"""
prepare_langenhagen_data.py - Convert raw Langenhagen weather/price exports in
langenhagen-data/ into the 5-min CSVs the environment reads (data/).

Inputs (either may be missing; whatever exists is converted):
    langenhagen-data/weather_langenhagen_<year>.csv
        Open-Meteo hourly export (metadata rows, then "time,temperature_2m (°C)")
        or plain "timestamp,T_out_C" (15-min).
    langenhagen-data/electricity_prices_<year>.csv
        Tibber export ("Datum von","Datum bis","Tibber <year>", DD.MM.YYYY HH:MM,
        comma decimals) or plain "timestamp,price" in EUR/kWh.

Outputs:
    data/langenhagen_outdoor_temperature_5min_<year>.csv   (timestamp, temp_amb [degC])
    data/langenhagen_price_<year>.csv                      (start, baseprice, unit, price_normalized)

price_normalized uses the min/max of the 2025 TRAINING prices, not the new
year's own range, so the trained agent sees prices on the same scale it learned.

Usage:
    python prepare_langenhagen_data.py --year 2026
"""
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--raw-dir", default="langenhagen-data")
parser.add_argument("--out-dir", default="data")
parser.add_argument("--train-price", default="data/langenhagen_price_2025.csv",
                    help="Prepared training price CSV whose min/max defines the normalization.")
parser.add_argument("--forecast", action="store_true",
                    help="Also download the real day-ahead temperature forecast error (Open-Meteo previous runs, "
                         "forecast issued 24 h before minus actual) -> data/langenhagen_forecast_error_5min_<year>.csv")
args = parser.parse_args()


def read_weather(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    # Open-Meteo puts location metadata above the real header; find the header row.
    header = next(i for i, l in enumerate(lines) if l.lower().startswith(("time,", "timestamp,")))
    df = pd.read_csv(path, skiprows=header, encoding="utf-8", encoding_errors="replace")
    ts = pd.to_datetime(df.iloc[:, 0])
    return pd.Series(df.iloc[:, 1].astype(float).values, index=ts)


def read_prices(path):
    df = pd.read_csv(path, dtype=str)
    if "Datum von" in df.columns:  # Tibber export (may be a full workbook sheet with extra columns)
        ts = pd.to_datetime(df["Datum von"], format="%d.%m.%Y %H:%M")
        price = df[f"Tibber {args.year}"].str.replace(",", ".").astype(float)
    else:
        ts = pd.to_datetime(df.iloc[:, 0])
        price = df.iloc[:, 1].str.replace(",", ".").astype(float)
    return pd.Series(price.values, index=ts)


MAX_FILL = 15  # 5-min steps (75 min): fills the DST spring-forward hour and the
# last hour of an hourly file, but leaves real data gaps (missing days) as NaN.


def to_5min(s, how):
    s = s[~s.index.duplicated(keep="first")].sort_index()  # DST fall-back duplicates
    end = s.index.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    grid = pd.date_range(s.index.min().normalize(), end, freq="5min")
    s = s.reindex(s.index.union(grid))
    if how == "linear":
        s = s.interpolate(method="time", limit=MAX_FILL, limit_area="inside")
    return s.reindex(grid).ffill(limit=MAX_FILL)


def report_gaps(name, s):
    missing = s.isna().groupby(s.index.date).any()
    if missing.any():
        days = missing[missing].index
        print(f"         {name}: {len(days)} day(s) with missing data, left empty: {days[0]} .. {days[-1]}")


weather_raw = os.path.join(args.raw_dir, f"weather_langenhagen_{args.year}.csv")
price_raw = os.path.join(args.raw_dir, f"electricity_prices_{args.year}.csv")

if os.path.exists(weather_raw):
    t = to_5min(read_weather(weather_raw), "linear")
    out = os.path.join(args.out_dir, f"langenhagen_outdoor_temperature_5min_{args.year}.csv")
    pd.DataFrame({"timestamp": t.index, "temp_amb [degC]": t.values.round(3)}).to_csv(out, index=False)
    print(f"weather: {out}  {len(t)} rows ({len(t)//288} days)  "
          f"{t.index[0]} .. {t.index[-1]}  range {t.min():.1f}..{t.max():.1f} degC")
    report_gaps("weather", t)
else:
    print(f"weather: {weather_raw} not found, skipped")

if os.path.exists(price_raw):
    p = to_5min(read_prices(price_raw), "step")  # prices are constant within their slot
    train = pd.read_csv(args.train_price)["baseprice"]
    lo, hi = train.min(), train.max()
    out = os.path.join(args.out_dir, f"langenhagen_price_{args.year}.csv")
    pd.DataFrame({
        "start": p.index, "baseprice": p.values, "unit": "EUR/kWh",
        "price_normalized": (p.values - lo) / (hi - lo),
    }).to_csv(out, index=False)
    outside = ((p < lo) | (p > hi)).mean() * 100
    print(f"prices : {out}  {len(p)} rows ({len(p)//288} days)  "
          f"{p.index[0]} .. {p.index[-1]}  range {p.min():.3f}..{p.max():.3f} EUR/kWh")
    print(f"         normalized with 2025 training range [{lo:.3f}, {hi:.3f}]; "
          f"{outside:.1f}% of {args.year} slots fall outside it")
    report_gaps("prices", p)
else:
    print(f"prices : {price_raw} not found, skipped (add it, then re-run)")

if args.forecast:
    import json
    import urllib.request
    w = read_weather(weather_raw)
    end = min(w.index.max().date(), pd.Timestamp(f"{args.year}-12-31").date())
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=52.44&longitude=9.74"
           "&hourly=temperature_2m,temperature_2m_previous_day1&timezone=Europe%2FBerlin"
           f"&start_date={args.year}-01-01&end_date={end}")
    h = json.load(urllib.request.urlopen(url, timeout=120))["hourly"]
    err = pd.Series(pd.Series(h["temperature_2m_previous_day1"]).values - pd.Series(h["temperature_2m"]).values,
                    index=pd.to_datetime(h["time"]))
    e = to_5min(err, "linear").fillna(0.0)
    out = os.path.join(args.out_dir, f"langenhagen_forecast_error_5min_{args.year}.csv")
    pd.DataFrame({"timestamp": e.index, "fc_err [K]": e.values.round(3)}).to_csv(out, index=False)
    print(f"forecast error: {out}  {len(e)} rows, RMSE {(e ** 2).mean() ** 0.5:.2f} K, bias {e.mean():+.2f} K")
