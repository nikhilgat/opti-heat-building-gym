"""
run_inference.py - Run a trained agent or a classical controller over a new period
of real data (e.g. Langenhagen 2026), day by day in calendar order. Test data only:
nothing here trains or updates a model.

Unlike run_evaluation.py (random held-out days, random weather window), this
walks every day from midnight, uses the real outdoor temperature and prices for
that exact day, and carries the indoor temperature over from one day to the
next. Days whose weather or prices (incl. the 30 h price look-ahead) are
missing are skipped, and the indoor temperature restarts after such a gap.
With --reward_mode band (default, the real building's rules) only heating-period
days (Oct-Apr) are simulated: heating only, comfort band 19-22 degC.

Prepare the data first:
    python prepare_langenhagen_data.py --year 2026

Then:
    python run_inference.py --year 2026                          # SAC, best model
    python run_inference.py --year 2026 --algorithm td3 --model final
    python run_inference.py --year 2026 --algorithm mpc          # also: pi, pid, fuzzy

Outputs (results/inference/<year>/<reward_mode>/):
    <tag>_live.png     refreshed every --plot-every days while running (open it in VS Code to watch)
    <tag>_summary.png  whole period, per day
    <tag>_steps.csv    one row per 5-min step
    <tag>_daily.csv    one row per day
<tag> = <algo>_<best|final> for RL agents, <algo> for classical controllers.
"""
import argparse
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from llec_building_gym import BaseBuildingGym, FuzzyController, MPCController, PIController, PIDController
from llec_building_gym.utils.heat_pump import HeatPump, heating_curve
from run_evaluation import get_model_path

logging.getLogger("llec_building_gym").setLevel(logging.WARNING)  # env logs one INFO line per reset

RL = {"sac": SAC, "ppo": PPO, "ddpg": DDPG, "td3": TD3, "a2c": A2C}
CLASSICAL = ["pi", "pid", "fuzzy", "mpc", "curve"]
OIL_KWH_PER_L, BOILER_EFF = 9.97, 0.90  # heating oil lower heating value; typical non-condensing boiler
OIL_EUR_PER_L = (0.95, 1.42)  # DE heating oil Jan 2026 low / Mar 2026 high (incl. VAT, CO2 levy)


class HeatingCurve:
    """Conventional weather-compensated control (no room sensor, no prices): heat output
    follows the outdoor temperature only, sized from the building's heat-loss
    coefficient to hold the target. Reads the current T_out from the C05/C06 observation
    (outdoor sensor); the action is a share of the heat pump's maximum output (the backup
    heater is automatic in the env)."""

    def __init__(self, K, T_target):
        self.K, self.T_target, self.hp = K, T_target, HeatPump()

    def predict(self, obs, deterministic=True):
        t_out = obs[4] * 10.0 + 10.0  # feature (T_out - 10) / 10
        cap_w = self.hp.get_performance(t_out, heating_curve(t_out)[0], 1.0)[0] * 1000.0
        return [min(max(self.K * (self.T_target - t_out) / cap_w, 0.0), 1.0)], None

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--algorithm", default="sac", choices=list(RL) + CLASSICAL)
parser.add_argument("--model", default="best", choices=["best", "final"],
                    help="RL only. best = best_<algo>/best_model.zip (highest eval score), final = <algo>_model_seed42.zip.")
parser.add_argument("--reward_mode", default="band", choices=["band", "combined"])
parser.add_argument("--obs_variant", default="C06")
parser.add_argument("--tag", default="", help="Model folder suffix used in training (e.g. ew5).")
parser.add_argument("--model-seed", type=int, default=42, dest="model_seed", help="RL only: training seed of the model.")
parser.add_argument("--backup-kw", type=float, default=6.0, dest="backup_kw",
                    help="Electric backup heater (kW, COP 1); IDM AERO ALM 4-12 indoor unit has a 6 kW rod.")
parser.add_argument("--classical-target", type=float, default=19.5, dest="classical_target",
                    help="Room temperature PI/PID regulate to (degC). Fuzzy and the heating curve keep the band centre.")
parser.add_argument("--mpc-horizon", type=int, default=288, dest="mpc_horizon", help="MPC horizon in 5-min steps (288 = 24 h).")
parser.add_argument("--mpc-replan", type=int, default=12, dest="mpc_replan", help="Re-solve the MPC every N steps (12 = hourly).")
parser.add_argument("--mpc-margin", type=float, default=0.0, dest="mpc_margin",
                    help="MPC soft buffer above the band's lower edge (degC): dipping into it is cheap, below the band is not.")
parser.add_argument("--data-dir", default="data", dest="data_dir")
parser.add_argument("--seed", type=int, default=58)
parser.add_argument("--T_in0", type=float, default=20.0, help="Indoor temperature on the first morning (degC).")
parser.add_argument("--plot-every", type=int, default=5, dest="plot_every", help="Refresh the live plot every N days.")
# Real Langenhagen building (the scripts' defaults are still the toy building)
parser.add_argument("--mC", type=float, default=74_600_000)
parser.add_argument("--K", type=float, default=366)
parser.add_argument("--Q_HP_Max", type=float, default=12_000)
parser.add_argument("--cop_heat", type=float, default=2.2)
parser.add_argument("--cop_cool", type=float, default=2.2)
parser.add_argument("--dt-scale", type=float, default=1.0, dest="dt_scale")
args = parser.parse_args()
TAG = f"{args.algorithm}_{args.model}" if args.algorithm in RL else args.algorithm
if args.algorithm in RL and args.model_seed != 42:
    TAG += f"_seed{args.model_seed}"
if args.algorithm == "mpc" and args.mpc_margin:
    TAG += f"_margin{args.mpc_margin:g}"
BAND = args.reward_mode == "band"
SUB = args.obs_variant + (f"_{args.tag}" if args.tag else "")

temp_path = f"{args.data_dir}/langenhagen_outdoor_temperature_5min_{args.year}.csv"
price_path = f"{args.data_dir}/langenhagen_price_{args.year}.csv"
fc_err_path = f"{args.data_dir}/langenhagen_forecast_error_5min_{args.year}.csv"  # optional (--forecast)
fc_err_path = fc_err_path if os.path.exists(fc_err_path) else None
for p in (temp_path, price_path):
    if not os.path.exists(p):
        raise SystemExit(f"Missing {p} - put the raw file in langenhagen-data/ and run "
                         f"`python prepare_langenhagen_data.py --year {args.year}` first.")

temp = pd.read_csv(temp_path)
price = pd.read_csv(price_path, parse_dates=["start"])
t_out = temp["temp_amb [degC]"].to_numpy()
start_ts = pd.to_datetime(temp["timestamp"].iloc[0])
if pd.Timestamp(price["start"].iloc[0]) != start_ts:
    raise SystemExit(f"Weather starts {start_ts} but prices start {price['start'].iloc[0]} - they must align.")
STEPS = 288  # 5-min steps per day
env = BaseBuildingGym(
    mC=args.mC, K=args.K, Q_HP_Max=args.Q_HP_Max, simulation_time=24 * 3600, control_step=300,
    schedule_type=temp_path, training=False, reward_mode=args.reward_mode,
    energy_price_path=price_path, outdoor_temperature_path=temp_path, obs_variant=args.obs_variant,
    cop_heat=args.cop_heat, cop_cool=args.cop_cool, dt_scale=args.dt_scale, use_scop=True,
    backup_kw=args.backup_kw, forecast_error_path=fc_err_path,
)
fc_err = env.building._fc_err_all  # None without a forecast error file (then forecasts are perfect)
LOOKAHEAD = env.prediction_horizon
n_days = min(len(t_out), len(price)) // STEPS
price_nan = price["price_normalized"].isna().to_numpy()
heat_mask = env.building.heating_mask  # None unless band mode


def usable(d):
    s = d * STEPS
    weather_ok = not pd.isna(t_out[s:s + STEPS]).any()
    prices_ok = not price_nan[s:min(s + STEPS + LOOKAHEAD, len(price_nan))].any()
    in_season = heat_mask is None or heat_mask[s:s + STEPS].all()
    return weather_ok and prices_ok and in_season


days = [d for d in range(n_days) if usable(d)]
skipped = n_days - len(days)

if args.algorithm in RL:
    model_path = get_model_path(args.algorithm, args.reward_mode, subfolder=SUB,
                                prefer_best=args.model == "best", model_seed=args.model_seed)
    model = RL[args.algorithm].load(model_path)
    source = model_path
else:
    b0 = env.building
    model = {
        "pi": lambda: PIController(Kp=0.4, Ki=1.5e-4),
        "pid": lambda: PIDController(Kp=0.4, Ki=1.5e-4, Kd=0.05),
        "fuzzy": lambda: FuzzyController(debug=False, use_gaussian=True, sigma=1.0, fine_tuning=True),
        "mpc": lambda: MPCController(
            dt=300, horizon=args.mpc_horizon, reward_mode=args.reward_mode, dt_scale=args.dt_scale,
            building_params={"mC": args.mC, "K": args.K, "Q_HP_Max": args.Q_HP_Max, "T_set": b0.T_set},
            price_max=float(b0.full_price_df["price_normalized"].max()), p_el_worst_w=env._P_el_worst_case,
            band=env.band, margin=args.mpc_margin),
        "curve": lambda: HeatingCurve(args.K, b0.T_set),
    }[args.algorithm]()
    target = args.classical_target if args.algorithm in ("pi", "pid") else env.building.T_set
    source = f"{args.algorithm} controller (target {target:.1f} degC)"
print(f"{TAG.upper()} ({source}): {len(days)} days from {start_ts.date()}, {skipped} skipped "
      f"({'outside heating period or ' if BAND else ''}missing data)")

out_dir = f"results/inference/{args.year}/{args.reward_mode}/{SUB}"
os.makedirs(out_dir, exist_ok=True)
COLS = ["timestamp", "day", "T_out", "T_in", "T_set", "action", "P_el_W", "cop", "price_eur_kwh", "cost_eur", "reward",
        "heat_W", "backup_W"]


def comfort_lines(ax):
    if BAND:
        ax.axhspan(*env.band, color="tab:green", alpha=0.12, label=f"band {env.band[0]:.0f}-{env.band[1]:.0f} °C")


def plot_live(rows, done):
    df = pd.DataFrame(rows[-7 * STEPS:], columns=COLS)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    comfort_lines(ax[0])
    ax[0].plot(df.timestamp, df.T_in, label="T_in (indoor)", lw=1.5)
    if not BAND:
        ax[0].plot(df.timestamp, df.T_set, label="T_set (target)", lw=1, ls="--", color="k")
    ax[0].plot(df.timestamp, df.T_out, label="T_out (outdoor)", lw=1, color="tab:gray")
    ax[0].set_ylabel("°C"); ax[0].legend(loc="upper left", ncol=3, fontsize=8)
    ax[1].plot(df.timestamp, df.action, lw=1, color="tab:red", label="action (heat pump load)")
    ax[1].plot(df.timestamp, df.P_el_W / 1000, lw=1, color="tab:orange", label="P_el (kW)")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].legend(loc="upper left", fontsize=8)
    ax[2].step(df.timestamp, df.price_eur_kwh, lw=1, color="tab:green", where="post")
    ax[2].set_ylabel("EUR/kWh")
    cost = sum(r[9] for r in rows)
    fig.suptitle(f"{TAG.upper()} inference {args.year} - day {done}/{len(days)} "
                 f"(last 7 days shown) - cost so far {cost:,.2f} EUR")
    fig.tight_layout(); fig.savefig(f"{out_dir}/{TAG}_live.png", dpi=90); plt.close(fig)


rows, T_in, prev, last_a, last_bk = [], args.T_in0, None, 0.0, False
for k, d in enumerate(days, 1):
    env.reset(seed=args.seed + d)
    b, s = env.building, d * STEPS
    # Replace the env's random weather window / shuffled price day with the real calendar day.
    b.T_out = t_out[s:s + 2 * STEPS].copy()  # next day included for C05/C06 / MPC forecasts
    b.T_out_measurement = b.T_out
    if fc_err is not None:
        b.T_out_fc_err = fc_err[s:s + 2 * STEPS].copy()
    continuous = prev == d - 1
    b.start = s
    b.T_in = T_in if continuous else args.T_in0  # restart after a data gap
    env.prev_action = last_a if continuous else 0.0
    env._backup_on = last_bk if continuous else False
    b._update_energy_price()
    env._update_set_point()
    obs = env._get_observation()
    for i in range(STEPS):
        if args.algorithm == "mpc" and BAND:
            if i % args.mpc_replan == 0:
                H = args.mpc_horizon
                # Same information as the C06 agent: measured T_out now, real day-ahead
                # forecast after that, only published prices (rest = same slot yesterday).
                T_fc = b.T_out[i:i + H] + (0.0 if b.T_out_fc_err is None else b.T_out_fc_err[i:i + H])
                T_fc[0] = b.T_out[i]
                model.predict(
                    obs, T_set_pred=[b.T_set] * (H + 1), T_out_pred=list(T_fc),
                    price_pred=b.price_forecast(H, first=0),  # env charges the current slot
                    heat_ok_pred=list(heat_mask[s + i:s + i + H]))
            action = model.last_plan[i % args.mpc_replan:i % args.mpc_replan + 1]
        elif args.algorithm in ("pi", "pid"):
            o = obs.copy()
            o[0] += b.T_set - args.classical_target  # regulate to the classical target, not the band centre
            action, _ = model.predict(o, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        T_set = b.T_set
        obs, reward, _, _, info = env.step(action)
        eur_kwh = price["baseprice"].iat[s + i]
        rows.append((start_ts + pd.Timedelta(minutes=5 * (s + i)), d, info["T_out"], b.T_in, T_set,
                     float(info["action"][0]), info["P_HP_el"], info["cop_used"], eur_kwh,
                     eur_kwh * info["P_HP_el"] / 1000 * (5 / 60), reward, info["Q_heat_W"], info["P_backup_W"]))
    T_in, prev, last_a, last_bk = b.T_in, d, env.prev_action, env._backup_on
    if k % args.plot_every == 0 or k == len(days):
        plot_live(rows, k)
        print(f"  day {k}/{len(days)} ({(start_ts + pd.Timedelta(days=d)).date()})", flush=True)

steps = pd.DataFrame(rows, columns=COLS)
steps["energy_kwh"] = steps["P_el_W"] / 1000 * (5 / 60)
steps["heat_kwh"] = steps["heat_W"] / 1000 * (5 / 60)  # heat delivered to the building
steps["backup_kwh"] = steps["backup_W"] / 1000 * (5 / 60)  # electric backup heater share
lo, hi = env.band
steps["below_K"] = (lo - steps["T_in"]).clip(lower=0)
steps["above_K"] = (steps["T_in"] - hi).clip(lower=0)
steps["in_band"] = (steps["below_K"] == 0) & (steps["above_K"] == 0)
steps["abs_dev"] = (steps["T_in"] - steps["T_set"]).abs()
daily = steps.groupby(steps["timestamp"].dt.date).agg(
    T_out_mean=("T_out", "mean"), T_in_mean=("T_in", "mean"), T_in_min=("T_in", "min"), T_in_max=("T_in", "max"),
    energy_kwh=("energy_kwh", "sum"), cost_eur=("cost_eur", "sum"), in_band_share=("in_band", "mean"),
    mean_abs_dev_K=("abs_dev", "mean"), reward=("reward", "sum"))
steps.drop(columns=["abs_dev"]).to_csv(f"{out_dir}/{TAG}_steps.csv", index=False)
daily.to_csv(f"{out_dir}/{TAG}_daily.csv")

# Whole-period summary (one point per day; skipped days show as gaps)
dd = daily.reindex(pd.date_range(start_ts.date(), periods=n_days).date)
x = pd.to_datetime(dd.index)
fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
comfort_lines(ax[0])
ax[0].plot(x, dd.T_out_mean, color="tab:gray", label="T_out daily mean")
ax[0].fill_between(x, dd.T_in_min, dd.T_in_max, color="tab:blue", alpha=0.25, label="T_in daily min-max")
ax[0].plot(x, dd.T_in_mean, color="tab:blue", label="T_in daily mean")
ax[0].set_ylabel("°C"); ax[0].legend(fontsize=8, ncol=4)
ax[1].bar(x, dd.energy_kwh, color="tab:orange"); ax[1].set_ylabel("kWh / day")
ax[2].bar(x, dd.cost_eur, color="tab:green"); ax[2].set_ylabel("EUR / day")
if BAND:
    ax[3].plot(x, dd.in_band_share * 100, color="tab:red"); ax[3].set_ylabel("% time in band"); ax[3].set_ylim(0, 105)
else:
    ax[3].plot(x, dd.mean_abs_dev_K, color="tab:red"); ax[3].set_ylabel("mean |T_in-T_set| K")
fig.suptitle(f"{TAG.upper()} inference {args.year}: {steps['energy_kwh'].sum():,.0f} kWh, "
             f"{steps['cost_eur'].sum():,.2f} EUR over {len(days)} days")
fig.tight_layout(); fig.savefig(f"{out_dir}/{TAG}_summary.png", dpi=90); plt.close(fig)

h = 5 / 60  # hours per step
print(f"\n{TAG.upper()} {steps['timestamp'].iloc[0].date()} .. {steps['timestamp'].iloc[-1].date()} "
      f"({len(days)} days simulated, {skipped} skipped)")
print(f"  electricity   {steps['energy_kwh'].sum():,.0f} kWh")
print(f"  cost          {steps['cost_eur'].sum():,.2f} EUR  (avg {steps['cost_eur'].sum() / max(steps['energy_kwh'].sum(), 1e-9):.3f} EUR/kWh paid)")
print(f"  comfort       {steps['in_band'].mean() * 100:.1f}% of time in {lo:.0f}-{hi:.0f} degC, "
      f"below {lo:.0f}: {steps['below_K'].sum() * h:,.1f} K*h, above {hi:.0f}: {steps['above_K'].sum() * h:,.1f} K*h, "
      f"min T_in {steps['T_in'].min():.1f} degC")
print(f"  avg reward    {daily['reward'].mean():.2f} per day")
oil_l = steps["heat_kwh"].sum() / BOILER_EFF / OIL_KWH_PER_L
print(f"  backup heater {steps['backup_kwh'].sum():,.0f} kWh electricity ({(steps['backup_W'] > 0).mean() * 100:.1f}% of steps)")
print(f"  heat          {steps['heat_kwh'].sum():,.0f} kWh delivered; same heat from an oil boiler "
      f"({BOILER_EFF:.0%}): {oil_l:,.0f} L = {oil_l * OIL_EUR_PER_L[0]:,.0f}-{oil_l * OIL_EUR_PER_L[1]:,.0f} EUR "
      f"at {OIL_EUR_PER_L[0]:.2f}-{OIL_EUR_PER_L[1]:.2f} EUR/L")
print(f"  saved         {out_dir}/{TAG}_summary.png, _live.png, _steps.csv, _daily.csv")
