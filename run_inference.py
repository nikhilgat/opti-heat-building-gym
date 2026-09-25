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
from run_evaluation import get_model_path

logging.getLogger("llec_building_gym").setLevel(logging.WARNING)  # env logs one INFO line per reset

RL = {"sac": SAC, "ppo": PPO, "ddpg": DDPG, "td3": TD3, "a2c": A2C}
CLASSICAL = ["pi", "pid", "fuzzy", "mpc"]
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--algorithm", default="sac", choices=list(RL) + CLASSICAL)
parser.add_argument("--model", default="best", choices=["best", "final"],
                    help="RL only. best = best_<algo>/best_model.zip (highest eval score), final = <algo>_model_seed42.zip.")
parser.add_argument("--reward_mode", default="band", choices=["band", "combined"])
parser.add_argument("--obs_variant", default="C05")
parser.add_argument("--mpc-horizon", type=int, default=288, dest="mpc_horizon", help="MPC horizon in 5-min steps (288 = 24 h).")
parser.add_argument("--mpc-replan", type=int, default=12, dest="mpc_replan", help="Re-solve the MPC every N steps (12 = hourly).")
parser.add_argument("--mpc-margin", type=float, default=0.0, dest="mpc_margin",
                    help="MPC plans inside [low+margin, high-margin] to absorb sensor noise (degC).")
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
if args.algorithm == "mpc" and args.mpc_margin:
    TAG += f"_margin{args.mpc_margin:g}"
BAND = args.reward_mode == "band"

temp_path = f"{args.data_dir}/langenhagen_outdoor_temperature_5min_{args.year}.csv"
price_path = f"{args.data_dir}/langenhagen_price_{args.year}.csv"
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
)
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
    model_path = get_model_path(args.algorithm, args.reward_mode, subfolder=args.obs_variant,
                                prefer_best=args.model == "best")
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
            band=(env.band[0] + args.mpc_margin, env.band[1] - args.mpc_margin)),
    }[args.algorithm]()
    source = f"{args.algorithm} controller (target {env.building.T_set:.1f} degC)"
print(f"{TAG.upper()} ({source}): {len(days)} days from {start_ts.date()}, {skipped} skipped "
      f"({'outside heating period or ' if BAND else ''}missing data)")

out_dir = f"results/inference/{args.year}/{args.reward_mode}"
os.makedirs(out_dir, exist_ok=True)
COLS = ["timestamp", "day", "T_out", "T_in", "T_set", "action", "P_el_W", "cop", "price_eur_kwh", "cost_eur", "reward"]


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


rows, T_in, prev = [], args.T_in0, None
for k, d in enumerate(days, 1):
    env.reset(seed=args.seed + d)
    b, s = env.building, d * STEPS
    # Replace the env's random weather window / shuffled price day with the real calendar day.
    b.T_out = t_out[s:s + 2 * STEPS].copy()  # next day included for C05 / MPC forecasts
    b.T_out_measurement = b.T_out
    b.start = s
    b.T_in = T_in if prev == d - 1 else args.T_in0  # restart after a data gap
    b._update_energy_price()
    env._update_set_point()
    obs = env._get_observation()
    for i in range(STEPS):
        if args.algorithm == "mpc" and BAND:
            if i % args.mpc_replan == 0:
                H = args.mpc_horizon
                model.predict(
                    obs, T_set_pred=[b.T_set] * (H + 1), T_out_pred=list(b.T_out[i:i + H]),
                    # the env charges step t with the price at start + t + 1
                    price_pred=list(price["price_normalized"].iloc[s + i + 1:s + i + 1 + H].ffill()),
                    heat_ok_pred=list(heat_mask[s + i:s + i + H]))
            action = model.last_plan[i % args.mpc_replan:i % args.mpc_replan + 1]
        else:
            action, _ = model.predict(obs, deterministic=True)
        T_set = b.T_set
        obs, reward, _, _, info = env.step(action)
        eur_kwh = price["baseprice"].iat[s + i]
        rows.append((start_ts + pd.Timedelta(minutes=5 * (s + i)), d, info["T_out"], b.T_in, T_set,
                     float(info["action"][0]), info["P_HP_el"], info["cop_used"], eur_kwh,
                     eur_kwh * info["P_HP_el"] / 1000 * (5 / 60), reward))
    T_in, prev = b.T_in, d
    if k % args.plot_every == 0 or k == len(days):
        plot_live(rows, k)
        print(f"  day {k}/{len(days)} ({(start_ts + pd.Timedelta(days=d)).date()})", flush=True)

steps = pd.DataFrame(rows, columns=COLS)
steps["energy_kwh"] = steps["P_el_W"] / 1000 * (5 / 60)
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
print(f"  saved         {out_dir}/{TAG}_summary.png, _live.png, _steps.csv, _daily.csv")
