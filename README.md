# Opti-Heat Building Gym

Reinforcement-learning vs. classical control of a **real residential heat pump** — a multi-family building in **Langenhagen (Hannover), Germany**, heated by an **IDM AERO ALM 4-12** air-source heat pump.

This project started as a fork of KIT's [LLECBuildingGym](https://github.com/KIT-IAI/LLECBuildingGym) (the code behind the paper *"Advanced Deep Reinforcement Learning for Heat Pump Control in Residential Buildings"*). That repo simulates a small *toy* building with a fixed-efficiency heat pump and weather from KIT Karlsruhe. We retrofitted it so the simulation describes **our** building: real thermal parameters, the real heat pump's datasheet, and a full year of real local weather and electricity prices. Then we retrained the RL agents and re-ran the whole controller benchmark.

> **Status (Sept 2026):** real-building model complete and validated. Latest benchmark: **SAC is the best controller** (reward 100.54), ahead of MPC (69.26), PPO (63.72) and Fuzzy (62.94). PI/PID are not yet retuned for this building. See [Where we are now](#6-where-we-are-now).

---

## 1. What is being simulated

Every 5 minutes a controller decides how hard to run the heat pump. The simulation then advances the building's indoor temperature and charges the electricity cost for that 5-minute slot. One **episode is one day** (288 steps).

```
 outdoor temp (Langenhagen 2025) ─┐
 electricity price (Tibber 2025) ─┤
                                  ▼
 controller ──action∈[-1,1]──►  heat pump (IDM ALM 4-12, 2D COP model)
    ▲                             │ heat Q_HP = action · Q_HP_Max
    │                             ▼
    └──── observation ◄──── building (1R1C thermal model) ──► reward = comfort − cost
```

- **Action** `a ∈ [-1, 1]`: fraction of maximum heat-pump power. Positive means heating, negative means cooling.
- **Controllers compared:** SAC and PPO (RL, Stable-Baselines3), PI, PID, Fuzzy logic, and MPC (Pyomo + IPOPT).

---

## 2. The building

### 2.1 Thermal model (1R1C)

The house is modelled as one thermal mass `mC` that loses heat to the outside through a conductance `K`:

```
mC · dT_in/dt = −K · (T_in − T_out) + Q_HP
```

| Parameter | Value | Meaning |
|---|---|---|
| `mC` | **74,600,000 J/K** | thermal capacitance of the building |
| `K` | **366 W/K** | overall heat-loss coefficient (UA) |
| `Q_HP_Max` | **12,000 W** | max heat output, the top of the IDM ALM 4-12 range |
| Room setpoint (MPC reference) | **20.2 °C** | |
| Thermal time constant `τ = mC/K` | **≈ 56.6 h** | how slowly the house cools with the heat pump off |

By comparison, the original toy building used `mC=300, K=20, Q_HP_Max=1500`.

**Validated.** `validate_dynamics.py` switches the heat pump off, simulates 48 h, fits the exponential cool-down and compares it to theory:

| Expected τ | Fitted τ | Error |
|---|---|---|
| 56.62 h | 56.58 h | **0.074 % ✔** |

This check also confirmed the key physics fix: the original code had a hidden `dt_scale = 1e-3` factor that made the building's time constant 1000× too long. With real SI values you must use `--dt-scale 1.0`.

### 2.2 Heat pump: IDM AERO ALM 4-12

The original repo used one fixed efficiency (COP) for every hour of the year. We replaced it with a model of the real machine, in `llec_building_gym/utils/heat_pump.py`. It was ported from the MATLAB model in `HEATPUMP/`:

1. **Weather-compensated flow temperature** (`heating_curve`). This is the EN 442 radiator law. Flow is 55 °C at −10 °C outside and drops to a floor of 35 °C in mild weather.
2. **2D COP = f(outdoor temp, flow temp)** (`HeatPump`). It interpolates the EN 14511 datasheet grid and includes duty-cycle averaging below the ≈4 kW modulation floor. Electrical power comes straight from the model.

| Outdoor | Flow temp | COP at full load |
|---|---|---|
| −20 °C | 55.0 °C | 1.66 |
| −7 °C | 52.3 °C | 2.16 |
| 2 °C | 43.6 °C | 2.87 |
| 10 °C | 35.0 °C | 4.98 |
| 20 °C | 35.0 °C | 5.43 |

Heating is therefore **about 3× cheaper per kWh of heat on a mild day than on a freezing one**. That matters for which controller wins (see §5). Enable the model with `--scop`. Cooling still uses a fixed `--cop_cool`, because the datasheet has no cooling data.

### 2.3 Data (Langenhagen, full year 2025)

| File | Content |
|---|---|
| `langenhagen-data/weather_langenhagen_2025.csv` | raw 15-min outdoor temperature |
| `langenhagen-data/electricity_prices_2025.csv` | raw 15-min Tibber dynamic prices (EUR/kWh) |
| `data/langenhagen_outdoor_temperature_5min.csv` | resampled to 5 min, 105,109 rows, −8.4 °C … 36.8 °C |
| `data/langenhagen_price_2025.csv` | resampled to 5 min, 105,118 rows, −0.086 … 0.906 EUR/kWh (real negative prices included), plus a `price_normalized` column in [0, 1] |

Days are split 80 % train / 20 % evaluation. Each episode picks a random day for weather and, independently, a random day for price.

### 2.4 Reward

```
reward_step = exp(−|T_in − T_set|)                              # comfort, ∈ (0, 1]
            − price · P_el / (max_price · P_el_worst_case)      # cost,    ∈ [−1, 0]
```

This is the `combined` reward mode, with both weights set to 1. `P_el` is the real electrical power from the COP model, so the cost term knows that mild-weather heating is cheap. An episode's score is the sum over 288 steps, so the best possible score is about 288 (perfect comfort, zero cost).

---

## 3. Controllers

| Controller | Type | Notes |
|---|---|---|
| **SAC** | RL (off-policy) | trained on the real building, observation variant `C04` |
| **PPO** | RL (on-policy) | trained on the real building, `C04` |
| **MPC** | optimisation (Pyomo/IPOPT) | uses the real building parameters, horizon 12 steps (1 h). Its cost term is still COP-blind (see §6). |
| **Fuzzy** | rule-based | reacts to temperature error |
| **PI / PID** | feedback | gains are still tuned for the toy building |

Observation variant `C04` = temperature deviation + current price + future prices + time of day + previous action.

---

## 4. Quick start

```bash
# install (Windows; use requirements.txt on Linux/HPC)
python -m venv llec_env && .\llec_env\Scripts\activate
pip install -r requirements_windows.txt
pip install -e .
```

> ⚠️ **The script defaults are still the toy building** (`mC=300, K=20, Q_HP_Max=1500, dt_scale=1e-3, COP=1`). To simulate our building, **always pass the real-building flags** below.

```bash
REAL="--mC 74600000 --K 366 --Q_HP_Max 12000 --cop_heat 2.2 --cop_cool 2.2 --scop --dt-scale 1.0"
```

**Validate the building physics**
```bash
python validate_dynamics.py --mC 74600000 --K 366 --dt-scale 1.0 --hours 48
```

**Train** (SAC ≈ 2 h and PPO ≈ 1.5 h for 1M steps on a laptop GPU. Env stepping on the CPU is the bottleneck.)
```bash
python run_train_rl.py --algorithm sac --reward_mode combined --obs_variant C04 --training --seed 42 \
  --timesteps 1e6 --num-envs 4 $REAL \
  --outdoor-temperature-path data/langenhagen_outdoor_temperature_5min.csv \
  --energy-price-path data/langenhagen_price_2025.csv \
  --tensorboard-log tb_logs          # optional; view with: tensorboard --logdir tb_logs
```

**Evaluate all controllers** (10 one-day episodes each)
```bash
python run_evaluation.py --algorithms sac ppo "PI Control" "PID Control" "Fuzzy Control" "MPC Control" \
  --reward_mode combined --obs_variant C04 $REAL \
  --outdoor_temperature_path data/langenhagen_outdoor_temperature_5min.csv \
  --energy_price_path data/langenhagen_price_2025.csv --prefer_best
```

**Summary table**
```bash
python compare_results.py --reward_mode combined --outdoor_temperature_path data/langenhagen_outdoor_temperature_5min.csv
```
Before re-running an evaluation, delete the old `results/combined/real_temp_data/*.csv`. Otherwise stale rows show up in the summary.

---

## 5. Results: how we got here

Scores are **average reward per one-day episode** (higher is better). **Rounds are not comparable with each other**, because each round changed the physics or the data. Compare controllers only within a round.

### Round 1: Original toy building (baseline)
Repo defaults, temperature-only reward, observation `T01`.

| MPC | PID | PI | PPO | SAC | DDPG | Fuzzy | A2C |
|---|---|---|---|---|---|---|---|
| 267.26 | 251.14 | 251.14 | 250.99 | 250.30 | 230.45 | 214.96 | 142.49 |

All controllers do well. It is an easy, fast-reacting toy building.

### Round 2: Real parameters, first attempt (superseded)
We plugged in `mC/K/Q_HP_Max` and found that **MPC was optimising against the toy building internally**. It was never given the real parameters. After fixing that and retraining: SAC 4.96, PPO 4.57, MPC 1.72, PI/PID/Fuzzy ≈ −103.
These numbers are **invalid**: they still used the `dt_scale = 1e-3` bug, which made the house react 1000× too slowly.

### Round 3: Correct physics (`dt_scale = 1.0`), 1D COP curve, KIT weather
| MPC | PPO | SAC | Fuzzy | PI / PID |
|---|---|---|---|---|
| **50.26** | 43.10 | 36.11 | 34.27 | −27.85 |

With correct physics, MPC led. The COP here was a 1D curve at a fixed 55 °C flow temperature (range 1.66–3.58).

### Round 4 (current): Full 2D COP + real Langenhagen weather & prices
SAC and PPO were retrained for 1M steps each (SAC 123 min, PPO 90 min), then all six controllers were evaluated on 10 held-out days.

| Rank | Controller | Avg. reward | Eval time |
|---|---|---|---|
| 🥇 | **SAC** | **100.54** | 36 s |
| 🥈 | MPC | 69.26 | 148 s |
| 🥉 | PPO | 63.72 | 36 s |
| 4 | Fuzzy | 62.94 | 24 s |
| 5 | PI | −21.13 | 31 s |
| 6 | PID | −21.14 | 32 s |

**Why SAC now beats MPC.** We checked that this is a real effect and not a bug. Per-step reward breakdown:

| | Comfort (norm.) | Cost (norm.) | Total per step | Mean \|action\| |
|---|---|---|---|---|
| MPC | 0.458 | −0.217 | 0.241 | 0.79 |
| SAC | 0.432 | **−0.083** | **0.349** | 0.30 |

MPC's internal cost model is `price × |action|`, so it is **blind to COP**. It doesn't know that heating on a mild day is about 3× cheaper than on a cold one. It heats hard all the time and buys slightly better comfort for about 2.6× the real energy cost. SAC was trained directly on the COP-aware reward and learned a much cheaper strategy that costs a little comfort. The effect grew in round 4 because the 2D COP varies far more (1.66–5.43) than the old 1D curve did.

PI/PID remain negative because their gains were never retuned for a 56-hour time-constant house. They saturate and waste energy.

---

## 6. Where we are now

**Done**
- [x] Real building parameters exposed on the CLI and wired into the env **and** into MPC
- [x] `dt_scale` physics bug fixed and validated (τ error 0.074 %)
- [x] Full 2D datasheet heat pump model (IDM AERO ALM 4-12) with weather-compensated flow temperature
- [x] Real Langenhagen 2025 weather + Tibber price data at 5-min resolution
- [x] SAC/PPO retrained on the real building (`models/combined/C04/`)
- [x] Full benchmark (round 4 above)
- [x] TensorBoard logging for training

**Open, roughly in priority order**
1. **Make MPC's objective COP-aware.** Its cost is `price × |action|` instead of `price × P_el(COP)`. Until this is fixed, the MPC vs. SAC comparison favours SAC.
2. **Retune PI/PID/Fuzzy gains** for the real building. They are currently not a fair baseline.
3. **Coordinated weather/price sampling.** An episode currently pairs one day's weather with a different random day's prices.
4. **Change the script defaults** to the real building so the flags aren't needed.
5. `cop_cool` is still a fixed assumption (2.2). There is no cooling data for this heat pump.
6. GPU utilisation is below 10 % during training because env stepping is the bottleneck. A `--gradient-steps` option would help.

**Model caveats**

| Path | Status |
|---|---|
| `models/combined/C04/{sac,ppo}_*` | ✅ current, trained on the real building (round 4) |
| `models/combined/C04/{a2c,ddpg}_*`, `models/combined/C01–C03/`, `models/temperature/` | ⚠️ original toy-building models. Don't use them with the real-building flags. |

A detailed, step-by-step log of every change is in **[CHANGELOG.md](CHANGELOG.md)**.

---

## 7. Repository layout

```
.
├── llec_building_gym/
│   ├── envs/base_building_gym.py   # Building (1R1C model) + BaseBuildingGym (Gymnasium env, reward)
│   ├── controllers/                # PI, PID, Fuzzy, MPC (Pyomo) — see README_MPC.md for solver setup
│   └── utils/
│       ├── heat_pump.py            # IDM ALM 4-12 model: heating_curve + 2D COP HeatPump
│       └── temporal_features.py
├── HEATPUMP/                       # original MATLAB heat pump model (reference for heat_pump.py)
├── data/                           # 5-min Langenhagen weather + price (used by the env)
├── langenhagen-data/               # raw 15-min source data
├── models/                         # trained RL models (see caveats above)
├── results/                        # evaluation CSVs + logs (git-ignored, regenerate with run_evaluation.py)
├── slurm_script/                   # batch scripts for HPC (SLURM) runs
├── run_train_rl.py                 # train SAC / PPO / DDPG / A2C
├── run_evaluation.py               # evaluate RL agents and classical controllers
├── compare_results.py              # summary table from results/
├── validate_dynamics.py            # free-response check of the building time constant
└── CHANGELOG.md                    # full history of the retrofit
```
