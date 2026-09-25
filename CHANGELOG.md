# LLECBuildingGym — Session Changelog

## 1. Setup

| Step | Detail |
|---|---|
| Repo cloned | github.com/KIT-IAI/LLECBuildingGym → `C:\Users\nikhi\Documents\buildinggym` (via `setup.bat`, git fetch/checkout) |
| Deps installed | `pip install -r requirements_windows.txt` |
| GPU fix | Default `pip install torch` gave CPU-only build. Reinstalled with `--index-url https://download.pytorch.org/whl/cu132` for RTX PRO 500 Blackwell (sm_120) support |

## 2. Baseline run (repo defaults, toy building)

Command:
```
python run_evaluation.py --algorithms ppo sac "MPC Control" --reward_mode temperature --obs_variant T01
```

| Algorithm | Avg. reward | Runtime |
|---|---|---|
| MPC Control | 267.26 | 233s |
| PID Control | 251.14 | 1.6s |
| PI Control | 251.14 | 1.6s |
| PPO | 250.99 | 3.4s |
| SAC | 250.30 | 5.9s |
| DDPG | 230.45 | 4.9s |
| Fuzzy Control | 214.96 | 1.8s |
| A2C | 142.49 | 3.9s |

Building params used: repo defaults (`mC=300, K=20, Q_HP_Max=1500`) — not your building.

## 3. Code changes: expose building params

Neither `run_evaluation.py` nor `run_train_rl.py` exposed building physics via CLI. Both hardcoded `mC=300, K=20, Q_HP_Max=1500, cop=1.0`.

| File | Change |
|---|---|
| `run_evaluation.py` | Added `--mC`, `--K`, `--Q_HP_Max`, `--cop_heat`, `--cop_cool` flags; wired into `BaseBuildingGym()` |
| `run_train_rl.py` | Same flags added; wired into `make_env()` calls for train + eval envs |
| `run_train_rl.py` | Added `--batch-size`, `--buffer-size` flags (previously hardcoded to 64 / 100k) |
| `compare_results.py` | New file — aggregates per-algorithm result CSVs into one summary table (script itself never printed one) |

## 4. Your building's real values (from project docs)

| Param | Value | Source |
|---|---|---|
| `mC` | 74,600,000 J/K | thermal capacitance C |
| `K` | 366 W/K | building UA |
| `Q_HP_Max` | 12,000 W | IDM AERO ALM 4-12, top of range |
| `cop_heat` | ~2.2 (winter-representative) | IDM datasheet, 55°C flow, -7°C to +2°C outdoor range (repo default of 3.0 overstates cold-weather efficiency) |
| Setpoint | 20.2°C | your MPC room reference |

IDM AERO ALM 4-12 real COP table (55°C flow, from `IDM_AEROALM412pagesdeleted.pdf`, EN14511):

| Outdoor temp | COP |
|---|---|
| 20°C | 3.68 |
| 15°C | 3.66 |
| 12°C | 3.45 |
| 10°C | 3.25 |
| 7°C | 3.02 |
| 2°C | 2.58 |
| -7°C | 2.09 |
| -10°C | 1.94 |
| -15°C | 1.73 |
| -20°C | 1.51 |

## 5. First run with real building params — bug found

Command:
```
python run_evaluation.py --algorithms "MPC Control" ppo sac --reward_mode combined --obs_variant C04 --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv" --mC 74600000 --K 366 --Q_HP_Max 12000 --cop_heat 3.0
```

| Algorithm | Avg. reward | Runtime |
|---|---|---|
| SAC (untrained on real params) | -94.57 | 25.8s |
| PPO (untrained on real params) | -97.84 | 27.3s |
| MPC Control | -103.28 | 236.5s |

**Bug found:** `MPCController` was instantiated without `building_params`, so it optimized using its own internal hardcoded default building (`mC=300, K=20, Q_HP_Max=1500`) while the actual environment used your real building. MPC's decisions were blind to your real physics — invalidating the comparison.

**Fix:** passed `building_params={"mC": mC, "K": K, "Q_HP_Max": Q_HP_Max, "T_set": 20.2}` into `MPCController()` in `run_evaluation.py`.

## 6. Retraining SAC/PPO on real building params

Command:
```
python run_train_rl.py --algorithm sac --reward_mode combined --obs_variant C04 --training --mC 74600000 --K 366 --Q_HP_Max 12000 --cop_heat 3.0 --timesteps 1e6 --num-envs 16 --eval-freq 50000 --batch-size 1024
```
(repeated with `--algorithm ppo`)

GPU note: utilization stayed under 10% throughout. Root cause is architectural, not a config error — SAC does one small gradient update per env step; env stepping (CPU/Python) is the bottleneck, not GPU compute. Larger `--batch-size` didn't change this. Not yet resolved (would need `--gradient-steps` support, not yet added).

GPU setup: needed `torch` reinstalled via `--index-url https://download.pytorch.org/whl/cu132` for RTX PRO 500 Blackwell (sm_120) — default `pip install torch` and even the cu124 index gave "not compatible" warnings since Blackwell wasn't yet supported in older CUDA builds.

## 7. Final benchmark (MPC fix + retrained models, real building params)

Command:
```
python run_evaluation.py --algorithms "MPC Control" "PI Control" "PID Control" "Fuzzy Control" ppo sac --reward_mode combined --obs_variant C04 --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv" --mC 74600000 --K 366 --Q_HP_Max 12000 --cop_heat 3.0 --prefer_best
```

| Algorithm | Avg. reward | Note |
|---|---|---|
| SAC (retrained) | 4.96 | best performer |
| PPO (retrained) | 4.57 | |
| MPC Control | 1.72 | now using correct building physics |
| PID Control | -103.29 | gains hardcoded for toy building scale, not retuned |
| PI Control | -103.29 | same issue |
| Fuzzy Control | -103.30 | same issue |

Note: `compare_results.py` will show duplicate/stale rows if old result CSVs aren't cleaned before rerunning — delete `results/combined/real_temp_data/*.csv` first, or always pass `--prefer_best` consistently.

## 8. Open items from original benchmark phase

| Item | Status |
|---|---|
| Rerun eval with `cop_heat=2.2` (realistic winter value vs. 3.0 used above) | Not yet run |
| Add `--gradient-steps` flag to increase real GPU work per env step | Not added |
| Retune PID/PI/Fuzzy gains for real building scale | Not done — currently invalid baseline |

## 9. Repo audit — what's real vs. placeholder (before starting the retrofit)

Decision: pivot from "benchmark reference" to full retrofit — MATLAB MPC and i4b RL tracks both currently broken, this repo is close enough to the target architecture to serve as the new base.

| Piece | Status | Detail |
|---|---|---|
| Weather data | Partial | `data/LLEC_outdoor_temperature_5min_data.csv` is real historical data, but from KIT's Karlsruhe site — not Langenhagen (your building) |
| Electricity prices | Real | `data/price_data_2025.csv` — real 2025 German day-ahead prices (ct/kWh + normalized), not synthetic |
| Building thermal model (1R1C) | Working | `mC`/`K`/`Q_HP_Max` already overridable via CLI (added earlier this session) |
| Heat pump COP | **Placeholder** | Confirmed in `base_building_gym.py` line ~682: one fixed constant (`self.COP_HEAT`/`self.COP_COOL`) applied to every action regardless of outdoor/flow temp. No datasheet, no interpolation. Biggest gap. |
| Indoor temp dynamics validation | Missing | No script validates that `mC`/`K` produce physically sane cooling/heating rates. Should be done before trusting any downstream results. |
| Train/test split | Partial | `--training` flag exists; not yet confirmed to be a proper chronological split vs. random |

## 10. Retrofit plan (per uploaded doc `LLECBuildingGym_Real_Building_HP_Retrofit.md`, scoped down)

Full 20-step retrofit plan considered but descoped — too much duplicate engineering vs. thesis timeline. Cherry-picked highest-value pieces:

1. **Validate indoor temp dynamics** — HP off, run several hours with real `mC`/`K`, sanity-check cooling rate against real physics. Not yet done.
2. **`HeatPump` class with real datasheet interpolation** — replace fixed `cop_heat` constant with `COP = f(T_out, T_flow)` built from the IDM AERO ALM 4-12 EN14511 table already extracted from `IDM_AEROALM412pagesdeleted.pdf` (see Section 4 of this changelog for the 55°C-flow COP table). Not yet started.
3. **Swap in real Langenhagen weather** — replace KIT Karlsruhe data with actual local weather history for the building. Not yet sourced.
4. Everything else in the uploaded retrofit doc (weather/price live APIs, occupancy, noise modeling, YAML config) deferred — lower priority than the two physics gaps above.

Next step recommended: start with the `HeatPump` datasheet-interpolation class (item 2), since the raw COP data is already on hand.

## 11. COP model decision: full 2D (T_air, T_flow), not a single fixed value

Two data points surfaced before implementing: (a) the EN14511 table in Section 4 matches `idm_alm412_datasheet.m`'s `COP_min` grid, not `COP_max` (which is what the earlier 1D `--scop` lookup used — now superseded); (b) a single "SCOP=5.12" seasonal-average figure was also floated as a simpler alternative. Decision: go full 2D `COP = f(T_out, T_flow)`, matching item 2's literal signature and the MATLAB model's fidelity — not the single-value shortcut, and not the 1D-fixed-55°C-flow curve built earlier in the session.

## 12. `HeatPump` class + weather-compensated flow temperature (item 2, done)

Ported from the MATLAB retrofit project into `llec_building_gym/utils/heat_pump.py`:

| Function | Ported from | Does |
|---|---|---|
| `heating_curve(T_air)` | `HEATPUMP/heating_curve.m` | EN 442 panel-radiator exponent law: flow temp = 55°C at ≤-10°C outdoor, floored at 35°C, exponent 1.3 in between |
| `HeatPump.get_performance(T_air, T_flow, u)` | `HEATPUMP/idm_alm412_interpolants.m` | 2D interpolation over the datasheet grid (`T_supply` in {35,45,50,55}°C — never exceeds 55°C so the NaN-filled 60/70°C rows are unused) + duty-cycle averaging below the ~4kW modulation floor |

Wired into `base_building_gym.py`: `--scop` now drives this full model (weather-compensated flow temp + full grid) instead of the old fixed-55°C-flow 1D lookup. `P_HP_el` now comes directly from the model's `Pel` output rather than `Q_HP_Max/COP` (avoids a divide-blowup near `action≈0`).

**Bug caught during smoke testing:** the datasheet's `T_air` axis was accidentally reversed twice (defined ascending, then reversed again "for clarity"), so COP came out backwards — e.g. 2.02 at +20°C outdoor (should be ~5.4, the machine's best-efficiency point) and 2.16 at -20°C (should be its worst, ~1.7). Fixed and re-verified across the full range:

| T_air | T_flow (weather-compensated) | COP @ full load |
|---|---|---|
| -20°C | 55.0°C | 1.66 |
| -7°C | 52.3°C | 2.16 |
| 2°C | 43.6°C | 2.87 |
| 10°C | 35.0°C | 4.98 |
| 20°C | 35.0°C | 5.43 |

Monotonically increasing with outdoor temp, as expected — and meaningfully better than the old fixed-55°C assumption at mild temps (e.g. 2.87 vs. 2.40 at 2°C), since the weather curve now drops the flow temp when the full 55°C isn't needed.

Also fixed while touching this code: `outdoor_temperature_path` CSV loading used to `outdoor_df.values.flatten()` the whole file — fine for the old single-column format, but would silently corrupt a `timestamp, T_out_C` file by interleaving strings and floats. Now selects the sole numeric column.

## 13. Real Langenhagen weather + price data (item 3, done)

User supplied real historical data for the actual building location: `langenhagen-data/weather_langenhagen_2025.csv` (15-min, full year 2025) and `langenhagen-data/electricity_prices_2025.csv` (15-min, Tibber EUR/kWh, full year 2025, German comma-decimal format) — replacing the KIT Karlsruhe weather data (wrong location) and the shorter (~116-day) price series used previously.

`prepare_langenhagen_data.py` (new): resamples both to the environment's 5-min control step (linear interpolation) and computes `price_normalized` (global min-max scale). One DST-fallback duplicate timestamp in the price data (Oct clock change) handled by keeping the first occurrence. Output:
- `data/langenhagen_outdoor_temperature_5min.csv` — 105,109 rows, range -8.4°C to 36.8°C
- `data/langenhagen_price_2025.csv` — 105,118 rows, raw range [-0.086, 0.906] EUR/kWh (includes real negative-price events) normalized to [0,1]

Full year at 5-min resolution for both — a big train/eval-split improvement over the old data (previously ~76 usable eval days at most; the old price series alone was only ~116 days total).

Note: weather and price are still sampled as *independent* random episode windows (pre-existing architecture, unchanged) — a "cold day" episode isn't guaranteed to pair with that same real day's price. Flagged, not fixed; would need a coordinated-sampling change to `Building.reset()`.

## 14. Dynamics validation (item 1, done)

`validate_dynamics.py` (new): runs `Building` with the heat pump off and a constant outdoor temperature, fits the exponential free-response time constant from the simulated trajectory, and compares it to the analytic `tau = mC / (dt_scale * K)` seconds.

```
python validate_dynamics.py --mC 74600000 --K 366 --dt-scale 1.0 --hours 48
```

| | |
|---|---|
| Expected tau (mC/(dt_scale·K)) | 203,825 s (56.62 h) |
| Fitted tau from simulation | 203,675 s (56.58 h) |
| Relative error | 0.074% — **PASS** |

Confirms the `dt_scale=1.0` fix (section 10) is dimensionally correct, not just "seems more reasonable."

## 15. Retraining with full 2D COP + real Langenhagen data

Command (repeated for `--algorithm ppo`):
```
python run_train_rl.py --algorithm sac --reward_mode combined --obs_variant C04 --training --seed 42 --timesteps 1e6 --num-envs 4 --mC 74600000 --K 366 --Q_HP_Max 12000 --cop_heat 2.2 --cop_cool 2.2 --scop --dt-scale 1.0 --outdoor-temperature-path data/langenhagen_outdoor_temperature_5min.csv --energy-price-path data/langenhagen_price_2025.csv
```
SAC: 1,000,000 steps in 123.4 min (~130 fps), final eval `mean_reward` 79.4. PPO: 1,001,088 steps in 89.6 min (~170 fps), final `ep_rew_mean` 53.3. Both noticeably higher training reward than the previous (1D-COP, KIT-weather) run — consistent with the full 2D model's much better COP at mild outdoor temps (weather-compensated flow drops below 55°C when not needed).

## 16. Final benchmark with real Langenhagen data + full 2D COP

Command:
```
python run_evaluation.py --algorithms sac ppo "PI Control" "PID Control" "Fuzzy Control" "MPC Control" --reward_mode combined --obs_variant C04 --mC 74600000 --K 366 --Q_HP_Max 12000 --cop_heat 2.2 --cop_cool 2.2 --scop --dt-scale 1.0 --outdoor_temperature_path data/langenhagen_outdoor_temperature_5min.csv --energy_price_path data/langenhagen_price_2025.csv --prefer_best
```

| Algorithm | Avg. reward | Runtime |
|---|---|---|
| **sac** | **100.54** | 36.4s |
| MPC Control | 69.26 | 147.7s |
| ppo | 63.72 | 36.3s |
| Fuzzy Control | 62.94 | 23.8s |
| PI Control | -21.13 | 31.1s |
| PID Control | -21.14 | 32.0s |

**SAC now beats MPC** (previously MPC led, 50.26 vs SAC's 36.11, in the 1D-COP run). Verified this is a real effect, not a bug — reward component breakdown:

| | Comfort (normalized) | Economic (normalized) | Total reward |
|---|---|---|---|
| MPC | 0.458 (slightly better) | **-0.217** | 0.241 |
| SAC | 0.432 | **-0.083** | 0.349 |

MPC's internal optimizer still assumes `price × |action|` (COP-blind, section 16's still-open item) — it doesn't know heating is often *cheap* now (COP up to 5.4 at mild temps), so it heats more aggressively than the true reward rewards (mean `|action|` 0.79 vs SAC's 0.30), buying marginally better comfort at ~2.6x the true economic cost. SAC, trained directly against the real COP-aware reward, finds a better tradeoff. This effect is much larger than in the previous run because the full 2D model's COP varies far more (1.66–5.43) than the old fixed-55°C curve (1.66–3.58), making MPC's flat-cost assumption a worse approximation of reality. PID/PI improved from -27.85 to -21.1 (cheaper COP softens their wasteful saturation, but the gains structure itself is still untuned).

Models: `models/combined/C04/{sac,ppo}_model_seed42.zip` + `best_{sac,ppo}/best_model.zip`, committed to worktree branch `worktree-real-building-scop-fix`.

## 17. TensorBoard logging added to run_train_rl.py

`model.learn()` only gave a console progress bar / periodic text blocks — no live view. Added `--tensorboard-log <dir>` (opt-in, off by default), wired into each SB3 model's `tensorboard_log` param with a per-run `tb_log_name` (`<algorithm>_<reward_mode>_<obs_variant>_seed<seed>`) so repeated/concurrent runs don't overwrite each other's logs.

```
python run_train_rl.py ... --tensorboard-log tb_logs
tensorboard --logdir tb_logs   # http://localhost:6006
```

## 18. Open items (updated)

| Item | Status |
|---|---|
| Retune PID/PI/Fuzzy gains for real building scale | Still not done |
| **MPC's own economic objective ignores COP** | Confirmed to materially matter now (section 16) — this is the clear next fix if MPC should be compared fairly |
| `cop_cool` | Still a fixed assumption (2.2) — no cooling data on the IDM ALM 4-12 datasheet, cooling isn't part of the 2D model |
| "Fix all controllers to have the same base parameters" | Checked: PID/PI/Fuzzy hold no copy of `mC/K/Q_HP_Max` at all (pure reactive controllers on temp-deviation + hand-tuned gains) — nothing to desync, unlike MPC's earlier bug. The real gap for them is untuned gains, not parameter mismatch. |
| Weather/price coordinated sampling (same real day for both) | Done in section 22 |
| Merge worktree branch into main checkout | Done (section 19) |

## 19. Project restructure

Worktree branch `real-building-scop-fix` (the up-to-date code) promoted to the single top-level project; old top-level copy, both `.git` dirs, `.claude/`, `__pycache__`, `setup.bat` (re-cloned upstream over the folder), empty slurm log dirs and the dead `plot-paper/` notebooks moved to `Documents/buildinggym_old_backup/`. Repo re-initialised by the user (`opti-heat-building-gym`). README shortened to building params, results, current state.

## 20. 2026 test data + continuous inference

Test data only — never used for training.

| File | Content |
|---|---|
| `langenhagen-data/weather_langenhagen_2026.csv` | Open-Meteo hourly, 1 Jan – 31 Aug 2026, −9.9 … 37.8 °C (colder than any 2025 value) |
| `langenhagen-data/electricity_prices_2026.csv` | Tibber 15-min, 1 Jan – 12 Sep 2026, trimmed to `Datum von, Datum bis, Tibber 2026` (full 30-column workbook export kept in `buildinggym_old_backup/`). **No prices 7 Jun – 6 Jul.** Range −0.385 … 0.906 EUR/kWh |

New scripts:
- `prepare_langenhagen_data.py --year 2026` — resamples to 5 min (temperature linear, prices step), normalises prices with the **2025 training** min/max, fills only gaps ≤ 75 min (DST hour), leaves missing days empty.
- `run_inference.py --year 2026 --algorithm <algo> --model best|final` — walks every day from midnight in calendar order with the real weather + prices, indoor temperature carried over day to day (one continuous run), skips days without data (incl. 30 h price look-ahead), writes `results/inference/2026/<algo>_<model>_{live,summary}.png` + step/daily CSVs.

## 21. Why the 2025 benchmark (sections 15–16) was misleading

First 2026 run with the C04 SAC model: house drifted to 5–10 °C in winter (mean |T_in − T_set| 7.36 K). Root causes found:

| Bug | Effect |
|---|---|
| `SeedWrapper` passed the **same seed on every reset** | every episode of a training env replayed the same day → C04 models trained on only **4 distinct days** |
| **Heating/cooling sign reversed** in `update_Tin` (`T_in -= dT` with `+Q_HP` inside) and MPC's model | positive action cooled the house, while the 2D COP model charged positive actions as heating |
| 1-day episodes starting at ~21 °C | with τ ≈ 56 h the house only loses ~4–5 K per day → "do nothing" scored well; all 10 eval episodes ended colder than they started, mean action ≈ 0 |
| `np.random.seed(42)` inside `_load_energy_price` on every reset | "random" initial T_in always 21.0 |
| "day" = `23*3600` s | price index drifted away from time of day |
| weather window random, independent of price day | weather and price from different days |
| DDPG/TD3 default `train_freq=(1,"episode")` | crash with >1 env |

## 22. Fixes + new observation variant C05

- `update_Tin`: `T_in += dt_scale·dt/mC·(−K(T_in−T_out) + Q_HP)`; MPC model same. +1 at 0 °C for 1 h: 20.00 → 20.22 °C.
- `SeedWrapper`: seed + 10000·n on each reset (reproducible, different day each episode).
- `--episode-days` (default 7) in `run_train_rl.py`; initial T_in uniform 16–23 °C (`self.np_random`).
- 24 h days, episodes start at midnight; weather window = same calendar days as prices; multi-day episodes fit inside the data.
- Local `RandomState(42)` for the train/eval day split (same split as before).
- `C05` = C04 + outdoor temperature now and at +1/3/6/12/24 h.
- DDPG/TD3: `train_freq=1, gradient_steps=1`; `td3` added to CLI.
- `validate_dynamics.py` still PASS (0.074 %).

## 23. Retraining (C05, 7-day episodes, 2025 data)

All five in parallel, 1M steps, 4 envs each, `--tensorboard-log tb_logs` (TensorBoard at http://localhost:6006):

| Algo | Time | Eval reward per 7-day episode: first / best (step) / last |
|---|---|---|
| SAC | 164.6 min | 443 / 876 (350k) / 735 |
| TD3 | 141.3 min | 1015 / 1015 (50k) / 931 |
| DDPG | 143.6 min | 690 / 777 (900k) / 173 |
| PPO | 135.2 min | −93 / 558 (250k) / 424 |
| A2C | 118.5 min | 399 / 399 (50k) / 323 |

Eval is noisy (each eval = 5 different held-out weeks), so "best" can be an early lucky snapshot (TD3, A2C). Models in `models/combined/C05/`; old C04 models untouched.

## 24. 2026 test results (1 Jan – 31 Aug, 211 days, continuous)

| Model | Reward/day | Mean \|T_in−T_set\| | Within 1 K | kWh | EUR |
|---|---|---|---|---|---|
| **SAC best** | **102.8** | **1.47 K** | **57.9 %** | 9,495 | 3,215 |
| SAC final | 82.7 | 1.60 K | 53.9 % | 11,808 | 4,006 |
| TD3 final | 73.8 | 6.35 K | 45.2 % | 9,018 | 3,044 |
| DDPG best | 69.9 | 3.61 K | 45.9 % | 9,774 | 3,264 |
| PPO best | 57.6 | 4.76 K | 29.1 % | 2,748 | 920 |
| A2C best | 57.6 | 2.05 K | 32.1 % | 4,986 | 1,708 |
| old C04 SAC | 33.2 | 7.36 K | 18.9 % | 3,382 | 1,151 |

SAC best holds ~18 °C all winter (old model: 5–10 °C). **But 38 % of its energy (3,627 kWh) is cooling**, and it heats even in Jul/Aug: the BA_RES setpoint alternates 21.7/18.3 °C and `exp(−|T_in−T_set|)` penalises too-warm like too-cold, so the agent heats to 21.7 then pays to cool to 18.3 daily. kWh/EUR above are therefore inflated.

## 25. New hard rules from the building owner (to implement before next retraining)

1. The heat pump **can only heat** — no cooling.
2. Heating period **1 Oct – 30 Apr** only; the heat pump is off outside it regardless of temperatures.
3. Comfort band **19–22 °C** (day 20–22, night down to 18; 19–22 chosen as one band).

Clarified with the owner: 19 °C is never acceptable to undercut (overheating > 22 °C penalised the same), band 19–22 °C around the clock, train only on heating-period days and skip May–Sep in tests, no domestic hot water, classical controllers re-run under the same rules, retrain all five.

Implemented as new reward mode `band` (env id `LLEC-HeatPumpHouse-1R1C-Band-v0`; `combined` untouched):
- action space `[0, 1]`; negative actions and any action outside Oct–Apr forced to 0 (`Building.heating_allowed()`, mask from the price CSV timestamps)
- train/eval start days only where the whole 7-day episode is inside Oct–Apr (2025: 161 train / 38 eval start days)
- reward = energy cost (normalized, COP-aware) − 10 × K outside [19, 22] per step (1 K below costs ~30× the worst per-step energy cost); observation deviation measured from band centre 20.5 °C
- PI/PID/Fuzzy were still tuned for the old reversed sign (would heat when too warm) → outputs negated / Fuzzy's normal path made consistent with its extreme branches
- MPC `band` mode: LP (cbc) mirroring the reward — COP-aware cost `price·a·Pel_full(T_out)`, band slack penalty, heating only, 0 outside the heating period; 24 h horizon, re-planned hourly (0.6 s/solve); optional `--mpc-margin`
- `run_inference.py`: `--reward_mode band` (default), classical controllers (`pi pid fuzzy mpc`), band metrics (time in band, K·h below/above), band shading in plots, outputs in `results/inference/2026/band/`

Known simplification: thermal output is always `action × 12 kW`; the real IDM delivers ~9.6 kW at −10 °C, so the coldest days are easier in simulation than in reality.

## 26. Band retraining + 2026 test under the hard rules

Training (C05, 7-day episodes, 1M steps, Oct–Apr 2025): SAC 167 min, DDPG 144, TD3 142, PPO 136, A2C 119. Best eval per 7-day episode: A2C −195, SAC −202, TD3 −237, DDPG −239, PPO −292. Models in `models/band/C05/`.

2026 test = 1 Jan – 30 Apr 2026 (120 days, continuous, real weather + Tibber prices). Sorted by cost:

| Controller | In 19–22 °C | Below 19 °C | Mean T_in | kWh | EUR | Avg EUR/kWh paid |
|---|---|---|---|---|---|---|
| MPC | 84.7 % | 34.8 K·h | 19.73 | 4,499 | 1,392 | 0.309 |
| **MPC, 0.3 K margin** | **100 %** | 0.06 K·h | 19.96 | 4,584 | **1,425** | **0.311** |
| **Fuzzy** | **100 %** | 0 | 19.85 | 4,387 | **1,457** | 0.332 |
| **A2C best** | **100 %** | 0 | 19.92 | 4,426 | **1,478** | 0.334 |
| PPO best | 86.3 % | 299 K·h | 20.35 | 4,447 | 1,480 | 0.333 |
| A2C final | 44.3 % | 520 K·h | 18.89 | 4,460 | 1,496 | 0.335 |
| **SAC best** | **100 %** | 0 | 19.76 | 4,519 | **1,511** | 0.334 |
| DDPG best | 99.9 % | 0.2 K·h | 19.86 | 4,681 | 1,549 | 0.331 |
| PI / PID | 100 % | 0 | 20.50 | 4,665 | 1,552 | 0.333 |
| DDPG final | 98.2 % | 2.9 K·h | 19.73 | 4,760 | 1,567 | 0.329 |
| TD3 final | 100 % | 0 | 20.49 | 4,811 | 1,591 | 0.331 |
| PPO final | 70.8 % | 1.8 K·h | 20.83 | 4,841 | 1,615 | 0.334 |
| TD3 best | 100 % | 0 | 20.38 | 4,850 | 1,622 | 0.334 |
| SAC final | 98.2 % | 0 | 21.13 | 4,940 | 1,660 | 0.336 |

Findings:
- Band-compliant spread is small: MPC+margin −8.2 % vs PI, Fuzzy −6.2 %, A2C best −4.8 %, SAC best −2.7 %.
- RL agents learned the hard rules (A2C/SAC/TD3 best: 100 % in band) but **don't time heating to prices** (0.334 EUR/kWh, same as PI); their savings come only from sitting lower in the band. MPC pays 0.311 EUR/kWh by pre-heating the 56 h-time-constant house in cheap hours (it has perfect 24 h weather + price forecasts; RL's C05 also sees perfect forecasts).
- Plain MPC sits exactly on 19.0 °C and the noisy temperature measurement pushes it below; a 0.3 K margin fixes it.
- PPO and A2C final break the band; "best" snapshot is the better pick for A2C/SAC, "final" for TD3.
- Comparison chart: `results/inference/2026/band/comparison.png` (+ `comparison.csv`).

Next options: make RL exploit prices (e.g. higher `gamma` for the 56 h time constant, smaller band penalty scale relative to cost, longer training), or use MPC+margin as the controller; model heat pump capacity vs outdoor temperature.
