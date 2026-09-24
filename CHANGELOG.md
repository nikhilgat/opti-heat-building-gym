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
| Weather/price coordinated sampling (same real day for both) | Not done — see section 13 note |
| Merge worktree branch into main checkout | Not done |
