# Opti-Heat Building Gym

RL vs. classical control of a real heat pump: a multi-family building in **Langenhagen (Hannover)** with an **IDM AERO ALM 4-12** air-source heat pump. The simulation runs in 5-min steps, and one episode is one day (288 steps).

## Building parameters

| Parameter | Value |
|---|---|
| Thermal capacitance `mC` | 74,600,000 J/K |
| Heat-loss coefficient `K` | 366 W/K |
| Max heat output `Q_HP_Max` | 12,000 W |
| Room setpoint | 20.2 °C |
| Time constant `mC/K` | ≈ 56.6 h (validated: 0.074 % error) |
| Heat pump COP | 2D datasheet model, from 1.66 (−20 °C) to 5.43 (+20 °C) |
| Weather / prices | Langenhagen 2025 (5-min), Tibber dynamic prices |

⚠️ The script defaults are still the old toy building, so always pass:
```
--mC 74600000 --K 366 --Q_HP_Max 12000 --cop_heat 2.2 --cop_cool 2.2 --scop --dt-scale 1.0
```

## Results

Scores are the average reward per one-day episode (higher is better), with a combined comfort + cost reward and observation variant `C04`.

**Current run: 2D COP + real Langenhagen data**

| Controller | Avg. reward |
|---|---|
| **SAC** | **100.54** |
| MPC | 69.26 |
| PPO | 63.72 |
| Fuzzy | 62.94 |
| PI | −21.13 |
| PID | −21.14 |

SAC beats MPC because MPC's cost model ignores COP. It heats harder (mean |action| 0.79 vs 0.30) and pays about 2.6× the energy cost for only slightly better comfort.

**Earlier runs** (different physics and data, so not comparable with the current run):

| Run | MPC | PPO | SAC | Fuzzy | PI/PID |
|---|---|---|---|---|---|
| Toy building (repo defaults) | 267.26 | 250.99 | 250.30 | 214.96 | 251.14 |
| Real params, 1D COP, KIT weather | 50.26 | 43.10 | 36.11 | 34.27 | −27.85 |

## Current state

**Done:** real building parameters, a validated thermal model, the 2D heat pump model, real Langenhagen data, and SAC/PPO retrained on this building (`models/combined/C04/`).

**Open:**
1. Make MPC's cost model use COP, so the comparison with SAC is fair.
2. Retune the PI/PID/Fuzzy gains for this building.
3. Sample weather and price from the same day in each episode.
4. Change the script defaults to this building.

Only the `C04` SAC/PPO models are current. Every other model in `models/` is from the old toy building. The full change history is in [CHANGELOG.md](CHANGELOG.md).
