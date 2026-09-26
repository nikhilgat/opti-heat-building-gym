"""
compare_inference.py - One table + chart from all run_inference.py results in a folder
(RL agents, classical controllers, heating curve and its oil-boiler equivalent).

Usage:
    python compare_inference.py results/inference/2026/band/C06
Writes <folder>/comparison.csv and <folder>/comparison.png.
"""
import glob
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OIL_KWH_PER_L, BOILER_EFF, OIL_EUR_PER_L = 9.97, 0.90, (0.95, 1.42)  # same as run_inference.py
LO, HI = 19.0, 22.0

folder = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(folder, "*_steps.csv"))):
    s = pd.read_csv(f)
    kwh = s.P_el_W.sum() / 12000
    rows.append(dict(
        controller=os.path.basename(f)[:-10],
        in_band=((s.T_in >= LO) & (s.T_in <= HI)).mean() * 100,
        below_Kh=(LO - s.T_in).clip(lower=0).sum() / 12, above_Kh=(s.T_in - HI).clip(lower=0).sum() / 12,
        mean_T_in=s.T_in.mean(), kWh=kwh, backup_kWh=s.backup_W.sum() / 12000 if "backup_W" in s else 0.0,
        EUR=s.cost_eur.sum(), EUR_per_kWh=s.cost_eur.sum() / kwh,
    ))
    if rows[-1]["controller"] == "curve" and "heat_W" in s:
        litres = s.heat_W.sum() / 12000 / BOILER_EFF / OIL_KWH_PER_L
        for tag, eur in zip(("low", "high"), OIL_EUR_PER_L):
            rows.append(dict(controller=f"oil_boiler_{tag}_{eur:.2f}EUR_L", in_band=rows[-1 if tag == "low" else -2]["in_band"],
                             below_Kh=0.0, above_Kh=0.0, mean_T_in=float("nan"), kWh=float("nan"), backup_kWh=0.0,
                             EUR=litres * eur, EUR_per_kWh=float("nan")))
t = pd.DataFrame(rows).sort_values("EUR").reset_index(drop=True)
t.to_csv(os.path.join(folder, "comparison.csv"), index=False)
pd.set_option("display.width", 200)
print(t.round(2).to_string())

ok = (t.below_Kh < 1) & (t.in_band >= 97)
fig, ax = plt.subplots(figsize=(11, 0.35 * len(t) + 1.5))
ax.barh(t.controller, t.EUR, color=["tab:green" if o else "tab:red" for o in ok]); ax.invert_yaxis()
ax.set_xlim(t.EUR.min() * 0.9, t.EUR.max() * 1.12)
for i, r in t.iterrows():
    ax.text(r.EUR, i, f" {r.EUR:,.0f} € | {r.in_band:.0f}% in band | backup {r.backup_kWh:,.0f} kWh", va="center", fontsize=8)
ax.set_xlabel("cost, EUR"); ax.set_title("2026 test (green = keeps 19-22 °C, red = breaks it)")
fig.tight_layout(); fig.savefig(os.path.join(folder, "comparison.png"), dpi=100)
