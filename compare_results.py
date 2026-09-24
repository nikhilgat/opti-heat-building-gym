"""
compare_results.py - Build a summary table (Algorithm | Avg. reward | Runtime)
from the CSVs written by run_evaluation.py.

Usage:
    python compare_results.py --reward_mode combined --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"
"""
import argparse
import os
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--reward_mode", default="temperature", choices=["temperature", "combined"])
parser.add_argument("--outdoor_temperature_path", default=None,
                     help="Pass the same value you used for run_evaluation.py (or omit if you used synthetic data).")
args = parser.parse_args()

subdir = "real_temp_data" if args.outdoor_temperature_path else "synthetic_temp_data"
results_dir = os.path.join("results", args.reward_mode, subdir)

rows = []
for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
    df = pd.read_csv(path)
    algo = df["algorithm"].iloc[0]
    avg_reward = df.groupby("episode")["reward"].sum().mean()
    runtime = df["evaluation_time_sec"].iloc[0]
    rows.append((algo, avg_reward, runtime))

if not rows:
    print(f"No result CSVs found in {results_dir}")
else:
    rows.sort(key=lambda r: -r[1])
    name_w = max(len(r[0]) for r in rows) + 2
    print(f"{'Algorithm':<{name_w}} {'Avg. reward':>12} {'Runtime':>10}")
    print("-" * (name_w + 25))
    for algo, avg_reward, runtime in rows:
        print(f"{algo:<{name_w}} {avg_reward:>12.2f} {runtime:>9.1f}s")
