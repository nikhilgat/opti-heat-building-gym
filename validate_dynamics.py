"""
validate_dynamics.py - Sanity-check that Building's indoor-temperature
dynamics (mC, K, dt_scale) match the expected physical thermal time
constant, with the heat pump off (free-floating decay/rise toward outdoor
temperature).

For a 1R1C model with constant T_out, HP off, the analytic solution is:
    T_in(t) = T_out + (T_in(0) - T_out) * exp(-t / tau)
where tau = mC / (dt_scale * K) seconds (see base_building_gym.py's
update_Tin: dT = dt_scale*(dt/mC)*(K*(Tin-Tout)+Q_HP), which for Q_HP=0
integrates to this exponential with rate dt_scale*K/mC per second). Note
the repo's historical default dt_scale=1e-3 makes tau = 1000*mC/K, i.e.
1000x longer than the "textbook" mC/K -- that factor is exactly the bug
documented in CHANGELOG.md section 10.

Usage:
    python validate_dynamics.py --mC 74600000 --K 366 --dt-scale 1.0
"""
import argparse
import numpy as np
from llec_building_gym.envs.base_building_gym import Building

parser = argparse.ArgumentParser()
parser.add_argument("--mC", type=float, default=74_600_000)
parser.add_argument("--K", type=float, default=366)
parser.add_argument("--dt-scale", type=float, dest="dt_scale", default=1.0)
parser.add_argument("--hours", type=float, default=48)
args = parser.parse_args()

control_step = 300  # 5 min, matches the environment default
simulation_time = int(args.hours * 3600)

b = Building(
    mC=args.mC, K=args.K, Q_HP_Max=12000,
    simulation_time=simulation_time, control_step=control_step,
    schedule_type="simple",  # flat T_out so the free-response is a clean exponential
    dt_scale=args.dt_scale,
)
b.reset(T_in=25.0, seed=1)
b.T_out[:] = 5.0  # override to a single constant outdoor temp for a clean test

expected_tau_s = args.mC / (args.dt_scale * args.K)
print(f"mC={args.mC:.0f} J/K, K={args.K} W/K, dt_scale={args.dt_scale}")
print(f"Expected time constant tau = mC/(dt_scale*K) = {expected_tau_s:.0f}s "
      f"({expected_tau_s/3600:.2f} h)")

T_out = 5.0
T_in0 = b.T_in
t = 0.0
log_t, log_T = [0.0], [T_in0]
while not b.is_done():
    b.update_Tin(action=0.0)  # HP off
    t += control_step
    log_t.append(t)
    log_T.append(b.T_in)

log_t = np.array(log_t)
log_T = np.array(log_T)

# Fit tau from the simulated trajectory: ln(T_in - T_out) should be linear in t
# with slope -1/tau (only valid while T_in > T_out).
mask = log_T > T_out + 1e-6
if mask.sum() > 2:
    slope, intercept = np.polyfit(log_t[mask], np.log(log_T[mask] - T_out), 1)
    fitted_tau_s = -1.0 / slope
else:
    fitted_tau_s = float("nan")

print(f"Simulated:  T_in {T_in0:.2f}C -> {log_T[-1]:.2f}C over {args.hours:.0f}h "
      f"(T_out={T_out:.1f}C)")
print(f"Fitted time constant from simulation: {fitted_tau_s:.0f}s ({fitted_tau_s/3600:.2f} h)")
rel_err = abs(fitted_tau_s - expected_tau_s) / expected_tau_s
print(f"Relative error vs. analytic tau: {rel_err*100:.3f}%  "
      f"{'PASS' if rel_err < 0.01 else 'FAIL'}")

# Quick reference table across dt_scale, so both the toy calibration and
# real-building calibration are visible at a glance.
print("\nReference (K in W/K, mC in J/K):")
for label, mc, k, ds in [
    ("toy default", 300, 20, 1e-3),
    ("real building (buggy, pre-fix)", 74_600_000, 366, 1e-3),
    ("real building (fixed)", 74_600_000, 366, 1.0),
]:
    tau = mc / (ds * k)
    unit = "s" if tau < 3600 else ("h" if tau < 86400 else "days")
    val = tau if unit == "s" else (tau / 3600 if unit == "h" else tau / 86400)
    print(f"  {label:35s} tau = {val:.2f} {unit}")
