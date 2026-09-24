"""
heat_pump.py - IDM AERO ALM 4-12 datasheet-based heat pump model.

Ported from the MATLAB retrofit project (HEATPUMP/heating_curve.m +
HEATPUMP/idm_alm412_interpolants.m / idm_alm412_datasheet.m):

- ``heating_curve(t_air)``: weather-compensated radiator flow temperature
  (EN 442 panel-radiator exponent law), replacing the fixed-55degC-flow
  assumption used by the earlier (1D) --scop lookup.
- ``HeatPump``: full 2D (T_air, T_flow) capacity/COP interpolation with
  duty-cycle averaging below the compressor's minimum-modulation floor,
  mirroring idm_alm412_interpolants.m's interpolatePerformance().

Only the T_supply in {35, 45, 50, 55} degC datasheet rows are used, since
heating_curve() never commands a flow temperature above the 55degC design
point -- this avoids the NaN-filled 60/70degC rows entirely (no NaN
handling needed, unlike the MATLAB version).
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# --- Weather-compensated flow temperature (heating_curve.m) ---
_T_ROOM_DESIGN = 20.0       # radiator design room temp (55/45/20 hydraulic scheme)
_T_AIR_DESIGN = -10.0       # design outdoor temp
_T_SUPPLY_DESIGN = 55.0     # design supply temp
_RADIATOR_EXPONENT = 1.3    # EN 442 panel-radiator exponent
_MIN_USEFUL_SUPPLY_C = 35.0 # floor, matches the old linear curve's upper endpoint
_HEATING_LIMIT_C = 15.0     # Heizgrenze


def heating_curve(t_air):
    """Weather-compensated radiator supply temperature (degC) at outdoor
    air temperature ``t_air`` (degC). Also returns whether t_air is below
    the Heizgrenze (informational only; does not gate heat pump output
    here -- the environment's action already controls that)."""
    if t_air <= _T_AIR_DESIGN:
        t_supply = _T_SUPPLY_DESIGN
    else:
        q_frac = (_T_ROOM_DESIGN - t_air) / (_T_ROOM_DESIGN - _T_AIR_DESIGN)
        q_frac = max(q_frac, 0.0)
        t_supply = _T_ROOM_DESIGN + (_T_SUPPLY_DESIGN - _T_ROOM_DESIGN) * q_frac ** (1.0 / _RADIATOR_EXPONENT)
        t_supply = max(t_supply, _MIN_USEFUL_SUPPLY_C)
    heating_on = t_air < _HEATING_LIMIT_C
    return t_supply, heating_on


# --- IDM AERO ALM 4-12 datasheet grid, T_supply in {35,45,50,55} only ---
# (HEATPUMP/idm_alm412_datasheet.m; T_air sorted ascending to match
# RegularGridInterpolator's requirement; these 4 rows have no NaNs.)
_T_AIR_ASC = np.array([-20, -15, -10, -7, 2, 7, 10, 12, 15, 20], dtype=float)
_T_SUPPLY = np.array([35, 45, 50, 55], dtype=float)

_PTH_MAX_KW = np.array([
    [7.95, 8.97, 9.85, 10.30, 11.80, 12.41, 12.95, 13.10, 13.12, 13.24],
    [7.94, 8.84, 9.72, 10.14, 11.27, 12.06, 12.51, 12.60, 12.68, 12.75],
    [7.86, 8.76, 9.65, 10.00, 11.01, 11.87, 12.29, 12.35, 12.46, 12.51],
    [7.99, 8.67, 9.58, 9.85, 10.74, 11.68, 12.07, 12.10, 12.24, 12.26],
])

_PTH_MIN_KW = np.array([
    [4.02, 4.00, 4.04, 4.02, 4.07, 4.04, 4.03, 3.89, 4.00, 3.98],
    [3.99, 4.02, 4.01, 4.00, 4.05, 4.07, 4.10, 3.98, 4.03, 4.00],
    [4.05, 4.06, 4.07, 4.08, 4.03, 4.04, 4.07, 3.96, 4.10, 4.05],
    [4.12, 4.09, 4.12, 4.16, 4.00, 4.01, 4.04, 3.94, 4.17, 4.10],
])

_PEL_MAX_KW = np.array([
    [3.94, 3.82, 3.79, 3.73, 3.64, 3.14, 2.60, 2.48, 2.45, 2.44],
    [4.42, 4.35, 4.21, 4.14, 4.00, 3.52, 3.17, 2.94, 2.89, 2.85],
    [4.61, 4.63, 4.45, 4.44, 4.22, 3.78, 3.46, 3.23, 3.15, 3.10],
    [4.82, 4.95, 4.72, 4.80, 4.48, 4.10, 3.83, 3.61, 3.48, 3.42],
])

_PEL_MIN_KW = np.array([
    [1.83, 1.60, 1.34, 1.24, 0.92, 0.76, 0.68, 0.61, 0.62, 0.61],
    [2.28, 1.91, 1.68, 1.54, 1.21, 1.03, 0.93, 0.85, 0.84, 0.82],
    [2.49, 2.12, 1.88, 1.74, 1.36, 1.16, 1.07, 0.97, 0.97, 0.94],
    [2.73, 2.36, 2.12, 1.99, 1.55, 1.33, 1.24, 1.14, 1.14, 1.11],
])


def _make_interp(grid):
    return RegularGridInterpolator((_T_SUPPLY, _T_AIR_ASC), grid, method="linear")


# Worst-case (maximum) electrical draw across the used grid, at full load.
# Used by BaseBuildingGym to normalize the economic reward term.
MAX_PEL_KW = float(_PEL_MAX_KW.max())


class HeatPump:
    """IDM AERO ALM 4-12 duty-cycle-averaged performance model.

    ``get_performance(t_air, t_supply, u)`` mirrors
    idm_alm412_interpolants.m's interpolatePerformance(): u in [0,1] is the
    requested fraction of MAXIMUM thermal capacity at the given
    (t_air, t_supply) operating point. Below the compressor's minimum
    modulation floor (Pth_min), output is duty-cycle averaged rather than
    held at a nonexistent "off state that still outputs ~4kW" -- see that
    file's header comment for why this matters.
    """

    def __init__(self):
        self._pth_max = _make_interp(_PTH_MAX_KW)
        self._pth_min = _make_interp(_PTH_MIN_KW)
        self._pel_max = _make_interp(_PEL_MAX_KW)
        self._pel_min = _make_interp(_PEL_MIN_KW)
        self._t_air_range = (_T_AIR_ASC.min(), _T_AIR_ASC.max())
        self._t_supply_range = (_T_SUPPLY.min(), _T_SUPPLY.max())

    def _clamp(self, t_air, t_supply):
        # RegularGridInterpolator raises outside its domain; clamp to mimic
        # MATLAB griddedInterpolant's 'nearest' extrapolation.
        t_air_c = min(max(t_air, self._t_air_range[0]), self._t_air_range[1])
        t_supply_c = min(max(t_supply, self._t_supply_range[0]), self._t_supply_range[1])
        return t_air_c, t_supply_c

    def get_performance(self, t_air, t_supply, u):
        """Returns (Pth_kW, Pel_kW, COP, duty) for requested capacity
        fraction ``u`` in [0,1] at outdoor temp ``t_air`` and flow temp
        ``t_supply`` (both degC)."""
        u = min(max(u, 0.0), 1.0)
        t_air_c, t_supply_c = self._clamp(t_air, t_supply)
        point = (t_supply_c, t_air_c)

        pth_max = float(self._pth_max(point))
        pth_min = float(self._pth_min(point))
        pel_max = float(self._pel_max(point))
        pel_min = float(self._pel_min(point))
        if not (pth_max > pth_min):
            pth_max = pth_min

        pth_req = u * pth_max

        if pth_req <= 0:
            pth, pel, duty = 0.0, 0.0, 0.0
        elif pth_req >= pth_min:
            span = pth_max - pth_min
            f = (pth_req - pth_min) / span if span > 0 else 0.0
            f = min(max(f, 0.0), 1.0)
            pth = pth_req
            pel = pel_min + f * (pel_max - pel_min)
            duty = 1.0
        else:
            duty = pth_req / pth_min
            pth = pth_req
            pel = duty * pel_min

        cop = pth / pel if pel > 0 else 0.0
        return pth, pel, cop, duty
