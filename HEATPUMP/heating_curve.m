function [T_supply_C, heating_on] = heating_curve(T_air_C)
%HEATING_CURVE Weather-compensated supply temp for the radiator system.
%
% EXPONENT-LAW curve (EN 442 radiator characteristic), replacing the
% previous linear interpolation -- see OPTIHEAT_PROJECT_STATUS.md Sec.5
% item 4. Radiator heat output scales with (T_supply - T_room)^n, n~1.3
% for panel radiators, NOT linearly with T_supply. Since building heat
% LOSS scales linearly with (T_room - T_air) (UA*(T_room-T_air)), setting
% load = radiator output and solving for T_supply as a function of T_air
% gives a curve that is concave, not straight -- it sits noticeably below
% the old linear line through most of the mid-load range, where most
% heating hours actually sit.
%
%   Q_frac(T_air) = (T_room_design - T_air) / (T_room_design - T_air_design)
%   T_supply_C    = T_room_design + (T_supply_design - T_room_design) * Q_frac^(1/n)
%
% Design point: 55 C supply at -10 C outdoor, room 20 C (radiator design
% 55/45/20, as-built hydraulic scheme). T_ROOM_DESIGN below is this
% hydraulic DESIGN figure, not the MPC's current control setpoint
% (T_room_ref = 20.2 C in init_adaptive_mpc.m) -- two different ~20 C
% numbers for two different reasons, don't merge them.
%
% Floored at MIN_USEFUL_SUPPLY_C = 35 C (unchanged from the old linear
% curve's upper endpoint) rather than letting the exponent law run down
% towards T_room_design as T_air approaches it: real radiator systems stop
% being useful below some minimum flow temperature, and this is that
% floor, not a point the exponent-law math is required to land on exactly
% at any particular T_air. With n=1.3 the floor is reached around
% T_air ~ 10 C, a few degrees before the separate 15 C Heizgrenze cutoff
% (HEATING_LIMIT_C below / is_heating_active.m) -- the curve is flat, not
% still descending, for the last few degrees before cutoff, which matches
% how real weather-compensation controllers commonly behave.
%
% CHECKED AGAINST THE REFERENCE NUMBER IN THE PROJECT DOCS, WITH A NOTED
% GAP: Sec.5 item 4 states the correct half-load supply temp is 40.1 C
% (vs. 44.6 C for the old linear curve). This implementation gives 40.5 C
% at the same Q_frac = 0.5 point (T_air = 5 C) -- close, same qualitative
% conclusion (several degrees below the linear curve through the mid-load
% range), but not bit-exact. The ~0.4 C gap is most likely because the
% reference figure used a mean-supply/return driving delta-T with the
% supply/return SPREAD also scaled by Q_frac, instead of this
% implementation's simpler supply-temperature-only delta-T (chosen because
% nothing else in this codebase models a return temperature at all -- see
% Sec.5 item 3, still open, for the separate T_supply<-Ttank question).
% Worth a quick sanity check against whatever produced the 40.1 C figure
% originally; not blocking, since the qualitative fix (curve sits several
% degrees below the old linear one through the mid-load range) holds
% under either convention.
%
% Second output HEATING_ON is the raw instantaneous Heizgrenze check. It is
% NOT the authoritative on/off gate anywhere in the model any more -- that
% role now belongs to is_heating_active.m's hybrid calendar+cold-snap gate
% (OPTIHEAT_PROJECT_STATUS.md Sec.5 item 1 / Sec.8 item 1). Kept as an
% output for callers that still want the raw instantaneous comparison;
% T_supply_C is still returned at its clamped minimum when this reads
% false, so downstream COP/capacity interpolants never see an invalid
% temperature.
%
% 15 C is taken from the as-built hydraulic scheme
% (LAN_Vir.6_HZ_SC_28.03.2025): "Heizgrenztemperatur und Bivalenzpunkt:
% Die Heizgrenztemperatur ist dabei immer 15 C." is_heating_active.m's
% COLD_SNAP_C deliberately reuses this same value -- see that file's
% header for why.

HEATING_LIMIT_C = 15;

T_room_design       = 20;    % radiator design room temp (55/45/20), NOT the MPC's T_room_ref
T_air_design         = -10;  % design outdoor temp, matches the old T1_air
T_supply_design      = 55;   % design supply temp, matches the old T1_sup
RADIATOR_EXPONENT    = 1.3;  % EN 442 panel-radiator exponent
MIN_USEFUL_SUPPLY_C  = 35;   % floor -- matches the old curve's upper (T2) endpoint

if T_air_C <= T_air_design
    T_supply_C = T_supply_design;
else
    Q_frac = (T_room_design - T_air_C) / (T_room_design - T_air_design);
    Q_frac = max(Q_frac, 0);   % T_air at/above room temp -> zero load, floor takes over
    T_supply_C = T_room_design + (T_supply_design - T_room_design) * Q_frac^(1/RADIATOR_EXPONENT);
    T_supply_C = max(T_supply_C, MIN_USEFUL_SUPPLY_C);
end

heating_on = T_air_C < HEATING_LIMIT_C;
end
