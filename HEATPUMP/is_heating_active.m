function tf = is_heating_active(dt, T_air_C)
%IS_HEATING_ACTIVE  Hybrid Heizperiode gate: calendar season OR real Heizgrenze.
%
% tf = is_heating_season(dt) | T_air_C < COLD_SNAP_C
%
% The pure calendar gate (is_heating_season.m) fixed the in-season
% capacity-cliff bug (OPTIHEAT_PROJECT_STATUS.md Sec.5 item 1) but, on its
% own, traded it for a worse one: the 2026-08-20 12-month smoke-test sweep
% (Sec.6a) showed June and July 2025 -- correctly, deliberately off per the
% calendar -- passively coasting to 17.5 C / 16.8 C for 12.5 h / 26.75 h
% with ZERO heat available, because a pure calendar has no way to react to
% a real cold snap landing inside the "official" off-season. This function
% restores that reaction as a floor UNDERNEATH the calendar gate, not a
% replacement for it.
%
% COLD_SNAP_C = 15, deliberately the SAME value as HEATING_LIMIT_C in
% heating_curve.m (the real IDM ALM 4-12 Heizgrenze, from the as-built
% hydraulic scheme). This was checked against the weather data, not
% guessed:
%   - It is the only threshold that actually reaches into the documented
%     June 15-17 (window min 13.1 C) and July 15-17 (window min 13.9 C)
%     smoke-test crash windows at all -- 12/13/14 C all sit BELOW both
%     windows' minima and would never fire.
%   - Lower "genuine cold snap" thresholds (10-13 C) looked appealing to
%     avoid firing on an ordinary mild night, but they don't fire during
%     the actual documented failures either, so they would not fix
%     anything.
%   - It is not a new number: it is already the site's real, documented
%     Heizgrenze, so outside the calendar season the machine now behaves
%     exactly as the real hardware does year-round (the as-built scheme
%     has no calendar override at all -- that was always a modelling
%     device to give the frozen-horizon Adaptive MPC a foreseeable
%     capacity signal, see is_heating_season.m).
%
% WHY THIS CANNOT REINTRODUCE THE ORIGINAL IN-SEASON BUG: the OR is
% short-circuited by the calendar term first. Within season (Oct-Apr),
% is_heating_season(dt) is unconditionally true, so heating_season_on is
% always true regardless of T_air_C -- the COLD_SNAP_C comparison never
% has any effect Oct-Apr and therefore cannot recreate the same-day
% capacity cliff that motivated the calendar gate in the first place. It
% only ever changes behaviour May-Sep.
%
% Residual known limitation, not yet resolved: within May-Sep, a stretch
% of nights hovering right around COLD_SNAP_C can still make the frozen-
% horizon Adaptive MPC's Qmax belief flip step to step, same mechanism as
% the original bug, just now confined to the off-season. Untested whether
% this matters in practice -- re-running run_smoke_tests.m after this
% change is how to check. Even imperfect, some heat during a real cold
% night is a strict improvement over the previous zero.
%
% Vectorized over DT and T_air_C exactly as far as is_heating_season is
% (datetime array or scalar); T_air_C must be the same size or scalar.

COLD_SNAP_C = 15;   % == HEATING_LIMIT_C in heating_curve.m -- see header.

tf = is_heating_season(dt) | T_air_C < COLD_SNAP_C;
end
