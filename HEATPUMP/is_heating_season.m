function tf = is_heating_season(dt)
%IS_HEATING_SEASON  German residential heating period (Heizperiode).
%
% Conventional German rental-law heating season, the near-universal
% convention written into standard lease contracts ("Mustermietvertrag")
% and cited by the Deutscher Mieterbund: 1 October through 30 April
% inclusive (7 months). There is no single nationwide statute fixing
% these exact calendar dates -- the underlying legal duty is a minimum
% indoor temperature, which in principle can require heating outside this
% window too during a cold snap -- but Oct 1 - Apr 30 is what building
% operators actually run to in practice, and it is a calendar fact known
% in advance rather than something that has to be read off the current
% outdoor temperature.
%
% This replaces the old instantaneous per-sample 15 C Heizgrenze gate as
% the heat pump's on/off criterion (see OPTIHEAT_PROJECT_STATUS.md Sec.5
% item 1): that gate re-evaluated every 15 min from T_air_C alone, so a
% single mild afternoon in an otherwise cold month could switch the
% compressor hard off for hours -- a cliff the frozen-horizon Adaptive MPC
% has no way to foresee. A calendar season gate removes that cliff at the
% source: within season the HP is available exactly like the real plant
% (load and price still decide how hard it runs), and outside it the HP is
% off entirely, exactly like a real heating plant shut down for summer.
%
% Vectorized over DT (datetime array or scalar).
tf = dt.Month >= 10 | dt.Month <= 4;
end
