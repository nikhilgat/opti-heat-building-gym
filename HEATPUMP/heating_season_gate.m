function heating_season_on = heating_season_gate(t_sim, T_air_C)
%HEATING_SEASON_GATE  Simulink-callable wrapper for is_heating_active.
%
% Stateflow charts (MPC Prediction Model, HP Physics) cannot manipulate
% datetime/duration values directly, even inside a coder.extrinsic-marked
% function body -- extrinsic only exempts the CALL itself from codegen
% restrictions, not the caller's own local arithmetic. This wrapper
% isolates all datetime arithmetic behind a single extrinsic call so the
% calling chart only ever touches T_SIM and T_AIR_C (double) in and a
% plain logical out.
%
% T_AIR_C feeds the hybrid gate's cold-snap override (see
% is_heating_active.m) -- both calling charts already carry T_air_C as an
% input for their own physics, so this needed no new Simulink signal, only
% this wrapper's signature growing by one argument.
%
% startDate is read from the base workspace once and cached: it is set by
% init_adaptive_mpc.m before the model loads, the same contract the rest
% of the model already relies on for plant_d0/mpcobj.

persistent startDate0
if isempty(startDate0)
    startDate0 = evalin('base', 'startDate');
end
heating_season_on = is_heating_active(startDate0 + seconds(t_sim), T_air_C);
end
