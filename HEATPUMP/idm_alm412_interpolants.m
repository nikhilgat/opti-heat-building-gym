function hp = idm_alm412_interpolants()

d = idm_alm412_datasheet();

[T_air_sorted, sort_idx] = sort(d.T_air, 'ascend');

P_th_max_sorted = d.P_th_max_kW(:, sort_idx);
P_th_min_sorted = d.P_th_min_kW(:, sort_idx);
P_el_max_sorted = d.P_el_max_kW(:, sort_idx);
P_el_min_sorted = d.P_el_min_kW(:, sort_idx);

hp.data = d;

hp.Pth_max = griddedInterpolant({d.T_supply, T_air_sorted}, P_th_max_sorted, 'linear', 'nearest');
hp.Pth_min = griddedInterpolant({d.T_supply, T_air_sorted}, P_th_min_sorted, 'linear', 'nearest');
hp.Pel_max = griddedInterpolant({d.T_supply, T_air_sorted}, P_el_max_sorted, 'linear', 'nearest');
hp.Pel_min = griddedInterpolant({d.T_supply, T_air_sorted}, P_el_min_sorted, 'linear', 'nearest');

hp.Tair_min = min(d.T_air);
hp.Tair_max = max(d.T_air);
hp.Tsupply_min = min(d.T_supply);
hp.Tsupply_max = max(d.T_supply);

hp.getPthMax = @(Tair,Tsupply) hp.Pth_max(Tsupply,Tair);
hp.getPthMin = @(Tair,Tsupply) hp.Pth_min(Tsupply,Tair);
hp.getPelMax = @(Tair,Tsupply) hp.Pel_max(Tsupply,Tair);
hp.getPelMin = @(Tair,Tsupply) hp.Pel_min(Tsupply,Tair);

% --- Operating envelope -------------------------------------------------
% Highest supply temperature with a measured (non-NaN) Pth_max at the
% nearest measured air-temperature column. Used to warn once when an
% operating point sits outside the datasheet grid.
hp.maxValidSupply = @(Tair) localMaxValidSupply(d, Tair);
hp.inEnvelope = @(Tair,Tsupply) ...
    (Tair >= min(d.T_air)) && (Tair <= max(d.T_air)) && ...
    (Tsupply >= min(d.T_supply)) && (Tsupply <= localMaxValidSupply(d, Tair));

hp.getPerformance = @(Tair,Tsupply,u) interpolatePerformance(hp,Tair,Tsupply,u);

end

function smax = localMaxValidSupply(d, Tair)
[~, col] = min(abs(d.T_air - Tair));      % nearest air-temp column
valid = ~isnan(d.P_th_max_kW(:, col));    % rows with measured data
if any(valid)
    smax = max(d.T_supply(valid));
else
    smax = min(d.T_supply);
end
end

function perf = interpolatePerformance(hp,Tair,Tsupply,u)
%INTERPOLATEPERFORMANCE Duty-cycle-averaged IDM ALM 4-12 performance.
%
% u is the requested fraction of MAXIMUM thermal capacity, 0..1.
% The delivered heat is therefore, by construction and in every branch:
%
%       Pth = u * Pth_max(Tair, Tsupply)
%
% This is the whole point of the rewrite. The MPC's internal input matrix
% is B(2,1) = Qmax/C_tank, i.e. it assumes heat = u * Qmax. The previous
% law was
%
%       Pth = Pth_min + u*(Pth_max - Pth_min)
%
% which delivered ~4 kW at u = 0 -- the machine had no OFF state at all --
% and had a slope of (Pth_max - Pth_min) instead of Pth_max. That is a
% ~4 kW offset error plus a ~30 % gain error against the controller's own
% model, and it is the dominant cause of the sustained limit cycling: the
% controller drove u to 0, still received 4 kW, pushed the room into its
% upper bound, and then swung back to full power.
%
% MINIMUM MODULATION. The real machine cannot deliver less than
% Pth_min ~ 4.0 kW while running. Below that it switches on and off. Over
% one 15-minute control interval that is exactly a duty cycle:
%
%       duty = Pth_req / Pth_min      (0 < duty < 1)
%
% and the interval-averaged heat is duty * Pth_min = Pth_req, while the
% interval-averaged electrical draw is duty * Pel_min. Averaging the
% compressor's on/off behaviour over the control interval keeps the plant
% linear and exactly matched to the controller's model, at the cost of not
% resolving sub-interval temperature ripple (negligible against a 500 L
% buffer and a 57 h building time constant).
%
% Returned fields:
%   Pth   [kW]  interval-average thermal output
%   Pel   [kW]  interval-average electrical input
%   COP   [-]   Pth/Pel, 0 when off
%   duty  [-]   fraction of the interval the compressor runs
%               0        = off for the whole interval
%               0<duty<1 = one on/off cycle inside this interval
%               1        = continuous modulated running

u = max(0, min(1, u));

Pth_max = hp.getPthMax(Tair, Tsupply);
Pth_min = hp.getPthMin(Tair, Tsupply);
Pel_max = hp.getPelMax(Tair, Tsupply);
Pel_min = hp.getPelMin(Tair, Tsupply);

% Guard against a degenerate grid point where the clamped datasheet gives
% a minimum at or above the maximum. Treat it as a fixed-output machine.
if ~(Pth_max > Pth_min)
    Pth_max = Pth_min;
end

Pth_req = u * Pth_max;

if Pth_req <= 0
    % --- Compressor off for the whole interval ---
    Pth  = 0;
    Pel  = 0;
    duty = 0;

elseif Pth_req >= Pth_min
    % --- Continuous inverter modulation between the min and max points ---
    span = Pth_max - Pth_min;
    if span > 0
        f = (Pth_req - Pth_min) / span;     % 0 at Pth_min, 1 at Pth_max
    else
        f = 0;
    end
    f    = max(0, min(1, f));
    Pth  = Pth_req;
    Pel  = Pel_min + f * (Pel_max - Pel_min);
    duty = 1;

else
    % --- Demand below the modulation floor: on/off inside the interval ---
    duty = Pth_req / Pth_min;               % strictly between 0 and 1
    Pth  = Pth_req;                         % == duty * Pth_min
    Pel  = duty * Pel_min;
end

if Pel > 0
    COP = Pth / Pel;
else
    COP = 0;
end

perf.Pth  = Pth;
perf.Pel  = Pel;
perf.COP  = COP;
perf.duty = duty;
end
