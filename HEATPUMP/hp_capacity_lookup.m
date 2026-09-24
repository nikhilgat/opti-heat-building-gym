function Q_W = hp_capacity_lookup(T_air_C, T_supply_C, mode)
%HP_CAPACITY_LOOKUP Scalar thermal capacity [W] from IDM ALM 4-12 interpolants.
% mode: 'max' or 'min' modulation.
%
% Envelope: the datasheet grids are NaN-filled inside
% idm_alm412_interpolants.m by holding the highest achievable flow
% temperature, so the interpolants no longer return NaN. The old
% RESISTIVE_FALLBACK_W = 30162 branch (inherited from
% BasicHeater_HouseHeatingSystemExample.m) has been removed: feeding a
% fictitious 30 kW capacity into getAdaptivePlant would have inflated the
% MPC's B matrix by ~3x and made the controller believe it had far more
% authority than the machine actually has. Out-of-envelope points now warn
% once and use the clamped datasheet value.

persistent hp warnedEnvelope
if isempty(hp)
    hp = idm_alm412_interpolants();
end
if isempty(warnedEnvelope)
    warnedEnvelope = false;
end

if ~warnedEnvelope && ~hp.inEnvelope(T_air_C, T_supply_C)
    warning('hp_capacity_lookup:outsideEnvelope', ...
        ['Operating point T_air = %.1f C, T_supply = %.1f C is outside the ' ...
         'measured IDM ALM 4-12 envelope (max supply here is %.1f C). ' ...
         'Using clamped datasheet values. This warning is issued once.'], ...
        T_air_C, T_supply_C, hp.maxValidSupply(T_air_C));
    warnedEnvelope = true;
end

switch mode
    case 'max'
        Q_kW = hp.getPthMax(T_air_C, T_supply_C);
    case 'min'
        Q_kW = hp.getPthMin(T_air_C, T_supply_C);
    otherwise
        error('hp_capacity_lookup: mode must be ''max'' or ''min'', got %s', mode);
end

if isnan(Q_kW)
    error('hp_capacity_lookup:nanCapacity', ...
        ['NaN capacity at T_air = %.1f C, T_supply = %.1f C, mode ''%s''. ' ...
         'The interpolant grids are NaN-filled, so this indicates a bug in ' ...
         'idm_alm412_interpolants.m rather than a missing data point.'], ...
        T_air_C, T_supply_C, mode);
end

Q_W = Q_kW * 1000;
end
