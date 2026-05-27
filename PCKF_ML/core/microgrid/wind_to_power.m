function Pw = wind_to_power(v, cfg)
% Cubic wind turbine power curve -- paper Section IV-A, Eq. (power_curve).
%
%   Pw = 0                                  v < v_ci  or  v > v_co
%   Pw = P_r * (v^3 - v_ci^3)/(v_r^3 - v_ci^3)    v_ci <= v < v_r
%   Pw = P_r                                v_r <= v <= v_co
%
% v can be a scalar or array.
% Returns Pw in per unit [pu], same base as cfg.P_r.

v_ci = cfg.v_ci;   % 3.5 m/s
v_r  = cfg.v_r;    % 12.0 m/s
v_co = cfg.v_co;   % 25.0 m/s
P_r  = cfg.P_r;    % 0.30 pu

Pw = zeros(size(v));

idx_cubic = (v >= v_ci) & (v < v_r);
idx_rated = (v >= v_r)  & (v <= v_co);

Pw(idx_cubic) = P_r * (v(idx_cubic).^3 - v_ci^3) / (v_r^3 - v_ci^3);
Pw(idx_rated) = P_r;

% below cut-in and above cut-out: Pw = 0 (already initialized)
end
