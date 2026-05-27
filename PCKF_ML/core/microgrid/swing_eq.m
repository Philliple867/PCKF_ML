function dxdt = swing_eq(~, x, params)
% Swing equations (continuous) -- paper Eqs. (delta)(omega).
% params.Pm_eff = T_m - P_w(v_t)

Pe = (params.E/params.Xd) * params.V * sin(x(1) - params.theta);
dxdt = [x(2) - params.ws;
        (params.ws/(2*params.H)) * (params.Pm_eff - Pe - params.D*(x(2)-params.ws))];
end
