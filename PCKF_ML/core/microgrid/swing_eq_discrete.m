function x_next = swing_eq_discrete(x, params)
% RK4 one-step integration of swing equations.
% params.Pm_eff = T_m - P_w(v_t) is the effective mechanical power
% at the current timestep -- matches paper Eq. (omega).

dt = params.dt;
k1 = rhs(x,             params);
k2 = rhs(x + .5*dt*k1,  params);
k3 = rhs(x + .5*dt*k2,  params);
k4 = rhs(x +    dt*k3,  params);
x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function dxdt = rhs(x, p)
% Paper Eqs. (delta) and (omega):
%   ddelta/dt = omega - omega_s
%   domega/dt = (omega_s/2H) * [P_m_eff - (E/Xd)*V*sin(delta-theta) - D*(omega-ws)]
Pe = (p.E/p.Xd) * p.V * sin(x(1) - p.theta);
dxdt = [x(2) - p.ws;
        (p.ws/(2*p.H)) * (p.Pm_eff - Pe - p.D*(x(2) - p.ws))];
end
