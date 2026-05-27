function [x_new, P_new, innov] = ekf(x, P, z, Q, R, params)
% EKF baseline -- linearized at current state.
% Jacobian of f derived from swing equations (paper Eqs. delta, omega).

dt  = params.dt;
H   = params.H;
D   = params.D;
Xd  = params.Xd;
E   = params.E;
V   = params.V;
th  = params.theta;
ws  = params.ws;

%% prediction (nonlinear propagation via RK4)
x_pr = swing_eq_discrete(x, params);

% Jacobian F = df/dx  (linearized at x, discrete)
% df1/ddelta=0, df1/domega=1  (times dt, then +I for discretization)
% df2/ddelta = -dt*(ws/2H)*(E/Xd)*V*cos(d-th)
% df2/domega = 1 - dt*(ws/2H)*D
dPe_dd = (E/Xd)*V*cos(x(1) - th);
F = [1,   dt;
    -dt*(ws/(2*H))*dPe_dd,   1 - dt*(ws/(2*H))*D];

P_pr = F*P*F' + Q;

%% observation
z_pr = pmu_model(x_pr, params);

% Jacobian Hobs = dh/dx at x_pr
dPe2 =  (E/Xd)*V*cos(x_pr(1) - th);
dQe2 = -(E/Xd)*V*sin(x_pr(1) - th);
Hobs = [dPe2, 0;
        dQe2, 0];

%% correction
S     = Hobs*P_pr*Hobs' + R;
K     = P_pr*Hobs' / S;
innov = z - z_pr;

x_new = x_pr + K*innov;
IKH   = eye(2) - K*Hobs;
P_new = IKH*P_pr*IKH' + K*R*K';
P_new = 0.5*(P_new + P_new');
end
