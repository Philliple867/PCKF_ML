function [x_new, P_new, innov] = pckf(x, P, z, Q, R, params, Xi, Hhat, Hinv)
% One step of PCKF -- Xu et al. (2019), Eqs. (14)-(30).
%
% Q can be:
%   Q_fixed (constant)  -- PCKF-fixed baseline
%   Q_t     (adaptive)  -- PCKF-ML proposed method, from Eq. (Qt)
%
% params.Pm_eff must hold the current P_m_eff = T_m - P_w(v_t).

N   = length(x);
ncp = 2*N + 1;

%% --- PREDICTION ---

sw = sqrt(max(diag(Q), 0));   % process noise std per state, [N x 1]

% Propagate each CP through dynamics -- Eqs. (16)(17)
Xhat = zeros(ncp, N);
for i = 1:ncp
    x_aug    = x + sw .* Xi(:, i);
    Xhat(i,:) = swing_eq_discrete(x_aug, params)';
end

% PC coefficient matrix -- Eq. (19): Ahat = Hinv * Xhat
Ahat = Hinv * Xhat;   % (2N+1) x N

% Predicted state mean -- Eq. (20): x_pred = A1 (first row)
x_pr = Ahat(1, :)';

% A2: rows 2..2N+1 carry second-moment info
A2   = Ahat(2:end, :);   % (2N) x N

% Predicted state covariance -- Eq. (21)
P_pr = A2' * A2 + Q;

%% --- OBSERVATION PREDICTION ---

Zhat = zeros(ncp, 2);
for i = 1:ncp
    Zhat(i,:) = pmu_model(Xhat(i,:)', params)';
end

W    = Hinv * Zhat;
z_pr = W(1,:)';     % predicted measurement mean -- Eq. (25)
W2   = W(2:end,:);  % (2N) x 2

%% --- CORRECTION ---  Eqs. (26)-(30)

Pzz  = W2' * W2 + R;       % [2 x 2]
Pxz  = A2' * W2;            % [N x 2]
K    = Pxz / Pzz;           % Kalman gain
innov = z - z_pr;

x_new = x_pr + K * innov;
P_new = P_pr - K * Pzz * K';

% symmetrize to prevent numerical drift
P_new = 0.5*(P_new + P_new');

% nudge if P loses positive-definiteness under large disturbances
[~, flag] = chol(P_new);
if flag ~= 0
    P_new = P_new + 1e-8 * eye(N);
end
end
