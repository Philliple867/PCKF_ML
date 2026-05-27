function [Hhat, Hinv] = gpc_basis(Xi, N)
% gPC basis matrix Hhat -- paper Eq. (15) in Xu 2019.
%
% After dimension reduction (paper Eq. pce_reduced), retained basis:
%   phi_0        = 1                           (constant)
%   phi_1..N     = xi_j                        (1st-order Hermite)
%   phi_N+1..2N  = (xi_j^2 - 1)/sqrt(2)       (2nd-order Hermite)
%
% Hhat is (2N+1) x (2N+1). Row i = phi evaluated at CP xi_i.
% Column ordering follows collocation_pts.m:
%   cols 1..N positive, cols N+1..2N negative, col 2N+1 center.

ncp  = 2*N + 1;
Hhat = zeros(ncp, ncp);

for i = 1:ncp
    xi = Xi(:, i);
    Hhat(i, 1) = 1.0;
    for j = 1:N
        Hhat(i, 1+j)   = xi(j);
        Hhat(i, N+1+j) = (xi(j)^2 - 1) / sqrt(2);
    end
end

% verify invertibility -- always holds for sqrt(3) collocation points
if rcond(Hhat) < 1e-12
    warning('gpc_basis: Hhat nearly singular (rcond=%.2e)', rcond(Hhat));
end
Hinv = Hhat \ eye(ncp);
end
