function Xi = collocation_pts(N)
% Collocation point matrix for PCKF -- paper Eq. (cp).
%
% Column layout (matches paper exactly):
%   cols 1..N      : xi_i  = +sqrt(3)*e_i   (positive perturbation)
%   cols N+1..2N   : xi_i  = -sqrt(3)*e_i   (negative perturbation)
%   col  2N+1      : xi    =  0              (center point)
%
% Roots come from He_3(xi) = (xi^3 - 3*xi)/sqrt(6) = 0
%   => xi in {-sqrt(3), 0, +sqrt(3)}

Xi = zeros(N, 2*N+1);
s3 = sqrt(3);

for i = 1:N
    Xi(i, i)     = +s3;    % positive block: cols 1..N
    Xi(i, N+i)   = -s3;    % negative block: cols N+1..2N
end
% col 2N+1 stays zero (center) -- already initialized

end
