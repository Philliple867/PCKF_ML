function loss = nll_loss(net, Xbatch, ybatch, lsig_min, lsig_max)
% Gaussian negative log-likelihood -- paper Eq. (NLL).
%
%   L = (1/Ns) * sum[ (1/2)*ln(sigma_t^2) + (v_t - mu_t)^2 / (2*sigma_t^2) ]
%
% Network outputs row 1 = mu_norm, row 2 = log_sigma_norm.
% Clamp log_sigma to [lsig_min, lsig_max] before exp().

pred      = predict(net, Xbatch);
mu        = pred(1,:);
log_sigma = pred(2,:);
log_sigma = max(min(log_sigma, dlarray(lsig_max)), dlarray(lsig_min));
sigma     = exp(log_sigma);

% Gaussian NLL: log(sigma) + (y-mu)^2/(2*sigma^2)
% Equivalent to (1/2)*ln(sigma^2) + (y-mu)^2/(2*sigma^2) -- paper Eq. (nll)
nll  = log(sigma) + (ybatch - mu).^2 ./ (2*sigma.^2);
loss = mean(nll);
end
