function [rmse_d, rmse_w] = rmse_metrics(x_true, x_est)
% RMSE for delta (col 1) and omega (col 2).
err    = x_true - x_est;
rmse_d = sqrt(mean(err(:,1).^2));
rmse_w = sqrt(mean(err(:,2).^2));
end
