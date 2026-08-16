clc; clear; close all;
addpath(genpath('../core'));
cfg = config();
rng(cfg.seed);

Xi           = collocation_pts(cfg.N);
[Hhat, Hinv] = gpc_basis(Xi, cfg.N);
fprintf('H_hat condition number: %.2f\n\n', cond(Hhat));

t_vec    = (0:cfg.Nsteps-1)' * cfg.dt;
v_mean   = zeros(cfg.Nsteps, 1);
sig_wind = zeros(cfg.Nsteps, 1);

if isfile(cfg.f_lstm)
    lstm = readtable(cfg.f_lstm,'Delimiter',',','VariableNamingRule','preserve');

    % find 5-hour peak-ramp window from real Kahuku 2019 LSTM output
    win = 5;  % hours
    n   = height(lstm);
    best_i = 1; best_score = 0;
    for i = 1:n-win
        seg   = lstm.sigma_wind(i:i+win-1);
        score = max(seg) - min(seg);   % sigma variation in window
        if score > best_score
            best_score = score;
            best_i     = i;
        end
    end

    seg_mu  = lstm.mu_wind(best_i:best_i+win-1);
    seg_sig = lstm.sigma_wind(best_i:best_i+win-1);
    t_seg   = linspace(0, cfg.T, win)';  

    v_mean   = max(interp1(t_seg, seg_mu,  t_vec, 'pchip'), 0);
    sig_wind = max(interp1(t_seg, seg_sig, t_vec, 'pchip'), 0.05);

    fprintf('Peak-ramp window: hour %d to %d (2019)\n', best_i, best_i+win-1);
    fprintf('sigma range: [%.3f, %.3f] m/s\n', min(sig_wind), max(sig_wind));
    fprintf('v_mean range: [%.3f, %.3f] m/s\n', min(v_mean), max(v_mean));
else
    % fallback
    sc = cfg.scenario;
    for s = 1:size(sc,1)
        idx = t_vec >= sc(s,1) & t_vec < sc(s,2);
        v_mean(idx)   = sc(s,3);
        sig_wind(idx) = sc(s,4);
    end
    fprintf('lstm_output.csv not found -- using synthetic\n');
end

%% ground
x0      = [cfg.d0; cfg.w0];
x_true  = zeros(cfg.Nsteps, 2);
Pm_true = zeros(cfg.Nsteps, 1);

xk = x0;
for k = 1:cfg.Nsteps
    v_k  = max(v_mean(k) + sig_wind(k)*randn(), 0);
    Pw_k = wind_to_power(v_k, cfg);
    Pm_k = cfg.Tm - Pw_k;
    Pm_true(k)   = Pm_k;
    x_true(k,:)  = xk';

    qd = max(min((cfg.alpha_d*sig_wind(k))^2, cfg.Q_max), cfg.Q_min);
    qw = max(min((cfg.alpha_w*sig_wind(k))^2, cfg.Q_max), cfg.Q_min);
    proc_nk = [sqrt(qd)*randn(); sqrt(qw)*randn()];

    p  = mg_params(cfg, Pm_k, [], []);
    xk = swing_eq_discrete(xk, p) + proc_nk;
end
fprintf('Ground truth OK. delta range: [%.4f, %.4f] rad\n',...
    min(x_true(:,1)), max(x_true(:,1)));

%% PMU measurements
z_meas = zeros(cfg.Nsteps, 2);
for k = 1:cfg.Nsteps
    p = mg_params(cfg, Pm_true(k),[],[]);
    z_meas(k,:) = pmu_model(x_true(k,:)',p)' + cfg.noise_R*randn(1,2);
end

%% filters
est_ekf = zeros(cfg.Nsteps,2);
est_pf  = zeros(cfg.Nsteps,2);
est_pm  = zeros(cfg.Nsteps,2);

xe=x0; Pe_=cfg.P0;
xf=x0; Pf=cfg.P0;
xm=x0; Pm_=cfg.P0;

for k = 1:cfg.Nsteps
    p  = mg_params(cfg, Pm_true(k),[],[]);
    zk = z_meas(k,:)';

    [xe,Pe_,~]   = ekf(xe,Pe_,zk,cfg.Q_fixed,cfg.R_meas,p);
    est_ekf(k,:) = xe';

    [xf,Pf,~]   = pckf(xf,Pf,zk,cfg.Q_fixed,cfg.R_meas,p,Xi,Hhat,Hinv);
    est_pf(k,:)  = xf';

    qd = max(min((cfg.alpha_d*sig_wind(k))^2,cfg.Q_max),cfg.Q_min);
    qw = max(min((cfg.alpha_w*sig_wind(k))^2,cfg.Q_max),cfg.Q_min);
    [xm,Pm_,~]  = pckf(xm,Pm_,zk,diag([qd,qw]),cfg.R_meas,p,Xi,Hhat,Hinv);
    est_pm(k,:)  = xm';
end

[rd_e,rw_e] = rmse_metrics(x_true,est_ekf);
[rd_f,rw_f] = rmse_metrics(x_true,est_pf);
[rd_m,rw_m] = rmse_metrics(x_true,est_pm);

fprintf('\n%-22s  %-14s  %-14s\n','Method','RMSE_delta(rad)','RMSE_omega(rad/s)');
fprintf('%s\n',repmat('-',1,52));
fprintf('%-22s  %-14.4e  %-14.4e\n','EKF',        rd_e, rw_e);
fprintf('%-22s  %-14.4e  %-14.4e\n','PCKF-fixed', rd_f, rw_f);
fprintf('%-22s  %-14.4e  %-14.4e\n','PCKF-ML',    rd_m, rw_m);
fprintf('\nPCKF-ML vs PCKF-fixed: delta %.1f%%  omega %.1f%%\n',...
    100*(rd_f-rd_m)/rd_f, 100*(rw_f-rw_m)/rw_f);

if ~isfolder(cfg.res_dir); mkdir(cfg.res_dir); end
save(fullfile(cfg.res_dir,'sim_single.mat'),...
    't_vec','x_true','est_ekf','est_pf','est_pm',...
    'z_meas','sig_wind','v_mean','Pm_true');
fprintf('\nSaved: sim_single.mat\n');
