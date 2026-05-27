clc; clear; close all;
addpath(genpath('../core'));
cfg = config();

if ~isfile(fullfile(cfg.res_dir,'sim_single.mat'))
    error('Run run_simulation.m first.');
end
load(fullfile(cfg.res_dir,'sim_single.mat'),'v_mean','sig_wind','Pm_true');

N_MC   = cfg.N_MC;
Nsteps = cfg.Nsteps;
x0     = [cfg.d0; cfg.w0];

Xi           = collocation_pts(cfg.N);
[Hhat, Hinv] = gpc_basis(Xi, cfg.N);

rmse_e  = zeros(N_MC,2);
rmse_pf = zeros(N_MC,2);
rmse_pm = zeros(N_MC,2);
cpu_e   = zeros(N_MC,1);
cpu_pf  = zeros(N_MC,1);
cpu_pm  = zeros(N_MC,1);

fprintf('Running %d MC trials...\n', N_MC);
marks = round(linspace(1,N_MC,6));

for mc = 1:N_MC
    rng(cfg.mc_seed0 + mc);

    
    x_true = zeros(Nsteps,2); Pm_mc = zeros(Nsteps,1);
    xk = x0;
    for k = 1:Nsteps
        v_k      = max(v_mean(k) + sig_wind(k)*randn(), 0);
        Pw_k     = wind_to_power(v_k, cfg);
        Pm_k     = cfg.Tm - Pw_k;
        Pm_mc(k) = Pm_k;
        x_true(k,:) = xk';

        qd = max(min((cfg.alpha_d*sig_wind(k))^2, cfg.Q_max), cfg.Q_min);
        qw = max(min((cfg.alpha_w*sig_wind(k))^2, cfg.Q_max), cfg.Q_min);
        proc_nk = 0.5 * [sqrt(qd)*randn(); sqrt(qw)*randn()];

        p  = mg_params(cfg, Pm_k,[],[]);
        xk = swing_eq_discrete(xk,p) + proc_nk;
    end

    zmeas = zeros(Nsteps,2);
    for k = 1:Nsteps
        p = mg_params(cfg,Pm_mc(k),[],[]);
        zmeas(k,:) = pmu_model(x_true(k,:)',p)' + cfg.noise_R*randn(1,2);
    end

    % EKF
    xe=x0; Pe_=cfg.P0; ee=zeros(Nsteps,2);
    t0=tic;
    for k=1:Nsteps
        p=mg_params(cfg,Pm_mc(k),[],[]);
        [xe,Pe_,~]=ekf(xe,Pe_,zmeas(k,:)',cfg.Q_fixed,cfg.R_meas,p);
        ee(k,:)=xe';
    end
    cpu_e(mc)=toc(t0);
    [rmse_e(mc,1),rmse_e(mc,2)]=rmse_metrics(x_true,ee);

    % PCKF-fixed
    xf=x0; Pf=cfg.P0; ef=zeros(Nsteps,2);
    t0=tic;
    for k=1:Nsteps
        p=mg_params(cfg,Pm_mc(k),[],[]);
        [xf,Pf,~]=pckf(xf,Pf,zmeas(k,:)',cfg.Q_fixed,cfg.R_meas,p,Xi,Hhat,Hinv);
        ef(k,:)=xf';
    end
    cpu_pf(mc)=toc(t0);
    [rmse_pf(mc,1),rmse_pf(mc,2)]=rmse_metrics(x_true,ef);

    % PCKF-ML
    xm=x0; Pm_=cfg.P0; em=zeros(Nsteps,2);
    t0=tic;
    for k=1:Nsteps
        qd=max(min((cfg.alpha_d*sig_wind(k))^2,cfg.Q_max),cfg.Q_min);
        qw=max(min((cfg.alpha_w*sig_wind(k))^2,cfg.Q_max),cfg.Q_min);
        p=mg_params(cfg,Pm_mc(k),[],[]);
        [xm,Pm_,~]=pckf(xm,Pm_,zmeas(k,:)',diag([qd,qw]),cfg.R_meas,p,Xi,Hhat,Hinv);
        em(k,:)=xm';
    end
    cpu_pm(mc)=toc(t0);
    [rmse_pm(mc,1),rmse_pm(mc,2)]=rmse_metrics(x_true,em);

    if ismember(mc,marks)
        fprintf('  run %3d/%d  delta: EKF=%.3e  PCKFf=%.3e  PCKFml=%.3e\n',...
            mc,N_MC,rmse_e(mc,1),rmse_pf(mc,1),rmse_pm(mc,1));
    end
end

fprintf('\n=== Monte Carlo (N=%d) ===\n', N_MC);
fprintf('%-22s  %-22s  %-22s  %s\n','Method','RMSE_delta','RMSE_omega','CPU ms');
fprintf('%s\n',repmat('-',1,78));
rows={rmse_e,rmse_pf,rmse_pm}; cpus={cpu_e,cpu_pf,cpu_pm};
names={'EKF','PCKF-fixed','PCKF-ML (proposed)'};
for i=1:3
    r=rows{i}; c=cpus{i};
    fprintf('%-22s  %.3e +/- %-10.3e  %.3e +/- %-10.3e  %.2f\n',...
        names{i},mean(r(:,1)),std(r(:,1)),mean(r(:,2)),std(r(:,2)),mean(c)*1e3);
end

imp_d = 100*(mean(rmse_pf(:,1))-mean(rmse_pm(:,1)))/mean(rmse_pf(:,1));
imp_w = 100*(mean(rmse_pf(:,2))-mean(rmse_pm(:,2)))/mean(rmse_pf(:,2));
fprintf('\nPCKF-ML vs PCKF-fixed: delta=%.1f%%  omega=%.1f%%\n',imp_d,imp_w);

save(fullfile(cfg.res_dir,'mc_results.mat'),...
    'rmse_e','rmse_pf','rmse_pm','cpu_e','cpu_pf','cpu_pm','names');
fprintf('\nSaved: mc_results.mat\n');