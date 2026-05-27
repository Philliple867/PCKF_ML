% sensitivity_noise.m
% Robustness test under varying PMU noise levels (30 runs each).

clc; clear; close all;
addpath(genpath('../core'));
cfg = config();

if ~isfile(fullfile(cfg.res_dir,'sim_single.mat'))
    error('Run run_simulation.m first.');
end
load(fullfile(cfg.res_dir,'sim_single.mat'),'v_mean','sig_wind');

x0=cfg.d0; x0=[x0;cfg.w0]; Nsteps=cfg.Nsteps; N_MC_s=30;
Xi=[]; [Hhat,Hinv]=gpc_basis(collocation_pts(cfg.N),cfg.N);
Xi=collocation_pts(cfg.N);

mults=cfg.noise_mults; n_lev=length(mults);
rmse_all=zeros(n_lev,3,2);

fprintf('Sensitivity: %d noise levels x %d runs\n\n',n_lev,N_MC_s);

for nl=1:n_lev
    nstd=cfg.noise_R*mults(nl); Rnl=diag([nstd^2,nstd^2]);
    tmp_e=zeros(N_MC_s,2); tmp_pf=zeros(N_MC_s,2); tmp_pm=zeros(N_MC_s,2);

    for mc=1:N_MC_s
        rng(500+nl*100+mc);
        proc_n=0.5*randn(Nsteps,2).*repmat(sqrt(diag(cfg.Q_fixed))',Nsteps,1);
        x_true=zeros(Nsteps,2); Pm_mc=zeros(Nsteps,1); xk=x0;
        for k=1:Nsteps
            v_k=max(v_mean(k)+sig_wind(k)*randn(),0);
            Pm_k=cfg.Tm-wind_to_power(v_k,cfg);
            Pm_mc(k)=Pm_k; x_true(k,:)=xk';
            p=mg_params(cfg,Pm_k,[],[]);
            xk=swing_eq_discrete(xk,p)+proc_n(k,:)';
        end
        zm=zeros(Nsteps,2);
        for k=1:Nsteps
            p=mg_params(cfg,Pm_mc(k),[],[]);
            zm(k,:)=pmu_model(x_true(k,:)',p)'+nstd*randn(1,2);
        end

        xe=x0;Pe_=cfg.P0;ee=zeros(Nsteps,2);
        xf=x0;Pf=cfg.P0;ef=zeros(Nsteps,2);
        xm=x0;Pm_=cfg.P0;em=zeros(Nsteps,2);
        for k=1:Nsteps
            p=mg_params(cfg,Pm_mc(k),[],[]);
            [xe,Pe_,~]=ekf(xe,Pe_,zm(k,:)',cfg.Q_fixed,Rnl,p); ee(k,:)=xe';
            [xf,Pf,~]=pckf(xf,Pf,zm(k,:)',cfg.Q_fixed,Rnl,p,Xi,Hhat,Hinv); ef(k,:)=xf';
            qd=max(min((cfg.alpha_d*sig_wind(k))^2,cfg.Q_max),cfg.Q_min);
            qw=max(min((cfg.alpha_w*sig_wind(k))^2,cfg.Q_max),cfg.Q_min);
            [xm,Pm_,~]=pckf(xm,Pm_,zm(k,:)',diag([qd,qw]),Rnl,p,Xi,Hhat,Hinv); em(k,:)=xm';
        end
        [tmp_e(mc,1),tmp_e(mc,2)]=rmse_metrics(x_true,ee);
        [tmp_pf(mc,1),tmp_pf(mc,2)]=rmse_metrics(x_true,ef);
        [tmp_pm(mc,1),tmp_pm(mc,2)]=rmse_metrics(x_true,em);
    end
    rmse_all(nl,1,:)=mean(tmp_e,1);
    rmse_all(nl,2,:)=mean(tmp_pf,1);
    rmse_all(nl,3,:)=mean(tmp_pm,1);
    fprintf('x%.1f (std=%.4f): EKF=%.3e  PCKFf=%.3e  PCKFml=%.3e\n',...
        mults(nl),nstd,rmse_all(nl,1,1),rmse_all(nl,2,1),rmse_all(nl,3,1));
end

save(fullfile(cfg.res_dir,'sensitivity_results.mat'),'rmse_all','mults');
fprintf('\nSaved: sensitivity_results.mat\n');
