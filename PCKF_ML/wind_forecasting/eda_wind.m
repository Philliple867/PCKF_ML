addpath(genpath('../core'));
cfg = config();
rng(cfg.seed);

ropts = {'Delimiter',',','VariableNamingRule','preserve'};
tr = readtable(cfg.f_train, ropts{:});
va = readtable(cfg.f_val,   ropts{:});
te = readtable(cfg.f_test,  ropts{:});

fprintf('DEBUG cols: %s\n', strjoin(tr.Properties.VariableNames,', '));
fprintf('DEBUG size: %d x %d\n', size(tr,1), size(tr,2));

sets  = {tr, va, te};
names = {'2017 (train)', '2018 (val)', '2019 (test)'};

fprintf('%-20s %6s %6s %6s %6s %7s %6s\n','','mean','std','min','max','skew','k');
fprintf('%s\n', repmat('-',1,58));

for s = 1:3
    v  = sets{s}.WindSpeed_100m_mps;
    pd = fitdist(v(v>0), 'Weibull');
    fprintf('%-20s %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f\n', ...
        names{s}, mean(v), std(v), min(v), max(v), skewness(v), pd.A);
end

v_tr = tr.WindSpeed_100m_mps;
acf  = autocorr(v_tr, 'NumLags', 48);
fprintf('\nAutocorr: lag1=%.3f  lag6=%.3f  lag24=%.3f\n', acf(2),acf(7),acf(25));
ramp = abs(diff(v_tr));
fprintf('Max hourly ramp: %.2f m/s/h\n', max(ramp));
fprintf('Hours with ramp > 3 m/s: %.1f%%\n', 100*mean(ramp>3));

v_all   = [tr.WindSpeed_100m_mps; va.WindSpeed_100m_mps; te.WindSpeed_100m_mps];
pd_all  = fitdist(v_all(v_all>0), 'Weibull');
fprintf('\nFull dataset: Weibull k=%.4f, lambda=%.4f\n', pd_all.A, pd_all.B);
fprintf('|skewness| = %.4f\n', abs(skewness(v_all)));

Pw_nom = wind_to_power(mean(v_all), cfg);
fprintf('\nP_w at mean wind (%.2f m/s) = %.4f pu\n', mean(v_all), Pw_nom);
fprintf('P_m_eff_nom = %.4f pu  -->  delta_0 = %.4f rad\n', cfg.Pm_eff_nom, cfg.d0);

stats.mu     = mean(v_all); stats.sigma = std(v_all);
stats.k_wei  = pd_all.A;   stats.lam   = pd_all.B;
stats.ac1    = acf(2);     stats.ac24  = acf(25);
if ~isfolder(cfg.res_dir); mkdir(cfg.res_dir); end
save(fullfile(cfg.res_dir,'eda_stats.mat'), 'stats');
fprintf('\nSaved: eda_stats.mat\n');