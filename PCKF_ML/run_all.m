function run_all(mode)
if nargin < 1; mode = 'full'; end
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root,'core')));
addpath(root);
cfg = config();
if ~isfolder(cfg.res_dir); mkdir(cfg.res_dir); end

fprintf('\n=== EPEC 2026 -- PCKF-ML Pipeline ===\n\n');
t0 = tic;

for fn = {cfg.f_train, cfg.f_val, cfg.f_test}
    if ~isfile(fn{1})
        error('Missing: %s\nPlace CSV files in data/', fn{1});
    end
end

if strcmp(mode,'full')
    run_s(fullfile(root,'wind_forecasting','eda_wind.m'),    root);
    run_s(fullfile(root,'wind_forecasting','train_lstm.m'),  root);
    run_s(fullfile(root,'wind_forecasting','eval_lstm.m'),   root);
end

run_s(fullfile(root,'dse_simulation','run_simulation.m'),    root);
run_s(fullfile(root,'dse_simulation','monte_carlo.m'),       root);
run_s(fullfile(root,'dse_simulation','sensitivity_noise.m'), root);

fprintf('\nDone. Total: %.1f min\nResults in: %s\n', toc(t0)/60, cfg.res_dir);
end

function run_s(p, root)
[~, n] = fileparts(p);
t = tic;
fprintf('[running] %s.m\n', n);
try
    evalin('base', sprintf("addpath(genpath('%s'));", ...
        strrep(fullfile(root,'core'),'\','\\')));
    evalin('base', sprintf("addpath('%s');", strrep(root,'\','\\')));
    evalin('base', sprintf("run('%s');",    strrep(p,  '\','\\')));
    fprintf('[done]    %s.m (%.1fs)\n\n', n, toc(t));
catch ME
    fprintf('[ERROR]   %s.m: %s\n', n, ME.message);
    rethrow(ME);
end
end