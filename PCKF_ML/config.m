function cfg = config()


cfg.seed = 42;

%% paths
cfg.root     = fileparts(mfilename('fullpath'));
cfg.data_dir = fullfile(cfg.root, 'data');
cfg.res_dir  = fullfile(cfg.root, 'results');

cfg.f_train  = fullfile(cfg.data_dir, 'train_wind_100m.csv');
cfg.f_val    = fullfile(cfg.data_dir, 'validate_wind_100m.csv');
cfg.f_test   = fullfile(cfg.data_dir, 'test_wind_100m.csv');
cfg.f_lstm   = fullfile(cfg.res_dir,  'lstm_output.csv');
cfg.f_model  = fullfile(cfg.res_dir,  'lstm_model.mat');

%% LSTM
cfg.seq_len  = 24;
cfg.features = {'WindSpeed_100m_mps', 'WindDirection_100m_deg', ...
                'Hour', 'Month', 'DayOfYear'};
cfg.target   = 'WindSpeed_100m_mps';
cfg.hidden1  = 128;
cfg.hidden2  = 64;
cfg.dropout  = 0.20;
cfg.epochs   = 60;
cfg.batchsz  = 64;
cfg.lr0      = 1e-3;
cfg.patience = 8;
cfg.lr_decay = 0.5;
cfg.gradclip = 1.0;
cfg.lsig_min = -4;
cfg.lsig_max =  2;

%% microgrid -- per unit, 100 MVA base, 60 Hz
cfg.H    = 5.0;       % inertia constant [s]
cfg.D    = 2.0;       % damping coefficient [pu]
cfg.Xd   = 0.30;      % transient reactance [pu]
cfg.E    = 1.05;      % internal voltage magnitude [pu]
cfg.V    = 1.00;      % terminal voltage magnitude [pu]
cfg.theta = 0.00;     % terminal voltage angle [rad]
cfg.ws   = 2*pi*60;   % synchronous speed [rad/s]
cfg.Tm   = 0.80;      % diesel mechanical power setpoint [pu]

%% wind turbine 
cfg.v_ci = 3.5;       % cut-in speed [m/s]
cfg.v_r  = 12.0;      % rated speed [m/s]
cfg.v_co = 25.0;      % cut-out speed [m/s]
cfg.P_r  = 0.30;      % rated wind power [pu]

% steady-state operating point WITH wind at Kahuku mean speed (7.9 m/s)
% P_m_eff = T_m - P_w(v_mean)  
v_nom   = 7.9;
Pw_nom  = cfg.P_r * (v_nom^3 - cfg.v_ci^3) / (cfg.v_r^3 - cfg.v_ci^3);
cfg.Pm_eff_nom = cfg.Tm - Pw_nom;   % ≈ 0.720 pu
% sin(d0) = P_m_eff * Xd / (E * V)
cfg.d0  = asin(cfg.Pm_eff_nom * cfg.Xd / (cfg.E * cfg.V));
cfg.w0  = cfg.ws;

%% simulation
cfg.dt     = 0.02;           % 50 Hz (PMU rate) [s]
cfg.T      = 100;            % total time [s]
cfg.Nsteps = cfg.T / cfg.dt; % = 5000 steps
cfg.noise_R = 0.01;          % PMU noise std [pu]

% Wind scenario: [t_start, t_end, v_wind_mean (m/s), sigma_wind (m/s)]
% v_wind_mean drives P_m_eff; sigma_wind drives Q_t and wind noise.
cfg.scenario = [
     0,  20,  7.9,  1.5;
    20,  40, 10.5,  4.5;
    40,  60,  5.5,  4.5;
    60,  80,  9.0,  4.5;
    80, 100,  7.0,  1.0];

%% PCKF
cfg.N       = 2;              % state dim: [delta, omega]
cfg.Q_fixed = diag([1e-4, 1e-4]);
cfg.R_meas  = diag([cfg.noise_R^2, cfg.noise_R^2]);
cfg.P0      = diag([0.01^2, 0.01^2]);
cfg.Q_min   = 1e-6;
cfg.Q_max   = 1e-2;

% adaptive Q scaling calibrated so Q_t = Q_fixed when sigma = mean_sigma
% mean sigma_wind from Kahuku LSTM output ≈ 3.18 m/s
mean_sig = 3.18;
cfg.alpha_d = sqrt(1e-4) / mean_sig;
cfg.alpha_w = sqrt(1e-4) / mean_sig;

%% Monte Carlo
cfg.N_MC     = 100;
cfg.mc_seed0 = 1000;

%% sensitivity analysis: noise multipliers on cfg.noise_R
cfg.noise_mults = [0.5, 1.0, 2.0, 5.0];

end
