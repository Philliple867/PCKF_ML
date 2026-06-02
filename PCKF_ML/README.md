# PCKF-ML: ML-Assisted PCKF for DSE in Islanded Microgrids

IEEE EPEC 2026, Banff, AB, Canada

---

## Requirements
- MATLAB R2022b+
- Deep Learning Toolbox (`ver('nnet')`)
- Statistics and Machine Learning Toolbox (`ver('stats')`)

## Setup
Place CSV files in `data/`:
```
train_wind_100m.csv       (2017, 8760 rows)
validate_wind_100m.csv    (2018, 8760 rows)
test_wind_100m.csv        (2019, 8760 rows)
```
Source: NREL WTK, Kahuku Wind Farm, Hawaii (21.667°N, 157.952°W), 100 m hub.

## Run
```matlab
>> run_all           % full pipeline
>> run_all('sim')    % skip LSTM training
```

## Project Structure
```
EPEC2026_PCKF_ML_v2/
├── run_all.m
├── config.m
├── data/
├── wind_forecasting/
│   ├── eda_wind.m
│   ├── train_lstm.m
│   └── eval_lstm.m
├── dse_simulation/
│   ├── run_simulation.m
│   ├── monte_carlo.m
│   └── sensitivity_noise.m
├── core/
│   ├── filters/
│   │   ├── pckf.m            <- PCKF, Xu 2019 Eqs. 14-30
│   │   ├── ekf.m             <- EKF baseline
│   │   ├── collocation_pts.m <- Xi matrix 
│   │   └── gpc_basis.m       <- Hhat matrix
│   ├── microgrid/
│   │   ├── swing_eq.m
│   │   ├── swing_eq_discrete.m
│   │   ├── wind_to_power.m   <- cubic power curve 
│   │   ├── pmu_model.m
│   │   └── mg_params.m
│   └── utils/
│       ├── sliding_window.m
│       ├── nll_loss.m        <- Gaussian NLL 
│       └── rmse_metrics.m
└── results/
```

## Key Changes vs v1
- `wind_to_power.m` added -- cubic power curve, matches paper exactly
- `swing_eq_discrete.m` uses `Pm_eff = Tm - Pw(v_t)` not raw Tm
- `collocation_pts.m` ordering matches paper Eq. (cp): positive→negative→center
- `nll_loss.m` uses correct 1/2 factors: `log(sigma) + (y-mu)^2/(2*sigma^2)`
- `config.m` has wind turbine params (v_ci, v_r, v_co, P_r) and correct d0
- All simulation scripts use `wind_to_power()` for physics-consistent disturbance
