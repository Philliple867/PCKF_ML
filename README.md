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
│   │   ├── collocation_pts.m <- Xi matrix (paper Eq. cp)
│   │   └── gpc_basis.m       <- Hhat matrix
│   ├── microgrid/
│   │   ├── swing_eq.m
│   │   ├── swing_eq_discrete.m
│   │   ├── wind_to_power.m   <- cubic power curve (paper Eq. power_curve)
│   │   ├── pmu_model.m
│   │   └── mg_params.m
│   └── utils/
│       ├── sliding_window.m
│       ├── nll_loss.m        <- Gaussian NLL (paper Eq. NLL)
│       └── rmse_metrics.m
└── results/
```


