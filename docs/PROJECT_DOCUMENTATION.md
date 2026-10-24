# Project documentation

## Abstract

Cloudie is a leak-free pipeline for tropical-cyclone intensity regression from single-channel infrared imagery. The target is maximum sustained wind in knots. A 5-class project label exists for stratified metrics. No experiment has been run in this repository. No performance number from the literature survey is a result of Cloudie.

## Methodology

Phase 0–1 implements the split and schema controls below. Phase 3 adds a training loop and an evaluation report. It does not implement feature baselines or a demo app. A smoke fit is not a result.

- Splits are `storm_wise`, `chronological` (a year block), or `holdout_basin`. Random frame splits raise `LeakageError`. A storm that crosses a year or basin boundary is not cut.
- Mean and standard deviation for normalization are fit on the train split only.
- Previous-frame links that leave the row's split are removed before any later temporal feature can see them.
- Primary-model inputs exclude `wind_speed_kt`, `r34_nmi`, `rmw_nmi`, `imd5_class`, and `intensity_bin`.
- The training loop reads `config/config.yaml`. It stops after `train.patience` validation checks without an improvement, writes a checkpoint under `artifacts/`, and compares the test split with the train-set mean. The wind loss is weighted Smooth L1. An auxiliary class loss uses weights fit on the train split only. There is no size term.
- Model interfaces: a train-set mean baseline, `SimpleCNN`, and ResNet-18 with a one-channel image repeated to three channels so a pretrained first convolution can be kept. Monte Carlo dropout puts BatchNorm in eval and Dropout in train. `pretrained: true` is refused unless a local environment variable allows the ImageNet download.

## Project classes

Configured edges in knots: `[17, 34)`, `[34, 48)`, `[48, 64)`, `[64, 91)`, and `[91, ∞)`, named `D_DD`, `CS`, `SCS`, `VSCS`, and `ESCS_SUCS`.

They group India Meteorological Department categories (Depression with Deep Depression, and Extremely Severe with Super). They are a metric label. The July 2024 IMD 3-minute sheet (Cyclone Warning Division SOP) places Very Severe at 64–89 knots and Extremely Severe at 90–119 knots. This config places 90 knots in `VSCS`. Do not stamp the names on a wind column whose averaging period is unrecorded.

## Survey gaps this scaffold addresses

| Gap | Control in this repo | Status |
| --- | --- | --- |
| Random frame splits leak near-duplicate images | `LeakageError` on random, frame, or shuffle strategies | Implemented, tested on smoke data |
| Best-track intensity used as an input | `forbidden_features` | Listed and rejected as a design rule; no feature builder yet |
| No uncertainty estimate | `mc_dropout_predict` | Interface only. The std is not a calibrated interval |
| Single-basin claims | `holdout_basin` | Split exists. No cross-basin score |
| Extreme-intensity error hidden by a single MAE | `metrics_by_class` | Function exists. The results table below stays empty |
| No training loop | `cloudie/train.py` | Runs on smoke data as a software check. No real-data score |
| Little use of storm history | `null_cross_split_links` | Links cannot cross a split. ConvLSTM is not implemented |

## Results

Leave this section empty until a training run on real labeled imagery. Do not paste a smoke-test loss here. Do not copy RMSE or MAE from papers into this file.

| Run | Split | MAE (kt) | RMSE (kt) | Bias (kt) | Notes |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Limitations

Best-track labels are noisy. Class imbalance, sensor shift, and weak skill on rare intense storms are expected risks. They are not measured findings. Cloudie does not issue warnings. A later service, if one is built, still needs a person to review any category change, any jump versus the last advisory, any crop that fails the missing-pixel gate, and any interval that crosses a class boundary.

## Viva notes

- Why refuse a random split? Adjacent frames of one storm are near-duplicates, so a frame split leaks the answer into the test set.
- Why is intensity not an input? At real time it is the quantity being estimated. Feeding the best-track wind back in is leakage.
- What is the Monte Carlo dropout standard deviation? It is the spread of stochastic forward passes. It becomes a prediction interval only after a storm-held-out coverage check, which has not been run.
- Why can NASA winds not be labeled with these class names blindly? Averaging periods differ. `wind_averaging` has to name the period, and a manifest with two periods is rejected.
- Why is 90 knots `VSCS` here when the current IMD sheet starts Extremely Severe at 90 knots? The project edge is the configured value 91 knots, grouping Extremely Severe with Super. The names are not an IMD warning.
