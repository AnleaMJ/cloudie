# Cloud agent handoff

Phase 2 lives in `src/`: features and classical baselines. Do not reimplement
that table and do not restart the package under a new name.

Shared modules are canonical in `src/`. `cloudie/` re-exports them so older
imports keep working: `config`, `data_loader`, `preprocessing`, `schema`,
`splits`, `metrics`, `model`, `make_smoke_data`, `inference`, `train`, and
`sequences`. The network classes (SimpleCNN, ResNet-18, ConvLSTM, MC-dropout)
live in `src/model.py`.

`src/train.py` is the training entrypoint (`python -m src.train`).
`src/sequences.py` holds the ConvLSTM windows. `cloudie/train.py` and
`cloudie/sequences.py` re-export them. Checkpoint scoring stays in
`cloudie/evaluate.py`. Prediction and the page stay in `cloudie/predict.py`,
`cloudie/app.py`, `cloudie/web.py`, and `cloudie/imd_satellite.py`.
Do not fold checkpoint scoring into `src/evaluate.py`, and do not copy the
held-out scorer into `cloudie/evaluate.py`. Commands are in
`docs/TRAINING.md`. A GPU run needs misty wind's approval. No results exist
yet.

Phase 3 is `cloudie/train.py` and `cloudie/evaluate.py`. Early stopping reads
`train.patience`. Checkpoints go under `paths.artifacts` and are gitignored.
The loss is weighted Smooth L1 on wind, with an auxiliary class loss weighted
by `train.class_loss_weight`. Class weights are fit on the train split only.
A size term uses `train.size_loss_weight` only when the train split has finite
`r34_nmi` and `rmw_nmi`. The comparison baseline is the train-set mean.
`notebooks/smoke_train.ipynb` says the images are synthetic. Do not paste a
smoke MAE into `docs/PROJECT_DOCUMENTATION.md`. `pretrained: true` still
refuses to download ImageNet weights unless `CLOUDIE_ALLOW_PRETRAINED=1`.

Phase 4 is `cloudie/predict.py` and `cloudie/app.py`. The page reports wind
speed in knots, the project intensity category of that wind, and the share
of model samples in that category (`model.mc_samples`). The button
`Pull current status from IMD` reads the public Infrared-1 JPEG from
`cloudie/imd_satellite.py` at request time and does not store it. Tests
pass a response in and must not download that JPEG. `present_as_forecast`
stays false. A missing checkpoint or a failed missing-pixel gate returns no
number. A frame outside the checkpoint's infrared values, or a wind outside
`data.plausible_wind_kt` (0–200), returns no number and no confidence.
Smoke checkpoints keep the synthetic banner. `predict.advisory_jump_kt`
is a review flag, not a warning threshold. Dropout standard deviation is not
the displayed score and is not a calibrated interval. Vercel loads the
repository-root `app.py` (`tool.vercel.entrypoint`). `streamlit run cloudie/app.py`
remains the local page. Weights stay gitignored, so a deployment with no
checkpoint file still returns no number.

Phase 5 is `ConvLSTMRegressor` and `cloudie/sequences.py`. `model.name: convlstm`
uses 4 frames from the same storm and the same split. A previous-frame link
that leaves the split rejects the window until `null_cross_split_links` clears
it. The image tensor does not contain wind or size. Southern Hemisphere frames
are not flipped. The size head is a linear map on image features and is absent
when the train split has no size labels.

## Hard constraints

- Splits: `storm_wise`, `chronological`, or `holdout_basin` only.
- Never random-split frames. Never cut one storm across train and test.
- Do not use wind speed, R34, RMW, or class labels as inputs.
- SMOTE or class weights stay inside the training fold. Regression does
  not use SMOTE. `src/baselines.py` already does this.
- Fit normalization and feature selection on train only.
- Do not commit data, checkpoints, or INSAT imagery.
- Do not download NASA or TCIR in CI.
- Do not write result numbers into `docs/PROJECT_DOCUMENTATION.md`.
  Smoke data is synthetic. `quote_as_result` stays `no`.
- Do not describe the repo as operational.
- Do not fill ADT, SATCON, or Dvorak columns unless those outputs are joined.
- The simplified DAV feature is not operational DAV. Do not rename it as if
  it were.
- Do not silently flip Southern Hemisphere images.
- Add pytest coverage for every new module.
- Read hyperparameters and paths from `config/config.yaml` or
  `config/config_smoke.yaml`. Do not keep a second copy in code.
- `data.image_size` is `native` (or null) unless a run sets a square resize.
  Classical features stay at native resolution. NASA TorchGeo chips are
  366×366. The smoke config may use a small square.
- Do not present Monte Carlo dropout std as a calibrated forecast interval.
- `data.wind_averaging` is `1min` for NASA HURDAT2 labels. The 17/34/48/64/91
  kt bins are the project's own grouping. IMD uses 3-minute winds.
- Do not apply the project class names unless `wind_averaging` is a single
  recorded period. The column is required (`1-min`, `3-min`, `10-min`, or
  another single label such as `synthetic`). Mixed periods fail closed.
- The 17/34/48/64/91 kt bins are the project's grouping, not official IMD
  classes. `imd5_class` stores that group. `src/evaluate.py` calls this
  grouping "our own grouping, IMD-style names" and always reports VSCS and
  ESCS/SuCS. It records `wind_averaging` and does not convert periods.
- `src/evaluate.py` scores supplied predictions on the test storms from
  `storm_ids_by_split`. Mean and ridge reuse `src/baselines.py`. Persistence
  is the previous same-storm wind, not a feature. Normalization calls
  `fit_split_normalization` in `src/train.py` (`cloudie/train.py` re-exports
  it). That helper is not defined on the splits module.
  `cloudie/splits.py` re-exports `src/splits.py`.
  Do not copy this evaluator into `cloudie/evaluate.py`. Monte Carlo dropout
  coverage reads a sample array. This module does not import torch. A
  coverage gap larger than the tolerance is labeled `uncalibrated`.
  `quote_as_result` stays `no`.
- Image statistics are pixel stats unless a brightness-temperature
  calibration table exists. Do not name raw digital numbers `tb`.

## Done in Phase 4 and Phase 5

- `cloudie/predict.py` and `streamlit run cloudie/app.py`.
- Optional 4-frame ConvLSTM via `model.name: convlstm`.
- Size head only after train-split `r34_nmi` and `rmw_nmi` exist. Best-track
  intensity is not an input to that head.

The training loss is weighted Smooth L1 on wind. A class loss is auxiliary.
There is no size term until size labels exist on the train split.
