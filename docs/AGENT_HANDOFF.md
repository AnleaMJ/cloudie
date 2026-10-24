# Cloud agent handoff

Phase 0–1 is in this repository. Continue from these modules. Do not restart the package under `src/`. The import root is `cloudie`.

## Hard constraints

- Splits: `storm_wise`, `chronological`, or `holdout_basin` only.
- Never random-split frames. Never cut one storm across train and test.
- Do not use wind speed, R34, RMW, or class labels as inputs.
- SMOTE or class weights, if a later phase uses them, stay inside the training fold.
- Fit normalization on the train split only.
- Do not commit data, checkpoints, or INSAT / MOSDAC imagery.
- Do not download NASA or TCIR in CI.
- Do not write result numbers. Smoke data is synthetic. `docs/PROJECT_DOCUMENTATION.md` keeps an empty results table on purpose.
- Do not describe the repo as operational, and do not present Monte Carlo dropout std as a calibrated forecast interval.
- Add pytest coverage for every new module.
- Read hyperparameters from `config/config.yaml`. `pretrained: true` downloads ImageNet weights and must not run in CI.
- Southern Hemisphere TCIR frames are not silently flipped.
- Do not apply the project class names to a wind column unless `wind_averaging` is a single recorded period.

## Next: Phase 2, classical baselines

Add `cloudie/features.py` and `cloudie/baselines.py`.

Features per image, fit and selected on train storms only:

- Pixel statistics. Call them brightness-temperature statistics only when the array is already calibrated.
- Cold-pixel coverage at thresholds recorded in config.
- A radial mean profile.
- A simplified gradient-angle variance, documented as not operational DAV.
- GLCM and Haar energies if those dependencies stay optional.
- A temporal delta against `prev_sample_id` after `null_cross_split_links`.

Models: the existing train-set mean predictor, then ridge, random forest, and XGBoost if it is installed. Group by `storm_id`. Write comparison tables with empty operational-baseline columns (ADT, SATCON, Dvorak stay blank until those outputs are actually joined). Map each feature to the literature survey in `docs/PROJECT_DOCUMENTATION.md` without claiming a skill gain.

## After Phase 2 tests pass

1. `cloudie/train.py` and `cloudie/evaluate.py`: early stopping from `train.patience`, checkpoints under `artifacts/`, MAE / RMSE / bias, per-class error, and a residual plot. Compare with `mean_baseline`. A Colab notebook can live in `notebooks/` and must say that smoke data is synthetic.
2. `cloudie/predict.py` and a Streamlit app. If the checkpoint is missing or was fit on smoke data, show a banner and refuse to present the number as a forecast.
3. Optional 4-frame ConvLSTM, and a size head only after `r34_nmi` / `rmw_nmi` exist as labels. Do not feed best-track intensity into that head.

Loss, when training arrives: weighted Smooth L1 on wind. A class loss is auxiliary. There is no size term until size labels exist.
