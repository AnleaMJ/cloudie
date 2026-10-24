# Cloud agent handoff

Phase 0–1 (schema, splits, smoke data, metrics, model interfaces) and
Phase 2 (features and classical baselines) are in this repository. Continue
from here. Do not restart from scratch and do not reimplement the baseline
table.

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
- Read hyperparameters from `config/config.yaml`.
- `data.image_size` is for the deep model. Classical features stay at native
  resolution.

## Next issue: Phase 3, deep models

`src/model.py` already defines `SimpleCNN`, `ResNetRegressor`, and
Monte Carlo dropout. Add `src/train.py` and `src/evaluate.py`:

- train only the training storms, early stopping on validation storms,
  leave the test split locked
- one checkpoint per stage, under `artifacts/`, which is gitignored
- loss is weighted smooth-L1 on wind. A class head is optional and must
  not become the headline metric
- repeat a one-channel image to three channels before ResNet-18 so a
  pretrained `conv1` can be kept. `pretrained=true` downloads weights and
  must not run in CI
- report MAE, RMSE, and bias, including inside VSCS and ESCS_SUCS, and
  compare with the Phase 2 mean and ridge rows from a real run
- MC-dropout standard deviation is epistemic noise under dropout, not a
  calibrated interval, until a storm-held-out coverage check says otherwise
- a Colab notebook may repeat the local script; it must not embed scores
  from smoke data

## Later, only after Phase 3 tests pass

1. `src/predict.py` and a Streamlit app. If the checkpoint is missing or
   was fit on smoke data, show a banner and refuse to present the number
   as a forecast. The app may load a Phase 2 joblib pipeline only after
   one is saved by an explicit export. Do not treat the smoke CSV as that
   artifact.
2. Optional 4-frame ConvLSTM and a size head. Size labels must already
   exist. Do not feed best-track intensity into that head.
