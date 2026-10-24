# Cyclone intensity estimation

Academic framework for estimating tropical-cyclone maximum sustained wind from
infrared imagery. The primary task is regression in knots. Five project
classes are an optional label for error tables, not an IMD warning.

Phase 2 is in the tree: image features and classical baselines. No deep
model has been trained. No score in this repository is a scientific result.

## Honesty note

- No trained weights are shipped. There are no results to quote.
- `src/make_smoke_data.py` writes synthetic images. They are not cyclones.
  A fit on them only proves the software runs.
- This is not an operational forecast or warning system. Best-track labels
  are noisy. Agreement with a best track is not ground truth.
- ADT, SATCON, and Dvorak columns in the baseline table are empty until
  those products are joined. Literature RMSE figures are not this repo's scores.
- Do not commit NASA, TCIR, IBTrACS, or INSAT-3D files. See `docs/DATA_SOURCES.md`.
- Add a code license before any public push. Third-party data stays under
  its own terms and is not included.

## What is implemented

Leak-free splits and quality gates (Phase 0–1), then classical baselines
(Phase 2):

- pixel or brightness-temperature stats, cold-pixel fractions, a radial
  profile, a warm-eye flag, simplified deviation-angle variance, GLCM,
  single-level Haar energy, and a same-storm temporal delta
- feature selection fit on the training storms only
- mean, ridge, k-nearest neighbors, RBF SVM, random forest, and XGBoost
  when it is installed, for wind regression and for the five project classes
- inner `StratifiedGroupKFold` on training storms, SMOTE inside those
  training folds only, and a single score on the validation storms
- the test split is not used for selection, resampling, or fitting

Deep training, Monte Carlo dropout evaluation, and the Streamlit demo are
later phases. See `docs/AGENT_HANDOFF.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m src.make_smoke_data --out data/smoke --profile balanced
python -m src.baselines --config config/config_smoke.yaml
```

`requirements-train.txt` adds torch for a later training phase. CI does not
install it and does not download imagery.

The baseline command writes `reports/smoke/baselines/`. That directory is
gitignored. The CSV has `quote_as_result=no` on every row. Read
`reports/smoke/baselines/README.txt` before looking at a number.

## Splits

Allowed: `storm_wise`, `chronological`, `holdout_basin`. Random frame splits
raise `LeakageError`. A storm that crosses a holdout year or basin is not
cut. Normalization statistics fit on train only. Augmentation refuses every
split except train. Wind speed, R34, and RMW are labels, not inputs.

`data.image_size` is the later deep-model resize. Classical features keep
the native crop and assume the storm is already near the center. This code
does not recenter with a best-track fix.

## Failure modes

Extreme-intensity underestimation is a known risk in the literature and is
not measured here. Other risks to keep visible: shear and exposed centers,
night-only visible imagery, sensor shift, the opposite sense of rotation in
the Southern Hemisphere (do not silently flip), mixed 1-minute and 10-minute
winds, and near-duplicate frames. The simplified DAV feature is not an
operational DAV product.
