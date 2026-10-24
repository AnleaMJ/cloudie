# Cyclone intensity estimation

Academic framework for estimating tropical-cyclone maximum sustained wind from
infrared imagery. The primary task is regression in knots. Five project
classes are an optional label for error tables, not an IMD warning.

`src/` holds Phase 2: image features and classical baselines. `cloudie/`
holds the deep path: Phase 3 training and evaluation, Phase 4 prediction,
and a Phase 5 four-frame ConvLSTM. Size labels (R34, radius of maximum wind)
stay out of the inputs. A size head is fit only when those columns exist on
the train split. No score in this repository is a scientific result.

## Honesty note

- No trained weights are shipped. There are no results to quote.
- `src/make_smoke_data.py` and `python -m cloudie.make_smoke_data` write
  synthetic images. They are not cyclones. A fit on them only proves the
  software runs.
- This is not an operational forecast or warning system. Best-track labels
  are noisy. Agreement with a best track is not ground truth.
- ADT, SATCON, and Dvorak columns in the baseline table are empty until
  those products are joined. Literature RMSE figures are not this repo's scores.
- Monte Carlo dropout returns a mean and a standard deviation. That standard
  deviation is dropout noise. It is not a calibrated prediction interval.
- Do not commit NASA, TCIR, IBTrACS, or INSAT-3D files. See `docs/DATA_SOURCES.md`.
- Add a code license before any public push. Third-party data stays under
  its own terms and is not included.

## What is implemented

Leak-free splits and quality gates (Phase 0–1), then classical baselines
(Phase 2, under `src/`):

- pixel or brightness-temperature stats, cold-pixel fractions, a radial
  profile, a warm-eye flag, simplified deviation-angle variance, GLCM,
  single-level Haar energy, and a same-storm temporal delta
- feature selection fit on the training storms only
- mean, ridge, k-nearest neighbors, RBF SVM, random forest, and XGBoost
  when it is installed, for wind regression and for the five project classes
- inner `StratifiedGroupKFold` on training storms, SMOTE inside those
  training folds only, and a single score on the validation storms
- the test split is not used for selection, resampling, or fitting

The deep path is `cloudie/`:

- Phase 3 trains with early stopping from `train.patience`, writes a
  checkpoint under `artifacts/`, and compares the held-out storms with the
  train-set mean. The wind loss is weighted Smooth L1. The class loss is
  auxiliary.
- Phase 4 is `cloudie/predict.py` and `streamlit run cloudie/app.py`. A
  missing checkpoint or a smoke checkpoint is not a forecast.
- Phase 5 is an optional 4-frame ConvLSTM (`model.name: convlstm`). The size
  head exists only when the train split has `r34_nmi` and `rmw_nmi`.

See `docs/AGENT_HANDOFF.md` and `docs/PROJECT_DOCUMENTATION.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m src.make_smoke_data --out data/smoke --profile balanced
python -m src.baselines --config config/config_smoke.yaml
```

`requirements-train.txt` adds PyTorch, torchvision, matplotlib, h5py, and
Streamlit. CI installs `requirements.txt` only and does not download imagery.
Training tests, the ConvLSTM tests, and the Streamlit page test skip when
those packages are absent.

The baseline command writes `reports/smoke/baselines/`. That directory is
gitignored. The CSV has `quote_as_result=no` on every row. Read
`reports/smoke/baselines/README.txt` before looking at a number.

A smoke training check needs the train extras. It does not produce a score
to quote, and it overwrites `data/smoke` with the `cloudie` synthetic set:

```bash
pip install -r requirements-train.txt
make train-smoke
streamlit run cloudie/app.py
```

If `artifacts/checkpoint.pt` is missing, or the checkpoint was fit on smoke
data, the page shows a banner and does not present a forecast. Dropout
standard deviation on that page is not a calibrated interval. Four uploaded
frames, named in time order, are the input when the checkpoint is the ConvLSTM.

## Splits

Allowed: `storm_wise`, `chronological`, `holdout_basin`. Random frame splits
raise `LeakageError`. A storm that crosses a holdout year or basin is not
cut. Normalization statistics fit on train only. Augmentation refuses every
split except train. Wind speed, R34, and RMW are labels, not inputs.

`data.image_size` is the deep-model resize. Classical features keep the
native crop and assume the storm is already near the center. This code
does not recenter with a best-track fix.

Project class names group India Meteorological Department categories for
metrics: `[17, 34)` `D_DD`, `[34, 48)` `CS`, `[48, 64)` `SCS`, `[64, 91)`
`VSCS`, and `91` and above `ESCS_SUCS`. The July 2024 IMD 3-minute sheet
starts Extremely Severe at 90 knots, so 90 knots is `VSCS` here. These names
are not a warning, and they do not transfer onto 1-minute winds unless
`wind_averaging` records the period.

## Failure modes

Extreme-intensity underestimation is a known risk in the literature and is
not measured here. Other risks to keep visible: shear and exposed centers,
night-only visible imagery, sensor shift, the opposite sense of rotation in
the Southern Hemisphere (do not silently flip), mixed 1-minute and 10-minute
winds, and near-duplicate frames. The simplified DAV feature is not an
operational DAV product.

## Layout

```
src/                     Phase 2 features and classical baselines
cloudie/                 deep path: train, evaluate, predict, ConvLSTM, Streamlit
config/                  config.yaml and config_smoke.yaml
tests/                   pytest, using temporary smoke data
docs/                    data sources, project notes, handoff
```
