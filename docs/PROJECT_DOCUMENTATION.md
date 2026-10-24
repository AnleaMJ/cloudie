# Project documentation

## Abstract

A leak-free pipeline for tropical-cyclone intensity regression from infrared
imagery. Classical feature baselines are implemented so a later neural net
has something honest to beat. The deep path can train, evaluate, and show a
model output. No experiment has been run. No performance number in the
literature survey is a result of this repository.

## Methodology

Outer splits are storm-wise, year-block, or basin-holdout. A storm is never
cut. The test side of that split is locked: feature selection, SMOTE, and
model fits see training storms only. Phase 2 scores the validation storms.
The deep path in `cloudie/` also reports the test split against the train-set
mean, and that report is a software check until the imagery is real.

Inner comparison for the classical models uses `StratifiedGroupKFold` on
training storms, stratified by the project class and grouped by `storm_id`.
SMOTE, when enabled, runs inside the current training fold and only for
classification. Regression uses the training labels as they are. Class
weights are the fallback when SMOTE is off or a fold is too small for it,
except for k-nearest neighbors, which has no class-weight parameter.

`cloudie/train.py` stops after `train.patience` validation checks without an
improvement and writes a checkpoint under `artifacts/`. The wind loss is
weighted Smooth L1. An auxiliary class loss uses weights fit on the train
split only. A size term is added only when the train split has finite
`r34_nmi` and `rmw_nmi`. Best-track wind is not an input to that head.

Model interfaces on the deep path: a train-set mean baseline, `SimpleCNN`,
ResNet-18 with a one-channel image repeated to three channels so a pretrained
first convolution can be kept, and an optional 4-frame ConvLSTM (Shi et al.,
NeurIPS 2015, written for precipitation nowcasting). The sequence model is
not a claim of intensity skill. Windows stay inside one storm and one split.
Monte Carlo dropout puts BatchNorm in eval and Dropout in train.
`pretrained: true` is refused unless a local environment variable allows the
ImageNet download. Southern Hemisphere frames are not flipped.

`cloudie/predict.py` and the Streamlit page read a checkpoint.
`present_as_forecast` stays false. A missing file or a failed missing-pixel
gate produces no number. The page reports wind speed in knots, the intensity
category of that wind, and the share of model samples in that category.
Pull current status from IMD reads the public Infrared-1 JPEG at request
time and does not store it.
A frame outside the checkpoint's infrared values, or a wind outside
0–200 kt, produces no number and no confidence. A jump versus a last
advisory and a project-class change are review notes.
The dropout standard deviation is not a calibrated interval.

Normalization statistics for the deep path fit on train only. Classical
features are computed on the native image, not on a z-scored tensor, so
cold thresholds stay in image units. Set `features.calibrated` only after a
real calibration table has been applied. The names then switch from
`pixel_*` to `tb_*`. This code does not look up a sensor calibration.

Images are assumed pre-centered. Best-track latitude and longitude are not
used to crop, and best-track intensity is not a feature.

## Feature dictionary

| Feature | What it is | Survey link |
| --- | --- | --- |
| `pixel_*` or `tb_*` | Min, max, mean, std, percentiles, interquartile range on finite pixels | Phase-2 mini-project brightness-temperature statistics. `tb_` is used only when the array is already calibrated. |
| `cold_frac_*` | Fraction of finite pixels below each configured threshold | Cold-cloud coverage from that mini-project. Thresholds are image units, not a hidden Kelvin scale. |
| `radial_mean_*` | Mean in equal-width annuli from the crop center | Radial structure. Wrong if the storm is not near the center. |
| `eye_center_mean`, `cdo_annulus_mean`, `eye_cdo_contrast`, `eye_present` | Warm center minus the surrounding annulus. The flag is contrast at or above `eye_contrast_min`. | A warm-eye versus central-dense-overcast cue. Not an eye bulletin. A cold core is not marked as an eye. |
| `dav_simplified_r*` | Circular variance of (gradient angle minus radial angle) at fixed fractions of the crop half-width. The ring is at least two pixels thick. Pixels with no gradient are ignored, because a zero gradient has no direction. | The survey's DAV work, implemented as a simplified gradient-angle variance. Not operational DAV: no radius search and no flight-level wind. |
| `glcm_*` | Contrast, dissimilarity, homogeneity, energy, correlation, and ASM, averaged over four angles at one distance | Grey-level texture. Levels and distance come from config. |
| `haar_energy_*` | Share of energy in the four single-level Haar bands | The survey's wavelet paper, reduced to one Haar level on one channel. Not a multichannel fusion model. |
| `has_prev`, `delta_*` | Change versus the previous frame of the same storm after cross-split links are removed. Missing history stays missing and is imputed inside the training fold. | Addresses the limited use of storm history. The link cannot cross train, val, and test. |

Wind speed, R34, RMW, and the class label are forbidden as inputs. That is
the operational hole called out for best-track-dependent models in the
survey, including the DeepTCNet setting. Stratifying a split on the label
is allowed. Feeding the label back in is not.

## Models

Regression: training-mean predictor, ridge, k-nearest neighbors, RBF SVR,
random forest, and XGBoost when the package imports. Classification of the
five project classes: most-frequent training class, logistic regression,
k-nearest neighbors, RBF SVC, random forest, and XGBoost. Inputs are
median-imputed and standardized inside the training fold.

The comparison table reports MAE, RMSE, and bias (prediction minus label),
the same MAE inside each project class, and macro-F1 with per-class recall.
Accuracy is not the headline. ADT, SATCON, and Dvorak columns are present
and empty.

The deep path adds `SimpleCNN`, ResNet-18, and the optional ConvLSTM. A
smoke fit is not a row in the results table below.

Project class edges group IMD sustained-wind bins: [17, 34) D_DD, [34, 48)
CS, [48, 64) SCS, [64, 91) VSCS, [91, ∞) ESCS_SUCS. They are not a warning
product. Rows below 17 kt are omitted from classification and kept for
regression. Names are not applied to 1-minute winds unless `wind_averaging`
says so, and a manifest may contain only one averaging period.

## Results

No experiment has been run. Leave this section empty until a real training
run on real imagery. Files written by `python -m src.baselines` are software
checks. Smoke rows are marked `synthetic=yes` and `quote_as_result=no`.
Do not paste those numbers here. Do not paste a smoke-test loss from
`cloudie` here either.

## Limitations

Noisy best-track labels, class imbalance, sensor shift, and weak skill on
rare intense storms are expected risks, not measured findings. Simplified
DAV, a single Haar level, and a one-frame delta do not reproduce the
papers they are drawn from. Southern Hemisphere images are not flipped.
Monte Carlo dropout is not a calibrated interval.
This code does not issue warnings. The Streamlit page still needs a person
to review any category change, any jump versus the last advisory, and any
crop that fails the missing-pixel gate.

## Viva notes

- Why not a random split? Adjacent frames of one storm are near-duplicates.
- Why is intensity not an input? It is unavailable, or leakage, in real time.
- Why is SMOTE inside the fold? Synthetic rows must not be built from
  validation or test storms.
- What is simplified DAV not? It is not the operational radius-search product.
- Why can NASA winds not be labeled with IMD names blindly? Averaging
  periods differ. The column `wind_averaging` has to say which one.
- What is a smoke MAE? A software check. It is not a result.
- What is the Monte Carlo dropout standard deviation? It is the spread of
  stochastic forward passes. It becomes a prediction interval only after a
  storm-held-out coverage check, which has not been run.
- Why is 90 knots `VSCS` here when the current IMD sheet starts Extremely
  Severe at 90 knots? The project edge is the configured value 91 knots,
  grouping Extremely Severe with Super. The names are not an IMD warning.
- Why does a smoke checkpoint produce no forecast in the demo? The checkpoint
  records `is_synthetic` and `present_as_forecast: false`. The page shows
  that banner and does not call the number a forecast.
- Why is the size head absent from a smoke run? The smoke manifest has no
  `r34_nmi` or `rmw_nmi`. The head is a linear map from image features, and
  best-track wind is not one of those features.
