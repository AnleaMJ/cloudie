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
model fits see training storms only. `select_features` takes those storms
from `storm_ids_by_split` and ignores val and test rows that are still in
the frame. Phase 2 scores the validation storms.
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

`src/evaluate.py` scores supplied wind predictions on the test storms from a
storm split. MAE, RMSE, and bias are reported overall and inside our own
grouping, IMD-style names, including VSCS and ESCS/SuCS. The report records
`wind_averaging` and does not convert 1-minute NASA HURDAT2 winds into the
3-minute winds used by IMD class definitions. Mean and ridge reuse
`src/baselines.py` and are fit on train storms. Persistence is the previous
wind in the same storm. Residual-quantile widths are fit on validation storms
and checked on test storms. Monte Carlo dropout coverage reads stored samples
and does not import torch. Intervals whose coverage misses the nominal level
by more than the tolerance are labeled `uncalibrated`. The written files are
software checks. `quote_as_result` stays `no`.

`cloudie/predict.py` and the Streamlit page read a checkpoint.
`present_as_forecast` stays false. A missing file or a failed missing-pixel
gate produces no number. The page reports wind speed in knots, the intensity
category of that wind, and the share of model samples in that category.
Pull current status from IMD reads the public Infrared-1 JPEG at request
time and does not store it. Vercel serves that page from the repository-root
`app.py`. `streamlit run cloudie/app.py` remains the local page.
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
| `pixel_*` or `tb_*` | Min, max, mean, std, percentiles, interquartile range on finite pixels | Pixel stats. Raw digital numbers stay `pixel_*`. `tb_` is used only after a brightness-temperature calibration table has been applied. |
| `cold_frac_*` | Fraction of finite pixels below each configured threshold | Cold-cloud coverage from that mini-project. Thresholds are image units, not a hidden Kelvin scale. |
| `radial_mean_*` | Mean in equal-width annuli from the crop center | Radial structure. Wrong if the storm is not near the center. |
| `eye_center_mean`, `cdo_annulus_mean`, `eye_cdo_contrast`, `eye_present` | Warm center minus the surrounding annulus. The flag is contrast at or above `eye_contrast_min`. | A warm-eye versus central-dense-overcast cue. Not an eye bulletin. A cold core is not marked as an eye. |
| `dav_simplified_r*` | Circular variance of (gradient angle minus radial angle) at fixed fractions of the crop half-width. The ring is at least two pixels thick. Pixels with no gradient are ignored, because a zero gradient has no direction. | P5 (Ritchie et al. 2012) DAV, implemented as a simplified gradient-angle variance. Not operational DAV: no radius search and no flight-level wind. |
| `glcm_*` | Contrast, dissimilarity, homogeneity, energy, correlation, and ASM, averaged over four angles at one distance | Grey-level texture. Levels and distance come from config. |
| `haar_energy_*` | Share of energy in the four single-level Haar bands | P8 (Zhang et al. 2025) used a stationary wavelet transform and chose Daubechies-2 after comparing it with Haar. This is a reduced form: one Haar level on one channel. Not a multichannel fusion model. |
| `has_prev`, `delta_*` | Change versus the previous frame of the same storm after cross-split links are removed. Missing history stays missing and is imputed inside the training fold. | Addresses the limited use of storm history. The link cannot cross train, val, and test. |

Wind speed, R34, RMW, and the class label are forbidden as inputs. The
survey flags inputs that are not available from the image alone: P4's best
model (Scheme 3) adds best-track intensity, and P6 (DeepTCNet) adds a TC
fullness value taken from TC Vitals or best track. Stratifying a split on the label
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

`data.image_size: native` keeps each chip's height and width. NASA TorchGeo
chips are 366×366 (Gurung et al. 2023). TCIR chips are 201×201. An integer
resizes the deep model to that square. Classical features stay on the native
crop either way.

Project class edges are this project's own grouping, not an IMD warning and
not the IMD 3-minute scale. IMD uses 3-minute winds. The edges are [17, 34)
D_DD, [34, 48) CS, [48, 64) SCS, [64, 91) VSCS, [91, ∞) ESCS_SUCS. Rows below
17 kt are omitted from classification and kept for regression. NASA
TropicalCyclone labels are HURDAT2 1-minute winds (`data.wind_averaging:
1min`). Names are applied only when the manifest records that single period.

## Literature mapping (P1 to P8)

Every row was checked against the paper itself with a live web search on
25 Oct 2026. Numbers are the papers' own results, never this repository's.

| ID | Paper | What it reports | Cloudie choice it supports | Gap it motivates |
| --- | --- | --- | --- | --- |
| P1 | Dharpure et al., "Deep Learning-Based Cyclone Intensity Estimation Using INSAT-3D IR Imagery: A Comparative Study", IJRPR 4(4):4359-4365, Apr 2023 (no DOI) | CNN, RNN and CNN-RNN on 1,184 INSAT-3D IR images over the North Indian Ocean. The abstract gives MAE 4.89 kt for CNN-RNN, but the introduction gives 7.87 kt. | INSAT-3D IR scope, CNN regression | Inconsistent metrics, no split or baselines, small data: storm-wise splits, classical baselines, gated pretrained weights |
| P2 | Jung et al., Frontiers in Earth Science 11:1285138, published 19 Jan 2024, doi:10.3389/feart.2023.1285138 | COMS to GK2A CNN transfer learning in the Western North Pacific. The image-only TLFFT3 model has a test RMSE of 11.17 kt, against 13.99 kt without transfer. The 3.79 kt 6-hour model also takes an earlier intensity as input. | Transfer across sensors, pretrained ResNet-18 | One basin, one test year, no uncertainty: basin-holdout and year-block splits, MC dropout, best-track-free inputs |
| P3 | Zhang et al., TCICENet, IEEE JSTARS 14:2070-2086, 2021, doi:10.1109/JSTARS.2021.3050767 | Cascaded classification then estimation on Northwest Pacific IR, tested on 2017-2019. The headline RMSE of 8.60 kt is the time-smoothed TCICENet-S. The unsmoothed model is 9.59 kt. | CNN regression, optional class head | Grade cascade errors, no temporal model: direct regression, `delta_*`, optional ConvLSTM |
| P4 | Baek et al., TC-SEM, Remote Sensing 14(2):426, 17 Jan 2022, doi:10.3390/rs14020426 | Multi-task CNN for RMW and R34 from COMS IR. The best scheme adds best-track intensity as an input, and the split is a random 68:6:26 frame split. | Size head for R34 and RMW | Best-track input and random frame split: size head without best-track wind, storm never cut |
| P5 | Ritchie et al., Weather and Forecasting 27(5):1264-1277, Oct 2012, doi:10.1175/WAF-D-11-00156.1 | Improved DAV on 95 North Atlantic storms (2004-2010). RMSE is 12.9 kt overall, 17.7 kt at Category 4 and 32.4 kt at Category 5. It uses the best-track center and picks the best radius per year. | `dav_simplified_r*` | Weak on extremes, needs a best-track center: simplified DAV, pre-centered images, per-class error reporting |
| P6 | Zhuo and Tan, DeepTCNet, Monthly Weather Review 149(7):2097-2113, Jul 2021, doi:10.1175/MWR-D-20-0333.1 | Physics-augmented VGG-13 on HURSAT-B1 IR, North Atlantic. RMSE is 8.7 kt on the 2017 and 2019 recon-aided test, using a TC fullness input. | CNN regression, multi-task size head | Needs a TC fullness input, Atlantic only: image-only inputs, North Indian Ocean and INSAT path |
| P7 | Manju et al., PLoS One 20(12):e0330705, 5 Dec 2025, doi:10.1371/journal.pone.0330705 | ConvLSTM on 4 consecutive 6-hour frames, 15 North Indian Ocean cyclones. The 81.1% grading accuracy (MeanShift) and the 7.79 kt RMSE (K-Means) come from different variants. | Optional 4-frame ConvLSTM, project classes | 15 storms, stratified frame split: storm-wise splits, SMOTE inside the fold, MC dropout |
| P8 | Zhang et al., IEEE JSTARS 18:4711-4735, 2025, doi:10.1109/JSTARS.2025.3531448 | Stationary wavelet transform plus IR, WV and PMW fusion on TCIR (2003-2017). MAE is 3.76 m/s and RMSE is 4.62 m/s. Daubechies-2 was chosen over Haar. | `haar_energy_*`, as a reduced form | Two-stage, no time evolution, no North Indian Ocean test: single-channel Haar, `has_prev` and `delta_*` |

Other sources checked the same day: ConvLSTM was introduced by Shi et al. at
NIPS 2015 (arXiv:1506.04214). TCIR (Chen, Chen and Lin, KDD 2018) stores
201x201 frames with IR1, WV, VIS and PMW channels. The NASA Tropical Storm
Wind Speed labels are HURDAT2 1-minute winds on 366x366 chips (Gurung et al.
2023, doi:10.1029/2022EA002693). IMD classes use 3-minute winds, and the IMD
FAQ and SOP Table 1.1 put Extremely Severe at 90 to 119 kt.

## Results

No experiment has been run. Leave this section empty until a real training
run on real imagery. Files written by `python -m src.baselines --config config/config_smoke.yaml` are software
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
- Why can NASA winds not be labeled with IMD names blindly? NASA labels are
  HURDAT2 1-minute winds. IMD categories use 3-minute winds. The
  17/34/48/64/91 kt bins are this project's grouping. `wind_averaging` has
  to record one period.
- What is a smoke MAE? A software check. It is not a result.
- What is the Monte Carlo dropout standard deviation? It is the spread of
  stochastic forward passes. `src/evaluate.py` can score coverage from stored
  samples, and from validation residual quantiles, on a storm-held-out split.
  No real-imagery coverage check has been run. A dropout standard deviation
  is not itself a calibrated interval. Intervals that miss the nominal
  coverage by more than the tolerance are labeled uncalibrated.
- Why is 90 knots `VSCS` here when the current IMD sheet starts Extremely
  Severe at 90 knots? The 17/34/48/64/91 kt bins are this project's own
  grouping, not official IMD classes. The project edge is the configured
  value 91 knots. The names are not an IMD warning.
- Why does a smoke checkpoint produce no forecast in the demo? The checkpoint
  records `is_synthetic` and `present_as_forecast: false`. The page shows
  that banner and does not call the number a forecast.
- Why is the size head absent from a smoke run? The smoke manifest has no
  `r34_nmi` or `rmw_nmi`. The head is a linear map from image features, and
  best-track wind is not one of those features.
