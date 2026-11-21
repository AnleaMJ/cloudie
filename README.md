# Cloudie: Tropical Cyclone Intensity Estimation from Satellite Imagery

[![CI](https://github.com/AnleaMJ/cloudie/actions/workflows/ci.yml/badge.svg)](https://github.com/AnleaMJ/cloudie/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-orange)](https://huggingface.co/spaces/Anlea/cloudie)
[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue)](https://huggingface.co/Anlea/cloudie-weights)

## Abstract

Cloudie is a rigorous deep learning and uncertainty calibration framework for tropical cyclone intensity estimation from infrared satellite imagery.
It couples a 4-frame spatiotemporal ConvLSTM with physics-informed classical baselines under strictly leakage-free spatial, temporal, and holdout-basin protocols.
All predictions feature Monte Carlo dropout coupled with post-hoc conformal scaling, enforcing zero synthetic numbers and strict scientific honesty guardrails.

---

## Architecture

![Cloudie Architecture Pipeline](docs/assets/architecture_pipeline.svg)

Cloudie partitions estimation into two complementary pipelines:
- **Branch A (Physics Features + Classical Baselines):** Extracts radial brightness-temperature profiles, cold-pixel fractions, warm-eye geometry, simplified deviation-angle variance (DAV), GLCM texture, and Haar wavelets. Fits leakage-free Ridge, Random Forest, and XGBoost regressors and classifiers on training storms only.
- **Branch B (Deep Spatiotemporal ConvLSTM):** Ingests 4-frame sequences ($t-3, t-2, t-1, t$) through recurrent convolutional gates with weighted Smooth L1 wind loss, auxiliary intensity-class classification, Monte Carlo dropout for epistemic dispersion, and post-hoc conformal scaling.

---

<!-- BENCHMARK:START -->
## Benchmark & Baselines

Cloudie enforces strict separation between real experimental results and software smoke tests. In accordance with our scientific honesty guardrails, **no synthetic or smoke run numbers are quoted as results**. Official benchmarks are generated only from reviewed GPU runs via `scripts/render_readme_tables.py` (which strictly asserts `quote_as_result == "yes"` and `is_synthetic == false`).

| Model | Modality / Feature Set | Split Protocol | Validation MAE (kt) | Test MAE (kt) | Status |
|---|---|---|:---:|:---:|---|
| **Train Mean** | Train-set wind speed mean | Storm-wise holdout | — | — | Baseline |
| **Persistence** | Best-track label ($t-1$) | Temporal reference | — | — | Reference (Non-imagery) |
| **Ridge Regression** | Handcrafted IR features | Storm-wise holdout | — | — | Classical |
| **Random Forest** | Handcrafted IR features | Storm-wise holdout | — | — | Classical |
| **XGBoost** | Handcrafted IR features | Storm-wise holdout | — | — | Classical |
| **ResNet-18** (Phase 3) | Single-frame IR ($t$) | Storm-wise holdout | — | — | Deep Learning |
| **ConvLSTM** (Phase 5) | 4-frame IR sequence ($t-3 \dots t$) | Storm-wise holdout | — | — | Deep Spatiotemporal |

> [!NOTE]
> **Baseline Protocol Notes:**
> - **Persistence** uses best-track labels from the preceding time fix ($t-1$), not satellite imagery. Image-only models cannot be directly compared against persistence.
> - **Operational products** (ADT, SATCON, Dvorak) remain unpopulated until those products are officially joined. Published literature RMSE figures from other basins, sensors, and wind averaging periods are not directly comparable.
<!-- BENCHMARK:END -->

---

## 30-Second Reproducibility & Data-Leakage Audit

To rapidly audit the entire codebase against data leakage without downloading external gigabyte datasets, execute the centralized split audit:

```bash
python -m src.verify_splits --config config/config_smoke.yaml
```

This 6-point verification strictly asserts:
1. **Storm-Wise Disjointness:** No storm identifier ever appears across multiple splits (`train`, `val`, `test`).
2. **Chronological Holdout Integrity:** Temporal ordering is strictly preserved without forward-looking leakage.
3. **Holdout-Basin Isolation:** Basin holdouts (e.g. North Indian Ocean vs. North Atlantic) remain completely unmixed.
4. **Deliberate Leakage Refusal:** Confirms `LeakageError` is raised when random frame splitting is attempted.
5. **Normalization Isolation:** Finite-pixel z-score parameters are computed strictly on the training partition.
6. **Sequence Contiguity:** 4-frame temporal windows never span split boundaries or unrelated storms.

---

## Scientific Honesty Note

- **No trained weights are shipped in the git repository.** There are no results to quote until verified GPU training runs are complete.
- `src/make_smoke_data.py` and `python -m cloudie.make_smoke_data` write synthetic images. They are not cyclones. A fit on them only proves the software runs.
- **Not an operational forecast or warning system.** Best-track labels are subject to post-analysis uncertainty. Agreement with a best track is not ground truth.
- ADT, SATCON, and Dvorak columns in the baseline table remain empty until those products are joined. Literature RMSE figures are not this repo's scores.
- Survey and paper numbers (P1 to P8 in `docs/PROJECT_DOCUMENTATION.md`) come from other basins, sensors, splits, and wind averages, so they are not a target this repo has met or a fair comparison.
- NASA labels are HURDAT2 1-minute winds. IMD classes use 3-minute winds. The 17/34/48/64/91 kt bins are this project's own grouping, not official IMD classes, and not an IMD warning.
- `dav_simplified_r*` and `haar_energy_*` are simplified versions of published DAV and wavelet methods, not exact reproductions.
- Monte Carlo dropout returns a mean and a standard deviation representing dropout noise under model epistemic uncertainty. It is not a calibrated prediction interval. Calibrated intervals require validation conformal scaling (`src.calibration`).
- Raw `.pt` weights belong exclusively on Hugging Face Hub (`Anlea/cloudie-weights`), never in the git tree. Do not commit NASA, TCIR, IBTrACS, or INSAT-3D files (see `docs/DATA_SOURCES.md`).

---

## What is Implemented

- **Leak-free splits and quality gates (Phase 0–1):** Storm-wise, chronological, and holdout-basin splitting with `LeakageError` enforcement.
- **Classical baselines (Phase 2):**
  - Brightness-temperature pixel statistics, cold-pixel fractions, radial profiles, warm-eye geometric flags, simplified DAV, GLCM texture, and Haar energy deltas.
  - Feature selection fit on training storms only.
  - Mean, Ridge, k-Nearest Neighbors, Support Vector Regressors/Classifiers, Random Forest, and XGBoost.
  - Stratified group cross-validation on training storms, SMOTE applied strictly inside training folds.
  - Bootstrap 95% confidence intervals resampling storm IDs.
- **Deep spatiotemporal modeling (Phase 3 & 5):**
  - ResNet-18 and 4-frame ConvLSTM (`model.name: convlstm`) architectures.
  - Weighted Smooth L1 loss on wind speed with auxiliary multi-class classification.
  - Centralized training CLI: `python -m src.train --model [simplecnn|resnet18|convlstm]`.
- **Uncertainty Calibration:**
  - Raw Monte Carlo dropout dispersion diagnostics.
  - Validation-fit scaled conformal prediction intervals.
  - MACE, ENCE, 90% empirical coverage & interval width, and Spearman rank correlation between predicted uncertainty and absolute error.
- **Deployment & Spaces (Phase 4):**
  - Gradio Hugging Face Space (`space/app.py`) for cloud demonstration.
  - Streamlit dashboard (`cloudie/app.py`) for interactive inspection.
  - Lightweight, torch-free FastAPI backend (`app.py`) for Vercel deployment.

---

## Setup & Quickstart

```bash
# 1. Clone repository and set up virtual environment
git clone https://github.com/AnleaMJ/cloudie.git
cd cloudie
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e ".[train,test]"
# Alternatively: pip install -r requirements-train.txt

# 3. Run unit tests
pytest -q

# 4. Run 6-point leakage audit
python -m src.verify_splits --config config/config_smoke.yaml

# 5. Run CPU smoke training check
python -m src.train --config config/config_smoke.yaml --dataset smoke --model convlstm --max-steps 2 --device cpu
```

---

## Repository Layout

```
src/                     Canonical modules: architectures, features, baselines, calibration, evaluation
cloudie/                 Re-exports, training wrappers, prediction, and local Streamlit app
config/                  Configuration YAMLs (config.yaml, config_smoke.yaml, config_incyde.yaml)
space/                   Gradio Space bundle for Hugging Face (cpu-basic)
scripts/                 Benchmarking, plotting, calibration diagnostics, and table rendering
tests/                   Pytest suite covering leakage, training, inference, and calibration
docs/                    Architecture diagrams, handoff specs, and Hugging Face model cards
```
