---
language:
- en
license: mit
tags:
- cyclone
- tropical-cyclone
- intensity-estimation
- satellite-imagery
- meteorological
- convlstm
- computer-vision
- pytorch
datasets:
- incyde
- tcir
pipeline_tag: image-to-text
---

# Cloudie Cyclone Intensity Weights (`Anlea/cloudie-weights`)

Checkpoint archive and run records for **Cloudie**, an academic tropical-cyclone intensity estimation framework from infrared satellite imagery.

> [!WARNING]
> **Operational Disclaimer & Scientific Honesty**
> - This model is **not an operational warning, forecast, or cyclone advisory system**.
> - Do not use these outputs for life-safety, navigation, or emergency-response decisions.
> - Best-track labels (HURDAT2 / IMD best track) are subject to observational and post-analysis uncertainty; agreement with labels is not ground truth.
> - Monte Carlo dropout variance captures model uncertainty under dropout noise, not physical meteorological forecast bounds. Calibrated intervals require validation conformal scaling.

---

## Model Details

- **Model Type:** 4-Frame Spatiotemporal ConvLSTM / SimpleCNN
- **Primary Task:** Regression of tropical-cyclone maximum sustained wind speed in **knots (IMD 3-minute standard)**
- **Auxiliary Task:** 5 project intensity categories: `D_DD` [17, 34), `CS` [34, 48), `SCS` [48, 64), `VSCS` [64, 91), `ESCS_SUCS` [91+] kt (project grouping; not official IMD warning classes)
- **Input Modality:** Infrared satellite imagery (e.g. INSAT-3D/3DR IR1 from INCYDE or GridSat-B1 IR1 from TCIR), single-channel, normalized with train-storm statistics only, circular-masked.
- **Source Code:** [GitHub: AnleaMJ/cloudie](https://github.com/AnleaMJ/cloudie)

---

## Expected Repository Layout

When mirror runs are published to `Anlea/cloudie-weights`, the repository mirrors the canonical `artifacts/runs/<run_id>/` structure required by `src.inference.RunCheckpoint`:

```
Anlea/cloudie-weights/
├── <run_id>/
│   ├── checkpoint.pt        # Torch model state dict + norm stats + config hash
│   ├── config.yaml          # Resolved training configuration
│   ├── metrics.json         # Evaluated test/val metrics (quote_as_result: yes)
│   ├── run_record.json      # Full provenance: git commit, hardware, seeds
│   ├── loss_curve.png       # Training & validation loss history
│   └── calibration.npz      # Test/val MC draws and targets for conformal evaluation
└── README.md
```

> [!IMPORTANT]
> Raw model weight files (`.pt`) must never be committed to the GitHub repository. Checkpoints belong exclusively on this Hugging Face Hub repository.

---

## Integration with Cloudie Hugging Face Space

The Cloudie Gradio Space ([`space/app.py`](https://github.com/AnleaMJ/cloudie/blob/main/space/app.py)) automatically queries this repository when configured.

To attach this model weights repository to the Space:

1. Open the Space settings on Hugging Face (`Anlea/cloudie` &rarr; Settings &rarr; Variables and secrets).
2. Set the Environment Variable:
   ```env
   CLOUDIE_HF_MODEL_REPO=Anlea/cloudie-weights
   CLOUDIE_RUN_ID=<run_id>
   ```
3. If the model weights repository is private, add a read-scoped secret:
   ```env
   HF_TOKEN=hf_...
   ```
4. `src.inference.RunCheckpoint` validates the remote run record and verifies that it passed held-out storm evaluation before serving predictions.

---

## Python Inference Usage

```python
import os
import torch
from src.config import load_config
from src.inference import RunCheckpoint, predict_images

# Configure target repository
os.environ["CLOUDIE_HF_MODEL_REPO"] = "Anlea/cloudie-weights"
os.environ["CLOUDIE_RUN_ID"] = "<run_id>"

config = load_config("config/config.yaml")

# RunCheckpoint handles Hub download, integrity check, and held-out validation gating
checkpoint_path = RunCheckpoint.resolve_path(config)
print(f"Loaded checkpoint from: {checkpoint_path}")
```

---

## Evaluation & Data Splitting Guardrails

- **Zero Data Leakage:** Storms are partitioned disjointly across train, validation, and test splits (`src.splits.assign_storms`). No storm crosses split boundaries.
- **Normalization Isolation:** Finite-pixel z-score parameters are computed exclusively on training storms.
- **6-Point Leakage Audit:** Audited by `src.verify_splits` across `storm_wise`, `chronological`, and `holdout_basin` splits.
- **Conformal Uncertainty:** 90% confidence intervals are evaluated using post-hoc scaled conformal prediction (`src.calibration.compare_calibration`).

---

## License

Code is licensed under the [MIT License](https://github.com/AnleaMJ/cloudie/blob/main/LICENSE). Trained weights and meteorological satellite data (INCYDE, TCIR, NASA) remain subject to their respective institutional terms of use and licenses.
