---
title: Cloudie
emoji: "🌀"
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.11"
hardware: cpu-basic
suggested_hardware: cpu-basic
short_description: Cyclone IR demo. Not a forecast.
pinned: false
fullWidth: false
---

# Cloudie

Academic demo that reads one infrared frame and, when a checkpoint is allowed
to answer, reports a wind speed in knots and this project's own intensity
grouping. The page shows the same permanent banner as the web demo
(`NO_TRAINED_MODEL_BANNER` in `src/inference.py`). It is not a forecast or a
warning system. No trained weights are in this bundle.

The greyscale cyclone-IR gate and `RunCheckpoint` live in `src/inference.py`.
A colour frame is refused with that gate's message and no wind. Confidence
and the wind interval stay labelled uncalibrated. The interval is a dash.
Nearest Indian coastal states appear only after latitude and longitude, and
they are labelled with the config note that this is not a landfall forecast.
The bundled `synthetic_ir.npy` sample is synthetic. It is not a satellite image.

A wind number appears in two cases:

- The checkpoint is the synthetic software check. The page says so.
- `paths.runs/<run_id>/` contains the checkpoint and `metrics.json` from
  `src.train`, or `CLOUDIE_HF_MODEL_REPO` / `paths.hf_model_repo` names a
  model repo with that same layout, and the record passes the held-out
  storm check in `src.inference.ConfidenceScore`. Until that record exists,
  the page shows no wind.

`CLOUDIE_RUN_ID` selects the run directory. `CLOUDIE_CHECKPOINT` may point
at one local file for the operator. The form does not ask the visitor for a path.

## Bundle

The GitHub repository keeps `src/` as the canonical copy. `cloudie/`
re-exports those modules. The Space imports `src` only. From a checkout:

```bash
python space/sync_bundle.py --out /tmp/cloudie-space
```

The directory passed to `--out` is what gets uploaded. It contains `app.py`,
`requirements.txt`, this `README.md`, `src/`, `config/`, and
`cloudie/samples/synthetic_ir.npy`. The script does not copy `data/`,
`artifacts/`, or checkpoints.

CPU torch is pinned in `requirements.txt` through the PyTorch CPU index.
Gradio is selected by `sdk_version` above and is not pinned again in
`requirements.txt`. Hardware is `cpu-basic`. Do not select a GPU flavor.

## Deploy

Do this after the GitHub change is merged, with a token that can create and
write the Space. This checkout does not push the Space.

1. Create a fine-grained token at <https://huggingface.co/settings/tokens>
   with write access limited to the Space `Anlea/cloudie`. To create the
   repo, the token also needs permission to create a Space under the user
   `Anlea` (contributor permission, or a classic token whose role is
   `write`). A read-only token cannot upload the bundle.
2. Build the bundle from the merged commit: `python space/sync_bundle.py --out /tmp/cloudie-space`.
3. Log in: `hf auth login`.
4. Create the Space on free CPU hardware:

```bash
hf repos create Anlea/cloudie --type space --space-sdk gradio --flavor cpu-basic --public --exist-ok
```

5. Upload the bundle, not the GitHub root:

```bash
hf upload Anlea/cloudie /tmp/cloudie-space . --repo-type space --commit-message "Cloudie CPU Space"
```

6. Leave the hardware on `cpu-basic`. If a later settings change is required:
   `hf spaces settings Anlea/cloudie --hardware cpu-basic`.

The running Space reads `CLOUDIE_HF_MODEL_REPO` from a Space variable when a
real checkpoint exists, and a Space secret named `HF_TOKEN` with read access
to that model repo when the repo is private. Leave both unset until a
held-out real-data record exists. Creating a Gradio Space can require a Hub
plan even though `cpu-basic` has no hourly charge. This app does not request
a GPU.
