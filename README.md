# Cloudie

Cloudie estimates tropical-cyclone maximum sustained wind speed, in knots, from single-channel infrared satellite imagery. Regression is the primary task. An optional 5-class label, defined in config, is there for stratified metrics. Size labels (R34, radius of maximum wind) stay out of scope until they exist and are kept out of the inputs.

Repository: https://github.com/AnleaMJ/cloudie

This revision is Phase 0–1 plus Phase 3. Phase 0–1 is the manifest schema and quality gates, storm-safe splits, image preprocessing, regression metrics, a synthetic smoke-data generator, YAML configs, and model interfaces (`SimpleCNN`, a ResNet-18 regressor, and Monte Carlo dropout). Phase 3 is `cloudie/train.py` and `cloudie/evaluate.py`: early stopping from `train.patience`, a checkpoint under `artifacts/`, MAE / RMSE / bias, per-class error, a residual plot, and a comparison with the train-set mean. Classical feature baselines and the Streamlit demo are still later work. See `docs/AGENT_HANDOFF.md` and `docs/PROJECT_DOCUMENTATION.md`.

## Limitations

- No trained weights are shipped. There are no skill scores to quote. Figures from papers are not results of this repository.
- `python -m cloudie.make_smoke_data` writes synthetic images. They are not cyclones. A fit on them only shows that the software runs.
- Cloudie does not issue forecasts or warnings. Best-track wind is a noisy label. Agreement with a best track is not ground truth.
- Monte Carlo dropout returns a mean and a standard deviation. That standard deviation is dropout noise. It is not a calibrated prediction interval.
- ImageNet weights, if a later run sets `pretrained: true`, are a regularizer. They are not a physical prior for infrared imagery. That flag downloads weights and stays off in CI.
- These are known difficulties for this kind of model, and none of them has been measured here: weak skill on rare intense storms, shear and exposed centers, visible imagery that exists only in daylight, shift across satellites, the opposite rotation sense of Southern Hemisphere storms, mixed wind-averaging periods, and leakage from near-duplicate frames of one storm.
- NASA, TCIR, IBTrACS, INSAT-3D, and MOSDAC files stay out of git. See the data-source notes below and `docs/DATA_SOURCES.md`.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m cloudie.make_smoke_data --out data/smoke
```

`make test` and `make smoke` run the same two commands.

`requirements-train.txt` adds PyTorch, torchvision, matplotlib, and h5py. Continuous integration installs `requirements.txt` only. Training tests and the SimpleCNN forward-pass test skip when torch is absent.

A smoke training check needs the train extras. It does not produce a score to quote:

```bash
pip install -r requirements-train.txt
make train-smoke
```

`notebooks/smoke_train.ipynb` runs the same check and states that the images are synthetic.

Smoke images and manifests under `data/` are gitignored, along with `artifacts/`, `reports/` (except the placeholder files), `*.pt`, `*.ckpt`, and `*.h5`.

## Tests

```bash
pytest
```

The suite builds smoke manifests in a temporary directory. It needs no network and no real imagery. After `pytest`, you can also write a local smoke set:

```bash
python -m cloudie.make_smoke_data --out data/smoke
```

That command prints a reminder that the images are synthetic. Do not report scores from `data/smoke`.

## Splits and quality gates

Allowed strategies, set in `config/config.yaml`: `storm_wise` (default), `chronological`, and `holdout_basin`. A random frame split raises `LeakageError`. A storm that touches a holdout year or a holdout basin stays entirely on the holdout side. Normalization statistics fit on the train split only. Previous-frame links that point outside the row's split are cleared.

A manifest needs `sample_id`, `image_path`, `storm_id`, and `wind_speed_kt`. Sample ids are unique. Wind must lie in `[0, 200]` knots. One `wind_averaging` value is allowed per manifest. An image whose non-finite fraction is above `0.30` is rejected. Values in `image_path` are relative to the manifest file.

These columns are labels, and they are forbidden as inputs to the primary model: `wind_speed_kt`, `r34_nmi`, `rmw_nmi`, `imd5_class`, `intensity_bin`.

Project class names group India Meteorological Department categories for metrics:

| Wind (kt) | Project class |
| --- | --- |
| below 17 | none |
| [17, 34) | `D_DD` |
| [34, 48) | `CS` |
| [48, 64) | `SCS` |
| [64, 91) | `VSCS` |
| 91 and above | `ESCS_SUCS` |

The July 2024 IMD 3-minute maximum-sustained-wind sheet starts Extremely Severe at 90 knots and Super Cyclonic Storm at 120 knots, so 90 knots is `VSCS` in this config and Extremely Severe on that sheet. These names are a project metric. They are not a warning product, and they do not transfer onto 1-minute winds unless `wind_averaging` records the period.

Brightness temperature is computed only when the caller supplies gain and offset. The code does not embed a sensor calibration table.

## Data sources

Nothing in this table is vendored in the repository. Licensing notes are the provider terms as published; confirm them again before you copy pixels.

| Source | Role in Cloudie | Access and license |
| --- | --- | --- |
| NASA Tropical Storm Wind Speed via TorchGeo `TropicalCyclone` | First real labeled infrared set. Atlantic and East Pacific storms, 2000–2019, native size about 366 px, wind in knots. | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) on the [Source Cooperative](https://data.source.coop/nasa/tropical-storm-competition/README.md) release. TorchGeo downloads with `azcopy`. Cite the paper TorchGeo names, [doi:10.1109/JSTARS.2020.3011907](https://doi.org/10.1109/JSTARS.2020.3011907), and the dataset [doi:10.34911/rdnt.xs53up](https://doi.org/10.34911/rdnt.xs53up). |
| Hugging Face [`torchgeo/tropical_cyclone`](https://huggingface.co/datasets/torchgeo/tropical_cyclone) | Mirror of the competition archive. | Same underlying CC-BY-4.0 data. The Hub dataset viewer currently fails to build (`DatasetGenerationError`). Do not make CI depend on `datasets.load_dataset`. Re-split by storm. The competition frame split is not the split this repo uses. |
| [TCIR](https://github.com/BoyoChen/TCIR) | Later multichannel option. 201×201 frames, channels IR1, water vapour, visible, and passive microwave. | The GitHub repository is MIT. Pixels come from GridSat and CMORPH, whose terms are separate from that MIT license. Southern Hemisphere storms rotate the other way; do not silently flip them. Keep the HDF5 files out of git. |
| IBTrACS v4 | Labels only: position, wind, pressure, and wind radii when an agency provided them. | [doi:10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16). NCEI distributes it for open research. It is not a model input. |
| INSAT-3D / 3DR via [MOSDAC](https://www.mosdac.gov.in/) | Later North Indian Ocean adapter. | Registration is required. Redistribution is not assumed. Imagery stays out of git. |
| TC PRIMED | Later infrared, microwave, and environment fields. | [doi:10.25921/dmy1-0595](https://doi.org/10.25921/dmy1-0595). Not required for Phase 0–1. |

`center_source` may be `dataset_precentered` or a center fix that existed at image time. This phase does not crop with a best-track center revised after the image time.

## Layout

```
cloudie/                 schema, splits, preprocessing, loader, metrics, smoke data, models, train, evaluate
config/                  config.yaml and config_smoke.yaml
tests/                   pytest, using temporary smoke data
docs/                    data sources, project notes, next-phase handoff
.github/workflows/ci.yml pip install -r requirements.txt && pytest
data/ artifacts/ reports/   gitignored outputs, with .gitkeep placeholders
notebooks/smoke_train.ipynb  synthetic training check; do not report scores
```

This repository does not choose a code license. Third-party data remains under the terms above and is not included.
