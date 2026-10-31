# Training

`src/train.py` fits a wind-speed model in knots. Every hyperparameter, path,
and seed comes from the YAML. The CLI can select the dataset, the model, a
step cap, and the device. Smooth L1 is the wind loss. The optimizer is
AdamW. Early stopping watches validation MAE. Dropout stays in the network
so Monte Carlo dropout still runs at inference.

NASA and TCIR are separate runs. A batch never mixes their resolutions.
NASA phase 1 keeps only 366×366 frames (`orig_size`); other frames are
excluded and the count is logged. The official NASA side stays in
`source_split` and is not the split. Storms are assigned with
`src.splits.assign_storms` on the train and test sides together.
TCIR is IR1 at native 201×201 via `src/tcir.py`. It has no official split.
Normalization is fit on train storms only.

Each run writes `artifacts/runs/<run_id>/`: the checkpoint (config hash,
seed, dataset, split storm ids, `is_synthetic`, `wind_averaging`), the
resolved config, the split manifest, a loss curve CSV and PNG, and
`metrics.json` with validation and test MAE, RMSE, and bias from
`src.evaluate`. That directory is gitignored. Do not commit it.

Smoke data is synthetic. Those runs set `is_synthetic` true and
`"result": false`. They are a software check. There are no results to quote.
No GPU run has been approved.

## CPU dry run

Synthetic images only. This does not download NASA or TCIR, and it is not a result.

```bash
python -m src.make_smoke_data
python -m src.train --config config/config_smoke.yaml --dataset smoke --model simplecnn --max-steps 2 --device cpu
```

## GPU run

This needs misty wind's approval before anyone starts it. Set the env var
only after that approval. No results exist yet. Do not quote a number from
this command until a real run has been reviewed.

```bash
CLOUDIE_ALLOW_GPU=1 python -m src.train --config config/config.yaml --dataset nasa --model resnet18 --device cuda
```

Without `CLOUDIE_ALLOW_GPU=1` the process refuses `--device cuda`.
`paths.nasa_root` must already point at a local NASA TropicalCyclone
tree. This entrypoint does not download imagery. A TCIR run is the same
command with `--dataset tcir` and `paths.tcir_files` set in the YAML. Do
not point one run at both datasets.
