# Training

`src/train.py` fits a wind-speed model in knots. Every hyperparameter, path,
and seed comes from the YAML. The CLI can select the dataset, the model, a
step cap, the device, a checkpoint to resume, and a checkpoint to fine-tune
from. Smooth L1 is the wind loss. The optimizer is AdamW. Early stopping
watches validation MAE. Dropout stays in the network so Monte Carlo dropout
still runs at inference.

The device order for `--device auto` (the default) is `xpu`, then `cuda`,
then `cpu`. `xpu` is native `torch.xpu` from the official PyTorch wheel.
Do not install `intel_extension_for_pytorch`. That package is archived and
is not used here. `cudnn.benchmark` is not turned on.

`CLOUDIE_ALLOW_GPU=1` gates XPU and CUDA the same way. Without it, an
explicit `--device xpu` or `--device cuda` stops. Auto that would have
picked an accelerator falls back to CPU and logs why.

Precision follows the device. XPU uses
`torch.autocast("xpu", dtype=torch.bfloat16)` when
`torch.xpu.is_bf16_supported(including_emulation=False)` says native bf16
is available, and fp32 otherwise. CUDA uses fp16 autocast with GradScaler.
GradScaler is not used on XPU or CPU. CPU stays fp32.

Before a non-CPU run, one fixed seeded batch is forwarded on that device
and on CPU. The max absolute gap in the wind output and the loss is stored
on the run record. Above `train.device_parity_tol` (a fraction of the CPU
magnitude, default 0.05, with a floor of 1) the run logs a warning and sets
`device_parity_ok` false. It does not stop, and it does not convert units.

NASA and TCIR are separate runs. A batch never mixes their resolutions.
NASA phase 1 keeps only 366×366 frames (`orig_size`); other frames are
excluded and the count is logged. The official NASA side stays in
`source_split` and is not the split. Storms are assigned with
`src.splits.assign_storms` on the train and test sides together.
TCIR is IR1 at native 201×201 via `src/tcir.py`. It has no official split.

Normalization is fit per dataset on that dataset's train storms only,
using finite pixels only. NASA pixels are 8-bit counts. TCIR pixels are
brightness temperature in kelvin. The stats are stored on the checkpoint
and the run record under `norm_by_dataset`, keyed by the dataset name.
Inference (`src.inference`, `cloudie.predict`) applies that dataset's
stats and refuses a checkpoint that mixes datasets. A fine-tune refits
stats on the new dataset. It does not reuse the parent's mean and std.

`train.grad_accum_steps` defaults to 1. `train.num_workers` defaults to 2,
and the loader sets `persistent_workers` when that number is above zero.
A checkpoint is written at the end of every epoch (`last.pt` in the run
directory) so a sleep can resume with `--resume`.

Each run writes `artifacts/runs/<run_id>/`: the checkpoint (config hash,
seed, git commit, dataset, per-dataset norm stats, split storm ids,
`is_synthetic`, `wind_averaging`, `wind_averaging_s`, the per-split sets of
both, device, precision, peak memory), the resolved config, the split
manifest, a loss curve CSV and PNG, and `metrics.json` with validation and
test MAE, RMSE, and bias from `src.evaluate`. Peak memory is
`torch.xpu.max_memory_allocated` or `torch.cuda.max_memory_allocated`, and
null on CPU. XPU peak memory is the reason to stay on the torch 2.15 line.
That directory is gitignored. Do not commit it.

`wind_averaging_s` is the label's duration: 60, 180, or 600 seconds for a
1-minute, 3-minute, or 10-minute label. It stays on every row. Synthetic
labels store null. The training target is `labels.label_column`
(`wind_kt_3min` in the YAML). Raw knots stay on `wind_speed_kt` and
`wind_kt`. The run record stores `label_column` and `wind_conversion`.

Validation metrics are written twice. `val` scores the training target.
`val_raw` scores the raw label. For a converted 1-minute row the raw
prediction is the 3-minute prediction divided by that row's
`wind_kt_scale` (the configured factor, 0.93 by default). INCYDE scale is
1, so the two validation blocks match. Test metrics use the same pair
when the test split is scored.

`--init-from` from a parent that also trained on `wind_kt_3min` does not
set `wind_averaging_mismatch`, even when the raw periods differ (1-minute
pretrain, 3-minute INCYDE). The flag is still set when raw labels are
forced (`labels.label_column` `wind_speed_kt` or `wind_kt`) and the raw
periods differ. That warning says that no further conversion is applied.

Smoke data is synthetic. Those runs set `is_synthetic` true and
`"result": false`. They are a software check. There are no results to quote.

## Intel Arc / XPU

Install on Windows first, against the official XPU wheel index. WSL2 is a
fallback for a machine where the native Windows wheel cannot see the GPU.
Use torch 2.15 from that index. Install a current Intel GPU driver whose
WHQL version is at least 32.0.101.9033.

```bash
pip3 install -r requirements-xpu.txt --index-url https://download.pytorch.org/whl/xpu
```

`requirements-xpu.txt` pins `torch==2.15.*` plus torchvision and torchaudio
from that same index. The GitHub Actions jobs keep the CPU wheels. They do
not install this file.

On WSL2, after the Windows driver is installed, the guest still needs the
Level Zero and OpenCL userspace packages, and the GPU node has to be
present:

```bash
sudo apt-get install -y libze1 libze-intel-gpu1 intel-opencl-icd
ls -l /dev/dxg
```

`PYTORCH_ENABLE_XPU_FALLBACK` is left unset. Do not turn it on by default.
To see which ops fell back to CPU, run the benchmark once with the debug
flag. That log is diagnostic. It is not a result.

```bash
PYTORCH_DEBUG_XPU_FALLBACK=1 CLOUDIE_ALLOW_GPU=1 python scripts/bench_device.py --device xpu --steps 200 --batch-size 4
```

`scripts/bench_device.py` times `simplecnn` and `resnet18` on synthetic
tensors at the native sizes (366 for NASA, 201 for TCIR). It prints JSON
and writes it under `artifacts/bench/`, which is gitignored. The payload
is `is_synthetic: true` and `result: false`. It honors
`CLOUDIE_ALLOW_GPU`. A CPU-only check is the same command with
`--device cpu`.

Optional pre-cache, so loader workers read fp16 `.npy` files instead of
the source frames. Training uses the cache when
`paths.frame_cache/<dataset>/index.csv` exists and otherwise reads the
source as usual. This does not download imagery.

```bash
python -m src.cache_frames --config config/config.yaml --dataset nasa
python -m src.cache_frames --config config/config.yaml --dataset tcir
```

## CPU dry run

Synthetic images only. This does not download NASA or TCIR, and it is not a result.

```bash
python -m src.make_smoke_data
python -m src.train --config config/config_smoke.yaml --dataset smoke --model simplecnn --max-steps 2 --device cpu
```

## Pretrain

TCIR and NASA stay in separate runs. Both train on `wind_kt_3min`.
Pretrain both before any fine-tune.
`paths.nasa_root` must already point at a local NASA TropicalCyclone tree,
and `paths.tcir_files` must already list local TCIR HDF5 files. These
commands do not download imagery. No speed or accuracy number from them
is a result until a real run has been reviewed.

```bash
CLOUDIE_ALLOW_GPU=1 python -m src.train --config config/config.yaml --dataset tcir --model resnet18 --device auto
CLOUDIE_ALLOW_GPU=1 python -m src.train --config config/config.yaml --dataset nasa --model resnet18 --device auto
```

`--device cuda` is the same gate, for a machine that has CUDA instead of
Arc. Without `CLOUDIE_ALLOW_GPU=1` the process refuses both.

A run that sleeps can continue from the last finished epoch:

```bash
python -m src.train --config config/config.yaml --resume artifacts/runs/<run_id>
```

## Label standard

The regression target is an IMD 3-minute wind, `wind_kt_3min`.

INCYDE labels are already IMD 3-minute sustained winds. Those rows pass
through: `wind_kt_3min` equals `wind_kt`, `wind_averaging` is `3min`, and
`wind_averaging_s` is 180. `wind_conversion` is empty.

NASA and TCIR labels are 1-minute winds (`wind_averaging_s` 60). They are
multiplied by `labels.wind_1min_to_3min` (0.93). The row's
`wind_conversion` text is:

`0.93, derived from WMO/TD-1555 Tables 1.1 and 1.2, at-sea exposure (not a WMO constant)`

That factor is derived from the at-sea gust factors in those tables. It is
not a WMO constant. It is meant for open water more than
`labels.open_water_km` (20) kilometres offshore.

NASA has no land column. TCIR has latitude and longitude and no distance
to land, and this repo has no coastline, so those rows are not treated as
open water. `near_land` is `unknown` and `conversion_exposure_flag` is
`unknown`. The 0.93 factor is still applied. A row that already carries
`distance_to_land_km`, `near_land`, or `landfall` is not converted in
silence. `labels.near_land_policy` `flag` (the default) multiplies by 0.93
and sets `conversion_exposure_flag` true. `raw` leaves that row at the
1-minute label and sets the same flag.

About 91% of INCYDE original-frame labels are interpolated between IMD
fixes, not observations. Only two storms are SuCS (KYAAR and AMPHAN), so
a SuCS score on this set is indicative.

The auxiliary class head still uses the project's 17/34/48/64/91 kt bins,
now read from `wind_kt_3min`, so a pretrain checkpoint can initialize
every tensor. The INCYDE column `imd_class` is the IMD scale on that same
3-minute wind: D ≤27, DD ≤33, CS ≤47, SCS ≤63, VSCS ≤89, ESCS ≤119,
SuCS ≥120. Below 17 kt is `<17`.

## INCYDE

`python -m src.train --dataset incyde` reads a local extract of
`Cropped_cyclone_data.zip` (Zenodo 8015544, MD5
`ed5dac254c689e5a327dee7580324347`). Nothing in training or CI downloads
it. `paths.incyde_root` is the extract. The optional manifest is
`incyde.manifest_csv`. Without it, the loader reads
`Cropped_cyclone_labels.csv` inside the root.

Images follow `DDMONYYYY_HHMM_<obj>[_aug].jpg`. The clock in the name is
IST. The manifest stores `time_ist` and `time_utc` (IST minus 5 hours 30
minutes). `Sequence` is the storm key and is joined to
`incyde/audit/storm_split.csv`. `obj` is `object_index`. It is not a
storm id. `on_fix` is omitted: the audit CSVs do not say which frames sit
on an IMD best-track time. There is no per-frame landfall flag either.
`basin` value `LAND` is an IMD basin code, not a landfall. `near_land` is
`unknown`.

The default split is Bean's grouped split in `storm_split.csv`: 60 train,
14 val, 17 test storms. Concurrent storms share a `cooccurrence_group`
and a split. `split.strategy: storm_cv` builds folds from that CSV with
`incyde.cv_seed` (42), over train and val storms only, and keeps each
`cooccurrence_group` in one fold. Test storms are not in the folds.
`incyde.cv_val_fold` (0) is the validation fold. `incyde.cv_folds_csv`
(columns `storm_id`, `fold`) replaces that shuffle and is checked the
same way: no test storm, no group across folds.

`split.strategy: year_holdout_2019` puts 2019 in test, 2021 in val, and
the other years in train. The year is the `yyyy` in
`INCYDE-Sxx_IMD-yyyy-nnn`, not the filename date, so one storm cannot
land in two years. This split is off by default. It is indicative only:
the years and the images are not those of Agrawal et al. 2024
(arXiv 2410.08218). Landfall frames are kept, because they are not
flagged in the audit outputs. `src.train` will not score INCYDE test
storms, including this holdout and the default test storms, unless
`--allow-final-test` is set. Validation still runs, so early stopping
has a score.

`incyde/audit/exclusions.csv` drops `default_exclude=true` rows. Those
are the Sequence 42 bad-date frames (`wrong_date_imd_daymonth_swap`) and
their copies. Other `qc_flag` values stay. `incyde.drop_qc_flags` names
extra flags to drop.

Augmentation from the zip: only `rot_180` copies are kept. They are
`is_synthetic=true`, they follow the source storm, and they are removed
from val and test. `flip_h` and `flip_v` are dropped because a mirror
reverses cyclone spin. `rot_90` and `rot_270` are dropped because they
add black bars. Training then applies its own rotation on INCYDE train
rows only, with a circular mask so corner fill does not enter the loss.
There is no flip in that augmentation. The mask leaves about a fifth of
the square as missing, under `data.max_missing_fraction` (0.30).

Normalization is per dataset, on INCYDE train storms, original frames
only, finite pixels only, stored under `norm_by_dataset` key `incyde`.
`rot_180` copies are not in that fit. A fine-tune refits on the INCYDE
train originals and does not reuse the parent's mean and std.

`config/config.yaml` leaves `data.image_size` at `native` for NASA and TCIR.
Set a square `data.image_size` for INCYDE when the extract is not one size.
The loader then resizes onto that grid.

```bash
python -m src.train --config config/config.yaml --dataset incyde --model simplecnn --max-steps 2 --device cpu
```

## Fine-tune

`--init-from` loads model weights from a prior checkpoint and does not load
the optimizer. Normalization is fit again on the new dataset's train
storms. The run record stores the parent `run_id` and the checkpoint
sha256. An architecture mismatch is refused. The class head stays the
project's five bins so a TCIR or NASA checkpoint matches an INCYDE run.

Both sides on `wind_kt_3min` do not set `wind_averaging_mismatch`. Forcing
the raw column does, when the raw periods differ.

```bash
CLOUDIE_ALLOW_GPU=1 python -m src.train --config config/config.yaml --dataset incyde --model resnet18 --device auto --init-from artifacts/runs/<tcir-run-id>/checkpoint.pt
```

The checkpoint file name is `paths.checkpoint_name` in the YAML. Add
`--allow-final-test` only when the INCYDE test storms should be scored.
