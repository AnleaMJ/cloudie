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
1-minute, 3-minute, or 10-minute label. It is not a wind conversion.
Synthetic labels store null. When `--init-from` crosses datasets whose
`wind_averaging` differs, the run logs a warning and sets
`wind_averaging_mismatch` true. No conversion factor is applied.

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

TCIR and NASA stay in separate runs. Pretrain both before any fine-tune.
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

## Fine-tune

`--init-from` loads model weights from a prior checkpoint and does not load
the optimizer. Normalization is fit again on the new dataset's train
storms. The run record stores the parent `run_id` and the checkpoint
sha256. An architecture mismatch is refused.

INSAT / INCYDE is the later fine-tune target. The flag is the same once
that dataset is wired. Until then, a second existing dataset shows the
command. If the parent's `wind_averaging` differs from the new dataset,
the log carries a warning and `wind_averaging_mismatch` is true. There is
no conversion factor.

```bash
CLOUDIE_ALLOW_GPU=1 python -m src.train --config config/config.yaml --dataset nasa --model resnet18 --device auto --init-from artifacts/runs/<tcir-run-id>/checkpoint.pt
```

The checkpoint file name is `paths.checkpoint_name` in the YAML.
