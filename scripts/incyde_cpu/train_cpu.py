"""Train and evaluate the INCYDE intensity model (wind kt + category).

Reuses main's code paths: ``src.train.build_net`` (SimpleCNN backbone with
wind and class heads), ``src.train.EarlyStopping`` (lower validation MAE is
better), ``src.train.load_init_weights`` (``--init-from``),
``src.train.class_sample_weights``, ``src.device`` (auto picks xpu, then cuda,
then cpu; GPU needs ``CLOUDIE_ALLOW_GPU=1``), ``src.schema.wind_to_imd5``
(main's project class bins) and ``src.metrics``.

Differences from ``src.train._fit_run`` (each logged in docs/MODEL_CHANGELOG.md):
the class head is trained jointly (``train.class_loss_weight``), images come
from an in-memory cache built by ``prep_incyde.py`` / ``prep_tcir.py``, and
training-time augmentation is rot_180 plus circle-masked rotations (no flips).

Splits: INCYDE uses ``incyde/audit/storm_split.csv`` (storm ID, with
``cooccurrence_group`` kept together). The test storms are never loaded for
training or model selection. Storm-grouped CV folds are drawn over train+val
by ``cooccurrence_group``; a frame's ``rot_180`` copy always follows its
original. TCIR pretraining uses ``src.splits.assign_storms`` (storm-wise).

Nothing here is an operational forecast.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import confusion_matrix

from src.device import resolve_device
from src.metrics import (
    classification_metrics,
    metrics_by_class,
    regression_metrics,
    storm_bootstrap_metrics,
)
from src.schema import wind_to_imd5
from src.splits import assign_storms
from src.train import EarlyStopping, build_net, class_sample_weights, load_init_weights

REPO = Path(__file__).resolve().parents[2]
NOT_A_FORECAST = (
    "Validation-only research model. Not an operational forecast or warning. "
    "Agreement with a best track is not ground truth."
)


# ----------------------------------------------------------------- helpers
def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def net_config(cfg: dict) -> dict:
    """The keys ``src.train.build_net`` reads."""
    return {
        "model": {"name": cfg["model"]["name"], "pretrained": False, "dropout": cfg["model"]["dropout"]},
        "data": {"channels": cfg["model"]["channels"], "imd5_names": cfg["data"]["imd5_names"]},
    }


IMD7_EDGES = (17, 28, 34, 48, 64, 90, 120)  # IMD scale on 3-minute winds (kt)
IMD7_NAMES = ("D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS")


def imd7_of(winds: np.ndarray) -> np.ndarray:
    """IMD class from a 3-minute wind: D 17-27, DD 28-33, CS 34-47, SCS 48-63,
    VSCS 64-89, ESCS 90-119, SuCS >= 120 kt; below 17 kt is ``below_D``."""
    out = []
    for w in np.asarray(winds, dtype=float):
        name = "below_D"
        for edge, label in zip(IMD7_EDGES, IMD7_NAMES):
            if w >= edge - 0.5:  # labels are whole knots; predictions round to the nearest knot
                name = label
        out.append(name)
    return np.array(out, dtype=object)


def class_of(winds: np.ndarray, cfg: dict) -> list[str | None]:
    edges = tuple(float(e) for e in cfg["data"]["imd5_edges_kt"])
    names = tuple(str(n) for n in cfg["data"]["imd5_names"])
    return [wind_to_imd5(float(w), edges, names) for w in winds]


def circle_mask(side: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    centre = (side - 1) / 2.0
    return (((yy - centre) ** 2 + (xx - centre) ** 2) <= (side / 2.0) ** 2).float()


def rotate_batch(images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angles), torch.sin(angles)
    zeros = torch.zeros_like(cos)
    theta = torch.stack(
        [torch.stack([cos, -sin, zeros], 1), torch.stack([sin, cos, zeros], 1)], 1
    )
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


class Data:
    """Normalised (N, 1, S, S) tensor plus labels for one set of rows."""

    def __init__(self, images, winds, labels, known, storms, alt=None):
        self.images = images
        self.winds = torch.tensor(winds, dtype=torch.float32)
        index = {name: i for i, name in enumerate(known)}
        self.cls = torch.tensor([index.get(l, -1) if l else -1 for l in labels], dtype=torch.long)
        self.labels = labels
        self.storms = np.asarray(storms)
        self.alt = alt  # row -> row of an INCYDE rot_180 copy, or -1

    def __len__(self):
        return int(self.images.shape[0])


def augment(batch: torch.Tensor, alt_batch: torch.Tensor | None, aug: dict, gen: torch.Generator):
    """rot_180 (INCYDE's own copy when present) and circle-masked rotation. No flips."""
    n = batch.shape[0]
    choice = torch.randint(0, 3, (n,), generator=gen)
    out = batch.clone()
    if aug.get("rot_180"):
        pick = choice == 1
        if pick.any():
            rotated = torch.rot90(batch[pick], 2, dims=(-2, -1))
            if alt_batch is not None:
                have = alt_batch[pick]
                ok = torch.isfinite(have.view(have.shape[0], -1)).all(1)
                rotated[ok] = have[ok]
            out[pick] = rotated
    if aug.get("masked_rotation"):
        pick = choice == 2
        if pick.any():
            angles = torch.rand(int(pick.sum()), generator=gen) * 2 * math.pi
            out[pick] = rotate_batch(batch[pick], angles)
    return out


def predict(model, data: Data, device, mask, batch_size=512):
    model.eval()
    winds, logits = [], []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x = data.images[start : start + batch_size].float().to(device)
            if mask is not None:
                x = x * mask.to(device)
            w, lg = model(x)[:2]
            winds.append(w.view(-1).cpu())
            logits.append(lg.cpu())
    return torch.cat(winds).numpy(), torch.cat(logits).numpy()


def fit(model, train: Data, val: Data | None, cfg: dict, device, *, epochs, aug, mask,
        on_epoch=None, fixed_epochs=False, alt_images=None, seed=42):
    tcfg = cfg["train"]
    known = list(cfg["data"]["imd5_names"])
    gen = torch.Generator().manual_seed(seed)
    if tcfg.get("init_wind_bias"):
        # start the wind head at the train-fold mean instead of ~0 kt
        with torch.no_grad():
            model.backbone.head[-1].bias.fill_(float(train.winds.mean()))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sched = None
    if tcfg.get("cosine"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = None
    if tcfg.get("class_weighting"):
        _s, cw_np = class_sample_weights(train.labels, known)
        cw = torch.tensor(cw_np, dtype=torch.float32, device=device)
    clw = float(tcfg["class_loss_weight"])
    stopper = EarlyStopping(int(tcfg["patience"]))
    history = []
    bs = int(tcfg["batch_size"])
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train), generator=gen)
        tot, cnt = 0.0, 0
        for start in range(0, len(train), bs):
            idx = perm[start : start + bs]
            x = train.images[idx].float()
            alt = None
            if alt_images is not None and train.alt is not None:
                rows = train.alt[idx.numpy()]
                alt = torch.full_like(x, float("nan"))
                have = rows >= 0
                if have.any():
                    alt[torch.from_numpy(have)] = alt_images[rows[have]].float()
            if aug.get("rot_180") or aug.get("masked_rotation"):
                x = augment(x, alt, aug, gen)
            if mask is not None:
                x = x * mask
            x = x.to(device)
            y = train.winds[idx].to(device)
            c = train.cls[idx].to(device)
            w, lg = model(x)[:2]
            if tcfg.get("loss", "smooth_l1") == "huber":
                loss = F.huber_loss(w.view(-1), y, delta=float(tcfg.get("huber_delta_kt", 10.0)))
            else:
                loss = F.smooth_l1_loss(w.view(-1), y)
            m = c >= 0
            if clw > 0 and m.any():
                loss = loss + clw * F.cross_entropy(lg[m], c[m], weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        if sched is not None:
            sched.step()
        rec = {"epoch": epoch, "train_loss": tot / cnt, "seconds": round(time.time() - t0, 1)}
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stop = False
        if val is not None:
            pw, _ = predict(model, val, device, mask)
            rec["val_mae"] = float(np.mean(np.abs(pw - val.winds.numpy())))
            stop = stopper.update(rec["val_mae"], state) and not fixed_epochs
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if on_epoch is not None:
            on_epoch(epoch, model, opt, history, stopper)
        if stop:
            break
    if val is not None and not fixed_epochs and stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return model, history, stopper


def evaluate(model, data: Data, cfg, device, mask, train_winds, train_labels, *, bootstrap=True):
    known = tuple(cfg["data"]["imd5_names"])
    pw, logits = predict(model, data, device, mask)
    truth = data.winds.numpy()
    out = {"wind": regression_metrics(truth, pw)}
    if bootstrap:
        out["wind_storm_bootstrap"] = storm_bootstrap_metrics(truth, pw, data.storms, n_resamples=1000, seed=42)
    true_cls = np.array([l if l else "below_17kt" for l in data.labels], dtype=object)
    labelled = np.array([l is not None for l in data.labels])
    out["wind_by_true_class"] = metrics_by_class(truth, pw, true_cls)
    heads = {
        "class_head": np.array([known[i] for i in logits.argmax(1)], dtype=object),
        "binned_wind": np.array([c if c else "below_17kt" for c in class_of(pw, cfg)], dtype=object),
    }
    for name, pred in heads.items():
        t, p = true_cls[labelled], pred[labelled]
        res = classification_metrics(t, p, known)
        res["accuracy"] = float(np.mean(t == p))
        res["confusion_labels"] = list(known)
        res["confusion_matrix"] = confusion_matrix(t, p, labels=list(known)).tolist()
        out[f"category_{name}"] = res
    # IMD 7-class scale from the predicted wind (no separate head)
    t7, p7 = imd7_of(truth), imd7_of(np.round(pw))
    res7 = classification_metrics(t7, p7, IMD7_NAMES)
    res7["accuracy"] = float(np.mean(t7 == p7))
    res7["confusion_labels"] = ["below_D", *IMD7_NAMES]
    res7["confusion_matrix"] = confusion_matrix(t7, p7, labels=["below_D", *IMD7_NAMES]).tolist()
    out["category_imd7_from_wind"] = res7
    out["wind_by_true_imd7"] = metrics_by_class(truth, pw, t7)
    if bootstrap:
        out["wind_by_true_class_bootstrap"] = {
            str(c): storm_bootstrap_metrics(truth[true_cls == c], pw[true_cls == c], data.storms[true_cls == c], n_resamples=1000, seed=42)
            for c in pd.unique(true_cls)
        }
        for c, row in out["wind_by_true_class_bootstrap"].items():
            row["indicative_n_storms_lt_3"] = bool(row["n_storms"] < 3)
    # naive baselines, fit on the training rows only
    mean_pred = np.full_like(truth, float(np.mean(train_winds)))
    tl = [l for l in train_labels if l]
    majority = max(set(tl), key=tl.count)
    t = true_cls[labelled]
    maj = np.array([majority] * len(t), dtype=object)
    out["baseline_train_mean_wind"] = regression_metrics(truth, mean_pred)
    bres = classification_metrics(t, maj, known)
    bres["accuracy"] = float(np.mean(t == maj))
    bres["majority_class"] = majority
    out["baseline_majority_class"] = bres
    tl7 = imd7_of(np.asarray(train_winds))
    vals, counts = np.unique(tl7, return_counts=True)
    maj7 = np.array([vals[int(np.argmax(counts))]] * len(t7), dtype=object)
    b7 = classification_metrics(t7, maj7, IMD7_NAMES)
    b7["accuracy"] = float(np.mean(t7 == maj7))
    b7["majority_class"] = str(vals[int(np.argmax(counts))])
    out["baseline_majority_imd7"] = b7
    out["n_rows"] = int(len(truth))
    out["n_storms"] = int(len(set(data.storms.tolist())))
    return out, pw, logits


# ------------------------------------------------------------------ data
def load_incyde(cache: Path, size: int):
    index = pd.read_csv(cache / "incyde_index.csv", low_memory=False)
    images = np.load(cache / f"incyde_{size}.npy", mmap_mode="r")
    return index, images


def incyde_sets(index, images, rows_train, rows_eval, cfg, known):
    """Normalise with train-fold stats only. Eval sets use originals only."""
    orig = index["is_original"].to_numpy()
    tr_orig = rows_train & orig
    tr_img = np.asarray(images[np.where(tr_orig)[0]], dtype=np.float32)
    mean, std = float(tr_img.mean()), float(tr_img.std())
    def tensor(arr):
        return torch.from_numpy(((np.asarray(arr, dtype=np.float32) - mean) / std)[:, None]).half()
    # rot_180 copies, mapped from their original
    rot = index[(~index["is_original"]) & rows_train]
    pos = {f: i for i, f in enumerate(index.loc[tr_orig, "filename"])}
    alt_rows = np.full(int(tr_orig.sum()), -1, dtype=np.int64)
    alt_imgs = []
    for k, (src_name, row) in enumerate(zip(rot["source_filename"], rot.index)):
        if src_name in pos:
            alt_rows[pos[src_name]] = len(alt_imgs)
            alt_imgs.append(images[row])
    alt = tensor(np.stack(alt_imgs)) if alt_imgs else None
    tr = index[tr_orig]
    train = Data(tensor(tr_img), tr["wind_kt_3min"].to_numpy(), class_of(tr["wind_kt_3min"].to_numpy(), cfg),
                 known, tr["storm_id"].to_numpy(), alt_rows)
    evals = {}
    for name, rows in rows_eval.items():
        sel = rows & orig
        ev = index[sel]
        evals[name] = Data(tensor(images[np.where(sel)[0]]), ev["wind_kt_3min"].to_numpy(),
                           class_of(ev["wind_kt_3min"].to_numpy(), cfg), known, ev["storm_id"].to_numpy())
    return train, evals, alt, {"mean": mean, "std": std, "source_split": "train"}


def build_cv_folds(n_folds: int, seed: int) -> pd.DataFrame:
    """Same rule as ``build_cv_folds`` in PR #35 (``src/incyde.py``): train+val
    storms from ``storm_split.csv``, ``cooccurrence_group``s sorted, shuffled with
    ``numpy.random.default_rng(seed).permutation`` and dealt round-robin."""
    storms = pd.read_csv(REPO / "incyde/audit/storm_split.csv")
    pool = storms.loc[storms["proposed_split"].isin(["train", "val"])].drop_duplicates("storm_id")
    groups = sorted({int(g) for g in pool["cooccurrence_group"]})
    order = np.random.default_rng(int(seed)).permutation(len(groups))
    group_fold = {groups[int(i)]: k % n_folds for k, i in enumerate(order)}
    out = pd.DataFrame({"storm_id": pool["storm_id"].astype(str),
                        "fold": [group_fold[int(g)] for g in pool["cooccurrence_group"]]})
    return out.sort_values(["fold", "storm_id"]).reset_index(drop=True)


def cv_folds(index, n_folds: int, seed: int, folds_csv: Path | None = None):
    """Fold per row. ``folds_csv`` (storm_id, fold), e.g. SWE's, is used verbatim."""
    table = pd.read_csv(folds_csv) if folds_csv else build_cv_folds(n_folds, seed)
    fold_of = dict(zip(table["storm_id"].astype(str), table["fold"].astype(int)))
    folds = index["storm_id"].astype(str).map(fold_of)
    pool = index["proposed_split"].isin(["train", "val"])
    if folds[~pool].notna().any():
        raise ValueError("cv folds include a test storm")
    if folds[pool].isna().any():
        raise ValueError("cv folds omit a train/val storm")
    if (index[pool].assign(f=folds[pool]).groupby("cooccurrence_group")["f"].nunique() > 1).any():
        raise ValueError("a cooccurrence_group spans folds")
    return folds, table


# ------------------------------------------------------------- checkpoint
def payload(model, cfg, stats, extra: dict) -> dict:
    out = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_name": cfg["model"]["name"],
        "dropout": float(cfg["model"]["dropout"]),
        "pretrained": False,
        "in_channels": int(cfg["model"]["channels"]),
        "n_classes": len(cfg["data"]["imd5_names"]),
        "class_names": list(cfg["data"]["imd5_names"]),
        "edges_kt": [float(e) for e in cfg["data"]["imd5_edges_kt"]],
        "norm_mean": stats["mean"],
        "norm_std": stats["std"],
        "norm_source_split": "train",
        "image_size": int(cfg["data"]["image_size"]),
        "circle_mask": bool(cfg["augment"].get("masked_rotation")),
        "is_synthetic": False,
        "present_as_forecast": False,
        "honesty": NOT_A_FORECAST,
        "n_frames": 1,
        "size_head": False,
        "git_commit": git_sha(),
    }
    out.update(extra)
    return out


def hardware() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


# ---------------------------------------------------------------- commands
def cmd_pretrain(args, cfg, device):
    index = pd.read_csv(args.cache / "tcir_index.csv", low_memory=False)
    images = np.load(args.cache / f"tcir_{cfg['data']['image_size']}.npy", mmap_mode="r")
    target = "wind_kt_3min" if args.convert else "wind_kt_raw"
    keep = np.ones(len(index), dtype=bool)
    if args.convert and not cfg["labels"]["pretrain_keep_near_land"]:
        keep &= ~index["near_land"].to_numpy()
    index = index.assign(_row=np.arange(len(index)))
    frame = index[keep].copy()
    frame["split"] = None
    frame = assign_storms(frame, "storm_wise", test_fraction=0.1, val_fraction=0.1, seed=int(cfg["project"]["seed"]))
    known = list(cfg["data"]["imd5_names"])
    sign = -1.0 if cfg.get("tcir_invert", True) else 1.0  # cold = bright, as in INCYDE JPGs
    tr = frame[frame["split"] == "train"]
    tr_img = sign * np.asarray(images[tr["_row"].to_numpy()], dtype=np.float32)
    mean, std = float(tr_img.mean()), float(tr_img.std())
    def make(sub, arr=None):
        arr = sign * np.asarray(images[sub["_row"].to_numpy()], dtype=np.float32) if arr is None else arr
        t = torch.from_numpy(((arr - mean) / std)[:, None]).half()
        w = sub[target].to_numpy()
        return Data(t, w, class_of(w, cfg), known, sub["storm_id"].to_numpy())
    train = make(tr, tr_img)
    del tr_img
    val = make(frame[frame["split"] == "val"])
    model = build_net(net_config(cfg))
    mask = circle_mask(int(cfg["data"]["image_size"])) if cfg["augment"].get("masked_rotation") else None
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stats = {"mean": mean, "std": std}
    meta = {"dataset": "tcir", "pixel_sign": sign, "label_target": target,
            "wind_averaging": "3min_derived" if args.convert else "1min",
            "wind_averaging_s": 180 if args.convert else 60,
            "conversion_factor": cfg["labels"]["one_to_three_min_factor"] if args.convert else None,
            "run_id": out.name}
    def on_epoch(epoch, model, opt, history, stopper):
        torch.save(payload(model, cfg, stats, {**meta, "epoch": epoch, "history": history}), out / f"epoch_{epoch:02d}.pt")
    t0 = time.time()
    model, history, stopper = fit(model, train, val, cfg, device, epochs=int(args.epochs or cfg["train"]["epochs"]),
                                  aug=cfg["augment"], mask=mask, on_epoch=on_epoch, seed=int(cfg["project"]["seed"]))
    wall = time.time() - t0
    torch.save(payload(model, cfg, stats, {**meta, "history": history, "best_val_mae": stopper.best}), out / "checkpoint.pt")
    ev, pw, _ = evaluate(model, val, cfg, device, mask, train.winds.numpy(), train.labels)
    # the same predictions scored in the other label space
    raw = frame[frame["split"] == "val"]["wind_kt_raw"].to_numpy()
    conv = frame[frame["split"] == "val"]["wind_kt_3min"].to_numpy()
    factor = float(cfg["labels"]["one_to_three_min_factor"])
    ev["cross_label_space"] = {
        "vs_raw_1min": regression_metrics(raw, pw if not args.convert else pw / factor),
        "vs_converted_3min": regression_metrics(conv, pw if args.convert else pw * factor),
        "note": "Predictions mapped into each label space with the same factor before scoring.",
    }
    manifest = {
        "kind": "pretrain", "git_commit": git_sha(), "config": cfg, "seed": int(cfg["project"]["seed"]),
        "wall_time_s": round(wall, 1), "hardware": hardware(), "device": str(device),
        "rows": {s: int((frame["split"] == s).sum()) for s in ("train", "val", "test")},
        "storms": {s: int(frame.loc[frame["split"] == s, "storm_id"].nunique()) for s in ("train", "val", "test")},
        "near_land_rows_dropped": int((~keep).sum()), "data_sha256": json.loads((args.cache / "tcir_prep.json").read_text()),
        "history": history, **meta,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(ev, indent=2, default=str))
    print(json.dumps({"val_wind": ev["wind"], "cross": ev["cross_label_space"]}, indent=2))


def build_model(cfg, init_from):
    model = build_net(net_config(cfg))
    parent = None
    if init_from:
        run_id, sha, _ = load_init_weights(model, init_from, net_config(cfg))
        parent = {"path": str(init_from), "sha256": sha, "run_id": run_id}
    return model, parent


def cmd_finetune(args, cfg, device):
    size = int(cfg["data"]["image_size"])
    index, images = load_incyde(args.cache, size)
    known = list(cfg["data"]["imd5_names"])
    mask = circle_mask(size) if cfg["augment"].get("masked_rotation") else None
    seed = int(cfg["project"]["seed"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    split = index["proposed_split"]
    results = {}
    t_all = time.time()
    # 1) fixed split: train -> val (early stopping on val MAE, main's rule)
    torch.manual_seed(seed)
    train, evals, alt, stats = incyde_sets(index, images, (split == "train").to_numpy(),
                                           {"val": (split == "val").to_numpy()}, cfg, known)
    model, parent = build_model(cfg, args.init_from)
    meta = {"dataset": "incyde", "wind_averaging": "3min", "wind_averaging_s": 180, "parent": parent, "run_id": out.name}
    def on_epoch(epoch, model, opt, history, stopper):
        torch.save(payload(model, cfg, stats, {**meta, "epoch": epoch, "history": history}), out / f"epoch_{epoch:02d}.pt")
    t0 = time.time()
    model, history, stopper = fit(model, train, evals["val"], cfg, device, epochs=int(cfg["train"]["epochs"]),
                                  aug=cfg["augment"], mask=mask, on_epoch=on_epoch, alt_images=alt, seed=seed)
    wall_main = time.time() - t0
    best_epoch = int(np.argmin([h["val_mae"] for h in history]))
    ev, pw, logits = evaluate(model, evals["val"], cfg, device, mask, train.winds.numpy(), train.labels)
    results["fixed_split_val"] = ev
    torch.save(payload(model, cfg, stats, {**meta, "history": history, "best_epoch": best_epoch,
                                           "best_val_mae": stopper.best, "split_storm_ids": {
                                               s: sorted(index.loc[split == s, "storm_id"].unique().tolist()) for s in ("train", "val", "test")}}),
               out / "checkpoint.pt")
    vi = index[(split == "val") & index["is_original"]]
    pd.DataFrame({"filename": vi["filename"].to_numpy(), "storm_id": vi["storm_id"].to_numpy(),
                  "wind_true_kt": evals["val"].winds.numpy(), "wind_pred_kt": pw,
                  "class_true": evals["val"].labels,
                  "class_pred_head": [known[i] for i in logits.argmax(1)]}).to_csv(out / "val_predictions.csv", index=False)
    del train, evals, alt
    # 2) storm-grouped CV over train+val, fixed epoch count = best epoch + 1
    folds, fold_table = cv_folds(index, int(cfg["cv"]["n_folds"]), int(cfg["cv"].get("seed", 42)), args.cv_folds_csv)
    fold_table.to_csv(out / "cv_folds.csv", index=False)
    cv = []
    if not args.skip_cv:
        for k in range(int(cfg["cv"]["n_folds"])):
            tr_rows = (folds.notna() & (folds != k) & split.isin(["train", "val"])).to_numpy()
            ev_rows = (folds == k).to_numpy() & split.isin(["train", "val"]).to_numpy()
            torch.manual_seed(seed + k)
            ftrain, fevals, falt, _ = incyde_sets(index, images, tr_rows, {"fold": ev_rows}, cfg, known)
            fmodel, _ = build_model(cfg, args.init_from)
            fmodel, _h, _s = fit(fmodel, ftrain, None, cfg, device, epochs=best_epoch + 1, aug=cfg["augment"],
                                 mask=mask, fixed_epochs=True, alt_images=falt, seed=seed + k)
            fev, _, _ = evaluate(fmodel, fevals["fold"], cfg, device, mask, ftrain.winds.numpy(), ftrain.labels, bootstrap=False)
            fev["fold"] = k
            fev["eval_storms"] = sorted(set(fevals["fold"].storms.tolist()))
            cv.append(fev)
            print(json.dumps({"fold": k, "mae": fev["wind"]["mae_kt"], "f1": fev["category_class_head"]["macro_f1"]}), flush=True)
            del ftrain, fevals, falt
    if cv:
        def ms(get):
            v = np.array([get(f) for f in cv], dtype=float)
            return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)), "per_fold": v.tolist()}
        results["storm_grouped_cv"] = {
            "n_folds": len(cv), "epochs_per_fold": best_epoch + 1,
            "wind_mae_kt": ms(lambda f: f["wind"]["mae_kt"]),
            "wind_rmse_kt": ms(lambda f: f["wind"]["rmse_kt"]),
            "wind_bias_kt": ms(lambda f: f["wind"]["bias_kt"]),
            "cat_head_accuracy": ms(lambda f: f["category_class_head"]["accuracy"]),
            "cat_head_macro_f1": ms(lambda f: f["category_class_head"]["macro_f1"]),
            "cat_binned_accuracy": ms(lambda f: f["category_binned_wind"]["accuracy"]),
            "cat_binned_macro_f1": ms(lambda f: f["category_binned_wind"]["macro_f1"]),
            "baseline_mean_mae_kt": ms(lambda f: f["baseline_train_mean_wind"]["mae_kt"]),
            "baseline_mean_rmse_kt": ms(lambda f: f["baseline_train_mean_wind"]["rmse_kt"]),
            "baseline_majority_accuracy": ms(lambda f: f["baseline_majority_class"]["accuracy"]),
            "baseline_majority_macro_f1": ms(lambda f: f["baseline_majority_class"]["macro_f1"]),
            "imd7_accuracy": ms(lambda f: f["category_imd7_from_wind"]["accuracy"]),
            "imd7_macro_f1": ms(lambda f: f["category_imd7_from_wind"]["macro_f1"]),
            "baseline_majority_imd7_accuracy": ms(lambda f: f["baseline_majority_imd7"]["accuracy"]),
            "baseline_majority_imd7_macro_f1": ms(lambda f: f["baseline_majority_imd7"]["macro_f1"]),
            "folds": cv,
        }
    prep = json.loads((args.cache / "incyde_prep.json").read_text())
    orig = index["is_original"]
    manifest = {
        "kind": "finetune" if args.init_from else "scratch", "git_commit": git_sha(), "config": cfg,
        "seed": seed, "init_from": parent, "device": str(device), "hardware": hardware(),
        "wall_time_s_fixed_split": round(wall_main, 1), "wall_time_s_total": round(time.time() - t_all, 1),
        "best_epoch": best_epoch, "history": history, "data": prep,
        "rows_originals": {s: int((orig & (split == s)).sum()) for s in ("train", "val", "test")},
        "rows_rot_180_train": int(((~orig) & (split == "train")).sum()),
        "storms": {s: int(index.loc[split == s, "storm_id"].nunique()) for s in ("train", "val", "test")},
        "cv_folds_source": str(args.cv_folds_csv) if args.cv_folds_csv else "built: seed 42, PR #35 rule (cooccurrence groups round-robin)",
        "test_split": "sealed: never loaded for training, model selection or reporting",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    v = results["fixed_split_val"]
    print(json.dumps({"val_mae": v["wind"]["mae_kt"], "val_rmse": v["wind"]["rmse_kt"],
                      "head_acc": v["category_class_head"]["accuracy"], "head_f1": v["category_class_head"]["macro_f1"],
                      "bin_acc": v["category_binned_wind"]["accuracy"], "bin_f1": v["category_binned_wind"]["macro_f1"],
                      "cv": {k: results.get("storm_grouped_cv", {}).get(k) for k in ("wind_mae_kt", "cat_head_macro_f1")}},
                     indent=1, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pretrain", "finetune"])
    p.add_argument("--config", type=Path, default=REPO / "config/incyde_cpu.yaml")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default=None, choices=("auto", "xpu", "cuda", "cpu"))
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--no-convert", dest="convert", action="store_false")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--cv-folds-csv", type=Path, default=None, help="storm_id,fold table used verbatim (e.g. SWE's)")
    p.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=0.0005")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    torch.manual_seed(int(cfg["project"]["seed"]))
    np.random.seed(int(cfg["project"]["seed"]))
    device = resolve_device(args.device or cfg["train"]["device"])
    print(f"device={device}", flush=True)
    if args.command == "pretrain":
        cmd_pretrain(args, cfg, device)
    else:
        cmd_finetune(args, cfg, device)


if __name__ == "__main__":
    main()
