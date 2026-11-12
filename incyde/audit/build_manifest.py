"""Build the INCYDE (Zenodo 8015544) frame manifest, storm-wise split, and QC flags.

CPU only. Does not download data. Paths are configurable; nothing is hardcoded
to a machine-specific location.

Inputs
------
- Parsed frame table (incyde2.pkl): labels CSV plus filename fields and pHash.
- Per-original IMD interpolation table (R.pkl).
- IMD best-track workbook, parsed with imdtrack.

Outputs
-------
- <root>/manifest.csv (full ~103k-row frame table; not committed)
- incyde/audit/storm_split.csv (one row per Sequence)
- incyde/audit/exclusions.csv (flagged frames)
- incyde/audit/proposed_split_groups.csv (co-occurrence groups used for the split)

The split is grouped by storms that share any original-frame timestamp, so
concurrent storms stay in the same split. Seed 20261112. The two SuCS storms
are placed by hand: the KYAAR group (Sequence 67) in test, AMPHAN in train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))

from match import SEQUENCE_TO_IMD  # noqa: E402


def imd_class(k: float) -> str:
    if k < 17:
        return "<17"
    if k <= 27:
        return "D"
    if k <= 33:
        return "DD"
    if k <= 47:
        return "CS"
    if k <= 63:
        return "SCS"
    if k <= 89:
        return "VSCS"
    if k <= 119:
        return "ESCS"
    return "SuCS"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Directory for manifest.csv and default input paths (default: cwd).",
    )
    p.add_argument(
        "--labels-pickle",
        type=Path,
        default=None,
        help="incyde2.pkl (labels + parsed filenames + pHash). Default: <root>/audit/incyde2.pkl or <audit-dir>/incyde2.pkl.",
    )
    p.add_argument(
        "--imd-interp-pickle",
        type=Path,
        default=None,
        help="R.pkl (per-original IMD interpolation). Default: <root>/audit/R.pkl or <audit-dir>/R.pkl.",
    )
    p.add_argument(
        "--imd-best-track",
        type=Path,
        default=None,
        help="IMD Best_Tracks_Data_1982-2026.xlsx. Default: <root>/../imd/imd_best_tracks_1982-2026.xlsx.",
    )
    p.add_argument("--output-manifest", type=Path, default=None, help="Default: <root>/manifest.csv")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory for split/exclusion CSVs. Default: this script's directory.")
    p.add_argument("--seed", type=int, default=20261112, help="RNG seed for the proposed split (default: 20261112).")
    return p.parse_args()


def _first_existing(candidates: list[Path], what: str) -> Path:
    for path in candidates:
        if path is not None and path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {what}. Looked at: " + ", ".join(str(c) for c in candidates if c is not None))


def build(args: argparse.Namespace) -> None:
    import imdtrack
    import networkx as nx

    root = args.root.resolve()
    out_dir = (args.output_dir or AUDIT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_pkl = args.labels_pickle or _first_existing(
        [root / "audit" / "incyde2.pkl", AUDIT_DIR / "incyde2.pkl", root / "incyde2.pkl"],
        "incyde2.pkl",
    )
    interp_pkl = args.imd_interp_pickle or _first_existing(
        [root / "audit" / "R.pkl", AUDIT_DIR / "R.pkl", root / "R.pkl"],
        "R.pkl",
    )
    best_track = args.imd_best_track or _first_existing(
        [
            root / "imd_best_tracks_1982-2026.xlsx",
            root.parent / "imd" / "imd_best_tracks_1982-2026.xlsx",
        ],
        "IMD best-track workbook",
    )
    manifest_path = args.output_manifest or (root / "manifest.csv")

    d = pd.read_pickle(labels_pkl)
    R = pd.read_pickle(interp_pkl)
    bt = imdtrack.load(path=str(best_track)).observations
    bt["t"] = pd.to_datetime(bt.time_resolved)

    S = bt.groupby("storm_id").agg(
        imd_name=("name", "first"),
        basin=("basin", lambda b: "|".join(sorted(set(map(str, b))))),
    )

    d["source_frame"] = np.where(
        d.aug == "orig",
        d.filename,
        d.filename.str.replace(r"_(flip_h|flip_v|rot_90|rot_180|rot_270)\.jpg$", ".jpg", regex=True),
    )
    d["imd_storm_id"] = d.Sequence.map(SEQUENCE_TO_IMD)
    d = d.join(S, on="imd_storm_id")
    nm = d.Name.fillna("").str.replace("’", "").str.strip().str.upper()
    nm = nm.where(~nm.isin(["", "-", "--"]), "")
    d["storm_name"] = np.where(nm != "", nm, d.imd_name.fillna("").str.upper())
    d["storm_id"] = "INCYDE-S" + d.Sequence.astype(str).str.zfill(2) + "_IMD-" + d.imd_storm_id
    d["timestamp_utc"] = (d.ts - pd.Timedelta(hours=5.5)).dt.strftime("%Y-%m-%dT%H:%M:00Z")
    d["filename_time_ist"] = d.ts.dt.strftime("%Y-%m-%d %H:%M")
    d["imd_class"] = d["MSWS kts"].apply(imd_class)
    d["slot_inferred"] = np.where(d.ts.dt.minute.isin([0, 30]), "hh00_hh30", "hh15_hh45")
    r = R.set_index("filename")[["imd_interp", "match"]]
    d = d.join(r, on="source_frame")

    flags = []
    for _, x in d.iterrows():
        f = []
        if x.Sequence == 42 and x.ts < pd.Timestamp("2017-10-01"):
            f.append("wrong_date_imd_daymonth_swap")
        elif pd.isna(x.imd_interp):
            f.append("outside_imd_track_span")
        elif not x.match:
            f.append("label_ne_imd_interp")
        flags.append(";".join(f))
    d["qc_flags"] = flags

    # Proposed split: storms that share any original-frame timestamp stay together.
    o = d[d.aug == "orig"]
    G = nx.Graph()
    G.add_nodes_from(o.Sequence.unique())
    for _, grp_ts in o.groupby("ts"):
        s = sorted(grp_ts.Sequence.unique())
        for a in s[1:]:
            G.add_edge(s[0], a)
    comp = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
    pk = o.groupby("Sequence")["MSWS kts"].max()
    grp = (
        pd.DataFrame({"peak": pk, "grp": pk.index.map(comp)})
        .groupby("grp")
        .agg(peak=("peak", "max"), seqs=("peak", lambda s: list(s.index)))
    )
    grp["pclass"] = grp.peak.apply(imd_class)
    rng = np.random.default_rng(args.seed)
    split = {}
    for c, gg in grp.groupby("pclass"):
        ids = list(gg.index)
        rng.shuffle(ids)
        n = len(ids)
        if c == "SuCS":
            # Only 2 SuCS storms: the KYAAR group (Sequence 67) -> test, AMPHAN -> train.
            for i in ids:
                split[i] = "test" if 67 in gg.loc[i, "seqs"] else "train"
            continue
        nt = max(1, round(0.15 * n))
        nv = max(1, round(0.15 * n)) if n >= 3 else 0
        for k, i in enumerate(ids):
            split[i] = "test" if k < nt else "val" if k < nt + nv else "train"
    seq2split = {s: split[g] for g, row in grp.iterrows() for s in row.seqs}
    d["proposed_split"] = d.Sequence.map(seq2split)
    d["storm_group"] = d.Sequence.map(comp)
    d["use_in_proposed"] = (d.qc_flags != "wrong_date_imd_daymonth_swap") & (
        (d.aug == "orig") | (d.proposed_split == "train")
    )

    man = pd.DataFrame(
        {
            "file": "Cropped_cyclone_data/Cropped_cyclone_images/" + d.filename,
            "storm_id": d.storm_id,
            "incyde_sequence": d.Sequence,
            "imd_storm_id": d.imd_storm_id,
            "storm_name": d.storm_name,
            "timestamp": d.timestamp_utc,
            "filename_time_ist": d.filename_time_ist,
            "basin": d.basin,
            "label_kt": d["MSWS kts"],
            "label_class_incyde": d.Category,
            "label_class_imd": d.imd_class,
            "is_augmented": d.aug != "orig",
            "aug_type": d.aug.replace("orig", ""),
            "source_frame": "Cropped_cyclone_data/Cropped_cyclone_images/" + d.source_frame,
            "object_index": d.obj,
            "incyde_set": d.Set,
            "proposed_split": d.proposed_split,
            "storm_group": d.storm_group,
            "use_in_proposed_split": d.use_in_proposed,
            "imd_interp_kt": d.imd_interp.round(2),
            "slot_inferred": d.slot_inferred,
            "width": d.w,
            "height": d.h,
            "phash": d.phash,
            "qc_flags": d.qc_flags,
        }
    )
    man = man.sort_values(["incyde_sequence", "timestamp", "object_index", "aug_type"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(manifest_path, index=False)

    grp.assign(split=grp.index.map(split)).to_csv(out_dir / "proposed_split_groups.csv")
    write_storm_split(man, out_dir / "storm_split.csv")
    write_exclusions(man, out_dir / "exclusions.csv")
    print(len(man), "rows written to", manifest_path)


def write_storm_split(man: pd.DataFrame, path: Path) -> None:
    o = man[~man.is_augmented]
    g = o.groupby("incyde_sequence", as_index=False).agg(
        storm_id=("storm_id", "first"),
        name=("storm_name", "first"),
        basin=("basin", "first"),
        peak_kt=("label_kt", "max"),
        proposed_split=("proposed_split", "first"),
        cooccurrence_group=("storm_group", "first"),
    )
    g["peak_imd_class"] = g["peak_kt"].apply(imd_class)
    g["name"] = g["name"].fillna("")
    g["Sequence"] = g["incyde_sequence"].astype(int)
    out = g[
        [
            "Sequence",
            "storm_id",
            "name",
            "basin",
            "peak_imd_class",
            "proposed_split",
            "cooccurrence_group",
        ]
    ].sort_values("Sequence")
    out.to_csv(path, index=False)


def write_exclusions(man: pd.DataFrame, path: Path) -> None:
    flagged = man[man.qc_flags.fillna("") != ""].copy()
    flagged["filename"] = flagged["file"].str.rsplit("/", n=1).str[-1]
    flagged["_orig"] = ~flagged.is_augmented
    flagged["_s42"] = flagged.incyde_sequence.eq(42) & flagged.qc_flags.str.contains("wrong_date")
    ex = flagged.sort_values(
        ["_s42", "_orig", "incyde_sequence", "filename"],
        ascending=[False, False, True, True],
    )[["filename", "qc_flags"]].rename(columns={"qc_flags": "qc_flag"})
    ex.to_csv(path, index=False)


if __name__ == "__main__":
    build(parse_args())
