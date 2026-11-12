"""Sanity-check a built INCYDE manifest and the proposed storm split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Path to manifest.csv")
    return p.parse_args()


def main() -> None:
    m = pd.read_csv(parse_args().manifest, keep_default_na=False, na_values=[""], low_memory=False)
    m["qc_flags"] = m.qc_flags.fillna("")
    print(m.qc_flags.value_counts())
    print(m.groupby("proposed_split").agg(storms=("storm_id", "nunique"), rows=("file", "size"), used=("use_in_proposed_split", "sum")))
    o = m[~m.is_augmented]
    print(o[o.use_in_proposed_split].groupby("proposed_split").size())
    print(
        o.groupby(["proposed_split", "storm_id"])
        .label_kt.max()
        .reset_index()
        .groupby("proposed_split")
        .apply(lambda g: ", ".join(f"{s[7:10]}:{v}" for s, v in zip(g.storm_id, g.label_kt)))
        .to_string()
    )
    print("storms in >1 proposed split:", (m.groupby("storm_id").proposed_split.nunique() > 1).sum())
    print("storms in >1 incyde set:", (m.groupby("storm_id").incyde_set.nunique() > 1).sum())
    u = m[m.use_in_proposed_split]
    print("source frames in >1 split (used rows):", (u.groupby("source_frame").proposed_split.nunique() > 1).sum())
    print("source frames in >1 incyde set:", (m.groupby("source_frame").incyde_set.nunique() > 1).sum())
    print(
        "timestamps shared across proposed splits:",
        (o.groupby("timestamp").proposed_split.nunique() > 1).sum(),
        " across incyde sets:",
        (o.groupby("timestamp").incyde_set.nunique() > 1).sum(),
    )
    gs = o.groupby("storm_group").storm_name.unique()
    print({k: list(v) for k, v in gs.items() if o[o.storm_group == k].storm_id.nunique() > 1})


if __name__ == "__main__":
    main()
