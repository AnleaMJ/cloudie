"""Print per-class and per-split tables from a built INCYDE manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ORDER = ["<17", "D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Path to manifest.csv")
    return p.parse_args()


def tab(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("label_class_imd")
        .agg(frames=("file", "size"), storms=("storm_id", "nunique"))
        .reindex(ORDER)
        .fillna(0)
        .astype(int)
    )


def main() -> None:
    m = pd.read_csv(parse_args().manifest, keep_default_na=False, na_values=[""], low_memory=False)
    m["qc_flags"] = m.qc_flags.fillna("")
    o = m[~m.is_augmented]
    for sp in ["train", "val", "test"]:
        g = o[o.proposed_split == sp].groupby("incyde_sequence").agg(
            n=("label_kt", "size"), pk=("label_kt", "max"), nm=("storm_name", "first")
        )
        print(
            sp,
            len(g),
            "; ".join(
                f"S{i}{'(' + str(r.nm) + ')' if isinstance(r.nm, str) and r.nm else ''}:{r.pk}"
                for i, r in g.iterrows()
            ),
        )
    print()
    print("ALL rows (incl aug)")
    print(tab(m).T.to_string())
    print()
    print("Original frames only")
    print(tab(o).T.to_string())
    print()
    print("Original frames by INCYDE set")
    print(pd.crosstab(o.label_class_imd, o.incyde_set).reindex(ORDER).fillna(0).astype(int).T.to_string())
    print(o.groupby(["incyde_set", "label_class_imd"]).storm_id.nunique().unstack().reindex(columns=ORDER).fillna(0).astype(int).to_string())
    u = m[m.use_in_proposed_split]
    print()
    print("Proposed split, used rows (frames incl. train aug)")
    print(pd.crosstab(u.label_class_imd, u.proposed_split).reindex(ORDER).fillna(0).astype(int).T.to_string())
    uo = u[~u.is_augmented]
    print("Proposed split, original frames")
    print(pd.crosstab(uo.label_class_imd, uo.proposed_split).reindex(ORDER).fillna(0).astype(int).T.to_string())
    print(uo.groupby(["proposed_split", "label_class_imd"]).storm_id.nunique().unstack().reindex(columns=ORDER).fillna(0).astype(int).to_string())
    for c in ["ESCS", "SuCS"]:
        x = o[o.label_class_imd == c].groupby(["storm_name", "incyde_set", "proposed_split"]).size()
        print(c)
        print(x.to_string())
    pk = o.groupby("storm_id").label_kt.max()
    print(
        "storms by peak class:",
        pd.cut(pk, [0, 16, 27, 33, 47, 63, 89, 119, 200], labels=ORDER).value_counts().reindex(ORDER).to_dict(),
    )
    print("INCYDE category ESCS present?", "ESCS" in set(m.label_class_incyde), " INCYDE VSCS label range:", m[m.label_class_incyde == "VSCS"].label_kt.agg(["min", "max"]).tolist())
    print("label value counts top end", o.label_kt.value_counts().sort_index().tail(10).to_dict())
    print("<17 kt original frames by storm", o[o.label_kt < 17].groupby("storm_name", dropna=False).size().to_dict(), o[o.label_kt < 17].incyde_sequence.unique())


if __name__ == "__main__":
    main()
