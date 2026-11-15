"""Write results/incyde_v0/*.json summaries and the tables in docs/RESULTS.md
from a finished run directory (metrics.json + run_manifest.json). No numbers
are typed by hand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(x, d=2):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def cm_table(res: dict) -> str:
    labels = res["confusion_labels"]
    lines = ["| true \\ predicted | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for name, row in zip(labels, res["confusion_matrix"]):
        lines.append(f"| **{name}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def tables(run: Path) -> str:
    m = json.loads((run / "metrics.json").read_text())
    man = json.loads((run / "run_manifest.json").read_text())
    v = m["fixed_split_val"]
    cv = m.get("storm_grouped_cv")
    out = []
    out.append(f"Run `{run.name}`, git `{man['git_commit']}`, best epoch {man['best_epoch']} "
               f"(of {len(man['history'])} run), fixed-split wall time {man['wall_time_s_fixed_split']:.0f} s, "
               f"total {man['wall_time_s_total']:.0f} s on {man['hardware']['cpu_count']} CPU threads "
               f"(`{man['hardware']['platform']}`, torch {man['hardware']['torch']}).\n")
    out.append("### Validation storms (fixed split: 60 train storms → 14 validation storms)\n")
    out.append("| Metric | Model | Baseline |\n|---|---|---|")
    w, b = v["wind"], v["baseline_train_mean_wind"]
    bs = v["wind_storm_bootstrap"]
    out.append(f"| Wind MAE (kt) | {f(w['mae_kt'])} (95% storm bootstrap {f(bs['mae_kt_ci_low'])}–{f(bs['mae_kt_ci_high'])}) | {f(b['mae_kt'])} (train-mean wind) |")
    out.append(f"| Wind RMSE (kt) | {f(w['rmse_kt'])} ({f(bs['rmse_kt_ci_low'])}–{f(bs['rmse_kt_ci_high'])}) | {f(b['rmse_kt'])} |")
    out.append(f"| Wind bias (kt, pred − true) | {f(w['bias_kt'])} | {f(b['bias_kt'])} |")
    c5, m5 = v["category_binned_wind"], v["baseline_majority_class"]
    c7, m7 = v["category_imd7_from_wind"], v["baseline_majority_imd7"]
    out.append(f"| Category accuracy, 5 project bins | {f(c5['accuracy'],3)} | {f(m5['accuracy'],3)} (majority `{m5['majority_class']}`) |")
    out.append(f"| Category macro-F1, 5 project bins | {f(c5['macro_f1'],3)} | {f(m5['macro_f1'],3)} |")
    out.append(f"| Category accuracy, IMD 7 classes | {f(c7['accuracy'],3)} | {f(m7['accuracy'],3)} (majority `{m7['majority_class']}`) |")
    out.append(f"| Category macro-F1, IMD 7 classes | {f(c7['macro_f1'],3)} | {f(m7['macro_f1'],3)} |")
    out.append(f"\nRows: {v['n_rows']} original frames from {v['n_storms']} storms.\n")
    out.append("#### Wind error by true class (validation, #33 metrics, 95% interval resamples storms)\n")
    out.append("| True class | Rows | Storms | MAE | RMSE | Bias | RMSE 95% CI | Note |\n|---|---|---|---|---|---|---|---|")
    for name, r in v["wind_by_true_class_bootstrap"].items():
        note = "indicative (n_storms<3)" if r["indicative_n_storms_lt_3"] else ""
        out.append(f"| {name} | {r['n_rows']} | {r['n_storms']} | {f(r['mae_kt'])} | {f(r['rmse_kt'])} | {f(r['bias_kt'])} | {f(r['rmse_kt_ci_low'])}–{f(r['rmse_kt_ci_high'])} | {note} |")
    out.append("\n#### Confusion matrix, 5 project bins (category = predicted wind binned)\n")
    out.append(cm_table(c5))
    out.append("\n#### Confusion matrix, IMD 7 classes (category = predicted wind on the IMD scale)\n")
    out.append(cm_table(c7))
    if cv:
        out.append(f"\n### Storm-grouped {cv['n_folds']}-fold CV over train + validation storms "
                   f"({cv['epochs_per_fold']} epochs per fold, no early stopping on the fold)\n")
        out.append(f"Folds: {man.get('cv_folds_source')}.\n")
        out.append("| Metric | Model mean ± std | Baseline mean ± std | Per fold (model) |\n|---|---|---|---|")
        def row(label, key, base, d=2):
            a = cv[key]; bb = cv[base] if base else None
            bstr = f"{f(bb['mean'],d)} ± {f(bb['std'],d)}" if bb else "–"
            out.append(f"| {label} | {f(a['mean'],d)} ± {f(a['std'],d)} | {bstr} | {', '.join(f(x,d) for x in a['per_fold'])} |")
        row("Wind MAE (kt)", "wind_mae_kt", "baseline_mean_mae_kt")
        row("Wind RMSE (kt)", "wind_rmse_kt", "baseline_mean_rmse_kt")
        row("Wind bias (kt)", "wind_bias_kt", None)
        row("Accuracy, 5 bins", "cat_binned_accuracy", "baseline_majority_accuracy", 3)
        row("Macro-F1, 5 bins", "cat_binned_macro_f1", "baseline_majority_macro_f1", 3)
        row("Accuracy, IMD 7", "imd7_accuracy", "baseline_majority_imd7_accuracy", 3)
        row("Macro-F1, IMD 7", "imd7_macro_f1", "baseline_majority_imd7_macro_f1", 3)
        out.append("\n#### Wind error by true class, pooled over CV folds\n")
        out.append("| True class | Rows | MAE | RMSE | Bias |\n|---|---|---|---|---|")
        pooled = {}
        for fold in cv["folds"]:
            for name, r in fold["wind_by_true_class"].items():
                p = pooled.setdefault(name, {"n": 0, "ae": 0.0, "se": 0.0, "e": 0.0})
                n = r["n"]; p["n"] += n; p["ae"] += r["mae_kt"] * n; p["se"] += r["rmse_kt"] ** 2 * n; p["e"] += r["bias_kt"] * n
        for name, p in pooled.items():
            out.append(f"| {name} | {int(p['n'])} | {f(p['ae']/p['n'])} | {f((p['se']/p['n'])**0.5)} | {f(p['e']/p['n'])} |")
        out.append("\n#### Storms per CV fold\n")
        for fold in cv["folds"]:
            out.append(f"- fold {fold['fold']}: {fold['n_rows']} frames, {fold['n_storms']} storms")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    print(tables(args.run))


if __name__ == "__main__":
    main()
