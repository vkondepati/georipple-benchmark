"""Summarize the confirmation batch (network seeds 10-19) across lane and sourcing conditions.

Reads results/confirm/C*.json and writes results/conditions.tex and
results/conditions_summary.json. H1-H6 are the pre-registered comparisons
(docs/confirmation-protocol.md); in C0 they are confirmatory (Holm-adjusted), in C1-C3
they are paired robustness checks on the same networks, reported descriptively.
"""
import argparse
import json
import os

from make_tables import (
    HYPOTHESES,
    PREDICTED_SIGNS,
    cell,
    clustered_stats,
    holm_tests,
    load_config,
    load_runs,
    miss_summary,
)

HERE = os.path.dirname(__file__)
CONDITIONS = [
    ("C0", "Straight lanes, 1 DC/dealer"),
    ("C1", "Road lanes, 1 DC/dealer"),
    ("C2", "Straight lanes, 2 DCs/dealer"),
    ("C3", "Road lanes, 2 DCs/dealer"),
]


def unmet_difference(a, b):
    return lambda r: r["resolution"][a]["unmet_units"] - r["resolution"][b]["unmet_units"]


def condition_summary(runs, top_k):
    plans = {(r["seed"], r["hazard"]): r for r in runs}
    return {
        "n_records": len(runs),
        "n_networks": len({r["seed"] for r in runs}),
        "recall_full": clustered_stats(runs, lambda r: r["prediction"]["Full"]["recall"], (0, 1)),
        "precision_full": clustered_stats(
            runs, lambda r: r["prediction"]["Full"]["precision"], (0, 1)),
        "mae_full": clustered_stats(runs, lambda r: r["prediction"]["Full"]["mae_days"]),
        "unmet_no_action": clustered_stats(
            runs, lambda r: r["resolution"]["NoAction"]["unmet_units"]),
        "unmet_full": clustered_stats(runs, lambda r: r["resolution"]["Full"]["unmet_units"]),
        "cost_full": clustered_stats(runs, lambda r: r["resolution"]["Full"]["cost_k"]),
        "hypotheses": holm_tests(runs, HYPOTHESES, PREDICTED_SIGNS),
        "unmet_full_minus_a2": clustered_stats(runs, unmet_difference("Full", "A2")),
        "unmet_full_minus_a3": clustered_stats(runs, unmet_difference("Full", "A3")),
        "unmet_b4_minus_no_action": clustered_stats(runs, unmet_difference("B4", "NoAction")),
        "unmet_b3_minus_no_action": clustered_stats(runs, unmet_difference("B3", "NoAction")),
        "stci_topk_changed_with_lambda0": int(
            sum(r["stci_top_changed_lambda0"] for r in plans.values())),
        "stci_plans": len(plans),
        "top_k": top_k,
        "stci_plans_with_nonzero_at_most_k": int(
            sum(r["stci_nonzero"] <= top_k for r in plans.values())),
        "misses": miss_summary(runs),
        "false_alarm": {
            key: clustered_stats(runs, lambda r, key=key: r["false_alarm"][key]["unmet_delta"])
            for key in runs[0]["false_alarm"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-dir", default=os.path.join(HERE, "..", "results", "confirm"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "results"))
    a = ap.parse_args()
    summary = {"conditions": {}, "labels": dict(CONDITIONS)}
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Confirmation on ten fresh networks (seeds 10--19) under four lane and "
        r"sourcing conditions: mean [descriptive 95\% network-clustered t interval]. "
        r"$\Delta$ columns are paired differences; negative unmet values favor GeoRipple.}",
        r"\label{tab:conditions}",
        r"\centering",
        r"\small\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"Condition & Recall & $\Delta$Recall vs B2 & $\Delta$MAE vs A1 & "
        r"\multicolumn{4}{c}{$\Delta$Unmet of GeoRipple vs} & $\lambda$ changes\\",
        r"\cmidrule(lr){5-8}",
        r" & & (H1) & (H2) & none (H3) & A4 (H4) & B4 (H5) & B3 (H6) & top-$K$\\",
        r"\midrule",
    ]
    pooled = []
    for cid, label in CONDITIONS:
        path = os.path.join(a.confirm_dir, f"{cid}.json")
        runs = load_runs(path)
        pooled += runs
        s = condition_summary(runs, load_config(path).get("top_k", 10))
        summary["conditions"][cid] = s
        h = s["hypotheses"]
        lines.append(
            f"{cid}: {label} & {cell(s['recall_full'], '{:.2f}')} & "
            f"{cell(h['H1_recall_full_minus_b2'], '{:.3f}')} & "
            f"{cell(h['H2_mae_full_minus_a1'], '{:.1f}')} & "
            f"{cell(h['H3_unmet_full_minus_no_action'], '{:.0f}')} & "
            f"{cell(h['H4_unmet_full_minus_a4'], '{:.0f}')} & "
            f"{cell(h['H5_unmet_full_minus_b4'], '{:.0f}')} & "
            f"{cell(h['H6_unmet_full_minus_b3'], '{:.0f}')} & "
            f"{s['stci_topk_changed_with_lambda0']}/{s['stci_plans']} \\\\")
    summary["misses_pooled"] = miss_summary(pooled)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    os.makedirs(a.outdir, exist_ok=True)
    for name, content in (
        ("conditions.tex", "\n".join(lines) + "\n"),
        ("conditions_summary.json", json.dumps(summary, indent=1, allow_nan=False) + "\n"),
    ):
        path = os.path.join(a.outdir, name)
        with open(path + ".tmp", "w") as f:
            f.write(content)
        os.replace(path + ".tmp", path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
