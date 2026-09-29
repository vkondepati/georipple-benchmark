"""Summarize the one-factor-at-a-time sensitivity artifact."""
import argparse
import json
import os

from make_tables import clustered_stats, load_runs

HERE = os.path.dirname(__file__)
LABELS = {
    "q50": r"Planning quantile $q=0.50$",
    "base": r"Reference: $q=0.75$, $K=10$, $M=8$",
    "q90": r"Planning quantile $q=0.90$",
    "k5": r"Top-$K=5$",
    "k15": r"Top-$K=15$",
    "m4": r"Ensemble size $M=4$",
    "m16": r"Ensemble size $M=16$",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", default=os.path.join(HERE, "..", "results", "sensitivity_runs.json")
    )
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "results"))
    args = ap.parse_args()
    runs = load_runs(args.runs)
    config_order = [config["id"] for config in json.load(open(args.runs))["configs"]]
    summary = {"n_records": len(runs), "configs": {}}
    lines = [
        r"\begin{table}[t]",
        r"\caption{One-factor-at-a-time sensitivity (means over five seed clusters). "
        r"$\Delta$ values are relative to paired no action.}",
        r"\label{tab:sensitivity}",
        r"\centering",
        r"\small\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Configuration & Recall & $\Delta$Unmet & $\Delta$Fill & Cost (kCU) & Actions\\",
        r"\midrule",
    ]
    for config_id in config_order:
        subset = [run for run in runs if run["config"] == config_id]
        stats = {
            "recall": clustered_stats(subset, lambda run: run["prediction"]["recall"], (0, 1)),
            "unmet_delta": clustered_stats(
                subset, lambda run: run["full"]["unmet_units"] - run["no_action"]["unmet_units"]
            ),
            "fill_delta": clustered_stats(
                subset, lambda run: run["full"]["fill_rate"] - run["no_action"]["fill_rate"]
            ),
            "cost_k": clustered_stats(subset, lambda run: run["full"]["cost_k"]),
            "actions": clustered_stats(subset, lambda run: run["plan_info"]["activated"]),
        }
        summary["configs"][config_id] = stats
        mean = {name: stat["mean"] for name, stat in stats.items()}
        lines.append(
            f"{LABELS[config_id]} & {mean['recall']:.2f} & {mean['unmet_delta']:.0f} & "
            f"{mean['fill_delta']:.3f} & {mean['cost_k']:.1f} & {mean['actions']:.1f} \\\\"
        )
    unmet = [value["unmet_delta"]["mean"] for value in summary["configs"].values()]
    summary["robustness"] = {
        "configs_reducing_unmet": sum(value < 0 for value in unmet),
        "n_configs": len(unmet),
        "unmet_delta_range": [min(unmet), max(unmet)],
    }
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    os.makedirs(args.outdir, exist_ok=True)
    for name, content in (
        ("sensitivity.tex", "\n".join(lines) + "\n"),
        ("sensitivity_summary.json", json.dumps(summary, indent=1, allow_nan=False) + "\n"),
    ):
        path = os.path.join(args.outdir, name)
        with open(path + ".tmp", "w") as f:
            f.write(content)
        os.replace(path + ".tmp", path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
