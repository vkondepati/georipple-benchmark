"""Aggregate runs with descriptive 95% t intervals clustered by network seed."""
import argparse
import json
import os

import numpy as np
from scipy.stats import t as student_t

HERE = os.path.dirname(__file__)
PRED_ROWS = [("B1", "B1 Topology-only"), ("B2", "B2 Node-only spatial"),
             ("A1", "A1 No buffers"), ("Full", "GeoRipple (full)")]
RES_ROWS = [("NoAction", "No action"), ("B3", "B3 Action-count-matched greedy"),
            ("A2", "A2 No hazard routing"), ("A3", r"A3 $\lambda=0$"),
            ("Full", "GeoRipple (full)")]


def clustered_stats(runs, getter, bounds=None):
    """Mean and descriptive t interval over seed means; hazards are fixed strata."""
    groups = {}
    for run in runs:
        value = getter(run)
        if value is not None and np.isfinite(value):
            groups.setdefault(run["seed"], []).append(float(value))
    cluster_means = np.array([np.mean(v) for v in groups.values()], float)
    if not len(cluster_means):
        return {"mean": None, "ci95": [None, None], "n_records": 0, "n_seeds": 0}
    mean = float(cluster_means.mean())
    if len(cluster_means) > 1:
        margin = float(student_t.ppf(0.975, len(cluster_means) - 1)
                       * cluster_means.std(ddof=1) / np.sqrt(len(cluster_means)))
    else:
        margin = None
    interval = [mean - margin, mean + margin] if margin is not None else [None, None]
    if bounds is not None and margin is not None:
        interval = [float(np.clip(x, *bounds)) for x in interval]
    return {"mean": mean, "ci95": interval,
            "n_records": int(sum(map(len, groups.values()))), "n_seeds": len(cluster_means)}


def cell(stat, fmt):
    if stat["mean"] is None or not np.isfinite(stat["mean"]):
        return "--"
    lo, hi = stat["ci95"]
    if lo is None or not np.isfinite(lo):
        return fmt.format(stat["mean"])
    return f"{fmt.format(stat['mean'])} [{fmt.format(lo)}, {fmt.format(hi)}]"


def finite_mean(values):
    values = [float(value) for value in values
              if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def load_runs(path):
    with open(path) as f:
        data = json.load(f)
    runs = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{path}: expected a non-empty 'runs' list")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "results", "runs.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "results"))
    a = ap.parse_args()
    try:
        runs = load_runs(a.runs)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        ap.error(str(exc))
    n = len(runs)
    n_seeds = len({r["seed"] for r in runs})
    summary = {"n_records": n, "n_seed_clusters": n_seeds,
               "prediction": {}, "resolution": {}}

    pred_caption = (r"\caption{Stockout prediction versus realized outcome "
                    rf"(mean [descriptive 95\% seed-clustered t interval]; {n} records, "
                    rf"{n_seeds} seeds)}}")
    lines = [r"\begin{table}[t]", pred_caption,
             r"\label{tab:pred}", r"\centering", r"\small\setlength{\tabcolsep}{2pt}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Method & Prec.$^{\dagger}$ & Recall & Brier & MAE $\sigma_v$ (d)\\", r"\midrule"]
    for key, label in PRED_ROWS:
        p = clustered_stats(
            runs, lambda r, key=key: r["prediction"][key]["precision"], (0, 1))
        rc = clustered_stats(
            runs, lambda r, key=key: r["prediction"][key]["recall"], (0, 1))
        br = clustered_stats(
            runs, lambda r, key=key: r["prediction"][key]["brier"], (0, 1))
        mae = clustered_stats(runs, lambda r, key=key: r["prediction"][key]["mae_days"])
        mae_tp = clustered_stats(
            runs, lambda r, key=key: r["prediction"][key]["mae_days_true_positive"])
        summary["prediction"][key] = {
            "precision": p, "recall": rc, "brier": br, "mae_days": mae,
            "mae_days_true_positive": mae_tp,
        }
        ptxt = cell(p, "{:.2f}")
        if p["n_records"] != n:
            ptxt += f"$^{{[{p['n_records']}]}}$"
        lines.append(f"{label} & {ptxt} & {cell(rc, '{:.2f}')} & "
                     f"{cell(br, '{:.3f}')} & {cell(mae, '{:.1f}')} \\\\")
    precision_note = (r"\parbox{\linewidth}{\footnotesize $^{\dagger}$Records without a "
                      r"positive prediction are excluded; bracketed superscript gives the "
                      r"included count.}")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{2pt}", precision_note,
              r"\end{table}", ""]

    resolution_caption = (r"\caption{Resolution quality under realized hazards "
                          rf"(mean [descriptive 95\% seed-clustered t interval]; {n} records, "
                          rf"{n_seeds} seeds)}}")
    lines += [r"\begin{table}[t]", resolution_caption,
              r"\label{tab:resolve}", r"\centering", r"\small\setlength{\tabcolsep}{1.5pt}",
              r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
              r"Method & Fill rate & Unmet (u) & Recovery & TTR$_{95}$ (d)$^{\ddagger}$ & Cost (kCU)\\",
              r"\midrule"]
    for key, label in RES_ROWS:
        fr = clustered_stats(
            runs, lambda r, key=key: r["resolution"][key]["fill_rate"], (0, 1))
        um = clustered_stats(runs, lambda r, key=key: r["resolution"][key]["unmet_units"])
        rr = clustered_stats(
            runs, lambda r, key=key: float(r["resolution"][key]["ttr95_recovered"]),
            (0, 1))
        tt = clustered_stats(runs, lambda r, key=key: r["resolution"][key]["ttr95_days"])
        co = clustered_stats(runs, lambda r, key=key: r["resolution"][key]["cost_k"])
        summary["resolution"][key] = {
            "fill_rate": fr, "unmet_units": um, "recovery_rate": rr,
            "ttr95_days": tt, "cost_k": co,
        }
        recovery = cell(rr, "{:.0%}").replace("%", r"\%")
        lines.append(f"{label} & {cell(fr, '{:.3f}')} & {cell(um, '{:.0f}')} & "
                     f"{recovery} & {cell(tt, '{:.1f}')} & "
                     f"{cell(co, '{:.1f}')} \\\\")
    recovery_note = (r"\parbox{\linewidth}{\footnotesize $^{\ddagger}$Among records with "
                     r"observed recovery; Recovery is the fraction recovering within the "
                     r"horizon.}")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{2pt}", recovery_note,
              r"\end{table}"]

    plan_latencies = {(r["seed"], r["hazard"]): r["latency_seconds_full"] for r in runs}
    lat = list(plan_latencies.values())
    summary["latency_seconds_full"] = {"mean": float(np.mean(lat)),
                                       "max": float(np.max(lat)),
                                       "n_plans": len(lat)}
    summary["interval_method"] = {
        "description": "descriptive t interval over network-seed means",
        "bounded_metrics": "intersected with the metric's [0, 1] parameter space",
        "hazards": "fixed strata",
    }
    summary["per_hazard"] = {}
    for hz in sorted({r["hazard"] for r in runs}):
        sub = [r for r in runs if r["hazard"] == hz]
        summary["per_hazard"][hz] = {
            "recall": {k: finite_mean(r["prediction"][k]["recall"] for r in sub)
                       for k, _ in PRED_ROWS},
            "fill_rate": {k: finite_mean(r["resolution"][k]["fill_rate"] for r in sub)
                          for k, _ in RES_ROWS},
            "unmet_units": {k: finite_mean(r["resolution"][k]["unmet_units"] for r in sub)
                            for k, _ in RES_ROWS},
        }
    plan_changes = {(r["seed"], r["hazard"]): r["stci_top_changed_lambda0"] for r in runs}
    summary["stci_topk_changed_with_lambda0"] = int(sum(plan_changes.values()))
    effects = {
        "recall_full_minus_b2": lambda r: (
            r["prediction"]["Full"]["recall"] - r["prediction"]["B2"]["recall"]),
        "unmet_full_minus_no_action": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["NoAction"]["unmet_units"]),
        "fill_full_minus_no_action": lambda r: (
            r["resolution"]["Full"]["fill_rate"]
            - r["resolution"]["NoAction"]["fill_rate"]),
        "unmet_full_minus_a2": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["A2"]["unmet_units"]),
        "unmet_full_minus_a3": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["A3"]["unmet_units"]),
    }
    summary["paired_effects"] = {
        name: clustered_stats(runs, getter) for name, getter in effects.items()
    }

    os.makedirs(a.outdir, exist_ok=True)
    tables_path = os.path.join(a.outdir, "tables.tex")
    with open(tables_path + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tables_path + ".tmp", tables_path)
    summary_path = os.path.join(a.outdir, "summary.json")
    with open(summary_path + ".tmp", "w") as f:
        json.dump(summary, f, indent=1, allow_nan=False)
    os.replace(summary_path + ".tmp", summary_path)
    print("\n".join(lines))
    print(json.dumps({k: summary[k] for k in (
        "latency_seconds_full", "per_hazard", "paired_effects",
        "stci_topk_changed_with_lambda0")}, indent=1))


if __name__ == "__main__":
    main()
