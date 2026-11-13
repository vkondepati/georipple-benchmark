"""Aggregate runs with descriptive 95% t intervals clustered by network seed."""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from georipple.resolve import SHORTAGE_PENALTY  # noqa: E402

KCU_PER_UNMET = SHORTAGE_PENALTY / 1000.0
PRED_ROWS = [("B1", "B1 Topology-only"), ("B2", "B2 Node-only"),
             ("A1", "A1 No buffers"), ("Full", "GeoRipple (full)")]
RES_ROWS = [("NoAction", "No action"), ("B3", "B3 Action-count-matched greedy"),
            ("B4", "B4 Mean-exposure MILP"),
            ("A2", "A2 No hazard routing"), ("A3", r"A3 $\lambda=0$"),
            ("A4", "A4 Absolute dispatch"),
            ("Full", "GeoRipple (full)")]
# Pre-registered confirmatory comparisons (docs/confirmation-protocol.md): two-sided
# paired t over network means, Holm across all six, confirmed only with the predicted sign.
def paired(a, b):
    """Difference of two per-record metrics; undefined (None) if either side is."""
    return None if a is None or b is None else a - b


def _pred(r, method, metric):
    return r["prediction"][method][metric]


def _unmet(r, method):
    return r["resolution"][method]["unmet_units"]


HYPOTHESES = {
    "H1_recall_full_minus_b2": lambda r: paired(_pred(r, "Full", "recall"), _pred(r, "B2", "recall")),
    "H2_mae_full_minus_a1": lambda r: paired(_pred(r, "Full", "mae_days"), _pred(r, "A1", "mae_days")),
    "H3_unmet_full_minus_no_action": lambda r: paired(_unmet(r, "Full"), _unmet(r, "NoAction")),
    "H4_unmet_full_minus_a4": lambda r: paired(_unmet(r, "Full"), _unmet(r, "A4")),
    "H5_unmet_full_minus_b4": lambda r: paired(_unmet(r, "Full"), _unmet(r, "B4")),
    "H6_unmet_full_minus_b3": lambda r: paired(_unmet(r, "Full"), _unmet(r, "B3")),
}
PREDICTED_SIGNS = {"H1_recall_full_minus_b2": 1, "H2_mae_full_minus_a1": -1,
                   "H3_unmet_full_minus_no_action": -1, "H4_unmet_full_minus_a4": -1,
                   "H5_unmet_full_minus_b4": -1, "H6_unmet_full_minus_b3": -1}


def objective(run, key):
    """Realized MILP objective in kCU: plan cost plus the shortage penalty on unmet demand."""
    res = run["resolution"][key]
    return res["cost_k"] + KCU_PER_UNMET * res["unmet_units"]


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


def seed_means(runs, getter):
    groups = {}
    for run in runs:
        value = getter(run)
        if value is not None and np.isfinite(value):
            groups.setdefault(run["seed"], []).append(float(value))
    return np.array([np.mean(v) for _, v in sorted(groups.items())], float)


def t_test_p(means):
    """Two-sided one-sample t p-value; 1.0 when undefined, exact when every mean is equal."""
    if len(means) < 2:
        return 1.0
    if np.ptp(means) == 0:
        return 1.0 if means[0] == 0 else 0.0
    p = float(ttest_1samp(means, 0.0).pvalue)
    return p if np.isfinite(p) else 1.0


def holm_tests(runs, hypotheses, signs):
    """Holm-adjusted paired tests over network means for a fixed family of hypotheses."""
    raw = {name: t_test_p(seed_means(runs, getter)) for name, getter in hypotheses.items()}
    adjusted, running, m = {}, 0.0, len(raw)
    for rank, name in enumerate(sorted(raw, key=lambda k: (raw[k], k))):
        running = max(running, min(1.0, (m - rank) * raw[name]))
        adjusted[name] = running
    out = {}
    for name, getter in hypotheses.items():
        stat = clustered_stats(runs, getter)
        right_sign = stat["mean"] is not None and np.sign(stat["mean"]) == signs[name]
        out[name] = {**stat, "p": raw[name], "p_holm": adjusted[name],
                     "predicted_sign": signs[name],
                     "confirmed": bool(adjusted[name] < 0.05 and right_sign)}
    return out


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


def plan_infos(runs):
    """plan_info of each distinct (network, hazard) plan."""
    return list({(r["seed"], r["hazard"]): r["plan_info"] for r in runs}.values())


def miss_summary(runs):
    """Share of missed stockouts that a perfect event forecast would still miss."""
    def share(sub):
        missed = sum(r["misses"]["missed"] for r in sub)
        delay = sum(r["misses"]["missed_delay"] for r in sub)
        return {"missed": missed, "missed_delay": delay,
                "missed_forecast": missed - delay,
                "delay_share": delay / missed if missed else None}
    return {"all": share(runs),
            "per_hazard": {hz: share([r for r in runs if r["hazard"] == hz])
                           for hz in sorted({r["hazard"] for r in runs})}}


def load_config(path):
    with open(path) as f:
        return json.load(f).get("config", {})


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
                    rf"(mean [descriptive 95\% network-clustered t interval]; {n} records, "
                    rf"{n_seeds} networks)}}")
    lines = [r"\begin{table}[t]", pred_caption,
             r"\label{tab:pred}", r"\centering", r"\small\setlength{\tabcolsep}{2pt}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Method & Precision$^{\dagger}$ & Recall & Brier & MAE $\sigma_v$ (d)\\", r"\midrule"]
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
    precision_note = (
        r"\parbox{\linewidth}{\footnotesize $^{\dagger}$Undefined precision "
        r"(no positive prediction) is excluded; $[n]$ gives the included count "
        r"when below the total.}"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\vspace{2pt}",
        precision_note,
        r"\end{table}",
        "",
    ]

    resolution_caption = (r"\caption{Resolution quality under realized hazards "
                          rf"(mean [descriptive 95\% network-clustered t interval]; {n} records, "
                          rf"{n_seeds} networks)}}")
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
    if all("stci_nonzero" in r for r in runs):
        nonzero = {(r["seed"], r["hazard"]): r["stci_nonzero"] for r in runs}
        top_k = load_config(a.runs).get("top_k", 10)
        summary["stci_saturation"] = {
            "n_plans": len(nonzero),
            "plans_with_at_most_k_nonzero": int(sum(v <= top_k for v in nonzero.values())),
            "mean_nonzero": float(np.mean(list(nonzero.values()))),
            "top_k": top_k,
        }
    if all("misses" in r for r in runs):
        summary["misses"] = miss_summary(runs)
    if all("false_alarm" in r for r in runs):
        summary["false_alarm"] = {
            key: {"unmet_delta": clustered_stats(
                      runs, lambda r, key=key: r["false_alarm"][key]["unmet_delta"]),
                  "cost_k": clustered_stats(
                      runs, lambda r, key=key: r["false_alarm"][key]["cost_k"])}
            for key in runs[0]["false_alarm"]}
    if all("stci_betweenness_dar_spearman" in r for r in runs):
        rho = [r["stci_betweenness_dar_spearman"] for r in
               {(r["seed"], r["hazard"]): r for r in runs}.values()]
        finite = [v for v in rho if v is not None]
        summary["stci_betweenness_dar_spearman"] = {
            "mean": float(np.mean(finite)) if finite else None,
            "min": float(np.min(finite)) if finite else None,
            "n_plans": len(finite)}
    effects = {
        "recall_full_minus_b2": lambda r: paired(
            r["prediction"]["Full"]["recall"], r["prediction"]["B2"]["recall"]),
        "unmet_full_minus_no_action": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["NoAction"]["unmet_units"]),
        "fill_full_minus_no_action": lambda r: paired(
            r["resolution"]["Full"]["fill_rate"], r["resolution"]["NoAction"]["fill_rate"]),
        "unmet_full_minus_a2": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["A2"]["unmet_units"]),
        "unmet_full_minus_a3": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["A3"]["unmet_units"]),
        "unmet_full_minus_b4": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["B4"]["unmet_units"]),
        "unmet_b4_minus_no_action": lambda r: (
            r["resolution"]["B4"]["unmet_units"]
            - r["resolution"]["NoAction"]["unmet_units"]),
        "unmet_a4_minus_no_action": lambda r: (
            r["resolution"]["A4"]["unmet_units"]
            - r["resolution"]["NoAction"]["unmet_units"]),
        "unmet_b4abs_minus_no_action": lambda r: (
            r["resolution"]["B4-abs"]["unmet_units"]
            - r["resolution"]["NoAction"]["unmet_units"]),
        "unmet_full_minus_a4": lambda r: (
            r["resolution"]["Full"]["unmet_units"]
            - r["resolution"]["A4"]["unmet_units"]),
        "mae_full_minus_a1": lambda r: paired(
            r["prediction"]["Full"]["mae_days"], r["prediction"]["A1"]["mae_days"]),
        "cost_full_minus_a4": lambda r: (
            r["resolution"]["Full"]["cost_k"] - r["resolution"]["A4"]["cost_k"]),
        "objective_full_minus_no_action": lambda r: (
            objective(r, "Full") - objective(r, "NoAction")),
        "objective_full_minus_b4": lambda r: objective(r, "Full") - objective(r, "B4"),
        "objective_full_minus_b3": lambda r: objective(r, "Full") - objective(r, "B3"),
    }
    summary["paired_effects"] = {
        name: clustered_stats(runs, getter) for name, getter in effects.items()
    }
    summary["hypotheses"] = holm_tests(runs, HYPOTHESES, PREDICTED_SIGNS)
    statuses = [p["status"] for info in plan_infos(runs) for p in info.values()]
    gaps = [p["gap"] for info in plan_infos(runs) for p in info.values()]
    summary["solver"] = {"n_plans": len(statuses),
                         "n_status_nonzero": int(sum(st != 0 for st in statuses)),
                         "max_gap": float(max(gaps)) if gaps else None}

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
        "latency_seconds_full", "per_hazard", "paired_effects", "hypotheses",
        "stci_topk_changed_with_lambda0", "stci_saturation", "misses", "false_alarm",
        "stci_betweenness_dar_spearman") if k in summary},
        indent=1))


if __name__ == "__main__":
    main()
