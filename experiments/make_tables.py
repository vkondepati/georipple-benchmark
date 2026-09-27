"""Aggregate results/runs.json into results/summary.json and results/tables.tex.

Values are mean (standard deviation) over all seed x hazard runs.
Precision is averaged only over runs where the method predicted at least
one stockout; the number of such runs is reported alongside.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
PRED_ROWS = [("B1", "B1 Topology-only"), ("B2", "B2 Node-only spatial"),
             ("A1", "A1 No buffers"), ("Full", "GeoRipple (full)")]
RES_ROWS = [("NoAction", "No action"), ("B3", "B3 Greedy nearest"),
            ("A2", "A2 No hazard routing"), ("A3", r"A3 $\lambda=0$"),
            ("Full", "GeoRipple (full)")]


def ms(vals, fmt):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return "--", np.nan, np.nan, 0
    return f"{fmt.format(v.mean())} ({fmt.format(v.std())})", float(v.mean()), float(v.std()), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "results", "runs.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "results"))
    a = ap.parse_args()
    data = json.load(open(a.runs))
    runs = data["runs"]
    n = len(runs)
    summary = {"n_runs": n, "prediction": {}, "resolution": {}}

    lines = [r"\begin{table}[t]",
             r"\caption{Stockout prediction versus realized outcome "
             rf"(mean (s.d.) over {n} runs)}}",
             r"\label{tab:pred}", r"\centering", r"\small\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Method & Prec.$^{\dagger}$ & Recall & MAE $\sigma_v$ (d)\\", r"\midrule"]
    for key, label in PRED_ROWS:
        p = ms([r["prediction"][key]["precision"] for r in runs], "{:.2f}")
        rc = ms([r["prediction"][key]["recall"] for r in runs], "{:.2f}")
        mae = ms([r["prediction"][key]["mae_days"] for r in runs], "{:.1f}")
        summary["prediction"][key] = {"precision": p[1:], "recall": rc[1:], "mae_days": mae[1:]}
        ptxt = p[0] if p[3] == n else f"{p[0]}$^{{[{p[3]}]}}$"
        lines.append(f"{label} & {ptxt} & {rc[0]} & {mae[0]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{2pt}",
              r"\parbox{\linewidth}{\footnotesize $^{\dagger}$Averaged over runs with at least one "
              r"predicted stockout; bracketed superscript gives that count when fewer than all runs.}",
              r"\end{table}", ""]

    lines += [r"\begin{table}[t]",
              r"\caption{Resolution quality under realized hazards "
              rf"(mean (s.d.) over {n} runs)}}",
              r"\label{tab:resolve}", r"\centering", r"\small\setlength{\tabcolsep}{2.5pt}",
              r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
              r"Method & Fill rate & Unmet (u) & TTR$_{95}$ (d) & Cost (\$k)\\", r"\midrule"]
    for key, label in RES_ROWS:
        fr = ms([r["resolution"][key]["fill_rate"] for r in runs], "{:.3f}")
        um = ms([r["resolution"][key]["unmet_units"] for r in runs], "{:.0f}")
        tt = ms([r["resolution"][key]["ttr95_days"] for r in runs], "{:.1f}")
        co = ms([r["resolution"][key]["cost_k"] for r in runs], "{:.1f}")
        summary["resolution"][key] = {"fill_rate": fr[1:], "unmet_units": um[1:],
                                      "ttr95_days": tt[1:], "cost_k": co[1:]}
        lines.append(f"{label} & {fr[0]} & {um[0]} & {tt[0]} & {co[0]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    lat = [r["latency_seconds_full"] for r in runs]
    summary["latency_seconds_full"] = [float(np.mean(lat)), float(np.max(lat))]
    summary["per_hazard"] = {}
    for hz in sorted({r["hazard"] for r in runs}):
        sub = [r for r in runs if r["hazard"] == hz]
        summary["per_hazard"][hz] = {
            "recall": {k: float(np.nanmean([r["prediction"][k]["recall"] for r in sub]))
                       for k, _ in PRED_ROWS},
            "fill_rate": {k: float(np.nanmean([r["resolution"][k]["fill_rate"] for r in sub]))
                          for k, _ in RES_ROWS},
            "unmet_units": {k: float(np.mean([r["resolution"][k]["unmet_units"] for r in sub]))
                            for k, _ in RES_ROWS},
        }
    summary["stci_topk_changed_with_lambda0"] = int(sum(r["stci_top_changed_lambda0"] for r in runs))

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "tables.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(lines))
    print(json.dumps({k: summary[k] for k in ("latency_seconds_full", "per_hazard",
                                              "stci_topk_changed_with_lambda0")}, indent=1))


if __name__ == "__main__":
    main()
