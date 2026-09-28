"""Render the paper's map figures from one benchmark run.

Wavefront figure: forecast footprints, exposed lanes, and dealers
shaded by predicted stockout day.
Resolution figure: activated recovery lanes and each dealer's change in
fill rate under the plan versus no action (ground-truth execution).

State outlines: data/us-states.json (PublicaMundi/MappingAPI, derived from
U.S. Census boundaries).

Usage: python experiments/make_figures.py --seed 0 --hazard 0
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from run_benchmark import PLAN_QUANTILE

from georipple.evaluate import execute_truth
from georipple.geo import project
from georipple.hazards import (
    base_hazards,
    edge_exposure,
    forecast_ensemble,
)
from georipple.network import DC, SUP, generate
from georipple.predict import ensemble_exposures, predict_wavefront, stci
from georipple.resolve import (
    execution_schedule,
    generate_candidates,
    plan_milp,
)
from georipple.routing import HazardRouter
from georipple.simulate import initial_pipeline

plt.rcParams.update({"font.family": "serif", "font.size": 8})


def draw_states(ax):
    with open(os.path.join(HERE, "..", "data", "us-states.json")) as f:
        gj = json.load(f)
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            x, y = project(ring[:, 0], ring[:, 1])
            ax.fill(x, y, facecolor="#f4f4f1", edgecolor="#b9b9b3", lw=0.4, zorder=0)


def draw_poly(ax, poly, **kw):
    for p in getattr(poly, "geoms", [poly]):
        x, y = p.exterior.xy
        ax.plot(x, y, **kw)


def frame(ax, net, lon=(-100.5, -87.5), lat=(26.0, 34.5)):
    x0, y0 = project(lon[0], lat[0])
    x1, y1 = project(lon[1], lat[1])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#888888"); s.set_linewidth(0.5)


def facilities(ax, net):
    s = net.kind == SUP
    d = net.kind == DC
    ax.scatter(*net.xy[s].T, marker="^", s=34, c="#333333", zorder=5, lw=0)
    ax.scatter(*net.xy[d].T, marker="s", s=16, c="#555555", zorder=5, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hazard", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "paper", "figures"))
    a = ap.parse_args()
    T, seed = a.horizon, a.seed
    h = base_hazards()[a.hazard]
    net = generate(seed)
    ens = forecast_ensemble(h, np.random.default_rng([seed, a.hazard]), M=8)
    exps = ensemble_exposures(net, ens, T)
    p, sig, dar = predict_wavefront(net, exps, T)
    xe_mean = np.mean([xe.max(1) for _, xe in exps], 0)
    dl = net.dealers
    os.makedirs(a.outdir, exist_ok=True)

    # ---------- Fig: wavefront ----------
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    draw_states(ax)
    for m in ens:
        draw_poly(ax, m.poly, color="#1f5fa8", lw=0.5, alpha=0.45, zorder=1)
    for e in np.where(xe_mean >= 0.5)[0]:
        x, y = net.geoms[e].xy
        ax.plot(x, y, ls=(0, (3, 2)), color="#c0392b", lw=0.7, zorder=2)
    facilities(ax, net)
    pred = dl[p[dl] >= 0.5]
    rest = dl[p[dl] < 0.5]
    ax.scatter(*net.xy[rest].T, s=5, c="#bbbbbb", zorder=3, lw=0)
    sc = ax.scatter(*net.xy[pred].T, s=11, c=sig[pred], cmap="magma_r",
                    vmin=0, vmax=14, zorder=4, lw=0.2, edgecolors="k")
    frame(ax, net)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("Predicted stockout day", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.legend(handles=[
        Line2D([], [], color="#1f5fa8", lw=0.8, label="Forecast members"),
        Line2D([], [], color="#c0392b", ls=(0, (3, 2)), lw=0.8, label="Exposed lane"),
        Line2D([], [], marker="^", ls="", color="#333333", label="Supplier"),
        Line2D([], [], marker="s", ls="", ms=4, color="#555555", label="DC"),
        Line2D([], [], marker="o", ls="", ms=3, color="#bbbbbb", label="Dealer, not at risk")],
        loc="lower right", fontsize=5.5, framealpha=0.9)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.outdir, f"fig_wavefront.{ext}"), dpi=300)
    plt.close(fig)

    # ---------- plan (same settings as the benchmark's full method) ----------
    p_exp = np.mean([xn.max(1) > 0 for xn, _ in exps], 0)
    router = HazardRouter([m.poly for m in ens])
    cand, _ = generate_candidates(net, p >= 0.5, stci(net, exps, dar), p_exp, router=router, K=10)
    q = PLAN_QUANTILE
    xn = np.quantile([x for x, _ in exps], q, axis=0)
    xe = np.quantile([x for _, x in exps], q, axis=0)
    xc = np.quantile([edge_exposure(cand.geoms, m, T) for m in ens], q, axis=0)
    fe, fc, y, _ = plan_milp(net, cand, xn, xe, xc, T, initial_pipeline(net, T))
    Se, Sc = execution_schedule(net, fe, fc, y, T)
    r_none, _ = execute_truth(net, h, T, seed)
    r_plan, cost = execute_truth(net, h, T, seed, cand, Se, Sc, y,
                                 baseline_shipped=r_none.shipped)
    gain = (r_plan.served[dl].sum(1) - r_none.served[dl].sum(1)) / (net.demand[dl] * T)

    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    draw_states(ax)
    draw_poly(ax, h.poly, color="#1f5fa8", lw=0.9, zorder=1)
    colors = {"lateral": "#1e8449", "alt_source": "#7d3c98"}
    for i in np.where(y)[0]:
        x_, y_ = cand.geoms[i].xy
        ax.plot(x_, y_, color=colors[cand.kind[i]], lw=1.1, zorder=3)
    facilities(ax, net)
    order = np.argsort(np.abs(gain))
    sc = ax.scatter(*net.xy[dl[order]].T, s=9, c=100 * gain[order], cmap="RdBu",
                    vmin=-15, vmax=15, zorder=4, lw=0.2, edgecolors="#444444")
    frame(ax, net)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("Fill-rate change vs. no action (pp)", fontsize=7); cb.ax.tick_params(labelsize=6)
    used = {cand.kind[i] for i in np.where(y)[0]}
    labels = {"lateral": "Lateral transshipment", "alt_source": "Alternate sourcing"}
    ax.legend(handles=[Line2D([], [], color="#1f5fa8", lw=0.9, label="Realized footprint")] +
              [Line2D([], [], color=colors[k], lw=1.1, label=labels[k]) for k in colors if k in used],
              loc="lower right", fontsize=5.5, framealpha=0.9)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.outdir, f"fig_resolution.{ext}"), dpi=300)
    plt.close(fig)

    stats = {"seed": seed, "hazard": h.name, "n_predicted": len(pred),
             "n_exposed_lanes": int((xe_mean >= 0.5).sum()),
             "activated": {k: int(sum(1 for i in np.where(y)[0] if cand.kind[i] == k))
                           for k in colors},
             "cost_k": cost / 1000, "unmet_none": float((net.demand[dl, None] - r_none.served[dl]).clip(0).sum()),
             "unmet_plan": float((net.demand[dl, None] - r_plan.served[dl]).clip(0).sum()),
             "dealers_improved": int((gain > 0.005).sum()), "dealers_worse": int((gain < -0.005).sum())}
    with open(os.path.join(a.outdir, "figure_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
