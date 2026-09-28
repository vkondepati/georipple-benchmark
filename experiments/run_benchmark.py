"""Run the GeoRipple benchmark and write results/runs.json.

Usage:
    python experiments/run_benchmark.py --seeds 5 --horizon 28
"""
import argparse
from importlib.metadata import version
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.network import generate, DEALER                       # noqa: E402
from georipple.hazards import Hazard, base_hazards, forecast_ensemble, edge_exposure  # noqa: E402
from georipple.predict import (ensemble_exposures, predict_wavefront,  # noqa: E402
                               predict_topology_only, stci)
from georipple.resolve import (generate_candidates, plan_milp,        # noqa: E402
                               execution_schedule, greedy_nearest)
from georipple.routing import HazardRouter                             # noqa: E402
from georipple.simulate import initial_pipeline                        # noqa: E402
from georipple.evaluate import (execute_truth, prediction_metrics,     # noqa: E402
                                resolution_metrics, attributable_stockouts)


PLAN_QUANTILE = 0.75
PACKAGES = ("numpy", "scipy", "shapely", "networkx")


def provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip() or None
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=False).stdout.strip())
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in PACKAGES},
            "git_commit": commit, "git_dirty": dirty}


def plan_and_execute(net, ens, exps, p, score, h, T, seed, router, K, time_limit,
                     hazard_aware=True):
    t0 = time.perf_counter()
    p_exp = np.mean([xn.max(1) > 0 for xn, _ in exps], 0)
    cand, top = generate_candidates(net, p >= 0.5, score, p_exp,
                                    router=router if hazard_aware else None, K=K)
    # risk-averse planning exposure: upper quartile across forecast members
    q = PLAN_QUANTILE
    xn = np.quantile([a for a, _ in exps], q, axis=0)
    xe = np.quantile([b for _, b in exps], q, axis=0)
    if len(cand) and hazard_aware:
        xc = np.quantile([edge_exposure(cand.geoms, m, T) for m in ens], q, axis=0)
    else:
        xc = np.zeros((len(cand), T))       # A2: candidate lanes assumed clear
    fe, fc, y, info = plan_milp(net, cand, xn, xe, xc, T, initial_pipeline(net, T),
                                time_limit=time_limit)
    plan_s = time.perf_counter() - t0
    S_exist, S_cand = execution_schedule(net, fe, fc, y, T)
    r, cost = execute_truth(net, h, T, seed, cand, S_exist, S_cand, y)
    info.update({"plan_seconds": plan_s, "activated": int(y.sum()), "top_k": top})
    return r, cost, info


def run(seeds, T, K, time_limit, M):
    runs = []
    for seed in range(seeds):
        for hi, h in enumerate(base_hazards()):
            t_start = time.perf_counter()
            net = generate(seed)
            ens = forecast_ensemble(h, np.random.default_rng([seed, hi]), M=M)
            exps = ensemble_exposures(net, ens, T)

            t_pred = time.perf_counter()
            p_full, s_full, dar = predict_wavefront(net, exps, T)
            pred_seconds = time.perf_counter() - t_pred
            p_b2, s_b2, _ = predict_wavefront(net, exps, T, use_edges=False)
            p_a1, s_a1, _ = predict_wavefront(net, exps, T, use_buffers=False)
            p_b1, s_b1 = predict_topology_only(net, ens, T)

            truth, _ = execute_truth(net, h, T, seed)
            calm = Hazard(h.name, h.poly, T + 1, T + 1, 0.0)      # same delays, no hazard
            nohaz, _ = execute_truth(net, calm, T, seed)
            affected = attributable_stockouts(net, truth.sigma, nohaz.sigma, T)
            pred = {
                "B1": prediction_metrics(net, p_b1, s_b1, truth.sigma, T, affected),
                "B2": prediction_metrics(net, p_b2, s_b2, truth.sigma, T, affected),
                "A1": prediction_metrics(net, p_a1, s_a1, truth.sigma, T, affected),
                "Full": prediction_metrics(net, p_full, s_full, truth.sigma, T, affected),
            }

            res = {"NoAction": resolution_metrics(net, truth, affected, h, T, 0.0)}
            score1 = stci(net, exps, dar, lam=1.0)
            score0 = stci(net, exps, dar, lam=0.0)
            t_r = time.perf_counter()
            router = HazardRouter([m.poly for m in ens])
            router_s = time.perf_counter() - t_r

            info = {}
            r, c, info["Full"] = plan_and_execute(net, ens, exps, p_full, score1, h, T, seed,
                                                  router, K, time_limit)
            res["Full"] = resolution_metrics(net, r, affected, h, T, c)
            r, c, info["A2"] = plan_and_execute(net, ens, exps, p_full, score1, h, T, seed,
                                                router, K, time_limit, hazard_aware=False)
            res["A2"] = resolution_metrics(net, r, affected, h, T, c)
            r, c, info["A3"] = plan_and_execute(net, ens, exps, p_full, score0, h, T, seed,
                                                router, K, time_limit)
            res["A3"] = resolution_metrics(net, r, affected, h, T, c)
            p_exp = np.mean([xn.max(1) > 0 for xn, _ in exps], 0)
            cand, Se, Sc, y = greedy_nearest(net, p_full >= 0.5, p_exp, T)
            r, c = execute_truth(net, h, T, seed, cand, Se, Sc, y)
            res["B3"] = resolution_metrics(net, r, affected, h, T, c)

            latency = pred_seconds + router_s + info["Full"]["plan_seconds"]
            rec = {"seed": seed, "hazard": h.name, "prediction": pred, "resolution": res,
                   "plan_info": {k: {kk: vv for kk, vv in v.items() if kk != "top_k"}
                                 for k, v in info.items()},
                   "stci_top_changed_lambda0": sorted(info["Full"]["top_k"]) != sorted(info["A3"]["top_k"]),
                   "latency_seconds_full": latency,
                   "wall_seconds": time.perf_counter() - t_start}
            runs.append(rec)
            print(f"seed={seed} {h.name:24s} affected={res['NoAction']['n_affected']:3d} "
                  f"fill: none={res['NoAction']['fill_rate']:.3f} B3={res['B3']['fill_rate']:.3f} "
                  f"full={res['Full']['fill_rate']:.3f}  latency={latency:.1f}s", flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "runs.json"))
    a = ap.parse_args()
    runs = run(a.seeds, a.horizon, a.top_k, a.time_limit, a.members)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    config = {k: v for k, v in vars(a).items() if k != "out"}
    with open(a.out, "w") as f:
        json.dump({"config": config, "provenance": provenance(), "runs": runs},
                  f, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
