"""Run the GeoRipple benchmark and write results/runs.json.

Usage:
    python experiments/run_benchmark.py --seeds 5 --horizon 28
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.evaluate import (
    attributable_stockouts,
    execute_truth,
    prediction_metrics,
    resolution_metrics,
)
from georipple.hazards import Hazard, base_hazards, edge_exposure, forecast_ensemble
from georipple.network import generate
from georipple.predict import (
    ensemble_exposures,
    predict_topology_only,
    predict_wavefront,
    stci,
)
from georipple.resolve import (
    execution_schedule,
    generate_candidates,
    greedy_nearest,
    plan_milp,
)
from georipple.routing import HazardRouter
from georipple.simulate import initial_pipeline

PLAN_QUANTILE = 0.75
LOCK_FILE = Path(__file__).resolve().parents[1] / "requirements.lock"


def locked_packages():
    return tuple(line.split("==", 1)[0] for line in LOCK_FILE.read_text().splitlines()
                 if "==" in line and not line.startswith((" ", "#")))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


def provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip() or None
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=False).stdout.strip())
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in locked_packages()},
            "git_commit": commit, "git_dirty": dirty}


def build_plan(net, ens, exps, p, score, T, router, K, time_limit,
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
    info.update({"plan_seconds": plan_s, "activated": int(y.sum()), "top_k": top})
    return (cand, S_exist, S_cand, y), info


def run(seeds, T, K, time_limit, M, truth_reps=3):
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

            score1 = stci(net, exps, dar, lam=1.0)
            score0 = stci(net, exps, dar, lam=0.0)
            t_r = time.perf_counter()
            router = HazardRouter([m.poly for m in ens])
            router_s = time.perf_counter() - t_r

            info = {}
            plans = {}
            plans["Full"], info["Full"] = build_plan(
                net, ens, exps, p_full, score1, T, router, K, time_limit)
            plans["A2"], info["A2"] = build_plan(
                net, ens, exps, p_full, score1, T, router, K, time_limit,
                hazard_aware=False)
            plans["A3"], info["A3"] = build_plan(
                net, ens, exps, p_full, score0, T, router, K, time_limit)
            p_exp = np.mean([xn.max(1) > 0 for xn, _ in exps], 0)
            plans["B3"] = greedy_nearest(
                net, p_full >= 0.5, p_exp, T,
                max_lanes=info["Full"]["activated"], priority=dar.mean(0))

            latency = pred_seconds + router_s + info["Full"]["plan_seconds"]
            top_changed = sorted(info["Full"]["top_k"]) != sorted(info["A3"]["top_k"])
            calm = Hazard(h.name, h.poly, T + 1, T + 1, 0.0)
            planning_wall = time.perf_counter() - t_start
            for truth_rep in range(truth_reps):
                t_exec = time.perf_counter()
                truth_seed = seed * 10_000 + truth_rep
                truth, _ = execute_truth(net, h, T, truth_seed)
                nohaz, _ = execute_truth(net, calm, T, truth_seed)
                affected = attributable_stockouts(net, truth.sigma, nohaz.sigma, T)
                pred = {
                    "B1": prediction_metrics(net, p_b1, s_b1, truth.sigma, T, affected),
                    "B2": prediction_metrics(net, p_b2, s_b2, truth.sigma, T, affected),
                    "A1": prediction_metrics(net, p_a1, s_a1, truth.sigma, T, affected),
                    "Full": prediction_metrics(net, p_full, s_full, truth.sigma, T, affected),
                }
                res = {"NoAction": resolution_metrics(net, truth, affected, h, T, 0.0)}
                for key, plan in plans.items():
                    cand, Se, Sc, y = plan
                    r, cost = execute_truth(net, h, T, truth_seed, cand, Se, Sc, y,
                                            baseline_shipped=truth.shipped)
                    res[key] = resolution_metrics(net, r, affected, h, T, cost)
                execution_seconds = time.perf_counter() - t_exec
                rec = {
                    "seed": seed, "truth_rep": truth_rep, "hazard": h.name,
                    "prediction": pred, "resolution": res,
                    "plan_info": {k: {kk: vv for kk, vv in v.items() if kk != "top_k"}
                                  for k, v in info.items()},
                    "stci_top_changed_lambda0": top_changed,
                    "latency_seconds_full": latency,
                    "execution_seconds": execution_seconds,
                    "wall_seconds": planning_wall + execution_seconds,
                }
                runs.append(rec)
                print(f"seed={seed} rep={truth_rep} {h.name:24s} "
                      f"affected={res['NoAction']['n_affected']:3d} "
                      f"fill: none={res['NoAction']['fill_rate']:.3f} "
                      f"B3={res['B3']['fill_rate']:.3f} full={res['Full']['fill_rate']:.3f} "
                      f"latency={latency:.1f}s", flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=positive_int, default=5)
    ap.add_argument("--horizon", type=positive_int, default=28)
    ap.add_argument("--top-k", type=positive_int, default=10)
    ap.add_argument("--members", type=positive_int, default=8)
    ap.add_argument("--truth-reps", type=positive_int, default=3)
    ap.add_argument("--time-limit", type=positive_float, default=60.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "runs.json"))
    a = ap.parse_args()
    runs = run(a.seeds, a.horizon, a.top_k, a.time_limit, a.members, a.truth_reps)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    config = {k: v for k, v in vars(a).items() if k != "out"}
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"config": config, "provenance": provenance(), "runs": runs},
                  f, indent=1, default=float, allow_nan=False)
    os.replace(tmp, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
