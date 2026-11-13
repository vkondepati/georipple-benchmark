"""Run the GeoRipple benchmark and write results/runs.json.

Usage:
    python experiments/run_benchmark.py --seeds 10 --horizon 28
    python experiments/run_benchmark.py --seeds 10 --seed-offset 10 --lanes road --dealer-dcs 2
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.evaluate import (
    attributable_stockouts,
    classify_misses,
    execute_truth,
    prediction_metrics,
    resolution_metrics,
    unmet_units,
)
from georipple.hazards import Hazard, base_hazards, edge_exposure, forecast_ensemble
from georipple.lanes import RoadLanes, StraightLanes, enrich_lanes
from georipple.network import DC, generate
from georipple.predict import (
    ensemble_exposures,
    predict_topology_only,
    predict_wavefront,
    stci,
    stci_components,
)
from georipple.resolve import (
    additive_schedule,
    execution_schedule,
    generate_candidates,
    greedy_nearest,
    plan_milp,
)
from georipple.roads import load_roads
from georipple.simulate import initial_pipeline

PLAN_QUANTILE = 0.75
DEALER_SHARES = {1: (1.0,), 2: (0.7, 0.3)}
LANE_MODELS = ("straight", "road")
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


def lane_model(name):
    if name == "straight":
        return StraightLanes()
    if name == "road":
        return RoadLanes(load_roads())
    raise ValueError(f"unknown lane model {name!r}")


def build_plan(net, ens, exps, p, score, T, router, K, time_limit,
               hazard_aware=True, quantile=PLAN_QUANTILE, free_router=None):
    """Solve one MILP and return its plan under both dispatch rules.

    ``router`` keeps candidate lanes clear of the forecast hazard; ``free_router`` draws
    them with no hazard check for ablation A2 (straight lanes if None). Returns
    ({"additive": plan, "absolute": plan}, info); a plan is (candidates, existing-lane
    schedule, candidate schedule, activation).
    """
    t0 = time.perf_counter()
    p_exp = np.mean([xn.max(1) > 0 for xn, _ in exps], 0)
    cand, top = generate_candidates(net, p >= 0.5, score, p_exp,
                                    router=router if hazard_aware else free_router, K=K)
    node_members = [a for a, _ in exps]
    edge_members = [b for _, b in exps]
    xn = np.mean(node_members, axis=0) if quantile is None else \
        np.quantile(node_members, quantile, axis=0)
    xe = np.mean(edge_members, axis=0) if quantile is None else \
        np.quantile(edge_members, quantile, axis=0)
    if len(cand) and hazard_aware:
        candidate_members = [edge_exposure(cand.geoms, m, T) for m in ens]
        xc = np.mean(candidate_members, axis=0) if quantile is None else \
            np.quantile(candidate_members, quantile, axis=0)
    else:
        xc = np.zeros((len(cand), T))       # A2: candidate lanes assumed clear
    fe, fc, y, info = plan_milp(net, cand, xn, xe, xc, T, initial_pipeline(net, T),
                                time_limit=time_limit)
    plans = {"additive": (cand, *additive_schedule(net, fe, fc, y, T), y),
             "absolute": (cand, *execution_schedule(net, fe, fc, y, T), y)}
    info.update({"plan_seconds": time.perf_counter() - t0, "activated": int(y.sum()),
                 "top_k": top, "exposure_aggregation": "mean" if quantile is None else quantile})
    return plans, info


def realized_hazard(base, seed, hazard_index):
    """Independent held-out event from the same family as the forecast."""
    rng = np.random.default_rng([seed, hazard_index, 947])
    return forecast_ensemble(base, rng, M=1)[0]


def betweenness_dar_spearman(net, exps, dar):
    """Rank correlation of betweenness and downstream DaR among exposed DCs.

    Suppliers are sources and dealers are sinks, so in this layered network only DCs
    have nonzero betweenness; this asks whether it adds anything beyond DaR there.
    """
    xhat, down, betw = stci_components(net, exps, dar)
    exposed = (xhat > 0) & (net.kind == DC)
    if exposed.sum() < 3 or np.ptp(betw[exposed]) == 0 or np.ptp(down[exposed]) == 0:
        return None
    return float(spearmanr(betw[exposed], down[exposed]).statistic)


FALSE_ALARM_KEYS = ("Full", "A4", "B4")


def run(seeds, T, K, time_limit, M, truth_reps=3, seed_offset=0, lanes="straight",
        dealer_dcs=1):
    model = lane_model(lanes)
    runs = []
    for seed in range(seed_offset, seed_offset + seeds):
        net = enrich_lanes(generate(seed, dealer_shares=DEALER_SHARES[dealer_dcs]), model)
        for hi, h in enumerate(base_hazards()):
            t_start = time.perf_counter()
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
            router = model.avoiding([m.poly for m in ens])
            router_s = time.perf_counter() - t_r

            info, plans = {}, {}
            full, info["Full"] = build_plan(net, ens, exps, p_full, score1, T, router, K,
                                            time_limit, free_router=model)
            plans["Full"], plans["A4"] = full["additive"], full["absolute"]
            a2, info["A2"] = build_plan(net, ens, exps, p_full, score1, T, router, K,
                                        time_limit, hazard_aware=False, free_router=model)
            plans["A2"] = a2["additive"]
            a3, info["A3"] = build_plan(net, ens, exps, p_full, score0, T, router, K,
                                        time_limit, free_router=model)
            plans["A3"] = a3["additive"]
            b4, info["B4"] = build_plan(net, ens, exps, p_full, score1, T, router, K,
                                        time_limit, quantile=None, free_router=model)
            plans["B4"], plans["B4-abs"] = b4["additive"], b4["absolute"]
            p_exp = np.mean([xn.max(1) > 0 for xn, _ in exps], 0)
            plans["B3"] = greedy_nearest(
                net, p_full >= 0.5, p_exp, T,
                max_lanes=info["Full"]["activated"], priority=dar.mean(0), router=model)

            latency = pred_seconds + router_s + info["Full"]["plan_seconds"]
            top_changed = sorted(info["Full"]["top_k"]) != sorted(info["A3"]["top_k"])
            b_dar = betweenness_dar_spearman(net, exps, dar)
            h_real = realized_hazard(h, seed, hi)
            calm = Hazard(h.name, h_real.poly, T + 1, T + 1, 0.0)
            _, replay_sigma, _ = predict_wavefront(net, ensemble_exposures(net, [h_real], T), T)
            replay = replay_sigma[net.dealers] < T
            planning_wall = time.perf_counter() - t_start
            for truth_rep in range(truth_reps):
                t_exec = time.perf_counter()
                truth_seed = seed * 10_000 + truth_rep
                truth, _ = execute_truth(net, h_real, T, truth_seed)
                nohaz, _ = execute_truth(net, calm, T, truth_seed)
                affected = attributable_stockouts(net, truth.sigma, nohaz.sigma, T)
                pred = {
                    "B1": prediction_metrics(net, p_b1, s_b1, truth.sigma, T, affected),
                    "B2": prediction_metrics(net, p_b2, s_b2, truth.sigma, T, affected),
                    "A1": prediction_metrics(net, p_a1, s_a1, truth.sigma, T, affected),
                    "Full": prediction_metrics(net, p_full, s_full, truth.sigma, T, affected),
                }
                res = {"NoAction": resolution_metrics(net, truth, affected, h_real, T, 0.0)}
                for key, plan in plans.items():
                    cand, Se, Sc, y = plan
                    r, cost = execute_truth(net, h_real, T, truth_seed, cand, Se, Sc, y,
                                            baseline_shipped=truth.shipped)
                    res[key] = resolution_metrics(net, r, affected, h_real, T, cost)
                false_alarm = {}                 # the same plans if the event never comes
                for key in FALSE_ALARM_KEYS:
                    cand, Se, Sc, y = plans[key]
                    r, cost = execute_truth(net, calm, T, truth_seed, cand, Se, Sc, y,
                                            baseline_shipped=nohaz.shipped)
                    false_alarm[key] = {"unmet_delta": unmet_units(net, r) - unmet_units(net, nohaz),
                                        "cost_k": cost / 1000.0}
                execution_seconds = time.perf_counter() - t_exec
                rec = {
                    "seed": seed, "truth_rep": truth_rep, "hazard": h.name,
                    "condition": {"lanes": lanes, "dealer_dcs": dealer_dcs},
                    "prediction": pred, "resolution": res, "false_alarm": false_alarm,
                    "misses": classify_misses(affected, p_full[net.dealers] >= 0.5, replay),
                    "plan_info": {k: {kk: vv for kk, vv in v.items() if kk != "top_k"}
                                  for k, v in info.items()},
                    "stci_top_changed_lambda0": top_changed,
                    "stci_nonzero": int((score1 > 0).sum()),
                    "stci_betweenness_dar_spearman": b_dar,
                    "latency_seconds_full": latency,
                    "execution_seconds": execution_seconds,
                    "wall_seconds": planning_wall + execution_seconds,
                }
                runs.append(rec)
                print(f"{lanes}/{dealer_dcs} seed={seed} rep={truth_rep} {h.name:24s} "
                      f"affected={res['NoAction']['n_affected']:3d} "
                      f"unmet: none={res['NoAction']['unmet_units']:.0f} "
                      f"A4={res['A4']['unmet_units']:.0f} B4={res['B4']['unmet_units']:.0f} "
                      f"full={res['Full']['unmet_units']:.0f} "
                      f"latency={latency:.1f}s", flush=True)
    return runs


def write_json(payload, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, default=float, allow_nan=False)
    os.replace(tmp, path)
    print("wrote", path)


def benchmark_arguments(ap):
    ap.add_argument("--seeds", type=positive_int, default=10)
    ap.add_argument("--seed-offset", type=nonnegative_int, default=0)
    ap.add_argument("--horizon", type=positive_int, default=28)
    ap.add_argument("--top-k", type=positive_int, default=10)
    ap.add_argument("--members", type=positive_int, default=8)
    ap.add_argument("--truth-reps", type=positive_int, default=3)
    ap.add_argument("--time-limit", type=positive_float, default=60.0)


def main():
    ap = argparse.ArgumentParser()
    benchmark_arguments(ap)
    ap.add_argument("--lanes", choices=LANE_MODELS, default="straight")
    ap.add_argument("--dealer-dcs", type=int, choices=sorted(DEALER_SHARES), default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "runs.json"))
    a = ap.parse_args()
    runs = run(a.seeds, a.horizon, a.top_k, a.time_limit, a.members, a.truth_reps,
               a.seed_offset, a.lanes, a.dealer_dcs)
    config = {k: v for k, v in vars(a).items() if k != "out"}
    config["truth_hazard"] = "independent held-out perturbation per seed and family"
    write_json({"config": config, "provenance": provenance(), "runs": runs}, a.out)


if __name__ == "__main__":
    main()
