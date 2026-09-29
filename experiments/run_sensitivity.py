"""One-factor-at-a-time sensitivity study for the full GeoRipple planner."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from run_benchmark import build_plan, positive_float, positive_int, provenance, realized_hazard

from georipple.evaluate import (
    attributable_stockouts,
    execute_truth,
    prediction_metrics,
    resolution_metrics,
)
from georipple.hazards import Hazard, base_hazards, forecast_ensemble
from georipple.network import generate
from georipple.predict import ensemble_exposures, predict_wavefront, stci
from georipple.routing import HazardRouter


def sensitivity_configs():
    return [
        {"id": "q50", "quantile": 0.50, "top_k": 10, "members": 8},
        {"id": "base", "quantile": 0.75, "top_k": 10, "members": 8},
        {"id": "q90", "quantile": 0.90, "top_k": 10, "members": 8},
        {"id": "k5", "quantile": 0.75, "top_k": 5, "members": 8},
        {"id": "k15", "quantile": 0.75, "top_k": 15, "members": 8},
        {"id": "m4", "quantile": 0.75, "top_k": 10, "members": 4},
        {"id": "m16", "quantile": 0.75, "top_k": 10, "members": 16},
    ]


def run(seeds, T, time_limit, truth_reps):
    records = []
    for config in sensitivity_configs():
        for seed in range(seeds):
            net = generate(seed)
            for hi, base in enumerate(base_hazards()):
                ens = forecast_ensemble(
                    base, np.random.default_rng([seed, hi]), M=config["members"]
                )
                exps = ensemble_exposures(net, ens, T)
                p, sig, dar = predict_wavefront(net, exps, T)
                score = stci(net, exps, dar)
                router = HazardRouter([member.poly for member in ens])
                plan, info = build_plan(
                    net,
                    ens,
                    exps,
                    p,
                    score,
                    T,
                    router,
                    config["top_k"],
                    time_limit,
                    quantile=config["quantile"],
                )
                cand, existing, candidate, activated = plan
                h_real = realized_hazard(base, seed, hi)
                calm = Hazard(base.name, h_real.poly, T + 1, T + 1, 0.0)
                for truth_rep in range(truth_reps):
                    truth_seed = seed * 10_000 + truth_rep
                    truth, _ = execute_truth(net, h_real, T, truth_seed)
                    nohaz, _ = execute_truth(net, calm, T, truth_seed)
                    affected = attributable_stockouts(net, truth.sigma, nohaz.sigma, T)
                    planned, cost = execute_truth(
                        net,
                        h_real,
                        T,
                        truth_seed,
                        cand,
                        existing,
                        candidate,
                        activated,
                        baseline_shipped=truth.shipped,
                    )
                    records.append(
                        {
                            "config": config["id"],
                            "seed": seed,
                            "truth_rep": truth_rep,
                            "hazard": base.name,
                            "prediction": prediction_metrics(
                                net, p, sig, truth.sigma, T, affected
                            ),
                            "no_action": resolution_metrics(
                                net, truth, affected, h_real, T, 0.0
                            ),
                            "full": resolution_metrics(
                                net, planned, affected, h_real, T, cost
                            ),
                            "plan_info": {
                                key: value for key, value in info.items() if key != "top_k"
                            },
                        }
                    )
                print(
                    f"config={config['id']:4s} seed={seed} hazard={base.name:24s} "
                    f"actions={info['activated']:2d}",
                    flush=True,
                )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=positive_int, default=5)
    ap.add_argument("--horizon", type=positive_int, default=28)
    ap.add_argument("--truth-reps", type=positive_int, default=3)
    ap.add_argument("--time-limit", type=positive_float, default=60.0)
    ap.add_argument(
        "--out", default=os.path.join(HERE, "..", "results", "sensitivity_runs.json")
    )
    args = ap.parse_args()
    records = run(args.seeds, args.horizon, args.time_limit, args.truth_reps)
    payload = {
        "config": {key: value for key, value in vars(args).items() if key != "out"},
        "configs": sensitivity_configs(),
        "provenance": provenance(),
        "runs": records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        json.dump(payload, f, indent=1, default=float, allow_nan=False)
    os.replace(args.out + ".tmp", args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
