"""Validate checked-in JSON and deterministic table artifacts."""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def strict_json(path):
    with path.open() as f:
        return json.load(
            f,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value} in {path}")
            ),
        )


def unsolved_plans(runs):
    """Plan keys whose MILP did not reach the optimality gap (status != 0)."""
    bad = set()
    for run in runs:
        info = run.get("plan_info", {})
        plans = {"plan": info} if "status" in info else info
        for key, plan in plans.items():
            if isinstance(plan, dict) and plan.get("status", 0) != 0:
                bad.add(key)
    return sorted(bad)


def validate_runs(payload, path, n_configs=1):
    runs = payload.get("runs", [])
    config = payload.get("config", {})
    n_hazards = len({run.get("hazard") for run in runs})
    expected = (
        config.get("seeds", 0)
        * config.get("truth_reps", 0)
        * n_hazards
        * n_configs
    )
    if not runs or len(runs) != expected:
        raise SystemExit(f"incomplete artifact: {path}; expected {expected} records")
    unsolved = unsolved_plans(runs)
    if unsolved:
        raise SystemExit(f"MILP not solved to the optimality gap in {path}: {unsolved}")
    if payload.get("provenance", {}).get("git_dirty") is not False:
        raise SystemExit(f"non-clean provenance: {path}; regenerate from a clean commit")


def equivalent_json(left, right):
    """Compare generated summaries while tolerating platform-level float noise."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equivalent_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equivalent_json(a, b) for a, b in zip(left, right)
        )
    numeric = (int, float)
    if (
        isinstance(left, numeric)
        and not isinstance(left, bool)
        and isinstance(right, numeric)
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    return left == right


CONDITIONS = ("C0", "C1", "C2", "C3")


def regenerated(tmp):
    """(script args, generated dir, checked-in dir, file names) for every derived artifact."""
    results = ROOT / "results"
    jobs = [
        (["make_tables.py", "--runs", str(results / "runs.json")], "dev", results,
         ("tables.tex", "summary.json")),
        (["make_sensitivity.py", "--runs", str(results / "sensitivity_runs.json")], "sens",
         results, ("sensitivity.tex", "sensitivity_summary.json")),
        (["make_conditions.py", "--confirm-dir", str(results / "confirm")], "cond", results,
         ("conditions.tex", "conditions_summary.json")),
    ]
    for cid in CONDITIONS:
        jobs.append((["make_tables.py", "--runs", str(results / "confirm" / f"{cid}.json")],
                     f"confirm_{cid}", results / "confirm" / cid,
                     ("tables.tex", "summary.json")))
    for args, sub, expected_dir, names in jobs:
        out = Path(tmp) / sub
        subprocess.run([sys.executable, str(ROOT / "experiments" / args[0]), *args[1:],
                        "--outdir", str(out)], check=True, stdout=subprocess.DEVNULL)
        for name in names:
            yield expected_dir / name, out / name


def main():
    results = ROOT / "results"
    runs = [results / "runs.json", results / "sensitivity_runs.json"]
    runs += [results / "confirm" / f"{cid}.json" for cid in CONDITIONS]
    summaries = [results / "summary.json", results / "sensitivity_summary.json",
                 results / "conditions_summary.json",
                 *(results / "confirm" / cid / "summary.json" for cid in CONDITIONS)]
    others = [ROOT / "paper" / "figures" / "figure_stats.json", ROOT / "data" / "us-states.json"]
    payloads = {path: strict_json(path) for path in runs + summaries + others}
    for path in runs:
        payload = payloads[path]
        validate_runs(payload, path, max(1, len(payload.get("configs", []))))

    sys.path.insert(0, str(ROOT))
    from georipple.roads import load_roads
    if not load_roads().is_connected():
        raise SystemExit("data/us-primary-roads.json.gz is not one connected road network")

    with tempfile.TemporaryDirectory() as tmp:
        for expected, generated in regenerated(tmp):
            if expected.suffix == ".json":
                matches = equivalent_json(strict_json(expected), strict_json(generated))
            else:
                matches = expected.read_bytes() == generated.read_bytes()
            if not matches:
                raise SystemExit(f"stale artifact: {expected}; run `make tables`")


if __name__ == "__main__":
    main()
