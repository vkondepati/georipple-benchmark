"""Validate checked-in JSON and deterministic table artifacts."""
import json
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
    if payload.get("provenance", {}).get("git_dirty") is not False:
        raise SystemExit(f"non-clean provenance: {path}; regenerate from a clean commit")


def main():
    paths = (
        ROOT / "results" / "runs.json",
        ROOT / "results" / "summary.json",
        ROOT / "results" / "sensitivity_runs.json",
        ROOT / "results" / "sensitivity_summary.json",
        ROOT / "paper" / "figures" / "figure_stats.json",
        ROOT / "data" / "us-states.json",
    )
    payloads = {path: strict_json(path) for path in paths}
    runs_path = ROOT / "results" / "runs.json"
    sensitivity_path = ROOT / "results" / "sensitivity_runs.json"
    validate_runs(payloads[runs_path], runs_path)
    validate_runs(
        payloads[sensitivity_path],
        sensitivity_path,
        len(payloads[sensitivity_path].get("configs", [])),
    )

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "make_tables.py"),
                "--runs",
                str(ROOT / "results" / "runs.json"),
                "--outdir",
                tmp,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "make_sensitivity.py"),
                "--runs",
                str(ROOT / "results" / "sensitivity_runs.json"),
                "--outdir",
                tmp,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for name in (
            "tables.tex",
            "summary.json",
            "sensitivity.tex",
            "sensitivity_summary.json",
        ):
            expected = ROOT / "results" / name
            generated = Path(tmp) / name
            if expected.read_bytes() != generated.read_bytes():
                raise SystemExit(f"stale artifact: {expected}; run `make tables`")


if __name__ == "__main__":
    main()
