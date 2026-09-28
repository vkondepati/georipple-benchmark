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


def main():
    paths = (
        ROOT / "results" / "runs.json",
        ROOT / "results" / "summary.json",
        ROOT / "paper" / "figures" / "figure_stats.json",
        ROOT / "data" / "us-states.json",
    )
    for path in paths:
        strict_json(path)

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
        for name in ("tables.tex", "summary.json"):
            expected = ROOT / "results" / name
            generated = Path(tmp) / name
            if expected.read_bytes() != generated.read_bytes():
                raise SystemExit(f"stale artifact: {expected}; run `make tables`")


if __name__ == "__main__":
    main()
