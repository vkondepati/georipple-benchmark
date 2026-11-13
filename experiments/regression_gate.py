"""Check that the current runner reproduces the published development artifact.

Methods whose definition did not change must match the published records exactly:
prediction B1/B2/A1/Full, resolution B3, the new A4 (absolute dispatch) against the old
Full, the new B4-abs against the old B4, and no action apart from TTR (which now counts
from the realized onset). Timing fields are ignored.

Usage: python experiments/regression_gate.py --new build/runs.json [--reference-commit ea465ae]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("prediction", m, m) for m in ("B1", "B2", "A1", "Full")] + [
    ("resolution", "B3", "B3"), ("resolution", "Full", "A4"), ("resolution", "B4", "B4-abs")]


def mismatches(old_runs, new_runs):
    if len(old_runs) != len(new_runs):
        return [f"record count {len(old_runs)} != {len(new_runs)}"]
    key = lambda r: (r["seed"], r["hazard"], r["truth_rep"])
    new_by_key = {key(r): r for r in new_runs}
    bad = []
    for old in old_runs:
        new = new_by_key.get(key(old))
        if new is None:
            bad.append(f"{key(old)} missing")
            continue
        for section, old_name, new_name in PAIRS:
            if old[section][old_name] != new[section][new_name]:
                bad.append(f"{key(old)} {section} {old_name}->{new_name}")
        a, b = old["resolution"]["NoAction"], new["resolution"]["NoAction"]
        if {k: v for k, v in a.items() if k != "ttr95_days"} != \
                {k: v for k, v in b.items() if k != "ttr95_days"}:
            bad.append(f"{key(old)} resolution NoAction")
        if old["stci_top_changed_lambda0"] != new["stci_top_changed_lambda0"]:
            bad.append(f"{key(old)} STCI top-K flag")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--reference-commit", default="ea465ae")
    a = ap.parse_args()
    published = subprocess.run(["git", "show", f"{a.reference_commit}:results/runs.json"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout
    old_runs = json.loads(published)["runs"]
    with open(a.new) as f:
        new_runs = json.load(f)["runs"]
    bad = mismatches(old_runs, new_runs)
    print(f"{len(old_runs)} records compared with {a.reference_commit}; {len(bad)} mismatches")
    for line in bad[:20]:
        print("  ", line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
