import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from make_tables import cell, clustered_stats, finite_mean, load_runs
from check_artifacts import validate_runs


def test_clustered_stats_uses_seed_as_independent_unit():
    runs = [
        {"seed": 0, "value": 1.0}, {"seed": 0, "value": 3.0},
        {"seed": 1, "value": 5.0}, {"seed": 1, "value": 7.0},
    ]
    stat = clustered_stats(runs, lambda r: r["value"])
    assert stat["mean"] == pytest.approx(4.0)
    assert stat["n_records"] == 4
    assert stat["n_seeds"] == 2
    assert stat["ci95"][0] < stat["mean"] < stat["ci95"][1]


def test_clustered_stats_clips_bounded_confidence_interval():
    runs = [{"seed": i, "value": value}
            for i, value in enumerate([0.8, 0.9, 1.0, 1.0, 1.0])]
    stat = clustered_stats(runs, lambda r: r["value"], bounds=(0, 1))
    assert stat["ci95"][1] == 1.0


def test_percentage_cells_can_be_escaped_for_latex():
    stat = {"mean": 0.9, "ci95": [0.8, 1.0]}
    assert cell(stat, "{:.0%}").replace("%", r"\%") == r"90\% [80\%, 100\%]"


def test_finite_mean_ignores_missing_values():
    assert finite_mean([None, 1.0, 3.0]) == 2.0
    assert finite_mean([None]) is None


def test_load_runs_rejects_empty_artifact(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text('{"runs": []}')
    with pytest.raises(ValueError, match="non-empty 'runs' list"):
        load_runs(path)


def test_validate_runs_requires_complete_clean_artifact(tmp_path):
    path = tmp_path / "runs.json"
    payload = {
        "config": {"seeds": 1, "truth_reps": 2},
        "provenance": {"git_dirty": False},
        "runs": [
            {"hazard": "flood"},
            {"hazard": "flood"},
            {"hazard": "storm"},
            {"hazard": "storm"},
        ],
    }
    validate_runs(payload, path)
    payload["provenance"]["git_dirty"] = True
    with pytest.raises(SystemExit, match="non-clean provenance"):
        validate_runs(payload, path)
