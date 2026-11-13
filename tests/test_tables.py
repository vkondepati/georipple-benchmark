import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from make_tables import cell, clustered_stats, finite_mean, load_runs
from check_artifacts import equivalent_json, validate_runs


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


def test_equivalent_json_tolerates_only_small_numeric_noise():
    assert equivalent_json({"x": [1.0]}, {"x": [1.0 + 1e-12]})
    assert not equivalent_json({"x": [1.0]}, {"x": [1.01]})
    assert not equivalent_json({"x": True}, {"x": 1})


def test_holm_adjustment_is_monotone_and_never_below_raw_p():
    from scipy.stats import ttest_1samp

    from make_tables import holm_tests

    strong = [3.0, 2.5, 3.5, 2.8, 3.1]
    weak = [0.4, -0.3, 0.9, -0.2, 0.6]
    runs = [{"seed": i, "a": s, "b": w} for i, (s, w) in enumerate(zip(strong, weak, strict=True))]
    out = holm_tests(runs, {"a": lambda r: r["a"], "b": lambda r: r["b"]}, {"a": 1, "b": 1})
    p_a, p_b = ttest_1samp(strong, 0.0).pvalue, ttest_1samp(weak, 0.0).pvalue
    assert out["a"]["p"] == pytest.approx(p_a) and out["b"]["p"] == pytest.approx(p_b)
    assert out["a"]["p_holm"] == pytest.approx(min(1.0, 2 * p_a))
    assert out["b"]["p_holm"] == pytest.approx(max(min(1.0, 2 * p_a), p_b))
    assert out["a"]["mean"] == pytest.approx(np.mean(strong))


def test_validate_runs_rejects_time_limited_milps(tmp_path):
    path = tmp_path / "runs.json"
    nested = {"config": {"seeds": 1, "truth_reps": 1}, "provenance": {"git_dirty": False},
              "runs": [{"hazard": "flood", "plan_info": {"Full": {"status": 1}}}]}
    with pytest.raises(SystemExit, match="optimality gap"):
        validate_runs(nested, path)
    flat = {"config": {"seeds": 1, "truth_reps": 1}, "provenance": {"git_dirty": False},
            "runs": [{"hazard": "flood", "plan_info": {"status": 1, "message": "time limit"}}]}
    with pytest.raises(SystemExit, match="optimality gap"):
        validate_runs(flat, path)
    flat["runs"][0]["plan_info"]["status"] = 0
    validate_runs(flat, path)


def test_hypotheses_skip_records_with_undefined_metrics():
    from make_tables import holm_tests, paired

    runs = [{"seed": 0, "a": 0.2, "b": 0.1}, {"seed": 0, "a": None, "b": 0.3},
            {"seed": 1, "a": 0.4, "b": 0.1}, {"seed": 2, "a": 0.5, "b": 0.2}]
    out = holm_tests(runs, {"h": lambda r: paired(r["a"], r["b"])}, {"h": 1})
    assert out["h"]["n_records"] == 3
    assert out["h"]["mean"] == pytest.approx(np.mean([0.1, 0.3, 0.3]))


def test_zero_variance_differences_give_a_defined_p_value():
    from make_tables import holm_tests

    zeros = [{"seed": i, "d": 0.0} for i in range(5)]
    constant = [{"seed": i, "d": -2.0} for i in range(5)]
    assert holm_tests(zeros, {"h": lambda r: r["d"]}, {"h": -1})["h"]["p"] == 1.0
    out = holm_tests(constant, {"h": lambda r: r["d"]}, {"h": -1})["h"]
    assert out["p"] == 0.0 and out["confirmed"] is True


def test_confirmation_needs_holm_significance_and_the_predicted_sign():
    from make_tables import holm_tests

    runs = [{"seed": i, "up": u, "down": -u}
            for i, u in enumerate([3.0, 2.5, 3.5, 2.8, 3.1])]
    out = holm_tests(runs, {"up": lambda r: r["up"], "down": lambda r: r["down"]},
                     {"up": -1, "down": -1})
    assert out["up"]["confirmed"] is False      # significant, wrong sign
    assert out["down"]["confirmed"] is True
    assert out["down"]["predicted_sign"] == -1


def _confirm_payload(lanes="road", dcs=1, offset=10, commit="a" * 40):
    from georipple.hazards import base_hazards

    runs = [{"seed": s, "hazard": h.name, "truth_rep": k,
             "condition": {"lanes": lanes, "dealer_dcs": dcs}}
            for s in range(10, 20) for h in base_hazards() for k in range(3)]
    return {"config": {"seeds": 10, "seed_offset": offset, "truth_reps": 3,
                       "lanes": lanes, "dealer_dcs": dcs},
            "provenance": {"git_dirty": False, "git_commit": commit}, "runs": runs}


def test_confirmation_files_must_hold_their_own_condition_and_fresh_seeds(tmp_path):
    from check_artifacts import validate_confirmation

    path = tmp_path / "C1.json"
    validate_confirmation(_confirm_payload(), path, "C1")
    with pytest.raises(SystemExit, match="condition"):
        validate_confirmation(_confirm_payload(lanes="straight"), path, "C1")
    with pytest.raises(SystemExit, match="seeds"):
        validate_confirmation(_confirm_payload(offset=0), path, "C1")
    mislabeled = _confirm_payload()
    mislabeled["runs"][5]["condition"] = {"lanes": "straight", "dealer_dcs": 1}
    with pytest.raises(SystemExit, match="condition"):
        validate_confirmation(mislabeled, path, "C1")
    duplicated = _confirm_payload()
    duplicated["runs"][1] = dict(duplicated["runs"][0])
    with pytest.raises(SystemExit, match="seeds"):
        validate_confirmation(duplicated, path, "C1")


def test_raw_artifacts_must_share_one_commit_and_the_pinned_environment(tmp_path):
    from check_artifacts import locked_pins, validate_environment

    pins = locked_pins()
    ok = {"provenance": {"python": "3.11.9", "packages": dict(pins), "git_commit": "b" * 40}}
    validate_environment([("x.json", ok)])
    other = {"provenance": {**ok["provenance"], "git_commit": "c" * 40}}
    with pytest.raises(SystemExit, match="commit"):
        validate_environment([("x.json", ok), ("y.json", other)])
    drifted = {"provenance": {**ok["provenance"], "packages": {**pins, "scipy": "0.0.1"}}}
    with pytest.raises(SystemExit, match="scipy"):
        validate_environment([("x.json", drifted)])
    with pytest.raises(SystemExit, match="Python"):
        validate_environment([("x.json", {"provenance": {**ok["provenance"], "python": "3.14.0"}})])
