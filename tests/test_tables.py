import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from make_tables import clustered_stats


def test_clustered_stats_uses_seed_as_independent_unit():
    runs = [
        {"seed": 0, "value": 1.0}, {"seed": 0, "value": 3.0},
        {"seed": 1, "value": 5.0}, {"seed": 1, "value": 7.0},
    ]
    stat = clustered_stats(runs, lambda r: r["value"])
    assert stat["mean"] == pytest.approx(4.0)
    assert stat["n_runs"] == 4
    assert stat["n_seeds"] == 2
    assert stat["ci95"][0] < stat["mean"] < stat["ci95"][1]
