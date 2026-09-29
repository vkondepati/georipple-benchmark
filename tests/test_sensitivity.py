import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from run_sensitivity import sensitivity_configs


def test_sensitivity_grid_is_one_factor_at_a_time():
    configs = sensitivity_configs()
    assert len({config["id"] for config in configs}) == len(configs)
    reference = next(config for config in configs if config["id"] == "base")
    for config in configs:
        changed = sum(config[key] != reference[key] for key in ("quantile", "top_k", "members"))
        assert changed <= 1
