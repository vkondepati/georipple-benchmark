import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.evaluate import classify_misses


def test_misses_split_into_forecast_and_delay_compounded():
    truth = np.array([True, True, True, False, True])
    predicted = np.array([True, False, False, False, False])
    replay_stockout = np.array([True, True, False, False, False])
    counts = classify_misses(truth, predicted, replay_stockout)
    assert counts == {"missed": 3, "missed_forecast": 1, "missed_delay": 2}
