import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from run_benchmark import nonnegative_int, positive_float, positive_int, realized_hazard
from georipple.hazards import base_hazards


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


@pytest.mark.parametrize("value", ["0", "-0.5"])
def test_positive_float_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(value)


def test_nonnegative_int_accepts_zero_and_rejects_negative():
    assert nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_int("-1")


def test_realized_hazard_is_deterministic_and_held_out():
    base = base_hazards()[0]
    first = realized_hazard(base, 3, 0)
    second = realized_hazard(base, 3, 0)
    assert first.poly.equals(second.poly)
    assert not first.poly.equals(base.poly)
