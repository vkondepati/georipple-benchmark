import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from run_benchmark import positive_float, positive_int


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


@pytest.mark.parametrize("value", ["0", "-0.5"])
def test_positive_float_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(value)
