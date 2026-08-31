import numpy as np
from peg_analysis.plots import _mean_ci95


def test_ci_single_value():
    mean, ci = _mean_ci95([3.0])
    assert mean == 3.0 and ci == 0.0


def test_ci_ignores_nan():
    mean, ci = _mean_ci95([1.0, np.nan, 3.0])
    assert mean == 2.0 and ci > 0
