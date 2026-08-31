import numpy as np
from peg_analysis.core import axial_above_thumb_length, detect_gap, feat, length_scale, peg_angle_deg


def test_gap():
    assert detect_gap([0, 10, 20, 100, 110])[1] == 3


def test_geometry_angle_scale():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 8:11] = True
    f = feat(mask)
    assert np.isclose(f["visible_length"], 9.0)
    assert np.isclose(length_scale(f, 18.0), 2.0)
    assert abs(peg_angle_deg(f["axis"])) < 1e-9


def test_thumb_and_axial_length():
    peg_mask = np.zeros((20, 20), dtype=bool); peg_mask[2:12, 9:12] = True
    hand_mask = np.zeros((20, 20), dtype=bool); hand_mask[7:10, 4:8] = True
    peg, hand = feat(peg_mask), feat(hand_mask)
    assert np.allclose(hand["thumb_tip"], [5.5, 9.0])
    assert np.isclose(axial_above_thumb_length(peg, hand), 7.0)


def test_visible_width_and_robust_noise():
    from peg_analysis.core import dimension_scale, robust_noise
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 8:11] = True
    f = feat(mask)
    assert np.isclose(f["visible_width"], 2.0)
    assert np.isclose(dimension_scale(f, 4.0, "visible_width"), 2.0)
    n, std, mad = robust_noise([1.0, 2.0, 3.0, np.nan])
    assert n == 3 and std > 0 and mad == 1.0
