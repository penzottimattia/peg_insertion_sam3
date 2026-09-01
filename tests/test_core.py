import numpy as np
from peg_analysis.core import axial_above_thumb_length, detect_gap, feat, lateral_thumb_lower_bbox_distance, length_scale, peg_angle_deg


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


def test_lateral_uses_lower_bbox_top_left_and_thumb_median():
    peg = np.zeros((20, 24), dtype=bool)
    hand = np.zeros((20, 24), dtype=bool)
    peg[9:18, 10:15] = True
    hand[8, 1:4] = True
    hand[8, 9:14] = True

    result = lateral_thumb_lower_bbox_distance(peg, hand)

    assert np.isclose(result["thumb_x"], 11.5)
    assert np.isclose(result["thumb_y"], 8.0)
    assert np.isclose(result["corner_x"], 10.0)
    assert np.isclose(result["corner_y"], 9.0)
    assert np.isclose(result["distance"], np.sqrt(3.25))


def test_lateral_selects_contiguous_segment_closest_to_lower_bbox():
    peg = np.zeros((20, 28), dtype=bool)
    hand = np.zeros((20, 28), dtype=bool)
    peg[9:18, 6:21] = True
    hand[8, 6:9] = True
    hand[8, 17:20] = True

    result = lateral_thumb_lower_bbox_distance(peg, hand)

    assert np.isclose(result["thumb_x"], 7.0)
    assert result["thumb_segment_min_x"] == 6
    assert result["thumb_segment_max_x"] == 8


def test_lateral_rejects_segment_outside_peg_bbox():
    peg = np.zeros((20, 24), dtype=bool)
    hand = np.zeros((20, 24), dtype=bool)
    peg[9:18, 10:15] = True
    hand[8, 2:5] = True
    assert lateral_thumb_lower_bbox_distance(peg, hand) is None


def test_lateral_requires_peg_strictly_below_thumb():
    peg = np.zeros((20, 24), dtype=bool)
    hand = np.zeros((20, 24), dtype=bool)
    peg[4:9, 10:15] = True
    hand[8, 10:13] = True
    assert lateral_thumb_lower_bbox_distance(peg, hand) is None
