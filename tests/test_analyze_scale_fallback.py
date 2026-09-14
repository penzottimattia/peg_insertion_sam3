from peg_analysis.analyze import _first_valid_pre_gap


def test_scale_uses_first_valid_pre_gap_feature():
    features = [None, None, {"visible_length": 10.0}, {"visible_length": 9.0}]
    index, feature = _first_valid_pre_gap(features, 3)
    assert index == 2
    assert feature is features[2]


def test_scale_never_uses_post_gap_feature():
    features = [None, None, {"visible_length": 10.0}]
    assert _first_valid_pre_gap(features, 1) == (None, None)
