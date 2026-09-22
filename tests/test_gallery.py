import numpy as np
import pandas as pd

from peg_analysis.gallery import _fill_group, _highlight_frame


def test_highlight_uses_last_when_missing_or_before_shallow_start():
    assert _highlight_frame({"highlight": np.nan}, 100, 240) == 240
    assert _highlight_frame({"highlight": 99}, 100, 240) == 240
    assert _highlight_frame({"highlight": 100}, 100, 240) == 100
    assert _highlight_frame({"highlight": 170}, 100, 240) == 170


def test_highlight_is_clamped_to_last_frame():
    assert _highlight_frame({"highlight": 999}, 100, 240) == 240


def test_fill_group_duplicates_to_ten_deterministically():
    group = pd.DataFrame({"demo": ["a", "b", "c"]})
    first = _fill_group(group, 10, np.random.default_rng(4))
    second = _fill_group(group, 10, np.random.default_rng(4))
    assert len(first) == 10
    assert first.demo.tolist() == second.demo.tolist()
    assert first.gallery_duplicate.tolist() == [False] * 3 + [True] * 7


def test_highlight_clamps_to_truncated_max_depth_frame():
    assert _highlight_frame({"highlight": 220}, 100, 180) == 180
    assert _highlight_frame({"highlight": np.nan}, 100, 180) == 180


def test_example_gallery_groups_by_tolerance_by_default():
    import json
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1] / "cumulative.example.json").read_text())
    assert config["video_gallery"]["group_by_tolerance"] is True


def test_highlight_can_be_shifted_earlier_using_source_timestamps():
    timestamps = np.arange(300, dtype=np.int64) * 50_000_000  # 20 Hz source
    assert _highlight_frame({"highlight": 170}, 100, 240, timestamps, 0.5) == 160
    assert _highlight_frame({"highlight": 105}, 100, 240, timestamps, 0.5) == 100


def test_highlight_early_offset_does_not_change_fallback_semantics():
    timestamps = np.arange(300, dtype=np.int64) * 50_000_000
    assert _highlight_frame({"highlight": np.nan}, 100, 240, timestamps, 0.5) == 240
    assert _highlight_frame({"highlight": 99}, 100, 240, timestamps, 0.5) == 240


def test_group_by_tolerance_documented_as_additive():
    from pathlib import Path
    text = (Path(__file__).parents[1] / "README.md").read_text()
    assert "group_by_tolerance` is additive" in text
    assert "per-dataset/method gallery is always rendered" in text


def test_anomalous_depth_overrides_success():
    from peg_analysis.gallery import _trial_success
    row = {"max_insertion_depth_mm": 30.0}
    assert _trial_success(row, 15.0, None)
    assert not _trial_success(row, 15.0, 29.0)
    assert _trial_success({"max_insertion_depth_mm": 20.0}, 15.0, 29.0)


def test_directory_gallery_accepts_spec_parameter():
    import inspect
    from peg_analysis.gallery import render_directory_gallery
    assert "spec_path" in inspect.signature(render_directory_gallery).parameters
