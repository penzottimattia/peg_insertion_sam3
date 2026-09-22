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
    assert _highlight_frame({"highlight": np.nan}, 100, 240, timestamps, 0.5) == 230
    assert _highlight_frame({"highlight": 99}, 100, 240, timestamps, 0.5) == 230


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




def test_acceleration_keeps_one_output_frame_per_source_frame():
    # The gallery loop length is source-frame count. At 6 Hz source encoded at
    # 24 fps, 120 frames therefore make a 5 s MP4 representing 20 s recording time.
    from peg_analysis.gallery import _recording_time
    source_frames, dataset_fps, export_fps = 120, 6.0, 24.0
    assert source_frames / export_fps == 5.0
    assert _recording_time(source_frames, dataset_fps) == 20.0


def test_fallback_highlight_shifts_three_seconds_on_host_clock():
    timestamps = np.arange(300, dtype=np.int64) * 50_000_000
    assert _highlight_frame({"highlight": np.nan}, 100, 240, timestamps, 3.0) == 180
    assert _highlight_frame({"highlight": 999}, 100, 240, timestamps, 3.0) == 180


def test_fallback_border_stays_visible_three_export_seconds_when_accelerated():
    from peg_analysis.gallery import _ensure_visible_highlight
    # 6 Hz source, 24 fps export, 4x acceleration. Host-time 3 s early would
    # normally be only 0.75 s on screen. Enforce 72 source frames of visibility,
    # which is 3 s at the 24 fps exported playback rate.
    assert _ensure_visible_highlight(180, 100, 240, 3.0, 6.0, 24.0, True) == 169


def test_valid_highlight_is_not_retimed_by_visibility_guard():
    from peg_analysis.gallery import _ensure_visible_highlight
    assert _ensure_visible_highlight(180, 100, 240, 3.0, 6.0, 24.0, False) == 180


def test_synthetic_gallery_row_uses_closest_physical_neighbor():
    import pandas as pd
    from peg_analysis.gallery import _replace_synthetic_with_nearest_physical
    data = pd.DataFrame([
        dict(trial_label="T01", method="m", tolerance=1.0, demo="demo_a", source_summary="a.csv", synthetic=False, max_insertion_depth_mm=10.0, max_abs_axial_slip_mm=2.0, initial_angular_error_deg=1.0),
        dict(trial_label="T02", method="m", tolerance=1.0, demo="demo_b", source_summary="b.csv", synthetic=False, max_insertion_depth_mm=30.0, max_abs_axial_slip_mm=8.0, initial_angular_error_deg=5.0),
        dict(trial_label="T03", method="m", tolerance=1.0, demo="synthetic_000001", source_summary="b.csv", synthetic=True, max_insertion_depth_mm=29.0, max_abs_axial_slip_mm=7.5, initial_angular_error_deg=4.8),
    ])
    out = _replace_synthetic_with_nearest_physical(data)
    replacement = out.iloc[2]
    assert replacement.trial_label == "T03"
    assert replacement.demo == "demo_b"
    assert replacement.source_summary == "b.csv"
    assert bool(replacement.gallery_replaced_synthetic)
    assert replacement.gallery_replacement_demo == "demo_b"


def test_synthetic_neighbor_never_crosses_method_or_tolerance():
    import pandas as pd
    from peg_analysis.gallery import _replace_synthetic_with_nearest_physical
    data = pd.DataFrame([
        dict(method="a", tolerance=1.0, demo="same_group", source_summary="a.csv", synthetic=False, max_insertion_depth_mm=0.0),
        dict(method="b", tolerance=1.0, demo="closer_wrong_method", source_summary="b.csv", synthetic=False, max_insertion_depth_mm=99.0),
        dict(method="a", tolerance=1.0, demo="synthetic_000001", source_summary="a.csv", synthetic=True, max_insertion_depth_mm=100.0),
    ])
    assert _replace_synthetic_with_nearest_physical(data).iloc[2].demo == "same_group"


def test_synthetic_highlight_jitter_uses_host_timestamp_clock():
    from peg_analysis.gallery import _jitter_highlight_host_time
    timestamps = np.array(
        [0, 100_000_000, 300_000_000, 600_000_000, 1_000_000_000],
        dtype=np.int64,
    )
    assert _jitter_highlight_host_time(3, 0, 4, timestamps, -0.25) == 2
    assert _jitter_highlight_host_time(2, 0, 4, timestamps, 0.25) == 3


def test_example_exposes_synthetic_highlight_jitter():
    import json
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1] / "cumulative.example.json").read_text())
    assert config["video_gallery"]["synthetic_highlight_jitter_seconds"] == 0.25


def test_directory_gallery_accepts_tolerance_parameter():
    import inspect
    from peg_analysis.gallery import render_directory_gallery
    assert "tolerance" in inspect.signature(render_directory_gallery).parameters

def test_cumulative_gallery_draws_tolerance_header():
    import inspect
    from peg_analysis import gallery
    source = inspect.getsource(gallery.render_cumulative_gallery)
    assert "tolerance_text" in source
    assert "tolerance_label" in source


def test_cumulative_gallery_separates_method_column_blocks():
    import inspect
    from peg_analysis import gallery
    source = inspect.getsource(gallery.render_cumulative_gallery)
    assert "column_gap" in source
    assert "gi * column_gap" in source
