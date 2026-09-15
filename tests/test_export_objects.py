import numpy as np
import pytest

from peg_analysis.export_objects import keyframes, parse_prompt, parse_prompts, save_merged_png


def test_parse_prompt_accepts_custom_name_and_text():
    assert parse_prompt("red_peg=the red cylinder") == ("red_peg", "the red cylinder")


def test_parse_prompts_rejects_invalid_and_duplicate_names():
    with pytest.raises(ValueError, match="NAME=TEXT"):
        parse_prompt("missing separator")
    with pytest.raises(ValueError, match="duplicate"):
        parse_prompts(["peg=red cylinder", "peg=other cylinder"])


def test_keyframes_are_first_pre_gap_and_last():
    selected, gap = keyframes(
        np.array([0, 10, 20, 100, 110, 120]),
        {"gap_mad_multiplier": 8.0, "gap_nominal_multiplier": 3.0},
    )
    assert selected == [("first", 0), ("pre_gap", 2), ("last", 5)]
    assert gap == 80


def test_save_merged_png_uses_union_of_all_masks(tmp_path):
    import cv2

    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    peg = np.zeros((4, 5), dtype=bool)
    holder = np.zeros((4, 5), dtype=bool)
    peg[1, 2] = True
    holder[2, 3] = True
    union = peg | holder
    path = tmp_path / "merged.png"
    assert save_merged_png(path, frame, {"peg": peg, "holder": holder}) == 2
    image = cv2.cvtColor(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
    )
    assert np.array_equal(image[union, :3], frame[union])
    assert (image[union, 3] == 255).all()
    assert (image[~union, 3] == 0).all()


def test_cli_export_objects_accepts_prompts_as_one_list(monkeypatch):
    import sys
    import peg_analysis.cli as cli

    monkeypatch.setattr(sys, "argv", [
        "peg-analysis", "export-objects",
        "--dataset-path", "dataset.h5",
        "--prompts", "peg=red cylinder", "holder=white cylinder",
    ])
    # Stop after argparse and config dispatch without requiring a real config/dataset.
    captured = {}
    monkeypatch.setattr(cli, "config", lambda *args, **kwargs: {
        "dataset_path": "dataset.h5", "sam3": {}, "onset": {}
    })
    import peg_analysis.export_objects as module
    monkeypatch.setattr(module, "export_objects", lambda c, prompts, **kwargs: captured.update(prompts=prompts))
    cli.main()
    assert captured["prompts"] == ["peg=red cylinder", "holder=white cylinder"]
