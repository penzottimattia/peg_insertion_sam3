import json
from pathlib import Path
import numpy as np
import pytest

from peg_analysis.analyze import load_saved_masks


def test_load_saved_masks(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    np.savez_compressed(
        masks_dir / "demo_000001_main.npz",
        peg=np.ones((2, 3, 4), dtype=bool),
        hand=np.ones((2, 3, 4), dtype=bool),
        holder=np.ones((2, 3, 4), dtype=bool),
    )
    (masks_dir / "demo_000001_main_meta.json").write_text(json.dumps({
        "insertion_start_frame": 1
    }))
    masks, meta = load_saved_masks(tmp_path, "demo_000001", "main")
    assert set(masks) == {"peg", "hand", "holder"}
    assert meta["insertion_start_frame"] == 1


def test_missing_saved_masks_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="segment --demo demo_000001"):
        load_saved_masks(tmp_path, "demo_000001", "main")
