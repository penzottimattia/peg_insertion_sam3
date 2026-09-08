import numpy as np

from peg_analysis.render import _crop_box, _rgb_trace, _sample_indices


def test_sample_indices_include_endpoints():
    assert _sample_indices(3, 9, 4).tolist() == [3, 5, 7, 9]


def test_rgb_trace_uses_original_pixels_without_mask_colours():
    frames = np.zeros((3, 12, 12, 3), dtype=np.uint8)
    frames[:] = [20, 30, 40]
    masks = np.zeros((3, 12, 12), dtype=bool)
    frames[0, 2:5, 2:4] = [180, 20, 10]
    frames[1, 4:7, 2:4] = [190, 25, 15]
    frames[2, 6:9, 2:4] = [200, 30, 20]
    masks[0, 2:5, 2:4] = True
    masks[1, 4:7, 2:4] = True
    masks[2, 6:9, 2:4] = True

    result = _rgb_trace(frames[0], frames, masks, [0, 1, 2], alpha=1.0)

    assert np.all(result[2:4, 2:4] == [180, 20, 10])
    assert np.all(result[4:6, 2:4] == [190, 25, 15])
    assert np.all(result[6:9, 2:4] == [200, 30, 20])
    assert np.all(result[0, 0] == [20, 30, 40])


def test_crop_box_contains_complete_task_union():
    masks = {
        "peg": np.zeros((3, 40, 50), dtype=bool),
        "holder": np.zeros((3, 40, 50), dtype=bool),
        "hand": np.zeros((3, 40, 50), dtype=bool),
    }
    masks["peg"][:, 12:20, 23:28] = True
    masks["holder"][:, 20:30, 15:36] = True
    x0, y0, x1, y1 = _crop_box(masks, [0, 1, 2], (40, 50, 3))
    assert x0 <= 15 and y0 <= 12
    assert x1 >= 36 and y1 >= 30
