import json
import numpy as np
from peg_analysis.extract_pixels import _automatic_frames, _save_rgb, _validate_frames

def test_save_rgb_combines_all_objects_into_one_image(tmp_path):
    import cv2
    frame=np.arange(5*6*3,dtype=np.uint8).reshape(5,6,3)
    peg=np.zeros((5,6),bool); peg[1,2]=True
    holder=np.zeros((5,6),bool); holder[3,4]=True
    hand=np.zeros((5,6),bool); hand[4,5]=True
    path=tmp_path/'frame.png'
    assert _save_rgb(path, frame, {"peg":peg,"holder":holder,"hand":hand}) == 3
    saved=cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    union=peg|holder|hand
    assert np.array_equal(saved[union, :3], frame[union])
    assert (saved[union, 3] == 255).all()
    assert (saved[~union, 3] == 0).all()


def test_validate_frames_deduplicates_and_rejects_range():
    import pytest
    assert _validate_frames([2,0,2],3,'demo','main') == [2,0]
    with pytest.raises(ValueError, match='outside'):
        _validate_frames([3],3,'demo','main')


def test_automatic_frames_uses_last_frame_before_timestamp_gap():
    onset={"gap_mad_multiplier": 8.0, "gap_nominal_multiplier": 3.0}
    assert _automatic_frames(np.array([0,10,20,100,110,120]), onset) == [0,2,5]
