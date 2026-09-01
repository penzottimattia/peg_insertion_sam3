import h5py
import numpy as np
import pytest

from peg_analysis.timestamp_gap import insert_timestamp_gap


def _dataset(path):
    with h5py.File(path, "w") as h5:
        cameras = h5.create_group("demos/demo_000002/cameras")
        for serial in ("main", "secondary"):
            camera = cameras.create_group(serial)
            camera.create_dataset(
                "host_timestamp_ns",
                data=np.array([0, 10, 20, 30, 40], dtype=np.int64),
            )


def test_insert_gap_copies_and_updates_all_cameras(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "updated.h5"
    _dataset(source)

    destination, results = insert_timestamp_gap(
        source, "2", 3, 100, output_path=output
    )

    assert destination == output.resolve()
    assert len(results) == 2
    with h5py.File(source, "r") as h5:
        assert h5["demos/demo_000002/cameras/main/host_timestamp_ns"][:].tolist() == [0, 10, 20, 30, 40]
    with h5py.File(output, "r") as h5:
        for serial in ("main", "secondary"):
            values = h5[f"demos/demo_000002/cameras/{serial}/host_timestamp_ns"][:]
            assert values.tolist() == [0, 10, 20, 130, 140]


def test_insert_gap_can_target_one_camera_in_place(tmp_path):
    source = tmp_path / "source.h5"
    _dataset(source)

    insert_timestamp_gap(
        source, "demo_000002", 2, 50, in_place=True, camera_serial="main"
    )

    with h5py.File(source, "r") as h5:
        assert h5["demos/demo_000002/cameras/main/host_timestamp_ns"][:].tolist() == [0, 10, 70, 80, 90]
        assert h5["demos/demo_000002/cameras/secondary/host_timestamp_ns"][:].tolist() == [0, 10, 20, 30, 40]


def test_insert_gap_rejects_first_or_out_of_range_frame(tmp_path):
    source = tmp_path / "source.h5"
    _dataset(source)

    with pytest.raises(ValueError, match="at least 1"):
        insert_timestamp_gap(source, "2", 0, 100)
    with pytest.raises(ValueError, match="outside"):
        insert_timestamp_gap(source, "2", 5, 100)
