from pathlib import Path
import shutil


def _resolve_demo(h5, demo):
    demos_group = h5.get("demos")
    if demos_group is None:
        raise ValueError("Dataset does not contain a /demos group")

    if demo in demos_group:
        return demo

    try:
        candidate = f"demo_{int(demo):06d}"
    except (TypeError, ValueError):
        candidate = None

    if candidate and candidate in demos_group:
        return candidate

    available = sorted(name for name in demos_group if name.startswith("demo_"))
    raise ValueError(
        f"Unknown demo {demo!r}. Available: {', '.join(available) or '(none)'}"
    )


def _selected_camera_groups(h5, demo, camera_serial=None):
    cameras_path = f"demos/{demo}/cameras"
    if cameras_path not in h5:
        raise ValueError(f"Demo has no cameras group: /{cameras_path}")

    cameras = h5[cameras_path]
    if camera_serial is not None:
        serial = str(camera_serial)
        if serial not in cameras:
            available = sorted(cameras.keys())
            raise ValueError(
                f"Unknown camera {serial!r} for {demo}. "
                f"Available: {', '.join(available) or '(none)'}"
            )
        return [(serial, cameras[serial])]

    return [(serial, cameras[serial]) for serial in sorted(cameras.keys())]


def insert_timestamp_gap(
    dataset_path,
    demo,
    frame,
    gap_ns,
    *,
    output_path=None,
    in_place=False,
    camera_serial=None,
):
    """Insert a timestamp discontinuity immediately before ``frame``.

    Every timestamp from ``frame`` onward is increased by ``gap_ns``. By
    default all cameras in the selected demo are updated so that the views
    remain synchronized. The source file is copied unless ``in_place`` is set.
    """
    import h5py
    import numpy as np

    source = Path(dataset_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if frame < 1:
        raise ValueError("frame must be at least 1 so the gap has a preceding frame")
    if gap_ns <= 0:
        raise ValueError("gap_ns must be greater than zero")
    if in_place and output_path is not None:
        raise ValueError("output_path and in_place are mutually exclusive")

    if in_place:
        destination = source
    else:
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else source.with_name(f"{source.stem}_with_gap{source.suffix}")
        )
        if destination == source:
            raise ValueError("output_path must differ from dataset_path unless --in-place is used")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    try:
        with h5py.File(destination, "r+") as h5:
            resolved_demo = _resolve_demo(h5, demo)
            selected = _selected_camera_groups(h5, resolved_demo, camera_serial)

            updates = []
            for serial, camera in selected:
                if "host_timestamp_ns" not in camera:
                    raise ValueError(
                        f"Missing host_timestamp_ns for {resolved_demo}/camera {serial}"
                    )
                timestamps = camera["host_timestamp_ns"]
                if timestamps.ndim != 1:
                    raise ValueError(
                        f"host_timestamp_ns is not one-dimensional for "
                        f"{resolved_demo}/camera {serial}"
                    )
                if frame >= len(timestamps):
                    raise ValueError(
                        f"frame {frame} is outside {resolved_demo}/camera {serial} "
                        f"with {len(timestamps)} frames"
                    )
                if not np.issubdtype(timestamps.dtype, np.integer):
                    raise ValueError(
                        f"host_timestamp_ns must use an integer dtype for "
                        f"{resolved_demo}/camera {serial}"
                    )

                tail = timestamps[frame:]
                max_value = np.iinfo(timestamps.dtype).max
                if len(tail) and int(tail.max()) > max_value - gap_ns:
                    raise OverflowError(
                        f"Adding {gap_ns} ns would overflow timestamps for "
                        f"{resolved_demo}/camera {serial}"
                    )
                updates.append((serial, timestamps, tail + gap_ns))

            for _, timestamps, shifted_tail in updates:
                timestamps[frame:] = shifted_tail

            results = []
            for serial, timestamps, _ in updates:
                actual_gap = int(timestamps[frame]) - int(timestamps[frame - 1])
                results.append(
                    {
                        "demo": resolved_demo,
                        "camera_serial": serial,
                        "frame": int(frame),
                        "added_gap_ns": int(gap_ns),
                        "resulting_step_ns": actual_gap,
                    }
                )
    except Exception:
        if not in_place and destination.exists():
            destination.unlink()
        raise

    return destination, results
