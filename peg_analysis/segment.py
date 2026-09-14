from pathlib import Path
import json
import numpy as np
from .core import camera_roles, configured_objects, demos, detect_gap, group, save
from .sam3_runner import Runner


def selected_demos(h, requested_demo=None):
    available = demos(h)
    if requested_demo is None:
        return available
    if requested_demo not in available:
        raise ValueError(f"Unknown complete demo {requested_demo!r}. Available: {', '.join(available)}")
    return [requested_demo]


def _segment_demo(c, demo, overwrite=False, runner=None):
    """Segment one demo and return the lazily-created runner."""
    import h5py
    output_dir = Path(c["output_dir"])
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(c["dataset_path"], "r") as h:
        for role, serial in camera_roles(c):
            mask_path = masks_dir / f"{demo}_{role}.npz"
            meta_path = masks_dir / f"{demo}_{role}_meta.json"
            # Reuse existing masks and migrate legacy metadata without rerunning SAM.
            if mask_path.exists() and not overwrite:
                g = group(h, demo, serial)
                timestamps = g["host_timestamp_ns"][:]
                metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                changed = False

                dataset_path = str(Path(c["dataset_path"]).expanduser().resolve())
                if metadata.get("dataset_path") != dataset_path:
                    metadata["dataset_path"] = dataset_path
                    changed = True

                defaults = {
                    "demo": demo,
                    "role": role,
                    "camera_serial": str(serial),
                    "frame_count": int(len(timestamps)),
                }
                for key, value in defaults.items():
                    if key not in metadata:
                        metadata[key] = value
                        changed = True

                if "insertion_start_frame" not in metadata or "pre_gap_frame" not in metadata:
                    pre_gap, insertion_start, gap_ns = detect_gap(
                        timestamps,
                        c["onset"]["gap_mad_multiplier"],
                        c["onset"]["gap_nominal_multiplier"],
                    )
                    metadata["pre_gap_frame"] = int(pre_gap)
                    metadata["insertion_start_frame"] = int(insertion_start)
                    metadata["timestamp_gap_ns"] = int(gap_ns)
                    changed = True

                if changed:
                    save(meta_path, metadata)
                    print(f"Updated existing metadata: {demo}/{role}")
                else:
                    print(f"Skipping existing masks: {demo}/{role}")
                continue
            if runner is None:
                runner = Runner(c["sam3"])
            g = group(h, demo, serial)
            timestamps = g["host_timestamp_ns"][:]
            pre_gap, insertion_start, gap_ns = detect_gap(
                timestamps, c["onset"]["gap_mad_multiplier"], c["onset"]["gap_nominal_multiplier"])
            frames = np.asarray(g["rgb"])
            masks, sam_meta = runner.all(frames, configured_objects(c, role), pre_gap,
                                         output_dir / "work" / demo / role)
            np.savez_compressed(mask_path, **masks)
            save(meta_path, {"demo": demo, "role": role, "camera_serial": str(serial),
                             "dataset_path": str(Path(c["dataset_path"]).expanduser().resolve()),
                             "frame_count": int(len(frames)), "pre_gap_frame": int(pre_gap),
                             "insertion_start_frame": int(insertion_start),
                             "timestamp_gap_ns": int(gap_ns), "sam": sam_meta})
            print(f"Saved masks: {demo}/{role}")


    return runner


def segment(c, requested_demo=None, overwrite=False):
    """Segment demos independently, warning and continuing after demo-local failures."""
    import h5py
    import warnings

    with h5py.File(c["dataset_path"], "r") as h:
        requested = selected_demos(h, requested_demo)

    runner = None
    completed = 0
    failed = 0
    for demo in requested:
        try:
            runner = _segment_demo(c, demo, overwrite=overwrite, runner=runner)
            completed += 1
        except Exception as exc:
            failed += 1
            warnings.warn(
                f"Skipping segmentation for {demo}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    print(f"Segmented {completed} demo(s)")
    if failed:
        print(f"Warning: skipped segmentation for {failed} demo(s)")
