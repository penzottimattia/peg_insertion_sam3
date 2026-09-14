"""Extract original RGB pixels selected by saved or freshly-computed prompt masks."""
from pathlib import Path
import json
import numpy as np
from .core import camera_roles, configured_objects, demos, detect_gap, group
from .analyze import load_saved_masks

OBJECTS = ("peg", "holder", "hand")


def _resolve_demo(h5, demo):
    available = demos(h5)
    if demo in available:
        return demo
    try:
        candidate = f"demo_{int(demo):06d}"
    except (TypeError, ValueError):
        candidate = None
    if candidate in available:
        return candidate
    raise ValueError(f"Unknown complete demo {demo!r}. Available: {', '.join(available)}")


def _automatic_frames(timestamps, onset):
    """First frame, last frame before the timestamp discontinuity, and last frame."""
    if len(timestamps) < 2:
        raise ValueError("At least two frames are required for automatic frame selection")
    handover, _, _ = detect_gap(
        timestamps, onset["gap_mad_multiplier"], onset["gap_nominal_multiplier"]
    )
    return [0, int(handover), len(timestamps) - 1]


def _validate_frames(indices, count, demo, role):
    result=[]
    seen=set()
    for value in indices:
        i=int(value)
        if i < 0 or i >= count:
            raise ValueError(f"Frame {i} is outside {demo}/{role} with {count} frames")
        if i not in seen:
            result.append(i); seen.add(i)
    if not result:
        raise ValueError("At least one frame index is required")
    return result


def _save_rgb(destination, frame, masks):
    """Save one RGBA PNG containing the union, transparent everywhere else."""
    import cv2
    frame = np.asarray(frame, dtype=np.uint8)
    union = np.zeros(frame.shape[:2], dtype=bool)
    for mask in masks.values():
        if mask is not None:
            union |= np.asarray(mask, dtype=bool)
    # Preserve real RGB on selected pixels and make all missing pixels fully transparent.
    image = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)
    image[union, :3] = frame[union]
    image[union, 3] = 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    if not cv2.imwrite(str(destination), bgra):
        raise IOError(f"Could not write image: {destination}")
    return int(union.sum())


def _extract_demo_pixels(c, demo, frame_indices, source="segmented", output_dir=None):
    """Extract peg/holder/hand RGB pixels for requested frames from both cameras.

    source='segmented' uses masks in c['output_dir']; source='hdf5' runs the three
    configured text prompts independently on just the requested HDF5 frames.
    """
    import h5py
    if source not in {"segmented", "hdf5"}:
        raise ValueError("source must be 'segmented' or 'hdf5'")
    destination = Path(output_dir).expanduser().resolve() if output_dir else Path(c["output_dir"])/"pixels"
    destination.mkdir(parents=True, exist_ok=True)
    manifest=[]
    runner=None
    with h5py.File(c["dataset_path"], "r") as h5:
        demo=_resolve_demo(h5, demo)
        for role, serial in camera_roles(c):
            camera=group(h5, demo, serial)
            indices = (
                _automatic_frames(camera["host_timestamp_ns"][:], c["onset"])
                if frame_indices is None
                else _validate_frames(frame_indices, len(camera["rgb"]), demo, role)
            )
            masks=None
            if source == "segmented":
                masks, _ = load_saved_masks(c["output_dir"], demo, role)
                if "peg" not in masks:
                    raise ValueError(f"Saved masks for {demo}/{role} are missing peg")
            else:
                if runner is None:
                    from .sam3_runner import Runner
                    runner=Runner(c["sam3"])
            for i in indices:
                frame=np.asarray(camera["rgb"][i])
                frame_masks = {}
                object_counts = {}
                objects = configured_objects(c, role)
                for name in objects:
                    if masks is None:
                        mask, _ = runner.prompt_image(
                            frame, objects[name],
                            destination/"work"/demo/role/f"frame_{i:06d}"/name,
                        )
                    else:
                        sequence = masks.get(name)
                        mask = sequence[i] if sequence is not None else None
                    frame_masks[name] = mask
                    object_counts[name] = int(np.asarray(mask, dtype=bool).sum()) if mask is not None else 0
                out=destination/demo/role/f"frame_{i:06d}.png"
                count=_save_rgb(out, frame, frame_masks)
                manifest.append(dict(
                    demo=demo, role=role,
                    camera_serial=str(c[f"{role}_camera_serial"]),
                    frame_index=i, pixel_count=count,
                    object_pixel_counts=object_counts,
                    file=str(out.relative_to(destination)),
                ))
    manifest_path=destination/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest)} RGB extraction image(s) into {destination}")
    print(f"Saved manifest: {manifest_path}")
    return destination


def extract_pixels(c, demo=None, frame_indices=None, source="segmented", output_dir=None):
    """Extract selected frames for one demo, or every complete demo when omitted."""
    import h5py
    if demo is not None:
        return _extract_demo_pixels(c, demo, frame_indices, source=source, output_dir=output_dir)
    with h5py.File(c["dataset_path"], "r") as h5:
        selected = demos(h5)
    if not selected:
        raise RuntimeError("No complete demonstrations found")
    destination = None
    for selected_demo in selected:
        destination = _extract_demo_pixels(
            c, selected_demo, frame_indices, source=source, output_dir=output_dir
        )
    return destination
