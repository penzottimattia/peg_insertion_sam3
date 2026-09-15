"""Export CLI-prompted objects from key frames of an HDF5 demonstration dataset."""
from pathlib import Path
import json
import re

import numpy as np

from .core import demos, detect_gap


def parse_prompt(value):
    """Parse NAME=TEXT, preserving spaces in the SAM text prompt."""
    name, separator, text = str(value).partition("=")
    name = name.strip()
    text = text.strip()
    if not separator or not name or not text:
        raise ValueError("prompts must use NAME=TEXT with a non-empty name and text")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(
            f"invalid prompt name {name!r}; use letters, digits, '.', '_' or '-'"
        )
    return name, text


def parse_prompts(values):
    prompts = {}
    for value in values:
        name, text = parse_prompt(value)
        if name in prompts:
            raise ValueError(f"duplicate prompt name: {name}")
        prompts[name] = text
    if not prompts:
        raise ValueError("at least one --prompt NAME=TEXT is required")
    return prompts


def resolve_demo(h5, requested):
    available = demos(h5)
    if requested in available:
        return requested
    try:
        candidate = f"demo_{int(requested):06d}"
    except (TypeError, ValueError):
        candidate = None
    if candidate in available:
        return candidate
    raise ValueError(
        f"Unknown complete demo {requested!r}. Available: {', '.join(available)}"
    )


def keyframes(timestamps, onset):
    """Return labeled first, last-pre-gap, and final frame indices."""
    if len(timestamps) < 2:
        raise ValueError("At least two frames are required for key-frame export")
    pre_gap, _, gap_ns = detect_gap(
        timestamps,
        onset["gap_mad_multiplier"],
        onset["gap_nominal_multiplier"],
    )
    selected = [("first", 0), ("pre_gap", int(pre_gap)), ("last", len(timestamps) - 1)]
    # Retain semantic labels even when a very short demo maps labels to one frame.
    return selected, int(gap_ns)


def save_merged_png(path, frame, masks):
    """Save the union of all prompted masks as one transparent RGBA image."""
    import cv2

    frame = np.asarray(frame, dtype=np.uint8)
    union = np.zeros(frame.shape[:2], dtype=bool)
    for mask in masks.values():
        if mask is None:
            continue
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != frame.shape[:2]:
            raise ValueError("SAM mask shape does not match RGB frame shape")
        union |= selected
    rgba = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)
    rgba[union, :3] = frame[union]
    rgba[union, 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
        raise IOError(f"Could not write image: {path}")
    return int(union.sum())


def _camera_groups(h5, demo, requested_serials=None):
    cameras = h5[f"demos/{demo}/cameras"]
    available = sorted(cameras.keys())
    selected = available if not requested_serials else [str(x) for x in requested_serials]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown camera(s) for {demo}: {', '.join(unknown)}. "
            f"Available: {', '.join(available)}"
        )
    return [(serial, cameras[serial]) for serial in selected]


def export_objects(c, prompt_values, requested_demo=None, camera_serials=None, output_dir=None):
    """Prompt each selected key frame independently and export per-object RGBA PNGs."""
    import h5py
    from .sam3_runner import Runner

    prompts = parse_prompts(prompt_values)
    dataset_path = Path(c["dataset_path"]).expanduser().resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else dataset_path.with_suffix("").with_name(dataset_path.stem + "_object_export")
    )
    destination.mkdir(parents=True, exist_ok=True)
    runner = Runner(c["sam3"])
    manifest = []

    with h5py.File(dataset_path, "r") as h5:
        selected_demos = (
            [resolve_demo(h5, requested_demo)] if requested_demo is not None else demos(h5)
        )
        if not selected_demos:
            raise RuntimeError("No complete demonstrations found")
        for demo in selected_demos:
            for serial, camera in _camera_groups(h5, demo, camera_serials):
                frames, gap_ns = keyframes(camera["host_timestamp_ns"][:], c["onset"])
                for frame_label, frame_index in frames:
                    frame = np.asarray(camera["rgb"][frame_index])
                    frame_masks = {}
                    object_results = {}
                    for object_name, prompt_text in prompts.items():
                        work = destination / "work" / demo / serial / frame_label / object_name
                        mask, info = runner.prompt_image(frame, prompt_text, work)
                        frame_masks[object_name] = mask
                        object_results[object_name] = {
                            "prompt": prompt_text,
                            "matched": bool(info.get("matched", False)),
                            "pixel_count": (
                                int(np.asarray(mask, dtype=bool).sum())
                                if mask is not None else 0
                            ),
                            "sam": info,
                        }
                    relative = Path(demo) / serial / f"{frame_label}.png"
                    union_pixel_count = save_merged_png(
                        destination / relative, frame, frame_masks
                    )
                    manifest.append({
                        "demo": demo,
                        "camera_serial": serial,
                        "frame_label": frame_label,
                        "frame_index": int(frame_index),
                        "host_timestamp_ns": int(camera["host_timestamp_ns"][frame_index]),
                        "timestamp_gap_ns": gap_ns,
                        "union_pixel_count": union_pixel_count,
                        "objects": object_results,
                        "file": str(relative),
                    })

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Exported {len(manifest)} merged frame image(s) from "
        f"{len({row['demo'] for row in manifest})} demo(s) into {destination}"
    )
    print(f"Saved manifest: {manifest_path}")
    return destination
