"""Render a real-pixel trajectory overlay for one demonstration."""
from pathlib import Path

import numpy as np

from .analyze import load_saved_masks
from .core import group


_ROLE_TITLE = {"main": "Main camera", "secondary": "Secondary camera"}


def _sample_indices(start, stop, count):
    """Return unique, evenly spaced indices including start and stop."""
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    if count < 2:
        raise ValueError("traces must be at least 2")
    count = min(int(count), stop - start + 1)
    return np.unique(np.rint(np.linspace(start, stop, count)).astype(int))


def _main_target_frame(output_dir, demo, insertion_start, frame_count):
    """Prefer the measured maximum-depth frame, falling back to the last frame."""
    import pandas as pd

    summary_path = Path(output_dir) / "summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        if "demo" in summary and "max_depth_frame" in summary:
            match = summary.loc[summary["demo"] == demo]
            if len(match):
                target = int(match.iloc[0]["max_depth_frame"])
                return min(max(target, insertion_start), frame_count - 1)
    return frame_count - 1


def _nearest_timestamp_frame(timestamps, target_timestamp, insertion_start):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    candidates = np.arange(insertion_start, len(timestamps))
    if not len(candidates):
        return len(timestamps) - 1
    distances = np.abs(timestamps[candidates] - int(target_timestamp))
    return int(candidates[int(np.argmin(distances))])


def _crop_box(masks, indices, image_shape, padding_fraction=0.12):
    """Find a stable task crop. Masks are never rendered."""
    height, width = image_shape[:2]
    union = np.zeros((height, width), dtype=bool)
    for name in ("peg", "holder", "hand"):
        sequence = masks.get(name)
        if sequence is None:
            continue
        for index in indices:
            union |= np.asarray(sequence[index], dtype=bool)

    y, x = np.nonzero(union)
    if not len(x):
        return 0, 0, width, height
    x0, x1 = int(x.min()), int(x.max()) + 1
    y0, y1 = int(y.min()), int(y.max()) + 1
    padding = max(12, int(padding_fraction * max(x1 - x0, y1 - y0)))
    return (
        max(0, x0 - padding), max(0, y0 - padding),
        min(width, x1 + padding), min(height, y1 + padding),
    )


def _rgb_trace(reference_frame, frames, peg_masks, indices, alpha=0.82):
    """Overlay real RGB peg pixels from many frames onto one RGB frame.

    The binary SAM masks act only as alpha mattes. No mask colours, heatmaps,
    fills, or contours are drawn. Earlier samples are composited first and the
    maximum-depth sample is fully opaque.
    """
    reference = np.asarray(reference_frame, dtype=np.uint8)
    canvas = reference.astype(np.float32).copy()

    for order, frame_index in enumerate(indices):
        mask = np.asarray(peg_masks[int(frame_index)], dtype=bool)
        if not mask.any():
            continue
        frame = np.asarray(frames[int(frame_index)], dtype=np.uint8)
        if frame.shape != reference.shape:
            raise ValueError("All RGB frames must have the same shape")
        sample_alpha = 1.0 if order == len(indices) - 1 else float(alpha)
        canvas[mask] = (
            (1.0 - sample_alpha) * canvas[mask]
            + sample_alpha * frame[mask]
        )

    return canvas.clip(0, 255).astype(np.uint8)


def render_demo(c, demo, roles=("main", "secondary"), traces=12,
                alpha=0.82, output_path=None, crop=True):
    """Render one real-pixel trajectory image for each selected camera.

    The last pre-insertion RGB frame is the fixed background. Saved peg masks
    select the original peg pixels from evenly sampled frames between insertion
    onset and maximum measured depth. All selected appearances are composited
    into one image, producing a photographic trace rather than a mask plot.
    """
    import h5py
    import matplotlib.pyplot as plt

    if traces < 2:
        raise ValueError("traces must be at least 2")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    unknown = set(roles) - {"main", "secondary"}
    if unknown:
        raise ValueError(f"Unknown camera role(s): {', '.join(sorted(unknown))}")
    if not roles:
        raise ValueError("At least one camera role is required")

    panels = []
    with h5py.File(c["dataset_path"], "r") as h5:
        main_camera = group(h5, demo, c["main_camera_serial"])
        _, main_meta = load_saved_masks(c["output_dir"], demo, "main")
        main_start = int(main_meta["insertion_start_frame"])
        main_target = _main_target_frame(
            c["output_dir"], demo, main_start, len(main_camera["rgb"])
        )
        target_timestamp = int(main_camera["host_timestamp_ns"][main_target])

        for role in roles:
            camera = group(h5, demo, c[f"{role}_camera_serial"])
            masks, metadata = load_saved_masks(c["output_dir"], demo, role)
            start = int(metadata["insertion_start_frame"])
            target = main_target if role == "main" else _nearest_timestamp_frame(
                camera["host_timestamp_ns"][:], target_timestamp, start
            )
            indices = _sample_indices(start, target, traces)
            reference_index = int(metadata.get("pre_gap_frame", max(start - 1, 0)))
            reference_index = min(max(reference_index, 0), len(camera["rgb"]) - 1)
            reference = np.asarray(camera["rgb"][reference_index])
            image = _rgb_trace(reference, camera["rgb"], masks["peg"], indices, alpha)

            if crop:
                x0, y0, x1, y1 = _crop_box(masks, indices, image.shape)
                image = image[y0:y1, x0:x1]
            panels.append((_ROLE_TITLE[role], image))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(7 * len(panels), 6), squeeze=False
    )
    for ax, (title, image) in zip(axes[0], panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{demo}: task evolution trace")
    fig.tight_layout()

    if output_path is None:
        output_path = Path(c["output_dir"]) / "renders" / f"{demo}_task_trace.png"
    else:
        output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved task trace: {output_path}")
    return output_path
