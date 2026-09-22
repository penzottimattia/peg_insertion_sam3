"""Render cumulative-comparison trials as tolerance-specific video galleries."""
from pathlib import Path
import json

import numpy as np

from .cumulative import load_cumulative_config


def _truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _source_dir(row):
    return Path(str(row["source_summary"])).expanduser().resolve().parent


def _dataset_path(source_dir, demo):
    candidates = sorted((source_dir / "masks").glob(f"{demo}_*_meta.json"))
    if not candidates:
        raise FileNotFoundError(f"No mask metadata for {demo} in {source_dir / 'masks'}")
    meta = json.loads(candidates[0].read_text())
    value = meta.get("dataset_path")
    if not value:
        raise ValueError(f"Mask metadata has no dataset_path: {candidates[0]}")
    return Path(value).expanduser().resolve()


def _camera_serial(source_dir, demo):
    meta_path = source_dir / "masks" / f"{demo}_main_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Main-camera metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    return str(meta["camera_serial"])


def _shallow_start_frame(row, shallow_depth_mm):
    """Infer first shallow-engagement frame from compact depth pixels and summary scale."""
    import pandas as pd

    source_dir = _source_dir(row)
    path = source_dir / f"{row['demo']}_timeseries.csv"
    series = pd.read_csv(path)
    main = series[(series.role == "main") & (series.phase == "insertion")].copy()
    depth_px = pd.to_numeric(main.get("insertion_depth_px"), errors="coerce")
    max_frame = int(row["max_depth_frame"])
    at_max = main.loc[main.frame_index == max_frame]
    if at_max.empty:
        raise ValueError(f"max_depth_frame {max_frame} absent from {path}")
    max_px = float(pd.to_numeric(at_max.iloc[0]["insertion_depth_px"], errors="coerce"))
    max_mm = float(row["max_insertion_depth_mm"])
    mm_per_px = max_mm / max_px if np.isfinite(max_px) and abs(max_px) > 1e-12 else np.nan
    if np.isfinite(mm_per_px) and mm_per_px > 0:
        depth_mm = depth_px * mm_per_px
        matches = main.loc[depth_mm >= float(shallow_depth_mm), "frame_index"]
        if len(matches):
            return int(matches.iloc[0])
    # Robust fallback: insertion onset is the earliest insertion-phase frame.
    if len(main):
        return int(main.frame_index.iloc[0])
    raise ValueError(f"No insertion-phase frames in {path}")


def _highlight_frame(row, start_frame, last_frame, timestamps=None, early_seconds=0.0):
    """Resolve the global highlight frame, optionally shifting it earlier in real time.

    Missing or pre-engagement highlights keep the last-frame fallback. For a
    valid highlight, ``early_seconds`` is applied using source HDF5 timestamps,
    not the output MP4 frame rate, then clamped to shallow engagement.
    """
    value = row.get("highlight", np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    if not np.isfinite(value) or int(value) < int(start_frame):
        return int(last_frame)

    resolved = min(max(int(value), int(start_frame)), int(last_frame))
    early_seconds = max(float(early_seconds), 0.0)
    if early_seconds <= 0.0 or timestamps is None:
        return resolved

    timestamps = np.asarray(timestamps, dtype=np.int64)
    target_ns = int(timestamps[resolved]) - round(early_seconds * 1_000_000_000)
    candidates = np.arange(int(start_frame), resolved + 1, dtype=int)
    eligible = candidates[timestamps[candidates] <= target_ns]
    return int(eligible[-1]) if len(eligible) else int(start_frame)


def _read_trial(row, shallow_depth_mm):
    import cv2
    import h5py

    source_dir = _source_dir(row)
    demo = str(row["demo"])
    dataset = _dataset_path(source_dir, demo)
    serial = _camera_serial(source_dir, demo)
    start = _shallow_start_frame(row, shallow_depth_mm)
    with h5py.File(dataset, "r") as h5:
        frames = h5[f"demos/{demo}/cameras/{serial}/rgb"]
        recorded_last = len(frames) - 1
        truncate_at_max_depth = _truthy(row.get("_gallery_truncate_at_max_depth", False))
        last = min(int(row["max_depth_frame"]), recorded_last) if truncate_at_max_depth else recorded_last
        start = min(max(start, 0), last)
        timestamps = h5[f"demos/{demo}/cameras/{serial}/host_timestamp_ns"][:]
        early_seconds = float(row.get("_gallery_highlight_early_seconds", 0.0))
        highlight = _highlight_frame(row, start, last, timestamps, early_seconds)
        data = [np.asarray(frames[i], dtype=np.uint8) for i in range(start, last + 1)]
    return data, start, highlight


def _fit(frame, width, height):
    import cv2
    frame = np.asarray(frame, dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _draw_cell(frame, success, highlighted, cell_size):
    import cv2
    width, height = cell_size
    image = _fit(frame, width, height)
    if highlighted:
        color = (38, 190, 70) if success else (220, 45, 45)  # RGB
        t = max(5, round(min(width, height) * 0.018))
        cv2.rectangle(image, (t // 2, t // 2), (width - 1 - t // 2, height - 1 - t // 2), color, t)
    return image


def _fill_group(group, count, rng):
    """Fill a method group to count by deterministic sampling with replacement."""
    if len(group) == 0:
        return group
    if len(group) >= count:
        return group.iloc[:count].copy()
    extra = group.iloc[rng.integers(0, len(group), size=count - len(group))].copy()
    extra["gallery_duplicate"] = True
    result = group.copy()
    result["gallery_duplicate"] = False
    return __import__("pandas").concat([result, extra], ignore_index=True, sort=False)



def _trial_success(row, threshold, anomalous_depth_mm=None):
    """Classify a trial; excessive depth is an optional anomalous failure."""
    depth = float(row["max_insertion_depth_mm"])
    succeeds = depth >= float(threshold)
    if anomalous_depth_mm is not None and depth >= float(anomalous_depth_mm):
        succeeds = False
    return succeeds

def render_cumulative_gallery(spec_path, trials_path=None, output_dir=None):
    """Create one MP4 per tolerance from cumulative plotted trials."""
    import cv2
    import pandas as pd

    spec_path = Path(spec_path).expanduser().resolve()
    cfg = load_cumulative_config(spec_path)
    raw = json.loads(spec_path.read_text())
    gallery = raw.get("video_gallery", {}) or {}
    shallow = float(gallery.get("shallow_engagement_depth_mm", 2.0))
    fps = float(gallery.get("fps", 30.0))
    cell_width = int(gallery.get("cell_width", 320))
    cell_height = int(gallery.get("cell_height", 240))
    seed = int(gallery.get("seed", 0))
    truncate_at_max_depth = bool(gallery.get("truncate_at_max_depth", False))
    group_by_tolerance = bool(gallery.get("group_by_tolerance", True))
    highlight_early_seconds = float(gallery.get("highlight_early_seconds", 0.0))
    anomalous_depth_mm = gallery.get("anomalous_depth_mm")
    anomalous_depth_mm = None if anomalous_depth_mm is None else float(anomalous_depth_mm)
    first_n = gallery.get("first_n")
    last_n = gallery.get("last_n")
    first_n = None if first_n is None else int(first_n)
    last_n = None if last_n is None else int(last_n)
    slots = int(gallery.get("trials_per_method", 10))
    if slots != 10:
        raise ValueError("video_gallery.trials_per_method must be 10 for the 5x2/2x5 layout")
    if shallow < 0 or fps <= 0 or cell_width < 32 or cell_height < 32:
        raise ValueError("Invalid video_gallery shallow depth, fps, or cell dimensions")
    if not np.isfinite(highlight_early_seconds) or highlight_early_seconds < 0:
        raise ValueError("video_gallery.highlight_early_seconds must be at least 0")
    if anomalous_depth_mm is not None and not np.isfinite(anomalous_depth_mm):
        raise ValueError("video_gallery.anomalous_depth_mm must be finite")
    if first_n is not None and last_n is not None:
        raise ValueError("video_gallery.first_n and last_n are mutually exclusive")
    if first_n is not None and first_n < 1:
        raise ValueError("video_gallery.first_n must be at least 1")
    if last_n is not None and last_n < 1:
        raise ValueError("video_gallery.last_n must be at least 1")

    if output_dir is None:
        output_dir = spec_path.parent / "cumulative_plots"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if trials_path is None:
        trials_path = output_dir / "plotted_trials.csv"
    trials_path = Path(trials_path).expanduser().resolve()
    data = pd.read_csv(trials_path)
    # Always refresh highlight from each original summary.csv. plotted_trials.csv
    # may have been created before highlight existed or may simply be stale.
    data["highlight"] = np.nan
    if "source_summary" not in data.columns or "demo" not in data.columns:
        raise ValueError("Gallery trials require source_summary and demo columns")
    summary_cache = {}
    for index, trial in data.iterrows():
        summary_path = Path(str(trial["source_summary"])).expanduser().resolve()
        if summary_path not in summary_cache:
            if not summary_path.is_file():
                raise FileNotFoundError(f"Source summary CSV not found: {summary_path}")
            original = pd.read_csv(summary_path)
            if "demo" not in original.columns:
                raise ValueError(f"Source summary CSV has no demo column: {summary_path}")
            summary_cache[summary_path] = original.set_index("demo", drop=False)
        original = summary_cache[summary_path]
        demo = str(trial["demo"])
        if demo in original.index and "highlight" in original.columns:
            match = original.loc[demo]
            if getattr(match, "ndim", 1) > 1:
                match = match.iloc[0]
            data.at[index, "highlight"] = match.get("highlight", np.nan)
    # Synthetic cumulative rows have no physical recording of their own.
    if "synthetic" in data.columns:
        data = data.loc[~data.synthetic.map(_truthy)].copy()
    threshold = cfg["insertion_depth_threshold"]
    if threshold is None:
        raise ValueError("Video gallery requires insertion_depth_threshold to classify success/failure")

    method_order = list(cfg["methods"])
    rng = np.random.default_rng(seed)
    outputs = []
    manifest = []
    # Per-dataset/method galleries are always produced. group_by_tolerance is additive:
    # when enabled, aggregate tolerance galleries are produced as well.
    gallery_units = [
        {"tolerance": item["tolerance"], "method": item["method"], "index": index, "kind": "method"}
        for index, item in enumerate(cfg["datasets"], start=1)
        if ((data.tolerance.astype(str) == str(item["tolerance"])) & (data.method == item["method"])).any()
    ]
    if group_by_tolerance:
        gallery_units.extend(
            {"tolerance": tolerance, "method": None, "index": None, "kind": "tolerance"}
            for tolerance in dict.fromkeys(d["tolerance"] for d in cfg["datasets"])
            if (data.tolerance.astype(str) == str(tolerance)).any()
        )

    for unit in gallery_units:
        tolerance = unit["tolerance"]
        tol = data[data.tolerance.astype(str) == str(tolerance)].copy()
        if unit["method"] is not None:
            tol = tol[tol.method == unit["method"]].copy()
        tol["_gallery_truncate_at_max_depth"] = truncate_at_max_depth
        tol["_gallery_highlight_early_seconds"] = highlight_early_seconds
        present = [m for m in method_order if (tol.method == m).any()]
        groups = []
        for method in present:
            selected = tol[tol.method == method].copy()
            if first_n is not None:
                selected = selected.head(first_n)
            elif last_n is not None:
                selected = selected.tail(last_n)
            group = _fill_group(selected, slots, rng)
            groups.append((method, group))

        # Multiple methods: each is a 2-column x 5-row block. One method: 5 columns x 2 rows.
        rows, cols_per_group = ((2, 5) if len(groups) == 1 else (5, 2))
        total_cols = cols_per_group * len(groups)
        prepared = []
        max_len = 0
        for method, group in groups:
            trials = []
            for _, row in group.iterrows():
                frames, start, highlight = _read_trial(row, shallow)
                success = _trial_success(row, threshold, anomalous_depth_mm)
                trials.append((row, frames, start, highlight, success))
                max_len = max(max_len, len(frames))
            prepared.append((method, trials))

        method_header = 38
        canvas_size = (total_cols * cell_width, rows * cell_height + method_header)
        safe_tol = str(tolerance).replace(".", "_")
        if unit["kind"] == "tolerance":
            path = output_dir / f"gallery_tolerance_{safe_tol}.mp4"
        else:
            method = unit["method"]
            path = output_dir / f"gallery_{unit['index']:02d}_{method}_tolerance_{safe_tol}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas_size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open MP4 writer: {path}")
        try:
            for out_i in range(max_len):
                canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
                for gi, (method, trials) in enumerate(prepared):
                    label = cfg["methods"][method]["label"]
                    x0 = gi * cols_per_group * cell_width
                    cv2.putText(canvas, label, (x0 + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                                (240, 240, 240), 2, cv2.LINE_AA)
                    for slot, (row, frames, start, highlight, success) in enumerate(trials):
                        rr, cc = divmod(slot, cols_per_group)
                        idx = min(out_i, len(frames) - 1)  # hold last frame after a shorter trial ends
                        global_frame = start + idx
                        cell = _draw_cell(frames[idx], success, global_frame >= highlight,
                                          (cell_width, cell_height))
                        y = method_header + rr * cell_height
                        x = x0 + cc * cell_width
                        canvas[y:y + cell_height, x:x + cell_width] = cell
                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        outputs.append(path)
        manifest.append({
            "tolerance": tolerance, "file": path.name, "fps": fps,
            "shallow_engagement_depth_mm": shallow, "insertion_depth_threshold_mm": threshold,
            "truncate_at_max_depth": truncate_at_max_depth,
            "highlight_early_seconds": highlight_early_seconds,
            "anomalous_depth_mm": anomalous_depth_mm,
            "first_n": first_n, "last_n": last_n,
            "group_by_tolerance": group_by_tolerance,
            "gallery_kind": unit["kind"],
            "dataset_index": unit["index"],
            "methods": present, "layout_rows": rows, "columns_per_method": cols_per_group,
        })
    manifest_path = output_dir / "gallery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for path in outputs:
        print(f"Saved gallery video: {path}")
    print(f"Saved gallery manifest: {manifest_path}")
    return outputs

def render_directory_gallery(input_dir, spec_path=None, first_n=None, last_n=None, output=None,
                             fps=None, shallow_depth_mm=None,
                             highlight_early_seconds=None,
                             truncate_at_max_depth=None,
                             insertion_depth_threshold_mm=None,
                             anomalous_depth_mm=None,
                             cell_width=None, cell_height=None):
    """Render a gallery directly from one analyzed directory.

    Trials retain summary.csv order. ``first_n`` and ``last_n`` are mutually
    exclusive selectors; when neither is given all physical summary rows are
    used. Unlike cumulative galleries, no duplication is performed.
    """
    import cv2
    import pandas as pd

    spec_raw = {}
    if spec_path is not None:
        spec_path = Path(spec_path).expanduser().resolve()
        spec_raw = json.loads(spec_path.read_text())
    gallery = spec_raw.get("video_gallery", {}) or {}

    # CLI values override video_gallery values; hard defaults apply only when both are absent.
    def choose(cli_value, key, default):
        return gallery.get(key, default) if cli_value is None else cli_value

    first_n = choose(first_n, "first_n", None)
    last_n = choose(last_n, "last_n", None)
    fps = float(choose(fps, "fps", 24.0))
    shallow_depth_mm = float(choose(shallow_depth_mm, "shallow_engagement_depth_mm", 2.0))
    highlight_early_seconds = float(choose(highlight_early_seconds, "highlight_early_seconds", 0.0))
    truncate_at_max_depth = bool(choose(truncate_at_max_depth, "truncate_at_max_depth", False))
    insertion_depth_threshold_mm = choose(insertion_depth_threshold_mm, "insertion_depth_threshold", None)
    if insertion_depth_threshold_mm is None:
        insertion_depth_threshold_mm = spec_raw.get("insertion_depth_threshold", 15.0)
    insertion_depth_threshold_mm = float(insertion_depth_threshold_mm)
    anomalous_depth_mm = choose(anomalous_depth_mm, "anomalous_depth_mm", None)
    anomalous_depth_mm = None if anomalous_depth_mm is None else float(anomalous_depth_mm)
    cell_width = int(choose(cell_width, "cell_width", 320))
    cell_height = int(choose(cell_height, "cell_height", 240))
    first_n = None if first_n is None else int(first_n)
    last_n = None if last_n is None else int(last_n)

    if first_n is not None and last_n is not None:
        raise ValueError("first_n and last_n are mutually exclusive")
    input_dir = Path(input_dir).expanduser().resolve()
    summary_path = input_dir / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    for name, value in (("first_n", first_n), ("last_n", last_n)):
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} must be at least 1")
    if fps <= 0 or cell_width < 32 or cell_height < 32:
        raise ValueError("Invalid gallery fps or cell dimensions")
    if shallow_depth_mm < 0 or highlight_early_seconds < 0:
        raise ValueError("Shallow depth and highlight_early_seconds must be non-negative")
    if anomalous_depth_mm is not None and not np.isfinite(anomalous_depth_mm):
        raise ValueError("anomalous_depth_mm must be finite")

    data = pd.read_csv(summary_path).copy()
    required = {"demo", "max_insertion_depth_mm", "max_depth_frame"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing: {', '.join(sorted(missing))}")
    if first_n is not None:
        data = data.head(int(first_n)).copy()
    elif last_n is not None:
        data = data.tail(int(last_n)).copy()
    if data.empty:
        raise ValueError("No trials selected")

    data["source_summary"] = str(summary_path)
    if "highlight" not in data.columns:
        data["highlight"] = np.nan
    data["_gallery_truncate_at_max_depth"] = bool(truncate_at_max_depth)
    data["_gallery_highlight_early_seconds"] = float(highlight_early_seconds)

    prepared = []
    max_len = 0
    audit = []
    for _, row in data.iterrows():
        frames, start, highlight = _read_trial(row, float(shallow_depth_mm))
        success = _trial_success(row, insertion_depth_threshold_mm, anomalous_depth_mm)
        prepared.append((row, frames, start, highlight, success))
        max_len = max(max_len, len(frames))
        audit.append({
            "demo": row["demo"],
            "source_highlight": None if not np.isfinite(row.get("highlight", np.nan)) else int(row["highlight"]),
            "display_start_frame": int(start),
            "display_last_frame": int(start + len(frames) - 1),
            "effective_highlight_frame": int(highlight),
            "success": bool(success),
        })

    count = len(prepared)
    # Keep the established 5-column visual grammar; additional trials add rows.
    cols = min(5, count)
    rows = int(np.ceil(count / cols))
    canvas_size = (cols * int(cell_width), rows * int(cell_height))
    output_path = (Path(output).expanduser().resolve() if output else
                   input_dir / "gallery.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), canvas_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {output_path}")
    try:
        for out_i in range(max_len):
            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            for slot, (row, frames, start, highlight, success) in enumerate(prepared):
                rr, cc = divmod(slot, cols)
                idx = min(out_i, len(frames) - 1)
                global_frame = start + idx
                cell = _draw_cell(frames[idx], success, global_frame >= highlight,
                                  (int(cell_width), int(cell_height)))
                y, x = rr * int(cell_height), cc * int(cell_width)
                canvas[y:y + int(cell_height), x:x + int(cell_width)] = cell
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    audit_path = output_path.with_name(output_path.stem + "_highlights.csv")
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    print(f"Saved directory gallery: {output_path}")
    print(f"Saved highlight audit: {audit_path}")
    return output_path
