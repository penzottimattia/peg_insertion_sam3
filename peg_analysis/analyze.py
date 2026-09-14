from pathlib import Path
import json
import warnings
import numpy as np
from .core import (axial_above_thumb_length, demos, detect_gap, dimension_scale,
                   camera_roles, feat, group, lateral_thumb_lower_bbox_distance, peg_angle_deg, robust_noise)


def selected_demos(h, requested_demo=None):
    available = demos(h)
    if requested_demo is None:
        return available
    if requested_demo not in available:
        raise ValueError(f"Unknown complete demo {requested_demo!r}. Available: {', '.join(available)}")
    return [requested_demo]


def load_saved_masks(output_dir, demo, role):
    masks_dir = Path(output_dir) / "masks"
    mask_path = masks_dir / f"{demo}_{role}.npz"
    meta_path = masks_dir / f"{demo}_{role}_meta.json"
    if not mask_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing saved segmentation for {demo}/{role}. Run: peg-analysis -c config.yaml segment --demo {demo}")
    with np.load(mask_path) as archive:
        masks = {name: archive[name].astype(bool) for name in archive.files}
    return masks, json.loads(meta_path.read_text())


def _median_pre(x, column, insertion_start, window):
    return x.loc[x[column].notna() & (x.frame_index < insertion_start), column].tail(window).median()


def _noise_fields(prefix, values):
    n, std, mad = robust_noise(values)
    return {f"pre_{prefix}_n": n, f"pre_{prefix}_std": std, f"pre_{prefix}_mad": mad}


def _first_valid_pre_gap(features, pre_gap_frame):
    """Return the first valid feature at or before the last pre-gap frame."""
    stop = min(int(pre_gap_frame), len(features) - 1)
    for frame_index in range(max(stop + 1, 0)):
        if features[frame_index] is not None:
            return frame_index, features[frame_index]
    return None, None


def _analyze_impl(c, requested_demo=None):
    import h5py
    import pandas as pd
    output_dir = Path(c["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    pre_window = int(c["onset"]["pre_window_frames"])
    with h5py.File(c["dataset_path"], "r") as h:
        for demo in selected_demos(h, requested_demo):
            camera_frames = []
            for role, serial in camera_roles(c):
                g = group(h, demo, serial)
                timestamps = g["host_timestamp_ns"][:]
                masks, meta = load_saved_masks(output_dir, demo, role)
                if "peg" not in masks or len(masks["peg"]) != len(timestamps):
                    raise ValueError(f"Invalid saved peg masks for {demo}/{role}")
                for name, sequence in masks.items():
                    if len(sequence) != len(timestamps):
                        raise ValueError(f"Invalid saved masks for {demo}/{role}/{name}")
                if "insertion_start_frame" in meta:
                    insertion_start = int(meta["insertion_start_frame"])
                    pre_gap_frame = int(meta.get("pre_gap_frame", insertion_start - 1))
                else:
                    pre_gap_frame, insertion_start, _ = detect_gap(
                        timestamps, c["onset"]["gap_mad_multiplier"],
                        c["onset"]["gap_nominal_multiplier"])
                peg_features = [feat(m) for m in masks["peg"]]
                hand_masks = masks.get("hand")
                hand_features = ([feat(m) for m in hand_masks] if hand_masks is not None
                                 else [None] * len(timestamps))
                scale_frame_index, scale_feature = _first_valid_pre_gap(
                    peg_features, pre_gap_frame
                )
                if scale_feature is None:
                    raise RuntimeError(
                        f"No valid pre-gap peg mask for scale: {demo}/{role}"
                    )
                if scale_frame_index != 0:
                    warnings.warn(
                        f"Invalid peg mask at frame 0 for {demo}/{role}; "
                        f"using first valid pre-gap frame {scale_frame_index} for scale",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if role == "main":
                    mm_per_px = dimension_scale(
                        scale_feature, c["dimensions"].get("peg_length_mm"), "visible_length")
                    scale_basis = "peg_length"
                else:
                    mm_per_px = dimension_scale(
                        scale_feature, c["dimensions"].get("peg_width_mm"), "visible_width")
                    scale_basis = "peg_width"
                if not np.isfinite(mm_per_px):
                    warnings.warn(f"Missing {scale_basis} scale for {demo}/{role}; millimetre outputs will be missing")
                rows = []
                for i, timestamp in enumerate(timestamps):
                    peg, hand = peg_features[i], hand_features[i]
                    row = dict(role=role, frame_index=i, host_timestamp_ns=int(timestamp),
                               elapsed_time_from_insertion_start_s=(int(timestamp)-int(timestamps[insertion_start]))/1e9,
                               phase="pre" if i < insertion_start else "insertion",
                               pre_gap_frame=pre_gap_frame,
                               insertion_start_frame=insertion_start,
                               peg_valid=peg is not None, hand_valid=hand is not None)
                    if peg is not None:
                        angle = peg_angle_deg(peg["axis"])
                        row.update(peg_center_x_px=peg["c"][0], peg_center_y_px=peg["c"][1],
                                   peg_axis_x=peg["axis"][0], peg_axis_y=peg["axis"][1],
                                   peg_visible_length_px=peg["visible_length"],
                                   peg_visible_width_px=peg["visible_width"],
                                   peg_top_x_px=peg["endpoint_top"][0], peg_top_y_px=peg["endpoint_top"][1],
                                   peg_angle_deg=angle, peg_angular_error_deg=abs(angle))
                    if hand is not None:
                        row.update(thumb_tip_x_px=hand["thumb_tip"][0], thumb_tip_y_px=hand["thumb_tip"][1])
                    if role == "secondary":
                        row["thumb_lower_bbox_corner_distance_px"] = np.nan
                    if peg is not None and hand is not None:
                        if role == "main":
                            row["peg_above_thumb_length_px"] = axial_above_thumb_length(peg, hand)
                        else:
                            lateral = lateral_thumb_lower_bbox_distance(
                                masks["peg"][i], hand_masks[i]
                            )
                            if lateral is not None:
                                row["thumb_lower_bbox_corner_distance_px"] = lateral["distance"]
                    rows.append(row)
                x = pd.DataFrame(rows)
                for optional_column in ("peg_above_thumb_length_px", "thumb_lower_bbox_corner_distance_px"):
                    if optional_column not in x:
                        x[optional_column] = np.nan
                baseline_length = _median_pre(x, "peg_visible_length_px", insertion_start, pre_window)
                if not np.isfinite(baseline_length):
                    raise RuntimeError(f"No valid pre-insertion peg lengths: {demo}/{role}")
                if role == "main":
                    x["insertion_depth_valid"] = x.peg_valid
                    x["insertion_depth_px"] = baseline_length - x["peg_visible_length_px"]
                    x["insertion_depth_mm"] = x.insertion_depth_px * mm_per_px
                    axial0 = _median_pre(x, "peg_above_thumb_length_px", insertion_start, pre_window)
                    if not np.isfinite(axial0):
                        warnings.warn(f"No valid pre-insertion axial baseline for {demo}; axial-slip output will be missing")
                    x["axial_slip_valid"] = x.peg_valid & x.hand_valid & x["peg_above_thumb_length_px"].notna() & np.isfinite(axial0)
                    x["axial_slip_px"] = x.peg_above_thumb_length_px - axial0
                    x["axial_slip_mm"] = x.axial_slip_px * mm_per_px
                else:
                    lateral0 = _median_pre(x, "thumb_lower_bbox_corner_distance_px", insertion_start, pre_window)
                    if not np.isfinite(lateral0):
                        warnings.warn(f"No valid pre-insertion lateral baseline for {demo}; lateral-slip output will be missing")
                    x["lateral_slip_valid"] = x.peg_valid & x.hand_valid & x["thumb_lower_bbox_corner_distance_px"].notna() & np.isfinite(lateral0)
                    x["lateral_slip_px"] = x.thumb_lower_bbox_corner_distance_px - lateral0
                    x["lateral_slip_mm"] = x.lateral_slip_px * mm_per_px
                x["mm_per_px"] = mm_per_px
                x["scale_basis"] = scale_basis
                x["scale_frame_index"] = int(scale_frame_index)
                x["visible_length_baseline_px"] = baseline_length
                camera_frames.append(x)
            combined = pd.concat(camera_frames, ignore_index=True)
            timeseries_columns = [
                "role", "frame_index", "host_timestamp_ns",
                "elapsed_time_from_insertion_start_s", "phase",
                "peg_valid", "hand_valid",
                "insertion_depth_valid", "insertion_depth_px",
                "axial_slip_valid", "axial_slip_px",
                "lateral_slip_valid", "lateral_slip_px",
                "peg_angle_deg", "peg_angular_error_deg",
            ]
            compact = combined.reindex(columns=timeseries_columns)
            compact.to_csv(output_dir / f"{demo}_timeseries.csv", index=False)
            main = combined[combined.role == "main"].copy()
            secondary = combined[combined.role == "secondary"].copy()
            main_insert = main[main.phase == "insertion"]
            secondary_insert = secondary[secondary.phase == "insertion"] if len(secondary) else secondary
            depth_valid = main_insert[main_insert.insertion_depth_valid & main_insert.insertion_depth_mm.notna()]
            if depth_valid.empty:
                raise RuntimeError(f"No valid main-camera insertion-phase depth metrics: {demo}")
            max_depth_idx = depth_valid.insertion_depth_mm.idxmax()
            initial = main.loc[main.frame_index == int(main.insertion_start_frame.iloc[0])].iloc[0]
            final = main.loc[max_depth_idx]
            axial = main_insert.loc[main_insert.axial_slip_valid, "axial_slip_mm"].dropna()
            lateral = (secondary_insert.loc[secondary_insert.lateral_slip_valid, "lateral_slip_mm"].dropna()
                       if "lateral_slip_valid" in secondary_insert else pd.Series(dtype=float))
            summary = dict(
                demo=demo,
                max_insertion_depth_mm=float(final.insertion_depth_mm),
                max_depth_frame=int(final.frame_index),
                max_abs_axial_slip_mm=float(axial.abs().max()) if len(axial) else np.nan,
                max_abs_lateral_slip_mm=float(lateral.abs().max()) if len(lateral) else np.nan,
                initial_angular_error_deg=float(initial.peg_angular_error_deg),
                final_angular_error_deg=float(final.peg_angular_error_deg),
                angular_error_change_deg=float(
                    final.peg_angular_error_deg - initial.peg_angular_error_deg
                ),
                main_valid_frame_fraction=float(main.peg_valid.mean()),
                secondary_valid_frame_fraction=(
                    float(secondary.lateral_slip_valid.mean())
                    if len(secondary) and "lateral_slip_valid" in secondary else np.nan
                ),
            )
            summaries.append(summary)
    return pd.DataFrame(summaries)

def analyze(c, requested_demo=None):
    """Analyze demos independently, warning and continuing after demo-local failures."""
    import h5py
    import pandas as pd

    output_dir = Path(c["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(c["dataset_path"], "r") as h:
        requested = selected_demos(h, requested_demo)

    summaries = []
    skipped = []
    for demo in requested:
        try:
            summaries.append(_analyze_impl(c, demo))
        except Exception as exc:
            warnings.warn(
                f"Skipping {demo}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped.append({
                "demo": demo,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    summary_path = output_dir / "summary.csv"
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    else:
        pd.DataFrame().to_csv(summary_path, index=False)

    skipped_path = output_dir / "skipped_demos.csv"
    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_path, index=False)
        print(f"Warning: skipped {len(skipped)} demo(s)")
        print(f"Saved skipped-demo report: {skipped_path}")
    elif skipped_path.exists() and requested_demo is None:
        skipped_path.unlink()

    print(f"Analyzed {len(summaries)} demo(s)")
    print(f"Saved metrics: {summary_path}")

