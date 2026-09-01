from pathlib import Path
import json
import warnings
import numpy as np
from .core import (axial_above_thumb_length, demos, detect_gap, dimension_scale,
                   feat, group, peg_angle_deg, robust_noise,
                   signed_point_axis_distance)


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
            for role, serial in (("main", c["main_camera_serial"]), ("secondary", c["secondary_camera_serial"])):
                g = group(h, demo, serial)
                timestamps = g["host_timestamp_ns"][:]
                masks, meta = load_saved_masks(output_dir, demo, role)
                for name in ("peg", "hand", "holder"):
                    if name not in masks or len(masks[name]) != len(timestamps):
                        raise ValueError(f"Invalid saved masks for {demo}/{role}/{name}")
                if "insertion_start_frame" in meta:
                    insertion_start = int(meta["insertion_start_frame"])
                    pre_gap_frame = int(meta.get("pre_gap_frame", insertion_start - 1))
                else:
                    pre_gap_frame, insertion_start, _ = detect_gap(
                        timestamps, c["onset"]["gap_mad_multiplier"],
                        c["onset"]["gap_nominal_multiplier"])
                peg_features = [feat(m) for m in masks["peg"]]
                hand_features = [feat(m) for m in masks["hand"]]
                if peg_features[0] is None:
                    raise RuntimeError(f"Invalid peg mask at scale frame 0: {demo}/{role}")
                if role == "main":
                    mm_per_px = dimension_scale(
                        peg_features[0], c["dimensions"].get("peg_length_mm"), "visible_length")
                    scale_basis = "peg_length"
                else:
                    mm_per_px = dimension_scale(
                        peg_features[0], c["dimensions"].get("peg_width_mm"), "visible_width")
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
                    if peg is not None and hand is not None:
                        row["thumb_peg_signed_distance_px"] = signed_point_axis_distance(hand["thumb_tip"], peg["c"], peg["normal"])
                        if role == "main":
                            row["peg_above_thumb_length_px"] = axial_above_thumb_length(peg, hand)
                    rows.append(row)
                x = pd.DataFrame(rows)
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
                    lateral0 = _median_pre(x, "thumb_peg_signed_distance_px", insertion_start, pre_window)
                    if not np.isfinite(lateral0):
                        warnings.warn(f"No valid pre-insertion lateral baseline for {demo}; lateral-slip output will be missing")
                    x["lateral_slip_valid"] = x.peg_valid & x.hand_valid & x["thumb_peg_signed_distance_px"].notna() & np.isfinite(lateral0)
                    x["lateral_slip_px"] = x.thumb_peg_signed_distance_px - lateral0
                    x["lateral_slip_mm"] = x.lateral_slip_px * mm_per_px
                x["mm_per_px"] = mm_per_px
                x["scale_basis"] = scale_basis
                x["visible_length_baseline_px"] = baseline_length
                camera_frames.append(x)
            combined = pd.concat(camera_frames, ignore_index=True)
            combined.to_csv(output_dir / f"{demo}_timeseries.csv", index=False)
            main = combined[combined.role == "main"].copy()
            secondary = combined[combined.role == "secondary"].copy()
            main_insert = main[main.phase == "insertion"]
            secondary_insert = secondary[secondary.phase == "insertion"]
            depth_valid = main_insert[main_insert.insertion_depth_valid & main_insert.insertion_depth_mm.notna()]
            if depth_valid.empty:
                raise RuntimeError(f"No valid main-camera insertion-phase depth metrics: {demo}")
            max_depth_idx = depth_valid.insertion_depth_mm.idxmax()
            initial = main.loc[main.frame_index == int(main.insertion_start_frame.iloc[0])].iloc[0]
            final = main.loc[max_depth_idx]
            axial = main_insert.loc[main_insert.axial_slip_valid, "axial_slip_mm"].dropna()
            lateral = secondary_insert.loc[secondary_insert.lateral_slip_valid, "lateral_slip_mm"].dropna()
            summary = dict(
                demo=demo,
                max_insertion_depth_mm=float(final.insertion_depth_mm),
                max_depth_frame=int(final.frame_index),
                max_axial_slip_mm=float(axial.max()) if len(axial) else np.nan,
                min_axial_slip_mm=float(axial.min()) if len(axial) else np.nan,
                max_abs_axial_slip_mm=float(axial.abs().max()) if len(axial) else np.nan,
                max_abs_lateral_slip_mm=float(lateral.abs().max()) if len(lateral) else np.nan,
                initial_angle_deg=float(initial.peg_angle_deg),
                initial_angular_error_deg=float(initial.peg_angular_error_deg),
                final_angle_deg=float(final.peg_angle_deg),
                final_angular_error_deg=float(final.peg_angular_error_deg),
                angular_error_change_deg=float(final.peg_angular_error_deg-initial.peg_angular_error_deg),
                main_valid_frame_fraction=float(main.peg_valid.mean()),
                secondary_valid_frame_fraction=float((secondary.peg_valid & secondary.hand_valid).mean()),
            )
            pre_main = main[main.phase == "pre"]
            pre_secondary = secondary[secondary.phase == "pre"]
            summary.update(_noise_fields("insertion_depth_mm", pre_main.insertion_depth_mm))
            summary.update(_noise_fields("axial_slip_mm", pre_main.axial_slip_mm))
            summary.update(_noise_fields("lateral_slip_mm", pre_secondary.lateral_slip_mm))
            summary.update(_noise_fields("angular_error_deg", pre_main.peg_angular_error_deg))
            summary.update(_noise_fields("peg_angle_deg", pre_main.peg_angle_deg))
            summary.update(_noise_fields("visible_length_px", pre_main.peg_visible_length_px))
            summary.update(_noise_fields("visible_width_px", pre_secondary.peg_visible_width_px))
            summaries.append(summary)
    summary_path = output_dir / (f"summary_{requested_demo}.csv" if requested_demo else "summary.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved metrics: {summary_path}")

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
            _analyze_impl(c, demo)
            demo_summary_path = output_dir / f"summary_{demo}.csv"
            summaries.append(pd.read_csv(demo_summary_path))
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

    summary_path = output_dir / (
        f"summary_{requested_demo}.csv" if requested_demo else "summary.csv"
    )
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

