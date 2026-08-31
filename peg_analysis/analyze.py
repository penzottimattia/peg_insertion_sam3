from pathlib import Path
import json
import numpy as np
from .core import (axial_above_thumb_length, demos, detect_gap, feat, group,
                   length_scale, peg_angle_deg, signed_point_axis_distance)


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


def analyze(c, requested_demo=None):
    import h5py
    import pandas as pd
    output_dir = Path(c["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
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
                # Backward compatibility: new metadata stores the insertion start.
                # For masks created by the original combined pipeline, recover it
                # from the HDF5 timestamps without rerunning SAM.
                if "insertion_start_frame" in meta:
                    t0 = int(meta["insertion_start_frame"])
                else:
                    _, t0, _ = detect_gap(
                        timestamps,
                        c["onset"]["gap_mad_multiplier"],
                        c["onset"]["gap_nominal_multiplier"],
                    )
                peg_features = [feat(m) for m in masks["peg"]]
                hand_features = [feat(m) for m in masks["hand"]]
                if peg_features[0] is None:
                    raise RuntimeError(f"Invalid peg mask at scale frame 0: {demo}/{role}")
                mm_per_px = length_scale(peg_features[0], c["dimensions"].get("peg_length_mm"))
                rows = []
                for i, timestamp in enumerate(timestamps):
                    peg, hand = peg_features[i], hand_features[i]
                    row = dict(role=role, frame_index=i, host_timestamp_ns=int(timestamp),
                               elapsed_time_from_t0_s=(int(timestamp)-int(timestamps[t0]))/1e9,
                               phase="pre" if i < t0 else "insertion", insertion_start_frame=t0,
                               peg_valid=peg is not None, hand_valid=hand is not None)
                    if peg is not None:
                        angle = peg_angle_deg(peg["axis"])
                        row.update(peg_center_x_px=peg["c"][0], peg_center_y_px=peg["c"][1],
                                   peg_axis_x=peg["axis"][0], peg_axis_y=peg["axis"][1],
                                   peg_visible_length_px=peg["visible_length"],
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
                pre_lengths = x.loc[x.peg_valid & (x.frame_index < t0), "peg_visible_length_px"].tail(int(c["onset"]["pre_window_frames"]))
                if pre_lengths.empty:
                    raise RuntimeError(f"No valid pre-insertion peg lengths: {demo}/{role}")
                baseline_length = float(pre_lengths.median())
                if role == "main":
                    x["insertion_depth_valid"] = x.peg_valid
                    x["axial_slip_valid"] = x.peg_valid & x.hand_valid & x["peg_above_thumb_length_px"].notna()
                    x["insertion_depth_px"] = baseline_length - x["peg_visible_length_px"]
                    x["insertion_depth_mm"] = x.insertion_depth_px * mm_per_px
                    axial0 = x.loc[x.frame_index == t0, "peg_above_thumb_length_px"].iloc[0]
                    x["axial_slip_px"] = x.peg_above_thumb_length_px - axial0
                    x["axial_slip_mm"] = x.axial_slip_px * mm_per_px
                else:
                    x["lateral_slip_valid"] = x.peg_valid & x.hand_valid & x["thumb_peg_signed_distance_px"].notna()
                    lateral0 = x.loc[x.frame_index == t0, "thumb_peg_signed_distance_px"].iloc[0]
                    x["lateral_slip_signed_px"] = x.thumb_peg_signed_distance_px - lateral0
                    x["lateral_slip_away_px"] = x.thumb_peg_signed_distance_px.abs() - abs(lateral0)
                    x["lateral_slip_signed_mm"] = x.lateral_slip_signed_px * mm_per_px
                    x["lateral_slip_away_mm"] = x.lateral_slip_away_px * mm_per_px
                x["mm_per_px"] = mm_per_px
                x["visible_length_baseline_px"] = baseline_length
                camera_frames.append(x)
            combined = pd.concat(camera_frames, ignore_index=True)
            combined.to_csv(output_dir / f"{demo}_timeseries.csv", index=False)
            main = combined[combined.role == "main"].copy()
            secondary = combined[combined.role == "secondary"].copy()
            depth_valid = main[main.insertion_depth_valid & main.insertion_depth_mm.notna()]
            if depth_valid.empty:
                raise RuntimeError(f"No valid main-camera depth metrics: {demo}")
            max_depth_idx = depth_valid.insertion_depth_mm.idxmax()
            initial = main.loc[main.frame_index == int(main.insertion_start_frame.iloc[0])].iloc[0]
            final = main.loc[max_depth_idx]
            axial = main.loc[main.axial_slip_valid, "axial_slip_mm"].dropna()
            lateral = secondary.loc[secondary.lateral_slip_valid, "lateral_slip_away_mm"].dropna()
            summaries.append(dict(
                demo=demo,
                max_insertion_depth_mm=float(final.insertion_depth_mm),
                max_depth_frame=int(final.frame_index),
                max_axial_slip_mm=float(axial.max()) if len(axial) else np.nan,
                min_axial_slip_mm=float(axial.min()) if len(axial) else np.nan,
                max_abs_axial_slip_mm=float(axial.abs().max()) if len(axial) else np.nan,
                max_lateral_slip_away_mm=float(lateral.max()) if len(lateral) else np.nan,
                initial_angle_deg=float(initial.peg_angle_deg),
                initial_angular_error_deg=float(initial.peg_angular_error_deg),
                final_angle_deg=float(final.peg_angle_deg),
                final_angular_error_deg=float(final.peg_angular_error_deg),
                angular_error_change_deg=float(final.peg_angular_error_deg-initial.peg_angular_error_deg),
                main_valid_frame_fraction=float(main.peg_valid.mean()),
                secondary_valid_frame_fraction=float((secondary.peg_valid & secondary.hand_valid).mean()),
            ))
    summary_path = output_dir / (f"summary_{requested_demo}.csv" if requested_demo else "summary.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved metrics: {summary_path}")
