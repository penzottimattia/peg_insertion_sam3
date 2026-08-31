from pathlib import Path
import numpy as np


def _mean_ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float(values.mean()) if len(values) else np.nan, 0.0
    from scipy.stats import t
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, half


def _distribution(ax, values, label, ylabel, trial_labels, colors):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    clean = values[valid]
    mean, ci = _mean_ci95(clean)
    ax.bar(
        [0], [mean], width=0.45, color="0.85", edgecolor="0.35",
        yerr=[ci], capsize=6, zorder=1,
    )
    if len(clean):
        valid_indices = np.flatnonzero(valid)
        offsets = (
            np.linspace(-0.13, 0.13, len(valid_indices))
            if len(valid_indices) > 1 else np.array([0.0])
        )
        value_span = float(np.ptp(clean))
        label_dy = 0.018 * value_span if value_span > 0 else 0.02 * max(abs(mean), 1.0)
        for offset, index in zip(offsets, valid_indices):
            color = colors[index]
            value = values[index]
            ax.scatter([offset], [value], color=color, edgecolor="white", linewidth=0.5,
                       s=42, zorder=3)
            # Alternate vertical displacement to reduce collisions while keeping
            # labels close to their corresponding points.
            direction = 1 if index % 2 == 0 else -1
            ax.annotate(
                trial_labels[index],
                xy=(offset, value),
                xytext=(4, direction * 5),
                textcoords="offset points",
                color=color,
                alpha=0.68,
                fontsize=7,
                ha="left",
                va="bottom" if direction > 0 else "top",
                clip_on=False,
                zorder=4,
            )
    ax.set_xlim(-0.28, 0.32)
    ax.set_xticks([0], [label])
    ax.set_ylabel(ylabel)


def _lateral_summary_column(data):
    """Prefer the current lateral metric while accepting legacy summaries."""
    current = "max_abs_lateral_slip_mm"
    legacy = "max_lateral_slip_away_mm"
    if current in data.columns:
        return current, "Maximum absolute lateral slip (mm)"
    if legacy in data.columns:
        return legacy, "Maximum slip away (mm) [legacy]"
    raise ValueError(
        "Summary CSV has neither max_abs_lateral_slip_mm nor "
        "max_lateral_slip_away_mm. Re-run analyze or inspect the CSV schema."
    )


def make_plots(c, summary_path=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    output_dir = Path(c["output_dir"])
    summary_path = Path(summary_path) if summary_path else output_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    data = pd.read_csv(summary_path)
    lateral_column, lateral_ylabel = _lateral_summary_column(data)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Stable short trial labels and colors are shared across every panel.
    trial_labels = [f"T{i+1:02d}" for i in range(len(data))]
    cmap = plt.colormaps.get_cmap("turbo")
    positions = np.linspace(0.05, 0.95, max(len(data), 1))
    colors = [cmap(position) for position in positions[:len(data)]]
    trial_key = data[["demo"]].copy()
    trial_key.insert(0, "trial_label", trial_labels)
    trial_key.to_csv(plots_dir / "trial_key.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    _distribution(
        axes[0], data.max_insertion_depth_mm, "Depth",
        "Maximum insertion depth (mm)", trial_labels, colors,
    )
    _distribution(
        axes[1], data.max_abs_axial_slip_mm, "Axial",
        "Maximum absolute slip (mm)", trial_labels, colors,
    )
    _distribution(
        axes[2], data[lateral_column], "Lateral",
        lateral_ylabel, trial_labels, colors,
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "outcome_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in data.iterrows():
        ax.plot([0, 1], [row.initial_angular_error_deg, row.final_angular_error_deg], color="0.7", linewidth=1)
        ax.scatter([0, 1], [row.initial_angular_error_deg, row.final_angular_error_deg], color="black", s=24, zorder=3)
    for x, col in enumerate(["initial_angular_error_deg", "final_angular_error_deg"]):
        mean, ci = _mean_ci95(data[col])
        ax.errorbar(x, mean, yerr=ci, fmt="o", color="tab:red", capsize=6, markersize=7, zorder=4)
    ax.set_xticks([0, 1], ["Initial", "At maximum depth"])
    ax.set_ylabel("Angular error (deg)")
    fig.tight_layout()
    fig.savefig(plots_dir / "angular_error_paired.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data.initial_angular_error_deg, data.max_insertion_depth_mm, color="black")
    ax.set_xlabel("Initial angular error (deg)")
    ax.set_ylabel("Maximum insertion depth (mm)")
    fig.tight_layout()
    fig.savefig(plots_dir / "initial_angle_vs_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots: {plots_dir}")


def merge_plots(output_dirs, destination="merged_plots"):
    """Merge summary.csv files from multiple analysis output directories and plot them."""
    import pandas as pd

    output_dirs = [Path(path).expanduser().resolve() for path in output_dirs]
    if len(output_dirs) < 2:
        raise ValueError("merge-plot requires at least two output directories")

    frames = []
    for output_dir in output_dirs:
        summary_path = output_dir / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
        frame = pd.read_csv(summary_path)
        frame.insert(0, "source_output_dir", str(output_dir))
        frame.insert(0, "source_label", output_dir.name)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True, sort=False)
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    merged_summary = destination / "merged_summary.csv"
    data.to_csv(merged_summary, index=False)

    plot_config = {"output_dir": str(destination)}
    make_plots(plot_config, merged_summary)

    # Keep merge-plot artifacts directly in the requested destination.
    generated_plots = destination / "plots"
    for generated in generated_plots.iterdir():
        generated.replace(destination / generated.name)
    generated_plots.rmdir()

    # Replace the basic demo-only key with a source-aware key.
    trial_labels = [f"T{i+1:02d}" for i in range(len(data))]
    key = data[["source_label", "source_output_dir", "demo"]].copy()
    key.insert(0, "trial_label", trial_labels)
    key.to_csv(destination / "trial_key.csv", index=False)

    print(f"Merged {len(data)} trials from {len(output_dirs)} output directories")
    print(f"Saved merged summary: {merged_summary}")
    print(f"Saved merged plots: {destination}")
