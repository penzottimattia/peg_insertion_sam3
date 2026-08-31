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


def _distribution(ax, values, label, ylabel):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean, ci = _mean_ci95(values)
    ax.bar([0], [mean], width=0.45, color="0.75", edgecolor="0.2", yerr=[ci], capsize=6)
    if len(values):
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(offsets, values, color="black", s=28, zorder=3)
    ax.set_xticks([0], [label])
    ax.set_ylabel(ylabel)


def make_plots(c, summary_path=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    output_dir = Path(c["output_dir"])
    summary_path = Path(summary_path) if summary_path else output_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    data = pd.read_csv(summary_path)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    _distribution(axes[0], data.max_insertion_depth_mm, "Depth", "Maximum insertion depth (mm)")
    _distribution(axes[1], data.max_abs_axial_slip_mm, "Axial", "Maximum absolute slip (mm)")
    _distribution(axes[2], data.max_lateral_slip_away_mm, "Lateral", "Maximum slip away (mm)")
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
