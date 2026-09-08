"""Configuration-driven method comparison plots across tolerance levels."""
from pathlib import Path
import json
import numpy as np

from .plots import _mean_ci95

_METRICS = (
    ("max_insertion_depth_mm", "Depth", "Maximum insertion depth (mm)"),
    ("max_abs_axial_slip_mm", "Axial", "Maximum absolute axial slip (mm)"),
    ("max_abs_lateral_slip_mm", "Lateral", "Maximum absolute lateral slip (mm)"),
)


def _resolve_summary(path, base_dir):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    return path / "summary.csv" if path.is_dir() else path


def load_cumulative_config(config_path):
    """Validate and normalize a cumulative-plot JSON configuration."""
    config_path = Path(config_path).expanduser().resolve()
    raw = json.loads(config_path.read_text())
    methods = raw.get("methods")
    datasets = raw.get("datasets")
    if not isinstance(methods, list) or not methods:
        raise ValueError("cumulative config requires a non-empty 'methods' list")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("cumulative config requires a non-empty 'datasets' list")

    normalized_methods = {}
    for method in methods:
        name = str(method.get("name", "")).strip()
        if not name:
            raise ValueError("every method requires a non-empty name")
        if name in normalized_methods:
            raise ValueError(f"duplicate method name: {name}")
        normalized_methods[name] = {
            "name": name,
            "label": str(method.get("label", name)),
            "color": method.get("color"),
        }

    normalized_datasets = []
    for index, dataset in enumerate(datasets):
        method = str(dataset.get("method", "")).strip()
        if method not in normalized_methods:
            raise ValueError(f"dataset {index} references unknown method: {method!r}")
        if "tolerance" not in dataset:
            raise ValueError(f"dataset {index} requires tolerance")
        data_dirs = dataset.get("data_dirs")
        if not isinstance(data_dirs, list) or not data_dirs:
            raise ValueError(f"dataset {index} requires a non-empty data_dirs list")
        normalized_datasets.append({
            "method": method,
            "tolerance": dataset["tolerance"],
            "summaries": [_resolve_summary(path, config_path.parent) for path in data_dirs],
        })

    normalization_type = raw.get("normalization_type")
    if normalization_type is None:
        # Backward compatibility with the earlier boolean option.
        normalization_type = "max" if raw.get("normalized_angle", False) else "none"
    normalization_type = str(normalization_type).strip().lower()
    if normalization_type not in {"none", "max", "minmax", "zscore"}:
        raise ValueError(
            "normalization_type must be one of: none, max, minmax, zscore"
        )

    return {
        "methods": normalized_methods,
        "datasets": normalized_datasets,
        "tolerance_label": str(raw.get("tolerance_label", "Tolerance")),
        "title": raw.get("title"),
        "insertion_depth_threshold": (
            None if raw.get("insertion_depth_threshold") is None
            else float(raw["insertion_depth_threshold"])
        ),
        "nmax_trials": (
            None if raw.get("nmax_trials") is None
            else int(raw["nmax_trials"])
        ),
        "normalization_type": normalization_type,
    }


def _load_trials(config):
    import pandas as pd

    frames = []
    for group in config["datasets"]:
        for part_index, summary_path in enumerate(group["summaries"], start=1):
            if not summary_path.is_file():
                raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
            frame = pd.read_csv(summary_path)
            required = {name for name, _, _ in _METRICS} | {"demo"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Summary CSV {summary_path} is missing: {', '.join(sorted(missing))}"
                )
            frame = frame.copy()
            frame.insert(0, "source_summary", str(summary_path))
            frame.insert(0, "part", part_index)
            frame.insert(0, "tolerance", group["tolerance"])
            frame.insert(0, "method", group["method"])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _method_distribution(
    ax, values, x, color, trial_labels, below_insertion_depth_threshold=None
):
    """Draw one method mean, 95% CI, trial markers, and trial labels."""
    values = np.asarray(values, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    clean = values[valid_indices]
    if below_insertion_depth_threshold is None:
        below_insertion_depth_threshold = np.zeros(len(values), dtype=bool)
    else:
        below_insertion_depth_threshold = np.asarray(
            below_insertion_depth_threshold, dtype=bool
        )
        if len(below_insertion_depth_threshold) != len(values):
            raise ValueError(
                "below_insertion_depth_threshold must match values length"
            )
    mean, ci = _mean_ci95(clean)
    if np.isfinite(mean):
        ax.bar(
            [x], [mean], width=0.62, color=color, alpha=0.24,
            edgecolor=color, linewidth=1.4, yerr=[ci], capsize=5, zorder=1,
        )
    if not len(clean):
        return

    offsets = (
        np.linspace(-0.20, 0.20, len(valid_indices))
        if len(valid_indices) > 1 else np.array([0.0])
    )
    for order, (offset, index) in enumerate(zip(offsets, valid_indices)):
        value = values[index]
        if below_insertion_depth_threshold[index]:
            ax.scatter(
                [x + offset], [value], marker="x", color=color,
                linewidths=2.0, s=72, zorder=5,
            )
        else:
            ax.scatter(
                [x + offset], [value], marker="o", color=color,
                edgecolor="white", linewidth=0.55, s=43, zorder=3,
            )
        direction = 1 if order % 2 == 0 else -1
        ax.annotate(
            trial_labels[index], xy=(x + offset, value),
            xytext=(3, direction * 5), textcoords="offset points",
            color=color, alpha=0.78, fontsize=7,
            ha="left", va="bottom" if direction > 0 else "top",
            clip_on=False, zorder=4,
        )


def cumulative_plots(
    config_path, output_dir=None, nmax_trials=None, normalization_type=None
):
    """Create method comparison distributions with one row per tolerance.

    The combined trial CSV always contains every configured trial. When
    nmax_trials is set, each method/tolerance group independently keeps its N
    trials with the greatest insertion depth for the figure and plotted CSV.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    config_path = Path(config_path).expanduser().resolve()
    config = load_cumulative_config(config_path)
    data = _load_trials(config)
    if data.empty:
        raise ValueError("No trials were found in the configured summaries")

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir else config_path.parent / "cumulative_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = data.copy()
    data.insert(0, "trial_label", [f"T{i + 1:02d}" for i in range(len(data))])
    data.to_csv(output_dir / "cumulative_trials.csv", index=False)

    selected_nmax = config["nmax_trials"] if nmax_trials is None else nmax_trials
    if selected_nmax is not None and selected_nmax < 1:
        raise ValueError("nmax_trials must be at least 1")
    if selected_nmax is None:
        plotted_data = data.copy()
    else:
        selected_indices = []
        for _, group in data.groupby(["tolerance", "method"], sort=False):
            selected_indices.extend(
                group["max_insertion_depth_mm"].nlargest(selected_nmax).index.tolist()
            )
        plotted_data = data.loc[data.index.isin(selected_indices)].copy()
    plotted_data.to_csv(output_dir / "plotted_trials.csv", index=False)

    tolerances = [
        tolerance for tolerance in dict.fromkeys(
            item["tolerance"] for item in config["datasets"]
        )
        if (plotted_data["tolerance"].astype(str) == str(tolerance)).any()
    ]
    methods = config["methods"]
    method_names = list(methods)
    fallback = plt.colormaps.get_cmap("tab10")
    colors = {
        name: method["color"] or fallback(index % 10)
        for index, (name, method) in enumerate(methods.items())
    }

    figure, axes = plt.subplots(
        len(tolerances), len(_METRICS),
        figsize=(5.0 * len(_METRICS), 3.7 * len(tolerances)),
        squeeze=False,
    )
    for row, tolerance in enumerate(tolerances):
        tolerance_data = plotted_data[
            plotted_data["tolerance"].astype(str) == str(tolerance)
        ]
        for column, (metric, short_label, ylabel) in enumerate(_METRICS):
            ax = axes[row, column]
            present_methods = [
                name for name in method_names
                if (tolerance_data.method == name).any()
            ]
            for method_index, method_name in enumerate(present_methods):
                method_data = tolerance_data[tolerance_data.method == method_name]
                threshold = config["insertion_depth_threshold"]
                below_threshold = (
                    np.zeros(len(method_data), dtype=bool)
                    if threshold is None else
                    method_data["max_insertion_depth_mm"].to_numpy(dtype=float)
                    < threshold
                )
                _method_distribution(
                    ax,
                    method_data[metric].to_numpy(dtype=float),
                    method_index,
                    colors[method_name],
                    method_data.trial_label.tolist(),
                    below_threshold,
                )
            ax.set_xticks(
                range(len(present_methods)),
                [methods[name]["label"] for name in present_methods],
            )
            ax.set_xlim(-0.55, max(len(present_methods) - 0.45, 0.55))
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.22)
            if column == 0 and config["insertion_depth_threshold"] is not None:
                ax.axhline(
                    config["insertion_depth_threshold"], color="0.35",
                    linestyle="--", linewidth=1.1, alpha=0.75, zorder=0,
                )
            if row == 0:
                ax.set_title(short_label)
            if column == 0:
                ax.text(
                    -0.29, 0.5,
                    f"{config['tolerance_label']}: {tolerance}",
                    transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=11, fontweight="bold",
                )

    shared_ymin = min(ax.get_ylim()[0] for ax in axes.flat)
    shared_ymax = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(shared_ymin, shared_ymax)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color=colors[name], linewidth=5,
            alpha=0.75, markeredgecolor="white", label=methods[name]["label"],
        )
        for name in method_names
    ]
    figure.legend(
        handles=legend_handles, loc="upper center", ncol=len(method_names),
        bbox_to_anchor=(0.5, 1.01), frameon=False,
    )
    if config["title"]:
        figure.suptitle(str(config["title"]), y=1.055)
    figure.tight_layout(rect=(0.03, 0, 1, 0.96))
    output_path = output_dir / "cumulative_outcomes.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    selected_normalization = (
        config["normalization_type"]
        if normalization_type is None else str(normalization_type).strip().lower()
    )
    if selected_normalization not in {"none", "max", "minmax", "zscore"}:
        raise ValueError("normalization_type must be one of: none, max, minmax, zscore")

    plotted_data["angle_normalization_type"] = selected_normalization
    plotted_data["initial_angular_error_group_mean_deg"] = 0.0
    plotted_data["initial_angular_error_group_min_deg"] = 0.0
    plotted_data["initial_angular_error_group_max_deg"] = 0.0
    plotted_data["initial_angular_error_group_scale_deg"] = 1.0
    plotted_data["normalized_initial_angular_error"] = plotted_data[
        "initial_angular_error_deg"
    ].astype(float)
    for _, group in plotted_data.groupby(["tolerance", "method"], sort=False):
        values = group["initial_angular_error_deg"].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        mean = float(np.mean(valid)) if len(valid) else 0.0
        group_min = float(np.min(valid)) if len(valid) else 0.0
        group_max = float(np.max(valid)) if len(valid) else 0.0
        if selected_normalization == "max":
            scale = float(np.max(np.abs(valid))) if len(valid) else 1.0
            center = 0.0
        elif selected_normalization == "minmax":
            scale = group_max - group_min
            center = group_min
        elif selected_normalization == "zscore":
            scale = float(np.std(valid, ddof=0)) if len(valid) else 1.0
            center = mean
        else:
            scale = 1.0
            center = 0.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        plotted_data.loc[group.index, "initial_angular_error_group_mean_deg"] = mean
        plotted_data.loc[group.index, "initial_angular_error_group_min_deg"] = group_min
        plotted_data.loc[group.index, "initial_angular_error_group_max_deg"] = group_max
        plotted_data.loc[group.index, "initial_angular_error_group_scale_deg"] = scale
        if selected_normalization != "none":
            plotted_data.loc[group.index, "normalized_initial_angular_error"] = (
                group["initial_angular_error_deg"].astype(float) - center
            ) / scale
    plotted_data.to_csv(output_dir / "plotted_trials.csv", index=False)

    scatter_figure, scatter_axes = plt.subplots(
        1, len(tolerances), figsize=(5.2 * len(tolerances), 4.25), squeeze=False,
        sharex=True, sharey=True,
    )
    threshold = config["insertion_depth_threshold"]
    for column, tolerance in enumerate(tolerances):
        ax = scatter_axes[0, column]
        tolerance_data = plotted_data[
            plotted_data["tolerance"].astype(str) == str(tolerance)
        ]
        for method_name in method_names:
            method_data = tolerance_data[tolerance_data.method == method_name]
            for order, (_, trial) in enumerate(method_data.iterrows()):
                failed = (
                    threshold is not None
                    and float(trial.max_insertion_depth_mm) < threshold
                )
                marker = "x" if failed else "o"
                scatter_kwargs = {
                    "marker": marker,
                    "color": colors[method_name],
                    "s": 72 if failed else 45,
                    "zorder": 5 if failed else 3,
                }
                if failed:
                    scatter_kwargs["linewidths"] = 2.0
                else:
                    scatter_kwargs.update(edgecolor="white", linewidth=0.55)
                angle_value = (
                    float(trial.normalized_initial_angular_error)
                    if selected_normalization != "none"
                    else float(trial.initial_angular_error_deg)
                )
                ax.scatter(
                    [angle_value], [trial.max_insertion_depth_mm],
                    **scatter_kwargs,
                )
                direction = 1 if order % 2 == 0 else -1
                ax.annotate(
                    trial.trial_label,
                    xy=(angle_value, trial.max_insertion_depth_mm),
                    xytext=(4, direction * 5), textcoords="offset points",
                    color=colors[method_name], alpha=0.78, fontsize=7,
                    ha="left", va="bottom" if direction > 0 else "top",
                    clip_on=False, zorder=4,
                )
        if threshold is not None:
            ax.axhline(
                threshold, color="0.35", linestyle="--", linewidth=1.1,
                alpha=0.75, zorder=0,
            )
        ax.set_title(f"{config['tolerance_label']}: {tolerance}")
        angle_labels = {
            "none": "Initial angular error (deg)",
            "max": "Max-normalized initial angular error",
            "minmax": "Min-max normalized initial angular error",
            "zscore": "Initial angular error z-score",
        }
        ax.set_xlabel(angle_labels[selected_normalization])
        ax.grid(alpha=0.22)
    scatter_axes[0, 0].set_ylabel("Maximum insertion depth (mm)")
    scatter_figure.legend(
        handles=legend_handles, loc="upper center", ncol=len(method_names),
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    scatter_figure.tight_layout(rect=(0, 0, 1, 0.94))
    scatter_path = output_dir / "initial_angle_vs_depth_by_tolerance.png"
    scatter_figure.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(scatter_figure)

    print(f"Loaded {len(data)} trials")
    print(f"Plotted {len(plotted_data)} trials across {len(tolerances)} tolerance level(s)")
    print(f"Saved cumulative plot: {output_path}")
    print(f"Saved angle-depth scatter plot: {scatter_path}")
    print(f"Saved all trials: {output_dir / 'cumulative_trials.csv'}")
    print(f"Saved plotted trials: {output_dir / 'plotted_trials.csv'}")
    return output_path
