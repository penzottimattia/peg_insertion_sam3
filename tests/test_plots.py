import numpy as np
from peg_analysis.plots import _mean_ci95


def test_ci_single_value():
    mean, ci = _mean_ci95([3.0])
    assert mean == 3.0 and ci == 0.0


def test_ci_ignores_nan():
    mean, ci = _mean_ci95([1.0, np.nan, 3.0])
    assert mean == 2.0 and ci > 0


def test_merge_plots_combines_summaries(tmp_path):
    import pandas as pd
    from peg_analysis.plots import merge_plots

    required = {
        "demo": ["demo_000001"],
        "max_insertion_depth_mm": [10.0],
        "max_abs_axial_slip_mm": [1.0],
        "max_abs_lateral_slip_mm": [2.0],
        "initial_angular_error_deg": [3.0],
        "final_angular_error_deg": [1.5],
    }
    inputs = []
    for name in ("run_a", "run_b"):
        output_dir = tmp_path / name
        output_dir.mkdir()
        pd.DataFrame(required).to_csv(output_dir / "summary.csv", index=False)
        inputs.append(output_dir)

    destination = tmp_path / "merged"
    merge_plots(inputs, destination)
    merged = pd.read_csv(destination / "merged_summary.csv")
    assert len(merged) == 2
    assert set(merged.source_label) == {"run_a", "run_b"}
    assert (destination / "outcome_distributions.png").is_file()
    assert (destination / "trial_key.csv").is_file()


def test_merge_plots_nmax_trials_keeps_highest_depths(tmp_path):
    import pandas as pd
    from peg_analysis.plots import merge_plots

    inputs = []
    for name, depths in (("run_a", [1.0, 9.0]), ("run_b", [5.0, 3.0])):
        output_dir = tmp_path / name
        output_dir.mkdir()
        pd.DataFrame({
            "demo": [f"demo_{i:06d}" for i in range(len(depths))],
            "max_insertion_depth_mm": depths,
            "max_abs_axial_slip_mm": [1.0] * len(depths),
            "max_abs_lateral_slip_mm": [2.0] * len(depths),
            "initial_angular_error_deg": [3.0] * len(depths),
            "final_angular_error_deg": [1.5] * len(depths),
        }).to_csv(output_dir / "summary.csv", index=False)
        inputs.append(output_dir)

    destination = tmp_path / "merged"
    merge_plots(inputs, destination, nmax_trials=2)
    merged = pd.read_csv(destination / "merged_summary.csv")
    plotted = pd.read_csv(destination / "plotted_summary.csv")
    assert len(merged) == 4
    assert sorted(plotted.max_insertion_depth_mm) == [5.0, 9.0]


def test_outcome_panels_use_same_y_scale(tmp_path, monkeypatch):
    import matplotlib.axes
    import pandas as pd
    from peg_analysis.plots import make_plots

    output_dir = tmp_path / "results"
    output_dir.mkdir()
    summary = output_dir / "summary.csv"
    pd.DataFrame({
        "demo": ["demo_000000", "demo_000001"],
        "max_insertion_depth_mm": [30.0, 20.0],
        "max_abs_axial_slip_mm": [3.0, 2.0],
        "max_abs_lateral_slip_mm": [1.0, 0.5],
        "initial_angular_error_deg": [2.0, 1.0],
        "final_angular_error_deg": [1.0, 0.5],
    }).to_csv(summary, index=False)

    saved_limits = []
    original_set_ylim = matplotlib.axes.Axes.set_ylim

    def recording_set_ylim(self, *args, **kwargs):
        result = original_set_ylim(self, *args, **kwargs)
        if len(args) >= 2:
            saved_limits.append((float(args[0]), float(args[1])))
        return result

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", recording_set_ylim)
    make_plots({"output_dir": str(output_dir)}, summary)
    assert len(saved_limits) >= 3
    assert saved_limits[-3] == saved_limits[-2] == saved_limits[-1]


def test_distribution_uses_x_below_insertion_depth_threshold():
    import matplotlib.pyplot as plt
    from peg_analysis.plots import _distribution

    fig, ax = plt.subplots()
    _distribution(
        ax, [5.0, 15.0], "Depth", "Depth (mm)", ["T01", "T02"],
        ["red", "blue"], [True, False],
    )
    trial_markers = ax.collections[-2:]
    x_path = trial_markers[0].get_paths()[0]
    circle_path = trial_markers[1].get_paths()[0]
    assert len(x_path.vertices) < len(circle_path.vertices)
    assert trial_markers[0].get_linewidths()[0] == 2.0
    plt.close(fig)


def test_make_plots_accepts_insertion_depth_threshold(tmp_path):
    import pandas as pd
    from peg_analysis.plots import make_plots

    output_dir = tmp_path / "results"
    output_dir.mkdir()
    summary = output_dir / "summary.csv"
    pd.DataFrame({
        "demo": ["demo_000000", "demo_000001"],
        "max_insertion_depth_mm": [10.0, 30.0],
        "max_abs_axial_slip_mm": [2.0, 3.0],
        "max_abs_lateral_slip_mm": [1.0, 1.5],
        "initial_angular_error_deg": [2.0, 1.0],
        "final_angular_error_deg": [1.0, 0.5],
    }).to_csv(summary, index=False)

    make_plots(
        {"output_dir": str(output_dir)}, summary,
        insertion_depth_threshold=20.0,
    )
    assert (output_dir / "plots" / "outcome_distributions.png").is_file()
