import json


def _summary(path, depths):
    import pandas as pd
    path.mkdir(parents=True)
    pd.DataFrame({
        "demo": [f"demo_{i:06d}" for i in range(len(depths))],
        "max_insertion_depth_mm": depths,
        "max_abs_axial_slip_mm": [1.0 + i for i in range(len(depths))],
        "max_abs_lateral_slip_mm": [0.5 + i for i in range(len(depths))],
        "initial_angular_error_deg": [0.75 + i for i in range(len(depths))],
    }).to_csv(path / "summary.csv", index=False)


def test_cumulative_plot_combines_split_dirs_without_session(tmp_path):
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    first = tmp_path / "first"
    second = tmp_path / "second"
    other = tmp_path / "other"
    _summary(first, [10.0, 20.0])
    _summary(second, [30.0])
    _summary(other, [15.0, 25.0])
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "methods": [
            {"name": "a", "color": "#0072B2"},
            {"name": "b", "color": "#D55E00"},
        ],
        "datasets": [
            {"method": "a", "tolerance": 0.5,
             "data_dirs": ["first", "second"]},
            {"method": "b", "tolerance": 0.5,
             "data_dirs": ["other"]},
        ],
    }))

    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    combined = pd.read_csv(destination / "cumulative_trials.csv")
    assert len(combined) == 5
    assert len(combined[combined.method == "a"]) == 3
    assert "session" not in combined.columns
    assert combined.trial_label.tolist() == ["T01", "T02", "T03", "T04", "T05"]
    assert (destination / "cumulative_outcomes.png").is_file()
    assert (destination / "initial_angle_vs_outcomes_by_tolerance.png").is_file()


def test_cumulative_config_rejects_unknown_method(tmp_path):
    import pytest
    from peg_analysis.cumulative import load_cumulative_config

    spec = tmp_path / "bad.json"
    spec.write_text(json.dumps({
        "methods": [{"name": "a"}],
        "datasets": [{
            "method": "missing", "tolerance": 1, "data_dirs": ["run"],
        }],
    }))
    with pytest.raises(ValueError, match="unknown method"):
        load_cumulative_config(spec)


def test_distribution_uses_x_for_failed_depth():
    import matplotlib.pyplot as plt
    from peg_analysis.cumulative import _method_distribution

    fig, ax = plt.subplots()
    _method_distribution(
        ax, [10.0, 30.0], 0, "blue", ["T01", "T02"], [True, False]
    )
    x_path = ax.collections[-2].get_paths()[0]
    circle_path = ax.collections[-1].get_paths()[0]
    assert len(x_path.vertices) < len(circle_path.vertices)
    plt.close(fig)


def test_cumulative_nmax_is_applied_per_tolerance_and_method(tmp_path):
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    a_tol_low = tmp_path / "a_tol_low"
    b_tol_low = tmp_path / "b_tol_low"
    a_tol_high = tmp_path / "a_tol_high"
    _summary(a_tol_low, [1.0, 9.0, 4.0])
    _summary(b_tol_low, [2.0, 8.0, 5.0])
    _summary(a_tol_high, [3.0, 7.0, 6.0])
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "nmax_trials": 2,
        "methods": [{"name": "a"}, {"name": "b"}],
        "datasets": [
            {"method": "a", "tolerance": 0.5, "data_dirs": ["a_tol_low"]},
            {"method": "b", "tolerance": 0.5, "data_dirs": ["b_tol_low"]},
            {"method": "a", "tolerance": 1.0, "data_dirs": ["a_tol_high"]},
        ],
    }))

    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    all_trials = pd.read_csv(destination / "cumulative_trials.csv")
    plotted = pd.read_csv(destination / "plotted_trials.csv")
    assert len(all_trials) == 9
    assert len(plotted) == 6
    assert sorted(plotted.query("method == 'a' and tolerance == 0.5").max_insertion_depth_mm) == [4.0, 9.0]
    assert sorted(plotted.query("method == 'b' and tolerance == 0.5").max_insertion_depth_mm) == [5.0, 8.0]
    assert sorted(plotted.query("method == 'a' and tolerance == 1.0").max_insertion_depth_mm) == [6.0, 7.0]

def test_cumulative_cli_nmax_overrides_json(tmp_path):
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    run = tmp_path / "run"
    _summary(run, [1.0, 4.0, 8.0])
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "nmax_trials": 1,
        "methods": [{"name": "a"}],
        "datasets": [
            {"method": "a", "tolerance": 1.0, "data_dirs": ["run"]},
        ],
    }))

    destination = tmp_path / "out"
    cumulative_plots(spec, destination, nmax_trials=2)
    plotted = pd.read_csv(destination / "plotted_trials.csv")
    assert sorted(plotted.max_insertion_depth_mm) == [4.0, 8.0]


def test_normalized_angle_and_threshold_line_are_supported(tmp_path, monkeypatch):
    import matplotlib.axes
    from peg_analysis.cumulative import cumulative_plots

    run = tmp_path / "run"
    _summary(run, [10.0, 30.0])
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "normalized_angle": True,
        "insertion_depth_threshold": 20.0,
        "methods": [{"name": "a"}],
        "datasets": [
            {"method": "a", "tolerance": 0.5, "data_dirs": ["run"]},
        ],
    }))

    horizontal_lines = []
    original_axhline = matplotlib.axes.Axes.axhline

    def recording_axhline(self, y=0, *args, **kwargs):
        horizontal_lines.append(float(y))
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", recording_axhline)
    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    assert horizontal_lines.count(20.0) == 2

    import pandas as pd
    plotted = pd.read_csv(destination / "plotted_trials.csv")
    assert plotted.normalized_initial_angular_error.max() == 1.0
    assert plotted.initial_angular_error_group_scale_deg.nunique() == 1


def test_normalized_angle_uses_separate_scale_per_method_and_tolerance(tmp_path):
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    runs = {}
    for name, depths in {
        "a_low": [10.0, 30.0],
        "b_low": [12.0, 32.0],
        "a_high": [14.0, 34.0],
    }.items():
        runs[name] = tmp_path / name
        _summary(runs[name], depths)

    # Give each group a distinct maximum initial angle.
    for name, angles in {
        "a_low": [1.0, 2.0],
        "b_low": [2.0, 8.0],
        "a_high": [3.0, 6.0],
    }.items():
        path = runs[name] / "summary.csv"
        frame = pd.read_csv(path)
        frame["initial_angular_error_deg"] = angles
        frame.to_csv(path, index=False)

    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "normalized_angle": True,
        "methods": [{"name": "a"}, {"name": "b"}],
        "datasets": [
            {"method": "a", "tolerance": 0.5, "data_dirs": ["a_low"]},
            {"method": "b", "tolerance": 0.5, "data_dirs": ["b_low"]},
            {"method": "a", "tolerance": 1.0, "data_dirs": ["a_high"]},
        ],
    }))

    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    plotted = pd.read_csv(destination / "plotted_trials.csv")
    scales = plotted.groupby(["method", "tolerance"])[
        "initial_angular_error_group_scale_deg"
    ].first().to_dict()
    assert scales == {("a", 0.5): 2.0, ("a", 1.0): 6.0, ("b", 0.5): 8.0}
    maxima = plotted.groupby(["method", "tolerance"])[
        "normalized_initial_angular_error"
    ].max()
    assert (maxima == 1.0).all()


def test_zscore_normalization_is_per_method_and_tolerance(tmp_path):
    import numpy as np
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _summary(run_a, [10.0, 20.0, 30.0])
    _summary(run_b, [12.0, 22.0, 32.0])
    for path, angles in ((run_a, [1.0, 2.0, 3.0]), (run_b, [10.0, 20.0, 30.0])):
        frame = pd.read_csv(path / "summary.csv")
        frame["initial_angular_error_deg"] = angles
        frame.to_csv(path / "summary.csv", index=False)

    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "normalization_type": "zscore",
        "methods": [{"name": "a"}, {"name": "b"}],
        "datasets": [
            {"method": "a", "tolerance": 0.5, "data_dirs": ["run_a"]},
            {"method": "b", "tolerance": 0.5, "data_dirs": ["run_b"]},
        ],
    }))
    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    plotted = pd.read_csv(destination / "plotted_trials.csv")
    for _, group in plotted.groupby(["method", "tolerance"]):
        assert np.isclose(group.normalized_initial_angular_error.mean(), 0.0)
        assert np.isclose(group.normalized_initial_angular_error.std(ddof=0), 1.0)
    assert set(plotted.angle_normalization_type) == {"zscore"}


def test_invalid_normalization_type_is_rejected(tmp_path):
    import pytest
    from peg_analysis.cumulative import load_cumulative_config

    spec = tmp_path / "bad_normalization.json"
    spec.write_text(json.dumps({
        "normalization_type": "unknown",
        "methods": [{"name": "a"}],
        "datasets": [
            {"method": "a", "tolerance": 0.5, "data_dirs": ["run"]},
        ],
    }))
    with pytest.raises(ValueError, match="none, max, minmax, zscore"):
        load_cumulative_config(spec)


def test_minmax_normalization_is_per_method_and_tolerance(tmp_path):
    import numpy as np
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _summary(run_a, [10.0, 20.0, 30.0])
    _summary(run_b, [12.0, 22.0, 32.0])
    for path, angles in ((run_a, [1.0, 2.0, 5.0]), (run_b, [10.0, 20.0, 40.0])):
        frame = pd.read_csv(path / "summary.csv")
        frame["initial_angular_error_deg"] = angles
        frame.to_csv(path / "summary.csv", index=False)

    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "normalization_type": "minmax",
        "methods": [{"name": "a"}, {"name": "b"}],
        "datasets": [
            {"method": "a", "tolerance": 0.5, "data_dirs": ["run_a"]},
            {"method": "b", "tolerance": 0.5, "data_dirs": ["run_b"]},
        ],
    }))
    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    plotted = pd.read_csv(destination / "plotted_trials.csv")
    for _, group in plotted.groupby(["method", "tolerance"]):
        assert np.isclose(group.normalized_initial_angular_error.min(), 0.0)
        assert np.isclose(group.normalized_initial_angular_error.max(), 1.0)
    assert set(plotted.angle_normalization_type) == {"minmax"}
    scales = plotted.groupby("method").initial_angular_error_group_scale_deg.first()
    assert set(scales) == {4.0, 30.0}
    assert set(plotted.groupby("method").initial_angular_error_group_min_deg.first()) == {1.0, 10.0}
    assert set(plotted.groupby("method").initial_angular_error_group_max_deg.first()) == {5.0, 40.0}


def test_cumulative_scatter_has_depth_and_axial_rows(tmp_path, monkeypatch):
    import matplotlib.figure
    from peg_analysis.cumulative import cumulative_plots

    run = tmp_path / "run"
    _summary(run, [10.0, 30.0])
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "methods": [{"name": "a"}],
        "datasets": [
            {"method": "a", "tolerance": 0.5, "data_dirs": ["run"]},
        ],
    }))

    saved_axis_labels = []
    saved_limits = []
    original_savefig = matplotlib.figure.Figure.savefig

    def recording_savefig(self, fname, *args, **kwargs):
        if str(fname).endswith("initial_angle_vs_outcomes_by_tolerance.png"):
            saved_axis_labels.extend(ax.get_ylabel() for ax in self.axes)
            saved_limits.extend((ax.get_xlim(), ax.get_ylim()) for ax in self.axes)
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    destination = tmp_path / "out"
    cumulative_plots(spec, destination)

    assert "Maximum insertion depth (mm)" in saved_axis_labels
    assert "Maximum absolute axial slip (mm)" in saved_axis_labels
    assert len(saved_limits) == 2
    assert saved_limits[0] == saved_limits[1]


def test_linear_analysis_is_drawn_and_exported_per_group(tmp_path, monkeypatch):
    import matplotlib.axes
    import numpy as np
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    run = tmp_path / "run"
    _summary(run, [10.0, 20.0, 30.0])
    frame = pd.read_csv(run / "summary.csv")
    frame["initial_angular_error_deg"] = [1.0, 2.0, 3.0]
    frame.to_csv(run / "summary.csv", index=False)
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "normalization_type": "zscore", "normalization_scope": "group",
        "linear_analysis": {"enabled": True, "show_fit": True, "show_statistics": True},
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 0.5, "data_dirs": ["run"]}],
    }))
    lines = []
    original = matplotlib.axes.Axes.plot
    def recording_plot(self, *args, **kwargs):
        lines.append(args)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    results = pd.read_csv(destination / "linear_correlations.csv")
    assert len(results) == 2
    assert set(results.outcome) == {"max_insertion_depth_mm", "max_abs_axial_slip_mm"}
    assert (results.n == 3).all()
    assert np.isclose(results.iloc[0].pearson_r, 1.0)
    assert results.iloc[0].interpretation == "evidence_of_linear_association"
    assert len(lines) >= 2


def test_linear_analysis_handles_insufficient_and_constant_groups(tmp_path):
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    run = tmp_path / "run"
    _summary(run, [10.0, 20.0])
    frame = pd.read_csv(run / "summary.csv")
    frame["initial_angular_error_deg"] = [1.0, 1.0]
    frame.to_csv(run / "summary.csv", index=False)
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "linear_analysis": True,
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 0.5, "data_dirs": ["run"]}],
    }))
    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    results = pd.read_csv(destination / "linear_correlations.csv")
    assert (results.interpretation == "insufficient_data").all()
    assert results.pearson_r.isna().all()


def test_tolerance_and_global_normalization_scopes(tmp_path):
    from peg_analysis.cumulative import load_cumulative_config
    for scope in ("tolerance", "global"):
        spec = tmp_path / f"{scope}.json"
        spec.write_text(json.dumps({
            "normalization_scope": scope,
            "methods": [{"name": "a"}],
            "datasets": [{"method": "a", "tolerance": 0.5, "data_dirs": ["run"]}],
        }))
        assert load_cumulative_config(spec)["normalization_scope"] == scope


def test_linear_analysis_line_width_is_loaded_and_validated(tmp_path):
    import pytest
    from peg_analysis.cumulative import load_cumulative_config

    base = {
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 0.5, "data_dirs": ["run"]}],
    }
    spec = tmp_path / "line_width.json"
    spec.write_text(json.dumps({**base, "linear_analysis": {"line_width": 3.25}}))
    assert load_cumulative_config(spec)["linear_analysis"]["line_width"] == 3.25

    spec.write_text(json.dumps({**base, "linear_analysis": {"line_width": 0}}))
    with pytest.raises(ValueError, match="line_width must be greater than zero"):
        load_cumulative_config(spec)


def test_linear_analysis_line_width_controls_fit(tmp_path, monkeypatch):
    import matplotlib.axes
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots

    run = tmp_path / "run"
    _summary(run, [10.0, 20.0, 30.0])
    frame = pd.read_csv(run / "summary.csv")
    frame["initial_angular_error_deg"] = [1.0, 2.0, 3.0]
    frame.to_csv(run / "summary.csv", index=False)
    spec = tmp_path / "cumulative.json"
    spec.write_text(json.dumps({
        "linear_analysis": {"line_width": 3.25},
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 0.5, "data_dirs": ["run"]}],
    }))
    widths = []
    original = matplotlib.axes.Axes.plot
    def recording_plot(self, *args, **kwargs):
        widths.append(kwargs.get("linewidth"))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
    cumulative_plots(spec, tmp_path / "out")
    assert widths and all(width == 3.25 for width in widths)


def test_jitter_and_synthetic_config_are_loaded(tmp_path):
    from peg_analysis.cumulative import load_cumulative_config
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "x_jitter": 0.2, "synthetic_n": 3, "synthetic_seed": 7,
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 1.0, "data_dirs": ["run"],
                      "x_jitter": 0.05, "synthetic_n": 2}],
    }))
    cfg = load_cumulative_config(spec)
    assert cfg["x_jitter"] == 0.2 and cfg["synthetic_n"] == 3 and cfg["synthetic_seed"] == 7
    assert cfg["datasets"][0]["x_jitter"] == 0.05
    assert cfg["datasets"][0]["synthetic_n"] == 2


def test_synthetic_n_adds_bootstrapped_rows(tmp_path):
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots
    run = tmp_path / "run"
    _summary(run, [10.0, 20.0])
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "synthetic_n": 3, "synthetic_seed": 4,
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 1.0, "data_dirs": ["run"]}],
    }))
    destination = tmp_path / "out"
    cumulative_plots(spec, destination)
    data = pd.read_csv(destination / "cumulative_trials.csv")
    assert len(data) == 5
    assert data.synthetic.sum() == 3
    assert set(data.loc[data.synthetic, "max_insertion_depth_mm"]) <= {10.0, 20.0}



def test_jitter_resamples_within_normalized_bounds_and_is_consistent_across_panes(tmp_path, monkeypatch):
    import matplotlib.axes
    import numpy as np
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots
    run = tmp_path / "run"
    _summary(run, [10.0, 20.0])
    frame = pd.read_csv(run / "summary.csv")
    frame["initial_angular_error_deg"] = [1.0, 2.0]
    frame.to_csv(run / "summary.csv", index=False)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "normalization_type": "max", "x_jitter": 0.5,
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 1.0, "data_dirs": ["run"]}],
    }))
    seen = []
    original = matplotlib.axes.Axes.scatter
    def recording(self, x, y, *args, **kwargs):
        if len(x) == 1:
            seen.append((self, float(x[0])))
        return original(self, x, y, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording)
    cumulative_plots(spec, tmp_path / "out")
    # Two trials are drawn in each of the two outcome panes.
    pane_points = {}
    for ax, x in seen:
        pane_points.setdefault(id(ax), []).append(x)
    panes = list(pane_points.values())[-2:]
    assert len(panes) == 2 and all(len(xs) == 2 for xs in panes)
    assert panes[0] == panes[1]
    assert max(panes[0]) == 1.0
    assert sum(np.isclose(x, 1.0) for x in panes[0]) == 1


def test_jitter_keeps_only_one_x_one_anchor_per_pane_across_methods(tmp_path, monkeypatch):
    import matplotlib.axes
    import numpy as np
    import pandas as pd
    from peg_analysis.cumulative import cumulative_plots
    for name, depths in (("a", [10.0, 20.0]), ("b", [12.0, 22.0])):
        run = tmp_path / name
        _summary(run, depths)
        frame = pd.read_csv(run / "summary.csv")
        frame["initial_angular_error_deg"] = [1.0, 2.0]
        frame.to_csv(run / "summary.csv", index=False)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "normalization_type": "max", "x_jitter": 0.25,
        "methods": [{"name": "a"}, {"name": "b"}],
        "datasets": [
            {"method": "a", "tolerance": 1.0, "data_dirs": ["a"]},
            {"method": "b", "tolerance": 1.0, "data_dirs": ["b"]},
        ],
    }))
    seen = []
    original = matplotlib.axes.Axes.scatter
    def recording(self, x, y, *args, **kwargs):
        if len(x) == 1:
            seen.append((id(self), float(x[0])))
        return original(self, x, y, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording)
    cumulative_plots(spec, tmp_path / "out")
    panes = {}
    for ax_id, x in seen:
        panes.setdefault(ax_id, []).append(x)
    scatter_panes = list(panes.values())[-2:]
    assert len(scatter_panes) == 2
    for xs in scatter_panes:
        assert sum(np.isclose(x, 1.0) for x in xs) == 1


def test_font_name_and_size_are_loaded(tmp_path):
    from peg_analysis.cumulative import load_cumulative_config
    spec = tmp_path / "fonts.json"
    spec.write_text(json.dumps({
        "font_name": "serif", "font_size": 14,
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 1.0, "data_dirs": ["run"]}],
    }))
    cfg = load_cumulative_config(spec)
    assert cfg["font_name"] == "serif"
    assert cfg["font_size"] == 14.0


def test_invalid_font_size_is_rejected(tmp_path):
    import pytest
    from peg_analysis.cumulative import load_cumulative_config
    spec = tmp_path / "fonts.json"
    spec.write_text(json.dumps({
        "font_size": 0,
        "methods": [{"name": "a"}],
        "datasets": [{"method": "a", "tolerance": 1.0, "data_dirs": ["run"]}],
    }))
    with pytest.raises(ValueError, match="font_size must be greater than zero"):
        load_cumulative_config(spec)


def test_cumulative_plot_has_no_hardcoded_font_sizes():
    import inspect
    import re
    import peg_analysis.cumulative as module
    source = inspect.getsource(module)
    assert not re.search(r"fontsize\s*=\s*[0-9]", source)
