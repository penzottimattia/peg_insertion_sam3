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
