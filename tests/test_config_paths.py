from pathlib import Path
from peg_analysis.core import config, default_output_dir


def test_default_output_dir_is_sibling_named_after_dataset(tmp_path):
    dataset = tmp_path / "trial.h5"
    assert default_output_dir(dataset) == tmp_path / "trial"


def test_cli_paths_override_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text('main_camera_serial: "main"\nsecondary_camera_serial: "secondary"\n')
    dataset = tmp_path / "data.h5"
    result = config(cfg, dataset_path=dataset)
    assert result["dataset_path"] == str(dataset.resolve())
    assert result["output_dir"] == str((tmp_path / "data").resolve())


def test_explicit_output_dir(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text('main_camera_serial: "main"\nsecondary_camera_serial: "secondary"\n')
    result = config(cfg, dataset_path=tmp_path / "data.h5", output_dir=tmp_path / "results")
    assert result["output_dir"] == str((tmp_path / "results").resolve())
