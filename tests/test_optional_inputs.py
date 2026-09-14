from peg_analysis.core import camera_roles, configured_objects, config


def test_secondary_camera_is_optional(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("main_camera_serial: main\ntext_prompts:\n  main:\n    peg: peg\n")
    result = config(cfg, dataset_path=tmp_path / "data.h5")
    assert camera_roles(result) == [("main", "main")]


def test_hand_and_holder_prompts_are_optional():
    c = {"text_prompts": {"main": {"peg": "peg", "hand": None}}}
    assert configured_objects(c, "main") == {"peg": "peg"}
