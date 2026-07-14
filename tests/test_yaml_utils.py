from lm_eval.utils import load_yaml_config


def test_load_yaml_config_reuses_local_function_module(tmp_path):
    module_path = tmp_path / "helpers.py"
    module_path.write_text(
        "def first():\n    return 'first'\n\ndef second():\n    return 'second'\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "first: !function helpers.first\nsecond: !function helpers.second\n",
        encoding="utf-8",
    )

    config = load_yaml_config(yaml_path=config_path, mode="full")

    assert config["first"]() == "first"
    assert config["second"]() == "second"
    assert config["first"].__globals__ is config["second"].__globals__
