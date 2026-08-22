from lm_eval.tasks._yaml_loader import load_yaml


def test_load_yaml_reuses_local_function_module(tmp_path):
    """Two `!function` tags into one file must land in one module object.

    Re-importing the file per tag repeats its module-level side effects --
    registry decorators among them -- and hands out functions that do not share
    state with each other.
    """
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

    config = load_yaml(config_path)

    assert config["first"]() == "first"
    assert config["second"]() == "second"
    assert config["first"].__globals__ is config["second"].__globals__


def test_load_yaml_resolves_function_module_beside_parent_dir(tmp_path):
    """`!function ../shared.fn` addresses a sibling directory, not a package.

    Task directories share helpers by pointing one level up. Treating the dots
    of "../" as package separators would look for "//shared.py" and fail.
    """
    (tmp_path / "shared.py").write_text(
        "def fn():\n    return 'shared'\n", encoding="utf-8"
    )
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    config_path = task_dir / "task.yaml"
    config_path.write_text("fn: !function ../shared.fn\n", encoding="utf-8")

    assert load_yaml(config_path)["fn"]() == "shared"


def test_load_yaml_include_list_inherits_right_to_left(tmp_path):
    """`include: [a, b]` lets `a` win over `b`."""
    (tmp_path / "base.yaml").write_text(
        "output_type: multiple_choice\nkept: base\n", encoding="utf-8"
    )
    (tmp_path / "overlay.yaml").write_text(
        "output_type: generate_until\n", encoding="utf-8"
    )
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "include: [overlay.yaml, base.yaml]\ntask: t\n", encoding="utf-8"
    )

    cfg = load_yaml(config_path)
    assert cfg["output_type"] == "generate_until"
    assert cfg["kept"] == "base"
    assert cfg["task"] == "t"


def test_load_yaml_allows_diamond_includes(tmp_path):
    """One base reached through two branches is a diamond, not a cycle."""
    (tmp_path / "base.yaml").write_text("kept: base\n", encoding="utf-8")
    (tmp_path / "left.yaml").write_text(
        "include: base.yaml\nleft: yes\n", encoding="utf-8"
    )
    (tmp_path / "right.yaml").write_text(
        "include: base.yaml\nright: yes\n", encoding="utf-8"
    )
    config_path = tmp_path / "task.yaml"
    config_path.write_text("include: [left.yaml, right.yaml]\n", encoding="utf-8")

    cfg = load_yaml(config_path)
    assert cfg["kept"] == "base"
    assert cfg["left"] is True and cfg["right"] is True


def test_load_yaml_still_rejects_a_real_cycle(tmp_path):
    import pytest

    (tmp_path / "a.yaml").write_text("include: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("include: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Include cycle"):
        load_yaml(tmp_path / "a.yaml")
