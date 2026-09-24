"""Same bounded scheduling and arithmetic as the qualified reference.

Reference text is parsed only; it is independent of the current implementation."""

import ast
from pathlib import Path

from sglang_omni.models.moss_speech.components import cfm_qkv_multistream as actual


def definition(path, name):
    tree = ast.parse(Path(path).read_text())
    return ast.dump(
        next(
            n
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
        ),
        include_attributes=False,
    )


def test_run_schedule_is_unchanged_from_full_cfm_parent():
    source = Path(__file__).resolve().parent / "fixtures/cfm_qkv_reference"
    for name, file in (
        ("ParallelTailBlock", "operations.py.txt"),
        ("QkvSchedule", "schedule.py.txt"),
    ):
        assert definition(actual.__file__, name) == definition(source / file, name)


def test_identity_only_changes_default_main_stream_admission():
    source = Path(__file__).resolve().parent / "fixtures/cfm_qkv_reference"
    old = ast.parse((source / "schedule.py.txt").read_text())
    new = ast.parse(Path(actual.__file__).read_text())
    old_fn = next(
        n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "identity"
    )
    new_fn = next(
        n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "identity"
    )

    def guard(fn):
        return next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and len(n.args) == 2
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value == "Invalid owner or device"
        )

    before, after = guard(old_fn), guard(new_fn)
    expected = ast.parse(
        "result[0][0] >= 0 and all(x[0] > 0 for x in result[1:]) and all(x[1] > 0 and x[2] > 0 for x in result) and len({x[3] for x in result}) == 1",
        mode="eval",
    ).body
    assert ast.dump(after.args[0]) == ast.dump(expected)
    after.args[0] = before.args[0]
    assert ast.dump(old_fn, include_attributes=False) == ast.dump(
        new_fn, include_attributes=False
    )
