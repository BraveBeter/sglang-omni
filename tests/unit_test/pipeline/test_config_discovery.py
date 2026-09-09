"""Architecture discovery reads metadata without executing checkpoint code."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.config.manager import resolve_config_cls_for_model_path


def test_custom_config_uses_raw_architecture_without_input_or_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "custom_discovery_fixture",
                "architectures": ["MetadataOnlyArchitecture"],
                "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            }
        )
    )
    marker = tmp_path / "executed"
    (tmp_path / "configuration_custom.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('Checkpoint code executed')\n"
    )
    prompt = Mock(return_value="n")
    monkeypatch.setattr("builtins.input", prompt)
    expected = object()
    lookup = Mock(return_value=expected)
    monkeypatch.setattr(
        "sglang_omni.config.manager.PIPELINE_CONFIG_REGISTRY.get_config", lookup
    )
    assert resolve_config_cls_for_model_path(str(tmp_path)) is expected
    lookup.assert_called_once_with("MetadataOnlyArchitecture")
    prompt.assert_not_called()
    assert not marker.exists()


def test_builtin_config_resolves_with_remote_code_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = Mock(return_value=SimpleNamespace(architectures=["BuiltinArchitecture"]))
    lookup = Mock(return_value=object())
    monkeypatch.setattr("sglang_omni.config.manager.AutoConfig.from_pretrained", load)
    monkeypatch.setattr(
        "sglang_omni.config.manager.PIPELINE_CONFIG_REGISTRY.get_config", lookup
    )
    assert resolve_config_cls_for_model_path("builtin-model") is lookup.return_value
    load.assert_called_once_with("builtin-model", trust_remote_code=False)
    lookup.assert_called_once_with("BuiltinArchitecture")
