# SPDX-License-Identifier: Apache-2.0
"""Resolve architecture metadata without interactive or remote Python execution."""

import builtins
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import AutoConfig

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

ARCHITECTURE = "Qwen3OmniMoeForConditionalGeneration"


def test_custom_config_uses_raw_architecture_without_input_or_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AutoConfig caught the denied prompt and silently hid interactive discovery."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "custom_architecture_for_discovery_test",
                "architectures": [ARCHITECTURE],
                "auto_map": {"AutoConfig": "custom_configuration.CustomConfig"},
            }
        )
    )
    prompts: list[str] = []

    def reject_input(prompt: str = "") -> str:
        prompts.append(prompt)
        raise AssertionError("Metadata discovery must not ask to execute code")

    monkeypatch.setattr(builtins, "input", reject_input)
    with patch(
        "transformers.models.auto.configuration_auto.get_class_from_dynamic_module",
        side_effect=AssertionError("Metadata discovery executed custom code"),
    ) as dynamic_config:
        resolved = resolve_config_cls_for_model_path(str(tmp_path))
        assert resolved is PIPELINE_CONFIG_REGISTRY.get_config(ARCHITECTURE)
        assert prompts == []
        dynamic_config.assert_not_called()


def test_builtin_config_resolves_with_remote_code_explicitly_disabled(
    tmp_path: Path,
) -> None:
    configuration = AutoConfig.for_model("qwen3_omni_moe", architectures=[ARCHITECTURE])
    configuration.to_json_file(tmp_path / "config.json")
    with patch.object(
        AutoConfig, "from_pretrained", wraps=AutoConfig.from_pretrained
    ) as load:
        resolved = resolve_config_cls_for_model_path(str(tmp_path))
        assert resolved is PIPELINE_CONFIG_REGISTRY.get_config(ARCHITECTURE)
        assert load.call_args.kwargs["trust_remote_code"] is False
