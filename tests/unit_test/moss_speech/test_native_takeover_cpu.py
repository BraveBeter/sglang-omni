# SPDX-License-Identifier: Apache-2.0
"""Factory takeover and valid audio-segment boundary regressions."""
from types import SimpleNamespace

import pytest

from . import sglang_cpu_env  # noqa: F401


def test_formal_factory_calls_native_builder_after_preflight(monkeypatch):
    from sglang_omni.models.moss_speech import stages
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    calls = []
    monkeypatch.setattr(
        stages,
        "validate_ar_preconditions",
        lambda *a, **k: calls.append(("preflight", a, k)),
    )
    marker = object()

    def build(self, path, **kwargs):
        calls.append(("native", path, kwargs, self.context_length))
        return marker

    monkeypatch.setattr(MossSpeechEngineBuilder, "build", build)
    result = stages.create_ar_engine_executor(
        "checkpoint", gpu_id=2, server_args_overrides={"mem_fraction_static": 0.6}
    )
    assert result is marker
    assert [c[0] for c in calls] == ["preflight", "native"]
    assert calls[1][2]["gpu_id"] == 2
    assert calls[1][2]["server_args_overrides"]["mem_fraction_static"] == 0.6
    assert calls[1][3] == stages.DEFAULT_CONTEXT_LIMIT


def test_audio_extraction_ignores_text_mode_audio_eosp():
    from sglang_omni.models.moss_speech.stages import extract_output_codes

    state = SimpleNamespace(
        output_grid=[
            [42, 16384],
            [151646, 10],
            [151667, 21],
            [151667, 22],
            [151667, 16384],
            [43, 99],
        ]
    )
    assert extract_output_codes(state) == [21, 22]


def test_audio_extraction_rejects_invalid_audio_codes():
    from sglang_omni.models.moss_speech.stages import extract_output_codes

    with pytest.raises(ValueError):
        extract_output_codes(SimpleNamespace(output_grid=[[151667, 16385]]))
