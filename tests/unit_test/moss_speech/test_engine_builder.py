# SPDX-License-Identifier: Apache-2.0
"""T3.2: engine builder policy tests (CPU-only; no infrastructure build)."""

from types import SimpleNamespace

import pytest

import tests.unit_test.moss_speech.sglang_cpu_env  # noqa: F401  (import side effects)
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder


def _server_args(**kw):
    base = dict(
        disable_cuda_graph=True,
        enable_torch_compile=False,
        disable_radix_cache=True,
        tp_size=1,
        dtype="bfloat16",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_builder_pins_arch_and_context():
    b = MossSpeechEngineBuilder()
    assert b.model_arch_override == "MossSpeechForCausalLM"
    assert b.context_length == 40960
    b2 = MossSpeechEngineBuilder(context_length=8192)
    assert b2.context_length == 8192


def test_generation_defaults_lock_v1_policy():
    d = MossSpeechEngineBuilder().generation_defaults(dtype="bfloat16")
    assert d["disable_cuda_graph"] and d["disable_radix_cache"]
    assert not d["enable_torch_compile"]
    assert d["tp_size"] == 1


def test_adjust_overrides_fail_closed():
    b = MossSpeechEngineBuilder()
    with pytest.raises(ValueError, match="rejects"):
        b.adjust_overrides({"enable_torch_compile": True, "tp_size": 4})
    overrides = {}
    b.adjust_overrides(overrides)
    assert overrides["disable_cuda_graph"] is True
    assert overrides["enable_torch_compile"] is False
    assert overrides["tp_size"] == 1
    assert overrides["mem_fraction_static"] == 0.72


def test_validate_rejects_policy_violations():
    b = MossSpeechEngineBuilder()
    with pytest.raises(ValueError, match="cuda_graph"):
        b.validate_before_infrastructure(_server_args(disable_cuda_graph=False))
    with pytest.raises(ValueError, match="tp_size"):
        b.validate_before_infrastructure(_server_args(tp_size=2))
    with pytest.raises(ValueError, match="dtype"):
        b.validate_before_infrastructure(_server_args(dtype="float16"))
    b.validate_before_infrastructure(_server_args())  # clean pass


def test_t34_adapters_wired():
    # make_adapters returns the request/result adapter pair (T3.4); the
    # runner construction needs a live model worker and is GPU-verified in
    # the engine parity driver instead.
    b = MossSpeechEngineBuilder()
    rb, ra = b.make_adapters(model=None)
    assert callable(rb) and callable(ra)


@pytest.mark.parametrize(
    "overrides", [{"attention_backend": "flashinfer"}, {"trust_remote_code": True}]
)
def test_builder_rejects_unqualified_attention_and_remote_config(overrides):
    with pytest.raises(ValueError, match="rejects"):
        MossSpeechEngineBuilder().adjust_overrides(overrides)
