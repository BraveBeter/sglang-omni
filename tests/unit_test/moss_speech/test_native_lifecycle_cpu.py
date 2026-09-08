# SPDX-License-Identifier: Apache-2.0
"""Regression tests for native request state and boundary semantics."""
import tests.unit_test.moss_speech.sglang_cpu_env  # noqa: F401  # isort: skip

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_speech import fsm
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.request_builders import build_sglang_moss_request
from sglang_omni.proto.request import StagePayload


def make_data(**values):
    state = MossSpeechState()
    state.input_grid = [[12, 512], [13, 512]]
    state.attention_mask = [1, 1]
    state.temperature = 0.0
    state.explicit_params = ["temperature"]
    for k, v in values.items():
        setattr(state, k, v)
    return build_sglang_moss_request(
        state,
        payload=StagePayload(
            request_id="regression", request=SimpleNamespace(), data=state.to_dict()
        ),
    )


def test_both_fsm_transitions_apply_in_reference_order():
    assert fsm.next_mode((fsm.SOSP, fsm.EOSP), fsm.MODE_TEXT) == fsm.MODE_TEXT


def test_zero_temperature_survives_other_sampling_parameters():
    data = make_data(top_p=0.9, top_k=20)
    assert data.params.temperature == 0.0
    assert not data.params.do_sample


def test_length_wire_field_reaches_scheduler():
    data = make_data(max_new_tokens=7)
    assert data.params.max_new_tokens == 7
    assert data.req.sampling_params.max_new_tokens == 7


def test_explicit_unit_temperature_means_sampling():
    data = make_data(temperature=1.0, top_p=1.0, top_k=-1)
    assert data.params.do_sample


def test_default_model_sampling_is_distinct_from_explicit_values():
    data = make_data(
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        explicit_params=[],
    )
    assert (
        data.params.temperature,
        data.params.top_p,
        data.params.top_k,
        data.params.repetition_penalty,
    ) == (0.6, 0.95, 20, 1.1)


def test_retry_restarts_request_rng():
    data = make_data(effective_seed=123)
    first = MossSpeechModelRunner._generator_for(data, torch.device("cpu"))
    expected = torch.rand(8, generator=first)
    data.reset()
    second = MossSpeechModelRunner._generator_for(data, torch.device("cpu"))
    assert torch.equal(torch.rand(8, generator=second), expected)


def test_unsupported_stop_is_rejected_before_model_execution():
    with pytest.raises(ValueError, match="stop"):
        make_data(stop=["stop here"])


def test_left_padding_is_removed_using_mask():
    data = make_data(
        input_grid=[[0, 512], [12, 512], [13, 512]], attention_mask=[0, 1, 1]
    )
    assert data.prompt_rows.tolist() == [[12, 512], [13, 512]]


def test_min_length_uses_generated_rows_not_framework_counter():
    data = make_data(min_new_tokens=2)
    data.generation_steps = 99  # framework bookkeeping must not affect FSM time
    text = torch.zeros(fsm.TEXT_VOCAB)
    audio = torch.zeros(fsm.AUDIO_VOCAB)
    audio[fsm.EOSP] = 10
    runner = object.__new__(MossSpeechModelRunner)
    row = runner._select_row(data, text, audio)
    assert row[1] != fsm.EOSP


def test_norm_rounds_residual_and_normalized_value_before_weight():
    from sglang_omni.models.moss_speech.sglang_model import MossSpeechRMSNorm

    gen = torch.Generator().manual_seed(921)
    x = torch.randn(3, 128, generator=gen).bfloat16()
    residual = torch.randn(3, 128, generator=gen).bfloat16()
    norm = MossSpeechRMSNorm(128, eps=1e-6).bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, generator=gen).bfloat16())
    original_x = x.clone()
    summed = x + residual
    f = summed.float()
    expected = (
        f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16() * norm.weight
    actual, actual_residual = norm(x, residual)
    assert torch.equal(actual_residual, summed)
    assert torch.equal(actual, expected)
    assert torch.equal(x, original_x)


def test_abort_releases_only_matching_request_state_and_logits():
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    builder = MossSpeechEngineBuilder()
    model = SimpleNamespace(
        _dual_logits_by_rid={
            ("regression", "decode"): (1, 2),
            ("other", "decode"): (3, 4),
        }
    )
    rb, _ = builder.make_adapters(model)
    seed_data = make_data()
    data = rb(seed_data.stage_payload)
    data.output_rows = [(14, 9)]
    data.pending_feedback_queue.append(torch.ones(3))
    callback = builder.make_abort_callback()
    callback("regression")
    callback("regression")
    assert data.output_rows == [] and not data.pending_feedback_queue
    assert ("regression", "decode") not in model._dual_logits_by_rid
    assert ("other", "decode") in model._dual_logits_by_rid


def test_duplicate_live_request_id_is_rejected():
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    rb, _ = MossSpeechEngineBuilder().make_adapters(None)
    payload = make_data().stage_payload
    rb(payload)
    with pytest.raises(ValueError, match="duplicate"):
        rb(payload)


def test_modalities_admit_in_fair_groups_and_cancel_pending():
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
    from sglang_omni.scheduling.types import DeferredAdmission

    builder = MossSpeechEngineBuilder()
    rb, _ = builder.make_adapters(None)
    payload = make_data().stage_payload

    def request(rid, modality):
        return rb(
            StagePayload(
                request_id=rid,
                request=SimpleNamespace(),
                data={**payload.data, "output_modality": modality},
            )
        )

    t1 = request("t1", "text")
    t2 = request("t2", "text")
    a1 = request("a1", "audio")
    t3 = request("t3", "text")
    assert not isinstance(t1, DeferredAdmission) and not isinstance(
        t2, DeferredAdmission
    )
    assert isinstance(a1, DeferredAdmission) and isinstance(t3, DeferredAdmission)
    done = builder.make_request_finished_callback()
    done("t1")
    assert not a1.ready.done()
    done("t2")
    assert a1.ready.done() and not t3.ready.done()
    builder.make_abort_callback()("a1")
    assert t3.ready.done()


def test_request_partitioned_linear_preserves_single_request_shapes(monkeypatch):
    from sglang_omni.models.moss_speech.sglang_model import request_linear

    seen = []
    original = torch.nn.functional.linear

    def recording(x, weight):
        seen.append(tuple(x.shape))
        return original(x, weight)

    monkeypatch.setattr(torch.nn.functional, "linear", recording)
    x = torch.randn(7, 4)
    weight = torch.randn(3, 4)
    result = request_linear(x, weight, [5, 2])
    assert seen == [(5, 4), (2, 4)]
    torch.testing.assert_close(result, original(x, weight))


def test_rope_frequency_initialization_is_cpu_float32():
    from sglang_omni.models.moss_speech.sglang_model import reference_rope_frequencies

    frequencies = reference_rope_frequencies(128, 1000000)
    expected = 1.0 / (1000000 ** (torch.arange(0, 128, 2).float() / 128))
    assert frequencies.device.type == "cpu"
    assert frequencies.dtype == torch.float32
    assert torch.equal(frequencies, expected)


@pytest.mark.parametrize(
    "rows", [[[-1, 512]], [[151680, 512]], [[12, -1]], [[12, 16512]]]
)
def test_out_of_vocabulary_native_input_rejected_on_cpu(rows):
    with pytest.raises(ValueError):
        make_data(input_grid=rows, attention_mask=[1])
