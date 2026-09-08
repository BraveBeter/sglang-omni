# SPDX-License-Identifier: Apache-2.0
"""T3.4 CPU tests: FSM table, sampling order, adapter contract, state
lifecycle. Pure-logic tests (no GPU, no engine construction)."""

from types import SimpleNamespace

import torch

import tests.unit_test.moss_speech.sglang_cpu_env  # noqa: F401
from sglang_omni.models.moss_speech import fsm
from sglang_omni.models.moss_speech.fsm import (
    AUDIO_PAD,
    EOSP,
    IM_END,
    MODALITY_PAD,
    MODE_AUDIO,
    MODE_TEXT,
    SOSP,
    TEXT_ENDOFTEXT,
    MossSamplingParams,
    apply_audio_constraints,
    apply_warpers,
    channel_histories,
    finalize_row,
    initial_mode,
    next_mode,
    repetition_penalty_scores,
    row_selected_token,
    sample_row,
    stop_hit,
)


# --------------------------------------------------------------------- FSM
def test_initial_mode_from_prompt_last_row():
    assert initial_mode(torch.tensor([[1, AUDIO_PAD]])) == MODE_TEXT
    assert initial_mode(torch.tensor([[SOSP, AUDIO_PAD]])) == MODE_TEXT
    # modality_pad in the text channel -> audio branch first
    assert initial_mode(torch.tensor([[MODALITY_PAD, 3000]])) == MODE_AUDIO
    # text branch overrides when the audio channel is the pad
    assert initial_mode(torch.tensor([[MODALITY_PAD, AUDIO_PAD]])) == MODE_TEXT


def test_next_mode_transition_table():
    assert next_mode((SOSP, 5), MODE_TEXT) == MODE_AUDIO
    assert next_mode((10, EOSP), MODE_AUDIO) == MODE_TEXT
    # no transition without the exact tokens
    assert next_mode((10, 5), MODE_TEXT) == MODE_TEXT
    assert next_mode((MODALITY_PAD, 5), MODE_AUDIO) == MODE_AUDIO
    # audio-mode sosp does not fire the text->audio rule
    assert next_mode((SOSP, 5), MODE_AUDIO) == MODE_AUDIO


def test_finalize_row_audio_overwrite_and_selection():
    assert finalize_row(999, 4242, MODE_AUDIO) == (MODALITY_PAD, 4242)
    assert finalize_row(999, 4242, MODE_TEXT) == (999, 4242)
    assert row_selected_token((MODALITY_PAD, 4242)) == 4242
    assert row_selected_token((999, 4242)) == 999


def test_stop_hit_watches_text_channel_only():
    assert stop_hit((IM_END, 123)) is True
    assert stop_hit((TEXT_ENDOFTEXT, 123)) is True
    assert stop_hit((MODALITY_PAD, EOSP)) is False
    assert stop_hit((123, EOSP)) is False


# ------------------------------------------------------------------ masking
def test_audio_constraints_hard_and_min_new():
    a = torch.zeros(16512)
    apply_audio_constraints(a, generating_length=1, min_new_tokens=0)
    assert torch.isinf(a[16385:]).all() and a[16384] == 0.0
    a2 = torch.zeros(16512)
    apply_audio_constraints(a2, generating_length=2, min_new_tokens=3)
    assert a2[16384] == float("-inf")
    a3 = torch.zeros(16512)
    apply_audio_constraints(a3, generating_length=3, min_new_tokens=3)
    assert a3[16384] == 0.0


def test_repetition_penalty_hf_math():
    logits = torch.tensor([2.0, -2.0, 1.0])
    out = repetition_penalty_scores(logits, torch.tensor([0, 1]), 2.0)
    assert out[0] == 1.0  # positive -> score/penalty
    assert out[1] == -4.0  # negative -> score*penalty
    assert out[2] == 1.0  # unseen untouched
    # penalty 1.0 is a no-op
    assert torch.equal(
        repetition_penalty_scores(logits, torch.tensor([0]), 1.0), logits
    )


def test_warper_order_temperature_topk_topp():
    base = torch.tensor([10.0, 9.0, 1.0, 0.5])
    p = MossSamplingParams(do_sample=True, temperature=2.0, top_k=2, top_p=1.0)
    out = apply_warpers(base, p)
    assert torch.isinf(out[2]) and torch.isinf(out[3])
    assert out[0] == 5.0 and out[1] == 4.5
    # top_p keeps the smallest head exceeding the threshold (HF semantics:
    # keep idx0 only here; 4.0's cumulative mass already passed 0.5)
    p2 = MossSamplingParams(do_sample=True, temperature=1.0, top_k=-1, top_p=0.5)
    out2 = apply_warpers(torch.tensor([5.0, 4.0, 1.0]), p2)
    assert out2[0] == 5.0 and torch.isinf(out2[1]) and torch.isinf(out2[2])


# ------------------------------------------------------------------ sampling
def test_greedy_tie_breaks_to_lowest_index():
    scores = torch.zeros(10)
    scores[7] = 3.0
    scores[2] = 3.0  # exact tie with 7
    assert fsm.sample_channel(scores, do_sample=False) == 2


def test_sample_row_reference_order():
    # audio head tries to pick 16390 (forbidden) -> falls to best allowed
    text_logits = torch.full((151680,), -10.0)
    text_logits[42] = 5.0
    audio_logits = torch.full((16512,), -10.0)
    audio_logits[16390] = 9.0
    audio_logits[100] = 4.0
    params = MossSamplingParams(do_sample=False, repetition_penalty=1.0)
    prompt = torch.tensor([[1, AUDIO_PAD], [2, AUDIO_PAD]])
    t_hist, a_hist = channel_histories(prompt, [])
    text_tok, audio_tok = sample_row(
        text_logits, audio_logits, params, t_hist, a_hist, 1, MODE_TEXT
    )
    assert text_tok == 42 and audio_tok == 100


def test_sample_row_rep_penalty_applies_per_channel():
    text_logits = torch.full((151680,), -10.0)
    text_logits[42] = 5.0
    text_logits[43] = 4.9
    audio_logits = torch.zeros(16512)
    params = MossSamplingParams(do_sample=False, repetition_penalty=2.0)
    prompt = torch.tensor([[42, AUDIO_PAD]])
    # 42 seen in text history: 5.0 -> 2.5, so 43 (4.9) wins
    t_hist, a_hist = channel_histories(prompt, [])
    text_tok, _ = sample_row(
        text_logits, audio_logits, params, t_hist, a_hist, 1, MODE_TEXT
    )
    assert text_tok == 43


def test_seeded_sampling_reproducible_and_isolated():
    text_logits = torch.randn(151680)
    audio_logits = torch.randn(16512)
    params = MossSamplingParams(do_sample=True, temperature=1.0, top_p=0.9)
    prompt = torch.tensor([[1, AUDIO_PAD]])
    t_hist, a_hist = channel_histories(prompt, [])
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    r1 = sample_row(text_logits, audio_logits, params, t_hist, a_hist, 1, MODE_TEXT, g1)
    r2 = sample_row(text_logits, audio_logits, params, t_hist, a_hist, 1, MODE_TEXT, g2)
    assert r1 == r2
    g3 = torch.Generator().manual_seed(124)
    # different seed need not differ (probabilistic), but must be valid tokens
    r3 = sample_row(text_logits, audio_logits, params, t_hist, a_hist, 1, MODE_TEXT, g3)
    assert 0 <= r3[0] < 151680 and 0 <= r3[1] < 16385


def test_channel_histories_include_prompt_and_generated():
    prompt = torch.tensor([[1, AUDIO_PAD], [SOSP, AUDIO_PAD]])
    gen = [(MODALITY_PAD, 5), (MODALITY_PAD, 6)]
    t, a = channel_histories(prompt, gen)
    assert t.tolist() == [1, SOSP, MODALITY_PAD, MODALITY_PAD]
    assert a.tolist() == [AUDIO_PAD, AUDIO_PAD, 5, 6]


# ------------------------------------------------------------------ adapter
def _state_fixture(**over):
    from sglang_omni.models.moss_speech.payload_types import MossSpeechState

    s = MossSpeechState()
    s.input_grid = [[[151644, AUDIO_PAD], [77091, AUDIO_PAD], [198, AUDIO_PAD]]]
    s.output_modality = "text"
    s.temperature = 0.0
    s.explicit_params = ["temperature", "top_p", "top_k", "repetition_penalty"]
    s.top_p = 1.0
    s.top_k = -1
    s.repetition_penalty = 1.1
    s.effective_seed = 77
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_build_sglang_moss_request_contract():
    from sglang_omni.models.moss_speech.request_builders import (
        build_sglang_moss_request,
    )
    from sglang_omni.proto.request import StagePayload

    payload = StagePayload(request_id="r1", request=SimpleNamespace(), data=None)
    data = build_sglang_moss_request(_state_fixture(), payload=payload)
    assert data.prompt_rows.shape == (3, 2)
    assert data.mode == MODE_TEXT
    assert data.params.repetition_penalty == 1.1
    assert data.params.do_sample is False  # explicit zero temperature is greedy
    assert data.effective_seed == 77
    assert data.stage_payload is payload
    # Req carries the selected-token stream and the text stop tokens
    assert list(data.req.origin_input_ids) == [151644, 77091, 198]
    assert data.req.eos_token_ids == {TEXT_ENDOFTEXT, IM_END}


def test_build_request_do_sample_flag_and_audio_prompt():
    from sglang_omni.models.moss_speech.request_builders import (
        build_sglang_moss_request,
    )
    from sglang_omni.proto.request import StagePayload

    payload = StagePayload(request_id="r2", request=SimpleNamespace(), data=None)
    s = _state_fixture(temperature=0.7, top_p=0.95, top_k=20)
    s.input_grid = [[[151644, AUDIO_PAD], [SOSP, AUDIO_PAD], [MODALITY_PAD, 500]]]
    data = build_sglang_moss_request(s, payload=payload)
    assert data.params.do_sample is True
    assert data.mode == MODE_AUDIO  # last row text==modality_pad
    assert list(data.req.origin_input_ids) == [151644, SOSP, 500]
    # temperature<=0 is greedy
    s2 = _state_fixture(temperature=0.0)
    d2 = build_sglang_moss_request(s2, payload=payload)
    assert d2.params.do_sample is False


def test_result_adapter_reconstructs_grid_and_cleans_state():
    from sglang_omni.models.moss_speech.request_builders import (
        make_moss_speech_scheduler_adapters,
    )
    from sglang_omni.proto.request import StagePayload

    rb, ra = make_moss_speech_scheduler_adapters(model=None)
    state = _state_fixture()
    payload = StagePayload(
        request_id="r3", request=SimpleNamespace(), data=state.to_dict()
    )
    data = rb(payload)
    data.output_rows = [(11, 2), (MODALITY_PAD, 5), (IM_END, 7)]
    out = ra(data)
    out_state = out.data
    assert out_state["output_grid"] == [[11, 2], [MODALITY_PAD, 5], [IM_END, 7]]
    # voice/seed fields survive the round trip
    assert out_state["effective_seed"] == 77
    # cleanup was idempotent and released per-request state
    assert data.finished is True
    ra(data)  # must not raise


def test_reset_clears_generation_keeps_prompt():
    from sglang_omni.models.moss_speech.request_builders import (
        build_sglang_moss_request,
    )
    from sglang_omni.proto.request import StagePayload

    payload = StagePayload(request_id="r4", request=SimpleNamespace(), data=None)
    data = build_sglang_moss_request(_state_fixture(), payload=payload)
    data.output_rows = [(1, 2)]
    data.generation_steps = 3
    data.pending_feedback_queue.append(torch.zeros(4))
    data.reset()
    assert data.output_rows == [] and data.generation_steps == 0
    assert len(data.pending_feedback_queue) == 0
    assert data.mode == MODE_TEXT
    assert data.prompt_rows.shape == (3, 2)


def test_engine_builder_t34_wiring_points():
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    b = MossSpeechEngineBuilder()
    rb, ra = b.make_adapters(model=None)
    assert callable(rb) and callable(ra)
