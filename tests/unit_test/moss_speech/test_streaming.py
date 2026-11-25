# SPDX-License-Identifier: Apache-2.0
"""MOSS-Speech increment, lookahead, final flush and cancellation contracts."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torchaudio
from diffusers.models.attention_processor import Attention

from sglang_omni.models.moss_speech.codec import MossSpeechCodec
from sglang_omni.models.moss_speech.config import MossSpeechFlowArgs
from sglang_omni.models.moss_speech.flow_attention import (
    install_flow_attention_broadcast,
)
from sglang_omni.models.moss_speech.flow_precision import (
    BF16FlowDecoder,
    FlowEulerDecoder,
)
from sglang_omni.models.moss_speech.fsm import AUDIO_PAD, EOSP, MODALITY_PAD, SOSP
from sglang_omni.models.moss_speech.request_builders import (
    RequestValidationError,
    build_result,
    build_sglang_moss_request,
    preprocess_request,
)
from sglang_omni.models.moss_speech.streaming import (
    MossSpeechStreamingVocoder,
    build_audio_stream,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.fixtures.moss_speech import (
    TextTokenizer,
    audio_reference,
    make_codec,
)


@pytest.mark.parametrize("flow_dtype", [torch.float32, torch.bfloat16])
def test_broadcast_preserves_causal_attention_values(flow_dtype: torch.dtype) -> None:
    attention = Attention(query_dim=8, heads=2, dim_head=4).to(dtype=flow_dtype)
    hidden_states = torch.arange(32, dtype=flow_dtype).reshape(1, 4, 8) / 32
    attention_bias = torch.full((1, 4, 4), float("-inf"), dtype=flow_dtype).triu(1)
    expected = attention(hidden_states, attention_mask=attention_bias)

    decoder = torch.nn.Module()
    decoder.estimator = attention
    install_flow_attention_broadcast(
        BF16FlowDecoder(decoder) if flow_dtype is torch.bfloat16 else decoder
    )

    broadcast_bias = attention.prepare_attention_mask(attention_bias, 4, 1)
    assert broadcast_bias.stride(1) == 0
    torch.testing.assert_close(
        attention(hidden_states, attention_mask=attention_bias), expected
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("configured_steps", "streaming", "expected_steps"),
    [(None, True, 10), (6, True, 6), (6, False, 10)],
)
def test_streaming_euler_grid(
    dtype: torch.dtype,
    configured_steps: int | None,
    streaming: bool,
    expected_steps: int,
) -> None:
    settings = MossSpeechFlowArgs(streaming_flow_steps=configured_steps)
    native_decoder = torch.nn.Module()
    native_decoder.rand_noise = torch.ones(1, 80, 12)
    native_decoder.t_scheduler = "cosine"
    native_decoder.solve_euler = Mock(return_value=torch.empty(0))
    decoder = FlowEulerDecoder(
        native_decoder, streaming_flow_steps=settings.streaming_flow_steps
    )
    conditioning = torch.zeros(1, 80, 5, dtype=dtype)
    decoder(conditioning, torch.ones(1, 1, 5), 10, streaming=streaming)
    noise, timesteps = native_decoder.solve_euler.call_args.args[:2]
    torch.testing.assert_close(noise, torch.ones_like(conditioning))
    expected = torch.linspace(0, 1, expected_steps + 1, dtype=dtype)
    assert torch.equal(timesteps, 1 - torch.cos(expected * 0.5 * torch.pi))


@pytest.mark.parametrize("steps", [0, -1])
def test_streaming_flow_steps_must_be_positive(steps: int) -> None:
    with pytest.raises(ValueError, match="streaming_flow_steps"):
        MossSpeechFlowArgs(streaming_flow_steps=steps)


@pytest.mark.parametrize("steps", [None, 6, 10])
def test_reference_graphs_require_the_four_step_grid(steps: int | None) -> None:
    codec = Mock(device=torch.device("cuda:0"))
    with pytest.raises(ValueError, match="four streaming Flow steps"):
        MossSpeechStreamingVocoder(
            codec,
            Path("codec"),
            TextTokenizer(),
            flow_dtype=torch.bfloat16,
            streaming_flow_steps=steps,
            prefill_streaming_reference=True,
        )


@pytest.mark.accelerator
def test_reference_prefix_is_independent_of_cache_hits_and_eviction(
    tmp_path: Path,
) -> None:
    """A returned stream must be reproducible after another reference evicts its prefix."""
    codec_path = os.environ.get("MOSS_SPEECH_CODEC_PATH")
    reference_audio = os.environ.get("MOSS_SPEECH_REFERENCE_AUDIO")
    if not torch.cuda.is_available() or not codec_path or not reference_audio:
        pytest.skip("Requires CUDA and local codec/reference assets")
    else:
        pass
    samples, sample_rate = torchaudio.load(reference_audio)
    if samples.size(1) < 6 * sample_rate:
        pytest.skip("Requires two references with complete Flow chunks")
    else:
        pass
    alternate = tmp_path / "alternate.wav"
    torchaudio.save(alternate, samples[:, : samples.size(1) // 2], sample_rate)
    codec = MossSpeechCodec(
        codec_path, torch.device("cuda:0"), flow_dtype=torch.bfloat16
    )
    vocoder = MossSpeechStreamingVocoder(
        codec,
        Path(codec_path),
        TextTokenizer(),
        flow_dtype=torch.bfloat16,
        prefill_streaming_reference=True,
        streaming_flow_steps=4,
    )
    chunks = []
    for index, reference_path in enumerate(
        [reference_audio, reference_audio, str(alternate), reference_audio]
    ):
        request_id = str(index)
        state = vocoder.create_stream_state(request_id)
        vocoder.latch_stream_contract(
            request_id,
            state,
            {"reference_audio": reference_path, "effective_seed": 42},
            origin="stream",
        )
        first = (
            25
            + (-state.voice["prompt_token"].size(1) % 25)
            + vocoder.flow.pre_lookahead_len
        )
        state.codes = [1] * first
        increment = vocoder.decode_delta(request_id, state, is_final=False)
        state.codes.extend([3, 9])
        waveform = vocoder.decode_delta(request_id, state, is_final=True)
        assert waveform is not None and increment is not None
        assert vocoder.flow.decoder.decoder.forward_estimator.graphs
        assert (
            vocoder.reference_prefix is not None
            and vocoder.reference_prefix.cached is not None
        )
        chunks.append(torch.cat((increment, waveform)))
    assert torch.equal(chunks[0], chunks[1]) and torch.equal(chunks[0], chunks[3])
    assert not torch.equal(chunks[0], chunks[2])
    vocoder.on_serving_stop()


class TestFlow:
    __test__ = False
    pre_lookahead_len: int = 3
    token_mel_ratio: int = 4
    conditioning_dtype: torch.dtype = torch.float32

    def inference(
        self,
        *,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        streaming: bool,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert streaming
        assert prompt_feat.dtype == embedding.dtype == self.conditioning_dtype
        assert prompt_token.dtype == torch.int32
        if finalize:
            committed_codes = token.shape[-1]
        else:
            committed_codes = token.shape[-1] - self.pre_lookahead_len
            assert (prompt_token.shape[-1] + committed_codes) % 25 == 0
        mel = token[:, None, :committed_codes].repeat_interleave(4, dim=-1)
        return mel.float(), mel.float()


class TestHiFT:
    __test__ = False

    def inference(
        self, *, speech_feat: torch.Tensor, cache_source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = speech_feat.repeat_interleave(480, dim=-1)
        waveform = (
            source.squeeze(1) * 0.001 + torch.rand_like(source.squeeze(1)) * 0.001
        )
        return waveform, source


@pytest.mark.parametrize("stream", [False, True])
def test_ar_stream_routes_only_generated_audio_through_first_eosp(stream: bool) -> None:
    prepared = preprocess_request(
        StagePayload(
            "request",
            OmniRequest(
                inputs="Hello",
                params={"stream": stream, "seed": 42},
                metadata={"tts_params": {"ref_audio": "reference.wav"}},
            ),
            {},
        ),
        TextTokenizer(),
    )
    native = build_sglang_moss_request(prepared)
    output = Mock()
    sent_messages = []
    for row in (
        (SOSP, AUDIO_PAD),
        (MODALITY_PAD, 512),
        (MODALITY_PAD, 7),
        (MODALITY_PAD, EOSP),
        (MODALITY_PAD, 8),
    ):
        native.output_rows.append(row)
        sent_messages.extend(build_audio_stream("request", native, output))
        assert build_audio_stream("request", native, output) == []
    assert [item.data.tolist() for item in sent_messages] == (
        [[512], [7]] if stream else []
    )
    if stream:
        first, second = sent_messages
        assert first.target == second.target == "audio_vocoder"
        assert first.metadata == {
            "stream": True,
            "reference_audio": "reference.wav",
            "effective_seed": 42,
        }
        assert second.metadata == {"stream": True}
    else:
        pass


@pytest.mark.parametrize(
    ("audio_codes", "flow_dtype"),
    [(0, None), (2, None), (68, None), (68, torch.bfloat16)],
)
def test_codec_flushes_each_sample_once_after_stream_done(
    audio_codes: int, flow_dtype: torch.dtype | None
) -> None:
    codec = make_codec()
    codec.decoder.hift = TestHiFT()
    scheduler = MossSpeechStreamingVocoder(
        codec, Path("unused"), TextTokenizer(), flow_dtype=flow_dtype
    )
    flow = TestFlow()
    flow.conditioning_dtype = flow_dtype or torch.float32
    scheduler.flow = flow
    reference = audio_reference()
    prepared = preprocess_request(
        StagePayload(
            "request",
            OmniRequest(
                inputs="Hello",
                params={"stream": True, "seed": 42},
                metadata={"tts_params": {"ref_audio": reference}},
            ),
            {},
        ),
        TextTokenizer(),
    )
    native = build_sglang_moss_request(prepared)
    initial_rng = torch.get_rng_state().clone()
    for code in range(audio_codes):
        native.output_rows.append((MODALITY_PAD, code))
        (message,) = build_audio_stream("request", native, Mock())
        scheduler.handle_stream_chunk(
            "request", StreamItem(code, message.data, "ar_engine", message.metadata)
        )
    if audio_codes == 68:
        assert scheduler.outbox.qsize() == 2
    else:
        assert scheduler.outbox.empty()
    native.finish_reason = "length"
    scheduler.handle_stream_done("request")
    terminal = build_result(native)
    if audio_codes:
        scheduler.handle_streaming_new_request("request", terminal)
        messages = [
            scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())
        ]
        samples = sum(
            message.data["audio_waveform_shape"][0]
            for message in messages[:-1]
            if message.type == "stream"
        )
        assert samples == audio_codes * 1920
        assert messages[-1].data.data == {
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": len(native.req.origin_input_ids),
                "completion_tokens": audio_codes,
                "total_tokens": len(native.req.origin_input_ids) + audio_codes,
            },
        }
        assert not scheduler.stream_states
    else:
        with pytest.raises(RequestValidationError, match="before producing audio"):
            scheduler.handle_streaming_new_request("request", terminal)
        scheduler.abort("request")
        assert scheduler.outbox.empty()
        assert not scheduler.stream_states
    assert torch.equal(torch.get_rng_state(), initial_rng)
