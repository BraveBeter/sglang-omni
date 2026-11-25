# SPDX-License-Identifier: Apache-2.0
"""MOSS-Speech increment, lookahead, final flush and cancellation contracts."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

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
