# SPDX-License-Identifier: Apache-2.0
"""Complete asynchronous Codec isolation evidence against an independent Torch owner."""

import pickle
import random
from typing import TypedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
from diffusers.models.attention_processor import Attention
from safetensors.torch import load_file

from sglang_omni.models.moss_speech.async_vocoder import MossSpeechAsyncVocoder
from sglang_omni.models.moss_speech.components.codec_async import CodecStepFailure
from sglang_omni.models.moss_speech.components.streaming_codec import (
    CodecStreamSession,
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.stages import create_audio_vocoder_executor
from sglang_omni.vendor.moss_speech_codec.matcha_components.bias_broadcast import (
    LiveBiasBroadcastProcessor,
)


class AsyncCodecEvidence(TypedDict):
    artifacts: dict[str, torch.Tensor]
    executions: dict[str, int]
    flow_slots: int
    pcm_exact: bool


@torch.no_grad()
def assert_async_codec(codec_path: str, reference_tensors: str) -> AsyncCodecEvidence:
    """Preserve complete511-code PCM, per-request RNG and cancellation across actual workers."""
    oracle = load_file(reference_tensors, device="cpu")
    codes = oracle["codes511.codes"][0].tolist()
    voice = {
        name: oracle["voice." + name].clone()
        for name in ["prompt_token", "prompt_feat", "embedding"]
    }
    request_codes = {"first": codes, "second": [(code + 11) % 16384 for code in codes]}
    voices = {
        name: {**voice, "embedding": torch.roll(voice["embedding"], index, dims=1)}
        for index, name in enumerate(request_codes)
    }
    before = (
        pickle.dumps(random.getstate()),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(),
        torch.cuda.get_rng_state().clone(),
    )
    independent = MossSpeechStreamingCodec(codec_path, chunk_size=25, device="cuda:0")
    executor = create_audio_vocoder_executor(
        "unused",
        codec_path=codec_path,
        device="cuda",
        gpu_id=0,
        chunk_size=25,
        broadcast_attention_bias=True,
        cfm_graph_mel_length=1500,
        codec_flow_slots=2,
    )
    assert isinstance(executor, MossSpeechAsyncVocoder)
    artifacts: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    expected_rng: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    try:
        executor.on_serving_start()
        assert executor.async_codec is not None and torch.backends.cudnn.enabled
        pool = executor.async_codec
        codec = executor.codec
        for index, request_id in enumerate(request_codes):
            independent.begin(request_id, voices[request_id], seed=42 + index)
            state = independent.sessions[request_id]
            expected[request_id] = torch.cat(
                independent.push(request_id, request_codes[request_id], final=True)
            )
            assert state.device_rng is not None
            expected_rng[request_id] = (state.cpu_rng.clone(), state.device_rng.clone())
            artifacts["async." + request_id + ".native_pcm"] = expected[
                request_id
            ].contiguous()
        solo: dict[str, torch.Tensor] = {}

        def run(request_ids: list[str]) -> dict[str, torch.Tensor]:
            outputs: dict[str, list[torch.Tensor]] = {name: [] for name in request_ids}
            states: dict[str, CodecStreamSession] = {}
            for name in request_ids:
                index = list(request_codes).index(name)
                codec.begin(name, voices[name], seed=42 + index)
                states[name] = codec.sessions[name]
                pool.ingest(name, request_codes[name], final=True)
            while codec.sessions or pool.pending:
                for name in list(codec.sessions):
                    pool.submit(name)
                for name, waveform in pool.collect(wait=True).items():
                    outputs[name].append(waveform)
            complete: dict[str, torch.Tensor] = {}
            for name, chunks in outputs.items():
                complete[name] = torch.cat(chunks)
                torch.testing.assert_close(
                    complete[name], expected[name], atol=1e-5, rtol=1e-5
                )
                assert states[name].cursor == len(codes)
                assert (
                    states[name].emitted_samples == len(codes) * codec.samples_per_code
                )
                assert states[name].ledger[-1]["final"]
                assert torch.equal(states[name].cpu_rng, expected_rng[name][0])
                assert torch.equal(states[name].device_rng, expected_rng[name][1])
            return complete

        for name in request_codes:
            solo.update(run([name]))
        paired = run(list(reversed(request_codes)))
        assert all(torch.equal(paired[name], solo[name]) for name in request_codes)
        for name in paired:
            artifacts["async." + name + ".solo_pcm"] = solo[name].contiguous()
            artifacts["async." + name + ".paired_pcm"] = paired[name].contiguous()
        codec.begin("unready", voice, seed=42)
        with patch.object(
            torch.random,
            "fork_rng",
            side_effect=AssertionError("Unready async entered RNG"),
        ):
            pool.ingest("unready", codes[:1])
            assert not pool.submit("unready")
        pool.cancel("unready")
        codec.begin("short-control", voice, seed=42)
        pool.ingest("short-control", codes[:32], final=True)
        assert pool.submit("short-control")
        short_control = pool.collect(wait=True)["short-control"]
        independent.begin("short-native", voice, seed=42)
        short_native = torch.cat(
            independent.push("short-native", codes[:32], final=True)
        )
        torch.testing.assert_close(short_control, short_native, atol=1e-5, rtol=1e-5)
        artifacts["async.short.native_pcm"] = short_native.contiguous()
        artifacts["async.short.solo_pcm"] = short_control.contiguous()
        codec.begin("reused", voice, seed=42)
        pool.ingest("reused", codes[:32], final=True)
        assert pool.submit("reused")
        pool.cancel("reused")
        codec.begin("reused", voice, seed=44)
        replacement = codec.sessions["reused"]
        while pool.pending:
            assert not pool.collect(wait=True)
        assert replacement.cursor == 0 and not replacement.ledger
        pool.cancel("reused")
        codec.begin("held", voice, seed=43)
        pool.ingest("held", codes[:1])
        held = codec.sessions["held"]
        codec.begin("failed", voice, seed=42)
        pool.ingest("failed", codes[:32], final=True)
        with patch.object(
            pool.slots[0].flow,
            "inference",
            side_effect=RuntimeError("Injected async Flow failure"),
        ):
            assert pool.submit("failed")
            with pytest.raises(CodecStepFailure, match="Injected async Flow failure"):
                pool.collect(wait=True)
        assert codec.sessions == {"held": held} and held.cursor == 0 and not held.ledger
        pool.cancel("held")
        codec.begin("retry", voice, seed=42)
        pool.ingest("retry", codes[:32], final=True)
        assert pool.submit("retry")
        retry = pool.collect(wait=True)["retry"]
        assert torch.equal(retry, short_control)
        artifacts["async.short.retry_pcm"] = retry.contiguous()
        assert not pool.pending and not codec.sessions
        assert len({slot.stream.cuda_stream for slot in pool.slots}) == 2
        assert all(
            slot.completions > 0 and slot.graph is not None and slot.graph.replays > 0
            for slot in pool.slots
        )
        processors = [
            module.processor
            for slot in pool.slots
            for module in slot.flow.modules()
            if isinstance(module, Attention)
            and isinstance(module.processor, LiveBiasBroadcastProcessor)
        ]
        assert processors and all(
            processor.last_head_stride == 0 and processor.broadcast_calls > 0
            for processor in processors
        )
        executions = {
            "codec-async": sum(slot.completions for slot in pool.slots),
            "codec-cfm-graph": sum(slot.graph.replays for slot in pool.slots),
            "codec-bias-broadcast": sum(
                processor.broadcast_calls for processor in processors
            ),
            "codec-unready-return": 1,
        }
    finally:
        executor.stop()
        independent.close()
    assert pickle.dumps(random.getstate()) == before[0]
    assert pickle.dumps(np.random.get_state()) == before[1]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert torch.equal(torch.cuda.get_rng_state(), before[3])
    return AsyncCodecEvidence(
        artifacts=artifacts, executions=executions, flow_slots=2, pcm_exact=True
    )
