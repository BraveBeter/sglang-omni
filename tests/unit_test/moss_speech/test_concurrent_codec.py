# SPDX-License-Identifier: Apache-2.0
"""B1 Flow concurrency must preserve request audio and drain before shutdown."""

import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_speech.codec import MossSpeechCodec, VoiceConditioning
from sglang_omni.models.moss_speech.concurrent_codec import MossSpeechConcurrentCodec
from sglang_omni.models.moss_speech.request_builders import MossSpeechState
from sglang_omni.models.moss_speech.stages import create_audio_vocoder_executor
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fixtures.moss_speech import audio_reference


class RequestsDrained(Protocol):
    def __call__(self) -> bool: ...


@pytest.mark.parametrize("flow_slots", [1, 2])
def test_factory_routes_flow_slots_to_codec_and_scheduler(flow_slots: int) -> None:
    """Enabling two Flow slots must enable two compute workers on that codec."""
    with (
        patch("sglang_omni.models.moss_speech.stages.MossSpeechCodec") as serial,
        patch(
            "sglang_omni.models.moss_speech.stages.MossSpeechConcurrentCodec"
        ) as dual,
        patch("sglang_omni.models.moss_speech.stages.AutoTokenizer"),
        patch(
            "sglang_omni.models.moss_speech.stages.resolve_concrete_device",
            return_value=torch.device("cuda:0"),
        ) as resolve,
    ):
        scheduler = create_audio_vocoder_executor(
            "model", codec_path="codec", device="cuda", gpu_id=0, flow_slots=flow_slots
        )
        selected = dual if flow_slots == 2 else serial
        resolve.assert_called_once_with("cuda", 0)
        selected.assert_called_once_with("codec", torch.device("cuda:0"))
        assert scheduler.max_concurrency == flow_slots
        assert scheduler.fn.func == selected.return_value.decode
        scheduler.stop()
        selected.return_value.close.assert_called_once_with()


@pytest.mark.accelerator
def test_dual_flow_preserves_pcm_rng_and_waits_for_active_requests() -> None:
    """Concurrent requests must equal serial PCM, recover a failed slot and drain."""
    require_cuda()
    codec_path = os.environ.get("MOSS_SPEECH_CODEC_PATH")
    if codec_path is None:
        pytest.skip("Set MOSS_SPEECH_CODEC_PATH to the frozen Codec checkpoint")
    device = torch.device("cuda:0")
    serial = MossSpeechCodec(codec_path, device)
    concurrent = MossSpeechConcurrentCodec(codec_path, device)
    requests = [
        MossSpeechState(
            reference_audio=audio_reference(
                frequency_hz, sample_count=reference_samples
            ),
            audio_codes=list(range(3, 3 + code_count)),
            effective_seed=seed,
        )
        for frequency_hz, code_count, seed, reference_samples in [
            (220, 24, 7, 27 * 24000),
            (880, 65, 19, 24000),
        ]
    ]
    cpu_rng = torch.get_rng_state()
    device_rng = torch.cuda.get_rng_state(device)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    expected = [
        serial.decode_waveform(state, str(i)) for i, state in enumerate(requests)
    ]
    try:
        assert concurrent.decoder is not None and serial.decoder is not None
        assert concurrent.slots[0].flow is not concurrent.slots[1].flow
        assert len({slot.stream.cuda_stream for slot in concurrent.slots}) == 2
        owner_threads: set[int] = set()
        reference = concurrent.encode_reference
        hift = concurrent.decoder.hift.inference

        def owned_reference(reference_audio: str) -> VoiceConditioning:
            owner_threads.add(threading.get_ident())
            return reference(reference_audio)

        def owned_hift(
            *, speech_feat: torch.Tensor, cache_source: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            owner_threads.add(threading.get_ident())
            return hift(speech_feat=speech_feat, cache_source=cache_source)

        with (
            ThreadPoolExecutor(max_workers=2) as executor,
            patch.object(concurrent, "encode_reference", side_effect=owned_reference),
            patch.object(concurrent.decoder.hift, "inference", side_effect=owned_hift),
        ):
            for _ in range(2):
                outputs = list(
                    executor.map(concurrent.decode_waveform, requests, ["0", "1"])
                )
                assert all(
                    torch.equal(left, right) for left, right in zip(expected, outputs)
                )
            with (
                patch.object(
                    concurrent.slots[0].flow,
                    "inference",
                    side_effect=RuntimeError("Flow failed"),
                ),
                patch.object(
                    concurrent.slots[1].flow,
                    "inference",
                    side_effect=RuntimeError("Flow failed"),
                ),
            ):
                with pytest.raises(RuntimeError, match="Flow failed"):
                    concurrent.decode_waveform(requests[0], "failed")
            assert concurrent.available_slots.qsize() == 2
            assert torch.equal(
                concurrent.decode_waveform(requests[0], "reused"), expected[0]
            )
            entered = threading.Event()
            release = threading.Event()
            draining = threading.Event()
            original = concurrent.encode_reference
            wait_for = concurrent.lifecycle.wait_for

            def paused_reference(reference_audio: str) -> VoiceConditioning:
                entered.set()
                assert release.wait(timeout=30)
                return original(reference_audio)

            def wait_for_requests(predicate: RequestsDrained) -> bool:
                draining.set()
                return wait_for(predicate)

            with (
                patch.object(
                    concurrent, "encode_reference", side_effect=paused_reference
                ),
                patch.object(
                    concurrent.lifecycle, "wait_for", side_effect=wait_for_requests
                ),
            ):
                active = executor.submit(
                    concurrent.decode_waveform, requests[1], "closing"
                )
                assert entered.wait(timeout=30)
                closing = executor.submit(concurrent.close)
                assert draining.wait(timeout=30)
                assert not closing.done()
                release.set()
                assert torch.equal(active.result(timeout=60), expected[1])
                closing.result(timeout=60)
        assert len(owner_threads) == 1
        assert concurrent.decoder is None and concurrent.encoder is None
        with pytest.raises(RuntimeError, match="closed"):
            concurrent.decode_waveform(requests[0], "late")
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(device), device_rng)
        assert random.getstate() == python_rng
        current_numpy_rng = np.random.get_state()
        assert np.array_equal(current_numpy_rng[1], numpy_rng[1])
        assert (
            current_numpy_rng[0] == numpy_rng[0]
            and current_numpy_rng[2:] == numpy_rng[2:]
        )
    finally:
        concurrent.close()
        serial.close()
