# SPDX-License-Identifier: Apache-2.0
"""Two private B1 Flow streams with serialized conditioning and HiFT."""

import copy
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue

import numpy as np
import torch

from sglang_omni.models.moss_speech.codec import (
    FlowDecoder,
    MossSpeechCodec,
    VoiceConditioning,
    codec_rng,
)
from sglang_omni.models.moss_speech.request_builders import MossSpeechState


@dataclass(kw_only=True)
class FlowSlot:
    flow: FlowDecoder
    stream: torch.cuda.Stream


@dataclass(kw_only=True)
class PreparedAudio:
    voice: VoiceConditioning
    token: torch.Tensor
    ready: torch.cuda.Event


class MossSpeechConcurrentCodec(MossSpeechCodec):
    def __init__(self, codec_path: str, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("Two Flow slots require a CUDA device")
        else:
            pass
        super().__init__(codec_path, device)
        assert self.decoder is not None
        self.owner = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="moss-codec-owner"
        )
        self.lifecycle = threading.Condition()
        self.active_requests: int = 0
        self.closed: bool = False
        self.owner_stream = torch.cuda.Stream(device=device)
        self.slots = [
            FlowSlot(flow=flow, stream=torch.cuda.Stream(device=device))
            for flow in (self.decoder.flow, copy.deepcopy(self.decoder.flow))
        ]
        torch.cuda.current_stream(device).synchronize()
        self.available_slots: Queue[FlowSlot] = Queue(maxsize=len(self.slots))
        for slot in self.slots:
            self.available_slots.put(slot)
        self.cudnn_deterministic: bool = torch.backends.cudnn.deterministic
        self.cudnn_benchmark: bool = torch.backends.cudnn.benchmark
        # note (BraveBeter): cuDNN flags are process-wide and cannot be scoped by concurrent requests.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @torch.inference_mode()
    def prepare_audio(self, state: MossSpeechState) -> PreparedAudio:
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self.owner_stream),
            codec_rng(),
        ):
            voice = self.encode_reference(state.reference_audio)
            token = torch.tensor(
                [state.audio_codes], dtype=torch.long, device=self.device
            )
            ready = torch.cuda.Event()
            ready.record()
            return PreparedAudio(voice=voice, token=token, ready=ready)

    @torch.inference_mode()
    def synthesize_audio(
        self, state: MossSpeechState, mel: torch.Tensor, complete: torch.cuda.Event
    ) -> torch.Tensor:
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self.owner_stream),
            codec_rng(),
        ):
            assert self.decoder is not None
            self.owner_stream.wait_event(complete)
            mel.record_stream(self.owner_stream)
            random.seed(state.effective_seed)
            np.random.seed(state.effective_seed % (1 << 32))
            torch.manual_seed(state.effective_seed)
            waveform, _ = self.decoder.hift.inference(
                speech_feat=mel, cache_source=torch.zeros(1, 1, 0)
            )
            return waveform.detach().cpu().reshape(-1)

    @torch.inference_mode()
    def decode_waveform(self, state: MossSpeechState, request_id: str) -> torch.Tensor:
        with self.lifecycle:
            if self.closed:
                raise RuntimeError("Codec is closed")
            else:
                self.active_requests += 1
        slot = self.available_slots.get()
        try:
            with torch.cuda.device(self.device):
                prepared = self.owner.submit(self.prepare_audio, state).result()
                voice, token = prepared.voice, prepared.token
                with torch.cuda.stream(slot.stream):
                    slot.stream.wait_event(prepared.ready)
                    mel, _ = slot.flow.inference(
                        token=token,
                        token_len=torch.tensor(
                            [token.shape[1]], dtype=torch.int32, device=self.device
                        ),
                        prompt_token=voice["prompt_token"],
                        prompt_token_len=torch.tensor(
                            [voice["prompt_token"].shape[1]],
                            dtype=torch.int32,
                            device=self.device,
                        ),
                        prompt_feat=voice["prompt_feat"],
                        prompt_feat_len=torch.tensor(
                            [voice["prompt_feat"].shape[1]],
                            dtype=torch.int32,
                            device=self.device,
                        ),
                        embedding=voice["embedding"],
                        streaming=False,
                        finalize=True,
                    )
                    complete = torch.cuda.Event()
                    complete.record()
                return self.owner.submit(
                    self.synthesize_audio, state, mel, complete
                ).result()
        finally:
            try:
                slot.stream.synchronize()
            finally:
                self.available_slots.put(slot)
                with self.lifecycle:
                    self.active_requests -= 1
                    self.lifecycle.notify_all()

    def close(self) -> None:
        with self.lifecycle:
            if self.closed:
                return
            else:
                self.closed = True
            self.lifecycle.wait_for(lambda: self.active_requests == 0)
        self.owner.shutdown(wait=True)
        self.owner_stream.synchronize()
        self.slots.clear()
        while not self.available_slots.empty():
            self.available_slots.get_nowait()
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        super().close()
