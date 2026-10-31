# SPDX-License-Identifier: Apache-2.0
"""Chunk-trained eager flow decoding with request-owned RNG and HiFT overlap."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from sglang_omni.models.moss_speech.hf_config import MAX_CONTEXT_TOKENS
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow import (
    CausalMaskedDiffWithXvec,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.hifigan_generator import (
    HiFTGenerator,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import (
    build_flow_decoder,
    build_hift,
    scoped_global_rng,
)


class HiFTCache(TypedDict):
    mel: torch.Tensor
    source: torch.Tensor
    speech: torch.Tensor


class StreamLedgerEntry(TypedDict):
    available: int
    cursor_before: int
    samples: int
    final: bool


def load_flow_weights(
    module: CausalMaskedDiffWithXvec, state: dict[str, torch.Tensor | int]
) -> None:
    """Discard checkpoint training counters and enforce every inference key."""
    module.load_state_dict(
        {name: value for name, value in state.items() if name not in {"epoch", "step"}},
        strict=True,
    )


@dataclass(kw_only=True)
class CodecStreamSession:
    voice: dict[str, torch.Tensor]
    seed: int
    cpu_rng: torch.Tensor
    device_rng: torch.Tensor | None
    codes: list[int] = field(default_factory=list)
    cursor: int = 0
    emitted_samples: int = 0
    cache: HiFTCache | None = None
    ledger: list[StreamLedgerEntry] = field(default_factory=list)
    final_requested: bool = False


class MossSpeechStreamingCodec:
    """Serialize eager codec calls while keeping stream state independent."""

    sample_rate = 24000
    samples_per_code = 1920
    mel_cache_len = 8
    source_cache_len = 8 * 480

    def __init__(
        self,
        codec_path: str | Path | None,
        *,
        chunk_size: int,
        device: str | torch.device,
        flow: CausalMaskedDiffWithXvec | None = None,
        hift: HiFTGenerator | None = None,
    ) -> None:
        if chunk_size not in (5, 25):
            raise ValueError("Streaming chunk size must match trained profile 5 or 25")
        else:
            pass
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.lock = threading.RLock()
        self.sessions: dict[str, CodecStreamSession] = {}
        self.speech_window = np.hamming(2 * self.source_cache_len)
        if flow is None or hift is None:
            if codec_path is None:
                raise ValueError("codec_path is required when loading weights")
            else:
                pass
            flow_directory = Path(codec_path) / "flow"
            with scoped_global_rng():
                flow = build_flow_decoder(chunk_size=chunk_size)
                load_flow_weights(
                    flow,
                    torch.load(
                        flow_directory / f"flow-chunk-{chunk_size}.pt",
                        map_location="cpu",
                        weights_only=True,
                    ),
                )
                hift = build_hift()
                hift.load_state_dict(
                    torch.load(
                        flow_directory / "hift.pt",
                        map_location="cpu",
                        weights_only=True,
                    ),
                    strict=True,
                )
            flow = flow.to(self.device).eval()
            hift = hift.to(self.device).eval()
        else:
            pass
        self.flow = flow
        self.hift = hift

    def begin(
        self, request_id: str, voice: dict[str, torch.Tensor], *, seed: int
    ) -> None:
        with self.lock:
            if request_id in self.sessions:
                raise ValueError(f"Codec session is already active: {request_id}")
            else:
                pass
            prompt_token = voice["prompt_token"]
            prompt_mel = voice["prompt_feat"]
            if (
                prompt_token.ndim != 2
                or prompt_token.shape[0] != 1
                or prompt_mel.shape != (1, prompt_token.shape[1] * 4, 80)
            ):
                raise ValueError("Voice token/mel shape mismatch")
            else:
                pass
            if voice["embedding"].shape != (1, 192):
                raise ValueError("Voice embedding shape mismatch")
            else:
                pass
            cpu_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
            if self.device.type == "cuda":
                device_rng = (
                    torch.Generator(device=self.device).manual_seed(seed).get_state()
                )
            else:
                device_rng = None
            self.sessions[request_id] = CodecStreamSession(
                voice={name: value.to(self.device) for name, value in voice.items()},
                seed=seed,
                cpu_rng=cpu_rng,
                device_rng=device_rng,
            )

    def next_hop(
        self, session: CodecStreamSession, *, cursor: int | None = None
    ) -> int:
        """Align the first generated hop to the prompt's trained chunk grid."""
        if cursor is None:
            cursor = session.cursor
        else:
            pass
        if cursor:
            return self.chunk_size
        else:
            return (
                self.chunk_size
                + (-session.voice["prompt_token"].shape[1]) % self.chunk_size
            )

    @torch.inference_mode()
    def push(
        self, request_id: str, codes: list[int], *, final: bool = False
    ) -> list[torch.Tensor]:
        with self.lock:
            session = self.sessions.get(request_id)
            if session is None:
                raise ValueError(f"Unknown codec session: {request_id}")
            else:
                pass
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            try:
                if session.final_requested and codes:
                    raise ValueError("Cannot append codes after final was requested")
                else:
                    pass
                if any(
                    isinstance(code, bool)
                    or not isinstance(code, int)
                    or not 0 <= code < 16384
                    for code in codes
                ):
                    raise ValueError("Invalid streaming audio code")
                else:
                    pass
                session.codes.extend(codes)
                if len(session.codes) > MAX_CONTEXT_TOKENS:
                    raise ValueError(
                        f"Streaming audio exceeds {MAX_CONTEXT_TOKENS}-code budget"
                    )
                else:
                    pass
                session.final_requested = session.final_requested or final
                if session.final_requested and not session.codes:
                    raise ValueError("Cannot finalize empty audio stream")
                else:
                    pass
                waveforms: list[torch.Tensor] = []
                with (
                    torch.random.fork_rng(devices=devices),
                    torch.backends.cudnn.flags(
                        enabled=torch.backends.cudnn.enabled,
                        benchmark=False,
                        deterministic=True,
                        allow_tf32=torch.backends.cudnn.allow_tf32,
                    ),
                ):
                    torch.set_rng_state(session.cpu_rng)
                    if devices:
                        torch.cuda.set_rng_state(session.device_rng, self.device)
                    else:
                        pass
                    try:
                        while (
                            len(session.codes) - session.cursor
                            >= self.next_hop(session) + self.flow.pre_lookahead_len
                        ):
                            hop = self.next_hop(session)
                            available = (
                                session.cursor + hop + self.flow.pre_lookahead_len
                            )
                            waveforms.append(
                                self.decode(session, available, final=False)
                            )
                            session.cursor += hop
                        if session.final_requested:
                            waveforms.append(
                                self.decode(session, len(session.codes), final=True)
                            )
                            session.cursor = len(session.codes)
                            if (
                                session.emitted_samples
                                != len(session.codes) * self.samples_per_code
                            ):
                                raise RuntimeError(
                                    "Streaming sample ledger does not match code length"
                                )
                            else:
                                pass
                        else:
                            pass
                    finally:
                        session.cpu_rng = torch.get_rng_state()
                        if devices:
                            session.device_rng = torch.cuda.get_rng_state(self.device)
                        else:
                            pass
                if session.final_requested:
                    self.cleanup(request_id)
                else:
                    pass
                return waveforms
            except BaseException:
                self.cleanup(request_id)
                raise

    def decode(
        self, session: CodecStreamSession, available: int, *, final: bool
    ) -> torch.Tensor:
        audio_token = torch.tensor(
            [session.codes[:available]], device=self.device, dtype=torch.long
        )
        voice = session.voice
        predicted_mel, _ = self.flow.inference(
            token=audio_token,
            token_len=torch.tensor([available], dtype=torch.int32, device=self.device),
            prompt_token=voice["prompt_token"],
            prompt_token_len=torch.tensor(
                [voice["prompt_token"].shape[1]], dtype=torch.int32, device=self.device
            ),
            prompt_feat=voice["prompt_feat"],
            prompt_feat_len=torch.tensor(
                [voice["prompt_feat"].shape[1]], dtype=torch.int32, device=self.device
            ),
            embedding=voice["embedding"],
            streaming=not final,
            finalize=final,
        )
        predicted_mel = predicted_mel[
            :, :, session.cursor * self.flow.token_mel_ratio :
        ]
        overlap = session.cache
        source = torch.zeros(1, 1, 0)
        if overlap is not None:
            predicted_mel = torch.cat([overlap["mel"], predicted_mel], dim=2)
            source = overlap["source"]
        else:
            pass
        waveform, generated_source = self.hift.inference(
            speech_feat=predicted_mel, cache_source=source
        )
        if overlap is not None:
            waveform_device = waveform.device
            waveform, previous_waveform = waveform.cpu(), overlap["speech"].cpu()
            overlap_samples = self.source_cache_len
            # note (MOSS-Speech): Preserve float64 window arithmetic and FP32 assignment.
            waveform[..., :overlap_samples] = (
                waveform[..., :overlap_samples] * self.speech_window[:overlap_samples]
                + previous_waveform[..., -overlap_samples:]
                * self.speech_window[overlap_samples:]
            )
            waveform = waveform.to(waveform_device)
        else:
            pass
        if not final:
            session.cache = {
                "mel": predicted_mel[:, :, -self.mel_cache_len :],
                "source": generated_source[:, :, -self.source_cache_len :],
                "speech": waveform[:, -self.source_cache_len :],
            }
            waveform = waveform[:, : -self.source_cache_len]
        else:
            pass
        waveform = waveform.detach().cpu().reshape(-1)
        if not waveform.numel() or not torch.isfinite(waveform).all():
            raise RuntimeError("Streaming codec emitted empty or non-finite audio")
        else:
            pass
        session.emitted_samples += waveform.numel()
        session.ledger.append(
            {
                "available": available,
                "cursor_before": session.cursor,
                "samples": waveform.numel(),
                "final": final,
            }
        )
        return waveform

    def cleanup(self, request_id: str) -> None:
        with self.lock:
            self.sessions.pop(request_id, None)

    def close(self) -> None:
        with self.lock:
            self.sessions.clear()
