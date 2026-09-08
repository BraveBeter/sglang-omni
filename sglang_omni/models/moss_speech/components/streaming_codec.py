# SPDX-License-Identifier: Apache-2.0
"""Chunk-trained flow with the locked CosyVoice2 prefix/HiFT overlap protocol.

This is a separate codec profile. The P1 offline AudioDecoder stays unchanged.
The flow recomputes only the available prefix with causal chunk masks; it never
requires future AR codes. HiFT overlap and RNG state belong to each request.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_flow_weights(module: Any, state: dict[str, Any]) -> None:
    """Discard only checkpoint training counters; enforce every inference key."""
    module.load_state_dict(
        {k: v for k, v in state.items() if k not in {"epoch", "step"}}, strict=True
    )


@dataclass
class CodecStreamSession:
    voice: dict[str, torch.Tensor]
    seed: int
    codes: list[int] = field(default_factory=list)
    cursor: int = 0
    emitted_samples: int = 0
    cache: dict[str, torch.Tensor] | None = None
    cpu_rng: torch.Tensor | None = None
    device_rng: torch.Tensor | None = None
    ledger: list[dict[str, Any]] = field(default_factory=list)


class MossSpeechStreamingCodec:
    """Serial codec calls with independently owned streaming sessions."""

    sample_rate = 24000
    samples_per_code = 1920
    mel_cache_len = 8
    source_cache_len = 8 * 480

    def __init__(
        self,
        codec_path: str | Path | None,
        *,
        chunk_size: int = 5,
        device: str = "cuda",
        flow: Any = None,
        hift: Any = None,
    ) -> None:
        if chunk_size not in (5, 25):
            raise ValueError("Streaming chunk size must match trained profile 5 or 25")
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.lock = threading.RLock()
        self.sessions: dict[str, CodecStreamSession] = {}
        self.speech_window = np.hamming(2 * self.source_cache_len)
        if flow is None or hift is None:
            from .hf_codec.modeling import (
                _build_flow_decoder,
                _build_hift,
                _scoped_global_rng,
            )

            if codec_path is None:
                raise ValueError("codec_path is required when loading weights")
            root = Path(codec_path) / "flow"
            with _scoped_global_rng():
                flow = _build_flow_decoder(chunk_size=chunk_size)
                load_flow_weights(
                    flow,
                    torch.load(
                        root / f"flow-chunk-{chunk_size}.pt",
                        map_location="cpu",
                        weights_only=True,
                    ),
                )
                hift = _build_hift()
                hift.load_state_dict(
                    torch.load(root / "hift.pt", map_location="cpu", weights_only=True),
                    strict=True,
                )
            flow, hift = flow.to(self.device).eval(), hift.to(self.device).eval()
        self.flow, self.hift = flow, hift

    def begin(
        self, request_id: str, voice: dict[str, torch.Tensor], *, seed: int
    ) -> None:
        with self.lock:
            if request_id in self.sessions:
                raise ValueError(f"Codec session is already active: {request_id}")
            tokens = voice["prompt_token"]
            feat = voice["prompt_feat"]
            if (
                tokens.ndim != 2
                or tokens.shape[0] != 1
                or feat.shape != (1, tokens.shape[1] * 4, 80)
            ):
                raise ValueError("Voice token/mel shape mismatch")
            if voice["embedding"].shape != (1, 192):
                raise ValueError("Voice embedding shape mismatch")
            state = CodecStreamSession(
                {k: v.to(self.device) for k, v in voice.items()}, seed
            )
            state.cpu_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
            if self.device.type == "cuda":
                state.device_rng = (
                    torch.Generator(device=self.device).manual_seed(seed).get_state()
                )
            self.sessions[request_id] = state

    def next_hop(self, state: CodecStreamSession) -> int:
        if state.cursor:
            return self.chunk_size
        return (
            self.chunk_size + (-state.voice["prompt_token"].shape[1]) % self.chunk_size
        )

    @torch.inference_mode()
    def push(
        self, request_id: str, codes: list[int], *, final: bool = False
    ) -> list[torch.Tensor]:
        with self.lock:
            state = self.sessions.get(request_id)
            if state is None:
                raise ValueError(f"Unknown codec session: {request_id}")
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            try:
                if any(
                    isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < 16384
                    for c in codes
                ):
                    raise ValueError("Invalid streaming audio code")
                state.codes.extend(codes)
                if len(state.codes) > 512:
                    raise ValueError("Streaming audio exceeds 512-code budget")
                if final and not state.codes:
                    raise ValueError("Cannot finalize empty audio stream")
                waves = []
                with (
                    torch.random.fork_rng(devices=devices),
                    torch.backends.cudnn.flags(
                        enabled=torch.backends.cudnn.enabled,
                        benchmark=False,
                        deterministic=True,
                        allow_tf32=torch.backends.cudnn.allow_tf32,
                    ),
                ):
                    torch.set_rng_state(state.cpu_rng)
                    if devices:
                        torch.cuda.set_rng_state(state.device_rng, self.device)
                    try:
                        while (
                            len(state.codes) - state.cursor
                            >= self.next_hop(state) + self.flow.pre_lookahead_len
                        ):
                            hop = self.next_hop(state)
                            available = state.cursor + hop + self.flow.pre_lookahead_len
                            waves.append(self._decode(state, available, final=False))
                            state.cursor += hop
                        if final:
                            waves.append(
                                self._decode(state, len(state.codes), final=True)
                            )
                            state.cursor = len(state.codes)
                            if (
                                state.emitted_samples
                                != len(state.codes) * self.samples_per_code
                            ):
                                raise RuntimeError(
                                    "Streaming sample ledger does not match code length"
                                )
                    finally:
                        state.cpu_rng = torch.get_rng_state()
                        if devices:
                            state.device_rng = torch.cuda.get_rng_state(self.device)
                if final:
                    self.cleanup(request_id)
                return waves
            except BaseException:
                self.cleanup(request_id)
                raise

    def _decode(
        self, state: CodecStreamSession, available: int, *, final: bool
    ) -> torch.Tensor:
        token = torch.tensor(
            [state.codes[:available]], device=self.device, dtype=torch.long
        )
        voice = state.voice
        mel, _ = self.flow.inference(
            token=token,
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
        mel = mel[:, :, state.cursor * self.flow.token_mel_ratio :]
        cache = state.cache
        source = torch.zeros(1, 1, 0)
        if cache is not None:
            mel = torch.cat([cache["mel"], mel], dim=2)
            source = cache["source"]
        wave, generated_source = self.hift.inference(
            speech_feat=mel, cache_source=source
        )
        if cache is not None:
            # Preserve upstream float64 NumPy window arithmetic and FP32 store.
            device = wave.device
            wave, previous = wave.cpu(), cache["speech"].cpu()
            n = self.source_cache_len
            wave[..., :n] = (
                wave[..., :n] * self.speech_window[:n]
                + previous[..., -n:] * self.speech_window[n:]
            )
            wave = wave.to(device)
        if not final:
            state.cache = {
                "mel": mel[:, :, -self.mel_cache_len :],
                "source": generated_source[:, :, -self.source_cache_len :],
                "speech": wave[:, -self.source_cache_len :],
            }
            wave = wave[:, : -self.source_cache_len]
        wave = wave.detach().cpu().reshape(-1)
        if not wave.numel() or not torch.isfinite(wave).all():
            raise RuntimeError("Streaming codec emitted empty or non-finite audio")
        state.emitted_samples += wave.numel()
        state.ledger.append(
            {
                "available": available,
                "cursor_before": state.cursor,
                "samples": wave.numel(),
                "final": final,
            }
        )
        return wave

    def cleanup(self, request_id: str) -> None:
        with self.lock:
            self.sessions.pop(request_id, None)

    def close(self) -> None:
        with self.lock:
            self.sessions.clear()
