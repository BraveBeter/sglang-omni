# SPDX-License-Identifier: Apache-2.0
"""Chunk-trained flow with the locked CosyVoice2 prefix/HiFT overlap protocol.

This profile retains its own chunk-trained flow weights; offline uses flow.pt.
The flow recomputes only the available prefix with causal chunk masks; it never
requires future AR codes. HiFT overlap and RNG state belong to each request.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def validate_startup_schedule(
    policy: str | None, chunk_size: int, small_first_chunk: bool
) -> None:
    """Admit one explicit output-changing policy before loading any assets."""
    if policy is not None and (type(policy) is not str or policy != "6_8_align25"):
        raise ValueError("Unsupported stream startup schedule")
    if policy is not None and chunk_size != 25:
        raise ValueError("Stream startup schedule requires the chunk25 profile")
    if policy is not None and small_first_chunk:
        raise ValueError("Stream startup schedule conflicts with small_first_chunk")


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
    final_requested: bool = False
    prefix_voice_digest: str | None = None


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
        codec_optimizations: bool = False,
        codec_memory_optimizations: bool = False,
        codec_final_bias_alignment: bool = False,
        codec_final_bias_elision: bool = False,
        codec_qkv_multistream: bool = False,
        small_first_chunk: bool = False,
        stream_startup_schedule: str | None = None,
    ) -> None:
        if type(codec_final_bias_elision) is not bool or (
            codec_final_bias_elision and not codec_final_bias_alignment
        ):
            raise ValueError(
                "Final bias elision requires explicit boolean and final bias alignment"
            )
        if codec_final_bias_alignment and (chunk_size != 25 or not codec_optimizations):
            raise ValueError(
                "Final bias alignment requires chunk25 and codec_optimizations"
            )
        self.codec_final_bias_alignment = codec_final_bias_alignment
        if type(codec_qkv_multistream) is not bool or (
            codec_qkv_multistream and (chunk_size != 25 or not codec_optimizations)
        ):
            raise ValueError(
                "QKV multistream requires explicit boolean and optimized chunk25"
            )
        self.codec_qkv_multistream = codec_qkv_multistream
        self._final_bias_alignment = None
        self._final_bias_alignment_logged = False
        if chunk_size not in (5, 25):
            raise ValueError("Streaming chunk size must match trained profile 5 or 25")
        if small_first_chunk and chunk_size != 25:
            raise ValueError("Small first chunk requires the chunk25 profile")
        validate_startup_schedule(
            stream_startup_schedule, chunk_size, small_first_chunk
        )
        self.stream_startup_schedule = stream_startup_schedule
        self.small_first_chunk = small_first_chunk
        self.codec_path = Path(codec_path) if codec_path is not None else None
        self._prefix_cache = None
        self._native_epoch = None
        self._native_epoch_logged = False
        self._prefix_cache_receipt: dict[str, Any] | None = None
        self._prepared_voice = None
        self._full_graphs = None
        self._full_graph_receipt: dict[str, Any] | None = None
        self.chunk_size = chunk_size
        self.codec_optimizations = codec_optimizations
        self.codec_memory_optimizations = codec_memory_optimizations
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
                flow = _build_flow_decoder(
                    chunk_size=chunk_size,
                    inference_optimizations=codec_optimizations,
                    **(
                        {"memory_optimizations": True}
                        if codec_memory_optimizations
                        else {}
                    ),
                )
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
        if codec_final_bias_alignment:
            from .cosyvoice_flow.decoder import CausalConditionalDecoder
            from .cosyvoice_flow.final_bias import FinalBiasAlignment

            estimator = getattr(getattr(flow, "decoder", None), "estimator", None)
            if type(estimator) is CausalConditionalDecoder:
                self._final_bias_alignment = FinalBiasAlignment(
                    **({"elide_zero_bias": True} if codec_final_bias_elision else {})
                )
                estimator._final_bias_alignment = self._final_bias_alignment

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
            digest = None
            if self._prefix_cache is not None:
                from .cfm_prefix_cache import cpu_voice_digest

                digest = cpu_voice_digest(voice)
            state = CodecStreamSession(
                {k: v.to(self.device) for k, v in voice.items()},
                seed,
                prefix_voice_digest=digest,
            )
            state.cpu_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
            if self.device.type == "cuda":
                state.device_rng = (
                    torch.Generator(device=self.device).manual_seed(seed).get_state()
                )
            self.sessions[request_id] = state

    def next_hop(self, state: CodecStreamSession, *, cursor: int | None = None) -> int:
        """Resolve a hop at the decoded cursor or an ingest-only lookahead cursor."""
        if cursor is None:
            cursor = state.cursor
        if self.stream_startup_schedule == "6_8_align25":
            if cursor == 0:
                return 6
            if cursor == 6:
                return 8
            if cursor == 14:
                # Rejoin the existing prompt-aligned chunk25 grid at hop three.
                return 25 + (-state.voice["prompt_token"].shape[1]) % 25 - 14
        if cursor:
            return self.chunk_size
        if self.small_first_chunk:
            hop = (-state.voice["prompt_token"].shape[1]) % self.chunk_size
            hop = hop or self.chunk_size
            if hop * self.samples_per_code <= self.source_cache_len:
                hop += self.chunk_size
            return hop
        return (
            self.chunk_size + (-state.voice["prompt_token"].shape[1]) % self.chunk_size
        )

    @torch.inference_mode()
    def push(
        self,
        request_id: str,
        codes: list[int],
        *,
        final: bool = False,
        max_hops: int | None = None,
    ) -> list[torch.Tensor]:
        with self.lock:
            state = self.sessions.get(request_id)
            if state is None:
                raise ValueError(f"Unknown codec session: {request_id}")
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            try:
                if max_hops is not None and (
                    isinstance(max_hops, bool)
                    or not isinstance(max_hops, int)
                    or max_hops <= 0
                ):
                    raise ValueError("max_hops must be a positive integer")
                if state.final_requested and codes:
                    raise ValueError("Cannot append codes after final was requested")
                if any(
                    isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < 16384
                    for c in codes
                ):
                    raise ValueError("Invalid streaming audio code")
                state.codes.extend(codes)
                if len(state.codes) > 512:
                    raise ValueError("Streaming audio exceeds 512-code budget")
                state.final_requested = state.final_requested or final
                if state.final_requested and not state.codes:
                    raise ValueError("Cannot finalize empty audio stream")
                if (
                    self.codec_optimizations
                    and not state.final_requested
                    and len(state.codes) - state.cursor
                    < self.next_hop(state) + self.flow.pre_lookahead_len
                ):
                    return []
                waves = []
                finalized = False
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
                        while len(state.codes) - state.cursor >= self.next_hop(
                            state
                        ) + self.flow.pre_lookahead_len and (
                            max_hops is None or len(waves) < max_hops
                        ):
                            hop = self.next_hop(state)
                            available = state.cursor + hop + self.flow.pre_lookahead_len
                            waves.append(self._decode(state, available, final=False))
                            state.cursor += hop
                        if state.final_requested and (
                            max_hops is None or len(waves) < max_hops
                        ):
                            waves.append(
                                self._decode(state, len(state.codes), final=True)
                            )
                            state.cursor = len(state.codes)
                            finalized = True
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
                if finalized:
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
        cache_scope = nullcontext()
        if self._prefix_cache is not None:
            from .cfm_prefix_cache import runtime_environment

            length = (
                voice["prompt_token"].shape[1]
                + available
                - (0 if final else self.flow.pre_lookahead_len)
            ) * self.flow.token_mel_ratio
            cache_scope = self._prefix_cache.scope(
                state.prefix_voice_digest,
                length,
                final,
                runtime_environment(self.device),
            )
        with cache_scope:
            mel, _ = self.flow.inference(
                token=token,
                token_len=torch.tensor(
                    [available], dtype=torch.int32, device=self.device
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
        if self.codec_optimizations:
            # The transport and the overlap window both consume host samples.
            # Keep the speech overlap here; mel/source remain on the device.
            wave = wave.cpu()
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

    def prefix_cache_status(self) -> dict[str, Any]:
        with self.lock:
            if self._prefix_cache is None:
                return {"enabled": False}
            return {
                **self._prefix_cache.status(),
                "startup": self._prefix_cache_receipt,
            }

    def prepare_prefix_cache(self, manifest_path: str | Path) -> dict[str, Any]:
        """Prepare and verify explicit conditioning before the service is ready."""
        from .cfm_prefix_cache import (
            PrefixCacheBank,
            load_prepared_voice,
            runtime_environment,
        )
        from .cosyvoice_flow.cfm_attention import runtime_supported

        with self.lock:
            if self.sessions:
                raise ValueError("Cannot prepare a prefix cache with active sessions")
            if self._prefix_cache is not None:
                raise ValueError("A prefix cache is already prepared")
            if self.chunk_size != 25 or not self.codec_optimizations:
                raise ValueError("Prefix cache requires optimized chunk25")
            if self.device.type != "cuda" or self.codec_path is None:
                raise ValueError("Prefix cache requires a qualified CUDA codec")
            environment = runtime_environment(self.device)
            if not runtime_supported(
                device_name=environment["device_name"],
                capability=tuple(environment["capability"]),
                torch_build=environment["torch"],
                cuda_build=environment["cuda"],
                dtype=torch.float32,
            ):
                raise ValueError(
                    "Prefix cache runtime has not passed strict validation"
                )
            started = time.perf_counter()
            prepared = load_prepared_voice(
                manifest_path,
                codec_root=self.codec_path,
                source_root=Path(__file__).resolve().parents[1],
                environment=environment,
            )
            receipt: dict[str, Any] = {
                "manifest_sha256": prepared.manifest_sha256,
                "voice_digest": prepared.voice_digest,
                "verified_files": prepared.verified_files,
                "source_sha256": prepared.source_sha256,
                "environment": environment,
                "outer_cudnn_deterministic": torch.backends.cudnn.deterministic,
                "outer_cudnn_benchmark": torch.backends.cudnn.benchmark,
            }
            torch.cuda.synchronize(self.device)
            initial_memory = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

            def calibration() -> tuple[Any, ...]:
                return self._calibrate_prepared_voice(prepared)

            tick = time.perf_counter()
            reference = calibration()
            receipt["native_codec_and_hift_warmup_seconds"] = time.perf_counter() - tick
            # The lazy fixed-noise copy originates under inference_mode. Give it
            # a version counter so later in-place changes invalidate the bank.
            with torch.inference_mode(False), torch.no_grad():
                for module in self.flow.modules():
                    for name in ("rand_noise", "_inference_noise"):
                        tensor = getattr(module, name, None)
                        if isinstance(tensor, torch.Tensor) and tensor.is_inference():
                            setattr(module, name, tensor.clone())
            shapes = (
                (1392, 1424, 1500)
                if self.stream_startup_schedule == "6_8_align25"
                else (1400, 1500) if self.small_first_chunk else (1500,)
            )
            bank = PrefixCacheBank(
                self.flow,
                prepared.voice_digest,
                shapes,
                environment,
            )
            self._prefix_cache = bank
            try:
                bank.mode = "build"
                tick = time.perf_counter()
                built = calibration()
                receipt["cache_build_codec_seconds"] = time.perf_counter() - tick
                if not self._same_calibration(reference, built) or len(
                    bank.entries
                ) != 560 * len(bank.binding.shapes):
                    raise RuntimeError(
                        "Prepared prefix build changed complete codec output or missed layers"
                    )
                bank.mode = "cached"
                tick = time.perf_counter()
                cached = calibration()
                receipt["cached_codec_validation_seconds"] = time.perf_counter() - tick
                if not self._same_calibration(
                    reference, cached
                ) or bank.hits != 560 * len(bank.binding.shapes):
                    raise RuntimeError(
                        "Prepared prefix reuse changed complete codec output or did not activate"
                    )
                receipt.update(
                    exact=True,
                    startup_attention_hits=bank.hits,
                    startup_ledger=reference[1],
                    initial_allocated_bytes=initial_memory,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    ready_allocated_bytes=torch.cuda.memory_allocated(self.device),
                    total_prepare_seconds=time.perf_counter() - started,
                )
                self._prefix_cache_receipt = receipt
                self._prepared_voice = prepared
                self._log_prefix_cache("ready")
                return self.prefix_cache_status()
            except BaseException:
                torch.cuda.synchronize(self.device)
                bank.close()
                self._prefix_cache = None
                self._prepared_voice = None
                raise

    def _calibrate_prepared_voice(self, prepared: Any) -> tuple[Any, ...]:
        """Run the original complete first hops, including HiFT and request RNG."""
        request_id = "__moss_prefix_startup__"
        self.begin(request_id, prepared.voice, seed=7)
        state = self.sessions[request_id]
        try:
            waves = self.push(request_id, list(prepared.warmup_audio_codes))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return (
                [wave.clone() for wave in waves],
                list(state.ledger),
                state.cpu_rng.clone(),
                state.device_rng.clone() if state.device_rng is not None else None,
            )
        finally:
            self.cleanup(request_id)

    @staticmethod
    def _same_calibration(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
        def same_tensor(a: Any, b: Any) -> bool:
            if a is None or b is None:
                return a is b
            return (
                a.shape == b.shape
                and a.dtype == b.dtype
                and torch.equal(
                    a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
                )
            )

        return (
            len(left[0]) == len(right[0])
            and all(same_tensor(a, b) for a, b in zip(left[0], right[0]))
            and left[1] == right[1]
            and same_tensor(left[2], right[2])
            and same_tensor(left[3], right[3])
        )

    def full_graph_status(self) -> dict[str, Any]:
        with self.lock:
            if self._full_graphs is None:
                return {"enabled": False}
            return {**self._full_graphs.status(), "startup": self._full_graph_receipt}

    @torch.inference_mode()
    def prepare_full_graphs(self) -> dict[str, Any]:
        """Prepare explicit complete CFM graphs and prove actual codec replay."""
        with self.lock:
            if self._full_graphs is not None:
                raise ValueError("Complete CFM graphs are already prepared")
            bank = self._prefix_cache
            if self.sessions or bank is None or self._prepared_voice is None:
                raise ValueError(
                    "Complete CFM graphs require an idle prepared prefix bank"
                )
            if not bank.binding.valid or bank.active_length is not None:
                raise ValueError("Complete CFM graphs require a valid idle prefix bank")
            if (
                self.chunk_size != 25
                or not self.codec_optimizations
                or self.device.type != "cuda"
            ):
                raise ValueError("Complete CFM graphs require optimized CUDA chunk25")
            from .cfm_full_graph import FullCfmGraphBank

            started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            initial_memory = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

            def calibration() -> tuple[Any, ...]:
                return self._calibrate_prepared_voice(self._prepared_voice)

            owner = None
            prefix_hits_before = bank.hits
            try:
                # Match push(), including the effective context at replay time.
                with torch.backends.cudnn.flags(
                    enabled=torch.backends.cudnn.enabled,
                    benchmark=False,
                    deterministic=True,
                    allow_tf32=torch.backends.cudnn.allow_tf32,
                ):
                    reference = calibration()
                    owner = FullCfmGraphBank(
                        self.flow.decoder,
                        bank,
                        **(
                            {"qkv_multistream": True}
                            if self.codec_qkv_multistream
                            else {}
                        ),
                    )
                    preparation = owner.prepare(calibration)
                    owner.install()
                    before = owner.status()["graph_replays"]
                    actual = calibration()
                    if not self._same_calibration(reference, actual):
                        raise RuntimeError(
                            "Complete CFM graphs changed complete codec output"
                        )
                    replay_count = owner.status()["graph_replays"] - before
                    if replay_count != len(bank.binding.shapes):
                        raise RuntimeError(
                            "Complete CFM startup missed actual graph replays"
                        )
                receipt = dict(
                    preparation=preparation,
                    exact=True,
                    startup_graph_replays=replay_count,
                    ready_graph_replays=owner.status()["graph_replays"],
                    startup_ledger=reference[1],
                    initial_allocated_bytes=initial_memory,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    ready_allocated_bytes=torch.cuda.memory_allocated(self.device),
                    total_prepare_seconds=time.perf_counter() - started,
                )
                self._full_graph_receipt = receipt
                self._full_graphs = owner
                self._log_full_graphs("ready")
                return self.full_graph_status()
            except BaseException:
                if owner is not None:
                    owner.close()
                self._full_graphs = None
                self._full_graph_receipt = None
                raise
            finally:
                if self._prefix_cache_receipt is not None:
                    self._prefix_cache_receipt["startup_attention_hits"] += (
                        bank.hits - prefix_hits_before
                    )

    def _log_full_graphs(self, event: str) -> None:
        if self._full_graphs is None:
            return
        summary = self._full_graphs.status()
        receipt = self._full_graph_receipt or {}
        summary["startup_graph_replays"] = receipt.get("startup_graph_replays", 0)
        summary["request_graph_replays"] = max(
            0,
            summary["graph_replays"] - receipt.get("ready_graph_replays", 0),
        )
        for key in (
            "total_prepare_seconds",
            "peak_allocated_bytes",
            "ready_allocated_bytes",
        ):
            if key in receipt:
                summary[key] = receipt[key]
        logging.getLogger(__name__).info(
            "MOSS complete CFM graphs %s: %s",
            event,
            json.dumps(summary, sort_keys=True),
        )

    def _log_prefix_cache(self, event: str) -> None:
        if self._prefix_cache is None:
            return
        receipt = self._prefix_cache_receipt or {}
        summary = self._prefix_cache.status()
        for key in (
            "voice_digest",
            "manifest_sha256",
            "startup_attention_hits",
            "total_prepare_seconds",
            "native_codec_and_hift_warmup_seconds",
            "cache_build_codec_seconds",
            "cached_codec_validation_seconds",
            "peak_allocated_bytes",
            "ready_allocated_bytes",
        ):
            if key in receipt:
                summary[key] = receipt[key]
        summary["request_attention_hits"] = max(
            0, summary["attention_hits"] - receipt.get("startup_attention_hits", 0)
        )
        logging.getLogger(__name__).info(
            "MOSS prefix cache %s: %s", event, json.dumps(summary, sort_keys=True)
        )

    def prepare_native_epoch(self, manifest_path: str | Path) -> dict[str, Any]:
        """Explicit worker-local activation after original prefix/graph preparation."""
        with self.lock:
            if self.sessions:
                raise ValueError("Cannot prepare native epoch with active sessions")
            if self._prefix_cache is None or self._prepared_voice is None:
                raise ValueError("Native epoch requires a prepared prefix cache")
            if self._native_epoch is not None:
                raise ValueError("Native epoch is already prepared")
            from .cfm_native_epoch import NativeEpochOwner
            from .cfm_prefix_cache import tensor_epoch

            owner = NativeEpochOwner(self.flow, manifest_path, tensor_epoch)
            try:
                owner.prepare()
                self._prefix_cache.install_epoch_owner(owner)
            except BaseException:
                owner.close()
                raise
            self._native_epoch = owner
            logging.getLogger(__name__).info(
                "MOSS native epoch ready: %s",
                json.dumps(owner.status(), sort_keys=True),
            )
            return owner.status()

    def native_epoch_status(self) -> dict[str, Any]:
        with self.lock:
            if self._native_epoch is None:
                return dict(enabled=False, reason="disabled or not prepared")
            return self._native_epoch.status()

    def final_bias_alignment_status(self) -> dict[str, Any]:
        with self.lock:
            if self._final_bias_alignment is None:
                return dict(enabled=False, reason="disabled or unsupported estimator")
            return self._final_bias_alignment.status()

    def close(self) -> None:
        with self.lock:
            self.sessions.clear()
            if self._full_graphs is not None:
                self._log_full_graphs("closed")
                self._full_graphs.close()
                self._full_graphs = None
            self._prepared_voice = None
            if self._prefix_cache is not None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self._log_prefix_cache("closed")
                self._prefix_cache.close()
                self._prefix_cache = None
            if (
                self._final_bias_alignment is not None
                and not self._final_bias_alignment_logged
            ):
                self._final_bias_alignment.close()
                logging.getLogger(__name__).info(
                    "MOSS final bias alignment closed: %s",
                    json.dumps(self._final_bias_alignment.status(), sort_keys=True),
                )
                self._final_bias_alignment_logged = True
            if self._native_epoch is not None and not self._native_epoch_logged:
                self._native_epoch.close()
                logging.getLogger(__name__).info(
                    "MOSS native epoch closed: %s",
                    json.dumps(self._native_epoch.status(), sort_keys=True),
                )
                self._native_epoch_logged = True
