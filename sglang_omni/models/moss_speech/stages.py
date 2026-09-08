# SPDX-License: Apache-2.0
"""Stage factories for the MOSS-Speech pipeline (P2/T2.4).

- preprocessing: Layout A (colocated validate + lower + codec encode) on the
  P1 encoder-only adapter; default voice precomputed once at factory time
  and transported in request payloads (no startup-queue handoff, no waiting
  inside the gpu_startup_lock).
- ar_engine: native SGLang dual-channel model with 40-layer paged KV.
- text_decode: reference text-channel decode rules (skip special tokens +
  empty/end_empty replacements).
- audio_vocoder: P1 decoder-only adapter, serial execution with a
  request-scoped RNG scope (save/set/restore around the HiFT-consuming
  decode), per-request cleanup, device-local voice cache keyed by content.

Codec/voice paths: ``codec_path`` defaults to the sibling
``<model_path>-Codec`` directory; ``voice_wav`` must be provided (YAML
stage_overrides.factory_args) — V1 has one config-level default voice.
"""

from __future__ import annotations

import os
import random
from typing import Any, List, Optional

import numpy as np
import torch

from sglang_omni.models.moss_speech.components.codec_adapter import (
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.processor import MossSpeechGridProcessor
from sglang_omni.models.moss_speech.payload_types import (
    EOSP_TOKEN_ID,
    MODALITY_PAD_TOKEN_ID,
    MossSpeechState,
)
from sglang_omni.models.moss_speech.request_builders import (
    RequestValidationError,
    cleanup_preprocessing_state,
    cleanup_vocoder_state,
    normalize_and_validate,
    pop_decoded_audio,
    remember_prepared,
    remember_vocoder_session,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

DEFAULT_CONTEXT_LIMIT = 10240  # SFT context bound (P0 contract §2)


def _resolve_codec_dir(model_path: str, codec_path: Optional[str]) -> str:
    if codec_path:
        if not os.path.isdir(codec_path):
            raise FileNotFoundError(f"codec_path {codec_path!r} not found")
        return codec_path
    sibling = f"{model_path.rstrip('/')}-Codec"
    if os.path.isdir(sibling):
        return sibling
    raise FileNotFoundError(
        f"codec directory not found: pass codec_path or materialize {sibling!r}"
    )


def _load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    from sglang_omni.models.moss_speech.hf_config import (
        ensure_moss_speech_config_registered,
    )

    ensure_moss_speech_config_registered()
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


def _state_from_payload(payload: StagePayload) -> MossSpeechState:
    data = payload.data
    if isinstance(data, MossSpeechState):
        return data
    if isinstance(data, dict):
        return MossSpeechState.from_dict(data)
    raise TypeError(f"unsupported stage data {type(data)!r}")


# ---------------------------------------------------------------- preprocessing
def create_preprocessing_executor(
    model_path: str,
    *,
    codec_path: Optional[str] = None,
    voice_wav: Optional[str] = None,
    encode_batch_size: int = 4,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> SimpleScheduler:
    codec_dir = _resolve_codec_dir(model_path, codec_path)
    voice_wav = voice_wav or os.environ.get("MOSS_SPEECH_VOICE_WAV")
    if not voice_wav or not os.path.isfile(voice_wav):
        raise FileNotFoundError(
            "default voice asset required for V1 audio output: pass voice_wav or set the "
            "MOSS_SPEECH_VOICE_WAV env default (PipelineConfig env_defaults in the YAML)"
        )
    adapter = MossSpeechCodecAdapter(codec_dir, load_encoder=True, load_decoder=False)
    tokenizer = _load_tokenizer(model_path)
    processor = MossSpeechGridProcessor(tokenizer)
    # default voice precomputed exactly once, on the encoder side (P1 chain)
    voice = adapter.encode_voice_ref(voice_wav)
    voice_key = (
        voice.meta.get("revision", "voice") + ":" + str(voice.prompt_token.shape[1])
    )

    def compute(payload: StagePayload) -> StagePayload:
        state = normalize_and_validate(payload.request, request_id=payload.request_id)
        decoded = pop_decoded_audio(state)
        codes: List[List[int]] = []
        if decoded:
            codes = adapter.encode(
                [(wav, sr) for wav, sr, _ in decoded], batch_size=encode_batch_size
            )
        state.audio_codes = codes
        built = processor.build(state.turns, codes, state.output_modality)
        if built.prompt_len > context_limit:
            raise RequestValidationError(
                f"context length {built.prompt_len} exceeds the {context_limit} limit "
                "(post-encode exact check)",
                reason="context_too_long",
            )
        state.input_grid = built.grid[0].tolist()
        state.attention_mask = built.attention_mask[0].tolist()
        state.prompt_grid_len = built.prompt_len
        if state.output_modality == "audio":
            state.voice_key = voice_key
            state.voice_token_ids = voice.prompt_token[0].to(torch.int32).tolist()
            state.voice_feat = voice.prompt_feat[0].tolist()
            state.voice_embedding = voice.embedding[0].tolist()
        remember_prepared(payload.request_id, state)
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=state.to_dict()
        )

    return SimpleScheduler(compute, abort_callback=cleanup_preprocessing_state)


# -------------------------------------------------------------------- native AR
def validate_ar_preconditions(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    codec_path: Optional[str] = None,
) -> Any:
    """CPU-only checkpoint/config validation, independent of model loading."""
    from transformers import AutoConfig

    from sglang_omni.models.moss_speech.hf_config import (
        MossSpeechConfig,
        ensure_moss_speech_config_registered,
    )

    if dtype not in ("bfloat16", "bf16"):
        raise ValueError("MOSS-Speech V1 requires BF16 AR weights")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path {model_path!r} not found")
    _resolve_codec_dir(model_path, codec_path)
    ensure_moss_speech_config_registered()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    if not isinstance(config, MossSpeechConfig):
        raise ValueError(f"Expected MossSpeechConfig, got {type(config).__name__}")
    return config


def create_ar_engine_executor(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    codec_path: Optional[str] = None,
    gpu_id: int | None = None,
    context_length: int = DEFAULT_CONTEXT_LIMIT,
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    """Build the validated native SGLang scheduler used by the formal YAML."""
    validate_ar_preconditions(model_path, dtype=dtype, codec_path=codec_path)
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    if not 1 <= context_length <= DEFAULT_CONTEXT_LIMIT:
        raise ValueError(
            f"V1 context_length must be within [1, {DEFAULT_CONTEXT_LIMIT}]"
        )
    overrides = dict(server_args_overrides or {})
    overrides["trust_remote_code"] = False
    return MossSpeechEngineBuilder(context_length=context_length).build(
        model_path,
        gpu_id=0 if gpu_id is None else gpu_id,
        dtype=dtype,
        server_args_overrides=overrides,
    )


# ----------------------------------------------------------------- text_decode
def create_text_decode_executor(model_path: str) -> SimpleScheduler:
    tokenizer = _load_tokenizer(model_path)

    def compute(payload: StagePayload) -> StagePayload:
        state = _state_from_payload(payload)
        grid = torch.tensor(state.output_grid or [], dtype=torch.long)
        if grid.numel() == 0:
            raise RequestValidationError("text_decode received an empty output grid")
        text_channel = grid[:, 0]
        text = tokenizer.decode(text_channel.tolist(), skip_special_tokens=True)
        # reference processor decode replacements (P0 contract §4)
        text = text.replace("<|empty|>", ".").replace("<|end_empty|>", ":")
        state.generated_text = text
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=state.to_dict()
        )

    return SimpleScheduler(compute)


def extract_output_codes(state: MossSpeechState) -> List[int]:
    """Extract the first audio segment; ignored text-mode audio cannot stop it."""
    codes: List[int] = []
    for text_token, audio_token in state.output_grid or []:
        if int(text_token) != MODALITY_PAD_TOKEN_ID:
            continue
        token = int(audio_token)
        if token == EOSP_TOKEN_ID:
            break
        if not 0 <= token < EOSP_TOKEN_ID:
            raise RequestValidationError(f"Invalid generated audio code {token}")
        codes.append(token)
    return codes


# --------------------------------------------------------------- audio_vocoder
def create_audio_vocoder_executor(
    model_path: str,
    *,
    codec_path: Optional[str] = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    codec_dir = _resolve_codec_dir(model_path, codec_path)
    adapter = MossSpeechCodecAdapter(
        codec_dir,
        load_encoder=False,
        load_decoder=True,
        device=f"cuda:{gpu_id}" if gpu_id is not None else "cuda",
    )
    _VOICE_CACHE: dict[str, Any] = {}  # device tensor copies keyed by voice_key

    def _voice_for(state: MossSpeechState) -> Any:
        key = state.voice_key
        if key is None:
            raise RequestValidationError("audio request missing voice conditioning")
        cached = _VOICE_CACHE.get(key)
        if cached is None:
            from sglang_omni.models.moss_speech.components.voice import (
                VoiceConditioning,
            )

            cached = VoiceConditioning(
                prompt_token=torch.tensor([state.voice_token_ids], dtype=torch.int32),
                prompt_feat=torch.tensor([state.voice_feat], dtype=torch.float32),
                embedding=torch.tensor([state.voice_embedding], dtype=torch.float32),
                meta={"sr": 24000},
            )
            _VOICE_CACHE[key] = cached
        return cached

    def compute(payload: StagePayload) -> StagePayload:
        state = _state_from_payload(payload)
        codes = extract_output_codes(state)
        voice = _voice_for(state)
        remember_vocoder_session(payload.request_id)
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        try:
            torch.manual_seed(state.effective_seed)
            torch.cuda.manual_seed_all(state.effective_seed)
            sr, wav = adapter.decode(codes, voice, request_id=payload.request_id)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if torch.cuda.is_available():
                for dev, st in enumerate(cuda_states):
                    torch.cuda.set_rng_state(st, dev)
            cleanup_vocoder_state(payload.request_id)
        state.audio_samples = wav.tolist()
        state.audio_sample_rate = int(sr)
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=state.to_dict()
        )

    return SimpleScheduler(compute, abort_callback=cleanup_vocoder_state)
