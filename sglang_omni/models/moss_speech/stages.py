# SPDX-License-Identifier: Apache-2.0
"""Factories for MOSS-Speech preprocessing, generation, and terminals."""

from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.models.moss_speech.audio_tasks import RequestValidationError
from sglang_omni.models.moss_speech.components.codec_adapter import (
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.processor import MossSpeechGridProcessor
from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.models.moss_speech.hf_config import (
    MossSpeechConfig,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.request_builders import (
    normalize_and_validate,
    pop_decoded_audio,
)
from sglang_omni.models.moss_speech.request_data import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.streaming import (
    MossSpeechStreamingText,
    MossSpeechStreamingVocoder,
)
from sglang_omni.models.moss_speech.voice_transport import (
    pack_voice_payload,
    voice_payload_digest,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device


def resolve_codec_dir(model_path: str, codec_path: str | None) -> str:
    codec_directory = Path(codec_path or f"{model_path.rstrip('/')}-Codec")
    if not codec_directory.is_dir():
        raise FileNotFoundError(f"Codec directory {codec_directory} does not exist")
    else:
        return str(codec_directory)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    ensure_moss_speech_config_registered()
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


def validate_prompt_length(length: int, limit: int) -> None:
    if length < 1 or length > limit:
        raise RequestValidationError(
            f"Canonical prompt length {length} exceeds the supported range [1, {limit}]",
            reason="context_too_long",
        )
    else:
        pass


def create_preprocessing_executor(
    model_path: str,
    *,
    encode_batch_size: int,
    context_limit: int,
    codec_path: str | None = None,
    voice_wav: str | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if voice_wav is None or not Path(voice_wav).is_file():
        raise FileNotFoundError(
            "Set preprocessing.factory.voice_wav to a reference WAV"
        )
    else:
        pass
    tokenizer = load_tokenizer(model_path)
    processor = MossSpeechGridProcessor(tokenizer)
    adapter = MossSpeechCodecAdapter(
        resolve_codec_dir(model_path, codec_path),
        device=str(resolve_concrete_device(device, gpu_id)),
        load_encoder=True,
        load_decoder=False,
    )
    try:
        voice = adapter.encode_voice_ref(voice_wav)
        voice_payload = pack_voice_payload(
            {
                "prompt_token": voice.prompt_token,
                "prompt_feat": voice.prompt_feat,
                "embedding": voice.embedding,
            }
        )
    except BaseException:
        adapter.close()
        raise
    voice_key = voice_payload_digest(voice_payload)

    def compute(payload: StagePayload) -> StagePayload:
        state = normalize_and_validate(
            payload.request, request_id=payload.request_id, allow_streaming=True
        )
        decoded_audio = pop_decoded_audio(state)
        if decoded_audio:
            codes = adapter.encode(
                [(waveform, sample_rate) for waveform, sample_rate, _ in decoded_audio],
                batch_size=encode_batch_size,
            )
        else:
            codes = []
        state.audio_codes = codes
        canonical = processor.build(state.turns, codes, state.output_modality)
        validate_prompt_length(canonical.prompt_len, context_limit)
        state.input_grid = canonical.grid[0].tolist()
        state.attention_mask = canonical.attention_mask[0].tolist()
        state.prompt_grid_len = canonical.prompt_len
        if state.output_modality == "audio":
            state.voice_key = voice_key
            state.voice_payload = voice_payload
        else:
            pass
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=state.to_dict()
        )

    return SimpleScheduler(compute, shutdown_callback=adapter.close)


def validate_ar_preconditions(model_path: str, *, dtype: str) -> MossSpeechConfig:
    if dtype not in {"bfloat16", "bf16"}:
        raise ValueError("MOSS-Speech requires BF16 AR weights")
    else:
        pass
    return MossSpeechConfig.from_pretrained(model_path)


def create_ar_engine_executor(
    model_path: str,
    *,
    dtype: str,
    context_length: int,
    native_norm_library: str | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[MossSpeechSGLangRequestData]:
    validate_ar_preconditions(model_path, dtype=dtype)
    overrides = dict(server_args_overrides or {})
    overrides["trust_remote_code"] = False
    if total_gpu_memory_fraction is not None:
        overrides.setdefault("mem_fraction_static", total_gpu_memory_fraction)
    elif process_total_gpu_memory_fraction is not None:
        overrides.setdefault("mem_fraction_static", process_total_gpu_memory_fraction)
    else:
        pass
    return MossSpeechEngineBuilder(
        context_length=context_length,
        streaming=True,
        native_norm_library=native_norm_library,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=overrides,
        total_gpu_memory_fraction=(
            process_total_gpu_memory_fraction
            if process_total_gpu_memory_fraction is not None
            else total_gpu_memory_fraction
        ),
    )


def create_text_decode_executor(model_path: str) -> MossSpeechStreamingText:
    return MossSpeechStreamingText(load_tokenizer(model_path))


def create_audio_vocoder_executor(
    model_path: str,
    *,
    chunk_size: int,
    codec_path: str | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
) -> MossSpeechStreamingVocoder:
    codec_directory = resolve_codec_dir(model_path, codec_path)
    concrete_device = str(resolve_concrete_device(device, gpu_id))
    offline_codec = MossSpeechCodecAdapter(
        codec_directory,
        device=concrete_device,
        load_encoder=False,
        load_decoder=True,
    )
    try:
        streaming_codec = MossSpeechStreamingCodec(
            codec_directory, chunk_size=chunk_size, device=concrete_device
        )
    except BaseException:
        offline_codec.close()
        raise
    return MossSpeechStreamingVocoder(streaming_codec, offline_codec)
