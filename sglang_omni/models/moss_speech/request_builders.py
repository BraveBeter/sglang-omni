# SPDX-License: Apache-2.0
"""Request building & canonical lowering for MOSS-Speech (P2/T2.3).

Pure-CPU normalize/validate implementing the P2 chat contract
(`docs/design/moss_speech/p2/01_chat_contract.md` §3/§6). The preprocessing
stage calls :func:`normalize_and_validate` before any codec/GPU work; the
AR-facing builder and routing helpers live here too.

Design notes:
- waveforms never cross the process boundary: preprocessing decodes audio
  and encodes to codec codes within the same stage; downstream state carries
  codes only;
- ``effective_seed``: explicit HTTP seed wins; otherwise derived from
  request_id via a stable blake2b hash (never Python ``hash()``);
- explicit parameter fidelity relies on the GAP-1 fix (length aliases are
  marked by the endpoint; see tests/unit_test/serve/).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import soundfile as _sf
import torch

from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY
from sglang_omni.proto.request import OmniRequest, StagePayload

from .payload_types import MossSpeechState

# V1 limits (chat contract §3.2; overridable via stage factory args later).
MAX_AUDIO_DURATION_S = 30.0
MAX_TOTAL_AUDIO_DURATION_S = 120.0
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000
SUPPORTED_AUDIO_FORMATS = {"wav", "flac", "ogg", "mp3", "m4a", "aac", "opus"}

TEXT_TERMINAL = "text_decode"
AUDIO_TERMINAL = "audio_vocoder"
_VALID_MODALITY_SETS = ({"text"}, {"audio"})


class RequestValidationError(ValueError):
    """Invalid request (maps to HTTP 400 at the boundary)."""

    def __init__(self, message: str, *, reason: str = "invalid_request") -> None:
        super().__init__(message)
        self.reason = reason


# --------------------------------------------------------------------- helpers
def _as_generate_request(inputs: Any):
    """Accept a client GenerateRequest object or its serialized dict form."""
    from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

    if isinstance(inputs, GenerateRequest):
        return inputs
    if isinstance(inputs, dict):
        data = dict(inputs)
        messages = data.get("messages")
        if isinstance(messages, list):
            data["messages"] = [Message(**m) if isinstance(m, dict) else m for m in messages]
        sampling = data.get("sampling")
        if isinstance(sampling, dict):
            data["sampling"] = SamplingParams(**sampling)
        return GenerateRequest(**data)
    raise RequestValidationError(f"unsupported request inputs type {type(inputs)!r}")


def _derive_effective_seed(request_id: str) -> int:
    digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _decode_audio_bytes(data: bytes, *, source: str) -> tuple[torch.Tensor, int]:
    """bytes -> (mono float32 (T,), original_sr); raises RequestValidationError."""
    if not data:
        raise RequestValidationError(f"audio {source} is empty", reason="invalid_audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise RequestValidationError(
            f"audio {source} exceeds {MAX_AUDIO_BYTES} bytes", reason="invalid_audio"
        )
    try:
        wav, sr = _sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a 400
        raise RequestValidationError(
            f"audio {source} could not be decoded: {exc}", reason="invalid_audio"
        ) from exc
    if wav.shape[0] == 0:
        raise RequestValidationError(f"audio {source} decodes to zero samples", reason="invalid_audio")
    if not MIN_SAMPLE_RATE <= sr <= MAX_SAMPLE_RATE:
        raise RequestValidationError(
            f"audio {source} sample rate {sr} outside [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]",
            reason="invalid_audio",
        )
    if not torch.isfinite(torch.from_numpy(wav)).all():
        raise RequestValidationError(f"audio {source} contains non-finite samples", reason="invalid_audio")
    duration = wav.shape[0] / sr
    if duration > MAX_AUDIO_DURATION_S:
        raise RequestValidationError(
            f"audio {source} duration {duration:.1f}s exceeds {MAX_AUDIO_DURATION_S}s",
            reason="invalid_audio",
        )
    mono = torch.from_numpy(wav[:, 0].copy())  # reference takes the first channel
    return mono, int(sr)


def _load_audio_source(source: str) -> tuple[torch.Tensor, int]:
    """Local file path or data URI; URLs are rejected in V1 (no new download stack)."""
    if source.startswith("data:"):
        _, _, b64 = source.partition(",")
        try:
            data = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise RequestValidationError(f"invalid data-URI audio: {exc}", reason="invalid_audio") from exc
        return _decode_audio_bytes(data, source="data-uri")
    if source.startswith(("http://", "https://", "ftp://")):
        raise RequestValidationError(
            "URL audio sources are not supported in V1; provide base64 input_audio or a local path",
            reason="not_supported",
        )
    if not os.path.isfile(source):
        raise RequestValidationError(f"audio path {source!r} not found", reason="invalid_audio")
    with open(source, "rb") as fh:
        return _decode_audio_bytes(fh.read(), source=source)


# ----------------------------------------------------------------- normalize
@dataclass
class NormalizedAudioTurn:
    waveform: torch.Tensor
    sample_rate: int
    source_key: str


def normalize_and_validate(request: OmniRequest, *, request_id: str) -> MossSpeechState:
    """HTTP-layered request -> canonical MossSpeechState (pure CPU).

    Raises RequestValidationError for every matrix row in the chat contract
    §6 before any GPU/codec work.
    """
    gen = _as_generate_request(request.inputs)

    # ---- V1 capability rejections ----------------------------------------
    if getattr(gen, "stream", False):
        raise RequestValidationError("streaming is not supported in V1", reason="not_supported")
    metadata = getattr(gen, "metadata", None) or {}
    if metadata.get("audio_config"):
        raise RequestValidationError(
            "request-level voice selection (audio config) is not supported in V1",
            reason="not_supported",
        )
    if getattr(gen, "stage_sampling", None) or getattr(gen, "stage_params", None):
        raise RequestValidationError("stage sampling/params overrides are not supported in V1", reason="not_supported")
    if getattr(gen, "extra_params", None):
        raise RequestValidationError("extra params are not supported in V1", reason="not_supported")
    if metadata.get("images") or metadata.get("videos"):
        raise RequestValidationError("image/video inputs are not supported in V1", reason="not_supported")

    # ---- output modality ---------------------------------------------------
    modalities = list(getattr(gen, "output_modalities", None) or ["text"])
    modality_set = {str(m) for m in modalities}
    if modality_set not in [set(s) for s in _VALID_MODALITY_SETS] or len(modalities) != 1:
        raise RequestValidationError(
            f"modalities must be exactly one of ['text'] or ['audio']; got {modalities!r}"
        )
    output_modality = modalities[0]

    # ---- messages -> canonical turns ---------------------------------------
    messages = list(getattr(gen, "messages", None) or [])
    if not messages:
        raise RequestValidationError("messages must not be empty")
    turns: List[dict[str, Any]] = []
    audio_turns: List[int] = []  # indices into turns
    inline_audio_sources: List[str] = []
    for i, msg in enumerate(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        if role not in ("system", "user", "assistant"):
            raise RequestValidationError(f"unsupported role {role!r} at message {i}")
        text_parts: List[str] = []
        audio_part: Optional[dict] = None
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type") if isinstance(part, dict) else None
                if ptype == "text":
                    text_parts.append(str(part.get("text", "")))
                elif ptype == "input_audio":
                    if audio_part is not None:
                        raise RequestValidationError(
                            f"message {i} contains multiple input_audio parts", reason="invalid_request"
                        )
                    audio_part = part.get("input_audio") or {}
                elif ptype in ("image_url", "image", "video", "video_url", "input_image"):
                    raise RequestValidationError(
                        f"content part type {ptype!r} is not supported in V1", reason="not_supported"
                    )
                elif ptype is None:
                    raise RequestValidationError(f"message {i} has a content part without a type")
                else:
                    raise RequestValidationError(
                        f"unsupported content part type {ptype!r} in message {i}", reason="not_supported"
                    )
        else:
            raise RequestValidationError(f"message {i} content must be string or parts list")
        if audio_part is not None and text_parts:
            raise RequestValidationError(
                f"message {i} mixes text and audio in one turn; single modality per turn required"
            )
        if audio_part is not None:
            turns.append({"role": role, "kind": "audio", "text": None})
            audio_turns.append(len(turns) - 1)
            inline_audio_sources.append(str(audio_part.get("data", "")))
        else:
            turns.append({"role": role, "kind": "text", "text": "".join(text_parts)})

    # ---- audios[] binding (contract §3.2) -----------------------------------
    audios_meta = metadata.get("audios")
    if audios_meta:
        if inline_audio_sources:
            raise RequestValidationError(
                "provide either per-turn input_audio parts or audios[], not both",
                reason="invalid_request",
            )
        sources = [str(a) for a in audios_meta]
        if len(sources) != len(audio_turns):
            raise RequestValidationError(
                f"audios[] has {len(sources)} items but the conversation has "
                f"{len(audio_turns)} audio turns; unambiguous 1:1 in-order binding required",
                reason="invalid_request",
            )
        inline_audio_sources = sources
    elif audio_turns and not inline_audio_sources:
        raise RequestValidationError(
            "audio turns present but no input_audio parts or audios[] provided",
            reason="invalid_request",
        )

    # ---- decode audio (CPU) --------------------------------------------------
    decoded: List[NormalizedAudioTurn] = []
    total_duration = 0.0
    for idx, source in zip(audio_turns, inline_audio_sources):
        if source.startswith("data:") or not source.startswith(("http://", "https://")) and not os.path.isfile(source):
            # inline base64 payload arrives as raw base64 string
            try:
                data = base64.b64decode(source, validate=False) if not source.startswith("data:") else None
            except (binascii.Error, ValueError) as exc:
                raise RequestValidationError(f"invalid base64 audio: {exc}", reason="invalid_audio") from exc
            if data is not None:
                wav, sr = _decode_audio_bytes(data, source=f"turn[{idx}]")
            else:
                wav, sr = _load_audio_source(source)
        else:
            wav, sr = _load_audio_source(source)
        total_duration += wav.shape[0] / sr
        digest = hashlib.blake2b(wav.numpy().tobytes(), digest_size=16).hexdigest()
        decoded.append(NormalizedAudioTurn(waveform=wav, sample_rate=sr, source_key=digest))
    if total_duration > MAX_TOTAL_AUDIO_DURATION_S:
        raise RequestValidationError(
            f"total audio duration {total_duration:.1f}s exceeds {MAX_TOTAL_AUDIO_DURATION_S}s",
            reason="invalid_audio",
        )

    # ---- conservative CPU context-length pre-check (contract §6) -------------
    _PRECHECK_CHARS_PER_TOKEN = 2.0
    _PRECHECK_TEMPLATE_TOKENS = 64
    est_tokens = _PRECHECK_TEMPLATE_TOKENS + sum(
        max(len(t.get("text") or "") / _PRECHECK_CHARS_PER_TOKEN, 1.0) for t in turns
    )
    est_tokens += sum(waveform.shape[0] / sr * 12.5 for waveform, sr in ((d.waveform, d.sample_rate) for d in decoded))
    if est_tokens > 10240:  # SFT context bound (P0 contract); exact check is post-encode
        raise RequestValidationError(
            f"estimated context length {int(est_tokens)} exceeds the 10240 limit", reason="context_too_long"
        )

    # ---- parameters -----------------------------------------------------------
    sampling = getattr(gen, "sampling", None)
    seed_val = getattr(sampling, "seed", None) if sampling is not None else None
    effective_seed = int(seed_val) if seed_val is not None else _derive_effective_seed(request_id)
    explicit = list(metadata.get(EXPLICIT_GENERATION_PARAMS_KEY, []) or [])

    state = MossSpeechState(
        output_modality=output_modality,
        turns=turns,
        explicit_params=explicit,
        effective_seed=effective_seed,
        temperature=getattr(sampling, "temperature", None) if sampling is not None else None,
        top_p=getattr(sampling, "top_p", None) if sampling is not None else None,
        top_k=getattr(sampling, "top_k", None) if sampling is not None else None,
        repetition_penalty=getattr(sampling, "repetition_penalty", None) if sampling is not None else None,
        max_new_tokens=getattr(gen, "max_tokens", None),
        stop=list(getattr(sampling, "stop", None) or []),
    )
    # stash decoded audio for the same-process encode step (never serialized)
    state.__dict__["_decoded_audio"] = [(d.waveform, d.sample_rate, d.source_key) for d in decoded]
    return state


def pop_decoded_audio(state: MossSpeechState) -> List[tuple[torch.Tensor, int, str]]:
    """Consume the process-local decoded audio stash (preprocessing only)."""
    return state.__dict__.pop("_decoded_audio", [])


# ------------------------------------------------------------------- routing
def resolve_output_terminal(request_id: str, output: Any) -> str:
    """route_fn(request_id, output) -> 'text_decode' | 'audio_vocoder'."""
    data = getattr(output, "data", output)
    modality = None
    if isinstance(data, MossSpeechState):
        modality = data.output_modality
    elif isinstance(data, dict):
        modality = data.get("output_modality")
    if modality == "text":
        return TEXT_TERMINAL
    if modality == "audio":
        return AUDIO_TERMINAL
    raise RequestValidationError(
        f"request {request_id}: cannot route without a valid output_modality (got {modality!r})"
    )


# ------------------------------------------------------------------ cleanup
# Per-stage owner registries (contract: idempotent, stage-scoped; the shared
# default voice cache is never released by request cleanup).
_PREPARED_LOCK = threading.Lock()
_PREPARED_REQUESTS: "dict[str, MossSpeechState]" = {}


def remember_prepared(request_id: str, state: MossSpeechState) -> None:
    with _PREPARED_LOCK:
        _PREPARED_REQUESTS[request_id] = state


def cleanup_preprocessing_state(request_id: str) -> None:
    """Idempotent abort cleanup for the preprocessing owner scope."""
    with _PREPARED_LOCK:
        _PREPARED_REQUESTS.pop(request_id, None)


_VOCODER_SESSIONS_LOCK = threading.Lock()
_VOCODER_SESSIONS: "dict[str, bool]" = {}


def remember_vocoder_session(request_id: str) -> None:
    with _VOCODER_SESSIONS_LOCK:
        _VOCODER_SESSIONS[request_id] = True


def cleanup_vocoder_state(request_id: str) -> None:
    """Idempotent abort cleanup for the vocoder owner scope (P2 stub registry)."""
    with _VOCODER_SESSIONS_LOCK:
        _VOCODER_SESSIONS.pop(request_id, None)
