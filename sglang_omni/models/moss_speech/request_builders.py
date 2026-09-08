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
import math
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import soundfile as _sf
import torch

from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY
from sglang_omni.proto.request import OmniRequest, StagePayload

from .payload_types import MossSpeechState

if TYPE_CHECKING:
    from .request_data import MossSpeechSGLangRequestData

MODALITY_PAD_TOKEN = 151667
TEXT_ENDOFTEXT_TOKEN = 151643
IM_END_TOKEN = 151645
TEXT_VOCAB_LIMIT = 151680

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
        super().__init__(f"Invalid model request: {message}")
        self.reason = reason


# --------------------------------------------------------------------- helpers
def _as_generate_request(inputs: Any) -> Any:
    """Accept a client GenerateRequest object or its serialized dict form."""
    from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

    if isinstance(inputs, GenerateRequest):
        return inputs
    if isinstance(inputs, dict):
        data = dict(inputs)
        messages = data.get("messages")
        if isinstance(messages, list):
            data["messages"] = [
                Message(**m) if isinstance(m, dict) else m for m in messages
            ]
        sampling = data.get("sampling")
        if isinstance(sampling, dict):
            data["sampling"] = SamplingParams(**sampling)
        return GenerateRequest(**data)
    raise RequestValidationError(f"unsupported request inputs type {type(inputs)!r}")


def _request_generate(request: OmniRequest) -> Any:
    """Restore the public Client's split wire form; retain P3 full requests."""
    from dataclasses import fields

    from sglang_omni.client.types import SamplingParams

    inputs = request.inputs
    if not isinstance(inputs, list) and not request.params and not request.metadata:
        return _as_generate_request(inputs)
    if isinstance(inputs, dict) and "sampling" in inputs:
        return _as_generate_request(inputs)
    metadata = dict(request.metadata or {})
    if isinstance(inputs, dict):
        messages = inputs.get("messages")
        for key in ("audios", "images", "videos"):
            if key in inputs:
                metadata[key] = inputs[key]
    else:
        messages = inputs
    if not isinstance(messages, list):
        raise RequestValidationError("MOSS-Speech requires chat messages")
    params = dict(request.params or {})
    keys = {field.name for field in fields(SamplingParams)}
    sampling = {key: value for key, value in params.items() if key in keys}
    extra = set(params) - keys - {"stream", "stage_sampling", "stage_params"}
    if extra:
        raise RequestValidationError(f"unsupported request parameters: {sorted(extra)}")
    return _as_generate_request(
        dict(
            messages=messages,
            metadata=metadata,
            sampling=sampling,
            output_modalities=metadata.get("output_modalities", ["text"]),
            stream=params.get("stream", False),
            max_tokens=params.get("max_new_tokens"),
            stage_sampling=params.get("stage_sampling"),
            stage_params=params.get("stage_params"),
        )
    )


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
        raise RequestValidationError(
            f"audio {source} decodes to zero samples", reason="invalid_audio"
        )
    if not MIN_SAMPLE_RATE <= sr <= MAX_SAMPLE_RATE:
        raise RequestValidationError(
            f"audio {source} sample rate {sr} outside [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]",
            reason="invalid_audio",
        )
    if not torch.isfinite(torch.from_numpy(wav)).all():
        raise RequestValidationError(
            f"audio {source} contains non-finite samples", reason="invalid_audio"
        )
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
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestValidationError(
                f"invalid data-URI audio: {exc}", reason="invalid_audio"
            ) from exc
        return _decode_audio_bytes(data, source="data-uri")
    if source.startswith(("http://", "https://", "ftp://")):
        raise RequestValidationError(
            "URL audio sources are not supported in V1; provide base64 input_audio or a local path",
            reason="not_supported",
        )
    if not os.path.isfile(source):
        raise RequestValidationError(
            f"audio path {source!r} not found", reason="invalid_audio"
        )
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
    gen = _request_generate(request)

    sampling = getattr(gen, "sampling", None)
    for name in ("temperature", "top_p", "repetition_penalty", "min_p"):
        value = getattr(sampling, name, None)
        if value is not None and not math.isfinite(float(value)):
            raise RequestValidationError(f"{name} must be finite")
    if getattr(sampling, "min_p", 0.0) not in (None, 0.0):
        raise RequestValidationError(
            "min_p is not supported in V1", reason="not_supported"
        )
    temperature = getattr(sampling, "temperature", None)
    top_p = getattr(sampling, "top_p", None)
    penalty = getattr(sampling, "repetition_penalty", None)
    if (
        temperature is not None
        and temperature < 0
        or top_p is not None
        and not 0 < top_p <= 1
        or penalty is not None
        and penalty <= 0
    ):
        raise RequestValidationError("invalid sampling parameters")

    top_k = getattr(sampling, "top_k", None)
    seed = getattr(sampling, "seed", None)
    length = getattr(gen, "max_tokens", None)
    if top_k is not None and top_k < -1:
        raise RequestValidationError("top_k must be -1, 0 or positive")
    if seed is not None and not -(2**63) <= seed < 2**64:
        raise RequestValidationError("seed must fit torch.Generator's 64-bit range")
    if length is not None and not 1 <= length <= 512:
        raise RequestValidationError("max_new_tokens must be within [1, 512]")

    # ---- V1 capability rejections ----------------------------------------
    if getattr(getattr(gen, "sampling", None), "stop", None):
        raise RequestValidationError(
            "custom stop strings are not supported in V1", reason="not_supported"
        )
    if getattr(gen, "stream", False):
        raise RequestValidationError(
            "streaming is not supported in V1", reason="not_supported"
        )
    metadata = getattr(gen, "metadata", None) or {}
    if metadata.get("audio_config"):
        raise RequestValidationError(
            "request-level voice selection (audio config) is not supported in V1",
            reason="not_supported",
        )
    if getattr(gen, "stage_sampling", None) or getattr(gen, "stage_params", None):
        raise RequestValidationError(
            "stage sampling/params overrides are not supported in V1",
            reason="not_supported",
        )
    if getattr(gen, "extra_params", None):
        raise RequestValidationError(
            "extra params are not supported in V1", reason="not_supported"
        )
    if metadata.get("images") or metadata.get("videos"):
        raise RequestValidationError(
            "image/video inputs are not supported in V1", reason="not_supported"
        )

    # ---- output modality ---------------------------------------------------
    modalities = list(getattr(gen, "output_modalities", None) or ["text"])
    modality_set = {str(m) for m in modalities}
    if (
        modality_set not in [set(s) for s in _VALID_MODALITY_SETS]
        or len(modalities) != 1
    ):
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
        role = getattr(msg, "role", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        content = (
            getattr(msg, "content", None)
            if not isinstance(msg, dict)
            else msg.get("content")
        )
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
                            f"message {i} contains multiple input_audio parts",
                            reason="invalid_request",
                        )
                    audio_part = part.get("input_audio") or {}
                elif ptype in (
                    "image_url",
                    "image",
                    "video",
                    "video_url",
                    "input_image",
                ):
                    raise RequestValidationError(
                        f"content part type {ptype!r} is not supported in V1",
                        reason="not_supported",
                    )
                elif ptype is None:
                    raise RequestValidationError(
                        f"message {i} has a content part without a type"
                    )
                else:
                    raise RequestValidationError(
                        f"unsupported content part type {ptype!r} in message {i}",
                        reason="not_supported",
                    )
        else:
            raise RequestValidationError(
                f"message {i} content must be string or parts list"
            )
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
        if any(inline_audio_sources):
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
        if (
            source.startswith("data:")
            or not source.startswith(("http://", "https://"))
            and not os.path.isfile(source)
        ):
            # inline base64 payload arrives as raw base64 string
            try:
                data = (
                    base64.b64decode(source, validate=True)
                    if not source.startswith("data:")
                    else None
                )
            except (binascii.Error, ValueError) as exc:
                raise RequestValidationError(
                    f"invalid base64 audio: {exc}", reason="invalid_audio"
                ) from exc
            if data is not None:
                wav, sr = _decode_audio_bytes(data, source=f"turn[{idx}]")
            else:
                wav, sr = _load_audio_source(source)
        else:
            wav, sr = _load_audio_source(source)
        total_duration += wav.shape[0] / sr
        digest = hashlib.blake2b(wav.numpy().tobytes(), digest_size=16).hexdigest()
        decoded.append(
            NormalizedAudioTurn(waveform=wav, sample_rate=sr, source_key=digest)
        )
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
    est_tokens += sum(
        waveform.shape[0] / sr * 12.5
        for waveform, sr in ((d.waveform, d.sample_rate) for d in decoded)
    )
    if (
        est_tokens > 10240
    ):  # SFT context bound (P0 contract); exact check is post-encode
        raise RequestValidationError(
            f"estimated context length {int(est_tokens)} exceeds the 10240 limit",
            reason="context_too_long",
        )

    # ---- parameters -----------------------------------------------------------
    sampling = getattr(gen, "sampling", None)
    seed_val = getattr(sampling, "seed", None) if sampling is not None else None
    effective_seed = (
        int(seed_val) if seed_val is not None else _derive_effective_seed(request_id)
    )
    explicit = list(metadata.get(EXPLICIT_GENERATION_PARAMS_KEY, []) or [])

    state = MossSpeechState(
        output_modality=output_modality,
        turns=turns,
        explicit_params=explicit,
        effective_seed=effective_seed,
        temperature=(
            getattr(sampling, "temperature", None) if sampling is not None else None
        ),
        top_p=getattr(sampling, "top_p", None) if sampling is not None else None,
        top_k=getattr(sampling, "top_k", None) if sampling is not None else None,
        repetition_penalty=(
            getattr(sampling, "repetition_penalty", None)
            if sampling is not None
            else None
        ),
        max_new_tokens=getattr(gen, "max_tokens", None),
        stop=list(getattr(sampling, "stop", None) or []),
    )
    # stash decoded audio for the same-process encode step (never serialized)
    state.__dict__["_decoded_audio"] = [
        (d.waveform, d.sample_rate, d.source_key) for d in decoded
    ]
    return state


def pop_decoded_audio(state: MossSpeechState) -> List[tuple[torch.Tensor, int, str]]:
    """Consume the process-local decoded audio stash (preprocessing only)."""
    return state.__dict__.pop("_decoded_audio", [])


# ------------------------------------------------------------------- routing
def resolve_terminal_stages(request: OmniRequest) -> List[str]:
    """terminal_stages_fn: which terminals the coordinator joins for a request.

    V1 output modalities are mutually exclusive (both -> rejected in
    normalize), so exactly one terminal is active per request.
    """
    inputs = getattr(request, "inputs", None)
    modalities = None
    if isinstance(inputs, dict):
        modalities = inputs.get("output_modalities")
    else:
        modalities = getattr(inputs, "output_modalities", None)
    modalities = list(
        modalities or (request.metadata or {}).get("output_modalities") or ["text"]
    )
    if modalities == ["audio"]:
        return [AUDIO_TERMINAL]
    if modalities == ["text"]:
        return [TEXT_TERMINAL]
    raise RequestValidationError(
        f"cannot resolve terminal stages for modalities {modalities!r}"
    )


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


# --------------------------------------------------------------------------
# SGLang AR request data (T3.4): per-request state for the native engine.
# Owner = the AR model runner; allocate on adapter build, reset/free on
# finish/abort/cleanup (idempotent).
# --------------------------------------------------------------------------
def __getattr__(name: str) -> Any:
    """Keep the historical native request type import path lazily available."""
    if name == "MossSpeechSGLangRequestData":
        from .request_data import MossSpeechSGLangRequestData

        return MossSpeechSGLangRequestData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_sglang_moss_request(
    state: Any, *, payload: Any = None
) -> MossSpeechSGLangRequestData:
    """MossSpeechState (P2 wire) -> engine request data.

    Consumes the frozen P2 fields: input grid rows, explicit/derived sampling
    parameters, effective seed. Explicit HTTP params win; the HTTP fill-in
    defaults (1.0/1.0/-1/1.0) arrive as explicit markers via
    EXPLICIT_GENERATION_PARAMS_KEY and are honored as-is (P2 chat contract).
    """
    from sglang_omni.models.moss_speech.fsm import MossSamplingParams, initial_mode

    from .request_data import MossSpeechSGLangRequestData

    explicit = set(getattr(state, "explicit_params", []) or [])
    defaults = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }

    def value(name: Any) -> Any:
        raw = getattr(state, name, None)
        return raw if name in explicit and raw is not None else defaults[name]

    temperature = float(value("temperature"))
    top_p = float(value("top_p"))
    top_k = int(value("top_k"))
    penalty = float(value("repetition_penalty"))
    max_new = getattr(state, "max_new_tokens", None)
    max_new = 200 if max_new is None else int(max_new)
    if not 0 <= temperature or not 0 < top_p <= 1 or top_k not in (-1, 0) and top_k < 1:
        raise RequestValidationError("invalid native sampling parameters")
    if (
        not all(math.isfinite(v) for v in (temperature, top_p, penalty))
        or penalty <= 0
        or max_new < 1
    ):
        raise RequestValidationError(
            "repetition_penalty and max_new_tokens must be positive"
        )
    if getattr(state, "stop", None):
        raise RequestValidationError(
            "custom stop strings are not supported in V1", reason="not_supported"
        )
    params = MossSamplingParams(
        do_sample=temperature > 0 and top_k != 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=penalty,
        min_new_tokens=int(getattr(state, "min_new_tokens", 0) or 0),
        max_new_tokens=max_new,
    )
    prompt_rows = torch.as_tensor(state.input_grid, dtype=torch.long)
    if prompt_rows.dim() == 3 and prompt_rows.shape[0] == 1:
        prompt_rows = prompt_rows[0]
    if prompt_rows.dim() != 2 or prompt_rows.shape[1] != 2 or not len(prompt_rows):
        raise RequestValidationError("native input_grid must have shape (L, 2)")
    mask = getattr(state, "attention_mask", None)
    if mask is not None and len(mask):
        mask = torch.as_tensor(mask).reshape(-1)
        if mask.numel() != len(prompt_rows) or not ((mask == 0) | (mask == 1)).all():
            raise RequestValidationError("invalid native attention_mask")
        if not mask.any() or (mask[1:] < mask[:-1]).any():
            raise RequestValidationError(
                "native attention_mask must be nonempty left padding"
            )
        prompt_rows = prompt_rows[mask.bool()]
    if ((prompt_rows[:, 0] < 0) | (prompt_rows[:, 0] >= TEXT_VOCAB_LIMIT)).any() or (
        (prompt_rows[:, 1] < 0) | (prompt_rows[:, 1] >= 16512)
    ).any():
        raise RequestValidationError("native input_grid contains out-of-vocabulary IDs")
    if not 0 <= params.min_new_tokens <= params.max_new_tokens:
        raise RequestValidationError(
            "min_new_tokens must be within [0, max_new_tokens]"
        )
    # 1-D scheduler representation: one selected token per grid row (the
    # embedding-selected channel value; lossless per P3-01 §3.1)
    selected = [
        int(r[1]) if int(r[0]) == MODALITY_PAD_TOKEN else int(r[0])
        for r in prompt_rows.tolist()
    ]
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    sp = SamplingParams(
        max_new_tokens=params.max_new_tokens,
        temperature=0.0,  # engine-side sampling is inert: the runner samples
    )
    sp.normalize(None)
    sp.verify(TEXT_VOCAB_LIMIT)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=selected,
        sampling_params=sp,
        eos_token_ids={TEXT_ENDOFTEXT_TOKEN, IM_END_TOKEN},
        vocab_size=TEXT_VOCAB_LIMIT,
    )
    req.tokenizer = None

    data = MossSpeechSGLangRequestData(
        req=req,
        input_ids=torch.tensor(selected, dtype=torch.long),
        prompt_rows=prompt_rows,
        mode=initial_mode(prompt_rows),
        params=params,
        effective_seed=int(getattr(state, "effective_seed", 0) or 0),
        max_new_tokens=params.max_new_tokens,
        enforce_request_limits=True,
    )
    data.stage_payload = payload
    # record which fields were explicit for the adapter contract test
    data.__dict__["_explicit_fields"] = sorted(explicit)
    return data


def make_moss_speech_scheduler_adapters(*, model: Any, lifecycle: Any = None) -> Any:
    """StagePayload <-> engine request adapters (moss_tts_local pattern)."""
    import traceback as _tb

    if lifecycle is None:
        from sglang_omni.models.moss_speech.request_lifecycle import (
            MossSpeechRequestLifecycle,
        )

        lifecycle = MossSpeechRequestLifecycle(model)

    def _logged(fn: Any) -> Any:
        def wrapper(*a: Any, **k: Any) -> Any:
            try:
                return fn(*a, **k)
            except Exception:
                _tb.print_exc()
                raise

        return wrapper

    @_logged
    def request_builder(payload: Any) -> Any:
        state = payload.data
        if not isinstance(state, MossSpeechState):
            state = MossSpeechState.from_dict(state)
        data = build_sglang_moss_request(state, payload=payload)
        return lifecycle.admit(data, state.output_modality)

    @_logged
    def result_adapter(data: Any) -> Any:
        try:
            import os as _osr

            if _osr.environ.get("MOSS_DUMP_DIR"):
                _r = getattr(data, "req", None)
                print(
                    "[adapter] finish_reason:",
                    getattr(data, "finish_reason", None),
                    "output_rows:",
                    len(data.output_rows),
                    "req_output_ids:",
                    (list(_r.output_ids)[-6:] if _r is not None else None),
                    "fr_detail:",
                    (str(_r.finished_reason)[:120] if _r is not None else None),
                    flush=True,
                )
            payload = data.stage_payload
            state = payload.data
            if not isinstance(state, MossSpeechState):
                state = MossSpeechState.from_dict(state)
            state.output_grid = [list(map(int, r)) for r in data.output_rows]
            state.finish_reason = data.finish_reason
            return StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=state.to_dict(),
            )
        finally:
            lifecycle.release(data.req.rid)

    return request_builder, result_adapter


def cleanup_ar_request_state(data: MossSpeechSGLangRequestData) -> None:
    """Idempotent AR-owner cleanup: release per-request state only."""
    if data is None:
        return
    (
        data.pending_feedback_queue.clear()
        if data.pending_feedback_queue is not None
        else None
    )
    data.output_rows = []
    data.rng_generator = None
    data.finished = True
