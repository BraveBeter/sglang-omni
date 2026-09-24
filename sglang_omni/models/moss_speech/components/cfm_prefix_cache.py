# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicitly prepared fixed-voice CFM prefix caches.

Conditioning is validated on the CPU before startup work or request transfers.
Unsupported requests retain the ordinary codec path; cache construction never
happens on a request's first audio chunk.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparedVoice:
    voice: dict[str, Any]
    voice_digest: str
    warmup_audio_codes: tuple[int, ...]
    manifest_sha256: str
    source_sha256: dict[str, str]
    codec_sha256: dict[str, str]
    environment: dict[str, Any]
    verified_files: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(source_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(source_root)): _sha256(path)
        for path in sorted(source_root.rglob("*.py"))
    }


def _checked_file(base: Path, description: dict[str, Any]) -> tuple[Path, str]:
    if not isinstance(description, dict) or not isinstance(
        description.get("path"), str
    ):
        raise ValueError("Prepared voice requires a local file and SHA256")
    path = Path(description["path"])
    path = (path if path.is_absolute() else base / path).resolve()
    expected = description.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Prepared file has no complete SHA256")
    try:
        actual = _sha256(path)
    except OSError as error:
        raise ValueError(f"Prepared input cannot be read: {path}") from error
    if actual != expected:
        raise ValueError(f"Prepared input changed: {path}")
    return path, actual


def load_prepared_voice(
    manifest_path: str | Path,
    *,
    codec_root: Path,
    source_root: Path,
    environment: dict[str, Any],
) -> PreparedVoice:
    """Validate immutable input bytes and runtime before constructing GPU state."""
    import torch

    from ..voice_transport import pack_voice_payload, voice_payload_digest

    path = Path(manifest_path).resolve()
    try:
        raw_manifest = path.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read prepared voice manifest") from error
    required = {
        "schema_version",
        "prompt_codes",
        "voice",
        "reference_wave",
        "codec_sha256",
        "source_sha256",
        "environment",
        "warmup_audio_codes",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Incomplete or unknown prepared voice manifest fields")
    if manifest["schema_version"] != 1 or manifest["prompt_codes"] != 342:
        raise ValueError("Prepared prefix cache requires schema 1 and 342 prompt codes")
    if manifest["environment"] != environment:
        raise ValueError("Prepared prefix cache environment changed")
    source_root, codec_root = Path(source_root).resolve(), Path(codec_root).resolve()
    sources = source_fingerprint(source_root)
    if not sources or manifest["source_sha256"] != sources:
        raise ValueError("Prepared CFM source fingerprint changed")
    codec_files = manifest["codec_sha256"]
    if not isinstance(codec_files, dict) or not {
        "flow/flow-chunk-25.pt",
        "flow/hift.pt",
    }.issubset(codec_files):
        raise ValueError("Prepared manifest must bind both flow and HiFT weights")
    verified: dict[str, str] = {}
    for relative, digest in codec_files.items():
        if not isinstance(relative, str):
            raise ValueError("Codec path is not a string")
        target = (codec_root / relative).resolve()
        if Path(relative).is_absolute() or not target.is_relative_to(codec_root):
            raise ValueError("Codec fingerprint escapes the codec directory")
        checked, actual = _checked_file(codec_root, dict(path=relative, sha256=digest))
        verified[str(checked)] = actual
    codes = manifest["warmup_audio_codes"]
    if (
        not isinstance(codes, list)
        or len(codes) != 36
        or any(type(code) is not int or not 0 <= code < 16384 for code in codes)
    ):
        raise ValueError("Prepared calibration requires 36 valid audio codes")
    voice_path, voice_sha = _checked_file(path.parent, manifest["voice"])
    wave_path, wave_sha = _checked_file(path.parent, manifest["reference_wave"])
    verified.update({str(voice_path): voice_sha, str(wave_path): wave_sha})
    try:
        voice = torch.load(voice_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Prepared voice is not a safe tensor artifact") from error
    if not isinstance(voice, dict) or set(voice) != {
        "prompt_token",
        "prompt_feat",
        "embedding",
    }:
        raise ValueError("Prepared voice requires exactly three conditioning tensors")
    if any(
        not isinstance(value, torch.Tensor) or value.device.type != "cpu"
        for value in voice.values()
    ):
        raise ValueError("Prepared voice conditioning must be CPU tensors")
    if (
        voice["prompt_token"].shape != (1, 342)
        or voice["prompt_feat"].shape != (1, 1368, 80)
        or voice["embedding"].shape != (1, 192)
        or voice["prompt_token"].dtype not in (torch.int32, torch.int64)
        or voice["prompt_feat"].dtype != torch.float32
        or voice["embedding"].dtype != torch.float32
    ):
        raise ValueError("Prepared voice shape or dtype changed")
    if not bool(((voice["prompt_token"] >= 0) & (voice["prompt_token"] < 16384)).all()):
        raise ValueError("Prepared voice contains invalid audio codes")
    digest = voice_payload_digest(pack_voice_payload(voice))
    if digest != manifest["voice"].get("payload_sha256"):
        raise ValueError("Prepared voice conditioning bytes changed")
    return PreparedVoice(
        voice=voice,
        voice_digest=digest,
        warmup_audio_codes=tuple(codes),
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        source_sha256=sources,
        codec_sha256=dict(codec_files),
        environment=dict(environment),
        verified_files=verified,
    )


@dataclass
class CacheBinding:
    voice_digest: str
    shapes: tuple[int, ...]
    environment: dict[str, Any]
    epoch: tuple[Any, ...]
    valid: bool = True

    def __post_init__(self) -> None:
        if (
            not self.shapes
            or len(set(self.shapes)) != len(self.shapes)
            or (
                not set(self.shapes).issubset({1400, 1500})
                and self.shapes != (1392, 1424, 1500)
            )
        ):
            raise ValueError(
                "A prefix bank permits only distinct measured shapes 1400/1500 "
                "or the explicit startup schedule 1392/1424/1500"
            )

    def matches(
        self,
        voice: str | None,
        length: int,
        final: bool,
        environment: dict[str, Any],
        epoch: tuple[Any, ...],
    ) -> bool:
        return (
            self.valid
            and not final
            and voice == self.voice_digest
            and length in self.shapes
            and environment == self.environment
            and epoch == self.epoch
        )

    def invalidate(self) -> None:
        self.valid = False


class PrefixCallLedger:
    def __init__(self, names: tuple[str, ...], steps: int = 10) -> None:
        if not names or steps <= 0:
            raise ValueError("Prefix execution needs layers and Euler steps")
        self.names, self.steps, self.count = names, steps, 0

    def next(self, name: str) -> tuple[int, str]:
        if (
            self.count >= len(self.names) * self.steps
            or name != self.names[self.count % len(self.names)]
        ):
            raise ValueError("CFM prefix call order changed")
        step = self.count // len(self.names)
        self.count += 1
        return step, name

    def finish(self, *, allow_empty: bool = False) -> None:
        if self.count != len(self.names) * self.steps and not (
            allow_empty and self.count == 0
        ):
            raise ValueError("CFM prefix execution was incomplete")


def tensor_epoch(module: Any) -> tuple[Any, ...]:
    """CPU metadata only; includes CFM's otherwise unregistered fixed noise."""
    import torch

    def describe(name: str, value: torch.Tensor) -> tuple[Any, ...]:
        try:
            version = value._version
        except RuntimeError:
            version = None
        return (
            name,
            id(value),
            value.data_ptr(),
            tuple(value.shape),
            value.stride(),
            value.storage_offset(),
            str(value.dtype),
            str(value.device),
            version,
        )

    entries = [describe(name, value) for name, value in module.named_parameters()]
    entries += [describe(name, value) for name, value in module.named_buffers()]
    for name, child in module.named_modules():
        entries.append((name, "training", child.training))
        noise = getattr(child, "rand_noise", None)
        if isinstance(noise, torch.Tensor):
            entries.append(describe(f"{name}.rand_noise", noise))
    return tuple(entries)


def cpu_voice_digest(voice: dict[str, Any]) -> str | None:
    import torch

    from ..voice_transport import pack_voice_payload, voice_payload_digest

    if any(
        not isinstance(value, torch.Tensor) or value.device.type != "cpu"
        for value in voice.values()
    ):
        return None
    if (
        set(voice) != {"prompt_token", "prompt_feat", "embedding"}
        or voice["prompt_token"].dtype not in (torch.int32, torch.int64)
        or any(
            voice[name].dtype != torch.float32 for name in ("prompt_feat", "embedding")
        )
    ):
        return None
    return voice_payload_digest(pack_voice_payload(voice))


def runtime_environment(device: Any) -> dict[str, Any]:
    """Describe the effective codec scope, not its caller's cuDNN defaults."""
    import torch

    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": True,
    }


def _execution_context() -> tuple[Any, ...]:
    import torch

    priority = getattr(torch._C, "_get_sdp_priority_order", lambda: ())
    return (
        torch.backends.cudnn.enabled,
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
        tuple(priority()),
    )


class PrefixCacheBank:
    """One codec owns at most two immutable layer/step prefix-output banks."""

    prefix = 1280

    def __init__(
        self,
        flow: Any,
        voice_digest: str,
        shapes: tuple[int, ...],
        environment: dict[str, Any],
    ) -> None:
        from diffusers.models.attention_processor import Attention

        from .cosyvoice_flow.cfm_attention import MossCfmAttnProcessor

        self.flow = flow
        self._epoch_owner = None
        self.binding = CacheBinding(
            voice_digest, shapes, dict(environment), tensor_epoch(flow)
        )
        self.entries: dict[tuple[int, int, str], Any] = {}
        self.mode = "cached"
        self.active_length: int | None = None
        self.ledger: PrefixCallLedger | None = None
        self.hits = self.solves = self.fallbacks = 0
        self.context = _execution_context()
        self.modules = tuple(
            (name, module, module.processor)
            for name, module in flow.decoder.estimator.named_modules()
            if type(module) is Attention
            and type(module.processor) is MossCfmAttnProcessor
        )
        if len(self.modules) != 56:
            raise ValueError(
                "Prepared cache requires the 56 native MOSS attention layers"
            )
        self.names = tuple(name for name, _, _ in self.modules)
        bank = self

        class PreparedHandler:
            def __init__(self, name: str) -> None:
                self.name = name

            def shape_supported(self, length: int, chunk_size: int) -> bool:
                from .cosyvoice_flow.cfm_attention import shape_supported

                return shape_supported(length, chunk_size) or (
                    bank.active_length == length
                    and length in (1392, 1400, 1424)
                    and chunk_size == 100
                )

            def attention(
                self, query: Any, segments: tuple, plan: Any, mask: Any
            ) -> Any:
                from .cosyvoice_flow.cfm_attention import _segmented_attention

                if bank.active_length != plan.length or plan.chunk_size != 100:
                    return _segmented_attention(query, segments)
                return bank.attention(self.name, query, segments, plan, mask)

        for name, module, original in self.modules:
            # Decoder provenance intentionally accepts the exact local type.
            # An instance-owned handler preserves that trust boundary.
            module.set_processor(
                MossCfmAttnProcessor(
                    original.original, prefix_cache_handler=PreparedHandler(name)
                )
            )

    def install_epoch_owner(self, owner: Any) -> None:
        if (
            self.active_length is not None
            or getattr(self, "_epoch_owner", None) is not None
        ):
            raise RuntimeError("Active or already-bound epoch owner")
        if owner.flow is not self.flow or not owner.active or owner.closed:
            raise ValueError("Epoch owner is not qualified for this flow")
        self._epoch_owner = owner

    def current_epoch(self) -> tuple[Any, ...]:
        owner = getattr(self, "_epoch_owner", None)
        if owner is None:
            return tensor_epoch(self.flow)
        return owner.snapshot(self.flow, tensor_epoch)

    def scope(
        self, voice: str | None, length: int, final: bool, environment: dict[str, Any]
    ) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def scoped() -> Any:
            if self.active_length is not None:
                raise RuntimeError("CFM prefix cache scope is already active")
            epoch = self.current_epoch()
            if epoch != self.binding.epoch:
                self.binding.invalidate()
            eligible = (
                self.binding.matches(voice, length, final, environment, epoch)
                and self.context == _execution_context()
            )
            if eligible:
                self.active_length, self.ledger = length, PrefixCallLedger(self.names)
                self.solves += 1
            else:
                self.fallbacks += 1
            try:
                yield
                if eligible:
                    self.ledger.finish(allow_empty=self.mode == "cached")
            finally:
                self.active_length, self.ledger = None, None

        return scoped()

    def attention(
        self, name: str, query: Any, segments: tuple, plan: Any, mask: Any
    ) -> Any:
        import torch
        import torch.nn.functional as F

        from .cosyvoice_flow.cfm_attention import _segmented_attention

        step, label = self.ledger.next(name)
        key_id = (plan.length, step, label)
        _, key, value, _ = segments[-1]
        if self.mode == "build":
            # Nonstandard startup lengths keep the native full-SDPA oracle.
            result = (
                F.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
                )
                if plan.length in (1392, 1400, 1424)
                else _segmented_attention(query, segments)
            )
            if key_id in self.entries:
                raise RuntimeError("Duplicate prepared prefix entry")
            self.entries[key_id] = result[..., : self.prefix, :].clone()
            return result
        prefix = self.entries.get(key_id)
        if prefix is None:
            raise RuntimeError("Prepared prefix bank is incomplete")
        tail_q, tail_mask = query[..., self.prefix :, :], mask[..., self.prefix :, :]
        expected = int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
        if (
            torch._fused_sdp_choice(tail_q, key, value, tail_mask, 0.0, False)
            != expected
        ):
            self.fallbacks += 1
            return (
                F.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
                )
                if plan.length in (1392, 1400, 1424)
                else _segmented_attention(query, segments)
            )
        tail = F.scaled_dot_product_attention(
            tail_q, key, value, attn_mask=tail_mask, dropout_p=0.0, is_causal=False
        )
        result = torch.empty_strided(
            query.shape,
            (plan.length * 512, 64, 512, 1),
            dtype=query.dtype,
            device=query.device,
        )
        torch.cat((prefix, tail), dim=-2, out=result)
        self.hits += 1
        return result

    def status(self) -> dict[str, Any]:
        return dict(
            enabled=self.binding.valid,
            shapes=list(self.binding.shapes),
            entries=len(self.entries),
            bytes=sum(t.numel() * t.element_size() for t in self.entries.values()),
            attention_hits=self.hits,
            eligible_solves=self.solves,
            fallback_calls=self.fallbacks,
        )

    def close(self) -> None:
        if self.active_length is not None:
            raise RuntimeError("Cannot release an active CFM prefix bank")
        self.binding.invalidate()
        for _, module, original in self.modules:
            module.set_processor(original)
        self.entries.clear()
        self._epoch_owner = None


def build_prepared_manifest(
    *,
    voice_path: Path,
    wave_path: Path,
    codec_root: Path,
    source_root: Path,
    environment: dict[str, Any],
    warmup_audio_codes: list[int],
) -> dict[str, Any]:
    """Build a CPU receipt; the explicit caller writes it before codec startup."""
    import torch

    voice_path, wave_path, codec_root = (
        Path(voice_path).resolve(),
        Path(wave_path).resolve(),
        Path(codec_root).resolve(),
    )
    voice = torch.load(voice_path, map_location="cpu", weights_only=True)
    digest = cpu_voice_digest(voice)
    if digest is None or voice["prompt_token"].shape != (1, 342):
        raise ValueError(
            "Prepared manifest needs canonical CPU conditioning with 342 codes"
        )
    if len(warmup_audio_codes) != 36 or any(
        type(code) is not int or not 0 <= code < 16384 for code in warmup_audio_codes
    ):
        raise ValueError("Prepared calibration requires 36 valid audio codes")
    return dict(
        schema_version=1,
        prompt_codes=342,
        voice=dict(
            path=str(voice_path), sha256=_sha256(voice_path), payload_sha256=digest
        ),
        reference_wave=dict(path=str(wave_path), sha256=_sha256(wave_path)),
        codec_sha256={
            name: _sha256(codec_root / name)
            for name in ("flow/flow-chunk-25.pt", "flow/hift.pt")
        },
        source_sha256=source_fingerprint(Path(source_root)),
        environment=dict(environment),
        warmup_audio_codes=list(warmup_audio_codes),
    )
