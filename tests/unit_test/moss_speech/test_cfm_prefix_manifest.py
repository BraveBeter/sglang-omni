"""Prepared startup conditioning binds voice bytes, source, weights and runtime."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from sglang_omni.models.moss_speech.components.cfm_prefix_cache import (
    load_prepared_voice,
)
from sglang_omni.models.moss_speech.voice_transport import (
    pack_voice_payload,
    voice_payload_digest,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared(tmp_path):
    voice = dict(
        prompt_token=torch.zeros(1, 342, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 1368, 80),
        embedding=torch.zeros(1, 192),
    )
    voice_path = tmp_path / "voice.pt"
    torch.save(voice, voice_path)
    wave = tmp_path / "reference.wav"
    wave.write_bytes(b"RIFF-test-reference")
    codec = tmp_path / "codec"
    (codec / "flow").mkdir(parents=True)
    for name in ("flow-chunk-25.pt", "hift.pt"):
        (codec / "flow" / name).write_bytes(name.encode())
    source = tmp_path / "source"
    source.mkdir()
    (source / "decoder.py").write_text("pass\n")
    environment = dict(
        torch="test-build",
        cuda="test-cuda",
        device_name="test-device",
        capability=[8, 0],
        matmul_allow_tf32=False,
        cudnn_allow_tf32=True,
        cudnn_deterministic=True,
    )
    manifest = dict(
        schema_version=1,
        prompt_codes=342,
        voice=dict(
            path="voice.pt",
            sha256=digest(voice_path),
            payload_sha256=voice_payload_digest(pack_voice_payload(voice)),
        ),
        reference_wave=dict(path="reference.wav", sha256=digest(wave)),
        codec_sha256={
            f"flow/{name}": digest(codec / "flow" / name)
            for name in ("flow-chunk-25.pt", "hift.pt")
        },
        source_sha256={"decoder.py": digest(source / "decoder.py")},
        environment=environment,
        warmup_audio_codes=list(range(36)),
    )
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(manifest))
    return (
        path,
        manifest,
        dict(codec_root=codec, source_root=source, environment=environment),
    )


def rewrite(prepared):
    path, manifest, _ = prepared
    path.write_text(json.dumps(manifest))


def test_complete_manifest_loads_cpu_voice_and_identity(prepared):
    path, manifest, kwargs = prepared
    result = load_prepared_voice(path, **kwargs)
    assert result.voice_digest == manifest["voice"]["payload_sha256"]
    assert result.manifest_sha256 == digest(path)
    assert result.warmup_audio_codes == tuple(range(36))
    assert all(value.device.type == "cpu" for value in result.voice.values())


@pytest.mark.parametrize(
    "name",
    [
        "voice.pt",
        "reference.wav",
        "codec/flow/flow-chunk-25.pt",
        "codec/flow/hift.pt",
        "source/decoder.py",
    ],
)
def test_changed_file_rejects_manifest(prepared, name):
    path, _, kwargs = prepared
    target = path.parent / name
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        load_prepared_voice(path, **kwargs)


def test_changed_voice_payload_digest_is_rejected(prepared):
    path, manifest, kwargs = prepared
    manifest["voice"]["payload_sha256"] = "0" * 64
    rewrite(prepared)
    with pytest.raises(ValueError):
        load_prepared_voice(path, **kwargs)


def test_runtime_and_complete_source_set_are_bound(prepared):
    path, _, kwargs = prepared
    with pytest.raises(ValueError):
        load_prepared_voice(path, **dict(kwargs, environment={"torch": "different"}))
    (kwargs["source_root"] / "new.py").write_text("pass")
    with pytest.raises(ValueError):
        load_prepared_voice(path, **kwargs)


@pytest.mark.parametrize(
    "codes", [[0] * 35, [0] * 35 + [16384], [0] * 35 + [-1], [0] * 35 + [True]]
)
def test_calibration_codes_must_be_complete_valid_audio(prepared, codes):
    path, manifest, kwargs = prepared
    manifest["warmup_audio_codes"] = codes
    rewrite(prepared)
    with pytest.raises(ValueError):
        load_prepared_voice(path, **kwargs)


def test_missing_required_codec_file_is_rejected(prepared):
    path, manifest, kwargs = prepared
    del manifest["codec_sha256"]["flow/hift.pt"]
    rewrite(prepared)
    with pytest.raises(ValueError):
        load_prepared_voice(path, **kwargs)


def test_non_cpu_voice_tensor_is_rejected(prepared):
    path, manifest, kwargs = prepared
    voice = dict(
        prompt_token=torch.zeros(1, 342, dtype=torch.int32, device="meta"),
        prompt_feat=torch.zeros(1, 1368, 80),
        embedding=torch.zeros(1, 192),
    )
    torch.save(voice, path.parent / "voice.pt")
    manifest["voice"]["sha256"] = digest(path.parent / "voice.pt")
    rewrite(prepared)
    with pytest.raises(ValueError):
        load_prepared_voice(path, **kwargs)


def test_manifest_builder_binds_exact_inputs_and_roundtrips(prepared):
    from sglang_omni.models.moss_speech.components.cfm_prefix_cache import (
        build_prepared_manifest,
    )

    path, original, kwargs = prepared
    result = build_prepared_manifest(
        voice_path=path.parent / "voice.pt",
        wave_path=path.parent / "reference.wav",
        codec_root=kwargs["codec_root"],
        source_root=kwargs["source_root"],
        environment=kwargs["environment"],
        warmup_audio_codes=list(range(36)),
    )
    path.write_text(json.dumps(result))
    restored = load_prepared_voice(path, **kwargs)
    assert restored.voice_digest == original["voice"]["payload_sha256"]
    assert restored.warmup_audio_codes == tuple(range(36))
