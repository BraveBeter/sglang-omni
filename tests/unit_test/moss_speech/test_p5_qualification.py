"""Qualification must not hide missing samples or uncalibrated quality gates."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.moss_speech.p5.quality import build_cases, error_counts, validate_pairs
from tests.test_model.moss_speech_ci_config import MossSpeechCiPreset


def test_quality_thresholds_cannot_be_enabled_without_calibration() -> None:
    assert not MossSpeechCiPreset().gate_thresholds
    with pytest.raises(ValueError, match="calibrat"):
        MossSpeechCiPreset(gate_thresholds=True)


def test_audio_assets_are_verified_before_generation(tmp_path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fixture")
    sample = dict(
        id="en_00",
        source_id="source",
        lang="en",
        source_audio=audio.name,
        source_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
        source_text="Source",
        target_text="Target",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [sample]}))
    cases = build_cases(manifest)
    assert {case["mode"] for case in cases} == {"t2t", "t2s", "s2t", "s2s"}
    assert cases[1]["body"]["modalities"] == ["audio"]
    assert cases[2]["expected_text"] == "Source"
    assert cases[3]["expected_text"] is None
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        build_cases(manifest)


@pytest.mark.parametrize(
    "lang,ref,hyp,numerator,denominator",
    [
        ("en", "red green blue", "red yellow", 2, 3),
        ("zh", "你好世界", "你好", 2, 4),
        ("en", "hello", "", 1, 1),
    ],
)
def test_corpus_counts_include_deletions(
    lang, ref, hyp, numerator, denominator
) -> None:
    result = error_counts(ref, hyp, lang)
    assert result["errors"] == numerator
    assert result["reference_units"] == denominator


def test_missing_or_divergent_outputs_fail_pair_gate() -> None:
    row = dict(
        id="a", input_grid=[[1, 512]], grid=[[151645, 2]], text="", finish_reason="stop"
    )
    assert validate_pairs([row], [dict(row)], ["a"])["pass"]
    assert not validate_pairs([row], [], ["a"])["pass"]
    assert not validate_pairs([row], [dict(row, grid=[[151645, 3]])], ["a"])["pass"]
    assert not validate_pairs([row], [row, row], ["a"])["pass"]


def test_chat_contract_import_does_not_load_sglang() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['sglang'] = None; "
            "from sglang_omni.models.moss_speech import request_builders; "
            "assert 'sglang_omni.models.moss_speech.request_data' not in sys.modules",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_24gb_probe_sets_engine_budget_not_only_placement(
    tmp_path, monkeypatch
) -> None:
    import sys

    import yaml

    from scripts.moss_speech.p5.make_config import main

    model = tmp_path / "ar"
    codec = tmp_path / "codec"
    model.mkdir()
    codec.mkdir()
    (model / "config.json").write_text("{}")
    (codec / "config.json").write_text("{}")
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"asset")
    output = tmp_path / "config.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_config",
            "--model-path",
            str(model),
            "--codec-path",
            str(codec),
            "--voice-wav",
            str(voice),
            "--runtime-dir",
            str(tmp_path / "runtime"),
            "--output",
            str(output),
            "--probe-24gb",
        ],
    )
    main()
    data = yaml.safe_load(output.read_text())
    ar = next(s for s in data["stages"] if s["name"] == "ar_engine")
    overrides = ar["factory_args"]["server_args_overrides"]
    assert overrides["mem_fraction_static"] == 0.80
    assert overrides["max_running_requests"] == 1
    assert overrides["max_total_tokens"] == 1024
    assert all(
        s["factory_args"].get("codec_path") == str(codec)
        for s in data["stages"]
        if s["name"] != "text_decode"
    )


def test_versioned_text_profile_keeps_hf_output_separate():
    from scripts.moss_speech.p5.quality import comparison_text, reference_text_fields

    tokenizer = type(
        "Tokenizer",
        (),
        {
            "decode": lambda self, ids, **k: "".join(
                {1: "hello", 2: " world", 3: "\ufffd", 151645: ""}[i] for i in ids
            )
        },
    )()
    for grid, expected, upstream in [
        ([[1, 0]], "hello", ""),
        ([[1, 0], [2, 0]], "hello world", "hello"),
        ([[1, 0], [151645, 0]], "hello", "hello"),
        ([[1, 0], [3, 0]], "hello", "hello"),
    ]:
        fields = reference_text_fields(tokenizer, grid, audio_output=False)
        assert fields["text"] == upstream
        assert comparison_text(fields) == expected
        row = dict(
            id="length",
            input_grid=[[10, 512]],
            grid=grid,
            finish_reason="length",
            **fields,
        )
        native = dict(row, text=expected)
        native.pop("service_text")
        assert validate_pairs([row], [native], ["length"])["pass"]
        wrong_grid = dict(native, grid=[[99, 0]])
        result = validate_pairs([row], [wrong_grid], ["length"])
        assert not result["pass"] and not result["grid_cases"]["length"]
    legacy = dict(
        id="a", input_grid=[[10, 0]], grid=[[1, 0]], text="", finish_reason="length"
    )
    # No silent migration of frozen unversioned references.
    assert not validate_pairs([legacy], [dict(legacy, text="hello")], ["a"])["pass"]
    assert not validate_pairs(
        [legacy], [dict(legacy, text_decode_profile="unknown")], ["a"]
    )["pass"]
    with pytest.raises(ValueError, match="profile"):
        comparison_text(dict(text="x", text_decode_profile="unknown"))


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([1], "hello"),
        ([1, 2], "hello world"),
        ([1, 151645], "hello"),
        ([1, 3], "hello"),
    ],
)
def test_terminal_text_preserves_complete_generated_prefix(monkeypatch, ids, expected):
    from types import SimpleNamespace

    from sglang_omni.models.moss_speech import stages
    from sglang_omni.models.moss_speech.payload_types import MossSpeechState
    from sglang_omni.proto.request import StagePayload

    tokenizer = SimpleNamespace(
        decode=lambda ids, **k: "".join(
            {1: "hello", 2: " world", 3: "\ufffd", 151645: ""}[i] for i in ids
        )
    )
    monkeypatch.setattr(stages, "_load_tokenizer", lambda _: tokenizer)
    payload = StagePayload(
        request_id="length",
        request=SimpleNamespace(),
        data=MossSpeechState(
            output_modality="text",
            output_grid=[[i, 512] for i in ids],
            finish_reason="length",
        ).to_dict(),
    )
    assert stages.create_text_decode_executor(".")._fn(payload).data["text"] == expected
