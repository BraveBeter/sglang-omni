#!/usr/bin/env python3
"""Shared fixed-manifest chat benchmark and strict adaptation comparisons."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

MODES = ("t2t", "t2s", "s2t", "s2s")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cases(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    cases = []
    seen = set()
    for sample in manifest["samples"]:
        sid = sample["id"]
        if sid in seen or sample["lang"] not in ("en", "zh"):
            raise ValueError("Duplicate sample or unsupported language")
        seen.add(sid)
        audio = (manifest_path.parent / sample["source_audio"]).resolve()
        if sha256(audio) != sample["source_sha256"]:
            raise ValueError(f"Audio SHA mismatch: {sid}")
        for mode in MODES:
            audio_out = mode.endswith("s")
            speech_in = mode.startswith("s")
            system = (
                "You are a helpful voice assistant. Answer the user's questions with spoken responses."
                if audio_out
                else "You are a helpful assistant. Answer the user's questions with text."
            )
            content: Any = (
                "Repeat the following text verbatim, without adding anything: "
                + sample["target_text"]
            )
            if speech_in:
                content = [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio.read_bytes()).decode(),
                            "format": "wav",
                        },
                    }
                ]
            if mode == "s2t":
                system = "Transcribe the user's audio verbatim in its original language. Output only the transcription."
            body = dict(
                model="moss-speech",
                messages=[
                    dict(role="system", content=system),
                    dict(role="user", content=content),
                ],
                modalities=["audio" if audio_out else "text"],
                stream=False,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.1,
                seed=0,
                max_tokens=512,
            )
            hf_messages = [
                dict(role="system", content=system),
                dict(
                    role="user",
                    content={"path": str(audio)} if speech_in else content,
                ),
            ]
            cases.append(
                dict(
                    id=f"{sid}_{mode}",
                    source_id=sample.get("source_id"),
                    lang=sample["lang"],
                    mode=mode,
                    body=body,
                    hf_messages=hf_messages,
                    expected_text=(
                        (
                            sample["source_text"]
                            if mode == "s2t"
                            else sample["target_text"]
                        )
                        if mode != "s2s"
                        else None
                    ),
                )
            )
    if not cases:
        raise ValueError("Empty qualification manifest")
    return cases


def error_counts(reference: str, hypothesis: str, lang: str) -> dict[str, Any]:
    from jiwer import process_words

    from benchmarks.tasks.asr import normalize_text

    ref = normalize_text(reference, lang)
    hyp = normalize_text(hypothesis, lang)
    counts = process_words(ref, hyp)
    errors = counts.substitutions + counts.deletions + counts.insertions
    units = counts.hits + counts.substitutions + counts.deletions
    return dict(
        errors=errors,
        reference_units=units,
        rate=errors / units if units else None,
        reference_normalized=ref,
        hypothesis_normalized=hyp,
    )


# Versioned integration semantics, deliberately independent of native decoding.
TEXT_DECODE_PROFILE = "moss-full-grid-unicode-prefix-v1"


def reference_text_fields(tokenizer: Any, grid: list, *, audio_output: bool) -> dict:
    """Keep the upstream result intact and export service semantics separately."""

    def decode(rows: list) -> str:
        return (
            tokenizer.decode([row[0] for row in rows], skip_special_tokens=True)
            .replace("<|empty|>", ".")
            .replace("<|end_empty|>", ":")
        )

    return dict(
        text="" if audio_output else decode(grid[:-1]),
        service_text="" if audio_output else decode(grid).rstrip("\ufffd"),
        text_decode_profile=TEXT_DECODE_PROFILE,
    )


def comparison_text(row: dict) -> str:
    """Legacy reports keep legacy semantics; unknown profiles cannot pass."""
    profile = row.get("text_decode_profile")
    if profile is None:
        return row["text"]
    if profile != TEXT_DECODE_PROFILE:
        raise ValueError(f"Unknown text decode profile: {profile!r}")
    return row["service_text"] if "service_text" in row else row["text"]


def validate_pairs(
    reference: list[dict], native: list[dict], ids: list[str]
) -> dict[str, Any]:
    ref = {x["id"]: x for x in reference}
    nat = {x["id"]: x for x in native}
    coverage = len(ref) == len(reference) == len(ids) == len(native) == len(
        nat
    ) and set(ref) == set(nat) == set(ids)
    checks = {}
    grid_checks = {}
    text_checks = {}
    for rid in ids:
        a, b = ref.get(rid, {}), nat.get(rid, {})
        valid = bool(a and b and not a.get("error") and not b.get("error"))
        grid_checks[rid] = valid and all(
            key in a and key in b and a[key] == b[key]
            for key in ("input_grid", "grid", "finish_reason")
        )
        try:
            for row in (a, b):
                if row.get("text_decode_profile") not in (None, TEXT_DECODE_PROFILE):
                    raise ValueError("Unknown text decode profile")
            # Frozen, unversioned references retain the old strict comparison.
            text_checks[rid] = valid and (
                a["text"] == b["text"]
                if not a.get("text_decode_profile")
                else a.get("text_decode_profile") == b.get("text_decode_profile")
                and comparison_text(a) == comparison_text(b)
            )
        except (KeyError, ValueError):
            text_checks[rid] = False
        checks[rid] = grid_checks[rid] and text_checks[rid]
    return {
        "pass": coverage and bool(checks) and all(checks.values()),
        "coverage": coverage,
        "cases": checks,
        "grid_cases": grid_checks,
        "text_cases": text_checks,
    }
