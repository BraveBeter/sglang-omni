#!/usr/bin/env python3
"""Score all frozen cases after generator shutdown using independent Whisper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from quality import build_cases, error_counts, sha256, validate_pairs


def acoustic_stats(path: Path) -> dict[str, Any]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    finite = bool(wav.size and np.isfinite(wav).all())
    return dict(
        valid=finite and sr == 24000 and wav.shape[1] == 1,
        sample_rate=sr,
        samples=len(wav),
        duration_s=len(wav) / sr,
        rms=float(np.sqrt(np.mean(wav**2))) if finite else None,
        peak=float(np.abs(wav).max()) if finite else None,
        clipping_fraction=float(np.mean(np.abs(wav) >= 0.999)) if finite else None,
        sha256=sha256(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--asr-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = build_cases(args.manifest)
    reference = json.loads(args.reference.read_text())
    native = json.loads((args.native_dir / "report.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        pass_=False,
        gate_thresholds=False,
        calibrated=False,
        human_mos=None,
        manifest_sha256=sha256(args.manifest),
        reference_sha256=sha256(args.reference),
        native_sha256=sha256(args.native_dir / "report.json"),
        asr_sha256=sha256(args.asr_checkpoint),
        latency_scope={
            "native": "HTTP end-to-end",
            "reference": "HF encode and AR, excluding P1 decode",
        },
        results=[],
        summary={},
    )
    report["pass"] = report.pop("pass_")
    try:
        import whisper

        torch.set_num_threads(1)
        model = whisper.load_model(str(args.asr_checkpoint), device="cuda")
        report["gpu"] = torch.cuda.get_device_name(0)
        transcript_cache = {}

        def transcribe(path: Path, lang: str) -> str:
            key = (sha256(path), lang)
            if key not in transcript_cache:
                output = model.transcribe(
                    str(path),
                    language=lang,
                    temperature=0.0,
                    beam_size=5,
                    fp16=False,
                    condition_on_previous_text=False,
                    verbose=False,
                )
                transcript_cache[key] = output["text"]
            return transcript_cache[key]

        ref = {x["id"]: x for x in reference["results"]}
        nat = {x["id"]: x for x in native["results"]}
        report["pairs"] = validate_pairs(
            reference["results"], native["results"], [c["id"] for c in cases]
        )
        for case in cases:
            rid = case["id"]
            row = dict(id=rid, mode=case["mode"], lang=case["lang"])
            try:
                for name, result in (("reference", ref[rid]), ("native", nat[rid])):
                    if result.get("error"):
                        raise RuntimeError(result["error"])
                    text = result["text"]
                    row[name] = dict(
                        finish_reason=result["finish_reason"], seconds=result["seconds"]
                    )
                    if case["mode"].endswith("s"):
                        wav = (
                            args.native_dir / "reference_wav"
                            if name == "reference"
                            else args.native_dir
                        ) / f"{rid}.wav"
                        row[name]["acoustics"] = acoustic_stats(wav)
                        assert row[name]["acoustics"]["valid"]
                        text = transcribe(wav, case["lang"])
                        if name == "native":
                            row[name]["rtf"] = (
                                result["seconds"] / row[name]["acoustics"]["duration_s"]
                            )
                    row[name]["transcript"] = text
                    if case["expected_text"] is not None:
                        row[name]["annotation_error"] = error_counts(
                            case["expected_text"], text, case["lang"]
                        )
                row["reference_native_error"] = error_counts(
                    row["reference"]["transcript"],
                    row["native"]["transcript"],
                    case["lang"],
                )
            except Exception as exc:
                row["error"] = repr(exc)
            report["results"].append(row)
            print(rid, row.get("error", "scored"), flush=True)
            (args.out_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        for mode in ("t2t", "t2s", "s2t", "s2s"):
            for lang in ("en", "zh"):
                rows = [
                    r
                    for r in report["results"]
                    if r["mode"] == mode and r["lang"] == lang
                ]
                stats = dict(
                    total=len(rows),
                    failed=sum(bool(r.get("error")) for r in rows),
                    metric="CER" if lang == "zh" else "WER",
                )
                for name in ("reference", "native"):
                    valid = [r[name] for r in rows if not r.get("error")]
                    errors = [r.get("annotation_error") for r in valid]
                    counted = [r for r in errors if r is not None]
                    units = sum(r["reference_units"] for r in counted)
                    numerator = sum(r["errors"] for r in counted)
                    stats[name] = dict(
                        scored=len(valid),
                        errors=numerator if counted else None,
                        reference_units=units if counted else None,
                        rate=numerator / units if units else None,
                        truncated=sum(r["finish_reason"] == "length" for r in valid),
                    )
                report["summary"][f"{mode}_{lang}"] = stats
        report["asr_unique_waveform_language_pairs"] = len(transcript_cache)
        report["pass"] = (
            reference["pass"]
            and native["pass"]
            and report["pairs"]["pass"]
            and all(not r.get("error") for r in report["results"])
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
