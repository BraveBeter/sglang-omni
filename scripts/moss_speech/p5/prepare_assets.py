#!/usr/bin/env python3
"""Stage a pinned bilingual SeedTTS subset and a checksummed ASR checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--asr-dir", required=True)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
    from benchmarks.dataset.seedtts import load_seedtts_samples

    manifest = {
        "dataset": SEEDTTS_DATASET_ID,
        "revision": SEEDTTS_DATASET_REVISION,
        "selection": f"first {args.count} rows in each split, no filtering",
        "samples": [],
    }
    for lang in ("en", "zh"):
        samples = load_seedtts_samples(
            SEEDTTS_DATASET_ID,
            args.count,
            split=lang,
            revision=SEEDTTS_DATASET_REVISION,
        )
        assert len(samples) == args.count
        for i, sample in enumerate(samples):
            filename = f"{lang}_{i:02d}.wav"
            shutil.copyfile(sample.ref_audio, out / filename)
            manifest["samples"].append(
                {
                    "id": f"{lang}_{i:02d}",
                    "source_id": sample.sample_id,
                    "lang": lang,
                    "source_audio": filename,
                    "source_sha256": hashlib.sha256(
                        (out / filename).read_bytes()
                    ).hexdigest(),
                    "source_text": sample.ref_text,
                    "target_text": sample.target_text,
                }
            )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    import whisper

    asr_dir = Path(args.asr_dir).resolve()
    asr_dir.mkdir(parents=True, exist_ok=True)
    url = whisper._MODELS["small"]
    checkpoint = Path(whisper._download(url, str(asr_dir), in_memory=False))
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha == url.split("/")[-2]
    (out / "asr.json").write_text(
        json.dumps(
            {
                "name": "openai-whisper small",
                "url": url,
                "sha256": sha,
                "checkpoint": str(checkpoint),
                "package": "openai-whisper==20250625",
                "decoding": {"temperature": 0.0, "beam_size": 5, "fp16": False},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Staged {len(manifest['samples'])} samples and {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
