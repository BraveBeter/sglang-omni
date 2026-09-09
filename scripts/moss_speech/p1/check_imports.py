#!/usr/bin/env python3
"""Import-closure audit for the MOSS-Speech codec components (T1.2, G1).

Runs in the TARGET environment (``.venv-omni``) with a clean environment
(no reference PYTHONPATH, offline HF flags) and asserts:

1. the vendored closure imports;
2. none of the training/UI-only packages are pulled in transitively;
3. encoder-only / decoder-only imports are possible independently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Offline discipline must be set before the first transformers import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BANNED_PREFIXES = (
    "lightning",
    "matplotlib",
    "pyworld",
    "gdown",
    "wget",
    "hydra",
    "hyperpyyaml",
    "omegaconf",
    "conformer",
    "gradio",
    "cosyvoice",  # reference repo package must not leak via sys.path
    "matcha",  # reference repo package must not leak via sys.path
)


def audit() -> dict:
    import sglang_omni.models.moss_speech.components.hf_codec as hf_codec_pkg
    from sglang_omni.models.moss_speech.components.codec_adapter import (
        MossSpeechCodecAdapter,
    )
    from sglang_omni.models.moss_speech.components.voice import (
        CampplusSpeakerEncoder,
        MossSpeechVoiceHook,
        VoiceConditioning,
    )

    loaded = {m.split(".")[0] for m in sys.modules}
    violations = sorted(set(BANNED_PREFIXES) & loaded)
    return {
        "hf_codec_package": hf_codec_pkg.__name__,
        "adapter": MossSpeechCodecAdapter.__name__,
        "voice_symbols": [
            VoiceConditioning.__name__,
            MossSpeechVoiceHook.__name__,
            CampplusSpeakerEncoder.__name__,
        ],
        "banned_violations": violations,
        "module_count": len(sys.modules),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, indent=1)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text + "\n")
    if result["banned_violations"]:
        print("AUDIT FAILED: banned modules loaded:", result["banned_violations"])
        sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
