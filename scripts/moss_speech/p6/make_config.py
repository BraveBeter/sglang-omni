#!/usr/bin/env python3
"""Render the explicit streaming variant with local model, codec and voice assets."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sglang_omni.models.moss_speech.config import MossSpeechStreamingPipelineConfig


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("model-path", "codec-path", "voice-wav", "runtime-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--chunk-size", type=int, choices=[5, 25], default=5)
    a = p.parse_args()
    for path in (
        a.model_path / "config.json",
        a.codec_path / "flow" / f"flow-chunk-{a.chunk_size}.pt",
        a.codec_path / "flow/flow.pt",
        a.codec_path / "flow/hift.pt",
        a.voice_wav,
    ):
        if not path.is_file():
            p.error(f"Required local asset missing: {path}")
    config = MossSpeechStreamingPipelineConfig(model_path=str(a.model_path.resolve()))
    config.endpoints.base_path = str(a.runtime_dir.resolve())
    config.env_defaults["MOSS_SPEECH_VOICE_WAV"] = str(a.voice_wav.resolve())
    for stage in config.stages:
        if stage.name != "text_decode":
            stage.factory_args["codec_path"] = str(a.codec_path.resolve())
        if stage.name == "audio_vocoder":
            stage.factory_args["chunk_size"] = a.chunk_size
    data = config.model_dump(mode="json")
    data["config_cls"] = type(config).__name__
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
