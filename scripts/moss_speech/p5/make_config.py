#!/usr/bin/env python3
"""Render an offline config with explicit asset paths; no remote model loading."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--codec-path", required=True, type=Path)
    parser.add_argument("--voice-wav", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-24gb", action="store_true")
    args = parser.parse_args()
    for path in (
        args.model_path / "config.json",
        args.codec_path / "config.json",
        args.voice_wav,
    ):
        if not path.is_file():
            parser.error(f"Required local asset is missing: {path}")
    config = MossSpeechPipelineConfig(model_path=str(args.model_path.resolve()))
    config.endpoints.base_path = str(args.runtime_dir.resolve())
    config.env_defaults["MOSS_SPEECH_VOICE_WAV"] = str(args.voice_wav.resolve())
    for stage in config.stages:
        if stage.name != "text_decode":
            stage.factory_args["codec_path"] = str(args.codec_path.resolve())
        if args.probe_24gb and stage.name == "ar_engine":
            stage.runtime.resources.total_gpu_memory_fraction = 0.80
            stage.factory_args["server_args_overrides"].update(
                max_running_requests=1,
                max_queued_requests=2,
                max_total_tokens=1024,
                mem_fraction_static=0.80,
            )
    output = config.model_dump(mode="json")
    output["config_cls"] = "MossSpeechPipelineConfig"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
