# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned MOSS-Speech smoke assets and a consumed pipeline configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download

MODEL_REPOSITORY = "OpenMOSS-Team/MOSS-Speech"
MODEL_REVISION = "cff025bb41d8459d59abac0b5e44aba7f659ec9e"
CODEC_REPOSITORY = "OpenMOSS-Team/MOSS-Speech-Codec"
CODEC_REVISION = "eeec733e4e1dea7da444d332d8e1621ef257414c"
REFERENCE_REVISION = "1ea408a10d07b9fdc7a27bce19d1211b62067784"
AUDIO_SHA256 = {
    "en": "5c1541599a64decdbb16962034e213ad89ca32be096404ee6d523b7752ee0790",
    "cn": "cde3e98fdf0c90d4a578aa5858e1d4f36d99d3dd96d7ebb302b8bb976c3d12da",
}


def derive_audio_input(
    source_path: Path, destination_path: Path
) -> dict[str, str | int | float]:
    with wave.open(str(source_path), "rb") as source:
        channels = source.getnchannels()
        sample_width_bytes = source.getsampwidth()
        sample_rate_hz = source.getframerate()
        source_frames = source.getnframes()
        selected_frames = min(source_frames, 20 * sample_rate_hz)
        if selected_frames <= 0:
            raise ValueError(f"Reference WAV has no audio frames: {source_path}")
        else:
            pass
        samples = source.readframes(selected_frames)
        if len(samples) != selected_frames * channels * sample_width_bytes:
            raise ValueError(f"Reference WAV is truncated: {source_path}")
        else:
            pass
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination_path), "wb") as destination:
        destination.setparams(
            (
                channels,
                sample_width_bytes,
                sample_rate_hz,
                selected_frames,
                "NONE",
                "not compressed",
            )
        )
        destination.writeframes(samples)
    return {
        "source_filename": source_path.name,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_frames": source_frames,
        "source_duration_seconds": source_frames / sample_rate_hz,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "selected_start_frame": 0,
        "selected_end_frame": selected_frames,
        "derived_filename": destination_path.name,
        "derived_frames": selected_frames,
        "derived_duration_seconds": selected_frames / sample_rate_hz,
        "derived_sha256": hashlib.sha256(destination_path.read_bytes()).hexdigest(),
    }


def prepare_assets(output_directory: Path, gpu_count: int) -> dict[str, str]:
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = Path(snapshot_download(MODEL_REPOSITORY, revision=MODEL_REVISION))
    codec_path = Path(snapshot_download(CODEC_REPOSITORY, revision=CODEC_REVISION))
    model_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_filenames = set(model_index["weight_map"].values())
    if not weight_filenames:
        raise ValueError("Empty AR weight index")
    else:
        pass
    for directory, filenames in (
        (model_path, ["config.json", "tokenizer_config.json", *weight_filenames]),
        (
            codec_path,
            [
                "config.json",
                "model.safetensors",
                "flow/config.yaml",
                "flow/flow.pt",
                "flow/flow-chunk-25.pt",
                "flow/hift.pt",
                "flow/campplus.onnx",
            ],
        ),
    ):
        for filename in filenames:
            artifact_path = directory / filename
            if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Missing or empty checkpoint artifact: {artifact_path}"
                )
            else:
                pass
    input_audio_provenance = {}
    for language, expected_sha256 in AUDIO_SHA256.items():
        audio_path = output_directory / f"prompt-{language}.wav"
        url = f"https://raw.githubusercontent.com/OpenMOSS/MOSS-Speech/{REFERENCE_REVISION}/assets/{audio_path.name}"
        with urlopen(url, timeout=60) as response:
            audio_bytes = response.read()
        if hashlib.sha256(audio_bytes).hexdigest() != expected_sha256:
            raise ValueError(f"Reference WAV hash mismatch: {url}")
        else:
            pass
        audio_path.write_bytes(audio_bytes)
        input_audio_provenance[language] = derive_audio_input(
            audio_path, output_directory / f"input-{language}.wav"
        )
    codec_gpu = gpu_count - 1
    configuration = {
        "config_cls": "MossSpeechPipelineConfig",
        "model_path": str(model_path),
        "stages": {
            "preprocessing": {
                "gpu": codec_gpu,
                "factory": {
                    "codec_path": str(codec_path),
                    "voice_wav": str(output_directory / "prompt-cn.wav"),
                },
            },
            "ar_engine": {"gpu": 0},
            "audio_vocoder": {
                "gpu": codec_gpu,
                "factory": {"codec_path": str(codec_path)},
            },
        },
    }
    config_path = output_directory / "moss_speech.yaml"
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    manifest = {
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "snapshot": str(model_path),
        },
        "codec": {
            "repository": CODEC_REPOSITORY,
            "revision": CODEC_REVISION,
            "snapshot": str(codec_path),
        },
        "reference_git_revision": REFERENCE_REVISION,
        "reference_wav_sha256": AUDIO_SHA256,
        "input_audio": input_audio_provenance,
        "gpu_count": gpu_count,
    }
    (output_directory / "prepared_assets.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return {
        "MOSS_SPEECH_CI_CONFIG": str(config_path),
        "MOSS_SPEECH_CI_AUDIO_EN": str(output_directory / "input-en.wav"),
        "MOSS_SPEECH_CI_AUDIO_ZH": str(output_directory / "input-cn.wav"),
        "MOSS_SPEECH_CI_OUTPUT_DIR": str(output_directory / "results"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--github-env", type=Path)
    arguments = parser.parse_args()
    environment = prepare_assets(arguments.output_directory, arguments.gpu_count)
    (arguments.output_directory / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    if arguments.github_env is not None:
        with arguments.github_env.open("a") as environment_file:
            for name, value in environment.items():
                if "\n" in value or "\r" in value:
                    raise ValueError(f"Invalid environment path for {name}")
                else:
                    pass
                environment_file.write(f"{name}={value}\n")
    else:
        pass
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
