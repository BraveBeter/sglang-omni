# SPDX-License-Identifier: Apache-2.0
"""Independent chat-native MOSS-Speech CI preset; no TTS endpoint assumptions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MossSpeechCiPreset:
    config_cls: str = "MossSpeechPipelineConfig"
    model_name: str = "moss-speech"
    endpoint: str = "/v1/chat/completions"
    startup_timeout: int = 600
    request_timeout: int = 600
    max_new_tokens: int = 512
    client_concurrency: int = 1
    gate_thresholds: bool = False
    calibrated: bool = False
    quality_thresholds: dict[str, float] | None = None
    calibration_report: str | None = None

    def __post_init__(self) -> None:
        if self.gate_thresholds and not (
            self.calibrated and self.quality_thresholds and self.calibration_report
        ):
            raise ValueError("Quality gates require actual CI calibration and evidence")
