# coding=utf-8
# Copyright 2025 OpenMOSS and HuggingFace Inc. teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers.configuration_utils import PretrainedConfig


class MossSpeechCodecConfig(PretrainedConfig):
    """Lightweight configuration for MossSpeech codec.

    This config is intentionally minimal since the codec assembles a Whisper-VQ
    encoder and a Flow/HiFT decoder from their own configs and checkpoints.
    """

    model_type = "moss_speech_codec"

    def __init__(self, sample_rate: int = 16000, return_dict: bool = True, **kwargs):
        self.sample_rate = int(sample_rate)
        self.return_dict = bool(return_dict)
        super().__init__(**kwargs)


__all__ = ["MossSpeechCodecConfig"]

