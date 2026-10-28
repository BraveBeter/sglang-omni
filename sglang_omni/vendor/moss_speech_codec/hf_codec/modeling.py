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
# Modified for SGLang-Omni inference; see ../VENDORED_SOURCES.md.

"""Explicit eager construction of the codec flow and HiFT checkpoint trees."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile
import torch
from torch import nn

from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow import (
    CausalMaskedDiffWithXvec,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow_matching import (
    CausalConditionalCFM,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.hifigan_f0_predictor import (
    ConvRNNF0Predictor,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.hifigan_generator import (
    HiFTGenerator,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.transformer.upsample_encoder import (
    UpsampleConformerEncoder,
)

OFFLINE_FLOW_CHUNK_SIZE = 5


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    samples, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(samples.copy()).T, sample_rate


@contextmanager
def scoped_global_rng() -> Iterator[None]:
    """Restore caller RNG state around deterministic model construction or decoding."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            pass


def build_flow_decoder(
    *,
    chunk_size: int,
    token_frame_rate: float = 12.5,
    token_mel_ratio: int = 4,
) -> CausalMaskedDiffWithXvec:
    """Construct the fixed checkpoint module tree without evaluating YAML."""
    cfm_params = SimpleNamespace(
        sigma_min=1e-06,
        solver="euler",
        t_scheduler="cosine",
        training_cfg_rate=0.2,
        inference_cfg_rate=0.7,
        reg_loss_type="l1",
    )
    estimator = CausalConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
        static_chunk_size=chunk_size * token_mel_ratio,
        num_decoding_left_chunks=-1,
    )
    decoder = CausalConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=cfm_params,
        estimator=estimator,
    )
    encoder = UpsampleConformerEncoder(
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        input_size=512,
        upsample_stride=4,
        use_cnn_module=False,
        macaron_style=False,
        static_chunk_size=chunk_size,
    )
    flow = CausalMaskedDiffWithXvec(
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=20480,
        input_frame_rate=token_frame_rate,
        only_mask_loss=True,
        token_mel_ratio=token_mel_ratio,
        pre_lookahead_len=3,
        encoder=encoder,
        decoder=decoder,
    )
    return flow


def build_hift() -> HiFTGenerator:
    return HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
    )


class AudioDecoder(nn.Module):
    """Decode a complete code sequence using tensor-native voice conditioning."""

    def __init__(
        self,
        flow_ckpt_path: str | Path,
        hift_ckpt_path: str | Path,
        *,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        with scoped_global_rng():
            self.flow = build_flow_decoder(chunk_size=OFFLINE_FLOW_CHUNK_SIZE)
            self.flow.load_state_dict(
                torch.load(flow_ckpt_path, map_location=self.device, weights_only=True),
                strict=False,
            )
            self.hift = build_hift()
            self.hift.load_state_dict(
                torch.load(hift_ckpt_path, map_location=self.device, weights_only=True),
                strict=True,
            )
        self.flow.to(self.device).eval()
        self.hift.to(self.device).eval()

    @torch.inference_mode()
    def decode(
        self, audio_token: torch.Tensor, voice: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        prompt_token = voice["prompt_token"]
        prompt_mel = voice["prompt_feat"]
        predicted_mel, _ = self.flow.inference(
            token=audio_token.to(self.device),
            token_len=torch.tensor(
                [audio_token.shape[1]], dtype=torch.int32, device=self.device
            ),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor(
                [prompt_token.shape[1]], dtype=torch.int32, device=self.device
            ),
            prompt_feat=prompt_mel.to(self.device),
            prompt_feat_len=torch.tensor(
                [prompt_mel.shape[1]], dtype=torch.int32, device=self.device
            ),
            embedding=voice["embedding"].to(self.device),
            streaming=False,
            finalize=True,
        )
        waveform, _ = self.hift.inference(
            speech_feat=predicted_mel, cache_source=torch.zeros(1, 1, 0)
        )
        return waveform
