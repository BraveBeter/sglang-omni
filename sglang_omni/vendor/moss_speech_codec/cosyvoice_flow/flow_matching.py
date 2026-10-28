# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
import torch

from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.utils_common import (
    set_all_random_seed,
)
from sglang_omni.vendor.moss_speech_codec.matcha_components.flow_matching_base import (
    BASECFM,
)


class ConditionalCFM(BASECFM):
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: torch.nn.Module = None,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)


class CausalConditionalCFM(ConditionalCFM):
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: torch.nn.Module = None,
    ):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        streaming=False,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self.rand_noise[:, :, : mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return (
            self.solve_euler(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                streaming=streaming,
            ),
            None,
        )
