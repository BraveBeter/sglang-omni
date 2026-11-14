# SPDX-License-Identifier: Apache-2.0
"""BF16 diffusion inputs for the native Flow Euler solver."""

import torch


class BF16FlowDecoder(torch.nn.Module):
    """Keep the native noise source and solver while using a BF16 time grid."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder: torch.nn.Module = decoder

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, None]:
        noise = self.decoder.rand_noise[:, :, : mu.size(2)].to(mu) * temperature
        timesteps = torch.linspace(
            0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype
        )
        if self.decoder.t_scheduler == "cosine":
            timesteps = 1 - torch.cos(timesteps * 0.5 * torch.pi)
        else:
            pass
        return (
            self.decoder.solve_euler(
                noise, timesteps, mu, mask, spks, cond, streaming=streaming
            ),
            None,
        )
