# SPDX-License-Identifier: Apache-2.0
"""Precision and step count for the native Flow Euler solver."""

import torch


class FlowEulerDecoder(torch.nn.Module):
    """Keep the native noise source and solver using a time grid in the input precision."""

    def __init__(
        self, decoder: torch.nn.Module, *, terminal_flow_steps: int | None = None
    ) -> None:
        super().__init__()
        self.decoder: torch.nn.Module = decoder
        self.terminal_flow_steps: int | None = terminal_flow_steps

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
        steps = (
            self.terminal_flow_steps
            if not streaming and self.terminal_flow_steps is not None
            else n_timesteps
        )
        timesteps = torch.linspace(0, 1, steps + 1, device=mu.device, dtype=mu.dtype)
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
