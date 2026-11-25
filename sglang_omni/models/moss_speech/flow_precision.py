# SPDX-License-Identifier: Apache-2.0
"""Time grids for the native Flow Euler solver."""

from typing import NamedTuple

import torch


class FlowCondition(NamedTuple):
    mu: torch.Tensor
    mask: torch.Tensor
    spks: torch.Tensor
    cond: torch.Tensor


class FlowEulerDecoder(torch.nn.Module):
    """Keep the native noise source and use the input precision for the time grid."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        streaming_flow_steps: int | None = None,
        terminal_flow_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder: torch.nn.Module = decoder
        self.collect_conditions: list[FlowCondition] | None = None
        self.streaming_flow_steps: int | None = streaming_flow_steps
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
        if self.collect_conditions is not None:
            assert streaming and spks is not None and cond is not None
            self.collect_conditions.append(FlowCondition(mu, mask, spks, cond))
            return mu, None
        else:
            pass
        noise = self.decoder.rand_noise[:, :, : mu.size(2)].to(mu) * temperature
        configured_steps = (
            self.streaming_flow_steps if streaming else self.terminal_flow_steps
        )
        steps = n_timesteps if configured_steps is None else configured_steps
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


BF16FlowDecoder = FlowEulerDecoder
