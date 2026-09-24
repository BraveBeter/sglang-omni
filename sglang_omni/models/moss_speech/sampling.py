# SPDX-License-Identifier: Apache-2.0
"""Request-owned device histories for the unchanged repetition-penalty math."""
from __future__ import annotations

import torch


class DeviceChannelHistory:
    """Upload the immutable prompt once, then append only newly committed rows."""

    def __init__(
        self, prompt_rows: torch.Tensor, max_generated: int, device: torch.device
    ) -> None:
        self.prompt_rows = prompt_rows
        self.prompt_length = len(prompt_rows)
        self.generated = 0
        self.rows = torch.empty(
            (2, self.prompt_length + max_generated), dtype=torch.long, device=device
        )
        self.rows[:, : self.prompt_length].copy_(prompt_rows.T)

    def histories(
        self, output_rows: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(output_rows)
        if count < self.generated:
            raise ValueError("Sampling history rewind requires a fresh request cache")
        end = self.prompt_length + count
        if end > self.rows.shape[1]:
            raise ValueError("Sampling history exceeds the request token budget")
        if count > self.generated:
            tail = torch.tensor(
                output_rows[self.generated :], dtype=torch.long, device=self.rows.device
            )
            self.rows[:, self.prompt_length + self.generated : end].copy_(tail.T)
            self.generated = count
        return self.rows[0, :end], self.rows[1, :end]
