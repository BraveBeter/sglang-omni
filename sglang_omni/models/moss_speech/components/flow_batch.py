# SPDX-License-Identifier: Apache-2.0
"""Research qualification for equal-length FP32 Flow solver batches."""

from dataclasses import dataclass

import torch

from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow_matching import (
    CausalConditionalCFM,
)


@dataclass(kw_only=True)
class FlowBatchInput:
    request_id: str
    noise: torch.Tensor
    timestep_grid: torch.Tensor
    encoded_mel: torch.Tensor
    mask: torch.Tensor
    speaker_embedding: torch.Tensor
    conditioning_mel: torch.Tensor
    streaming: bool


class EqualLengthFlowBatch:
    """Combine prepared requests while preserving the native single-row fallback."""

    def __init__(self, decoder: CausalConditionalCFM) -> None:
        self.decoder = decoder
        self.executions: dict[int, int] = {2: 0, 4: 0}
        self.fallback_requests = 0

    def compatible(self, requests: list[FlowBatchInput]) -> bool:
        if len(requests) not in (2, 4) or self.decoder.training:
            return False
        else:
            first = requests[0]
        mel_frames = first.noise.shape[-1]
        shapes = (
            (1, 80, mel_frames),
            (11,),
            (1, 80, mel_frames),
            (1, 1, mel_frames),
            (1, 80),
            (1, 80, mel_frames),
        )
        for request in requests:
            tensors = (
                request.noise,
                request.timestep_grid,
                request.encoded_mel,
                request.mask,
                request.speaker_embedding,
                request.conditioning_mel,
            )
            if (
                request.streaming != first.streaming
                or any(
                    tensor.shape != shape
                    or tensor.dtype != torch.float32
                    or tensor.device != first.noise.device
                    for tensor, shape in zip(tensors, shapes, strict=True)
                )
                or not torch.equal(request.timestep_grid, first.timestep_grid)
                or torch.is_autocast_enabled(first.noise.device.type)
            ):
                return False
            else:
                pass
        return True

    @torch.inference_mode()
    def solve(self, requests: list[FlowBatchInput]) -> dict[str, torch.Tensor]:
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("Duplicate Flow batch request IDs")
        elif not self.compatible(requests):
            self.fallback_requests += len(requests)
            return {
                request.request_id: self.decoder.solve_euler(
                    request.noise,
                    request.timestep_grid,
                    request.encoded_mel,
                    request.mask,
                    request.speaker_embedding,
                    request.conditioning_mel,
                    streaming=request.streaming,
                ).clone()
                for request in requests
            }
        else:
            batch_size = len(requests)
        noise = torch.cat([request.noise for request in requests])
        mean = torch.cat([request.encoded_mel for request in requests])
        mask = torch.cat([request.mask for request in requests])
        speaker = torch.cat([request.speaker_embedding for request in requests])
        conditioning = torch.cat([request.conditioning_mel for request in requests])
        noise_input = torch.zeros_like(noise).repeat(2, 1, 1)
        mask_input = mask.repeat(2, 1, 1)
        mean_input = torch.cat([mean, torch.zeros_like(mean)])
        speaker_input = torch.cat([speaker, torch.zeros_like(speaker)])
        conditioning_input = torch.cat([conditioning, torch.zeros_like(conditioning)])
        timestep_input = torch.zeros(2 * batch_size, device=noise.device)
        timestep_grid = requests[0].timestep_grid
        timestep = timestep_grid[0].unsqueeze(0)
        timestep_delta = timestep_grid[1] - timestep_grid[0]
        for step in range(1, len(timestep_grid)):
            noise_input[:batch_size] = noise
            noise_input[batch_size:] = noise
            timestep_input[:] = timestep.unsqueeze(0)
            derivative = self.decoder.forward_estimator(
                noise_input,
                mask_input,
                mean_input,
                timestep_input,
                speaker_input,
                conditioning_input,
                requests[0].streaming,
            )
            conditional, unconditional = derivative.split(batch_size)
            derivative = (
                1.0 + self.decoder.inference_cfg_rate
            ) * conditional - self.decoder.inference_cfg_rate * unconditional
            noise = noise + timestep_delta * derivative
            timestep = timestep + timestep_delta
            if step < len(timestep_grid) - 1:
                timestep_delta = timestep_grid[step + 1] - timestep
            else:
                pass
        self.executions[batch_size] += 1
        return {
            request.request_id: noise[index : index + 1].clone()
            for index, request in enumerate(requests)
        }
