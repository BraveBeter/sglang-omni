# SPDX-License-Identifier: Apache-2.0
"""Replay fixed-size HiFT convolutions between the native spectral transforms."""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The first, second and subsequent grouped streaming packets include the HiFT overlap.
STREAM_MEL_FRAMES = (32, 108, 308)


def predicted_spectrum(
    network: torch.nn.Module, mel: torch.Tensor, source_spectrum: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = network.conv_pre(mel)
    for i, upsample in enumerate(network.ups):
        hidden = upsample(F.leaky_relu(hidden, network.lrelu_slope))
        if i == network.num_upsamples - 1:
            hidden = network.reflection_pad(hidden)
        else:
            pass
        hidden = hidden + network.source_resblocks[i](
            network.source_downs[i](source_spectrum)
        )
        first = i * network.num_kernels
        residual = network.resblocks[first](hidden)
        for j in range(1, network.num_kernels):
            residual += network.resblocks[first + j](hidden)
        hidden = residual / network.num_kernels
    logits = network.conv_post(F.leaky_relu(hidden))
    bins = network.istft_params["n_fft"] // 2 + 1
    return logits[:, :bins].exp(), logits[:, bins:].sin()


class HiFTConvolutionGraph:
    def __init__(
        self, network: torch.nn.Module, mel: torch.Tensor, source_spectrum: torch.Tensor
    ) -> None:
        self.mel = mel.clone()
        self.source_spectrum = source_spectrum.clone()
        owner = torch.cuda.current_stream(mel.device)
        capture_stream = torch.cuda.Stream(device=mel.device)
        capture_stream.wait_stream(owner)
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                predicted_spectrum(network, self.mel, self.source_spectrum)
        owner.wait_stream(capture_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=capture_stream):
            self.magnitude, self.phase = predicted_spectrum(
                network, self.mel, self.source_spectrum
            )
        owner.wait_stream(capture_stream)

    def replay(
        self, mel: torch.Tensor, source_spectrum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.mel.copy_(mel)
        self.source_spectrum.copy_(source_spectrum)
        self.graph.replay()
        return self.magnitude, self.phase


class HiFTGraphDecoder:
    def __init__(self, network: torch.nn.Module, device: torch.device) -> None:
        self.network = network
        self.stream = torch.cuda.default_stream(device)
        self.graphs: dict[int, HiFTConvolutionGraph | None] = {}

    def __call__(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        frames = x.shape[-1]
        if (
            frames not in STREAM_MEL_FRAMES
            or x.shape[0] != 1
            or x.device != self.stream.device
            or x.dtype != torch.float32
            or s.dtype != torch.float32
            or torch.cuda.current_stream(x.device) != self.stream
            or (frames in self.graphs and self.graphs[frames] is None)
        ):
            return self.network.decode(x=x, s=s)
        else:
            components = self.network._stft(
                s.squeeze(1)
            )  # noqa: leading-underscore -- native spectral transform
            source_spectrum = torch.cat(components, dim=1)
        if frames not in self.graphs:
            try:
                self.graphs[frames] = HiFTConvolutionGraph(
                    self.network, x, source_spectrum
                )
            except RuntimeError as error:
                logger.warning(
                    f"HiFT graph capture failed for {frames} mel frames: {error}"
                )
                self.graphs[frames] = None
        else:
            pass
        graph = self.graphs[frames]
        if graph is None or graph.source_spectrum.shape != source_spectrum.shape:
            return self.network.decode(x=x, s=s)
        else:
            magnitude, phase = graph.replay(x, source_spectrum)
            waveform = self.network._istft(
                magnitude, phase
            )  # noqa: leading-underscore -- native inverse transform
            return waveform.clamp(-self.network.audio_limit, self.network.audio_limit)
