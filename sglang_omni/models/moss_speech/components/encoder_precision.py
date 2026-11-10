# SPDX-License-Identifier: Apache-2.0
"""Scoped Flow condition-encoder BF16 with explicit FP32 boundaries."""

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from sglang_omni.models.moss_speech.components.flow_precision import (
    BF16Forward,
    PrecisionOperator,
    TensorForward,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.transformer.upsample_encoder import (
    UpsampleConformerEncoder,
)
from sglang_omni.vendor.moss_speech_codec.matcha_components.bias_broadcast import (
    LiveBiasBroadcastProcessor,
)

logger = logging.getLogger(__name__)
EncoderPrecisionMode = Literal["linear", "attention", "full"]


class EncoderForward(Protocol):
    def __call__(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        context: torch.Tensor | None = None,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class FP32BoundaryForward:
    def __init__(self, original: TensorForward) -> None:
        self.original = original
        self.executions: int = 0
        self.last_compute_dtype: torch.dtype | None = None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(hidden_states.device.type, enabled=False):
            output = self.original(hidden_states.float())
        self.executions += 1
        self.last_compute_dtype = output.dtype
        return output


@dataclass(kw_only=True)
class FP32Operator:
    module: nn.Module
    forward: FP32BoundaryForward


class EncoderBF16Forward:
    def __init__(
        self, encoder: UpsampleConformerEncoder, mode: EncoderPrecisionMode
    ) -> None:
        self.encoder = encoder
        self.original: EncoderForward = encoder.forward
        self.mode = mode
        self.executions: int = 0
        self.last_output_dtype: torch.dtype | None = None
        self.last_mask_dtype: torch.dtype | None = None

    def __call__(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        context: torch.Tensor | None = None,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.encoder.training
            or torch.is_grad_enabled()
            or xs.dtype is not torch.float32
        ):
            raise RuntimeError(
                "Flow encoder BF16 requires no-grad evaluation and FP32 inputs"
            )
        else:
            pass
        with torch.autocast(
            xs.device.type, dtype=torch.bfloat16, enabled=self.mode != "linear"
        ):
            hidden_states, masks = self.original(
                xs,
                xs_lens,
                context=torch.empty(0, 0, 0) if context is None else context,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                streaming=streaming,
            )
        if self.executions == 0:
            logger.info(f"MOSS_FLOW_ENCODER_BF16_ACTIVE mode={self.mode}")
        else:
            pass
        self.executions += 1
        hidden_states = hidden_states.float()
        self.last_output_dtype = hidden_states.dtype
        self.last_mask_dtype = masks.dtype
        return hidden_states, masks


class FlowEncoderBF16Scope:
    """Lower condition-encoder compute while preserving normalization and solver inputs."""

    def __init__(
        self,
        encoder: UpsampleConformerEncoder,
        mode: EncoderPrecisionMode = "attention",
    ) -> None:
        if type(encoder) is not UpsampleConformerEncoder or mode not in (
            "linear",
            "attention",
            "full",
        ):
            raise ValueError(
                "Flow encoder BF16 requires a supported native encoder mode"
            )
        else:
            pass
        self.encoder = encoder
        self.mode = mode
        self.forward = EncoderBF16Forward(encoder, mode)
        self.operators: list[PrecisionOperator] = []
        self.fp32_operators: list[FP32Operator] = []
        self.handles: list[RemovableHandle] = []
        self.convolution_executions: int = 0
        self.bf16_convolution_executions: int = 0
        self.sdpa_processors: tuple[LiveBiasBroadcastProcessor, ...] = ()
        for name, module in encoder.named_modules():
            if type(module) is nn.Linear:
                self.operators.append(
                    PrecisionOperator(
                        name=name,
                        module=module,
                        forward=BF16Forward(
                            module,
                            module.forward,
                            resident_weights=True,
                            store_bf16_parameters=True,
                        ),
                    )
                )
            elif type(module) is nn.LayerNorm or (
                mode == "attention" and type(module) is nn.Conv1d
            ):
                self.fp32_operators.append(
                    FP32Operator(
                        module=module, forward=FP32BoundaryForward(module.forward)
                    )
                )
            else:
                pass
        self.linear_forwards: dict[nn.Module, BF16Forward] = {
            operator.module: operator.forward for operator in self.operators
        }
        self.active: bool = False

    def record_convolution(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        assert type(module) is nn.Conv1d and inputs
        self.convolution_executions += 1
        self.bf16_convolution_executions += int(output.dtype is torch.bfloat16)

    def record_linear(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        assert inputs and inputs[0].dtype in (torch.float32, torch.bfloat16)
        forward = self.linear_forwards[module]
        forward.executions += 1
        forward.bf16_executions += int(output.dtype is torch.bfloat16)
        forward.last_compute_dtype = output.dtype

    def __enter__(self) -> "FlowEncoderBF16Scope":
        modules = [
            self.encoder,
            *(operator.module for operator in self.operators),
            *(operator.module for operator in self.fp32_operators),
        ]
        if self.active or any("forward" in vars(module) for module in modules):
            raise ValueError(
                "Flow encoder precision requires unmodified operator forwards"
            )
        else:
            pass
        try:
            for operator in self.operators:
                operator.forward.prepare()
                if self.mode == "linear":
                    operator.module.forward = operator.forward
                else:
                    self.handles.append(
                        operator.module.register_forward_hook(self.record_linear)
                    )
            for operator in self.fp32_operators:
                operator.module.forward = operator.forward
            if self.mode == "full":
                for module in self.encoder.modules():
                    if type(module) is nn.Conv1d:
                        self.handles.append(
                            module.register_forward_hook(self.record_convolution)
                        )
                    else:
                        pass
            else:
                pass
            self.encoder.forward = self.forward
            self.active = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.encoder.forward is self.forward:
            delattr(self.encoder, "forward")
        else:
            pass
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for operator in self.fp32_operators:
            if operator.module.forward is operator.forward:
                delattr(operator.module, "forward")
            else:
                pass
        for operator in reversed(self.operators):
            if operator.module.forward is operator.forward:
                delattr(operator.module, "forward")
            else:
                pass
            operator.forward.release()
        self.active = False
