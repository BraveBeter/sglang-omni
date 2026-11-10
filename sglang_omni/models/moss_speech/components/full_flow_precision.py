# SPDX-License-Identifier: Apache-2.0
"""Reversible full-Flow BF16 storage, conditioning and Euler state."""

import logging
from dataclasses import dataclass
from types import TracebackType

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleLinear

from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow import (
    CausalMaskedDiffWithXvec,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.transformer.embedding import (
    EspnetRelPositionalEncoding,
    PositionalEncoding,
)
from sglang_omni.vendor.moss_speech_codec.matcha_components.bias_broadcast import (
    LiveBiasBroadcastProcessor,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class StoredFlowTensor:
    tensor: torch.Tensor
    backup: torch.Tensor


class FullFlowBF16Scope:
    """Use native BF16 operators and solver state with exact CPU restoration."""

    def __init__(self, flow: CausalMaskedDiffWithXvec) -> None:
        if type(flow) is not CausalMaskedDiffWithXvec:
            raise ValueError("Full Flow BF16 requires the native Flow model")
        else:
            pass
        self.flow = flow
        self.parameters: list[StoredFlowTensor] = []
        self.buffers: list[StoredFlowTensor] = []
        self.position_tables: list[
            tuple[PositionalEncoding | EspnetRelPositionalEncoding, StoredFlowTensor]
        ] = []
        self.processors: list[LiveBiasBroadcastProcessor] = []
        self.active: bool = False
        self.bf16_parameter_bytes: int = 0
        self.fp32_cpu_backup_bytes: int = 0

    def __enter__(self) -> "FullFlowBF16Scope":
        if (
            self.active
            or self.flow.inference_dtype is not None
            or any(
                module.training or "forward" in vars(module)
                for module in self.flow.modules()
            )
            or any(
                parameter.dtype is not torch.float32 or parameter.grad is not None
                for parameter in self.flow.parameters()
            )
            or any(
                isinstance(module, LoRACompatibleLinear)
                and module.lora_layer is not None
                for module in self.flow.modules()
            )
        ):
            raise ValueError("Full Flow BF16 requires idle unmodified FP32 evaluation")
        else:
            pass
        self.processors.clear()
        for module in self.flow.decoder.estimator.modules():
            if type(module) is Attention:
                if (
                    type(module.processor) is not LiveBiasBroadcastProcessor
                    or module.processor.use_bf16_sdpa
                    or module.processor.bf16_inputs
                ):
                    raise ValueError(
                        "Full Flow BF16 requires native live bias processors"
                    )
                else:
                    self.processors.append(module.processor)
            else:
                pass
        try:
            for tensor in self.flow.parameters():
                self.parameters.append(
                    StoredFlowTensor(
                        tensor=tensor,
                        backup=tensor.detach().to(device="cpu", copy=True),
                    )
                )
                tensor.data = tensor.detach().bfloat16()
            for tensor in self.flow.buffers():
                if tensor.is_floating_point():
                    self.buffers.append(
                        StoredFlowTensor(
                            tensor=tensor,
                            backup=tensor.detach().to(device="cpu", copy=True),
                        )
                    )
                    tensor.data = tensor.detach().bfloat16()
                else:
                    pass
            for module in self.flow.encoder.modules():
                if isinstance(
                    module, (PositionalEncoding, EspnetRelPositionalEncoding)
                ):
                    if module.pe is not None:
                        self.position_tables.append(
                            (
                                module,
                                StoredFlowTensor(
                                    tensor=module.pe,
                                    backup=module.pe.detach().to(
                                        device="cpu", copy=True
                                    ),
                                ),
                            )
                        )
                        module.pe.data = module.pe.detach().bfloat16()
                    else:
                        pass
                else:
                    pass
            for processor in self.processors:
                processor.use_bf16_sdpa = True
                processor.bf16_inputs = True
            self.flow.inference_dtype = torch.bfloat16
            self.flow.decoder.solver_dtype = torch.bfloat16
            self.flow.decoder.estimator.time_embeddings.compute_dtype = torch.bfloat16
            self.bf16_parameter_bytes = sum(
                state.tensor.numel() * state.tensor.element_size()
                for state in self.parameters
            )
            self.fp32_cpu_backup_bytes = sum(
                state.backup.numel() * state.backup.element_size()
                for state in self.parameters
            )
            self.active = True
            logger.info(
                f"MOSS_CODEC_FULL_FLOW_BF16_ACTIVE parameter_bytes={self.bf16_parameter_bytes}"
            )
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
        self.flow.inference_dtype = None
        self.flow.decoder.solver_dtype = None
        self.flow.decoder.estimator.time_embeddings.compute_dtype = torch.float32
        for processor in self.processors:
            processor.use_bf16_sdpa = False
            processor.bf16_inputs = False
        self.processors.clear()
        for module, original in self.position_tables:
            original.tensor.data = original.backup.to(original.tensor.device)
            module.pe = original.tensor
        self.position_tables.clear()
        for state in reversed(self.buffers + self.parameters):
            state.tensor.data = state.backup.to(state.tensor.device)
        self.parameters.clear()
        self.buffers.clear()
        self.active = False
