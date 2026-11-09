# SPDX-License-Identifier: Apache-2.0
"""Scoped component BF16 experiments with FP32 estimator boundaries."""

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleLinear
from torch import nn

from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)
from sglang_omni.vendor.moss_speech_codec.matcha_components.bias_broadcast import (
    LiveBiasBroadcastProcessor,
)

logger = logging.getLogger(__name__)

FlowPrecisionStage = Literal[
    "ffn_first",
    "ffn",
    "attention_first",
    "attention",
    "convolutions_first",
    "convolutions",
]

FlowPrecisionProfile = (
    FlowPrecisionStage
    | Literal[
        "m1",
        "ffn_attention",
        "ffn_attention_sdpa",
        "ffn_attention_sdpa_resident",
        "ffn_attention_sdpa_storage",
    ]
)


class TensorForward(Protocol):
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


class BF16Forward:
    """Run one original operator under autocast and retain its FP32 interface."""

    def __init__(
        self,
        module: nn.Module,
        original: TensorForward,
        *,
        resident_weights: bool = False,
        store_bf16_parameters: bool = False,
    ) -> None:
        if store_bf16_parameters and not resident_weights:
            raise ValueError("BF16 parameter storage requires resident inference")
        else:
            pass
        self.module = module
        self.original = original
        self.resident_weights: bool = resident_weights
        self.store_bf16_parameters: bool = store_bf16_parameters
        self.fp32_cpu_parameters: dict[str, torch.Tensor] = {}
        self.resident_weight: torch.Tensor | None = None
        self.resident_bias: torch.Tensor | None = None
        self.parameter_copies: int = 0
        self.executions: int = 0
        self.bf16_executions: int = 0
        self.last_compute_dtype: torch.dtype | None = None

    @property
    def resident_parameter_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.resident_weight, self.resident_bias)
            if tensor is not None
        )

    @property
    def cpu_backup_parameter_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.fp32_cpu_parameters.values()
        )

    def prepare(self) -> None:
        if self.resident_weights:
            module = self.module
            if (
                type(module) not in (nn.Linear, LoRACompatibleLinear)
                or (
                    type(module) is LoRACompatibleLinear
                    and module.lora_layer is not None
                )
                or any(
                    parameter.dtype is not torch.float32
                    for parameter in module.parameters()
                )
            ):
                raise ValueError(
                    "Resident BF16 requires unadapted FP32 Linear parameters"
                )
            else:
                if self.store_bf16_parameters and (
                    module.training
                    or any(
                        parameter.grad is not None for parameter in module.parameters()
                    )
                ):
                    raise ValueError(
                        "BF16 parameter storage requires idle evaluation parameters"
                    )
                else:
                    pass
                self.resident_weight = module.weight.detach().bfloat16()
                self.resident_bias = (
                    module.bias.detach().bfloat16() if module.bias is not None else None
                )
                self.parameter_copies += 1 + int(module.bias is not None)
                if self.store_bf16_parameters:
                    self.fp32_cpu_parameters = {
                        name: parameter.detach().to(device="cpu", copy=True)
                        for name, parameter in module.named_parameters(recurse=False)
                    }
                    module.weight.data = self.resident_weight
                    if module.bias is not None:
                        module.bias.data = self.resident_bias
                    else:
                        pass
                else:
                    pass
        else:
            pass

    def release(self) -> None:
        for name, parameter in self.module.named_parameters(recurse=False):
            if name in self.fp32_cpu_parameters:
                parameter.data = self.fp32_cpu_parameters.pop(name).to(parameter.device)
            else:
                pass
        self.resident_weight = None
        self.resident_bias = None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_cpu_parameters and (
            self.module.training
            or torch.is_grad_enabled()
            or hidden_states.dtype != torch.float32
        ):
            raise RuntimeError(
                "BF16 parameter storage requires no-grad evaluation and FP32 inputs"
            )
        else:
            pass
        if (
            self.module.training
            or hidden_states.dtype != torch.float32
            or (self.resident_weights and torch.is_grad_enabled())
        ):
            return self.original(hidden_states)
        else:
            if self.resident_weights:
                if self.resident_weight is None:
                    raise RuntimeError(
                        "Resident BF16 requires an active precision scope"
                    )
                else:
                    output = F.linear(
                        hidden_states.bfloat16(),
                        self.resident_weight,
                        self.resident_bias,
                    )
                    if self.executions == 0:
                        logger.info(
                            f"MOSS_FLOW_BF16_RESIDENT_ACTIVE bytes={self.resident_parameter_bytes}"
                        )
                        if self.store_bf16_parameters:
                            logger.info(
                                f"MOSS_FLOW_BF16_STORAGE_ACTIVE cpu_backup_bytes={self.cpu_backup_parameter_bytes}"
                            )
                        else:
                            pass
                    else:
                        pass
            else:
                with torch.autocast(hidden_states.device.type, dtype=torch.bfloat16):
                    output = self.original(hidden_states)
            self.executions += 1
            self.bf16_executions += int(output.dtype == torch.bfloat16)
            self.last_compute_dtype = output.dtype
            return output.float()


@dataclass(kw_only=True)
class PrecisionOperator:
    name: str
    module: nn.Module
    forward: BF16Forward


class FlowBF16Scope:
    """Select estimator operators and optionally retain BF16 compute-weight copies."""

    def __init__(
        self,
        estimator: CausalConditionalDecoder,
        stage: FlowPrecisionStage,
        *,
        bf16_sdpa: bool = False,
        resident_weights: bool = False,
        store_bf16_parameters: bool = False,
    ) -> None:
        if resident_weights and stage not in ("ffn", "attention"):
            raise ValueError(
                "Resident BF16 requires the complete FFN or attention stage"
            )
        else:
            pass
        if bf16_sdpa and stage != "attention":
            raise ValueError(
                "BF16 SDPA requires the complete attention projection stage"
            )
        else:
            processors: list[LiveBiasBroadcastProcessor] = []
        if bf16_sdpa:
            for module in estimator.modules():
                if type(module) is Attention:
                    if type(module.processor) is not LiveBiasBroadcastProcessor:
                        raise ValueError("BF16 SDPA requires live bias broadcasting")
                    else:
                        processors.append(module.processor)
                else:
                    pass
            if not processors:
                raise ValueError("No live attention processors matched BF16 SDPA")
            else:
                pass
        else:
            pass
        self.sdpa_processors: tuple[LiveBiasBroadcastProcessor, ...] = tuple(processors)
        if stage not in (
            "ffn_first",
            "ffn",
            "attention_first",
            "attention",
            "convolutions_first",
            "convolutions",
        ):
            raise ValueError(f"Unsupported Flow precision stage: {stage}")
        else:
            self.operators: list[PrecisionOperator] = []
        for name, module in estimator.named_modules():
            selected = (
                (
                    stage in ("ffn_first", "ffn")
                    and ".ff." in name
                    and isinstance(module, nn.Linear)
                )
                or (
                    stage in ("attention_first", "attention")
                    and (".attn1." in name or ".attn2." in name)
                    and isinstance(module, nn.Linear)
                )
                or (
                    stage in ("convolutions_first", "convolutions")
                    and isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))
                    and not name.startswith(("final_block.", "final_proj."))
                )
            )
            if selected:
                self.operators.append(
                    PrecisionOperator(
                        name=name,
                        module=module,
                        forward=BF16Forward(
                            module,
                            module.forward,
                            resident_weights=resident_weights,
                            store_bf16_parameters=store_bf16_parameters,
                        ),
                    )
                )
            else:
                pass
        if stage.endswith("_first"):
            self.operators = self.operators[:1]
        else:
            pass
        if not self.operators:
            raise ValueError("No estimator operators matched the precision stage")
        else:
            self.active: bool = False

    def __enter__(self) -> "FlowBF16Scope":
        if (
            self.active
            or any("forward" in vars(operator.module) for operator in self.operators)
            or any(processor.use_bf16_sdpa for processor in self.sdpa_processors)
        ):
            raise ValueError("Precision scope requires unmodified operator forwards")
        else:
            try:
                for operator in self.operators:
                    operator.forward.prepare()
                    operator.module.forward = operator.forward
                for processor in self.sdpa_processors:
                    processor.use_bf16_sdpa = True
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
        for processor in self.sdpa_processors:
            processor.use_bf16_sdpa = False
        for operator in reversed(self.operators):
            if operator.module.forward is operator.forward:
                delattr(operator.module, "forward")
            else:
                pass
            operator.forward.release()
        self.active = False
