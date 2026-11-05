# SPDX-License-Identifier: Apache-2.0
"""Scoped component BF16 experiments with FP32 estimator boundaries."""

from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol

import torch
from torch import nn

from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)

FlowPrecisionStage = Literal[
    "ffn_first",
    "ffn",
    "attention_first",
    "attention",
    "convolutions_first",
    "convolutions",
]


class TensorForward(Protocol):
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


class BF16Forward:
    """Run one original operator under autocast and retain its FP32 interface."""

    def __init__(self, module: nn.Module, original: TensorForward) -> None:
        self.module = module
        self.original = original
        self.executions: int = 0
        self.bf16_executions: int = 0
        self.last_compute_dtype: torch.dtype | None = None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.module.training or hidden_states.dtype != torch.float32:
            return self.original(hidden_states)
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
    """Temporarily select estimator operators without changing parameter storage."""

    def __init__(
        self, estimator: CausalConditionalDecoder, stage: FlowPrecisionStage
    ) -> None:
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
                        forward=BF16Forward(module, module.forward),
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
        if self.active or any(
            "forward" in vars(operator.module) for operator in self.operators
        ):
            raise ValueError("Precision scope requires unmodified operator forwards")
        else:
            for operator in self.operators:
                operator.module.forward = operator.forward
            self.active = True
            return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for operator in reversed(self.operators):
            if operator.module.forward is operator.forward:
                delattr(operator.module, "forward")
            else:
                pass
        self.active = False
