# SPDX-License-Identifier: Apache-2.0
"""MOSS-Speech AR engine builder (P3 D1 decision).

Model-specific subclass of the COMMON generation builder
(:class:`sglang_omni.scheduling.engine_factory.SGLangGenerationEngineBuilder`)
— no hand-rolled scheduler assembly. Responsibilities:

- ``model_arch_override="MossSpeechForCausalLM"`` rides the common bootstrap
  and triggers ModelWorker's 40-attention-layer accounting branch
  (num_shared + 2 * num_modality = 40; HF num_hidden_layers stays 36).
- V1 policy is locked here: CUDA graph / torch compile / radix cache /
  quantization / TP>1 must stay OFF; violating operator overrides raise
  instead of silently enabling (P3-01 §3.5).
- The model runner and request/result adapters are T3.4 deliverables; this
  module only fixes their wiring points so T3.3's minimal driver and T3.7's
  formal takeover share ONE build path (P3 task dependency note).
"""

from __future__ import annotations

from typing import Any

from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder


class MossSpeechEngineBuilder(SGLangGenerationEngineBuilder):
    model_name = "MOSS-Speech AR"
    # max_position_embeddings from the locked checkpoint config.
    context_length = 40960
    model_arch_override = "MossSpeechForCausalLM"

    def __init__(self, context_length: int | None = None) -> None:
        super().__init__()
        if context_length is not None:
            self.context_length = int(context_length)

    # ------------------------------------------------------------ policy
    def generation_defaults(self, dtype: str = "bfloat16") -> dict[str, Any]:
        return {
            "dtype": dtype,
            "disable_cuda_graph": True,
            "enable_torch_compile": False,
            "disable_radix_cache": True,
            "tp_size": 1,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # V1 boundaries are non-negotiable: force them even if an operator
        # passed conflicting values (fail-closed, no silent downgrade).
        overrides["disable_cuda_graph"] = True
        overrides["enable_torch_compile"] = False
        overrides["disable_radix_cache"] = True
        overrides["tp_size"] = 1
        overrides.setdefault("mem_fraction_static", 0.72)

    def customize_server_args(self, server_args: Any) -> None:
        server_args.disable_cuda_graph = True
        server_args.enable_torch_compile = False
        server_args.disable_radix_cache = True
        if int(getattr(server_args, "tp_size", 1)) != 1:
            raise ValueError("MOSS-Speech V1 supports TP=1 only")

    def validate_before_infrastructure(self, server_args: Any) -> None:
        sa = server_args
        problems = []
        if not sa.disable_cuda_graph:
            problems.append("cuda_graph enabled")
        if getattr(sa, "enable_torch_compile", False):
            problems.append("torch_compile enabled")
        if not sa.disable_radix_cache:
            problems.append("radix_cache enabled")
        if int(getattr(sa, "tp_size", 1)) != 1:
            problems.append(f"tp_size={sa.tp_size}")
        if str(getattr(sa, "dtype", "bfloat16")) not in ("bfloat16", "bf16"):
            problems.append(f"dtype={sa.dtype}")
        if problems:
            raise ValueError(
                f"{self.model_name}: V1 policy violated: {', '.join(problems)}"
            )

    # ------------------------------------------------- T3.4 wiring points
    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        raise NotImplementedError(
            "MossSpeechEngineBuilder.setup_model lands with the P3/T3.4 model "
            "runner (single build path shared by the T3.3 spike and the T3.7 "
            "formal AR factory takeover)"
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError(
            "MossSpeechEngineBuilder.make_model_runner lands with P3/T3.4"
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        raise NotImplementedError(
            "MossSpeechEngineBuilder.make_adapters lands with P3/T3.4"
        )
