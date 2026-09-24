# SPDX-License-Identifier: Apache-2.0
"""MOSS-Speech AR engine builder (P3 D1 decision).

Model-specific subclass of the COMMON generation builder
(:class:`sglang_omni.scheduling.engine_factory.SGLangGenerationEngineBuilder`)
— no hand-rolled scheduler assembly. Responsibilities:

- ``model_arch_override="MossSpeechForCausalLM"`` rides the common bootstrap
  and triggers ModelWorker's 40-attention-layer accounting branch
  (num_shared + 2 * num_modality = 40; HF num_hidden_layers stays 36).
- Default policy is locked here: CUDA graph / torch compile / radix cache /
  quantization / TP>1 must stay OFF; violating operator overrides raise
  instead of silently enabling. Prefix caching is a separate opt-in profile
  using model-specific canonical keys over the common KV pools.
- The model runner and request/result adapters are T3.4 deliverables; this
  module only fixes their wiring points so T3.3's minimal driver and T3.7's
  formal takeover share ONE build path (P3 task dependency note).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder


class MossSpeechEngineBuilder(SGLangGenerationEngineBuilder):
    model_name = "MOSS-Speech AR"
    # max_position_embeddings from the locked checkpoint config.
    context_length = 40960
    model_arch_override = "MossSpeechForCausalLM"

    def __init__(
        self,
        context_length: int | None = None,
        *,
        streaming: bool = False,
        prefix_cache: bool = False,
        eager_cpu_metadata: bool = False,
        mixed_modalities: bool = False,
        decode_projection_graphs: bool = False,
        decode_full_graphs: bool = False,
        prefill_full_graphs: bool = False,
        sampling_optimizations: bool = False,
    ) -> None:
        super().__init__()
        self.streaming = streaming
        self.prefix_cache = prefix_cache
        self.eager_cpu_metadata = eager_cpu_metadata
        self.decode_full_graphs = decode_full_graphs
        self._decode_graph_manager = None
        self.prefill_full_graphs = prefill_full_graphs
        self._prefill_graph_manager = None
        self.decode_projection_graphs = decode_projection_graphs
        self.sampling_optimizations = sampling_optimizations
        from sglang_omni.models.moss_speech.request_lifecycle import (
            MossSpeechRequestLifecycle,
        )

        self.request_lifecycle = MossSpeechRequestLifecycle(
            mixed_modalities=mixed_modalities
        )
        if context_length is not None:
            self.context_length = int(context_length)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.moss_speech.hf_config import (
            ensure_moss_speech_config_registered,
        )

        ensure_moss_speech_config_registered()

    # ------------------------------------------------------------ policy
    def generation_defaults(self, dtype: str = "bfloat16") -> dict[str, Any]:
        return {
            "max_running_requests": 16,
            "attention_backend": "torch_native",
            "dtype": dtype,
            "disable_cuda_graph": True,
            "enable_torch_compile": False,
            "disable_radix_cache": not self.prefix_cache,
            "tp_size": 1,
            # chunked prefill splits prompts across multiple EXTEND blocks,
            # firing post_prefill per chunk; the reference-semantics sampling
            # (one first token per request) is only verified for single-block
            # prefill in V1 — keep it off (task note: unverified configs off).
            "chunked_prefill_size": -1,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        required = {
            "attention_backend": "torch_native",
            "trust_remote_code": False,
            "disable_cuda_graph": True,
            "enable_torch_compile": False,
            "disable_radix_cache": not self.prefix_cache,
            "tp_size": 1,
            "chunked_prefill_size": -1,
        }
        for key, value in required.items():
            if key in overrides and overrides[key] != value:
                raise ValueError(f"MOSS-Speech V1 rejects {key}={overrides[key]!r}")
            overrides[key] = value
        if overrides.get("quantization") is not None:
            raise ValueError("MOSS-Speech V1 does not support quantization")
        if overrides.get("kv_cache_dtype", "auto") not in ("auto", "bfloat16", "bf16"):
            raise ValueError("MOSS-Speech V1 requires BF16 KV")
        overrides.setdefault("mem_fraction_static", 0.72)

    def customize_server_args(self, server_args: Any) -> None:
        server_args.disable_cuda_graph = True
        server_args.enable_torch_compile = False
        server_args.disable_radix_cache = not self.prefix_cache
        if int(getattr(server_args, "tp_size", 1)) != 1:
            raise ValueError("MOSS-Speech V1 supports TP=1 only")

    def validate_before_infrastructure(self, server_args: Any) -> None:
        sa = server_args
        problems = []
        if not sa.disable_cuda_graph:
            problems.append("cuda_graph enabled")
        if getattr(sa, "enable_torch_compile", False):
            problems.append("torch_compile enabled")
        if bool(sa.disable_radix_cache) == self.prefix_cache:
            problems.append("radix_cache does not match the selected profile")
        if int(getattr(sa, "chunked_prefill_size", -1) or -1) != -1:
            problems.append(f"chunked_prefill_size={sa.chunked_prefill_size}")
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
        from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel

        model = model_worker.model_runner.model
        if not isinstance(model, MossSpeechSGLangModel):
            raise RuntimeError(
                f"{self.model_name}: expected MossSpeechSGLangModel, got {type(model)}"
            )
        model.assert_heads_independent()
        if self.decode_projection_graphs:
            for layer in (
                list(model.layers) + list(model.text_block) + list(model.audio_block)
            ):
                layer.enable_projection_graphs = True
        if self.eager_cpu_metadata:
            from sglang.srt.layers.attention.torch_native_backend import (
                TorchNativeAttnBackend,
            )

            from .attention import MossSpeechTorchNativeBackend

            runner = model_worker.model_runner
            if type(runner.attn_backend) is not TorchNativeAttnBackend:
                raise ValueError(
                    "Host metadata requires the eager torch_native backend"
                )
            runner.attn_backend = MossSpeechTorchNativeBackend(runner)
            runner.attn_backend.prefill_attention_backend_str = "torch_native"
            runner.attn_backend.decode_attention_backend_str = "torch_native"
        # the decode-feedback staging embedding is not a checkpoint tensor;
        # the weight loader leaves it on the init device — move it here
        target = torch.device(device)
        if model._decode_input_embedding.weight.device != target:
            model._decode_input_embedding = model._decode_input_embedding.to(target)
        if self.decode_full_graphs:
            from .decode_graphs import MossSpeechDecodeGraphManager

            self._decode_graph_manager = MossSpeechDecodeGraphManager(
                model, model_worker.model_runner
            )
            model._decode_graph_manager = self._decode_graph_manager

        if self.prefill_full_graphs:
            from .prefill_graphs import MossSpeechPrefillGraphManager

            self._prefill_graph_manager = MossSpeechPrefillGraphManager(
                model, model_worker.model_runner
            )
            model._prefill_graph_manager = self._prefill_graph_manager

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        if self._prefill_graph_manager is not None:
            if scheduler._running or scheduler._scheduler_thread_id is not None:
                raise RuntimeError(
                    "Prefill graphs must initialize before scheduler starts"
                )
            self._prefill_graph_manager.initialize(model_runner)

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del server_args, generation_cuda_graph_enabled
        if self.decode_full_graphs:
            self._decode_graph_manager = model._decode_graph_manager
            self._decode_graph_manager.initialize()

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        if self._prefill_graph_manager is None:
            return (
                {}
                if self._decode_graph_manager is None
                else {"shutdown_callback": self._decode_graph_manager.close}
            )
        return {"shutdown_callback": self.cleanup_build_failure}

    def cleanup_build_failure(self) -> None:
        if self._prefill_graph_manager is not None:
            self._prefill_graph_manager.close()
        if self._decode_graph_manager is not None:
            self._decode_graph_manager.close()

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        import importlib

        runner_mod = importlib.import_module(
            "sglang_omni.models.moss_speech.model_runner"
        )
        runner = runner_mod.MossSpeechModelRunner(model_worker, output_proc)
        runner.sampling_optimizations = self.sampling_optimizations
        return runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.moss_speech.request_builders import (
            make_moss_speech_scheduler_adapters,
        )

        self.request_lifecycle.model = model
        return make_moss_speech_scheduler_adapters(
            model=model,
            lifecycle=self.request_lifecycle,
            prefix_cache=self.prefix_cache,
        )

    def _build_runtime(self, **kwargs: Any) -> tuple[Any, Any]:
        if self.prefix_cache:
            from sglang.srt.mem_cache.cache_init_params import CacheInitParams

            from .prefix_cache import MossSpeechRadixCache

            # Infrastructure has allocated the shared pools but admitted no
            # requests yet. Bind the semantic cache before constructing runtime.
            args = kwargs["server_args"]
            cache = MossSpeechRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=kwargs["req_to_token_pool"],
                    token_to_kv_pool_allocator=kwargs["token_to_kv_pool_allocator"],
                    page_size=args.page_size,
                    chunked_prefill_size=args.chunked_prefill_size,
                )
            )
            kwargs["tree_cache"] = cache
            kwargs["prefill_manager"].tree_cache = cache
        return super()._build_runtime(**kwargs)

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        if not self.streaming:
            return {}
        from .streaming import make_stream_output_builder

        return {"stream_output_builder": make_stream_output_builder()}

    def make_abort_callback(self) -> Callable[[str], None]:
        return lambda rid: self.request_lifecycle.release(rid, aborted=True)

    def make_request_finished_callback(self) -> Callable[[str], None]:
        return self.request_lifecycle.release
