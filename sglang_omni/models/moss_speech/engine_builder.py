# SPDX-License-Identifier: Apache-2.0
"""Eager AR policy and model-specific adapters on the shared engine builder."""

from __future__ import annotations

from typing import Protocol

import torch
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_speech.hf_config import (
    KV_RESERVE_TOKENS,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.request_builders import (
    RequestBuilder,
    ResultAdapter,
    make_moss_speech_scheduler_adapters,
)
from sglang_omni.models.moss_speech.request_data import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.request_lifecycle import MossSpeechRequestLifecycle
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
from sglang_omni.models.moss_speech.streaming import make_stream_output_builder
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class RequestCallback(Protocol):
    def __call__(self, request_id: str) -> None: ...


class MossSpeechEngineBuilder(
    SGLangGenerationEngineBuilder[MossSpeechSGLangRequestData]
):
    model_name = "MOSS-Speech AR"
    context_length = 40960
    model_arch_override = "MossSpeechForCausalLM"
    supports_context_length_override = True

    def __init__(self, context_length: int | None = None, *, streaming: bool) -> None:
        super().__init__()
        self.streaming = streaming
        self.request_lifecycle: MossSpeechRequestLifecycle | None = None
        if context_length is None:
            pass
        else:
            self.context_length = context_length

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        ensure_moss_speech_config_registered()

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        max_running_requests = self.server_args_overrides.get("max_running_requests")
        if (
            isinstance(max_running_requests, bool)
            or not isinstance(max_running_requests, int)
            or max_running_requests <= 0
        ):
            raise ValueError(
                "MOSS-Speech requires a positive max_running_requests in the stage engine configuration"
            )
        else:
            return {
                "max_running_requests": max_running_requests,
                "attention_backend": "torch_native",
                "dtype": dtype,
                "disable_cuda_graph": True,
                "enable_torch_compile": False,
                "disable_radix_cache": True,
                "tp_size": 1,
                "chunked_prefill_size": -1,
                "trust_remote_code": False,
            }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        required = {
            "attention_backend": "torch_native",
            "trust_remote_code": False,
            "disable_cuda_graph": True,
            "enable_torch_compile": False,
            "disable_radix_cache": True,
            "tp_size": 1,
            "chunked_prefill_size": -1,
        }
        for name, value in required.items():
            if name in overrides and overrides[name] != value:
                raise ValueError(
                    f"MOSS-Speech eager inference rejects {name}={overrides[name]!r}"
                )
            else:
                overrides[name] = value
        if overrides.get("quantization") is not None:
            raise ValueError(
                "MOSS-Speech eager inference does not support quantization"
            )
        else:
            pass
        if overrides.get("kv_cache_dtype", "auto") not in ("auto", "bfloat16", "bf16"):
            raise ValueError("MOSS-Speech requires BF16 KV storage")
        else:
            pass
        if overrides.get("dtype") not in ("bfloat16", "bf16"):
            raise ValueError("MOSS-Speech requires BF16 model weights")
        else:
            pass

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        configuration = resolved_view(server_args)
        problems = []
        if not configuration.disable_cuda_graph or configuration.enable_torch_compile:
            problems.append("generation must use eager execution")
        else:
            pass
        if (
            not configuration.disable_radix_cache
            or configuration.chunked_prefill_size != -1
        ):
            problems.append("prefix caching and chunked prefill must be disabled")
        else:
            pass
        if configuration.tp_size != 1:
            problems.append("tensor parallel size must be one")
        else:
            pass
        if configuration.dtype not in ("bfloat16", "bf16"):
            problems.append("model dtype must be BF16")
        else:
            pass
        if attention_backends_of(configuration) != ("torch_native", "torch_native"):
            problems.append("prefill and decode attention must use torch_native")
        else:
            pass
        if problems:
            raise ValueError(f"{self.model_name}: {', '.join(problems)}")
        else:
            pass

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        model = model_worker.model_runner.model
        assert isinstance(model, MossSpeechSGLangModel)
        model.assert_heads_independent()
        configuration = resolved_view(server_args)
        if (
            configuration.max_running_requests
            > model.decode_input_embedding.num_embeddings
        ):
            raise ValueError(
                "max_running_requests exceeds the MOSS-Speech decode embedding capacity"
            )
        else:
            pass
        model.decode_input_embedding = model.decode_input_embedding.to(
            torch.device(device)
        )

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> MossSpeechModelRunner:
        return MossSpeechModelRunner(model_worker, output_proc)

    def post_scheduler_setup(
        self,
        scheduler: OmniScheduler[MossSpeechSGLangRequestData],
        model_runner: MossSpeechModelRunner,
    ) -> None:
        assert scheduler.model_runner is model_runner
        context_tokens = min(
            scheduler.model_config.context_len,
            scheduler.max_total_num_tokens - KV_RESERVE_TOKENS,
        )
        # note (Codex): Upstream clipping subtracts one from this sentinel; physical KV guards remain reserved.
        scheduler.max_req_len = context_tokens + 1
        scheduler.max_req_input_len = context_tokens - 1

    def make_adapters(
        self, model: torch.nn.Module
    ) -> tuple[RequestBuilder, ResultAdapter]:
        assert isinstance(model, MossSpeechSGLangModel)
        self.request_lifecycle = MossSpeechRequestLifecycle(model=model)
        return make_moss_speech_scheduler_adapters(lifecycle=self.request_lifecycle)

    def extra_scheduler_kwargs(self) -> SchedulerExtras[MossSpeechSGLangRequestData]:
        if self.streaming:
            return {"stream_output_builder": make_stream_output_builder()}
        else:
            return {}

    def abort_request(self, request_id: str) -> None:
        assert self.request_lifecycle is not None
        self.request_lifecycle.release(request_id, aborted=True)

    def finish_request(self, request_id: str) -> None:
        assert self.request_lifecycle is not None
        self.request_lifecycle.release(request_id)

    def make_abort_callback(self) -> RequestCallback:
        return self.abort_request

    def make_request_finished_callback(self) -> RequestCallback:
        return self.finish_request
