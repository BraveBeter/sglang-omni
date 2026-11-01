# SPDX-License-Identifier: Apache-2.0
"""Small native MOSS models and scoped attention pools for regression tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import DynamicGradMode

from sglang_omni.models.moss_speech.attention import MossSpeechTorchNativeBackend
from sglang_omni.models.moss_speech.decode_graphs import (
    LENGTHS,
    DecodeEntry,
    DecodeGraph,
)
from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig
from sglang_omni.models.moss_speech.model_layers import MossSpeechRMSNorm
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.native_norm import NativeNorm
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.prefill_graphs import PrefillEntry, PrefillGraphs
from sglang_omni.models.moss_speech.projection_graphs import MossSpeechProjectionGraphs
from sglang_omni.models.moss_speech.request_builders import (
    make_moss_speech_scheduler_adapters,
)
from sglang_omni.models.moss_speech.request_data import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.request_lifecycle import MossSpeechRequestLifecycle
from sglang_omni.models.moss_speech.sglang_model import (
    MossSpeechModelOutput,
    MossSpeechSGLangModel,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class AttentionPools:
    device: str
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool


class RowCommitRunner(MossSpeechModelRunner):
    """Replace engine allocation while retaining the real row-commit methods."""

    def __init__(self, model: MossSpeechSGLangModel) -> None:
        self.model = model
        self.report_mixed_batch = False


@contextmanager
def native_runtime(device: torch.device) -> Iterator[None]:
    with (
        get_context().override_server_args(
            device=device.type,
            dtype="bfloat16",
            attention_backend="torch_native",
            disable_cuda_graph=True,
            enable_torch_compile=False,
            disable_radix_cache=True,
            tp_size=1,
        ),
        get_parallel().override(
            tp_rank=0,
            pp_rank=0,
            attn_tp_rank=0,
            attn_dp_rank=0,
            attn_cp_rank=0,
            moe_tp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
            tp_group=None,
            attn_tp_group=None,
        ),
        torch.device(device),
        torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ),
    ):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(previous_dtype)


def tiny_model(
    device: torch.device, *, audio_vocab_size: int = 64, decode_batch_capacity: int = 2
) -> MossSpeechSGLangModel:
    configuration = MossSpeechConfig(
        vocab_size=256,
        audio_vocab_size=audio_vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_shared_layers=2,
        num_modality_layers=1,
        num_hidden_layers=3,
        max_position_embeddings=16,
        moss_decode_buffer_bs=decode_batch_capacity,
    )
    model = MossSpeechSGLangModel(configuration, init_device=device)
    generator = torch.Generator(device=device).manual_seed(183)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2, generator=generator)
    return model


def checkpoint_weights(
    configuration: MossSpeechConfig,
) -> list[tuple[str, torch.Tensor]]:
    """Construct the external checkpoint schema without the native name mapper."""
    assert configuration.audio_vocab_size is not None
    hidden_size = configuration.hidden_size
    query_rows = configuration.num_attention_heads * configuration.head_dim
    key_value_rows = configuration.num_key_value_heads * configuration.head_dim
    shapes: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (configuration.vocab_size, hidden_size)),
        ("model.audio_embed.weight", (configuration.audio_vocab_size, hidden_size)),
        ("model.text_norm.weight", (hidden_size,)),
        ("model.audio_norm.weight", (hidden_size,)),
        ("text_lm_head.weight", (configuration.vocab_size, hidden_size)),
        ("audio_lm_head.weight", (configuration.audio_vocab_size, hidden_size)),
    ]
    layer_shapes = [
        ("self_attn.q_proj.weight", (query_rows, hidden_size)),
        ("self_attn.k_proj.weight", (key_value_rows, hidden_size)),
        ("self_attn.v_proj.weight", (key_value_rows, hidden_size)),
        ("self_attn.o_proj.weight", (hidden_size, query_rows)),
        ("self_attn.q_norm.weight", (configuration.head_dim,)),
        ("self_attn.k_norm.weight", (configuration.head_dim,)),
        ("mlp.gate_proj.weight", (configuration.intermediate_size, hidden_size)),
        ("mlp.up_proj.weight", (configuration.intermediate_size, hidden_size)),
        ("mlp.down_proj.weight", (hidden_size, configuration.intermediate_size)),
        ("input_layernorm.weight", (hidden_size,)),
        ("post_attention_layernorm.weight", (hidden_size,)),
    ]
    for block, depth in (
        ("shared_block", configuration.num_shared_layers),
        ("text_block", configuration.num_modality_layers),
        ("audio_block", configuration.num_modality_layers),
    ):
        for layer_index in range(depth):
            shapes.extend(
                (f"model.{block}.layers.{layer_index}.{leaf}", shape)
                for leaf, shape in layer_shapes
            )
    return [
        (name, torch.full(shape, index + 1, dtype=torch.bfloat16))
        for index, (name, shape) in enumerate(shapes)
    ]


def prefill(
    model: MossSpeechSGLangModel,
    prompt_embeddings: list[torch.Tensor],
    request_ids: list[str],
    *,
    use_host_attention_metadata: bool = False,
    request_slots: list[int] | None = None,
    max_context_len: int = 16,
) -> tuple[MossSpeechModelOutput, TorchNativeAttnBackend]:
    device = model.embed_tokens.weight.device
    token_lengths = [embeddings.shape[0] for embeddings in prompt_embeddings]
    token_count = sum(token_lengths)
    if request_slots is None:
        request_slots = list(range(1, len(token_lengths) + 1))
    else:
        assert len(request_slots) == len(token_lengths)
    request_pool = ReqToTokenPool(
        size=max(request_slots),
        max_context_len=max(max_context_len, max(token_lengths) + 2),
        device=str(device),
        enable_memory_saver=False,
    )
    kv_pool = MHATokenToKVPool(
        size=max(32, token_count + len(token_lengths) + 2, 2 * max_context_len),
        page_size=1,
        dtype=torch.bfloat16,
        head_num=model.config.num_key_value_heads,
        head_dim=model.head_dim,
        layer_num=model.total_attention_layers,
        device=str(device),
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    next_slot = 1
    for request_index, token_length in zip(request_slots, token_lengths, strict=True):
        request_pool.req_to_token[request_index, :token_length] = torch.arange(
            next_slot, next_slot + token_length, device=device
        )
        next_slot += token_length
    backend_class = (
        MossSpeechTorchNativeBackend
        if use_host_attention_metadata
        else TorchNativeAttnBackend
    )
    backend = backend_class(
        AttentionPools(
            device=str(device),
            req_to_token_pool=request_pool,
            token_to_kv_pool=kv_pool,
        )
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(token_lengths),
        input_ids=torch.zeros(token_count, dtype=torch.long, device=device),
        req_pool_indices=torch.tensor(request_slots, device=device),
        seq_lens=torch.tensor(token_lengths, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(token_lengths, dtype=torch.int64, device="cpu"),
        req_pool_indices_cpu=torch.tensor(request_slots, device="cpu"),
        out_cache_loc=torch.arange(1, token_count + 1, device=device),
        seq_lens_sum=token_count,
        positions=torch.cat(
            [torch.arange(length, device=device) for length in token_lengths]
        ),
        extend_seq_lens=torch.tensor(token_lengths, dtype=torch.int32, device=device),
        extend_prefix_lens=torch.zeros(
            len(token_lengths), dtype=torch.int32, device=device
        ),
        extend_seq_lens_cpu=token_lengths,
        extend_prefix_lens_cpu=[0] * len(token_lengths),
        extend_num_tokens=token_count,
        rids=request_ids,
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        output = model(
            batch.input_ids,
            batch.positions,
            batch,
            input_embeds=torch.cat(prompt_embeddings),
        )
    return output, backend


def decode(
    model: MossSpeechSGLangModel,
    backend: TorchNativeAttnBackend,
    prompt_lengths: list[int],
    feedback_embeddings: torch.Tensor,
    request_ids: list[str],
    *,
    request_slots: list[int] | None = None,
) -> MossSpeechModelOutput:
    device = model.embed_tokens.weight.device
    batch_size = len(prompt_lengths)
    if request_slots is None:
        request_slots = list(range(1, batch_size + 1))
    else:
        assert len(request_slots) == batch_size
    token_slots = torch.arange(
        sum(prompt_lengths) + 1,
        sum(prompt_lengths) + 1 + batch_size,
        device=device,
    )
    for batch_index, (request_index, token_length) in enumerate(
        zip(request_slots, prompt_lengths, strict=True)
    ):
        backend.req_to_token_pool.req_to_token[request_index, token_length] = (
            token_slots[batch_index]
        )
    model.decode_input_embedding.weight[:batch_size].copy_(feedback_embeddings)
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=torch.arange(batch_size, device=device),
        req_pool_indices=torch.tensor(request_slots, device=device),
        seq_lens=torch.tensor(
            [length + 1 for length in prompt_lengths], dtype=torch.int32, device=device
        ),
        out_cache_loc=token_slots,
        seq_lens_sum=sum(prompt_lengths) + batch_size,
        seq_lens_cpu=torch.tensor(
            [length + 1 for length in prompt_lengths], dtype=torch.int64, device="cpu"
        ),
        positions=torch.tensor(prompt_lengths, device=device),
        rids=request_ids,
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        return model(batch.input_ids, batch.positions, batch)


def assert_full_decode_graph(model: MossSpeechSGLangModel, *, capacity=16) -> dict:
    """Replay full dual heads and all KV layers, including output reuse and close."""
    device = model.embed_tokens.weight.device
    prompt = model.embed_tokens(torch.tensor([1, 2, 3], device=device))
    _, backend = prefill(model, [prompt], ["prompt"], max_context_len=capacity)
    graph_lengths = tuple(range(1, min(capacity, 512) + 1))
    manager = DecodeGraph(model, backend, lengths=graph_lengths)
    model.decode_graph = manager
    try:
        memory_before = {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
        }
        started = perf_counter()
        manager.initialize()
        startup_seconds = perf_counter() - started
        memory_after = {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
        }
        assert len(manager.entries) == len(graph_lengths), manager.reason
        snapshots = []
        generator = torch.Generator(device=device).manual_seed(9901)
        lengths = ((512, 1023) if capacity > 512 else ()) + tuple(
            n - 1 for n in graph_lengths
        )
        for length in lengths:
            # Fill all earlier KV rows deterministically, including unused-gap rows.
            backend.req_to_token_pool.req_to_token[1, :length] = torch.arange(
                1, length + 1, device=device
            )
            for cache in (
                *backend.token_to_kv_pool.k_buffer,
                *backend.token_to_kv_pool.v_buffer,
            ):
                cache[1 : length + 1].normal_(mean=0.0, std=0.125, generator=generator)
            feedback = model.embed_tokens(torch.tensor([length], device=device))
            model.decode_graph = None
            expected = decode(model, backend, [length], feedback, ["eager"])
            expected_kv = [
                t.clone()
                for t in (
                    *backend.token_to_kv_pool.k_buffer,
                    *backend.token_to_kv_pool.v_buffer,
                )
            ]
            model.decode_graph = manager
            actual = decode(model, backend, [length], feedback, ["graph"])
            for name in ("text_logits", "audio_logits", "hidden_states"):
                torch.testing.assert_close(
                    getattr(actual, name), getattr(expected, name), rtol=0, atol=0
                )
            for actual_kv, reference in zip(
                (
                    *backend.token_to_kv_pool.k_buffer,
                    *backend.token_to_kv_pool.v_buffer,
                ),
                expected_kv,
            ):
                torch.testing.assert_close(
                    actual_kv[1 : length + 2], reference[1 : length + 2], rtol=0, atol=0
                )
            snapshots.append((actual, actual.text_logits.clone()))
        assert manager.hits == len(graph_lengths)
        for actual, saved in snapshots:
            torch.testing.assert_close(actual.text_logits, saved, rtol=0, atol=0)
        # Sampling may mutate an owned output without poisoning the next replay.
        snapshots[-1][0].text_logits.fill_(float("-inf"))
        again = decode(model, backend, [lengths[-1]], feedback, ["reuse"])
        torch.testing.assert_close(
            again.text_logits, expected.text_logits, rtol=0, atol=0
        )
        observed = []
        handle = model.layers[0].register_forward_pre_hook(
            lambda module, args: observed.append(len(args))
        )
        previous_hits = manager.hits
        try:
            hooked = decode(model, backend, [lengths[-1]], feedback, ["hooked"])
            torch.testing.assert_close(
                hooked.text_logits, expected.text_logits, rtol=0, atol=0
            )
            assert observed == [5] and manager.hits == previous_hits
        finally:
            handle.remove()
        manager.close()
        manager.close()
        after = decode(model, backend, [lengths[-1]], feedback, ["closed"])
        torch.testing.assert_close(
            after.audio_logits, expected.audio_logits, rtol=0, atol=0
        )
        original_body = DecodeEntry.body

        def fail_warmup(entry: DecodeEntry) -> MossSpeechModelOutput:
            original_body(entry)
            raise RuntimeError("injected warmup failure")

        failed = DecodeGraph(model, backend, lengths=(1,))
        with (
            patch.object(DecodeEntry, "body", fail_warmup),
            patch(
                "torch.cuda.synchronize", wraps=torch.cuda.synchronize
            ) as synchronize,
            pytest.raises(RuntimeError, match="injected warmup failure"),
        ):
            failed.initialize()
        assert synchronize.call_count > 0
        assert failed.closed and not failed.entries
        return {
            "captures": len(graph_lengths),
            "startup_seconds": startup_seconds,
            "memory_before_bytes": memory_before,
            "memory_after_bytes": memory_after,
            "replays": manager.hits,
            "complete_heads_exact": True,
            "kv_layers": model.total_attention_layers,
            "owned_output_and_close": True,
            "partial_warmup_failure_synchronized": True,
            "failed_warmup_synchronize_calls": synchronize.call_count,
        }
    finally:
        manager.close()
        model.decode_graph = None
        model.dual_logits_by_request_id.clear()


def assert_full_prefill_graph(model: MossSpeechSGLangModel, lengths=(3, 7)) -> dict:
    """Full heads/KV exact under graph reuse and noncontiguous allocated slots."""
    device = model.embed_tokens.weight.device
    max_length = max(lengths)
    prompt = model.embed_tokens(torch.arange(1, max_length + 1, device=device))
    _, backend = prefill(model, [prompt], ["initial"], max_context_len=max_length)
    manager = PrefillGraphs(model, backend, lengths)
    model.prefill_graph = manager
    try:
        manager.initialize()
        assert len(manager.entries) == len(lengths), manager.reason
        snapshots = []
        for length in (*lengths, lengths[0]):
            model.prefill_graph = None
            expected, reference_backend = prefill(
                model, [prompt[:length]], ["reference"], max_context_len=max_length
            )
            locations = torch.arange(2, 2 * length + 1, 2, device=device)
            backend.req_to_token_pool.req_to_token[1, :length] = locations
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=torch.zeros(length, dtype=torch.long, device=device),
                positions=torch.arange(length, device=device),
                req_pool_indices=torch.tensor([1], device=device),
                seq_lens=torch.tensor([length], dtype=torch.int32, device=device),
                out_cache_loc=locations,
                seq_lens_sum=length,
                extend_seq_lens=torch.tensor(
                    [length], dtype=torch.int32, device=device
                ),
                extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
                extend_seq_lens_cpu=[length],
                extend_prefix_lens_cpu=[0],
                extend_num_tokens=length,
                rids=["graph"],
            )
            model.prefill_graph = manager
            with forward_context(ForwardContext(attn_backend=backend)):
                actual = model(
                    batch.input_ids,
                    batch.positions,
                    batch,
                    input_embeds=prompt[:length],
                )
            for name in ("text_logits", "audio_logits", "hidden_states"):
                torch.testing.assert_close(
                    getattr(actual, name), getattr(expected, name), atol=0, rtol=0
                )
            for cache, oracle in zip(
                (
                    *backend.token_to_kv_pool.k_buffer,
                    *backend.token_to_kv_pool.v_buffer,
                ),
                (
                    *reference_backend.token_to_kv_pool.k_buffer,
                    *reference_backend.token_to_kv_pool.v_buffer,
                ),
            ):
                torch.testing.assert_close(
                    cache[locations], oracle[1 : length + 1], atol=0, rtol=0
                )
            snapshots.append((actual, actual.text_logits.clone()))
        assert manager.hits == len(lengths) + 1
        for output, saved in snapshots:
            torch.testing.assert_close(output.text_logits, saved, atol=0, rtol=0)
        output.text_logits.fill_(float("-inf"))
        with forward_context(ForwardContext(attn_backend=backend)):
            again = model(
                batch.input_ids, batch.positions, batch, input_embeds=prompt[:length]
            )
        torch.testing.assert_close(
            again.text_logits, expected.text_logits, atol=0, rtol=0
        )
        observed = []
        handle = model.layers[0].register_forward_pre_hook(
            lambda module, args: observed.append(len(args))
        )
        previous_hits = manager.hits
        try:
            with forward_context(ForwardContext(attn_backend=backend)):
                hooked = model(
                    batch.input_ids,
                    batch.positions,
                    batch,
                    input_embeds=prompt[:length],
                )
            torch.testing.assert_close(
                hooked.text_logits, expected.text_logits, rtol=0, atol=0
            )
            assert observed == [5] and manager.hits == previous_hits
        finally:
            handle.remove()
        manager.close()
        manager.close()
        assert (
            manager.try_replay(batch.input_ids, batch.positions, batch, prompt[:length])
            is None
        )
        original_body = PrefillEntry.body

        def fail_warmup(entry: PrefillEntry) -> MossSpeechModelOutput:
            original_body(entry)
            raise RuntimeError("injected warmup failure")

        failed = PrefillGraphs(model, backend, (lengths[0],))
        with (
            patch.object(PrefillEntry, "body", fail_warmup),
            patch(
                "torch.cuda.synchronize", wraps=torch.cuda.synchronize
            ) as synchronize,
            pytest.raises(RuntimeError, match="injected warmup failure"),
        ):
            failed.initialize()
        assert synchronize.call_count > 0
        assert failed.closed and not failed.entries
        return {
            "captures": len(lengths),
            "replays": manager.hits,
            "lengths": list(lengths),
            "kv_layers": model.total_attention_layers,
            "complete_heads_and_scattered_kv_exact": True,
            "owned_outputs_and_close": True,
            "partial_warmup_failure_synchronized": True,
            "failed_warmup_synchronize_calls": synchronize.call_count,
        }
    finally:
        manager.close()
        model.prefill_graph = None
        model.dual_logits_by_request_id.clear()


def assert_ar_combination(
    model: MossSpeechSGLangModel,
    optimizations: list[str],
    projection_batch_size: int = 4,
    graph_lengths: tuple[int, ...] = (3, 7),
    native_norm_library: str | None = None,
    decode_graph_lengths: tuple[int, ...] = LENGTHS,
) -> dict:
    """Exercise selected AR paths together against uncaptured Torch heads and KV."""
    selected = set(optimizations)
    device = model.embed_tokens.weight.device
    assert device.type == "cuda"
    projection_batch_size = (
        projection_batch_size if "ar-projection-graph" in selected else 1
    )
    request_count = (
        max(2, projection_batch_size)
        if "mixed-modalities" in selected
        else projection_batch_size
    )
    assert model.decode_input_embedding.num_embeddings >= request_count
    if "context-prefill-graph" in selected:
        assert graph_lengths and all(1 <= length <= 512 for length in graph_lengths)
    else:
        graph_lengths = (3, 7)
    norm_modules = [
        module for module in model.modules() if isinstance(module, MossSpeechRMSNorm)
    ]
    previous_native = [module.native_norm for module in norm_modules]
    native = NativeNorm(native_norm_library) if "native-norm" in selected else None
    for module in norm_modules:
        module.native_norm = native
    host_enabled = "ar-host-metadata" in selected
    lifecycle = MossSpeechRequestLifecycle(
        model, mixed_modalities="mixed-modalities" in selected
    )
    build_request, _ = make_moss_speech_scheduler_adapters(lifecycle=lifecycle)
    runner = RowCommitRunner(model)
    requests = []
    graph_owners = []
    saved_outputs = []
    activations = {
        name: {"executions": 0, "complete_heads_exact": True, "all_kv_exact": True}
        for name in selected
    }

    @contextmanager
    def eager_reference() -> Iterator[None]:
        saved_graphs = model.decode_graph, model.prefill_graph, model.projection_graphs
        model.decode_graph = model.prefill_graph = model.projection_graphs = None
        for module in norm_modules:
            module.native_norm = None
        try:
            yield
        finally:
            model.decode_graph, model.prefill_graph, model.projection_graphs = (
                saved_graphs
            )
            for module in norm_modules:
                module.native_norm = native

    def compare_outputs(
        actual: MossSpeechModelOutput, expected: MossSpeechModelOutput
    ) -> None:
        for attribute in ("text_logits", "audio_logits", "hidden_states"):
            torch.testing.assert_close(
                getattr(actual, attribute), getattr(expected, attribute), rtol=0, atol=0
            )
        saved_outputs.append(
            (actual, actual.text_logits.clone(), actual.audio_logits.clone())
        )

    try:
        with DynamicGradMode():
            lengths = [
                graph_lengths[index % len(graph_lengths)]
                for index in range(request_count)
            ]
            prompts = []
            request_ids = []
            for index, length in enumerate(lengths):
                request_id = f"combined-{index}"
                modality = (
                    "audio" if "mixed-modalities" in selected and index % 2 else "text"
                )
                rows = (
                    [[151667, 7]] * length
                    if modality == "audio"
                    else [[12, 512]] * length
                )
                state = MossSpeechState(
                    input_grid=rows,
                    output_modality=modality,
                    max_new_tokens=4,
                    temperature=0,
                    top_k=1,
                    repetition_penalty=1,
                )
                with torch.device("cpu"):
                    request = build_request(
                        StagePayload(
                            request_id, OmniRequest(inputs=None), state.to_dict()
                        )
                    )
                assert isinstance(request, MossSpeechSGLangRequestData)
                requests.append(request)
                request_ids.append(request_id)
                prompts.append(runner.rows_to_embeddings(request.prompt_rows))
            assert lifecycle.snapshot() == {
                "requests": request_count,
                "active": request_count,
                "pending": 0,
            }
            context_capacity = max(
                1024, sum(graph_lengths) + request_count * max(graph_lengths) + 32
            )
            actual, backend = prefill(
                model,
                prompts,
                request_ids,
                use_host_attention_metadata=host_enabled,
                max_context_len=context_capacity,
            )
            with eager_reference():
                expected, reference_backend = prefill(
                    model, prompts, request_ids, max_context_len=context_capacity
                )
            compare_outputs(actual, expected)
            written_slots = set(range(1, sum(lengths) + 1))
            next_slot = sum(lengths) + 1
            layers = [*model.layers, *model.text_block, *model.audio_block]
            if "ar-decode-graph" in selected:
                model.decode_graph = DecodeGraph(
                    model, backend, lengths=decode_graph_lengths
                )
                graph_owners.append(model.decode_graph)
                model.decode_graph.initialize()
                assert model.decode_graph.entries, model.decode_graph.reason
            else:
                pass
            if "context-prefill-graph" in selected:
                model.prefill_graph = PrefillGraphs(model, backend, graph_lengths)
                graph_owners.append(model.prefill_graph)
                model.prefill_graph.initialize()
                assert model.prefill_graph.entries, model.prefill_graph.reason
            else:
                pass
            if "ar-projection-graph" in selected:
                model.projection_graphs = MossSpeechProjectionGraphs(
                    layers, batch_size=projection_batch_size
                )
                graph_owners.append(model.projection_graphs)
            else:
                pass

            def compare_kv() -> None:
                slots = torch.tensor(sorted(written_slots), device=device)
                for actual_cache, expected_cache in zip(
                    (
                        *backend.token_to_kv_pool.k_buffer,
                        *backend.token_to_kv_pool.v_buffer,
                    ),
                    (
                        *reference_backend.token_to_kv_pool.k_buffer,
                        *reference_backend.token_to_kv_pool.v_buffer,
                    ),
                    strict=True,
                ):
                    torch.testing.assert_close(
                        actual_cache[slots], expected_cache[slots], rtol=0, atol=0
                    )

            # Reuse the private prefill bank with live slots, then consume its KV in decode.
            for length in graph_lengths:
                locations = torch.arange(next_slot, next_slot + length, device=device)
                next_slot += length
                for attention_backend in (backend, reference_backend):
                    attention_backend.req_to_token_pool.req_to_token[1, :length] = (
                        locations
                    )
                prompt = model.embed_tokens(torch.arange(1, length + 1, device=device))
                batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=torch.zeros(length, dtype=torch.long, device=device),
                    positions=torch.arange(length, device=device),
                    req_pool_indices=torch.tensor([1], device=device),
                    req_pool_indices_cpu=torch.tensor([1], device="cpu"),
                    seq_lens=torch.tensor([length], dtype=torch.int32, device=device),
                    seq_lens_cpu=torch.tensor([length], device="cpu"),
                    out_cache_loc=locations,
                    seq_lens_sum=length,
                    extend_seq_lens=torch.tensor(
                        [length], dtype=torch.int32, device=device
                    ),
                    extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
                    extend_seq_lens_cpu=[length],
                    extend_prefix_lens_cpu=[0],
                    extend_num_tokens=length,
                    rids=[request_ids[0]],
                )
                reference_backend.init_forward_metadata(batch)
                with (
                    eager_reference(),
                    forward_context(ForwardContext(attn_backend=reference_backend)),
                ):
                    expected = model(
                        batch.input_ids, batch.positions, batch, input_embeds=prompt
                    )
                backend.init_forward_metadata(batch)
                with forward_context(ForwardContext(attn_backend=backend)):
                    actual = model(
                        batch.input_ids, batch.positions, batch, input_embeds=prompt
                    )
                compare_outputs(actual, expected)
                lengths[0] = length
                written_slots.update(locations.tolist())
                compare_kv()

            def run_decode(schedule: list[int]) -> None:
                nonlocal next_slot
                locations = torch.arange(
                    next_slot, next_slot + len(schedule), device=device
                )
                next_slot += len(schedule)
                prompt_lengths = [lengths[slot - 1] for slot in schedule]
                for attention_backend in (backend, reference_backend):
                    for index, slot in enumerate(schedule):
                        attention_backend.req_to_token_pool.req_to_token[
                            slot, prompt_lengths[index]
                        ] = locations[index]
                feedback = model.embed_tokens(
                    torch.tensor([20 + slot for slot in schedule], device=device)
                )
                model.decode_input_embedding.weight[: len(schedule)].copy_(feedback)
                batch = ForwardBatch(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=len(schedule),
                    input_ids=torch.arange(len(schedule), device=device),
                    positions=torch.tensor(prompt_lengths, device=device),
                    req_pool_indices=torch.tensor(schedule, device=device),
                    seq_lens=torch.tensor(
                        [length + 1 for length in prompt_lengths],
                        dtype=torch.int32,
                        device=device,
                    ),
                    seq_lens_cpu=torch.tensor(
                        [length + 1 for length in prompt_lengths], device="cpu"
                    ),
                    out_cache_loc=locations,
                    seq_lens_sum=sum(prompt_lengths) + len(schedule),
                    rids=[request_ids[slot - 1] for slot in schedule],
                )
                reference_backend.init_forward_metadata(batch)
                with (
                    eager_reference(),
                    forward_context(ForwardContext(attn_backend=reference_backend)),
                ):
                    expected = model(batch.input_ids, batch.positions, batch)
                backend.init_forward_metadata(batch)
                with forward_context(ForwardContext(attn_backend=backend)):
                    actual = model(batch.input_ids, batch.positions, batch)
                compare_outputs(actual, expected)
                written_slots.update(locations.tolist())
                compare_kv()
                for slot in schedule:
                    lengths[slot - 1] += 1

            schedules = [
                [1],
                list(range(1, request_count + 1)),
                [request_count],
                list(range(request_count, 0, -1)),
            ]
            if "ar-decode-graph" in selected:
                schedules.extend([[1], [1]])
            elif projection_batch_size == 1 and "ar-projection-graph" in selected:
                schedules.append([1])
            else:
                pass
            for step_index, schedule in enumerate(schedules):
                if step_index >= 4:
                    lengths[0] = (
                        511
                        if step_index == 4 and "ar-decode-graph" in selected
                        else 512
                    )
                    previous_slots = torch.arange(1, lengths[0] + 1, device=device)
                    for attention_backend in (backend, reference_backend):
                        attention_backend.req_to_token_pool.req_to_token[
                            1, : lengths[0]
                        ] = previous_slots
                    generator = torch.Generator(device=device).manual_seed(791)
                    for candidate_cache, reference_cache in zip(
                        (
                            *backend.token_to_kv_pool.k_buffer,
                            *backend.token_to_kv_pool.v_buffer,
                        ),
                        (
                            *reference_backend.token_to_kv_pool.k_buffer,
                            *reference_backend.token_to_kv_pool.v_buffer,
                        ),
                        strict=True,
                    ):
                        initialized = (
                            torch.randn(
                                candidate_cache[1 : lengths[0] + 1].shape,
                                device=device,
                                dtype=candidate_cache.dtype,
                                generator=generator,
                            )
                            * 0.1
                        )
                        candidate_cache[1 : lengths[0] + 1].copy_(initialized)
                        reference_cache[1 : lengths[0] + 1].copy_(initialized)
                    written_slots.update(range(1, lengths[0] + 1))
                    next_slot = lengths[0] + 1
                else:
                    pass
                run_decode(schedule)
                if step_index == 3:
                    # Cancel a live request and reuse its admission identity independently of graph buffers.
                    survivor = requests[1] if request_count > 1 else None
                    if survivor is not None:
                        survivor.rng_generator = torch.Generator(
                            device=device
                        ).manual_seed(survivor.effective_seed)
                        survivor_rng = survivor.rng_generator.get_state().clone()
                        survivor_prompt = survivor.prompt_rows.clone()
                        survivor_rows = list(survivor.output_rows)
                    else:
                        pass
                    lifecycle.release(request_ids[0], aborted=True)
                    assert not any(
                        owner[0] == request_ids[0]
                        for owner in model.dual_logits_by_request_id
                    )
                    with torch.device("cpu"):
                        replacement_state = MossSpeechState(
                            input_grid=[[14, 512]] * graph_lengths[0],
                            output_modality="text",
                            max_new_tokens=4,
                            temperature=0,
                            top_k=1,
                            repetition_penalty=1,
                        )
                        replacement = build_request(
                            StagePayload(
                                "combined-replacement",
                                OmniRequest(inputs=None),
                                replacement_state.to_dict(),
                            )
                        )
                    assert isinstance(replacement, MossSpeechSGLangRequestData)
                    assert "combined-replacement" in lifecycle.active
                    replacement_prompt = runner.rows_to_embeddings(
                        replacement.prompt_rows
                    )
                    # Both pools reuse the retired request slot, while the new request owns its output.
                    replacement_slots = torch.arange(
                        1, len(replacement_prompt) + 1, device=device
                    )
                    for attention_backend in (backend, reference_backend):
                        attention_backend.req_to_token_pool.req_to_token[
                            1, : len(replacement_prompt)
                        ] = replacement_slots
                    replacement_batch = ForwardBatch(
                        forward_mode=ForwardMode.EXTEND,
                        batch_size=1,
                        input_ids=torch.zeros(
                            len(replacement_prompt), dtype=torch.long, device=device
                        ),
                        positions=torch.arange(len(replacement_prompt), device=device),
                        req_pool_indices=torch.tensor([1], device=device),
                        req_pool_indices_cpu=torch.tensor([1], device="cpu"),
                        seq_lens=torch.tensor(
                            [len(replacement_prompt)], dtype=torch.int32, device=device
                        ),
                        seq_lens_cpu=torch.tensor(
                            [len(replacement_prompt)], device="cpu"
                        ),
                        out_cache_loc=replacement_slots,
                        seq_lens_sum=len(replacement_prompt),
                        extend_seq_lens=torch.tensor(
                            [len(replacement_prompt)], dtype=torch.int32, device=device
                        ),
                        extend_prefix_lens=torch.zeros(
                            1, dtype=torch.int32, device=device
                        ),
                        extend_seq_lens_cpu=[len(replacement_prompt)],
                        extend_prefix_lens_cpu=[0],
                        extend_num_tokens=len(replacement_prompt),
                        rids=["combined-replacement"],
                    )
                    reference_backend.init_forward_metadata(replacement_batch)
                    with (
                        eager_reference(),
                        forward_context(ForwardContext(attn_backend=reference_backend)),
                    ):
                        expected = model(
                            replacement_batch.input_ids,
                            replacement_batch.positions,
                            replacement_batch,
                            input_embeds=replacement_prompt,
                        )
                    backend.init_forward_metadata(replacement_batch)
                    with forward_context(ForwardContext(attn_backend=backend)):
                        actual = model(
                            replacement_batch.input_ids,
                            replacement_batch.positions,
                            replacement_batch,
                            input_embeds=replacement_prompt,
                        )
                    compare_outputs(actual, expected)
                    compare_kv()
                    request_ids[0] = "combined-replacement"
                    requests[0] = replacement
                    lengths[0] = len(replacement_prompt)
                    if survivor is not None:
                        run_decode([2])
                        assert (
                            not survivor.finished
                            and survivor.effective_seed
                            == survivor.rng_generator.initial_seed()
                        )
                        assert survivor.output_rows == survivor_rows
                        torch.testing.assert_close(
                            survivor.prompt_rows, survivor_prompt, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            survivor.rng_generator.get_state(),
                            survivor_rng,
                            rtol=0,
                            atol=0,
                        )
                    else:
                        pass
                else:
                    pass

            guard_ids = torch.zeros(1, dtype=torch.long, device=device)
            guard_positions = torch.ones(1, dtype=torch.long, device=device)
            guard_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=1,
                input_ids=guard_ids,
                positions=guard_positions,
                req_pool_indices=torch.ones(1, dtype=torch.long, device=device),
                seq_lens=torch.tensor([2], dtype=torch.int32, device=device),
                seq_lens_cpu=torch.tensor([2], device="cpu"),
                out_cache_loc=torch.ones(1, dtype=torch.long, device=device),
                seq_lens_sum=2,
                rids=["guard"],
            )
            if model.decode_graph is not None:
                caches = (
                    *backend.token_to_kv_pool.k_buffer,
                    *backend.token_to_kv_pool.v_buffer,
                )
                saved_rows = [cache[1].clone() for cache in caches]
                hits_before = model.decode_graph.hits
                with forward_context(ForwardContext(attn_backend=backend)):
                    assert (
                        model.decode_graph.try_replay(
                            guard_ids, guard_positions, guard_batch
                        )
                        is not None
                    )
                assert model.decode_graph.hits == hits_before + 1
                for cache, row in zip(caches, saved_rows, strict=True):
                    cache[1].copy_(row)
            else:
                pass
            projection_hidden = model.decode_input_embedding.weight[
                :projection_batch_size
            ]
            projection_positions = torch.arange(projection_batch_size, device=device)
            if model.projection_graphs is not None:
                assert model.projection_graphs.can_replay(
                    projection_hidden, projection_positions, is_decode=True
                )
            else:
                pass

            def assert_full_graph_fallbacks() -> None:
                for graph, ids, positions, request_batch, embeds in (
                    (model.decode_graph, guard_ids, guard_positions, guard_batch, None),
                    (
                        model.prefill_graph,
                        replacement_batch.input_ids,
                        replacement_batch.positions,
                        replacement_batch,
                        replacement_prompt,
                    ),
                ):
                    if graph is not None:
                        hits_before = graph.hits
                        assert graph.identity() != graph.signature
                        with forward_context(ForwardContext(attn_backend=backend)):
                            assert (
                                graph.try_replay(ids, positions, request_batch, embeds)
                                is None
                            )
                        assert graph.hits == hits_before
                    else:
                        pass

            # A changed norm implementation must invalidate every captured family.
            norm = norm_modules[0]
            original_epsilon = norm.variance_epsilon
            norm.variance_epsilon *= 2
            try:
                assert_full_graph_fallbacks()
                if model.projection_graphs is not None:
                    assert (
                        model.projection_graphs.model_state()
                        != model.projection_graphs.prepared_model_state
                    )
                    if native is None:
                        replays_before = model.projection_graphs.replays
                        assert not model.projection_graphs.can_replay(
                            projection_hidden, projection_positions, is_decode=True
                        )
                        assert model.projection_graphs.replays == replays_before
                    else:
                        pass
                else:
                    pass
            finally:
                norm.variance_epsilon = original_epsilon
            if native is not None:
                assert native.calls > 0
                activations["native-norm"]["executions"] = native.calls
                if graph_owners:
                    assert any(
                        graph.native_capture_calls > 0 for graph in graph_owners
                    ) or native.calls > sum(
                        graph.native_capture_calls for graph in graph_owners
                    )
                else:
                    pass
                norm.native_norm = None
                try:
                    assert_full_graph_fallbacks()
                    if model.projection_graphs is not None:
                        assert (
                            model.projection_graphs.model_state()
                            != model.projection_graphs.prepared_model_state
                        )
                        replays_before = model.projection_graphs.replays
                        assert not model.projection_graphs.can_replay(
                            projection_hidden, projection_positions, is_decode=True
                        )
                        assert model.projection_graphs.replays == replays_before
                    else:
                        pass
                finally:
                    norm.native_norm = native
            else:
                pass
            for feature, graph in (
                ("ar-decode-graph", model.decode_graph),
                ("context-prefill-graph", model.prefill_graph),
                ("ar-projection-graph", model.projection_graphs),
            ):
                if graph is not None:
                    replays = (
                        graph.replays
                        if feature == "ar-projection-graph"
                        else graph.hits
                    )
                    captures = (
                        len(graph.sections)
                        if feature == "ar-projection-graph"
                        else len(graph.entries)
                    )
                    assert replays > 0 and captures > 0
                    activations[feature].update(
                        executions=replays,
                        captures=captures,
                        replays=replays,
                        native_capture_calls=graph.native_capture_calls,
                    )
                else:
                    pass
            if host_enabled:
                assert (
                    isinstance(backend, MossSpeechTorchNativeBackend)
                    and backend.executions > 0
                )
                activations["ar-host-metadata"]["executions"] = backend.executions
            else:
                pass
            if "mixed-modalities" in selected:
                assert {lifecycle.active[request_id] for request_id in request_ids} == {
                    "text",
                    "audio",
                }
                runner.report_mixed_batch = True
                scheduled_requests = [
                    SchedulerRequest(request_id=request.req.rid, data=request)
                    for request in requests
                ]
                with torch.device("cpu"):
                    runner.collect_step(
                        GenerationBatchResult(), scheduled_requests, phase="decode"
                    )
                assert not runner.report_mixed_batch
                activations["mixed-modalities"]["executions"] = 1
            else:
                pass
            for output, text_snapshot, audio_snapshot in saved_outputs:
                torch.testing.assert_close(
                    output.text_logits, text_snapshot, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    output.audio_logits, audio_snapshot, rtol=0, atol=0
                )
            return {
                "optimizations": activations,
                "complete_heads_exact": True,
                "all_kv_exact": True,
                "owned_outputs": True,
                "norm_mutation_guard": True,
            }
    finally:
        try:
            for graph in reversed(graph_owners):
                graph.close()
        finally:
            model.decode_graph = model.prefill_graph = model.projection_graphs = None
            for request_id in list(lifecycle.requests):
                lifecycle.release(request_id)
            for module, previous in zip(norm_modules, previous_native, strict=True):
                module.native_norm = previous
