# SPDX-License-Identifier: Apache-2.0
"""Request-local replay preserves whole-batch ownership and failure boundaries."""

import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor import forward_context as context_api
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang_omni.models.moss_speech.decode_graphs import DecodeEntry, DecodeGraph


@pytest.fixture
def serial_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DecodeGraph, ForwardBatch, list[int]]:
    model = torch.nn.Module()
    model.decode_input_embedding = torch.nn.Embedding(8, 3)
    model.eval()
    backend = object.__new__(TorchNativeAttnBackend)
    table = torch.zeros(9, 129, dtype=torch.int64)
    for i in range(4):
        table[2 * i + 1, :128] = torch.arange(1, 129) + i * 128
    backend.req_to_token_pool = SimpleNamespace(req_to_token=table)
    backend.token_to_kv_pool = SimpleNamespace(k_buffer=[torch.zeros(514, 1, 1)])
    graph = DecodeGraph(
        model, backend, lengths=(4, 8, 32, 128), serial_batch_replay=True
    )
    graph.identity = Mock(return_value=())
    graph.signature, graph.stream = (), 17
    executions = []
    for length in graph.lengths:
        entry = object.__new__(DecodeEntry)
        entry.ids = torch.zeros(1, dtype=torch.int64)
        entry.positions = torch.zeros_like(entry.ids)
        entry.req = torch.zeros_like(entry.ids)
        entry.locations = torch.zeros_like(entry.ids)
        entry.output = SimpleNamespace(
            text_logits=torch.zeros(1, 8),
            audio_logits=torch.zeros(1, 4),
            hidden_states=torch.zeros(2, 1, 3),
        )

        def compute(entry: DecodeEntry = entry, length: int = length) -> None:
            executions.append(length)
            marker = (
                int(entry.ids[0]) * 1000
                + int(entry.req[0]) * 10
                + int(entry.locations[0])
            )
            entry.output.text_logits.fill_(marker)
            entry.output.audio_logits.fill_(marker + 1)
            entry.output.hidden_states.fill_(marker + 2)
            backend.token_to_kv_pool.k_buffer[0][entry.locations] = marker

        entry.graph = SimpleNamespace(replay=Mock(side_effect=compute))
        graph.entries[length] = entry
    monkeypatch.setattr(context_api, "has_forward_context", lambda: True)
    monkeypatch.setattr(context_api, "get_attn_backend", lambda: backend)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=17)
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
        input_ids=torch.arange(4),
        positions=torch.tensor([7] * 4),
        req_pool_indices=torch.tensor([1, 3, 5, 7]),
        out_cache_loc=torch.tensor([8, 136, 264, 392]),
        seq_lens_cpu=torch.tensor([8] * 4),
        seq_lens=torch.tensor([8] * 4, dtype=torch.int32),
        seq_lens_sum=32,
    )
    return graph, batch, executions


@torch.no_grad()
def test_serial_replay_preserves_rows_and_owned_outputs(
    serial_graph: tuple[DecodeGraph, ForwardBatch, list[int]]
) -> None:
    """Packing and repeated entries cannot reorder or overwrite another request."""
    graph, batch, executions = serial_graph
    for count in (2, 4):
        for order in itertools.permutations(range(count)):
            for heterogeneous in (False, True):
                lengths = [graph.lengths[i] if heterogeneous else 8 for i in order]
                batch.batch_size = count
                batch.input_ids = torch.tensor(order)
                batch.positions = torch.tensor([length - 1 for length in lengths])
                batch.req_pool_indices = torch.tensor([2 * i + 1 for i in order])
                batch.out_cache_loc = torch.tensor(
                    [128 * i + length for i, length in zip(order, lengths)]
                )
                batch.seq_lens_cpu = torch.tensor(lengths)
                batch.seq_lens = torch.tensor(lengths, dtype=torch.int32)
                batch.seq_lens_sum = sum(lengths)
                original = [
                    value.clone()
                    for value in (
                        batch.input_ids,
                        batch.positions,
                        batch.req_pool_indices,
                        batch.out_cache_loc,
                    )
                ]
                actual = graph.try_replay(batch.input_ids, batch.positions, batch)
                expected = (
                    batch.input_ids * 1000
                    + batch.req_pool_indices * 10
                    + batch.out_cache_loc
                )
                assert torch.equal(actual.text_logits[:, 0], expected.float())
                assert torch.equal(actual.audio_logits[:, 0], expected.float() + 1)
                assert actual.hidden_states.shape == (2, count, 3)
                assert torch.equal(actual.hidden_states[0, :, 0], expected.float() + 2)
                retained = actual.audio_logits.clone()
                actual.text_logits.fill_(float("-inf"))
                assert torch.isfinite(actual.next_token_logits).all()
                again = graph.try_replay(batch.input_ids, batch.positions, batch)
                assert torch.isfinite(again.text_logits).all()
                assert torch.equal(actual.audio_logits, retained)
                assert all(
                    torch.equal(a, b)
                    for a, b in zip(
                        original,
                        (
                            batch.input_ids,
                            batch.positions,
                            batch.req_pool_indices,
                            batch.out_cache_loc,
                        ),
                    )
                )
                assert executions[-count:] == lengths
    assert graph.serial_batches[2] > 0 and graph.serial_batches[4] > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("seq_lens_cpu", torch.tensor([8, 8, 8, 513])),
        ("seq_lens_cpu", torch.tensor([8, 8, 8, 40960])),
        ("seq_lens_cpu", torch.tensor([8, 8, 8, 7])),
        ("seq_lens_cpu", torch.tensor([8, 8, 8])),
        ("seq_lens_cpu", None),
        ("seq_lens_cpu", torch.tensor([8.0] * 4)),
        ("seq_lens", torch.tensor([8, 8, 8, 9], dtype=torch.int32)),
        ("seq_lens", torch.tensor([8] * 3, dtype=torch.int32)),
        ("seq_lens_sum", 31),
        ("input_ids", torch.arange(4, dtype=torch.int32)),
        ("input_ids", torch.arange(8)[::2]),
        ("input_ids", torch.tensor([0, 1, 2, 8])),
        ("positions", torch.tensor([7, 7, 7, 8])),
        ("req_pool_indices", torch.tensor([1, 3, 5, 5])),
        ("req_pool_indices", torch.tensor([1, 3, 5, 9])),
        ("out_cache_loc", torch.tensor([8, 136, 264, 8])),
        ("out_cache_loc", torch.tensor([8, 136, 264, 513])),
    ],
)
@torch.no_grad()
def test_serial_replay_rejects_entire_batch_before_mutation(
    serial_graph: tuple[DecodeGraph, ForwardBatch, list[int]],
    field: str,
    value: torch.Tensor | int | None,
) -> None:
    """An invalid final row must not replay or write KV for earlier valid rows."""
    graph, batch, executions = serial_graph
    setattr(batch, field, value)
    before = graph.backend.token_to_kv_pool.k_buffer[0].clone()
    assert graph.try_replay(batch.input_ids, batch.positions, batch) is None
    assert executions == [] and graph.hits == 0
    assert torch.equal(before, graph.backend.token_to_kv_pool.k_buffer[0])


@torch.no_grad()
def test_serial_replay_propagates_partial_failure(
    serial_graph: tuple[DecodeGraph, ForwardBatch, list[int]]
) -> None:
    """A later replay error propagates instead of retrying partially mutated KV."""
    graph, batch, executions = serial_graph
    entry = graph.entries[8]
    compute = entry.graph.replay.side_effect
    attempts = 0

    def fail_second() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("second-row failure")
        else:
            compute()

    entry.graph.replay.side_effect = fail_second
    with pytest.raises(RuntimeError, match="second-row failure"):
        graph.try_replay(batch.input_ids, batch.positions, batch)
    assert attempts == 2 and executions == [8]


@pytest.mark.parametrize(
    "fallback",
    [
        "disabled",
        "batch3",
        "training",
        "closed",
        "identity",
        "prefill",
        "hook",
        "embedding",
    ],
)
@torch.no_grad()
def test_serial_replay_keeps_default_and_runtime_fallbacks(
    serial_graph: tuple[DecodeGraph, ForwardBatch, list[int]], fallback: str
) -> None:
    """The new opt-in cannot expand unsupported runtime or prefill execution."""
    graph, batch, executions = serial_graph
    embedding = None
    if fallback == "disabled":
        graph.serial_batch_replay = False
    elif fallback == "batch3":
        batch.batch_size = 3
    elif fallback == "training":
        graph.model.train()
    elif fallback == "closed":
        graph.closed = True
    elif fallback == "identity":
        graph.signature = None
    elif fallback == "prefill":
        batch.forward_mode = ForwardMode.EXTEND
    elif fallback == "hook":
        graph.model.register_forward_pre_hook(lambda *_: None)
    else:
        embedding = torch.zeros(4, 3)
    assert graph.try_replay(batch.input_ids, batch.positions, batch, embedding) is None
    assert executions == [] and graph.hits == 0
