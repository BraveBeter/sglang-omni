# SPDX-License-Identifier: Apache-2.0
"""Real radix-tree tests with CPU KV indices and canonical dual-channel keys."""
from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey

from sglang_omni.models.moss_speech.prefix_cache import (
    MossSpeechRadixCache,
    attach_cache_key,
    canonical_key,
    record_cache_output,
)


class Pool:
    def __init__(self) -> None:
        self.req_to_token = torch.arange(32, dtype=torch.int64).reshape(2, 16)

    def write(self, indices, values) -> None:
        self.req_to_token[indices] = values


class Allocator:
    device = "cpu"

    def __init__(self) -> None:
        self.freed: list[int] = []

    def free(self, values: torch.Tensor) -> None:
        self.freed.extend(values.tolist())


@pytest.fixture
def cache() -> MossSpeechRadixCache:
    return MossSpeechRadixCache(CacheInitParams(False, Pool(), Allocator(), 1))


def request(rows: list[tuple[int, int]], slot: int = 0) -> SimpleNamespace:
    selected = array("q", [a if t == 151667 else t for t, a in rows])
    req = SimpleNamespace(
        origin_input_ids=selected,
        output_ids=array("q"),
        extra_key=None,
        req_pool_idx=slot,
        cache_protected_len=0,
        last_node=None,
        priority=0,
        session=None,
    )
    req.get_fill_ids = lambda: req.origin_input_ids + req.output_ids
    attach_cache_key(req, rows)
    return req


def match(cache: MossSpeechRadixCache, req, limit: int | None = None) -> torch.Tensor:
    return cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(req.get_fill_ids(), extra_key=req.extra_key, limit=limit),
            req=req,
        )
    ).device_indices


def test_same_selected_ids_do_not_alias_embeddings(cache) -> None:
    text = request([(42, 512), (8, 512)])
    audio = request([(151667, 42), (8, 512)])
    assert text.origin_input_ids == audio.origin_input_ids
    cache.insert(InsertParams(key=canonical_key(text), value=torch.tensor([3, 4])))
    assert match(cache, text).tolist() == [3, 4]
    assert match(cache, audio).numel() == 0
    assert text.origin_input_ids == array("q", [42, 8])


def test_shared_prefix_and_matching_limit(cache) -> None:
    first = request([(11, 512), (151667, 900), (12, 512)])
    second = request([(11, 512), (151667, 900), (13, 512)])
    cache.insert(InsertParams(key=canonical_key(first), value=torch.tensor([1, 2, 3])))
    assert match(cache, second).tolist() == [1, 2]
    assert match(cache, first, 1).tolist() == [1]
    assert match(cache, first, 0).numel() == 0


def test_uncommitted_feedback_is_not_part_of_cache_key(cache) -> None:
    req = request([(11, 512)])
    record_cache_output(req, (151667, 42))
    assert len(canonical_key(req)) == 1
    req.output_ids.append(42)
    cache.insert(InsertParams(key=canonical_key(req), value=torch.tensor([1, 2])))
    other = request([(11, 512)])
    record_cache_output(other, (42, 512))
    other.output_ids.append(42)
    assert req.get_fill_ids() == other.get_fill_ids()
    assert match(cache, other).tolist() == [1]


def test_finished_request_inserts_keys_and_releases_duplicate_slots(cache) -> None:
    first = request([(11, 512), (151667, 42)])
    cache.cache_finished_req(first, kv_len_to_handle=2)
    second = request([(11, 512), (151667, 42)], slot=1)
    assert match(cache, second).tolist() == [0, 1]
    cache.cache_finished_req(second, kv_len_to_handle=2)
    assert cache.token_to_kv_pool_allocator.freed == [16, 17]
    cache.evict(EvictParams(num_tokens=2))
    assert match(cache, second).numel() == 0
    assert set(cache.token_to_kv_pool_allocator.freed) == {0, 1, 16, 17}


def test_unfinished_request_updates_real_owner_and_drops_lock_on_finish(cache) -> None:
    req = request([(11, 512), (151667, 42)])
    req.last_node = cache.root_node
    cache.cache_unfinished_req(req)
    assert req.cache_protected_len == 2
    assert req.prefix_indices.tolist() == [0, 1]
    assert req.last_node is not cache.root_node
    cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=2)
    cache.evict(EvictParams(num_tokens=2))
    assert set(cache.token_to_kv_pool_allocator.freed) == {0, 1}


def test_raw_key_without_canonical_request_cannot_match(cache) -> None:
    req = request([(42, 512)])
    cache.insert(InsertParams(key=canonical_key(req), value=torch.tensor([1])))
    assert (
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [42])))
        ).device_indices.numel()
        == 0
    )


def test_missing_committed_row_fails_closed(cache) -> None:
    req = request([(42, 512)])
    req.output_ids.append(7)
    with pytest.raises(ValueError, match="committed"):
        canonical_key(req)


def test_different_prefill_shapes_do_not_share_kv():
    """BF16 kernels depend on the original prefill matrix row count."""
    from types import SimpleNamespace

    def key(rows):
        req = SimpleNamespace(extra_key=None, output_ids=[])
        attach_cache_key(req, rows)
        return canonical_key(req)

    a = key([[10, 512], [11, 512]])
    b = key([[10, 512], [11, 512], [12, 512]])
    assert a.extra_key != b.extra_key
