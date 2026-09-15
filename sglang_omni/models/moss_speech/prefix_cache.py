# SPDX-License-Identifier: Apache-2.0
"""Canonical dual-channel keys over SGLang's existing radix KV storage.

The packed integers are cache identities only. Model inputs and scheduler
sampling IDs retain their original representation, with one KV slot per row.
"""
from __future__ import annotations

from array import array
from dataclasses import replace
from typing import Any, Iterable

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

_NAMESPACE = "moss-speech-dual-grid-v1:"


def _row_key(row: Iterable[int]) -> int:
    text, audio = (int(value) for value in row)
    if not 0 <= text < 151680 or not 0 <= audio < 16512:
        raise ValueError("MOSS-Speech cache row is outside the canonical vocabularies")
    return (text << 32) | audio


def attach_cache_key(req: Any, rows: Iterable[Iterable[int]]) -> None:
    req._moss_prompt_cache_ids = array("q", (_row_key(row) for row in rows))
    req._moss_output_cache_ids = array("q")


def record_cache_output(req: Any, row: Iterable[int]) -> None:
    output = getattr(req, "_moss_output_cache_ids", None)
    if output is not None:
        output.append(_row_key(row))


class _CacheRequest:
    """Expose canonical IDs while forwarding ownership updates to the real Req."""

    def __init__(self, req: Any) -> None:
        object.__setattr__(self, "_request", req)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._request, name, value)

    @property
    def origin_input_ids(self) -> array:
        return self._request._moss_prompt_cache_ids

    @property
    def output_ids(self) -> array:
        committed = len(self._request.output_ids)
        rows = self._request._moss_output_cache_ids
        if len(rows) < committed:
            raise ValueError("MOSS-Speech cache is missing committed output rows")
        return rows[:committed]

    @property
    def extra_key(self) -> str:
        # BF16 projection kernels can round differently at different M.
        # Reuse only KV created with the same original prefill row count.
        return _NAMESPACE + repr((len(self.origin_input_ids), self._request.extra_key))

    def get_fill_ids(self) -> array:
        return (self.origin_input_ids + self.output_ids)[
            : len(self._request.get_fill_ids())
        ]


def canonical_key(req: Any, limit: int | None = None) -> RadixKey:
    view = _CacheRequest(req)
    return RadixKey(
        view.origin_input_ids + view.output_ids,
        extra_key=view.extra_key,
        limit=limit,
    )


class MossSpeechRadixCache(RadixCache):
    """Use canonical keys for both prefix lookups and all KV insertion paths."""

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if params.req is not None:
            key = canonical_key(params.req, len(params.key))
        elif str(params.key.extra_key).startswith(_NAMESPACE):
            # Internal matching performed by cache_unfinished_req after insertion.
            key = params.key
        else:
            # A token-only scheduling estimate has insufficient model semantics.
            key = RadixKey(array("q"), extra_key=_NAMESPACE)
        return super().match_prefix(replace(params, key=key))

    def cache_finished_req(
        self, req: Any, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:
        super().cache_finished_req(
            _CacheRequest(req), is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )

    def cache_unfinished_req(self, req: Any, chunked: bool = False) -> None:
        super().cache_unfinished_req(_CacheRequest(req), chunked=chunked)
