"""Regression tests for explicit generation-parameter marking (GAP-1).

A user-specified length must be recorded in ``explicit_generation_params``
regardless of the alias used (``max_tokens`` / ``max_completion_tokens`` /
``max_new_tokens``); before the fix only ``max_new_tokens`` was inspected,
which never exists on ``ChatCompletionRequest``, so user lengths were
indistinguishable from endpoint defaults.
"""

from __future__ import annotations

from sglang_omni.serve.openai_api import _explicit_generation_params
from sglang_omni.serve.protocol import ChatCompletionRequest


def _req(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kwargs
    )


def test_max_tokens_alias_is_explicit() -> None:
    assert _explicit_generation_params(_req(max_tokens=16)) == ["max_tokens"]


def test_max_completion_tokens_alias_is_explicit() -> None:
    assert _explicit_generation_params(_req(max_completion_tokens=16)) == [
        "max_completion_tokens"
    ]


def test_length_aliases_do_not_collapse_before_marking() -> None:
    # both aliases set: both are recorded (downstream resolves priority)
    req = _req(max_tokens=8, max_completion_tokens=16)
    assert _explicit_generation_params(req) == ["max_completion_tokens", "max_tokens"]


def test_unset_length_is_not_marked() -> None:
    assert _explicit_generation_params(_req(temperature=0.7)) == ["temperature"]


def test_sampling_defaults_filled_by_endpoint_are_not_marked() -> None:
    # explicit values equal to endpoint defaults are still explicit (marked),
    # while absent fields stay unmarked — the core contract.
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=1.0,  # explicit but equal to the fill-in default
    )
    assert _explicit_generation_params(req) == ["temperature"]
