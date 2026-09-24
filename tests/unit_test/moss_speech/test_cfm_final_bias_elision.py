"""Original admission must remain intact when omitting proven final bias."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "original_final_bias_tests",
    Path(__file__).resolve().with_name("test_cfm_final_bias.py"),
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture
def data(monkeypatch):
    d = reference.data.__wrapped__(monkeypatch)
    d.owner = d.module.FinalBiasAlignment(elide_zero_bias=True)
    return d


def test_exact_optin_solve_cache_and_no_allocation(data, monkeypatch):
    d = data
    monkeypatch.setattr(
        d.module.F, "pad", Mock(side_effect=AssertionError("bias allocation"))
    )
    monkeypatch.setattr(
        torch.Tensor, "item", Mock(side_effect=AssertionError("device read"))
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        Mock(side_effect=AssertionError("device synchronization")),
    )
    with torch.inference_mode():
        assert reference.apply(d) is None
        assert reference.apply(d) is None
        assert reference.apply(d, cache={}) is None
    state = d.owner.status()
    assert state["elide_zero_bias"] is True
    assert (
        state["prepared"] == 2 and state["reused"] == 1 and state["fallback_calls"] == 0
    )
    assert len(d.cache) == 1 and list(d.cache.values()) == [None]
    assert not any(isinstance(v, torch.Tensor) for v in vars(d.owner).values())
    d.owner.close()
    d.owner.close()
    with torch.inference_mode():
        assert reference.apply(d) is d.bias


@pytest.mark.parametrize(
    "case",
    [
        "streaming",
        "no_proof",
        "fake_proof",
        "mask_alias",
        "no_cache",
        "layout",
        "dtype",
        "length",
        "subclass",
        "method",
        "mask_function",
        "training",
        "grad",
        "policy",
        "runtime",
        "closed",
    ],
)
def test_all_original_fallbacks(data, monkeypatch, case):
    reference.test_unknown_context_returns_original_bias(data, monkeypatch, case)


def test_only_explicit_boolean_mode():
    module = reference.api()
    assert module.FinalBiasAlignment().status()["elide_zero_bias"] is False
    for value in (1, "true", None):
        with pytest.raises(ValueError):
            module.FinalBiasAlignment(elide_zero_bias=value)
