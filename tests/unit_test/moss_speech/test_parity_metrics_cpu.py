# SPDX-License-Identifier: Apache-2.0
"""Parity metrics must reject errors hidden by sparse/NaN reductions."""
import torch

from tests.unit_test.moss_speech.reference import compare_logits as compare


def test_nan_and_inf_patterns_are_hard_failures():
    assert not compare(torch.tensor([float("nan")]), torch.tensor([float("nan")]))[
        "pass"
    ]
    assert not compare(torch.tensor([float("inf")]), torch.tensor([-float("inf")]))[
        "pass"
    ]
    assert compare(
        torch.tensor([-float("inf"), 1.0]), torch.tensor([-float("inf"), 1.0])
    )["pass"]


def test_sparse_large_error_fails_absolute_cap():
    ref = torch.zeros(100000)
    value = ref.clone()
    value[0] = 5.0
    assert not compare(value, ref)["pass"]


def test_fraction_and_combined_bound_are_independent():
    assert not compare(torch.full((100,), 1.01), torch.zeros(100))["pass"]
    assert compare(torch.full((100,), 102.0), torch.full((100,), 100.0))["pass"]
