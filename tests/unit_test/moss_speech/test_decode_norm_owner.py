"""Actual CPU Tensor contracts for a startup-only complete-decode owner."""

from __future__ import annotations

import ast
import importlib.util
import threading
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import torch

MODEL = Path(__file__).resolve().parents[3] / "sglang_omni/models/moss_speech"
SPEC = importlib.util.spec_from_file_location(
    "decode_norm_under_test", MODEL / "decode_norm_pointwise.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def native_class(path):
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MossSpeechRMSNorm"
    )
    scope = dict(
        torch=torch,
        RMSNorm=torch.nn.Module,
        Any=Any,
        Optional=Optional,
        graph_norm=helper.graph_norm,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["MossSpeechRMSNorm"]


Norm = native_class(MODEL / "sglang_model.py")


class OriginalNorm(torch.nn.Module):
    """Preserve the checkpoint's explicit BF16 rounding boundaries.

    Residual addition is rounded before normalization; the normalized value
    is rounded before multiplying the learned weight. A fused float32
    residual+norm or norm+weight operation changes these semantics.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Any:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        value = self.weight * value.to(dtype)
        return value if residual is None else (value, residual)


def norm(width=4096, cls=Norm):
    value = cls()
    value.weight = torch.nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
    value.variance_epsilon = 1e-6
    return value.eval()


def native(x, weight, epsilon):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    return dict(output=weight * value.to(x.dtype))


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.norms = torch.nn.ModuleList(
            [norm(4096 if i < 82 else 128) for i in range(162)]
        )
        self.model.eval()
        self.owner = helper.DecodeNormOwner(self.model, Norm, native)
        self.model._moss_decode_norm_owner = self.owner

    def body(self, capacity=128):
        for i, module in enumerate(self.model.norms):
            shape = (1, 4096) if i < 82 else (1, 32, 128) if i < 122 else (1, 8, 128)
            module(torch.ones(shape, dtype=torch.bfloat16))

    def test_no_token_preserves_original_autograd_and_residual(self):
        actual, reference = self.model.norms[0], norm(cls=OriginalNorm)
        for residual_enabled in (False, True):
            x = torch.ones(1, 4096, dtype=torch.bfloat16, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            residual = torch.ones_like(x) if residual_enabled else None
            post = torch.ones_like(x) if residual_enabled else None
            a, b = actual(x, residual, post), reference(y, residual, post)
            ao, bo = a[0] if residual_enabled else a, b[0] if residual_enabled else b
            self.assertTrue(torch.equal(ao, bo))
            self.assertEqual(type(ao.grad_fn), type(bo.grad_fn))
            ao.sum().backward()
            bo.sum().backward()
            self.assertTrue(torch.equal(x.grad, y.grad))
            self.assertTrue(torch.equal(actual.weight.grad, reference.weight.grad))
            if residual_enabled:
                self.assertTrue(torch.equal(a[1], b[1]))
            actual.weight.grad = reference.weight.grad = None
        self.assertEqual(self.owner.status()["norm_calls"], 0)

    def test_three_capacities_three_bodies_exact_1458_calls(self):
        with (
            patch.object(helper, "_cuda_supported", return_value=True),
            torch.inference_mode(),
        ):
            for capacity in (128, 512, 1024):
                for _ in range(3):
                    with helper.norm_capture(self.owner, self.model, capacity):
                        self.body(capacity)
        status = self.owner.status()
        self.assertEqual(status["norm_calls"], 1458)
        self.assertEqual(status["bodies"], {"128": 3, "512": 3, "1024": 3})
        self.assertEqual(status["fallback_calls"], 0)
        self.assertEqual(status["partial_projection_calls"], 0)
        self.assertIsNone(helper.active_owner())
        self.assertTrue(all(n.weight.requires_grad for n in self.model.norms))

    def test_nested_and_wrong_context_rejected_and_restored(self):
        with self.assertRaisesRegex(ValueError, "inference"):
            with helper.norm_capture(self.owner, self.model, 128):
                pass
        with (
            torch.inference_mode(),
            patch.object(helper, "_cuda_supported", return_value=True),
        ):
            with self.assertRaisesRegex(ValueError, "nested"):
                with helper.norm_capture(self.owner, self.model, 128):
                    with helper.norm_capture(self.owner, self.model, 128):
                        pass
        self.assertIsNone(helper.active_owner())

    def test_missing_duplicate_or_unsupported_call_fails_closed(self):
        for mode in ("missing", "duplicate", "unsupported"):
            with (
                self.subTest(mode=mode),
                torch.inference_mode(),
                patch.object(
                    helper, "_cuda_supported", return_value=mode != "unsupported"
                ),
            ):
                with self.assertRaises(ValueError):
                    with helper.norm_capture(self.owner, self.model, 128):
                        if mode == "missing":
                            continue
                        module = self.model.norms[0]
                        module(torch.ones(1, 4096, dtype=torch.bfloat16))
                        if mode == "duplicate":
                            module(torch.ones(1, 4096, dtype=torch.bfloat16))
            self.assertIsNone(helper.active_owner())

    def test_exception_restores_tls_and_original_exception(self):
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, "sentinel"):
            with helper.norm_capture(self.owner, self.model, 128):
                raise RuntimeError("sentinel")
        self.assertIsNone(helper.active_owner())
        self.assertEqual(self.owner.status()["bodies"], {})

    def test_thread_does_not_inherit_token(self):
        seen = []
        with torch.inference_mode(), self.assertRaises(ValueError):
            with helper.norm_capture(self.owner, self.model, 128):
                worker = threading.Thread(
                    target=lambda: seen.append(helper.active_owner())
                )
                worker.start()
                worker.join()
        self.assertEqual(seen, [None])
        self.assertIsNone(helper.active_owner())

    def test_actual_partial_projection_rejected(self):
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "partial"):
            with helper.norm_capture(self.owner, self.model, 128):
                self.owner.observe_partial_projection()
        self.assertEqual(self.owner.status()["partial_projection_calls"], 1)
        self.assertIsNone(helper.active_owner())

    def test_replaced_weight_forward_or_training_rejected(self):
        module = self.model.norms[0]
        for mode in ("weight", "forward", "training"):
            with self.subTest(mode=mode):
                old_weight, old_training = (
                    module.weight,
                    module.training,
                )
                try:
                    if mode == "weight":
                        module.weight = torch.nn.Parameter(
                            module.weight.detach().clone()
                        )
                    elif mode == "forward":
                        module.forward = lambda x: x
                    else:
                        module.train()
                    with torch.inference_mode(), self.assertRaises(ValueError):
                        with helper.norm_capture(self.owner, self.model, 128):
                            pass
                finally:
                    module.weight, module.training = old_weight, old_training
                    if mode == "forward":
                        del module.forward
        self.assertIsNone(helper.active_owner())

    def test_cpu_dtype_layout_and_input_grad_fall_back(self):
        module = self.model.norms[0]
        values = [
            torch.ones(1, 4096, dtype=torch.bfloat16),
            torch.ones(1, 4096),
            torch.ones(1, 4096, dtype=torch.bfloat16, requires_grad=True),
        ]
        for value in values:
            with (
                self.subTest(dtype=value.dtype, grad=value.requires_grad),
                torch.inference_mode(),
            ):
                with self.assertRaises(ValueError):
                    with helper.norm_capture(self.owner, self.model, 128):
                        expected = native(
                            value, module.weight, module.variance_epsilon
                        )["output"]
                        self.assertTrue(torch.equal(module(value), expected))
        self.assertIsNone(helper.active_owner())

    def test_close_idempotent_and_busy_preserves_owner(self):
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "busy"):
            with helper.norm_capture(self.owner, self.model, 128):
                self.owner.close()
        self.assertFalse(self.owner.status()["closed"])
        helper.close_norm_owner(self.model)
        helper.close_norm_owner(self.model)
        self.assertTrue(self.owner.status()["closed"])
        self.assertIsNone(self.owner.engine)
        self.assertFalse(hasattr(self.model, "_moss_decode_norm_owner"))

    def test_size_one_stride_or_offset_outside_fixture_falls_back(self):
        values = [
            torch.ones(4096, dtype=torch.bfloat16).as_strided((1, 4096), (0, 1)),
            torch.ones(8192, dtype=torch.bfloat16).as_strided(
                (1, 4096), (4096, 1), 4096
            ),
        ]
        for x in values:
            with self.subTest(stride=x.stride(), offset=x.storage_offset()):
                before = self.owner.status()["norm_calls"]
                with (
                    torch.inference_mode(),
                    patch.object(helper, "_cuda_supported", return_value=True),
                ):
                    with self.assertRaises(ValueError):
                        with helper.norm_capture(self.owner, self.model, 128):
                            self.model.norms[0](x)
                            self.assertEqual(self.owner.status()["norm_calls"], before)

    def test_closed_status_preserves_actual_changed_flags(self):
        self.model.norms[0].weight.requires_grad_(False)
        self.assertFalse(self.owner.status()["norm_parameter_flags_preserved"])
        helper.close_norm_owner(self.model)
        self.assertFalse(self.owner.status()["norm_parameter_flags_preserved"])

    def test_unknown_capacity_and_fourth_body_rejected(self):
        with torch.inference_mode(), self.assertRaises(ValueError):
            with helper.norm_capture(self.owner, self.model, 127):
                pass
        with (
            patch.object(helper, "_cuda_supported", return_value=True),
            torch.inference_mode(),
        ):
            for _ in range(3):
                with helper.norm_capture(self.owner, self.model, 128):
                    self.body()
            with self.assertRaises(ValueError):
                with helper.norm_capture(self.owner, self.model, 128):
                    pass

    def test_none_owner_is_original_native_path(self):
        with helper.norm_capture(None, self.model, 128):
            self.assertIsNone(helper.active_owner())
            self.body()
        self.assertEqual(self.owner.status()["norm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
