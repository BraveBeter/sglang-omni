"""CPU tests for the model-private capture provider, without importing Torch."""

from __future__ import annotations

import ctypes as C
import importlib.util
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni/models/moss_speech/components/cfm_direct_lt.py"
)
spec = importlib.util.spec_from_file_location("cfm_direct_lt_test_target", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Device:
    type = "cuda"

    def __init__(self, index=0):
        self.index = index

    def __eq__(self, other):
        return isinstance(other, Device) and self.index == other.index

    def __str__(self):
        return f"cuda:{self.index}"


class Tensor:
    pointer = 0x10000

    def __init__(self, shape, *, dtype="float32", device=None, events=None):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = device or Device()
        self._base = None
        Tensor.pointer += 0x1000000
        self.pointer = Tensor.pointer
        self.events = events if events is not None else []
        self.copied_from = None
        self.custom_stride = None

    def data_ptr(self):
        return self.pointer

    def stride(self):
        if self.custom_stride is not None:
            return self.custom_stride
        result, product = [], 1
        for dimension in reversed(self.shape):
            result.insert(0, product)
            product *= dimension
        return tuple(result)

    def is_contiguous(self):
        return self.custom_stride is None

    def storage_offset(self):
        return 0

    def copy_(self, other):
        self.events.append("copy")
        self.copied_from = other
        return self


class Stream:
    def __init__(self, pointer, events):
        self.cuda_stream = pointer
        self.events = events

    def synchronize(self):
        self.events.append(("stream_sync", self.cuda_stream))


class Torch:
    float32 = "float32"
    uint8 = "uint8"
    __version__ = "2.9.1+cu128"

    def __init__(self, events):
        self.events = events
        self.device_index = 0
        self.stream = Stream(0x40, events)
        self.capturing = False
        self.sync_error = False
        self.allocated = []
        self.cuda = types.SimpleNamespace(
            current_device=lambda: self.device_index,
            current_stream=lambda device=None: self.stream,
            is_current_stream_capturing=lambda: self.capturing,
            synchronize=self.synchronize,
            get_device_capability=lambda device: (8, 0),
            get_device_name=lambda device: "NVIDIA A800-SXM4-80GB",
        )

    def device(self, value):
        return (
            value
            if isinstance(value, Device)
            else Device(int(str(value).split(":")[-1]))
        )

    def empty(self, shape, **kwargs):
        self.events.append("allocate")
        result = Tensor(shape, events=self.events, **kwargs)
        self.allocated.append(result)
        return result

    def synchronize(self, device=None):
        self.events.append("device_sync")
        if self.sync_error:
            raise RuntimeError("synchronize failed")


class Function:
    def __init__(self, body):
        self.body = body

    def __call__(self, *args):
        return self.body(*args)


class Lt:
    """Mock the actual C ABI, including readback sizes and typed arguments."""

    def __init__(self, events):
        self.events = events
        self.version = 120804
        self.fail = None
        self.wrong_size = False
        self.workspace = 4096
        self.alignment = 256
        self.split_k_support = 1
        self.reduction_scheme_mask = 1
        self.capability_reads = []
        self.values = {}
        self.config = {}
        self.pointer = 0x100
        self.launches = []
        self.calls = []
        names = (
            "GetVersion Create Destroy MatmulDescCreate MatmulDescDestroy "
            "MatmulDescSetAttribute MatmulDescGetAttribute MatrixLayoutCreate "
            "MatrixLayoutDestroy MatrixLayoutGetAttribute MatmulAlgoInit "
            "MatmulAlgoConfigSetAttribute MatmulAlgoConfigGetAttribute "
            "MatmulAlgoCapGetAttribute MatmulAlgoCheck Matmul"
        ).split()
        for name in names:
            setattr(
                self,
                f"cublasLt{name}",
                Function(lambda *a, name=name: self.call(name, *a)),
            )

    def allocate(self, output):
        self.pointer += 1
        output._obj.value = self.pointer
        return self.pointer

    def call(self, name, *args):
        self.calls.append(name)
        if name == self.fail:
            return 7
        if name == "GetVersion":
            return self.version
        if name == "Create":
            self.allocate(args[0])
        elif name == "MatmulDescCreate":
            ptr = self.allocate(args[0])
            self.values[ptr] = {0: args[1], 1: args[2], 8: None, 26: -1}
        elif name == "MatmulDescSetAttribute":
            owner, attr, value, size = args
            self.values[owner.value][attr] = value._obj.value
            expected_size = 8 if attr in (8, 10) else 4
            assert size == expected_size
        elif name == "MatrixLayoutCreate":
            output, dtype, rows, cols, ld = args
            ptr = self.allocate(output)
            self.values[ptr] = {0: dtype, 1: 0, 2: rows, 3: cols, 4: ld, 5: 1, 6: 0}
        elif name in ("MatmulDescGetAttribute", "MatrixLayoutGetAttribute"):
            owner, attr, output, size, written = args
            output._obj.value = self.values[owner.value][attr]
            written._obj.value = 1 if self.wrong_size else size
        elif name == "MatmulAlgoInit":
            assert args[1:8] == (68, 0, 0, 0, 0, 0, 0)
            self.config[C.addressof(args[8]._obj)] = {0: 0}
        elif name == "MatmulAlgoConfigSetAttribute":
            algo, attr, value, size = args
            assert size == (2 if attr >= 7 else 4)
            self.config[C.addressof(algo._obj)][attr] = value._obj.value
        elif name == "MatmulAlgoConfigGetAttribute":
            algo, attr, value, size, written = args
            value._obj.value = self.config[C.addressof(algo._obj)][attr]
            written._obj.value = size
        elif name == "MatmulAlgoCapGetAttribute":
            _, attr, value, size, written = args
            assert attr in (0, 1, 16, 17, 18, 19)
            self.capability_reads.append(attr)
            value._obj.value = (
                self.split_k_support
                if attr == 0
                else self.reduction_scheme_mask if attr == 1 else self.alignment
            )
            written._obj.value = size
        elif name == "MatmulAlgoCheck":
            args[-1]._obj.state = 0
            args[-1]._obj.workspaceSize = self.workspace
        elif name == "Matmul":
            self.launches.append(args)
            self.events.append("launch")
        elif name.endswith("Destroy"):
            self.events.append((name, args[0].value))
        return 0


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.torch = Torch(self.events)
        self.lib = Lt(self.events)
        self.patches = [
            patch.object(module, "_get_torch", return_value=self.torch),
            patch.object(
                module,
                "_open_loaded_library",
                return_value=(self.lib, Path("/mock/libcublasLt.so.12")),
            ),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patches)])

    def parameters(self, k=512):
        return Tensor((256, k)), Tensor((256,))

    def test_constructor_sets_exact_descriptors_and_reuses_context_workspace(self):
        context = module.DirectLtContext(Device())
        plans = []
        for k in (512, 1024):
            for rows in (240, 440, 2800, 3000):
                w, b = self.parameters(k)
                plan = module.DirectLtProjection(w, b, rows, context=context)
                plans.append(plan)
                info = plan.describe()
                self.assertEqual(
                    info["config"],
                    {str(i): v for i, v in enumerate((0, 14, 1, 0, 0, 0, 0, 0, 0))},
                )
                self.assertEqual(info["rows"], rows)
                self.assertEqual(info["k"], k)
                desc = self.lib.values[plan.desc.value]
                self.assertEqual(desc[7], 4)
                self.assertEqual(desc[8], b.data_ptr())
                self.assertEqual(desc[3], 1)
                self.assertEqual(desc[4], 0)
        workspaces = [x for x in self.torch.allocated if x.dtype == "uint8"]
        self.assertEqual(len(workspaces), 1)
        self.assertEqual(workspaces[0].shape, (1048576,))
        context.close()
        self.assertTrue(all(plan.closed for plan in plans))

    def test_run_copies_input_and_uses_exact_sixteen_argument_abi(self):
        w, b = self.parameters()
        plan = module.DirectLtProjection(w, b, 240)
        value = Tensor((2, 120, 512))
        before = len(self.lib.calls)
        allocations = len(self.torch.allocated)
        self.events.clear()
        self.torch.capturing = True
        output = plan.run(value)
        self.assertIs(output, plan.output)
        self.assertIs(plan.input.copied_from, value)
        self.assertIs(plan(value), output)
        self.assertEqual(self.events, ["copy", "launch", "copy", "launch"])
        self.assertEqual(self.lib.calls[before:], ["Matmul", "Matmul"])
        self.assertEqual(len(self.torch.allocated), allocations)
        args = self.lib.launches[-1]
        self.assertEqual(len(args), 16)
        self.assertEqual(args[3].value, w.data_ptr())
        self.assertEqual(args[5].value, plan.input.data_ptr())
        self.assertEqual(args[8].value, output.data_ptr())
        self.assertEqual(args[10].value, output.data_ptr())
        self.assertEqual(args[9].value, args[11].value)
        self.assertEqual(args[2]._obj.value, 1.0)
        self.assertEqual(args[7]._obj.value, 0.0)
        self.assertEqual(args[13].value, plan.context.workspace.data_ptr())
        self.assertEqual(args[14], 1048576)
        self.assertEqual(args[15].value, 0x40)
        self.assertEqual(plan.status()["calls"], 2)
        self.torch.capturing = False
        plan.close()

    def test_prepare_stream_orders_handoff_and_refuses_during_capture(self):
        context = module.DirectLtContext(Device())
        plan = module.DirectLtProjection(*self.parameters(), 240, context=context)
        self.events.clear()
        self.torch.stream = Stream(0x80, self.events)
        with self.assertRaisesRegex(RuntimeError, "stream"):
            plan.run(Tensor((2, 120, 512)))
        context.prepare_stream()
        self.assertEqual(self.events, [("stream_sync", 0x40)])
        plan.run(Tensor((2, 120, 512)))
        self.assertEqual(self.lib.launches[-1][-1].value, 0x80)
        self.torch.capturing = True
        with self.assertRaisesRegex(RuntimeError, "capture"):
            context.prepare_stream()
        with self.assertRaisesRegex(RuntimeError, "capture"):
            context.close()
        self.torch.capturing = False
        context.close()

    def test_other_thread_can_use_the_prepared_stream_serially(self):
        plan = module.DirectLtProjection(*self.parameters(), 240)
        results = []
        worker = threading.Thread(
            target=lambda: results.append(plan.run(Tensor((2, 120, 512))))
        )
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [plan.output])
        plan.close()

    def test_rejects_changed_current_device_tensor_layout_dtype_and_parameters(self):
        plan = module.DirectLtProjection(*self.parameters(), 240)
        for value in (
            Tensor((2, 120, 1024)),
            Tensor((2, 120, 512), dtype="float16"),
            Tensor((2, 120, 512), device=Device(1)),
        ):
            with self.assertRaises((ValueError, RuntimeError)):
                plan.run(value)
        value = Tensor((2, 120, 512))
        value.custom_stride = (1500 * 512, 512, 1)
        with self.assertRaises((ValueError, RuntimeError)):
            plan.run(value)
        self.torch.device_index = 1
        with self.assertRaisesRegex(RuntimeError, "device"):
            plan.run(Tensor((2, 120, 512)))
        self.torch.device_index = 0
        plan.weight.pointer += 256
        with self.assertRaisesRegex(RuntimeError, "parameter"):
            plan.run(Tensor((2, 120, 512)))
        self.assertEqual(len(self.lib.launches), 0)
        plan.close()

    def test_rejects_unsupported_shape_and_noncontiguous_weights(self):
        for k, rows in ((256, 240), (512, 120), (512, 300)):
            with self.assertRaises(ValueError):
                module.DirectLtProjection(*self.parameters(k), rows)
        weight, bias = self.parameters()
        weight.custom_stride = (1, 256)
        with self.assertRaises(ValueError):
            module.DirectLtProjection(weight, bias, 240)
        self.assertEqual(self.lib.calls, [])

    def test_abi_readback_workspace_alignment_and_status_fail_closed(self):
        self.assertEqual(C.sizeof(module._Algo), 64)
        self.assertEqual(C.sizeof(module._HeuristicResult), 96)
        for field, bad in (
            ("version", 120803),
            ("wrong_size", True),
            ("workspace", 2097152),
            ("alignment", 3),
            ("fail", "MatmulAlgoCheck"),
        ):
            with self.subTest(field=field):
                original = getattr(self.lib, field)
                setattr(self.lib, field, bad)
                with self.assertRaises(RuntimeError):
                    module.DirectLtProjection(*self.parameters(), 240)
                setattr(self.lib, field, original)

    def test_launch_error_is_propagated_and_context_cannot_launch_again(self):
        plan = module.DirectLtProjection(*self.parameters(), 240)
        self.lib.fail = "Matmul"
        with self.assertRaisesRegex(RuntimeError, "Matmul.*7"):
            plan.run(Tensor((2, 120, 512)))
        self.assertEqual(plan.calls, 0)
        self.lib.fail = None
        with self.assertRaisesRegex(RuntimeError, "failed"):
            plan.run(Tensor((2, 120, 512)))
        plan.close()

    def test_shared_projection_close_does_not_close_context(self):
        context = module.DirectLtContext(Device())
        first = module.DirectLtProjection(*self.parameters(), 240, context=context)
        second = module.DirectLtProjection(*self.parameters(1024), 240, context=context)
        first.close()
        self.assertTrue(first.closed)
        self.assertFalse(context.closed)
        second.run(Tensor((2, 120, 1024)))
        self.events.clear()
        context.close()
        context.close()
        self.assertEqual(self.events[0], "device_sync")
        self.assertEqual(sum(e == "device_sync" for e in self.events), 1)
        self.assertEqual(self.events[-1][0], "Destroy")
        self.assertIsNone(second.output)
        self.assertIsNone(context.workspace)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            second.run(Tensor((2, 120, 1024)))

    def test_sync_failure_retains_live_buffers_and_allows_close_retry(self):
        plan = module.DirectLtProjection(*self.parameters(), 240)
        output = plan.output
        self.torch.sync_error = True
        with self.assertRaisesRegex(RuntimeError, "synchronize failed"):
            plan.close()
        self.assertFalse(plan.closed)
        self.assertIs(plan.output, output)
        self.torch.sync_error = False
        plan.close()
        self.assertTrue(plan.closed)

    def test_partial_constructor_failure_destroys_only_created_resources(self):
        context = module.DirectLtContext(Device())
        self.lib.fail = "MatmulAlgoInit"
        self.events.clear()
        with self.assertRaisesRegex(RuntimeError, "AlgoInit"):
            module.DirectLtProjection(*self.parameters(), 240, context=context)
        destroyed = [e[0] for e in self.events if isinstance(e, tuple)]
        self.assertEqual(destroyed, ["MatrixLayoutDestroy"] * 3 + ["MatmulDescDestroy"])
        self.assertEqual(context.status()["projections"], 0)
        self.lib.fail = None
        context.close()

    def test_context_close_waits_for_an_inflight_host_launch(self):
        from concurrent.futures import ThreadPoolExecutor

        context = module.DirectLtContext(Device())
        plan = module.DirectLtProjection(*self.parameters(), 240, context=context)
        entered, release, closing, finished = (threading.Event() for _ in range(4))
        original = self.lib.cublasLtMatmul.body

        def launch(*args):
            entered.set()
            self.assertTrue(release.wait(5))
            return original(*args)

        def close():
            closing.set()
            context.close()
            finished.set()

        self.lib.cublasLtMatmul.body = launch
        with ThreadPoolExecutor(max_workers=2) as workers:
            future = workers.submit(plan.run, Tensor((2, 120, 512)))
            self.assertTrue(entered.wait(5))
            closing_future = workers.submit(close)
            self.assertTrue(closing.wait(5))
            self.assertFalse(finished.wait(0.05))
            self.assertFalse(plan.closed)
            release.set()
            future.result(timeout=5)
            closing_future.result(timeout=5)
        self.assertLess(self.events.index("launch"), self.events.index("device_sync"))
        self.assertTrue(plan.closed)

    def test_close_attempts_all_descriptors_after_destroy_failure(self):
        context = module.DirectLtContext(Device())
        plan = module.DirectLtProjection(*self.parameters(), 240, context=context)
        self.assertEqual(plan.owners, (plan.input, plan.output, plan.weight, plan.bias))
        self.assertEqual(context.owners, (context.workspace,))
        self.lib.fail = "MatrixLayoutDestroy"
        self.lib.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "MatrixLayoutDestroy.*7"):
            context.close()
        self.assertEqual(self.lib.calls.count("MatrixLayoutDestroy"), 3)
        self.assertIn("MatmulDescDestroy", self.lib.calls)
        self.assertEqual(self.lib.calls[-1], "Destroy")
        self.assertTrue(context.closed)
        self.assertEqual(plan.owners, ())
        self.assertEqual(context.owners, ())
        context.close()

    def test_explicit_ff_first_profile_uses_independently_qualified_geometry(self):
        context = module.DirectLtContext(Device())
        for rows in (240, 440, 2800, 3000):
            weight, bias = Tensor((1024, 256)), Tensor((1024,))
            plan = module.DirectLtProjection(
                weight, bias, rows, context=context, profile="ff_first"
            )
            info = plan.describe()
            self.assertEqual(info["profile"], "ff_first")
            self.assertEqual(info["output_features"], 1024)
            self.assertEqual(info["k"], 256)
            self.assertEqual(
                info["config"],
                {
                    str(i): v
                    for i, v in enumerate(
                        (0, 15 if rows in (240, 440) else 20, 1, 0, 0, 0, 0, 0, 0)
                    )
                },
            )
            self.assertEqual(
                [(row["2"], row["3"], row["4"]) for row in info["layouts"]],
                [(256, 1024, 256), (256, rows, 256), (1024, rows, 1024)],
            )
            self.assertEqual(plan.input.shape, (2, rows // 2, 256))
            self.assertEqual(plan.output.shape, (2, rows // 2, 1024))
            self.assertEqual(info["descriptor"]["8"], bias.data_ptr())
            self.assertEqual(info["descriptor"]["7"], 4)
        self.assertEqual(sum(t.dtype == "uint8" for t in self.torch.allocated), 1)
        context.close()

    def test_tail_tile_does_not_mutate_full_controls_or_other_profiles(self):
        context = module.DirectLtContext(Device())
        weight, bias = Tensor((1024, 256)), Tensor((1024,))
        tail = module.DirectLtProjection(
            weight, bias, 240, context=context, profile="ff_first"
        )
        full = module.DirectLtProjection(
            weight, bias, 2800, context=context, profile="ff_first"
        )
        other_tail = module.DirectLtProjection(
            weight, bias, 440, context=context, profile="ff_first"
        )
        other_full = module.DirectLtProjection(
            weight, bias, 3000, context=context, profile="ff_first"
        )
        ordinary = module.DirectLtProjection(*self.parameters(), 240, context=context)
        qkv = module.DirectLtProjection(
            Tensor((512, 256)), None, 240, context=context, profile="qkv_1400"
        )
        self.assertEqual(
            [
                p.describe()["config"]["1"]
                for p in (tail, full, other_tail, other_full, ordinary, qkv)
            ],
            [15, 20, 15, 20, 14, 16],
        )
        self.assertEqual(module._FF_FIRST_CONFIG[1], 20)
        self.assertEqual(module._CONFIG[1], 14)
        self.assertEqual(tail.describe()["config"]["2"], 1)
        self.assertEqual(tail.describe()["config"]["3"], 0)
        context.close()

    def test_profiles_reject_implicit_or_mixed_geometry(self):
        with self.assertRaises(ValueError):
            module.DirectLtProjection(Tensor((1024, 256)), Tensor((1024,)), 240)
        for profile, weight, bias in (
            ("ff_first", Tensor((256, 512)), Tensor((256,))),
            ("ff_first", Tensor((1024, 512)), Tensor((1024,))),
            ("ff_first", Tensor((1024, 256)), Tensor((256,))),
            ("unknown", Tensor((256, 512)), Tensor((256,))),
        ):
            with self.assertRaises(ValueError):
                module.DirectLtProjection(weight, bias, 240, profile=profile)
        self.assertEqual(self.lib.launches, [])

    def test_ff_first_and_old_profiles_share_context_without_hot_getters(self):
        context = module.DirectLtContext(Device())
        old = module.DirectLtProjection(*self.parameters(), 240, context=context)
        first = module.DirectLtProjection(
            Tensor((1024, 256)),
            Tensor((1024,)),
            240,
            context=context,
            profile="ff_first",
        )
        self.assertEqual(old.describe()["config"]["1"], 14)
        self.assertEqual(old.describe()["output_features"], 256)
        self.assertEqual(old.describe()["profile"], "output")
        self.assertEqual(first.describe()["config"]["1"], 15)
        self.events.clear()
        before = len(self.lib.calls)
        allocations = len(self.torch.allocated)
        value = Tensor((2, 120, 256))
        self.torch.capturing = True
        result = first(value)
        self.assertIs(first(value), result)
        self.assertIs(result, first.output)
        self.assertEqual(self.events, ["copy", "launch", "copy", "launch"])
        self.assertEqual(self.lib.calls[before:], ["Matmul", "Matmul"])
        self.assertEqual(len(self.torch.allocated), allocations)
        self.assertEqual(self.lib.launches[-1][13].value, context.workspace.data_ptr())
        self.assertEqual(self.lib.launches[-1][10].value, result.data_ptr())
        self.assertEqual(old.describe()["config"]["1"], 14)
        self.torch.capturing = False
        context.close()
        self.assertTrue(first.closed and old.closed)
        self.assertEqual(first.owners, ())

    def test_qkv_profiles_pin_shape_split_and_no_bias_descriptor(self):
        context = module.DirectLtContext(Device())
        for profile, rows_values, split in (
            ("qkv_1400", (240, 2800), 3),
            ("qkv_1500", (440, 3000), 2),
        ):
            for rows in rows_values:
                with self.subTest(profile=profile, rows=rows):
                    weight = Tensor((512, 256))
                    plan = module.DirectLtProjection(
                        weight, None, rows, context=context, profile=profile
                    )
                    info = plan.describe()
                    self.assertEqual(info["profile"], profile)
                    self.assertEqual(
                        info["config"],
                        {
                            str(i): v
                            for i, v in enumerate((0, 16, split, 1, 0, 0, 0, 0, 0))
                        },
                    )
                    self.assertEqual(info["descriptor"]["7"], 1)
                    self.assertIsNone(info["descriptor"]["8"])
                    self.assertEqual(info["descriptor"]["0"], 68)
                    self.assertEqual(info["descriptor"]["1"], 0)
                    self.assertEqual(
                        info["capabilities"],
                        {"split_k_support": 1, "reduction_scheme_mask": 1},
                    )
                    self.assertEqual(
                        [(x["2"], x["3"], x["4"]) for x in info["layouts"]],
                        [(256, 512, 256), (256, rows, 256), (512, rows, 512)],
                    )
                    self.assertEqual(plan.input.shape, (2, rows // 2, 256))
                    self.assertEqual(plan.output.shape, (2, rows // 2, 512))
                    self.assertEqual(plan.owners, (plan.input, plan.output, weight))
        self.assertEqual(self.lib.capability_reads.count(0), 4)
        self.assertEqual(self.lib.capability_reads.count(1), 4)
        self.assertEqual(self.lib.calls.count("MatmulAlgoCheck"), 4)
        self.assertEqual(sum(t.dtype == "uint8" for t in self.torch.allocated), 1)
        context.close()
        self.assertEqual(plan.owners, ())

    def test_qkv_profiles_reject_cross_shape_bias_and_mixed_geometry(self):
        for profile, rows, weight, bias in (
            ("qkv_1400", 440, Tensor((512, 256)), None),
            ("qkv_1400", 3000, Tensor((512, 256)), None),
            ("qkv_1500", 240, Tensor((512, 256)), None),
            ("qkv_1500", 2800, Tensor((512, 256)), None),
            ("qkv_1400", 240, Tensor((512, 256)), Tensor((512,))),
            ("qkv_1500", 440, Tensor((256, 512)), None),
            ("qkv_1500", 440, Tensor((512, 512)), None),
            ("output", 240, Tensor((256, 512)), None),
            ("ff_first", 240, Tensor((1024, 256)), None),
        ):
            with self.subTest(profile=profile, rows=rows):
                with self.assertRaises(ValueError):
                    module.DirectLtProjection(weight, bias, rows, profile=profile)
        self.assertEqual(self.lib.calls, [])

    def test_qkv_requires_split_and_inplace_capabilities_before_algorithm_check(self):
        context = module.DirectLtContext(Device())
        for field, value in (
            ("split_k_support", 0),
            ("reduction_scheme_mask", 0),
            ("reduction_scheme_mask", 2),
        ):
            with self.subTest(field=field, value=value):
                original = getattr(self.lib, field)
                setattr(self.lib, field, value)
                self.lib.calls.clear()
                with self.assertRaisesRegex(RuntimeError, "capabilit"):
                    module.DirectLtProjection(
                        Tensor((512, 256)),
                        None,
                        240,
                        context=context,
                        profile="qkv_1400",
                    )
                self.assertNotIn("MatmulAlgoCheck", self.lib.calls)
                self.assertNotIn("Matmul", self.lib.calls)
                self.assertIn("MatmulDescDestroy", self.lib.calls)
                self.assertEqual(context.plans, [])
                self.assertFalse(context.closed or context.failed)
                setattr(self.lib, field, original)
        # Extra supported reduction bits do not change the selected INPLACE1.
        self.lib.reduction_scheme_mask = 3
        plan = module.DirectLtProjection(
            Tensor((512, 256)), None, 440, context=context, profile="qkv_1500"
        )
        self.assertEqual(plan.describe()["config"]["3"], 1)
        context.close()

    def test_qkv_capability_query_or_algorithm_check_failure_cleans_descriptors(self):
        context = module.DirectLtContext(Device())
        for operation in ("MatmulAlgoCapGetAttribute", "MatmulAlgoCheck"):
            with self.subTest(operation=operation):
                self.lib.fail = operation
                self.lib.calls.clear()
                with self.assertRaisesRegex(RuntimeError, "status 7"):
                    module.DirectLtProjection(
                        Tensor((512, 256)),
                        None,
                        240,
                        context=context,
                        profile="qkv_1400",
                    )
                self.assertEqual(context.plans, [])
                self.assertEqual(self.lib.calls.count("MatrixLayoutDestroy"), 3)
                self.assertIn("MatmulDescDestroy", self.lib.calls)
                self.assertEqual(self.lib.launches, [])
        self.lib.fail = None
        context.close()

    def test_qkv_run_has_no_getters_allocations_or_fake_bias_and_returns_borrowed(self):
        context = module.DirectLtContext(Device())
        old = module.DirectLtProjection(*self.parameters(), 240, context=context)
        first = module.DirectLtProjection(
            Tensor((1024, 256)),
            Tensor((1024,)),
            240,
            context=context,
            profile="ff_first",
        )
        plans = [
            module.DirectLtProjection(
                Tensor((512, 256)), None, rows, context=context, profile=profile
            )
            for profile, rows in (("qkv_1400", 240), ("qkv_1500", 440))
        ]
        self.torch.capturing = True
        for plan in plans:
            value = Tensor((2, plan.rows // 2, 256))
            self.events.clear()
            before = len(self.lib.calls)
            allocations = len(self.torch.allocated)
            output = plan(value)
            self.assertIs(plan(value), output)
            self.assertIs(output, plan.output)
            self.assertIs(plan.input.copied_from, value)
            self.assertEqual(self.events, ["copy", "launch", "copy", "launch"])
            self.assertEqual(self.lib.calls[before:], ["Matmul", "Matmul"])
            self.assertEqual(len(self.torch.allocated), allocations)
            args = self.lib.launches[-1]
            self.assertEqual(args[2]._obj.value, 1.0)
            self.assertEqual(args[7]._obj.value, 0.0)
            self.assertEqual(args[3].value, plan.weight.data_ptr())
            self.assertEqual(args[5].value, plan.input.data_ptr())
            self.assertEqual(args[8].value, output.data_ptr())
            self.assertEqual(args[10].value, output.data_ptr())
            self.assertEqual(args[13].value, context.workspace.data_ptr())
            self.assertEqual(plan.calls, 2)
        self.assertEqual(old.describe()["config"]["1"], 14)
        self.assertEqual(first.describe()["config"]["1"], 15)
        self.torch.capturing = False
        context.close()
        self.assertTrue(all(plan.closed for plan in plans))

    def test_qkv_rejects_parameter_mutation_and_obeys_stream_handoff(self):
        plan = module.DirectLtProjection(
            Tensor((512, 256)), None, 240, profile="qkv_1400"
        )
        value = Tensor((2, 120, 256))
        plan.bias = Tensor((512,))
        with self.assertRaisesRegex(RuntimeError, "parameter"):
            plan(value)
        plan.bias = None
        plan.weight.pointer += 256
        with self.assertRaisesRegex(RuntimeError, "parameter"):
            plan(value)
        plan.weight.pointer -= 256
        self.torch.stream = Stream(0x80, self.events)
        with self.assertRaisesRegex(RuntimeError, "stream"):
            plan(value)
        plan.context.prepare_stream()
        results = []
        worker = threading.Thread(target=lambda: results.append(plan(value)))
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [plan.output])
        self.assertEqual(self.lib.launches[-1][-1].value, 0x80)
        plan.close()

    def test_original_tail_strides_copy_directly_to_same_prepared_compact_input(self):
        context = module.DirectLtContext(Device())
        for rows, length in ((240, 1400), (440, 1500)):
            for profile, k, output_features in (
                (f"qkv_{length}", 256, 512),
                ("output", 512, 256),
                ("ff_first", 256, 1024),
            ):
                with self.subTest(rows=rows, profile=profile):
                    bias = (
                        None
                        if profile.startswith("qkv")
                        else Tensor((output_features,))
                    )
                    plan = module.DirectLtProjection(
                        Tensor((output_features, k)),
                        bias,
                        rows,
                        context=context,
                        profile=profile,
                    )
                    source = Tensor((2, rows // 2, k))
                    source.custom_stride = (length * k, k, 1)
                    source.storage_offset = lambda: 1280 * k
                    identity = (
                        source.data_ptr(),
                        source.stride(),
                        source.storage_offset(),
                    )
                    before = len(self.lib.calls)
                    allocations = len(self.torch.allocated)
                    self.events.clear()
                    result = plan(source)
                    self.assertIs(plan.input.copied_from, source)
                    self.assertEqual(
                        identity,
                        (source.data_ptr(), source.stride(), source.storage_offset()),
                    )
                    self.assertEqual(self.events, ["copy", "launch"])
                    self.assertEqual(self.lib.calls[before:], ["Matmul"])
                    self.assertEqual(len(self.torch.allocated), allocations)
                    self.assertIs(result, plan.output)
                    self.assertEqual(plan.input.stride(), (rows // 2 * k, k, 1))
                    self.assertEqual(self.lib.launches[-1][5].value, plan.input_pointer)
        context.close()

    def test_strided_admission_refuses_unknown_batch_inner_length_precision(self):
        plan = module.DirectLtProjection(
            Tensor((512, 256)), None, 240, profile="qkv_1400"
        )
        for stride in (
            (1500 * 256, 256, 1),
            (1400 * 256, 257, 1),
            (1400 * 256, 256, 2),
            (0, 256, 1),
        ):
            value = Tensor((2, 120, 256))
            value.custom_stride = stride
            with self.assertRaises(ValueError):
                plan(value)
        for shape, dtype in (((2, 121, 256), "float32"), ((2, 120, 256), "float16")):
            value = Tensor(shape, dtype=dtype)
            value.custom_stride = (1400 * 256, 256, 1)
            with self.assertRaises(ValueError):
                plan(value)
        full = module.DirectLtProjection(
            Tensor((512, 256)), None, 2800, profile="qkv_1400"
        )
        value = Tensor((2, 1400, 256))
        value.custom_stride = (2680 * 256, 256, 1)
        with self.assertRaises(ValueError):
            full(value)
        self.assertEqual(self.lib.launches, [])
        plan.close()
        full.close()

    def test_real_cpu_view_copy_preserves_source_offset_and_signed_zero_bits(self):
        try:
            import torch as actual_torch
        except ImportError:
            self.skipTest("Actual CPU Tensor view test requires Torch")
        for length, k in ((1400, 256), (1500, 256), (1400, 512), (1500, 512)):
            source = actual_torch.arange(
                2 * length * k, dtype=actual_torch.float32
            ).reshape(2, length, k)
            source[:, 1280, 0] = -0.0
            saved = source.clone()
            view = source[:, 1280:, :]
            plan = module.DirectLtProjection.__new__(module.DirectLtProjection)
            plan.rows, plan.k = 2 * (length - 1280), k
            plan.closed = False
            plan.calls = 0
            plan.input = actual_torch.empty(2, length - 1280, k)
            plan.output = actual_torch.zeros(2, length - 1280, 512 if k == 256 else 256)
            plan.weight = actual_torch.ones(512 if k == 256 else 256, k)
            plan.bias = None
            plan.bias_pointer = None
            plan.input_pointer = plan.input.data_ptr()
            plan.output_pointer = plan.output.data_ptr()
            plan.weight_pointer = plan.weight.data_ptr()
            workspace = actual_torch.empty(1048576, dtype=actual_torch.uint8)
            plan.workspace_pointer = workspace.data_ptr()

            def launch(*args):
                self.assertEqual(args[5].value, plan.input.data_ptr())
                self.assertTrue(
                    actual_torch.equal(
                        plan.input.contiguous().view(actual_torch.uint8),
                        view.contiguous().view(actual_torch.uint8),
                    )
                )
                return 0

            plan.context = types.SimpleNamespace(
                lock=threading.RLock(),
                torch=actual_torch,
                device=source.device,
                workspace=workspace,
                _check_stream=lambda: 7,
                lib=types.SimpleNamespace(cublasLtMatmul=launch),
                handle=C.c_void_p(1),
            )
            plan.desc = C.c_void_p(2)
            plan.layouts = [C.c_void_p(i) for i in (3, 4, 5)]
            plan.algo = module._Algo()
            plan.alpha = C.c_float(1)
            plan.beta = C.c_float(0)
            self.assertIs(plan(view), plan.output)
            self.assertEqual(view.storage_offset(), 1280 * k)
            self.assertEqual(view.data_ptr(), source.data_ptr() + 1280 * k * 4)
            self.assertTrue(
                actual_torch.equal(
                    source.view(actual_torch.uint8), saved.view(actual_torch.uint8)
                )
            )

    def test_unique_loaded_library_resolution_is_strict(self):
        mapping = "1-2 r-xp 0 00:00 1 /tmp/libcublasLt.so.12.8.4\n"
        self.assertEqual(
            module._find_loaded_library(mapping), Path("/tmp/libcublasLt.so.12.8.4")
        )
        for bad in (
            "",
            mapping + mapping.replace("/tmp/", "/other/"),
            mapping.rstrip() + " (deleted)\n",
        ):
            with self.assertRaises(RuntimeError):
                module._find_loaded_library(bad)


if __name__ == "__main__":
    unittest.main()
