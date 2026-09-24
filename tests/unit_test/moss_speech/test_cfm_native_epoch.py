"""Per-flow live epoch owner: no cached tuple, no global replacement."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni/models/moss_speech/components/cfm_native_epoch.py"
)


def module():
    spec = importlib.util.spec_from_file_location("test_private_native_epoch", SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Flow:
    def __init__(self):
        self.value = 1


class Provider:
    def __init__(self):
        self.calls = 0
        self.fallback = False
        self.wrong = False
        self.closed = False

    def epoch(self, flow, original, *, stats):
        self.calls += 1
        if self.fallback:
            stats.update(route="original", reason="unsupported registry")
            return original(flow)
        stats.update(
            route="native",
            modules=1,
            parameters=1,
            buffers=0,
            noises=0,
            entries=2,
            gil_held=True,
        )
        return ((flow.value + int(self.wrong),), ("training", False))

    def close(self):
        self.closed = True

    identity = {"manifest_sha256": "a" * 64, "library_sha256": "b" * 64}


class OwnerTests(unittest.TestCase):
    def setup_owner(self):
        self.api = module()
        self.flow = Flow()
        self.provider = Provider()
        self.original_calls = []

        def original(flow):
            self.original_calls.append(flow.value)
            return ((flow.value,), ("training", False))

        self.original = original
        self.owner = self.api.NativeEpochOwner(
            self.flow, "manifest.json", original, loader=lambda path: self.provider
        )
        return self.owner

    def test_constructor_does_not_load_and_prepare_controls_do_not_count_hits(self):
        api = module()
        with patch.object(
            api, "load_package", side_effect=AssertionError("early load")
        ):
            owner = api.NativeEpochOwner(Flow(), "manifest.json", lambda f: ())
            self.assertFalse(owner.status()["enabled"])
            owner.close()
        owner = self.setup_owner()
        owner.prepare()
        self.assertEqual(self.original_calls, [1, 1])
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(owner.status()["native_calls"], 0)
        self.assertTrue(owner.status()["enabled"])

    def test_every_snapshot_observes_live_change_and_preserves_oracle(self):
        owner = self.setup_owner()
        owner.prepare()
        self.original_calls.clear()
        first = owner.snapshot(self.flow, self.original)
        self.flow.value = 9
        second = owner.snapshot(self.flow, self.original)
        self.assertNotEqual(first, second)
        self.assertEqual(self.original_calls, [])
        self.assertEqual(owner.status()["native_calls"], 2)
        self.assertFalse(owner.status()["caches_epoch_or_registration_lists"])

    def test_changed_current_oracle_and_foreign_flow_use_current_original(self):
        owner = self.setup_owner()
        owner.prepare()

        def current(flow):
            return ("current", flow.value)

        self.assertEqual(owner.snapshot(self.flow, current), ("current", 1))
        other = Flow()
        other.value = 8
        self.assertEqual(
            owner.snapshot(other, self.original), ((8,), ("training", False))
        )
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(owner.status()["fallback_calls"], 2)

    def test_unsupported_uses_exact_original_and_exceptions_are_not_swallowed(self):
        owner = self.setup_owner()
        owner.prepare()
        self.provider.fallback = True
        self.assertEqual(
            owner.snapshot(self.flow, self.original), self.original(self.flow)
        )

        def bad(flow):
            raise KeyError("original error")

        with self.assertRaisesRegex(KeyError, "original error"):
            owner.snapshot(self.flow, bad)
        self.assertEqual(owner.status()["fallback_calls"], 1)
        self.assertEqual(owner.status()["failed_calls"], 1)

    def test_raw_prepare_mismatch_does_not_activate_and_releases_provider(self):
        owner = self.setup_owner()
        self.provider.wrong = True
        with self.assertRaisesRegex(ValueError, "tuple"):
            owner.prepare()
        self.assertFalse(owner.status()["enabled"])
        self.assertTrue(self.provider.closed)
        owner.close()
        self.assertTrue(owner.status()["flow_released"])

    def test_prepare_fallback_is_not_native_qualification(self):
        owner = self.setup_owner()
        self.provider.fallback = True
        with self.assertRaises(ValueError):
            owner.prepare()
        self.assertFalse(owner.status()["enabled"])
        self.assertTrue(self.provider.closed)

    def test_prepare_double_and_close_double_release_once(self):
        owner = self.setup_owner()
        owner.prepare()
        with self.assertRaises(RuntimeError):
            owner.prepare()
        owner.close()
        state = owner.status()
        owner.close()
        self.assertEqual(state, owner.status())
        self.assertTrue(state["closed"])
        self.assertTrue(state["flow_released"])
        self.assertTrue(state["provider_released"])
        with self.assertRaises(RuntimeError):
            owner.snapshot(self.flow, self.original)

    def test_pid_crossing_is_rejected_without_native_call(self):
        owner = self.setup_owner()
        owner.prepare()
        with (
            patch.object(os, "getpid", return_value=owner.pid + 1),
            self.assertRaises(RuntimeError),
        ):
            owner.snapshot(self.flow, self.original)
        self.assertEqual(self.provider.calls, 1)

    def test_original_mutates_during_prepare_is_rejected(self):
        owner = self.setup_owner()

        def original(flow):
            result = ((flow.value,), ("training", False))
            flow.value += 1
            return result

        owner.original = original
        with self.assertRaises(ValueError):
            owner.prepare()
        self.assertFalse(owner.status()["enabled"])


if __name__ == "__main__":
    unittest.main()
