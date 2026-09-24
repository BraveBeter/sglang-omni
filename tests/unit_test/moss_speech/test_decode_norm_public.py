"""Public startup-only norm package admission and spawned worker configuration."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech import stages
from sglang_omni.models.moss_speech.config import MossSpeechStreamingPipelineConfig
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder


class PublicNormTests(unittest.TestCase):
    def test_default_none_and_explicit_false_override(self):
        value = MossSpeechStreamingPipelineConfig(model_path="/model")
        self.assertIsNone(value.decode_norm_pointwise)
        stage = next(s for s in value.stages if s.name == "ar_engine")
        self.assertNotIn("decode_norm_pointwise", stage.factory_args)
        stage.factory_args.update(
            decode_norm_pointwise=True, decode_norm_kernel_manifest="/old/manifest.json"
        )
        inherited = MossSpeechStreamingPipelineConfig(**value.model_dump())
        self.assertIs(
            next(s for s in inherited.stages if s.name == "ar_engine").factory_args[
                "decode_norm_pointwise"
            ],
            True,
        )
        merged = ConfigManager(inherited).merge_config({"decode_norm_pointwise": False})
        self.assertIs(
            next(s for s in merged.stages if s.name == "ar_engine").factory_args[
                "decode_norm_pointwise"
            ],
            False,
        )

    def test_enabled_requires_complete_graph_and_manifest(self):
        for options in (
            dict(decode_norm_pointwise=True),
            dict(decode_norm_pointwise=True, decode_full_graphs=True),
            dict(decode_norm_pointwise=1),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MossSpeechEngineBuilder(**options)

    def test_off_does_not_read_manifest_or_load_binary(self):
        with (
            patch.object(
                Path, "read_text", side_effect=AssertionError("off read manifest")
            ),
            patch("ctypes.CDLL", side_effect=AssertionError("off dlopen")),
        ):
            builder = MossSpeechEngineBuilder(
                decode_norm_pointwise=False,
                decode_norm_kernel_manifest="/missing/package.json",
            )
        self.assertFalse(builder.decode_norm_pointwise)

    def test_factory_forwards_public_worker_options(self):
        with (
            patch.object(stages, "validate_ar_preconditions"),
            patch(
                "sglang_omni.models.moss_speech.engine_builder.MossSpeechEngineBuilder"
            ) as builder,
        ):
            stages.create_ar_engine_executor(
                "/model",
                decode_full_graphs=True,
                decode_norm_pointwise=True,
                decode_norm_kernel_manifest="/qualified/manifest.json",
            )
        self.assertIs(builder.call_args.kwargs["decode_norm_pointwise"], True)
        self.assertEqual(
            builder.call_args.kwargs["decode_norm_kernel_manifest"],
            "/qualified/manifest.json",
        )


class PackageTests(unittest.TestCase):
    def api(self):
        return importlib.import_module(
            "sglang_omni.models.moss_speech.decode_norm_kernel"
        )

    def fixture(self, directory):
        api = self.api()
        files = {}
        for name in api.QUALIFIED_FILES:
            path = directory / name
            path.write_bytes(name.encode())
            files[name] = api.sha(path)
        manifest = deepcopy(api.MANIFEST_FIELDS)
        manifest["files"] = files
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return api, path, files

    def test_relative_package_and_environment_are_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            api, path, files = self.fixture(Path(folder))
            with patch.object(api, "QUALIFIED_FILES", files):
                row = api.validate_manifest(path)
                self.assertEqual(row["manifest_sha256"], api.sha(path))
                api.validate_environment(
                    row["manifest"],
                    dict(torch="2.9.1+cu128", cuda="12.8", compute_capability=[8, 0]),
                )
                with self.assertRaises(ValueError):
                    api.validate_environment(
                        row["manifest"],
                        dict(
                            torch="2.9.1+cu128", cuda="12.6", compute_capability=[8, 0]
                        ),
                    )

    def test_changed_binary_or_unqualified_flags_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            api, path, files = self.fixture(Path(folder))
            with patch.object(api, "QUALIFIED_FILES", files):
                value = json.loads(path.read_text())
                value["compiler"]["flags"].append("--use_fast_math")
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    api.validate_manifest(path)
                value = deepcopy(api.MANIFEST_FIELDS)
                value["files"] = files
                path.write_text(json.dumps(value))
                (Path(folder) / "native.so").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    api.validate_manifest(path)

    def test_absolute_escape_missing_and_unknown_manifest_fields_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            api, path, files = self.fixture(Path(folder))
            with patch.object(api, "QUALIFIED_FILES", files):
                for change in (
                    lambda v: v["files"].update({"../native.so": files["native.so"]}),
                    lambda v: v.update(unknown=True),
                    lambda v: v.pop("qualification"),
                    lambda v: v.update(abi_version=True),
                    lambda v: v["environment"].update(compute_capability=[8, False]),
                ):
                    value = deepcopy(api.MANIFEST_FIELDS)
                    value["files"] = files.copy()
                    change(value)
                    path.write_text(json.dumps(value))
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        api.validate_manifest(path)


class WorkerStartupTests(unittest.TestCase):
    def run_startup(self, *, fail=False, incomplete=False):
        from sglang_omni.models.moss_speech import (
            decode_norm_kernel,
            decode_norm_pointwise,
        )

        builder = MossSpeechEngineBuilder(decode_full_graphs=True)
        builder.decode_norm_pointwise = True
        builder.decode_norm_kernel_manifest = "explicit.json"
        events = []
        model = SimpleNamespace()
        status = dict(
            closed=False,
            busy=False,
            norm_calls=1458,
            bodies={"128": 3, "512": 3, "1024": 3},
            fallback_calls=0,
            partial_projection_calls=0,
            failed_bodies=0,
            norm_parameter_flags_preserved=True,
        )
        if incomplete:
            status["norm_calls"] = 1457
        owner = SimpleNamespace(status=lambda: status)

        def initialize():
            self.assertIs(model._moss_decode_norm_owner, owner)
            events.append("initialize")
            if fail:
                raise RuntimeError("capture error")

        def close():
            events.append("close")
            del model._moss_decode_norm_owner

        manager = SimpleNamespace(
            initialize=initialize, close=close, cache=SimpleNamespace(size=3)
        )
        model._decode_graph_manager = manager
        engine = SimpleNamespace(identity=dict(manifest="bound"))
        with (
            patch.object(decode_norm_kernel, "Engine", return_value=engine) as load,
            patch.object(decode_norm_pointwise, "DecodeNormOwner", return_value=owner),
        ):
            if fail or incomplete:
                with self.assertRaises((RuntimeError, ValueError)):
                    builder.setup_model_resources(
                        model, None, generation_cuda_graph_enabled=False
                    )
                self.assertEqual(events, ["initialize", "close"])
                self.assertFalse(hasattr(model, "_moss_decode_norm_owner"))
            else:
                builder.setup_model_resources(
                    model, None, generation_cuda_graph_enabled=False
                )
                self.assertEqual(events, ["initialize"])
            self.assertEqual(load.call_args.args[1], "explicit.json")

    def test_worker_loads_before_capture(self):
        self.run_startup()

    def test_capture_exception_closes_original_manager(self):
        self.run_startup(fail=True)

    def test_incomplete_body_rejects_and_closes(self):
        self.run_startup(incomplete=True)


if __name__ == "__main__":
    unittest.main()
