# SPDX-License-Identifier: Apache-2.0
"""P6 codec comparison must exercise and report the selected optimization mode."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/moss_speech/p6/compare_codec.py"


def load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("p6_codec_cli_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CodecCliTests(unittest.TestCase):
    def test_parser_default_keeps_optimizations_disabled(self) -> None:
        module = load_script()
        args = module.parser().parse_args(
            [
                "--codec-path",
                "codec",
                "--state",
                "state",
                "--reference-dir",
                "ref",
                "--out-dir",
                "out",
            ]
        )
        self.assertIs(args.codec_optimizations, False)

    def test_parser_accepts_explicit_optimization(self) -> None:
        module = load_script()
        args = module.parser().parse_args(
            [
                "--codec-path",
                "codec",
                "--state",
                "state",
                "--reference-dir",
                "ref",
                "--out-dir",
                "out",
                "--codec-optimizations",
            ]
        )
        self.assertIs(args.codec_optimizations, True)

    def test_main_passes_selected_flag_to_real_constructor_site_and_report(
        self,
    ) -> None:
        module = load_script()

        class ConstructorReached(RuntimeError):
            pass

        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                reference = root / "reference"
                reference.mkdir()
                (reference / "report.json").write_text(json.dumps({"pass": True}))
                output = root / "output"
                calls = []

                def constructor(*args: object, **kwargs: object) -> None:
                    calls.append((args, kwargs))
                    raise ConstructorReached("Stop before any codec inference")

                source = dict(
                    voice_token_ids=[1], voice_feat=[[0.0]], voice_embedding=[0.0]
                )
                torch = types.SimpleNamespace(
                    set_num_threads=lambda _: None,
                    backends=types.SimpleNamespace(
                        cuda=types.SimpleNamespace(
                            matmul=types.SimpleNamespace(allow_tf32=None)
                        )
                    ),
                    load=lambda *args, **kwargs: source,
                    tensor=lambda values, **kwargs: values,
                    int32="int32",
                    cuda=types.SimpleNamespace(
                        get_device_name=lambda _: "CPU test double"
                    ),
                )
                argv = [
                    str(SCRIPT),
                    "--codec-path",
                    str(root / "codec"),
                    "--state",
                    str(root / "state.pt"),
                    "--reference-dir",
                    str(reference),
                    "--out-dir",
                    str(output),
                ]
                if enabled:
                    argv.append("--codec-optimizations")
                modules = {
                    "torch": torch,
                    "sglang_omni.models.moss_speech.components.streaming_codec": types.SimpleNamespace(
                        MossSpeechStreamingCodec=constructor
                    ),
                }
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, modules):
                    with self.assertRaises(ConstructorReached):
                        module.main()
                self.assertEqual(
                    calls,
                    [
                        (
                            (root / "codec",),
                            {"chunk_size": 5, "codec_optimizations": enabled},
                        )
                    ],
                )
                report = json.loads((output / "report.json").read_text())
                self.assertIs(report["codec_optimizations"], enabled)
                self.assertFalse(report["pass"])


if __name__ == "__main__":
    unittest.main()
