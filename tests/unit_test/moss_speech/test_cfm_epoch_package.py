"""Explicit portable package identity, no implicit compile or module import."""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

COMPONENTS = (
    Path(__file__).resolve().parents[3] / "sglang_omni/models/moss_speech/components"
)


def module():
    spec = importlib.util.spec_from_file_location(
        "test_native_epoch_package", COMPONENTS / "cfm_epoch_package.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.api = module()

    def test_strict_maps_unique_versions_and_deleted(self):
        line = "0-1 r-xp 0 00:00 0 /tmp/libstdc++.so.6.0.30"
        self.assertEqual(
            self.api.loaded_library(line, "libstdc++.so.6"),
            Path("/tmp/libstdc++.so.6.0.30"),
        )
        for maps in (
            "",
            line + " (deleted)",
            line + "\n" + line.replace("/tmp/", "/other/"),
        ):
            with self.subTest(maps=maps), self.assertRaises(ValueError):
                self.api.loaded_library(maps, "libstdc++.so.6")

    def test_strict_abi_integer_bool_no_coercion(self):
        expected = {
            "implementation": "cpython",
            "python_version": [3, 10, 12],
            "soabi": "x",
            "machine": "x86_64",
            "torch": "2.9.1+cu128",
            "cxx11_abi": True,
        }
        self.api.validate_abi(expected, dict(expected))
        for key, value in [
            ("python_version", [3, 11, 0]),
            ("cxx11_abi", 1),
            ("torch", "2.9.0"),
        ]:
            actual = dict(expected)
            actual[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_abi(expected, actual)

    def fixture(self, folder):
        root = Path(folder)
        (root / "epoch.cpp").write_text("source")
        (root / "native.so").write_bytes(b"library")
        body = {
            "files": {n: self.api.sha(root / n) for n in ["epoch.cpp", "native.so"]}
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(body))
        return path

    def test_manifest_relocation_same_bytes_and_modified_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "a"
            source.mkdir()
            manifest = self.fixture(source)
            with patch.object(
                self.api, "QUALIFIED_MANIFEST_SHA256", self.api.sha(manifest)
            ):
                first = self.api.validate_manifest(manifest)
                target = Path(temp) / "b"
                shutil.copytree(source, target)
                second = self.api.validate_manifest(target / "manifest.json")
                self.assertEqual(first["manifest"], second["manifest"])
                self.assertNotEqual(first["library"], second["library"])
                (target / "native.so").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    self.api.validate_manifest(target / "manifest.json")

    def test_package_symlink_escape_missing_extra_or_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "pkg"
            root.mkdir()
            manifest = self.fixture(root)
            saved = manifest.read_bytes()
            with patch.object(
                self.api, "QUALIFIED_MANIFEST_SHA256", self.api.sha(manifest)
            ):
                (root / "native.so").rename(Path(temp) / "outside.so")
                (root / "native.so").symlink_to(Path(temp) / "outside.so")
                with self.assertRaises(ValueError):
                    self.api.validate_manifest(manifest)
                (root / "native.so").unlink()
                with self.assertRaises(ValueError):
                    self.api.validate_manifest(manifest)
                manifest.write_bytes(saved + b" ")
                with self.assertRaises(ValueError):
                    self.api.validate_manifest(manifest)

    def test_preload_rejected_before_package_read_or_torch_import(self):
        with (
            patch.dict(os.environ, LD_PRELOAD="foreign.so"),
            patch.object(
                self.api, "validate_manifest", side_effect=AssertionError("read")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "LD_PRELOAD"):
                self.api.NativeEpochPackage("missing")


if __name__ == "__main__":
    unittest.main()
