# SPDX-License-Identifier: Apache-2.0
"""Registry assertions: MossSpeechForCausalLM resolves to the native class in
a fresh interpreter (subprocess), and the 40-layer arch override applies."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

CODE = r"""
from types import SimpleNamespace

import tests.unit_test.moss_speech.sglang_cpu_env  # noqa: F401
from sglang.srt.models.registry import ModelRegistry
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner

# Runtime order mirrors the engine: _register_omni_model() runs inside
# SGLModelRunner.__init__ BEFORE the model loader resolves the architecture.
# The method body never touches instance state, so a bare namespace suffices.
SGLModelRunner._register_omni_model(SimpleNamespace())
cls = ModelRegistry.models.get("MossSpeechForCausalLM")
assert cls is not None, "MossSpeechForCausalLM not in ModelRegistry"
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
assert cls is MossSpeechSGLangModel, f"registry resolved {cls}"
print("REGISTRY_OK")
"""


def test_registry_resolves_in_fresh_process(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", CODE],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "TMPDIR": "/dev/shm",
            "PYTHONPATH": str(repo_root),
        },
    )
    assert "REGISTRY_OK" in proc.stdout, f"stderr tail: {proc.stderr[-800:]}"
    assert proc.returncode == 0


def test_arch_override_sets_40_attention_layers():
    from sglang_omni.model_runner.model_worker import ModelWorker

    mc = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["MossSpeechForCausalLM"],
            num_shared_layers=32,
            num_modality_layers=4,
            num_hidden_layers=36,
        ),
        num_hidden_layers=36,
    )
    ModelWorker._apply_arch_override(mc, "MossSpeechForCausalLM")
    assert mc.num_attention_layers == 40
    assert mc.num_hidden_layers == 36  # HF semantics preserved
    assert mc.hf_text_config is mc.hf_config
