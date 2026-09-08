# SPDX-License-Identifier: Apache-2.0
"""T3.2 CPU tests: native model structure, 446-tensor weight mapping,
anti-tying and error paths.

Heavy sglang layer construction needs the CPU shims: this module imports
``sglang_cpu_env`` FIRST (it installs a single-process gloo group, a tolerant
ServerArgs stand-in and a vllm._custom_ops stub before sglang resolves its
module-level platform probes).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tests.unit_test.moss_speech.sglang_cpu_env  # noqa: F401  (import side effects)
from sglang_omni.models.moss_speech.sglang_model import (
    ARCH_KEY,
    MossSpeechSGLangModel,
    moss_attention_layer_count,
)

MANIFEST = Path(__file__).parent / "weight_manifest_446.txt"

# Tiny-but-structurally-faithful config: shared=2, modality=1 -> attention
# layer ids {0,1} trunk, {2} text tail, {3} audio tail.
TINY = dict(
    hidden_size=256,
    intermediate_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    max_position_embeddings=512,
    vocab_size=256,
    audio_vocab_size=64,
    num_shared_layers=2,
    num_modality_layers=1,
    num_hidden_layers=3,
    attention_bias=False,
    modality_pad_token_id=151667,
)


def make_tiny_model() -> MossSpeechSGLangModel:
    return MossSpeechSGLangModel(
        SimpleNamespace(**TINY), init_device=torch.device("cpu")
    )


def tiny_checkpoint_names():
    names = [
        "model.embed_tokens.weight",
        "model.audio_embed.weight",
        "model.text_norm.weight",
        "model.audio_norm.weight",
        "text_lm_head.weight",
        "audio_lm_head.weight",
    ]
    per_layer = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ]
    for i in range(TINY["num_shared_layers"]):
        names += [f"model.shared_block.layers.{i}.{p}.weight" for p in per_layer]
    for i in range(TINY["num_modality_layers"]):
        names += [f"model.text_block.layers.{i}.{p}.weight" for p in per_layer]
        names += [f"model.audio_block.layers.{i}.{p}.weight" for p in per_layer]
    return names


def sentinel_weights(model: MossSpeechSGLangModel, names):
    """Unique per-tensor sentinel fill so mis-routing is detectable by value.

    Shard targets (qkv/gate-up) get shard-row-shaped tensors; direct targets
    get full-shape tensors.
    """
    params = dict(model.named_parameters())
    weights = []
    for idx, name in enumerate(names):
        mapped = model._map_checkpoint_name(name)
        assert mapped is not None, name
        target, shard = mapped
        if shard is None:
            shape = tuple(params[target].shape)
        else:
            start, end = model._shard_row_span(shard)
            shape = (end - start, params[target].shape[1])
        weights.append((name, torch.full(shape, idx + 1, dtype=torch.float32)))
    return weights


def test_real_config_attention_layer_count():
    # HF semantic num_hidden_layers=36 preserved; runtime attention layers=40.
    real = dict(num_shared_layers=32, num_modality_layers=4, num_hidden_layers=36)
    assert moss_attention_layer_count(real) == 40


def test_tiny_layer_ids_cover_all_attention_slots():
    m = make_tiny_model()
    ids = m.attention_layer_ids()
    assert sorted(ids) == [0, 1, 2, 3]
    assert moss_attention_layer_count(TINY) == 4
    # tail layer ids sit above the trunk (32/36 offsets for the real shape)
    assert ids[2] > ids[1] and ids[3] > ids[2]


def test_manifest_446_all_mapped():
    names = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    assert len(names) == 446
    m = make_tiny_model()
    params = dict(m.named_parameters())
    for name in names:
        mapped = MossSpeechSGLangModel._map_checkpoint_name(name)
        assert mapped is not None, f"unmapped checkpoint key: {name}"
        target = mapped[0]
        # normalize block index modulo the tiny depth so real-shape names hit
        # the tiny module tree
        parts = target.split(".")
        if parts[0] in ("layers", "text_block", "audio_block") and parts[1].isdigit():
            depth = (
                TINY["num_shared_layers"]
                if parts[0] == "layers"
                else TINY["num_modality_layers"]
            )
            parts[1] = str(int(parts[1]) % depth)
            target = ".".join(parts)
        assert target in params, f"{name} -> {target} not a module param"


def test_load_weights_sentinel_routing():
    m = make_tiny_model()
    names = tiny_checkpoint_names()
    m.load_weights(sentinel_weights(m, names))
    params = dict(m.named_parameters())
    for idx, name in enumerate(names):
        target, shard = m._map_checkpoint_name(name)
        if shard is None:
            assert torch.all(
                params[target] == idx + 1
            ), f"{name} -> {target}: sentinel lost"
        else:
            start, end = m._shard_row_span(shard)
            rows = params[target][start:end]
            assert torch.all(
                rows == idx + 1
            ), f"{name} -> {target}[{shard}]: sentinel lost"
    # the staged decode embedding is NOT a checkpoint tensor
    assert "_decode_input_embedding.weight" not in {
        n for n, _ in sentinel_weights(m, names)
    }


def test_load_weights_missing_tensor_raises():
    m = make_tiny_model()
    names = tiny_checkpoint_names()[:-1]  # drop one
    with pytest.raises(RuntimeError, match="missing"):
        m.load_weights(sentinel_weights(m, names))


def test_load_weights_unexpected_key_raises():
    m = make_tiny_model()
    good = sentinel_weights(m, tiny_checkpoint_names())
    with pytest.raises(RuntimeError, match="unexpected"):
        m.load_weights(good + [("model.lm_head.weight", torch.zeros(1))])


def test_load_weights_broadcast_shape_is_rejected():
    m = make_tiny_model()
    weights = sentinel_weights(m, tiny_checkpoint_names())
    for i, (name, value) in enumerate(weights):
        if name.endswith("self_attn.q_proj.weight"):
            weights[i] = (name, value[:1])
            break
    with pytest.raises(RuntimeError, match="shape"):
        m.load_weights(weights)


def test_load_weights_duplicate_source_is_rejected():
    m = make_tiny_model()
    weights = sentinel_weights(m, tiny_checkpoint_names())
    with pytest.raises(RuntimeError, match="duplicate"):
        m.load_weights(weights + weights[:1])


def test_load_weights_shape_mismatch_raises():
    m = make_tiny_model()
    names = tiny_checkpoint_names()
    weights = sentinel_weights(m, names)
    name, t = weights[0]
    weights[0] = (name, torch.zeros(tuple(d + 3 for d in t.shape)))
    with pytest.raises(Exception):
        m.load_weights(weights)


def test_heads_must_not_be_tied():
    m = make_tiny_model()
    m.load_weights(sentinel_weights(m, tiny_checkpoint_names()))
    m.assert_heads_independent()  # distinct sentinels -> passes

    m2 = make_tiny_model()
    m2.load_weights(sentinel_weights(m2, tiny_checkpoint_names()))
    with torch.no_grad():
        # force sharing of VALUES (not storage) to prove the guard trips on
        # value equality, the failure mode of an accidental tie
        m2.text_lm_head.weight.copy_(m2.embed_tokens.weight)
    with pytest.raises(RuntimeError, match="tying forbidden"):
        m2.assert_heads_independent()


def test_decode_staging_buffer_exists_and_frozen():
    m = make_tiny_model()
    buf = m._decode_input_embedding
    assert buf.weight.shape == (128, TINY["hidden_size"])
    assert not buf.weight.requires_grad


def test_arch_key_matches_hf_architectures():
    import json

    hf_archs = (
        json.load(
            open(
                "/remote-home1/xrluan/SGLang_experiments/models/MOSS-Speech/config.json"
            )
        )["architectures"]
        if Path(
            "/remote-home1/xrluan/SGLang_experiments/models/MOSS-Speech/config.json"
        ).exists()
        else ["MossSpeechForCausalLM"]
    )
    assert ARCH_KEY in hf_archs
