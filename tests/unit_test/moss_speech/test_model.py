# SPDX-License-Identifier: Apache-2.0
"""Native weight-loading, numerical, and dual-tail attention contracts."""

import os
from pathlib import Path

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from transformers import Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_speech import model_layers, sglang_model
from sglang_omni.models.moss_speech.hf_config import (
    MossSpeechConfig,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_layers import (
    MossSpeechRMSNorm,
    reference_rope_frequencies,
    request_linear,
)
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fixtures.moss_speech_model import (
    checkpoint_weights,
    decode,
    native_runtime,
    prefill,
    qualify_batched_rope,
    tiny_model,
)

NATIVE_CPU = pytest.mark.skipif(
    os.environ.get("SGLANG_USE_CPU_ENGINE") != "1",
    reason="Native CPU construction requires SGLANG_USE_CPU_ENGINE=1 before imports",
)


@NATIVE_CPU
def test_actual_worker_allocates_both_modality_tails_without_rewriting_hf_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = MossSpeechConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=256,
        audio_vocab_size=64,
        max_position_embeddings=16,
        architectures=["MossSpeechForCausalLM"],
    )
    configuration.save_pretrained(tmp_path)
    monkeypatch.setitem(
        CONFIG_MAPPING._extra_content, "moss_speech", MossSpeechConfig
    )  # noqa: leading-underscore  # Transformers registry spelling.
    ensure_moss_speech_config_registered()
    worker_configuration = ModelConfig(
        str(tmp_path), trust_remote_code=False, dtype="bfloat16"
    )
    ModelWorker.apply_arch_override(worker_configuration, "MossSpeechForCausalLM")
    assert worker_configuration.num_attention_layers == 40
    assert worker_configuration.num_hidden_layers == 36
    assert worker_configuration.hf_config.num_hidden_layers == 36
    with native_runtime(torch.device("cpu")):
        model = MossSpeechSGLangModel(configuration)
        weights = checkpoint_weights(configuration)
        assert len(weights) == 446
        model.load_weights(weights)
        assert model.attention_layer_ids() == list(range(40))


@NATIVE_CPU
def test_checkpoint_sentinels_reach_every_independent_parameter() -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        weights = dict(checkpoint_weights(model.config))
        model.load_weights(weights.items())
        for parameter, source in (
            (model.embed_tokens.weight, "model.embed_tokens.weight"),
            (model.audio_embed.weight, "model.audio_embed.weight"),
            (model.text_norm.weight, "model.text_norm.weight"),
            (model.audio_norm.weight, "model.audio_norm.weight"),
            (model.text_lm_head.weight, "text_lm_head.weight"),
            (model.audio_lm_head.weight, "audio_lm_head.weight"),
        ):
            torch.testing.assert_close(parameter, weights[source], rtol=0, atol=0)
        for layers, source_block in (
            (model.layers, "shared_block"),
            (model.text_block, "text_block"),
            (model.audio_block, "audio_block"),
        ):
            for layer_index, layer in enumerate(layers):
                prefix = f"model.{source_block}.layers.{layer_index}."
                expected_qkv = torch.cat(
                    [
                        weights[prefix + f"self_attn.{projection}_proj.weight"]
                        for projection in ("q", "k", "v")
                    ]
                )
                expected_gate_up = torch.cat(
                    [
                        weights[prefix + f"mlp.{projection}_proj.weight"]
                        for projection in ("gate", "up")
                    ]
                )
                torch.testing.assert_close(
                    layer.self_attn.qkv_proj.weight, expected_qkv, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    layer.mlp.gate_up_proj.weight, expected_gate_up, rtol=0, atol=0
                )
                for parameter, leaf in (
                    (layer.self_attn.o_proj.weight, "self_attn.o_proj.weight"),
                    (layer.self_attn.q_norm.weight, "self_attn.q_norm.weight"),
                    (layer.self_attn.k_norm.weight, "self_attn.k_norm.weight"),
                    (layer.mlp.down_proj.weight, "mlp.down_proj.weight"),
                    (layer.input_layernorm.weight, "input_layernorm.weight"),
                    (
                        layer.post_attention_layernorm.weight,
                        "post_attention_layernorm.weight",
                    ),
                ):
                    torch.testing.assert_close(
                        parameter, weights[prefix + leaf], rtol=0, atol=0
                    )


@NATIVE_CPU
@pytest.mark.parametrize(
    "defect,message",
    [
        ("missing", "cover"),
        ("unexpected", "Unexpected"),
        ("duplicate", "Duplicate"),
        ("broadcast", "shape"),
        ("shape", "shape"),
    ],
)
def test_incomplete_duplicate_and_malformed_checkpoints_fail(
    defect: str, message: str
) -> None:
    with native_runtime(torch.device("cpu")):
        model = tiny_model(torch.device("cpu"))
        weights = checkpoint_weights(model.config)
        if defect == "missing":
            weights.pop()
        elif defect == "unexpected":
            weights.append(("model.lm_head.weight", torch.zeros(1)))
        elif defect == "duplicate":
            weights.append(weights[0])
        elif defect == "broadcast":
            source_name, tensor = weights[6]
            weights[6] = (source_name, tensor[:1])
        else:
            source_name, tensor = weights[0]
            weights[0] = (
                source_name,
                torch.zeros(tuple(length + 3 for length in tensor.shape)),
            )
        with pytest.raises(RuntimeError, match=message):
            model.load_weights(weights)


@NATIVE_CPU
@pytest.mark.parametrize("channel", ["text", "audio"])
def test_equal_embedding_and_head_weights_are_rejected(channel: str) -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        model.load_weights(checkpoint_weights(model.config))
        model.assert_heads_independent()
        embedding, head = (
            (model.embed_tokens, model.text_lm_head)
            if channel == "text"
            else (model.audio_embed, model.audio_lm_head)
        )
        head.weight.copy_(embedding.weight)
        with pytest.raises(RuntimeError, match="independent"):
            model.assert_heads_independent()


def test_bf16_normalization_rounds_residual_and_normalized_value() -> None:
    """Combining FP32 residual addition or weight multiplication changes checkpoint rounding."""
    generator = torch.Generator().manual_seed(921)
    hidden_states = torch.randn(3, 128, generator=generator).bfloat16()
    residual = torch.randn(3, 128, generator=generator).bfloat16()
    original_hidden = hidden_states.clone()
    with torch.random.fork_rng(devices=[]):
        normalization = MossSpeechRMSNorm(128, eps=1e-6).bfloat16()
    with torch.no_grad():
        normalization.weight.copy_(torch.randn(128, generator=generator).bfloat16())
    rounded_residual = hidden_states + residual
    reference = Qwen3RMSNorm(128, eps=1e-6).bfloat16()
    with torch.no_grad():
        reference.weight.copy_(normalization.weight)
    expected = reference(rounded_residual)
    actual, actual_residual = normalization(hidden_states, residual)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_residual, rounded_residual)
    assert torch.equal(hidden_states, original_hidden)


def test_projection_preserves_each_request_gemm_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(78)
    hidden_states = torch.randn(7, 128, generator=generator).bfloat16()
    weight = torch.randn(32, 128, generator=generator).bfloat16()
    linear = torch.nn.functional.linear
    expected = torch.cat(
        [linear(hidden_states[:2], weight), linear(hidden_states[2:], weight)]
    )
    token_shapes: list[tuple[int, ...]] = []

    def record_linear(
        request_hidden: torch.Tensor, request_weight: torch.Tensor
    ) -> torch.Tensor:
        token_shapes.append(tuple(request_hidden.shape))
        return linear(request_hidden, request_weight)

    monkeypatch.setattr(torch.nn.functional, "linear", record_linear)
    actual = request_linear(hidden_states, weight, [2, 5])
    assert token_shapes == [(2, 128), (5, 128)]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("head_dim,theta", [(8, 10000.0), (128, 1_000_000.0)])
def test_rope_frequencies_keep_cpu_float32_reference_values(
    head_dim: int, theta: float
) -> None:
    frequencies = reference_rope_frequencies(head_dim, theta)
    assert frequencies.device.type == "cpu"
    assert frequencies.dtype == torch.float32
    with torch.device("cpu"):
        reference = Qwen3RotaryEmbedding(
            Qwen3Config(head_dim=head_dim, rope_theta=theta)
        )
    torch.testing.assert_close(frequencies, reference.inv_freq, rtol=0, atol=0)


def assert_native_request_isolation(device: torch.device) -> None:
    with native_runtime(device), torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(183)
        model = tiny_model(device)
        prompt_embeddings = [
            torch.randn((length, 128), generator=generator, device=device)
            for length in (2, 3)
        ]
        feedback_embeddings = torch.randn((2, 128), generator=generator, device=device)
        packed_prefill, packed_backend = prefill(
            model, prompt_embeddings, ["first", "second"]
        )
        packed_decode = decode(
            model, packed_backend, [2, 3], feedback_embeddings, ["first", "second"]
        )
        for request_index, prompt in enumerate(prompt_embeddings):
            isolated_prefill, isolated_backend = prefill(model, [prompt], ["isolated"])
            isolated_decode = decode(
                model,
                isolated_backend,
                [len(prompt)],
                feedback_embeddings[request_index : request_index + 1],
                ["isolated"],
            )
            for packed, isolated in (
                (packed_prefill.text_logits, isolated_prefill.text_logits),
                (packed_prefill.audio_logits, isolated_prefill.audio_logits),
                (packed_decode.text_logits, isolated_decode.text_logits),
                (packed_decode.audio_logits, isolated_decode.audio_logits),
            ):
                assert packed is not None and isolated is not None
                torch.testing.assert_close(
                    packed[request_index], isolated[0], rtol=0, atol=0
                )
        first_tail = model.config.num_shared_layers
        second_tail = first_tail + model.config.num_modality_layers
        text_keys = packed_backend.token_to_kv_pool.get_key_buffer(first_tail)
        audio_keys = packed_backend.token_to_kv_pool.get_key_buffer(second_tail)
        assert text_keys.data_ptr() != audio_keys.data_ptr()
        assert torch.count_nonzero(text_keys[1:8]) > 0
        assert torch.count_nonzero(audio_keys[1:8]) > 0
        assert not torch.equal(text_keys[1:8], audio_keys[1:8])


@NATIVE_CPU
def test_native_cpu_prefill_decode_keep_requests_and_tail_kv_independent() -> None:
    assert_native_request_isolation(torch.device("cpu"))


@pytest.mark.accelerator
def test_native_cuda_prefill_decode_keep_requests_and_tail_kv_independent() -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        assert_native_request_isolation(torch.device("cuda", 0))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@NATIVE_CPU
def test_decode_rope_reuse_keeps_full_outputs_and_request_kv(batch_size: int) -> None:
    assert_decode_rope_reuse(torch.device("cpu"), batch_size)


@pytest.mark.accelerator
@pytest.mark.parametrize("batch_size", [2, 4])
def test_cuda_decode_rope_reuse_keeps_full_outputs_and_request_kv(
    batch_size: int,
) -> None:
    require_cuda()
    assert_decode_rope_reuse(torch.device("cuda", 0), batch_size)


def assert_decode_rope_reuse(device: torch.device, batch_size: int) -> None:
    """Reusing position terms must not change logits, retained outputs or either KV tail."""
    calls = 0
    original_terms = model_layers.reference_rope_terms

    def counted_terms(
        positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original_terms(positions, head_dim, theta, dtype)

    with (
        native_runtime(device),
        torch.no_grad(),
        pytest.MonkeyPatch.context() as patcher,
    ):
        patcher.setattr(model_layers, "reference_rope_terms", counted_terms)
        patcher.setattr(sglang_model, "reference_rope_terms", counted_terms)
        eager = tiny_model(device, batch_size=batch_size).eval()
        candidate = tiny_model(device, batch_size=batch_size).eval()
        assert candidate.decode_rope_reuse is False
        candidate.decode_rope_reuse = True
        generator = torch.Generator(device=device).manual_seed(713)
        lengths = list(range(1, batch_size + 1))
        prompts = [
            torch.randn(length, 128, device=device, generator=generator)
            for length in lengths
        ]
        request_ids = [f"request-{i}" for i in range(batch_size)]
        eager_prefill, eager_backend = prefill(eager, prompts, request_ids)
        candidate_prefill, candidate_backend = prefill(candidate, prompts, request_ids)
        assert candidate.decode_rope_reuse_forwards == 0
        for name in ("text_logits", "audio_logits", "hidden_states"):
            torch.testing.assert_close(
                getattr(candidate_prefill, name),
                getattr(eager_prefill, name),
                rtol=0,
                atol=0,
            )
        retained = []
        for step in range(2):
            feedback = torch.randn(batch_size, 128, device=device, generator=generator)
            before = calls
            eager_output = decode(eager, eager_backend, lengths, feedback, request_ids)
            assert calls - before == eager.total_attention_layers
            before = calls
            candidate_output = decode(
                candidate, candidate_backend, lengths, feedback, request_ids
            )
            assert calls - before == (
                1 if batch_size >= 2 else candidate.total_attention_layers
            )
            for name in ("text_logits", "audio_logits", "hidden_states"):
                actual = getattr(candidate_output, name)
                torch.testing.assert_close(
                    actual, getattr(eager_output, name), rtol=0, atol=0
                )
                retained.append((actual, actual.clone()))
            for layer_index in candidate.attention_layer_ids():
                for request_index, length in enumerate(lengths, 1):
                    slots = candidate_backend.req_to_token_pool.req_to_token[
                        request_index, : length + 1
                    ].long()
                    for cache_name in ("get_key_buffer", "get_value_buffer"):
                        actual = getattr(
                            candidate_backend.token_to_kv_pool, cache_name
                        )(layer_index)[slots]
                        expected = getattr(eager_backend.token_to_kv_pool, cache_name)(
                            layer_index
                        )[slots]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert candidate.decode_rope_reuse_forwards == (
                step + 1 if batch_size >= 2 else 0
            )
            lengths = [length + 1 for length in lengths]
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@NATIVE_CPU
@pytest.mark.parametrize(
    "incompatible", ["theta", "training", "grad", "hook", "autocast"]
)
def test_decode_rope_reuse_falls_back_for_incompatible_execution(
    incompatible: str,
) -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu")).eval()
        model.decode_rope_reuse = True
        prompts = [torch.ones(1, 128), torch.ones(2, 128)]
        _, backend = prefill(model, prompts, ["one", "two"])
        if incompatible == "theta":
            model.text_block[0].self_attn.rope_theta *= 2
        elif incompatible == "training":
            model.train()
        elif incompatible == "hook":
            model.layers[0].register_forward_pre_hook(lambda module, arguments: None)
        else:
            pass
        with (
            torch.set_grad_enabled(incompatible == "grad"),
            torch.autocast(
                "cpu", dtype=torch.bfloat16, enabled=incompatible == "autocast"
            ),
        ):
            decode(model, backend, [1, 2], torch.ones(2, 128), ["one", "two"])
        assert model.decode_rope_reuse_forwards == 0


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(torch.device("cuda", 0), id="cuda", marks=pytest.mark.accelerator),
    ],
)
def test_decode_rope_terms_preserve_full_context_reference_rounding(
    batch_size: int,
    device: torch.device,
) -> None:
    if device.type == "cuda":
        require_cuda()
    else:
        pass
    positions = torch.tensor([0, 40959, 512, 40958][:batch_size], device=device)
    reference = Qwen3RotaryEmbedding(Qwen3Config(head_dim=128, rope_theta=10000.0)).to(
        device
    )
    hidden = torch.zeros(1, batch_size, 256, dtype=torch.bfloat16, device=device)
    cosine, sine = model_layers.reference_rope_terms(
        positions, 128, 10000.0, hidden.dtype
    )
    expected_cosine, expected_sine = reference(hidden, positions[None, :])
    torch.testing.assert_close(cosine[:, 0], expected_cosine[0], rtol=0, atol=0)
    torch.testing.assert_close(sine[:, 0], expected_sine[0], rtol=0, atol=0)


@NATIVE_CPU
def test_batched_rope_qualification_returns_complete_heads_and_restores_state() -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"), batch_size=4).eval()
        feedback = model.decode_input_embedding.weight.clone()
        grids = {
            "short": torch.tensor([[2, 0], [3, 0], [4, 0]]),
            "long": torch.tensor([[5, 0], [6, 0], [7, 0], [8, 0]]),
        }
        proof, outputs = qualify_batched_rope(model, grids)
        assert proof["batch_sizes"] == [2, 4]
        assert all(count > 0 for count in proof["executions"].values())
        assert proof["complete_heads_exact"] and proof["all_kv_exact"]
        assert proof["kv_layers"] == model.total_attention_layers
        for batch in ("2", "4"):
            assert set(outputs[batch]) == set(grids)
            for values in outputs[batch].values():
                assert values["text"].shape == (model.text_vocab_size,)
                assert values["audio"].shape == (model.audio_vocab_size,)
                assert values["text"].dtype == values["audio"].dtype == torch.bfloat16
        assert model.decode_rope_reuse is False and model.training is False
        assert not model.dual_logits_by_request_id
        torch.testing.assert_close(
            model.decode_input_embedding.weight, feedback, rtol=0, atol=0
        )
