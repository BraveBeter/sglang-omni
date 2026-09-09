# SPDX-License: Apache-2.0
"""HF config port for MOSS-Speech (P2/T2.2).

Faithful port of the locked `configuration_moss_speech.py` from
`OpenMOSS-Team/MOSS-Speech` @ snapshot `cff025bb`. SGLang does not execute
``trust_remote_code``, so the config class is vendored here and registered
under its ``model_type``.

Deviations from the source (documented; no semantic change):
1. Prefer modern ``PreTrainedConfig`` RoPE normalization and validation
   methods, with the legacy free-function fallback for transformers 4.x.
   Layer-type validation remains standalone for compatibility.
2. Checkpoint fields are preserved verbatim, including
   ``num_hidden_layers=36`` (32 shared + 4 modality per the config's own
   invariant). The 40-layer (32+4+4) KV accounting is a P3 memory-pool
   concern and is NOT reflected here — the HF config is not falsified.
"""

from __future__ import annotations

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _rope_config_validation(config: PretrainedConfig) -> None:
    # New releases may retain the deprecated free function, so prefer the
    # instance API while preserving its normalization-before-validation order.
    standardize = getattr(config, "standardize_rope_params", None)
    validate = getattr(config, "validate_rope", None)
    if callable(standardize) and callable(validate):
        standardize()
        validate()
    else:
        from transformers.modeling_rope_utils import rope_config_validation

        rope_config_validation(config)


def _layer_type_validation(layer_types: list[str]) -> None:
    """transformers>=5 replaced the free function with an instance method.

    The original signature took the list; replicating its check standalone
    keeps behavior identical on transformers 4.x callers while 5.x configs
    validate themselves via ``PreTrainedConfig.__init__``.
    """
    allowed = {"full_attention", "sliding_attention"}
    if layer_types and any(lt not in allowed for lt in layer_types):
        raise ValueError(
            f"unsupported layer types: {sorted(set(layer_types) - allowed)}"
        )


class MossSpeechConfig(PretrainedConfig):
    """Configuration for ``MossSpeechForCausalLM`` (locked revision port)."""

    model_type = "moss_speech"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151680,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=None,
        layer_types=None,
        attention_dropout=0.0,
        audio_vocab_size=None,
        modality_pad_token_id=0,
        sosp_token_id=None,
        eosp_token_id=None,
        num_shared_layers=32,
        num_modality_layers=4,
        **kwargs,
    ):
        assert num_shared_layers + num_modality_layers == num_hidden_layers, (
            f"num_shared_layers ({num_shared_layers}) + num_modality_layers ({num_modality_layers}) "
            f"must equal to num_hidden_layers ({num_hidden_layers})"
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.audio_vocab_size = audio_vocab_size
        self.modality_pad_token_id = int(modality_pad_token_id)
        self.sosp_token_id = None if sosp_token_id is None else int(sosp_token_id)
        self.eosp_token_id = None if eosp_token_id is None else int(eosp_token_id)
        self.num_shared_layers = int(num_shared_layers)
        self.num_modality_layers = int(num_modality_layers)

        if num_key_value_heads is None:  # backward compatibility
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        _rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        _layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def ensure_moss_speech_config_registered() -> None:
    """Idempotent ``AutoConfig`` registration (safe at import time)."""
    from transformers import AutoConfig

    AutoConfig.register("moss_speech", MossSpeechConfig, exist_ok=True)


ensure_moss_speech_config_registered()

__all__ = ["MossSpeechConfig", "ensure_moss_speech_config_registered"]
