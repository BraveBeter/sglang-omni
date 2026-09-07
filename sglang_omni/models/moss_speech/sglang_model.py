# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-Speech AR model.

Dual-channel (text/audio) decoder: shared 32-layer trunk, two independent
4-layer tails (text layer ids 32-35, audio 36-39) each with its own final
RMSNorm and LM head. One grid row == one KV position in all 40 attention
layers (see docs/design/moss_speech/p3/01_native_design.md §2-3).

Weight mapping (checkpoint -> module), 446 source tensors, all consumed:

  model.embed_tokens.weight            -> embed_tokens.weight
  model.audio_embed.weight             -> audio_embed.weight
  model.shared_block.layers.{0..31}.*  -> layers.{0..31}.*
  model.text_block.layers.{0..3}.*     -> text_block.{0..3}.*   (layer_id 32+i)
  model.audio_block.layers.{0..3}.*    -> audio_block.{0..3}.*  (layer_id 36+i)
  model.text_norm.weight               -> text_norm.weight
  model.audio_norm.weight              -> audio_norm.weight
  text_lm_head.weight                  -> text_lm_head.weight
  audio_lm_head.weight                 -> audio_lm_head.weight

The four large matrices (two embeddings, two heads) carry INDEPENDENT values in
the checkpoint (maxdiff 0.56/0.22, verified T3.1); tying is forbidden.
`num_hidden_layers` stays 36 (checkpoint semantics); the runtime attention
layer count is num_shared_layers + 2 * num_modality_layers = 40 (see
ModelWorker._apply_arch_override branch + tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3DecoderLayer

logger = logging.getLogger(__name__)

# Token ids frozen by MODEL_CONTRACT / P3-01 §2.
MODALITY_PAD_ID = 151667


@dataclass
class MossSpeechModelOutput(LogitsProcessorOutput):
    """Extends the engine-facing output with the dual-channel heads."""

    text_logits: Optional[torch.Tensor] = None
    audio_logits: Optional[torch.Tensor] = None


def as_qwen3_layer_config(config: Any) -> Any:
    """Build a Qwen3-shaped config for the reusable decoder layers.

    ``num_hidden_layers`` is set to the RUNTIME ATTENTION LAYER COUNT (40) so
    that LayerScatterModes / layer bookkeeping inside the reused Qwen3 layers
    stay consistent with the 40-slot KV allocator. The HF semantic field
    (num_hidden_layers=36) on the original config object is untouched.
    """
    from transformers import Qwen3Config

    total = moss_attention_layer_count(config)
    if isinstance(config, dict):
        src = config
    else:
        src = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    cfg = Qwen3Config(
        hidden_size=int(src["hidden_size"]),
        intermediate_size=int(src["intermediate_size"]),
        num_attention_heads=int(src["num_attention_heads"]),
        num_key_value_heads=int(src["num_key_value_heads"]),
        head_dim=int(src.get("head_dim") or 0) or None,
        rms_norm_eps=float(src["rms_norm_eps"]),
        rope_theta=float(src.get("rope_theta", 1_000_000.0)),
        max_position_embeddings=int(src.get("max_position_embeddings", 40960)),
        vocab_size=int(src["vocab_size"]),
        num_hidden_layers=total,
        attention_bias=bool(src.get("attention_bias", False)),
        tie_word_embeddings=False,
    )
    return cfg


def moss_attention_layer_count(config: Any) -> int:
    """num_shared + 2 * num_modality (both tails hold KV every position)."""
    get = (lambda k, d=None: config.get(k, d)) if isinstance(config, dict) else (
        lambda k, d=None: getattr(config, k, d)
    )
    return int(get("num_shared_layers")) + 2 * int(get("num_modality_layers"))


class MossSpeechSGLangModel(torch.nn.Module):
    """MOSS-Speech AR backbone (architecture key: MossSpeechForCausalLM).

    TP=1 only in V1. The scheduler sees a 1-D selected-token stream; the
    dual-channel prompt rows are folded by the model runner
    (custom_prefill_forward) which passes ``input_embeds`` directly. During
    decode the runner stages each request's feedback embedding into
    ``_decode_input_embedding`` rows and rewrites ``input_ids`` to row ids
    (moss_tts precedent).
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        init_device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.text_vocab_size = int(config.vocab_size)
        self.audio_vocab_size = int(config.audio_vocab_size)
        self.num_shared_layers = int(config.num_shared_layers)
        self.num_modality_layers = int(config.num_modality_layers)
        self.total_attention_layers = moss_attention_layer_count(config)
        self.head_dim = int(getattr(config, "head_dim", 0) or
                            (int(config.hidden_size) // int(config.num_attention_heads)))

        qcfg = as_qwen3_layer_config(config)

        self.embed_tokens = VocabParallelEmbedding(
            self.text_vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix="embed_tokens",
        )
        self.audio_embed = VocabParallelEmbedding(
            self.audio_vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix="audio_embed",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3DecoderLayer(
                qcfg,
                layer_id=i,
                start_layer=0,
                quant_config=quant_config,
                prefix=f"layers.{i}",
            )
            for i in range(self.num_shared_layers)
        )
        text_start = self.num_shared_layers
        audio_start = text_start + self.num_modality_layers
        self.text_block = torch.nn.ModuleList(
            Qwen3DecoderLayer(
                qcfg,
                layer_id=text_start + i,
                start_layer=0,
                quant_config=quant_config,
                prefix=f"text_block.{i}",
            )
            for i in range(self.num_modality_layers)
        )
        self.audio_block = torch.nn.ModuleList(
            Qwen3DecoderLayer(
                qcfg,
                layer_id=audio_start + i,
                start_layer=0,
                quant_config=quant_config,
                prefix=f"audio_block.{i}",
            )
            for i in range(self.num_modality_layers)
        )

        self.text_norm = RMSNorm(self.hidden_size, eps=qcfg.rms_norm_eps)
        self.audio_norm = RMSNorm(self.hidden_size, eps=qcfg.rms_norm_eps)
        self.text_lm_head = ParallelLMHead(
            self.text_vocab_size, self.hidden_size, quant_config=quant_config
        )
        self.audio_lm_head = ParallelLMHead(
            self.audio_vocab_size, self.hidden_size, quant_config=quant_config
        )

        # Staged decode-feedback embedding rows (moss_tts pattern): the model
        # runner writes per-request feedback embeddings into rows [0..bs) and
        # rewrites forward_batch.input_ids to row indices before decode.
        buffer_bs = int(getattr(config, "moss_decode_buffer_bs", 128))
        self._decode_input_embedding = torch.nn.Embedding(
            buffer_bs, self.hidden_size,
            device=init_device or torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self._decode_input_embedding.weight.requires_grad_(False)
        # Attention layers must cover exactly the allocator's 40 slots.
        ids = [l.self_attn.attn.layer_id for l in list(self.layers) + list(self.text_block) + list(self.audio_block)]
        assert sorted(ids) == list(range(self.total_attention_layers)), (
            f"attention layer ids {sorted(ids)} != 0..{self.total_attention_layers - 1}"
        )

    # ------------------------------------------------------------------ utils
    def attention_layer_ids(self) -> List[int]:
        ids = [l.self_attn.attn.layer_id for l in self.layers]
        ids += [l.self_attn.attn.layer_id for l in self.text_block]
        ids += [l.self_attn.attn.layer_id for l in self.audio_block]
        return ids

    def _last_token_indices(self, hidden: torch.Tensor, forward_batch: Any) -> torch.Tensor:
        """Indices of each request's last position in the packed token dim."""
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or getattr(mode, "is_decode", None) and mode.is_decode():
            return torch.arange(hidden.shape[0], device=hidden.device)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None or int(extend_seq_lens.numel()) == 0:
            return torch.tensor([hidden.shape[0] - 1], device=hidden.device)
        cum = torch.cumsum(extend_seq_lens.flatten(), dim=0)
        return (cum - 1).to(device=hidden.device, dtype=torch.long)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Any = None,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MossSpeechModelOutput:
        del kwargs
        if input_embeds is None:
            if input_ids is None:
                raise ValueError("require input_ids or input_embeds")
            forward_mode = getattr(forward_batch, "forward_mode", None) if forward_batch is not None else None
            is_decode = (
                forward_mode is not None
                and getattr(forward_mode, "is_decode", None) is not None
                and bool(forward_mode.is_decode())
            )
            if is_decode:
                input_embeds = self._decode_input_embedding(input_ids)
            else:
                # 1-D text-token fallback path (tests / smoke only); the
                # canonical dual-channel prefill always passes input_embeds.
                input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        text_hidden, text_residual = hidden_states, residual
        for layer in self.text_block:
            text_hidden, text_residual = layer(
                positions=positions,
                hidden_states=text_hidden,
                forward_batch=forward_batch,
                residual=text_residual,
            )

        audio_hidden, audio_residual = hidden_states, residual
        for layer in self.audio_block:
            audio_hidden, audio_residual = layer(
                positions=positions,
                hidden_states=audio_hidden,
                forward_batch=forward_batch,
                residual=audio_residual,
            )

        last_idx = self._last_token_indices(text_hidden, forward_batch)
        text_final, _ = self.text_norm(text_hidden[last_idx], text_residual[last_idx] if text_residual is not None else None)
        audio_final, _ = self.audio_norm(audio_hidden[last_idx], audio_residual[last_idx] if audio_residual is not None else None)
        text_logits = self.text_lm_head(text_final)
        audio_logits = self.audio_lm_head(audio_final)
        return MossSpeechModelOutput(
            next_token_logits=text_logits,
            text_logits=text_logits,
            audio_logits=audio_logits,
        )

    # ------------------------------------------------------------ weight load
    _CKPT_RENAMES = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.audio_embed.weight": "audio_embed.weight",
        "model.text_norm.weight": "text_norm.weight",
        "model.audio_norm.weight": "audio_norm.weight",
        "text_lm_head.weight": "text_lm_head.weight",
        "audio_lm_head.weight": "audio_lm_head.weight",
    }

    @staticmethod
    def _map_checkpoint_name(name: str) -> Optional[Tuple[str, Optional[str]]]:
        """checkpoint key -> (native param name, shard id or None).

        Shard ids exist because the reused Qwen3 layers keep sglang's fused
        representations: ``qkv_proj`` consumes q/k/v (rows: q | k | v) and
        ``gate_up_proj`` consumes gate/up (rows: gate | up). TP=1 only (V1),
        so a plain row-slice copy is exact.
        """
        renames = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.audio_embed.weight": "audio_embed.weight",
            "model.text_norm.weight": "text_norm.weight",
            "model.audio_norm.weight": "audio_norm.weight",
            "text_lm_head.weight": "text_lm_head.weight",
            "audio_lm_head.weight": "audio_lm_head.weight",
        }
        if name in renames:
            return renames[name], None
        for ckpt_prefix, native_prefix in (
            ("model.shared_block.layers.", "layers."),
            ("model.text_block.layers.", "text_block."),
            ("model.audio_block.layers.", "audio_block."),
        ):
            if name.startswith(ckpt_prefix):
                rest = name[len(ckpt_prefix):]
                break
        else:
            return None  # unexpected checkpoint key
        fused = {
            "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", "q"),
            "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", "k"),
            "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", "v"),
            "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", "gate"),
            "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", "up"),
        }
        layer_idx, sep, leaf = rest.partition(".")
        assert sep, f"malformed checkpoint layer key {name}"
        if leaf in fused:
            target, shard = fused[leaf]
            return f"{native_prefix}{layer_idx}.{target}", shard
        return f"{native_prefix}{rest}", None

    def _shard_row_span(self, shard: str) -> Tuple[int, int]:
        """Row range of ``shard`` inside its fused matrix (TP=1)."""
        head_dim = self.head_dim if getattr(self, "head_dim", None) else int(
            getattr(self.config, "head_dim", 0)
        )
        num_heads = int(self.config.num_attention_heads)
        num_kv = int(self.config.num_key_value_heads)
        inter = int(self.config.intermediate_size)
        q_rows, kv_rows = num_heads * head_dim, num_kv * head_dim
        spans = {
            "q": (0, q_rows),
            "k": (q_rows, q_rows + kv_rows),
            "v": (q_rows + kv_rows, q_rows + 2 * kv_rows),
            "gate": (0, inter),
            "up": (inter, 2 * inter),
        }
        return spans[shard]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        consumed_sources = 0
        shards_seen: dict[str, set] = {}
        direct_seen: set[str] = set()
        for name, tensor in weights:
            mapped = self._map_checkpoint_name(name)
            if mapped is None:
                logger.warning("moss_speech load_weights: unexpected checkpoint key %s", name)
                continue
            target, shard = mapped
            if target not in params:
                raise KeyError(f"mapped module {target} not found for checkpoint key {name}")
            data = tensor.data if hasattr(tensor, "data") else tensor
            param = params[target]
            if shard is None:
                default_weight_loader(param, data)
                direct_seen.add(target)
            else:
                start, end = self._shard_row_span(shard)
                with torch.no_grad():
                    param[start:end].copy_(data.to(param.dtype))
                shards_seen.setdefault(target, set()).add(shard)
            consumed_sources += 1

        # coverage: every source tensor of the 446-checkpoint layout consumed
        per_layer_direct = [
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
        ]
        per_layer_fused = {
            "self_attn.qkv_proj.weight": {"q", "k", "v"},
            "mlp.gate_up_proj.weight": {"gate", "up"},
        }
        for i in range(self.num_shared_layers):
            prefix = f"layers.{i}."
            for d in per_layer_direct:
                if prefix + d not in direct_seen:
                    raise RuntimeError(f"moss_speech load_weights: missing {prefix + d}")
            for fused, shards in per_layer_fused.items():
                got = shards_seen.get(prefix + fused, set())
                if got != shards:
                    raise RuntimeError(
                        f"moss_speech load_weights: {prefix + fused} shards {sorted(got)} != {sorted(shards)}"
                    )
        for i in range(self.num_modality_layers):
            for blk in ("text_block", "audio_block"):
                prefix = f"{blk}.{i}."
                for d in per_layer_direct:
                    if prefix + d not in direct_seen:
                        raise RuntimeError(f"moss_speech load_weights: missing {prefix + d}")
                for fused, shards in per_layer_fused.items():
                    got = shards_seen.get(prefix + fused, set())
                    if got != shards:
                        raise RuntimeError(
                            f"moss_speech load_weights: {prefix + fused} shards {sorted(got)} != {sorted(shards)}"
                        )
        for d in ("embed_tokens.weight", "audio_embed.weight", "text_norm.weight",
                  "audio_norm.weight", "text_lm_head.weight", "audio_lm_head.weight"):
            if d not in direct_seen:
                raise RuntimeError(f"moss_speech load_weights: missing {d}")
        expected_sources = (
            6
            + (self.num_shared_layers + 2 * self.num_modality_layers) * 11
        )
        logger.info(
            "moss_speech load_weights: %d/%d source tensors consumed "
            "(4 independent large matrices; qkv/gate-up fused per sglang layout)",
            consumed_sources, expected_sources,
        )
        if consumed_sources != expected_sources:
            raise RuntimeError(
                f"moss_speech load_weights: consumed {consumed_sources} sources, "
                f"expected {expected_sources} (dedup or missing checkpoint tensors)"
            )

    # -------------------------------------------------- anti-tying assertion
    def assert_heads_independent(self) -> None:
        """The checkpoint's embed/head matrices are independent values; tying
        would silently corrupt both channels (T3.1 verified maxdiff 0.56/0.22)."""
        with torch.no_grad():
            for emb, head, label in (
                (self.embed_tokens, self.text_lm_head, "text"),
                (self.audio_embed, self.audio_lm_head, "audio"),
            ):
                ew = emb.weight
                hw = head.weight if hasattr(head, "weight") else head.logits_processor
                if ew.shape == hw.shape and torch.equal(ew, hw):
                    raise RuntimeError(f"{label} embedding and lm_head share storage/values (tying forbidden)")

EntryClass = MossSpeechSGLangModel  # symmetry with package-level discovery

ARCH_KEY = "MossSpeechForCausalLM"
