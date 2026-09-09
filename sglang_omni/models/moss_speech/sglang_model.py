# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-Speech AR model.

Dual-channel (text/audio) decoder: shared 32-layer trunk, two independent
4-layer tails (text layer ids 32-35, audio 36-39) each with its own final
RMSNorm and LM head. One grid row == one KV position in all 40 attention
layers.

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
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, List, Optional, Tuple

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3DecoderLayer

logger = logging.getLogger(__name__)

# Token IDs from the locked checkpoint configuration.
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
    get = (
        (lambda k, d=None: config.get(k, d))
        if isinstance(config, dict)
        else (lambda k, d=None: getattr(config, k, d))
    )
    return int(get("num_shared_layers")) + 2 * int(get("num_modality_layers"))


class MossSpeechRMSNorm(RMSNorm):
    """Preserve the checkpoint's explicit BF16 rounding boundaries.

    Residual addition is rounded before normalization; the normalized value
    is rounded before multiplying the learned weight. A fused float32
    residual+norm or norm+weight operation changes these semantics.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Any:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        value = self.weight * value.to(dtype)
        return value if residual is None else (value, residual)


@lru_cache(maxsize=8)
def reference_rope_frequencies(head_dim: int, theta: float) -> torch.Tensor:
    """Reference initializes frequencies on CPU before moving them to CUDA.

    CUDA pow differs by one ULP for some frequencies, which first changes
    BF16 sin/cos at position 459 for the locked checkpoint.
    """
    exponents = (
        torch.arange(0, head_dim, 2, device="cpu", dtype=torch.int64).float() / head_dim
    )
    return 1.0 / (theta**exponents)


def request_linear(
    x: torch.Tensor, weight: torch.Tensor, lengths: list[int]
) -> torch.Tensor:
    """Keep each request's GEMM shape independent of unrelated batch members."""
    if sum(lengths) != x.shape[0]:
        raise ValueError("request lengths do not cover the packed tensor")
    if len(lengths) <= 1:
        return torch.nn.functional.linear(x, weight)
    return torch.cat(
        [torch.nn.functional.linear(part, weight) for part in x.split(lengths, dim=0)],
        dim=0,
    )


def request_norm(
    norm: Any,
    x: torch.Tensor,
    lengths: list[int],
    residual: Optional[torch.Tensor] = None,
) -> Any:
    """Evaluate reduction shapes independently for each packed request."""
    if len(lengths) <= 1:
        return norm(x, residual) if residual is not None else norm(x)
    chunks = x.split(lengths, dim=0)
    if residual is None:
        return torch.cat([norm(chunk) for chunk in chunks], dim=0)
    results = [
        norm(chunk, r) for chunk, r in zip(chunks, residual.split(lengths, dim=0))
    ]
    return tuple(torch.cat([result[i] for result in results], dim=0) for i in range(2))


class MossSpeechDecoderLayer(Qwen3DecoderLayer):
    """TP=1 decoder that does not bypass model-specific Q/K norm rounding."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = forward_batch._moss_request_lengths
        if post_residual_addition is not None:
            raise ValueError("Post residual addition is not supported in V1")
        if residual is None:
            residual = hidden_states
            hidden_states = request_norm(self.input_layernorm, hidden_states, lengths)
        else:
            hidden_states, residual = request_norm(
                self.input_layernorm, hidden_states, lengths, residual
            )
        attn = self.self_attn
        # Keep separate GEMMs and BF16 elementwise rounding as in the checkpoint.
        # Packed storage remains unchanged for SGLang's weight loader.
        q_weight, k_weight, v_weight = attn.qkv_proj.weight.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=0
        )
        q = request_linear(hidden_states, q_weight, lengths)
        k = request_linear(hidden_states, k_weight, lengths)
        v = request_linear(hidden_states, v_weight, lengths)
        q = request_norm(
            attn.q_norm, q.reshape(-1, attn.num_heads, attn.head_dim), lengths
        )
        k = request_norm(
            attn.k_norm, k.reshape(-1, attn.num_kv_heads, attn.head_dim), lengths
        )
        inv_freq = reference_rope_frequencies(attn.head_dim, attn.rope_theta).to(
            positions.device
        )
        with torch.autocast(device_type=positions.device.type, enabled=False):
            freqs = (
                inv_freq[None, :, None] @ positions[None, None, :].float()
            ).transpose(1, 2)
            angles = torch.cat((freqs, freqs), dim=-1)
            cos = angles.cos().to(q.dtype)[0, :, None, :]
            sin = angles.sin().to(q.dtype)[0, :, None, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            left, right = x.chunk(2, dim=-1)
            return torch.cat((-right, left), dim=-1)

        q = (q * cos + rotate(q) * sin).reshape(-1, attn.q_size)
        k = (k * cos + rotate(k) * sin).reshape(-1, attn.kv_size)
        hidden_states = attn.attn(q, k, v, forward_batch)
        hidden_states = request_linear(hidden_states, attn.o_proj.weight, lengths)
        hidden_states, residual = request_norm(
            self.post_attention_layernorm, hidden_states, lengths, residual
        )
        gate_weight, up_weight = self.mlp.gate_up_proj.weight.chunk(2, dim=0)
        gate = request_linear(hidden_states, gate_weight, lengths)
        up = request_linear(hidden_states, up_weight, lengths)
        hidden_states = request_linear(
            torch.nn.functional.silu(gate) * up, self.mlp.down_proj.weight, lengths
        )
        return hidden_states, residual


def make_moss_decoder_layer(config: Any, **kwargs: Any) -> MossSpeechDecoderLayer:
    """Reuse native projections/attention/KV, retaining reference norm math."""
    layer = MossSpeechDecoderLayer(config, **kwargs)
    for name in ("input_layernorm", "post_attention_layernorm"):
        norm = MossSpeechRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        setattr(layer, name, norm)
        setattr(layer.layer_communicator, name, norm)
    for name in ("q_norm", "k_norm"):
        setattr(
            layer.self_attn,
            name,
            MossSpeechRMSNorm(layer.self_attn.head_dim, eps=config.rms_norm_eps),
        )
    return layer


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
        _dt = str(getattr(config, "dtype", "bfloat16") or "bfloat16")
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(_dt, torch.bfloat16)
        self.hidden_size = int(config.hidden_size)
        self.text_vocab_size = int(config.vocab_size)
        self.audio_vocab_size = int(config.audio_vocab_size)
        self.num_shared_layers = int(config.num_shared_layers)
        self.num_modality_layers = int(config.num_modality_layers)
        self.total_attention_layers = moss_attention_layer_count(config)
        self.head_dim = int(
            getattr(config, "head_dim", 0)
            or (int(config.hidden_size) // int(config.num_attention_heads))
        )

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
            make_moss_decoder_layer(
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
            make_moss_decoder_layer(
                qcfg,
                layer_id=text_start + i,
                start_layer=0,
                quant_config=quant_config,
                prefix=f"text_block.{i}",
            )
            for i in range(self.num_modality_layers)
        )
        self.audio_block = torch.nn.ModuleList(
            make_moss_decoder_layer(
                qcfg,
                layer_id=audio_start + i,
                start_layer=0,
                quant_config=quant_config,
                prefix=f"audio_block.{i}",
            )
            for i in range(self.num_modality_layers)
        )

        self.text_norm = MossSpeechRMSNorm(self.hidden_size, eps=qcfg.rms_norm_eps)
        self.audio_norm = MossSpeechRMSNorm(self.hidden_size, eps=qcfg.rms_norm_eps)
        self.text_lm_head = ParallelLMHead(
            self.text_vocab_size, self.hidden_size, quant_config=quant_config
        )
        self.audio_lm_head = ParallelLMHead(
            self.audio_vocab_size, self.hidden_size, quant_config=quant_config
        )
        # Per-channel LogitsProcessor (moss_tts precedent, PR #608): sizes
        # each channel's logits to its own vocab (strips ParallelLMHead vocab
        # padding) and honors the head's sampler-contract forward path.
        self.text_logits_processor = self._make_logits_processor(
            config, self.text_vocab_size
        )
        self.audio_logits_processor = self._make_logits_processor(
            config, self.audio_vocab_size
        )

        # Staged decode-feedback embedding rows (moss_tts pattern): the model
        # runner writes per-request feedback embeddings into rows [0..bs) and
        # rewrites forward_batch.input_ids to row indices before decode.
        buffer_bs = int(getattr(config, "moss_decode_buffer_bs", 128))
        self._decode_input_embedding = torch.nn.Embedding(
            buffer_bs,
            self.hidden_size,
            device=init_device or torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self._decode_input_embedding.weight.requires_grad_(False)
        # per-request dual logits from the most recent forward (rid-keyed;
        # the runner pops entries when consuming)
        self._dual_logits_by_rid: dict = {}
        # Attention layers must cover exactly the allocator's 40 slots.
        ids = [
            layer.self_attn.attn.layer_id
            for layer in list(self.layers)
            + list(self.text_block)
            + list(self.audio_block)
        ]
        assert sorted(ids) == list(
            range(self.total_attention_layers)
        ), f"attention layer ids {sorted(ids)} != 0..{self.total_attention_layers - 1}"

    @staticmethod
    def _make_logits_processor(config: Any, vocab_size: int) -> LogitsProcessor:
        from copy import copy

        channel_config = copy(config)
        channel_config.vocab_size = int(vocab_size)
        return LogitsProcessor(channel_config)

    # ------------------------------------------------------------------ utils
    def attention_layer_ids(self) -> List[int]:
        ids = [layer.self_attn.attn.layer_id for layer in self.layers]
        ids += [layer.self_attn.attn.layer_id for layer in self.text_block]
        ids += [layer.self_attn.attn.layer_id for layer in self.audio_block]
        return ids

    def _last_token_indices(
        self, hidden: torch.Tensor, forward_batch: Any
    ) -> torch.Tensor:
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
            forward_mode = (
                getattr(forward_batch, "forward_mode", None)
                if forward_batch is not None
                else None
            )
            is_decode = (
                forward_mode is not None
                and getattr(forward_mode, "is_decode", None) is not None
                and bool(forward_mode.is_decode())
            )
            if is_decode:
                staging = self._decode_input_embedding
                if staging.weight.device != input_ids.device:
                    # not a checkpoint tensor: the loader never migrates it
                    staging = staging.to(input_ids.device)
                    self._decode_input_embedding = staging
                input_embeds = staging(input_ids)
            else:
                # 1-D text-token fallback path (tests / smoke only); the
                # canonical dual-channel prefill always passes input_embeds.
                input_embeds = self.embed_tokens(input_ids)

        mode = getattr(forward_batch, "forward_mode", None)
        if mode is not None and mode.is_decode():
            lengths = [1] * input_embeds.shape[0]
        else:
            seq_lengths = getattr(forward_batch, "extend_seq_lens", None)
            lengths = (
                seq_lengths.tolist()
                if seq_lengths is not None
                else [input_embeds.shape[0]]
            )
        forward_batch._moss_request_lengths = lengths
        self._last_request_lengths = lengths
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        # sglang's fused RMSNorm updates (hidden, residual) IN PLACE inside
        # the decoder layers; both tails fork from the trunk output, so the
        # second tail must start from a private copy (the reference runs the
        # tails over the same shared output with independent streams).
        audio_hidden = hidden_states.clone()
        audio_residual = residual.clone() if residual is not None else None
        text_hidden, text_residual = hidden_states, residual
        for layer in self.text_block:
            text_hidden, text_residual = layer(
                positions=positions,
                hidden_states=text_hidden,
                forward_batch=forward_batch,
                residual=text_residual,
            )

        for layer in self.audio_block:
            audio_hidden, audio_residual = layer(
                positions=positions,
                hidden_states=audio_hidden,
                forward_batch=forward_batch,
                residual=audio_residual,
            )

        last_idx = self._last_token_indices(text_hidden, forward_batch)
        text_final, _ = request_norm(
            self.text_norm, text_hidden, lengths, text_residual
        )
        audio_final, _ = request_norm(
            self.audio_norm, audio_hidden, lengths, audio_residual
        )
        self._last_head_indices = last_idx
        # Final hidden states are consumed by the heads in this forward.
        self._last_text_hidden = text_final.detach()
        self._last_audio_hidden = audio_final.detach()
        # Preserve the standard engine hidden-state result as well.
        dual_hidden = torch.stack(
            [self._last_text_hidden[last_idx], self._last_audio_hidden[last_idx]], dim=0
        )
        out = self.compute_dual_logits()
        # rid-keyed transport: forward_batch is rebuilt between the forward
        # and the post hooks, and extra forwards (sampling/logprob recompute)
        # clobber any single-slot stash. Keying by the batch's rids binds
        # each request's dual logits to exactly this forward.
        rids = getattr(forward_batch, "rids", None)
        if rids is not None:
            _fm = getattr(forward_batch, "forward_mode", None)
            _is_dec = bool(
                _fm is not None
                and getattr(_fm, "is_decode", None) is not None
                and _fm.is_decode()
            )
            _mode_key = "decode" if _is_dec else "prefill"
            for _i, _rid in enumerate(rids):
                # clone: the engine's sampler warps next_token_logits
                # (same storage as text_logits) IN PLACE inside
                # forward_batch_generation; slices would see the warped
                # distribution before the model runner consumes it.
                self._dual_logits_by_rid[(_rid, _mode_key)] = (
                    out.text_logits[_i].clone(),
                    out.audio_logits[_i].clone(),
                )
        out.hidden_states = dual_hidden
        import os as _os2

        _d2 = _os2.environ.get("MOSS_DUMP_DIR")
        if _d2:
            import pathlib as _pl2

            _pl2.Path(_d2).mkdir(parents=True, exist_ok=True)
            _n2 = sum(1 for _ in _pl2.Path(_d2).glob("hidden_*.pt"))
            torch.save(
                {
                    "text_final": self._last_text_hidden.float().cpu(),
                    "audio_final": self._last_audio_hidden.float().cpu(),
                    "trunk_h": hidden_states.detach().float().cpu(),
                    "trunk_r": (
                        residual.detach().float().cpu()
                        if residual is not None
                        else None
                    ),
                    "audio_in_h": audio_hidden.detach().float().cpu(),
                    "audio_in_r": (
                        audio_residual.detach().float().cpu()
                        if audio_residual is not None
                        else None
                    ),
                    "last_idx": last_idx.detach().cpu(),
                    "rids": (list(rids) if isinstance(rids, (list, tuple)) else None),
                    "out_text_head5": out.text_logits[0][:5].detach().float().cpu(),
                    "recompute_head5": torch.nn.functional.linear(
                        self._last_text_hidden, self.text_lm_head.weight
                    )[0][:5]
                    .detach()
                    .float()
                    .cpu(),
                    "hidden_head5": self._last_text_hidden[0][:5]
                    .detach()
                    .float()
                    .cpu(),
                },
                _pl2.Path(_d2) / f"hidden_{_n2:03d}.pt",
            )
        try:
            forward_batch._moss_dual_logits = (out.text_logits, out.audio_logits)
            return out
        except AttributeError:
            # object.__new__-style test batches may not accept attributes
            return self.compute_dual_logits()

    def compute_dual_logits(self) -> MossSpeechModelOutput:
        """Dual-head logits from the stashed final hiddens of the last
        forward. Must be called synchronously after forward (single
        engine thread).

        Plain matmul against the head weights (TP=1, unquantized): the
        per-channel LogitsProcessor path mangled the distribution (argmax
        flipped to unrelated tokens at cos -0.33 while a direct
        ``hidden @ W.T`` reproduced the reference at cos 0.999 — see the
        T3.4 first-step dumps). The engine sampler is never used for this
        model (the runner injects next_token_ids), so the sampler-contract
        forwarding of ParallelLMHead is not needed here.
        """
        text_logits = request_linear(
            self._last_text_hidden, self.text_lm_head.weight, self._last_request_lengths
        )
        audio_logits = request_linear(
            self._last_audio_hidden,
            self.audio_lm_head.weight,
            self._last_request_lengths,
        )
        # Reference projects the complete prefill before selecting its last
        # row. Selecting hidden first changes the BF16 GEMM shape and ties.
        indices = self._last_head_indices
        text_logits = text_logits[indices]
        audio_logits = audio_logits[indices]
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
                rest = name[len(ckpt_prefix) :]
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
        head_dim = (
            self.head_dim
            if getattr(self, "head_dim", None)
            else int(getattr(self.config, "head_dim", 0))
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
        source_names: set[str] = set()
        for name, tensor in weights:
            if name in source_names:
                raise RuntimeError(f"duplicate checkpoint source {name}")
            source_names.add(name)
            mapped = self._map_checkpoint_name(name)
            if mapped is None:
                raise RuntimeError(f"unexpected checkpoint key {name}")
            target, shard = mapped
            if target not in params:
                raise KeyError(
                    f"mapped module {target} not found for checkpoint key {name}"
                )
            data = tensor.data if hasattr(tensor, "data") else tensor
            param = params[target]
            expected_shape = param.shape
            if shard is not None:
                start, end = self._shard_row_span(shard)
                expected_shape = param[start:end].shape
            if data.shape != expected_shape:
                raise RuntimeError(
                    f"checkpoint shape mismatch for {name}: {tuple(data.shape)} != {tuple(expected_shape)}"
                )
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
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]
        per_layer_fused = {
            "self_attn.qkv_proj.weight": {"q", "k", "v"},
            "mlp.gate_up_proj.weight": {"gate", "up"},
        }
        for i in range(self.num_shared_layers):
            prefix = f"layers.{i}."
            for d in per_layer_direct:
                if prefix + d not in direct_seen:
                    raise RuntimeError(
                        f"moss_speech load_weights: missing {prefix + d}"
                    )
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
                        raise RuntimeError(
                            f"moss_speech load_weights: missing {prefix + d}"
                        )
                for fused, shards in per_layer_fused.items():
                    got = shards_seen.get(prefix + fused, set())
                    if got != shards:
                        raise RuntimeError(
                            f"moss_speech load_weights: {prefix + fused} shards {sorted(got)} != {sorted(shards)}"
                        )
        for d in (
            "embed_tokens.weight",
            "audio_embed.weight",
            "text_norm.weight",
            "audio_norm.weight",
            "text_lm_head.weight",
            "audio_lm_head.weight",
        ):
            if d not in direct_seen:
                raise RuntimeError(f"moss_speech load_weights: missing {d}")
        expected_sources = (
            6 + (self.num_shared_layers + 2 * self.num_modality_layers) * 11
        )
        logger.info(
            "moss_speech load_weights: %d/%d source tensors consumed "
            "(4 independent large matrices; qkv/gate-up fused per sglang layout)",
            consumed_sources,
            expected_sources,
        )
        if consumed_sources != expected_sources:
            raise RuntimeError(
                f"moss_speech load_weights: consumed {consumed_sources} sources, "
                f"expected {expected_sources} (dedup or missing checkpoint tensors)"
            )

        self._weight_load_report = {
            "consumed_sources": consumed_sources,
            "unique_sources": len(source_names),
            "expected_sources": expected_sources,
            "unexpected_sources": 0,
            "exact_shapes_checked": True,
        }

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
                    raise RuntimeError(
                        f"{label} embedding and lm_head share storage/values (tying forbidden)"
                    )


EntryClass = MossSpeechSGLangModel  # symmetry with package-level discovery

ARCH_KEY = "MossSpeechForCausalLM"
