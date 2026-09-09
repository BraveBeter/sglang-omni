"""Minimal Whisper-VQ encoder for MossSpeech codec.

This file provides only the components used by
`MossSpeechCodec/modeling_moss_speech_codec.py` during inference:
 - vector quantization helper
 - causal conv for streaming
 - SDPA attention for encoder
 - WhisperVQEncoderLayer and WhisperVQEncoder
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from .utils import WhisperVQConfig


@dataclass
class QuantizedBaseModelOutput(BaseModelOutput):
    quantized_token_ids: Optional[torch.LongTensor] = None


@dataclass
class QuantizedBaseModelOutputWithCache(QuantizedBaseModelOutput):
    past_key_value: Optional[EncoderDecoderCache] = None
    conv1_cache: Optional[torch.Tensor] = None
    conv2_cache: Optional[torch.Tensor] = None


def vector_quantize(inputs: torch.Tensor, codebook: torch.Tensor):
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook**2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)
    distances = torch.addmm(
        codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0
    )
    _, indices_flatten = torch.min(distances, dim=1)
    codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
    return codes_flatten.view_as(inputs), indices_flatten, distances


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        causal_padding = (self.kernel_size[0] - 1) * self.dilation[0]
        x = nn.functional.pad(x, (causal_padding, 0))
        return super().forward(x)

    def forward_causal(
        self, inp: torch.Tensor, conv_cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k, d = self.kernel_size[0], self.dilation[0]
        if conv_cache is None:
            inp_pad = nn.functional.pad(inp, (k - 1, 0))
        else:
            inp_pad = torch.cat((conv_cache, inp), dim=-1)
        out = super().forward(inp_pad)
        new_cache = inp_pad[:, :, -(k - 1) * d :]
        return out, new_cache


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask,
    sequence_length,
    target_length,
    cache_position=None,
    dtype=torch.float32,
    device=None,
    min_dtype=None,
    batch_size=None,
):
    if batch_size is None:
        batch_size = attention_mask.shape[0] if attention_mask is not None else 1
    if device is None:
        device = attention_mask.device if attention_mask is not None else None
    if min_dtype is None:
        min_dtype = torch.finfo(dtype).min
    if cache_position is None:
        target_length = sequence_length
        sequence_length = target_length
        if attention_mask is not None:
            mask_length = attention_mask.shape[-1]
            target_length = mask_length
    causal_mask = attention_mask
    if causal_mask is None:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    else:
        causal_mask = (
            causal_mask[:, None, None, :]
            .expand(batch_size, 1, sequence_length, target_length)
            .to(dtype)
        )
        causal_mask = (1.0 - causal_mask) * min_dtype
    if attention_mask is not None:
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)
    return causal_mask


class WhisperAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[WhisperVQConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )


class WhisperSdpaAttention(WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ):
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = (
            query_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        is_cross_attention = key_value_states is not None
        current_states = key_value_states if is_cross_attention else hidden_states
        key_states = (
            self.k_proj(current_states)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            self.v_proj(current_states)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        causal_mask = attention_mask
        sign = False
        if self.is_causal and causal_mask is None and tgt_len > 1:
            if cache_position is not None:
                dtype, device = query_states.dtype, query_states.device
                min_dtype = torch.finfo(dtype).min
                causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                    None,
                    query_states.shape[-2],
                    key_states.shape[-2],
                    cache_position=cache_position,
                    dtype=dtype,
                    device=device,
                    min_dtype=min_dtype,
                    batch_size=query_states.shape[0],
                )
            else:
                sign = True

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=sign,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, None, None


WHISPER_ATTENTION_CLASSES = {
    "sdpa": WhisperSdpaAttention,
}


class WhisperVQEncoderLayer(nn.Module):
    def __init__(self, config: WhisperVQConfig, is_causal=True, layer_idx=None):
        super().__init__()
        self.embed_dim = config.d_model
        self.kv_cache = True
        impl = getattr(config, "_attn_implementation", "sdpa")
        if impl not in WHISPER_ATTENTION_CLASSES:
            impl = "sdpa"
        self.self_attn = WHISPER_ATTENTION_CLASSES[impl](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_causal=is_causal,
            layer_idx=layer_idx,
            config=config,
        )
        self.is_causal = is_causal
        if self.is_causal:
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward_causal(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        past_key_value: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask if not self.is_causal else None,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
            use_cache=self.kv_cache,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if self.kv_cache:
            outputs += (present_key_value,)
        return outputs, cache_position


class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperVQConfig
    base_model_prefix = "model"
    main_input_name = "input_features"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperVQEncoder):
            with torch.no_grad():
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError("channels must be even for sinusoidal positional embeddings")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class WhisperVQEncoder(WhisperPreTrainedModel):
    def __init__(self, config: WhisperVQConfig):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        if config.encoder_causal_convolution:
            conv_class = CausalConv1d
        else:
            conv_class = nn.Conv1d
        self.conv1 = conv_class(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = conv_class(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        if config.quantize_encoder_only:
            self.layers = nn.ModuleList(
                [
                    WhisperVQEncoderLayer(
                        config,
                        is_causal=config.encoder_causal_attention
                        or config.quantize_causal_encoder,
                        layer_idx=i,
                    )
                    for i in range(config.quantize_position)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    WhisperVQEncoderLayer(
                        config,
                        is_causal=config.encoder_causal_attention
                        or (
                            config.quantize_causal_encoder
                            and layer_id < config.quantize_position
                        ),
                        layer_idx=layer_id,
                    )
                    for layer_id in range(config.encoder_layers)
                ]
            )
            self.layer_norm = nn.LayerNorm(config.d_model)

        self.pooling_layer = None
        if config.pooling_kernel_size is not None:
            self.pooling_layer = (
                nn.AvgPool1d(kernel_size=config.pooling_kernel_size)
                if config.pooling_type == "avg"
                else nn.MaxPool1d(kernel_size=config.pooling_kernel_size)
            )

        self.codebook = None
        self.embed_positions2 = None
        if config.quantize_vocab_size is not None:
            self.codebook = nn.Embedding(config.quantize_vocab_size, config.d_model)
            pos2_len = self.max_source_positions // max(
                int(config.pooling_kernel_size or 1), 1
            )
            self.embed_positions2 = nn.Embedding(pos2_len, config.d_model)
            self.embed_positions2.requires_grad_(False)

        self.post_init()

    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        quantized_token_ids: Optional[torch.LongTensor] = None,
        conv1_cache: Optional[torch.Tensor] = None,
        conv2_cache: Optional[torch.Tensor] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        device = input_features.device
        if input_features.dim() != 3:
            raise ValueError(
                "`input_features` should be (batch, feature_size, seq_len)"
            )

        if input_features.shape[-1] % 2 == 1:
            input_features = nn.functional.pad(input_features, (0, 1))
        if input_features.shape[1] != self.num_mel_bins:
            raise ValueError(
                f"Expected {self.num_mel_bins} mel bins, got {input_features.shape[1]}"
            )

        if isinstance(self.conv1, CausalConv1d):
            conv1_output, new_conv1_cache = self.conv1.forward_causal(
                input_features, conv1_cache
            )
        else:
            conv1_output = self.conv1(input_features)
            new_conv1_cache = None
        x = nn.functional.gelu(conv1_output)
        if isinstance(self.conv2, CausalConv1d):
            conv2_output, new_conv2_cache = self.conv2.forward_causal(x, conv2_cache)
        else:
            conv2_output = self.conv2(x)
            new_conv2_cache = None
        x = nn.functional.gelu(conv2_output)
        x = x.permute(0, 2, 1)
        batch_size, seq_len, _ = x.shape
        if attention_mask is not None:
            attention_mask = attention_mask[
                :, :: self.conv1.stride[0] * self.conv2.stride[0]
            ]
        if cache_position is None:
            cache_position = torch.arange(0, seq_len, device=device)
        embed_pos = self.embed_positions.weight
        hidden_states = x + embed_pos[cache_position]
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if past_key_values is None:
            # transformers 5.x removed EncoderDecoderCache.from_legacy_cache;
            # the encoder path only uses the cache object when one is passed in
            # (streaming conv-cache decode), so a bare None needs no conversion.
            _from_legacy = getattr(EncoderDecoderCache, "from_legacy_cache", None)
            if _from_legacy is not None:
                past_key_values = _from_legacy(past_key_values)
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs, _ = layer.forward_causal(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
                past_key_value=past_key_values if past_key_values is not None else None,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if (
                idx + 1 == self.config.pooling_position
                and self.pooling_layer is not None
            ):
                hs = hidden_states.permute(0, 2, 1)
                if hs.shape[-1] % self.config.pooling_kernel_size != 0:
                    hs = nn.functional.pad(
                        hs,
                        (
                            0,
                            self.config.pooling_kernel_size
                            - hs.shape[-1] % self.config.pooling_kernel_size,
                        ),
                    )
                hidden_states = self.pooling_layer(hs).permute(0, 2, 1)
            if idx + 1 == self.config.quantize_position and self.codebook is not None:
                if quantized_token_ids is not None:
                    hidden_states = self.codebook(quantized_token_ids)
                else:
                    hidden_quantized, indices_flat, _ = vector_quantize(
                        hidden_states, self.codebook.weight
                    )
                    quantized_token_ids = indices_flat.reshape(
                        batch_size, hidden_quantized.shape[1]
                    )
                    hidden_states = hidden_quantized
                hidden_states = (
                    hidden_states
                    + self.embed_positions2.weight[: hidden_states.shape[1]]
                )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return QuantizedBaseModelOutputWithCache(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            quantized_token_ids=quantized_token_ids,
            past_key_value=past_key_values,
            conv1_cache=new_conv1_cache,
            conv2_cache=new_conv2_cache,
        )
