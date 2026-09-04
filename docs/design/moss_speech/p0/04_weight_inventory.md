# P0-04 Weight Inventory & Mapping (T0.5)

Full tensor dump: `artifacts/p0/weight_shapes.txt` (446 tensors, all BF16). Integrity: `artifacts/p0/weights_sha256.txt`.

## Summary

| namespace | tensors | shapes |
|---|---|---|
| `model.embed_tokens.weight` | 1 | [151680, 4096] |
| `model.audio_embed.weight` | 1 | [16512, 4096] |
| `model.shared_block.layers.{0..31}` | 10 each | q/k/v/o_proj [4096,4096]/[1024,4096]/[1024,4096]/[4096,4096]; q/k_norm [128]; ln ×2 [4096]; mlp gate/up [12288,4096], down [4096,12288] |
| `model.text_block.layers.{0..3}` | 10 each | same as shared |
| `model.audio_block.layers.{0..3}` | 10 each | same as shared |
| `model.text_norm.weight` / `model.audio_norm.weight` | 1+1 | [4096] |
| `text_lm_head.weight` | 1 | [151680, 4096] |
| `audio_lm_head.weight` | 1 | [16512, 4096] |

- No missing/unexpected keys when loading `MossSpeechForCausalLM` (446/446).
- Both lm heads are **explicit checkpoint tensors** → not tied despite `_tied_weights_keys` (that list only affects save-time dedup / load-time fallback when keys are absent).

## HF → SGLang mapping (flattened 40-layer view)

```
model.embed_tokens.weight            -> embed_tokens (text)
model.audio_embed.weight             -> audio embed (second table)
model.shared_block.layers.N.*        -> trunk layer N        (N: 0..31)
model.text_block.layers.N.*          -> text tail layer 32+N (N: 0..3)
model.audio_block.layers.N.*         -> audio tail layer 36+N (N: 0..3)
model.text_norm.weight               -> text branch final norm
model.audio_norm.weight              -> audio branch final norm
text_lm_head.weight                  -> text lm head (151680)
audio_lm_head.weight                 -> audio lm head (16512)
```

All per-layer tensor names follow the Qwen3 convention (q_norm/k_norm per-head RMSNorm) — the SGLang Qwen3 loader covers the projection set; the delta is dual-embed/dual-head/three-cache-block topology.

## Codec weights (`models/MOSS-Speech-Codec`)

`model.safetensors` (encoder+quantizer, `WhisperVQEncoder`) + `flow/{flow.pt, flow-chunk-5.pt, flow-chunk-25.pt, hift.pt, campplus.onnx}` (decoder stack). Hashes in `weights_sha256.txt`.

## KV accounting (see 05_trace_and_kv.md for measured values)

40 layer instances × 2 (K+V) × 8 kv-heads × 128 head_dim × 2 B = **160 KB/token** (bf16):

| context | KV GiB |
|---|---|
| 1024 | 0.16 |
| 4096 | 0.62 |
| 8192 | 1.25 |
| 10240 | 1.53 |
