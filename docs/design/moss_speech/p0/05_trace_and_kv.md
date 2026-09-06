# P0-05 Model Trace & KV/VRAM Accounting (T0.6)

Date: 2026-09-04. Node: A800-SXM4-80GB. Script: `sglang-omni/scripts/moss_speech/p0/trace_model.py`; raw: `artifacts/p0/trace/{trace_summary.json,trace_events.json}` (log `sbatch_trace_3632.log`). Probe case: T2S "用一句话介绍长城。", greedy, 91 generated steps.

## Structural claims (plan §5.4 / §6.1)

| claim | result | evidence |
|---|---|---|
| C1 both tails execute every forward | ✅ | tail calls text=91, audio=91; every forward's `modalities == ["text","audio"]` |
| C2 dual-channel sampling each step | ✅ | 90 audio-mode steps: text channel overwritten with modality_pad(151667); 1 text-mode step; audio channel sampled at every step (asymmetric padding — audio channel NOT padded during text steps; ignored on input side) |
| C3 text tail crosses audio-segment KV | ✅ structurally | all three caches share seq_len=125 (34 prompt + 91 gen); post-eosp text-mode step(s) execute and attend over KV written at audio positions. **Empirical nuance:** in all observed runs (trace probe, t2s/s2s/mixed fixtures) the model emits `<|im_end|>` immediately after eosp — no content-bearing text tail. Cross-reading is real but the resumed segment is the stop token only. |
| C4 no single-tail shortcut | ✅ | tail call counts equal; no modality filtering anywhere in forward |

## KV accounting

- Topology: `past_key_values_dict = {shared: 32-layer DynamicCache, text: 4, audio: 4}` — **40 layer instances** total; "tail reuses trunk layer slots" is structurally impossible (three separate caches; plan §6.1 confirmed).
- Per token: 40 × 2 × (8 kv-heads × 128) × 2 B = **160 KB** (bf16).
- Context curve: 1K→0.16 GiB, 4K→0.62 GiB, 8K→1.25 GiB, 10K→1.56 GiB.

## VRAM (colocated AR + codec, bf16, A800)

| phase | allocated GiB |
|---|---|
| after model+codec load | 35.73 |
| AR generation peak | 35.78 |
| codec decode peak | 36.47 |
| torch reserved (peak) | 38.97 |

Decomposition: AR ≈ 17.1 GiB (bf16) + codec stack ≈ 18.6 GiB (whisper encoder + flow + hift + campplus; flow/hift loaded fp32). **4090-24G colocated is NOT feasible at current dtypes** — P5 hardware variant needs codec downcast/fp16 or split placement (E6 precedent: split raised throughput 3.6×). 80G headroom is ample.

## Port implications (input to P2/P3)

1. 40-layer flattened KV accounting is mandatory; organize as 32/4/4 data flow inside the model (plan §6.1 spike question answered affirmatively).
2. Both tails must run per step even in pure-text output mode (logits_all consumed by the sampler); no execution shortcut available for V1.
3. The post-eosp text step exists (1 step to im_end); the runner FSM must keep text-channel sampling active after eosp until im_end.
4. Asymmetric channel padding (text→pad during audio; audio→sampled-but-ignored during text) must be reproduced exactly in the model runner hooks; audio logits masking `[16385:]=−inf` is per-step on the audio channel.

> **Correction (2026-09-06, from P1 T1.2 smoke):** the "after model+codec load = 35.73 GiB" figure above is dominated by the **reference AR loaded in fp32** (transformers `AutoModel` without `torch_dtype` in the `.venv-p0` environment), not by the codec. Direct measurement of the codec stack alone: encoder-only 1.31 GiB, decoder-only 0.54 GiB, full codec ≈ 2.4 GiB allocated. For the V1 target topology (native AR bf16 ≈ 17.1 GiB + codec fp32 ≈ 2.4 GiB ≈ **19.5 GiB**), a 24 GB card is provisionally feasible — to be confirmed by T1.4 multi-process placement measurements. The 40-layer KV accounting and all structural claims are unaffected.

