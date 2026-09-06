# P1-03 Performance & Concurrency Profile (T1.4)

Date: 2026-09-06. Node: slurmd-6, A800-SXM4-80GB, driver 5xx/cu128, torch 2.9.1.
Scripts: `scripts/moss_speech/p1/{bench_codec,ab_placement,ar_load}.py`.
Raw: `artifacts/p1/perf/{bench_codec.json, ab_*.json, *.wav}` (logs `sbatch_bench_3670`, `sbatch_ab_3671`, `sbatch_abar_3674`).

## 1. Encoder sweep (12-item mixed manifest: 6×3 s + 6×27 s)

| internal batch_size | wall p50 (ms) | per-item (ms) | per-code equal to bs=1 |
|---|---|---|---|
| 1 | 528 | 44.0 | — (baseline) |
| 4 | 554 | 46.2 | ✅ |
| 8 | 775 | 64.6 | ✅ |
| 16 | 771 | 64.3 | ✅ |
| 32 | 772 | 64.3 | ✅ |
| 64 | 773 | 64.4 | ✅ |
| 128 | 772 | 64.3 | ✅ |

- No OOM at any batch size (peak 2.69 GiB).
- **Batching does not pay on mixed lengths** (padding to the longest item dominates); bs=4 ≈ bs=1. Homogeneous-length coalescing may differ — not needed for V1.
- Batch safety: **per-code equality holds at every tested batch size** (A1/A2 of the alignment run independently confirm).

## 2. Decoder serial service (V1 contract: execution concurrency 1)

| codes | decode p50 (ms) | arrival threads 2 (ms) |
|---|---|---|
| 100 | 1126 | 2275 |
| 200 | 1462 | 2946 |
| 500 | 2825 | 5669 |

- ~4.2 ms/code slope + ~0.7 s prompt-prefix overhead (27 s default voice → 342-token prompt prefix).
- Arrival backlog queues linearly behind the decode lock (expected; queue concurrency ≠ execution concurrency).
- **Zero VRAM growth over 50 repeat decode+cleanup cycles** (0.0 MiB).
- RNG: per-request seeding required (HiFT draws; T1.3 finding) — implemented in the A/B decoder as the P2 contract.

## 3. VRAM breakdown (per process, fp32 codec)

| component | alloc (GiB) | peak (GiB) |
|---|---|---|
| encoder-only (whisper-VQ fp32) | 1.31 | 2.69 |
| decoder-only (flow+hift fp32) | 0.65 | 3.10 |
| campplus/voice features | (in encoder proc) | — |
| reference AR **bf16** (load / peak with KV) | 16.94 | 18.53 |

## 4. Placement A/B (single GPU, one encoder instance, decoder terminal process, default voice precomputed and delivered through the queue; 24 req/round = 12 speech (6×3 s + 6×27 s) + 12 text, arrival 0.4 s; 3 warmup + 3 recorded rounds; transport = framework SimpleScheduler queues with pickled payloads)

| config | throughput (rps) | e2e p50 (s) | e2e p95 (s) | errors |
|---|---|---|---|---|
| A (preproc+encoder colocated), no AR | 0.631 | 15.19 | 25.36 | 0 |
| B (separate encoder process), no AR | 0.630 | 15.20 | 25.39 | 0 |
| A + AR load (bf16, busy=1.0, 248 iters) | 0.350 | 30.77 | 52.32 | 0 |
| B + AR load | 0.351 | 30.63 | 52.09 | 0 |

- **A ≡ B within noise** in both regimes; the pipeline is decode-serial-bound
  (decode p50 1.34 s vs encode ~15–500 ms and preproc ~0).
- AR contention costs ~44% throughput / ~2× p50 at this load — V1 admission
  must budget codec decode against AR GPU time (plan §6.3's "preprocessing/codec
  限并发" applies to the decoder terminal most of all).
- Aggregate steady-state VRAM (AR + decoder + encoder + 3 CUDA contexts):
  ≈ 23.5 GiB — **24 GB is borderline-feasible at short contexts** (KV at SFT
  10K context ≈ +1.6 GiB → over budget; P5 must re-measure with real KV use).

## 5. Optional probes — status

- chunk5/25 streaming reference probe: **not run** (optional track, D4; V1.1
  input remains the P0 API-surface record).
- bf16/fp16 codec: **not run** (optional track; D2 default FP32 unchanged).
