# P1-04 Placement & P2 Handoff Decision (T1.5)

Date: 2026-09-06. Evidence: `03_perf.md` (runs 3670/3671/3674), `02_alignment.md` (run 3669), `01_framework_notes.md`.

## Decisions

### D-1 Encoder placement: **Layout A — colocated in preprocessing** (colocated preprocessing+encoder, one process)

- Evidence: A ≡ B within measurement noise in both no-AR and AR-contention
  regimes (0.631/0.630 and 0.350/0.351 rps; p50 delta < 0.5%); encode cost
  (15 ms short / ~0.5 s long) is 3 orders below decode-serial service time.
- Per the frozen rule ("both pass, no decisive benefit → pick the simpler
  A"), the extra process/queue/context of Layout B buys nothing at V1 loads.
- B remains the documented fallback if P4 native-AR profiling shows encoder
  spikes blocking admission (re-measure then; reference-AR numbers are not
  native-AR numbers — P4 re-runs this workload).

### D-2 Decoder terminal: **dedicated `audio_vocoder` process**, SimpleScheduler, execution concurrency = 1

- decode is the pipeline bottleneck (1.1–2.8 s/request); execution
  concurrency >1 is NOT enabled: HiFT consumes global RNG and AudioDecoder
  state is per-uuid — raising concurrency requires the P4-verified
  request-scoped RNG + state isolation and a measured win (D5 unchanged).
- Request-scoped RNG contract (from T1.3): seed per request-id before each
  decode; save/restore global RNG around the call (encode/voice paths are
  already RNG-free).

### D-3 Process→stage→device→component→dtype→replica table (V1 baseline)

| process | stage(s) | device | components | dtype | replicas |
|---|---|---|---|---|---|
| preproc+encoder | `preprocessing` (+inline audio encode) | GPU 0 | MossSpeechCodecAdapter(load_encoder=True, load_decoder=False) + campplus + voice hook | fp32 | 1 |
| AR engine | `tts_engine`-analogue (chat AR) | GPU 0 | native SGLang model (P3) | bf16 | 1 (TP=1) |
| decoder terminal | `audio_vocoder` | GPU 0 | MossSpeechCodecAdapter(load_encoder=False, load_decoder=True) + default voice (delivered via queue at startup) | fp32 | 1 |

- Default voice: precomputed in the encoder-side process at startup,
  serialized once through the stage queue (exercised in the A/B runs);
  per-request ad-hoc voices later via ReferenceEncodeService (hook ready).
- Encode batching: internal `batch_size=4` (sweep optimum; safety proven to
  128 — larger sizes only waste padding on mixed lengths).
- Concurrency gates: preprocessing unlimited (CPU-bound validation), encoder
  natural backpressure via colocated compute_fn, decoder admission-limited
  (serial); overall pipeline admission budget must account for AR+codec GPU
  contention (−44% throughput at full AR load in this profile).

### D-4 Hardware boundary statements

- 80G A800: comfortable (AR 17 + codec ~4.5 + KV ≤1.6 + activations).
- 24G: **borderline at short context** (≈23.5 GiB steady-state measured
  incl. contexts); at SFT 10K-context KV it exceeds budget. P5 must verify
  with real KV usage; candidates: codec bf16 (needs its own frozen quality
  gate) or CPU-offloaded decode. Not a V1 commitment.
- AR/codec split across GPUs: V2 track (E6 precedent), not assessed here.

### D-5 Scope & re-verification responsibility

- These are **reference-AR-load conclusions**. P2 wires the real DAG
  (StageConfig below) and validates payload/device binding; P4 re-runs the
  same workload manifest with the native AR and adjusts concurrency/memory
  budgets; any layout regression found in P4 reopens this decision.
- Not changed: V1 non-streaming, no radix cache, no CUDA graph, TP=1.

## StageConfig draft fields for P2 (indicative, non-exhaustive)

```yaml
stages:
  - name: preprocessing          # colocated encode (Layout A)
    device: cuda:0
    factory: moss_speech.preprocessing
    scheduler: SimpleScheduler    # max_concurrency=1 default; compute_fn = validate+encode
  - name: ar_engine              # P3 native model
    device: cuda:0
    scheduler: OmniScheduler
  - name: audio_vocoder          # terminal (audio-modality requests only)
    device: cuda:0
    factory: moss_speech.vocoder
    scheduler: SimpleScheduler    # serial; request-scoped RNG seed per request_id
codec:
  path: models/MOSS-Speech-Codec   # locked eeec733e materialization
  dtype: float32
  encode_batch_size: 4
  default_voice: <asset path + sha256>  # precomputed at startup, IPC-delivered
limits:
  decoder_in_flight: 1
```
