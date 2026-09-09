# P4 gate report

> Historical evidence: `scripts/p*.sbatch` names below identify workspace-local
> cluster wrappers used for the recorded jobs; they are not files delivered in
> Git and are not runnable reproduction instructions for a fresh checkout.
> Use the current [cookbook](../../../cookbook/moss_speech.md) for serving and
> [P5 reproduction](../p5/05_gate_report.md) / [P6 reproduction](../p6/02_reproduction.md)
> for the delivered Python and shell entry points. Historical receipts remain
> bound to their original commits and dependency versions.


Date: 2026-09-08. **P4 G1–G4 all passed; T4.1–T4.6 complete.**
P5 has not started. Source commits: shared HTTP `6b22b8d`, MOSS serving and
validation drivers `9dae728`. P3 remains published at `dc2c517`; parity v1/E1
is unchanged. Delivery branch: `fork/feat/moss-speech`.

The machine-readable evidence index is
`sglang-omni/docs/design/moss_speech/p4/gate_summary.json` (workspace copy:
`artifacts/p4/gate_summary.json`). It includes raw report, source, static-check,
retained-failure and Slurm SHA256 values.

| Gate | Evidence | Result |
|---|---|---|
| G1 HTTP four modes | 3864: frozen BF16 input/output grids, text and WAV; 3866: actual CLI | PASS |
| G2 validation | 14 HTTP invalid cases, post-encode exact context 400, CPU matrix | PASS |
| G3 lifecycle | AR-event disconnect, survivors/recovery, random seed isolation, real 24/25 admission, all child exits | PASS |
| G4 native load | 3865: 144 requests, 3 warmup +3 measured rounds; 3867: serial/interleaved WAV+grid equality | PASS |

## Completed evidence

- Real TCP HTTP + public Client + formal native YAML: `artifacts/p4/http_3864/report.json`, pass true.
- Four frozen BF16 reference inputs and grids; text decoded from those grids;
  WAV bytes exactly match the same reference codes through P1 codec/voice/seed.
- Inline audio and audios[] placeholder binding; 14 invalid request cases;
  exact post-encode oversized prompt returns 400 before AR.
- HTTP disconnect after AR prefill emits an actual aborted model-path event;
  survivor/recovery are unchanged. Concurrent random same-seed outputs match,
  changed seed changes output.
- Four 512-row prompts with a requested 512-row generation budget each produce
  476 rows and stop. This is not 512 actual generated rows.
- Real coordinator capacity 24, 25th HTTP request returns 503; all 24 cancelled
  requests clear and the following request reproduces baseline.
- All four stage child processes exit normally. CLI launch, both terminals,
  preflight hook and complete subprocess shutdown independently pass in
  `artifacts/p4/cli_3866/report.json`.
- NVML sampling interval 0.5s; HTTP-run device peak observed 29.188 GiB
  on NVIDIA A800-SXM4-80GB (GPU-5bc54404-37a4-00d7-45ab-904f0c51e6ae). This is sampled device usage,
  not an instantaneous allocator peak. Final KV is 4096 positions ×163840 bytes,
  approximately 0.625 GiB plus the physical padding slot.

## Retained failures and scope

3861 used old P0 text files as expected values; P3 BF16 grids and audio were
already exact. The driver now decodes frozen P3 grids for text expectations;
no reference asset was changed.

3862 OOMed on a roughly 6K prompt with the old large KV reservation. The final
serving envelope is prompt512/new512/context1024, AR running4/queued20, KV4096.
3863 passed that envelope before KV capping; 3864 revalidated after capping.
Longer-context capacity needs independent expansion qualification. P3 numerical
parity, checkpoint metadata, and previous failures are not rewritten.

The broad CPU HTTP suite has one existing M4A binary-dependency failure. It
reproduces using the exact unchanged P3 HTTP module at dc2c517, with torchcodec
reporting a missing torch CUDA ABI symbol. See
`artifacts/p4/preexisting_m4a_failure.log`. Do not present that suite as fully
passing. MOSS WAV and targeted HTTP/lifecycle/launcher tests pass separately.

## Native load and placement decision

Run 3865 repeats the P1 input composition: 24 requests per round, 12 text and
12 speech (6×3s +6×27s), 0.4s arrivals. All request audio outputs come from
actual native AR; max_new_tokens=200. Three warmup and three measured rounds
complete with no errors. Repeated input hashes have identical grids throughout.
The measured requests generate 13,932 grid rows in total;
maximum prompt length is 375. Separate run3867 compares
both full HTTP WAV and grid hashes for 3 serial baselines +12 interleaved
requests (client concurrency4), all equal; actual stage child PIDs exit normally.

| Metric | Measured result |
|---|---:|
| Measured / warmup requests | 72 / 72 |
| Aggregate throughput | 0.1031 requests/s |
| E2E p50 / p95 (includes queue) | 124.88 / 222.75 s |
| Native batch observed | 4 |
| Workload NVML sampled device peak | 27.026 GiB |
| Measured round-end memory spread | 0.0 MiB |
| AR / encoder-process / vocoder-process sampled peaks | 21.260 / 2.104 / 2.893 GiB |
| HTTP boundary-run sampled peak (3864) | 29.188 GiB |

These are **24-request bursts**, followed by draining the backlog. An arrival
interval of0.4s is not a sustainable2.5rps service claim. P1 used synthetic codes
and background reference AR, so these values cannot be interpreted as a P1
speedup/slowdown ratio. Client concurrency1/2/4 probes each complete 8 short text
requests in15.54/12.90/10.16s respectively; use1 for latency-sensitive work,
and measure the application's own workload before choosing2 or4.

Retain **Layout A**, one inline encoder in preprocessing, decoder in its own
serial terminal, encode internal batch4. Measured stage p50/p95:
preprocessing0.020/0.157s; vocoder1.943/3.321s; AR37.765/38.521s. AR elapsed is
prefill-start to model-path-end, excluding admission waiting. Codec stage
elapsed is dispatch to completion, including its local queue. Encoder admission
is not the bottleneck in this manifest; no evidence triggers the Layout B
fallback. Do not increase codec execution concurrency on this evidence.

## Validation and reproduction

- CPU release:148 passed,0 skipped (`artifacts/p4/cpu_release_final.log`).
- Public chat regression:7 passed (`artifacts/p4/cpu_http_launcher.log`);
  launcher/IPC regression:14 passed (`artifacts/p4/cpu_launcher.log`).
- 12 changed/new Python files: black/isort/ruff(py310)/AST pass;
  git diff --check passes (`artifacts/p4/static_checks.json`).
- GPU runs3864/3865/3866/3867 and their actual child processes exit successfully;
  accounting in `artifacts/p4/slurm_accounting.txt`. No live Slurm jobs remain.

From the workspace root in the configured offline environments:

```bash
sbatch scripts/p4_http.sbatch
sbatch scripts/p4_http.sbatch --workload
sbatch scripts/p4_http.sbatch --soak
sbatch scripts/p4_cli.sbatch
.venv-omni/bin/python sglang-omni/scripts/moss_speech/p4/summarize.py \
  --http artifacts/p4/http_3864 --workload artifacts/p4/http_3865 \
  --cli artifacts/p4/cli_3866 --soak artifacts/p4/http_3867 \
  --cpu artifacts/p4/cpu_release_final.log artifacts/p4/cpu_http_launcher.log artifacts/p4/cpu_launcher.log \
  --output artifacts/p4/gate_summary.json
```

Substitute fresh run directories when reproducing. The aggregate tool derives
metrics from existing reports and refuses failed/incomplete evidence; the
archived index additionally records source commits/hashes and retained failures.
The final driver also records WAV hashes per request. Historical3865's grid-only
long-run record is supplemented by3867; its raw report has not been rewritten.

## P5 handoff

P5 starts from the bounded, tested A800 configuration. Reuse these HTTP drivers
and the cookbook at `sglang-omni/docs/design/moss_speech/p4/02_serving.md`.
Add quality benchmarks and CI presets, resolve checkpoint license metadata
before upstream packaging, and qualify optional hardware/context expansion.
Start24G investigation from current measured footprints; do not presume4K or
24G feasibility. Larger context requires an explicit memory/parity/load gate.
Repair the existing torchcodec ABI issue before claiming general M4A endpoint
support. Streaming stays in P6. Performance work in P7 should target native AR
first while preserving the per-request numerical shape rules from P3.
