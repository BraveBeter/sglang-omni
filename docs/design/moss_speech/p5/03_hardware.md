# P5 hardware qualification

Date:2026-09-08. **The tested BF16-AR/FP32-codec serving configurations do not
qualify on RTX4090-24GB.** A80080GB remains the only fully qualified hardware.
This is a measured configuration boundary, not a claim that every possible
24GB/offload/quantized implementation is impossible. No supported24GB YAML is
published. 32GB/48GB devices have not been tested.

All trials use real Slurm RTX4090 allocations (GPU UUID
GPU-0d448cad-0cd5-8206-3fef-ee7cb1fa7084), unchanged native math, BF16 KV,
FP32 encoder/vocoder and local/offline checkpoints. TP1; codec concurrency1.
NVML reports23.988GiB total while CUDA's OOM diagnostic reports23.55GiB available
capacity; these are different accounting interfaces, not interchangeable peaks.

| Run | Settings | Observed outcome |
|---|---|---|
| 3870 | Unchanged P4 YAML: AR4, KV4096, static fraction0.72 | AR weights load, but static budget leaves no KV capacity; initialization rejects, all children exit |
| 3873 | AR1, KV1024, placement fraction0.80; actual SGLang fraction still0.72 | Same initialization rejection; exposed a configuration-generator mistake, not an independent physical-limit result |
| 3875 | AR1, KV1024, actual SGLang fraction0.80 | Initialization requires fraction>0.845 in that startup state; all children exit |
| 3879 | AR1, KV1024, actual SGLang fraction0.90; prompt/output budgets still512 | Service starts and attempts all32 requests;25 succeed and7 return500 due to vocoder CUDA OOM; all four children later exit0 |

The last run reaches a sampled NVML device peak of23.929GiB. The first failure
is zh_00_s2s: FP32 codec attention attempts an additional358MiB with only63MiB
CUDA free (and87MiB reserved but unallocated in that process). The OOM diagnostic
accounts for the AR at18.46GiB, encoder process2.07GiB and decoder process2.93GiB.
This demonstrates actual working-set pressure even after reducing AR concurrency
and the KV pool; it cannot be resolved merely by raising the static fraction.

The failed trial's A800-reference token comparisons are **not** a calibration of
4090 kernels: BF16 results can differ across GPU architectures, and its reference
also predates the explicit logits_to_keep correction. The independent physical
OOM and HTTP failures are sufficient to reject this serving configuration. Do
not describe every cross-GPU token difference as an implementation defect or
reuse the trial as a same-hardware quality result.

Evidence (workspace-relative):

- `artifacts/p5/24gb_baseline/report.json`, `artifacts/p5/24gb_3870.log`.
- `artifacts/p5/native_3873/report.json`, `artifacts/p5/probe_24gb.yaml`.
- `artifacts/p5/native_3875/report.json`, `artifacts/p5/probe_24gb_v2.yaml`.
- `artifacts/p5/native_3879/report.json`, `artifacts/p5/native_3879.log`,
  `artifacts/p5/probe_24gb_v3.yaml`.

The probe generator's default emits the0.80 experiment. The final v3 artifact
changes only `ar_engine.factory_args.server_args_overrides.mem_fraction_static`
to0.90 and the IPC directory. Placement metadata and SGLang's actual static
fraction are distinct; the regression test now verifies both are explicitly
accounted for. These trial YAMLs remain artifacts, not supported examples.

P5 closes the optional24GB investigation with this negative result. A future
CPU-offload, codec low-precision, shorter-output or alternate-placement proposal
must specify its own budget and then pass same-hardware reference/codec quality,
full HTTP, boundary/load, abort/recovery and cleanup gates. Existing P3/P4
reference artifacts and the A800 deployment budget remain unchanged.
