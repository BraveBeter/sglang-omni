#!/usr/bin/env python3
"""Codec performance & concurrency profile (T1.4, part 1: single process).

Sections:
  1. encoder sweep: internal batch_size {1,4,8,16,32,64,128} on a mixed
     short/long manifest; per-code equality vs bs=1 gates each step (stop on
     OOM or inequality); wall + CUDA-event device time.
  2. decoder serial service with arrival concurrency {1,2,4,8} (queue
     backlog pressure; execution stays 1 per the V1 contract), codes
     {100,200,500}, per-request seeded RNG.
  3. VRAM breakdown: encoder-only / decoder-only / voice-feature-only
     weights, allocated/reserved peaks, load transients; repeat-decode
     growth check.

Run on a compute node in `.venv-omni` (clean env).
"""

from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import statistics
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import torch

from sglang_omni.models.moss_speech.components.codec_adapter import MossSpeechCodecAdapter


def cuda_time_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"gpu": torch.cuda.get_device_name(0)}

    short = str(Path(args.assets_dir) / "short_en_3s.wav")
    long_ = str(Path(args.assets_dir) / "long_cn_27s.wav")
    manifest = [short] * 6 + [long_] * 6

    # ---------------- 1. encoder sweep --------------------------------------
    torch.cuda.reset_peak_memory_stats()
    adapter = MossSpeechCodecAdapter(args.codec_path, load_decoder=False)
    base_codes = adapter.encode(manifest, batch_size=1)
    enc_rows = []
    for bs in (1, 4, 8, 16, 32, 64, 128):
        try:
            codes = adapter.encode(manifest, batch_size=bs)
            equal = all(a == b for a, b in zip(codes, base_codes))
            if not equal:
                enc_rows.append({"bs": bs, "status": "STOPPED: per-code inequality vs bs=1"})
                break
            # warmup x1 then timed rounds
            for _ in range(1):
                adapter.encode(manifest, batch_size=bs)
            walls, devs = [], []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                _, dev = cuda_time_ms(lambda: adapter.encode(manifest, batch_size=bs))
                walls.append((time.perf_counter() - t0) * 1000)
                devs.append(dev)
            enc_rows.append({
                "bs": bs, "equal_to_bs1": equal,
                "wall_ms_p50": round(statistics.median(walls), 1),
                "wall_ms_p95": round(sorted(walls)[int(0.95 * len(walls)) - 1], 1) if len(walls) > 1 else round(walls[0], 1),
                "device_ms_p50": round(statistics.median(devs), 1),
                "per_item_ms_p50": round(statistics.median(walls) / len(manifest), 2),
            })
        except torch.cuda.OutOfMemoryError:
            enc_rows.append({"bs": bs, "status": "OOM"})
            break
    res["encoder_sweep"] = enc_rows
    res["encoder_vram"] = {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    adapter.close()
    torch.cuda.empty_cache()

    # ---------------- 2. decoder serial + arrival concurrency ----------------
    torch.cuda.reset_peak_memory_stats()
    dec = MossSpeechCodecAdapter(args.codec_path, load_encoder=False)
    voice = None  # decoder-only has no encoder; use a precomputed voice file
    # precompute voice with a temporary encoder-included adapter is wasteful;
    # instead reuse the full adapter for voice once, then close encoder side.
    full_tmp = MossSpeechCodecAdapter(args.codec_path, load_encoder=True, load_decoder=False)
    import shutil, tempfile

    # voice from the long asset (default voice = prompt-cn semantics)
    voice = full_tmp.encode_voice_ref(long_)
    full_tmp.close()
    torch.cuda.empty_cache()

    dec_rows = []
    for n_arrival in (1, 2, 4, 8):
        for n_codes in (100, 200, 500):
            codes = ([100] * n_codes)
            # warmup
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            dec.decode(codes, voice, request_id="warm")
            walls = []
            for r in range(args.rounds):
                q: "queue_mod.Queue[str]" = queue_mod.Queue()
                for i in range(20):
                    q.put(f"r{r}-{i}")

                def worker() -> None:
                    while True:
                        try:
                            rid = q.get_nowait()
                        except queue_mod.Empty:
                            return
                        torch.manual_seed(0)
                        torch.cuda.manual_seed_all(0)
                        t0 = time.perf_counter()
                        dec.decode(codes, voice, request_id=rid)
                        walls.append((time.perf_counter() - t0) * 1000)

                ts = [threading.Thread(target=worker) for _ in range(n_arrival)]
                t_start = time.perf_counter()
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                # walls across threads: single decode lock serializes them
            dec_rows.append({
                "arrival_threads": n_arrival, "n_codes": n_codes,
                "decode_ms_p50": round(statistics.median(walls), 1),
                "decode_ms_p95": round(sorted(walls)[int(0.95 * len(walls)) - 1], 1),
            })
    res["decoder_serial"] = dec_rows
    res["decoder_vram"] = {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }

    # growth check: repeat decode/cleanup cycles, allocated must not grow
    a0 = torch.cuda.memory_allocated()
    for i in range(50):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        dec.decode([100] * 200, voice, request_id=f"growth-{i}")
    res["decode_repeat_growth_mib"] = round((torch.cuda.memory_allocated() - a0) / 2**20, 2)
    dec.close()

    (out_dir / "bench_codec.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
