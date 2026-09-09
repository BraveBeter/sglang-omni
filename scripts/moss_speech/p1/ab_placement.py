#!/usr/bin/env python3
"""Placement A/B experiment for the MOSS-Speech codec (T1.4).

Layout A: preprocessing + encoder in ONE process (colocated compute_fn does
validation + encode inline).
Layout B: preprocessing process (CPU passthrough) -> mp.Queue -> separate
encoder process. Both: one encoder instance, single shared GPU, independent
decoder terminal process, text requests skip the encoder, default voice
precomputed on the encoder side and delivered to the decoder through the
queue (exercising the real serialization boundary).

Scheduler transport uses the framework's actual classes: SimpleScheduler
inboxes/outboxes (mp-safe queues, pickled payloads).

Usage:
  python ab_placement.py --layout A|B [--with-ar] --codec-path ... --out ...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

N_REQ = 24  # per round: 12 speech (6 short + 6 long), 12 text
CODES_POOL = [100, 200, 500]


# --------------------------------------------------------------------- workers
def _preproc_worker(layout: str, codec_path: str, in_q, out_q, stats) -> None:
    """Layout A: validate + encode inline. Layout B: CPU passthrough."""
    import torch  # local import per spawned process

    adapter = None
    if layout == "A":
        from sglang_omni.models.moss_speech.components.codec_adapter import (
            MossSpeechCodecAdapter,
        )

        adapter = MossSpeechCodecAdapter(
            codec_path, load_encoder=True, load_decoder=False
        )
        # precompute default voice and forward it to the decoder process
        voice = adapter.encode_voice_ref(str(Path(stats["assets"]) / "long_cn_27s.wav"))
        out_q.put({"type": "voice", "voice": voice})
    while True:
        msg = in_q.get()
        if msg.get("type") == "stop":
            break
        t0 = time.perf_counter()
        req = msg["req"]
        if req["kind"] == "speech":
            if adapter is not None:
                codes = adapter.encode([req["wav"]], batch_size=128)[0]
                req["codes_in"] = codes
        req["preproc_done_t"] = time.perf_counter()
        req["stage_wall"] = {"preproc": time.perf_counter() - t0}
        out_q.put({"type": "req", "req": req})
    if adapter is not None:
        adapter.close()
    stats["proc_preproc"] = {
        "alloc_gib": (
            round(torch.cuda.memory_allocated() / 2**30, 3)
            if torch.cuda.is_available()
            else 0
        ),
        "peak_gib": (
            round(torch.cuda.max_memory_allocated() / 2**30, 3)
            if torch.cuda.is_available()
            else 0
        ),
    }


def _encoder_worker_b(codec_path: str, in_q, out_q, stats) -> None:
    """Layout B only: dedicated encoder process using a real SimpleScheduler."""
    import torch

    from sglang_omni.models.moss_speech.components.codec_adapter import (
        MossSpeechCodecAdapter,
    )
    from sglang_omni.scheduling.messages import IncomingMessage
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    adapter = MossSpeechCodecAdapter(codec_path, load_encoder=True, load_decoder=False)
    voice = adapter.encode_voice_ref(str(Path(stats["assets"]) / "long_cn_27s.wav"))
    out_q.put({"type": "voice", "voice": voice})

    def compute_fn(data):
        req = data
        if req["kind"] == "speech":
            t0 = time.perf_counter()
            req["codes_in"] = adapter.encode([req["wav"]], batch_size=128)[0]
            req.setdefault("stage_wall", {})["encode"] = time.perf_counter() - t0
        req["enc_done_t"] = time.perf_counter()
        return req

    sched = SimpleScheduler(compute_fn)
    threading.Thread(target=lambda: (sched.start()), daemon=True).start()

    while True:
        msg = in_q.get()
        if msg.get("type") == "stop":
            break
        req = msg["req"]
        sched.inbox.put(
            IncomingMessage(request_id=req["id"], type="new_request", data=req)
        )
        out_m = sched.outbox.get()
        if out_m.type == "error":
            req["error"] = str(out_m.data)[:100]
        else:
            req = out_m.data
        out_q.put({"type": "req", "req": req})
    adapter.close()
    stats["proc_encoder"] = {
        "alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }


def _decoder_worker(codec_path: str, in_q, out_q, stats) -> None:
    import torch

    from sglang_omni.models.moss_speech.components.codec_adapter import (
        MossSpeechCodecAdapter,
    )

    adapter = MossSpeechCodecAdapter(codec_path, load_encoder=False, load_decoder=True)
    voice = in_q.get()["voice"]  # delivered via the real serialization boundary

    while True:
        msg = in_q.get()
        if msg.get("type") == "stop":
            break
        req = msg["req"]
        n = CODES_POOL[req["slot"] % len(CODES_POOL)]
        codes = req.get("codes_in") or req["codes_assigned"]
        torch.manual_seed(req["id_hash"] & 0x7FFFFFFF)
        torch.cuda.manual_seed_all(req["id_hash"] & 0x7FFFFFFF)
        t0 = time.perf_counter()
        try:
            sr, wav = adapter.decode(codes[:n], voice, request_id=req["id"])
            req["n_samples"] = int(wav.shape[-1])
        except Exception as e:  # noqa: BLE001
            req["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        req.setdefault("stage_wall", {})["decode"] = time.perf_counter() - t0
        req["dec_done_t"] = time.perf_counter()
        out_q.put({"type": "req", "req": req})
    adapter.close()
    stats["proc_decoder"] = {
        "alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }


# --------------------------------------------------------------------- driver
def run_round(driver_q, preproc_q, round_id: int, assets: Path, results: list) -> None:
    reqs = []
    for i in range(N_REQ):
        kind = "speech" if i % 2 == 0 else "text"
        wav = str(
            assets / ("short_en_3s.wav" if (i // 2) % 2 == 0 else "long_cn_27s.wav")
        )
        rid = f"r{round_id}-{i:02d}"
        reqs.append(
            {
                "id": rid,
                "id_hash": abs(hash(rid)) % (2**31),
                "kind": kind,
                "wav": wav,
                "slot": i,
                "codes_assigned": [100 + (7 * j) % 16000 for j in range(500)],
                "enqueue_t": None,
            }
        )
    t_start = time.perf_counter()
    for i, req in enumerate(reqs):
        req["enqueue_t"] = time.perf_counter()
        preproc_q.put({"type": "req", "req": req})
        time.sleep(0.4)  # fixed arrival schedule
    # collect all
    got = {}
    deadline = time.perf_counter() + 300
    while len(got) < N_REQ and time.perf_counter() < deadline:
        try:
            msg = driver_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if msg.get("type") == "req":
            got[msg["req"]["id"]] = msg["req"]
    wall = time.perf_counter() - t_start
    results.append(
        {
            "round": round_id,
            "wall_s": round(wall, 2),
            "n_done": len(got),
            "reqs": list(got.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["A", "B"], required=True)
    parser.add_argument("--with-ar", action="store_true")
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--model-path", default="models/MOSS-Speech")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    assets = Path(args.assets_dir)
    mp_ctx = mp.get_context("spawn")
    preproc_in = mp_ctx.Queue()
    enc_in = mp_ctx.Queue()  # layout B only
    dec_in = mp_ctx.Queue()
    driver_q = mp_ctx.Queue()

    stats = {"assets": str(assets)}
    mgr = mp_ctx.Manager()
    stats_proxy = mgr.dict(stats)

    preproc_target = _preproc_worker
    preproc_args = (
        args.layout,
        args.codec_path,
        preproc_in,
        enc_in if args.layout == "B" else dec_in,
        stats_proxy,
    )
    procs = [mp_ctx.Process(target=preproc_target, args=preproc_args, name="preproc")]

    if args.layout == "B":
        procs.append(
            mp_ctx.Process(
                target=_encoder_worker_b,
                args=(args.codec_path, enc_in, dec_in, stats_proxy),
                name="encoder",
            )
        )
    procs.append(
        mp_ctx.Process(
            target=_decoder_worker,
            args=(args.codec_path, dec_in, driver_q, stats_proxy),
            name="decoder",
        )
    )

    ar_proc = None
    ar_stats_path = Path(args.out).with_suffix(".ar.json")
    if args.with_ar:
        ar_cmd = [
            sys.executable.replace(".venv-omni", ".venv-p0"),
            str(Path(__file__).parent / "ar_load.py"),
            "--model-path",
            args.model_path,
            "--out",
            str(ar_stats_path),
        ]
        ar_proc = subprocess.Popen(ar_cmd, env={**os.environ})

    for p in procs:
        p.start()
    time.sleep(45)  # model load + voice delivery warmup

    results: list = []
    for r in range(3):  # warmup rounds (not recorded)
        run_round(driver_q, preproc_in, 100 + r, assets, [])
    for r in range(args.rounds):
        run_round(driver_q, preproc_in, r, assets, results)

    for q_ in (preproc_in, enc_in, dec_in):
        q_.put({"type": "stop"})
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    summary = {
        "layout": args.layout,
        "with_ar": args.with_ar,
        "procs": dict(stats_proxy),
    }
    rounds_out = []
    for r in results:
        e2e = [
            x["dec_done_t"] - x["enqueue_t"]
            for x in r["reqs"]
            if "dec_done_t" in x and "enqueue_t" in x
        ]
        enc_wall = [
            x.get("stage_wall", {}).get("encode", 0.0)
            + x.get("stage_wall", {}).get("preproc", 0.0)
            for x in r["reqs"]
        ]
        dec_wall = [x.get("stage_wall", {}).get("decode", 0.0) for x in r["reqs"]]
        errs = [x["id"] for x in r["reqs"] if x.get("error")]
        rounds_out.append(
            {
                "round": r["round"],
                "wall_s": r["wall_s"],
                "n_done": r["n_done"],
                "errors": errs,
                "throughput_rps": round(r["n_done"] / r["wall_s"], 3),
                "e2e_p50_s": round(statistics.median(e2e), 3) if e2e else None,
                "e2e_p95_s": (
                    round(sorted(e2e)[max(int(0.95 * len(e2e)) - 1, 0)], 3)
                    if e2e
                    else None
                ),
                "preproc_enc_ms_p50": (
                    round(statistics.median(enc_wall) * 1000, 1) if enc_wall else None
                ),
                "decode_ms_p50": (
                    round(statistics.median(dec_wall) * 1000, 1) if dec_wall else None
                ),
            }
        )
    summary["rounds"] = rounds_out
    if ar_proc is not None:
        ar_proc.terminate()
        ar_proc.wait(timeout=30)
        if ar_stats_path.exists():
            summary["ar"] = json.loads(ar_stats_path.read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
