#!/usr/bin/env python3
"""P3 / T3.4: engine-level parity through the FORMAL path.

Builds the complete engine with MossSpeechEngineBuilder.build() (common
bootstrap + arch override + adapters + MossSpeechModelRunner), submits
canonical requests through the OmniScheduler inbox, and compares the
generated dual-channel grids against the T3.1 BF16 reference (greedy).
This is the decisive test of the hand-built-ForwardBatch detour: the
engine constructs every scheduler contract itself.
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import torch  # noqa: E402

import faulthandler  # noqa: E402
import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
faulthandler.dump_traceback_later(420, exit=True)  # hang diagnosis


def build_state(case_dir: Path, modality: str) -> Dict[str, Any]:
    canon = json.loads((case_dir / "canonical_input.json").read_text())
    return {
        "output_modality": modality,
        "input_grid": canon["input_ids"],
        "attention_mask": canon["attention_mask"],
        "prompt_grid_len": len(canon["input_ids"]),
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.1,
        "effective_seed": 0,
        "explicit_params": [],
        "turns": [],
    }


def submit_case(scheduler, case_dir: Path, modality: str) -> str:
    """Queue the request BEFORE start(): OmniScheduler.start() runs the event
    loop synchronously on the calling thread (the stage runtime dedicates a
    worker to it), so the driver starts it in a background thread."""
    from sglang_omni.proto.request import StagePayload
    from sglang_omni.scheduling.messages import IncomingMessage

    rid = f"parity-{case_dir.name}"
    payload = StagePayload(
        request_id=rid,
        request=SimpleNamespace(request_id=rid),
        data=build_state(case_dir, modality),
    )
    scheduler.inbox.put(IncomingMessage(request_id=rid, type="new_request", data=payload))
    return rid


def run_case(scheduler, case_dir: Path, modality: str, timeout_s: float):
    rid = submit_case(scheduler, case_dir, modality)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = scheduler.outbox.get(timeout=1.0)
        except queue.Empty:
            continue
        if out.request_id != rid:
            continue
        if out.type == "error":
            import traceback as _tbe
            err = out.data
            print(f"[drv-error] {rid}: {err!r}", flush=True)
            if isinstance(err, BaseException) and err.__traceback__ is not None:
                _tbe.print_exception(type(err), err, err.__traceback__)
            raise RuntimeError(f"engine error for {rid}: {err}")
        if out.type == "result":
            return out.data
    raise TimeoutError(f"no result for {rid} within {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/MOSS-Speech")
    ap.add_argument("--ref-dir", default="artifacts/p3/reference")
    ap.add_argument("--out", default="artifacts/p3/engine_parity.json")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--mem-fraction", type=float, default=0.72)
    args = ap.parse_args()

    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
    from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig  # noqa: F401

    builder = MossSpeechEngineBuilder()
    overrides = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(overrides)
    overrides["mem_fraction_static"] = args.mem_fraction
    t0 = time.time()
    print("[boot] building engine (common bootstrap)", flush=True)
    scheduler = builder.build(args.model_path, gpu_id=0, server_args_overrides=overrides)
    boot_s = time.time() - t0
    print(f"[boot] done in {boot_s:.1f}s", flush=True)
    # pre-queue both requests, then run the event loop on a worker thread
    import os as _os3

    only = _os3.environ.get("MOSS_ONLY_CASE")
    cases = [("t2t_short", "text"), ("t2s_short_trans", "audio")]
    if only:
        cases = [c for c in cases if c[0] == only]
    rids = {}
    if os.environ.get("MOSS_SERIAL"):
        # one request at a time: isolates mixed-batch prefill effects
        pass
    else:
        for case, modality in cases:
            case_dir = Path(args.ref_dir) / ("natural_transition/t2s_short_trans" if case == "t2s_short_trans" else case)
            rids[case] = submit_case(scheduler, case_dir, modality)
            print(f"[case] {case} queued", flush=True)

    import threading

    # surface the failure stack: _emit_request_error carries the exception
    # object with its own __traceback__ — print the origin stack directly
    import traceback as _tb

    _orig_emit = scheduler._emit_request_error

    def _emit_with_tb(request_id, error):
        print(f"[emit_error] {request_id}: {error!r}", flush=True)
        if isinstance(error, BaseException) and error.__traceback__ is not None:
            _tb.print_exception(type(error), error, error.__traceback__)
        return _orig_emit(request_id, error)

    scheduler._emit_request_error = _emit_with_tb

    for _name in ("run_batch", "_process_batch_result", "_handle_batch_failure",
                  "_admit_or_defer_built_request", "_resolve_pending_async"):
        _orig = getattr(scheduler, _name, None)
        if _orig is None:
            continue

        def _wrap(orig=_orig, name=_name):
            def inner(*a, **k):
                try:
                    return orig(*a, **k)
                except Exception:
                    print(f"[scheduler:{name}] raised:", flush=True)
                    _tb.print_exc()
                    raise
            return inner

        setattr(scheduler, _name, _wrap())

    loop = threading.Thread(target=scheduler.start, name="omni-loop", daemon=True)
    loop.start()
    report: Dict[str, Any] = {"boot_seconds": round(boot_s, 1)}

    results = {}
    for case, modality in cases:
        print(f"[case] {case} awaiting", flush=True)
        case_dir = Path(args.ref_dir) / ("natural_transition/t2s_short_trans" if case == "t2s_short_trans" else case)
        ref_grid = torch.load(case_dir / "tokens_grid.pt")
        t1 = time.time()
        try:
            data = run_case(scheduler, case_dir, modality, args.timeout)
            inner = getattr(data, "data", data)  # result is a StagePayload
            grid = inner.get("output_grid") if isinstance(inner, dict) else getattr(inner, "output_grid", None)
            print("[result] finish_reason:", inner.get("finish_reason") if isinstance(inner, dict) else getattr(inner, "finish_reason", None), flush=True)
            gen = [list(map(int, r)) for r in grid]
            print("[result] first rows:", gen[:3], flush=True)
            n = min(len(gen), len(ref_grid))
            equal_rows = sum(1 for i in range(n) if gen[i] == ref_grid[i].tolist())
            first_diff = next((i for i in range(n) if gen[i] != ref_grid[i].tolist()), None)
            Path("artifacts/p3").mkdir(exist_ok=True)
            torch.save(torch.tensor(gen), f"artifacts/p3/gen_{case}.pt")
            results[case] = {
                "gen_len": len(gen), "ref_len": len(ref_grid),
                "equal_rows": equal_rows, "first_diff": first_diff,
                "bitwise_equal": bool(len(gen) == len(ref_grid) and first_diff is None),
                "gen_seconds": round(time.time() - t1, 1),
            }
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb2
            print(f"[case-error] {case}: {exc!r}", flush=True)
            _tb2.print_exception(type(exc), exc, exc.__traceback__)
            ctx = exc.__context__
            while ctx is not None:
                print(f"[case-error-context] {ctx!r}", flush=True)
                _tb2.print_exception(type(ctx), ctx, ctx.__traceback__)
                ctx = ctx.__context__
            results[case] = {"error": repr(exc)}
    report["cases"] = results

    scheduler.stop()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("ENGINE PARITY DONE")


if __name__ == "__main__":
    main()
