#!/usr/bin/env python3
"""Auditable native-engine parity: one submission, explicit forced/free runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

CASES = {
    "t2t_short": "t2t_short",
    "t2s_cn": "t2s_cn",
    "s2t_cn": "s2t_cn",
    "s2s_cn": "s2s_cn",
    "mixed_multiturn": "mixed_multiturn",
    "t2s_short_trans": "natural_transition/t2s_short_trans",
}


def compare_logits(native: Any, reference: Any) -> Any:
    """Frozen v1 per-head metrics, rejecting NaNs and nonfinite mismatches."""
    native, reference = native.float(), reference.float()
    if native.shape != reference.shape:
        return {
            "pass": False,
            "shape_mismatch": [list(native.shape), list(reference.shape)],
        }
    finite = torch.isfinite(reference)
    pattern = bool(
        torch.equal(torch.isfinite(native), finite)
        and torch.equal(torch.isposinf(native), torch.isposinf(reference))
        and torch.equal(torch.isneginf(native), torch.isneginf(reference))
        and not native.isnan().any()
        and not reference.isnan().any()
    )
    delta = (native[finite] - reference[finite]).abs()
    bad = delta > 1.0 + 0.02 * reference[finite].abs()
    frac = float(bad.float().mean()) if bad.numel() else 0.0
    max_abs = float(delta.max()) if delta.numel() else 0.0
    return {
        "pass": pattern and frac <= 1e-4 and max_abs <= 4.5,
        "nonfinite_match": pattern,
        "max_abs": max_abs,
        "exceed_fraction": frac,
        "exceed_count": int(bad.sum()),
    }


def capture_scores(text: Any, audio: Any, data: Any) -> Any:
    from sglang_omni.models.moss_speech import fsm

    t, a = text.detach().float().cpu(), audio.detach().float().cpu()
    raw = torch.cat([t, a])
    a = fsm.apply_audio_constraints(
        a.clone(), len(data.output_rows) + 1, data.params.min_new_tokens
    )
    masked = torch.cat([t, a])
    th, ah = fsm.channel_histories(data.prompt_rows, data.output_rows)
    scored = torch.cat(
        [
            fsm.repetition_penalty_scores(t, th, data.params.repetition_penalty),
            fsm.repetition_penalty_scores(a, ah, data.params.repetition_penalty),
        ]
    )
    return {"raw": raw, "masked": masked, "scored": scored}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/MOSS-Speech")
    ap.add_argument("--ref-dir", default="artifacts/p3/reference")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--supplement-dir")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument(
        "--modes", nargs="+", default=["forced", "free"], choices=["forced", "free"]
    )
    ap.add_argument("--attention-backend", default="torch_native")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "protocol": "v1",
        "cases": {},
        "run": str(out),
        "attention_backend": args.attention_backend,
        "commit": subprocess.check_output(
            ["git", "-C", "sglang-omni", "rev-parse", "HEAD"], text=True
        ).strip(),
        "submissions": [],
        "model": None,
        "pass": False,
    }
    diff = subprocess.check_output(["git", "-C", "sglang-omni", "diff", "HEAD"])
    (out / "working_tree.patch").write_bytes(diff)
    report["diff_sha256"] = hashlib.sha256(diff).hexdigest()
    report["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("sglang-omni/sglang_omni/models/moss_speech").rglob("*.py")
    }
    report["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    scheduler = None
    loop = None
    try:
        from sglang_omni.models.moss_speech.engine_builder import (
            MossSpeechEngineBuilder,
        )
        from sglang_omni.proto.request import StagePayload
        from sglang_omni.scheduling.messages import IncomingMessage

        builder = MossSpeechEngineBuilder()
        scheduler = builder.build(
            args.model_path,
            gpu_id=0,
            server_args_overrides={
                "attention_backend": args.attention_backend,
                "mem_fraction_static": 0.60,
                "max_running_requests": 4,
            },
        )
        runner = scheduler._model_runner
        report["model"] = type(runner.model).__name__
        original_select = runner._select_row
        active = {}

        def select(data: Any, text: Any, audio: Any) -> Any:
            state = active[data.req.rid]
            step = len(data.output_rows)
            captured = capture_scores(text, audio, data)
            prefix = torch.cat(
                [
                    data.prompt_rows,
                    torch.tensor(data.output_rows, dtype=torch.long).reshape(-1, 2),
                ]
            )
            torch.save(
                {
                    **captured,
                    "rid": data.req.rid,
                    "step": step,
                    "prefix_sha256": hashlib.sha256(
                        prefix.numpy().tobytes()
                    ).hexdigest(),
                },
                state["out"] / f"step_{step:04d}.pt",
            )
            chosen = original_select(data, text, audio)
            item = {"step": step, "mode": int(data.mode), "native_choice": list(chosen)}
            if state["mode"] == "forced":
                ref_path = state["ref"] / f"step_{step:04d}.pt"
                if not ref_path.exists() and args.supplement_dir:
                    supplement = Path(args.supplement_dir)
                    manifest = json.loads((supplement / "manifest.json").read_text())
                    assert manifest["pass"]
                    assert (
                        manifest["cases"][state["ref"].name]["grid_sha256"]
                        == hashlib.sha256(
                            (state["ref"] / "tokens_grid.pt").read_bytes()
                        ).hexdigest()
                    )
                    ref_path = supplement / state["ref"].name / ref_path.name
                ref = torch.load(ref_path, map_location="cpu", weights_only=True)
                item["metrics"] = {
                    head: compare_logits(
                        captured["raw"][lo:hi], ref["raw"].reshape(-1)[lo:hi]
                    )
                    for head, lo, hi in [("text", 0, 151680), ("audio", 151680, 168192)]
                }
            state["steps"].append(item)
            if state["mode"] == "forced":
                return tuple(state["grid"][step].tolist())
            return chosen

        runner._select_row = select
        loop = threading.Thread(target=scheduler.start, daemon=True)
        loop.start()
        for case in args.cases:
            ref_dir = Path(args.ref_dir) / CASES[case]
            canon = json.loads((ref_dir / "canonical_input.json").read_text())
            grid = torch.load(
                ref_dir / "tokens_grid.pt", map_location="cpu", weights_only=True
            )
            for mode in args.modes:
                rid = f"{out.name}-{case}-{mode}"
                case_out = out / f"{case}_{mode}"
                case_out.mkdir()
                state = {
                    "out": case_out,
                    "ref": ref_dir,
                    "grid": grid,
                    "steps": [],
                    "mode": mode,
                }
                active[rid] = state
                payload = StagePayload(
                    request_id=rid,
                    request=SimpleNamespace(request_id=rid),
                    data={
                        "output_modality": (
                            "audio"
                            if case in ("t2s_cn", "s2s_cn", "t2s_short_trans")
                            else "text"
                        ),
                        "input_grid": canon["input_ids"],
                        "attention_mask": canon["attention_mask"],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "repetition_penalty": 1.1,
                        "max_new_tokens": len(grid) if mode == "forced" else 200,
                        "effective_seed": 0,
                        "explicit_params": [
                            "temperature",
                            "top_p",
                            "top_k",
                            "repetition_penalty",
                        ],
                    },
                )
                assert rid not in report["submissions"]
                report["submissions"].append(rid)
                scheduler.inbox.put(
                    IncomingMessage(request_id=rid, type="new_request", data=payload)
                )
                deadline = time.monotonic() + args.timeout
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError(rid)
                    try:
                        result = scheduler.outbox.get(timeout=1)
                    except queue.Empty:
                        continue
                    if result.request_id != rid:
                        raise RuntimeError(
                            f"unexpected result {result.request_id}, expected {rid}"
                        )
                    if result.type == "error":
                        raise RuntimeError(repr(result.data))
                    if result.type == "result":
                        break
                result_state = result.data.data
                generated = torch.tensor(result_state["output_grid"], dtype=torch.long)
                torch.save(generated, case_out / "tokens_grid.pt")
                same_len = generated.shape == grid.shape
                exact = same_len and torch.equal(generated, grid)
                metrics_pass = (
                    all(
                        all(m["pass"] for m in step["metrics"].values())
                        for step in state["steps"]
                    )
                    if mode == "forced"
                    else None
                )
                summary = {
                    "mode": mode,
                    "generated_length": len(generated),
                    "reference_length": len(grid),
                    "finish_reason": result_state.get("finish_reason"),
                    "sample_calls": len(state["steps"]),
                    "grid_equal": exact,
                    "logits_pass": metrics_pass,
                    "steps": state["steps"],
                    "pass": (
                        metrics_pass and len(state["steps"]) == len(grid)
                        if mode == "forced"
                        else exact
                    ),
                }
                if same_len:
                    summary["equal_by_channel"] = (generated == grid).sum(0).tolist()
                report["cases"][f"{case}_{mode}"] = summary
                print(
                    case,
                    mode,
                    {k: v for k, v in summary.items() if k != "steps"},
                    flush=True,
                )
                del active[rid]
        report["pass"] = all(x["pass"] for x in report["cases"].values())
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if scheduler is not None:
            scheduler.stop()
        if loop is not None:
            loop.join(timeout=20)
            report["loop_exited"] = not loop.is_alive()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
