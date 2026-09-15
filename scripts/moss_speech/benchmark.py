# SPDX-License-Identifier: Apache-2.0
"""Measure native AR profiles against staged, independent dual-grid fixtures.

Run on an allocated GPU. Fixtures use the canonical_input.json/tokens_grid.pt
layout produced by scripts/moss_speech/p3/reference_baseline.py. This measures AR
latency and allocated memory; codec and HTTP timings are intentionally separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--prompt-case", choices=["t2s_cn", "s2t_cn"], default="t2s_cn")
    ap.add_argument(
        "--profile",
        choices=["baseline", "metadata", "mixed", "cache", "combined", "graph"],
        required=True,
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    builder = MossSpeechEngineBuilder(
        context_length=1024,
        decode_projection_graphs=args.profile in ("graph", "combined"),
        prefix_cache=args.profile in ("cache", "combined"),
        eager_cpu_metadata=args.profile in ("metadata", "combined"),
        mixed_modalities=args.profile in ("mixed", "combined"),
    )
    from sglang_omni.models.moss_speech import engine_builder

    source_root = Path(engine_builder.__file__).parent
    report = {
        "profile": args.profile,
        "prompt_case": args.prompt_case,
        "pass": False,
        "device": torch.cuda.get_device_name(),
        "runs": [],
        "source_root": str(source_root),
        "model_path": str(args.model_path.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "versions": {
            name: version(name) for name in ["torch", "sglang", "transformers"]
        },
        "source_sha256": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_root.rglob("*.py")
        },
        "reference_sha256": {
            str(p.relative_to(args.reference_dir)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for case in ["t2t_short", "t2s_cn", "s2t_cn", "s2s_cn"]
            for p in [
                args.reference_dir / case / "canonical_input.json",
                args.reference_dir / case / "tokens_grid.pt",
            ]
        },
    }

    scheduler = None
    loop = None
    try:
        scheduler = builder.build(
            str(args.model_path),
            gpu_id=0,
            server_args_overrides={
                "max_total_tokens": 4096,
                "max_running_requests": 4,
                "mem_fraction_static": 0.6,
            },
        )
        fixtures = {}
        for case in ["t2t_short", "t2s_cn", "s2t_cn", "s2s_cn"]:
            folder = args.reference_dir / case
            fixtures[case] = (
                json.loads((folder / "canonical_input.json").read_text()),
                torch.load(
                    folder / "tokens_grid.pt", map_location="cpu", weights_only=True
                ),
            )
        counters = Counter()
        hits = []
        first = {}
        steps = Counter()
        runner = scheduler._model_runner
        original_collect = runner._collect_step
        original_prefill = runner.custom_prefill_forward

        def collect(
            result: Any,
            forward_batch: Any,
            requests: list[Any],
            *,
            phase: str = "prefill",
        ) -> None:
            counters[(phase, len(requests))] += 1
            original_collect(result, forward_batch, requests, phase=phase)
            for request in requests:
                first.setdefault(request.request_id, time.perf_counter())
                steps[request.request_id] += 1

        def prefill(
            forward_batch: Any, schedule_batch: Any, requests: list[Any]
        ) -> Any:
            hits.extend(
                {
                    "rid": r.data.req.rid,
                    "cached": len(r.data.req.prefix_indices),
                    "extend": r.data.req.extend_range.length,
                }
                for r in requests
            )
            return original_prefill(forward_batch, schedule_batch, requests)

        runner._collect_step = collect
        runner.custom_prefill_forward = prefill
        loop = threading.Thread(target=scheduler.start, daemon=True)
        loop.start()

        def run(label: str, cases: list[str], *, abort: bool = False) -> None:
            counters.clear()
            hits.clear()
            pending = {}
            rids = []
            start = time.perf_counter()
            for i, case in enumerate(cases):
                rid = f"{out.name}-{label}-{i}"
                rids.append(rid)
                canon, grid = fixtures[case]
                pending[rid] = grid[:64]
                payload = StagePayload(
                    request_id=rid,
                    request=SimpleNamespace(request_id=rid),
                    data={
                        "output_modality": (
                            "audio" if case in ("t2s_cn", "s2s_cn") else "text"
                        ),
                        "input_grid": canon["input_ids"],
                        "attention_mask": canon["attention_mask"],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "repetition_penalty": 1.1,
                        "max_new_tokens": 64,
                        "effective_seed": 0,
                        "explicit_params": [
                            "temperature",
                            "top_p",
                            "top_k",
                            "repetition_penalty",
                        ],
                    },
                )
                scheduler.inbox.put(
                    IncomingMessage(request_id=rid, type="new_request", data=payload)
                )
            cancelled = False
            received = []
            while pending:
                assert time.perf_counter() - start < 180, label
                if abort and not cancelled and steps[rids[-1]] >= 4:
                    scheduler.abort(rids[-1])
                    scheduler.abort(rids[-1])
                    pending.pop(rids[-1])
                    cancelled = True
                try:
                    result = scheduler.outbox.get(timeout=0.02)
                except queue.Empty:
                    continue
                if result.request_id not in pending:
                    assert cancelled and result.request_id == rids[-1], result
                    continue
                if result.type == "error":
                    raise RuntimeError(repr(result.data))
                if result.type != "result":
                    continue
                grid = torch.tensor(result.data.data["output_grid"])
                expected = pending.pop(result.request_id)
                exact = torch.equal(grid, expected)
                received.append(
                    {
                        "rid": result.request_id,
                        "elapsed_s": time.perf_counter() - start,
                        "first_s": first[result.request_id] - start,
                        "grid_equal": exact,
                        "length": len(grid),
                    }
                )
                assert exact, (
                    label,
                    result.request_id,
                    (
                        (grid == expected).sum(0).tolist()
                        if grid.shape == expected.shape
                        else (grid.shape, expected.shape)
                    ),
                )
            elapsed = time.perf_counter() - start
            deadline = time.perf_counter() + 5
            while (
                builder.request_lifecycle.snapshot()["requests"]
                and time.perf_counter() < deadline
            ):
                time.sleep(0.01)
            assert builder.request_lifecycle.snapshot() == {
                "requests": 0,
                "active": 0,
                "pending": 0,
            }
            while True:
                try:
                    extra = scheduler.outbox.get_nowait()
                    assert cancelled and extra.request_id == rids[-1]
                except queue.Empty:
                    break
            item = {
                "label": label,
                "cases": cases,
                "elapsed_s": elapsed,
                "results": received,
                "batches": {f"{phase}_{bs}": n for (phase, bs), n in counters.items()},
                "cache_hits": list(hits),
                "cancelled": cancelled,
                "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            report["runs"].append(item)
            (out / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(item), flush=True)

        for batch_size in [1, 2, 4]:
            cases = [args.prompt_case] * batch_size
            run(f"warmup-{batch_size}", cases)
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(3):
                run(f"batch-{batch_size}-{repeat}", cases)
        mixed = ["t2t_short", "t2s_cn", "s2t_cn", "s2s_cn"]
        run("warmup-mixed", mixed)
        for repeat in range(3):
            run(f"mixed-{repeat}", mixed)
        run("abort", mixed, abort=True)
        run("recovery", mixed)
        if args.profile in ("cache", "combined"):
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            tree = scheduler.tree_cache
            assert tree.protected_size() == 0
            before = tree.total_size()
            eviction = tree.evict(EvictParams(num_tokens=before))
            assert tree.total_size() == 0 and eviction.num_tokens_evicted == before
            report["eviction"] = {
                "before_tokens": before,
                "evicted_tokens": eviction.num_tokens_evicted,
            }
            run("after-eviction", mixed)
            assert all(hit["cached"] == 0 for hit in report["runs"][-1]["cache_hits"])
        report["pass"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if scheduler:
            model = scheduler._model_runner.model
            report["graph_replays"] = [
                {
                    str(key): graph.replay_count
                    for key, graph in getattr(layer, "_projection_graphs", {}).items()
                }
                for layer in list(model.layers)
                + list(model.text_block)
                + list(model.audio_block)
            ]
            scheduler.stop()
        if loop:
            loop.join(timeout=20)
            report["loop_exited"] = not loop.is_alive()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
