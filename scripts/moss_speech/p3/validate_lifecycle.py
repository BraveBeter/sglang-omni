#!/usr/bin/env python3
"""Real scheduler batches, seeded sampling, cancellation and KV reuse checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


class Harness:
    def __init__(self, out: Path) -> None:
        from sglang_omni.models.moss_speech.engine_builder import (
            MossSpeechEngineBuilder,
        )

        self.out = out
        self.builder = MossSpeechEngineBuilder()
        self.scheduler = self.builder.build(
            "models/MOSS-Speech",
            gpu_id=0,
            server_args_overrides={
                "mem_fraction_static": 0.60,
                "max_running_requests": 4,
            },
        )
        self.runner = self.scheduler._model_runner
        self.steps = {}
        self.results = {}
        self.batches = []
        self.captures = {}
        self.forced = {}
        self.raw = {}
        self.errors = []
        original_collect = self.runner._collect_step
        original_select = self.runner._select_row

        def collect(
            result: Any, fb: Any, requests: Any, *, phase: Any = "prefill"
        ) -> Any:
            self.batches.append(
                {"phase": phase, "rids": [r.data.req.rid for r in requests]}
            )
            return original_collect(result, fb, requests, phase=phase)

        def select(data: Any, text: Any, audio: Any) -> Any:
            rid = data.req.rid
            step = len(data.output_rows)
            self.steps[rid] = step + 1
            if rid in self.captures and step in self.captures[rid]:
                self.raw[(rid, step)] = (
                    text.detach().float().cpu(),
                    audio.detach().float().cpu(),
                )
            if rid in self.forced and step < len(self.forced[rid]):
                return tuple(self.forced[rid][step].tolist())
            return original_select(data, text, audio)

        self.slots = {}
        original_before = self.runner.before_decode

        def before(fb: Any, sb: Any, requests: Any, **kwargs: Any) -> Any:
            original_before(fb, sb, requests, **kwargs)
            runtime = self.scheduler.model_worker.model_runner
            for i, req in enumerate(requests):
                rid = req.data.req.rid
                if rid.startswith("reuse-") and rid not in self.slots:
                    self.slots[rid] = (
                        runtime.req_to_token_pool.req_to_token[
                            fb.req_pool_indices[i], : fb.seq_lens[i]
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                    )

        self.runner.before_decode = before
        self.runner._collect_step = collect
        self.runner._select_row = select
        self.loop = threading.Thread(target=self.scheduler.start, daemon=True)
        self.loop.start()
        self.initial_free = int(
            self.scheduler.token_to_kv_pool_allocator.available_size()
        )

    def submit(
        self,
        rid: Any,
        case: Any = "t2t_short",
        *,
        seed: Any = 7,
        limit: Any = 24,
        temperature: Any = 0.65,
        top_k: Any = 20,
        top_p: Any = 0.9,
        penalty: Any = 1.1,
        rows: Any = None,
        padding: Any = 0,
        modality: Any = None,
    ) -> Any:
        from sglang_omni.proto.request import StagePayload
        from sglang_omni.scheduling.messages import IncomingMessage

        canon = json.loads(
            (Path("artifacts/p3/reference") / case / "canonical_input.json").read_text()
        )
        rows = canon["input_ids"] if rows is None else rows
        if len(rows) == 1 and isinstance(rows[0][0], list):
            rows = rows[0]
        mask = [0] * padding + [1] * len(rows)
        rows = [[151643, 0]] * padding + rows
        data = {
            "input_grid": rows,
            "attention_mask": mask,
            "output_modality": modality or ("audio" if case == "t2s_cn" else "text"),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": penalty,
            "max_new_tokens": limit,
            "effective_seed": seed,
            "explicit_params": ["temperature", "top_p", "top_k", "repetition_penalty"],
        }
        payload = StagePayload(
            request_id=rid, request=SimpleNamespace(request_id=rid), data=data
        )
        self.scheduler.inbox.put(
            IncomingMessage(request_id=rid, type="new_request", data=payload)
        )

    def wait(self, rids: Any, timeout: Any = 300) -> Any:
        deadline = time.monotonic() + timeout
        while not all(r in self.results for r in rids):
            if time.monotonic() > deadline:
                raise TimeoutError(f"waiting for {rids}; steps={self.steps}")
            try:
                result = self.scheduler.outbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if result.type == "error":
                self.errors.append(
                    {"rid": result.request_id, "error": repr(result.data)}
                )
                raise RuntimeError(self.errors[-1])
            if result.type == "result":
                assert result.request_id not in self.results
                self.results[result.request_id] = result.data.data
        return [torch.tensor(self.results[r]["output_grid"]) for r in rids]

    def idle(self, timeout: Any = 30) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.builder.request_lifecycle.snapshot()
            free = int(self.scheduler.token_to_kv_pool_allocator.available_size())
            if not any(snapshot.values()) and free == self.initial_free:
                assert not self.runner.model._dual_logits_by_rid
                return {
                    "lifecycle": snapshot,
                    "free_slots": free,
                    "initial_free_slots": self.initial_free,
                }
            time.sleep(0.02)
        raise AssertionError(
            f"resources not reclaimed: {snapshot}, free={free}/{self.initial_free}"
        )

    def close(self) -> None:
        self.scheduler.stop()
        self.loop.join(20)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        assert not self.loop.is_alive()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "pass": False,
        "checks": {},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("sglang-omni/sglang_omni/models/moss_speech").glob("*.py")
        },
    }
    harness = None
    try:
        harness = Harness(out)
        specs = {
            "a": {"seed": 7, "limit": 24},
            "b": {
                "seed": 19,
                "limit": 16,
                "temperature": 0.85,
                "top_k": 12,
                "top_p": 0.8,
                "penalty": 1.2,
            },
        }
        for case in ["t2t_short", "t2s_cn"]:
            singles = {}
            for name, spec in specs.items():
                rid = f"{case}-single-{name}"
                harness.submit(rid, case, **spec)
                singles[name] = harness.wait([rid])[0]
            for order in [("a", "b"), ("b", "a")]:
                rids = [f"{case}-batch-{order[0]}-{name}" for name in order]
                for rid, name in zip(rids, order):
                    harness.submit(rid, case, **specs[name])
                grids = harness.wait(rids)
                for rid, name, grid in zip(rids, order, grids):
                    equal = torch.equal(grid, singles[name])
                    report["checks"][rid] = equal
                    torch.save(
                        {"single": singles[name], "batch": grid}, out / f"{rid}.pt"
                    )
            rid = f"{case}-padded"
            harness.submit(rid, case, padding=5, **specs["a"])
            report["checks"][rid] = torch.equal(harness.wait([rid])[0], singles["a"])
            # Actual decode cancellation, verified after at least four samples.
            survivor, victim = f"{case}-survivor", f"{case}-abort"
            harness.submit(victim, case, seed=31, limit=100)
            harness.submit(survivor, case, **specs["a"])
            deadline = time.monotonic() + 60
            while harness.steps.get(victim, 0) < 4:
                if time.monotonic() > deadline:
                    raise TimeoutError("abort never reached decode")
                time.sleep(0.005)
            report[f"{case}_abort_step"] = harness.steps[victim]
            harness.scheduler.abort(victim)
            harness.scheduler.abort(victim)
            report["checks"][survivor] = torch.equal(
                harness.wait([survivor])[0], singles["a"]
            )
            report[f"{case}_cleanup"] = harness.idle()
            recovery = f"{case}-recovery"
            harness.submit(recovery, case, **specs["a"])
            report["checks"][recovery] = torch.equal(
                harness.wait([recovery])[0], singles["a"]
            )
        # Exact greedy has stricter tie sensitivity than non-greedy samples.
        greedy = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "limit": 31, "seed": 0}
        for rid in ["greedy-single", "greedy-batch-1", "greedy-batch-2"]:
            harness.captures[rid] = set(range(31))
        harness.submit("greedy-single", **greedy)
        single = harness.wait(["greedy-single"])[0]
        harness.submit("greedy-batch-1", **greedy)
        harness.submit("greedy-batch-2", **greedy)
        generated = harness.wait(["greedy-batch-1", "greedy-batch-2"])
        for rid, grid in zip(["greedy-batch-1", "greedy-batch-2"], generated):
            report["checks"][rid] = torch.equal(single, grid)
        torch.save(harness.raw, out / "greedy_raw.pt")
        report["checks"]["greedy_batch_raw_exact"] = all(
            torch.equal(a, b)
            for rid in ["greedy-batch-1", "greedy-batch-2"]
            for step in range(31)
            for a, b in zip(
                harness.raw[("greedy-single", step)], harness.raw[(rid, step)]
            )
        )
        harness.submit("seed-only-7", seed=7)
        seed7 = harness.wait(["seed-only-7"])[0]
        harness.submit("seed-only-19", seed=19)
        seed19 = harness.wait(["seed-only-19"])[0]
        report["checks"]["seed_only_changes_sample"] = not torch.equal(seed7, seed19)
        # Exercise the actual GPU sampler's mask and top-k at a nonzero temperature.
        from sglang_omni.models.moss_speech import fsm

        t = torch.zeros(fsm.TEXT_VOCAB, device="cuda")
        t[42] = 20
        a = torch.zeros(fsm.AUDIO_VOCAB, device="cuda")
        a[73] = 20
        a[16385] = 1000
        params = fsm.MossSamplingParams(
            do_sample=True, temperature=0.7, top_k=1, top_p=0.9, repetition_penalty=1.0
        )
        token = fsm.sample_row(
            t,
            a,
            params,
            torch.empty(0, dtype=torch.long, device="cuda"),
            torch.empty(0, dtype=torch.long, device="cuda"),
            1,
            fsm.MODE_TEXT,
            torch.Generator(device="cuda").manual_seed(0),
        )
        report["checks"]["gpu_sampler_mask_topk"] = tuple(token) == (42, 73)
        # Reuse the SAME physical slots, retaining the previous request's KV.
        # The allocator reset only reorders its free list, after all owners
        # have released their slots. It never clears the KV tensor contents.
        harness.idle()
        allocator = harness.scheduler.token_to_kv_pool_allocator
        allocator.clear()
        harness.submit("reuse-old-audio", "t2s_cn", limit=24)
        harness.wait(["reuse-old-audio"])
        harness.idle()
        runtime = harness.scheduler.model_worker.model_runner
        kv = runtime.token_to_kv_pool
        stale = kv.get_key_buffer(0)[harness.slots["reuse-old-audio"]].clone()
        assert stale.count_nonzero().item() > 0
        allocator.clear()
        report["checks"]["allocator_reset_keeps_stale_kv"] = torch.equal(
            stale, kv.get_key_buffer(0)[harness.slots["reuse-old-audio"]]
        )
        harness.submit("reuse-new-text", **greedy)
        reused = harness.wait(["reuse-new-text"])[0]
        overlap = sorted(
            set(harness.slots["reuse-old-audio"]) & set(harness.slots["reuse-new-text"])
        )
        report["slot_reuse"] = {
            "old_slots": harness.slots["reuse-old-audio"],
            "new_slots": harness.slots["reuse-new-text"],
            "overlap": overlap,
        }
        report["checks"]["actual_slot_reuse"] = bool(overlap)
        report["checks"]["reused_slots_no_old_history"] = torch.equal(reused, single)
        buffers = list(kv.k_buffer) + list(kv.v_buffer)
        per_slot_bytes = sum(b[0].numel() * b.element_size() for b in buffers)
        report["kv_structure"] = {
            "layers": harness.runner.model.attention_layer_ids(),
            "dtype": str(buffers[0].dtype),
            "per_slot_bytes": per_slot_bytes,
            "allocator_slots": int(allocator.size),
            "physical_slots_with_padding": int(buffers[0].shape[0]),
            "pool_bytes": sum(b.numel() * b.element_size() for b in buffers),
            "all_buffers_distinct": len({b.data_ptr() for b in buffers}) == 80,
            "weight_load": harness.runner.model._weight_load_report,
        }
        report["checks"]["kv_40_layers_160kib"] = (
            harness.runner.model.attention_layer_ids() == list(range(40))
            and per_slot_bytes == 163840
            and buffers[0].dtype == torch.bfloat16
        )
        report["checks"]["kv_no_alias_capacity"] = report["kv_structure"][
            "all_buffers_distinct"
        ] and all(b.shape[0] == allocator.size + 1 for b in buffers)
        report["checks"]["weight_coverage_446"] = (
            report["kv_structure"]["weight_load"]["consumed_sources"] == 446
        )
        # Mixed arrivals exercise DeferredAdmission. Cancel one queued modality.
        harness.submit("mixed-text", limit=24)
        harness.submit("mixed-audio-cancelled", "t2s_cn", limit=24)
        harness.submit("mixed-audio", "t2s_cn", limit=24)
        harness.scheduler.abort("mixed-audio-cancelled")
        harness.wait(["mixed-text", "mixed-audio"])
        report["cleanup"] = harness.idle()
        report["checks"]["cancelled_no_result"] = (
            "mixed-audio-cancelled" not in harness.results
        )
        report["checks"]["real_decode_batch"] = any(
            b["phase"] == "decode" and len(b["rids"]) >= 2 for b in harness.batches
        )
        report["checks"]["no_mixed_output_batch"] = all(
            not ("mixed-text" in b["rids"] and "mixed-audio" in b["rids"])
            for b in harness.batches
        )
        report["pass"] = all(report["checks"].values())
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if harness:
            report["batches"] = harness.batches
            report["steps"] = harness.steps
            harness.close()
            report["loop_exited"] = True
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
