#!/usr/bin/env python3
"""Diagnostic teacher forcing for full-prefix/cached/reference and tail KV reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from validate_lifecycle import Harness
from validate_native import compare_logits


def metrics(actual: Any, expected: Any) -> Any:
    return {
        head: compare_logits(a.reshape(-1), b.reshape(-1))
        for head, a, b in zip(("text", "audio"), actual, expected)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"pass": False, "probes": {}, "diagnostic_only": True}
    h = None
    try:
        h = Harness(out)
        original_select = h.runner._select_row
        original_before = h.runner.before_decode
        perturbations = {}

        def select(data: Any, text: Any, audio: Any) -> Any:
            if data.req.rid in h.forced:
                # Diagnostic prefixes deliberately include EOS. Suppress only
                # EOS stopping in this driver so every prescribed row is read.
                data.req.sampling_params.ignore_eos = True
            return original_select(data, text, audio)

        h.runner._select_row = select

        def before(fb: Any, sb: Any, requests: Any, **kwargs: Any) -> Any:
            original_before(fb, sb, requests, **kwargs)
            for i, req in enumerate(requests):
                rid = req.data.req.rid
                if rid not in perturbations:
                    continue
                prefix, layers = perturbations[rid]
                if len(req.data.output_rows) != len(prefix) - 1:
                    continue
                runtime = h.scheduler.model_worker.model_runner
                positions = (
                    (prefix[:-1, 0] == 151667)
                    .nonzero()
                    .flatten()
                    .to(fb.req_pool_indices.device)
                )
                slots = runtime.req_to_token_pool.req_to_token[
                    fb.req_pool_indices[i], positions
                ]
                assert len(slots) > 0
                for layer in layers:
                    runtime.token_to_kv_pool.get_key_buffer(layer)[slots] = 0
                report.setdefault("perturbed_slots", {})[rid] = len(slots)

        h.runner.before_decode = before
        all_pass = True
        for path in sorted(Path("artifacts/p3/reference").glob("probe_*.pt")):
            blob = torch.load(path, map_location="cpu", weights_only=True)
            prefix = blob["prefix"][0]
            n = len(prefix)
            name = path.stem
            fresh = name + "-fresh"
            h.captures[fresh] = {0}
            h.submit(fresh, rows=prefix.tolist(), limit=1, temperature=0)
            h.wait([fresh])
            cached = name + "-cached"
            h.captures[cached] = {n - 1}
            h.forced[cached] = prefix[1:]
            h.submit(cached, rows=prefix[:1].tolist(), limit=n, temperature=0)
            h.wait([cached])
            fresh_logits, cached_logits = h.raw[(fresh, 0)], h.raw[(cached, n - 1)]
            result = {
                "fresh_reference": metrics(fresh_logits, blob["fresh"]),
                "cached_reference": metrics(cached_logits, blob["cached"]),
                "cached_fresh": metrics(cached_logits, fresh_logits),
                "prefix_length": n,
            }
            result["pass"] = all(
                m["pass"]
                for key in ("fresh_reference", "cached_reference", "cached_fresh")
                for m in result[key].values()
            )
            all_pass &= result["pass"]
            report["probes"][name] = result
            torch.save(
                {"fresh": fresh_logits, "cached": cached_logits}, out / (name + ".pt")
            )
            print(name, result, flush=True)
            if name == "probe_t2s_cn_audio_eosp_30":
                deltas = {}
                for target, layers in [
                    ("text", range(32, 36)),
                    ("audio", range(36, 40)),
                ]:
                    rid = name + "-perturb-" + target
                    h.captures[rid] = {n - 1}
                    h.forced[rid] = prefix[1:]
                    perturbations[rid] = (prefix, layers)
                    h.submit(rid, rows=prefix[:1].tolist(), limit=n, temperature=0)
                    h.wait([rid])
                    deltas[target] = {
                        head: float((a - b).abs().max())
                        for head, a, b in zip(
                            ("text", "audio"), h.raw[(rid, n - 1)], cached_logits
                        )
                    }
                passed = (
                    deltas["text"]["text"] > 1
                    and deltas["text"]["audio"] == 0
                    and deltas["audio"]["audio"] > 1
                    and deltas["audio"]["text"] == 0
                )
                report["perturbation"] = {"deltas": deltas, "pass": passed}
                all_pass &= passed
        report["cleanup"] = h.idle()
        report["pass"] = all_pass and len(report["probes"]) == 4
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if h:
            h.close()
            report["loop_exited"] = True
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
