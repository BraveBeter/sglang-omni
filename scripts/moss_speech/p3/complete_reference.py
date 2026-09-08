#!/usr/bin/env python3
"""Supplement missing reference steps without changing any frozen expected."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from reference_baseline import greedy_generate, load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases", nargs="+", default=["s2t_cn", "mixed_multiturn"])
    args = p.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    model = load_model("models/MOSS-Speech", torch.bfloat16)
    report = {
        "pass": False,
        "cases": {},
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
    }
    try:
        for case in args.cases:
            source = Path("artifacts/p3/reference") / case
            canonical = json.loads((source / "canonical_input.json").read_text())
            inputs = {
                k: torch.tensor([v], dtype=torch.long) for k, v in canonical.items()
            }
            output, raw = greedy_generate(model, inputs, 200, 0, 1.1, 0, 151643, 151645)
            expected = torch.load(
                source / "tokens_grid.pt", map_location="cpu", weights_only=True
            )
            grid = output["sequences"][0].cpu()
            assert torch.equal(grid, expected), f"reference grid changed: {case}"
            target = outdir / case
            target.mkdir()
            count = 0
            for i in range(len(expected)):
                captured = {
                    "raw": torch.cat(raw[i]),
                    "masked": torch.cat(
                        [x[0].float().cpu() for x in output["logits"][i]]
                    ),
                    "scored": torch.cat(
                        [x[0].float().cpu() for x in output["scores"][i]]
                    ),
                }
                old = source / f"step_{i:04d}.pt"
                if old.exists():
                    prior = torch.load(old, map_location="cpu", weights_only=True)
                    assert all(
                        torch.equal(captured[k], prior[k]) for k in captured
                    ), f"existing capture changed: {old}"
                else:
                    torch.save(captured, target / old.name)
                    count += 1
            report["cases"][case] = {
                "grid_equal": True,
                "existing_captures_equal": True,
                "added_steps": count,
                "grid_sha256": hashlib.sha256(
                    (source / "tokens_grid.pt").read_bytes()
                ).hexdigest(),
            }
            print(case, report["cases"][case], flush=True)
        report["pass"] = True
    finally:
        (outdir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
