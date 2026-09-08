#!/usr/bin/env python3
"""Observe per-layer activations for newly detected first-step parity failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["reference", "native"], required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--cases", nargs="+", default=["en_00_s2t", "zh_00_t2t", "en_00_t2t"]
    )
    p.add_argument("--logits-to-keep", type=int, default=None)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = {r["id"]: r for r in json.loads(args.reference.read_text())["results"]}
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    harness = None
    if args.backend == "reference":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "models/MOSS-Speech",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        layers = [
            *model.model.shared_block.layers,
            *model.model.text_block.layers,
            *model.model.audio_block.layers,
        ]
        norms = [model.model.text_norm, model.model.audio_norm]
    else:
        from validate_lifecycle import Harness

        harness = Harness(args.out_dir)
        model = harness.runner.model
        layers = [*model.layers, *model.text_block, *model.audio_block]
        norms = [model.text_norm, model.audio_norm]
    caps: dict[str, Any] = {}

    def snapshot(key: str, pair: bool = False) -> Any:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            value = output
            if isinstance(output, tuple):
                value = output[0] + output[1] if pair else output[0]
            caps[key] = value.detach().reshape(-1, value.shape[-1]).cpu()

        return hook

    hooks = [
        layer.register_forward_hook(
            snapshot(f"layer_{i:02d}", args.backend == "native")
        )
        for i, layer in enumerate(layers)
    ]
    if args.backend == "reference":
        for name in ("text_lm_head", "audio_lm_head"):

            def head_input(module: Any, inputs: Any, key: str = name) -> None:
                caps[key + "_input_shape"] = list(inputs[0].shape)

            hooks.append(getattr(model, name).register_forward_pre_hook(head_input))
    hooks += [
        norm.register_forward_hook(snapshot(f"norm_{i}"))
        for i, norm in enumerate(norms)
    ]
    try:
        for rid in args.cases:
            caps.clear()
            prefix = rows[rid]["input_grid"]
            if harness:
                harness.captures[rid] = {0}
                harness.submit(
                    rid, rows=prefix, limit=1, temperature=0, seed=0, top_p=1, top_k=-1
                )
                grid = harness.wait([rid])[0].tolist()
                raw = harness.raw[(rid, 0)]
            else:
                import run_reference as rr
                from transformers import GenerationConfig
                from utils.interface import MIMOStopper

                captured = {}

                def capture_output(module: Any, inputs: Any, output: Any) -> None:
                    captured["raw"] = tuple(
                        t[0, -1].detach().float().cpu() for t in output.logits_all
                    )

                head_hook = model.register_forward_hook(capture_output)
                with torch.inference_mode():
                    grid = model.generate(
                        **(
                            {"logits_to_keep": args.logits_to_keep}
                            if args.logits_to_keep is not None
                            else {}
                        ),
                        input_ids=torch.tensor([prefix], device="cuda"),
                        attention_mask=torch.ones(
                            1, len(prefix), dtype=torch.long, device="cuda"
                        ),
                        generation_config=GenerationConfig(
                            do_sample=False,
                            repetition_penalty=1.1,
                            max_new_tokens=1,
                            min_new_tokens=0,
                            use_cache=True,
                        ),
                        stopping_criteria=[MIMOStopper(151643), MIMOStopper(151645)],
                        streamer=rr._NoopStreamer(),
                    )[0].tolist()
                head_hook.remove()
                raw = captured["raw"]
            torch.save(
                {**caps, "raw": raw, "grid": grid, "prefix": prefix},
                args.out_dir / f"{rid}.pt",
            )
            print(rid, len(prefix), grid, flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        if harness:
            harness.close()


if __name__ == "__main__":
    main()
