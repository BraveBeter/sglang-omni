#!/usr/bin/env python3
"""Generate a new same-precision HF reference on the pre-registered manifest."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from quality import build_cases, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    cases = build_cases(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {"pass": False, "manifest_sha256": sha256(args.manifest), "results": []}
    try:
        import run_reference as rr
        from transformers import AutoModel, AutoProcessor, GenerationConfig
        from utils.interface import MIMOStopper

        rr._install_torchaudio_load_shim()
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            codec_path=args.codec_path,
            device="cuda",
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
        report["gpu"] = torch.cuda.get_device_name(0)
        report["precision"] = dict(
            ar="bfloat16", matmul_tf32=False, cudnn_tf32=True, logits_to_keep=0
        )
        for case in cases:
            row = {"id": case["id"], "mode": case["mode"], "lang": case["lang"]}
            try:
                start = time.monotonic()
                inputs = processor([case["hf_messages"]], case["body"]["modalities"])
                canonical = inputs["input_ids"][0].tolist()
                assert inputs["input_ids"].shape == (1, len(canonical), 2)
                assert len(canonical) <= 512
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                with torch.inference_mode():
                    output = model.generate(
                        logits_to_keep=0,
                        input_ids=inputs["input_ids"].to("cuda"),
                        attention_mask=inputs["attention_mask"].to("cuda"),
                        generation_config=GenerationConfig(
                            do_sample=False,
                            repetition_penalty=1.1,
                            max_new_tokens=512,
                            min_new_tokens=0,
                            use_cache=True,
                        ),
                        stopping_criteria=[MIMOStopper(151643), MIMOStopper(151645)],
                        streamer=rr._NoopStreamer(),
                    )
                grid = output[0].cpu().tolist()
                assert 0 < len(grid) <= 512 and all(len(x) == 2 for x in grid)
                text = (
                    processor.tokenizer.decode(
                        [x[0] for x in grid[:-1]], skip_special_tokens=True
                    )
                    .replace("<|empty|>", ".")
                    .replace("<|end_empty|>", ":")
                )
                row.update(
                    input_grid=canonical,
                    grid=grid,
                    text="" if case["mode"].endswith("s") else text,
                    finish_reason=(
                        "stop" if grid[-1][0] in (151643, 151645) else "length"
                    ),
                    seconds=time.monotonic() - start,
                )
            except Exception as exc:
                row["error"] = repr(exc)
            report["results"].append(row)
            (args.out_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                case["id"],
                len(row.get("grid", [])),
                row.get("error", row.get("finish_reason")),
                flush=True,
            )
        report["pass"] = len(report["results"]) == len(cases) and all(
            not x.get("error") for x in report["results"]
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
