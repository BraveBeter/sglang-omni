#!/usr/bin/env python3
"""Reference-AR load process for the T1.4 placement A/B experiment.

Loads the AR model ONLY (bf16, torch_dtype explicit — corrects the P0 fp32
accident, no codec) under `.venv-p0` and runs greedy generate loops on a P0
canonical t2t input, logging busy fraction and VRAM. Stopped via SIGTERM by
the A/B driver; final stats are flushed by the driver-side atexit hook.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "p0"))
sys.path.insert(0, "/remote-home1/xrluan/SGLang_experiments/repos/MOSS-Speech")
sys.path.insert(0, "/remote-home1/xrluan/SGLang_experiments/repos/MOSS-Speech/Matcha-TTS")
import run_reference as rr  # noqa: E402

rr._install_torchaudio_load_shim()

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer, GenerationConfig  # noqa: E402
from utils.interface import MIMOStopper  # noqa: E402

_running = True


def _stop(*_a) -> None:
    global _running
    _running = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    stats = {
        "load_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
        "busy_fraction": None,
        "n_generate_iters": 0,
    }
    atexit.register(lambda: (Path(args.out).write_text(json.dumps(stats, indent=1)), print(json.dumps(stats))))

    ids = tok(
        "<|im_start|>user\nIntroduce yourself in one sentence.<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt",
    ).input_ids.to("cuda")
    L = ids.shape[1]
    grid = torch.zeros(1, L, 2, dtype=torch.long, device="cuda")
    grid[:, :, 0] = ids
    grid[:, :, 1] = 512  # audio_pad on the ignored channel
    attn = torch.ones(1, L, dtype=torch.long, device="cuda")
    gen_cfg = GenerationConfig(do_sample=False, repetition_penalty=1.1, max_new_tokens=64, min_new_tokens=0, use_cache=True)
    stoppers = [MIMOStopper(tok.pad_token_id), MIMOStopper(tok.convert_tokens_to_ids("<|im_end|>"))]

    busy = 0.0
    t_start = time.time()
    while _running:
        torch.manual_seed(0)
        t0 = time.perf_counter()
        out = model.generate(
            input_ids=grid, attention_mask=attn, generation_config=gen_cfg,
            stopping_criteria=stoppers, streamer=rr._NoopStreamer(),
        )
        busy += time.perf_counter() - t0
        stats["n_generate_iters"] += 1
        stats["busy_fraction"] = round(busy / (time.time() - t_start), 3)
        stats["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        stats["reserved_gib"] = round(torch.cuda.memory_reserved() / 2**30, 2)


if __name__ == "__main__":
    main()
