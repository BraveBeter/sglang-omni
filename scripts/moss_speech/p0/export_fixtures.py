#!/usr/bin/env python3
"""Golden-fixture export for MOSS-Speech parity (P0 / T0.8).

For 5 cases (t2t, t2s, s2t, s2s, mixed multi-turn) exports, under greedy
decoding with fixed seed and bf16:

  - canonical chat request (JSON conversation)
  - processor canonical input grid (text/audio channels + attention mask)
  - input-side codec codes (for audio user turns)
  - full generated token grid + per-step raw logits (both channels; first 64
    steps stored as fp32, full dump path recorded)
  - text output, audio codes (extracted from grid), audio wav + metadata
  - environment fingerprint

Re-run determinism gate: t2t_short is generated twice; logits max-abs diff and
token equality must hold (bf16 tolerance 1e-3 on fp32-cast logits).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import GenerationConfig  # noqa: E402

import run_reference as rr  # noqa: E402
from utils.interface import MIMOStopper  # noqa: E402

MODALITY_PAD = 151667
SOSP = 151646
EOSP = 16384


def extract_audio_codes(grid: torch.Tensor) -> List[int]:
    """grid: (L, 2) channels-last token grid (generate output). Returns codes
    between the first sosp/eosp bracket on the audio channel."""
    text_ch = grid[:, 0].tolist()
    audio_ch = grid[:, 1].tolist()
    if SOSP not in text_ch:
        return []
    start = text_ch.index(SOSP) + 1
    stop = audio_ch.index(EOSP) if EOSP in audio_ch else len(audio_ch)
    return [int(t) for t in audio_ch[start:stop]]


def export_case(
    engine: rr.Inference,
    out_dir: Path,
    case_id: str,
    task: str,
    conversation: List[Dict[str, Any]],
    decoder_audio_prompt: Optional[str],
    gen_cfg: GenerationConfig,
    seed: int,
    keep_steps: int = 64,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    modality = rr._task_output_modality(task)
    full_conv = [{"role": "system", "content": rr.system_prompt_for(task)}] + conversation
    inputs = engine.processor([full_conv], [modality])
    canon = {
        "input_ids": inputs["input_ids"][0].cpu().tolist(),  # (L, 2)
        "attention_mask": inputs["attention_mask"][0].cpu().tolist(),
    }

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = engine.model.generate(
        input_ids=inputs["input_ids"].to(engine.device),
        attention_mask=inputs["attention_mask"].to(engine.device),
        generation_config=GenerationConfig(
            **{k: getattr(gen_cfg, k) for k in ("do_sample", "repetition_penalty", "max_new_tokens", "min_new_tokens", "use_cache")}
        ),
        # NOTE: the remote generate() reads these from **kwargs, not from the
        # generation config; passing them inside the config takes the tensor
        # code path and crashes.
        return_dict_in_generate=True,
        output_logits=True,
        output_scores=True,
        stopping_criteria=[
            MIMOStopper(engine.processor.tokenizer.pad_token_id),
            MIMOStopper(engine.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")),
        ],
        streamer=rr._NoopStreamer(),
    )
    grid = out["sequences"][0].cpu()  # (L, 2)
    logits_steps = out["logits"]  # tuple of (text_logits, audio_logits), each (B, V)

    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "case_id": case_id,
        "task": task,
        "seed": seed,
        "conversation": conversation,
        "n_new_steps": int(grid.shape[0]),
        "n_logits_steps": len(logits_steps),
    }

    # logits: first keep_steps, fp32, both channels concatenated
    kept = []
    for st in logits_steps[:keep_steps]:
        kept.append(torch.cat([st[0][0], st[1][0]]).float().cpu())
    torch.save(torch.stack(kept) if kept else torch.zeros(0), case_dir / "logits_first_steps.pt")
    meta["logits_kept_steps"] = len(kept)
    meta["text_vocab"] = int(logits_steps[0][0].shape[-1]) if logits_steps else None
    meta["audio_vocab"] = int(logits_steps[0][1].shape[-1]) if logits_steps else None

    torch.save(grid, case_dir / "tokens_grid.pt")
    meta["audio_codes_extracted"] = extract_audio_codes(grid)[:20] + ["..."]

    # decode output (text or audio)
    audio, text = rr.decode_tokens(engine, grid.unsqueeze(0), task, decoder_audio_prompt, seed)
    if audio is not None:
        sr, wav = audio
        sf.write(case_dir / "audio.wav", wav.numpy() if hasattr(wav, "numpy") else wav, sr)
        meta["audio_meta"] = rr.audio_meta(audio)
        meta["audio_codes_full_len"] = len(extract_audio_codes(grid))
    if text is not None:
        (case_dir / "text.txt").write_text(text)
        meta["text"] = text

    # input-side codes for audio user turns
    in_codes = {}
    for i, turn in enumerate(conversation):
        c = turn.get("content")
        if isinstance(c, dict) and c.get("path"):
            codes = engine.processor.audio_codec.encode([c["path"]])[0]
            in_codes[f"turn{i}"] = {"path": c["path"], "n_codes": len(codes), "first10": codes[:10]}
    meta["input_audio_codes"] = in_codes

    (case_dir / "canonical_input.json").write_text(json.dumps(canon))
    (case_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/MOSS-Speech")
    parser.add_argument("--codec-path", default="models/MOSS-Speech-Codec")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    rr._install_torchaudio_load_shim()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_env.json").write_text(json.dumps(rr.env_fingerprint(argparse.Namespace(
        model_path=args.model_path, codec_path=args.codec_path, seed=args.seed, sampling="greedy",
        temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.1,
        max_new_tokens=args.max_new_tokens, min_new_tokens=0)), indent=1))

    engine = rr.Inference(args.model_path, codec_path=args.codec_path, device="cuda")
    gen_cfg = GenerationConfig(
        do_sample=False, repetition_penalty=1.1,
        max_new_tokens=args.max_new_tokens, min_new_tokens=0, use_cache=True,
    )
    assets = Path(args.assets_dir)
    cn = str(assets / "prompt-cn.wav")

    cases = [
        ("t2t_short", "text_instruct_text_response", [rr.user_text_turn("Introduce yourself in one sentence.")], None),
        ("t2s_cn", "text_instruct_speech_response", [rr.user_text_turn("用中文介绍一下上海的三到四个著名景点。")], cn),
        ("s2t_cn", "speech_instruct_text_response", [rr.user_audio_turn(cn)], None),
        ("s2s_cn", "speech_instruct_speech_response", [rr.user_audio_turn(cn)], cn),
        # Self-contained mixed multi-turn fixture: fixed literal assistant turn,
        # then an audio user turn with text output (cross-modality history).
        (
            "mixed_multiturn",
            "speech_instruct_text_response",
            [
                rr.user_text_turn("My name is Alice and I like hiking."),
                rr.assistant_text_turn("Nice to meet you, Alice! Hiking is a great way to enjoy nature."),
                rr.user_audio_turn(cn),
            ],
            None,
        ),
    ]
    summary = []
    for cid, task, conv, prompt in cases:
        print(f"[fixture] {cid}", flush=True)
        summary.append(export_case(engine, out_dir, cid, task, conv, prompt, gen_cfg, args.seed))

    # determinism gate on t2t_short
    m1 = summary[0]
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    g2 = export_case(engine, out_dir / "_rerun", "t2t_short_rerun", cases[0][1], cases[0][2], None, gen_cfg, args.seed)
    tok1 = torch.load(out_dir / "t2t_short" / "tokens_grid.pt")
    tok2 = torch.load(out_dir / "_rerun" / "t2t_short_rerun" / "tokens_grid.pt")
    l1 = torch.load(out_dir / "t2t_short" / "logits_first_steps.pt")
    l2 = torch.load(out_dir / "_rerun" / "t2t_short_rerun" / "logits_first_steps.pt")
    max_diff = float("nan")
    if l1.numel() and l1.shape == l2.shape:
        # audio logits carry -inf in the masked [16385:] region; -inf minus -inf
        # yields NaN even for identical runs — sanitize before diffing.
        l1f = torch.nan_to_num(l1, nan=0.0, posinf=1e30, neginf=-1e30)
        l2f = torch.nan_to_num(l2, nan=0.0, posinf=1e30, neginf=-1e30)
        max_diff = float((l1f - l2f).abs().max())
    det = {
        "tokens_equal": bool(torch.equal(tok1, tok2)),
        "logits_max_abs_diff_sanitized": max_diff,
        "pass": bool(torch.equal(tok1, tok2)) and (max_diff < 1e-3 if max_diff == max_diff else False),
    }
    (out_dir / "_determinism.json").write_text(json.dumps(det, indent=1))
    summary.append({"determinism": det})
    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(det, indent=1))
    print("FIXTURES DONE", flush=True)


if __name__ == "__main__":
    main()
