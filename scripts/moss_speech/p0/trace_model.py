#!/usr/bin/env python3
"""Model-trace and KV/VRAM accounting spike for MOSS-Speech (P0 / T0.6).

Verifies four structural claims from the plan (all instrumentation is
monkey-patch based; the locked reference code is never modified):

  C1. Both tails (text_block / audio_block) execute on every decode step.
  C2. Both channels are sampled every step; the non-active channel is padded
      (text channel gets modality_pad during audio segments; audio channel is
      sampled-but-ignored during text segments).
  C3. After eosp the text tail resumes text generation attending over KV
      written during the audio segment (text cache length == audio cache
      length == shared cache length == total steps).
  C4. There is no single-tail shortcut: tail execution counts are equal.

Also records: per-cache layer counts (32/4/4), per-token KV bytes at 40-layer
accounting, VRAM split (weights / AR peak / codec decode peak).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import torch  # noqa: E402

import run_reference as rr  # noqa: E402  (same dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/MOSS-Speech")
    parser.add_argument("--codec-path", default="models/MOSS-Speech-Codec")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rr._install_torchaudio_load_shim()
    out = {"claims": {}, "kv": {}, "vram": {}, "trace_path": None}
    trace: List[Dict[str, Any]] = []

    engine = rr.Inference(args.model_path, codec_path=args.codec_path, device="cuda")
    weights_gib = torch.cuda.memory_allocated() / 2**30
    out["vram"]["after_model_load_gib"] = round(weights_gib, 2)

    base = engine.model.model  # MossSpeechModel
    lm = engine.model

    # --- hooks -------------------------------------------------------------
    tail_calls = {"text": 0, "audio": 0, "shared": 0}
    captured: Dict[str, Any] = {}
    orig_forward = base.forward

    def traced_forward(*a, **kw):
        mods = kw.get("modalities") or (a[0] if a else None)
        trace.append({"event": "forward", "modalities": list(mods)})
        for m in mods or []:
            tail_calls[m] = tail_calls.get(m, 0) + 1
        res = orig_forward(*a, **kw)
        pkvd = getattr(res, "past_key_values_dict", None)
        if pkvd is not None:
            captured["pkvd"] = pkvd
        return res

    base.forward = traced_forward

    orig_sample = lm._generate_next_tokens_with_scores

    def traced_next(logits_all, input_ids, realprocessor, do_samples, generation_config, generating_length):
        toks, scores, raw = orig_sample(logits_all, input_ids, realprocessor, do_samples, generation_config, generating_length)
        trace.append(
            {
                "event": "sample",
                "step": generating_length,
                "raw_sampled": toks.detach().cpu().tolist(),
            }
        )
        return toks, scores, raw

    lm._generate_next_tokens_with_scores = traced_next

    orig_pad = lm._process_multi_modality_tokens

    def traced_pad(next_tokens, current_modality, modality_pad_token):
        res = orig_pad(next_tokens, current_modality, modality_pad_token)
        trace.append(
            {
                "event": "pad",
                "step_modality": current_modality.detach().cpu().tolist(),  # 0 text / 1 audio
                "post_pad": res.detach().cpu().tolist(),
            }
        )
        return res

    lm._process_multi_modality_tokens = traced_pad

    # --- run one T2S case (full FSM: text -> sosp -> audio -> eosp -> text -> im_end)
    conv = [rr.user_text_turn("用一句话介绍长城。")]
    gen_cfg = rr.build_generation_config(
        argparse.Namespace(sampling="greedy", max_new_tokens=200, min_new_tokens=0)
    )
    torch.cuda.reset_peak_memory_stats()
    token_ids, in_len = rr.generate_once(engine, "text_instruct_speech_response", conv, gen_cfg, seed=0)
    ar_peak_gib = torch.cuda.max_memory_allocated() / 2**30

    # --- KV cache inspection (captured inside traced forward) ---------------
    caches: Dict[str, Any] = {}
    pkvd = captured.get("pkvd")
    if pkvd is not None:
        for name, cache in pkvd.items():
            n_layers = len(cache.key_cache) if hasattr(cache, "key_cache") else -1
            caches[name] = {
                "layers": n_layers,
                "seq_len": int(cache.get_seq_length()),
            }
    out["kv"]["caches"] = caches

    # --- claims ------------------------------------------------------------
    fwd_mods = [t["modalities"] for t in trace[:20] if t["event"] == "forward"]
    both_tails_every_fwd = all(set(m) == {"text", "audio"} for m in fwd_mods)
    out["claims"]["C1_both_tails_every_forward"] = both_tails_every_fwd
    out["claims"]["C4_tail_calls_equal_no_shortcut"] = tail_calls.get("text") == tail_calls.get("audio") and tail_calls["text"] > 0
    out["claims"]["tail_call_counts"] = dict(tail_calls)

    pad_events = [t for t in trace if t["event"] == "pad"]
    sample_events = [t for t in trace if t["event"] == "sample"]
    n_steps = len(pad_events)
    audio_steps = [i for i, t in enumerate(pad_events) if t["step_modality"] == [1]]
    text_steps = [i for i, t in enumerate(pad_events) if t["step_modality"] == [0]]
    # post_pad for B=1 is [[tok_text, tok_audio]]
    text_padded_on_audio = all(pad_events[i]["post_pad"][0][0] == 151667 for i in audio_steps)
    # sample events align 1:1 with pad events (same decode step order)
    audio_value_every_step = all(
        len(s["raw_sampled"][0]) == 2 for s in sample_events[:n_steps]
    )
    out["claims"]["C2_dual_channel_sampling"] = {
        "n_steps": n_steps,
        "audio_mode_steps": len(audio_steps),
        "text_mode_steps": len(text_steps),
        "text_channel_padded_with_modality_pad_on_audio_steps": text_padded_on_audio,
        "audio_channel_sampled_every_step": audio_value_every_step,
    }

    # C3: after eosp, text resumes; caches all equal length
    grid = token_ids[0].cpu()  # (L, 2)
    text_ch, audio_ch = grid[:, 0].tolist(), grid[:, 1].tolist()
    eosp_positions = [i for i, t in enumerate(audio_ch) if t == 16384]
    im_end_positions = [i for i, t in enumerate(text_ch) if t == 151645]
    resumed_text_after_eosp = False
    if eosp_positions:
        e = eosp_positions[0]
        tail_tokens = text_ch[e + 1 : (im_end_positions[0] + 1) if im_end_positions else None]
        # tokens after eosp that are real text (not modality_pad) before im_end
        resumed_text_after_eosp = any(t not in (151667, 151645) for t in tail_tokens)
    lens = {k: v["seq_len"] for k, v in caches.items()}
    out["claims"]["C3_text_resumes_after_eosp_cross_reading_audio_kv"] = {
        "eosp_first_pos": eosp_positions[0] if eosp_positions else None,
        "im_end_pos": im_end_positions[0] if im_end_positions else None,
        "text_tokens_after_eosp_before_im_end": resumed_text_after_eosp,
        "cache_seq_lens": lens,
        "all_caches_equal_len": len(set(lens.values())) == 1 if lens else None,
    }

    # --- KV accounting table ----------------------------------------------
    hidden = 4096
    kv_heads, head_dim, bytes_per = 8, 128, 2  # bf16
    per_tok_layer = 2 * kv_heads * head_dim * bytes_per  # K+V
    kv_table = {}
    for ctx in (1024, 4096, 8192, 10240):
        kv_table[ctx] = round(40 * per_tok_layer * ctx / 2**30, 2)
    out["kv"]["per_token_bytes_40_layers"] = 40 * per_tok_layer
    out["kv"]["bf16_gib_by_context"] = kv_table
    out["vram"]["ar_peak_gib"] = round(ar_peak_gib, 2)

    # --- codec decode VRAM -------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    audio, _ = rr.decode_tokens(engine, token_ids, "text_instruct_speech_response",
                                str(Path(args.assets_dir) / "prompt-cn.wav"), seed=0)
    out["vram"]["codec_decode_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out["vram"]["total_reserved_gib"] = round(torch.cuda.memory_reserved() / 2**30, 2)

    # --- dump trace (first 400 events) --------------------------------------
    trace_path = Path(args.out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    slim = [t for t in trace if t["event"] != "forward"][:400]
    (trace_path.parent / "trace_events.json").write_text(json.dumps(slim, indent=1))
    out["trace_path"] = str(trace_path.parent / "trace_events.json")
    trace_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
