#!/usr/bin/env python3
"""P3 / T3.1: BF16 reference baseline, capture-point calibration and
teacher-forced transition probes for MOSS-Speech native parity.

Background (verified against the locked reference code):
  * The P0 fixture exporter loaded the AR via ``AutoModel.from_pretrained``
    WITHOUT ``torch_dtype`` -> the model ran in FP32. The native engine runs
    BF16, so P0 logits are NOT a same-compute-precision baseline. This script
    exports an explicit-BF16 baseline (FP32 kept only as a dtype-gap probe)
    without touching any P0 artifact.
  * Reference capture points per generation step (modeling_moss_speech.py,
    ``MossSpeechGenerationMixin._sample`` / ``_generate_next_tokens_with_scores``):
      raw    : ``outputs.logits_all[:, -1, :]`` right out of the two LM heads,
               before any mask or upcast (captured here via a forward wrapper).
      masked : raw.clone().float() + audio constraints
               (audio ch [16385:] = -inf; eosp -inf while generating_length <
                min_new_tokens)  -- exactly what P0 stored as "logits".
      scored : masked after the per-channel logits processor (greedy:
               repetition penalty 1.1; warpers are not built for do_sample=
               False) -- exactly what P0 stored as "scores".
  * FSM (applied BEFORE the forward of each step, reading the last grid row):
      text-mode  & text ch == sosp(151646) -> audio mode
      audio-mode & audio ch == eosp(16384) -> text mode
    In audio mode the sampled text token is overwritten with modality_pad
    (151667) before the row is appended; in text mode the audio sample is kept
    but ignored by embedding selection (per-row, purely token-driven:
    text != 151667 -> text embed, else audio embed of the audio-channel id).
  * Stop: MIMOStopper(pad=<|endoftext|>) or MIMOStopper(im_end=151645) on the
    TEXT channel of the last appended row (row is part of the output grid).

Outputs (artifacts/p3/reference/):
  <case>/canonical_input.json, tokens_grid.pt, steps_k*.pt (raw/masked/scored),
  meta.json; transition_probes.json; natural_transition/; _determinism.json;
  _fp32_gap.json; _manifest.json (SHAs, dtypes, versions, TF32 flags, attempts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor, GenerationConfig  # noqa: E402

import run_reference as rr  # noqa: E402
from utils.interface import MIMOStopper  # noqa: E402

STOP_IDS = (151643, 151645)  # (<|endoftext|>, im_end); overwritten in main() from tokenizer

MODALITY_PAD = 151667
SOSP = 151646
EOSP = 16384
IM_END = 151645
AUDIO_CH_FORBIDDEN_FROM = 16385
TEXT_VOCAB = 151680
AUDIO_VOCAB = 16512


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_fingerprint(model_path: str, seed: int) -> Dict[str, Any]:
    import subprocess

    def sh(cmd: List[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"<failed: {exc}>"

    return {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "model_path": model_path,
        "model_sha256_4shards": "see _manifest.json",
        "seed": seed,
        "commit": sh(["git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "HEAD"]),
        "hostname": os.uname().nodename,
    }


def load_model(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()
    dts = {str(p.dtype) for p in model.parameters()}
    assert dts == {str(dtype)}, f"expected all-{dtype} params, got {dts}"
    return model


class RawLogitsCapture:
    """Zero-intrusion observation wrapper: records per-forward last-position
    dual-head logits before any mask/upcast. Patches the instance's
    ``forward`` (torch's _call_impl resolves ``self.forward`` on the instance,
    unlike a dunder ``__call__`` attribute, which Python ignores on instances).

    The reference's own capture of `logits`/`scores` via output_logits/
    output_scores is untouched."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._orig = model.forward
        self.steps: List[Tuple[torch.Tensor, torch.Tensor]] = []
        model.forward = self._wrap  # type: ignore[method-assign]

    def _wrap(self, *args, **kwargs):
        out = self._orig(*args, **kwargs)
        la = out.logits_all
        self.steps.append(
            (
                la[0][0, -1, :].detach().to(torch.float32).cpu(),
                la[1][0, -1, :].detach().to(torch.float32).cpu(),
            )
        )
        return out

    def detach_wrapper(self, model: torch.nn.Module) -> None:
        model.forward = self._orig  # type: ignore[method-assign]


def greedy_generate(model, inputs, max_new_tokens: int, seed: int, rep: float, min_new: int,
                    pad_id: int, im_end_id: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cap = RawLogitsCapture(model)
    try:
        out = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            generation_config=GenerationConfig(
                do_sample=False,
                repetition_penalty=rep,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new,
                use_cache=True,
            ),
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            stopping_criteria=[
                MIMOStopper(pad_id),
                MIMOStopper(im_end_id),
            ],
            streamer=rr._NoopStreamer(),
        )
    finally:
        cap.detach_wrapper(model)
    return out, cap.steps


def transition_neighborhoods(grid: torch.Tensor) -> List[int]:
    """Step indices worth keeping beyond the head window: any row whose text
    channel is sosp/modality_pad/im_end/pad-stop or whose audio channel is
    eosp, plus one row before/after each."""
    keep: set = set()
    rows, cols = grid.shape[0], grid.shape[1]
    assert cols == 2
    text_ch = grid[:, 0].tolist()
    audio_ch = grid[:, 1].tolist()
    for i in range(rows):
        interesting = (
            text_ch[i] in (SOSP, MODALITY_PAD, IM_END)
            or audio_ch[i] == EOSP
            or text_ch[i] < 0
        )
        if interesting:
            keep.update({max(0, i - 1), i, min(rows - 1, i + 1)})
    return sorted(keep)


def export_case_steps(
    case_dir: Path,
    grid: torch.Tensor,
    raw_steps: List[Tuple[torch.Tensor, torch.Tensor]],
    masked_steps: List[Tuple[torch.Tensor, torch.Tensor]],
    scored_steps: List[Tuple[torch.Tensor, torch.Tensor]],
    keep_steps: int,
) -> Dict[str, Any]:
    n = grid.shape[0]
    print(f"[debug] {case_dir.name}: grid_rows={n} raw={len(raw_steps)} "
          f"masked={len(masked_steps)} scored={len(scored_steps)}", flush=True)
    assert len(raw_steps) == len(masked_steps) == len(scored_steps) == n
    neigh = set(transition_neighborhoods(grid))
    keep_idx = sorted(set(range(min(keep_steps, n))) | neigh)
    for k in keep_idx:
        torch.save(
            {
                "raw": torch.cat(raw_steps[k]),
                "masked": torch.cat(masked_steps[k]),
                "scored": torch.cat(scored_steps[k]),
            },
            case_dir / f"step_{k:04d}.pt",
        )
    return {
        "n_steps": n,
        "kept_steps": keep_idx,
        "kept_count": len(keep_idx),
    }


def run_case(
    engine, model, out_dir: Path, case_id: str, task: str, conversation, seed: int,
    keep_steps: int, max_new: int, full_capture: bool = False,
) -> Dict[str, Any]:
    modality = rr._task_output_modality(task)
    full_conv = [{"role": "system", "content": rr.system_prompt_for(task)}] + conversation
    inputs = engine.processor([full_conv], [modality])
    canon = {
        "input_ids": inputs["input_ids"][0].cpu().tolist(),
        "attention_mask": inputs["attention_mask"][0].cpu().tolist(),
    }
    out, raw_steps = greedy_generate(model, inputs, max_new, seed, rep=1.1, min_new=0,
                                          pad_id=STOP_IDS[0], im_end_id=STOP_IDS[1])
    grid = out["sequences"][0].cpu()
    masked_steps = [(t[0][0].float().cpu(), t[1][0].float().cpu()) for t in out["logits"]]
    scored_steps = [(t[0][0].float().cpu(), t[1][0].float().cpu()) for t in out["scores"]]

    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "canonical_input.json").write_text(json.dumps(canon))
    torch.save(grid, case_dir / "tokens_grid.pt")
    keep = keep_steps if not full_capture else 10**9
    steps_meta = export_case_steps(case_dir, grid, raw_steps, masked_steps, scored_steps, keep)

    meta: Dict[str, Any] = {
        "case_id": case_id,
        "task": task,
        "seed": seed,
        "conversation": conversation,
        "max_new_tokens": max_new,
        "compute_dtype": "bf16",
        **steps_meta,
    }
    text_ch = grid[:, 0]
    audio_ch = grid[:, 1]
    meta["n_sosp_rows"] = int((text_ch == SOSP).sum())
    meta["n_eosp_rows"] = int((audio_ch == EOSP).sum())
    meta["n_modality_pad_rows"] = int((text_ch == MODALITY_PAD).sum())
    meta["last_text_token"] = int(text_ch[-1])
    (case_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


@torch.no_grad()
def teacher_forced_logits(model, prefix: torch.Tensor, cached: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Last-position dual-head raw logits over a (1, L, 2) prefix.

    cached=False: single full-prefix forward (use_cache=False).
    cached=True : stepwise forward maintaining past_key_values_dict (the
                  reference's cached decode path).
    """
    device = next(model.parameters()).device
    prefix = prefix.to(device)
    Lp = prefix.shape[1]
    ones_mask = torch.ones((1, Lp), dtype=torch.long, device=device)
    if not cached:
        out = model(
            input_ids=prefix,
            attention_mask=ones_mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        la = out.logits_all
        return la[0][:, -1, :].float().cpu(), la[1][:, -1, :].float().cpu()
    past: Optional[Dict[str, Any]] = None
    for pos in range(Lp):
        out = model(
            input_ids=prefix[:, pos : pos + 1],
            attention_mask=ones_mask[:, : pos + 1],
            past_key_values_dict=past,
            use_cache=True,
            cache_position=torch.tensor([pos], device=device),
            logits_to_keep=0 if pos < Lp - 1 else 1,
            return_dict=True,
        )
        past = out.past_key_values_dict
    la = out.logits_all
    return la[0][:, -1, :].float().cpu(), la[1][:, -1, :].float().cpu()


def make_probe(model, prompt_grid: torch.Tensor, gen_grid: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """Build teacher-forced prefixes at generation step k.

    mode 'text_sosp': replace row k text token with sosp -> next step audio.
    mode 'audio_eosp': replace row k audio token with eosp -> next step text.
    mode 'text_eos': replace row k text token with im_end -> stop fires.
    Returns prefix (1, Lp, 2); for 'text_eos' also returns the stop probe row.
    """
    g = gen_grid.clone()
    if mode == "text_sosp":
        g[k, 0] = SOSP
    elif mode == "audio_eosp":
        g[k, 1] = EOSP
    elif mode == "text_eos":
        g[k, 0] = IM_END
    else:
        raise ValueError(mode)
    return torch.cat([prompt_grid.unsqueeze(0), g[: k + 1].unsqueeze(0)], dim=1)


def finite_stats(delta: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> Dict[str, Any]:
    finite = torch.isfinite(delta)
    return {
        "n_finite": int(finite.sum()),
        "n_total": int(delta.numel()),
        "max_abs": float(delta[finite].abs().max()) if finite.any() else None,
        "p999_abs": float(torch.quantile(delta[finite].abs().float(), 0.999)) if finite.any() else None,
        "mean_abs": float(delta[finite].abs().float().mean()) if finite.any() else None,
        "n_over_bound": int((delta[finite].abs() > atol + rtol * ref[finite].abs()).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/MOSS-Speech")
    parser.add_argument("--codec-path", default="models/MOSS-Speech-Codec")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--keep-steps", type=int, default=64)
    args = parser.parse_args()

    rr._install_torchaudio_load_shim()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"env": env_fingerprint(args.model_path, args.seed)}

    # record shard SHAs once (weights unchanged since P0; re-verified)
    shard_shas = {
        f"shard_{i + 1}": sha256_file(Path(args.model_path) / f"model-{i + 1:05d}-of-00004.safetensors")
        for i in range(4)
    }
    p0_shas = Path("artifacts/p0/weights_sha256.txt")
    if p0_shas.exists():
        p0 = {
            line.split()[1].split("/")[-1]: line.split()[0]
            for line in p0_shas.read_text().splitlines()
            if "model-" in line
        }
        mismatch = {
            k: (v, p0.get(f"model-{int(k.split('_')[1]):05d}-of-00004.safetensors"))
            for k, v in shard_shas.items()
            if p0.get(f"model-{int(k.split('_')[1]):05d}-of-00004.safetensors") not in (None, v)
        }
        manifest["weights_match_p0"] = not mismatch
        manifest["weights_mismatch"] = mismatch
    manifest["shard_sha256"] = shard_shas

    engine = rr.Inference(args.model_path, codec_path=args.codec_path, device="cuda")
    tok = engine.processor.tokenizer
    stop_ids = (int(tok.pad_token_id), int(tok.convert_tokens_to_ids("<|im_end|>")))
    manifest["stop_ids"] = {"pad": stop_ids[0], "im_end": stop_ids[1]}
    global STOP_IDS
    STOP_IDS = stop_ids
    model = load_model(args.model_path, torch.bfloat16)
    engine.model = model  # share processor/stoppers; run under explicit bf16
    attn_impl = getattr(model.config, "_attn_implementation", None)
    manifest["attn_implementation"] = attn_impl
    manifest["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    manifest["kv_cache_dtype_class"] = str(type(model.model.shared_block.layers[0].self_attn).__name__)

    assets = Path(args.assets_dir)
    cn = str(assets / "prompt-cn.wav")
    cases = [
        ("t2t_short", "text_instruct_text_response", [rr.user_text_turn("Introduce yourself in one sentence.")], None),
        ("t2s_cn", "text_instruct_speech_response", [rr.user_text_turn("用中文介绍一下上海的三到四个著名景点。")], cn),
        ("s2t_cn", "speech_instruct_text_response", [rr.user_audio_turn(cn)], None),
        ("s2s_cn", "speech_instruct_speech_response", [rr.user_audio_turn(cn)], cn),
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
    grids: Dict[str, torch.Tensor] = {}
    prompts: Dict[str, torch.Tensor] = {}
    for cid, task, conv, _prompt in cases:
        print(f"[bf16-baseline] {cid}", flush=True)
        m = run_case(engine, model, out_dir, cid, task, conv, args.seed, args.keep_steps, args.max_new_tokens)
        summary.append(m)
        grids[cid] = torch.load(out_dir / cid / "tokens_grid.pt")
        prompts[cid] = torch.tensor(json.loads((out_dir / cid / "canonical_input.json").read_text())["input_ids"])

    # cross-check canonical inputs against P0 (processor determinism)
    p0_fixtures = Path("sglang-omni/tests/fixtures/moss_speech")
    cross = {}
    for cid in grids:
        p0c = p0_fixtures / cid / "canonical_input.json"
        if p0c.exists():
            p0_ids = torch.tensor(json.loads(p0c.read_text())["input_ids"])
            cross[cid] = bool(torch.equal(p0_ids, prompts[cid]))
    manifest["canonical_input_matches_p0"] = cross

    # ---- teacher-forced transition probes (cached vs full-prefix, bf16) ----
    probes: Dict[str, Any] = {}
    for cid, mode, k in [
        ("t2t_short", "text_sosp", 8),
        ("t2t_short", "text_eos", 10),
        ("t2s_cn", "audio_eosp", 30),
        ("t2s_cn", "audio_eosp", 60),
    ]:
        prefix = make_probe(model, prompts[cid], grids[cid], k, mode)
        fresh_t, fresh_a = teacher_forced_logits(model, prefix, cached=False)
        cache_t, cache_a = teacher_forced_logits(model, prefix, cached=True)
        probe = {
            "case": cid,
            "mode": mode,
            "k": k,
            "prefix_len": int(prefix.shape[1]),
            "cached_vs_fresh_text": finite_stats(cache_t - fresh_t, fresh_t, 0.0, 0.0),
            "cached_vs_fresh_audio": finite_stats(cache_a - fresh_a, fresh_a, 0.0, 0.0),
            "fresh_text_argmax": int(fresh_t.argmax()),
            "fresh_audio_argmax": int(fresh_a.argmax()),
            "cached_text_argmax": int(cache_t.argmax()),
            "cached_audio_argmax": int(cache_a.argmax()),
        }
        probes[f"{cid}:{mode}:{k}"] = probe
        torch.save({"prefix": prefix.cpu(), "fresh": (fresh_t, fresh_a), "cached": (cache_t, cache_a)},
                   out_dir / f"probe_{cid}_{mode}_{k}.pt")
    (out_dir / "transition_probes.json").write_text(json.dumps(probes, indent=1))

    # ---- natural full-transition search (sosp -> audio -> eosp -> text -> im_end) ----
    nat_dir = out_dir / "natural_transition"
    nat_dir.mkdir(exist_ok=True)
    attempts = []
    found = None
    candidates = [
        ("请只说两个字：你好。", 120),
        ("Say exactly one word: hello.", 120),
        ("请用中文说：早上好。", 120),
        ("Reply with the single word: thanks.", 120),
    ]
    for i, (text, mx) in enumerate(candidates):
        if found is not None:
            break
        conv = [rr.user_text_turn(text)]
        try:
            inputs = engine.processor(
                [{"role": "system", "content": rr.system_prompt_for("text_instruct_speech_response")}, *conv],
                ["audio"],
            )
            out, raw_steps = greedy_generate(model, inputs, mx, args.seed, rep=1.1, min_new=0,
                                          pad_id=STOP_IDS[0], im_end_id=STOP_IDS[1])
            grid = out["sequences"][0].cpu()
            text_ch = grid[:, 0].tolist()
            audio_ch = grid[:, 1].tolist()
            has_audio = any(t == MODALITY_PAD for t in text_ch)
            has_eosp = EOSP in audio_ch
            back_to_text = has_eosp and any(
                text_ch[j] not in (MODALITY_PAD,) for j in range(audio_ch.index(EOSP) + 1, len(text_ch))
            )
            stopped_im_end = int(text_ch[-1]) == IM_END
            ok = has_audio and has_eosp and back_to_text and stopped_im_end
            attempts.append(
                {"i": i, "text": text, "steps": len(text_ch), "has_audio": has_audio,
                 "has_eosp": has_eosp, "back_to_text": back_to_text, "stopped_im_end": stopped_im_end,
                 "accepted": ok}
            )
            if ok:
                found = (i, text)
                cid = "t2s_short_trans"
                case_dir = nat_dir / cid
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "canonical_input.json").write_text(json.dumps(
                    {"input_ids": inputs["input_ids"][0].cpu().tolist(),
                     "attention_mask": inputs["attention_mask"][0].cpu().tolist()}))
                torch.save(grid, case_dir / "tokens_grid.pt")
                masked_steps = [(t[0][0].float().cpu(), t[1][0].float().cpu()) for t in out["logits"]]
                scored_steps = [(t[0][0].float().cpu(), t[1][0].float().cpu()) for t in out["scores"]]
                sm = export_case_steps(case_dir, grid, raw_steps, masked_steps, scored_steps, 10**9)
                (case_dir / "meta.json").write_text(json.dumps(
                    {"case_id": cid, "task": "text_instruct_speech_response", "seed": args.seed,
                     "conversation": conv, "compute_dtype": "bf16", **sm}, ensure_ascii=False, indent=1))
        except Exception as exc:  # noqa: BLE001
            attempts.append({"i": i, "text": text, "error": repr(exc)})
    manifest["natural_transition_attempts"] = attempts
    manifest["natural_transition_found"] = found

    # ---- determinism: rerun t2t_short under bf16 ----
    m1 = run_case(engine, model, out_dir / "_rerun", "t2t_short_rerun", cases[0][1], cases[0][2],
                  args.seed, args.keep_steps, args.max_new_tokens)
    g1 = torch.load(out_dir / "t2t_short" / "tokens_grid.pt")
    g2 = torch.load(out_dir / "_rerun" / "t2t_short_rerun" / "tokens_grid.pt")
    s1 = torch.load(out_dir / "t2t_short" / "step_0000.pt")
    s2 = torch.load(out_dir / "_rerun" / "t2t_short_rerun" / "step_0000.pt")
    rdiff = finite_stats(s1["raw"] - s2["raw"], s2["raw"], 0.0, 0.0)
    det = {
        "tokens_equal": bool(torch.equal(g1, g2)),
        "step0_raw_max_abs": rdiff["max_abs"],
        "pass": bool(torch.equal(g1, g2)) and (rdiff["max_abs"] or 0.0) == 0.0,
    }
    (out_dir / "_determinism.json").write_text(json.dumps(det, indent=1))
    manifest["determinism"] = det

    # ---- fp32 dtype-gap probe on t2t_short (P0-equivalent load path) ----
    del model
    torch.cuda.empty_cache()
    engine.model = load_model(args.model_path, torch.float32)
    m_fp32 = run_case(engine, engine.model, out_dir / "_fp32", "t2t_short_fp32", cases[0][1], cases[0][2],
                      args.seed, args.keep_steps, args.max_new_tokens)
    g32 = torch.load(out_dir / "_fp32" / "t2t_short_fp32" / "tokens_grid.pt")
    bf_grid = torch.load(out_dir / "t2t_short" / "tokens_grid.pt")
    gap = {"tokens_equal": bool(torch.equal(g32, bf_grid))}
    step_stats = []
    n = min(bf_grid.shape[0], g32.shape[0])
    for k in range(n):
        f32 = torch.load(out_dir / "_fp32" / "t2t_short_fp32" / f"step_{k:04d}.pt") if (
            out_dir / "_fp32" / "t2t_short_fp32" / f"step_{k:04d}.pt").exists() else None
        bf16s = torch.load(out_dir / "t2t_short" / f"step_{k:04d}.pt") if (
            out_dir / "t2t_short" / f"step_{k:04d}.pt").exists() else None
        if f32 is None or bf16s is None:
            continue
        st = finite_stats(bf16s["raw"] - f32["raw"], f32["raw"], 0.0, 0.0)
        st["k"] = k
        step_stats.append(st)
    if step_stats:
        gap["per_step_max_abs_max"] = max(s["max_abs"] for s in step_stats)
        gap["per_step_p999_max"] = max(s["p999_abs"] for s in step_stats)
        gap["n_over_combined_atol_rtol_examples"] = [
            {"k": s["k"], "max_abs": s["max_abs"], "p999": s["p999_abs"]} for s in step_stats[:5]
        ]
    (out_dir / "_fp32_gap.json").write_text(json.dumps(gap, indent=1))
    manifest["fp32_gap"] = gap

    (out_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({"determinism": det, "fp32_gap": gap, "natural": found}, indent=1))
    print("REFERENCE BASELINE DONE", flush=True)


if __name__ == "__main__":
    main()
