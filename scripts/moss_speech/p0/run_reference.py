#!/usr/bin/env python3
"""Headless reference driver for MOSS-Speech four-mode baseline (P0 / T0.4).

Replicates the official `utils/interface.py` inference path (OpenMOSS/MOSS-Speech
GitHub repo, feat/docs @ 1ea408a) with three minimal, documented deviations:

1. `do_sample` is switchable (`--sampling greedy|default`); the official code
   hardcodes `do_sample=True`. Greedy uses temperature=1.0/top_p=1.0/top_k=0
   (HF 4.57 semantics) and keeps repetition_penalty.
2. The generated `token_ids` tensor is captured before `processor.decode` and
   saved for determinism checks and later fixture export.
3. No gradio UI: uses the `Inference` engine directly (gradio is still imported
   transitively by `utils.interface` at module top level).

Runs on a Slurm compute node in offline mode. See
docs/design/moss_speech/p0/{01_version_lock,02_deps}.md for environment details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Offline discipline (AGENT.md section 2).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import GenerationConfig  # noqa: E402
from transformers.generation.streamers import BaseStreamer  # noqa: E402

# utils.interface lives in the locked GitHub clone; sys.path is set by the
# launcher (PYTHONPATH includes repos/MOSS-Speech and Matcha-TTS).
from utils.interface import Inference, MIMOStopper  # noqa: E402


class _NoopStreamer(BaseStreamer):
    """No-op streamer: transformers 4.57.1 `generate()` drops `streamer=None`
    from kwargs, which breaks the remote `_sample(streamer)` positional
    signature (written for transformers 4.57.0.dev0). Passing this no-op keeps
    the reference code pristine and semantically unchanged."""

    def put(self, value: torch.Tensor) -> None:  # noqa: D102
        return None

    def end(self) -> None:  # noqa: D102
        return None


def _install_torchaudio_load_shim() -> None:
    """Replace `torchaudio.load` with a soundfile-backed loader.

    torchaudio 2.9 routes `load` through TorchCodec, whose wheels are built
    against torch/cuda versions we cannot use here (0.16 needs cu13; older
    builds core-dump on cluster nodes). This shim is IO-only and matches
    torchaudio semantics: float32 tensor of shape (channels, samples), plus
    sample rate. Reference code stays pristine.
    """

    import soundfile as _sf
    import torchaudio as _ta

    def _load(filepath, *args, **kwargs):
        data, sr = _sf.read(str(filepath), dtype="float32", always_2d=True)
        return torch.from_numpy(data.copy()).T, sr

    _ta.load = _load


SYSTEM_PROMPT_SPEECH = "You are a helpful voice assistant. Answer the user's questions with spoken responses."
SYSTEM_PROMPT_TEXT = (
    "You are a helpful assistant. Answer the user's questions with text."
)


@dataclass
class CaseResult:
    case_id: str
    task: str
    sampling: str
    seed: int
    text: Optional[str]
    audio_path: Optional[str]
    audio_meta: Optional[Dict[str, Any]]
    n_steps: Optional[int]  # number of newly generated grid steps (incl. stop step)
    wall_s: float
    peak_vram_gib: float
    tokens_path: Optional[str] = None
    determinism_repeat_ok: Optional[bool] = None


def _task_output_modality(task: str) -> str:
    if task.endswith("speech_response"):
        return "audio"
    if task.endswith("text_response"):
        return "text"
    raise ValueError(f"Unknown task: {task}")


def system_prompt_for(task: str) -> str:
    return (
        SYSTEM_PROMPT_SPEECH
        if _task_output_modality(task) == "audio"
        else SYSTEM_PROMPT_TEXT
    )


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    if args.sampling == "greedy":
        return GenerationConfig(
            do_sample=False,
            repetition_penalty=1.1,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            use_cache=True,
        )
    return GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        use_cache=True,
    )


def generate_once(
    engine: Inference,
    task: str,
    conversation: List[Dict[str, Any]],
    gen_cfg: GenerationConfig,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    """One forward pass mirroring Inference.forward, returning raw token ids
    and the prompt grid length (for step accounting)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_modalities = [_task_output_modality(task)]
    full_conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt_for(task)}
    ]
    full_conversation.extend(conversation)
    inputs = engine.processor([full_conversation], output_modalities)
    in_len = int(inputs["input_ids"].shape[-1])
    stopping_criteria = [
        MIMOStopper(engine.processor.tokenizer.pad_token_id),
        MIMOStopper(engine.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")),
    ]
    token_ids = engine.model.generate(
        input_ids=inputs["input_ids"].to(engine.device),
        attention_mask=inputs["attention_mask"].to(engine.device),
        generation_config=gen_cfg,
        stopping_criteria=stopping_criteria,
        streamer=_NoopStreamer(),
    )
    return token_ids, in_len


def decode_tokens(
    engine: Inference,
    token_ids: torch.Tensor,
    task: str,
    decoder_audio_prompt_path: Optional[str],
    seed: int,
) -> Tuple[Tuple[int, torch.Tensor] | None, str | None]:
    """Deterministic processor.decode with an explicit voice prompt (the
    official default prompt path is buggy; callers always pass a locked asset)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    modality = _task_output_modality(task)
    results = engine.processor.decode(
        token_ids.to(engine.device),
        [modality],
        decoder_audio_prompt_path=decoder_audio_prompt_path,
    )
    resp = results[0]
    if modality == "audio" and getattr(resp, "audio", None) is not None:
        return (int(resp.sampling_rate), resp.audio.squeeze(0).cpu()), None
    return None, getattr(resp, "generated_text", None)


def audio_meta(audio: Tuple[int, torch.Tensor]) -> Dict[str, Any]:
    sr, wav = audio
    arr = wav.numpy() if isinstance(wav, torch.Tensor) else wav
    digest = hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
    return {
        "sampling_rate": sr,
        "channels": int(1 if arr.ndim == 1 else arr.shape[-1]),
        "duration_s": round(len(arr) / sr, 3),
        "dtype": str(arr.dtype),
        "waveform_blake2b": digest,
    }


def env_fingerprint(args: argparse.Namespace) -> Dict[str, Any]:
    import transformers

    fp: Dict[str, Any] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        fp["gpu"] = torch.cuda.get_device_name(0)
        fp["gpu_count"] = torch.cuda.device_count()
    fp["model_path"] = str(Path(args.model_path).resolve())
    fp["codec_path"] = str(Path(args.codec_path).resolve())
    fp["seed"] = args.seed
    fp["sampling"] = args.sampling
    return fp


def save_case(out_dir: Path, res: CaseResult, tokens: Optional[torch.Tensor]) -> None:
    case_dir = out_dir / res.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "case_id": res.case_id,
        "task": res.task,
        "sampling": res.sampling,
        "seed": res.seed,
        "text": res.text,
        "audio": res.audio_meta,
        "n_new_steps": res.n_steps,
        "wall_s": round(res.wall_s, 2),
        "peak_vram_gib": round(res.peak_vram_gib, 2),
        "determinism_repeat_ok": res.determinism_repeat_ok,
    }
    (case_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    if res.text is not None:
        (case_dir / "text.txt").write_text(res.text)
    if tokens is not None:
        torch.save(tokens.cpu(), case_dir / "tokens.pt")


def run_case(
    engine: Inference,
    out_dir: Path,
    case_id: str,
    task: str,
    conversation: List[Dict[str, Any]],
    args: argparse.Namespace,
    decoder_audio_prompt: Optional[str],
) -> CaseResult:
    gen_cfg = build_generation_config(args)
    modality = _task_output_modality(task)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    token_ids, in_len = generate_once(engine, task, conversation, gen_cfg, args.seed)
    gen_wall = time.time() - t0
    gen_peak = (
        torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    )

    # Deterministic decode with the locked voice prompt (audio) or text decode.
    audio_meta_dict: Optional[Dict[str, Any]] = None
    audio_out_path: Optional[str] = None
    text: Optional[str] = None
    if modality == "audio":
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        audio, _ = decode_tokens(
            engine, token_ids, task, decoder_audio_prompt, args.seed
        )
        dec_wall = time.time() - t1
        dec_peak = (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0.0
        )
        if audio is not None:
            sr, wav = audio
            audio_meta_dict = audio_meta(audio)
            audio_meta_dict["decode_wall_s"] = round(dec_wall, 2)
            audio_meta_dict["decode_peak_vram_gib"] = round(dec_peak, 2)
            case_dir = out_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            sf.write(
                case_dir / "audio.wav",
                wav.numpy() if hasattr(wav, "numpy") else wav,
                sr,
            )
            audio_out_path = str(case_dir / "audio.wav")
    else:
        _, text = decode_tokens(engine, token_ids, task, None, args.seed)

    # Determinism: re-run generation with same seed/settings, compare tokens.
    det_ok: Optional[bool] = None
    if args.check_determinism:
        token_ids2, _ = generate_once(engine, task, conversation, gen_cfg, args.seed)
        det_ok = bool(
            token_ids.shape == token_ids2.shape
            and torch.equal(token_ids.cpu(), token_ids2.cpu())
        )

    n_new_steps = (
        int(token_ids.shape[1]) if token_ids is not None else None
    )  # grid is (B, L, 2); generate(output_only=True) strips prompt
    res = CaseResult(
        case_id=case_id,
        task=task,
        sampling=args.sampling,
        seed=args.seed,
        text=text if modality == "text" else None,
        audio_path=audio_out_path,
        audio_meta=audio_meta_dict,
        n_steps=n_new_steps,
        wall_s=gen_wall,
        peak_vram_gib=gen_peak,
        determinism_repeat_ok=det_ok,
    )
    save_case(out_dir, res, token_ids)
    return res


def user_audio_turn(path: str) -> Dict[str, Any]:
    return {"role": "user", "content": {"path": path, "type": "audio/wav"}}


def assistant_audio_turn(path: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": {"path": path, "type": "filepath"}}


def assistant_text_turn(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text}


def user_text_turn(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/MOSS-Speech")
    parser.add_argument("--codec-path", default="models/MOSS-Speech-Codec")
    parser.add_argument(
        "--assets-dir", required=True, help="locked GitHub clone assets dir"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sampling", choices=["greedy", "default"], default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--skip-mixed", action="store_true")
    args = parser.parse_args()

    _install_torchaudio_load_shim()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_env.json").write_text(json.dumps(env_fingerprint(args), indent=1))

    engine = Inference(args.model_path, codec_path=args.codec_path, device="cuda")

    assets = Path(args.assets_dir)
    prompt_cn = str(assets / "prompt-cn.wav")
    prompt_en = str(assets / "prompt-en.wav")

    long_text = (
        "Please summarize the following passage in three bullet points: The industrial "
        "revolution transformed manufacturing, transportation, and daily life across Europe "
        "and North America. Steam power replaced human and animal labor in factories, while "
        "railways connected distant markets for the first time. Urban populations exploded "
        "as workers moved to cities, and new social classes emerged around industrial capital."
    )
    cases: List[Tuple[str, str, List[Dict[str, Any]], Optional[str]]] = [
        (
            "t2t_short",
            "text_instruct_text_response",
            [user_text_turn("Introduce yourself in one sentence.")],
            None,
        ),
        ("t2t_long", "text_instruct_text_response", [user_text_turn(long_text)], None),
        (
            "t2s_cn",
            "text_instruct_speech_response",
            [user_text_turn("用中文介绍一下上海的三到四个著名景点。")],
            prompt_cn,
        ),
        (
            "t2s_en",
            "text_instruct_speech_response",
            [user_text_turn("Say something encouraging to a student before an exam.")],
            prompt_en,
        ),
        ("s2t_cn", "speech_instruct_text_response", [user_audio_turn(prompt_cn)], None),
        ("s2t_en", "speech_instruct_text_response", [user_audio_turn(prompt_en)], None),
        (
            "s2s_cn",
            "speech_instruct_speech_response",
            [user_audio_turn(prompt_cn)],
            prompt_cn,
        ),
        (
            "s2s_en",
            "speech_instruct_speech_response",
            [user_audio_turn(prompt_en)],
            prompt_en,
        ),
    ]

    summary: List[Dict[str, Any]] = []
    for case_id, task, conv, dec_prompt in cases:
        print(f"[case] {case_id} ({task})", flush=True)
        res = run_case(engine, out_dir, case_id, task, conv, args, dec_prompt)
        summary.append(
            {
                "case_id": res.case_id,
                "task": res.task,
                "wall_s": res.wall_s,
                "n_new_steps": res.n_steps,
                "det_ok": res.determinism_repeat_ok,
                "audio": res.audio_meta,
                "text": (res.text or "")[:80],
            }
        )
        print(f"  -> {summary[-1]}", flush=True)

    if not args.skip_mixed:
        # Mixed multi-turn: text Q/A, then audio user turn -> audio A, then
        # audio user turn -> text A (exercises text-after-audio continuation
        # across turns and per-turn processor dispatch).
        conv: List[Dict[str, Any]] = [
            user_text_turn("My name is Alice and I like hiking.")
        ]
        r1 = run_case(
            engine,
            out_dir,
            "mixed_t1_text",
            "text_instruct_text_response",
            conv,
            args,
            None,
        )
        assert r1.text, "mixed turn 1 must produce text"
        conv.append(assistant_text_turn(r1.text))
        conv.append(user_audio_turn(prompt_en))
        r2 = run_case(
            engine,
            out_dir,
            "mixed_t2_s2s",
            "speech_instruct_speech_response",
            conv,
            args,
            prompt_en,
        )
        if r2.audio_path:
            conv.append(assistant_audio_turn(r2.audio_path))
        conv.append(user_audio_turn(prompt_cn))
        r3 = run_case(
            engine,
            out_dir,
            "mixed_t3_s2t",
            "speech_instruct_text_response",
            conv,
            args,
            None,
        )
        summary.append(
            {
                "case_id": "mixed",
                "turns": ["t1_text", "t2_s2s", "t3_s2t"],
                "t3_text": (r3.text or "")[:120],
            }
        )

    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    print("ALL CASES DONE", flush=True)


if __name__ == "__main__":
    main()
