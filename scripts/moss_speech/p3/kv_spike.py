#!/usr/bin/env python3
"""P3 / T3.3: real-weight load, 40-layer KV accounting and cache equivalence.

Drives the NATIVE model through the same common bootstrap the formal engine
uses (create_sglang_infrastructure_defer_cuda_graph with the arch override),
then teacher-forces the T3.1 transition probes with hand-built ForwardBatch
objects — no scheduler loop, no sampling FSM (those are T3.4+).

Checks (frozen acceptance, Tasks.md T3.3):
  1. bf16 real weights: 446-source coverage (loader asserts), all-param dtype
     census, pre/post-load GPU memory, pool budget fields.
  2. Runtime layer accounting: ModelConfig.num_hidden_layers==36 (HF
     semantics) AND num_attention_layers/layer_info/allocator/model layer ids
     all consistent at 40 BEFORE any forward.
  3. KV theory vs practice: 40*2*8*128*2B = 160 KiB/token; per-layer buffer
     shape/dtype; tail buffers not aliased (storage identity + sentinel
     write isolation).
  4. Transition probes (text+sosp, audio+eosp x2, text EOS): native cached
     decode vs native full-prefix recompute vs the T3.1 BF16 reference
     (frozen tolerance: |d| <= 1.0 + 0.02|ref|, over-bound <= 1e-4,
     max-abs <= 4.5; argmax equality recorded).
  5. Directed KV perturbation: zeroing the TEXT-tail layers' history at audio
     positions changes text logits; zeroing the AUDIO tail leaves text logits
     intact (proves each tail reads its OWN history; no cross-tail reads).
  6. Slot release/reuse: freed slots re-allocated to an identical second
     request reproduce the first request's logits bit-exactly.
  7. Teardown: pools freed, process group destroyed, exit recorded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

MODALITY_PAD = 151667
TEXT_VOCAB = 151680


def gpu_mem() -> Dict[str, float]:
    torch.cuda.synchronize()
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
        "avail_gib": round(
            (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_reserved()
            )
            / 2**30,
            3,
        ),
    }


def build_input_embeds(model, prefix_rows: torch.Tensor) -> torch.Tensor:
    """(L, 2) grid rows -> (L, H) embeds via the reference's token-driven rule:
    text != modality_pad -> text embed(text); else audio embed(audio id)."""
    text_ch = prefix_rows[:, 0].to(model.embed_tokens.weight.device)
    audio_ch = prefix_rows[:, 1].to(text_ch.device)
    is_audio = text_ch == MODALITY_PAD
    text_safe = text_ch.masked_fill(is_audio, 0)
    text_emb = model.embed_tokens(text_safe)
    audio_emb = model.audio_embed(audio_ch)
    sel = (~is_audio).unsqueeze(-1).to(text_emb.dtype)
    return text_emb * sel + audio_emb * (1 - sel)


def write_req_slots(runner, req_idx: int, slots: torch.Tensor) -> None:
    """req_to_token_pool is the attention metadata source for kv indices;
    leaving it untouched means the backend reads uninitialized slots."""
    slots_dev = slots.to(runner.device, dtype=torch.int64).flatten()
    # canonical index form (radix_cache.py precedent): req_to_token[i, 0:L]
    runner.req_to_token_pool.write((req_idx, slice(0, slots_dev.shape[0])), slots_dev)


def make_forward_batch(runner, *, mode, seq_lens, out_cache_loc, input_ids, positions):
    """Single-request ForwardBatch with explicit positions/slots/mask/dtype."""
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    fb = object.__new__(ForwardBatch)
    fb.forward_mode = mode
    fb.batch_size = len(seq_lens)
    fb.input_ids = input_ids.to(runner.device, dtype=torch.int64)
    fb.req_pool_indices = torch.zeros(
        len(seq_lens), dtype=torch.int64, device=runner.device
    )
    fb.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=runner.device)
    fb.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    fb.seq_lens_sum = int(sum(seq_lens))
    fb.out_cache_loc = out_cache_loc.to(runner.device, dtype=torch.int64)
    fb.positions = positions.to(runner.device, dtype=torch.int64)
    if mode == ForwardMode.EXTEND:
        fb.extend_seq_lens = fb.seq_lens.clone()
        fb.extend_prefix_lens = torch.zeros_like(fb.seq_lens)
        fb.extend_seq_lens_cpu = fb.extend_seq_lens.cpu()
        fb.extend_prefix_lens_cpu = fb.extend_prefix_lens.cpu()
        fb.extend_seq_lens_sum = int(fb.extend_seq_lens.sum())
    else:
        fb.extend_seq_lens = None
        fb.extend_prefix_lens = None
        fb.extend_seq_lens_cpu = None
        fb.extend_prefix_lens_cpu = None
        fb.extend_seq_lens_sum = None
    fb.device = runner.device
    return fb


def run_forward(runner, model, fb, input_embeds):
    """init metadata + forward under an explicit ForwardContext (mirrors
    ModelRunner.forward: the global context carries the attn backend that
    RadixAttention resolves at call time)."""
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    assert runner.attn_backend is not None
    with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        runner.attn_backend.init_forward_metadata(forward_batch=fb)
        return model(
            input_ids=fb.input_ids,
            positions=fb.positions,
            forward_batch=fb,
            input_embeds=input_embeds.to(runner.device),
        )


def native_probe_logits(runner, model, alloc, prefix: torch.Tensor, cached: bool):
    """Teacher-forced native logits for a (Lp, 2) prefix.

    cached=False: one EXTEND over the full prefix (full-prefix recompute).
    cached=True : EXTEND over prefix[:-1] then a DECODE step on the last row
    (the reference's cached decode path semantics from T3.1).
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    Lp = prefix.shape[0]
    if not cached:
        slots = alloc(Lp)
        write_req_slots(runner, 0, slots)
        fb = make_forward_batch(
            runner,
            mode=ForwardMode.EXTEND,
            seq_lens=[Lp],
            out_cache_loc=slots,
            input_ids=torch.zeros(Lp),
            positions=torch.arange(Lp),
        )
        out = run_forward(runner, model, fb, build_input_embeds(model, prefix))
        return (
            out.text_logits[0].float().cpu(),
            out.audio_logits[0].float().cpu(),
            slots,
        )
    slots = alloc(Lp - 1)
    write_req_slots(runner, 0, slots)
    fb = make_forward_batch(
        runner,
        mode=ForwardMode.EXTEND,
        seq_lens=[Lp - 1],
        out_cache_loc=slots,
        input_ids=torch.zeros(Lp - 1),
        positions=torch.arange(Lp - 1),
    )
    run_forward(runner, model, fb, build_input_embeds(model, prefix[:-1]))
    last_slot = alloc(1)
    all_slots = torch.cat([slots, last_slot.to(slots.device)])
    write_req_slots(runner, 0, all_slots)
    fb2 = make_forward_batch(
        runner,
        mode=ForwardMode.DECODE,
        seq_lens=[Lp],
        out_cache_loc=last_slot,
        input_ids=torch.zeros(1),
        positions=torch.tensor([Lp - 1]),
    )
    out = run_forward(runner, model, fb2, build_input_embeds(model, prefix[-1:]))
    return (
        out.text_logits[0].float().cpu(),
        out.audio_logits[0].float().cpu(),
        all_slots,
    )


def compare(native: torch.Tensor, ref: torch.Tensor) -> Dict[str, Any]:
    """Frozen P3-02 §3 protocol: non-finite pattern, combined bound, max-abs."""
    n_fin = torch.isfinite(native)
    r_fin = torch.isfinite(ref)
    pattern_ok = bool(torch.equal(n_fin, r_fin))
    if n_fin.numel() != ref.numel():
        return {"shape_mismatch": [list(native.shape), list(ref.shape)], "pass": False}
    d = (native - ref)[n_fin & r_fin]
    stats = {
        "nonfinite_pattern_equal": pattern_ok,
        "n_finite": int((n_fin & r_fin).sum()),
        "max_abs": float(d.abs().max()) if d.numel() else 0.0,
    }
    if d.numel():
        over = d.abs() > (1.0 + 0.02 * ref[n_fin & r_fin].abs())
        stats["over_bound_frac"] = float(over.float().mean())
        stats["pass"] = bool(
            pattern_ok and stats["max_abs"] <= 4.5 and stats["over_bound_frac"] <= 1e-4
        )
    else:
        stats["pass"] = pattern_ok
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/MOSS-Speech")
    parser.add_argument("--probes-dir", default="artifacts/p3/reference")
    parser.add_argument("--out", default="artifacts/p3/kv_spike.json")
    parser.add_argument("--mem-fraction", type=float, default=0.72)
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    }
    pre = gpu_mem()

    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
    from sglang_omni.models.moss_speech.hf_config import (  # noqa: F401 (AutoConfig registration)
        MossSpeechConfig,
    )
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling.sglang_backend.server_args_builder import (
        build_sglang_server_args,
    )

    builder = MossSpeechEngineBuilder()
    overrides = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(overrides)
    overrides["mem_fraction_static"] = args.mem_fraction
    server_args = build_sglang_server_args(
        args.model_path, context_length=builder.context_length, **overrides
    )
    builder.validate_before_infrastructure(server_args)

    (_want_graph, infra) = (
        scheduling_bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args, 0, model_arch_override="MossSpeechForCausalLM"
        )
    )
    (model_worker, tree_cache, req_pool, kv_allocator, _pm, _dm, model_config) = infra
    runner = model_worker.model_runner
    model = runner.model

    from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel

    assert isinstance(model, MossSpeechSGLangModel), f"loaded {type(model)}"
    post = gpu_mem()
    report["load"] = {"pre": pre, "post": post}

    # ---- 1/2: structure + layer accounting --------------------------------
    li = runner.layer_info
    report["layers"] = {
        "hf_num_hidden_layers": int(model_config.num_hidden_layers),
        "num_attention_layers": int(model_config.num_attention_layers),
        "layer_info": {
            "start": int(li.start_layer),
            "end": int(li.end_layer),
            "num_effective": int(li.num_effective_layers),
        },
        "model_layer_ids": model.attention_layer_ids(),
    }
    assert int(model_config.num_hidden_layers) == 36
    assert int(model_config.num_attention_layers) == 40
    assert int(li.num_effective_layers) == 40 and int(li.end_layer) == 40
    assert sorted(model.attention_layer_ids()) == list(range(40))

    dtypes: Dict[str, int] = {}
    for p in model.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    model.assert_heads_independent()
    report["weights"] = {
        "param_dtypes": dtypes,
        "heads_independent": True,
        "coverage": "load_weights RuntimeError-on-missing (446-source assert)",
    }

    # ---- 3: KV pool theory vs practice ------------------------------------
    pool = runner.token_to_kv_pool
    kv = pool if hasattr(pool, "k_buffer") else pool.token_to_kv_pool
    layer_num = int(getattr(kv, "layer_num", len(kv.k_buffer)))
    k0 = kv.k_buffer[0]
    per_token_bytes = (
        layer_num
        * 2
        * int(k0.shape[-2])
        * int(k0.shape[-1])
        * torch.tensor([], dtype=k0.dtype).element_size()
    )
    report["kv_pool"] = {
        "class": type(kv).__name__,
        "layer_num": layer_num,
        "dtype": str(k0.dtype),
        "k_shape": list(k0.shape),
        "size_slots": int(kv.size),
        "page_size": int(kv.page_size),
        "per_token_bytes": per_token_bytes,
        "per_token_kib": per_token_bytes / 1024,
        "theory_kib": 40 * 2 * 8 * 128 * 2 / 1024,
        "allocator_size": int(kv_allocator.size),
    }
    assert layer_num == 40, layer_num
    assert per_token_bytes == 40 * 2 * 8 * 128 * 2, per_token_bytes
    # tail buffers not aliased (storage identity + sentinel isolation)
    probe_slot = int(torch.tensor([0]).to(kv.k_buffer[0].device))
    sent = torch.full_like(kv.k_buffer[32][probe_slot], 7.5)
    orig32 = kv.k_buffer[32][probe_slot].clone()
    orig36 = kv.k_buffer[36][probe_slot].clone()
    kv.k_buffer[32][probe_slot] = sent
    no_alias = (
        bool(torch.equal(kv.k_buffer[36][probe_slot], orig36))
        and kv.k_buffer[32].data_ptr() != kv.k_buffer[36].data_ptr()
    )
    kv.k_buffer[32][probe_slot] = orig32
    report["kv_pool"]["tail_no_alias"] = no_alias
    assert no_alias

    # ---- 4: teacher-forced probes ------------------------------------------
    def alloc(n: int) -> torch.Tensor:
        return kv_allocator.alloc(n)

    probes_report = {}
    probe_files = sorted(Path(args.probes_dir).glob("probe_*.pt"))
    assert probe_files, f"no probes under {args.probes_dir}"
    for pf in probe_files:
        blob = torch.load(pf, map_location="cpu")
        prefix = blob["prefix"][0]  # (Lp, 2)
        entry: Dict[str, Any] = {"prefix_len": int(prefix.shape[0])}
        ref_fresh_t, ref_fresh_a = blob["fresh"]
        ref_cached_t, ref_cached_a = blob["cached"]
        nt_f, na_f, slots_f = native_probe_logits(
            runner, model, alloc, prefix, cached=False
        )
        entry["fresh_vs_ref"] = {
            "text": compare(nt_f, ref_fresh_t),
            "audio": compare(na_f, ref_fresh_a),
        }
        nt_c, na_c, slots_c = native_probe_logits(
            runner, model, alloc, prefix, cached=True
        )
        entry["cached_vs_ref"] = {
            "text": compare(nt_c, ref_cached_t),
            "audio": compare(na_c, ref_cached_a),
        }
        entry["cached_vs_fresh_native"] = {
            "text": compare(nt_c, nt_f),
            "audio": compare(na_c, na_f),
        }
        entry["argmax"] = {
            "fresh_text": [int(nt_f.argmax()), int(ref_fresh_t.argmax())],
            "fresh_audio": [int(na_f.argmax()), int(ref_fresh_a.argmax())],
            "cached_text": [int(nt_c.argmax()), int(ref_cached_t.argmax())],
            "cached_audio": [int(na_c.argmax()), int(ref_cached_a.argmax())],
        }
        probes_report[pf.name] = entry
        kv_allocator.free(slots_f.to(torch.int64))
        kv_allocator.free(slots_c.to(torch.int64))
    report["probes"] = probes_report

    # ---- 5: directed KV perturbation on one audio->text transition probe ----
    pf = Path(args.probes_dir) / "probe_t2s_cn_audio_eosp_30.pt"
    blob = torch.load(pf, map_location="cpu")
    prefix = blob["prefix"][0]
    text_rows = prefix[:, 0] != MODALITY_PAD
    audio_positions = (~text_rows).nonzero().flatten()
    base_t, base_a, p_slots = native_probe_logits(
        runner, model, alloc, prefix, cached=True
    )
    audio_slot_ids = p_slots[audio_positions.to(p_slots.device)]

    # Replay ONLY the final decode step against the already-written KV
    # (re-running the full teacher-forced path would rewrite the buffers and
    # erase the perturbation -- that was the earlier zero-delta artifact).
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    def replay_last_step():
        fb2 = make_forward_batch(
            runner,
            mode=ForwardMode.DECODE,
            seq_lens=[prefix.shape[0]],
            out_cache_loc=p_slots[-1:].to(runner.device),
            input_ids=torch.zeros(1),
            positions=torch.tensor([prefix.shape[0] - 1]),
        )
        out = run_forward(runner, model, fb2, build_input_embeds(model, prefix[-1:]))
        return out.text_logits[0].float().cpu(), out.audio_logits[0].float().cpu()

    def perturb(layers: List[int], zero: bool = True):
        saved = []
        for lyr in layers:
            buf = kv.k_buffer[lyr]
            idx = audio_slot_ids.to(buf.device)
            saved.append((buf, idx, buf[idx].clone()))
            if zero:
                buf[idx] = 0
        return saved

    def restore(saved):
        for buf, idx, orig in saved:
            buf[idx] = orig

    base_t2, base_a2 = replay_last_step()
    L_TEXT = list(range(32, 36))
    L_AUDIO = list(range(36, 40))
    saved = perturb(L_TEXT)
    pt_t, pt_a = replay_last_step()
    restore(saved)
    text_delta_on_text_perturb = float((pt_t - base_t2).abs().max())
    saved = perturb(L_AUDIO)
    pa_t, pa_a = replay_last_step()
    restore(saved)
    text_delta_on_audio_perturb = float((pa_t - base_t2).abs().max())
    audio_delta_on_audio_perturb = float((pa_a - base_a2).abs().max())
    report["perturbation"] = {
        "n_audio_positions": int(audio_positions.numel()),
        "text_logits_delta_text_tail_zeroed": text_delta_on_text_perturb,
        "text_logits_delta_audio_tail_zeroed": text_delta_on_audio_perturb,
        "audio_logits_delta_audio_tail_zeroed": audio_delta_on_audio_perturb,
        "pass": bool(
            text_delta_on_text_perturb > 1.0
            and text_delta_on_audio_perturb < 0.5
            and audio_delta_on_audio_perturb > 1.0
        ),
    }
    kv_allocator.free(p_slots.to(torch.int64))

    # ---- 6: slot release/reuse ---------------------------------------------
    blob = torch.load(
        Path(args.probes_dir) / "probe_t2t_short_text_sosp_8.pt", map_location="cpu"
    )
    prefix = blob["prefix"][0]
    t1, a1, s1 = native_probe_logits(runner, model, alloc, prefix, cached=False)
    kv_allocator.free(s1.to(torch.int64))
    t2, a2, s2 = native_probe_logits(runner, model, alloc, prefix, cached=False)
    kv_allocator.free(s2.to(torch.int64))
    report["slot_reuse"] = {
        "second_alloc_overlaps_freed": bool(
            len(set(s1.tolist()) & set(s2.tolist())) > 0
        ),
        "text_bit_equal": bool(torch.equal(t1, t2)),
        "audio_bit_equal": bool(torch.equal(a1, a2)),
        "pass": bool(torch.equal(t1, t2) and torch.equal(a1, a2)),
    }

    # ---- 7: teardown ---------------------------------------------------------
    del model, runner, model_worker
    torch.cuda.empty_cache()
    try:
        import torch.distributed as dist
        from sglang.srt.distributed import parallel_state as pstate

        pstate.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
        destroyed = True
    except Exception as exc:  # noqa: BLE001
        destroyed = f"error: {exc!r}"
    report["teardown"] = {"distributed_destroyed": destroyed, "post_mem": gpu_mem()}

    ok = (
        all(
            v["pass"]
            for p in probes_report.values()
            for k in ("fresh_vs_ref", "cached_vs_ref")
            for v in p[k].values()
        )
        and report["perturbation"]["pass"]
        and report["slot_reuse"]["pass"]
    )
    report["pass"] = bool(ok)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "kv": report["kv_pool"],
                "perturbation": report["perturbation"],
                "slot_reuse": report["slot_reuse"],
            },
            indent=1,
        )
    )
    print("KV SPIKE DONE", flush=True)


if __name__ == "__main__":
    main()
