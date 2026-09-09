#!/usr/bin/env python3
"""T3.3 diagnostic: localize the native-vs-reference logits gap.

Single t2t prompt, fresh EXTEND only. Prints activation norms per stage and
top-k overlap vs the T3.1 reference step-0 raw logits, and tries multiple
attention backends to expose metadata assumptions of hand-built batches.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

MODALITY_PAD = 151667


def build_input_embeds(model, rows: torch.Tensor) -> torch.Tensor:
    text_ch = rows[:, 0].to(model.embed_tokens.weight.device)
    audio_ch = rows[:, 1].to(text_ch.device)
    is_audio = text_ch == MODALITY_PAD
    te = model.embed_tokens(text_ch.masked_fill(is_audio, 0))
    ae = model.audio_embed(audio_ch)
    sel = (~is_audio).unsqueeze(-1).to(te.dtype)
    return te * sel + ae * (1 - sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/MOSS-Speech")
    ap.add_argument("--case-dir", default="artifacts/p3/reference/t2t_short")
    ap.add_argument("--out", default="artifacts/p3/diag_forward.json")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
    from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig  # noqa: F401
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling.sglang_backend.server_args_builder import (
        build_sglang_server_args,
    )

    builder = MossSpeechEngineBuilder()
    ov = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(ov)
    ov["mem_fraction_static"] = 0.30
    if args.backend:
        ov["attention_backend"] = args.backend
    sa = build_sglang_server_args(args.model_path, context_length=40960, **ov)
    (_, infra) = scheduling_bootstrap.create_sglang_infrastructure_defer_cuda_graph(
        sa, 0, model_arch_override="MossSpeechForCausalLM"
    )
    (mw, _tc, req_pool, kv_alloc, _pm, _dm, mc) = infra
    runner, model = mw.model_runner, mw.model_runner.model

    canon = json.loads((Path(args.case_dir) / "canonical_input.json").read_text())
    prefix = torch.tensor(canon["input_ids"])  # (L, 2)
    L = prefix.shape[0]
    ref = torch.load(Path(args.case_dir) / "step_0000.pt")["raw"]  # concat text|audio
    ref_t, ref_a = ref[:151680], ref[151680:]

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    embeds = build_input_embeds(model, prefix)
    slots = kv_alloc.alloc(L)
    slots_dev = slots.to(runner.device, dtype=torch.int64).flatten()
    runner.req_to_token_pool.write((0, slice(0, L)), slots_dev)
    fb = object.__new__(ForwardBatch)
    fb.forward_mode = ForwardMode.EXTEND
    fb.batch_size = 1
    fb.input_ids = torch.zeros(L, dtype=torch.int64, device=runner.device)
    fb.req_pool_indices = torch.zeros(1, dtype=torch.int64, device=runner.device)
    fb.seq_lens = torch.tensor([L], dtype=torch.int64, device=runner.device)
    fb.seq_lens_cpu = torch.tensor([L], dtype=torch.int64)
    fb.seq_lens_sum = L
    fb.out_cache_loc = slots.to(runner.device)
    fb.positions = torch.arange(L, dtype=torch.int64, device=runner.device)
    fb.extend_seq_lens = fb.seq_lens.clone()
    fb.extend_prefix_lens = torch.zeros_like(fb.seq_lens)
    fb.extend_seq_lens_cpu = fb.extend_seq_lens.cpu()
    fb.extend_prefix_lens_cpu = fb.extend_prefix_lens.cpu()
    fb.extend_seq_lens_sum = L
    fb.device = runner.device

    rep: dict = {"L": L, "backend": str(type(runner.attn_backend).__name__)}
    acts: dict = {}
    hooks = []

    def snap(name):
        def hook(mod, inp, out):
            with torch.no_grad():
                if isinstance(out, tuple):
                    t = out[0]
                else:
                    t = out
                if torch.is_tensor(t):
                    acts.setdefault(name, []).append(
                        [float(t.float().norm()), float(t.float().abs().mean())]
                    )

        return hook

    hooks.append(model.layers[0].register_forward_hook(snap("trunk_l0")))
    hooks.append(model.layers[-1].register_forward_hook(snap("trunk_l31")))
    hooks.append(model.text_block[-1].register_forward_hook(snap("text_l3")))
    hooks.append(model.audio_block[-1].register_forward_hook(snap("audio_l3")))

    with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        runner.attn_backend.init_forward_metadata(forward_batch=fb)
        out = model(
            input_ids=fb.input_ids,
            positions=fb.positions,
            forward_batch=fb,
            input_embeds=embeds.to(runner.device),
        )
    for h in hooks:
        h.remove()
    rep["acts"] = acts
    nt, na = out.text_logits[0].float().cpu(), out.audio_logits[0].float().cpu()

    def topk_overlap(a, b, k=20):
        return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist()))

    rep["embed_norm"] = float(embeds.float().norm())
    # minimal discriminator: VocabParallelEmbedding.forward vs direct weight indexing
    with torch.no_grad():
        two = torch.tensor([151644, 77091], device=runner.device)
        via_forward = model.embed_tokens(two)
        via_index = model.embed_tokens.weight[two]
        rep["embed_discriminator"] = {
            "forward_rows_norm": [
                float(via_forward[i].float().norm()) for i in range(2)
            ],
            "index_rows_norm": [float(via_index[i].float().norm()) for i in range(2)],
            "equal": bool(torch.equal(via_forward, via_index)),
            "forward_max_minus_index": float((via_forward - via_index).abs().max()),
        }
    print("DISCRIMINATOR", rep["embed_discriminator"], flush=True)
    rep["text_top5_native"] = nt.topk(5).indices.tolist()
    rep["text_top5_ref"] = ref_t.topk(5).indices.tolist()
    rep["audio_top5_native"] = na.topk(5).indices.tolist()
    rep["audio_top5_ref"] = ref_a.topk(5).indices.tolist()
    rep["top20_overlap"] = {
        "text": topk_overlap(nt, ref_t),
        "audio": topk_overlap(na, ref_a),
    }
    rep["max_abs"] = {
        "text": float((nt - ref_t).abs().max()),
        "audio": float((na - ref_a).abs().max()),
    }
    rep["native_text_absmean"] = float(nt.abs().mean())
    rep["ref_text_absmean"] = float(ref_t.abs().mean())

    # ---- layer-by-layer bisect vs exported reference hiddens ------------
    ref_blob = torch.load("artifacts/p3/diag/ref_hidden.pt", map_location="cpu")

    caps: dict[str, torch.Tensor] = {}

    def snap_full(name):
        def hook(mod, inp, out):
            with torch.no_grad():
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    caps[name] = t.detach().float().cpu()

        return hook

    hs = [
        model.layers[0].register_forward_hook(snap_full("shared_0")),
        model.layers[1].register_forward_hook(snap_full("shared_1")),
        model.layers[31].register_forward_hook(snap_full("shared_31")),
        model.text_block[3].register_forward_hook(snap_full("text_last")),
        model.audio_block[3].register_forward_hook(snap_full("audio_last")),
    ]
    embeds_cpu = embeds.detach().float().cpu()
    with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        runner.attn_backend.init_forward_metadata(forward_batch=fb)
        out2 = model(
            input_ids=fb.input_ids,
            positions=fb.positions,
            forward_batch=fb,
            input_embeds=embeds.to(runner.device),
        )
    for h in hs:
        h.remove()

    def cmp(name, native_t, ref_t):
        if native_t is None:
            return None
        n, r = native_t.float().cpu(), ref_t.float().cpu()
        if r.dim() == 3:
            r = r[0]
        if n.dim() == 3:
            n = n[0]
        L = min(n.shape[0], r.shape[0])
        n, r = n[L - 1], r[L - 1]  # last position row
        cos = torch.nn.functional.cosine_similarity(n, r, dim=0)
        cos = cos.mean() if cos.numel() > 1 else cos
        return {
            "cos_last": float(cos),
            "max_abs": float((n - r).abs().max()),
            "n_norm": float(n.norm()),
            "r_norm": float(r.norm()),
        }

    bisect = {"embeds": cmp("embeds", embeds_cpu, ref_blob["embeds"])}
    for k in ("shared_0", "shared_1", "shared_31", "text_last", "audio_last"):
        bisect[k] = cmp(k, caps.get(k), ref_blob[k])
    # logits
    bisect["text_logits"] = cmp("tl", nt, ref_blob["text_logits"])
    bisect["audio_logits"] = cmp("al", na, ref_blob["audio_logits"])
    rep["bisect"] = bisect

    # ---- direct weight verification vs checkpoint ------------------------
    import json as _json

    from safetensors import safe_open

    mp = Path(args.model_path)
    idx = _json.loads((mp / "model.safetensors.index.json").read_text())["weight_map"]
    wrep = {}
    for ck, attr in (
        ("model.embed_tokens.weight", "embed_tokens"),
        ("model.audio_embed.weight", "audio_embed"),
        ("text_lm_head.weight", "text_lm_head"),
    ):
        with safe_open(mp / idx[ck], framework="pt", device="cpu") as f:
            ck_w = f.get_tensor(ck)
        native_w = getattr(model, attr).weight.detach().float().cpu()
        wrep[attr] = {
            "ckpt_shape": list(ck_w.shape),
            "native_shape": list(native_w.shape),
            "rows_equal": bool(
                torch.equal(
                    native_w[:3].to(ck_w.dtype),
                    (
                        ck_w[:3].to(native_w.dtype)
                        if False
                        else ck_w[:3].to(native_w.dtype)
                    ),
                )
            ),
            "first_row_max_abs": float((native_w[0] - ck_w[0].float()).abs().max()),
            "native_norm": float(native_w.norm()),
            "ckpt_norm": float(ck_w.float().norm()),
        }
    rep["weights"] = wrep
    print(json.dumps(wrep, indent=1))

    # ---- layer-tuple bisect: delta+residual == HF full stream ----------
    def snap_pair(name):
        def hook(mod, inp, out):
            with torch.no_grad():
                if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
                    caps2[name] = (out[0] + out[1]).detach().float().cpu()
                    caps2[name + "__h"] = float(out[0].detach().float().norm())
                    caps2[name + "__r"] = float(out[1].detach().float().norm())
                    if name == "shared_0":
                        r0 = out[1].detach().float().cpu()
                        h0 = out[0].detach().float().cpu()
                        caps2["r0_dump"] = {
                            "r_row_norms_head": [float(x) for x in r0.norm(dim=-1)[:6]],
                            "r_first8": [float(x) for x in r0[0, :8]],
                            "h_first8": [float(x) for x in h0[0, :8]],
                            "r_dtype": str(out[1].dtype),
                            "r_shape": list(out[1].shape),
                        }

        return hook

    caps2: dict = {}
    hp = [
        model.layers[0].register_forward_hook(snap_pair("shared_0")),
        model.layers[1].register_forward_hook(snap_pair("shared_1")),
        model.layers[31].register_forward_hook(snap_pair("shared_31")),
        model.text_block[3].register_forward_hook(snap_pair("text_last")),
        model.audio_block[3].register_forward_hook(snap_pair("audio_last")),
    ]
    with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        runner.attn_backend.init_forward_metadata(forward_batch=fb)
        model(
            input_ids=fb.input_ids,
            positions=fb.positions,
            forward_batch=fb,
            input_embeds=embeds.to(runner.device),
        )
    for h in hp:
        h.remove()

    bisect2 = {}
    for k, rv in (
        ("shared_0", ref_blob["shared_0"][0]),
        ("shared_1", ref_blob["shared_1"][0]),
        ("shared_31", ref_blob["shared_31"][0]),
        ("text_last", ref_blob["text_last"][0]),
        ("audio_last", ref_blob["audio_last"][0]),
    ):
        n = caps2.get(k)
        if n is None:
            bisect2[k] = None
            continue
        L = min(n.shape[0], rv.shape[0])
        cos = torch.nn.functional.cosine_similarity(n[L - 1], rv[L - 1].float(), dim=0)
        bisect2[k] = {
            "cos_last": float(cos),
            "max_abs": float((n[L - 1] - rv[L - 1].float()).abs().max()),
            "n_row_norm": float(n[L - 1].norm()),
            "r_row_norm": float(rv[L - 1].float().norm()),
        }
    rep["layer_bisect"] = bisect2
    rep["layer_hr_norms"] = {
        k: v for k, v in caps2.items() if k.endswith("__h") or k.endswith("__r")
    }
    print("LAYER_HR", json.dumps(rep["layer_hr_norms"]), flush=True)
    print("R0_DUMP", json.dumps(caps2.get("r0_dump")), flush=True)

    # ---- weight audit: native layers[0] vs checkpoint, element-wise -------
    from safetensors import safe_open as _sopen

    audit = {}
    with torch.no_grad():
        ck_names = {
            "qkv": "model.shared_block.layers.0.self_attn.q_proj.weight",
            "k": "model.shared_block.layers.0.self_attn.k_proj.weight",
            "v": "model.shared_block.layers.0.self_attn.v_proj.weight",
            "o": "model.shared_block.layers.0.self_attn.o_proj.weight",
            "qn": "model.shared_block.layers.0.self_attn.q_norm.weight",
            "ln1": "model.shared_block.layers.0.input_layernorm.weight",
            "gate": "model.shared_block.layers.0.mlp.gate_proj.weight",
            "down": "model.shared_block.layers.0.mlp.down_proj.weight",
        }
        ckT = {}
        for k2, full in ck_names.items():
            with _sopen(mp / idx[full], framework="pt", device="cpu") as f:
                ckT[k2] = f.get_tensor(full).float()
        nv = dict(model.layers[0].named_parameters())
        fused = nv["self_attn.qkv_proj.weight"].detach().float().cpu()
        audit["qkv_q_match"] = bool(torch.equal(fused[:4096], ckT["qkv"]))
        audit["qkv_k_match"] = bool(torch.equal(fused[4096:5120], ckT["k"]))
        audit["qkv_v_match"] = bool(torch.equal(fused[5120:6144], ckT["v"]))
        audit["o_match"] = bool(
            torch.equal(nv["self_attn.o_proj.weight"].detach().float().cpu(), ckT["o"])
        )
        audit["qnorm_match"] = bool(
            torch.equal(nv["self_attn.q_norm.weight"].detach().float().cpu(), ckT["qn"])
        )
        audit["ln1_match"] = bool(
            torch.equal(nv["input_layernorm.weight"].detach().float().cpu(), ckT["ln1"])
        )
        gu = nv["mlp.gate_up_proj.weight"].detach().float().cpu()
        audit["gate_match"] = bool(torch.equal(gu[:12288], ckT["gate"]))
        audit["down_match"] = bool(
            torch.equal(nv["mlp.down_proj.weight"].detach().float().cpu(), ckT["down"])
        )
        audit["fused_shape"] = list(fused.shape)
    rep["weight_audit_l0"] = audit
    print("WEIGHT_AUDIT_L0", json.dumps(audit), flush=True)

    # ---- intra-layer activation norms (layer 0) --------------------------
    intra = {}

    def snapnorm(name):
        def hook(mod, inp, out):
            with torch.no_grad():
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    intra[name] = float(t.detach().float().norm())

        return hook

    hln = model.layers[0].input_layernorm.register_forward_hook(snapnorm("ln1_out"))
    hat = model.layers[0].self_attn.register_forward_hook(snapnorm("attn_out"))
    hqk = model.layers[0].self_attn.qkv_proj.register_forward_hook(snapnorm("qkv_out"))
    hmlp = model.layers[0].mlp.register_forward_hook(snapnorm("mlp_out"))
    hrope = model.layers[0].self_attn.rotary_emb.register_forward_hook(
        snapnorm("rope_out")
    )
    with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        runner.attn_backend.init_forward_metadata(forward_batch=fb)
        model(
            input_ids=fb.input_ids,
            positions=fb.positions,
            forward_batch=fb,
            input_embeds=embeds.to(runner.device),
        )
    for h in (hln, hat, hqk, hmlp, hrope):
        h.remove()
    rep["intra_l0"] = intra
    print("INTRA_L0", json.dumps(intra), flush=True)
    print("LAYER_BISECT", json.dumps(bisect2, indent=1), flush=True)

    # ---- layer bisect via causality-safe runner -------------------------
    def run_rows_capture(n_rows: int, layer_indices: list) -> dict:
        rows = prefix[:n_rows]
        e = build_input_embeds(model, rows)
        sl = kv_alloc.alloc(n_rows)
        sd = sl.to(runner.device, dtype=torch.int64).flatten()
        runner.req_to_token_pool.write((0, slice(0, n_rows)), sd)
        fb2 = object.__new__(ForwardBatch)
        fb2.forward_mode = ForwardMode.EXTEND
        fb2.batch_size = 1
        fb2.input_ids = torch.zeros(n_rows, dtype=torch.int64, device=runner.device)
        fb2.req_pool_indices = torch.zeros(1, dtype=torch.int64, device=runner.device)
        fb2.seq_lens = torch.tensor([n_rows], dtype=torch.int64, device=runner.device)
        fb2.seq_lens_cpu = torch.tensor([n_rows], dtype=torch.int64)
        fb2.seq_lens_sum = n_rows
        fb2.out_cache_loc = sd
        fb2.positions = torch.arange(n_rows, dtype=torch.int64, device=runner.device)
        fb2.extend_seq_lens = fb2.seq_lens.clone()
        fb2.extend_prefix_lens = torch.zeros_like(fb2.seq_lens)
        fb2.extend_seq_lens_cpu = fb2.extend_seq_lens.cpu()
        fb2.extend_prefix_lens_cpu = fb2.extend_prefix_lens.cpu()
        fb2.extend_seq_lens_sum = n_rows
        fb2.device = runner.device
        caps3: dict = {}
        hooks = [
            model.layers[li].register_forward_hook(
                (
                    lambda li: lambda m, i, o: caps3.__setitem__(
                        li, (o[0] + o[1]).detach().float().cpu()
                    )
                )(li)
            )
            for li in layer_indices
        ]
        with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            runner.attn_backend.init_forward_metadata(forward_batch=fb2)
            model(
                input_ids=fb2.input_ids,
                positions=fb2.positions,
                forward_batch=fb2,
                input_embeds=e.to(runner.device),
            )
        for h in hooks:
            h.remove()
        kv_alloc.free(sl.to(torch.int64))
        return caps3

    short_caps = run_rows_capture(6, [0])
    full_caps = run_rows_capture(34, [0, 1, 2, 31])
    row5_short, row5_full = short_caps[0][5], full_caps[0][5]
    rep["causality_probe"] = {
        "row5_max_abs_short_vs_full": float((row5_short - row5_full).abs().max()),
        "row5_bit_equal": bool(torch.equal(row5_short, row5_full)),
    }
    ref_layers = {
        0: ref_blob["shared_0"][0],
        1: ref_blob["shared_1"][0],
        2: None,
        31: ref_blob["shared_31"][0],
    }
    layer_stats = {}
    for li in (0, 1, 2, 31):
        native = full_caps[li]
        stats = []
        for r in (0, 5, 33):
            rr = ref_layers[li]
            if rr is None:
                stats.append({"row": r, "note": "no ref"})
                continue
            cos = torch.nn.functional.cosine_similarity(native[r], rr[r].float(), dim=0)
            stats.append(
                {
                    "row": r,
                    "cos": round(float(cos), 5),
                    "n_norm": round(float(native[r].norm()), 3),
                    "r_norm": round(float(rr[r].float().norm()), 3),
                }
            )
        layer_stats[f"layer{li}"] = stats
    rep["layer_bisect_rows"] = layer_stats
    print("LAYER_BISECT_ROWS", json.dumps(layer_stats), flush=True)
    print("CAUSALITY", rep["causality_probe"], flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(bisect, indent=1))
    print("DIAG DONE")


if __name__ == "__main__":
    main()
