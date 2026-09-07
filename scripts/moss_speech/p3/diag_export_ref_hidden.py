#!/usr/bin/env python3
"""T3.3 diagnostic (reference side, .venv-p0): export per-stage hidden states
for one t2t prompt from the HF bf16 model for layer-by-layer bisecting."""
from __future__ import annotations
import json, os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import torch
from transformers import AutoModel

def main() -> None:
    model = AutoModel.from_pretrained(
        "models/MOSS-Speech", trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda")
    model.eval()
    canon = json.loads(Path("artifacts/p3/reference/t2t_short/canonical_input.json").read_text())
    ids = torch.tensor(canon["input_ids"])[None].cuda()  # (1, L, 2)
    L = ids.shape[1]

    caps: dict[str, torch.Tensor] = {}

    inner = model.model  # MossSpeechModel
    orig_embed_call = inner.__class__.forward

    def mk(name):
        def hook(mod, inp, out):
            with torch.no_grad():
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    caps.setdefault(name, []).append(t.detach().float().cpu())
        return hook

    hooks = [model.model.shared_block.layers[i].register_forward_hook(mk(f"shared_{i}"))
             for i in (0, 1, 31)]
    hooks.append(model.model.text_block.layers[3].register_forward_hook(mk("text_last")))
    hooks.append(model.model.audio_block.layers[3].register_forward_hook(mk("audio_last")))

    with torch.no_grad():
        # embeddings: call the packed-embed path directly
        text_ids = ids[:, :, 0]
        audio_ids = ids[:, :, 1]
        pad = int(getattr(model.config, "modality_pad_token_id", 151667))
        text_safe = text_ids.masked_fill(text_ids == pad, 0)
        te = model.model.embed_tokens(text_safe)
        ae = model.model.audio_embed(audio_ids)
        sel = (text_ids != pad).unsqueeze(-1).to(te.dtype)
        caps["embeds"] = [(te * sel + ae * (1 - sel)).detach().float().cpu()]
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1, return_dict=True)
        # tail final norms: recompute from captured tail outputs? hook the norms instead
    for h in hooks:
        h.remove()

    # final normed hiddens: run norms manually on captured tail outputs? The
    # tail hook captures the transformer-block output (hidden, weight?) --
    # MossSpeechTransformerBlock returns tensor; recompute norm:
    tl = caps["text_last"][0]
    al = caps["audio_last"][0]
    with torch.no_grad():
        t_normed = model.model.text_norm(tl.cuda().to(torch.bfloat16)).float().cpu()
        a_normed = model.model.audio_norm(al.cuda().to(torch.bfloat16)).float().cpu()
    caps["text_normed"] = [t_normed]
    caps["audio_normed"] = [a_normed]
    caps["text_logits"] = [out.logits_all[0][0, -1].float().cpu()]
    caps["audio_logits"] = [out.logits_all[1][0, -1].float().cpu()]

    blob = {k: v[0] for k, v in caps.items()}
    blob["L"] = L
    Path("artifacts/p3/diag").mkdir(parents=True, exist_ok=True)
    torch.save(blob, "artifacts/p3/diag/ref_hidden.pt")
    for k, v in blob.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), "norm", float(v.float().norm()))
    print("REF HIDDEN EXPORTED")

if __name__ == "__main__":
    main()
