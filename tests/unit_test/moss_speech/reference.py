# SPDX-License-Identifier: Apache-2.0
"""Independent text and logits oracles for the model contracts."""
from typing import Any

import torch

TEXT_DECODE_PROFILE = "moss-full-grid-unicode-prefix-v1"


def reference_text_fields(tokenizer: Any, grid: list, *, audio_output: bool) -> dict:
    """Keep the upstream result intact and export service semantics separately."""

    def decode(rows: list) -> str:
        return (
            tokenizer.decode([row[0] for row in rows], skip_special_tokens=True)
            .replace("<|empty|>", ".")
            .replace("<|end_empty|>", ":")
        )

    return dict(
        text="" if audio_output else decode(grid[:-1]),
        service_text="" if audio_output else decode(grid).rstrip("\ufffd"),
        text_decode_profile=TEXT_DECODE_PROFILE,
    )


def compare_logits(native: Any, reference: Any) -> Any:
    """Frozen v1 per-head metrics, rejecting NaNs and nonfinite mismatches."""
    native, reference = native.float(), reference.float()
    if native.shape != reference.shape:
        return {
            "pass": False,
            "shape_mismatch": [list(native.shape), list(reference.shape)],
        }
    finite = torch.isfinite(reference)
    pattern = bool(
        torch.equal(torch.isfinite(native), finite)
        and torch.equal(torch.isposinf(native), torch.isposinf(reference))
        and torch.equal(torch.isneginf(native), torch.isneginf(reference))
        and not native.isnan().any()
        and not reference.isnan().any()
    )
    delta = (native[finite] - reference[finite]).abs()
    bad = delta > 1.0 + 0.02 * reference[finite].abs()
    frac = float(bad.float().mean()) if bad.numel() else 0.0
    max_abs = float(delta.max()) if delta.numel() else 0.0
    return {
        "pass": pattern and frac <= 1e-4 and max_abs <= 4.5,
        "nonfinite_match": pattern,
        "max_abs": max_abs,
        "exceed_fraction": frac,
        "exceed_count": int(bad.sum()),
    }
