#!/usr/bin/env python3
"""CPU config/registry/DAG/preflight regression after native P3 takeover.

This command does not initialize the AR engine. The native GPU factory is
validated separately by scripts/moss_speech/p3/takeover.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

REPORT: dict = {"sections": {}}


def _ok(section: str, detail: dict) -> None:
    REPORT["sections"][section] = {"pass": True, **detail}


def _fail(section: str, detail: dict) -> None:
    REPORT["sections"][section] = {"pass": False, **detail}
    REPORT["pass"] = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="examples/configs/moss_speech.yaml"
    )
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    REPORT["pass"] = True

    # ---------------- A1: config load via the CLI path ---------------------
    from sglang_omni.config.manager import ConfigManager

    try:
        manager = ConfigManager.from_file(args.config)
        config = manager.config
        _ok(
            "yaml_load",
            {
                "config_cls": type(config).__name__,
                "entry": config.resolved_entry_stage,
                "terminals": config.terminal_stages,
                "stages": [s.name for s in config.stages],
            },
        )
    except Exception as exc:  # noqa: BLE001
        _fail("yaml_load", {"error": f"{type(exc).__name__}: {exc}"})
        _write(args)
        return

    # ---------------- A2: registry resolution by model_path -----------------
    try:
        from sglang_omni.config.manager import resolve_config_cls_for_model_path
        from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig

        resolved = resolve_config_cls_for_model_path(config.model_path)
        assert resolved is MossSpeechPipelineConfig, f"registry resolved {resolved!r}"
        _ok("registry_resolution", {"resolved": resolved.__name__})
    except Exception as exc:  # noqa: BLE001
        _fail("registry_resolution", {"error": f"{type(exc).__name__}: {exc}"})

    # ---------------- A3: DAG shape assertions -------------------------------
    try:
        names = {s.name for s in config.stages}
        assert {"preprocessing", "ar_engine", "text_decode", "audio_vocoder"} == names
        ar = next(s for s in config.stages if s.name == "ar_engine")
        ar_next = ar.next if isinstance(ar.next, list) else [ar.next]
        assert set(ar_next) == {"text_decode", "audio_vocoder"}
        assert ar.route_fn and ar.terminal is not True
        assert all(
            next(s for s in config.stages if s.name == n).terminal
            for n in ("text_decode", "audio_vocoder")
        )
        processes = {s.process for s in config.stages}
        assert len(processes) == 4, f"expected 4 processes, got {processes}"
        _ok("dag_shape", {"processes": sorted(processes), "route_fn": ar.route_fn})
    except Exception as exc:  # noqa: BLE001
        _fail("dag_shape", {"error": str(exc)})

    # ---------------- A4: hf_config parse of the locked checkpoint ------------
    try:
        from transformers import AutoConfig

        from sglang_omni.models.moss_speech.hf_config import (
            MossSpeechConfig,
            ensure_moss_speech_config_registered,
        )

        ensure_moss_speech_config_registered()
        parsed = AutoConfig.from_pretrained(config.model_path, trust_remote_code=False)
        assert type(parsed) is MossSpeechConfig and parsed.num_hidden_layers == 36
        _ok(
            "hf_config_parse",
            {"type": type(parsed).__name__, "layers": parsed.num_hidden_layers},
        )
    except Exception as exc:  # noqa: BLE001
        _fail("hf_config_parse", {"error": f"{type(exc).__name__}: {exc}"})

    # CPU preflight is separate from the native GPU factory after P3 takeover.
    try:
        from sglang_omni.models.moss_speech.stages import validate_ar_preconditions

        parsed = validate_ar_preconditions(config.model_path, dtype="bfloat16")
        _ok(
            "ar_preflight",
            {"config_type": type(parsed).__name__, "gpu_initialized": False},
        )
    except Exception as exc:
        _fail("ar_preflight", {"error": f"{type(exc).__name__}: {exc}"})

    # ---------------- B: negative configs ----------------------------------------
    negative: dict = {}
    try:
        bad = Path(args.config).read_text().replace("model_path:", "model_pathx:")
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(bad)
            bad_path = fh.name
        try:
            ConfigManager.from_file(bad_path)
            negative["invalid_field"] = "NO ERROR (unexpected)"
        except Exception as exc:  # noqa: BLE001
            negative["invalid_field"] = f"{type(exc).__name__}"
        os.unlink(bad_path)

        from sglang_omni.config import PipelineConfig, StageConfig

        # Framework-validated constructs (schema.py _validate_general):
        try:
            PipelineConfig(
                model_path="m",
                stages=[
                    StageConfig(
                        name="t",
                        process="p",
                        factory="builtins:print",
                        terminal=True,
                        next="x",
                        route_fn="builtins:print",
                    ),
                ],
            )
            negative["terminal_route_fn"] = "NO ERROR (unexpected)"
        except Exception as exc:  # noqa: BLE001
            negative["terminal_route_fn"] = f"{type(exc).__name__}"
        try:
            PipelineConfig(
                model_path="m",
                stages=[
                    StageConfig(
                        name="a", process="p", factory="builtins:print", next="ghost"
                    ),
                ],
            )
            negative["next_missing_stage"] = (
                "NO ERROR (construct-time unknown-target not validated; runtime routing would reject)"
            )
        except Exception as exc:  # noqa: BLE001
            negative["next_missing_stage"] = f"{type(exc).__name__}"
        # NOTE (recorded framework fact): cycles and unknown `next` targets are
        # NOT rejected at construct time by _validate_general; unknown targets
        # are rejected by the runtime routing wrapper (stage_workers). The
        # multi-process smoke exercises the real wrapper on our topology.
        hard_failures = {
            k: v
            for k, v in negative.items()
            if v.startswith("NO ERROR") and k == "terminal_route_fn"
        }
        if hard_failures:
            _fail("negative_configs", {"results": negative})
        else:
            _ok("negative_configs", {"results": negative})
    except Exception as exc:  # noqa: BLE001
        _fail("negative_configs", {"error": f"{type(exc).__name__}: {exc}"})

    _write(args)
    raise SystemExit(0 if REPORT["pass"] else 1)


def _write(args) -> None:
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(REPORT, indent=1))
    print(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    main()
