#!/usr/bin/env python3
"""Formal YAML native multiprocess takeover. No fixture injection or AR stub."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def chat(text: Any, *, audio: Any = False, limit: Any = 200) -> Any:
    from sglang_omni.proto.request import OmniRequest
    from sglang_omni.serve.openai_api import _build_chat_generate_request
    from sglang_omni.serve.protocol import ChatCompletionRequest

    system = (
        "You are a helpful voice assistant. Answer the user's questions with spoken responses."
        if audio
        else "You are a helpful assistant. Answer the user's questions with text."
    )
    request = ChatCompletionRequest(
        model="moss-speech",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        modalities=["audio"] if audio else ["text"],
        seed=0,
        temperature=0.0,
        top_p=1.0,
        max_tokens=limit,
        repetition_penalty=1.1,
        top_k=-1,
    )
    return OmniRequest(inputs=_build_chat_generate_request(request).to_dict())


def blake(samples: Any) -> Any:
    return hashlib.blake2b(
        np.asarray(samples, dtype=np.float32).tobytes(), digest_size=16
    ).hexdigest()


def events(path: Any) -> Any:
    result = []
    for file in path.glob("events_*.jsonl"):
        for line in file.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # An actively appended last line.
    return result


async def run(args: Any, report: Any) -> Any:
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.profiler.profiler_control import ProfilerControlClient

    config = ConfigManager.from_file(args.config).config
    ar = next(s for s in config.stages if s.name == "ar_engine")
    assert (
        ar.factory == "sglang_omni.models.moss_speech.stages.create_ar_engine_executor"
    )
    report["ar_factory"] = ar.factory
    if args.startup_failure:
        ar.factory_args["dtype"] = "float32"
    runner = MultiProcessPipelineRunner(config)
    processes = {}
    collecting = True

    async def monitor() -> Any:
        while collecting:
            for group in runner._groups:
                for proc in group.processes:
                    if proc.pid:
                        processes[proc.pid] = proc
            await asyncio.sleep(0.02)

    monitor_task = asyncio.create_task(monitor())
    profiler = None
    audio_result = None
    try:
        try:
            await runner.start(timeout=600)
        except Exception as exc:
            if not args.startup_failure:
                raise
            report["startup_error"] = str(exc)
            assert "BF16" in str(exc) or "bfloat16" in str(exc), str(exc)
            report["checks"]["expected_startup_rejection"] = True
            return
        assert not args.startup_failure, "invalid dtype unexpectedly started"
        report["ready"] = True
        profiler = ProfilerControlClient(runner.stage_control_endpoints)
        await profiler.broadcast_start(
            run_id=Path(args.out_dir).name,
            trace_path_template=str(Path(args.out_dir) / "unused"),
            event_dir=str(Path(args.out_dir) / "events"),
            enable_torch=False,
        )

        async def submit(rid: Any, request: Any) -> Any:
            result = await asyncio.wait_for(
                runner.coordinator.submit(rid, request), timeout=600
            )
            data = result.data if hasattr(result, "data") else result
            torch.save(data, Path(args.out_dir) / (rid + ".pt"))
            return data

        text_prompt = "Introduce yourself in one sentence."
        audio_prompt = "用中文介绍一下上海的三到四个著名景点。"
        for case, prompt, audio in [
            ("t2t_short", text_prompt, False),
            ("t2s_cn", audio_prompt, True),
        ]:
            data = await submit(case, chat(prompt, audio=audio))
            ref = Path("artifacts/p3/reference") / case
            expected = torch.load(ref / "tokens_grid.pt", weights_only=True)
            canonical = json.loads((ref / "canonical_input.json").read_text())
            report["checks"][case + "_input"] = (
                data["input_grid"] == canonical["input_ids"]
            )
            report["checks"][case + "_grid"] = torch.equal(
                torch.tensor(data["output_grid"]), expected
            )
            report["checks"][case + "_seed"] = data["effective_seed"] == 0
            report["checks"][case + "_terminal"] = bool(
                data.get("audio_samples") if audio else data.get("generated_text")
            )
            if audio:
                audio_result = data
            print(
                case,
                {k: v for k, v in report["checks"].items() if k.startswith(case)},
                flush=True,
            )
        # Overlapping requests through the actual coordinator and child AR.
        first, second = await asyncio.gather(
            submit("parallel-1", chat(text_prompt)),
            submit("parallel-2", chat(text_prompt)),
        )
        report["checks"]["parallel_equal"] = (
            first["output_grid"]
            == second["output_grid"]
            == torch.load(
                "artifacts/p3/reference/t2t_short/tokens_grid.pt", weights_only=True
            ).tolist()
        )
        # Wait for a real AR prefill-end event, then abort while decoding.
        victim = asyncio.create_task(
            submit("cancel-native", chat(audio_prompt, audio=True))
        )
        survivor = asyncio.create_task(
            submit("survivor-native", chat(audio_prompt, audio=True))
        )
        deadline = time.monotonic() + 90
        matched = []
        while time.monotonic() < deadline:
            matched = [
                e
                for e in events(Path(args.out_dir) / "events")
                if e["request_id"] == "cancel-native"
                and e["stage"] == "ar_engine"
                and e["event_name"] == "scheduler_prefill_end"
            ]
            if matched:
                break
            await asyncio.sleep(0.02)
        assert matched, "AR prefill event not observed before cancellation"
        report["abort_event"] = matched[0]
        report["checks"]["abort_ack"] = await runner.coordinator.abort("cancel-native")
        try:
            await asyncio.wait_for(victim, timeout=60)
            report["checks"]["abort_finalized"] = False
        except asyncio.CancelledError:
            report["checks"]["abort_finalized"] = True
        surviving = await survivor
        report["checks"]["survivor_grid"] = (
            surviving["output_grid"] == audio_result["output_grid"]
        )
        report["checks"]["survivor_audio"] = blake(surviving["audio_samples"]) == blake(
            audio_result["audio_samples"]
        )
        recovered = await submit("recovered-native", chat(text_prompt))
        report["checks"]["recovered_grid"] = (
            recovered["output_grid"] == first["output_grid"]
        )
    finally:
        if profiler:
            await profiler.close()
        # Capture before stop clears groups; monitor also covers start failures.
        for group in runner._groups:
            for proc in group.processes:
                if proc.pid:
                    processes[proc.pid] = proc
        await runner.stop()
        collecting = False
        await monitor_task
        report["processes"] = [
            {"pid": pid, "exitcode": proc.exitcode, "alive": proc.is_alive()}
            for pid, proc in processes.items()
        ]
        report["checks"]["children_exited"] = len(processes) == 4 and all(
            not p.is_alive() and p.exitcode is not None for p in processes.values()
        )
    if audio_result:
        # Expected waveform is decoded from the frozen HF grid using the P1
        # component and the same conditioning/seed, after child GPU release.
        from sglang_omni.models.moss_speech.stages import create_audio_vocoder_executor
        from sglang_omni.proto.request import StagePayload

        expected_state = dict(audio_result)
        expected_state["output_grid"] = torch.load(
            "artifacts/p3/reference/t2s_cn/tokens_grid.pt", weights_only=True
        ).tolist()
        expected_state["audio_samples"] = None
        vocoder = create_audio_vocoder_executor(config.model_path)
        expected = vocoder._fn(
            StagePayload(
                request_id="reference-codec",
                request=SimpleNamespace(),
                data=expected_state,
            )
        ).data
        report["checks"]["audio_reference_hash"] = blake(
            expected["audio_samples"]
        ) == blake(audio_result["audio_samples"])
        report["checks"]["audio_sample_rate"] = (
            audio_result["audio_sample_rate"] == 24000
        )
        report["audio_hash"] = blake(audio_result["audio_samples"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sglang-omni/examples/configs/moss_speech.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--startup-failure", action="store_true")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=False)
    report = {"pass": False, "checks": {}, "startup_failure_test": args.startup_failure}
    try:
        asyncio.run(run(args, report))
        report["pass"] = bool(report["checks"]) and all(report["checks"].values())
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (Path(args.out_dir) / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
