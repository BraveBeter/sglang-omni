#!/usr/bin/env python3
"""Real HTTP quality/CI driver using the unchanged native model stage factories."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import torch
import uvicorn
from quality import build_cases, sha256, validate_pairs

from tests.test_model.moss_speech_ci_config import MossSpeechCiPreset


async def run(args: Any, report: dict[str, Any]) -> None:
    from sglang_omni.client import Client
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.serve.openai_api import create_app

    preset = MossSpeechCiPreset()
    cases = build_cases(args.manifest)
    reference = json.loads(args.reference.read_text())
    assert reference["pass"] and reference["manifest_sha256"] == sha256(args.manifest)
    config = ConfigManager.from_file(str(args.config)).config
    runner = MultiProcessPipelineRunner(config)
    states, processes = {}, {}
    server = server_task = None
    running = True
    report["memory"] = []
    report["quality_gate_thresholds"] = preset.gate_thresholds
    report["quality_calibrated"] = preset.calibrated

    async def monitor() -> None:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = None
            while running:
                for group in runner._groups:
                    for proc in group.processes:
                        if proc.pid:
                            processes[proc.pid] = proc
                if handle is None:
                    for index in range(pynvml.nvmlDeviceGetCount()):
                        candidate = pynvml.nvmlDeviceGetHandleByIndex(index)
                        if any(
                            p.pid in processes
                            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(
                                candidate
                            )
                        ):
                            handle = candidate
                            report["gpu"] = pynvml.nvmlDeviceGetName(handle)
                            report["gpu_uuid"] = pynvml.nvmlDeviceGetUUID(handle)
                            break
                if handle is not None:
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    report["memory"].append(
                        dict(t=time.time(), used=memory.used, total=memory.total)
                    )
                await asyncio.sleep(0.5)
        finally:
            pynvml.nvmlShutdown()

    monitoring = asyncio.create_task(monitor())

    def capture(rid: str, result: Any) -> Any:
        state = result.data if hasattr(result, "data") else result
        states[rid] = state
        return Client._default_result_builder(rid, state)

    try:
        await runner.start(timeout=preset.startup_timeout)
        client = Client(runner.coordinator, result_builder=capture)
        app = create_app(
            client,
            model_name=config.name,
            chat_request_validator=config.validate_chat_request,
            supports_uploaded_voice_references=config.supports_uploaded_voice_references(),
            architectures=[config.architecture],
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=10)
        )
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("HTTP server exited before readiness")
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
            timeout=preset.request_timeout,
        ) as http:
            assert (await http.get("/health")).status_code == 200
            for case in cases:
                rid = case["id"]
                row = dict(id=rid, mode=case["mode"], lang=case["lang"])
                try:
                    start = time.monotonic()
                    response = await http.post(
                        preset.endpoint, json={**case["body"], "request_id": rid}
                    )
                    row["seconds"] = time.monotonic() - start
                    response.raise_for_status()
                    data = response.json()
                    (args.out_dir / f"{rid}.json").write_text(
                        json.dumps(data, indent=2) + "\n"
                    )
                    state = states[rid]
                    choice = data["choices"][0]
                    row.update(
                        input_grid=state["input_grid"],
                        grid=state["output_grid"],
                        text=state["generated_text"],
                        finish_reason=choice["finish_reason"],
                    )
                    if case["mode"].endswith("s"):
                        wav = args.out_dir / f"{rid}.wav"
                        wav.write_bytes(
                            base64.b64decode(choice["message"]["audio"]["data"])
                        )
                        row["waveform_sha256"] = sha256(wav)
                    torch.save(state, args.out_dir / f"{rid}.pt")
                except Exception as exc:
                    row["error"] = repr(exc)
                report["results"].append(row)
                print(
                    rid,
                    len(row.get("grid", [])),
                    row.get("error", row.get("finish_reason")),
                    flush=True,
                )
                (args.out_dir / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            report["futures_empty"] = not runner.coordinator._completion_futures
    finally:
        try:
            if server:
                server.should_exit = True
            if server_task:
                await asyncio.wait_for(server_task, 30)
        finally:
            await runner.stop()
            running = False
            await monitoring
            report["processes"] = [
                dict(pid=p.pid, alive=p.is_alive(), exitcode=p.exitcode)
                for p in processes.values()
            ]
            report["processes_exit"] = len(processes) == 4 and all(
                not p.is_alive() and p.exitcode == 0 for p in processes.values()
            )
    report["pairs"] = validate_pairs(
        reference["results"], report["results"], [c["id"] for c in cases]
    )
    # Decode the independent HF grids with the already qualified P1 decoder,
    # using the same configured voice. The native HTTP grids are never replaced.
    from sglang_omni.client.audio import audio_to_base64
    from sglang_omni.models.moss_speech.stages import create_audio_vocoder_executor
    from sglang_omni.proto.request import StagePayload

    stage = next(s for s in config.stages if s.name == "audio_vocoder")
    vocoder = create_audio_vocoder_executor(config.model_path, **stage.factory_args)
    ref_wav_dir = args.out_dir / "reference_wav"
    ref_wav_dir.mkdir()
    wave_checks = {}
    for row in reference["results"]:
        rid = row["id"]
        if not row["mode"].endswith("s") or rid not in states or row.get("error"):
            continue
        state = {**states[rid], "output_grid": row["grid"], "audio_samples": None}
        decoded = vocoder._fn(
            StagePayload(
                request_id=f"reference-{rid}", request=SimpleNamespace(), data=state
            )
        ).data
        wav = ref_wav_dir / f"{rid}.wav"
        wav.write_bytes(
            base64.b64decode(
                audio_to_base64(
                    decoded["audio_samples"], sample_rate=24000, output_format="wav"
                )
            )
        )
        wave_checks[rid] = sha256(wav) == sha256(args.out_dir / f"{rid}.wav")
    report["waveforms_equal"] = len(wave_checks) == len(cases) // 2 and all(
        wave_checks.values()
    )
    report["waveform_checks"] = wave_checks
    report["pass"] = (
        all(
            report.get(key)
            for key in ("futures_empty", "processes_exit", "waveforms_equal")
        )
        and report["pairs"]["pass"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        pass_=False,
        manifest_sha256=sha256(args.manifest),
        reference_sha256=sha256(args.reference),
        config_sha256=sha256(args.config),
        results=[],
    )
    report["pass"] = report.pop("pass_")
    repo = Path(__file__).resolve().parents[3]
    source_paths = [
        *sorted((repo / "scripts/moss_speech/p5").glob("*.py")),
        repo / "scripts/moss_speech/ci/run_gpu.sh",
        repo / "tests/test_model/moss_speech_ci_config.py",
        *sorted((repo / "sglang_omni/models/moss_speech").rglob("*.py")),
    ]
    report["source_sha256"] = {
        "sglang-omni/" + str(path.relative_to(repo)): sha256(path)
        for path in source_paths
    }
    try:
        asyncio.run(run(args, report))
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
