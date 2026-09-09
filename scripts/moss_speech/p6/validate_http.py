#!/usr/bin/env python3
"""Exercise real SSE through the native multiprocessing streaming variant."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
import torch
import uvicorn

from scripts.moss_speech.p4.validate_http import events
from scripts.moss_speech.p5.quality import (
    TEXT_DECODE_PROFILE,
    build_cases,
    comparison_text,
    sha256,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def run(args: Any, report: dict) -> None:
    from sglang_omni.client import Client
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.profiler.profiler_control import ProfilerControlClient
    from sglang_omni.serve.openai_api import create_app

    cases = build_cases(args.manifest)
    if args.smoke:
        cases = [c for c in cases if c["id"].startswith(("en_00_", "zh_00_"))]
    reference = json.loads(args.reference.read_text())
    refs = {r["id"]: r for r in reference["results"]}
    config = ConfigManager.from_file(str(args.config)).config
    runner = MultiProcessPipelineRunner(config)
    states, processes = {}, {}
    server = server_task = profiler = None
    running = True
    report["memory"] = []

    def save():
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    async def monitor():
        import pynvml

        pynvml.nvmlInit()
        handle = None
        try:
            while running:
                for group in runner._groups:
                    for proc in group.processes:
                        if proc.pid:
                            processes[proc.pid] = proc
                if handle is None:
                    for i in range(pynvml.nvmlDeviceGetCount()):
                        candidate = pynvml.nvmlDeviceGetHandleByIndex(i)
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
                    m = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    report["memory"].append(
                        dict(t=time.time(), used=m.used, total=m.total)
                    )
                await asyncio.sleep(0.5)
        finally:
            pynvml.nvmlShutdown()

    monitoring = asyncio.create_task(monitor())

    def capture(rid, result):
        state = result.data if hasattr(result, "data") else result
        states[rid] = state
        return Client._default_result_builder(rid, state)

    try:
        await runner.start(timeout=600)
        profiler = ProfilerControlClient(runner.stage_control_endpoints)
        await profiler.broadcast_start(
            run_id=args.out_dir.name,
            trace_path_template=str(args.out_dir / "unused"),
            event_dir=str(args.out_dir / "events"),
            enable_torch=False,
        )
        client = Client(runner.coordinator, result_builder=capture)
        app = create_app(
            client,
            model_name=config.name,
            chat_request_validator=config.validate_chat_request,
            supports_uploaded_voice_references=False,
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
                raise RuntimeError("HTTP startup failed")
            await asyncio.sleep(0.02)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=900
        ) as http:

            async def post(case, rid=None, audio_format="pcm", overrides=None):
                rid = rid or case["id"]
                body = {
                    **case["body"],
                    "stream": True,
                    "request_id": rid,
                    **(overrides or {}),
                }
                if case["mode"].endswith("s"):
                    body["audio"] = {"format": audio_format}
                row = dict(
                    id=rid,
                    case_id=case["id"],
                    mode=case["mode"],
                    lang=case["lang"],
                    events=[],
                    audio_format=audio_format,
                )
                start = time.perf_counter_ns()
                pcm = []
                text = ""
                finish = []
                done = 0
                first = None
                try:
                    async with http.stream(
                        "POST", "/v1/chat/completions", json=body
                    ) as response:
                        response.raise_for_status()
                        row["content_type"] = response.headers.get("content-type")
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            elapsed = (time.perf_counter_ns() - start) / 1e9
                            value = line[6:]
                            if value == "[DONE]":
                                done += 1
                                row["events"].append(dict(t=elapsed, done=True))
                                continue
                            event = json.loads(value)
                            if "error" in event:
                                raise RuntimeError(str(event["error"]))
                            choice = event["choices"][0]
                            delta = choice["delta"]
                            entry = dict(
                                t=elapsed, finish_reason=choice.get("finish_reason")
                            )
                            if choice.get("finish_reason"):
                                finish.append(choice["finish_reason"])
                                row["usage"] = event.get("usage")
                            text += delta.get("content") or ""
                            if delta.get("content"):
                                entry["text"] = delta["content"]
                            if delta.get("audio", {}).get("data"):
                                encoded = base64.b64decode(delta["audio"]["data"])
                                if audio_format == "wav":
                                    samples, sr = sf.read(
                                        io.BytesIO(encoded), dtype="int16"
                                    )
                                    assert sr == 24000 and samples.ndim == 1
                                    chunk = samples.astype("<i2").tobytes()
                                else:
                                    chunk = encoded
                                assert chunk and len(chunk) % 2 == 0
                                if first is None:
                                    first = elapsed
                                pcm.append(chunk)
                                entry.update(
                                    samples=len(chunk) // 2, pcm_sha256=digest(chunk)
                                )
                            row["events"].append(entry)
                    row["seconds"] = (time.perf_counter_ns() - start) / 1e9
                    state = states[rid]
                    torch.save(state, args.out_dir / f"{rid}.pt")
                    expected = dict(refs[case["id"]])
                    if overrides and "max_tokens" in overrides:
                        cap = overrides["max_tokens"]
                        assert case["mode"].endswith("s") and cap < len(
                            expected["grid"]
                        )
                        expected.update(
                            grid=expected["grid"][:cap], finish_reason="length"
                        )
                    row.update(
                        input_grid=state["input_grid"],
                        grid=state["output_grid"],
                        text=text,
                        text_decode_profile=TEXT_DECODE_PROFILE,
                        finish_reason=finish[-1] if finish else None,
                        done_count=done,
                        finish_count=len(finish),
                        ttfa_s=first,
                    )
                    row["matches_reference"] = all(
                        row[k] == expected[k]
                        for k in ("input_grid", "grid", "finish_reason")
                    )
                    row["matches_reference"] &= text == comparison_text(expected)
                    row["wire_complete"] = (
                        done == len(finish) == 1
                        and row["usage"] is not None
                        and row["usage"]["completion_tokens"] == len(row["grid"])
                    )
                    if case["mode"].endswith("s"):
                        data = b"".join(pcm)
                        (args.out_dir / f"{rid}.pcm").write_bytes(data)
                        wave = np.frombuffer(data, dtype="<i2")
                        sf.write(
                            args.out_dir / f"{rid}.wav", wave, 24000, subtype="PCM_16"
                        )
                        row.update(
                            samples=wave.size,
                            pcm_sha256=digest(data),
                            audio_chunks=len(pcm),
                            duration_s=wave.size / 24000,
                            rtf=row["seconds"] / (wave.size / 24000),
                            codec_ledger=state.get("stream_codec_ledger"),
                            stream_samples=state.get("stream_samples"),
                        )
                        row["sample_ledger"] = (
                            wave.size
                            == state["stream_samples"]
                            == sum(e["samples"] for e in row["codec_ledger"])
                        )
                    row["pass"] = (
                        row["matches_reference"]
                        and row["wire_complete"]
                        and row.get("sample_ledger", True)
                        and "\ufffd" not in text
                    )
                except Exception as exc:
                    row.update(error=repr(exc), pass_=False)
                    row["pass"] = row.pop("pass_")
                report["results"].append(row)
                save()
                print(
                    rid,
                    row["pass"],
                    row.get("error"),
                    row.get("audio_chunks"),
                    flush=True,
                )
                return row

            for case in cases:
                row = await post(case)
                if not row["pass"]:
                    raise RuntimeError(
                        f'First failing SSE case: {case["id"]}: {row.get("error")}'
                    )
            # WAV containers are decoded independently; never concatenate headers.
            wave_case = next(c for c in cases if c["mode"] == "t2s")
            report["wav_transport"] = (
                await post(wave_case, rid="wav-transport", audio_format="wav")
            )["pass"]
            if args.lifecycle:
                from scripts.moss_speech.p6.lifecycle import exercise

                report["lifecycle"] = await exercise(
                    http, client, runner, post, cases, report, args.out_dir
                )
            report["futures_empty"] = not runner.coordinator._completion_futures
            report["checks"] = {
                "requests": all(r["pass"] for r in report["results"]),
                "futures_empty": report["futures_empty"],
                "lifecycle": all(
                    report.get("lifecycle", {"not_requested": True}).values()
                ),
            }
    finally:
        if profiler is not None:
            try:
                await profiler.broadcast_stop()
            except Exception:
                pass
        if server is not None:
            server.should_exit = True
        try:
            if server_task is not None:
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
    timeline = events(args.out_dir / "events")
    for row in report["results"]:
        starts = [
            e["timestamp_ns"]
            for e in timeline
            if e["request_id"] == row["id"]
            and e["stage"] == "ar_engine"
            and e["event_name"] == "stage_first_stream_chunk_sent"
        ]
        audio = [
            e["timestamp_ns"]
            for e in timeline
            if e["request_id"] == row["id"]
            and e["stage"] == "audio_vocoder"
            and e["event_name"] == "stage_first_stream_chunk_sent"
        ]
        ends = [
            e["timestamp_ns"]
            for e in timeline
            if e["request_id"] == row["id"] and e["event_name"] == "model_path_end"
        ]
        if audio and ends:
            row["first_pcm_before_ar_end"] = min(audio) < max(ends)
            row["ar_end_minus_first_pcm_s"] = (max(ends) - min(audio)) / 1e9
        row["ar_first_stream_ns"] = min(starts) if starts else None
    report["early_pcm_proven"] = any(
        r.get("first_pcm_before_ar_end") for r in report["results"]
    )
    report["pass"] = (
        bool(report.get("checks"))
        and all(report["checks"].values())
        and report["processes_exit"]
        and report["early_pcm_proven"]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("manifest", "reference", "config", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--lifecycle", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "pass": False,
        "results": [],
        "manifest_sha256": sha256(args.manifest),
        "reference_sha256": sha256(args.reference),
        "config_sha256": sha256(args.config),
        "smoke": args.smoke,
        "lifecycle_requested": args.lifecycle,
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[3])): sha256(p)
            for root in (
                "sglang_omni/models/moss_speech",
                "scripts/moss_speech/p6",
                "sglang_omni/serve",
            )
            for p in (Path(__file__).resolve().parents[3] / root).rglob("*.py")
        },
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
