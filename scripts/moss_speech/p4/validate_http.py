#!/usr/bin/env python3
"""Real TCP HTTP validation with the formal native multiprocess pipeline."""
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
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import soundfile as sf
import torch
import uvicorn


def body(kind: str, *, limit: int = 200, source: str | None = None) -> dict[str, Any]:
    audio_out = kind.endswith("s")
    system = (
        "You are a helpful voice assistant. Answer the user's questions with spoken responses."
        if audio_out
        else "You are a helpful assistant. Answer the user's questions with text."
    )
    content: Any = (
        "用中文介绍一下上海的三到四个著名景点。"
        if audio_out
        else "Introduce yourself in one sentence."
    )
    if kind.startswith("s"):
        data = base64.b64encode(
            Path(source or "repos/MOSS-Speech/assets/prompt-cn.wav").read_bytes()
        ).decode()
        content = [
            {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
        ]
    return dict(
        model="moss-speech",
        messages=[
            dict(role="system", content=system),
            dict(role="user", content=content),
        ],
        modalities=["audio" if audio_out else "text"],
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.1,
        seed=0,
        max_tokens=limit,
        stream=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def events(directory: Path) -> list[dict[str, Any]]:
    result = []
    for path in directory.glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


async def run(args: Any, report: dict[str, Any]) -> None:
    from sglang_omni.client import Client
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.profiler.profiler_control import ProfilerControlClient
    from sglang_omni.serve.openai_api import create_app

    out = Path(args.out_dir)
    config = ConfigManager.from_file(args.config).config
    runner = MultiProcessPipelineRunner(config)
    processes = {}
    running = True
    raw = {}
    checks = report["checks"]
    report["memory"] = []
    report["requests"] = []
    report["rounds"] = []
    profiler = server = server_task = None
    client = None

    async def monitor() -> None:
        import pynvml

        pynvml.nvmlInit()
        handle = None
        while running:
            for group in runner._groups:
                for proc in group.processes:
                    if proc.pid:
                        processes[proc.pid] = proc
            if handle is None:
                for index in range(pynvml.nvmlDeviceGetCount()):
                    candidate = pynvml.nvmlDeviceGetHandleByIndex(index)
                    entries = pynvml.nvmlDeviceGetComputeRunningProcesses(candidate)
                    if any(p.pid in processes for p in entries):
                        handle = candidate
                        report["gpu_uuid"] = pynvml.nvmlDeviceGetUUID(handle)
                        report["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
                        break
                if handle is None:
                    await asyncio.sleep(0.5)
                    continue
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            entries = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            report["memory"].append(
                dict(
                    t=time.time(),
                    used=memory.used,
                    free=memory.free,
                    total=memory.total,
                    processes={
                        str(p.pid): p.usedGpuMemory
                        for p in entries
                        if p.pid in processes
                    },
                )
            )
            await asyncio.sleep(0.5)
        pynvml.nvmlShutdown()

    monitor_task = asyncio.create_task(monitor())

    def capture(rid: str, result: Any) -> Any:
        data = result.data if hasattr(result, "data") else result
        raw[rid] = data
        return Client._default_result_builder(rid, data)

    try:
        await runner.start(timeout=600)
        profiler = ProfilerControlClient(runner.stage_control_endpoints)
        await profiler.broadcast_start(
            run_id=out.name,
            trace_path_template=str(out / "unused"),
            event_dir=str(out / "events"),
            enable_torch=False,
        )
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
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=10)
        )
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if server_task.done():
                await server_task
            await asyncio.sleep(0.05)
        report["url"] = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(
            base_url=report["url"], timeout=600, limits=httpx.Limits(max_connections=32)
        ) as http:
            assert (await http.get("/health")).status_code == 200

            async def post(
                rid: str, request: dict[str, Any], *, keep: bool = True
            ) -> dict[str, Any]:
                start = time.monotonic()
                response = await http.post(
                    "/v1/chat/completions", json={**request, "request_id": rid}
                )
                elapsed = time.monotonic() - start
                assert response.status_code == 200, response.text
                result = response.json()
                assert result["id"] == "chatcmpl-" + rid
                assert len(result["choices"]) == 1
                assert result["choices"][0]["finish_reason"] in ("stop", "length")
                state = raw[rid]
                assert result["usage"]["prompt_tokens"] == len(state["input_grid"])
                assert result["usage"]["completion_tokens"] == len(state["output_grid"])
                report["requests"].append(
                    dict(
                        id=rid,
                        seconds=elapsed,
                        request_hash=digest(request),
                        grid_hash=digest(state["output_grid"]),
                        audio_hash=digest(
                            result["choices"][0]["message"].get("audio", {}).get("data")
                        ),
                        prompt_tokens=len(state["input_grid"]),
                        completion_tokens=len(state["output_grid"]),
                    )
                )
                if keep:
                    (out / (rid + ".json")).write_text(json.dumps(result))
                    torch.save(state, out / (rid + ".pt"))
                else:
                    raw.pop(rid)
                return result

            if args.soak:
                variants = [
                    body("t2s"),
                    body("s2s", source="artifacts/p1/perf/short_en_3s.wav"),
                    body("s2s", source="artifacts/p1/perf/long_cn_27s.wav"),
                ]
                for i, request in enumerate(variants):
                    await post(f"baseline-{i}", request, keep=False)
                sem = asyncio.Semaphore(4)

                async def repeat(i: int) -> None:
                    async with sem:
                        await post(f"repeat-{i}", variants[i % 3], keep=False)

                await asyncio.gather(*[repeat(i) for i in range(12)])
                previous = {}
                for request in report["requests"]:
                    value = (request["grid_hash"], request["audio_hash"])
                    assert previous.setdefault(request["request_hash"], value) == value
                checks["waveforms_and_grids_equal_serial_vs_interleaved"] = True
                checks["fifteen_completed"] = len(report["requests"]) == 15
                checks["completion_futures_empty"] = (
                    not runner.coordinator._completion_futures
                )
            elif not args.workload:
                for mode, fixture in [
                    ("t2t", "t2t_short"),
                    ("t2s", "t2s_cn"),
                    ("s2t", "s2t_cn"),
                    ("s2s", "s2s_cn"),
                ]:
                    result = await post(fixture, body(mode))
                    expected = torch.load(
                        Path("artifacts/p3/reference") / fixture / "tokens_grid.pt",
                        weights_only=True,
                    ).tolist()
                    canonical = json.loads(
                        (
                            Path("artifacts/p3/reference")
                            / fixture
                            / "canonical_input.json"
                        ).read_text()
                    )
                    checks[fixture + "_input"] = (
                        raw[fixture]["input_grid"] == canonical["input_ids"]
                    )
                    checks[fixture + "_grid"] = raw[fixture]["output_grid"] == expected
                    msg = result["choices"][0]["message"]
                    if mode.endswith("s"):
                        samples, sr = sf.read(
                            io.BytesIO(base64.b64decode(msg["audio"]["data"]))
                        )
                        checks[fixture + "_waveform"] = (
                            sr == 24000
                            and samples.size > 0
                            and bool(np.isfinite(samples).all())
                        )
                    else:
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(
                            config.model_path,
                            trust_remote_code=False,
                            local_files_only=True,
                        )
                        expected_text = (
                            tokenizer.decode(
                                [row[0] for row in expected], skip_special_tokens=True
                            )
                            .replace("<|empty|>", ".")
                            .replace("<|end_empty|>", ":")
                        )
                        checks[fixture + "_text"] = msg["content"] == expected_text
                    print(
                        fixture,
                        {k: v for k, v in checks.items() if k.startswith(fixture)},
                        flush=True,
                    )
                bound = body("s2t")
                data = bound["messages"][1]["content"][0]["input_audio"]["data"]
                bound["messages"][1]["content"][0]["input_audio"]["data"] = ""
                bound["audios"] = [data]
                await post("bound-audios", bound)
                checks["audios_binding"] = (
                    raw["bound-audios"]["output_grid"] == raw["s2t_cn"]["output_grid"]
                )
                invalid = [
                    {"stream": True},
                    {"modalities": ["text", "audio"]},
                    {"min_p": 0.1},
                    {"stop": ["x"]},
                    {"audio": {"voice": "x"}},
                    {"temperature": -1},
                    {"max_tokens": 0},
                    {"max_tokens": -1},
                    {"max_completion_tokens": 0},
                    {"max_completion_tokens": 17},
                    {"top_k": -2},
                    {"seed": 2**64},
                    {"messages": []},
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {"data": "bad", "format": "wav"},
                                    }
                                ],
                            }
                        ]
                    },
                ]
                report["invalid"] = []
                for i, extra in enumerate(invalid):
                    rid = f"invalid-{i}"
                    response = await http.post(
                        "/v1/chat/completions",
                        json={**body("t2t"), **extra, "request_id": rid},
                    )
                    report["invalid"].append(
                        dict(id=rid, status=response.status_code, detail=response.text)
                    )
                    checks[rid] = (
                        response.status_code in (400, 422)
                        and runner.coordinator.get_request_info(rid) is None
                    )
                oversized = body("t2t")
                oversized["messages"][1]["content"] = "😀" * 6000
                response = await http.post(
                    "/v1/chat/completions",
                    json={**oversized, "request_id": "exact-context-error"},
                )
                checks["exact_context_400"] = response.status_code == 400
                report["exact_context_error"] = response.text
                victim = asyncio.create_task(
                    post("disconnected", body("t2s", limit=500))
                )
                survivor = asyncio.create_task(post("survivor", body("t2s")))
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    found = [
                        e
                        for e in events(out / "events")
                        if e["request_id"] == "disconnected"
                        and e["event_name"] == "scheduler_prefill_end"
                    ]
                    if found:
                        break
                    await asyncio.sleep(0.02)
                assert found, "no AR prefill before disconnect"
                report["disconnect_event"] = found[0]
                victim.cancel()
                await asyncio.gather(victim, return_exceptions=True)
                await survivor
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    info = runner.coordinator.get_request_info("disconnected")
                    if info is None or info.state.value == "aborted":
                        break
                    await asyncio.sleep(0.05)
                checks["disconnect_abort"] = (
                    info is None or info.state.value == "aborted"
                ) and any(
                    e["request_id"] == "disconnected"
                    and e.get("metadata", {}).get("status") == "aborted"
                    for e in events(out / "events")
                )
                checks["survivor_grid"] = (
                    raw["survivor"]["output_grid"] == raw["t2s_cn"]["output_grid"]
                )
                checks["survivor_wave"] = (
                    raw["survivor"]["audio_samples"] == raw["t2s_cn"]["audio_samples"]
                )
                await post("recovered", body("t2t"))
                checks["recovery"] = (
                    raw["recovered"]["output_grid"] == raw["t2t_short"]["output_grid"]
                )
                # The exact prompt boundary, through the same HTTP/tokenizer path.
                from transformers import AutoTokenizer

                from sglang_omni.models.moss_speech.components.processor import (
                    MossSpeechGridProcessor,
                )

                processor = MossSpeechGridProcessor(
                    AutoTokenizer.from_pretrained(
                        config.model_path,
                        trust_remote_code=False,
                        local_files_only=True,
                    )
                )
                boundary = body("t2s", limit=512)
                boundary["messages"][1]["content"] = " a"

                def grid_len(value: dict[str, Any]) -> int:
                    turns = [
                        dict(role=m["role"], kind="text", text=m["content"])
                        for m in value["messages"]
                    ]
                    return processor.build(turns, [], "audio").prompt_len

                base_len = grid_len(boundary)
                boundary["messages"][1]["content"] = " a" * (513 - base_len)
                assert grid_len(boundary) == 512
                await asyncio.gather(
                    *[post(f"boundary-{i}", boundary) for i in range(4)]
                )
                checks["boundary_four_equal"] = all(
                    raw[f"boundary-{i}"]["output_grid"]
                    == raw["boundary-0"]["output_grid"]
                    for i in range(4)
                )
                checks["boundary_prompt_512"] = (
                    len(raw["boundary-0"]["input_grid"]) == 512
                )
                report["boundary_generated_rows"] = len(
                    raw["boundary-0"]["output_grid"]
                )
                for seed, rid in [(12, "random-alone"), (13, "random-other")]:
                    await post(
                        rid,
                        {
                            **body("t2t", limit=32),
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                            "seed": seed,
                        },
                    )
                await asyncio.gather(
                    *[
                        post(
                            rid,
                            {
                                **body("t2t", limit=32),
                                "temperature": 0.6,
                                "top_p": 0.95,
                                "top_k": 20,
                                "seed": 12,
                            },
                        )
                        for rid in ["random-a", "random-b"]
                    ]
                )
                checks["seed_repro"] = (
                    raw["random-alone"]["output_grid"]
                    == raw["random-a"]["output_grid"]
                    == raw["random-b"]["output_grid"]
                )
                checks["seed_changes"] = (
                    raw["random-alone"]["output_grid"]
                    != raw["random-other"]["output_grid"]
                )
                queued = [
                    asyncio.create_task(post(f"overload-{i}", body("t2s", limit=512)))
                    for i in range(24)
                ]
                deadline = time.monotonic() + 15
                while (
                    time.monotonic() < deadline
                    and len(runner.coordinator._requests) < 24
                ):
                    await asyncio.sleep(0.01)
                assert (
                    len(runner.coordinator._requests) == 24
                ), "admission cap not filled"
                rejected = await http.post(
                    "/v1/chat/completions",
                    json={**body("t2t"), "request_id": "over-cap"},
                )
                checks["overload_503"] = rejected.status_code == 503
                for task in queued:
                    task.cancel()
                await asyncio.gather(*queued, return_exceptions=True)
                deadline = time.monotonic() + 30
                while runner.coordinator._requests and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                checks["overload_cleanup"] = not runner.coordinator._requests
                await post("after-overload", body("t2t"))
                checks["overload_recovery"] = (
                    raw["after-overload"]["output_grid"]
                    == raw["t2t_short"]["output_grid"]
                )
                checks["completion_futures_empty"] = (
                    not runner.coordinator._completion_futures
                )
            else:
                # P1 arrival manifest; actual generation replaces synthetic codes.
                for concurrency in (1, 2, 4):
                    sem = asyncio.Semaphore(concurrency)

                    async def probe(i: int) -> None:
                        async with sem:
                            await post(
                                f"c{concurrency}-{i}", body("t2t", limit=32), keep=False
                            )

                    start = time.monotonic()
                    await asyncio.gather(*[probe(i) for i in range(8)])
                    report["rounds"].append(
                        dict(
                            kind="concurrency",
                            concurrency=concurrency,
                            seconds=time.monotonic() - start,
                        )
                    )
                for round_id in range(6):
                    start = time.monotonic()
                    tasks = []

                    async def arrival(i: int) -> None:
                        await asyncio.sleep(max(0, start + i * 0.4 - time.monotonic()))
                        source = (
                            "artifacts/p1/perf/short_en_3s.wav"
                            if (i // 2) % 2 == 0
                            else "artifacts/p1/perf/long_cn_27s.wav"
                        )
                        await post(
                            f"r{round_id}-{i}",
                            body("s2s" if i % 2 == 0 else "t2s", source=source),
                            keep=False,
                        )

                    tasks = [asyncio.create_task(arrival(i)) for i in range(24)]
                    await asyncio.gather(*tasks)
                    elapsed = time.monotonic() - start
                    latencies = [
                        v["seconds"]
                        for v in report["requests"]
                        if v["id"].startswith(f"r{round_id}-")
                    ]
                    report["rounds"].append(
                        dict(
                            kind="warmup" if round_id < 3 else "measured",
                            round=round_id,
                            seconds=elapsed,
                            rps=24 / elapsed,
                            p50=float(np.percentile(latencies, 50)),
                            p95=float(np.percentile(latencies, 95)),
                            memory=report["memory"][-1],
                        )
                    )
                    print("round", report["rounds"][-1], flush=True)
                hashes = {}
                for item in report["requests"]:
                    if item["id"].startswith("r"):
                        previous = hashes.setdefault(
                            item["request_hash"], item["grid_hash"]
                        )
                        assert (
                            previous == item["grid_hash"]
                        ), "repeat request grid changed"
                checks["workload_reproducible"] = True
                checks["workload_144_completed"] = (
                    len([r for r in report["requests"] if r["id"].startswith("r")])
                    == 144
                )
                idle = [
                    r["memory"]["used"]
                    for r in report["rounds"]
                    if r["kind"] == "measured"
                ]
                checks["steady_memory"] = max(idle) - min(idle) <= 256 * 1024**2
                checks["completion_futures_empty"] = (
                    not runner.coordinator._completion_futures
                )
    finally:
        if server:
            server.should_exit = True
        if server_task:
            await asyncio.wait_for(server_task, 30)
        if profiler:
            await profiler.close()
        await runner.stop()
        running = False
        await monitor_task
        report["processes"] = [
            dict(pid=pid, exitcode=p.exitcode, alive=p.is_alive())
            for pid, p in processes.items()
        ]
        checks["processes_exit"] = len(processes) == 4 and all(
            p.exitcode == 0 and not p.is_alive() for p in processes.values()
        )
    if not args.workload and not args.soak:
        from sglang_omni.client.audio import audio_to_base64
        from sglang_omni.models.moss_speech.stages import create_audio_vocoder_executor
        from sglang_omni.proto.request import StagePayload

        vocoder = create_audio_vocoder_executor(config.model_path)
        for fixture in ("t2s_cn", "s2s_cn"):
            state = dict(raw[fixture])
            state["output_grid"] = torch.load(
                Path("artifacts/p3/reference") / fixture / "tokens_grid.pt",
                weights_only=True,
            ).tolist()
            state["audio_samples"] = None
            expected = vocoder._fn(
                StagePayload(
                    request_id="reference-" + fixture,
                    request=SimpleNamespace(),
                    data=state,
                )
            ).data
            response = json.loads((out / (fixture + ".json")).read_text())
            checks[fixture + "_encoded_reference"] = (
                audio_to_base64(
                    expected["audio_samples"], sample_rate=24000, output_format="wav"
                )
                == response["choices"][0]["message"]["audio"]["data"]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="sglang-omni/examples/configs/moss_speech.yaml"
    )
    parser.add_argument("--out-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--workload", action="store_true")
    mode.add_argument("--soak", action="store_true")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=False)
    report = {"pass": False, "checks": {}, "workload": args.workload, "soak": args.soak}
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
        print("FINAL", report["pass"], report["checks"], flush=True)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
