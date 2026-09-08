"""Real TCP disconnects, controlled transport failure and repeated stream checks."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import aclosing
from pathlib import Path
from typing import Any

from scripts.moss_speech.p4.validate_http import events


async def exercise(
    http: Any, client: Any, runner: Any, post: Any, cases: list, report: dict, out: Any
) -> dict[str, bool]:
    from scripts.moss_speech.p5.quality import sha256

    report["executed_lifecycle_sha256"] = sha256(Path(__file__))
    checks = {}
    case = next(c for c in cases if c["id"] == "en_00_t2s")
    other = next(c for c in cases if c["id"] == "zh_00_t2s")
    baseline = {r["case_id"]: r for r in report["results"] if r["id"] == r["case_id"]}
    report["invalid"], report["disconnects"], report["idle_memory"] = [], [], []

    for i, extra in enumerate(
        [
            {"audio": {"format": "mp3"}},
            {"audio": {"voice": "unsupported"}},
            {"modalities": ["text", "audio"]},
            {"max_tokens": 0},
            {"temperature": -1},
            {"seed": 2**64},
        ]
    ):
        rid = f"invalid-stream-{i}"
        response = await http.post(
            "/v1/chat/completions",
            json={
                **case["body"],
                "stream": True,
                "request_id": rid,
                **extra,
            },
        )
        ok = (
            response.status_code in (400, 422)
            and runner.coordinator.get_request_info(rid) is None
        )
        report["invalid"].append(dict(id=rid, status=response.status_code, pass_=ok))
        checks[rid] = ok

    # The frozen greedy AR path must keep its prefix when only the cap changes.
    for limit in (1, 5, 7, 8, 10, 13):
        row = await post(case, rid=f"length-{limit}", overrides={"max_tokens": limit})
        checks[f"length-{limit}"] = row["pass"] and row["finish_reason"] == "length"

    async def wait_aborted(rid: str) -> bool:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            info = runner.coordinator.get_request_info(rid)
            timeline = [e for e in events(out / "events") if e["request_id"] == rid]
            ar_aborted = any(
                e.get("metadata", {}).get("status") == "aborted" for e in timeline
            )
            released = [
                e
                for e in timeline
                if e["event_name"] == "moss_codec_session_released"
                and e["metadata"].get("session_active") is False
            ]
            report.setdefault("release_events", {})[rid] = released
            # Before first PCM the request may be cancelled before codec creation.
            observed = bool(released) or (rid == "disconnect-before_pcm" and ar_aborted)
            if (info is None or info.state.value == "aborted") and observed:
                return rid not in runner.coordinator._completion_futures
            await asyncio.sleep(0.05)
        return False

    for phase in ("before_pcm", "after_pcm", "ar_done_pending_flush"):
        rid = f"disconnect-{phase}"
        seen = {"chunks": 0, "done": False, "ar_end": False}
        headers, audio_seen = asyncio.Event(), asyncio.Event()

        async def victim() -> None:
            async with http.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    **case["body"],
                    "stream": True,
                    "request_id": rid,
                    "audio": {"format": "pcm"},
                },
            ) as response:
                response.raise_for_status()
                headers.set()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line[6:] == "[DONE]":
                        seen["done"] = True
                        continue
                    event = json.loads(line[6:])
                    if (
                        event.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("audio", {})
                        .get("data")
                    ):
                        seen["chunks"] += 1
                        audio_seen.set()

        task = asyncio.create_task(victim())
        survivor = asyncio.create_task(post(other, rid=f"survivor-{phase}"))
        await asyncio.wait_for(headers.wait(), 60)
        if phase == "before_pcm":
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and not task.done():
                seen["prefill"] = any(
                    e["request_id"] == rid
                    and e["event_name"] == "scheduler_prefill_end"
                    for e in events(out / "events")
                )
                if seen["prefill"]:
                    break
                await asyncio.sleep(0.01)
        elif phase == "after_pcm":
            await asyncio.wait_for(audio_seen.wait(), 120)
        elif phase == "ar_done_pending_flush":
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and not task.done():
                seen["ar_end"] = any(
                    e["request_id"] == rid and e["event_name"] == "model_path_end"
                    for e in events(out / "events")
                )
                if seen["ar_end"]:
                    break
                await asyncio.sleep(0.01)
        seen["cancelled_before_done"] = not task.done() and not seen["done"]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        seen["aborted"] = await wait_aborted(rid)
        survivor_row = await survivor
        seen["survivor_equal"] = (
            survivor_row["pass"]
            and survivor_row.get("pcm_sha256") == baseline[other["id"]]["pcm_sha256"]
        )
        recovery = await post(case, rid=f"recovery-{phase}")
        seen["recovery_equal"] = (
            recovery["pass"]
            and recovery.get("pcm_sha256") == baseline[case["id"]]["pcm_sha256"]
        )
        phase_reached = (
            seen.get("prefill", False) and seen["chunks"] == 0
            if phase == "before_pcm"
            else seen["chunks"] > 0 if phase == "after_pcm" else seen["ar_end"]
        )
        checks[rid] = phase_reached and all(
            seen[k]
            for k in (
                "cancelled_before_done",
                "aborted",
                "survivor_equal",
                "recovery_equal",
            )
        )
        report["disconnects"].append(dict(id=rid, **seen))

    # Inject after a real backend delta; close the owned iterator to exercise abort.
    original = client.completion_stream

    async def fail_after_delta(*args: Any, **kwargs: Any) -> Any:
        async with aclosing(original(*args, **kwargs)) as stream:
            async for delta in stream:
                yield delta
                if kwargs.get("request_id") == "controlled-stream-error":
                    raise RuntimeError("controlled post-header failure")

    client.completion_stream = fail_after_delta
    wire = []
    try:
        async with http.stream(
            "POST",
            "/v1/chat/completions",
            json={
                **case["body"],
                "stream": True,
                "request_id": "controlled-stream-error",
                "audio": {"format": "pcm"},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    wire.append(line[6:])
    finally:
        client.completion_stream = original
    errors = [json.loads(v) for v in wire if v != "[DONE]" and "error" in json.loads(v)]
    checks["controlled_error"] = (
        len(errors) == 1
        and "[DONE]" not in wire
        and all(
            not json.loads(v).get("choices", [{}])[0].get("finish_reason") for v in wire
        )
        and await wait_aborted("controlled-stream-error")
    )
    report["controlled_error"] = dict(wire=wire, pass_=checks["controlled_error"])

    # Repeated two-request waves detect cross-request output or retained allocation.
    for repeat in range(4):
        rows = await asyncio.gather(
            *[post(c, rid=f"repeat-{repeat}-{c['id']}") for c in (case, other)]
        )
        checks[f"repeat-{repeat}"] = all(
            r["pass"] and r["pcm_sha256"] == baseline[r["case_id"]]["pcm_sha256"]
            for r in rows
        )
        await asyncio.sleep(1)
        report["idle_memory"].append(report["memory"][-1])
    idle = [m["used"] for m in report["idle_memory"]][1:]
    # One warmup wave, then a predeclared 64 MiB allocator/NVML tolerance.
    checks["idle_memory_bounded"] = max(idle) - min(idle) <= 64 * 1024**2
    checks["completion_futures_empty"] = not runner.coordinator._completion_futures
    return checks
