#!/usr/bin/env python3
"""Validate installed console commands from outside the checkout, under Slurm."""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

from scripts.moss_speech.p5.quality import (
    build_cases,
    comparison_text,
    reference_text_fields,
    sha256,
)


def check_text_lengths(
    http: httpx.Client, args: Any, refs: dict, streaming: bool
) -> list[dict]:
    """Compare actual HTTP text to prefixes of an immutable independent HF grid."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    results = []
    for case in build_cases(args.manifest):
        if case["id"] not in ("en_00_t2t", "zh_00_t2t"):
            continue
        grid = refs[case["id"]]["grid"]
        caps = {1, 2, 3}
        # Exercise a real incomplete multibyte token when present in the reference.
        for cap in range(1, min(len(grid), 32)):
            if tokenizer.decode(
                [row[0] for row in grid[:cap]], skip_special_tokens=True
            ).endswith("\ufffd"):
                caps.add(cap)
                break
        for cap in sorted(caps):
            assert cap < len(grid)
            expected = reference_text_fields(tokenizer, grid[:cap], audio_output=False)
            body = {**case["body"], "max_tokens": cap, "stream": streaming}
            text, finish, done = "", [], 0
            if streaming:
                with http.stream("POST", "/v1/chat/completions", json=body) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        if line[6:] == "[DONE]":
                            done += 1
                            continue
                        event = json.loads(line[6:])
                        assert "error" not in event, event
                        choice = event["choices"][0]
                        text += choice["delta"].get("content") or ""
                        if choice.get("finish_reason"):
                            finish.append(choice["finish_reason"])
                assert done == 1
            else:
                response = http.post("/v1/chat/completions", json=body)
                response.raise_for_status()
                result = response.json()
                choice = result["choices"][0]
                text = choice["message"].get("content") or ""
                finish = [choice["finish_reason"]]
                assert result["usage"]["completion_tokens"] == cap
            assert finish == ["length"] and text == expected["service_text"], (
                case["id"],
                cap,
                text,
                expected,
            )
            results.append(
                dict(id=case["id"], cap=cap, actual_text=text, **expected, pass_=True)
            )
    assert len(results) >= 6
    return results


def check_profile(args: Any, streaming: bool, report: dict[str, Any]) -> None:
    out = args.out_dir / ("streaming" if streaming else "offline")
    out.mkdir()
    executable = shutil.which("sglang-omni")
    assert executable, "Install the checkout so sglang-omni is on PATH"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        executable,
        "serve",
        "--model-path",
        str(args.model_path),
        "--codec-path",
        str(args.codec_path),
        "--voice-wav",
        str(args.voice_wav),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if streaming:
        command += ["--config", str(args.streaming_config)]
    report.update(command=command, cwd=str(out), checks={}, results=[])
    env = dict(os.environ)
    # The installed console script must resolve the package by itself.
    env.pop("PYTHONPATH", None)
    env.pop("MOSS_SPEECH_VOICE_WAV", None)
    ref = json.loads(args.reference.read_text())
    refs = {r["id"]: r for r in ref["results"]}
    children: dict[int, psutil.Process] = {}
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            cwd=out,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        report["pid"] = process.pid
        parent = psutil.Process(process.pid)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=900) as http:
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"CLI exited before ready: {process.returncode}"
                        )
                    children.update({p.pid: p for p in parent.children(recursive=True)})
                    try:
                        if http.get("/health", timeout=1).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("CLI readiness timed out")
                report["checks"]["ready"] = True
                for case in [
                    c
                    for c in build_cases(args.manifest)
                    if c["id"].startswith("en_00_")
                ]:
                    body = {**case["body"], "stream": streaming}
                    audio = case["mode"].endswith("s")
                    text, encoded, finish = "", b"", []
                    if streaming:
                        if audio:
                            body["audio"] = {"format": "pcm"}
                        done = 0
                        with http.stream(
                            "POST", "/v1/chat/completions", json=body
                        ) as response:
                            response.raise_for_status()
                            for line in response.iter_lines():
                                if not line.startswith("data: "):
                                    continue
                                if line[6:] == "[DONE]":
                                    done += 1
                                    continue
                                event = json.loads(line[6:])
                                assert "error" not in event, event
                                choice = event["choices"][0]
                                delta = choice["delta"]
                                text += delta.get("content") or ""
                                if delta.get("audio", {}).get("data"):
                                    encoded += base64.b64decode(delta["audio"]["data"])
                                if choice.get("finish_reason"):
                                    finish.append(choice["finish_reason"])
                        assert done == 1
                    else:
                        response = http.post("/v1/chat/completions", json=body)
                        response.raise_for_status()
                        choice = response.json()["choices"][0]
                        text = choice["message"].get("content") or ""
                        if audio:
                            encoded = base64.b64decode(
                                choice["message"]["audio"]["data"]
                            )
                        finish = [choice["finish_reason"]]
                    rid = case["id"]
                    assert finish == [refs[rid]["finish_reason"]]
                    if audio:
                        expected = (
                            args.streaming_baseline
                            if streaming
                            else args.offline_baseline
                        ) / (rid + (".pcm" if streaming else ".wav"))
                        assert encoded == expected.read_bytes(), f"Audio changed: {rid}"
                        (out / expected.name).write_bytes(encoded)
                    else:
                        assert text == comparison_text(
                            refs[rid]
                        ), f"Text changed: {rid}"
                    report["results"].append(
                        {"id": rid, "pass": True, "bytes": len(encoded)}
                    )
                    print(out.name, rid, "passed", flush=True)
                report["checks"]["four_modes"] = len(report["results"]) == 4
                report["text_lengths"] = check_text_lengths(http, args, refs, streaming)
                report["checks"]["text_lengths"] = all(
                    r["pass_"] for r in report["text_lengths"]
                )
                if not streaming:
                    response = http.post(
                        "/v1/chat/completions", json={**body, "stream": True}
                    )
                    report["checks"]["offline_rejects_streaming"] = (
                        response.status_code == 400
                    )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            time.sleep(1)
            report["returncode"] = process.returncode
            report["children"] = [
                {
                    "pid": pid,
                    "alive": p.is_running() and p.status() != psutil.STATUS_ZOMBIE,
                }
                for pid, p in children.items()
            ]
            report["checks"]["exit"] = (
                process.returncode == 0
                and bool(children)
                and not any(p["alive"] for p in report["children"])
            )
    report["checks"]["noninteractive_discovery"] = (
        "Do you wish to run the custom code?" not in (out / "server.log").read_text()
    )
    report["pass"] = all(report["checks"].values())
    assert report["pass"]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "model-path",
        "codec-path",
        "voice-wav",
        "streaming-config",
        "manifest",
        "reference",
        "offline-baseline",
        "streaming-baseline",
        "out-dir",
    ):
        parser.add_argument(
            "--" + name, required=True, type=lambda value: Path(value).resolve()
        )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "pass": False,
        "source_sha256": sha256(Path(__file__)),
        "reference_sha256": sha256(args.reference),
        "profiles": {},
    }
    try:
        for streaming in (False, True):
            profile: dict[str, Any] = {}
            report["profiles"][str(streaming)] = profile
            check_profile(args, streaming, profile)
        report["pass"] = True
    finally:
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
