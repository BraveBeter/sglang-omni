#!/usr/bin/env python3
"""Smoke the documented CLI command and its owned subprocess shutdown."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psutil


def run(out: Path, report: dict[str, Any]) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        "sglang-omni/examples/configs/moss_speech.yaml",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    report["command"] = command
    children = {}
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        report["pid"] = process.pid
        parent = psutil.Process(process.pid)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as http:
                deadline = time.monotonic() + 600
                ready = False
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"CLI exited {process.returncode} before readiness"
                        )
                    for child in parent.children(recursive=True):
                        children[child.pid] = child
                    try:
                        ready = http.get("/health", timeout=1).status_code == 200
                    except httpx.TransportError:
                        pass
                    if ready:
                        break
                    time.sleep(1)
                assert ready, "CLI readiness timed out"
                report["checks"]["ready"] = True
                base = {
                    "model": "moss-speech",
                    "messages": [{"role": "user", "content": "Hello."}],
                    "max_tokens": 32,
                    "temperature": 0,
                    "seed": 0,
                }
                for modality in ("text", "audio"):
                    response = http.post(
                        "/v1/chat/completions", json={**base, "modalities": [modality]}
                    )
                    assert response.status_code == 200, response.text
                    value = response.json()
                    (out / (modality + ".json")).write_text(json.dumps(value))
                    message = value["choices"][0]["message"]
                    report["checks"][modality] = bool(
                        message.get("content")
                        if modality == "text"
                        else message.get("audio", {}).get("data")
                    )
                rejected = http.post(
                    "/v1/chat/completions", json={**base, "stream": True}
                )
                report["checks"]["preflight_hook"] = rejected.status_code == 400
                report["models"] = http.get("/v1/models").json()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            report["returncode"] = process.returncode
            time.sleep(1)
            report["children"] = [
                {
                    "pid": pid,
                    "alive": child.is_running()
                    and child.status() != psutil.STATUS_ZOMBIE,
                }
                for pid, child in children.items()
            ]
            report["checks"]["exit"] = (
                process.returncode == 0
                and bool(children)
                and not any(c["alive"] for c in report["children"])
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"pass": False, "checks": {}}
    try:
        run(out, report)
        report["pass"] = all(report["checks"].values())
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
