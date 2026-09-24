"""Explicit per-flow live metadata owner. Call only under the codec's own lock.

No epoch or registration list is retained. Every snapshot executes the qualified
GIL-held native traversal, or the current original Python implementation. Graph
and prefix-bank owners must be closed before this owner releases its references.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable


def load_package(path: Any) -> Any:
    from .cfm_epoch_package import NativeEpochPackage

    return NativeEpochPackage(path)


class NativeEpochOwner:
    def __init__(self, flow: Any, manifest: Any, original: Callable, *, loader=None):
        self.flow, self.manifest, self.original = flow, manifest, original
        self.loader = loader or load_package
        self.pid = os.getpid()
        self.provider = None
        self.active = self.closed = False
        self.calls = self.native_calls = self.fallback_calls = self.failed_calls = 0
        self.prepare_seconds = 0.0
        self.identity = self.qualification = None

    def _check(self) -> None:
        if self.closed or os.getpid() != self.pid:
            raise RuntimeError("Closed or cross-process native epoch owner")

    def prepare(self) -> None:
        self._check()
        if self.active or self.provider is not None:
            raise RuntimeError("Native epoch owner already prepared")
        started = time.perf_counter()
        try:
            self.provider = self.loader(self.manifest)
            before = self.original(self.flow)
            stats = {}
            native = self.provider.epoch(self.flow, self.original, stats=stats)
            after = self.original(self.flow)
            if stats.get("route") != "native" or not before == native == after:
                raise ValueError(
                    "Original/native/original complete epoch tuple differs"
                )
            if stats.get("gil_held") is not True:
                raise ValueError("Native snapshot did not retain the GIL")
            self.identity = dict(self.provider.identity)
            self.qualification = dict(
                stats,
                original_controls=2,
                native_controls=1,
                full_tuple_exact=True,
                entries=len(native),
            )
            self.active = True
        except BaseException:
            if self.provider is not None:
                self.provider.close()
                self.provider = None
            raise
        finally:
            self.prepare_seconds = time.perf_counter() - started

    def snapshot(self, flow: Any, original: Callable) -> tuple:
        self._check()
        if not self.active:
            raise RuntimeError("Native epoch owner is not qualified")
        self.calls += 1
        stats = {}
        try:
            if flow is self.flow and original is self.original:
                value = self.provider.epoch(flow, original, stats=stats)
            else:
                stats["route"] = "original"
                value = original(flow)
            if stats.get("route") == "native":
                self.native_calls += 1
            elif stats.get("route") == "original":
                self.fallback_calls += 1
            else:
                raise RuntimeError("Native epoch provider returned no route")
            return value
        except BaseException:
            self.failed_calls += 1
            raise

    def status(self) -> dict:
        return dict(
            enabled=self.active,
            closed=self.closed,
            pid=self.pid,
            calls=self.calls,
            native_calls=self.native_calls,
            fallback_calls=self.fallback_calls,
            failed_calls=self.failed_calls,
            prepare_seconds=self.prepare_seconds,
            binding=self.identity,
            qualification=self.qualification,
            flow_released=self.flow is None,
            provider_released=self.provider is None,
            caches_epoch_or_registration_lists=False,
        )

    def close(self) -> None:
        if self.closed:
            return
        self._check()
        if self.provider is not None:
            self.provider.close()
        self.provider = self.flow = self.original = self.loader = None
        self.active = False
        self.closed = True
