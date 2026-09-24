# SPDX-License-Identifier: Apache-2.0
"""Private ownership guard for prepared CFM graph storage.

GPU stream order and the codec lock govern tensor use. CPU startup and executor
threads may differ; graph inputs must never be overwritten concurrently.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class ReplayLease:
    """Retain cache tensors and admit one caller on the prepared replay stream."""

    def __init__(
        self,
        owners: tuple[Any, ...],
        describe: Callable[[Any], Any],
        *,
        stream: int,
    ) -> None:
        if not owners or not callable(describe):
            raise ValueError("Graph replay requires owned storage and metadata")
        self.owners = tuple(owners)
        self._describe = describe
        self._signatures = tuple(describe(t) for t in self.owners)
        self._stream = stream
        self._lock = threading.Lock()
        self._busy = self._closed = self._invalid = False

    @contextmanager
    def use(self, *, stream: int) -> Iterator[None]:
        """A failed metadata check permanently invalidates this prepared state."""
        with self._lock:
            if self._closed:
                raise RuntimeError("CFM graph lease is closed")
            if self._busy:
                raise RuntimeError("CFM graph lease is busy")
            if self._invalid:
                raise RuntimeError("CFM graph lease is invalid")
            if stream != self._stream:
                raise RuntimeError("CFM graph replay stream changed")
            try:
                signatures = tuple(self._describe(t) for t in self.owners)
            except BaseException:
                self._invalid = True
                raise
            if signatures != self._signatures:
                self._invalid = True
                raise RuntimeError("CFM graph storage changed")
            self._busy = True
        try:
            yield
        finally:
            with self._lock:
                self._busy = False

    def close(self) -> None:
        """Caller releases graph executables before dropping their tensor owners."""
        with self._lock:
            if self._busy:
                raise RuntimeError("CFM graph lease is busy")
            self._closed = True
            self.owners = ()
            self._signatures = ()
