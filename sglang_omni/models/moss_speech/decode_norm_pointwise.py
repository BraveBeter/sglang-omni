# SPDX-License-Identifier: Apache-2.0
"""Capture-only owner for explicitly qualified complete-decode norm arithmetic."""
from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch

logger = logging.getLogger(__name__)
_LOCAL = threading.local()
_GEOMETRIES = {(1, 4096): 82, (1, 32, 128): 40, (1, 8, 128): 40}
_LAYOUTS = {
    (1, 4096): (4096, 1),
    (1, 32, 128): (4096, 128, 1),
    (1, 8, 128): (1024, 128, 1),
}
_CAPACITIES = (128, 512, 1024)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def active_owner() -> DecodeNormOwner | None:
    return getattr(_LOCAL, "owner", None)


def _signature(module: Any) -> tuple:
    weight = module.weight
    return (
        id(module),
        type(module),
        getattr(module.forward, "__func__", None),
        module.training,
        id(weight),
        weight.data_ptr(),
        tuple(weight.shape),
        weight.stride(),
        weight.storage_offset(),
        weight.dtype,
        weight.device,
        weight.requires_grad,
        weight.is_inference(),
        weight._version,
        module.variance_epsilon,
    )


def _cuda_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.device.type == "cuda"
        and x.device == weight.device
        and torch.cuda.current_device() == x.device.index
        and x.data_ptr() % 256 == weight.data_ptr() % 256 == 0
    )


class DecodeNormOwner:
    """Keep native launch code alive through all three complete decode graphs."""

    def __init__(
        self, model: Any, norm_type: type, engine: Callable[..., dict]
    ) -> None:
        self.model, self.norm_type, self.engine = model, norm_type, engine
        self.norms = tuple(n for n in model.modules() if isinstance(n, norm_type))
        _require(len(self.norms) == 162, "Expected162 original norms")
        _require(len({id(n) for n in self.norms}) == 162, "Duplicate native norm")
        _require(callable(engine), "Missing explicit native launch owner")
        for norm in self.norms:
            _require(
                type(norm) is norm_type and not norm.training, "Only native eval norms"
            )
            _require(
                type(norm.weight) is torch.nn.Parameter
                and norm.weight.requires_grad
                and not norm.weight.is_inference()
                and norm.weight.dtype == torch.bfloat16
                and norm.weight.stride() == (1,)
                and tuple(norm.weight.shape) in ((4096,), (128,)),
                "Original norm parameter metadata changed",
            )
            _require(
                math.isfinite(norm.variance_epsilon) and norm.variance_epsilon > 0,
                "Invalid native epsilon",
            )
        _require(
            Counter(tuple(n.weight.shape) for n in self.norms)
            == {(4096,): 82, (128,): 80},
            "Wrong native parameter geometry coverage",
        )
        self._signatures = tuple(_signature(n) for n in self.norms)
        self._norm_ids = {id(n) for n in self.norms}
        self._thread = threading.get_ident()
        self._busy = self._closed = False
        self._flags_preserved = True
        self._seen: set[int] = set()
        self._shapes: Counter = Counter()
        self._bodies: Counter = Counter()
        self._calls = self._fallbacks = self._partial = self._failed = 0
        self._lock = threading.RLock()

    @contextmanager
    def body(self, model: Any, capacity: int) -> Iterator[None]:
        with self._lock:
            _require(
                torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),
                "Original inference capture scope required",
            )
            _require(
                active_owner() is None and not self._busy, "No nested norm capture"
            )
            _require(
                not self._closed and model is self.model, "Closed or foreign norm owner"
            )
            _require(threading.get_ident() == self._thread, "Different capture thread")
            _require(
                type(capacity) is int and capacity in _CAPACITIES,
                "Unknown decode capacity",
            )
            _require(
                self._bodies[capacity] < 3, "No request-time or fourth capture body"
            )
            _require(
                tuple(_signature(n) for n in self.norms) == self._signatures,
                "Original native norm state changed",
            )
            current = tuple(n for n in model.modules() if isinstance(n, self.norm_type))
            _require(
                tuple(map(id, current)) == tuple(map(id, self.norms)),
                "Native norm structure changed",
            )
            self._busy = True
            self._seen, self._shapes = set(), Counter()
            _LOCAL.owner = self
            success = False
            try:
                yield
                _require(
                    len(self._seen) == 162 and self._shapes == Counter(_GEOMETRIES),
                    "Incomplete162 norm capture body",
                )
                _require(
                    tuple(_signature(n) for n in self.norms) == self._signatures,
                    "Norm state changed during capture",
                )
                self._bodies[capacity] += 1
                success = True
            finally:
                _LOCAL.owner = None
                self._busy = False
                if not success:
                    self._failed += 1
                self._seen, self._shapes = set(), Counter()

    def apply(self, norm: Any, x: torch.Tensor) -> torch.Tensor | None:
        _require(self._busy and active_owner() is self, "No active norm owner")
        shape = tuple(x.shape)
        supported = (
            id(norm) in self._norm_ids
            and type(norm) is self.norm_type
            and type(x) is torch.Tensor
            and not norm.training
            and torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled()
            and not x.requires_grad
            and x.dtype == torch.bfloat16
            and shape in _GEOMETRIES
            and x.stride() == _LAYOUTS[shape]
            and x.storage_offset() == 0
            and tuple(norm.weight.shape) == (shape[-1],)
            and _cuda_supported(x, norm.weight)
        )
        if not supported:
            self._fallbacks += 1
            return None
        _require(id(norm) not in self._seen, "Duplicate norm in capture body")
        self._seen.add(id(norm))
        self._shapes[shape] += 1
        self._calls += 1
        value = self.engine(x, norm.weight, norm.variance_epsilon)["output"]
        _require(
            type(value) is torch.Tensor
            and value.shape == x.shape
            and value.dtype == x.dtype
            and value.device == x.device
            and value.stride() == _LAYOUTS[shape]
            and value.storage_offset() == 0
            and value.data_ptr() not in (x.data_ptr(), norm.weight.data_ptr())
            and value.is_inference()
            and not value.requires_grad
            and value.grad_fn is None,
            "Candidate output differs from native inference metadata",
        )
        return value

    def observe_partial_projection(self) -> None:
        self._partial += 1
        raise ValueError("Unexpected partial projection during complete norm capture")

    def status(self) -> dict:
        return dict(
            closed=self._closed,
            busy=self._busy,
            norm_calls=self._calls,
            fallback_calls=self._fallbacks,
            partial_projection_calls=self._partial,
            failed_bodies=self._failed,
            bodies={str(k): v for k, v in sorted(self._bodies.items())},
            norm_parameter_flags_preserved=(
                self._flags_preserved
                if self._closed
                else tuple(_signature(n) for n in self.norms) == self._signatures
            ),
        )

    def close(self) -> None:
        with self._lock:
            _require(not self._busy, "Cannot close busy norm owner")
            if self._closed:
                return
            self._flags_preserved = (
                tuple(_signature(n) for n in self.norms) == self._signatures
            )
            self._closed = True
            self.engine = None
            self.model = None
            self.norms = ()
            self._norm_ids.clear()
            self._signatures = ()


@contextmanager
def norm_capture(
    owner: DecodeNormOwner | None, model: Any, capacity: int
) -> Iterator[None]:
    if owner is None:
        yield
    else:
        with owner.body(model, capacity):
            yield


def graph_norm(norm: Any, x: torch.Tensor) -> torch.Tensor | None:
    owner = active_owner()
    return None if owner is None else owner.apply(norm, x)


def close_norm_owner(model: Any) -> None:
    owner = getattr(model, "_moss_decode_norm_owner", None)
    if owner is not None:
        identity = getattr(owner.engine, "identity", None)
        manager = getattr(model, "_decode_graph_manager", None)
        replays = getattr(manager, "hits", 0)
        owner.close()
        delattr(model, "_moss_decode_norm_owner")
        logger.info(
            "MOSS decode norm graphs closed: %s",
            json.dumps(
                dict(**owner.status(), kernel=identity, replays=replays), sort_keys=True
            ),
        )
