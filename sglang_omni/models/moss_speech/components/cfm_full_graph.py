# SPDX-License-Identifier: Apache-2.0
"""Prepared model-local complete CFM graphs with owned replay storage.

Euler arithmetic follows the vendored CosyVoice solver (Alibaba Inc., 2024/2025).
Only explicitly prepared native prefix-bank calls enter these fixed-shape graphs.
"""
from __future__ import annotations

import threading
import time
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterator

INPUT_KEYS = ("x", "t_span", "mu", "mask", "spks", "cond")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def raw_equal(left: Any, right: Any) -> bool:
    import torch

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def ff_first_tail(module: Any, value: Any, projection: Callable[[Any], Any]) -> Any:
    """Evaluate a qualified compact projection and the original GELU."""
    import torch
    from diffusers.models.activations import GELU

    require(
        type(module) is GELU
        and module.approximate in ("none", "tanh")
        and type(module.proj) is torch.nn.Linear
        and module.proj.in_features == 256
        and module.proj.out_features == 1024
        and module.proj.bias is not None,
        "Unknown FF first/GELU module",
    )
    require(
        tuple(value.shape)
        in ((2, 1392, 256), (2, 1400, 256), (2, 1424, 256), (2, 1500, 256))
        and value.is_contiguous(),
        "Unknown FF first/GELU input",
    )
    projected = projection(value[:, 1280:, :])
    require(
        projected.shape == (2, value.shape[1] - 1280, 1024),
        "Unknown FF first projection output",
    )
    return module.gelu(projected)


def tensor_layout(value: Any) -> tuple[Any, ...]:
    return (
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
    )


def tensor_storage(value: Any) -> tuple[Any, ...]:
    # Graph inputs are intentionally overwritten. Their version is not immutable.
    return id(value), value.data_ptr(), value.storage_offset(), tensor_layout(value)


def storage_bytes(values: tuple[Any, ...]) -> int:
    storages = {
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
        for value in values
    }
    return sum(storages.values())


@contextmanager
def temporary_attribute(target: Any, name: str, value: Any) -> Iterator[None]:
    owned = name in vars(target)
    original = vars(target).get(name)
    setattr(target, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(target, name, original)
        else:
            delattr(target, name)


def execution_context(cfm: Any) -> tuple[Any, ...]:
    import torch

    priority = getattr(torch._C, "_get_sdp_priority_order", lambda: ())
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
        tuple(priority()),
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
        cfm.inference_cfg_rate,
        cfm.inference_optimizations,
        id(cfm.estimator),
        getattr(cfm.forward_estimator, "__func__", cfm.forward_estimator),
    )


class EulerBody:
    """The original ten Euler steps, including original CFG operation order."""

    def __init__(self, cfm: Any, values: dict[str, Any], proof: Any) -> None:
        import torch

        from .cosyvoice_flow.cfm_attention import expand_cfg_provenance

        require(
            cfm.inference_optimizations, "Complete graphs need optimized native CFM"
        )
        self.cfm, self.values = cfm, values
        x = values["x"]
        self.x_in = torch.zeros((2, 80, x.size(2)), device=x.device, dtype=x.dtype)
        self.mask_in = torch.zeros((2, 1, x.size(2)), device=x.device, dtype=x.dtype)
        self.mu_in = torch.zeros_like(self.x_in)
        self.t_in = torch.zeros((2,), device=x.device, dtype=x.dtype)
        self.spks_in = torch.zeros((2, 80), device=x.device, dtype=x.dtype)
        self.cond_in = torch.zeros_like(self.x_in)
        self.mask_in[:] = values["mask"]
        self.proof = expand_cfg_provenance(proof, values["mask"], self.mask_in)
        self.bias_cache: dict[Any, Any] = {}
        self.last_estimator_calls = 0

    @property
    def owners(self) -> tuple[Any, ...]:
        return (
            self.x_in,
            self.mask_in,
            self.mu_in,
            self.t_in,
            self.spks_in,
            self.cond_in,
            *self.bias_cache.values(),
        )

    def __call__(self) -> Any:
        import torch

        values = self.values
        x, t_span = values["x"], values["t_span"]
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        self.mask_in[:] = values["mask"]
        self.mu_in[0] = values["mu"]
        self.spks_in[0] = values["spks"]
        self.cond_in[0] = values["cond"]
        self.last_estimator_calls = 0
        for step in range(1, len(t_span)):
            self.x_in[:] = x
            self.t_in[:] = t.unsqueeze(0)
            dphi_dt = self.cfm.forward_estimator(
                self.x_in,
                self.mask_in,
                self.mu_in,
                self.t_in,
                self.spks_in,
                self.cond_in,
                True,
                attention_bias_cache=self.bias_cache,
                full_valid_mask=self.proof,
            )
            self.last_estimator_calls += 1
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (
                1.0 + self.cfm.inference_cfg_rate
            ) * dphi_dt - self.cfm.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x.float()


class _ShapeGraph:
    def __init__(
        self, owner: FullCfmGraphBank, length: int, sample: dict[str, Any]
    ) -> None:
        import torch

        from .cfm_graph_lifetime import ReplayLease
        from .cosyvoice_flow.cfm_attention import new_full_valid_mask

        self.owner, self.length = owner, length
        self.device = sample["values"]["x"].device
        self.context = execution_context(owner.cfm)
        self.layouts = {
            key: tensor_layout(value) for key, value in sample["values"].items()
        }
        self.inputs = {key: value.clone() for key, value in sample["values"].items()}
        require(
            all(
                tensor_layout(self.inputs[key]) == layout
                for key, layout in self.layouts.items()
            ),
            "CFM input clone changed layout",
        )
        mask, proof = new_full_valid_mask(self.inputs["mask"], length)
        require(
            raw_equal(mask, self.inputs["mask"]), "Prepared CFM mask is not all valid"
        )
        self.inputs["mask"] = mask
        self.body = EulerBody(owner.cfm, self.inputs, proof)
        self.graph = None
        self.lease = None
        self.closed = False
        self.actual_bodies: list[dict[str, int]] = []
        self.parent_entries = tuple(
            (key, value)
            for key, value in owner.bank.entries.items()
            if key[0] == length
        )
        require(len(self.parent_entries) == 560, "Incomplete shared SDPA prefix bank")
        self.shared_bytes = storage_bytes(
            tuple(value for _, value in self.parent_entries)
        )
        self.bytes = 0
        self.own_tensors: tuple[Any, ...] = ()
        self.receipt: dict[str, Any] = {"length": length}
        with torch.cuda.device(self.device):
            caller = torch.cuda.current_stream(self.device)
            self.stream = caller.cuda_stream
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(caller)
            started = time.perf_counter()
            try:
                with owner.block_scope(length), torch.cuda.stream(side):
                    owner._lt_context.prepare_stream()
                    owner._bind_qkv(length)
                    for _ in range(2):
                        self._body()
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=side):
                        self.output = self._body()
                caller.wait_stream(side)
                torch.cuda.synchronize(self.device)
                require(
                    len(self.actual_bodies) == 3
                    and all(
                        row == owner._expected_body(length)
                        for row in self.actual_bodies
                    ),
                    "Incomplete actual CFM graph capture",
                )
                own_tensors = (
                    *self.inputs.values(),
                    *self.body.owners,
                    self.output,
                    owner._query_slots[length],
                    *(
                        value
                        for key, value in owner._prefixes.items()
                        if key[0] == length
                    ),
                    *(
                        value
                        for key, value in owner._kv_slots.items()
                        if key[0] == length
                    ),
                    *(
                        value
                        for key, plan in owner._runtime_plans(length).items()
                        if key[0] == length
                        for value in (plan.input, plan.output)
                    ),
                    *owner._lt_context.owners,
                    *owner._branch_owners(length),
                )
                parameter_owners = tuple(
                    value
                    for key, plan in owner._runtime_plans(length).items()
                    if key[0] == length
                    for value in (plan.weight, plan.bias)
                    if value is not None
                )
                shared = tuple(value for _, value in self.parent_entries)
                self.own_tensors = tuple(own_tensors)
                self.lease = ReplayLease(
                    self.own_tensors + shared + parameter_owners,
                    tensor_storage,
                    stream=self.stream,
                )
                self.bytes = storage_bytes(self.own_tensors)
                result = self.run(sample["values"])
                require(
                    raw_equal(result, sample["reference"])
                    and result.stride() == sample["reference"].stride(),
                    "Complete CFM graph startup raw/layout gate failed",
                )
                if owner.qkv_multistream:
                    self.receipt.update(
                        qkv_multistream=owner._multistream(length),
                        qkv_branch_identity=(
                            list(owner._shape_schedules[length].initial)
                            if owner._multistream(length)
                            else []
                        ),
                    )
                self.receipt.update(
                    prepare_seconds=time.perf_counter() - started,
                    actual_bodies=self.actual_bodies,
                    raw_exact=True,
                    startup_validation_replays=1,
                    owned_tensor_bytes=self.bytes,
                    shared_prefix_bytes=self.shared_bytes,
                )
            except BaseException:
                self.close()
                raise

    def _body(self) -> Any:
        from .cfm_prefix_cache import PrefixCallLedger

        owner, bank = self.owner, self.owner.bank
        before = owner._tail_totals(self.length)
        before_branches = owner._branch_calls(self.length)
        before_hits, before_fallbacks = bank.hits, bank.fallbacks
        owner._tail_ledger = PrefixCallLedger(bank.names)
        with bank.scope(
            bank.binding.voice_digest, self.length, False, bank.binding.environment
        ):
            require(
                bank.active_length == self.length,
                "Prepared bank rejected graph capture",
            )
            output = self.body()
            delta = tuple(
                a - b for a, b in zip(owner._tail_totals(self.length), before)
            )
            row = dict(
                blocks=owner._tail_ledger.count,
                estimator_calls=self.body.last_estimator_calls,
                attention_hits=bank.hits - before_hits,
                attention_fallbacks=bank.fallbacks - before_fallbacks,
                block_tail_calls=delta[0],
                tail_sdpa_calls=delta[1],
                direct_projection_calls=delta[2],
            )
            if owner.qkv_multistream:
                row["qkv_fork_join_calls"] = (
                    owner._branch_calls(self.length) - before_branches
                )
            require(
                row == owner._expected_body(self.length) and bank.ledger.count == 0,
                "Incomplete block tail graph body or unexpected parent attention",
            )
            owner._tail_ledger.finish()
        self.actual_bodies.append(row)
        return output

    def matches(self, values: dict[str, Any], streaming: bool, proof: Any) -> bool:
        import torch

        from .cosyvoice_flow.cfm_attention import FullValidMask

        return (
            not self.closed
            and streaming
            and type(proof) is FullValidMask
            and proof.matches(values["mask"])
            and execution_context(self.owner.cfm) == self.context
            and torch.cuda.current_stream(self.device).cuda_stream == self.stream
            and all(
                hasattr(values[key], "shape") and tensor_layout(values[key]) == layout
                for key, layout in self.layouts.items()
            )
            and all(
                self.owner.bank.entries.get(key) is value
                for key, value in self.parent_entries
            )
        )

    def run(self, values: dict[str, Any]) -> Any:
        import torch

        with torch.cuda.device(self.device):
            with self.lease.use(
                stream=torch.cuda.current_stream(self.device).cuda_stream
            ):
                # Captured inputs are inference tensors even if a caller's outer
                # context changes while handling an exception or shutdown.
                with torch.inference_mode():
                    for key in INPUT_KEYS:
                        self.inputs[key].copy_(values[key])
                    self.graph.replay()
                return self.output.clone()

    def close(self) -> None:
        if self.closed:
            return
        import torch

        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            if self.graph is not None:
                self.graph.reset()
                self.graph = None
            if self.lease is not None:
                self.lease.close()
        self.own_tensors = ()
        self.inputs.clear()
        self.parent_entries = ()
        self.body = None
        self.output = None
        self.closed = True


class FullCfmGraphBank:
    """A finite policy-specific shape bank; unsupported solves remain eager."""

    def __init__(
        self, cfm: Any, prefix_bank: Any, *, qkv_multistream: bool = False
    ) -> None:
        require(prefix_bank.flow.decoder is cfm, "CFM and prefix bank differ")
        require(
            type(qkv_multistream) is bool, "QKV multistream must be an explicit boolean"
        )
        self.qkv_multistream = qkv_multistream
        self._qkv_branches = None
        self._branch_plans = {}
        self._branch_keys = set()
        self._shape_schedules = {}
        self.cfm, self.bank = cfm, prefix_bank
        self._original = cfm.solve_euler
        self._had_override = "solve_euler" in vars(cfm)
        self._override = vars(cfm).get("solve_euler")
        self._installed = None
        self._closed = self._prepared = False
        self._lock = threading.Lock()
        self._graphs: dict[int, _ShapeGraph] = {}
        self._samples: dict[int, dict[str, Any]] = {}
        self._prefixes: dict[tuple[int, int, str, str], Any] = {}
        self._query_slots: dict[int, Any] = {}
        self._kv_slots: dict[tuple[int, int, str, str], Any] = {}
        self._raw_keys: set[tuple[int, int, str, str]] = set()
        self._tail_blocks: dict[tuple[int, int, str], Any] = {}
        self._tail_ledger = None
        self._whole_block_checks = self._block_intermediate_checks = 0
        self._qkv_keys: set[tuple[int, int, str, str]] = set()
        self._tail_plans: dict[tuple[int, str, str], Any] = {}
        self._full_plans: dict[tuple[int, str, str], Any] = {}
        self._lt_context = None
        self._blocks: tuple[tuple[str, Any], ...] = ()
        self._collecting: int | None = None
        self._ff_first_checks = 0
        self._ff_first_keys: set[tuple[int, int, str, str]] = set()
        self._raw_checks = 0
        self._projection_calls = self._replays = self._fallbacks = 0

    def _multistream(self, length: int) -> bool:
        return self.qkv_multistream and length in (1392, 1424)

    def _runtime_plans(self, length: int) -> dict:
        plans = {
            key: plan for key, plan in self._tail_plans.items() if key[0] == length
        }
        if self._multistream(length):
            plans.update(
                {
                    key: plan
                    for key, plan in self._branch_plans.items()
                    if key[0] == length
                }
            )
        return plans

    def _expected_body(self, length: int) -> dict:
        row = dict(
            blocks=560,
            estimator_calls=10,
            attention_hits=0,
            attention_fallbacks=0,
            block_tail_calls=560,
            tail_sdpa_calls=560,
            direct_projection_calls=3360,
        )
        if self.qkv_multistream:
            row["qkv_fork_join_calls"] = 560 if self._multistream(length) else 0
        return row

    def _bind_qkv(self, length: int) -> None:
        if not self._multistream(length):
            return
        schedule = self._qkv_branches.bind()
        self._shape_schedules[length] = schedule
        for key, body in self._tail_blocks.items():
            if key[0] == length:
                body.schedule = schedule

    def _branch_calls(self, length: int) -> int:
        schedule = self._shape_schedules.get(length)
        return schedule.calls if schedule is not None else 0

    def _branch_owners(self, length: int) -> tuple:
        if not self._multistream(length):
            return ()
        return (
            *self._qkv_branches.owners,
            *(
                t
                for key, body in self._tail_blocks.items()
                if key[0] == length
                for t in body.capture_owners
            ),
        )

    def _collect_branch(self, key, module, value, output) -> None:
        import torch

        from .cfm_direct_lt import DirectLtContext, DirectLtProjection
        from .cfm_qkv_multistream import QkvBranches

        length, _, name, kind = key
        require(
            self._multistream(length)
            and kind in ("k", "v")
            and key not in self._branch_keys,
            "Duplicate or unsupported branch projection",
        )
        if self._qkv_branches is None:
            self._qkv_branches = QkvBranches(
                torch.cuda, DirectLtContext, self._lt_context
            )
            self._qkv_branches.prepare()
        plan_key = (length, name, kind)
        if plan_key not in self._branch_plans:
            with torch.cuda.stream(self._qkv_branches.streams[kind]):
                self._branch_plans[plan_key] = DirectLtProjection(
                    module.weight,
                    module.bias,
                    2 * (length - 1280),
                    context=self._qkv_branches.contexts[kind],
                    profile=f"qkv_{length}",
                )
            require(
                self._branch_plans[plan_key].describe()
                == self._tail_plans[plan_key].describe(),
                "Parallel projection changes original algorithm/descriptors",
            )
        actual = self._qkv_branches.call(
            kind, self._branch_plans[plan_key], value[:, 1280:, :]
        )
        require(
            raw_equal(actual, output[:, 1280:, :]),
            "Original branch tail projection differs",
        )
        self._branch_keys.add(key)
        if length not in self._shape_schedules:
            self._bind_qkv(length)

    def _key(
        self, name: str, *, before_attention: bool = False
    ) -> tuple[int, int, str]:
        bank = self.bank
        require(
            bank.active_length is not None and bank.ledger is not None,
            "CFM projection outside prepared scope",
        )
        index = bank.ledger.count if before_attention else bank.ledger.count - 1
        require(
            0 <= index < 560 and bank.names[index % 56] == name,
            "CFM projection/attention call order changed",
        )
        return bank.active_length, index // 56, name

    def _collect_projection(
        self, name: str, kind: str, module: Any, args: tuple, output: Any
    ) -> None:
        if self._collecting is None:
            return
        import torch

        from .cfm_direct_lt import DirectLtContext, DirectLtProjection

        qkv = kind in ("q", "k", "v")
        position = self._key(name, before_attention=True) if qkv else self._key(name)
        key = (*position, kind)
        length = key[0]
        require(
            kind in ("q", "k", "v", "to_out", "ff", "ff_first"),
            "Unknown projection kind",
        )
        first = kind == "ff_first"
        seen = (
            self._qkv_keys if qkv else self._ff_first_keys if first else self._raw_keys
        )
        require(
            length == self._collecting and key not in seen,
            "Duplicate or wrong-shape projection prefix",
        )
        value = args[0]
        features = 512 if qkv else 1024 if first else 256
        if qkv:
            require(
                type(module) is torch.nn.Linear
                and module.bias is None
                and value.shape == (2, length, 256)
                and value.is_contiguous(),
                "Unknown original QKV projection",
            )
        require(
            output.shape == (2, length, features), "Unknown original projection shape"
        )
        profile = f"qkv_{length}" if qkv else "ff_first" if first else "output"
        with torch.cuda.device(value.device):
            if self._lt_context is None:
                self._lt_context = DirectLtContext(value.device)
                self._lt_context.prepare_stream()
            plan_key = (length, name, kind)
            if plan_key not in self._tail_plans:
                self._tail_plans[plan_key] = DirectLtProjection(
                    module.weight,
                    module.bias,
                    2 * (length - 1280),
                    context=self._lt_context,
                    profile=profile,
                )
                self._full_plans[plan_key] = DirectLtProjection(
                    module.weight,
                    module.bias,
                    2 * length,
                    context=self._lt_context,
                    profile=profile,
                )
            full = self._full_plans[plan_key](value.contiguous())
            require(
                raw_equal(full, output) and full.stride() == output.stride(),
                "Original full direct-Lt projection differs",
            )
            tail_input = value[:, 1280:, :]
            if kind == "ff":
                tail_input = tail_input.contiguous()
            tail = self._tail_plans[plan_key](tail_input)
            require(
                raw_equal(tail, output[:, 1280:, :]),
                "Original tail direct-Lt projection differs",
            )
            if qkv:
                if self._multistream(length) and kind in ("k", "v"):
                    self._collect_branch(key, module, value, output)
                self._qkv_keys.add(key)
                if kind == "q":
                    if length not in self._query_slots:
                        self._query_slots[length] = output.new_zeros((2, length, 512))
                else:
                    # Full slots are the sole persistent K/V storage, bound to this step/layer.
                    self._kv_slots[key] = output.clone()
            elif first:
                self._ff_first_keys.add(key)
            else:
                self._raw_checks += 1
                self._raw_keys.add(key)

    def _collect_ff_first(
        self, name: str, module: Any, args: tuple, output: Any
    ) -> None:
        if self._collecting is None:
            return
        length, _, _ = self._key(name)
        # Direct-Lt/GELU calls do not re-enter the original module hooks.
        actual = ff_first_tail(
            module, args[0], self._tail_plans[(length, name, "ff_first")]
        )
        require(
            raw_equal(actual, output[:, 1280:, :]),
            "Original FF first/GELU tail differs",
        )
        self._ff_first_checks += 1

    def _observe(
        self,
        x: Any,
        t_span: Any,
        mu: Any,
        mask: Any,
        spks: Any = None,
        cond: Any = None,
        streaming: bool = False,
        full_valid_mask: Any = None,
    ) -> Any:
        from .cosyvoice_flow.cfm_attention import FullValidMask

        values = dict(zip(INPUT_KEYS, (x, t_span, mu, mask, spks, cond)))
        length = self.bank.active_length
        if length not in self.bank.binding.shapes:
            return self._original(
                **values, streaming=streaming, full_valid_mask=full_valid_mask
            )
        require(
            self._collecting is None and length not in self._samples,
            "Duplicate CFM calibration solve",
        )
        require(
            streaming
            and type(full_valid_mask) is FullValidMask
            and full_valid_mask.matches(mask),
            "Calibration mask lacks native provenance",
        )
        require(
            x.shape == (1, 80, length)
            and mask.shape == (1, 1, length)
            and mu.shape == cond.shape == x.shape
            and spks.shape == (1, 80)
            and t_span.shape == (11,),
            "Unknown CFM calibration inputs",
        )
        require(
            all(
                value.device == x.device and value.dtype == x.dtype
                for value in values.values()
            ),
            "Mixed CFM calibration tensors",
        )
        captured = {key: value.clone() for key, value in values.items()}
        self._collecting = length
        try:
            result = self._original(
                **values, streaming=streaming, full_valid_mask=full_valid_mask
            )
        finally:
            self._collecting = None
        require(
            self.bank.ledger.count == 560, "Calibration missed native attention layers"
        )
        self._samples[length] = dict(values=captured, reference=result.clone())
        return result

    def _tail_totals(self, length: int) -> tuple[int, int, int]:
        rows = [
            body.counts() for key, body in self._tail_blocks.items() if key[0] == length
        ]
        return tuple(sum(row[index] for row in rows) for index in range(3))

    @staticmethod
    def _block_arguments(args: tuple, kwargs: dict) -> tuple[Any, Any]:
        require(
            not args
            and set(kwargs)
            <= {"hidden_states", "attention_mask", "timestep", "cross_attention_kwargs"}
            and {"hidden_states", "attention_mask"} <= set(kwargs)
            and set(kwargs.get("cross_attention_kwargs") or {})
            <= {"moss_cfm_mask_plan"},
            "Unknown native block arguments",
        )
        return kwargs["hidden_states"], kwargs["attention_mask"]

    def _collect_block(
        self, name: str, block: Any, original: Callable, args: tuple, kwargs: dict
    ) -> Any:
        if self._collecting is None:
            return original(*args, **kwargs)
        from .cfm_block_tail import IntermediateGate, TailBlock, capture_native

        hidden, mask = self._block_arguments(args, kwargs)
        key = self._key(name, before_attention=True)
        require(
            key[0] == self._collecting and key not in self._tail_blocks,
            "Duplicate or wrong original whole-block identity",
        )
        result, captured = capture_native(block, original, args, kwargs)
        require(self._key(name) == key, "Original whole-block attention order changed")
        prefix_key = (*key, "block")
        self._prefixes[prefix_key] = result[:, :1280, :].clone()
        body_type = TailBlock
        extra = {}
        if self._multistream(key[0]):
            from .cfm_qkv_multistream import ParallelTailBlock

            body_type = ParallelTailBlock
            extra["schedule"] = self._shape_schedules[key[0]]
        runtime_plans = self._runtime_plans(key[0])
        body = body_type(
            block,
            key,
            self._query_slots[key[0]],
            self._kv_slots[(*key, "k")],
            self._kv_slots[(*key, "v")],
            self._prefixes[prefix_key],
            {
                kind: runtime_plans[(key[0], name, kind)]
                for kind in ("q", "k", "v", "to_out", "ff_first", "ff")
            },
            **extra,
        )
        check = IntermediateGate(captured, key[0])
        candidate = body.run(hidden, mask, key=key, observe=check)
        count = check.finish()
        require(
            raw_equal(candidate, result)
            and candidate.stride() == result.stride()
            and body.counts() == (1, 1, 6),
            "Original complete block tail differs",
        )
        self._tail_blocks[key] = body
        self._whole_block_checks += 1
        self._block_intermediate_checks += count
        return result

    @contextmanager
    def block_scope(self, length: int) -> Iterator[None]:
        with ExitStack() as stack:
            for name, block in self._blocks:

                def forward(*args: Any, label: str = name, **kwargs: Any) -> Any:
                    hidden, mask = self._block_arguments(args, kwargs)
                    require(
                        self._tail_ledger is not None,
                        "Whole-block scope has no execution ledger",
                    )
                    step, actual = self._tail_ledger.next(label)
                    key = (length, step, actual)
                    require(
                        key in self._tail_blocks, "Missing whole-block prefix identity"
                    )
                    return self._tail_blocks[key].run(hidden, mask, key=key)

                stack.enter_context(temporary_attribute(block, "forward", forward))
            yield

    def prepare(self, calibration: Callable[[], tuple[Any, ...]]) -> dict[str, Any]:
        require(
            not self._closed and not self._prepared,
            "CFM graphs already prepared or closed",
        )
        require(
            self.bank.active_length is None, "Cannot prepare in an active prefix scope"
        )
        require(
            self.bank.binding.valid and self.bank.mode == "cached",
            "CFM graphs need a valid cached prefix bank",
        )
        require(self._lock.acquire(blocking=False), "CFM graph owner is busy")
        started = time.perf_counter()
        try:
            import torch

            from .cfm_block_tail import validate_native_block
            from .cosyvoice_flow.cfm_attention import runtime_supported
            from .matcha_components.transformer import BasicTransformerBlock

            device = next(self.cfm.estimator.parameters()).device
            require(
                runtime_supported(
                    device_name=torch.cuda.get_device_name(device),
                    capability=torch.cuda.get_device_capability(device),
                    torch_build=torch.__version__,
                    cuda_build=torch.version.cuda,
                    dtype=next(self.cfm.estimator.parameters()).dtype,
                ),
                "Unqualified full-CFM runtime",
            )
            require(
                torch.is_inference_mode_enabled()
                and not torch.is_grad_enabled()
                and not torch.backends.cuda.matmul.allow_tf32,
                "Complete CFM preparation requires exact inference context",
            )
            blocks = []
            for name, attention, _ in self.bank.modules:
                require(name.endswith(".attn1"), "Unknown native attention path")
                block = self.cfm.estimator.get_submodule(name.rsplit(".", 1)[0])
                require(
                    type(block) is BasicTransformerBlock
                    and block.attn1 is attention
                    and len(block.ff.net) == 3
                    and block.ff.net[1].p == 0.0,
                    "Unknown native CFM block",
                )
                validate_native_block(block)
                blocks.append((name, block))
            require(len(blocks) == 56, "Incomplete native CFM blocks")
            self._blocks = tuple(blocks)
            memory_before = torch.cuda.memory_allocated(device)
            handles = []
            try:
                for name, block in self._blocks:
                    for kind, module in (
                        ("q", block.attn1.to_q),
                        ("k", block.attn1.to_k),
                        ("v", block.attn1.to_v),
                        ("to_out", block.attn1.to_out[0]),
                        ("ff", block.ff.net[2]),
                        ("ff_first", block.ff.net[0].proj),
                    ):
                        handles.append(
                            module.register_forward_hook(
                                lambda mod, args, output, label=name, op=kind: self._collect_projection(
                                    label, op, mod, args, output
                                )
                            )
                        )
                    handles.append(
                        block.ff.net[0].register_forward_hook(
                            lambda mod, args, output, label=name: self._collect_ff_first(
                                label, mod, args, output
                            )
                        )
                    )
                with ExitStack() as stack, torch.cuda.device(device):
                    stack.enter_context(
                        temporary_attribute(self.cfm, "solve_euler", self._observe)
                    )
                    for name, block in self._blocks:
                        original = block.forward

                        def collect(
                            *args: Any,
                            label: str = name,
                            module: Any = block,
                            native: Callable = original,
                            **kwargs: Any,
                        ) -> Any:
                            return self._collect_block(
                                label, module, native, args, kwargs
                            )

                        stack.enter_context(
                            temporary_attribute(block, "forward", collect)
                        )
                    calibration()
            finally:
                for handle in handles:
                    handle.remove()
            shapes = self.bank.binding.shapes
            require(
                set(self._samples) == set(shapes)
                and len(self._prefixes) == 560 * len(shapes)
                and len(self._kv_slots) == 1120 * len(shapes)
                and len(self._tail_blocks) == 560 * len(shapes)
                and len(self._raw_keys) == 1120 * len(shapes)
                and self._whole_block_checks == 560 * len(shapes)
                and self._block_intermediate_checks == 13 * 560 * len(shapes)
                and len(self._qkv_keys) == 1680 * len(shapes)
                and set(self._query_slots) == set(shapes)
                and self._raw_checks == 1120 * len(shapes)
                and self._ff_first_checks == 560 * len(shapes)
                and len(self._ff_first_keys) == 560 * len(shapes),
                "Incomplete original prefix/direct-Lt/FF first qualification",
            )
            require(
                len(self._branch_keys)
                == 1120 * sum(self._multistream(n) for n in shapes),
                "Incomplete original branch projection qualification",
            )
            calibration_allocated = torch.cuda.memory_allocated(device)
            full_control_bytes = storage_bytes(
                tuple(
                    t
                    for plan in self._full_plans.values()
                    for t in (plan.input, plan.output)
                )
            )
            with torch.cuda.device(device):
                for plan in self._full_plans.values():
                    plan.close()
                self._full_plans.clear()
            with torch.cuda.device(device):
                for length in shapes:
                    self._graphs[length] = _ShapeGraph(
                        self, length, self._samples[length]
                    )
            transient_serial_keys = [
                key
                for key in self._tail_plans
                if self._multistream(key[0]) and key[2] in ("k", "v")
            ]
            transient_serial_bytes = storage_bytes(
                tuple(
                    t
                    for key in transient_serial_keys
                    for t in (self._tail_plans[key].input, self._tail_plans[key].output)
                )
            )
            # Graph leases reference only branch K/V plans. Original serial controls
            # survive all startup raw gates, then relinquish their temporary buffers.
            with torch.cuda.device(device):
                for key in transient_serial_keys:
                    self._tail_plans[key].close()
                    del self._tail_plans[key]
            self._samples.clear()
            self._prepared = True
            return dict(
                prepare_seconds=time.perf_counter() - started,
                graph_replays=self._replays,
                qkv_multistream=self.qkv_multistream,
                qkv_branch_projection_checks=len(self._branch_keys),
                original_projection_checks=self._raw_checks,
                qkv_projection_checks=len(self._qkv_keys),
                ff_first_checks=self._ff_first_checks,
                ff_first_projection_checks=len(self._ff_first_keys),
                whole_block_checks=self._whole_block_checks,
                block_intermediate_checks=self._block_intermediate_checks,
                startup_memory=dict(
                    pre_calibration_allocated_bytes=memory_before,
                    calibration_allocated_bytes=calibration_allocated,
                    transient_full_control_bytes=full_control_bytes,
                    transient_serial_qkv_control_bytes=transient_serial_bytes,
                    prepared_allocated_bytes=torch.cuda.memory_allocated(device),
                    device_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_scope="device cumulative startup peak; includes calibration stages and full Lt controls",
                ),
                shapes=[entry.receipt for entry in self._graphs.values()],
            )
        finally:
            self._lock.release()

    def install(self) -> None:
        require(self._prepared and not self._closed, "CFM graph bank is not prepared")
        if self._installed is None:
            self._installed = self.solve_euler
            self.cfm.solve_euler = self._installed

    def solve_euler(
        self,
        x: Any,
        t_span: Any,
        mu: Any,
        mask: Any,
        spks: Any = None,
        cond: Any = None,
        streaming: bool = False,
        full_valid_mask: Any = None,
    ) -> Any:
        values = dict(zip(INPUT_KEYS, (x, t_span, mu, mask, spks, cond)))
        require(self._lock.acquire(blocking=False), "CFM graph owner is busy")
        try:
            entry = (
                self._graphs.get(self.bank.active_length)
                if self._prepared
                and not self._closed
                and self.bank.binding.valid
                and self.bank.mode == "cached"
                else None
            )
            if entry is None or not entry.matches(values, streaming, full_valid_mask):
                self._fallbacks += 1
                return self._original(
                    **values, streaming=streaming, full_valid_mask=full_valid_mask
                )
            result = entry.run(values)
            self._replays += 1
            return result
        finally:
            self._lock.release()

    def status(self) -> dict[str, Any]:
        result = dict(
            enabled=self._prepared and not self._closed and self.bank.binding.valid,
            shapes=list(self._graphs),
            graph_replays=self._replays,
            fallback_calls=self._fallbacks,
            bytes=storage_bytes(
                tuple(
                    t
                    for entry in self._graphs.values()
                    for t in getattr(entry, "own_tensors", ())
                )
            ),
            shared_prefix_bytes=sum(
                entry.shared_bytes for entry in self._graphs.values()
            ),
        )

        if self.qkv_multistream:
            result.update(
                qkv_multistream=True,
                qkv_multistream_shapes=[
                    n for n in self._graphs if self._multistream(n)
                ],
                qkv_branch_projection_checks=len(self._branch_keys),
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        require(self._lock.acquire(blocking=False), "CFM graph owner is busy")
        try:
            if (
                self._installed is not None
                and getattr(self.cfm, "solve_euler") is self._installed
            ):
                if self._had_override:
                    self.cfm.solve_euler = self._override
                else:
                    delattr(self.cfm, "solve_euler")
            self._installed = None
            for entry in self._graphs.values():
                entry.close()
            self._graphs.clear()
            if self._qkv_branches is not None:
                import torch

                with torch.cuda.device(self._lt_context.device):
                    self._qkv_branches.close()
            if self._lt_context is not None:
                import torch

                with torch.cuda.device(self._lt_context.device):
                    self._lt_context.close()
            self._qkv_branches = None
            self._branch_plans.clear()
            self._branch_keys.clear()
            self._shape_schedules.clear()
            self._tail_plans.clear()
            self._full_plans.clear()
            self._ff_first_keys.clear()
            self._qkv_keys.clear()
            self._prefixes.clear()
            self._query_slots.clear()
            self._kv_slots.clear()
            self._raw_keys.clear()
            self._tail_blocks.clear()
            self._tail_ledger = None
            self._samples.clear()
            self._blocks = ()
            self._closed = True
        finally:
            self._lock.release()
