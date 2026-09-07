# SPDX-License-Identifier: Apache-2.0
"""CPU-test environment shims for constructing sglang model layers without
CUDA or an initialized distributed process group.

IMPORTANT: import this module FIRST (before anything that imports sglang).
On a CUDA-build-but-GPU-less host sglang's platform probes report neither
cuda nor cpu, which makes ``RotaryEmbedding.__init__`` import
``vllm._custom_ops`` (absent here). Module import (1) pre-registers a stub
``vllm._custom_ops`` module so that lazy import succeeds (the stub is never
executed on the CPU test paths), and (2) installs the remaining shims:

- ``get_server_args`` (as consumed inside ``sglang.srt.models.qwen3`` and via
  the runtime context) returns a namespace with ``rl_on_policy_target=None``
  (the only field the reused Qwen3 decoder layer reads at construction time).
- A REAL single-process gloo process group is initialized (world size 1)
  via sglang's own ``init_distributed_environment``/``initialize_model_parallel``,
  so VocabParallelEmbedding / LayerCommunicator / dp_attention group getters
  all resolve. A TCP store on a per-process random loopback port avoids
  clashes between concurrent pytest workers.
"""

from __future__ import annotations

import types
from typing import Any

import sys

def _install_vllm_stub() -> None:
    if "vllm._custom_ops" in sys.modules:
        return
    vllm = sys.modules.get("vllm")
    if vllm is None:
        vllm = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm
    ops = types.ModuleType("vllm._custom_ops")

    def _unreachable(*args: Any, **kwargs: Any):  # pragma: no cover
        raise AssertionError("vllm rotary stub must not execute on CPU tests")

    ops.rotary_embedding = _unreachable
    sys.modules["vllm._custom_ops"] = ops


_install_vllm_stub()

class _ServerArgsStandin:
    """Tolerant ServerArgs stand-in: any un-set attribute reads as False
    (bool-flag semantics); the few non-bool fields are declared explicitly."""

    speculative_algorithm = "NONE"
    attention_backend = None
    decode_attention_backend = None
    prefill_attention_backend=None
    kntransformers_cfg = None
    page_size = 1
    spec_num_draft_tokens = 0
    torch_compile_max_bs = 0
    rl_on_policy_target = None
    device = "cpu"
    # numeric sizes (a False default would divide-modulo to 0 in dp math)
    dp_size = 1
    tp_size = 1
    attn_cp_size = 1
    moe_dense_tp_size = 1
    ep_size = 1
    max_ep_size = 0
    ep_join_rank_offset = 0
    pipeline_parallel_size = 1

    def __getattr__(self, name: str) -> bool:
        return False


SERVER_ARGS_STANDIN = _ServerArgsStandin()


_SHIMMED = False


def _init_single_process_groups() -> None:
    import torch as _torch

    torch_dtype_bf16 = _torch.bfloat16
    import os as _os
    import random as _random
    import socket as _socket

    import torch.distributed as dist
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if dist.is_initialized():
        return
    with _socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", str(port))
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    initialize_dp_attention(
        SERVER_ARGS_STANDIN,
        types.SimpleNamespace(
            hidden_size=0,
            dtype=torch_dtype_bf16,
            hf_config=types.SimpleNamespace(hybrid_override_pattern=None),
        ),
    )


def install_cpu_shims() -> None:
    global _SHIMMED
    if _SHIMMED:
        return

    import sglang.srt.models.qwen3 as q3

    if not getattr(q3.get_server_args, "_moss_cpu_shim", False):
        def _fake_server_args() -> Any:
            return SERVER_ARGS_STANDIN

        _fake_server_args._moss_cpu_shim = True  # type: ignore[attr-defined]
        q3.get_server_args = _fake_server_args

    from sglang.srt.runtime_context import get_context

    get_context().set_server_args(SERVER_ARGS_STANDIN)
    _init_single_process_groups()
    _SHIMMED = True
    assert "sglang" in sys.modules


install_cpu_shims()
