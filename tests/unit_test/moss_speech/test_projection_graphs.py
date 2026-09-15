# SPDX-License-Identifier: Apache-2.0
"""Static projection graphs must replay fresh inputs without changing math."""
import pytest
import torch

from sglang_omni.models.moss_speech.projection_graphs import ProjectionGraph


def test_projection_graph_rejects_cpu_inputs():
    with pytest.raises(ValueError, match="CUDA"):
        ProjectionGraph(lambda x: x + 1, (torch.ones(2),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_projection_graph_replays_new_inputs_bit_exact(batch_size):
    torch.manual_seed(731)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(batch_size, 64, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    def forward(hidden, residual, optional):
        assert optional is None
        summed = hidden + residual
        normalized = (
            summed.float()
            * torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        return torch.nn.functional.linear(normalized, weight), summed

    graph = ProjectionGraph(forward, (x, residual, None))
    for i in range(3):
        x = torch.randn_like(x)
        expected = forward(x, residual, None)
        actual = graph(x, residual, None)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert graph.replay_count == i + 1
    with pytest.raises(ValueError, match="shape"):
        graph(x[:0], residual, None)
