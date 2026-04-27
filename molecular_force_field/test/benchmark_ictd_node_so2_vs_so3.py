from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [str(REPO_ROOT)] + [p for p in sys.path if p != str(REPO_ROOT)]

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_node_so2 import (
    PureCartesianICTDNodeSO2TransformerLayer,
)


def _make_graph(
    device: torch.device,
    dtype: torch.dtype,
    *,
    num_nodes: int = 256,
    avg_degree: int = 24,
    seed: int = 0,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype, generator=g) * 2.0
    A = torch.randint(1, 6, (num_nodes,), device=device, generator=g)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device, generator=g)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _bench_model(model_cls, *, warmup: int = 3, steps: int = 10) -> dict[str, float | int | bool]:
    device = torch.device("cuda")
    dtype = torch.float32
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_graph(device, dtype)
    model = model_cls(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=16,
        hidden_dim_conv=32,
        hidden_dim_sh=32,
        hidden_dim=32,
        lmax=2,
        num_interaction=3,
        save_readout_mode="multiple-contraction",
        internal_compute_dtype=torch.float64,
    ).to(device=device, dtype=dtype)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        p = pos.detach().clone().requires_grad_(True)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        out.square().mean().backward()
        opt.step()
    torch.cuda.synchronize()

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    step_ms: list[float] = []
    peak_mem_mb = 0.0
    out_mean = 0.0
    out_std = 0.0
    grad_norm = 0.0
    finite = True
    for _ in range(steps):
        p = pos.detach().clone().requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = out.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        fwd_ms.append((t1 - t0) * 1000.0)
        bwd_ms.append((t2 - t1) * 1000.0)
        step_ms.append((t3 - t0) * 1000.0)
        peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
        out_mean = float(out.mean().item())
        out_std = float(out.std().item())
        grad_norm = float(p.grad.norm().item())
        finite = finite and bool(torch.isfinite(out).all() and torch.isfinite(p.grad).all())

    return {
        "params": _count_params(model),
        "forward_ms_mean": statistics.mean(fwd_ms),
        "forward_ms_median": statistics.median(fwd_ms),
        "forward_ms_std": statistics.pstdev(fwd_ms),
        "backward_ms_mean": statistics.mean(bwd_ms),
        "backward_ms_median": statistics.median(bwd_ms),
        "backward_ms_std": statistics.pstdev(bwd_ms),
        "step_ms_mean": statistics.mean(step_ms),
        "step_ms_median": statistics.median(step_ms),
        "step_ms_std": statistics.pstdev(step_ms),
        "peak_mem_mb": peak_mem_mb,
        "out_mean": out_mean,
        "out_std": out_std,
        "grad_norm": grad_norm,
        "finite": finite,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "config": {
                    "save_readout_mode": "multiple-contraction",
                    "lmax": 2,
                    "hidden_dim_conv": 32,
                    "num_interaction": 3,
                    "main_number_of_basis": 16,
                    "internal_compute_dtype": "torch.float64",
                    "num_nodes": 256,
                    "avg_degree": 24,
                },
            },
            ensure_ascii=False,
        )
    )
    results = {
        "so3": _bench_model(PureCartesianICTDTransformerLayer),
        "node_so2": _bench_model(PureCartesianICTDNodeSO2TransformerLayer),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
