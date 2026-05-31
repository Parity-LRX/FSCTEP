#!/usr/bin/env python3
"""Integration + safety check for Trainer.train_compiled_autograd.

Drives the EXACT trainer method (Trainer._train_step_backward) on the real model
for both routes and verifies:
  - baseline route: compiled-autograd compiles the double-backward and gradients
    match eager to machine precision; feature stays enabled.
  - fusion route: compiled-autograd raises (known PyTorch limitation); the probe
    catches it, permanently disables the feature, and falls back to an eager
    backward that still produces correct gradients. Training never crashes.

Run: python -m molecular_force_field.test.check_train_compiled_autograd
"""
from __future__ import annotations

import types

import torch

from molecular_force_field.training.trainer import Trainer
from molecular_force_field.test.bench_ictd_fix_trainstep import (
    build_model, make_fixed_graph, _flat_grads,
)


def _build_stub(model):
    stub = types.SimpleNamespace(
        train_compiled_autograd=True,
        distributed=False,
        e3trans=model,
        _compiled_autograd_warned=False,
        _ca_checked=False,
        _ca_disabled=False,
    )
    for name in ("_compiled_autograd_ctx",
                 "_maybe_freeze_fusion_bias_for_compiled_autograd",
                 "_train_step_backward"):
        setattr(stub, name, types.MethodType(getattr(Trainer, name), stub))
    return stub


def _run_route(route: str) -> bool:
    dev = torch.device("cuda")
    dt = torch.float64
    torch.manual_seed(0)
    model = build_model(channels=32, lmax=2, num_interaction=2, route=route,
                        product_backend="ictd-pure-u", dtype=dt, device=dev)
    model.eval()
    graph = make_fixed_graph(num_nodes=128, avg_degree=24, dtype=dt, device=dev)
    rest = tuple(graph[1:])

    stub = _build_stub(model)
    # The trainer's real auto-freeze: no-op for baseline; freezes fusion_readouts biases for fusion.
    stub._maybe_freeze_fusion_bias_for_compiled_autograd()
    n_frozen = sum(1 for n, p in model.named_parameters()
                   if n.endswith(".bias") and "fusion_readouts" in n and not p.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    stub.optimizer = torch.optim.SGD(params, lr=0.0)

    def fresh_loss():
        for p in params:
            p.grad = None
        pos = graph[0].detach().clone().requires_grad_(True)
        e = model(pos, *rest)
        if isinstance(e, tuple):
            e = e[0]
        energy = e.sum()
        g = torch.autograd.grad(energy, pos, create_graph=True)[0]
        return energy ** 2 + 10.0 * ((-g) ** 2).mean()

    # eager reference (force loss)
    fresh_loss().backward()
    ge = _flat_grads(params).clone()

    # via the trainer's real backward path
    import torch._dynamo as dynamo
    dynamo.reset()
    for p in params:
        p.grad = None
    stub._train_step_backward(fresh_loss())
    torch.cuda.synchronize()
    gt = _flat_grads(params).clone()

    dgrad = (gt - ge).abs().max().item()
    rel = dgrad / (ge.abs().max().item() + 1e-30)
    used_ca = not stub._ca_disabled
    grads_ok = rel <= 1e-9
    frozen_ok = (n_frozen > 0) if route == "fusion" else (n_frozen == 0)
    print(f"[{route}] frozen_fusion_bias={n_frozen} compiled_autograd_used={used_ca} "
          f"grads rel|d|={rel:.3e} grads_ok={grads_ok}")
    # After auto-freeze, compiled-autograd should be USED (not fallback) for BOTH routes.
    route_ok = grads_ok and used_ca and frozen_ok
    print(f"[{route}] {'PASS' if route_ok else 'FAIL'}")
    return route_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("requires CUDA"); return 1
    ok = True
    for route in ("baseline", "fusion"):
        ok &= _run_route(route)
    print(f"[overall] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
