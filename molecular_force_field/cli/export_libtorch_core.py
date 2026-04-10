#!/usr/bin/env python3
"""
Export TorchScript core model loadable by LibTorch (C++) via torch.jit.save.

Difference from torch.save(obj):
- torch.save() saves Python pickle objects (including custom classes), not directly loadable from C++.
- Pure C++ pipeline requires TorchScript files exported via torch.jit.save(ScriptModule, path).

This script exports core.pt with:
- forward signature:
    (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor) -> atom_energies
- **LAMMPS 接口仅需能量和力**：TorchScript trace 时 model 不输出物理张量（dipole/polarizability 等），
  只输出 per-atom energy；力由 C++ 侧 dE/dpos 计算。
- **Optional embedded E0** (option B): embed per-element constant energy (E0) from preprocessing/fitting
  into TorchScript; exported core.pt outputs per-atom energy as "network energy + E0(Z)".
  Note: E0 does not affect forces (constant gradient w.r.t. coordinates is zero).

Recommended usage:
- Usually pass only `--checkpoint`, `--elements`, and export/runtime options.
- Model-structure hyperparameters default to checkpoint metadata.
- If explicit CLI values conflict with checkpoint metadata, the CLI wins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _pick_device(req: str) -> str:
    if req == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    if req == "cuda":
        torch.cuda.set_device(0)
        return "cuda:0"
    return req


def _parse_dtype(s: Optional[str]) -> Optional[torch.dtype]:
    if s is None:
        return None
    mapping = {
        "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
        "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
    }
    dt = mapping.get(s.lower().strip())
    if dt is None:
        raise ValueError(f"Unsupported dtype: {s!r}, options: float32, float64")
    return dt


def _set_ictd_internal_compute_dtype_for_export(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Force ICTD-family internal TP compute dtype for deployment export only.

    We keep training/eager defaults unchanged and only rewrite the internal
    compute dtype on the model instance being traced to TorchScript.
    """
    cache_attrs = (
        "_proj_cache",
        "_recon_cache",
        "_cg_cache_by_dev_dtype",
        "_proj_group_cache_by_dev_dtype",
        "_proj_sparse_cache_by_dev_dtype",
        "_proj_bucket_cache_by_dev_dtype",
    )
    touched = 0
    for module in model.modules():
        if hasattr(module, "internal_compute_dtype"):
            setattr(module, "internal_compute_dtype", dtype)
            touched += 1
        for attr in cache_attrs:
            if hasattr(module, attr):
                cache = getattr(module, attr)
                if isinstance(cache, dict):
                    cache.clear()
    if touched:
        print(f"[export_core] forced ICTD internal_compute_dtype={str(dtype).replace('torch.', '')} on {touched} modules")


def _e0_lut_from_keys_values(
    keys: torch.Tensor, values: torch.Tensor, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build a TorchScript-friendly lookup table lut[Z] = E0(Z)."""
    keys = keys.to(dtype=torch.long, device="cpu").contiguous()
    values = values.to(dtype=dtype, device="cpu").contiguous()
    max_z = int(keys.max().item()) if keys.numel() > 0 else 0
    size = max(119, max_z + 1)  # cover periodic table by default
    lut = torch.zeros(size, dtype=dtype)
    for k, v in zip(keys.tolist(), values.tolist()):
        if 0 <= int(k) < size:
            lut[int(k)] = float(v)
    return lut.to(device=device)


_DEFAULT_CUE_NATIVE_TRACE_BUCKETS: list[tuple[str, int, int]] = [
    ("small", 648, 35804),
    ("medium", 1296, 80000),
    ("large", 4096, 262144),
]


def _default_trace_num_nodes_edges(mode: str) -> tuple[int, int]:
    if mode in {"pure-cartesian-ictd", "pure-cartesian-ictd-o3", "pure-cartesian-ictd-save", "pure-cartesian-ictd-save-o3"}:
        return 2048, 32000
    return 32, 256


def _parse_trace_buckets(spec: str | None, *, mode: str, native_ops: bool) -> list[tuple[str, int, int]]:
    if spec is None or not str(spec).strip():
        if mode == "spherical-save-cue" and native_ops:
            return list(_DEFAULT_CUE_NATIVE_TRACE_BUCKETS)
        return []
    buckets: list[tuple[str, int, int]] = []
    raw_parts = [part.strip() for part in str(spec).split(",") if part.strip()]
    for idx, part in enumerate(raw_parts):
        if ":" not in part:
            raise ValueError(
                f"Invalid trace bucket {part!r}; expected nodes:edges or name=nodes:edges"
            )
        name = f"bucket{idx + 1}"
        sizes = part
        if "=" in part:
            left, right = part.split("=", 1)
            name = left.strip()
            sizes = right.strip()
        nodes_s, edges_s = [x.strip() for x in sizes.split(":", 1)]
        nodes = int(nodes_s)
        edges = int(edges_s)
        if nodes <= 0 or edges <= 0:
            raise ValueError(f"Trace bucket must be positive, got {part!r}")
        buckets.append((name, nodes, edges))
    return buckets


class _E0WrappedModel(torch.nn.Module):
    """Wrap an eager model to add E0(Z) into per-atom energies before tracing."""

    def __init__(self, model: torch.nn.Module, e0_lut: torch.Tensor):
        super().__init__()
        self.model = model
        self.match_e0_to_output = os.environ.get("MFF_EXPORT_E0_MATCH_OUTPUT", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        conv = getattr(model, "e3_conv_emb", None)
        ext_rank = getattr(model, "external_tensor_rank", None)
        if ext_rank is None:
            ext_rank = getattr(conv, "external_tensor_rank", None)
        self.external_tensor_rank = int(ext_rank) if ext_rank is not None else None
        ext_specs = getattr(model, "external_tensor_specs", None)
        if ext_specs is None:
            ext_specs = getattr(conv, "external_tensor_specs", None)
        self.external_tensor_specs = list(ext_specs) if ext_specs is not None else None
        ext_total = getattr(model, "external_tensor_total_numel", None)
        if ext_total is None:
            ext_total = getattr(conv, "external_tensor_total_numel", None)
        self.external_tensor_total_numel = int(ext_total) if ext_total is not None else 0
        ext_irrep = getattr(model, "external_tensor_irrep", None)
        if ext_irrep is None:
            ext_irrep = getattr(conv, "external_tensor_irrep", None)
        self.external_tensor_irrep = str(ext_irrep) if ext_irrep is not None else None
        self.num_fidelity_levels = int(getattr(model, "num_fidelity_levels", 0) or 0)
        self.multi_fidelity_mode = str(getattr(model, "multi_fidelity_mode", "conditioning") or "conditioning")
        self.physical_tensor_heads = getattr(model, "physical_tensor_heads", None)
        self.has_physical_tensor_heads = (
            hasattr(model, "physical_tensor_heads") and getattr(model, "physical_tensor_heads", None) is not None
        )
        self.register_buffer("e0_lut", e0_lut)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        *,
        precomputed_edge_vec: Optional[torch.Tensor] = None,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
        sync_after_scatter=None,
    ):
        # Keep the same signature the framework expects.
        kwargs = {
            "precomputed_edge_vec": precomputed_edge_vec,
            "sync_after_scatter": sync_after_scatter,
        }
        if return_physical_tensors and self.has_physical_tensor_heads:
            kwargs["return_physical_tensors"] = True
        if return_reciprocal_source:
            kwargs["return_reciprocal_source"] = True
        if external_tensor is not None:
            kwargs["external_tensor"] = external_tensor
        if fidelity_ids is not None:
            kwargs["fidelity_ids"] = fidelity_ids
        out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)
        atom_energy = out[0] if isinstance(out, tuple) else out
        # E0 lookup: e0_lut[Z]
        e0 = self.e0_lut.index_select(0, A.to(torch.long))
        # Some exported backends may return per-atom energy only for the leading
        # physical atoms even when A includes extra ghost entries from the
        # deployment runtime. Match the E0 lookup length to the actual output
        # after the dynamic gather, so the source length is not frozen by the
        # trace-time node count.
        if self.match_e0_to_output:
            e0 = e0.narrow(0, 0, atom_energy.size(0))
        # Broadcast e0 to match out (usually (N,1)).
        if atom_energy.dim() == 2:
            e0 = e0.unsqueeze(1)
        atom_energy = atom_energy + e0.to(dtype=atom_energy.dtype, device=atom_energy.device)
        if isinstance(out, tuple):
            return (atom_energy, *out[1:])
        return atom_energy


class _FixedFidelityWrappedModel(torch.nn.Module):
    """Wrap an eager model and bind a graph-level fidelity id before tracing."""

    def __init__(self, model: torch.nn.Module, fixed_fidelity_id: int):
        super().__init__()
        self.model = model
        self.fixed_fidelity_id = int(fixed_fidelity_id)
        conv = getattr(model, "e3_conv_emb", None)
        ext_rank = getattr(model, "external_tensor_rank", None)
        if ext_rank is None:
            ext_rank = getattr(conv, "external_tensor_rank", None)
        self.external_tensor_rank = int(ext_rank) if ext_rank is not None else None
        ext_specs = getattr(model, "external_tensor_specs", None)
        if ext_specs is None:
            ext_specs = getattr(conv, "external_tensor_specs", None)
        self.external_tensor_specs = list(ext_specs) if ext_specs is not None else None
        ext_total = getattr(model, "external_tensor_total_numel", None)
        if ext_total is None:
            ext_total = getattr(conv, "external_tensor_total_numel", None)
        self.external_tensor_total_numel = int(ext_total) if ext_total is not None else 0
        ext_irrep = getattr(model, "external_tensor_irrep", None)
        if ext_irrep is None:
            ext_irrep = getattr(conv, "external_tensor_irrep", None)
        self.external_tensor_irrep = str(ext_irrep) if ext_irrep is not None else None
        self.num_fidelity_levels = int(getattr(model, "num_fidelity_levels", 0) or 0)
        self.multi_fidelity_mode = str(getattr(model, "multi_fidelity_mode", "conditioning") or "conditioning")
        self.physical_tensor_heads = getattr(model, "physical_tensor_heads", None)
        self.has_physical_tensor_heads = (
            hasattr(model, "physical_tensor_heads") and getattr(model, "physical_tensor_heads", None) is not None
        )

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        *,
        precomputed_edge_vec: Optional[torch.Tensor] = None,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
        sync_after_scatter=None,
    ):
        kwargs = {
            "precomputed_edge_vec": precomputed_edge_vec,
            "sync_after_scatter": sync_after_scatter,
        }
        if return_physical_tensors and self.has_physical_tensor_heads:
            kwargs["return_physical_tensors"] = True
        if return_reciprocal_source:
            kwargs["return_reciprocal_source"] = True
        if external_tensor is not None:
            kwargs["external_tensor"] = external_tensor
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        kwargs["fidelity_ids"] = torch.full(
            (num_graphs,),
            self.fixed_fidelity_id,
            device=batch.device,
            dtype=torch.long,
        )
        return self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)


@torch.jit.interface
class _ExportOnlyTraceCoreInterface:
    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


@torch.jit.interface
class _ExportOnlyTraceCoreInterfaceWithFidelity:
    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
        fidelity_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class _ExportOnlyTraceCoreAdapter(torch.nn.Module):
    """Script-friendly positional export wrapper around a traced numeric core."""

    def __init__(self, core):
        super().__init__()
        self.core = torch.jit.Attribute(core, _ExportOnlyTraceCoreInterface)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
    ):
        return self.core.forward(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor)


class _ExportOnlyTraceCoreAdapterWithFidelity(torch.nn.Module):
    """Script-friendly positional export wrapper around a traced numeric core."""

    def __init__(self, core):
        super().__init__()
        self.core = torch.jit.Attribute(core, _ExportOnlyTraceCoreInterfaceWithFidelity)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
        fidelity_ids: torch.Tensor,
    ):
        return self.core.forward(
            pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor, fidelity_ids
        )


def _trace_model_for_export(
    model: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    export_reciprocal_source: bool = False,
    trace_num_nodes: int,
    trace_num_edges: int,
    jit_mode: str = "trace",
) -> torch.jit.ScriptModule:
    from molecular_force_field.interfaces.lammps_mliap import (
        _TorchScriptEdgeVecCore,
        _TorchScriptEdgeVecCoreWithFidelity,
        _resolve_model_external_tensor_total_numel,
    )

    model.eval()
    num_fidelity_levels = int(getattr(model, "num_fidelity_levels", 0) or 0)
    fixed_fidelity_id = getattr(model, "fixed_fidelity_id", None)
    runtime_fidelity = num_fidelity_levels > 0 and fixed_fidelity_id is None
    core_cls = _TorchScriptEdgeVecCoreWithFidelity if runtime_fidelity else _TorchScriptEdgeVecCore
    core = core_cls(model, export_reciprocal_source=export_reciprocal_source).to(device=device)

    ext_total_numel = _resolve_model_external_tensor_total_numel(model)
    N = max(int(trace_num_nodes), 1)
    E = max(int(trace_num_edges), 1)
    pos = torch.zeros(N, 3, device=device, dtype=dtype)
    A = torch.ones(N, device=device, dtype=torch.long)
    batch = torch.zeros(N, device=device, dtype=torch.long)
    edge_src = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_dst = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
    cell = (torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 100.0)
    edge_vec = torch.randn(E, 3, device=device, dtype=dtype)
    external_tensor = (
        torch.empty(0, device=device, dtype=dtype)
        if ext_total_numel <= 0
        else torch.zeros(ext_total_numel, device=device, dtype=dtype)
    )
    fidelity_ids = torch.zeros(1, device=device, dtype=torch.long)

    try:
        with torch.no_grad():
            for m in core.modules():
                prewarm = getattr(m, "prewarm_caches", None)
                if callable(prewarm):
                    prewarm(device=device, dtype=dtype)
            if runtime_fidelity:
                _ = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor, fidelity_ids)
            else:
                _ = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor)
    except Exception:
        pass

    trace_inputs = (
        (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor, fidelity_ids)
        if runtime_fidelity
        else (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor)
    )
    core_ts = torch.jit.trace(core, trace_inputs, check_trace=False, strict=False)
    try:
        core_ts = torch.jit.freeze(core_ts.eval())
    except Exception:
        core_ts = core_ts.eval()
    if jit_mode == "trace":
        return core_ts
    if jit_mode != "hybrid":
        raise ValueError(f"Unsupported jit_mode={jit_mode!r}; expected 'trace' or 'hybrid'")
    adapter: torch.nn.Module
    if runtime_fidelity:
        adapter = _ExportOnlyTraceCoreAdapterWithFidelity(core_ts)
    else:
        adapter = _ExportOnlyTraceCoreAdapter(core_ts)
    try:
        hybrid_ts = torch.jit.script(adapter)
    except Exception as exc:
        raise RuntimeError(
            "jit_mode=hybrid is still experimental for this model/backend combination; "
            "TorchScript rejected the export-only wrapper around the traced numeric core."
        ) from exc
    try:
        hybrid_ts = torch.jit.freeze(hybrid_ts.eval())
    except Exception:
        hybrid_ts = hybrid_ts.eval()
    return hybrid_ts


def _export_single_core(
    *,
    checkpoint: str,
    elements: List[str],
    device: str,
    max_radius: Optional[float],
    num_interaction: Optional[int],
    out_pt: str,
    tensor_product_mode: Optional[str] = None,
    force_dtype: Optional[torch.dtype] = None,
    embed_e0: bool = True,
    e0_csv: Optional[str] = None,
    native_ops: bool = False,
    export_reciprocal_source: bool = False,
    export_fidelity_id: int | None = None,
    trace_num_nodes: int | None = None,
    trace_num_edges: int | None = None,
    jit_mode: str = "trace",
) -> dict:
    from molecular_force_field.interfaces.lammps_mliap import (
        LAMMPS_MLIAP_MFF,
        _resolve_model_external_tensor_rank,
        _resolve_model_external_tensor_specs,
        _resolve_model_external_tensor_total_numel,
    )
    from molecular_force_field.utils.config import ModelConfig

    _ts_supported = (
        "pure-cartesian-sparse",
        "pure-cartesian-sparse-save",
        "pure-cartesian-ictd",
        "pure-cartesian-ictd-o3",
        "pure-cartesian-ictd-save",
        "pure-cartesian-ictd-save-o3",
        "spherical-save-cue",
    )

    # Resolve mode and metadata: explicit CLI override > checkpoint > fallback.
    if tensor_product_mode is not None:
        mode = tensor_product_mode
    else:
        ckpt_peek = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mode = ckpt_peek.get("tensor_product_mode", None)
        del ckpt_peek
        if mode is None:
            raise ValueError(
                f"tensor_product_mode not saved in checkpoint; specify via --mode."
                f"\nTorchScript-supported modes: {_ts_supported}"
            )
        print(f"[export_core] Read tensor_product_mode={mode!r} from checkpoint")
    ckpt_peek = torch.load(checkpoint, map_location="cpu", weights_only=False)
    effective_radius = float(ckpt_peek.get("max_radius", max_radius if max_radius is not None else 5.0))
    if num_interaction is None:
        num_interaction = int(ckpt_peek.get("model_hyperparameters", {}).get("num_interaction", 2))
    del ckpt_peek
    if mode not in _ts_supported:
        raise ValueError(
            f"Mode {mode!r} does not support TorchScript export."
            f"\nSupported modes: {_ts_supported}"
        )
    default_nodes, default_edges = _default_trace_num_nodes_edges(mode)
    if trace_num_nodes is None:
        trace_num_nodes = default_nodes
    if trace_num_edges is None:
        trace_num_edges = default_edges

    # Load atomic E0 from preprocessing output if requested.
    atomic_energy_keys = None
    atomic_energy_values = None
    if e0_csv:
        cfg = ModelConfig(dtype=torch.float64)
        cfg.load_atomic_energies_from_file(e0_csv)
        atomic_energy_keys = cfg.atomic_energy_keys.tolist()
        atomic_energy_values = [float(x) for x in cfg.atomic_energy_values.tolist()]

    # spherical-save-cue uses cuEquivariance custom ops on CUDA.
    # --native-ops: keep those ops in core.pt (requires MFF_CUSTOM_OPS_LIB at LAMMPS runtime).
    # default:      build on CUDA with force_naive (pure-PyTorch ops, correct device constants).
    force_naive = False
    if mode == "spherical-save-cue" and not native_ops:
        build_device = device
        trace_device = device
        force_naive = True
        print("[export_core] spherical-save-cue: built with CUDA + force_naive, core.pt needs no cuEquivariance runtime")
    elif mode == "spherical-save-cue" and native_ops:
        build_device = device
        trace_device = device
        print("[export_core] spherical-save-cue --native-ops: using native cuEquivariance CUDA ops")
        print("[export_core]   LAMMPS runtime must set MFF_CUSTOM_OPS_LIB to cuequivariance ops .so")
    else:
        build_device = device
        trace_device = device

    obj = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=checkpoint,
        element_types=elements,
        max_radius=effective_radius,
        atomic_energy_keys=atomic_energy_keys,
        atomic_energy_values=atomic_energy_values,
        device=build_device,
        tensor_product_mode=mode,
        num_interaction=num_interaction,
        torchscript=False,
        force_naive=force_naive,
    )

    actual_dtype = obj.dtype
    if force_dtype is not None:
        actual_dtype = force_dtype
        obj.wrapper = obj.wrapper.to(dtype=force_dtype)
        obj.wrapper.model = obj.wrapper.model.to(dtype=force_dtype)

    model_eager = obj.wrapper.model
    metadata_model = model_eager
    num_fidelity_levels = int(getattr(model_eager, "num_fidelity_levels", 0) or 0)
    multi_fidelity_mode = str(getattr(model_eager, "multi_fidelity_mode", "conditioning") or "conditioning")
    runtime_fidelity_input = False
    if num_fidelity_levels > 0:
        if export_fidelity_id is not None:
            if int(export_fidelity_id) < 0 or int(export_fidelity_id) >= num_fidelity_levels:
                raise ValueError(
                    f"--export-fidelity-id must be in [0, {num_fidelity_levels - 1}], got {export_fidelity_id}"
                )
            model_eager = _FixedFidelityWrappedModel(model_eager, int(export_fidelity_id)).to(
                device=torch.device(trace_device)
            )
        else:
            runtime_fidelity_input = True
    external_tensor_rank = _resolve_model_external_tensor_rank(model_eager)
    external_tensor_specs = _resolve_model_external_tensor_specs(model_eager)
    external_tensor_total_numel = _resolve_model_external_tensor_total_numel(model_eager)
    external_tensor_irrep = getattr(model_eager, "external_tensor_irrep", None)
    if external_tensor_irrep is None:
        conv = getattr(model_eager, "e3_conv_emb", None)
        external_tensor_irrep = getattr(conv, "external_tensor_irrep", None)
    o3_irrep_preset = getattr(model_eager, "o3_irrep_preset", None)
    o3_active_irreps = getattr(model_eager, "active_irreps_str", None)

    if mode == "spherical-save-cue" and not native_ops:
        if hasattr(model_eager, "make_torchscript_portable"):
            model_eager.make_torchscript_portable()
            print("[export_core] Replaced product_3/product_5 with pure PyTorch impl (no cuequivariance custom ops)")

    if mode in {"pure-cartesian-ictd", "pure-cartesian-ictd-save"}:
        _set_ictd_internal_compute_dtype_for_export(model_eager, torch.float32)

    # Optional: embed E0(Z) into per-atom energies before tracing.
    if embed_e0:
        aek = obj.wrapper.atomic_energy_keys.detach().cpu()
        aev = obj.wrapper.atomic_energy_values.detach().cpu()
        lut = _e0_lut_from_keys_values(aek, aev, dtype=actual_dtype, device=torch.device(trace_device))
        model_eager = _E0WrappedModel(model_eager, lut).to(device=torch.device(trace_device))

    # Trace to TorchScript core (edge_vec positional arg) and export its ScriptModule.
    core = _trace_model_for_export(
        model_eager,
        device=torch.device(trace_device),
        dtype=actual_dtype,
        export_reciprocal_source=export_reciprocal_source,
        trace_num_nodes=int(trace_num_nodes),
        trace_num_edges=int(trace_num_edges),
        jit_mode=jit_mode,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    core.eval()
    torch.jit.save(core, out_pt)
    print(f"Exported LibTorch-loadable TorchScript core: {out_pt}")

    runtime_backend = str(getattr(metadata_model, "long_range_runtime_backend", "none"))
    runtime_source_kind = str(getattr(metadata_model, "long_range_runtime_source_kind", "none"))
    runtime_source_channels = int(getattr(metadata_model, "long_range_runtime_source_channels", 0))
    runtime_source_layout = str(getattr(metadata_model, "long_range_runtime_source_layout", "none"))
    runtime_source_boundary = str(getattr(metadata_model, "long_range_runtime_source_boundary", "periodic"))
    runtime_source_slab_padding_factor = int(
        getattr(metadata_model, "long_range_runtime_source_slab_padding_factor", 2)
    )
    if not export_reciprocal_source:
        runtime_backend = "none"
        runtime_source_kind = "none"
        runtime_source_channels = 0
        runtime_source_layout = "none"
    long_range_module = getattr(metadata_model, "long_range_module", None)
    long_range_screening = None
    long_range_softening = None
    long_range_energy_scale = None
    if long_range_module is not None and hasattr(long_range_module, "screening_raw"):
        long_range_screening = float(F.softplus(long_range_module.screening_raw.detach()).cpu().item())
    if long_range_module is not None and hasattr(long_range_module, "softening_raw"):
        long_range_softening = float((F.softplus(long_range_module.softening_raw.detach()) + 1.0e-6).cpu().item())
    if long_range_module is not None and hasattr(long_range_module, "energy_scale"):
        energy_scale = getattr(long_range_module, "energy_scale")
        if energy_scale is not None:
            long_range_energy_scale = float(energy_scale.detach().cpu().item())

    meta = {
        "elements": elements,
        "tensor_product_mode": mode,
        "device_exported_from": device,
        "max_radius": float(effective_radius),
        "num_interaction": int(num_interaction) if num_interaction is not None else None,
        "dtype": str(actual_dtype).replace("torch.", ""),
        "internal_compute_dtype_export": (
            "float32" if mode in {"pure-cartesian-ictd", "pure-cartesian-ictd-save"} else str(actual_dtype).replace("torch.", "")
        ),
        "embed_e0": bool(embed_e0),
        "trace_num_nodes": int(trace_num_nodes),
        "trace_num_edges": int(trace_num_edges),
        "jit_mode": str(jit_mode),
        "export_reciprocal_source": bool(export_reciprocal_source),
        "e0_source": (str(e0_csv) if e0_csv else "from_checkpoint_or_default"),
        "forward_signature": [
            "pos(N,3)",
            "A(N,) atomic number (int64)",
            "batch(N,) (int64)",
            "edge_src(E,) (int64)",
            "edge_dst(E,) (int64)",
            "edge_shifts(E,3)",
            "cell(1,3,3)",
            "edge_vec(E,3)",
            "external_tensor(rank-dependent tensor or empty tensor)",
        ] + (["fidelity_ids(n_graphs,) int64"] if runtime_fidelity_input else []),
        "external_tensor_rank": (
            int(external_tensor_rank) if external_tensor_rank is not None else None
        ),
        "external_tensor_irrep": (str(external_tensor_irrep) if external_tensor_irrep is not None else None),
        "external_tensor_specs": external_tensor_specs,
        "external_tensor_total_numel": int(external_tensor_total_numel),
        "external_tensor_has_field_1o": bool(
            external_tensor_specs is not None and any(str(spec.get("name")) == "external_field" and str(spec.get("irrep")) == "1o" for spec in external_tensor_specs)
        ),
        "external_tensor_has_field_1e": bool(
            external_tensor_specs is not None and any(str(spec.get("name")) == "magnetic_field" and str(spec.get("irrep")) == "1e" for spec in external_tensor_specs)
        ),
        "o3_irrep_preset": (str(o3_irrep_preset) if o3_irrep_preset is not None else None),
        "o3_active_irreps": list(o3_active_irreps) if o3_active_irreps is not None else None,
        "num_fidelity_levels": int(num_fidelity_levels),
        "multi_fidelity_mode": str(multi_fidelity_mode),
        "export_fidelity_id": (int(export_fidelity_id) if export_fidelity_id is not None else None),
        "runtime_fidelity_input": bool(runtime_fidelity_input),
        "reciprocal_source_channels": runtime_source_channels,
        "reciprocal_source_boundary": runtime_source_boundary,
        "reciprocal_source_slab_padding_factor": int(
            runtime_source_slab_padding_factor
        ),
        "long_range_runtime_backend": runtime_backend,
        "long_range_source_kind": runtime_source_kind,
        "long_range_source_channels": runtime_source_channels,
        "long_range_source_layout": runtime_source_layout,
        "long_range_source_boundary": runtime_source_boundary,
        "long_range_source_slab_padding_factor": runtime_source_slab_padding_factor,
        "long_range_boundary": str(getattr(metadata_model, "long_range_boundary", "nonperiodic")),
        "long_range_backend": str(getattr(metadata_model, "long_range_backend", "dense_pairwise")),
        "long_range_mesh_size": int(getattr(metadata_model, "long_range_mesh_size", 16)),
        "long_range_slab_padding_factor": int(getattr(metadata_model, "long_range_slab_padding_factor", 2)),
        "long_range_reciprocal_backend": str(getattr(metadata_model, "long_range_reciprocal_backend", "direct_kspace")),
        "long_range_energy_partition": str(getattr(metadata_model, "long_range_energy_partition", "potential")),
        "long_range_neutralize": bool(getattr(metadata_model, "long_range_neutralize", True)),
        "long_range_green_mode": str(getattr(metadata_model, "long_range_green_mode", "poisson")),
        "long_range_mesh_fft_full_ewald": bool(getattr(metadata_model, "long_range_mesh_fft_full_ewald", False)),
        "long_range_theta": float(getattr(metadata_model, "long_range_theta", 0.5)),
        "long_range_leaf_size": int(getattr(metadata_model, "long_range_leaf_size", 32)),
        "long_range_multipole_order": int(getattr(metadata_model, "long_range_multipole_order", 0)),
        "long_range_screening": long_range_screening,
        "long_range_softening": long_range_softening,
        "long_range_energy_scale": long_range_energy_scale,
        "feature_spectral_boundary": str(getattr(metadata_model, "feature_spectral_boundary", "periodic")),
        "feature_spectral_slab_padding_factor": int(
            getattr(metadata_model, "feature_spectral_slab_padding_factor", 2)
        ),
        "feature_spectral_assignment": str(getattr(metadata_model, "feature_spectral_assignment", "cic")),
        "notes": [
            "Core model: outputs per-atom energy.",
            "If export_reciprocal_source=true: core tuple includes reciprocal_source as the last tensor.",
            "The reciprocal_source output slot is kept for backward compatibility and may carry a generic runtime long-range source.",
            "If embed_e0=true: output includes E0(Z) constant bias (from preprocessing fit or e0_csv).",
            "For multi-fidelity checkpoints, omitting export_fidelity_id exports a runtime fidelity_ids input.",
            "For multi-fidelity checkpoints, export_fidelity_id freezes one fidelity branch into the exported core.pt.",
            "Forces: dE/d(pos) computed via autograd on C++ side.",
            "Loadable: C++ torch::jit::load(path).",
            "external_tensor is required at runtime when external_tensor_rank is not null.",
            "jit_mode=hybrid is experimental and should be benchmarked before production use.",
        ],
    }
    meta_path = out_pt + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote metadata: {meta_path}")
    return meta


def export_core(
    *,
    checkpoint: str,
    elements: List[str],
    device: str,
    max_radius: Optional[float],
    num_interaction: Optional[int],
    out_pt: str,
    tensor_product_mode: Optional[str] = None,
    force_dtype: Optional[torch.dtype] = None,
    embed_e0: bool = True,
    e0_csv: Optional[str] = None,
    native_ops: bool = False,
    export_reciprocal_source: bool = False,
    export_fidelity_id: int | None = None,
    trace_num_nodes: int | None = None,
    trace_num_edges: int | None = None,
    bundle_out: str | None = None,
    trace_buckets: str | None = None,
    jit_mode: str | None = None,
) -> None:
    ckpt_peek = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mode = tensor_product_mode if tensor_product_mode is not None else ckpt_peek.get("tensor_product_mode", None)
    del ckpt_peek
    if mode is None:
        raise ValueError("tensor_product_mode not saved in checkpoint; specify via --mode")

    effective_jit_mode = str(jit_mode) if jit_mode is not None else (
        "hybrid"
        if (
            (mode == "spherical-save-cue" and native_ops)
            or mode in {"pure-cartesian-ictd", "pure-cartesian-ictd-o3", "pure-cartesian-ictd-save", "pure-cartesian-ictd-save-o3"}
        )
        else "trace"
    )

    bucket_defs = _parse_trace_buckets(trace_buckets, mode=mode, native_ops=bool(native_ops))
    if bundle_out is None:
        _export_single_core(
            checkpoint=checkpoint,
            elements=elements,
            device=device,
            max_radius=max_radius,
            num_interaction=num_interaction,
            out_pt=out_pt,
            tensor_product_mode=mode,
            force_dtype=force_dtype,
            embed_e0=embed_e0,
            e0_csv=e0_csv,
            native_ops=native_ops,
            export_reciprocal_source=export_reciprocal_source,
            export_fidelity_id=export_fidelity_id,
            trace_num_nodes=trace_num_nodes,
            trace_num_edges=trace_num_edges,
            jit_mode=effective_jit_mode,
        )
        return

    bundle_dir = Path(bundle_out).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if not bucket_defs:
        n_default, e_default = _default_trace_num_nodes_edges(mode)
        bucket_defs = [("default", int(trace_num_nodes or n_default), int(trace_num_edges or e_default))]

    variant = "cue-native-kk" if mode == "spherical-save-cue" and native_ops else mode
    bucket_entries: list[dict] = []
    for name, nodes, edges in bucket_defs:
        core_name = f"core_{name}.pt"
        core_path = bundle_dir / core_name
        meta = _export_single_core(
            checkpoint=checkpoint,
            elements=elements,
            device=device,
            max_radius=max_radius,
            num_interaction=num_interaction,
            out_pt=str(core_path),
            tensor_product_mode=mode,
            force_dtype=force_dtype,
            embed_e0=embed_e0,
            e0_csv=e0_csv,
            native_ops=native_ops,
            export_reciprocal_source=export_reciprocal_source,
            export_fidelity_id=export_fidelity_id,
            trace_num_nodes=nodes,
            trace_num_edges=edges,
            jit_mode=effective_jit_mode,
        )
        bucket_entries.append(
            {
                "name": str(name),
                "core_path": core_name,
                "max_nodes": int(nodes),
                "max_edges": int(edges),
                "dtype": str(meta["dtype"]),
                "trace_num_nodes": int(meta["trace_num_nodes"]),
                "trace_num_edges": int(meta["trace_num_edges"]),
                "jit_mode": str(meta.get("jit_mode", effective_jit_mode)),
            }
        )

    manifest = {
        "version": 1,
        "tensor_product_mode": mode,
        "variant": variant,
        "selector": "smallest-fitting",
        "jit_mode": str(effective_jit_mode),
        "buckets": bucket_entries,
    }
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote bundle manifest: {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export LibTorch-loadable TorchScript core model. "
                    "Model-structure hyperparameters default to checkpoint metadata; explicit CLI values override."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint (.pth). Structure hyperparameters are resolved with priority: "
                        "explicit CLI > checkpoint metadata > defaults.")
    p.add_argument("--elements", nargs="+", default=["H", "O"], help="Element order (LAMMPS type order)")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--mode", type=str, default=None,
                   help="Model mode. If not set, restore from checkpoint metadata. "
                        "Supported: pure-cartesian-ictd, pure-cartesian-ictd-o3, pure-cartesian-ictd-save, pure-cartesian-ictd-save-o3, spherical-save-cue")
    p.add_argument("--max-radius", type=float, default=None,
                   help="Override checkpoint cutoff radius (Å). If not set, restore from checkpoint metadata.")
    p.add_argument("--num-interaction", type=int, default=None,
                   help="Override checkpoint num_interaction. If not set, restore from checkpoint metadata.")
    p.add_argument("--dtype", type=str, default=None,
                   help="Force export precision: float32 or float64. If not set, follow checkpoint metadata.")
    p.add_argument("--embed-e0", action="store_true",
                   help="Backward-compatible alias. E0 embedding is now enabled by default unless --no-embed-e0 is passed. "
                        "When enabled, E0 is taken from --e0-csv if provided, else from checkpoint when available.")
    p.add_argument("--no-embed-e0", action="store_true", help="Do not embed E0 into TorchScript (export network energy only)")
    p.add_argument("--e0-csv", type=str, default=None,
                   help="E0 CSV path (Atom,E0 columns). Highest priority override for E0; "
                        "if not set, prefer checkpoint E0 when available.")
    p.add_argument("--native-ops", action="store_true",
                   help="spherical-save-cue: keep native cuEquivariance CUDA ops (faster, but LAMMPS requires MFF_CUSTOM_OPS_LIB). "
                        "Default: pure PyTorch ops (portable, no extra deps).")
    p.add_argument("--export-reciprocal-source", action="store_true",
                   help="Export core.pt with an additional reciprocal_source output tensor for the USER-MFFTORCH reciprocal solver. "
                        "Enable this when the checkpoint uses long-range reciprocal runtime evaluation "
                        "(recommended LES-style setup: reciprocal-spectral-v1 + mesh_fft + poisson + potential + cic).")
    p.add_argument("--export-fidelity-id", type=int, default=None,
                   help="For multi-fidelity checkpoints, freeze a single graph-level fidelity id into the exported core.pt. "
                        "If omitted, core.pt keeps a runtime fidelity_ids input.")
    p.add_argument("--trace-num-nodes", type=int, default=None,
                   help="Representative node count used during TorchScript tracing. "
                        "Default: 2048 for pure-cartesian-ictd / pure-cartesian-ictd-o3 / pure-cartesian-ictd-save / pure-cartesian-ictd-save-o3, else 32.")
    p.add_argument("--trace-num-edges", type=int, default=None,
                   help="Representative edge count used during TorchScript tracing. "
                        "Default: 32000 for pure-cartesian-ictd / pure-cartesian-ictd-o3 / pure-cartesian-ictd-save / pure-cartesian-ictd-save-o3, else 256.")
    p.add_argument("--bundle-out", type=str, default=None,
                   help="Export a multi-core bundle directory instead of a single core.pt. "
                        "Writes per-bucket cores plus manifest.json.")
    p.add_argument("--trace-buckets", type=str, default=None,
                   help="Comma-separated bucket list for --bundle-out. Format: nodes:edges or name=nodes:edges. "
                        "Default for spherical-save-cue --native-ops: small=648:35804,medium=1296:80000,large=4096:262144.")
    p.add_argument("--jit-mode", type=str, default=None, choices=["trace", "hybrid"],
                   help="Export mode. Default: hybrid for spherical-save-cue with --native-ops and for pure-cartesian-ictd / pure-cartesian-ictd-o3 / pure-cartesian-ictd-save / pure-cartesian-ictd-save-o3; else trace. "
                        "hybrid scripts an export-only wrapper around the traced numeric core.")
    p.add_argument("--out", type=str, default="core.pt", help="Output TorchScript file path")
    args = p.parse_args()

    device = _pick_device(args.device)
    force_dtype = _parse_dtype(args.dtype)
    embed_e0 = not bool(args.no_embed_e0)
    export_core(
        checkpoint=args.checkpoint,
        elements=list(args.elements),
        device=device,
        max_radius=(float(args.max_radius) if args.max_radius is not None else None),
        num_interaction=(int(args.num_interaction) if args.num_interaction is not None else None),
        out_pt=str(args.out),
        tensor_product_mode=args.mode,
        force_dtype=force_dtype,
        embed_e0=embed_e0,
        e0_csv=args.e0_csv,
        native_ops=bool(args.native_ops),
        export_reciprocal_source=bool(args.export_reciprocal_source),
        export_fidelity_id=(int(args.export_fidelity_id) if args.export_fidelity_id is not None else None),
        trace_num_nodes=(int(args.trace_num_nodes) if args.trace_num_nodes is not None else None),
        trace_num_edges=(int(args.trace_num_edges) if args.trace_num_edges is not None else None),
        bundle_out=args.bundle_out,
        trace_buckets=args.trace_buckets,
        jit_mode=(str(args.jit_mode) if args.jit_mode is not None else None),
    )


if __name__ == "__main__":
    main()
