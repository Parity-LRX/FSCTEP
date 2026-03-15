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


class _E0WrappedModel(torch.nn.Module):
    """Wrap an eager model to add E0(Z) into per-atom energies before tracing."""

    def __init__(self, model: torch.nn.Module, e0_lut: torch.Tensor):
        super().__init__()
        self.model = model
        conv = getattr(model, "e3_conv_emb", None)
        ext_rank = getattr(model, "external_tensor_rank", None)
        if ext_rank is None:
            ext_rank = getattr(conv, "external_tensor_rank", None)
        self.external_tensor_rank = int(ext_rank) if ext_rank is not None else None
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
        out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)
        # E0 lookup: e0_lut[Z]
        e0 = self.e0_lut.index_select(0, A.to(torch.long))
        atom_energy = out[0] if isinstance(out, tuple) else out
        # Broadcast e0 to match out (usually (N,1)).
        if atom_energy.dim() == 2:
            e0 = e0.unsqueeze(1)
        atom_energy = atom_energy + e0.to(dtype=atom_energy.dtype, device=atom_energy.device)
        if isinstance(out, tuple):
            return (atom_energy, *out[1:])
        return atom_energy


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
) -> None:
    from molecular_force_field.interfaces.lammps_mliap import (
        LAMMPS_MLIAP_MFF,
        _maybe_torchscript_trace_model,
        _resolve_model_external_tensor_rank,
    )
    from molecular_force_field.utils.config import ModelConfig

    _ts_supported = (
        "pure-cartesian-sparse",
        "pure-cartesian-sparse-save",
        "pure-cartesian-ictd",
        "pure-cartesian-ictd-save",
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
    external_tensor_rank = _resolve_model_external_tensor_rank(model_eager)

    if mode == "spherical-save-cue" and not native_ops:
        if hasattr(model_eager, "make_torchscript_portable"):
            model_eager.make_torchscript_portable()
            print("[export_core] Replaced product_3/product_5 with pure PyTorch impl (no cuequivariance custom ops)")

    # Optional: embed E0(Z) into per-atom energies before tracing.
    if embed_e0:
        aek = obj.wrapper.atomic_energy_keys.detach().cpu()
        aev = obj.wrapper.atomic_energy_values.detach().cpu()
        lut = _e0_lut_from_keys_values(aek, aev, dtype=actual_dtype, device=torch.device(trace_device))
        model_eager = _E0WrappedModel(model_eager, lut).to(device=torch.device(trace_device))

    # Trace to TorchScript core (edge_vec positional arg) and export its ScriptModule.
    ts_model = _maybe_torchscript_trace_model(
        model_eager,
        device=torch.device(trace_device),
        dtype=actual_dtype,
        enable=True,
        export_reciprocal_source=export_reciprocal_source,
    )
    core = getattr(ts_model, "core", None)
    if core is None or not isinstance(core, torch.jit.ScriptModule):
        raise RuntimeError("Failed to obtain TorchScript core module (trace failed)")

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
        "embed_e0": bool(embed_e0),
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
        ],
        "external_tensor_rank": (
            int(external_tensor_rank) if external_tensor_rank is not None else None
        ),
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
        "notes": [
            "Core model: outputs per-atom energy.",
            "If export_reciprocal_source=true: core tuple includes reciprocal_source as the last tensor.",
            "The reciprocal_source output slot is kept for backward compatibility and may carry a generic runtime long-range source.",
            "If embed_e0=true: output includes E0(Z) constant bias (from preprocessing fit or e0_csv).",
            "Forces: dE/d(pos) computed via autograd on C++ side.",
            "Loadable: C++ torch::jit::load(path).",
            "external_tensor is required at runtime when external_tensor_rank is not null.",
        ],
    }
    meta_path = out_pt + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote metadata: {meta_path}")


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
                        "Supported: pure-cartesian-ictd, pure-cartesian-ictd-save, spherical-save-cue")
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
    )


if __name__ == "__main__":
    main()
