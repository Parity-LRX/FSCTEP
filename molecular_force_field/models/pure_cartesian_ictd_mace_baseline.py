"""Standalone MACE-baseline extracted from PureCartesianICTDFix.

This is the `(ictd_fix_route="baseline", ictd_fix_product_backend="native-mace")`
path of `PureCartesianICTDFix`, pulled out into its own clean model with the
fusion route, the other product backends (ictd / ictd-bridge-u / ictd-pure-u),
the lmax>3 fallback, feature-spectral and long-range branches all stripped out.

What it keeps (identical submodules, so weights transfer 1:1):
  node_embedding
  -> N x ICTDResidualInteractionBlock        (ICTD message passing -- unchanged)
  -> N x NativeMACEProductBasisBlockSO3       (native MACE symmetric contraction)
  -> per-layer EquivariantScalarReadoutSO3 (first N-1) + MACEStyleScalarReadoutSO3 (last)
  -> energy = sum of per-layer readouts.

It is numerically identical to
`PureCartesianICTDFix(ictd_fix_route="baseline", ictd_fix_product_backend="native-mace", ...)`
for the same hyper-parameters and weights (validated in
test/_mace_baseline_equiv.py): same submodule classes, same construction args,
same forward, so `load_state_dict(fix.state_dict(), strict=False)` reproduces it
bit-for-bit.

Purpose: a clean, well-understood reference path to build the long-range
interaction module onto first, before touching the ICTD+fusion architecture.
Long-range is intentionally NOT wired here yet (next step).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from molecular_force_field.models.ictd_irreps import direction_harmonics_all
from molecular_force_field.models.radial_basis import mace_radial_embedding
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    EquivariantScalarReadoutSO3,
)
from molecular_force_field.models.pure_cartesian_ictd_fix import (
    ICTDResidualInteractionBlock,
    MACEStyleScalarReadoutSO3,
    NativeMACEProductBasisBlockSO3,
    _resolve_internal_compute_dtype,
)
from molecular_force_field.models.long_range import build_long_range_module


class PureCartesianICTDMaceBaseline(nn.Module):
    """ICTD-interaction + native-MACE-contraction baseline (no fusion).

    Energy = sum_l layer_readout_l(h_l). Forces / virials come from
    differentiating the energy, exactly like the parent model.
    """

    def __init__(
        self,
        max_embed_radius: float,
        main_number_of_basis: int,
        hidden_dim_conv: int,
        *,
        atomic_numbers: list[int] | tuple[int, ...] | None = None,
        max_atomvalue: int = 10,
        num_interaction: int = 2,
        lmax: int = 2,
        function_type_main: str = "bessel",
        polynomial_cutoff_p: int | None = 6,
        save_contraction_order: int = 3,
        avg_num_neighbors: float | None = None,
        # --- ICTD interaction knobs (native-mace keeps the ICTD interaction) ---
        ictd_save_tp_mode: str = "fully-connected",
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        ictd_tp_backend: str = "pytorch",
        equivariant_post_linear: bool = False,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_fix_interaction_scale: str = "none",
        ictd_fix_interaction_rms_norm: bool = False,
        # --- readout knobs ---
        ictd_fix_readout_head_scale_init: float = 1.0,
        ictd_fix_readout_head_scale_trainable: bool = False,
        ictd_fix_layer_readout_output_init_std: float = 0.003,
        # --- long-range (off by default; additive scalar energy term) ---
        long_range_mode: str = "none",
        long_range_hidden_dim: int = 64,
        long_range_boundary: str = "nonperiodic",
        long_range_neutralize: bool = True,
        long_range_filter_hidden_dim: int = 64,
        long_range_kmax: int = 2,
        long_range_mesh_size: int = 16,
        long_range_slab_padding_factor: int = 2,
        long_range_include_k0: bool = False,
        long_range_source_channels: int = 1,
        long_range_backend: str = "dense_pairwise",
        long_range_reciprocal_backend: str = "direct_kspace",
        long_range_energy_partition: str = "potential",
        long_range_green_mode: str = "poisson",
        long_range_assignment: str = "cic",
        long_range_mesh_fft_full_ewald: bool = False,
        long_range_theta: float = 0.5,
        long_range_leaf_size: int = 32,
        long_range_multipole_order: int = 0,
        device=None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if int(num_interaction) < 2:
            raise ValueError(f"num_interaction must be >= 2, got {num_interaction}")
        if ictd_fix_interaction_scale not in {"none", "mace-rms"}:
            raise ValueError(
                f"ictd_fix_interaction_scale must be 'none' or 'mace-rms', got {ictd_fix_interaction_scale!r}"
            )

        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.num_interaction = int(num_interaction)
        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.ictd_fix_interaction_scale = str(ictd_fix_interaction_scale)
        # compat: trainer / harness may introspect this; baseline has no fusion route.
        self.ictd_fix_route = "baseline"
        self.ictd_fix_interaction_rms_norm = bool(ictd_fix_interaction_rms_norm)
        self.ictd_fix_readout_head_scale_init = float(ictd_fix_readout_head_scale_init)
        self.ictd_fix_readout_head_scale_trainable = bool(ictd_fix_readout_head_scale_trainable)
        self.ictd_fix_layer_readout_output_init_std = float(ictd_fix_layer_readout_output_init_std)
        self.polynomial_cutoff_p = (
            None
            if polynomial_cutoff_p is None or int(polynomial_cutoff_p) <= 0
            else int(polynomial_cutoff_p)
        )
        self.max_atomvalue = int(max_atomvalue)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.edge_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)

        if atomic_numbers is None:
            atomic_numbers = tuple(range(self.max_atomvalue))
        else:
            atomic_numbers = tuple(sorted({int(z) for z in atomic_numbers}))
            if len(atomic_numbers) == 0:
                raise ValueError("atomic_numbers must not be empty")
        self.atomic_numbers = atomic_numbers
        self.num_elements = len(self.atomic_numbers)
        map_size = max(self.max_atomvalue, max(self.atomic_numbers) + 1)
        atomic_number_to_index = torch.full((map_size,), -1, dtype=torch.long)
        for idx, z in enumerate(self.atomic_numbers):
            if z < 0:
                raise ValueError(f"atomic_numbers must be non-negative, got {z}")
            atomic_number_to_index[z] = idx
        self.register_buffer("atomic_number_to_index", atomic_number_to_index, persistent=False)

        self.node_embedding = nn.Linear(self.num_elements, self.channels, bias=False)

        # last layer contracts to scalar (target_lmax=0); earlier layers keep lmax.
        product_target_lmax = [
            self.lmax if layer_idx < self.num_interaction - 1 else 0
            for layer_idx in range(self.num_interaction)
        ]
        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        for layer_idx, target_lmax in enumerate(product_target_lmax):
            input_lmax = 0 if layer_idx == 0 else self.lmax
            sc_lmax = target_lmax  # baseline route: no fusion last-layer sc widening
            message_scale_init = None
            sc_scale_init = None
            if self.ictd_fix_interaction_scale == "mace-rms":
                message_presets = {
                    0: [0.625, 0.561, 0.540, 0.403],
                    1: [0.489, 0.745, 0.741, 0.620],
                }
                preset = message_presets.get(layer_idx, [0.5] * (self.lmax + 1))
                message_scale_init = preset[: self.lmax + 1]
                if sc_lmax == 0 and layer_idx > 0:
                    sc_scale_init = [0.342]
                elif sc_lmax > 0 and layer_idx > 0:
                    sc_scale_init = [0.342] + [0.5] * sc_lmax
            self.interactions.append(
                ICTDResidualInteractionBlock(
                    channels=self.channels,
                    lmax=self.lmax,
                    input_lmax=input_lmax,
                    target_lmax=self.lmax,
                    sc_lmax=sc_lmax,
                    number_of_basis=self.number_of_basis,
                    num_elements=self.num_elements,
                    function_type=self.function_type,
                    ictd_save_tp_mode=ictd_save_tp_mode,
                    ictd_tp_path_policy=ictd_tp_path_policy,
                    ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                    ictd_tp_backend=ictd_tp_backend,
                    equivariant_post_linear=equivariant_post_linear,
                    use_self_connection=(layer_idx > 0),
                    avg_num_neighbors=self.avg_num_neighbors,
                    message_scale_init=message_scale_init,
                    sc_scale_init=sc_scale_init,
                    use_rms_norm=self.ictd_fix_interaction_rms_norm,
                )
            )
            self.products.append(
                NativeMACEProductBasisBlockSO3(
                    num_elements=self.num_elements,
                    channels=self.channels,
                    lmax=self.lmax,
                    target_lmax=target_lmax,
                    correlation=save_contraction_order,
                )
            )

        self.layer_energy_readouts = nn.ModuleList(
            [
                EquivariantScalarReadoutSO3(
                    self.channels, self.lmax, output_init_std=self.ictd_fix_layer_readout_output_init_std
                )
                for _ in range(self.num_interaction - 1)
            ]
        )
        self.last_layer_energy_readout = MACEStyleScalarReadoutSO3(
            self.channels,
            hidden_channels=16,
            output_init_std=self.ictd_fix_layer_readout_output_init_std,
        )
        if self.ictd_fix_readout_head_scale_trainable:
            self.readout_head_scales = nn.Parameter(
                torch.full((2,), self.ictd_fix_readout_head_scale_init, dtype=torch.get_default_dtype())
            )
        else:
            self.readout_head_scales = None

        # --- long-range interaction module (None when mode=="none") ---
        # Fed the final per-atom SCALAR descriptor (layer_states[-1], invariant) ->
        # latent charge -> additive [N,1] long-range energy. energy_scale inits to 0,
        # so even when enabled the term starts at exactly zero (no init perturbation).
        self.long_range_mode = str(long_range_mode)
        self.long_range_module = build_long_range_module(
            mode=self.long_range_mode,
            feature_dim=self.channels,
            hidden_dim=long_range_hidden_dim,
            boundary=long_range_boundary,
            neutralize=long_range_neutralize,
            filter_hidden_dim=long_range_filter_hidden_dim,
            kmax=long_range_kmax,
            mesh_size=long_range_mesh_size,
            slab_padding_factor=long_range_slab_padding_factor,
            include_k0=long_range_include_k0,
            source_channels=long_range_source_channels,
            backend=long_range_backend,
            reciprocal_backend=long_range_reciprocal_backend,
            energy_partition=long_range_energy_partition,
            green_mode=long_range_green_mode,
            assignment=long_range_assignment,
            mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
            theta=long_range_theta,
            leaf_size=long_range_leaf_size,
            multipole_order=long_range_multipole_order,
            cutoff_radius=self.max_radius,
        )
        self.long_range_exports_reciprocal_source = (
            bool(getattr(self.long_range_module, "exports_reciprocal_source", False))
            if self.long_range_module is not None
            else False
        )

    def _readout_head_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.readout_head_scales is None:
            # new_zeros(()) keeps this CUDA-graph capturable; equals the scalar.
            return ref.new_zeros(()) + float(self.ictd_fix_readout_head_scale_init)
        return self.readout_head_scales[index].to(dtype=ref.dtype, device=ref.device)

    def forward(
        self,
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        *,
        precomputed_edge_vec=None,
        return_combined_features: bool = False,
        sync_after_scatter: "callable | None" = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        if return_physical_tensors:
            raise ValueError("pure-cartesian-ictd-mace-baseline does not support return_physical_tensors=True")

        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec[sort_idx]
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs

        edge_length = edge_vec.norm(dim=1)
        n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        edge_mask = (edge_length <= self.max_radius).to(dtype=pos.dtype).unsqueeze(-1)
        Y_list = direction_harmonics_all(n.to(dtype=dtype), self.lmax)
        edge_attrs = {l: Y_list[l].to(dtype=dtype).unsqueeze(-2) for l in range(self.lmax + 1)}
        edge_feats = mace_radial_embedding(
            edge_length,
            r_max=self.max_radius,
            number_of_basis=self.number_of_basis,
            function_type=self.function_type,
            polynomial_cutoff_p=self.polynomial_cutoff_p,
        ).to(dtype=dtype)

        A_long = A.long()
        if not getattr(self, "skip_input_validation", False):
            if int(A_long.max().item()) >= self.atomic_number_to_index.numel():
                raise ValueError(
                    f"Encountered atomic number {int(A_long.max().item())}, but compact mapping supports only up to "
                    f"{self.atomic_number_to_index.numel() - 1}. atomic_numbers={self.atomic_numbers}"
                )
        compact_idx = self.atomic_number_to_index[A_long]
        if not getattr(self, "skip_input_validation", False):
            if torch.any(compact_idx < 0):
                bad = torch.unique(A_long[compact_idx < 0]).tolist()
                raise ValueError(
                    f"Encountered atomic numbers without compact mapping: {bad}. "
                    f"Configured atomic_numbers={self.atomic_numbers}"
                )
        node_attrs = F.one_hot(compact_idx, num_classes=self.num_elements).to(dtype=dtype)
        h = self.node_embedding(node_attrs)

        layer_states: List[torch.Tensor] = []
        total_energy = None
        for layer_idx, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            message, sc = interaction(
                node_attrs=node_attrs,
                node_feats=h,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                edge_mask=edge_mask,
                sync_after_scatter=sync_after_scatter,
            )
            h = product(node_feats=message, sc=sc, node_attrs=node_attrs)
            layer_states.append(h)
            if layer_idx < self.num_interaction - 1:
                e_layer = self.layer_energy_readouts[layer_idx](h)
                e_layer = self._readout_head_scale(0, e_layer) * e_layer
            else:
                e_layer = self.last_layer_energy_readout(h)
                e_layer = self._readout_head_scale(1, e_layer) * e_layer
            total_energy = e_layer if total_energy is None else (total_energy + e_layer)

        combined_features = torch.cat(layer_states, dim=-1)
        out = total_energy.sum(dim=-1, keepdim=True)

        # --- long-range additive term (skipped entirely when module is None) ---
        reciprocal_source = None
        if self.long_range_module is not None:
            lr_feat = layer_states[-1].reshape(layer_states[-1].shape[0], -1)  # final scalar descriptor [N, channels]
            defer = False
            if return_reciprocal_source and self.long_range_exports_reciprocal_source:
                long_range_energy, reciprocal_source = self.long_range_module(
                    lr_feat, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst, return_source=True
                )
                defer = reciprocal_source.numel() > 0
            else:
                long_range_energy = self.long_range_module(
                    lr_feat, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst
                )
            if long_range_energy is not None and not defer:
                out = out + long_range_energy

        if return_combined_features:
            if return_reciprocal_source:
                rs = reciprocal_source if reciprocal_source is not None else out.new_empty((out.size(0), 0))
                return out, combined_features, rs
            return out, combined_features
        if return_reciprocal_source:
            rs = reciprocal_source if reciprocal_source is not None else out.new_empty((out.size(0), 0))
            return out, rs
        return out
