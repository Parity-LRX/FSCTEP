"""
No-cross-layer ablation for pure-cartesian ICTD save-multiple.

This variant keeps the same contraction depth, fusion, product5 readout, and
layerwise 0e auxiliary heads as the main model, but removes the explicit
cross-layer input from the second contraction branch:

    full:    g_mix = C_mix([f1, ..., f_{N-1}, g_last])
    ablation g_mix = C_mix([f_last, g_last])

The goal is to isolate whether cross-layer fusion itself contributes beyond
simply stacking two contraction blocks.
"""

from __future__ import annotations

import torch
from e3nn.math import soft_one_hot_linspace

from molecular_force_field.utils.scatter import scatter
from molecular_force_field.models.long_range import apply_long_range_modules
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer,
    _split_irreps,
    _merge_irreps,
    apply_channel_adapter_per_l,
    direction_harmonics_all,
)


class PureCartesianICTDTransformerLayerNoCrossFusion(PureCartesianICTDTransformerLayer):
    """Ablation model that disables cross-layer inputs to the second contraction."""

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
        sync_after_scatter: callable | None = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        if return_physical_tensors:
            raise ValueError("pure-cartesian-ictd-save does not currently support return_physical_tensors=True")
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

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

        Y_list = direction_harmonics_all(n.to(dtype=next(self.parameters()).dtype), self.lmax)
        f1 = self.e3_conv_emb(
            pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
            precomputed_n=n,
            precomputed_edge_length=edge_length,
            precomputed_Y_list=Y_list,
        )
        f1 = self.post_conv_linears[0](f1)
        if sync_after_scatter is not None:
            f1 = sync_after_scatter(f1)
        features = [f1]

        num_nodes = pos.size(0)
        emb_base = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True,
        ).mul(self.number_of_basis ** 0.5)
        neighbor_count = scatter(
            torch.ones_like(edge_dst, dtype=f1.dtype),
            edge_dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ).clamp(min=1.0)
        for layer_idx, (tp2, fc2) in enumerate(zip(self.tp2_layers, self.fc2_layers), start=1):
            f_prev = features[-1]
            emb = emb_base.to(dtype=f_prev.dtype)
            gates = fc2(emb)

            Y = {l: Y_list[l].to(dtype=f_prev.dtype).unsqueeze(-2) for l in range(self.lmax + 1)}
            x1 = _split_irreps(f_prev, self.channels, self.lmax)
            x1e = {l: x1[l][edge_src] for l in range(self.lmax + 1)}
            edge_blocks = tp2(x1e, Y, gates)
            edge_flat = _merge_irreps(edge_blocks, self.channels, self.lmax)
            edge_flat = edge_flat * edge_mask.to(dtype=edge_flat.dtype)
            f_next = scatter(edge_flat, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
            f_next = f_next / neighbor_count.to(dtype=edge_flat.dtype).sqrt().unsqueeze(-1)
            f_next = self.post_conv_linears[layer_idx](f_next)
            if sync_after_scatter is not None:
                f_next = sync_after_scatter(f_next)
            features.append(f_next)

        f_combine = torch.cat(features, dim=-1)

        if self.save_readout_mode == "elementwise-scalar":
            xb = _split_irreps(f_combine, self.channels * self.num_interaction, self.lmax)
            scalars = self.product_3(xb, xb)
            combined_features = f_combine
        else:
            g_last = self.g_last_gate * self.multiple_contraction_last(features[-1])
            # Ablation: remove explicit shallow-feature input from the second contraction.
            mix_inputs = torch.cat([features[-1], g_last], dim=-1)
            g_mix = self.multiple_contraction_mix(mix_inputs)
            fused_input = torch.cat([g_last, g_mix], dim=-1)
            f_contract = self.multiple_contract_fuse(fused_input)
            contract_blocks = _split_irreps(f_contract, self.channels, self.lmax)
            combined_features = torch.cat(features + [f_contract], dim=-1)

        splits = [_split_irreps(f, self.channels, self.lmax) for f in features]
        T_blocks: dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            parts = []
            for i in range(len(features)):
                b_l = apply_channel_adapter_per_l(splits[i][l], self._p5_adapt[i][str(l)])
                parts.append(b_l)
            T_blocks[l] = torch.cat(parts, dim=-2)
        if self.save_readout_mode == "elementwise-scalar":
            T_blocks[0] = torch.cat([T_blocks[0], scalars.unsqueeze(-1)], dim=-2)
        else:
            for l in range(self.lmax + 1):
                c_l = apply_channel_adapter_per_l(contract_blocks[l], self._p5_contract_adapt[str(l)])
                T_blocks[l] = torch.cat([T_blocks[l], c_l], dim=-2)
        f_prod5, long_range_energy, reciprocal_source, defer_long_range_to_runtime = apply_long_range_modules(
            self,
            self.product_5(T_blocks, T_blocks),
            pos,
            batch,
            cell,
            edge_src=edge_src,
            edge_dst=edge_dst,
            return_reciprocal_source=return_reciprocal_source,
        )

        e_out = self.proj_total(f_prod5)
        aux_energy = None
        for idx, (readout, feat) in enumerate(zip(self.layer_energy_readouts, features)):
            e_layer = self.layer_energy_gates[idx] * readout(feat)
            aux_energy = e_layer if aux_energy is None else (aux_energy + e_layer)
        if aux_energy is not None:
            e_out = e_out + aux_energy
        out = e_out.sum(dim=-1, keepdim=True)
        if long_range_energy is not None and not defer_long_range_to_runtime:
            out = out + long_range_energy
        if return_combined_features:
            if return_reciprocal_source:
                if reciprocal_source is None:
                    reciprocal_source = out.new_empty((out.size(0), 0))
                return out, combined_features, reciprocal_source
            return out, combined_features
        if return_reciprocal_source:
            if reciprocal_source is None:
                reciprocal_source = out.new_empty((out.size(0), 0))
            return out, reciprocal_source
        return out
