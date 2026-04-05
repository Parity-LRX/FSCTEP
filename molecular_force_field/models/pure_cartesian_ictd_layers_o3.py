"""
O(3) parity-aware ICTD transformer.

This keeps the existing ICTD polynomial / harmonic basis, but promotes internal
features from l-only blocks to (l, parity) blocks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.math import soft_one_hot_linspace

from molecular_force_field.models.long_range import build_feature_spectral_module, build_long_range_module
from molecular_force_field.models.ictd_irreps import (
    HarmonicElementwiseProductO3,
    HarmonicFullyConnectedTensorProductO3,
    canonical_irrep_parity_sign,
    parity_letter_to_sign,
    parity_sign_to_letter,
    parse_irreps_string,
    direction_harmonics_all,
)
from molecular_force_field.models.mlp import MainNet, RobustScalarWeightedSum
from molecular_force_field.models.pure_cartesian_ictd_layers import PhysicalTensorICTDEmbedding
from molecular_force_field.utils.external_tensor_specs import (
    external_tensor_total_numel,
    normalize_external_tensor_specs,
    unpack_external_tensor,
)
from molecular_force_field.utils.fidelity import (
    apply_delta_energy_heads,
    apply_fidelity_embedding,
    zero_init_module_output,
)


def _irrep_key(l: int, parity: int) -> tuple[int, int]:
    return (int(l), 1 if int(parity) >= 0 else -1)


def _irrep_key_str(l: int, parity: int) -> str:
    return f"{int(l)}{parity_sign_to_letter(parity)}"


def _o3_active_irreps(lmax: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for l in range(int(lmax) + 1):
        out.append((l, 1))
        out.append((l, -1))
    return out


def _canonical_o3_irreps(lmax: int) -> list[tuple[int, int]]:
    return [(int(l), canonical_irrep_parity_sign(int(l))) for l in range(int(lmax) + 1)]


def _sorted_unique_irreps(keys: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({_irrep_key(l, p) for l, p in keys}, key=lambda item: (item[0], 0 if item[1] > 0 else 1))


def _parse_o3_active_irreps_spec(spec: str | Iterable[str] | None, *, lmax: int) -> list[tuple[int, int]] | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return None
        parsed = parse_irreps_string(text.replace(",", " + "))
    else:
        entries = [str(x).strip() for x in spec if str(x).strip()]
        if not entries:
            return None
        parsed = parse_irreps_string(" + ".join(entries))
    out: list[tuple[int, int]] = []
    for mul, l_val, parity in parsed:
        if mul != 1:
            raise ValueError(f"o3_active_irreps only supports multiplicity-1 entries, got {spec!r}")
        if not (0 <= int(l_val) <= int(lmax)):
            raise ValueError(f"o3_active_irreps entry {l_val}{parity_sign_to_letter(parity)} outside 0..{lmax}")
        out.append((int(l_val), int(parity)))
    return _sorted_unique_irreps(out)


def _required_o3_irreps(
    *,
    lmax: int,
    external_tensor_irrep: str | None,
    external_tensor_specs: list[dict] | None,
    physical_tensor_outputs: dict[str, dict] | None,
) -> list[tuple[int, int]]:
    required: list[tuple[int, int]] = [(0, 1)]
    normalized_specs = normalize_external_tensor_specs(
        external_tensor_specs,
        external_tensor_rank=None,
        external_tensor_irrep=None,
        external_tensor_parity=None,
    )
    if normalized_specs is not None:
        for spec in normalized_specs:
            for mul, l_val, parity in parse_irreps_string(str(spec["irrep"])):
                if mul != 1:
                    raise ValueError(f"external tensor spec must describe a single irrep, got {spec!r}")
                if int(l_val) <= int(lmax):
                    required.append((int(l_val), int(parity)))
    elif external_tensor_irrep is not None:
        for mul, l_val, parity in parse_irreps_string(str(external_tensor_irrep)):
            if mul != 1:
                raise ValueError(f"external_tensor_irrep must describe a single irrep, got {external_tensor_irrep!r}")
            if int(l_val) <= int(lmax):
                required.append((int(l_val), int(parity)))
    if physical_tensor_outputs:
        for spec_in in physical_tensor_outputs.values():
            irreps = spec_in.get("irreps")
            if irreps is not None:
                parsed = parse_irreps_string(" + ".join(str(x) for x in irreps) if isinstance(irreps, (list, tuple)) else str(irreps))
                for mul, l_val, parity in parsed:
                    if mul != 1:
                        raise ValueError(f"physical tensor irreps must use multiplicity 1, got {irreps!r}")
                    if int(l_val) <= int(lmax):
                        required.append((int(l_val), int(parity)))
                continue
            for l_val in spec_in.get("ls", []):
                required.append((int(l_val), canonical_irrep_parity_sign(int(l_val))))
    return _sorted_unique_irreps(required)


def _balanced_o3_irreps(
    *,
    lmax: int,
    seed_irreps: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = set(seed_irreps)
    geom = _canonical_o3_irreps(lmax)
    for l1, p1 in seed_irreps:
        for l2, p2 in geom:
            p3 = int(p1) * int(p2)
            l_lo = abs(int(l1) - int(l2))
            l_hi = min(int(l1) + int(l2), int(lmax))
            for l3 in range(l_lo, l_hi + 1):
                out.add((l3, p3))
    return _sorted_unique_irreps(out)


def resolve_o3_active_irreps(
    *,
    lmax: int,
    preset: str = "auto",
    explicit_irreps: str | Iterable[str] | None = None,
    external_tensor_irrep: str | None = None,
    external_tensor_specs: list[dict] | None = None,
    physical_tensor_outputs: dict[str, dict] | None = None,
) -> list[tuple[int, int]]:
    required = _required_o3_irreps(
        lmax=lmax,
        external_tensor_irrep=external_tensor_irrep,
        external_tensor_specs=external_tensor_specs,
        physical_tensor_outputs=physical_tensor_outputs,
    )
    explicit = _parse_o3_active_irreps_spec(explicit_irreps, lmax=lmax)
    if explicit is not None:
        return _sorted_unique_irreps(list(explicit) + required)

    preset_norm = str(preset or "auto").strip().lower()
    if preset_norm == "full":
        return _o3_active_irreps(lmax)
    if preset_norm == "minimal":
        return required
    if preset_norm == "auto":
        return _sorted_unique_irreps(_canonical_o3_irreps(lmax) + required)
    if preset_norm == "balanced":
        return _balanced_o3_irreps(
            lmax=lmax,
            seed_irreps=_sorted_unique_irreps(_canonical_o3_irreps(lmax) + required),
        )
    raise ValueError(f"Unknown o3_irrep_preset={preset!r}; expected auto|minimal|balanced|full")


def _irreps_total_dim_o3(channels: int, active_irreps: list[tuple[int, int]]) -> int:
    return int(channels) * sum(2 * int(l) + 1 for l, _ in active_irreps)


def _split_irreps_o3(
    x: torch.Tensor,
    channels: int,
    active_irreps: list[tuple[int, int]],
) -> dict[tuple[int, int], torch.Tensor]:
    out: dict[tuple[int, int], torch.Tensor] = {}
    idx = 0
    for l, p in active_irreps:
        d = channels * (2 * l + 1)
        blk = x[..., idx : idx + d]
        idx += d
        out[(l, p)] = blk.view(*x.shape[:-1], channels, 2 * l + 1)
    return out


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


def _merge_irreps_o3(
    blocks: dict[tuple[int, int], torch.Tensor],
    channels: int,
    active_irreps: list[tuple[int, int]],
) -> torch.Tensor:
    parts = []
    for key in active_irreps:
        blk = blocks[key]
        parts.append(blk.reshape(*blk.shape[:-2], channels * blk.shape[-1]))
    return torch.cat(parts, dim=-1)


def _apply_channel_adapter_per_irrep(x_lp: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    if isinstance(adapter, nn.Identity):
        return x_lp
    y = adapter(x_lp.movedim(-2, -1))
    return y.movedim(-1, -2)


def _normalize_external_tensor_irrep(
    *,
    rank: int | None,
    irrep: str | None,
    parity: str | int | None,
) -> str | None:
    if rank is None:
        if irrep is not None or parity is not None:
            raise ValueError("external_tensor_irrep/parity requires external_tensor_rank to be set")
        return None
    if irrep is not None:
        parts = parse_irreps_string(str(irrep))
        if len(parts) != 1 or parts[0][0] != 1:
            raise ValueError(f"external_tensor_irrep must describe exactly one irrep, got {irrep!r}")
        _, l_val, p = parts[0]
        if l_val != int(rank):
            raise ValueError(f"external_tensor_irrep {irrep!r} does not match external_tensor_rank={rank}")
        return _irrep_key_str(l_val, p)
    if parity is None:
        if int(rank) == 1:
            return "1o"
        return _irrep_key_str(int(rank), canonical_irrep_parity_sign(int(rank)))
    p = parity_letter_to_sign(str(parity)) if isinstance(parity, str) else int(parity)
    return _irrep_key_str(int(rank), p)


def _normalize_physical_tensor_spec(name: str, spec_in: dict, *, lmax: int) -> dict:
    spec = dict(spec_in)
    irreps = spec.get("irreps", None)
    ls = spec.get("ls", None)
    parity_by_l: dict[int, int] = {}
    if irreps is not None:
        parsed = parse_irreps_string(" + ".join(str(x) for x in irreps) if isinstance(irreps, (list, tuple)) else str(irreps))
        ls = []
        for mul, l_val, parity in parsed:
            if mul != 1:
                raise ValueError(f"physical_tensor_outputs[{name!r}] irreps must use multiplicity 1 entries, got {irreps!r}")
            if l_val not in ls:
                ls.append(l_val)
            parity_by_l[l_val] = parity
    if ls is None:
        raise ValueError(f"physical_tensor_outputs[{name!r}] missing 'ls' or 'irreps'")
    ls = [int(l) for l in ls]
    for l in ls:
        if not (0 <= l <= lmax):
            raise ValueError(f"physical_tensor_outputs[{name!r}] l={l} out of range 0..lmax={lmax}")
        parity_by_l.setdefault(l, canonical_irrep_parity_sign(l))
    if len(ls) != len(set(ls)):
        raise ValueError(f"physical_tensor_outputs[{name!r}] duplicate l values are not allowed")
    reduce = str(spec.get("reduce", "sum")).strip().lower()
    if reduce not in ("sum", "mean", "none"):
        raise ValueError(f"physical_tensor_outputs[{name!r}] reduce must be 'sum'|'mean'|'none', got {reduce!r}")
    channels_out = spec.get("channels_out", 1)
    if isinstance(channels_out, int):
        ch_out_by_l = {l: int(channels_out) for l in ls}
    elif isinstance(channels_out, dict):
        ch_out_by_l = {int(k): int(v) for k, v in channels_out.items()}
        for l in ls:
            if l not in ch_out_by_l:
                raise ValueError(f"physical_tensor_outputs[{name!r}] channels_out missing l={l}")
    else:
        raise ValueError(f"physical_tensor_outputs[{name!r}] channels_out must be int or dict, got {type(channels_out)}")
    for l in ls:
        if ch_out_by_l[l] <= 0:
            raise ValueError(f"physical_tensor_outputs[{name!r}] channels_out[{l}] must be positive")
    spec["ls"] = ls
    spec["channels_out"] = ch_out_by_l
    spec["reduce"] = reduce
    spec["irreps"] = [_irrep_key_str(l, parity_by_l[l]) for l in ls]
    spec["parity_by_l"] = {int(l): int(parity_by_l[l]) for l in ls}
    return spec


class ICTDO3E3Conv(nn.Module):
    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_out: int,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        lmax: int = 2,
        function_type: str = "gaussian",
        path_policy: str = "full",
        max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "auto",
        external_tensor_rank: int | None = None,
        external_tensor_irrep: str | None = None,
        external_tensor_specs: list[dict] | None = None,
        active_irreps: list[tuple[int, int]] | None = None,
        external_tensor_input_repr: str = "cartesian",
        external_tensor_channels_in: int = 1,
        external_tensor_include_trace_chain: bool = True,
        num_fidelity_levels: int = 0,
        multi_fidelity_mode: str = "conditioning",
        external_tensor_scale_init: float = 0.0,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = int(channels_out)
        self.output_size = int(output_size)
        self.lmax = int(lmax)
        self.function_type = function_type
        self.active_irreps = _sorted_unique_irreps(active_irreps or _o3_active_irreps(self.lmax))
        self.active_irrep_set = set(self.active_irreps)

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )
        self.num_fidelity_levels = int(num_fidelity_levels)
        self.fidelity_embedding = (
            nn.Embedding(self.num_fidelity_levels, output_size) if self.num_fidelity_levels > 0 else None
        )
        self.tp2 = HarmonicFullyConnectedTensorProductO3(
            mul_in1=output_size,
            mul_in2=output_size,
            mul_out=channels_out,
            lmax=self.lmax,
            active_irreps=self.active_irreps,
            internal_weights=True,
            path_policy=path_policy,
            max_rank_other=max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
        )
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.num_paths),
        )

        self.external_tensor_specs = normalize_external_tensor_specs(
            external_tensor_specs,
            external_tensor_rank=external_tensor_rank,
            external_tensor_irrep=external_tensor_irrep,
            external_tensor_parity=None,
        )
        self.external_tensor_total_numel = external_tensor_total_numel(self.external_tensor_specs)
        single_external_spec = self.external_tensor_specs[0] if self.external_tensor_specs and len(self.external_tensor_specs) == 1 else None
        self.external_tensor_rank = int(single_external_spec["rank"]) if single_external_spec is not None else None
        self.external_tensor_irrep = str(single_external_spec["irrep"]) if single_external_spec is not None else None
        if self.external_tensor_specs is not None and len(self.external_tensor_specs) > 1:
            self.external_tensor_embed = None
            self.external_tensor_scale_by_irrep = None
            self.external_tensor_embeds = nn.ModuleDict(
                {
                    str(spec["name"]): PhysicalTensorICTDEmbedding(
                        rank=int(spec["rank"]),
                        lmax_out=self.lmax,
                        channels_in=int(external_tensor_channels_in),
                        channels_out=1,
                        input_repr=str(external_tensor_input_repr),
                        include_trace_chain=bool(external_tensor_include_trace_chain),
                        rank2_mode="full" if int(spec["rank"]) == 2 else "symmetric",
                        internal_compute_dtype=internal_compute_dtype,
                    )
                    for spec in self.external_tensor_specs
                }
            )
            self.external_tensor_scale_by_spec = nn.Parameter(
                torch.full((len(self.external_tensor_specs),), float(external_tensor_scale_init))
            )
        elif self.external_tensor_rank is not None:
            self.external_tensor_embed = PhysicalTensorICTDEmbedding(
                rank=self.external_tensor_rank,
                lmax_out=self.lmax,
                channels_in=int(external_tensor_channels_in),
                channels_out=1,
                input_repr=str(external_tensor_input_repr),
                include_trace_chain=bool(external_tensor_include_trace_chain),
                rank2_mode="full" if self.external_tensor_rank == 2 else "symmetric",
                internal_compute_dtype=internal_compute_dtype,
            )
            self.external_tensor_scale_by_irrep = nn.Parameter(
                torch.full((len(self.active_irreps),), float(external_tensor_scale_init))
            )
            self.external_tensor_embeds = None
            self.external_tensor_scale_by_spec = None
        else:
            self.external_tensor_embed = None
            self.external_tensor_scale_by_irrep = None
            self.external_tensor_embeds = None
            self.external_tensor_scale_by_spec = None
        self.output_dim = _irreps_total_dim_o3(self.channels_out, self.active_irreps)

    def _empty_blocks(self, batch_shape: tuple[int, ...], mul: int, device: torch.device, dtype: torch.dtype) -> dict[tuple[int, int], torch.Tensor]:
        out = {}
        for l, p in self.active_irreps:
            out[(l, p)] = torch.zeros(*batch_shape, mul, 2 * l + 1, device=device, dtype=dtype)
        return out

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
        precomputed_n=None,
        precomputed_edge_length=None,
        precomputed_Y_list=None,
        external_tensor: torch.Tensor | None = None,
        fidelity_ids: torch.Tensor | None = None,
    ):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        if precomputed_n is None or precomputed_edge_length is None:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
            edge_length = edge_vec.norm(dim=1)
            n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        else:
            n = precomputed_n
            edge_length = precomputed_edge_length

        Ai = self.atom_mlp(self.atom_embedding(A.long()))
        Ai = apply_fidelity_embedding(Ai, batch, fidelity_ids, self.fidelity_embedding)
        n = n.to(dtype=Ai.dtype)
        edge_length = edge_length.to(dtype=Ai.dtype)

        if precomputed_Y_list is None:
            Y_list = direction_harmonics_all(n, self.lmax)
        else:
            Y_list = precomputed_Y_list

        f_in = self._empty_blocks((edge_src.shape[0],), self.output_size, Ai.device, Ai.dtype)
        for l in range(self.lmax + 1):
            p = canonical_irrep_parity_sign(l)
            key = (l, p)
            if key in self.active_irrep_set:
                f_in[key] = Ai[edge_src].unsqueeze(-1) * Y_list[l].unsqueeze(-2)

        if external_tensor is not None:
            if self.external_tensor_specs is not None and len(self.external_tensor_specs) > 1:
                unpacked = unpack_external_tensor(external_tensor, self.external_tensor_specs)
                for spec_idx, spec in enumerate(self.external_tensor_specs):
                    blocks_i = self.external_tensor_embeds[str(spec["name"])](unpacked[str(spec["name"])], return_blocks=True)
                    _, ext_l, ext_p = parse_irreps_string(str(spec["irrep"]))[0]
                    key = (int(ext_l), int(ext_p))
                    t = blocks_i.get(int(ext_l))
                    if t is None:
                        continue
                    num_nodes = Ai.shape[0]
                    if t.dim() == 2:
                        t = t.unsqueeze(0).expand(num_nodes, *t.shape)
                    elif t.dim() == 3:
                        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
                        if num_graphs > 0 and t.shape[0] == num_graphs:
                            t = t[batch]
                        elif t.shape[0] == 1 and num_nodes > 1:
                            t = t.expand(num_nodes, *t.shape[1:])
                    t = t[edge_src].to(device=Ai.device, dtype=Ai.dtype)
                    if t.shape[-2] == 1:
                        t = t * Ai[edge_src].unsqueeze(-1)
                    scale = self.external_tensor_scale_by_spec[spec_idx]
                    f_in[key] = f_in[key] + t * scale
            elif self.external_tensor_embed is None or self.external_tensor_irrep is None:
                raise ValueError("external_tensor was provided but ICTDO3E3Conv was not configured with external tensor support")
            else:
                ext_blocks = self.external_tensor_embed(external_tensor, return_blocks=True)
                ext_l = self.external_tensor_rank
                ext_p = parity_letter_to_sign(self.external_tensor_irrep[-1])
                key = (int(ext_l), int(ext_p))
                t = ext_blocks.get(int(ext_l))
                if t is not None:
                    num_nodes = Ai.shape[0]
                    if t.dim() == 2:
                        t = t.unsqueeze(0).expand(num_nodes, *t.shape)
                    elif t.dim() == 3:
                        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
                        if num_graphs > 0 and t.shape[0] == num_graphs:
                            t = t[batch]
                        elif t.shape[0] == 1 and num_nodes > 1:
                            t = t.expand(num_nodes, *t.shape[1:])
                    t = t[edge_src].to(device=Ai.device, dtype=Ai.dtype)
                    if t.shape[-2] == 1:
                        t = t * Ai[edge_src].unsqueeze(-1)
                    scale = self.external_tensor_scale_by_irrep[self.active_irreps.index(key)]
                    f_in[key] = f_in[key] + t * scale

        x2 = self._empty_blocks((edge_src.shape[0],), self.output_size, Ai.device, Ai.dtype)
        x2[(0, 1)] = Ai[edge_dst].unsqueeze(-1)

        emb = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True)
        emb = emb.mul(self.number_of_basis ** 0.5).to(dtype=Ai.dtype)
        gates = self.fc(emb)
        out_blocks = self.tp2(f_in, x2, gates)
        edge_features = _merge_irreps_o3(out_blocks, self.channels_out, self.active_irreps)
        num_nodes = pos.size(0)
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1).to(edge_features.dtype)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
        return out


class PureCartesianICTDO3TransformerLayer(nn.Module):
    def __init__(
        self,
        max_embed_radius: float,
        main_max_radius: float,
        main_number_of_basis: int,
        hidden_dim_conv: int,
        hidden_dim_sh: int,
        hidden_dim: int,
        channel_in2: int = 32,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "auto",
        invariant_channels: int = 32,
        physical_tensor_outputs: dict[str, dict] | None = None,
        external_tensor_rank: int | None = None,
        external_tensor_irrep: str | None = None,
        external_tensor_parity: str | int | None = None,
        external_tensor_specs: list[dict] | None = None,
        o3_irrep_preset: str = "auto",
        o3_active_irreps: str | list[str] | tuple[str, ...] | None = None,
        external_tensor_input_repr: str = "cartesian",
        external_tensor_channels_in: int = 1,
        external_tensor_include_trace_chain: bool = True,
        num_fidelity_levels: int = 0,
        multi_fidelity_mode: str = "conditioning",
        external_tensor_scale_init: float = 0.0,
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
        long_range_far_source_dim: int = 16,
        long_range_far_num_shells: int = 3,
        long_range_far_shell_growth: float = 2.0,
        long_range_far_tail: bool = True,
        long_range_far_tail_bins: int = 2,
        long_range_far_stats: str = "mean,count,mean_r,rms_r",
        long_range_far_max_radius_multiplier: float | None = None,
        long_range_far_source_norm: bool = True,
        long_range_far_gate_init: float = 0.0,
        feature_spectral_mode: str = "none",
        feature_spectral_bottleneck_dim: int = 8,
        feature_spectral_mesh_size: int = 16,
        feature_spectral_filter_hidden_dim: int = 64,
        feature_spectral_boundary: str = "periodic",
        feature_spectral_slab_padding_factor: int = 2,
        feature_spectral_neutralize: bool = True,
        feature_spectral_include_k0: bool = False,
        feature_spectral_assignment: str = "cic",
        feature_spectral_gate_init: float = 0.0,
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.lmax = int(lmax)
        self.channels = int(hidden_dim_conv)
        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")
        self.long_range_mode = str(long_range_mode)
        self.long_range_hidden_dim = int(long_range_hidden_dim)
        self.long_range_boundary = str(long_range_boundary)
        self.long_range_neutralize = bool(long_range_neutralize)
        self.long_range_filter_hidden_dim = int(long_range_filter_hidden_dim)
        self.long_range_kmax = int(long_range_kmax)
        self.long_range_mesh_size = int(long_range_mesh_size)
        self.long_range_slab_padding_factor = int(long_range_slab_padding_factor)
        self.long_range_include_k0 = bool(long_range_include_k0)
        self.long_range_source_channels = int(long_range_source_channels)
        self.long_range_backend = str(long_range_backend)
        self.long_range_reciprocal_backend = str(long_range_reciprocal_backend)
        self.long_range_energy_partition = str(long_range_energy_partition)
        self.long_range_green_mode = str(long_range_green_mode)
        self.long_range_assignment = str(long_range_assignment)
        self.long_range_mesh_fft_full_ewald = bool(long_range_mesh_fft_full_ewald)
        self.long_range_theta = float(long_range_theta)
        self.long_range_leaf_size = int(long_range_leaf_size)
        self.long_range_multipole_order = int(long_range_multipole_order)
        self.long_range_far_source_dim = int(long_range_far_source_dim)
        self.long_range_far_num_shells = int(long_range_far_num_shells)
        self.long_range_far_shell_growth = float(long_range_far_shell_growth)
        self.long_range_far_tail = bool(long_range_far_tail)
        self.long_range_far_tail_bins = int(long_range_far_tail_bins)
        self.long_range_far_stats = str(long_range_far_stats)
        self.long_range_far_max_radius_multiplier = (
            float(long_range_far_max_radius_multiplier) if long_range_far_max_radius_multiplier is not None else None
        )
        self.long_range_far_source_norm = bool(long_range_far_source_norm)
        self.long_range_far_gate_init = float(long_range_far_gate_init)
        self.feature_spectral_mode = str(feature_spectral_mode)
        self.feature_spectral_bottleneck_dim = int(feature_spectral_bottleneck_dim)
        self.feature_spectral_mesh_size = int(feature_spectral_mesh_size)
        self.feature_spectral_filter_hidden_dim = int(feature_spectral_filter_hidden_dim)
        self.feature_spectral_boundary = str(feature_spectral_boundary)
        self.feature_spectral_slab_padding_factor = int(feature_spectral_slab_padding_factor)
        self.feature_spectral_neutralize = bool(feature_spectral_neutralize)
        self.feature_spectral_include_k0 = bool(feature_spectral_include_k0)
        self.feature_spectral_assignment = str(feature_spectral_assignment)
        self.feature_spectral_gate_init = float(feature_spectral_gate_init)

        self.external_tensor_specs = normalize_external_tensor_specs(
            external_tensor_specs,
            external_tensor_rank=external_tensor_rank,
            external_tensor_irrep=external_tensor_irrep,
            external_tensor_parity=external_tensor_parity,
        )
        self.external_tensor_total_numel = external_tensor_total_numel(self.external_tensor_specs)
        single_external_spec = self.external_tensor_specs[0] if self.external_tensor_specs and len(self.external_tensor_specs) == 1 else None
        self.external_tensor_rank = int(single_external_spec["rank"]) if single_external_spec is not None else None
        self.external_tensor_irrep = str(single_external_spec["irrep"]) if single_external_spec is not None else None
        self.num_fidelity_levels = int(num_fidelity_levels)
        self.multi_fidelity_mode = str(multi_fidelity_mode or "conditioning").strip().lower()
        if self.multi_fidelity_mode not in {"conditioning", "delta-baseline"}:
            raise ValueError(f"Unsupported multi_fidelity_mode {self.multi_fidelity_mode!r}")
        self.o3_irrep_preset = str(o3_irrep_preset or "auto").strip().lower()
        self.o3_active_irreps_config = list(o3_active_irreps) if isinstance(o3_active_irreps, (list, tuple)) else o3_active_irreps
        self.physical_tensor_representation = "ictd"
        self.symmetry_mode = "o3"

        normalized_physical_tensor_specs: dict[str, dict] | None = None
        if physical_tensor_outputs is not None:
            normalized_physical_tensor_specs = {
                name: _normalize_physical_tensor_spec(name, spec_in, lmax=self.lmax)
                for name, spec_in in physical_tensor_outputs.items()
            }
        self.active_irreps = resolve_o3_active_irreps(
            lmax=self.lmax,
            preset=self.o3_irrep_preset,
            explicit_irreps=self.o3_active_irreps_config,
            external_tensor_irrep=self.external_tensor_irrep,
            external_tensor_specs=self.external_tensor_specs,
            physical_tensor_outputs=normalized_physical_tensor_specs,
        )
        self.active_irrep_set = set(self.active_irreps)
        self.active_irreps_str = [_irrep_key_str(l, p) for l, p in self.active_irreps]
        self.irreps_dim = _irreps_total_dim_o3(self.channels, self.active_irreps)

        self.max_radius = main_max_radius
        self.number_of_basis = main_number_of_basis
        self.function_type = function_type_main
        self.invariant_channels = int(invariant_channels)

        self.e3_conv_emb = ICTDO3E3Conv(
            max_radius=max_embed_radius,
            number_of_basis=main_number_of_basis,
            channels_out=self.channels,
            embedding_dim=embedding_dim,
            max_atomvalue=max_atomvalue,
            output_size=output_size,
            lmax=self.lmax,
            function_type=function_type_main,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
            external_tensor_rank=external_tensor_rank,
            external_tensor_irrep=self.external_tensor_irrep,
            external_tensor_specs=self.external_tensor_specs,
            active_irreps=self.active_irreps,
            external_tensor_input_repr=external_tensor_input_repr,
            external_tensor_channels_in=external_tensor_channels_in,
            external_tensor_include_trace_chain=external_tensor_include_trace_chain,
            num_fidelity_levels=self.num_fidelity_levels,
            external_tensor_scale_init=external_tensor_scale_init,
        )
        self.tp2_layers = nn.ModuleList()
        self.fc2_layers = nn.ModuleList()
        for _ in range(self.num_interaction - 1):
            tp2 = HarmonicFullyConnectedTensorProductO3(
                mul_in1=self.channels,
                mul_in2=1,
                mul_out=self.channels,
                lmax=self.lmax,
                active_irreps=self.active_irreps,
                internal_weights=True,
                path_policy=ictd_tp_path_policy,
                max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
                ictd_tp_backend=ictd_tp_backend,
            )
            fc2 = nn.Sequential(
                nn.Linear(main_number_of_basis, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, tp2.num_paths),
            )
            self.tp2_layers.append(tp2)
            self.fc2_layers.append(fc2)

        combined_channels = self.channels * self.num_interaction
        scalar_channels = (self.num_interaction - 1) * self.invariant_channels
        self.W_read = nn.ParameterDict()
        for l, p in self.active_irreps:
            self.W_read[_irrep_key_str(l, p)] = nn.Parameter(
                torch.randn(scalar_channels, combined_channels, combined_channels) * 0.02
            )
        self.readout_linear = nn.Sequential(
            nn.Linear(scalar_channels + combined_channels * len(self.active_irreps), embed_size[0]),
            nn.SiLU(),
            nn.Linear(embed_size[0], 17),
        )
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")

        self._p5_adapt = nn.ModuleList()
        self.product5_muls_by_irrep = {key: self.channels for key in self.active_irreps}
        for _ in range(self.num_interaction):
            layer_adapt = nn.ModuleDict()
            for l, p in self.active_irreps:
                layer_adapt[_irrep_key_str(l, p)] = nn.Identity()
            self._p5_adapt.append(layer_adapt)
        self.product_5 = HarmonicElementwiseProductO3(
            active_irreps=self.active_irreps,
            mul=combined_channels,
            irreps_out="0e",
            internal_compute_dtype=internal_compute_dtype,
        )

        sum_mul = self.channels * len(self.active_irreps)
        self.proj_total = MainNet(self.num_interaction * sum_mul + scalar_channels, embed_size, 17)
        self.delta_proj_total: nn.ModuleDict | None = None
        self.delta_weighted_sum: nn.ModuleDict | None = None
        if self.multi_fidelity_mode == "delta-baseline" and self.num_fidelity_levels > 1:
            self.delta_proj_total = nn.ModuleDict()
            self.delta_weighted_sum = nn.ModuleDict()
            for fid in range(1, self.num_fidelity_levels):
                head = MainNet(self.num_interaction * sum_mul + scalar_channels, embed_size, 17)
                zero_init_module_output(head)
                self.delta_proj_total[str(fid)] = head
                sum_head = RobustScalarWeightedSum(17)
                with torch.no_grad():
                    sum_head.weights.zero_()
                self.delta_weighted_sum[str(fid)] = sum_head
        self.long_range_module = build_long_range_module(
            mode=self.long_range_mode,
            feature_dim=self.proj_total.input_size if hasattr(self.proj_total, "input_size") else self.num_interaction * sum_mul + scalar_channels,
            hidden_dim=self.long_range_hidden_dim,
            boundary=self.long_range_boundary,
            neutralize=self.long_range_neutralize,
            filter_hidden_dim=self.long_range_filter_hidden_dim,
            kmax=self.long_range_kmax,
            mesh_size=self.long_range_mesh_size,
            slab_padding_factor=self.long_range_slab_padding_factor,
            include_k0=self.long_range_include_k0,
            source_channels=self.long_range_source_channels,
            backend=self.long_range_backend,
            reciprocal_backend=self.long_range_reciprocal_backend,
            energy_partition=self.long_range_energy_partition,
            green_mode=self.long_range_green_mode,
            assignment=self.long_range_assignment,
            mesh_fft_full_ewald=self.long_range_mesh_fft_full_ewald,
            theta=self.long_range_theta,
            leaf_size=self.long_range_leaf_size,
            multipole_order=self.long_range_multipole_order,
            far_source_dim=self.long_range_far_source_dim,
            far_num_shells=self.long_range_far_num_shells,
            far_shell_growth=self.long_range_far_shell_growth,
            far_tail=self.long_range_far_tail,
            far_tail_bins=self.long_range_far_tail_bins,
            far_stats=self.long_range_far_stats,
            far_max_radius_multiplier=self.long_range_far_max_radius_multiplier,
            far_source_norm=self.long_range_far_source_norm,
            far_gate_init=self.long_range_far_gate_init,
            cutoff_radius=self.max_radius,
        )
        self.long_range_num_k = getattr(self.long_range_module, "num_k", None) if self.long_range_module is not None else None
        self.feature_spectral_module = build_feature_spectral_module(
            mode=self.feature_spectral_mode,
            feature_dim=self.proj_total.input_size if hasattr(self.proj_total, "input_size") else self.num_interaction * sum_mul + scalar_channels,
            bottleneck_dim=self.feature_spectral_bottleneck_dim,
            mesh_size=self.feature_spectral_mesh_size,
            filter_hidden_dim=self.feature_spectral_filter_hidden_dim,
            boundary=self.feature_spectral_boundary,
            slab_padding_factor=self.feature_spectral_slab_padding_factor,
            neutralize=self.feature_spectral_neutralize,
            include_k0=self.feature_spectral_include_k0,
            assignment=self.feature_spectral_assignment,
            gate_init=self.feature_spectral_gate_init,
        )
        self.long_range_runtime_backend = "none"
        self.long_range_runtime_source_kind = "none"
        self.long_range_runtime_source_channels = 0
        self.long_range_runtime_source_layout = "none"
        self.long_range_runtime_source_boundary = self.long_range_boundary
        self.long_range_runtime_source_slab_padding_factor = self.long_range_slab_padding_factor
        if self.long_range_module is not None and bool(getattr(self.long_range_module, "exports_reciprocal_source", False)):
            self.long_range_runtime_backend = str(getattr(self.long_range_module, "runtime_backend", "none"))
            self.long_range_runtime_source_kind = str(getattr(self.long_range_module, "source_kind", "latent_charge"))
            self.long_range_runtime_source_channels = int(getattr(self.long_range_module, "source_channels", self.long_range_source_channels))
            self.long_range_runtime_source_layout = str(getattr(self.long_range_module, "source_layout", "channels_last"))
            self.long_range_runtime_source_boundary = self.long_range_boundary
            self.long_range_runtime_source_slab_padding_factor = self.long_range_slab_padding_factor
            self.reciprocal_source_channels = self.long_range_runtime_source_channels
            self.reciprocal_source_boundary = self.long_range_runtime_source_boundary
            self.reciprocal_source_slab_padding_factor = self.long_range_runtime_source_slab_padding_factor
        elif self.feature_spectral_module is not None:
            self.long_range_runtime_backend = "mesh_fft"
            self.long_range_runtime_source_kind = "feature_bottleneck"
            self.long_range_runtime_source_channels = self.feature_spectral_bottleneck_dim
            self.long_range_runtime_source_layout = "channels_last"
            self.long_range_runtime_source_boundary = self.feature_spectral_boundary
            self.long_range_runtime_source_slab_padding_factor = self.feature_spectral_slab_padding_factor
            self.reciprocal_source_channels = self.feature_spectral_bottleneck_dim
            self.reciprocal_source_boundary = self.feature_spectral_boundary
            self.reciprocal_source_slab_padding_factor = self.feature_spectral_slab_padding_factor
        else:
            self.reciprocal_source_channels = 0
            self.reciprocal_source_boundary = "periodic"
            self.reciprocal_source_slab_padding_factor = 2

        self._physical_tensor_specs: dict[str, dict] | None = None
        self.physical_tensor_heads: nn.ModuleDict | None = None
        if normalized_physical_tensor_specs is not None:
            self._physical_tensor_specs = {}
            heads = nn.ModuleDict()
            for name, spec in normalized_physical_tensor_specs.items():
                per_l = nn.ModuleDict()
                for l in spec["ls"]:
                    if spec["channels_out"][l] == combined_channels:
                        per_l[str(l)] = nn.Identity()
                    else:
                        per_l[str(l)] = nn.Linear(combined_channels, spec["channels_out"][l], bias=False)
                heads[name] = per_l
                self._physical_tensor_specs[name] = spec
                for l in spec["ls"]:
                    key = (int(l), int(spec["parity_by_l"][l]))
                    if key not in self.active_irrep_set:
                        raise ValueError(
                            f"physical_tensor_outputs[{name!r}] requires active irrep {_irrep_key_str(*key)} "
                            f"but active_irreps are {self.active_irreps_str}"
                        )
            self.physical_tensor_heads = heads

    def prewarm_caches(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        warm_device = device if device is not None else next(self.parameters()).device
        warm_dtype = dtype if dtype is not None else next(self.parameters()).dtype
        self.e3_conv_emb.tp2.prewarm_caches(device=warm_device, dtype=warm_dtype)
        for tp2 in self.tp2_layers:
            tp2.prewarm_caches(device=warm_device, dtype=warm_dtype)

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
        sync_after_scatter: callable | None = None,
        external_tensor: torch.Tensor | None = None,
        fidelity_ids: torch.Tensor | None = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
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

        Y_list = direction_harmonics_all(n.to(dtype=dtype), self.lmax)
        f1 = self.e3_conv_emb(
            pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
            precomputed_n=n,
            precomputed_edge_length=edge_length,
            precomputed_Y_list=Y_list,
            external_tensor=external_tensor,
            fidelity_ids=fidelity_ids,
        )
        if sync_after_scatter is not None:
            f1 = sync_after_scatter(f1)
        features = [f1]

        num_nodes = pos.size(0)
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1)
        emb_base = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        for tp2, fc2 in zip(self.tp2_layers, self.fc2_layers):
            f_prev = features[-1]
            emb = emb_base.to(dtype=f_prev.dtype)
            gates = fc2(emb)
            x1 = _split_irreps_o3(f_prev, self.channels, self.active_irreps)
            x1e = {key: blk[edge_src] for key, blk in x1.items()}
            Y = {}
            for l in range(self.lmax + 1):
                key = (l, canonical_irrep_parity_sign(l))
                if key in self.active_irrep_set:
                    Y[key] = Y_list[l].to(dtype=f_prev.dtype).unsqueeze(-2)
            edge_blocks = tp2(x1e, Y, gates)
            edge_flat = _merge_irreps_o3(edge_blocks, self.channels, self.active_irreps)
            neighbor_count = neighbor_count.to(edge_flat.dtype)
            f_next = scatter(edge_flat, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
            if sync_after_scatter is not None:
                f_next = sync_after_scatter(f_next)
            features.append(f_next)

        f_combine = torch.cat(features, dim=-1)
        xb = _split_irreps_o3(f_combine, self.channels * self.num_interaction, self.active_irreps)

        physical_out: dict[str, dict[int, torch.Tensor]] | None = None
        if return_physical_tensors:
            if self.physical_tensor_heads is None or self._physical_tensor_specs is None:
                raise ValueError("return_physical_tensors=True but physical_tensor_outputs was not set in __init__")
            physical_out = {}
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            counts = None
            for name, per_l in self.physical_tensor_heads.items():
                spec = self._physical_tensor_specs[name]
                reduce = spec["reduce"]
                out_blocks: dict[int, torch.Tensor] = {}
                for l in spec["ls"]:
                    p = spec["parity_by_l"][l]
                    y_l = _apply_channel_adapter_per_irrep(xb[(l, p)], per_l[str(l)])
                    if reduce == "none":
                        out_blocks[l] = y_l
                    else:
                        yg = scatter(y_l, batch, dim=0, dim_size=num_graphs, reduce="sum")
                        if reduce == "mean":
                            if counts is None:
                                counts = scatter(
                                    torch.ones_like(batch, dtype=yg.dtype),
                                    batch,
                                    dim=0,
                                    dim_size=num_graphs,
                                    reduce="sum",
                                ).clamp(min=1.0)
                            yg = yg / counts.view(-1, 1, 1)
                        out_blocks[l] = yg
                physical_out[name] = out_blocks

        scalars = torch.zeros(
            f_combine.shape[0],
            (self.num_interaction - 1) * self.invariant_channels,
            device=f_combine.device,
            dtype=f_combine.dtype,
        )
        for l, p in self.active_irreps:
            t = xb[(l, p)]
            gram = torch.einsum("ncm,ndm->ncd", t, t) / math.sqrt(2 * l + 1)
            scalars = scalars + torch.einsum("ocd,ncd->no", self.W_read[_irrep_key_str(l, p)], gram)

        T_blocks: dict[tuple[int, int], torch.Tensor] = {}
        splits = [_split_irreps_o3(f, self.channels, self.active_irreps) for f in features]
        for key in self.active_irreps:
            kstr = _irrep_key_str(*key)
            parts = []
            for i in range(len(features)):
                b_lp = splits[i][key]
                b_lp = _apply_channel_adapter_per_irrep(b_lp, self._p5_adapt[i][kstr])
                parts.append(b_lp)
            T_blocks[key] = torch.cat(parts, dim=-2)
        T_blocks[(0, 1)] = torch.cat([T_blocks[(0, 1)], scalars.unsqueeze(-1)], dim=-2)
        f_prod5 = self.product_5(T_blocks, T_blocks)
        feature_reciprocal_source = None
        if self.feature_spectral_module is not None:
            f_prod5, feature_reciprocal_source = self.feature_spectral_module(f_prod5, pos, batch, cell)

        long_range_energy = None
        reciprocal_source = None
        defer_long_range_to_runtime = False
        if self.long_range_module is not None:
            if return_reciprocal_source and bool(getattr(self.long_range_module, "exports_reciprocal_source", False)):
                long_range_energy, reciprocal_source = self.long_range_module(
                    f_prod5, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst, return_source=True
                )
                defer_long_range_to_runtime = reciprocal_source.numel() > 0
            else:
                long_range_energy = self.long_range_module(
                    f_prod5, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst
                )
        if reciprocal_source is None:
            reciprocal_source = feature_reciprocal_source

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        e_scalar = e_out.sum(dim=-1, keepdim=True)
        if self.multi_fidelity_mode == "delta-baseline":
            e_scalar = apply_delta_energy_heads(
                e_scalar,
                f_prod5,
                batch,
                fidelity_ids,
                self.delta_proj_total,
                self.delta_weighted_sum,
            )
        if long_range_energy is not None and not defer_long_range_to_runtime:
            e_scalar = e_scalar + long_range_energy
        if reciprocal_source is None and return_reciprocal_source:
            reciprocal_source = e_scalar.new_empty((e_scalar.size(0), 0))
        if return_physical_tensors and return_reciprocal_source:
            return e_scalar, physical_out, reciprocal_source
        if return_physical_tensors:
            return e_scalar, physical_out
        if return_reciprocal_source:
            return e_scalar, reciprocal_source
        return e_scalar
