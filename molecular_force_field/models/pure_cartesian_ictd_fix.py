from __future__ import annotations

import math
from typing import Dict, List

import opt_einsum_fx
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import soft_one_hot_linspace

from molecular_force_field.models.ictd_irreps import (
    EdgeWeightedPathPreservingTensorProduct,
    EquivariantChannelLinearSO3,
    EquivariantChannelLinearSO3Rect,
    HarmonicElementwiseProduct,
    HarmonicPathWeightedTensorProduct,
    direction_harmonics_all,
    ictd_u_matrix_so3,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    EquivariantScalarReadoutSO3,
    _irreps_total_dim,
    _merge_irreps,
    _split_irreps,
    apply_channel_adapter_per_l,
    resolve_save_multiple_mix_channels,
)
from molecular_force_field.models._mace_symmetric_contraction import MaceSymmetricContraction
from molecular_force_field.models.mlp import MainNet
from molecular_force_field.utils.scatter import scatter


_CONTRACTION_BATCH_EXAMPLE = 10
_CONTRACTION_ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


def _node_type_indices(node_attrs: torch.Tensor) -> torch.Tensor:
    if node_attrs.dim() == 1:
        return node_attrs.long()
    return node_attrs.argmax(dim=-1).long()


def _init_contraction_basis_logits_(logits: torch.Tensor, first_order_logit: float = 4.0) -> None:
    """Start from a stable order-1-dominant contraction instead of free large mixing."""
    with torch.no_grad():
        logits.zero_()
        logits[:, 0, :].fill_(float(first_order_logit))


def _init_contraction_basis_weight_(weight: torch.Tensor, higher_order_std: float = 0.02) -> None:
    """Free ablation: order-1 starts as passthrough, higher orders start small."""
    with torch.no_grad():
        weight.zero_()
        weight[:, 0, :].fill_(1.0)
        if weight.shape[1] > 1 and higher_order_std > 0:
            weight[:, 1:, :].normal_(mean=0.0, std=float(higher_order_std))


def _init_contraction_path_weight_(weight: torch.Tensor, std: float = 0.02) -> None:
    with torch.no_grad():
        weight.fill_(1.0)
        if std > 0:
            weight.add_(torch.randn_like(weight) * float(std))


def _init_path_tp_weight_to_one_(module: nn.Module | None) -> None:
    """ICTD path TPs multiply radial/contraction weights; do not start them near zero."""
    if module is not None and hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        with torch.no_grad():
            module.weight.fill_(1.0)


def _init_linear_identity_(module: nn.Module | None) -> None:
    if not isinstance(module, nn.Linear):
        return
    if module.weight.shape[0] != module.weight.shape[1]:
        return
    with torch.no_grad():
        nn.init.eye_(module.weight)
        if module.bias is not None:
            module.bias.zero_()


def _init_so3_linear_identity_(module: nn.Module | None) -> None:
    adapters = getattr(module, "adapters", None)
    if adapters is None:
        return
    for adapter in adapters.values():
        _init_linear_identity_(adapter)


def _init_element_conditioned_identity_(module: nn.Module | None) -> None:
    weights = getattr(module, "weights", None)
    if weights is None:
        return
    with torch.no_grad():
        for weight in weights.values():
            if weight.shape[-2] != weight.shape[-1]:
                continue
            weight.zero_()
            eye = torch.eye(weight.shape[-1], dtype=weight.dtype, device=weight.device)
            weight.copy_(eye.unsqueeze(0).expand_as(weight))
        bias = getattr(module, "bias", None)
        if bias is not None:
            for value in bias.values():
                value.zero_()


def _hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{int(channels)}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(lmax) + 1)))


def _so3_flat_to_mace_features(x: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    blocks = _split_irreps(x, int(channels), int(lmax))
    return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)


def _merge_blocks_subset(blocks: Dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    return torch.cat([blocks[l].reshape(blocks[l].shape[0], int(channels) * (2 * l + 1)) for l in range(int(lmax) + 1)], dim=-1)


def _concat_so3_states_by_l(states: List[torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    """
    Concatenate multiple SO3-flat states by channel within each l-block.

    Each state is laid out as [l0 | l1 | ...]. Directly concatenating states
    along the flat dimension would produce [s0_l0 | s0_l1 | ... | s1_l0 | ...],
    which is not a valid SO3-flat layout for a larger channel count. Equivariant
    operators expect [all_l0_channels | all_l1_channels | ...].
    """
    if len(states) == 0:
        raise ValueError("states must contain at least one SO3-flat tensor")
    split_states = [_split_irreps(state, int(channels), int(lmax)) for state in states]
    parts = []
    for l in range(int(lmax) + 1):
        block = torch.cat([split_state[l] for split_state in split_states], dim=-2)
        parts.append(block.reshape(*block.shape[:-2], block.shape[-2] * block.shape[-1]))
    return torch.cat(parts, dim=-1)


def _tp_allowed_paths_from_target_lmax(lmax_in1: int, lmax_in2: int, lmax_target: int) -> List[tuple[int, int, int]]:
    """
    Mirror MACE/e3nn instruction pruning at the SO3 level:
    keep only paths (l1, l2, l3) whose output irrep l3 is present in the target set.

    For our current ICTD fix baseline, target irreps are exactly all l=0..lmax_target.
    This helper still makes the path set explicit and keeps interaction TP
    aligned with the same contract used by MACE TensorProduct instructions.
    """
    paths: List[tuple[int, int, int]] = []
    target_ls = set(range(int(lmax_target) + 1))
    for l1 in range(int(lmax_in1) + 1):
        for l2 in range(int(lmax_in2) + 1):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                if l3 not in target_ls:
                    continue
                if l3 > int(lmax_target):
                    continue
                if (l1 + l2 + l3) % 2 == 1:
                    continue
                paths.append((l1, l2, l3))
    return paths


def _tp_allowed_paths_to_output_l(lmax_in1: int, lmax_in2: int, output_l: int) -> List[tuple[int, int, int]]:
    paths: List[tuple[int, int, int]] = []
    l3 = int(output_l)
    for l1 in range(int(lmax_in1) + 1):
        for l2 in range(int(lmax_in2) + 1):
            if not (abs(l1 - l2) <= l3 <= l1 + l2):
                continue
            if (l1 + l2 + l3) % 2 == 1:
                continue
            paths.append((l1, l2, l3))
    return paths


class ElementConditionedLinearSO3(nn.Module):
    def __init__(self, num_elements: int, channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.num_elements = int(num_elements)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.dim = _irreps_total_dim(self.channels, self.lmax)
        self.weights = nn.ParameterDict(
            {
                str(l): nn.Parameter(torch.randn(self.num_elements, self.channels, self.channels) * 0.02)
                for l in range(self.lmax + 1)
            }
        )
        if bias:
            self.bias = nn.ParameterDict(
                {
                    str(l): nn.Parameter(torch.zeros(self.num_elements, self.channels))
                    for l in range(self.lmax + 1)
                }
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        attrs = node_attrs.to(dtype=x.dtype)
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            weight = self.weights[str(l)].to(dtype=x.dtype)
            mixed_weight = torch.einsum("ne,eoi->noi", attrs, weight)
            out_block = torch.einsum("noi,nid->nod", mixed_weight, blocks[l])
            if self.bias is not None:
                mixed_bias = torch.einsum("ne,eo->no", attrs, self.bias[str(l)].to(dtype=x.dtype))
                out_block = out_block + mixed_bias.unsqueeze(-1)
            out_blocks[l] = out_block
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class PerLScaleSO3(nn.Module):
    def __init__(self, channels: int, lmax: int, init_scales: list[float] | tuple[float, ...]):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        if len(init_scales) != self.lmax + 1:
            raise ValueError(f"Expected {self.lmax + 1} init scales, got {len(init_scales)}")
        scales = torch.as_tensor(init_scales, dtype=torch.get_default_dtype()).clamp_min(1e-6)
        self.log_scale = nn.Parameter(scales.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        scales = self.log_scale.to(dtype=x.dtype, device=x.device).exp()
        for l in range(self.lmax + 1):
            out_blocks[l] = blocks[l] * scales[l]
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class PathPreservingLinearSO3(nn.Module):
    def __init__(self, in_channels_by_l: Dict[int, int], out_channels: int, lmax: int):
        super().__init__()
        self.in_channels_by_l = {int(k): int(v) for k, v in in_channels_by_l.items()}
        self.out_channels = int(out_channels)
        self.lmax = int(lmax)
        self.weights = nn.ParameterDict()
        for l in range(self.lmax + 1):
            in_channels = self.in_channels_by_l.get(l, 0)
            weight = nn.Parameter(torch.empty(self.out_channels, in_channels))
            if in_channels > 0:
                nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(float(in_channels)))
            else:
                nn.init.zeros_(weight)
            self.weights[str(l)] = weight

    def forward(self, blocks: Dict[int, torch.Tensor]) -> torch.Tensor:
        out_blocks: Dict[int, torch.Tensor] = {}
        sample = next(iter(blocks.values()))
        for l in range(self.lmax + 1):
            x_l = blocks[l]
            weight = self.weights[str(l)].to(dtype=x_l.dtype, device=x_l.device)
            if x_l.shape[-2] == 0:
                out_blocks[l] = torch.zeros(
                    *x_l.shape[:-2], self.out_channels, 2 * l + 1, dtype=x_l.dtype, device=x_l.device
                )
            else:
                out_blocks[l] = torch.einsum("oc,ncm->nom", weight, x_l)
        return _merge_irreps(out_blocks, self.out_channels, self.lmax)


class ICTDSymmetricContractionSO3(nn.Module):
    """
    MACE-style symmetric contraction implemented with ICTD-SO3 operators.

    This keeps the higher-order ICTD-SO3 paths explicit as a basis list and then
    combines those basis terms with compact element-conditioned coefficients.
    That is closer in spirit to MACE's explicit product basis than the previous
    shared-contraction-plus-output-gating implementation.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        self.num_elements = int(num_elements)
        if contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(f"contraction_combine must be 'softmax', 'free', or 'path-free', got {contraction_combine!r}")
        self.contraction_combine = str(contraction_combine)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.reduce)
        self.order_mix = nn.ModuleList(
            [EquivariantChannelLinearSO3(self.hidden_channels, self.lmax, bias=False) for _ in range(self.correlation)]
        )
        _init_so3_linear_identity_(self.order_mix[0])
        self.tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        for tp in self.tp_layers:
            _init_path_tp_weight_to_one_(tp)
        self.tp_path_weight = nn.ParameterList()
        if self.contraction_combine == "path-free":
            for tp in self.tp_layers:
                weight = nn.Parameter(torch.empty(self.num_elements, tp.num_paths, self.hidden_channels))
                _init_contraction_path_weight_(weight)
                self.tp_path_weight.append(weight)
        self.out_linear = EquivariantChannelLinearSO3(
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.out_linear)
        if self.contraction_combine == "softmax":
            self.basis_logits = nn.ParameterDict(
                {
                    str(l): nn.Parameter(
                        torch.zeros(self.num_elements, self.correlation, self.hidden_channels)
                    )
                    for l in range(self.lmax + 1)
                }
            )
            self.basis_weight = None
            for logits in self.basis_logits.values():
                _init_contraction_basis_logits_(logits)
        else:
            self.basis_logits = None
            self.basis_weight = nn.ParameterDict(
                {
                    str(l): nn.Parameter(
                        torch.empty(self.num_elements, self.correlation, self.hidden_channels)
                    )
                    for l in range(self.lmax + 1)
                }
            )
            for weight in self.basis_weight.values():
                _init_contraction_basis_weight_(weight)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        element_index = _node_type_indices(node_attrs)

        basis_terms = [self.order_mix[0](base)]
        if self.correlation > 1:
            base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
            current_blocks = base_blocks
            for order_idx, tp in enumerate(self.tp_layers, start=1):
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.tp_path_weight[order_idx - 1][element_index].to(dtype=base.dtype)
                current_blocks = tp(current_blocks, base_blocks, path_channel_weights=path_weight)
                current_flat = _merge_irreps(current_blocks, self.hidden_channels, self.lmax)
                basis_terms.append(self.order_mix[order_idx](current_flat))

        basis_blocks = [_split_irreps(term, self.hidden_channels, self.lmax) for term in basis_terms]
        combined_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            if self.contraction_combine == "softmax":
                coeff = torch.softmax(self.basis_logits[str(l)][element_index].to(dtype=base.dtype), dim=1)
            else:
                coeff = self.basis_weight[str(l)][element_index].to(dtype=base.dtype)
            stack = torch.stack([term_blocks[l] for term_blocks in basis_blocks], dim=1)
            combined_blocks[l] = torch.sum(stack * coeff.unsqueeze(-1), dim=1)
        combined = _merge_irreps(combined_blocks, self.hidden_channels, self.lmax)
        return self.out_linear(combined)


class ICTDProductBasisBlock(nn.Module):
    """
    MACE-style product block:
      h_{t+1} = linear( symmetric_contraction_ictd(message, node_attrs) ) + sc
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.symmetric_contractions = ICTDSymmetricContractionSO3(
            num_elements=num_elements,
            in_channels=channels,
            hidden_channels=channels,
            lmax=lmax,
            correlation=correlation,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
            contraction_combine=contraction_combine,
        )
        self.linear = EquivariantChannelLinearSO3(channels, lmax, bias=False)
        _init_so3_linear_identity_(self.linear)
        self.output_norm = nn.Identity()

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        contracted = self.symmetric_contractions(node_feats, node_attrs)
        out = self.linear(contracted)
        if sc is not None:
            out = out + sc
        return self.output_norm(out)


class ICTDScalarSymmetricContractionSO3(nn.Module):
    """
    Scalar-target MACE-style contraction.

    This mirrors MACE's final product target `64x0e`: the last TP step is
    instruction-pruned to l_out=0, and all order mixing/output projection happens
    only in scalar channel space.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        self.num_elements = int(num_elements)
        if contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(f"contraction_combine must be 'softmax', 'free', or 'path-free', got {contraction_combine!r}")
        self.contraction_combine = str(contraction_combine)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.reduce)
        self.scalar_order_mix = nn.ModuleList(
            [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False) for _ in range(self.correlation)]
        )
        _init_linear_identity_(self.scalar_order_mix[0])
        self.full_tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 2, 0))
            ]
        )
        scalar_paths = _tp_allowed_paths_to_output_l(self.lmax, self.lmax, 0)
        self.final_scalar_tp = (
            HarmonicPathWeightedTensorProduct(
                channels=self.hidden_channels,
                lmax=self.lmax,
                allowed_paths=scalar_paths,
                path_policy=ictd_tp_path_policy,
                max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
            )
            if self.correlation > 1
            else None
        )
        for tp in self.full_tp_layers:
            _init_path_tp_weight_to_one_(tp)
        _init_path_tp_weight_to_one_(self.final_scalar_tp)
        self.full_tp_path_weight = nn.ParameterList()
        self.final_scalar_path_weight = None
        if self.contraction_combine == "path-free":
            for tp in self.full_tp_layers:
                weight = nn.Parameter(torch.empty(self.num_elements, tp.num_paths, self.hidden_channels))
                _init_contraction_path_weight_(weight)
                self.full_tp_path_weight.append(weight)
            if self.final_scalar_tp is not None:
                self.final_scalar_path_weight = nn.Parameter(
                    torch.empty(self.num_elements, self.final_scalar_tp.num_paths, self.hidden_channels)
                )
                _init_contraction_path_weight_(self.final_scalar_path_weight)
        self.out_linear = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        _init_linear_identity_(self.out_linear)
        if self.contraction_combine == "softmax":
            self.basis_logits = nn.Parameter(
                torch.zeros(self.num_elements, self.correlation, self.hidden_channels)
            )
            self.basis_weight = None
            _init_contraction_basis_logits_(self.basis_logits)
        else:
            self.basis_logits = None
            self.basis_weight = nn.Parameter(
                torch.empty(self.num_elements, self.correlation, self.hidden_channels)
            )
            _init_contraction_basis_weight_(self.basis_weight)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
        element_index = _node_type_indices(node_attrs)

        basis_terms = [self.scalar_order_mix[0](base_blocks[0].squeeze(-1))]
        current_blocks = base_blocks
        for order_idx in range(1, self.correlation):
            if order_idx == self.correlation - 1:
                if self.final_scalar_tp is None:
                    raise RuntimeError("final_scalar_tp unexpectedly missing")
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.final_scalar_path_weight[element_index].to(dtype=base.dtype)
                current_blocks = self.final_scalar_tp(current_blocks, base_blocks, path_channel_weights=path_weight)
            else:
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.full_tp_path_weight[order_idx - 1][element_index].to(dtype=base.dtype)
                current_blocks = self.full_tp_layers[order_idx - 1](
                    current_blocks,
                    base_blocks,
                    path_channel_weights=path_weight,
                )
            scalar = current_blocks[0].squeeze(-1)
            basis_terms.append(self.scalar_order_mix[order_idx](scalar))

        if self.contraction_combine == "softmax":
            coeff = torch.softmax(self.basis_logits[element_index].to(dtype=base.dtype), dim=1)
        else:
            coeff = self.basis_weight[element_index].to(dtype=base.dtype)
        stack = torch.stack(basis_terms, dim=1)
        combined = torch.sum(stack * coeff, dim=1)
        return self.out_linear(combined)


class ICTDScalarProductBasisBlock(nn.Module):
    """
    MACE-style final product block for keep_last_layer_irreps=False.

    Native MACE changes the last product target irreps to scalar-only. This
    block uses a scalar-target ICTD contraction instead of building the full
    output irreps and slicing l=0 afterward.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.symmetric_contractions = ICTDScalarSymmetricContractionSO3(
            num_elements=num_elements,
            in_channels=channels,
            hidden_channels=channels,
            lmax=lmax,
            correlation=correlation,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
            contraction_combine=contraction_combine,
        )
        self.linear = nn.Linear(self.channels, self.channels, bias=False)
        _init_linear_identity_(self.linear)
        self.output_norm = nn.Identity()

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        contracted = self.symmetric_contractions(node_feats, node_attrs)
        out = self.linear(contracted)
        if sc is not None:
            if sc.shape[-1] == self.channels:
                sc_scalar = sc
            else:
                sc_scalar = _split_irreps(sc, self.channels, self.lmax)[0].squeeze(-1)
            out = out + sc_scalar
        return self.output_norm(out)


class SO3ToE3NNBasisBridge(nn.Module):
    """
    Fixed per-l orthogonal bridge between the ICTD SO3 basis and e3nn/MACE basis.

    `direction_harmonics_all` now matches e3nn component normalization in RMS, but
    each l block can still differ by an orthogonal basis convention. Native MACE
    contraction assumes the e3nn convention, while ICTD interaction emits ICTD
    convention features. The bridge uses a deterministic least-squares/SVD fit
    from sampled directions to construct Q_l such that:

        Y_ictd_l @ Q_l ~= Y_e3nn_l
    """

    def __init__(self, channels: int, lmax: int, num_samples: int = 8192):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260426)
        dirs = torch.randn(int(num_samples), 3, generator=generator, dtype=torch.float64)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y_ictd = direction_harmonics_all(dirs, self.lmax)
        y_e3nn = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(self.lmax),
            dirs,
            normalize=True,
            normalization="component",
        )
        offset = 0
        for l in range(self.lmax + 1):
            width = 2 * l + 1
            a = y_ictd[l].to(dtype=torch.float64)
            b = y_e3nn[:, offset : offset + width].to(dtype=torch.float64)
            offset += width
            u, _, vh = torch.linalg.svd(a.T @ b)
            q = (u @ vh).to(dtype=torch.get_default_dtype())
            self.register_buffer(f"q_{l}", q, persistent=True)

    def _q(self, l: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return getattr(self, f"q_{int(l)}").to(dtype=dtype, device=device)

    def ictd_flat_to_e3nn_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,mp->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def e3nn_flat_to_ictd_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,pm->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def ictd_flat_to_e3nn_features(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)

    def ictd_flat_to_e3nn_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))

    def e3nn_flat_to_ictd_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.e3nn_flat_to_ictd_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))


class NativeMACEProductBasisBlockSO3(nn.Module):
    """
    Hybrid product block: ICTD-SO3 interaction features are interpreted in
    MACE/e3nn mul-ir layout, then contracted by the native MACE symmetric
    contraction implementation.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=False,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, self.lmax)

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        x = self.basis_bridge.ictd_flat_to_e3nn_features(node_feats, self.lmax)
        out = self.linear(self.symmetric_contractions(x, node_attrs))
        if sc is not None:
            sc = self.basis_bridge.ictd_flat_to_e3nn_flat(sc, self.target_lmax)
            if sc.shape[-1] == out.shape[-1]:
                out = out + sc
            elif self.target_lmax == 0:
                if sc.shape[-1] == self.channels:
                    out = out + sc
                else:
                    out = out + _split_irreps(sc, self.channels, self.lmax)[0].squeeze(-1)
            else:
                raise ValueError(f"Cannot add sc shape {tuple(sc.shape)} to native product output {tuple(out.shape)}")
        if self.target_lmax > 0:
            out = self.basis_bridge.e3nn_flat_to_ictd_flat(out, self.target_lmax)
        return out


class ICTDBridgeUSymmetricContractionSO3(nn.Module):
    """
    Bridge-U symmetric contraction expressed directly in the ICTD basis.

    This is algebraically equivalent to:
      ICTD features -> e3nn basis -> MACE SymmetricContraction -> ICTD basis
    but the per-l basis change is folded into the stored U tensors once at
    initialization. The forward path therefore consumes and returns ICTD-basis
    flat SO3 features. This backend is the stable high-l bridge used when
    pure ICTD U generation is not numerically reliable.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, self.lmax)
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=False,
        )
        self._fold_basis_change_into_u_tensors()

    def _input_q(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        blocks = [self.basis_bridge._q(l, dtype=dtype, device=device) for l in range(self.lmax + 1)]
        return torch.block_diag(*blocks)

    def _transform_u_tensor(self, u_tensor: torch.Tensor, output_l: int) -> torch.Tensor:
        dtype = u_tensor.dtype
        device = u_tensor.device
        q_in = self._input_q(dtype=dtype, device=device)
        nu = int(u_tensor.dim() - 1) if int(output_l) == 0 else int(u_tensor.dim() - 2)
        if int(output_l) == 0:
            if nu == 1:
                return torch.einsum("ai,ip->ap", q_in, u_tensor)
            if nu == 2:
                return torch.einsum("ai,bj,ijp->abp", q_in, q_in, u_tensor)
            if nu == 3:
                return torch.einsum("ai,bj,ck,ijkp->abcp", q_in, q_in, q_in, u_tensor)
        q_out = self.basis_bridge._q(output_l, dtype=dtype, device=device)
        if nu == 1:
            return torch.einsum("ro,ai,oip->rap", q_out, q_in, u_tensor)
        if nu == 2:
            return torch.einsum("ro,ai,bj,oijp->rabp", q_out, q_in, q_in, u_tensor)
        if nu == 3:
            return torch.einsum("ro,ai,bj,ck,oijkp->rabcp", q_out, q_in, q_in, q_in, u_tensor)
        raise NotImplementedError(f"ICTD bridge-U contraction currently supports correlation<=3, got nu={nu}")

    def _fold_basis_change_into_u_tensors(self) -> None:
        with torch.no_grad():
            for (mul, ir), contraction in zip(self.target_irreps, self.symmetric_contractions.contractions):
                del mul
                output_l = int(ir.l)
                for nu in range(1, int(contraction.correlation) + 1):
                    name = f"U_matrix_{nu}"
                    old = getattr(contraction, name)
                    new = self._transform_u_tensor(old, output_l)
                    old.copy_(new.to(dtype=old.dtype, device=old.device))

    def forward(self, node_feats: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        return self.symmetric_contractions(x, node_attrs)


class ICTDBridgeUProductBasisBlockSO3(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = ICTDBridgeUSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.symmetric_contractions(node_feats, node_attrs))
        if sc is not None:
            if sc.shape[-1] == out.shape[-1]:
                out = out + sc
            elif self.target_lmax == 0:
                if sc.shape[-1] == self.channels:
                    out = out + sc
                else:
                    out = out + _split_irreps(sc, self.channels, self.lmax)[0].squeeze(-1)
            else:
                raise ValueError(f"Cannot add sc shape {tuple(sc.shape)} to bridge-U output {tuple(out.shape)}")
        return out


# Backward-compatible aliases for checkpoints/scripts that still refer to the old name.
ICTDMACEUSymmetricContractionSO3 = ICTDBridgeUSymmetricContractionSO3
ICTDMACEUProductBasisBlockSO3 = ICTDBridgeUProductBasisBlockSO3


class _ICTDPureUContraction(nn.Module):
    """MACE contraction recursion over caller-provided ICTD U tensors."""

    def __init__(
        self,
        *,
        u_tensors: Dict[int, torch.Tensor],
        output_l: int,
        num_elements: int,
        num_features: int,
    ):
        super().__init__()
        self.output_l = int(output_l)
        self.correlation = int(max(u_tensors))
        self.num_elements = int(num_elements)
        self.num_features = int(num_features)
        for nu in range(1, self.correlation + 1):
            self.register_buffer(f"U_matrix_{nu}", u_tensors[nu].contiguous())

        self.contractions_weighting = nn.ModuleList()
        self.contractions_features = nn.ModuleList()
        self.weights = nn.ParameterList([])

        for i in range(self.correlation, 0, -1):
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * self.output_l + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == self.correlation:
                parse_subscript_main = (
                    [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum("".join(parse_subscript_main), x, y, w, z)
                )
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn([num_equivariance] + [num_ell] * i + [num_params]).squeeze(0),
                        torch.randn((self.num_elements, num_params, self.num_features)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                    ),
                )
                self.weights_max = nn.Parameter(
                    torch.randn((self.num_elements, num_params, self.num_features)) / max(num_params, 1)
                )
            else:
                parse_subscript_weighting = (
                    [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1))]
                    + ["k,ekc,be->bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i - 1 + min(self.output_l, 1))]
                    + ["i,bci->bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i - 1 + min(self.output_l, 1))]
                )

                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum("".join(parse_subscript_weighting), x, y, z)
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )
                self.contractions_weighting.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_weighting,
                        example_inputs=(
                            torch.randn([num_equivariance] + [num_ell] * i + [num_params]).squeeze(0),
                            torch.randn((self.num_elements, num_params, self.num_features)),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                        ),
                    )
                )
                self.contractions_features.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_features,
                        example_inputs=(
                            torch.randn(
                                [_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_equivariance]
                                + [num_ell] * i
                            ).squeeze(2),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        ),
                    )
                )
                self.weights.append(
                    nn.Parameter(torch.randn((self.num_elements, num_params, self.num_features)) / max(num_params, 1))
                )

    def U_tensors(self, nu: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"U_matrix_{int(nu)}"]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.graph_opt_main(self.U_tensors(self.correlation), self.weights_max, x, y)
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(self.U_tensors(self.correlation - i - 1), weight, y)
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)
        return out.view(out.shape[0], -1)


class ICTDPureUSymmetricContractionSO3(nn.Module):
    """
    MACE-style symmetric contraction with U tensors generated from ICTD CG only.

    This keeps the optimized MACE contraction/einsum wrapper and trainable
    per-element weights, but replaces every `U_matrix_real` buffer with the
    corresponding ICTD-basis U generated by `ictd_u_matrix_so3`. It is the
    pure-ICTD contraction ablation against `ictd-bridge-u`.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.contractions = nn.ModuleList()
        dtype = torch.get_default_dtype()
        for mul, ir in self.target_irreps:
            del mul
            output_l = int(ir.l)
            u_tensors = {
                nu: ictd_u_matrix_so3(
                    lmax=self.lmax,
                    output_l=output_l,
                    correlation=nu,
                    irrep_normalization="component",
                    dtype=dtype,
                )
                for nu in range(1, int(correlation) + 1)
            }
            self.contractions.append(
                _ICTDPureUContraction(
                    u_tensors=u_tensors,
                    output_l=output_l,
                    num_elements=int(num_elements),
                    num_features=self.channels,
                )
            )

    def forward(self, node_feats: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        return torch.cat([contraction(x, node_attrs) for contraction in self.contractions], dim=-1)


class ICTDPureUProductBasisBlockSO3(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = ICTDPureUSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.symmetric_contractions(node_feats, node_attrs))
        if sc is not None:
            if sc.shape[-1] == out.shape[-1]:
                out = out + sc
            elif self.target_lmax == 0:
                if sc.shape[-1] == self.channels:
                    out = out + sc
                else:
                    out = out + _split_irreps(sc, self.channels, self.lmax)[0].squeeze(-1)
            else:
                raise ValueError(f"Cannot add sc shape {tuple(sc.shape)} to ICTD pure-U output {tuple(out.shape)}")
        return out


class ICTDPureUFusionMixBlockSO3(nn.Module):
    """
    Fusion-only full-CG ICTD contraction.

    This replaces the older `MultipleContractionSO3` path in `ictd-fix` fusion:
    concat layer states -> SO3 channel reduction -> full-CG ICTD contraction
    -> SO3 linear. The contraction is the same `ictd-pure-u` implementation used
    by the MACE-style product blocks, so fusion does not reintroduce the old
    natural-parity-only contraction.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        self.symmetric_contractions = ICTDPureUSymmetricContractionSO3(
            num_elements=int(num_elements),
            channels=self.hidden_channels,
            lmax=self.lmax,
            target_lmax=self.lmax,
            correlation=int(correlation),
        )
        self.linear = o3.Linear(_hidden_irreps(self.hidden_channels, self.lmax), _hidden_irreps(self.hidden_channels, self.lmax))

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        return self.linear(self.symmetric_contractions(reduced, node_attrs))


class ICTDBridgeUFusionMixBlockSO3(nn.Module):
    """
    Fusion-only bridge-U contraction.

    This mirrors `ICTDPureUFusionMixBlockSO3`, but uses the bridge-U MACE
    U tensors folded into ICTD coordinates. It is the fallback for lmax > 3,
    where directly generated pure ICTD U tensors are not stable enough.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        self.symmetric_contractions = ICTDBridgeUSymmetricContractionSO3(
            num_elements=int(num_elements),
            channels=self.hidden_channels,
            lmax=self.lmax,
            target_lmax=self.lmax,
            correlation=int(correlation),
        )
        self.linear = o3.Linear(
            _hidden_irreps(self.hidden_channels, self.lmax),
            _hidden_irreps(self.hidden_channels, self.lmax),
        )

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        return self.linear(self.symmetric_contractions(reduced, node_attrs))


class FusionProduct5ReadoutSO3(nn.Module):
    """Old product5/proj_total readout applied directly to fused feature blocks."""

    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        num_feature_blocks: int,
        hidden_sizes: list[int] | tuple[int, ...],
        output_init_std: float = 0.003,
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.num_feature_blocks = int(num_feature_blocks)
        if self.num_feature_blocks < 1:
            raise ValueError(f"num_feature_blocks must be >= 1, got {self.num_feature_blocks}")
        self.product_5 = HarmonicElementwiseProduct(
            lmax=self.lmax,
            mul=self.channels * self.num_feature_blocks,
            irreps_out="0e",
            internal_compute_dtype=internal_compute_dtype,
        )
        proj_in_dim = self.channels * self.num_feature_blocks * (self.lmax + 1)
        self.proj_total = MainNet(proj_in_dim, list(hidden_sizes), 1, output_init_std=output_init_std)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(features) != self.num_feature_blocks:
            raise ValueError(f"Expected {self.num_feature_blocks} feature blocks, got {len(features)}")
        split_features = [_split_irreps(x, self.channels, self.lmax) for x in features]
        t_blocks = {
            l: torch.cat([blocks[l] for blocks in split_features], dim=-2)
            for l in range(self.lmax + 1)
        }
        f_prod5 = self.product_5(t_blocks, t_blocks)
        return self.proj_total(f_prod5)


class FusionProduct5ReadoutSO3Mixed(nn.Module):
    """Product5 readout for SO3 blocks with different channel counts."""

    def __init__(
        self,
        *,
        channels_per_block: list[int] | tuple[int, ...],
        lmax: int,
        hidden_sizes: list[int] | tuple[int, ...],
        output_init_std: float = 0.003,
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels_per_block = tuple(int(c) for c in channels_per_block)
        self.lmax = int(lmax)
        self.num_feature_blocks = len(self.channels_per_block)
        if self.num_feature_blocks < 1:
            raise ValueError("channels_per_block must contain at least one feature block")
        if any(c < 1 for c in self.channels_per_block):
            raise ValueError(f"channels_per_block entries must be positive, got {self.channels_per_block}")
        self.total_channels = int(sum(self.channels_per_block))
        self.product_5 = HarmonicElementwiseProduct(
            lmax=self.lmax,
            mul=self.total_channels,
            irreps_out="0e",
            internal_compute_dtype=internal_compute_dtype,
        )
        proj_in_dim = self.total_channels * (self.lmax + 1)
        self.proj_total = MainNet(proj_in_dim, list(hidden_sizes), 1, output_init_std=output_init_std)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(features) != self.num_feature_blocks:
            raise ValueError(f"Expected {self.num_feature_blocks} feature blocks, got {len(features)}")
        split_features = [
            _split_irreps(x, channels, self.lmax)
            for x, channels in zip(features, self.channels_per_block)
        ]
        t_blocks = {
            l: torch.cat([blocks[l] for blocks in split_features], dim=-2)
            for l in range(self.lmax + 1)
        }
        f_prod5 = self.product_5(t_blocks, t_blocks)
        return self.proj_total(f_prod5)


class MACEStyleScalarReadoutSO3(nn.Module):
    """Native MACE final readout shape for this config: Cx0e -> 16x0e -> 1x0e."""

    def __init__(self, channels: int, hidden_channels: int = 16, output_init_std: float = 0.003):
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.linear_1 = nn.Linear(self.channels, self.hidden_channels, bias=True)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(self.hidden_channels, 1, bias=True)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, scalar_feats: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(scalar_feats)))


class ICTDResidualInteractionBlock(nn.Module):
    """
    ICTD-SO3 interaction block with MACE-like interface.

    Returns:
      - message: scatter-aggregated neighbor message
      - sc:      element-conditioned self-connection
    """

    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        input_lmax: int | None = None,
        target_lmax: int | None = None,
        sc_lmax: int | None = None,
        number_of_basis: int,
        num_elements: int,
        function_type: str = "gaussian",
        ictd_save_tp_mode: str = "fully-connected",
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        equivariant_post_linear: bool = False,
        use_self_connection: bool = True,
        avg_num_neighbors: float | None = None,
        message_scale_init: list[float] | tuple[float, ...] | None = None,
        sc_scale_init: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.input_lmax = self.lmax if input_lmax is None else int(input_lmax)
        self.target_lmax = self.lmax if target_lmax is None else int(target_lmax)
        self.sc_lmax = self.input_lmax if sc_lmax is None else int(sc_lmax)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)
        self.equivariant_post_linear = bool(equivariant_post_linear)
        self.use_self_connection = bool(use_self_connection)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        allowed_paths = _tp_allowed_paths_from_target_lmax(
            lmax_in1=self.input_lmax,
            lmax_in2=self.lmax,
            lmax_target=self.target_lmax,
        )
        self.linear_up = EquivariantChannelLinearSO3(self.channels, self.input_lmax, bias=False)
        self.tp = EdgeWeightedPathPreservingTensorProduct(
            channels=self.channels,
            lmax=self.lmax,
            allowed_paths=allowed_paths,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
        )
        _init_path_tp_weight_to_one_(self.tp)
        self.fc = nn.Sequential(
            nn.Linear(self.number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.num_paths * self.channels),
        )
        self.message_linear = PathPreservingLinearSO3(
            {
                l: self.channels * int(self.tp.path_counts_by_l.get(l, 0))
                for l in range(self.target_lmax + 1)
            },
            out_channels=self.channels,
            lmax=self.target_lmax,
        )
        _init_so3_linear_identity_(self.linear_up)
        self.message_selector = (
            ElementConditionedLinearSO3(
                num_elements=num_elements,
                channels=self.channels,
                lmax=self.target_lmax,
                bias=False,
            )
            if not self.use_self_connection
            else None
        )
        _init_element_conditioned_identity_(self.message_selector)
        self.self_connection = (
            ElementConditionedLinearSO3(
                num_elements=num_elements,
                channels=self.channels,
                lmax=self.sc_lmax,
                bias=False,
            )
            if self.use_self_connection
            else None
        )
        _init_element_conditioned_identity_(self.self_connection)
        self.message_norm = nn.Identity()
        self.sc_norm = nn.Identity()
        self.message_output_scale = (
            PerLScaleSO3(self.channels, self.target_lmax, message_scale_init)
            if message_scale_init is not None
            else nn.Identity()
        )
        self.sc_output_scale = (
            PerLScaleSO3(self.channels, self.sc_lmax, sc_scale_init)
            if sc_scale_init is not None
            else nn.Identity()
        )

    def forward(
        self,
        *,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: Dict[int, torch.Tensor],
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        sync_after_scatter: callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        num_nodes = node_feats.size(0)

        node_feats_for_sc = node_feats
        node_feats = self.linear_up(node_feats)
        gates = self.fc(edge_feats)
        x1 = _split_irreps(node_feats, self.channels, self.input_lmax)
        x1e = {l: x1[l][edge_src] for l in range(self.input_lmax + 1)}
        edge_blocks = self.tp(x1e, edge_attrs, gates)
        if edge_mask is not None:
            mask = edge_mask.to(dtype=node_feats.dtype)
            edge_blocks = {l: block * mask.unsqueeze(-1) for l, block in edge_blocks.items()}
        message_blocks = {
            l: scatter(edge_blocks[l], edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
            for l in range(self.target_lmax + 1)
        }
        if self.avg_num_neighbors is None:
            if edge_mask is not None:
                avg_num_neighbors = float(edge_mask.detach().sum().item()) / float(max(num_nodes, 1))
            else:
                avg_num_neighbors = float(edge_src.numel()) / float(max(num_nodes, 1))
        else:
            avg_num_neighbors = self.avg_num_neighbors
        message = self.message_linear(message_blocks) / max(avg_num_neighbors, 1e-8)
        if sync_after_scatter is not None:
            message = sync_after_scatter(message)
        if not self.use_self_connection:
            message = self.message_selector(message, node_attrs)
        message = self.message_norm(message)
        message = self.message_output_scale(message)
        sc = None
        if self.self_connection is not None:
            if self.sc_lmax == self.input_lmax:
                sc_input = node_feats_for_sc
            elif self.sc_lmax == 0:
                sc_input = _split_irreps(node_feats_for_sc, self.channels, self.input_lmax)[0].reshape(
                    node_feats_for_sc.shape[0], self.channels
                )
            else:
                raise ValueError(
                    f"Unsupported ICTD self-connection projection input_lmax={self.input_lmax}, sc_lmax={self.sc_lmax}"
                )
            sc = self.self_connection(sc_input, node_attrs)
            sc = self.sc_norm(sc)
            sc = self.sc_output_scale(sc)
        return message, sc


class PureCartesianICTDFix(nn.Module):
    """
    ICTD-SO3 model organized with a MACE-style backbone:

      h_t -> interaction_t(node_attrs, h_t, edge_*) = (m_t, sc_t)
          -> product_t(m_t, sc_t, node_attrs) = h_{t+1}
          -> layer_readout_t(h_{t+1})

    Optional route:
      - baseline: sum(layerwise readouts)
      - fusion:   sum(layerwise readouts) + E_fusion(h_1, ..., h_N)
    """

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
        atomic_numbers: list[int] | tuple[int, ...] | None = None,
        output_size: int = 8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        ictd_Lmax: int = 6,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        max_rank_other: int = 1,
        k_policy: str = "k0",
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        product5_muls_by_l: dict[int, int] | None = None,
        invariant_channels: int = 32,
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
        equivariant_post_linear: bool = False,
        ictd_save_tp_mode: str = "fully-connected",
        ictd_fix_route: str = "baseline",
        ictd_fix_contraction_combine: str = "softmax",
        ictd_fix_product_backend: str = "ictd-pure-u",
        ictd_fix_interaction_scale: str = "none",
        ictd_fix_fusion_scale_init: float = 0.1,
        ictd_fix_fusion_heads: int = 1,
        ictd_fix_fusion_head_weight_mode: str = "softmax",
        ictd_fix_fusion_input_scale_init: float = 1.0,
        ictd_fix_fusion_input_scale_trainable: bool = False,
        ictd_fix_gmix_gate_init: float = 1.0,
        ictd_fix_gmix_gate_trainable: bool = False,
        ictd_fix_readout_head_scale_init: float = 1.0,
        ictd_fix_readout_head_scale_trainable: bool = False,
        ictd_fix_fusion_readout_mixed_channels: bool = True,
        save_contraction_order: int = 3,
        save_multiple_mix_channels: int | None = None,
        avg_num_neighbors: float | None = None,
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if int(num_interaction) < 2:
            raise ValueError(f"num_interaction must be >= 2, got {num_interaction}")
        if ictd_fix_route not in {"baseline", "fusion"}:
            raise ValueError(f"ictd_fix_route must be 'baseline' or 'fusion', got {ictd_fix_route!r}")
        if ictd_fix_contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(
                f"ictd_fix_contraction_combine must be 'softmax', 'free', or 'path-free', got {ictd_fix_contraction_combine!r}"
            )
        requested_product_backend = str(ictd_fix_product_backend)
        if requested_product_backend == "ictd-mace-u":
            requested_product_backend = "ictd-bridge-u"
        if requested_product_backend not in {"ictd", "native-mace", "ictd-bridge-u", "ictd-pure-u"}:
            raise ValueError(
                "ictd_fix_product_backend must be 'ictd', 'native-mace', 'ictd-bridge-u', "
                f"'ictd-mace-u' alias, or 'ictd-pure-u', got {ictd_fix_product_backend!r}"
            )
        if ictd_fix_interaction_scale not in {"none", "mace-rms"}:
            raise ValueError(
                f"ictd_fix_interaction_scale must be 'none' or 'mace-rms', got {ictd_fix_interaction_scale!r}"
            )
        if ictd_fix_fusion_head_weight_mode not in {"softmax", "free"}:
            raise ValueError(
                "ictd_fix_fusion_head_weight_mode must be 'softmax' or 'free', "
                f"got {ictd_fix_fusion_head_weight_mode!r}"
            )
        if long_range_mode != "none" or feature_spectral_mode != "none":
            raise NotImplementedError("pure-cartesian-ictd-fix currently supports only long_range_mode=none and feature_spectral_mode=none")

        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.num_interaction = int(num_interaction)
        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.ictd_fix_route = str(ictd_fix_route)
        self.ictd_fix_contraction_combine = str(ictd_fix_contraction_combine)
        self.ictd_fix_requested_product_backend = requested_product_backend
        self.ictd_fix_product_backend = (
            "ictd-bridge-u"
            if requested_product_backend == "ictd-pure-u" and self.lmax > 3
            else requested_product_backend
        )
        self.ictd_fix_product_backend_fallback = self.ictd_fix_product_backend != self.ictd_fix_requested_product_backend
        self.ictd_fix_interaction_scale = str(ictd_fix_interaction_scale)
        self.ictd_fix_fusion_scale_init = float(ictd_fix_fusion_scale_init)
        self.ictd_fix_fusion_heads = int(ictd_fix_fusion_heads)
        self.ictd_fix_fusion_head_weight_mode = str(ictd_fix_fusion_head_weight_mode)
        self.ictd_fix_fusion_input_scale_init = float(ictd_fix_fusion_input_scale_init)
        self.ictd_fix_fusion_input_scale_trainable = bool(ictd_fix_fusion_input_scale_trainable)
        self.ictd_fix_gmix_gate_init = float(ictd_fix_gmix_gate_init)
        self.ictd_fix_gmix_gate_trainable = bool(ictd_fix_gmix_gate_trainable)
        self.ictd_fix_readout_head_scale_init = float(ictd_fix_readout_head_scale_init)
        self.ictd_fix_readout_head_scale_trainable = bool(ictd_fix_readout_head_scale_trainable)
        self.ictd_fix_fusion_readout_mixed_channels = bool(ictd_fix_fusion_readout_mixed_channels)
        if self.ictd_fix_fusion_heads < 1:
            raise ValueError(f"ictd_fix_fusion_heads must be >= 1, got {self.ictd_fix_fusion_heads}")
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
        product_target_lmax = [
            self.lmax if layer_idx < self.num_interaction - 1 else 0
            for layer_idx in range(self.num_interaction)
        ]
        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.ictd_fix_effective_product_backends: list[str] = []
        for layer_idx, target_lmax in enumerate(product_target_lmax):
            effective_product_backend = self.ictd_fix_product_backend
            self.ictd_fix_effective_product_backends.append(effective_product_backend)
            input_lmax = 0 if layer_idx == 0 else self.lmax
            sc_lmax = (
                self.lmax
                if (self.ictd_fix_route == "fusion" and layer_idx == self.num_interaction - 1)
                else target_lmax
            )
            message_scale_init = None
            sc_scale_init = None
            if self.ictd_fix_interaction_scale == "mace-rms":
                # Initialized from the ICTD/native-MACE basisbridge diagnostic on
                # aspirin lmax=3/ch64. The scales are learnable, so this is a
                # stabilization prior rather than a fixed calibration.
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
                )
            )
            if effective_product_backend == "native-mace":
                self.products.append(
                    NativeMACEProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                    )
                )
            elif effective_product_backend == "ictd-bridge-u":
                self.products.append(
                    ICTDBridgeUProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                    )
                )
            elif effective_product_backend == "ictd-pure-u":
                self.products.append(
                    ICTDPureUProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                    )
                )
            elif target_lmax == self.lmax:
                self.products.append(
                    ICTDProductBasisBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        ictd_tp_path_policy=ictd_tp_path_policy,
                        ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                        internal_compute_dtype=internal_compute_dtype,
                        ictd_tp_backend=ictd_tp_backend,
                        contraction_combine=self.ictd_fix_contraction_combine,
                    )
                )
            else:
                self.products.append(
                    ICTDScalarProductBasisBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        ictd_tp_path_policy=ictd_tp_path_policy,
                        ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                        internal_compute_dtype=internal_compute_dtype,
                        ictd_tp_backend=ictd_tp_backend,
                        contraction_combine=self.ictd_fix_contraction_combine,
                    )
                )
        self.layer_energy_readouts = nn.ModuleList(
            [EquivariantScalarReadoutSO3(self.channels, self.lmax, output_init_std=0.003) for _ in range(self.num_interaction - 1)]
        )
        self.last_layer_energy_readout = MACEStyleScalarReadoutSO3(
            self.channels,
            hidden_channels=16,
            output_init_std=0.003,
        )
        if self.ictd_fix_readout_head_scale_trainable:
            self.readout_head_scales = nn.Parameter(
                torch.full((2,), self.ictd_fix_readout_head_scale_init, dtype=torch.get_default_dtype())
            )
        else:
            self.readout_head_scales = None

        if self.ictd_fix_route == "fusion":
            self.save_multiple_mix_channels = resolve_save_multiple_mix_channels(
                self.channels,
                self.num_interaction,
                save_multiple_mix_channels,
            )
            fusion_mix_cls = ICTDBridgeUFusionMixBlockSO3 if self.lmax > 3 else ICTDPureUFusionMixBlockSO3
            self.ictd_fix_fusion_mix_backend = "ictd-bridge-u" if self.lmax > 3 else "ictd-pure-u"
            self.multiple_contraction_mix = fusion_mix_cls(
                num_elements=self.num_elements,
                in_channels=self.channels * self.num_interaction,
                hidden_channels=self.save_multiple_mix_channels,
                lmax=self.lmax,
                correlation=save_contraction_order,
            )
            self.multiple_contract_fuse = (
                nn.Identity()
                if self.ictd_fix_fusion_readout_mixed_channels or self.save_multiple_mix_channels == self.channels
                else EquivariantChannelLinearSO3Rect(
                    self.save_multiple_mix_channels,
                    self.channels,
                    self.lmax,
                    bias=False,
                )
            )
            if self.ictd_fix_fusion_input_scale_trainable:
                self.fusion_input_scales = nn.Parameter(
                    torch.full(
                        (self.num_interaction,),
                        self.ictd_fix_fusion_input_scale_init,
                        dtype=torch.get_default_dtype(),
                    )
                )
            else:
                self.fusion_input_scales = None
            if self.ictd_fix_gmix_gate_trainable:
                self.g_mix_gate = nn.Parameter(
                    torch.tensor(self.ictd_fix_gmix_gate_init, dtype=torch.get_default_dtype())
                )
            else:
                self.g_mix_gate = None
            if self.ictd_fix_fusion_readout_mixed_channels:
                fusion_readout_cls = FusionProduct5ReadoutSO3Mixed
                fusion_readout_kwargs = dict(
                    channels_per_block=[self.channels] * self.num_interaction + [self.save_multiple_mix_channels],
                    lmax=self.lmax,
                    hidden_sizes=main_hidden_sizes3,
                    output_init_std=0.003,
                    internal_compute_dtype=internal_compute_dtype,
                )
            else:
                fusion_readout_cls = FusionProduct5ReadoutSO3
                fusion_readout_kwargs = dict(
                    channels=self.channels,
                    lmax=self.lmax,
                    num_feature_blocks=self.num_interaction + 1,
                    hidden_sizes=main_hidden_sizes3,
                    output_init_std=0.003,
                    internal_compute_dtype=internal_compute_dtype,
                )
            self.fusion_readouts = nn.ModuleList(
                [fusion_readout_cls(**fusion_readout_kwargs) for _ in range(self.ictd_fix_fusion_heads)]
            )
            self.fusion_readout = self.fusion_readouts[0]
            if self.ictd_fix_fusion_heads == 1:
                self.fusion_head_logits = None
                self.fusion_head_weights = None
                self.fusion_energy_scale = nn.Parameter(
                    torch.tensor(self.ictd_fix_fusion_scale_init, dtype=torch.get_default_dtype())
                )
            elif self.ictd_fix_fusion_head_weight_mode == "softmax":
                self.fusion_head_logits = nn.Parameter(
                    torch.zeros(self.ictd_fix_fusion_heads, dtype=torch.get_default_dtype())
                )
                self.fusion_head_weights = None
                self.fusion_energy_scale = nn.Parameter(
                    torch.tensor(self.ictd_fix_fusion_scale_init, dtype=torch.get_default_dtype())
                )
            else:
                self.fusion_head_logits = None
                self.fusion_head_weights = nn.Parameter(
                    torch.full(
                        (self.ictd_fix_fusion_heads,),
                        self.ictd_fix_fusion_scale_init / self.ictd_fix_fusion_heads,
                        dtype=torch.get_default_dtype(),
                    )
                )
                self.fusion_energy_scale = None
        else:
            self.save_multiple_mix_channels = None
            self.multiple_contraction_mix = None
            self.multiple_contract_fuse = None
            self.ictd_fix_fusion_mix_backend = None
            self.fusion_readouts = None
            self.fusion_readout = None
            self.fusion_head_logits = None
            self.fusion_head_weights = None
            self.fusion_energy_scale = None
            self.fusion_input_scales = None
            self.g_mix_gate = None

    def _readout_head_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.readout_head_scales is None:
            return ref.new_tensor(self.ictd_fix_readout_head_scale_init)
        return self.readout_head_scales[index].to(dtype=ref.dtype, device=ref.device)

    def _fusion_input_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.fusion_input_scales is None:
            return ref.new_tensor(self.ictd_fix_fusion_input_scale_init)
        return self.fusion_input_scales[index].to(dtype=ref.dtype, device=ref.device)

    def _g_mix_gate(self, ref: torch.Tensor) -> torch.Tensor:
        if self.g_mix_gate is None:
            return ref.new_tensor(self.ictd_fix_gmix_gate_init)
        return self.g_mix_gate.to(dtype=ref.dtype, device=ref.device)

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
            raise ValueError("pure-cartesian-ictd-fix does not currently support return_physical_tensors=True")

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
        edge_feats = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True,
        ).mul(self.number_of_basis ** 0.5).to(dtype=dtype)

        A_long = A.long()
        if int(A_long.max().item()) >= self.atomic_number_to_index.numel():
            raise ValueError(
                f"Encountered atomic number {int(A_long.max().item())}, but compact mapping supports only up to "
                f"{self.atomic_number_to_index.numel() - 1}. atomic_numbers={self.atomic_numbers}"
            )
        compact_idx = self.atomic_number_to_index[A_long]
        if torch.any(compact_idx < 0):
            bad = torch.unique(A_long[compact_idx < 0]).tolist()
            raise ValueError(
                f"Encountered atomic numbers without compact mapping: {bad}. "
                f"Configured atomic_numbers={self.atomic_numbers}"
            )
        node_attrs = F.one_hot(compact_idx, num_classes=self.num_elements).to(dtype=dtype)
        h = self.node_embedding(node_attrs)

        layer_states: List[torch.Tensor] = []
        last_preproduct_state: torch.Tensor | None = None
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
            if self.ictd_fix_route == "fusion" and layer_idx == self.num_interaction - 1:
                last_preproduct_state = message if sc is None else message + sc
                layer_states.append(last_preproduct_state)
                h_last = product(node_feats=message, sc=sc, node_attrs=node_attrs)
                e_layer = self.last_layer_energy_readout(h_last)
                e_layer = self._readout_head_scale(1, e_layer) * e_layer
            else:
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
        if self.ictd_fix_route == "fusion":
            if last_preproduct_state is None:
                raise RuntimeError("fusion route expected a last_preproduct_state but none was produced")
            fusion_mix_states = [
                self._fusion_input_scale(idx, state) * state
                for idx, state in enumerate(layer_states)
            ]
            mix_inputs = _concat_so3_states_by_l(fusion_mix_states, self.channels, self.lmax)
            g_mix = self.multiple_contraction_mix(mix_inputs, node_attrs)
            g_mix = self._g_mix_gate(g_mix) * g_mix
            g_mix = self.multiple_contract_fuse(g_mix)
            fusion_inputs = layer_states + [g_mix]
            if self.ictd_fix_fusion_heads == 1:
                fusion_energy = self.fusion_readout(fusion_inputs)
                fusion_energy = self.fusion_energy_scale.to(dtype=fusion_energy.dtype, device=fusion_energy.device) * fusion_energy
            else:
                head_energies = torch.stack(
                    [readout(fusion_inputs) for readout in self.fusion_readouts],
                    dim=0,
                )
                if self.ictd_fix_fusion_head_weight_mode == "softmax":
                    head_weights = torch.softmax(
                        self.fusion_head_logits.to(dtype=head_energies.dtype, device=head_energies.device),
                        dim=0,
                    )
                    head_weights = self.fusion_energy_scale.to(
                        dtype=head_energies.dtype,
                        device=head_energies.device,
                    ) * head_weights
                else:
                    head_weights = self.fusion_head_weights.to(
                        dtype=head_energies.dtype,
                        device=head_energies.device,
                    )
                head_weights = head_weights.view(self.ictd_fix_fusion_heads, 1, 1)
                fusion_energy = (head_energies * head_weights).sum(dim=0)
            total_energy = fusion_energy if total_energy is None else (total_energy + fusion_energy)
            combined_features = torch.cat([combined_features, g_mix], dim=-1)

        out = total_energy.sum(dim=-1, keepdim=True)
        if return_combined_features:
            if return_reciprocal_source:
                return out, combined_features, out.new_empty((out.size(0), 0))
            return out, combined_features
        if return_reciprocal_source:
            return out, out.new_empty((out.size(0), 0))
        return out
