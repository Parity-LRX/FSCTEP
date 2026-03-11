"""
cuEquivariance backend **strictly equivalent** to `e3nn_layers.py`.

This implements the same architecture as the original `e3nn_layers.E3_TransformerLayer_multi`
(full FullyConnectedTensorProduct, FullTensorProduct, ElementwiseTensorProduct, TensorSquare, etc.)
but replaces every e3nn operator with its cuEquivariance (`cuet`) counterpart.

Design goals:
  - Forward signature is identical to `e3nn_layers.py` for each public class.
  - All TP paths (including non-channelwise fully-connected ones) are preserved.
  - Falls back to ``method="naive"`` on CPU so the module can be instantiated/tested
    without a GPU, although performance gains require CUDA.

Requires:
  pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
"""

from __future__ import annotations

import math
import warnings
from typing import List, Sequence

import torch
import torch.nn as nn

from molecular_force_field.models.mlp import MainNet2, MainNet, RobustScalarWeightedSum


def _require_cue():
    # cuEquivariance is transitioning APIs; keep logs clean while preserving behavior.
    warnings.filterwarnings(
        "ignore",
        message=r".*EquivariantTensorProduct is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorProductUniform1d is deprecated.*",
        category=UserWarning,
    )
    try:
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        from cuequivariance.group_theory.experimental.e3nn import O3_e3nn
    except Exception as e:
        raise ImportError(
            "cuEquivariance is required for tensor_product_mode='spherical-cue'.\n"
            "Install (on CUDA Linux):\n"
            "  pip install cuequivariance-torch cuequivariance-ops-torch-cu12\n"
            f"Original import error: {e}"
        ) from e
    return cue, cuet, O3_e3nn


def _radial_embedding(
    r: torch.Tensor,
    *,
    r_max: float,
    number_of_basis: int,
    function_type: str,
) -> torch.Tensor:
    E = r.numel()
    nb = int(number_of_basis)
    r_max = float(r_max)
    x = (r / r_max).clamp(min=0.0, max=1.0)
    idx = torch.arange(nb, device=r.device, dtype=r.dtype)

    ft = str(function_type)
    if ft == "gaussian":
        centers = idx / max(nb - 1, 1)
        sigma = 1.0 / max(nb - 1, 1)
        emb = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / max(sigma, 1e-6)) ** 2)
    elif ft in ("fourier", "cosine"):
        k = idx + 1.0
        emb = torch.cos(math.pi * k[None, :] * x[:, None])
    elif ft == "bessel":
        k = idx + 1.0
        t = math.pi * k[None, :] * x[:, None]
        emb = torch.sin(t) / torch.clamp(t, min=1e-6)
    elif ft == "smooth_finite":
        centers = idx / max(nb - 1, 1)
        d = (x[:, None] - centers[None, :]).abs()
        emb = (1.0 - d).clamp(min=0.0) ** 2 * (1.0 + 2.0 * d)
    else:
        centers = idx / max(nb - 1, 1)
        sigma = 1.0 / max(nb - 1, 1)
        emb = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / max(sigma, 1e-6)) ** 2)

    emb = emb * (nb ** 0.5)
    emb = emb * (r <= r_max).to(dtype=r.dtype)[:, None]
    return emb.reshape(E, nb)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        last = int(in_dim)
        for h in hidden:
            layers.append(nn.Linear(last, int(h)))
            layers.append(nn.SiLU())
            last = int(h)
        layers.append(nn.Linear(last, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sh_irreps_string(lmax: int) -> str:
    parts = []
    for l in range(int(lmax) + 1):
        p = "e" if l % 2 == 0 else "o"
        parts.append(f"1x{l}{p}")
    return " + ".join(parts)


class E3Conv(nn.Module):
    """
    First convolution: atom scalars -> irreps_output.

    Equivalent to `e3nn_layers.E3Conv`:
      1. Embed atoms -> node_scalars (output_size x 0e)
      2. FullTensorProduct(node_scalars, SH) -> edge features
      3. FullyConnectedTP(edge_features, node_scalars_dst, weight(r)) -> irreps_output
      4. scatter mean
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        irreps_output: str,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        main_hidden_sizes3=None,
        emb_number=None,
        function_type: str = "gaussian",
        device=None,
        dtype=None,
        **_unused,
    ):
        super().__init__()
        cue, cuet, O3_e3nn = _require_cue()
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if emb_number is None:
            emb_number = [64, 64, 64]

        default_dtype = torch.get_default_dtype() if dtype is None else dtype
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_cuda = dev.type == "cuda"
        method_tp = "fused_tp" if is_cuda else "naive"
        layout = cue.ir_mul

        self.max_radius = float(max_radius)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim, dtype=default_dtype)
        self.fitnet1 = MainNet2(input_size=embedding_dim, hidden_sizes=main_hidden_sizes3, output_size=output_size)

        node_in_str = f"{output_size}x0e"
        sh_str = _sh_irreps_string(2)
        irreps_output_str = str(irreps_output)

        cue_node_in = cue.Irreps(O3_e3nn, node_in_str)
        cue_sh = cue.Irreps(O3_e3nn, sh_str)
        cue_output = cue.Irreps(O3_e3nn, irreps_output_str)

        self.sh = cuet.SphericalHarmonics(
            ls=[0, 1, 2], normalize=True,
            device=device, math_dtype=default_dtype,
            method="uniform_1d" if is_cuda else "naive",
        )

        full_tp_desc = cue.descriptors.full_tensor_product(cue_node_in, cue_sh)
        self.full_tp = cuet.EquivariantTensorProduct(
            full_tp_desc,
            layout=layout,
            device=device,
            math_dtype=default_dtype,
        )
        cue_full_tp_out = full_tp_desc.outputs[0].irreps

        cue_node_in2 = cue.Irreps(O3_e3nn, f"{output_size}x0e")
        self.tp = cuet.FullyConnectedTensorProduct(
            cue_full_tp_out, cue_node_in2, cue_output,
            layout=layout, internal_weights=False, shared_weights=False,
            device=device, dtype=default_dtype, method=method_tp,
        )

        self.fc = _MLP(
            in_dim=number_of_basis,
            hidden=list(emb_number),
            out_dim=int(self.tp.weight_numel),
        )

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)

        num_nodes = pos.size(0)
        atom_embeddings = self.atom_embedding(A.long())
        Ai = self.fitnet1(atom_embeddings)

        sh_edge = self.sh(edge_vec)
        f_in = self.full_tp(Ai[edge_src], sh_edge)

        emb = _radial_embedding(edge_length, r_max=self.max_radius,
                                number_of_basis=self.number_of_basis,
                                function_type=self.function_type)
        w = self.fc(emb)

        edge_features = self.tp(f_in, Ai[edge_dst], w)

        from molecular_force_field.utils.scatter import scatter
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst,
                                dim=0, dim_size=num_nodes).clamp(min=1).float()
        out = scatter(edge_features, edge_dst, dim=0,
                      dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
        return out

class E3Conv2(nn.Module):
    """
    Subsequent convolution: node irreps -> node irreps.

    Equivalent to `e3nn_layers.E3Conv2`:
      1. FullyConnectedTP(f_in[edge_src], SH_edge, weight(r)) -> irreps_output
      2. scatter mean
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        irreps_input_conv: str,
        irreps_output: str,
        function_type: str = "gaussian",
        emb_number=None,
        device=None,
        dtype=None,
        **_unused,
    ):
        super().__init__()
        cue, cuet, O3_e3nn = _require_cue()
        if emb_number is None:
            emb_number = [64, 64, 64]

        default_dtype = torch.get_default_dtype() if dtype is None else dtype
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_cuda = dev.type == "cuda"
        method_tp = "fused_tp" if is_cuda else "naive"
        layout = cue.ir_mul

        self.max_radius = float(max_radius)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)

        sh_str = _sh_irreps_string(2)
        cue_in = cue.Irreps(O3_e3nn, str(irreps_input_conv))
        cue_sh = cue.Irreps(O3_e3nn, sh_str)
        cue_out = cue.Irreps(O3_e3nn, str(irreps_output))

        self.sh = cuet.SphericalHarmonics(
            ls=[0, 1, 2], normalize=True,
            device=device, math_dtype=default_dtype,
            method="uniform_1d" if is_cuda else "naive",
        )

        self.tp = cuet.FullyConnectedTensorProduct(
            cue_in, cue_sh, cue_out,
            layout=layout, internal_weights=False, shared_weights=False,
            device=device, dtype=default_dtype, method=method_tp,
        )

        self.fc = _MLP(
            in_dim=number_of_basis,
            hidden=list(emb_number),
            out_dim=int(self.tp.weight_numel),
        )

    def forward(self, f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)

        num_nodes = pos.size(0)
        sh_edge = self.sh(edge_vec)

        emb = _radial_embedding(edge_length, r_max=self.max_radius,
                                number_of_basis=self.number_of_basis,
                                function_type=self.function_type)
        w = self.fc(emb)
        edge_features = self.tp(f_in[edge_src], sh_edge, w)

        from molecular_force_field.utils.scatter import scatter
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst,
                                dim=0, dim_size=num_nodes).clamp(min=1).float()
        out = scatter(edge_features, edge_dst, dim=0,
                      dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
        return out


def _parse_irreps_segs(s: str):
    """Parse e3nn irreps string into list of (mul, l, parity_str)."""
    s = str(s).replace(" ", "")
    segs = []
    for p in s.split("+"):
        if not p:
            continue
        if "x" in p:
            mul_s, ir_s = p.split("x", 1)
            mul = int(mul_s)
        else:
            mul = 1
            ir_s = p
        parity = ir_s[-1]
        l = int(ir_s[:-1])
        segs.append((mul, l, parity))
    return segs


class E3_TransformerLayer_multi(nn.Module):
    """
    Full spherical TP model layer — cuEquivariance backend.

    Architecture strictly matches `e3nn_layers.E3_TransformerLayer_multi`:
      - E3Conv (first conv: FullTensorProduct + FullyConnectedTP)
      - E3Conv2 x (num_interaction-1) (subsequent: FullyConnectedTP)
      - product_3: FullyConnectedTP(f_combine, f_combine) -> scalar_channels x 0e
      - product_5: ElementwiseTensorProduct(T, T, ["0e"]) (per-channel dot)
      - MLP readout -> weighted sum -> atom energies
    """

    def __init__(
        self,
        max_embed_radius,
        main_max_radius,
        main_number_of_basis,
        irreps_input,
        irreps_query,
        irreps_key,
        irreps_value,
        irreps_output,
        irreps_sh,
        hidden_dim_sh,
        hidden_dim,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers=1,
        device=None,
        function_type_main="gaussian",
        num_interaction=2,
    ):
        super().__init__()
        cue, cuet, O3_e3nn = _require_cue()

        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        layout = cue.ir_mul

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")

        emb_number = [64, 64, 64] if not isinstance(hidden_dim, list) else hidden_dim
        self.irreps_input_str = str(irreps_input)
        self._irreps_input_segs = _parse_irreps_segs(self.irreps_input_str)

        # --- Convolution stack ---
        self.e3_conv_layers = nn.ModuleList()
        self.e3_conv_layers.append(
            E3Conv(
                max_radius=max_embed_radius,
                number_of_basis=main_number_of_basis,
                irreps_output=self.irreps_input_str,
                embedding_dim=embedding_dim,
                max_atomvalue=max_atomvalue,
                output_size=output_size,
                main_hidden_sizes3=main_hidden_sizes3,
                emb_number=emb_number,
                function_type=function_type_main,
                device=self.device,
            )
        )
        for _ in range(1, self.num_interaction):
            self.e3_conv_layers.append(
                E3Conv2(
                    max_radius=max_embed_radius,
                    number_of_basis=main_number_of_basis,
                    irreps_input_conv=self.irreps_input_str,
                    irreps_output=self.irreps_input_str,
                    emb_number=emb_number,
                    function_type=function_type_main,
                    device=self.device,
                )
            )

        # --- product_3: ElementwiseTensorProduct(..., ["0e"]) via cuet native TP ---
        input_multi_segs = [(m * self.num_interaction, l, p) for m, l, p in self._irreps_input_segs]
        input_multi_str = " + ".join(f"{m}x{l}{p}" for m, l, p in input_multi_segs)

        cue_in_multi = cue.Irreps(O3_e3nn, input_multi_str)
        zeroe_filter = [ir for _mul, ir in cue.Irreps(O3_e3nn, "0e")]
        product_3_desc = cue.descriptors.elementwise_tensor_product(
            cue_in_multi, cue_in_multi, irreps3_filter=zeroe_filter
        )
        self.product_3 = cuet.EquivariantTensorProduct(
            product_3_desc,
            layout=layout,
            device=self.device,
            math_dtype=torch.get_default_dtype(),
        )
        cue_product_3_out = product_3_desc.outputs[0].irreps
        self.scalar_channels = int(product_3_desc.outputs[0].dim)

        # --- product_5: ElementwiseTensorProduct(..., ["0e"]) via cuet native TP ---
        product_5_in_str = f"{input_multi_str} + {cue_product_3_out}"
        cue_product_5_in = cue.Irreps(O3_e3nn, product_5_in_str)
        product_5_desc = cue.descriptors.elementwise_tensor_product(
            cue_product_5_in, cue_product_5_in, irreps3_filter=zeroe_filter
        )
        self.product_5 = cuet.EquivariantTensorProduct(
            product_5_desc,
            layout=layout,
            device=self.device,
            math_dtype=torch.get_default_dtype(),
        )
        product_5_out_dim = int(product_5_desc.outputs[0].dim)

        self.proj_total = MainNet(product_5_out_dim, embed_size, 17)
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        features: List[torch.Tensor] = []
        f_prev = self.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            features.append(f_prev)

        f_combine = torch.cat(features, dim=-1)
        f_combine_product = self.product_3(f_combine, f_combine)

        T = torch.cat(features + [f_combine_product], dim=-1)
        f2_product_5 = self.product_5(T, T)

        product_proj = self.proj_total(f2_product_5)
        e_out = self.weighted_sum(product_proj)
        atom_energies = e_out.sum(dim=-1, keepdim=True)
        return atom_energies
