from __future__ import annotations

import math

import torch
from e3nn import o3

from molecular_force_field.models.ictd_irreps import (
    HarmonicFullyConnectedTensorProductO3,
    direction_harmonics_fast,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full_o3 import (
    PureCartesianICTDO3TransformerLayer,
    resolve_o3_active_irreps,
)
from molecular_force_field.utils.external_tensor_specs import pack_external_tensor_dict


def _make_o3_blocks(batch: int = 2, mul: int = 1, dtype: torch.dtype = torch.float64):
    active = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    blocks = {}
    for l, p in active:
        blocks[(l, p)] = torch.zeros(batch, mul, 2 * l + 1, dtype=dtype)
    return active, blocks


def _e3_to_ictd_basis_cpu(l: int) -> torch.Tensor:
    l = int(l)
    dim = 2 * l + 1
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1729 + l)
    xyz = torch.randn(max(128, 16 * dim), 3, generator=gen, dtype=torch.float64)
    xyz = torch.nn.functional.normalize(xyz, dim=-1)
    y_e3 = o3.spherical_harmonics(l, xyz, normalize=True, normalization="component")
    y_ictd = direction_harmonics_fast(xyz, l)
    return torch.linalg.lstsq(y_e3, y_ictd).solution.contiguous()


def _ictd_to_e3_basis_cpu(l: int) -> torch.Tensor:
    return torch.linalg.inv(_e3_to_ictd_basis_cpu(int(l))).contiguous()


def _entry_e3_to_ictd(entry: torch.Tensor, l: int) -> torch.Tensor:
    mat = _e3_to_ictd_basis_cpu(int(l)).to(device=entry.device, dtype=entry.dtype)
    return torch.matmul(entry, mat)


def _entry_ictd_to_e3(entry: torch.Tensor, l: int) -> torch.Tensor:
    mat = _ictd_to_e3_basis_cpu(int(l)).to(device=entry.device, dtype=entry.dtype)
    return torch.matmul(entry, mat)


def test_o3_tp_respects_parity_multiplication_rule() -> None:
    active, x1 = _make_o3_blocks()
    _, x2 = _make_o3_blocks()
    x1[(1, 1)][..., 0] = 1.0
    x2[(1, -1)][..., 0] = 1.0

    tp = HarmonicFullyConnectedTensorProductO3(
        mul_in1=1,
        mul_in2=1,
        mul_out=1,
        lmax=2,
        active_irreps=active,
        internal_weights=True,
    )
    weights = torch.ones(x1[(0, 1)].shape[0], tp.num_paths, dtype=torch.float64)
    out = tp(x1, x2, weights)

    nonzero_even = sum(v.abs().sum().item() for (l, p), v in out.items() if p == 1)
    nonzero_odd = sum(v.abs().sum().item() for (l, p), v in out.items() if p == -1)
    assert nonzero_odd > 0.0
    assert nonzero_even == 0.0


def test_canonical_o3_scalar_rhs_multiplicity_scale_matches_e3nn() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    batch = 4
    hidden_mul = 8
    attr_mul = 2
    lmax = 3
    hidden_irreps = o3.Irreps("8x0e + 8x1o + 8x2e + 8x3o")
    attr_irreps = o3.Irreps("2x0e")
    active_irreps = [(l, 1 if l % 2 == 0 else -1) for l in range(lmax + 1)]
    allowed_paths = [(l, p, 0, 1, l, p) for l, p in active_irreps]

    tp_ref = o3.FullyConnectedTensorProduct(
        hidden_irreps,
        attr_irreps,
        hidden_irreps,
        shared_weights=True,
        internal_weights=True,
    ).to(dtype=dtype)
    tp_ictd = HarmonicFullyConnectedTensorProductO3(
        mul_in1=hidden_mul,
        mul_in2=attr_mul,
        mul_out=hidden_mul,
        lmax=lmax,
        active_irreps=active_irreps,
        internal_weights=True,
        allowed_paths=allowed_paths,
    ).to(dtype=dtype)
    with torch.no_grad():
        for idx, w_view in enumerate(tp_ref.weight_views()):
            tp_ictd.weight[idx].copy_(w_view.permute(2, 0, 1))

    x1_entries_e3 = []
    x1_ictd = {}
    for mul, ir in hidden_irreps:
        entry = torch.randn(batch, mul, ir.dim, dtype=dtype)
        x1_entries_e3.append(entry)
        x1_ictd[(int(ir.l), int(ir.p))] = _entry_e3_to_ictd(entry, int(ir.l))
    x1_e3 = torch.cat([entry.reshape(batch, -1) for entry in x1_entries_e3], dim=-1)

    x2_entry_e3 = torch.randn(batch, attr_mul, 1, dtype=dtype)
    x2_e3 = x2_entry_e3.reshape(batch, -1)
    x2_ictd = {(0, 1): _entry_e3_to_ictd(x2_entry_e3, 0)}

    y_ref_flat = tp_ref(x1_e3, x2_e3)
    y_ref = {}
    start = 0
    for mul, ir in hidden_irreps:
        width = mul * ir.dim
        y_ref[(int(ir.l), int(ir.p))] = y_ref_flat[:, start : start + width].reshape(batch, mul, ir.dim)
        start += width

    y_ictd = {key: _entry_ictd_to_e3(val, key[0]) for key, val in tp_ictd(x1_ictd, x2_ictd).items()}

    ref_norm_sq = 0.0
    diff_sq = 0.0
    for key, target in y_ref.items():
        ref_norm_sq += float(target.square().sum().item())
        diff_sq += float((y_ictd[key] - target).square().sum().item())
    rel = math.sqrt(diff_sq / ref_norm_sq)

    assert rel < 1.0e-6


def test_o3_model_forward_external_and_physical_heads() -> None:
    dtype = torch.float64
    model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        channel_in2=8,
        embedding_dim=8,
        max_atomvalue=10,
        output_size=4,
        embed_size=[8, 8],
        main_hidden_sizes3=[8, 8],
        num_layers=1,
        num_interaction=2,
        lmax=2,
        external_tensor_rank=1,
        external_tensor_irrep="1e",
        physical_tensor_outputs={
            "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
            "born_effective_charge_per_atom": {"irreps": ["0e", "1e", "2e"], "channels_out": 1, "reduce": "none"},
        },
    ).to(dtype=dtype)

    pos = torch.randn(4, 3, dtype=dtype)
    A = torch.tensor([1, 1, 8, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    edge_src = torch.tensor([0, 0, 1, 2, 3, 1], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 2, 3, 0, 3], dtype=torch.long)
    edge_shifts = torch.zeros(edge_src.shape[0], 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0)
    out_e, phys = model(
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        external_tensor=torch.tensor([0.0, 0.0, 0.1], dtype=dtype),
        return_physical_tensors=True,
    )
    assert out_e.shape == (4, 1)
    assert phys["magnetic_moment"][1].shape == (1, 1, 3)
    assert phys["born_effective_charge_per_atom"][0].shape == (4, 1, 1)
    assert phys["born_effective_charge_per_atom"][1].shape == (4, 1, 3)
    assert phys["born_effective_charge_per_atom"][2].shape == (4, 1, 5)


def test_o3_active_irrep_presets_and_explicit_override() -> None:
    phys = {
        "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
        "born_effective_charge_per_atom": {"irreps": ["0e", "1e", "2e"], "channels_out": 1, "reduce": "none"},
    }
    auto_irreps = resolve_o3_active_irreps(
        lmax=2,
        preset="auto",
        external_tensor_irrep="1e",
        physical_tensor_outputs=phys,
    )
    minimal_irreps = resolve_o3_active_irreps(
        lmax=2,
        preset="minimal",
        external_tensor_irrep="1e",
        physical_tensor_outputs=phys,
    )
    explicit_irreps = resolve_o3_active_irreps(
        lmax=2,
        preset="full",
        explicit_irreps="0e,1e",
        external_tensor_irrep="1e",
        physical_tensor_outputs=phys,
    )

    assert auto_irreps == [(0, 1), (1, 1), (1, -1), (2, 1)]
    assert minimal_irreps == [(0, 1), (1, 1), (2, 1)]
    assert explicit_irreps == [(0, 1), (1, 1), (2, 1)]


def test_o3_model_minimal_preset_tracks_required_irreps() -> None:
    model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        channel_in2=8,
        embedding_dim=8,
        max_atomvalue=10,
        output_size=4,
        embed_size=[8, 8],
        main_hidden_sizes3=[8, 8],
        num_layers=1,
        num_interaction=2,
        lmax=2,
        external_tensor_rank=1,
        external_tensor_irrep="1e",
        physical_tensor_outputs={
            "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
        },
        o3_irrep_preset="minimal",
    ).to(dtype=torch.float64)

    assert model.active_irreps == [(0, 1), (1, 1)]


def test_o3_model_supports_simultaneous_electric_and_magnetic_fields() -> None:
    dtype = torch.float64
    specs = [
        {"name": "external_field", "rank": 1, "irrep": "1o"},
        {"name": "magnetic_field", "rank": 1, "irrep": "1e"},
    ]
    model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        channel_in2=8,
        embedding_dim=8,
        max_atomvalue=10,
        output_size=4,
        embed_size=[8, 8],
        main_hidden_sizes3=[8, 8],
        num_layers=1,
        num_interaction=2,
        lmax=2,
        external_tensor_specs=specs,
        physical_tensor_outputs={
            "dipole": {"irreps": ["1o"], "channels_out": 1, "reduce": "sum"},
            "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
        },
        o3_irrep_preset="auto",
    ).to(dtype=dtype)

    pos = torch.randn(4, 3, dtype=dtype)
    A = torch.tensor([1, 1, 8, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    edge_src = torch.tensor([0, 0, 1, 2, 3, 1], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 2, 3, 0, 3], dtype=torch.long)
    edge_shifts = torch.zeros(edge_src.shape[0], 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0)
    packed = pack_external_tensor_dict(
        {
            "external_field": torch.tensor([0.1, -0.2, 0.3], dtype=dtype),
            "magnetic_field": torch.tensor([-0.3, 0.4, 0.2], dtype=dtype),
        },
        specs,
        dtype=dtype,
    )
    out_e, phys = model(
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        external_tensor=packed,
        return_physical_tensors=True,
    )

    assert packed is not None
    assert packed.shape == (6,)
    assert out_e.shape == (4, 1)
    assert phys["dipole"][1].shape == (1, 1, 3)
    assert phys["magnetic_moment"][1].shape == (1, 1, 3)
