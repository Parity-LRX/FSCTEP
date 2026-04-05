from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from molecular_force_field.models import (
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    E3_TransformerLayer_multi,
    PureCartesianICTDTransformerLayer,
    PureCartesianSparseTransformerLayer,
    PureCartesianSparseTransformerLayerSave,
    PureCartesianTransformerLayer,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.utils.config import ModelConfig


def _build_dummy_graph(dtype: torch.dtype = torch.float64):
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.0],
            [0.3, 1.0, 0.4],
            [1.2, 1.1, 0.8],
        ],
        dtype=dtype,
    )
    A = torch.tensor([1, 6, 8, 1], dtype=torch.long)
    batch = torch.zeros(pos.size(0), dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 12.0
    edge_src = []
    edge_dst = []
    for i in range(pos.size(0)):
        for j in range(pos.size(0)):
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
    edge_src = torch.tensor(edge_src, dtype=torch.long)
    edge_dst = torch.tensor(edge_dst, dtype=torch.long)
    edge_shifts = torch.zeros((edge_src.numel(), 3), dtype=dtype)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _common_long_range_kwargs() -> dict:
    return dict(
        long_range_mode="latent-coulomb",
        long_range_hidden_dim=16,
        long_range_boundary="nonperiodic",
        long_range_neutralize=True,
        long_range_filter_hidden_dim=16,
        long_range_kmax=1,
        long_range_mesh_size=8,
        long_range_slab_padding_factor=2,
        long_range_include_k0=False,
        long_range_source_channels=1,
        long_range_backend="dense_pairwise",
        long_range_reciprocal_backend="direct_kspace",
        long_range_energy_partition="potential",
        long_range_green_mode="poisson",
        long_range_assignment="cic",
        long_range_theta=0.5,
        long_range_leaf_size=16,
        long_range_multipole_order=0,
        long_range_far_source_dim=8,
        long_range_far_num_shells=2,
        long_range_far_shell_growth=2.0,
        long_range_far_tail=True,
        long_range_far_tail_bins=2,
        long_range_far_stats="mean,count,mean_r,rms_r",
        long_range_far_max_radius_multiplier=2.0,
        long_range_far_source_norm=True,
        long_range_far_gate_init=0.0,
        feature_spectral_mode="none",
        feature_spectral_bottleneck_dim=4,
        feature_spectral_mesh_size=8,
        feature_spectral_filter_hidden_dim=16,
        feature_spectral_boundary="periodic",
        feature_spectral_slab_padding_factor=2,
        feature_spectral_neutralize=True,
        feature_spectral_include_k0=False,
        feature_spectral_gate_init=0.0,
    )


def _build_config() -> ModelConfig:
    cfg = ModelConfig(dtype=torch.float64)
    cfg.atomic_energy_keys = torch.tensor([1, 6, 8], dtype=torch.long)
    cfg.atomic_energy_values = torch.tensor([-13.6, -37.8, -75.0], dtype=torch.float64)
    return cfg


def _build_model(mode: str, cfg: ModelConfig) -> torch.nn.Module:
    long_range_kwargs = _common_long_range_kwargs()
    spherical_kwargs = dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        irreps_input=cfg.get_irreps_output_conv(),
        irreps_query=cfg.get_irreps_query_main(),
        irreps_key=cfg.get_irreps_key_main(),
        irreps_value=cfg.get_irreps_value_main(),
        irreps_output=cfg.get_irreps_output_conv_2(),
        irreps_sh=cfg.get_irreps_sh_transformer(),
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=2,
        function_type_main=cfg.function_type,
        device=torch.device("cpu"),
        **long_range_kwargs,
    )
    cartesian_kwargs = dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=cfg.channel_in,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=2,
        function_type_main=cfg.function_type,
        lmax=cfg.lmax,
        device=torch.device("cpu"),
        **long_range_kwargs,
    )
    if mode == "spherical":
        return E3_TransformerLayer_multi(**spherical_kwargs).to(dtype=cfg.dtype)
    if mode == "spherical-save":
        return E3_TransformerLayer_multi_channelwise(**spherical_kwargs).to(dtype=cfg.dtype)
    if mode == "spherical-save-cue":
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue,
        )
        return E3_TransformerLayer_multi_cue(**spherical_kwargs).to(dtype=cfg.dtype)
    if mode == "pure-cartesian":
        return PureCartesianTransformerLayer(**cartesian_kwargs).to(dtype=cfg.dtype)
    if mode == "pure-cartesian-sparse":
        return PureCartesianSparseTransformerLayer(max_rank_other=1, k_policy="k0", **cartesian_kwargs).to(dtype=cfg.dtype)
    if mode == "pure-cartesian-sparse-save":
        return PureCartesianSparseTransformerLayerSave(max_rank_other=1, k_policy="k0", **cartesian_kwargs).to(dtype=cfg.dtype)
    if mode == "partial-cartesian":
        return CartesianTransformerLayer(**cartesian_kwargs).to(dtype=cfg.dtype)
    if mode == "partial-cartesian-loose":
        return CartesianTransformerLayerLoose(**cartesian_kwargs).to(dtype=cfg.dtype)
    if mode == "pure-cartesian-ictd":
        return PureCartesianICTDTransformerLayerFull(
            internal_compute_dtype=cfg.internal_compute_dtype,
            **cartesian_kwargs,
        ).to(dtype=cfg.dtype)
    if mode == "pure-cartesian-ictd-save":
        return PureCartesianICTDTransformerLayer(
            internal_compute_dtype=cfg.internal_compute_dtype,
            **cartesian_kwargs,
        ).to(dtype=cfg.dtype)
    raise ValueError(f"Unsupported mode: {mode}")


def _make_checkpoint(path: Path, mode: str, cfg: ModelConfig) -> None:
    model = _build_model(mode, cfg)
    torch.save(
        {
            "tensor_product_mode": mode,
            "dtype": cfg.dtype,
            "max_radius": cfg.max_radius,
            "max_radius_main": cfg.max_radius_main,
            "atomic_energy_keys": cfg.atomic_energy_keys,
            "atomic_energy_values": cfg.atomic_energy_values,
            "model_hyperparameters": {
                "channel_in": cfg.channel_in,
                "channel_in2": cfg.channel_in2,
                "channel_in3": cfg.channel_in3,
                "channel_in4": cfg.channel_in4,
                "channel_in5": cfg.channel_in5,
                "max_atomvalue": cfg.max_atomvalue,
                "embedding_dim": cfg.embedding_dim,
                "main_hidden_sizes3": cfg.main_hidden_sizes3,
                "embed_size": cfg.embed_size,
                "output_size": cfg.output_size,
                "lmax": cfg.lmax,
                "function_type": cfg.function_type,
                "num_layers": cfg.num_layers,
                "number_of_basis": cfg.number_of_basis,
                "number_of_basis_main": cfg.number_of_basis_main,
                "emb_number_main_2": cfg.emb_number_main_2,
                "max_radius_main": cfg.max_radius_main,
                "num_interaction": 2,
                **_common_long_range_kwargs(),
            },
            "e3trans_state_dict": model.state_dict(),
        },
        path,
    )


ALL_MODES = [
    "spherical",
    "spherical-save",
    "pure-cartesian",
    "pure-cartesian-sparse",
    "pure-cartesian-sparse-save",
    "partial-cartesian",
    "partial-cartesian-loose",
    "pure-cartesian-ictd",
    "pure-cartesian-ictd-save",
]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_long_range_all_modes_train_and_infer_smoke(mode: str) -> None:
    cfg = _build_config()
    model = _build_model(mode, cfg)
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _build_dummy_graph(cfg.dtype)

    model.train()
    pos_train = pos.clone().requires_grad_(True)
    atom_energy = model(pos_train, A, batch, edge_src, edge_dst, edge_shifts, cell)
    loss = atom_energy.sum()
    loss.backward()
    assert torch.isfinite(atom_energy).all()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)

    model.eval()
    with torch.no_grad():
        atom_energy_eval = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    assert atom_energy_eval.shape == (pos.size(0), 1)
    assert torch.isfinite(atom_energy_eval).all()


@pytest.mark.parametrize(
    "mode",
    ["spherical", "spherical-save", "pure-cartesian-ictd", "pure-cartesian-ictd-save"],
)
def test_long_range_export_mliap_supported_modes(mode: str) -> None:
    cfg = _build_config()
    with tempfile.TemporaryDirectory(prefix=f"long-range-{mode}-") as td:
        ckpt_path = Path(td) / f"{mode}.pth"
        out_path = Path(td) / f"{mode}-mliap.pt"
        _make_checkpoint(ckpt_path, mode, cfg)
        mliap_obj = LAMMPS_MLIAP_MFF.from_checkpoint(
            checkpoint_path=str(ckpt_path),
            element_types=["H", "C", "O"],
            device="cpu",
            tensor_product_mode=mode,
        )
        torch.save(mliap_obj, out_path)
        reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
        assert isinstance(reloaded, LAMMPS_MLIAP_MFF)


def test_long_range_spherical_save_cue_smoke_if_available() -> None:
    try:
        cfg = _build_config()
        model = _build_model("spherical-save-cue", cfg)
    except ImportError:
        pytest.skip("cuEquivariance is not available in this environment")
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _build_dummy_graph(cfg.dtype)
    with torch.no_grad():
        atom_energy = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    assert atom_energy.shape == (pos.size(0), 1)
