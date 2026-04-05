"""Offline self-test for the LAMMPS Python potential wrapper.

This does NOT require LAMMPS. It validates the core contract:
- The wrapper can load a checkpoint and compute (E, F)
- Forces are consistent with energy gradients via finite-difference check

Run:
    python -m molecular_force_field.test.self_test_lammps_potential
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass

import numpy as np
import torch

from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
)
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.interfaces.lammps_potential import LAMMPSPotential


def _make_dummy_checkpoint_spherical_save(path: str, device: torch.device) -> ModelConfig:
    """创建 spherical-save (e3nn_layers_channelwise) 的 dummy checkpoint。"""
    from molecular_force_field.models.e3nn_layers_channelwise import (
        E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
    )

    config = ModelConfig(dtype=torch.float64)
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    model = E3_TransformerLayer_multi_channelwise(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        irreps_input=config.get_irreps_output_conv(),
        irreps_query=config.get_irreps_query_main(),
        irreps_key=config.get_irreps_key_main(),
        irreps_value=config.get_irreps_value_main(),
        irreps_output=config.get_irreps_output_conv_2(),
        irreps_sh=config.get_irreps_sh_transformer(),
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        device=device,
    ).to(device)

    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
        "tensor_product_mode": "spherical-save",
    }
    torch.save(ckpt, path)
    return config


def _make_dummy_checkpoint_spherical_save_cue(
    path: str,
    device: torch.device,
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
) -> ModelConfig:
    """创建 spherical-save-cue (cue_layers_channelwise, cuEquivariance) 的 dummy checkpoint。"""
    try:
        import cuequivariance_torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "spherical-save-cue requires cuEquivariance. "
            "Install: pip install cuequivariance-torch cuequivariance-ops-torch-cu12"
        ) from e
    from molecular_force_field.models.cue_layers_channelwise import (
        E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise_cue,
    )

    config = ModelConfig(dtype=torch.float64)
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    if long_range_far_max_radius_multiplier is None:
        long_range_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
            long_range_far_num_shells,
            long_range_far_shell_growth,
        )

    model = E3_TransformerLayer_multi_channelwise_cue(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        irreps_input=config.get_irreps_output_conv(),
        irreps_query=config.get_irreps_query_main(),
        irreps_key=config.get_irreps_key_main(),
        irreps_value=config.get_irreps_value_main(),
        irreps_output=config.get_irreps_output_conv_2(),
        irreps_sh=config.get_irreps_sh_transformer(),
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        device=device,
        long_range_mode=long_range_mode,
        long_range_hidden_dim=long_range_hidden_dim,
        long_range_boundary=long_range_boundary,
        long_range_neutralize=long_range_neutralize,
        long_range_filter_hidden_dim=long_range_filter_hidden_dim,
        long_range_kmax=long_range_kmax,
        long_range_mesh_size=long_range_mesh_size,
        long_range_slab_padding_factor=long_range_slab_padding_factor,
        long_range_include_k0=long_range_include_k0,
        long_range_source_channels=long_range_source_channels,
        long_range_backend=long_range_backend,
        long_range_reciprocal_backend=long_range_reciprocal_backend,
        long_range_energy_partition=long_range_energy_partition,
        long_range_green_mode=long_range_green_mode,
        long_range_assignment=long_range_assignment,
        long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
        long_range_theta=long_range_theta,
        long_range_leaf_size=long_range_leaf_size,
        long_range_multipole_order=long_range_multipole_order,
        long_range_far_source_dim=long_range_far_source_dim,
        long_range_far_num_shells=long_range_far_num_shells,
        long_range_far_shell_growth=long_range_far_shell_growth,
        long_range_far_tail=long_range_far_tail,
        long_range_far_tail_bins=long_range_far_tail_bins,
        long_range_far_stats=long_range_far_stats,
        long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
        long_range_far_source_norm=long_range_far_source_norm,
        long_range_far_gate_init=long_range_far_gate_init,
        feature_spectral_mode=feature_spectral_mode,
        feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
        feature_spectral_mesh_size=feature_spectral_mesh_size,
        feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
        feature_spectral_boundary=feature_spectral_boundary,
        feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
        feature_spectral_neutralize=feature_spectral_neutralize,
        feature_spectral_include_k0=feature_spectral_include_k0,
        feature_spectral_assignment=feature_spectral_assignment,
        feature_spectral_gate_init=feature_spectral_gate_init,
    ).to(device)

    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
        "tensor_product_mode": "spherical-save-cue",
    }
    _write_long_range_hparams(
        ckpt,
        long_range_mode=long_range_mode,
        long_range_hidden_dim=long_range_hidden_dim,
        long_range_boundary=long_range_boundary,
        long_range_neutralize=long_range_neutralize,
        long_range_filter_hidden_dim=long_range_filter_hidden_dim,
        long_range_kmax=long_range_kmax,
        long_range_mesh_size=long_range_mesh_size,
        long_range_slab_padding_factor=long_range_slab_padding_factor,
        long_range_include_k0=long_range_include_k0,
        long_range_source_channels=long_range_source_channels,
        long_range_backend=long_range_backend,
        long_range_reciprocal_backend=long_range_reciprocal_backend,
        long_range_energy_partition=long_range_energy_partition,
        long_range_green_mode=long_range_green_mode,
        long_range_assignment=long_range_assignment,
        long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
        long_range_theta=long_range_theta,
        long_range_leaf_size=long_range_leaf_size,
        long_range_multipole_order=long_range_multipole_order,
        long_range_far_source_dim=long_range_far_source_dim,
        long_range_far_num_shells=long_range_far_num_shells,
        long_range_far_shell_growth=long_range_far_shell_growth,
        long_range_far_tail=long_range_far_tail,
        long_range_far_tail_bins=long_range_far_tail_bins,
        long_range_far_stats=long_range_far_stats,
        long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
        long_range_far_source_norm=long_range_far_source_norm,
        long_range_far_gate_init=long_range_far_gate_init,
    )
    _write_feature_spectral_hparams(
        ckpt,
        feature_spectral_mode=feature_spectral_mode,
        feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
        feature_spectral_mesh_size=feature_spectral_mesh_size,
        feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
        feature_spectral_boundary=feature_spectral_boundary,
        feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
        feature_spectral_neutralize=feature_spectral_neutralize,
        feature_spectral_include_k0=feature_spectral_include_k0,
        feature_spectral_assignment=feature_spectral_assignment,
        feature_spectral_gate_init=feature_spectral_gate_init,
    )
    torch.save(ckpt, path)
    return config


def _write_long_range_hparams(
    ckpt: dict,
    *,
    long_range_mode: str,
    long_range_hidden_dim: int,
    long_range_boundary: str,
    long_range_neutralize: bool,
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
) -> None:
    if long_range_mode == "none":
        return
    ckpt["model_hyperparameters"] = ckpt.get("model_hyperparameters", {})
    ckpt["model_hyperparameters"]["long_range_mode"] = long_range_mode
    ckpt["model_hyperparameters"]["long_range_hidden_dim"] = int(long_range_hidden_dim)
    ckpt["model_hyperparameters"]["long_range_boundary"] = str(long_range_boundary)
    ckpt["model_hyperparameters"]["long_range_neutralize"] = bool(long_range_neutralize)
    ckpt["model_hyperparameters"]["long_range_filter_hidden_dim"] = int(long_range_filter_hidden_dim)
    ckpt["model_hyperparameters"]["long_range_kmax"] = int(long_range_kmax)
    ckpt["model_hyperparameters"]["long_range_mesh_size"] = int(long_range_mesh_size)
    ckpt["model_hyperparameters"]["long_range_slab_padding_factor"] = int(long_range_slab_padding_factor)
    ckpt["model_hyperparameters"]["long_range_include_k0"] = bool(long_range_include_k0)
    ckpt["model_hyperparameters"]["long_range_source_channels"] = int(long_range_source_channels)
    ckpt["model_hyperparameters"]["long_range_backend"] = str(long_range_backend)
    ckpt["model_hyperparameters"]["long_range_reciprocal_backend"] = str(long_range_reciprocal_backend)
    ckpt["model_hyperparameters"]["long_range_energy_partition"] = str(long_range_energy_partition)
    ckpt["model_hyperparameters"]["long_range_green_mode"] = str(long_range_green_mode)
    ckpt["model_hyperparameters"]["long_range_assignment"] = str(long_range_assignment)
    ckpt["model_hyperparameters"]["long_range_mesh_fft_full_ewald"] = bool(long_range_mesh_fft_full_ewald)
    ckpt["model_hyperparameters"]["long_range_theta"] = float(long_range_theta)
    ckpt["model_hyperparameters"]["long_range_leaf_size"] = int(long_range_leaf_size)
    ckpt["model_hyperparameters"]["long_range_multipole_order"] = int(long_range_multipole_order)
    ckpt["model_hyperparameters"]["long_range_far_source_dim"] = int(long_range_far_source_dim)
    ckpt["model_hyperparameters"]["long_range_far_num_shells"] = int(long_range_far_num_shells)
    ckpt["model_hyperparameters"]["long_range_far_shell_growth"] = float(long_range_far_shell_growth)
    ckpt["model_hyperparameters"]["long_range_far_tail"] = bool(long_range_far_tail)
    ckpt["model_hyperparameters"]["long_range_far_tail_bins"] = int(long_range_far_tail_bins)
    ckpt["model_hyperparameters"]["long_range_far_stats"] = str(long_range_far_stats)
    if long_range_far_max_radius_multiplier is None:
        long_range_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
            long_range_far_num_shells,
            long_range_far_shell_growth,
        )
    ckpt["model_hyperparameters"]["long_range_far_max_radius_multiplier"] = float(long_range_far_max_radius_multiplier)
    ckpt["model_hyperparameters"]["long_range_far_source_norm"] = bool(long_range_far_source_norm)
    ckpt["model_hyperparameters"]["long_range_far_gate_init"] = float(long_range_far_gate_init)


def _write_feature_spectral_hparams(
    ckpt: dict,
    *,
    feature_spectral_mode: str,
    feature_spectral_bottleneck_dim: int = 8,
    feature_spectral_mesh_size: int = 16,
    feature_spectral_filter_hidden_dim: int = 64,
    feature_spectral_boundary: str = "periodic",
    feature_spectral_slab_padding_factor: int = 2,
    feature_spectral_neutralize: bool = True,
    feature_spectral_include_k0: bool = False,
    feature_spectral_assignment: str = "cic",
    feature_spectral_gate_init: float = 0.0,
) -> None:
    if feature_spectral_mode == "none":
        return
    ckpt["model_hyperparameters"] = ckpt.get("model_hyperparameters", {})
    ckpt["model_hyperparameters"]["feature_spectral_mode"] = str(feature_spectral_mode)
    ckpt["model_hyperparameters"]["feature_spectral_bottleneck_dim"] = int(feature_spectral_bottleneck_dim)
    ckpt["model_hyperparameters"]["feature_spectral_mesh_size"] = int(feature_spectral_mesh_size)
    ckpt["model_hyperparameters"]["feature_spectral_filter_hidden_dim"] = int(feature_spectral_filter_hidden_dim)
    ckpt["model_hyperparameters"]["feature_spectral_boundary"] = str(feature_spectral_boundary)
    ckpt["model_hyperparameters"]["feature_spectral_slab_padding_factor"] = int(feature_spectral_slab_padding_factor)
    ckpt["model_hyperparameters"]["feature_spectral_neutralize"] = bool(feature_spectral_neutralize)
    ckpt["model_hyperparameters"]["feature_spectral_include_k0"] = bool(feature_spectral_include_k0)
    ckpt["model_hyperparameters"]["feature_spectral_assignment"] = str(feature_spectral_assignment)
    ckpt["model_hyperparameters"]["feature_spectral_gate_init"] = float(feature_spectral_gate_init)


def _make_dummy_checkpoint_pure_cartesian_ictd(
    path: str,
    device: torch.device,
    external_tensor_rank: int | None = None,
    external_tensor_irrep: str | None = None,
    external_tensor_specs: list[dict] | None = None,
    physical_tensor_outputs: dict[str, dict] | None = None,
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
    num_fidelity_levels: int = 0,
    multi_fidelity_mode: str = "conditioning",
) -> ModelConfig:
    """创建 pure-cartesian-ictd (pure_cartesian_ictd_layers_full) 的 dummy checkpoint。"""
    from molecular_force_field.models.pure_cartesian_ictd_layers_full import PureCartesianICTDTransformerLayer

    config = ModelConfig(dtype=torch.float64)
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    if long_range_far_max_radius_multiplier is None:
        long_range_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
            long_range_far_num_shells,
            long_range_far_shell_growth,
        )

    model = PureCartesianICTDTransformerLayer(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        hidden_dim_conv=config.channel_in,
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        lmax=config.lmax,
        physical_tensor_outputs=physical_tensor_outputs,
        external_tensor_rank=external_tensor_rank,
        external_tensor_irrep=external_tensor_irrep,
        external_tensor_specs=external_tensor_specs,
        internal_compute_dtype=config.internal_compute_dtype,
        device=device,
        long_range_mode=long_range_mode,
        long_range_hidden_dim=long_range_hidden_dim,
        long_range_boundary=long_range_boundary,
        long_range_neutralize=long_range_neutralize,
        long_range_filter_hidden_dim=long_range_filter_hidden_dim,
        long_range_kmax=long_range_kmax,
        long_range_mesh_size=long_range_mesh_size,
        long_range_slab_padding_factor=long_range_slab_padding_factor,
        long_range_include_k0=long_range_include_k0,
        long_range_source_channels=long_range_source_channels,
        long_range_backend=long_range_backend,
        long_range_reciprocal_backend=long_range_reciprocal_backend,
        long_range_energy_partition=long_range_energy_partition,
        long_range_green_mode=long_range_green_mode,
        long_range_assignment=long_range_assignment,
        long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
        long_range_theta=long_range_theta,
        long_range_leaf_size=long_range_leaf_size,
        long_range_multipole_order=long_range_multipole_order,
        long_range_far_source_dim=long_range_far_source_dim,
        long_range_far_num_shells=long_range_far_num_shells,
        long_range_far_shell_growth=long_range_far_shell_growth,
        long_range_far_tail=long_range_far_tail,
        long_range_far_tail_bins=long_range_far_tail_bins,
        long_range_far_stats=long_range_far_stats,
        long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
        long_range_far_source_norm=long_range_far_source_norm,
        long_range_far_gate_init=long_range_far_gate_init,
        feature_spectral_mode=feature_spectral_mode,
        feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
        feature_spectral_mesh_size=feature_spectral_mesh_size,
        feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
        feature_spectral_boundary=feature_spectral_boundary,
        feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
        feature_spectral_neutralize=feature_spectral_neutralize,
        feature_spectral_include_k0=feature_spectral_include_k0,
        feature_spectral_assignment=feature_spectral_assignment,
        feature_spectral_gate_init=feature_spectral_gate_init,
        num_fidelity_levels=num_fidelity_levels,
        multi_fidelity_mode=multi_fidelity_mode,
    ).to(device)

    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
        "tensor_product_mode": "pure-cartesian-ictd",
    }
    if external_tensor_rank is not None:
        ckpt["external_tensor_rank"] = int(external_tensor_rank)
    if external_tensor_irrep is not None:
        ckpt["external_tensor_irrep"] = str(external_tensor_irrep)
    if external_tensor_specs is not None:
        ckpt["external_tensor_specs"] = external_tensor_specs
    if physical_tensor_outputs is not None:
        ckpt["physical_tensor_outputs"] = physical_tensor_outputs
    ckpt.setdefault("model_hyperparameters", {})
    ckpt["model_hyperparameters"]["num_fidelity_levels"] = int(num_fidelity_levels)
    ckpt["model_hyperparameters"]["multi_fidelity_mode"] = str(multi_fidelity_mode)
    _write_long_range_hparams(
        ckpt,
        long_range_mode=long_range_mode,
        long_range_hidden_dim=long_range_hidden_dim,
        long_range_boundary=long_range_boundary,
        long_range_neutralize=long_range_neutralize,
        long_range_filter_hidden_dim=long_range_filter_hidden_dim,
        long_range_kmax=long_range_kmax,
        long_range_mesh_size=long_range_mesh_size,
        long_range_slab_padding_factor=long_range_slab_padding_factor,
        long_range_include_k0=long_range_include_k0,
        long_range_source_channels=long_range_source_channels,
        long_range_backend=long_range_backend,
        long_range_reciprocal_backend=long_range_reciprocal_backend,
        long_range_energy_partition=long_range_energy_partition,
        long_range_green_mode=long_range_green_mode,
        long_range_assignment=long_range_assignment,
        long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
        long_range_theta=long_range_theta,
        long_range_leaf_size=long_range_leaf_size,
        long_range_multipole_order=long_range_multipole_order,
        long_range_far_source_dim=long_range_far_source_dim,
        long_range_far_num_shells=long_range_far_num_shells,
        long_range_far_shell_growth=long_range_far_shell_growth,
        long_range_far_tail=long_range_far_tail,
        long_range_far_tail_bins=long_range_far_tail_bins,
        long_range_far_stats=long_range_far_stats,
        long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
        long_range_far_source_norm=long_range_far_source_norm,
        long_range_far_gate_init=long_range_far_gate_init,
    )
    _write_feature_spectral_hparams(
        ckpt,
        feature_spectral_mode=feature_spectral_mode,
        feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
        feature_spectral_mesh_size=feature_spectral_mesh_size,
        feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
        feature_spectral_boundary=feature_spectral_boundary,
        feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
        feature_spectral_neutralize=feature_spectral_neutralize,
        feature_spectral_include_k0=feature_spectral_include_k0,
        feature_spectral_assignment=feature_spectral_assignment,
        feature_spectral_gate_init=feature_spectral_gate_init,
    )
    torch.save(ckpt, path)
    return config


def _make_dummy_checkpoint_pure_cartesian_ictd_o3(
    path: str,
    device: torch.device,
    external_tensor_rank: int | None = None,
    external_tensor_irrep: str | None = None,
    external_tensor_specs: list[dict] | None = None,
    physical_tensor_outputs: dict[str, dict] | None = None,
    o3_irrep_preset: str = "auto",
    o3_active_irreps: str | None = None,
) -> ModelConfig:
    """创建 pure-cartesian-ictd-o3 的 dummy checkpoint。"""
    from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import PureCartesianICTDO3TransformerLayer

    config = ModelConfig(dtype=torch.float64)
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        hidden_dim_conv=config.channel_in,
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        lmax=config.lmax,
        physical_tensor_outputs=physical_tensor_outputs,
        external_tensor_rank=external_tensor_rank,
        external_tensor_irrep=external_tensor_irrep,
        external_tensor_specs=external_tensor_specs,
        o3_irrep_preset=o3_irrep_preset,
        o3_active_irreps=o3_active_irreps,
        internal_compute_dtype=config.internal_compute_dtype,
        device=device,
    ).to(device)

    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
        "tensor_product_mode": "pure-cartesian-ictd-o3",
        "o3_irrep_preset": o3_irrep_preset,
        "o3_active_irreps": o3_active_irreps,
    }
    if external_tensor_rank is not None:
        ckpt["external_tensor_rank"] = int(external_tensor_rank)
    if external_tensor_irrep is not None:
        ckpt["external_tensor_irrep"] = str(external_tensor_irrep)
    if external_tensor_specs is not None:
        ckpt["external_tensor_specs"] = external_tensor_specs
    if physical_tensor_outputs is not None:
        ckpt["physical_tensor_outputs"] = physical_tensor_outputs
    torch.save(ckpt, path)
    return config


def _make_dummy_checkpoint_pure_cartesian_ictd_save(path: str, device: torch.device) -> ModelConfig:
    """创建 pure-cartesian-ictd-save (pure_cartesian_ictd_layers) 的 dummy checkpoint。"""
    from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer

    config = ModelConfig(dtype=torch.float64)
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    model = PureCartesianICTDTransformerLayer(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        hidden_dim_conv=config.channel_in,
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        lmax=config.lmax,
        ictd_tp_path_policy="full",
        internal_compute_dtype=config.internal_compute_dtype,
        device=device,
    ).to(device)

    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
        "tensor_product_mode": "pure-cartesian-ictd-save",
    }
    torch.save(ckpt, path)
    return config


@dataclass
class SelfTestResult:
    energy_kcalmol: float
    max_abs_force_kcalmol_per_ang: float
    max_abs_fd_err_kcalmol_per_ang: float


def _make_dummy_checkpoint(path: str, device: torch.device) -> ModelConfig:
    # Keep dtype explicit and stable
    config = ModelConfig(dtype=torch.float64)

    # Provide some atomic energies to avoid relying on fitted_E0.csv
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    model = E3_TransformerLayer_multi(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        irreps_input=config.get_irreps_output_conv(),
        irreps_query=config.get_irreps_query_main(),
        irreps_key=config.get_irreps_key_main(),
        irreps_value=config.get_irreps_value_main(),
        irreps_output=config.get_irreps_output_conv_2(),
        irreps_sh=config.get_irreps_sh_transformer(),
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        function_type_main=config.function_type,
        device=device,
    ).to(device)

    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
    }
    torch.save(ckpt, path)
    return config


def run_self_test(seed: int = 0, eps: float = 1e-4) -> SelfTestResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = f"{td}/dummy_model.pth"
        _ = _make_dummy_checkpoint(ckpt_path, device=device)

        pot = LAMMPSPotential(
            checkpoint_path=ckpt_path,
            device="cpu",
            max_radius=3.0,
            atomic_energy_keys=[1, 8],
            atomic_energy_values=[-13.6, -75.0],
            # Here we intentionally test the common LAMMPS convention:
            # type 1 -> H (Z=1), type 2 -> O (Z=8)
            type_to_Z={1: 1, 2: 8},
        )

        # Build a tiny non-periodic system: H-O-H
        nlocal = 3
        nall = 3
        tag = np.arange(1, nall + 1, dtype=np.int32)
        type_array = np.array([1, 2, 1], dtype=np.int32)
        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
            dtype=np.float64,
        )
        boxlo = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        boxhi = np.array([50.0, 50.0, 50.0], dtype=np.float64)
        pbc = np.array([0, 0, 0], dtype=np.int32)

        E0, F0 = pot.compute(nlocal, nall, tag, type_array, x, boxlo, boxhi, pbc)
        max_abs_force = float(np.max(np.abs(F0)))

        # Finite difference check for a subset of coordinates
        # dE/dx ≈ (E(x+eps) - E(x-eps)) / (2eps), and F = -dE/dx
        coords_to_check = [(0, 0), (1, 1), (2, 2)]  # (atom_idx, dim)
        fd_errs = []
        for i, d in coords_to_check:
            x_p = x.copy()
            x_m = x.copy()
            x_p[i, d] += eps
            x_m[i, d] -= eps

            E_p, _ = pot.compute(nlocal, nall, tag, type_array, x_p, boxlo, boxhi, pbc)
            E_m, _ = pot.compute(nlocal, nall, tag, type_array, x_m, boxlo, boxhi, pbc)

            dE_dx = (E_p - E_m) / (2.0 * eps)  # kcal/mol/Ang
            fd_force = -dE_dx
            fd_errs.append(abs(fd_force - F0[i, d]))

        max_abs_fd_err = float(np.max(fd_errs)) if fd_errs else 0.0

        return SelfTestResult(
            energy_kcalmol=float(E0),
            max_abs_force_kcalmol_per_ang=max_abs_force,
            max_abs_fd_err_kcalmol_per_ang=max_abs_fd_err,
        )


def main():
    try:
        res = run_self_test()
    except ImportError as e:
        # Most common missing deps: torch_cluster / torch_scatter
        raise SystemExit(f"Self-test failed due to missing dependency: {e}")

    print("LAMMPS potential offline self-test:")
    print(f"  Energy (kcal/mol): {res.energy_kcalmol:.6f}")
    print(f"  Max |Force| (kcal/mol/Ang): {res.max_abs_force_kcalmol_per_ang:.6f}")
    print(f"  Max finite-diff error (kcal/mol/Ang): {res.max_abs_fd_err_kcalmol_per_ang:.6e}")

    # Heuristic threshold; random model is noisy, but FD should still broadly match.
    # If this is huge, something is wrong in sign/units/grad path.
    if res.max_abs_fd_err_kcalmol_per_ang > 1e-2:
        raise SystemExit(
            "Self-test FAILED: finite-difference error too large. "
            "This suggests a bug in energy/force sign, unit conversion, or graph construction."
        )

    print("Self-test PASSED.")


if __name__ == "__main__":
    main()
