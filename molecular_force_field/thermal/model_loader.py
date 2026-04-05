from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models import (
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    E3_TransformerLayer_multi,
    PureCartesianSparseTransformerLayer,
    PureCartesianSparseTransformerLayerSave,
    PureCartesianTransformerLayer,
    PureCartesianICTDTransformerLayer,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.models.zbl import maybe_wrap_model_with_zbl
from molecular_force_field.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
)


@dataclass
class ModelLoadOptions:
    checkpoint: str
    device: str = "cpu"
    atomic_energy_file: Optional[str] = None
    atomic_energy_keys: Optional[list[int]] = None
    atomic_energy_values: Optional[list[float]] = None
    max_radius: Optional[float] = None
    tensor_product_mode: Optional[str] = None
    dtype: Optional[str] = None
    max_atomvalue: Optional[int] = None
    embedding_dim: Optional[int] = None
    embed_size: Optional[list[int]] = None
    output_size: Optional[int] = None
    lmax: Optional[int] = None
    irreps_output_conv_channels: Optional[int] = None
    function_type: Optional[str] = None
    num_interaction: Optional[int] = None


def _infer_physical_tensor_outputs_from_state_dict(sd: dict) -> dict | None:
    per_name: dict[str, dict[int, int]] = {}
    pattern = re.compile(r"^physical_tensor_heads\.([^.]+)\.(\d+)\.weight$")
    for key, value in sd.items():
        match = pattern.match(key)
        if not match:
            continue
        name = match.group(1)
        l_value = int(match.group(2))
        ch_out = int(value.shape[0]) if hasattr(value, "shape") and len(value.shape) >= 1 else 1
        per_name.setdefault(name, {})[l_value] = ch_out

    if not per_name:
        return None

    outputs = {}
    for name, ch_by_l in per_name.items():
        ls = sorted(ch_by_l.keys())
        outputs[name] = {
            "ls": ls,
            "channels_out": {l_value: ch_by_l[l_value] for l_value in ls},
            "reduce": "sum",
        }
    return outputs


def _resolve_dtype(dtype_value: Optional[object]) -> Optional[torch.dtype]:
    if dtype_value is None:
        return None
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        lowered = dtype_value.strip().lower()
        if lowered in {"float64", "double", "fp64"}:
            return torch.float64
        if lowered in {"float32", "float", "fp32"}:
            return torch.float32
    raise ValueError(f"Unsupported dtype value: {dtype_value!r}")


def _get_resolved_option(explicit_value, metadata: dict, key: str, fallback):
    if explicit_value is not None:
        return explicit_value
    return metadata.get(key, fallback)


def _build_model(
    *,
    tensor_product_mode: str,
    config: ModelConfig,
    device: torch.device,
    num_interaction: int,
    physical_tensor_outputs: dict | None,
    external_tensor_rank: int | None,
    long_range_mode: str,
    long_range_hidden_dim: int,
    long_range_boundary: str,
    long_range_neutralize: bool,
    long_range_filter_hidden_dim: int,
    long_range_kmax: int,
    long_range_mesh_size: int,
    long_range_slab_padding_factor: int,
    long_range_include_k0: bool,
    long_range_source_channels: int,
    long_range_reciprocal_backend: str,
    long_range_energy_partition: str,
    long_range_green_mode: str,
    long_range_assignment: str,
    long_range_mesh_fft_full_ewald: bool,
    feature_spectral_mode: str,
    feature_spectral_bottleneck_dim: int,
    feature_spectral_mesh_size: int,
    feature_spectral_filter_hidden_dim: int,
    feature_spectral_boundary: str,
    feature_spectral_slab_padding_factor: int,
    feature_spectral_neutralize: bool,
    feature_spectral_include_k0: bool,
    feature_spectral_assignment: str,
    feature_spectral_gate_init: float,
    invariant_channels: int,
):
    if tensor_product_mode == "pure-cartesian":
        model = PureCartesianTransformerLayer(
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
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
        )
    elif tensor_product_mode == "pure-cartesian-ictd":
        model = PureCartesianICTDTransformerLayerFull(
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
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            internal_compute_dtype=config.internal_compute_dtype,
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=external_tensor_rank,
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
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
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
        )
    elif tensor_product_mode == "pure-cartesian-ictd-save":
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
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            internal_compute_dtype=config.internal_compute_dtype,
            device=device,
        )
    elif tensor_product_mode in {"pure-cartesian-sparse", "pure-cartesian-sparse-save"}:
        sparse_cls = (
            PureCartesianSparseTransformerLayerSave
            if tensor_product_mode == "pure-cartesian-sparse-save"
            else PureCartesianSparseTransformerLayer
        )
        model = sparse_cls(
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
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            max_rank_other=max_rank_other,
            k_policy=k_policy,
            device=device,
        )
    elif tensor_product_mode == "partial-cartesian":
        model = CartesianTransformerLayer(
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
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
        )
    elif tensor_product_mode == "partial-cartesian-loose":
        model = CartesianTransformerLayerLoose(
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
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
        )
    elif tensor_product_mode == "spherical-save":
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
            num_interaction=num_interaction,
            function_type_main=config.function_type,
            device=device,
        )
    elif tensor_product_mode == "spherical-save-cue":
        try:
            import cuequivariance_torch  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "tensor_product_mode='spherical-save-cue' requires cuEquivariance. "
                "Install via `pip install -e \".[cue]\"` or `pip install -r requirements-cue.txt`."
            ) from exc
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise_cue,
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
            num_interaction=num_interaction,
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
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
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
        )
    else:
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
        )
    return model.to(device)


def load_model_and_calculator(options: ModelLoadOptions):
    if options.atomic_energy_keys is not None or options.atomic_energy_values is not None:
        if options.atomic_energy_keys is None or options.atomic_energy_values is None:
            raise ValueError("Both atomic_energy_keys and atomic_energy_values must be provided together.")
        if len(options.atomic_energy_keys) != len(options.atomic_energy_values):
            raise ValueError("atomic_energy_keys and atomic_energy_values must have the same length.")

    device = torch.device(
        options.device if options.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(options.checkpoint, map_location=device, weights_only=False)
    state_dict_ckpt = checkpoint.get("e3trans_state_dict", {})
    arch_meta = checkpoint.get("model_hyperparameters", {})

    dtype = (
        _resolve_dtype(options.dtype)
        or _resolve_dtype(checkpoint.get("dtype"))
        or _resolve_dtype(arch_meta.get("dtype"))
        or torch.float64
    )
    config = ModelConfig(
        dtype=dtype,
        channel_in=int(_get_resolved_option(None, arch_meta, "channel_in", 64)),
        channel_in2=int(_get_resolved_option(None, arch_meta, "channel_in2", 32)),
        channel_in3=int(_get_resolved_option(None, arch_meta, "channel_in3", 32)),
        channel_in4=int(_get_resolved_option(None, arch_meta, "channel_in4", 32)),
        channel_in5=int(_get_resolved_option(None, arch_meta, "channel_in5", 32)),
        max_atomvalue=int(_get_resolved_option(options.max_atomvalue, arch_meta, "max_atomvalue", 10)),
        embedding_dim=int(_get_resolved_option(options.embedding_dim, arch_meta, "embedding_dim", 16)),
        main_hidden_sizes3=list(_get_resolved_option(None, arch_meta, "main_hidden_sizes3", [64, 32])),
        embed_size=list(_get_resolved_option(options.embed_size, arch_meta, "embed_size", [128, 128, 128])),
        output_size=int(_get_resolved_option(options.output_size, arch_meta, "output_size", 8)),
        irreps_output_conv_channels=_get_resolved_option(
            options.irreps_output_conv_channels, arch_meta, "irreps_output_conv_channels", None
        ),
        lmax=int(_get_resolved_option(options.lmax, arch_meta, "lmax", 2)),
        function_type=str(_get_resolved_option(options.function_type, arch_meta, "function_type", "gaussian")),
        num_layers=int(_get_resolved_option(None, arch_meta, "num_layers", 1)),
        number_of_basis=int(_get_resolved_option(None, arch_meta, "number_of_basis", 8)),
        number_of_basis_main=int(_get_resolved_option(None, arch_meta, "number_of_basis_main", 8)),
        emb_number_main_2=list(_get_resolved_option(None, arch_meta, "emb_number_main_2", [64, 64, 64])),
        max_radius=float(options.max_radius or checkpoint.get("max_radius") or arch_meta.get("max_radius", 5.0)),
        max_radius_main=float(
            _get_resolved_option(
                options.max_radius,
                {"max_radius": checkpoint.get("max_radius") or arch_meta.get("max_radius", 5.0)},
                "max_radius",
                arch_meta.get("max_radius_main", checkpoint.get("max_radius", 5.0)),
            )
        ),
    )

    if options.atomic_energy_keys is not None:
        config.atomic_energy_keys = torch.tensor(options.atomic_energy_keys, dtype=torch.long)
        config.atomic_energy_values = torch.tensor(options.atomic_energy_values, dtype=config.dtype)
    else:
        e0_path = options.atomic_energy_file or "fitted_E0.csv"
        loaded = config.load_atomic_energies_from_file(e0_path)
        if not loaded and options.atomic_energy_file and not os.path.exists(options.atomic_energy_file):
            raise FileNotFoundError(f"Atomic energy file not found: {options.atomic_energy_file}")

    physical_tensor_outputs = checkpoint.get("physical_tensor_outputs")
    if physical_tensor_outputs is None:
        physical_tensor_outputs = _infer_physical_tensor_outputs_from_state_dict(state_dict_ckpt)

    external_tensor_rank = checkpoint.get("external_tensor_rank")
    if external_tensor_rank is None and "e3_conv_emb.external_tensor_scale_by_l" in state_dict_ckpt:
        external_tensor_rank = 1
    long_range_mode = str(_get_resolved_option(None, arch_meta, "long_range_mode", "none"))
    long_range_hidden_dim = int(_get_resolved_option(None, arch_meta, "long_range_hidden_dim", 64))
    long_range_boundary = str(_get_resolved_option(None, arch_meta, "long_range_boundary", "nonperiodic"))
    long_range_neutralize = bool(_get_resolved_option(None, arch_meta, "long_range_neutralize", True))
    long_range_filter_hidden_dim = int(_get_resolved_option(None, arch_meta, "long_range_filter_hidden_dim", 64))
    long_range_kmax = int(_get_resolved_option(None, arch_meta, "long_range_kmax", 2))
    long_range_mesh_size = int(_get_resolved_option(None, arch_meta, "long_range_mesh_size", 16))
    long_range_slab_padding_factor = int(_get_resolved_option(None, arch_meta, "long_range_slab_padding_factor", 2))
    long_range_include_k0 = bool(_get_resolved_option(None, arch_meta, "long_range_include_k0", False))
    long_range_source_channels = int(_get_resolved_option(None, arch_meta, "long_range_source_channels", 1))
    long_range_backend = str(_get_resolved_option(None, arch_meta, "long_range_backend", "dense_pairwise"))
    long_range_reciprocal_backend = str(_get_resolved_option(None, arch_meta, "long_range_reciprocal_backend", "direct_kspace"))
    long_range_energy_partition = str(_get_resolved_option(None, arch_meta, "long_range_energy_partition", "potential"))
    long_range_green_mode = str(_get_resolved_option(None, arch_meta, "long_range_green_mode", "poisson"))
    long_range_assignment = str(_get_resolved_option(None, arch_meta, "long_range_assignment", "cic"))
    long_range_mesh_fft_full_ewald = bool(_get_resolved_option(None, arch_meta, "long_range_mesh_fft_full_ewald", False))
    long_range_theta = float(_get_resolved_option(None, arch_meta, "long_range_theta", 0.5))
    long_range_leaf_size = int(_get_resolved_option(None, arch_meta, "long_range_leaf_size", 32))
    long_range_multipole_order = int(_get_resolved_option(None, arch_meta, "long_range_multipole_order", 0))
    long_range_far_source_dim = int(_get_resolved_option(None, arch_meta, "long_range_far_source_dim", 16))
    long_range_far_num_shells = int(_get_resolved_option(None, arch_meta, "long_range_far_num_shells", 3))
    long_range_far_shell_growth = float(_get_resolved_option(None, arch_meta, "long_range_far_shell_growth", 2.0))
    long_range_far_tail = bool(_get_resolved_option(None, arch_meta, "long_range_far_tail", True))
    long_range_far_tail_bins = int(_get_resolved_option(None, arch_meta, "long_range_far_tail_bins", 2))
    long_range_far_stats = str(_get_resolved_option(None, arch_meta, "long_range_far_stats", "mean,count,mean_r,rms_r"))
    raw_far_max_radius_multiplier = arch_meta.get("long_range_far_max_radius_multiplier")
    if raw_far_max_radius_multiplier is None:
        long_range_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
            long_range_far_num_shells,
            long_range_far_shell_growth,
        )
    else:
        long_range_far_max_radius_multiplier = float(raw_far_max_radius_multiplier)
    long_range_far_source_norm = bool(_get_resolved_option(None, arch_meta, "long_range_far_source_norm", True))
    long_range_far_gate_init = float(_get_resolved_option(None, arch_meta, "long_range_far_gate_init", 0.0))
    feature_spectral_mode = str(_get_resolved_option(None, arch_meta, "feature_spectral_mode", "none"))
    feature_spectral_bottleneck_dim = int(_get_resolved_option(None, arch_meta, "feature_spectral_bottleneck_dim", 8))
    feature_spectral_mesh_size = int(_get_resolved_option(None, arch_meta, "feature_spectral_mesh_size", 16))
    feature_spectral_filter_hidden_dim = int(_get_resolved_option(None, arch_meta, "feature_spectral_filter_hidden_dim", 64))
    feature_spectral_boundary = str(_get_resolved_option(None, arch_meta, "feature_spectral_boundary", "periodic"))
    feature_spectral_slab_padding_factor = int(_get_resolved_option(None, arch_meta, "feature_spectral_slab_padding_factor", 2))
    feature_spectral_neutralize = bool(_get_resolved_option(None, arch_meta, "feature_spectral_neutralize", True))
    feature_spectral_include_k0 = bool(_get_resolved_option(None, arch_meta, "feature_spectral_include_k0", False))
    feature_spectral_assignment = str(_get_resolved_option(None, arch_meta, "feature_spectral_assignment", "cic"))
    feature_spectral_gate_init = float(_get_resolved_option(None, arch_meta, "feature_spectral_gate_init", 0.0))

    tensor_product_mode = options.tensor_product_mode or checkpoint.get("tensor_product_mode", "spherical")
    if checkpoint.get("tensor_product_mode") and options.tensor_product_mode and options.tensor_product_mode != checkpoint.get("tensor_product_mode"):
        logging.info(
            "Using user-specified tensor_product_mode=%s instead of checkpoint mode=%s",
            options.tensor_product_mode,
            checkpoint.get("tensor_product_mode"),
        )

    model = _build_model(
        tensor_product_mode=tensor_product_mode,
        config=config,
        device=device,
        num_interaction=int(_get_resolved_option(options.num_interaction, arch_meta, "num_interaction", 2)),
        physical_tensor_outputs=physical_tensor_outputs,
        external_tensor_rank=external_tensor_rank,
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
        invariant_channels=int(_get_resolved_option(None, arch_meta, "invariant_channels", 32)),
    )

    if tensor_product_mode == "spherical-save-cue":
        load_result = model.load_state_dict(state_dict_ckpt, strict=False)
        cue_auto_buffers = [k for k in load_result.missing_keys if ".graphs." in k and ".graph.c" in k]
        missing_learned = [k for k in load_result.missing_keys if k not in cue_auto_buffers]
        if load_result.unexpected_keys:
            logging.warning(
                "Ignored %d unexpected checkpoint keys for spherical-save-cue",
                len(load_result.unexpected_keys),
            )
        if missing_learned:
            raise RuntimeError(
                "Checkpoint is missing learned spherical-save-cue parameters: "
                f"{missing_learned[:10]}"
            )
    else:
        model.load_state_dict(state_dict_ckpt, strict=True)
    model = maybe_wrap_model_with_zbl(model, arch_meta)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    atomic_energy_dict: Dict[int, float] = {
        int(key): float(value)
        for key, value in zip(config.atomic_energy_keys.tolist(), config.atomic_energy_values.tolist())
    }
    calculator = MyE3NNCalculator(model, atomic_energy_dict, device, float(config.max_radius))
    metadata = {
        "tensor_product_mode": tensor_product_mode,
        "dtype": str(config.dtype).replace("torch.", ""),
        "device": str(device),
        "max_radius": float(config.max_radius),
        "checkpoint": options.checkpoint,
    }
    return model, calculator, atomic_energy_dict, metadata
