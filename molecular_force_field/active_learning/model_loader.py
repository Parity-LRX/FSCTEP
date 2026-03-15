"""Load E3NN model from checkpoint for active learning (model deviation, etc.)."""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from molecular_force_field.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
    infer_physical_tensor_outputs_from_state_dict,
)
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.models.zbl import maybe_wrap_model_with_zbl

logger = logging.getLogger(__name__)


def build_e3trans_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    atomic_energy_file: Optional[str] = None,
    tensor_product_mode: Optional[str] = None,
    num_interaction: int = 2,
) -> Tuple[torch.nn.Module, ModelConfig]:
    """
    Build e3trans model from checkpoint and load state_dict.

    Returns:
        (e3trans, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = tensor_product_mode or ckpt.get("tensor_product_mode")
    if mode is None:
        raise ValueError(
            "tensor_product_mode not in checkpoint. Specify via --tensor-product-mode."
        )
    max_radius = float(ckpt.get("max_radius", 5.0))
    arch_meta = ckpt.get("model_hyperparameters", {})

    external_tensor_rank = ckpt.get("external_tensor_rank")
    if external_tensor_rank is None:
        state = ckpt.get("e3trans_state_dict", {})
        if "e3_conv_emb.external_tensor_scale_by_l" in state:
            external_tensor_rank = 1
    physical_tensor_outputs = ckpt.get("physical_tensor_outputs")
    if physical_tensor_outputs is None:
        physical_tensor_outputs = arch_meta.get("physical_tensor_outputs")
    if physical_tensor_outputs is None:
        physical_tensor_outputs = infer_physical_tensor_outputs_from_state_dict(ckpt.get("e3trans_state_dict", {}))

    config = ModelConfig(dtype=torch.float64, max_radius=max_radius)
    if atomic_energy_file:
        if os.path.exists(atomic_energy_file):
            config.load_atomic_energies_from_file(atomic_energy_file)
        else:
            logger.warning(
                "atomic_energy_file=%s not found; falling back to checkpoint/default atomic energies.",
                atomic_energy_file,
            )

    checkpoint_atomic_keys = ckpt.get("atomic_energy_keys")
    checkpoint_atomic_values = ckpt.get("atomic_energy_values")
    if (
        config.atomic_energy_keys is None
        and checkpoint_atomic_keys is not None
        and checkpoint_atomic_values is not None
    ):
        if isinstance(checkpoint_atomic_keys, torch.Tensor):
            checkpoint_atomic_keys = checkpoint_atomic_keys.detach().cpu().to(torch.long)
        else:
            checkpoint_atomic_keys = torch.tensor(checkpoint_atomic_keys, dtype=torch.long)
        if isinstance(checkpoint_atomic_values, torch.Tensor):
            checkpoint_atomic_values = checkpoint_atomic_values.detach().cpu().to(torch.float64)
        else:
            checkpoint_atomic_values = torch.tensor(checkpoint_atomic_values, dtype=torch.float64)
        config.atomic_energy_keys = checkpoint_atomic_keys
        config.atomic_energy_values = checkpoint_atomic_values
        logger.info("Using atomic energies stored in checkpoint.")
    elif config.atomic_energy_keys is None:
        config.atomic_energy_keys = torch.tensor([1, 6, 7, 8], dtype=torch.long)
        config.atomic_energy_values = torch.tensor(
            [-430.53, -821.03, -1488.19, -2044.35], dtype=torch.float64
        )
        logger.warning("Using default atomic energies; pass --atomic-energy-file for fitted E0.")

    cfg = config
    dtype = cfg.dtype
    long_range_mode = str(arch_meta.get("long_range_mode", "none"))
    long_range_hidden_dim = int(arch_meta.get("long_range_hidden_dim", 64))
    long_range_boundary = str(arch_meta.get("long_range_boundary", "nonperiodic"))
    long_range_neutralize = bool(arch_meta.get("long_range_neutralize", True))
    long_range_filter_hidden_dim = int(arch_meta.get("long_range_filter_hidden_dim", 64))
    long_range_kmax = int(arch_meta.get("long_range_kmax", 2))
    long_range_mesh_size = int(arch_meta.get("long_range_mesh_size", 16))
    long_range_slab_padding_factor = int(arch_meta.get("long_range_slab_padding_factor", 2))
    long_range_include_k0 = bool(arch_meta.get("long_range_include_k0", False))
    long_range_source_channels = int(arch_meta.get("long_range_source_channels", 1))
    long_range_backend = str(arch_meta.get("long_range_backend", "dense_pairwise"))
    long_range_reciprocal_backend = str(arch_meta.get("long_range_reciprocal_backend", "direct_kspace"))
    long_range_energy_partition = str(arch_meta.get("long_range_energy_partition", "potential"))
    long_range_green_mode = str(arch_meta.get("long_range_green_mode", "poisson"))
    long_range_assignment = str(arch_meta.get("long_range_assignment", "cic"))
    long_range_theta = float(arch_meta.get("long_range_theta", 0.5))
    long_range_leaf_size = int(arch_meta.get("long_range_leaf_size", 32))
    long_range_multipole_order = int(arch_meta.get("long_range_multipole_order", 0))
    long_range_far_source_dim = int(arch_meta.get("long_range_far_source_dim", 16))
    long_range_far_num_shells = int(arch_meta.get("long_range_far_num_shells", 3))
    long_range_far_shell_growth = float(arch_meta.get("long_range_far_shell_growth", 2.0))
    long_range_far_tail = bool(arch_meta.get("long_range_far_tail", True))
    long_range_far_tail_bins = int(arch_meta.get("long_range_far_tail_bins", 2))
    long_range_far_stats = str(arch_meta.get("long_range_far_stats", "mean,count,mean_r,rms_r"))
    if arch_meta.get("long_range_far_max_radius_multiplier") is None:
        long_range_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
            long_range_far_num_shells,
            long_range_far_shell_growth,
        )
    else:
        long_range_far_max_radius_multiplier = float(arch_meta.get("long_range_far_max_radius_multiplier"))
    long_range_far_source_norm = bool(arch_meta.get("long_range_far_source_norm", True))
    long_range_far_gate_init = float(arch_meta.get("long_range_far_gate_init", 0.0))
    feature_spectral_mode = str(arch_meta.get("feature_spectral_mode", "none"))
    feature_spectral_bottleneck_dim = int(arch_meta.get("feature_spectral_bottleneck_dim", 8))
    feature_spectral_mesh_size = int(arch_meta.get("feature_spectral_mesh_size", 16))
    feature_spectral_filter_hidden_dim = int(arch_meta.get("feature_spectral_filter_hidden_dim", 64))
    feature_spectral_boundary = str(arch_meta.get("feature_spectral_boundary", "periodic"))
    feature_spectral_slab_padding_factor = int(arch_meta.get("feature_spectral_slab_padding_factor", 2))
    feature_spectral_neutralize = bool(arch_meta.get("feature_spectral_neutralize", True))
    feature_spectral_include_k0 = bool(arch_meta.get("feature_spectral_include_k0", False))
    feature_spectral_gate_init = float(arch_meta.get("feature_spectral_gate_init", 0.0))
    common_long_range_kwargs = dict(
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
        feature_spectral_gate_init=feature_spectral_gate_init,
    )
    invariant_channels = int(arch_meta.get("invariant_channels", 32))
    k_spherical = dict(
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
        num_interaction=num_interaction,
        function_type_main=cfg.function_type,
        device=device,
        **common_long_range_kwargs,
    )
    k_cartesian = dict(
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
        num_interaction=num_interaction,
        invariant_channels=invariant_channels,
        function_type_main=cfg.function_type,
        lmax=cfg.lmax,
        device=device,
        **common_long_range_kwargs,
    )

    from molecular_force_field.models import (
        E3_TransformerLayer_multi,
        CartesianTransformerLayer,
        CartesianTransformerLayerLoose,
        PureCartesianTransformerLayer,
        PureCartesianSparseTransformerLayer,
        PureCartesianSparseTransformerLayerSave,
        PureCartesianICTDTransformerLayer,
    )
    from molecular_force_field.models.e3nn_layers_channelwise import (
        E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
    )
    from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
        PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
    )

    if mode == "spherical":
        e3trans = E3_TransformerLayer_multi(**k_spherical, invariant_channels=invariant_channels)
    elif mode == "spherical-save":
        e3trans = E3_TransformerLayer_multi_channelwise(**k_spherical, invariant_channels=invariant_channels)
    elif mode == "spherical-save-cue":
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue,
        )
        e3trans = E3_TransformerLayer_multi_cue(
            **k_spherical,
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
            feature_spectral_gate_init=feature_spectral_gate_init,
        )
    elif mode == "partial-cartesian":
        e3trans = CartesianTransformerLayer(**k_cartesian)
    elif mode == "partial-cartesian-loose":
        e3trans = CartesianTransformerLayerLoose(**k_cartesian)
    elif mode == "pure-cartesian":
        e3trans = PureCartesianTransformerLayer(**k_cartesian)
    elif mode in {"pure-cartesian-sparse", "pure-cartesian-sparse-save"}:
        sparse_cls = (
            PureCartesianSparseTransformerLayerSave
            if mode == "pure-cartesian-sparse-save"
            else PureCartesianSparseTransformerLayer
        )
        e3trans = sparse_cls(
            **k_cartesian,
            max_rank_other=1,
            k_policy="k0",
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=external_tensor_rank,
        )
    elif mode == "pure-cartesian-ictd":
        ictd_kwargs = dict(
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
        if external_tensor_rank is not None:
            ictd_kwargs["external_tensor_rank"] = external_tensor_rank
        ictd_kwargs["long_range_mode"] = long_range_mode
        ictd_kwargs["long_range_hidden_dim"] = long_range_hidden_dim
        ictd_kwargs["long_range_boundary"] = long_range_boundary
        ictd_kwargs["long_range_neutralize"] = long_range_neutralize
        ictd_kwargs["long_range_filter_hidden_dim"] = long_range_filter_hidden_dim
        ictd_kwargs["long_range_kmax"] = long_range_kmax
        ictd_kwargs["long_range_mesh_size"] = long_range_mesh_size
        ictd_kwargs["long_range_slab_padding_factor"] = long_range_slab_padding_factor
        ictd_kwargs["long_range_include_k0"] = long_range_include_k0
        ictd_kwargs["long_range_source_channels"] = long_range_source_channels
        ictd_kwargs["long_range_backend"] = long_range_backend
        ictd_kwargs["long_range_reciprocal_backend"] = long_range_reciprocal_backend
        ictd_kwargs["long_range_energy_partition"] = long_range_energy_partition
        ictd_kwargs["long_range_green_mode"] = long_range_green_mode
        ictd_kwargs["long_range_assignment"] = long_range_assignment
        ictd_kwargs["long_range_theta"] = long_range_theta
        ictd_kwargs["long_range_leaf_size"] = long_range_leaf_size
        ictd_kwargs["long_range_multipole_order"] = long_range_multipole_order
        ictd_kwargs["long_range_far_source_dim"] = long_range_far_source_dim
        ictd_kwargs["long_range_far_num_shells"] = long_range_far_num_shells
        ictd_kwargs["long_range_far_shell_growth"] = long_range_far_shell_growth
        ictd_kwargs["long_range_far_tail"] = long_range_far_tail
        ictd_kwargs["long_range_far_tail_bins"] = long_range_far_tail_bins
        ictd_kwargs["long_range_far_stats"] = long_range_far_stats
        ictd_kwargs["long_range_far_max_radius_multiplier"] = long_range_far_max_radius_multiplier
        ictd_kwargs["long_range_far_source_norm"] = long_range_far_source_norm
        ictd_kwargs["long_range_far_gate_init"] = long_range_far_gate_init
        ictd_kwargs["feature_spectral_mode"] = feature_spectral_mode
        ictd_kwargs["feature_spectral_bottleneck_dim"] = feature_spectral_bottleneck_dim
        ictd_kwargs["feature_spectral_mesh_size"] = feature_spectral_mesh_size
        ictd_kwargs["feature_spectral_filter_hidden_dim"] = feature_spectral_filter_hidden_dim
        ictd_kwargs["feature_spectral_boundary"] = feature_spectral_boundary
        ictd_kwargs["feature_spectral_slab_padding_factor"] = feature_spectral_slab_padding_factor
        ictd_kwargs["feature_spectral_neutralize"] = feature_spectral_neutralize
        ictd_kwargs["feature_spectral_include_k0"] = feature_spectral_include_k0
        ictd_kwargs["feature_spectral_gate_init"] = feature_spectral_gate_init
        e3trans = PureCartesianICTDTransformerLayerFull(
            **k_cartesian,
            physical_tensor_outputs=physical_tensor_outputs,
            **ictd_kwargs,
        )
    elif mode == "pure-cartesian-ictd-save":
        e3trans = PureCartesianICTDTransformerLayer(
            **k_cartesian,
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
    else:
        raise ValueError(
            f"Unsupported tensor_product_mode: {mode}. "
            "Supported: spherical, spherical-save, spherical-save-cue, "
            "partial-cartesian, partial-cartesian-loose, pure-cartesian, "
            "pure-cartesian-sparse, pure-cartesian-sparse-save, pure-cartesian-ictd, pure-cartesian-ictd-save"
        )

    e3trans = e3trans.to(device=device, dtype=dtype)
    state = ckpt.get("e3trans_ema_state_dict") or ckpt["e3trans_state_dict"]
    e3trans.load_state_dict(state, strict=True)
    e3trans = maybe_wrap_model_with_zbl(e3trans, arch_meta)
    e3trans.eval()
    return e3trans, config
