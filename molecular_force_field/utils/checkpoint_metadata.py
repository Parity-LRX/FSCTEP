from __future__ import annotations

import math
import os
import re
from typing import Any, Mapping

import torch


DEFAULT_MODEL_ARCHITECTURE: dict[str, Any] = {
    "dtype": "float64",
    "max_atomvalue": 10,
    "embedding_dim": 16,
    "embed_size": [128, 128, 128],
    "output_size": 8,
    "lmax": 2,
    "irreps_output_conv_channels": None,
    "function_type": "gaussian",
    "tensor_product_mode": "spherical",
    "num_interaction": 2,
    "invariant_channels": 32,
    "max_radius": 5.0,
    "max_rank_other": 1,
    "k_policy": "k0",
    "ictd_tp_path_policy": "full",
    "ictd_tp_max_rank_other": None,
    "long_range_mode": "none",
    "long_range_hidden_dim": 64,
    "long_range_boundary": "nonperiodic",
    "long_range_neutralize": True,
    "long_range_filter_hidden_dim": 64,
    "long_range_kmax": 2,
    "long_range_mesh_size": 16,
    "long_range_slab_padding_factor": 2,
    "long_range_include_k0": False,
    "long_range_source_channels": 1,
    "long_range_backend": "dense_pairwise",
    "long_range_reciprocal_backend": "direct_kspace",
    "long_range_energy_partition": "potential",
    "long_range_green_mode": "poisson",
    "long_range_assignment": "cic",
    "long_range_theta": 0.5,
    "long_range_leaf_size": 32,
    "long_range_multipole_order": 0,
    "long_range_far_source_dim": 16,
    "long_range_far_num_shells": 3,
    "long_range_far_shell_growth": 2.0,
    "long_range_far_tail": True,
    "long_range_far_tail_bins": 2,
    "long_range_far_stats": "mean,count,mean_r,rms_r",
    "long_range_far_max_radius_multiplier": 8.0,
    "long_range_far_source_norm": True,
    "long_range_far_gate_init": 0.0,
    "long_range_num_k": None,
    "feature_spectral_mode": "none",
    "feature_spectral_bottleneck_dim": 8,
    "feature_spectral_mesh_size": 16,
    "feature_spectral_filter_hidden_dim": 64,
    "feature_spectral_boundary": "periodic",
    "feature_spectral_slab_padding_factor": 2,
    "feature_spectral_neutralize": True,
    "feature_spectral_include_k0": False,
    "feature_spectral_gate_init": 0.0,
    "zbl_enabled": False,
    "zbl_inner_cutoff": 0.8,
    "zbl_outer_cutoff": 1.2,
    "zbl_exponent": 0.23,
    "zbl_energy_scale": 1.0,
}


def derive_long_range_far_max_radius_multiplier(
    far_num_shells: int,
    far_shell_growth: float,
) -> float:
    far_num_shells = int(far_num_shells)
    far_shell_growth = float(far_shell_growth)
    if far_num_shells < 1:
        raise ValueError("long_range_far_num_shells must be >= 1")
    if far_shell_growth <= 1.0:
        raise ValueError("long_range_far_shell_growth must be > 1")
    return float(math.pow(far_shell_growth, far_num_shells))


def maybe_load_checkpoint(path: str | None, *, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint if isinstance(checkpoint, dict) else None


def get_arch_metadata(checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}
    arch_meta = checkpoint.get("model_hyperparameters", {})
    return arch_meta if isinstance(arch_meta, Mapping) else {}


def normalize_dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    if value == torch.float64:
        return "float64"
    if value == torch.float32:
        return "float32"
    text = str(value).strip().lower()
    if text in {"torch.float64", "float64", "double"}:
        return "float64"
    if text in {"torch.float32", "float32", "float"}:
        return "float32"
    return text or None


def _resolve_value(
    overrides: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    arch_meta: Mapping[str, Any],
    key: str,
    default: Any,
    *,
    checkpoint_key: str | None = None,
) -> Any:
    if overrides.get(key) is not None:
        return overrides[key]
    if checkpoint_key is not None and checkpoint is not None and checkpoint.get(checkpoint_key) is not None:
        return checkpoint.get(checkpoint_key)
    if arch_meta.get(key) is not None:
        return arch_meta.get(key)
    return default


def infer_physical_tensor_outputs_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    per_name: dict[str, dict[int, int]] = {}
    pat = re.compile(r"^physical_tensor_heads\.([^.]+)\.(\d+)\.weight$")
    for key, value in state_dict.items():
        match = pat.match(key)
        if not match:
            continue
        name = match.group(1)
        l_value = int(match.group(2))
        channels_out = int(value.shape[0]) if hasattr(value, "shape") and len(value.shape) >= 1 else 1
        per_name.setdefault(name, {})[l_value] = channels_out

    if not per_name:
        return None

    outputs: dict[str, dict[str, Any]] = {}
    for name, channels_by_l in per_name.items():
        ls = sorted(channels_by_l.keys())
        outputs[name] = {
            "ls": ls,
            "channels_out": {l_value: channels_by_l[l_value] for l_value in ls},
            "reduce": "sum",
        }
    return outputs


def infer_external_tensor_rank_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    if "e3_conv_emb.external_tensor_scale_by_l" in state_dict:
        return 1
    return None


def resolve_model_architecture(
    checkpoint: Mapping[str, Any] | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    arch_meta = get_arch_metadata(checkpoint)
    resolved: dict[str, Any] = {}

    resolved["dtype"] = (
        normalize_dtype_name(overrides.get("dtype"))
        or normalize_dtype_name(checkpoint.get("dtype") if checkpoint is not None else None)
        or normalize_dtype_name(arch_meta.get("dtype"))
        or DEFAULT_MODEL_ARCHITECTURE["dtype"]
    )

    resolved["max_radius"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "max_radius",
            DEFAULT_MODEL_ARCHITECTURE["max_radius"],
            checkpoint_key="max_radius",
        )
    )
    resolved["max_atomvalue"] = int(_resolve_value(overrides, checkpoint, arch_meta, "max_atomvalue", DEFAULT_MODEL_ARCHITECTURE["max_atomvalue"]))
    resolved["embedding_dim"] = int(_resolve_value(overrides, checkpoint, arch_meta, "embedding_dim", DEFAULT_MODEL_ARCHITECTURE["embedding_dim"]))
    resolved["embed_size"] = list(_resolve_value(overrides, checkpoint, arch_meta, "embed_size", DEFAULT_MODEL_ARCHITECTURE["embed_size"]))
    resolved["output_size"] = int(_resolve_value(overrides, checkpoint, arch_meta, "output_size", DEFAULT_MODEL_ARCHITECTURE["output_size"]))
    resolved["lmax"] = int(_resolve_value(overrides, checkpoint, arch_meta, "lmax", DEFAULT_MODEL_ARCHITECTURE["lmax"]))
    resolved["irreps_output_conv_channels"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "irreps_output_conv_channels",
        DEFAULT_MODEL_ARCHITECTURE["irreps_output_conv_channels"],
    )
    resolved["function_type"] = str(_resolve_value(overrides, checkpoint, arch_meta, "function_type", DEFAULT_MODEL_ARCHITECTURE["function_type"]))
    resolved["tensor_product_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "tensor_product_mode",
            DEFAULT_MODEL_ARCHITECTURE["tensor_product_mode"],
            checkpoint_key="tensor_product_mode",
        )
    )
    resolved["num_interaction"] = int(_resolve_value(overrides, checkpoint, arch_meta, "num_interaction", DEFAULT_MODEL_ARCHITECTURE["num_interaction"]))
    resolved["invariant_channels"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "invariant_channels",
            DEFAULT_MODEL_ARCHITECTURE["invariant_channels"],
        )
    )
    resolved["max_rank_other"] = int(_resolve_value(overrides, checkpoint, arch_meta, "max_rank_other", DEFAULT_MODEL_ARCHITECTURE["max_rank_other"]))
    resolved["k_policy"] = str(_resolve_value(overrides, checkpoint, arch_meta, "k_policy", DEFAULT_MODEL_ARCHITECTURE["k_policy"]))
    resolved["ictd_tp_path_policy"] = str(
        _resolve_value(overrides, checkpoint, arch_meta, "ictd_tp_path_policy", DEFAULT_MODEL_ARCHITECTURE["ictd_tp_path_policy"])
    )
    resolved["ictd_tp_max_rank_other"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "ictd_tp_max_rank_other",
        DEFAULT_MODEL_ARCHITECTURE["ictd_tp_max_rank_other"],
    )
    resolved["long_range_mode"] = str(
        _resolve_value(overrides, checkpoint, arch_meta, "long_range_mode", DEFAULT_MODEL_ARCHITECTURE["long_range_mode"])
    )
    resolved["long_range_hidden_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_hidden_dim",
            DEFAULT_MODEL_ARCHITECTURE["long_range_hidden_dim"],
        )
    )
    resolved["long_range_boundary"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_boundary",
            DEFAULT_MODEL_ARCHITECTURE["long_range_boundary"],
        )
    )
    resolved["long_range_neutralize"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_neutralize",
            DEFAULT_MODEL_ARCHITECTURE["long_range_neutralize"],
        )
    )
    resolved["long_range_filter_hidden_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_filter_hidden_dim",
            DEFAULT_MODEL_ARCHITECTURE["long_range_filter_hidden_dim"],
        )
    )
    resolved["long_range_kmax"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_kmax",
            DEFAULT_MODEL_ARCHITECTURE["long_range_kmax"],
        )
    )
    resolved["long_range_mesh_size"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_mesh_size",
            DEFAULT_MODEL_ARCHITECTURE["long_range_mesh_size"],
        )
    )
    resolved["long_range_slab_padding_factor"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_slab_padding_factor",
            DEFAULT_MODEL_ARCHITECTURE["long_range_slab_padding_factor"],
        )
    )
    resolved["long_range_include_k0"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_include_k0",
            DEFAULT_MODEL_ARCHITECTURE["long_range_include_k0"],
        )
    )
    resolved["long_range_source_channels"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_source_channels",
            DEFAULT_MODEL_ARCHITECTURE["long_range_source_channels"],
        )
    )
    resolved["long_range_backend"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_backend",
            DEFAULT_MODEL_ARCHITECTURE["long_range_backend"],
        )
    )
    resolved["long_range_reciprocal_backend"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_reciprocal_backend",
            DEFAULT_MODEL_ARCHITECTURE["long_range_reciprocal_backend"],
        )
    )
    resolved["long_range_energy_partition"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_energy_partition",
            DEFAULT_MODEL_ARCHITECTURE["long_range_energy_partition"],
        )
    )
    resolved["long_range_green_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_green_mode",
            DEFAULT_MODEL_ARCHITECTURE["long_range_green_mode"],
        )
    )
    resolved["long_range_assignment"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_assignment",
            DEFAULT_MODEL_ARCHITECTURE["long_range_assignment"],
        )
    )
    resolved["long_range_theta"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_theta",
            DEFAULT_MODEL_ARCHITECTURE["long_range_theta"],
        )
    )
    resolved["long_range_leaf_size"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_leaf_size",
            DEFAULT_MODEL_ARCHITECTURE["long_range_leaf_size"],
        )
    )
    resolved["long_range_multipole_order"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_multipole_order",
            DEFAULT_MODEL_ARCHITECTURE["long_range_multipole_order"],
        )
    )
    resolved["long_range_far_source_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_source_dim",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_source_dim"],
        )
    )
    resolved["long_range_far_num_shells"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_num_shells",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_num_shells"],
        )
    )
    resolved["long_range_far_shell_growth"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_shell_growth",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_shell_growth"],
        )
    )
    resolved["long_range_far_tail"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_tail",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_tail"],
        )
    )
    resolved["long_range_far_tail_bins"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_tail_bins",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_tail_bins"],
        )
    )
    resolved["long_range_far_stats"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_stats",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_stats"],
        )
    )
    raw_far_max_radius_multiplier = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "long_range_far_max_radius_multiplier",
        None,
    )
    if raw_far_max_radius_multiplier is None:
        resolved["long_range_far_max_radius_multiplier"] = derive_long_range_far_max_radius_multiplier(
            resolved["long_range_far_num_shells"],
            resolved["long_range_far_shell_growth"],
        )
    else:
        resolved["long_range_far_max_radius_multiplier"] = float(raw_far_max_radius_multiplier)
    resolved["long_range_far_source_norm"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_source_norm",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_source_norm"],
        )
    )
    resolved["long_range_far_gate_init"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_gate_init",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_gate_init"],
        )
    )
    resolved["feature_spectral_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_mode",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_mode"],
        )
    )
    resolved["feature_spectral_bottleneck_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_bottleneck_dim",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_bottleneck_dim"],
        )
    )
    resolved["feature_spectral_mesh_size"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_mesh_size",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_mesh_size"],
        )
    )
    resolved["feature_spectral_filter_hidden_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_filter_hidden_dim",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_filter_hidden_dim"],
        )
    )
    resolved["feature_spectral_boundary"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_boundary",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_boundary"],
        )
    )
    resolved["feature_spectral_slab_padding_factor"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_slab_padding_factor",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_slab_padding_factor"],
        )
    )
    resolved["feature_spectral_neutralize"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_neutralize",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_neutralize"],
        )
    )
    resolved["feature_spectral_include_k0"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_include_k0",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_include_k0"],
        )
    )
    resolved["feature_spectral_gate_init"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_gate_init",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_gate_init"],
        )
    )
    resolved["zbl_enabled"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_enabled",
            DEFAULT_MODEL_ARCHITECTURE["zbl_enabled"],
        )
    )
    resolved["zbl_inner_cutoff"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_inner_cutoff",
            DEFAULT_MODEL_ARCHITECTURE["zbl_inner_cutoff"],
        )
    )
    resolved["zbl_outer_cutoff"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_outer_cutoff",
            DEFAULT_MODEL_ARCHITECTURE["zbl_outer_cutoff"],
        )
    )
    resolved["zbl_exponent"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_exponent",
            DEFAULT_MODEL_ARCHITECTURE["zbl_exponent"],
        )
    )
    resolved["zbl_energy_scale"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_energy_scale",
            DEFAULT_MODEL_ARCHITECTURE["zbl_energy_scale"],
        )
    )
    resolved["long_range_num_k"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "long_range_num_k",
        DEFAULT_MODEL_ARCHITECTURE["long_range_num_k"],
    )

    state_dict = checkpoint.get("e3trans_state_dict", {}) if checkpoint is not None else {}
    physical_tensor_outputs = checkpoint.get("physical_tensor_outputs") if checkpoint is not None else None
    if physical_tensor_outputs is None:
        physical_tensor_outputs = arch_meta.get("physical_tensor_outputs")
    if physical_tensor_outputs is None and state_dict:
        physical_tensor_outputs = infer_physical_tensor_outputs_from_state_dict(state_dict)
    resolved["physical_tensor_outputs"] = physical_tensor_outputs

    external_tensor_rank = checkpoint.get("external_tensor_rank") if checkpoint is not None else None
    if external_tensor_rank is None:
        external_tensor_rank = arch_meta.get("external_tensor_rank")
    if external_tensor_rank is None and state_dict:
        external_tensor_rank = infer_external_tensor_rank_from_state_dict(state_dict)
    resolved["external_tensor_rank"] = external_tensor_rank

    resolved["inference_output_physical_tensors"] = (
        checkpoint.get("inference_output_physical_tensors") if checkpoint is not None else None
    )

    return resolved


def get_checkpoint_atomic_energies(
    checkpoint: Mapping[str, Any] | None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(checkpoint, Mapping):
        return None

    keys = checkpoint.get("atomic_energy_keys")
    values = checkpoint.get("atomic_energy_values")
    if keys is None or values is None:
        return None

    if isinstance(keys, torch.Tensor):
        keys_tensor = keys.detach().cpu().to(dtype=torch.long)
    else:
        keys_tensor = torch.tensor(list(keys), dtype=torch.long)

    if isinstance(values, torch.Tensor):
        values_tensor = values.detach().cpu().to(dtype=dtype)
    else:
        values_tensor = torch.tensor(list(values), dtype=dtype)

    if keys_tensor.numel() != values_tensor.numel():
        return None

    return keys_tensor, values_tensor
