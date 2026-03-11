from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import pandas as pd


VOIGT_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass
class CallawayParameters:
    grain_size_nm: Optional[float] = None
    specularity: float = 0.0
    point_defect_coeff: float = 0.0
    dislocation_coeff: float = 0.0
    interface_coeff: float = 0.0


def _sanitize_array(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=np.float64, copy=True)
    out[~np.isfinite(out)] = 0.0
    return out


def load_phono3py_kappa_hdf5(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        required = ["temperature", "kappa", "mode_kappa", "gamma", "group_velocity", "frequency"]
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(
                f"{path} is missing datasets required for Callaway post-processing: {missing}"
            )

        data = {name: _sanitize_array(handle[name][...]) for name in required}
        if "weight" in handle:
            data["weight"] = _sanitize_array(handle["weight"][...])
        else:
            data["weight"] = np.ones(data["frequency"].shape[0], dtype=np.float64)
    return data


def _boundary_rate_ps(group_velocity: np.ndarray, grain_size_nm: Optional[float], specularity: float) -> np.ndarray:
    if grain_size_nm is None or grain_size_nm <= 0:
        return np.zeros(group_velocity.shape[:-1], dtype=np.float64)
    length_angstrom = grain_size_nm * 10.0
    velocity_norm = np.linalg.norm(group_velocity, axis=-1)
    p = np.clip(specularity, 0.0, 0.999999)
    prefactor = (1.0 - p) / (1.0 + p)
    return prefactor * velocity_norm / length_angstrom


def _point_defect_rate_ps(frequency_thz: np.ndarray, coeff: float) -> np.ndarray:
    if coeff <= 0:
        return np.zeros_like(frequency_thz, dtype=np.float64)
    omega = np.abs(frequency_thz)
    return coeff * omega**4


def _dislocation_rate_ps(frequency_thz: np.ndarray, coeff: float) -> np.ndarray:
    if coeff <= 0:
        return np.zeros_like(frequency_thz, dtype=np.float64)
    omega = np.abs(frequency_thz)
    return coeff * omega**2


def _interface_rate_ps(frequency_thz: np.ndarray, coeff: float) -> np.ndarray:
    if coeff <= 0:
        return np.zeros_like(frequency_thz, dtype=np.float64)
    omega = np.abs(frequency_thz)
    return coeff * omega


def apply_engineering_scattering(
    phono3py_data: Dict[str, np.ndarray],
    params: CallawayParameters,
) -> Dict[str, np.ndarray]:
    temperatures = phono3py_data["temperature"]
    intrinsic_kappa = phono3py_data["kappa"]
    mode_kappa = phono3py_data["mode_kappa"]
    gamma = phono3py_data["gamma"]
    group_velocity = phono3py_data["group_velocity"]
    frequency = phono3py_data["frequency"]
    weight = phono3py_data["weight"]

    if mode_kappa.ndim != 4:
        raise ValueError(
            "Expected mode_kappa to have shape (n_temperature, n_q, n_band, 6); "
            f"got {mode_kappa.shape}"
        )
    if gamma.shape[:3] != mode_kappa.shape[:3]:
        raise ValueError(
            "gamma and mode_kappa shapes are inconsistent: "
            f"{gamma.shape} vs {mode_kappa.shape}"
        )

    # phono3py gamma is in THz, so 1 THz = 1 / ps.
    intrinsic_rate_ps = np.where(gamma > 0.0, 2.0 * gamma, 0.0)
    boundary_rate_ps = _boundary_rate_ps(group_velocity, params.grain_size_nm, params.specularity)
    point_defect_rate_ps = _point_defect_rate_ps(frequency, params.point_defect_coeff)
    dislocation_rate_ps = _dislocation_rate_ps(frequency, params.dislocation_coeff)
    interface_rate_ps = _interface_rate_ps(frequency, params.interface_coeff)
    extrinsic_rate_ps = (
        boundary_rate_ps[None, :, :]
        + point_defect_rate_ps[None, :, :]
        + dislocation_rate_ps[None, :, :]
        + interface_rate_ps[None, :, :]
    )

    total_rate_ps = intrinsic_rate_ps + extrinsic_rate_ps
    scale = np.ones_like(total_rate_ps, dtype=np.float64)
    mask = intrinsic_rate_ps > 0.0
    scale[mask] = intrinsic_rate_ps[mask] / total_rate_ps[mask]
    scaled_mode_kappa = mode_kappa * scale[..., None]

    weight_sum = np.sum(weight)
    if weight_sum <= 0:
        raise ValueError("Invalid q-point weights: sum(weight) must be positive.")
    kappa_engineered = scaled_mode_kappa.sum(axis=2).sum(axis=1) / weight_sum

    isotropic_intrinsic = intrinsic_kappa[:, :3].mean(axis=1)
    isotropic_engineered = kappa_engineered[:, :3].mean(axis=1)

    return {
        "temperature": temperatures,
        "kappa_intrinsic": intrinsic_kappa,
        "kappa_engineered": kappa_engineered,
        "kappa_isotropic_intrinsic": isotropic_intrinsic,
        "kappa_isotropic_engineered": isotropic_engineered,
        "mode_scaling": scale,
        "intrinsic_rate_ps": intrinsic_rate_ps,
        "extrinsic_rate_ps": extrinsic_rate_ps,
        "parameters": asdict(params),
    }


def component_to_index(component: str) -> Optional[int]:
    lowered = component.lower()
    if lowered == "isotropic":
        return None
    if lowered not in VOIGT_COMPONENTS:
        raise ValueError(f"Unsupported component {component!r}. Choices: isotropic, {', '.join(VOIGT_COMPONENTS)}")
    return VOIGT_COMPONENTS.index(lowered)


def extract_component(result: Dict[str, np.ndarray], component: str, engineered: bool = True) -> np.ndarray:
    index = component_to_index(component)
    if index is None:
        key = "kappa_isotropic_engineered" if engineered else "kappa_isotropic_intrinsic"
        return result[key]
    key = "kappa_engineered" if engineered else "kappa_intrinsic"
    return result[key][:, index]


def interpolate_component(result: Dict[str, np.ndarray], component: str, temperatures: np.ndarray, engineered: bool = True) -> np.ndarray:
    values = extract_component(result, component, engineered=engineered)
    source_t = result["temperature"]
    return np.interp(temperatures, source_t, values)


def save_callaway_summary(
    result: Dict[str, np.ndarray],
    output_prefix: str | Path,
    component: str = "xx",
) -> Dict[str, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    engineered_component = extract_component(result, component, engineered=True)
    intrinsic_component = extract_component(result, component, engineered=False)

    df = pd.DataFrame(
        {
            "temperature_K": result["temperature"],
            f"{component}_intrinsic_WmK": intrinsic_component,
            f"{component}_engineered_WmK": engineered_component,
            "isotropic_intrinsic_WmK": result["kappa_isotropic_intrinsic"],
            "isotropic_engineered_WmK": result["kappa_isotropic_engineered"],
        }
    )
    csv_path = output_prefix.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    json_ready = {
        "temperature": result["temperature"].tolist(),
        "kappa_intrinsic": result["kappa_intrinsic"].tolist(),
        "kappa_engineered": result["kappa_engineered"].tolist(),
        "kappa_isotropic_intrinsic": result["kappa_isotropic_intrinsic"].tolist(),
        "kappa_isotropic_engineered": result["kappa_isotropic_engineered"].tolist(),
        "parameters": result["parameters"],
        "component": component,
    }
    json_path = output_prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready, handle, indent=2, ensure_ascii=False)

    return {"csv": csv_path, "json": json_path}
