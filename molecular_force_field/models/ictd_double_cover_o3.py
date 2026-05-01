from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from molecular_force_field.models.ictd_irreps import (
    _harmonic_rotation_generator_cpu_f64,
    canonical_irrep_parity_sign,
)


_AXES = ("x", "y", "z")
_C128 = torch.complex128
_DEVICE_TENSOR_CACHE: Dict[Tuple[str, Tuple[object, ...], str, str], torch.Tensor] = {}


@dataclass(frozen=True)
class DoubleCoverO3Irrep:
    """Double-cover O(3) carrier label `(parent_l, j, parity)`.

    `two_j` stores `2*j`, so half-integer spinor irreps have odd `two_j`.
    `parent_l` keeps the ICTD orbital parent that supplies inversion parity
    bookkeeping and, for spinor carriers, the product space
    `H_l^{ICTD} tensor S_{1/2}` used to construct the total-j subspace.

    The proper-rotation convention matches `ictd_irreps.py`: infinitesimal
    generators satisfy `[J_x, J_y] = -J_z`.
    """

    parent_l: int
    two_j: int
    parity: int | None = None

    def __post_init__(self) -> None:
        l = int(self.parent_l)
        two_j = int(self.two_j)
        if l < 0:
            raise ValueError(f"parent_l must be non-negative, got {self.parent_l}")
        if two_j < 0:
            raise ValueError(f"two_j must be non-negative, got {self.two_j}")
        p = canonical_irrep_parity_sign(l) if self.parity is None else int(self.parity)
        if p not in (-1, 1):
            raise ValueError(f"parity must be +1 or -1, got {self.parity}")

        if two_j % 2 == 0:
            if two_j != 2 * l:
                raise ValueError(
                    "integer orbital-like carriers require two_j == 2*parent_l; "
                    f"got parent_l={l}, two_j={two_j}"
                )
        else:
            allowed = {1} if l == 0 else {2 * l - 1, 2 * l + 1}
            if two_j not in allowed:
                raise ValueError(
                    "spinor carriers require two_j in {2*l-1, 2*l+1} "
                    f"(or two_j=1 for l=0); got parent_l={l}, two_j={two_j}"
                )

        object.__setattr__(self, "parent_l", l)
        object.__setattr__(self, "two_j", two_j)
        object.__setattr__(self, "parity", p)

    @property
    def dim(self) -> int:
        return int(self.two_j) + 1

    @property
    def j(self) -> float:
        return 0.5 * float(self.two_j)

    @property
    def is_spinor(self) -> bool:
        return int(self.two_j) % 2 == 1

    def label(self) -> str:
        p = "e" if int(self.parity) > 0 else "o"
        if int(self.two_j) % 2 == 0:
            j_s = str(int(self.two_j) // 2)
        else:
            j_s = f"{int(self.two_j)}/2"
        return f"(l={int(self.parent_l)},j={j_s},{p})"


def _as_irrep(irrep: DoubleCoverO3Irrep | Tuple[int, int, int]) -> DoubleCoverO3Irrep:
    if isinstance(irrep, DoubleCoverO3Irrep):
        return irrep
    return DoubleCoverO3Irrep(*irrep)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    dev = torch.device("cpu" if device is None else device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested, but torch.cuda.is_available() is False")
    return dev


def _resolve_complex_dtype(dtype: torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.complex128
    if dtype in (torch.complex128, torch.float64):
        return torch.complex128
    if dtype in (torch.complex64, torch.float32):
        return torch.complex64
    raise ValueError(f"dtype must be complex64/complex128 or float32/float64, got {dtype}")


def _to_device_dtype(
    x: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return x.to(device=_resolve_device(device), dtype=_resolve_complex_dtype(dtype))


def _cached_to_device_dtype(
    x: torch.Tensor,
    *,
    namespace: str,
    key: Tuple[object, ...],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move canonical CPU tensors with a small device/dtype cache.

    The cached tensors are intended to be treated as read-only.  This mirrors
    the ICTD-O3 pattern where expensive algebraic objects are built once and
    reused on the active runtime device.
    """
    dev = _resolve_device(device)
    cdtype = _resolve_complex_dtype(dtype)
    cache_key = (str(namespace), tuple(key), str(dev), str(cdtype))
    cached = _DEVICE_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    moved = x.to(device=dev, dtype=cdtype)
    _DEVICE_TENSOR_CACHE[cache_key] = moved
    return moved


def clear_double_cover_o3_device_cache() -> None:
    """Clear runtime device/dtype tensor cache, preserving CPU algebra caches."""
    _DEVICE_TENSOR_CACHE.clear()


def _phase_fix_columns(mat: torch.Tensor, eps: float = 1.0e-14) -> torch.Tensor:
    """Fix each column's largest component to have positive real phase."""
    out = mat.clone()
    for col in range(out.shape[1]):
        v = out[:, col]
        idx = int(v.abs().argmax().item())
        pivot = v[idx]
        if pivot.abs().item() <= eps:
            continue
        phase = pivot / pivot.abs()
        out[:, col] = v / phase
        if out[idx, col].real.item() < 0:
            out[:, col] = -out[:, col]
    return out


def _phase_fix_global(tensor: torch.Tensor, eps: float = 1.0e-14) -> torch.Tensor:
    """Fix a tensor's single global complex phase."""
    out = tensor.clone()
    idx = int(out.abs().argmax().item())
    pivot = out.flatten()[idx]
    if pivot.abs().item() <= eps:
        return out
    phase = pivot / pivot.abs()
    out = out / phase
    if out.flatten()[idx].real.item() < 0:
        out = -out
    return out


def _spin_half_generators_cpu() -> Dict[str, torch.Tensor]:
    """Spin-1/2 generators matching ICTD's `[Jx, Jy] = -Jz` convention."""
    i = torch.tensor(1j, dtype=_C128)
    sx = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=_C128)
    sy = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=_C128)
    sz = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=_C128)
    # +i*sigma/2 gives anti-Hermitian generators with the same sign convention
    # as the ICTD Cartesian orbital generators.
    return {"x": 0.5 * i * sx, "y": 0.5 * i * sy, "z": 0.5 * i * sz}


@lru_cache(maxsize=None)
def _orbital_generators_cpu(parent_l: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l = int(parent_l)
    return tuple(
        _harmonic_rotation_generator_cpu_f64(l, axis).to(dtype=_C128) for axis in _AXES
    )  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _product_spinor_generators_cpu(parent_l: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l = int(parent_l)
    orbital = dict(zip(_AXES, _orbital_generators_cpu(l)))
    spin = _spin_half_generators_cpu()
    d_orb = 2 * l + 1
    i_orb = torch.eye(d_orb, dtype=_C128)
    i_spin = torch.eye(2, dtype=_C128)
    return tuple(
        torch.kron(orbital[axis], i_spin) + torch.kron(i_orb, spin[axis])
        for axis in _AXES
    )  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _carrier_embedding_cpu(irrep: DoubleCoverO3Irrep) -> torch.Tensor:
    """CPU complex128 embedding with orthonormal columns.

    For spinor carriers the embedding maps carrier coordinates into
    `H_l^{ICTD} tensor S_{1/2}`. For integer orbital-like carriers it is the
    identity embedding in the ICTD harmonic carrier.
    """
    ir = _as_irrep(irrep)
    if not ir.is_spinor:
        return torch.eye(ir.dim, dtype=_C128)

    gens = dict(zip(_AXES, _product_spinor_generators_cpu(ir.parent_l)))
    casimir = -sum(g @ g for g in gens.values())
    casimir = 0.5 * (casimir + casimir.conj().T)
    evals, evecs = torch.linalg.eigh(casimir)
    target = ir.j * (ir.j + 1.0)
    mask = (evals.real - target).abs() < 1.0e-8
    if int(mask.sum().item()) != ir.dim:
        raise RuntimeError(
            f"Failed to isolate {ir.label()} Casimir eigenspace: "
            f"target={target}, matches={int(mask.sum().item())}, expected={ir.dim}, "
            f"evals={evals.real.tolist()}"
        )
    sub = evecs[:, mask]

    # Choose a deterministic weight basis inside the degenerate Casimir
    # eigenspace by diagonalizing -i*Jz.
    hz = (-1j) * (sub.conj().T @ gens["z"] @ sub)
    hz = 0.5 * (hz + hz.conj().T)
    _, vz = torch.linalg.eigh(hz)
    emb = sub @ vz
    q, _ = torch.linalg.qr(emb)
    return _phase_fix_columns(q[:, : ir.dim]).contiguous()


@lru_cache(maxsize=None)
def _carrier_generators_cpu(irrep: DoubleCoverO3Irrep) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ir = _as_irrep(irrep)
    if ir.is_spinor:
        emb = _carrier_embedding_cpu(ir)
        prod_gens = _product_spinor_generators_cpu(ir.parent_l)
        gens = tuple((emb.conj().T @ g @ emb).contiguous() for g in prod_gens)
    else:
        gens = _orbital_generators_cpu(ir.parent_l)
    return gens  # type: ignore[return-value]


def orbital_spinor_basis(
    irrep: DoubleCoverO3Irrep | Tuple[int, int, int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.complex128,
) -> torch.Tensor:
    """Return the carrier embedding basis on the requested device.

    The returned matrix has orthonormal columns and maps carrier coordinates
    into the parent product space. For integer orbital-like irreps this is an
    identity matrix in the ICTD harmonic basis.
    """
    ir = _as_irrep(irrep)
    return _cached_to_device_dtype(
        _carrier_embedding_cpu(ir),
        namespace="basis",
        key=(ir,),
        device=device,
        dtype=dtype,
    )


def double_cover_o3_generators(
    irrep: DoubleCoverO3Irrep | Tuple[int, int, int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.complex128,
) -> Dict[str, torch.Tensor]:
    ir = _as_irrep(irrep)
    gens = _carrier_generators_cpu(ir)
    return {
        axis: _cached_to_device_dtype(
            gens[idx],
            namespace="generator",
            key=(ir, axis),
            device=device,
            dtype=dtype,
        )
        for idx, axis in enumerate(_AXES)
    }


def double_cover_o3_rotation(
    irrep: DoubleCoverO3Irrep | Tuple[int, int, int],
    axis: torch.Tensor,
    angle: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.complex128,
) -> torch.Tensor:
    dev = _resolve_device(device if device is not None else axis.device)
    cdtype = _resolve_complex_dtype(dtype)
    exp_dtype = torch.complex128 if cdtype == torch.complex64 else cdtype
    axis_t = axis.to(device=dev, dtype=torch.float64)
    norm = axis_t.norm()
    if float(norm.item()) <= 1.0e-30:
        raise ValueError("axis must be non-zero")
    axis_t = axis_t / norm
    gens = double_cover_o3_generators(irrep, device=dev, dtype=exp_dtype)
    generator = sum(axis_t[i].to(dtype=exp_dtype) * gens[_AXES[i]] for i in range(3))
    return torch.linalg.matrix_exp(float(angle) * generator).to(dtype=cdtype)


def double_cover_o3_inversion(
    irrep: DoubleCoverO3Irrep | Tuple[int, int, int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.complex128,
) -> torch.Tensor:
    ir = _as_irrep(irrep)
    dev = _resolve_device(device)
    cdtype = _resolve_complex_dtype(dtype)
    return int(ir.parity) * torch.eye(ir.dim, device=dev, dtype=cdtype)


def _triangle_allowed(ir1: DoubleCoverO3Irrep, ir2: DoubleCoverO3Irrep, ir3: DoubleCoverO3Irrep) -> bool:
    tj1, tj2, tj3 = int(ir1.two_j), int(ir2.two_j), int(ir3.two_j)
    if abs(tj1 - tj2) > tj3 or tj3 > tj1 + tj2:
        return False
    return (tj1 + tj2 + tj3) % 2 == 0


def _zero_cg_cpu(ir1: DoubleCoverO3Irrep, ir2: DoubleCoverO3Irrep, ir3: DoubleCoverO3Irrep) -> torch.Tensor:
    return torch.zeros(ir1.dim, ir2.dim, ir3.dim, dtype=_C128)


def _normalize_cg(C: torch.Tensor, ir3: DoubleCoverO3Irrep, normalization: str) -> torch.Tensor:
    if normalization not in {"component", "norm", "none"}:
        raise ValueError(
            "normalization must be one of 'component', 'norm', or 'none', "
            f"got {normalization!r}"
        )
    norm = C.norm().item()
    if norm <= 1.0e-30 or normalization == "none":
        return C
    if normalization == "norm":
        return C / norm
    return C * (math.sqrt(float(ir3.dim)) / norm)


@lru_cache(maxsize=None)
def _build_double_cover_o3_cg_cpu(
    ir1: DoubleCoverO3Irrep,
    ir2: DoubleCoverO3Irrep,
    ir3: DoubleCoverO3Irrep,
    normalization: str,
) -> torch.Tensor:
    ir1, ir2, ir3 = _as_irrep(ir1), _as_irrep(ir2), _as_irrep(ir3)
    if int(ir3.parity) != int(ir1.parity) * int(ir2.parity):
        return _zero_cg_cpu(ir1, ir2, ir3)
    if not _triangle_allowed(ir1, ir2, ir3):
        return _zero_cg_cpu(ir1, ir2, ir3)

    gens1 = dict(zip(_AXES, _carrier_generators_cpu(ir1)))
    gens2 = dict(zip(_AXES, _carrier_generators_cpu(ir2)))
    gens3 = dict(zip(_AXES, _carrier_generators_cpu(ir3)))
    d1, d2, d3 = ir1.dim, ir2.dim, ir3.dim
    eye1 = torch.eye(d1, dtype=_C128)
    eye2 = torch.eye(d2, dtype=_C128)
    n_unknown = d3 * d1 * d2
    equations = []
    for axis in _AXES:
        jin = torch.kron(gens1[axis], eye2) + torch.kron(eye1, gens2[axis])
        a = torch.zeros(d3 * d1 * d2, n_unknown, dtype=_C128)
        col = 0
        for q in range(d3):
            for r in range(d1 * d2):
                cmat = torch.zeros(d3, d1 * d2, dtype=_C128)
                cmat[q, r] = 1.0
                a[:, col] = (gens3[axis] @ cmat - cmat @ jin).reshape(-1)
                col += 1
        equations.append(a)
    constraint = torch.cat(equations, dim=0)
    _, s, vh = torch.linalg.svd(constraint)
    cutoff = max(1.0e-10, 1.0e-9 * float(s[0].real.item() if s.numel() else 1.0))
    nullity = int((s <= cutoff).sum().item())
    if nullity != 1:
        raise RuntimeError(
            f"Expected one-dimensional intertwiner nullspace for "
            f"{ir1.label()} x {ir2.label()} -> {ir3.label()}, got nullity={nullity}; "
            f"smallest singular values={s[-min(6, s.numel()):].real.tolist()}"
        )
    # torch.linalg.svd returns Vh.  The right singular vector is therefore the
    # conjugate of the last Vh row; using the row directly is only valid for
    # real-valued nullspace solves.
    cmat = vh[-1].conj().reshape(d3, d1 * d2).contiguous()
    C = cmat.T.reshape(d1, d2, d3).contiguous()
    C = _phase_fix_global(C)
    return _normalize_cg(C, ir3, str(normalization)).contiguous()


def build_double_cover_o3_cg(
    ir1: DoubleCoverO3Irrep | Tuple[int, int, int],
    ir2: DoubleCoverO3Irrep | Tuple[int, int, int],
    ir3: DoubleCoverO3Irrep | Tuple[int, int, int],
    normalization: str = "component",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.complex128,
) -> torch.Tensor:
    ir1 = _as_irrep(ir1)
    ir2 = _as_irrep(ir2)
    ir3 = _as_irrep(ir3)
    C = _build_double_cover_o3_cg_cpu(
        ir1,
        ir2,
        ir3,
        str(normalization),
    )
    return _cached_to_device_dtype(
        C,
        namespace="cg",
        key=(ir1, ir2, ir3, str(normalization)),
        device=device,
        dtype=dtype,
    )


def apply_double_cover_o3_tp(x1: torch.Tensor, x2: torch.Tensor, cg: torch.Tensor) -> torch.Tensor:
    """Apply a double-cover O(3) tensor product with trailing irrep dims."""
    if x1.shape[:-1] != x2.shape[:-1]:
        x1, x2 = torch.broadcast_tensors(
            x1.unsqueeze(-1),
            x2.unsqueeze(-2),
        )
        x1 = x1[..., :, 0]
        x2 = x2[..., 0, :]
    if x1.shape[-1] != cg.shape[0] or x2.shape[-1] != cg.shape[1]:
        raise ValueError(
            f"Input dimensions {x1.shape[-1]}, {x2.shape[-1]} do not match CG shape {tuple(cg.shape)}"
        )
    return torch.einsum("...i,...j,ijk->...k", x1, x2, cg)


def _normalize_irrep_sequence(
    irreps: Sequence[DoubleCoverO3Irrep | Tuple[int, int, int]],
) -> Tuple[DoubleCoverO3Irrep, ...]:
    return tuple(_as_irrep(ir) for ir in irreps)


def enumerate_double_cover_o3_paths(
    irreps_in1: Sequence[DoubleCoverO3Irrep | Tuple[int, int, int]],
    irreps_in2: Sequence[DoubleCoverO3Irrep | Tuple[int, int, int]],
    irreps_out: Sequence[DoubleCoverO3Irrep | Tuple[int, int, int]],
) -> List[Tuple[int, int, int]]:
    """Enumerate parity- and triangle-valid double-cover O(3) TP paths.

    Returns index triples `(i1, i2, i3)` into the three irrep sequences.  The
    expensive CG construction is intentionally not performed here.
    """
    in1 = _normalize_irrep_sequence(irreps_in1)
    in2 = _normalize_irrep_sequence(irreps_in2)
    out = _normalize_irrep_sequence(irreps_out)
    paths: List[Tuple[int, int, int]] = []
    for i1, ir1 in enumerate(in1):
        for i2, ir2 in enumerate(in2):
            target_parity = int(ir1.parity) * int(ir2.parity)
            for i3, ir3 in enumerate(out):
                if int(ir3.parity) != target_parity:
                    continue
                if not _triangle_allowed(ir1, ir2, ir3):
                    continue
                paths.append((i1, i2, i3))
    return paths


class DoubleCoverO3TensorProduct(nn.Module):
    """Cached tensor product module for double-cover O(3) carriers.

    Inputs and outputs are dictionaries keyed by `DoubleCoverO3Irrep`, with
    trailing irrep dimensions `(..., dim)`.  The module builds all valid CG
    tensors once in `__init__`, registers them as buffers, and only performs
    tensor contractions in `forward`.
    """

    def __init__(
        self,
        irreps_in1: Sequence[DoubleCoverO3Irrep | Tuple[int, int, int]],
        irreps_in2: Sequence[DoubleCoverO3Irrep | Tuple[int, int, int]],
        irreps_out: Sequence[DoubleCoverO3Irrep | Tuple[int, int, int]],
        *,
        normalization: str = "component",
        dtype: torch.dtype = torch.complex128,
    ):
        super().__init__()
        self.irreps_in1 = _normalize_irrep_sequence(irreps_in1)
        self.irreps_in2 = _normalize_irrep_sequence(irreps_in2)
        self.irreps_out = _normalize_irrep_sequence(irreps_out)
        self.normalization = str(normalization)
        self.paths = enumerate_double_cover_o3_paths(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
        )
        self.num_paths = len(self.paths)
        cg_by_path: List[torch.Tensor] = []
        for p_idx, (i1, i2, i3) in enumerate(self.paths):
            cg = build_double_cover_o3_cg(
                self.irreps_in1[i1],
                self.irreps_in2[i2],
                self.irreps_out[i3],
                normalization=self.normalization,
                device="cpu",
                dtype=dtype,
            )
            cg_by_path.append(cg)
            self.register_buffer(f"cg_{p_idx}", cg, persistent=False)

        self._groups: List[Dict[str, object]] = []
        groups_tmp: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for p_idx, (i1, i2, _i3) in enumerate(self.paths):
            groups_tmp.setdefault((i1, i2), []).append((p_idx, _i3))
        for group_idx, ((i1, i2), items) in enumerate(sorted(groups_tmp.items())):
            ir1 = self.irreps_in1[i1]
            ir2 = self.irreps_in2[i2]
            k_total = 0
            segments: List[Tuple[int, int, int, int]] = []
            for p_idx, i3 in items:
                dim3 = self.irreps_out[i3].dim
                segments.append((p_idx, i3, k_total, k_total + dim3))
                k_total += dim3
            U = torch.zeros(ir1.dim * ir2.dim, k_total, dtype=cg_by_path[items[0][0]].dtype)
            for p_idx, _i3, start, stop in segments:
                U[:, start:stop] = cg_by_path[p_idx].reshape(ir1.dim * ir2.dim, stop - start)
            self._groups.append(
                {
                    "i1": i1,
                    "i2": i2,
                    "segments": segments,
                    "k_total": k_total,
                }
            )
            self.register_buffer(f"group_U_{group_idx}", U.contiguous(), persistent=False)

    def _buffer_cg(self, path_idx: int) -> torch.Tensor:
        return self.get_buffer(f"cg_{int(path_idx)}")

    def _buffer_group_U(self, group_idx: int) -> torch.Tensor:
        return self.get_buffer(f"group_U_{int(group_idx)}")

    def forward(
        self,
        x1: Dict[DoubleCoverO3Irrep, torch.Tensor],
        x2: Dict[DoubleCoverO3Irrep, torch.Tensor],
        path_weights: torch.Tensor | None = None,
    ) -> Dict[DoubleCoverO3Irrep, torch.Tensor]:
        if not self.paths:
            sample = next(iter(x1.values()), None)
            if sample is None:
                sample = next(iter(x2.values()))
            return {
                ir: sample.new_zeros(*sample.shape[:-1], ir.dim)
                for ir in self.irreps_out
            }
        if path_weights is not None and path_weights.shape[-1] != self.num_paths:
            raise ValueError(
                f"path_weights last dimension must be num_paths={self.num_paths}, "
                f"got {tuple(path_weights.shape)}"
            )

        outputs: Dict[DoubleCoverO3Irrep, torch.Tensor] = {}
        for group_idx, group in enumerate(self._groups):
            i1 = int(group["i1"])
            i2 = int(group["i2"])
            ir1 = self.irreps_in1[i1]
            ir2 = self.irreps_in2[i2]
            if ir1 not in x1 or ir2 not in x2:
                continue
            a = x1[ir1]
            b = x2[ir2].to(device=a.device)
            U = self._buffer_group_U(group_idx)
            if U.device != a.device:
                U = U.to(device=a.device)
            a_comp = a.to(dtype=U.dtype)
            b_comp = b.to(dtype=U.dtype)
            outer = a_comp.unsqueeze(-1) * b_comp.unsqueeze(-2)
            flat = outer.reshape(*outer.shape[:-2], ir1.dim * ir2.dim)
            y = torch.matmul(flat, U)
            for p_idx, i3, start, stop in group["segments"]:  # type: ignore[assignment]
                ir3 = self.irreps_out[int(i3)]
                contrib = y[..., int(start): int(stop)]
                if path_weights is not None:
                    weight = path_weights[..., int(p_idx)].to(device=contrib.device, dtype=contrib.dtype)
                    contrib = contrib * weight.unsqueeze(-1)
                if ir3 in outputs:
                    outputs[ir3] = outputs[ir3] + contrib
                else:
                    outputs[ir3] = contrib

        sample = next(iter(outputs.values()), None)
        if sample is None:
            sample = next(iter(x1.values()))
        for ir in self.irreps_out:
            if ir not in outputs:
                outputs[ir] = sample.new_zeros(*sample.shape[:-1], ir.dim)
        return outputs


def _act(D: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,...j->...i", D, x)


def _relative_error(diff: torch.Tensor, ref: torch.Tensor) -> float:
    denom = max(float(ref.norm().item()), 1.0e-30)
    return float(diff.norm().item()) / denom


def check_double_cover_o3_equivariance(
    ir1: DoubleCoverO3Irrep | Tuple[int, int, int] | None = None,
    ir2: DoubleCoverO3Irrep | Tuple[int, int, int] | None = None,
    ir3: DoubleCoverO3Irrep | Tuple[int, int, int] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.complex128,
    ntrials: int = 8,
    atol: float | None = None,
    seed: int = 0,
) -> Dict[str, float | bool | str]:
    """Check tensor-product equivariance for random axis-angle rotations."""
    dev = _resolve_device(device)
    cdtype = _resolve_complex_dtype(dtype)
    tol = 1.0e-5 if cdtype == torch.complex64 else 1.0e-8
    if atol is not None:
        tol = float(atol)
    ir1 = DoubleCoverO3Irrep(0, 1, 1) if ir1 is None else _as_irrep(ir1)
    ir2 = DoubleCoverO3Irrep(1, 2, -1) if ir2 is None else _as_irrep(ir2)
    ir3 = DoubleCoverO3Irrep(1, 1, -1) if ir3 is None else _as_irrep(ir3)
    cg = build_double_cover_o3_cg(ir1, ir2, ir3, device=dev, dtype=cdtype)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    max_abs = 0.0
    max_rel = 0.0
    for _ in range(int(ntrials)):
        axis = torch.randn(3, generator=gen, dtype=torch.float64)
        angle = float(torch.randn((), generator=gen, dtype=torch.float64).item())
        D1 = double_cover_o3_rotation(ir1, axis, angle, device=dev, dtype=cdtype)
        D2 = double_cover_o3_rotation(ir2, axis, angle, device=dev, dtype=cdtype)
        D3 = double_cover_o3_rotation(ir3, axis, angle, device=dev, dtype=cdtype)
        x1 = torch.randn(ir1.dim, generator=gen, dtype=torch.float64).to(dev, cdtype)
        x2 = torch.randn(ir2.dim, generator=gen, dtype=torch.float64).to(dev, cdtype)
        x1 = x1 + 1j * torch.randn(ir1.dim, generator=gen, dtype=torch.float64).to(dev, cdtype)
        x2 = x2 + 1j * torch.randn(ir2.dim, generator=gen, dtype=torch.float64).to(dev, cdtype)
        lhs = apply_double_cover_o3_tp(_act(D1, x1), _act(D2, x2), cg)
        rhs = _act(D3, apply_double_cover_o3_tp(x1, x2, cg))
        diff = lhs - rhs
        max_abs = max(max_abs, float(diff.abs().max().item()))
        max_rel = max(max_rel, _relative_error(diff, rhs))
    return {
        "passed": bool(max_abs <= tol or max_rel <= tol),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": tol,
        "device": str(dev),
        "dtype": str(cdtype),
        "path": f"{ir1.label()} x {ir2.label()} -> {ir3.label()}",
    }


def _assert_close(name: str, value: torch.Tensor, target: torch.Tensor, atol: float) -> None:
    err = float((value - target).abs().max().item())
    if err > atol:
        raise AssertionError(f"{name} failed: max_abs={err} > atol={atol}")


def _test_lie_algebra_and_casimir(device: torch.device, dtype: torch.dtype) -> None:
    tol = 1.0e-5 if dtype == torch.complex64 else 1.0e-8
    irreps = [DoubleCoverO3Irrep(0, 0, 1), DoubleCoverO3Irrep(0, 1, 1)]
    for l in range(1, 4):
        irreps.extend(
            [
                DoubleCoverO3Irrep(l, 2 * l, canonical_irrep_parity_sign(l)),
                DoubleCoverO3Irrep(l, 2 * l - 1, canonical_irrep_parity_sign(l)),
                DoubleCoverO3Irrep(l, 2 * l + 1, canonical_irrep_parity_sign(l)),
            ]
        )
    for ir in irreps:
        g = double_cover_o3_generators(ir, device=device, dtype=dtype)
        _assert_close(f"{ir.label()} [Jx,Jy]", g["x"] @ g["y"] - g["y"] @ g["x"], -g["z"], tol)
        _assert_close(f"{ir.label()} [Jy,Jz]", g["y"] @ g["z"] - g["z"] @ g["y"], -g["x"], tol)
        _assert_close(f"{ir.label()} [Jz,Jx]", g["z"] @ g["x"] - g["x"] @ g["z"], -g["y"], tol)
        cas = -(g["x"] @ g["x"] + g["y"] @ g["y"] + g["z"] @ g["z"])
        target = (ir.j * (ir.j + 1.0)) * torch.eye(ir.dim, device=device, dtype=dtype)
        _assert_close(f"{ir.label()} Casimir", cas, target, tol)
        for axis in _AXES:
            _assert_close(
                f"{ir.label()} anti-Hermitian {axis}",
                g[axis].conj().T,
                -g[axis],
                tol,
            )


def _test_decomposition() -> None:
    for l in range(4):
        if l == 0:
            b = orbital_spinor_basis(DoubleCoverO3Irrep(0, 1, 1))
            _assert_close("l=0 spinor orthonormal", b.conj().T @ b, torch.eye(2, dtype=_C128), 1.0e-10)
            continue
        low = orbital_spinor_basis(DoubleCoverO3Irrep(l, 2 * l - 1))
        high = orbital_spinor_basis(DoubleCoverO3Irrep(l, 2 * l + 1))
        if low.shape[1] + high.shape[1] != 2 * (2 * l + 1):
            raise AssertionError(f"l={l} spinor dimensions do not exhaust product space")
        _assert_close(f"l={l} low orthonormal", low.conj().T @ low, torch.eye(low.shape[1], dtype=_C128), 1.0e-10)
        _assert_close(f"l={l} high orthonormal", high.conj().T @ high, torch.eye(high.shape[1], dtype=_C128), 1.0e-10)
        _assert_close(
            f"l={l} low/high orthogonal",
            low.conj().T @ high,
            torch.zeros(low.shape[1], high.shape[1], dtype=_C128),
            1.0e-10,
        )


def _test_rotation_inversion(device: torch.device, dtype: torch.dtype) -> None:
    tol = 1.0e-5 if dtype == torch.complex64 else 1.0e-8
    axis = torch.tensor([0.3, -0.5, 0.7], dtype=torch.float64)
    for ir in [DoubleCoverO3Irrep(1, 1, -1), DoubleCoverO3Irrep(1, 3, -1), DoubleCoverO3Irrep(1, 2, 1)]:
        D = double_cover_o3_rotation(ir, axis, 0.37, device=device, dtype=dtype)
        eye = torch.eye(ir.dim, device=device, dtype=dtype)
        _assert_close(f"{ir.label()} unitary", D.conj().T @ D, eye, tol)
        inv = double_cover_o3_inversion(ir, device=device, dtype=dtype)
        _assert_close(f"{ir.label()} inversion", inv, int(ir.parity) * eye, tol)
        if device.type == "cuda" and D.device.type != "cuda":
            raise AssertionError("rotation did not stay on CUDA")


def _test_tensor_product_equivariance(device: torch.device, dtype: torch.dtype) -> None:
    examples = [
        (DoubleCoverO3Irrep(0, 1, 1), DoubleCoverO3Irrep(1, 2, -1), DoubleCoverO3Irrep(1, 1, -1)),
        (DoubleCoverO3Irrep(0, 1, 1), DoubleCoverO3Irrep(1, 2, -1), DoubleCoverO3Irrep(1, 3, -1)),
        (DoubleCoverO3Irrep(0, 1, 1), DoubleCoverO3Irrep(0, 1, 1), DoubleCoverO3Irrep(0, 0, 1)),
        (DoubleCoverO3Irrep(0, 1, 1), DoubleCoverO3Irrep(0, 1, 1), DoubleCoverO3Irrep(1, 2, 1)),
    ]
    tol = 1.0e-5 if dtype == torch.complex64 else 1.0e-8
    for ir1, ir2, ir3 in examples:
        result = check_double_cover_o3_equivariance(
            ir1,
            ir2,
            ir3,
            device=device,
            dtype=dtype,
            ntrials=6,
            atol=tol,
            seed=1729 + ir1.two_j + 3 * ir2.two_j + 7 * ir3.two_j,
        )
        if not bool(result["passed"]):
            raise AssertionError(f"equivariance failed: {result}")
        cg = build_double_cover_o3_cg(ir1, ir2, ir3, device=device, dtype=dtype)
        if cg.device != device:
            raise AssertionError(f"CG device mismatch: got {cg.device}, expected {device}")

    wrong = build_double_cover_o3_cg(
        DoubleCoverO3Irrep(0, 1, 1),
        DoubleCoverO3Irrep(1, 2, -1),
        DoubleCoverO3Irrep(1, 1, 1),
        device=device,
        dtype=dtype,
    )
    if float(wrong.abs().max().item()) != 0.0:
        raise AssertionError("wrong-parity CG path should be exactly zero")


def _test_cache_and_module_tensor_product(device: torch.device, dtype: torch.dtype) -> None:
    clear_double_cover_o3_device_cache()
    ir1 = DoubleCoverO3Irrep(0, 1, 1)
    ir2 = DoubleCoverO3Irrep(1, 2, -1)
    out_low = DoubleCoverO3Irrep(1, 1, -1)
    out_high = DoubleCoverO3Irrep(1, 3, -1)

    cg_a = build_double_cover_o3_cg(ir1, ir2, out_low, device=device, dtype=dtype)
    cg_b = build_double_cover_o3_cg(ir1, ir2, out_low, device=device, dtype=dtype)
    if cg_a.data_ptr() != cg_b.data_ptr():
        raise AssertionError("device/dtype CG cache did not return the cached tensor")
    gen_a = double_cover_o3_generators(out_high, device=device, dtype=dtype)["z"]
    gen_b = double_cover_o3_generators(out_high, device=device, dtype=dtype)["z"]
    if gen_a.data_ptr() != gen_b.data_ptr():
        raise AssertionError("device/dtype generator cache did not return the cached tensor")

    tp = DoubleCoverO3TensorProduct(
        [ir1],
        [ir2],
        [out_low, out_high],
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    if tp.num_paths != 2:
        raise AssertionError(f"expected 2 module paths, got {tp.num_paths}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(4401)
    x1 = torch.randn(4, ir1.dim, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    x2 = torch.randn(4, ir2.dim, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    x1 = x1 + 1j * torch.randn(4, ir1.dim, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    x2 = x2 + 1j * torch.randn(4, ir2.dim, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    out = tp({ir1: x1}, {ir2: x2})
    for ir in (out_low, out_high):
        direct = apply_double_cover_o3_tp(
            x1,
            x2,
            build_double_cover_o3_cg(ir1, ir2, ir, device=device, dtype=dtype),
        )
        _assert_close(f"module direct {ir.label()}", out[ir], direct, 1.0e-5 if dtype == torch.complex64 else 1.0e-8)

    axis = torch.tensor([0.1, -0.7, 0.4], dtype=torch.float64)
    angle = 0.83
    D1 = double_cover_o3_rotation(ir1, axis, angle, device=device, dtype=dtype)
    D2 = double_cover_o3_rotation(ir2, axis, angle, device=device, dtype=dtype)
    lhs = tp({ir1: _act(D1, x1)}, {ir2: _act(D2, x2)})
    rhs_raw = tp({ir1: x1}, {ir2: x2})
    for ir in (out_low, out_high):
        D3 = double_cover_o3_rotation(ir, axis, angle, device=device, dtype=dtype)
        rhs = _act(D3, rhs_raw[ir])
        _assert_close(f"module equivariance {ir.label()}", lhs[ir], rhs, 1.0e-5 if dtype == torch.complex64 else 1.0e-8)


def run_self_tests(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> Dict[str, str]:
    dev = _resolve_device(device)
    cdtype = _resolve_complex_dtype(dtype)
    _test_decomposition()
    _test_lie_algebra_and_casimir(dev, cdtype)
    _test_rotation_inversion(dev, cdtype)
    _test_tensor_product_equivariance(dev, cdtype)
    _test_cache_and_module_tensor_product(dev, cdtype)

    if dev.type == "cuda":
        axis = torch.tensor([0.2, 0.4, -0.3], dtype=torch.float64)
        ir = DoubleCoverO3Irrep(1, 3, -1)
        rot_cpu = double_cover_o3_rotation(ir, axis, 0.41, device="cpu", dtype=torch.complex128)
        rot_gpu = double_cover_o3_rotation(ir, axis.to(dev), 0.41, device=dev, dtype=cdtype).cpu().to(torch.complex128)
        tol = 1.0e-5 if cdtype == torch.complex64 else 1.0e-10
        _assert_close("CPU/GPU rotation", rot_gpu, rot_cpu, tol)
        cg_cpu = build_double_cover_o3_cg(
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(1, 2, -1),
            DoubleCoverO3Irrep(1, 3, -1),
            device="cpu",
            dtype=torch.complex128,
        )
        cg_gpu = build_double_cover_o3_cg(
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(1, 2, -1),
            DoubleCoverO3Irrep(1, 3, -1),
            device=dev,
            dtype=cdtype,
        ).cpu().to(torch.complex128)
        _assert_close("CPU/GPU CG", cg_gpu, cg_cpu, tol)

    return {"device": str(dev), "dtype": str(cdtype), "status": "passed"}


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"complex128", "c128", "float64", "f64"}:
        return torch.complex128
    if value in {"complex64", "c64", "float32", "f32"}:
        return torch.complex64
    raise ValueError(f"Unsupported dtype argument {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-test ICTD double-cover O3 backend.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", default="complex128", help="complex128 or complex64")
    args = parser.parse_args()
    result = run_self_tests(device=args.device, dtype=_parse_dtype(args.dtype))
    print(
        "ICTD double-cover O3 self-tests passed "
        f"(device={result['device']}, dtype={result['dtype']})"
    )


if __name__ == "__main__":
    main()
