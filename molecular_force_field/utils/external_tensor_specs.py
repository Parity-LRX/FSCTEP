from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def canonical_irrep_parity_sign(rank: int) -> int:
    return 1 if int(rank) % 2 == 0 else -1


def parity_letter_to_sign(parity: str | int | None) -> int:
    if parity is None:
        raise ValueError("parity cannot be None")
    if isinstance(parity, int):
        if parity == 0:
            raise ValueError("parity sign cannot be 0")
        return 1 if int(parity) > 0 else -1
    text = str(parity).strip().lower()
    if text in {"e", "even", "+1", "1"}:
        return 1
    if text in {"o", "odd", "-1"}:
        return -1
    raise ValueError(f"Unsupported parity value {parity!r}")


def parity_sign_to_letter(parity: int) -> str:
    return "e" if int(parity) >= 0 else "o"


def normalize_external_tensor_irrep(
    *,
    rank: int | None,
    irrep: str | None,
    parity: str | int | None,
) -> str | None:
    if rank is None:
        if irrep is not None or parity is not None:
            raise ValueError("external_tensor_irrep/parity requires external_tensor_rank to be set")
        return None
    rank = int(rank)
    if irrep is not None:
        text = str(irrep).strip()
        if len(text) < 2:
            raise ValueError(f"Invalid external tensor irrep {irrep!r}")
        l_val = int(text[:-1])
        if l_val != rank:
            raise ValueError(f"external_tensor_irrep {irrep!r} does not match external_tensor_rank={rank}")
        p = parity_letter_to_sign(text[-1])
        return f"{l_val}{parity_sign_to_letter(p)}"
    if parity is None:
        if rank == 1:
            return "1o"
        return f"{rank}{parity_sign_to_letter(canonical_irrep_parity_sign(rank))}"
    return f"{rank}{parity_sign_to_letter(parity_letter_to_sign(parity))}"


def external_tensor_numel_for_rank(rank: int) -> int:
    rank = int(rank)
    if rank < 0:
        raise ValueError("external tensor rank must be >= 0")
    return 3 ** rank if rank > 0 else 1


def canonicalize_external_tensor_shape(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    rank = int(rank)
    if rank == 0:
        if tensor.shape[-1:] == (1,):
            return tensor
        if tensor.ndim == 0:
            return tensor.reshape(1)
        raise ValueError(f"rank-0 external tensor must have shape (..., 1), got {tuple(tensor.shape)}")
    if rank == 1:
        if tensor.shape[-1:] == (3,):
            return tensor
        raise ValueError(f"rank-1 external tensor must have shape (..., 3), got {tuple(tensor.shape)}")
    if rank == 2:
        if tensor.shape[-2:] == (3, 3):
            return tensor
        if tensor.shape[-1:] == (9,):
            return tensor.reshape(*tensor.shape[:-1], 3, 3)
        if tensor.shape[-1:] == (6,):
            xx, yy, zz, xy, xz, yz = tensor.unbind(dim=-1)
            row0 = torch.stack((xx, xy, xz), dim=-1)
            row1 = torch.stack((xy, yy, yz), dim=-1)
            row2 = torch.stack((xz, yz, zz), dim=-1)
            return torch.stack((row0, row1, row2), dim=-2)
        raise ValueError(f"rank-2 external tensor must have shape (..., 3, 3), (..., 9), or (..., 6), got {tuple(tensor.shape)}")
    expected = (3,) * rank
    if tensor.shape[-rank:] == expected:
        return tensor
    if tensor.shape[-1:] == (3 ** rank,):
        return tensor.reshape(*tensor.shape[:-1], *expected)
    raise ValueError(f"rank-{rank} external tensor must have trailing shape {expected} or (..., {3 ** rank}), got {tuple(tensor.shape)}")


def flatten_external_tensor(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    tensor = canonicalize_external_tensor_shape(tensor, rank)
    return tensor.reshape(*tensor.shape[:-int(rank)], external_tensor_numel_for_rank(rank)) if int(rank) > 0 else tensor.reshape(*tensor.shape[:-1], 1)


def reshape_flat_external_tensor(flat_tensor: torch.Tensor, rank: int) -> torch.Tensor:
    rank = int(rank)
    numel = external_tensor_numel_for_rank(rank)
    if flat_tensor.shape[-1] != numel:
        raise ValueError(f"Expected {numel} values for rank-{rank} external tensor, got shape {tuple(flat_tensor.shape)}")
    if rank == 0:
        return flat_tensor
    return flat_tensor.reshape(*flat_tensor.shape[:-1], *([3] * rank))


def normalize_external_tensor_specs(
    specs: Iterable[Mapping[str, Any]] | None = None,
    *,
    external_tensor_rank: int | None = None,
    external_tensor_irrep: str | None = None,
    external_tensor_parity: str | int | None = None,
    default_name: str = "external_field",
) -> list[dict[str, Any]] | None:
    normalized: list[dict[str, Any]] = []
    if specs is not None:
        seen_names: set[str] = set()
        for spec_in in specs:
            name = str(spec_in.get("name", "")).strip()
            if not name:
                raise ValueError(f"external_tensor_specs entry missing name: {spec_in!r}")
            if name in seen_names:
                raise ValueError(f"Duplicate external tensor spec name {name!r}")
            seen_names.add(name)
            rank = spec_in.get("rank")
            if rank is None:
                raise ValueError(f"external_tensor_specs[{name!r}] missing rank")
            irrep = normalize_external_tensor_irrep(
                rank=int(rank),
                irrep=spec_in.get("irrep"),
                parity=spec_in.get("parity"),
            )
            normalized.append(
                {
                    "name": name,
                    "rank": int(rank),
                    "irrep": irrep,
                    "numel": external_tensor_numel_for_rank(int(rank)),
                }
            )
    elif external_tensor_rank is not None:
        rank = int(external_tensor_rank)
        normalized.append(
            {
                "name": str(default_name),
                "rank": rank,
                "irrep": normalize_external_tensor_irrep(
                    rank=rank,
                    irrep=external_tensor_irrep,
                    parity=external_tensor_parity,
                ),
                "numel": external_tensor_numel_for_rank(rank),
            }
        )
    return normalized or None


def build_standard_external_tensor_specs(
    *,
    external_tensor_rank: int | None,
    external_tensor_irrep: str | None,
    external_tensor_parity: str | int | None = None,
    include_magnetic_field: bool = False,
    default_name: str = "external_field",
) -> list[dict[str, Any]] | None:
    specs = normalize_external_tensor_specs(
        None,
        external_tensor_rank=external_tensor_rank,
        external_tensor_irrep=external_tensor_irrep,
        external_tensor_parity=external_tensor_parity,
        default_name=default_name,
    )
    if include_magnetic_field:
        magnetic_specs = normalize_external_tensor_specs(
            None,
            external_tensor_rank=1,
            external_tensor_irrep="1e",
            external_tensor_parity=None,
            default_name="magnetic_field",
        )
        specs = (specs or []) + (magnetic_specs or [])
    return specs or None


def external_tensor_total_numel(specs: Iterable[Mapping[str, Any]] | None) -> int:
    if not specs:
        return 0
    specs_list = [dict(spec) for spec in specs]
    if any("numel" not in spec for spec in specs_list):
        specs_list = normalize_external_tensor_specs(specs_list) or []
    return int(sum(int(spec["numel"]) for spec in specs_list))


def get_external_tensor_spec(specs: Iterable[Mapping[str, Any]] | None, name: str) -> dict[str, Any] | None:
    if not specs:
        return None
    specs_list = [dict(spec) for spec in specs]
    if any("numel" not in spec for spec in specs_list):
        specs_list = normalize_external_tensor_specs(specs_list) or []
    for spec in specs_list:
        if str(spec.get("name")) == str(name):
            return dict(spec)
    return None


def unpack_external_tensor(
    external_tensor: torch.Tensor,
    specs: Iterable[Mapping[str, Any]] | None,
) -> dict[str, torch.Tensor]:
    if not specs:
        return {}
    specs_list = [dict(spec) for spec in specs]
    if any("numel" not in spec for spec in specs_list):
        specs_list = normalize_external_tensor_specs(specs_list) or []
    total = external_tensor_total_numel(specs_list)
    if external_tensor.shape[-1] != total:
        raise ValueError(f"Packed external tensor trailing dim must be {total}, got {tuple(external_tensor.shape)}")
    out: dict[str, torch.Tensor] = {}
    start = 0
    for spec in specs_list:
        end = start + int(spec["numel"])
        out[str(spec["name"])] = reshape_flat_external_tensor(external_tensor[..., start:end], int(spec["rank"]))
        start = end
    return out


def pack_external_tensor_dict(
    tensors: Mapping[str, torch.Tensor | None],
    specs: Iterable[Mapping[str, Any]] | None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    batch_size: int | None = None,
) -> torch.Tensor | None:
    if not specs:
        return None
    specs_list = [dict(spec) for spec in specs]
    if any("numel" not in spec for spec in specs_list):
        specs_list = normalize_external_tensor_specs(specs_list) or []
    parts: list[torch.Tensor] = []
    leading_shape: tuple[int, ...] | None = None
    for spec in specs_list:
        name = str(spec["name"])
        rank = int(spec["rank"])
        numel = int(spec["numel"])
        tensor = tensors.get(name)
        if tensor is None:
            if leading_shape is None:
                if batch_size is None:
                    leading_shape = ()
                else:
                    leading_shape = (int(batch_size),)
            zeros = torch.zeros(*leading_shape, numel, device=device, dtype=dtype)
            parts.append(zeros)
            continue
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        flat = flatten_external_tensor(tensor, rank)
        if leading_shape is None:
            leading_shape = tuple(flat.shape[:-1])
        elif tuple(flat.shape[:-1]) != leading_shape:
            raise ValueError(f"External tensor {name!r} leading shape {tuple(flat.shape[:-1])} != {leading_shape}")
        parts.append(flat)
    if not parts:
        return None
    return torch.cat(parts, dim=-1)


def pack_external_tensor_from_extras(
    extras: Mapping[str, Any] | None,
    specs: Iterable[Mapping[str, Any]] | None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if not specs:
        return None
    extras = extras or {}
    tensors: dict[str, torch.Tensor | None] = {}
    batch_size: int | None = None
    for spec in specs:
        name = str(spec["name"])
        tensor = extras.get(name)
        if torch.is_tensor(tensor):
            tensors[name] = tensor
            if batch_size is None and tensor.ndim >= 2:
                batch_size = int(tensor.shape[0])
        else:
            tensors[name] = None
    if not any(torch.is_tensor(v) for v in tensors.values()):
        return None
    return pack_external_tensor_dict(tensors, specs, device=device, dtype=dtype, batch_size=batch_size)
