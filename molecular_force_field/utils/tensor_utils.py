"""Tensor utility functions for molecular modeling."""

import torch
from molecular_force_field.models.pure_cartesian_ictd_layers import PhysicalTensorICTDEmbedding


def map_tensor_values(x, keys, values):
    """
    Map tensor values according to key-value pairs.
    
    Args:
        x: Input tensor to be mapped
        keys: Tensor of mapping keys
        values: Tensor of mapping values, one-to-one correspondence with keys
        
    Returns:
        Tensor with values replaced according to mapping rules
        
    Raises:
        ValueError: If keys and values have different lengths
    """
    # Check if keys and values have the same length
    if keys.size(0) != values.size(0):
        raise ValueError("`keys` and `values` must have the same length.")

    # Robust, vectorized mapping.
    # We assume `x` represents atomic numbers (or values convertible to ints).
    # Previous implementation relied on equality + nonzero, which can silently misbehave
    # when any value in `x` is not present in `keys`.
    x_long = x.to(dtype=torch.long)
    keys_long = keys.to(dtype=torch.long)

    # Sort keys once per call (keys are small; overhead negligible vs model forward)
    sorted_keys, sort_idx = torch.sort(keys_long)
    sorted_values = values[sort_idx]

    # searchsorted gives insertion positions; clamp and verify exact matches
    pos = torch.searchsorted(sorted_keys, x_long)
    pos = pos.clamp(min=0, max=sorted_keys.numel() - 1)
    matched = sorted_keys[pos] == x_long
    if not bool(torch.all(matched)):
        missing = torch.unique(x_long[~matched]).detach().cpu().tolist()
        raise KeyError(
            f"map_tensor_values: found values not present in keys. Missing={missing}. "
            f"Provide atomic_energy_keys/values that cover all elements in the system."
        )

    return sorted_values[pos]


def build_physical_tensor_label_blocks(
    tensor: torch.Tensor,
    *,
    rank: int,
    lmax: int,
    include_trace_chain: bool,
    representation: str,
    device: torch.device,
    cache: dict | None = None,
) -> dict[int, torch.Tensor]:
    """Convert Cartesian labels to the block representation expected by a model."""
    rep = str(representation).strip().lower()
    if rep == "cartesian":
        tensor = tensor.to(device)
        if rank == 0:
            if tensor.shape[-1:] == (1,):
                tensor = tensor[..., 0]
            return {0: tensor.unsqueeze(-1).unsqueeze(-1)}
        if rank == 1:
            if tensor.shape[-1] != 3:
                raise ValueError(f"rank-1 Cartesian label must have trailing dim 3, got {tuple(tensor.shape)}")
            return {1: tensor.unsqueeze(-2)}
        if rank == 2:
            if tensor.shape[-1:] == (6,):
                xx, yy, zz, xy, xz, yz = tensor.unbind(dim=-1)
                row0 = torch.stack((xx, xy, xz), dim=-1)
                row1 = torch.stack((xy, yy, yz), dim=-1)
                row2 = torch.stack((xz, yz, zz), dim=-1)
                tensor = torch.stack((row0, row1, row2), dim=-2)
            elif tensor.shape[-1:] == (9,):
                tensor = tensor.reshape(*tensor.shape[:-1], 3, 3)
            elif tensor.shape[-2:] != (3, 3):
                raise ValueError(
                    f"rank-2 Cartesian label must have shape (...,3,3), (...,9), or (...,6), got {tuple(tensor.shape)}"
                )
            tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
            out = {}
            if include_trace_chain:
                trace = tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
                out[0] = trace.unsqueeze(-1).unsqueeze(-1)
                eye = torch.eye(3, device=tensor.device, dtype=tensor.dtype)
                tensor = tensor - trace.unsqueeze(-1).unsqueeze(-1) * eye
            out[2] = tensor.unsqueeze(-3)
            return out
        raise ValueError(f"Unsupported Cartesian label rank={rank}")

    key = (rank, include_trace_chain, lmax)
    if cache is None:
        cache = {}
    embedder = cache.get(key)
    if embedder is None:
        embedder = PhysicalTensorICTDEmbedding(
            rank=rank,
            lmax_out=lmax,
            channels_in=1,
            channels_out=1,
            input_repr="cartesian",
            include_trace_chain=include_trace_chain,
        ).to(device)
        cache[key] = embedder
    return embedder(tensor.to(device), return_blocks=True)
