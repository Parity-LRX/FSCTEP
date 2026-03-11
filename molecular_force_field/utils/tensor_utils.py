"""Tensor utility functions for molecular modeling."""

import torch


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