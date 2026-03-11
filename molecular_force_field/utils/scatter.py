"""
Compatibility scatter utilities.

This project historically depended on `torch_scatter.scatter`. However, binary wheels for
`torch_scatter` are tightly coupled to specific PyTorch + CUDA versions, and importing a
mismatched build raises an `OSError` at import time (undefined symbols).

To make the codebase more robust:
  - We try to use `torch_scatter.scatter` when it is importable.
  - Otherwise, we fall back to a pure-PyTorch implementation based on `index_add`.

The fallback supports the usage patterns in this repo:
  - `dim=0`
  - `reduce in {"sum", "add", "mean"}`
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import warnings


_HAS_TORCH_SCATTER = False
_torch_scatter_scatter = None
_WARNED_FALLBACK = False

try:  # pragma: no cover
    from torch_scatter import scatter as _scatter  # type: ignore

    _HAS_TORCH_SCATTER = True
    _torch_scatter_scatter = _scatter
except Exception:  # pragma: no cover
    _HAS_TORCH_SCATTER = False
    _torch_scatter_scatter = None


def scatter_backend() -> str:
    """
    Return which implementation is currently used.
    """
    return "torch_scatter" if _HAS_TORCH_SCATTER else "torch.index_add(fallback)"


def require_torch_scatter(*, reason: str = "") -> None:
    """
    Enforce that `torch_scatter` is available.

    If `MFF_REQUIRE_TORCH_SCATTER=1` is set in the environment, we raise an ImportError
    when torch_scatter is not importable (instead of silently falling back).
    """
    require = os.environ.get("MFF_REQUIRE_TORCH_SCATTER", "").strip() in ("1", "true", "True", "YES", "yes")
    if require and not _HAS_TORCH_SCATTER:
        msg = "torch_scatter is required but is not available/importable."
        if reason:
            msg += f" Reason: {reason}"
        msg += " Install a torch_scatter wheel matching your PyTorch/CUDA."
        raise ImportError(msg)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    A subset-compatible replacement for `torch_scatter.scatter`.
    """
    red = str(reduce)
    if _HAS_TORCH_SCATTER and _torch_scatter_scatter is not None:
        return _torch_scatter_scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce=red)

    require_torch_scatter(reason="scatter fallback would slow down performance.")

    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:  # pragma: no cover
        _WARNED_FALLBACK = True
        warnings.warn(
            "torch_scatter is not available/importable; falling back to a pure-PyTorch scatter (index_add). "
            "This is correct but can be significantly slower. "
            "To speed up, install a torch_scatter wheel matching your PyTorch/CUDA.",
            stacklevel=2,
        )

    if dim != 0:
        raise NotImplementedError("Fallback scatter only supports dim=0.")

    if index.dtype != torch.long:
        index = index.long()

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    if out is None:
        out = src.new_zeros((int(dim_size),) + src.shape[1:])
    else:
        out.zero_()

    if red in ("sum", "add"):
        return out.index_add(0, index, src)

    if red == "mean":
        out = out.index_add(0, index, src)
        counts = src.new_zeros((int(dim_size),))
        ones = torch.ones_like(index, dtype=counts.dtype)
        counts = counts.index_add(0, index, ones)
        # Broadcast counts to src dims
        view = (int(dim_size),) + (1,) * (src.dim() - 1)
        return out / counts.clamp_min(1.0).view(view)

    raise NotImplementedError(f"Fallback scatter does not support reduce={reduce!r}.")

