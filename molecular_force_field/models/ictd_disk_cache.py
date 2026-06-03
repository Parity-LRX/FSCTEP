"""On-disk (L2) cache for ICTD Clebsch-Gordan and symmetric-contraction U tensors.

These tensors are pure, deterministic functions of a few small config params
(the l's, lmax, correlation, normalization) and are mathematically exact in
float64. Computing them is expensive (lmax=3 correlation=3 pure-U build ~24s) and
currently happens on *every* fresh process -- each training run, each DDP worker,
each LAMMPS model load, each test. This module persists the float64 results to
disk keyed by the value-determining params, so any later process loads them in
well under a second.

Design / why it is safe (numerical consistency + equivariance are hard constraints):
  * Stored as **float64 CPU** tensors -- the canonical high-precision values.
    Callers cast to the model dtype: float64 -> bit-identical to on-the-fly;
    float32 -> a downcast of the float64 value (strictly >= as accurate as the
    current native-float32 compute path).
  * The cache directory path embeds CACHE_VERSION. Bump CACHE_VERSION whenever the
    CG/U math changes and every stale entry is ignored (a new dir is used).
  * Each entry stores a sha256 of its own tensor bytes; load re-checks it and
    treats a mismatch (corruption / truncation) as a miss -> recompute.
  * Writes are atomic (temp file + os.replace), so a crash mid-write never leaves
    a torn file that a later process would read.
  * Loads prefer weights_only=True.
  * Kill switch: env FSCETP_ICTD_CACHE=0 disables the cache entirely (the builders
    fall back to exact on-the-fly computation -- identical to pre-cache behaviour).
  * Any failure at all (unwritable dir, unreadable/corrupt file, torch.load error)
    falls back to computing the value. The cache can never change a returned value
    and can never crash a model build; worst case it is a no-op.

Layout:  <root>/<CACHE_VERSION>/<namespace>/<key>.pt
  root  = $FSCETP_ICTD_CACHE_DIR, else $XDG_CACHE_HOME/fscetp_ictd, else
          ~/.cache/fscetp_ictd
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Callable, Sequence

import torch

# Bump this string on ANY change to the CG / U mathematics or storage format.
# A new value uses a fresh cache directory, so stale precomputed tensors are
# never read.
CACHE_VERSION = "v1"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def is_enabled() -> bool:
    """Cache on unless FSCETP_ICTD_CACHE is explicitly falsey."""
    return os.environ.get("FSCETP_ICTD_CACHE", "1").strip().lower() not in _FALSE


def _validate_on_load() -> bool:
    """If FSCETP_ICTD_CACHE_VALIDATE is set, recompute+compare on every hit (tests/CI)."""
    return os.environ.get("FSCETP_ICTD_CACHE_VALIDATE", "0").strip().lower() in _TRUE


def _default_root() -> str:
    """In-repo cache directory (ships with the code).

    Placing the cache next to the source means the precomputed CG/U tensors are
    committed and read directly on every checkout (4090 / Parity / local) with no
    recompute. Override with FSCETP_ICTD_CACHE_DIR to point elsewhere (e.g. a shared
    scratch dir, or to keep a working tree clean during exploratory runs).
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ictd_cache")


def cache_root() -> str:
    d = os.environ.get("FSCETP_ICTD_CACHE_DIR") or _default_root()
    return os.path.join(d, CACHE_VERSION)


def _key_parts(key: Sequence) -> list:
    return [str(k) for k in key]


def _entry_path(namespace: str, key: Sequence) -> str:
    parts = _key_parts(key)
    raw = namespace + "__" + "_".join(parts)
    safe = raw.replace(os.sep, "_").replace("/", "_").replace(" ", "")
    # Keep file names bounded; long/odd keys (e.g. e3nn irreps strings) get hashed.
    if len(safe) > 180 or any(c in safe for c in '\\:*?"<>|'):
        safe = namespace + "__" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return os.path.join(cache_root(), namespace, safe + ".pt")


def _tensor_sha(t: torch.Tensor) -> str:
    arr = t.detach().to("cpu", torch.float64).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _load(path: str, key: Sequence) -> torch.Tensor | None:
    if not os.path.exists(path):
        return None
    try:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Our own file in our own cache dir; tolerate older torch without
            # weights_only dict support.
            blob = torch.load(path, map_location="cpu")
    except Exception as e:  # unreadable / truncated
        logging.debug("ictd_disk_cache: load failed %s: %s", path, e)
        return None
    if not isinstance(blob, dict) or "tensor" not in blob:
        return None
    t = blob["tensor"]
    if not torch.is_tensor(t):
        return None
    # Integrity + key/version guards: any mismatch -> treat as miss.
    if blob.get("version") != CACHE_VERSION:
        return None
    if blob.get("key") != _key_parts(key):
        return None
    try:
        if blob.get("sha") != _tensor_sha(t):
            logging.debug("ictd_disk_cache: sha mismatch %s -> recompute", path)
            return None
    except Exception:
        return None
    return t.to(torch.float64).contiguous()


def _store(path: str, key: Sequence, namespace: str, t: torch.Tensor) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob = {
            "tensor": t,
            "sha": _tensor_sha(t),
            "key": _key_parts(key),
            "namespace": namespace,
            "version": CACHE_VERSION,
        }
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        os.close(fd)
        try:
            torch.save(blob, tmp)
            os.replace(tmp, path)  # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    except Exception as e:  # unwritable dir / disk full / race -> just skip
        logging.debug("ictd_disk_cache: store skipped %s: %s", path, e)


def load_or_compute(namespace: str, key: Sequence, compute_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Return the float64 CPU tensor for (namespace, key).

    Loads from disk when present+valid, otherwise computes via compute_fn(),
    stores it, and returns it. Always returns a float64 CPU tensor; callers cast
    to the model device/dtype exactly as before. Never raises due to the cache:
    on any cache error it falls back to compute_fn().
    """
    if not is_enabled():
        return compute_fn().detach().to("cpu", torch.float64).contiguous()

    path = _entry_path(namespace, key)
    cached = _load(path, key)
    if cached is not None:
        if _validate_on_load():
            fresh = compute_fn().detach().to("cpu", torch.float64).contiguous()
            if not torch.equal(fresh, cached):
                logging.warning("ictd_disk_cache: VALIDATE mismatch %s -> using fresh + rewriting", path)
                _store(path, key, namespace, fresh)
                return fresh
        return cached

    t = compute_fn().detach().to("cpu", torch.float64).contiguous()
    _store(path, key, namespace, t)
    return t


def cache_info() -> dict:
    """Summary of the on-disk cache (for the precompute CLI / debugging)."""
    root = cache_root()
    n, total = 0, 0
    by_ns: dict[str, int] = {}
    if os.path.isdir(root):
        for ns in sorted(os.listdir(root)):
            nsdir = os.path.join(root, ns)
            if not os.path.isdir(nsdir):
                continue
            for fn in os.listdir(nsdir):
                if fn.endswith(".pt"):
                    n += 1
                    by_ns[ns] = by_ns.get(ns, 0) + 1
                    try:
                        total += os.path.getsize(os.path.join(nsdir, fn))
                    except OSError:
                        pass
    return {"root": root, "enabled": is_enabled(), "entries": n,
            "by_namespace": by_ns, "total_bytes": total}
