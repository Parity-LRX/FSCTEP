"""Configuration selector for active learning (DPGen2-style trust levels).

Supports configurable handling of ``fail`` frames (max_devi_f >= level_f_hi):

- ``discard``      : drop all fail frames (original behaviour).
- ``sample_topk``  : promote the *least extreme* fail frames (closest to
                     ``level_f_hi``) into the candidate pool so the model
                     can learn from the boundary region.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from ase.io import read, write

logger = logging.getLogger(__name__)


class ConfSelector:
    """Select candidate configurations based on model deviation.

    Parameters
    ----------
    level_f_lo, level_f_hi : force-deviation trust-level window.
    conv_accuracy          : fraction of *accurate* frames required for convergence.
    fail_strategy          : ``"discard"`` | ``"sample_topk"``.
    fail_max_select        : how many fail frames to promote (only for *sample_topk*).
    """

    def __init__(
        self,
        level_f_lo: float = 0.05,
        level_f_hi: float = 0.50,
        level_v_lo: Optional[float] = None,
        level_v_hi: Optional[float] = None,
        conv_accuracy: float = 0.9,
        fail_strategy: str = "discard",
        fail_max_select: int = 0,
    ):
        self.level_f_lo = level_f_lo
        self.level_f_hi = level_f_hi
        self.level_v_lo = level_v_lo
        self.level_v_hi = level_v_hi
        self.conv_accuracy = conv_accuracy
        self.fail_strategy = fail_strategy
        self.fail_max_select = fail_max_select

    def select(
        self,
        traj_path: str,
        model_devi_path: str,
        output_candidate_path: str = "candidate.xyz",
    ) -> Tuple[List[int], List[int], List[int], bool]:
        """Select configurations from trajectory using model_devi.out.

        Returns
        -------
        (candidate_indices, accurate_indices, fail_indices, converged)

        ``candidate_indices`` may include promoted fail frames when
        ``fail_strategy == "sample_topk"``.
        """
        atoms_list = read(traj_path, index=":")
        max_devi_f = []
        with open(model_devi_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    max_devi_f.append(float(parts[1]))
        max_devi_f = np.array(max_devi_f)
        if len(max_devi_f) != len(atoms_list):
            raise ValueError(
                f"model_devi.out has {len(max_devi_f)} lines but trajectory "
                f"has {len(atoms_list)} frames"
            )

        id_candidate = np.where(
            (max_devi_f >= self.level_f_lo) & (max_devi_f < self.level_f_hi)
        )[0].tolist()
        id_accurate = np.where(max_devi_f < self.level_f_lo)[0].tolist()
        id_fail = np.where(max_devi_f >= self.level_f_hi)[0].tolist()

        # ---- convergence (always based on the original classification) ----
        n_total = len(max_devi_f)
        n_accurate = len(id_accurate)
        converged = (n_accurate / n_total) >= self.conv_accuracy if n_total > 0 else True

        # ---- Layer 0: fail recovery ----
        n_promoted = 0
        if self.fail_strategy == "sample_topk" and id_fail and self.fail_max_select > 0:
            fail_devi = max_devi_f[id_fail]
            order = np.argsort(fail_devi)  # ascending — least extreme first
            n_take = min(self.fail_max_select, len(order))
            promoted = [id_fail[order[j]] for j in range(n_take)]
            id_candidate = id_candidate + promoted
            id_fail = [idx for idx in id_fail if idx not in set(promoted)]
            n_promoted = n_take

        # ---- write candidates ----
        if id_candidate:
            candidate_atoms = [atoms_list[i] for i in id_candidate]
            write(output_candidate_path, candidate_atoms, format="extxyz")

        # ---- logging ----
        promo_msg = f" (+{n_promoted} from fail)" if n_promoted else ""
        logger.info(
            f"Layer 1 (uncertainty): accurate={n_accurate}, "
            f"candidate={len(id_candidate)}{promo_msg}, "
            f"fail={len(id_fail)}, converged={converged}"
        )
        return id_candidate, id_accurate, id_fail, converged
