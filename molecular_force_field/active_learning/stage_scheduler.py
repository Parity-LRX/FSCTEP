"""
Multi-stage scheduler for active learning (DPGen2-style).

Concept:
  - An ExplorationStage holds the exploration parameters for one stage
    (e.g. temperature, MD steps, trust levels).
  - A StageScheduler manages a list of stages and drives the outer loop:
      for each stage, run iterations until converged, then move to next stage.
"""

import dataclasses
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExplorationStage:
    """
    Parameters for one exploration stage.

    Attributes
    ----------
    temperature: float
        MD temperature in K (default: 300.0)
    timestep: float
        MD timestep in fs (default: 1.0)
    nsteps: int
        Number of MD steps per iteration (default: 1000)
    friction: float
        Langevin friction coefficient (default: 0.01)
    relax_fmax: float
        Max force for pre-relaxation; 0 = skip (default: 0.05)
    log_interval: int
        Steps between trajectory writes (default: 10)
    level_f_lo: float
        Lower force deviation threshold (default: 0.05)
    level_f_hi: float
        Upper force deviation threshold (default: 0.50)
    conv_accuracy: float
        Fraction of accurate frames required for convergence (default: 0.9)
    max_iters: int
        Max iterations for this stage (default: 20)
    name: str
        Human-readable stage name (default: auto)
    """

    temperature: float = 300.0
    timestep: float = 1.0
    nsteps: int = 1000
    friction: float = 0.01
    relax_fmax: float = 0.05
    log_interval: int = 10
    level_f_lo: float = 0.05
    level_f_hi: float = 0.50
    conv_accuracy: float = 0.9
    max_iters: int = 20
    name: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplorationStage":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class StageScheduler:
    """
    Drive a list of ExplorationStage objects.

    Usage
    -----
    scheduler = StageScheduler(stages=[...])
    for stage_idx, stage in scheduler:
        # run iterations for this stage
        converged = scheduler.run_stage(stage_idx, explore_fn, ...)
        if converged:
            scheduler.mark_converged(stage_idx)
    """

    def __init__(self, stages: List[ExplorationStage]):
        if not stages:
            raise ValueError("At least one stage is required.")
        self.stages = stages
        self._converged: List[bool] = [False] * len(stages)
        self._n_iters_done: List[int] = [0] * len(stages)

    @classmethod
    def from_dicts(cls, stage_dicts: List[Dict[str, Any]]) -> "StageScheduler":
        return cls([ExplorationStage.from_dict(d) for d in stage_dicts])

    @classmethod
    def from_json(cls, path: str) -> "StageScheduler":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls.from_dicts(data)
        return cls.from_dicts(data["stages"])

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self):
        for i, s in enumerate(self.stages):
            yield i, s

    def mark_converged(self, stage_idx: int) -> None:
        self._converged[stage_idx] = True

    def is_converged(self, stage_idx: int) -> bool:
        return self._converged[stage_idx]

    def all_converged(self) -> bool:
        return all(self._converged)

    def increment_iter(self, stage_idx: int) -> None:
        self._n_iters_done[stage_idx] += 1

    def n_iters_done(self, stage_idx: int) -> int:
        return self._n_iters_done[stage_idx]

    def summary(self) -> str:
        lines = ["Stage scheduler summary:"]
        for i, s in enumerate(self.stages):
            name = s.name or f"stage_{i}"
            status = "converged" if self._converged[i] else f"{self._n_iters_done[i]} iter(s)"
            lines.append(f"  [{i}] {name} T={s.temperature}K steps={s.nsteps}: {status}")
        return "\n".join(lines)


def make_single_stage_scheduler(
    temperature: float = 300.0,
    timestep: float = 1.0,
    nsteps: int = 1000,
    friction: float = 0.01,
    relax_fmax: float = 0.05,
    log_interval: int = 10,
    level_f_lo: float = 0.05,
    level_f_hi: float = 0.50,
    conv_accuracy: float = 0.9,
    max_iters: int = 20,
    name: str = "stage_0",
) -> StageScheduler:
    """Create a StageScheduler with a single stage (backward compatible)."""
    stage = ExplorationStage(
        temperature=temperature,
        timestep=timestep,
        nsteps=nsteps,
        friction=friction,
        relax_fmax=relax_fmax,
        log_interval=log_interval,
        level_f_lo=level_f_lo,
        level_f_hi=level_f_hi,
        conv_accuracy=conv_accuracy,
        max_iters=max_iters,
        name=name,
    )
    return StageScheduler([stage])
