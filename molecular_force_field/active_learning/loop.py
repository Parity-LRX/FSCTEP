"""Active learning main loop (DPGen2-style), with multi-stage scheduling
and multi-layer candidate filtering.

Filtering layers
----------------
0. **Fail recovery** – optionally promote a bounded number of high-deviation
   (fail) frames so the model can learn from boundary regions.
1. **Uncertainty gate** – keep frames whose ``max_devi_f`` falls within
   ``[level_f_lo, level_f_hi)`` (DPGen2-style trust window).
2. **Diversity selection** – use structural fingerprints (SOAP, deviation-
   histogram, …) + Farthest Point Sampling to pick a maximally diverse
   subset and cap the DFT budget.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Union

import numpy as np
from ase.io import read as ase_read, write as ase_write

from molecular_force_field.active_learning.conf_selector import ConfSelector
from molecular_force_field.active_learning.data_merge import merge_training_data
from molecular_force_field.active_learning.diversity_selector import (
    DiversitySelector,
    load_per_atom_devi,
    parse_max_devi_f,
)
from molecular_force_field.active_learning.model_devi import ModelDeviCalculator
from molecular_force_field.active_learning.pes_coverage import evaluate_pes_coverage
from molecular_force_field.active_learning.stage_scheduler import (
    ExplorationStage,
    StageScheduler,
    make_single_stage_scheduler,
)
from molecular_force_field.active_learning.train_ensemble import train_ensemble

logger = logging.getLogger(__name__)


def _state_file(work_dir: str) -> str:
    return os.path.join(work_dir, "al_state.json")


def _merge_done_file(iter_path: str) -> str:
    return os.path.join(iter_path, "merge.done")


def _converged_done_file(iter_path: str) -> str:
    return os.path.join(iter_path, "converged.done")


def _collect_existing_checkpoints(iter_path: str, n_models: int) -> Optional[List[str]]:
    ckpt_dir = os.path.join(iter_path, "checkpoint")
    if not os.path.isdir(ckpt_dir):
        return None

    checkpoint_paths: List[str] = []
    for model_idx in range(n_models):
        found = None
        for name in sorted(os.listdir(ckpt_dir)):
            if name.startswith(f"model_{model_idx}_") and name.endswith(".pth"):
                found = os.path.join(ckpt_dir, name)
        if found is None:
            return None
        checkpoint_paths.append(found)
    return checkpoint_paths


def _save_loop_state(work_dir: str, scheduler: StageScheduler) -> None:
    payload = {
        "version": 1,
        "stages": [
            {
                "converged": scheduler.is_converged(i),
                "n_iters_done": scheduler.n_iters_done(i),
            }
            for i in range(len(scheduler.stages))
        ],
        "total_iters_done": sum(scheduler.n_iters_done(i) for i in range(len(scheduler.stages))),
    }
    with open(_state_file(work_dir), "w") as f:
        json.dump(payload, f, indent=2)


def _load_loop_state(work_dir: str, scheduler: StageScheduler) -> bool:
    state_path = _state_file(work_dir)
    if not os.path.exists(state_path):
        return False

    with open(state_path) as f:
        payload = json.load(f)

    stage_states = payload.get("stages", [])
    if len(stage_states) != len(scheduler.stages):
        raise ValueError(
            f"State file {state_path} has {len(stage_states)} stage(s), "
            f"but current scheduler has {len(scheduler.stages)}."
        )

    scheduler._converged = [bool(s.get("converged", False)) for s in stage_states]
    scheduler._n_iters_done = [int(s.get("n_iters_done", 0)) for s in stage_states]
    return True


def _run_one_stage(
    stage_idx: int,
    stage: ExplorationStage,
    scheduler: StageScheduler,
    work_dir: str,
    data_dir: str,
    explore_fn: Callable,
    label_fn: Callable,
    n_models: int,
    device: str,
    e0_path: str,
    max_radius: float,
    train_args: Optional[list],
    global_iter_offset: int,
    start_local_iter: int = 0,
    diversity_selector: Optional[DiversitySelector] = None,
    fail_strategy: str = "discard",
    fail_max_select: int = 0,
    explore_structures: Optional[List[str]] = None,
    explore_n_workers: int = 1,
    initial_checkpoint_paths: Optional[List[str]] = None,
    n_gpu: int = 1,
    max_parallel: int = 0,
    nnodes: int = 1,
    master_addr: str = "auto",
    master_port: int = 29500,
    launcher: str = "auto",
    external_field: Optional[list] = None,
) -> int:
    """Run iterations for one stage until converged or max_iters reached.

    Returns the number of iterations actually run.
    """
    stage_name = stage.name or f"stage_{stage_idx}"
    iter_dir = os.path.join(work_dir, "iterations")
    os.makedirs(iter_dir, exist_ok=True)

    iters_started = start_local_iter
    for local_iter in range(start_local_iter, stage.max_iters):
        global_iter = global_iter_offset + local_iter
        logger.info(
            f"=== Stage [{stage_idx}] {stage_name}  "
            f"local_iter {local_iter + 1}/{stage.max_iters}  "
            f"(global iter {global_iter + 1}) ==="
        )
        iter_path = os.path.join(iter_dir, f"iter_{global_iter}")
        os.makedirs(iter_path, exist_ok=True)

        # ---- 1. Train ensemble ----
        bootstrap_single_checkpoint = False
        checkpoints = _collect_existing_checkpoints(iter_path, n_models)
        if checkpoints is not None:
            logger.info("Reusing existing training checkpoints for this iteration.")
        elif initial_checkpoint_paths is not None and global_iter == 0:
            if len(initial_checkpoint_paths) == n_models:
                checkpoints = list(initial_checkpoint_paths)
                logger.info(
                    "Using user-provided initial checkpoint(s) for iteration 0; "
                    "skipping ensemble training."
                )
            else:
                checkpoints = [initial_checkpoint_paths[0]]
                bootstrap_single_checkpoint = True
                logger.info(
                    "Using a single user-provided initial checkpoint for bootstrap "
                    "iteration 0; skipping ensemble training and uncertainty gating."
                )
        else:
            checkpoints = train_ensemble(
                data_dir=data_dir,
                work_dir=iter_path,
                n_models=n_models,
                base_seed=42 + global_iter * 100,
                train_args=train_args,
                n_gpu=n_gpu,
                max_parallel=max_parallel,
                nnodes=nnodes,
                master_addr=master_addr,
                master_port=master_port,
                launcher=launcher,
            )

        # ---- 2. Explore (single or multi-structure) ----
        traj_path = os.path.join(iter_path, "explore_traj.xyz")
        if os.path.exists(traj_path):
            logger.info(f"Reusing existing exploration trajectory: {traj_path}")
        elif explore_structures and len(explore_structures) > 1:
            sub_trajs = [
                os.path.join(iter_path, f"explore_traj_{s_idx}.xyz")
                for s_idx in range(len(explore_structures))
            ]
            n_workers = min(explore_n_workers, len(explore_structures))
            if n_workers > 1:
                logger.info(
                    f"  Parallel exploration: {len(explore_structures)} structures, "
                    f"{n_workers} workers"
                )
                futures = {}
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for s_idx, (struct_path, sub_traj) in enumerate(
                        zip(explore_structures, sub_trajs)
                    ):
                        fut = pool.submit(
                            explore_fn,
                            global_iter,
                            checkpoints[0],
                            stage,
                            input_structure=struct_path,
                            output_traj=sub_traj,
                        )
                        futures[fut] = (s_idx, struct_path, sub_traj)
                    for fut in as_completed(futures):
                        s_idx, struct_path, sub_traj = futures[fut]
                        exc = fut.exception()
                        if exc:
                            logger.error(
                                f"  Structure {s_idx} ({os.path.basename(struct_path)}) "
                                f"exploration failed: {exc}"
                            )
                        else:
                            logger.info(
                                f"  Structure {s_idx} ({os.path.basename(struct_path)}) done"
                            )
            else:
                logger.info(
                    f"  Sequential exploration: {len(explore_structures)} structures"
                )
                for s_idx, (struct_path, sub_traj) in enumerate(
                    zip(explore_structures, sub_trajs)
                ):
                    explore_fn(
                        global_iter, checkpoints[0], stage,
                        input_structure=struct_path, output_traj=sub_traj,
                    )

            combined_atoms = []
            for s_idx, (struct_path, sub_traj) in enumerate(
                zip(explore_structures, sub_trajs)
            ):
                if os.path.exists(sub_traj):
                    sub_atoms = ase_read(sub_traj, index=":")
                    combined_atoms.extend(sub_atoms)
                    logger.info(
                        f"  Structure {s_idx} ({os.path.basename(struct_path)}): "
                        f"{len(sub_atoms)} frames"
                    )
                else:
                    logger.warning(
                        f"  Structure {s_idx}: trajectory not found ({sub_traj})"
                    )
            traj_path = os.path.join(iter_path, "explore_traj.xyz")
            if not combined_atoms:
                raise FileNotFoundError(
                    "Multi-structure exploration produced no frames."
                )
            ase_write(traj_path, combined_atoms, format="extxyz")
            logger.info(
                f"Combined {len(combined_atoms)} frames from "
                f"{len(explore_structures)} structures"
            )
        else:
            traj_path = explore_fn(global_iter, checkpoints[0], stage)
            if not os.path.exists(traj_path):
                raise FileNotFoundError(f"Exploration failed: {traj_path}")

        # ---- 3. Model deviation (+ per-atom data for diversity) ----
        candidate_path = os.path.join(iter_path, "candidate.xyz")
        model_devi_path = os.path.join(iter_path, "model_devi.out")
        converged = False
        candidate_ids = []
        if bootstrap_single_checkpoint:
            if os.path.exists(candidate_path):
                candidate_atoms = ase_read(candidate_path, index=":")
                candidate_ids = list(range(len(candidate_atoms)))
                logger.info(
                    f"Reusing bootstrap candidates: {len(candidate_ids)} frame(s)"
                )
            else:
                bootstrap_atoms = ase_read(traj_path, index=":")
                candidate_ids = list(range(len(bootstrap_atoms)))
                ase_write(candidate_path, bootstrap_atoms, format="extxyz")
                logger.info(
                    f"Bootstrap iteration: promoted all {len(candidate_ids)} explored "
                    "frame(s) to candidates without uncertainty gating."
                )
        else:
            per_atom_path = model_devi_path.replace(".out", "_per_atom.txt")
            need_per_atom = (
                diversity_selector is not None
                and diversity_selector.metric == "devi_hist"
            )
            if os.path.exists(model_devi_path) and (not need_per_atom or os.path.exists(per_atom_path)):
                logger.info(f"Reusing existing model deviation file: {model_devi_path}")
            else:
                import torch
                calc = ModelDeviCalculator(
                    checkpoint_paths=checkpoints,
                    device=torch.device(device),
                    atomic_energy_file=e0_path,
                    external_field=external_field,
                )
                calc.compute_from_trajectory(traj_path, output_path=model_devi_path)

            # ---- 4. Layer 0 + 1: uncertainty gate (+ fail recovery) ----
            selector = ConfSelector(
                level_f_lo=stage.level_f_lo,
                level_f_hi=stage.level_f_hi,
                conv_accuracy=stage.conv_accuracy,
                fail_strategy=fail_strategy,
                fail_max_select=fail_max_select,
            )
            candidate_ids, _, _, converged = selector.select(
                traj_path, model_devi_path, output_candidate_path=candidate_path
            )

            if converged:
                logger.info(
                    f"Stage [{stage_idx}] {stage_name} converged at local_iter {local_iter + 1}."
                )
                scheduler.increment_iter(stage_idx)
                scheduler.mark_converged(stage_idx)
                with open(_converged_done_file(iter_path), "w") as f:
                    f.write("ok\n")
                _save_loop_state(work_dir, scheduler)
                return scheduler.n_iters_done(stage_idx) - iters_started

            if not os.path.exists(candidate_path) or not candidate_ids:
                logger.warning("No candidates selected; continuing to next iteration.")
                scheduler.increment_iter(stage_idx)
                _save_loop_state(work_dir, scheduler)
                continue

        # ---- 5. Layer 2: diversity selection ----
        if diversity_selector is not None and diversity_selector.metric != "none":
            candidate_atoms = ase_read(candidate_path, index=":")
            if len(candidate_atoms) > diversity_selector.max_select:
                candidate_max_devi = None
                if (not bootstrap_single_checkpoint) and os.path.exists(model_devi_path):
                    all_max_devi = parse_max_devi_f(model_devi_path)
                    candidate_max_devi = (
                        all_max_devi[np.array(candidate_ids)]
                        if len(all_max_devi) > 0
                        else None
                    )

                per_atom_devi_cand = None
                if diversity_selector.metric == "devi_hist" and bootstrap_single_checkpoint:
                    logger.warning(
                        "Bootstrap single-checkpoint mode does not have ensemble "
                        "deviation histograms; skipping Layer 2 devi_hist filtering."
                    )
                elif diversity_selector.metric == "devi_hist":
                    per_atom_path = model_devi_path.replace(".out", "_per_atom.txt")
                    if os.path.exists(per_atom_path):
                        all_per_atom = load_per_atom_devi(per_atom_path)
                        per_atom_devi_cand = [all_per_atom[i] for i in candidate_ids]

                diverse_local_ids = diversity_selector.select(
                    candidate_atoms,
                    max_devi_f=candidate_max_devi,
                    per_atom_devi=per_atom_devi_cand,
                )
                diverse_atoms = [candidate_atoms[i] for i in diverse_local_ids]
                ase_write(candidate_path, diverse_atoms, format="extxyz")
                logger.info(
                    f"Layer 2 (diversity): {len(candidate_atoms)} -> "
                    f"{len(diverse_atoms)} candidates"
                )

        # ---- 6. Label ----
        labeled_path = os.path.join(iter_path, "labeled.xyz")
        if os.path.exists(labeled_path):
            logger.info(f"Reusing existing labeled data: {labeled_path}")
        else:
            label_fn(candidate_path, labeled_path, iter_path, checkpoint_path=checkpoints[0])

        # ---- 7. Merge ----
        merge_done = _merge_done_file(iter_path)
        if os.path.exists(merge_done):
            logger.info(f"Merge already completed for this iteration: {merge_done}")
        else:
            merge_training_data(
                data_dir=data_dir,
                new_xyz_path=labeled_path,
                e0_csv_path=e0_path,
                max_radius=max_radius,
                external_field=external_field,
            )
            with open(merge_done, "w") as f:
                f.write("ok\n")
        scheduler.increment_iter(stage_idx)
        _save_loop_state(work_dir, scheduler)

    logger.warning(
        f"Stage [{stage_idx}] {stage_name} reached max_iters={stage.max_iters} without convergence."
    )
    return scheduler.n_iters_done(stage_idx) - iters_started


def run_active_learning_loop(
    work_dir: str,
    data_dir: str,
    explore_fn: Callable,
    label_fn: Callable,
    n_models: int = 4,
    n_iterations: int = 20,
    level_f_lo: float = 0.05,
    level_f_hi: float = 0.50,
    conv_accuracy: float = 0.9,
    pre_eval: bool = True,
    explore_structure: Optional[str] = None,
    device: str = "cuda",
    atomic_energy_file: Optional[str] = None,
    max_radius: float = 5.0,
    train_args: Optional[list] = None,
    scheduler: Optional[StageScheduler] = None,
    diversity_selector: Optional[DiversitySelector] = None,
    fail_strategy: str = "discard",
    fail_max_select: int = 0,
    explore_structures: Optional[List[str]] = None,
    explore_n_workers: int = 1,
    initial_checkpoint_paths: Optional[List[str]] = None,
    resume: bool = False,
    n_gpu: int = 1,
    max_parallel: int = 0,
    nnodes: int = 1,
    master_addr: str = "auto",
    master_port: int = 29500,
    launcher: str = "auto",
    external_field: Optional[list] = None,
) -> None:
    """Run DPGen2-style active learning loop with multi-layer filtering.

    Parameters
    ----------
    diversity_selector :
        A :class:`DiversitySelector` instance.  ``None`` disables Layer 2
        (diversity) filtering — backward compatible.
    fail_strategy :
        ``"discard"`` (default) or ``"sample_topk"`` (Layer 0).
    fail_max_select :
        Number of fail frames to promote when ``fail_strategy="sample_topk"``.
    explore_n_workers :
        Number of parallel workers for multi-structure exploration.
        ``1`` (default) = sequential.  ``>1`` = concurrent threads via
        ``ThreadPoolExecutor``.  Has no effect when only one structure is given.
    initial_checkpoint_paths :
        Optional list of checkpoint paths used for iteration 0. When given,
        the first iteration skips ensemble training and directly explores with
        these checkpoint(s).
    resume :
        Resume from ``work_dir/al_state.json`` and reuse any existing per-step
        outputs inside ``iterations/iter_*``.
    n_gpu :
        Number of GPUs for training each ensemble model (1 = single process).
    max_parallel :
        Max ensemble models trained simultaneously.  0 = auto
        (``available_gpus // n_gpu``).
    nnodes :
        Number of nodes for multi-node DDP (default 1 = single node).
    master_addr :
        Rendezvous address (``'auto'`` resolves via SLURM or hostname).
    master_port :
        Base rendezvous port (default 29500).
    launcher :
        ``'auto'`` / ``'local'`` / ``'slurm'``.

    Other parameters are unchanged from the original API.
    """
    os.makedirs(work_dir, exist_ok=True)
    e0_path = atomic_energy_file or os.path.join(data_dir, "fitted_E0.csv")

    if initial_checkpoint_paths is not None:
        initial_checkpoint_paths = [os.path.abspath(p) for p in initial_checkpoint_paths]
        for p in initial_checkpoint_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Initial checkpoint not found: {p}")
        if len(initial_checkpoint_paths) not in (1, n_models):
            raise ValueError(
                "initial_checkpoint_paths must contain either 1 checkpoint "
                f"or exactly n_models={n_models} checkpoints."
            )

    if pre_eval:
        logger.info("Running PES coverage evaluation...")
        evaluate_pes_coverage(
            dataset_path=data_dir,
            output_path=os.path.join(work_dir, "pes_coverage_report.json"),
        )

    # Build scheduler if not provided (backward-compatible single-stage path)
    if scheduler is None:
        scheduler = make_single_stage_scheduler(
            level_f_lo=level_f_lo,
            level_f_hi=level_f_hi,
            conv_accuracy=conv_accuracy,
            max_iters=n_iterations,
            name="stage_0",
        )
        # Wrap old-style explore_fn (2 args) to new 3-arg signature
        _raw_explore_fn = explore_fn
        import inspect
        sig = inspect.signature(_raw_explore_fn)
        if len(sig.parameters) < 3:
            def explore_fn(iter_idx, ckpt, stage):  # noqa: F811
                return _raw_explore_fn(iter_idx, ckpt)

    logger.info(
        f"Starting active learning: {len(scheduler)} stage(s), "
        f"{sum(s.max_iters for s in scheduler.stages)} max total iterations."
    )
    if diversity_selector is not None and diversity_selector.metric != "none":
        logger.info(
            f"Diversity filter: metric={diversity_selector.metric}, "
            f"max_select={diversity_selector.max_select}"
        )
    if fail_strategy != "discard":
        logger.info(f"Fail recovery: strategy={fail_strategy}, max_select={fail_max_select}")
    if n_gpu > 1 or max_parallel > 1 or nnodes > 1:
        parts = [f"{n_gpu} GPU(s)/model"]
        if nnodes > 1:
            parts.append(f"{nnodes} node(s)")
        parts.append(f"max_parallel={'auto' if max_parallel == 0 else max_parallel}")
        if launcher != "auto":
            parts.append(f"launcher={launcher}")
        logger.info(f"Training config: {', '.join(parts)}")
    if explore_structures and len(explore_structures) > 1:
        logger.info(
            f"Multi-structure exploration: {len(explore_structures)} structures, "
            f"explore_n_workers={explore_n_workers}"
        )
        for i, sp in enumerate(explore_structures):
            logger.info(f"  [{i}] {sp}")
    if initial_checkpoint_paths is not None:
        logger.info(
            f"Initial checkpoint bootstrap enabled: {len(initial_checkpoint_paths)} checkpoint(s)"
        )
    if resume:
        loaded = _load_loop_state(work_dir, scheduler)
        if loaded:
            logger.info(f"Resuming active learning from {_state_file(work_dir)}")
        else:
            logger.info("Resume requested but no state file found; starting from scratch.")

    for stage_idx, stage in scheduler:
        if scheduler.is_converged(stage_idx):
            logger.info(f"Skipping already converged stage [{stage_idx}]")
            continue
        stage_name = stage.name or f"stage_{stage_idx}"
        stage_start_local_iter = scheduler.n_iters_done(stage_idx)
        stage_global_offset = sum(
            scheduler.n_iters_done(i) for i in range(stage_idx)
        )
        logger.info(
            f">>> Entering stage [{stage_idx}] {stage_name}  "
            f"T={stage.temperature}K  steps={stage.nsteps}  "
            f"level_f=[{stage.level_f_lo}, {stage.level_f_hi}]  "
            f"conv_accuracy={stage.conv_accuracy}  max_iters={stage.max_iters}  "
            f"resume_local_iter={stage_start_local_iter}"
        )
        _run_one_stage(
            stage_idx=stage_idx,
            stage=stage,
            scheduler=scheduler,
            work_dir=work_dir,
            data_dir=data_dir,
            explore_fn=explore_fn,
            label_fn=label_fn,
            n_models=n_models,
            device=device,
            e0_path=e0_path,
            max_radius=max_radius,
            train_args=train_args,
            global_iter_offset=stage_global_offset,
            start_local_iter=stage_start_local_iter,
            diversity_selector=diversity_selector,
            fail_strategy=fail_strategy,
            fail_max_select=fail_max_select,
            explore_structures=explore_structures,
            explore_n_workers=explore_n_workers,
            initial_checkpoint_paths=initial_checkpoint_paths,
            n_gpu=n_gpu,
            max_parallel=max_parallel,
            nnodes=nnodes,
            master_addr=master_addr,
            master_port=master_port,
            launcher=launcher,
            external_field=external_field,
        )

    logger.info(scheduler.summary())
    logger.info("Active learning loop finished.")
