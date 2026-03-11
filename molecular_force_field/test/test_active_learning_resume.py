"""Smoke tests for active-learning bootstrap checkpoint and resume logic.

Run:
    python -m molecular_force_field.test.test_active_learning_resume
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

from ase import Atoms
from ase.io import write as ase_write

from molecular_force_field.active_learning import loop as al_loop
from molecular_force_field.active_learning.stage_scheduler import (
    make_single_stage_scheduler,
)


class _DummyModelDeviCalculator:
    def __init__(self, checkpoint_paths, device, atomic_energy_file=None):
        self.checkpoint_paths = checkpoint_paths

    def compute_from_trajectory(self, traj_path: str, output_path: str = "model_devi.out"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("# frame_id max_devi_f min_devi_f avg_devi_f devi_e\n")
            f.write("0 1.000000e-01 1.000000e-01 1.000000e-01 0.000000e+00\n")
        return []


def _write_single_frame_xyz(path: str) -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    ase_write(path, [atoms], format="extxyz")


def test_initial_checkpoint_skips_training():
    print("== test_initial_checkpoint_skips_training ==")
    with tempfile.TemporaryDirectory() as td:
        work_dir = os.path.join(td, "work")
        data_dir = os.path.join(td, "data")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        init_struct = os.path.join(td, "init.xyz")
        _write_single_frame_xyz(init_struct)
        init_ckpt = os.path.join(td, "init_model.pth")
        with open(init_ckpt, "w") as f:
            f.write("dummy")

        calls = {"train": 0, "explore": 0, "label": 0, "merge": 0}

        def fake_train_ensemble(**kwargs):
            calls["train"] += 1
            raise AssertionError("train_ensemble should be skipped in iteration 0")

        def explore_fn(iter_idx, checkpoint_path, stage, **kwargs):
            calls["explore"] += 1
            assert checkpoint_path == os.path.abspath(init_ckpt)
            out_traj = kwargs.get(
                "output_traj",
                os.path.join(work_dir, "iterations", f"iter_{iter_idx}", "explore_traj.xyz"),
            )
            _write_single_frame_xyz(out_traj)
            return out_traj

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            calls["label"] += 1
            shutil.copyfile(candidate_path, output_path)

        def fake_merge_training_data(data_dir, new_xyz_path, e0_csv_path, max_radius):
            calls["merge"] += 1

        old_train = al_loop.train_ensemble
        old_merge = al_loop.merge_training_data
        old_model_devi = al_loop.ModelDeviCalculator
        try:
            al_loop.train_ensemble = fake_train_ensemble
            al_loop.merge_training_data = fake_merge_training_data
            al_loop.ModelDeviCalculator = _DummyModelDeviCalculator

            scheduler = make_single_stage_scheduler(
                level_f_lo=0.05,
                level_f_hi=0.5,
                conv_accuracy=1.1,
                max_iters=1,
                name="bootstrap",
            )
            al_loop.run_active_learning_loop(
                work_dir=work_dir,
                data_dir=data_dir,
                explore_fn=explore_fn,
                label_fn=label_fn,
                n_models=4,
                pre_eval=False,
                explore_structure=init_struct,
                explore_structures=[init_struct],
                device="cpu",
                atomic_energy_file=None,
                max_radius=5.0,
                scheduler=scheduler,
                initial_checkpoint_paths=[init_ckpt],
                resume=False,
            )
        finally:
            al_loop.train_ensemble = old_train
            al_loop.merge_training_data = old_merge
            al_loop.ModelDeviCalculator = old_model_devi

        assert calls["train"] == 0, calls
        assert calls["explore"] == 1, calls
        assert calls["label"] == 1, calls
        assert calls["merge"] == 1, calls
        assert os.path.exists(os.path.join(work_dir, "iterations", "iter_0", "candidate.xyz"))
        assert os.path.exists(os.path.join(work_dir, "iterations", "iter_0", "merge.done"))
        print("passed")


def test_resume_reuses_partial_iteration_outputs():
    print("== test_resume_reuses_partial_iteration_outputs ==")
    with tempfile.TemporaryDirectory() as td:
        work_dir = os.path.join(td, "work")
        data_dir = os.path.join(td, "data")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        init_struct = os.path.join(td, "init.xyz")
        _write_single_frame_xyz(init_struct)

        calls = {"train": 0, "explore": 0, "label": 0, "merge": 0}

        def fake_train_ensemble(**kwargs):
            calls["train"] += 1
            ckpt_dir = os.path.join(kwargs["work_dir"], "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "model_0_spherical.pth")
            with open(ckpt_path, "w") as f:
                f.write("dummy")
            return [ckpt_path]

        def explore_fn(iter_idx, checkpoint_path, stage, **kwargs):
            calls["explore"] += 1
            out_traj = kwargs.get(
                "output_traj",
                os.path.join(work_dir, "iterations", f"iter_{iter_idx}", "explore_traj.xyz"),
            )
            _write_single_frame_xyz(out_traj)
            return out_traj

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            calls["label"] += 1
            shutil.copyfile(candidate_path, output_path)

        def fake_merge_training_data(data_dir, new_xyz_path, e0_csv_path, max_radius):
            calls["merge"] += 1
            if calls["merge"] == 1:
                raise RuntimeError("simulated crash during merge")

        old_train = al_loop.train_ensemble
        old_merge = al_loop.merge_training_data
        old_model_devi = al_loop.ModelDeviCalculator
        try:
            al_loop.train_ensemble = fake_train_ensemble
            al_loop.merge_training_data = fake_merge_training_data
            al_loop.ModelDeviCalculator = _DummyModelDeviCalculator

            scheduler = make_single_stage_scheduler(
                level_f_lo=0.05,
                level_f_hi=0.5,
                conv_accuracy=1.1,
                max_iters=1,
                name="resume",
            )
            try:
                al_loop.run_active_learning_loop(
                    work_dir=work_dir,
                    data_dir=data_dir,
                    explore_fn=explore_fn,
                    label_fn=label_fn,
                    n_models=1,
                    pre_eval=False,
                    explore_structure=init_struct,
                    explore_structures=[init_struct],
                    device="cpu",
                    atomic_energy_file=None,
                    max_radius=5.0,
                    scheduler=scheduler,
                    resume=False,
                )
            except RuntimeError as exc:
                assert "simulated crash" in str(exc)
            else:
                raise AssertionError("expected simulated crash")

            scheduler2 = make_single_stage_scheduler(
                level_f_lo=0.05,
                level_f_hi=0.5,
                conv_accuracy=1.1,
                max_iters=1,
                name="resume",
            )
            al_loop.run_active_learning_loop(
                work_dir=work_dir,
                data_dir=data_dir,
                explore_fn=explore_fn,
                label_fn=label_fn,
                n_models=1,
                pre_eval=False,
                explore_structure=init_struct,
                explore_structures=[init_struct],
                device="cpu",
                atomic_energy_file=None,
                max_radius=5.0,
                scheduler=scheduler2,
                resume=True,
            )
        finally:
            al_loop.train_ensemble = old_train
            al_loop.merge_training_data = old_merge
            al_loop.ModelDeviCalculator = old_model_devi

        assert calls["train"] == 1, calls
        assert calls["explore"] == 1, calls
        assert calls["label"] == 1, calls
        assert calls["merge"] == 2, calls
        assert os.path.exists(os.path.join(work_dir, "iterations", "iter_0", "merge.done"))
        assert os.path.exists(os.path.join(work_dir, "al_state.json"))
        print("passed")


def main():
    test_initial_checkpoint_skips_training()
    test_resume_reuses_partial_iteration_outputs()
    print("All active-learning resume tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}")
        raise
    sys.exit(0)
