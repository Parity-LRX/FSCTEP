from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from molecular_force_field.active_learning.conf_selector import ConfSelector
from molecular_force_field.active_learning.data_merge import _read_existing_blocks, merge_training_data
from molecular_force_field.active_learning.loop import _select_exploration_checkpoint
from molecular_force_field.cli.init_data import (
    _filter_labeled_atoms_by_force,
    _sample_uniform_indices,
)
from molecular_force_field.cli.preprocess import _source_tail_split_indices


def _write_dummy_labeled_xyz(path: Path, n_frames: int = 5) -> None:
    atoms_list = []
    for i in range(n_frames):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75 + 0.02 * i, 0.0, 0.0]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-1.0 + 0.05 * i,
            forces=np.zeros((2, 3), dtype=float),
        )
        atoms_list.append(atoms)
    write(path, atoms_list, format="extxyz")


def test_conf_selector_raises_on_nonfinite_model_deviation(tmp_path: Path) -> None:
    traj_path = tmp_path / "traj.xyz"
    write(traj_path, [Atoms("H", positions=[[0.0, 0.0, 0.0]])], format="extxyz")

    model_devi_path = tmp_path / "model_devi.out"
    model_devi_path.write_text("# frame_id max_devi_f min_devi_f avg_devi_f devi_e\n0 nan 0.0 0.0 0.0\n")

    selector = ConfSelector()
    with pytest.raises(FloatingPointError):
        selector.select(str(traj_path), str(model_devi_path), str(tmp_path / "candidate.xyz"))


def test_merge_training_data_updates_train_and_val(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "molecular_force_field.active_learning.data_merge.save_to_h5_parallel",
        lambda *args, **kwargs: None,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"Atom": [1], "E0": [0.0]}).to_csv(data_dir / "fitted_E0.csv", index=False)

    labeled_path = tmp_path / "labeled.xyz"
    _write_dummy_labeled_xyz(labeled_path, n_frames=5)

    added = merge_training_data(
        data_dir=str(data_dir),
        new_xyz_path=str(labeled_path),
        val_ratio=0.4,
        random_seed=7,
    )

    assert added == 5
    train_blocks = _read_existing_blocks(str(data_dir), "train")[0]
    val_blocks = _read_existing_blocks(str(data_dir), "val")[0]
    assert train_blocks is not None
    assert val_blocks is not None
    assert len(train_blocks) == 3
    assert len(val_blocks) == 2


def test_select_exploration_checkpoint_round_robin() -> None:
    checkpoints = ["model_0.pth", "model_1.pth", "model_2.pth"]
    ckpt, idx = _select_exploration_checkpoint(checkpoints, global_iter=4, slot_idx=1)
    assert ckpt == "model_2.pth"
    assert idx == 2


def test_source_tail_split_prefers_farther_perturbations(tmp_path: Path) -> None:
    input_path = tmp_path / "cold_start.xyz"
    atoms_list = []
    for displacement in [0.0, 0.02, 0.05, 0.30]:
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75 + displacement, 0.0, 0.0]])
        atoms.info["source"] = "seed_A"
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-1.0,
            forces=np.zeros((2, 3), dtype=float),
        )
        atoms_list.append(atoms)
    write(input_path, atoms_list, format="extxyz")

    train_idx, val_idx = _source_tail_split_indices(str(input_path), data_size=4, train_ratio=0.75, seed=0)
    assert list(val_idx) == [3]
    assert list(train_idx) == [0, 1, 2]


def test_filter_labeled_atoms_by_force() -> None:
    low = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    low.calc = SinglePointCalculator(low, energy=-1.0, forces=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    high = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.80, 0.0, 0.0]])
    high.calc = SinglePointCalculator(high, energy=-0.8, forces=np.array([[0.0, 0.0, 0.0], [7.5, 0.0, 0.0]]))

    kept, dropped = _filter_labeled_atoms_by_force([low, high], max_force_filter=6.0)
    assert len(kept) == 1
    assert kept[0].get_potential_energy() == pytest.approx(-1.0)
    assert len(dropped) == 1
    assert dropped[0][0] == 1
    assert dropped[0][1] == pytest.approx(7.5)


def test_sample_uniform_indices() -> None:
    indices = _sample_uniform_indices(1000, 100)
    assert len(indices) == 100
    assert len(set(indices)) == 100
    assert indices[0] == 0
    assert indices[-1] == 999
