"""Tests for:
  1. MyE3NNCalculator with external_tensor (ICTD + electric field)
  2. MyE3NNCalculator with return_physical_tensors (ICTD + dipole/polarizability heads)
  3. Multi-structure parallel exploration (explore_n_workers > 1)

Run:
    python -m molecular_force_field.test.test_calculator_and_parallel_explore
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer,
)
from molecular_force_field.utils.config import ModelConfig


def _make_atoms(n=3):
    return Atoms(
        symbols=["H", "O", "H"][:n],
        positions=np.array(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]],
            dtype=np.float64,
        )[:n],
        pbc=False,
    )


def _build_ictd_model(device, external_rank=None, physical_tensor_outputs=None):
    config = ModelConfig(dtype=torch.float64)
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    kwargs = dict(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        hidden_dim_conv=config.channel_in,
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        lmax=config.lmax,
        internal_compute_dtype=config.dtype,
        device=device,
    )
    if external_rank is not None:
        kwargs["external_tensor_rank"] = external_rank
    if physical_tensor_outputs is not None:
        kwargs["physical_tensor_outputs"] = physical_tensor_outputs

    model = PureCartesianICTDTransformerLayer(**kwargs).to(device)
    model.eval()

    ref_dict = {
        int(k): float(v)
        for k, v in zip(
            config.atomic_energy_keys.tolist(),
            config.atomic_energy_values.tolist(),
        )
    }
    return model, ref_dict


# ------------------------------------------------------------------ #
# Test 1 — external_tensor support
# ------------------------------------------------------------------ #
def test_external_tensor(device="cpu"):
    print("=" * 60)
    print("Test 1: MyE3NNCalculator with external_tensor (rank-1 field)")
    print("=" * 60)

    model, ref_dict = _build_ictd_model(device, external_rank=1)

    E_field = torch.tensor([0.0, 0.0, 0.01], device=device, dtype=torch.float64)
    calc_field = MyE3NNCalculator(
        model, ref_dict, torch.device(device), max_radius=3.0,
        external_tensor=E_field,
    )
    calc_no_field = MyE3NNCalculator(
        model, ref_dict, torch.device(device), max_radius=3.0,
        external_tensor=None,
    )

    atoms = _make_atoms()
    atoms_f = atoms.copy()

    atoms.calc = calc_no_field
    atoms_f.calc = calc_field

    e_no = atoms.get_potential_energy()
    f_no = atoms.get_forces()
    e_with = atoms_f.get_potential_energy()
    f_with = atoms_f.get_forces()

    print(f"  E (no field) = {e_no:.6f} eV")
    print(f"  E (field)    = {e_with:.6f} eV")
    print(f"  Forces shape : {f_no.shape}")

    assert np.isfinite(e_no) and np.isfinite(e_with), "Energy not finite"
    assert f_no.shape == (3, 3), f"Bad forces shape {f_no.shape}"
    assert f_with.shape == (3, 3), f"Bad forces shape {f_with.shape}"

    print("  => PASSED\n")


# ------------------------------------------------------------------ #
# Test 2 — physical_tensor outputs
# ------------------------------------------------------------------ #
def test_physical_tensors(device="cpu"):
    print("=" * 60)
    print("Test 2: MyE3NNCalculator with return_physical_tensors")
    print("=" * 60)

    pto = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
    }
    model, ref_dict = _build_ictd_model(device, physical_tensor_outputs=pto)

    calc = MyE3NNCalculator(
        model, ref_dict, torch.device(device), max_radius=3.0,
        return_physical_tensors=True,
    )
    atoms = _make_atoms()
    atoms.calc = calc

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(f"  E = {e:.6f} eV, Forces shape = {f.shape}")

    pt = calc.results.get("physical_tensors")
    assert pt is not None, "physical_tensors not in results"
    assert "dipole" in pt, "dipole missing"
    assert "polarizability" in pt, "polarizability missing"

    dipole_l1 = pt["dipole"][1]
    print(f"  dipole[l=1] shape  = {dipole_l1.shape}")
    assert dipole_l1.shape[-1] == 3, f"dipole l=1 should be (batch, ch, 3), got {dipole_l1.shape}"

    pol_l0 = pt["polarizability"][0]
    pol_l2 = pt["polarizability"][2]
    print(f"  polarizability[l=0] shape = {pol_l0.shape}")
    print(f"  polarizability[l=2] shape = {pol_l2.shape}")
    assert pol_l2.shape[-1] == 5, f"polarizability l=2 should end in 5, got {pol_l2.shape}"

    print("  => PASSED\n")


# ------------------------------------------------------------------ #
# Test 3 — multi-structure parallel exploration
# ------------------------------------------------------------------ #
def test_parallel_explore():
    print("=" * 60)
    print("Test 3: Multi-structure parallel exploration (explore_n_workers)")
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _dummy_explore(s_idx, struct_path, sub_traj):
        """Simulate an explore_fn: read structure, write perturbed frames."""
        atoms = ase_read(struct_path)
        frames = []
        for i in range(5):
            a = atoms.copy()
            a.positions += np.random.default_rng(s_idx * 100 + i).normal(0, 0.05, a.positions.shape)
            frames.append(a)
        ase_write(sub_traj, frames, format="extxyz")
        return sub_traj

    with tempfile.TemporaryDirectory() as td:
        structs = []
        for i, (sym, pos) in enumerate([
            (["H", "H"], [[0, 0, 0], [0.74, 0, 0]]),
            (["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]]),
            (["C", "O"], [[0, 0, 0], [1.13, 0, 0]]),
        ]):
            p = os.path.join(td, f"struct_{i}.xyz")
            atoms = Atoms(symbols=sym, positions=pos, pbc=False)
            ase_write(p, atoms, format="extxyz")
            structs.append(p)

        sub_trajs = [os.path.join(td, f"traj_{i}.xyz") for i in range(len(structs))]

        # --- Sequential ---
        t0 = time.perf_counter()
        for i, (sp, st) in enumerate(zip(structs, sub_trajs)):
            _dummy_explore(i, sp, st)
        t_seq = time.perf_counter() - t0

        # --- Parallel (n_workers=3) ---
        sub_trajs_par = [os.path.join(td, f"traj_par_{i}.xyz") for i in range(len(structs))]
        t0 = time.perf_counter()
        futures = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, (sp, st) in enumerate(zip(structs, sub_trajs_par)):
                fut = pool.submit(_dummy_explore, i, sp, st)
                futures[fut] = i
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    raise RuntimeError(f"Worker {futures[fut]} failed: {exc}")
        t_par = time.perf_counter() - t0

        # Verify all trajectories exist and have correct frame count
        total_seq = 0
        total_par = 0
        for st_seq, st_par in zip(sub_trajs, sub_trajs_par):
            assert os.path.exists(st_seq), f"Missing {st_seq}"
            assert os.path.exists(st_par), f"Missing {st_par}"
            f_seq = ase_read(st_seq, index=":")
            f_par = ase_read(st_par, index=":")
            assert len(f_seq) == 5, f"Expected 5 frames, got {len(f_seq)}"
            assert len(f_par) == 5, f"Expected 5 frames, got {len(f_par)}"
            total_seq += len(f_seq)
            total_par += len(f_par)

        print(f"  Sequential: {total_seq} frames, {t_seq:.4f} s")
        print(f"  Parallel:   {total_par} frames, {t_par:.4f} s")

        # Combine — same as loop.py does
        combined = []
        for st in sub_trajs_par:
            combined.extend(ase_read(st, index=":"))
        combined_path = os.path.join(td, "explore_traj.xyz")
        ase_write(combined_path, combined, format="extxyz")
        total = len(ase_read(combined_path, index=":"))
        assert total == 15, f"Expected 15 combined frames, got {total}"
        print(f"  Combined trajectory: {total} frames")
        print("  => PASSED\n")


# ------------------------------------------------------------------ #
# Test 4 — external_tensor + physical_tensors together
# ------------------------------------------------------------------ #
def test_external_plus_physical(device="cpu"):
    print("=" * 60)
    print("Test 4: external_tensor + return_physical_tensors combined")
    print("=" * 60)

    pto = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
    }
    model, ref_dict = _build_ictd_model(
        device, external_rank=1, physical_tensor_outputs=pto,
    )

    E_field = torch.tensor([0.0, 0.0, 0.01], device=device, dtype=torch.float64)
    calc = MyE3NNCalculator(
        model, ref_dict, torch.device(device), max_radius=3.0,
        external_tensor=E_field,
        return_physical_tensors=True,
    )
    atoms = _make_atoms()
    atoms.calc = calc

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    pt = calc.results["physical_tensors"]
    dipole = pt["dipole"][1]

    print(f"  E = {e:.6f} eV")
    print(f"  Forces shape = {f.shape}")
    print(f"  dipole[l=1] shape = {dipole.shape}")

    assert np.isfinite(e), "Energy not finite"
    assert f.shape == (3, 3)
    assert dipole.shape[-1] == 3

    print("  => PASSED\n")


def main():
    ok = True
    for fn in [test_external_tensor, test_physical_tensors,
               test_parallel_explore, test_external_plus_physical]:
        try:
            fn()
        except Exception as exc:
            print(f"  => FAILED: {exc}")
            import traceback
            traceback.print_exc()
            ok = False

    if ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
