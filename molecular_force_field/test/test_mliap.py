"""Test ML-IAP unified interface: verify atom/edge forces numerical consistency.

Tests:
1. AtomForcesWrapper: autograd on dummy pos → per-atom forces  (new, default)
2. EdgeForcesWrapper: autograd on edge_vec → per-pair forces   (legacy)
Both should give forces identical to the traditional pos-gradient approach.

Run:
    python -m molecular_force_field.test.test_mliap
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu
from molecular_force_field.utils.tensor_utils import map_tensor_values

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _build_test_system(n_atoms: int = 5, device: str = "cpu"):
    """Build a small periodic test system."""
    rng = np.random.default_rng(42)
    pos = torch.tensor(rng.uniform(0.5, 9.5, (n_atoms, 3)), dtype=torch.float64, device=device)
    A = torch.tensor([1, 8, 1, 8, 1][:n_atoms], dtype=torch.long, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
    batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
    return pos, A, cell, batch


def _make_dummy_checkpoint_for_mode(mode: str, path: str, device: torch.device):
    """Create dummy checkpoint for the given tensor_product_mode."""
    from molecular_force_field.test.self_test_lammps_potential import (
        _make_dummy_checkpoint,
        _make_dummy_checkpoint_pure_cartesian_ictd,
        _make_dummy_checkpoint_pure_cartesian_ictd_save,
        _make_dummy_checkpoint_spherical_save,
        _make_dummy_checkpoint_spherical_save_cue,
    )
    if mode == "spherical-save-cue":
        return _make_dummy_checkpoint_spherical_save_cue(path, device)
    if mode == "spherical-save":
        return _make_dummy_checkpoint_spherical_save(path, device)
    if mode == "pure-cartesian-ictd":
        return _make_dummy_checkpoint_pure_cartesian_ictd(path, device)
    if mode == "pure-cartesian-ictd-save":
        return _make_dummy_checkpoint_pure_cartesian_ictd_save(path, device)
    return _make_dummy_checkpoint(path, device)


def _build_model_for_mode(mode: str, config: ModelConfig, device: str):
    """Build model for the given tensor_product_mode."""
    if mode == "spherical-save":
        from molecular_force_field.models.e3nn_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
        )
        return E3_TransformerLayer_multi_channelwise(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
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
            device=torch.device(device),
        ).to(device)
    if mode == "spherical-save-cue":
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue,
        )
        return E3_TransformerLayer_multi_cue(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
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
            device=torch.device(device),
        ).to(device)
    # spherical
    return E3_TransformerLayer_multi(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        irreps_input=config.get_irreps_output_conv(),
        irreps_query=config.get_irreps_query_main(),
        irreps_key=config.get_irreps_key_main(),
        irreps_value=config.get_irreps_value_main(),
        irreps_output=config.get_irreps_output_conv_2(),
        irreps_sh=config.get_irreps_sh_transformer(),
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        function_type_main=config.function_type,
        device=torch.device(device),
    ).to(device)


def test_edge_forces_consistency(device: str = "cpu", tensor_product_mode: str = "spherical"):
    """Verify that per-pair forces aggregated to per-atom match pos-gradient forces."""
    if tensor_product_mode in ("pure-cartesian-ictd", "pure-cartesian-ictd-save"):
        print("SKIP: edge forces test uses spherical architecture (pure-cartesian-ictd has different forward)")
        return True
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "dummy.pth")
        _make_dummy_checkpoint_for_mode(tensor_product_mode, ckpt_path, torch.device(device))

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ModelConfig(dtype=torch.float64)
        config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
        config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=torch.float64)

        model = _build_model_for_mode(tensor_product_mode, config, device)
        model.load_state_dict(ckpt["e3trans_state_dict"], strict=True)
        model.eval()

        pos, A, cell, batch = _build_test_system(5, device)

        # --- Method 1: Traditional pos gradient ---
        from molecular_force_field.interfaces.lammps_potential import _radius_graph_pbc_cpu_simple
        edge_src, edge_dst, edge_shifts = _radius_graph_pbc_cpu_simple(pos, 3.0, cell)

        pos1 = pos.clone().requires_grad_(True)
        atom_e1 = model(pos1, A, batch, edge_src, edge_dst, edge_shifts, cell)
        mapped_A = map_tensor_values(
            A.to(dtype=torch.float64),
            config.atomic_energy_keys,
            config.atomic_energy_values,
        )
        E1 = atom_e1.sum() + mapped_A.sum()
        grad1 = torch.autograd.grad(E1, pos1)[0]
        forces_pos = -grad1  # (N, 3)

        # --- Method 2: edge_vec gradient (ML-IAP way) ---
        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        edge_vec = (pos[edge_dst] - pos[edge_src] + shift_vecs).detach().clone().requires_grad_(True)

        # pos is unused when precomputed_edge_vec is given, but model still needs it for shape
        pos_dummy = torch.zeros_like(pos)
        atom_e2 = model(
            pos_dummy, A, batch, edge_src, edge_dst, edge_shifts, cell,
            precomputed_edge_vec=edge_vec,
        )
        E2 = atom_e2.sum() + mapped_A.sum()
        grad_ev = torch.autograd.grad(E2, edge_vec)[0]  # (E, 3), dE/d(edge_vec)

        # Aggregate edge_vec gradients to per-atom forces:
        # edge_vec = pos[dst] - pos[src] + shift
        # dE/d(pos_i) = sum_{k: src[k]=i} -dE/d(ev_k) + sum_{k: dst[k]=i} dE/d(ev_k)
        N = pos.size(0)
        forces_ev = torch.zeros(N, 3, dtype=torch.float64, device=device)
        forces_ev.index_add_(0, edge_src, -grad_ev)  # src contributes -dE/d(ev)
        forces_ev.index_add_(0, edge_dst, grad_ev)    # dst contributes +dE/d(ev)
        forces_ev = -forces_ev  # F = -dE/dpos

        # --- Compare ---
        max_diff = (forces_pos - forces_ev).abs().max().item()
        print(f"Max |F_pos - F_edge_vec|: {max_diff:.2e}")

        if max_diff > 1e-6:
            print("FAIL: edge forces inconsistent with pos forces")
            print(f"  forces_pos:\n{forces_pos}")
            print(f"  forces_ev:\n{forces_ev}")
            return False

        print("PASS: edge forces consistent with pos forces")
        return True


def test_atom_forces_consistency(device: str = "cpu", tensor_product_mode: str = "spherical"):
    """Verify AtomForcesWrapper (dE/d(pos) via dummy pos) matches traditional pos-gradient forces."""
    if tensor_product_mode in ("pure-cartesian-ictd", "pure-cartesian-ictd-save"):
        print("SKIP: atom forces test uses spherical architecture (pure-cartesian-ictd has different forward)")
        return True
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "dummy.pth")
        _make_dummy_checkpoint_for_mode(tensor_product_mode, ckpt_path, torch.device(device))

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ModelConfig(dtype=torch.float64)
        config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
        config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=torch.float64)

        model = _build_model_for_mode(tensor_product_mode, config, device)
        model.load_state_dict(ckpt["e3trans_state_dict"], strict=True)
        model.eval()

        pos, A, cell, batch = _build_test_system(5, device)

        # --- Method 1: Traditional pos gradient (ground truth) ---
        from molecular_force_field.interfaces.lammps_potential import _radius_graph_pbc_cpu_simple
        edge_src, edge_dst, edge_shifts = _radius_graph_pbc_cpu_simple(pos, 3.0, cell)

        pos1 = pos.clone().requires_grad_(True)
        atom_e1 = model(pos1, A, batch, edge_src, edge_dst, edge_shifts, cell)
        mapped_A = map_tensor_values(
            A.to(dtype=torch.float64),
            config.atomic_energy_keys,
            config.atomic_energy_values,
        )
        E1 = atom_e1.sum() + mapped_A.sum()
        grad1 = torch.autograd.grad(E1, pos1)[0]
        forces_pos = -grad1  # (N, 3)

        # --- Method 2: AtomForcesWrapper (dE/d(pos) via dummy pos + rij) ---
        from molecular_force_field.interfaces.lammps_mliap import AtomForcesWrapper

        wrapper = AtomForcesWrapper(model, config.atomic_energy_keys, config.atomic_energy_values)
        wrapper = wrapper.to(dtype=torch.float64).to(device)

        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        rij = pos[edge_dst] - pos[edge_src] + shift_vecs

        N = pos.size(0)
        _, _, atom_forces = wrapper(
            rij, A, batch, edge_src, edge_dst, edge_shifts, cell, nlocal=N,
        )

        # --- Compare ---
        max_diff = (forces_pos - atom_forces).abs().max().item()
        print(f"Max |F_pos - F_atom_wrapper|: {max_diff:.2e}")

        if max_diff > 1e-6:
            print("FAIL: AtomForcesWrapper forces inconsistent with pos forces")
            print(f"  forces_pos:\n{forces_pos}")
            print(f"  atom_forces:\n{atom_forces}")
            return False

        print("PASS: AtomForcesWrapper forces consistent with pos forces")
        return True


def test_export_and_load(device: str = "cpu", tensor_product_mode: str = "spherical"):
    """Test export script and verify the saved object can be loaded."""
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "dummy.pth")
        _make_dummy_checkpoint_for_mode(tensor_product_mode, ckpt_path, torch.device(device))

        mliap_path = os.path.join(td, "dummy-mliap.pt")
        from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
        obj = LAMMPS_MLIAP_MFF.from_checkpoint(
            ckpt_path,
            element_types=["H", "O"],
            max_radius=3.0,
            atomic_energy_keys=[1, 8],
            atomic_energy_values=[-13.6, -75.0],
            device=device,
        )
        torch.save(obj, mliap_path)

        loaded = torch.load(mliap_path, map_location=device, weights_only=False)
        assert isinstance(loaded, LAMMPS_MLIAP_MFF), f"Expected LAMMPS_MLIAP_MFF, got {type(loaded)}"
        assert loaded.element_types == ["H", "O"]
        assert loaded.rcutfac == 3.0
        print(f"PASS: export/load round-trip OK ({os.path.getsize(mliap_path)} bytes)")
        return True


def test_lammps_mliap_online(device: str = "cpu", tensor_product_mode: str = "spherical"):
    """Test ML-IAP unified interface with actual LAMMPS (requires ML-IAP build)."""
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "dummy.pth")
        _make_dummy_checkpoint_for_mode(tensor_product_mode, ckpt_path, torch.device(device))
        mliap_path = os.path.join(td, "dummy-mliap.pt")

        from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
        obj = LAMMPS_MLIAP_MFF.from_checkpoint(
            ckpt_path,
            element_types=["H", "O"],
            max_radius=3.0,
            atomic_energy_keys=[1, 8],
            atomic_energy_values=[-13.6, -75.0],
            device=device,
        )
        torch.save(obj, mliap_path)

        try:
            import lammps
            from lammps.mliap import activate_mliappy, load_unified
        except (ImportError, OSError):
            print("SKIP: lammps.mliap not available")
            return True

        lmp = lammps.lammps(cmdargs=["-nocite", "-log", "none"])
        try:
            activate_mliappy(lmp)
        except (ImportError, OSError) as e:
            print(f"SKIP: activate_mliappy failed: {e}")
            lmp.close()
            return True

        loaded = torch.load(mliap_path, map_location=device, weights_only=False)
        load_unified(loaded)

        lmp.commands_string(f"""
units metal
atom_style atomic
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 2 single 0.96 0.0 0.0
create_atoms 1 single 0.24 0.93 0.0
mass 1 1.008
mass 2 15.999
pair_style mliap unified {mliap_path} 0
pair_coeff * * H O
""".strip())

        try:
            lmp.command("run 0")
            e_total = lmp.get_thermo("pe")
            print(f"LAMMPS ML-IAP run 0: E = {e_total:.4f} eV")
            lmp.close()
            print("PASS: LAMMPS ML-IAP online test")
            return True
        except Exception as e:
            print(f"FAIL: LAMMPS ML-IAP run 0 error: {e}")
            lmp.close()
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ML-IAP interface tests")
    parser.add_argument("--mode", "--tensor-product-mode", dest="mode", type=str,
                        default="spherical",
                        choices=["spherical", "spherical-save", "spherical-save-cue", "pure-cartesian-ictd", "pure-cartesian-ictd-save"],
                        help="Model mode to test (default: spherical)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("ML-IAP Interface Tests")
    print("=" * 60)
    print(f"Mode: {args.mode}")

    ok = True

    print("\n--- Test 1: atom forces consistency (AtomForcesWrapper) ---")
    ok = test_atom_forces_consistency(args.device, args.mode) and ok

    print("\n--- Test 2: edge forces consistency (legacy EdgeForcesWrapper) ---")
    ok = test_edge_forces_consistency(args.device, args.mode) and ok

    print("\n--- Test 3: export/load round-trip ---")
    ok = test_export_and_load(args.device, args.mode) and ok

    print("\n--- Test 4: LAMMPS ML-IAP online test ---")
    ok = test_lammps_mliap_online(args.device, args.mode) and ok

    print("\n" + "=" * 60)
    if ok:
        print("All ML-IAP tests PASSED.")
    else:
        print("Some ML-IAP tests FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
