import torch

from molecular_force_field.models.zbl import ZBLConfig, maybe_wrap_model_with_zbl


class _ZeroAtomEnergyModel(torch.nn.Module):
    def forward(
        self,
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        *,
        precomputed_edge_vec=None,
        external_tensor=None,
        return_physical_tensors=False,
        return_reciprocal_source=False,
        sync_after_scatter=None,
    ):
        return torch.zeros((A.shape[0], 1), dtype=pos.dtype, device=pos.device)


def test_zbl_adds_repulsive_energy_and_force():
    model = maybe_wrap_model_with_zbl(
        _ZeroAtomEnergyModel(),
        ZBLConfig(enabled=True, inner_cutoff=0.6, outer_cutoff=1.2),
    )

    pos = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    A = torch.tensor([14, 14], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    edge_src = torch.tensor([0], dtype=torch.long)
    edge_dst = torch.tensor([1], dtype=torch.long)
    edge_shifts = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0)

    atom_energy = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    total_energy = atom_energy.sum()
    forces = -torch.autograd.grad(total_energy, pos)[0]

    assert total_energy.item() > 0.0
    assert forces[0, 0].item() < 0.0
    assert forces[1, 0].item() > 0.0
    assert torch.allclose(forces[0], -forces[1], atol=1.0e-10, rtol=1.0e-10)


def test_zbl_switches_off_beyond_outer_cutoff():
    model = maybe_wrap_model_with_zbl(
        _ZeroAtomEnergyModel(),
        ZBLConfig(enabled=True, inner_cutoff=0.6, outer_cutoff=1.2),
    )

    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    A = torch.tensor([14, 14], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    edge_src = torch.tensor([0], dtype=torch.long)
    edge_dst = torch.tensor([1], dtype=torch.long)
    edge_shifts = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0)

    atom_energy = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    total_energy = atom_energy.sum()
    forces = -torch.autograd.grad(total_energy, pos)[0]

    assert torch.allclose(total_energy, torch.zeros_like(total_energy), atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(forces, torch.zeros_like(forces), atol=1.0e-12, rtol=1.0e-12)
