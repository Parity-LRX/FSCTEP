import torch

from molecular_force_field.models.ictd_irreps import (
    HarmonicChannelWiseTensorProduct,
    HarmonicChannelWiseTensorProductO3,
)


def test_harmonic_channelwise_tp_so3_forward_backward():
    torch.manual_seed(0)
    mod = HarmonicChannelWiseTensorProduct(
        mul_in1=4,
        mul_in2=1,
        mul_out=6,
        lmax=2,
        internal_compute_dtype=torch.float32,
    ).to(dtype=torch.float32)
    x1 = {l: torch.randn(7, 4, 2 * l + 1, dtype=torch.float32, requires_grad=True) for l in range(3)}
    x2 = {l: torch.randn(7, 1, 2 * l + 1, dtype=torch.float32, requires_grad=True) for l in range(3)}
    gates = torch.randn(7, mod.num_paths, dtype=torch.float32)
    out = mod(x1, x2, gates)
    assert set(out.keys()) == {0, 1, 2}
    loss = sum(v.square().mean() for v in out.values())
    loss.backward()
    assert x1[0].grad is not None
    assert x2[0].grad is not None
    assert mod.weight.grad is not None


def test_harmonic_channelwise_tp_o3_forward_backward():
    torch.manual_seed(0)
    active_irreps = [(0, 1), (1, -1), (2, 1)]
    mod = HarmonicChannelWiseTensorProductO3(
        mul_in1=4,
        mul_in2=4,
        mul_out=6,
        lmax=2,
        active_irreps=active_irreps,
        internal_compute_dtype=torch.float32,
    ).to(dtype=torch.float32)
    x1 = {
        key: torch.randn(5, 4, 2 * key[0] + 1, dtype=torch.float32, requires_grad=True)
        for key in active_irreps
    }
    x2 = {(0, 1): torch.randn(5, 4, 1, dtype=torch.float32, requires_grad=True)}
    gates = torch.randn(5, mod.num_paths, dtype=torch.float32)
    out = mod(x1, x2, gates)
    assert set(out.keys()) == set(active_irreps)
    loss = sum(v.square().mean() for v in out.values())
    loss.backward()
    assert x1[(0, 1)].grad is not None
    assert x2[(0, 1)].grad is not None
    assert mod.weight.grad is not None
