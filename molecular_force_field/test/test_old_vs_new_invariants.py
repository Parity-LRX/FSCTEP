"""
Numerical comparison: old _irreps_elementwise_tensor_product_0e vs new HarmonicElementwiseProduct("0e").

Tests on both pure_cartesian_ictd_layers.py and pure_cartesian_ictd_layers_full.py,
covering:
  1) Default path (all adapters = Identity)
  2) Adapter path (product5_muls_by_l with non-default values)
  3) Multiple num_interaction (2, 3)
  4) Multiple lmax (2, 3)
  5) End-to-end f_prod5 comparison (invs + inv3 concat)
"""
import math
import torch
import torch.nn as nn

from molecular_force_field.models.ictd_irreps import HarmonicElementwiseProduct

# ── reference (old) implementations ──

def _split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    idx = 0
    for l in range(lmax + 1):
        d = channels * (2 * l + 1)
        blk = x[..., idx: idx + d]
        idx += d
        out[l] = blk.view(*x.shape[:-1], channels, 2 * l + 1)
    return out


def _old_elementwise_tp_0e(x1, x2, channels, lmax):
    """Old _irreps_elementwise_tensor_product_0e (identity adapter path)."""
    b1 = _split_irreps(x1, channels, lmax)
    b2 = _split_irreps(x2, channels, lmax)
    outs = []
    for l in range(lmax + 1):
        outs.append((b1[l] * b2[l]).sum(dim=-1) / ((2 * l + 1) ** 0.5))
    return torch.cat(outs, dim=-1)


def _old_elementwise_tp_0e_blocks(b1, b2, muls_by_l, lmax):
    """Old _elementwise_tensor_product_0e_blocks (adapter path)."""
    outs = []
    for l in range(lmax + 1):
        x = b1[l]
        y = b2[l]
        outs.append((x * y).sum(dim=-1) / math.sqrt(2 * l + 1))
    return torch.cat(outs, dim=-1)


def _apply_channel_adapter_per_l(x_l, adapter):
    if isinstance(adapter, nn.Identity):
        return x_l
    y = adapter(x_l.movedim(-2, -1))
    return y.movedim(-1, -2)


# ── test helpers ──

SEP = "─" * 80


def compare(name, old, new):
    abs_diff = torch.abs(old - new)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff = (abs_diff / old.abs().clamp(min=1e-30)).max().item()
    match = torch.allclose(old, new, atol=1e-12, rtol=1e-10)
    status = "PASS ✓" if match else "FAIL ✗"
    print(f"  [{status}] {name}")
    print(f"    shape:    {old.shape}")
    print(f"    max_abs:  {max_diff:.2e}")
    print(f"    mean_abs: {mean_diff:.2e}")
    print(f"    max_rel:  {rel_diff:.2e}")
    return match


# ── Test 1: default path (all Identity adapters) ──

def test_default_path():
    print(f"\n{SEP}")
    print("Test 1: Default path (all adapters = Identity)")
    print(SEP)
    all_pass = True
    for lmax in [2, 3]:
        for channels in [8, 16]:
            for num_interaction in [2, 3]:
                batch = 32
                dim = channels * (lmax + 1) ** 2
                torch.manual_seed(42)
                features = [torch.randn(batch, dim) for _ in range(num_interaction)]

                product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

                # old
                old_invs = [_old_elementwise_tp_0e(f, f, channels, lmax) for f in features]
                old_cat = torch.cat(old_invs, dim=-1)

                # new
                new_invs = []
                for f in features:
                    b = _split_irreps(f, channels, lmax)
                    new_invs.append(product_5(b, b))
                new_cat = torch.cat(new_invs, dim=-1)

                tag = f"lmax={lmax} ch={channels} n_int={num_interaction}"
                ok = compare(tag, old_cat, new_cat)
                all_pass = all_pass and ok
    return all_pass


# ── Test 2: adapter path (non-uniform muls_by_l) ──

def test_adapter_path():
    print(f"\n{SEP}")
    print("Test 2: Adapter path (product5_muls_by_l non-default)")
    print(SEP)
    all_pass = True
    for lmax in [2, 3]:
        channels = 16
        num_interaction = 2
        batch = 32
        dim = channels * (lmax + 1) ** 2

        muls_by_l = {l: max(4, channels // (l + 1)) for l in range(lmax + 1)}

        # Build adapters (same logic as PureCartesianICTDTransformerLayer.__init__)
        adapters = []
        for _ in range(num_interaction):
            layer_adapt = nn.ModuleDict()
            for l in range(lmax + 1):
                out_ch = muls_by_l[l]
                if out_ch == channels:
                    layer_adapt[str(l)] = nn.Identity()
                else:
                    layer_adapt[str(l)] = nn.Linear(channels, out_ch, bias=False)
            adapters.append(layer_adapt)

        product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

        torch.manual_seed(99)
        features = [torch.randn(batch, dim) for _ in range(num_interaction)]

        # old path
        old_invs = []
        for i, f in enumerate(features):
            b = _split_irreps(f, channels, lmax)
            ba = {}
            for l in range(lmax + 1):
                ba[l] = _apply_channel_adapter_per_l(b[l], adapters[i][str(l)])
            old_invs.append(_old_elementwise_tp_0e_blocks(ba, ba, muls_by_l, lmax))
        old_cat = torch.cat(old_invs, dim=-1)

        # new path
        new_invs = []
        for i, f in enumerate(features):
            b = _split_irreps(f, channels, lmax)
            for l in range(lmax + 1):
                b[l] = _apply_channel_adapter_per_l(b[l], adapters[i][str(l)])
            new_invs.append(product_5(b, b))
        new_cat = torch.cat(new_invs, dim=-1)

        tag = f"lmax={lmax} muls={dict(muls_by_l)}"
        ok = compare(tag, old_cat, new_cat)
        all_pass = all_pass and ok
    return all_pass


# ── Test 3: full f_prod5 (invs + inv3 = scalars*scalars) ──

def test_full_f_prod5():
    print(f"\n{SEP}")
    print("Test 3: Full f_prod5 = cat(invs + [inv3]) end-to-end")
    print(SEP)
    all_pass = True
    for lmax in [2, 3]:
        channels = 16
        num_interaction = 2
        batch = 32
        dim = channels * (lmax + 1) ** 2
        combined_channels = channels * num_interaction
        scalar_channels = (num_interaction - 1) * 32

        torch.manual_seed(7)
        features = [torch.randn(batch, dim) for _ in range(num_interaction)]
        f_combine = torch.cat(features, dim=-1)

        # Simulate W_read scalars
        W_read = [torch.randn(scalar_channels, combined_channels, combined_channels) * 0.02
                   for _ in range(lmax + 1)]
        xb = _split_irreps(f_combine, combined_channels, lmax)
        scalars = torch.zeros(batch, scalar_channels)
        for l in range(lmax + 1):
            t = xb[l]
            gram = torch.einsum("ncm,ndm->ncd", t, t) / ((2 * l + 1) ** 0.5)
            scalars = scalars + torch.einsum("ocd,ncd->no", W_read[l], gram)

        inv3 = scalars * scalars

        product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

        # old f_prod5
        old_invs = [_old_elementwise_tp_0e(f, f, channels, lmax) for f in features]
        old_f_prod5 = torch.cat(old_invs + [inv3], dim=-1)

        # new f_prod5
        new_invs = []
        for f in features:
            b = _split_irreps(f, channels, lmax)
            new_invs.append(product_5(b, b))
        new_f_prod5 = torch.cat(new_invs + [inv3], dim=-1)

        tag = f"lmax={lmax} f_prod5 dim={old_f_prod5.shape[-1]}"
        ok = compare(tag, old_f_prod5, new_f_prod5)
        all_pass = all_pass and ok
    return all_pass


# ── Test 4: float64 precision ──

def test_float64():
    print(f"\n{SEP}")
    print("Test 4: float64 precision")
    print(SEP)
    lmax, channels, batch, num_interaction = 2, 16, 32, 2
    dim = channels * (lmax + 1) ** 2

    torch.manual_seed(0)
    features = [torch.randn(batch, dim, dtype=torch.float64) for _ in range(num_interaction)]

    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

    old_invs = [_old_elementwise_tp_0e(f, f, channels, lmax) for f in features]
    old_cat = torch.cat(old_invs, dim=-1)

    new_invs = []
    for f in features:
        b = _split_irreps(f, channels, lmax)
        new_invs.append(product_5(b, b))
    new_cat = torch.cat(new_invs, dim=-1)

    return compare("float64 lmax=2 ch=16 n_int=2", old_cat, new_cat)


# ── main ──

if __name__ == "__main__":
    results = []
    results.append(("Default path", test_default_path()))
    results.append(("Adapter path", test_adapter_path()))
    results.append(("Full f_prod5", test_full_f_prod5()))
    results.append(("float64", test_float64()))

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"{'=' * 80}")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        all_ok = all_ok and ok

    print(f"\n{'All tests PASSED.' if all_ok else 'Some tests FAILED!'}")
