"""Export a checkpoint to LAMMPS ML-IAP unified format.

Only the following five models support ML-IAP (precomputed_edge_vec / edge forces):
- e3nn_layers.py (spherical)
- e3nn_layers_channelwise.py (spherical-save)
- cue_layers_channelwise.py (spherical-save-cue, cuEquivariance GPU acceleration)
- pure_cartesian_ictd_layers.py (pure-cartesian-ictd-save)
- pure_cartesian_ictd_layers_full.py (pure-cartesian-ictd)

Usage:
    python -m molecular_force_field.cli.export_mliap checkpoint.pth \\
        --elements H O \\
        --atomic-energy-keys 1 8 \\
        --atomic-energy-values -13.6 -75.0 \\
        --output model-mliap.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Export checkpoint to LAMMPS ML-IAP unified format (.pt). "
                    "Model-structure hyperparameters default to checkpoint metadata; explicit CLI values override.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=str,
                        help="Path to model checkpoint (.pth). Structure hyperparameters are resolved with priority: "
                             "explicit CLI > checkpoint metadata > defaults.")
    parser.add_argument("--elements", nargs="+", required=True,
                        help="Element symbols in LAMMPS type order (e.g. H O)")
    parser.add_argument("--atomic-energy-keys", nargs="+", type=int, default=None,
                        help="Atomic numbers for baseline energies (e.g. 1 8). Highest priority override; "
                             "if not set, prefer checkpoint E0 when available.")
    parser.add_argument("--atomic-energy-values", nargs="+", type=float, default=None,
                        help="Baseline energies in eV (e.g. -13.6 -75.0). Highest priority override; "
                             "if not set, prefer checkpoint E0 when available.")
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Override checkpoint cutoff radius (Angstrom). If not set, restore from checkpoint metadata.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--embed-size", nargs="+", type=int, default=None,
                        help="Override checkpoint embed_size. If not set, restore from checkpoint metadata.")
    parser.add_argument("--output-size", type=int, default=None,
                        help="Override checkpoint output_size. If not set, auto-read from checkpoint metadata.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output .pt file (default: <checkpoint>-mliap.pt)")
    parser.add_argument("--tensor-product-mode", type=str, default=None,
                        choices=["spherical", "spherical-save", "spherical-save-cue", "pure-cartesian-ictd", "pure-cartesian-ictd-save"],
                        help="Override tensor_product_mode. If not set, restore from checkpoint metadata, else fall back to spherical. "
                             "spherical-save-cue uses cuEquivariance for GPU acceleration.")
    parser.add_argument("--num-interaction", type=int, default=None,
                        help="Override checkpoint num_interaction (applies to all tensor-product-mode). "
                             "If not set, restore from checkpoint metadata.")
    parser.add_argument("--torchscript", action="store_true",
                        help="Trace model to TorchScript before export. Auto-enabled for spherical-save-cue.")
    args = parser.parse_args()

    from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

    effective_mode = args.tensor_product_mode
    if effective_mode is None:
        ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        effective_mode = ckpt_peek.get("tensor_product_mode", "spherical")
        del ckpt_peek

    use_torchscript = bool(args.torchscript)
    if effective_mode == "spherical-save-cue" and not use_torchscript:
        print("[export_mliap] Auto-enabling TorchScript export for spherical-save-cue")
        use_torchscript = True

    mliap_obj = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=args.checkpoint,
        element_types=args.elements,
        max_radius=args.max_radius,
        atomic_energy_keys=args.atomic_energy_keys,
        atomic_energy_values=args.atomic_energy_values,
        device=args.device,
        embed_size=args.embed_size,
        output_size=args.output_size,
        tensor_product_mode=args.tensor_product_mode,
        num_interaction=args.num_interaction,
        torchscript=use_torchscript,
    )

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(args.checkpoint)[0]
        out_path = f"{base}-mliap.pt"

    torch.save(mliap_obj, out_path)
    print(f"Exported ML-IAP unified object to: {out_path}")
    print(f"  Elements: {args.elements}")
    print(f"  Cutoff:   {mliap_obj.rcutfac} Angstrom")
    print()
    print("LAMMPS input example:")
    print(f"  neighbor 1.0 bin   # reduce ghost inflation for ML-IAP")
    print(f"  pair_style mliap unified {os.path.basename(out_path)} 0")
    print(f"  pair_coeff * * {' '.join(args.elements)}")


if __name__ == "__main__":
    main()
