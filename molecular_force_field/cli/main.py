from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mff",
        description="Unified FSCETP CLI entrypoint",
        allow_abbrev=False,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run the training CLI")
    group.add_argument("--evaluate", action="store_true", help="Run the evaluation CLI")
    group.add_argument("--preprocess", action="store_true", help="Run the preprocessing CLI")
    group.add_argument("--convert-dataset", action="store_true", help="Run the dataset conversion CLI")
    group.add_argument("--lammps", action="store_true", help="Run the LAMMPS interface CLI")
    group.add_argument("--export-core", action="store_true", help="Run the TorchScript core export CLI")
    group.add_argument("--evaluate-pes-coverage", action="store_true", help="Run the PES coverage evaluation CLI")
    group.add_argument("--active-learn", action="store_true", help="Run the active learning CLI")
    group.add_argument("--init-data", action="store_true", help="Run the data initialization CLI")
    group.add_argument("--merge-multifidelity", action="store_true", help="Merge processed H5 datasets and generate fidelity ids")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.train:
        from molecular_force_field.cli.train import main as submain
    elif args.evaluate:
        from molecular_force_field.cli.evaluate import main as submain
    elif args.preprocess:
        from molecular_force_field.cli.preprocess import main as submain
    elif args.convert_dataset:
        from molecular_force_field.cli.convert_dataset import main as submain
    elif args.lammps:
        from molecular_force_field.cli.lammps_interface import main as submain
    elif args.export_core:
        from molecular_force_field.cli.export_libtorch_core import main as submain
    elif args.evaluate_pes_coverage:
        from molecular_force_field.cli.evaluate_pes_coverage import main as submain
    elif args.active_learn:
        from molecular_force_field.cli.active_learning import main as submain
    elif args.init_data:
        from molecular_force_field.cli.init_data import main as submain
    elif args.merge_multifidelity:
        from molecular_force_field.cli.merge_multifidelity_h5 import main as submain
    else:  # pragma: no cover
        parser.error("one subcommand flag is required")
        return

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        submain()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
