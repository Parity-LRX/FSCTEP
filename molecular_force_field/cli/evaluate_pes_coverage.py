"""CLI for PES coverage evaluation."""

import argparse
import logging
import os

from molecular_force_field.active_learning.pes_coverage import evaluate_pes_coverage


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PES (Potential Energy Surface) coverage of a dataset using SOAP descriptors."
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Path to dataset XYZ file. If not set, uses --data-dir/train.xyz or first XYZ in data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Data directory (used when --dataset-file not set)",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="Optional path to reference structures XYZ (e.g. exploration trajectory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pes_coverage_report.json",
        help="Output report path (default: pes_coverage_report.json)",
    )
    parser.add_argument(
        "--soap-rcut",
        type=float,
        default=5.0,
        help="SOAP cutoff radius in Å (default: 5.0)",
    )
    parser.add_argument(
        "--soap-nmax",
        type=int,
        default=8,
        help="SOAP radial basis count (default: 8)",
    )
    parser.add_argument(
        "--soap-lmax",
        type=int,
        default=6,
        help="SOAP spherical harmonics max degree (default: 6)",
    )
    parser.add_argument(
        "--r-cov",
        type=float,
        default=0.5,
        help="Coverage threshold for reference set (default: 0.5)",
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Treat structures as periodic",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for SOAP (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset_path = args.dataset_file
    if dataset_path is None:
        dataset_path = args.data_dir
        train_xyz = os.path.join(args.data_dir, "train.xyz")
        if os.path.exists(train_xyz):
            dataset_path = train_xyz

    report = evaluate_pes_coverage(
        dataset_path=dataset_path,
        reference_path=args.reference_file,
        output_path=args.output,
        soap_rcut=args.soap_rcut,
        soap_nmax=args.soap_nmax,
        soap_lmax=args.soap_lmax,
        r_cov=args.r_cov,
        periodic=args.periodic,
        n_jobs=args.n_jobs,
    )

    print(f"Report saved to {args.output}")
    if "coverage" in report:
        print(f"Coverage: {report['coverage']:.2%}, fill_distance: {report['fill_distance']:.4f}")
    else:
        print(f"k-NN mean distance: {report.get('knn_mean_distance', 0):.4f}")


if __name__ == "__main__":
    main()
