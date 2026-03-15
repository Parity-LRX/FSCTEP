from __future__ import annotations

import argparse
import os
from typing import Sequence

import h5py
import numpy as np


def _sample_keys(handle: h5py.File) -> list[str]:
    keys = [k for k in handle.keys() if k.startswith("sample_")]
    return sorted(keys, key=lambda x: int(x.split("_", 1)[1]))


def merge_processed_h5_with_fidelity(
    inputs: Sequence[str],
    fidelity_ids: Sequence[int],
    output_h5: str,
    output_fidelity_npy: str,
) -> np.ndarray:
    if len(inputs) == 0:
        raise ValueError("At least one input H5 file is required")
    if len(inputs) != len(fidelity_ids):
        raise ValueError("inputs and fidelity_ids must have the same length")

    merged_ids: list[int] = []
    os.makedirs(os.path.dirname(os.path.abspath(output_h5)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_fidelity_npy)) or ".", exist_ok=True)

    with h5py.File(output_h5, "w") as out_h5:
        out_index = 0
        for path, fid in zip(inputs, fidelity_ids):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input H5 not found: {path}")
            fid_int = int(fid)
            if fid_int < 0:
                raise ValueError(f"fidelity_id must be >= 0, got {fid_int} for {path}")
            with h5py.File(path, "r") as in_h5:
                for key in _sample_keys(in_h5):
                    in_h5.copy(in_h5[key], out_h5, name=f"sample_{out_index}")
                    merged_ids.append(fid_int)
                    out_index += 1

    merged = np.asarray(merged_ids, dtype=np.int64)
    np.save(output_fidelity_npy, merged)
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple processed_*.h5 datasets and generate graph-level fidelity_id.npy"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Input processed H5 files, e.g. processed_low.h5 processed_high.h5",
    )
    parser.add_argument(
        "--fidelity-ids",
        type=int,
        nargs="+",
        required=True,
        help="Per-input fidelity id, one integer per --inputs entry, e.g. 0 1",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        required=True,
        help="Output merged processed H5 path",
    )
    parser.add_argument(
        "--output-fidelity-npy",
        type=str,
        required=True,
        help="Output fidelity_id.npy path aligned with merged sample order",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    merged = merge_processed_h5_with_fidelity(
        inputs=args.inputs,
        fidelity_ids=args.fidelity_ids,
        output_h5=args.output_h5,
        output_fidelity_npy=args.output_fidelity_npy,
    )
    print(
        f"Merged {len(args.inputs)} inputs into {args.output_h5} with "
        f"{merged.shape[0]} samples; wrote fidelity ids to {args.output_fidelity_npy}"
    )


if __name__ == "__main__":
    main()
