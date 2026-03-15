from __future__ import annotations

import h5py
import numpy as np

from molecular_force_field.cli.merge_multifidelity_h5 import merge_processed_h5_with_fidelity


def _write_processed_h5(path, values: list[float]) -> None:
    with h5py.File(path, "w") as h5:
        for idx, value in enumerate(values):
            g = h5.create_group(f"sample_{idx}")
            g.create_dataset("pos", data=np.full((1, 3), value, dtype=np.float64))
            g.create_dataset("A", data=np.array([1], dtype=np.int64))
            g.create_dataset("y", data=np.array(value, dtype=np.float64))
            g.create_dataset("force", data=np.zeros((1, 3), dtype=np.float64))
            g.create_dataset("edge_src", data=np.zeros((0,), dtype=np.int64))
            g.create_dataset("edge_dst", data=np.zeros((0,), dtype=np.int64))
            g.create_dataset("edge_shifts", data=np.zeros((0, 3), dtype=np.float64))
            g.create_dataset("cell", data=np.eye(3, dtype=np.float64))


def test_merge_processed_h5_with_fidelity(tmp_path) -> None:
    low = tmp_path / "processed_low.h5"
    high = tmp_path / "processed_high.h5"
    merged = tmp_path / "processed_merged.h5"
    fidelity = tmp_path / "fidelity_id.npy"
    _write_processed_h5(low, [1.0, 2.0])
    _write_processed_h5(high, [10.0])

    merged_ids = merge_processed_h5_with_fidelity(
        inputs=[str(low), str(high)],
        fidelity_ids=[0, 1],
        output_h5=str(merged),
        output_fidelity_npy=str(fidelity),
    )

    assert merged_ids.tolist() == [0, 0, 1]
    assert np.load(fidelity).tolist() == [0, 0, 1]
    with h5py.File(merged, "r") as h5:
        assert sorted(h5.keys()) == ["sample_0", "sample_1", "sample_2"]
        assert h5["sample_0"]["y"][()] == 1.0
        assert h5["sample_1"]["y"][()] == 2.0
        assert h5["sample_2"]["y"][()] == 10.0
