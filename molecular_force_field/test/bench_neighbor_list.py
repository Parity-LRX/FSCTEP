"""
ASE vs matscipy 邻域列表速度对比。

测试不同体系大小（原子数）和截断半径下的单帧计算时间。
"""
import time
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list


def make_bulk(n_repeat: int):
    """生成简单立方 FCC-like 体系，原子数 ≈ 4 * n_repeat^3。"""
    a = 3.6  # Å, Cu-like lattice constant
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]) * a
    positions = []
    for ix in range(n_repeat):
        for iy in range(n_repeat):
            for iz in range(n_repeat):
                offset = np.array([ix, iy, iz]) * a
                for b in basis:
                    positions.append(b + offset)
    positions = np.array(positions, dtype=np.float64)
    cell = np.eye(3) * a * n_repeat
    atom_types = np.full(len(positions), 29, dtype=np.int64)  # Cu
    return positions, atom_types, cell


def bench_ase(pos, atom_types, cell, pbc, cutoff, n_runs=5):
    atoms = Atoms(numbers=atom_types, positions=pos, cell=cell, pbc=pbc)
    # warmup
    ase_neighbor_list('ijS', atoms, cutoff)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        i, j, S = ase_neighbor_list('ijS', atoms, cutoff)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    n_edges = len(i)
    return np.median(times), n_edges


def bench_matscipy(pos, atom_types, cell, pbc, cutoff, n_runs=5):
    # warmup
    matscipy_neighbour_list('ijS', positions=pos, cell=cell, pbc=pbc, cutoff=cutoff)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        i, j, S = matscipy_neighbour_list('ijS', positions=pos, cell=cell, pbc=pbc, cutoff=cutoff)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    n_edges = len(i)
    return np.median(times), n_edges


def main():
    cutoffs = [5.0, 8.0]
    repeats = [3, 5, 7, 10]  # → ~108, 500, 1372, 4000 atoms
    pbc = [True, True, True]

    print(f"{'Atoms':>8} {'Cutoff':>8} {'Edges':>10} {'ASE (ms)':>10} {'matscipy (ms)':>14} {'Speedup':>8}")
    print("-" * 70)

    for n_rep in repeats:
        pos, atom_types, cell = make_bulk(n_rep)
        n_atoms = len(pos)
        for cutoff in cutoffs:
            t_ase, n_edges_ase = bench_ase(pos, atom_types, cell, pbc, cutoff)
            t_mat, n_edges_mat = bench_matscipy(pos, atom_types, cell, pbc, cutoff)
            assert n_edges_ase == n_edges_mat, (
                f"Edge count mismatch! ASE={n_edges_ase}, matscipy={n_edges_mat}"
            )
            speedup = t_ase / t_mat if t_mat > 0 else float('inf')
            print(f"{n_atoms:>8} {cutoff:>8.1f} {n_edges_ase:>10} {t_ase*1000:>10.2f} {t_mat*1000:>14.2f} {speedup:>8.2f}x")

    print()
    print("Speedup = ASE_time / matscipy_time")


if __name__ == "__main__":
    main()
