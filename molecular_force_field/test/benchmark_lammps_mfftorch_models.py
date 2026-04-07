#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import median
import math


AVOGADRO = 6.02214076e23
ANGSTROM3_PER_CM3 = 1.0e24
ATOMIC_MASS = {
    1: 1.008,
    2: 15.999,
}


LOOP_RE = re.compile(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def generate_positions(n1: int, n2: int, box: float, min_separation: float, seed: int) -> list[tuple[int, float, float, float]]:
    rng = random.Random(seed)
    positions: list[tuple[int, float, float, float]] = []
    min_sep2 = max(min_separation, 0.0) ** 2
    max_attempts_per_atom = 20000

    atom_types = [1] * max(n1, 0) + [2] * max(n2, 0)

    def periodic_d2(a: tuple[int, float, float, float], b: tuple[int, float, float, float]) -> float:
        dx = a[1] - b[1]
        dy = a[2] - b[2]
        dz = a[3] - b[3]
        dx -= round(dx / box) * box
        dy -= round(dy / box) * box
        dz -= round(dz / box) * box
        return dx * dx + dy * dy + dz * dz

    for atype in atom_types:
        placed = False
        for _ in range(max_attempts_per_atom):
            cand = (atype, rng.random() * box, rng.random() * box, rng.random() * box)
            if min_sep2 > 0.0 and any(periodic_d2(cand, prev) < min_sep2 for prev in positions):
                continue
            positions.append(cand)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place atoms; enlarge box or reduce min separation")
    return positions


def neighbor_stats(positions: list[tuple[int, float, float, float]], box: tuple[float, float, float], cutoff: float) -> tuple[int, float]:
    pairs = 0
    n = len(positions)
    cutoff2 = cutoff * cutoff
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][1] - positions[j][1]
            dy = positions[i][2] - positions[j][2]
            dz = positions[i][3] - positions[j][3]
            dx -= round(dx / box[0]) * box[0]
            dy -= round(dy / box[1]) * box[1]
            dz -= round(dz / box[2]) * box[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= cutoff2:
                pairs += 1
    avg_neigh = (2.0 * pairs / n) if n else 0.0
    return pairs, avg_neigh


def read_lammps_atomic_positions(path: Path) -> tuple[list[tuple[int, float, float, float]], tuple[float, float, float]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    xlo = xhi = ylo = yhi = zlo = zhi = None
    atoms_start = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            xlo, xhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            ylo, yhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            zlo, zhi = float(parts[0]), float(parts[1])
        elif line.strip().startswith("Atoms"):
            atoms_start = i + 2
            break
    if None in (xlo, xhi, ylo, yhi, zlo, zhi) or atoms_start is None:
        raise RuntimeError(f"failed to parse LAMMPS data file {path}")

    boxx = float(xhi) - float(xlo)
    boxy = float(yhi) - float(ylo)
    boxz = float(zhi) - float(zlo)
    box = (boxx, boxy, boxz)

    positions: list[tuple[int, float, float, float]] = []
    for line in lines[atoms_start:]:
        s = line.strip()
        if not s:
            if positions:
                break
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        # full atom_style: id mol type q x y z ...
        lammps_type = int(parts[2])
        x = float(parts[4]) - float(xlo)
        y = float(parts[5]) - float(ylo)
        z = float(parts[6]) - float(zlo)
        # model expects pair_coeff order H O, so remap: H->1, O->2
        model_type = 2 if lammps_type == 1 else 1
        positions.append((model_type, x, y, z))
    if not positions:
        raise RuntimeError(f"no atoms parsed from {path}")
    return positions, box


def write_input(path: Path, core_pt: Path, positions: list[tuple[int, float, float, float]], cutoff: float, box: tuple[float, float, float], warmup: int, steps: int) -> None:
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "",
        f"region box block 0 {box[0]:.6f} 0 {box[1]:.6f} 0 {box[2]:.6f}",
        "create_box 2 box",
    ]
    for atype, x, y, z in positions:
        lines.append(f"create_atoms {atype} single {x:.8f} {y:.8f} {z:.8f} units box")
    lines += [
        "mass 1 1.008",
        "mass 2 15.999",
        "",
        "neighbor 1.0 bin",
        "",
        f"pair_style mff/torch {cutoff:.6f} cuda",
        f"pair_coeff * * {core_pt} H O",
        "",
        "velocity all create 300 42",
        "fix 1 all nve",
        "thermo 10",
        f"run {warmup}",
        "reset_timestep 0",
        f"run {steps}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_last_loop(text: str) -> dict[str, float]:
    matches = list(LOOP_RE.finditer(text))
    if not matches:
        raise RuntimeError("failed to parse LAMMPS loop time")
    m = matches[-1]
    loop = float(m.group(1))
    procs = int(m.group(2))
    steps = int(m.group(3))
    atoms = int(m.group(4))
    return {
        "loop_s": loop,
        "procs": procs,
        "steps": steps,
        "atoms": atoms,
        "steps_per_s": steps / loop,
        "katom_steps_per_s": (steps * atoms) / loop / 1000.0,
    }


def water_density_box(n1: int, n2: int, density_g_cm3: float) -> float:
    total_mass_g_per_mol = n1 * ATOMIC_MASS[1] + n2 * ATOMIC_MASS[2]
    volume_cm3_per_mol = total_mass_g_per_mol / density_g_cm3
    volume_a3 = volume_cm3_per_mol / AVOGADRO * ANGSTROM3_PER_CM3
    return volume_a3 ** (1.0 / 3.0)


def run_case(lmp: Path, torch_lib: Path, core_pt: Path, n1: int, n2: int, cutoff: float, box: tuple[float, float, float], min_separation: float, warmup: int, steps: int, seed: int, base_positions: list[tuple[int, float, float, float]] | None = None, keep_dir: Path | None = None) -> dict[str, float]:
    positions = base_positions if base_positions is not None else generate_positions(n1=n1, n2=n2, box=max(box), min_separation=min_separation, seed=seed)
    if keep_dir is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="bench-mfftorch-"))
    else:
        tmpdir = keep_dir
        tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        in_file = tmpdir / "in.mfftorch"
        write_input(in_file, core_pt=core_pt, positions=positions, cutoff=cutoff, box=box, warmup=warmup, steps=steps)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        proc = subprocess.run(
            [str(lmp), "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "off", "neigh", "full", "-in", str(in_file)],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        (tmpdir / "run.log").write_text(proc.stdout, encoding="utf-8")
        return parse_last_loop(proc.stdout)
    finally:
        if keep_dir is None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark LAMMPS USER-MFFTORCH throughput for multiple core.pt models.")
    ap.add_argument("--lmp", type=Path, required=True)
    ap.add_argument("--torch-lib", type=Path, default=Path("/usr/local/lib/python3.10/dist-packages/torch/lib"))
    ap.add_argument("--cue-core", type=Path, required=True)
    ap.add_argument("--ictd-core", type=Path, required=True)
    ap.add_argument("--ictd-save-core", type=Path, required=True)
    ap.add_argument("--n1", type=int, default=512)
    ap.add_argument("--n2", type=int, default=256)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--box", type=float, default=0.0, help="0 means auto from atom count")
    ap.add_argument("--density-g-cm3", type=float, default=0.0, help="If > 0, compute cubic box from total H/O mass and target density.")
    ap.add_argument("--lammps-data", type=Path, default=None, help="Use atom coordinates from a LAMMPS data file instead of random placement.")
    ap.add_argument("--min-separation", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    if args.lammps_data is not None:
        base_positions, box = read_lammps_atomic_positions(args.lammps_data)
        total_atoms = len(base_positions)
        args.n1 = sum(1 for t, *_ in base_positions if t == 1)
        args.n2 = sum(1 for t, *_ in base_positions if t == 2)
    else:
        total_atoms = args.n1 + args.n2
        if args.density_g_cm3 > 0:
            a = water_density_box(args.n1, args.n2, args.density_g_cm3)
            box = (a, a, a)
        else:
            a = args.box if args.box > 0 else max(60.0, 3.0 * (total_atoms ** (1.0 / 3.0)) * 2.5)
            box = (a, a, a)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.lammps_data is not None:
        preview_positions = base_positions
    else:
        preview_positions = generate_positions(n1=args.n1, n2=args.n2, box=max(box), min_separation=args.min_separation, seed=args.seed)
    pair_count, avg_neigh = neighbor_stats(preview_positions, box=box, cutoff=args.cutoff)

    models = {
        "spherical-save-cue": args.cue_core,
        "ictd": args.ictd_core,
        "ictd-save": args.ictd_save_core,
    }
    results: dict[str, dict[str, float]] = {}
    for name, core in models.items():
        per_run = []
        for run_idx in range(args.runs):
            keep_dir = None
            if args.keep_artifacts:
                keep_dir = Path(f"/home/rebuild/.bench_mfftorch_{name.replace('/', '_')}_{run_idx}")
            metrics = run_case(
                lmp=args.lmp,
                torch_lib=args.torch_lib,
                core_pt=core,
                n1=args.n1,
                n2=args.n2,
                cutoff=args.cutoff,
                box=box,
                min_separation=args.min_separation,
                warmup=args.warmup,
                steps=args.steps,
                seed=args.seed,
                base_positions=(base_positions if args.lammps_data is not None else None),
                keep_dir=keep_dir,
            )
            per_run.append(metrics)
            print(f"{name} run{run_idx+1}: loop={metrics['loop_s']:.6f}s katom-step/s={metrics['katom_steps_per_s']:.6f}", flush=True)
        results[name] = {
            "loop_s": median(x["loop_s"] for x in per_run),
            "steps_per_s": median(x["steps_per_s"] for x in per_run),
            "katom_steps_per_s": median(x["katom_steps_per_s"] for x in per_run),
            "atoms": per_run[0]["atoms"],
            "steps": per_run[0]["steps"],
        }

    ranking = sorted(results.items(), key=lambda kv: kv[1]["katom_steps_per_s"], reverse=True)
    lines = []
    lines.append(
        f"config atoms={total_atoms} n1={args.n1} n2={args.n2} box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f}) cutoff={args.cutoff} "
        f"warmup={args.warmup} steps={args.steps} runs={args.runs} density_g_cm3={args.density_g_cm3:.6f} "
        f"avg_neighbors={avg_neigh:.6f} pairs={pair_count}"
    )
    for name, metrics in ranking:
        lines.append(
            f"{name}\tloop_s={metrics['loop_s']:.6f}\tsteps_per_s={metrics['steps_per_s']:.6f}\tkatom_steps_per_s={metrics['katom_steps_per_s']:.6f}"
        )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
