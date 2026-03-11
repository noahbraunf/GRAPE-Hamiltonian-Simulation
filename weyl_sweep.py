#!/usr/bin/env python3
"""Weyl chamber sweep: compare direct, mediator, and combined generator sets."""
import argparse
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np

from grape_lib import (
    build_combined_basis,
    build_direct_basis_lifted,
    build_multiqubit_mediator_basis,
    build_pauli_basis,
    build_qudit_mediated_basis,
    build_weyl_grid,
    build_weyl_target,
    gate_fidelity,
    optimize_gate,
)


def get_generators_and_target(
    config: str, d_qudit: int, c1: float, c2: float, c3: float
):
    """Return (generators, target_unitary, labels) for a given configuration."""
    if config == "direct":
        gens, labels = build_pauli_basis()
        target = build_weyl_target(c1, c2, c3, d_qudit=1)
    elif config == "mediator":
        gens, labels = build_qudit_mediated_basis(d_qudit)
        target = build_weyl_target(c1, c2, c3, d_qudit=d_qudit)
    elif config == "combined":
        gens, labels = build_combined_basis(d_qudit)
        target = build_weyl_target(c1, c2, c3, d_qudit=d_qudit)
    elif config.startswith("multiqubit_"):
        n_anc = int(config.split("_")[1])
        gens, labels = build_multiqubit_mediator_basis(n_anc)
        target = build_weyl_target(c1, c2, c3, d_qudit=2**n_anc)
    else:
        raise ValueError(f"Unknown config: {config}")
    return gens, target, labels


def _d_label(config: str, d_qudit: int) -> int:
    """Canonical d label for filenames. Direct config always uses d=1."""
    if config == "direct":
        return 1
    if config.startswith("multiqubit_"):
        n_anc = int(config.split("_")[1])
        return 2**n_anc
    return d_qudit


def mode_run(args):
    """Run a single (config, d_qudit, weyl_idx, T_idx) optimization."""
    grid = build_weyl_grid(args.weyl_step_denom)
    T_values = jnp.linspace(args.T_min, args.T_max, args.n_T)

    task_id = int(
        os.environ.get("SLURM_ARRAY_TASK_ID", args.weyl_idx * args.n_T + args.T_idx)
    )
    weyl_idx = task_id // args.n_T
    T_idx = task_id % args.n_T

    c1, c2, c3 = grid[weyl_idx]
    T = float(T_values[T_idx])

    generators, U_target, labels = get_generators_and_target(
        args.config, args.d_qudit, c1, c2, c3
    )

    d_lab = _d_label(args.config, args.d_qudit)
    print(
        f"[run] config={args.config} d={d_lab} "
        f"weyl={weyl_idx} ({c1:.4f},{c2:.4f},{c3:.4f}) "
        f"T_idx={T_idx} T={T:.4f} "
        f"n_gen={generators.shape[0]} dim={generators.shape[1]} "
        f"M={args.M} restarts={args.n_restarts} epochs={args.epochs}"
    )

    best_params, fidelity, _, all_loss_histories = optimize_gate(
        generators,
        U_target,
        n_steps=args.M,
        time=T,
        n_restarts=args.n_restarts,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"[run] done  F={fidelity:.8f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"{args.config}_d{d_lab}_w{weyl_idx:02d}_T{T_idx:04d}.npz",
    )
    np.savez(
        out_path,
        config=args.config,
        d_qudit=args.d_qudit,
        weyl_idx=weyl_idx,
        c1=c1,
        c2=c2,
        c3=c3,
        T=T,
        T_idx=T_idx,
        fidelity=fidelity,
        best_omegas=np.array(best_params),
        all_loss_histories=np.array(all_loss_histories),
        M=args.M,
        n_generators=generators.shape[0],
    )
    print(f"[run] saved → {out_path}")


def mode_aggregate(args):
    """Aggregate all run files for a (config, d_qudit) into a single sweep file."""
    d_lab = _d_label(args.config, args.d_qudit)
    pattern = os.path.join(
        args.output_dir, f"{args.config}_d{d_lab}_w*_T*.npz"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    print(f"[aggregate] found {len(files)} files for config={args.config} d={d_lab}")

    records = [np.load(f) for f in files]

    grid = build_weyl_grid(args.weyl_step_denom)
    n_weyl = len(grid)
    T_values = np.linspace(args.T_min, args.T_max, args.n_T)

    fidelities = np.full((n_weyl, args.n_T), np.nan)
    best_omegas = {}
    for r in records:
        wi = int(r["weyl_idx"])
        ti = int(r["T_idx"])
        fidelities[wi, ti] = float(r["fidelity"])
        best_omegas[(wi, ti)] = r["best_omegas"]

    n_filled = int(np.sum(~np.isnan(fidelities)))
    n_total = n_weyl * args.n_T
    print(f"[aggregate] filled {n_filled}/{n_total} grid points")

    out_path = os.path.join(
        args.output_dir, f"{args.config}_d{d_lab}_sweep.npz"
    )
    save_dict = dict(
        config=args.config,
        d_qudit=args.d_qudit,
        weyl_coords=np.array(grid),
        T_values=T_values,
        fidelities=fidelities,
        T_min=args.T_min,
        T_max=args.T_max,
        n_T=args.n_T,
        weyl_step_denom=args.weyl_step_denom,
    )
    for (wi, ti), omegas in best_omegas.items():
        save_dict[f"omegas_w{wi:02d}_T{ti:04d}"] = omegas
    np.savez(out_path, **save_dict)
    print(f"[aggregate] saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weyl chamber sweep experiments")
    parser.add_argument("--mode", choices=["run", "aggregate"], default="run")
    parser.add_argument(
        "--config",
        choices=[
            "direct", "mediator", "combined",
            "multiqubit_1", "multiqubit_2", "multiqubit_3",
        ],
        required=True,
    )
    parser.add_argument("--d_qudit", type=int, default=2)
    parser.add_argument(
        "--weyl_step_denom", type=int, default=8,
        help="Grid step = pi/N. 8 gives 10 points, 16 gives 35.",
    )
    parser.add_argument("--weyl_idx", type=int, default=0)
    parser.add_argument("--T_idx", type=int, default=0)
    parser.add_argument("--T_min", type=float, default=0.05)
    parser.add_argument("--T_max", type=float, default=4.0)
    parser.add_argument("--n_T", type=int, default=20)
    parser.add_argument("--M", type=int, default=40)
    parser.add_argument("--n_restarts", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/weyl")

    args = parser.parse_args()

    if args.mode == "run":
        mode_run(args)
    elif args.mode == "aggregate":
        mode_aggregate(args)
