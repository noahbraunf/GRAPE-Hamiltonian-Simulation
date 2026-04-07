import jax
import jax.numpy as jnp
import optax
from jax import lax


def generate_spin_operators(levels: int):
    s = (levels - 1) / 2
    ms = jnp.arange(s, -s - 1, -1)

    Sp_coeffs = jnp.sqrt(s * (s + 1) - ms[1:] * (ms[1:] + 1))
    Sp = jnp.diag(Sp_coeffs, k=1)
    Sm = Sp.conj().T

    Sz = jnp.diag(ms)
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / 2j

    return {"I": jnp.eye(levels), "x": Sx, "y": Sy, "z": Sz}


pauli_operators = {
    "I": jnp.eye(2),
    "x": jnp.array([[0, 1], [1, 0]]),
    "y": jnp.array([[0, -1j], [1j, 0]]),
    "z": jnp.array([[1, 0], [0, -1]]),
}


def build_pauli_basis():
    """15 two-qubit Pauli tensor products {sigma^a (x) sigma^b}, excluding I(x)I."""
    generators, labels = [], []
    for a, sa in pauli_operators.items():
        for b, sb in pauli_operators.items():
            if a == "I" and b == "I":
                continue
            generators.append(jnp.kron(sa, sb))
            labels.append(f"\u03c3{a}\u2297\u03c3{b}")
    return jnp.stack(generators), labels


def build_qudit_mediated_basis(d_qudit: int):
    """Hermitian generators for two qubits coupled to a d_qudit-level qudit.

    Hilbert space: qubit1 (x) qubit2 (x) qudit.
    Includes all sigma_a (x) I (x) S_b and I (x) sigma_a (x) S_b terms,
    excluding the global identity (I_2 (x) I_2 (x) I_d).
    Deduplicates I (x) I (x) S_b terms that appear for both qubit slots.
    Yields 27 generators for all supported d values.
    """
    I2 = jnp.eye(2, dtype=jnp.complex64)
    if d_qudit == 2:
        spin_ops = pauli_operators
    else:
        spin_ops = generate_spin_operators(d_qudit)

    generators, labels = [], []
    seen = set()

    for qubit_idx in [1, 2]:
        for a, sigma_a in pauli_operators.items():
            for b, S_b in spin_ops.items():
                if a == "I" and b == "I":
                    continue  # global identity does not change state
                # I (x) I (x) S_b is qudit-only; same regardless of qubit_idx
                if qubit_idx == 1:
                    label = (
                        f"$I \\otimes I \\otimes S{b}$"
                        if a == "I"
                        else f"$\\sigma^{a}_{qubit_idx} \\otimes I \\otimes S{b}$"
                    )
                elif qubit_idx == 2:
                    label = (
                        f"$I \\otimes I \\otimes S{b}$"
                        if a == "I"
                        else f"$\\sigma^{a}_{qubit_idx} \\otimes I \\otimes S{b}$"
                    )
                else:
                    raise ValueError(f"Invalid qubit index: {qubit_idx}")
                if label in seen:  # duplicate
                    continue
                seen.add(label)
                if qubit_idx == 1:
                    op = _tensor_product_all([sigma_a, I2, S_b])
                else:
                    op = _tensor_product_all([I2, sigma_a, S_b])
                generators.append(op)
                labels.append(label)

    return jnp.stack(generators), labels


def build_swap_qudit_target(d_qudit: int) -> jax.Array:
    """SWAP (x) I_d: SWAP on the two qubits, identity on the qudit."""
    U_swap = jnp.zeros((4, 4), dtype=jnp.complex64)
    for i in range(2):
        for j in range(2):
            U_swap = U_swap.at[j * 2 + i, i * 2 + j].set(1.0)
    return jnp.kron(U_swap, jnp.eye(d_qudit, dtype=jnp.complex64))


def _tensor_product_all(ops):
    from math import prod

    n = len(ops)

    inputs = []
    outputs = []

    for k in range(n):
        i = 2 * k
        j = 2 * k + 1
        inputs.extend((ops[k], [i, j]))  # (A, (0, 1)), (B, (2, 3)
        outputs.append(i)

    for k in range(n):
        outputs.append(2 * k + 1)

    # def _scan_product(carry, op):
    #     return carry * op.shape[0], _

    # dimension, _ = jax.lax.scan(_scan_product, 1, ops)
    dimension = prod(op.shape[0] for op in ops)

    return jnp.einsum(*inputs, outputs).reshape(dimension, dimension)


def hamiltonian_generators_spin_mediated(n_qubits=2, mediator_dimension: int = 2):
    generators = []
    labels = []
    spin_operators = generate_spin_operators(mediator_dimension)

    for qubit in range(n_qubits):
        for p_label, pauli in pauli_operators.items():
            for s_label, spin in spin_operators.items():
                if p_label != "I" and s_label != "I":
                    operators = [jnp.eye(2) for _ in range(n_qubits)] + [spin]
                    operators[qubit] = pauli

                    cur_labels = ["I_2" for _ in range(n_qubits)] + [
                        f"I_{mediator_dimension}" if s_label == "I" else "S^" + s_label
                    ]
                    cur_labels[qubit] = (
                        "I_2" if p_label == "I" else "\\sigma^" + p_label
                    )

                    generators.append(_tensor_product_all(operators))
                    labels.append("$" + " \\otimes ".join(cur_labels) + "$")
    return jnp.stack(generators), labels


@jax.jit
def gate_fidelity(U, V):
    d = U.shape[-1]
    overlap = jnp.einsum(
        "...ij,...ij->", jnp.conj(U), V
    )  # hilbert schmidt inner product
    return (jnp.abs(overlap) ** 2 + d) / (d * (d + 1))  # average gate fidelity


def evolve_H(
    coupling_strengths: jax.Array,
    generators: jax.Array,
    time,
) -> jax.Array:
    """Create unitary which is derived from coupling strengths and generators (i.e. the Hamiltonian)

    Args:
        coupling_strengths (jax.Array[time_step, id]): each coupling strength for a generator of the unitary
        generators (jax.Array[id, N, N]): each generator of the Hamiltonian
        time: Time to evolve Hamiltonian over
        constraint_fn: pure function which is a constraint applied to coupling_strengths

    Returns:
        jax.Array[N, N]
    """

    dt = time / coupling_strengths.shape[0]

    H_all_timesteps = jnp.einsum("ij,jkl->ikl", coupling_strengths, generators)

    U_steps = jax.vmap(jax.scipy.linalg.expm)(-1j * H_all_timesteps * dt)

    U_prefix = jax.lax.associative_scan(lambda u0, u1: u1 @ u0, U_steps)
    return U_prefix[-1]


def evolve_H_trotter(
    coupling_strengths: jax.Array,
    generators: jax.Array,
    time,
) -> jax.Array:
    """First-order Trotter approximation of simultaneous evolution.

    U = prod_t prod_j exp(-i * omega_j(t) * G_j * dt),  dt = T/M

    Each macro step approximates exp(-i * sum_j omega_j G_j * dt) by
    splitting into n_gen sequential factors of duration dt each.  In the
    M -> infinity limit this reproduces evolve_H exactly.
    """
    d = generators.shape[1]
    dt = time / coupling_strengths.shape[0]
    sub_dt = dt

    def single_trotter_step(omega_t):
        def apply_gen(U, gen_omega):
            g, w = gen_omega
            return jax.scipy.linalg.expm(-1j * w * g * sub_dt) @ U, None

        U_step, _ = jax.lax.scan(
            apply_gen, jnp.eye(d, dtype=generators.dtype), (generators, omega_t)
        )
        return U_step

    U_steps = jax.vmap(single_trotter_step)(coupling_strengths)
    U_total = jax.lax.associative_scan(lambda u0, u1: u1 @ u0, U_steps)
    return U_total[-1]


def normalize_frob_norm(generators: jax.Array, target_norm: float = 1.0):
    """Return a normalize_fn that projects ||H(t)||_F = target_norm per time step.

    H(t) = sum_j omega_j(t) G_j.  Rescales omega at each t so the Frobenius
    norm sqrt(trace(H^dagger H)) of the Hermitian H equals target_norm.
    """

    def _normalize(thetas: jax.Array) -> jax.Array:
        H = jnp.einsum("tj,jkl->tkl", thetas, generators)  # (M, d, d)
        norms = jnp.sqrt(jnp.sum(jnp.abs(H) ** 2, axis=(-2, -1)))  # (M,)
        scale = target_norm / jnp.maximum(norms, 1e-10)  # (M,)
        return thetas * scale[:, None]

    return _normalize


def optimize_gate(
    generators: jax.Array,
    U_target: jax.Array,
    n_steps,
    time,
    n_restarts=50,
    epochs=4000,
    lr=0.01,
    initial_scale=1.0,
    seed=42,
    normalize_fn=None,
    evolve_fn=None,
):
    optimizer = optax.adam(lr)
    if normalize_fn is None:
        normalize_fn = normalize_frob_norm(generators)
    if evolve_fn is None:
        evolve_fn = evolve_H

    def loss_fun(thetas: jax.Array):
        U = evolve_fn(normalize_fn(thetas), generators, time)
        return 1.0 - gate_fidelity(U, U_target)

    def single_restart(key):
        # currently this is not trotterized, so we either need to convert to trotterized form,
        # which can be done by instead of having a thetas be (n_steps, n_generators), it is theta(n_steps, generator_id)=coupling strength
        # we then have to map that into evolve_H, one hot style
        #
        # another potential way (easier, TODO: first)
        # is to bound the entire norm by 1
        # Frobenius vs. Operator TBD, but this is easy enough to change in sim
        thetas = initial_scale * jax.random.normal(key, (n_steps, generators.shape[0]))
        opt_state = optimizer.init(thetas)

        def step(carry, _):
            thetas, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fun)(thetas)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(thetas, updates), new_opt_state), loss

        (final_thetas, _), losses = lax.scan(
            step, (thetas, opt_state), None, length=epochs
        )
        return normalize_fn(final_thetas), losses

    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    all_params, all_losses = jax.vmap(single_restart)(keys)  # (n_restarts, epochs)
    best_id = jnp.argmin(all_losses[:, -1])
    return (
        all_params[best_id],
        1.0 - all_losses[best_id, -1],
        all_losses[best_id],
        all_losses,
    )


@jax.jit
def sweep_fidelity(
    generators,
    U_target,
    T_values,
    M,
    label,
    n_restarts=30,
    epochs=4000,
    lr=0.005,
    init_scale=0.3,
    verbose=False,
):
    def optimize_for_T(T):
        _, fid, _, _ = optimize_gate(
            generators, U_target, M, T, n_restarts, epochs, lr, init_scale
        )
        return fid

    return jax.vmap(optimize_for_T)(jnp.array(T_values))


if __name__ == "__main__":
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser(description="Qudit-mediated SWAP GRAPE sweep")
    parser.add_argument(
        "--mode",
        choices=["run", "aggregate", "check", "analyze", "test"],
        default="run",
    )
    # Physics
    parser.add_argument("--d_qudit", type=int, default=2, choices=[2, 3, 4, 5, 6])
    # Sweep grid
    parser.add_argument(
        "--T_idx",
        type=int,
        default=0,
        help="Index into T_values array (overridden by SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument("--T_min", type=float, default=0.3)
    parser.add_argument("--T_max", type=float, default=12.0)
    parser.add_argument("--n_T", type=int, default=50)
    # Optimizer
    parser.add_argument("--M", type=int, default=40, help="Number of time steps")
    parser.add_argument("--n_restarts", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize",
        choices=["frob_norm", "cosine"],
        default="frob_norm",
        help="Hamiltonian norm constraint: frob_norm (||H||_F=1) or cosine (w_j=cos(t_j))",
    )
    parser.add_argument(
        "--trotter",
        action="store_true",
        help="Use Trotterized evolution (sequential per-generator expm per step)",
    )
    # I/O
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # check mode
    if args.mode == "check":
        for d in [2, 4]:
            gens, labels = build_qudit_mediated_basis(d)
            target = build_swap_qudit_target(d)
            n, N = gens.shape[0], gens.shape[-1]
            max_err = float(
                max(jnp.max(jnp.abs(gens[k] - gens[k].conj().T)) for k in range(n))
            )
            print(f"d={d}: {n} generators, dim={N}, max Hermitian err={max_err:.2e}")
            assert N == 4 * d
        print("All checks passed.")
        raise SystemExit(0)

    # run mode
    if args.mode == "run":
        import numpy as np

        # SLURM_ARRAY_TASK_ID overrides --T_idx when running on cluster
        T_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.T_idx))
        T_values = jnp.linspace(args.T_min, args.T_max, args.n_T)
        T = float(T_values[T_idx])

        generators, labels = build_qudit_mediated_basis(args.d_qudit)
        U_target = build_swap_qudit_target(args.d_qudit)
        n_generators = generators.shape[0]

        print(
            f"[run] d={args.d_qudit}  T_idx={T_idx}  T={T:.4f}  "
            f"n_gen={n_generators}  M={args.M}  "
            f"restarts={args.n_restarts}  epochs={args.epochs}"
        )

        norm_fn = (
            (lambda thetas: jnp.cos(thetas)) if args.normalize == "cosine" else None
        )
        evol_fn = evolve_H_trotter if args.trotter else evolve_H
        best_params, fidelity, _best_hist, all_loss_histories = optimize_gate(
            generators,
            U_target,
            n_steps=args.M,
            time=T,
            n_restarts=args.n_restarts,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            normalize_fn=norm_fn,
            evolve_fn=evol_fn,
        )
        print(f"[run] done  F={fidelity:.8f}")

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"d{args.d_qudit}_T{T_idx:04d}.npz")
        np.savez(
            out_path,
            T=T,
            T_idx=T_idx,
            fidelity=fidelity,
            best_omegas=np.array(best_params),
            all_loss_histories=np.array(all_loss_histories),
            d_qudit=args.d_qudit,
            M=args.M,
            n_generators=n_generators,
        )
        print(f"[run] saved -> {out_path}")
    # aggregate mode
    elif args.mode == "aggregate":
        import numpy as np

        pattern = os.path.join(args.output_dir, f"d{args.d_qudit}_T*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern}")
        print(f"[aggregate] found {len(files)} files for d={args.d_qudit}")

        # Load and sort by T_idx
        records = [np.load(f) for f in files]
        records.sort(key=lambda r: int(r["T_idx"]))

        T_values_expected = jnp.linspace(args.T_min, args.T_max, args.n_T)
        for r in records:
            idx = int(r["T_idx"])
            expected_T = float(T_values_expected[idx])
            actual_T = float(r["T"])
            if abs(actual_T - expected_T) > 1e-4:
                raise ValueError(
                    f"T_idx={idx}: file has T={actual_T:.6f} but expected "
                    f"T={expected_T:.6f} from --T_min={args.T_min} "
                    f"--T_max={args.T_max} --n_T={args.n_T}. "
                    f"Ensure all run files used the same sweep parameters."
                )

        # Collect generator labels once
        _, labels = build_qudit_mediated_basis(args.d_qudit)

        M_vals = {int(r["M"]) for r in records}
        if len(M_vals) != 1:
            raise ValueError(f"Inconsistent --M across run files: {M_vals}")

        out_path = os.path.join(args.output_dir, f"d{args.d_qudit}_sweep.npz")
        np.savez(
            out_path,
            T_values=np.array([r["T"] for r in records]),
            fidelities=np.array([r["fidelity"] for r in records]),
            best_omegas=np.stack([r["best_omegas"] for r in records]),
            all_loss_histories=np.stack([r["all_loss_histories"] for r in records]),
            generator_labels=np.array(labels, dtype=object),
            d_qudit=args.d_qudit,
            M=int(records[0]["M"]),
            n_generators=int(records[0]["n_generators"]),
            n_restarts=args.n_restarts,
            epochs=args.epochs,
            lr=args.lr,
            T_min=args.T_min,
            T_max=args.T_max,
            n_T=args.n_T,
        )
        print(f"[aggregate] saved -> {out_path}")
        data = np.load(out_path, allow_pickle=True)
        print(f"\t\tT_values shape:\t{data['T_values'].shape}")
        print(f"\t\tfidelities shape:\t{data['fidelities'].shape}")
        print(f"\t\tbest_omegas shape:\t{data['best_omegas'].shape}")
        print(f"\t\tall_loss_hist:\t{data['all_loss_histories'].shape}")

    # analyze mode
    elif args.mode == "analyze":
        import numpy as np

        sweep_path = os.path.join(args.output_dir, f"d{args.d_qudit}_sweep.npz")
        data = np.load(sweep_path, allow_pickle=True)
        print(f"[analyze] loaded {sweep_path}")

        generators, _ = build_qudit_mediated_basis(args.d_qudit)
        T_values = jnp.array(data["T_values"])
        best_omegas = jnp.array(data["best_omegas"])  # (n_T, M, n_gen)
        M = int(data["M"])
        M_half = M // 2

        # Compute U_half for every T simultaneously via vmap
        def compute_U_half(omega, T):
            return evolve_H(omega[:M_half], generators, T * M_half / M)

        U_halves = jax.vmap(compute_U_half)(best_omegas, T_values)  # (n_T, N, N)
        print(f"[analyze] U_halves shape={U_halves.shape}  M_half={M_half}")

        out = dict(T_values=np.array(T_values), U_halves=np.array(U_halves))

        if args.d_qudit == 2:
            # SWAP(q1,anc) (x) I(q2) in q1 (x) q2 (x) anc basis: |i,j,k> -> |k,j,i>
            rows = [
                k * 4 + j * 2 + i for i in range(2) for j in range(2) for k in range(2)
            ]
            cols = [
                i * 4 + j * 2 + k for i in range(2) for j in range(2) for k in range(2)
            ]
            U_swap_q1_anc = (
                jnp.zeros((8, 8), dtype=jnp.complex64)
                .at[jnp.array(rows), jnp.array(cols)]
                .set(1.0)
            )
            half_fids = jax.vmap(lambda U: gate_fidelity(U, U_swap_q1_anc))(U_halves)
            out["half_fidelities_vs_swap_q1_anc"] = np.array(half_fids)
            best_T = float(T_values[jnp.argmax(half_fids)])
            print(
                f"[analyze] max half-fidelity vs SWAP(q1,anc) (x) I(q2): "
                f"{float(half_fids.max()):.6f}  at T={best_T:.4f}"
            )

        elif args.d_qudit == 4:
            # q1 (x) q2 (x) anc indexing: |i,j,k> -> 8i + 4j + k
            idx_j0 = jnp.array([8 * i + k for i in range(2) for k in range(4)])
            idx_j1 = jnp.array([8 * i + 4 + k for i in range(2) for k in range(4)])

            def block_analysis(U):
                V_j0 = U[jnp.ix_(idx_j0, idx_j0)]
                V_j1 = U[jnp.ix_(idx_j1, idx_j1)]
                return V_j0, jnp.max(jnp.abs(V_j0 - V_j1))

            V_halves, block_diffs = jax.vmap(block_analysis)(U_halves)
            out["V_halves"] = np.array(V_halves)  # (n_T, 8, 8) q1 (x) anc unitary
            out["half_block_diffs"] = np.array(block_diffs)  # (n_T,)
            MT_BOUND = 2 * 3 * float(jnp.pi) / 4
            print(f"[analyze] MT bound 2*3pi/4 = {MT_BOUND:.4f}")
            print(
                f"[analyze] min block diff {float(block_diffs.min()):.6f}  "
                f"at T={float(T_values[jnp.argmin(block_diffs)]):.4f}"
            )

        out_path = os.path.join(args.output_dir, f"d{args.d_qudit}_analysis.npz")
        np.savez(out_path, **out)
        print(f"[analyze] saved -> {out_path}")

    # test mode: verify 2-qubit SWAP speed limit under both constraints
    elif args.mode == "test":
        import numpy as np

        generators, labels = build_pauli_basis()
        U_swap = jnp.zeros((4, 4), dtype=jnp.complex64)
        for i in range(2):
            for j in range(2):
                U_swap = U_swap.at[j * 2 + i, i * 2 + j].set(1.0)

        T_star_op = 3 * jnp.pi / 4  # QSL for SWAP with ||H||_op <= 1 (reference)
        T_values = jnp.linspace(0.3, 4.0, 15)

        print("[test] 2-qubit SWAP speed limit verification")
        print(f"[test] {generators.shape[0]} generators, M={args.M}")
        print(f"[test] restarts={args.n_restarts}, epochs={args.epochs}")

        opt_kwargs = dict(
            n_restarts=args.n_restarts,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )

        # frobenius-norm ||H||_F = 1, simultaneous evolution
        print("\n[test] FROB-NORM (||H||_F=1, simultaneous)")
        print(f"[test] Op-norm reference T* = 3pi/4 = {float(T_star_op):.4f}")
        frob_fids = []
        for T in T_values:
            _, fid, _, _ = optimize_gate(
                generators,
                U_swap,
                args.M,
                float(T),
                normalize_fn=normalize_frob_norm(generators, 1.0),
                evolve_fn=evolve_H,
                **opt_kwargs,
            )
            fid = float(fid)
            frob_fids.append(fid)
            marker = " <- op T*" if abs(float(T) - float(T_star_op)) < 0.15 else ""
            print(f"  T={float(T):.3f}  F={fid:.6f}{marker}")

        # cosine + Trotterized
        print("\n[test] COSINE (Trotterized)")
        trot_fids = []
        for T in T_values:
            _, fid, _, _ = optimize_gate(
                generators,
                U_swap,
                args.M,
                float(T),
                normalize_fn=lambda thetas: jnp.cos(thetas),
                evolve_fn=evolve_H_trotter,
                **opt_kwargs,
            )
            fid = float(fid)
            trot_fids.append(fid)
            print(f"\t\tT={float(T):.3f}\tF={fid:.6f}")

        print("\n[test] COMPARISON")
        print(f"\t\t{'T':>6s}\t{'frob-norm':>10s}\t{'trot+cos':>10s}")
        for i, T in enumerate(T_values):
            print(f"\t\t{float(T):6.3f}\t{frob_fids[i]:10.6f}\t{trot_fids[i]:10.6f}")

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "swap_speed_limit_test.npz")
        np.savez(
            out_path,
            T_values=np.array(T_values),
            frob_norm_fidelities=np.array(frob_fids),
            trotter_cosine_fidelities=np.array(trot_fids),
            T_star_op_norm=float(T_star_op),
        )
        print(f"[test] saved -> {out_path}")
