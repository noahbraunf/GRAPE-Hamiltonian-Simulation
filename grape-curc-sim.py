from enum import Enum

import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import lax

from lbfgs_b import lbfgsb


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

    Hilbert space: qubit1 ⊗ qubit2 ⊗ qudit.
    Includes all sigma_a ⊗ I ⊗ S_b and I ⊗ sigma_a ⊗ S_b terms,
    excluding the global identity (a="I" AND b="I").
    Deduplicates I ⊗ I ⊗ S_b terms that appear for both qubit slots.
    Yields 27 generators for all supported d values.
    """
    I2 = jnp.eye(2, dtype=jnp.complex64)
    if d_qudit == 2:
        spin_ops = pauli_operators
    else:
        spin_ops = generate_spin_operators(d_qudit)

    generators, labels = [], []
    seen: set[str] = set()

    for qubit_idx in [1, 2]:
        for a, sigma_a in pauli_operators.items():
            for b, S_b in spin_ops.items():
                if a == "I" and b == "I":
                    continue  # global identity — trivial
                # I⊗I⊗S_b is qudit-only; same regardless of qubit_idx
                label = f"I⊗I⊗S{b}" if a == "I" else f"q{qubit_idx}:σ{a}⊗S{b}"
                if label in seen:
                    continue
                seen.add(label)
                if qubit_idx == 1:
                    op = jnp.kron(jnp.kron(sigma_a, I2), S_b)
                else:
                    op = jnp.kron(jnp.kron(I2, sigma_a), S_b)
                generators.append(op)
                labels.append(label)

    return jnp.stack(generators), labels


def build_swap_qudit_target(d_qudit: int) -> jax.Array:
    """SWAP ⊗ I_d: SWAP on the two qubits, identity on the qudit."""
    U_swap = jnp.zeros((4, 4), dtype=jnp.complex64)
    for i in range(2):
        for j in range(2):
            U_swap = U_swap.at[j * 2 + i, i * 2 + j].set(1.0)
    return jnp.kron(U_swap, jnp.eye(d_qudit, dtype=jnp.complex64))


def hamiltonian_generators_spin_mediated(n_qubits=2, mediator_dimension: int = 2):
    generators = []
    labels = []
    spin_operators = generate_spin_operators(mediator_dimension)

    def tensor_product_all(ops):
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

        dimension = prod(op.shape[0] for op in ops)

        return jnp.einsum(*inputs, outputs).reshape(dimension, dimension)

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

                    generators.append(tensor_product_all(operators))
                    labels.append("$" + " \\otimes ".join(cur_labels) + "$")
    return jnp.stack(generators), labels


@jax.jit
def gate_fidelity(U, V):
    d = U.shape[-1]
    overlap = jnp.einsum("...ij,...ij->", jnp.conj(U), V)
    return (jnp.abs(overlap) ** 2 + d) / (d * (d + 1))


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
):
    n_generators = generators.shape[0]
    optimizer = lbfgsb(
        lower=-jnp.ones((n_generators,)),
        upper=jnp.ones((n_generators,)),
        learning_rate=lr,
    )

    def loss_fun(omegas: jax.Array):
        U = evolve_H(
            omegas,
            generators,
            time,
        )
        return 1.0 - gate_fidelity(U, U_target)

    def single_restart(key):
        params = initial_scale * jax.random.normal(key, (n_steps, n_generators))
        opt_state = optimizer.init(params)

        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fun)(params)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, params, value=loss, value_fn=loss_fun, grad=grads
            )
            return (optax.apply_updates(params, updates), new_opt_state), loss

        (final_params, _), losses = lax.scan(
            step, (params, opt_state), None, length=epochs
        )
        return final_params, losses

    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    all_params, all_losses = jax.vmap(single_restart)(
        keys
    )  # all_losses: (n_restarts, epochs)
    best_id = jnp.argmin(all_losses[:, -1])
    return all_params[best_id], 1.0 - all_losses[best_id, -1], all_losses[best_id], all_losses


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
    U_swap = jnp.zeros((4, 4), dtype=jnp.complex64)
    for i in range(2):
        for j in range(2):
            U_swap = U_swap.at[j * 2 + i, i * 2 + j].set(1.0)

    generators_t0, labels_t0 = build_pauli_basis()
    M_t0 = 20  # timesteps
    T_t0 = 1.0  # total evolution time (arbitrary when Omega is unbounded)

    print(f"Tier 0: {len(labels_t0)} generators, M = {M_t0}, T = {T_t0}")
    print(f"Total parameters: {len(labels_t0)} x {M_t0} = {M_t0 * len(labels_t0)}")

    omegas_t0, fid_t0, hist_t0, _ = optimize_gate(
        generators_t0,
        U_swap,
        n_steps=M_t0,
        time=T_t0,
        n_restarts=50,
        epochs=3000,
        lr=0.01,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(hist_t0)
    ax.set(xlabel="Epoch", ylabel="Infidelity (1 - F)")
    ax.set_title(f"Tier 0: Two-Qubit SWAP (unbounded) — F = {fid_t0:.10f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
