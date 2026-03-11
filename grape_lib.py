"""Shared GRAPE library for quantum gate synthesis experiments.

Extracted from grape-curc-sim.py. All generator-building functions run at
initialization time (outside JIT). The optimization hot path (evolve_H,
gate_fidelity, optimize_gate) is fully JIT-compatible with no Python
control flow or mutability.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax


# ── Pauli operators (qubit) ─────────────────────────────────────────────

pauli_operators = {
    "I": jnp.eye(2),
    "x": jnp.array([[0, 1], [1, 0]]),
    "y": jnp.array([[0, -1j], [1j, 0]]),
    "z": jnp.array([[1, 0], [0, -1]]),
}

_PAULI_LABELS = ("I", "x", "y", "z")
_PAULI_STACK = jnp.stack([pauli_operators[k] for k in _PAULI_LABELS])  # (4, 2, 2)


# ── Spin operators (qudit) ──────────────────────────────────────────────

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


# ── Generator basis builders (run once at init, not JIT-traced) ─────────

def build_pauli_basis():
    """15 two-qubit Pauli tensor products {σ_a ⊗ σ_b}, excluding I⊗I."""
    generators, labels = [], []
    for a, sa in pauli_operators.items():
        for b, sb in pauli_operators.items():
            if a == "I" and b == "I":
                continue
            generators.append(jnp.kron(sa, sb))
            labels.append(f"σ{a}⊗σ{b}")
    return jnp.stack(generators), labels


def build_qudit_mediated_basis(d_qudit: int):
    """Hermitian generators for two qubits coupled to a d_qudit-level qudit.

    Hilbert space: qubit1 ⊗ qubit2 ⊗ qudit.
    27 generators for all supported d values.
    """
    I2 = jnp.eye(2, dtype=jnp.complex64)
    spin_ops = pauli_operators if d_qudit == 2 else generate_spin_operators(d_qudit)

    generators, labels = [], []
    seen: set[str] = set()

    for qubit_idx in [1, 2]:
        for a, sigma_a in pauli_operators.items():
            for b, S_b in spin_ops.items():
                if a == "I" and b == "I":
                    continue
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


def build_direct_basis_lifted(d_qudit: int):
    """15 two-qubit Pauli generators lifted to q1⊗q2⊗qudit space.

    Each generator is σ_a ⊗ σ_b ⊗ I_d (qubit-qubit coupling, identity on ancilla).
    """
    gens_4, labels = build_pauli_basis()
    I_d = jnp.eye(d_qudit, dtype=jnp.complex64)
    gens_lifted = jnp.stack([jnp.kron(gens_4[k], I_d) for k in range(gens_4.shape[0])])
    return gens_lifted, labels


def build_combined_basis(d_qudit: int):
    """Union of mediator and direct generators, deduplicated.

    Returns 36 generators: 27 mediator + 9 new qubit-qubit coupling terms.
    Deduplication via vectorized Frobenius inner products.
    """
    med_gens, med_labels = build_qudit_mediated_basis(d_qudit)
    dir_gens, dir_labels = build_direct_basis_lifted(d_qudit)

    # Vectorized overlap: (n_dir, n_med) matrix of |Tr(D† M)| / (||D|| ||M||)
    # Flatten to (n, dim^2) for dot products
    n_med, n_dir = med_gens.shape[0], dir_gens.shape[0]
    med_flat = med_gens.reshape(n_med, -1)
    dir_flat = dir_gens.reshape(n_dir, -1)

    overlaps = jnp.abs(dir_flat.conj() @ med_flat.T)  # (n_dir, n_med)
    dir_norms = jnp.sqrt(jnp.sum(jnp.abs(dir_flat) ** 2, axis=1, keepdims=True))
    med_norms = jnp.sqrt(jnp.sum(jnp.abs(med_flat) ** 2, axis=1, keepdims=True))
    normalized = overlaps / (dir_norms @ med_norms.T + 1e-12)

    # A direct generator is a duplicate if its max overlap with any mediator gen > 0.99
    is_dup = jnp.max(normalized, axis=1) > 0.99  # (n_dir,)

    new_gens = [dir_gens[k] for k in range(n_dir) if not bool(is_dup[k])]
    new_labels = [f"direct:{dir_labels[k]}" for k in range(n_dir) if not bool(is_dup[k])]

    combined_gens = jnp.concatenate([med_gens, jnp.stack(new_gens)], axis=0)
    return combined_gens, med_labels + new_labels


def build_multiqubit_mediator_basis(n_anc: int):
    """Generators for two qubits coupled to n ancilla qubits.

    Hilbert space: q1 ⊗ q2 ⊗ anc1 ⊗ ... ⊗ anc_n,  dim = 4 * 2^n.
    Includes: 1-body terms on each subsystem, 2-body qubit-ancilla couplings.
    No ancilla-ancilla couplings (only 1 and 2-body interactions).
    """
    n_total = 2 + n_anc
    d_total = 2 ** n_total

    # Precompute identity chain for each subsystem count
    # Build operator via: I ⊗ ... ⊗ Op_site ⊗ ... ⊗ I
    def _single_site_op(site: int, op):
        """Embed a 2x2 operator at `site` into the full d_total space."""
        result = jnp.eye(1, dtype=jnp.complex64)
        for i in range(n_total):
            result = jnp.kron(result, op if i == site else jnp.eye(2, dtype=jnp.complex64))
        return result

    def _two_site_op(site_a: int, op_a, site_b: int, op_b):
        """Embed two 2x2 operators at sites a and b into the full space."""
        result = jnp.eye(1, dtype=jnp.complex64)
        for i in range(n_total):
            if i == site_a:
                result = jnp.kron(result, op_a)
            elif i == site_b:
                result = jnp.kron(result, op_b)
            else:
                result = jnp.kron(result, jnp.eye(2, dtype=jnp.complex64))
        return result

    generators, labels = [], []

    # 1-body terms on all subsystems
    for site in range(n_total):
        site_label = f"q{site + 1}" if site < 2 else f"anc{site - 1}"
        for a in ("x", "y", "z"):
            generators.append(_single_site_op(site, pauli_operators[a]))
            labels.append(f"{site_label}:σ{a}")

    # 2-body qubit-ancilla coupling terms
    for q_idx in range(2):
        for a_idx in range(n_anc):
            anc_site = 2 + a_idx
            for pa in ("x", "y", "z"):
                for pb in ("x", "y", "z"):
                    generators.append(_two_site_op(
                        q_idx, pauli_operators[pa],
                        anc_site, pauli_operators[pb],
                    ))
                    labels.append(f"q{q_idx + 1}:σ{pa}⊗anc{a_idx + 1}:σ{pb}")

    return jnp.stack(generators), labels


def build_swap_qudit_target(d_qudit: int) -> jax.Array:
    """SWAP ⊗ I_d: SWAP on the two qubits, identity on the qudit."""
    U_swap = jnp.zeros((4, 4), dtype=jnp.complex64)
    for i in range(2):
        for j in range(2):
            U_swap = U_swap.at[j * 2 + i, i * 2 + j].set(1.0)
    return jnp.kron(U_swap, jnp.eye(d_qudit, dtype=jnp.complex64))


def build_weyl_target(c1: float, c2: float, c3: float, d_qudit: int = 1) -> jax.Array:
    """Canonical two-qubit gate exp(-i(c1 XX + c2 YY + c3 ZZ)) ⊗ I_d.

    Weyl chamber: c1 >= c2 >= c3 >= 0, c1 <= pi/4.
    d_qudit=1 gives the pure 4x4 two-qubit gate.
    """
    XX = jnp.kron(pauli_operators["x"], pauli_operators["x"])
    YY = jnp.kron(pauli_operators["y"], pauli_operators["y"])
    ZZ = jnp.kron(pauli_operators["z"], pauli_operators["z"])
    H_weyl = c1 * XX + c2 * YY + c3 * ZZ
    U_can = jax.scipy.linalg.expm(-1j * H_weyl.astype(jnp.complex64))
    if d_qudit == 1:
        return U_can
    return jnp.kron(U_can, jnp.eye(d_qudit, dtype=jnp.complex64))


def build_weyl_grid(step_denom: int = 8) -> list[tuple[float, float, float]]:
    """Weyl chamber tetrahedron grid with step = pi/step_denom.

    Returns list of (c1, c2, c3) with c1 >= c2 >= c3 >= 0, c1 <= pi/4.
    step_denom=8 gives 10 points; step_denom=16 gives 35 points.
    """
    step = np.pi / step_denom
    points = []
    for c1 in np.arange(0, np.pi / 4 + 1e-10, step):
        for c2 in np.arange(0, c1 + 1e-10, step):
            for c3 in np.arange(0, c2 + 1e-10, step):
                points.append((float(c1), float(c2), float(c3)))
    return points


# ── Core physics (all JIT-compatible, no Python control flow) ───────────

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
    """Unitary from piecewise-constant Hamiltonian evolution.

    Args:
        coupling_strengths: (M, n_gen) coupling strength per timestep per generator
        generators: (n_gen, N, N) Hermitian basis operators
        time: total evolution time

    Returns:
        (N, N) unitary
    """
    dt = time / coupling_strengths.shape[0]
    H_all = jnp.einsum("ij,jkl->ikl", coupling_strengths, generators)
    U_steps = jax.vmap(jax.scipy.linalg.expm)(-1j * H_all * dt)
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
    """Multi-restart GRAPE optimization with Adam and cosine reparametrization.

    Note: default lr=0.01 here; CLI scripts typically override to 0.005.
    """
    optimizer = optax.adam(lr)

    def loss_fun(thetas: jax.Array):
        U = evolve_H(jnp.cos(thetas), generators, time)
        return 1.0 - gate_fidelity(U, U_target)

    def single_restart(key):
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
        return jnp.cos(final_thetas), losses

    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    all_params, all_losses = jax.vmap(single_restart)(keys)
    best_id = jnp.argmin(all_losses[:, -1])
    return (
        all_params[best_id],
        1.0 - all_losses[best_id, -1],
        all_losses[best_id],
        all_losses,
    )
