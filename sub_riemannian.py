#!/usr/bin/env python3
"""Sub-Riemannian analysis of mediator vs direct generator sets.

Computes:
1. First-order effective Hamiltonian (projection onto ancilla |0>)
2. Second-order effective coupling via BCH commutators
3. Lower bound comparison: effective coupling strength vs direct coupling
"""
import jax
import jax.numpy as jnp
import numpy as np

from grape_lib import build_qudit_mediated_basis, pauli_operators


def project_to_qubit_subspace(generator: jax.Array, d_qudit: int) -> jax.Array:
    """Project a q1⊗q2⊗anc generator onto the ancilla ground state |0><0|.

    Returns the effective 4x4 two-qubit operator:
        G_eff[i,j] = G[(i, anc=0), (j, anc=0)]
    """
    G = generator.reshape(4, d_qudit, 4, d_qudit)
    return G[:, 0, :, 0]


_PAULI_BASIS = jnp.stack([
    pauli_operators["I"], pauli_operators["x"],
    pauli_operators["y"], pauli_operators["z"],
])  # (4, 2, 2)


def classify_two_qubit_op(op_4x4: jax.Array, tol: float = 1e-6):
    """Classify a 4x4 two-qubit operator via Pauli decomposition.

    Returns (coupling_norm, max_single_norm):
        coupling_norm: max |c_{ab}| for a!=I, b!=I (qubit-qubit coupling strength)
        max_single_norm: max |c_{ab}| for a=I xor b=I (single-qubit strength)
    """
    # Decompose: c_{ab} = Tr(σ_a⊗σ_b · O) / 4
    pauli_kron = jnp.einsum("aij,bkl->abikjl", _PAULI_BASIS, _PAULI_BASIS)
    pauli_kron = pauli_kron.reshape(4, 4, 4, 4)  # (a, b, row, col)
    coeffs = jnp.einsum("abij,ji->ab", pauli_kron, op_4x4) / 4  # (4, 4)

    # Coupling: both indices non-identity (a>0 and b>0)
    coupling_coeffs = jnp.abs(coeffs[1:, 1:])
    coupling_norm = float(jnp.max(coupling_coeffs))

    # Single-qubit: exactly one index is identity
    single_q1 = jnp.abs(coeffs[1:, 0])  # σ_a ⊗ I
    single_q2 = jnp.abs(coeffs[0, 1:])  # I ⊗ σ_b
    max_single = float(jnp.maximum(jnp.max(single_q1), jnp.max(single_q2)))

    return coupling_norm, max_single


def first_order_analysis(d_qudit: int):
    """Project all mediator generators onto ancilla |0> and classify."""
    gens, labels = build_qudit_mediated_basis(d_qudit)
    print(f"\n{'=' * 60}")
    print(f"First-order effective Hamiltonian (d={d_qudit})")
    print(f"{'=' * 60}")

    n_zero = 0
    n_single = 0
    n_coupling = 0

    for k in range(gens.shape[0]):
        G_eff = project_to_qubit_subspace(gens[k], d_qudit)
        norm = float(jnp.max(jnp.abs(G_eff)))
        if norm < 1e-8:
            status = "ZERO"
            n_zero += 1
        else:
            coupling, single = classify_two_qubit_op(G_eff)
            if coupling > 1e-6:
                status = f"COUPLING (strength {coupling:.4f})"
                n_coupling += 1
            else:
                status = f"SINGLE-QUBIT (strength {single:.4f})"
                n_single += 1
        print(f"  {labels[k]:22s} → {status}")

    print(f"\n  Summary: {n_zero} zero, {n_single} single-qubit, {n_coupling} coupling")
    return n_coupling


def second_order_coupling(d_qudit: int):
    """Compute effective qubit-qubit coupling from commutators [G_j, G_k].

    At second order in the Magnus expansion, the effective Hamiltonian
    picks up terms ~ [H_1, H_2] dt^2. We project [G_j, G_k] onto the
    ancilla ground state and check for qubit-qubit coupling.
    """
    gens, labels = build_qudit_mediated_basis(d_qudit)
    print(f"\n{'=' * 60}")
    print(f"Second-order effective coupling (d={d_qudit})")
    print(f"{'=' * 60}")

    n = gens.shape[0]

    # Vectorized: compute all commutators at once
    # comm[j,k] = -i (G_j G_k - G_k G_j)
    products_jk = jnp.einsum("jab,kbc->jkac", gens, gens)  # (n, n, dim, dim)
    comms = -1j * (products_jk - products_jk.transpose(1, 0, 2, 3))  # (n, n, dim, dim)

    # Project each commutator onto ancilla ground state
    comms_4d = comms.reshape(n, n, 4, d_qudit, 4, d_qudit)
    comms_eff = comms_4d[:, :, :, 0, :, 0]  # (n, n, 4, 4)

    # Compute norms of effective commutators (upper triangle only)
    norms = jnp.max(jnp.abs(comms_eff), axis=(-2, -1))  # (n, n)

    # Find coupling commutators (act nontrivially on both qubits)
    coupling_comms = []
    for j in range(n):
        for k in range(j + 1, n):
            norm_val = float(norms[j, k])
            if norm_val > 1e-8:
                coupling, _ = classify_two_qubit_op(comms_eff[j, k])
                if coupling > 1e-6:
                    coupling_comms.append((j, k, coupling))

    print(f"  Found {len(coupling_comms)} coupling commutator pairs")

    if coupling_comms:
        coupling_comms.sort(key=lambda x: -x[2])
        print(f"  Top coupling commutators:")
        for j, k, c_norm in coupling_comms[:10]:
            print(
                f"    [{labels[j]}, {labels[k]}]: "
                f"coupling strength {c_norm:.4f}"
            )

    max_eff = max((x[2] for x in coupling_comms), default=0.0)
    print(f"\n  Max effective 2nd-order coupling strength: {max_eff:.4f}")
    print(f"  Direct coupling norm (each σ_a⊗σ_b):  1.0000")
    print(f"  Ratio (effective/direct):              {max_eff:.4f}")

    return max_eff


def sub_riemannian_lower_bound(d_qudit: int, c1: float, c2: float, c3: float):
    """Lower bound on time to reach U_can(c1,c2,c3) via mediator generators.

    The effective coupling rate from second-order processes bounds the
    minimum gate time: T_mediator >= weyl_distance / g_eff.
    """
    max_eff = second_order_coupling.__wrapped_max_eff.get(d_qudit)
    if max_eff is None:
        max_eff = second_order_coupling(d_qudit)
        second_order_coupling.__wrapped_max_eff[d_qudit] = max_eff

    weyl_sum = abs(c1) + abs(c2) + abs(c3)
    T_direct = weyl_sum
    T_mediator_lb = weyl_sum / max_eff if max_eff > 0 else float("inf")

    print(f"\n  Weyl point ({c1:.4f}, {c2:.4f}, {c3:.4f}):")
    print(f"    Direct T* ≈ {T_direct:.4f}")
    print(f"    Mediator lower bound T* ≥ {T_mediator_lb:.4f}")
    if T_direct > 0:
        print(f"    Slowdown factor ≥ {T_mediator_lb / T_direct:.2f}x")

    return T_direct, T_mediator_lb


# Cache for second_order results
second_order_coupling.__wrapped_max_eff = {}


if __name__ == "__main__":
    for d in [2, 4]:
        n_ent = first_order_analysis(d)
        assert n_ent == 0, f"d={d}: found {n_ent} entangling first-order terms (expect 0)"
        max_eff = second_order_coupling(d)
        second_order_coupling.__wrapped_max_eff[d] = max_eff

    print(f"\n{'=' * 60}")
    print("Sub-Riemannian lower bounds for key gates")
    print(f"{'=' * 60}")

    gates = {
        "CNOT": (np.pi / 4, 0.0, 0.0),
        "iSWAP": (np.pi / 4, np.pi / 4, 0.0),
        "SWAP": (np.pi / 4, np.pi / 4, np.pi / 4),
        "√SWAP": (np.pi / 8, np.pi / 8, np.pi / 8),
    }
    for d in [2, 4]:
        print(f"\nd={d}:")
        for name, (c1, c2, c3) in gates.items():
            print(f"\n  --- {name} ---")
            sub_riemannian_lower_bound(d, c1, c2, c3)
