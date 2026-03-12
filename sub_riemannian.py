#!/usr/bin/env python3
"""Sub-Riemannian analysis of mediator vs direct generator sets.

Computes:
1. First-order effective Hamiltonian (projection onto ancilla |0>)
2. Second-order unconditional cancellation (Tr_anc kills all coupling)
3. Third-order coupling and single-step numerical bounds

Key results:
- First-order projection: zero qubit-qubit coupling for all d.
- Unconditional constraint (U = U_gate ⊗ I_d) kills SECOND-order coupling:
  Tr_anc([G_j, G_k])/d has zero coupling because ⟨m|S_e|m⟩ varies with m
  but the target requires identical coupling for ALL ancilla states.
- Coupling starts at THIRD order in the Magnus expansion (T³ scaling).
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


def partial_trace_anc(operator: jax.Array, d_qudit: int) -> jax.Array:
    """Partial trace over ancilla: Tr_anc(O)/d → 4x4 qubit operator.

    For unconditional gates U = U_gate ⊗ I_d, the I_d component of any
    operator is extracted by Tr_anc(O)/d.
    """
    M = operator.reshape(4, d_qudit, 4, d_qudit)
    return jnp.trace(M, axis1=1, axis2=3) / d_qudit


_PAULI_BASIS = jnp.stack([
    pauli_operators["I"], pauli_operators["x"],
    pauli_operators["y"], pauli_operators["z"],
])  # (4, 2, 2)

_PAULI_LABELS = ["I", "X", "Y", "Z"]

# Precompute 16 Pauli tensor products for decomposition
_PAULI_KRON = jnp.einsum(
    "aij,bkl->abikjl", _PAULI_BASIS, _PAULI_BASIS
).reshape(4, 4, 4, 4)


def pauli_coupling_norm(op_4x4: jax.Array) -> float:
    """Max |c_{ab}| for a!=I, b!=I in the Pauli decomposition of a 4x4 op."""
    coeffs = jnp.einsum("abij,ji->ab", _PAULI_KRON, op_4x4) / 4
    return float(jnp.max(jnp.abs(coeffs[1:, 1:])))


def classify_two_qubit_op(op_4x4: jax.Array):
    """Returns (coupling_norm, single_qubit_norm)."""
    coeffs = jnp.einsum("abij,ji->ab", _PAULI_KRON, op_4x4) / 4
    coupling = float(jnp.max(jnp.abs(coeffs[1:, 1:])))
    single = float(jnp.maximum(
        jnp.max(jnp.abs(coeffs[1:, 0])),
        jnp.max(jnp.abs(coeffs[0, 1:])),
    ))
    return coupling, single


# ── 1. First-order analysis ───────────────────────────────────────────

def first_order_analysis(d_qudit: int):
    """Project all mediator generators onto ancilla |0> and classify."""
    gens, labels = build_qudit_mediated_basis(d_qudit)
    print(f"\n{'=' * 60}")
    print(f"First-order effective Hamiltonian (d={d_qudit})")
    print(f"{'=' * 60}")

    n_zero, n_single, n_coupling = 0, 0, 0
    for k in range(gens.shape[0]):
        G_eff = project_to_qubit_subspace(gens[k], d_qudit)
        norm = float(jnp.max(jnp.abs(G_eff)))
        if norm < 1e-8:
            n_zero += 1
            status = "ZERO"
        else:
            c, s = classify_two_qubit_op(G_eff)
            if c > 1e-6:
                n_coupling += 1
                status = f"COUPLING ({c:.4f})"
            else:
                n_single += 1
                status = f"SINGLE-QUBIT ({s:.4f})"
        print(f"  {labels[k]:22s} → {status}")

    print(f"\n  Summary: {n_zero} zero, {n_single} single-qubit, {n_coupling} coupling")
    return n_coupling


# ── 2. Second-order unconditional analysis ────────────────────────────

def second_order_unconditional(d_qudit: int):
    """Show that Tr_anc([G_j, G_k])/d has zero qubit-qubit coupling.

    Physical reason: [G_j, G_k] projected onto ancilla state |m> gives
    coupling proportional to ⟨m|S_e|m⟩ (diagonal element of a spin op).
    Since ⟨m|S_e|m⟩ varies with m but the unconditional constraint
    requires identical coupling for ALL m, the only consistent solution
    is zero coupling. Mathematically: Tr(S_e) = 0 for e ∈ {x,y,z}.
    """
    gens, labels = build_qudit_mediated_basis(d_qudit)
    n = gens.shape[0]

    print(f"\n{'=' * 60}")
    print(f"Second-order unconditional analysis (d={d_qudit})")
    print(f"{'=' * 60}")

    products = jnp.einsum("jab,kbc->jkac", gens, gens)
    comms = -1j * (products - products.transpose(1, 0, 2, 3))

    # Projection onto |0> only (conditional)
    comms_r = comms.reshape(n, n, 4, d_qudit, 4, d_qudit)
    comms_proj0 = comms_r[:, :, :, 0, :, 0]

    max_proj0 = 0.0
    for j in range(n):
        for k in range(j + 1, n):
            c = pauli_coupling_norm(comms_proj0[j, k])
            max_proj0 = max(max_proj0, c)

    # Unconditional: Tr_anc/d
    comms_traced = jnp.trace(comms_r, axis1=3, axis2=5) / d_qudit

    max_uncond = 0.0
    for j in range(n):
        for k in range(j + 1, n):
            c = pauli_coupling_norm(comms_traced[j, k])
            max_uncond = max(max_uncond, c)

    print(f"  Conditional (⟨0| only):   max coupling = {max_proj0:.4f}")
    print(f"  Unconditional (Tr_anc/d): max coupling = {max_uncond:.8f}")
    print(f"  → Second-order coupling {'VANISHES' if max_uncond < 1e-6 else 'NONZERO'} "
          f"under unconditional constraint")

    return max_uncond


# ── 3. Third-order analysis ───────────────────────────────────────────

def third_order_unconditional(d_qudit: int):
    """Third-order Magnus: Tr_anc([[G_p,G_q],G_r])/d.

    The triple commutator picks up the anticommutator {S_e, S_g} which
    has a nonzero trace (∝ δ_{eg}). This is the LEADING ORDER coupling
    for unconditional mediator gates.

    Returns G3: maximum per-triple coupling coefficient.
    """
    gens, labels = build_qudit_mediated_basis(d_qudit)
    n = gens.shape[0]

    print(f"\n{'=' * 60}")
    print(f"Third-order unconditional coupling (d={d_qudit})")
    print(f"{'=' * 60}")

    products = jnp.einsum("jab,kbc->jkac", gens, gens)
    comms = -1j * (products - products.transpose(1, 0, 2, 3))

    # [[G_p,G_q], G_r] via [comms[p,q], G_r]
    # comms[p,q] is Hermitian, so [comms,G] is anti-Hermitian
    triple = (jnp.einsum("pqab,rbc->pqrac", comms, gens)
              - jnp.einsum("rab,pqbc->pqrac", gens, comms))

    # Unconditional projection
    triple_r = triple.reshape(n, n, n, 4, d_qudit, 4, d_qudit)
    Q = jnp.trace(triple_r, axis1=4, axis2=6) / d_qudit  # (n,n,n,4,4)

    # Pauli coupling coefficients (purely imaginary for anti-Hermitian Q)
    T_coeffs = jnp.einsum("abil,pqrli->pqrab", _PAULI_KRON, Q) / 4
    coupling_abs = jnp.abs(T_coeffs[:, :, :, 1:, 1:])

    max_per_triple = float(jnp.max(coupling_abs))

    # Find the top contributing triples
    top_triples = []
    for p in range(n):
        for q in range(n):
            for r in range(n):
                c = float(jnp.max(coupling_abs[p, q, r]))
                if c > max_per_triple * 0.9:
                    top_triples.append((p, q, r, c))
    top_triples.sort(key=lambda x: -x[3])

    print(f"  Max per-triple coupling: {max_per_triple:.4f}")
    print(f"  {len(top_triples)} triples at >90% of max")
    if top_triples:
        p, q, r, c = top_triples[0]
        print(f"  Example: [[{labels[p]}, {labels[q]}], {labels[r]}] → {c:.4f}")

    # Count nonzero triples
    n_nonzero = int(jnp.sum(jnp.max(coupling_abs, axis=(-2, -1)) > 1e-6))
    print(f"  {n_nonzero} nonzero triples out of {n**3}")

    return max_per_triple


# ── 4. Single-step numerical bound ───────────────────────────────────

def single_step_coupling_bound(d_qudit: int, T_values=None):
    """Numerically compute max unconditional coupling from exp(-iHT).

    For each T, optimize coupling over ||ω||_∞ ≤ 1:
        g(T) = max_ω coupling(Tr_anc(exp(-i Σ ω_j G_j T)) / d)

    If g(T)/T < 1 for all T, the mediator is strictly slower per step.
    """
    gens, _ = build_qudit_mediated_basis(d_qudit)
    n_gen = gens.shape[0]

    if T_values is None:
        T_values = np.concatenate([
            np.linspace(0.05, 0.5, 10),
            np.linspace(0.5, 2.0, 16),
            np.linspace(2.0, 5.0, 7),
        ])

    print(f"\n{'=' * 60}")
    print(f"Single-step numerical coupling bound (d={d_qudit})")
    print(f"{'=' * 60}")

    rng = np.random.default_rng(42)
    n_restarts = 200
    results = []

    for T in T_values:
        best_coupling = 0.0
        for _ in range(n_restarts):
            omega = rng.uniform(-1, 1, size=n_gen)
            H = jnp.einsum("j,jab->ab", omega, gens)
            U = jax.scipy.linalg.expm(-1j * H * T)
            Q = partial_trace_anc(U, d_qudit)
            c = pauli_coupling_norm(Q)
            best_coupling = max(best_coupling, c)
        rate = best_coupling / T if T > 0 else 0.0
        results.append((T, best_coupling, rate))

    print(f"  {'T':>6s}  {'coupling':>10s}  {'rate=c/T':>10s}  {'vs direct':>10s}")
    max_rate = 0.0
    for T, c, r in results:
        max_rate = max(max_rate, r)
        marker = " *" if r > 1.0 else ""
        print(f"  {T:6.3f}  {c:10.6f}  {r:10.6f}  {'FASTER' if r > 1 else 'slower':>10s}{marker}")

    print(f"\n  Max coupling rate (g/T): {max_rate:.6f}")
    if max_rate < 1.0:
        print(f"  → Single-step mediator coupling ALWAYS slower than direct")
    else:
        print(f"  → Single-step bound inconclusive (rate > 1 at some T)")

    return results


# ── 5. Time lower bounds ──────────────────────────────────────────────

def mediator_time_bound(d_qudit: int, c1: float, c2: float, c3: float,
                        g3_per_triple: float):
    """Lower bound from third-order Magnus (unconditional).

    Since both first and second order coupling vanish, coupling ∝ T³.
    Bound: T_med ≥ (6·d_W / G₃)^{1/3} where G₃ = trilinear max.
    Note: this bound is generally inconclusive because G₃ grows as n³.
    """
    d_W = abs(c1) + abs(c2) + abs(c3)
    T_direct = d_W

    print(f"\n  Weyl ({c1:.4f}, {c2:.4f}, {c3:.4f}), d_W = {d_W:.4f}:")
    print(f"    Direct:   T ≥ {T_direct:.4f}")
    print(f"    Mediator: coupling starts at O(T³), per-triple coeff = {g3_per_triple:.4f}")

    return T_direct


if __name__ == "__main__":
    for d in [2, 4]:
        # 1. First order: zero coupling
        n_coupling = first_order_analysis(d)
        assert n_coupling == 0, f"d={d}: found {n_coupling} coupling terms (expect 0)"

        # 2. Second order: unconditional cancellation
        max_2nd = second_order_unconditional(d)
        assert max_2nd < 1e-6, f"d={d}: 2nd-order unconditional coupling = {max_2nd}"

        # 3. Third order: first nonzero coupling
        g3 = third_order_unconditional(d)

    # 4. Single-step numerical bound
    print(f"\n{'=' * 60}")
    print("Single-step numerical bounds")
    print(f"{'=' * 60}")
    for d in [2, 4]:
        single_step_coupling_bound(d)
