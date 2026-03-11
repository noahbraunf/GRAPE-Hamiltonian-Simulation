from typing import Any, Callable, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

FlatValueGrad: TypeAlias = Callable[[jax.Array], tuple[jax.Array, jax.Array]]

# A JAX scalar array holding a float (shape ``()``, e.g. ``jnp.float64``).
FloatScalar: TypeAlias = jax.Array
# A JAX scalar array holding an int (shape ``()``, dtype ``int32``).
IntScalar: TypeAlias = jax.Array
# A JAX scalar array holding a bool (shape ``()``, dtype ``bool``).
BoolScalar: TypeAlias = jax.Array


class LBFGSBState(NamedTuple):
    """Full solver state for L-BFGS-B.

    All fields are JAX arrays so the entire state is a valid pytree and
    can be passed through ``lax.scan`` / ``lax.while_loop``.

    Attributes
    ---
    x : Array, shape ``(n,)``
        Current parameter vector.
    f : FloatScalar
        Objective value at ``x``.
    g : Array, shape ``(n,)``
        Gradient at ``x``.
    s_hist : Array, shape ``(m, n)``
        Ring buffer of displacement vectors ``s_k = x_{k+1} - x_k``.
    y_hist : Array, shape ``(m, n)``
        Ring buffer of gradient difference vectors ``y_k = g_{k+1} - g_k``.
    rho_hist : Array, shape ``(m,)``
        Ring buffer of curvature reciprocals ``1 / (s_k^T y_k)``.
    head : IntScalar
        Index of the *next* insertion slot in the ring buffers.
    size : IntScalar
        Number of valid correction pairs currently stored (≤ m).
    k : IntScalar
        Iteration counter.
    converged : BoolScalar
        ``True`` once the projected-gradient norm drops below ``gtol``.
    """

    x: jax.Array
    f: FloatScalar
    g: jax.Array
    s_hist: jax.Array
    y_hist: jax.Array
    rho_hist: jax.Array
    sy_gram: jax.Array  # (m, m) cached S^T Y
    ss_gram: jax.Array  # (m, m) cached S^T S
    head: IntScalar
    size: IntScalar
    k: IntScalar
    converged: BoolScalar


class LBFGSBTrace(NamedTuple):
    """Per-iteration diagnostic record (one per ``lax.scan`` step).

    Attributes
    ---
    f : FloatScalar
        Objective value after the step.
    proj_grad_inf : FloatScalar
        Infinity norm of the projected gradient *before* the step.
    step_norm : FloatScalar
        ``||x_new − x_old||_2``.  Zero when the step is skipped.
    accepted : BoolScalar
        Whether the line search accepted a step.
    """

    f: FloatScalar
    proj_grad_inf: FloatScalar
    step_norm: FloatScalar
    accepted: BoolScalar


class ScaleLBFGSBState(NamedTuple):
    """Optax-facing state for :func:`scale_by_lbfgsb`.

    Attributes
    ----------
    s_hist : Array, shape ``(m, n)``
        Ring buffer of displacement vectors.
    y_hist : Array, shape ``(m, n)``
        Ring buffer of gradient differences.
    rho_hist : Array, shape ``(m,)``
        Ring buffer of curvature reciprocals.
    head : IntScalar
        Next insertion slot.
    size : IntScalar
        Number of valid pairs stored.
    num_updates : IntScalar
        Total number of ``update`` calls so far.
    prev_params : PyTree
        Parameters from the previous ``update`` call (arbitrary pytree,
        same structure as the model parameters).
    prev_grad : PyTree
        Gradient from the previous ``update`` call (same structure as
        ``prev_params``).
    """

    s_hist: jax.Array
    y_hist: jax.Array
    rho_hist: jax.Array
    head: IntScalar
    size: IntScalar
    num_updates: IntScalar
    prev_params: PyTree
    prev_grad: PyTree


def make_flat_value_and_grad(
    loss_fn: Callable[[PyTree], jax.Array], params0: PyTree
) -> tuple[jax.Array, Callable[[jax.Array], PyTree], FlatValueGrad]:
    """Flatten pytree loss into 1D value and grad function"""
    x0, unravel = ravel_pytree(params0)

    def flat_loss(x: jax.Array) -> jax.Array:
        return jnp.asarray(loss_fn(unravel(x)), dtype=x.dtype)

    return x0, unravel, jax.value_and_grad(flat_loss)


def convergence_measure(
    x: jax.Array, g: jax.Array, lower: jax.Array, upper: jax.Array
) -> jax.Array:
    """Eq. (6.1): ``||P(x-g, l, u) - x||_inf``"""
    return jnp.max(jnp.abs(jnp.clip(x - g, lower, upper) - x))


def _compute_theta(
    s_hist: jax.Array, y_hist: jax.Array, head: IntScalar, size: IntScalar
) -> FloatScalar:
    """theta = y_k^T y_k / (s_k^T y_k) for most recent pair (from algorithm step 7)"""
    m = s_hist.shape[0]
    newest = (head - jnp.int32(1)) % m
    s_last, y_last = s_hist[newest], y_hist[newest]
    sty = jnp.dot(s_last, y_last)
    yty = jnp.dot(y_last, y_last)
    return jnp.where(size > 0, yty / (sty + 1e-30), 1.0)


def _chronological_order(
    s_hist: jax.Array,
    y_hist: jax.Array,
    rho_hist: jax.Array,
    head: IntScalar,
    size: IntScalar,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return history from oldest to newest plus float validity mask"""
    m = s_hist.shape[0]
    id = (head - size + jnp.arange(m, dtype=head.dtype)) % m
    valid = (jnp.arange(m, dtype=size.dtype) < size).astype(s_hist.dtype)
    return s_hist[id], y_hist[id], rho_hist[id], valid


def _wt_times_v(
    s_ord: jax.Array,
    y_ord: jax.Array,
    valid: jax.Array,
    theta: FloatScalar,
    v: jax.Array,
) -> jax.Array:
    """W^T v = [Y^T v; theta S^T v] from precomputed chronological arrays."""
    yt_v = (y_ord @ v) * valid
    st_v = (s_ord @ v) * valid
    return jnp.concatenate([yt_v, theta * st_v])


def _w_times_u(
    s_ord: jax.Array,
    y_ord: jax.Array,
    valid: jax.Array,
    theta: FloatScalar,
    u: jax.Array,
) -> jax.Array:
    """W u = Y u_1 + theta S u_2 from precomputed chronological arrays."""
    m = s_ord.shape[0]
    u1 = u[:m] * valid
    u2 = u[m:] * valid
    return y_ord.T @ u1 + theta * (s_ord.T @ u2)


def _nmat_solve(nmat_lu: tuple[jax.Array, jax.Array], v: jax.Array) -> jax.Array:
    return jax.scipy.linalg.lu_solve(nmat_lu, v)


def _bv_product(
    v: jax.Array,
    theta: FloatScalar,
    s_ord: jax.Array,
    y_ord: jax.Array,
    valid: jax.Array,
    nmat_lu: tuple[jax.Array, jax.Array],
) -> jax.Array:
    wt_v = _wt_times_v(s_ord, y_ord, valid, theta, v)
    m_wt_v = _nmat_solve(nmat_lu, wt_v)
    return theta * v - _w_times_u(s_ord, y_ord, valid, theta, m_wt_v)


def _two_loop_recursion(
    grad: jax.Array,
    s_hist: jax.Array,
    y_hist: jax.Array,
    rho_hist: jax.Array,
    head: IntScalar,
    size: IntScalar,
    *,
    unroll: int = 4,
) -> jax.Array:
    s_ord, y_ord, rho_ord, valid = _chronological_order(
        s_hist, y_hist, rho_hist, head, size
    )

    def reverse_pass(
        q: jax.Array, elems: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        s_i, y_i, rho_i, ok = elems
        alpha_i = jnp.where(ok > 0.5, rho_i * jnp.dot(s_i, q), 0.0)
        return q - alpha_i * y_i, alpha_i

    q, alpha_forward = lax.scan(
        reverse_pass, grad, (s_ord, y_ord, rho_ord, valid), unroll=unroll, reverse=True
    )

    m = s_hist.shape[0]
    newest = (head - jnp.int32(1)) % m
    gamma = jnp.where(
        size > 0,
        jnp.dot(s_hist[newest], y_hist[newest])
        / (jnp.dot(y_hist[newest], y_hist[newest]) + 1e-30),
        1.0,
    )
    r0 = gamma * q

    def forward_pass(
        r: jax.Array,
        elems: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        s_i, y_i, rho_i, alpha_i, ok = elems
        beta_i = jnp.where(ok > 0.5, rho_i * jnp.dot(y_i, r), 0.0)
        return r + jnp.where(ok > 0.5, alpha_i - beta_i, 0.0) * s_i, None

    r, _ = lax.scan(
        forward_pass, r0, (s_ord, y_ord, rho_ord, alpha_forward, valid), unroll=unroll
    )

    return -r


def _build_nmat_lu(
    sy_gram: jax.Array,
    ss_gram: jax.Array,
    theta: FloatScalar,
    head: IntScalar,
    size: IntScalar,
) -> tuple[jax.Array, jax.Array]:
    m = sy_gram.shape[0]

    # reorder grams into chronological order
    id = (head - size + jnp.arange(m, dtype=head.dtype)) % m
    valid = (jnp.arange(m, dtype=size.dtype) < size).astype(sy_gram.dtype)

    sy_ord = sy_gram[id][:, id] * valid[:, None] * valid[None, :]
    ss_ord = ss_gram[id][:, id] * valid[:, None] * valid[None, :]

    D = jnp.diag(jnp.diag(sy_ord))
    L = jnp.tril(sy_ord, -1)

    Nmat = jnp.block([[-D, L.T], [L, theta * ss_ord]])

    diag_reg = jnp.concatenate([valid, valid])
    return jax.scipy.linalg.lu_factor(Nmat + jnp.diag(1.0 - diag_reg))


def _generalized_cauchy_point(
    x: jax.Array,
    g: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    theta: FloatScalar,
    s_ord: jax.Array,
    y_ord: jax.Array,
    valid: jax.Array,
    nmat_lu: tuple[jax.Array, jax.Array],
    *,
    max_segments: int = 10_000,
) -> tuple[jax.Array, jax.Array]:
    n = x.shape[0]
    m = s_ord.shape[0]

    bound = jnp.where(g < 0, upper, lower)
    valid = (g != 0) & jnp.isfinite(bound)

    t_break = jnp.where(valid, (x - bound) / g, jnp.inf)
    order = jnp.argsort(t_break)
    t_sorted = t_break[order]

    d = jnp.where(t_break > 0, -g, 0.0)
    p = _wt_times_v(s_ord, y_ord, valid, theta, d)
    c = jnp.zeros(2 * m, dtype=x.dtype)
    fp = jnp.dot(g, d)
    fpp = -theta * fp - jnp.dot(p, _nmat_solve(nmat_lu, p))

    valid = (fp < 0) & (fpp > 0)
    ratio = -fp / fpp
    dt_min = jnp.where(valid, ratio, jnp.where(fp >= 0, 0.0, jnp.inf))

    def _get_wb(b: IntScalar) -> jax.Array:
        y_row = y_ord[:, b]
        s_row = s_ord[:, b]
        return jnp.concatenate([y_row * valid, theta * s_row * valid])

    def _cond(carry: tuple) -> BoolScalar:
        seg_id, _, _, _, _, _, _, _, _, found = carry
        return (~found) & (seg_id < jnp.minimum(n, max_segments))

    def _body(carry: tuple) -> tuple:
        seg_id, t_old, xc, d, fp, fpp, dt_min, p, c, found = carry
        t_j = t_sorted[seg_id]
        dt = jnp.maximum(t_j - t_old, 0.0)

        inside = (dt_min >= 0) & (dt_min < dt)
        xc_a = xc + dt_min * d
        c_a = c + dt_min * p

        xc_b = xc + dt * d
        c_b = c + dt * p

        b = order[seg_id]
        bound_b = jnp.where(g[b] > 0, lower[b], upper[b])

        xc_b = xc_b.at[b].set(bound_b)
        z_b = bound_b - x[b]

        w_b = _get_wb(b)
        g_b = g[b]

        m_wb = _nmat_solve(nmat_lu, w_b)

        fp_new = (
            fp
            + dt * fpp
            + g_b**2
            + theta * g_b * z_b
            - g_b * jnp.dot(w_b, _nmat_solve(nmat_lu, c_b))
        )

        fpp_new = (
            fpp
            - theta * g_b**2
            - 2.0 * g_b * jnp.dot(m_wb, p)
            - g_b**2 * jnp.dot(w_b, m_wb)
        )

        p_new = p + g_b * w_b
        d_new = d.at[b].set(0.0)

        valid = (fp_new < 0) & (fpp_new > 0)
        ratio = -fp_new / fpp_new
        dt_min_new = jnp.where(
            valid,
            ratio,
            jnp.where(fp_new >= 0, 0.0, jnp.inf),
        )

        xc_out = jnp.where(inside, xc_a, xc_b)
        c_out = jnp.where(inside, c_a, c_b)
        d_out = jnp.where(inside, d, d_new)
        fp_out = jnp.where(inside, fp, fp_new)
        fpp_out = jnp.where(inside, fpp, fpp_new)
        p_out = jnp.where(inside, p, p_new)
        dt_min_out = jnp.where(inside, dt_min, dt_min_new)

        return (
            seg_id + 1,
            t_j,
            xc_out,
            d_out,
            fp_out,
            fpp_out,
            dt_min_out,
            p_out,
            c_out,
            found | inside,
        )

    init = (jnp.int32(0), 0.0, x, d, fp, fpp, dt_min, p, c, jnp.array(False))
    final = lax.while_loop(_cond, _body, init)

    _, _, xc, d_f, _, _, dt_min_f, _, c_aux, found = final
    dt_min_f = jnp.maximum(dt_min_f, 0.0)
    return jnp.clip(xc, lower, upper), c_aux


def _subspace_minimize_cg(
    x: jax.Array,
    xc: jax.Array,
    g: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    theta: FloatScalar,
    s_ord: jax.Array,
    y_ord: jax.Array,
    valid: jax.Array,
    nmat_lu: tuple[jax.Array, jax.Array],
    c_aux: jax.Array,
    *,
    max_cg_iters: int = 50,
) -> jax.Array:
    dtype = x.dtype
    free_f = (
        ((xc > lower) | ~jnp.isfinite(lower)) & ((xc < upper) | ~jnp.isfinite(upper))
    ).astype(dtype)

    mc = _nmat_solve(nmat_lu, c_aux)
    wmc = _w_times_u(s_ord, y_ord, valid, theta, mc)
    rc = (g + theta * (xc - x) - wmc) * free_f
    rc_norm = jnp.linalg.norm(rc)
    tol = jnp.minimum(0.1, jnp.sqrt(rc_norm)) * rc_norm

    def _bff(v: jax.Array) -> jax.Array:
        return free_f * _bv_product(v, theta, s_ord, y_ord, valid, nmat_lu)

    def _cond(
        carry: tuple[jax.Array, jax.Array, jax.Array, FloatScalar, IntScalar],
    ) -> BoolScalar:
        _, _, r, _, i = carry
        return (jnp.linalg.norm(r) > tol) & (i < max_cg_iters)

    def _body(
        carry: tuple[jax.Array, jax.Array, jax.Array, FloatScalar, IntScalar],
    ) -> tuple:
        d_hat, p_cg, r_hat, rho1, i = carry
        q = _bff(p_cg)
        pTq = jnp.dot(p_cg, q)
        alpha2 = jnp.where(pTq > 0, rho1 / (pTq + 1e-30), 0.0)

        alpha1_ub = jnp.where(
            p_cg > 1e-30,
            (upper - xc - d_hat) / p_cg,
            jnp.where(p_cg < -1e-30, (lower - xc - d_hat) / p_cg, jnp.inf),
        )
        alpha1 = jnp.maximum(jnp.min(jnp.where(free_f > 0.5, alpha1_ub, jnp.inf)), 0.0)

        truncated = alpha2 > alpha1
        alpha_use = jnp.where(truncated, alpha1, alpha2)

        d_new = d_hat + alpha_use * p_cg
        r_new = (r_hat + alpha_use * q) * free_f
        rho2 = jnp.dot(r_new, r_new)
        beta = rho2 / (rho1 + 1e-30)
        p_new = (-r_new + beta * p_cg) * free_f
        r_out = jnp.where(truncated, jnp.zeros_like(r_new), r_new)
        return d_new, p_new, r_out, rho2, i + 1

    init = (jnp.zeros_like(x), -rc, rc, jnp.dot(rc, rc), jnp.int32(0))
    d_final, _, _, _, _ = lax.while_loop(_cond, _body, init)
    return jnp.clip(xc + d_final * free_f, lower, upper)


def _bounded_line_search(
    x: jax.Array,
    f: FloatScalar,
    g: jax.Array,
    d: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    value_and_grad: FlatValueGrad,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_steps: int = 20,
) -> tuple[BoolScalar, jax.Array, FloatScalar, jax.Array]:
    pos_step = jnp.where((d > 0) & jnp.isfinite(upper), (upper - x) / d, jnp.inf)
    neg_step = jnp.where((d < 0) & jnp.isfinite(lower), (lower - x) / d, jnp.inf)
    alpha0 = jnp.where(
        jnp.isfinite(jnp.min(jnp.minimum(pos_step, neg_step))),
        jnp.minimum(1.0, jnp.min(jnp.minimum(pos_step, neg_step))),
        1.0,
    )
    slope0 = jnp.dot(g, d)

    def cond(carry: tuple) -> BoolScalar:
        i, alpha, accepted, _, _, _ = carry
        return (~accepted) & (i < max_steps) & (alpha > 1e-16)

    def body(carry: tuple) -> tuple:
        i, alpha, accepted, x_acc, f_acc, g_acc = carry
        x_trial = jnp.clip(x + alpha * d, lower, upper)
        f_trial, g_trial = value_and_grad(x_trial)
        ok: BoolScalar = f_trial <= f + c1 * alpha * slope0
        return (
            i + 1,
            jnp.where(ok, alpha, 0.5 * alpha),
            accepted | ok,
            jnp.where(ok, x_trial, x_acc),
            jnp.where(ok, f_trial, f_acc),
            jnp.where(ok, g_trial, g_acc),
        )

    init = (jnp.int32(0), alpha0, jnp.array(False), x, f, g)
    _, _, accepted, x_new, f_new, g_new = lax.while_loop(cond, body, init)
    return accepted, x_new, f_new, g_new


def _push_pair(
    s: jax.Array,
    y: jax.Array,
    s_hist: jax.Array,
    y_hist: jax.Array,
    rho_hist: jax.Array,
    sy_gram: jax.Array,
    ss_gram: jax.Array,
    head: IntScalar,
    size: IntScalar,
    curvature_eps: float = 2.2e-16,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, IntScalar, IntScalar]:
    sty: FloatScalar = jnp.dot(s, y)
    yty: FloatScalar = jnp.dot(y, y)
    keep: BoolScalar = sty > curvature_eps * yty

    rho: FloatScalar = 1.0 / (sty + 1e-30)
    m: int = s_hist.shape[0]
    head_next: IntScalar = (head + 1) % m
    size_next: IntScalar = jnp.minimum(size + 1, m)

    s_new = jnp.where(keep, s_hist.at[head].set(s), s_hist)
    y_new = jnp.where(keep, y_hist.at[head].set(y), y_hist)
    rho_new = jnp.where(keep, rho_hist.at[head].set(rho), rho_hist)

    sy_row = y_new @ s
    sy_col = s_new @ y
    sy_up = sy_gram.at[head, :].set(sy_row).at[:, head].set(sy_col)
    sy_up = sy_up.at[head, head].set(sty)

    ss_row = s_new @ s
    ss_up = ss_gram.at[head, :].set(ss_row).at[:, head].set(ss_row)
    ss_up = ss_up.at[head, head].set(jnp.dot(s, s))

    sy_out = jnp.where(keep, sy_up, sy_gram)
    ss_out = jnp.where(keep, ss_up, ss_gram)
    head_out = jnp.where(keep, head_next, head)
    size_out = jnp.where(keep, size_next, size)
    return s_new, y_new, rho_new, sy_out, ss_out, head_out, size_out


def _lbfgsb_step(
    state: LBFGSBState,
    lower: jax.Array,
    upper: jax.Array,
    value_and_grad: FlatValueGrad,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    gtol: float = 1e-5,
    curvature_eps: float = 2.2e-16,
    max_line_search_steps: int = 20,
    max_cauchy_segments: int = 10_000,
    max_cg_iters: int = 50,
) -> tuple[LBFGSBState, LBFGSBTrace]:
    pg_inf = convergence_measure(state.x, state.g, lower, upper)

    def _converged(_: None) -> tuple[LBFGSBState, LBFGSBTrace]:
        return (
            state._replace(k=state.k + 1, converged=jnp.array(True)),
            LBFGSBTrace(
                f=state.f,
                proj_grad_inf=pg_inf,
                step_norm=jnp.zeros((), dtype=state.x.dtype),
                accepted=jnp.array(True),
            ),
        )

    def _iterate(_: None) -> tuple[LBFGSBState, LBFGSBTrace]:
        # Compute once, pass everywhere
        theta = _compute_theta(state.s_hist, state.y_hist, state.head, state.size)
        s_ord, y_ord, _, valid = _chronological_order(
            state.s_hist, state.y_hist, state.rho_hist, state.head, state.size
        )
        nmat_lu = _build_nmat_lu(
            state.sy_gram, state.ss_gram, theta, state.head, state.size
        )

        xc, c_aux = _generalized_cauchy_point(
            state.x,
            state.g,
            lower,
            upper,
            theta,
            s_ord,
            y_ord,
            valid,
            nmat_lu,
            max_segments=max_cauchy_segments,
        )

        z = _subspace_minimize_cg(
            state.x,
            xc,
            state.g,
            lower,
            upper,
            theta,
            s_ord,
            y_ord,
            valid,
            nmat_lu,
            c_aux,
            max_cg_iters=max_cg_iters,
        )

        d_dir = z - state.x
        accepted, x_new, f_new, g_new = _bounded_line_search(
            state.x,
            state.f,
            state.g,
            d_dir,
            lower,
            upper,
            value_and_grad,
            c1=c1,
            c2=c2,
            max_steps=max_line_search_steps,
        )

        s = x_new - state.x
        y = g_new - state.g
        s_h, y_h, rho_h, sy_g, ss_g, hd, sz = _push_pair(
            s,
            y,
            state.s_hist,
            state.y_hist,
            state.rho_hist,
            state.sy_gram,
            state.ss_gram,
            state.head,
            state.size,
            curvature_eps=curvature_eps,
        )

        new_state = LBFGSBState(
            x=x_new,
            f=f_new,
            g=g_new,
            s_hist=s_h,
            y_hist=y_h,
            rho_hist=rho_h,
            sy_gram=sy_g,
            ss_gram=ss_g,
            head=hd,
            size=sz,
            k=state.k + 1,
            converged=jnp.array(False),
        )
        return new_state, LBFGSBTrace(
            f=f_new,
            proj_grad_inf=pg_inf,
            step_norm=jnp.linalg.norm(s),
            accepted=accepted,
        )

    return lax.cond(pg_inf <= gtol, _converged, _iterate, operand=None)


@jax.jit(
    static_argnames=(
        "loss_fn",
        "max_iters",
        "memory",
        "c1",
        "c2",
        "gtol",
        "curvature_eps",
        "max_line_search_steps",
        "max_cauchy_segments",
        "max_cg_iters",
    )
)
def solve_lbfgsb(
    params0: PyTree,
    lower_tree: PyTree,
    upper_tree: PyTree,
    loss_fn: Callable[[PyTree], jax.Array],
    *,
    max_iters: int = 200,
    memory: int = 10,
    c1: float = 1e-4,
    c2: float = 0.9,
    gtol: float = 1e-5,
    curvature_eps: float = 2.2e-16,
    max_line_search_steps: int = 20,
    max_cauchy_segments: int = 10_000,
    max_cg_iters: int = 50,
) -> tuple[PyTree, LBFGSBState, LBFGSBTrace]:
    x0, unravel, vg = make_flat_value_and_grad(loss_fn, params0)
    lower, _ = ravel_pytree(lower_tree)
    upper, _ = ravel_pytree(upper_tree)
    lower = jnp.asarray(lower, dtype=x0.dtype)
    upper = jnp.asarray(upper, dtype=x0.dtype)
    f0, g0 = vg(x0)
    n = x0.shape[0]

    init = LBFGSBState(
        x=x0,
        f=f0,
        g=g0,
        s_hist=jnp.zeros((memory, n), dtype=x0.dtype),
        y_hist=jnp.zeros((memory, n), dtype=x0.dtype),
        rho_hist=jnp.zeros((memory,), dtype=x0.dtype),
        sy_gram=jnp.zeros((memory, memory), dtype=x0.dtype),
        ss_gram=jnp.zeros((memory, memory), dtype=x0.dtype),
        head=jnp.int32(0),
        size=jnp.int32(0),
        k=jnp.int32(0),
        converged=jnp.array(False),
    )

    def body(state: LBFGSBState, _: None) -> tuple[LBFGSBState, LBFGSBTrace]:
        return lax.cond(
            state.converged,
            lambda s: (
                s,
                LBFGSBTrace(
                    f=s.f,
                    proj_grad_inf=convergence_measure(s.x, s.g, lower, upper),
                    step_norm=jnp.zeros((), dtype=s.x.dtype),
                    accepted=jnp.array(True),
                ),
            ),
            lambda s: _lbfgsb_step(
                s,
                lower,
                upper,
                vg,
                c1=c1,
                c2=c2,
                gtol=gtol,
                curvature_eps=curvature_eps,
                max_line_search_steps=max_line_search_steps,
                max_cauchy_segments=max_cauchy_segments,
                max_cg_iters=max_cg_iters,
            ),
            state,
        )

    final, trace = lax.scan(body, init, xs=None, length=max_iters)
    return unravel(final.x), final, trace


def scale_by_lbfgsb(
    memory_size: int = 10,
    scale_init_precond: bool = True,
) -> optax.GradientTransformation:
    def init_fn(params: PyTree) -> ScaleLBFGSBState:
        flat, _ = ravel_pytree(params)
        n, dtype = flat.shape[0], flat.dtype
        return ScaleLBFGSBState(
            s_hist=jnp.zeros((memory_size, n), dtype=dtype),
            y_hist=jnp.zeros((memory_size, n), dtype=dtype),
            rho_hist=jnp.zeros((memory_size,), dtype=dtype),
            head=jnp.int32(0),
            size=jnp.int32(0),
            num_updates=jnp.int32(0),
            prev_params=params,
            prev_grad=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(
        updates: PyTree, state: ScaleLBFGSBState, params: PyTree
    ) -> tuple[PyTree, ScaleLBFGSBState]:
        flat_g, unravel_g = ravel_pytree(updates)
        flat_p, _ = ravel_pytree(params)
        flat_prev_p, _ = ravel_pytree(state.prev_params)
        flat_prev_g, _ = ravel_pytree(state.prev_grad)
        s, y = flat_p - flat_prev_p, flat_g - flat_prev_g
        sty = jnp.dot(s, y)
        yty = jnp.dot(y, y)
        keep = sty > 2.2e-16 * yty
        rho = 1.0 / (sty + 1e-30)
        m = state.s_hist.shape[0]
        s_h = jnp.where(keep, state.s_hist.at[state.head].set(s), state.s_hist)
        y_h = jnp.where(keep, state.y_hist.at[state.head].set(y), state.y_hist)
        rho_h = jnp.where(keep, state.rho_hist.at[state.head].set(rho), state.rho_hist)
        hd = jnp.where(keep, (state.head + 1) % m, state.head)
        sz = jnp.where(keep, jnp.minimum(state.size + 1, m), state.size)

        direction = _two_loop_recursion(flat_g, s_h, y_h, rho_h, hd, sz)
        init_scale = jnp.where(
            scale_init_precond & (sz == 0), 1.0 / (jnp.linalg.norm(flat_g) + 1e-30), 1.0
        )
        direction = jnp.where(sz == 0, -init_scale * flat_g, direction)

        new_state = ScaleLBFGSBState(
            s_hist=s_h,
            y_hist=y_h,
            rho_hist=rho_h,
            head=hd,
            size=sz,
            num_updates=state.num_updates + 1,
            prev_params=params,
            prev_grad=updates,
        )
        return unravel_g(direction), new_state

    return optax.GradientTransformation(init_fn, update_fn)


def lbfgsb(
    lower: PyTree | None = None,
    upper: PyTree | None = None,
    *,
    learning_rate: optax.ScalarOrSchedule | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    linesearch: optax.GradientTransformation | None = None,
) -> optax.GradientTransformationExtraArgs:
    if linesearch is None:
        linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=20)
    inner_parts = [scale_by_lbfgsb(memory_size, scale_init_precond)]
    if learning_rate is not None:
        inner_parts.append(optax.scale_by_learning_rate(learning_rate))
    inner_parts.append(linesearch)
    inner = optax.chain(*inner_parts)

    def init_fn(params: PyTree) -> Any:
        return inner.init(params)

    def update_fn(
        updates: PyTree, state: Any, params: PyTree, **extra_args: Any
    ) -> tuple[PyTree, Any]:
        new_updates, new_state = inner.update(updates, state, params, **extra_args)
        if lower is not None or upper is not None:

            def _project(p: jax.Array, u: jax.Array) -> jax.Array:
                result = p + u
                if lower is not None:
                    result = jnp.maximum(result, jnp.asarray(lower, dtype=result.dtype))
                if upper is not None:
                    result = jnp.minimum(result, jnp.asarray(upper, dtype=result.dtype))
                return result - p

            new_updates = jax.tree.map(_project, params, new_updates)
        return new_updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
