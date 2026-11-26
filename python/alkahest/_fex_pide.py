"""Partial integro-differential equations for FEX (FEX-PG).

Hardwick, Liang and Yang (*Solving high-dimensional partial integral
differential equations: the finite expression method*, arXiv 2410.00835,
JCP 2025) extend FEX to PIDEs from Lévy-driven SDEs,

.. math::

    u_t + b(x)\\cdot\\nabla u + \\tfrac12 \\mathrm{Tr}(\\sigma\\sigma^T \\nabla^2 u)
    + \\int \\big(u(x + G(x, z)) - u(x) - G(x, z)\\cdot\\nabla u(x)\\big)\\,\\nu(dz)
    = f(t, x),\\qquad u(T, x) = g(x),

with ``ν(dz) = λ φ(z) dz``. Their two devices are a **Taylor-moment
replacement** of the integral term and **parameter grouping**. This module
provides both, plus what they do not have: an *exact* expectation of the
jump term for closed-form candidates, so :func:`alkahest.fex.finalize` can
prove a recovered solution against the true integral operator.

* :class:`GaussianJump` — jumps ``G = z`` (additive) or ``G = x(e^z − 1)``
  (multiplicative, ``x ↦ x e^z``), with i.i.d. ``N(μ, σ²)`` components and
  intensity ``λ``. Its constants are kept exact (a float literal is read as
  the decimal it prints as: ``0.3`` is ``3/10``).
* :func:`taylor_moment` — the replacement, done symbolically: the jump term
  becomes the **local** part ``−λu − λ E[G]·∇u`` (added to the operator as
  an expression in the jet symbols) plus a **nonlocal** part
  ``λ (u(x + μ) + ½σ² Δu(x + μ))`` evaluated at shifted points by the
  Rust interpreter (one :class:`NonlocalTerm`), exactly as in the paper's
  eq. (23). It is *exact* — not an approximation — on every candidate that
  is a polynomial of degree ≤ 3 in each coordinate (the third central
  moment of a Gaussian vanishes and nothing of order 4 is left), and for
  ``σ = 0``. A multiplicative jump uses an ``n``-node Gauss–Hermite rule in
  ``z`` instead (the paper uses the trapezoid rule in 1-D).
* :func:`gaussian_expectation` — ``E[u(x + z)]`` (or ``E[u(x e^z)]``) of a
  closed form, in closed form, from the Gaussian moment generating
  function: polynomials in ``z`` times ``exp`` of linear forms, and
  ``sin`` / ``cos`` of linear forms. Used by ``finalize`` to verify a PIDE
  solution exactly, at the fitted ``d`` and (for affine-in-sums
  candidates) for symbolic ``d``; anything outside that class is reported
  as not verifiable rather than approximated.
* Drift and diffusion: ``drift`` is a list of expressions (one per spatial
  coordinate) or :class:`RadialDrift` (``b = β(|x|)·x``, which lifts to
  symbolic ``d``); ``diffusion`` is a number ``θ`` (``σ = θ I``, so the
  diffusion term is ``½θ²Δu``), :class:`BidiagonalDiffusion` (the paper's
  ``σ = θ(I + c·S)``, ``S`` the subdiagonal shift) or a matrix. A
  non-diagonal ``σσᵀ`` is passed to the interpreter as a
  :class:`SecondOrderForm`, so ``Tr(σσᵀ∇²u)`` is exact in the fused
  forward pass, with no off-diagonal Hessian ever formed.

Parameter grouping is :func:`alkahest.fex.finalize`'s tying step (leaf
weights that agree become one parameter, refitted under the loss gate);
:func:`group_weights` exposes the paper's clustering of a weight vector with
threshold ``η`` for comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

import alkahest as _ak

__all__ = [
    "BidiagonalDiffusion",
    "GaussianJump",
    "RadialDrift",
    "gaussian_expectation",
    "group_weights",
    "taylor_moment",
]

_REMEDIATION = (
    "give each nonlocal term one shift (and optionally one scale) per coordinate and finite "
    "coefficients, a second-order form with one diagonal entry per coordinate and off-diagonal "
    "entries (i, j, a) with i < j, a jump law with a finite mean and variance, and use a "
    "least-squares or centred loss"
)


def _err(message: str) -> Exception:
    from .alkahest import FexError

    exc = FexError(f"[E-FEX-015] {message}\nRemediation: {_REMEDIATION}")
    exc.code = "E-FEX-015"
    exc.remediation = _REMEDIATION
    exc.span = None
    return exc


# ---------------------------------------------------------------------------
# Exact constants
# ---------------------------------------------------------------------------


def exact_number(v: Any) -> Fraction | None:
    """``v`` as an exact rational: ints and Fractions as they are, a float as
    the decimal it prints as (``0.3`` → ``3/10``); ``None`` for an Expr."""
    if isinstance(v, bool):
        raise TypeError("a boolean is not a PIDE constant")
    if isinstance(v, (int, Fraction)):
        return Fraction(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise _err(f"the constant {v!r} is not finite")
        return Fraction(repr(v))
    return None


def to_expr(v: Any, pool: Any) -> Any:
    """An exact Expr for a number (see :func:`exact_number`) or an Expr."""
    q = exact_number(v)
    if q is None:
        return v
    if q.denominator == 1:
        return pool.integer(q.numerator)
    return pool.rational(q.numerator, q.denominator)


def _float(v: Any) -> float:
    q = exact_number(v)
    if q is not None:
        return float(q)
    return float(_ak.evalf(v)) if hasattr(_ak, "evalf") else float(str(v))


# ---------------------------------------------------------------------------
# The jump law, drift and diffusion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianJump:
    """Jumps with i.i.d. ``N(mean, var)`` components on the spatial
    coordinates, arriving at rate ``rate`` (``ν(dz) = rate · φ(z) dz``).

    ``multiplicative=False``: ``G(x, z) = z`` (``u(x + z)``);
    ``multiplicative=True``: ``G(x, z) = x(e^z − 1)`` (``u(x e^z)``, the
    option-pricing jumps of the paper's first examples). ``nodes`` is the
    Gauss–Hermite rule used for a multiplicative jump in the search."""

    rate: Any
    mean: Any
    var: Any
    multiplicative: bool = False
    nodes: int = 24

    def __post_init__(self) -> None:
        for name in ("rate", "mean", "var"):
            v = getattr(self, name)
            q = exact_number(v)
            if q is None and not isinstance(v, _ak.Expr):
                raise _err(f"GaussianJump.{name} must be a number or an Expr, got {v!r}")
        if exact_number(self.var) is not None and exact_number(self.var) < 0:
            raise _err("GaussianJump.var must be non-negative")
        if self.nodes < 1 or self.nodes > 200:
            raise _err("GaussianJump.nodes must be between 1 and 200")

    def floats(self) -> tuple[float, float, float]:
        return _float(self.rate), _float(self.mean), _float(self.var)

    def exprs(self, pool: Any) -> tuple[Any, Any, Any]:
        return to_expr(self.rate, pool), to_expr(self.mean, pool), to_expr(self.var, pool)


@dataclass(frozen=True)
class RadialDrift:
    """``b(x) = β(|x|) · x`` over the spatial coordinates; ``beta`` is a
    number or a callable of ``r = |x|`` (an Expr) returning an Expr. Lifts to
    symbolic ``d``."""

    beta: Any

    def coefficient(self, r: Any, pool: Any) -> Any:
        return self.beta(r) if callable(self.beta) else to_expr(self.beta, pool)


@dataclass(frozen=True)
class BidiagonalDiffusion:
    """``σ = θ (I + c·S)`` with ``S`` the subdiagonal shift (``σ`` lower
    bidiagonal: ``θ`` on the diagonal, ``c·θ`` below it) — the diffusion of
    the paper's second high-dimensional example (``c = 1``)."""

    theta: Any
    sub: Any = 1

    def matrix(self, n: int, pool: Any) -> list[list[Any]]:
        th = to_expr(self.theta, pool)
        c = to_expr(self.sub, pool)
        zero = pool.integer(0)
        m = [[zero] * n for _ in range(n)]
        for i in range(n):
            m[i][i] = th
            if i > 0:
                m[i][i - 1] = c * th
        return m


def _diffusion_matrix(diffusion: Any, n: int, pool: Any) -> list[list[Any]] | None:
    """``σ`` as an exact ``n × n`` matrix (``None`` for no diffusion)."""
    if diffusion is None:
        return None
    if isinstance(diffusion, BidiagonalDiffusion):
        return diffusion.matrix(n, pool)
    if exact_number(diffusion) is not None or isinstance(diffusion, _ak.Expr):
        th = to_expr(diffusion, pool)
        zero = pool.integer(0)
        return [[th if i == j else zero for j in range(n)] for i in range(n)]
    rows = [list(r) for r in diffusion]
    if len(rows) != n or any(len(r) != n for r in rows):
        raise _err(f"the diffusion matrix must be {n} × {n}")
    return [[to_expr(v, pool) for v in r] for r in rows]


def _half_sigma_sigma_t(sig: list[list[Any]], pool: Any) -> list[list[Any]]:
    n = len(sig)
    half = pool.rational(1, 2)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            acc = None
            for k in range(n):
                a, b = sig[i][k], sig[j][k]
                if _ak.is_zero(a) is True or _ak.is_zero(b) is True:
                    continue
                t = a * b
                acc = t if acc is None else acc + t
            row.append(pool.integer(0) if acc is None else _ak.simplify(half * acc).value)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# The Taylor-moment replacement
# ---------------------------------------------------------------------------


def taylor_moment(jump: GaussianJump, jet: Any, spatial: list[int]) -> dict:
    """The jump term ``λ(E[u(x + G)] − u − E[G]·∇u)`` in the form the
    interpreter evaluates.

    Returns ``{"local": Expr in the jet symbols, "nonlocal": [(scale, shift,
    c_u, c_Δ)], "formula": str, "exact_for": str}``. Additive jumps follow
    the paper's eq. (23) (second-order Taylor expansion about ``x + μ``);
    multiplicative jumps use a Gauss–Hermite rule in ``z`` and the exact
    ``E[e^z] = e^{μ + σ²/2}`` in the local part."""
    pool = jet.pool
    d = len(jet.x)
    lam, mu, var = jump.exprs(pool)
    lam_f, mu_f, var_f = jump.floats()
    u = jet.u
    if not jump.multiplicative:
        gsum = None
        for i in spatial:
            gsum = jet.grad[i] if gsum is None else gsum + jet.grad[i]
        local = -lam * u - lam * mu * gsum
        shift = [mu_f if i in spatial else 0.0 for i in range(d)]
        terms = [([], shift, lam_f, 0.5 * lam_f * var_f)]
        formula = "λ·(u(x + μ) + ½σ²·Δu(x + μ)) − λ·u − λμ·Σᵢ ∂ᵢu"
        exact_for = (
            "polynomials of degree ≤ 3 in each coordinate (and every u when σ = 0); "
            "otherwise the paper's second-order approximation"
        )
    else:
        em1 = _ak.exp(mu + var / 2) - 1
        xg = None
        for i in spatial:
            t = jet.x[i] * jet.grad[i]
            xg = t if xg is None else xg + t
        local = -lam * u - lam * em1 * xg
        # Gauss–Hermite: E[f(z)] = Σ w_k/√π f(μ + √2 σ t_k).
        t_k, w_k = np.polynomial.hermite.hermgauss(jump.nodes)
        sd = math.sqrt(var_f)
        terms = []
        for t, w in zip(t_k, w_k):
            s = math.exp(mu_f + math.sqrt(2.0) * sd * t)
            scale = [s if i in spatial else 1.0 for i in range(d)]
            terms.append((scale, [0.0] * d, lam_f * w / math.sqrt(math.pi), 0.0))
            if sd == 0.0:
                terms = [(scale, [0.0] * d, lam_f, 0.0)]
                break
        formula = (
            f"λ·Σₖ wₖ u(x·e^{{zₖ}}) ({jump.nodes}-node Gauss–Hermite) − λ·u − "
            "λ(e^{μ+σ²/2} − 1)·Σᵢ xᵢ∂ᵢu"
        )
        exact_for = "u polynomial in x of degree below 2·nodes in each coordinate"
    return {"local": local, "nonlocal": terms, "formula": formula, "exact_for": exact_for}


def group_weights(w: Any, eta: float) -> list[int]:
    """The paper's parameter grouping (§3.3): hierarchical clustering of a
    leaf's weight vector with distance threshold ``eta`` (single linkage on
    sorted values, which is what SciPy's ``fcluster(..., 'distance')`` gives
    in 1-D); returns one cluster label per weight, ``0 …``."""
    w = np.asarray(w, dtype=np.float64).ravel()
    order = np.argsort(w)
    labels = np.empty(len(w), dtype=int)
    lab = 0
    for k, idx in enumerate(order):
        if k > 0 and w[idx] - w[order[k - 1]] > eta:
            lab += 1
        labels[idx] = lab
    return [int(v) for v in labels]


# ---------------------------------------------------------------------------
# Building the problem
# ---------------------------------------------------------------------------


def build(
    dim: int,
    *,
    time: int,
    drift: Any,
    diffusion: Any,
    jump: GaussianJump | None,
    rhs: Any,
    terminal: Any,
    T: Any,
    bc_weight: float,
    extra: Any,
    jet: Any,
) -> tuple[Any, dict, dict]:
    """``(operator Expr, Problem kwargs, spec["pide"])`` for
    :meth:`alkahest.fex.Problem.pide`."""
    pool = jet.pool
    x = jet.x
    spatial = [i for i in range(dim) if i != time]
    n = len(spatial)
    op = jet.ut
    # Drift b·∇u.
    drift_desc: Any = None
    if drift is not None:
        if isinstance(drift, RadialDrift):
            r2 = None
            for i in spatial:
                r2 = x[i] ** 2 if r2 is None else r2 + x[i] ** 2
            beta = drift.coefficient(r2 ** pool.rational(1, 2), pool)
            if _ak.is_zero(beta) is not True:
                bg = None
                for i in spatial:
                    t = x[i] * jet.grad[i]
                    bg = t if bg is None else bg + t
                op = op + beta * bg
            drift_desc = drift
        else:
            bs = list(drift(x) if callable(drift) else drift)
            if len(bs) != n:
                raise _err(f"the drift needs {n} components (one per spatial coordinate)")
            for i, b in zip(spatial, bs):
                b = to_expr(b, pool)
                if _ak.is_zero(b) is True:
                    continue
                op = op + b * jet.grad[i]
            drift_desc = [to_expr(b, pool) for b in bs]
    # Diffusion ½Tr(σσᵀ∇²u).
    sig = _diffusion_matrix(diffusion, n, pool)
    a_exact = None if sig is None else _half_sigma_sigma_t(sig, pool)
    second_order = None
    if a_exact is not None:
        diag = [a_exact[k][k] for k in range(n)]
        off = [
            (spatial[i], spatial[j], a_exact[i][j])
            for i in range(n)
            for j in range(i + 1, n)
            if _ak.is_zero(a_exact[i][j]) is not True
        ]
        same = all(_ak.is_zero(v - diag[0]) is True for v in diag)
        if same and not off:
            op = op + diag[0] * jet.lap
        else:
            dvals = [0.0] * dim
            for k, i in enumerate(spatial):
                dvals[i] = _float(_num_or_eval(diag[k]))
            second_order = (dvals, [(i, j, _float(_num_or_eval(c))) for i, j, c in off])
            op = op + jet.lap
    # The jump term.
    nonlocal_terms: list = []
    taylor = None
    if jump is not None:
        taylor = taylor_moment(jump, jet, spatial)
        op = op + taylor["local"]
        nonlocal_terms = taylor["nonlocal"]
    if extra is not None:
        op = op + (extra(jet) if callable(extra) else extra)
    kw = {
        "nonlocal_terms": nonlocal_terms or None,
        "second_order": second_order,
    }
    desc = {
        "time": time,
        "T": to_expr(T, pool),
        "drift": drift_desc,
        "diffusion": diffusion,
        "sigma": sig,
        "a": a_exact,
        "jump": jump,
        "taylor": None if taylor is None else {k: taylor[k] for k in ("formula", "exact_for")},
        "extra": extra,
    }
    return op, kw, desc


def _num_or_eval(e: Any) -> Any:
    from ._fex_finalize import _number_value

    v = _number_value(e)
    if v is not None:
        return v
    return float(_ak.simplify(e).value.node()[1]) if e.node()[0] == "float" else e


# ---------------------------------------------------------------------------
# Exact Gaussian expectations
# ---------------------------------------------------------------------------


class NoClosedForm(Exception):
    """``E[u(x + z)]`` is not in the class :func:`gaussian_expectation` handles."""


def _kids(e: Any) -> list[Any]:
    from ._fex_finalize import _children

    return _children(e)


def _free(e: Any, zs: set) -> bool:
    from ._fex_finalize import _symbols_in

    return not (_symbols_in(e) & zs)


def _linear_coeffs(arg: Any, zs: list[Any]) -> tuple[Any, list[Any]] | None:
    """``arg = a₀ + Σ cᵢ zᵢ`` with ``a₀, cᵢ`` free of the ``z``'s, or ``None``."""
    zset = set(zs)
    cs = []
    for z in zs:
        c = _ak.simplify(_ak.diff(arg, z).value).value
        if not _free(c, zset):
            return None
        cs.append(c)
    a0 = _ak.subs(arg, {z: z._pool.integer(0) for z in zs})
    return a0, cs


def _mgf_derivative(k: int, c: Any, mean: Any, var: Any, pool: Any) -> Any:
    """``E[z^k e^{c z}]`` for ``z ~ N(mean, var)``: the ``k``-th derivative of
    the moment generating function ``exp(c·mean + c²·var/2)`` at ``c``."""
    s = pool.symbol("__fex_mgf_c")
    m = _ak.exp(s * mean + s**2 * var / 2)
    for _ in range(k):
        m = _ak.diff(m, s).value
    return _ak.subs(m, {s: c})


def _term_expectation(term: Any, zs: list[Any], mean: Any, var: Any, pool: Any) -> Any:
    zset = set(zs)
    nd = term.node()
    factors = _kids(term) if nd[0] == "mul" else [term]
    const = pool.integer(1)
    powers = dict.fromkeys(zs, 0)
    exp_arg = None
    trig: list[tuple[str, Any]] = []
    for f in factors:
        if _free(f, zset):
            const = const * f
            continue
        fn = f.node()
        if fn[0] == "symbol" and f in zset:
            powers[f] += 1
            continue
        if fn[0] == "pow":
            b, ex = _kids(f)
            k = ex.node()
            if b in zset and k[0] == "integer" and int(k[1]) >= 0:
                powers[b] += int(k[1])
                continue
            if b.node()[0] == "func" and b.node()[1] == "exp" and k[0] == "integer":
                (a,) = _kids(b)
                t = a * ex
                exp_arg = t if exp_arg is None else exp_arg + t
                continue
            raise NoClosedForm(f"factor {f}")
        if fn[0] == "func" and fn[1] == "exp":
            (a,) = _kids(f)
            exp_arg = a if exp_arg is None else exp_arg + a
            continue
        if fn[0] == "func" and fn[1] in ("sin", "cos"):
            (a,) = _kids(f)
            trig.append((fn[1], a))
            continue
        raise NoClosedForm(f"factor {f}")
    if trig:
        # cos/sin of a linear form, alone in its z's: the characteristic
        # function. E[cos(a + c·z)] = e^{-c²σ²/2} cos(a + c·μ), same for sin.
        if len(trig) > 1 or exp_arg is not None or any(powers.values()):
            raise NoClosedForm("a trigonometric factor shares its jump with another factor")
        name, a = trig[0]
        lc = _linear_coeffs(a, zs)
        if lc is None:
            raise NoClosedForm(f"{name} of a non-linear function of the jump")
        a0, cs = lc
        c2 = None
        csum = None
        for c in cs:
            if _ak.is_zero(c) is True:
                continue
            c2 = c**2 if c2 is None else c2 + c**2
            csum = c if csum is None else csum + c
        if csum is None:
            return const * term._pool.func(name, [a0])
        return const * _ak.exp(-c2 * var / 2) * term._pool.func(name, [a0 + csum * mean])
    out = const
    cs = [pool.integer(0)] * len(zs)
    if exp_arg is not None:
        lc = _linear_coeffs(exp_arg, zs)
        if lc is None:
            raise NoClosedForm("exp of a non-linear function of the jump")
        a0, cs = lc
        out = out * _ak.exp(a0)
    for z, c in zip(zs, cs):
        k = powers[z]
        if k == 0 and _ak.is_zero(c) is True:
            continue
        out = out * _mgf_derivative(k, c, mean, var, pool)
    return out


def _sum_terms(e: Any, cap: int = 20000) -> list[Any]:
    """``e`` as a list of product terms (``expand`` may keep a numeric
    content factor outside a sum: ``¼·(x² + 2xz + …)``)."""
    nd = e.node()
    if nd[0] == "add":
        out: list[Any] = []
        for c in _kids(e):
            out.extend(_sum_terms(c, cap))
        return out
    if nd[0] == "mul":
        acc: list[Any] = [None]
        for c in _kids(e):
            ts = _sum_terms(c, cap)
            if len(acc) * len(ts) > cap:
                raise NoClosedForm("the expanded candidate is too large")
            acc = [t if a is None else a * t for a in acc for t in ts]
        return acc if acc != [None] else [e]
    return [e]


def gaussian_expectation(
    expr: Any,
    coords: list[Any],
    mean: Any,
    var: Any,
    *,
    multiplicative: bool = False,
) -> Any:
    """``E[u(x + z)]`` (or ``E[u(x e^z)]``) for i.i.d. ``zᵢ ~ N(mean, var)``,
    one per entry of ``coords``, in closed form.

    Handles every term that is (after expansion) a product of factors free of
    ``z``, powers of the ``zᵢ``, ``exp`` of linear forms in ``z`` and at most
    one ``sin`` / ``cos`` of a linear form (alone in its jumps). Raises
    :class:`NoClosedForm` otherwise — it never approximates."""
    pool = coords[0]._pool
    zs = [pool.symbol(f"__fex_z{k}") for k in range(len(coords))]
    if multiplicative:
        mp = {x: x * _ak.exp(z) for x, z in zip(coords, zs)}
    else:
        mp = {x: x + z for x, z in zip(coords, zs)}
    e = _ak.expand(_ak.subs(expr, mp))
    terms = _sum_terms(e)
    out = None
    for t in terms:
        v = _term_expectation(t, zs, mean, var, pool)
        out = v if out is None else out + v
    return pool.integer(0) if out is None else out


def jump_operator_exact(
    u: Any, coords: list[Any], spatial: list[int], jump: GaussianJump, pool: Any
) -> Any:
    """``λ(E[u(x + G)] − u − E[G]·∇u)`` of a closed form, exactly."""
    lam, mu, var = jump.exprs(pool)
    xs = [coords[i] for i in spatial]
    eu = gaussian_expectation(u, xs, mu, var, multiplicative=jump.multiplicative)
    if jump.multiplicative:
        g = _ak.exp(mu + var / 2) - 1
        lin = None
        for x in xs:
            t = g * x * _ak.diff(u, x).value
            lin = t if lin is None else lin + t
    else:
        lin = None
        for x in xs:
            t = mu * _ak.diff(u, x).value
            lin = t if lin is None else lin + t
    return lam * (eu - u - lin)


def jump_operator_taylor(
    u: Any, coords: list[Any], spatial: list[int], jump: GaussianJump, pool: Any
) -> Any | None:
    """The Taylor-moment replacement applied to a closed form (additive
    jumps): ``λ(u(x+μ) + ½σ²Δu(x+μ) − u − μ Σ ∂ᵢu)``."""
    if jump.multiplicative:
        return None
    lam, mu, var = jump.exprs(pool)
    xs = [coords[i] for i in spatial]
    lap = None
    lin = None
    for x in xs:
        d1 = _ak.diff(u, x).value
        t = _ak.diff(d1, x).value
        lap = t if lap is None else lap + t
        lin = mu * d1 if lin is None else lin + mu * d1
    shifted = {x: x + mu for x in xs}
    us = _ak.subs(u, shifted)
    ls = _ak.subs(lap, shifted)
    return lam * (us + var / 2 * ls - u - lin)


def pide_residual_concrete(u: Any, problem: Any, src: dict, *, exact_jump: bool = True) -> Any:
    """``u_t + b·∇u + ½Tr(σσᵀ∇²u) + A u (+ extra) − f`` at the fitted ``d``,
    with the jump term ``A u`` exact (or its Taylor replacement)."""
    spec = problem.spec
    desc = spec["pide"]
    pool = problem.jet.pool
    coords = problem.x
    time = desc["time"]
    spatial = [i for i in range(len(coords)) if i != time]
    r = _ak.diff(u, coords[time]).value
    drift = desc["drift"]
    if isinstance(drift, RadialDrift):
        r2 = None
        for i in spatial:
            r2 = coords[i] ** 2 if r2 is None else r2 + coords[i] ** 2
        beta = drift.coefficient(r2 ** pool.rational(1, 2), pool)
        for i in spatial:
            r = r + beta * coords[i] * _ak.diff(u, coords[i]).value
    elif drift is not None:
        for i, b in zip(spatial, drift):
            r = r + b * _ak.diff(u, coords[i]).value
    a = desc["a"]
    if a is not None:
        for p, i in enumerate(spatial):
            di = _ak.diff(u, coords[i]).value
            for q, j in enumerate(spatial):
                if _ak.is_zero(a[p][q]) is True:
                    continue
                r = r + a[p][q] * _ak.diff(di, coords[j]).value
    jump = desc["jump"]
    if jump is not None:
        if exact_jump:
            r = r + jump_operator_exact(u, coords, spatial, jump, pool)
        else:
            t = jump_operator_taylor(u, coords, spatial, jump, pool)
            if t is None:
                raise NoClosedForm("no Taylor form for a multiplicative jump")
            r = r + t
    extra = desc.get("extra")
    if extra is not None:
        raise NoClosedForm("an extra operator term is not verified exactly")
    if src.get("rhs") is not None:
        r = r - src["rhs"]
    return r


# ---------------------------------------------------------------------------
# Symbolic d
# ---------------------------------------------------------------------------


def _sum_nodes(e: Any) -> list[Any]:
    out, stack, seen = [], [e], set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        nd = n.node()
        if nd[0] == "func" and nd[1] == "Sum":
            out.append(n)
            continue
        stack.extend(_kids(n))
    return out


def _affine_in(e: Any, syms: list[Any]) -> bool:
    for s in syms:
        d1 = _ak.simplify(_ak.diff(e, s).value).value
        if not _free(d1, set(syms)):
            return False
    return True


def jump_operator_symbolic(u: Any, fs: Any, jump: GaussianJump) -> Any:
    """``λ(E[u(x + z)] − u − μ Σⱼ ∂ⱼu)`` for symbolic ``d``, for ``u`` affine in
    its sums ``Σᵢ f(xᵢ)`` (so the expectation passes inside each sum, where it
    is one-dimensional)."""
    from .experimental import Indexed, Sum

    pool = fs.pool
    if jump.multiplicative:
        raise NoClosedForm("multiplicative jumps are verified at the fitted d only")
    lam, mu, var = jump.exprs(pool)
    sums = _sum_nodes(u)
    stand = [pool.symbol(f"__fex_S{k}") for k in range(len(sums))]
    from ._fex_finalize import _rebuild

    mp = dict(zip(sums, stand))

    def f(node: Any, _kids_: list) -> Any:
        return mp.get(node)

    g = _rebuild(u, f)
    if not _affine_in(g, stand):
        raise NoClosedForm("the candidate is not affine in its sums over the coordinates")
    eu = g
    rep = {}
    for s_node, s_sym in zip(sums, stand):
        args = s_node.node()[2]
        body, k, lo, hi = args[0], args[1], args[2], args[3]
        xk = Indexed(fs.x, k)
        w = pool.symbol("__fex_w1")
        body1 = _ak.subs(body, {xk: w})
        eb = gaussian_expectation(body1, [w], mu, var)
        eb = _ak.subs(eb, {w: xk})
        rep[s_sym] = Sum(eb, k, lo, hi)
    eu = _ak.subs(g, rep)
    from ._fex_finalize import _sum_partials_indexed

    return lam * (eu - u - mu * _sum_partials_indexed(u, fs))


def _first_coordinate_second_partial(u: Any, fs: Any) -> Any:
    """``∂²u/∂x₁²`` for ``u`` affine in its sums ``Σᵢ fₖ(xᵢ)``:
    ``Σₖ (∂u/∂Sₖ)·fₖ''(x₁)``; refuses any other ``u``."""
    from ._fex_finalize import _rebuild
    from .experimental import Indexed

    pool = fs.pool
    sums = _sum_nodes(u)
    stand = [pool.symbol(f"__fex_S{k}") for k in range(len(sums))]
    mp = dict(zip(sums, stand))
    g = _rebuild(u, lambda node, _k: mp.get(node))
    if not _affine_in(g, stand):
        raise NoClosedForm("a bidiagonal diffusion lifts only for u affine in its sums")
    x1 = Indexed(fs.x, pool.integer(1))
    out = pool.integer(0)
    for s_node, s_sym in zip(sums, stand):
        args = s_node.node()[2]
        body, k = args[0], args[1]
        w = pool.symbol("__fex_w1")
        f = _ak.subs(body, {Indexed(fs.x, k): w})
        f2 = _ak.diff(_ak.diff(f, w).value, w).value
        out = out + _ak.diff(g, s_sym).value * _ak.subs(f2, {w: x1})
    return out


def pide_operator_symbolic(u: Any, fs: Any, desc: dict, src: dict) -> Any:
    """The PIDE residual for symbolic ``d``."""
    from .experimental import Indexed, Sum, laplacian_indexed, simplify_sums

    pool = fs.pool
    if fs.t is None:
        raise NoClosedForm("a PIDE needs its time coordinate")
    r = _ak.diff(u, fs.t).value
    drift = desc["drift"]
    if drift is not None:
        if not isinstance(drift, RadialDrift):
            raise NoClosedForm("only a radial drift β(|x|)·x lifts to symbolic d")
        j = pool.symbol("j", "integer")
        xj = Indexed(fs.x, j)
        R = Sum(fs.xi() ** 2, fs.i, 1, fs.d) ** pool.rational(1, 2)
        beta = drift.coefficient(R, pool)
        if _ak.is_zero(beta) is not True:
            one = pool.integer(1)
            d1 = simplify_sums(_ak.diff(u, xj).value, [(j, one, fs.d)])
            r = r + beta * simplify_sums(Sum(xj * d1, j, 1, fs.d))
    diff_ = desc["diffusion"]
    if diff_ is not None:
        if isinstance(diff_, BidiagonalDiffusion):
            # σσᵀ has (1, 1 + c², …, 1 + c²) on its diagonal and c next to
            # it. For u affine in its sums the Hessian is diagonal, so
            # ½Tr(σσᵀ∇²u) = ½θ²((1 + c²)Δu − c²·∂₁²u).
            th = to_expr(diff_.theta, pool)
            c = to_expr(diff_.sub, pool)
            d11 = _first_coordinate_second_partial(u, fs)
            lap = laplacian_indexed(u, fs.x, 1, fs.d)
            r = r + th**2 / 2 * ((1 + c**2) * lap - c**2 * d11)
        elif exact_number(diff_) is not None or isinstance(diff_, _ak.Expr):
            th = to_expr(diff_, pool)
            r = r + th**2 / 2 * laplacian_indexed(u, fs.x, 1, fs.d)
        else:
            raise NoClosedForm("a general diffusion matrix does not lift to symbolic d")
    jump = desc["jump"]
    if jump is not None:
        r = r + jump_operator_symbolic(u, fs, jump)
    if desc.get("extra") is not None:
        raise NoClosedForm("an extra operator term does not lift to symbolic d")
    if src.get("rhs") is not None:
        r = r - src["rhs"]
    return r
