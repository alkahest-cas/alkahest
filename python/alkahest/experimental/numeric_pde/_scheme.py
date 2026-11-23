"""Linear constant-coefficient difference schemes: von Neumann stability,
modified equations and consistency order.

A scheme on a grid ``u_j^n ≈ u(j Δx, n Δt)`` is

    Σ_{l, k} a_{l,k} u_{j+k}^{n+l} = 0,

with time levels ``l = 0 … L`` (``L`` is the newest level) and space offsets
``k``. Coefficients are exact (Laurent) polynomials in symbols such as the
CFL number ``c``, the diffusion number ``r`` or ``Δt``, ``Δx``, ``ν``.

Von Neumann analysis
--------------------
Substituting ``u_j^n = g^n e^{ijθ}`` gives the *amplification polynomial*
``Φ(g, θ) = Σ_l (Σ_k a_{l,k} e^{ikθ}) g^l``. The scheme satisfies the von
Neumann condition when, for every θ, ``Φ(·, θ)`` keeps its degree and is a
*simple von Neumann polynomial*: all roots in the closed unit disc and those
on the unit circle simple (Strikwerda, *Finite Difference Schemes and PDEs*,
§4.3; Miller, *J. Inst. Math. Appl.* 8 (1971)). For a two-level scheme this
is ``|G(θ)| ≤ 1`` with ``G = −(Σ_k a_{0,k} e^{ikθ}) / (Σ_k a_{1,k} e^{ikθ})``.

Miller's recursion turns that into a Boolean combination of polynomial sign
conditions in ``(cos θ, sin θ)`` and the parameters; the half-angle
substitution ``t = tan(θ/2)`` makes them polynomial in ``t`` (θ = π is
checked separately). The universally quantified sentence ``∀θ`` is then
decided **exactly**:

* no free parameter — by exact real-root isolation in ``t``;
* one free parameter (or several that enter only through one monomial, such
  as ``ν Δt / Δx²``) — by a one-parameter cylindrical algebraic
  decomposition: the parameter line is cut at the real roots of the leading
  coefficients and principal subresultant coefficients of all atoms (the
  CAD projection), and the sentence is decided exactly at a rational sample
  in every open cell and at every rational cell boundary.

Nothing is decided by sampling floating-point values. Irrational cell
boundaries whose membership cannot be decided exactly are reported in
``undecided_points``, and the result is then ``proof_status="unknown"``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any
from math import factorial

from ...exceptions import NumericPdeError
from ._exact import (
    AlgElem,
    AlgField,
    KPoly,
    MPoly,
    UPoly,
    _num_of,
    bderiv,
    expr_free_symbols,
    expr_to_mpoly,
    frac_to_expr,
    isolate_real_roots,
    k_isolate,
    k_sign_at,
    mpoly_to_expr,
    psc,
    rational_between,
    sign_at,
    upoly_to_expr,
)

__all__ = [
    "ConsistencyResult",
    "ModifiedEquation",
    "Scheme",
    "VonNeumannResult",
    "consistency",
    "modified_equation",
    "von_neumann",
]

_T = "\x00t"  # internal name of the half-angle variable (cannot clash)


def _pool_of(*objs):
    import alkahest

    for o in objs:
        if hasattr(o, "_pool"):
            return o._pool
    p = alkahest.active_pool()
    return p if p is not None else alkahest.default_pool()


def _cancel(e):
    import alkahest

    return alkahest.cancel(e)


def _to_expr(pool, v):
    if hasattr(v, "node"):
        return v
    if isinstance(v, bool):
        raise NumericPdeError(f"invalid coefficient {v!r}", code="E-PDE-351")
    if isinstance(v, float):
        raise NumericPdeError(
            f"floating-point coefficient {v!r}: pass an exact rational", code="E-PDE-351"
        )
    if isinstance(v, int | Fraction):
        return frac_to_expr(pool, v)
    if isinstance(v, str):
        return frac_to_expr(pool, Fraction(v))
    raise NumericPdeError(f"invalid coefficient {v!r}", code="E-PDE-351")


# ---------------------------------------------------------------------------
# Rational functions as (numerator, denominator) MPoly pairs (no gcd)
# ---------------------------------------------------------------------------


def _expr_to_ratfunc(expr, vars):
    """(N, D) with expr = N / D, N and D Laurent polynomials in ``vars``."""
    try:
        return expr_to_mpoly(expr, vars), MPoly.const(vars, 1)
    except NumericPdeError:
        pass
    vars = tuple(vars)

    def walk(e):
        n = e.node()
        tag = n[0]
        if tag in ("integer", "rational", "symbol", "float"):
            return expr_to_mpoly(e, vars), MPoly.const(vars, 1)
        if tag == "add":
            N, D = MPoly(vars), MPoly.const(vars, 1)
            for c in n[1]:
                n2, d2 = walk(c)
                N, D = N * d2 + n2 * D, D * d2
            return N, D
        if tag == "mul":
            N, D = MPoly.const(vars, 1), MPoly.const(vars, 1)
            for c in n[1]:
                n2, d2 = walk(c)
                N, D = N * n2, D * d2
            return N, D
        if tag == "pow":
            ex = _num_of(n[2].node())
            if ex is None or ex.denominator != 1:
                raise NumericPdeError(
                    f"non-integer power in {e}: coefficients must be rational functions",
                    code="E-PDE-351",
                )
            k = int(ex)
            N, D = walk(n[1])
            if k < 0:
                N, D, k = D, N, -k
            return N**k, D**k
        raise NumericPdeError(f"{e} is not a rational function of {vars}", code="E-PDE-351")

    return walk(expr)


# ---------------------------------------------------------------------------
# Scheme
# ---------------------------------------------------------------------------


def _contains_func(e, name):
    n = e.node()
    if n[0] == "func":
        return n[1] == name or any(hasattr(c, "node") and _contains_func(c, name) for c in n[2])
    if n[0] in ("add", "mul"):
        return any(_contains_func(c, name) for c in n[1])
    if n[0] == "pow":
        return _contains_func(n[1], name) or _contains_func(n[2], name)
    return False


class Scheme:
    """A linear constant-coefficient finite-difference scheme.

    Parameters
    ----------
    coeffs : dict
        ``{(l, k): a_lk}`` — the coefficient of ``u_{j+k}^{n+l}``; ``k`` may
        be a tuple for several space dimensions. Coefficients are ints,
        :class:`~fractions.Fraction` or exact ``Expr``.
    dt, dx : Expr, optional
        The time step and grid spacing symbols, when the coefficients use
        them (needed to state a modified equation in physical form).

    Use :meth:`from_expr` to read a scheme written with applied grid
    functions ``u(n + l, j + k)``.
    """

    def __init__(self, coeffs, dt=None, dx=None, pool=None):
        if not coeffs:
            raise NumericPdeError("empty scheme", code="E-PDE-353")
        pool = pool or _pool_of(dt, dx, *list(coeffs.values()))
        norm = {}
        dims: Any = None
        for key, v in coeffs.items():
            lv, k = key
            k = tuple(k) if isinstance(k, tuple | list) else (k,)
            if dims is None:
                dims = len(k)
            elif len(k) != dims:
                raise NumericPdeError("inconsistent offset dimensions", code="E-PDE-352")
            if not isinstance(lv, int) or not all(isinstance(x, int) for x in k):
                raise NumericPdeError(
                    f"time level and space offsets must be integers, got {key!r}",
                    code="E-PDE-352",
                )
            e = _to_expr(pool, v)
            if (lv, k) in norm:
                e = norm[(lv, k)] + e
            norm[(lv, k)] = e
        lmin = min(lv for lv, _ in norm)
        self.coeffs = {(lv - lmin, k): e for (lv, k), e in norm.items()}
        self.dims = dims
        self.levels = max(lv for lv, _ in self.coeffs) + 1
        self.dt = dt
        self.dx = dx
        self._pool = pool

    @classmethod
    def from_expr(cls, expr, u, n, j, dt=None, dx=None):
        """Read ``Σ a · u(n + l, j + k) = 0`` from an expression.

        ``expr`` is an ``Expr`` meaning ``expr = 0`` (or a pair
        ``(lhs, rhs)``), ``u`` the grid-function name, ``n`` the time index
        symbol and ``j`` the space index symbol (or a list of them).
        """
        import alkahest

        if isinstance(expr, tuple):
            expr = expr[0] - expr[1]
        pool = expr._pool
        js = list(j) if isinstance(j, list | tuple) else [j]
        ex = alkahest.expand(expr)
        node = ex.node()
        terms = node[1] if node[0] == "add" else [ex]
        coeffs = {}

        def shift(arg, base):
            q = _num_of(_cancel(arg - base).node())
            if q is None or q.denominator != 1:
                raise NumericPdeError(
                    f"grid index {arg} is not {base} plus an integer", code="E-PDE-352"
                )
            return int(q)

        for t in terms:
            tn = t.node()
            factors = tn[1] if tn[0] == "mul" else [t]
            grid = [f for f in factors if f.node()[0] == "func" and f.node()[1] == u]
            if len(grid) != 1:
                raise NumericPdeError(
                    f"term {t} is not linear in the grid function {u!r}", code="E-PDE-352"
                )
            g = grid[0]
            gi = next(
                i for i, f in enumerate(factors) if f.node()[0] == "func" and f.node()[1] == u
            )
            rest = factors[:gi] + factors[gi + 1 :]
            if any(_contains_func(f, u) for f in rest):
                raise NumericPdeError(f"term {t} is not linear in {u!r}", code="E-PDE-352")
            args = list(g.node()[2])
            if len(args) != 1 + len(js):
                raise NumericPdeError(f"{g} should have {1 + len(js)} indices", code="E-PDE-352")
            lv = shift(args[0], n)
            k = tuple(shift(a, b) for a, b in zip(args[1:], js))
            c = pool.mul(rest) if len(rest) > 1 else (rest[0] if rest else pool.integer(1))
            key = (lv, k if len(js) > 1 else k[0])
            coeffs[key] = coeffs[key] + c if key in coeffs else c
        return cls(coeffs, dt=dt, dx=dx, pool=pool)

    def __repr__(self):
        return f"Scheme(levels={self.levels}, dims={self.dims}, points={len(self.coeffs)})"

    def symbols(self):
        out = {}
        for e in self.coeffs.values():
            out.update(expr_free_symbols(e))
        return out

    def substitute(self, subs):
        import alkahest

        if not subs:
            return self
        return Scheme(
            {
                (lv, k if self.dims > 1 else k[0]): alkahest.subs(e, subs)
                for (lv, k), e in self.coeffs.items()
            },
            dt=self.dt,
            dx=self.dx,
            pool=self._pool,
        )


# ---------------------------------------------------------------------------
# Normalising coefficients to polynomials in (ideally) one parameter
# ---------------------------------------------------------------------------


def _normalised(scheme):
    """Polynomial coefficients {(l,k): MPoly} with the common content removed,
    the variable names, a name → Expr map, and the monomial substitution
    (``None`` or ``(new_name, expr)``) used to merge parameters."""
    syms = scheme.symbols()
    names = tuple(sorted(syms))
    rf = {key: _expr_to_ratfunc(e, names) for key, e in scheme.coeffs.items()}
    dens = [d for (_, d) in rf.values() if not d.is_monomial()]
    polys = {}
    for key, (N, D) in rf.items():
        p = N
        if D.is_monomial():
            p = p.div_monomial(D)
        for d2 in dens:
            if d2 is not D:
                p = p * d2
        polys[key] = p
    nz = [p for p in polys.values() if not p.is_zero()]
    if not nz:
        raise NumericPdeError("all scheme coefficients vanish", code="E-PDE-356")
    # remove the monomial content (min exponent of every variable)
    mins = [min(e[i] for p in nz for e in p.terms) for i in range(len(names))]
    content = MPoly(names, {tuple(mins): 1})
    polys = {k: p.div_monomial(content) for k, p in polys.items()}
    # drop the rational content too (makes coefficients small)
    allc = [c for p in polys.values() for c in p.terms.values()]
    from math import gcd as _g

    num = 0
    den = 1
    for c in allc:
        num = _g(num, c.numerator)
        den = den * c.denominator // _g(den, c.denominator)
    scale = Fraction(den, num) if num else Fraction(1)
    polys = {k: p * scale for k, p in polys.items()}
    symmap = dict(syms)
    used = set()
    for p in polys.values():
        used |= p.free_vars()
    merged = None
    if len(used) > 1:
        merged = _merge_monomial(polys, names, used)
        if merged is not None:
            polys, names, newname, mono = merged
            pool = scheme._pool
            symmap[newname] = None  # filled below
            mono_expr = mpoly_to_expr(pool, mono, syms)
            merged = (newname, mono_expr)
    return polys, names, symmap, merged


def _merge_monomial(polys, names, used):
    """If all exponent vectors differ from one another by integer multiples of
    a single primitive vector ``v`` (the coefficients are, up to a common
    monomial factor, polynomials in ``μ = x^v``), rewrite them in ``μ``."""
    from math import gcd as _g

    vecs = sorted({e for p in polys.values() for e in p.terms})
    if len(vecs) < 2:
        return None
    e0 = vecs[0]
    diffs = {e: tuple(a - b for a, b in zip(e, e0)) for e in vecs}
    base = next(d for d in diffs.values() if any(d))
    g = 0
    for x in base:
        g = _g(g, abs(x))
    v = tuple(x // g for x in base)
    for x in v:
        if x:
            if x < 0:
                v = tuple(-y for y in v)
            break
    piv = next(i for i, x in enumerate(v) if x)
    mult = {}
    for e, d in diffs.items():
        if d[piv] % v[piv]:
            return None
        m = d[piv] // v[piv]
        if tuple(m * x for x in v) != d:
            return None
        mult[e] = m
    mn = min(mult.values())
    newname = "\x00mu"
    nn = (newname,)
    out = {}
    for key, p in polys.items():
        t = {}
        for e, c in p.terms.items():
            m = mult[e] - mn
            t[(m,)] = t.get((m,), 0) + c
        out[key] = MPoly(nn, t)
    mono = MPoly(names, {v: 1})
    return out, nn, newname, mono


# ---------------------------------------------------------------------------
# Complex polynomials and Miller's recursion
# ---------------------------------------------------------------------------


class _C:
    __slots__ = ("im", "re")

    def __init__(self, re, im):
        self.re, self.im = re, im

    def __add__(self, o):
        return _C(self.re + o.re, self.im + o.im)

    def __sub__(self, o):
        return _C(self.re - o.re, self.im - o.im)

    def __mul__(self, o):
        if isinstance(o, _C):
            return _C(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
        return _C(self.re * o, self.im * o)

    def conj(self):
        return _C(self.re, -self.im)

    def abs2(self):
        return self.re * self.re + self.im * self.im

    def is_zero(self):
        return self.re.is_zero() and self.im.is_zero()


TRUE = ("const", True)
FALSE = ("const", False)


def _atom(p, rel):
    if p.is_const():
        v = p.const_value()
        ok = {">": v > 0, ">=": v >= 0, "==": v == 0}[rel]
        return TRUE if ok else FALSE
    return ("atom", p, rel)


def _and(*fs):
    out = []
    for f in fs:
        if f == FALSE:
            return FALSE
        if f == TRUE:
            continue
        if f[0] == "and":
            out.extend(f[1])
        else:
            out.append(f)
    if not out:
        return TRUE
    return out[0] if len(out) == 1 else ("and", out)


def _or(*fs):
    out = []
    for f in fs:
        if f == TRUE:
            return TRUE
        if f == FALSE:
            continue
        if f[0] == "or":
            out.extend(f[1])
        else:
            out.append(f)
    # (p > 0) ∨ (p == 0)  →  p ≥ 0
    atoms = [f for f in out if f[0] == "atom"]
    for a in atoms:
        for b in atoms:
            if a is not b and a[1] == b[1] and {a[2], b[2]} == {">", "=="}:
                out = [f for f in out if f is not a and f is not b] + [("atom", a[1], ">=")]
                return _or(*out)
    if not out:
        return FALSE
    return out[0] if len(out) == 1 else ("or", out)


def _phi1(phi):
    d = len(phi) - 1
    cd, c0 = phi[d], phi[0]
    return [cd.conj() * phi[k + 1] - c0 * phi[d - k - 1].conj() for k in range(d)]


def _schur(phi):
    d = len(phi) - 1
    if d == 0:
        return TRUE
    a, b = phi[d].abs2(), phi[0].abs2()
    return _and(_atom(a - b, ">"), _schur(_phi1(phi)))


def _simple_vn(phi):
    d = len(phi) - 1
    if d == 0:
        return TRUE
    a, b = phi[d].abs2(), phi[0].abs2()
    p1 = _phi1(phi)
    branch_a = _and(_atom(a - b, ">"), _simple_vn(p1))
    zero = _and(*[_atom(c.re, "==") for c in p1], *[_atom(c.im, "==") for c in p1])
    branch_b = zero if zero == FALSE else _and(zero, _schur([phi[k] * k for k in range(1, d + 1)]))
    return _or(branch_a, branch_b)


def _vn_poly(phi):
    """Miller's test for a von Neumann polynomial (all roots in the closed
    unit disc, multiplicities allowed)."""
    d = len(phi) - 1
    if d == 0:
        return TRUE
    a, b = phi[d].abs2(), phi[0].abs2()
    p1 = _phi1(phi)
    branch_a = _and(_atom(a - b, ">"), _vn_poly(p1))
    zero = _and(*[_atom(c.re, "==") for c in p1], *[_atom(c.im, "==") for c in p1])
    branch_b = (
        zero if zero == FALSE else _and(zero, _vn_poly([phi[k] * k for k in range(1, d + 1)]))
    )
    return _or(branch_a, branch_b)


def _von_neumann_formula(phi, second_order=False):
    d = len(phi) - 1
    body = _vn_poly(phi) if second_order else _simple_vn(phi)
    return _and(_atom(phi[d].abs2(), ">"), body)


def _atoms(f, acc):
    if f[0] == "atom":
        acc.append(f[1])
    elif f[0] in ("and", "or"):
        for g in f[1]:
            _atoms(g, acc)
    return acc


def _eval(f, sign):
    tag = f[0]
    if tag == "const":
        return f[1]
    if tag == "atom":
        s = sign(f[1])
        return {">": s > 0, ">=": s >= 0, "==": s == 0}[f[2]]
    if tag == "and":
        return all(_eval(g, sign) for g in f[1])
    return any(_eval(g, sign) for g in f[1])


# ---------------------------------------------------------------------------
# Exact decision of ∀t formulas
# ---------------------------------------------------------------------------


def _lcm_sqf(acc, q):
    if q.deg() <= 0:
        return acc
    q = q.squarefree()
    if acc is None:
        return q
    g = acc.gcd(q)
    return (acc * q.exquo(g)).monic()


def _forall_t(formula_t, formula_pi, subst):
    """Decide ``(∀t ∈ ℝ: formula_t) ∧ formula_pi`` after substituting the
    parameter values in ``subst`` (a dict name → Fraction)."""

    def specialise(p):
        for k, v in subst.items():
            p = p.subs_value(k, v)
        return p

    cache = {}

    def up(p):
        key = id(p)
        if key not in cache:
            cache[key] = (p, specialise(p).to_upoly(_T))
        return cache[key][1]

    # θ = π
    def sign_pi(p):
        v = specialise(p).const_value()
        return (v > 0) - (v < 0)

    if not _eval(formula_pi, sign_pi):
        return False
    atoms = _atoms(formula_t, [])
    polys = [up(p) for p in atoms]
    S = None
    for q in polys:
        if not q.is_zero():
            S = _lcm_sqf(S, q)
    roots = isolate_real_roots(S) if S is not None else []
    samples = []
    if roots:
        lo = roots[0].value if roots[0].is_rational else roots[0].lo
        hi = roots[-1].value if roots[-1].is_rational else roots[-1].hi
        samples.append(lo - 1)
        for a, b in zip(roots, roots[1:]):
            samples.append(rational_between(a, b))
        samples.append(hi + 1)
    else:
        samples.append(Fraction(0))
    for s in samples:
        if not _eval(formula_t, lambda p, _s=s: _sgn(up(p)(_s))):
            return False
    return all(_eval(formula_t, lambda p, _r=rt: sign_at(up(p), _r)) for rt in roots)


def _forall_t_alg(formula_t, formula_pi, pname, alpha):
    """As :func:`_forall_t`, at the irrational parameter value ``alpha``,
    with exact arithmetic in Q(alpha)."""
    F = AlgField(alpha)

    def elem(up):
        return AlgElem(F, up)

    def sign_pi(p):
        return elem(p.to_upoly(pname)).sign()

    if not _eval(formula_pi, sign_pi):
        return False
    cache = {}

    def kp(p):
        key = id(p)
        if key not in cache:
            cache[key] = (p, KPoly(F, [elem(u) for u in p.to_bivariate(_T, pname)]))
        return cache[key][1]

    atoms = _atoms(formula_t, [])
    S = None
    for p in atoms:
        q = kp(p)
        if q.deg() <= 0:
            continue
        q = q.squarefree()
        if S is None:
            S = q
        else:
            g = S.gcd(q)
            S = (S * (q.divmod(g)[0] if g.deg() > 0 else q)).monic()
    roots = k_isolate(S) if S is not None else []
    samples = []
    if roots:
        lo = roots[0].value if roots[0].is_rational else roots[0].lo
        hi = roots[-1].value if roots[-1].is_rational else roots[-1].hi
        samples.append(lo - 1)
        for a, b in zip(roots, roots[1:]):
            samples.append(rational_between(a, b))
        samples.append(hi + 1)
    else:
        samples.append(Fraction(0))
    for s in samples:
        if not _eval(formula_t, lambda p, _s=s: kp(p)(_s).sign()):
            return False
    return all(_eval(formula_t, lambda p, _r=rt: k_sign_at(kp(p), _r)) for rt in roots)


def _sgn(x):
    return (x > 0) - (x < 0)


class _Cell:
    __slots__ = ("hi", "kind", "lo", "point", "stable")
    hi: Any
    kind: Any
    lo: Any
    point: Any
    stable: Any

    def __init__(self, kind, lo=None, hi=None, point=None, stable=None):
        self.kind, self.lo, self.hi, self.point, self.stable = kind, lo, hi, point, stable


def _decide_one_parameter(formula_t, formula_pi, pname):
    atoms_t = _atoms(formula_t, [])
    atoms_pi = _atoms(formula_pi, [])
    crit = None
    biv = []
    seen = set()
    for p in atoms_t:
        if p in seen:
            continue
        seen.add(p)
        biv.append(p.to_bivariate(_T, pname))
    for B in biv:
        crit = _lcm_sqf(crit, B[-1])
        if len(B) > 1:
            for q in psc(B, bderiv(B)):
                if not q.is_zero():
                    crit = _lcm_sqf(crit, q)
    for i in range(len(biv)):
        for j in range(i + 1, len(biv)):
            if len(biv[i]) > 1 and len(biv[j]) > 1:
                for q in psc(biv[i], biv[j]):
                    if not q.is_zero():
                        crit = _lcm_sqf(crit, q)
    for p in atoms_pi:
        crit = _lcm_sqf(crit, p.to_upoly(pname))
    roots = isolate_real_roots(crit) if crit is not None else []
    cells = []
    if not roots:
        cells.append(_Cell("open", None, None, Fraction(0)))
    else:
        first = roots[0].value if roots[0].is_rational else roots[0].lo
        cells.append(_Cell("open", None, roots[0], first - 1))
        for i, r in enumerate(roots):
            cells.append(_Cell("point", r, r, r))
            if i + 1 < len(roots):
                cells.append(_Cell("open", r, roots[i + 1], rational_between(r, roots[i + 1])))
        last = roots[-1].value if roots[-1].is_rational else roots[-1].hi
        cells.append(_Cell("open", roots[-1], None, last + 1))
    for c in cells:
        if c.kind == "open":
            c.stable = _forall_t(formula_t, formula_pi, {pname: c.point})
        elif c.point.is_rational:
            c.stable = _forall_t(formula_t, formula_pi, {pname: c.point.value})
        else:
            try:
                c.stable = _forall_t_alg(formula_t, formula_pi, pname, c.point)
            except (ZeroDivisionError, ArithmeticError, RecursionError):  # pragma: no cover
                c.stable = None  # undecided irrational boundary
    return cells, crit


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


class _DictLike:
    _keys: tuple = ()

    def __getitem__(self, k):
        if k not in self._keys:
            raise KeyError(k)
        return getattr(self, k)

    def keys(self):
        return list(self._keys)

    def get(self, k, default=None):
        return getattr(self, k, default) if k in self._keys else default

    def __contains__(self, k):
        return k in self._keys


class VonNeumannResult(_DictLike):
    """Result of :func:`von_neumann`.

    Attributes
    ----------
    G : Expr or None
        The amplification factor ``G(θ)`` (two-level schemes); ``None`` for
        multi-level schemes, where ``amplification_polynomial`` is the object.
    amplification_polynomial : Expr
        ``Φ(g, θ) = Σ_l (Σ_k a_{l,k} e^{ikθ}) g^l`` written with ``cos`` and
        ``sin``.
    abs_G_squared : Expr or None
        ``|G(θ)|²`` (two-level schemes), a rational function of ``cos kθ``.
    stability_condition : Expr
        The exact set of parameter values satisfying the von Neumann
        condition, as a predicate (``True``/``False`` without parameters).
        With ``proof_status == "unknown"`` it is the part proved stable — a
        sufficient condition.
    proof_status : str
        ``"proved"`` (decided exactly) or ``"unknown"`` (undecided boundary
        points, or more than one independent parameter).
    parameter : Expr or None
        The parameter the condition is stated in (a symbol, or a monomial such
        as ``ν Δt / Δx²`` when several symbols enter only through it).
    intervals : list
        Maximal stable pieces ``(lo, hi, lo_closed, hi_closed)`` with exact
        endpoints (``None`` for ±∞); points appear as ``(a, a, True, True)``.
    undecided_points : list
        Irrational boundary points whose membership was not decided.
    method : str
    """

    _keys = (
        "G",
        "amplification_polynomial",
        "abs_G_squared",
        "stability_condition",
        "proof_status",
        "parameter",
        "intervals",
        "undecided_points",
        "method",
        "theta",
    )
    G: Any
    amplification_polynomial: Any
    abs_G_squared: Any
    stability_condition: Any
    proof_status: Any
    parameter: Any
    intervals: Any
    undecided_points: Any
    method: Any
    theta: Any

    def __init__(self, **kw):
        for k in self._keys:
            setattr(self, k, kw.get(k))

    @property
    def stable(self):
        """``True``/``False`` for a parameter-free scheme decided exactly,
        else ``None``."""
        if self.parameter is None and self.proof_status == "proved":
            return bool(self.intervals)
        return None

    def __repr__(self):
        return (
            f"VonNeumannResult(stability_condition={self.stability_condition}, "
            f"proof_status={self.proof_status!r})"
        )


def _alg_to_expr(pool, a, crit, var):
    if a.is_rational:
        return frac_to_expr(pool, a.value)
    import alkahest

    # index of a among the real roots of its defining polynomial
    q = a.poly
    rts = isolate_real_roots(q)
    idx = None
    for i, b in enumerate(rts):
        if not b.is_rational and b.lo < a.hi and a.lo < b.hi:
            # disjoint isolating intervals of the same polynomial: overlap ⇒ same root
            idx = i
            break
    ints = q.primitive_int()
    poly = upoly_to_expr(pool, UPoly(ints), var)
    if idx is None:  # pragma: no cover
        return pool.float(float(a))
    return alkahest.root_of(poly, var, idx)


def _condition_expr(pool, cells, param_expr, crit, var_for_root):
    """Merge decided cells into maximal intervals and build a predicate."""
    pieces = []
    cur = None
    for c in cells:
        ok = c.stable is True
        if ok:
            if cur is None:
                if c.kind == "open":
                    cur = [c.lo, c.hi, False, False]
                else:
                    cur = [c.point, c.point, True, True]
            else:
                if c.kind == "open":
                    cur[1], cur[3] = c.hi, False
                else:
                    cur[1], cur[3] = c.point, True
        else:
            if cur is not None:
                pieces.append(tuple(cur))
                cur = None
    if cur is not None:
        pieces.append(tuple(cur))

    def ex(a):
        return _alg_to_expr(pool, a, crit, var_for_root)

    preds = []
    for lo, hi, lc, hc in pieces:
        if lo is not None and hi is not None and lo is hi:
            preds.append(pool.pred_eq(param_expr, ex(lo)))
            continue
        parts = []
        if lo is not None:
            parts.append(pool.ge(param_expr, ex(lo)) if lc else pool.gt(param_expr, ex(lo)))
        if hi is not None:
            parts.append(pool.le(param_expr, ex(hi)) if hc else pool.lt(param_expr, ex(hi)))
        if not parts:
            preds.append(pool.pred_true())
        elif len(parts) == 1:
            preds.append(parts[0])
        else:
            preds.append(pool.pred_and(parts))
    if not preds:
        cond = pool.pred_false()
    elif len(preds) == 1:
        cond = preds[0]
    else:
        cond = pool.pred_or(preds)

    def out_end(a):
        if a is None:
            return None
        return a.value if a.is_rational else ex(a)

    intervals = [(out_end(lo), out_end(hi), lc, hc) for lo, hi, lc, hc in pieces]
    return cond, intervals


def _exp_itheta(pool, k, theta):
    """``e^{ikθ}`` split as (real Expr, imag Expr)."""
    import alkahest

    if k == 0:
        return pool.integer(1), pool.integer(0)
    arg = theta if k == 1 else pool.integer(k) * theta
    return alkahest.cos(arg), alkahest.sin(arg)


def _symbol_expr(pool, row, theta, names, symmap):
    """``Σ_k a_k e^{ikθ}`` as an Expr (cos/sin form, conjugate pairs merged)."""
    import alkahest

    imag = pool.imaginary_unit()
    ks = sorted({abs(k) for k in row})
    terms = []
    for k in ks:
        ap = row.get(k)
        am = row.get(-k)
        ap = ap if ap is not None else None
        if k == 0:
            terms.append(mpoly_to_expr(pool, row[0], symmap))
            continue
        zero = MPoly(names)
        a = ap if ap is not None else zero
        b = am if am is not None else zero
        s, d = a + b, a - b
        arg = theta if k == 1 else pool.integer(k) * theta
        if not s.is_zero():
            terms.append(mpoly_to_expr(pool, s, symmap) * alkahest.cos(arg))
        if not d.is_zero():
            terms.append(imag * mpoly_to_expr(pool, d, symmap) * alkahest.sin(arg))
    if not terms:
        return pool.integer(0)
    return terms[0] if len(terms) == 1 else pool.add(terms)


def _abs2_expr(pool, row, theta, symmap):
    """``|Σ_k a_k e^{ikθ}|² = Σ_{k,m} a_k a_m cos((k−m)θ)`` as an Expr."""
    import alkahest

    acc = {}
    for k, a in row.items():
        for m, b in row.items():
            d = abs(k - m)
            acc[d] = acc[d] + a * b if d in acc else a * b
    terms = []
    for d in sorted(acc):
        c = acc[d]
        if c.is_zero():
            continue
        ce = mpoly_to_expr(pool, c, symmap)
        if d == 0:
            terms.append(ce)
        else:
            arg = theta if d == 1 else pool.integer(d) * theta
            terms.append(ce * alkahest.cos(arg))
    if not terms:
        return pool.integer(0)
    return terms[0] if len(terms) == 1 else pool.add(terms)


def _binom_poly(vars, re_sign, m):
    """``(1 + s i t)^m`` as a complex MPoly pair (s = ±1)."""
    one = MPoly.const(vars, 1)
    out = _C(one, MPoly(vars))
    base = _C(one, MPoly.var(vars, _T) * re_sign)
    for _ in range(m):
        out = out * base
    return out


def von_neumann(scheme, subs=None, theta=None, time_order=None):
    """Von Neumann stability analysis of a linear constant-coefficient scheme.

    Parameters
    ----------
    scheme : Scheme
    subs : dict, optional
        Substitutions applied to the coefficients first, e.g.
        ``{dt: r * dx**2 / nu}`` to state the result in the diffusion number.
    theta : Expr, optional
        The wave-number symbol used in the returned ``G`` (default ``theta``).
    time_order : {None, 1, 2}
        Order in time of the equation the scheme approximates. ``None``
        detects it: a scheme whose amplification polynomial has a double root
        ``g = 1`` at ``θ = 0`` (``Φ(1, 0) = ∂Φ/∂g(1, 0) = 0``, e.g. leapfrog
        for ``u_tt = c² u_xx``) is treated as second order in time, and then
        Strikwerda's criterion (Thm 8.2.1) applies: roots in the closed unit
        disc, at most double on the unit circle. It is decided for
        amplification polynomials of degree ≤ 2, where it coincides with the
        von Neumann-polynomial condition; higher degrees give ``"unknown"``.

    Returns
    -------
    VonNeumannResult
        Mapping-like: ``result["G"]``, ``result["stability_condition"]``,
        ``result["proof_status"]``.

    Raises
    ------
    NumericPdeError
        ``E-PDE-353`` for more than one space dimension, ``E-PDE-351`` for
        coefficients that are not rational functions, ``E-PDE-356`` when the
        newest time level vanishes identically.
    """
    if subs:
        scheme = scheme.substitute(subs)
    if scheme.dims != 1:
        raise NumericPdeError(
            "von_neumann supports one space dimension; for a d-dimensional scheme "
            "analyse each direction or substitute a specific wave vector",
            code="E-PDE-353",
        )
    pool = scheme._pool
    if theta is None:
        theta = pool.symbol("theta")
    polys, names, symmap, merged = _normalised(scheme)
    if merged is not None:
        newname, mono_expr = merged
        symmap = {newname: mono_expr}
    L = scheme.levels - 1
    rows = [{} for _ in range(L + 1)]
    for (lv, k), p in polys.items():
        if not p.is_zero():
            rows[lv][k[0]] = p
    if not rows[L]:
        raise NumericPdeError("the newest time level has no coefficients", code="E-PDE-356")
    if L == 0:
        raise NumericPdeError("a scheme needs at least two time levels", code="E-PDE-353")

    # --- expressions for the user ---------------------------------------------

    sym_rows = [_symbol_expr(pool, r, theta, names, symmap) if r else pool.integer(0) for r in rows]
    g = pool.symbol("g")
    phi_terms = [
        sym_rows[lv] * (g**lv if lv > 1 else (g if lv == 1 else pool.integer(1)))
        for lv in range(L + 1)
    ]
    phi_expr = pool.add(phi_terms) if len(phi_terms) > 1 else phi_terms[0]
    G = absG2 = None
    if L == 1:
        one_den = list(rows[1]) == [0] and rows[1][0].is_const() and rows[1][0].const_value() == 1
        if one_den:
            G = pool.integer(-1) * sym_rows[0]
            absG2 = _abs2_expr(pool, rows[0], theta, symmap)
        else:
            G = pool.integer(-1) * sym_rows[0] * sym_rows[1] ** -1
            absG2 = (
                _abs2_expr(pool, rows[0], theta, symmap)
                * _abs2_expr(pool, rows[1], theta, symmap) ** -1
            )

    # --- Miller conditions in t = tan(θ/2) -------------------------------------
    pnames = tuple(n for n in names)
    vars = (_T, *pnames)

    def lift_poly(p):
        return MPoly(vars, {(0, *e): c for e, c in p.terms.items()})

    K = max(abs(k) for r in rows for k in r)
    phi_t = []
    phi_pi = []
    for lv in range(L + 1):
        acc = _C(MPoly(vars), MPoly(vars))
        accpi = MPoly(vars)
        for k, p in rows[lv].items():
            lp = lift_poly(p)
            w = _binom_poly(vars, 1, K + k) * _binom_poly(vars, -1, K - k)
            acc = acc + w * lp
            accpi = accpi + lp * (1 if k % 2 == 0 else -1)
        phi_t.append(acc)
        phi_pi.append(_C(accpi, MPoly(vars)))
    tot = MPoly(names)
    dtot = MPoly(names)
    for lv, row in enumerate(rows):
        for p in row.values():
            tot = tot + p
            dtot = dtot + p * lv
    if time_order is None:
        time_order = 2 if (tot.is_zero() and dtot.is_zero()) else 1
    if time_order not in (1, 2):
        raise NumericPdeError("time_order must be 1 or 2", code="E-PDE-353")
    second = time_order == 2
    if second and L > 2:
        return VonNeumannResult(
            G=G,
            amplification_polynomial=phi_expr,
            abs_G_squared=absG2,
            stability_condition=None,
            proof_status="unknown",
            parameter=None,
            intervals=[],
            undecided_points=[],
            method="second order in time with more than three levels is not decided",
            theta=theta,
        )
    f_t = _von_neumann_formula(phi_t, second)
    f_pi = _von_neumann_formula(phi_pi, second)

    used = set()
    for p in _atoms(f_t, []) + _atoms(f_pi, []):
        used |= p.free_vars()
    used.discard(_T)
    method = (
        "miller-recursion"
        + (" (second order in time: roots in the closed disc, at most double)" if second else "")
        + " + half-angle substitution + exact real-root isolation"
    )
    if not used:
        ok = _forall_t(f_t, f_pi, {})
        return VonNeumannResult(
            G=G,
            amplification_polynomial=phi_expr,
            abs_G_squared=absG2,
            stability_condition=pool.pred_true() if ok else pool.pred_false(),
            proof_status="proved",
            parameter=None,
            intervals=[(None, None, False, False)] if ok else [],
            undecided_points=[],
            method=method,
            theta=theta,
        )
    if len(used) > 1:
        return VonNeumannResult(
            G=G,
            amplification_polynomial=phi_expr,
            abs_G_squared=absG2,
            stability_condition=None,
            proof_status="unknown",
            parameter=None,
            intervals=[],
            undecided_points=[],
            method=method + "; more than one independent parameter — fix all but one with subs=",
            theta=theta,
        )
    (pname,) = used
    cells, crit = _decide_one_parameter(f_t, f_pi, pname)
    param_expr = symmap[pname]
    var_for_root = param_expr if param_expr.node()[0] == "symbol" else pool.symbol("x")
    cond, intervals = _condition_expr(pool, cells, param_expr, crit, var_for_root)
    undecided = [c.point for c in cells if c.stable is None]
    und_exprs = [_alg_to_expr(pool, a, crit, var_for_root) for a in undecided]
    return VonNeumannResult(
        G=G,
        amplification_polynomial=phi_expr,
        abs_G_squared=absG2,
        stability_condition=cond,
        proof_status="proved" if not undecided else "unknown",
        parameter=param_expr,
        intervals=intervals,
        undecided_points=und_exprs,
        method=method + " + one-parameter CAD",
        theta=theta,
    )


# ---------------------------------------------------------------------------
# Modified equation (Warming–Hyett via the exact symbol)
# ---------------------------------------------------------------------------


class _QF:
    """Element N / d^k of Q(params) with a fixed denominator polynomial d."""

    __slots__ = ("d", "k", "n")

    def __init__(self, n, k, d):
        self.n, self.k, self.d = n, k, d

    def _align(self, o):
        if self.k == o.k:
            return self.n, o.n, self.k
        if self.k < o.k:
            return self.n * self.d ** (o.k - self.k), o.n, o.k
        return self.n, o.n * self.d ** (self.k - o.k), self.k

    def __add__(self, o):
        a, b, k = self._align(o)
        return _QF(a + b, k, self.d)

    def __sub__(self, o):
        a, b, k = self._align(o)
        return _QF(a - b, k, self.d)

    def __mul__(self, o):
        if isinstance(o, _QF):
            return _QF(self.n * o.n, self.k + o.k, self.d)
        return _QF(self.n * o, self.k, self.d)

    def div_d(self):
        return _QF(self.n, self.k + 1, self.d)

    def is_zero(self):
        return self.n.is_zero()


def _ser_mul(a, b, N, zero):
    out = [zero for _ in range(N + 1)]
    for i, x in enumerate(a):
        if x.is_zero():
            continue
        for j in range(N + 1 - i):
            y = b[j]
            if not y.is_zero():
                out[i + j] = out[i + j] + x * y
    return out


def _principal_log_series(scheme_rows, names, N):
    """γ_1..γ_N with log g(s) = Σ γ_q s^q, s = iθ, as _QF elements."""
    L = len(scheme_rows) - 1
    total = MPoly(names)
    dpoly = MPoly(names)
    for lv, row in enumerate(scheme_rows):
        for p in row.values():
            total = total + p
            dpoly = dpoly + p * lv
    if not total.is_zero():
        raise NumericPdeError(
            "the scheme does not preserve constant states (Σ a_{l,k} ≠ 0), so it is "
            "not consistent with any equation u_t = L u",
            code="E-PDE-354",
        )
    if dpoly.is_zero():
        raise NumericPdeError(
            "the principal amplification root is not simple at θ = 0 "
            "(∂Φ/∂g(1, 0) = 0); no modified equation",
            code="E-PDE-355",
        )
    zero = _QF(MPoly(names), 0, dpoly)
    one = _QF(MPoly.const(names, 1), 0, dpoly)
    # C_l(s) = Σ_k a_{l,k} e^{k s}
    C = []
    for row in scheme_rows:
        ser = []
        for n in range(N + 1):
            acc = MPoly(names)
            for k, p in row.items():
                if k or n == 0:
                    acc = acc + p * Fraction(k**n, factorial(n))
            ser.append(_QF(acc, 0, dpoly))
        C.append(ser)
    g = [one] + [zero] * N
    for _ in range(N + 1):
        # Φ(g, s) by Horner in g
        acc = list(C[L])
        for lv in range(L - 1, -1, -1):
            acc = _ser_mul(acc, g, N, zero)
            acc = [acc[i] + C[lv][i] for i in range(N + 1)]
        if all(x.is_zero() for x in acc):
            break
        g = [g[i] - acc[i].div_d() for i in range(N + 1)]
    w = [zero, *g[1:]]
    logs = [zero] * (N + 1)
    power = list(w)
    for m in range(1, N + 1):
        coef = Fraction((-1) ** (m + 1), m)
        logs = [logs[i] + power[i] * coef for i in range(N + 1)]
        power = _ser_mul(power, w, N, zero)
    return logs, dpoly


class ModifiedEquation(_DictLike):
    """Result of :func:`modified_equation`.

    ``u_t = Σ_q coefficients[q] · ∂_x^q u`` is the equation the scheme solves
    to higher order (Warming & Hyett, *J. Comput. Phys.* 14 (1974)).

    Attributes
    ----------
    coefficients : dict
        ``{q: μ_q}`` in physical form (``Δx^q / Δt`` scaling applied) when
        ``dt``/``dx`` are known, else ``None``.
    symbol_coefficients : dict
        ``{q: γ_q}`` with ``log g(θ) = Σ_q γ_q (iθ)^q`` — dimensionless.
    physical, numerical : dict
        The split of ``coefficients`` into the target equation's terms and
        the truncation terms.
    leading_dissipation, leading_dispersion : tuple or None
        ``(q, μ_q)`` of the first even (dissipative, ``q ≥ 2``) and first odd
        (dispersive, ``q ≥ 3``) numerical term.
    order : int
        Highest ``q`` computed.
    """

    _keys = (
        "coefficients",
        "symbol_coefficients",
        "physical",
        "numerical",
        "leading_dissipation",
        "leading_dispersion",
        "order",
    )
    coefficients: Any
    symbol_coefficients: Any
    physical: Any
    numerical: Any
    leading_dissipation: Any
    leading_dispersion: Any
    order: Any
    _pool: Any

    def __init__(self, **kw):
        for k in self._keys:
            setattr(self, k, kw.get(k))
        self._pool = kw.get("pool")

    def rhs(self, u=None, x=None, which="all"):
        """``Σ_q μ_q ∂_x^q u`` as an ``Expr`` (``which`` = all/physical/numerical)."""
        import alkahest

        from ._fd import _default_function, _nth_derivative

        src = {"all": self.coefficients, "physical": self.physical, "numerical": self.numerical}[
            which
        ]
        if src is None:
            raise NumericPdeError("physical coefficients need dt and dx", code="E-PDE-353")
        p = self._pool
        u, (x,) = _default_function(p, 1, u, None if x is None else [x])
        terms = [c * _nth_derivative(u, x, q) for q, c in sorted(src.items())]
        del alkahest
        if not terms:
            return p.integer(0)
        return terms[0] if len(terms) == 1 else p.add(terms)

    def __repr__(self):
        return f"ModifiedEquation(coefficients={self.coefficients})"


def _is_zero_expr(e):
    n = e.node()
    return n[0] == "integer" and n[1] == "0"


def modified_equation(scheme, order=4, dt=None, dx=None, subs=None, pde=None):
    """Modified equation of a two- or multi-level scheme in one space dimension.

    The coefficients come from the exact symbol: with ``s = iθ`` the
    principal root ``g(s)`` of ``Φ(g, s) = 0`` (``g(0) = 1``) is expanded as
    a power series, and ``log g = Σ γ_q s^q`` gives
    ``u_t = Σ_q γ_q (Δx^q/Δt) ∂_x^q u``. This equals the Warming–Hyett
    procedure without its error-prone elimination of time derivatives.

    Parameters
    ----------
    order : int
        Highest spatial derivative order ``q`` to compute (default 4).
    dt, dx : Expr, optional
        Time-step and spacing symbols (default: ``scheme.dt``/``scheme.dx``).
    subs : dict, optional
        Applied to the dimensionless coefficients before scaling, e.g.
        ``{c: a*dt/dx}`` turns a CFL-number scheme into physical form.
    pde : dict, optional
        ``{q: c_q}`` for the target equation ``u_t = Σ c_q ∂_x^q u``. If
        omitted, terms whose coefficient is free of ``dt`` and ``dx`` are
        taken as the physical ones.
    """
    import alkahest

    if scheme.dims != 1:
        raise NumericPdeError("modified_equation supports one space dimension", code="E-PDE-353")
    pool = scheme._pool
    dt = dt if dt is not None else scheme.dt
    dx = dx if dx is not None else scheme.dx
    syms = scheme.symbols()
    names = tuple(sorted(syms))
    rf = {key: _expr_to_ratfunc(e, names) for key, e in scheme.coeffs.items()}
    # clear denominators by the product of all denominators
    dens = [d for (_, d) in rf.values()]
    L = scheme.levels - 1
    rows = [{} for _ in range(L + 1)]
    for (lv, k), (N, D) in rf.items():
        p = N
        for d2 in dens:
            if d2 is not D:
                p = p * d2
        if not p.is_zero():
            rows[lv][k[0]] = rows[lv].get(k[0], MPoly(names)) + p
    logs, dpoly = _principal_log_series(rows, names, order)
    sym_coeffs = {}
    for q in range(1, order + 1):
        e = logs[q]
        if e.is_zero():
            continue
        num = mpoly_to_expr(pool, e.n, syms)
        if e.k:
            num = num * mpoly_to_expr(pool, dpoly, syms) ** (-e.k)
        val = _cancel(num)
        if not _is_zero_expr(val):
            sym_coeffs[q] = val
    coeffs = physical = numerical = None
    lead_diss = lead_disp = None
    if dt is not None and dx is not None:
        coeffs = {}
        for q, g in sym_coeffs.items():
            v = alkahest.subs(g, subs) if subs else g
            v = _cancel(v * dx**q * dt**-1)
            if not _is_zero_expr(v):
                coeffs[q] = v
        physical, numerical = {}, {}
        if pde is not None:
            target = {q: _to_expr(pool, c) for q, c in pde.items()}
            for q in sorted(set(coeffs) | set(target)):
                diff = _cancel(coeffs.get(q, pool.integer(0)) - target.get(q, pool.integer(0)))
                if q in target:
                    physical[q] = target[q]
                if not _is_zero_expr(diff):
                    numerical[q] = diff
        else:
            dname = {dt.node()[1], dx.node()[1]}
            for q, v in coeffs.items():
                if set(expr_free_symbols(v)) & dname:
                    numerical[q] = v
                else:
                    physical[q] = v
        for q in sorted(numerical):
            if q % 2 == 0 and q >= 2 and lead_diss is None:
                lead_diss = (q, numerical[q])
            if q % 2 == 1 and q >= 3 and lead_disp is None:
                lead_disp = (q, numerical[q])
    return ModifiedEquation(
        coefficients=coeffs,
        symbol_coefficients=sym_coeffs,
        physical=physical,
        numerical=numerical,
        leading_dissipation=lead_diss,
        leading_dispersion=lead_disp,
        order=order,
        pool=pool,
    )


# ---------------------------------------------------------------------------
# Consistency order (local truncation error)
# ---------------------------------------------------------------------------


class ConsistencyResult(_DictLike):
    """Result of :func:`consistency`.

    Attributes
    ----------
    consistent : bool or None
        ``True`` when every truncation term vanishes as ``Δt, Δx → 0``
        independently; ``False`` when a term survives; ``None`` when a term
        such as ``Δx²/Δt`` vanishes only on some refinement paths.
    time_order, space_order : int or None
        ``p`` and ``q`` in ``τ = O(Δt^p + Δx^q)`` (from the pure ``Δt^a`` and
        ``Δx^b`` terms; ``None`` when no such term appeared up to ``max_order``).
    mixed_terms : list
        Monomials ``Δt^a Δx^b`` (``a, b ≠ 0``) present in the error.
    truncation_error : Expr
        ``τ = Σ_m τ_m ∂_x^m u`` (terms up to ``max_order``), normalised so the
        scheme approximates ``u_t − L u``.
    terms : dict
        ``{m: τ_m}``.
    """

    _keys = (
        "consistent",
        "time_order",
        "space_order",
        "mixed_terms",
        "truncation_error",
        "terms",
    )
    consistent: Any
    time_order: Any
    space_order: Any
    mixed_terms: Any
    truncation_error: Any
    terms: Any

    def __init__(self, **kw):
        for k in self._keys:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return (
            f"ConsistencyResult(consistent={self.consistent}, time_order={self.time_order}, "
            f"space_order={self.space_order})"
        )


def consistency(scheme, pde, dt=None, dx=None, subs=None, max_order=8, u=None, x=None):
    """Local truncation error and order of accuracy of a scheme for
    ``u_t = Σ_q c_q ∂_x^q u``.

    The exact solution satisfies ``∂_t^p u = L^p u``, so the scheme applied to
    it has the formal symbol ``Σ_{l,k} a_{l,k} exp(l Δt L(D) + k Δx D)``; this
    is expanded in ``D = ∂_x`` and normalised by ``Δt Σ_{l,k} l a_{l,k}``.

    Parameters
    ----------
    pde : dict
        ``{q: c_q}`` (``Expr`` or numbers).
    dt, dx : Expr
        The step symbols (default ``scheme.dt``/``scheme.dx``); the scheme's
        coefficients must be written in them (use ``subs`` to rewrite CFL or
        diffusion numbers, e.g. ``{r: nu*dt/dx**2}``).
    max_order : int
        Highest power of ``D`` expanded.
    """
    from ._fd import _default_function, _nth_derivative

    if scheme.dims != 1:
        raise NumericPdeError("consistency supports one space dimension", code="E-PDE-353")
    if subs:
        scheme = scheme.substitute(subs)
    pool = scheme._pool
    dt = dt if dt is not None else scheme.dt
    dx = dx if dx is not None else scheme.dx
    if dt is None or dx is None:
        raise NumericPdeError("consistency needs the dt and dx symbols", code="E-PDE-353")
    pde = {q: _to_expr(pool, c) for q, c in pde.items()}
    syms = scheme.symbols()
    for c in pde.values():
        syms.update(expr_free_symbols(c))
    syms[dt.node()[1]] = dt
    syms[dx.node()[1]] = dx
    names = tuple(sorted(syms))
    tn, xn = dt.node()[1], dx.node()[1]
    a = {}
    for (lv, k), e in scheme.coeffs.items():
        a[(lv, k[0])] = expr_to_mpoly(e, names)
    Lop = {q: expr_to_mpoly(c, names) for q, c in pde.items()}
    DT = MPoly.var(names, tn)
    DX = MPoly.var(names, xn)
    M = max_order
    zero = MPoly(names)

    def smul(A, B):
        out = [zero] * (M + 1)
        for i, x in enumerate(A):
            if x.is_zero():
                continue
            for j in range(M + 1 - i):
                if not B[j].is_zero():
                    out[i + j] = out[i + j] + x * B[j]
        return out

    total = [zero] * (M + 1)
    for (lv, k), coef in a.items():
        E = [zero] * (M + 1)
        if lv:
            for q, c in Lop.items():
                if q <= M:
                    E[q] = E[q] + DT * c * lv
        if k and M >= 1:
            E[1] = E[1] + DX * k
        ex = [MPoly.const(names, 1)] + [zero] * M
        pw = [MPoly.const(names, 1)] + [zero] * M
        for n in range(1, M + 1):
            pw = smul(pw, E)
            if all(p.is_zero() for p in pw):
                break
            ex = [ex[i] + pw[i] * Fraction(1, factorial(n)) for i in range(M + 1)]
        total = [total[i] + ex[i] * coef for i in range(M + 1)]
    norm = zero
    for (lv, _k), coef in a.items():
        norm = norm + coef * lv
    norm = norm * DT
    if norm.is_zero():
        raise NumericPdeError(
            "the scheme has no time-derivative content (Σ l·a_{l,k} = 0)", code="E-PDE-354"
        )
    ti, xi = names.index(tn), names.index(xn)
    if norm.is_monomial():
        terms = [t.div_monomial(norm) for t in total]
        norm_expr = None
    else:
        # strip only the (dt, dx) monomial content; the rest is a nonzero factor
        mins = [0] * len(names)
        mins[ti] = min(e[ti] for e in norm.terms)
        mins[xi] = min(e[xi] for e in norm.terms)
        mono = MPoly(names, {tuple(mins): 1})
        terms = [t.div_monomial(mono) for t in total]
        norm_expr = mpoly_to_expr(pool, norm.div_monomial(mono), syms)
    consistent = True
    pure_t, pure_x, mixed = [], [], []
    for t in terms:
        for e in t.terms:
            ea, eb = e[ti], e[xi]
            if ea < 0 or eb < 0:
                if consistent is True:
                    consistent = None
            elif ea == 0 and eb == 0:
                consistent = False
            elif eb == 0:
                pure_t.append(ea)
            elif ea == 0:
                pure_x.append(eb)
            else:
                mixed.append((ea, eb))
    # the u_t - L u part itself must have cancelled: terms[m] for the target
    texpr = {}
    for m, t in enumerate(terms):
        if t.is_zero():
            continue
        v = mpoly_to_expr(pool, t, syms)
        if norm_expr is not None:
            v = _cancel(v * norm_expr**-1)
        texpr[m] = v
    uu, (xx,) = _default_function(pool, 1, u, None if x is None else [x])
    parts = [c * _nth_derivative(uu, xx, m) for m, c in sorted(texpr.items())]
    tau = pool.integer(0) if not parts else (parts[0] if len(parts) == 1 else pool.add(parts))
    return ConsistencyResult(
        consistent=consistent,
        time_order=min(pure_t) if pure_t else None,
        space_order=min(pure_x) if pure_x else None,
        mixed_terms=sorted(set(mixed)),
        truncation_error=tau,
        terms=texpr,
    )
