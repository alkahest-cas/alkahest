"""Finalising a FEX solution (milestone F3): exact constants, symbolic ``d``,
verification and certificates.

:func:`finalize` takes a fitted candidate (a :class:`~alkahest.fex.SearchResult`
or a :class:`~alkahest.fex.FitResult` with its template) and turns it into an
exact closed form:

1. **Coefficient processing.** Parameters that do not change the function
   family are pinned (:meth:`Template.redundant_params`); coefficients below
   ``tau`` are zeroed (Song–Cameron–Yang filtering); leaf weights that agree
   are grouped and refitted as one tied parameter (Hardwick–Liang–Yang); each
   free value is then *snapped* to an exact constant — a rational (continued
   fractions), a rational multiple of ``π, √2, √3, e, log 2`` or of a product
   or quotient of two of them, or a two-term relation ``r₀ + r₁·K`` found by
   PSLQ (:func:`alkahest.guess_relation`) or LLL
   (:func:`alkahest.lattice.lll_reduce_rows`). After every snap the remaining
   free parameters are refitted, and the snap is kept only if the loss stays
   at or below the tolerance.
2. **Lifting to symbolic d.** A leaf ``Σᵢ α φ(xᵢ)`` with one tied weight
   becomes ``α·Sum(φ(Indexed(x, i)), i, 1, d)``; the problem's operator is
   written for symbolic ``d`` (``laplacian_indexed``, ``Σⱼ ∂u/∂xⱼ``) and its
   source terms are lifted the same way. Constants may scale with the
   dimension (``−10`` at ``d = 10`` is ``−d``); the alternatives are tried
   and the first family in which the candidate is an exact solution is kept,
   reported, and checked to coincide with the given problem at the fitted
   ``d``.
3. **Verification** with the D9 vocabulary of
   :func:`alkahest.experimental.verify_pde_solution`: ``exactly_verified``
   only from exact symbolic reasoning, for all ``d`` and on every face of the
   box (the other coordinates' sums are free symbols, so the proof covers
   every point of every face). At the fitted ``d`` the residuals are checked
   exactly too; when no exact form exists, an Arb-ball enclosure of the
   residual at the sample points is reported — evidence, never a proof.
4. **Certificates:** a Lean 4 theorem for the concrete-``d`` PDE residual
   when it is cheap (see :mod:`alkahest._fex_lean`); otherwise a withheld
   reason.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

import alkahest as _ak

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

_REMEDIATION = {
    "E-FEX-009": (
        "pass the SearchResult returned by fex.search, or a FitResult together with "
        "template=; the problem must be the fex.Problem the candidate was fitted to"
    ),
    "E-FEX-010": (
        "build the family from fex.family_symbols(problem): rhs / bc / potential as "
        "expressions in d, Indexed(x, i) and t that reduce to the problem's own source "
        "terms at the problem's dimension"
    ),
    "E-FEX-011": (
        "pass two Expr from the same ExprPool (or a SearchResult / finalize result for 'found')"
    ),
}


def _err(code: str, message: str) -> Exception:
    from .alkahest import FexError

    rem = _REMEDIATION[code]
    exc = FexError(f"[{code}] {message}\nRemediation: {rem}")
    exc.code = code
    exc.remediation = rem
    exc.span = None
    return exc


# ---------------------------------------------------------------------------
# Exact constants
# ---------------------------------------------------------------------------

#: The transcendental / irrational basis for snapping, with a complexity cost.
_BASIS = {
    "pi": (math.pi, 1.0),
    "sqrt2": (math.sqrt(2.0), 1.5),
    "sqrt3": (math.sqrt(3.0), 1.5),
    "e": (math.e, 1.5),
    "log2": (math.log(2.0), 2.0),
    "log3": (math.log(3.0), 2.5),
}


def _monomials() -> list[tuple[tuple[tuple[str, int], ...], float, float]]:
    """``1``, each basis constant to the power ±1, ±2, and products /
    quotients of two different ones: ``(powers, value, cost)``."""
    out: list[tuple[tuple[tuple[str, int], ...], float, float]] = [((), 1.0, 0.0)]
    names = list(_BASIS)
    for n in names:
        v, c = _BASIS[n]
        for p in (1, -1, 2, -2):
            out.append((((n, p),), v**p, c * abs(p) + (0.5 if p < 0 else 0.0)))
    for a, b in itertools.combinations(names, 2):
        va, ca = _BASIS[a]
        vb, cb = _BASIS[b]
        for pa, pb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            out.append(
                (
                    ((a, pa), (b, pb)),
                    va**pa * vb**pb,
                    ca + cb + 0.5 * ((pa < 0) + (pb < 0)),
                )
            )
    return out


_MONOMIALS = _monomials()


def _qcost(q: Fraction) -> float:
    return math.log2(1 + abs(q.numerator)) + math.log2(q.denominator)


@dataclass(frozen=True)
class Const:
    """An exact constant ``Σ rₖ · Πⱼ Kⱼ^pⱼ`` over the snapping basis, with
    rational ``rₖ`` (at most two terms)."""

    terms: tuple[tuple[Fraction, tuple[tuple[str, int], ...]], ...]

    @staticmethod
    def rational(q: Fraction | int) -> Const:
        q = Fraction(q)
        return Const(((q, ()),)) if q != 0 else Const(())

    def is_zero(self) -> bool:
        return not self.terms

    def is_rational(self) -> bool:
        return all(not m for _, m in self.terms)

    def as_fraction(self) -> Fraction | None:
        if not self.terms:
            return Fraction(0)
        if len(self.terms) == 1 and not self.terms[0][1]:
            return self.terms[0][0]
        return None

    def value(self) -> float:
        return float(
            sum(float(r) * math.prod(_BASIS[n][0] ** p for n, p in m) for r, m in self.terms)
        )

    def cost(self) -> float:
        c = 0.0
        for r, m in self.terms:
            c += _qcost(r) + sum(_BASIS[n][1] * abs(p) for n, p in m)
        return c + 2.0 * max(0, len(self.terms) - 1)

    def scaled(self, q: Fraction) -> Const:
        return Const(tuple((r * q, m) for r, m in self.terms if r * q != 0))

    def to_expr(self, pool: Any) -> Any:
        if not self.terms:
            return pool.integer(0)
        out = None
        for r, m in self.terms:
            t = _qexpr(r, pool)
            for n, p in m:
                t = t * _basis_expr(n, pool) ** p
            out = t if out is None else out + t
        return out

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for r, m in self.terms:
            s = str(r)
            if m:
                s += "*" + "*".join(f"{n}^{p}" if p != 1 else n for n, p in m)
            parts.append(s)
        return " + ".join(parts)


def _qexpr(q: Fraction, pool: Any) -> Any:
    if q.denominator == 1:
        return pool.integer(q.numerator)
    return pool.rational(q.numerator, q.denominator)


def _basis_expr(name: str, pool: Any) -> Any:
    if name == "pi":
        return pool.pi()
    if name == "e":
        return _ak.exp(pool.integer(1))
    if name == "sqrt2":
        return _ak.sqrt(pool.integer(2))
    if name == "sqrt3":
        return _ak.sqrt(pool.integer(3))
    if name == "log2":
        return _ak.log(pool.integer(2))
    if name == "log3":
        return _ak.log(pool.integer(3))
    raise KeyError(name)


def _close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def snap_candidates(
    v: float, *, max_den: int = 1000, rel_tol: float = 1e-7, limit: int = 8
) -> list[tuple[Const, str]]:
    """Exact constants close to ``v``, simplest first.

    A candidate of description cost ``c`` (bits of its numerators,
    denominators and basis constants) is accepted only within
    ``rel_tol · 2^-c`` of ``v``: a more complicated constant has to match
    more digits, so a dense supply of complicated candidates cannot "match"
    an arbitrary number.

    Returns ``(Const, how)`` with ``how`` one of ``"rational"`` (continued
    fractions), ``"constant"`` (a rational multiple of a basis monomial),
    ``"pslq"`` (``r₀ + r₁·K`` from :func:`alkahest.guess_relation`) and
    ``"lll"`` (``r₀ + r₁·K₁ + r₂·K₂`` from an LLL-reduced lattice).
    """
    if not math.isfinite(v):
        return []
    found: dict[Const, str] = {}
    if v == 0.0:
        return [(Const(()), "rational")]
    # Rationals and rational multiples of a monomial: continued fractions
    # (Fraction.limit_denominator is the best approximation with bounded
    # denominator, i.e. a convergent or semiconvergent).
    for m, mv, _cost in _MONOMIALS:
        r = v / mv
        q = Fraction(r).limit_denominator(max_den)
        if q != 0:
            c = Const(((q, m),))
            if _close(c.value(), v, rel_tol * 2.0 ** -c.cost()):
                found.setdefault(c, "rational" if not m else "constant")
    # Two-term relations v = r0 + r1*K with PSLQ.
    for n, (kv, _kc) in _BASIS.items():
        try:
            rel = _ak.guess_relation([v, 1.0, kv], max_abs_coeff=max_den)
        except Exception:  # E-PSLQ refusals and bad input both mean "no relation"
            rel = None
        if rel and rel[0] != 0 and rel[1] != 0 and rel[2] != 0:
            a, b, c = rel
            const = Const(((Fraction(-b, a), ()), (Fraction(-c, a), ((n, 1),))))
            if _close(const.value(), v, rel_tol * 2.0 ** -const.cost()):
                found.setdefault(const, "pslq")
    # Three-term relations v = r0 + r1*K1 + r2*K2 with LLL.
    from .lattice import lll_reduce_rows

    scale = 2**36
    for a, b in itertools.combinations(_BASIS, 2):
        va, vb = _BASIS[a][0], _BASIS[b][0]
        rows = [
            [1, 0, 0, 0, round(scale * v)],
            [0, 1, 0, 0, scale],
            [0, 0, 1, 0, round(scale * va)],
            [0, 0, 0, 1, round(scale * vb)],
        ]
        try:
            red = lll_reduce_rows(rows)
        except Exception:
            continue
        for row in red[:2]:
            c0, c1, c2, c3, _ = row
            if c0 == 0 or max(abs(c0), abs(c1), abs(c2), abs(c3)) > max_den:
                continue
            if c2 == 0 or c3 == 0:
                continue  # one-constant relations are found above
            const = Const(
                (
                    (Fraction(-c1, c0), ()),
                    (Fraction(-c2, c0), ((a, 1),)),
                    (Fraction(-c3, c0), ((b, 1),)),
                )
            )
            const = Const(tuple(t for t in const.terms if t[0] != 0))
            if _close(const.value(), v, rel_tol * 2.0 ** -const.cost()):
                found.setdefault(const, "lll")
    out = sorted(found.items(), key=lambda kv: (kv[0].cost(), abs(kv[0].value() - v)))
    return out[:limit]


def _exact_float(x: float) -> Const | None:
    """The exact constant a source float stands for (``0.2`` → ``1/5``,
    ``0.785…`` → ``π/4``), accepted only to near machine precision."""
    if x == 0.0:
        return Const(())
    cands = snap_candidates(x, max_den=10**6, rel_tol=1e-14, limit=1)
    return cands[0][0] if cands else None


# ---------------------------------------------------------------------------
# Refitting with tied and fixed parameters
# ---------------------------------------------------------------------------


class _Refitter:
    """Fit the free parameter groups of one candidate with the others held."""

    def __init__(self, ev: Any, e: list[int], theta: np.ndarray, least_squares: bool):
        self.ev = ev
        self.e = list(e)
        self.n = len(theta)
        self.theta = np.array(theta, dtype=np.float64)
        self.ls = least_squares
        self.evals = 0

    def expand(self, groups: list[list[int]], fixed: dict[int, float], phi: np.ndarray):
        th = self.theta.copy()
        for k, v in fixed.items():
            th[k] = v
        for g, ks in enumerate(groups):
            th[ks] = phi[g]
        return th

    def loss(self, th: np.ndarray) -> float:
        self.evals += 1
        val = self.ev.loss(self.e, th)
        return float(val) if np.isfinite(val) else math.inf

    def fit(
        self,
        groups: list[list[int]],
        fixed: dict[int, float],
        phi0: np.ndarray,
        max_iter: int = 60,
    ) -> tuple[np.ndarray, float]:
        """Minimise over ``phi`` (one value per group). Returns the best
        ``theta`` and its loss."""
        if not groups:
            th = self.expand(groups, fixed, phi0)
            return th, self.loss(th)
        T = np.zeros((self.n, len(groups)))
        for g, ks in enumerate(groups):
            T[ks, g] = 1.0
        if self.ls:
            return self._lm(groups, fixed, np.array(phi0, dtype=float), T, max_iter)
        return self._bfgs(groups, fixed, np.array(phi0, dtype=float), T, max_iter * 3)

    def _lm(self, groups, fixed, phi, T, max_iter):
        th = self.expand(groups, fixed, phi)
        r, J = self.ev.residual_and_jacobian(self.e, th)
        self.evals += 1
        f = float(r @ r) if np.all(np.isfinite(r)) else math.inf
        if not math.isfinite(f):
            return th, math.inf
        lam = 1e-3
        for _ in range(max_iter):
            Jp = J @ T
            g = Jp.T @ r
            A = Jp.T @ Jp
            improved = False
            for _ in range(12):
                M = A + lam * np.diag(np.maximum(np.diag(A), 1e-12))
                try:
                    step = -np.linalg.solve(M, g)
                except np.linalg.LinAlgError:
                    step = -np.linalg.lstsq(M, g, rcond=None)[0]
                phi_new = phi + step
                th_new = self.expand(groups, fixed, phi_new)
                r_new, J_new = self.ev.residual_and_jacobian(self.e, th_new)
                self.evals += 1
                f_new = float(r_new @ r_new) if np.all(np.isfinite(r_new)) else math.inf
                if f_new < f:
                    phi, th, r, J, f = phi_new, th_new, r_new, J_new, f_new
                    lam = max(lam / 5.0, 1e-15)
                    improved = True
                    break
                lam *= 8.0
            if not improved or f < 1e-32:
                break
        return th, f

    def _bfgs(self, groups, fixed, phi, T, max_iter):
        def fg(p):
            th = self.expand(groups, fixed, p)
            val, grad = self.ev.loss_and_grad(self.e, th)
            self.evals += 1
            val = float(val)
            if not math.isfinite(val):
                return math.inf, np.zeros_like(p), th
            return val, T.T @ grad, th

        f, g, th = fg(phi)
        if not math.isfinite(f):
            return th, math.inf
        H = np.eye(len(phi))
        for _ in range(max_iter):
            d = -H @ g
            if g @ d >= 0:
                H = np.eye(len(phi))
                d = -g
            a = 1.0
            ok = False
            for _ in range(30):
                p2 = phi + a * d
                f2, g2, th2 = fg(p2)
                if f2 <= f + 1e-4 * a * (g @ d):
                    ok = True
                    break
                a *= 0.5
            if not ok:
                break
            s, y = p2 - phi, g2 - g
            phi, f, g, th = p2, f2, g2, th2
            sy = s @ y
            if sy > 1e-300:
                rho = 1.0 / sy
                eye = np.eye(len(phi))
                H = (eye - rho * np.outer(s, y)) @ H @ (
                    eye - rho * np.outer(y, s)
                ) + rho * np.outer(s, s)
            if np.linalg.norm(g) < 1e-14 * max(1.0, abs(f)):
                break
        return th, f


# ---------------------------------------------------------------------------
# Templates as trees
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    kind: str  # "leaf" | "unary" | "binary"
    slot: int
    a: _Node | None = None
    b: _Node | None = None


def _parse_shape(shape: str) -> _Node:
    s = shape.replace(" ", "")
    pos = 0
    counter = [0]

    def node() -> _Node:
        nonlocal pos
        c = s[pos].upper()
        pos += 1
        if c == "L":
            n = _Node("leaf", counter[0])
            counter[0] += 1
            return n
        if c == "U":
            pos += 1  # (
            child = node()
            pos += 1  # )
            n = _Node("unary", counter[0], child)
            counter[0] += 1
            return n
        pos += 1  # (
        left = node()
        slot = counter[0]
        counter[0] += 1
        pos += 1  # ,
        right = node()
        pos += 1  # )
        return _Node("binary", slot, left, right)

    return node()


def _unary(op: str, a: Any, pool: Any) -> Any:
    if op == "0":
        return pool.integer(0)
    if op == "1":
        return pool.integer(1)
    if op == "id":
        return a
    if op == "sq":
        return a**2
    if op == "cube":
        return a**3
    if op == "quartic":
        return a**4
    if op == "exp":
        return _ak.exp(a)
    if op == "sin":
        return _ak.sin(a)
    if op == "cos":
        return _ak.cos(a)
    if op == "log":
        return _ak.log(a)
    if op == "abs":
        return _ak.abs(a)
    raise ValueError(op)


def _binary(op: str, a: Any, b: Any) -> Any:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def _children(e: Any) -> list[Any]:
    nd = e.node()
    out = []
    for a in nd[1:]:
        if isinstance(a, (list, tuple)):
            out.extend(c for c in a if isinstance(c, _ak.Expr))
        elif isinstance(a, _ak.Expr):
            out.append(a)
    return out


def _symbols_in(e: Any, acc: set | None = None) -> set:
    acc = set() if acc is None else acc
    stack = [e]
    seen = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        nd = n.node()
        if nd[0] == "symbol":
            acc.add(n)
        else:
            stack.extend(_children(n))
    return acc


def _number_value(e: Any) -> Fraction | float | None:
    """The value of a numeric literal node (integer, rational, float, or a
    power of one), else ``None``."""
    nd = e.node()
    k = nd[0]
    if k == "integer":
        return Fraction(int(nd[1]))
    if k == "rational":
        return Fraction(int(nd[1]), int(nd[2]))
    if k == "float":
        return float(nd[1])
    if k == "pow":
        base, ex = nd[1], nd[2] if len(nd) > 2 else None
        if isinstance(base, list):
            base, ex = base
        b = _number_value(base)
        x = _number_value(ex) if ex is not None else None
        if (
            isinstance(b, Fraction)
            and isinstance(x, Fraction)
            and x.denominator == 1
            and b != 0
            and abs(x) <= 64
        ):
            return b ** int(x)
    return None


def _rebuild(e: Any, f: Any) -> Any:
    """Rebuild ``e`` bottom-up, applying ``f(node, new_children)`` →
    ``Expr | None`` (``None`` keeps the default reconstruction)."""
    nd = e.node()
    kind = nd[0]
    kids = _children(e)
    if not kids:
        r = f(e, [])
        return e if r is None else r
    new = [_rebuild(c, f) for c in kids]
    r = f(e, new)
    if r is not None:
        return r
    if kind == "add":
        out = new[0]
        for c in new[1:]:
            out = out + c
        return out
    if kind == "mul":
        out = new[0]
        for c in new[1:]:
            out = out * c
        return out
    if kind == "pow":
        return new[0] ** new[1]
    if kind == "func":
        return e._pool.func(nd[1], new)
    return e


# ---------------------------------------------------------------------------
# The symbolic-d setting
# ---------------------------------------------------------------------------


@dataclass
class FamilySymbols:
    """Symbols for writing a problem family in symbolic dimension ``d``:
    ``d`` (number of spatial coordinates), the summation index ``i``, the
    indexed base ``x`` (``Indexed(x, i)`` is ``xᵢ``), the time symbol ``t``
    (when the problem has one), and helpers."""

    pool: Any
    d: Any
    i: Any
    x: Any
    t: Any | None

    def xi(self, index: Any = None) -> Any:
        """``Indexed(x, index)`` (default: the summation index ``i``)."""
        from .experimental import Indexed

        return Indexed(self.x, self.i if index is None else index)

    def sum(self, body: Any) -> Any:
        """``Σ_{i=1}^{d} body(xᵢ)`` for a callable ``body``, or of an
        expression in ``Indexed(x, i)``."""
        from .experimental import Sum

        b = body(self.xi()) if callable(body) else body
        return Sum(b, self.i, 1, self.d)


def family_symbols(problem: Any) -> FamilySymbols:
    """The :class:`FamilySymbols` of ``problem`` (in its pool), for writing a
    ``family=`` for :func:`finalize`."""
    pool = problem.jet.pool
    t = pool.symbol("t") if problem.time is not None else None
    return FamilySymbols(
        pool=pool,
        d=pool.symbol("d", "positive"),
        i=pool.symbol("i", "integer"),
        x=pool.symbol("x"),
        t=t,
    )


class _Lift:
    """Lifting concrete-``d`` expressions to symbolic ``d``.

    Numeric constants become placeholder symbols with a list of exact
    alternatives (the constant itself, and ``c·n^-k · d^k``)."""

    def __init__(self, fs: FamilySymbols, coords: list[Any], time: int | None):
        self.fs = fs
        self.pool = fs.pool
        self.coords = coords
        self.time = time
        self.spatial = [c for k, c in enumerate(coords) if k != time]
        self.n = len(self.spatial)
        self.xi_sym = self.pool.symbol("__fex_xi")
        self.placeholders: list[tuple[Any, Const, list[tuple[Const, int]]]] = []
        self._by_value: dict[Any, Any] = {}
        self.failure: str | None = None

    def const(self, c: Const, key: Any = None, scalable: bool = True) -> Any:
        """A placeholder for exact constant ``c`` (``0`` / ``±1`` stay).
        Occurrences with the same ``key`` share one placeholder (a tied
        parameter); without a key every occurrence gets its own."""
        f = c.as_fraction()
        if f is not None and f in (0, 1, -1):
            return _qexpr(f, self.pool)
        key = ("site", len(self.placeholders)) if key is None else key
        if key in self._by_value:
            return self._by_value[key]
        sym = self.pool.symbol(f"__fex_c{len(self.placeholders)}")
        alts: list[tuple[Const, int]] = [(c, 0)]
        if scalable:
            for k in (1, -1, 2, -2):
                a = c.scaled(Fraction(self.n) ** (-k))
                if all(r.denominator <= 10**6 and abs(r.numerator) <= 10**6 for r, _ in a.terms):
                    alts.append((a, k))
        self.placeholders.append((sym, c, alts))
        self._by_value[key] = sym
        return sym

    def lift_source(self, e: Any) -> Any | None:
        """``e`` (an expression in the concrete coordinates) with every
        symmetric sum over the spatial coordinates replaced by a ``Sum`` and
        every numeric constant by a placeholder; ``None`` (with
        ``self.failure`` set) when a coordinate is left over."""
        if e is None:
            return None
        if isinstance(e, (int, float, Fraction)):
            c = Const.rational(e) if not isinstance(e, float) else _exact_float(e)
            if c is None:
                self.failure = f"source constant {e!r} has no exact form"
                return None
            return self.const(c)
        out = self._lift(e)
        if out is None:
            return None
        return out

    def _lift(self, e: Any) -> Any | None:
        nv = _number_value(e)
        if nv is not None:
            c = Const.rational(nv) if isinstance(nv, Fraction) else _exact_float(nv)
            if c is None:
                self.failure = f"constant {nv!r} has no exact form"
                return None
            return self.const(c)
        nd = e.node()
        kind = nd[0]
        if kind == "symbol":
            if e in self.spatial:
                self.failure = f"coordinate {e} is not part of a symmetric sum"
                return None
            if self.time is not None and e == self.coords[self.time]:
                return self.fs.t
            return e
        if kind == "add":
            return self._lift_add(_children(e))
        if kind == "pow":
            base, ex = _children(e)
            b = self._lift(base)
            if b is None:
                return None
            if _number_value(ex) is not None:
                return b**ex
            x = self._lift(ex)
            return None if x is None else b**x
        kids = _children(e)
        if not kids:
            return e
        new = []
        for c in kids:
            lc = self._lift(c)
            if lc is None:
                return None
            new.append(lc)
        if kind == "mul":
            out = new[0]
            for c in new[1:]:
                out = out * c
            return out
        if kind == "func":
            return self.pool.func(nd[1], new)
        self.failure = f"cannot lift a {kind} node"
        return None

    def _lift_add(self, terms: list[Any]) -> Any | None:
        groups: dict[str, list[tuple[Any, Any]]] = {}
        rest = []
        for t in terms:
            syms = _symbols_in(t)
            hit = [c for c in self.spatial if c in syms]
            if len(hit) == 1:
                pat = _ak.subs(t, {hit[0]: self.xi_sym})
                groups.setdefault(str(pat), []).append((hit[0], pat))
            else:
                rest.append(t)
        out = None
        for _key, members in groups.items():
            if sorted(str(c) for c, _ in members) != sorted(str(c) for c in self.spatial):
                self.failure = (
                    "a sum over the coordinates is not symmetric (it does not cover "
                    "every spatial coordinate exactly once with the same term)"
                )
                return None
            pat = members[0][1]
            body = self._lift(pat)
            if body is None:
                return None
            body = _ak.subs(body, {self.xi_sym: self.fs.xi()})
            term = self.fs.sum(body)
            out = term if out is None else out + term
        for t in rest:
            lt = self._lift(t)
            if lt is None:
                return None
            out = lt if out is None else out + lt
        return out if out is not None else self.pool.integer(0)


# ---------------------------------------------------------------------------
# Operators: concrete coordinates and symbolic d
# ---------------------------------------------------------------------------


def _simp(e: Any) -> Any:
    return _ak.simplify(e).value


def _sum_partials_indexed(u: Any, fs: FamilySymbols) -> Any:
    """``Σ_{j=1}^{d} ∂u/∂x_j`` with the Kronecker deltas collapsed."""
    from .experimental import Indexed, Sum, simplify_sums

    j = fs.pool.symbol("j", "integer")
    one = fs.pool.integer(1)
    d1 = _ak.diff(u, Indexed(fs.x, j)).value
    d1 = simplify_sums(d1, [(j, one, fs.d)])
    return simplify_sums(Sum(d1, j, 1, fs.d))


def _operator_symbolic(kind: str, u: Any, fs: FamilySymbols, src: dict[str, Any]) -> Any:
    from .experimental import laplacian_indexed

    if kind in ("poisson", "cubic_schrodinger", "eigen"):
        lap = laplacian_indexed(u, fs.x, 1, fs.d)
        r = -lap
        if kind == "cubic_schrodinger":
            r = r + u**3
        if src.get("potential") is not None:
            r = r + src["potential"] * u
        if kind == "eigen":
            r = r - src["eigenvalue"] * u
        if src.get("rhs") is not None:
            r = r - src["rhs"]
        return r
    if kind == "transport":
        r = src["space_coeff"] * _sum_partials_indexed(u, fs)
        if fs.t is not None:
            r = r + src["time_coeff"] * _ak.diff(u, fs.t).value
        if src.get("rhs") is not None:
            r = r - src["rhs"]
        return r
    raise ValueError(kind)


def _operator_concrete(
    kind: str, u: Any, coords: list[Any], time: int | None, src: dict[str, Any]
) -> Any:
    spatial = [c for k, c in enumerate(coords) if k != time]
    if kind in ("poisson", "cubic_schrodinger", "eigen"):
        lap = None
        for x in spatial:
            t = _ak.diff(_ak.diff(u, x).value, x).value
            lap = t if lap is None else lap + t
        assert lap is not None, "a Laplacian needs a spatial coordinate"
        r = -lap
        if kind == "cubic_schrodinger":
            r = r + u**3
        if src.get("potential") is not None:
            r = r + src["potential"] * u
        if kind == "eigen":
            r = r - src["eigenvalue"] * u
        if src.get("rhs") is not None:
            r = r - src["rhs"]
        return r
    if kind == "transport":
        r = None
        for k, x in enumerate(coords):
            t = src["coeffs"][k] * _ak.diff(u, x).value
            r = t if r is None else r + t
        if src.get("rhs") is not None:
            r = r - src["rhs"]
        return r
    raise ValueError(kind)


def _expand_args(e: Any) -> Any:
    """Expand the arguments of every function application, so that
    ``sin(π/4·(x₁ + x₂))`` and ``sin(π/4·x₁ + π/4·x₂)`` become the same atom
    for the zero test."""

    def f(node: Any, kids: list) -> Any:
        nd = node.node()
        if nd[0] == "func" and nd[1] not in ("Sum", "Indexed", "KroneckerDelta"):
            return node._pool.func(nd[1], [_ak.expand(k) for k in kids])
        return None

    return _rebuild(e, f)


def _relaxed_face(c: dict) -> bool:
    """A face residual written with stand-in symbols for the sums over the
    other coordinates: a proof of zero holds for every value of them, but a
    "nonzero" verdict does not (the stand-ins are not independent)."""
    return "__fex_rest" in str(c["residual"])


def _simplify_args(e: Any) -> Any:
    """Simplify the arguments of every function application (the
    simplifier collects ``x/3 + y/3`` as ``(x + y)/3``), so equal arguments
    written differently become the same atom."""

    def f(node: Any, kids: list) -> Any:
        nd = node.node()
        if nd[0] == "func" and nd[1] not in ("Sum", "Indexed", "KroneckerDelta"):
            return node._pool.func(nd[1], [_simp(_ak.expand(k)) for k in kids])
        return None

    return _rebuild(e, f)


def _verify_residuals(pool: Any, pde: list[Any], bc: list[Any], ic: list[Any]) -> dict:
    """Run residuals (already formed) through the PDE verifier's D9 pipeline:
    simplify sums, alpha-normalise, zero test, with ``proved_nonzero``
    downgraded when an opaque ``Sum`` is involved.

    When that does not prove every residual, it is repeated on further
    normal forms of the *same* residuals — the full simplifier (which
    combines exponentials), function arguments collected,
    expanded, then the whole residual expanded — and a proof (or refutation) from any of
    them stands: nothing is sampled, so none of this weakens the verdict."""
    from .alkahest import _pde_verify

    s = pool.symbol("__fex_s")
    w = pool.func("__fex_w", [s])

    def run(p, b, i):
        conds = [("bc", r, pool.integer(0)) for r in b] + [("ic", r, pool.integer(0)) for r in i]
        return _pde_verify(p, [w], [pool.integer(0)], conds)

    def finish(v: dict) -> dict:
        for c in v["bc"]:
            if c["status"] == "proved_nonzero" and _relaxed_face(c):
                c["status"] = "unknown"
                c["method"] = "none"
                c["evidence"] = [
                    "face residual with stand-ins for the other coordinates' sums: "
                    "zero would be a proof, nonzero is not a refutation"
                ]
        st = [c["status"] for key in ("pde", "bc", "ic") for c in v[key]]
        if all(x == "proved_zero" for x in st):
            status = "exactly_verified"
        elif any(x == "proved_nonzero" for x in st):
            status = "refuted"
        elif all(x in ("proved_zero", "probably_zero") for x in st):
            status = "numerically_checked"
        else:
            status = "unverified"
        v["status"] = status
        v["verified"] = status == "exactly_verified"
        v["evidence"] = {
            "exactly_verified": "symbolic_residual",
            "refuted": "nonzero_residual",
            "numerically_checked": "ball_arithmetic_samples",
        }.get(status, "none")
        return v

    v = finish(run(pde, bc, ic))
    if v["verified"]:
        return v
    rank = {"proved_zero": 3, "proved_nonzero": 3, "probably_zero": 1, "unknown": 0}
    forms = (
        _simp,
        _simplify_args,
        _expand_args,
        lambda r: _ak.expand(_expand_args(r)),
    )
    for form in forms:
        try:
            v2 = finish(run(*([form(r) for r in rs] for rs in (pde, bc, ic))))
        except Exception:
            continue
        for key in ("pde", "bc", "ic"):
            for j, c2 in enumerate(v2[key]):
                if rank.get(c2["status"], 0) > rank.get(v[key][j]["status"], 0):
                    v[key][j] = c2
        v = finish(v)
        if v["verified"]:
            break
    return v


def _interior_verified(v: dict | None) -> bool:
    """Every PDE residual (not the boundary ones) proved zero."""
    return bool(v) and bool(v.get("pde")) and all(c["status"] == "proved_zero" for c in v["pde"])


def _status_counts(v: dict) -> dict:
    return {
        "status": v["status"],
        "verified": v["verified"],
        "pde": [(c["status"], c["method"]) for c in v["pde"]],
        "bc": [(c["status"], c["method"]) for c in v["bc"]],
        "ic": [(c["status"], c["method"]) for c in v["ic"]],
    }


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def _problem_kind(problem: Any) -> str:
    spec = problem.spec
    kind = spec.get("kind")
    if kind == "poisson" and spec.get("loss") == "rayleigh":
        return "eigen"
    return kind


def _box_from_points(pts: np.ndarray | None, default: tuple[float, float]) -> tuple:
    if pts is None or len(pts) == 0:
        return default
    lo, hi = float(np.min(pts)), float(np.max(pts))
    return (lo, hi)


def _exact_bound(v: float) -> Const:
    """A domain bound as an exact constant (a float is a dyadic rational
    when it has no simpler exact form)."""
    c = _exact_float(v)
    return c if c is not None else Const.rational(Fraction(v))


def _family_float(v: float) -> Const:
    c = _exact_float(v)
    if c is None:
        raise _err("E-FEX-010", f"the family constant {v!r} has no exact form; pass an Expr")
    return c


def finalize(
    result: Any,
    problem: Any,
    *,
    template: Any = None,
    points: Any = None,
    boundary: Any = None,
    tau: float = 0.05,
    group_tol: float = 1e-6,
    max_den: int = 1000,
    snap_tol: float = 1e-4,
    loss_tol: float | None = None,
    family: dict | None = None,
    domain: tuple[float, float] | None = None,
    symbolic_d: bool = True,
    certificate: bool = True,
    interval_points: int = 64,
    rng: np.random.Generator | None = None,
) -> dict:
    """Turn a fitted FEX candidate into an exact, verified closed form.

    Parameters
    ----------
    result : SearchResult or FitResult
        The candidate (``e``, ``theta``). A :class:`SearchResult` carries its
        template and points; a :class:`FitResult` needs ``template=`` (and
        points, or they are sampled from ``domain``).
    problem : Problem
        The problem the candidate was fitted to.
    tau : float
        Filtering threshold: coefficients with ``|θ| < tau`` are tried as
        ``0`` (kept only when the refit passes the loss gate).
    group_tol : float
        Leaf weights within this relative distance are tied.
    max_den, snap_tol : int, float
        Snapping: denominators / relation coefficients up to ``max_den``;
        a candidate of description cost ``c`` bits must lie within
        ``snap_tol · 2^-c`` (relative) of the fitted value.
    loss_tol : float, optional
        The gate for every pin, filter, tie and snap. Default:
        ``max(100·loss₀, 1e-20)`` for residual losses, ``loss₀ + 1e-4·|loss₀|``
        for a Rayleigh quotient (``loss₀``: the loss of ``result`` refitted).
    family : dict, optional
        The problem family in symbolic ``d``, written with
        :func:`family_symbols`: any of ``rhs``, ``bc``, ``potential``,
        ``time_coeff``, ``space_coeff``, ``eigenvalue``. Without it the
        family is inferred from the problem (see the module docs) and
        ``verification["symbolic_d"]["family_inferred"]`` is ``True``.
    domain : (lo, hi), optional
        The box ``[lo, hi]^d`` whose faces carry the boundary condition
        (default: the range of the boundary points, else ``[-1, 1]``). For a
        problem with a time coordinate the condition is on ``t = 0``.

    Returns
    -------
    dict
        ``expr`` — the closed form at the fitted ``d`` (exact when every
        parameter snapped, else with float parameters); ``expr_symbolic_d`` —
        the same for every ``d`` (``None`` when it does not lift);
        ``snapped_params`` — one entry per parameter (``value``, ``exact``,
        ``how``, ``group``); ``verification`` — ``concrete`` (the D9 dict at
        the fitted ``d``), ``symbolic_d`` (the D9 dict for all ``d``, with
        the family it was proved for), ``interval`` (an Arb enclosure of the
        residual at sample points when nothing exact is available; never a
        certification); ``certificate`` — a Lean 4 source and its status, or
        a withheld reason; and ``loss``, ``loss_initial``, ``loss_tol``,
        ``log``.

    Raises
    ------
    FexError
        ``E-FEX-009`` for a malformed ``result`` / ``problem`` pair;
        ``E-FEX-010`` when ``family`` does not reduce to ``problem`` at the
        fitted dimension.
    """
    from . import fex as _fex

    rng = rng if rng is not None else np.random.default_rng(0)
    tpl = template if template is not None else getattr(result, "template", None)
    e = getattr(result, "e", None)
    theta = getattr(result, "theta", None)
    if tpl is None or e is None or theta is None:
        raise _err(
            "E-FEX-009",
            "finalize needs a candidate with e, theta and its template "
            "(a SearchResult, or a FitResult with template=)",
        )
    if not isinstance(problem, _fex.Problem) or not problem.spec:
        raise _err("E-FEX-009", "problem must be an alkahest.fex.Problem")
    if problem.dim != tpl.dim:
        raise _err(
            "E-FEX-009",
            f"the template has dimension {tpl.dim} and the problem {problem.dim}",
        )
    e = [int(v) for v in e]
    theta = np.asarray(theta, dtype=np.float64).ravel()
    if len(e) != tpl.n_slots or len(theta) != tpl.n_params:
        raise _err(
            "E-FEX-009",
            f"the candidate has {len(e)} opcodes and {len(theta)} parameters; the "
            f"template needs {tpl.n_slots} and {tpl.n_params}",
        )
    kind = _problem_kind(problem)
    spec = problem.spec
    time = problem.time
    dim = tpl.dim
    if family is not None:
        fpool = problem.jet.pool
        family = {
            k: (
                v
                if v is None or isinstance(v, _ak.Expr)
                else (
                    _family_float(v).to_expr(fpool)
                    if isinstance(v, float)
                    else _qexpr(Fraction(v), fpool)
                )
            )
            for k, v in family.items()
        }
    pts = points if points is not None else getattr(result, "points", None)
    bd = boundary if boundary is not None else getattr(result, "boundary", None)
    if domain is None:
        default = (0.0, 1.0) if time is not None else (-1.0, 1.0)
        src_pts = bd if bd is not None and time is None else pts
        domain = _box_from_points(src_pts, default)
    if pts is None:
        pts = _fex.sample_box(2000, dim, domain[0], domain[1], rng)
        if bd is None and spec.get("bc_weight", 100.0) > 0:
            if time is not None:
                bd = _fex.sample_initial_slice(500, dim, domain[0], domain[1], time, 0.0, rng)
            else:
                bd = _fex.sample_box_boundary(500, dim, domain[0], domain[1], rng)
    log: list[str] = []
    eig: dict[str, Any] = {}
    fit_problem = problem
    fit_bd = bd
    if kind == "eigen":
        # The Monte Carlo Rayleigh quotient's minimiser is biased; the
        # residual −Δu + V u − λ u vanishes pointwise at an exact eigenpair,
        # whatever the points. Alternate a least-squares fit of θ for fixed λ
        # with the projection λ = ⟨−Δu + V u, u⟩ / ⟨u, u⟩ (the postprocessing
        # of Liang & Yang, §4.4.2), then snap λ and fit to the exact λ.
        theta, lam, lam_exact, fit_problem = _refine_eigen(
            problem, tpl, e, theta, pts, max_den, snap_tol, log
        )
        fit_bd = None
        eig = {"estimate": lam, "exact": lam_exact}
    ev = _fex.Evaluator(tpl, fit_problem, pts, fit_bd)
    least_squares = fit_problem.loss == "least_squares"
    rf = _Refitter(ev, e, theta, least_squares)
    n = tpl.n_params
    kinds = tpl.slot_kinds
    offs = tpl.param_offsets

    # Parameter bookkeeping: `fixed` holds exact snapped values, `groups` the
    # free (possibly tied) parameters.
    fixed: dict[int, Const] = {}
    how: dict[int, str] = {}
    groups: list[list[int]] = [[k] for k in range(n)]

    def fixed_f() -> dict[int, float]:
        return {k: c.value() for k, c in fixed.items()}

    def phi_of(th: np.ndarray, gs: list[list[int]]) -> np.ndarray:
        return np.array([float(np.mean(th[ks])) for ks in gs])

    th_cur, loss_cur = rf.fit(groups, {}, theta.copy())
    loss_initial = float(ev.loss(e, theta))
    if not (loss_cur <= loss_initial):
        th_cur, loss_cur = theta.copy(), loss_initial
    if loss_tol is None:
        if fit_problem.loss == "rayleigh":
            loss_tol = loss_cur + 1e-4 * abs(loss_cur)
        else:
            loss_tol = loss_cur + max(1e-20, min(100.0, 1e-3 / max(loss_cur, 1e-300)) * loss_cur)
    log.append(f"refit: loss {loss_initial:.3e} -> {loss_cur:.3e}; gate {loss_tol:.3e}")

    def attempt(new_fixed: dict[int, Const], new_groups: list[list[int]], label: str) -> bool:
        nonlocal th_cur, loss_cur, fixed, groups
        ff = {k: c.value() for k, c in new_fixed.items()}
        th, lo = rf.fit(new_groups, ff, phi_of(th_cur, new_groups))
        if lo <= loss_tol:
            fixed, groups, th_cur, loss_cur = new_fixed, new_groups, th, lo
            log.append(f"{label}: accepted (loss {lo:.3e})")
            return True
        log.append(f"{label}: rejected (loss {lo:.3e})")
        return False

    def without(gs: list[list[int]], ks: set[int]) -> list[list[int]]:
        return [g for g in ([k for k in g if k not in ks] for g in gs) if g]

    # 1. Pin parameters that do not change the family.
    red = tpl.redundant_params(e)
    pins = {
        k: Const.rational(Fraction(v).limit_denominator(1000))
        for k, v in enumerate(red)
        if v is not None
    }
    if pins and attempt(
        {**fixed, **pins}, without(groups, set(pins)), f"pin {len(pins)} redundant"
    ):
        for k in pins:
            how[k] = "redundant"

    # 2. Filtering: coefficients below tau become 0.
    small = sorted(
        (k for g in groups for k in g if abs(th_cur[k]) < tau), key=lambda k: abs(th_cur[k])
    )
    if small:
        zeros = {k: Const(()) for k in small}
        if attempt({**fixed, **zeros}, without(groups, set(small)), f"filter {len(small)}"):
            for k in small:
                how[k] = "filtered"
        else:
            for k in small:
                if k in fixed:
                    continue
                if attempt({**fixed, k: Const(())}, without(groups, {k}), f"filter θ{k}"):
                    how[k] = "filtered"

    # 3. Grouping: tie the free spatial weights of each leaf that agree.
    def leaf_weights(slot: int) -> list[int]:
        o = offs[slot]
        return [o + i for i in range(dim) if i != time]

    for slot, kd in enumerate(kinds):
        if kd != "leaf":
            continue
        ws = [k for k in leaf_weights(slot) if k not in fixed]
        if len(ws) < 2:
            continue
        ws.sort(key=lambda k: th_cur[k])
        clusters: list[list[int]] = [[ws[0]]]
        for k in ws[1:]:
            prev = th_cur[clusters[-1][-1]]
            if _close(th_cur[k], prev, group_tol) or abs(th_cur[k] - prev) < group_tol * 1e-3:
                clusters[-1].append(k)
            else:
                clusters.append([k])
        for cl in clusters:
            if len(cl) < 2:
                continue
            s = set(cl)
            ng = [g for g in groups if not (set(g) & s)] + [sorted(cl)]
            attempt(dict(fixed), ng, f"tie {len(cl)} weights of leaf slot {slot}")

    # 4. Degenerate directions: a scale shared by a unary slot and the
    # subtree below it (α·(w·s) = (α·w)·s), or a shift shared by two biases,
    # lets a parameter take any value. Pin such a parameter to ±1 / 0 —
    # inner unary slots nearest the root first, so the leaves keep the
    # meaningful values — unless it already has a simple exact value.
    def simple(v: float) -> bool:
        c = snap_candidates(v, max_den=max_den, rel_tol=snap_tol, limit=1)
        return bool(c) and c[0][0].cost() <= 4.0

    def is_bias(k: int) -> bool:
        slot = max(s for s in range(len(offs)) if offs[s] <= k and kinds[s] != "binary")
        return (kinds[slot] == "leaf" and k == offs[slot] + dim) or (
            kinds[slot] == "unary" and k == offs[slot] + 1
        )

    def order_key(g: list[int]) -> tuple:
        k = g[0]
        slot = max(s for s in range(len(offs)) if offs[s] <= k and kinds[s] != "binary")
        return (0 if is_bias(k) else 1, 0 if kinds[slot] == "unary" else 1, -slot, k)

    for g in sorted(groups, key=order_key):
        if g not in groups:
            continue
        v = float(np.mean(th_cur[g]))
        if simple(v):
            continue
        targets = (0,) if is_bias(g[0]) else (1 if v >= 0 else -1, 0)
        for target in targets:
            c = Const.rational(target)
            nf = dict(fixed)
            for k in g:
                nf[k] = c
            if attempt(nf, [h for h in groups if h is not g], f"pin θ{g} -> {target} (degenerate)"):
                for k in g:
                    how[k] = "degenerate"
                break

    # 5. Snapping, one group at a time, simplest first.
    tried: set[tuple[int, Const]] = set()
    progress = True
    while progress and groups:
        progress = False
        options = []
        for gi, g in enumerate(groups):
            v = float(np.mean(th_cur[g]))
            for c, hw in snap_candidates(v, max_den=max_den, rel_tol=snap_tol):
                if (g[0], c) in tried:
                    continue
                rel = abs(c.value() - v) / max(abs(v), 1e-300)
                options.append((c.cost() + 0.1 * math.log10(max(rel, 1e-17)), gi, c, hw))
        options.sort(key=lambda o: o[0])
        for _score, gi, c, hw in options[:24]:
            g = groups[gi]
            tried.add((g[0], c))
            nf = dict(fixed)
            for k in g:
                nf[k] = c
            if attempt(nf, [h for h in groups if h is not g], f"snap θ{g} -> {c} ({hw})"):
                for k in g:
                    how[k] = hw
                progress = True
                break

    all_exact = not groups
    pool = problem.jet.pool
    coords = problem.x
    if all_exact:
        fixed = _drop_half_turns(tpl, e, fixed, coords, log)
    group_of: dict[int, int] = {}
    for gi, g in enumerate(groups):
        for k in g:
            group_of[k] = gi
    snapped = []
    for k in range(n):
        slot = max(s for s in range(len(offs)) if offs[s] <= k and (kinds[s] != "binary"))
        snapped.append(
            {
                "index": k,
                "slot": slot,
                "value": float(th_cur[k]),
                "exact": str(fixed[k]) if k in fixed else None,
                "how": how.get(k),
                "group": group_of.get(k),
            }
        )

    # Exact (or float) instantiation at the fitted d.
    if all_exact:
        params = [fixed[k].to_expr(pool) for k in range(n)]
        expr = _simp(tpl.instantiate_symbolic(e, params, coords))
    else:
        expr = tpl.instantiate(e, th_cur, coords)

    verification: dict[str, Any] = {"concrete": None, "symbolic_d": None, "interval": None}
    out: dict[str, Any] = {
        "expr": expr,
        "expr_symbolic_d": None,
        "exact": all_exact,
        "snapped_params": snapped,
        "verification": verification,
        "certificate": None,
        "loss": loss_cur,
        "loss_initial": loss_initial,
        "loss_tol": loss_tol,
        "theta": th_cur,
        "e": e,
        "log": log,
    }

    if kind == "eigen":
        out["eigenvalue"] = str(eig["exact"]) if eig.get("exact") is not None else None
        out["eigenvalue_estimate"] = eig.get("estimate")
    # Problem constants in exact form.
    src_exact, src_note = _exact_sources(problem, kind, time)
    out["problem_constants"] = src_note
    trivial = all_exact and _ak.is_zero(expr) is True
    out["trivial"] = trivial
    if trivial:
        # u = 0 solves every homogeneous problem; it is never a recovery.
        verification["concrete"] = {
            "status": "trivial_solution",
            "verified": False,
            "reason": "the candidate is identically zero",
        }
    elif kind == "expr":
        verification["concrete"] = {
            "status": "unverified",
            "verified": False,
            "reason": "operators given as expressions are not verified exactly yet",
        }
    elif all_exact and src_exact is not None:
        if kind == "eigen":
            lam_c = eig.get("exact")
            src_exact["eigenvalue"] = lam_c.to_expr(pool) if lam_c is not None else None
        if kind != "eigen" or src_exact.get("eigenvalue") is not None:
            verification["concrete"] = _verify_concrete(
                kind, expr, coords, time, src_exact, domain, spec, pool
            )
    if verification["concrete"] is None or not verification["concrete"].get("verified"):
        verification["interval"] = _interval_residual(
            kind, tpl, e, th_cur, coords, time, src_exact, problem, pts, interval_points, rng
        )

    # Lift to symbolic d.
    if (
        symbolic_d
        and not trivial
        and kind in ("poisson", "transport", "cubic_schrodinger", "eigen")
    ):
        lifted = _lift_and_verify(
            tpl, e, fixed, groups, problem, kind, time, domain, family, all_exact, out
        )
        out["expr_symbolic_d"] = lifted.get("expr")
        verification["symbolic_d"] = lifted
    if certificate:
        from ._fex_lean import concrete_certificate

        out["certificate"] = concrete_certificate(
            kind, expr, coords, time, src_exact, verification["concrete"], all_exact
        )
    return out


def _drop_half_turns(tpl, e, fixed: dict, coords, log: list) -> dict:
    """Remove shifts by multiples of ``π/2`` inside ``sin`` / ``cos``
    (``−sin(a + π) = sin(a)``): each parameter that is a multiple of ``π/2``
    is tried at ``0``, alone or with one inner unary scale negated, and the
    change is kept only when the two closed forms are *proved* equal."""
    pool = coords[0]._pool if coords else None
    if pool is None or not any(
        tpl.unary[e[s]] in ("sin", "cos") for s in range(len(e)) if tpl.slot_kinds[s] != "binary"
    ):
        return fixed
    n = tpl.n_params

    def inst(fx):
        return _simp(tpl.instantiate_symbolic(e, [fx[k].to_expr(pool) for k in range(n)], coords))

    base = inst(fixed)
    scales = [tpl.param_offsets[s] for s, k in enumerate(tpl.slot_kinds) if k == "unary"]
    changed = True
    while changed:
        changed = False
        for k in range(n):
            c = fixed[k]
            if len(c.terms) != 1 or c.terms[0][1] != (("pi", 1),):
                continue
            q = c.terms[0][0]
            if (2 * q).denominator != 1:
                continue
            for flip in [None, *scales]:
                fx = dict(fixed)
                fx[k] = Const(())
                if flip is not None:
                    if flip == k or fixed[flip].is_zero():
                        continue
                    fx[flip] = fixed[flip].scaled(Fraction(-1))
                cand = inst(fx)
                if _ak.is_zero(_simp(cand - base)) is True:
                    log.append(
                        f"half-turn: θ{k} = {c} -> 0"
                        + (f", θ{flip} negated" if flip is not None else "")
                    )
                    fixed, base, changed = fx, cand, True
                    break
            if changed:
                break
    return fixed


def _exact_sources(problem: Any, kind: str, time: int | None) -> tuple[dict | None, list]:
    """The problem's source terms with float constants replaced by the exact
    constants they stand for; ``(None, notes)`` if one has no exact form."""
    spec = problem.spec
    pool = problem.jet.pool
    notes: list[str] = []
    out: dict[str, Any] = {}

    def exactify(v: Any, name: str) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, Fraction)):
            return _qexpr(Fraction(v), pool)
        if isinstance(v, float):
            c = _exact_float(v)
            if c is None:
                raise ValueError(f"{name} = {v!r}")
            if not c.is_rational() or Fraction(v) != c.as_fraction():
                notes.append(f"{name}: {v!r} read as {c}")
            return c.to_expr(pool)

        def f(node: Any, _kids: list) -> Any:
            nv = _number_value(node)
            if isinstance(nv, float):
                c = _exact_float(nv)
                if c is None:
                    raise ValueError(f"{name}: constant {nv!r}")
                notes.append(f"{name}: {nv!r} read as {c}")
                return c.to_expr(pool)
            return None

        return _rebuild(v, f)

    try:
        for key in ("rhs", "bc", "potential"):
            out[key] = exactify(spec.get(key), key)
        if kind == "transport":
            out["coeffs"] = [
                exactify(float(c), f"coefficient {k}") for k, c in enumerate(spec["grad_coeffs"])
            ]
    except ValueError as exc:
        notes.append(f"no exact form for {exc}")
        return None, notes
    return out, notes


def _faces(coords: list[Any], time: int | None, domain: tuple, pool: Any) -> list[tuple[Any, Any]]:
    lo, hi = (_exact_bound(domain[0]), _exact_bound(domain[1]))
    out = []
    for k, x in enumerate(coords):
        if k == time:
            continue
        for b in (lo, hi):
            out.append((x, b.to_expr(pool)))
    return out


def _verify_concrete(kind, expr, coords, time, src, domain, spec, pool) -> dict:
    res = _operator_concrete(kind, expr, coords, time, src)
    bcs, ics = [], []
    has_bc = spec.get("bc_weight", 100.0) > 0
    g = src.get("bc")
    g = pool.integer(0) if g is None else g
    if has_bc:
        if time is not None:
            t = coords[time]
            ics.append(_ak.subs(expr - g, {t: pool.integer(0)}))
        else:
            for x, b in _faces(coords, time, domain, pool):
                bcs.append(_ak.subs(expr - g, {x: b}))
    v = _verify_residuals(pool, [res], bcs, ics)
    v["interior_verified"] = _interior_verified(v)
    v["boundary"] = (
        "initial slice t = 0"
        if time is not None and has_bc
        else (f"faces of [{domain[0]}, {domain[1]}]^d" if has_bc else "none")
    )
    return v


def _potential_values(problem: Any, pts: np.ndarray) -> np.ndarray:
    pot = problem.spec.get("potential")
    if pot is None:
        return np.zeros(len(pts))
    xs = problem.x
    return np.asarray(
        [float(_ak.eval_expr(pot, {x: float(v) for x, v in zip(xs, p)})) for p in pts]
    )


def _eigen_twin(problem: Any, lam: Any) -> Any:
    """``−Δu + V u − λ u = 0`` with the normalised loss ``mean(r²) / mean(u²)``
    (scale-invariant, so the trivial ``u = 0`` is not a minimiser); no
    boundary rows."""
    from . import fex as _fex

    jet = problem.jet
    pot = problem.spec.get("potential")
    pool = jet.pool
    lam_e = lam if isinstance(lam, _ak.Expr) else pool.float(float(lam), 53)
    op = -jet.lap - lam_e * jet.u
    if pot is not None:
        op = op + pot * jet.u
    return _fex.Problem.from_operator(
        jet, op, rhs=0.0, bc=None, bc_weight=0.0, loss="normalized", power=2.0
    )


def _refine_eigen(problem, tpl, e, theta, pts, max_den, snap_tol, log):
    from . import fex as _fex

    sel = np.asarray(pts)[: min(len(pts), 4000)]
    V = _potential_values(problem, sel)
    th = np.array(theta, dtype=float)
    ev0 = _fex.Evaluator(tpl, problem, sel, None)

    def project(t):
        u, _g, lap = ev0.jets(e, t, sel)
        den = float(u @ u)
        return float(((-lap + V * u) @ u) / den) if den > 0 else math.nan

    # The eigenproblem is homogeneous: rescale u so the root scale is ±1
    # (a Rayleigh fit may return any scale), and hold it there.
    root = len(tpl.slot_kinds) - 1
    hold: dict[int, float] = {}
    if tpl.slot_kinds[root] == "unary":
        o = tpl.param_offsets[root]
        if th[o] != 0.0:
            c = abs(th[o])
            th[o] /= c
            th[o + 1] /= c
        hold = {o: float(th[o])}
    lam = project(th)
    free = [[k] for k in range(tpl.n_params) if k not in hold]
    loss = math.inf
    rounds = 0
    for _ in range(30):
        rounds += 1
        ev = _fex.Evaluator(tpl, _eigen_twin(problem, lam), sel, None)
        rf = _Refitter(ev, e, th, False)
        th_new, loss = rf.fit(free, hold, np.array([th[g[0]] for g in free]), max_iter=40)
        lam_new = project(th_new)
        if not math.isfinite(lam_new):
            break
        th = th_new
        done = abs(lam_new - lam) <= 1e-13 * max(1.0, abs(lam))
        lam = lam_new
        if done:
            break
    log.append(f"eigen refinement: λ ≈ {lam!r} after {rounds} rounds (residual loss {loss:.3e})")
    cands = snap_candidates(lam, max_den=max_den, rel_tol=snap_tol, limit=1)
    lam_exact = cands[0][0] if cands else None
    if lam_exact is None:
        log.append("eigenvalue: no exact form; continuing with the float λ")
        return th, lam, None, _eigen_twin(problem, lam)
    log.append(f"eigenvalue snapped: λ = {lam_exact}")
    return th, lam, lam_exact, _eigen_twin(problem, lam_exact.to_expr(problem.jet.pool))


def _interval_residual(kind, tpl, e, th, coords, time, src, problem, pts, m, rng) -> dict | None:
    """Arb-ball enclosures of the residual of the float candidate at sample
    points — rigorous at those points, a statement about nothing else."""
    if src is None or kind not in ("poisson", "transport", "cubic_schrodinger"):
        return None
    try:
        u = tpl.instantiate(e, th, coords)
        res = _operator_concrete(kind, u, coords, time, src)
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
    if pts is None:
        return None
    idx = rng.choice(len(pts), size=min(m, len(pts)), replace=False)
    worst = 0.0
    count = 0
    for p in np.asarray(pts)[idx]:
        b = {x: _ak.ArbBall(float(v)) for x, v in zip(coords, p)}
        try:
            ball = _ak.interval_eval(res, b)
        except Exception:
            continue
        hi = max(abs(float(ball.lo)), abs(float(ball.hi)))
        worst = max(worst, hi)
        count += 1
    return {
        "status": "enclosure",
        "verified": False,
        "max_abs_residual_upper": worst,
        "points": count,
        "note": "rigorous enclosure of the float candidate's residual at sample points only; "
        "not a proof that the residual vanishes",
    }


# ---------------------------------------------------------------------------
# Lifting and the symbolic-d proof
# ---------------------------------------------------------------------------


def _lift_and_verify(tpl, e, fixed, groups, problem, kind, time, domain, family, all_exact, out):
    pool = problem.jet.pool
    fs = family_symbols(problem)
    coords = problem.x
    lift = _Lift(fs, coords, time)
    report: dict[str, Any] = {"status": "not_lifted", "verified": False, "expr": None}
    if not all_exact:
        report["reason"] = "not every parameter has an exact value"
        return report
    dim = tpl.dim
    offs = tpl.param_offsets
    unary, binary = tpl.unary, tpl.binary

    def P(k: int) -> Any:
        return lift.const(fixed[k], key=("param", k))

    root = _parse_shape(tpl.shape)

    def build(nd: _Node) -> Any | None:
        if nd.kind == "leaf":
            op = unary[e[nd.slot]]
            o = offs[nd.slot]
            ws = [k for k in range(o, o + dim) if (k - o) != time]
            vals = {fixed[k] for k in ws}
            if op in ("0", "1"):
                acc = P(o + dim)
                if op == "1":
                    # Σᵢ αᵢ·1 = (Σ αᵢ): a constant of the fitted d.
                    tot = Fraction(0)
                    for k in range(o, o + dim):
                        q = fixed[k].as_fraction()
                        if q is None:
                            return None
                        tot += q
                    acc = acc + lift.const(Const.rational(tot))
                return acc
            terms = []
            if len(vals) == 1:
                (c,) = vals
                if not c.is_zero():
                    terms.append(
                        lift.const(c, key=("weights", nd.slot))
                        * fs.sum(lambda xi: _unary(op, xi, pool))
                    )
            else:
                report["reason"] = (
                    f"leaf slot {nd.slot}: the spatial weights are not all equal "
                    f"({len(vals)} distinct exact values)"
                )
                return None
            if time is not None and not fixed[o + time].is_zero():
                terms.append(P(o + time) * _unary(op, fs.t, pool))
            b = fixed[o + dim]
            if not b.is_zero():
                terms.append(P(o + dim))
            if not terms:
                return pool.integer(0)
            s = terms[0]
            for t in terms[1:]:
                s = s + t
            return s
        assert nd.a is not None
        if nd.kind == "unary":
            a = build(nd.a)
            if a is None:
                return None
            o = offs[nd.slot]
            return P(o) * _unary(unary[e[nd.slot]], a, pool) + P(o + 1)
        assert nd.b is not None
        a = build(nd.a)
        b = build(nd.b)
        if a is None or b is None:
            return None
        return _binary(binary[e[nd.slot]], a, b)

    u_ph = build(root)
    if u_ph is None:
        report.setdefault("reason", "the candidate does not lift")
        return report
    n_cand = len(lift.placeholders)

    # The family's source terms.
    spec = problem.spec
    src_ph: dict[str, Any] = {}
    if family is None:
        for key in ("rhs", "bc", "potential"):
            v = spec.get(key)
            if v is None:
                src_ph[key] = None
                continue
            lv = lift.lift_source(v)
            if lv is None:
                report["reason"] = f"the problem's {key} does not lift: {lift.failure}"
                return report
            src_ph[key] = lv
        if kind == "transport":
            cs = [float(c) for c in spec["grad_coeffs"]]
            sp = [c for k, c in enumerate(cs) if k != time]
            if any(c != sp[0] for c in sp):
                report["reason"] = "the spatial transport coefficients are not all equal"
                return report
            cc = _exact_float(sp[0])
            if cc is None:
                report["reason"] = "a transport coefficient has no exact form"
                return report
            src_ph["space_coeff"] = lift.const(cc)
            if time is not None:
                ct = _exact_float(cs[time])
                if ct is None:
                    report["reason"] = "the time coefficient has no exact form"
                    return report
                src_ph["time_coeff"] = lift.const(ct)
        report["family_inferred"] = True
    else:
        report["family_inferred"] = False
        for key in ("rhs", "bc", "potential", "time_coeff", "space_coeff", "eigenvalue"):
            src_ph[key] = family.get(key)
        if kind == "transport":
            src_ph.setdefault("space_coeff", None)
            if src_ph["space_coeff"] is None:
                src_ph["space_coeff"] = pool.integer(1)
            if src_ph["time_coeff"] is None:
                src_ph["time_coeff"] = pool.integer(0)
        _check_family(problem, kind, time, family, fs, out)
    if kind == "eigen" and src_ph.get("eigenvalue") is None:
        lam = out.get("eigenvalue")
        if lam is None:
            report["reason"] = "no exact eigenvalue"
            return report
        cands = [c for c, _ in snap_candidates(out["eigenvalue_estimate"], rel_tol=1e-4)]
        src_ph["eigenvalue"] = lift.const(cands[0])

    # Residuals in placeholders.
    try:
        res_ph = _operator_symbolic(kind, u_ph, fs, src_ph)
    except Exception as exc:
        report["reason"] = f"symbolic operator failed: {exc}"
        return report
    has_bc = spec.get("bc_weight", 100.0) > 0
    g_ph = src_ph.get("bc")
    g_ph = pool.integer(0) if g_ph is None else g_ph
    bc_ph, ic_ph = [], []
    if has_bc:
        if time is not None:
            ic_ph.append(_ak.subs(u_ph - g_ph, {fs.t: pool.integer(0)}))
        else:
            lo, hi = _exact_bound(domain[0]), _exact_bound(domain[1])
            for b in (lo, hi):
                face = _face_restrict(u_ph - g_ph, fs, b.to_expr(pool))
                if face is None:
                    report["reason"] = "the boundary residual does not split on a face"
                    return report
                bc_ph.append(face)

    # Try the alternatives, fewest d-scalings first.
    # Order: fewest d-scalings of the candidate's own constants first (the
    # solution formula should change least), then the cheapest family.
    alts = [a for _sym, _c, a in lift.placeholders]
    syms = [s for s, _c, _a in lift.placeholders]
    cand = [j < n_cand for j in range(len(alts))]
    space = math.prod(len(a) for a in alts) if alts else 1
    if space > 200_000:
        alts = [[x for x in a if abs(x[1]) <= 1] for a in alts]
        space = math.prod(len(a) for a in alts)
    if space > 200_000:
        alts = [a if not c else a[:1] for a, c in zip(alts, cand)]
    combos = list(itertools.product(*[range(len(a)) for a in alts])) if alts else [()]

    def combo_key(cmb):
        return sum(a[j][0].cost() + 1.5 * abs(a[j][1]) for j, a in zip(cmb, alts))

    combos.sort(key=combo_key)

    def family_of(mp: dict) -> dict:
        return {
            key: str(_simp(_ak.subs(val, mp))) for key, val in src_ph.items() if val is not None
        }

    tried = 0
    best = None
    others: list[dict] = []
    extra_budget = 48
    for cmb in combos[:256]:
        if best is not None and best[0]["verified"]:
            extra_budget -= 1
            if extra_budget < 0 or len(others) >= 2:
                break
        tried += 1
        mp = {}
        for s, j, a in zip(syms, cmb, alts):
            c, k = a[j]
            val = c.to_expr(pool)
            if k:
                val = val * fs.d**k
            mp[s] = val
        pde_r = [_ak.subs(res_ph, mp)]
        bcs = [_ak.subs(r, mp) for r in bc_ph]
        ics = [_ak.subs(r, mp) for r in ic_ph]
        v = _verify_residuals(pool, pde_r, bcs, ics)
        # Rank: fully verified > the PDE itself verified > neither.
        score = 2 if v["verified"] else (1 if _interior_verified(v) else 0)
        if best is None or score > best[2]:
            best = (v, mp, score)
        elif v["verified"]:
            others.append({"expr": _simp(_ak.subs(u_ph, mp)), "family": family_of(mp)})
    assert best is not None
    v, mp, _score = best
    report.update(v)
    report["interior_verified"] = _interior_verified(v)
    report["alternatives_tried"] = tried
    report["expr"] = _simp(_ak.subs(u_ph, mp))
    report["family"] = family_of(mp)
    # Other families (cost order) that the candidate also solves for every
    # d and that agree with the problem at the fitted d: the lift from one
    # dimension is not unique, so they are reported, not hidden.
    report["other_families"] = others
    report["operator"] = _operator_text(kind)
    report["candidate_constants"] = n_cand
    if has_bc:
        report["boundary"] = (
            "initial slice t = 0, every x"
            if time is not None
            else f"every face x_j = {domain[0]}, {domain[1]} (j = 1..d), every other coordinate"
        )
    report["status_note"] = (
        "exact for every d >= 1" if v["verified"] else "not proved for symbolic d"
    )
    report["status"] = v["status"]
    report["verified"] = v["verified"]
    return report


def _operator_text(kind: str) -> str:
    return {
        "poisson": "-Δu = rhs",
        "cubic_schrodinger": "-Δu + u³ + V·u = rhs",
        "eigen": "-Δu + V·u = λ·u",
        "transport": "time_coeff·u_t + space_coeff·Σⱼ u_{xⱼ} = rhs",
    }[kind]


def _face_restrict(r: Any, fs: FamilySymbols, b: Any) -> Any | None:
    """``r`` on the face ``x_j = b``: every ``Sum(body(xᵢ), i, 1, d)``
    becomes ``body(b) + R`` with ``R`` a fresh real symbol standing for the
    sum over the other coordinates (one per distinct body). A residual that
    vanishes identically in the ``R``'s vanishes on the face, for every
    ``j`` and every value of the other coordinates."""
    pool = fs.pool
    rest: dict[str, Any] = {}

    def f(node: Any, kids: list) -> Any:
        nd = node.node()
        if nd[0] == "func" and nd[1] == "Sum":
            from .experimental import Indexed

            args = nd[2]
            body, k = args[0], args[1]
            body = _simp(_ak.subs(body, {k: fs.i}))
            # Σ c·f(xᵢ) = c·Σ f(xᵢ): pull numeric factors out so equal sums
            # written differently share one stand-in.
            coeff = pool.integer(1)
            bn = body.node()
            if bn[0] == "mul":
                num = [c for c in _children(body) if _number_value(c) is not None]
                rest_f = [c for c in _children(body) if _number_value(c) is None]
                if num and rest_f:
                    for c in num:
                        coeff = coeff * c
                    body = rest_f[0]
                    for c in rest_f[1:]:
                        body = body * c
                    body = _simp(body)
            key = str(body)
            if key not in rest:
                rest[key] = pool.symbol(f"__fex_rest{len(rest)}")
            xk = Indexed(fs.x, fs.i)
            return coeff * (_ak.subs(body, {xk: b}) + rest[key])
        return None

    try:
        return _rebuild(r, f)
    except Exception:
        return None


def _check_family(problem, kind, time, family, fs, out) -> None:
    """``E-FEX-010`` unless the family's source terms at the problem's
    dimension equal the problem's own (exactly)."""
    from .experimental import Indexed, expand_finite_sums

    pool = problem.jet.pool
    coords = problem.x
    spatial = [c for k, c in enumerate(coords) if k != time]
    n = len(spatial)
    src, _notes = _exact_sources(problem, kind, time)
    if src is None:
        return

    def at_n(v: Any) -> Any:
        v = _ak.subs(v, {fs.d: pool.integer(n)})
        v = expand_finite_sums(v)
        mp = {Indexed(fs.x, pool.integer(k + 1)): spatial[k] for k in range(n)}
        if fs.t is not None:
            mp[fs.t] = coords[time]
        return _ak.subs(v, mp)

    for key in ("rhs", "bc", "potential"):
        fv = family.get(key)
        pv = src.get(key)
        if fv is None and pv is None:
            continue
        fv = pool.integer(0) if fv is None else fv
        pv = pool.integer(0) if pv is None else pv
        if _ak.is_zero(at_n(fv) - pv) is not True:
            raise _err(
                "E-FEX-010",
                f"the family's {key} at d = {n} is {at_n(fv)}, not the problem's {pv}",
            )
    if kind == "transport":
        cs = src["coeffs"]
        sp = [c for k, c in enumerate(cs) if k != time]
        want_s = family.get("space_coeff", pool.integer(1))
        for c in sp:
            if _ak.is_zero(at_n(want_s) - c) is not True:
                raise _err("E-FEX-010", f"space_coeff at d = {n} is not {c}")
        if time is not None:
            want_t = family.get("time_coeff", pool.integer(0))
            if _ak.is_zero(at_n(want_t) - cs[time]) is not True:
                raise _err("E-FEX-010", f"time_coeff at d = {n} is not {cs[time]}")


# ---------------------------------------------------------------------------
# Recovery metric
# ---------------------------------------------------------------------------


def _decide_zero(e: Any) -> bool | None:
    """``is_zero`` on ``e`` and on the normal forms :func:`_verify_residuals`
    uses; the first decided verdict (each is exact, none samples)."""
    forms = (
        lambda r: r,
        _simp,
        _simplify_args,
        _expand_args,
        lambda r: _ak.expand(_expand_args(r)),
    )
    for form in forms:
        try:
            v = _ak.is_zero(form(e))
        except Exception:
            continue
        if v is not None:
            return v
    return None


def _split_sum(p: str) -> list[str]:
    """The top-level terms of a skeleton string ``[a+b+…]``."""
    if not (p.startswith("[") and p.endswith("]")):
        return [p]
    inner, depth, cur, out = p[1:-1], 0, "", []
    for ch in inner:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "+" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    if len(out) == 1:
        return [p]
    return out


def _skeleton(e: Any) -> str:
    """A constant-free normal form: every numeric literal (outside integer
    exponents) and named constant becomes ``C``; a constant *factor* is
    dropped (a scale is a free parameter, so ``C·f`` and ``f`` share a
    skeleton, and ``C·(a + b)`` reads as ``a + b``), constant *terms* merge
    into one ``C``, and commutative arguments are sorted."""

    def go(x: Any) -> str:
        nv = _number_value(x)
        nd = x.node()
        if nv is not None:
            return "C"
        k = nd[0]
        if k == "symbol":
            name = nd[1]
            return "C" if name in ("pi", "E", "e") else name
        if k == "pow":
            base, ex = _children(x)
            ev = _number_value(ex)
            if isinstance(ev, Fraction):
                return f"({go(base)})^{ev}"
            return f"({go(base)})^({go(ex)})"
        if k == "func":
            args = [go(a) for a in _children(x)]
            if nd[1] in ("exp", "log", "sqrt") and all(a == "C" for a in args):
                return "C"
            return f"{nd[1]}({','.join(args)})"
        if k == "add":
            parts = [go(a) for a in _children(x)]
            merged = [p for p in parts if p != "C"]
            if len(merged) < len(parts):
                merged.append("C")
            flat: list[str] = []
            for p in merged:
                flat.extend(_split_sum(p))
            flat = sorted(set(flat)) if len(flat) > 1 else flat
            return flat[0] if len(flat) == 1 else "[" + "+".join(flat) + "]"
        if k == "mul":
            parts = sorted(g for g in (go(a) for a in _children(x)) if g != "C")
            if not parts:
                return "C"
            return parts[0] if len(parts) == 1 else "[" + "*".join(parts) + "]"
        return str(x)

    def ratify(node: Any, _kids: list) -> Any:
        # Floats become nearby rationals: the skeleton abstracts every
        # constant anyway, and expand() refuses float coefficients.
        nv = _number_value(node)
        if isinstance(nv, float) and math.isfinite(nv):
            return _qexpr(Fraction(nv).limit_denominator(10**9), node._pool)
        return None

    e = _rebuild(e, ratify)
    try:
        e = _ak.expand(_expand_args(_simp(e)))
    except Exception:
        e = _simp(e)
    return go(e)


def recovery_metric(found: Any, truth: Any, *, coords: Any = None) -> dict:
    """Has ``found`` recovered ``truth``?

    The definition used by SSDE and SymPlex: after simplification, ``found``
    and ``truth`` have the same *skeleton* — the same expression tree once
    every numeric constant is abstracted to a placeholder (so
    ``0.4999·Σx²`` and ``½Σx²`` agree, ``sin(Σx)`` and ``Σx²`` do not).

    Returns a dict: ``skeleton_equivalent`` (that definition), ``exact`` —
    ``True`` when ``found − truth`` is *proved* zero (``False`` proved
    nonzero, ``None`` undecided) —, and the two skeletons. ``found`` may also
    be a :func:`finalize` result (its exact ``expr``) or a
    :class:`SearchResult`.
    """
    if isinstance(found, dict) and "expr" in found:
        found = found["expr"]
    elif hasattr(found, "expression") and not isinstance(found, _ak.Expr):
        found = found.expression
    if not isinstance(found, _ak.Expr) or not isinstance(truth, _ak.Expr):
        raise _err("E-FEX-011", "recovery_metric compares two Expr")
    sf, st = _skeleton(found), _skeleton(truth)
    exact = _decide_zero(found - truth)
    return {
        "skeleton_equivalent": sf == st,
        "exact": exact,
        "found_skeleton": sf,
        "truth_skeleton": st,
    }
