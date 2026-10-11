"""Linear time-invariant systems: transfer functions, state space, Lyapunov and Riccati.

This is the controls layer (G10). Everything here is built from the exact
kernel — :func:`alkahest.cancel`, :func:`alkahest.solve` (with its ``RootOf``
roots), :class:`alkahest.Matrix` (``inverse``, ``det``, ``matrix_exp``) and
:func:`alkahest.experimental.inverse_laplace_transform` — so a transfer
function may carry symbolic parameters (``m``, ``b``, ``k``, a gain ``K``)
and every exact answer is exact.

* :class:`TransferFunction` — SISO ``G(s) = num(s)/den(s)``: poles, zeros,
  DC gain, series / parallel / feedback, minimal realisation, step and
  impulse responses, Tustin / ZOH discretisation, controllable canonical form.
* :class:`StateSpace` — MIMO ``(A, B, C, D)``: transfer matrix, poles,
  controllability / observability matrices and their (generic) rank, series /
  parallel / feedback interconnection, ZOH / Tustin discretisation.
* :func:`kronecker`, :func:`lyapunov`, :func:`care`, :func:`lqr` — exact for
  small symbolic or rational data, numeric (numpy) otherwise.
* :func:`bode`, :func:`margins` — numeric frequency response, gain and phase
  margins and their crossover frequencies.

**Side conditions.** A result that holds only for generic parameter values —
a pole–zero cancellation, a leading coefficient that was divided by, the
stabilising branch of a Riccati equation — states its hypotheses. They are
read with :func:`controls_side_conditions` right after the call, one string
per condition (``"m ≠ 0"``, ``"r > 0"``), the same out-of-band pattern as
:func:`alkahest.solve_side_conditions`.

**Refusals** raise :class:`ControlsError` with a stable code:

* ``E-CTRL-001`` — malformed input: not a rational function of the variable,
  shapes that do not conform, a MIMO system where a SISO one is needed, an
  unknown method.
* ``E-CTRL-002`` — an improper transfer function where a proper one is
  needed (state-space realisation, impulse response, margins).
* ``E-CTRL-003`` — a numeric operation (Bode, margins, numeric Lyapunov /
  Riccati) on data that still has free parameters; substitute values first.
* ``E-CTRL-004`` — the requested object does not exist: a Lyapunov operator
  that is singular (``λᵢ + λⱼ = 0``), a Riccati equation with no stabilising
  solution, an infinite DC gain.
* ``E-CTRL-005`` — out of the exact engine's reach (symbolic size, an
  unsolvable polynomial system, a stabilising branch that could not be
  selected); the remediation names the numeric route.
* ``E-CTRL-006`` — a numeric routine needs ``numpy``, which is not installed.

Example
-------
>>> import alkahest as ak
>>> from alkahest.controls import TransferFunction
>>> p = ak.ExprPool(); s = p.symbol("s")
>>> G = TransferFunction(p.integer(1), s**2 + 3*s + 2, s)
>>> G.poles()
[-2, -1]
"""

from __future__ import annotations

import cmath
import math
import threading
from fractions import Fraction

from .exceptions import AlkahestError

__all__ = [
    "ControlsError",
    "Margins",
    "StateSpace",
    "TransferFunction",
    "bode",
    "care",
    "controls_side_conditions",
    "kronecker",
    "lqr",
    "lyapunov",
    "margins",
]


# ---------------------------------------------------------------------------
# Errors and side conditions
# ---------------------------------------------------------------------------


class ControlsError(AlkahestError):
    """A controls-layer operation refused (``E-CTRL-001`` … ``E-CTRL-006``).

    ``001`` malformed input; ``002`` an improper transfer function where a
    proper one is needed; ``003`` a numeric routine on data with free
    parameters; ``004`` the requested object does not exist (singular Lyapunov
    operator, no stabilising Riccati solution, infinite DC gain); ``005`` out
    of the exact engine's reach — use the numeric route; ``006`` numpy is
    required and missing.
    """

    def __init__(
        self,
        message: str,
        code: str = "E-CTRL-001",
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code=code, remediation=remediation, span=span)


_STATE = threading.local()


def _reset_conditions():
    _STATE.conds = []


def _note(cond: str):
    conds = getattr(_STATE, "conds", None)
    if conds is None:
        conds = _STATE.conds = []
    if cond not in conds:
        conds.append(cond)


def _note_nonzero(expr):
    """Record ``expr ≠ 0`` unless ``expr`` is a non-zero rational constant."""
    if _as_fraction(expr) is None:
        _note(f"{expr} ≠ 0")


def controls_side_conditions() -> list:
    """Hypotheses the most recent controls call on this thread rests on.

    One string per condition, e.g. ``"m ≠ 0"`` (a leading coefficient that was
    divided by) or ``"(s + a) ≠ 0"`` (a cancelled pole–zero pair). An empty list
    means the last result holds for every value of its parameters. Each public
    entry point of :mod:`alkahest.controls` resets the list.
    """
    return list(getattr(_STATE, "conds", []))


def _forward_solver_conditions():
    ak = _ak()
    for c in ak.solve_side_conditions():
        _note(str(c))


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def _ak():
    import alkahest

    return alkahest


def _simp(e):
    ak = _ak()
    return ak.simplify(e).value


def _radsimp(e):
    """Reduce ``sqrt(c)^k`` to ``c^(k//2)·sqrt(c)^(k%2)`` throughout, then simplify.

    The kernel's simplifier leaves ``sqrt(3)^2`` alone; exact Riccati solutions
    for rational data are full of such powers.
    """
    ak = _ak()
    if not isinstance(e, ak.Expr):
        return e

    def walk(x):
        node = x.node()
        tag = node[0]
        if tag == "add":
            return x._pool.add([walk(c) for c in node[1]])
        if tag == "mul":
            return x._pool.mul([walk(c) for c in node[1]])
        if tag == "func":
            return x._pool.func(node[1], [walk(c) for c in node[2]])
        if tag == "pow":
            base, ex = walk(node[1]), node[2]
            k = _as_fraction(ex)
            bn = base.node()
            if (
                k is not None
                and k.denominator == 1
                and k >= 2
                and bn[0] == "func"
                and bn[1] == "sqrt"
                and len(bn[2]) == 1
            ):
                arg = bn[2][0]
                q, r = divmod(int(k), 2)
                out = arg**q
                if r:
                    out = out * base
                return out
            return base.pow_expr(walk(ex))
        return x

    try:
        return _simp(walk(e))
    except Exception:
        return e


def _canon(e):
    """``cancel`` then ``simplify``: lowest terms, with rational constants folded."""
    ak = _ak()
    return _simp(ak.cancel(e))


def _as_fraction(e):
    """Exact rational value of a numeric literal expression, else ``None``."""
    ak = _ak()
    if isinstance(e, bool):
        return None
    if isinstance(e, int):
        return Fraction(e)
    if isinstance(e, Fraction):
        return e
    if not isinstance(e, ak.Expr):
        return None
    node = e._node_exact()
    if node[0] == "integer":
        return Fraction(node[1])
    if node[0] == "rational":
        return Fraction(node[1], node[2])
    return None


def _has_float(e) -> bool:
    ak = _ak()
    if isinstance(e, float):
        return True
    if not isinstance(e, ak.Expr):
        return False
    stack = [e]
    while stack:
        x = stack.pop()
        node = x.node()
        tag = node[0]
        if tag == "float":
            return True
        stack.extend(_children(node))
    return False


def _children(node):
    tag = node[0]
    if tag in ("add", "mul"):
        return list(node[1])
    if tag == "pow":
        return [node[1], node[2]]
    if tag == "func":
        return list(node[2])
    if tag == "piecewise":
        out = []
        for c, v in node[1]:
            out += [c, v]
        out.append(node[2])
        return out
    if tag == "predicate":
        return list(node[2])
    return []


def _free_symbols(e) -> list:
    """Free symbols of ``e`` in first-seen order (``RootOf`` has none)."""
    ak = _ak()
    if not isinstance(e, ak.Expr):
        return []
    seen = []
    stack = [e]
    while stack:
        x = stack.pop()
        node = x.node()
        if node[0] == "symbol":
            if x not in seen:
                seen.append(x)
            continue
        stack.extend(reversed(_children(node)))
    return seen


def _lift(like, v):
    """Lift a Python number (int, Fraction, float) into ``like``'s pool."""
    ak = _ak()
    v = getattr(v, "value", v) if isinstance(v, ak.DerivedResult) else v
    if isinstance(v, ak.Expr):
        return v
    pool = like._pool
    if isinstance(v, bool):
        raise ControlsError(f"expected a number or expression, got {v!r}")
    if isinstance(v, int):
        return pool.integer(v)
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return pool.integer(v.numerator)
        return pool.rational(v.numerator, v.denominator)
    if isinstance(v, float):
        return pool.float(v)
    raise ControlsError(
        f"expected a number or expression, got {type(v).__name__}",
        code="E-CTRL-001",
    )


def _is_zero(e) -> bool:
    """True only when ``e`` is *proven* to be zero (by simplify or cancel)."""
    ak = _ak()
    f = _as_fraction(e)
    if f is not None:
        return f == 0
    z = _simp(e)
    f = _as_fraction(z)
    if f is not None:
        return f == 0
    try:
        c = ak.cancel(z)
    except Exception:
        return False
    f = _as_fraction(c)
    return f is not None and f == 0


def _num_den(e):
    """Split ``cancel(e)`` into ``(numerator, denominator)`` expressions."""
    ak = _ak()
    c = ak.cancel(e)
    node = c.node()
    factors = list(node[1]) if node[0] == "mul" else [c]
    num, den = [], []
    for f in factors:
        fn = f._node_exact()
        if fn[0] == "pow":
            ex = _as_fraction(fn[2])
            if ex is not None and ex < 0 and ex.denominator == 1:
                base = fn[1]
                den.append(base if ex == -1 else base ** int(-ex))
                continue
        if fn[0] == "rational":
            num.append(c._pool.integer(fn[1]))
            den.append(c._pool.integer(fn[2]))
            continue
        num.append(f)
    one = c._pool.integer(1)
    n = _simp(c._pool.mul(num)) if len(num) > 1 else (num[0] if num else one)
    d = _simp(c._pool.mul(den)) if len(den) > 1 else (den[0] if den else one)
    return n, d


def _poly_coeffs(e, var) -> list:
    """Coefficients of the polynomial ``e`` in ``var``, lowest degree first.

    Coefficients may involve other symbols. Raises ``E-CTRL-001`` when ``e`` is
    not a polynomial in ``var``.
    """
    ak = _ak()
    e = _simp(e)
    gens = [var] + [x for x in _free_symbols(e) if x != var]
    try:
        expanded = ak.poly_normal(e, gens)
    except Exception:
        expanded = _simp(ak.simplify_expanded(e).value)
    node = expanded.node()
    terms = list(node[1]) if node[0] == "add" else [expanded]
    buckets: dict = {}
    for t in terms:
        tn = t._node_exact()
        factors = list(tn[1]) if tn[0] == "mul" else [t]
        deg = 0
        rest = []
        for f in factors:
            fn = f._node_exact()
            if f == var:
                deg += 1
            elif fn[0] == "pow" and fn[1] == var:
                ex = _as_fraction(fn[2])
                if ex is None or ex.denominator != 1 or ex < 0:
                    raise ControlsError(
                        f"{e} is not a polynomial in {var}",
                        code="E-CTRL-001",
                        remediation="pass a rational function of the transfer variable",
                    )
                deg += int(ex)
            else:
                if var in _free_symbols(f):
                    raise ControlsError(
                        f"{e} is not a polynomial in {var} (the factor {f} depends on it)",
                        code="E-CTRL-001",
                        remediation="pass a rational function of the transfer variable",
                    )
                rest.append(f)
        if not rest:
            coeff = var._pool.integer(1)
        elif len(rest) == 1:
            coeff = rest[0]
        else:
            coeff = var._pool.mul(rest)
        buckets.setdefault(deg, []).append(coeff)
    if not buckets:
        return [var._pool.integer(0)]
    n = max(buckets)
    out = []
    for k in range(n + 1):
        cs = buckets.get(k)
        if not cs:
            out.append(var._pool.integer(0))
        elif len(cs) == 1:
            out.append(_simp(cs[0]))
        else:
            out.append(_simp(var._pool.add(cs)))
    # Drop trailing coefficients that are proven zero.
    while len(out) > 1 and _is_zero(out[-1]):
        out.pop()
    return out


def _from_coeffs(coeffs, var):
    """Polynomial expression from lowest-first coefficients."""
    pool = var._pool
    terms = []
    for k, c in enumerate(coeffs):
        c = _lift(var, c)
        if _is_zero(c):
            continue
        if k == 0:
            terms.append(c)
        elif k == 1:
            terms.append(c * var)
        else:
            terms.append(c * var**k)
    if not terms:
        return pool.integer(0)
    return _simp(pool.add(terms)) if len(terms) > 1 else _simp(terms[0])


def _fraction_coeffs(coeffs):
    """All coefficients as Fractions, or ``None`` if any is not rational."""
    out = []
    for c in coeffs:
        f = _as_fraction(c)
        if f is None:
            f = _as_fraction(_simp(c))
        if f is None:
            return None
        out.append(f)
    return out


def _complex_value(e, bindings=None) -> complex:
    ak = _ak()
    f = _as_fraction(e)
    if f is not None:
        return complex(float(f))
    try:
        return complex(ak.evaluate(e, bindings or {}, mode="complex").value)
    except Exception as exc:  # unbound parameter, undefined point
        free = _free_symbols(e)
        if free:
            raise ControlsError(
                f"cannot evaluate {e} numerically: free parameters {', '.join(map(str, free))}",
                code="E-CTRL-003",
                remediation="substitute numeric values for the parameters first, "
                "e.g. G.subs({K: 2})",
            ) from exc
        raise


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a test dependency
        raise ControlsError(
            "this numeric routine needs numpy",
            code="E-CTRL-006",
            remediation="pip install numpy",
        ) from exc
    return np


# --- polynomial arithmetic on Fraction / complex coefficient lists (low first)


def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _psub(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) for i in range(n)]


def _ptrim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _pdiff(a):
    return [k * a[k] for k in range(1, len(a))] or [0]


def _pdivmod(a, b):
    a = _ptrim(a)
    b = _ptrim(b)
    if len(a) < len(b):
        return [0], a
    q = [Fraction(0)] * (len(a) - len(b) + 1)
    r = [Fraction(x) for x in a]
    lb = Fraction(b[-1])
    for k in range(len(a) - len(b), -1, -1):
        coef = r[k + len(b) - 1] / lb
        q[k] = coef
        for j, bj in enumerate(b):
            r[k + j] -= coef * bj
    return _ptrim(q), _ptrim(r[: len(b) - 1] or [Fraction(0)])


def _pgcd(a, b):
    a, b = _ptrim(a), _ptrim(b)
    while not (len(b) == 1 and b[0] == 0):
        _, r = _pdivmod(a, b)
        a, b = b, r
    lc = Fraction(a[-1])
    return [Fraction(x) / lc for x in a]


def _squarefree(a):
    """Yun's square-free decomposition over ℚ: ``[(factor, multiplicity)]``."""
    a = _ptrim([Fraction(x) for x in a])
    if len(a) <= 1:
        return []
    out = []
    da = _pdiff(a)
    g = _pgcd(a, da)
    b, _ = _pdivmod(a, g)
    c, _ = _pdivmod(da, g)
    d = _psub(c, _pdiff(b))
    i = 1
    while len(_ptrim(b)) > 1:
        aa = _pgcd(b, d)
        if len(aa) > 1:
            out.append((aa, i))
        b, _ = _pdivmod(b, aa)
        c, _ = _pdivmod(d, aa)
        d = _psub(c, _pdiff(b))
        i += 1
    return out


def _horner(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


# ---------------------------------------------------------------------------
# Roots (poles / zeros)
# ---------------------------------------------------------------------------


def _roots(coeffs, var) -> list:
    """Roots of the polynomial with lowest-first ``coeffs``, with multiplicity.

    Rational coefficients: square-free decomposition over ℚ, then each factor is
    solved exactly by :func:`alkahest.solve` (radicals or ``RootOf``), so a root
    of multiplicity ``m`` appears ``m`` times. Parametric coefficients: the
    distinct roots :func:`alkahest.solve` returns, for generic parameter values
    (the solver's own side conditions are forwarded).
    """
    ak = _ak()
    coeffs = list(coeffs)
    while len(coeffs) > 1 and _is_zero(coeffs[-1]):
        coeffs.pop()
    if len(coeffs) <= 1:
        return []
    fr = _fraction_coeffs(coeffs)
    out = []
    if fr is not None:
        for factor, mult in _squarefree(fr):
            if len(factor) == 2:
                rts = [_lift(var, -factor[0] / factor[1])]
            else:
                poly = _from_coeffs(factor, var)
                sols = ak.solve([poly], [var])
                rts = [sol[var] for sol in sols]
            for r in rts:
                out.extend([r] * mult)
        return _sort_roots(out)
    lc = coeffs[-1]
    _note_nonzero(lc)
    poly = _from_coeffs(coeffs, var)
    try:
        sols = ak.solve([poly], [var])
    except Exception as exc:
        raise ControlsError(
            f"could not solve {poly} = 0 for {var} exactly: {exc}",
            code="E-CTRL-005",
            remediation="substitute numeric values for the parameters, then ask again",
        ) from exc
    _forward_solver_conditions()
    return [sol[var] for sol in sols]


def _sort_roots(rs):
    def key(r):
        try:
            z = _complex_value(r)
            return (0, z.real, z.imag)
        except Exception:
            return (1, 0.0, 0.0)

    return sorted(rs, key=key)


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def _as_matrix(M, like=None, name="matrix"):
    ak = _ak()
    if isinstance(M, ak.Matrix):
        return M
    rows = [list(r) for r in M]
    if like is None:
        for r in rows:
            for x in r:
                if isinstance(x, ak.Expr):
                    like = x
                    break
            if like is not None:
                break
    if like is None:
        raise ControlsError(
            f"{name}: entries are plain Python numbers and no expression gives the pool",
            code="E-CTRL-001",
            remediation="build the matrix with alkahest.Matrix or include at least one "
            "pool expression",
        )
    return ak.Matrix([[_lift(like, x) for x in r] for r in rows])


def _entries(M):
    return [list(r) for r in M.to_list()]


def _any_entry(M):
    for r in M.to_list():
        for x in r:
            return x
    return None


def _zeros(like, r, c):
    ak = _ak()
    z = like._pool.integer(0)
    return ak.Matrix([[z] * c for _ in range(r)])


def _eye(like, n):
    ak = _ak()
    z, o = like._pool.integer(0), like._pool.integer(1)
    return ak.Matrix([[o if i == j else z for j in range(n)] for i in range(n)])


def _block(blocks):
    """Assemble a block matrix from a list of block rows (all ak.Matrix)."""
    ak = _ak()
    rows = []
    for brow in blocks:
        h = brow[0].rows
        for i in range(h):
            row = []
            for B in brow:
                row.extend(B.to_list()[i])
            rows.append(row)
    return ak.Matrix(rows)


def _sub(M, r0, r1, c0, c1):
    ak = _ak()
    L = M.to_list()
    return ak.Matrix([L[i][c0:c1] for i in range(r0, r1)])


def _msimp(M):
    ak = _ak()
    return ak.Matrix([[_simp(x) for x in r] for r in M.to_list()])


def _mcancel(M):
    ak = _ak()
    return ak.Matrix([[_simp(ak.cancel(x)) for x in r] for r in M.to_list()])


def _mscale(M, c):
    ak = _ak()
    return ak.Matrix([[_simp(c * x) for x in r] for r in M.to_list()])


def _matrix_has_float(M):
    return any(_has_float(x) for r in M.to_list() for x in r)


def _matrix_free(M):
    out = []
    for r in M.to_list():
        for x in r:
            for v in _free_symbols(x):
                if v not in out:
                    out.append(v)
    return out


def _is_zero_matrix(M):
    return all(_is_zero(x) for r in M.to_list() for x in r)


def _to_numpy(M, name="matrix"):
    np = _numpy()
    free = _matrix_free(M)
    if free:
        raise ControlsError(
            f"{name} has free parameters {', '.join(map(str, free))}; "
            "the numeric solver needs numbers",
            code="E-CTRL-003",
            remediation="substitute numeric values for the parameters, or use method='exact'",
        )
    vals = [[_complex_value(x) for x in r] for r in M.to_list()]
    arr = np.array(vals, dtype=complex)
    if np.all(np.abs(arr.imag) <= 1e-14 * (1 + np.abs(arr.real))):
        return arr.real.copy()
    return arr


def kronecker(A, B):
    """Kronecker product ``A ⊗ B``.

    For ``A`` of shape ``(m, n)`` and ``B`` of shape ``(p, q)`` returns the
    ``(m·p, n·q)`` block matrix ``[[a₁₁ B, …, a₁ₙ B], …]``. Entries may be
    symbolic. Satisfies ``vec(A X B) = (Bᵀ ⊗ A) vec(X)`` with column-major
    ``vec``, which is how :func:`lyapunov` builds its linear system.

    Example
    -------
    >>> kronecker(Matrix([[a, 0], [0, b]]), Matrix([[1, 1]]))
    [[a, a, 0, 0], [0, 0, b, b]]
    """
    ak = _ak()
    _reset_conditions()
    A = _as_matrix(A, name="A")
    B = _as_matrix(B, like=_any_entry(A), name="B")
    LA, LB = A.to_list(), B.to_list()
    rows = []
    for i in range(A.rows):
        for k in range(B.rows):
            row = []
            for j in range(A.cols):
                for l in range(B.cols):  # noqa: E741
                    row.append(_simp(LA[i][j] * LB[k][l]))
            rows.append(row)
    return ak.Matrix(rows)


def _vec(M):
    """Column-major vectorisation as a list."""
    L = M.to_list()
    return [L[i][j] for j in range(M.cols) for i in range(M.rows)]


def _unvec(v, n, m):
    ak = _ak()
    return ak.Matrix([[v[j * n + i] for j in range(m)] for i in range(n)])


# ---------------------------------------------------------------------------
# Transfer functions
# ---------------------------------------------------------------------------


class TransferFunction:
    """SISO transfer function ``G(var) = num(var) / den(var)``.

    Parameters
    ----------
    num : Expr | int
        Numerator polynomial in ``var`` — or, when ``den`` is omitted, any
        rational function of ``var`` (it is then split by :func:`cancel`).
    den : Expr | int, optional
        Denominator polynomial in ``var``.
    var : Expr
        The transform variable (``s``, or ``z`` for a discrete system).
    dt : Expr | float, optional
        Sampling period of a discrete-time system; ``None`` for continuous.

    ``num`` and ``den`` are kept **as given** — common factors are not
    cancelled behind the caller's back, because a pole–zero cancellation can
    hide an unstable mode. :meth:`minreal` cancels them and states the
    hypothesis. Coefficients may carry symbolic parameters.

    Example
    -------
    >>> G = TransferFunction(1, m*s**2 + b*s + k, s)
    >>> G.to_ss().A        # controllable canonical form (scipy's tf2ss)
    [[-b/m, -k/m], [1, 0]]
    """

    def __init__(self, num, den=None, var=None, *, dt=None):
        ak = _ak()
        if var is None:
            raise ControlsError(
                "TransferFunction needs its variable, e.g. TransferFunction(num, den, s)",
                code="E-CTRL-001",
            )
        if not isinstance(var, ak.Expr) or var.node()[0] != "symbol":
            raise ControlsError(f"the transfer variable must be a symbol, got {var!r}")
        self.var = var
        self.dt = dt
        num = _lift(var, num)
        if den is None:
            n, d = _num_den(num)
        else:
            den = _lift(var, den)
            if _is_zero(den):
                raise ControlsError("the denominator is identically zero", code="E-CTRL-001")
            n1, d1 = _num_den(num)
            n2, d2 = _num_den(den)
            n = _simp(n1 * d2)
            d = _simp(d1 * n2)
        self._num_c = _poly_coeffs(n, var)
        self._den_c = _poly_coeffs(d, var)
        if all(_is_zero(c) for c in self._den_c):
            raise ControlsError("the denominator is identically zero", code="E-CTRL-001")

    # -- construction -----------------------------------------------------

    @classmethod
    def from_coeffs(cls, num, den, var, *, dt=None):
        """Build from coefficient lists, **highest power first** (scipy order)."""
        num_e = _from_coeffs([_lift(var, c) for c in reversed(list(num))], var)
        den_e = _from_coeffs([_lift(var, c) for c in reversed(list(den))], var)
        return cls(num_e, den_e, var, dt=dt)

    @classmethod
    def _raw(cls, num_c, den_c, var, dt):
        obj = cls.__new__(cls)
        obj.var = var
        obj.dt = dt
        obj._num_c = list(num_c)
        obj._den_c = list(den_c)
        while len(obj._num_c) > 1 and _is_zero(obj._num_c[-1]):
            obj._num_c.pop()
        while len(obj._den_c) > 1 and _is_zero(obj._den_c[-1]):
            obj._den_c.pop()
        return obj

    # -- accessors --------------------------------------------------------

    @property
    def num(self):
        """Numerator polynomial (an expression in ``var``)."""
        return _from_coeffs(self._num_c, self.var)

    @property
    def den(self):
        """Denominator polynomial (an expression in ``var``)."""
        return _from_coeffs(self._den_c, self.var)

    @property
    def num_coeffs(self) -> list:
        """Numerator coefficients, highest power first (scipy order)."""
        return list(reversed(self._num_c))

    @property
    def den_coeffs(self) -> list:
        """Denominator coefficients, highest power first (scipy order)."""
        return list(reversed(self._den_c))

    @property
    def expr(self):
        """``num / den`` as one expression."""
        return _simp(self.num / self.den)

    @property
    def is_discrete(self) -> bool:
        return self.dt is not None

    def degree(self) -> tuple:
        """``(deg num, deg den)`` for generic parameter values."""
        return (len(self._num_c) - 1, len(self._den_c) - 1)

    def is_proper(self) -> bool:
        return len(self._num_c) <= len(self._den_c)

    def is_strictly_proper(self) -> bool:
        return len(self._num_c) < len(self._den_c) or all(_is_zero(c) for c in self._num_c)

    def _same_family(self, other):
        if other.var != self.var:
            raise ControlsError(
                f"cannot combine systems in {self.var} and {other.var}", code="E-CTRL-001"
            )
        if (self.dt is None) != (other.dt is None):
            raise ControlsError(
                "cannot combine a continuous-time and a discrete-time system",
                code="E-CTRL-001",
            )

    def _coerce(self, other):
        ak = _ak()
        if isinstance(other, TransferFunction):
            self._same_family(other)
            return other
        if isinstance(other, (int, Fraction, float, ak.Expr)):
            return TransferFunction(_lift(self.var, other), None, self.var, dt=self.dt)
        return NotImplemented

    # -- evaluation -------------------------------------------------------

    def __call__(self, value):
        """``G(value)`` as an expression (exact)."""
        ak = _ak()
        v = _lift(self.var, value)
        return _simp(ak.subs(self.expr, {self.var: v}))

    def subs(self, mapping):
        """Substitute parameter values; returns a new :class:`TransferFunction`."""
        ak = _ak()
        mp = {k: _lift(self.var, v) for k, v in mapping.items()}
        n = [_simp(ak.subs(c, mp)) for c in self._num_c]
        d = [_simp(ak.subs(c, mp)) for c in self._den_c]
        return TransferFunction(
            _from_coeffs(n, self.var), _from_coeffs(d, self.var), self.var, dt=self.dt
        )

    def freqresp(self, w) -> list:
        """Complex frequency response ``G(jω)`` (or ``G(e^{jωT})``) at each ω."""
        num = [_complex_value(c) for c in self._num_c]
        den = [_complex_value(c) for c in self._den_c]
        ws = _as_list(w)
        out = []
        for x in ws:
            if self.dt is None:
                z = 1j * float(x)
            else:
                z = cmath.exp(1j * float(x) * float(_complex_value(_lift(self.var, self.dt)).real))
            dv = _horner(den, z)
            nv = _horner(num, z)
            out.append(complex("inf") if dv == 0 else nv / dv)
        return out

    # -- structure --------------------------------------------------------

    def poles(self) -> list:
        """Roots of the denominator, with multiplicity for rational coefficients.

        Exact: radicals or ``RootOf`` (via :func:`alkahest.solve`), sorted by
        real then imaginary part when numeric. Side conditions (e.g. the
        leading coefficient ``≠ 0``) are in :func:`controls_side_conditions`.
        """
        _reset_conditions()
        return _roots(self._den_c, self.var)

    def zeros(self) -> list:
        """Roots of the numerator (see :meth:`poles`)."""
        _reset_conditions()
        return _roots(self._num_c, self.var)

    def dc_gain(self):
        """Steady-state gain: ``G(0)`` (continuous) or ``G(1)`` (discrete).

        Computed after :meth:`minreal`, so a cancelled pole at the origin does
        not count. A remaining pole at ``0`` (``1``) refuses with
        ``E-CTRL-004`` — the gain is infinite. A symbolic ``den(0)`` is stated
        as a side condition.
        """
        ak = _ak()
        _reset_conditions()
        g = self._minreal_quiet()
        pt = g.var._pool.integer(0 if self.dt is None else 1)
        n0 = _simp(ak.subs(g.num, {g.var: pt}))
        d0 = _simp(ak.subs(g.den, {g.var: pt}))
        if _is_zero(d0):
            raise ControlsError(
                f"the system has a pole at {pt}: its DC gain is infinite",
                code="E-CTRL-004",
            )
        _note_nonzero(d0)
        return _canon(n0 / d0)

    # -- interconnection --------------------------------------------------

    def series(self, other):
        """``other ∘ self`` — the product ``G₁·G₂`` (no cancellation)."""
        _reset_conditions()
        o = self._coerce(other)
        return TransferFunction._raw(
            _coef_mul(self._num_c, o._num_c), _coef_mul(self._den_c, o._den_c), self.var, self.dt
        )

    def parallel(self, other):
        """``G₁ + G₂`` over the common denominator ``d₁·d₂`` (no cancellation)."""
        _reset_conditions()
        o = self._coerce(other)
        n = _coef_add(_coef_mul(self._num_c, o._den_c), _coef_mul(o._num_c, self._den_c))
        return TransferFunction._raw(n, _coef_mul(self._den_c, o._den_c), self.var, self.dt)

    def feedback(self, other=1, sign=-1):
        """Closed loop ``G / (1 − sign·G·H)``; ``sign=-1`` is negative feedback.

        With ``G = n₁/d₁`` and ``H = n₂/d₂`` this is
        ``n₁·d₂ / (d₁·d₂ − sign·n₁·n₂)``, not cancelled.
        """
        _reset_conditions()
        o = self._coerce(other)
        if sign not in (1, -1):
            raise ControlsError("feedback sign must be +1 or -1", code="E-CTRL-001")
        n = _coef_mul(self._num_c, o._den_c)
        nn = _coef_mul(self._num_c, o._num_c)
        dd = _coef_mul(self._den_c, o._den_c)
        d = _coef_add(dd, nn) if sign == -1 else _coef_add(dd, [_simp(-x) for x in nn])
        if all(_is_zero(c) for c in d):
            raise ControlsError(
                "the closed loop is ill-posed: 1 - sign·G·H vanishes identically",
                code="E-CTRL-004",
            )
        return TransferFunction._raw(n, d, self.var, self.dt)

    def __mul__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return o
        return self.series(o)

    __rmul__ = __mul__

    def __add__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return o
        return self.parallel(o)

    __radd__ = __add__

    def __neg__(self):
        return TransferFunction._raw(
            [_simp(-c) for c in self._num_c], self._den_c, self.var, self.dt
        )

    def __sub__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return o
        return self.parallel(-o)

    def __rsub__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return o
        return o.parallel(-self)

    def __truediv__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return o
        if all(_is_zero(c) for c in o._num_c):
            raise ControlsError("division by the zero transfer function", code="E-CTRL-001")
        return TransferFunction._raw(
            _coef_mul(self._num_c, o._den_c), _coef_mul(self._den_c, o._num_c), self.var, self.dt
        )

    def __pow__(self, k):
        if not isinstance(k, int) or k < 0:
            raise ControlsError("only non-negative integer powers", code="E-CTRL-001")
        out = TransferFunction._raw(
            [self.var._pool.integer(1)], [self.var._pool.integer(1)], self.var, self.dt
        )
        for _ in range(k):
            out = out.series(self)
        return out

    def equals(self, other) -> bool:
        """Whether ``self`` and ``other`` are the same rational function."""
        o = self._coerce(other)
        return _is_zero(_simp(self.num * o.den - o.num * self.den))

    # -- minimal realisation ---------------------------------------------

    def _minreal_quiet(self):
        ak = _ak()
        n, d = _num_den(self.num / self.den)
        g = TransferFunction(n, d, self.var, dt=self.dt)
        removed = ak.cancel(self.den / g.den)
        rn, _ = _num_den(removed)
        # The cancelled factor: its roots in s are hidden modes, and a factor in
        # the parameters alone (m in m·s/(m·s²)) is a genericity hypothesis.
        _note_nonzero(rn)
        return g

    def minreal(self):
        """Cancel common factors of numerator and denominator.

        Returns the reduced :class:`TransferFunction`. It equals ``self`` away
        from the roots of the cancelled factor, which is stated as a side
        condition (``"(s + 1) ≠ 0"``) — those are the hidden modes. Cancellation
        is over ℚ(parameters), so it is minimal for **generic** parameter
        values; on a special locus (``a = b`` in ``(s+a)/(s+b)``) a further
        cancellation can appear.
        """
        _reset_conditions()
        return self._minreal_quiet()

    # -- state space -------------------------------------------------------

    def to_ss(self):
        """Controllable canonical form, matching :func:`scipy.signal.tf2ss`.

        With ``den = s^n + a₁ s^{n-1} + … + aₙ`` (after dividing by the leading
        coefficient — a side condition when it is symbolic) and the numerator
        padded to degree ``n``: ``A`` has first row ``[-a₁ … -aₙ]`` and ones on
        the sub-diagonal, ``B = e₁``, ``C = [b₁ − a₁b₀, …, bₙ − aₙb₀]``,
        ``D = b₀``. Refuses an improper ``G`` with ``E-CTRL-002`` and a static
        gain (no states) with ``E-CTRL-001``.
        """
        _reset_conditions()
        return self._to_ss()

    def _to_ss(self):
        ak = _ak()
        if not self.is_proper():
            raise ControlsError(
                f"{self.expr} is improper (deg num > deg den) and has no state-space realisation",
                code="E-CTRL-002",
            )
        n = len(self._den_c) - 1
        if n == 0:
            raise ControlsError(
                "a static gain has no states; use StateSpace with empty A instead",
                code="E-CTRL-001",
            )
        lc = self._den_c[-1]
        _note_nonzero(lc)
        a = [_canon(c / lc) for c in reversed(self._den_c)]  # a[0] = 1
        b = [_canon(c / lc) for c in reversed(self._num_c)]
        b = [self.var._pool.integer(0)] * (n + 1 - len(b)) + b
        pool = self.var._pool
        z, o = pool.integer(0), pool.integer(1)
        A = [[_simp(-a[j + 1]) for j in range(n)]]
        for i in range(1, n):
            A.append([o if j == i - 1 else z for j in range(n)])
        B = [[o]] + [[z] for _ in range(n - 1)]
        C = [[_canon(b[j + 1] - a[j + 1] * b[0]) for j in range(n)]]
        D = [[_simp(b[0])]]
        return StateSpace(ak.Matrix(A), ak.Matrix(B), ak.Matrix(C), ak.Matrix(D), dt=self.dt)

    # -- time responses ----------------------------------------------------

    def impulse_response(self, t):
        """``y(t) = L⁻¹{G(s)}(t)`` for ``t > 0`` (continuous, strictly proper).

        Uses :func:`alkahest.experimental.inverse_laplace_transform`, whose
        refusals (e.g. an irreducible denominator factor of degree > 2,
        ``E-TRANSFORM-002``) propagate. Side conditions from the transform are
        forwarded.
        """
        _reset_conditions()
        return self._ilt(self.expr, t, "impulse")

    def step_response(self, t):
        """Unit-step response ``y(t) = L⁻¹{G(s)/s}(t)`` for ``t > 0``."""
        _reset_conditions()
        return self._ilt(self.expr / self.var, t, "step")

    def _ilt(self, F, t, what):
        ak = _ak()
        from . import experimental as ex

        if self.dt is not None:
            raise ControlsError(
                f"{what}_response is for continuous-time systems",
                code="E-CTRL-001",
            )
        if what == "impulse" and not self.is_strictly_proper():
            raise ControlsError(
                "the impulse response of a biproper G contains δ(t); split off D = G(∞) first",
                code="E-CTRL-002",
            )
        if not self.is_proper():
            raise ControlsError(f"{self.expr} is improper", code="E-CTRL-002")
        y = ex.inverse_laplace_transform(ak.cancel(F), self.var, t)
        for c in ex.transform_side_conditions():
            _note(str(c))
        return y

    # -- discretisation ----------------------------------------------------

    def discretize(self, T, method="tustin", var=None):
        """Discrete-time equivalent with sampling period ``T``.

        ``method="tustin"`` (alias ``"bilinear"``) substitutes
        ``s = (2/T)·(z − 1)/(z + 1)``; ``"zoh"`` goes through
        :meth:`StateSpace.discretize` (exact ``matrix_exp``). ``var`` is the
        new variable (default: a symbol ``z`` in the same pool).
        """
        _reset_conditions()
        if self.dt is not None:
            raise ControlsError("the system is already discrete", code="E-CTRL-001")
        z = var if var is not None else self.var._pool.symbol("z")
        T = _lift(self.var, T)
        if method in ("tustin", "bilinear"):
            ak = _ak()
            s_sub = 2 * (z - 1) / (T * (z + 1))
            Gz = ak.subs(self.expr, {self.var: s_sub})
            n, d = _num_den(Gz)
            return TransferFunction(n, d, z, dt=T)
        if method == "zoh":
            sysd = self._to_ss()._discretize(T, "zoh")
            return sysd._to_tf(z)
        raise ControlsError(f"unknown discretisation method {method!r}", code="E-CTRL-001")

    # -- frequency domain (numeric) -----------------------------------------

    def bode(self, w):
        """``(magnitude_dB, phase_deg)`` at the frequencies ``w``; see :func:`bode`."""
        return bode(self, w)

    def margins(self):
        """Gain/phase margins; see :func:`margins`."""
        return margins(self)

    def __repr__(self):
        tag = "" if self.dt is None else f", dt={self.dt}"
        return f"TransferFunction(({self.num}) / ({self.den}), {self.var}{tag})"


def _coef_mul(a, b):
    out = [None] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            t = x * y
            out[i + j] = t if out[i + j] is None else out[i + j] + t
    return [_simp(x) for x in out]


def _coef_add(a, b):
    n = max(len(a), len(b))
    out = []
    for i in range(n):
        if i < len(a) and i < len(b):
            out.append(_simp(a[i] + b[i]))
        else:
            out.append(a[i] if i < len(a) else b[i])
    return out


def _as_list(w):
    try:
        return [float(x) for x in w]
    except TypeError:
        return [float(w)]


# ---------------------------------------------------------------------------
# State space
# ---------------------------------------------------------------------------


class StateSpace:
    """LTI system ``ẋ = A x + B u``, ``y = C x + D u`` (MIMO).

    ``A`` (n×n), ``B`` (n×m), ``C`` (p×n), ``D`` (p×m, default zero) are
    :class:`alkahest.Matrix` or nested lists; entries may be symbolic.
    ``dt`` marks a discrete-time system ``x[k+1] = A x[k] + B u[k]``.
    """

    def __init__(self, A, B, C, D=None, *, dt=None):
        A = _as_matrix(A, name="A")
        like = _any_entry(A)
        B = _as_matrix(B, like=like, name="B")
        C = _as_matrix(C, like=like, name="C")
        if A.rows != A.cols:
            raise ControlsError(f"A must be square, got {A.rows}×{A.cols}", code="E-CTRL-001")
        n = A.rows
        if B.rows != n:
            raise ControlsError(f"B must have {n} rows, got {B.rows}", code="E-CTRL-001")
        if C.cols != n:
            raise ControlsError(f"C must have {n} columns, got {C.cols}", code="E-CTRL-001")
        D = _zeros(like, C.rows, B.cols) if D is None else _as_matrix(D, like=like, name="D")
        if D.rows != C.rows or D.cols != B.cols:
            raise ControlsError(
                f"D must be {C.rows}×{B.cols}, got {D.rows}×{D.cols}", code="E-CTRL-001"
            )
        self.A, self.B, self.C, self.D = A, B, C, D
        self.dt = dt
        self._like = like

    @property
    def nstates(self) -> int:
        return self.A.rows

    @property
    def ninputs(self) -> int:
        return self.B.cols

    @property
    def noutputs(self) -> int:
        return self.C.rows

    def is_siso(self) -> bool:
        return self.ninputs == 1 and self.noutputs == 1

    def _require_siso(self, what):
        if not self.is_siso():
            raise ControlsError(
                f"{what} is implemented for SISO systems; this one is "
                f"{self.noutputs}×{self.ninputs}",
                code="E-CTRL-001",
            )

    def _same_family(self, other):
        if (self.dt is None) != (other.dt is None):
            raise ControlsError(
                "cannot combine a continuous-time and a discrete-time system",
                code="E-CTRL-001",
            )

    # -- transfer matrix --------------------------------------------------

    def characteristic_polynomial(self, var):
        """``det(var·I − A)``, expanded."""
        ak = _ak()
        M = _mscale(_eye(self._like, self.nstates), var) - self.A
        return _simp(ak.simplify_expanded(M.det()).value)

    def to_tf(self, var):
        """Transfer function ``C (var·I − A)⁻¹ B + D``.

        Returned over the common denominator ``det(var·I − A)`` (degree ``n``,
        not cancelled — as :func:`scipy.signal.ss2tf` does; call
        :meth:`TransferFunction.minreal` to cancel). A SISO system gives one
        :class:`TransferFunction`; a MIMO one a list of rows of them.
        """
        _reset_conditions()
        return self._to_tf(var)

    def _to_tf(self, var):
        ak = _ak()
        n = self.nstates
        sI_A = _mscale(_eye(self._like, n), var) - self.A
        chi = _simp(ak.simplify_expanded(sI_A.det()).value)
        inv = sI_A.inverse()
        adj = _mcancel(_mscale(inv, chi))
        G = self.C @ adj @ self.B
        rows = []
        for i in range(self.noutputs):
            row = []
            for j in range(self.ninputs):
                nij = ak.cancel(G.get(i, j) + self.D.get(i, j) * chi)
                tf = TransferFunction._raw(
                    _poly_coeffs(nij, var), _poly_coeffs(chi, var), var, self.dt
                )
                row.append(tf)
            rows.append(row)
        if self.is_siso():
            return rows[0][0]
        return rows

    def poles(self, var=None) -> list:
        """Eigenvalues of ``A`` — roots of ``det(sI − A)`` with multiplicity."""
        _reset_conditions()
        v = var if var is not None else self._like._pool.symbol("s")
        chi = self.characteristic_polynomial(v)
        return _roots(_poly_coeffs(chi, v), v)

    def zeros(self, var=None) -> list:
        """Zeros of a SISO system (roots of the numerator of :meth:`to_tf`)."""
        self._require_siso("zeros")
        _reset_conditions()
        v = var if var is not None else self._like._pool.symbol("s")
        return _roots(self._to_tf(v)._num_c, v)

    def dc_gain(self):
        """``D − C A⁻¹ B`` (continuous) or ``D + C (I − A)⁻¹ B`` (discrete).

        ``det A ≠ 0`` (resp. ``det(I − A) ≠ 0``) is a side condition when it is
        symbolic; a proven singular one refuses with ``E-CTRL-004``.
        """
        ak = _ak()
        _reset_conditions()
        M = self.A if self.dt is None else _eye(self._like, self.nstates) - self.A
        try:
            Minv = M.inverse()
        except Exception as exc:
            raise ControlsError(
                "A has an eigenvalue at the origin (1 for discrete time): the DC gain is infinite",
                code="E-CTRL-004",
            ) from exc
        for c in ak.matrix_inverse_side_conditions():
            _note(str(c))
        CB = self.C @ Minv @ self.B
        out = self.D - CB if self.dt is None else self.D + CB
        return _mcancel(out)

    # -- controllability / observability -----------------------------------

    def controllability_matrix(self):
        """``[B, AB, …, A^{n−1}B]`` (n × n·m)."""
        blocks = [self.B]
        for _ in range(self.nstates - 1):
            blocks.append(_msimp(self.A @ blocks[-1]))
        return _block([blocks])

    def observability_matrix(self):
        """``[C; CA; …; CA^{n−1}]`` (n·p × n)."""
        blocks = [self.C]
        for _ in range(self.nstates - 1):
            blocks.append(_msimp(blocks[-1] @ self.A))
        return _block([[b] for b in blocks])

    def _rank(self, M):
        ak = _ak()
        r = M.rank()
        for c in ak.matrix_rank_side_conditions():
            _note(c)
        return r

    def controllability_rank(self) -> int:
        """Generic rank of the controllability matrix.

        Exact where the pivot hypotheses in :func:`controls_side_conditions`
        hold (e.g. ``"a ≠ 0"`` for an input gain ``a``), an upper bound
        everywhere.
        """
        _reset_conditions()
        return self._rank(self.controllability_matrix())

    def observability_rank(self) -> int:
        """Generic rank of the observability matrix (see :meth:`controllability_rank`)."""
        _reset_conditions()
        return self._rank(self.observability_matrix())

    def is_controllable(self) -> bool:
        """Generic controllability (rank ``n``), with its side conditions stated."""
        return self.controllability_rank() == self.nstates

    def is_observable(self) -> bool:
        """Generic observability (rank ``n``), with its side conditions stated."""
        return self.observability_rank() == self.nstates

    # -- interconnection ---------------------------------------------------

    def series(self, other):
        """``other ∘ self``: the output of ``self`` drives ``other``."""
        _reset_conditions()
        self._same_family(other)
        n1, n2 = self.nstates, other.nstates
        if other.ninputs != self.noutputs:
            raise ControlsError("series: output/input sizes do not conform", code="E-CTRL-001")
        L = self._like
        A = _block([[self.A, _zeros(L, n1, n2)], [_msimp(other.B @ self.C), other.A]])
        B = _block([[self.B], [_msimp(other.B @ self.D)]])
        C = _block([[_msimp(other.D @ self.C), other.C]])
        D = _msimp(other.D @ self.D)
        return StateSpace(A, B, C, D, dt=self.dt)

    def parallel(self, other):
        """Sum of the outputs of two systems driven by the same input."""
        _reset_conditions()
        self._same_family(other)
        if other.ninputs != self.ninputs or other.noutputs != self.noutputs:
            raise ControlsError("parallel: sizes do not conform", code="E-CTRL-001")
        n1, n2 = self.nstates, other.nstates
        L = self._like
        A = _block([[self.A, _zeros(L, n1, n2)], [_zeros(L, n2, n1), other.A]])
        B = _block([[self.B], [other.B]])
        C = _block([[self.C, other.C]])
        D = _msimp(self.D + other.D)
        return StateSpace(A, B, C, D, dt=self.dt)

    def feedback(self, other=None, sign=-1):
        """Closed loop ``u = r + sign·y₂``, ``y₂ = other(y₁)`` (``sign=-1``: negative).

        ``other`` is a :class:`StateSpace` or a static gain matrix (default:
        unity). Needs ``I − sign·D₁D₂`` invertible; a symbolic determinant is
        a side condition, a singular one refuses with ``E-CTRL-004``.
        """
        ak = _ak()
        _reset_conditions()
        if sign not in (1, -1):
            raise ControlsError("feedback sign must be +1 or -1", code="E-CTRL-001")
        L = self._like
        if other is None:
            if self.ninputs != self.noutputs:
                raise ControlsError("unity feedback needs a square system", code="E-CTRL-001")
            other = _eye(L, self.noutputs)
        if not isinstance(other, StateSpace):
            K = _as_matrix(other, like=L, name="feedback gain")
            other = _StaticGain(K, self.dt, L)
        self._same_family(other)
        p, m = self.noutputs, self.ninputs
        if other.ninputs != p or other.noutputs != m:
            raise ControlsError("feedback: sizes do not conform", code="E-CTRL-001")
        sgn = L._pool.integer(sign)
        M = _eye(L, p) - _mscale(self.D @ other.D, sgn)
        try:
            E = M.inverse()
        except Exception as exc:
            raise ControlsError(
                "the feedback loop is ill-posed: I − sign·D₁·D₂ is singular",
                code="E-CTRL-004",
            ) from exc
        for c in ak.matrix_inverse_side_conditions():
            _note(str(c))
        # y1 = E (C1 x1 + sign D1 C2 x2 + D1 r);  u = r + sign (C2 x2 + D2 y1)
        Cy_x1 = _msimp(E @ self.C)
        Dy = _msimp(E @ self.D)
        # u in terms of (x1, x2, r)
        Cu_x1 = _mscale(other.D @ Cy_x1, sgn)
        Du = _msimp(_eye(L, m) + _mscale(other.D @ Dy, sgn))
        A11 = _msimp(self.A + self.B @ Cu_x1)
        if isinstance(other, StateSpace) and other.nstates:
            Cy_x2 = _mscale(E @ self.D @ other.C, sgn)
            Cu_x2 = _msimp(_mscale(other.C, sgn) + _mscale(other.D @ Cy_x2, sgn))
            A12 = _msimp(self.B @ Cu_x2)
            A21 = _msimp(other.B @ Cy_x1)
            A22 = _msimp(other.A + other.B @ Cy_x2)
            A = _block([[A11, A12], [A21, A22]])
            B = _block([[_msimp(self.B @ Du)], [_msimp(other.B @ Dy)]])
            C = _block([[Cy_x1, Cy_x2]])
        else:
            A, B, C = A11, _msimp(self.B @ Du), Cy_x1
        return StateSpace(_mcancel(A), _mcancel(B), _mcancel(C), _mcancel(Dy), dt=self.dt)

    # -- responses ---------------------------------------------------------

    def impulse_response(self, t):
        """Impulse response (SISO: an expression; MIMO: a matrix), via :meth:`to_tf`."""
        return self._resp(t, "impulse")

    def step_response(self, t):
        """Unit-step response (SISO: an expression; MIMO: a matrix), via :meth:`to_tf`."""
        return self._resp(t, "step")

    def _resp(self, t, what):
        ak = _ak()
        _reset_conditions()
        v = self._like._pool.symbol("s")
        G = self._to_tf(v)
        rows = [[G]] if self.is_siso() else G
        out = []
        for r in rows:
            row = []
            for g in r:
                gm = g._minreal_quiet()
                if what == "step":
                    y = gm._ilt(gm.expr / v, t, "step")
                else:
                    y = gm._ilt(gm.expr, t, "impulse")
                row.append(y)
            out.append(row)
        if self.is_siso():
            return out[0][0]
        return ak.Matrix(out)

    # -- discretisation ----------------------------------------------------

    def discretize(self, T, method="zoh"):
        """Discrete-time equivalent with sampling period ``T``.

        ``"zoh"``: exact ``Ad = e^{AT}``, ``Bd = ∫₀ᵀ e^{Aτ} dτ B``, read off
        ``exp([[A, B], [0, 0]]·T)`` (no integral needed); ``Cd = C``,
        ``Dd = D``. ``"tustin"`` / ``"bilinear"``: :func:`scipy.signal.cont2discrete`'s
        bilinear formulas with α = ½ — ``Ad = (I − AT/2)⁻¹(I + AT/2)``,
        ``Bd = (I − AT/2)⁻¹ B T``, ``Cd = C (I − AT/2)⁻¹``, ``Dd = D + C Bd / 2``.
        """
        _reset_conditions()
        return self._discretize(T, method)

    def _discretize(self, T, method):
        ak = _ak()
        if self.dt is not None:
            raise ControlsError("the system is already discrete", code="E-CTRL-001")
        L = self._like
        T = _lift(L, T)
        n, m = self.nstates, self.ninputs
        if method == "zoh":
            M = _block([[self.A, self.B], [_zeros(L, m, n), _zeros(L, m, m)]])
            E = _mscale(M, T).matrix_exp()
            for c in ak.matrix_exp_side_conditions():
                _note(str(c))
            E = _msimp(E)
            Ad = _sub(E, 0, n, 0, n)
            Bd = _sub(E, 0, n, n, n + m)
            return StateSpace(Ad, Bd, self.C, self.D, dt=T)
        if method in ("tustin", "bilinear"):
            half = L._pool.rational(1, 2)
            Id = _eye(L, n)
            AT2 = _mscale(self.A, _simp(T * half))
            Minv = (Id - AT2).inverse()
            for c in ak.matrix_inverse_side_conditions():
                _note(str(c))
            Ad = _mcancel(Minv @ (Id + AT2))
            Bd = _mcancel(_mscale(Minv @ self.B, T))
            Cd = _mcancel(self.C @ Minv)
            Dd = _mcancel(self.D + _mscale(self.C @ Bd, half))
            return StateSpace(Ad, Bd, Cd, Dd, dt=T)
        raise ControlsError(f"unknown discretisation method {method!r}", code="E-CTRL-001")

    def minreal(self, var=None):
        """Minimal realisation of a SISO system (via :meth:`TransferFunction.minreal`)."""
        self._require_siso("minreal")
        _reset_conditions()
        v = var if var is not None else self._like._pool.symbol("s")
        g = self._to_tf(v)._minreal_quiet()
        return g._to_ss()

    def subs(self, mapping):
        """Substitute parameter values in all four matrices."""
        ak = _ak()
        L = self._like
        mp = {k: _lift(L, v) for k, v in mapping.items()}

        def f(M):
            return ak.Matrix([[_simp(ak.subs(x, mp)) for x in r] for r in M.to_list()])

        return StateSpace(f(self.A), f(self.B), f(self.C), f(self.D), dt=self.dt)

    def __repr__(self):
        tag = "" if self.dt is None else f", dt={self.dt}"
        return f"StateSpace(A={self.A}, B={self.B}, C={self.C}, D={self.D}{tag})"


class _StaticGain:
    """Zero-state system ``y = K u`` used inside :meth:`StateSpace.feedback`."""

    def __init__(self, K, dt, like):
        self.D = K
        self.dt = dt
        self._like = like
        self.nstates = 0
        self.ninputs = K.cols
        self.noutputs = K.rows
        self.A = self.B = self.C = None


# ---------------------------------------------------------------------------
# Lyapunov
# ---------------------------------------------------------------------------

_EXACT_SYMBOLIC_LYAP_MAX = 3
_EXACT_RATIONAL_LYAP_MAX = 8


def lyapunov(A, Q, method="auto"):
    """Solve the continuous Lyapunov equation ``A X + X Aᵀ + Q = 0``.

    (python-control's ``lyap(A, Q)`` convention; :func:`scipy.linalg.solve_continuous_lyapunov`
    solves ``A X + X Aᴴ = Q``, i.e. pass ``-Q`` there.)

    ``method="exact"`` vectorises: ``(I ⊗ A + A ⊗ I) vec X = −vec Q`` and
    solves the linear system with an exact inverse, then **verifies** the
    residual cancels to zero. The operator is invertible iff no two
    eigenvalues of ``A`` satisfy ``λᵢ + λⱼ = 0``; a symbolic determinant is a
    side condition (``controls_side_conditions``), a proven singular one
    refuses with ``E-CTRL-004``. Exact for rational ``A`` up to 8×8 and
    symbolic ``A`` up to 3×3 (``E-CTRL-005`` beyond).

    ``method="numeric"`` solves the same system with numpy and returns a
    ``numpy.ndarray``. ``"auto"`` picks exact unless an entry is a float.

    Returns
    -------
    Matrix | numpy.ndarray

    Example
    -------
    >>> lyapunov(Matrix([[0, 1], [-k, -b]]), Matrix([[1, 0], [0, 1]]))  # X for AX+XAᵀ+I=0
    """
    _reset_conditions()
    A = _as_matrix(A, name="A")
    Q = _as_matrix(Q, like=_any_entry(A), name="Q")
    n = A.rows
    if A.cols != n or Q.rows != n or Q.cols != n:
        raise ControlsError("lyapunov: A and Q must be square of the same size", code="E-CTRL-001")
    method = _pick_method(method, A, Q)
    if method == "numeric":
        np = _numpy()
        An, Qn = _to_numpy(A, "A"), _to_numpy(Q, "Q")
        Id = np.eye(n)
        K = np.kron(Id, An) + np.kron(An, Id)
        if np.linalg.matrix_rank(K) < n * n:
            raise ControlsError(
                "the Lyapunov operator is singular: A has eigenvalues with λᵢ + λⱼ = 0",
                code="E-CTRL-004",
            )
        x = np.linalg.solve(K, -Qn.reshape(-1, order="F"))
        return x.reshape((n, n), order="F")
    symbolic = bool(_matrix_free(A) or _matrix_free(Q))
    limit = _EXACT_SYMBOLIC_LYAP_MAX if symbolic else _EXACT_RATIONAL_LYAP_MAX
    if n > limit:
        raise ControlsError(
            f"exact Lyapunov solve limited to {limit}×{limit} for "
            f"{'symbolic' if symbolic else 'rational'} data",
            code="E-CTRL-005",
            remediation="substitute numbers and use method='numeric'",
        )
    return _lyap_exact(A, Q)


def _pick_method(method, *Ms):
    if method not in ("auto", "exact", "numeric"):
        raise ControlsError(f"unknown method {method!r}", code="E-CTRL-001")
    if method == "auto":
        return "numeric" if any(_matrix_has_float(M) for M in Ms) else "exact"
    return method


def _lyap_exact(A, Q):
    ak = _ak()
    n = A.rows
    L = _any_entry(A)
    Id = _eye(L, n)
    K = _msimp(kronecker(Id, A) + kronecker(A, Id))
    try:
        Kinv = K.inverse()
    except Exception as exc:
        code = getattr(exc, "code", "")
        if code == "E-MAT-004":
            raise ControlsError(
                "could not decide whether the Lyapunov operator I⊗A + A⊗I is singular",
                code="E-CTRL-005",
                remediation="substitute values for the parameters",
            ) from exc
        raise ControlsError(
            "the Lyapunov operator is singular: A has eigenvalues with λᵢ + λⱼ = 0, "
            "so A X + X Aᵀ + Q = 0 has no unique solution",
            code="E-CTRL-004",
        ) from exc
    for c in ak.matrix_inverse_side_conditions():
        _note(str(c))
    q = ak.Matrix([[_simp(-x)] for x in _vec(Q)])
    x = Kinv @ q
    X = _unvec([_canon(x.get(i, 0)) for i in range(n * n)], n, n)
    R = A @ X + X @ A.transpose() + Q
    if not _is_zero_matrix(_mcancel(R)):
        raise ControlsError(
            "internal check failed: the computed X does not satisfy A X + X Aᵀ + Q = 0",
            code="E-CTRL-005",
            remediation="use method='numeric' with numeric data",
        )
    return X


# ---------------------------------------------------------------------------
# Riccati / LQR
# ---------------------------------------------------------------------------


def care(A, B, Q, R, method="auto", sample=None):
    """Stabilising solution of the continuous algebraic Riccati equation.

    ``Aᵀ X + X A − X B R⁻¹ Bᵀ X + Q = 0`` (scipy's ``solve_continuous_are``
    convention). The stabilising solution is the one with
    ``A − B R⁻¹ Bᵀ X`` Hurwitz; it is unique when it exists, and real.

    ``method="exact"`` (n ≤ 2, rational or symbolic data): solves the
    polynomial system for the entries of a symmetric ``X`` with
    :func:`alkahest.solve`, computes the Routh–Hurwitz conditions of each
    candidate's closed loop, and returns the candidate whose conditions hold
    at ``sample`` — a ``{symbol: number}`` point inside the parameter region
    of interest (default: every parameter set to a distinct value in
    ``(1, 2)``). The Hurwitz conditions of the returned branch are **stated**
    in :func:`controls_side_conditions` (e.g. ``"… r … > 0"``): the result is
    the stabilising solution exactly where they hold, whichever sample was
    used. No stabilising candidate refuses with ``E-CTRL-004``.

    ``method="numeric"``: the Hamiltonian ``[[A, −BR⁻¹Bᵀ], [−Q, −Aᵀ]]``'s
    stable invariant subspace (ordered Schur via scipy when available, else
    eigenvectors), ``X = U₂U₁⁻¹``; returns a ``numpy.ndarray``. Eigenvalues on
    the imaginary axis (no stabilising solution) refuse with ``E-CTRL-004``.

    ``"auto"``: exact when every entry is exact and ``n ≤ 2``, otherwise
    numeric (which needs numeric data — ``E-CTRL-003`` if parameters remain).

    Example
    -------
    >>> X = care([[0, 1], [0, 0]], [[0], [1]], [[1, 0], [0, 1]], [[r]])  # double integrator
    >>> controls_side_conditions()   # r ≠ 0 and the closed-loop Hurwitz conditions (r > 0)
    """
    _reset_conditions()
    A = _as_matrix(A, name="A")
    L = _any_entry(A)
    B = _as_matrix(B, like=L, name="B")
    Q = _as_matrix(Q, like=L, name="Q")
    R = _as_matrix(R, like=L, name="R")
    n, m = A.rows, B.cols
    if A.cols != n or B.rows != n or Q.rows != n or Q.cols != n or R.rows != m or R.cols != m:
        raise ControlsError("care: shapes do not conform", code="E-CTRL-001")
    if method not in ("auto", "exact", "numeric"):
        raise ControlsError(f"unknown method {method!r}", code="E-CTRL-001")
    floats = any(_matrix_has_float(M) for M in (A, B, Q, R))
    if method == "auto":
        method = "exact" if (not floats and n <= 2) else "numeric"
    if method == "numeric":
        return _care_numeric(A, B, Q, R)
    if n > 2:
        raise ControlsError(
            "exact CARE is limited to n ≤ 2",
            code="E-CTRL-005",
            remediation="substitute numbers and use method='numeric'",
        )
    return _care_exact(A, B, Q, R, sample)


def _care_numeric(A, B, Q, R):
    np = _numpy()
    An, Bn, Qn, Rn = (_to_numpy(M, nm) for M, nm in ((A, "A"), (B, "B"), (Q, "Q"), (R, "R")))
    n = An.shape[0]
    G = Bn @ np.linalg.solve(Rn, Bn.T)
    H = np.block([[An, -G], [-Qn, -An.T]])
    ev = np.linalg.eigvals(H)
    scale = max(1.0, float(np.max(np.abs(ev))))
    if np.any(np.abs(ev.real) <= 1e-10 * scale):
        raise ControlsError(
            "the Hamiltonian has eigenvalues on the imaginary axis: no stabilising solution "
            "((A, B) not stabilisable or (Q, A) has unobservable modes on the axis)",
            code="E-CTRL-004",
        )
    try:
        import scipy.linalg as sl

        _, Z, _ = sl.schur(H, sort="lhp")
        U = Z[:, :n]
    except ImportError:
        w, V = np.linalg.eig(H)
        U = V[:, w.real < 0]
    if U.shape[1] != n:
        raise ControlsError(
            "no stabilising solution (stable subspace has wrong dimension)", code="E-CTRL-004"
        )
    U1, U2 = U[:n, :], U[n:, :]
    if np.linalg.cond(U1) > 1e12:
        raise ControlsError("no stabilising solution (U₁ is singular)", code="E-CTRL-004")
    X = np.real(U2 @ np.linalg.inv(U1))
    X = (X + X.T) / 2
    res = An.T @ X + X @ An - X @ G @ X + Qn
    if np.max(np.abs(res)) > 1e-8 * max(1.0, np.max(np.abs(X))):
        raise ControlsError(
            "numeric Riccati residual too large; the problem is ill-conditioned", code="E-CTRL-005"
        )
    return X


def _care_exact(A, B, Q, R, sample):
    ak = _ak()
    L = _any_entry(A)
    pool = L._pool
    n = A.rows
    try:
        Rinv = R.inverse()
    except Exception as exc:
        raise ControlsError("R is singular", code="E-CTRL-001") from exc
    for c in ak.matrix_inverse_side_conditions():
        _note(str(c))
    G = _mcancel(B @ Rinv @ B.transpose())
    xs = {}
    unknowns = []
    for i in range(n):
        for j in range(i, n):
            v = pool.symbol(f"_care_x{i}{j}")
            xs[(i, j)] = xs[(j, i)] = v
            unknowns.append(v)
    X = ak.Matrix([[xs[(i, j)] for j in range(n)] for i in range(n)])
    Ric = A.transpose() @ X + X @ A - X @ G @ X + Q
    eqs = []
    for i in range(n):
        for j in range(i, n):
            num, _ = _num_den(Ric.get(i, j))
            if not _is_zero(num):
                eqs.append(num)
    try:
        sols = ak.solve(eqs, unknowns)
    except Exception as exc:
        raise ControlsError(
            f"could not solve the Riccati system exactly: {exc}",
            code="E-CTRL-005",
            remediation="substitute numbers and use method='numeric'",
        ) from exc
    if not isinstance(sols, list) or not sols:
        raise ControlsError("the Riccati equation has no solution", code="E-CTRL-004")
    for c in ak.solve_side_conditions():
        _note(str(c))
    params = []
    for M in (A, B, Q, R):
        for v in _matrix_free(M):
            if v not in params:
                params.append(v)
    default = {v: 1.0 + (k + 1) / (len(params) + 2) for k, v in enumerate(params)}
    sample = {**default, **(sample or {})}
    bindings = {k: float(v) for k, v in sample.items()}
    s = pool.symbol("_care_s")
    winners = []
    for sol in sols:
        if any(u not in sol for u in unknowns):
            continue  # a positive-dimensional component: not an isolated solution
        Xc = ak.Matrix([[_radsimp(_simp(sol[xs[(i, j)]])) for j in range(n)] for i in range(n)])
        try:
            vals = [[_complex_value(x, bindings) for x in r] for r in Xc.to_list()]
        except Exception:
            continue
        if any(abs(v.imag) > 1e-9 * (1 + abs(v.real)) for r in vals for v in r):
            continue  # not real at the sample
        Ric_c = A.transpose() @ Xc + Xc @ A - Xc @ G @ Xc + Q
        try:
            res = max(abs(_complex_value(x, bindings)) for r in Ric_c.to_list() for x in r)
        except Exception:
            continue
        scale = 1 + max(abs(v) for r in vals for v in r)
        if res > 1e-8 * scale * scale:
            continue  # not a solution at the sample (guards the solver)
        Acl = A - G @ Xc
        chi = _simp(ak.simplify_expanded((_mscale(_eye(L, n), s) - Acl).det()).value)
        coeffs = _poly_coeffs(chi, s)
        conds = _hurwitz_conditions(coeffs)
        try:
            ok = all(_complex_value(c, bindings).real > 1e-12 for c in conds)
        except Exception:
            ok = False
        if ok:
            winners.append((Xc, conds))
    if not winners:
        raise ControlsError(
            "no candidate solution of the Riccati equation is stabilising at the sample point",
            code="E-CTRL-004",
            remediation="check stabilisability of (A, B) and detectability of (Q, A), "
            "or pass sample={param: value} inside the parameter region of interest",
        )
    if len(winners) > 1:
        raise ControlsError(
            "more than one candidate passed the stabilising test at the sample point",
            code="E-CTRL-005",
            remediation="use method='numeric' with numeric data",
        )
    Xc, conds = winners[0]
    for c in conds:
        c = _simp(c)
        if _free_symbols(c):  # constant conditions were decided above
            _note(f"{c} > 0")
    return Xc


def _hurwitz_conditions(coeffs):
    """Routh first-column expressions that must all be ``> 0`` (lowest-first coeffs).

    Divides by the leading coefficient, so the conditions are those of the
    monic polynomial.
    """
    ak = _ak()
    lc = coeffs[-1]
    c = [ak.cancel(x / lc) for x in reversed(coeffs)]  # highest first, c[0] = 1
    n = len(c) - 1
    if n == 0:
        return []
    row0 = c[0::2]
    row1 = c[1::2]
    first = [row1[0]] if row1 else []
    rows = [row0, row1]
    for _ in range(n - 1):
        a, b = rows[-2], rows[-1]
        if not b or _is_zero(b[0]):
            return [*first, b[0] if b else _simp(c[0] * 0)]
        new = []
        for k in range(len(a) - 1):
            bk = b[k + 1] if k + 1 < len(b) else 0
            new.append(ak.cancel((b[0] * a[k + 1] - a[0] * bk) / b[0]))
        if not new:
            break
        first.append(new[0])
        rows.append(new)
    return first


def lqr(A, B, Q, R, method="auto", sample=None):
    """Continuous LQR: ``(K, X)`` with ``K = R⁻¹ Bᵀ X`` and ``X`` from :func:`care`.

    ``u = −K x`` minimises ``∫ xᵀQx + uᵀRu dt``. Exact or numeric as
    :func:`care`; side conditions as :func:`care` states them.
    """
    X = care(A, B, Q, R, method=method, sample=sample)
    ak = _ak()
    if isinstance(X, ak.Matrix):
        A = _as_matrix(A, name="A")
        L = _any_entry(A)
        B = _as_matrix(B, like=L, name="B")
        R = _as_matrix(R, like=L, name="R")
        K = _mcancel(R.inverse() @ B.transpose() @ X)
        K = ak.Matrix([[_radsimp(x) for x in r] for r in K.to_list()])
        return K, X
    np = _numpy()
    Bn = _to_numpy(_as_matrix(B, like=_any_entry(_as_matrix(A)), name="B"))
    Rn = _to_numpy(_as_matrix(R, like=_any_entry(_as_matrix(A)), name="R"))
    return np.linalg.solve(Rn, Bn.T @ X), X


# ---------------------------------------------------------------------------
# Frequency response (numeric)
# ---------------------------------------------------------------------------


def _siso_tf(sys):
    if isinstance(sys, TransferFunction):
        return sys
    if isinstance(sys, StateSpace):
        sys._require_siso("frequency response")
        v = sys._like._pool.symbol("s")
        return sys._to_tf(v)
    raise ControlsError("expected a TransferFunction or StateSpace", code="E-CTRL-001")


def bode(sys, w):
    """Bode data of a SISO system: ``(magnitude_dB, phase_deg)`` at frequencies ``w``.

    Matches :func:`scipy.signal.bode`: magnitude ``20·log₁₀|G(jω)|`` and the
    unwrapped phase in degrees. Numeric: the system must have no free
    parameters (``E-CTRL-003``; use ``G.subs({...})``). Returns numpy arrays
    when numpy is available, else lists.
    """
    _reset_conditions()
    G = _siso_tf(sys)
    resp = G.freqresp(w)
    mag = [20 * math.log10(abs(z)) if z != 0 else -math.inf for z in resp]
    ph = [math.atan2(z.imag, z.real) for z in resp]
    # unwrap like numpy.unwrap
    out = []
    off = 0.0
    for k, p in enumerate(ph):
        if k:
            d = p + off - out[-1]
            while d > math.pi:
                off -= 2 * math.pi
                d -= 2 * math.pi
            while d < -math.pi:
                off += 2 * math.pi
                d += 2 * math.pi
        out.append(p + off)
    phase = [math.degrees(p) for p in out]
    try:
        import numpy as np

        return np.array(mag), np.array(phase)
    except ImportError:  # pragma: no cover
        return mag, phase


class Margins(tuple):
    """``(gain_margin, phase_margin, phase_crossover, gain_crossover)``.

    Same order and conventions as ``control.margin``: ``gain_margin`` is an
    absolute ratio (``inf`` when the Nyquist plot never crosses the negative
    real axis), ``phase_margin`` in degrees (``inf`` when ``|G|`` never
    crosses 1); ``phase_crossover`` is where the gain margin is measured
    (``∠G = −180°``) and ``gain_crossover`` where ``|G(jω)| = 1`` (``nan`` when
    absent). ``gain_margin_db`` is ``20·log₁₀`` of the ratio.
    """

    def __new__(cls, gm, pm, wpc, wgc):
        return super().__new__(cls, (gm, pm, wpc, wgc))

    gain_margin = property(lambda self: self[0])
    phase_margin = property(lambda self: self[1])
    phase_crossover = property(lambda self: self[2])
    gain_crossover = property(lambda self: self[3])

    @property
    def gain_margin_db(self):
        gm = self[0]
        return math.inf if gm == math.inf else 20 * math.log10(gm)

    def __repr__(self):
        return (
            f"Margins(gain_margin={self[0]!r}, phase_margin={self[1]!r}, "
            f"phase_crossover={self[2]!r}, gain_crossover={self[3]!r})"
        )


def margins(sys) -> Margins:
    """Gain and phase margins of a continuous SISO loop transfer function.

    The crossover frequencies are roots of real polynomials in ω:
    ``Im[N(jω)·D(−jω)] = 0`` (phase crossover; kept where ``G(jω) ≤ 0``) and
    ``|N(jω)|² − |D(jω)|² = 0`` (gain crossover). For rational coefficients
    the roots are isolated **exactly** (:func:`alkahest.real_roots`) and
    refined by exact bisection; otherwise numpy's polynomial roots are used.
    The reported margin is the one ``control.margin`` reports: the gain margin
    closest to 1 in log scale and the smallest-magnitude phase margin.

    Refuses discrete-time systems and systems with free parameters
    (``E-CTRL-003``).
    """
    _reset_conditions()
    G = _siso_tf(sys)
    if G.dt is not None:
        raise ControlsError("margins: continuous-time systems only", code="E-CTRL-001")
    for c in G._num_c + G._den_c:
        if _free_symbols(c):
            raise ControlsError(
                "margins needs numeric coefficients",
                code="E-CTRL-003",
                remediation="substitute values first, e.g. G.subs({K: 2})",
            )
    nf = _fraction_coeffs(G._num_c)
    df = _fraction_coeffs(G._den_c)
    exact = nf is not None and df is not None
    if exact:
        num = nf
        den = df
    else:
        num = [_complex_value(c).real for c in G._num_c]
        den = [_complex_value(c).real for c in G._den_c]

    def iw(poly):
        # p(jω) = Σ c_k j^k ω^k  →  (real part coeffs, imag part coeffs) in ω
        re = [0] * len(poly)
        im = [0] * len(poly)
        for k, c in enumerate(poly):
            r = k % 4
            if r == 0:
                re[k] = c
            elif r == 1:
                im[k] = c
            elif r == 2:
                re[k] = -c
            else:
                im[k] = -c
        return re, im

    nr, ni = iw(num)
    dr, di = iw(den)
    # Im(N/D) ∝ Im(N · conj D) = ni·dr − nr·di
    im_poly = _ptrim(_psub(_pmul(ni, dr), _pmul(nr, di)))
    mag_poly = _ptrim(
        _psub(
            [a + b for a, b in _zip_pad(_pmul(nr, nr), _pmul(ni, ni))],
            [a + b for a, b in _zip_pad(_pmul(dr, dr), _pmul(di, di))],
        )
    )
    w180 = [w for w in _real_nonneg_roots(im_poly, exact) if w >= 0]
    wc = [w for w in _real_nonneg_roots(mag_poly, exact) if w > 0]
    resp180 = G.freqresp(w180) if w180 else []
    keep = [(w, r) for w, r in zip(w180, resp180) if r.real <= 0 and abs(r) != math.inf]
    keep.sort()
    gms = [(1.0 / abs(r) if r != 0 else math.inf, w) for w, r in keep]
    wc.sort()
    respc = G.freqresp(wc) if wc else []
    pms = [((math.degrees(cmath.phase(r)) % 360.0) - 180.0, w) for w, r in zip(wc, respc)]
    if gms and not all(g == math.inf for g, _ in gms):
        best = min(abs(math.log(g)) for g, _ in gms if g != math.inf)
        gm, wpc = next((g, w) for g, w in gms if g != math.inf and abs(math.log(g)) == best)
    else:
        gm, wpc = math.inf, math.nan
    if pms:
        best = min(abs(p) for p, _ in pms)
        pm, wgc = next((p, w) for p, w in pms if abs(p) == best)
    else:
        pm, wgc = math.inf, math.nan
    return Margins(gm, pm, wpc, wgc)


def _zip_pad(a, b):
    n = max(len(a), len(b))
    return [((a[i] if i < len(a) else 0), (b[i] if i < len(b) else 0)) for i in range(n)]


def _real_nonneg_roots(poly, exact):
    poly = _ptrim(poly)
    if len(poly) <= 1:
        return []
    if exact:
        return _exact_real_roots(poly)
    np = _numpy()
    rts = np.roots(list(reversed([float(c) for c in poly])))
    scale = max(1.0, float(np.max(np.abs(rts)))) if len(rts) else 1.0
    return sorted(float(r.real) for r in rts if abs(r.imag) <= 1e-9 * scale and r.real >= -1e-12)


def _exact_real_roots(poly):
    """Non-negative real roots of a ℚ-polynomial, isolated exactly, as floats."""
    out = []
    # strip the root at 0
    k = 0
    while k < len(poly) and poly[k] == 0:
        k += 1
    if k:
        out.append(0.0)
        poly = poly[k:]
    if len(poly) <= 1:
        return out
    for f, _mult in _squarefree(poly):
        out.extend(_isolate_positive(f))
    return sorted(out)


def _isolate_positive(f):
    """Positive roots of a square-free ℚ-polynomial by Sturm sequences + bisection."""
    f = [Fraction(x) for x in f]
    seq = [f, _pdiff(f)]
    while len(_ptrim(seq[-1])) > 1 or _ptrim(seq[-1])[0] != 0:
        _, r = _pdivmod(seq[-2], seq[-1])
        r = [-x for x in r]
        if len(_ptrim(r)) == 1 and _ptrim(r)[0] == 0:
            break
        seq.append(_ptrim(r))

    def sign_changes(x):
        vals = []
        for p in seq:
            v = _horner(p, x)
            if v != 0:
                vals.append(v > 0)
        return sum(1 for a, b in zip(vals, vals[1:]) if a != b)

    # Cauchy bound
    lc = abs(f[-1])
    bound = 1 + max((abs(c) / lc for c in f[:-1]), default=Fraction(0))
    roots = []

    def count(a, b):
        return sign_changes(a) - sign_changes(b)

    def solve_interval(a, b):
        c = count(a, b)
        if c == 0:
            return
        if c == 1:
            lo, hi = a, b
            flo = _horner(f, lo)
            for _ in range(200):
                if hi - lo <= abs(hi) * Fraction(1, 10**17) + Fraction(1, 10**300):
                    break
                mid = (lo + hi) / 2
                fm = _horner(f, mid)
                if fm == 0:
                    lo = hi = mid
                    break
                if (fm > 0) == (flo > 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            roots.append(float((lo + hi) / 2))
            return
        mid = (a + b) / 2
        if _horner(f, mid) == 0:
            roots.append(float(mid))
            solve_interval(a, mid - (b - a) / 10**30)
            solve_interval(mid + (b - a) / 10**30, b)
            return
        solve_interval(a, mid)
        solve_interval(mid, b)

    eps = Fraction(0)
    if _horner(f, eps) == 0:
        eps = bound / 10**40
    solve_interval(eps, bound)
    return roots
