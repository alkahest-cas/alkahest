"""Spectral collocation differentiation matrices (Chebyshev and Fourier).

Exact mode returns an :class:`alkahest.Matrix` whose entries are exact
expressions in ``cos(jπ/N)`` (simplified to rationals or radicals where
alkahest knows the value, e.g. ``cos(π/3) = 1/2``, ``cos(π/4) = √2/2``);
numeric mode returns NumPy arrays built with the standard formulas of
Trefethen, *Spectral Methods in MATLAB* (``cheb.m``, ``fourdif``).
"""

from __future__ import annotations

from fractions import Fraction

from ...exceptions import NumericPdeError
from ._exact import frac_to_expr

__all__ = ["cheb_diff_matrix", "cheb_nodes", "fourier_diff_matrix", "fourier_nodes"]


def _pool(pool):
    import alkahest

    if pool is not None:
        return pool
    p = alkahest.active_pool()
    return p if p is not None else alkahest.default_pool()


def _simp(e):
    import alkahest

    try:
        return alkahest.simplify(e).value
    except Exception:  # pragma: no cover - simplify never refuses plain trig
        return e


def _check_n(N, minimum):
    if not isinstance(N, int) or isinstance(N, bool) or minimum > N:
        raise NumericPdeError(f"N must be an integer >= {minimum}, got {N!r}", code="E-PDE-357")


class _Quad:
    """Exact element ``a + b√d`` of a real quadratic field (d ∈ {2, 3, 5})."""

    __slots__ = ("a", "b", "d")

    def __init__(self, a, b=0, d=0):
        self.a, self.b, self.d = Fraction(a), Fraction(b), d if b else 0

    @staticmethod
    def lift(x):
        return x if isinstance(x, _Quad) else _Quad(x)

    def _field(self, o):
        if self.d and o.d and self.d != o.d:
            return None
        return self.d or o.d

    def __add__(self, o):
        o = _Quad.lift(o)
        d = self._field(o)
        if d is None:
            return None
        return _Quad(self.a + o.a, self.b + o.b, d)

    def __neg__(self):
        return _Quad(-self.a, -self.b, self.d)

    def __sub__(self, o):
        o = _Quad.lift(o)
        return self + (-o)

    def __rsub__(self, o):
        return _Quad.lift(o) - self

    def __mul__(self, o):
        o = _Quad.lift(o)
        d = self._field(o)
        if d is None:
            return None
        return _Quad(self.a * o.a + self.b * o.b * d, self.a * o.b + self.b * o.a, d)

    __rmul__ = __mul__

    def inv(self):
        n = self.a * self.a - self.b * self.b * self.d
        return _Quad(self.a / n, -self.b / n, self.d)

    def __truediv__(self, o):
        o = _Quad.lift(o)
        if self._field(o) is None:
            return None
        return self * o.inv()

    def __rtruediv__(self, o):
        return _Quad.lift(o) / self

    def is_zero(self):
        return self.a == 0 and self.b == 0

    def to_expr(self, pool):
        import alkahest

        if not self.b:
            return frac_to_expr(pool, self.a)
        rt = alkahest.sqrt(pool.integer(self.d))
        tb = rt if self.b == 1 else frac_to_expr(pool, self.b) * rt
        if not self.a:
            return tb
        return pool.add([frac_to_expr(pool, self.a), tb])


_COS_TABLE = {
    Fraction(0): _Quad(1),
    Fraction(1, 3): _Quad(Fraction(1, 2)),
    Fraction(1, 2): _Quad(0),
    Fraction(2, 3): _Quad(Fraction(-1, 2)),
    Fraction(1): _Quad(-1),
    Fraction(1, 4): _Quad(0, Fraction(1, 2), 2),
    Fraction(3, 4): _Quad(0, Fraction(-1, 2), 2),
    Fraction(1, 6): _Quad(0, Fraction(1, 2), 3),
    Fraction(5, 6): _Quad(0, Fraction(-1, 2), 3),
    Fraction(1, 5): _Quad(Fraction(1, 4), Fraction(1, 4), 5),
    Fraction(2, 5): _Quad(Fraction(-1, 4), Fraction(1, 4), 5),
    Fraction(3, 5): _Quad(Fraction(1, 4), Fraction(-1, 4), 5),
    Fraction(4, 5): _Quad(Fraction(-1, 4), Fraction(-1, 4), 5),
}


def _cos_pi_rational(pool, q):
    """``cos(q π)`` for rational q: an exact :class:`_Quad` when the value lies
    in Q, Q(√2), Q(√3) or Q(√5), else an ``Expr``."""
    import alkahest

    q = Fraction(q) % 2
    r = 2 - q if q > 1 else q
    if r in _COS_TABLE:
        return _COS_TABLE[r]
    return _simp(alkahest.cos(pool.pi() * frac_to_expr(pool, q)))


def _sin_pi_rational(pool, q):
    import alkahest

    r = (Fraction(1, 2) - Fraction(q)) % 2
    r = 2 - r if r > 1 else r
    if r in _COS_TABLE:
        return _COS_TABLE[r]
    return _simp(alkahest.sin(pool.pi() * frac_to_expr(pool, Fraction(q))))


def _e(pool, v):
    if isinstance(v, _Quad):
        return v.to_expr(pool)
    return frac_to_expr(pool, v) if isinstance(v, Fraction | int) else v


def _exact_or_expr(pool, fn, *args):
    """Evaluate ``fn`` on _Quad arguments when all are exact and share a
    field; else on Expr arguments (then simplified)."""
    if all(isinstance(a, _Quad | Fraction | int) for a in args):
        try:
            v = fn(*[_Quad.lift(a) for a in args])
        except (TypeError, AttributeError):
            v = None
        if v is not None:
            return v
    return _simp(fn(*[_e(pool, a) for a in args]))


def cheb_nodes(N, numeric=False, pool=None):
    """Chebyshev–Gauss–Lobatto points ``x_j = cos(jπ/N)``, ``j = 0..N``."""
    _check_n(N, 1)
    if numeric:
        import numpy as np

        return np.cos(np.pi * np.arange(N + 1) / N)
    p = _pool(pool)
    return [_e(p, _cos_pi_rational(p, Fraction(j, N))) for j in range(N + 1)]


def cheb_diff_matrix(N, numeric=False, pool=None):
    """Chebyshev collocation differentiation matrix on ``N + 1`` Gauss–Lobatto
    points ``x_j = cos(jπ/N)``.

    Returns ``(D, x)``. With ``numeric=False`` (default) ``D`` is an exact
    :class:`alkahest.Matrix` and ``x`` a list of exact ``Expr`` nodes; with
    ``numeric=True`` both are NumPy arrays, the diagonal computed by the
    negative-sum trick for accuracy. ``D @ f(x)`` is exact for polynomials of
    degree ≤ N.
    """
    _check_n(N, 1)
    if numeric:
        import numpy as np

        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
        X = np.tile(x, (N + 1, 1)).T
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
        D = D - np.diag(D.sum(axis=1))
        return D, x
    import alkahest

    p = _pool(pool)
    xs = [_cos_pi_rational(p, Fraction(j, N)) for j in range(N + 1)]
    c = [2 if j in (0, N) else 1 for j in range(N + 1)]
    rows = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            if i == j:
                if i == 0:
                    v = Fraction(2 * N * N + 1, 6)
                elif i == N:
                    v = -Fraction(2 * N * N + 1, 6)
                else:
                    v = _exact_or_expr(p, lambda x: -x / (2 * (1 - x * x)), xs[i])
            else:
                sgn = -1 if (i + j) % 2 else 1
                ratio = Fraction(c[i], c[j]) * sgn
                v = _exact_or_expr(p, lambda a, b, _r=ratio: _r / (a - b), xs[i], xs[j])
            row.append(_e(p, v))
        rows.append(row)
    return alkahest.Matrix(rows), [_e(p, v) for v in xs]


def fourier_nodes(N, numeric=False, pool=None):
    """Equispaced periodic nodes ``x_j = 2πj/N``, ``j = 0..N-1``."""
    _check_n(N, 2)
    if numeric:
        import numpy as np

        return 2 * np.pi * np.arange(N) / N
    p = _pool(pool)
    return [p.pi() * frac_to_expr(p, Fraction(2 * j, N)) if j else p.integer(0) for j in range(N)]


def _fourier_entry(p, N, k, order):
    """Entry ``D[i, j]`` with ``k = i - j`` (mod N) of the order-1 or order-2
    Fourier matrix, exact."""
    k %= N
    sgn = -1 if k % 2 else 1
    if k == 0:
        if order == 1:
            return Fraction(0)
        if N % 2 == 0:
            return Fraction(-(N * N), 12) - Fraction(1, 6)
        return Fraction(-(N * N), 12) + Fraction(1, 12)
    q = Fraction(k, N)  # (k h / 2) / π  with h = 2π/N
    s = _sin_pi_rational(p, q)
    c = _cos_pi_rational(p, q)
    half = Fraction(sgn, 2)
    if order == 1:
        if N % 2 == 0:  # ½ (-1)^k cot(kh/2)
            return _exact_or_expr(p, lambda cc, ss: half * cc / ss, c, s)
        return _exact_or_expr(p, lambda ss: half / ss, s)  # ½ (-1)^k csc(kh/2)
    if N % 2 == 0:  # -½ (-1)^k csc²(kh/2)
        return _exact_or_expr(p, lambda ss: -half / (ss * ss), s)
    # -½ (-1)^k csc(kh/2) cot(kh/2)
    return _exact_or_expr(p, lambda cc, ss: -half * cc / (ss * ss), c, s)


def fourier_diff_matrix(N, order=1, numeric=False, pool=None):
    """Fourier (periodic) collocation differentiation matrix on ``N``
    equispaced points ``x_j = 2πj/N`` of ``[0, 2π)``.

    ``order`` is 1 or 2. Returns ``(D, x)`` — an exact
    :class:`alkahest.Matrix` (entries ``½(−1)^k cot(kπ/N)`` for even ``N``,
    ``½(−1)^k csc(kπ/N)`` for odd ``N``, ``k = i − j``) or, with
    ``numeric=True``, NumPy arrays. ``D`` differentiates trigonometric
    interpolants exactly: ``D @ f(x) = f'(x)`` for trigonometric polynomials
    of degree < N/2.
    """
    _check_n(N, 2)
    if order not in (1, 2):
        raise NumericPdeError(
            "fourier_diff_matrix supports order 1 and 2 (use powers of D for higher odd N orders)",
            code="E-PDE-357",
        )
    if numeric:
        import numpy as np

        k = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
        sgn = np.where(k % 2 == 1, -1.0, 1.0)
        half = np.pi * k / N
        with np.errstate(divide="ignore", invalid="ignore"):
            if order == 1:
                D = 0.5 * sgn / np.tan(half) if N % 2 == 0 else 0.5 * sgn / np.sin(half)
                D[k == 0] = 0.0
            else:
                if N % 2 == 0:
                    D = -0.5 * sgn / np.sin(half) ** 2
                    diag = -(N**2) / 12.0 - 1.0 / 6.0
                else:
                    D = -0.5 * sgn * np.cos(half) / np.sin(half) ** 2
                    diag = -(N**2) / 12.0 + 1.0 / 12.0
                D[k == 0] = diag
        return D, 2 * np.pi * np.arange(N) / N
    import alkahest

    p = _pool(pool)
    rows = [[_e(p, _fourier_entry(p, N, i - j, order)) for j in range(N)] for i in range(N)]
    return alkahest.Matrix(rows), fourier_nodes(N, pool=p)
