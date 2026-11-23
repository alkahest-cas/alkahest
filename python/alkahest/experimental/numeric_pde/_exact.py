"""Exact polynomial arithmetic over Q used by the numeric-PDE analysis layer.

Everything here works with :class:`fractions.Fraction` coefficients and is
exact: no floating-point value ever decides a sign. The pieces are

* :class:`UPoly` — dense univariate polynomials (Sturm sequences, real-root
  isolation, the sign of a polynomial at a real algebraic number);
* :class:`MPoly` — sparse multivariate *Laurent* polynomials keyed by
  exponent tuples (negative exponents are allowed, so ``1/dt`` is a monomial);
* subresultant principal coefficients of bivariate polynomials, the
  projection operator of a one-parameter cylindrical algebraic decomposition;
* conversion between :class:`alkahest.Expr` and :class:`MPoly`.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any
from math import gcd

from ...exceptions import NumericPdeError

# ---------------------------------------------------------------------------
# Univariate polynomials
# ---------------------------------------------------------------------------


class UPoly:
    """Dense univariate polynomial over Q, coefficients low → high."""

    __slots__ = ("c",)

    def __init__(self, coeffs=()):
        c = [Fraction(x) for x in coeffs]
        while c and c[-1] == 0:
            c.pop()
        self.c = c

    # -- basics -------------------------------------------------------------
    @staticmethod
    def const(a):
        return UPoly([a])

    @staticmethod
    def x():
        return UPoly([0, 1])

    def deg(self):
        return len(self.c) - 1

    def is_zero(self):
        return not self.c

    def lc(self):
        return self.c[-1] if self.c else Fraction(0)

    def __eq__(self, other):
        return isinstance(other, UPoly) and self.c == other.c

    def __hash__(self):
        return hash(tuple(self.c))

    def __repr__(self):
        return f"UPoly({[str(x) for x in self.c]})"

    def __add__(self, o):
        a, b = self.c, o.c
        n = max(len(a), len(b))
        return UPoly([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])

    def __neg__(self):
        return UPoly([-x for x in self.c])

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if isinstance(o, (int, Fraction)):
            return UPoly([x * o for x in self.c])
        a, b = self.c, o.c
        if not a or not b:
            return UPoly()
        out = [Fraction(0)] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    out[i + j] += x * y
        return UPoly(out)

    __rmul__ = __mul__

    def __call__(self, x):
        acc = Fraction(0)
        for coef in reversed(self.c):
            acc = acc * x + coef
        return acc

    def deriv(self):
        return UPoly([i * self.c[i] for i in range(1, len(self.c))])

    def divmod(self, o):
        if o.is_zero():
            raise ZeroDivisionError("polynomial division by zero")
        r = list(self.c)
        q = [Fraction(0)] * max(0, len(r) - len(o.c) + 1)
        dl = o.c[-1]
        while len(r) >= len(o.c) and r:
            k = len(r) - len(o.c)
            f = r[-1] / dl
            q[k] = f
            for i, y in enumerate(o.c):
                r[i + k] -= f * y
            r.pop()
            while r and r[-1] == 0:
                r.pop()
        return UPoly(q), UPoly(r)

    def exquo(self, o):
        q, r = self.divmod(o)
        if not r.is_zero():
            raise ArithmeticError("inexact polynomial division")
        return q

    def monic(self):
        if not self.c:
            return self
        lc = self.c[-1]
        return UPoly([x / lc for x in self.c])

    def gcd(self, o):
        a, b = self, o
        while not b.is_zero():
            a, b = b, a.divmod(b)[1]
        return a.monic()

    def squarefree(self):
        if self.deg() <= 0:
            return UPoly([1]) if not self.is_zero() else UPoly()
        g = self.gcd(self.deriv())
        return self.exquo(g).monic()

    def primitive_int(self):
        """Scale to a primitive polynomial with integer coefficients."""
        if not self.c:
            return []
        den = 1
        for x in self.c:
            den = den * x.denominator // gcd(den, x.denominator)
        ints = [int(x * den) for x in self.c]
        g = 0
        for v in ints:
            g = gcd(g, v)
        return [v // g for v in ints]

    # -- real roots -----------------------------------------------------------
    def sturm(self):
        seq = [self, self.deriv()]
        while not seq[-1].is_zero():
            r = seq[-2].divmod(seq[-1])[1]
            seq.append(-r)
        seq.pop()
        return seq

    def root_bound(self):
        """A power of two strictly larger than |root| for every root."""
        lc = abs(self.c[-1])
        m = max((abs(x) for x in self.c[:-1]), default=Fraction(0))
        b = 1 + m / lc
        p = Fraction(1)
        while p <= b:
            p *= 2
        return p


def _sign(x):
    return (x > 0) - (x < 0)


def _variations(seq, x):
    signs = [_sign(p(x)) for p in seq]
    signs = [s for s in signs if s]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _divisors(n, cap=200000):
    n = abs(n)
    if n == 0:
        return [0]
    out = []
    i = 1
    while i * i <= n:
        if i > cap:
            return None
        if n % i == 0:
            out.append(i)
            if i * i != n:
                out.append(n // i)
        i += 1
    return out


def rational_roots(p):
    """All rational roots of ``p`` (distinct), or ``None`` if too costly."""
    if p.deg() <= 0:
        return []
    ints = p.primitive_int()
    roots = set()
    # strip x^k factor
    k = 0
    while ints and ints[0] == 0:
        ints = ints[1:]
        k += 1
    if k:
        roots.add(Fraction(0))
    if len(ints) <= 1:
        return sorted(roots)
    if len(ints) == 2:
        roots.add(Fraction(-ints[0], ints[1]))
        return sorted(roots)
    dp = _divisors(ints[0])
    dq = _divisors(ints[-1])
    if dp is None or dq is None or len(dp) * len(dq) > 20000:
        return None
    q = UPoly(ints)
    for a in dp:
        for b in dq:
            for s in (1, -1):
                cand = Fraction(s * a, b)
                if cand not in roots and q(cand) == 0:
                    roots.add(cand)
    return sorted(roots)


class RealAlg:
    """A real algebraic number: either an exact rational, or the unique root
    of the squarefree polynomial ``poly`` in the open interval ``(lo, hi)``
    with ``poly(lo) * poly(hi) < 0``."""

    __slots__ = ("hi", "lo", "poly", "value")
    hi: Any
    lo: Any
    poly: Any
    value: Any

    def __init__(self, poly=None, lo=None, hi=None, value=None):
        self.poly, self.lo, self.hi, self.value = poly, lo, hi, value

    @property
    def is_rational(self):
        return self.value is not None

    def refine(self):
        if self.is_rational:
            return
        m = (self.lo + self.hi) / 2
        pm = self.poly(m)
        if pm == 0:
            self.value = m
            return
        if _sign(pm) == _sign(self.poly(self.lo)):
            self.lo = m
        else:
            self.hi = m

    def approx(self, width=Fraction(1, 2**60)):
        while not self.is_rational and self.hi - self.lo > width:
            self.refine()
        return self.value if self.is_rational else (self.lo + self.hi) / 2

    def __float__(self):
        return float(self.approx())

    def __repr__(self):
        if self.is_rational:
            return f"RealAlg({self.value})"
        return f"RealAlg(root of {self.poly} in ({float(self.lo)}, {float(self.hi)}))"


def isolate_real_roots(p):
    """Isolate the distinct real roots of ``p`` (non-zero) in increasing order.

    Rational roots are returned exactly; irrational ones as isolating
    intervals of the squarefree part.
    """
    if p.is_zero():
        raise ValueError("zero polynomial has no isolated roots")
    sf = p.squarefree()
    if sf.deg() <= 0:
        return []
    rats = rational_roots(sf) or []
    rest = sf
    for r in rats:
        rest = rest.exquo(UPoly([-r, 1]))
    out = [RealAlg(value=r) for r in rats]
    if rest.deg() >= 1:
        seq = rest.sturm()
        bnd = rest.root_bound()
        stack = [(-bnd, bnd)]
        while stack:
            lo, hi = stack.pop()
            n = _variations(seq, lo) - _variations(seq, hi)  # roots in (lo, hi]
            if n == 0:
                continue
            if rest(hi) == 0:  # a rational root the candidate search skipped
                out.append(RealAlg(value=hi))
                n -= 1
                if n == 0:
                    continue
                m = (lo + hi) / 2
                stack.append((lo, m))
                stack.append((m, hi))
                continue
            if n == 1 and _sign(rest(lo)) * _sign(rest(hi)) < 0:
                out.append(RealAlg(rest, lo, hi))
                continue
            m = (lo + hi) / 2
            stack.append((lo, m))
            stack.append((m, hi))
        # drop duplicate exact roots
        seen = set()
        uniq = []
        for a in out:
            if a.is_rational:
                if a.value in seen:
                    continue
                seen.add(a.value)
            uniq.append(a)
        out = uniq

    # sort: separate overlapping rational / interval entries
    def key(a):
        return a.value if a.is_rational else a.lo

    out.sort(key=key)
    # make intervals disjoint from the rational roots and each other
    changed = True
    while changed:
        changed = False
        for i in range(len(out) - 1):
            a, b = out[i], out[i + 1]
            if not _strictly_less(a, b):
                (a if not a.is_rational else b).refine()
                changed = True
        out.sort(key=key)
    return out


def _upper(a):
    return a.value if a.is_rational else a.hi


def _lower(a):
    return a.value if a.is_rational else a.lo


def _strictly_less(a, b):
    """True when the representations of a and b are already separated."""
    if a.is_rational and b.is_rational:
        return a.value < b.value
    return _upper(a) <= _lower(b) and not (a.is_rational and b.is_rational)


def sign_at(f, alpha):
    """Exact sign of the polynomial ``f`` at the real algebraic ``alpha``."""
    if f.is_zero():
        return 0
    if alpha.is_rational:
        return _sign(f(alpha.value))
    g = alpha.poly.gcd(f)
    if g.deg() >= 1 and _sign(g(alpha.lo)) * _sign(g(alpha.hi)) < 0:
        return 0
    fs = f.squarefree()
    seq = fs.sturm()
    while True:
        if alpha.is_rational:
            return _sign(f(alpha.value))
        lo, hi = alpha.lo, alpha.hi
        flo = fs(lo)
        if flo != 0 and fs(hi) != 0 and _variations(seq, lo) - _variations(seq, hi) == 0:
            return _sign(f(lo))
        alpha.refine()


def rational_between(a, b):
    """A rational strictly between the real algebraic numbers a < b."""
    for _ in range(10000):
        lo, hi = _upper(a), _lower(b)
        if lo < hi:
            return _simple_rational(lo, hi)
        if a.is_rational and b.is_rational:
            break
        if not a.is_rational:
            a.refine()
        if not b.is_rational:
            b.refine()
    raise AssertionError("roots not separated")


def _simple_rational(lo, hi):
    """A rational in the open interval (lo, hi) with small denominator."""
    if lo < 0 < hi:
        return Fraction(0)
    d = 1
    while True:
        n = (lo * d).__floor__() + 1
        c = Fraction(n, d)
        if lo < c < hi:
            return c
        d *= 2


# ---------------------------------------------------------------------------
# Multivariate Laurent polynomials
# ---------------------------------------------------------------------------


class MPoly:
    """Sparse multivariate Laurent polynomial over Q.

    ``terms`` maps exponent tuples (one entry per variable in ``vars``) to
    non-zero :class:`Fraction` coefficients. All polynomials combined in one
    computation must share the same ``vars`` tuple.
    """

    __slots__ = ("terms", "vars")

    def __init__(self, vars, terms=None):
        self.vars = tuple(vars)
        t = {}
        if terms:
            for e, c in terms.items():
                c = Fraction(c)
                if c:
                    t[tuple(e)] = c
        self.terms = t

    @classmethod
    def const(cls, vars, c):
        return cls(vars, {(0,) * len(vars): c})

    @classmethod
    def var(cls, vars, name, power=1):
        e = [0] * len(vars)
        e[list(vars).index(name)] = power
        return cls(vars, {tuple(e): 1})

    def is_zero(self):
        return not self.terms

    def is_const(self):
        return all(not any(e) for e in self.terms)

    def const_value(self):
        if not self.terms:
            return Fraction(0)
        if not self.is_const():
            raise ValueError("not a constant")
        return next(iter(self.terms.values()))

    def is_monomial(self):
        return len(self.terms) == 1

    def __eq__(self, o):
        return isinstance(o, MPoly) and self.terms == o.terms

    def __hash__(self):
        return hash(frozenset(self.terms.items()))

    def __repr__(self):
        return f"MPoly({self.vars}, {self.terms})"

    def _lift(self, o):
        if isinstance(o, MPoly):
            return o
        return MPoly.const(self.vars, o)

    def __add__(self, o):
        o = self._lift(o)
        t = dict(self.terms)
        for e, c in o.terms.items():
            v = t.get(e, 0) + c
            if v:
                t[e] = v
            else:
                t.pop(e, None)
        r = MPoly(self.vars)
        r.terms = t
        return r

    __radd__ = __add__

    def __neg__(self):
        r = MPoly(self.vars)
        r.terms = {e: -c for e, c in self.terms.items()}
        return r

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        if not isinstance(o, MPoly):
            o = Fraction(o)
            if not o:
                return MPoly(self.vars)
            r = MPoly(self.vars)
            r.terms = {e: c * o for e, c in self.terms.items()}
            return r
        t = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in o.terms.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                v = t.get(e, 0) + c1 * c2
                if v:
                    t[e] = v
                else:
                    t.pop(e, None)
        r = MPoly(self.vars)
        r.terms = t
        return r

    __rmul__ = __mul__

    def __pow__(self, n):
        if n < 0:
            if not self.is_monomial():
                raise NumericPdeError(
                    "negative power of a non-monomial is not a Laurent polynomial",
                    code="E-PDE-351",
                )
            ((e, c),) = self.terms.items()
            return MPoly(self.vars, {tuple(x * n for x in e): Fraction(1) / c**-n})
        r = MPoly.const(self.vars, 1)
        b = self
        while n:
            if n & 1:
                r = r * b
            b = b * b
            n >>= 1
        return r

    def div_monomial(self, m):
        ((e, c),) = m.terms.items()
        r = MPoly(self.vars)
        r.terms = {tuple(a - b for a, b in zip(k, e)): v / c for k, v in self.terms.items()}
        return r

    def free_vars(self):
        out = set()
        for e in self.terms:
            for i, x in enumerate(e):
                if x:
                    out.add(self.vars[i])
        return out

    def degree_in(self, name):
        i = self.vars.index(name)
        return max((e[i] for e in self.terms), default=0)

    def min_degree_in(self, name):
        i = self.vars.index(name)
        return min((e[i] for e in self.terms), default=0)

    def subs_value(self, name, value):
        """Substitute a rational value for one variable (exponent >= 0 or value != 0)."""
        i = self.vars.index(name)
        value = Fraction(value)
        t = {}
        for e, c in self.terms.items():
            k = e[i]
            if k < 0 and value == 0:
                raise ZeroDivisionError("Laurent term at zero")
            v = c * value**k
            ne = (*e[:i], 0, *e[i + 1 :])
            t[ne] = t.get(ne, 0) + v
        return MPoly(self.vars, t)

    def to_upoly(self, name):
        """As a univariate polynomial in ``name``; all other exponents must be 0."""
        i = self.vars.index(name)
        coeffs = {}
        for e, c in self.terms.items():
            if any(x for j, x in enumerate(e) if j != i):
                raise ValueError("polynomial involves other variables")
            if e[i] < 0:
                raise ValueError("negative exponent")
            coeffs[e[i]] = c
        n = max(coeffs, default=-1)
        return UPoly([coeffs.get(k, 0) for k in range(n + 1)])

    def to_bivariate(self, tname, rname):
        """As a list (indexed by degree in ``tname``) of :class:`UPoly` in ``rname``."""
        ti, ri = self.vars.index(tname), self.vars.index(rname)
        rows = {}
        for e, c in self.terms.items():
            if any(x for j, x in enumerate(e) if j not in (ti, ri)):
                raise ValueError("polynomial involves other variables")
            rows.setdefault(e[ti], {})[e[ri]] = c
        n = max(rows, default=-1)
        out = []
        for k in range(n + 1):
            row = rows.get(k, {})
            m = max(row, default=-1)
            out.append(UPoly([row.get(j, 0) for j in range(m + 1)]))
        return out


# ---------------------------------------------------------------------------
# Bivariate projection: subresultant principal coefficients over Q[r]
# ---------------------------------------------------------------------------


def _bareiss_det(M):
    """Determinant of a square matrix of UPoly entries (fraction-free)."""
    n = len(M)
    if n == 0:
        return UPoly([1])
    A = [row[:] for row in M]
    sign = 1
    prev = UPoly([1])
    for k in range(n - 1):
        if A[k][k].is_zero():
            for i in range(k + 1, n):
                if not A[i][k].is_zero():
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    break
            else:
                return UPoly()
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]).exquo(prev)
        prev = A[k][k]
    d = A[n - 1][n - 1]
    return -d if sign < 0 else d


def psc(P, Q):
    """Principal subresultant coefficients ``psc_0 … psc_{min(m,n)-1}`` of the
    bivariate polynomials P, Q (lists of UPoly by t-degree)."""
    m, n = len(P) - 1, len(Q) - 1
    out = []
    for j in range(min(m, n)):
        size = m + n - 2 * j
        rows = []
        for i in range(n - j):
            row = [UPoly() for _ in range(size)]
            for k in range(m + 1):
                col = i + (m - k)
                if col < size:
                    row[col] = P[k]
            rows.append(row)
        for i in range(m - j):
            row = [UPoly() for _ in range(size)]
            for k in range(n + 1):
                col = i + (n - k)
                if col < size:
                    row[col] = Q[k]
            rows.append(row)
        out.append(_bareiss_det(rows))
    return out


def bderiv(P):
    return [P[k] * k for k in range(1, len(P))]


# ---------------------------------------------------------------------------
# Expr <-> MPoly
# ---------------------------------------------------------------------------


def _num_of(node):
    tag = node[0]
    if tag == "integer":
        return Fraction(int(node[1]))
    if tag == "rational":
        return Fraction(int(node[1]), int(node[2]))
    return None


def expr_free_symbols(expr):
    """Names of the free symbols of ``expr`` (excluding ``I`` and ``pi``)."""
    out = {}

    def walk(e):
        n = e.node()
        tag = n[0]
        if tag == "symbol":
            out[n[1]] = e
        elif tag in ("add", "mul"):
            for c in n[1]:
                walk(c)
        elif tag == "pow":
            walk(n[1])
            walk(n[2])
        elif tag == "func":
            for c in n[2]:
                if hasattr(c, "node"):
                    walk(c)

    walk(expr)
    return out


def expr_to_mpoly(expr, vars):
    """Convert an alkahest ``Expr`` to an :class:`MPoly` in ``vars`` (names).

    Accepts integers, rationals, symbols listed in ``vars``, sums, products and
    integer powers (negative powers only of monomials). Anything else —
    floats, transcendental functions, unlisted symbols — raises
    ``E-PDE-351``.
    """
    vars = tuple(vars)

    def walk(e):
        n = e.node()
        tag = n[0]
        v = _num_of(n)
        if v is not None:
            return MPoly.const(vars, v)
        if tag == "symbol":
            if n[1] not in vars:
                raise NumericPdeError(
                    f"symbol {n[1]!r} is not one of the declared variables {vars}",
                    code="E-PDE-351",
                )
            return MPoly.var(vars, n[1])
        if tag == "add":
            acc = MPoly(vars)
            for c in n[1]:
                acc = acc + walk(c)
            return acc
        if tag == "mul":
            acc = MPoly.const(vars, 1)
            for c in n[1]:
                acc = acc * walk(c)
            return acc
        if tag == "pow":
            ex = _num_of(n[2].node())
            if ex is None or ex.denominator != 1:
                raise NumericPdeError(
                    f"non-integer power in {e}; coefficients must be (Laurent) polynomials",
                    code="E-PDE-351",
                )
            return walk(n[1]) ** int(ex)
        if tag == "float":
            raise NumericPdeError(
                f"floating-point coefficient {e}: pass an exact rational "
                "(pool.rational(p, q) or fractions.Fraction) instead",
                code="E-PDE-351",
            )
        raise NumericPdeError(
            f"{e} is not a polynomial in {vars}; scheme coefficients must be "
            "(Laurent) polynomials with rational coefficients",
            code="E-PDE-351",
        )

    return walk(expr)


def frac_to_expr(pool, q):
    q = Fraction(q)
    if q.denominator == 1:
        return pool.integer(q.numerator)
    return pool.rational(q.numerator, q.denominator)


def mpoly_to_expr(pool, p, symbols):
    """Convert back to an Expr; ``symbols`` maps variable name → Expr."""
    if p.is_zero():
        return pool.integer(0)
    terms = []

    def key(item):
        e, _ = item
        return (sum(abs(x) for x in e), tuple(-x for x in e))

    for e, c in sorted(p.terms.items(), key=key):
        factors = []
        if c != 1 or not any(e):
            factors.append(frac_to_expr(pool, c))
        for name, k in zip(p.vars, e):
            if k == 1:
                factors.append(symbols[name])
            elif k:
                factors.append(symbols[name] ** k)
        terms.append(factors[0] if len(factors) == 1 else pool.mul(factors))
    return terms[0] if len(terms) == 1 else pool.add(terms)


def upoly_to_expr(pool, p, var):
    return mpoly_to_expr(
        pool,
        MPoly(("v",), {(i,): c for i, c in enumerate(p.c)}),
        {"v": var},
    )


# ---------------------------------------------------------------------------
# Arithmetic in Q(α) for a real algebraic α, and polynomials over it
# ---------------------------------------------------------------------------


def _interval_eval(p, lo, hi):
    """A rigorous enclosure of ``{p(x) : lo ≤ x ≤ hi}`` (interval Horner)."""
    a = b = Fraction(0)
    for c in reversed(p.c):
        prods = (a * lo, a * hi, b * lo, b * hi)
        a, b = min(prods) + c, max(prods) + c
    return a, b


class AlgField:
    """The field Q(α), α the root of ``alpha.poly`` in its isolating interval.

    ``alpha.poly`` may be reducible; whenever an inversion meets a factor that
    vanishes elsewhere, the modulus is replaced by the cofactor that still
    vanishes at α (dynamic evaluation), so every element stays well defined.
    """

    def __init__(self, alpha):
        if alpha.is_rational:
            raise ValueError("use rational arithmetic for a rational point")
        self.alpha = RealAlg(alpha.poly.monic(), alpha.lo, alpha.hi)

    @property
    def q(self):
        return self.alpha.poly

    def reduce(self, p):
        return p.divmod(self.q)[1] if p.deg() >= self.q.deg() else p

    def sign(self, p):
        return sign_at(p, self.alpha)

    def inverse(self, p):
        while True:
            r0, r1 = self.q, self.reduce(p)
            s0, s1 = UPoly(), UPoly([1])
            while not r1.is_zero():
                qq, rr = r0.divmod(r1)
                r0, r1 = r1, rr
                s0, s1 = s1, s0 - qq * s1
            g = r0
            if g.deg() == 0:
                return self.reduce(s0 * (Fraction(1) / g.c[0]))
            # g(α) ≠ 0 because p(α) ≠ 0: drop the factor g from the modulus
            newq = self.q.exquo(g).monic()
            self.alpha = RealAlg(newq, self.alpha.lo, self.alpha.hi)


class AlgElem:
    """An element of Q(α), stored as a polynomial in α."""

    __slots__ = ("F", "p")

    def __init__(self, F, p):
        self.F = F
        self.p = F.reduce(p)

    def __add__(self, o):
        return AlgElem(self.F, self.p + o.p)

    def __sub__(self, o):
        return AlgElem(self.F, self.p - o.p)

    def __neg__(self):
        return AlgElem(self.F, -self.p)

    def __mul__(self, o):
        if isinstance(o, AlgElem):
            return AlgElem(self.F, self.p * o.p)
        return AlgElem(self.F, self.p * Fraction(o))

    def sign(self):
        return self.F.sign(self.F.reduce(self.p))

    def is_zero(self):
        return self.p.is_zero() or self.sign() == 0

    def inv(self):
        if self.is_zero():
            raise ZeroDivisionError("inverse of zero in Q(alpha)")
        return AlgElem(self.F, self.F.inverse(self.p))

    def enclosure(self):
        return _interval_eval(self.F.reduce(self.p), self.F.alpha.lo, self.F.alpha.hi)


class KPoly:
    """Univariate polynomial with coefficients in Q(α) (``AlgElem``)."""

    __slots__ = ("F", "c")

    def __init__(self, F, coeffs):
        self.F = F
        c = list(coeffs)
        while c and c[-1].is_zero():
            c.pop()
        self.c = c

    def deg(self):
        return len(self.c) - 1

    def is_zero(self):
        return not self.c

    def _z(self):
        return AlgElem(self.F, UPoly())

    def __neg__(self):
        return KPoly(self.F, [-x for x in self.c])

    def __mul__(self, o):
        if not self.c or not o.c:
            return KPoly(self.F, [])
        out = [self._z() for _ in range(len(self.c) + len(o.c) - 1)]
        for i, x in enumerate(self.c):
            for j, y in enumerate(o.c):
                out[i + j] = out[i + j] + x * y
        return KPoly(self.F, out)

    def __call__(self, x):
        acc = self._z()
        for c in reversed(self.c):
            acc = acc * x + c
        return acc

    def deriv(self):
        return KPoly(self.F, [self.c[i] * i for i in range(1, len(self.c))])

    def divmod(self, o):
        r = list(self.c)
        inv = o.c[-1].inv()
        q = [self._z() for _ in range(max(0, len(r) - len(o.c) + 1))]
        while len(r) >= len(o.c) and r:
            k = len(r) - len(o.c)
            f = r[-1] * inv
            q[k] = f
            for i, y in enumerate(o.c):
                r[i + k] = r[i + k] - f * y
            r.pop()
            while r and r[-1].is_zero():
                r.pop()
        return KPoly(self.F, q), KPoly(self.F, r)

    def monic(self):
        inv = self.c[-1].inv()
        return KPoly(self.F, [x * inv for x in self.c])

    def gcd(self, o):
        a, b = self, o
        while not b.is_zero():
            a, b = b, a.divmod(b)[1]
        return a.monic() if not a.is_zero() else a

    def squarefree(self):
        if self.deg() <= 0:
            return self
        g = self.gcd(self.deriv())
        if g.deg() <= 0:
            return self.monic()
        return self.divmod(g)[0].monic()

    def sturm(self):
        seq = [self, self.deriv()]
        while not seq[-1].is_zero():
            seq.append(-(seq[-2].divmod(seq[-1])[1]))
        seq.pop()
        return seq

    def root_bound(self):
        lc = self.c[-1]
        while True:
            a, b = lc.enclosure()
            if a > 0 or b < 0:
                break
            self.F.alpha.refine()
        low = min(abs(a), abs(b))
        m = Fraction(0)
        for c in self.c[:-1]:
            x, y = c.enclosure()
            m = max(m, abs(x), abs(y))
        bnd = 1 + m / low
        p = Fraction(1)
        while p <= bnd:
            p *= 2
        return p


def _kvariations(seq, x):
    signs = [s for s in (p(x).sign() for p in seq) if s]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


class KRoot:
    """A real root of a squarefree KPoly: exact rational, or isolated in (lo, hi)."""

    __slots__ = ("hi", "lo", "poly", "value")
    hi: Any
    lo: Any
    poly: Any
    value: Any

    def __init__(self, poly, lo=None, hi=None, value=None):
        self.poly, self.lo, self.hi, self.value = poly, lo, hi, value

    @property
    def is_rational(self):
        return self.value is not None

    def refine(self):
        if self.is_rational:
            return
        m = (self.lo + self.hi) / 2
        pm = self.poly(m)
        if pm.is_zero():
            self.value = m
            return
        if pm.sign() == self.poly(self.lo).sign():
            self.lo = m
        else:
            self.hi = m


def k_isolate(p):
    """Distinct real roots of the non-zero KPoly ``p``, increasing."""
    sf = p.squarefree()
    if sf.deg() <= 0:
        return []
    seq = sf.sturm()
    bnd = sf.root_bound()
    out = []
    stack = [(-bnd, bnd)]
    while stack:
        lo, hi = stack.pop()
        n = _kvariations(seq, lo) - _kvariations(seq, hi)  # roots in (lo, hi]
        if n and sf(hi).is_zero():
            out.append(KRoot(sf, value=hi))
            n -= 1
            if n == 0:
                continue
            m = (lo + hi) / 2
            stack.append((lo, m))
            stack.append((m, hi))
            continue
        if n == 0:
            continue
        if n == 1 and sf(lo).sign() * sf(hi).sign() < 0:
            out.append(KRoot(sf, lo, hi))
            continue
        m = (lo + hi) / 2
        stack.append((lo, m))
        stack.append((m, hi))
    out.sort(key=lambda r: r.value if r.is_rational else r.lo)
    ded = []
    for r in out:
        if ded and r.is_rational and ded[-1].is_rational and ded[-1].value == r.value:
            continue
        ded.append(r)
    return ded


def k_sign_at(f, root):
    """Exact sign of the KPoly ``f`` at a KRoot."""
    if f.is_zero():
        return 0
    if root.is_rational:
        return f(root.value).sign()
    g = root.poly.gcd(f)
    if g.deg() >= 1 and g(root.lo).sign() * g(root.hi).sign() < 0:
        return 0
    fs = f.squarefree()
    seq = fs.sturm()
    while True:
        if root.is_rational:
            return f(root.value).sign()
        lo, hi = root.lo, root.hi
        if (
            not fs(lo).is_zero()
            and not fs(hi).is_zero()
            and _kvariations(seq, lo) - _kvariations(seq, hi) == 0
        ):
            return f(lo).sign()
        root.refine()
