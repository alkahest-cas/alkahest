"""Exact linear algebra and polynomial helpers for :mod:`alkahest.controls`.

Everything here works on plain Python lists of coefficients / entries and has
no dependency on the kernel: ``Fraction`` data for the rational algorithms
(structured Lyapunov solve, Hamiltonian stable subspace, polynomial
arithmetic in ``ℚ[r]/(f)``), and a tiny *field* protocol for the elimination
routines that also serve symbolic data (Kalman decomposition):

* ``K.zero`` / ``K.one`` — the identities,
* ``K.canon(x)`` — bring an entry to normal form after arithmetic,
* ``K.iszero(x)`` — whether ``x`` is *proven* zero,
* ``K.pivot(x)`` — called with every entry chosen as a pivot (the symbolic
  field records ``x ≠ 0`` as a side condition there).

Polynomials are coefficient lists, **lowest degree first**.
"""

from __future__ import annotations

from fractions import Fraction

from .alkahest import check_budget as _tick  # cooperative Budget checkpoint

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


class _QQ:
    """The rational field."""

    zero = Fraction(0)
    one = Fraction(1)

    @staticmethod
    def canon(x):
        return x

    @staticmethod
    def iszero(x):
        return x == 0

    @staticmethod
    def pivot(x):
        return None


QQ = _QQ()


class Refusal(Exception):
    """Internal: a structured algorithm gave up (the caller maps it to a code)."""

    def __init__(self, kind, message=""):
        super().__init__(message or kind)
        self.kind = kind


# ---------------------------------------------------------------------------
# Dense matrices over a field (lists of rows)
# ---------------------------------------------------------------------------


def zeros(K, r, c):
    return [[K.zero for _ in range(c)] for _ in range(r)]


def eye(K, n):
    return [[K.one if i == j else K.zero for j in range(n)] for i in range(n)]


def transpose(M):
    return [list(col) for col in zip(*M)] if M else []


def madd(K, A, B):
    return [[K.canon(a + b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def msub(K, A, B):
    return [[K.canon(a - b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mscale(K, c, A):
    return [[K.canon(c * a) for a in row] for row in A]


def mmul(K, A, B):
    n, m = len(A), len(B)
    p = len(B[0]) if B else 0
    Bt = transpose(B)
    out = []
    for i in range(n):
        _tick()
        row = []
        Ai = A[i]
        for j in range(p):
            acc = None
            Bj = Bt[j]
            for k in range(m):
                a, b = Ai[k], Bj[k]
                if _triv_zero(a) or _triv_zero(b):
                    continue
                t = a * b
                acc = t if acc is None else acc + t
            row.append(K.zero if acc is None else K.canon(acc))
        out.append(row)
    return out


def _triv_zero(x):
    """Cheap syntactic zero test (``Fraction`` zero; expressions never match)."""
    return isinstance(x, (int, Fraction)) and x == 0


def rref(K, M):
    """Reduced row echelon form and the pivot columns."""
    R = [list(r) for r in M]
    rows = len(R)
    cols = len(R[0]) if rows else 0
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        p = None
        for i in range(r, rows):
            if not K.iszero(R[i][c]):
                p = i
                break
        if p is None:
            # normalise entries proven non-pivots so later tests are cheap
            continue
        _tick()
        R[r], R[p] = R[p], R[r]
        piv = R[r][c]
        K.pivot(piv)
        R[r] = [K.canon(x / piv) for x in R[r]]
        for i in range(rows):
            if i != r and not K.iszero(R[i][c]):
                f = R[i][c]
                R[i] = [K.canon(x - f * y) for x, y in zip(R[i], R[r])]
        pivots.append(c)
        r += 1
    return R, pivots


def inverse(K, A):
    """``A⁻¹`` or ``None`` when ``A`` is (provably) singular."""
    n = len(A)
    aug = [list(A[i]) + eye(K, n)[i] for i in range(n)]
    R, piv = rref(K, aug)
    if piv[:n] != list(range(n)):
        return None
    return [row[n:] for row in R[:n]]


def nullspace(K, M, ncols=None):
    """Basis of ``{v : M v = 0}`` as a list of column vectors."""
    cols = ncols if ncols is not None else (len(M[0]) if M else 0)
    if not M:
        return [[K.one if i == j else K.zero for i in range(cols)] for j in range(cols)]
    R, piv = rref(K, M)
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = [K.zero] * cols
        v[f] = K.one
        for row, pc in enumerate(piv):
            v[pc] = K.canon(-R[row][f])
        basis.append(v)
    return basis


def column_basis(K, M):
    """Independent columns spanning the column space of ``M`` (list of columns)."""
    if not M or not M[0]:
        return []
    _, piv = rref(K, M)
    Mt = transpose(M)
    return [Mt[c] for c in piv]


def complete_basis(K, cols, n):
    """Standard basis vectors completing the independent ``cols`` to a basis of K^n."""
    if not cols:
        return [[K.one if i == j else K.zero for i in range(n)] for j in range(n)]
    Vt = [list(c) for c in cols]  # r × n
    _, piv = rref(K, Vt)
    return [[K.one if i == j else K.zero for i in range(n)] for j in range(n) if j not in piv]


def hstack_cols(cols, n):
    """Matrix (n × len(cols)) whose columns are ``cols``."""
    return [[c[i] for c in cols] for i in range(n)]


def matpoly(K, coeffs, M):
    """``Σ coeffs[k] M^k`` (Horner)."""
    n = len(M)
    acc = mscale(K, coeffs[-1], eye(K, n))
    for c in reversed(coeffs[:-1]):
        acc = madd(K, mmul(K, acc, M), mscale(K, c, eye(K, n)))
    return acc


def charpoly(A):
    """Characteristic polynomial ``det(tI − A)`` over ℚ (Faddeev–LeVerrier), low first."""
    n = len(A)
    c = [Fraction(0)] * (n + 1)
    c[n] = Fraction(1)
    Mk = zeros(QQ, n, n)
    for k in range(1, n + 1):
        Mk = madd(QQ, mmul(QQ, A, Mk), mscale(QQ, c[n - k + 1], eye(QQ, n)))
        AM = mmul(QQ, A, Mk)
        tr = sum(AM[i][i] for i in range(n))
        c[n - k] = -tr / k
    return c


# ---------------------------------------------------------------------------
# Structured Sylvester / Lyapunov solve over ℚ
# ---------------------------------------------------------------------------


def sylvester_solve(A, B, C):
    """Solve ``A X + X B = C`` exactly over ℚ without forming the Kronecker system.

    With ``p`` the characteristic polynomial of ``A`` and the commuting
    operators ``𝓛 X = A X``, ``𝓡 X = X B``: Cayley–Hamilton gives
    ``p(𝓛) = 0``, hence ``(𝓛 + 𝓡) · g = p(−𝓡)`` for the polynomial
    ``g = −Σ_{k} c_k Σ_{i<k} 𝓛^i (−𝓡)^{k−1−i}``. So with
    ``Y = C · p(−B)⁻¹`` the matrix ``X = −g(Y)`` solves the equation. Only
    ``O(n)`` matrix products are needed (against an ``n²×n²`` elimination);
    ``p(−B)`` is invertible exactly when no ``λᵢ(A) + μⱼ(B)`` vanishes.
    Raises :class:`Refusal` ``"singular"`` otherwise.
    """
    n = len(A)
    c = charpoly(A)
    nB = mscale(QQ, Fraction(-1), B)
    pB = matpoly(QQ, c, nB)
    inv = inverse(QQ, pB)
    if inv is None:
        raise Refusal("singular", "the Sylvester operator is singular")
    Y = mmul(QQ, C, inv)
    # P_{n-1} = c_n I;  P_{l-1} = A P_l + c_l I  (descending l), Horner over l.
    P = mscale(QQ, c[n], eye(QQ, n))
    W = mmul(QQ, P, Y)
    for l in range(n - 1, 0, -1):  # noqa: E741
        P = madd(QQ, mmul(QQ, A, P), mscale(QQ, c[l], eye(QQ, n)))  # P_{l-1}
        W = madd(QQ, mmul(QQ, P, Y), mmul(QQ, W, nB))
    return mscale(QQ, Fraction(-1), W)


# ---------------------------------------------------------------------------
# Polynomials over ℚ (low-first lists of Fractions)
# ---------------------------------------------------------------------------


def ptrim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a or [Fraction(0)]


def padd(a, b):
    n = max(len(a), len(b))
    return ptrim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])


def pscale(a, c):
    return ptrim([x * c for x in a])


def pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return ptrim(out)


def pdivmod(a, b):
    a = ptrim([Fraction(x) for x in a])
    b = ptrim([Fraction(x) for x in b])
    if len(b) == 1 and b[0] == 0:
        raise ZeroDivisionError("polynomial division by zero")
    if len(a) < len(b):
        return [Fraction(0)], a
    q = [Fraction(0)] * (len(a) - len(b) + 1)
    r = list(a)
    lb = b[-1]
    for k in range(len(a) - len(b), -1, -1):
        coef = r[k + len(b) - 1] / lb
        q[k] = coef
        if coef:
            for j, bj in enumerate(b):
                r[k + j] -= coef * bj
    return ptrim(q), ptrim(r[: len(b) - 1] or [Fraction(0)])


def pmod(a, m):
    return pdivmod(a, m)[1]


def pegcd(a, b):
    """``(g, s, t)`` with ``s·a + t·b = g`` and ``g`` monic."""
    r0, r1 = ptrim(a), ptrim(b)
    s0, s1 = [Fraction(1)], [Fraction(0)]
    t0, t1 = [Fraction(0)], [Fraction(1)]
    while not (len(r1) == 1 and r1[0] == 0):
        q, r = pdivmod(r0, r1)
        r0, r1 = r1, r
        s0, s1 = s1, padd(s0, pscale(pmul(q, s1), Fraction(-1)))
        t0, t1 = t1, padd(t0, pscale(pmul(q, t1), Fraction(-1)))
    lc = r0[-1]
    return pscale(r0, 1 / lc), pscale(s0, 1 / lc), pscale(t0, 1 / lc)


def pinv_mod(a, m):
    """Inverse of ``a`` modulo ``m`` (they must be coprime)."""
    g, s, _ = pegcd(a, m)
    if len(g) != 1:
        raise Refusal("not-coprime", "polynomial not invertible modulo m")
    return pmod(s, m)


def ppow(a, k):
    out = [Fraction(1)]
    for _ in range(k):
        out = pmul(out, a)
    return out


def pderiv(a):
    return ptrim([k * a[k] for k in range(1, len(a))] or [Fraction(0)])


def pcompose_neg(a):
    """``a(−x)``."""
    return ptrim([c if k % 2 == 0 else -c for k, c in enumerate(a)])


def pdigits(a, f):
    """``f``-adic digits of ``a``: ``a = Σ d_k f^k`` with ``deg d_k < deg f``."""
    out = []
    a = ptrim(a)
    while not (len(a) == 1 and a[0] == 0):
        q, r = pdivmod(a, f)
        out.append(r)
        a = q
    return out or [[Fraction(0)]]


def binom(n, k):
    from math import comb

    return comb(n, k)


def ptaylor(a, shift_var_poly_ring_mod):
    """Coefficients of ``a(x + ε)`` in ``ε``: ``[a^{(i)}(x)/i!]`` as polys in ``x``."""
    out = []
    for i in range(len(a)):
        coeff = [Fraction(0)] * max(1, len(a) - i)
        for k in range(i, len(a)):
            coeff[k - i] += a[k] * binom(k, i)
        out.append(ptrim(coeff))
    return out


# ---------------------------------------------------------------------------
# Hurwitz test (exact)
# ---------------------------------------------------------------------------


def is_hurwitz(p):
    """Whether every root of the ℚ-polynomial ``p`` has negative real part (exact Routh)."""
    p = ptrim(p)
    n = len(p) - 1
    if n == 0:
        return True
    c = [x / p[-1] for x in reversed(p)]  # highest first, c[0] = 1
    if any(x <= 0 for x in c):
        return False
    rows = [c[0::2], c[1::2]]
    for _ in range(n - 1):
        a, b = rows[-2], rows[-1]
        if not b or b[0] <= 0:
            return False
        rows.append(
            [
                (b[0] * a[k + 1] - a[0] * (b[k + 1] if k + 1 < len(b) else 0)) / b[0]
                for k in range(len(a) - 1)
            ]
        )
    return all(r and r[0] > 0 for r in rows)


# ---------------------------------------------------------------------------
# Riccati: exact stable invariant subspace of the Hamiltonian over ℚ
# ---------------------------------------------------------------------------


def care_hamiltonian(A, B, Q, R, factor):
    """Exact stabilising solution of ``AᵀX + XA − X B R⁻¹ Bᵀ X + Q = 0`` over ℚ.

    ``factor(coeffs)`` factors a ℚ-polynomial into ``[(monic_factor, mult)]``.
    The characteristic polynomial of the Hamiltonian ``H = [[A, −G], [−Q, −Aᵀ]]``
    is split into ℚ-irreducible factors; each is classified *exactly* by Routh
    as having all roots in the open left half plane (stable), all in the right
    one, or neither. Raises :class:`Refusal` with ``kind`` ``"mixed"`` when an
    irreducible factor has roots on both sides (the stable subspace is then not
    defined over ℚ), ``"axis"`` for roots on the imaginary axis, ``"singular"``
    when ``U₁`` is singular, ``"R"`` when ``R`` is singular.
    """
    n = len(A)
    Rinv = inverse(QQ, R)
    if Rinv is None:
        raise Refusal("R", "R is singular")
    G = mmul(QQ, mmul(QQ, B, Rinv), transpose(B))
    AT = transpose(A)
    H = [[*A[i], *[-x for x in G[i]]] for i in range(n)] + [
        [*[-x for x in Q[i]], *[-x for x in AT[i]]] for i in range(n)
    ]
    chi = charpoly(H)
    stable = []
    for f, m in factor(chi):
        if len(f) <= 1:
            continue
        if is_hurwitz(f):
            stable.append((f, m))
        elif is_hurwitz(pcompose_neg(f)):
            continue
        else:
            raise Refusal("mixed" if _no_axis_roots(f) else "axis")
    if sum((len(f) - 1) * m for f, m in stable) != n:
        raise Refusal("axis", "the stable subspace has the wrong dimension")
    basis = []
    for f, m in stable:
        Fm = matpoly(QQ, ppow(f, m), H)
        basis.extend(nullspace(QQ, Fm, 2 * n))
    if len(basis) != n:
        raise Refusal("axis", "the stable subspace has the wrong dimension")
    U = hstack_cols(basis, 2 * n)
    U1, U2 = U[:n], U[n:]
    U1i = inverse(QQ, U1)
    if U1i is None:
        raise Refusal("singular", "U1 is singular")
    X = mmul(QQ, U2, U1i)
    # Verify exactly: symmetric and a root of the Riccati equation.
    XT = transpose(X)
    if XT != X:
        raise Refusal("verify", "the Hamiltonian solution is not symmetric")
    res = madd(
        QQ,
        msub(QQ, madd(QQ, mmul(QQ, AT, X), mmul(QQ, X, A)), mmul(QQ, mmul(QQ, X, G), X)),
        Q,
    )
    if any(x != 0 for r in res for x in r):
        raise Refusal("verify", "Riccati residual is not zero")
    return X


def _no_axis_roots(f):
    """Heuristic classification of a mixed factor: no root with real part ~ 0."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return True
    rts = np.roots([float(c) for c in reversed(f)])
    scale = max(1.0, float(np.max(np.abs(rts)))) if len(rts) else 1.0
    return not bool(np.any(np.abs(rts.real) <= 1e-9 * scale))


# ---------------------------------------------------------------------------
# Partial fractions of N / f^m over ℚ[r]/(f) and their Laplace inverse
# ---------------------------------------------------------------------------


def kmul(a, b, f):
    return pmod(pmul(a, b), f)


def series_inverse(g, order, f):
    """Inverse of the power series ``g(ε) = Σ g_i ε^i`` (coefficients in ``ℚ[r]/(f)``)."""
    g0i = pinv_mod(g[0], f)
    out = [g0i]
    for k in range(1, order):
        acc = [Fraction(0)]
        for i in range(1, min(k, len(g) - 1) + 1):
            acc = padd(acc, kmul(g[i], out[k - i], f))
        out.append(pmod(pscale(kmul(acc, g0i, f), Fraction(-1)), f))
    return out


def series_mul(a, b, order, f):
    out = []
    for k in range(order):
        _tick()
        acc = [Fraction(0)]
        for i in range(k + 1):
            if i < len(a) and (k - i) < len(b):
                acc = padd(acc, kmul(a[i], b[k - i], f))
        out.append(pmod(acc, f))
    return out


def residue_coefficients(num, f, m):
    """Exponential-polynomial coefficients of ``L⁻¹{num / f^m}`` at a root ``r`` of ``f``.

    ``f`` is monic irreducible, ``deg num < m·deg f``. Returns a list
    ``c[i]`` (``i = 0 … m−1``), each a polynomial in ``r`` modulo ``f``, such
    that ``L⁻¹{num/f^m}(t) = Σ_{f(r)=0} e^{r t} Σ_i c_i(r) t^i``.
    """
    f = ptrim(f)
    digits = pdigits(num, f)  # digits[k] multiplies f^k
    digits += [[Fraction(0)]] * (m - len(digits))
    fT = ptaylor(f, None)  # f(r+ε) coefficients as polynomials in r
    g = [pmod(fT[i + 1], f) for i in range(len(f) - 1)]  # f(r+ε)/ε
    coeffs = [[Fraction(0)] for _ in range(m)]
    for j in range(1, m + 1):  # term  v_j / f^j,  v_j = digits[m − j]
        v = digits[m - j]
        if len(v) == 1 and v[0] == 0:
            continue
        psi = [pmod(c, f) for c in ptaylor(v, None)]
        ginv = series_inverse(g, j, f)
        gj = [[Fraction(1)]] + [[Fraction(0)]] * (j - 1)
        for _ in range(j):
            gj = series_mul(gj, ginv, j, f)
        prod = series_mul(psi, gj, j, f)
        for i in range(j):  # coefficient of t^i/i! is prod[j − 1 − i]
            fact = 1
            for q in range(2, i + 1):
                fact *= q
            coeffs[i] = padd(coeffs[i], pscale(prod[j - 1 - i], Fraction(1, fact)))
    return [pmod(c, f) for c in coeffs]


def partial_fraction_numerators(num, factors):
    """CRT split of ``num / Π fᵢ^mᵢ`` into ``[(fᵢ, mᵢ, nᵢ)]``, ``deg nᵢ < mᵢ·deg fᵢ``."""
    mods = [ppow(f, m) for f, m in factors]
    total = [Fraction(1)]
    for d in mods:
        total = pmul(total, d)
    out = []
    for (f, m), Di in zip(factors, mods):
        rest, _ = pdivmod(total, Di)
        e = pinv_mod(rest, Di)
        out.append((f, m, pmod(pmul(num, e), Di)))
    return out
