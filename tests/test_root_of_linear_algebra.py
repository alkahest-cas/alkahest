"""``RootOf`` completeness (F7): printed ``RootOf`` reads back, and
``eigenvects`` / ``jordan_form`` / ``matrix_exp`` work on ``RootOf`` spectra.

G1 (#458) introduced ``RootOf(p, k)`` for irreducible factors of degree ≥ 3
without a radical form, but left three gaps:

* ``parse(str(root_of(...)))`` raised ``E-PARSE-003`` (unknown function);
* ``eigenvects``, ``jordan_form`` and ``matrix_exp`` refused any ``RootOf``
  eigenvalue (``E-EIGEN-004`` / ``E-LINALG``), because their elimination
  decides pivots with a zero test that cannot use ``p(α) = 0``.

The linear algebra now runs in the number field ``ℚ(α) = ℚ[x]/(p)`` (FLINT
``nf_elem``), where pivots are exact; each result is a polynomial of degree
``< deg p`` in the ``RootOf`` node, valid for every conjugate.

Oracles: SymPy reduces ``A·v − λ·v`` modulo ``p`` (an exact check in ℚ(α)
that shares no code with Alkahest), NumPy ``eig`` for eigenvalues and
eigenvector directions, SciPy ``expm`` for the matrix exponential.
"""

import re

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")
sympy = pytest.importorskip("sympy")


def _companion(p, coeffs):
    """Companion matrix of the monic ``x^n + c_{n-1} x^{n-1} + … + c_0``
    (``coeffs`` = ``[c_0, …, c_{n-1}]``)."""
    n = len(coeffs)
    rows = [[p.integer(0) for _ in range(n)] for _ in range(n)]
    for i in range(1, n):
        rows[i][i - 1] = p.integer(1)
    for i in range(n):
        rows[i][n - 1] = p.integer(-coeffs[i])
    return rows


def _np_companion(coeffs):
    n = len(coeffs)
    a = np.zeros((n, n))
    for i in range(1, n):
        a[i, i - 1] = 1.0
    for i in range(n):
        a[i, n - 1] = -coeffs[i]
    return a


_ROOT_OF = re.compile(r"RootOf\(([^,]+), (\d+)\)")


def _to_sympy_in_alpha(expr):
    """``(sympy expression in t, p(t))`` for an Alkahest expression that is a
    rational function of exactly one ``RootOf(p, k)``."""
    s = str(expr)
    found = set(_ROOT_OF.findall(s))
    assert len(found) <= 1, s
    t = sympy.Symbol("t")
    if not found:
        return sympy.sympify(s.replace("^", "**")), None
    (poly_s, _k) = next(iter(found))
    body = _ROOT_OF.sub("t", s).replace("^", "**")
    x = sympy.Symbol("x")
    p_t = sympy.Poly(sympy.sympify(poly_s.replace("^", "**")).subs(x, t), t)
    return sympy.sympify(body, locals={"t": t}), p_t


def _is_zero_mod_p(expr):
    """Exact: ``expr`` is zero in ℚ(α) = ℚ[t]/(p)."""
    e, p_t = _to_sympy_in_alpha(expr)
    if p_t is None:
        return sympy.simplify(e) == 0
    t = p_t.gen
    num, den = sympy.fraction(sympy.together(e))
    num = sympy.Poly(sympy.expand(num), t)
    den = sympy.Poly(sympy.expand(den), t)
    assert not den.rem(p_t).is_zero, "division by zero in Q(alpha)"
    return num.rem(p_t).is_zero


def _complex(expr, env=None):
    r = ak.evaluate(expr, env or {}, mode="complex")
    assert r.status == "ok", (str(expr), r.reason)
    return complex(r.value)


# ---------------------------------------------------------------------------
# Parse round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "coeffs",
    [[-1, -1, 0, 1], [1, 1, 0, 0, 1], [3, -3, 0, 0, 0, 2], [-2, 0, 0, 1], [3, 0, 1, 0, 0, 1]],
)
def test_printed_root_of_reads_back_as_itself(coeffs):
    p = ak.ExprPool()
    x = p.symbol("x")
    poly = sum((c * x**k for k, c in enumerate(coeffs) if c), p.integer(0))
    for k in range(len(coeffs) - 1):
        r = ak.root_of(poly, x, k)
        s = str(r)
        assert s.startswith("RootOf("), s
        assert ak.parse(s, p) == r
        # Inside a larger expression, with the caller's `x` bound to something
        # else: the placeholder is bound by RootOf and does not leak.
        t = p.symbol("t")
        syms = {"x": 2 * t, "t": t}
        e = t * r**2 + 1
        assert ak.parse(str(e), p, syms) == e
        assert set(syms) == {"x", "t"}
        assert syms["x"] == 2 * t


def test_root_of_parse_forms_and_refusals():
    p = ak.ExprPool()
    r = ak.parse("RootOf(y^3 - y - 1, y, 0)", p)
    assert str(r) == "RootOf(x^3 - x - 1, 0)"
    assert abs(ak.eval_expr(r, {}) - 1.324717957244746) < 1e-15
    # A reducible polynomial names the root it means.
    assert str(ak.parse("RootOf((x - 2)*(x^2 - 2), 2)", p)) == "2"
    for bad in [
        "RootOf(x^3 - x - 1)",
        "RootOf(x^3 - x - 1, 3)",
        "RootOf(x^3 - x - 1, -1)",
        "RootOf(x^3 - a*x - 1, 0)",
        "RootOf(x^3 - a*x - 1, x, 0)",
        "RootOf(7, 0)",
        "RootOf(sin(x), 0)",
    ]:
        with pytest.raises(ak.ParseError):
            ak.parse(bad, p)


def test_solve_output_round_trips():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([x**5 - x - 1], [x])
    for sol in sols:
        assert ak.parse(str(sol[x]), p) == sol[x]


# ---------------------------------------------------------------------------
# Eigenvectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "coeffs",
    [
        [1, 1, 0, 0],  # x^4 + x + 1
        [-1, -1, 0, 0],  # x^4 - x - 1
        [1, 0, 1, 0, 0],  # x^5 + x^2 + 1
        [-1, -1, 0, 0, 0],  # x^5 - x - 1
    ],
)
def test_companion_eigenvects_exact_and_numeric(coeffs):
    p = ak.ExprPool()
    m = ak.Matrix(_companion(p, coeffs))
    n = len(coeffs)
    a_np = _np_companion(coeffs)
    np_vals = np.linalg.eigvals(a_np)
    got = m.eigenvects()
    assert sum(mult for _, mult, _ in got) == n
    seen = []
    for lam, mult, vecs in got:
        assert str(lam).startswith("RootOf("), str(lam)
        assert (mult, len(vecs)) == (1, 1)
        v = vecs[0]
        # Exact: (A v − λ v)_i ≡ 0 mod p, reduced by SymPy.
        for i in range(n):
            row = sum((m.get(i, j) * v.get(j, 0) for j in range(n)), p.integer(0)) - lam * v.get(
                i, 0
            )
            assert _is_zero_mod_p(row), str(row)
        # Numeric: λ is a NumPy eigenvalue and v is its eigenvector.
        lv = _complex(lam)
        assert min(abs(lv - w) for w in np_vals) < 1e-9
        seen.append(lv)
        vv = np.array([_complex(v.get(i, 0)) for i in range(n)])
        assert np.linalg.norm(a_np @ vv - lv * vv) < 1e-9 * max(1.0, np.linalg.norm(vv))
    # Every eigenvalue appears exactly once.
    for w in np_vals:
        assert min(abs(w - s) for s in seen) < 1e-9


@pytest.mark.parametrize(
    "coeffs",
    [
        [-1, -1, 0],  # x^3 - x - 1: one real root, Cardano radicals
        [1, -3, 0],  # x^3 - 3x + 1: three real roots, the cos(2πk/9) form
        [-2, 0, 0],  # x^3 - 2: pure cubic
    ],
)
def test_irreducible_cubic_companion_eigenvects(coeffs):
    """Irreducible cubics keep their radical eigenvalues, whose symbolic
    elimination used to stall (E-LINALG-010). The kernel is now computed in
    ℚ[x]/(f) and read at the radical, so the entries are polynomials of degree
    ≤ 2 in it; checked against NumPy."""
    p = ak.ExprPool()
    m = ak.Matrix(_companion(p, coeffs))
    a_np = _np_companion(coeffs)
    np_vals = np.linalg.eigvals(a_np)
    got = m.eigenvects()
    assert sum(mult for _, mult, _ in got) == 3
    for lam, mult, vecs in got:
        assert (mult, len(vecs)) == (1, 1)
        lv = _complex(lam)
        assert min(abs(lv - w) for w in np_vals) < 1e-9
        vv = np.array([_complex(vecs[0].get(i, 0)) for i in range(3)])
        assert np.linalg.norm(vv) > 1e-6
        assert np.linalg.norm(a_np @ vv - lv * vv) < 1e-9 * np.linalg.norm(vv)
    pm, j = m.jordan_form()
    pn = np.array([[_complex(pm.get(r, c)) for c in range(3)] for r in range(3)])
    jn = np.array([[_complex(j.get(r, c)) for c in range(3)] for r in range(3)])
    assert np.allclose(a_np @ pn, pn @ jn, atol=1e-9)
    assert abs(np.linalg.det(pn)) > 1e-8


def test_diagonalize_root_of_spectrum():
    p = ak.ExprPool()
    m = ak.Matrix(_companion(p, [1, 1, 0, 0]))
    pm, d = m.diagonalize()
    a_np = _np_companion([1, 1, 0, 0])
    pn = np.array([[_complex(pm.get(i, j)) for j in range(4)] for i in range(4)])
    dn = np.array([[_complex(d.get(i, j)) for j in range(4)] for i in range(4)])
    assert np.allclose(a_np @ pn, pn @ dn, atol=1e-9)
    assert abs(np.linalg.det(pn)) > 1e-6


# ---------------------------------------------------------------------------
# Jordan form
# ---------------------------------------------------------------------------


def _dot(a, r, b, c, n, p):
    """``(A·B)[r, c]`` summed over the terms with no literal-zero factor."""
    terms = [
        a.get(r, k) * b.get(k, c)
        for k in range(n)
        if str(a.get(r, k)) != "0" and str(b.get(k, c)) != "0"
    ]
    return sum(terms, p.integer(0))


def _block_upper(p, c_rows):
    """``[[C, I], [0, C]]`` — characteristic polynomial ``χ_C²`` with one
    Jordan block of size 2 per root of ``χ_C``."""
    n = len(c_rows)
    rows = [[p.integer(0) for _ in range(2 * n)] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            rows[i][j] = c_rows[i][j]
            rows[i + n][j + n] = c_rows[i][j]
        rows[i][i + n] = p.integer(1)
    return rows


def test_jordan_form_repeated_irreducible_quartic():
    p = ak.ExprPool()
    coeffs = [1, 1, 0, 0]  # x^4 + x + 1
    m = ak.Matrix(_block_upper(p, _companion(p, coeffs)))
    pm, j = m.jordan_form()
    n = 8
    # J: four 2×2 blocks, one per root of x^4 + x + 1.
    diag = [str(j.get(i, i)) for i in range(n)]
    assert all(s.startswith("RootOf(x^4 + x + 1") for s in diag), diag
    assert [str(j.get(i, i + 1)) for i in range(0, n, 2)] == ["1"] * 4
    assert [str(j.get(i, i + 1)) for i in range(1, n - 1, 2)] == ["0"] * 3
    # Exact: M·P = P·J column by column, each column over one RootOf.
    for c in range(n):
        for r in range(n):
            # Only the non-zero terms: a column involves one conjugate, and a
            # literal `0 * RootOf(…, other)` term is not part of the identity.
            lhs = _dot(m, r, pm, c, n, p)
            rhs = _dot(pm, r, j, c, n, p)
            assert _is_zero_mod_p(lhs - rhs), (r, c)
    # Numeric: P invertible and M = P J P⁻¹.
    a_np = np.array([[_complex(m.get(r, c)) for c in range(n)] for r in range(n)])
    pn = np.array([[_complex(pm.get(r, c)) for c in range(n)] for r in range(n)])
    jn = np.array([[_complex(j.get(r, c)) for c in range(n)] for r in range(n)])
    assert abs(np.linalg.det(pn)) > 1e-8
    assert np.allclose(a_np, pn @ jn @ np.linalg.inv(pn), atol=1e-8)


# ---------------------------------------------------------------------------
# Matrix exponential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repeated", [False, True])
def test_matrix_exp_of_root_of_spectrum_matches_scipy(repeated):
    linalg = pytest.importorskip("scipy.linalg")
    p = ak.ExprPool()
    t = p.symbol("t")
    c = _companion(p, [1, 1, 0, 0])
    rows = _block_upper(p, c) if repeated else c
    n = len(rows)
    a = ak.Matrix(rows)
    a_np = np.array([[float(str(a.get(i, j))) for j in range(n)] for i in range(n)])
    e = ak.Matrix([[rows[i][j] * t for j in range(n)] for i in range(n)]).matrix_exp()
    for tv in (0.0, 0.37, -1.1, 2.0):
        want = linalg.expm(a_np * tv)
        got = np.array([[ak.eval_expr(e.get(i, j), {t: tv}) for j in range(n)] for i in range(n)])
        assert np.allclose(got, want, rtol=1e-8, atol=1e-9), (tv, got - want)
    # The rational matrix itself (s = 1).
    e1 = a.matrix_exp()
    got = np.array([[ak.eval_expr(e1.get(i, j), {}) for j in range(n)] for i in range(n)])
    assert np.allclose(got, linalg.expm(a_np), rtol=1e-8, atol=1e-9)


def test_matrix_exp_mixed_spectrum_matches_scipy():
    """A rational eigenvalue, a complex quadratic pair and a RootOf quartic in
    one block-diagonal matrix (conjugated by a unimodular matrix)."""
    linalg = pytest.importorskip("scipy.linalg")
    p = ak.ExprPool()
    blocks = [[[2]], [[0, -2], [1, 0]], [[0, 0, 0, -1], [1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]]]
    n = 7
    d = np.zeros((n, n))
    o = 0
    for b in blocks:
        k = len(b)
        d[o : o + k, o : o + k] = b
        o += k
    u = np.eye(n) + np.triu(np.ones((n, n)), 1)  # unimodular
    a_np = u @ d @ np.linalg.inv(u)
    a_int = np.rint(a_np).astype(int)
    assert np.allclose(a_int, a_np)
    a = ak.Matrix([[p.integer(int(a_int[i, j])) for j in range(n)] for i in range(n)])
    e = a.matrix_exp()
    got = np.array([[ak.eval_expr(e.get(i, j), {}) for j in range(n)] for i in range(n)])
    assert np.allclose(got, linalg.expm(a_int.astype(float)), rtol=1e-8, atol=1e-8)


def test_symbolic_entries_with_root_of_spectrum_still_refuse():
    """Outside ℚ (and ℚ·t) there is no number field to compute in."""
    p = ak.ExprPool()
    a, b = p.symbol("a"), p.symbol("b")
    rows = _companion(p, [1, 1, 0, 0])
    rows[0][0] = a
    rows[1][1] = b
    with pytest.raises(Exception):
        ak.Matrix(rows).eigenvects()
