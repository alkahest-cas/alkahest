"""``solve`` past degree 2: factor first, then formula, radicals or ``RootOf``
(report 10-9, G1).

``solve`` refused every univariate of degree ≥ 3 with ``E-SOLVE-002`` —
``s**3 + 6*s**2 + 11*s + 6`` (roots -1, -2, -3), ``s*(s**2 - 5)``,
``x**3 + 2*x**2 - 3*x``, the Butterworth denominator ``s**4 + 1``, and 9 of 30
random integer polynomials. Rational-coefficient univariates are now factored
over ℚ; each irreducible factor is solved by formula (degree ≤ 2), by an even /
palindromic / pure-cubic reduction to radicals, or as ``RootOf(f, k)`` — an
exact root with a documented index order (real roots ascending, then complex
by real part, then imaginary part) that evaluates to any precision.

Every check here is against an independent oracle: NumPy's companion-matrix
roots, SymPy's distinct-root count, mpmath at 50 digits, or the exact value.
"""

import random

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "solve_numerical"),
    reason="native module built without groebner feature",
)


def _upoly(p, x, coeffs):
    """``Σ c_k x^k`` from ascending integer coefficients."""
    expr = None
    for k, c in enumerate(coeffs):
        if c == 0:
            continue
        term = c * x**k if k else p.integer(c)
        expr = term if expr is None else expr + term
    return expr


def _values(sols, var):
    out = []
    for sol in sols:
        r = ak.evaluate(sol[var], {}, mode="complex")
        assert r.status == "ok", (sol[var], r.reason)
        out.append(complex(r.value))
    return out


def _match(got, want, tol=1e-7):
    """Every wanted root has a distinct partner in ``got`` within ``tol``."""
    got = list(got)
    assert len(got) == len(want), (got, want)
    for w in want:
        i = min(range(len(got)), key=lambda j: abs(got[j] - w))
        assert abs(got[i] - w) <= tol * max(1.0, abs(w)), (got, want)
        got.pop(i)


def _distinct_numpy_roots(coeffs):
    """Distinct roots of the polynomial via SymPy's square-free part, then
    NumPy's companion matrix (well conditioned once roots are simple)."""
    sympy = pytest.importorskip("sympy")
    t = sympy.Symbol("t")
    p = sympy.Poly(list(reversed(coeffs)), t)
    sf = sympy.Poly(sympy.sqf_part(p.as_expr()), t)
    return [complex(r) for r in np.roots([float(c) for c in sf.all_coeffs()])]


# ---------------------------------------------------------------------------
# The report's repros
# ---------------------------------------------------------------------------


def test_cubic_with_rational_roots_is_exact():
    p = ak.ExprPool()
    s = p.symbol("s")
    sols = ak.solve([s**3 + 6 * s**2 + 11 * s + 6], [s])
    assert sorted(str(sol[s]) for sol in sols) == ["-1", "-2", "-3"]


@pytest.mark.parametrize(
    "coeffs",
    [
        [0, -5, 0, 1],  # s(s^2 - 5)
        [0, -3, 2, 1],  # x^3 + 2x^2 - 3x
        [1, 0, 0, 0, 1],  # s^4 + 1, Butterworth n = 4
        [1, 0, 0, 0, 0, 0, 0, 0, 1],  # s^8 + 1, Butterworth n = 8
    ],
    ids=["s(s2-5)", "x3+2x2-3x", "s4+1", "s8+1"],
)
def test_report_repros_return_every_root(coeffs):
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([_upoly(p, x, coeffs)], [x])
    _match(_values(sols, x), _distinct_numpy_roots(coeffs))


def test_exact_roots_substitute_back_rigorously():
    p = ak.ExprPool()
    x = p.symbol("x")
    eq = x * (x**2 - 5)
    sols = ak.solve([eq], [x])
    assert len(sols) == 3
    for sol in sols:
        # A rigorous 256-bit enclosure of the residual must contain 0 and be
        # tiny (a wrong root would be separated from 0 by far more).
        ball = ak.interval_eval(ak.subs(eq, {x: sol[x]}), {}, 256)
        assert ball.contains(0.0)
        assert ball.rad < 1e-60


# ---------------------------------------------------------------------------
# Random integer polynomials, degree 3–8
# ---------------------------------------------------------------------------


def test_random_integer_polynomials_match_numpy():
    rng = random.Random(20261010)
    p = ak.ExprPool()
    x = p.symbol("x")
    for _ in range(30):
        deg = rng.randint(3, 8)
        coeffs = [rng.randint(-6, 6) for _ in range(deg)] + [rng.choice([1, 2, 3, -1])]
        if all(c == 0 for c in coeffs[:-1]):
            coeffs[0] = 1
        sols = ak.solve([_upoly(p, x, coeffs)], [x])
        _match(_values(sols, x), _distinct_numpy_roots(coeffs), tol=1e-6)


def test_numeric_true_agrees_with_the_exact_answer():
    p = ak.ExprPool()
    x = p.symbol("x")
    coeffs = [-1, -1, 0, 0, 0, 1]  # x^5 - x - 1, no radical form
    num = ak.solve([_upoly(p, x, coeffs)], [x], numeric=True)
    got = [complex(sol[x]) for sol in num]
    _match(got, _distinct_numpy_roots(coeffs))


# ---------------------------------------------------------------------------
# RootOf
# ---------------------------------------------------------------------------


def test_irreducible_quintic_returns_root_of_in_documented_order():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([x**5 - x - 1], [x])
    assert [str(sol[x]) for sol in sols] == [f"RootOf(x^5 - x - 1, {k})" for k in range(5)]
    vals = _values(sols, x)
    # Real root first, then the two conjugate pairs by real part, lower first.
    assert vals[0].imag == 0
    pairs = vals[1:]
    assert pairs[0].real < pairs[2].real
    assert pairs[0].imag < 0 < pairs[1].imag
    assert pairs[2].imag < 0 < pairs[3].imag


def test_root_of_evaluates_to_thirty_digits():
    """A 200-bit enclosure of the plastic number, checked against mpmath at
    50 digits and by evaluating the cubic on the enclosure itself."""
    from fractions import Fraction

    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    want = next(r for r in mpmath.polyroots([1, 0, -1, -1]) if abs(mpmath.im(r)) < 1e-40)
    ref = Fraction(mpmath.nstr(mpmath.re(want), 48))
    p = ak.ExprPool()
    x = p.symbol("x")
    r = ak.root_of(x**3 - x - 1, x, 0)
    assert str(r) == "RootOf(x^3 - x - 1, 0)"
    ball = ak.interval_eval(r, {}, 200)
    assert ball.rad < 1e-50
    diff = ak.interval_eval(r - p.rational(ref.numerator, ref.denominator), {}, 200)
    assert abs(diff.mid) + diff.rad < 1e-30
    residual = ak.interval_eval(r**3 - r - 1, {}, 200)
    assert residual.contains(0.0)
    assert residual.rad < 1e-50


def test_root_of_is_a_constant():
    p = ak.ExprPool()
    x = p.symbol("x")
    r = ak.root_of(x**3 - x - 1, x, 0)
    assert str(ak.diff(r, x).value) == "0"
    assert ak.subs(r, {x: p.integer(5)}) == r
    assert abs(ak.eval_expr(r, {}) - 1.324717957244746) < 1e-15
    # Not an undefined function: no Derivative node, no E-EVAL-005.
    assert not ak._is_undefined_function_name("RootOf")
    t = p.symbol("t")
    e = r * t + 1
    assert str(ak.diff(e, t).value) == "RootOf(x^3 - x - 1, 0)"
    assert abs(ak.eval_expr(e, {t: 2.0}) - 3.649435914489492) < 1e-14
    # Compiled code reads it as the constant it is.
    f = ak.compile_expr(e, [t])
    assert abs(f([2.0]) - 3.649435914489492) < 1e-14
    assert r"\operatorname{RootOf}" in ak.latex(e)


def test_root_of_reduces_and_refuses():
    p = ak.ExprPool()
    x = p.symbol("x")
    # (x - 2)(x^2 - 2): real roots -√2, √2, 2.
    poly = (x - 2) * (x**2 - 2)
    assert str(ak.root_of(poly, x, 2)) == "2"
    r1 = ak.root_of(poly, x, 1)
    assert abs(ak.eval_expr(r1, {}) - 2**0.5) < 1e-15
    with pytest.raises(ak.SolverError) as exc:
        ak.root_of(poly, x, 3)
    assert exc.value.code == "E-SOLVE-008"


def test_domain_real_keeps_only_real_roots():
    p = ak.ExprPool()
    x = p.symbol("x")
    assert ak.solve([x**4 + 1], [x], domain="real") == []
    real = ak.solve([x**5 - x - 1], [x], domain="real")
    assert [str(s[x]) for s in real] == ["RootOf(x^5 - x - 1, 0)"]


# ---------------------------------------------------------------------------
# Domain workflows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4])
def test_gauss_legendre_nodes(n):
    p = ak.ExprPool()
    x = p.symbol("x")
    legendre = {3: (5 * x**3 - 3 * x) / 2, 4: (35 * x**4 - 30 * x**2 + 3) / 8}[n]
    sols = ak.solve([legendre], [x])
    nodes, _ = np.polynomial.legendre.leggauss(n)
    vals = _values(sols, x)
    assert all(abs(v.imag) < 1e-12 for v in vals)
    _match([v.real for v in vals], list(nodes), tol=1e-12)


def test_parametric_cubic_with_a_rational_root():
    p = ak.ExprPool()
    s, a = p.symbol("s"), p.symbol("a")
    sols = ak.solve([s**3 - a * s**2 - s + a], [s])
    assert sorted(str(sol[s]) for sol in sols) == ["-1", "1", "a"]


def test_parametric_irreducible_cubic_still_refuses():
    p = ak.ExprPool()
    s, a = p.symbol("s"), p.symbol("a")
    with pytest.raises(ak.SolverError) as exc:
        ak.solve([s**3 + a * s + 1], [s])
    assert exc.value.code == "E-SOLVE-002"


def test_solow_steady_state():
    p = ak.ExprPool()
    z = p.symbol("z")
    sols = ak.solve([p.rational(3, 10) * z - p.rational(1, 10) * z**3], [z])
    _match(_values(sols, z), [0, 3**0.5, -(3**0.5)], tol=1e-12)


def test_lqr_riccati_with_numeric_weights():
    """Double integrator, Q = I, r = 1: the Lex basis ends in
    p22^4 - 2 p22^2 - 3 = (p22^2 - 3)(p22^2 + 1), which used to stop the solve."""
    p = ak.ExprPool()
    p11, p12, p22 = p.symbol("p11"), p.symbol("p12"), p.symbol("p22")
    eqs = [-(p12**2) + 1, p11 - p12 * p22, 2 * p12 - p22**2 + 1]
    sols = ak.solve(eqs, [p11, p12, p22])
    assert len(sols) == 4
    real = ak.solve(eqs, [p11, p12, p22], domain="real")
    got = sorted(tuple(round(ak.eval_expr(s[v], {}), 12) for v in (p11, p12, p22)) for s in real)
    r3 = round(3**0.5, 12)
    assert got == [(-r3, 1.0, -r3), (r3, 1.0, r3)]


# ---------------------------------------------------------------------------
# Eigenvalues and Routh–Hurwitz on factored characteristic polynomials
# ---------------------------------------------------------------------------


def _seir_jacobian(p):
    b, g, s_, mu = (p.symbol(n, ak.Domain.Positive) for n in ("beta", "gamma", "sigma", "mu"))
    z = p.integer(0)
    m = ak.Matrix([[-mu, z, -b], [z, -(s_ + mu), b], [z, s_, -(g + mu)]])
    return m, (b, g, s_, mu)


def test_seir_eigenvalues_factor():
    p = ak.ExprPool()
    m, (b, g, s_, mu) = _seir_jacobian(p)
    ev = m.eigenvals()
    assert sum(ev.values()) == 3
    env = {b: 0.4, g: 0.1, s_: 0.2, mu: 0.01}
    got = [ak.evaluate(e, env, mode="complex").value for e in ev]
    a = np.array([[-0.01, 0, -0.4], [0, -0.21, 0.4], [0, 0.2, -0.11]])
    _match(got, list(np.linalg.eigvals(a)), tol=1e-9)


def test_three_site_ring_has_a_double_eigenvalue():
    p = ak.ExprPool()
    t = p.symbol("t")
    z = p.integer(0)
    m = ak.Matrix([[z, -t, -t], [-t, z, -t], [-t, -t, z]])
    ev = m.eigenvals()
    got = sorted((round(ak.eval_expr(e, {t: 1.0}), 12), k) for e, k in ev.items())
    assert got == [(-2.0, 1), (1.0, 2)]


def test_huckel_four_site_chain():
    p = ak.ExprPool()
    a, b = p.symbol("alpha"), p.symbol("beta")
    z = p.integer(0)
    m = ak.Matrix([[a, b, z, z], [b, a, b, z], [z, b, a, b], [z, z, b, a]])
    ev = m.eigenvals()
    assert sum(ev.values()) == 4
    got = [ak.evaluate(e, {a: 0.3, b: -1.1}, mode="complex").value for e in ev]
    phi = (1 + 5**0.5) / 2
    _match(got, [0.3 - 1.1 * k for k in (phi, phi - 1, -phi, 1 - phi)], tol=1e-12)


def test_routh_hurwitz_accepts_the_factored_characteristic_polynomial():
    p = ak.ExprPool()
    m, _ = _seir_jacobian(p)
    cp, lam = m.characteristic_polynomial_lambda_minus_m()
    rh = ak.routh_hurwitz(cp, lam)
    assert rh is not None
