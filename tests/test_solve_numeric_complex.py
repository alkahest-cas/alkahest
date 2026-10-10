"""``solve(..., numeric=True)`` over ℂ (report 10-9, W2).

The default solve domain is complex, but the numeric read-out used to go
through the real interpreter and the degree > 2 fallback through the real-only
homotopy solver, so:

* ``solve([s**2 + 1], [s], numeric=True)`` was ``[{s: nan}, {s: nan}]``,
* ``solve([s**4 + 1], [s], numeric=True)`` was ``[]``,
* ``solve([s**3 + 1], [s], numeric=True)`` was only the root ``-1``,
* ``solve([x - y], [x], numeric=True)`` was ``[{x: nan}]``.

Non-real roots now come back as Python ``complex``; a value that depends on a
free parameter refuses with ``E-SOLVE-006``.
"""

import cmath
import random

import alkahest
import pytest

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "solve_numerical"),
    reason="native module built without groebner feature",
)


def _roots(sols, var):
    return [complex(s[var]) for s in sols]


def _assert_same_multiset(got, want, tol=1e-7):
    got = list(got)
    assert len(got) == len(want), (got, want)
    for w in want:
        best = min(range(len(got)), key=lambda i: abs(got[i] - w))
        assert abs(got[best] - w) <= tol * (1 + abs(w)), (got, want)
        got.pop(best)


def _poly(pool, var, coeffs_desc):
    """Polynomial with descending integer coefficients (numpy.roots order)."""
    deg = len(coeffs_desc) - 1
    expr = pool.integer(0)
    for k, c in enumerate(coeffs_desc):
        if c:
            expr = expr + pool.integer(int(c)) * var ** (deg - k)
    return expr


def test_s_squared_plus_one_is_plus_minus_i():
    p = alkahest.ExprPool()
    s = p.symbol("s")
    sols = alkahest.solve([s**2 + 1], [s], numeric=True)
    vals = _roots(sols, s)
    assert all(not cmath.isnan(v) for v in vals)
    _assert_same_multiset(vals, [1j, -1j], tol=1e-14)


def test_s_fourth_plus_one_has_four_roots():
    p = alkahest.ExprPool()
    s = p.symbol("s")
    sols = alkahest.solve([s**4 + 1], [s], numeric=True)
    want = [cmath.exp(1j * cmath.pi * (2 * k + 1) / 4) for k in range(4)]
    _assert_same_multiset(_roots(sols, s), want, tol=1e-10)


def test_s_cubed_plus_one_has_three_roots_and_the_real_one_is_float():
    p = alkahest.ExprPool()
    s = p.symbol("s")
    sols = alkahest.solve([s**3 + 1], [s], numeric=True)
    want = [-1, 0.5 + 0.75**0.5 * 1j, 0.5 - 0.75**0.5 * 1j]
    _assert_same_multiset(_roots(sols, s), want, tol=1e-10)
    reals = [sol[s] for sol in sols if isinstance(sol[s], float)]
    assert reals == [-1.0]


def test_free_parameter_is_a_coded_refusal_not_nan():
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(alkahest.SolverError) as ei:
        alkahest.solve([x - y], [x], numeric=True)
    assert ei.value.code == "E-SOLVE-006"
    assert "y" in str(ei.value)
    # The symbolic answer is still there.
    (sol,) = alkahest.solve([x - y], [x])
    assert sol[x] == y


def test_free_parameter_on_the_high_degree_path_is_refused_too():
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(alkahest.SolverError) as ei:
        alkahest.solve([x**3 - y], [x], numeric=True)
    assert ei.value.code == "E-SOLVE-006"


def test_system_with_complex_solutions_counts_all_of_them():
    # x² + y² = 1, x = y³  ⇒  y⁶ + y² − 1 = 0: six solutions, two real.
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sols = alkahest.solve([x**2 + y**2 - 1, x - y**3], [x, y], numeric=True)
    want_y = np.roots([1, 0, 0, 0, 1, 0, -1])
    _assert_same_multiset(_roots(sols, y), list(want_y), tol=1e-8)
    for sol in sols:
        xv, yv = complex(sol[x]), complex(sol[y])
        assert abs(xv**2 + yv**2 - 1) < 1e-9
        assert abs(xv - yv**3) < 1e-9


def test_domain_real_drops_the_complex_roots_of_the_numeric_fallback():
    p = alkahest.ExprPool()
    s = p.symbol("s")
    assert alkahest.solve([s**4 + 1], [s], numeric=True, domain="real") == []
    sols = alkahest.solve([s**3 + 1], [s], numeric=True, domain="real")
    assert [sol[s] for sol in sols] == [-1.0]


@pytest.mark.parametrize("seed", range(24))
def test_random_integer_polynomials_match_numpy_roots(seed):
    rng = random.Random(seed)
    deg = 2 + seed % 7  # 2..8
    while True:
        coeffs = [rng.randint(-9, 9) for _ in range(deg + 1)]
        if coeffs[0] == 0 or coeffs[-1] == 0:
            continue
        want = np.roots(coeffs)
        # Keep the comparison well-posed: squarefree, well-separated roots.
        gaps = [abs(a - b) for i, a in enumerate(want) for b in want[i + 1 :]]
        if min(gaps) > 1e-2:
            break
    p = alkahest.ExprPool()
    s = p.symbol("s")
    sols = alkahest.solve([_poly(p, s, coeffs)], [s], numeric=True)
    got = _roots(sols, s)
    assert len(got) == deg
    assert all(not cmath.isnan(v) for v in got)
    _assert_same_multiset(got, list(want), tol=1e-6)


def test_roots_at_infinity_are_not_counted_and_not_lost():
    # Bézout bound 9, but only the three cube roots are finite.
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sols = alkahest.solve([x**2 * y - 1, x * y**2 - 2], [x, y], numeric=True)
    xs = _roots(sols, x)
    want = [0.5 ** (1 / 3) * cmath.exp(2j * cmath.pi * k / 3) for k in range(3)]
    _assert_same_multiset(xs, want, tol=1e-9)
    for sol in sols:
        assert abs(complex(sol[y]) - 2 * complex(sol[x])) < 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_random_dense_systems_return_the_full_bezout_count(seed):
    # Dense random systems have exactly d1*d2 finite solutions.  The affine
    # tracker this replaced dropped one of them (a jumped path) about 7% of
    # the time; seed 33 of this generator was one such case.
    import itertools

    rng = random.Random(seed + 30)
    d1, d2 = rng.choice([(2, 2), (2, 3), (3, 3), (3, 2)])
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")

    def dense(d):
        e = p.integer(rng.randint(1, 9))
        for i, j in itertools.product(range(d + 1), repeat=2):
            if 0 < i + j <= d:
                c = rng.randint(-9, 9)
                if i + j == d and c == 0:
                    c = 1
                if c:
                    e = e + p.integer(c) * x**i * y**j
        return e

    eqs = [dense(d1), dense(d2)]
    sols = alkahest.solve(eqs, [x, y], numeric=True)
    assert len(sols) == d1 * d2
    pts = {(complex(s[x]), complex(s[y])) for s in sols}
    for a, b in pts:
        # Rational coefficients: the solution set is closed under conjugation.
        assert any(abs(a.conjugate() - c) < 1e-8 and abs(b.conjugate() - d) < 1e-8 for c, d in pts)


def test_homotopy_real_mode_still_reports_only_real_roots():
    p = alkahest.ExprPool()
    s = p.symbol("s")
    pts = alkahest.solve_numerical([s**3 + 1], [s])
    assert [round(pt.to_dict()[s], 12) for pt in pts] == [-1.0]
    assert alkahest.solve_numerical([s**4 + 1], [s]) == []
