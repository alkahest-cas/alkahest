"""Symbolic ``solve`` on polynomial *systems* whose eliminant has degree > 2
(benchmark 2026-10-25, B4).

The 3.11.0 wheel refused ``[x² + y² + z² − 4, xy − z − 1, x + y + z³ − 2]``
with ``E-SOLVE-002`` ("back-substitution requires solving a degree-6
univariate polynomial") although ``numeric=True`` solved it. SymPy's ``solve``
and ``solve_poly_system`` return ``[]`` on the same system — a silent wrong
answer. Back-substitution now factors the eliminant over ℚ and returns
``RootOf(f, k)`` for irreducible factors past the radical cases, so every one
of the 12 complex solutions comes back exactly.

Every solution here is checked against independent oracles:

* substituted into the equations at 80 digits with mpmath, each ``RootOf``
  evaluated from ``mpmath.polyroots`` in the documented root order (real
  roots ascending, then complex by real part, then imaginary part) — so a
  wrong index or a wrong branch fails the residual check;
* the solution set is compared, point by point, with ``numeric=True``;
* where ``y`` is a polynomial in the ``RootOf`` (shape position), the
  residual is reduced exactly modulo the minimal polynomial with SymPy.

Solution counts were cross-checked against Wolfram ``NSolve`` when writing
the tests (12, 6, 6, 8).
"""

import random

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "solve_numerical"),
    reason="native module built without groebner feature",
)

DPS = 80

# ---------------------------------------------------------------------------
# An mpmath evaluator for solver output (integers, rationals, +, ·, ^, sqrt,
# RootOf), independent of alkahest's own numeric evaluators.
# ---------------------------------------------------------------------------

_ROOTS: dict = {}


def _ordered_roots(coeffs_desc):
    key = tuple(coeffs_desc)
    if key not in _ROOTS:
        with mpmath.workdps(DPS + 20):
            rs = mpmath.polyroots(list(coeffs_desc), maxsteps=800, extraprec=4 * DPS)
            tiny = mpmath.mpf(10) ** -(DPS // 2)
            real = sorted(mpmath.re(r) for r in rs if abs(mpmath.im(r)) < tiny)
            cplx = sorted(
                (r for r in rs if abs(mpmath.im(r)) >= tiny),
                key=lambda r: (mpmath.re(r), mpmath.im(r)),
            )
        _ROOTS[key] = [mpmath.mpc(r) for r in real] + cplx
    return _ROOTS[key]


def _mp(e):
    node = e.node()
    tag = node[0]
    if tag == "integer":
        return mpmath.mpf(int(node[1]))
    if tag == "rational":
        return mpmath.mpf(int(node[1])) / int(node[2])
    if tag == "add":
        return mpmath.fsum(_mp(c) for c in node[1])
    if tag == "mul":
        out = mpmath.mpf(1)
        for c in node[1]:
            out *= _mp(c)
        return out
    if tag == "pow":
        exp = node[2].node()
        if exp[0] == "integer":
            return _mp(node[1]) ** int(exp[1])
        return mpmath.power(_mp(node[1]), _mp(node[2]))
    if tag == "func" and node[1] == "sqrt":
        return mpmath.sqrt(_mp(node[2][0]))
    if tag == "func" and node[1] == "RootOf":
        k = int(node[2][0].node()[1])
        coeffs = [int(c.node()[1]) for c in node[2][1:]]
        return _ordered_roots(coeffs)[k]
    raise AssertionError(f"unexpected node in a solution: {node}")


def _check(eq_strs, names, *, expected_count, root_of=True):
    """Solve exactly, then check every tuple and the count against oracles."""
    pool = ak.ExprPool()
    syms = [pool.symbol(n) for n in names]
    eqs = [ak.parse(s, pool) for s in eq_strs]
    sols = ak.solve(eqs, syms)
    assert isinstance(sols, list)
    assert len(sols) == expected_count
    if root_of:
        assert any("RootOf" in str(s[v]) for s in sols for v in syms)
    pts = []
    with mpmath.workdps(DPS):
        for s in sols:
            assert not any("." in str(s[v]) for v in syms), "a float leaked into an exact solution"
            env = {n: _mp(s[v]) for n, v in zip(names, syms)}
            scale = 1 + max(abs(c) for c in env.values())
            for eq in eq_strs:
                r = eval(eq, {"__builtins__": {}}, dict(env))
                assert abs(r) < mpmath.mpf(10) ** -(DPS // 2) * scale**6, (eq, s)
            pts.append([complex(env[n]) for n in names])
    # Pairwise distinct.
    for i in range(len(pts)):
        for j in range(i):
            assert sum(abs(a - b) for a, b in zip(pts[i], pts[j])) > 1e-8
    # Same set as the homotopy / numeric solver, including complex roots.
    numeric = ak.solve(eqs, syms, numeric=True)
    assert len(numeric) == expected_count

    def cval(v):
        try:
            return complex(v)
        except TypeError:
            return complex(ak.evaluate(v, {}, mode="complex").value)

    for s in numeric:
        q = [cval(s[v]) for v in syms]
        assert any(sum(abs(a - b) for a, b in zip(pt, q)) < 1e-6 for pt in pts), q
    return pool, syms, sols


def test_benchmark_repro_sextic_eliminant():
    _, (x, y, z), sols = _check(
        ["x**2+y**2+z**2-4", "x*y-z-1", "x+y+z**3-2"], "xyz", expected_count=12
    )
    assert all(str(s[z]).startswith("RootOf(") for s in sols)
    real = [
        [float(ak.evaluate(s[v], {}, mode="complex").value.real) for v in (x, y, z)]
        for s in sols
        if all(
            abs(complex(ak.evaluate(s[v], {}, mode="complex").value).imag) < 1e-12
            for v in (x, y, z)
        )
    ]
    assert len(real) == 2
    assert any(
        abs(p[0] - 0.24267) < 1e-5 and abs(p[1] - 1.91145) < 1e-5 and abs(p[2] + 0.53615) < 1e-5
        for p in real
    )


def test_circle_meets_cubic():
    _check(["x**2+y**2-4", "y-x**3-x-1"], "xy", expected_count=6)


def test_folium_meets_parabola():
    _check(["x**3+y**3-3*x*y-1", "x**2-y-2"], "xy", expected_count=6)


def test_circle_meets_cubic_through_radicals():
    # Eliminant factors into pieces solvable by radicals: no RootOf needed.
    _check(["x**2+y**2-1", "y-x**3+x"], "xy", expected_count=6, root_of=False)


def test_katsura_3():
    _check(
        [
            "x+2*y+2*z+2*w-1",
            "x**2+2*y**2+2*z**2+2*w**2-x",
            "2*x*y+2*y*z+2*z*w-y",
            "y**2+2*x*z+2*y*w-z",
        ],
        "xyzw",
        expected_count=8,
    )


def test_shape_position_is_a_polynomial_in_the_root_of():
    """``⟨x − y² − y, y⁵ − y − 1⟩``: ``x`` is a polynomial in the ``RootOf``,
    and its residual reduces to 0 exactly modulo the minimal polynomial."""
    sympy = pytest.importorskip("sympy")
    _, (x, y), sols = _check(["x-y**2-y", "y**5-y-1"], "xy", expected_count=5)
    t = sympy.Symbol("t")
    f = t**5 - t - 1
    for k, s in enumerate(sorted(sols, key=lambda s: str(s[y]))):
        assert str(s[y]) == f"RootOf(x^5 - x - 1, {k})"
        assert "sqrt" not in str(s[x])
        xs = sympy.sympify(str(s[x]).replace(str(s[y]), "t").replace("^", "**"))
        assert sympy.rem(sympy.expand(xs - t**2 - t), f, t) == 0


def test_root_of_systems_honour_budget():
    pool = ak.ExprPool()
    x, y, z = (pool.symbol(n) for n in "xyz")
    eqs = [ak.parse(s, pool) for s in ["x**2+y**2+z**2-4", "x*y-z-1", "x+y+z**3-2"]]
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=0)):
        ak.solve(eqs, [x, y, z])


def _random_poly(rng, degree):
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            c = rng.randint(-3, 3)
            if c:
                terms.append(f"({c})*x**{i}*y**{j}")
    return "+".join(terms) or "1"


@pytest.mark.parametrize("seed", range(6))
def test_random_dense_systems_match_numeric(seed):
    """Dense random 2×2 systems of degree (3, 2) or (3, 3): eliminants of
    degree 6–9 that are almost always irreducible."""
    rng = random.Random(1000 + seed)
    eqs = [_random_poly(rng, 3), _random_poly(rng, rng.choice([2, 3]))]
    pool = ak.ExprPool()
    syms = [pool.symbol("x"), pool.symbol("y")]
    count = len(ak.solve([ak.parse(s, pool) for s in eqs], syms, numeric=True))
    _check(eqs, "xy", expected_count=count, root_of=False)
