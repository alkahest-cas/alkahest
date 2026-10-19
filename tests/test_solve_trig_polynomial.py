"""``solve`` for trigonometric polynomial equations, systems through invertible
functions, and complex exponential families (F6).

* Equations polynomial in ``sin``/``cos``/``tan`` of integer multiples of one
  angle (``2*sin(x)**2 = 1``, ``sin(x) = cos(x)``, ``sin(2x) = cos(x)``) return
  whole periodic families in an integer index ``k``.
* Systems in which an unknown occurs only inside one invertible function
  (``exp(x) + y = 3, exp(x) - y = 1``) are solved through that function.
* ``domain="complex"`` (or a complex unknown) gives ``exp(x) = c`` the family
  ``log(c) + 2*pi*I*k``.

Every family member is substituted back; completeness is checked by a
sign-change scan on [-10, 10], and the points in [-10, 10] are compared with
SymPy's ``solveset`` in both directions.
"""

import cmath
import math

import alkahest as ak
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "solve_numerical"),
    reason="native module built without groebner feature",
)


def _indices(pool):
    names = [c.split(" ")[0] for c in ak.solve_side_conditions() if "Integer" in c]
    return [pool.symbol(n, "integer") for n in names]


def _members(sols, x, pool, ks=(-8, 9)):
    """Every family member for the given index values (as floats)."""
    idx = _indices(pool)
    out = []
    for s in sols:
        for kv in range(*ks):
            env = {k: float(kv) for k in idx}
            out.append(ak.eval_expr(s[x], env))
    return out


def _bisect(f, lo, hi):
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _scan_roots(f, lo=-10.0, hi=10.0, n=40000):
    """Sign changes of ``f`` on [lo, hi] (poles of tan excluded)."""
    roots = []
    step = (hi - lo) / n
    prev_x, prev = lo, f(lo)
    for i in range(1, n + 1):
        xv = lo + i * step
        cur = f(xv)
        if abs(prev) < 1 and abs(cur) < 1 and (prev < 0) != (cur < 0):
            roots.append(_bisect(f, prev_x, xv))
        prev_x, prev = xv, cur
    return roots


CASES = [
    # (name, alkahest builder, numeric f, sympy builder)
    (
        "2sin^2=1",
        lambda x, p: 2 * ak.sin(x) ** 2 - 1,
        lambda v: 2 * math.sin(v) ** 2 - 1,
        lambda sp, x: 2 * sp.sin(x) ** 2 - 1,
    ),
    (
        "sin=cos",
        lambda x, p: ak.sin(x) - ak.cos(x),
        lambda v: math.sin(v) - math.cos(v),
        lambda sp, x: sp.sin(x) - sp.cos(x),
    ),
    (
        "sin2x=cosx",
        lambda x, p: ak.sin(2 * x) - ak.cos(x),
        lambda v: math.sin(2 * v) - math.cos(v),
        lambda sp, x: sp.sin(2 * x) - sp.cos(x),
    ),
    (
        "sinx*cosx=1/4",
        lambda x, p: ak.sin(x) * ak.cos(x) - p.rational(1, 4),
        lambda v: math.sin(v) * math.cos(v) - 0.25,
        lambda sp, x: sp.sin(x) * sp.cos(x) - sp.Rational(1, 4),
    ),
    (
        "cos^2-sin=1",
        lambda x, p: ak.cos(x) ** 2 - ak.sin(x) - 1,
        lambda v: math.cos(v) ** 2 - math.sin(v) - 1,
        lambda sp, x: sp.cos(x) ** 2 - sp.sin(x) - 1,
    ),
    (
        "sinx+cosx=-1",
        lambda x, p: ak.sin(x) + ak.cos(x) + 1,
        lambda v: math.sin(v) + math.cos(v) + 1,
        lambda sp, x: sp.sin(x) + sp.cos(x) + 1,
    ),
    (
        "tan^2=3",
        lambda x, p: ak.tan(x) ** 2 - 3,
        lambda v: math.tan(v) ** 2 - 3,
        lambda sp, x: sp.tan(x) ** 2 - 3,
    ),
    (
        "cos2x=cosx",
        lambda x, p: ak.cos(2 * x) - ak.cos(x),
        lambda v: math.cos(2 * v) - math.cos(v),
        lambda sp, x: sp.cos(2 * x) - sp.cos(x),
    ),
    (
        "2sin^2(x/2)=1-cos3x... (half angles)",
        lambda x, p: ak.sin(x / 2) - ak.cos(x),
        lambda v: math.sin(v / 2) - math.cos(v),
        lambda sp, x: sp.sin(x / 2) - sp.cos(x),
    ),
    (
        "sqrt3 sin + cos = 1",
        lambda x, p: p.integer(3) ** p.rational(1, 2) * ak.sin(x) + ak.cos(x) - 1,
        lambda v: math.sqrt(3) * math.sin(v) + math.cos(v) - 1,
        lambda sp, x: sp.sqrt(3) * sp.sin(x) + sp.cos(x) - 1,
    ),
]


@pytest.mark.parametrize(("name", "make", "f", "_sym"), CASES, ids=[c[0] for c in CASES])
def test_families_are_solutions_and_complete(name, make, f, _sym):
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([make(x, p)], [x])
    assert sols, name
    members = _members(sols, x, p)
    for m in members:
        assert abs(f(m)) < 1e-9, (name, m)
    for r in _scan_roots(f):
        assert any(abs(r - m) < 1e-7 for m in members), (name, r)


@pytest.mark.parametrize(("name", "make", "f", "sym"), CASES, ids=[c[0] for c in CASES])
def test_matches_sympy_solveset(name, make, f, sym):
    sympy = pytest.importorskip("sympy")
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([make(x, p)], [x])
    sx = sympy.Symbol("x", real=True)
    theirs = sympy.solveset(sym(sympy, sx), sx, sympy.Interval(-10, 10))
    if not isinstance(theirs, sympy.FiniteSet):
        pytest.skip(f"SymPy returned {type(theirs).__name__}")
    want = sorted(float(t) for t in theirs)
    got = sorted(m for m in _members(sols, x, p) if -10 <= m <= 10)
    for w in want:
        assert any(abs(w - g) < 1e-9 for g in got), (name, w)
    for g in got:
        assert any(abs(w - g) < 1e-9 for w in want), (name, g)


def test_sin_equals_cos_is_one_pi_periodic_family():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([ak.sin(x) - ak.cos(x)], [x])
    assert len(sols) == 1
    (k,) = _indices(p)
    a = ak.eval_expr(sols[0][x], {k: 0.0})
    b = ak.eval_expr(sols[0][x], {k: 1.0})
    assert abs(b - a - math.pi) < 1e-12
    assert abs(math.tan(a) - 1) < 1e-12


def test_two_sin_squared_merges_into_two_families():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([2 * ak.sin(x) ** 2 - 1], [x])
    assert len(sols) == 2


def test_trig_polynomial_numeric_reports_k0_members():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([ak.sin(x) - ak.cos(x)], [x], numeric=True)
    assert len(sols) == 1
    assert abs(math.tan(sols[0][x]) - 1) < 1e-12
    assert any("periodic" in c for c in ak.solve_side_conditions())


def test_no_real_solution():
    p = ak.ExprPool()
    x = p.symbol("x")
    assert ak.solve([ak.sin(x) ** 2 + ak.cos(x) - 3], [x]) == []


def test_identity_is_refused():
    p = ak.ExprPool()
    x = p.symbol("x")
    with pytest.raises(ak.SolverError) as ei:
        ak.solve([ak.sin(x) ** 2 + ak.cos(x) ** 2 - 1], [x])
    assert ei.value.code == "E-SOLVE-001"


def test_tangent_undefined_family_is_dropped():
    # tan(x)·cos(x) = sin(x) holds wherever tan is defined, so after clearing
    # the denominator it would be an identity; tan(x) = sin(x)/cos(x) - 1 ...
    # here: tan x = 2 sin x  →  sin x = 0 or cos x = 1/2, never cos x = 0.
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([ak.tan(x) - 2 * ak.sin(x)], [x])
    members = _members(sols, x, p)
    for m in members:
        assert abs(math.cos(m)) > 1e-6
        assert abs(math.tan(m) - 2 * math.sin(m)) < 1e-9
    for r in _scan_roots(lambda v: math.tan(v) - 2 * math.sin(v)):
        assert any(abs(r - m) < 1e-7 for m in members)


# ---------------------------------------------------------------------------
# Systems through invertible functions
# ---------------------------------------------------------------------------


def test_exp_system():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sols = ak.solve([ak.exp(x) + y - 3, ak.exp(x) - y - 1], [x, y])
    assert len(sols) == 1
    xv = ak.eval_expr(sols[0][x], {})
    yv = ak.eval_expr(sols[0][y], {})
    assert abs(xv - math.log(2)) < 1e-12
    assert abs(yv - 1) < 1e-12


def test_exp_system_with_multiple_and_negative_root():
    # exp(2x) - y = 0, exp(x) + y = 6  →  u² + u − 6 = 0, u = 2 (u = −3 has no
    # real logarithm).
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sols = ak.solve([ak.exp(2 * x) - y, ak.exp(x) + y - 6], [x, y])
    assert len(sols) == 1
    xv = ak.eval_expr(sols[0][x], {})
    yv = ak.eval_expr(sols[0][y], {})
    assert abs(math.exp(2 * xv) - yv) < 1e-12
    assert abs(math.exp(xv) + yv - 6) < 1e-12


def test_log_system():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sols = ak.solve([ak.log(x) + y - 2, ak.log(x) - y], [x, y])
    assert len(sols) == 1
    assert abs(ak.eval_expr(sols[0][x], {}) - math.e) < 1e-12


def test_trig_system_families():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sols = ak.solve([ak.sin(x) + y - 1, ak.sin(x) - y], [x, y])
    assert len(sols) == 2
    (k,) = _indices(p)
    for s in sols:
        assert abs(ak.eval_expr(s[y], {}) - 0.5) < 1e-12
        for kv in range(-3, 4):
            assert abs(math.sin(ak.eval_expr(s[x], {k: float(kv)})) - 0.5) < 1e-12
    num = ak.solve([ak.sin(x) + y - 1, ak.sin(x) - y], [x, y], numeric=True)
    assert sorted(round(s[x], 12) for s in num) == sorted(
        round(v, 12) for v in (math.pi / 6, 5 * math.pi / 6)
    )


def test_unknown_inside_and_outside_still_refuses():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(ak.SolverError):
        ak.solve([ak.exp(x) + x - y, y - 1], [x, y])


# ---------------------------------------------------------------------------
# Complex exponential families
# ---------------------------------------------------------------------------


def _complex_members(sols, x, pool, ks=(-4, 5)):
    idx = _indices(pool)
    out = []
    for s in sols:
        for kv in range(*ks):
            env = {k: float(kv) for k in idx}
            r = ak.evaluate(s[x], env, mode="complex")
            assert r.status == "ok", r
            out.append(complex(r.value))
    return out


def test_exp_complex_family():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([ak.exp(x) - 3], [x], domain="complex")
    assert len(sols) == 1
    ms = _complex_members(sols, x, p)
    for m in ms:
        assert abs(cmath.exp(m) - 3) < 1e-9
    for n in range(-4, 5):
        want = complex(math.log(3), 2 * math.pi * n)
        assert any(abs(want - m) < 1e-9 for m in ms)
    # The real default is unchanged.
    real = ak.solve([ak.exp(x) - 3], [x])
    assert len(real) == 1
    assert abs(ak.eval_expr(real[0][x], {}) - math.log(3)) < 1e-12


def test_exp_negative_complex_family_vs_sympy():
    sympy = pytest.importorskip("sympy")
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([ak.exp(2 * x) + 4], [x], domain="complex")
    ms = _complex_members(sols, x, p, ks=(-6, 7))
    for m in ms:
        assert abs(cmath.exp(2 * m) + 4) < 1e-9
    sx = sympy.Symbol("x")
    theirs = sympy.solveset(sympy.exp(2 * sx) + 4, sx, sympy.S.Complexes)
    pts = []
    for part in theirs.args if isinstance(theirs, sympy.Union) else [theirs]:
        lam, _ = part.args
        for n in range(-2, 3):
            pts.append(complex(sympy.N(lam(n))))
    assert pts
    for w in pts:
        assert any(abs(w - m) < 1e-9 for m in ms), w
    # ...and over the reals there is nothing.
    assert ak.solve([ak.exp(2 * x) + 4], [x]) == []


def test_complex_symbol_selects_complex_mode():
    p = ak.ExprPool()
    z = p.symbol("z", "complex")
    sols = ak.solve([ak.exp(z) - 2], [z])
    ms = _complex_members(sols, z, p)
    assert any(abs(m.imag) > 1 for m in ms)


def test_complex_sin_out_of_range():
    p = ak.ExprPool()
    x = p.symbol("x")
    assert ak.solve([ak.sin(x) - 2], [x]) == []
    sols = ak.solve([ak.sin(x) - 2], [x], domain="complex")
    assert len(sols) == 2
    for m in _complex_members(sols, x, p):
        assert abs(cmath.sin(m) - 2) < 1e-9


def test_complex_lambert_refuses():
    p = ak.ExprPool()
    x = p.symbol("x")
    with pytest.raises(ak.SolverError):
        ak.solve([x * ak.exp(x) - 1], [x], domain="complex")


def test_complex_exp_system():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sols = ak.solve([ak.exp(x) + y - 3, ak.exp(x) - y - 1], [x, y], domain="complex")
    ms = _complex_members(sols, x, p)
    for m in ms:
        assert abs(cmath.exp(m) - 2) < 1e-9
    assert any(abs(m.imag) > 1 for m in ms)


def test_complex_lambert_refuses_even_with_two_real_branches():
    # Over the reals x·eˣ = −1/4 has two roots (W₀ and W₋₁); over ℂ it has one
    # on every branch, which cannot be listed.
    p = ak.ExprPool()
    x = p.symbol("x")
    eq = x * ak.exp(x) + p.rational(1, 4)
    real = ak.solve([eq], [x], numeric=True)
    assert len(real) == 2
    for s in real:
        assert abs(s[x] * math.exp(s[x]) + 0.25) < 1e-12
    with pytest.raises(ak.SolverError):
        ak.solve([eq], [x], domain="complex")


def test_interval_and_complex_domain_conflict():
    p = ak.ExprPool()
    x = p.symbol("x")
    with pytest.raises(ValueError):
        ak.solve([ak.cos(x) - x], [x], numeric=True, interval=(0, 1), domain="complex")


def test_complex_numeric_reports_k0_member():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([ak.exp(x) + 2], [x], numeric=True, domain="complex")
    assert len(sols) == 1
    v = complex(sols[0][x])
    assert abs(cmath.exp(v) + 2) < 1e-12
    assert any("periodic" in c for c in ak.solve_side_conditions())
