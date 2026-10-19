"""F8 -- non-smooth functions: abs / sign / min / max / floor / ceil / round.

* derivatives (``diff``, ``diff_forward``, ``symbolic_grad``) are the
  almost-everywhere derivative, with the kink set stated as a side condition;
* ``rewrite_piecewise`` lowers ``abs`` / ``min`` / ``max`` to ``Piecewise``;
* ``integrate`` splits at the breakpoints of ``Piecewise`` integrands and glues
  a continuous antiderivative (checked against mpmath quadrature);
* ``emit_c_expr`` compiles ``Piecewise`` / ``abs`` / ``min`` / ``max`` (any
  arity) and agrees with ``eval_expr``.
"""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess

import alkahest as ak
import mpmath
import pytest


@pytest.fixture
def pool():
    return ak.ExprPool()


def const(x, n):  # an integer constant in x's pool
    return x._pool.integer(n)


def fd(expr, x, at, h=1e-6):
    hi = ak.eval_expr(expr, {x: at + h})
    lo = ak.eval_expr(expr, {x: at - h})
    return (hi - lo) / (2 * h)


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def _forms(expr, x):
    """The three differentiation routes, as expressions."""
    return {
        "diff": ak.diff(expr, x).value,
        "diff_forward": ak.diff_forward(expr, x).value,
        "grad": ak.symbolic_grad(expr, [x])[0],
    }


@pytest.mark.parametrize(
    ("build", "points"),
    [
        (lambda x: ak.abs(x**2 - 1), [-2.3, -0.4, 0.3, 1.7]),
        (lambda x: ak.abs(ak.sin(x)) * x, [0.7, 2.0, 4.0]),
        (lambda x: ak.max(x, x**2), [-1.5, 0.4, 2.5]),
        (lambda x: ak.min(x, x**2), [-1.5, 0.4, 2.5]),
        (lambda x: ak.max(x, ak.max(x**2, 2 * x + 5)), [-4.0, 0.4, 3.5]),
        (lambda x: x * ak.floor(x) + ak.sin(x), [0.3, 1.4, -2.6]),
        (lambda x: ak.ceil(3 * x) * ak.exp(x), [0.1, 1.3]),
        (lambda x: ak.round(x) + x**3, [0.2, 0.9, -1.7]),
        (lambda x: ak.sign(x) * x**2, [-1.2, 0.8]),
    ],
)
def test_derivatives_match_finite_differences_away_from_kinks(pool, build, points):
    x = pool.symbol("x")
    expr = build(x)
    for name, d in _forms(expr, x).items():
        for at in points:
            got = ak.eval_expr(d, {x: at})
            assert got == pytest.approx(fd(expr, x, at), rel=1e-5, abs=1e-5), (name, at)


def test_abs_derivative_is_sign_with_side_condition(pool):
    x = pool.symbol("x")
    r = ak.diff(ak.abs(x), x)
    assert str(r.value) == "sign(x)"
    assert r.side_conditions == ["x ≠ 0"]
    # the condition is also on the derivation steps
    assert any(s["side_conditions"] for s in r.steps)


@pytest.mark.parametrize("fn", [ak.floor, ak.ceil, ak.round, ak.sign])
def test_piecewise_constant_derivative_is_zero_only_with_a_stated_condition(pool, fn):
    x = pool.symbol("x")
    r = ak.diff(fn(x), x)
    assert str(r.value) == "0"
    assert r.side_conditions, "a zero derivative must carry its side condition"
    # a constant argument is smooth: no condition
    assert ak.diff(fn(pool.integer(3)), x).side_conditions == []


def test_floor_condition_names_the_integers(pool):
    x = pool.symbol("x")
    conds = ak.diff(ak.floor(x), x).side_conditions
    assert len(conds) == 1
    assert "floor(x)" in conds[0]


def test_min_max_are_piecewise_and_conditioned(pool):
    x = pool.symbol("x")
    for fn in (ak.min, ak.max):
        r = ak.diff(fn(x, x**2), x)
        assert "Piecewise" in str(r.value)
        assert r.side_conditions
        rf = ak.diff_forward(fn(x, x**2), x)
        assert "Piecewise" in str(rf.value)
        assert rf.side_conditions


def test_chain_rule_through_abs(pool):
    x = pool.symbol("x")
    e = ak.abs(ak.sin(x)) ** 2 + ak.abs(x - 3) * x
    d = ak.diff(e, x).value
    for at in (0.5, 2.0, 4.5):
        assert ak.eval_expr(d, {x: at}) == pytest.approx(fd(e, x, at), rel=1e-5)


def test_smooth_derivatives_unchanged(pool):
    x = pool.symbol("x")
    assert ak.diff(ak.sin(x), x).side_conditions == []
    assert ak.diff(x**3 + ak.exp(x), x).side_conditions == []


def test_multivariate_reverse_gradient(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    e = ak.abs(x - y) * y + ak.max(x, 2 * y)
    gx, gy = ak.symbolic_grad(e, [x, y])
    for px, py in [(2.0, 0.5), (-1.0, 3.0), (4.0, 1.0)]:
        env = {x: px, y: py}
        h = 1e-6
        fx = (ak.eval_expr(e, {x: px + h, y: py}) - ak.eval_expr(e, {x: px - h, y: py})) / (2 * h)
        fy = (ak.eval_expr(e, {x: px, y: py + h}) - ak.eval_expr(e, {x: px, y: py - h})) / (2 * h)
        assert ak.eval_expr(gx, env) == pytest.approx(fx, rel=1e-5, abs=1e-6)
        assert ak.eval_expr(gy, env) == pytest.approx(fy, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# rewrite_piecewise
# ---------------------------------------------------------------------------


def test_rewrite_piecewise_lowers_and_agrees_pointwise(pool):
    x = pool.symbol("x")
    for e in [
        ak.abs(x - 1),
        ak.min(x, x**2),
        ak.max(x, ak.max(x**2, 3 - x)),
        ak.sign(x),
        ak.abs(ak.abs(x) - 2) + x,
    ]:
        low = ak.rewrite_piecewise(e)
        assert "Piecewise" in str(low)
        for head in ("abs(", "min(", "max("):
            assert head not in str(low)
        for v in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0]:
            assert ak.eval_expr(low, {x: v}) == pytest.approx(ak.eval_expr(e, {x: v}))


def test_rewrite_piecewise_leaves_smooth_expressions(pool):
    x = pool.symbol("x")
    e = ak.sin(x) + x**2
    assert ak.rewrite_piecewise(e) == e


# ---------------------------------------------------------------------------
# integrate
# ---------------------------------------------------------------------------


def test_definite_abs_five_halves(pool):
    x = pool.symbol("x")
    r = ak.integrate(ak.abs(x), x, pool.integer(-1), pool.integer(2))
    assert str(r.value) == "5/2"


@pytest.mark.parametrize(
    ("build", "a", "b"),
    [
        (lambda x: ak.max(x, x**2), -1, 2),
        (lambda x: ak.min(x, x**2), -2, 3),
        (lambda x: ak.abs(x**2 - 2), 0, 2),
        (lambda x: ak.abs(x**2 - 1) * x, -2, 3),
        (lambda x: ak.abs(x - 1) + ak.abs(x + 1), -3, 4),
        (lambda x: ak.max(x**2 - 2, const(x, 0)), -3, 3),
        (lambda x: ak.abs(2 * x - 1) * ak.exp(-x), -1, 2),
        (lambda x: ak.abs(x**2 - x - 1), -1, 3),
        (lambda x: ak.min(const(x, 1), ak.abs(x)), -2, 2),
    ],
)
def test_definite_integrals_match_mpmath(pool, build, a, b):
    x = pool.symbol("x")
    e = build(x)
    r = ak.integrate(e, x, pool.integer(a), pool.integer(b)).value
    got = ak.eval_expr(r, {})
    f = lambda t: ak.eval_expr(e, {x: float(t)})  # noqa: E731
    # split at all the kinks mpmath would otherwise have to find
    kinks = sorted(
        (
            -math.sqrt(2),
            math.sqrt(2),
            -1,
            0,
            1,
            2,
            0.5,
            (1 - math.sqrt(5)) / 2,
            (1 + math.sqrt(5)) / 2,
        )
    )
    pts = [a] + [k for k in kinks if a < k < b] + [b]
    want = mpmath.quad(f, pts)
    assert got == pytest.approx(float(want), rel=1e-9, abs=1e-9)


def test_reversed_bounds_negate(pool):
    x = pool.symbol("x")
    e = ak.max(x, x**2)
    fwd = ak.eval_expr(ak.integrate(e, x, pool.integer(-1), pool.integer(2)).value, {})
    bwd = ak.eval_expr(ak.integrate(e, x, pool.integer(2), pool.integer(-1)).value, {})
    assert fwd == pytest.approx(19 / 6)
    assert bwd == pytest.approx(-19 / 6)


def test_improper_integral_over_pieces(pool):
    x = pool.symbol("x")
    oo = ak.oo if hasattr(ak, "oo") else None
    if oo is None:
        pytest.skip("no oo")
    # int_0^oo min(1, x) e^{-x} dx = 1 - 1/e
    e = ak.min(const(x, 1), x) * ak.exp(-x)
    r = ak.integrate(e, x, pool.integer(0), oo).value
    assert ak.eval_expr(r, {}) == pytest.approx(1 - math.exp(-1), rel=1e-12)


@pytest.mark.parametrize(
    "build",
    [
        lambda x: ak.abs(x),
        lambda x: ak.max(x, x**2),
        lambda x: ak.min(x, x**2),
        lambda x: ak.abs(x**2 - 2),
        lambda x: ak.abs(x - 1) * x,
        lambda x: ak.max(x**2 - 2, const(x, 0)) + ak.abs(x + 1),
        lambda x: ak.abs(x**2 - 1) * ak.exp(x),
    ],
)
def test_indefinite_is_a_continuous_antiderivative(pool, build):
    x = pool.symbol("x")
    e = build(x)
    F = ak.integrate(e, x).value
    assert "Piecewise" in str(F)
    grid = [-3 + 0.0173 * i for i in range(400)]
    vals = [ak.eval_expr(F, {x: g}) for g in grid]
    # continuity: no jump larger than the integrand times the step (+ slack)
    for i in range(len(grid) - 1):
        step = grid[i + 1] - grid[i]
        bound = max(abs(ak.eval_expr(e, {x: g})) for g in (grid[i], grid[i + 1])) * step
        assert abs(vals[i + 1] - vals[i]) <= 3 * bound + 1e-9, (grid[i], vals[i], vals[i + 1])
    # and it is an antiderivative off the kinks
    for g in grid[3::37]:
        near = min(abs(g - k) for k in (-math.sqrt(2), math.sqrt(2), -1, 0, 1, 0.5))
        if near > 1e-3:
            assert fd(F, x, g) == pytest.approx(ak.eval_expr(e, {x: g}), rel=1e-5, abs=1e-5)


def test_explicit_piecewise_integrand(pool):
    x = pool.symbol("x")
    zero, one = pool.integer(0), pool.integer(1)
    pw = ak.piecewise([(pool.lt(x, zero), -x), (pool.lt(x, one), x**2)], pool.integer(1))
    r = ak.integrate(pw, x, pool.integer(-1), pool.integer(3)).value
    assert ak.eval_expr(r, {}) == pytest.approx(0.5 + 1 / 3 + 2)
    F = ak.integrate(pw, x).value
    assert ak.eval_expr(F, {x: 3.0}) - ak.eval_expr(F, {x: -1.0}) == pytest.approx(0.5 + 1 / 3 + 2)


def test_unsupported_conditions_are_refused_not_guessed(pool):
    x = pool.symbol("x")
    # irreducible cubic breakpoint
    with pytest.raises(Exception, match="E-INT"):
        ak.integrate(ak.abs(x**3 + x - 1), x)
    # transcendental condition
    with pytest.raises(Exception, match="E-INT"):
        ak.integrate(ak.abs(ak.sin(x)), x)
    # parametric breakpoint
    a = pool.symbol("a")
    with pytest.raises(Exception, match="E-INT"):
        ak.integrate(ak.abs(x - a), x)


# ---------------------------------------------------------------------------
# emit_c
# ---------------------------------------------------------------------------

CC = next(
    (
        c
        for c in ("/usr/bin/gcc", "/usr/bin/cc", shutil.which("gcc") or "")
        if c and shutil.os.access(c, shutil.os.X_OK)
    ),
    None,
)


def _compile(code, tmp_path, nargs):
    src = tmp_path / "f.c"
    so = tmp_path / "f.so"
    src.write_text("#include <math.h>\n" + code)
    r = subprocess.run(
        [
            CC,
            "-std=c99",
            "-pedantic",
            "-Werror",
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(so),
            str(src),
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + code
    fn = ctypes.CDLL(str(so)).f
    fn.restype = ctypes.c_double
    fn.argtypes = [ctypes.c_double] * nargs
    return fn


@pytest.mark.skipif(CC is None, reason="no C compiler")
@pytest.mark.parametrize(
    "build",
    [
        lambda x: ak.abs(x - 1) + ak.sign(x),
        lambda x: ak.min(x, x**2),
        lambda x: ak.max(x, ak.max(x**2, 2 * x + 5)),
        lambda x: ak.min(x, ak.min(const(x, 3), ak.min(x**2, 1 - x))),
        lambda x: ak.floor(x) + ak.ceil(x) * ak.round(x),
        lambda x: ak.rewrite_piecewise(ak.abs(x) + ak.max(x, const(x, 1))),
        lambda x: ak.piecewise([(x._pool.lt(x, const(x, 0)), -x)], x**2),
    ],
)
def test_emit_c_matches_eval(pool, tmp_path, build):
    x = pool.symbol("x")
    e = build(x)
    code = ak.emit_c_expr(e, x)
    fn = _compile(code, tmp_path, 1)
    for v in [-3.7, -1.0, -0.25, 0.0, 0.5, 1.0, 2.5, 4.0]:
        assert fn(v) == pytest.approx(ak.eval_expr(e, {x: v}), abs=1e-12), (v, code)


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_emit_c_uses_libm_for_abs_min_max(pool):
    x = pool.symbol("x")
    code = ak.emit_c_expr(
        ak.abs(x) + ak.min(x, const(x, 1)) + ak.max(x, ak.max(const(x, 2), const(x, 3))), x
    )
    for helper in ("fabs(", "fmin(", "fmax("):
        assert helper in code
