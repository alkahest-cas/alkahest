"""Multivariate Horner: ``alkahest.horner`` with a list of variables, and the
compile pipeline's polynomial pass (``compile_expr(..., optimize=...)``).
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from fractions import Fraction
from math import comb

import alkahest
import pytest
from alkahest import ExprPool, compile_expr, eval_expr, horner


def _dense(p, x, y, n):
    """(1 + x + y)^n written out as a sum of monomials."""
    terms = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            c = comb(n, i) * comb(n - i, j)
            terms.append(p.integer(c) * x**i * y**j)
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total


def test_horner_single_variable_is_unchanged():
    p = ExprPool()
    x = p.symbol("x")
    e = x**3 + p.integer(2) * x**2 + p.integer(3) * x + p.integer(4)
    h = horner(e, x)
    for v in (0.0, 1.5, -2.25):
        assert eval_expr(h, {x: v}) == eval_expr(e, {x: v})
    # The univariate contract still refuses a second symbol.
    y = p.symbol("y")
    with pytest.raises(ValueError):
        horner(x * y + x, x)


@pytest.mark.parametrize("vars_arg", ["list", "tuple", "none"])
def test_horner_multivariate_matches(vars_arg):
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    e = _dense(p, x, y, 6)
    arg = {"list": [x, y], "tuple": (x, y), "none": None}[vars_arg]
    h = horner(e, arg) if arg is not None else horner(e)
    assert h != e
    for a, b in [(0.3, 0.7), (-1.25, 0.5), (2.0, -3.0)]:
        want = (1 + a + b) ** 6
        got = eval_expr(h, {x: a, y: b})
        assert math.isclose(got, want, rel_tol=1e-13, abs_tol=1e-13)


def test_horner_multivariate_parameters_and_rationals():
    p = ExprPool()
    x, y, a = p.symbol("x"), p.symbol("y"), p.symbol("a")
    e = a * x**2 * y + p.rational(1, 3) * x * y + p.rational(-5, 7) * y**3 + a
    h = horner(e, [x, y])
    env = {x: 0.4, y: -1.1, a: 2.5}
    assert math.isclose(eval_expr(h, env), eval_expr(e, env), rel_tol=1e-14)


def test_horner_multivariate_refuses_non_polynomial():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(ValueError):
        horner(alkahest.sin(x) + y, [x, y])
    with pytest.raises(TypeError):
        horner(x + y, "x")


def _points(n, lo=0.05, hi=0.95, seed=7):
    rng = random.Random(seed)
    return [(rng.uniform(lo, hi), rng.uniform(lo, hi)) for _ in range(n)]


def _exact_dense(a, b, n):
    return (1 + Fraction(a) + Fraction(b)) ** n


def test_compiled_poly20_accuracy_and_opt_out():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    e = _dense(p, x, y, 20)
    fast = compile_expr(e, [x, y], expected_evals=1 << 20)
    exact = compile_expr(e, [x, y], expected_evals=1 << 20, optimize=False)
    worst_fast = worst_exact = 0.0
    for a, b in _points(2000):
        want = float(_exact_dense(a, b, 20))
        worst_fast = max(worst_fast, abs(fast([a, b]) - want) / want)
        worst_exact = max(worst_exact, abs(exact([a, b]) - want) / want)
        # Opting out compiles the sum as written: bit-exact with eval_expr.
        assert exact([a, b]) == eval_expr(e, {x: a, y: b})
    assert worst_fast < 1e-14, (worst_fast, worst_exact)


def test_compiled_vector_and_batch_agree_with_scalar():
    np = pytest.importorskip("numpy")
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    poly = _dense(p, x, y, 10)
    outs = [poly, alkahest.sin(poly), poly * alkahest.exp(x)]
    vec = compile_expr(outs, [x, y])
    xs = np.linspace(0.05, 0.95, 5000)
    ys = np.linspace(0.9, 0.1, 5000)
    rows = alkahest.numpy_eval(vec, xs, ys)
    for k, o in enumerate(outs):
        f = compile_expr(o, [x, y], expected_evals=len(xs))
        single = alkahest.numpy_eval(f, xs, ys)
        ref = np.array([eval_expr(o, {x: a, y: b}) for a, b in zip(xs[:200], ys[:200])])
        assert np.allclose(rows[k], single, rtol=1e-13, atol=0)
        # sin(poly) with poly ~ 10^3: an ulp of the argument is ~1e-13 absolute.
        assert np.allclose(single[:200], ref, rtol=1e-12, atol=1e-10)


def test_small_polynomials_stay_bit_exact():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    e = p.rational(2, 5) * x**3 - p.rational(7, 3) * x**2 * y + y + p.rational(1, 7)
    f = compile_expr(e, [x, y], expected_evals=1 << 20)
    for a, b in _points(200, -2.0, 2.0):
        assert f([a, b]) == eval_expr(e, {x: a, y: b})


def test_environment_flag_disables_the_pass():
    code = (
        "import alkahest as a\n"
        "from math import comb\n"
        "p = a.ExprPool(); x, y = p.symbol('x'), p.symbol('y')\n"
        "e = p.integer(0)\n"
        "terms = [p.integer(comb(12, i) * comb(12 - i, j)) * x**i * y**j "
        "for i in range(13) for j in range(13 - i)]\n"
        "e = terms[0]\n"
        "for t in terms[1:]: e = e + t\n"
        "f = a.compile_expr(e, [x, y], expected_evals=1 << 20)\n"
        "pts = [(0.1 + 0.0137 * k, 0.9 - 0.011 * k) for k in range(60)]\n"
        "print(all(f([u, v]) == a.eval_expr(e, {x: u, y: v}) for u, v in pts))\n"
    )
    env = dict(os.environ, ALKAHEST_JIT_HORNER="0")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True"


def _shared_pairs(p, x, y):
    """Outputs sharing the product x·y·sin(x)·cos(y) next to different factors."""
    s, c = alkahest.sin(x), alkahest.cos(y)
    return [x * y * s * c * f for f in (s, c, alkahest.exp(x * y), x**2)] + [x * y * s + x * y * c]


def test_common_pair_elimination_agrees_and_opts_out():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    outs = _shared_pairs(p, x, y)
    fast = compile_expr(outs, [x, y])
    exact = compile_expr(outs, [x, y], optimize=False)
    for a, b in _points(300, -2.0, 2.0):
        got, plain = fast([a, b]), exact([a, b])
        for k, o in enumerate(outs):
            want = eval_expr(o, {x: a, y: b})
            assert plain[k] == want
            assert abs(got[k] - want) <= 1e-14 * max(1.0, abs(want)), (k, got[k], want)


def test_cpe_environment_flag():
    code = (
        "import alkahest as a\n"
        "p = a.ExprPool(); x, y = p.symbol('x'), p.symbol('y')\n"
        "e = x * y * a.sin(x) + x * y * a.cos(x) + x * y * a.exp(y)\n"
        "f = a.compile_expr(e, [x, y], expected_evals=1 << 20)\n"
        "pts = [(0.1 + 0.0137 * k, 0.9 - 0.011 * k) for k in range(60)]\n"
        "print(all(f([u, v]) == a.eval_expr(e, {x: u, y: v}) for u, v in pts))\n"
    )
    env = dict(os.environ, ALKAHEST_JIT_CPE="0")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True"
