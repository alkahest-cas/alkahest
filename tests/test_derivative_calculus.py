"""Derivative nodes through calculus and ODE input (follow-up F9 to G3/G9).

* ``integrate`` of exact derivatives of undefined functions (``f' -> f``,
  chain-rule reversal, polynomial-by-parts) checked against SymPy on concrete
  instantiations of ``f``;
* ``series`` of expressions with ``f(t)`` about a point (Taylor coefficients in
  ``f(a), f'(a), ...``) checked against SymPy's ``series``;
* ``limit`` of opaque functions: refused unless declared continuous;
* ``dsolve_system`` / ``ODE.from_equations`` taking equations written with
  ``y(t)`` and ``Derivative(y(t), t)``;
* real-form exits (``dsolve_system``, ``eigenvals``, ``solve``).
"""

from __future__ import annotations

import random

import alkahest as ak
import alkahest.experimental as akx
import pytest

sympy = pytest.importorskip("sympy")


@pytest.fixture
def pool():
    return ak.ExprPool()


def _instantiate(expr, binds_funcs, pool, t):
    """Replace each undefined function in ``expr`` by a concrete body."""
    out = expr
    for f, body in binds_funcs:
        out = ak.replace_function(out, f, body)
    return out


# ---------------------------------------------------------------------------
# integrate
# ---------------------------------------------------------------------------

F_BODY = lambda t: ak.sin(t) * t + ak.exp(t / 3)  # noqa: E731  (smooth, nonzero)


def _check_antiderivative(pool, integrand, antiderivative, t, f):
    """d/dt(antiderivative) == integrand once f is a concrete function."""
    d = ak.diff(antiderivative, t).value
    diff_expr = d - integrand
    inst = ak.replace_function(diff_expr, f, F_BODY(t))
    rng = random.Random(7)
    for _ in range(5):
        x = rng.uniform(0.3, 1.7)
        assert abs(ak.eval_expr(inst, {t: x})) < 1e-9


@pytest.mark.parametrize(
    "build",
    [
        lambda f, fp, fpp, t: fp,
        lambda f, fp, fpp, t: fpp,
        lambda f, fp, fpp, t: f * fp,
        lambda f, fp, fpp, t: fp / f,
        lambda f, fp, fpp, t: fp * ak.exp(f),
        lambda f, fp, fpp, t: fpp * fp,
        lambda f, fp, fpp, t: 3 * fp * ak.cos(f),
        lambda f, fp, fpp, t: fp * (1 + f) ** 2,
        lambda f, fp, fpp, t: fp / (1 + f**2),
        lambda f, fp, fpp, t: t * fpp,
        lambda f, fp, fpp, t: (t**2 + 1) * ak.diff(fpp, t).value,
    ],
)
def test_integral_of_exact_derivative(pool, build):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    fp = ak.diff(f, t).value
    fpp = ak.diff(fp, t).value
    integrand = build(f, fp, fpp, t)
    got = ak.integrate(integrand, t).value
    _check_antiderivative(pool, integrand, got, t, f)


def test_integrate_derivative_gives_the_function(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    assert ak.integrate(ak.diff(f, t).value, t).value == f
    fpp = ak.diff(ak.diff(f, t).value, t).value
    assert ak.integrate(fpp, t).value == ak.diff(f, t).value


def test_chain_rule_reversal_with_inner_function(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    # d/dt f(t^2) = 2 t f'(t^2)
    ft2 = pool.func("f", [t**2])
    integrand = ak.diff(ft2, t).value
    assert ak.integrate(integrand, t).value == ft2
    # g'(t) h(g(t)) with g = f(t): f' * sin(f)
    fp = ak.diff(f, t).value
    got = ak.integrate(fp * ak.sin(f), t).value
    assert "cos" in str(got)
    _check_antiderivative(pool, fp * ak.sin(f), got, t, f)


def test_integral_matches_sympy_definite(pool):
    sp_t = sympy.Symbol("t")
    t = pool.symbol("t")
    f = pool.func("f", [t])
    fp = ak.diff(f, t).value
    integrand = fp * ak.exp(f) + t * ak.diff(fp, t).value
    F = ak.integrate(integrand, t).value
    body = F_BODY(t)
    a, b = 0.4, 1.3
    ours = ak.eval_expr(ak.replace_function(F, f, body), {t: b}) - ak.eval_expr(
        ak.replace_function(F, f, body), {t: a}
    )
    sp_f = sympy.sin(sp_t) * sp_t + sympy.exp(sp_t / 3)
    sp_integrand = sympy.diff(sp_f, sp_t) * sympy.exp(sp_f) + sp_t * sympy.diff(sp_f, sp_t, 2)
    want = float(sympy.Integral(sp_integrand, (sp_t, a, b)).evalf(20))
    assert ours == pytest.approx(want, rel=1e-9)


def test_definite_integral_is_difference_of_endpoint_values(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    fp = ak.diff(f, t).value
    r = ak.integrate(fp * f, t, pool.integer(1), pool.integer(2)).value
    f1, f2 = pool.func("f", [pool.integer(1)]), pool.func("f", [pool.integer(2)])
    assert ak.simplify(r - (f2**2 - f1**2) / 2).value == pool.integer(0)


@pytest.mark.parametrize("which", ["f", "t*f", "f*f", "sin(f)"])
def test_integral_without_closed_form_is_refused(pool, which):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    expr = {"f": f, "t*f": t * f, "f*f": f * f, "sin(f)": ak.sin(f)}[which]
    with pytest.raises(ak.IntegrationError):
        ak.integrate(expr, t)


def test_improper_integral_of_undefined_function_is_refused(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    with pytest.raises(ak.IntegrationError):
        ak.integrate(ak.diff(f, t).value, t, pool.integer(0), pool.pos_infinity())


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------


def test_taylor_series_of_f_in_terms_of_derivatives_at_a(pool):
    t, a = pool.symbol("t"), pool.symbol("a")
    f = pool.func("f", [t])
    s = ak.series(f, t, a, 4)
    # Compare with SymPy's series(f(x), x, a, 4) after instantiating f.
    sx, sa = sympy.symbols("x a")
    want = sympy.series(sympy.Function("f")(sx), sx, sa, 4).removeO()
    body = lambda u: ak.sin(u) + u**2  # noqa: E731
    sp_body = lambda u: sympy.sin(u) + u**2  # noqa: E731
    ours = ak.replace_function(s.truncated(), f, body(t))
    theirs = want.replace(sympy.Function("f"), sp_body).doit()
    for av, tv in [(0.3, 0.55), (1.1, 0.9)]:
        got = ak.eval_expr(ours, {a: av, t: tv})
        ref = float(theirs.subs({sx: tv, sa: av}).evalf())
        assert got == pytest.approx(ref, rel=1e-9)


def test_series_with_f_and_derivative_nodes_has_the_right_coefficients(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    zero = pool.integer(0)
    s = ak.series(ak.exp(f) * ak.sin(t), t, zero, 3).truncated()
    # exp(f(t)) sin(t) = e^{f(0)} t + e^{f(0)} f'(0) t^2 + O(t^3)
    f0 = pool.func("f", [zero])
    f0p = ak.subs(ak.diff(f, t).value, {t: zero})
    want = ak.exp(f0) * t + ak.exp(f0) * f0p * t**2
    assert ak.simplify(s - want).value == pool.integer(0)


# ---------------------------------------------------------------------------
# limit
# ---------------------------------------------------------------------------


def test_limit_of_undefined_function_is_refused_unless_declared(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    two = pool.integer(2)
    with pytest.raises(ak.LimitError) as exc:
        ak.limit(f * ak.cos(t), t, two)
    assert exc.value.code == "E-LIMIT-007"
    got = ak.limit(f * ak.cos(t), t, two, continuous=[f]).value
    want = pool.func("f", [two]) * ak.cos(two)
    assert ak.simplify(got - want).value == pool.integer(0)


def test_limit_refuses_derivative_quotient_even_when_continuous(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    a = pool.integer(1)
    q = (f - pool.func("f", [a])) / (t - a)
    with pytest.raises(ak.LimitError) as exc:
        ak.limit(q, t, a, continuous=[f])
    assert exc.value.code == "E-LIMIT-007"


def test_limit_of_derivative_node_needs_its_own_declaration(pool):
    t = pool.symbol("t")
    f = pool.func("f", [t])
    fp = ak.diff(f, t).value
    with pytest.raises(ak.LimitError):
        ak.limit(f + fp, t, pool.integer(0), continuous=[f])
    ak.limit(f + fp, t, pool.integer(0), continuous=[f, fp])


def test_limit_without_functions_unchanged(pool):
    x = pool.symbol("x")
    assert ak.limit(ak.sin(x) / x, x, pool.integer(0)).value == pool.integer(1)
    assert ak.limit(ak.sin(x) / x, x, pool.integer(0), continuous=[]).value == pool.integer(1)


# ---------------------------------------------------------------------------
# dsolve_system / ODE from y(t) equations
# ---------------------------------------------------------------------------


def test_dsolve_system_scalar_second_order_in_y_of_t(pool):
    t = pool.symbol("t")
    y = pool.func("y", [t])
    yp = ak.diff(y, t).value
    ypp = ak.diff(yp, t).value
    eq = ypp + 3 * yp + 2 * y
    sol = akx.dsolve_system([eq], functions=[y], t=t)
    yt = sol["functions"][y]
    # numeric residual of y'' + 3 y' + 2 y = 0 by finite differences
    c = dict(zip(sol["constants"], (0.7, -1.3)))
    h = 1e-4

    def val(x):
        return ak.eval_expr(yt, {t: x, **c})

    for x0 in (0.2, 0.9, 1.5):
        ypp_n = (val(x0 + h) - 2 * val(x0) + val(x0 - h)) / h**2
        yp_n = (val(x0 + h) - val(x0 - h)) / (2 * h)
        assert abs(ypp_n + 3 * yp_n + 2 * val(x0)) < 1e-4
    # the symbol-form result (hand-lowered x' = v, v' = -3v - 2x) is the same
    ode = ak.ODE.from_equations([eq], [y], t)
    sol2 = akx.dsolve_system(ode)
    assert len(sol2["y_of_t"]) == 2
    xs, vs = pool.symbol("x"), pool.symbol("v")
    hand = akx.dsolve_system(ak.ODE([xs, vs], [vs, -3 * vs - 2 * xs], t))
    for ours, theirs in zip(sol["functions"].values(), hand["y_of_t"]):
        for t0 in (0.3, 0.8):
            a = ak.eval_expr(ours, {t: t0, **c})
            b = ak.eval_expr(theirs, {t: t0, **dict(zip(hand["constants"], (0.7, -1.3)))})
            assert a == pytest.approx(b, rel=1e-9)


def test_dsolve_system_coupled_second_order_matches_symbol_form(pool):
    t = pool.symbol("t")
    x = pool.func("x", [t])
    z = pool.func("z", [t])
    xpp = ak.diff(ak.diff(x, t).value, t).value
    zpp = ak.diff(ak.diff(z, t).value, t).value
    # x'' = -2x + z,  z'' = x - 2z  (normal modes)
    eqs = [xpp + 2 * x - z, zpp - x + 2 * z]
    sol = akx.dsolve_system(eqs, functions=[x, z], t=t)
    assert set(sol["functions"].keys()) >= {x, z}
    consts = sol["constants"]
    assert len(consts) == 4
    rng = random.Random(3)
    c = {k: rng.uniform(-1, 1) for k in consts}
    h = 1e-4

    def at(expr, tv):
        return ak.eval_expr(expr, {t: tv, **c})

    for t0 in (0.3, 1.1):
        for u, other in ((x, z), (z, x)):
            f = sol["functions"][u]
            g = sol["functions"][other]
            upp = (at(f, t0 + h) - 2 * at(f, t0) + at(f, t0 - h)) / h**2
            # u'' = -2u + other
            assert abs(upp - (-2 * at(f, t0) + at(g, t0))) < 1e-4


def test_dsolve_system_first_order_system_in_y_of_t_with_forcing(pool):
    t = pool.symbol("t")
    p = pool.func("p", [t])
    q = pool.func("q", [t])
    pp = ak.diff(p, t).value
    qp = ak.diff(q, t).value
    eqs = [pp - q, qp + p - ak.sin(2 * t)]
    sol = akx.dsolve_system(eqs, functions=[p, q], t=t)
    c = dict.fromkeys(sol["constants"], 0.5)
    fp, fq = sol["functions"][p], sol["functions"][q]
    h = 1e-5
    for t0 in (0.4, 1.2):
        dp = (ak.eval_expr(fp, {t: t0 + h, **c}) - ak.eval_expr(fp, {t: t0 - h, **c})) / (2 * h)
        assert dp == pytest.approx(ak.eval_expr(fq, {t: t0, **c}), abs=1e-5)


def test_dsolve_system_equation_errors(pool):
    t = pool.symbol("t")
    y = pool.func("y", [t])
    yp = ak.diff(y, t).value
    with pytest.raises(ValueError):
        akx.dsolve_system([yp - y])  # no functions / t
    with pytest.raises(ak.OdeError) as exc:
        # y never appears differentiated: algebraic
        akx.dsolve_system([y - 1], functions=[y], t=t)
    assert exc.value.code == "E-ODE-030"
    z = pool.func("z", [t])
    zp = ak.diff(z, t).value
    with pytest.raises(ak.OdeError) as exc:
        # singular leading matrix: y' + z' = 0 twice
        akx.dsolve_system([yp + zp, yp + zp + y], functions=[y, z], t=t)
    assert exc.value.code == "E-ODE-030"


def test_dsolve_system_symbolic_mass_reports_condition(pool):
    t = pool.symbol("t")
    m, k = pool.symbol("m"), pool.symbol("k")
    x = pool.func("x", [t])
    xpp = ak.diff(ak.diff(x, t).value, t).value
    sol = akx.dsolve_system([m * xpp + k * x], functions=[x], t=t)
    assert any("m" in c for c in sol["side_conditions"])


# ---------------------------------------------------------------------------
# real-form exits
# ---------------------------------------------------------------------------


def _osc_ode(pool):
    t = pool.symbol("t")
    x, v = pool.symbol("x"), pool.symbol("v")
    w = pool.symbol("w", "positive")
    return ak.ODE([x, v], [v, -(w**2) * x], t), t, w


def test_dsolve_system_real_form_is_real_and_opt_out_is_complex(pool):
    ode, t, w = _osc_ode(pool)
    real = akx.dsolve_system(ode)
    raw = akx.dsolve_system(ode, real_form=False)
    assert "I" not in str(real["fundamental_matrix"])
    assert "cos" in str(real["fundamental_matrix"])
    # real-mode evaluation equals the complex-mode real part
    rng = random.Random(1)
    consts = real["constants"]
    for row_r in real["y_of_t"]:
        binds = {t: 0.8, w: 1.7, **{c: rng.uniform(-1, 1) for c in consts}}
        r = ak.eval_expr(row_r, binds)
        z = ak.evaluate(row_r, binds, mode="complex")
        assert z.value.real == pytest.approx(r, abs=1e-12)
        assert abs(z.value.imag) < 1e-12
    # the raw form is the same function
    for row_r, row_c in zip(real["fundamental_matrix"], raw["fundamental_matrix"]):
        for er, ec in zip(row_r, row_c):
            b = {t: 0.8, w: 1.7}
            zc = ak.evaluate(ec, b, mode="complex").value
            assert zc.real == pytest.approx(ak.eval_expr(er, b), abs=1e-12)


def test_dsolve_system_complex_symbol_data_is_not_rewritten(pool):
    t = pool.symbol("t")
    x = pool.symbol("x")
    c = pool.symbol("c", "complex")
    ode = ak.ODE([x], [c * x], t)
    sol = akx.dsolve_system(ode)
    assert "exp" in str(sol["fundamental_matrix"])


def test_eigenvals_real_form(pool):
    a, b = pool.symbol("a"), pool.symbol("b")
    m = ak.Matrix([[a, b], [b, -a]])
    ev = m.eigenvals()
    raw = m.eigenvals(real_form=False)
    # sqrt(-4*(-a^2-b^2))/2 is sqrt(a^2+b^2): real, so the radical of a negative is gone
    assert all("-1 * a^2" not in str(k) for k in ev)
    assert len(ev) == len(raw) == 2
    for av, bv in [(0.7, 1.9), (-1.2, 0.4)]:
        got = sorted(ak.eval_expr(k, {a: av, b: bv}) for k in ev)
        want = sorted(ak.evaluate(k, {a: av, b: bv}, mode="complex").value.real for k in raw)
        assert got == pytest.approx(want, abs=1e-12)
    # a genuinely complex spectrum is untouched
    zero, one = pool.integer(0), pool.integer(1)
    rot = ak.Matrix([[zero, one], [-one, zero]])
    assert set(rot.eigenvals()) == set(rot.eigenvals(real_form=False))


def test_eigenvals_sqrt_of_square_loses_radical(pool):
    # symmetric 2x2 with a perfect-square discriminant
    a = pool.symbol("a", "positive")
    m = ak.Matrix([[0, 2 * a], [2 * a, 0]])
    ev = m.eigenvals()
    assert all("sqrt" not in str(k) and "^(1/2)" not in str(k) for k in ev)


def test_solve_results_are_realified_when_proved_real(pool):
    a = pool.symbol("a", "positive")
    x = pool.symbol("x")
    sols = ak.solve([x**2 - 4 * a**2], [x])
    shown = {str(s[x]) for s in sols}
    assert all("sqrt" not in s and "^(1/2)" not in s for s in shown)
    for s in sols:
        v = ak.eval_expr(s[x], {a: 1.3})
        assert abs(abs(v) - 2.6) < 1e-12
    raw = ak.solve([x**2 - 4 * a**2], [x], real_form=False)
    assert len(raw) == len(sols)


def test_solve_leaves_genuinely_complex_roots_alone(pool):
    x = pool.symbol("x")
    sols = ak.solve([x**2 + 1], [x])
    raw = ak.solve([x**2 + 1], [x], real_form=False)
    assert len(sols) == 2
    assert [s[x] for s in sols] == [s[x] for s in raw]
    assert all("sqrt(-4)" in str(s[x]) for s in sols)
