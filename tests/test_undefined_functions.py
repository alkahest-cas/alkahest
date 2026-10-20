"""Undefined functions ``f(t)`` under ``diff`` (report 10-9, G3).

Through 3.12.0 ``diff(f(t), t)`` raised ``E-DIFF-001`` ("cannot differentiate
unknown function"), so Euler–Lagrange for any multibody system, an FLRW metric
with ``a(t)``, and an ODE written as ``y(t)`` all had to be hand-rolled with
symbols.  The derivative is now a canonical ``Derivative(f(u…), k…)`` node
counted per argument slot; these tests check it against SymPy and against
finite differences of concrete instantiations.
"""

from __future__ import annotations

import math
import pickle
import random

import alkahest as ak
import alkahest.experimental as akx
import pytest

sympy = pytest.importorskip("sympy")


@pytest.fixture
def pool():
    return ak.ExprPool()


def _num(expr, binds):
    return ak.eval_expr(expr, binds)


# ---------------------------------------------------------------------------
# The report's repro and the representation
# ---------------------------------------------------------------------------


def test_report_repro_diff_of_applied_function(pool):
    t = pool.symbol("t")
    q = pool.func("q", [t])
    d1 = ak.diff(q, t).value
    assert str(d1) == "Derivative(q(t), t)"
    d2 = ak.diff(d1, t).value
    assert str(d2) == "Derivative(q(t), t, t)"
    # hash-consed: the same node however it is reached
    assert ak.diff(ak.diff(q, t), t).value == d2
    # independent of other variables
    x = pool.symbol("x")
    assert ak.diff(q, x).value == pool.integer(0)


def test_mixed_partials_are_canonical(pool):
    x, t = pool.symbol("x"), pool.symbol("t")
    u = pool.func("u", [x, t])
    a = ak.diff(ak.diff(u, x).value, t).value
    b = ak.diff(ak.diff(u, t).value, x).value
    assert a == b
    assert str(a) == "Derivative(u(x, t), x, t)"


def test_builtins_and_reserved_names_still_refuse(pool):
    x = pool.symbol("x")
    for name in ("zeta_prime", "binomial", "factorial"):
        with pytest.raises(ak.DiffError) as exc:
            ak.diff(pool.func(name, [x]), x)
        assert exc.value.code == "E-DIFF-001"
    with pytest.raises(ak.DiffError):
        ak.diff(pool.func("polygamma", [x, x * x]), x)


# ---------------------------------------------------------------------------
# Chain rule vs SymPy
# ---------------------------------------------------------------------------


def _to_sympy(expr, names):
    """Read alkahest's printed form with SymPy (the spelling is SymPy's)."""
    local = {n: sympy.Function(n) for n in names}
    local.update({s: sympy.Symbol(s) for s in ("t", "x", "y", "_xi_1", "_xi_2")})
    return sympy.sympify(str(expr).replace("^", "**"), locals=local)


@pytest.mark.parametrize(
    ("build", "sym"),
    [
        (lambda p, t, x, f, g: f(t**2), lambda t, x, f, g: f(t**2)),
        (lambda p, t, x, f, g: f(ak.sin(t)) * g(t), lambda t, x, f, g: f(sympy.sin(t)) * g(t)),
        (lambda p, t, x, f, g: f(g(t)), lambda t, x, f, g: f(g(t))),
        (
            lambda p, t, x, f, g: ak.exp(f(t)) * f(t) ** 3,
            lambda t, x, f, g: sympy.exp(f(t)) * f(t) ** 3,
        ),
        (
            lambda p, t, x, f, g: p.func("h", [x * t, t]),
            lambda t, x, f, g: sympy.Function("h")(x * t, t),
        ),
    ],
)
def test_chain_rule_matches_sympy(pool, build, sym):
    t, x = pool.symbol("t"), pool.symbol("x")

    def f(a):
        return pool.func("f", [a])

    def g(a):
        return pool.func("g", [a])

    e = build(pool, t, x, f, g)
    St, Sx = sympy.symbols("t x")
    Sf, Sg = sympy.Function("f"), sympy.Function("g")
    want = sympy.diff(sym(St, Sx, Sf, Sg), St, 2)
    got = ak.diff(ak.diff(e, t).value, t).value
    # The printed form is SymPy's spelling: SymPy reads it.
    ours = _to_sympy(got, ["f", "g", "h"])
    # Compare after instantiating the functions concretely (SymPy cannot
    # identify two `Subs` that differ only in a dummy's name symbolically).
    X, T = sympy.symbols("X T")
    defs = {
        "f": (X, sympy.sin(X) * X + sympy.exp(X / 3)),
        "g": (X, sympy.cos(2 * X) + X**3),
        "h": ((X, T), X**2 * sympy.sin(T) + sympy.cos(X * T)),
    }

    def inst(expr):
        lam = {sympy.Function(n): sympy.Lambda(a, b) for n, (a, b) in defs.items()}
        return expr.subs(lam).doit()

    AX, AT = pool.symbol("X"), pool.symbol("T")
    ak_defs = {
        "f": ([AX], ak.sin(AX) * AX + ak.exp(AX / 3)),
        "g": ([AX], ak.cos(2 * AX) + AX**3),
        "h": ([AX, AT], AX**2 * ak.sin(AT) + ak.cos(AX * AT)),
    }
    ak_inst = got
    for name, (args, body) in ak_defs.items():
        ak_inst = ak.replace_function(ak_inst, pool.func(name, args), body)
    for tv, xv in ((0.3, -0.7), (1.1, 0.4)):
        point = {St: tv, Sx: xv}
        w = float(inst(want).subs(point))
        assert math.isclose(float(inst(ours).subs(point)), w, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(_num(ak_inst, {t: tv, x: xv}), w, rel_tol=1e-9, abs_tol=1e-9)


def test_chain_rule_numerically_against_finite_differences(pool):
    # Instantiate f := sin(X)·exp(T) + X²T and compare d/dt with a central
    # difference of the instantiated expression.
    t, x = pool.symbol("t"), pool.symbol("x")
    X, T = pool.symbol("X"), pool.symbol("T")
    f = pool.func("f", [x * t, ak.cos(t)])
    e = ak.sin(f) * ak.diff(f, x).value + f**2
    body = ak.sin(X) * ak.exp(T) + X**2 * T
    pat = pool.func("f", [X, T])
    for mode in ("diff", "diff_forward", "grad"):
        if mode == "diff":
            d = ak.diff(e, t).value
        elif mode == "diff_forward":
            d = ak.diff_forward(e, t).value
        else:
            d = ak.symbolic_grad(e, [x, t])[1]
        di = ak.replace_function(d, pat, body)
        ei = ak.replace_function(e, pat, body)
        rng = random.Random(7)
        for _ in range(3):
            xv, tv = rng.uniform(-1, 1), rng.uniform(-1, 1)
            h = 1e-6
            fd = (_num(ei, {x: xv, t: tv + h}) - _num(ei, {x: xv, t: tv - h})) / (2 * h)
            assert math.isclose(_num(di, {x: xv, t: tv}), fd, rel_tol=1e-6, abs_tol=1e-6), mode


# ---------------------------------------------------------------------------
# subs / replace_function ("doit")
# ---------------------------------------------------------------------------


def test_subs_definition_differentiates_derivatives(pool):
    t = pool.symbol("t")
    y = pool.func("y", [t])
    ode = ak.diff(ak.diff(y, t).value, t).value + y
    assert ak.simplify(ak.subs(ode, {y: ak.sin(t)})).value == pool.integer(0)
    # subs at a point is structural: y'(0) stays y'(0)
    y0 = ak.subs(ak.diff(y, t).value, {t: pool.integer(0)})
    assert str(y0) == "Subs(Derivative(y(_xi_1), _xi_1), _xi_1, 0)"


def test_replace_function_everywhere(pool):
    t, x = pool.symbol("t"), pool.symbol("x")
    f = lambda a: pool.func("f", [a])  # noqa: E731
    e = ak.diff(f(f(t)) + f(t**2), t).value
    r = ak.simplify(ak.replace_function(e, f(x), ak.exp(x))).value
    want = ak.diff(ak.exp(ak.exp(t)) + ak.exp(t**2), t).value
    assert ak.simplify(r - want).value == pool.integer(0)
    with pytest.raises(ak.DiffError) as exc:
        ak.replace_function(e, f(pool.integer(2)), ak.exp(x))
    assert exc.value.code == "E-DIFF-005"


# ---------------------------------------------------------------------------
# Printing, parsing, pickling
# ---------------------------------------------------------------------------


def test_print_parse_pickle_round_trip(pool):
    t, x = pool.symbol("t"), pool.symbol("x")
    f = pool.func("f", [t])
    u = pool.func("u", [x, t])
    exprs = [
        ak.diff(f, t).value,
        ak.diff(ak.diff(u, x).value, t).value,
        ak.diff(pool.func("f", [t**2]), t).value,
        ak.diff(ak.diff(pool.func("w", [t, t]), t).value, t).value,
    ]
    syms = {"t": t, "x": x, "f": f, "u": u, "w": pool.func("w", [t, t])}
    for e in exprs:
        assert ak.parse(str(e), pool, dict(syms)) == e, str(e)
        assert pickle.loads(pickle.dumps(e)) == e
        assert ak.latex(e)
        assert ak.unicode_str(e)
    assert ak.unicode_str(exprs[0]) == "f′(t)"
    assert ak.latex(exprs[1]).startswith(r"\frac{\partial^{2}}{\partial x \partial t}")
    # An undeclared unknown name is still a parse error.
    with pytest.raises(ak.ParseError):
        ak.parse("g(t)", pool, {"t": t})


def test_evaluation_refuses_derivative_nodes(pool):
    t = pool.symbol("t")
    d = ak.diff(pool.func("f", [t]), t).value
    with pytest.raises(ak.AlkahestError) as exc:
        ak.eval_expr(d, {t: 0.5})
    assert exc.value.code == "E-EVAL-005"


# ---------------------------------------------------------------------------
# Euler–Lagrange vs SymPy
# ---------------------------------------------------------------------------


def _compare_eoms(pool, t, coords, lag, sym_coords, sym_lag, params, sym_params):
    from sympy.calculus.euler import euler_equations

    eoms = ak.euler_lagrange(lag, coords, t)
    flat, mapping = ak.functions_to_symbols(eoms, coords, t)
    # SymPy: Eq(∂L/∂q − d/dt ∂L/∂q̇, 0); ours is the negative.
    St = sympy.Symbol("t")
    sym_eoms = [-eq.lhs for eq in euler_equations(sym_lag, sym_coords, St)]
    rng = random.Random(11)
    for _ in range(5):
        binds, sbinds = {}, {}
        for p, sp in zip(params, sym_params):
            v = rng.uniform(0.5, 2.0)
            binds[p], sbinds[sp] = v, v
        # one value per q, q̇, q̈ — independent, as a state is
        for symbol in mapping.values():
            binds[symbol] = rng.uniform(-1.5, 1.5)
        sym_subs = []  # highest derivative first, so q̈ is not read as d/dt(q̇)
        for qa, qs in zip(coords, sym_coords):
            nodes = [qa, ak.diff(qa, t).value]
            nodes.append(ak.diff(nodes[1], t).value)
            for k in (2, 1, 0):
                v = binds[mapping[nodes[k]]] if nodes[k] in mapping else 0.0
                sym_subs.append((qs.diff(St, k) if k else qs, v))
        for ours, theirs in zip(flat, sym_eoms):
            val = theirs
            for key, v in sym_subs:
                val = val.subs(key, v)
            want = float(val.subs(sbinds))
            got = _num(ours, binds)
            assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9), (str(ours), got, want)
    return eoms, mapping


def test_double_pendulum_matches_sympy(pool):
    t = pool.symbol("t")
    m1, m2, l1, l2, g = (pool.symbol(n) for n in ("m1", "m2", "l1", "l2", "g"))
    th1, th2 = pool.func("theta1", [t]), pool.func("theta2", [t])
    d1, d2 = ak.diff(th1, t).value, ak.diff(th2, t).value
    T = (
        pool.rational(1, 2) * (m1 + m2) * l1**2 * d1**2
        + pool.rational(1, 2) * m2 * l2**2 * d2**2
        + m2 * l1 * l2 * d1 * d2 * ak.cos(th1 - th2)
    )
    V = -(m1 + m2) * g * l1 * ak.cos(th1) - m2 * g * l2 * ak.cos(th2)
    L = T - V

    St = sympy.Symbol("t")
    sm1, sm2, sl1, sl2, sg = sympy.symbols("m1 m2 l1 l2 g")
    a1, a2 = sympy.Function("theta1")(St), sympy.Function("theta2")(St)
    sT = (
        sympy.Rational(1, 2) * (sm1 + sm2) * sl1**2 * a1.diff(St) ** 2
        + sympy.Rational(1, 2) * sm2 * sl2**2 * a2.diff(St) ** 2
        + sm2 * sl1 * sl2 * a1.diff(St) * a2.diff(St) * sympy.cos(a1 - a2)
    )
    sV = -(sm1 + sm2) * sg * sl1 * sympy.cos(a1) - sm2 * sg * sl2 * sympy.cos(a2)
    _eoms, mapping = _compare_eoms(
        pool, t, [th1, th2], L, [a1, a2], sT - sV, [m1, m2, l1, l2, g], [sm1, sm2, sl1, sl2, sg]
    )
    names = sorted(str(s) for s in mapping.values())
    assert names == sorted(
        ["theta1", "theta1_dot", "theta1_ddot", "theta2", "theta2_dot", "theta2_ddot"]
    )


def test_cart_pole_matches_sympy(pool):
    t = pool.symbol("t")
    M, m, ln, g = (pool.symbol(n) for n in ("M", "m", "l", "g"))
    x, th = pool.func("x", [t]), pool.func("theta", [t])
    xd, thd = ak.diff(x, t).value, ak.diff(th, t).value
    L = (
        pool.rational(1, 2) * (M + m) * xd**2
        + m * ln * xd * thd * ak.cos(th)
        + pool.rational(1, 2) * m * ln**2 * thd**2
        - m * g * ln * ak.cos(th)
    )
    St = sympy.Symbol("t")
    sM, sm, sl, sg = sympy.symbols("M m l g")
    sx, sth = sympy.Function("x")(St), sympy.Function("theta")(St)
    sL = (
        sympy.Rational(1, 2) * (sM + sm) * sx.diff(St) ** 2
        + sm * sl * sx.diff(St) * sth.diff(St) * sympy.cos(sth)
        + sympy.Rational(1, 2) * sm * sl**2 * sth.diff(St) ** 2
        - sm * sg * sl * sympy.cos(sth)
    )
    _compare_eoms(pool, t, [x, th], L, [sx, sth], sL, [M, m, ln, g], [sM, sm, sl, sg])


def test_euler_lagrange_refusals(pool):
    t, x = pool.symbol("t"), pool.symbol("x")
    with pytest.raises(ak.DiffError) as exc:
        ak.euler_lagrange(x**2, [ak.sin(t)], t)
    assert exc.value.code == "E-DIFF-005"
    with pytest.raises(ak.DiffError) as exc:
        ak.euler_lagrange(x**2, [pool.func("q", [x])], t)
    assert exc.value.code == "E-DIFF-007"


# ---------------------------------------------------------------------------
# FLRW: Ricci scalar with a(t)
# ---------------------------------------------------------------------------


def test_flrw_ricci_scalar(pool):
    t, r, th, ph, k = (pool.symbol(n) for n in ("t", "r", "theta", "phi", "k"))
    a = pool.func("a", [t])
    coords = [t, r, th, ph]
    g = [
        pool.integer(-1),
        a**2 / (1 - k * r**2),
        a**2 * r**2,
        a**2 * r**2 * ak.sin(th) ** 2,
    ]  # diagonal metric
    ginv = [1 / gi for gi in g]
    n = 4

    def d(e, i):
        return ak.diff(e, coords[i]).value

    # Γ^a_bc for a diagonal metric
    gamma = [[[pool.integer(0)] * n for _ in range(n)] for _ in range(n)]
    for A in range(n):
        for B in range(n):
            for C in range(n):
                term = pool.integer(0)
                if A == C:
                    term = term + d(g[A], B)
                if A == B:
                    term = term + d(g[A], C)
                if B == C:
                    term = term - d(g[B], A)
                gamma[A][B][C] = ak.simplify(pool.rational(1, 2) * ginv[A] * term).value
    R = pool.integer(0)
    for B in range(n):
        ric = pool.integer(0)
        for A in range(n):
            ric = ric + d(gamma[A][B][B], A) - d(gamma[A][B][A], B)
            for E in range(n):
                ric = ric + gamma[A][A][E] * gamma[E][B][B] - gamma[A][B][E] * gamma[E][B][A]
        R = R + ginv[B] * ric
    ad = ak.diff(a, t).value
    add = ak.diff(ad, t).value
    want = 6 * (k + ad**2 + a * add) / a**2
    (flat_R, flat_want), mapping = ak.functions_to_symbols([R, want], [a], t)
    rng = random.Random(3)
    for _ in range(5):
        binds = {s: rng.uniform(0.3, 1.7) for s in mapping.values()}
        binds.update(
            {r: rng.uniform(0.1, 0.5), th: rng.uniform(0.2, 1.3), ph: 0.4, k: rng.uniform(-1, 1)}
        )
        assert math.isclose(_num(flat_R, binds), _num(flat_want, binds), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# dsolve with y(t)
# ---------------------------------------------------------------------------


def test_dsolve_accepts_applied_function(pool):
    t = pool.symbol("t")
    y = pool.func("y", [t])
    ode = ak.diff(ak.diff(y, t).value, t).value + y  # y'' + y = 0 (no y' term)
    sols = akx.dsolve(ode, t, y)
    assert sols
    assert sols[0]["form"] == "explicit"
    ysol = sols[0]["y_of_x"]
    residual = ak.subs(ode, {y: ysol})
    assert ak.simplify(residual).value == pool.integer(0)
    # A plain symbol without derivative symbols is still an error.
    with pytest.raises(ValueError):
        akx.dsolve(ode, t, pool.symbol("z"))
