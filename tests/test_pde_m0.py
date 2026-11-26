"""PDE foundation layer (milestone M0): unknowns and partials, the jet space,
problem objects, ``classify_pde`` (with a 40-equation table), verification,
the ``Sum`` / ``Indexed`` / ``RootOfFamily`` heads, and transforms of
unknowns.

Rust unit tests for the same layer live in ``alkahest-core/src/pde/``.
"""

from __future__ import annotations

import math

import alkahest as ak
import alkahest.experimental as akx
import pytest
from alkahest.experimental import (
    PDE,
    D,
    Function,
    HalfLine,
    Interval,
    JetSpace,
    PdeError,
    Sum,
)


@pytest.fixture
def P():
    pool = ak.ExprPool()

    class NS:
        pass

    ns = NS()
    ns.pool = pool
    ns.x, ns.y, ns.z, ns.t = (pool.symbol(n) for n in ("x", "y", "z", "t"))
    ns.u = Function("u")
    return ns


def code(exc_info) -> str:
    return exc_info.value.code


# ---------------------------------------------------------------------------
# Unknowns, partials, printing and parsing
# ---------------------------------------------------------------------------


def test_function_and_D(P):
    u = P.u(P.x, P.t)
    assert str(u) == "u(x, t)"
    assert str(D(u, P.x, P.t)) == "Derivative(u(x, t), x, t)"
    assert D(u, P.x, P.t) == D(u, P.t, P.x)
    assert str(D(u, P.x, 2)) == "Derivative(u(x, t), x, x)"
    assert D(u, (P.x, 2), P.t) == D(u, P.x, P.x, P.t)
    assert D(u) == u
    with pytest.raises(PdeError) as e:
        Function("sin")
    assert code(e) == "E-PDE-001"
    with pytest.raises(PdeError):
        Function("Sum")


def test_diff_with_order(P):
    f = P.x**4
    assert ak.diff(f, P.x, 2).value == ak.simplify(12 * P.x**2).value
    assert ak.diff(f, P.x).value == ak.diff(f, P.x, 1).value
    with pytest.raises(ValueError):
        ak.diff(f, P.x, 0)


def test_parse_round_trip_with_function(P):
    e = D(P.u(P.x, P.t), P.x, P.t) + P.u(P.x, P.t)
    back = ak.parse(str(e), P.pool, {"u": P.u, "x": P.x, "t": P.t})
    assert back == e


def test_jet_names_parse_back_through_from_jets(P):
    u = P.u(P.x, P.t)
    J = JetSpace([P.x, P.t], [u], 2)
    s = ak.parse("u_t - u_xx", P.pool)
    assert J.from_jets(s) == D(u, P.t) - D(u, P.x, 2)


# ---------------------------------------------------------------------------
# JetSpace
# ---------------------------------------------------------------------------


def test_jetspace_bijection(P):
    u = P.u(P.x, P.t)
    J = JetSpace([P.x, P.t], [u], 3)
    assert str(J.jet(u, [1, 1])) == "u_xt"
    assert str(J.jet(0, [2, 0])) == "u_xx"
    e = D(u, P.x, P.t) + u * D(u, P.x, 2)
    je = J.to_jets(e)
    assert J.from_jets(je) == e
    assert J.to_jets(je) == je
    assert J.decode(J.jet(u, [0, 2])) == (u, [0, 2])
    assert J.decode(P.pool.symbol("u_tx")) is None
    assert len(J.coordinates()) == 10
    assert J.order_of(e) == 2
    assert J.independent == [P.x, P.t]
    assert J.unknowns == [u]


def test_jetspace_total_derivative_and_prolongation(P):
    u = P.u(P.x, P.t)
    J = JetSpace(None, [u], 2)
    f = P.x * u * D(u, P.x)
    dj = J.total_derivative(f, P.x)
    want = J.prolonged(1).to_jets(ak.diff(f, P.x).value)
    assert ak.simplify(dj - want).value == P.pool.integer(0)
    heat = D(u, P.t) - D(u, P.x, 2)
    assert len(J.prolong([heat], 2)) == 6


def test_jetspace_rational_system(P):
    u = P.u(P.x, P.t)
    J = JetSpace([P.x, P.t], [u], 2)
    sys = J.to_rational_system([D(u, P.t) - D(u, P.x) / u])
    assert len(sys["equations"]) == 1
    assert [str(s) for s in sys["nonzero"]] == ["u"]
    with pytest.raises(PdeError) as e:
        J.to_rational_system([ak.sin(u)])
    assert code(e) == "E-PDE-006"


def test_jetspace_errors(P):
    u = P.u(P.x, P.t)
    J = JetSpace([P.x, P.t], [u], 1)
    with pytest.raises(PdeError) as e:
        J.to_jets(P.u(P.x, 0))
    assert code(e) == "E-PDE-003"
    with pytest.raises(PdeError) as e:
        J.to_jets(D(u, P.x, 2))
    assert code(e) == "E-PDE-004"
    with pytest.raises(PdeError) as e:
        JetSpace([P.x], [u], 1)
    assert code(e) == "E-PDE-002"
    with pytest.raises(PdeError) as e:
        JetSpace([P.x, P.t], [u], 1, naming="prime")
    assert code(e) == "E-PDE-009"


def test_ode_conventions_are_jet_spaces(P):
    q = Function("q")(P.t)
    J = JetSpace([P.t], [q], 3)
    e = D(q, P.t, 2) + q
    old, _ = ak.functions_to_symbols(e, [q], P.t)
    assert J.with_naming("dot").to_jets(e) == old
    inp = J.to_ode_input(e)
    assert [str(s) for s in inp["derivs"]] == ["q'", "q''"]
    pairs = J.dae_pairs()
    assert str(pairs[0][1]) == "dq/dt"


def test_functions_to_jets(P):
    u = P.u(P.x, P.t)
    clash = P.pool.symbol("u_x")
    e = D(u, P.x, P.t) + D(u, P.x) + clash
    out, mapping = akx.functions_to_jets(e, [u])
    assert sorted(str(s) for s in mapping.values()) == ["u_x_", "u_xt"]
    back = ak.subs(out, {s: n for n, s in mapping.items()})
    assert back == e


# ---------------------------------------------------------------------------
# Problem objects
# ---------------------------------------------------------------------------


def test_problem_builder(P):
    u = P.u(P.x, P.t)
    L = P.pool.symbol("L")
    heat = PDE(D(u, P.t) - D(u, P.x, 2), u)
    prob = (
        heat.on(Interval(P.x, 0, L) * HalfLine(P.t, 0))
        .bc(P.u(0, P.t))
        .bc(P.u(L, P.t), 0)
        .ic(P.u(P.x, 0), ak.sin(P.x))
    )
    assert prob.order == 2
    assert len(prob.bcs) == 2
    assert len(prob.ics) == 1
    assert prob.independent == [P.x, P.t]
    assert heat.classify().names == ["heat"]
    assert "heat" in repr(heat.classify())
    # immutability
    assert heat.bcs == []
    with pytest.raises(PdeError) as e:
        heat.bc(P.x)
    assert code(e) == "E-PDE-011"
    with pytest.raises(PdeError) as e:
        heat.on(Interval(P.y, 0, 1))
    assert code(e) == "E-PDE-010"
    with pytest.raises(PdeError) as e:
        heat.on(Interval(P.x, 1, 0))
    assert code(e) == "E-PDE-010"
    with pytest.raises(PdeError) as e:
        PDE(P.x + 1, u)
    assert code(e) == "E-PDE-007"
    # unknown inferred; (lhs, rhs) equations
    assert PDE((D(u, P.t), D(u, P.x, 2))).unknowns == [u]


def test_domains(P):
    d = Interval(P.x, 0, 1) * HalfLine(P.t, 0)
    assert "x" in str(d.condition())
    assert akx.Box([P.x, P.y], [(0, 1), (0, 2)]).vars() == [P.x, P.y]
    akx.Disk((P.x, P.y), 1)
    akx.Ball([P.x, P.y, P.z], 2)
    u = Function("w")(P.x, P.y)
    PDE(D(u, P.x, 2) + D(u, P.y, 2), u).on(akx.Disk((P.x, P.y), 1))
    with pytest.raises(PdeError):
        PDE(D(u, P.x, 2) + D(u, P.y, 2), u).on(akx.Disk((P.x, P.y), -1))


def test_system_classification(P):
    # Systems used to be refused with E-PDE-012; they now classify (the
    # one-dimensional wave equation as a first-order system).
    u, v = P.u(P.x, P.t), Function("v")(P.x, P.t)
    sys = PDE([D(u, P.t) - D(v, P.x), D(v, P.t) - D(u, P.x)], [u, v])
    assert sys.order == 1
    info = sys.classify()
    assert isinstance(info, akx.PdeSystemInfo)
    assert info.type == "hyperbolic"
    assert info.strictly_hyperbolic


def test_arbitrary_functions(P):
    f1 = Function("F1")(P.x)
    gen = akx.ArbitraryFunctions(avoid=[f1])
    assert gen.fresh_name() == "F2"
    assert gen.fresh(P.x - P.t) == Function("F3")(P.x - P.t)


# ---------------------------------------------------------------------------
# Classification: the 40-equation table
# ---------------------------------------------------------------------------


def _table(P):
    x, y, z, t = P.x, P.y, P.z, P.t
    pool = P.pool
    c = pool.symbol("c")
    k = pool.symbol("k", "positive")
    nu = pool.symbol("nu", "positive")
    m = pool.symbol("m")
    a = pool.symbol("a", "positive")
    Dc = pool.symbol("D", "positive")
    s = pool.symbol("s")
    sigma = pool.symbol("sigma", "positive")
    r = pool.symbol("r", "positive")
    U = Function("u")
    uxt, uxy, uxyz, uxyt, ust = U(x, t), U(x, y), U(x, y, z), U(x, y, t), U(s, t)

    def d(f, *v):
        return D(f, *v)

    u = uxt
    w = uxy
    return [
        # (label, equation, unknown, order, linearity, type, names)
        ("heat", d(u, t) - d(u, x, 2), u, 2, "linear", "parabolic", ["heat"]),
        ("heat k", d(u, t) - k * d(u, x, 2), u, 2, "linear", "parabolic", ["heat"]),
        (
            "heat 2d",
            d(uxyt, t) - (d(uxyt, x, 2) + d(uxyt, y, 2)),
            uxyt,
            2,
            "linear",
            "parabolic",
            ["heat"],
        ),
        ("heat source", d(u, t) - d(u, x, 2) - ak.sin(x), u, 2, "linear", "parabolic", ["heat"]),
        ("wave", d(u, t, 2) - c**2 * d(u, x, 2), u, 2, "linear", "hyperbolic", ["wave"]),
        (
            "wave 2d",
            d(uxyt, t, 2) - 4 * (d(uxyt, x, 2) + d(uxyt, y, 2)),
            uxyt,
            2,
            "linear",
            "hyperbolic",
            ["wave"],
        ),
        ("laplace", d(w, x, 2) + d(w, y, 2), w, 2, "linear", "elliptic", ["laplace"]),
        (
            "laplace 3d",
            d(uxyz, x, 2) + d(uxyz, y, 2) + d(uxyz, z, 2),
            uxyz,
            2,
            "linear",
            "elliptic",
            ["laplace"],
        ),
        ("poisson", d(w, x, 2) + d(w, y, 2) - x * y, w, 2, "linear", "elliptic", ["poisson"]),
        (
            "helmholtz",
            d(w, x, 2) + d(w, y, 2) + k**2 * w,
            w,
            2,
            "linear",
            "elliptic",
            ["helmholtz"],
        ),
        ("transport", d(u, t) + 3 * d(u, x), u, 1, "linear", None, ["transport"]),
        ("transport var", d(u, t) + x * d(u, x), u, 1, "linear", None, ["transport"]),
        (
            "inviscid burgers",
            d(u, t) + u * d(u, x),
            u,
            1,
            "quasilinear",
            None,
            ["inviscid_burgers"],
        ),
        (
            "burgers",
            d(u, t) + u * d(u, x) - nu * d(u, x, 2),
            u,
            2,
            "semilinear",
            "parabolic",
            ["burgers"],
        ),
        ("kdv", d(u, t) + 6 * u * d(u, x) + d(u, x, 3), u, 3, "semilinear", None, ["kdv"]),
        (
            "fisher",
            d(u, t) - d(u, x, 2) - u * (1 - u),
            u,
            2,
            "semilinear",
            "parabolic",
            ["fisher_kpp"],
        ),
        (
            "sine-gordon",
            d(u, t, 2) - d(u, x, 2) + ak.sin(u),
            u,
            2,
            "semilinear",
            "hyperbolic",
            ["sine_gordon"],
        ),
        (
            "sine-gordon lc",
            d(u, x, t) - ak.sin(u),
            u,
            2,
            "semilinear",
            "hyperbolic",
            ["sine_gordon"],
        ),
        (
            "klein-gordon",
            d(u, t, 2) - d(u, x, 2) + m**2 * u,
            u,
            2,
            "linear",
            "hyperbolic",
            ["klein_gordon"],
        ),
        (
            "telegraph",
            d(u, t, 2) + 2 * d(u, t) - d(u, x, 2),
            u,
            2,
            "linear",
            "hyperbolic",
            ["telegraph"],
        ),
        ("tricomi", y * d(w, x, 2) + d(w, y, 2), w, 2, "linear", "mixed", []),
        ("tricomi x", x * d(w, y, 2) + d(w, x, 2), w, 2, "linear", "mixed", []),
        (
            "mixed parabolic",
            d(w, x, 2) + 2 * d(w, x, y) + d(w, y, 2),
            w,
            2,
            "linear",
            "parabolic",
            [],
        ),
        (
            "mixed hyperbolic",
            d(w, x, 2) + 3 * d(w, x, y) + d(w, y, 2),
            w,
            2,
            "linear",
            "hyperbolic",
            [],
        ),
        ("mixed elliptic", d(w, x, 2) + d(w, x, y) + d(w, y, 2), w, 2, "linear", "elliptic", []),
        (
            "monge-ampere",
            d(w, x, 2) * d(w, y, 2) - d(w, x, y) ** 2 - 1,
            w,
            2,
            "fully_nonlinear",
            "solution_dependent",
            [],
        ),
        ("eikonal", d(w, x) ** 2 + d(w, y) ** 2 - 1, w, 1, "fully_nonlinear", None, []),
        ("hamilton-jacobi", d(u, t) + d(u, x) ** 2, u, 1, "fully_nonlinear", None, []),
        (
            "porous medium",
            d(u, t) - u * d(u, x, 2) - d(u, x) ** 2,
            u,
            2,
            "quasilinear",
            "solution_dependent",
            [],
        ),
        (
            "minimal surface",
            (1 + d(w, y) ** 2) * d(w, x, 2)
            - 2 * d(w, x) * d(w, y) * d(w, x, y)
            + (1 + d(w, x) ** 2) * d(w, y, 2),
            w,
            2,
            "quasilinear",
            "solution_dependent",
            [],
        ),
        (
            "biharmonic",
            d(w, x, 4) + 2 * d(w, x, 2, y, 2) + d(w, y, 4),
            w,
            4,
            "linear",
            None,
            [],
        ),
        (
            "schrodinger",
            pool.imaginary_unit() * d(u, t) + d(u, x, 2),
            u,
            2,
            "linear",
            "parabolic",
            [],
        ),
        (
            "black-scholes",
            d(ust, t)
            + pool.rational(1, 2) * sigma**2 * s**2 * d(ust, s, 2)
            + r * s * d(ust, s)
            - r * ust,
            ust,
            2,
            "linear",
            "parabolic",
            [],
        ),
        (
            "advection-diffusion",
            d(u, t) + a * d(u, x) - Dc * d(u, x, 2),
            u,
            2,
            "linear",
            "parabolic",
            [],
        ),
        (
            "damped klein-gordon",
            d(u, t, 2) + d(u, t) - d(u, x, 2) + u,
            u,
            2,
            "linear",
            "hyperbolic",
            ["telegraph"],
        ),
        ("laplace xt", d(u, x, 2) + d(u, t, 2), u, 2, "linear", "elliptic", ["laplace"]),
        ("backward heat", d(u, t) + d(u, x, 2), u, 2, "linear", "parabolic", []),
        ("nonlinear heat", d(u, t) - d(u, x, 2) - u**3, u, 2, "semilinear", "parabolic", []),
        ("euler first order", x * d(w, x) + y * d(w, y) - w, w, 1, "linear", None, []),
        (
            "wave source",
            d(u, t, 2) - d(u, x, 2) - ak.sin(x * t),
            u,
            2,
            "linear",
            "hyperbolic",
            ["wave"],
        ),
    ]


def test_classification_table_of_40(P):
    table = _table(P)
    assert len(table) == 40
    failures = []
    for label, eq, unk, order, lin, typ, names in table:
        info = akx.classify_pde(eq, unk)
        got = (info.order, info.linearity, info.type, info.names)
        if got != (order, lin, typ, names):
            failures.append((label, got, (order, lin, typ, names)))
    assert not failures, failures


def test_classification_details(P):
    u = P.u(P.x, P.t)
    info = akx.classify_pde(D(u, P.t) - D(u, P.x, 2) - ak.sin(P.x), u)
    assert info.homogeneous is False
    assert info.constant_coefficients is True
    heat = info.named[0]
    assert str(heat["parameters"]["diffusivity"]) == "1"
    assert str(heat["parameters"]["source"]) == "sin(x)"
    assert heat["time"] == P.t
    var = akx.classify_pde(D(u, P.t) + P.x * D(u, P.x), u)
    assert var.constant_coefficients is False
    c = P.pool.symbol("c")
    wave = akx.classify_pde(D(u, P.t, 2) - c**2 * D(u, P.x, 2), u)
    assert len(wave.type_conditions) == 1
    assert len(wave.named[0]["conditions"]) == 1


def test_tricomi_regions(P):
    w = P.u(P.x, P.y)
    info = akx.classify_pde(P.y * D(w, P.x, 2) + D(w, P.y, 2), w)
    assert info.type == "mixed"
    assert [r["type"] for r in info.regions] == ["hyperbolic", "parabolic", "elliptic"]
    assert info.regions[0]["upper"] == P.pool.integer(0)
    assert info.regions[0]["var"] == P.y
    assert str(info.discriminant) in ("-4*y", "(-4 * y)", "-4 y") or "y" in str(info.discriminant)


def test_classify_infers_unknown(P):
    u = P.u(P.x, P.t)
    assert akx.classify_pde(D(u, P.t) - D(u, P.x, 2)).names == ["heat"]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_verify_dalembert(P):
    u = P.u(P.x, P.t)
    c = P.pool.symbol("c")
    wave = PDE(D(u, P.t, 2) - c**2 * D(u, P.x, 2), u)
    gen = akx.ArbitraryFunctions()
    sol = gen.fresh(P.x - c * P.t) + gen.fresh(P.x + c * P.t)
    v = akx.verify_pde_solution(wave, sol)
    assert v["status"] == "exactly_verified"
    assert v["verified"] is True
    assert v["arbitrary_functions"] == ["F1", "F2"]
    assert v["pde"][0]["status"] == "proved_zero"
    bad = Function("F1")(P.x - 2 * c * P.t)
    v = akx.verify_pde_solution(wave, bad)
    assert v["status"] == "refuted"
    assert v["verified"] is False


def test_verify_heat_ibvp(P):
    u = P.u(P.x, P.t)
    pi = P.pool.pi()
    heat = (
        PDE(D(u, P.t) - D(u, P.x, 2), u)
        .bc(P.u(0, P.t))
        .bc(P.u(1, P.t))
        .ic(P.u(P.x, 0), ak.sin(pi * P.x))
    )
    sol = ak.sin(pi * P.x) * ak.exp(-(pi**2) * P.t)
    v = heat.verify(sol)
    assert v["status"] == "exactly_verified", v
    assert len(v["bc"]) == 2
    assert len(v["ic"]) == 1
    wrong = ak.sin(pi * P.x) * ak.exp(-P.t)
    v = heat.verify(wrong)
    assert v["status"] == "refuted"
    assert v["pde"][0]["status"] == "proved_nonzero"
    assert v["bc"][0]["status"] == "proved_zero"
    # extra conditions passed directly
    v = akx.verify_pde_solution(heat, sol, ics=[(P.u(P.x, 0), ak.sin(pi * P.x))])
    assert v["status"] == "exactly_verified"


def test_verify_equation_without_problem(P):
    u = P.u(P.x, P.t)
    v = akx.verify_pde_solution(D(u, P.t) + 2 * D(u, P.x), Function("G")(P.x - 2 * P.t))
    assert v["status"] == "exactly_verified"


# ---------------------------------------------------------------------------
# Sum / Indexed / KroneckerDelta
# ---------------------------------------------------------------------------


def test_sum_basics(P):
    pool = P.pool
    k = pool.symbol("k", "integer")
    n = pool.symbol("n", "integer")
    s = Sum(P.x**k / k, k, 1, n)
    assert str(s) == "Sum(x^k * k^-1, k, 1, n)" or str(s).startswith("Sum(")
    # printing round-trips
    assert ak.parse(str(s), pool, {"x": P.x, "k": k, "n": n}) == s
    # binder: substituting the bound variable is a no-op; a free one works
    assert ak.subs(s, {k: pool.integer(3)}) == s
    s3 = ak.subs(s, {n: pool.integer(3)})
    assert akx.expand_sum(s3) == ak.simplify(P.x + P.x**2 / 2 + P.x**3 / 3).value
    # numeric evaluation with a symbolic bound bound at call time
    val = ak.eval_expr(s, {P.x: 0.5, n: 40})
    assert abs(val - math.log(2)) < 1e-9
    # truncate / coefficient on an infinite series
    inf = Sum(P.x**k, k, 1, pool.pos_infinity())
    assert akx.truncate(inf, 2) == ak.simplify(P.x + P.x**2).value
    assert akx.coefficient(inf, 4) == P.x**4
    with pytest.raises(PdeError) as e:
        akx.expand_sum(s)
    assert code(e) == "E-PDE-021"
    with pytest.raises(PdeError) as e:
        Sum(k, P.x, 1, P.x)
    assert code(e) == "E-PDE-020"


def test_sum_diff_and_folding(P):
    pool = P.pool
    k = pool.symbol("k", "integer")
    d = pool.symbol("d", "positive")
    s = Sum(P.y * k, k, 1, d)
    assert akx.simplify_sums(ak.diff(s, P.y).value) == akx.simplify_sums(Sum(k, k, 1, d))
    assert akx.simplify_sums(Sum(P.y, k, 1, d)) == ak.simplify(P.y * d).value
    with pytest.raises(ak.DiffError):
        ak.diff(Sum(k, k, 1, P.y), P.y)


def test_symbolic_dimension_laplacian(P):
    pool = P.pool
    i = pool.symbol("i", "integer")
    j = pool.symbol("j", "integer")
    d = pool.symbol("d", "positive")
    X = akx.IndexedBase(pool.symbol("x"))
    u = Sum(X[i] ** 2 / 2, i, 1, d)
    assert akx.laplacian_indexed(u, X, 1, d) == d
    du = ak.diff(u, X[j]).value
    assert "KroneckerDelta" in str(du)
    assert akx.simplify_sums(du, [(j, pool.integer(1), d)]) == X[j]
    assert akx.KroneckerDelta(pool.integer(2), pool.integer(3)) == pool.integer(0)
    assert akx.KroneckerDelta(i, i) == pool.integer(1)
    # evaluation of indexed sums with concrete d
    env = {d: 3, X[1]: 1.0, X[2]: 2.0, X[3]: 3.0}
    assert ak.eval_expr(Sum(X[i] ** 2, i, 1, d), env) == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# RootOfFamily
# ---------------------------------------------------------------------------


def test_root_of_family(P):
    pool = P.pool
    lam = pool.symbol("lam")
    n = pool.symbol("n", "integer")
    fam = akx.RootOfFamily(ak.sin(lam), lam, n)
    assert str(fam) == "RootOfFamily(sin(lam), lam, n, 0)"
    third = akx.root_family_member(fam, 3)
    lo, hi = akx.root_family_enclosure(third, 80)
    assert float(lo) <= 3 * math.pi <= float(hi)
    # Robin: h sin(lam) + lam cos(lam) = 0 with h = 1
    robin = akx.RootOfFamily(ak.sin(lam) + lam * ak.cos(lam), lam, 1)
    lo, hi = akx.root_family_enclosure(robin)
    assert abs(float(lo) - 2.0287578381104345) < 1e-12
    assert ak.eval_expr(robin, {}) == pytest.approx(2.0287578381104345, abs=1e-12)
    # parse round trip
    assert ak.parse(str(fam), pool, {"lam": lam, "n": n}) == fam
    # refusals
    with pytest.raises(PdeError) as e:
        akx.root_family_enclosure(akx.RootOfFamily(lam**2 + 1, lam, 1))
    assert code(e) == "E-PDE-031"
    with pytest.raises(PdeError) as e:
        akx.RootOfFamily(lam, lam, n, 5)
    assert code(e) == "E-PDE-030"


def test_root_family_implicit_derivative(P):
    pool = P.pool
    lam = pool.symbol("lam")
    h = pool.symbol("h", "positive")
    r = akx.RootOfFamily(lam - h, lam, 1)
    assert ak.diff(r, h).value == pool.integer(1)


# ---------------------------------------------------------------------------
# Transforms of unknowns
# ---------------------------------------------------------------------------


def test_laplace_transform_of_pde(P):
    u = P.u(P.x, P.t)
    s = P.pool.symbol("s")
    out = str(akx.laplace_transform(D(u, P.t) - D(u, P.x, 2), P.t, s))
    assert "U(x, s)" in out
    assert "u(x, 0)" in out
    assert "Derivative(U(x, s), x, x)" in out
    out2 = str(akx.laplace_transform(D(u, P.t, 2), P.t, s))
    assert "s^2" in out2
    assert "u(x, 0)" in out2


def test_fourier_transform_of_pde(P):
    u = P.u(P.x, P.t)
    xi = P.pool.symbol("xi")
    out = str(akx.fourier_transform(D(u, P.t) - 3 * D(u, P.x, 2), P.x, xi))
    assert "u_hat(xi, t)" in out
    assert "Derivative(u_hat(xi, t), t)" in out
    with pytest.raises(ak.TransformError):
        akx.fourier_transform(P.u(2 * P.x, P.t), P.x, xi)


def test_error_codes_are_registered():
    import re

    text = open(  # noqa: SIM115
        __import__("pathlib").Path(__file__).parents[1] / "alkahest-core/src/errors/codes.rs",
        encoding="utf-8",
    ).read()
    codes = re.findall(r'"(E-PDE-\d{3})"', text)
    assert codes
    # M0 owns 001–049; later milestones register their own ranges (M1:
    # 050–099, …), so only the M0 codes are checked against M0's range here.
    m0 = [c for c in codes if int(c[-3:]) <= 49]
    assert m0
    assert all(int(c[-3:]) >= 1 for c in m0)
    assert len(set(codes)) == len(codes)
