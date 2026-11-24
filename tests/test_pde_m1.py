"""``pdsolve`` — milestone M1 (first-order PDEs): API, tiers, verification,
Cauchy problems and refusals.  The corpus regression lives in
``tests/pde_corpus/``."""

from __future__ import annotations

import alkahest as ak
import pytest

akx = pytest.importorskip("alkahest.experimental")
if not hasattr(akx, "pdsolve"):
    pytest.skip("pdsolve not built", allow_module_level=True)


@pytest.fixture
def xy():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    U = akx.Function("u")
    return p, x, y, U, U(x, y)


def test_transport_general_solution(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    sol = akx.pdsolve(2 * akx.D(u, x) + 3 * akx.D(u, y), u)
    assert isinstance(sol, akx.PdeSolution)
    assert sol.kind == "general"
    assert sol.method == "first_order_linear"
    assert sol.verified
    assert sol.verification["status"] == "exactly_verified"
    assert sol.arbitrary_functions == ["F1"]
    assert not sol.implicit
    assert sol.rhs is not None
    # Independent re-check through M0's verifier.
    assert akx.verify_pde_solution(akx.PDE(2 * akx.D(u, x) + 3 * akx.D(u, y), u), sol.value)[
        "verified"
    ]


def test_unknown_inferred_and_solvefun(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    sol = akx.pdsolve(u + akx.D(u, x) + akx.D(u, y), solvefun="G")
    assert sol.arbitrary_functions == ["G1"]


def test_sympy_inhomogeneous(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    eq = -2 * akx.D(u, x) + 4 * akx.D(u, y) + 5 * u - ak.exp(x + 3 * y)
    sol = akx.pdsolve(eq, u)
    assert sol.verified
    assert sol.kind == "general"


def test_variable_coefficients(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    eq = x * akx.D(u, x) - y * akx.D(u, y) + y**2 * u - y**2
    sol = akx.pdsolve(eq, u)
    assert sol.verified


def test_three_variables():
    p = ak.ExprPool()
    x, y, z = p.symbol("x"), p.symbol("y"), p.symbol("z")
    u = akx.Function("u")(x, y, z)
    sol = akx.pdsolve(x * akx.D(u, x) + y * akx.D(u, y) + z * akx.D(u, z) - 3 * u, u)
    assert sol.verified
    assert sol.kind == "general"


def test_quasilinear_implicit(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    eq = (y - u) * akx.D(u, x) + (u - x) * akx.D(u, y) - (x - y)
    sol = akx.pdsolve(eq, u)
    assert sol.implicit
    assert sol.kind == "implicit"
    assert sol.method == "first_order_quasilinear"
    assert sol.rhs is None
    assert sol.verified


def test_cauchy_problem_explicit(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    sol = akx.pdsolve(akx.D(u, x) + akx.D(u, y) - u, u, ics=[(U(x, 0), ak.cos(x))])
    assert sol.kind == "particular"
    assert sol.verified
    assert [c["status"] for c in sol.verification["ic"]] == ["proved_zero"]


def test_burgers_breaking_time():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    U = akx.Function("u")
    u = U(x, t)
    pde = akx.PDE(akx.D(u, t) + u * akx.D(u, x), u).ic(U(x, 0), -x)
    sol = akx.pdsolve(pde)
    assert sol.verified
    rendered = [s[2] for s in sol.side_conditions]
    assert any(s[0] == "positive" for s in sol.side_conditions), rendered
    assert any("t" in r for r in rendered)


def test_burgers_sine_is_implicit():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    U = akx.Function("u")
    u = U(x, t)
    sol = akx.pdsolve(akx.PDE(akx.D(u, t) + u * akx.D(u, x), u).ic(U(x, 0), ak.sin(x)))
    assert sol.verified
    assert sol.implicit


def test_characteristic_data_refused(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    with pytest.raises(akx.PdeError) as ei:
        akx.pdsolve(akx.D(u, x), u, ics=[(U(x, 0), x)])
    assert ei.value.code == "E-PDE-052"


def test_complete_and_singular_integral(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    px, py = akx.D(u, x), akx.D(u, y)
    sol = akx.pdsolve(u - x * px - y * py - px * py, u)
    assert sol.kind == "complete_integral"
    assert sol.method == "clairaut"
    assert len(sol.constants) == 2
    assert [a.kind for a in sol.alternatives] == ["singular_integral"]
    assert sol.alternatives[0].verified


def test_charpit_and_separable(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    px, py = akx.D(u, x), akx.D(u, y)
    assert akx.pdsolve(px * py - x * y, u).verified
    s = akx.pdsolve(px**2 + py**2 - x - y, u)
    assert s.verified
    assert s.kind == "complete_integral"


def test_hints(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    names = [h[0] for h in akx.pdsolve_hints()]
    assert names[0] == "first_order_linear"
    assert "charpit" in names
    sol = akx.pdsolve(akx.D(u, x) + akx.D(u, y), u, hint="first_order_quasilinear")
    assert sol.method == "first_order_quasilinear"
    with pytest.raises(akx.PdeError) as ei:
        akx.pdsolve(akx.D(u, x), u, hint="no_such_hint")
    assert ei.value.code == "E-PDE-050"


def test_all_returns_one_per_tier(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    sols = akx.pdsolve(akx.D(u, x) + akx.D(u, y), u, all=True)
    assert len(sols) >= 2
    assert all(s.verified for s in sols)
    assert len({s.method for s in sols}) == len(sols)


def test_second_order_refused_for_now(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    with pytest.raises(akx.PdeError) as ei:
        akx.pdsolve(akx.D(u, x, 2) - akx.D(u, y), u)
    assert ei.value.code == "E-PDE-050"


def test_log_source_by_parts(xy):
    # SymPy only classifies this one; the particular part needs ∫ eˣ log x dx.
    p, x, y, U, u = xy  # noqa: RUF059
    sol = akx.pdsolve(u + akx.D(u, x) + akx.D(u, y) + ak.log(x), u)
    assert sol.verified


def test_unverifiable_is_refused_not_returned(xy):
    # No tier solves this; the answer must be a refusal, never a candidate
    # that failed verification.
    p, x, y, U, u = xy  # noqa: RUF059
    with pytest.raises(akx.PdeError) as ei:
        akx.pdsolve(akx.D(u, x) ** 2 + x * akx.D(u, y) ** 3 + u * y, u)
    assert ei.value.code in ("E-PDE-051",)


# ---------------------------------------------------------------------------
# Integral head and answers with unevaluated integrals
# ---------------------------------------------------------------------------


def test_integral_head_calculus(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    t = p.symbol("t")
    f = akx.Function("f")
    i = akx.Integral(f(t) * y, t, 1, x)
    # Leibniz: d/dx is the integrand at x; d/dy differentiates under the sign.
    assert str(ak.simplify(ak.diff(i, x).value - f(x) * y).value) == "0"
    assert "Integral" in str(ak.diff(i, y).value)
    # The bound variable is shadowed by subs; the free bound is not.
    assert ak.subs(i, {t: p.integer(3)}) == i
    assert "Integral" in str(ak.subs(i, {x: p.integer(2)}))
    # Indefinite form: the fundamental theorem.
    ii = akx.Integral(f(x), x)
    assert ak.diff(ii, x).value == f(x)
    # Parsing round-trips.
    back = ak.parse(str(i), p, {"x": x, "y": y, "t": t, "f": f})
    assert str(back) == str(i)
    with pytest.raises(akx.PdeError) as ei:
        akx.Integral(f(t), t, t, x)
    assert ei.value.code == "E-PDE-054"


def test_arbitrary_coefficient_needs_unevaluated_integral(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    f = akx.Function("f")
    sol = akx.pdsolve(akx.D(u, x) - f(x), u)
    assert sol.verified
    assert "Integral" in str(sol.value)
    g = akx.Function("g")
    sol2 = akx.pdsolve(f(x) * akx.D(u, x) + g(y) * akx.D(u, y), u)
    assert sol2.verified


def test_riccati_characteristic_with_parameters(xy):
    p, x, y, U, u = xy  # noqa: RUF059
    a, b, m, n = (p.symbol(s) for s in "abmn")
    # y' = a x^n y^2 + b m x^(m-1) - a b^2 x^(n+2m): particular solution b x^m.
    rhs = a * x**n * y**2 + b * m * x ** (m - 1) - a * b**2 * x ** (n + 2 * m)
    sol = akx.pdsolve(akx.D(u, x) + rhs * akx.D(u, y), u)
    assert sol.verified
