"""Milestone M6 of the PDE plan: Lie point symmetries, conservation laws,
reductions (traveling waves, expansion methods, similarity solutions) and
linearising transformations.  Every returned object is verified; the tests
also check the refusals (``E-PDE-300…349``)."""

from __future__ import annotations

import alkahest as ak
import pytest

akx = pytest.importorskip("alkahest.experimental")
if not hasattr(akx, "pde_symmetries"):  # pragma: no cover - old extension
    pytest.skip("extension built without the M6 layer", allow_module_level=True)

D = akx.D


@pytest.fixture
def xt():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    u = akx.Function("u")(x, t)
    return p, x, t, u


# ---------------------------------------------------------------------------
# Point symmetries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "make", "dim", "infinite"),
    [
        ("heat", lambda u, x, t, p: D(u, t) - D(u, x, 2), 6, "superposition"),
        ("burgers", lambda u, x, t, p: D(u, t) + u * D(u, x) - D(u, x, 2), 5, None),
        ("kdv", lambda u, x, t, p: D(u, t) + u * D(u, x) + D(u, x, 3), 4, None),
        ("mkdv", lambda u, x, t, p: D(u, t) + u**2 * D(u, x) + D(u, x, 3), 3, None),
        ("fisher", lambda u, x, t, p: D(u, t) - D(u, x, 2) - u * (1 - u), 2, None),
        ("sine_gordon", lambda u, x, t, p: D(D(u, x), t) - ak.sin(u), 3, None),
        ("wave", lambda u, x, t, p: D(u, t, 2) - D(u, x, 2), 1, "passive"),
        ("klein_gordon_cubic", lambda u, x, t, p: D(u, t, 2) - D(u, x, 2) + u**3, 4, None),
    ],
)
def test_symmetry_table(xt, name, make, dim, infinite):
    p, x, t, u = xt
    S = akx.pde_symmetries(make(u, x, t, p), u)
    assert len(S.generators) == dim, (name, S)
    assert all(g.verified for g in S.generators)
    assert S.complete
    assert (S.infinite["kind"] if S.infinite else None) == infinite
    if dim:
        assert S.structure_constants is not None


def test_heat_lie_algebra_and_verify(xt):
    p, x, t, u = xt
    eq = D(u, t) - D(u, x, 2)
    S = akx.pde_symmetries(eq, u)
    alg = S.lie_algebra()
    assert len(alg.generators) == 6
    # Every generator passes the independent check; a non-symmetry fails.
    for g in S.generators:
        assert akx.verify_symmetry(eq, u, g) == "proved_zero"
    w = p.symbol("u")
    assert akx.verify_symmetry(eq, u, [x, t, p.integer(0)]) != "proved_zero"
    assert akx.verify_symmetry(eq, u, [x, 2 * t, p.integer(0)]) == "proved_zero"
    assert w is not None
    # [∂_x, x∂_x + 2t∂_t] = ∂_x
    c = akx.commutator(
        [x, t, w], [p.integer(1), p.integer(0), p.integer(0)], [x, 2 * t, p.integer(0)]
    )
    assert str(c[0]) == "1"
    assert str(c[1]) == "0"


def test_group_classification_nonlinear_diffusion(xt):
    p, x, t, u = xt
    m = p.symbol("m")
    eq = D(u, t) - D(u**m * D(u, x), x)
    cases = akx.pde_symmetries(eq, u, cases=True)
    by_case = {tuple(str(v) for _, v in c.case): len(c.generators) for c in cases}
    assert by_case[()] == 4
    assert by_case[("-4/3",)] == 5
    assert by_case[("0",)] == 6


def test_determining_equations_are_linear_in_infinitesimals(xt):
    _, x, t, u = xt
    eqs = akx.determining_equations(D(u, t) - D(u, x, 2), u)
    assert eqs
    assert any("eta" in str(e) for e in eqs)


def test_symmetry_refusal_not_polynomial(xt):
    _, x, t, u = xt
    with pytest.raises(akx.PdeError) as ei:
        akx.pde_symmetries(D(u, t) - ak.sin(D(u, x, 2)), u)
    assert ei.value.code in ("E-PDE-300", "E-PDE-301")


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------


def test_kdv_conservation_laws(xt):
    _, x, t, u = xt
    eq = D(u, t) + u * D(u, x) + D(u, x, 3)
    laws = akx.conservation_laws(eq, u, order=2)
    assert len(laws) == 4
    assert all(law.verified for law in laws)
    for law in laws:
        st = akx.verify_conservation_law(eq, u, law.multiplier, law.components)
        assert st == "proved_zero"
    dens = {str(law.density) for law in laws}
    assert "u" in dens


def test_conserved_flux_refuses_non_multiplier(xt):
    p, x, t, u = xt
    eq = D(u, t) + u * D(u, x) + D(u, x, 3)
    with pytest.raises(akx.PdeError) as ei:
        akx.conserved_flux(eq, u, p.symbol("u_x"))
    assert ei.value.code == "E-PDE-306"


def test_euler_operator(xt):
    p, x, _, u = xt
    # L = u_x^2/2 - u^3/6  ->  E(L) = -u_xx - u^2/2, in jet symbols.
    L = D(u, x) ** 2 / 2 - u**3 / 6
    e = akx.euler_operator(L, u)
    expected = -p.symbol("u_xx") - p.symbol("u") ** 2 / 2
    assert str(ak.simplify(e - expected).value) == "0"


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda u, x, t: D(u, t) + 6 * u * D(u, x) + D(u, x, 3),  # KdV soliton
        lambda u, x, t: D(u, t) + u * D(u, x) - D(u, x, 2),  # Burgers shock
        lambda u, x, t: D(u, t) - D(u, x, 2) - u * (1 - u),  # Fisher front
        lambda u, x, t: D(D(u, x), t) - ak.sin(u),  # sine-Gordon kink
        lambda u, x, t: D(u, t, 2) - D(u, x, 2) - u + u**3,  # phi^4 kink
    ],
)
def test_traveling_waves_verified(xt, make):
    _, x, t, u = xt
    eq = make(u, x, t)
    sols = akx.traveling_wave_solutions(eq, u)
    assert sols
    for s in sols:
        assert s.verified
        assert akx.verify_pde_solution(akx.PDE(eq, u), s.value)["status"] == "exactly_verified"


def test_sine_gordon_kink_form(xt):
    _, x, t, u = xt
    sols = akx.traveling_wave_solutions(D(D(u, x), t) - ak.sin(u), u)
    assert any("atan(exp(" in str(s.value) for s in sols)


def test_similarity_heat_source(xt):
    _, x, t, u = xt
    sols = akx.similarity_solutions(D(u, t) - D(u, x, 2), u)
    assert any("exp(" in str(s.value) and "t^(-1/2)" in str(s.value) for s in sols)
    assert all(s.verified for s in sols)


def test_reduce_by_generator(xt):
    p, x, t, u = xt
    r = akx.reduce_by_generator(D(u, t) - D(u, x, 2), u, [x, 2 * t, p.integer(0)])
    assert "F" in str(r["ode"])


def test_ansatz_refuses_unknown_method(xt):
    _, x, t, u = xt
    with pytest.raises(akx.PdeError):
        akx.ansatz_solutions(D(u, t) + u * D(u, x) - D(u, x, 2), u, method="nope")


def test_pdsolve_nonlinear_tiers(xt):
    _, x, t, u = xt
    sol = akx.pdsolve(D(u, t) + 6 * u * D(u, x) + D(u, x, 3), u)
    assert sol.verified
    assert sol.kind == "special"
    assert sol.method == "traveling_wave"
    sols = akx.pdsolve(D(u, t) + u * D(u, x) - D(u, x, 2), u, all=True)
    assert {s.method for s in sols} >= {"cole_hopf", "traveling_wave"}
    names = [h[0] for h in akx.pdsolve_hints()]
    assert {"cole_hopf", "traveling_wave", "similarity"} <= set(names)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_cole_hopf(xt):
    _, x, t, u = xt
    ch = akx.cole_hopf(D(u, t) + 2 * u * D(u, x) - 3 * D(u, x, 2), u)
    assert ch["verified"]
    with pytest.raises(akx.PdeError) as ei:
        akx.cole_hopf(D(u, t) + u * D(u, x) + D(u, x, 3), u)
    assert ei.value.code == "E-PDE-304"


def test_miura(xt):
    _, x, t, u = xt
    w = akx.Function("w")(x, t)
    m = akx.miura(D(u, t) - 6 * u**2 * D(u, x) + D(u, x, 3), u, w)
    assert m["verified"]


def test_hodograph(xt):
    _, x, t, u = xt
    h = akx.Function("h")(x, t)
    eqs = [D(u, t) + u * D(u, x) + D(h, x), D(h, t) + h * D(u, x) + u * D(h, x)]
    r = akx.hodograph(eqs, [u, h])
    assert r["verified"]
    assert len(r["equations"]) == 2


def test_hopf_lax(xt):
    _, x, t, u = xt
    r = akx.hopf_lax(D(u, t) + D(u, x) ** 2 / 2, u)
    assert r["verified"]
    assert "g(y)" in str(r["objective"])
