"""PDE milestone M3: the Sturm–Liouville solver and the transform methods.

The corpus in ``tests/pde_corpus/`` drives most of this file:

* ``sturm_liouville_problems.json`` — eigenproblems from Haberman ch. 2/5/7,
  Strauss ch. 4/9/10 and DLMF 18; reference eigenvalues in
  ``mathematica/sturm_liouville_ref.txt`` (Mathematica 14.3:
  ``DEigenvalues`` where it solves the problem symbolically, otherwise an
  independent route — ``FindRoot`` on the textbook eigenvalue condition,
  ``BesselJZero``);
* ``transform_problems.json`` — heat / wave / Laplace / Schrödinger /
  Klein–Gordon / telegraph problems on the line and the half-line; reference
  values in ``mathematica/transforms_ref.txt`` (``DSolveValue``).

Every family / solution returned must be **exactly verified**; the numbers
must match the references.
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import alkahest as ak
import alkahest.experimental as akx
import pytest

CORPUS = pathlib.Path(__file__).parent / "pde_corpus"


def _ref_lines(name: str) -> dict:
    out = {}
    for line in (CORPUS / "mathematica" / name).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out[parts[0]] = parts[1:]
    return out


def _wl_numbers(s: str):
    s = re.sub(r"`[0-9.]*", "", s).replace("*^", "e")
    s = s.replace("{", "[").replace("}", "]").replace("*I", "j")
    s = re.sub(r"(\d)\s*j", r"\1j", s)
    return eval(s, {"Indeterminate": None})  # noqa: S307 - our own reference file


SL_PROBLEMS = json.loads((CORPUS / "sturm_liouville_problems.json").read_text())
SL_REF = _ref_lines("sturm_liouville_ref.txt")
TR_PROBLEMS = json.loads((CORPUS / "transform_problems.json").read_text())
TR_REF = _ref_lines("transforms_ref.txt")


def _end(spec, pool, syms):
    if isinstance(spec, str):
        return spec
    return (spec[0], ak.parse(spec[1], pool, syms), ak.parse(spec[2], pool, syms))


def _solve_sl(entry, enclosures=6):
    pool = ak.ExprPool()
    var = pool.symbol(entry["var"])
    syms = {entry["var"]: var}
    kind = entry["kind"]
    if kind == "general":
        P = lambda s: ak.parse(s, pool, syms)  # noqa: E731
        bc = entry["bc"]
        bc = bc if bc == "periodic" else tuple(_end(e, pool, syms) for e in bc)
        return akx.sturm_liouville(
            var,
            (P(entry["a"]), P(entry["b"])),
            bc,
            p=P(entry["p"]),
            q=P(entry["q"]),
            w=P(entry["w"]),
            enclosures=enclosures,
        )
    if kind == "bessel":
        bc = entry["bc"]
        bc = bc if isinstance(bc, str) else _end(bc, pool, syms)
        sl = akx.SturmLiouville.bessel(
            var, ak.parse(entry["order"], pool), ak.parse(entry["radius"], pool), bc
        )
        return sl.solve(enclosures=enclosures)
    return getattr(akx.SturmLiouville, kind)(var).solve(enclosures=enclosures)


def _spectrum(es) -> list[float]:
    """The first eigenvalues, with multiplicity, from every part of the
    result: certified negative ones, unconditional isolated modes, and the
    families' enclosures."""
    vals = []
    if es.lower["status"] == "present":
        vals += [(lo + hi) / 2 for _, lo, hi in es.lower["family"].enclosures]
    for m in es.modes:
        if m.condition is None:
            vals.append(float(ak.evaluate(m.eigenvalue, {}).value))
    for f in es.families:
        for _, lo, hi in f.enclosures:
            vals += [(lo + hi) / 2] * len(f.eigenfunctions)
    return sorted(vals)


@pytest.mark.parametrize("entry", SL_PROBLEMS, ids=[e["id"] for e in SL_PROBLEMS])
def test_sturm_liouville_corpus(entry):
    es = _solve_sl(entry)
    for f in es.families:
        assert f.verification["status"] == "exactly_verified", (entry["id"], f.verification)
        assert f.closed_form == entry["closed_form"], (entry["id"], f.frequency)
        if not f.closed_form:
            assert f.defining_equation is not None
            assert "tan" not in str(f.defining_equation), "the defining equation must be pole-free"
    for m in es.modes:
        assert m.verification in ("exactly_verified",), (entry["id"], m)
    ref = [float(v) for v in _wl_numbers(SL_REF[entry["id"]][1])]
    got = _spectrum(es)
    n = min(len(ref), len(got))
    assert n >= 3, (entry["id"], got, ref)
    for a, b in zip(got[:n], ref[:n]):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (entry["id"], got, ref)


def test_sl_symbolic_robin_family_and_enclosures():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    L = pool.symbol("L", "positive")
    h = pool.symbol("h", "positive")
    es = akx.sturm_liouville(x, (0, L), ("dirichlet", ("robin", h, 1)))
    (f,) = es.families
    assert not f.closed_form
    assert f.enclosures == []  # symbolic parameters
    de = str(f.defining_equation)
    assert "sin" in de and "cos" in de and "h" in de
    assert es.lower["status"] == "none"
    assert es.modes == []
    encs = f.enclosures_with(3, {h: 1, L: 1})
    assert math.isclose((encs[0][1] + encs[0][2]) / 2, 2.028757838110434**2, rel_tol=1e-12)
    assert encs[0][1] <= encs[0][2]
    # the second member as an expression, and its eigenfunction
    lam2 = f.eigenvalue_at(2)
    assert "RootOfFamily" in str(lam2)
    (x2,) = f.eigenfunctions_at(2)
    assert "sin" in str(x2)


def test_sl_symbolic_h_conditional_lower_spectrum():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    h = pool.symbol("h")  # sign unknown
    es = akx.sturm_liouville(x, (0, 1), ("dirichlet", ("robin", h, 1)))
    assert es.lower["status"] == "conditional"
    assert "h" in str(es.lower["condition"])
    # λ = 0 is an eigenvalue exactly when h + 1 = 0
    (m,) = es.modes
    assert m.condition is not None and "h" in str(m.condition)


def test_sl_from_ode_and_closed_forms():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    lam = pool.symbol("lam")
    X = akx.Function("X")(x)
    eq = akx.D(X, x, 2) + lam * X
    es = akx.SturmLiouville.from_ode(eq, X, lam, (0, pool.symbol("pi")), ("dirichlet", "dirichlet")).solve()
    (f,) = es.families
    assert str(f.eigenvalue) == str(f.index**2)
    assert f.eigenvalue_at(3) == pool.integer(9)
    (x3,) = f.eigenfunctions_at(3)
    assert "sin" in str(x3)
    assert ak.simplify(f.norms_squared[0] - pool.symbol("pi") / 2) == pool.integer(0)


def test_sl_legendre_members_and_norms():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    es = akx.SturmLiouville.legendre(x).solve()
    (f,) = es.families
    assert f.first == 0
    assert f.verification["status"] == "exactly_verified"
    (p2,) = f.eigenfunctions_at(2)
    assert ak.simplify(p2 - (3 * x**2 - 1) / 2) == pool.integer(0)
    assert "legendre_p" in str(f.eigenfunctions[0])


def test_sl_errors():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    with pytest.raises(akx.PdeError) as e:
        akx.sturm_liouville(x, (0, 1), (("robin", 0, 0), "dirichlet"))
    assert "E-PDE-151" in str(e.value)
    with pytest.raises(akx.PdeError) as e:
        akx.sturm_liouville(x, (0, 1), ("bounded", "dirichlet"))
    assert "E-PDE-151" in str(e.value)


# ---------------------------------------------------------------------------
# Transform methods
# ---------------------------------------------------------------------------


def _build_transform_problem(entry):
    pool = ak.ExprPool()
    x, t = pool.symbol("x"), pool.symbol("t")
    u = akx.Function("u")(x, t)
    syms = {"x": x, "t": t, "u": akx.Function("u")}
    eq = ak.parse(entry["eq"], pool, syms)
    pde = akx.PDE(eq, u)
    if entry["domain"] == "line":
        pde = pde.on(akx.Line(x) * akx.HalfLine(t, 0))
    else:
        pde = pde.on(akx.HalfLine(x, 0) * akx.HalfLine(t, 0))
    for j, rhs in enumerate(entry["ic"]):
        lhs = ak.subs(akx.D(u, t, j) if j else u, {t: pool.integer(0)})
        pde = pde.ic(lhs, ak.parse(rhs, pool, syms))
    for bc in entry["bc"]:
        if entry["domain"] == "line":
            # Laplace in the half-plane: the condition at t (= y) = 0
            pde = pde.bc(ak.subs(u, {t: pool.integer(0)}), ak.parse(bc, pool, syms))
        else:
            order, rhs = bc
            lhs = ak.subs(akx.D(u, x, order) if order else u, {x: pool.integer(0)})
            pde = pde.bc(lhs, ak.parse(rhs, pool, syms))
    return pool, x, t, pde


def _value_at(pool, expr, x, t, px, pt):
    env = {x: px, t: pt}
    for name in ("k", "c"):
        env[pool.symbol(name)] = 1
    r = ak.evaluate(expr, env, mode="complex") if "I" in str(expr) else ak.evaluate(expr, env)
    return r.value


@pytest.mark.parametrize("entry", TR_PROBLEMS, ids=[e["id"] for e in TR_PROBLEMS])
def test_transform_corpus(entry):
    pool, x, t, pde = _build_transform_problem(entry)
    sol = akx.transform_solve(pde)
    assert sol.verification["status"] == "exactly_verified", (entry["id"], sol.verification, sol.value)
    if entry["expect"] != "any":
        assert sol.kind == entry["expect"], (entry["id"], sol.kind, sol.value)
    ref = TR_REF.get(entry["id"])
    if sol.kind == "closed_form" and ref and ref[0] == "closed" and "Integral" not in str(sol.value):
        want = _wl_numbers(ref[1])
        for (px, pt), w in zip([(0.3, 0.7), (1.1, 0.2)], want):
            got = _value_at(pool, sol.value, x, t, px, pt)
            if isinstance(w, (int, float, complex)) and got is not None:
                assert abs(complex(got) - complex(w)) < 1e-9 * max(1, abs(w)), (entry["id"], got, w)


def test_transform_heat_kernel_shape():
    entry = next(e for e in TR_PROBLEMS if e["id"] == "heat_line_symbolic")
    pool, x, t, pde = _build_transform_problem(entry)
    sol = akx.transform_solve(pde)
    (term,) = sol.terms
    assert str(term["density"]) == "f(s)"
    assert "exp" in str(term["kernel"])
    checks = {c[0]: c[2] for c in sol.verification["checks"]}
    assert checks["kernel_pde"] == "proved_zero"


def test_sine_cosine_transforms():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    w = pool.symbol("w", "positive")
    a = pool.symbol("a", "positive")
    f = ak.exp(-a * x)
    s = akx.fourier_sine_transform(f, x, w)
    c = akx.fourier_cosine_transform(f, x, w)
    assert ak.simplify(s - w / (a**2 + w**2)) == pool.integer(0)
    assert ak.simplify(c - a / (a**2 + w**2)) == pool.integer(0)
    back = akx.inverse_fourier_cosine_transform(c, w, x)
    assert ak.simplify(back - f) == pool.integer(0)
    with pytest.raises(Exception) as e:
        akx.fourier_sine_transform(ak.sin(x), x, w)
    assert "E-TRANSFORM-021" in str(e.value)


def test_transform_errors():
    pool = ak.ExprPool()
    x, t = pool.symbol("x"), pool.symbol("t")
    u = akx.Function("u")(x, t)
    burgers = akx.PDE(akx.D(u, t) + u * akx.D(u, x), u).on(akx.Line(x) * akx.HalfLine(t, 0))
    with pytest.raises(akx.PdeError) as e:
        akx.transform_solve(burgers)
    assert "E-PDE-160" in str(e.value)
    heat = akx.PDE(akx.D(u, t) - akx.D(u, x, 2), u).on(akx.Line(x) * akx.HalfLine(t, 0))
    with pytest.raises(akx.PdeError) as e:
        akx.transform_solve(heat)
    assert "E-PDE-161" in str(e.value)
