"""Regression suite for ``experimental.dsolve`` over ``ode_corpus.CORPUS``.

Every answer ``dsolve`` returns is re-checked here, independently of its own
substitution gate: an explicit ``y(x)`` is substituted (constants fixed,
derivatives by ``alkahest.diff``) and the residual evaluated at sample points;
an implicit relation ``G(x, y) = 0`` is checked by implicit differentiation
(``y' = -Gx/Gy``, ``y'' = (d/dx + y' d/dy) y'``, ...) at sample points
``(x, y)``. A returned answer that fails this is a wrong answer and fails the
test. ``SOLVED`` is the floor: those entries must keep being solved.
"""

from __future__ import annotations

import cmath
import math
import re

import alkahest as ak
import alkahest.experimental as ex
import pytest
from ode_corpus import CORPUS, SYSTEMS

DSYMS = ["yp", "ypp", "yppp", "ypppp"]
CVALS = [1.7, 2.3, -0.5, 1.4]
XS = (0.2, 0.4, 0.7, 1.1, 2.3, 3.7)
YS = (0.45, 0.9, 1.6)

# Entries solved (explicitly or implicitly, and verified) when the suite was
# added. A regression drops one of these.
SOLVED = {
    *(f"cc{i:02d}" for i in range(1, 15)),
    *(f"cf{i:02d}" for i in range(1, 12)),
    *(f"eu{i:02d}" for i in range(1, 11)),
    "fo01", "fo02", "fo03", "fo04", "fo05", "fo06", "fo07", "fo08", "fo09",
    "fo10", "fo11", "fo12", "fo13", "fo14", "fo15", "fo16", "fo17", "fo18",
    "fo19", "fo20", "fo21", "fo22", "fo23", "fo25",
    "nl01", "nl02", "nl03", "nl04", "nl07", "nl08", "nl10",
    "ex01", "ex02", "ex03", "ex05",
    "sp01", "sp02", "sp03", "sp08", "sp09",
    *(f"ba{i:02d}" for i in range(1, 11)),
    # elliptic / special-quadrature / Lagrange round
    "nl05", "nl06", "nl09", "nl11", "nl12", "ex04", "sp04", "sp05", "sp06", "sp07", "fo24",
    *(f"nl{i}" for i in range(14, 20)),
    "fo26", "fo27", "sp10", "sp11", "ex06",
}  # fmt: skip


def _num(e, env):
    try:
        v = complex(ak.eval_expr(e, env))
    except Exception:
        # Values reached only through a complex intermediate.
        r = ak.evaluate(e, env, mode="complex")
        if r.status != "ok":
            return None
        v = complex(r.value)
    if not cmath.isfinite(v):
        return None
    return v


def _setup(entry):
    _oid, _cat, F, order, params = entry
    pool = ak.ExprPool()
    syms = {n: pool.symbol(n) for n in ["x", "y", *DSYMS[:order], *params]}
    eq = ak.parse(F, pool, dict(syms))
    return pool, syms, eq


def _bind(e, pool, consts, params, syms):
    m = {c: pool.float(v) for c, v in zip(consts, CVALS)}
    m.update({syms[k]: pool.float(v) for k, v in params.items()})
    return ak.subs(e, m) if m else e


def _check_explicit(entry, pool, syms, eq, sol):
    _oid, _cat, _F, order, params = entry
    x = syms["x"]
    y = _bind(sol["y_of_x"], pool, sol["constants"], params, syms)
    ders = [y]
    for _ in range(order):
        ders.append(ak.diff(ders[-1], x).value)
    m = {syms["y"]: ders[0]}
    for k in range(order):
        m[syms[DSYMS[k]]] = ders[k + 1]
    r = ak.subs(_bind(eq, pool, [], params, syms), m)
    ok = 0
    for xv in XS:
        v, s = _num(r, {x: xv}), _num(y, {x: xv})
        if v is None or s is None:
            continue
        assert abs(v) <= 1e-6 * (1 + abs(s)), f"residual {v} at x = {xv}"
        ok += 1
    return ok


def _check_implicit(entry, pool, syms, eq, sol):
    _oid, _cat, _F, order, params = entry
    x, ysym = syms["x"], syms["y"]
    g = _bind(sol["implicit_relation"], pool, sol["constants"], params, syms)
    gx, gy = ak.diff(g, x).value, ak.diff(g, ysym).value
    slope = -gx / gy
    ders = [slope]
    for _ in range(order - 1):
        d = ders[-1]
        ders.append(ak.diff(d, x).value + slope * ak.diff(d, ysym).value)
    m = {syms[DSYMS[k]]: ders[k] for k in range(order)}
    r = ak.subs(_bind(eq, pool, [], params, syms), m)
    ok = 0
    for xv in XS:
        for yv in YS:
            env = {x: xv, ysym: yv}
            v, s = _num(r, env), _num(ders[-1], env)
            if v is None or s is None:
                continue
            assert abs(v) <= 1e-6 * (1 + abs(s)), f"residual {v} at ({xv}, {yv})"
            ok += 1
    return ok


def _check_parametric(entry, pool, syms, eq, sol):
    """A curve ``(x(p), y(p))`` solves ``F(x, y, y') = 0`` when
    ``F(x(p), y(p), p) = 0`` and ``dy/dp = p·dx/dp`` along it."""
    _oid, _cat, _F, order, params = entry
    assert order == 1
    p = sol["parameter"]
    xp = _bind(sol["x_of_p"], pool, sol["constants"], params, syms)
    yp = _bind(sol["y_of_p"], pool, sol["constants"], params, syms)
    on_eq = ak.subs(
        _bind(eq, pool, [], params, syms), {syms["x"]: xp, syms["y"]: yp, syms["yp"]: p}
    )
    slope = ak.diff(yp, p).value - p * ak.diff(xp, p).value
    ok = 0
    for pv in (0.3, 0.7, 1.3, 2.1, -0.6):
        for r in (on_eq, slope):
            v, s = _num(r, {p: pv}), _num(xp, {p: pv})
            if v is None or s is None:
                continue
            assert abs(v) <= 1e-6 * (1 + abs(s)), f"parametric residual {v} at p = {pv}"
            ok += 1
    for line in sol["singular_solutions"]:
        r = ak.subs(eq, {syms["y"]: line, syms["yp"]: ak.diff(line, syms["x"]).value})
        for xv in XS:
            v = _num(r, {syms["x"]: xv})
            assert v is not None, f"singular solution {line} did not evaluate"
            assert abs(v) <= 1e-9, f"singular solution {line}: {v}"
    return ok


@pytest.mark.parametrize("entry", CORPUS, ids=[e[0] for e in CORPUS])
def test_corpus_entry(entry):
    oid, _cat, _F, order, _params = entry
    pool, syms, eq = _setup(entry)
    try:
        sols = ex.dsolve(eq, syms["x"], syms["y"], [syms[d] for d in DSYMS[:order]])
    except ak.AlkahestError as e:
        if oid in SOLVED:
            pytest.fail(f"{oid} regressed: {e}")
        return
    sol = sols[0]
    assert len(sol["constants"]) == order, "a general solution carries `order` constants"
    if sol["form"] == "parametric":
        ok = _check_parametric(entry, pool, syms, eq, sol)
    elif sol["form"] == "implicit":
        ok = _check_implicit(entry, pool, syms, eq, sol)
    else:
        ok = _check_explicit(entry, pool, syms, eq, sol)
    assert ok >= 2, f"{oid}: too few sample points evaluated"


def test_no_float_coefficients_in_constant_coefficient_answers():
    for entry in CORPUS:
        if not entry[1].startswith("const-coeff") and entry[1] != "euler":
            continue
        _pool, syms, eq = _setup(entry)
        order = entry[3]
        try:
            sols = ex.dsolve(eq, syms["x"], syms["y"], [syms[d] for d in DSYMS[:order]])
        except ak.AlkahestError:
            continue
        y = sols[0]["y_of_x"]
        assert not _has_float(y), f"{entry[0]}: {y}"


_FLOAT = re.compile(r"\d\.\d+e[-+]?\d+|\d+\.\d{5,}")


def _has_float(e):
    return _FLOAT.search(str(e)) is not None


def test_corpus_size():
    assert len(CORPUS) >= 90
    assert math.isfinite(len(SOLVED))


@pytest.mark.parametrize("entry", SYSTEMS, ids=[e[0] for e in SYSTEMS])
def test_system_entry(entry):
    _oid, names, rhs_src = entry
    pool = ak.ExprPool()
    t = pool.symbol("t")
    vs = [pool.symbol(n) for n in names]
    table = {"t": t, **dict(zip(names, vs))}
    rhs = [ak.parse(r, pool, dict(table)) for r in rhs_src]
    sol = ex.dsolve_system(ak.ODE(vs, rhs, t))
    ys = sol["y_of_t"]
    assert len(sol["constants"]) == len(vs)
    for y in ys:
        assert not _has_float(y), str(y)
    cvals = {c: pool.float(v) for c, v in zip(sol["constants"], CVALS)}
    ys = [ak.subs(y, cvals) for y in ys]
    for i, (y, r) in enumerate(zip(ys, rhs)):
        lhs = ak.diff(y, t).value
        rr = ak.subs(r, dict(zip(vs, ys)))
        for tv in XS:
            a, b = _num(lhs, {t: tv}), _num(rr, {t: tv})
            assert a is not None, f"component {i} did not evaluate"
            assert b is not None, f"component {i} did not evaluate"
            assert abs(a - b) <= 1e-8 * (1 + abs(a)), f"component {i}: {a} vs {b} at t = {tv}"


@pytest.mark.parametrize("oid", ["sp04", "sp05", "sp10"])
def test_hermite_second_solutions_use_real_erfi(oid):
    # `erfi` is a real primitive: no `-i*erf(i*z)` spelling, and the solution
    # evaluates in plain real arithmetic.
    entry = next(e for e in CORPUS if e[0] == oid)
    pool, syms, eq = _setup(entry)
    sol = ex.dsolve(eq, syms["x"], syms["y"], [syms[d] for d in DSYMS[:2]])[0]
    shown = str(sol["y_of_x"])
    assert "erfi(" in shown, shown
    assert "erf(" not in shown, shown
    assert "I" not in shown, shown
    y = _bind(sol["y_of_x"], pool, sol["constants"], {}, syms)
    assert math.isfinite(ak.eval_expr(y, {syms["x"]: 0.7}))
