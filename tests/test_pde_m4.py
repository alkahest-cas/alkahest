"""PDE milestone M4: separation of variables and eigenfunction series
(``alkahest.experimental.separate`` / ``separation_solve``)."""

from __future__ import annotations

import json
import math
import os
import sys

import alkahest as ak
import alkahest.experimental as akx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pde_corpus"))


def _heat_problem(data=None, bcs=("dirichlet", "dirichlet")):
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    L, k = p.symbol("L", "positive"), p.symbol("k", "positive")
    u = akx.Function("u")(x, t)
    pde = akx.PDE(akx.D(u, t) - k * akx.D(u, x, 2), u).on(
        akx.Interval(x, 0, L) * akx.HalfLine(t, 0)
    )
    for end, kind in zip((p.integer(0), L), bcs):
        lhs = u if kind == "dirichlet" else akx.D(u, x)
        pde = pde.bc(ak.subs(lhs, {x: end}), 0)
    pde = pde.ic(ak.subs(u, {t: p.integer(0)}), x * (L - x) if data is None else data(x, L))
    return p, x, t, L, k, u, pde


def test_separate_heat_and_polar():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    k = p.symbol("k", "positive")
    u = akx.Function("u")(x, t)
    s = akx.separate(akx.D(u, t) - k * akx.D(u, x, 2), u)
    assert s.kind == "product"
    assert len(s.constants) == 1
    assert len(s.odes) == 2
    assert "mu1" in str(s.odes[0])
    r, th = p.symbol("r", "positive"), p.symbol("theta")
    v = akx.Function("u")(r, th, t)
    eq = akx.D(v, t) - akx.curvilinear_laplacian(v, "polar", [r, th])
    s = akx.separate(eq, v)
    assert str(s.equations[0]["var"]) == "theta"
    assert len(s.constants) == 2
    assert akx.is_separable(eq, v)
    assert not akx.is_separable(akx.D(u, t) - akx.D(u, x, 2) - (x + t) ** 2 * u, u)
    with pytest.raises(Exception) as ei:
        akx.separate(akx.D(u, t) - akx.D(u, x, 2) - (x + t) ** 2 * u, u)
    assert "E-PDE-200" in str(ei.value)
    # Additive (Hamilton–Jacobi).
    s = akx.separate(akx.D(u, x) ** 2 + akx.D(u, t), u, kind="additive")
    assert s.kind == "additive"


def test_heat_series_verified_truncate_compile():
    _p, x, t, L, k, _u, pde = _heat_problem()
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert sol.kind == "series"
    assert sol.convergence == "uniform"
    assert "Sum(" in str(sol.value)
    assert sol.projection
    assert sol.projection[0]["status"] == "numerically_checked"
    g = sol.modes[0]
    assert g.indices
    assert str(g.indices[0][0]) == "n"
    # Truncation is explicit; compile evaluates fast.
    tr = sol.truncate(3)
    assert "Sum" not in str(tr)
    f = sol.compile(60, [x, t], {L: 1, k: 1})
    assert abs(f([0.5, 0.0]) - 0.25) < 1e-3
    # Exact solution value at t = 0.1 from the series with many terms.
    exact = sum(
        8 / (n * math.pi) ** 3 * math.sin(n * math.pi * 0.5) * math.exp(-((n * math.pi) ** 2) * 0.1)
        for n in range(1, 400, 2)
    )
    assert abs(f([0.5, 0.1]) - exact) < 1e-9


def test_closed_form_when_data_is_finite():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    c = p.symbol("c", "positive")
    u = akx.Function("u")(x, t)
    pi = ak.parse("pi", p)
    pde = (
        akx.PDE(akx.D(u, t, 2) - c**2 * akx.D(u, x, 2), u)
        .on(akx.Interval(x, 0, pi) * akx.HalfLine(t, 0))
        .bc(ak.subs(u, {x: p.integer(0)}), 0)
        .bc(ak.subs(u, {x: pi}), 0)
        .ic(ak.subs(u, {t: p.integer(0)}), ak.sin(x) + 3 * ak.sin(2 * x))
        .ic(ak.subs(akx.D(u, t), {t: p.integer(0)}), 0)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert sol.kind == "closed_form"
    assert pde.verify(sol.value)["status"] == "exactly_verified"


def test_neumann_robin_periodic_and_gibbs():
    p, x, t, _L, _k, u, pde = _heat_problem(data=lambda x, L: x, bcs=("neumann", "neumann"))
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert sol.convergence == "uniform"  # cosine series of x: continuous even extension
    # Robin end: RootOfFamily eigenvalues, numerically projected coefficients.
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    u = akx.Function("u")(x, t)
    one = p.integer(1)
    pde = (
        akx.PDE(akx.D(u, t) - akx.D(u, x, 2), u)
        .on(akx.Interval(x, 0, 1) * akx.HalfLine(t, 0))
        .bc(ak.subs(u, {x: p.integer(0)}), 0)
        .bc(ak.subs(akx.D(u, x), {x: one}) + ak.subs(u, {x: one}), 0)
        .ic(ak.subs(u, {t: p.integer(0)}), one)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert "RootOfFamily" in str(sol.value)
    assert sol.convergence == "l2"
    assert any("Gibbs" in n for n in sol.notes)
    assert sol.projection[0]["status"] == "numerically_checked"
    f = sol.compile(40, [x, t])
    assert 0.0 < f([0.5, 0.2]) < 1.0


def test_lifting_steady_state_and_duhamel():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    u = akx.Function("u")(x, t)
    base = akx.PDE(akx.D(u, t) - akx.D(u, x, 2), u).on(akx.Interval(x, 0, 1) * akx.HalfLine(t, 0))
    pde = (
        base.bc(ak.subs(u, {x: p.integer(0)}), 1)
        .bc(ak.subs(u, {x: p.integer(1)}), 2)
        .ic(ak.subs(u, {t: p.integer(0)}), 0)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert "steady_state" in sol.method
    assert str(sol.steady_state) in ("(1 + x)", "(x + 1)")
    pde = (
        base.bc(ak.subs(u, {x: p.integer(0)}), 0)
        .bc(ak.subs(u, {x: p.integer(1)}), ak.sin(t))
        .ic(ak.subs(u, {t: p.integer(0)}), 0)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert "lifting" in sol.method
    assert "duhamel" in sol.method
    # Symbolic boundary data: the Duhamel integral stays unevaluated and is
    # verified through Leibniz's rule.
    g = akx.Function("g")(t)
    pde = (
        base.bc(ak.subs(u, {x: p.integer(0)}), 0)
        .bc(ak.subs(u, {x: p.integer(1)}), g)
        .ic(ak.subs(u, {t: p.integer(0)}), 0)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert "Integral(" in str(sol.value)


def test_symbolic_data_unevaluated_coefficients():
    _p, _x, _t, _L, _k, _u, pde = _heat_problem(data=lambda x, L: akx.Function("f")(x))
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert "Integral(" in str(sol.value)
    assert any(c[3] == "projection_definition" for c in sol.verification["checks"])
    f = akx.Function("f")
    # A symbolic datum: the projection check has nothing to evaluate.
    assert all(pc["status"] != "refuted" for pc in sol.projection)
    del f


def test_disk_laplace_poisson_integral_and_cartesian_disk():
    p = ak.ExprPool()
    r, th = p.symbol("r", "positive"), p.symbol("theta")
    v = akx.Function("u")(r, th)
    pde = (
        akx.PDE(akx.curvilinear_laplacian(v, "polar", [r, th]), v)
        .on(akx.PolarDisk(r, th, 1))
        .bc(ak.subs(v, {r: p.integer(1)}), ak.sin(th) ** 3)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert sol.kind == "closed_form"
    assert any(n == "poisson_integral" for n, _ in sol.alternatives)
    # Heat on a Cartesian disk: solved in polar coordinates.
    x, y, t = p.symbol("x"), p.symbol("y"), p.symbol("t")
    u = akx.Function("w")(x, y, t)
    pde = (
        akx.PDE(akx.D(u, t) - akx.D(u, x, 2) - akx.D(u, y, 2), u)
        .on(akx.Disk((x, y), 1) * akx.HalfLine(t, 0))
        .bc(u, 0)
        .ic(ak.subs(u, {t: p.integer(0)}), 1 - x**2 - y**2)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert sol.coordinates[0] == "polar"
    assert "bessel_j0" in str(sol.value)


def test_ball_cylinder_annulus_sector():
    p = ak.ExprPool()
    r, th, t, z = p.symbol("r", "positive"), p.symbol("theta"), p.symbol("t"), p.symbol("z")
    v = akx.Function("u")(r, th)
    pde = (
        akx.PDE(akx.curvilinear_laplacian(v, "spherical", [r, th, None]), v)
        .on(akx.SphericalBall(r, th, None, 1))
        .bc(ak.subs(v, {r: p.integer(1)}), ak.cos(th) ** 2)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert sol.kind == "closed_form"
    w = akx.Function("u")(r, t)
    pde = (
        akx.PDE(akx.D(w, t) - akx.curvilinear_laplacian(w, "spherical", [r, None, None]), w)
        .on(akx.SphericalBall(r, None, None, 1) * akx.HalfLine(t, 0))
        .bc(ak.subs(w, {r: p.integer(1)}), 0)
        .ic(ak.subs(w, {t: p.integer(0)}), 1)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert "sin(" in str(sol.value)
    c = akx.Function("u")(r, z, t)
    pi = ak.parse("pi", p)
    pde = (
        akx.PDE(akx.D(c, t) - akx.curvilinear_laplacian(c, "cylindrical", [r, None, z]), c)
        .on(akx.Cylinder(r, None, z, 1, (0, 1)) * akx.HalfLine(t, 0))
        .bc(ak.subs(c, {r: p.integer(1)}), 0)
        .bc(ak.subs(c, {z: p.integer(0)}), 0)
        .bc(ak.subs(c, {z: p.integer(1)}), 0)
        .ic(ak.subs(c, {t: p.integer(0)}), (1 - r**2) * ak.sin(pi * z))
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    pde = (
        akx.PDE(akx.curvilinear_laplacian(v, "polar", [r, th]), v)
        .on(akx.Annulus(r, th, 1, 2))
        .bc(ak.subs(v, {r: p.integer(1)}), 0)
        .bc(ak.subs(v, {r: p.integer(2)}), ak.sin(th))
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    assert sol.kind == "closed_form"
    pde = (
        akx.PDE(akx.curvilinear_laplacian(v, "polar", [r, th]), v)
        .on(akx.Sector(r, th, 1, pi / 2))
        .bc(ak.subs(v, {th: p.integer(0)}), 0)
        .bc(ak.subs(v, {th: pi / 2}), 0)
        .bc(ak.subs(v, {r: p.integer(1)}), th * (pi / 2 - th))
    )
    sol = akx.separation_solve(pde)
    assert sol.verified


def test_errors():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    u = akx.Function("u")(x, t)
    # Nonlinear.
    pde = (
        akx.PDE(akx.D(u, t) - akx.D(u, x, 2) - u**2, u)
        .on(akx.Interval(x, 0, 1) * akx.HalfLine(t, 0))
        .bc(ak.subs(u, {x: p.integer(0)}), 0)
        .bc(ak.subs(u, {x: p.integer(1)}), 0)
        .ic(ak.subs(u, {t: p.integer(0)}), 1)
    )
    with pytest.raises(Exception, match="E-PDE-201"):
        akx.separation_solve(pde)
    # A missing condition.
    pde = (
        akx.PDE(akx.D(u, t) - akx.D(u, x, 2), u)
        .on(akx.Interval(x, 0, 1) * akx.HalfLine(t, 0))
        .bc(ak.subs(u, {x: p.integer(0)}), 0)
        .ic(ak.subs(u, {t: p.integer(0)}), 1)
    )
    with pytest.raises(Exception, match="E-PDE-203"):
        akx.separation_solve(pde)
    # Wave without initial velocity: refused, not assumed zero.
    pde = (
        akx.PDE(akx.D(u, t, 2) - akx.D(u, x, 2), u)
        .on(akx.Interval(x, 0, 1) * akx.HalfLine(t, 0))
        .bc(ak.subs(u, {x: p.integer(0)}), 0)
        .bc(ak.subs(u, {x: p.integer(1)}), 0)
        .ic(ak.subs(u, {t: p.integer(0)}), 1)
    )
    with pytest.raises(Exception, match="E-PDE-203"):
        akx.separation_solve(pde)
    # Neumann problem whose flux does not balance.
    y = p.symbol("y")
    v = akx.Function("u")(x, y)
    pde = (
        akx.PDE(akx.D(v, x, 2) + akx.D(v, y, 2), v)
        .on(akx.Interval(x, 0, 1) * akx.Interval(y, 0, 1))
        .bc(ak.subs(akx.D(v, x), {x: p.integer(0)}), 0)
        .bc(ak.subs(akx.D(v, x), {x: p.integer(1)}), 0)
        .bc(ak.subs(akx.D(v, y), {y: p.integer(0)}), 0)
        .bc(ak.subs(akx.D(v, y), {y: p.integer(1)}), 1)
    )
    with pytest.raises(Exception, match="E-PDE-207"):
        akx.separation_solve(pde)


def _fd_heat_robin(n=400, t_end=0.1):
    """Crank–Nicolson for u_t = u_xx on (0, 1), u(0) = 0, u_x(1) + u(1) = 0,
    u(x, 0) = 1 — an independent numerical reference."""
    np = pytest.importorskip("numpy")
    h = 1.0 / n
    x = np.linspace(0.0, 1.0, n + 1)
    u = np.ones(n + 1)
    u[0] = 0.0
    # Ghost point at x = 1: u_{n+1} = u_{n-1} - 2h u_n.
    a = np.zeros((n + 1, n + 1))
    for i in range(1, n):
        a[i, i - 1], a[i, i], a[i, i + 1] = 1.0, -2.0, 1.0
    a[n, n - 1], a[n, n] = 2.0, -2.0 - 2.0 * h
    a /= h * h
    a[0, :] = 0.0
    steps = 2000
    dt = t_end / steps
    eye = np.eye(n + 1)
    lhs = eye - 0.5 * dt * a
    rhs = eye + 0.5 * dt * a
    lhs[0, :] = 0.0
    lhs[0, 0] = 1.0
    rhs[0, :] = 0.0
    step = np.linalg.solve(lhs, rhs)
    for _ in range(steps):
        u = step @ u
    return x, u


def test_series_against_finite_differences():
    """The truncated Robin series agrees with an independent Crank–Nicolson
    solution at t = 0.1 (the series converges fast for t > 0)."""
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    u = akx.Function("u")(x, t)
    one = p.integer(1)
    pde = (
        akx.PDE(akx.D(u, t) - akx.D(u, x, 2), u)
        .on(akx.Interval(x, 0, 1) * akx.HalfLine(t, 0))
        .bc(ak.subs(u, {x: p.integer(0)}), 0)
        .bc(ak.subs(akx.D(u, x), {x: one}) + ak.subs(u, {x: one}), 0)
        .ic(ak.subs(u, {t: p.integer(0)}), one)
    )
    sol = akx.separation_solve(pde)
    assert sol.verified
    f = sol.compile(30, [x, t])
    xs, ref = _fd_heat_robin()
    for xv in (0.25, 0.5, 0.75, 1.0):
        i = round(xv * (len(xs) - 1))
        assert abs(f([xv, 0.1]) - ref[i]) < 2e-4, (xv, f([xv, 0.1]), ref[i])


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

with open(os.path.join(HERE, "pde_corpus", "separation_expected.json")) as _fh:
    EXPECTED = json.load(_fh)

# Seconds per corpus problem; a slower one is skipped as a performance gap
# unless it is on the regression list.
CORPUS_TIMEOUT = 120


def _corpus(name):
    with open(os.path.join(HERE, "pde_corpus", name)) as fh:
        return json.load(fh)


def _run_isolated(corpus, entry_id):
    """Solve one corpus problem in a child process, so a crash or a hang in
    one problem is reported against that problem, not the whole session."""
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    runner = os.path.join(HERE, "pde_corpus", "run_separation.py")
    try:
        cp = subprocess.run(
            [sys.executable, runner, "--one", corpus, entry_id],
            capture_output=True,
            text=True,
            timeout=CORPUS_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"id": entry_id, "status": "timeout"}
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("{")]
    if cp.returncode != 0 or not lines:
        return {
            "id": entry_id,
            "status": "crashed",
            "returncode": cp.returncode,
            "stderr": cp.stderr[-2000:],
        }
    return json.loads(lines[-1])


def _check_corpus_entry(corpus, entry):
    rec = _run_isolated(corpus, entry["id"])
    expected = EXPECTED.get(entry["id"])
    # Never a refuted answer, never a crash.
    assert rec["status"] not in ("refuted", "crashed"), rec
    if expected == "exactly_verified":
        assert rec["status"] == "exactly_verified", rec
    elif rec["status"] != "exactly_verified":
        # Not (yet) solved: reported as a skip with the reason, so the -v log
        # of a run is the corpus score.
        pytest.skip(f"{rec['status']}: {rec.get('code', '')} {rec.get('error', '')[:120]}")


@pytest.mark.parametrize("entry", _corpus("separation_corpus.json"), ids=lambda e: e["id"])
def test_textbook_corpus(entry):
    _check_corpus_entry("separation_corpus.json", entry)


@pytest.mark.parametrize("entry", _corpus("abbasi_bounded.json"), ids=lambda e: e["id"])
def test_abbasi_bounded_corpus(entry):
    _check_corpus_entry("abbasi_bounded.json", entry)
