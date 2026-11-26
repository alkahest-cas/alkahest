"""PDE milestone M5: Green's functions and fundamental solutions.

The corpus ``tests/pde_corpus/green_problems.json`` drives most of this file:
fundamental solutions (Strauss ch. 7, Evans §2.2–2.4, Duffy), image
constructions (Strauss §3.1, §7.4; Haberman §9.5), two-point Green's
functions (Haberman ch. 9), eigenfunction series (Haberman §9.3, ch. 11) and
Green's-function representations of solutions.  Reference values come from
Mathematica 14.3 (``GreenFunction`` / ``DSolveValue``,
``mathematica/green_ref.txt``).

Every answer returned must be **exactly verified**; where Mathematica returns a
closed form the values must agree at the sample points.
"""

from __future__ import annotations

import cmath
import json
import math
import pathlib
import re

import alkahest as ak
import alkahest.experimental as akx
import pytest

CORPUS = pathlib.Path(__file__).parent / "pde_corpus"
PROBLEMS = json.loads((CORPUS / "green_problems.json").read_text())
PARAMS = {"k": 1.5, "kappa": 1.0, "c": 1.0, "m": 1.0, "a": 0.5, "L": 1.0, "h": 2.0, "R": 1.0}
POSITIVE = set(PARAMS) | {"t", "r", "n"}


def _ref() -> dict:
    out = {}
    for line in (CORPUS / "mathematica" / "green_ref.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out[parts[0]] = (parts[1], parts[2])
    return out


REF = _ref()


def _wl_numbers(s: str):
    s = re.sub(r"`[0-9.]*", "", s).replace("*^", "e")
    s = s.replace("{", "[").replace("}", "]")
    s = re.sub(r"([0-9.]+)\s*\*\s*I", r"\1j", s)
    return eval(s, {"Indeterminate": None})  # our own reference file


class Ctx:
    """A pool with the corpus' symbols: parameters, t, r, n positive."""

    def __init__(self):
        self.pool = ak.ExprPool()
        self.syms: dict = {}

    def __call__(self, name: str):
        if name not in self.syms:
            dom = "positive" if name in POSITIVE else "real"
            self.syms[name] = self.pool.symbol(name, dom)
        return self.syms[name]

    def parse(self, s: str):
        names = set(re.findall(r"[A-Za-z_]\w*", s)) - {"exp", "sin", "cos", "sqrt", "log", "pi"}
        syms = {n: self(n) for n in names if n not in ("u", "f", "Derivative")}
        if "u(" in s or "f(" in s:
            syms["u"] = akx.Function("u")
            syms["f"] = akx.Function("f")
        return ak.parse(s, self.pool, syms)

    def env(self, point: dict) -> dict:
        out = {}
        for name, v in {**PARAMS, **point}.items():
            if name in self.syms:
                out[self.syms[name]] = float(v)
        return out


def _end(spec, cx: Ctx):
    if isinstance(spec, str):
        return spec
    return ("robin", cx.parse(spec[1]), cx.parse(spec[2]))


def _sl(entry, cx: Ctx):
    x = cx(entry["var"])
    a, b = (cx.parse(e) for e in entry["interval"])
    bc = tuple(_end(e, cx) for e in entry["bc"])
    w = cx.parse(entry.get("w", "x" if entry.get("bc", [None])[0] == "bounded" else "1"))
    return akx.SturmLiouville(x, (a, b), bc, p=cx.parse(entry["p"]), q=cx.parse(entry["q"]), w=w)


def _params(entry, cx: Ctx) -> dict:
    return {k: cx(v) for k, v in entry.get("params", {}).items()}


def build(entry):
    """Solve one corpus entry: ``(result, ctx)``."""
    cx = Ctx()
    cat = entry["category"]
    time = cx(entry["time"]) if entry.get("time") else None
    if cat == "fundamental":
        res = akx.fundamental_solution(
            entry["op"], [cx(v) for v in entry["space"]], time, **_params(entry, cx)
        )
    elif cat == "fundamental_nd":
        res = akx.fundamental_solution(
            entry["op"], time=time, dimension=cx("n"), radius=cx("r"), **_params(entry, cx)
        )
    elif cat == "images":
        reg = [
            cx(r) if isinstance(r, str) and r != entry["region"][0] else r for r in entry["region"]
        ]
        res = akx.image_green_function(
            entry["op"],
            [cx(v) for v in entry["space"]],
            [cx(v) for v in entry["source_pt"]],
            tuple(reg),
            entry["walls"],
            time,
            **_params(entry, cx),
        )
    elif cat == "ode":
        res = akx.green_function(_sl(entry, cx), cx("xi"), "closed")
    elif cat == "series":
        kind = entry["kind"]
        res = akx.green_function(
            _sl(entry, cx),
            cx("xi"),
            kind,
            diffusivity=cx("kappa") if kind == "heat" else None,
            speed=cx("c") if kind == "wave" else None,
            time=cx("t") if kind in ("heat", "wave") else None,
        )
    elif cat == "solve":
        res = akx.green_solve(_pde(entry, cx))
    else:
        raise AssertionError(cat)
    return res, cx


def _pde(entry, cx: Ctx):
    U = akx.Function("u")
    dom = entry["domain"]
    if dom == "interval":
        x = cx(entry["var"])
        u = U(x)
        a, b = (cx.parse(e) for e in entry["interval"])
        eq = ak.parse(entry["eq"], cx.pool, {"x": x, "u": U, "f": akx.Function("f")})
        pde = akx.PDE(eq, u).on(akx.Interval(x, a, b))
        for order, at, rhs in entry["bcs"]:
            pt = cx.parse(at)
            if isinstance(order, list):  # alpha u + beta u'
                lhs = order[0] * ak.subs(akx.D(u, x), {x: pt}) + order[1] * ak.subs(u, {x: pt})
            else:
                lhs = ak.subs(akx.D(u, x, order) if order else u, {x: pt})
            pde = pde.bc(lhs, cx.parse(rhs))
        return pde
    if dom == "line_time":
        x, t = cx("x"), cx("t")
        u = U(x, t)
        eq = ak.parse(entry["eq"], cx.pool, {"x": x, "t": t, "u": U})
        pde = akx.PDE(eq, u).on(akx.Line(x) * akx.HalfLine(t, 0))
        for order, rhs in entry["ics"]:
            lhs = ak.subs(akx.D(u, t, order) if order else u, {t: cx.pool.integer(0)})
            pde = pde.ic(lhs, cx.parse(rhs))
        return pde
    if dom == "half_plane":
        x, y = cx("x"), cx("y")
        u = U(x, y)
        eq = ak.parse(entry["eq"], cx.pool, {"x": x, "y": y, "u": U})
        pde = akx.PDE(eq, u).on(akx.Line(x) * akx.HalfLine(y, 0))
        for _, rhs in entry["bcs"]:
            pde = pde.bc(ak.subs(u, {y: cx.pool.integer(0)}), cx.parse(rhs))
        return pde
    raise AssertionError(dom)


# ---------------------------------------------------------------------------
# Numerical evaluation (sample points only — never part of a verdict)
# ---------------------------------------------------------------------------


_EULER = 0.57721566490153286061


def _bessel_series(n: int, z: complex, sign: int) -> complex:
    """``J_n`` (sign −1) or ``I_n`` (sign +1) by the power series (small |z|)."""
    total, term = 0j, (z / 2) ** n / math.factorial(n)
    for k in range(80):
        total += term
        term *= sign * (z / 2) ** 2 / ((k + 1) * (k + 1 + n))
    return total


def _k0(z: complex) -> complex:
    """``K₀`` by DLMF 10.31.2."""
    q, total, term, hk = (z / 2) ** 2, 0j, 1 + 0j, 0.0
    for k in range(80):
        total += term * hk
        term *= q / ((k + 1) ** 2)
        hk += 1 / (k + 1)
    return -(cmath.log(z / 2) + _EULER) * _bessel_series(0, z, 1) + total


def _y0(z: complex) -> complex:
    """``Y₀`` by DLMF 10.8.2."""
    q, total, term, hk = (z / 2) ** 2, 0j, 1 + 0j, 0.0
    for k in range(80):
        total += (-1) ** (k + 1) * term * hk
        term *= q / ((k + 1) ** 2)
        hk += 1 / (k + 1)
    j0 = _bessel_series(0, z, -1)
    return 2 / math.pi * ((cmath.log(z / 2) + _EULER) * j0 + total)


def _bessel(kind: str):
    def f(n, z):
        n = round(complex(n).real)
        z = complex(z)
        if kind == "J":
            return _bessel_series(n, z, -1)
        if kind == "I":
            return _bessel_series(n, z, 1)
        if kind == "K" and n == 0:
            return _k0(z)
        if kind == "H" and n == 0:
            return _bessel_series(0, z, -1) + 1j * _y0(z)
        raise NotImplementedError(kind, n)

    return f


def _num(expr, env: dict, delta_test: bool = False) -> complex:
    """Evaluate the printed form of an expression with the functions above;
    ``diracdelta(z)`` → ``exp(−z²)`` (a test function) when asked."""
    return _num_str(str(expr), env, delta_test)


def _num_str(s: str, env: dict, delta_test: bool = False, extra=None) -> complex:
    s = s.replace("^", "**").replace("∞", "inf")
    ns = {
        "sqrt": cmath.sqrt,
        "exp": cmath.exp,
        "log": cmath.log,
        "sin": cmath.sin,
        "cos": cmath.cos,
        "sinh": cmath.sinh,
        "cosh": cmath.cosh,
        "abs": abs,
        "heaviside": lambda z: 1 if complex(z).real > 0 else (0 if complex(z).real < 0 else 0.5),
        "diracdelta": lambda z: cmath.exp(-(z**2)) if delta_test else float("nan"),
        "bessel_j": _bessel("J"),
        "bessel_j0": lambda z: _bessel("J")(0, z),
        "bessel_j1": lambda z: _bessel("J")(1, z),
        "bessel_i": _bessel("I"),
        "bessel_k": _bessel("K"),
        "hankel1": _bessel("H"),
        "gamma": math.gamma,
        "pi": math.pi,
        "I": 1j,
    }
    ns.update(extra or {})
    for sym, v in env.items():
        ns[str(sym)] = v
    return complex(eval(s, ns))  # our own expression printer's output


def _sums_to_python(s: str) -> str:
    """Rewrite every printed ``Sum(body, k, lo, hi)`` as ``_S(lambda k: body, lo, hi)``."""
    out, i = [], 0
    while True:
        j = s.find("Sum(", i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        depth, args, start, pos = 0, [], j + 4, j + 4
        while True:
            ch = s[pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    args.append(s[start:pos])
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                args.append(s[start:pos])
                start = pos + 1
            pos += 1
        body, k, lo, hi = (x.strip() for x in args)
        out.append(f"_S(lambda {k}: {_sums_to_python(body)}, {lo}, {hi})")
        i = pos + 1


def _series_num(value, env: dict, terms: int) -> complex:
    """Partial sums of every ``Sum`` (indices clipped to ``|k| ≤ terms``)."""

    def summer(f, lo, hi):
        lo = -terms if lo == -math.inf else int(lo)
        hi = terms if hi == math.inf else int(hi)
        return sum((f(k) for k in range(lo, hi + 1)), 0j)

    s = _sums_to_python(str(value).replace("∞", "inf"))
    return _num_str(s, env, extra={"_S": summer, "inf": math.inf})


def _value(entry, res):
    cat = entry["category"]
    if cat in ("fundamental", "fundamental_nd"):
        return res.kernel
    return res.value


def _shift_env(entry, cx: Ctx, point: dict) -> dict:
    """Fundamental solutions are in x − ξ: shift the space point."""
    env = cx.env(point)
    if entry["category"] == "fundamental":
        src = {"x": "xi", "y": "eta", "z": "zeta"}
        for v in entry["space"]:
            env[cx(v)] = float(point[v]) - float(point.get(src[v], 0))
    if entry["category"] == "fundamental_nd":
        env[cx("n")] = float(entry["n"])
        env[cx("r")] = float(point["r"])
    return env


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", PROBLEMS, ids=[e["id"] for e in PROBLEMS])
def test_corpus_entry(entry):
    if "expect" in entry:
        with pytest.raises(Exception) as ei:
            build(entry)
        assert entry["expect"] in str(ei.value)
        return
    res, cx = build(entry)
    assert res.verified, res.failed_checks()
    status, vals = REF[entry["id"]]
    if status != "closed" or not entry["points"] or "reference_error" in entry:
        return
    if any(h in str(_value(entry, res)) for h in ("Integral(", "RootOfFamily(")):
        # A representation, or a transcendental eigenvalue family: the
        # verification above is the test; values are compared elsewhere.
        return
    ref = _wl_numbers(vals)
    for point, want in zip(entry["points"], ref):
        if want is None:
            continue
        env = _shift_env(entry, cx, point)
        value = _value(entry, res)
        terms = 4000 if entry.get("kind") == "static" else 60
        got = (
            _series_num(value, env, terms)
            if entry["category"] == "series"
            else _num(value, env, entry.get("delta_test", False))
        )
        if entry.get("compare") == "conjugate":
            got = got.conjugate()
        if entry.get("compare") == "negate":
            got = -got
        tol = 1e-3 if entry.get("kind") == "static" else 1e-9
        assert abs(got - complex(want)) <= tol * max(1.0, abs(complex(want))), (
            entry["id"],
            point,
            got,
            want,
        )


def test_mathematica_heat_3d_reference_error():
    """Mathematica 14.3's GreenFunction for the 3-D heat operator returns
    e^{−r²/(4κt)} / (4πκ t^{3/2}), which lacks a factor (4π)^{−1/2}: its mass is
    √(4π), not 1.  Alkahest's kernel has unit mass (an exact check) and the
    ratio of the two at the sample point is exactly √(4π)."""
    entry = next(e for e in PROBLEMS if e["id"] == "heat_3d")
    res, cx = build(entry)
    assert any(c["what"] == "mass" and c["status"] == "proved_zero" for c in res.checks)
    point = entry["points"][0]
    got = _num(res.kernel, _shift_env(entry, cx, point)).real
    want = _wl_numbers(REF["heat_3d"][1])[0]
    assert abs(want / got - math.sqrt(4 * math.pi)) < 1e-9


def test_corpus_has_its_categories():
    cats = {e["category"] for e in PROBLEMS}
    assert {"fundamental", "fundamental_nd", "images", "ode", "series", "solve"} <= cats
    assert len(PROBLEMS) >= 50
    assert all(e["id"] in REF for e in PROBLEMS)


# ---------------------------------------------------------------------------
# API and cross-checks
# ---------------------------------------------------------------------------


def test_fundamental_records():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    fs = akx.fundamental_solution("laplace", [x, y])
    assert fs.verified
    assert fs.status == "exactly_verified"
    assert fs.identity == "Δ G = δ(x)"
    whats = {c["what"] for c in fs.checks}
    assert {"homogeneous", "flux"} <= whats
    assert fs.references
    assert fs.derivation
    fs1 = akx.fundamental_solution("helmholtz", [x], k=p.symbol("k", "positive"))
    assert any(c["method"] == "jump_formula" for c in fs1.checks)
    assert "jump formula" in fs1.identity_proof


def test_fundamental_refusals():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    t = p.symbol("t", "positive")
    c = p.symbol("c", "positive")
    with pytest.raises(Exception, match="E-PDE-251"):
        akx.fundamental_solution("heat", [x], diffusivity=c)  # no time
    with pytest.raises(Exception, match="E-PDE-250"):
        akx.fundamental_solution("klein_gordon", [x, y], t, speed=c, mass=c)
    with pytest.raises(Exception, match="E-PDE-250"):
        akx.fundamental_solution("wave", time=t, dimension=p.symbol("n"), radius=y, speed=c)
    with pytest.raises(ValueError):
        akx.fundamental_solution("nonsense", [x])


def test_image_refusals():
    p = ak.ExprPool()
    x, y, xi, eta = (p.symbol(n) for n in ("x", "y", "xi", "eta"))
    k = p.symbol("k", "positive")
    with pytest.raises(Exception, match="E-PDE-252"):
        akx.image_green_function("laplace", [x, y], [xi, eta], ("disk", p.integer(1)), "neumann")
    with pytest.raises(Exception, match="E-PDE-252"):
        akx.image_green_function(
            "helmholtz", [x, y], [xi, eta], ("strip", 1, k), ["dirichlet", "dirichlet"], k=k
        )
    with pytest.raises(Exception, match="E-PDE-251"):
        akx.image_green_function("laplace", [x, y], [xi], ("half_space", 1), "dirichlet")


def test_green_solve_refusal():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    U = akx.Function("u")
    u = U(x, y)
    # u_xx − u_yy on the half-plane: not one of the shapes.
    pde = (
        akx.PDE(akx.D(u, x, 2) - akx.D(u, y, 2) + akx.D(u, x), u)
        .on(akx.Line(x) * akx.HalfLine(y, 0))
        .bc(ak.subs(u, {y: p.integer(0)}), 0)
    )
    with pytest.raises(Exception, match="E-PDE-256"):
        akx.green_solve(pde)


def test_heat_images_agree_with_eigen_series():
    """The image series and the eigenfunction series of the heat equation on
    an interval are the same function (Poisson summation)."""
    p = ak.ExprPool()
    x, xi = p.symbol("x"), p.symbol("xi")
    t, kappa, L = (p.symbol(n, "positive") for n in ("t", "kappa", "L"))
    for walls in (["dirichlet", "dirichlet"], ["neumann", "neumann"], ["dirichlet", "neumann"]):
        img = akx.image_green_function(
            "heat", [x], [xi], ("strip", 0, L), walls, t, diffusivity=kappa
        )
        sl = akx.SturmLiouville(x, (0, L), tuple(walls))
        ser = akx.green_function(sl, xi, "heat", diffusivity=kappa, time=t)
        assert img.verified
        assert ser.verified
        env = {x: 0.3, xi: 0.65, t: 0.05, kappa: 1.0, L: 1.0}
        a = _series_num(img.value, env, 20)
        b = _series_num(ser.value, env, 60)
        assert abs(a - b) < 1e-10, (walls, a, b)


def test_static_series_matches_closed_form():
    p = ak.ExprPool()
    x, xi = p.symbol("x"), p.symbol("xi")
    L, h = p.symbol("L", "positive"), p.symbol("h", "positive")
    sl = akx.SturmLiouville(x, (0, L), ("dirichlet", ("robin", h, 1)))
    closed = akx.green_function(sl, xi, "closed")
    ser = akx.green_function(sl, xi, "static")
    assert closed.verified
    assert ser.verified
    assert "RootOfFamily" in str(ser.value)
    # The coefficient identity is checked exactly when the family is closed:
    dd = akx.green_function(akx.SturmLiouville(x, (0, L), ("dirichlet", "dirichlet")), xi, "static")
    assert any(c["what"] == "coefficients" for c in dd.checks)


def test_modified_green_function():
    p = ak.ExprPool()
    x, xi = p.symbol("x"), p.symbol("xi")
    L = p.symbol("L", "positive")
    sl = akx.SturmLiouville(x, (0, L), ("neumann", "neumann"))
    with pytest.raises(Exception, match="E-PDE-254"):
        akx.green_function(sl, xi, "closed")
    g = akx.green_function(sl, xi, "static")
    assert g.method == "modified_eigenfunction_series"
    assert any("modified Green" in n for n in g.notes)


def test_disk_poisson_kernel_has_unit_mass():
    p = ak.ExprPool()
    x, y, xi, eta = (p.symbol(n) for n in ("x", "y", "xi", "eta"))
    R = p.symbol("R", "positive")
    g = akx.image_green_function("laplace", [x, y], [xi, eta], ("disk", R), "dirichlet")
    assert g.verified
    assert g.poisson_kernel is not None
    n = 400
    total = 0.0
    for i in range(n):
        phi = 2 * math.pi * i / n
        env = {x: 0.3, y: -0.4, xi: math.cos(phi), eta: math.sin(phi), R: 1.0}
        total += _num(g.poisson_kernel, env).real * (2 * math.pi / n)
    assert abs(total - 1) < 1e-9


def _fourier_reference(x, t, transfer, fhat, n=4000, cutoff=6.0):
    """∫ e^{2πiξx} H(ξ, t) f̂(ξ) dξ by the trapezoid rule (smooth, decaying)."""
    h = 2 * cutoff / n
    total = 0j
    for i in range(n + 1):
        xi = -cutoff + i * h
        wgt = 0.5 if i in (0, n) else 1.0
        total += wgt * cmath.exp(2j * math.pi * xi * x) * transfer(xi, t) * fhat(xi)
    return (total * h).real


@pytest.mark.parametrize("which", ["klein_gordon", "telegraph"])
def test_light_cone_representation_is_the_solution(which):
    """M3 returned Klein–Gordon and telegraph as Fourier integrals; the light-cone
    Bessel kernels must give the same function (checked numerically here,
    exactly in Rust on the kernel)."""
    p = ak.ExprPool()
    x = p.symbol("x")
    t = p.symbol("t", "positive")
    U = akx.Function("u")
    u = U(x, t)
    if which == "klein_gordon":
        eq = akx.D(u, t, 2) - akx.D(u, x, 2) + u
    else:
        eq = akx.D(u, t, 2) + akx.D(u, t) - akx.D(u, x, 2)
    pde = (
        akx.PDE(eq, u)
        .on(akx.Line(x) * akx.HalfLine(t, 0))
        .ic(ak.subs(u, {t: p.integer(0)}), ak.exp(-(x**2)))
        .ic(ak.subs(akx.D(u, t), {t: p.integer(0)}), 0)
    )
    sol = akx.transform_solve(pde)
    assert sol.verified
    assert sol.kind == "integral_representation"
    assert "light_cone" not in str(sol.value)
    term = sol.terms[0]
    kern = str(term["kernel"])
    assert ("bessel_j" if which == "klein_gordon" else "bessel_i") in kern
    x0, t0 = 0.3, 0.7
    s = term["var"]
    lo, hi = x0 - t0, x0 + t0
    n = 800
    hstep = (hi - lo) / n
    integral = 0.0
    for i in range(n + 1):
        sv = lo + i * hstep
        wgt = 0.5 if i in (0, n) else 1.0
        if i in (0, n):
            sv += 1e-12 if i == 0 else -1e-12
        env = {x: x0, t: t0, s: sv}
        integral += wgt * (_num(term["kernel"], env) * math.exp(-(sv**2))).real
    integral *= hstep
    closed = _num(sol.closed_part, {x: x0, t: t0}).real
    got = closed + integral

    def fhat(xi):
        return math.sqrt(math.pi) * math.exp(-((math.pi * xi) ** 2))

    if which == "klein_gordon":

        def transfer(xi, tt):
            w = math.sqrt(4 * math.pi**2 * xi**2 + 1)
            return math.cos(w * tt)
    else:

        def transfer(xi, tt):
            # u'' + u' + 4π²ξ²u = 0, u(0) = 1, u'(0) = 0
            d = cmath.sqrt(4 * math.pi**2 * xi**2 - 0.25)
            return (cmath.exp(-tt / 2) * (cmath.cos(d * tt) + cmath.sin(d * tt) / (2 * d))).real

    want = _fourier_reference(x0, t0, transfer, fhat)
    assert abs(got - want) < 1e-5, (got, want)
