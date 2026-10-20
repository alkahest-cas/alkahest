"""C emission of special functions (F13): helpers checked against mpmath.

``emit_c_expr`` / ``emit_c_vec`` used to refuse every special function that is
not in ``<math.h>``.  Bessel J/Y/I/K of integer order, digamma, trigamma,
``polygamma`` of order 0/1, the dilogarithm, the complete elliptic integrals,
the Fresnel integrals and the exponential-integral family now lower to
self-contained ``static`` C99 helpers (no POSIX ``jn``/``yn``, no complex
type).  Every numeric check compiles the emitted C with the system compiler
(``-std=c99 -pedantic -Werror``) and compares it with mpmath at random points
of the helper's documented domain, and with ``eval_expr`` where the
interpreter has the function, so a convention slip (Fresnel's ``pi/2``,
elliptic parameter ``m = k^2``) cannot hide behind a self-consistent oracle.
"""

from __future__ import annotations

import math
import os
import random
import subprocess

import alkahest as ak
import pytest
from test_emit_c_cse import CC, _Lib, requires_cc

mpmath = pytest.importorskip("mpmath")
mpmath.mp.dps = 40


def _err(got, want, floor=0.0):
    """Error relative to ``max(|want|, floor)``; both infinite and equal is 0."""
    if math.isinf(want) or math.isinf(got):
        return 0.0 if got == want else math.inf
    return abs(got - want) / max(abs(want), floor, 1e-300)


# name -> (emitted function name, arity)
def _env():
    p = ak.ExprPool()
    return p, p.symbol("x"), p.symbol("n")


def _lib(tmp_path, builders, nargs=1):
    """Compile ``builders`` (name -> expression) into one translation unit."""
    p, x, n = _env()
    src = ""
    vs = [x] if nargs == 1 else [n, x]
    for name, build in builders.items():
        src += ak.emit_c_expr(build(p, x), vs, fn_name=name)
    return _Lib(src, tmp_path)


def _rng_points(rng, lo, hi, k, log=False):
    if log:
        return [10 ** rng.uniform(math.log10(lo), math.log10(hi)) for _ in range(k)]
    return [rng.uniform(lo, hi) for _ in range(k)]


# (symbol, mpmath reference, domain samplers, tolerance, floor)
# The floor is the scale the error is measured against where the function
# crosses zero: there the helpers promise an *absolute* error (see the comment
# above each helper in ``alkahest-core/src/horner/c_helpers``).
UNARY = {
    "digamma": (
        mpmath.digamma,
        [(0.01, 60, False), (-30, -0.01, False), (10, 1e12, True)],
        1e-13,
        0.05,
    ),
    "trigamma": (
        lambda x: mpmath.polygamma(1, x),
        [(0.01, 60, False), (-30, -0.01, False), (10, 1e12, True)],
        2e-15,
        0.0,
    ),
    "dilog": (lambda x: mpmath.polylog(2, x), [(-1000, 1, False), (-1e6, -1, False)], 2e-15, 0.0),
    "EllipticK": (mpmath.ellipk, [(-100, 0.999999, False), (-1e6, -1, False)], 2e-15, 0.0),
    "EllipticE": (mpmath.ellipe, [(-100, 1, False), (-1e6, -1, False)], 4e-15, 0.0),
    "fresnels": (mpmath.fresnels, [(-6, 6, False), (0, 40, False), (10, 1e6, True)], 4e-15, 0.1),
    "fresnelc": (mpmath.fresnelc, [(-6, 6, False), (0, 40, False), (10, 1e6, True)], 4e-15, 0.1),
    "Ei": (mpmath.ei, [(0.5, 60, False), (-60, -0.01, False), (40, 700, False)], 1e-14, 0.0),
    "li": (mpmath.li, [(1.6, 1e6, False), (0.01, 0.99, False)], 1e-14, 0.0),
    "Si": (mpmath.si, [(-50, 50, False), (10, 1e6, True)], 1e-14, 0.1),
    "Ci": (mpmath.ci, [(0.01, 50, False), (10, 1e6, True)], 1e-14, 0.1),
    "Shi": (mpmath.shi, [(-40, 40, False), (10, 700, False)], 1e-14, 0.0),
    "Chi": (mpmath.chi, [(0.01, 40, False), (10, 700, False)], 1e-14, 0.1),
}


@requires_cc
def test_unary_special_helpers_against_mpmath_and_interpreter(tmp_path):
    p, x, _ = _env()
    lib = _lib(tmp_path, {f"f_{k}": (lambda p, x, k=k: p.func(k, [x])) for k in UNARY})
    rng = random.Random(2026)
    for name, (ref, domains, tol, floor) in UNARY.items():
        f = lib.scalar(f"f_{name}", 1)
        for lo, hi, log in domains:
            for v in _rng_points(rng, lo, hi, 400, log):
                want = float(ref(mpmath.mpf(v)))
                got = f(v)
                assert _err(got, want, floor) < tol, (name, v, got, want)
        # The interpreter agrees on the convention at a spot value.
        v = 0.7 if name not in ("Ci", "Chi", "li") else 1.7
        if name == "dilog":
            v = 0.3
        spot = ak.eval_expr(p.func(name, [x]), {x: v})
        assert _err(f(v), spot, floor) < 1e-12, (name, v, f(v), spot)


@requires_cc
def test_special_helpers_domain_edges(tmp_path):
    lib = _lib(tmp_path, {f"f_{k}": (lambda p, x, k=k: p.func(k, [x])) for k in UNARY})
    inf, nan = math.inf, math.nan
    f = lambda name: lib.scalar(f"f_{name}", 1)  # noqa: E731
    for pole in (0.0, -1.0, -7.0):
        assert math.isnan(f("digamma")(pole))
        assert math.isnan(f("trigamma")(pole))
    assert math.isnan(f("digamma")(nan))
    assert math.isnan(f("digamma")(-inf))
    assert f("digamma")(inf) == inf
    assert f("trigamma")(inf) == 0.0
    assert math.isnan(f("dilog")(1.5))
    assert f("dilog")(-inf) == -inf
    assert f("dilog")(1.0) == pytest.approx(math.pi**2 / 6, rel=1e-15)
    assert f("dilog")(0.0) == 0.0
    assert f("EllipticK")(1.0) == inf
    assert math.isnan(f("EllipticK")(1.5))
    assert f("EllipticE")(1.0) == 1.0
    assert math.isnan(f("EllipticE")(2.0))
    assert f("EllipticK")(0.0) == pytest.approx(math.pi / 2, rel=1e-15)
    assert f("fresnels")(inf) == 0.5
    assert f("fresnelc")(-inf) == -0.5
    assert f("fresnels")(0.0) == 0.0
    assert f("Ei")(0.0) == -inf
    assert f("Ei")(inf) == inf
    assert f("Ei")(-inf) == 0.0
    assert f("Ei")(800.0) == inf
    assert f("li")(0.0) == 0.0
    assert f("li")(1.0) == -inf
    assert math.isnan(f("li")(-1.0))
    assert f("Si")(inf) == pytest.approx(math.pi / 2)
    assert f("Si")(-inf) == pytest.approx(-math.pi / 2)
    assert f("Si")(0.0) == 0.0
    assert f("Ci")(0.0) == -inf
    assert f("Ci")(inf) == 0.0
    assert math.isnan(f("Ci")(-1.0))
    assert f("Chi")(0.0) == -inf
    assert math.isnan(f("Chi")(-1.0))
    assert f("Shi")(0.0) == 0.0
    assert f("Shi")(800.0) == inf
    assert f("Shi")(-800.0) == -inf


# kind -> (mpmath reference, extra domain)
BESSEL = {
    "bessel_j": lambda n, x: mpmath.re(mpmath.besselj(n, x)),
    "bessel_y": lambda n, x: mpmath.bessely(n, x),
    "bessel_i": lambda n, x: mpmath.re(mpmath.besseli(n, x)),
    "bessel_k": lambda n, x: mpmath.besselk(n, x),
}


@requires_cc
def test_bessel_integer_order_against_mpmath(tmp_path):
    lib = _lib(
        tmp_path,
        {f"b_{k}": (lambda p, x, k=k: p.func(k, [p.symbol("n"), x])) for k in BESSEL},
        nargs=2,
    )
    rng = random.Random(7)
    for kind, ref in BESSEL.items():
        f = lib.scalar(f"b_{kind}", 2)
        oscillatory = kind in ("bessel_j", "bessel_y")
        regions = [(0.001, 2.0, 12), (2.0, 25.0, 30), (25.0, 300.0, 50)]
        for lo, hi, nmax in regions:
            for _ in range(300):
                order = rng.randint(0, nmax)
                v = rng.uniform(lo, hi)
                want = float(ref(order, mpmath.mpf(v)))
                # Near a zero of J/Y the error is absolute, measured against
                # the envelope sqrt(2/(pi x)).
                floor = math.sqrt(2 / (math.pi * max(v, 1))) if oscillatory else 0.0
                assert _err(f(float(order), v), want, floor) < 1e-14, (kind, order, v)
        # Large argument, and the sign rules for negative order / argument.
        for _ in range(60):
            order = rng.randint(0, 3)
            v = 10 ** rng.uniform(2, 5)
            floor = math.sqrt(2 / (math.pi * v)) if oscillatory else 0.0
            if kind == "bessel_i":
                v = rng.uniform(100, 700)
            if kind == "bessel_k":
                v = rng.uniform(100, 700)
            want = float(ref(order, mpmath.mpf(v)))
            assert _err(f(float(order), v), want, floor) < 1e-14, (kind, order, v)
        for order in (-3, -2, 5):
            v = rng.uniform(1.0, 12.0)
            if kind in ("bessel_j", "bessel_i"):
                for vv in (v, -v):
                    want = float(ref(order, mpmath.mpf(vv)))
                    assert _err(f(float(order), vv), want, 1e-3) < 1e-13, (kind, order, vv)
            else:
                want = float(ref(order, mpmath.mpf(v)))
                assert _err(f(float(order), v), want, 1e-3) < 1e-13, (kind, order, v)


@requires_cc
def test_bessel_edges_and_non_integer_runtime_order(tmp_path):
    lib = _lib(
        tmp_path,
        {f"b_{k}": (lambda p, x, k=k: p.func(k, [p.symbol("n"), x])) for k in BESSEL},
        nargs=2,
    )
    inf = math.inf
    j, y, i, k = (lib.scalar(f"b_{m}", 2) for m in BESSEL)
    # A symbolic order that is not an integer at run time gives NAN, never a
    # value for a different order.
    for f in (j, y, i, k):
        assert math.isnan(f(0.5, 1.0))
        assert math.isnan(f(math.nan, 1.0))
        assert math.isnan(f(20000.0, 1.0))
    assert j(0.0, 0.0) == 1.0
    assert j(3.0, 0.0) == 0.0
    assert j(2.0, inf) == 0.0
    assert y(0.0, 0.0) == -inf
    assert math.isnan(y(1.0, -1.0))
    assert y(1.0, inf) == 0.0
    assert i(0.0, 0.0) == 1.0
    assert i(2.0, -inf) == inf
    assert i(1.0, -inf) == -inf
    assert i(0.0, 720.0) == inf
    assert k(1.0, 0.0) == inf
    assert math.isnan(k(1.0, -1.0))
    assert k(0.0, inf) == 0.0
    # Overflow/underflow stay IEEE values rather than NaN.
    assert y(5.0, 1e-100) == -inf
    assert k(5.0, 1e-100) == inf
    assert k(0.0, 800.0) == 0.0
    # bessel_j0 / bessel_j1 lower to the same helper.
    lib2 = _lib(
        tmp_path,
        {"j0": lambda p, x: p.func("bessel_j0", [x]), "j1": lambda p, x: p.func("bessel_j1", [x])},
    )
    for v in (0.3, 2.5, 17.0, 40.0, 123.4):
        assert _err(lib2.scalar("j0", 1)(v), float(mpmath.besselj(0, v)), 0.05) < 1e-14
        assert _err(lib2.scalar("j1", 1)(v), float(mpmath.besselj(1, v)), 0.05) < 1e-14


@requires_cc
def test_polygamma_order_0_and_1_lower_to_digamma_trigamma(tmp_path):
    p, x, _ = _env()
    pg0 = ak.emit_c_expr(p.func("polygamma", [p.integer(0), x]), x, fn_name="pg0")
    pg1 = ak.emit_c_expr(p.func("polygamma", [p.integer(1), x]), x, fn_name="pg1")
    assert "alkahest_digamma" in pg0
    assert "alkahest_trigamma" in pg1
    lib = _Lib(pg0 + pg1, tmp_path)
    for v in (0.4, 1.3, 7.7, 25.0):
        assert _err(lib.scalar("pg0", 1)(v), float(mpmath.digamma(v)), 0.05) < 1e-13
        assert _err(lib.scalar("pg1", 1)(v), float(mpmath.polygamma(1, v))) < 1e-14
    with pytest.raises(ak.AlkahestError) as ei:
        ak.emit_c_expr(p.func("polygamma", [p.integer(2), x]), x)
    assert ei.value.code == "E-CODEGEN-001"


def test_remaining_refusals_stay_coded():
    p, x, _ = _env()
    for e in (
        p.func("bessel_j", [p.rational(1, 2), x]),  # non-integer literal order
        p.func("bessel_k", [p.float(0.5, 53), x]),
        p.func("EllipticE", [x, p.integer(2)]),  # incomplete
        p.func("EllipticF", [x, p.integer(2)]),
        p.func("airy_ai", [x]),
        p.func("zeta", [x]),
        p.func("polylog", [p.integer(3), x]),
        p.func("spherical_jn", [p.integer(1), x]),
    ):
        with pytest.raises(ak.AlkahestError) as ei:
            ak.emit_c_expr(e, x)
        assert ei.value.code == "E-CODEGEN-001", e


@requires_cc
def test_helpers_compile_warning_free_on_their_own(tmp_path):
    """Each emitted function compiles alone with ``-Wall -Wextra -Werror``: no
    unused ``static`` helper rides along."""
    p, x, _n = _env()
    cases = [p.func(k, [x]) for k in UNARY] + [p.func(k, [p.integer(3), x]) for k in BESSEL]
    for e in cases:
        src = "#include <math.h>\n" + ak.emit_c_expr(e, x, fn_name="f")
        path = tmp_path / "w.c"
        path.write_text(src)
        r = subprocess.run(
            [
                CC,
                "-std=c99",
                "-pedantic",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-O2",
                "-c",
                "-o",
                os.devnull,
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{e}:\n{r.stderr}"


@requires_cc
def test_special_functions_inside_larger_expressions(tmp_path):
    """Helpers coexist in one function, mixed with CSE'd elementary code."""
    p, x, _ = _env()
    y = p.symbol("y")
    e = (
        p.func("bessel_j", [p.integer(2), x]) * p.func("digamma", [y])
        + p.func("Ei", [x]) / (1 + p.func("Si", [y]) ** 2)
        + p.func("fresnelc", [x * y])
    )
    code = ak.emit_c_expr(e, [x, y], fn_name="f")
    for helper in ("bessel_j", "digamma", "ei", "si", "fresnel_c"):
        assert code.count(f"static double alkahest_{helper}(") == 1, helper
    f = _Lib(code, tmp_path).scalar("f", 2)
    rng = random.Random(3)
    for _ in range(50):
        a, b = rng.uniform(0.5, 9), rng.uniform(0.5, 9)
        assert _err(f(a, b), ak.eval_expr(e, {x: a, y: b}), 1e-3) < 1e-10
