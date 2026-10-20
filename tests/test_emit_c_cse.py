"""C emission quality: common-subexpression elimination, constants, coverage.

``emit_c_vec`` used to print every shared sub-expression once per use: the
n = 4 pendulum's equations of motion came out as ~320 kB of C, and a
Black–Scholes price + gradient + Hessian as 35 kB with 664 ``sqrt`` and 131
``log`` calls.  ``pi`` was emitted as a free symbol (refused unless passed as an
input), and ``heaviside`` / ``lambert_w`` raised an uncoded ``ValueError``.

Every numeric check here compiles the emitted C with the system compiler and
compares it against an independent value — ``eval_expr`` (the interpreter),
``math`` or ``mpmath`` — at random points.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import shutil
import subprocess
import tempfile

import alkahest as ak
import pytest


def _cc_works(path):
    d = tempfile.mkdtemp()
    src = os.path.join(d, "t.c")
    out = os.path.join(d, "t.so")
    with open(src, "w") as f:
        f.write("#include <math.h>\ndouble g(double x){return erf(x) + log1p(x);}\n")
    try:
        r = subprocess.run(
            [path, "-O2", "-shared", "-fPIC", "-o", out, src, "-lm"], capture_output=True
        )
        return r.returncode == 0
    except OSError:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _find_cc():
    for cc in ("/usr/bin/gcc", "/usr/bin/cc", "cc", "gcc", "clang"):
        path = cc if os.path.isabs(cc) else shutil.which(cc)
        if path and os.access(path, os.X_OK) and _cc_works(path):
            return path
    return None


CC = _find_cc()
requires_cc = pytest.mark.skipif(CC is None, reason="no working C compiler")


class _Lib:
    """A compiled emitted function, callable from Python through ctypes."""

    def __init__(self, source, tmp_path):
        src = tmp_path / "f.c"
        so = tmp_path / "f.so"
        src.write_text("#include <math.h>\n" + source)
        # -std=c99 -pedantic: the emitted code must be portable C, not GNU C.
        r = subprocess.run(
            [
                CC,
                "-std=c99",
                "-pedantic",
                "-Werror",
                "-O2",
                "-shared",
                "-fPIC",
                "-o",
                str(so),
                str(src),
                "-lm",
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"emitted C does not compile:\n{r.stderr}\n{source[:2000]}"
        self.lib = ctypes.CDLL(str(so))

    def scalar(self, name, nargs):
        fn = getattr(self.lib, name)
        fn.restype = ctypes.c_double
        fn.argtypes = [ctypes.c_double] * nargs
        return fn

    def vector(self, name, nargs, nout):
        fn = getattr(self.lib, name)
        fn.restype = None
        fn.argtypes = [ctypes.c_double] * nargs + [ctypes.POINTER(ctypes.c_double)]

        def call(*args):
            out = (ctypes.c_double * nout)()
            fn(*args, out)
            return list(out)

        return call


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _count(code, call):
    return code.count(call + "(")


# ---------------------------------------------------------------------------
# Black–Scholes price + gradient + Hessian (report §4.4)
# ---------------------------------------------------------------------------


def _black_scholes():
    p = ak.ExprPool()
    S, K, r, s, T = (p.symbol(n) for n in ("S", "K", "r", "sigma", "T"))
    two = p.integer(2)

    def Phi(z):
        return (1 + ak.erf(z / ak.sqrt(two))) / 2

    d1 = (ak.log(S / K) + (r + s**2 / 2) * T) / (s * ak.sqrt(T))
    d2 = d1 - s * ak.sqrt(T)
    C = S * Phi(d1) - K * ak.exp(-r * T) * Phi(d2)
    vs = [S, K, r, s, T]
    g = [ak.diff(C, v).value for v in vs]
    H = [ak.diff(gi, v).value for gi in g for v in vs]
    return [C, *g, *H], vs


def test_black_scholes_greeks_share_their_transcendentals():
    outs, vs = _black_scholes()
    code = ak.emit_c_vec(outs, vs, ["S", "K", "r", "sigma", "T"], "bs_all")
    # One sqrt(T); sqrt(2) and sqrt(pi) are folded to literals.  Was 664 sqrt.
    assert _count(code, "sqrt") <= 2, code
    # One log(S/K).  Was 131.
    assert _count(code, "log") == 1, code
    # erf(d1), erf(d2) and the two Gaussian densities' exps, plus exp(-rT).
    assert _count(code, "erf") <= 2, code
    assert _count(code, "exp") <= 3, code
    # pi is a literal, not a parameter.
    assert "3.141592653589793" in code or "1.7724538509055159" in code
    assert len(code) < 12_000, len(code)  # was ~35 kB
    # 383 `*` and 115 `+`/`-` before operand-pair sharing, 172 and 104 after.
    assert code.count(" * ") < 200, code.count(" * ")
    assert _ops(code) < 110, _ops(code)


@requires_cc
def test_black_scholes_greeks_compiled_match_the_interpreter(tmp_path):
    outs, vs = _black_scholes()
    code = ak.emit_c_vec(outs, vs, ["S", "K", "r", "sigma", "T"], "bs_all")
    f = _Lib(code, tmp_path).vector("bs_all", 5, len(outs))
    rng = random.Random(7)
    for _ in range(40):
        pt = [rng.uniform(50, 150), rng.uniform(50, 150), rng.uniform(0.0, 0.1)]
        pt += [rng.uniform(0.1, 0.6), rng.uniform(0.1, 3.0)]
        got = f(*pt)
        env = dict(zip(vs, pt))
        for i, e in enumerate(outs):
            want = ak.eval_expr(e, env)
            # Absolute floor: second-order greeks near zero cancel.
            assert abs(got[i] - want) <= 1e-12 * max(abs(want), 1.0) * 100, (i, pt, got[i], want)
    # Independent oracle for the price: the closed form via math.erf.
    S0, K0, r0, s0, T0 = 100.0, 95.0, 0.05, 0.2, 1.0
    d1 = (math.log(S0 / K0) + (r0 + s0 * s0 / 2) * T0) / (s0 * math.sqrt(T0))
    d2 = d1 - s0 * math.sqrt(T0)

    def N(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    price = S0 * N(d1) - K0 * math.exp(-r0 * T0) * N(d2)
    assert _rel(f(S0, K0, r0, s0, T0)[0], price) < 1e-13


# ---------------------------------------------------------------------------
# n = 4 pendulum equations of motion (report §4.2, embedded autocode)
# ---------------------------------------------------------------------------


def _pendulum(n):
    p = ak.ExprPool()
    integer = p.integer
    q = [p.symbol(f"q{i}") for i in range(n)]
    qd = [p.symbol(f"qd{i}") for i in range(n)]
    qdd = [p.symbol(f"qdd{i}") for i in range(n)]
    m = [p.symbol(f"m{i}") for i in range(n)]
    L = [p.symbol(f"L{i}") for i in range(n)]
    g = p.symbol("g")

    def D(e, v):
        return ak.diff(e, v).value

    def ddt(e):
        return sum((D(e, q[j]) * qd[j] + D(e, qd[j]) * qdd[j] for j in range(n)), integer(0))

    x = integer(0)
    y = integer(0)
    T = integer(0)
    V = integer(0)
    for i in range(n):
        x = x + L[i] * ak.sin(q[i])
        y = y - L[i] * ak.cos(q[i])
        T = T + m[i] * (ddt(x) ** 2 + ddt(y) ** 2) / 2
        V = V + m[i] * g * y
    Lg = T - V
    eom = [ddt(D(Lg, qd[i])) - D(Lg, q[i]) for i in range(n)]
    Mm = ak.jacobian(eom, qdd)
    Ms = ak.Matrix([[ak.simplify_trig(Mm.get(i, j)).value for j in range(n)] for i in range(n)])
    rhs = ak.Matrix([[ak.simplify(-ak.subs(e, dict.fromkeys(qdd, 0))).value] for e in eom])
    acc = Ms.inverse() @ rhs
    acc = [ak.simplify(acc.get(i, 0)).value for i in range(n)]
    return acc, m + L + [g] + q + qd


@pytest.fixture(scope="module")
def pendulum4():
    return _pendulum(4)


def test_pendulum_eom_c_is_compact(pendulum4):
    acc, inputs = pendulum4
    code = ak.emit_c_vec(acc, inputs, fn_name="npend")
    # 320 kB before CSE; aim for at least 5x smaller.  Measured ~25 kB.
    assert len(code) < 64_000, len(code)
    # Operand-pair sharing in the n-ary products and sums: 2882 `*` and 461
    # `+`/`-` before it, 953 and 455 after.
    assert code.count(" * ") < 1100, code.count(" * ")
    assert _ops(code) < 470, _ops(code)
    # Each sin(q_i) / cos(q_i) / sin(q_i - q_j) is computed once.
    for i in range(4):
        assert code.count(f"sin(q{i})") <= 1, code
        assert code.count(f"cos(q{i})") <= 1, code


@requires_cc
def test_pendulum_eom_compiled_matches_the_interpreter(pendulum4, tmp_path):
    acc, inputs = pendulum4
    code = ak.emit_c_vec(acc, inputs, fn_name="npend")
    f = _Lib(code, tmp_path).vector("npend", len(inputs), len(acc))
    rng = random.Random(1)
    for _ in range(25):
        pt = [rng.uniform(0.3, 1.5) for _ in range(8)] + [9.81]
        pt += [rng.uniform(-1.5, 1.5) for _ in range(4)] + [rng.uniform(-2, 2) for _ in range(4)]
        got = f(*pt)
        env = dict(zip(inputs, pt))
        for i, a in enumerate(acc):
            want = ak.eval_expr(a, env)
            assert _rel(got[i], want) < 1e-12, (i, pt, got[i], want)


# ---------------------------------------------------------------------------
# CSE and strength reduction on small cases
# ---------------------------------------------------------------------------


def test_shared_calls_are_computed_once_across_outputs():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    code = ak.emit_c_vec([ak.sin(x) * ak.cos(y), ak.sin(x) + ak.cos(y), ak.sin(x) ** 2], [x, y])
    assert _count(code, "sin") == 1, code
    assert _count(code, "cos") == 1, code


def test_integer_and_half_integer_powers_avoid_pow():
    p = ak.ExprPool()
    x = p.symbol("x")
    exprs = [
        x**2 + x**3 + x**4,
        x ** (-1) + x ** (-2),
        x ** p.rational(1, 2) + x ** p.rational(-1, 2) + x ** p.rational(3, 2),
    ]
    code = ak.emit_c_vec(exprs, x)
    assert "pow(" not in code, code
    assert _count(code, "sqrt") == 1, code
    # x*x is shared by x^2, x^3, x^4 and x^-2.
    assert code.count("(x * x)") == 1, code


@requires_cc
def test_strength_reduced_powers_match_the_interpreter(tmp_path):
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    exprs = [
        x**2 + x**3 * y - x**4 / y**2,
        x ** p.rational(3, 2) - y * x ** p.rational(-5, 2),
        (x + y) ** -3 - 2 * x / (y * (x + 1) ** 2),
        x**7 - x ** (-7) + x ** p.rational(1, 3),
    ]
    f = _Lib(ak.emit_c_vec(exprs, [x, y]), tmp_path).vector("eval_vec", 2, len(exprs))
    rng = random.Random(3)
    for _ in range(200):
        pt = [rng.uniform(0.1, 4.0), rng.uniform(-3.0, 3.0)]
        if abs(pt[1]) < 0.05:
            continue
        got = f(*pt)
        for i, e in enumerate(exprs):
            want = ak.eval_expr(e, {x: pt[0], y: pt[1]})
            assert _rel(got[i], want) < 1e-12, (i, pt, got[i], want)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_pi_and_e_are_literals():
    p = ak.ExprPool()
    x = p.symbol("x")
    pi = p.symbol("pi")
    code = ak.emit_c_expr(ak.sin(pi * x) + ak.exp(p.integer(1)) * x, x)
    assert "3.141592653589793" in code, code
    assert "2.718281828459045" in code, code
    assert "exp(" not in code, code
    # Binding pi explicitly still makes it a parameter.
    code2 = ak.emit_c_expr(ak.sin(pi * x), [x, pi], var_names=["x", "p"])
    assert "3.14159" not in code2
    assert "p" in code2


@requires_cc
def test_pi_compiled_value(tmp_path):
    p = ak.ExprPool()
    x = p.symbol("x")
    pi = p.symbol("pi")
    e = ak.sin(pi * x) / ak.sqrt(pi) + p.rational(1, 3) * x
    f = _Lib(ak.emit_c_expr(e, x), tmp_path).scalar("f", 1)
    for v in (-2.5, -0.1, 0.3, 1.7, 9.0):
        want = math.sin(math.pi * v) / math.sqrt(math.pi) + v / 3
        assert _rel(f(v), want) < 1e-14, v


# ---------------------------------------------------------------------------
# Function coverage
# ---------------------------------------------------------------------------


@requires_cc
def test_piecewise_and_step_functions_match_the_interpreter(tmp_path):
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    pw = ak.piecewise([(p.lt(x, p.integer(0)), -x), (p.le(x, y), x * y)], x**2)
    exprs = [
        p.func("heaviside", [x]),
        ak.sign(x - y),
        ak.min(x, y) + ak.max(x, y),
        ak.abs(x) + ak.floor(y) + ak.ceil(x),
        pw,
        ak.asinh(x) + ak.atanh(y / 4) + ak.acosh(ak.abs(x) + 1),
        ak.erfc(x) + ak.gamma(ak.abs(y) + p.rational(1, 2)),
    ]
    f = _Lib(ak.emit_c_vec(exprs, [x, y]), tmp_path).vector("eval_vec", 2, len(exprs))
    pts = [(0.0, 0.0), (0.0, 1.0), (-1.5, 2.0), (2.0, -1.0), (1.0, 1.0), (0.5, 3.0)]
    rng = random.Random(5)
    pts += [(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(100)]
    for xv, yv in pts:
        got = f(xv, yv)
        for i, e in enumerate(exprs):
            want = ak.eval_expr(e, {x: xv, y: yv})
            assert _rel(got[i], want) < 1e-12, (i, xv, yv, got[i], want)


@requires_cc
def test_lambert_w_helper_against_mpmath(tmp_path):
    mpmath = pytest.importorskip("mpmath")
    p = ak.ExprPool()
    x = p.symbol("x")
    code = ak.emit_c_expr(ak.lambert_w(x), x, fn_name="lw")
    assert "alkahest_lambert_w0" in code
    # Two emitted functions using the helper can share one translation unit.
    code2 = code + ak.emit_c_expr(ak.lambert_w(x) * 2, x, fn_name="lw2")
    lib = _Lib(code2, tmp_path)
    lw = lib.scalar("lw", 1)
    em = -1 / math.e
    rng = random.Random(11)
    pts = [1e-300, 1e-10, 1e-3, 0.5, 1.0, math.e, 10.0, 1e5, 1e100, 1e300, -1e-300, -0.1, -0.3]
    pts += [rng.uniform(em + 1e-3, 5.0) for _ in range(300)]
    pts += [10 ** rng.uniform(-300, 300) for _ in range(300)]
    for v in pts:
        want = float(mpmath.lambertw(mpmath.mpf(v)).real)
        assert _rel(lw(v), want) < 1e-14, (v, lw(v), want)
    assert lw(0.0) == 0.0
    assert lw(em) == -1.0
    assert math.isnan(lw(-0.5))
    assert lib.scalar("lw2", 1)(1.0) == 2 * lw(1.0)


# ---------------------------------------------------------------------------
# Coded refusals
# ---------------------------------------------------------------------------


def test_refusals_carry_codegen_codes():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(ak.AlkahestError) as ei:
        ak.emit_c_expr(p.func("diracdelta", [x]), x)
    assert ei.value.code == "E-CODEGEN-001"
    assert "diracdelta" in str(ei.value)
    assert isinstance(ei.value, ValueError)  # unchanged for existing callers
    with pytest.raises(ak.AlkahestError) as ei:
        ak.emit_c_vec([x + y], x)
    assert ei.value.code == "E-CODEGEN-003"
    with pytest.raises(ak.AlkahestError) as ei:
        ak.emit_c_expr(x + p.big_o(x**2), x)
    assert ei.value.code == "E-CODEGEN-002"


# ---------------------------------------------------------------------------
# Sharing operand pairs inside n-ary sums and products (F13)
# ---------------------------------------------------------------------------


def _ops(code):
    return code.count(" + ") + code.count(" - ")


def test_shared_pair_inside_two_sums_is_computed_once():
    p = ak.ExprPool()
    a, b, c, d = (p.symbol(n) for n in "abcd")
    code = ak.emit_c_vec([a + b + c, a + b + d], [a, b, c, d])
    # Four additions without sharing; a + b once and two more make three.
    assert _ops(code) == 3, code
    assert code.count("(a + b)") + code.count("(b + a)") == 1, code


def test_shared_pair_in_products_and_negated_sums():
    p = ak.ExprPool()
    a, b, c, d = (p.symbol(n) for n in "abcd")
    code = ak.emit_c_vec([a * b * c, a * b * d], [a, b, c, d])
    assert code.count(" * ") == 3, code
    # -a - b is the negation of a + b, and b - a of a - b: one shared node each.
    outs = [a + b + c, -a - b + d, a - b + c, b - a + d]
    code = ak.emit_c_vec(outs, [a, b, c, d])
    assert _ops(code) <= 6, code  # eight without sharing


@requires_cc
def test_pair_sharing_matches_the_interpreter(tmp_path):
    p = ak.ExprPool()
    xs = [p.symbol(f"x{i}") for i in range(9)]
    rng = random.Random(5)
    outs = []
    for _ in range(40):
        picks = rng.sample(xs, rng.randint(3, 8))
        e = None
        for t in picks:
            term = -t if rng.random() < 0.3 else t
            e = term if e is None else e + term
        outs.append(e)
        picks = rng.sample(xs, rng.randint(3, 6))
        m = picks[0]
        for t in picks[1:]:
            m = m * t
        outs.append(m)
    code = ak.emit_c_vec(outs, xs, fn_name="f")
    f = _Lib(code, tmp_path).vector("f", len(xs), len(outs))
    for _ in range(20):
        pt = [rng.uniform(-2, 2) for _ in xs]
        got = f(*pt)
        for o, g in zip(outs, got):
            want = ak.eval_expr(o, dict(zip(xs, pt)))
            assert abs(g - want) <= 1e-12 * max(1.0, abs(want)), (o, g, want)
