"""Small API gaps left after 3.13.0.

1. ``ArbBall`` (and the results that carry one or that users get back from
   evaluation and root isolation) pickles, bit-for-bit, at any precision.
2. A user symbol named ``pi`` is stored under an invisibly marked name (#534);
   no code emitter may leak that marker, and the variable must not collide
   with ``M_PI`` / π or a name the emitted code declares.
3. ``Fraction(1, 2) ** x`` has the exact base ``1/2`` (Python 3.9–3.11 hand
   ``Expr.__rpow__`` a float).
"""

from __future__ import annotations

import ctypes
import math
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
from fractions import Fraction

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")

MARKER = "​"


# ---------------------------------------------------------------------------
# 1. Pickling
# ---------------------------------------------------------------------------


def _ball_state(b):
    return (b.prec, b.mid_exact(), b.rad_exact(), b.mid_str(), b.rad_str(), repr(b))


def _sin_ball(prec):
    """sin(1) as a ball at `prec` bits: a full-width mantissa and a radius."""
    p = ak.ExprPool()
    x = p.symbol("x")
    return ak.interval_eval(ak.sin(x), {x: ak.ArbBall(1.0, 0.0, prec)})


def _identity(obj):
    return obj


def _ball_in_worker(prec):
    return _sin_ball(prec)


@pytest.mark.parametrize("prec", [53, 300, 3400])
def test_arb_ball_round_trips_bit_for_bit(prec):
    b = _sin_ball(prec)
    assert b.prec == prec
    # A real full-precision midpoint, not a short dyadic.
    assert b.mid_exact().denominator.bit_length() > prec // 2
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        c = pickle.loads(pickle.dumps(b, protocol=proto))
        assert type(c) is ak.ArbBall
        assert _ball_state(c) == _ball_state(b)
    # Independent oracle: the round-tripped ball still encloses sin(1).
    c = pickle.loads(pickle.dumps(b))
    with mpmath.workprec(prec + 64):
        m, r = c.to_mpf()
        assert abs(mpmath.sin(1) - m) <= r


@pytest.mark.parametrize("prec", [53, 300, 3400])
def test_arb_ball_through_multiprocessing(prec):
    balls = [_sin_ball(prec), ak.ArbBall(0.1, 1e-300, prec), ak.ArbBall(-0.0, 0.0, prec)]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2) as pool:
        back = pool.map(_identity, balls)
        made = pool.map(_ball_in_worker, [prec])
    for a, b in zip(balls, back):
        assert _ball_state(a) == _ball_state(b)
    # A ball computed in the worker is the one computed here.
    assert _ball_state(made[0]) == _ball_state(_sin_ball(prec))


def test_arb_ball_special_values():
    for b in (
        ak.ArbBall(1.0, math.inf, 64),
        ak.ArbBall(-0.0, 0.0, 64),
        ak.ArbBall(math.inf, 0.0, 64),
        ak.ArbBall(-math.inf, 0.0, 64),
    ):
        c = pickle.loads(pickle.dumps(b))
        assert repr(c) == repr(b)
        assert (c.mid, c.rad, c.prec) == (b.mid, b.rad, b.prec) or math.isnan(b.mid)
        assert math.copysign(1.0, c.mid) == math.copysign(1.0, b.mid)
    n = pickle.loads(pickle.dumps(ak.ArbBall(math.nan, 0.0, 64)))
    assert math.isnan(n.mid)


def test_arb_ball_bad_payload_refused():
    with pytest.raises(ValueError):
        ak.ArbBall._from_pickle(64, (64, "f", 1, 0), (64, "f", -1, 0))  # negative radius
    with pytest.raises(ValueError):
        ak.ArbBall._from_pickle(64, (8, "f", 1 << 20, 0), (64, "+0", 0, 0))  # mantissa > prec
    with pytest.raises(ValueError):
        ak.ArbBall._from_pickle(64, (64, "?", 0, 0), (64, "+0", 0, 0))


def test_evaluation_result_round_trips():
    p = ak.ExprPool()
    x = p.symbol("x")
    r = ak.evaluate(ak.sin(x), {x: ak.ArbBall(1.0, 0.0, 300)}, mode="interval", precision_bits=300)
    assert r.enclosure is not None
    c = pickle.loads(pickle.dumps(r))
    for attr in (
        "status",
        "backend",
        "requested_mode",
        "requested_precision_bits",
        "achieved_precision_bits",
        "reason",
        "is_enclosure",
    ):
        assert getattr(c, attr) == getattr(r, attr), attr
    assert _ball_state(c.enclosure) == _ball_state(r.enclosure)
    assert repr(c.value) == repr(r.value)


def test_root_interval_round_trips():
    p = ak.ExprPool()
    x = p.symbol("x")
    roots = ak.real_roots(x**3 - 2, x)
    back = pickle.loads(pickle.dumps(roots))
    assert [(r.lo_exact(), r.hi_exact()) for r in back] == [
        (r.lo_exact(), r.hi_exact()) for r in roots
    ]
    # The isolating interval still contains 2^(1/3) (mpmath oracle).
    lo = Fraction(*map(int, back[0].lo_exact()))
    hi = Fraction(*map(int, back[0].hi_exact()))
    with mpmath.workdps(40):
        cbrt2 = mpmath.cbrt(2)
        assert mpmath.mpf(lo.numerator) / lo.denominator <= cbrt2
        assert cbrt2 <= mpmath.mpf(hi.numerator) / hi.denominator
    with pytest.raises(ValueError):
        ak.RootInterval._from_pickle(Fraction(2), Fraction(1))


def test_series_and_domain_round_trip():
    p = ak.ExprPool()
    x = p.symbol("x")
    s = ak.series(ak.sin(x), x, 0, 6)
    c = pickle.loads(pickle.dumps(s))
    assert type(c) is ak.Series
    assert c.expr == s.expr  # same process: lands in the same pool
    assert str(c.truncated()) == str(s.truncated())
    for d in (ak.Domain.Real, ak.Domain.Complex, ak.Domain.Integer, ak.Domain.Positive):
        assert pickle.loads(pickle.dumps(d)) == d
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        (s2, d2, r2) = pool.map(_identity, [(s, ak.Domain.NonZero, ak.real_roots(x**2 - 2, x))])[0]
    assert s2.expr == s.expr
    assert d2 == ak.Domain.NonZero
    assert [r.lo_exact() for r in r2] == [r.lo_exact() for r in ak.real_roots(x**2 - 2, x)]


# ---------------------------------------------------------------------------
# 2. A user symbol named `pi` in every emitter
# ---------------------------------------------------------------------------


def _cc_works(path):
    import tempfile

    d = tempfile.mkdtemp()
    try:
        src = os.path.join(d, "t.c")
        with open(src, "w") as f:
            f.write("#include <math.h>\ndouble f(double x) { return sin(x); }\n")
        r = subprocess.run(
            [path, "-shared", "-fPIC", "-o", os.path.join(d, "t.so"), src, "-lm"],
            capture_output=True,
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


@pytest.fixture
def cc():
    if CC is None:
        if os.environ.get("CI"):
            pytest.fail("no working C compiler in CI")
        pytest.skip("no working C compiler")
    return CC


def _compile(cc, source, tmp_path, name):
    src = tmp_path / f"{name}.c"
    so = tmp_path / f"{name}.so"
    src.write_text("#include <math.h>\n" + source)
    r = subprocess.run(
        [
            cc,
            "-std=c99",
            "-Wall",
            "-Werror",
            "-O1",
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
    assert r.returncode == 0, f"emitted C does not compile:\n{source}\n{r.stderr[:3000]}"
    return ctypes.CDLL(str(so))


@pytest.fixture
def user_pi():
    p = ak.ExprPool()
    u = p.symbol("pi")
    x = p.symbol("x")
    # The user variable, the constant π and a function of each.
    e = u * x + ak.sin(u) + p.pi()
    return p, u, x, e


def _want(u, x):
    return u * x + math.sin(u) + math.pi


def test_c_scalar_user_pi(cc, user_pi, tmp_path):
    _p, u, x, e = user_pi
    src = ak.emit_c_expr(e, [u, x])
    assert MARKER not in src
    assert "double f(double pi, double x)" in src
    fn = _compile(cc, src, tmp_path, "scalar").f
    fn.restype = ctypes.c_double
    fn.argtypes = [ctypes.c_double] * 2
    assert fn(2.0, 3.0) == pytest.approx(_want(2.0, 3.0), rel=1e-15)


def test_c_vec_user_pi(cc, user_pi, tmp_path):
    p, u, x, e = user_pi
    src = ak.emit_c_vec([e, u, p.pi()], [u, x])
    assert MARKER not in src
    fn = _compile(cc, src, tmp_path, "vec").eval_vec
    fn.restype = None
    fn.argtypes = [ctypes.c_double] * 2 + [ctypes.POINTER(ctypes.c_double)]
    out = (ctypes.c_double * 3)()
    fn(2.0, 3.0, out)
    assert list(out) == pytest.approx([_want(2.0, 3.0), 2.0, math.pi], rel=1e-15)


def test_c_names_never_collide(cc, tmp_path):
    """Variables named like what the emitted code or <math.h> declares."""
    p = ak.ExprPool()
    names = ["out", "sin", "M_PI", "_t0", "f", "double", "v", "in", "x y", "1x", "α"]
    vs = [p.symbol(n) for n in names]
    e = vs[0]
    for k, v in enumerate(vs[1:], start=2):
        e = e + k * v
    e = e + ak.sin(vs[0]) * ak.sin(vs[0]) + p.pi()
    args = [0.25 * (i + 1) for i in range(len(vs))]
    want = sum((i + 1) * a for i, a in enumerate(args)) + math.sin(args[0]) ** 2 + math.pi
    src = ak.emit_c_expr(e, vs)
    lib = _compile(cc, src, tmp_path, "names")
    fn = lib.f
    fn.restype = ctypes.c_double
    fn.argtypes = [ctypes.c_double] * len(vs)
    assert fn(*args) == pytest.approx(want, rel=1e-14)
    # Vector form: `out` must not shadow the output pointer.
    src = ak.emit_c_vec([e, vs[0]], vs)
    fv = _compile(cc, src, tmp_path, "names_vec").eval_vec
    fv.restype = None
    fv.argtypes = [ctypes.c_double] * len(vs) + [ctypes.POINTER(ctypes.c_double)]
    out = (ctypes.c_double * 2)()
    fv(*args, out)
    assert list(out) == pytest.approx([want, args[0]], rel=1e-14)


def test_c_missing_variable_names_the_plain_symbol(user_pi):
    _p, _u, x, e = user_pi
    with pytest.raises(ValueError) as info:
        ak.emit_c_expr(e, [x])
    assert MARKER not in str(info.value)
    assert "pi" in str(info.value)


def test_stablehlo_user_pi(user_pi):
    from test_stablehlo_refusals import _run

    _p, u, x, e = user_pi
    src = ak.to_stablehlo(e, [u, x])
    assert MARKER not in src
    assert _run(src, [2.0, 3.0]) == pytest.approx(_want(2.0, 3.0), rel=1e-15)
    with pytest.raises(ak.AlkahestError) as info:
        ak.to_stablehlo(e, [x])
    assert MARKER not in str(info.value)


def test_jax_user_pi(user_pi):
    pytest.importorskip("jax")
    _p, u, x, e = user_pi
    f = ak.to_jax(e, [u, x])
    import numpy as np

    got = float(np.asarray(f(np.array([2.0]), np.array([3.0])))[0])
    # JAX computes in float32 unless x64 is enabled; π for `pi` would be 12.57.
    assert got == pytest.approx(_want(2.0, 3.0), rel=1e-6)


def test_latex_and_unicode_user_pi(user_pi):
    p, u, _x, e = user_pi
    for text in (ak.latex(e), ak.unicode_str(e), str(e), repr(e)):
        assert MARKER not in text
    assert ak.latex(u) == "pi"
    assert ak.latex(p.pi()) == r"\pi"


def test_to_sympy_user_pi(user_pi):
    sp = pytest.importorskip("sympy")
    from alkahest.crosscheck import to_sympy

    p, _u, _x, e = user_pi
    s = to_sympy(e)
    names = {str(sym) for sym in s.free_symbols}
    assert names == {"pi", "x"}
    assert all(MARKER not in n for n in names)
    # The constant is sympy.pi, the variable a Symbol; bound to 2.0 they agree
    # with the direct evaluation.
    val = s.subs({sp.Symbol("pi"): 2, sp.Symbol("x"): 3})
    assert float(val) == pytest.approx(_want(2.0, 3.0), rel=1e-15)
    assert sp.simplify(to_sympy(ak.sin(p.pi()))) == 0
    assert to_sympy(p.pi()) is sp.pi


# ---------------------------------------------------------------------------
# 3. Fraction ** Expr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [Fraction(1, 2), Fraction(1, 3), Fraction(-2, 7), Fraction(5)])
def test_fraction_base_is_exact(q):
    p = ak.ExprPool()
    x = p.symbol("x")
    got = q**x
    assert got == p.rational(q.numerator, q.denominator) ** x
    assert got.node()[0] == "pow"
    assert got.node()[1].node()[0] in ("rational", "integer")


def test_float_base_stays_float():
    p = ak.ExprPool()
    x = p.symbol("x")
    # 0.5 is what Fraction(1, 2) turns into on 3.9-3.11; written as a float it
    # is a float and must stay one.
    assert (0.5**x).node()[1].node()[0] == "float"
    assert x.__rpow__(0.5).node()[1].node()[0] == "float"


def test_documented_exact_spelling():
    p = ak.ExprPool()
    x = p.symbol("x")
    e = p.rational(1, 2) ** x
    assert e.node()[1].node()[0] == "rational"
    assert ak.eval_expr(e, {x: 3}) == 0.125


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
