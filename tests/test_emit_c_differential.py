"""Randomised differential test: compiled ``emit_c_expr`` / ``emit_c_vec`` vs ``eval_expr``.

Report10-21 N1: the operand-pair sharing added to the C emitter (#490) counted
every slot pair of an n-ary sum or product, so when an operand list held the
same operand twice the pair was "shared" with itself through one slot and an
operand was silently dropped — ``m*v*v/2`` compiled to ``((m*v)*0.5)``,
``x + y + y`` to ``x + y``, ``x*x*y`` to ``(x*y)*(x*y)``.  The existing fuzzers
built sums from *distinct* operands and never saw it.

Here every expression is assembled from a small pool of sub-expressions, so
operands repeat inside one sum or product, repeat nested at several levels, and
repeat across the outputs of one ``emit_c_vec`` call.  The pool mixes powers,
``Piecewise``, step functions and special functions that the emitter lowers to
its own C helpers.  Each expression is compiled by the system C compiler with
``-std=c99 -O2`` and compared with ``eval_expr`` at random points.
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


@pytest.fixture
def cc():
    if CC is None:
        # CI images ship a compiler: a missing one there must not turn this
        # oracle into a silent skip.
        if os.environ.get("CI"):
            pytest.fail("no working C compiler in CI; the emit_c differential test needs one")
        pytest.skip("no working C compiler")
    return CC


def _compile(source, tmp_path, name):
    src = tmp_path / f"{name}.c"
    so = tmp_path / f"{name}.so"
    src.write_text("#include <math.h>\n" + source)
    r = subprocess.run(
        [CC, "-std=c99", "-O2", "-shared", "-fPIC", "-o", str(so), str(src), "-lm"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"emitted C does not compile:\n{r.stderr[:3000]}"
    return ctypes.CDLL(str(so))


def _scalar(lib, name, nargs):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_double
    fn.argtypes = [ctypes.c_double] * nargs
    return fn


def _vector(lib, name, nargs, nout):
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = [ctypes.c_double] * nargs + [ctypes.POINTER(ctypes.c_double)]

    def call(*args):
        out = (ctypes.c_double * nout)()
        fn(*args, out)
        return list(out)

    return call


# ---------------------------------------------------------------------------
# Random expressions with repeated operands
# ---------------------------------------------------------------------------

P = ak.ExprPool()
VARS = [P.symbol(n) for n in ("x", "y", "z")]
VNAMES = ["x", "y", "z"]
X, Y, Z = VARS


def _atoms():
    half = P.rational(1, 2)
    return [X, Y, Z, X, Y, P.integer(2), P.integer(-3), half, P.rational(3, 4)]


def _special(rng, a):
    """A function the emitter lowers to libm or to one of its own helpers,
    applied to `a` mapped into a domain where it is finite and smooth."""
    pos = ak.abs(a) + P.integer(1)  # >= 1
    unit = ak.tanh(a) * P.rational(1, 2)  # in (-1/2, 1/2)
    choices = [
        lambda: ak.sin(a),
        lambda: ak.cos(a),
        lambda: ak.exp(ak.tanh(a)),
        lambda: ak.log(pos),
        lambda: ak.sqrt(pos),
        lambda: ak.atan(a),
        lambda: ak.erf(a),
        lambda: ak.lambert_w(pos),
        lambda: P.func("digamma", [pos]),
        lambda: P.func("trigamma", [pos]),
        lambda: P.func("dilog", [unit]),
        lambda: P.func("Si", [a]),
        lambda: P.func("Ei", [pos]),
        lambda: P.func("bessel_j", [P.integer(rng.randint(0, 2)), a]),
        lambda: P.func("fresnels", [unit]),
        lambda: ak.piecewise([(P.lt(a, P.integer(0)), a * a), (P.le(a, Y), a + Y)], a - Z),
        lambda: P.func("heaviside", [a]),
        lambda: ak.sign(a) * a,
        lambda: ak.min(a, Y) + ak.max(a, Z),
    ]
    return rng.choice(choices)()


def _nary(rng, pool, op):
    """A sum or product of 2-6 operands drawn *with replacement* from `pool`,
    so the same operand repeats inside one operand list."""
    k = rng.randint(2, 6)
    ops = [rng.choice(pool) for _ in range(k)]
    # Force at least one repeat most of the time.
    if rng.random() < 0.7:
        ops.append(rng.choice(ops))
    rng.shuffle(ops)
    acc = ops[0]
    for t in ops[1:]:
        if op == "*":
            acc = acc * t
        elif rng.random() < 0.25:
            acc = acc - t
        else:
            acc = acc + t
    return acc


def _gen(rng, depth=3):
    pool = list(_atoms())
    for _ in range(depth):
        new = []
        for _ in range(rng.randint(2, 4)):
            r = rng.random()
            if r < 0.35:
                new.append(_nary(rng, pool, "+"))
            elif r < 0.65:
                new.append(_nary(rng, pool, "*"))
            elif r < 0.75:
                new.append(rng.choice(pool) ** rng.randint(2, 4))
            elif r < 0.8:
                new.append(rng.choice(pool) / (rng.choice(pool) ** 2 + P.integer(1)))
            else:
                new.append(_special(rng, rng.choice(pool)))
        # Keep the old entries: later levels reuse both, which nests repeats.
        pool = pool + new
    return _nary(rng, pool[-6:], rng.choice("+*"))


def _close(got, want, tol=1e-12):
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    return abs(got - want) <= tol * max(1.0, abs(want))


_POINTS = [(0.7, -1.3, 0.4), (-0.45, 0.9, 1.6), (1.25, 0.35, -0.8), (-1.1, -0.6, 0.15)]


def _eval(e, pt):
    try:
        return ak.eval_expr(e, dict(zip(VARS, pt)))
    except Exception:
        return None


# Hand-picked repeats from the report, always included.
def _report_cases():
    return [
        X * Y * Y / 2,
        X + Y + Y,
        X * X * Y,
        X + X + 1,
        X * Y * Y,
        X + Y + Y + Z + Z,
        (X + Y) * (X + Y) * Z,
        X - Y - Y + X,
        X * X * Y * Y * Z,
    ]


def _cases(seed, n):
    rng = random.Random(seed)
    exprs = list(_report_cases())
    while len(exprs) < n:
        e = _gen(rng, depth=rng.randint(1, 3))
        exprs.append(e)
    return exprs


N_EXPRS = 600


def test_emit_c_expr_matches_eval_expr_on_repeated_operands(cc, tmp_path):
    exprs = _cases(20261022, N_EXPRS)
    sources, kept = [], []
    for e in exprs:
        try:
            sources.append(ak.emit_c_expr(e, VARS, VNAMES, f"f{len(kept)}"))
        except ak.AlkahestError:
            continue
        kept.append(e)
    assert len(kept) >= 500, f"only {len(kept)} expressions emitted"
    # Helpers are `#ifndef`-guarded, so many functions share one translation unit.
    bad, checked = [], 0
    chunk = 100
    for c0 in range(0, len(kept), chunk):
        lib = _compile("\n".join(sources[c0 : c0 + chunk]), tmp_path, f"s{c0}")
        for i in range(c0, min(c0 + chunk, len(kept))):
            f = _scalar(lib, f"f{i}", 3)
            for pt in _POINTS:
                want = _eval(kept[i], pt)
                if want is None:
                    continue
                checked += 1
                got = f(*pt)
                if not _close(got, want):
                    bad.append((str(kept[i])[:120], pt, got, want))
    assert checked >= 1500, checked
    assert not bad, f"{len(bad)} mismatches, e.g. {bad[:5]}"


def test_emit_c_vec_matches_eval_expr_on_repeated_operands(cc, tmp_path):
    exprs = _cases(20261023, N_EXPRS)
    rng = random.Random(7)
    # Outputs share sub-expressions (and repeated pairs) across one call.
    groups, i = [], 0
    while i < len(exprs):
        k = rng.randint(2, 6)
        groups.append(exprs[i : i + k])
        i += k
    sources, kept = [], []
    for g in groups:
        try:
            sources.append(ak.emit_c_vec(g, VARS, VNAMES, f"g{len(kept)}"))
        except ak.AlkahestError:
            continue
        kept.append(g)
    assert sum(len(g) for g in kept) >= 500
    bad, checked = [], 0
    chunk = 30
    for c0 in range(0, len(kept), chunk):
        lib = _compile("\n".join(sources[c0 : c0 + chunk]), tmp_path, f"v{c0}")
        for j in range(c0, min(c0 + chunk, len(kept))):
            g = kept[j]
            f = _vector(lib, f"g{j}", 3, len(g))
            for pt in _POINTS:
                got = f(*pt)
                for e, gv in zip(g, got):
                    want = _eval(e, pt)
                    if want is None:
                        continue
                    checked += 1
                    if not _close(gv, want):
                        bad.append((str(e)[:120], pt, gv, want))
    assert checked >= 1500, checked
    assert not bad, f"{len(bad)} mismatches, e.g. {bad[:5]}"


def test_report_repros(cc, tmp_path):
    """The minimal repros from report10-21 N1, exactly as reported."""
    m, v = P.symbol("m"), P.symbol("v")
    lib = _compile(
        ak.emit_c_expr(m * v * v / 2, [m, v], ["m", "v"], "ke")
        + ak.emit_c_expr(X + Y + Y, [X, Y], ["x", "y"], "s")
        + ak.emit_c_expr(X * X * Y, [X, Y], ["x", "y"], "p")
        + ak.emit_c_expr(X + X + 1, [X, Y], ["x", "y"], "t")
        + ak.emit_c_vec([X + Y + Y, X * Y * Y], [X, Y], ["x", "y"], "vv"),
        tmp_path,
        "repro",
    )
    assert _scalar(lib, "ke", 2)(2.0, 3.0) == 9.0
    assert _scalar(lib, "s", 2)(1.0, 2.0) == 5.0
    assert _scalar(lib, "p", 2)(2.0, 3.0) == 12.0
    assert _scalar(lib, "t", 2)(1.0, 0.0) == 3.0
    assert _vector(lib, "vv", 2, 2)(1.0, 2.0) == [5.0, 4.0]
