"""Numeric-correctness fuzz for the codegen / evaluation paths.

Cross-checks three independent evaluators of the same random symbolic
expression:

1. an independent Python ``math`` reference (ground truth, built alongside
   each generated expression),
2. :func:`alkahest.eval_expr` (the f64 interpreter path), and
3. the ``gcc``-compiled output of :func:`alkahest.emit_c_expr` (the C codegen
   path, which lowers to ``libm``).

Any disagreement beyond a tight tolerance localizes a bug to a specific path.
This suite was added after a fuzz run found that ``eval_expr``'s ``erf``/``erfc``
used a low-accuracy Abramowitz & Stegun approximation (abs error ~1.4e-7,
and ~6.6e-4 *relative* on ``erfc`` of moderate-to-large arguments via
catastrophic cancellation), while the emitted C used full-precision ``libm``.
:func:`test_erf_erfc_accuracy` is the targeted regression guard for that fix
and needs no C compiler.
"""

import math
import os
import random
import shutil
import subprocess
import tempfile

import alkahest as ak
import pytest

P = ak.ExprPool()
VARS = [P.symbol("x"), P.symbol("y"), P.symbol("z")]
VNAMES = ["x", "y", "z"]

UNARY = [
    (ak.sin, math.sin),
    (ak.cos, math.cos),
    (ak.tan, math.tan),
    (ak.exp, math.exp),
    (ak.log, math.log),
    (ak.sqrt, math.sqrt),
    (ak.sinh, math.sinh),
    (ak.cosh, math.cosh),
    (ak.tanh, math.tanh),
    (ak.atan, math.atan),
    (ak.asin, math.asin),
    (ak.acos, math.acos),
    (ak.erf, math.erf),
    (ak.erfc, math.erfc),
    (ak.floor, math.floor),
    (ak.ceil, math.ceil),
]


def _c_compiler():
    for cc in ("cc", "gcc-12", "gcc", "clang", "gcc-11"):
        path = shutil.which(cc)
        if path and _cc_works(path):
            return path
    return None


def _cc_works(path):
    # Probe an actual math.h include + libm link, since the emitted code needs
    # both — a compiler that builds a bare `main` may still lack C headers.
    d = tempfile.mkdtemp()
    src = os.path.join(d, "t.c")
    out = os.path.join(d, "t.out")
    with open(src, "w") as f:
        f.write("#include <math.h>\nint main(){return (int)erf(0.5);}\n")
    try:
        r = subprocess.run([path, "-O2", "-o", out, src, "-lm"], capture_output=True)
        return r.returncode == 0
    except OSError:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _gen(depth, rng):
    """Return (alkahest_expr, python_fn(vals)->float) for the same computation."""
    if depth <= 0 or rng.random() < 0.28:
        r = rng.random()
        if r < 0.5:
            i = rng.randint(0, len(VARS) - 1)
            return VARS[i], (lambda vs, i=i: vs[i])
        k = rng.randint(-5, 5)
        return P.integer(k), (lambda vs, k=k: float(k))
    kind = rng.random()
    if kind < 0.45:
        ae, af = _gen(depth - 1, rng)
        be, bf = _gen(depth - 1, rng)
        op = rng.choice("+-*/")
        if op == "+":
            return ae + be, (lambda vs, af=af, bf=bf: af(vs) + bf(vs))
        if op == "-":
            return ae - be, (lambda vs, af=af, bf=bf: af(vs) - bf(vs))
        if op == "*":
            return ae * be, (lambda vs, af=af, bf=bf: af(vs) * bf(vs))
        return ae / be, (lambda vs, af=af, bf=bf: af(vs) / bf(vs))
    if kind < 0.62:
        ae, af = _gen(depth - 1, rng)
        n = rng.randint(-3, 4)
        return ae**n, (lambda vs, af=af, n=n: af(vs) ** n)
    if kind < 0.78:
        ae, af = _gen(depth - 1, rng)
        be, bf = _gen(depth - 1, rng)
        return ak.atan2(ae, be), (lambda vs, af=af, bf=bf: math.atan2(af(vs), bf(vs)))
    build, pf = rng.choice(UNARY)
    ae, af = _gen(depth - 1, rng)
    return build(ae), (lambda vs, pf=pf, af=af: pf(af(vs)))


def _isclose(a, b, rtol=1e-9, atol=1e-11):
    if a is None or b is None:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.isnan(a) == math.isnan(b)
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def test_erf_erfc_accuracy():
    """eval_expr(erf/erfc) must match math to full double precision (no compiler needed)."""
    p = ak.ExprPool()
    x = p.symbol("x")
    worst = 0.0
    v = -4.0
    while v <= 4.0:
        for expr, ref in ((ak.erf(x), math.erf(v)), (ak.erfc(x), math.erfc(v))):
            got = ak.eval_expr(expr, {x: v})
            rel = abs(got - ref) / max(abs(ref), 1e-300)
            worst = max(worst, rel)
        v += 0.01
    assert worst < 1e-12, f"erf/erfc eval accuracy regressed: worst relative error {worst:.2e}"


@pytest.mark.skipif(_c_compiler() is None, reason="no working C compiler for emit_c_expr fuzz")
def test_emit_c_matches_reference_and_eval():
    """Fuzz: gcc-compiled emit_c_expr and eval_expr must both match a Python reference."""
    cc = _c_compiler()
    rng = random.Random(20260723)
    n = 150
    points = 6

    exprs = []
    tries = 0
    while len(exprs) < n and tries < n * 6:
        tries += 1
        try:
            e, pfn = _gen(rng.randint(2, 5), rng)
            csrc = ak.emit_c_expr(e, VARS, VNAMES, f"f{len(exprs)}")
        except Exception:
            continue
        exprs.append((e, pfn, csrc))

    lines = ["#include <math.h>", "#include <stdio.h>"]
    lines += [c for (_, _, c) in exprs]
    lines.append("int main(){char L[512];int idx;double x,y,z;")
    lines.append(
        'while(fgets(L,sizeof L,stdin)){if(sscanf(L,"%d %lf %lf %lf",'
        "&idx,&x,&y,&z)!=4)continue;double r=0.0/0.0;switch(idx){"
    )
    lines += [f"case {i}: r=f{i}(x,y,z); break;" for i in range(len(exprs))]
    lines.append('}printf("%.17g\\n",r);}return 0;}')
    csrc = "\n".join(lines)

    d = tempfile.mkdtemp()
    try:
        cpath = os.path.join(d, "fuzz.c")
        bpath = os.path.join(d, "fuzz")
        with open(cpath, "w") as f:
            f.write(csrc)
        r = subprocess.run([cc, "-O2", "-o", bpath, cpath, "-lm"], capture_output=True, text=True)
        assert r.returncode == 0, f"C compile failed:\n{r.stderr[:2000]}"

        jobs = []
        stdin_lines = []
        for idx, (e, pfn, _c) in enumerate(exprs):
            got = 0
            attempts = 0
            while got < points and attempts < points * 8:
                attempts += 1
                xv, yv, zv = (round(rng.uniform(-4, 4), 4) for _ in range(3))
                vs = (xv, yv, zv)
                try:
                    ref = pfn(vs)
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
                if not isinstance(ref, float) or not math.isfinite(ref) or abs(ref) > 1e12:
                    continue
                ev = ak.eval_expr(e, {VARS[0]: xv, VARS[1]: yv, VARS[2]: zv})
                jobs.append((idx, xv, yv, zv, ref, ev))
                stdin_lines.append(f"{idx} {xv!r} {yv!r} {zv!r}")
                got += 1

        out = subprocess.run(
            [bpath], input="\n".join(stdin_lines) + "\n", capture_output=True, text=True
        )
        cvals = [float(s) for s in out.stdout.split()]
        assert len(cvals) == len(jobs)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    c_bugs = []
    e_bugs = []
    for (idx, xv, yv, zv, ref, ev), cv in zip(jobs, cvals):
        if not _isclose(cv, ref):
            c_bugs.append((idx, xv, yv, zv, ref, cv))
        if not _isclose(ev, ref):
            e_bugs.append((idx, xv, yv, zv, ref, ev))

    assert not c_bugs, f"emit_c_expr disagrees with reference ({len(c_bugs)}): {c_bugs[:5]}"
    assert not e_bugs, f"eval_expr disagrees with reference ({len(e_bugs)}): {e_bugs[:5]}"
