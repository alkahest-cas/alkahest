"""JIT / numeric suite: compiled evaluation throughput and compile time."""

from __future__ import annotations

import random
import re

from _harness import WL, Case, Suite, shared_array, shared_array_path, wl_str

SCRIPT = "benchmarks/website/numeric.py"

ARGS = ["x", "y", "z"]


def _model(nterms: int, seed: int) -> str:
    """A sum of damped oscillatory terms in three variables (seeded)."""
    rng = random.Random(seed)
    terms = []
    for _ in range(nterms):
        a, b, c = (round(rng.uniform(-2, 2), 3) for _ in range(3))
        w = round(rng.uniform(0.1, 1.5), 3)
        kind = rng.randrange(3)
        if kind == 0:
            terms.append(f"({w})*sin(({a})*x + ({b})*y)*exp(-({abs(c)})*z**2)")
        elif kind == 1:
            terms.append(f"({w})*cos(({a})*x*y + ({c}))/(1 + ({abs(b)})*z**2)")
        else:
            terms.append(f"({w})*sqrt(1 + ({a})**2*x**2 + y**2)*log(2 + z**2)")
    return " + ".join(terms)


def _wl(s: str) -> str:
    t = wl_str(s)
    for py, w in (("exp", "Exp"), ("log", "Log"), ("sqrt", "Sqrt"), ("sin", "Sin"), ("cos", "Cos")):
        t = re.sub(rf"\b{py}\(", w + "(", t)
    out, stack = [], []
    for ch in t:
        if ch == "(":
            call = bool(out) and out[-1].isalpha()
            stack.append(call)
            out.append("[" if call else "(")
        elif ch == ")":
            out.append("]" if stack.pop() else ")")
        else:
            out.append(ch)
    return "".join(out)


MODEL = _model(24, 7)
SAMPLES = [0, 31337, 5_000_000, 9_999_999]


def _arrays(n):
    return [shared_array(f"num_{a}", n, 400 + i, -2.0, 2.0) for i, a in enumerate(ARGS)]


def _fp(out, n):
    import math

    import numpy as np

    a = np.asarray(out, dtype=float).ravel()
    return [math.fsum(a) / n, *[float(a[i % n]) for i in SAMPLES]]


def _ref(n):
    def ref():
        import mpmath as mp

        mp.mp.dps = 30

        def go():
            import math

            import numpy as np

            arrs = _arrays(n)
            env = {k: getattr(np, k) for k in ("sin", "cos", "exp", "sqrt", "log")}
            f = eval("lambda x, y, z: " + MODEL, env)
            menv = {k: getattr(mp, k) for k in ("sin", "cos", "exp", "sqrt", "log")}
            g = eval("lambda x, y, z: " + MODEL, menv)
            vals = [math.fsum(f(*arrs)) / n]
            for i in SAMPLES:
                vals.append(float(g(*[mp.mpf(float(a[i % n])) for a in arrs])))
            return vals

        return go, lambda r: r

    return ref


def _alk(n, parallel=False):
    def impl():
        import alkahest as ak

        p = ak.ExprPool()
        syms = {a: p.symbol(a) for a in ARGS}
        e = ak.parse(MODEL, p, syms)
        fn = ak.compile_expr(e, [syms[a] for a in ARGS], expected_evals=n)
        arrs = _arrays(n)
        ev = ak.numpy_eval_par if parallel else ak.numpy_eval

        def go():
            return ev(fn, *arrs)

        return go, lambda r: _fp(r, n)

    return impl


def _numpy(n):
    def impl():
        import numpy as np

        arrs = _arrays(n)
        env = {k: getattr(np, k) for k in ("sin", "cos", "exp", "sqrt", "log")}
        f = eval("lambda x, y, z: " + MODEL, env)

        def go():
            return f(*arrs)

        return go, lambda r: _fp(r, n)

    return impl


def _sympy(n):
    def impl():
        import sympy as sp

        syms = sp.symbols(ARGS)
        e = sp.sympify(MODEL, locals=dict(zip(ARGS, syms)))
        f = sp.lambdify(syms, e, modules="numpy", cse=True)
        arrs = _arrays(n)

        def go():
            return f(*arrs)

        return go, lambda r: _fp(r, n)

    return impl


def _symengine(n):
    def impl():
        import numpy as np
        import symengine as se

        syms = se.symbols(ARGS)
        e = se.sympify(MODEL)
        f = se.Lambdify(list(syms), [e], backend="llvm", cse=True)
        X = np.ascontiguousarray(np.stack(_arrays(n), axis=1))

        def go():
            return f(X)

        return go, lambda r: _fp(r, n)

    return impl


def _wl_eval(n) -> WL:
    reads = "; ".join(
        f'{a}A = BinaryReadList["{shared_array_path(f"num_{a}", n, 400 + i, -2.0, 2.0)}", "Real64"]'
        for i, a in enumerate(ARGS)
    )
    idx = ", ".join(str(i % n + 1) for i in SAMPLES)
    return WL(
        setup=f"{reads}; e = {_wl(MODEL)}; rules = {{x -> xA, y -> yA, z -> zA}}",
        run="e /. rules",
        fp=f"Join[{{Mean[res]}}, res[[{{{idx}}}]]]",
    )


N1 = 10_000_000

EVAL = Case(
    id="jit_eval_1e7",
    task="Evaluate a 24-term expression at 10M points",
    detail="A sum of 24 damped sin/cos/sqrt·log terms in x, y, z (~250 nodes) at 10,000,000 "
    "float64 points, single thread. Alkahest: compile_expr + numpy_eval; NumPy: the same "
    "expression as NumPy ufunc calls; SymPy: lambdify(numpy, cse=True); SymEngine: "
    "Lambdify(LLVM, cse=True); Mathematica: packed-array evaluation. Compilation not timed. "
    "Checked: mean and sampled values against mpmath.",
    systems={
        "alkahest": _alk(N1),
        "numpy": _numpy(N1),
        "sympy": _sympy(N1),
        "symengine": _symengine(N1),
        "mathematica": _wl_eval(N1),
    },
    reference=_ref(N1),
    tol=1e-8,
    repeats=5,
)

EVAL_PAR = Case(
    id="jit_eval_1e7_parallel",
    task="Same 10M-point evaluation, multithreaded Alkahest",
    detail="As above, but Alkahest uses numpy_eval_par (Rayon, 8 threads under the 8-CPU cap). "
    "NumPy and SymEngine have no built-in multithreaded path for this and run one thread; "
    "Mathematica's packed-array arithmetic uses its own threading where it applies.",
    systems={
        "alkahest": _alk(N1, parallel=True),
        "numpy": _numpy(N1),
        "symengine": _symengine(N1),
        "mathematica": _wl_eval(N1),
    },
    reference=_ref(N1),
    tol=1e-8,
    note="Not a like-for-like thread count: this row shows what the parallel path buys.",
)

# ---------------------------------------------------------------------------
# Compile time for a large expression
# ---------------------------------------------------------------------------

BIG = _model(2000, 11)
BIG_PT = (0.3, -0.7, 1.1)


def _big_ref():
    import mpmath as mp

    mp.mp.dps = 30

    def go():
        env = {k: getattr(mp, k) for k in ("sin", "cos", "exp", "sqrt", "log")}
        f = eval("lambda x, y, z: " + BIG, env)
        return [float(f(*[mp.mpf(v) for v in BIG_PT]))]

    return go, lambda r: r


def _big_alk():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {a: p.symbol(a) for a in ARGS}
    e = ak.parse(BIG, p, syms)
    xs = [syms[a] for a in ARGS]

    def go():
        return ak.compile_expr(e, xs, expected_evals=10**6)

    def fp(fn):
        import numpy as np

        return [float(ak.numpy_eval(fn, *[np.array([v]) for v in BIG_PT])[0])]

    return go, fp


def _big_sympy():
    import sympy as sp

    syms = sp.symbols(ARGS)
    e = sp.sympify(BIG, locals=dict(zip(ARGS, syms)))

    def go():
        return sp.lambdify(syms, e, modules="numpy", cse=True)

    return go, lambda f: [float(f(*BIG_PT))]


def _big_symengine():
    import symengine as se

    syms = se.symbols(ARGS)
    e = se.sympify(BIG)

    def go():
        return se.Lambdify(list(syms), [e], backend="llvm", cse=True)

    return go, lambda f: [float(f(list(BIG_PT))[0])]


COMPILE = Case(
    id="jit_compile_large",
    task="Compile a 2000-term expression",
    detail="Time to turn a ~20,000-node expression (2000 damped oscillatory terms in x, y, z) "
    "into a native vectorised function, from an already-built expression. Alkahest: "
    "compile_expr (Cranelift); SymPy: lambdify(numpy, cse=True); SymEngine: Lambdify(LLVM, "
    "cse=True). Checked: the compiled function's value at one point against mpmath.",
    systems={"alkahest": _big_alk, "sympy": _big_sympy, "symengine": _big_symengine},
    reference=_big_ref,
    tol=1e-9,
)

SUITES = [
    Suite(
        id="numeric",
        label="JIT / numeric",
        blurb="Symbolic expressions compiled to native code and evaluated over millions of "
        "points, against NumPy, SymPy's lambdify, SymEngine's LLVM backend and Mathematica.",
        script=SCRIPT,
        cases=[EVAL, EVAL_PAR, COMPILE],
    )
]
