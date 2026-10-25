"""Pharma / PK-PD suite: compartment-model closed forms, exposure metrics,
parameter sensitivities and population-scale evaluation."""

from __future__ import annotations

from _harness import WL, Case, Suite, shared_array, shared_array_path, wl_str

SCRIPT = "benchmarks/website/pharma.py"

# ---------------------------------------------------------------------------
# Model definitions (Python syntax; every system parses these strings)
# ---------------------------------------------------------------------------

# Two-compartment model with first-order absorption: depot -> central <-> peripheral.
RATES = ["ka", "k10", "k12", "k21"]
A_ROWS = [["-ka", "0", "0"], ["ka", "-(k10 + k12)", "k21"], ["0", "k12", "-k21"]]
PT = {"ka": 1.3, "k10": 0.25, "k12": 0.4, "k21": 0.15, "t": 2.7}

_S = "(k10 + k12 + k21)"
_ROOT = f"sqrt({_S}**2 - 4*k10*k21)"
ALPHA = f"(({_S} + {_ROOT})/2)"
BETA = f"(({_S} - {_ROOT})/2)"
# Central-compartment concentration after an oral dose D with bioavailability F.
C2_ORAL = (
    f"F*D*ka/V*((k21 - {ALPHA})/((ka - {ALPHA})*({BETA} - {ALPHA}))*exp(-{ALPHA}*t)"
    f" + (k21 - {BETA})/((ka - {BETA})*({ALPHA} - {BETA}))*exp(-{BETA}*t)"
    f" + (k21 - ka)/(({ALPHA} - ka)*({BETA} - ka))*exp(-ka*t))"
)
C2_PARAMS = ["ka", "k10", "k12", "k21", "V"]
C2_PT = {"ka": 1.3, "k10": 0.25, "k12": 0.4, "k21": 0.15, "V": 22.0, "F": 0.9, "D": 100.0, "t": 2.7}

# One-compartment oral model in clearance parameterisation.
C1_ORAL = "F*D*ka/(V*(ka - CL/V))*(exp(-CL/V*t) - exp(-ka*t))"
C1_PT = {"F": 0.9, "D": 100.0, "ka": 1.3, "CL": 3.1, "V": 22.0}


def _wl(s: str) -> str:
    """Python expression string -> WL (exp/sqrt only)."""
    import re

    t = wl_str(s)
    t = re.sub(r"\bD\b", "dose", t)  # D is WL's derivative operator
    t = re.sub(r"\bexp\(", "Exp(", t)
    t = re.sub(r"\bsqrt\(", "Sqrt(", t)
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


def _wl_rules(d: dict) -> str:
    return "{" + ", ".join(f"{'dose' if k == 'D' else k} -> {v}" for k, v in d.items()) + "}"


# ---------------------------------------------------------------------------
# 1. Symbolic matrix exponential of the 2-compartment oral model
# ---------------------------------------------------------------------------


def _mexp_ref():
    import numpy as np
    import scipy.linalg as sl

    def go():
        env = {k: v for k, v in PT.items() if k != "t"}
        An = np.array([[eval(c, {}, env) for c in row] for row in A_ROWS], dtype=float)
        return list(sl.expm(An * PT["t"]).ravel())

    return go, lambda r: r


def _mexp_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n, ak.Domain.Positive) for n in [*RATES, "t"]}
    A = ak.Matrix.from_rows([[ak.parse(c, p, syms) for c in row] for row in A_ROWS])
    t = syms["t"]

    def go():
        return (A * t).matrix_exp()

    def fp(E):
        env = {syms[k]: v for k, v in PT.items()}
        return [ak.eval_expr(E.get(i, j), env) for i in range(3) for j in range(3)]

    return go, fp


def _mexp_sympy():
    import sympy as sp

    syms = {n: sp.Symbol(n, positive=True) for n in [*RATES, "t"]}
    A = sp.Matrix([[sp.sympify(c, locals=syms) for c in row] for row in A_ROWS])
    t = syms["t"]

    def go():
        return (A * t).exp()

    def fp(E):
        env = {syms[k]: v for k, v in PT.items()}
        return [complex(sp.N(E[i, j].subs(env))).real for i in range(3) for j in range(3)]

    return go, fp


_WL_A = "{" + ", ".join("{" + ", ".join(_wl(c) for c in row) + "}" for row in A_ROWS) + "}"

MEXP = Case(
    id="pk_matrix_exp",
    task="Closed-form exp(A·t), 2-compartment oral model",
    detail="Symbolic 3×3 matrix exponential of the depot/central/peripheral rate matrix with "
    "symbolic ka, k10, k12, k21 and t (the full analytic solution of the linear PK model). "
    "Checked: all nine entries at a parameter point against scipy.linalg.expm.",
    systems={
        "alkahest": _mexp_alkahest,
        "sympy": _mexp_sympy,
        "mathematica": WL(
            setup=f"A = {_WL_A}",
            run="MatrixExp[A t]",
            fp=f"Re[N[Flatten[res /. {_wl_rules(PT)}]]]",
        ),
    },
    reference=_mexp_ref,
    tol=1e-9,
)

# ---------------------------------------------------------------------------
# 2. AUC and AUMC (exposure metrics) of the 1-compartment oral model
# ---------------------------------------------------------------------------

C1_SYMS = ["F", "D", "ka", "CL", "V", "t"]


def _auc_ref():
    import mpmath as mp

    mp.mp.dps = 30

    def go():
        e = C1_PT
        ke = e["CL"] / e["V"]

        def c(t):
            return e["F"] * e["D"] * e["ka"] / (e["V"] * (e["ka"] - ke)) * (mp.exp(-ke * t) - mp.exp(-e["ka"] * t))

        return [float(mp.quad(c, [0, mp.inf])), float(mp.quad(lambda t: t * c(t), [0, mp.inf]))]

    return go, lambda r: r


def _auc_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n, ak.Domain.Positive) for n in C1_SYMS}
    C = ak.parse(C1_ORAL, p, syms)
    t = syms["t"]
    zero, oo = p.integer(0), p.pos_infinity()

    def go():
        return ak.integrate(C, t, zero, oo).value, ak.integrate(t * C, t, zero, oo).value

    def fp(r):
        env = {syms[k]: v for k, v in C1_PT.items()}
        return [ak.eval_expr(r[0], env), ak.eval_expr(r[1], env)]

    return go, fp


def _auc_sympy():
    import sympy as sp

    syms = {n: sp.Symbol(n, positive=True) for n in C1_SYMS}
    C = sp.sympify(C1_ORAL, locals=syms)
    t = syms["t"]

    def go():
        return sp.integrate(C, (t, 0, sp.oo)), sp.integrate(t * C, (t, 0, sp.oo))

    def fp(r):
        env = {syms[k]: v for k, v in C1_PT.items()}
        out = []
        for e in r:
            if e.has(sp.Integral):
                out.append("FAILED")
            else:
                out.append(float(sp.N(e.subs(env))))
        return out

    return go, fp


AUC = Case(
    id="pk_auc_aumc",
    task="AUC₀^∞ and AUMC₀^∞, 1-compartment oral",
    detail="Exact improper integrals ∫₀^∞ C(t) dt and ∫₀^∞ t·C(t) dt of the Bateman function "
    "with symbolic positive F, D, ka, CL, V. Checked against mpmath quadrature at a parameter "
    "point.",
    systems={
        "alkahest": _auc_alkahest,
        "sympy": _auc_sympy,
        "mathematica": WL(
            setup=f"c = {_wl(C1_ORAL)}; as = {{F > 0, dose > 0, ka > 0, CL > 0, V > 0, ka != CL/V}}",
            run="{Integrate[c, {t, 0, Infinity}, Assumptions -> as], "
            "Integrate[t c, {t, 0, Infinity}, Assumptions -> as]}",
            fp=f"N[res /. {_wl_rules(C1_PT)}]",
        ),
    },
    reference=_auc_ref,
    tol=1e-9,
    note="A small problem: every system answers in well under a second; the gap is per-call "
    "overhead of the integrator.",
)

# ---------------------------------------------------------------------------
# 3. Parameter sensitivities: gradient + Hessian of the 2-compartment model
# ---------------------------------------------------------------------------

PAIRS = [(i, j) for i in range(len(C2_PARAMS)) for j in range(i, len(C2_PARAMS))]
C2_ALL = ["F", "D", "ka", "k10", "k12", "k21", "V", "t"]


def _sens_ref():
    import mpmath as mp

    mp.mp.dps = 40
    env = {"exp": mp.exp, "sqrt": mp.sqrt}

    def go():
        f = eval("lambda F, D, ka, k10, k12, k21, V, t: " + C2_ORAL, env)
        x0 = [mp.mpf(C2_PT[n]) for n in C2_ALL]
        idx = [C2_ALL.index(n) for n in C2_PARAMS]
        vals = []
        for i in idx:
            order = [0] * len(C2_ALL)
            order[i] = 1
            vals.append(float(mp.diff(f, x0, tuple(order))))
        for i, j in PAIRS:
            order = [0] * len(C2_ALL)
            order[idx[i]] += 1
            order[idx[j]] += 1
            vals.append(float(mp.diff(f, x0, tuple(order))))
        return vals

    return go, lambda r: r


def _sens_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n) for n in C2_ALL}
    C = ak.parse(C2_ORAL, p, syms)
    ps = [syms[n] for n in C2_PARAMS]

    def go():
        g = [ak.diff(C, v).value for v in ps]
        h = [ak.diff(g[i], ps[j]).value for i, j in PAIRS]
        return g + h

    def fp(r):
        env = {syms[k]: v for k, v in C2_PT.items()}
        return [ak.eval_expr(e, env) for e in r]

    return go, fp


def _sens_sympy():
    import sympy as sp

    syms = {n: sp.Symbol(n) for n in C2_ALL}
    C = sp.sympify(C2_ORAL, locals=syms)
    ps = [syms[n] for n in C2_PARAMS]

    def go():
        g = [sp.diff(C, v) for v in ps]
        h = [sp.diff(g[i], ps[j]) for i, j in PAIRS]
        return g + h

    def fp(r):
        env = {syms[k]: v for k, v in C2_PT.items()}
        return [float(e.evalf(subs=env)) for e in r]

    return go, fp


def _sens_symengine():
    import symengine as se

    syms = {n: se.Symbol(n) for n in C2_ALL}
    C = se.sympify(C2_ORAL)
    ps = [syms[n] for n in C2_PARAMS]

    def go():
        g = [se.diff(C, v) for v in ps]
        h = [se.diff(g[i], ps[j]) for i, j in PAIRS]
        return g + h

    def fp(r):
        env = {syms[k]: v for k, v in C2_PT.items()}
        return [float(e.subs(env)) for e in r]

    return go, fp


_WL_PS = "{" + ", ".join(C2_PARAMS) + "}"

SENS = Case(
    id="pk_sensitivities",
    task="Gradient + Hessian of a 2-compartment model",
    detail="All 5 first and 15 second partial derivatives of the closed-form 2-compartment oral "
    "concentration (eigenvalues written with the square root, ~120 nodes) with respect to "
    "ka, k10, k12, k21, V. Checked: all 20 values against 40-digit mpmath numerical "
    "differentiation.",
    systems={
        "alkahest": _sens_alkahest,
        "sympy": _sens_sympy,
        "symengine": _sens_symengine,
        "mathematica": WL(
            setup=f"c = {_wl(C2_ORAL)}; ps = {_WL_PS}; "
            f"pairs = {{{', '.join('{' + str(i + 1) + ', ' + str(j + 1) + '}' for i, j in PAIRS)}}}",
            run="Module[{g = D[c, #] & /@ ps}, Join[g, D[g[[#[[1]]]], ps[[#[[2]]]]] & /@ pairs]]",
            fp=f"N[res /. {_wl_rules(C2_PT)}]",
        ),
    },
    reference=_sens_ref,
    tol=1e-8,
    note="Raw (unsimplified) derivatives in every system; timing is differentiation only.",
)

# ---------------------------------------------------------------------------
# 4. Population evaluation: 1e6 virtual subjects
# ---------------------------------------------------------------------------

N_SUBJ = 1_000_000
POP_RANGES = {
    "ka": (0.5, 3.0, 101),
    "k10": (0.05, 0.5, 102),
    "k12": (0.1, 1.0, 103),
    "k21": (0.05, 0.5, 104),
    "V": (10.0, 40.0, 105),
    "t": (0.5, 24.0, 106),
}
POP_ARGS = list(POP_RANGES)
POP_FIXED = {"F": 0.9, "D": 100.0}
POP_SAMPLES = [0, 1234, 500_000, 999_999]
POP_EXPR = C2_ORAL.replace("F*D*", f"{POP_FIXED['F']}*{POP_FIXED['D']}*")


def _pop_arrays():
    return [shared_array(f"pop_{n}", N_SUBJ, s, lo, hi) for n, (lo, hi, s) in POP_RANGES.items()]


def _pop_fp(out):
    import math

    import numpy as np

    out = np.asarray(out, dtype=float).ravel()
    return [math.fsum(out) / len(out), *[float(out[i]) for i in POP_SAMPLES]]


def _pop_ref():
    import mpmath as mp

    mp.mp.dps = 30

    def go():
        import math

        import numpy as np

        arrs = _pop_arrays()
        f = eval("lambda ka, k10, k12, k21, V, t: " + POP_EXPR, {"exp": np.exp, "sqrt": np.sqrt})
        mean = math.fsum(f(*arrs)) / N_SUBJ
        g = eval("lambda ka, k10, k12, k21, V, t: " + POP_EXPR, {"exp": mp.exp, "sqrt": mp.sqrt})
        samples = [float(g(*[mp.mpf(float(a[i])) for a in arrs])) for i in POP_SAMPLES]
        return [mean, *samples]

    return go, lambda r: r


def _pop_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n) for n in POP_ARGS}
    e = ak.parse(POP_EXPR, p, syms)
    fn = ak.compile_expr(e, [syms[n] for n in POP_ARGS], expected_evals=N_SUBJ)
    arrs = _pop_arrays()

    def go():
        return ak.numpy_eval(fn, *arrs)

    return go, _pop_fp


def _pop_numpy():
    import numpy as np

    arrs = _pop_arrays()
    F, D = POP_FIXED["F"], POP_FIXED["D"]

    def model(ka, k10, k12, k21, V, t):
        s = k10 + k12 + k21
        r = np.sqrt(s * s - 4 * k10 * k21)
        a = (s + r) / 2
        b = (s - r) / 2
        return (
            F
            * D
            * ka
            / V
            * (
                (k21 - a) / ((ka - a) * (b - a)) * np.exp(-a * t)
                + (k21 - b) / ((ka - b) * (a - b)) * np.exp(-b * t)
                + (k21 - ka) / ((a - ka) * (b - ka)) * np.exp(-ka * t)
            )
        )

    def go():
        return model(*arrs)

    return go, _pop_fp


def _pop_sympy():
    import sympy as sp

    syms = [sp.Symbol(n) for n in POP_ARGS]
    e = sp.sympify(POP_EXPR, locals=dict(zip(POP_ARGS, syms)))
    f = sp.lambdify(syms, e, modules="numpy", cse=True)
    arrs = _pop_arrays()

    def go():
        return f(*arrs)

    return go, _pop_fp


def _pop_symengine():
    import numpy as np
    import symengine as se

    syms = [se.Symbol(n) for n in POP_ARGS]
    e = se.sympify(POP_EXPR)
    f = se.Lambdify(syms, [e], backend="llvm", cse=True)
    X = np.ascontiguousarray(np.stack(_pop_arrays(), axis=1))

    def go():
        return f(X)

    return go, _pop_fp


def _wl_pop() -> WL:
    reads = "; ".join(
        f'{n}A = BinaryReadList["{shared_array_path(f"pop_{n}", N_SUBJ, s, lo, hi)}", "Real64"]'
        for n, (lo, hi, s) in POP_RANGES.items()
    )
    rules = "{" + ", ".join(f"{n} -> {n}A" for n in POP_ARGS) + "}"
    return WL(
        setup=f"{reads}; c = {_wl(POP_EXPR)}; rules = {rules}",
        run="c /. rules",
        fp=f"Join[{{Mean[res]}}, res[[{{{', '.join(str(i + 1) for i in POP_SAMPLES)}}}]]]",
    )


POP = Case(
    id="pk_population_1e6",
    task="Evaluate a 2-compartment model on 1M virtual subjects",
    detail="The closed-form 2-compartment oral concentration with per-subject ka, k10, k12, k21, "
    "V and sampling time t (1,000,000 subjects, float64). Alkahest: compile_expr + numpy_eval "
    "(one thread); NumPy: hand-written vectorised code; SymPy: lambdify(numpy, cse=True); "
    "SymEngine: Lambdify(LLVM, cse=True); Mathematica: vectorised packed-array evaluation. "
    "Compilation is not timed. Checked: mean and four sampled subjects against 30-digit mpmath.",
    systems={
        "alkahest": _pop_alkahest,
        "numpy": _pop_numpy,
        "sympy": _pop_sympy,
        "symengine": _pop_symengine,
        "mathematica": _wl_pop(),
    },
    reference=_pop_ref,
    tol=1e-8,
)

SUITES = [
    Suite(
        id="pharma",
        label="Pharma / PK-PD",
        blurb="Linear compartment models solved in closed form, exposure metrics, parameter "
        "sensitivities, and the compiled model evaluated across a virtual population.",
        script=SCRIPT,
        cases=[MEXP, AUC, SENS, POP],
    )
]
