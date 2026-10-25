"""Finance suite: Black–Scholes greeks, SABR sensitivities, and compiled
pricing over a large grid."""

from __future__ import annotations

import re

from _harness import WL, Case, Suite, shared_array, shared_array_path, wl_str

SCRIPT = "benchmarks/website/finance.py"


def _wl(s: str) -> str:
    t = wl_str(s)
    for py, w in (("exp", "Exp"), ("log", "Log"), ("sqrt", "Sqrt"), ("erf", "Erf")):
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


# ---------------------------------------------------------------------------
# Shared machinery: all first and second partials of a scalar model
# ---------------------------------------------------------------------------


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def _deriv_case(cid, task, detail, expr, params, allvars, point, tol=1e-8, note=None) -> Case:
    pairs = _pairs(len(params))
    mp_env_names = ("exp", "log", "sqrt", "erf")

    def ref():
        import mpmath as mp

        mp.mp.dps = 40
        env = {n: getattr(mp, n) for n in mp_env_names}

        def go():
            f = eval("lambda " + ", ".join(allvars) + ": " + expr, env)
            x0 = [mp.mpf(point[v]) for v in allvars]
            idx = [allvars.index(v) for v in params]
            vals = []
            for i in idx:
                o = [0] * len(allvars)
                o[i] = 1
                vals.append(float(mp.diff(f, x0, tuple(o))))
            for i, j in pairs:
                o = [0] * len(allvars)
                o[idx[i]] += 1
                o[idx[j]] += 1
                vals.append(float(mp.diff(f, x0, tuple(o))))
            return vals

        return go, lambda r: r

    def alk():
        import alkahest as ak

        p = ak.ExprPool()
        syms = {n: p.symbol(n) for n in allvars}
        e = ak.parse(expr, p, syms)
        ps = [syms[n] for n in params]

        def go():
            g = [ak.diff(e, v).value for v in ps]
            return g + [ak.diff(g[i], ps[j]).value for i, j in pairs]

        def fp(r):
            env = {syms[k]: v for k, v in point.items()}
            return [ak.eval_expr(x, env) for x in r]

        return go, fp

    def sym():
        import sympy as sp

        syms = {n: sp.Symbol(n) for n in allvars}
        e = sp.sympify(expr, locals=syms)
        ps = [syms[n] for n in params]

        def go():
            g = [sp.diff(e, v) for v in ps]
            return g + [sp.diff(g[i], ps[j]) for i, j in pairs]

        def fp(r):
            env = {syms[k]: v for k, v in point.items()}
            return [float(x.evalf(subs=env)) for x in r]

        return go, fp

    def seng():
        import symengine as se

        syms = {n: se.Symbol(n) for n in allvars}
        e = se.sympify(expr)
        ps = [syms[n] for n in params]

        def go():
            g = [se.diff(e, v) for v in ps]
            return g + [se.diff(g[i], ps[j]) for i, j in pairs]

        def fp(r):
            env = {syms[k]: v for k, v in point.items()}
            return [float(x.subs(env)) for x in r]

        return go, fp

    rules = "{" + ", ".join(f"{k} -> {v}" for k, v in point.items()) + "}"
    wl_pairs = "{" + ", ".join(f"{{{i + 1}, {j + 1}}}" for i, j in pairs) + "}"
    return Case(
        id=cid,
        task=task,
        detail=detail,
        systems={
            "alkahest": alk,
            "sympy": sym,
            "symengine": seng,
            "mathematica": WL(
                setup=f"e = {_wl(expr)}; ps = {{{', '.join(params)}}}; pairs = {wl_pairs}",
                run="Module[{g = D[e, #] & /@ ps}, Join[g, D[g[[#[[1]]]], ps[[#[[2]]]]] & /@ pairs]]",
                fp=f"N[res /. {rules}]",
            ),
        },
        reference=ref,
        tol=tol,
        note=note,
    )


# ---------------------------------------------------------------------------
# 1. Black–Scholes: every first and second-order greek
# ---------------------------------------------------------------------------

_D1 = "((log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T)))"
_D2 = f"({_D1} - sigma*sqrt(T))"
BS_CALL = f"S*(1 + erf({_D1}/sqrt(2)))/2 - K*exp(-r*T)*(1 + erf({_D2}/sqrt(2)))/2"
BS_VARS = ["S", "K", "T", "r", "sigma"]
BS_PT = {"S": 105.0, "K": 100.0, "T": 0.75, "r": 0.03, "sigma": 0.22}

BS = _deriv_case(
    "bs_greeks",
    "Black–Scholes: all greeks to 2nd order",
    "European call written with erf; all 4 first (delta, vega, theta, rho) and 10 second-order "
    "partials (gamma, vanna, volga, charm, ...) in S, σ, T, r, unsimplified. Checked against "
    "40-digit mpmath numerical differentiation.",
    BS_CALL,
    ["S", "sigma", "T", "r"],
    BS_VARS,
    BS_PT,
    note="A small expression: this measures per-derivative overhead more than algorithmic scaling.",
)

# ---------------------------------------------------------------------------
# 2. SABR (Hagan 2002) implied volatility: gradient and Hessian
# ---------------------------------------------------------------------------

_FK = "(F*K)"
_LFK = "log(F/K)"
_Z = f"(nu/alpha*{_FK}**((1 - beta)/2)*{_LFK})"
_X = f"log((sqrt(1 - 2*rho*{_Z} + {_Z}**2) + {_Z} - rho)/(1 - rho))"
SABR = (
    f"alpha/({_FK}**((1 - beta)/2)*(1 + (1 - beta)**2/24*{_LFK}**2 + (1 - beta)**4/1920*{_LFK}**4))"
    f"*({_Z}/{_X})"
    f"*(1 + ((1 - beta)**2/24*alpha**2/{_FK}**(1 - beta) + rho*beta*nu*alpha/(4*{_FK}**((1 - beta)/2))"
    f" + (2 - 3*rho**2)/24*nu**2)*T)"
)
SABR_VARS = ["F", "K", "T", "alpha", "beta", "rho", "nu"]
SABR_PT = {"F": 0.035, "K": 0.04, "T": 2.0, "alpha": 0.036, "beta": 0.5, "rho": -0.25, "nu": 0.42}

SABR_CASE = _deriv_case(
    "sabr_hessian",
    "SABR implied vol: gradient + Hessian",
    "Hagan et al. (2002) lognormal SABR implied-volatility formula; all 4 first and 10 second "
    "partials in α, β, ρ, ν (the model-calibration Jacobian and Hessian), unsimplified. Checked "
    "against 40-digit mpmath numerical differentiation.",
    SABR,
    ["alpha", "beta", "rho", "nu"],
    SABR_VARS,
    SABR_PT,
    tol=1e-7,
)

# ---------------------------------------------------------------------------
# 3. Price + delta + gamma + vega over 1M options
# ---------------------------------------------------------------------------

N_OPT = 1_000_000
OPT_RANGES = {
    "S": (50.0, 150.0, 301),
    "K": (60.0, 140.0, 302),
    "T": (0.05, 3.0, 303),
    "sigma": (0.05, 0.8, 304),
}
OPT_ARGS = list(OPT_RANGES)
R_FIX = 0.03
_d1 = f"((log(S/K) + ({R_FIX} + sigma**2/2)*T)/(sigma*sqrt(T)))"
_d2 = f"({_d1} - sigma*sqrt(T))"
_pdf = f"(exp(-{_d1}**2/2)/sqrt(2*3.141592653589793))"
GREEKS = [
    f"S*(1 + erf({_d1}/sqrt(2)))/2 - K*exp(-{R_FIX}*T)*(1 + erf({_d2}/sqrt(2)))/2",
    f"(1 + erf({_d1}/sqrt(2)))/2",
    f"{_pdf}/(S*sigma*sqrt(T))",
    f"S*{_pdf}*sqrt(T)",
]
OPT_SAMPLES = [0, 4242, 500_000, 999_999]


def _opt_arrays():
    return [shared_array(f"opt_{n}", N_OPT, s, lo, hi) for n, (lo, hi, s) in OPT_RANGES.items()]


def _opt_fp(out):
    import math

    import numpy as np

    a = np.asarray(out, dtype=float)
    if a.shape[0] != 4:
        a = a.reshape(N_OPT, 4).T
    vals = [math.fsum(row) / N_OPT for row in a]
    for i in OPT_SAMPLES:
        vals += [float(a[0, i]), float(a[2, i])]
    return vals


def _opt_ref():
    import mpmath as mp

    mp.mp.dps = 30

    def go():
        import math

        import numpy as np
        from scipy.special import erf

        arrs = _opt_arrays()
        env = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "erf": erf}
        fs = [eval("lambda " + ", ".join(OPT_ARGS) + ": " + g, env) for g in GREEKS]
        vals = [math.fsum(f(*arrs)) / N_OPT for f in fs]
        menv = {n: getattr(mp, n) for n in ("exp", "log", "sqrt", "erf")}
        mfs = [eval("lambda " + ", ".join(OPT_ARGS) + ": " + GREEKS[k], menv) for k in (0, 2)]
        for i in OPT_SAMPLES:
            x = [mp.mpf(float(a[i])) for a in arrs]
            vals += [float(mfs[0](*x)), float(mfs[1](*x))]
        return vals

    return go, lambda r: r


def _opt_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n) for n in OPT_ARGS}
    es = [ak.parse(g, p, syms) for g in GREEKS]
    fn = ak.compile_expr(es, [syms[n] for n in OPT_ARGS], expected_evals=N_OPT)
    arrs = _opt_arrays()

    def go():
        return ak.numpy_eval(fn, *arrs)

    return go, _opt_fp


def _opt_numpy():
    import numpy as np
    from scipy.special import erf

    S, K, T, sig = _opt_arrays()
    r = R_FIX

    def go():
        sq = np.sqrt(T)
        d1 = (np.log(S / K) + (r + sig * sig / 2) * T) / (sig * sq)
        d2 = d1 - sig * sq
        n1 = (1 + erf(d1 / np.sqrt(2))) / 2
        n2 = (1 + erf(d2 / np.sqrt(2))) / 2
        pdf = np.exp(-d1 * d1 / 2) / np.sqrt(2 * np.pi)
        return np.stack([S * n1 - K * np.exp(-r * T) * n2, n1, pdf / (S * sig * sq), S * pdf * sq])

    return go, _opt_fp


def _opt_sympy():
    import numpy as np
    import sympy as sp

    syms = [sp.Symbol(n) for n in OPT_ARGS]
    loc = dict(zip(OPT_ARGS, syms))
    es = [sp.sympify(g, locals=loc) for g in GREEKS]
    f = sp.lambdify(syms, es, modules=["scipy", "numpy"], cse=True)
    arrs = _opt_arrays()

    def go():
        return f(*arrs)

    return go, lambda r: _opt_fp(np.stack(r))


def _opt_symengine():
    import numpy as np
    import symengine as se

    syms = [se.Symbol(n) for n in OPT_ARGS]
    es = [se.sympify(g) for g in GREEKS]
    f = se.Lambdify(syms, es, backend="llvm", cse=True)
    X = np.ascontiguousarray(np.stack(_opt_arrays(), axis=1))

    def go():
        return f(X)

    return go, _opt_fp


def _opt_wl() -> WL:
    reads = "; ".join(
        f'{n}A = BinaryReadList["{shared_array_path(f"opt_{n}", N_OPT, s, lo, hi)}", "Real64"]'
        for n, (lo, hi, s) in OPT_RANGES.items()
    )
    rules = "{" + ", ".join(f"{n} -> {n}A" for n in OPT_ARGS) + "}"
    samples = ", ".join(f"res[[1, {i + 1}]], res[[3, {i + 1}]]" for i in OPT_SAMPLES)
    return WL(
        setup=f"{reads}; gs = {{{', '.join(_wl(g) for g in GREEKS)}}}; rules = {rules}",
        run="gs /. rules",
        fp=f"Join[Mean /@ res, {{{samples}}}]",
    )


GRID = Case(
    id="bs_grid_1e6",
    task="Price + delta + gamma + vega for 1M options",
    detail="Four closed-form Black–Scholes outputs evaluated for 1,000,000 options with "
    "per-option S, K, T, σ (float64). Alkahest: compile_expr (one fused 4-output kernel) + "
    "numpy_eval, one thread; NumPy: hand-written vectorised code with scipy.special.erf; "
    "SymPy: lambdify(scipy/numpy, cse=True); SymEngine: Lambdify(LLVM, cse=True); Mathematica: "
    "packed-array evaluation. Compilation not timed. Checked: means and sampled values against "
    "mpmath.",
    systems={
        "alkahest": _opt_alkahest,
        "numpy": _opt_numpy,
        "sympy": _opt_sympy,
        "symengine": _opt_symengine,
        "mathematica": _opt_wl(),
    },
    reference=_opt_ref,
    tol=1e-8,
)

SUITES = [
    Suite(
        id="finance",
        label="Finance",
        blurb="Closed-form option greeks and SABR calibration derivatives derived symbolically, "
        "then compiled and evaluated over a million options.",
        script=SCRIPT,
        cases=[BS, SABR_CASE, GRID],
    )
]
