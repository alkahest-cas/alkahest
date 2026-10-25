"""Core CAS suite: expansion, factorisation, GCD, integration, Gröbner bases, real roots."""

from __future__ import annotations

import re

from _harness import WL, Case, Suite, wl_str
from _inputs import (
    cyclic,
    katsura,
    legendre_integer,
    rand_dense_univariate,
    rand_sparse_multivariate,
)

SCRIPT = "benchmarks/website/core.py"
V4 = ["x", "y", "z", "w"]

# ---------------------------------------------------------------------------
# 1. Fateman's expansion benchmark
# ---------------------------------------------------------------------------

FATEMAN_N = 20
FATEMAN_F = f"(1 + x + y + z + w)**{FATEMAN_N}"
POINT4 = {"x": 2, "y": -3, "z": 5, "w": 7}


def _fateman_ref():
    def go():
        u = 1 + sum(POINT4.values())
        f = u**FATEMAN_N
        from math import comb

        # number of monomials of total degree <= 2N in 4 variables
        nterms = comb(2 * FATEMAN_N + 4, 4)
        return [nterms, str(f * (f + 1))]

    return go, lambda r: r


def _fateman_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n) for n in V4}
    f = ak.parse(FATEMAN_F, p, syms)
    e = f * (f + 1)
    vs = [syms[n] for n in V4]

    def go():
        return ak.poly_normal(e, vs)

    def fp(r):
        nterms = len(r.node()[1])
        val = ak.simplify(ak.subs(r, {syms[k]: p.integer(v) for k, v in POINT4.items()})).value
        return [nterms, str(val)]

    return go, fp


def _fateman_sympy():
    import sympy as sp

    syms = sp.symbols(V4)
    f = sp.sympify(FATEMAN_F, locals=dict(zip(V4, syms)))
    e = f * (f + 1)

    def go():
        return sp.expand(e)

    def fp(r):
        return [len(r.args), str(r.subs(dict(zip(syms, POINT4.values()))))]

    return go, fp


def _fateman_symengine():
    import symengine as se

    f = se.sympify(FATEMAN_F)
    e = f * (f + 1)

    def go():
        return se.expand(e)

    def fp(r):
        return [len(r.args), str(r.subs({se.Symbol(k): v for k, v in POINT4.items()}))]

    return go, fp


_PT4_WL = "{" + ", ".join(f"{k} -> {v}" for k, v in POINT4.items()) + "}"

FATEMAN = Case(
    id="expand_fateman",
    task="Expand f·(f+1), f = (1+x+y+z+w)^20",
    detail="Fateman's polynomial-expansion benchmark: 4 variables, 135,751 terms in the result. "
    "Checked by term count and the exact value at (2,-3,5,7).",
    systems={
        "alkahest": _fateman_alkahest,
        "sympy": _fateman_sympy,
        "symengine": _fateman_symengine,
        "mathematica": WL(
            setup=f"f = {wl_str(FATEMAN_F)}; e = f (f + 1)",
            run="Expand[e]",
            fp=f"{{Length[res], res /. {_PT4_WL}}}",
        ),
    },
    reference=_fateman_ref,
    repeats=5,
    timeout_s=300,
    note="SymPy is timed with sympy.expand, its standard expansion entry point.",
)

# ---------------------------------------------------------------------------
# 2. Univariate factorisation over Z
# ---------------------------------------------------------------------------

UF_PARTS = [(60, 1), (60, 2), (80, 3)]  # (degree, seed)
UF_BITS = 20
UF_POLY = " * ".join(f"({rand_dense_univariate(d, UF_BITS, s)})" for d, s in UF_PARTS)
UF_DEGREE = sum(d for d, _ in UF_PARTS)


def _degrees_ref(degs):
    def ref():
        return (lambda: None), (lambda _r: [len(degs), *sorted(degs)])

    return ref


def _ufactor_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    x = p.symbol("x")
    up = ak.UniPoly.from_symbolic(ak.poly_normal(ak.parse(UF_POLY, p, {"x": x}), [x]), x)

    def go():
        return up.factor_z()

    def fp(r):
        degs = []
        for f, m in r.factor_list():
            if f.degree > 0:
                degs += [f.degree] * m
        return [len(degs), *sorted(degs)]

    return go, fp


def _ufactor_sympy():
    import sympy as sp

    x = sp.Symbol("x")
    P = sp.Poly(sp.expand(sp.sympify(UF_POLY, locals={"x": x})), x)

    def go():
        return P.factor_list()

    def fp(r):
        degs = []
        for f, m in r[1]:
            degs += [f.degree()] * m
        return [len(degs), *sorted(degs)]

    return go, fp


UFACTOR = Case(
    id="factor_univariate",
    task=f"Factor a dense degree-{UF_DEGREE} polynomial over Z",
    detail=f"Product of three random dense polynomials of degree {', '.join(str(d) for d, _ in UF_PARTS)} "
    f"with {UF_BITS}-bit coefficients, expanded (degree {UF_DEGREE}). "
    "Checked: the irreducible factor degrees are recovered.",
    systems={
        "alkahest": _ufactor_alkahest,
        "sympy": _ufactor_sympy,
        "mathematica": WL(
            setup=f"p = Expand[{wl_str(UF_POLY)}]",
            run="FactorList[p]",
            fp="Module[{d = Flatten[Table[ConstantArray[Exponent[f[[1]], x], f[[2]]], {f, "
            "Select[res, Exponent[#[[1]], x] > 0 &]}]]}, Join[{Length[d]}, Sort[d]]]",
        ),
    },
    reference=_degrees_ref([d for d, _ in UF_PARTS]),
)

# ---------------------------------------------------------------------------
# 3. Multivariate factorisation over Z
# ---------------------------------------------------------------------------

MF_PARTS = [rand_sparse_multivariate(4, 24, 10, 50, seed, V4) for seed in (11, 12, 13)]
MF_POLY = " * ".join(f"({q})" for q in MF_PARTS)


def _mfactor_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n) for n in V4}
    vs = [syms[n] for n in V4]
    mp = ak.MultiPoly.from_symbolic(ak.poly_normal(ak.parse(MF_POLY, p, syms), vs), vs)

    def go():
        return mp.factor_z()

    def fp(r):
        degs = []
        for f, m in r.factor_list():
            if f.total_degree > 0:
                degs += [f.total_degree] * m
        return [len(degs), *sorted(degs)]

    return go, fp


def _mfactor_sympy():
    import sympy as sp

    syms = sp.symbols(V4)
    P = sp.Poly(sp.expand(sp.sympify(MF_POLY, locals=dict(zip(V4, syms)))), *syms)

    def go():
        return P.factor_list()

    def fp(r):
        degs = []
        for f, m in r[1]:
            degs += [f.total_degree()] * m
        return [len(degs), *sorted(degs)]

    return go, fp


def _mfactor_ref():
    import sympy as sp

    syms = sp.symbols(V4)
    degs = [sp.Poly(sp.sympify(q, locals=dict(zip(V4, syms))), *syms).total_degree() for q in MF_PARTS]
    return (lambda: None), (lambda _r: [len(degs), *sorted(degs)])


MFACTOR = Case(
    id="factor_multivariate",
    task="Factor a 4-variable polynomial over Z",
    detail="Product of three random sparse 4-variable polynomials (25 terms, total degree <= 10 "
    "each), expanded to 5,069 terms. Checked: the three irreducible factors and their degrees are recovered.",
    systems={
        "alkahest": _mfactor_alkahest,
        "sympy": _mfactor_sympy,
        "mathematica": WL(
            setup=f"p = Expand[{wl_str(MF_POLY)}]; vs = {{x, y, z, w}}",
            run="FactorList[p]",
            fp="Module[{d = Flatten[Table[ConstantArray[Max[Total /@ CoefficientRules[f[[1]], vs][[All, 1]]], "
            "f[[2]]], {f, Select[res, !NumericQ[#[[1]]] &]}]]}, Join[{Length[d]}, Sort[d]]]",
        ),
    },
    reference=_mfactor_ref,
)

# ---------------------------------------------------------------------------
# 4. Multivariate GCD
# ---------------------------------------------------------------------------

G_COMMON = rand_sparse_multivariate(4, 60, 12, 99, 21, V4)
G_A = rand_sparse_multivariate(4, 60, 12, 99, 22, V4)
G_B = rand_sparse_multivariate(4, 60, 12, 99, 23, V4)
GCD_P = f"({G_COMMON}) * ({G_A})"
GCD_Q = f"({G_COMMON}) * ({G_B})"


def _nterms_str(s: str) -> int:
    return len(re.split(r" [+-] ", s.strip()))


def _gcd_ref():
    import sympy as sp

    syms = sp.symbols(V4)
    g = sp.Poly(sp.sympify(G_COMMON, locals=dict(zip(V4, syms))), *syms)
    return (lambda: None), (lambda _r: [g.total_degree(), len(g.terms())])


def _gcd_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {n: p.symbol(n) for n in V4}
    vs = [syms[n] for n in V4]
    A = ak.MultiPoly.from_symbolic(ak.poly_normal(ak.parse(GCD_P, p, syms), vs), vs)
    B = ak.MultiPoly.from_symbolic(ak.poly_normal(ak.parse(GCD_Q, p, syms), vs), vs)

    def go():
        return A.gcd(B)

    def fp(r):
        return [r.total_degree, _nterms_str(str(r))]

    return go, fp


def _gcd_sympy():
    import sympy as sp

    syms = sp.symbols(V4)
    loc = dict(zip(V4, syms))
    A = sp.Poly(sp.expand(sp.sympify(GCD_P, locals=loc)), *syms)
    B = sp.Poly(sp.expand(sp.sympify(GCD_Q, locals=loc)), *syms)

    def go():
        return A.gcd(B)

    def fp(r):
        return [r.total_degree(), len(r.terms())]

    return go, fp


GCD = Case(
    id="gcd_multivariate",
    task="GCD of two 4-variable polynomials",
    detail="g·a and g·b with g, a, b random sparse 4-variable polynomials (61 terms, total degree "
    "<= 12 each); the inputs have ~2,100 terms each. Checked: the GCD has g's degree and term count.",
    systems={
        "alkahest": _gcd_alkahest,
        "sympy": _gcd_sympy,
        "mathematica": WL(
            setup=f"a = Expand[{wl_str(GCD_P)}]; b = Expand[{wl_str(GCD_Q)}]; vs = {{x, y, z, w}}",
            run="PolynomialGCD[a, b]",
            fp="{Max[Total /@ CoefficientRules[res, vs][[All, 1]]], Length[CoefficientRules[res, vs]]}",
        ),
    },
    reference=_gcd_ref,
)

# ---------------------------------------------------------------------------
# 5. Indefinite integration of a batch of integrands
# ---------------------------------------------------------------------------

INTEGRANDS = [
    "(x**3 + 2)/((x - 1)**2*(x**2 + 1))",
    "(3*x**5 - 2*x**3 + x - 7)/((x + 2)**3*(x**2 + x + 1)**2)",
    "1/(x**4 - 1)",
    "1/(x**3 + 1)",
    "(2*x + 1)/(x**2 + x + 1)**3",
    "x**2*exp(x)*sin(x)",
    "x**3*log(x)**2",
    "1/(x*log(x))",
    "exp(x)/(1 + exp(x))",
    "sin(x)**3*cos(x)**2",
    "x*atan(x)",
    "log(x**2 + 1)",
    "x**4*cos(2*x)",
    "exp(2*x)*cos(3*x)",
    "x*exp(x**2)",
    "(x**2 + 1)/(x**4 + x**2 + 1)",
]
A_PT, B_PT = 1.3, 2.1


def _int_ref():
    import mpmath as mp

    mp.mp.dps = 30

    def go():
        vals = []
        for s in INTEGRANDS:
            f = eval(
                "lambda x: " + s,
                {"exp": mp.exp, "sin": mp.sin, "cos": mp.cos, "log": mp.log, "atan": mp.atan},
            )
            vals.append(float(mp.quad(f, [A_PT, B_PT])))
        return vals

    return go, lambda r: r


def _int_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    x = p.symbol("x")
    fs = [ak.parse(s, p, {"x": x}) for s in INTEGRANDS]

    def go():
        out = []
        for f in fs:
            try:
                out.append(ak.integrate(f, x).value)
            except Exception:
                out.append(None)
        return out

    def fp(rs):
        vals = []
        for F in rs:
            if F is None:
                vals.append("FAILED")
            else:
                vals.append(ak.eval_expr(F, {x: B_PT}) - ak.eval_expr(F, {x: A_PT}))
        return vals

    return go, fp


def _int_sympy():
    import sympy as sp

    x = sp.Symbol("x")
    fs = [sp.sympify(s, locals={"x": x}) for s in INTEGRANDS]

    def go():
        return [sp.integrate(f, x) for f in fs]

    def fp(rs):
        vals = []
        for F in rs:
            if F.has(sp.Integral):
                vals.append("FAILED")
                continue
            v = complex(sp.N(F.subs(x, B_PT) - F.subs(x, A_PT), 20))
            vals.append(v.real if abs(v.imag) < 1e-9 else "COMPLEX")
        return vals

    return go, fp


def _wl_integrands() -> str:
    # Translate function-call syntax f(...) to f[...] for the few heads used.
    out = []
    for s in INTEGRANDS:
        t = wl_str(s)
        for py, wl in (
            ("exp", "Exp"),
            ("sin", "Sin"),
            ("cos", "Cos"),
            ("log", "Log"),
            ("atan", "ArcTan"),
        ):
            t = re.sub(rf"\b{py}\(", wl + "(", t)
        out.append(_paren_to_brackets(t))
    return "{" + ", ".join(out) + "}"


def _paren_to_brackets(s: str) -> str:
    """Turn ``Head(args)`` into ``Head[args]`` for capitalised WL heads."""
    res = []
    stack = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            j = len(res) - 1
            is_call = j >= 0 and (res[j].isalpha())
            stack.append(is_call)
            res.append("[" if is_call else "(")
        elif ch == ")":
            res.append("]" if stack.pop() else ")")
        else:
            res.append(ch)
        i += 1
    return "".join(res)


INTEGRATE = Case(
    id="integrate_batch",
    task=f"Indefinite integrals: batch of {len(INTEGRANDS)}",
    detail="16 rational and elementary integrands (partial fractions with repeated and "
    "irreducible quadratic factors, x^n·exp·sin, log powers, atan, ...). Each antiderivative "
    f"is checked by F({B_PT}) - F({A_PT}) against mpmath quadrature (rel. tol 1e-9).",
    systems={
        "alkahest": _int_alkahest,
        "sympy": _int_sympy,
        "mathematica": WL(
            setup=f"fs = {_wl_integrands()}",
            run="Map[Integrate[#, x] &, fs]",
            fp=f'Map[If[!FreeQ[#, Integrate], "FAILED", '
            f"Module[{{v = N[(# /. x -> {B_PT}) - (# /. x -> {A_PT}), 20]}}, "
            'If[Abs[Im[v]] < 10^-9, Re[v], "COMPLEX"]]] &, res]',
        ),
    },
    reference=_int_ref,
    tol=1e-9,
)

# ---------------------------------------------------------------------------
# 6. Gröbner bases
# ---------------------------------------------------------------------------


def _gb_case(name: str, eqs: list[str], vs: list[str], label: str, note: str | None = None) -> Case:
    def _alk():
        import alkahest as ak

        p = ak.ExprPool()
        syms = {n: p.symbol(n) for n in vs}
        es = [ak.parse(e, p, syms) for e in eqs]
        xs = [syms[n] for n in vs]

        def go():
            return ak.GroebnerBasis.compute(es, xs, order="grevlex")

        def fp(gb):
            polys = gb.polynomials()
            degs = [max(sum(m) for m, _c in g.terms()) for g in polys]
            return [len(polys), sum(degs)]

        return go, fp

    def _sym():
        import sympy as sp

        syms = sp.symbols(vs)
        loc = dict(zip(vs, syms))
        es = [sp.sympify(e, locals=loc) for e in eqs]

        def go():
            return sp.groebner(es, *syms, order="grevlex")

        def fp(gb):
            return [len(gb.exprs), sum(sp.Poly(e, *syms).total_degree() for e in gb.exprs)]

        return go, fp

    wl_vars = "{" + ", ".join(vs) + "}"
    wl_eqs = "{" + ", ".join(wl_str(e) for e in eqs) + "}"
    return Case(
        id=name,
        task=label,
        detail=f"Reduced Gröbner basis, grevlex, {len(vs)} variables, over Q. Checked: number of "
        "basis elements and the sum of their total degrees (the reduced basis is unique).",
        systems={
            "alkahest": _alk,
            "sympy": _sym,
            "mathematica": WL(
                setup=f"eqs = {wl_eqs}; vs = {wl_vars}",
                run="GroebnerBasis[eqs, vs, MonomialOrder -> DegreeReverseLexicographic]",
                fp="{Length[res], Total[Map[Max[Total /@ CoefficientRules[#, vs][[All, 1]]] &, res]]}",
            ),
        },
        reference=None,  # consensus of the systems that finish
        repeats=3,
        note=note,
    )


GB_CASES = [
    _gb_case("groebner_cyclic5", *cyclic(5), "Gröbner basis: cyclic-5"),
    _gb_case("groebner_katsura5", *katsura(5), "Gröbner basis: katsura-5"),
    _gb_case("groebner_katsura6", *katsura(6), "Gröbner basis: katsura-6"),
    _gb_case("groebner_cyclic6", *cyclic(6), "Gröbner basis: cyclic-6"),
]

# ---------------------------------------------------------------------------
# 7. Real-root isolation
# ---------------------------------------------------------------------------

LEG_N = 200
LEG_POLY = legendre_integer(LEG_N)


def _rr_ref():
    return (lambda: None), (lambda _r: [LEG_N])


def _rr_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    x = p.symbol("x")
    e = ak.parse(LEG_POLY, p, {"x": x})

    def go():
        return ak.real_roots(e, x)

    return go, lambda r: [len(r)]


def _rr_sympy():
    import sympy as sp

    x = sp.Symbol("x")
    P = sp.Poly(sp.sympify(LEG_POLY, locals={"x": x}), x)

    def go():
        return P.intervals()

    return go, lambda r: [len(r)]


RROOTS = Case(
    id="real_roots_legendre",
    task=f"Isolate all real roots, degree {LEG_N}",
    detail=f"2^{LEG_N}·P_{LEG_N}(x) (Legendre; its roots are the Gauss–Legendre nodes, clustered "
    "near ±1). Exact isolation into disjoint rational intervals. Checked: all 200 roots found.",
    systems={
        "alkahest": _rr_alkahest,
        "sympy": _rr_sympy,
        "mathematica": WL(
            setup=f"p = {wl_str(LEG_POLY)}",
            run="RootIntervals[p]",
            fp="{Length[res[[1]]]}",
        ),
    },
    reference=_rr_ref,
)

SUITES = [
    Suite(
        id="core",
        label="Core CAS",
        blurb="Polynomial arithmetic, factorisation, GCD, integration, Gröbner bases and real "
        "roots — the kernels every other workload sits on.",
        script=SCRIPT,
        cases=[FATEMAN, UFACTOR, MFACTOR, GCD, INTEGRATE, *GB_CASES, RROOTS],
    )
]
