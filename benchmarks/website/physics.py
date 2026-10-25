"""Physics suite: curvature invariants in general relativity, angular-momentum
coupling coefficients, and special functions to 1000 digits."""

from __future__ import annotations

from fractions import Fraction

from _harness import WL, Case, Suite

SCRIPT = "benchmarks/website/physics.py"

# ---------------------------------------------------------------------------
# 1-2. Kretschmann scalar R_abcd R^abcd (Schwarzschild, Kerr)
# ---------------------------------------------------------------------------

COORDS = ["t", "r", "theta", "phi"]
METRICS = {
    "schwarzschild": [
        ["-(1 - 2*M/r)", "0", "0", "0"],
        ["0", "1/(1 - 2*M/r)", "0", "0"],
        ["0", "0", "r**2", "0"],
        ["0", "0", "0", "r**2*sin(theta)**2"],
    ],
    "kerr": [
        [
            "-(1 - 2*M*r/(r**2 + a**2*cos(theta)**2))",
            "0",
            "0",
            "-2*M*r*a*sin(theta)**2/(r**2 + a**2*cos(theta)**2)",
        ],
        ["0", "(r**2 + a**2*cos(theta)**2)/(r**2 - 2*M*r + a**2)", "0", "0"],
        ["0", "0", "r**2 + a**2*cos(theta)**2", "0"],
        [
            "-2*M*r*a*sin(theta)**2/(r**2 + a**2*cos(theta)**2)",
            "0",
            "0",
            "(r**2 + a**2 + 2*M*r*a**2*sin(theta)**2/(r**2 + a**2*cos(theta)**2))*sin(theta)**2",
        ],
    ],
}
KPT = {"M": 1.0, "a": 0.6, "r": 3.0, "theta": 0.7}


def _k_closed_form(which: str) -> float:
    import math

    M, a, r, th = KPT["M"], KPT["a"], KPT["r"], KPT["theta"]
    if which == "schwarzschild":
        return 48 * M**2 / r**6
    c = math.cos(th)
    S = r**2 + a**2 * c**2
    return 48 * M**2 * (r**6 - 15 * r**4 * a**2 * c**2 + 15 * r**2 * a**4 * c**4 - a**6 * c**6) / S**6


def _k_ref(which):
    def ref():
        return (lambda: None), (lambda _r: [_k_closed_form(which)])

    return ref


def _k_alkahest(which):
    def impl():
        import alkahest as ak

        p = ak.ExprPool()
        syms = {n: p.symbol(n) for n in [*COORDS, "M", "a"]}
        g = [[ak.parse(c, p, syms) for c in row] for row in METRICS[which]]
        xs = [syms[n] for n in COORDS]

        def go():
            return ak.Metric(g, xs).kretschmann()

        def fp(K):
            return [ak.eval_expr(K, {syms[k]: v for k, v in KPT.items()})]

        return go, fp

    return impl


def _k_sympy(which):
    """Textbook component computation; Christoffels and the scalar simplified."""

    def impl():
        import sympy as sp

        syms = {n: sp.Symbol(n) for n in [*COORDS, "M", "a"]}
        g = sp.Matrix([[sp.sympify(c, locals=syms) for c in row] for row in METRICS[which]])
        xs = [syms[n] for n in COORDS]
        n = 4

        def go():
            gi = sp.simplify(g.inv())
            Gam = [
                [
                    [
                        sp.simplify(
                            sum(
                                gi[a, d] * (sp.diff(g[d, b], xs[c]) + sp.diff(g[d, c], xs[b]) - sp.diff(g[b, c], xs[d]))
                                for d in range(n)
                            )
                            / 2
                        )
                        for c in range(n)
                    ]
                    for b in range(n)
                ]
                for a in range(n)
            ]
            # R^a_{bcd}
            R = {}
            for a in range(n):
                for b in range(n):
                    for c in range(n):
                        for d in range(c + 1, n):
                            e = sp.diff(Gam[a][b][d], xs[c]) - sp.diff(Gam[a][b][c], xs[d])
                            e += sum(Gam[a][c][k] * Gam[k][b][d] - Gam[a][d][k] * Gam[k][b][c] for k in range(n))
                            if e != 0:
                                R[(a, b, c, d)] = e
            # lower the first index, raise the last three: K = sum R_abcd R^abcd
            Rl = {}
            for a in range(n):
                for b in range(n):
                    for c in range(n):
                        for d in range(c + 1, n):
                            e = sum(g[a, k] * R.get((k, b, c, d), 0) for k in range(n))
                            if e != 0:
                                Rl[(a, b, c, d)] = e
            # R^{abcd} = g^{bf} g^{ch} g^{dk} R^a_{fhk}, using antisymmetry in (h, k)
            K = 0
            for (a, b, c, d), lo in Rl.items():
                up = sum(
                    gi[b, f] * (gi[c, h] * gi[d, k] - gi[c, k] * gi[d, h]) * e
                    for (a2, f, h, k), e in R.items()
                    if a2 == a
                )
                K += 2 * lo * up  # (c, d) and (d, c) contribute equally
            return sp.simplify(K)

        def fp(K):
            return [float(K.evalf(subs={syms[k]: v for k, v in KPT.items()}))]

        return go, fp

    return impl


def _k_wl(which: str) -> WL:
    import re

    rows = []
    for row in METRICS[which]:
        cells = []
        for c in row:
            t = c.replace("**", "^")
            t = re.sub(r"\b(sin|cos)\(", lambda m: m.group(1).capitalize() + "[", t)
            # close the bracket of sin[...]/cos[...] (arguments here are bare symbols)
            t = re.sub(r"(Sin|Cos)\[theta\)", r"\1[theta]", t)
            cells.append(t)
        rows.append("{" + ", ".join(cells) + "}")
    g = "{" + ", ".join(rows) + "}"
    rules = "{" + ", ".join(f"{k} -> {v}" for k, v in KPT.items()) + "}"
    run = (
        "Module[{n = 4, gi, Gam, R, Rl, Ru},"
        " gi = Simplify[Inverse[g]];"
        " Gam = Table[Simplify[Sum[gi[[a, d]] (D[g[[d, b]], xs[[c]]] + D[g[[d, c]], xs[[b]]] - "
        "D[g[[b, c]], xs[[d]]]), {d, n}]/2], {a, n}, {b, n}, {c, n}];"
        " R = Table[D[Gam[[a, b, d]], xs[[c]]] - D[Gam[[a, b, c]], xs[[d]]] + "
        "Sum[Gam[[a, c, k]] Gam[[k, b, d]] - Gam[[a, d, k]] Gam[[k, b, c]], {k, n}], "
        "{a, n}, {b, n}, {c, n}, {d, n}];"
        " Rl = Table[Sum[g[[a, k]] R[[k, b, c, d]], {k, n}], {a, n}, {b, n}, {c, n}, {d, n}];"
        " Ru = Table[Sum[gi[[b, f]] gi[[c, h]] gi[[d, k]] R[[a, f, h, k]], {f, n}, {h, n}, {k, n}], "
        "{a, n}, {b, n}, {c, n}, {d, n}];"
        " Simplify[Total[Flatten[Rl Ru]]]]"
    )
    return WL(setup=f"g = {g}; xs = {{t, r, theta, phi}}", run=run, fp=f"{{N[res /. {rules}]}}")


def _k_case(which: str, label: str) -> Case:
    return Case(
        id=f"kretschmann_{which}",
        task=f"Kretschmann scalar, {label} metric",
        detail=f"R_abcd R^abcd from the {label} metric in Boyer–Lindquist/Schwarzschild "
        "coordinates with symbolic M"
        + (", a" if which == "kerr" else "")
        + ": inverse metric, Christoffel symbols, Riemann tensor, full contraction, simplified "
        "result. SymPy and Mathematica run the textbook component algorithm (Christoffels and the "
        "scalar simplified); Alkahest calls Metric.kretschmann(). Checked against the closed form "
        + ("48M²/r⁶." if which == "schwarzschild" else "of Henry (2000)."),
        systems={
            "alkahest": _k_alkahest(which),
            "sympy": _k_sympy(which),
            "mathematica": _k_wl(which),
        },
        reference=_k_ref(which),
        tol=1e-9,
        repeats=5 if which == "schwarzschild" else 3,
    )


# ---------------------------------------------------------------------------
# 3. Clebsch–Gordan coefficients: every coefficient up to j = 3
# ---------------------------------------------------------------------------

CG_JMAX2 = 6  # 2*jmax


def _cg_list():
    """All non-trivially-allowed (j1, j2, J, m1, m2, M) with j1, j2 <= jmax."""
    out = []
    for tj1 in range(CG_JMAX2 + 1):
        for tj2 in range(CG_JMAX2 + 1):
            for tJ in range(abs(tj1 - tj2), tj1 + tj2 + 1, 2):
                for tm1 in range(-tj1, tj1 + 1, 2):
                    for tm2 in range(-tj2, tj2 + 1, 2):
                        tM = tm1 + tm2
                        if abs(tM) <= tJ:
                            out.append((tj1, tj2, tJ, tm1, tm2, tM))
    return out


def _cg_fp_from(vals):
    import math

    return [
        len(vals),
        math.fsum(vals),
        math.fsum(v * v for v in vals),
        math.fsum((i % 7 - 3) * v for i, v in enumerate(vals)),
    ]


def _cg_racah(tj1, tj2, tJ, tm1, tm2, tM) -> float:
    """Racah's closed formula, exact rationals, independent of every system."""
    import math
    from math import factorial as f

    j1, j2, J = Fraction(tj1, 2), Fraction(tj2, 2), Fraction(tJ, 2)
    m1, m2, M = Fraction(tm1, 2), Fraction(tm2, 2), Fraction(tM, 2)

    def F(x):
        assert x.denominator == 1
        assert x >= 0
        return f(int(x))

    pre = Fraction((tJ + 1) * F(J + j1 - j2) * F(J - j1 + j2) * F(j1 + j2 - J), F(j1 + j2 + J + 1))
    pre *= F(J + M) * F(J - M) * F(j1 - m1) * F(j1 + m1) * F(j2 - m2) * F(j2 + m2)
    s = Fraction(0)
    k = 0
    while True:
        args = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if any(Fraction(a) < 0 for a in args[1:4]):
            break
        if all(Fraction(a) >= 0 for a in args):
            den = 1
            for a in args:
                den *= F(Fraction(a))
            s += Fraction((-1) ** k, den)
        k += 1
    return float(s) * math.sqrt(float(pre))


def _cg_ref():
    def go():
        return _cg_fp_from([_cg_racah(*c) for c in _cg_list()])

    return go, lambda r: r


def _cg_alkahest():
    import alkahest as ak

    cs = [tuple(Fraction(v, 2) for v in c) for c in _cg_list()]

    def go():
        return [ak.clebsch_gordan(*c) for c in cs]

    def fp(res):
        return _cg_fp_from([ak.eval_expr(e, {}) for e in res])

    return go, fp


def _cg_sympy():
    import sympy as sp
    from sympy.physics.wigner import clebsch_gordan

    cs = [tuple(sp.Rational(v, 2) for v in c) for c in _cg_list()]

    def go():
        return [clebsch_gordan(*c) for c in cs]

    return go, lambda res: _cg_fp_from([float(e) for e in res])


def _cg_wl() -> WL:
    lst = "{" + ", ".join("{" + ", ".join(f"{v}/2" for v in c) + "}" for c in _cg_list()) + "}"
    return WL(
        setup=f"cs = {lst}",
        run="Map[ClebschGordan[{#[[1]], #[[4]]}, {#[[2]], #[[5]]}, {#[[3]], #[[6]]}] &, cs]",
        fp="Module[{v = N[res]}, {Length[v], Total[v], Total[v^2], "
        "Total[Table[(Mod[i - 1, 7] - 3) v[[i]], {i, Length[v]}]]}]",
    )


CG = Case(
    id="clebsch_gordan_batch",
    task="All Clebsch–Gordan coefficients up to j = 3",
    detail=f"Every ⟨j1 m1; j2 m2 | J M⟩ with j1, j2 ∈ {{0, 1/2, …, 3}} ({len(_cg_list()):,} "
    "coefficients), exactly (square roots of rationals). Checked: count, sum, sum of squares "
    "and a weighted sum against Racah's formula evaluated independently.",
    systems={"alkahest": _cg_alkahest, "sympy": _cg_sympy, "mathematica": _cg_wl()},
    reference=_cg_ref,
    tol=1e-9,
)

# ---------------------------------------------------------------------------
# 4. Special functions at 1000 significant digits
# ---------------------------------------------------------------------------

DIGITS = 1000
CHECK_DIGITS = 990
SF = [  # (label, alkahest builder, mpmath, Wolfram)
    ("Γ(1/3)", "gamma(R(1,3))", "mp.gamma(mp.mpf(1)/3)", "Gamma[1/3]"),
    ("ζ(3)", "zeta(I(3))", "mp.zeta(3)", "Zeta[3]"),
    ("ζ(1/2)", "zeta(R(1,2))", "mp.zeta(mp.mpf(1)/2)", "Zeta[1/2]"),
    ("erf(2)", "erf(I(2))", "mp.erf(2)", "Erf[2]"),
    ("erfc(5)", "erfc(I(5))", "mp.erfc(5)", "Erfc[5]"),
    ("J₀(10)", "bessel_j(I(0), I(10))", "mp.besselj(0, 10)", "BesselJ[0, 10]"),
    ("J₃(7)", "bessel_j(I(3), I(7))", "mp.besselj(3, 7)", "BesselJ[3, 7]"),
    ("Y₀(3)", "bessel_y(I(0), I(3))", "mp.bessely(0, 3)", "BesselY[0, 3]"),
    ("K₁(2)", "bessel_k(I(1), I(2))", "mp.besselk(1, 2)", "BesselK[1, 2]"),
    ("Ai(2)", "airy_ai(I(2))", "mp.airyai(2)", "AiryAi[2]"),
    ("Ei(1)", "exp_integral_ei(I(1))", "mp.ei(1)", "ExpIntegralEi[1]"),
    ("W(1)", "lambert_w(I(1))", "mp.lambertw(1).real", "ProductLog[1]"),
    ("Li₂(1/3)", "polylog(I(2), R(1,3))", "mp.polylog(2, mp.mpf(1)/3)", "PolyLog[2, 1/3]"),
    ("ψ(1/4)", "digamma(R(1,4))", "mp.digamma(mp.mpf(1)/4)", "PolyGamma[0, 1/4]"),
    ("Si(3)", "sin_integral(I(3))", "mp.si(3)", "SinIntegral[3]"),
    ("exp(π√163)", "exp(PI*sqrt(I(163)))", "mp.exp(mp.pi*mp.sqrt(163))", "Exp[Pi Sqrt[163]]"),
]


def digits_key(dec: str, nd: int = CHECK_DIGITS) -> str:
    """Canonical 'sign|exponent|first nd significant digits' of a decimal string."""
    from decimal import Decimal, getcontext

    getcontext().prec = 4000
    d = Decimal(dec)
    sign, digs, exp = d.as_tuple()
    s = "".join(map(str, digs)).lstrip("0")
    e = len("".join(map(str, digs))) + exp - (len("".join(map(str, digs))) - len(s))
    return f"{'-' if sign else '+'}|{e}|{s[:nd]}"


def _sf_mpmath_vals(dps):
    import mpmath as mp

    mp.mp.dps = dps
    return [eval(m, {"mp": mp}) for _l, _a, m, _w in SF]


def _sf_ref():
    def go():
        import mpmath as mp

        vals = _sf_mpmath_vals(DIGITS + 200)
        return [digits_key(mp.nstr(v, DIGITS + 150, strip_zeros=False)) for v in vals]

    return go, lambda r: r


def _sf_mpmath():
    import mpmath as mp

    def go():
        return _sf_mpmath_vals(DIGITS)

    def fp(vals):
        return [digits_key(mp.nstr(v, DIGITS, strip_zeros=False)) for v in vals]

    return go, fp


def _sf_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    env = {
        "R": p.rational,
        "I": p.integer,
        "PI": ak.parse("pi", p),
        **{n: getattr(ak, n) for n in ("gamma", "zeta", "erf", "erfc", "bessel_j", "bessel_y", "bessel_k")},
        **{n: getattr(ak, n) for n in ("airy_ai", "exp_integral_ei", "lambert_w", "polylog")},
        **{n: getattr(ak, n) for n in ("digamma", "sin_integral", "exp", "sqrt")},
    }
    exprs = [eval(a, env) for _l, a, _m, _w in SF]
    prec_bits = int(DIGITS * 3.3219280948873622) + 32

    def go():
        return [ak.interval_eval(e, {}, prec=prec_bits) for e in exprs]

    def fp(balls):
        from decimal import Decimal, getcontext

        getcontext().prec = 4000
        out = []
        for b in balls:
            mid, rad = Decimal(b.mid_str()), Decimal(b.rad_str())
            # the ball must certify the digits we compare
            ok = mid != 0 and rad / abs(mid) < Decimal(10) ** (-(CHECK_DIGITS + 2))
            out.append(digits_key(b.mid_str()) if ok else "FAILED")
        return out

    return go, fp


def _sf_sympy():
    import sympy as sp

    exprs = [
        sp.gamma(sp.Rational(1, 3)),
        sp.zeta(3),
        sp.zeta(sp.Rational(1, 2)),
        sp.erf(2),
        sp.erfc(5),
        sp.besselj(0, 10),
        sp.besselj(3, 7),
        sp.bessely(0, 3),
        sp.besselk(1, 2),
        sp.airyai(2),
        sp.Ei(1),
        sp.LambertW(1),
        sp.polylog(2, sp.Rational(1, 3)),
        sp.digamma(sp.Rational(1, 4)),
        sp.Si(3),
        sp.exp(sp.pi * sp.sqrt(163)),
    ]

    def go():
        return [sp.N(e, DIGITS) for e in exprs]

    def fp(vals):
        return [digits_key(sp.Float(v, DIGITS).__str__()) if v.is_Float else "FAILED" for v in vals]

    return go, fp


SPECFUN = Case(
    id="special_functions_1000_digits",
    task="16 special-function values to 1000 digits",
    detail="Γ(1/3), ζ(3), ζ(1/2), erf(2), erfc(5), J₀(10), J₃(7), Y₀(3), K₁(2), Ai(2), Ei(1), "
    "W(1), Li₂(1/3), ψ(1/4), Si(3), exp(π√163) at 1000 significant digits. Alkahest: rigorous "
    "Arb balls via interval_eval (radius checked to certify the digits); mpmath at dps=1000 "
    "(the engine behind SymPy's evalf); SymPy N(·, 1000); Mathematica N[·, 1000]. Checked: the "
    "first 990 digits of every value against mpmath at 1200 digits.",
    systems={
        "alkahest": _sf_alkahest,
        "mpmath": _sf_mpmath,
        "sympy": _sf_sympy,
        "mathematica": WL(
            setup="xs = Hold[{" + ", ".join(w for _l, _a, _m, w in SF) + "}]",
            run=f"N[ReleaseHold[xs], {DIGITS}]",
            fp="Map[Module[{rd = RealDigits[#, 10, "
            f'{CHECK_DIGITS}]}}, If[# < 0, "-", "+"] <> "|" <> ToString[rd[[2]]] <> "|" <> '
            "StringJoin[ToString /@ rd[[1]]]] &, res]",
        ),
    },
    reference=_sf_ref,
    note="mpmath and Arb cache constants such as π across calls; the warm-up run populates "
    "those caches for every system alike.",
)

SUITES = [
    Suite(
        id="physics",
        label="Physics",
        blurb="Curvature invariants of black-hole metrics, angular-momentum coupling "
        "coefficients, and special functions evaluated to a thousand digits.",
        script=SCRIPT,
        cases=[_k_case("schwarzschild", "Schwarzschild"), _k_case("kerr", "Kerr"), CG, SPECFUN],
    )
]
