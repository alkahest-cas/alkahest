"""Engineering / robotics suite: multibody equations of motion, manipulator
kinematics, and compiled evaluation of the kinematics at scale."""

from __future__ import annotations

import json

from _harness import DATA_DIR, WL, Case, Suite, shared_array, shared_array_path

SCRIPT = "benchmarks/website/engineering.py"

# ---------------------------------------------------------------------------
# 1-2. Euler–Lagrange equations of an n-link planar pendulum
# ---------------------------------------------------------------------------


def _state(n: int) -> dict:
    return {
        "q": [0.1 * (i + 1) - 0.25 for i in range(n)],
        "qd": [0.3 - 0.07 * i for i in range(n)],
        "qdd": [0.05 * i - 0.2 for i in range(n)],
        "m": [1.0 + 0.1 * i for i in range(n)],
        "l": [1.0 - 0.05 * i for i in range(n)],
        "g": 9.81,
    }


def _eom_ref(n: int):
    """Closed-form n-link pendulum dynamics: M(q) q'' + c(q, q') + g(q)."""

    def ref():
        import math

        s = _state(n)
        q, qd, qdd, m, ln, g = s["q"], s["qd"], s["qdd"], s["m"], s["l"], s["g"]

        def mu(i, j):
            return sum(m[k] for k in range(max(i, j), n))

        out = []
        for i in range(n):
            r = 0.0
            for j in range(n):
                c = mu(i, j) * ln[i] * ln[j]
                r += c * math.cos(q[i] - q[j]) * qdd[j] + c * math.sin(q[i] - q[j]) * qd[j] ** 2
            r += mu(i, i) * g * ln[i] * math.sin(q[i])
            out.append(r)
        return (lambda: None), (lambda _r: out)

    return ref


def _eom_alkahest(n: int):
    def impl():
        import alkahest as ak

        p = ak.ExprPool()
        t = p.symbol("t")
        g = p.symbol("g")
        m = [p.symbol(f"m{i}") for i in range(n)]
        ln = [p.symbol(f"l{i}") for i in range(n)]
        q = [p.func(f"q{i}", [t]) for i in range(n)]
        px, py = p.integer(0), p.integer(0)
        T, V = p.integer(0), p.integer(0)
        for i in range(n):
            px = px + ln[i] * ak.sin(q[i])
            py = py - ln[i] * ak.cos(q[i])
            vx, vy = ak.diff(px, t).value, ak.diff(py, t).value
            T = T + m[i] * (vx**2 + vy**2) / 2
            V = V + m[i] * g * py
        L = T - V

        def go():
            return ak.euler_lagrange(L, q, t)

        def fp(eoms):
            exprs, mp = ak.functions_to_symbols(eoms, q, t)
            syms = {str(v): v for v in mp.values()}
            s = _state(n)
            env = {g: s["g"]}
            for i in range(n):
                env[syms[f"q{i}"]] = s["q"][i]
                env[syms[f"q{i}_dot"]] = s["qd"][i]
                env[syms[f"q{i}_ddot"]] = s["qdd"][i]
                env[m[i]] = s["m"][i]
                env[ln[i]] = s["l"][i]
            return [ak.eval_expr(e, env) for e in exprs]

        return go, fp

    return impl


def _eom_sympy(n: int):
    def impl():
        import sympy as sp
        from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols

        q = dynamicsymbols(f"q0:{n}")
        t = dynamicsymbols._t
        g = sp.Symbol("g")
        m = sp.symbols(f"m0:{n}")
        ln = sp.symbols(f"l0:{n}")
        px = py = 0
        T = V = 0
        for i in range(n):
            px += ln[i] * sp.sin(q[i])
            py += -ln[i] * sp.cos(q[i])
            T += m[i] * (px.diff(t) ** 2 + py.diff(t) ** 2) / 2
            V += m[i] * g * py
        L = T - V

        def go():
            return LagrangesMethod(L, q).form_lagranges_equations()

        def fp(eoms):
            s = _state(n)
            Q = sp.symbols(f"Q0:{n}")
            Qd = sp.symbols(f"Qd0:{n}")
            Qdd = sp.symbols(f"Qdd0:{n}")
            rep = {q[i].diff(t, 2): Qdd[i] for i in range(n)}
            rep1 = {q[i].diff(t): Qd[i] for i in range(n)}
            rep2 = {q[i]: Q[i] for i in range(n)}
            env = {g: s["g"]}
            for i in range(n):
                env.update({Q[i]: s["q"][i], Qd[i]: s["qd"][i], Qdd[i]: s["qdd"][i]})
                env.update({m[i]: s["m"][i], ln[i]: s["l"][i]})
            return [float(e.subs(rep).subs(rep1).subs(rep2).evalf(subs=env)) for e in eoms]

        return go, fp

    return impl


def _eom_wl(n: int) -> WL:
    s = _state(n)
    rules = []
    for i in range(n):
        rules += [
            f"Derivative[2][q{i}][t] -> {s['qdd'][i]}",
            f"Derivative[1][q{i}][t] -> {s['qd'][i]}",
            f"q{i}[t] -> {s['q'][i]}",
            f"m{i} -> {s['m'][i]}",
            f"l{i} -> {s['l'][i]}",
        ]
    rules.append(f"g -> {s['g']}")
    setup = (
        f'n = {n}; qs = Table[Symbol["q" <> ToString[i]][t], {{i, 0, n - 1}}]; '
        'ms = Table[Symbol["m" <> ToString[i]], {i, 0, n - 1}]; '
        'ls = Table[Symbol["l" <> ToString[i]], {i, 0, n - 1}]; '
        "xs = Accumulate[ls Sin[qs]]; ys = Accumulate[-ls Cos[qs]]; "
        "L = Total[ms (D[xs, t]^2 + D[ys, t]^2)/2] - Total[ms g ys]"
    )
    return WL(
        setup=setup,
        run="Table[D[D[L, D[qs[[i]], t]], t] - D[L, qs[[i]]], {i, n}]",
        fp=f"N[res //. {{{', '.join(rules)}}}]",
    )


def _eom_case(n: int) -> Case:
    return Case(
        id=f"eom_{n}link",
        task=f"Euler–Lagrange equations, {n}-link pendulum",
        detail=f"Derive the {n} equations of motion d/dt(∂L/∂q̇ᵢ) − ∂L/∂qᵢ of a planar {n}-link "
        "pendulum with symbolic masses, lengths and g, from its Lagrangian (built untimed). "
        "SymPy uses sympy.physics.mechanics.LagrangesMethod. Checked: each equation at a "
        "numeric state against the closed-form M(q)q̈ + c(q,q̇) + g(q).",
        systems={
            "alkahest": _eom_alkahest(n),
            "sympy": _eom_sympy(n),
            "mathematica": _eom_wl(n),
        },
        reference=_eom_ref(n),
        tol=1e-9,
    )


# ---------------------------------------------------------------------------
# 3. Forward kinematics + Jacobian of a 6R arm (PUMA-560 DH table)
# ---------------------------------------------------------------------------

# alpha_i given by (cos, sin) so the DH matrices stay exact
DH = [  # (d, a, cos(alpha), sin(alpha))
    ("0", "0", 0, -1),
    ("0", "a2", 1, 0),
    ("d3", "a3", 0, 1),
    ("d4", "0", 0, -1),
    ("0", "0", 0, 1),
    ("d6", "0", 1, 0),
]
LINK = {"a2": 0.4318, "a3": 0.0203, "d3": 0.15005, "d4": 0.4318, "d6": 0.056}
QPT = [0.3, -0.7, 0.45, 1.1, -0.35, 0.8]


def _dh_rows(i: int, c, s, sym, num):
    """DH matrix rows for joint i using the given constructors."""
    d, a, ca, sa = DH[i]
    q = sym(f"q{i + 1}")
    ct, st = c(q), s(q)
    dd = sym(d) if d != "0" else num(0)
    aa = sym(a) if a != "0" else num(0)
    return [
        [ct, -st * num(ca), st * num(sa), aa * ct],
        [st, ct * num(ca), -ct * num(sa), aa * st],
        [num(0), num(sa), num(ca), dd],
        [num(0), num(0), num(0), num(1)],
    ]


def _fk_numeric(qs, mpmod):
    import mpmath as mp

    T = mp.eye(4)
    for i in range(6):
        d, a, ca, sa = DH[i]
        dd = LINK.get(d, 0)
        aa = LINK.get(a, 0)
        ct, st = mp.cos(qs[i]), mp.sin(qs[i])
        A = mp.matrix(
            [
                [ct, -st * ca, st * sa, aa * ct],
                [st, ct * ca, -ct * sa, aa * st],
                [0, sa, ca, dd],
                [0, 0, 0, 1],
            ]
        )
        T = T * A
    return [T[0, 3], T[1, 3], T[2, 3]]


def _jac_ref():
    import mpmath as mp

    mp.mp.dps = 30

    def go():
        x0 = [mp.mpf(v) for v in QPT]
        pos = [float(v) for v in _fk_numeric(x0, mp)]
        J = []
        for r in range(3):
            for j in range(6):
                order = [0] * 6
                order[j] = 1
                J.append(float(mp.diff(lambda *qq, r=r: _fk_numeric(list(qq), mp)[r], x0, tuple(order))))
        return pos + J

    return go, lambda r: r


def _jac_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    cache: dict = {}

    def sym(n):
        if n not in cache:
            cache[n] = p.symbol(n)
        return cache[n]

    rows = [_dh_rows(i, ak.cos, ak.sin, sym, p.integer) for i in range(6)]
    qs = [sym(f"q{i + 1}") for i in range(6)]

    def go():
        T = ak.Matrix(rows[0])
        for i in range(1, 6):
            T = T * ak.Matrix(rows[i])
        P = [T.get(r, 3) for r in range(3)]
        return P, ak.jacobian(P, qs)

    def fp(res):
        P, J = res
        env = {sym(k): v for k, v in LINK.items()}
        env.update(dict(zip(qs, QPT)))
        return [ak.eval_expr(e, env) for e in P] + [ak.eval_expr(e, env) for row in J.to_list() for e in row]

    return go, fp


def _jac_sympy():
    import sympy as sp

    cache: dict = {}

    def sym(n):
        return cache.setdefault(n, sp.Symbol(n))

    rows = [_dh_rows(i, sp.cos, sp.sin, sym, sp.Integer) for i in range(6)]
    qs = [sym(f"q{i + 1}") for i in range(6)]

    def go():
        T = sp.Matrix(rows[0])
        for i in range(1, 6):
            T = T * sp.Matrix(rows[i])
        P = sp.Matrix([T[r, 3] for r in range(3)])
        return P, P.jacobian(qs)

    def fp(res):
        P, J = res
        env = {sym(k): v for k, v in LINK.items()}
        env.update(dict(zip(qs, QPT)))
        return [float(e.evalf(subs=env)) for e in P] + [float(e.evalf(subs=env)) for e in J]

    return go, fp


def _jac_symengine():
    import symengine as se

    cache: dict = {}

    def sym(n):
        return cache.setdefault(n, se.Symbol(n))

    rows = [_dh_rows(i, se.cos, se.sin, sym, se.Integer) for i in range(6)]
    qs = [sym(f"q{i + 1}") for i in range(6)]

    def go():
        T = se.DenseMatrix(rows[0])
        for i in range(1, 6):
            T = T * se.DenseMatrix(rows[i])
        P = se.DenseMatrix([T[r, 3] for r in range(3)])
        return P, P.jacobian(se.DenseMatrix(qs))

    def fp(res):
        P, J = res
        env = {sym(k): v for k, v in LINK.items()}
        env.update(dict(zip(qs, QPT)))
        return [float(P[r].subs(env)) for r in range(3)] + [
            float(J[r, c].subs(env)) for r in range(3) for c in range(6)
        ]

    return go, fp


def _jac_wl() -> WL:
    mats = []
    for i in range(6):
        d, a, ca, sa = DH[i]
        q = f"q{i + 1}"
        dd = d if d != "0" else "0"
        aa = a if a != "0" else "0"
        mats.append(
            f"{{{{Cos[{q}], -Sin[{q}] ({ca}), Sin[{q}] ({sa}), {aa} Cos[{q}]}}, "
            f"{{Sin[{q}], Cos[{q}] ({ca}), -Cos[{q}] ({sa}), {aa} Sin[{q}]}}, "
            f"{{0, {sa}, {ca}, {dd}}}, {{0, 0, 0, 1}}}}"
        )
    rules = ", ".join([f"{k} -> {v}" for k, v in LINK.items()] + [f"q{i + 1} -> {v}" for i, v in enumerate(QPT)])
    return WL(
        setup=f"mats = {{{', '.join(mats)}}}; qs = {{q1, q2, q3, q4, q5, q6}}",
        run="Module[{T = Dot @@ mats, P}, P = T[[1 ;; 3, 4]]; {P, D[P, {qs}]}]",
        fp=f"N[Flatten[res /. {{{rules}}}]]",
    )


JAC = Case(
    id="robot_6r_jacobian",
    task="Forward kinematics + Jacobian of a 6R arm",
    detail="PUMA-560 Denavit–Hartenberg chain with symbolic joint angles and link parameters "
    "(a2, a3, d3, d4, d6): product of six 4×4 transforms, then the 3×6 position Jacobian. "
    "Checked: position and all 18 Jacobian entries against 30-digit mpmath differentiation "
    "of the numeric chain.",
    systems={
        "alkahest": _jac_alkahest,
        "sympy": _jac_sympy,
        "symengine": _jac_symengine,
        "mathematica": _jac_wl(),
    },
    reference=_jac_ref,
    tol=1e-9,
)

# ---------------------------------------------------------------------------
# 4. Compiled evaluation of the 6R Jacobian at 1M configurations
# ---------------------------------------------------------------------------

N_CFG = 1_000_000
J_SAMPLES = [0, 777, 500_000, 999_999]


def _jac_strings() -> list[str]:
    """The 18 Jacobian entries (PUMA-560 numeric link lengths) as strings.

    Derived once with SymPy and cached, so every system compiles the very same
    expressions."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "jac6r.json"
    if path.exists():
        return json.loads(path.read_text())
    import sympy as sp

    cache: dict = {}

    def sym(n):
        if n in LINK:
            return sp.Float(LINK[n])
        return cache.setdefault(n, sp.Symbol(n))

    rows = [_dh_rows(i, sp.cos, sp.sin, sym, sp.Integer) for i in range(6)]
    T = sp.Matrix(rows[0])
    for i in range(1, 6):
        T = T * sp.Matrix(rows[i])
    P = sp.Matrix([T[r, 3] for r in range(3)])
    J = P.jacobian([sym(f"q{i + 1}") for i in range(6)])
    out = [sp.sstr(e, full_prec=True) for e in J]
    path.write_text(json.dumps(out))
    return out


def _cfg_paths():
    return [shared_array_path(f"cfg_q{i + 1}", N_CFG, 200 + i, -3.14159, 3.14159) for i in range(6)]


def _cfg_arrays():
    return [shared_array(f"cfg_q{i + 1}", N_CFG, 200 + i, -3.14159, 3.14159) for i in range(6)]


def _jeval_fp(out):
    import math

    import numpy as np

    a = np.asarray(out, dtype=float)
    if a.shape[0] != 18:
        a = a.reshape(N_CFG, 18).T
    vals = [math.fsum(a.ravel()) / a.size]
    for i in J_SAMPLES:
        vals += [float(a[0, i]), float(a[10, i])]
    return vals


def _jeval_ref():
    import mpmath as mp

    mp.mp.dps = 30

    def go():
        import math

        import numpy as np

        strs = _jac_strings()
        arrs = _cfg_arrays()
        names = [f"q{i + 1}" for i in range(6)]
        fn = [eval("lambda " + ",".join(names) + ": " + s, {"sin": np.sin, "cos": np.cos}) for s in strs]
        tot = math.fsum(math.fsum(np.broadcast_to(f(*arrs), (N_CFG,))) for f in fn)
        vals = [tot / (18 * N_CFG)]
        mfn = [
            eval("lambda " + ",".join(names) + ": " + s, {"sin": mp.sin, "cos": mp.cos}) for s in (strs[0], strs[10])
        ]
        for i in J_SAMPLES:
            x = [mp.mpf(float(a[i])) for a in arrs]
            vals += [float(mfn[0](*x)), float(mfn[1](*x))]
        return vals

    return go, lambda r: r


def _jeval_alkahest():
    import alkahest as ak

    p = ak.ExprPool()
    syms = {f"q{i + 1}": p.symbol(f"q{i + 1}") for i in range(6)}
    es = [ak.parse(s, p, syms) for s in _jac_strings()]
    fn = ak.compile_expr(es, list(syms.values()), expected_evals=N_CFG)
    arrs = _cfg_arrays()

    def go():
        return ak.numpy_eval(fn, *arrs)

    return go, _jeval_fp


def _jeval_sympy():
    import numpy as np
    import sympy as sp

    syms = sp.symbols("q1:7")
    loc = {str(s): s for s in syms}
    es = [sp.sympify(s, locals=loc) for s in _jac_strings()]
    f = sp.lambdify(syms, es, modules="numpy", cse=True)
    arrs = _cfg_arrays()

    def go():
        return f(*arrs)

    def fp(r):
        return _jeval_fp(np.stack([np.broadcast_to(x, (N_CFG,)) for x in r]))

    return go, fp


def _jeval_symengine():
    import numpy as np
    import symengine as se

    syms = se.symbols("q1:7")
    es = [se.sympify(s) for s in _jac_strings()]
    f = se.Lambdify(list(syms), es, backend="llvm", cse=True)
    X = np.ascontiguousarray(np.stack(_cfg_arrays(), axis=1))

    def go():
        return f(X)

    return go, _jeval_fp


def _jeval_wl() -> WL:
    import re

    reads = "; ".join(f'q{i + 1}A = BinaryReadList["{path}", "Real64"]' for i, path in enumerate(_cfg_paths()))
    exprs = []
    for s in _jac_strings():
        t = s.replace("**", "^")
        t = re.sub(r"\b(sin|cos)\(", lambda m: m.group(1).capitalize() + "(", t)
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
        # SymPy prints floats like 0.4318; WL reads them as machine reals.
        exprs.append("".join(out))
    rules = "{" + ", ".join(f"q{i + 1} -> q{i + 1}A" for i in range(6)) + "}"
    samples = ", ".join(f"res[[1, {i + 1}]], res[[11, {i + 1}]]" for i in J_SAMPLES)
    return WL(
        setup=f"{reads}; js = {{{', '.join(exprs)}}}; rules = {rules}",
        run="Map[If[VectorQ[#], #, ConstantArray[#, Length[q1A]]] &, js /. rules]",
        fp=f"Join[{{Mean[Flatten[res]]}}, {{{samples}}}]",
    )


JEVAL = Case(
    id="robot_jacobian_eval_1e6",
    task="Evaluate a 6R Jacobian at 1M configurations",
    detail="The 18 entries of the PUMA-560 position Jacobian (derived once with SymPy, so every "
    "system compiles identical expressions) evaluated at 1,000,000 random joint "
    "configurations. Alkahest: compile_expr (one fused multi-output kernel) + numpy_eval, one "
    "thread; SymPy: lambdify(numpy, cse=True); SymEngine: Lambdify(LLVM, cse=True); "
    "Mathematica: packed-array evaluation. Compilation not timed. Checked: mean and sampled "
    "entries against mpmath.",
    systems={
        "alkahest": _jeval_alkahest,
        "sympy": _jeval_sympy,
        "symengine": _jeval_symengine,
        "mathematica": _jeval_wl(),
    },
    reference=_jeval_ref,
    tol=1e-8,
)

SUITES = [
    Suite(
        id="engineering",
        label="Engineering / robotics",
        blurb="Equations of motion of multibody systems, manipulator kinematics, and the "
        "compiled kinematics evaluated over a million configurations.",
        script=SCRIPT,
        cases=[_eom_case(4), _eom_case(6), JAC, JEVAL],
    )
]
