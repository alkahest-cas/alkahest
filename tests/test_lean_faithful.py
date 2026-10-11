"""``to_lean`` states the computed result, binds every variable, never returns ``""``.

W9 (report 10-9): ``to_lean(resultant(x**2 - 2, x**3 - 3, x))`` emitted

    example : x^2 + (-1)*2 = 1 := by ring_nf; simp

— the resultant log records ``p ↦ Res(p, q)`` (an algorithm's input and
output), and the exporter printed that pair as an equality, with ``x`` left
unbound. ``to_lean`` also returned ``""`` for a ``Piecewise`` result and raised a
bare ``TypeError`` on a ``ZeilbergerCertificate``.

These tests check the *statement*, independently of Lean: every emitted
``example`` is parsed back into Python and checked numerically against the
mathematics it claims (equalities at random points, derivatives by central
differences, interval integrals by quadrature, determinants with numpy, finite
sums term by term), its final right-hand side is checked against the value the
API returned, and every ``(v : ℝ)`` in a goal must be bound. When a Lean 4
toolchain with the Mathlib project in ``lean/`` is available the emitted files
are also compiled with ``-DautoImplicit=false`` (the CI ``Verify Lean 4
proofs`` job compiles the same shapes through ``tests/lean_corpus.py``).
"""

from __future__ import annotations

import math
import os
import random
import re
import shutil
import subprocess

import alkahest as ak
import pytest

# ---------------------------------------------------------------------------
# Lean term → Python
# ---------------------------------------------------------------------------

_FUNCS = {
    "Real.sinh": "math.sinh",
    "Real.cosh": "math.cosh",
    "Real.tanh": "math.tanh",
    "Real.sin": "math.sin",
    "Real.cos": "math.cos",
    "Real.tan": "math.tan",
    "Real.exp": "math.exp",
    "Real.log": "math.log",
    "Real.sqrt": "math.sqrt",
    "Real.arctan": "math.atan",
    "Real.arcsin": "math.asin",
    "Real.Gamma": "math.gamma",
    "Nat.factorial": "math.factorial",
}


def _match_paren(s: str, i: int) -> int:
    """Index of the bracket closing the one opened at ``s[i]``."""
    depth = 0
    for j in range(i, len(s)):
        depth += s[j] == "("
        depth -= s[j] == ")"
        if depth == 0:
            return j
    raise ValueError(f"unbalanced: {s}")


def _apply_lambdas(s: str) -> str:
    """``(fun t : ℕ => B) a`` → ``(lambda t: B)(a)`` (application by juxtaposition)."""
    while (i := s.find("(fun t : ℕ => ")) != -1:
        end = _match_paren(s, i)
        body = s[i + len("(fun t : ℕ => ") : end]
        rest = s[end + 1 :].lstrip()
        if rest.startswith("("):
            arg_end = _match_paren(rest, 0) + 1
        else:
            arg_end = re.match(r"[\w.]+", rest).end()
        s = s[:i] + f"(lambda t: {body})({rest[:arg_end]})" + rest[arg_end:]
    return s


def lean_to_py(term: str) -> str:
    """Translate an emitted Lean real-valued term into a Python expression."""
    s = _apply_lambdas(term)
    s = re.sub(r"\s*:\s*[ℝℕ]\s*\)", ")", s)
    for lean, py in _FUNCS.items():
        s = s.replace(lean, py)
    s = s.replace("rexp", "math.exp")
    s = s.replace("π", "math.pi")
    # `(b)⁻¹ ^ n` is `((b)⁻¹) ^ n`; a method call binds tighter than `**`.
    s = s.replace(")⁻¹", ").__pow__(-1)")
    s = s.replace("^", "**")
    return s


def py_eval(term: str, env: dict[str, float]) -> float:
    ns = {"math": math, "pi": math.pi}
    ns.update(env)
    return float(eval(lean_to_py(term), ns))  # test-only: evaluates emitted text


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# Statement parsing
# ---------------------------------------------------------------------------


def split_top(s: str, sep: str) -> list[str]:
    """Split ``s`` on ``sep`` at bracket depth 0."""
    parts, depth, i, start = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(s[start:])
    return parts


def statements(src: str) -> list[tuple[str, str]]:
    """``(binders, goal)`` for every ``example`` in ``src``."""
    out = []
    for m in re.finditer(r"^(?:example|theorem \w+)(.*?):=", src, flags=re.M | re.S):
        header = " " + " ".join(m.group(1).split()) + " "
        # Binders are leading `( … )` groups; the goal follows the first
        # depth-0 ` : `.
        head, goal = split_top(header, " : ")[0], " : ".join(split_top(header, " : ")[1:])
        out.append((head.strip(), " ".join(goal.split())))
    return out


def binder_names(binders: str) -> tuple[dict[str, str], list[str]]:
    """``({name: type}, [hypothesis texts])`` of an ``example`` binder list."""
    names, hyps = {}, []
    for group in re.findall(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", binders):
        lhs, _, typ = group.partition(" : ")
        typ = typ.strip()
        for n in lhs.split():
            if typ in ("ℝ", "ℕ"):
                names[n] = typ
            else:
                hyps.append(typ)
    return names, hyps


def unbound_variables(binders: str, goal: str) -> set[str]:
    bound = set(binder_names(binders)[0])
    bound |= set(re.findall(r"fun \((\w+) : [ℝℕ]\)", goal))
    bound |= set(re.findall(r"fun (\w+) : ℕ", goal))
    bound |= set(re.findall(r"[∫∑∏∀] (\w+) [∈i]", goal))
    used = set(re.findall(r"\((\w+) : ℝ\)", goal))
    return {u for u in used if not re.fullmatch(r"-?\d+", u)} - bound


def sample_env(binders: str, rng: random.Random) -> dict[str, float]:
    names, hyps = binder_names(binders)
    env = {}
    for n, typ in names.items():
        env[n] = rng.randint(1, 6) if typ == "ℕ" else rng.uniform(0.3, 0.9)
    # Hypotheses the emitter writes are positivity / nonvanishing / |x| < 1,
    # all satisfied by values in (0.3, 0.9) away from cos / sin zeros.
    for h in hyps:
        assert re.search(r"0 < |≠ 0|-1 < ", h), f"unexpected hypothesis {h!r}"
    return env


# ---------------------------------------------------------------------------
# Numeric check of one goal
# ---------------------------------------------------------------------------


def check_goal(binders: str, goal: str, rng: random.Random) -> str | None:
    """Check ``goal`` numerically; return the RHS term (for value checks)."""
    if goal.startswith("Filter.Tendsto"):
        return None  # limit statements: checked against the API value below
    if len(split_top(goal, " ≤ ")) == 2:
        lo, hi = split_top(goal, " ≤ ")
        for _ in range(4):
            env = sample_env(binders, rng)
            assert py_eval(lo, env) <= py_eval(hi, env) + 1e-9, f"false inequality: {goal}"
        return None
    lhs_rhs = split_top(goal, " = ")
    assert len(lhs_rhs) == 2, f"not an equation: {goal}"
    lhs, rhs = lhs_rhs
    for _ in range(4):
        env = sample_env(binders, rng)
        m = re.fullmatch(r"deriv \(fun \((\w+) : ℝ\) => (.*)\) (\w+)", lhs)
        if m:
            var, body, point = m.groups()
            x0 = env[point]
            h = 1e-5
            fp = py_eval(body, {**env, var: x0 + h})
            fm = py_eval(body, {**env, var: x0 - h})
            got = (fp - fm) / (2 * h)
            want = py_eval(rhs, env)
            assert close(got, want, 1e-4), f"deriv mismatch {got} vs {want}: {goal}"
            continue
        m = re.fullmatch(r"∫ (\w+) in \((.*)\)\.\.\((.*)\), (.*)", lhs)
        if m:
            var, lo, hi, body = m.groups()
            a, b = py_eval(lo, env), py_eval(hi, env)
            n = 2000
            hstep = (b - a) / n
            total = sum(
                (1 if i in (0, n) else 4 if i % 2 else 2)
                * py_eval(body, {**env, var: a + i * hstep})
                for i in range(n + 1)
            )
            got = total * hstep / 3
            assert close(got, py_eval(rhs, env), 1e-6), f"integral mismatch: {goal}"
            continue
        m = re.fullmatch(r"Matrix\.det !!\[(.*)\]", lhs)
        if m:
            np = pytest.importorskip("numpy")
            rows = [split_top(r, ", ") for r in split_top(m.group(1), "; ")]
            mat = np.array([[py_eval(e, env) for e in row] for row in rows], dtype=float)
            assert close(float(np.linalg.det(mat)), py_eval(rhs, env), 1e-8), goal
            continue
        m = re.fullmatch(r"∑ (\w+) ∈ (Icc|range) (?:\((\d+) : ℕ\) )?(\w+), (.*)", lhs)
        if m:
            k, kind, lo, hi, body = m.groups()
            top = int(env[hi]) if hi in env else int(hi)
            idx = range(int(lo), top + 1) if kind == "Icc" else range(top)
            got = sum(py_eval(body, {**env, k: i}) for i in idx)
            assert close(got, py_eval(rhs, env)), f"sum mismatch: {goal}"
            continue
        if lhs.startswith("∏") or "(fun t : ℕ" in lhs:
            return None  # factorial product / Gosper Δ-step: checked by Lean
        got, want = py_eval(lhs, env), py_eval(rhs, env)
        assert close(got, want), f"false equality {got} != {want}: {goal}"
    return rhs


def value_of(expr: ak.Expr, pool: ak.ExprPool, env: dict[str, float]) -> float:
    bindings = {pool.symbol(n): float(v) for n, v in env.items()}
    return float(ak.evaluate(expr, bindings, mode="f64").value)


def audit(src: str, result=None, pool=None, final_rhs_is_value: bool = True) -> list[str]:
    """Run every structural + numeric check over one emitted certificate."""
    assert src, "to_lean must never return an empty string"
    for token in ("sorry", "admit", "axiom"):
        assert token not in src
    stmts = statements(src)
    assert stmts, f"no example in:\n{src}"
    rng = random.Random(9)
    rhss = []
    for binders, goal in stmts:
        unbound = unbound_variables(binders, goal)
        assert not unbound, f"unbound {unbound} in: example {binders} : {goal}"
        rhss.append((binders, check_goal(binders, goal, rng)))
    if result is not None and final_rhs_is_value:
        binders, rhs = rhss[-1]
        if rhs is not None:
            env = sample_env(binders, rng)
            assert close(py_eval(rhs, env), value_of(result.value, pool, env)), (
                f"statement RHS {rhs} is not the returned value {result.value}"
            )
    return [g for _, g in stmts]


# ---------------------------------------------------------------------------
# W9 repro
# ---------------------------------------------------------------------------


def test_resultant_certificate_states_the_sylvester_determinant():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.resultant(x**2 - 2, x**3 - 3, x)
    src = ak.to_lean(r)
    goals = audit(src, r, pool)
    assert goals[0].startswith("Matrix.det !![")
    # The false restatement `p = Res(p, q)` must be gone.
    assert "(x : ℝ)) ^ (2 : ℕ) + ((2 : ℝ) * (-1 : ℝ))) = (1 : ℝ)" not in src
    assert r.certificate == src


RESULTANT_CASES = [
    ("x**2 - 2", "x**3 - 3", "x"),
    ("x**2 - 5*x + 6", "x - 2", "x"),
    ("x**2 + y**2 - 1", "y - x", "y"),
    ("x**2 + y*x - 1", "x**3 - y", "x"),
    ("-2*x**2 + 3", "3*x - 1", "x"),
    ("x - 3", "5", "x"),
]


@pytest.mark.parametrize(("ps", "qs", "vs"), RESULTANT_CASES)
def test_resultant_certificate_is_sylvester_of_inputs(ps, qs, vs):
    """The matrix is the Sylvester matrix of (p, q) and its det is the value."""
    sympy = pytest.importorskip("sympy")
    from sympy.polys.subresultants_qq_zz import sylvester

    pool = ak.ExprPool()
    syms = {n: pool.symbol(n) for n in ("x", "y")}
    p = ak.parse(ps.replace("**", "^"), pool, syms)
    q = ak.parse(qs.replace("**", "^"), pool, syms)
    r = ak.resultant(p, q, syms[vs])
    src = ak.to_lean(r)
    goals = audit(src, r, pool)
    sx, sy = sympy.symbols("x y")
    sp, sq, sv = sympy.sympify(ps), sympy.sympify(qs), sympy.Symbol(vs)
    expected = sylvester(sympy.Poly(sp, sv), sympy.Poly(sq, sv), sv, 1)
    rows = [
        split_top(row, ", ")
        for row in split_top(re.fullmatch(r"Matrix\.det !!\[(.*)\] = .*", goals[0]).group(1), "; ")
    ]
    assert len(rows) == expected.rows
    env = {"x": 0.61, "y": 0.37}
    for i, row in enumerate(rows):
        for j, entry in enumerate(row):
            want = float(sympy.sympify(expected[i, j]).subs({sx: env["x"], sy: env["y"]}))
            assert close(py_eval(entry, env), want), (i, j, entry, expected[i, j])
    oracle = sympy.resultant(sp, sq, sv)
    assert close(value_of(r.value, pool, env), float(oracle.subs({sx: env["x"], sy: env["y"]})))


def test_resultant_too_large_refuses_with_code():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.resultant(x**3 - 2, x**3 + x + 1, x)  # 6×6 Sylvester matrix
    assert r.certificate is None
    with pytest.raises(ak.CertificateUnavailableError) as info:
        ak.to_lean(r)
    assert info.value.code == "E-CERT-001"
    assert r.certificate_status["reason"] == "withheld_resultant_shape"


def test_piecewise_and_zeilberger_refuse_with_code():
    pool = ak.ExprPool()
    x, n, k = pool.symbol("x"), pool.symbol("n"), pool.symbol("k")
    pw = ak.simplify(ak.piecewise([(pool.gt(x, pool.integer(0)), x)], -x))
    with pytest.raises(ak.CertificateUnavailableError) as info:
        ak.to_lean(pw)
    assert info.value.code == "E-CERT-001"

    one = pool.integer(1)
    g = ak.gamma
    binom = g(n + one) / (g(k + one) * g(n - k + one))
    z = ak.zeilberger(binom**2, n, k)
    with pytest.raises(ak.CertificateUnavailableError) as info:
        ak.to_lean(z)
    assert info.value.code == "E-CERT-001"
    assert "ZeilbergerCertificate" in str(info.value)


def test_non_alkahest_object_is_still_a_type_error():
    with pytest.raises(TypeError):
        ak.to_lean(42)


# ---------------------------------------------------------------------------
# Every dispatch kind
# ---------------------------------------------------------------------------


def _kinds():
    pool = ak.ExprPool()
    x, y, k, n = (pool.symbol(s) for s in ("x", "y", "k", "n"))
    zero, one = pool.integer(0), pool.integer(1)
    pos = ak.Assumptions(pool)
    pos.refine(pool.gt(x, zero))
    pos.refine(pool.gt(y, zero))
    return pool, [
        # (label, result-or-expr, compare-final-RHS-with-.value)
        ("expr_add_zero", x * y + zero, False),
        ("simplify_collect", ak.simplify(x * 2 + x * 3), True),
        ("simplify_trig", ak.simplify_trig(ak.sin(x) ** 2 + ak.cos(x) ** 2), True),
        ("simplify_log_exp", ak.simplify_log_exp(ak.log(ak.exp(x))), True),
        ("assumptions_log_product", pos.simplify(ak.log(x * y)), True),
        ("diff_param", ak.diff(y * ak.sin(x), x), False),
        ("diff_poly", ak.diff(x**3 + 2 * x, x), False),
        ("integrate_sin", ak.integrate(ak.sin(x), x), False),
        ("integrate_log", ak.integrate(ak.log(x), x), False),
        ("definite_cos", ak.integrate(ak.cos(x), x, zero, one), True),
        ("definite_cos_param", ak.integrate(ak.cos(x), x, zero, y), True),
        ("limit_sinc", ak.limit(ak.sin(x) / x, x, zero), False),
        ("sum_definite", ak.sum_definite(k, k, one, n), True),
        ("sum_indefinite", ak.sum_indefinite(k, k), False),
        ("product_factorial", ak.product_definite(k, k, one, n), False),
        ("resultant_param", ak.resultant(x**2 + y**2 - 1, y - x, y), True),
        ("sos", ak.sos_decompose(x**2 - 2 * x + 2, [x]), False),
    ]


_POOL, _KINDS = _kinds()


@pytest.mark.parametrize(("label", "obj", "final_is_value"), _KINDS, ids=[k[0] for k in _KINDS])
def test_every_dispatch_kind_is_faithful_and_bound(label, obj, final_is_value):
    src = ak.to_lean(obj)
    result = obj if isinstance(obj, ak.DerivedResult) else None
    if isinstance(obj, ak.DerivedResult):
        assert src == obj.certificate, f"{label}: to_lean must equal .certificate"
    audit(src, result, _POOL, final_rhs_is_value=final_is_value)


def test_limit_statement_names_the_returned_limit():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.limit(ak.sin(x) / x, x, pool.integer(0))
    goal = statements(ak.to_lean(r))[0][1]
    m = re.search(r"\(nhds (.*)\)$", goal)
    assert m
    assert close(py_eval(m.group(1), {}), value_of(r.value, pool, {}))


# ---------------------------------------------------------------------------
# The rewrite-log gate (Rust side): a step whose sides differ never certifies
# ---------------------------------------------------------------------------


def test_withheld_results_raise_never_empty():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    for obj in (
        ak.simplify(x**2),  # nothing derived
        ak.integrate(x * ak.exp(x**2), x),  # outside the FTC fragment
        ak.limit(ak.sin(pool.symbol("y") * x) / x, x, pool.integer(0)),
    ):
        assert obj.certificate is None
        with pytest.raises(ak.CertificateUnavailableError):
            ak.to_lean(obj)


# ---------------------------------------------------------------------------
# Optional: compile with Lean when the toolchain + Mathlib project are present
# ---------------------------------------------------------------------------

# The Lake project CI verifies with; override to point at a checkout whose
# Mathlib is already built (e.g. when running from a worktree).
_LEAN_DIR = os.environ.get(
    "ALKAHEST_LEAN_PROJECT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lean"),
)


def _lake():
    lake = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
    built = os.path.isdir(os.path.join(_LEAN_DIR, ".lake", "packages", "mathlib", ".lake", "build"))
    return lake if os.path.exists(lake) and built else None


@pytest.mark.skipif(_lake() is None, reason="Lean toolchain / built Mathlib not available")
def test_emitted_statements_compile_without_auto_implicit(tmp_path):
    """Compile every dispatch kind's certificate in one file (Mathlib loads once)."""
    srcs = [ak.to_lean(obj) for _, obj, _ in _KINDS]
    pool = ak.ExprPool()
    x = pool.symbol("x")
    srcs.append(ak.to_lean(ak.resultant(x**2 - 2, x**3 - 3, x)))
    imports, bodies = [], []
    for src in srcs:
        body = []
        for line in src.splitlines():
            if line.startswith("import "):
                if line not in imports:
                    imports.append(line)
            else:
                body.append(line)
        bodies.append("section\n" + "\n".join(body) + "\nend\n")
    path = tmp_path / "all_kinds.lean"
    path.write_text("\n".join(imports) + "\n\n" + "\n".join(bodies))
    proc = subprocess.run(
        [_lake(), "env", "lean", "-DwarningAsError=true", "-DautoImplicit=false", str(path)],
        cwd=_LEAN_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"{proc.stdout}{proc.stderr}"
