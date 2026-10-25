"""Cancellation limits at +∞ against SymPy (benchmark finding B1, 2026-10-25).

* ``(exp(1/x − e^{−x}) − exp(1/x))/e^{−x}`` — the textbook Gruntz example —
  came back as ``−∞`` from the 3.11.0 wheel (true ``−1``). Fixed on main by the
  Gruntz leading-term work (#520); pinned here.
* ``x·(log(2/x) − log(1/x))`` (``= x·log 2``) came back as ``0`` (true ``+∞``):
  after ``x ↦ 1/t`` L'Hôpital read ``(log(2t) − log t)/t`` at ``t = 0`` as
  ``0/0``, because the substituted numerator ``log 0 − log 0`` simplifies to
  ``0`` as ``a − a``.

The randomised part builds the two shapes the finding belongs to —
``(exp(f + g) − exp(f))/h`` and ``(F(a + ε) − F(a))/ε`` with ``g, ε → 0``
exponentially or polynomially — and checks every answer against SymPy. A
refusal is allowed; a different value is not.
"""

from __future__ import annotations

import random

import alkahest as ak
import pytest

sympy = pytest.importorskip("sympy")

TIMEOUT = 600

PINNED = [
    ("(exp(1/x - exp(-x)) - exp(1/x))/exp(-x)", "-1"),
    ("exp(x)*(exp(1/x - exp(-x)) - exp(1/x))", "-1"),
    ("(exp(-exp(-x)) - 1)/exp(-x)", "-1"),
    ("(exp(1/x + exp(-x)/x) - exp(1/x))/(exp(-x)/x)", "1"),
    ("(exp(1/x**2 - 1/x) - exp(1/x**2))/(1/x)", "-1"),
    ("x*(log(2/x) - log(1/x))", "oo"),
    ("(log(1/x + 1/x) - log(1/x))/(1/x)", "oo"),
    ("x**2*(log(3/x) - log(1/x))", "oo"),
    ("(log(1/x + 1/x**2) - log(1/x))/(1/x)", "1"),
]

F_PARTS = ["1/x", "0", "2", "1/x**2", "-1/x", "1/(x+1)", "log(x)/x"]
G_PARTS = ["-exp(-x)", "exp(-2*x)", "-1/x", "1/x**2", "-exp(-x**2)", "3*exp(-x)", "exp(-x)/x"]
H_PARTS = ["exp(-x)", "exp(-2*x)", "1/x", "1/x**2", "exp(-x**2)", "exp(-x)/x", "x*exp(-x)"]
HEADS = ["exp", "log", "sin", "cos", "X**3", "1/X", "log(1+X)", "exp(1/X)"]
POINTS = ["1/x", "1 + 1/x", "2"]
EPSILONS = ["exp(-x)", "1/x", "exp(-x**2)", "1/x**2", "x*exp(-x)"]


def _apply(head: str, arg: str) -> str:
    return head.replace("X", f"({arg})") if "X" in head else f"{head}({arg})"


def _random_cases(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            f, g, h = rng.choice(F_PARTS), rng.choice(G_PARTS), rng.choice(H_PARTS)
            out.append(f"(exp({f} + {g}) - exp({f}))/({h})")
        else:
            head, a, e = rng.choice(HEADS), rng.choice(POINTS), rng.choice(EPSILONS)
            out.append(f"({_apply(head, f'{a} + {e}')} - {_apply(head, a)})/({e})")
    return out


def _check(src: str, want=None):
    """Return ``None`` (agree or refused) or a description of a wrong answer."""
    x_sp = sympy.Symbol("x", positive=True)
    if want is None:
        want = sympy.limit(sympy.sympify(src, locals={"x": x_sp}), x_sp, sympy.oo)
    else:
        want = sympy.sympify(want)
    p = ak.ExprPool()
    x = p.symbol("x")
    try:
        got_ak = ak.limit(ak.parse(src.replace("**", "^"), p, {"x": x}), x, p.pos_infinity()).value
    except ak.AlkahestError:
        return None
    got = sympy.sympify(str(got_ak).replace("∞", "oo").replace("π", "pi").replace("^", "**"))
    if got == want:
        return None
    if any(v.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan) for v in (got, want)):
        return f"{src}: alkahest {got}, sympy {want}"
    if abs(complex(sympy.N(got - want, 30))) < 1e-9:
        return None
    return f"{src}: alkahest {got}, sympy {want}"


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(("src", "want"), PINNED)
def test_pinned_cancellations(src, want):
    p = ak.ExprPool()
    x = p.symbol("x")
    got = ak.limit(ak.parse(src.replace("**", "^"), p, {"x": x}), x, p.pos_infinity()).value
    if want == "oo":
        assert got == p.pos_infinity(), f"{src} -> {got}"
    else:
        assert got == p.integer(int(want)), f"{src} -> {got}"


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("seed", range(8))
def test_random_cancellations_against_sympy(seed):
    wrong = [w for w in (_check(src) for src in _random_cases(seed, 15)) if w]
    assert not wrong, "\n".join(wrong)
