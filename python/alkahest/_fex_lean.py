"""Lean 4 certificates for finalised FEX solutions at a concrete dimension.

For ``−Δu = f`` (Poisson) and ``Σ cₖ ∂u/∂xₖ = f`` (linear transport) with an
exact closed form ``u(x₀, …, x_{n−1})`` the emitter writes

.. code-block:: lean

    noncomputable def fexU (x0 … : ℝ) : ℝ := u
    theorem fex_pde (x0 … : ℝ) :
        -(deriv (deriv (fun r => fexU r x1 …)) x0 + …) = f

(or ``c₀ * deriv (fun r => fexU r …) x0 + … = f``). The statement uses only
Mathlib's ``deriv`` and the candidate itself — none of Alkahest's printed
derivatives. The proof is mechanical: for each coordinate, a ``HasDerivAt``
term is assembled node by node from the lemma for that node
(``hasDerivAt_id'``, ``hasDerivAt_const``, ``.add``, ``.mul``, ``.pow``,
``.sin``, ``.cos``, ``.exp``), matched to the partial derivative Alkahest
computed with ``convert … using 1`` and closed by ``ring_nf``; ``deriv`` is
rewritten with ``HasDerivAt.deriv`` (through ``funext`` for the inner one of a
second derivative), and the final identity is closed by ``ring_nf``. No
``sorry``.

It is *emitted*, never checked in-process: ``externally_verified`` is
``False``. A candidate outside that grammar, another operator, or a large
dimension yields ``status = "withheld"`` with a stable reason code.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import alkahest as _ak

#: Largest dimension a certificate is emitted for (the proof has one
#: ``HasDerivAt`` block per coordinate, two for a second derivative).
MAX_DIM = 24

WITHHELD_NOT_VERIFIED = "withheld_not_exactly_verified"
WITHHELD_OPERATOR = "withheld_operator"
WITHHELD_NODE = "withheld_unsupported_node"
WITHHELD_TOO_LARGE = "withheld_dimension_too_large"
WITHHELD_INEXACT = "withheld_inexact_parameters"

HEADER = (
    "import Mathlib.Tactic\n"
    "import Mathlib.Analysis.Calculus.Deriv.Basic\n"
    "import Mathlib.Analysis.Calculus.Deriv.Add\n"
    "import Mathlib.Analysis.Calculus.Deriv.Pow\n"
    "import Mathlib.Analysis.Calculus.Deriv.Mul\n"
    "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv\n"
    "import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n"
    "\n"
    "set_option linter.unreachableTactic false\n"
    "set_option linter.unusedTactic false\n\n"
    "open Real\n\n"
)


class _Unsupported(Exception):
    pass


def _num(e: Any) -> Fraction | None:
    nd = e.node()
    if nd[0] == "integer":
        return Fraction(int(nd[1]))
    if nd[0] == "rational":
        return Fraction(int(nd[1]), int(nd[2]))
    return None


def _kids(e: Any) -> list:
    from ._fex_finalize import _children

    return _children(e)


def _q(q: Fraction) -> str:
    if q.denominator == 1:
        return f"({q.numerator} : ℝ)"
    return f"(({q.numerator} : ℝ) / {q.denominator})"


def lean_expr(e: Any, names: dict) -> str:
    """Print ``e`` as a Lean real expression (``names`` maps symbols to Lean
    identifiers / terms)."""
    q = _num(e)
    if q is not None:
        return _q(q)
    nd = e.node()
    k = nd[0]
    if k == "symbol":
        if e in names:
            return names[e]
        if nd[1] == "pi":
            return "Real.pi"
        raise _Unsupported(f"symbol {nd[1]}")
    if k == "add":
        return "(" + " + ".join(lean_expr(c, names) for c in _kids(e)) + ")"
    if k == "mul":
        return "(" + " * ".join(lean_expr(c, names) for c in _kids(e)) + ")"
    if k == "pow":
        b, x = _kids(e)
        n = _num(x)
        if n is None or n.denominator != 1 or n < 0:
            raise _Unsupported("non-natural exponent")
        return f"({lean_expr(b, names)} ^ {n.numerator})"
    if k == "func":
        f = nd[1]
        args = _kids(e)
        if f in ("sin", "cos", "exp") and len(args) == 1:
            if f == "exp" and _num(args[0]) == 1:
                return "(Real.exp 1)"
            return f"(Real.{f} {lean_expr(args[0], names)})"
        if f == "sqrt" and len(args) == 1 and _num(args[0]) is not None:
            return f"(Real.sqrt {lean_expr(args[0], names)})"
        raise _Unsupported(f"function {f}")
    raise _Unsupported(k)


def _mentions(e: Any, s: Any) -> bool:
    from ._fex_finalize import _symbols_in

    return s in _symbols_in(e)


def hasderiv_term(e: Any, s: Any, names: dict) -> str:
    """A Lean term ``HasDerivAt (fun s => e) (…) s`` built node by node.
    The variable is printed as ``names[s]``."""
    v = names[s]
    if not _mentions(e, s):
        return f"(hasDerivAt_const {v} {lean_expr(e, names)})"
    nd = e.node()
    k = nd[0]
    if k == "symbol":
        return f"(hasDerivAt_id' {v})"
    if k == "add":
        kids = _kids(e)
        acc = hasderiv_term(kids[0], s, names)
        for c in kids[1:]:
            acc = f"({acc}.add {hasderiv_term(c, s, names)})"
        return acc
    if k == "mul":
        # Left fold over the factors; constant factors use const_mul /
        # mul_const so the derivative has no 0 * … terms.
        kids = _kids(e)
        acc: str | None = None  # HasDerivAt term of the product so far
        const_prefix: list[str] = []  # leading factors free of s
        for c in kids:
            if not _mentions(c, s):
                if acc is None:
                    const_prefix.append(lean_expr(c, names))
                else:
                    acc = f"({acc}.mul_const {lean_expr(c, names)})"
                continue
            t = hasderiv_term(c, s, names)
            acc = t if acc is None else f"({acc}.mul {t})"
        if const_prefix:
            acc = f"({acc}.const_mul ({' * '.join(const_prefix)}))"
        return acc
    if k == "pow":
        b, x = _kids(e)
        n = _num(x)
        if n is None or n.denominator != 1 or n < 0:
            raise _Unsupported("non-natural exponent")
        return f"({hasderiv_term(b, s, names)}.pow {n.numerator})"
    if k == "func" and nd[1] in ("sin", "cos", "exp"):
        (a,) = _kids(e)
        return f"({hasderiv_term(a, s, names)}.{nd[1]})"
    raise _Unsupported(k)


_CLOSE = (
    "all_goals first | rfl | (ring_nf; done) | (funext y; ring_nf; done) "
    "| (push_cast; ring_nf; done) | (norm_num; ring_nf; done) | (simp; done)"
)


def concrete_certificate(kind, expr, coords, time, src, concrete, exact) -> dict[str, Any]:
    """The Lean certificate dict for a finalised concrete-``d`` solution."""
    out: dict[str, Any] = {"status": "withheld", "externally_verified": False}
    if not exact:
        out["reason"] = WITHHELD_INEXACT
        return out
    if not concrete or not concrete.get("interior_verified"):
        out["reason"] = WITHHELD_NOT_VERIFIED
        return out
    if kind not in ("poisson", "transport") or src is None:
        out["reason"] = WITHHELD_OPERATOR
        out["detail"] = "certificates cover -Δu = f and Σ c_k ∂u/∂x_k = f"
        return out
    if len(coords) > MAX_DIM:
        out["reason"] = WITHHELD_TOO_LARGE
        out["detail"] = f"dimension {len(coords)} > {MAX_DIM}"
        return out
    try:
        src_text = _emit(kind, expr, coords, time, src)
    except _Unsupported as exc:
        out["reason"] = WITHHELD_NODE
        out["detail"] = str(exc)
        return out
    out.update(
        {
            "status": "emitted",
            "reason": "emitted",
            "theorem": "fex_pde",
            "lean": src_text,
            "covers": "the PDE residual at every point of ℝ^d (boundary conditions are "
            "not part of the theorem)",
        }
    )
    return out


def _emit(kind, expr, coords, time, src) -> str:
    pool = coords[0]._pool
    n = len(coords)
    names = {c: f"x{k}" for k, c in enumerate(coords)}
    binders = " ".join(f"x{k}" for k in range(n))
    body = lean_expr(expr, names)
    lines = [HEADER, f"noncomputable def fexU ({binders} : ℝ) : ℝ :=\n  {body}\n"]
    s_sym = pool.symbol("__fex_lean_s")
    rhs = src.get("rhs")
    rhs_text = lean_expr(rhs, names) if rhs is not None else "0"

    def slice_fn(k: int) -> str:
        args = " ".join("r" if j == k else f"x{j}" for j in range(n))
        return f"(fun r => fexU {args})"

    proof: list[str] = []
    terms: list[str] = []
    for k, x in enumerate(coords):
        if kind == "poisson" and k == time:
            continue
        coeff = None
        if kind == "transport":
            c = src["coeffs"][k]
            if _ak.is_zero(c) is True:
                continue
            coeff = lean_expr(c, names)
        # First partial: HasDerivAt of the slice at every s.
        e_s = _ak.subs(expr, {x: s_sym})
        nm = dict(names)
        nm[s_sym] = "s"
        d1 = _ak.simplify(_ak.diff(e_s, s_sym).value).value
        proof.append(
            f"  have h{k} : ∀ s : ℝ, HasDerivAt {slice_fn(k)} {lean_expr(d1, nm)} s := by\n"
            f"    intro s\n"
            f"    have h := {hasderiv_term(e_s, s_sym, nm)}\n"
            f"    unfold fexU\n"
            f"    convert h using 1\n"
            f"    {_CLOSE}"
        )
        if kind == "poisson":
            proof.append(
                f"  have e{k} : deriv {slice_fn(k)} = fun s => {lean_expr(d1, nm)} :=\n"
                f"    funext fun s => (h{k} s).deriv"
            )
            d2 = _ak.simplify(_ak.diff(d1, s_sym).value).value
            nm2 = dict(names)
            nm2[s_sym] = f"x{k}"
            proof.append(
                f"  have k{k} : HasDerivAt (fun s : ℝ => {lean_expr(d1, nm)}) "
                f"{lean_expr(d2, nm2)} x{k} := by\n"
                f"    have h := {hasderiv_term(d1, s_sym, nm2)}\n"
                f"    convert h using 1\n"
                f"    {_CLOSE}"
            )
            terms.append(f"deriv (deriv {slice_fn(k)}) x{k}")
        else:
            terms.append(f"{coeff} * deriv {slice_fn(k)} x{k}")
    if kind == "poisson":
        stmt = f"-({' + '.join(terms)}) = {rhs_text}"
        rws = ", ".join(f"e{k}, k{k}.deriv" for k in range(n) if k != time)
    else:
        stmt = f"{' + '.join(terms)} = {rhs_text}"
        rws = ", ".join(
            f"(h{k} x{k}).deriv" for k in range(n) if _ak.is_zero(src["coeffs"][k]) is not True
        )
    lines.append(f"theorem fex_pde ({binders} : ℝ) :\n    {stmt} := by")
    lines.extend(proof)
    lines.append(f"  rw [{rws}]")
    lines.append(f"  {_CLOSE}")
    return "\n".join(lines) + "\n"
