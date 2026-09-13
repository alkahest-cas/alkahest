"""Silent-error cases for codegen.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import re
from typing import Callable

import alkahest as ak
from contracts import Case, RefusesOr, Returns

from ._shared import POOL, X, _int, _rat

_C_LITERAL = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+\.\d*(?:[eE][-+]?\d+)?")
_MLIR_CONSTANT = re.compile(r"dense<([^>]+)>")
_AGREEMENT_EXPRS: list[tuple[str, Callable[[], ak.Expr]]] = [
    ("rational_2_5", lambda: _rat(2, 5) * X),
    ("rational_neg_7_3", lambda: _rat(-7, 3) + X),
    ("rational_355_113", lambda: _rat(355, 113) * X),
    ("large_integer", lambda: _int(10**30) + X),
    ("large_rational", lambda: POOL.rational(3 * 10**400 + 1, 2 * 10**400) * X),
    ("mixed_poly", lambda: _rat(2, 5) * X**3 + _rat(-7, 3) * X**2 + _rat(1, 7)),
    ("transcendental", lambda: ak.sin(X) * ak.cos(X) + _rat(1, 3) * X),
]
_AGREEMENT_POINTS = (-8.0, -1.5, -0.25, 0.5, 1.0, 2.0, 7.0)


def _c_body(code: str) -> str:
    """The single statement inside an emitted C function, whitespace-normalised."""
    return " ".join(code.splitlines()[1].split())


def _largest_c_literal(code: str) -> float:
    """The largest-magnitude `double` literal in an emitted C function."""
    return max((float(m) for m in _C_LITERAL.findall(code)), key=abs)


def _largest_mlir_constant(mlir: str) -> float:
    """The largest-magnitude `stablehlo.constant` in an emitted MLIR module."""
    return max((float(m) for m in _MLIR_CONSTANT.findall(mlir)), key=abs)


def _mlir_literals_missing_a_decimal_point(mlir: str) -> int:
    """How many emitted constants MLIR's grammar would reject.

    ``float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?`` — the point is
    required, and ``mlir-opt`` answers ``error: expected '>'`` without one.
    """
    return sum(1 for lit in _MLIR_CONSTANT.findall(mlir) if "." not in lit)


def _evaluators_disagreeing_on() -> int:
    """Count the (expression, point) pairs where the four f64 paths differ.

    ``eval_expr`` (registry interpreter), ``evaluate(mode="f64")`` (the
    ``eval::eval_f64`` facade the verification gates call), ``compile_expr``
    and ``numpy_eval`` all run the same arithmetic over the same DAG, so any
    difference means at least one of them is wrong — and the gates and the user
    are then not looking at the same number.  Comparison is exact, not
    tolerance-based: a tolerance is exactly what hides an ulp-level literal bug.
    """
    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - numpy is absent on some CI shards
        np = None

    disagreements = 0
    for _name, build in _AGREEMENT_EXPRS:
        expr = build()
        compiled = ak.compile_expr(expr, [X])
        for point in _AGREEMENT_POINTS:
            values = {
                float(ak.eval_expr(expr, {X: point})),
                float(ak.evaluate(expr, {X: point}, mode="f64").value),
                float(compiled([point])),
            }
            if np is not None:
                values.add(float(ak.numpy_eval(compiled, np.array([point]))[0]))
            if len(values) > 1:
                disagreements += 1
    return disagreements


CASES: list[Case] = [
    Case(
        id="codegen_horner_preserves_coefficients_past_i64",
        subsystem="codegen",
        statement="horner((x+1)^80) is the same polynomial: 2^80 at x = 1",
        op=lambda: float(ak.eval_expr(ak.horner((X + _int(1)) ** _int(80), X), {X: 1.0})),
        contract=Returns(float(2**80), tol=0.0),
        verified_by=(
            "(1+1)^80 = 2^80 = 1208925819614629174706176, exactly representable in binary64. "
            "C(80,40) = 1.075e23 exceeds i64::MAX = 9.22e18, so a coefficient vector taken "
            "as i64 wraps modulo 2^64 in the middle of the row."
        ),
        note=(
            "horner is a *rewrite*; it promises the same polynomial. Going through "
            "coefficients_i64 made that promise false with no error and no flag, and "
            "returned 2.12e20 here."
        ),
    ),
    Case(
        id="codegen_control_horner_small_coefficients",
        subsystem="codegen",
        statement="horner(x²+2x+1) = 16 at x = 3 — the ordinary case is unchanged",
        op=lambda: float(ak.eval_expr(ak.horner(X**2 + _int(2) * X + _int(1), X), {X: 3.0})),
        contract=Returns(16.0, tol=0.0),
        verified_by="(3+1)² = 16. Control for codegen_horner_preserves_coefficients_past_i64.",
    ),
    Case(
        id="codegen_emit_c_preserves_coefficients_past_i64",
        subsystem="codegen",
        statement="emit_c of (2^70+3)·x + 1 keeps the coefficient rather than wrapping it to 3",
        op=lambda: _largest_c_literal(ak.emit_c(_int(2**70 + 3) * X + _int(1), X)),
        contract=Returns(1.1805916207174113e21, tol=0.0),
        verified_by=(
            "float(2**70 + 3) = 1.1805916207174113e21 (Python rounds correctly). "
            "(2**70 + 3) % 2**64 = 3, which is what the emitted C used to contain."
        ),
    ),
    Case(
        id="codegen_emit_c_expr_matches_the_interpreter_on_a_rational",
        subsystem="codegen",
        statement="the C emitted for (2/5)·x carries the same double the interpreter uses",
        op=lambda: _largest_c_literal(ak.emit_c_expr(_rat(2, 5) * X, [X], var_names=["x"])),
        contract=Returns(0.4, tol=0.0),
        verified_by=(
            "float(Fraction(2, 5)) = 0.4. Emitting 0.39999999999999997 instead would make "
            "the compiled artefact disagree with eval_expr by an ulp on the same expression."
        ),
    ),
    Case(
        id="codegen_emit_c_expr_emits_compilable_c_for_a_non_finite_literal",
        subsystem="codegen",
        statement="a non-finite constant emits the <math.h> macro, not the bare token `NaN`",
        op=lambda: _c_body(ak.emit_c_expr(POOL.float(float("inf")) * X, [X], var_names=["x"])),
        contract=Returns("return (x * INFINITY);"),
        verified_by=(
            "C has no identifiers `inf` or `NaN`; INFINITY and NAN are the <math.h> macros, "
            "and the emitted code already documents that <math.h> is required. Checked by "
            "compiling both forms with cc: the bare token is an undeclared identifier."
        ),
        note="emit_c_expr reported success while returning a function that does not compile.",
    ),
    Case(
        id="codegen_stablehlo_preserves_coefficients_past_i64",
        subsystem="codegen",
        statement="to_stablehlo of 2^70·x + 1 does not lower the coefficient to the constant 0",
        op=lambda: _largest_mlir_constant(ak.to_stablehlo(_int(2**70) * X + _int(1), [X])),
        contract=Returns(1.1805916207174113e21, tol=0.0),
        verified_by=(
            "float(2**70) = 1.1805916207174113e21. The emitter read the coefficient with "
            "to_i64().unwrap_or(0), so the module it produced computed x·0 + 1 = 1 — valid "
            "MLIR, no diagnostic, a different function."
        ),
    ),
    Case(
        id="codegen_stablehlo_emits_parseable_float_literals",
        subsystem="codegen",
        statement="every emitted MLIR constant carries the decimal point the grammar requires",
        op=lambda: _mlir_literals_missing_a_decimal_point(
            ak.to_stablehlo(_int(10**30) * X + _rat(1, 10**30), [X])
        ),
        contract=Returns(0),
        verified_by=(
            "MLIR's float-literal grammar requires the point, and mlir-opt 15.0.4 was run "
            "on `dense<1e30>`, `dense<1e16>` and `dense<5e-324>`: each is rejected with "
            "`error: expected '>'`, while `dense<1.0e30>` and `dense<0.4>` parse."
        ),
        note=(
            "to_stablehlo documents its output as valid input to mlir-opt / XLA. Rust's "
            "shortest-round-trip float format drops the point whenever the mantissa is a "
            "single digit, so the module did not parse and the call still reported success."
        ),
    ),
    Case(
        id="codegen_control_stablehlo_small_coefficient",
        subsystem="codegen",
        statement="to_stablehlo of 7·x + 1 carries the coefficient 7",
        op=lambda: _largest_mlir_constant(ak.to_stablehlo(_int(7) * X + _int(1), [X])),
        contract=Returns(7.0, tol=0.0),
        verified_by="7 is exact in binary64. Control for the i64-overflow case above.",
    ),
    Case(
        id="codegen_compiled_fn_agrees_with_the_interpreter_bit_for_bit",
        subsystem="codegen",
        statement="eval_expr, evaluate(f64), compile_expr and numpy_eval give one value, not four",
        op=_evaluators_disagreeing_on,
        contract=Returns(0),
        verified_by=(
            "Not a mathematical claim but a consistency one: the four paths run the same "
            "arithmetic on the same DAG, so any difference means at least one is wrong. "
            "The count is of (expression, point) pairs where the four do not agree exactly."
        ),
        note=(
            "This is the case that catches a literal- or lowering-level divergence "
            "regardless of which side is wrong; the per-value cases above pin down which."
        ),
    ),
    Case(
        id="codegen_compiled_fn_does_not_invent_a_value_at_a_pole",
        subsystem="codegen",
        statement="compile_expr(1/x)([0.0]) must not return a finite number",
        op=lambda: float(ak.compile_expr(_int(1) / X, [X])([0.0])),
        contract=RefusesOr(),
        verified_by="1/0 is undefined; no real number is the value of 1/x at 0.",
        note=(
            "Passes via a *weak* refusal: the compiled path returns ±inf/NaN where eval_expr "
            "raises E-EVAL-009. That asymmetry is documented (CompiledFn has no error "
            "channel) and is the reason this case exists rather than a Raises() one."
        ),
    ),
]
