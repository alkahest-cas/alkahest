"""Fixtures the subsystem modules share.

The expression pool, the common symbols, and every helper or constant
that more than one subsystem needs.  Expressions are immutable and
pool-scoped, so one ``POOL`` for the whole corpus keeps case
construction cheap and interning consistent.  A helper used by exactly
one subsystem lives beside that subsystem's cases instead.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import alkahest as ak

POOL = ak.ExprPool()

X = POOL.symbol("x")

#: Second variable, for the two-variable `decide` cases.
Y = POOL.symbol("y")

N = POOL.symbol("n")


def _int(v: int) -> ak.Expr:
    return POOL.integer(v)


def _rat(a: int, b: int) -> ak.Expr:
    return POOL.rational(a, b)


def _num(value: Any) -> float:
    """Reduce an Expr / DerivedResult / number to a float."""
    if isinstance(value, ak.DerivedResult):
        value = value.value
    if isinstance(value, (int, float)):
        return float(value)
    return float(ak.eval_expr(value, {}))


PI = POOL.symbol("pi")

# --- Transcendental rank traps -------------------------------------------
# ``exp(a)²`` and ``exp(2a)`` are the same function written two ways.  Any
# elimination that cannot see that will "clear" a column it has not cleared.
_A = POOL.symbol("a")

#: ``[[a, 1], [0, b]]``: diagonalisable for ``a != b`` and defective at ``a = b``,
#: where ``e^A`` is the confluent form and the generic formula divides by
#: ``a − b``.  Which branch holds is not decidable from the matrix alone.
_B = POOL.symbol("b")

_E = math.e

CALCULUS = "first-course calculus fact, re-derived by hand"


def _survives_a_panic(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap *fn* so a Rust panic fails this case instead of killing the run.

    PyO3 turns an escaping Rust panic into ``pyo3_runtime.PanicException``,
    which inherits ``BaseException``.  That is the whole reason the class
    matters — a loop's ``except Exception`` does not catch it — but it also
    means an unwrapped op would take the gate process down with it and no case
    would be reported at all.  Re-raising as ``RuntimeError`` keeps the failure
    (scored ``no_answer``: neither an answer nor a refusal) while leaving the
    rest of the corpus scoreable.
    """

    def op() -> Any:
        try:
            return fn()
        except Exception:
            raise
        except BaseException as exc:  # PanicException is a BaseException — the point
            raise RuntimeError(
                f"escaping Rust panic: {type(exc).__module__}.{type(exc).__name__}: {exc}"
            ) from exc

    return op


#: Third variable, for the three-variable monomial-ideal cases.
_Z = POOL.symbol("z")
