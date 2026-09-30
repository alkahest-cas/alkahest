"""Compiled functions: constants in batch, and symbols the inputs do not bind.

Two silent wrong answers from the audit (A11):

* a function of no inputs batched to the caller's zero-filled buffer —
  ``call_batch_raw([], 0, 3)`` on ``compile_expr(5, [])`` gave ``[0, 0, 0]``
  while ``f([])`` gave ``5``;
* ``compile_expr(x + y, [x])`` compiled, and returned ``nan`` at every point,
  where ``eval_expr`` on the same expression raises.
"""

from __future__ import annotations

import math

import alkahest
import pytest
from alkahest import CompileCache, ExprPool, compile_expr


def test_a_constant_function_batches_to_its_constant():
    p = ExprPool()
    f = compile_expr(p.integer(5), [])
    assert f([]) == 5.0
    assert f.call_batch_raw([], 0, 3) == [5.0, 5.0, 5.0]
    # Large enough to recompile on a native tier when one is built in.
    n = 100_000
    assert set(f.call_batch_raw([], 0, n)) == {5.0}
    if hasattr(f, "call_batch_raw_par"):
        assert set(f.call_batch_raw_par([], 0, n)) == {5.0}


def test_a_constant_function_batches_through_the_buffer_path():
    np = pytest.importorskip("numpy")
    p = ExprPool()
    f = compile_expr(p.integer(5), [])
    if not hasattr(f, "call_batch_buffer"):
        pytest.skip("buffer protocol path not built (limited API < 3.11)")
    out = np.zeros(4)
    f.call_batch_buffer([], out)
    assert out.tolist() == [5.0] * 4


def test_compile_refuses_a_symbol_the_inputs_do_not_bind():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(alkahest.JitError) as info:
        compile_expr(x + y, [x])
    assert info.value.code == "E-JIT-005"
    assert "y" in str(info.value)
    # ValueError, like every other coded alkahest error.
    with pytest.raises(ValueError):
        compile_expr(x + y, [x], expected_evals=1_000_000)
    with pytest.raises(alkahest.JitError) as info:
        CompileCache().compile(x + y, [x])
    assert info.value.code == "E-JIT-005"


def test_binding_every_symbol_still_compiles():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    assert compile_expr(x + y, [x, y])([1.0, 2.0]) == 3.0
    # pi is a constant, not a free symbol.
    f = compile_expr(alkahest.sin(p.symbol("pi") * x), [x])
    assert math.isclose(f([0.5]), 1.0)
