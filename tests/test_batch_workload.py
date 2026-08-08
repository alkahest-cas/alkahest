"""Batch and streaming fan-out (``alkahest._batch``).

Covers the properties that matter for a search loop driving hundreds of
candidates: one bad element never raises and never gets dropped, input order
is preserved by :func:`alkahest.batch_map` regardless of ``parallel``,
:func:`alkahest.batch_map_iter` streams in the documented order for each
mode, and a captured error carries a real diagnostic code — the integrator's
own ``E-INT-*`` code when the exception has one, ``E-BATCH-001`` otherwise.
"""

from __future__ import annotations

import time

import alkahest as ak
import pytest
from alkahest._batch import UNEXPECTED_ERROR_CODE, BatchItem
from alkahest.exceptions import AlkahestError


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_public_names_exported_from_package_root():
    for name in (
        "BatchItem",
        "batch_map",
        "batch_map_iter",
        "integrate_many",
        "simplify_many",
        "diff_many",
    ):
        assert hasattr(ak, name), f"alkahest.{name} not exported"
        assert name in ak.__all__


# ---------------------------------------------------------------------------
# batch_map: never raises, preserves order, mixed success/failure
# ---------------------------------------------------------------------------


def test_batch_map_never_raises_for_a_bad_element(pool):
    x = pool.symbol("x")
    # log(log(x)) has no elementary antiderivative the kernel implements today.
    outs = ak.batch_map(lambda e: ak.integrate(e, x), [x**2, ak.log(ak.log(x)), ak.sin(x)])

    assert len(outs) == 3
    assert [o.ok for o in outs] == [True, False, True]
    assert outs[1].value is None
    assert outs[1].error is not None
    assert outs[0].error is None
    assert outs[2].error is None


def test_batch_map_preserves_input_order_sequential(pool):
    x = pool.symbol("x")
    exprs = [x**n for n in range(1, 8)]
    outs = ak.batch_map(lambda e: ak.diff(e, x), exprs)
    assert [o.index for o in outs] == list(range(7))
    assert all(o.ok for o in outs)


def test_batch_map_preserves_input_order_parallel():
    # Deliberately vary sleep so completion order differs from input order;
    # batch_map must still return results aligned to the original index.
    delays = [0.05, 0.01, 0.03, 0.0, 0.02]

    def _work(i):
        time.sleep(delays[i])
        return i * 10

    outs = ak.batch_map(_work, range(len(delays)), parallel=True)
    assert [o.index for o in outs] == list(range(len(delays)))
    assert [o.value for o in outs] == [i * 10 for i in range(len(delays))]
    assert all(o.ok for o in outs)


def test_batch_map_result_count_matches_input_even_with_all_failures():
    def _boom(_item):
        raise ValueError("always fails")

    outs = ak.batch_map(_boom, range(5))
    assert len(outs) == 5
    assert all(not o.ok for o in outs)
    assert all(o.value is None for o in outs)


def test_batch_map_empty_input_returns_empty_list():
    assert ak.batch_map(lambda x: x, []) == []
    assert ak.batch_map(lambda x: x, [], parallel=True) == []


def test_batch_map_forwards_kwargs_to_fn(pool):
    x = pool.symbol("x")
    outs = ak.batch_map(ak.simplify, [x + 0 * x], assumptions=None)
    assert outs[0].ok


# ---------------------------------------------------------------------------
# Error capture: stable codes, remediation, unexpected-failure fallback
# ---------------------------------------------------------------------------


def test_error_code_preserved_from_native_alkahest_exception(pool):
    x = pool.symbol("x")
    outs = ak.batch_map(lambda e: ak.integrate(e, x), [ak.log(ak.log(x))])
    error = outs[0].error
    assert error["code"] == "E-INT-001"
    assert error["remediation"]
    assert "log" in error["message"] or "integrate" in error["message"]
    assert error["type"]


def test_error_code_preserved_from_python_alkahest_error():
    class _CustomError(AlkahestError):
        def __init__(self, message):
            super().__init__(message, code="E-CUSTOM-042", remediation="do the other thing")

    def _raise(_item):
        raise _CustomError("nope")

    outs = ak.batch_map(_raise, [1])
    assert outs[0].error == {
        "code": "E-CUSTOM-042",
        "message": "nope",
        "remediation": "do the other thing",
        "type": "_CustomError",
    }


def test_unexpected_error_without_code_gets_batch_fallback_code():
    def _raise(_item):
        raise RuntimeError("no .code attribute here")

    outs = ak.batch_map(_raise, [1])
    assert outs[0].error["code"] == UNEXPECTED_ERROR_CODE
    assert outs[0].error["code"] == "E-BATCH-001"
    assert outs[0].error["remediation"] is None
    assert outs[0].error["type"] == "RuntimeError"


def test_keyboard_interrupt_is_not_captured():
    def _raise(_item):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ak.batch_map(_raise, [1])


# ---------------------------------------------------------------------------
# BatchItem shape
# ---------------------------------------------------------------------------


def test_batch_item_ok_success_shape():
    outs = ak.batch_map(lambda i: i * 2, [21])
    item = outs[0]
    assert isinstance(item, BatchItem)
    assert item.ok is True
    assert item.value == 42
    assert item.error is None
    assert item.elapsed_ms is not None
    assert item.elapsed_ms >= 0.0


def test_batch_item_is_frozen():
    item = BatchItem(index=0, ok=True, value=1)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError is a subclass
        item.value = 2


# ---------------------------------------------------------------------------
# batch_map_iter: order guarantees
# ---------------------------------------------------------------------------


def test_batch_map_iter_sequential_is_input_order(pool):
    x = pool.symbol("x")
    exprs = [x**n for n in range(1, 6)]
    items = list(ak.batch_map_iter(lambda e: ak.diff(e, x), exprs))
    assert [item.index for item in items] == list(range(5))


def test_batch_map_iter_sequential_matches_batch_map(pool):
    x = pool.symbol("x")
    exprs = [x**2, ak.log(ak.log(x)), ak.sin(x)]
    mapped = ak.batch_map(lambda e: ak.integrate(e, x), exprs)
    streamed = list(ak.batch_map_iter(lambda e: ak.integrate(e, x), exprs))
    assert [o.ok for o in mapped] == [o.ok for o in streamed]
    assert [o.index for o in mapped] == [o.index for o in streamed]


def test_batch_map_iter_parallel_streams_in_completion_order():
    # Item 0 sleeps longest, so it must be the *last* one yielded even though
    # it was submitted first — this is exactly what streaming buys a caller.
    delays = [0.08, 0.0, 0.0, 0.0]

    def _work(i):
        time.sleep(delays[i])
        return i

    order = [item.index for item in ak.batch_map_iter(_work, range(4), parallel=True)]
    assert order[-1] == 0
    assert set(order) == {0, 1, 2, 3}


def test_batch_map_iter_parallel_never_raises_and_covers_every_index():
    def _work(i):
        if i % 2 == 0:
            raise ValueError(f"bad item {i}")
        return i

    items = list(ak.batch_map_iter(_work, range(6), parallel=True))
    assert {item.index for item in items} == set(range(6))
    for item in items:
        if item.index % 2 == 0:
            assert not item.ok
            assert item.error["code"] == UNEXPECTED_ERROR_CODE
        else:
            assert item.ok
            assert item.value == item.index


def test_batch_map_iter_empty_input_yields_nothing():
    assert list(ak.batch_map_iter(lambda x: x, [])) == []
    assert list(ak.batch_map_iter(lambda x: x, [], parallel=True)) == []


# ---------------------------------------------------------------------------
# integrate_many / simplify_many / diff_many
# ---------------------------------------------------------------------------


def test_integrate_many_mixed_success_and_failure(pool):
    x = pool.symbol("x")
    outs = ak.integrate_many([x**2, ak.log(ak.log(x)), ak.sin(x)], x)
    assert [o.ok for o in outs] == [True, False, True]
    assert outs[1].error["code"] == "E-INT-001"
    assert str(outs[0].value.value) == str(ak.integrate(x**2, x).value)


def test_integrate_many_definite_bounds(pool):
    x = pool.symbol("x")
    zero, one = pool.integer(0), pool.integer(1)
    outs = ak.integrate_many([x**2, x**3], x, zero, one)
    assert all(o.ok for o in outs)
    assert str(outs[0].value.value) == str(ak.integrate(x**2, x, zero, one).value)


def test_integrate_many_parallel_preserves_order(pool):
    x = pool.symbol("x")
    exprs = [x**n for n in range(1, 12)]
    outs = ak.integrate_many(exprs, x, parallel=True)
    assert [o.index for o in outs] == list(range(len(exprs)))
    assert all(o.ok for o in outs)


def test_simplify_many_mixed_success_and_failure(pool):
    x = pool.symbol("x")
    outs = ak.simplify_many([x + 0 * x, x / x])
    assert all(o.ok for o in outs)
    assert str(outs[0].value.value) == "x"


def test_diff_many_mixed(pool):
    x = pool.symbol("x")
    outs = ak.diff_many([x**2, ak.sin(x), ak.cos(x)], x)
    assert all(o.ok for o in outs)
    assert str(outs[0].value.value) == str(ak.diff(x**2, x).value)


def test_many_helpers_never_raise_on_a_bad_element(pool):
    x = pool.symbol("x")
    # A non-Expr sentinel forces a failure path inside the wrapped call.
    outs = ak.simplify_many([x, "not an expr"])
    assert outs[0].ok
    assert not outs[1].ok
    assert outs[1].error is not None


def test_many_helpers_are_batch_map_over_the_underlying_op(pool):
    x = pool.symbol("x")
    exprs = [x**2, x**3]
    direct = [ak.batch_map(lambda e: ak.diff(e, x), exprs)[i].value.value for i in range(2)]
    via_helper = [ak.diff_many(exprs, x)[i].value.value for i in range(2)]
    assert [str(v) for v in direct] == [str(v) for v in via_helper]
