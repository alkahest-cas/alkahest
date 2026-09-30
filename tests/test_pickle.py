"""Pickling ``Expr`` and ``ExprPool`` (see ``alkahest._pickle`` for the rules).

``pickle.dumps(expr)`` used to raise ``TypeError: cannot pickle
'builtins.Expr'``, so an expression could not be sent to a
``multiprocessing`` worker, cached with ``joblib``, or stored in any pickled
object.
"""

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
import operator
import pickle
import subprocess
import sys
import textwrap

import alkahest as ak
import pytest
from alkahest import ExprPool


@pytest.fixture
def pool():
    return ExprPool()


def _sample(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    return x, y, ak.sin(x) ** 2 + pool.rational(3, 7) * y + pool.integer(10**40) + pool.float(0.25)


def test_expr_round_trips_in_process(pool):
    x, _y, e = _sample(pool)
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        got = pickle.loads(pickle.dumps(e, protocol=proto))
        assert got == e  # same pool, same node
        assert got + x == e + x


def test_copy_and_deepcopy(pool):
    _, _, e = _sample(pool)
    assert copy.copy(e) == e
    assert copy.deepcopy(e) == e
    assert copy.deepcopy([e, e]) == [e, e]


def test_pool_round_trips_to_itself_in_process(pool):
    assert pickle.loads(pickle.dumps(pool)) is pool


def test_payload_is_the_expression_not_the_pool(pool):
    x = pool.symbol("x")
    small = pickle.dumps(x + 1)
    for i in range(5000):  # grow the pool with unrelated nodes
        _ = x + i
    assert len(pickle.dumps(x + 1)) <= len(small) + 64


def test_deep_expression_pickles(pool):
    e = pool.symbol("x")
    for _ in range(50_000):
        e = ak.sin(e)
    assert pickle.loads(pickle.dumps(e)) == e


def test_malformed_payload_is_an_io_error(pool):
    from alkahest._pickle import _unpickle_expr

    with pytest.raises(ak.IoError):
        _unpickle_expr(pool, b"ALKD\x05\x00\x00\x00")
    with pytest.raises(ak.IoError):
        _unpickle_expr(pool, b"not an expression")


_CHILD = textwrap.dedent(
    """
    import pickle, sys
    import alkahest as ak
    data = sys.stdin.buffer.read()
    first, second = data.split(b"--SPLIT--")
    x, y, e = pickle.loads(first)          # pickled together
    z = pickle.loads(second)               # pickled separately, same source pool
    # Cross-pool arithmetic raises, so each of these proves a shared pool.
    print(str(e + x))
    print(str(z * y))
    print(str(ak.diff(e, x).value))
    print(x == pickle.loads(first)[0])
    """
)


def test_exprs_pickled_together_share_one_pool_in_a_new_process(pool):
    x, y, e = _sample(pool)
    z = x * y + 1
    payload = pickle.dumps((x, y, e)) + b"--SPLIT--" + pickle.dumps(z)
    r = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=payload,
        capture_output=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr.decode()[-3000:]
    lines = r.stdout.decode().splitlines()
    assert lines[0] == str(e + x)
    assert lines[1] == str(z * y)
    assert lines[2] == str(ak.diff(e, x).value)
    assert lines[3] == "True"


def test_multiprocessing_round_trip_comes_home_to_the_original_pool(pool):
    x, y, e = _sample(pool)
    lhs = [x, e, y**2]
    rhs = [y, x, e]
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        products = list(ex.map(operator.mul, lhs, rhs))
    for a, b, got in zip(lhs, rhs, products):
        # Built in the worker, returned into *this* pool: the same node the
        # parent builds, and usable with expressions that never left.
        assert got == a * b
        assert got + x == a * b + x
