"""Rust panics must not reach Python as ``PanicException``.

A panic that crosses the PyO3 boundary surfaces as ``pyo3_runtime.PanicException``,
which inherits from ``BaseException``, **not** ``Exception``.  An unattended loop
that wraps each candidate in ``except Exception`` therefore does not catch it,
and the run dies — the same failure mode as a segfault, minus the core dump.

Each test below drives an argument that used to panic inside Rust and asserts
both that it raises something ``except Exception`` can see, and that it is the
specific error type a caller would expect.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from alkahest.alkahest import Fps


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x", "real")


def assert_ordinary_exception(fn) -> Exception:
    """Call ``fn`` and require the failure to be a catchable ``Exception``.

    The ``BaseException`` arm is what a ``PanicException`` would land in — the
    regression this whole module is about.
    """
    try:
        fn()
    except Exception as e:  # deliberately broad: catching it here is the test
        return e
    except BaseException as e:  # pragma: no cover - reaching this arm is the bug
        pytest.fail(f"escaped as {type(e).__name__}, which except Exception misses")
    pytest.fail("expected a refusal")


# ---------------------------------------------------------------------------
# rug precision: Float::with_val panics outside [1, i32::MAX]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prec", [0, 4_000_000_000])
def test_arbball_rejects_out_of_range_precision(prec: int):
    e = assert_ordinary_exception(lambda: ak.ArbBall(1.0, 0.0, prec))
    assert isinstance(e, ValueError)


@pytest.mark.parametrize("prec", [0, 4_000_000_000])
def test_pool_float_rejects_out_of_range_precision(pool: ak.ExprPool, prec: int):
    e = assert_ordinary_exception(lambda: pool.float(1.5, prec))
    assert isinstance(e, ValueError)


def test_interval_eval_rejects_zero_precision(pool: ak.ExprPool, x: ak.Expr):
    """``evaluate`` already validated this; ``interval_eval`` was its
    unguarded twin and panicked deep inside the ball evaluator."""
    e = assert_ordinary_exception(lambda: ak.interval_eval(x, {x: ak.ArbBall(0.5, 0.1)}, prec=0))
    assert isinstance(e, ValueError)


def test_guess_relation_rejects_zero_precision():
    e = assert_ordinary_exception(lambda: ak.guess_relation([1.0, 2.0], precision_bits=0))
    assert isinstance(e, ValueError)


def test_bound_on_box_rejects_zero_precision(pool: ak.ExprPool, x: ak.Expr):
    e = assert_ordinary_exception(lambda: ak.bound_on_box(x, [(x, 0.0, 1.0)], prec=0))
    assert isinstance(e, ValueError)


def test_a_valid_precision_still_works(pool: ak.ExprPool, x: ak.Expr):
    """The validator must not have narrowed the useful range."""
    assert ak.ArbBall(1.0, 0.0, 256).mid == 1.0
    assert ak.interval_eval(x, {x: ak.ArbBall(0.5, 0.1)}, prec=256) is not None


# ---------------------------------------------------------------------------
# Matrix element access
# ---------------------------------------------------------------------------


def test_matrix_get_out_of_range_raises_index_error(pool: ak.ExprPool, x: ak.Expr):
    """``Matrix::get`` indexes a flat vector with no bounds check, so an
    off-by-one loop bound used to panic rather than raise."""
    m = ak.Matrix([[x, x], [x, x]])
    e = assert_ordinary_exception(lambda: m.get(0, 5))
    assert isinstance(e, IndexError)
    e = assert_ordinary_exception(lambda: m.get(5, 0))
    assert isinstance(e, IndexError)
    # A huge row index used to wrap `r * cols` and silently read the wrong
    # element instead of failing.
    assert_ordinary_exception(lambda: m.get(2**62, 0))
    assert m.get(1, 1) is not None


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_enormous_wall_ms_saturates_instead_of_panicking():
    """``Duration::from_secs_f64`` panics past ``u64::MAX`` seconds, and
    ``wall_ms=1e30`` is a plausible way to spell 'effectively unlimited'."""
    with ak.context(budget=ak.Budget(wall_ms=1e30)):
        assert ak.is_budget_active()


# ---------------------------------------------------------------------------
# A Python callback invoked from Rust
# ---------------------------------------------------------------------------


def test_raising_sparse_interp_oracle_propagates_its_own_exception():
    """The oracle is user code called from Rust through an infallible
    signature; it used to be ``.expect()``-ed, so *any* exception it raised
    became a ``PanicException``."""

    class Sentinel(Exception):
        pass

    def oracle(_x: int) -> int:
        raise Sentinel("oracle exploded")

    with pytest.raises(Sentinel):
        ak.sparse_interp_univariate(oracle, 3, 997)


def test_sparse_interp_oracle_returning_a_bad_type_is_a_type_error():
    e = assert_ordinary_exception(
        lambda: ak.sparse_interp_univariate(lambda _x: "not an int", 3, 997)
    )
    assert isinstance(e, TypeError)


def test_a_working_sparse_interp_oracle_is_unaffected():
    p = 997

    def f(v: int) -> int:
        return (v**5 + 3) % p

    terms = ak.sparse_interp_univariate(f, 3, p)
    assert sorted(terms) == [(1, 5), (3, 0)]


# ---------------------------------------------------------------------------
# Unbounded allocation from a user-supplied size
# ---------------------------------------------------------------------------


def test_plot_svg_rejects_an_absurd_point_count(pool: ak.ExprPool, x: ak.Expr):
    """``n_pts`` reached ``Vec::with_capacity`` unchecked: a capacity-overflow
    panic, or an allocation the OOM killer resolves."""
    e = assert_ordinary_exception(lambda: ak.plot_svg(x, x, n=10**15))
    assert isinstance(e, ValueError)
    assert ak.plot_svg(x, x, n=50).startswith("<svg")


def test_series_rejects_an_absurd_order(pool: ak.ExprPool, x: ak.Expr):
    """``series(sin(x), x, 0, 2**31 - 1)`` sized a coefficient vector out of
    the process.  Rust's allocator *aborts* on failure — ``SIGABRT``, no
    unwinding — so this was a process kill, not even a panic."""
    e = assert_ordinary_exception(lambda: ak.series(ak.sin(x), x, pool.integer(0), 2**31 - 1))
    assert isinstance(e, ValueError)
    assert ak.series(ak.sin(x), x, pool.integer(0), 8) is not None


def test_fps_rejects_absurd_coefficient_counts(pool: ak.ExprPool, x: ak.Expr):
    """Same allocator abort by a different door: ``order``/``n`` on the formal
    power series API were forwarded to a ``vec![...; n]`` unchecked."""
    fps = Fps.from_expr(ak.sin(x), x, 8)
    for call in (
        lambda: Fps.from_expr(ak.sin(x), x, 2**40),
        lambda: fps.coeff(2**40),
        lambda: fps.coeffs(2**40),
        lambda: fps.to_expr(x, 2**31 - 1),
    ):
        assert isinstance(assert_ordinary_exception(call), ValueError)
    assert len(fps.coeffs(5)) == 5
