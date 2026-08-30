"""``limit`` must always come back — with a value, or with a coded refusal.

``ak.limit(ak.sqrt(x**2 + x) - x, x, oo)`` used to never return. It ran inside
Rust holding the GIL, so a Python-side ``SIGALRM`` or thread timeout could not
stop it, and the Gruntz/expansion path had no ``budget::check`` checkpoint, so
``context(budget=...)`` could not bound it either.

Every test here carries a ``pytest.mark.timeout`` so a regression fails loudly
instead of wedging CI.
"""

from __future__ import annotations

import alkahest as ak
import pytest

# Generous next to the sub-second times these take when they work, and still far
# below "the process is stuck".
TIMEOUT = 60


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x", "real")


@pytest.fixture(autouse=True)
def _clear_cancel_before_and_after():
    """Cancellation is a process-wide flag — never let a failing assertion in
    one test leave it set for the next test to inherit."""
    ak.clear_cancel()
    yield
    ak.clear_cancel()


def _hard_limit(pool: ak.ExprPool, x: ak.Expr) -> ak.Expr:
    """A radical limit the engine does not solve.

    ``sqrt(sqrt(x**2 + x) + x)`` sits on a half-integer scale, so the
    leading-order route declines it and it falls through to the expansion path
    — the one that used to run away. It is the natural probe for "does the
    engine stop when it cannot answer".
    """
    return ak.sqrt(ak.sqrt(x**2 + x) + x)


# ---------------------------------------------------------------------------
# The reported expression
# ---------------------------------------------------------------------------


@pytest.mark.timeout(TIMEOUT)
def test_sqrt_x_squared_plus_x_minus_x_at_infinity(pool, x):
    """The reported hang. ``sqrt(x**2 + x) - x -> 1/2`` (multiply by the
    conjugate: ``x / (sqrt(x**2 + x) + x)``)."""
    got = ak.limit(ak.sqrt(x**2 + x) - x, x, pool.pos_infinity()).value
    assert got == pool.rational(1, 2), f"got {got}"


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda p, x: ak.sqrt(x**2 + x) - x, "1/2"),
        (lambda p, x: x - ak.sqrt(x**2 + x), "-1/2"),
        (lambda p, x: ak.sqrt(x**2 + 3 * x) - x, "3/2"),
        (lambda p, x: ak.sqrt(x**2 + 1) - x, "0"),
        (lambda p, x: ak.sqrt(x**2 + x) / x, "1"),
        (lambda p, x: ak.sqrt(x**2 + x), "∞"),
    ],
)
def test_algebraic_limits_at_infinity(pool, x, build, expected):
    """The ∞−∞ family the conjugate trick covers, plus the neighbours that must
    keep working."""
    assert str(ak.limit(build(pool, x), x, pool.pos_infinity()).value) == expected


# ---------------------------------------------------------------------------
# Bounded and interruptible, whatever the algorithm does
# ---------------------------------------------------------------------------


@pytest.mark.timeout(TIMEOUT)
def test_unsolvable_limit_refuses_instead_of_hanging(pool, x):
    """With no budget set, an unsolvable search must hit the internal work
    ceiling and refuse with ``E-LIMIT-004`` rather than spin forever."""
    with pytest.raises(ak.LimitError) as excinfo:
        ak.limit(_hard_limit(pool, x), x, pool.pos_infinity())
    assert excinfo.value.code == "E-LIMIT-004"
    assert excinfo.value.remediation


@pytest.mark.timeout(TIMEOUT)
def test_wall_budget_stops_a_hard_limit(pool, x):
    """``Budget(wall_ms=...)`` bounds the call and reports ``E-BUDGET-001``."""
    with (
        pytest.raises(ak.BudgetExceededError) as excinfo,
        ak.context(budget=ak.Budget(wall_ms=50)),
    ):
        ak.limit(_hard_limit(pool, x), x, pool.pos_infinity())
    assert excinfo.value.code == "E-BUDGET-001"


@pytest.mark.timeout(TIMEOUT)
def test_step_budget_stops_a_hard_limit(pool, x):
    """``Budget(max_steps=...)`` bounds the call and reports ``E-BUDGET-002``."""
    with (
        pytest.raises(ak.BudgetExceededError) as excinfo,
        ak.context(budget=ak.Budget(max_steps=5)),
    ):
        ak.limit(_hard_limit(pool, x), x, pool.pos_infinity())
    assert excinfo.value.code == "E-BUDGET-002"


@pytest.mark.timeout(TIMEOUT)
def test_request_cancel_interrupts_a_limit(pool, x):
    """``request_cancel()`` reaches the limit engine's checkpoints, exactly as
    it reaches ``integrate``'s (``E-BUDGET-003``)."""
    ak.request_cancel()
    with pytest.raises(ak.BudgetExceededError) as excinfo:
        ak.limit(_hard_limit(pool, x), x, pool.pos_infinity())
    assert excinfo.value.code == "E-BUDGET-003"
    ak.clear_cancel()
    # And the engine is usable again straight afterwards.
    assert ak.limit(ak.sqrt(x**2 + x) - x, x, pool.pos_infinity()).value == pool.rational(1, 2)


@pytest.mark.timeout(TIMEOUT)
def test_a_solvable_limit_under_a_generous_budget_still_answers(pool, x):
    """The checkpoints must not turn working limits into budget errors."""
    with ak.context(budget=ak.Budget(wall_ms=10_000, max_steps=1_000_000)):
        assert ak.limit(ak.sqrt(x**2 + x) - x, x, pool.pos_infinity()).value == pool.rational(
            1, 2
        )
        assert str(ak.limit(ak.sin(x) / x, x, pool.integer(0)).value) == "1"


# ---------------------------------------------------------------------------
# Existing coverage must not narrow
# ---------------------------------------------------------------------------


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(
    ("build", "point", "direction", "expected"),
    [
        (lambda x: ak.sin(x) / x, 0, "+-", "1"),
        (lambda x: (1 - ak.cos(x)) / x**2, 0, "+-", "1/2"),
        (lambda x: x * ak.log(x), 0, "+", "0"),
        (lambda x: (x**8 - 1) / (x - 1), 1, "+-", "8"),
        (lambda x: x * ak.sin(1 / x), 0, "+-", "0"),
        (lambda x: (1 + x) ** (1 / x), 0, "+-", "exp(1)"),
        (lambda x: (1 + 1 / x) ** x, "oo", "+-", "exp(1)"),
        (lambda x: (1 + 2 / x) ** x, "oo", "+-", "exp(2)"),
        (lambda x: ak.exp(x) / x**2, "oo", "+-", "∞"),
        (lambda x: ak.exp(-x), "oo", "+-", "0"),
        (lambda x: x * ak.exp(-x), "oo", "+-", "0"),
        (lambda x: ak.exp(ak.exp(x)), "oo", "+-", "∞"),
        (lambda x: ak.exp(2 * x) / ak.exp(3 * x), "oo", "+-", "0"),
        (lambda x: x / (x + 1), "oo", "+-", "1"),
        (lambda x: x**2, "oo", "+-", "∞"),
        (lambda x: 1 + 1 / x, "oo", "+-", "1"),
    ],
)
def test_limits_that_already_worked_still_work(pool, x, build, point, direction, expected):
    """A spread across every route — direct substitution, L'Hôpital, the series
    expansion, the indeterminate-power rewrite, Gruntz. The termination guard
    must not have narrowed any of them."""
    pt = pool.pos_infinity() if point == "oo" else pool.integer(point)
    assert str(ak.limit(build(x), x, pt, direction).value) == expected


@pytest.mark.timeout(TIMEOUT)
def test_limits_that_were_refused_are_still_refused(pool, x):
    """Refusals are coverage too: the guard must not have started answering
    limits that do not exist."""
    for expr in (ak.sin(x), ak.cos(x)):
        with pytest.raises(ak.LimitError):
            ak.limit(expr, x, pool.pos_infinity())
    # x/|x| at 0 has one-sided limits ∓1 and no two-sided one.
    with pytest.raises(ak.LimitError):
        ak.limit(x / ak.abs(x), x, pool.integer(0))
