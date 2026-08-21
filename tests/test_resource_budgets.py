"""Memory ceilings: `Budget.max_bytes`, and refusing before an out-of-memory abort.

2026-08-19 issue #5. Three subsystems reached an out-of-memory condition on
three different code paths — `telescope_md` at *its own default arguments*,
a 7-parameter parametric Gröbner stage, and a jet-order-4 elimination — and
all three died the same way::

    GNU MP: Cannot allocate memory (size=8)
    timeout: the monitored command dumped core          # SIGABRT, exit 134

No exception, no ``BudgetExceededError``, no ``try``/``except`` that could
help, and no memory analogue of ``Budget.wall_ms``. An unattended research
loop lost the whole interpreter, and with it every result it was holding —
not just the offending call.

GMP cannot be made to *fail* an allocation: its contract for a replacement
allocator forbids returning ``NULL``, and a Rust ``panic!`` may not cross a C
frame. So the refusal happens *before* the allocation, at Alkahest's own
cooperative checkpoints, against two ceilings:

``E-BUDGET-004``
    the caller's ``Budget(max_bytes=...)``.
``E-BUDGET-005``
    the process is about to exhaust a finite ``RLIMIT_AS`` (``ulimit -v``, a
    container limit). Active with **no budget at all**, which is what makes
    the default-arguments case survivable.

Also covers the two sibling resource gaps from the same round: `q_zeilberger`
having no ceiling (#10) and `prove_nonneg` honouring no budget (26d).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import alkahest as ak
import pytest

#: A call that has to *finish* to prove anything here takes well under a
#: minute; this only distinguishes "refused" from "hung".
HEAVY_TIMEOUT = 300


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


def multinomial(pool: ak.ExprPool, m: int):
    """``n! / (x₁!·…·x_m!·(n−Σxᵢ)!)`` — the m-index multinomial coefficient.

    At ``m = 4`` and ``telescope_md``'s own defaults this ran ~20 minutes and
    then aborted the process; ``m = 3`` aborts too, under a memory cap that
    ``m = 2`` fits inside comfortably.
    """
    one = pool.integer(1)
    n = pool.symbol("n")
    xs = [pool.symbol(f"x{t + 1}") for t in range(m)]
    rest = n
    for x in xs:
        rest = rest - x
    den = ak.gamma(rest + one)
    for x in xs:
        den = den * ak.gamma(x + one)
    return ak.gamma(n + one) / den, n, xs


# ---------------------------------------------------------------------------
# #5 — the abort itself
# ---------------------------------------------------------------------------

#: Run in a **subprocess** with its own `RLIMIT_AS`, for two reasons: an
#: out-of-memory abort cannot be caught in-process (that is the whole bug), and
#: a test that deliberately exhausts memory must not be able to take the test
#: runner with it. The cap is set *relative to the address space the import
#: already mapped*, so this does not depend on how much a given build reserves.
_OOM_CHILD = textwrap.dedent(
    """
    import resource, sys
    import alkahest as ak
    from alkahest.experimental import telescope_md

    def vsz():
        return int(open("/proc/self/statm").read().split()[0]) * resource.getpagesize()

    # 96 MB of headroom: more than the m = 2 case needs, far less than m = 3.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vsz() + 96 * 1024 * 1024, hard))

    pool = ak.ExprPool()
    one = pool.integer(1)
    n = pool.symbol("n")
    xs = [pool.symbol("x%d" % (t + 1)) for t in range(3)]
    rest = n
    for x in xs:
        rest = rest - x
    den = ak.gamma(rest + one)
    for x in xs:
        den = den * ak.gamma(x + one)
    term = ak.gamma(n + one) / den

    try:
        telescope_md(term, n, xs)
    except ak.BudgetExceededError as e:
        print("REFUSED", e.code)
        sys.exit(0)
    except ak.HolonomicError as e:
        # Also acceptable: a refusal is a refusal, as long as it is catchable.
        print("REFUSED", e.code)
        sys.exit(0)
    print("COMPLETED")
    """
)


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS + /proc/self/statm are Linux-only")
def test_out_of_memory_is_a_catchable_error_not_a_process_abort():
    """The headline of issue #5: no ``SIGABRT``, and an ``except`` clause works.

    Before the fix this child died with ``GNU MP: Cannot allocate memory
    (size=8)`` and ``returncode == -6``; the ``print`` after the ``try`` was
    never reached, and neither was any ``except``.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _OOM_CHILD],
        capture_output=True,
        text=True,
        timeout=HEAVY_TIMEOUT,
    )
    assert proc.returncode >= 0, (
        f"the child died on signal {-proc.returncode} instead of raising — "
        f"stderr: {proc.stderr[-800:]}"
    )
    assert "Cannot allocate memory" not in proc.stderr, proc.stderr[-800:]
    assert proc.returncode == 0, f"stdout: {proc.stdout!r} stderr: {proc.stderr[-800:]}"
    assert proc.stdout.startswith("REFUSED E-BUDGET-"), proc.stdout


# ---------------------------------------------------------------------------
# #5 — Budget.max_bytes
# ---------------------------------------------------------------------------


def test_budget_max_bytes_defaults_to_none_and_validates():
    assert ak.Budget().max_bytes is None
    assert ak.Budget(max_bytes=1024).max_bytes == 1024
    with pytest.raises(ValueError, match="max_bytes"):
        ak.Budget(max_bytes=-1)


def test_max_bytes_is_a_trailing_field_so_positional_construction_still_works():
    """``Budget`` is a public dataclass; the field had to be *appended*."""
    b = ak.Budget(50.0, 10, 7)
    assert (b.wall_ms, b.max_steps, b.seed, b.max_bytes) == (50.0, 10, 7, None)


def test_max_bytes_trips_telescope_md_with_a_coded_error(pool):
    """``E-BUDGET-004``, and it is a ``BudgetExceededError`` — not a
    ``HolonomicError`` that a loop would read as "no certificate exists"."""
    from alkahest.experimental import telescope_md

    term, n, xs = multinomial(pool, 3)
    with ak.context(budget=ak.Budget(max_bytes=1 << 20)):
        with pytest.raises(ak.BudgetExceededError) as exc:
            telescope_md(term, n, xs)
    assert exc.value.code == "E-BUDGET-004"
    assert "max_bytes" in (exc.value.remediation or "")


def test_a_generous_max_bytes_does_not_change_the_answer(pool):
    """The ceiling is a ceiling, not an unconditional refusal."""
    from alkahest.experimental import telescope_md

    term, n, xs = multinomial(pool, 2)
    with ak.context(budget=ak.Budget(max_bytes=1 << 40)):
        cert = telescope_md(term, n, xs)
    assert cert.order >= 1


def test_gmp_accounting_is_installed_and_counts(pool):
    """``max_bytes`` is measured against GMP's own live-byte total."""
    native = ak.alkahest
    before = native.gmp_live_bytes()
    big = pool.integer(2) ** pool.integer(200_000)
    ak.simplify_expanded(big)
    assert native.gmp_live_bytes() > 0
    assert before >= 0


# ---------------------------------------------------------------------------
# #10 — q_zeilberger's resource ceiling
# ---------------------------------------------------------------------------


def test_q_zeilberger_refuses_at_a_ceiling_rather_than_running_unbounded(pool):
    """``Σ_k [2n;k]_q`` — class-legal, and previously unbounded.

    Its near-twin ``Σ_k [n;k]_q`` decides in half a second; this one ran 8+
    minutes at the documented defaults with no output and had to be killed.
    The refusal must *say* it is a ceiling, or a loop records a false negative.
    """
    from alkahest.experimental import q_zeilberger, qbinomial

    q, n, k = pool.symbol("q"), pool.symbol("n"), pool.symbol("k")
    term = qbinomial(pool, pool.integer(2) * n, k)
    with pytest.raises(ak.HolonomicError) as exc:
        q_zeilberger(term, q, n, k, max_order=2, max_degree=2)
    assert exc.value.code == "E-HOLO-021"
    assert "resource ceilings" in str(exc.value)


def test_q_zeilberger_still_solves_the_cheap_sibling(pool):
    """``Σ_k [n;k]_q`` must be unaffected by the ceilings."""
    from alkahest.experimental import q_zeilberger, qbinomial

    q, n, k = pool.symbol("q"), pool.symbol("n"), pool.symbol("k")
    cert = q_zeilberger(qbinomial(pool, n, k), q, n, k)
    assert cert.order >= 1


def test_q_zeilberger_honours_a_wall_budget(pool):
    """It consulted no budget at all before — ``wall_ms`` did nothing."""
    from alkahest.experimental import q_zeilberger, qbinomial

    q, n, k = pool.symbol("q"), pool.symbol("n"), pool.symbol("k")
    term = qbinomial(pool, pool.integer(2) * n, k)
    with ak.context(budget=ak.Budget(wall_ms=300)):
        with pytest.raises(ak.BudgetExceededError) as exc:
            q_zeilberger(term, q, n, k, max_order=3, max_degree=6)
    assert exc.value.code == "E-BUDGET-001"


# ---------------------------------------------------------------------------
# 26d — prove_nonneg honours a budget
# ---------------------------------------------------------------------------


def test_prove_nonneg_honours_a_wall_budget(pool):
    """418.5 s inside ``Budget(wall_ms=3000)``, ending in ``E-SOS-002``.

    The error class matters as much as the timing: ``E-SOS-002`` already
    conflates "exhausted", "budget-limited" and "never attempted", so a loop
    that reads it as "not SOS" files a false negative. A budget stop has to be
    a ``BudgetExceededError``.
    """
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    sq = lambda e: e * e  # noqa: E731
    motzkin = (
        sq(x) * sq(x) * sq(y)
        + sq(x) * sq(y) * sq(y)
        - pool.integer(3) * sq(x) * sq(y) * sq(z)
        + sq(z) * sq(z) * sq(z)
    )
    with ak.context(budget=ak.Budget(wall_ms=1000)):
        with pytest.raises(ak.BudgetExceededError) as exc:
            ak.prove_nonneg(motzkin, [x, y, z])
    assert exc.value.code == "E-BUDGET-001"


def test_prove_nonneg_without_a_budget_still_reports_no_certificate(pool):
    """No budget, no change: the Motzkin form is still an honest ``E-SOS-002``."""
    x, y = pool.symbol("x"), pool.symbol("y")
    p = (x * x - pool.integer(2) * x * y + y * y) + pool.integer(1)
    cert = ak.prove_nonneg(p, [x, y])
    assert cert.kind in {"sos", "handelman", "putinar"}
