"""Calls that hung, ran out of memory or ignored their budget (audit C4–C6).

Each case runs in a **subprocess** with a wall-clock cap (and, on Linux, an
address-space cap), so a regression fails this test instead of hanging the
suite or taking the runner's memory with it.

* C4 — ``limit`` of an expression with an undefined constant in it:
  ``limit(0**-1 + x, x, 1)`` ran for hours, ``limit(x*0**-1, x, 1)`` and
  ``limit(log(0) + x, x, 1)`` exhausted memory. ``0**-1`` stays unevaluated,
  and differentiating it for a Taylor coefficient triples its size per order.
  The expression is undefined at every ``x``: refused up front (``E-LIMIT-005``).
* C5 — ``integrate((2x + log x)**(-1/2))`` took 400 s to decline, and a 3 s
  budget took 8.6 s to fire: the tower Risch-DE ansatz had no checkpoint.
* C6 — ``factorint`` of ``(2**89-1)(2**107-1)(2**127-1)`` ignored
  ``Budget(wall_ms=1000)`` (one uninterruptible FLINT call); ``apart`` and
  ``integrate`` of ``1/(x**(2**31)+1)`` built a dense degree-``2**31``
  polynomial by repeated multiplication, with no checkpoint.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

_PRELUDE = textwrap.dedent(
    """
    import time
    try:  # Linux only: macOS rejects RLIMIT_AS, Windows has no `resource`.
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    except (ImportError, ValueError, OSError):
        pass
    import alkahest as ak
    p = ak.ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    one = p.integer(1)
    t0 = time.time()
    def report(fn):
        try:
            fn()
            print("OK")
        except Exception as e:
            print("ERR", type(e).__name__, getattr(e, "code", ""))
        print("T", time.time() - t0)
    """
)


def _run(body, wall=60):
    code = _PRELUDE + textwrap.dedent(body)
    env = dict(os.environ, MALLOC_ARENA_MAX="2")
    t = time.time()
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=wall,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.split("\n")
    verdict = next(line for line in lines if line.startswith(("OK", "ERR")))
    elapsed = float(next(line for line in lines if line.startswith("T ")).split()[1])
    return verdict, elapsed, time.time() - t


@pytest.mark.parametrize(
    "call",
    [
        "ak.limit(z**-1 + x, x, one)",
        "ak.limit(x * z**-1, x, one)",
        "ak.limit(1 / (one - one) + x, x, p.pos_infinity())",
        "ak.limit(ak.log(z) + x, x, one)",
        "ak.limit(x * ak.log(z), x, p.pos_infinity())",
    ],
)
def test_c4_limit_of_an_undefined_constant_is_refused_promptly(call):
    verdict, elapsed, _ = _run(f"report(lambda: {call})")
    assert verdict.startswith("ERR"), verdict
    assert "E-LIMIT" in verdict
    assert elapsed < 10.0


def test_c4_an_ordinary_limit_is_unaffected():
    verdict, _, _ = _run("report(lambda: print(ak.limit(ak.sin(x) / x, x, z).value))")
    assert verdict == "OK"


@pytest.mark.parametrize(
    "integrand",
    [
        "(2 * x + ak.log(x)) ** p.rational(-1, 2)",
        "ak.asin(x * x * ak.log(x))",
    ],
)
def test_c5_integrate_honours_its_budget(integrand):
    verdict, elapsed, _ = _run(
        f"""
        with ak.context(budget=ak.Budget(wall_ms=1000)):
            report(lambda: ak.integrate({integrand}, x))
        """,
        wall=120,
    )
    # Either answer (a refusal or a budget trip) is fine; hanging is not, and
    # neither is a budget that fires seconds late (the audit measured 3 s ->
    # 8.6 s; this is ~1.1 s now, the slack is for slow CI runners).
    assert verdict.startswith("ERR"), verdict
    assert elapsed < 4.0, f"a 1 s budget took {elapsed:.1f} s"


def test_c6_factorint_honours_its_budget():
    verdict, elapsed, _ = _run(
        """
        n = (2**89 - 1) * (2**107 - 1) * (2**127 - 1)
        with ak.context(budget=ak.Budget(wall_ms=1000)):
            report(lambda: ak.number_theory.factorint(n))
        """,
        wall=120,
    )
    assert verdict.startswith("ERR"), verdict
    assert "BudgetExceededError" in verdict, verdict
    # One ladder pass of overshoot (about 1 s here); the audit measured > 20 s.
    assert elapsed < 6.0, f"a 1 s budget took {elapsed:.1f} s"


def test_c6_factorint_under_a_budget_still_factors():
    """The ladder must not cost coverage: small and medium composites factor
    fully under a budget, and agree with the unbudgeted call."""
    verdict, _, _ = _run(
        """
        cases = [2**64 + 1, 2**128 + 1, -(3**40 * 7**3 * 1000003), 600851475143,
                 (2**61 - 1) * (2**89 - 1), 10**40 + 1]
        def go():
            for n in cases:
                with ak.context(budget=ak.Budget(wall_ms=20000)):
                    a = ak.number_theory.factorint(n)
                b = ak.number_theory.factorint(n)
                assert a == b, (n, a, b)
                prod = 1
                for q, e in a.items():
                    prod *= q**e
                assert prod == n, (n, a)
        report(go)
        """,
        wall=120,
    )
    assert verdict == "OK", verdict


@pytest.mark.parametrize("op", ["ak.apart(f, x)", "ak.integrate(f, x)"])
def test_c6_huge_degree_rational_honours_its_budget(op):
    verdict, elapsed, _ = _run(
        f"""
        f = 1 / (x ** p.integer(2**31) + 1)
        with ak.context(budget=ak.Budget(wall_ms=1000)):
            report(lambda: {op})
        """,
        wall=120,
    )
    assert elapsed < 6.0, f"a 1 s budget took {elapsed:.1f} s ({verdict})"
