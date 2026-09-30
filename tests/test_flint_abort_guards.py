"""Inputs that used to kill the interpreter inside FLINT or GMP.

FLINT and GMP cannot fail gracefully once called: a composite modulus, an
integer too large to represent or an allocation that cannot succeed ends in
``abort()`` (or ``SIGFPE``), which no ``try`` can catch. Each case below runs in
a fresh interpreter so that a regression shows up as a failed test rather than
as the death of the test runner, and must now end in an ordinary exception.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import alkahest
import pytest

PRELUDE = "import alkahest as a, alkahest.alkahest as m, alkahest.number_theory as nt\n"

# (id, snippet, expected exception class name, expected code prefix)
CASES = [
    ("fmod_15", "m.factor_univariate_mod_p([6, 5, 1], 15)", "FactorError", "E-POLY-009"),
    ("fmod_4", "m.factor_univariate_mod_p([0, 0, 1], 4)", "FactorError", "E-POLY-009"),
    ("fmod_9", "m.factor_univariate_mod_p([1, 0, 1], 9)", "FactorError", "E-POLY-009"),
    ("fmod_12", "m.factor_univariate_mod_p([1, 2, 1], 12)", "FactorError", "E-POLY-009"),
    ("fmod_21", "m.factor_univariate_mod_p([1, 0, 0, 1], 21)", "FactorError", "E-POLY-009"),
    ("sigma_1e12", "m.divisor_sigma(10**12, 2)", "BudgetExceededError", "E-BUDGET-"),
    ("sigma_2e32", "m.divisor_sigma(2**32 + 1, 2)", "BudgetExceededError", "E-BUDGET-"),
    ("sigma_2e40", "m.divisor_sigma(2**40, 3)", "BudgetExceededError", "E-BUDGET-"),
    ("singleton_huge", "m.singleton_bound(2**40, 3, 2)", "BudgetExceededError", "E-BUDGET-"),
    ("hamming_bound_huge", "m.hamming_bound(2**40, 3, 2)", "BudgetExceededError", "E-BUDGET-"),
    (
        "hamming_code_huge_field",
        "m.LinearCode.hamming(m.FiniteField(4294967291), 2)",
        "CodingError",
        "E-CODE-004",
    ),
    ("sos_series", "m.sum_of_squares(6, 2**64 + 1)", "ArithmeticError", "E-NT-006"),
    ("cyclo_1e12", "m.cyclotomic_polynomial_coeffs(10**12)", "ArithmeticError", "E-NT-006"),
    ("cyclo_2e40", "m.cyclotomic_polynomial_coeffs(2**40)", "ArithmeticError", "E-NT-006"),
    (
        "nf_pow",
        "K = m.NumberField([-2, 0, 1]); (K.generator() + K.one()) ** (10**12)",
        "BudgetExceededError",
        "E-BUDGET-",
    ),
    (
        "nf_pow_2e40",
        "K = m.NumberField([-2, 0, 1]); (K.generator() + K.one()) ** (2**40)",
        "BudgetExceededError",
        "E-BUDGET-",
    ),
    (
        "nf_pow_neg",
        "K = m.NumberField([-2, 0, 1]); (K.generator() + K.one()) ** (-(10**12))",
        "BudgetExceededError",
        "E-BUDGET-",
    ),
    (
        "gf_zeros_budget",
        "F = m.FiniteField(2)\n"
        "with a.context(budget=a.Budget(max_bytes=10**8)):\n"
        "    m.GfMatrix.zeros(F, 65536, 65536)",
        "BudgetExceededError",
        "E-BUDGET-",
    ),
    (
        "gf_mul_budget",
        "F = m.FiniteField(7)\n"
        "c = m.GfMatrix.zeros(F, 65536, 1); r = m.GfMatrix.zeros(F, 1, 65536)\n"
        "with a.context(budget=a.Budget(max_bytes=10**8)):\n"
        "    c.mul(r)",
        "BudgetExceededError",
        "E-BUDGET-",
    ),
]


def _run(snippet: str) -> subprocess.CompletedProcess:
    code = (
        PRELUDE
        + "try:\n"
        + textwrap.indent(snippet, "    ")
        + "\nexcept Exception as exc:\n"
        + '    print("RAISED", type(exc).__name__, getattr(exc, "code", None))\n'
        + "else:\n"
        + '    print("RETURNED")\n'
    )
    env = dict(os.environ)
    env.setdefault("MALLOC_ARENA_MAX", "8")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


@pytest.mark.parametrize(
    ("snippet", "exc", "code"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_refused_not_aborted(snippet: str, exc: str, code: str) -> None:
    r = _run(snippet)
    if r.returncode < 0:
        pytest.fail(f"process killed by {signal.Signals(-r.returncode).name}: {r.stderr[-400:]}")
    assert r.returncode == 0, r.stderr[-800:]
    out = r.stdout.strip().splitlines()[-1]
    parts = out.split()
    assert parts[0] == "RAISED", out
    assert exc in parts[1] or parts[1].endswith(exc), out
    assert parts[2].startswith(code), out


LIMIT_AS = "import resource\nresource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is enforced on Linux")
@pytest.mark.parametrize(
    "snippet",
    [
        "m.GfMatrix.zeros(m.FiniteField(2), 65536, 65536)",
        "m.GfMatrix.zeros(m.FiniteField(7), 40000, 40000)",
        "m.GfMatrix.zeros(m.FiniteField(7, 3), 30000, 30000)",
    ],
    ids=["gf2_65536", "gf7_40000", "gf343_30000"],
)
def test_gf_matrix_under_an_address_space_limit(snippet: str) -> None:
    """Under ``ulimit -v 8G`` these shapes used to abort inside FLINT's allocator."""
    r = _run(LIMIT_AS + snippet)
    if r.returncode < 0:
        pytest.fail(f"process killed by {signal.Signals(-r.returncode).name}: {r.stderr[-400:]}")
    out = r.stdout.strip().splitlines()[-1]
    assert out.startswith("RAISED"), out
    # Which check fires first depends on the machine: E-BUDGET-005 when the
    # address-space limit is the tighter bound, E-BUDGET-006 when physical
    # memory is below it (e.g. a CI runner with less than 8 GB).
    assert "E-BUDGET-005" in out or "E-BUDGET-006" in out or "E-GFQ-012" in out, out


def test_factor_mod_p_keeps_the_unit() -> None:
    r = alkahest.factor_univariate_mod_p([5], 7)
    assert (r.unit, r.factor_list()) == (5, [])
    r = alkahest.factor_univariate_mod_p([0, 2], 7)
    assert (r.unit, r.factor_list()) == (2, [([0, 1], 1)])
    r = alkahest.factor_univariate_mod_p([1, 0, 1], 2)
    assert (r.unit, r.factor_list()) == (1, [([1, 1], 2)])


def test_factor_mod_p_zero_polynomial_raises() -> None:
    for coeffs in ([], [7], [0, 14]):
        with pytest.raises(alkahest.FactorError) as info:
            alkahest.factor_univariate_mod_p(coeffs, 7)
        assert info.value.code == "E-POLY-008"


def test_discrete_log_negative_residue() -> None:
    from alkahest import number_theory as nt

    # Signature is (residue, base, modulus): 2**2 = 4 ≡ -1 (mod 5).
    assert nt.discrete_log(-1, 2, 5) == 2
    assert nt.discrete_log(4, -3, 5) == 2
    assert nt.discrete_log(-16, 3, 17) == 0
    assert nt.nthroot_mod(-1, 2, 5) in (2, 3)


def test_root_of_unity_power_still_computed() -> None:
    K = alkahest.alkahest.NumberField.cyclotomic(5)
    z = K.generator()
    assert z ** (10**12 + 3) == z**3


def test_singleton_bound_not_truncated_to_u32() -> None:
    """The exponent used to be cast to ``u32``: ``2**(2**32 + 5)`` came out as ``2**5``."""
    r = _run("m.singleton_bound(2**32 + 5, 1, 2)")
    assert r.returncode == 0, r.stderr[-400:]
    out = r.stdout.strip().splitlines()[-1]
    parts = out.split()
    assert parts[0] == "RAISED", out
    assert parts[1].endswith("BudgetExceededError"), out
    assert parts[2].startswith("E-BUDGET-"), out
