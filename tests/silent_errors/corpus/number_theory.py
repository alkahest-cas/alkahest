"""Silent-error cases for number theory.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

from typing import Any, Callable

import alkahest as ak
import alkahest.number_theory as nt
from contracts import Case, Raises, RefusesOr, Returns

#: Apéry A005259, index-shifted to `Σ_i a_i(n)·A(n+i) = 0`:
#: `(n+2)³A(n+2) − (34n³+153n²+231n+117)A(n+1) + (n+1)³A(n) = 0`. Built once —
#: construction is cheap, but the corpus is a hot path on every pull request.
_APERY_MOD = ak.ModularRecurrence(
    [[1, 3, 3, 1], [-117, -231, -153, -34], [8, 12, 6, 1]],
    [1, 5],
)


def _relation_residual(values: list[int]) -> Callable[[], int]:
    """Answer = the *exact* integer residual ``Σ aᵢ·valuesᵢ`` of the relation
    ``guess_relation`` reports.  Zero, or nothing at all, are the only honest
    answers; any other integer means the reported "relation" is not one."""

    def op() -> int:
        coeffs = ak.guess_relation(values)
        if coeffs is None:
            raise ak.PslqError("guess_relation reported no relation")
        return sum(a * v for a, v in zip(coeffs, values))

    return op


#: Eight 20-digit decimal strings and the relation ``guess_relation`` returned
#: over eight such constants during the 2026-08-13 autoresearch run
#: (``temp-alkahest/testing/autoresearch-issues-2026-08-13.md`` §2).
#: Re-evaluating that relation at 60 digits gives 3.59e-14 — it is noise bought
#: with 20 digits, and ``relation_confidence`` called it credible.  Only the
#: *count* of constants and the declared precision enter the verdict, so these
#: stand in for the run's values at the same length.
_PURCHASED_20_DIGIT_CONSTANTS = [f"1.{str(k + 1) * 20}"[:22] for k in range(8)]
_PURCHASED_20_DIGIT_COEFFS = [-19, -13, 28, 1, 26, -11, 20, -65]
#: ``relation_confidence``'s three-valued verdict as a word, so that *unknown*
#: is scored as its own answer rather than collapsing into a truthy pass.
_CONFIDENCE_WORD = {True: "credible", False: "purchasable", None: "unknown"}


def _relation_verdict(constants: list, coeffs: list, **kwargs: Any) -> Callable[[], str]:
    """Answer = ``relation_confidence``'s verdict on a *found* relation:
    ``"credible"``, ``"purchasable"``, or ``"unknown"`` when the inputs' own
    precision is not knowable."""

    def op() -> str:
        return _CONFIDENCE_WORD[ak.relation_confidence(constants, coeffs, **kwargs)["credible"]]

    return op


CASES: list[Case] = [
    # Number theory at 0, 1, negatives, and the pseudoprime traps.
    # -----------------------------------------------------------------------
    Case(
        id="nt_isprime_one",
        subsystem="number_theory",
        statement="1 is not prime",
        op=lambda: nt.isprime(1),
        contract=Returns(False),
        verified_by="A prime has exactly two distinct positive divisors; 1 has one. Excluding 1 "
        "is what makes factorisation unique.",
    ),
    Case(
        id="nt_isprime_two",
        subsystem="number_theory",
        statement="2 is prime",
        op=lambda: nt.isprime(2),
        contract=Returns(True),
        verified_by="Divisors 1 and 2. The control for the edge cases: 'always False' must fail.",
    ),
    Case(
        id="nt_isprime_zero",
        subsystem="number_theory",
        statement="0 is not prime",
        op=lambda: nt.isprime(0),
        contract=RefusesOr(False),
        verified_by="0 has infinitely many divisors.",
        note="alkahest refuses with E-NT-002 (expects a positive integer) rather than "
        "answering False; both are safe.",
    ),
    Case(
        id="nt_isprime_negative",
        subsystem="number_theory",
        statement="-7 is not prime under the standard positive-integer definition",
        op=lambda: nt.isprime(-7),
        contract=RefusesOr(False),
        verified_by="Primality is defined on integers > 1. (-7 is a prime *element* of ℤ, which "
        "is a different statement and must not be conflated.)",
        note="alkahest refuses with E-NT-002.",
    ),
    Case(
        id="nt_isprime_carmichael_561",
        subsystem="number_theory",
        statement="561 = 3·11·17 is composite despite passing the Fermat test for every "
        "coprime base",
        op=lambda: nt.isprime(561),
        contract=Returns(False),
        verified_by="561 = 3·11·17; it is the smallest Carmichael number, so a^560 ≡ 1 (mod "
        "561) for every a coprime to 561. A Fermat-test primality check answers True.",
    ),
    Case(
        id="nt_isprime_strong_pseudoprime_2047",
        subsystem="number_theory",
        statement="2047 = 23·89 is composite despite being a strong pseudoprime to base 2",
        op=lambda: nt.isprime(2047),
        contract=Returns(False),
        verified_by="2047 = 2¹¹-1 = 23·89. It is the smallest strong pseudoprime to base 2, so "
        "a single-base Miller–Rabin with a=2 answers True.",
    ),
    Case(
        id="nt_factorint_zero",
        subsystem="number_theory",
        statement="0 has no prime factorisation",
        op=lambda: nt.factorint(0),
        contract=Raises("E-NT-002"),
        verified_by="0 is divisible by every prime to every power; no finite factorisation exists.",
    ),
    Case(
        id="nt_factorint_one",
        subsystem="number_theory",
        statement="1 factors as the empty product",
        op=lambda: nt.factorint(1),
        contract=Returns({}),
        verified_by="1 is the empty product. Reporting {1: 1} would break unique factorisation.",
    ),
    Case(
        id="nt_totient_one",
        subsystem="number_theory",
        statement="φ(1) = 1",
        op=lambda: nt.totient(1),
        contract=Returns(1),
        verified_by="The only k in [1,1] with gcd(k,1)=1 is k=1. Πp|n(1-1/p) over an empty "
        "prime set gives 1, so the formula agrees.",
    ),
    Case(
        id="nt_totient_zero",
        subsystem="number_theory",
        statement="φ(0) is undefined",
        op=lambda: nt.totient(0),
        contract=RefusesOr(),
        verified_by="Euler's totient is defined on positive integers.",
        note="alkahest refuses with E-NT-002.",
    ),
    Case(
        id="nt_totient_negative",
        subsystem="number_theory",
        statement="φ(-5) is undefined",
        op=lambda: nt.totient(-5),
        contract=RefusesOr(),
        verified_by="Euler's totient is defined on positive integers.",
        note="alkahest refuses with E-NT-002.",
    ),
    Case(
        id="nt_jacobi_even_denominator",
        subsystem="number_theory",
        statement="the Jacobi symbol (2/4) is undefined — the denominator must be odd",
        op=lambda: nt.jacobi_symbol(2, 4),
        contract=Raises("E-NT-002"),
        verified_by="(a/n) is defined as a product of Legendre symbols over the odd prime "
        "factorisation of n; even n has no such factorisation.",
    ),
    Case(
        id="nt_nthroot_mod_non_residue",
        subsystem="number_theory",
        statement="3 is a quadratic non-residue mod 7, so √3 mod 7 does not exist",
        op=lambda: nt.nthroot_mod(3, 2, 7),
        contract=Raises("E-NT-003"),
        verified_by="Squares mod 7 are {1,2,4} (1²=1, 2²=4, 3²=2). 3 is not among them.",
    ),
    Case(
        id="nt_discrete_log_no_solution",
        subsystem="number_theory",
        statement="3 is not a power of 2 mod 7, so log_2 3 mod 7 does not exist",
        op=lambda: nt.discrete_log(3, 2, 7),
        contract=Raises("E-NT-003"),
        verified_by="⟨2⟩ = {2,4,1} mod 7 has order 3 and does not contain 3.",
    ),
    Case(
        id="nt_discrete_log_control",
        subsystem="number_theory",
        statement="3² ≡ 2 (mod 7), so log_3 2 mod 7 = 2",
        op=lambda: nt.discrete_log(2, 3, 7),
        contract=Returns(2),
        verified_by="3² = 9 ≡ 2 (mod 7). The control for nt_discrete_log_no_solution.",
    ),
    Case(
        id="nt_nextprime_one",
        subsystem="number_theory",
        statement="the smallest prime greater than 1 is 2",
        op=lambda: nt.nextprime(1),
        contract=Returns(2),
        verified_by="2 is the smallest prime.",
    ),
    # ── integer relations ───────────────────────────────────────────────────
    Case(
        id="pslq_exact_integer_inputs_are_not_rounded",
        subsystem="number_theory",
        statement="guess_relation([2⁶⁰+1, 2⁶⁰, 1]) must report a relation that actually holds",
        op=_relation_residual([2**60 + 1, 2**60, 1]),
        contract=Returns(0),
        verified_by=(
            "-(2⁶⁰+1) + 2⁶⁰ + 1 = 0 exactly, so [-1, 1, 1] is a relation. alkahest returned "
            "[-1, 1, 0], whose residual over the values supplied is -1, and relation_confidence "
            "called it credible with available_digits = inf: the binding extracted every Python "
            "int through f64 first, discarding the low bit that the guard then assumed was exact."
        ),
    ),
    Case(
        id="pslq_control_small_rational_relation",
        subsystem="number_theory",
        statement="guess_relation([1, 2, 3]) must find a genuine relation among exact integers",
        op=_relation_residual([1, 2, 3]),
        contract=Returns(0),
        verified_by=(
            "1, 2, 3 are integers, so integer relations certainly exist (e.g. [1, 1, -1]). The "
            "control for pslq_exact_integer_inputs_are_not_rounded: refusing every integer input "
            "must not pass the gate."
        ),
    ),
    Case(
        id="pslq_confidence_declared_precision_refutes_a_purchased_relation",
        subsystem="number_theory",
        statement=(
            "relation_confidence must not call 8 coefficients of size ≤ 65 credible when the "
            "inputs carry 20 digits"
        ),
        op=_relation_verdict(_PURCHASED_20_DIGIT_CONSTANTS, _PURCHASED_20_DIGIT_COEFFS, digits=20),
        contract=Returns("purchasable"),
        verified_by=(
            "A counting argument, independent of alkahest: there are ~H^n = 65^8 ≈ 10^14.5 integer "
            "vectors with |aᵢ| ≤ 65, and the smallest |Σ aᵢcᵢ| among them is ~10^-14.5 for *any* 8 "
            "constants. At 20 digits such a vector is therefore always findable and is evidence of "
            "nothing. These are the coefficients guess_relation returned on 2026-08-13 over 8 "
            "constants at 20 digits; re-evaluating that relation at 60 digits gives 3.59e-14, the "
            "noise floor the counting argument predicts, not the ~1e-20 a true relation would give."
        ),
    ),
    Case(
        id="pslq_confidence_unknown_precision_is_not_a_pass",
        subsystem="number_theory",
        statement=(
            "relation_confidence must answer 'unknown', not 'credible', when the inputs are "
            "decimal strings of undeclared accuracy"
        ),
        op=_relation_verdict(_PURCHASED_20_DIGIT_CONSTANTS, _PURCHASED_20_DIGIT_COEFFS),
        contract=Returns("unknown"),
        verified_by=(
            "A decimal string is both an exact rational and the way a truncated constant is "
            "spelled, and the string does not say which — so no verdict is derivable from it. The "
            "old contract assumed exact and answered 'credible' for every relation among strings, "
            "which is the input a PSLQ loop actually produces."
        ),
        note="The gate's own shape: 'unknown' is the honest answer, and must not read as a pass.",
    ),
    Case(
        id="pslq_control_confidence_credible_at_full_precision",
        subsystem="number_theory",
        statement=(
            "relation_confidence must still call the same relation credible when the inputs "
            "carry 200 digits"
        ),
        op=_relation_verdict(_PURCHASED_20_DIGIT_CONSTANTS, _PURCHASED_20_DIGIT_COEFFS, digits=200),
        contract=Returns("credible"),
        verified_by=(
            "The control for the two cases above: 8 coefficients of size ≤ 65 cost ~14.5 digits, "
            "so 200 digits of agreement is ~185 digits more than the relation could have been "
            "bought with. A gate that answered 'purchasable' here would be passing by refusing "
            "everything."
        ),
    ),
    # M6 — a holonomic sequence mod p^k, at the indices where the recurrence
    # cannot simply be divided through.
    # -----------------------------------------------------------------------
    Case(
        id="modular_recurrence_through_a_singular_index",
        subsystem="number_theory",
        statement="A(13) mod 13³ = 5, reached only by dividing by (n+2)³ = 13³ at n = 11",
        op=lambda: _APERY_MOD.value_mod(13, 13, 3),
        contract=Returns(5),
        verified_by=(
            "A(13) = Σ_k C(13,k)²·C(13+k,k)² was summed as an exact Python integer from the "
            "definition and reduced mod 13³, independently of any recurrence. The recurrence "
            "route must cross n = 11, where the leading coefficient (n+2)³ is exactly 13³ and "
            "has no inverse mod 13³ at all."
        ),
        note=(
            "The classic shape: dividing by a non-unit. A modular inverse routine that returns "
            "something for a non-unit — or a Fermat-style pow(a, m-2, m), which is wrong for a "
            "prime *power* — produces a plausible residue here with no error of any kind. "
            "alkahest measures v_p of the leading coefficient first and runs the forward pass "
            "at 13⁶ so the three lost digits are ones it had already bought."
        ),
    ),
    Case(
        id="modular_recurrence_refuses_a_vanishing_leading_coefficient",
        subsystem="number_theory",
        statement="(n−4)·S(n+1) = S(n) determines no S(5), at any modulus",
        op=lambda: ak.ModularRecurrence([[-1], [-4, 1]], [1]).value_mod(5, 7, 3),
        contract=Raises("E-HOLO-007"),
        verified_by=(
            "At n = 4 the relation reads 0·S(5) = S(4) with S(4) = 1/24 ≠ 0, so no S(5) "
            "satisfies it — over ℤ, over ℚ, or over ℤ/7³. There is no right answer to return, "
            "and a larger modulus does not create one."
        ),
        note=(
            "The step before it, S(4) = 1/((−4)(−3)(−2)(−1)) = 1/24, is ordinary and is "
            "answered; only the undetermined one refuses. A gate that refused the whole "
            "recurrence would pass this case for the wrong reason."
        ),
    ),
    Case(
        id="binomial_mod_prime_power_far_above_the_prime",
        subsystem="number_theory",
        statement="binomial(2p−1, p−1) ≡ 1 (mod p³) at p = 101 — Wolstenholme",
        op=lambda: ak.binomial_mod(201, 100, 101, 3),
        contract=Returns(1),
        verified_by=(
            "Wolstenholme's theorem, and separately math.comb(201, 100) % 101**3 = 1. The "
            "argument is far larger than p, so Lucas' theorem alone (a mod-p statement) cannot "
            "reach it; the prime-power machinery has to."
        ),
        note=(
            "The trap is answering the mod-p question and labelling it mod-p³: Lucas gives "
            "1 here too, and a k>1 implementation that silently degrades to k=1 agrees with "
            "the truth on exactly this input while being wrong on most others."
        ),
    ),
]
