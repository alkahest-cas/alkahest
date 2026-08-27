"""Integer-relation search must not manufacture relations from thin input.

``guess_relation`` accepts ``float`` values — 53 bits, ~16 significant digits —
while defaulting to ``precision_bits=664``, about 200 digits. The floats get
zero-padded to that width, which turns them into *exact rationals*, and exact
integer relations among exact rationals genuinely exist. The search dutifully
found them and returned them as if they were discoveries::

    guess_relation([float(pi), float(e), float(log(2))])
    ->  [-60771139, 67263243, 11653676]

Nothing about that result distinguishes it from a real find. The same three
constants supplied as 200-digit decimal strings correctly return ``None``.

In an autoresearch loop this is the poisoned-branch case the whole silent-error
discipline exists to prevent: stage 3 asks "is this constant a rational
combination of these?", gets a confident yes, and every later step inherits a
false lemma.

The guard judges the *relation that was found*, not the input alone: pinning
down `n` coefficients bounded by `H` picks one of the `(2H+1)ⁿ` integer vectors
in that box, so it takes about `n·log10(2H+1)` digits of agreement, and when that
exceeds the digits the inputs carry, the relation was purchasable from the
available precision and is evidence of nothing.

Testing the result rather than guessing the caller's intent is what lets
`[1.0, 2.0, 3.0]` keep working: that relation costs almost no precision to pin
down, so it is returned normally even though the inputs are floats.

`relation_confidence` used to answer this question for *every* input type, on
the premise that "decimal strings and ints are exactly the values they spell".
That premise is false for the one way PSLQ is actually driven — a decimal string
standing for a truncated numerical value — so the gate returned
``credible: True`` unconditionally on the input an experimental-mathematics loop
produces, including for three relations that re-evaluation at 60 digits refutes
(``temp-alkahest/testing/autoresearch-issues-2026-08-13.md`` §2). A gate that
cannot fail is worse than no gate, because loop authors wire it into promotion
logic. It now answers ``credible: None`` — *unknown* — unless the input's
precision is knowable (``float``) or declared (``digits=`` / ``precision_bits=``).

The 2026-08-19 run then found that the gate read the input's *type*, and that
four value-preserving conversions change the type without changing the value
(``autoresearch-issues-2026-08-19.md`` §4). ``mpmath.mpf`` reported the *ambient*
``mp.dps`` rather than anything about itself, so it now answers *unknown* like a
decimal string; and ``int``/``Fraction`` inputs, whose infinite ``available``
made ``credible: True`` unfalsifiable, are now decided by evaluating
``Σ aᵢ·cᵢ`` in exact arithmetic. ``guess_relation`` gained the ``digits=``
declaration it was missing, so the entry point that raises the refusal finally
has an escape hatch.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import alkahest as ak
import pytest

#: pi, e and log 2 to 60 significant digits — no integer relation is known.
PI_60 = "3.14159265358979323846264338327950288419716939937510582097494"
E_60 = "2.71828182845904523536028747135266249775724709369995957496697"
LOG2_60 = "0.693147180559945309417232121458176568075500134360255254120680"

#: sqrt(2) and twice it, to 60 digits — relation [-2, 1] holds exactly.
SQRT2_60 = "1.41421356237309504880168872420969807856967187537694806841301"
TWO_SQRT2_60 = "2.82842712474619009760337744841939615713934375075389613682602"

#: 60 digits is ~199 bits; search below that so the guard is satisfied.
BITS_60_DIGITS = 190

#: The three relations `guess_relation` returned during the 2026-08-13
#: autoresearch run that re-evaluation at 60 digits refutes, with the number of
#: constants and the digits of input precision each was bought with.
#: `relation_confidence` called every one of them ``credible: True``.
#:
#: Only the *count* of constants and the declared precision enter the verdict,
#: so the constants themselves are stand-ins of the right length; the run's
#: actual values are in the report.
SPURIOUS_RELATIONS = [
    (8, 20, [-19, -13, 28, 1, 26, -11, 20, -65]),
    (10, 25, [-46, 21, -24, -40, -25, 8, 31, 40, 14, 5]),
    (10, 20, [-18, -7, 2, -8, 1, -4, 4, 2, -1, 24]),
]


def _decimal_strings(n: int, digits: int) -> list[str]:
    """*n* distinct decimal strings carrying *digits* significant figures."""
    return [f"1.{str(k + 1) * digits}"[: digits + 2] for k in range(n)]


class TestRefusesUnderprecisedInput:
    def test_floats_at_default_precision_are_refused(self):
        """The reported silent error, now an honest refusal."""
        with pytest.raises(ak.PslqError) as excinfo:
            ak.guess_relation([float(math.pi), float(math.e), float(math.log(2))])
        assert excinfo.value.code == "E-PSLQ-004"

    def test_a_true_small_relation_is_not_collateral_damage(self):
        """`[0.5, 0.3333333, 1.0]` -> `[-2, 0, 1]` is *correct*, and must survive.

        `-2(0.5) + 0 + 1(1.0) = 0` holds exactly, and costs no precision to pin
        down. An earlier draft of this guard refused it, which was
        over-correction: the fault it hunts is a relation too *large* to be
        justified, not any relation among floats.
        """
        coeffs = ak.guess_relation([0.5, 0.3333333, 1.0])
        assert coeffs is not None
        assert sum(c * v for c, v in zip(coeffs, [0.5, 0.3333333, 1.0])) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_refusal_explains_what_to_do(self):
        """A refusal an agent cannot act on is barely better than a wrong answer."""
        with pytest.raises(ak.PslqError) as excinfo:
            ak.guess_relation([float(math.pi), float(math.e)])
        remediation = excinfo.value.remediation
        assert remediation is not None
        assert "decimal strings" in remediation

    def test_small_relations_among_floats_are_kept(self):
        """The criterion is the relation's size, not the input's type.

        `[1.0, 2.0, 3.0]` are floats, but `1 + 2 - 3 = 0` costs essentially no
        precision to pin down, so refusing it would be over-correction. This is
        the case that caught an earlier, intent-guessing version of the guard.
        """
        assert ak.guess_relation([1.0, 2.0, 3.0], precision_bits=384) is not None


class TestGenuinePrecisionIsUnaffected:
    def test_true_relation_is_still_found(self):
        coeffs = ak.guess_relation([SQRT2_60, TWO_SQRT2_60], precision_bits=BITS_60_DIGITS)
        assert coeffs is not None
        a, b = coeffs
        assert a * 2 + b * 1 == 0 or (a, b) == (-2, 1) or (a, b) == (2, -1)

    def test_unrelated_constants_still_return_none(self):
        """The correct answer for pi, e, log 2 — and the one the bug hid."""
        assert ak.guess_relation([PI_60, E_60, LOG2_60], precision_bits=BITS_60_DIGITS) is None

    def test_search_at_the_precision_actually_supplied(self):
        """Fewer digits is fine as long as the request matches the data."""
        coeffs = ak.guess_relation([SQRT2_60[:32], TWO_SQRT2_60[:32]], precision_bits=99)
        assert coeffs is not None


class TestOptOutAndReporting:
    def test_check_precision_false_restores_old_behaviour(self):
        """Deliberate opt-out for a relation among the supplied rationals themselves."""
        coeffs = ak.guess_relation([float(math.pi), float(math.e)], check_precision=False)
        assert coeffs is not None

    def test_relation_confidence_reports_the_shortfall(self):
        verdict = ak.relation_confidence([0.1, 0.2, 0.7], [60771139, 67263243, 11653676])
        assert verdict["credible"] is False
        assert verdict["available_digits"] == pytest.approx(16.0, abs=0.5)
        assert verdict["consumed_digits"] > verdict["available_digits"]
        assert verdict["precision_source"] == "float"

    def test_relation_confidence_accepts_a_cheap_relation(self):
        verdict = ak.relation_confidence([1.0, 2.0, 3.0], [1, 1, -1])
        assert verdict["credible"] is True
        assert verdict["spare_digits"] > 0

    def test_exact_inputs_are_not_doubted_on_precision_grounds(self):
        """An int *is* the rational it spells, so precision cannot be the fault.

        What decides the exact branch instead is whether the relation is *true*
        — see :class:`TestExactInputsAreEvaluatedNotAssumed`. A relation with
        coefficients far larger than the constants is fine as long as it holds.
        """
        verdict = ak.relation_confidence([1, 2, 3], [10**9, 10**9, -(10**9)])
        assert verdict["credible"] is True
        assert verdict["available_digits"] == math.inf
        assert verdict["precision_source"] == "exact"
        assert verdict["exact_residual"] == 0


class TestUnknownPrecisionIsNotAPass:
    """A decimal string may be exact or may be a truncation; nothing says which.

    The old contract assumed exact and answered ``credible: True``. The new one
    answers ``None`` and makes the caller say how many digits are trustworthy.
    """

    def test_decimal_strings_are_not_judged(self):
        verdict = ak.relation_confidence([PI_60, E_60], [10**9, -(10**9), 1])
        assert verdict["credible"] is None
        assert verdict["available_digits"] is None
        assert verdict["spare_digits"] is None
        assert verdict["precision_source"] == "unknown"

    @pytest.mark.parametrize(("n", "digits", "coeffs"), SPURIOUS_RELATIONS)
    def test_the_spurious_relations_are_no_longer_called_credible(self, n, digits, coeffs):
        """The three relations from the 2026-08-13 run, as they were produced.

        Each was returned by ``guess_relation`` over `n` decimal strings of
        `digits` digits, and each is refuted by re-evaluation at 60 digits. The
        old gate called all three ``credible: True``.
        """
        constants = _decimal_strings(n, digits)
        assert ak.relation_confidence(constants, coeffs)["credible"] is None

    @pytest.mark.parametrize(("n", "digits", "coeffs"), SPURIOUS_RELATIONS)
    def test_declaring_the_precision_refutes_them(self, n, digits, coeffs):
        """Told what the strings are worth, the gate fails — which is the point."""
        constants = _decimal_strings(n, digits)
        verdict = ak.relation_confidence(constants, coeffs, digits=digits)
        assert verdict["credible"] is False
        assert verdict["available_digits"] == pytest.approx(float(digits))
        assert verdict["precision_source"] == "declared"

    @pytest.mark.parametrize(("n", "digits", "coeffs"), SPURIOUS_RELATIONS)
    def test_the_same_relations_survive_at_200_digits(self, n, digits, coeffs):
        """Not a gate that refuses everything: 200 digits justifies all three."""
        constants = _decimal_strings(n, digits)
        assert ak.relation_confidence(constants, coeffs, digits=200)["credible"] is True

    def test_precision_bits_is_the_binary_spelling(self):
        verdict = ak.relation_confidence([PI_60, E_60], [1, -1], precision_bits=664)
        assert verdict["available_digits"] == pytest.approx(200.0, abs=0.5)
        assert verdict["credible"] is True

    def test_digits_and_precision_bits_are_mutually_exclusive(self):
        with pytest.raises(ValueError):
            ak.relation_confidence([PI_60, E_60], [1, -1], digits=60, precision_bits=199)

    def test_a_declaration_is_a_cap_not_an_override(self):
        """A float cannot hold 200 digits however loudly the caller declares it."""
        verdict = ak.relation_confidence([0.1, 0.2, 0.7], [1, 1, -1], digits=200)
        assert verdict["available_digits"] == pytest.approx(16.0, abs=0.5)
        assert verdict["precision_source"] == "float"

    def test_the_margin_is_what_catches_a_purchased_relation(self):
        """PSLQ returns the *smallest* relation the precision buys, so a
        purchased one lands just under ``consumed <= available`` rather than
        over it. ``margin_digits=0`` is the old, toothless criterion."""
        n, digits, coeffs = SPURIOUS_RELATIONS[0]
        constants = _decimal_strings(n, digits)
        assert ak.relation_confidence(constants, coeffs, digits=digits)["credible"] is False
        raw = ak.relation_confidence(constants, coeffs, digits=digits, margin_digits=0)
        assert raw["credible"] is True
        assert raw["consumed_digits"] < raw["available_digits"]

    def test_mpmath_values_are_unknown_until_their_accuracy_is_declared(self):
        """An ``mpf`` is a decimal string with a type on it.

        It used to report ``value.context.prec``, which is the *ambient*
        ``mp.dps`` when it is asked rather than anything about the value — see
        :class:`TestAmbientPrecisionCannotDecideTheVerdict`. Declaring the
        accuracy is what turns it back into a judgement.
        """
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workprec(80):  # ~24 digits
            constants = [+mpmath.pi, +mpmath.e]
            expensive = [5144503108, -5945642943]
            verdict = ak.relation_confidence(constants, expensive)
            assert verdict["precision_source"] == "unknown"
            assert verdict["available_digits"] is None
            assert verdict["credible"] is None
            declared = ak.relation_confidence(constants, expensive, digits=24)
            assert declared["precision_source"] == "declared"
            assert declared["credible"] is False
            assert ak.relation_confidence(constants, [1, -1], digits=24)["credible"] is True

    def test_one_known_input_can_refute_without_the_rest_being_known(self):
        """Unknown is not a licence to give up.

        Available precision is a ``min`` over the inputs, so a single ``float``
        among decimal strings caps the whole search at ~16 digits however many
        digits the strings carry. A relation that already costs more than that
        is refutable outright — reporting ``None`` here would be the same
        can't-fail shrug the gate was rewritten to remove.
        """
        expensive = [5144503108, -5945642943]  # ~19.5 digits, over the float cap
        verdict = ak.relation_confidence([0.1, PI_60], expensive)
        assert verdict["credible"] is False
        # Still honest about *how much* room there was: unknowable.
        assert verdict["available_digits"] is None
        assert verdict["spare_digits"] is None
        assert verdict["precision_source"] == "unknown"

    def test_a_known_input_only_refutes_when_it_actually_rules_the_relation_out(self):
        """The converse guard: the shortcut must not manufacture verdicts.

        The same mixed inputs with cheap coefficients stay ``None`` — 16 digits
        does not rule this relation out, and the strings' precision is still
        unknown, so no judgement is available either way.
        """
        verdict = ak.relation_confidence([0.1, PI_60], [1, -1])
        assert verdict["credible"] is None
        assert verdict["precision_source"] == "unknown"

    def test_guess_relation_still_returns_unjudged_string_relations(self):
        """Unknown precision must not become a refusal either — only a refusal
        to *endorse*. `guess_relation` is unchanged for decimal strings."""
        coeffs = ak.guess_relation([SQRT2_60, TWO_SQRT2_60], precision_bits=BITS_60_DIGITS)
        assert coeffs is not None
        assert ak.relation_confidence([SQRT2_60, TWO_SQRT2_60], coeffs)["credible"] is None
        assert (
            ak.relation_confidence([SQRT2_60, TWO_SQRT2_60], coeffs, digits=60)["credible"] is True
        )


# ---------------------------------------------------------------------------
# The 2026-08-19 autoresearch run, issue #4: four value-preserving conversions
# each switched this guard off, and the `exact` branch certified a relation that
# is false for the very numbers supplied.
# ---------------------------------------------------------------------------

#: The spurious relation the 2026-08-13 run produced from `float(pi)`,
#: `float(e)`, `float(log 2)` — the one this whole guard exists to refuse. It
#: holds exactly among the three *doubles*, because 664-bit zero-padding makes
#: them exact rationals; re-evaluated against pi, e and log 2 themselves at 300
#: digits the residual is ~4.8e-9, so it is not a relation among the constants
#: anyone meant to supply.
SPURIOUS_FLOAT_RELATION = [-60771139, 67263243, 11653676]


def _float_constants() -> list[float]:
    """`pi`, `e`, `log 2` as plain doubles — ~16 digits, however they print."""
    return [float(math.pi), float(math.e), float(math.log(2))]


def _exact_spellings() -> dict[str, list[Fraction]]:
    """The same three doubles, spelled as exact rationals three ordinary ways.

    Each is a conversion a caller writes without a second thought — and each one
    changes the input's *type* without changing what the caller believes the
    numbers are, routing them to ``precision_source: "exact"`` and an
    unfalsifiable ``credible: True``.
    """
    mpmath = pytest.importorskip("mpmath")
    floats = _float_constants()
    with mpmath.workdps(40):
        truncated = [
            Fraction(Decimal(mpmath.nstr(v, 20))) for v in (mpmath.pi, mpmath.e, mpmath.log(2))
        ]
    return {
        "Fraction(str(x))": [Fraction(str(x)) for x in floats],
        "Fraction(Decimal(repr(x)))": [Fraction(Decimal(repr(x))) for x in floats],
        "20-digit nstr truncation": truncated,
    }


class TestConversionsThatPreserveTheValueMustNotSwitchTheGuardOff:
    """The guard read the input's *type*, and ``mpf(x)`` changes the type without
    changing the value. At ``mp.dps = 300`` the lifted floats reported 301
    available digits — read off the ambient working precision — and the relation
    the release was written to refuse came back ``credible: True`` with 277
    "spare" digits.
    """

    def test_lifting_floats_to_mpf_does_not_buy_precision(self):
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(300):
            lifted = [mpmath.mpf(x) for x in _float_constants()]
            verdict = ak.relation_confidence(lifted, SPURIOUS_FLOAT_RELATION)
        assert verdict["credible"] is not True
        assert verdict["precision_source"] == "unknown"
        assert verdict["available_digits"] is None

    def test_guess_relation_does_not_endorse_the_lifted_floats(self):
        """It may still *return* the relation — unknown precision is not a
        refusal — but nothing downstream may read it as endorsed."""
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(300):
            lifted = [mpmath.mpf(x) for x in _float_constants()]
            coeffs = ak.guess_relation(lifted)
            assert coeffs is not None
            assert ak.relation_confidence(lifted, coeffs)["credible"] is not True

    def test_declaring_the_real_accuracy_refuses_the_lifted_floats(self):
        """The values came out of doubles, so 16 digits is the truth about them."""
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(300):
            lifted = [mpmath.mpf(x) for x in _float_constants()]
            assert (
                ak.relation_confidence(lifted, SPURIOUS_FLOAT_RELATION, digits=16)["credible"]
                is False
            )


class TestAmbientPrecisionCannotDecideTheVerdict:
    """``_supplied_bits`` used ``value.context.prec`` — the ambient ``mp.dps``
    at the moment of asking, not a property of the value. The same objects
    judged before and after an unrelated ``mp.dps = 300`` got opposite verdicts:
    ``False`` at 16 digits, ``True`` at 300, with nothing about the numbers
    changed. A gate whose answer moves with a global in another library is not
    reporting evidence.
    """

    def test_the_verdict_does_not_move_with_mp_dps(self):
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(16):
            constants = [+mpmath.pi, +mpmath.e, +mpmath.log(2)]
            low = ak.relation_confidence(constants, SPURIOUS_FLOAT_RELATION)
        with mpmath.workdps(300):
            high = ak.relation_confidence(constants, SPURIOUS_FLOAT_RELATION)
        assert low["credible"] == high["credible"]
        assert low["available_digits"] == high["available_digits"]
        assert low["precision_source"] == high["precision_source"]
        assert high["credible"] is not True

    def test_only_a_declaration_moves_it_and_it_moves_the_same_way(self):
        mpmath = pytest.importorskip("mpmath")
        for dps in (16, 300):
            with mpmath.workdps(dps):
                constants = [+mpmath.pi, +mpmath.e, +mpmath.log(2)]
            with mpmath.workdps(300):
                verdict = ak.relation_confidence(constants, SPURIOUS_FLOAT_RELATION, digits=16)
            assert verdict["credible"] is False, f"computed at dps={dps}"


class TestExactInputsAreEvaluatedNotAssumed:
    """``precision_source: "exact"`` was an unfalsifiable pass.

    ``available_digits`` is ``inf`` on that branch, so no affordability test can
    fire and ``credible`` was ``True`` for *any* coefficients — including a
    relation whose exact residual is not zero. Precision genuinely cannot be the
    fault for an exact rational, but *arithmetic* can be, and one line of
    ``Fraction`` arithmetic settles it. This is what makes the exact branch
    falsifiable for the first time.
    """

    @pytest.mark.parametrize("spelling", ["Fraction(str(x))", "Fraction(Decimal(repr(x)))"])
    def test_a_false_relation_among_exact_rationals_is_refuted(self, spelling):
        constants = _exact_spellings()[spelling]
        residual = sum(Fraction(a) * c for a, c in zip(SPURIOUS_FLOAT_RELATION, constants))
        assert residual != 0, "the premise: this relation is false for these very numbers"
        verdict = ak.relation_confidence(constants, SPURIOUS_FLOAT_RELATION)
        assert verdict["credible"] is False
        assert verdict["precision_source"] == "exact"
        assert verdict["exact_residual"] == residual

    def test_a_truncated_decimal_read_back_as_a_fraction_is_refuted(self):
        """`Fraction(Decimal(nstr(pi, 20)))` is exactly a 20-digit truncation —
        an exact rational that is *not* the constant it stands for, and the
        relation found among such truncations does not hold for them."""
        constants = _exact_spellings()["20-digit nstr truncation"]
        with pytest.raises(ak.PslqError) as excinfo:
            ak.guess_relation(constants)
        assert excinfo.value.code == "E-PSLQ-005"
        assert "false for the values supplied" in str(excinfo.value)
        coeffs = ak.guess_relation(constants, check_precision=False)
        verdict = ak.relation_confidence(constants, coeffs)
        assert verdict["credible"] is False
        assert verdict["exact_residual"] != 0

    def test_the_refusal_says_what_to_do_about_it(self):
        constants = _exact_spellings()["Fraction(str(x))"]
        with pytest.raises(ak.PslqError) as excinfo:
            ak.guess_relation(constants)
        assert excinfo.value.code == "E-PSLQ-005"
        assert "digits=" in excinfo.value.remediation

    def test_a_true_relation_among_exact_rationals_still_passes(self):
        """The control. Refusing every exact input would be the same
        can't-fail gate with the sign flipped.

        Dyadic denominators so that the search — which reaches the constants
        through ``f64`` — sees the same numbers the verdict is computed from.
        """
        constants = [Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)]
        verdict = ak.relation_confidence(constants, [1, 1, -1])
        assert verdict["credible"] is True
        assert verdict["exact_residual"] == 0
        coeffs = ak.guess_relation(constants)
        assert coeffs is not None
        assert sum(Fraction(a) * c for a, c in zip(coeffs, constants)) == 0

    def test_a_declaration_turns_the_exact_check_off(self):
        """``digits=`` says "treat these rationals as approximations", so the
        affordability test decides and no exact residual is reported."""
        constants = _exact_spellings()["Fraction(str(x))"]
        verdict = ak.relation_confidence(constants, SPURIOUS_FLOAT_RELATION, digits=200)
        assert verdict["exact_residual"] is None
        assert verdict["precision_source"] == "declared"
        assert verdict["credible"] is True


class TestTheKnownWrongFixStaysOut:
    """`min(bitcount(mantissa), context.prec)` was the finder's proposed fix.

    It stops the attack and reports **0.30 digits** for ``mpf(1), mpf(2),
    mpf(3)``, flipping the true relation ``[1, 1, -1]`` to ``credible: False``.
    Every ``mpf`` is exactly a dyadic rational, so accuracy is not recoverable
    from the object and a mantissa bitcount measures how *round* a number is,
    not how accurate. A gate that refutes a true relation is a worse failure
    than one that shrugs at a false one, so this is an explicit guard: the
    verdict here may be ``True`` or unknown, and must never be ``False``.
    """

    def test_mpf_one_two_three_is_not_refuted(self):
        mpmath = pytest.importorskip("mpmath")
        constants = [mpmath.mpf(1), mpmath.mpf(2), mpmath.mpf(3)]
        verdict = ak.relation_confidence(constants, [1, 1, -1])
        assert verdict["credible"] is not False
        # Unknown, on the same grounds as a decimal string: nothing in an `mpf`
        # says how many of its digits mean anything.
        assert verdict["credible"] is None
        assert ak.guess_relation(constants) is not None

    def test_declaring_any_plausible_precision_makes_it_credible(self):
        """And the escape hatch confirms the relation really is cheap: 3
        coefficients bounded by 1 cost 1.4 digits, so any declaration from ~12
        digits up clears the 10-digit margin."""
        mpmath = pytest.importorskip("mpmath")
        constants = [mpmath.mpf(1), mpmath.mpf(2), mpmath.mpf(3)]
        for digits in (16, 50, 300):
            assert ak.relation_confidence(constants, [1, 1, -1], digits=digits)["credible"] is True

    def test_the_integer_spelling_of_the_same_relation_is_credible(self):
        """`[1, 2, 3]` as ints has no precision question at all, and the exact
        evaluation added for issue #4 confirms the relation rather than
        assuming it."""
        verdict = ak.relation_confidence([1, 2, 3], [1, 1, -1])
        assert verdict["credible"] is True
        assert verdict["exact_residual"] == 0


class TestTheCostFormulaDoesNotCollapseAtUnitCoefficients:
    """Item 26n. ``n·log10(H)`` is 0 for every relation with ``H = 1``, however
    many constants it spans, so a 40-term ±1 relation was free. The count is
    ``(2H+1)ⁿ`` — coefficients run over ±H *and zero* — so the cost is
    ``n·log10(2H+1)``, which is ``n·log10(3)`` at ``H = 1``.
    """

    @pytest.mark.parametrize("n", [2, 3, 8, 40])
    def test_unit_coefficients_are_not_free(self, n):
        coeffs = [1 if k % 2 else -1 for k in range(n)]
        consumed = ak.relation_confidence(_decimal_strings(n, 60), coeffs)["consumed_digits"]
        assert consumed == pytest.approx(n * math.log10(3))
        assert consumed > 0

    def test_a_long_unit_relation_is_refused_at_low_precision(self):
        """The end the collapsed formula could not reach: 40 ±1 coefficients
        cost ~19 digits, which 16 digits of float cannot buy."""
        n = 40
        coeffs = [1 if k % 2 else -1 for k in range(n)]
        verdict = ak.relation_confidence([0.1 * k + 0.5 for k in range(n)], coeffs)
        assert verdict["consumed_digits"] == pytest.approx(40 * math.log10(3))
        assert verdict["credible"] is False

    def test_the_general_form_is_still_the_box_count(self):
        n, h = 5, 65
        coeffs = [h, -h, 1, 0, -1]
        consumed = ak.relation_confidence(_decimal_strings(n, 60), coeffs)["consumed_digits"]
        assert consumed == pytest.approx(n * math.log10(2 * h + 1))


class TestGuessRelationHasAnEscapeHatch:
    """`digits=` rescued `relation_confidence`, but `guess_relation` — the entry
    point that actually raises `E-PSLQ-004` — had no way to say what the inputs
    are worth: its `precision_bits` is the width of the *search*. A caller who
    genuinely knew their input precision had no move except turning the guard
    off entirely with `check_precision=False`.
    """

    def test_digits_lets_a_high_precision_caller_through(self):
        """200-digit strings judged as such: the relation is affordable and is
        returned, where without the declaration it comes back unjudged."""
        coeffs = ak.guess_relation([SQRT2_60, TWO_SQRT2_60], precision_bits=BITS_60_DIGITS)
        assert coeffs is not None
        assert (
            ak.guess_relation([SQRT2_60, TWO_SQRT2_60], precision_bits=BITS_60_DIGITS, digits=60)
            == coeffs
        )

    def test_digits_makes_the_guard_fire_on_otherwise_unjudged_input(self):
        """The other direction, and the one that matters: declaring the real
        accuracy of `mpf` values turns an unjudged return into `E-PSLQ-004`."""
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(300):
            lifted = [mpmath.mpf(x) for x in _float_constants()]
            assert ak.guess_relation(lifted) is not None
            with pytest.raises(ak.PslqError) as excinfo:
                ak.guess_relation(lifted, digits=16)
        assert excinfo.value.code == "E-PSLQ-004"

    def test_digits_refuses_a_purchased_string_relation(self):
        """The case the docstring used to send the caller elsewhere for.

        pi, e and log 2 truncated to 20 digits buy a 10-digit-per-coefficient
        relation at the 664-bit default. Undeclared, `guess_relation` returns it
        unjudged — a decimal string may be exact — and `digits=20` is how the
        caller says it is not, in the one call that used to require a second.
        """
        constants = [PI_60[:21], E_60[:21], LOG2_60[:22]]
        unjudged = ak.guess_relation(constants)
        assert unjudged is not None
        assert ak.relation_confidence(constants, unjudged)["credible"] is None
        with pytest.raises(ak.PslqError) as excinfo:
            ak.guess_relation(constants, digits=20)
        assert excinfo.value.code == "E-PSLQ-004"
        assert "purchasable" in str(excinfo.value)

    def test_digits_is_keyword_only_so_it_cannot_be_confused_with_the_search_width(self):
        with pytest.raises(TypeError):
            ak.guess_relation([1.0, 2.0, 3.0], 384, None, True, 16)

    def test_digits_and_precision_bits_do_not_collide(self):
        """`precision_bits` still means the search width on `guess_relation`, so
        passing both is not the mutual-exclusion error `relation_confidence`
        raises — they are different quantities here."""
        coeffs = ak.guess_relation([1.0, 2.0, 3.0], precision_bits=384, digits=16)
        assert coeffs is not None
