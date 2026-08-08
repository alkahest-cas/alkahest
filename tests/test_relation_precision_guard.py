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
down `n` coefficients of magnitude `H` takes about `n·log10(H)` digits of
agreement, and when that exceeds the digits the inputs carry, the relation was
purchasable from the available precision and is evidence of nothing.

Testing the result rather than guessing the caller's intent is what lets
`[1.0, 2.0, 3.0]` keep working: that relation costs almost no precision to pin
down, so it is returned normally even though the inputs are floats.
"""

from __future__ import annotations

import math

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

    def test_relation_confidence_accepts_a_cheap_relation(self):
        verdict = ak.relation_confidence([1.0, 2.0, 3.0], [1, 1, -1])
        assert verdict["credible"] is True
        assert verdict["spare_digits"] > 0

    def test_exact_inputs_are_never_doubted(self):
        """Strings and ints spell exact rationals, so a relation among them holds."""
        for constants in ([PI_60, E_60], [1, 2, 3], ["1", "2", "3"]):
            verdict = ak.relation_confidence(constants, [10**9, -(10**9), 1])
            assert verdict["credible"] is True
            assert verdict["available_digits"] == math.inf
