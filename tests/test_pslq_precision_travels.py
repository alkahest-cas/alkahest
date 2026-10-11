"""W7 (report 10-9): integer-relation search must run at the precision its inputs carry.

Three faults, one root cause — the precision of the *search* was decoupled from
the precision of the *data*:

* ``mpmath.mpf`` values reached the engine through ``float()``: a 120-digit
  ``ζ(4)`` was searched as a double and came back with a junk relation
  ``[-54836443260404016, 609293814004489]`` (residual −6.49), which
  ``relation_confidence(..., digits=110)`` then called ``credible``.
* 115-digit decimal strings were searched at the 664-bit (200-digit) default,
  i.e. on 85 digits of zero padding: ``90·ζ(4) = π⁴`` came back ``None`` and the
  degree-6 minimal polynomial search raised a raw ``OverflowError``.
* Coefficients were narrowed to ``i64``, so a large relation escaped as
  ``OverflowError: coefficient overflows i64`` instead of reaching the guard.

Every expected relation below is checked against mpmath at a higher precision
than the search ran at, so the oracle is independent of alkahest.
"""

from __future__ import annotations

import contextlib
import math

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")

PRECISIONS = [30, 60, 115, 250]


def _normalise(coeffs):
    """Sign-normalise so the last nonzero coefficient is positive."""
    if coeffs is None:
        return None
    lead = next(c for c in reversed(coeffs) if c != 0)
    return [-c for c in coeffs] if lead < 0 else list(coeffs)


def _holds(values_fn, coeffs, dps):
    """Oracle: ``Σ aᵢ·cᵢ`` at ``dps + 40`` digits is below ``10^-(dps-5)``·scale."""
    with mpmath.workdps(dps + 40):
        vals = values_fn()
        total = mpmath.fsum(a * v for a, v in zip(coeffs, vals))
        scale = mpmath.fsum(abs(a * v) for a, v in zip(coeffs, vals))
        return abs(total) <= scale * mpmath.mpf(10) ** (-(dps - 5))


def _zeta4_pi4():
    return [mpmath.zeta(4), mpmath.pi**4]


def _machin():
    return [mpmath.pi, mpmath.atan(mpmath.mpf(1) / 5), mpmath.atan(mpmath.mpf(1) / 239)]


def _alpha_powers():
    alpha = mpmath.cbrt(2) + mpmath.sqrt(3)
    return [alpha**k for k in range(7)]


def _li2_half():
    return [mpmath.polylog(2, mpmath.mpf(1) / 2), mpmath.pi**2, mpmath.log(2) ** 2]


#: (name, values, expected sign-normalised relation, smallest precision tested)
IDENTITIES = [
    ("90ζ(4)=π⁴", _zeta4_pi4, [-90, 1], 30),
    ("Machin", _machin, [1, -16, 4], 30),
    ("Li₂(1/2)", _li2_half, [12, -1, 6], 30),
    ("minpoly 2^(1/3)+√3", _alpha_powers, [-23, -36, 27, -4, -9, 0, 1], 30),
]


def _spell(kind, values, dps):
    if kind == "mpf":
        return list(values)
    return [mpmath.nstr(v, dps, strip_zeros=False) for v in values]


@pytest.mark.parametrize("kind", ["mpf", "str"])
@pytest.mark.parametrize("dps", PRECISIONS)
@pytest.mark.parametrize(("name", "values_fn", "expected", "min_dps"), IDENTITIES)
class TestKnownIdentitiesAtEveryPrecision:
    def test_found_at_default_settings(self, name, values_fn, expected, min_dps, dps, kind):
        if dps < min_dps:
            pytest.skip(f"{name} needs more than {dps} digits")
        assert _holds(values_fn, expected, dps), "oracle: the identity itself"
        with mpmath.workdps(dps):
            constants = _spell(kind, values_fn(), dps)
        assert _normalise(ak.guess_relation(constants)) == expected

    def test_found_and_endorsed_with_digits(self, name, values_fn, expected, min_dps, dps, kind):
        if dps < min_dps:
            pytest.skip(f"{name} needs more than {dps} digits")
        with mpmath.workdps(dps):
            constants = _spell(kind, values_fn(), dps)
        coeffs = ak.guess_relation(constants, digits=dps - 2)
        assert _normalise(coeffs) == expected
        verdict = ak.relation_confidence(constants, coeffs, digits=dps - 2)
        assert verdict["credible"] is True
        assert verdict["excess_digits"] >= verdict["margin_digits"]


class TestTheReportedRepros:
    def test_mpf_is_not_searched_as_a_double(self):
        with mpmath.workdps(120):
            constants = _zeta4_pi4()
        assert _normalise(ak.guess_relation(constants)) == [-90, 1]
        assert _normalise(ak.guess_relation(constants, digits=110)) == [-90, 1]

    def test_the_junk_relation_is_not_credible(self):
        """The relation the float path returned; its residual is −6.49."""
        junk = [-54836443260404016, 609293814004489]
        with mpmath.workdps(120):
            constants = _zeta4_pi4()
            residual = junk[0] * constants[0] + junk[1] * constants[1]
        assert abs(residual) > 1, "oracle: the relation is false"
        verdict = ak.relation_confidence(constants, junk, digits=110)
        assert verdict["credible"] is False
        assert verdict["residual_digits"] < 20
        assert verdict["excess_digits"] < 0

    def test_115_digit_strings_find_zeta4_and_the_minimal_polynomial(self):
        with mpmath.workdps(120):
            z = [mpmath.nstr(v, 115, strip_zeros=False) for v in _zeta4_pi4()]
            a = [mpmath.nstr(v, 115, strip_zeros=False) for v in _alpha_powers()]
        assert _normalise(ak.guess_relation(z, digits=115)) == [-90, 1]
        assert _normalise(ak.guess_relation(a, digits=115)) == [-23, -36, 27, -4, -9, 0, 1]

    def test_an_explicit_search_width_is_capped_too(self):
        """``precision_bits`` above the data's precision is the same zero-padding."""
        with mpmath.workdps(65):
            z = [mpmath.nstr(v, 60, strip_zeros=False) for v in _zeta4_pi4()]
        assert _normalise(ak.guess_relation(z, precision_bits=2000)) == [-90, 1]


class TestNegativeControls:
    @pytest.mark.parametrize("dps", PRECISIONS)
    @pytest.mark.parametrize("kind", ["mpf", "str"])
    def test_catalan_pi_squared_log2_has_no_relation(self, dps, kind):
        with mpmath.workdps(dps):
            constants = _spell(kind, [mpmath.catalan, mpmath.pi**2, mpmath.log(2)], dps)
        assert ak.guess_relation(constants) is None
        assert ak.guess_relation(constants, digits=dps - 2) is None

    @pytest.mark.parametrize("dps", PRECISIONS)
    def test_a_purchased_relation_among_them_is_not_credible(self, dps):
        """Force a relation out of them — read the strings back as exact
        Fractions (which carry no precision cap) and search 2.5× wider — and
        the gate must refuse it at the precision the strings actually carry."""
        from decimal import Decimal
        from fractions import Fraction

        with mpmath.workdps(dps + 5):
            strings = [
                mpmath.nstr(v, dps, strip_zeros=False)
                for v in (mpmath.catalan, mpmath.pi**2, mpmath.log(2))
            ]
        exact = [Fraction(Decimal(s)) for s in strings]
        junk = ak.guess_relation(exact, precision_bits=int(dps * 3.33 * 2.5), check_precision=False)
        assert junk is not None, "exact rationals always have integer relations"
        assert sum(Fraction(a) * c for a, c in zip(junk, exact)) == 0
        assert ak.relation_confidence(strings, junk)["credible"] is False
        assert ak.relation_confidence(strings, junk, digits=dps)["credible"] is False
        with mpmath.workdps(dps + 5):
            mpfs = [mpmath.mpf(s) for s in strings]
        assert ak.relation_confidence(mpfs, junk)["credible"] is False

    def test_catalan_and_pi_squared_cheap_false_relation(self):
        with mpmath.workdps(60):
            constants = [mpmath.catalan, mpmath.pi**2]
        for coeffs in ([1, -1], [10, -1], [-1, 0]):
            assert ak.relation_confidence(constants, coeffs, digits=55)["credible"] is False


class TestFloatsCannotEndorseADegreeSixPolynomial:
    def test_float_minimal_polynomial_is_refused_not_endorsed(self):
        alpha = 2 ** (1 / 3) + 3**0.5
        powers = [alpha**k for k in range(7)]
        with pytest.raises(ak.PslqError) as excinfo:
            ak.guess_relation(powers)
        assert excinfo.value.code == "E-PSLQ-004"
        unjudged = ak.guess_relation(powers, check_precision=False)
        if unjudged is not None:
            assert ak.relation_confidence(powers, unjudged)["credible"] is False

    def test_a_cheap_float_relation_still_comes_back_endorsed(self):
        z, p4 = float(mpmath.zeta(4)), float(mpmath.pi**4)
        coeffs = ak.guess_relation([z, p4])
        assert _normalise(coeffs) == [-90, 1]
        assert ak.relation_confidence([z, p4], coeffs)["credible"] is True


class TestNoRawOverflowError:
    """Coefficients come back as Python ints of any size, never ``OverflowError``."""

    @pytest.mark.parametrize("dps", [30, 60, 115])
    def test_degree_six_search_never_overflows(self, dps):
        with mpmath.workdps(dps + 5):
            vals = _alpha_powers()
        for constants in (
            [mpmath.nstr(v, dps, strip_zeros=False) for v in vals],
            vals,
            [float(v) for v in vals],
        ):
            for kwargs in ({}, {"precision_bits": 2000}, {"check_precision": False}):
                # A coded refusal is fine; an OverflowError is not.
                with contextlib.suppress(ak.PslqError):
                    ak.guess_relation(constants, **kwargs)

    def test_large_coefficients_are_python_ints(self):
        """Exact rationals whose only relation needs a coefficient past i64."""
        from fractions import Fraction

        constants = [Fraction(1, 2**70), 1]
        coeffs = ak.guess_relation(constants)
        assert _normalise(coeffs) == [-(2**70), 1]
        assert all(isinstance(c, int) for c in coeffs)
        assert sum(Fraction(a) * c for a, c in zip(coeffs, constants)) == 0


class TestSiblingInputTypes:
    def test_fraction_is_not_rounded_through_float(self):
        """A Fraction used to reach the engine as a double, so the relation was
        searched among different numbers than the ones supplied."""
        from fractions import Fraction

        third, seventh = Fraction(1, 3), Fraction(1, 7)
        coeffs = ak.guess_relation([third, seventh, 1])
        assert coeffs is not None
        assert sum(Fraction(a) * c for a, c in zip(coeffs, [third, seventh, 1])) == 0

    def test_decimal_is_not_rounded_through_float(self):
        from decimal import Decimal, localcontext

        with localcontext() as ctx:
            ctx.prec = 80
            with mpmath.workdps(85):
                z = [Decimal(mpmath.nstr(v, 80)) for v in _zeta4_pi4()]
        assert _normalise(ak.guess_relation(z, digits=78)) == [-90, 1]

    def test_arb_ball_precision_comes_from_its_radius(self):
        """An ArbBall is accepted and searched at no more than its midpoint
        (a double) carries; its radius bounds what it can endorse."""
        balls = [ak.ArbBall(1.0, 0.0, 53), ak.ArbBall(2.0, 0.0, 53), ak.ArbBall(3.0, 0.0, 53)]
        coeffs = ak.guess_relation(balls)
        assert coeffs is not None
        assert ak.relation_confidence(balls, coeffs)["credible"] is True
        wide = [ak.ArbBall(1.0, 1e-3, 53), ak.ArbBall(2.0, 1e-3, 53), ak.ArbBall(3.0, 1e-3, 53)]
        verdict = ak.relation_confidence(wide, [1, 1, -1])
        assert verdict["available_digits"] == pytest.approx(math.log10(1e3), abs=0.5)
        assert verdict["credible"] is False
