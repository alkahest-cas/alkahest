"""`E-INT-004` is a theorem, not a status code.

`integrate` may only answer "no elementary antiderivative exists" when it has
actually proved that.  It once answered it for `∫x dx/√(1−x⁴) = ½asin(x²)`, and
for a whole family around it, because the proof rested on two premises that were
never established:

* *"the residue divisor is empty"* — the places over `x = ∞` on `y² = 1−x⁴`
  carry residues `±i`, and the enumeration that looked for them worked only over
  `ℚ`, so it found nothing and the code read that as there being nothing.  The
  residue theorem `Σ res = 0` did not catch it: an empty list sums to zero
  vacuously, and so does an omitted conjugate pair.
* *"there is no algebraic primitive"* — a Risch differential equation whose
  solver *declined*, which is not the same as one that has no solution.

Two rules follow, and this file pins both:

1. Nothing certified non-elementary may have an antiderivative.  For every
   integrand below that Alkahest solves, `d/dx F = f` is checked numerically —
   no display-string matching anywhere in this file.
2. Integrals that *are* genuinely non-elementary must keep their certificates.
   Downgrading everything to `E-INT-001` would also make rule 1 hold, and would
   be worthless.
"""

import alkahest as ak
import pytest
from alkahest.alkahest import ArbBall, ExprPool, diff, integrate, interval_eval

_TOL = 1e-8


def _verify(src, points):
    """Integrate `src` and check `d/dx F = f` at `points`.  Returns `F`."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = ak.parse(src, pool)
    cap = integrate(f, x).value
    d = diff(cap, x).value
    checked = 0
    for pt in points:
        bindings = {x: ArbBall(pt)}
        lhs = interval_eval(d, bindings).mid
        rhs = interval_eval(f, bindings).mid
        if lhs != lhs or rhs != rhs:  # NaN outside the real domain — skip
            continue
        assert abs(lhs - rhs) < _TOL * (1 + abs(rhs)), (
            f"{src}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs}\n  F = {cap}"
        )
        checked += 1
    assert checked >= 3, f"{src}: only {checked} usable sample points"
    return cap


def _code(src):
    """The error code `integrate` raises for `src`, or `'ok'` if it succeeds."""
    pool = ExprPool()
    x = pool.symbol("x")
    try:
        integrate(ak.parse(src, pool), x)
    except Exception as exc:
        return getattr(exc, "code", type(exc).__name__)
    return "ok"


# ---------------------------------------------------------------------------
# The reproducers: every one of these was E-INT-004 and every one is elementary
# ---------------------------------------------------------------------------

_INSIDE_UNIT = (0.17, 0.31, 0.48, 0.63, 0.79)

_WAS_FALSELY_CERTIFIED = [
    # ∫x^{k−1} dx/√(1 − x^{2k}) = (1/k)·asin(x^k)
    ("x/sqrt(1-x^4)", _INSIDE_UNIT),
    ("x^2/sqrt(1-x^6)", _INSIDE_UNIT),
    ("x^3/sqrt(1-x^8)", _INSIDE_UNIT),
    ("x^4/sqrt(1-x^10)", _INSIDE_UNIT),
    # scaled radicands
    ("x/sqrt(4-x^4)", (0.2, 0.5, 0.9, 1.2)),
    ("x/sqrt(9-x^4)", (0.3, 0.7, 1.1, 1.5)),
    ("x^2/sqrt(4-x^6)", (0.2, 0.5, 0.9, 1.2)),
    # second kind: the same pullback, weight √P instead of 1/√P
    ("x*sqrt(1-x^4)", _INSIDE_UNIT),
    ("x^2*sqrt(1-x^6)", _INSIDE_UNIT),
    ("x^3*sqrt(1-x^8)", _INSIDE_UNIT),
    ("x^5/sqrt(1-x^4)", _INSIDE_UNIT),
    # A pole at x = 0 on an *irrational* sheet (a(0) = −1).  The rational-Puiseux
    # routine cannot see it and the algebraic one never got the chance, because
    # the factoriser strips the factor `x` before returning — so between them the
    # residue divisor looked empty.
    ("sqrt(x^4-1)/x", (1.3, 1.7, 2.4, 3.1)),
    ("1/(x*sqrt(x^4-1))", (1.3, 1.7, 2.4, 3.1)),
]


@pytest.mark.parametrize(("src", "points"), _WAS_FALSELY_CERTIFIED)
def test_falsely_certified_integrals_are_solved_and_correct(src, points):
    _verify(src, points)


# The `+` half of the family was an honest E-INT-001 decline rather than a false
# certificate, and closes through the same route: `(1/k)·asinh(x^k)`.
@pytest.mark.parametrize(
    ("src", "points"),
    [
        ("x/sqrt(1+x^4)", (0.2, 0.7, 1.3, 2.1)),
        ("x^2/sqrt(1+x^6)", (0.2, 0.7, 1.3, 2.1)),
        ("x*sqrt(1+x^4)", (0.2, 0.9, 1.4, 2.2)),
    ],
)
def test_asinh_half_of_the_family(src, points):
    _verify(src, points)


def test_the_general_family_for_k_2_3_4():
    """`∫x^{k−1} dx/√(1 − x^{2k}) = (1/k)·asin(x^k)`, checked by differentiation."""
    for k in (2, 3, 4):
        _verify(f"x^{k - 1}/sqrt(1-x^{2 * k})", _INSIDE_UNIT)


# ---------------------------------------------------------------------------
# The certificates that must survive
# ---------------------------------------------------------------------------

# Each of these is genuinely non-elementary, and the engine now proves it:
# the residue divisor is empty over a *certified-complete* enumeration, and the
# Risch DE `b' + (P'/2P)b = B` is *decided* to have no rational solution.
_GENUINELY_NON_ELEMENTARY = [
    # ½∫du/√(1−u³) under u = x² — an elliptic integral of the first kind.  The
    # pullback route can see this reduction and is deliberately not allowed to
    # use it to talk the certificate down.
    "x/sqrt(1-x^6)",
    "1/sqrt(x^5+1)",
    "x/sqrt(x^5+1)",
    "x^3/sqrt(x^5+1)",
    "1/((x+2)*sqrt(x^5+1))",
    "1/sqrt(1-x^3)",
]


@pytest.mark.parametrize("src", _GENUINELY_NON_ELEMENTARY)
def test_genuine_certificates_survive(src):
    assert _code(src) == "E-INT-004", (
        f"{src} is genuinely non-elementary and must keep its certificate"
    )


def test_a_declined_risch_de_never_certifies():
    """An *undecided* integral part must produce E-INT-001, never E-INT-004.

    `∫x dx/√(1−x⁴)` is the witness: its residue divisor is non-empty (the `±i`
    over `∞`) and its integral part has no rational solution, so under the old
    reading — empty-looking divisor plus a declining solver — it was certified.
    Whatever this route ends up returning, it may not be a certificate, because
    the integral is elementary.
    """
    assert _code("x/sqrt(1-x^4)") == "ok"
    # …and the shapes the pullback cannot close still decline honestly rather
    # than certifying: `∫du/√(1−4u²)` is a gap in the genus-0 quadratic route.
    assert _code("x/sqrt(1-4*x^4)") == "E-INT-001"


def test_simple_radical_route_never_certifies():
    """`∫R(x, x^{1/n}) dx` is *always* elementary — `x = uⁿ` makes it rational.

    The simple-radical route decides only the integral part `vⱼ·yʲ` of the
    Liouville decomposition and reports `NonElementary` when its component Risch
    DE has no rational solution, without ever looking at the logarithmic part.
    `∫∛x/(x²+1) dx` is the witness: it equals

        −½log(u²+1) + ¼log(u⁴−u²+1) + (√3/2)·atan((2√3u²−√3)/3),   u = x^{1/3}

    and was certified non-elementary.  Whether or not the engine can produce
    that answer, it may not claim there is none.
    """
    for src in ("x^(1/3)/(x^2+1)", "x^(1/3)/(x^3-1)", "x^(2/5)/(x-1)"):
        pool = ExprPool()
        x = pool.symbol("x")
        # `^(1/3)` parses to an unevaluated exponent; normalise so the route is
        # actually reached rather than skipped on a spelling technicality.
        f = ak.simplify(ak.parse(src, pool)).value
        code = "ok"
        try:
            integrate(f, x)
        except Exception as exc:
            code = getattr(exc, "code", type(exc).__name__)
        assert code != "E-INT-004", (
            f"{src} is elementary (x = u^n rationalises it) — no certificate allowed"
        )


def test_no_certificate_for_anything_with_a_pullback_to_a_solved_integral():
    """Sweep the pullback family: nothing in it may come back E-INT-004."""
    bad = []
    for k in (2, 3, 4):
        for c in (1, 2, 4):
            for shape in (f"x^{k - 1}/sqrt({c}-x^{2 * k})", f"x^{k - 1}*sqrt({c}-x^{2 * k})"):
                if _code(shape) == "E-INT-004":
                    bad.append(shape)
    assert not bad, f"false non-elementarity certificates: {bad}"
