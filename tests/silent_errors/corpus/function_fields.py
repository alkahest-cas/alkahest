"""Silent-error cases for function fields: divisors, Pic⁰ and Riemann–Roch.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.

Curve theory is unusually rich in clean, plausible, wrong answers, and three
families of them live here.

* **Riemann–Roch applied outside its hypothesis.** ``dim L(D) = deg D + 1 − g``
  is an *equality* only for ``deg D > 2g − 2``. Below that it is an
  inequality, and evaluating the formula anyway produces a number — sometimes
  a negative one — where the truth is larger. ``dim L(0) = 1`` on a genus-2
  curve, not ``0 + 1 − 2 = −1``; ``dim L(K) = g``, not ``(2g − 2) + 1 − g``.
  Both are checkable by hand and both are wrong in the same direction, which
  is what makes them a trap rather than a typo.
* **A divisor whose support is not representable.** Every place this library
  can hold has degree 1. When ``div(u)`` lands on a conjugate pair over a
  non-square, the *available* wrong answer is to drop the place — which
  returns a divisor of the wrong degree that still looks like a divisor, and
  still satisfies every cheap invariant a caller might spot-check except
  ``deg div(u) = 0``.
* **Torsion reported where none was established.** "No small multiple was
  principal" is not "the class has infinite order", and "the order could not
  be decided" is not "the class is not torsion". The two are separate codes
  (``E-FFLD-007`` a verdict, ``E-FFLD-006`` an undecided) precisely so that a
  caller cannot read one as the other.

Each refusal case is paired with its control: the nearest neighbour where the
same computation *does* apply and a value must come back.
"""

from __future__ import annotations

from typing import Any, Callable

import alkahest.experimental as ex
from contracts import Case, Raises, Returns

# y² = x³ − x, genus 1.  Rational points: ∞ and the three branch points.
_E1 = [0, -1, 0, 1]
# y² = x⁵ + 1, genus 2.
_C2 = [1, 0, 0, 0, 0, 1]
# y² = x⁴ + 1, genus 1 but an even-degree ("real") model.
_REAL = [1, 0, 0, 0, 1]
# y² = x³ − 2, genus 1, rank 1: (3, 5) generates a non-torsion class.
_RANK1 = [-2, 0, 0, 1]


def _field(a: list[int]) -> ex.FunctionField:
    return ex.FunctionField.hyperelliptic(a)


def _genus(a: list[int]) -> Callable[[], int]:
    """Answer = the genus of ``y² = a(x)``."""

    def op() -> int:
        return int(_field(a).genus)

    return op


def _dim_at_infinity(a: list[int], n: int) -> Callable[[], int]:
    """Answer = ``dim L(n·∞)``."""

    def op() -> int:
        f = _field(a)
        d = ex.Divisor(f, [(ex.Place.infinity(), n)])
        return int(ex.riemann_roch(d).dimension)

    return op


def _dim_canonical(a: list[int]) -> Callable[[], int]:
    """Answer = ``dim L(K)`` for the canonical divisor."""

    def op() -> int:
        f = _field(a)
        return int(ex.riemann_roch(f.canonical_divisor()).dimension)

    return op


def _deg_canonical(a: list[int]) -> Callable[[], int]:
    """Answer = ``deg K``."""

    def op() -> int:
        return int(_field(a).canonical_divisor().degree)

    return op


def _divisor_degree(a: list[int], p: list[int], q: list[int]) -> Callable[[], int]:
    """Answer = ``deg div((p + q·y))``, which must be 0 whenever it is returned."""

    def op() -> int:
        f = _field(a)
        return int(ex.FunctionFieldElement(f, p, q).divisor().degree)

    return op


def _order_of_point(a: list[int], x: int, y: int) -> Callable[[], int]:
    """Answer = the order of the class of ``(x, y) − ∞``."""

    def op() -> int:
        f = _field(a)
        d = ex.Divisor(f, [(ex.Place.finite(x, y), 1), (ex.Place.infinity(), -1)])
        return int(d.divisor_class().order())

    return op


def _canonical_divisor(a: list[int]) -> Callable[[], Any]:
    """Answer = the canonical divisor rendered as a string."""

    def op() -> str:
        return str(_field(a).canonical_divisor())

    return op


CASES: list[Case] = [
    # ── Riemann–Roch below the canonical degree ─────────────────────────────
    #
    # The equality dim L(D) = deg D + 1 − g holds only above deg K = 2g − 2.
    # Applying it anyway is the single most available wrong answer in this
    # subsystem, and it is available because the formula is the one everybody
    # remembers and the hypothesis is the one everybody drops.
    Case(
        id="ffld_dim_l_zero_is_one_genus_two",
        subsystem="function_fields",
        statement="dim L(0) = 1 on y² = x⁵ + 1 (genus 2): only the constants",
        op=_dim_at_infinity(_C2, 0),
        contract=Returns(1),
        verified_by=(
            "L(0) is the ring of functions regular everywhere on a projective curve, which "
            "over ℚ is exactly the constants, so dim L(0) = 1 on every curve of every genus "
            "(Hartshorne II.6, Stichtenoth I.4.6). The dangerous answer is the Riemann-Roch "
            "equality evaluated outside its hypothesis: deg D + 1 − g = 0 + 1 − 2 = −1, a "
            "negative dimension, or the clamp of it to 0 — which says a genus-2 curve has no "
            "non-zero global regular function, not even 1."
        ),
    ),
    Case(
        id="ffld_dim_l_canonical_is_g_genus_two",
        subsystem="function_fields",
        statement="dim L(K) = g = 2 on y² = x⁵ + 1, not deg K + 1 − g = 1",
        op=_dim_canonical(_C2),
        contract=Returns(2),
        verified_by=(
            "dim L(K) = g is the definition of the genus via the canonical class, and for a "
            "hyperelliptic curve L(K) is spanned by 1, x, …, x^{g−1} (the holomorphic "
            "differentials x^i dx/y). deg K = 2g − 2 = 2 sits exactly *on* the boundary the "
            "Riemann-Roch equality excludes, so the formula gives 2 + 1 − 2 = 1 and is off by "
            "exactly one — the index of speciality, which is the whole content of the theorem."
        ),
    ),
    Case(
        id="ffld_dim_l_canonical_is_g_genus_one",
        subsystem="function_fields",
        statement="dim L(K) = 1 on the elliptic curve y² = x³ − x, where K = 0",
        op=_dim_canonical(_E1),
        contract=Returns(1),
        verified_by=(
            "An elliptic curve has trivial canonical class, K = 0, and dim L(0) = 1. Here the "
            "Riemann-Roch formula happens to agree (0 + 1 − 1 = 1), which is why it is the "
            "control for the genus-2 case above: the trap is not that the formula is always "
            "wrong below 2g − 2, it is that it is wrong *sometimes*, with nothing marking which."
        ),
    ),
    Case(
        id="ffld_deg_canonical_is_two_g_minus_two",
        subsystem="function_fields",
        statement="deg K = 2g − 2 = 2 on y² = x⁵ + 1",
        op=_deg_canonical(_C2),
        contract=Returns(2),
        verified_by=(
            "deg K = 2g − 2 for every curve of genus g (Riemann-Roch applied to D = K, or "
            "the degree of the divisor of any non-zero differential). For y² = f(x) with "
            "deg f = 2g + 1, div(dx/y) = (2g − 2)·∞."
        ),
    ),
    Case(
        id="ffld_dim_negative_degree_is_zero",
        subsystem="function_fields",
        statement="dim L(−3·∞) = 0: a divisor of negative degree has no sections",
        op=_dim_at_infinity(_E1, -3),
        contract=Returns(0),
        verified_by=(
            "A non-zero u with div(u) ≥ −D would give 0 = deg div(u) ≥ −deg D > 0. The "
            "dangerous answer is the formula's deg D + 1 − g = −3, a negative dimension "
            "reported as a dimension."
        ),
    ),
    Case(
        id="ffld_dim_above_canonical_degree_is_the_formula",
        subsystem="function_fields",
        statement="dim L(5·∞) = 5 + 1 − 2 = 4 on y² = x⁵ + 1, where the equality does hold",
        op=_dim_at_infinity(_C2, 5),
        contract=Returns(4),
        verified_by=(
            "deg D = 5 > 2g − 2 = 2, so Riemann-Roch is an equality: dim = deg D + 1 − g = 4. "
            "The control for the two cases above — the formula must be used where it applies, "
            "not refused everywhere out of caution."
        ),
    ),
    # ── a divisor whose support cannot be represented ───────────────────────
    #
    # Dropping an unrepresentable place is the silent failure: it returns a
    # divisor object of the wrong degree, and deg div(u) = 0 is the only cheap
    # invariant that catches it.
    Case(
        id="ffld_div_y_refuses_on_an_irrational_branch_locus",
        subsystem="function_fields",
        statement="div(y) on y² = x⁵ + 1 is supported at places of degree > 1 and must refuse",
        op=_divisor_degree(_C2, [0], [1]),
        contract=Raises("E-FFLD-003"),
        verified_by=(
            "div(y) = Σ_{α: f(α)=0} (α, 0) − 5·∞, and x⁵ + 1 = (x + 1)(x⁴ − x³ + x² − x + 1) "
            "where the quartic is irreducible over ℚ (it is the 10th cyclotomic polynomial). "
            "Four of the five branch points are therefore not rational and have no degree-1 "
            "representation. The available wrong answer is to report only the rational one: "
            "(−1, 0) − 5·∞, a 'divisor' of degree −4 for a function, which is impossible."
        ),
    ),
    Case(
        id="ffld_div_y_returns_degree_zero_when_the_locus_is_rational",
        subsystem="function_fields",
        statement="div(y) on y² = x³ − x has degree 0: the branch locus is rational",
        op=_divisor_degree(_E1, [0], [1]),
        contract=Returns(0),
        verified_by=(
            "x³ − x = x(x−1)(x+1) splits over ℚ, so div(y) = (−1,0) + (0,0) + (1,0) − 3·∞, of "
            "degree 3 − 3 = 0. Every divisor of a function has degree zero (Stichtenoth I.4.11). "
            "The control for the refusal above."
        ),
    ),
    Case(
        id="ffld_div_x_returns_degree_zero",
        subsystem="function_fields",
        statement="deg div(x) = 0 on y² = x³ − x, where x = 0 is a branch point",
        op=_divisor_degree(_E1, [0, 1], [0]),
        contract=Returns(0),
        verified_by=(
            "x = 0 is a branch point, so v_P(x) = 2 there and div(x) = 2·(0,0) − 2·∞. The "
            "ramification is the trap: reading the fibre as two distinct places of "
            "multiplicity 1 each gives the same degree, but the wrong divisor, and shows up "
            "only when the divisor is used."
        ),
    ),
    # ── torsion: three outcomes that must stay apart ────────────────────────
    Case(
        id="ffld_non_torsion_is_a_verdict_not_a_number",
        subsystem="function_fields",
        statement="(3,5) − ∞ on y² = x³ − 2 has infinite order; no finite order may be reported",
        op=_order_of_point(_RANK1, 3, 5),
        contract=Raises("E-FFLD-007"),
        verified_by=(
            "y² = x³ − 2 is the Mordell curve of rank 1 with generator (3, 5), and its torsion "
            "subgroup over ℚ is trivial (Cremona 1728.n1; Silverman, AEC X.§6). So the class "
            "has infinite order. The dangerous answer is a finite one: a search that stops at "
            "some bound and reports the bound, or reports the order of the reduction modulo a "
            "single prime — both are plausible small integers with nothing marking them as "
            "unproven."
        ),
    ),
    Case(
        id="ffld_two_torsion_at_a_branch_point",
        subsystem="function_fields",
        statement="(0,0) − ∞ on y² = x³ − x has order exactly 2",
        op=_order_of_point(_E1, 0, 0),
        contract=Returns(2),
        verified_by=(
            "A branch point P = (α, 0) satisfies 2(P − ∞) = div(x − α), so the class is "
            "2-torsion, and it is non-trivial because P ≠ ∞. The control for the non-torsion "
            "case: the machinery must return an order where one exists."
        ),
    ),
    Case(
        id="ffld_order_three_on_y2_x3_plus_1",
        subsystem="function_fields",
        statement="(0,1) − ∞ on y² = x³ + 1 has order exactly 3, not 6",
        op=_order_of_point([1, 0, 0, 1], 0, 1),
        contract=Returns(3),
        verified_by=(
            "The duplication formula at (0,1) gives λ = 3x²/(2y) = 0, hence 2P = (0, −1) = −P "
            "and 3P = O. The trap is reporting the order of the *group* (E(ℚ)_tors ≅ ℤ/6 for "
            "this curve, Cremona 36a1) instead of the order of the element — a multiple of the "
            "right answer that kills the class just as well, so every 'is N·δ principal?' "
            "check on it passes."
        ),
    ),
    # ── model boundaries: refuse, do not extrapolate ────────────────────────
    Case(
        id="ffld_even_degree_model_reports_genus",
        subsystem="function_fields",
        statement="y² = x⁴ + 1 has genus 1; the genus does not depend on the model",
        op=_genus(_REAL),
        contract=Returns(1),
        verified_by=(
            "For squarefree a of degree 2g + 2 the genus is g (the Riemann-Hurwitz count over "
            "the 2g + 2 branch points, with no ramification at infinity). Refusing to state "
            "the genus because the divisor machinery does not cover the model would be an "
            "unnecessary refusal — the control that this module's boundary is drawn in the "
            "right place."
        ),
    ),
    Case(
        id="ffld_even_degree_model_refuses_divisors",
        subsystem="function_fields",
        statement="y² = x⁴ + 1 has two places above x = ∞, so K = (2g−2)·∞ is meaningless there",
        op=_canonical_divisor(_REAL),
        contract=Raises("E-FFLD-002"),
        verified_by=(
            "On an even-degree model x has two poles ∞₊ and ∞₋, each of degree 1, and the "
            "canonical class is (g−1)(∞₊ + ∞₋). Writing it as (2g−2)·∞ presumes a single "
            "rational place at infinity that does not exist. The answer has the right degree, "
            "which is exactly why it is dangerous: every degree check passes."
        ),
    ),
    Case(
        id="ffld_higher_degree_model_refuses",
        subsystem="function_fields",
        statement="y³ = x⁴ − 1 is not a quadratic model and must be refused, not guessed at",
        op=lambda: int(ex.FunctionField([[1, 0, 0, 0, -1], [], [], [1]]).genus),
        contract=Raises("E-FFLD-001"),
        verified_by=(
            "The superelliptic curve y³ = x⁴ − 1 has genus 3 by the Riemann-Hurwitz formula "
            "(m·n − m − n + 2 − gcd(m,n))/2 with m = 4, n = 3. This module implements only "
            "n = 2, so the honest answer is a refusal. The dangerous answer is the "
            "hyperelliptic formula applied anyway — ⌊(4−1)/2⌋ = 1 — a genus that is off by two "
            "and indistinguishable from a correct one."
        ),
    ),
    Case(
        id="ffld_place_off_the_curve_is_refused",
        subsystem="function_fields",
        statement="(2, 0) is not on y² = x³ − x, since 2³ − 2 = 6 ≠ 0",
        op=lambda: str(
            ex.Divisor(ex.FunctionField.hyperelliptic(_E1), [(ex.Place.finite(2, 0), 1)])
        ),
        contract=Raises("E-FFLD-004"),
        verified_by=(
            "y² = a(x) at x = 2 requires y² = 6, and 6 is not a rational square, so no rational "
            "place lies over x = 2 at all. Accepting (2, 0) would make every subsequent "
            "divisor computation quietly meaningless while remaining type-correct."
        ),
    ),
    Case(
        id="ffld_nonzero_degree_divisor_has_no_class",
        subsystem="function_fields",
        statement="a divisor of degree 1 has no class in Pic⁰ and must not be projected into it",
        op=lambda: str(
            ex.Divisor(
                ex.FunctionField.hyperelliptic(_E1), [(ex.Place.finite(0, 0), 1)]
            ).divisor_class()
        ),
        contract=Raises("E-FFLD-005"),
        verified_by=(
            "Pic⁰ is the group of degree-zero classes. The available wrong answer is to drop "
            "the degree silently — to treat (0,0) as (0,0) − ∞ — which gives a class that is "
            "a perfectly good element of Pic⁰ and is the class of a *different* divisor."
        ),
    ),
]
