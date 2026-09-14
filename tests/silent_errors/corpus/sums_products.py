"""Silent-error cases for sums products.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Callable

import alkahest as ak
import alkahest.experimental as ex
from contracts import Case, RefusesOr, Returns

from ._shared import POOL, N, _int, _num, _rat

K = POOL.symbol("k")
#: Symbolic geometric ratio, for the `Σ rᵏ` cases.
R = POOL.symbol("r")


def _rsolve_residual(equation: ak.Expr, initials: dict[int, ak.Expr]) -> Callable[[], float]:
    """Answer = the worst residual of ``rsolve``'s answer *in the given equation*.

    Substituting the closed form back into the very equation that was passed in
    is self-certifying: it needs no oracle, and it is the one property a
    recurrence solver may never get wrong.  A solver that quietly re-indexes the
    equation returns the solution of a *different* problem, which is a clean,
    plausible, wrong sequence.
    """

    def op() -> float:
        closed = ak.rsolve(equation, N, "f", initials)
        residual = ak.simplify(_substitute_sequence(equation, closed)).value
        return max(abs(float(ak.eval_expr(residual, {N: float(j)}))) for j in range(6))

    return op


#: Shifts the recurrence cases are written with.
_SEQ_SHIFTS = (2, 1, 0, -1, -2)


def _seq(shift: int) -> ak.Expr:
    """``f(n + shift)`` — the term shape ``rsolve`` reads."""
    return POOL.func("f", [N if shift == 0 else N + _int(shift)])


def _substitute_sequence(equation: ak.Expr, closed: ak.Expr) -> ak.Expr:
    """``equation`` with every ``f(n + c)`` replaced by ``closed`` shifted by c.

    Written against the fixed shift set the recurrence cases are built from
    (:data:`_SEQ_SHIFTS`, via :func:`_seq`) rather than by walking the expression
    tree, so the substitution itself stays obviously correct.
    """
    out = equation
    for c in _SEQ_SHIFTS:
        arg = N if c == 0 else N + _int(c)
        shifted = closed if c == 0 else ak.subs(closed, {N: arg})
        out = ak.subs(out, {_seq(c): shifted})
    return out


def _basis_independence(equation: ak.Expr) -> Callable[[], bool]:
    """Answer = whether ``rsolve``'s *general* solution spans two dimensions.

    The general solution of a second-order linear recurrence is a two-parameter
    family.  Returning ``C₀·rⁿ + C₁·rⁿ`` for a repeated root looks like one but
    is not: both basis elements are the same function, so the family is
    one-dimensional and cannot meet two independent initial conditions.
    """

    def op() -> bool:
        general = ak.rsolve(equation, N, "f", None)
        c0 = POOL.symbol("C0")
        c1 = POOL.symbol("C1")
        rows = []
        for at in (0.0, 1.0):
            first = ak.eval_expr(general, {c0: 1.0, c1: 0.0, N: at})
            second = ak.eval_expr(general, {c0: 0.0, c1: 1.0, N: at})
            rows.append((float(first), float(second)))
        det = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        return abs(det) > 1e-9

    return op


def _constant_terms(report: Any) -> list[float]:
    """The values of every term of an asymptotic expansion that does not move.

    A term with the same value at ``n = 10`` and ``n = 20`` is a constant, and a
    constant claimed for a sum whose closed form is a polynomial with zero
    constant term is a fabricated one.
    """
    out = []
    for term in report.terms:
        lo = float(ak.eval_expr(term, {N: 10.0}))
        hi = float(ak.eval_expr(term, {N: 20.0}))
        if abs(lo - hi) <= 1e-9 * max(1.0, abs(lo)):
            out.append(lo)
    return out


def _binom(top: ak.Expr, bot: ak.Expr) -> ak.Expr:
    """``C(top, bot)`` as a Γ-quotient, the shape ``zeilberger`` parses."""
    return ak.gamma(top + _int(1)) / (ak.gamma(bot + _int(1)) * ak.gamma(top - bot + _int(1)))


def _zeilberger_sum_recurrence_defect(
    term: ak.Expr, exact_sum: Callable[[int], Fraction], disclosure_counts: bool
) -> Callable[[], float]:
    """Answer = how badly the *sum* recurrence read off the certificate fails.

    Zeilberger verifies ``Σ_i a_i(n)·F(n+i,k) = G(n,k+1) − G(n,k)``, an identity
    in ``k``.  Summing it gives ``Σ_i a_i(n)·S(n+i) = G(n,k_hi+1) − G(n,k_lo)``,
    so the familiar homogeneous recurrence needs that boundary difference to
    vanish — a hypothesis the algorithm does not establish.

    With *disclosure_counts* the case is satisfied either way an honest library
    can behave: prove the hypothesis (residual genuinely zero) or state it as a
    side condition on the certificate.  Silently omitting it scores the residual,
    which is what a caller who trusted the recurrence would inherit.
    """

    def op() -> float:
        cert = ak.zeilberger(term, N, K)
        if disclosure_counts:
            conditions = getattr(cert, "side_conditions", ())
            if any("boundary" in str(c).lower() for c in conditions):
                return 0.0
        worst = 0.0
        for ni in range(1, 6):
            total = Fraction(0)
            for i, a in enumerate(cert.coeffs):
                coeff = Fraction(float(ak.eval_expr(a, {N: float(ni)}))).limit_denominator(10**9)
                total += coeff * exact_sum(ni + i)
            worst = max(worst, abs(float(total)))
        return worst

    return op


def _zeilberger_boundary_tag(term: ak.Expr) -> Callable[[], str]:
    """Answer = ``cert.boundary``, the three-valued verdict on the *sum*.

    The certificate is an identity about the summand and always holds; the
    verdict is the separate claim that a recurrence for ``S(n)`` follows from
    it.  Scoring the verdict rather than the coefficients is what makes a
    ``"vanishes"`` about a sum that does not exist a silent error rather than a
    detail buried in a side-condition string.
    """

    def op() -> str:
        return str(ak.zeilberger(term, N, K).boundary)

    return op


def _zeilberger_boundary_at_tag(term: ak.Expr, lo: int, hi: int) -> Callable[[], str]:
    """Answer = the same verdict over an explicit *constant* range ``k = lo..hi``.

    ``cert.boundary`` asks about the natural ``k = 0..n``; ``boundary_at`` asks
    the same certificate about a range that does not move with ``n``.  That is
    the cheapest place to see an endpoint whose value depends on which ``n`` you
    are at, because the endpoint ``k = hi + 1`` is then a fixed integer while the
    summand's Γ arguments still slide past their poles as ``n`` changes.
    """

    def op() -> str:
        cert = ak.zeilberger(term, N, K)
        return str(cert.boundary_at(_int(lo), _int(hi))["boundary"])

    return op


def _sum_binomial_over_k_plus_one(m: int) -> Fraction:
    """``Σ_{k=0}^{m} C(m,k)/(k+1) = (2^{m+1} − 1)/(m+1)``, by hand."""
    return sum((Fraction(math.comb(m, j), j + 1) for j in range(m + 1)), Fraction(0))


def _sum_binomial_row(m: int) -> Fraction:
    """``Σ_{k=0}^{m} C(m,k) = 2^m``."""
    return Fraction(2**m)


def _perron_growth_rate(polys: list[tuple[int, ...]], terms: list[int]) -> float:
    """The growth rate ``asymptotics_from_recurrence`` is prepared to claim.

    NaN when it claims none — which the probe reads as a refusal, and which is
    the right answer whenever Poincaré–Perron's hypotheses fail. A confident
    root reported where the hypotheses do not hold is the silent error.
    """
    r = ex.asymptotics_from_recurrence(polys, N, terms=terms)
    if r.growth_rate is None or r.follows_dominant_root is False:
        return float("nan")
    return float(r.growth_rate)


def _perron_connection_constant(polys: list[tuple[int, ...]], terms: list[int]) -> float:
    r = ex.asymptotics_from_recurrence(polys, N, terms=terms)
    if not r.connection_constant_converged:
        return float("nan")
    return float(r.connection_constant)


CASES: list[Case] = [
    # Divergent sums and products.  Every one of these has a famous "value"
    # attached to it by some summation method; none of them converges.
    # -----------------------------------------------------------------------
    Case(
        id="sum_harmonic_divergent",
        subsystem="sums_products",
        statement="Σ_{k≥1} 1/k diverges",
        op=lambda: _num(ak.sum_definite(1 / K, K, _int(1), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="Partial sums exceed log n; the classic Oresme grouping argument.",
    ),
    Case(
        id="sum_geometric_divergent",
        subsystem="sums_products",
        statement="Σ_{k≥1} 2^k diverges",
        op=lambda: _num(ak.sum_definite(_int(2) ** K, K, _int(1), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="|r| = 2 > 1. Blindly applying a/(1-r) gives 2/(1-2) = -2, a clean wrong "
        "number for a sum of positive terms.",
    ),
    Case(
        id="sum_grandi_divergent",
        subsystem="sums_products",
        statement="Σ_{k≥0} (-1)^k diverges (Grandi's series)",
        op=lambda: _num(ak.sum_definite(_int(-1) ** K, K, _int(0), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="Partial sums alternate 1,0,1,0,… and have no limit. The Abel and Cesàro "
        "sums are 1/2 — the canonical plausible wrong answer.",
    ),
    Case(
        id="product_divergent_constant",
        subsystem="sums_products",
        statement="Π_{k≥1} 2 diverges",
        op=lambda: _num(ak.product_definite(_int(2), K, _int(1), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="The partial products are 2^n, which grow without bound.",
        note="Weak refusal: alkahest returns the symbol 2^∞, which does not reduce to a float.",
    ),
    Case(
        id="sum_control_first_ten",
        subsystem="sums_products",
        statement="Σ_{k=1}^{10} k = 55",
        op=lambda: _num(ak.sum_definite(K, K, _int(1), _int(10)).value),
        contract=Returns(55.0),
        verified_by="10·11/2 = 55.",
    ),
    Case(
        id="sum_control_faulhaber_symbolic",
        subsystem="sums_products",
        statement="Σ_{k=1}^{n} k = n(n+1)/2; at n=7 that is 28",
        op=lambda: float(ak.eval_expr(ak.sum_definite(K, K, _int(1), N).value, {N: 7})),
        contract=Returns(28.0),
        verified_by="7·8/2 = 28. Checks the closed form rather than its printed shape.",
    ),
    Case(
        id="product_control_factorial",
        subsystem="sums_products",
        statement="Π_{k=1}^{5} k = 120",
        op=lambda: _num(ak.product_definite(K, K, _int(1), _int(5)).value),
        contract=Returns(120.0),
        verified_by="1·2·3·4·5 = 120, i.e. 5! computed by hand.",
    ),
    Case(
        id="product_control_contains_zero",
        subsystem="sums_products",
        statement="Π_{k=0}^{5} k = 0 — the k=0 factor annihilates the product",
        op=lambda: _num(ak.product_definite(K, K, _int(0), _int(5)).value),
        contract=Returns(0.0),
        verified_by="0·1·2·3·4·5 = 0. A gamma-quotient closed form that forgets the pole at "
        "k=0 would report 120.",
    ),
    # The dropped rational scale in `product_definite`, and poles strictly
    # inside a summation range.  Both were reported in
    # `temp-alkahest/testing/3.8-silent-error-hunt-2.md` and fixed for 3.8.0.
    # -----------------------------------------------------------------------
    Case(
        id="product_definite_keeps_rational_scale",
        subsystem="sums_products",
        statement="Π_{k=1}^{5} 1/2 = 1/32",
        op=lambda: _num(ak.product_definite(_rat(1, 2), K, _int(1), _int(5))),
        contract=Returns(1.0 / 32.0, tol=1e-12),
        verified_by="Five factors of 1/2 multiply to 2^-5 = 1/32, by the definition of a product.",
    ),
    Case(
        id="product_definite_wallis_partial_product",
        subsystem="sums_products",
        statement="Π_{k=1}^{6} (2k-1)/(2k) = C(12,6)/4^6 = 924/4096",
        op=lambda: _num(
            ak.product_definite(
                (_int(2) * K - _int(1)) * (_int(2) * K) ** _int(-1), K, _int(1), _int(6)
            )
        ),
        contract=Returns(924.0 / 4096.0, tol=1e-9),
        verified_by=(
            "1·3·5·7·9·11 / (2·4·6·8·10·12) = 10395/46080 = 924/4096 = 0.2255859375, multiplied "
            "out by hand; it is also the standard Π(2k-1)/(2k) = C(2n,n)/4ⁿ at n = 6. alkahest "
            "returned 14.4375, which is 2⁶ times too large — one factor of the denominator's "
            "leading coefficient per index, from the scale ratuni_poly_to_univ discarded."
        ),
    ),
    Case(
        id="product_definite_empty_range_of_a_zero_term",
        subsystem="sums_products",
        statement="Π_{k=1}^{0} 0 = 1 — an empty product takes no factors at all",
        op=lambda: _num(ak.product_definite(_int(0), K, _int(1), _int(0))),
        contract=Returns(1.0),
        verified_by=(
            "The empty product is 1 by universal convention, whatever the term is: no factor is "
            "ever taken. alkahest returned 0 here while returning 1 for Π_{k=1}^{0} k, so its own "
            "two answers for the same empty range disagreed — the zero-numerator shortcut ran "
            "before the empty-range check."
        ),
    ),
    Case(
        id="product_control_integer_coefficient_ratio",
        subsystem="sums_products",
        statement="Π_{k=1}^{4} (k+1)/k = 5 — telescoping, no denominators to clear",
        op=lambda: _num(ak.product_definite((K + _int(1)) * K ** _int(-1), K, _int(1), _int(4))),
        contract=Returns(5.0, tol=1e-9),
        verified_by=(
            "(2/1)(3/2)(4/3)(5/4) telescopes to 5/1 = 5. The control for the rational-scale "
            "cases: this one has monic numerator and denominator, so it was already correct "
            "before the fix and must stay correct after it — a product_definite that started "
            "refusing every rational term would not pass here."
        ),
    ),
    Case(
        id="sum_definite_interior_pole_refused",
        subsystem="sums_products",
        statement="Σ_{k=1}^{10} 1/((k-3)(k-2)) is undefined — the k=2 and k=3 terms divide by zero",
        op=lambda: _num(
            ak.sum_definite(((K - _int(3)) * (K - _int(2))) ** _int(-1), K, _int(1), _int(10))
        ),
        contract=RefusesOr(),
        verified_by=(
            "The k=2 term is 1/((-1)·0) and the k=3 term is 1/(0·1); neither is a number, so the "
            "sum has no value. alkahest returned -5/8. Its own docstring promises E-SUM-003 for "
            "exactly this."
        ),
    ),
    Case(
        id="sum_definite_interior_pole_negative_lower_bound",
        subsystem="sums_products",
        statement="Σ_{k=-2}^{5} 1/(k(k+1)) is undefined — the k=-1 and k=0 terms divide by zero",
        op=lambda: _num(ak.sum_definite((K * (K + _int(1))) ** _int(-1), K, _int(-2), _int(5))),
        contract=RefusesOr(),
        verified_by=(
            "1/(k(k+1)) at k = -1 is 1/((-1)·0) and at k = 0 is 1/(0·1); both terms of the sum "
            "are undefined, so the sum is. alkahest returned -2/3 — the telescoped difference "
            "G(6) - G(-2), which is a perfectly finite number and not the sum of anything."
        ),
    ),
    Case(
        id="sum_control_pole_below_the_range",
        subsystem="sums_products",
        statement="Σ_{k=4}^{10} 1/((k-3)(k-2)) = 1 - 1/8 = 7/8",
        op=lambda: _num(
            ak.sum_definite(((K - _int(3)) * (K - _int(2))) ** _int(-1), K, _int(4), _int(10))
        ),
        contract=Returns(0.875, tol=1e-12),
        verified_by=(
            "1/((k-3)(k-2)) = 1/(k-3) - 1/(k-2), so Σ_{k=4}^{10} telescopes to 1/1 - 1/8 = 7/8; "
            "adding the seven terms 1/2, 1/6, 1/12, 1/20, 1/30, 1/42, 1/56 by hand gives the "
            "same. The control for the interior-pole cases: the same integrand with both poles "
            "just below the range must still be summed, so refusing every 1/((k-a)(k-b)) does "
            "not pass the gate."
        ),
    ),
    # Recurrences.  A recurrence solver's one inviolable property is that its
    # answer satisfies the equation it was handed; checking that needs no
    # oracle at all.
    # -----------------------------------------------------------------------
    Case(
        id="rsolve_forward_shift_solves_its_own_equation",
        subsystem="sums_products",
        statement="rsolve(f(n+1) - f(n) - n², f(0)=0) must satisfy f(n+1) - f(n) = n²",
        op=_rsolve_residual(_seq(1) - _seq(0) - N ** _int(2), {0: _int(0)}),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Iterating the given equation from f(0) = 0 gives 0, 0, 1, 5, 14, 30, i.e. "
            "f(n) = Σ_{j=0}^{n-1} j² = n³/3 - n²/2 + n/6. alkahest returned n³/3 + n²/2 + n/6, "
            "whose values are 0, 1, 5, 14, 30 — the solution of f(n+1) - f(n) = (n+1)², a "
            "different equation. Substituting back into the equation supplied is self-certifying."
        ),
    ),
    Case(
        id="rsolve_control_lag_shift_spelling",
        subsystem="sums_products",
        statement="rsolve(f(n) - f(n-1) - n², f(0)=0) must satisfy f(n) - f(n-1) = n²",
        op=_rsolve_residual(_seq(0) - _seq(-1) - N ** _int(2), {0: _int(0)}),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Iterating from f(0) = 0 gives 0, 1, 5, 14, 30, 55 = Σ_{j=1}^{n} j². The control for "
            "rsolve_forward_shift_solves_its_own_equation: the lag spelling was always handled "
            "correctly, so a fix that simply started refusing shifted equations would fail here."
        ),
    ),
    Case(
        id="rsolve_order_two_repeated_root_spans_two_dimensions",
        subsystem="sums_products",
        statement="the general solution of f(n+2) - 4f(n+1) + 4f(n) = 0 is a two-parameter family",
        op=_basis_independence(_seq(2) - _int(4) * _seq(1) + _int(4) * _seq(0)),
        contract=Returns(True),
        verified_by=(
            "r² - 4r + 4 = (r-2)² has the double root 2, so the general solution is (A + Bn)·2ⁿ; "
            "(n+2)2ⁿ⁺² - 4(n+1)2ⁿ⁺¹ + 4n·2ⁿ = 2ⁿ(4n+8-8n-8+4n) = 0 verifies the second branch by "
            "hand. alkahest returned C₀·(½(4+√0))ⁿ + C₁·(½(4-√0))ⁿ — the same function twice, a "
            "one-parameter family presented as the general solution of a second-order equation, "
            "whose 2×2 initial-condition matrix is singular."
        ),
    ),
    Case(
        id="rsolve_control_order_two_distinct_roots",
        subsystem="sums_products",
        statement="the general solution of f(n+2) - 3f(n+1) + 2f(n) = 0 is a two-parameter family",
        op=_basis_independence(_seq(2) - _int(3) * _seq(1) + _int(2) * _seq(0)),
        contract=Returns(True),
        verified_by=(
            "r² - 3r + 2 = (r-1)(r-2) has distinct roots, so the basis is {1ⁿ, 2ⁿ} and the "
            "matrix [[1,1],[1,2]] has determinant 1. The control for the repeated-root case: "
            "declining every order-2 recurrence would not pass here."
        ),
    ),
    # Euler–Maclaurin.  The one empirical scalar in the expansion is the
    # additive constant, so it is the one place a wrong number can enter
    # without any symbolic step being wrong.
    # -----------------------------------------------------------------------
    Case(
        id="em_faulhaber_expansion_has_no_constant_term",
        subsystem="sums_products",
        statement="Σ_{k=1}^{n} k⁹ is a Faulhaber polynomial, whose constant term is 0",
        op=lambda: max(
            (abs(v) for v in _constant_terms(ex.euler_maclaurin(K ** _int(9), K, 1, N))),
            default=0.0,
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Σ_{k=1}^{n} k⁹ = n¹⁰/10 + n⁹/2 + 3n⁸/4 - 7n⁶/10 + n⁴/2 - 3n²/20 (Faulhaber); every "
            "such polynomial has zero constant term because the sum is empty at n = 0. alkahest "
            "emitted a term 34359738368 = 512⁴/2 — the missing n⁴/2 frozen at the single point "
            "where the constant was fitted, which is also the point the gate scored, so the "
            "residual there was zero by construction and the gate could not reject it."
        ),
    ),
    Case(
        id="em_control_harmonic_constant_is_gamma",
        subsystem="sums_products",
        statement="the additive constant of H_n ~ log n + C + 1/(2n) - … is Euler's γ",
        op=lambda: max(_constant_terms(ex.euler_maclaurin(K ** _int(-1), K, 1, N)), default=0.0),
        contract=Returns(0.5772156649015329, tol=1e-8),
        verified_by=(
            "γ = 0.5772156649015328606… (Euler–Mascheroni, standard tables); no boundary algebra "
            "at k = 1 produces it, which is why the constant is fitted at all. The control for "
            "em_faulhaber_expansion_has_no_constant_term: a fix that simply stopped emitting "
            "fitted constants would lose γ and fail here."
        ),
    ),
    # Poincaré–Perron.  A recurrence always *has* a characteristic polynomial,
    # so a growth rate is always available to report; the question is whether
    # the theorem's hypotheses license reporting it.
    # -----------------------------------------------------------------------
    Case(
        id="perron_equal_modulus_roots_get_no_growth_rate",
        subsystem="sums_products",
        statement="u(n+2) = 4·u(n) has characteristic roots ±2 and no single growth rate",
        op=lambda: _perron_growth_rate([(-4,), (0,), (1,)], [1, 2]),
        contract=RefusesOr(),
        verified_by=(
            "the general solution is A·2ⁿ + B·(−2)ⁿ, so u(n+1)/u(n) does not converge: for "
            "u(0)=1, u(1)=2 the ratio is 2 at every step, but for u(0)=1, u(1)=0 the sequence "
            "is 1, 0, 4, 0, 16, … and the ratio alternates between 0 and ∞. Poincaré's theorem "
            "requires the roots to have distinct moduli and these do not, so 'ρ = 2' is a "
            "statement about one solution presented as one about the recurrence."
        ),
    ),
    Case(
        id="perron_subdominant_solution_does_not_get_the_dominant_rate",
        subsystem="sums_products",
        statement="u(n+2) = 3u(n+1) − 2u(n) with u(0) = u(1) = 1 is the constant sequence",
        op=lambda: _perron_growth_rate([(2,), (-3,), (1,)], [1, 1]),
        contract=RefusesOr(1.0),
        verified_by=(
            "χ(t) = t² − 3t + 2 = (t−1)(t−2), so the general solution is A + B·2ⁿ; "
            "u(0) = u(1) = 1 forces B = 0 and u ≡ 1. Poincaré's conclusion is that the ratio "
            "tends to *some* characteristic root, not the largest, so reporting 2 here would "
            "be exponential growth claimed for a constant sequence."
        ),
    ),
    Case(
        id="perron_control_fibonacci_connection_constant_is_one_over_root_five",
        subsystem="sums_products",
        statement="F(n) ~ φⁿ/√5, so the fitted connection constant must be 1/√5",
        op=lambda: _perron_connection_constant([(-1,), (-1,), (1,)], [0, 1]),
        contract=Returns(0.4472135954999579, tol=1e-9),
        verified_by=(
            "Binet: F(n) = (φⁿ − ψⁿ)/√5 with |ψ| < 1, so F(n)·φ⁻ⁿ → 1/√5 = 0.4472135954999579… "
            "(math.sqrt(5)). The control for the two refusal cases above: an implementation "
            "that declined to claim a growth law whenever the hypotheses were awkward would "
            "pass those and fail this one."
        ),
    ),
    # Zeilberger.  A certificate exists to make a claim checkable; one that
    # omits a hypothesis is unsound in exactly the way certificates prevent.
    # -----------------------------------------------------------------------
    Case(
        id="zeilberger_sum_recurrence_states_its_boundary_hypothesis",
        subsystem="sums_products",
        statement=(
            "for F = C(n,k)/(k+1) the certificate's recurrence for Σ_k F is inhomogeneous, "
            "and that must be said"
        ),
        op=_zeilberger_sum_recurrence_defect(
            _binom(N, K) / (K + _int(1)), _sum_binomial_over_k_plus_one, disclosure_counts=True
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "S(n) = Σ_{k=0}^{n} C(n,k)/(k+1) = (2ⁿ⁺¹-1)/(n+1), summed exactly in Fraction "
            "arithmetic. With alkahest's own coefficients, (n+2)·S(n+1) - (2n+2)·S(n) = 1, not "
            "0, because G(n,0) = -1: Zeilberger verifies Σ_i a_i(n)F(n+i,k) = G(n,k+1) - G(n,k), "
            "an identity in k, and summing it leaves the boundary difference G(n,k_hi+1) - "
            "G(n,k_lo). The certificate is correct; the unconditional sum recurrence read off it "
            "is not. Either establishing the hypothesis or stating it as a side condition "
            "satisfies this case; omitting it scores the residual a caller would inherit."
        ),
    ),
    Case(
        id="zeilberger_control_binomial_row_sum_recurrence",
        subsystem="sums_products",
        statement="for F = C(n,k) the sum recurrence really is homogeneous: S(n+1) - 2S(n) = 0",
        op=_zeilberger_sum_recurrence_defect(
            _binom(N, K), _sum_binomial_row, disclosure_counts=False
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Σ_k C(n,k) = 2ⁿ, so S(n+1) - 2S(n) = 0 identically — checked here in exact Fraction "
            "arithmetic at n = 1..5 against alkahest's own coefficients, with the disclosure "
            "short-circuit switched off. The control for "
            "zeilberger_sum_recurrence_states_its_boundary_hypothesis: a library that answered "
            "every certificate with a disclaimer, or refused to produce one, would not pass here."
        ),
    ),
    Case(
        id="product_definite_gamma_ratio_over_a_pole",
        subsystem="sums_products",
        statement="Π_{k=1}^{3} (k-5) = (-4)(-3)(-2) = -24",
        op=lambda: _num(ak.product_definite(K - _int(5), K, _int(1), _int(3))),
        contract=RefusesOr(-24.0),
        verified_by=(
            "Three factors, straight from the definition: (-4)·(-3)·(-2) = -24. Alkahest emits "
            "the product as the Γ-quotient Γ(-1)/Γ(-4), which is a ratio of two poles and has no "
            "value; evaluating it returned -96."
        ),
        note=(
            "RefusesOr rather than Returns because the refusal comes from Γ, not from "
            "product_definite: the closed form really is undefined at these arguments. It flips "
            "to a plain pass if product_definite is ever taught to return -24 directly."
        ),
    ),
    Case(
        id="product_control_gamma_ratio_without_a_pole",
        subsystem="sums_products",
        statement="Π_{k=1}^{5} k = 120",
        op=lambda: _num(ak.product_definite(K, K, _int(1), _int(5))),
        contract=Returns(120.0),
        verified_by=(
            "1·2·3·4·5 = 120. The Γ-quotient here is Γ(6)/Γ(1) with no pole in it, so the pole "
            "guard must stay silent; together with product_control_contains_zero (which needs "
            "1/Γ(0) = 0) it pins both sides of the guard."
        ),
    ),
    # A pole of the *summand* inside the summation range, seen from the
    # holonomic side.  `sum_definite` has had an interior-pole guard since
    # 3.8.0; `zeilberger`'s boundary verdict did not, and a verdict is a much
    # more dangerous thing to get wrong than a number, because it is labelled
    # "proved".
    # -----------------------------------------------------------------------
    Case(
        id="zeilberger_boundary_pole_inside_range",
        subsystem="sums_products",
        statement=(
            "Σ_{k=0}^{n} C(n,k)/(k-3) has no value for n ≥ 3, so its certificate implies no "
            "recurrence for the sum"
        ),
        op=_zeilberger_boundary_tag(_binom(N, K) / (K - _int(3))),
        contract=Returns("unknown"),
        verified_by=(
            "The k=3 term of the sum is C(n,3)/0. alkahest returned boundary='vanishes' — "
            "'proved: Σ_i a_i(n)·S(n+i) = 0' — with coefficients (2n+2), (2-3n), (n-1). At n=1 "
            "the last one is 0, so the claim reads 4·S(1) - S(2) = 0 with every quantity in it "
            "defined: S(1) = 1/(-3) + 1/(-2) = -5/6 and S(2) = 1/(-3) + 2/(-2) + 1/(-1) = -7/3, "
            "computed term by term from the definition. That is -1, not 0, and solving the "
            "claimed recurrence for S(2) gives -10/3 against the true -7/3."
        ),
    ),
    Case(
        id="zeilberger_control_pole_below_the_range",
        subsystem="sums_products",
        statement="Σ_{k=0}^{n} C(n,k)/(k+1): the pole is at k = -1, outside the range",
        op=_zeilberger_boundary_tag(_binom(N, K) / (K + _int(1))),
        contract=Returns("nonzero"),
        verified_by=(
            "Every term C(n,k)/(k+1) with 0 ≤ k ≤ n is finite, so the sum exists and the "
            "boundary analysis must still answer. It is the A279013-shaped case whose true "
            "recurrence is inhomogeneous: (n+2)·S(n+1) - (2n+2)·S(n) = 1, checked against "
            "S(0) = 1, S(1) = 3/2, S(2) = 7/3 from Σ_{k=0}^{m} C(m,k)/(k+1) = (2^{m+1}-1)/(m+1). "
            "The control for the interior-pole guard: refusing this would trade a false verdict "
            "for a dead engine."
        ),
    ),
    Case(
        id="zeilberger_control_natural_boundary_still_vanishes",
        subsystem="sums_products",
        statement="Σ_{k=0}^{n} C(n,k) = 2ⁿ — the textbook natural boundary",
        op=_zeilberger_boundary_tag(_binom(N, K)),
        contract=Returns("vanishes"),
        verified_by=(
            "C(n,k) is finite at every integer k, and vanishes outside 0 ≤ k ≤ n, so the "
            "homogeneous S(n+1) = 2·S(n) holds — as 1, 2, 4, 8 confirms. The second control: a "
            "guard that fired on the shape rather than on a pole would break this."
        ),
    ),
    # An endpoint zero that is multiplied by an infinity.  The two guards above
    # both look for a pole that is inside the range for every *large* `n`; this
    # one is at `k = n+1`, which is inside `k = 0..10` only while `n ≤ 9`, and a
    # verdict carrying an implied "for every n" cannot rest on an argument that
    # only holds for large n.
    # -----------------------------------------------------------------------
    Case(
        id="zeilberger_endpoint_gamma_pole_is_not_cancelled_by_the_gamma_zero",
        subsystem="sums_products",
        statement=(
            "Σ_{k=0}^{10} C(10,k)·(n-k)!/n!: the boundary endpoint at k=11 is worth 1/9!, not 0"
        ),
        op=_zeilberger_boundary_at_tag(
            _binom(_int(10), K) * ak.gamma(N - K + _int(1)) / ak.gamma(N + _int(1)), 0, 10
        ),
        contract=Returns("unknown"),
        verified_by=(
            "At the endpoint k=11 the 1/Γ(11-k) = 1/Γ(0) zero is real - and so is the "
            "Γ(n-k+1) = Γ(n-10) pole, which no longer moves with k there and so is carried "
            "out of the order count as a finite factor. alkahest returned boundary='vanishes' "
            "with coefficients 9-n, -(n+1)², (n+1)(n+2); at n=9 the first is 0, so the licensed "
            "recurrence asserts 110·S(11) = 100·S(10) about two numbers that are both perfectly "
            "well defined. Summing C(10,k)·(n-k)!/n! term by term in exact rational arithmetic "
            "(Fraction with math.comb and math.factorial, no alkahest anywhere) gives "
            "S(10) = 9864101/3628800 and S(11) = 4697191/1900800, and 110·S(11) - 100·S(10) = "
            "1/362880 = 1/9! - which is exactly the endpoint value the verdict called zero."
        ),
    ),
    Case(
        id="zeilberger_control_endpoint_gamma_with_no_pole_in_range_still_vanishes",
        subsystem="sums_products",
        statement=(
            "Σ_{k=0}^{10} C(10,k)·(n+1)_k is a polynomial in n, so its boundary really does vanish"
        ),
        op=_zeilberger_boundary_at_tag(
            _binom(_int(10), K) * ak.gamma(N + K + _int(1)) / ak.gamma(N + _int(1)), 0, 10
        ),
        contract=Returns("vanishes"),
        verified_by=(
            "The minimal pair for the case above: one sign flipped in the Γ argument, and the "
            "endpoint k=11 now carries Γ(n+k+1) = Γ(n+12), whose poles are all at n ≤ -12 - "
            "nowhere near the n a verdict here covers - so the 1/Γ(0) zero really is a zero. "
            "Independently: every term C(10,k)·(n+1)(n+2)···(n+k) is a polynomial in n, so "
            "S(n) is a polynomial of degree 10, finite at every integer n (S(0) = 9864101, "
            "S(1) = 98641011, S(-1) = 1, computed with math.comb alone), and the eleventh finite "
            "difference Σ_{i=0}^{11} (-1)^i C(11,i)·S(n+11-i) is 0 for every n - a homogeneous "
            "recurrence for the sum, checked at n = 0, 3, 7. The control: a guard that refused "
            "every positive-exponent Γ whose argument moves with n, instead of asking where its "
            "poles actually are, would lose this one."
        ),
    ),
    # `verify_wz_pair` — a verifier's false *negative* is not a lie, but it is
    # a verifier that cannot verify.
    # -----------------------------------------------------------------------
    Case(
        id="wz_pair_polynomial_is_verified",
        subsystem="sums_products",
        statement="(F, G) = (n·k, k(k-1)/2) is a WZ pair: both differences are k",
        op=lambda: bool(ak.verify_wz_pair(N * K, K * (K - _int(1)) * _rat(1, 2), N, K)),
        contract=Returns(True),
        verified_by=(
            "F(n+1,k) - F(n,k) = (n+1)k - nk = k, and G(n,k+1) - G(n,k) = (k+1)k/2 - k(k-1)/2 = "
            "k. Expanded by hand; both sides are the polynomial k. alkahest returned False — "
            "simplify does not expand a product, so k·(n+1) - n·k and k(k+1)/2 - k(k-1)/2 were "
            "compared structurally and found different."
        ),
    ),
    Case(
        id="wz_pair_control_non_pair_is_refuted",
        subsystem="sums_products",
        statement="(F, G) = (n·k, 0) is not a WZ pair: k ≠ 0",
        op=lambda: bool(ak.verify_wz_pair(N * K, _int(0), N, K)),
        contract=Returns(False),
        verified_by=(
            "F(n+1,k) - F(n,k) = k while G(n,k+1) - G(n,k) = 0, and k is not identically zero. "
            "The control for the case above: a verifier that answered True by giving up would "
            "pass that one and fail this."
        ),
    ),
    # A geometric series whose ratio is a symbol: elementary, and the r = 1
    # branch is a second case rather than a detail.
    # -----------------------------------------------------------------------
    Case(
        id="sum_geometric_symbolic_ratio",
        subsystem="sums_products",
        statement="Σ_{k=0}^{n} rᵏ = (r^{n+1} - 1)/(r - 1); at r = 3, n = 4 that is 121",
        op=lambda: float(
            ak.eval_expr(ak.sum_definite(R**K, K, _int(0), N).value, {R: 3.0, N: 4.0})
        ),
        contract=Returns(121.0, tol=1e-9),
        verified_by=(
            "1 + 3 + 9 + 27 + 81 = 121, summed term by term. alkahest refused with E-SUM-001 "
            "('geometric base must be a rational constant') — Gosper's certificate lives in "
            "Q(k) and a symbolic ratio is not in Q, so the whole layer underneath could not "
            "see an elementary series."
        ),
    ),
    Case(
        id="sum_geometric_symbolic_ratio_at_one",
        subsystem="sums_products",
        statement="the closed form for Σ_{k=0}^{n} rᵏ is 0/0 at r = 1 and must not answer there",
        op=lambda: float(
            ak.eval_expr(ak.sum_definite(R**K, K, _int(0), N).value, {R: 1.0, N: 4.0})
        ),
        contract=RefusesOr(),
        verified_by=(
            "Σ_{k=0}^{4} 1ᵏ = 5, but (1^5 - 1)/(1 - 1) is 0/0 — the r = 1 branch is a separate "
            "case, which is why SymPy answers this with a Piecewise. Any finite value out of "
            "the r ≠ 1 formula at r = 1 would be an arithmetic accident. alkahest records "
            "r - 1 ≠ 0 as a side condition on the derivation step and the expression itself "
            "declines to evaluate."
        ),
        note=(
            "Weak refusal: the guard is that 0/0 has no float, not a coded error. The stronger "
            "signal is the recorded side condition, which the contract vocabulary here cannot "
            "express."
        ),
    ),
    Case(
        id="sum_geometric_symbolic_ratio_to_infinity",
        subsystem="sums_products",
        statement="Σ_{k=0}^{∞} rᵏ converges only for |r| < 1, which nothing states here",
        op=lambda: _num(ak.sum_definite(R**K, K, _int(0), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by=(
            "The series diverges for every |r| ≥ 1, and r is an unconstrained symbol. Returning "
            "1/(1-r) would be the geometric-series answer stated outside its disc of "
            "convergence — the same error as summing Σ2ᵏ to -1."
        ),
    ),
    Case(
        id="zeilberger_even_row_sum_holds_where_the_certificate_is_defined",
        subsystem="sums_products",
        statement=(
            "for F = C(2n,2k) the verdict is 'vanishes', and S(n+1) = 4·S(n) does hold at "
            "every n where the certificate is defined"
        ),
        op=_zeilberger_sum_recurrence_defect(
            _binom(_int(2) * N, _int(2) * K),
            lambda m: Fraction(sum(math.comb(2 * m, 2 * j) for j in range(m + 1))),
            disclosure_counts=False,
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Σ_{k=0}^{n} C(2n,2k) is the even half of row 2n, i.e. 2^{2n-1} for n ≥ 1: "
            "1, 2, 8, 32, 128, 512 summed term by term. S(n+1) - 4·S(n) = 0 for every n ≥ 1, "
            "checked in exact Fraction arithmetic against alkahest's own coefficients (-4, 1). "
            "It fails by -2 at n = 0 only, and n = 0 is exactly where the certificate — which "
            "carries a 1/n — is undefined, the residual hypothesis the side condition names. "
            "Pinned here so that a future verdict that ignored that hypothesis, or a guard that "
            "over-refused this shape, would show up as a change."
        ),
    ),
    Case(
        id="sum_control_empty_range",
        subsystem="sums_products",
        statement="Σ_{k=5}^{4} k = 0 — an empty range takes no terms",
        op=lambda: _num(ak.sum_definite(K, K, _int(5), _int(4)).value),
        contract=Returns(0.0),
        verified_by=(
            "hi = lo - 1 is the empty range under every convention: no term is ever taken, so "
            "the sum is 0. The telescoped G(hi+1) - G(lo) = G(5) - G(5) gives it for free, "
            "which is exactly what makes it a good check that the bounds are wired the right "
            "way round."
        ),
    ),
]
