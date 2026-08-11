"""Ansatz families and coefficient fitting (``alkahest.ansatz``).

Covers the properties a conjecture-generation loop depends on: a known
polynomial round-trips exactly, an underdetermined family reports its free
parameters instead of picking a member, an inconsistent family raises
``E-ANSATZ-003`` (a *result* — the branch is closed), an oversized family is
refused before anything is materialised, the Padé/rational transform keeps the
solve linear, sample points are reproducible from a seed, and a fitted solution
drops into a :class:`~alkahest.research.ClaimGraph` unchanged.

The honesty contract is the point of most of these: :func:`alkahest.ansatz.fit`
may solve heuristically, but it grades itself by exact back-substitution, and
``"exactly_verified"`` is reachable only when the residual provably normalises
to zero.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import alkahest as ak
import pytest
from alkahest import ansatz as ansatz_module
from alkahest.ansatz import (
    DEFAULT_SEED,
    Ansatz,
    AnsatzSolution,
    certify_nonneg,
    enumerate_family,
    exponential_polynomial,
    fit,
    linear_combination,
    polynomial,
    quadratic_form,
    rational,
)
from alkahest.exceptions import AnsatzError
from alkahest.research import MACHINE_CHECKED_STATUSES, STATUS_BADGES


@pytest.fixture
def pool():
    return ak.ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


@pytest.fixture
def y(pool):
    return pool.symbol("y")


def assert_identical(left, right) -> None:
    """Assert ``left - right`` is identically zero.

    The kernel has several normalisers with different strengths (``simplify``
    does not collect like terms with rational coefficients; ``cancel`` refuses
    non-integer coefficients outright), so a round-trip check tries each of the
    public ones and passes if any of them closes.
    """
    difference = left - right
    for normalise in (ak.simplify_expanded, ak.cancel, ak.simplify, ak.together):
        try:
            candidate = normalise(difference)
        except Exception:
            continue
        candidate = getattr(candidate, "value", candidate)
        if str(candidate).strip() == "0":
            return
    raise AssertionError(f"{left} is not identical to {right}")


def _fit_code(*args, **kwargs) -> str:
    """Return the ``E-ANSATZ-*`` code raised by a call, failing if none is."""
    with pytest.raises(AnsatzError) as excinfo:
        fit(*args, **kwargs)
    return excinfo.value.code


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_reachable_from_package_root():
    assert "ansatz" in ak.__all__
    assert ak.ansatz.fit is fit


def test_public_names_are_exported():
    for name in (
        "Ansatz",
        "AnsatzSolution",
        "certify_nonneg",
        "enumerate_family",
        "exponential_polynomial",
        "fit",
        "linear_combination",
        "polynomial",
        "quadratic_form",
        "rational",
    ):
        assert name in ak.ansatz.__all__, f"{name} missing from alkahest.ansatz.__all__"
        assert hasattr(ak.ansatz, name)


# ---------------------------------------------------------------------------
# Family construction
# ---------------------------------------------------------------------------


def test_polynomial_names_are_predictable_and_graded(pool, x, y):
    univariate = polynomial(pool, [x], degree=3)
    assert univariate.names == ("c_0", "c_1", "c_2", "c_3")

    bivariate = polynomial(pool, [x, y], degree=2, name="a")
    # Graded, then lexicographic with the first variable heaviest.
    assert bivariate.names == ("a_0_0", "a_1_0", "a_0_1", "a_2_0", "a_1_1", "a_0_2")
    assert len(bivariate) == len(bivariate.unknowns) == len(bivariate.basis) == 6
    assert bivariate.vars == (x, y)
    assert set(bivariate.symbols) == set(bivariate.names)


def test_polynomial_min_degree_gives_homogeneous_forms(pool, x, y):
    homogeneous = polynomial(pool, [x, y], degree=2, min_degree=2, name="h")
    assert homogeneous.names == ("h_2_0", "h_1_1", "h_0_2")


def test_unknowns_and_vars_stay_disjoint(pool, x):
    family = polynomial(pool, [x], degree=2)
    assert set(map(str, family.unknowns)).isdisjoint(map(str, family.vars))


def test_coefficient_name_collision_is_refused(pool, x):
    # The pool already has a symbol c_0 reachable from the variables handed in,
    # so a family prefixed "c" would silently fit the wrong thing.
    collider = pool.symbol("c_0")
    with pytest.raises(AnsatzError) as excinfo:
        polynomial(pool, [x], degree=1, reserved=[collider])
    assert excinfo.value.code == "E-ANSATZ-001"
    assert "name" in (excinfo.value.remediation or "")
    # ... and a different prefix is accepted rather than gensym-ed.
    assert polynomial(pool, [x], degree=1, name="k", reserved=[collider]).names == ("k_0", "k_1")


def test_collision_check_sees_symbols_of_the_variables(pool):
    weird = pool.symbol("c_0")
    with pytest.raises(AnsatzError) as excinfo:
        polynomial(pool, [weird], degree=1)
    assert excinfo.value.code == "E-ANSATZ-001"


def test_with_prefix_renames_the_whole_family(pool, x):
    family = polynomial(pool, [x], degree=2)
    renamed = family.with_prefix("t")
    assert renamed.names == ("t_0", "t_1", "t_2")
    assert renamed.family == family.family


def test_rational_denominator_is_monic_by_default(pool, x):
    family = rational(pool, [x], num_degree=2, den_degree=2)
    assert family.names == ("a_0", "a_1", "a_2", "b_1", "b_2")
    assert family.metadata["numerator_terms"] == 3
    # Without the normalisation p/q and (2p)/(2q) are the same function, so the
    # extra denominator constant would show up as a spurious free parameter.
    unnormalised = rational(pool, [x], 1, 1, name="u", den_name="v", monic_denominator=False)
    assert unnormalised.names == ("u_0", "u_1", "v_0", "v_1")


def test_exponential_polynomial_shape(pool, x):
    family = exponential_polynomial(pool, x, [1, -1], degree=1)
    assert family.names == ("c_0_0", "c_0_1", "c_1_0", "c_1_1")
    assert family.metadata["degrees"] == (1, 1)


def test_quadratic_form_carries_only_the_upper_triangle(pool, x, y):
    family = quadratic_form(pool, [x, y])
    assert family.names == ("q_0_0", "q_0_1", "q_1_1")


def test_linear_combination_infers_its_variables(pool, x):
    family = linear_combination(pool, [ak.sin(x), ak.cos(x)])
    assert family.names == ("c_0", "c_1")
    assert [str(v) for v in family.vars] == ["x"]


def test_reserved_names_do_not_become_independent_variables(pool, x, y):
    # `reserved=` says "this name is taken", not "this is a variable". Inferring
    # vars from it would silently change what the fit is asked to prove: the
    # family below is a combination of sin and cos in x alone, and y is only
    # kept away from the coefficient names.
    family = linear_combination(pool, [ak.sin(x), ak.cos(x)], name="rv", reserved=[y])
    assert [str(v) for v in family.vars] == ["x"]
    solution = fit(family, family.expr - ak.sin(x))
    assert solution.status == "exactly_verified"
    # ... and the reservation still does its own job.
    with pytest.raises(AnsatzError) as excinfo:
        linear_combination(pool, [ak.sin(x)], name="k", reserved=[pool.symbol("k_0")])
    assert excinfo.value.code == "E-ANSATZ-001"


def test_max_terms_refuses_before_materialising(pool, x, y):
    with pytest.raises(AnsatzError) as excinfo:
        polynomial(pool, [x, y], degree=40, max_terms=32)
    assert excinfo.value.code == "E-ANSATZ-002"
    assert "861" in str(excinfo.value)  # C(42, 2) — counted, never built
    assert "max_terms" in (excinfo.value.remediation or "")


def test_max_terms_is_checked_for_every_family(pool, x, y):
    for call in (
        lambda: rational(pool, [x], num_degree=20, den_degree=20, max_terms=8),
        lambda: quadratic_form(pool, [x, y], max_terms=2),
        lambda: exponential_polynomial(pool, x, [1, 2], degree=9, max_terms=4),
        lambda: linear_combination(pool, [x, y, x * y], max_terms=2),
    ):
        with pytest.raises(AnsatzError) as excinfo:
            call()
        assert excinfo.value.code == "E-ANSATZ-002"


def test_instantiate_accepts_names_and_exprs(pool, x):
    family = polynomial(pool, [x], degree=1)
    assert str(family.instantiate({"c_0": 2, "c_1": 0})) == "2"
    assert str(family.instantiate({family.unknowns[0]: 0, family.unknowns[1]: 1})) == "x"


def test_instantiate_rejects_an_unknown_name(pool, x):
    family = polynomial(pool, [x], degree=1)
    with pytest.raises(AnsatzError) as excinfo:
        family.instantiate({"nope": 1})
    assert excinfo.value.code == "E-ANSATZ-001"


# ---------------------------------------------------------------------------
# Enumeration (stage 2)
# ---------------------------------------------------------------------------


def test_enumerate_family_is_lazy_and_complete(pool, x):
    family = polynomial(pool, [x], degree=1)
    members = [str(m) for m in enumerate_family(family, [0, 1])]
    assert members == ["0", "x", "1", "(x + 1)"]


def test_enumerate_family_is_bounded_before_the_first_member(pool, x):
    family = polynomial(pool, [x], degree=5)
    generator = enumerate_family(family, range(100), max_members=1000)
    with pytest.raises(AnsatzError) as excinfo:
        next(generator)
    assert excinfo.value.code == "E-ANSATZ-002"


def test_enumerate_family_output_is_evaluable(pool, x):
    family = polynomial(pool, [x], degree=2, name="e")
    for member in enumerate_family(family, [-1, 1]):
        assert isinstance(ak.eval_expr(member, {x: 0.5}), float)


# ---------------------------------------------------------------------------
# Fitting — the round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("coefficients", "expected_terms"),
    [
        ((2, -3, 1), {"2", "x^2"}),
        ((0, 0, 5), {"x^2"}),
        ((7, 0, 0), {"7"}),
    ],
)
def test_polynomial_round_trip_recovers_the_target(pool, x, coefficients, expected_terms):
    family = polynomial(pool, [x], degree=2)
    target = pool.integer(0)
    for power, coefficient in enumerate(coefficients):
        target = target + pool.integer(coefficient) * x**power
    target = ak.simplify(target).value

    solution = fit(family, family.expr - target)

    assert solution.status == "exactly_verified"
    assert solution.free == ()
    assert solution.determined
    assert solution.rank == len(family)
    assert_identical(solution.expr, target)
    assert expected_terms <= set(str(solution.expr).replace("(", " ").replace(")", " ").split())


def test_multivariate_round_trip(pool, x, y):
    family = polynomial(pool, [x, y], degree=2, name="m")
    target = x * y + y**2 - pool.integer(3)
    solution = fit(family, family.expr - target)
    assert solution.status == "exactly_verified"
    assert solution.rank == len(family)
    assert_identical(solution.expr, target)


def test_linear_combination_round_trip(pool, x):
    family = linear_combination(pool, [ak.sin(x), ak.cos(x)], name="L")
    target = pool.integer(3) * ak.sin(x) - pool.integer(2) * ak.cos(x)
    solution = fit(family, family.expr - target)
    assert solution.status == "exactly_verified"
    assert solution.rank == 2


def test_exponential_polynomial_round_trip(pool, x):
    family = exponential_polynomial(pool, x, [1, -1], name="g")
    target = pool.integer(2) * ak.exp(x) + pool.integer(3) * ak.exp(-x)
    solution = fit(family, family.expr - target)
    assert solution.status == "exactly_verified"
    assert solution.rank == 2


def test_quadratic_form_round_trip(pool, x, y):
    family = quadratic_form(pool, [x, y], name="w")
    target = x * x + pool.integer(2) * x * y + pool.integer(5) * y * y
    solution = fit(family, family.expr - target)
    assert solution.status == "exactly_verified"
    assert solution.rank == 3


def test_fit_result_is_an_ansatz_solution(pool, x):
    family = polynomial(pool, [x], degree=1)
    solution = fit(family, family.expr - x)
    assert isinstance(solution, AnsatzSolution)
    assert isinstance(solution.ansatz, Ansatz)
    assert solution.value is solution.expr
    assert solution.certificate is None
    assert solution.points  # the points that built the system, for reproduction
    assert "AnsatzSolution" in repr(solution)


# ---------------------------------------------------------------------------
# Fitting — underdetermined, inconsistent, and over-sampling
# ---------------------------------------------------------------------------


def test_underdetermined_family_returns_free_parameters(pool, x):
    # d/dx of a cubic vanishing identically pins every coefficient but the
    # constant: the members that work are a one-dimensional family.
    family = polynomial(pool, [x], degree=3, name="d")
    solution = fit(family, ak.diff(family.expr, x).value)

    assert solution.rank == 3
    assert [str(f) for f in solution.free] == ["d_0"]
    assert not solution.determined
    assert solution.status == "exactly_verified"
    # No arbitrary member was picked: the free parameter is still symbolic.
    assert "d_0" in str(solution.expr)


def test_underdetermined_family_is_solved_for_any_free_value(pool, x):
    family = polynomial(pool, [x], degree=3, name="u")
    solution = fit(family, ak.diff(family.expr, x).value)
    free = solution.free[0]
    for value in (-2, 0, 7):
        member = ak.subs(solution.expr, {free: value})
        assert_identical(ak.diff(member, x).value, pool.integer(0))


def test_dependent_basis_shows_up_as_a_free_parameter(pool, x):
    # sin and 2*sin are dependent, so fitting them to sin(x) is a family.
    family = linear_combination(pool, [ak.sin(x), pool.integer(2) * ak.sin(x)], name="p")
    solution = fit(family, family.expr - ak.sin(x))
    assert solution.rank == 1
    assert len(solution.free) == 1


def test_inconsistent_family_raises_e_ansatz_003(pool, x):
    family = polynomial(pool, [x], degree=1, name="s")
    with pytest.raises(AnsatzError) as excinfo:
        fit(family, family.expr - x**2)
    error = excinfo.value
    assert error.code == "E-ANSATZ-003"
    # Worded as a result, not a malfunction.
    assert "no member of this" in str(error)
    assert "result, not a malfunction" in (error.remediation or "")


def test_inconsistency_survives_enlarging_only_by_enlarging(pool, x):
    small = polynomial(pool, [x], degree=1, name="v")
    assert _fit_code(small, small.expr - x**3) == "E-ANSATZ-003"
    big = polynomial(pool, [x], degree=3, name="V")
    assert fit(big, big.expr - x**3).status == "exactly_verified"


def test_rank_is_read_off_the_reduction_not_the_point_count(pool, x):
    # 4 unknowns, and fit() draws strictly more equations than that; a family
    # that is only rank 3 must still report 3.
    family = polynomial(pool, [x], degree=3, name="r")
    solution = fit(family, ak.diff(family.expr, x).value)
    assert solution.rank == 3 < len(family)


def test_oversample_can_be_tightened(pool, x):
    family = polynomial(pool, [x], degree=2, name="o")
    solution = fit(family, family.expr - x**2, oversample=0)
    assert solution.rank == 3
    assert len(solution.points) == 3


def test_points_with_a_vanishing_denominator_are_skipped(pool, x):
    # Put a removable singularity exactly where the sampler looks first: the
    # fit must skip that point rather than build a row out of 0/0.
    probe = fit(polynomial(pool, [x], degree=1, name="pp"), pool.symbol("pp_1") * pool.integer(0))
    pole = Fraction(probe.points[0][0])
    pole_expr = pool.rational(pole.numerator, pole.denominator)

    family = polynomial(pool, [x], degree=1, name="q")
    target = (x * x - pole_expr * pole_expr) / (x - pole_expr)  # = x + pole, undefined at x = pole
    solution = fit(family, family.expr - target)

    assert all(point[0] != str(pole) for point in solution.points)
    assert solution.rank == 2
    assert_identical(solution.expr, x + pole_expr)


# ---------------------------------------------------------------------------
# Fitting — the rational / Padé transform
# ---------------------------------------------------------------------------


def test_rational_fit_recovers_a_rational_target(pool, x):
    family = rational(pool, [x], num_degree=1, den_degree=1)
    target = (x + pool.integer(2)) / (x - pool.integer(1))
    solution = fit(family, family.expr - target)
    assert solution.status == "exactly_verified"
    assert_identical(solution.expr, target)


def test_rational_fit_clears_the_denominator_automatically(pool, x):
    family = rational(pool, [x], num_degree=1, den_degree=1, name="n", den_name="m")
    target = pool.integer(1) / (pool.integer(1) + x)
    solution = fit(family, family.expr - target)
    rules = [step["rule"] for step in solution.steps]
    assert "ansatz_clear_denominator" in rules
    assert solution.status == "exactly_verified"


def test_ansatz_residual_states_the_denominator_clearing(pool, x):
    family = rational(pool, [x], num_degree=1, den_degree=1, name="p", den_name="r")
    residual = family.residual(pool.integer(1))
    # p - 1*q is affine in the unknowns; p/q - 1 is not.
    assert "^-1" not in str(residual)
    solution = fit(family, residual)
    assert solution.status == "exactly_verified"


def test_pade_of_exp_matches_the_taylor_coefficients(pool, x):
    # A Padé approximant is a *local* match, not an identity, so this is the
    # exact-system route with an explicit degree bound.
    family = rational(pool, [x], num_degree=2, den_degree=2, name="u", den_name="v")
    solution = fit(family, family.residual(ak.exp(x)), certify="exact", degree_bound=4)

    # The (2,2) Pade approximant of exp: (1 + x/2 + x^2/12) / (1 - x/2 + x^2/12).
    by_name = {str(k): str(v) for k, v in solution.assignment.items()}
    assert by_name == {
        "u_0": "1",
        "u_1": "1/2",
        "u_2": "1/12",
        "v_1": "-1/2",
        "v_2": "1/12",
    }

    # ... and the honesty gate correctly refuses to call it an identity.
    assert solution.status == "numerically_checked"
    assert solution.verification["residual"] != "0"


def test_fitting_a_rational_family_to_exp_as_an_identity_is_inconsistent(pool, x):
    family = rational(pool, [x], num_degree=2, den_degree=2, name="i", den_name="j")
    assert _fit_code(family, family.residual(ak.exp(x))) == "E-ANSATZ-003"


# ---------------------------------------------------------------------------
# Fitting — the honesty contract
# ---------------------------------------------------------------------------


def test_exactly_verified_only_when_the_residual_normalises_to_zero(pool, x):
    family = polynomial(pool, [x], degree=2, name="hv")
    solution = fit(family, family.expr - (x**2 + pool.integer(1)))
    assert solution.verification["status"] == "exactly_verified"
    assert solution.verification["residual"] == "0"
    assert solution.verification["externally_verified"] is False
    assert solution.status in MACHINE_CHECKED_STATUSES


def test_status_strings_come_from_the_existing_vocabulary(pool, x):
    family = polynomial(pool, [x], degree=1, name="sv")
    for certify, expected in (("residual", "exactly_verified"), ("none", "unverified")):
        solution = fit(family, family.expr - x, certify=certify)
        assert solution.status == expected
        assert solution.status in STATUS_BADGES
        assert solution.badge == STATUS_BADGES[expected]


def test_certify_none_records_no_verification_and_no_recheck_recipe(pool, x):
    family = polynomial(pool, [x], degree=1, name="cn")
    solution = fit(family, family.expr - x, certify="none")
    assert solution.verification["evidence"] == "none"
    assert solution.check == {}


def test_certify_rejects_an_unknown_mode(pool, x):
    family = polynomial(pool, [x], degree=1, name="cr")
    with pytest.raises(ValueError, match="certify"):
        fit(family, family.expr - x, certify="wishful")


def test_derivation_log_uses_the_step_schema(pool, x):
    family = polynomial(pool, [x], degree=2, name="dl")
    solution = fit(family, family.expr - x**2)
    assert solution.steps
    for step in solution.steps:
        assert set(ak.STEP_FIELDS) <= set(step)
        assert isinstance(step["side_conditions"], list)
    assert [s["rule"] for s in solution.steps][-1] == "ansatz_back_substitution"


def test_collocation_and_derivative_extraction_agree(pool, x, y):
    for family, target in (
        (polynomial(pool, [x], degree=3, name="A1"), x**3 - pool.integer(2) * x),
        (polynomial(pool, [x, y], degree=2, name="A2"), x * y + pool.integer(4)),
    ):
        by_points = fit(family, family.expr - target)
        by_taylor = fit(family, family.expr - target, certify="exact")
        assert by_points.rank == by_taylor.rank
        assert by_points.status == by_taylor.status == "exactly_verified"
        assert_identical(by_points.expr, by_taylor.expr)


def _break_probe_after(monkeypatch, successes: int) -> None:
    """Make ``_probe`` fail from its *successes*-th call onward, every attempt."""
    real_probe = ansatz_module._probe
    calls = itertools.count()

    def flaky(*args, **kwargs):
        if next(calls) % (successes + 1) == successes:
            raise ZeroDivisionError("undefined here")
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(ansatz_module, "_probe", flaky)


def test_a_half_built_taylor_system_is_never_returned(pool, x, monkeypatch):
    # Every base point dies on its second multi-index. The rows that did get
    # built are not a system anybody asked for, and returning them alongside a
    # shorter list of multi-indices would misreport which equations exist.
    family = polynomial(pool, [x], degree=2, name="ht")
    _break_probe_after(monkeypatch, successes=1)
    rows, used, skipped = ansatz_module._derivative_rows(
        family, family.expr - x**2, seed=DEFAULT_SEED, degree_bound=3
    )
    assert rows == []
    assert used == []
    assert skipped == 8


def test_an_unbuildable_taylor_system_is_reported_not_silently_truncated(pool, x, monkeypatch):
    family = polynomial(pool, [x], degree=2, name="ut")
    _break_probe_after(monkeypatch, successes=1)
    with pytest.raises(AnsatzError) as excinfo:
        fit(family, family.expr - x**2, certify="exact")
    assert excinfo.value.code == "E-ANSATZ-003"
    # Worded as the build failure it is, not as a claim about the family.
    assert "could not extract the Taylor system" in str(excinfo.value)


def test_max_points_does_not_shrink_the_taylor_system(pool, x):
    # max_points caps sample-point *draws*; the Taylor path draws one base
    # point and derives its equations from the degree bound, so a small cap
    # must not quietly hand back a smaller (weaker) system.
    family = polynomial(pool, [x], degree=3, name="mp")
    residual = family.expr - x**3

    def extraction(solution):
        return next(s for s in solution.steps if s["rule"] == "ansatz_taylor_extraction")

    uncapped = fit(family, residual, certify="exact")
    capped = fit(family, residual, certify="exact", max_points=2)
    assert extraction(capped)["after"] == extraction(uncapped)["after"]
    assert capped.rank == uncapped.rank == 4
    assert capped.status == "exactly_verified"
    assert capped.free == ()
    # The bound the system actually covers is stated, not implied.
    assert any("total degree <=" in c for c in extraction(capped)["side_conditions"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_sample_points_are_reproducible_without_a_budget(pool, x):
    family = polynomial(pool, [x], degree=2, name="D1")
    first = fit(family, family.expr - x**2)
    second = fit(family, family.expr - x**2)
    assert first.points == second.points


def test_budget_seed_drives_the_sample_points(pool, x):
    family = polynomial(pool, [x], degree=2, name="D2")
    residual = family.expr - x**2

    with ak.context(budget=ak.Budget(seed=7)):
        seven_a = fit(family, residual)
    with ak.context(budget=ak.Budget(seed=7)):
        seven_b = fit(family, residual)
    with ak.context(budget=ak.Budget(seed=99)):
        ninety_nine = fit(family, residual)

    assert seven_a.points == seven_b.points
    assert seven_a.points != ninety_nine.points
    # Different points, same mathematics.
    assert_identical(seven_a.expr, ninety_nine.expr)


def test_explicit_seed_overrides_the_budget(pool, x):
    family = polynomial(pool, [x], degree=2, name="D3")
    residual = family.expr - x**2
    with ak.context(budget=ak.Budget(seed=7)):
        overridden = fit(family, residual, seed=DEFAULT_SEED)
    assert overridden.points == fit(family, residual).points


def test_the_default_seed_is_a_fixed_constant():
    # Two machines with no budget active must draw the same points.
    assert isinstance(DEFAULT_SEED, int)


@pytest.mark.timeout(120)
def test_the_sampler_outlives_its_initial_draw_window():
    # A fixed draw window (-12..12 over 1..4) holds only 65 distinct points in
    # one variable. Asking for more than that from a stream that rejects
    # repeats but never widens hangs forever inside next(), which no caller's
    # attempt counter can interrupt, so this must run rather than wedge.
    points = list(itertools.islice(ansatz_module._sample_points(1, DEFAULT_SEED), 400))
    assert len(set(points)) == 400
    # Determinism is unaffected: the same seed still replays the same stream,
    # starting from the same first points.
    again = list(itertools.islice(ansatz_module._sample_points(1, DEFAULT_SEED), 400))
    assert again == points


@pytest.mark.timeout(120)
def test_a_fit_can_ask_for_more_points_than_the_initial_window_holds(pool, x):
    # Same hazard reached through the public API: 203 collocation rows in one
    # variable is more than the un-widened window could ever supply.
    family = polynomial(pool, [x], degree=2, name="wide")
    solution = fit(family, family.expr - x**2, oversample=200)
    assert len(solution.points) == 203
    assert len(set(solution.points)) == 203
    assert solution.status == "exactly_verified"
    assert_identical(solution.expr, x**2)


# ---------------------------------------------------------------------------
# Nonlinear escalation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ak.capabilities().get("groebner", False), reason="needs a groebner build")
def test_nonlinear_residual_escalates_to_solve(pool, x):
    family = polynomial(pool, [x], degree=0, name="nl")
    unknown = family.unknowns[0]
    solution = fit(family, unknown * unknown - pool.integer(4) + pool.integer(0) * x)
    assert "ansatz_nonlinear_escalation" in [s["rule"] for s in solution.steps]
    assert str(solution.expr) in {"2", "-2"}
    assert solution.status == "exactly_verified"


@pytest.mark.skipif(
    ak.capabilities().get("groebner", False), reason="only meaningful without groebner"
)
def test_nonlinear_residual_refuses_without_groebner(pool, x):  # pragma: no cover - build gated
    family = polynomial(pool, [x], degree=0, name="nl")
    unknown = family.unknowns[0]
    assert _fit_code(family, unknown * unknown - pool.integer(4)) == "E-ANSATZ-004"


@pytest.mark.skipif(not ak.capabilities().get("groebner", False), reason="needs a groebner build")
def test_unsatisfiable_nonlinear_residual_is_also_e_ansatz_003(pool, x):
    # c^2 * x = 1 cannot hold at two different sample points at once.
    family = polynomial(pool, [x], degree=0, name="ns")
    unknown = family.unknowns[0]
    assert _fit_code(family, unknown * unknown * x - pool.integer(1)) == "E-ANSATZ-003"


# ---------------------------------------------------------------------------
# Hand-off to positivity
# ---------------------------------------------------------------------------


def test_certify_nonneg_hands_a_lyapunov_candidate_to_prove_nonneg(pool, x, y):
    family = quadratic_form(pool, [x, y], name="P")
    target = x * x + pool.integer(2) * x * y + pool.integer(5) * y * y
    solution = fit(family, family.expr - target)

    certificate = certify_nonneg(solution)
    # Returned verbatim from alkahest.prove_nonneg — nothing reinterpreted here.
    assert isinstance(certificate, ak.PositivityCertificate)
    assert certificate.verify()


def test_certify_nonneg_passes_a_refutation_through(pool, x, y):
    family = quadratic_form(pool, [x, y], name="N")
    target = x * x - pool.integer(4) * y * y
    solution = fit(family, family.expr - target)
    with pytest.raises(ak.SosError) as excinfo:
        certify_nonneg(solution)
    # The positivity subsystem's own three-valued outcome, unmodified.
    assert excinfo.value.code.startswith("E-SOS-")


def test_certify_nonneg_refuses_an_undetermined_family(pool, x, y):
    family = quadratic_form(pool, [x, y], name="U")
    solution = fit(family, pool.integer(0) * family.expr)
    assert solution.free
    with pytest.raises(AnsatzError) as excinfo:
        certify_nonneg(solution)
    assert excinfo.value.code == "E-ANSATZ-003"
    assert "family, not a candidate" in str(excinfo.value)


def test_certify_nonneg_accepts_a_bare_expression(pool, x):
    assert certify_nonneg(x * x, [x]).verify()


# ---------------------------------------------------------------------------
# Claim-graph integration
# ---------------------------------------------------------------------------


def test_solution_records_into_a_claim_graph_unchanged(pool, x):
    family = polynomial(pool, [x], degree=2, name="cg")
    with ak.research.session(title="ansatz", pool=pool) as session:
        solution = fit(family, family.expr - (x**2 + pool.integer(1)))
        claim = session.record(
            solution,
            method="ansatz.fit",
            label="degree-2 fit",
            check=solution.check,
        )

    assert claim.status == "exactly_verified"
    assert claim.machine_checked
    assert claim.badge == STATUS_BADGES["exactly_verified"]
    assert claim.derivation  # the fit's own steps travelled with it
    assert claim.certificate is None
    assert len(session.graph) == 1
    assert session.graph.summary() == {"exactly_verified": 1}


def test_recorded_solution_survives_a_json_round_trip(pool, x):
    family = polynomial(pool, [x], degree=1, name="js")
    with ak.research.session(title="ansatz", pool=pool) as session:
        session.record(fit(family, family.expr - x), method="ansatz.fit")
    restored = ak.research.ClaimGraph.from_json(session.graph.to_json())
    assert restored.digest() == session.graph.digest()


def test_recheck_recipe_does_not_refute_a_good_fit(pool, x):
    family = polynomial(pool, [x], degree=2, name="rc")
    with ak.research.session(title="ansatz", pool=pool) as session:
        solution = fit(family, family.expr - (x**2 - pool.integer(1)))
        session.record(solution, method="ansatz.fit", check=solution.check)
    report = session.graph.verify()
    assert report.ok
    assert report.failed == ()


def test_an_unverified_fit_is_recorded_as_unverified(pool, x):
    family = polynomial(pool, [x], degree=1, name="uv")
    with ak.research.session(title="ansatz", pool=pool) as session:
        claim = session.record(fit(family, family.expr - x, certify="none"), method="ansatz.fit")
    assert claim.status == "unverified"
    assert not claim.machine_checked


# ---------------------------------------------------------------------------
# Composition with the rest of the loop plumbing
# ---------------------------------------------------------------------------


def test_fit_composes_with_batch_map(pool, x):
    families = [polynomial(pool, [x], degree=d, name=f"b{d}") for d in (1, 2, 3)]
    target = x**2

    outcomes = ak.batch_map(lambda f: fit(f, f.expr - target), families)
    assert [o.ok for o in outcomes] == [False, True, True]
    # The failing candidate carries the ansatz code, not a generic batch code.
    assert (outcomes[0].error or {})["code"] == "E-ANSATZ-003"


def test_enumerated_members_compile(pool, x):
    family = polynomial(pool, [x], degree=1, name="jt")
    for member in enumerate_family(family, [0, 1]):
        compiled = ak.compile_expr(member, [x])
        assert isinstance(compiled([0.5]), float)
