"""M9 — Gröbner bases over the coefficient field ``Q(params)``.

``GroebnerBasis.compute(polys, vars, params=[...])`` moves the listed symbols
into the *coefficient field* instead of the polynomial ring: they never enter
the monomial order, never generate S-pairs, and never enlarge the staircase.
The trade is that the basis is only correct where its leading coefficients —
now elements of ``Q(params)`` rather than ``Q`` — do not vanish, so every
computation also returns the hypersurfaces it assumed non-zero
(:meth:`~alkahest.experimental.ParametricGroebnerBasis.conditions`) and
refuses to :meth:`~alkahest.experimental.ParametricGroebnerBasis.specialize`
on them.

These tests check three things the Rust unit tests (`alkahest-core/src/poly/
groebner/parametric.rs`) already cover at the data-structure level, but not
through the Python surface a caller actually uses:

1. specialising the generic basis at a regular point agrees with computing
   the basis directly over ℚ at that same point — the actual mathematical
   content of the ``Q(params)`` result;
2. the degeneracy locus is genuinely a locus of *disagreement*, not a
   conservative label that never fires — both by exhibiting a point where the
   direct and specialised bases differ, and a point that is flagged but
   happens to agree anyway (the "sufficient, not necessary" clause in the
   docs);
3. the whole thing reads back as :class:`~alkahest.Expr`, end to end, on a
   structural-identifiability example — the reason the feature exists at all.
"""

from __future__ import annotations

from fractions import Fraction

import alkahest as ak
import alkahest.experimental as ake
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(ake, "ParametricGroebnerBasis"),
    reason="native module built without groebner feature",
)


@pytest.fixture
def pool():
    return ak.ExprPool()


def _exact_value(pool, expr, mapping):
    """Evaluate *expr* at exact (int/Fraction) bindings, as an exact
    :class:`Fraction`.

    ``eval_expr`` returns an IEEE double, which is the wrong tool here: the
    input-output relation's coefficients are large exact rationals, and
    comparing them via ``float`` rounds away the distinctions this test
    exists to check. ``ak.evaluate`` with :class:`Fraction` bindings routes
    through the ``exact_rational`` backend and returns an exact
    :class:`Fraction`, never a float approximation.
    """
    del pool  # kept for a uniform call signature across the test file
    result = ak.evaluate(expr, mapping)
    assert result.backend == "exact_rational", (
        f"expected the exact-rational backend, got {result.backend!r}"
    )
    return result.value


# ---------------------------------------------------------------------------
# 1. The basic shape: conditions, specialize, vanishing_conditions
# ---------------------------------------------------------------------------


def test_compute_with_params_returns_a_parametric_basis(pool):
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])

    assert isinstance(gb, ake.ParametricGroebnerBasis)
    assert gb.n_params == 1
    assert gb.order == "lex"
    assert [str(v) for v in gb.variables()] == ["x", "y"]
    assert [str(p) for p in gb.parameters()] == ["a"]


def test_omitting_params_returns_the_ordinary_basis(pool):
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    # No params kwarg, and params=[] both take the Q[vars] path unchanged.
    gb1 = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y, a])
    gb2 = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y, a], params=[])
    assert isinstance(gb1, ak.GroebnerBasis)
    assert isinstance(gb2, ak.GroebnerBasis)


def test_conditions_flag_the_actual_singular_locus(pool):
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])
    conds = [str(c) for c in gb.conditions()]

    # y = a/(a+1), x = 1/(a+1): the system is singular exactly at a = -1.
    assert "(a + 1)" in conds
    assert gb.is_regular_at([3]) is True
    assert gb.is_regular_at([-1]) is False
    assert [str(c) for c in gb.vanishing_conditions([-1])] != []
    assert [str(c) for c in gb.vanishing_conditions([3])] == []


# ---------------------------------------------------------------------------
# 2. Correctness cross-check: specialize vs. computing over Q directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_val", [2, 3, -5, 7, Fraction(1, 2), Fraction(-3, 4)])
def test_specialize_agrees_with_direct_computation_at_regular_points(pool, a_val):
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])
    assert gb.is_regular_at([a_val])

    spec = gb.specialize([a_val])
    spec_generators = sorted(str(g) for g in spec.to_exprs())

    a_expr = (
        pool.rational(a_val.numerator, a_val.denominator)
        if isinstance(a_val, Fraction)
        else pool.integer(a_val)
    )
    direct = ak.GroebnerBasis.compute([a_expr * x - y, x + y - one], [x, y])
    direct_generators = sorted(str(g) for g in direct.to_exprs())

    assert spec_generators == direct_generators


def test_degenerate_point_is_refused_and_genuinely_disagrees(pool):
    """a = -1 is flagged *and* is a real disagreement, not a false positive.

    At a = -1 the system {-x - y, x + y - 1} is inconsistent (its basis over
    Q is the unit ideal {1}), which is not the specialisation of the generic
    2-generator basis — so refusing here is the only correct answer, and the
    degeneracy report is not merely conservative at this particular point.
    """
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])
    assert len(gb) == 2, "expected a triangular 2-generator generic basis"

    with pytest.raises(ak.ParamGroebnerError) as excinfo:
        gb.specialize([-1])
    assert excinfo.value.code == "E-PARAMGB-004"

    neg_one = pool.integer(-1)
    direct = ak.GroebnerBasis.compute([neg_one * x - y, x + y - one], [x, y])
    assert len(direct) == 1
    # The unit ideal: 1 = 0, i.e. inconsistent — definitely not a 2-generator
    # triangular basis, so the generic formula really does break down here.
    assert str(direct.to_exprs()[0]) == "1"


def test_flagged_point_can_still_agree_by_a_removable_coincidence(pool):
    """conditions() is sufficient, not necessary: a = 0 is flagged (the
    algorithm inverts `a` to make `a*x - y` monic) but the specialised system
    is still consistent there, and the direct computation over Q confirms
    the *value* the generic formula predicts in the limit is exactly right —
    the conservative report costs a refusal, not a wrong answer.
    """
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one, zero = pool.integer(1), pool.integer(0)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])
    assert not gb.is_regular_at([0])
    with pytest.raises(ak.ParamGroebnerError):
        gb.specialize([0])

    # Direct computation at a = 0 still succeeds (a different leading
    # monomial for the first generator, but a perfectly good basis) and its
    # solution matches the a -> 0 limit of the generic formula x = 1/(a+1),
    # y = a/(a+1) -> x = 1, y = 0.
    direct = ak.GroebnerBasis.compute([zero * x - y, x + y - one], [x, y])
    values = {str(g).replace(" ", "") for g in direct.to_exprs()}
    assert values == {"y", "(x+-1)"}


# ---------------------------------------------------------------------------
# 3. Reading back as Expr: ParametricGbPoly.to_expr / terms / specialize
# ---------------------------------------------------------------------------


def test_generators_read_back_as_expr_not_write_only(pool):
    """Issue #11 was a write-only bug: a result nothing could read back.
    ParametricGbPoly must support the same read path as GbPoly.
    """
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])

    polys = gb.polynomials()
    assert len(polys) == len(gb) == 2
    for p in polys:
        e = p.to_expr()
        assert isinstance(e, ak.Expr)
        # terms() must also work and be non-empty, with Expr coefficients.
        terms = p.terms()
        assert terms
        for exps, coeff in terms:
            assert isinstance(exps, tuple)
            assert isinstance(coeff, ak.Expr)

    # to_exprs() at the basis level is the one-call form.
    exprs = gb.to_exprs()
    assert len(exprs) == 2
    assert all(isinstance(e, ak.Expr) for e in exprs)

    # Indexing and iteration both produce the same ParametricGbPoly type.
    assert isinstance(gb[0], ake.ParametricGbPoly)
    assert [p.to_expr() for p in gb] == [p.to_expr() for p in polys]


def test_specialize_generators_readable_as_gbpoly_expr(pool):
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])
    spec = gb.specialize([3])
    assert isinstance(spec, ak.GroebnerBasis)
    for g in spec:
        e = g.to_expr()
        assert isinstance(e, ak.Expr)


def test_single_polynomial_specialize_raises_on_pole(pool):
    a, x = pool.symbol("a"), pool.symbol("x")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - one], [x], params=[a])
    p = gb.polynomials()[0]
    # x - 1/a: specializing at a = 0 hits the pole in the coefficient.
    with pytest.raises(ak.ParamGroebnerError):
        p.specialize([0])
    spec = p.specialize([2])
    assert isinstance(spec, ak.GbPoly)


# ---------------------------------------------------------------------------
# 4. Errors
# ---------------------------------------------------------------------------


def test_wrong_arity_is_reported(pool):
    a, b, x = pool.symbol("a"), pool.symbol("b"), pool.symbol("x")
    gb = ak.GroebnerBasis.compute([a * x - b], [x], params=[a, b])
    with pytest.raises(ak.ParamGroebnerError) as excinfo:
        gb.specialize([1])
    assert excinfo.value.code == "E-PARAMGB-003"


def test_no_polys_or_vars_is_a_value_error(pool):
    a = pool.symbol("a")
    with pytest.raises(ValueError):
        ak.GroebnerBasis.compute([], [], params=[a])


def test_param_and_var_clash_is_rejected(pool):
    a, x = pool.symbol("a"), pool.symbol("x")
    with pytest.raises(ValueError):
        ak.GroebnerBasis.compute([a * x - x], [x, a], params=[a])


# ---------------------------------------------------------------------------
# 5. Elimination ideal over Q(params)
# ---------------------------------------------------------------------------


def test_eliminate_drops_generators_and_keeps_conditions(pool):
    # Implicitize (t, t^2) with an unused parameter `a` in the field: the
    # elimination ideal is <y - x^2> regardless of a, and has no conditions.
    a, t, x, y = pool.symbol("a"), pool.symbol("t"), pool.symbol("x"), pool.symbol("y")

    gb = ak.GroebnerBasis.compute([x - t, y - t * t], [t, x, y], params=[a])
    el = gb.eliminate([t])
    assert len(el) == 1
    rel = str(el.to_exprs()[0]).replace(" ", "")
    # y - x^2 = 0 (up to sign/ordering of terms).
    assert "y" in rel
    assert "x^2" in rel
    assert el.conditions() == []


def test_eliminate_rejects_a_parameter(pool):
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)
    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])
    with pytest.raises(ValueError):
        gb.eliminate([a])


# ---------------------------------------------------------------------------
# 6. Structural identifiability, end to end: a two-compartment ODE model
# ---------------------------------------------------------------------------
#
# Linear catenary two-compartment model, output = first compartment:
#
#   x1' = -(k01 + k21)*x1 + k12*x2
#   x2' =        k21 *x1 - k12  *x2
#   y   = x1
#
# The classical input-output equation (e.g. Bellman & Åström 1970; DAISY's
# worked examples) for a linear system x' = M x, y = c^T x is the
# characteristic polynomial of M applied to y:
#
#   y'' - tr(M)*y' + det(M)*y = 0
#
# which for this M (tr = -(k01+k21+k12), det = k01*k12) works out to
#
#   y'' + (k01 + k21 + k12)*y' + k01*k12*y = 0.
#
# This is derived here purely by eliminating the two internal states x1, x2
# from the jet equations {y - x1, y' - x1', y'' - x1''} (with x1', x1''
# substituted from the ODE), with the rate constants in the coefficient
# field — exactly the M9 use case the module docs describe.


def _two_compartment_io_equation(pool):
    x1, x2 = pool.symbol("x1"), pool.symbol("x2")
    y, yp, ypp = pool.symbol("y"), pool.symbol("yp"), pool.symbol("ypp")
    k01, k12, k21 = pool.symbol("k01"), pool.symbol("k12"), pool.symbol("k21")

    a_ = -(k01 + k21)  # dx1/dt coefficient of x1
    b_ = k12  # dx1/dt coefficient of x2
    c_ = k21  # dx2/dt coefficient of x1
    d_ = -k12  # dx2/dt coefficient of x2

    eq_y = x1 - y
    eq_yp = (a_ * x1 + b_ * x2) - yp
    eq_ypp = ((a_ * a_ + b_ * c_) * x1 + (a_ * b_ + b_ * d_) * x2) - ypp

    gb = ak.GroebnerBasis.compute(
        [eq_y, eq_yp, eq_ypp],
        [x1, x2, y, yp, ypp],
        params=[k01, k12, k21],
        order="lex",
    )
    el = gb.eliminate([x1, x2])
    return el, (y, yp, ypp, k01, k12, k21)


def test_two_compartment_identifiability_end_to_end(pool):
    el, (y, yp, ypp, k01, k12, k21) = _two_compartment_io_equation(pool)

    assert len(el) == 1, "expected a single input-output relation"
    io_expr = el.to_exprs()[0]
    assert isinstance(io_expr, ak.Expr)

    # Check the relation numerically at several (k01, k12, k21, y, yp, ypp)
    # points that lie on the surface it defines: pick k's and an assumed
    # solution trajectory value (y, yp, ypp) satisfying
    # ypp + (k01+k21+k12)*yp + k01*k12*y = 0, then confirm io_expr evaluates
    # to 0 there, and to something non-zero off the surface.
    for k01v, k12v, k21v, yv, ypv in [
        (2, 3, 5, 7, 11),
        (1, 1, 1, 2, -3),
        (Fraction(1, 2), Fraction(3, 2), 4, -1, 6),
    ]:
        yppv = -(k01v + k21v + k12v) * ypv - k01v * k12v * yv
        env = {k01: k01v, k12: k12v, k21: k21v, y: yv, yp: ypv, ypp: yppv}
        val = _exact_value(pool, io_expr, env)
        assert val == 0, f"IO relation should vanish on the trajectory: got {val}"

        # Off the surface (perturb ypp by 1) it must not vanish.
        env_off = dict(env)
        env_off[ypp] = yppv + 1
        val_off = _exact_value(pool, io_expr, env_off)
        assert val_off != 0


def test_two_compartment_io_equation_matches_trace_and_determinant(pool):
    """The IO relation's coefficients, cleared of denominators, are exactly
    the characteristic polynomial of the state matrix: 1, tr(M), det(M).
    """
    el, (y, yp, ypp, k01, k12, k21) = _two_compartment_io_equation(pool)
    io_expr = el.to_exprs()[0]

    for k01v, k12v, k21v in [(2, 3, 5), (1, 1, 1), (7, 2, 4)]:
        tr = -(k01v + k21v + k12v)  # tr(M)
        det = k01v * k12v  # det(M)

        # y'' - tr*y' + det*y should be proportional to io_expr (same
        # coefficients up to one overall nonzero scalar). Reading off the
        # coefficient of each of y, y', y'' individually (by zeroing the
        # other two) needs only that io_expr is linear in (y, yp, ypp),
        # which it is by construction.
        c_y = _exact_value(pool, io_expr, {y: 1, yp: 0, ypp: 0, k01: k01v, k12: k12v, k21: k21v})
        c_yp = _exact_value(pool, io_expr, {y: 0, yp: 1, ypp: 0, k01: k01v, k12: k12v, k21: k21v})
        c_ypp = _exact_value(pool, io_expr, {y: 0, yp: 0, ypp: 1, k01: k01v, k12: k12v, k21: k21v})
        assert c_ypp != 0
        # (c_y, c_yp, c_ypp) proportional to (det, -tr, 1).
        assert c_y / c_ypp == Fraction(det)
        assert c_yp / c_ypp == Fraction(-tr)


# ---------------------------------------------------------------------------
# 7. No-parameters path reproduces the plain-Q basis exactly
# ---------------------------------------------------------------------------


def test_empty_params_list_is_the_ordinary_engine(pool):
    x = pool.symbol("x")
    one = pool.integer(1)

    gb_param = ak.GroebnerBasis.compute([x * x - one], [x], params=[])
    gb_plain = ak.GroebnerBasis.compute([x * x - one], [x])
    assert isinstance(gb_param, ak.GroebnerBasis)
    assert [str(g) for g in gb_param.to_exprs()] == [str(g) for g in gb_plain.to_exprs()]


# ---------------------------------------------------------------------------
# 8. Feeding a parametric basis its own output back in (2026-08-19 issue #8)
# ---------------------------------------------------------------------------
#
# The basis lives in Q(params)[vars], so its generators carry `den**-1` factors
# in the parameters by construction.  `contains` / `reduce` used to route those
# through the denominator-free Expr -> GbPoly conversion and refuse with
# "negative exponent -1 in polynomial", which made the trivially-true
# `gb.contains(gb.to_exprs()[i])` unrunnable -- and with it the question a loop
# actually needs: do these two parametric bases generate the same ideal?


def test_contains_accepts_the_basis_own_generators(pool):
    a, x = pool.symbol("a"), pool.symbol("x")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - one], [x], params=[a])

    # The generator really does carry a parameter denominator.
    assert "a^-1" in str(gb.to_exprs()[0])

    for i, e in enumerate(gb.to_exprs()):
        assert gb.contains(e) is True, f"generator {i} not recognised: {e}"
        assert gb.reduce(e).is_zero is True
    # Denominator-free input keeps working.
    assert gb.contains(a * x - one) is True
    # ...as does the ParametricGbPoly read path.
    assert all(gb.contains(p.to_expr()) for p in gb.polynomials())


def test_contains_accepts_own_generators_multivariate(pool):
    a, x, y = pool.symbol("a"), pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - y, x + y - one], [x, y], params=[a])
    for i, e in enumerate(gb.to_exprs()):
        assert gb.contains(e) is True, f"generator {i} not recognised: {e}"


def test_nonparametric_control_still_accepts_its_own_generators(pool):
    x = pool.symbol("x")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([2 * x - one], [x])
    for e in gb.to_exprs():
        assert gb.contains(e) is True


def test_a_denominator_in_a_ring_variable_is_still_refused(pool):
    a, x = pool.symbol("a"), pool.symbol("x")
    one = pool.integer(1)

    gb = ak.GroebnerBasis.compute([a * x - one], [x], params=[a])
    with pytest.raises(ValueError, match="negative exponent"):
        gb.contains(x**-1)


def test_parametric_input_may_be_rational_in_the_parameters(pool):
    a, x = pool.symbol("a"), pool.symbol("x")
    one = pool.integer(1)

    # x - 1/a is the same ideal as a*x - 1, written with a denominator.
    gb = ak.GroebnerBasis.compute([x - a**-1], [x], params=[a])
    assert gb.contains(a * x - one) is True


def test_equals_ideal_answers_the_two_bases_question(pool):
    a, x = pool.symbol("a"), pool.symbol("x")
    one = pool.integer(1)

    g1 = ak.GroebnerBasis.compute([a * x - one], [x], params=[a])
    g2 = ak.GroebnerBasis.compute([a * a * x - a], [x], params=[a])
    g3 = ak.GroebnerBasis.compute([x - one], [x], params=[a])

    assert g1.equals_ideal(g2) is True
    assert g2.equals_ideal(g1) is True
    assert g1.equals_ideal(g3) is False
    assert g1.contains_ideal(g1) is True

    # One-directional containment is reported as such: <x> is not <a*x - 1>,
    # but <a*x - 1, x> == <1> contains both.
    big = ak.GroebnerBasis.compute([a * x - one, x], [x], params=[a])
    assert big.contains_ideal(g1) is True
    assert g1.contains_ideal(big) is False


def test_equals_ideal_refuses_to_compare_across_rings(pool):
    a, b, x = pool.symbol("a"), pool.symbol("b"), pool.symbol("x")
    one = pool.integer(1)

    g1 = ak.GroebnerBasis.compute([a * x - one], [x], params=[a])
    g2 = ak.GroebnerBasis.compute([b * x - one], [x], params=[a, b])
    assert g1.equals_ideal(g2) is False


# ---------------------------------------------------------------------------
# 9. specialize(values, verify=True) (2026-08-19 issue #14)
# ---------------------------------------------------------------------------
#
# `conditions()` is sufficient but not necessary, and on small-integer grids a
# quarter to a half of the refusals are unnecessary -- dominated by parameters
# equal to exactly 0, which are intermediate leading-coefficient inversions
# that cancelled.  `verify=True` recomputes and compares instead of refusing on
# the recorded conditions alone.  The refusals were separately verified *sound*,
# so this is a completeness fix; the sound path must not weaken.


@pytest.fixture
def cramer(pool):
    """a*x + b*y = 1, c*x + d*y = 1 over Q(a, b, c, d)."""
    a, b, c, d = (pool.symbol(s) for s in "abcd")
    x, y = pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)
    gb = ak.GroebnerBasis.compute(
        [a * x + b * y - one, c * x + d * y - one], [x, y], params=[a, b, c, d]
    )
    return gb


def test_verify_accepts_a_point_the_recorded_conditions_refuse(cramer):
    # a = 0 is listed in conditions() although the only denominator in the
    # returned basis is a*d - b*c, which is -1 here.
    assert cramer.is_regular_at([0, 1, 1, 1]) is False
    assert "a" in [str(c) for c in cramer.conditions()]

    with pytest.raises(ak.ParamGroebnerError) as exc:
        cramer.specialize([0, 1, 1, 1])
    assert exc.value.code == "E-PARAMGB-004"

    verified = cramer.specialize([0, 1, 1, 1], verify=True)
    assert isinstance(verified, ak.GroebnerBasis)
    # b*y = 1 and x + y = 1  =>  y = 1, x = 0.
    assert sorted(str(e).replace(" ", "") for e in verified.to_exprs()) == [
        "(y+-1)",
        "x",
    ]


def test_verify_still_refuses_a_genuinely_singular_point(cramer):
    # a*d - b*c = 0: the basis really does have a pole, and the specialised
    # system is a different (dependent) system.
    for pt in ([1, 1, 1, 1], [1, 1, 2, 2]):
        with pytest.raises(ak.ParamGroebnerError) as exc:
            cramer.specialize(pt, verify=True)
        assert exc.value.code == "E-PARAMGB-004"


def test_verify_agrees_with_the_direct_basis_where_it_accepts(pool, cramer):
    _a, _b, _c, _d = (pool.symbol(s) for s in "abcd")
    x, y = pool.symbol("x"), pool.symbol("y")
    one = pool.integer(1)

    pt = [0, 1, 1, 1]
    verified = cramer.specialize(pt, verify=True)
    direct = ak.GroebnerBasis.compute(
        [pool.integer(0) * x + one * y - one, one * x + one * y - one], [x, y]
    )
    assert sorted(str(e).replace(" ", "") for e in verified.to_exprs()) == sorted(
        str(e).replace(" ", "") for e in direct.to_exprs()
    )


def test_verify_defaults_off_and_leaves_the_sound_path_alone(pool, cramer):
    # Regular points are unaffected either way.
    for verify in (False, True):
        gb = cramer.specialize([1, 2, 3, 4], verify=verify)
        assert isinstance(gb, ak.GroebnerBasis)
    with pytest.raises(ak.ParamGroebnerError):
        cramer.specialize([1, 1, 1, 1])


def test_verify_recovers_a_measurable_share_of_a_small_integer_grid(cramer):
    """A quarter to a half of refusals on small-integer grids are unnecessary.

    Kept to the 4-parameter `cramer` system on {-1, 0, 1} so it stays a
    CI-tier test; the full 8-system {-2..2} sweep gives 646 refused / 334
    recovered / 239 genuine poles.
    """
    import itertools

    refused = recovered = 0
    for pt in itertools.product([-1, 0, 1], repeat=4):
        pt = list(pt)
        if cramer.is_regular_at(pt):
            continue
        refused += 1
        try:
            cramer.specialize(pt, verify=True)
            recovered += 1
        except ak.ParamGroebnerError:
            pass
    assert refused > 0
    # Every recovered point is a real recovery; the interesting claim is that
    # the share is substantial rather than a rounding error.
    assert 0.25 <= recovered / refused <= 0.5, (refused, recovered)


# ---------------------------------------------------------------------------
# 10. rosenfeld_groebner(dae, params=[...]) (2026-08-19 issues #16 and #13)
# ---------------------------------------------------------------------------


def _decay_dae(pool):
    """x' = -a*x with output y = x, as a DAE.  IO relation: y' + a*y = 0."""
    t, x, y = pool.symbol("t"), pool.symbol("x"), pool.symbol("y")
    dx, dy = pool.symbol("dx/dt"), pool.symbol("dy/dt")
    a = pool.symbol("a")
    dae = ak.DAE.new([dx + a * x, y - x], [x, y], [dx, dy], t)
    return dae, x, y, a


def test_rosenfeld_groebner_accepts_params(pool):
    """Issue #16: this raised TypeError -- M9 did not compose with V2-13."""
    dae, _x, _y, a = _decay_dae(pool)

    r = ak.rosenfeld_groebner(dae, params=[a], max_prolong_rounds=1, order="lex")

    assert isinstance(r.final_basis(), ake.ParametricGroebnerBasis)
    assert r.consistent is True
    # `a` is a coefficient, not a ring variable.
    assert "a" not in [str(v) for v in r.variables()]
    assert [str(p) for p in r.parameters()] == ["a"]
    assert r.final_basis().n_params == 1


def test_rosenfeld_groebner_parametric_gives_the_io_relation(pool):
    dae, x, y, a = _decay_dae(pool)

    r = ak.rosenfeld_groebner(dae, params=[a], eliminate=[x], minimal=True, order="lex")
    basis = r.final_basis()
    state_jets = [v for v in r.variables() if str(v).lstrip("d").split("/")[0] == "x"]
    io = basis.eliminate(state_jets)

    assert len(io) == 1
    # y + y'/a = 0, i.e. y' = -a*y.
    dy = pool.symbol("dy/dt")
    assert io.contains(a * y + dy) is True


def test_rosenfeld_groebner_minimal_stops_at_the_first_informative_round(pool):
    dae, x, _y, a = _decay_dae(pool)

    minimal = ak.rosenfeld_groebner(
        dae,
        params=[a],
        eliminate=[x],
        minimal=True,
        max_prolong_rounds=3,
        order="lex",
    )
    assert minimal.minimal_prolongation_rounds == 1
    assert minimal.prolongation_rounds == 1
    assert minimal.truncated is True

    with pytest.warns(UserWarning, match="already non-empty after 1"):
        over = ak.rosenfeld_groebner(
            dae, params=[a], eliminate=[x], max_prolong_rounds=3, order="lex"
        )
    assert over.prolongation_rounds == 3
    assert over.minimal_prolongation_rounds == 1

    # Over-supplying is *correct*, just more expensive -- that is exactly why
    # nothing else signals it.
    state_jets = [v for v in over.variables() if str(v).lstrip("d").split("/")[0] == "x"]
    assert len(over.final_basis().eliminate(state_jets)) > len(
        minimal.final_basis().eliminate(
            [v for v in minimal.variables() if str(v).lstrip("d").split("/")[0] == "x"]
        )
    )


def test_rosenfeld_groebner_sir_minimal_matches_the_hand_relation(pool):
    """SIR at the minimal informative jet order (2026-08-19 issue #13).

    S' = -b*S*I, I' = b*S*I - g*I, R' = g*I, y0 = I.  By hand,
    I*I'' - I'^2 + b*I^2*I' + b*g*I^3 = 0.  The state decouples -- R never
    enters the relation -- so "as many derivatives as there are states" is one
    too many, and one too many is four orders of magnitude here.
    """
    t = pool.symbol("t")
    S, infected, R, y0 = (pool.symbol(s) for s in ("S", "I", "R", "y0"))
    dS, dI, dR, dy0 = (pool.symbol(s) for s in ("dS/dt", "dI/dt", "dR/dt", "dy0/dt"))
    b, g = pool.symbol("b"), pool.symbol("g")
    dae = ak.DAE.new(
        [
            dS + b * S * infected,
            dI - b * S * infected + g * infected,
            dR - g * infected,
            y0 - infected,
        ],
        [S, infected, R, y0],
        [dS, dI, dR, dy0],
        t,
    )

    r = ak.rosenfeld_groebner(
        dae,
        params=[b, g],
        eliminate=[S, infected, R],
        minimal=True,
        max_prolong_rounds=4,
        order="lex",
    )
    # Two derivatives of the output, not three.
    assert r.minimal_prolongation_rounds == 2

    state_jets = [v for v in r.variables() if str(v).lstrip("d").split("/")[0] in ("S", "I", "R")]
    io = r.final_basis().eliminate(state_jets)
    assert len(io) == 1

    y1, y2 = pool.symbol("dy0/dt"), pool.symbol("ddy0/dt/dt")
    hand = y0 * y2 - y1 * y1 + b * y0 * y0 * y1 + b * g * y0 * y0 * y0
    assert io.contains(hand) is True
    # ...and the relation is the *whole* content: 4 terms, not 233.
    assert io.polynomials()[0].n_terms == 4


def test_eliminate_and_minimal_require_params(pool):
    dae, x, _y, _a = _decay_dae(pool)

    with pytest.raises(ValueError, match="params"):
        ak.rosenfeld_groebner(dae, eliminate=[x])
    with pytest.raises(ValueError, match="params"):
        ak.rosenfeld_groebner(dae, minimal=True)


def test_minimal_requires_eliminate(pool):
    dae, _x, _y, a = _decay_dae(pool)

    with pytest.raises(ValueError, match="eliminate"):
        ak.rosenfeld_groebner(dae, params=[a], minimal=True)


def test_rosenfeld_groebner_without_params_is_unchanged(pool):
    t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")
    dae = ak.DAE.new([dx - x], [x], [dx], t)

    r = ak.rosenfeld_groebner(dae, max_prolong_rounds=1)
    assert isinstance(r.final_basis(), ak.GroebnerBasis)
    assert r.consistent is True
