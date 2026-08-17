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
