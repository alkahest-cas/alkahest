"""A computed basis must be readable — `GbPoly`/`GroebnerBasis` round trips.

Before this, `GbPoly` exposed only `is_zero`/`n_vars` and `GroebnerBasis` only
its constructors plus `reduce`/`contains`, so nothing that returned a basis —
`rosenfeld_groebner`, `triangularize`, `primary_decomposition`, parametric
`solve` — could be inspected at all.  `expr_to_gbpoly` was named in
`compute_raw`'s own docstring but was not exported, so `reduce()` could not
even be called: a caller had no way to build its argument.
"""

import alkahest
import pytest
from alkahest import DAE, ExprPool

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "GroebnerBasis"),
    reason="native module built without groebner feature",
)


@pytest.fixture
def circle_line():
    """`x**2 + y**2 = 1` intersected with `x = y`, and its lex basis."""
    pool = ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    two, one = pool.integer(2), pool.integer(1)
    polys = [x**two + y**two - one, x - y]
    gb = alkahest.GroebnerBasis.compute(polys, [x, y])
    return pool, x, y, gb


# ---------------------------------------------------------------------------
# GroebnerBasis is a readable sequence
# ---------------------------------------------------------------------------


def test_basis_is_a_sequence_of_gbpoly(circle_line):
    _, _, _, gb = circle_line

    assert len(gb) == 2
    assert len(list(gb)) == len(gb)
    assert all(isinstance(g, alkahest.GbPoly) for g in gb)
    assert isinstance(gb[0], alkahest.GbPoly)
    assert gb.polynomials()[0].terms() == gb[0].terms()


def test_basis_indexing_bounds(circle_line):
    _, _, _, gb = circle_line

    assert gb[-1].terms() == gb[len(gb) - 1].terms()
    with pytest.raises(IndexError):
        gb[len(gb)]
    with pytest.raises(IndexError):
        gb[-len(gb) - 1]


def test_basis_reports_its_order_and_variables(circle_line):
    _, x, y, gb = circle_line

    assert gb.order == "lex"
    assert [str(v) for v in gb.variables()] == ["x", "y"]
    assert [str(v) for v in gb[0].variables()] == ["x", "y"]

    grevlex = alkahest.GroebnerBasis.compute([x - y], [x, y], "grevlex")
    assert grevlex.order == "grevlex"


# ---------------------------------------------------------------------------
# GbPoly ↔ Expr round trip
# ---------------------------------------------------------------------------


def test_expr_to_gbpoly_round_trips(circle_line):
    pool, x, y, _ = circle_line
    expr = x ** pool.integer(2) + y ** pool.integer(2) - pool.integer(1)

    p = alkahest.expr_to_gbpoly(expr, [x, y])

    assert p.n_vars == 2
    assert p.n_terms == 3
    # `to_expr` rebuilds a flat sum rather than the caller's exact tree, so the
    # round trip is closed on the canonical side: Expr → GbPoly → Expr → GbPoly.
    assert alkahest.expr_to_gbpoly(p.to_expr(), [x, y]).terms() == p.terms()
    assert str(p.to_expr()) == "(x^2 + y^2 + -1)"


def test_gbpoly_terms_are_exact(circle_line):
    pool, x, y, _ = circle_line
    # x**2 - (3/2)*y
    expr = x ** pool.integer(2) - pool.rational(3, 2) * y

    terms = dict(alkahest.expr_to_gbpoly(expr, [x, y]).terms())

    from fractions import Fraction

    assert terms[(2, 0)] == 1
    assert terms[(0, 1)] == Fraction(-3, 2)


def test_basis_generators_convert_back_to_expr(circle_line):
    _pool, _x, _y, gb = circle_line

    exprs = gb.to_exprs()

    assert len(exprs) == len(gb)
    assert [str(e) for e in exprs] == [str(g.to_expr()) for g in gb]
    # The lex basis eliminates x: one generator is univariate in y, and it is
    # the relation 2y**2 = 1 that the intersection actually satisfies.
    assert "(y^2 + -1/2)" in [str(e) for e in exprs]


def test_round_trip_expr_to_gbpoly_to_expr_is_in_the_ideal(circle_line):
    """Read a generator back out, feed it in again, and it is still in `I`."""
    _pool, x, y, gb = circle_line

    for g in gb:
        recovered = g.to_expr()
        assert gb.contains(recovered)
        assert alkahest.expr_to_gbpoly(recovered, [x, y]).n_terms == g.n_terms


def test_to_expr_accepts_an_explicit_variable_list(circle_line):
    pool, _x, _y, gb = circle_line
    a, b = pool.symbol("a"), pool.symbol("b")

    # Same exponent vectors, different names.
    renamed = gb[0].to_expr([a, b])
    assert "y" not in str(renamed)


def test_to_expr_refuses_a_short_variable_list(circle_line):
    _pool, x, _y, gb = circle_line

    with pytest.raises(ValueError, match=r"only 1 were named|more variables"):
        gb.to_exprs([x])


def test_zero_polynomial_converts_to_zero(circle_line):
    _pool, x, y, gb = circle_line

    # Anything in the ideal reduces to zero.
    remainder = gb.reduce(x - y)
    assert remainder.is_zero
    assert str(remainder.to_expr()) == "0"


# ---------------------------------------------------------------------------
# reduce() / contains() are usable on a basis the caller has in hand
# ---------------------------------------------------------------------------


def test_reduce_accepts_expr_and_gbpoly(circle_line):
    pool, x, y, gb = circle_line
    cube = x ** pool.integer(3)

    from_expr = gb.reduce(cube)
    from_poly = gb.reduce(alkahest.expr_to_gbpoly(cube, [x, y]))

    assert from_expr.terms() == from_poly.terms()
    # x**3 = y*y**2 = y/2 on the intersection.
    assert str(from_expr.to_expr()) == "(y * 1/2)"


def test_compute_raw_keeps_the_variable_context(circle_line):
    pool, x, y, _ = circle_line
    p = alkahest.expr_to_gbpoly(x ** pool.integer(2) - y, [x, y])

    gb = alkahest.GroebnerBasis.compute_raw([p])

    assert [str(e) for e in gb.to_exprs()] == [str(p.to_expr())]
    assert gb.contains(x ** pool.integer(2) - y)


def test_empty_vars_falls_back_to_the_stored_context(circle_line):
    """An empty `vars` list means "use what you know", not "name nothing"."""
    _, _, _, gb = circle_line

    assert [str(e) for e in gb.to_exprs([])] == [str(e) for e in gb.to_exprs()]


# ---------------------------------------------------------------------------
# eliminate() — documented in docs/mdbook/src/solving.md, previously unbound
# ---------------------------------------------------------------------------


def test_eliminate_implicitizes_a_parametric_curve():
    pool = ExprPool()
    t, x, y = pool.symbol("t"), pool.symbol("x"), pool.symbol("y")

    # (t, t**2), with the parameter ordered first so lex eliminates it.
    gb = alkahest.GroebnerBasis.compute([x - t, y - t ** pool.integer(2)], [t, x, y])
    implicit = gb.eliminate([t])

    assert [str(e) for e in implicit.to_exprs()] == ["((y * -1) + x^2)"]
    assert [str(v) for v in implicit.variables()] == ["t", "x", "y"]


def test_eliminate_rejects_an_unknown_variable(circle_line):
    pool, _x, _y, gb = circle_line

    with pytest.raises(ValueError, match="not written over"):
        gb.eliminate([pool.symbol("q")])


# ---------------------------------------------------------------------------
# The three APIs issue #11 called write-only
# ---------------------------------------------------------------------------


def test_rosenfeld_groebner_basis_is_readable():
    pool = ExprPool()
    t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")

    result = alkahest.rosenfeld_groebner(DAE.new([dx - x], [x], [dx], t), max_prolong_rounds=1)
    basis = result.final_basis()

    assert len(basis) > 0
    assert [str(v) for v in result.variables()] == [str(v) for v in basis.variables()]
    exprs = basis.to_exprs()
    assert len(exprs) == len(basis)
    # x' = x prolongs to x = x'' and x' = x''.
    assert [str(e) for e in exprs] == [
        "(x + (-1 * ddx/dt/dt))",
        "(dx/dt + (-1 * ddx/dt/dt))",
    ]


def test_rosenfeld_working_dae_variables_can_be_named():
    """The prolonged system reports extra variables; they now have names."""
    pool = ExprPool()
    t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")

    result = alkahest.rosenfeld_groebner(DAE.new([dx - x], [x], [dx], t), max_prolong_rounds=1)
    working = result.working_dae()

    assert len(working.variables()) == working.n_variables
    assert [str(v) for v in working.derivatives()] == ["dx/dt", "ddx/dt/dt"]


def test_triangularize_chain_is_readable():
    pool = ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    two, one = pool.integer(2), pool.integer(1)

    chains = alkahest.triangularize([x**two + y**two - one, x - y], [x, y])

    assert chains
    for chain in chains:
        assert [str(v) for v in chain.variables()] == ["x", "y"]
        assert [str(e) for e in chain.to_exprs()] == [str(p.to_expr()) for p in chain.polys()]


def test_primary_decomposition_components_are_readable():
    pool = ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    two = pool.integer(2)

    comps = alkahest.primary_decomposition([x**two, y], [x, y])

    assert comps
    for c in comps:
        assert c.primary().to_exprs()
        assert c.associated_prime().to_exprs()


def test_parametric_solve_basis_is_readable():
    """`solve` returns a basis for a positive-dimensional ideal; read it."""
    pool = ExprPool()
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")

    result = alkahest.solve([x ** pool.integer(2) - y * z], [x, y])

    assert isinstance(result, alkahest.GroebnerBasis)
    # Free parameters are appended after the solve variables, and the basis
    # names all of them — otherwise `to_exprs()` could not run at all.
    assert [str(v) for v in result.variables()] == ["x", "y", "z"]
    assert [str(e) for e in result.to_exprs()] == ["(x^2 + (y * z * -1))"]
