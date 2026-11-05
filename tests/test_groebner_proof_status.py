"""``GroebnerBasis.proof_status`` and ``solve_proof_status`` follow the path taken.

A multimodular basis of an inhomogeneous ideal is verified over ℚ to generate an
ideal *containing* the input; equality rests on random primes. That has to be
visible, and every route that really is a proof has to say so too.
"""

from __future__ import annotations

from fractions import Fraction
from functools import reduce

import pytest

alk = pytest.importorskip("alkahest")

if not hasattr(alk, "solve_proof_status"):
    pytest.skip("built without the groebner feature", allow_module_level=True)


@pytest.fixture
def ctx():
    pool = alk.ExprPool()
    return pool, pool.symbol("x"), pool.symbol("y")


def test_homogeneous_multimodular_is_proved(ctx):
    _, x, y = ctx
    gb = alk.GroebnerBasis.compute([x**2 - y**2, x * y + y**2], [x, y], order="grevlex")
    assert gb.proof_status == "proved"
    assert len(gb) > 1


def test_inhomogeneous_multimodular_is_probabilistic(ctx):
    pool, x, y = ctx
    one = pool.integer(1)
    polys = [x**2 + y**2 - one, x * y - one]
    gb = alk.GroebnerBasis.compute(polys, [x, y], order="grevlex")
    assert gb.proof_status == "probabilistic"
    # A read-only attribute, not a setting.
    with pytest.raises(AttributeError):
        gb.proof_status = "proved"


def test_lex_through_fglm_inherits_the_grevlex_status(ctx):
    pool, x, y = ctx
    one = pool.integer(1)
    lex = alk.GroebnerBasis.compute([x**2 + y**2 - one, x * y - one], [x, y])
    assert lex.order == "lex"
    assert lex.proof_status == "probabilistic"
    # Elimination keeps generators of the same basis, so the same status.
    assert lex.eliminate([x]).proof_status == "probabilistic"
    # Homogeneous and zero-dimensional: the grevlex basis is proved, and FGLM's
    # own verification is exact, so the lex basis is too.
    hom = alk.GroebnerBasis.compute([x**2, y**2, x * y], [x, y])
    assert hom.proof_status == "proved"


@pytest.mark.parametrize("build", ["grlex", "f5_grevlex", "f5_lex", "pos_dim_lex", "fallback"])
def test_exact_routes_are_proved(ctx, build):
    pool, x, y = ctx
    one = pool.integer(1)
    polys = [x**2 + y**2 - one, x * y - one]
    if build == "grlex":
        gb = alk.GroebnerBasis.compute(polys, [x, y], order="grlex")
    elif build == "f5_grevlex":
        gb = alk.GroebnerBasis.compute_f5(polys, [x, y], order="grevlex")
    elif build == "f5_lex":
        gb = alk.GroebnerBasis.compute_f5(polys, [x, y], order="lex")
    elif build == "pos_dim_lex":
        # The curve x*y = 1 in three variables: FGLM does not apply.
        z = pool.symbol("z")
        gb = alk.GroebnerBasis.compute([x * y - one, x**2 - z], [x, y, z])
    else:
        # Exponents past the modular engine's 16-bit monomials: Buchberger.
        gb = alk.GroebnerBasis.compute([x**70000 + y - one, y**2 - one], [x, y], order="grevlex")
    assert gb.proof_status == "proved"


def test_trivial_inputs_are_proved_under_grevlex(ctx):
    pool, x, y = ctx
    one = pool.integer(1)
    single = alk.GroebnerBasis.compute([x**2 + y - one], [x, y], order="grevlex")
    assert single.proof_status == "proved"
    linear = alk.GroebnerBasis.compute([x + y - one, x - y], [x, y], order="grevlex")
    assert linear.proof_status == "proved"


def test_unit_ideal_is_proved_under_both_orders(ctx):
    """``{1}`` passes the verification over ℚ untested, so it has its own proof."""
    pool, x, y = ctx
    polys = [x**2 + y**2 - pool.integer(1), x**2 + y**2 - pool.integer(2)]
    grevlex = alk.GroebnerBasis.compute(polys, [x, y], order="grevlex")
    assert [g.to_expr() for g in grevlex] == [pool.integer(1)]
    assert grevlex.proof_status == "proved"
    lex = alk.GroebnerBasis.compute(polys, [x, y])
    assert [g.to_expr() for g in lex] == [pool.integer(1)]
    assert lex.proof_status == "proved"


def test_solve_on_an_inconsistent_inhomogeneous_system_is_proved(ctx):
    pool, x, y = ctx
    polys = [x**2 + y**2 - pool.integer(1), x**2 + y**2 - pool.integer(2)]
    assert alk.solve(polys, [x, y]) == []
    assert alk.solve_proof_status() == "proved"


def test_proving_a_hard_inconsistency_is_cheap():
    """Cyclic-5 plus a generic quadric: Buchberger over ℚ takes over three minutes."""
    pool = alk.ExprPool()
    xs = [pool.symbol(f"x{i}") for i in range(5)]
    eqs = []
    for d in range(1, 5):
        s = pool.integer(0)
        for i in range(5):
            t = pool.integer(1)
            for j in range(d):
                t = t * xs[(i + j) % 5]
            s = s + t
        eqs.append(s)
    product = pool.integer(1)
    for v in xs:
        product = product * v
    eqs.append(product - pool.integer(1))
    quadric = pool.integer(-97)
    for i, v in enumerate(xs):
        quadric = quadric + pool.integer(2 * i + 3) * v * xs[(i + 1) % 5]
    eqs.append(quadric)
    with alk.context(budget=alk.Budget(wall_ms=30_000)):
        assert alk.solve(eqs, xs) == []
    assert alk.solve_proof_status() == "proved"


def test_solve_reports_probabilistic_and_the_status_is_per_call(ctx):
    pool, x, y = ctx
    one = pool.integer(1)
    # Consistent: every returned tuple still substitutes back to zero.
    sols = alk.solve([x**2 + y**2 - one, x - y], [x, y])
    assert alk.solve_proof_status() == "probabilistic"
    assert len(sols) == 2
    assert alk.solve_proof_status() == "probabilistic", "reads must not consume the status"
    alk.solve([x + y - one, x - y], [x, y])
    assert alk.solve_proof_status() == "proved"
    alk.solve([x**2 - one], [x])
    assert alk.solve_proof_status() == "proved"


def test_contains_false_is_consistent_with_the_exact_engine(ctx):
    """``contains == False`` is the half of membership a probabilistic basis proves."""
    pool, x, y = ctx
    one = pool.integer(1)
    polys = [x**2 + y**2 - one, x * y - one]
    gb = alk.GroebnerBasis.compute(polys, [x, y], order="grevlex")
    exact = alk.GroebnerBasis.compute_f5(polys, [x, y], order="grevlex")
    assert gb.proof_status == "probabilistic"
    for probe in [x, x - y, x**2 + y**2 - one, x * y - one, x**4 + y**4 - pool.integer(-1)]:
        assert gb.contains(probe) == exact.contains(probe)


def _primes_from(start, count):
    """The first ``count`` primes at or above ``start``, by trial division."""
    found = []
    n = start | 1
    while len(found) < count:
        if all(n % d for d in range(3, int(n**0.5) + 1, 2)):
            found.append(n)
        n += 2
    return found


@pytest.mark.parametrize("n_primes", [1, 2, 8])
def test_unlucky_primes_cannot_fake_an_inconsistency(ctx, n_primes):
    """``⟨y², y² + N·x − 1⟩`` has the zero ``x = 1/N``; modulo ``p | N`` it is ``⟨1⟩``.

    ``N`` is a product of primes in the engine's range, so the input's bad primes
    are real ones it might draw. A correct answer here is the exact ``x = 1/N``.
    """
    pool, x, y = ctx
    primes = _primes_from(2**30, n_primes)
    big_n = reduce(lambda a, b: a * b, primes)
    polys = [y**2, y**2 + pool.integer(big_n) * x - pool.integer(1)]
    sols = alk.solve(polys, [x, y])
    assert len(sols) == 1
    assert Fraction(str(sols[0][x])) == Fraction(1, big_n)
