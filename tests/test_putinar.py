"""F29 — Putinar (quadratic-module) positivity certificates on semialgebraic sets.

`prove_nonneg(constraints=...)` is Handelman-only and cannot prove `1 − x² ≥ 0`
on the unit disk at any level, although `y² + 1·(1 − x² − y²)` is a one-line
certificate. `prove_nonneg_on` searches `f = σ₀ + Σ σᵢ gᵢ + Σ λⱼ hⱼ`.

Every certificate is checked here against an independent oracle: its summands
(`cert.multipliers`) are converted to SymPy and re-expanded exactly, and the
sum must equal the target identically. Refutation witnesses are checked in
exact rational arithmetic to lie in the set and make the target negative.
"""

import alkahest as ak
import pytest

sympy = pytest.importorskip("sympy")
from alkahest.crosscheck import to_sympy  # noqa: E402


def _reexpand(cert):
    """Σ weight·c·q² over the certificate's summands, expanded by SymPy."""
    total = sympy.Integer(0)
    for weight, squares in cert.multipliers:
        w = to_sympy(weight)
        for c, q in squares:
            assert sympy.nsimplify(to_sympy(c)) > 0
            total += w * to_sympy(c) * to_sympy(q) ** 2
    return sympy.expand(total)


def _assert_certifies(cert, f):
    assert cert.kind == "putinar"
    assert cert.verify()
    assert sympy.expand(_reexpand(cert) - to_sympy(f)) == 0


@pytest.fixture
def xy():
    pool = ak.ExprPool()
    return pool, pool.symbol("x"), pool.symbol("y")


def test_report_repro_one_minus_x_squared_on_the_disk(xy):
    _pool, x, y = xy
    disk = 1 - x**2 - y**2
    f = 1 - x**2
    with pytest.raises(ak.SosError) as e:
        ak.prove_nonneg(f, [x, y], constraints=[disk], level=4)
    assert e.value.code == "E-SOS-002"

    cert = ak.prove_nonneg_on(f, [x, y], [disk], level=1)
    _assert_certifies(cert, f)
    assert "putinar" in repr(cert)


def test_one_minus_quartics_on_the_disk(xy):
    _pool, x, y = xy
    f = 1 - x**4 - y**4
    cert = ak.prove_nonneg_on(f, [x, y], [1 - x**2 - y**2])
    _assert_certifies(cert, f)
    assert cert.degree == 2
    lean = cert.to_lean()
    assert lean is not None
    assert "sorry" not in lean


def test_x_on_unit_interval_from_a_quadratic_constraint():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    cert = ak.prove_nonneg_on(x, [x], [x * (1 - x)])
    _assert_certifies(cert, x)


def test_predicate_constraints_are_accepted(xy):
    pool, x, y = xy
    f = 1 - x**2
    cert = ak.prove_nonneg_on(f, [x, y], [pool.ge(1 - x**2 - y**2, pool.integer(0))])
    _assert_certifies(cert, f)


def test_a_case_that_needs_level_two():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = 1 - x**2
    box = [1 - x, 1 + x]
    with pytest.raises(ak.SosError) as e:
        ak.prove_nonneg_on(f, [x], box, level=1)
    assert e.value.code == "E-SOS-002"
    assert "level=2" in str(e.value)
    cert = ak.prove_nonneg_on(f, [x], box, level=2)
    _assert_certifies(cert, f)


def test_motzkin_on_the_box(xy):
    """The Motzkin polynomial is not a sum of squares, but on [-1, 1]² it is
    in the quadratic module of the box constraints at level 4 (it vanishes at
    the four corners, so the certificate is tight)."""
    _pool, x, y = xy
    m = x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1
    cert = ak.prove_nonneg_on(m, [x, y], [1 - x**2, 1 - y**2], level=4)
    _assert_certifies(cert, m)


def test_shifted_motzkin_on_the_box_goes_through_the_psd_search(xy):
    """``M + 1/100`` at level 3 has no diagonally dominant certificate; the
    numeric block-diagonal Gram search proposes one and exact rounding lands
    it."""
    pool, x, y = xy
    m = x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1 + pool.rational(1, 100)
    cert = ak.prove_nonneg_on(m, [x, y], [1 - x**2, 1 - y**2], level=3)
    _assert_certifies(cert, m)
    assert not any("exact DD stage: certificate found" in line for line in cert.log)


def test_equalities_get_free_multipliers(xy):
    pool, x, y = xy
    f = pool.rational(1, 2) - x * y
    cert = ak.prove_nonneg_on(f, [x, y], equalities=[x**2 + y**2 - 1])
    _assert_certifies(cert, f)


def test_false_claim_is_refuted_with_an_exact_witness(xy):
    pool, x, y = xy
    f = pool.rational(1, 2) - x**2 - y**2
    disk = 1 - x**2 - y**2
    with pytest.raises(ak.SosError) as e:
        ak.prove_nonneg_on(f, [x, y], [disk])
    assert e.value.code == "E-SOS-003"
    witness = e.value.witness
    point = {to_sympy(k): sympy.nsimplify(to_sympy(v)) for k, v in witness.items()}
    assert to_sympy(f).subs(point) < 0
    assert to_sympy(disk).subs(point) >= 0


def test_unbounded_set_refusal_names_the_archimedean_condition(xy):
    _pool, x, y = xy
    with pytest.raises(ak.SosError) as e:
        ak.prove_nonneg_on(x * y, [x, y], [x, y], level=1)
    assert e.value.code == "E-SOS-002"
    assert "Archimedean" in str(e.value)


def test_level_out_of_range_is_invalid_input(xy):
    _pool, x, y = xy
    with pytest.raises(ak.SosError) as e:
        ak.prove_nonneg_on(1 - x**2, [x, y], [1 - x**2 - y**2], level=0)
    assert e.value.code == "E-SOS-004"
