"""Optional PARI/GP interop (``alkahest.pari``).

Tests that need PARI are skipped when no binding (``cypari`` / ``cypari2``)
is installed.  Expected values are published invariants (LMFDB / Cremona /
classical tables), not read off the implementation.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from alkahest import pari

needs_pari = pytest.mark.skipif(not pari.available(), reason="no PARI binding installed")


# ---------------------------------------------------------------------------
# PARI-free behaviour
# ---------------------------------------------------------------------------


def test_unavailable_is_a_coded_refusal(monkeypatch):
    monkeypatch.setattr(pari, "_load", lambda: None)
    assert pari.available() is False
    assert pari.backend() is None
    assert pari.pari_version() is None
    with pytest.raises(pari.PariUnavailable) as ei:
        pari.number_field_invariants([5, 0, 1])
    assert ei.value.code == "E-PARI-001"
    assert isinstance(ei.value, ImportError)
    with pytest.raises(pari.PariUnavailable):
        pari.elliptic_curve_invariants([0, 0, 1, -1, 0])
    with pytest.raises(pari.PariUnavailable):
        pari.modular_forms(1, 12)


def test_import_alkahest_does_not_load_pari():
    import subprocess
    import sys

    code = (
        "import alkahest, sys; assert 'cypari' not in sys.modules and 'cypari2' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ramanujan_tau_oracle():
    # Classical table (Lehmer / OEIS A000594).
    assert [pari.ramanujan_tau(n) for n in range(1, 11)] == [
        1,
        -24,
        252,
        -1472,
        4830,
        -6048,
        -16744,
        84480,
        -113643,
        -115920,
    ]
    # multiplicativity and tau(p^2) recursion, exact
    t = {n: pari.ramanujan_tau(n) for n in range(1, 31)}
    assert t[6] == t[2] * t[3]
    assert t[15] == t[3] * t[5]
    assert t[4] == t[2] ** 2 - 2**11


@pytest.mark.parametrize(
    ("D", "h"), [(-3, 1), (-4, 1), (-20, 2), (-23, 3), (-71, 7), (-163, 1), (-87, 6)]
)
def test_reduced_form_class_numbers(D, h):
    assert pari._imag_quadratic_class_number(D) == h


def test_lattice_index():
    assert pari._lattice_index([[2, 0], [0, 3]], 2) == 6
    # gcd of the 2x2 minors (6, 6, -12)
    assert pari._lattice_index([[2, 1], [0, 3], [4, 5]], 2) == 6
    assert pari._lattice_index([[1, 0]], 2) == 0


# ---------------------------------------------------------------------------
# Number fields
# ---------------------------------------------------------------------------


@needs_pari
@pytest.mark.parametrize(
    ("poly", "disc", "h", "cyc"),
    [
        ([5, 0, 1], -20, 2, [2]),  # Q(sqrt -5)
        ([23, 0, 1], -23, 3, [3]),  # Q(sqrt -23)
        ([71, 0, 1], -71, 7, [7]),
        ([1, 1], 1, 1, []),  # Q itself
    ],
)
def test_class_groups_match_tables(poly, disc, h, cyc):
    d = pari.number_field_invariants(poly, galois=False)
    assert d.discriminant == disc
    assert d.class_number == h
    assert d.class_group == cyc
    assert d.certification == "certified"


@needs_pari
def test_real_quadratic_units_and_integral_basis():
    from alkahest.experimental import NumberField

    K = NumberField([-5, 0, 1])  # Q(sqrt 5): d_K = 5, Z[(1+a)/2]
    d = pari.number_field_invariants(K)
    assert d.discriminant == 5
    assert d.signature == (2, 0)
    assert d.class_number == 1
    assert K.polynomial_discriminant == 20  # != d_K, index 2
    (u,) = d.fundamental_units
    assert abs(u.norm()) == 1
    # fundamental unit of Q(sqrt 5) is the golden ratio (or its inverse/negative)
    phi = K.element([Fraction(1, 2), Fraction(1, 2)])
    assert u in (phi, phi.inverse(), -phi, -phi.inverse())
    assert d.torsion_units[0] == 2
    assert d.verification["unit_norms"]
    assert d.verification["integral_basis"]


@needs_pari
def test_cubic_field_units_primes_galois():
    d = pari.number_field_invariants([-2, 0, 0, 1], primes=[2, 3, 5, 7, 31])  # Q(2^(1/3))
    assert d.discriminant == -108
    assert d.signature == (1, 1)
    assert d.class_number == 1
    assert d.galois["order"] == 6
    assert d.galois["parity"] == "odd"
    assert len(d.fundamental_units) == 1
    assert abs(d.fundamental_units[0].norm()) == 1
    split = {p: sorted((i.e, i.f) for i in v) for p, v in d.primes.items()}
    assert split[2] == [(3, 1)]
    assert split[3] == [(3, 1)]
    assert split[5] == [(1, 1), (1, 2)]  # 5 = 2 mod 3: x^3-2 has one root mod 5
    assert split[7] == [(1, 3)]  # 2 is not a cube mod 7: inert
    assert split[31] == [(1, 1)] * 3  # 2^10 = 1 mod 31: 2 is a cube, splits
    for p, ideals in d.primes.items():
        assert sum(i.e * i.f for i in ideals) == 3
        for i in ideals:
            assert i.norm == p**i.f


@needs_pari
def test_cyclotomic_field_s5_quintic_and_non_monic_refusal():
    d = pari.number_field_invariants([-1, -1, 0, 0, 0, 1])  # x^5 - x - 1
    assert d.discriminant == 2869
    assert d.galois["order"] == 120
    from alkahest.experimental import NumberField

    with pytest.raises(pari.PariError) as ei:
        pari.number_field_invariants(NumberField([1, 0, 2]))
    assert ei.value.code == "E-PARI-002"


@needs_pari
def test_cyclotomic_q_zeta5():
    from alkahest.experimental import NumberField

    d = pari.number_field_invariants(NumberField.cyclotomic(5), primes=[5, 11])
    assert d.discriminant == 125
    assert d.torsion_units[0] == 10
    assert d.galois["order"] == 4
    assert [(i.e, i.f) for i in d.primes[5]] == [(4, 1)]
    assert sorted((i.e, i.f) for i in d.primes[11]) == [(1, 1)] * 4  # 11 = 1 mod 5


@needs_pari
def test_wrong_pari_answer_is_caught(monkeypatch):
    real = pari._gp

    def bad(expr, *args):
        if expr == "(n)->n.disc":
            return real("-24")
        return real(expr, *args)

    monkeypatch.setattr(pari, "_gp", bad)
    with pytest.raises(pari.PariError) as ei:
        pari.number_field_invariants([5, 0, 1])
    assert ei.value.code == "E-PARI-003"


@needs_pari
def test_grh_conditional_when_not_certified():
    d = pari.number_field_invariants([23, 0, 1], certify=False)
    assert d.class_number == 3
    assert d.certification == "grh_conditional"
    assert (
        pari.number_field_invariants([23, 0, 1], class_group=False).certification == "not_computed"
    )


# ---------------------------------------------------------------------------
# Elliptic curves
# ---------------------------------------------------------------------------


@needs_pari
def test_curve_11a1():
    e = pari.elliptic_curve_invariants([0, -1, 1, -10, -20])  # Cremona 11a1
    assert e.conductor == 11
    assert e.torsion["order"] == 5
    assert e.torsion["structure"] == [5]
    assert e.analytic_rank == 0
    assert e.analytic_rank_status == "proven"
    assert e.mw_rank_bounds == (0, 0)
    assert e.mw_rank_status == "proven"
    assert (e.ap[2], e.ap[3], e.ap[5], e.ap[7], e.ap[11]) == (-2, -1, 1, -2, 1)
    # L(E,1) = 0.2538418608559106843...  (LMFDB 11.a2)
    assert e.l_value.rad < 1e-12
    assert abs(e.l_value.mid - 0.25384186085591068) <= e.l_value.rad + 1e-15
    assert e.verification["torsion_generators_pic0"]


@needs_pari
def test_curve_37a1_rank_one():
    e = pari.elliptic_curve_invariants([0, 0, 1, -1, 0])
    assert e.conductor == 37
    assert e.root_number == -1
    assert e.analytic_rank == 1
    assert e.mw_rank_bounds == (1, 1)
    assert e.mw_rank_status == "proven"
    assert e.torsion["order"] == 1
    # L'(E,1) = 0.3059997738340523...
    assert abs(e.l_value.mid - 0.30599977383405230) <= e.l_value.rad + 1e-15
    (pt,) = e.mw_points
    assert pt in [
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1)),
    ]
    assert (e.ap[2], e.ap[3], e.ap[5]) == (-2, -3, -2)


@needs_pari
def test_curve_389a1_rank_two():
    e = pari.elliptic_curve_invariants([0, 1, 1, -2, 0])
    assert e.conductor == 389
    assert e.root_number == 1
    assert e.analytic_rank == 2
    assert e.analytic_rank_status == "numerical"
    assert e.l_value is None
    assert e.mw_rank_bounds == (2, 2)
    assert len(e.mw_points) == 2
    assert e.verification["mw_points_infinite_order_pic0"]


@needs_pari
def test_torsion_compared_with_pic0_for_z6_and_z2xz2():
    # 36a1: y^2 = x^3 + 1, torsion Z/6 (also the tests/test_funcfield.py fixture)
    e = pari.elliptic_curve_invariants([0, 0, 0, 0, 1], rank=False)
    assert e.conductor == 36
    assert e.torsion["structure"] == [6]
    # 32a: y^2 = x^3 - x has torsion Z/2 x Z/2
    e = pari.elliptic_curve_invariants([-1, 0], rank=False)
    assert e.conductor == 32
    assert sorted(e.torsion["structure"]) == [2, 2]


@needs_pari
def test_curve_errors():
    with pytest.raises(pari.PariError) as ei:
        pari.elliptic_curve_invariants([0, 0])  # singular
    assert ei.value.code == "E-PARI-002"
    with pytest.raises(pari.PariError):
        pari.elliptic_curve_invariants([Fraction(1, 2), 1])


@needs_pari
def test_l_value_matches_mpmath_lfun_free_formula():
    """Independent recomputation of L(11a1, 1) from PARI's own a_n via mpmath."""
    import mpmath as mp

    e = pari.elliptic_curve_invariants([0, -1, 1, -10, -20], ap_bound=2)
    c = 2 * mp.pi / mp.sqrt(11)
    an = pari._py(pari._gp("(e,n)->ellan(ellinit(e),n)", pari._gp("[0,-1,1,-10,-20]"), 400))
    val = 2 * mp.fsum(mp.mpf(an[n - 1]) / n * mp.exp(-c * n) for n in range(1, 401))
    assert abs(float(val) - e.l_value.mid) < 1e-12


# ---------------------------------------------------------------------------
# Modular forms
# ---------------------------------------------------------------------------


@needs_pari
def test_delta_is_ramanujan_tau():
    m = pari.modular_forms(1, 12, nmax=10)
    assert m.dimension == 1
    assert m.new_dimension == 1
    assert m.basis[0][1:] == [pari.ramanujan_tau(n) for n in range(1, 11)]
    assert m.basis[0][1:] == [1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920]
    assert m.orbits[0].degree == 1
    assert m.orbits[0].coefficients == m.basis[0][1:]
    assert m.verification["delta_matches_eta_product"]


@needs_pari
def test_s2_gamma0_11_is_elliptic_curve_11a():
    m = pari.modular_forms(11, 2, nmax=12)
    assert m.dimension == 1
    # q prod (1-q^n)^2 (1-q^{11n})^2 = q - 2q^2 - q^3 + 2q^4 + q^5 + 2q^6 - 2q^7 ...
    assert m.orbits[0].coefficients[:7] == [1, -2, -1, 2, 1, 2, -2]
    e = pari.elliptic_curve_invariants([0, -1, 1, -10, -20], ap_bound=12)
    for p, a in e.ap.items():
        assert m.orbits[0].coefficients[p - 1] == a


@needs_pari
def test_s2_gamma0_23_number_field_eigenvalues():
    m = pari.modular_forms(23, 2, nmax=10)
    assert m.dimension == 2
    assert m.new_dimension == 2
    (orb,) = m.orbits
    assert orb.degree == 2
    assert orb.polynomial == [-1, -1, 1]
    K = orb.field
    a = orb.coefficients
    assert a[0].is_one()
    # a_2 has minimal polynomial x^2 + x - 1 (LMFDB 23.2.a.a: Hecke field Q(sqrt 5))
    assert a[1].trace() == -1
    assert a[1].norm() == -1
    # a_4 = a_2^2 - 2 = y^2 - 2 = y - 1 holds exactly in Q[y]/(y^2-y-1)
    assert a[3] == a[1] * a[1] - K.element([2])


@needs_pari
def test_s2_gamma0_37_has_two_rational_orbits():
    m = pari.modular_forms(37, 2, nmax=8)
    assert m.dimension == 2
    assert len(m.orbits) == 2
    assert sorted(o.coefficients[1] for o in m.orbits) == [-2, 0]  # a_2 of 37a and 37b


@needs_pari
def test_modular_forms_bad_input_and_empty_space():
    with pytest.raises(pari.PariError) as ei:
        pari.modular_forms(0, 2)
    assert ei.value.code == "E-PARI-002"
    m = pari.modular_forms(1, 2)
    assert m.dimension == 0
    assert m.orbits == []
    assert m.basis == []
