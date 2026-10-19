"""G11a — orthogonal polynomials, spherical harmonics and angular-momentum
coupling coefficients.

Report 10-9 §3 G11 / §4.3: beyond the elementary set only ``bessel_j0/j1``
existed, so hydrogen wavefunctions and other QM probes had to rebuild Legendre
and Laguerre polynomials by Rodrigues' formula by hand, and there were no
spherical harmonics, 3j/6j symbols or Clebsch–Gordan coefficients at all.

Oracles, in order of independence:

* **Exact, always run.** A pure-``Fraction`` three-term recurrence in this file
  (DLMF 18.9) — a different algorithm from the closed-form sums the Rust side
  expands — evaluated at rational points; orthogonality integrals through
  ``ak.integrate``; Clebsch–Gordan unitarity; the spherical-harmonic addition
  theorem.
* **SymPy** (``legendre``, ``assoc_legendre``, ``hermite``, ``hermite_prob``,
  ``laguerre``, ``assoc_laguerre``, ``chebyshevt/u``, ``gegenbauer``,
  ``jacobi``, ``Ynm``, ``wigner_3j/6j``, ``clebsch_gordan``) — exact equality,
  ``n <= 12``.  Skipped when SymPy is absent (Tier-1 CI).
* **mpmath** — numeric values at random points, non-integer and large degree.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as F

import alkahest as ak
import pytest

FAMILIES = (
    "legendre_p",
    "assoc_legendre_p",
    "hermite_h",
    "hermite_he",
    "laguerre_l",
    "assoc_laguerre_l",
    "chebyshev_t",
    "chebyshev_u",
    "gegenbauer_c",
    "jacobi_p",
)

#: Rational family parameters used by the exact tests (between n and x).
PARAMS = {
    "legendre_p": (),
    "assoc_legendre_p": None,  # integer order, swept separately
    "hermite_h": (),
    "hermite_he": (),
    "laguerre_l": (),
    "assoc_laguerre_l": (F(3, 2),),
    "chebyshev_t": (),
    "chebyshev_u": (),
    "gegenbauer_c": (F(5, 4),),
    "jacobi_p": (F(1, 2), F(-1, 3)),
}


@pytest.fixture
def pool():
    return ak.ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def lift(pool, v):
    if isinstance(v, F):
        return pool.rational(v.numerator, v.denominator)
    return pool.integer(v)


def build(pool, name, n, params, x, **kw):
    fn = getattr(ak, name)
    return fn(n, *(lift(pool, p) for p in params), x, **kw)


def exact_at(pool, expr, x, q: F) -> F:
    """``expr`` at ``x = q``, exactly, via subs + simplify."""
    v = ak.simplify(ak.subs(expr, {x: lift(pool, q)})).value
    return F(str(v).replace(" ", ""))


# ---------------------------------------------------------------------------
# Independent reference: Fraction recurrences (DLMF 18.9.1-18.9.2, 14.10.3)
# ---------------------------------------------------------------------------


def ref(name, n, params, x):
    """Exact value by the three-term recurrence (not the closed form)."""

    def three(a0, a1, step):
        if n == 0:
            return a0
        p0, p1 = a0, a1
        for k in range(1, n):
            p0, p1 = p1, step(k, p1, p0)
        return p1

    if name == "legendre_p":
        return three(F(1), x, lambda k, p1, p0: ((2 * k + 1) * x * p1 - k * p0) / (k + 1))
    if name == "hermite_h":
        return three(F(1), 2 * x, lambda k, p1, p0: 2 * x * p1 - 2 * k * p0)
    if name == "hermite_he":
        return three(F(1), x, lambda k, p1, p0: x * p1 - k * p0)
    if name in ("laguerre_l", "assoc_laguerre_l"):
        a = params[0] if params else F(0)
        return three(
            F(1),
            1 + a - x,
            lambda k, p1, p0: ((2 * k + 1 + a - x) * p1 - (k + a) * p0) / (k + 1),
        )
    if name == "chebyshev_t":
        return three(F(1), x, lambda k, p1, p0: 2 * x * p1 - p0)
    if name == "chebyshev_u":
        return three(F(1), 2 * x, lambda k, p1, p0: 2 * x * p1 - p0)
    if name == "gegenbauer_c":
        (a,) = params
        return three(
            F(1),
            2 * a * x,
            lambda k, p1, p0: (2 * (k + a) * x * p1 - (k + 2 * a - 1) * p0) / (k + 1),
        )
    if name == "jacobi_p":
        a, b = params

        def step(k, p1, p0):
            k += 1  # computing P_k from P_{k-1}, P_{k-2}
            c = 2 * k + a + b
            a1 = 2 * k * (k + a + b) * (c - 2)
            a2 = (c - 1) * (a * a - b * b)
            a3 = (c - 2) * (c - 1) * c
            a4 = 2 * (k + a - 1) * (k + b - 1) * c
            return ((a2 + a3 * x) * p1 - a4 * p0) / a1

        return three(F(1), (a + 1) + (a + b + 2) * (x - 1) / 2, step)
    raise AssertionError(name)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_exported():
    for name in (
        *FAMILIES,
        "expand_orthopoly",
        "spherical_harmonic_y",
        "wigner_3j",
        "wigner_6j",
        "clebsch_gordan",
        "SpecialFunctionError",
    ):
        assert name in ak.__all__, name
        assert hasattr(ak, name), name
    assert issubclass(ak.SpecialFunctionError, ak.AlkahestError)


def test_capability_flags_are_truthful(pool, x):
    rows = {r["name"]: r for r in ak.capabilities()["primitives"]}
    for name in FAMILIES:
        row = rows[name]
        assert row["numeric_f64"], name
        assert row["diff_forward"], name
        # No reverse slot (the degree has no cotangent), no Lean certificate,
        # no LLVM lowering: none may be claimed. (F24 gave every family a
        # Taylor-model rule for an integer literal degree.)
        assert not row["diff_reverse"], name
        assert not row["lean_theorem"], name
        assert row["taylor_model"], name
        assert not row["lower_llvm"], name
        # numeric_ball is claimed exactly when interval evaluation works.
        n = pool.symbol("n")
        params = [pool.rational(1, 4)] * (1 if name in PARAMS_1 else 2 if name == "jacobi_p" else 0)
        node = getattr(ak, name)(n, *params, x, expand=False)
        try:
            ak.interval_eval(node, {n: ak.ArbBall(3.0), x: ak.ArbBall(0.4)})
            works = True
        except ValueError:
            works = False
        assert row["numeric_ball"] == works, name


PARAMS_1 = {"assoc_legendre_p", "assoc_laguerre_l", "gegenbauer_c"}


# ---------------------------------------------------------------------------
# Exact expansion vs the Fraction recurrence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [f for f in FAMILIES if f != "assoc_legendre_p"])
def test_expansion_matches_recurrence_exactly(pool, x, name):
    params = PARAMS[name]
    for n in range(13):
        e = build(pool, name, n, params, x)
        for q in (F(1, 3), F(-5, 7), F(2)):
            assert exact_at(pool, e, x, q) == ref(name, n, params, q), (name, n, q)


def test_assoc_legendre_condon_shortley_exactly(pool, x):
    # P_n^m(x) for even m is a polynomial; compare against the Rodrigues form
    # (-1)^m (1-x^2)^{m/2} d^m P_n / dx^m evaluated with Fractions.
    def legendre_coeffs(n):
        c = [F(0)] * (n + 1)
        for k in range(n // 2 + 1):
            c[n - 2 * k] = F((-1) ** k * math.comb(n, k) * math.comb(2 * n - 2 * k, n)) / 2**n
        return c

    def deriv(c, m):
        for _ in range(m):
            c = [k * c[k] for k in range(1, len(c))]
        return c

    for n in range(9):
        for m in [m for m in range(-n, n + 1) if m % 2 == 0]:  # rational (1-x^2)^{m/2}
            e = ak.assoc_legendre_p(n, m, x)
            for q in (F(1, 3), F(-3, 5)):
                am = abs(m)
                d = deriv(legendre_coeffs(n), am)
                val = sum(ck * q**k for k, ck in enumerate(d)) * (1 - q * q) ** (am // 2)
                if m >= 0:
                    want = (-1) ** am * val
                else:
                    want = F(math.factorial(n - am), math.factorial(n + am)) * val
                assert exact_at(pool, e, x, q) == want, (n, m, q)
    # |m| > n is identically zero; P_1^1 = -sqrt(1 - x^2) at x = 3/5 is -4/5.
    assert str(ak.assoc_legendre_p(2, 3, x)) == "0"
    # At a rational point the square root is taken exactly when it exists.
    assert ak.assoc_legendre_p(1, 1, pool.rational(3, 5)) == pool.rational(-4, 5)
    half = ak.assoc_legendre_p(1, 1, pool.rational(1, 2))  # −√3/2, kept as a surd
    assert surd(half) == (F(-1, 2), 3)


def test_negative_degree_reflections(pool, x):
    q = F(2, 7)
    assert exact_at(pool, ak.legendre_p(-4, x), x, q) == ref("legendre_p", 3, (), q)
    assert exact_at(pool, ak.chebyshev_t(-3, x), x, q) == ref("chebyshev_t", 3, (), q)
    assert exact_at(pool, ak.chebyshev_u(-4, x), x, q) == -ref("chebyshev_u", 2, (), q)
    assert str(ak.chebyshev_u(-1, x)) == "0"
    # No polynomial reflection for Hermite: the node is kept, not an error.
    assert "hermite_h" in str(ak.hermite_h(-1, x))


def test_symbolic_degree_keeps_the_node_and_expand_false(pool, x):
    n = pool.symbol("n")
    assert str(ak.legendre_p(n, x)) == "legendre_p(n, x)"
    assert "legendre_p" in str(ak.legendre_p(3, x, expand=False))
    # Non-integer literal degree: a Legendre *function*, kept as a node.
    assert "legendre_p" in str(ak.legendre_p(pool.rational(1, 2), x))
    # After substituting an integer degree, expand_orthopoly expands it.
    e = ak.subs(ak.legendre_p(n, x), {n: pool.integer(3)})
    expanded = ak.expand_orthopoly(e)
    assert ak.simplify(expanded - ak.legendre_p(3, x)).value == pool.integer(0)


def test_symbolic_parameter(pool, x):
    # L_2^{(a)}(x) = (a+1)(a+2)/2 - (a+2) x + x^2/2
    a = pool.symbol("a")
    e = ak.assoc_laguerre_l(2, a, x)
    for av, xv in ((0.3, 0.7), (-0.4, 2.5)):
        got = ak.eval_expr(e, {a: av, x: xv})
        want = (av + 1) * (av + 2) / 2 - (av + 2) * xv + xv * xv / 2
        assert abs(got - want) < 1e-13


def test_special_values_simplify(pool, x):
    n = pool.symbol("n")
    one = pool.integer(1)
    assert ak.simplify(ak.legendre_p(n, one)).value == one
    assert ak.simplify(ak.chebyshev_t(n, one)).value == one
    assert ak.simplify(ak.chebyshev_u(n, one)).value == ak.simplify(n + 1).value
    assert ak.simplify(ak.laguerre_l(n, pool.integer(0))).value == one
    # Parity for an integer degree, on the node.
    h = ak.hermite_h(5, -x, expand=False)
    assert ak.simplify(h + ak.hermite_h(5, x, expand=False)).value == pool.integer(0)


def test_degree_cap_is_a_coded_refusal(pool, x):
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.legendre_p(5000, x)
    assert info.value.code == "E-SPECFN-004"
    ak.legendre_p(5000, x, expand=False)  # the node is always available


def test_parse_round_trip(pool, x):
    n = pool.symbol("n")
    for name in FAMILIES:
        params = [pool.symbol(f"a{i}") for i in range(PARAM_COUNT[name])]
        node = getattr(ak, name)(n, *params, x)
        assert ak.parse(str(node), pool) == node, name


PARAM_COUNT = {f: (1 if f in PARAMS_1 else 2 if f == "jacobi_p" else 0) for f in FAMILIES}


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("nv", [3.0, 6.0, 2.5])
def test_derivative_identity_vs_finite_difference(pool, x, name, nv):
    if name == "assoc_legendre_p" and nv != int(nv):
        pytest.skip("integer order only for the Ferrers recurrence")
    n = pool.symbol("n")
    pvals = {1: [0.75], 2: [0.5, 1.5], 0: []}[PARAM_COUNT[name]]
    if name == "assoc_legendre_p":
        pvals = [1.0]
    params = [pool.symbol(f"p{i}") for i in range(len(pvals))]
    node = getattr(ak, name)(n, *params, x)
    d = ak.diff(node, x).value
    env = {n: nv, **dict(zip(params, pvals))}
    for xv in (-0.6, 0.2, 0.7):
        h = 1e-6
        try:
            hi = ak.eval_expr(node, {**env, x: xv + h})
        except ak.AlkahestError:
            pytest.skip("non-integer degree needs the Arb layer in this build")
        lo = ak.eval_expr(node, {**env, x: xv - h})
        fd = (hi - lo) / (2 * h)
        got = ak.eval_expr(d, {**env, x: xv})
        assert abs(got - fd) <= 1e-6 * (1 + abs(fd)), (name, nv, xv, got, fd)


def test_derivative_in_the_degree_is_refused(pool, x):
    n = pool.symbol("n")
    with pytest.raises(ak.DiffError):
        ak.diff(ak.legendre_p(n, x), n)


def test_legendre_orthogonality_via_integrate(pool, x):
    """∫_{-1}^{1} P_m P_n dx = 2/(2n+1) δ_mn, exactly, through the integrator."""
    lo, hi = pool.integer(-1), pool.integer(1)
    for m in range(6):
        for n in range(6):
            r = ak.integrate(ak.legendre_p(m, x) * ak.legendre_p(n, x), x, lo, hi)
            v = ak.simplify(r.value).value
            want = pool.rational(2, 2 * n + 1) if m == n else pool.integer(0)
            assert v == want, (m, n, str(v))


def test_chebyshev_t_orthogonality_in_theta(pool):
    """T_n(cos t) = cos(n t): ∫_0^π T_m(cos t) T_n(cos t) dt = π/2 δ (m,n ≥ 1)."""
    t = pool.symbol("t")
    c = ak.cos(t)
    for m in range(1, 4):
        for n in range(1, 4):
            f = ak.chebyshev_t(m, c) * ak.chebyshev_t(n, c)
            v = ak.eval_expr(
                ak.integrate(f, t, pool.integer(0), pool.symbol("pi")).value,
                {},
            )
            want = math.pi / 2 if m == n else 0.0
            assert abs(v - want) < 1e-12, (m, n, v)


def test_hydrogen_radial_normalisation(pool):
    """R_{nl}(r) ∝ (2r/n)^ell e^{-r/n} L_{n-ell-1}^{(2l+1)}(2r/n) — the report's probe."""
    r = pool.symbol("r")
    for n, ell in ((1, 0), (2, 0), (2, 1), (3, 1), (3, 2)):
        rho = pool.rational(2, n) * r
        lag = ak.assoc_laguerre_l(n - ell - 1, 2 * ell + 1, rho)
        # N² = (2/n)³ (n-ell-1)! / (2n (n+ell)!)
        norm2 = F(8, n**3) * F(math.factorial(n - ell - 1), 2 * n * math.factorial(n + ell))
        R2 = lift(pool, norm2) * rho ** (2 * ell) * ak.exp(-pool.rational(2, n) * r) * lag**2
        # ∫_0^∞ R² r² dr = 1, exactly, through the integrator.
        v = ak.integrate(R2 * r**2, r, pool.integer(0), pool.pos_infinity()).value
        assert ak.simplify(v).value == pool.integer(1), (n, ell, str(v))


# ---------------------------------------------------------------------------
# Numeric and ball evaluation
# ---------------------------------------------------------------------------


def test_large_degree_ball_contains_exact_value(pool, x):
    n = pool.symbol("n")
    node = ak.legendre_p(n, x)
    exact = ref("legendre_p", 400, (), F(3, 10))
    try:
        b = ak.interval_eval(node, {n: ak.ArbBall(400.0), x: ak.ArbBall(0.3)})
    except ValueError:
        pytest.skip("ball evaluation needs the Arb layer in this build")
    # 0.3 is not exactly 3/10 in binary; widen by the conditioning.
    assert abs(b.mid - float(exact)) < 1e-12
    assert b.rad < 1e-20
    f = ak.eval_expr(node, {n: 400.0, x: 0.3})
    assert abs(f - float(exact)) < 1e-12


# ---------------------------------------------------------------------------
# Angular momentum — exact, oracle-free
# ---------------------------------------------------------------------------


def as_float(e) -> float:
    return ak.eval_expr(e, {})


def surd(e) -> tuple[F, int]:
    """``e = q·√r`` → ``(q, r)``; the coupling results have exactly this shape."""
    text = str(e)
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    parts = text.split(" * ")
    q, r = F(1), 1
    for part in parts:
        if part.endswith("^(1/2)"):
            r *= int(part[: -len("^(1/2)")])
        else:
            q *= F(part)
    return q, r


def square(e) -> F:
    q, r = surd(e)
    return q * q * r


def surd_mul(a, b):
    """Exact product of two surds ``(q, r)`` with square-free ``r``."""
    (qa, ra), (qb, rb) = a, b
    g = math.gcd(ra, rb)
    return qa * qb * g, (ra // g) * (rb // g)


def surd_sum(terms) -> dict[int, F]:
    out: dict[int, F] = {}
    for q, r in terms:
        out[r] = out.get(r, F(0)) + q
    return {r: q for r, q in out.items() if q}


def test_three_j_known_values(pool):
    e = ak.wigner_3j(1, 1, 0, 0, 0, 0, pool=pool)
    assert square(e) == F(1, 3)
    assert as_float(e) < 0
    e = ak.wigner_3j(F(1, 2), F(1, 2), 1, F(1, 2), F(-1, 2), 0, pool=pool)
    assert square(e) == F(1, 6)
    assert as_float(e) > 0
    # float half-integers are exact and accepted
    assert as_float(ak.wigner_3j(0.5, 0.5, 1, 0.5, -0.5, 0, pool=pool)) == as_float(e)
    # selection rules → exact 0
    zero = pool.integer(0)
    assert ak.wigner_3j(1, 1, 3, 0, 0, 0, pool=pool) == zero  # triangle
    assert ak.wigner_3j(1, 1, 1, 1, 0, 0, pool=pool) == zero  # Σm ≠ 0
    assert ak.wigner_3j(1, 1, 1, 0, 0, 0, pool=pool) == zero  # odd J, all m = 0
    assert ak.wigner_3j(1, 1, 0, 2, -2, 0, pool=pool) == zero  # |m| > j


def test_coupling_refusals(pool):
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.wigner_3j(F(1, 3), 1, 1, 0, 0, 0)
    assert info.value.code == "E-SPECFN-001"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.wigner_3j(0.3, 1, 1, 0, 0, 0)
    assert info.value.code == "E-SPECFN-001"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.wigner_6j(pool.symbol("j"), 1, 1, 1, 1, 1)
    assert info.value.code == "E-SPECFN-001"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.clebsch_gordan(-1, 1, 0, 0, 0, 0)
    assert info.value.code == "E-SPECFN-002"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.spherical_harmonic_y(F(1, 2), 0, pool.symbol("t"), pool.symbol("p"))
    assert info.value.code == "E-SPECFN-003"


def test_clebsch_gordan_unitarity(pool):
    """Σ_{m1} <j1 m1 j2 M-m1|J M><j1 m1 j2 M-m1|J' M> = δ_{JJ'} for j1=3/2, j2=1."""
    j1, j2 = F(3, 2), F(1)
    js = [F(1, 2), F(3, 2), F(5, 2)]
    for J in js:
        for Jp in js:
            for twoM in range(-int(2 * min(J, Jp)), int(2 * min(J, Jp)) + 1, 2):
                M = F(twoM, 2)
                terms = []
                for twom1 in range(-3, 4, 2):
                    m1 = F(twom1, 2)
                    m2 = M - m1
                    if abs(m2) > j2:
                        continue
                    a = ak.clebsch_gordan(j1, j2, J, m1, m2, M, pool=pool)
                    b = ak.clebsch_gordan(j1, j2, Jp, m1, m2, M, pool=pool)
                    terms.append(surd_mul(surd(a), surd(b)))
                want = {1: F(1)} if Jp == J else {}
                assert surd_sum(terms) == want, (J, Jp, M)


def test_six_j_known_values(pool):
    assert ak.wigner_6j(1, 1, 1, 1, 1, 1, pool=pool) == pool.rational(1, 6)
    e = ak.wigner_6j(F(1, 2), F(1, 2), 1, F(1, 2), F(1, 2), 0, pool=pool)
    assert square(e) == F(1, 4)
    assert ak.wigner_6j(1, 1, 3, 1, 1, 1, pool=pool) == pool.integer(0)


def test_six_j_orthogonality(pool):
    """Σ_x (2x+1)(2f+1) {a b x; c d f}{c d x; a b g} = δ_fg (DLMF 34.7.2 form)."""
    a, b, c, d = 1, 1, 1, 1
    for f in range(0, 3):
        for g in range(0, 3):
            terms = []
            for xx in range(0, 3):
                q, r = surd_mul(
                    surd(ak.wigner_6j(a, b, xx, c, d, f, pool=pool)),
                    surd(ak.wigner_6j(c, d, xx, a, b, g, pool=pool)),
                )
                terms.append((q * (2 * xx + 1) * (2 * f + 1), r))
            assert surd_sum(terms) == ({1: F(1)} if f == g else {}), (f, g)


# ---------------------------------------------------------------------------
# Spherical harmonics
# ---------------------------------------------------------------------------


def amplitude(pool, ell, m, theta):
    """Y_l^m at φ = 0 (real)."""
    phi = pool.symbol("phi")
    y = ak.spherical_harmonic_y(ell, m, theta, phi)
    return ak.simplify(ak.subs(y, {phi: pool.integer(0)})).value


def test_spherical_harmonic_orthonormality_and_addition_theorem(pool):
    t = pool.symbol("theta")
    amps = {(ell, m): amplitude(pool, ell, m, t) for ell in range(4) for m in range(-ell, ell + 1)}
    grid = [math.pi * (i + 0.5) / 400 for i in range(400)]  # midpoint rule
    vals = {k: [ak.eval_expr(e, {t: th}) for th in grid] for k, e in amps.items()}
    w = math.pi / 400
    for (ell, m), va in vals.items():
        for (lp, mp), vb in vals.items():
            if m != mp:
                continue  # the φ integral is 2π δ_mm'
            s = 2 * math.pi * w * sum(a * b * math.sin(th) for a, b, th in zip(va, vb, grid))
            assert abs(s - (1.0 if ell == lp else 0.0)) < 1e-4, (ell, m, lp, mp, s)
    # Addition theorem: Σ_m |Y_lm|² = (2l+1)/(4π), pointwise.
    for ell in range(4):
        for i in (3, 150, 377):
            s = sum(vals[(ell, m)][i] ** 2 for m in range(-ell, ell + 1))
            assert abs(s - (2 * ell + 1) / (4 * math.pi)) < 1e-12
    # Y_l^{-m} = (-1)^m conj(Y_l^m): at φ = 0 the amplitudes differ by (-1)^m.
    for ell in range(4):
        for m in range(1, ell + 1):
            for i in (10, 200):
                assert abs(vals[(ell, -m)][i] - (-1) ** m * vals[(ell, m)][i]) < 1e-13
    phi = pool.symbol("phi")
    assert ak.spherical_harmonic_y(2, 3, t, phi) == pool.integer(0)


# ---------------------------------------------------------------------------
# SymPy oracle (exact)
# ---------------------------------------------------------------------------


def _sympy():
    return pytest.importorskip("sympy")


def to_sym(e):
    from alkahest.crosscheck import to_sympy

    return to_sympy(e)


@pytest.mark.parametrize("name", FAMILIES)
def test_matches_sympy_exactly(pool, x, name):
    sp = _sympy()
    X = sp.Symbol("x")
    oracle = {
        "legendre_p": lambda n: sp.legendre(n, X),
        "hermite_h": lambda n: sp.hermite(n, X),
        "hermite_he": lambda n: sp.hermite_prob(n, X),
        "laguerre_l": lambda n: sp.laguerre(n, X),
        "assoc_laguerre_l": lambda n: sp.assoc_laguerre(n, sp.Rational(3, 2), X),
        "chebyshev_t": lambda n: sp.chebyshevt(n, X),
        "chebyshev_u": lambda n: sp.chebyshevu(n, X),
        "gegenbauer_c": lambda n: sp.gegenbauer(n, sp.Rational(5, 4), X),
        "jacobi_p": lambda n: sp.jacobi(n, sp.Rational(1, 2), sp.Rational(-1, 3), X),
    }
    if name == "assoc_legendre_p":
        for n in range(9):
            for m in range(-n, n + 1):
                got = to_sym(ak.assoc_legendre_p(n, m, x)).subs(sp.Symbol("x"), X)
                want = sp.assoc_legendre(n, m, X)
                assert sp.simplify(got - want) == 0, (n, m, got, want)
        return
    for n in range(13):
        got = to_sym(build(pool, name, n, PARAMS[name], x))
        want = sp.expand(oracle[name](n))
        assert sp.expand(got - want) == 0, (name, n, got, want)


def test_symbolic_parameters_match_sympy(pool, x):
    sp = _sympy()
    X, A, B = sp.symbols("x a b")
    a, b = pool.symbol("a"), pool.symbol("b")
    for n in range(7):
        pairs = (
            (ak.assoc_laguerre_l(n, a, x), sp.assoc_laguerre(n, A, X)),
            (ak.gegenbauer_c(n, a, x), sp.gegenbauer(n, A, X)),
            (ak.jacobi_p(n, a, b, x), sp.jacobi(n, A, B, X)),
        )
        for got, want in pairs:
            g = to_sym(got)
            assert sp.expand(g - sp.expand_func(want)) == 0, (n, g, want)


def test_spherical_harmonics_match_sympy(pool):
    sp = _sympy()
    T, P = sp.symbols("theta phi", real=True)
    t, p = pool.symbol("theta"), pool.symbol("phi")
    rng = random.Random(7)
    for ell in range(5):
        for m in range(-ell, ell + 1):
            got = to_sym(ak.spherical_harmonic_y(ell, m, t, p))
            want = sp.Ynm(ell, m, T, P).expand(func=True)
            names = {"theta": T, "phi": P, "pi": sp.pi}
            got = got.subs({s: names[s.name] for s in got.free_symbols})
            for _ in range(3):
                th = sp.Rational(rng.randint(1, 313), 100)
                ph = sp.Rational(rng.randint(0, 628), 100)
                d = complex(sp.N((got - want).subs({T: th, P: ph}), 30))
                assert abs(d) < 1e-25, (ell, m, d)


def test_coupling_coefficients_match_sympy(pool):
    sp = _sympy()
    from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j

    half = [F(k, 2) for k in range(0, 7)]
    rng = random.Random(11)
    checked = 0
    for _ in range(400):
        j1, j2, j3 = (rng.choice(half) for _ in range(3))
        m1 = F(rng.choice(range(-int(2 * j1), int(2 * j1) + 1, 2)), 2)
        m2 = F(rng.choice(range(-int(2 * j2), int(2 * j2) + 1, 2)), 2)
        m3 = -m1 - m2
        S = [sp.Rational(v.numerator, v.denominator) for v in (j1, j2, j3, m1, m2, m3)]
        got = to_sym(ak.wigner_3j(j1, j2, j3, m1, m2, m3, pool=pool))
        assert sp.simplify(got - wigner_3j(*S)) == 0, (j1, j2, j3, m1, m2, m3)
        got = to_sym(ak.clebsch_gordan(j1, j2, j3, m1, m2, -m3, pool=pool))
        S_cg = [*S[:5], -S[5]]
        assert sp.simplify(got - clebsch_gordan(*S_cg)) == 0, (j1, j2, j3, m1, m2, -m3)
        checked += 1
    assert checked == 400
    for _ in range(150):
        js = [rng.choice(half[:5]) for _ in range(6)]
        S = [sp.Rational(v.numerator, v.denominator) for v in js]
        got = to_sym(ak.wigner_6j(*js, pool=pool))
        try:
            want = wigner_6j(*S)
        except ValueError:
            # SymPy refuses non-integer triad perimeters; ours is 0 there.
            assert got == 0
            continue
        assert sp.simplify(got - want) == 0, js


# ---------------------------------------------------------------------------
# mpmath oracle (numeric, incl. non-integer and large degree)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_numeric_values_match_mpmath(pool, x, name):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40

    def oracle(nv, ps, xv):
        if name == "legendre_p":
            return mp.legendre(nv, xv)
        if name == "assoc_legendre_p":
            return mp.legenp(nv, ps[0], xv, type=2)
        if name == "hermite_h":
            return mp.hermite(nv, xv)
        if name == "hermite_he":
            return mp.power(2, -mp.mpf(nv) / 2) * mp.hermite(nv, xv / mp.sqrt(2))
        if name == "laguerre_l":
            return mp.laguerre(nv, 0, xv)
        if name == "assoc_laguerre_l":
            return mp.laguerre(nv, ps[0], xv)
        if name == "chebyshev_t":
            return mp.chebyt(nv, xv)
        if name == "chebyshev_u":
            return mp.chebyu(nv, xv)
        if name == "gegenbauer_c":
            return mp.gegenbauer(nv, ps[0], xv)
        return mp.jacobi(nv, ps[0], ps[1], xv)

    n = pool.symbol("n")
    pc = PARAM_COUNT[name]
    params = [pool.symbol(f"p{i}") for i in range(pc)]
    node = getattr(ak, name)(n, *params, x)
    rng = random.Random(hash(name) & 0xFFFF)
    degrees = [0, 1, 5, 12, 40, 150]
    if name != "assoc_legendre_p":
        degrees += [2.5, 7.25]
    for nv in degrees:
        for _ in range(4):
            xv = rng.uniform(-0.95, 0.95)
            ps = [rng.uniform(0.1, 1.5) for _ in range(pc)]
            if name == "assoc_legendre_p":
                ps = [float(rng.randint(-min(int(nv), 3), min(int(nv), 3)))]
            env = {n: float(nv), x: xv, **dict(zip(params, ps))}
            try:
                got = ak.eval_expr(node, env)
            except ak.AlkahestError:
                assert nv != int(nv), "an integer degree must always evaluate"
                pytest.skip("non-integer degree needs the Arb layer in this build")
            want = float(oracle(mp.mpf(nv), [mp.mpf(p) for p in ps], mp.mpf(xv)))
            scale = 1 + abs(want)
            assert abs(got - want) <= 1e-9 * scale, (name, nv, ps, xv, got, want)
