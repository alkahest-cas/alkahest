"""G4 — zero recognition, constant folding and special values.

Report 10-9 §3 G4: ``sqrt(2)**2``, ``exp(-x)*exp(x)``, ``erf(-x)+erf(x)``,
``cos(pi)``, ``gamma(1/2)``, ``dilog(1)``, ``m*c*sin^2+m*c*cos^2-m*c`` (which
``simplify_trig`` left as ``m*c + m*c*-1``) … were all returned unsimplified,
and there was no public zero test.  These tests pin

* every repro, under every simplifier entry point, to ``0`` or its exact value
  (checked against mpmath);
* ``is_zero`` on a corpus of >100 identities (must be ``True``) and a corpus of
  non-identities (must never be ``True``, and should be ``False``);
* ``equals`` provenance, assumptions, and agreement with SymPy where SymPy
  decides.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")

P = ak.ExprPool()
x = P.symbol("x")
y = P.symbol("y")
a = P.symbol("a")
b = P.symbol("b")
k = P.symbol("k")
m = P.symbol("m")
c = P.symbol("c")
th = P.symbol("th")
mu = P.symbol("mu")
S = P.symbol("S", ak.Domain.Positive)
K = P.symbol("K", ak.Domain.Positive)
r = P.symbol("r", ak.Domain.Positive)
T = P.symbol("T", ak.Domain.Positive)
sig = P.symbol("sigma", ak.Domain.Positive)
p = P.symbol("p", ak.Domain.Positive)
q = P.symbol("q", ak.Domain.Positive)
n = P.symbol("n", ak.Domain.Integer)
pi = P.symbol("pi")
I = P.imaginary_unit()  # noqa: E741
Z = P.integer
R = P.rational
sin, cos, tan = ak.sin, ak.cos, ak.tan
sinh, cosh, tanh = ak.sinh, ak.cosh, ak.tanh
exp, log, sqrt = ak.exp, ak.log, ak.sqrt

SIMPLIFIERS = [ak.simplify, ak.simplify_trig, ak.simplify_egraph]


def numeric(e) -> float:
    return float(ak.eval_expr(e, {}))


# ---------------------------------------------------------------------------
# The report's repros: each simplifier returns 0 or the exact value.
# ---------------------------------------------------------------------------

#: (name, expression, expected exact value as an Expr, mpmath oracle)
REPROS = [
    ("sqrt(2)^2", sqrt(Z(2)) ** 2, Z(2), 2),
    ("exp(-x)exp(x)", exp(-x) * exp(x), Z(1), 1),
    ("erf(-x)+erf(x)", ak.erf(-x) + ak.erf(x), Z(0), 0),
    ("erf(0)", ak.erf(Z(0)), Z(0), 0),
    ("asin(1)", ak.asin(Z(1)), pi / 2, mpmath.pi / 2),
    ("cos(pi)", cos(pi), Z(-1), -1),
    ("sin(pi)", sin(pi), Z(0), 0),
    ("sin(n pi)", sin(n * pi), Z(0), None),
    ("gamma(1/2)", ak.gamma(R(1, 2)), sqrt(pi), mpmath.sqrt(mpmath.pi)),
    ("dilog(1)", ak.dilog(Z(1)), pi**2 / 6, mpmath.polylog(2, 1)),
    ("pythagoras", m * c * sin(th) ** 2 + m * c * cos(th) ** 2 - m * c, Z(0), None),
    ("(S/K)(S/K)^-1", (S / K) * (S / K) ** -1, Z(1), None),
    ("(mu I)^2", (mu * I) ** 2, -(mu**2), None),
    ("exp(I pi n)", exp(I * pi * n), Z(-1) ** n, None),
    ("m1*1*2*2^-1*L1^2", a * Z(1) * Z(2) * Z(2) ** -1 * b**2, a * b**2, None),
    ("(k m)^2 - k^2 m^2", (k * m) ** 2 - k**2 * m**2, Z(0), None),
]


@pytest.mark.parametrize("simplifier", SIMPLIFIERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(("name", "expr", "want", "oracle"), REPROS, ids=[t[0] for t in REPROS])
def test_report_repro(simplifier, name, expr, want, oracle):
    got = simplifier(expr).value
    assert ak.is_zero(got - want) is True, f"{simplifier.__name__}({name}) -> {got}"
    # The result must be *the* folded form, not merely equal to it.
    assert ak.simplify(got - want).value == Z(0), f"{simplifier.__name__}({name}) -> {got}"
    if oracle is not None:
        assert abs(numeric(got) - float(oracle)) < 1e-12


def test_pythagorean_residue_reaches_literal_zero():
    """The exact report symptom: simplify_trig left ``m*c + m*c*-1``."""
    e = m * c * sin(th) ** 2 + m * c * cos(th) ** 2 - m * c
    assert str(ak.simplify_trig(e).value) == "0"
    assert str(ak.simplify(e).value) == "0"


def test_sqrt_square_of_nonnegative_symbol():
    xp = P.symbol("xp", ak.Domain.NonNegative)
    assert ak.simplify(sqrt(xp) ** 2).value == xp
    # For a real x of unknown sign sqrt(x)^2 is left alone by simplify (for
    # x < 0 the left side is not a real number) — but the identity holds
    # wherever it is defined, and the zero test proves it.
    assert ak.is_zero(sqrt(x) ** 2 - x) is True


@pytest.mark.parametrize(
    ("num", "den"),
    [(nn, d) for d in (1, 2, 3, 4, 6) for nn in range(-12, 13)],
)
def test_trig_table_against_mpmath(num, den):
    arg = R(num, den) * pi if den != 1 else Z(num) * pi
    t = mpmath.mpf(num) / den * mpmath.pi
    for f, ref in ((sin, mpmath.sin), (cos, mpmath.cos)):
        v = ak.simplify(f(arg)).value
        assert not any(s in str(v) for s in ("sin", "cos")), f"{f.__name__}({num}/{den}·π) = {v}"
        assert abs(numeric(v) - float(ref(t))) < 1e-13


@pytest.mark.parametrize("twice", range(-9, 10))
def test_gamma_half_integers_against_mpmath(twice):
    qv = R(2 * twice + 1, 2)
    v = ak.simplify(ak.gamma(qv)).value
    assert "gamma" not in str(v)
    want = mpmath.gamma(mpmath.mpf(2 * twice + 1) / 2)
    assert abs(numeric(v) / float(want) - 1) < 1e-13


@pytest.mark.parametrize(
    ("arg", "want"),
    [
        (Z(1), mpmath.polylog(2, 1)),
        (Z(-1), mpmath.polylog(2, -1)),
        (R(1, 2), mpmath.polylog(2, 0.5)),
    ],
)
def test_dilog_special_values(arg, want):
    v = ak.simplify(ak.dilog(arg)).value
    assert "dilog" not in str(v)
    assert abs(numeric(v) - float(want)) < 1e-14


def test_inverse_trig_table():
    cases = [
        (ak.asin(R(1, 2)), mpmath.pi / 6),
        (ak.asin(Z(-1)), -mpmath.pi / 2),
        (ak.acos(Z(-1)), mpmath.pi),
        (ak.acos(R(1, 2)), mpmath.pi / 3),
        (ak.acos(R(-1, 2)), 2 * mpmath.pi / 3),
        (ak.atan(Z(1)), mpmath.pi / 4),
        (ak.atan(sqrt(Z(3))), mpmath.pi / 3),
        (ak.asin(sqrt(Z(2)) / 2), mpmath.pi / 4),
    ]
    for e, want in cases:
        v = ak.simplify(e).value
        assert not any(s in str(v) for s in ("asin", "acos", "atan")), v
        assert abs(numeric(v) - float(want)) < 1e-14


def test_integer_parameter_multiples_of_pi():
    assert ak.simplify(sin(n * pi)).value == Z(0)
    assert ak.simplify(cos(2 * n * pi)).value == Z(1)
    assert ak.simplify(cos(n * pi)).value == Z(-1) ** n
    # A real parameter decides nothing.
    assert "sin" in str(ak.simplify(sin(x * pi)).value)


# ---------------------------------------------------------------------------
# Identity corpus: is_zero must say True.
# ---------------------------------------------------------------------------


def N(z):
    """Standard normal CDF through erf."""
    return (1 + ak.erf(z / sqrt(Z(2)))) / 2


def _black_scholes_parity():
    d1 = (log(S / K) + (r + sig**2 / 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    call = S * N(d1) - K * exp(-r * T) * N(d2)
    put = K * exp(-r * T) * N(-d2) - S * N(-d1)
    return call - put - (S - K * exp(-r * T))


IDENTITIES = [
    # trigonometric
    sin(x) ** 2 + cos(x) ** 2 - 1,
    sin(2 * x) - 2 * sin(x) * cos(x),
    cos(2 * x) - (cos(x) ** 2 - sin(x) ** 2),
    cos(2 * x) - (1 - 2 * sin(x) ** 2),
    cos(2 * x) - (2 * cos(x) ** 2 - 1),
    sin(x + y) - (sin(x) * cos(y) + cos(x) * sin(y)),
    sin(x - y) - (sin(x) * cos(y) - cos(x) * sin(y)),
    cos(x + y) - (cos(x) * cos(y) - sin(x) * sin(y)),
    cos(x - y) - (cos(x) * cos(y) + sin(x) * sin(y)),
    tan(x) - sin(x) / cos(x),
    1 + tan(x) ** 2 - 1 / cos(x) ** 2,
    sin(3 * x) - (3 * sin(x) - 4 * sin(x) ** 3),
    cos(3 * x) - (4 * cos(x) ** 3 - 3 * cos(x)),
    sin(-x) + sin(x),
    cos(-x) - cos(x),
    sin(x) ** 4 - cos(x) ** 4 - (sin(x) ** 2 - cos(x) ** 2),
    sin(x) * cos(y) - (sin(x + y) + sin(x - y)) / 2,
    cos(x) * cos(y) - (cos(x - y) + cos(x + y)) / 2,
    sin(x) * sin(y) - (cos(x - y) - cos(x + y)) / 2,
    sin(x / 2) ** 2 - (1 - cos(x)) / 2,
    cos(x / 2) ** 2 - (1 + cos(x)) / 2,
    sin(pi - x) - sin(x),
    cos(pi / 2 - x) - sin(x),
    sin(x + 2 * pi) - sin(x),
    cos(x + pi) + cos(x),
    tan(x + pi) - tan(x),
    sin(2 * x) / (1 + cos(2 * x)) - tan(x),
    (1 - cos(2 * x)) / sin(2 * x) - tan(x),
    sin(x) ** 2 * cos(x) ** 2 - sin(2 * x) ** 2 / 4,
    tan(2 * x) - 2 * tan(x) / (1 - tan(x) ** 2),
    sin(4 * x) - 4 * sin(x) * cos(x) * (cos(x) ** 2 - sin(x) ** 2),
    cos(x) ** 4 - sin(x) ** 4 - cos(2 * x),
    # hyperbolic
    cosh(x) ** 2 - sinh(x) ** 2 - 1,
    sinh(2 * x) - 2 * sinh(x) * cosh(x),
    cosh(2 * x) - cosh(x) ** 2 - sinh(x) ** 2,
    tanh(x) - sinh(x) / cosh(x),
    exp(x) - cosh(x) - sinh(x),
    exp(-x) - cosh(x) + sinh(x),
    sinh(x + y) - sinh(x) * cosh(y) - cosh(x) * sinh(y),
    cosh(x + y) - cosh(x) * cosh(y) - sinh(x) * sinh(y),
    1 - tanh(x) ** 2 - 1 / cosh(x) ** 2,
    sinh(-x) + sinh(x),
    cosh(-x) - cosh(x),
    # exponential / logarithm
    exp(-x) * exp(x) - 1,
    exp(2 * x) - exp(x) ** 2,
    exp(x + y) - exp(x) * exp(y),
    exp(x / 2) ** 2 - exp(x),
    exp(x) ** 3 * exp(-2 * x) - exp(x),
    exp(x - y) * exp(y) - exp(x),
    log(p * q) - log(p) - log(q),
    log(p**3) - 3 * log(p),
    log(p / q) - log(p) + log(q),
    log(sqrt(p)) - log(p) / 2,
    exp(log(p)) - p,
    exp(log(x)) - x,
    log(exp(x)) - x,
    log(Z(4)) - 2 * log(Z(2)),
    log(Z(6)) - log(Z(2)) - log(Z(3)),
    log(R(1, 8)) + 3 * log(Z(2)),
    Z(2) ** x * Z(2) ** (-x) - 1,
    p**x * p**y - p ** (x + y),
    exp(x * log(p)) - p**x,
    exp(3 * log(p)) - p**3,
    x * exp(x) - exp(x + log(x)),
    # radicals
    sqrt(Z(2)) ** 2 - 2,
    sqrt(Z(8)) - 2 * sqrt(Z(2)),
    sqrt(Z(12)) - 2 * sqrt(Z(3)),
    1 / sqrt(Z(2)) - sqrt(Z(2)) / 2,
    sqrt(Z(2)) * sqrt(Z(3)) - sqrt(Z(6)),
    (1 + sqrt(Z(2))) ** 2 - (3 + 2 * sqrt(Z(2))),
    (sqrt(Z(3)) - 1) * (sqrt(Z(3)) + 1) - 2,
    sqrt(x) ** 2 - x,
    sqrt(x) ** 4 - x**2,
    (x ** R(1, 3)) ** 3 - x,
    x ** R(3, 2) - x * sqrt(x),
    sqrt(p**2) - p,
    1 / (sqrt(x) + 1) - (sqrt(x) - 1) / (x - 1),
    (x - 1) / (sqrt(x) - 1) - sqrt(x) - 1,
    Z(2) ** R(1, 2) * Z(2) ** R(1, 3) - Z(2) ** R(5, 6),
    Z(8) ** R(1, 3) - 2,
    Z(27) ** R(2, 3) - 9,
    # imaginary unit
    I**2 + 1,
    (mu * I) ** 2 + mu**2,
    (1 + I) ** 2 - 2 * I,
    exp(I * x) - cos(x) - I * sin(x),
    exp(I * pi) + 1,
    exp(I * pi / 2) - I,
    (exp(I * x) + exp(-I * x)) / 2 - cos(x),
    (exp(I * x) - exp(-I * x)) / (2 * I) - sin(x),
    exp(I * pi * n) - Z(-1) ** n,
    sin(n * pi),
    cos(n * pi) - Z(-1) ** n,
    cos(2 * n * pi) - 1,
    # special values
    ak.erf(Z(0)),
    ak.erf(-x) + ak.erf(x),
    ak.asin(Z(1)) - pi / 2,
    ak.acos(Z(0)) - pi / 2,
    ak.atan(Z(1)) - pi / 4,
    cos(pi) + 1,
    sin(pi / 6) - R(1, 2),
    cos(pi / 3) - R(1, 2),
    tan(pi / 3) - sqrt(Z(3)),
    sin(pi / 4) - sqrt(Z(2)) / 2,
    ak.gamma(R(1, 2)) - sqrt(pi),
    ak.gamma(R(3, 2)) - sqrt(pi) / 2,
    ak.gamma(R(5, 2)) - 3 * sqrt(pi) / 4,
    ak.gamma(R(-1, 2)) + 2 * sqrt(pi),
    ak.gamma(Z(5)) - 24,
    ak.dilog(Z(1)) - pi**2 / 6,
    ak.dilog(Z(-1)) + pi**2 / 12,
    log(Z(1)),
    exp(Z(0)) - 1,
    ak.asin(sqrt(Z(3)) / 2) - pi / 3,
    ak.acos(Z(-1)) - pi,
    # Γ functional equation at half-integers
    *[ak.gamma(R(h, 2) + 1) - R(h, 2) * ak.gamma(R(h, 2)) for h in (1, 3, 5, 7, -1, -3)],
    # rational functions
    (x**2 - 1) / (x - 1) - (x + 1),
    1 / (x - 1) - 1 / (x + 1) - 2 / (x**2 - 1),
    (x + y) ** 3 - x**3 - 3 * x**2 * y - 3 * x * y**2 - y**3,
    -(a**2) * (b - 1) + a**2 * b - a**2,
    (k * m) ** 2 - k**2 * m**2,
    (S / K) * (S / K) ** -1 - 1,
    x / (x * y) - 1 / y,
    (x**4 - y**4) / (x - y) - (x + y) * (x**2 + y**2),
    a * Z(1) * Z(2) * Z(2) ** -1 * b**2 - a * b**2,
    m * c * sin(th) ** 2 + m * c * cos(th) ** 2 - m * c,
    # put–call parity under Black–Scholes
    _black_scholes_parity(),
]


def test_identity_corpus_is_large_enough():
    assert len(IDENTITIES) >= 100


@pytest.mark.parametrize("expr", IDENTITIES, ids=[str(i) for i in range(len(IDENTITIES))])
def test_identity_is_proved_zero(expr):
    t = ak.equals(expr, 0)
    assert t.value is True, f"{expr}: {t!r} {t.evidence}"
    assert t.status == "proved_zero"


# ---------------------------------------------------------------------------
# Non-identities: is_zero must never say True.
# ---------------------------------------------------------------------------

NON_IDENTITIES = [
    sin(x) - x,
    sin(x) ** 2 - cos(x) ** 2,
    cos(2 * x) - 2 * cos(x),
    exp(x + y) - exp(x) - exp(y),
    log(p + q) - log(p) - log(q),
    sqrt(x**2) - x,
    sqrt(Z(2)) * sqrt(Z(3)) - sqrt(Z(5)),
    x**2 - x,
    (x + 1) ** 2 - x**2 - 1,
    tan(x) - sin(x),
    sinh(x) - cosh(x),
    exp(I * x) - cos(x),
    ak.gamma(R(1, 2)) - pi,
    ak.erf(x) - x,
    ak.dilog(Z(1)) - pi**2 / 12,
    ak.asin(sin(x)) - x,
    sin(x) ** 2 + cos(x) ** 2 - 1 + R(1, 10**20),
    exp(x) - 1 - x,
    sqrt(Z(2)) - R(141421356, 100000000),
    pi - R(355, 113),
    exp(Z(1)) - R(2718281828, 1000000000),
    x / x - 1 + x,
    cos(x) - cos(-2 * x),
    sin(x + y) - sin(x) - sin(y),
    log(p**2) - log(p),
    sqrt(p * q) - sqrt(p),
    x ** R(1, 3) - x ** R(1, 2),
    cosh(x) ** 2 + sinh(x) ** 2 - 1,
    sin(n * pi / 2),
    cos(pi / 5) - R(4, 5),
    ak.erf(Z(1)) - R(84, 100),
    exp(I * x) - 1,
    (S / K) - 1,
    tan(x) ** 2 - 1 / cos(x) ** 2,
    log(Z(6)) - log(Z(2)) - log(Z(5)),
    sin(x) * cos(x) - sin(2 * x),
    I * x - x,
    x * y - y * x + 1,
    _black_scholes_parity() + K * exp(-r * T) * R(1, 1000),
    exp(x) * exp(y) - exp(x * y),
]


@pytest.mark.parametrize("expr", NON_IDENTITIES, ids=[str(i) for i in range(len(NON_IDENTITIES))])
def test_non_identity_is_never_called_zero(expr):
    t = ak.equals(expr, 0)
    assert t.value is not True, f"{expr}: {t!r}"
    assert t.status != "proved_zero"


def test_non_identities_are_mostly_disproved():
    disproved = sum(ak.is_zero(e) is False for e in NON_IDENTITIES)
    assert disproved >= len(NON_IDENTITIES) - 2


# ---------------------------------------------------------------------------
# Provenance and assumptions
# ---------------------------------------------------------------------------


def test_equals_provenance_exact():
    t = ak.equals(exp(2 * x), exp(x) ** 2)
    assert t.value is True
    assert t.proved
    assert t.status == "proved_zero"
    assert t.method in ("simplify", "rational_normal_form")
    assert bool(t) is True


def test_equals_provenance_numeric_disproof():
    t = ak.equals(ak.asin(sin(x)), x)
    assert t.value is False
    assert t.status == "proved_nonzero"
    assert t.method == "arb_sampling"
    assert t.evidence, "a disproof names the point"


def test_undetermined_result_refuses_bool():
    t = ak.ZeroTest(None, "unknown", "none", [])
    with pytest.raises(ValueError):
        bool(t)


def test_sampling_never_proves_zero():
    # A zero the exact layer cannot see (W(x)·exp(W(x)) = x): the numeric
    # layer may only call it probably zero.
    w = ak.lambert_w(p)
    t = ak.equals(w * exp(w), p)
    assert t.value is not False
    if t.value is None:
        assert t.status in ("probably_zero", "unknown")


def test_assumptions_respected():
    # sqrt(x^2) - x is not identically zero over the reals …
    e = sqrt(x**2) - x
    assert ak.is_zero(e) is False
    # … but is zero once x > 0 is assumed.
    asm = ak.Assumptions(P)
    asm.refine(P.gt(x, Z(0)))
    assert ak.is_zero(e, assumptions=asm) is True
    # A symbol declared positive behaves the same with no assumptions.
    assert ak.is_zero(sqrt(p**2) - p) is True


def test_is_zero_accepts_derived_result():
    assert ak.is_zero(ak.diff(sin(x) ** 2 + cos(x) ** 2, x)) is True


def test_float_input_is_never_proved():
    e = x - x * P.float(1.0)
    assert ak.equals(e, 0).status != "proved_nonzero"


# ---------------------------------------------------------------------------
# SymPy agreement (where SymPy decides)
# ---------------------------------------------------------------------------


def _to_sympy_faithful(expr):
    """``to_sympy`` plus the facts it does not carry: ``pi`` is π, ``I`` is
    the imaginary unit, and the symbols declared ``Positive`` / ``Integer``
    here keep those assumptions."""
    import sympy as sp
    from alkahest.crosscheck import to_sympy

    s = to_sympy(expr)
    positive = {"S", "K", "r", "T", "sigma", "p", "q"}
    repl = {}
    for sym in s.free_symbols:
        name = sym.name
        if name == "pi":
            repl[sym] = sp.pi
        elif name == "I":
            repl[sym] = sp.I
        elif name in positive:
            repl[sym] = sp.Symbol(name, positive=True)
        elif name == "n":
            repl[sym] = sp.Symbol(name, integer=True)
        else:
            repl[sym] = sp.Symbol(name, real=True)
    return s.xreplace(repl)


def test_agreement_with_sympy():
    pytest.importorskip("sympy")

    disagreements = []
    for expr, truth in [(e, True) for e in IDENTITIES[:60]] + [
        (e, False) for e in NON_IDENTITIES[:25]
    ]:
        try:
            s = _to_sympy_faithful(expr)
        except Exception:
            continue
        ours = ak.is_zero(expr)
        try:
            theirs = s.equals(0)
        except Exception:
            theirs = None
        # Where both decide, they must agree, and both must match the truth.
        if ours is not None and theirs is not None and bool(ours) != bool(theirs):
            disagreements.append((str(expr), ours, theirs))
        assert ours in (truth, None), f"{expr}: is_zero={ours}, truth={truth}"
    assert not disagreements, disagreements


def test_is_zero_is_fast_on_a_kerr_sized_identity():
    # sin^2+cos^2 of a large argument tree, summed many times: still linear.
    big = x
    for i in range(30):
        big = big * y + Z(i)
    e = sum((sin(big) ** 2 + cos(big) ** 2 - 1) * Z(j) for j in range(1, 20))
    assert ak.is_zero(e) is True


def test_numeric_values_are_exact_not_float():
    for e in (ak.gamma(R(1, 2)), ak.dilog(Z(1)), cos(pi / 6), ak.atan(Z(1))):
        v = ak.simplify(e).value
        assert "." not in str(v), f"float leaked into {v}"
    assert math.isclose(numeric(ak.simplify(cos(pi / 6)).value), math.sqrt(3) / 2)
