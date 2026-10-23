"""``parse(str(e)) == e`` for every named constant alkahest prints (report N10).

``str`` printed the imaginary unit as ``I`` but ``parse("I")`` made a free *real*
symbol, so ``is_zero(parse(str(I**2 + 1)))`` was False while ``is_zero(I**2+1)``
was True.  ``E`` and ``GoldenRatio`` (SymPy's spellings) also came back as free
symbols, and the printed ``+oo`` (``∞``) did not lex at all.

Convention (documented in ``parse``): the reserved names are ``pi``,
``EulerGamma``, ``Catalan``, ``Glaisher``, ``GoldenRatio`` (real constant
atoms), ``I`` (imaginary unit), ``E`` (``exp(1)``) and ``oo``/``∞`` (+∞).  A
caller who wants a *variable* with one of those names passes it in ``symbols``.
"""

import alkahest as ak
import pytest
from alkahest import ExprPool, parse
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

CONSTANT_NAMES = ["pi", "EulerGamma", "Catalan", "Glaisher", "GoldenRatio", "I", "E", "oo"]


@pytest.fixture
def pool():
    return ExprPool()


@pytest.mark.parametrize("name", CONSTANT_NAMES)
def test_every_constant_round_trips(pool, name):
    c = parse(name, pool)
    assert parse(str(c), pool) == c
    # A second, independent parse (fresh symbol map) agrees.
    assert parse(name, pool) == c


def test_imaginary_unit_is_the_pool_atom(pool):
    assert parse("I", pool) == pool.imaginary_unit()
    e = pool.imaginary_unit() ** 2 + 1
    assert ak.is_zero(e)
    assert ak.is_zero(parse(str(e), pool))


def test_e_is_exp_one(pool):
    assert parse("E", pool) == ak.exp(pool.integer(1))
    assert ak.is_zero(ak.log(parse("E", pool)) - 1)


def test_goldenratio_is_a_real_constant(pool):
    phi = parse("GoldenRatio", pool)
    assert ak.eval_expr(phi, {}) == pytest.approx((1 + 5**0.5) / 2, rel=1e-15)


def test_infinity_prints_and_parses(pool):
    inf = pool.pos_infinity()
    assert parse(str(inf), pool) == inf
    assert parse("oo", pool) == inf


@pytest.mark.parametrize("name", ["I", "E", "oo"])
def test_a_bound_symbol_overrides_the_reserved_name(pool, name):
    v = pool.symbol("v_" + name)
    assert parse(f"{name} + 1", pool, {name: v}) == v + 1


# ---------------------------------------------------------------------------
# Randomised round trip over constants, functions and operators
# ---------------------------------------------------------------------------

UNARY = [
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "sinh",
    "cosh",
    "tanh",
    "atan",
    "asin",
    "acos",
    "erf",
    "gamma",
    "abs",
    "digamma",
]


def _exprs(pool):
    leaves = st.one_of(
        st.integers(-9, 9).map(pool.integer),
        st.builds(pool.rational, st.integers(-9, 9), st.integers(2, 9)),
        st.sampled_from(["x", "y", "z"]).map(pool.symbol),
        st.sampled_from(CONSTANT_NAMES).map(lambda n: parse(n, pool)),
        st.floats(0.25, 100.0, allow_nan=False, allow_infinity=False).map(
            lambda f: pool.float(round(f, 3))
        ),
    )

    def extend(children):
        un = st.builds(lambda n, a: getattr(ak, n)(a), st.sampled_from(UNARY), children)
        add = st.lists(children, min_size=2, max_size=3).map(lambda xs: _fold(xs, "+"))
        mul = st.lists(children, min_size=2, max_size=3).map(lambda xs: _fold(xs, "*"))
        pw = st.builds(lambda a, b: a**b, children, children)
        neg = children.map(lambda a: -a)
        return st.one_of(un, add, mul, pw, neg)

    return st.recursive(leaves, extend, max_leaves=8)


def _fold(xs, op):
    acc = xs[0]
    for y in xs[1:]:
        acc = acc + y if op == "+" else acc * y
    return acc


@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(data=st.data())
def test_random_expressions_round_trip(data):
    pool = ExprPool()
    e = data.draw(_exprs(pool))
    text = str(e)
    back = parse(text, pool)
    assert back == e, f"{text!r} re-read as {back}"
    # Idempotent: printing the re-read tree gives the same text.
    assert str(back) == text
