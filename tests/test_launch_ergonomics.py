"""First-hour ergonomics for a user arriving from SymPy (launch-readiness 2026-10-24).

Each block pins one item: Python numbers as operands, ``Matrix`` from Python
numbers and one ``Matrix`` type, the top-level constants and conveniences
(``pi``/``E``/``I``/``oo``, ``factor``, ``expand``, ``binomial``,
``factorial``), a user symbol named ``pi``, ``capabilities()`` agreeing with
``diff_forward``, ``from alkahest.experimental import *``, ``latex(Series)``
and pickling of results.  Values are checked against an independent oracle
(``math`` / ``mpmath`` / SymPy) rather than against alkahest itself.
"""

from __future__ import annotations

import math
import multiprocessing
import pickle
import random
from fractions import Fraction

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")


@pytest.fixture
def pool():
    return ak.ExprPool()


def _num(e, bindings=None):
    return float(ak.eval_expr(e, bindings or {}))


# ---------------------------------------------------------------------------
# 1. Python numbers as operands and as function arguments
# ---------------------------------------------------------------------------


class TestPythonNumberOperands:
    def test_int_base_power(self, pool):
        x = pool.symbol("x")
        e = 2**x
        assert isinstance(e, ak.Expr)
        assert e.node()[0] == "pow"
        base = e.node()[1]
        assert base.node() == ["integer", "2"]  # exact, not 2.0
        for v in (0, 1, 3, 10):
            assert _num(e, {x: v}) == 2.0**v

    def test_alternating_sign(self, pool):
        k = pool.symbol("k")
        e = (-1) ** k
        for v in range(6):
            assert _num(e, {k: v}) == (-1.0) ** v

    def test_fraction_exponent_and_float_base(self, pool):
        x = pool.symbol("x")
        assert (x ** Fraction(1, 3)).node()[2].node() == ["rational", "1", "3"]
        # A float base stays a float: no false exactness.
        assert (0.5**x).node()[1].node()[0] == "float"
        # (`Fraction(1, 2) ** x` is decided by `Fraction.__pow__`, which turns
        # itself into a float before Expr is asked; `pool.rational(1, 2) ** x`
        # is the exact spelling.)
        assert (pool.rational(1, 2) ** x).node()[1].node() == ["rational", "1", "2"]

    def test_int_on_either_side(self, pool):
        x = pool.symbol("x")
        for e, f in [
            (3 * x, lambda v: 3 * v),
            (x * 3, lambda v: 3 * v),
            (1 - x, lambda v: 1 - v),
            (x - 1, lambda v: v - 1),
            (1 / x, lambda v: 1 / v),
            (Fraction(1, 3) + x, lambda v: v + 1 / 3),
        ]:
            assert _num(e, {x: 2.5}) == pytest.approx(f(2.5))

    @pytest.mark.parametrize(
        ("ctor", "ref"),
        [
            (ak.sin, math.sin),
            (ak.cos, math.cos),
            (ak.exp, math.exp),
            (ak.sqrt, math.sqrt),
            (ak.log, math.log),
            (ak.atan, math.atan),
            (ak.gamma, math.gamma),
        ],
    )
    def test_unary_functions_take_ints(self, ctor, ref):
        e = ctor(2)
        assert isinstance(e, ak.Expr)
        assert e.node()[2][0].node() == ["integer", "2"]
        assert _num(e) == pytest.approx(ref(2), rel=1e-12)

    def test_unary_function_fraction_is_exact(self):
        e = ak.sin(Fraction(1, 3))
        assert e.node()[2][0].node() == ["rational", "1", "3"]
        assert _num(e) == pytest.approx(math.sin(1 / 3), rel=1e-12)

    def test_unary_function_float_stays_float(self):
        assert ak.sin(0.5).node()[2][0].node()[0] == "float"

    def test_exp1_is_E(self):
        assert ak.exp(1) == ak.E
        assert _num(ak.E) == pytest.approx(math.e)

    def test_sqrt2_squared(self):
        r = ak.simplify(ak.sqrt(2) ** 2)
        assert _num(r.value) == pytest.approx(2.0)

    def test_binary_functions_take_numbers(self, pool):
        x = pool.symbol("x")
        assert _num(ak.atan2(1, 1)) == pytest.approx(math.pi / 4)
        assert _num(ak.atan2(x, 1), {x: 1.0}) == pytest.approx(math.pi / 4)

    def test_non_numbers_still_refused(self, pool):
        with pytest.raises(TypeError):
            ak.sin("2")
        with pytest.raises(TypeError):
            pool.symbol("x") + "a"

    def test_number_only_calls_use_default_pool(self):
        assert ak.sin(2)._pool is ak.default_pool()
        assert ak.parse("x")._pool is ak.default_pool()
        assert ak.symbol("x")._pool is ak.default_pool()

    def test_number_only_calls_use_context_pool(self, pool):
        with ak.context(pool=pool):
            assert ak.sin(2)._pool is pool
            assert ak.pi._pool is pool
            assert ak.Matrix([[1]]).to_list()[0][0]._pool is pool

    def test_default_pool_constants_mix_into_any_pool(self, pool):
        x = pool.symbol("x")
        for e in (x + ak.pi, ak.pi + x, x * ak.sqrt(2), ak.sin(2) - x, x**ak.E):
            assert e._pool is pool
        assert _num(x + ak.pi, {x: 1.0}) == pytest.approx(1 + math.pi)
        assert _num(ak.sin(x * ak.pi / 2), {x: 1.0}) == pytest.approx(1.0)
        r = ak.subs(x + 1, {x: ak.pi})
        assert _num(getattr(r, "value", r)) == pytest.approx(1 + math.pi)

    def test_module_constants_as_bounds_and_points(self, pool):
        x = pool.symbol("x")
        assert _num(ak.integrate(ak.exp(-x), x, 0, ak.oo).value) == pytest.approx(1.0)
        assert _num(ak.integrate(ak.sin(x), x, 0, ak.pi).value) == pytest.approx(2.0)
        assert _num(ak.limit(1 / x, x, ak.oo).value) == 0.0
        assert _num(ak.limit(ak.atan(x), x, ak.pi).value) == pytest.approx(math.atan(math.pi))
        # Python numbers as points and bounds, too.
        assert _num(ak.limit(ak.sin(x) / x, x, 0).value) == 1.0
        k = pool.symbol("k")
        assert _num(ak.sum_definite(k, k, 1, 10).value) == 55.0
        assert _num(ak.integrate(x, x, 0, Fraction(1, 2)).value) == 0.125

    def test_cross_pool_variables_still_refused(self, pool):
        other = ak.ExprPool()
        with pytest.raises(ak.PoolError):
            pool.symbol("x") + other.symbol("y")
        # A default-pool *variable* is not a value: it is not moved either.
        with pytest.raises(ak.PoolError):
            pool.symbol("x") + ak.symbol("t")
        with pytest.raises(ak.PoolError):
            ak.sin(pool.symbol("x")) + other.symbol("y")


# ---------------------------------------------------------------------------
# 2. Matrix from Python numbers; one Matrix type
# ---------------------------------------------------------------------------


class TestMatrix:
    def test_int_matrix(self):
        m = ak.Matrix([[1, 2], [3, 4]])
        assert _num(m.det()) == -2.0

    def test_fraction_and_mixed_entries(self, pool):
        x = pool.symbol("x")
        m = ak.Matrix([[Fraction(1, 2), 1], [x, 3]])
        assert m.to_list()[0][0].node() == ["rational", "1", "2"]
        assert m.to_list()[1][0]._pool is pool
        assert _num(m.det(), {x: 2.0}) == pytest.approx(1.5 - 2.0)

    def test_from_rows_is_the_same_type(self, pool):
        one = pool.integer(1)
        a = ak.Matrix.from_rows([[one]])
        assert isinstance(a, ak.Matrix)
        assert isinstance(a @ a, ak.Matrix)
        assert type(ak.Matrix([[1]])) is type(a)

    def test_controls_accept_from_rows(self, pool):
        def i(n):
            return pool.integer(n)

        a = ak.Matrix.from_rows([[i(0), i(1)], [i(-2), i(-3)]])
        b = ak.Matrix.from_rows([[i(0)], [i(1)]])
        c = ak.Matrix.from_rows([[i(1), i(0)]])
        d = ak.Matrix.from_rows([[i(0)]])
        ss = ak.StateSpace(a, b, c, d)
        assert ss is not None
        q = ak.Matrix.from_rows([[i(1), i(0)], [i(0), i(1)]])
        x = ak.lyapunov(a, q)
        # Independent check: A X + X Aᵀ + Q = 0 with numpy.
        np = pytest.importorskip("numpy")
        an = np.array([[0, 1], [-2, -3]], dtype=float)
        xn = np.array([[_num(e) for e in row] for row in x.to_list()])
        residual = an @ xn + xn @ an.T + np.eye(2)
        assert np.abs(residual).max() < 1e-9

    def test_controls_accept_int_matrices(self):
        x = ak.lyapunov(ak.Matrix([[0, 1], [-2, -3]]), ak.Matrix([[1, 0], [0, 1]]))
        assert isinstance(x, ak.Matrix)


# ---------------------------------------------------------------------------
# 3. Top-level conveniences and the `pi` symbol trap
# ---------------------------------------------------------------------------


class TestConstants:
    def test_values(self):
        assert _num(ak.pi) == pytest.approx(math.pi)
        assert _num(ak.E) == pytest.approx(math.e)
        assert ak.oo == ak.default_pool().pos_infinity()
        r = ak.evaluate(ak.I, {}, mode="complex")
        assert r.value == pytest.approx(1j)

    def test_constants_match_parse(self):
        for name in ("pi", "E", "I", "oo"):
            assert getattr(ak, name) == ak.parse(name), name

    def test_star_import(self):
        ns: dict = {}
        exec("from alkahest import *", ns)
        for name in ("pi", "E", "I", "oo", "factor", "expand", "binomial", "factorial"):
            assert name in ns, name

    def test_names_in_all(self):
        for name in (
            "pi",
            "E",
            "I",
            "oo",
            "factor",
            "expand",
            "binomial",
            "factorial",
            "default_pool",
        ):
            assert name in ak.__all__
            assert getattr(ak, name).__doc__ or name in ("pi", "E", "I", "oo")

    def test_simplify_sin_pi(self):
        assert ak.simplify(ak.sin(ak.pi)).value == ak.default_pool().integer(0)


class TestUserSymbolNamedPi:
    def test_pool_symbol_pi_is_a_variable(self, pool):
        p = pool.symbol("pi")
        assert p != ak.parse("pi", pool)
        assert p != pool.pi()
        # Used to simplify to 0: the variable was π.
        r = ak.simplify(ak.sin(p)).value
        assert r != pool.integer(0)
        assert str(p) == "pi"
        # It is a free variable: evaluates only when bound.
        assert _num(ak.sin(p), {p: 1.0}) == pytest.approx(math.sin(1.0))

    @pytest.mark.parametrize("name", ["EulerGamma", "Catalan", "Glaisher", "GoldenRatio"])
    def test_other_reserved_names(self, pool, name):
        v = pool.symbol(name)
        assert v != ak.parse(name, pool)
        assert str(v) == name

    def test_parse_pi_is_the_constant(self, pool):
        assert ak.parse("pi", pool) == pool.pi()
        assert _num(ak.parse("pi", pool)) == pytest.approx(math.pi)
        assert ak.simplify(ak.sin(pool.pi())).value == pool.integer(0)

    def test_bound_name_wins_in_parse(self, pool):
        p = pool.symbol("pi")
        assert ak.parse("2*pi", pool, {"pi": p}) == 2 * p


class TestFactor:
    def test_univariate(self, pool):
        x = pool.symbol("x")
        f = ak.factor(x**2 - 1)
        assert isinstance(f, ak.Expr)
        assert f.node()[0] == "mul"
        self._same(f, x**2 - 1, [x])

    def test_multivariate(self, pool):
        x, y = pool.symbol("x"), pool.symbol("y")
        f = ak.factor(x**2 - y**2)
        assert f.node()[0] == "mul"
        assert len(f.node()[1]) == 2
        self._same(f, x**2 - y**2, [x, y])

    def test_content_multiplicity_and_denominator(self, pool):
        x = pool.symbol("x")
        e = (2 * x**3 + 4 * x**2 + 2 * x) / (x**2 - 1)
        f = ak.factor(e)
        self._same(f, e, [x])
        sp = pytest.importorskip("sympy")
        sx = sp.Symbol("x")
        assert (
            sp.simplify(
                sp.sympify(str(f).replace("^", "**"))
                - sp.factor((2 * sx**3 + 4 * sx**2 + 2 * sx) / (sx**2 - 1))
            )
            == 0
        )

    def test_rational_coefficients(self, pool):
        x = pool.symbol("x")
        e = x**2 / 2 - Fraction(1, 2)
        self._same(ak.factor(e), e, [x])

    def test_against_sympy_random(self, pool):
        sp = pytest.importorskip("sympy")
        rng = random.Random(20261025)
        x, y = pool.symbol("x"), pool.symbol("y")
        sx, sy = sp.symbols("x y")
        for _ in range(8):
            a, b, c = (rng.randint(-3, 3) for _ in range(3))
            e = (x + a) ** 2 * (x * y + b) * (y - c)
            se = (sx + a) ** 2 * (sx * sy + b) * (sy - c)
            f = ak.factor(ak.expand(e))
            n_ours = len(f.node()[1]) if f.node()[0] == "mul" else 1
            assert n_ours >= 1
            self._same(f, e, [x, y])
            # Same number of irreducible factors (with multiplicity) as SymPy.
            ours = sp.factor_list(sp.sympify(str(f).replace("^", "**")))
            theirs = sp.factor_list(sp.expand(se))
            assert sorted(m for _, m in ours[1]) == sorted(m for _, m in theirs[1])

    def test_float_refused(self, pool):
        x = pool.symbol("x")
        with pytest.raises(ak.AlkahestError):
            ak.factor(0.5 * x**2 - 0.5)

    @staticmethod
    def _same(a, b, vs):
        rng = random.Random(7)
        for _ in range(5):
            pt = {v: rng.uniform(1.5, 3.0) for v in vs}
            assert _num(a, pt) == pytest.approx(_num(b, pt), rel=1e-10)


class TestExpand:
    def test_binomial_power(self, pool):
        x = pool.symbol("x")
        e = ak.expand((x + 1) ** 13)
        assert e.node()[0] == "add"
        assert len(e.node()[1]) == 14
        assert _num(e, {x: 1.0}) == 2.0**13

    def test_trinomial_power_not_capped(self, pool):
        x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
        e = ak.expand((x + y + z) ** 8)
        assert len(e.node()[1]) == math.comb(10, 2)  # monomials of degree 8 in 3 vars
        assert _num(e, {x: 1.0, y: 1.0, z: 1.0}) == 3.0**8

    def test_collects_like_terms(self, pool):
        x = pool.symbol("x")
        e = ak.expand((x + 1) ** 2 - x**2)
        assert _num(e, {x: 5.0}) == 11.0
        assert len(e.node()[1]) == 2  # 2*x + 1

    def test_denominator_distributed(self, pool):
        x = pool.symbol("x")
        e = ak.expand((x + 1) ** 2 / x)
        assert e.node()[0] == "add"
        assert len(e.node()[1]) == 3  # x + 2 + 1/x
        assert _num(e, {x: 2.0}) == pytest.approx(4.5)

    def test_against_sympy(self, pool):
        sp = pytest.importorskip("sympy")
        x, y = pool.symbol("x"), pool.symbol("y")
        sx, sy = sp.symbols("x y")
        e = (x - 2 * y) ** 3 * (x + y) + ak.sin(x) * (x + 1)
        se = (sx - 2 * sy) ** 3 * (sx + sy) + sp.sin(sx) * (sx + 1)
        ours = sp.sympify(str(ak.expand(e)).replace("^", "**"))
        assert sp.expand(ours - sp.expand(se)) == 0
        assert len(sp.Add.make_args(ours)) == len(sp.Add.make_args(sp.expand(se)))

    def test_zero(self, pool):
        x = pool.symbol("x")
        assert ak.expand((x + 1) ** 2 - x**2 - 2 * x - 1) == pool.integer(0)


class TestBinomialFactorial:
    def test_exact_integers(self):
        assert ak.factorial(5) == ak.default_pool().integer(120)
        assert ak.factorial(0) == ak.default_pool().integer(1)
        assert ak.factorial(30).node() == ["integer", str(math.factorial(30))]
        for n in range(-4, 9):
            for k in range(-1, 9):
                want = math.comb(n, k) if n >= 0 and k >= 0 else None
                got = ak.binomial(n, k)._node_exact()[1]
                if want is not None:
                    assert got == want, (n, k)
                elif k < 0:
                    assert got == 0
                else:
                    # Generalised: C(n, k) = n(n-1)...(n-k+1)/k!
                    assert got == pytest.approx(float(mpmath.binomial(n, k)))

    def test_negative_factorial_refused(self):
        with pytest.raises(ak.DomainError) as ei:
            ak.factorial(-1)
        assert ei.value.code == "E-DOMAIN-001"

    def test_symbolic(self, pool):
        n, k = pool.symbol("n"), pool.symbol("k")
        b = ak.binomial(n, k)
        assert b._pool is pool
        assert _num(b, {n: 7.5, k: 2.25}) == pytest.approx(
            float(mpmath.binomial(7.5, 2.25)), rel=1e-10
        )
        f = ak.factorial(n)
        assert _num(f, {n: 4.5}) == pytest.approx(float(mpmath.factorial(4.5)), rel=1e-10)
        assert ak.binomial(n, 2)._pool is pool


# ---------------------------------------------------------------------------
# 4. capabilities() agrees with diff_forward
# ---------------------------------------------------------------------------


def _cnum(e, bindings):
    """A complex value of *e* (some primitives, e.g. Hankel, are complex on
    the reals), or ``None`` where it does not evaluate to a finite number."""
    try:
        v = _num(e, bindings)
        if math.isfinite(v):
            return complex(v)
    except (ak.AlkahestError, ValueError, ZeroDivisionError, TypeError):
        pass
    try:
        r = ak.evaluate(e, bindings, mode="complex")
    except (ak.AlkahestError, ValueError, ZeroDivisionError):
        return None
    if r.status != "ok":
        return None
    z = complex(r.value)
    return z if math.isfinite(z.real) and math.isfinite(z.imag) else None


def _probe_args(pool, x, arity, slot):
    consts = [Fraction(1, 3), Fraction(2, 5), Fraction(1, 7), Fraction(3, 11)]
    return [
        x if i == slot else pool.rational(c.numerator, c.denominator)
        for i, c in enumerate(consts[:arity])
    ]


def test_capabilities_diff_forward_is_truthful():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    claimed = [r["name"] for r in ak.capabilities()["primitives"] if r["diff_forward"]]
    assert len(claimed) > 30
    checked = []
    for name in claimed:
        ok_slot = False
        for arity in (1, 2, 3, 4):
            for slot in range(arity):
                try:
                    e = pool.func(name, _probe_args(pool, x, arity, slot))
                except ak.AlkahestError:  # wrong arity for this function
                    break
                try:
                    sym = ak.diff(e, x).value
                except ak.AlkahestError:
                    continue
                fwd = ak.diff_forward(e, x).value  # must not refuse where diff works
                agree = 0
                for v in (0.21, 0.37, 0.55, 0.8, 1.3, 2.2):
                    a, b = _cnum(fwd, {x: v}), _cnum(sym, {x: v})
                    if a is None or b is None:
                        continue
                    assert a == pytest.approx(b, rel=1e-8, abs=1e-10), (name, arity, slot, v)
                    agree += 1
                # A distribution-valued derivative (`heaviside` → `diracdelta`)
                # has no point value to compare: then the two must agree
                # symbolically.
                if agree or fwd == sym or ak.simplify(fwd - sym).value == pool.integer(0):
                    ok_slot = True
            if ok_slot:
                break
        assert ok_slot, (
            f"capabilities() claims diff_forward for {name}, but no argument slot differentiates"
        )
        checked.append(name)
    for must in ("tan", "asin", "acos", "sinh", "cosh", "tanh", "asinh", "erf", "gamma"):
        assert must in checked


@pytest.mark.parametrize(
    ("ctor", "ref"),
    [
        (ak.tan, lambda v: 1 / math.cos(v) ** 2),
        (ak.asin, lambda v: 1 / math.sqrt(1 - v * v)),
        (ak.acos, lambda v: -1 / math.sqrt(1 - v * v)),
        (ak.sinh, math.cosh),
        (ak.cosh, math.sinh),
        (ak.tanh, lambda v: 1 / math.cosh(v) ** 2),
        (ak.asinh, lambda v: 1 / math.sqrt(1 + v * v)),
        (ak.erf, lambda v: 2 / math.sqrt(math.pi) * math.exp(-v * v)),
    ],
)
def test_diff_forward_new_rules_against_closed_form(pool, ctor, ref):
    x = pool.symbol("x")
    d = ak.diff_forward(ctor(x**2), x).value  # chain rule through the argument
    for v in (0.1, 0.3, 0.6):
        assert _num(d, {x: v}) == pytest.approx(2 * v * ref(v * v), rel=1e-10)


# ---------------------------------------------------------------------------
# 5. from alkahest.experimental import *
# ---------------------------------------------------------------------------


def test_experimental_star_import():
    ns: dict = {}
    exec("from alkahest.experimental import *", ns)
    import alkahest.experimental as exp

    for name in exp.__all__:
        assert name in ns


# ---------------------------------------------------------------------------
# 6. latex(Series)
# ---------------------------------------------------------------------------


def test_latex_accepts_series_and_derived_result(pool):
    x = pool.symbol("x")
    s = ak.series(ak.exp(x), x, pool.integer(0), 4)
    t = ak.latex(s)
    assert isinstance(t, str)
    assert t == ak.latex(s.expr)
    assert "O" in t
    assert isinstance(ak.unicode_str(s), str)
    r = ak.diff(x**3, x)
    assert ak.latex(r) == ak.latex(r.value)


# ---------------------------------------------------------------------------
# 7. Pickling results
# ---------------------------------------------------------------------------


def test_pickle_derived_result_round_trip(pool):
    x = pool.symbol("x")
    for r in (
        ak.simplify(x + x),
        ak.diff(ak.sin(x) * x**2, x),
        ak.integrate(x**2 * ak.exp(x), x),
        ak.integrate(x**2, x, 0, 1),
    ):
        r2 = pickle.loads(pickle.dumps(r))
        assert isinstance(r2, ak.DerivedResult)
        assert r2.value == r.value  # same process: same pool
        assert r2.steps == r.steps
        assert r2.derivation == r.derivation
        assert r2.verification == r.verification


def test_pickle_matrix_round_trip(pool):
    x = pool.symbol("x")
    m = ak.Matrix([[x, 1], [Fraction(1, 2), ak.sin(x)]])
    m2 = pickle.loads(pickle.dumps(m))
    assert isinstance(m2, ak.Matrix)
    assert m2.shape() == (2, 2)
    assert [[str(e) for e in r] for r in m2.to_list()] == [[str(e) for e in r] for r in m.to_list()]
    assert m2.to_list()[0][0] == x


def test_pickle_default_pool_constant():
    e = ak.pi / 2
    assert pickle.loads(pickle.dumps(e)) == e
    assert pickle.loads(pickle.dumps(e))._pool is ak.default_pool()


def _worker_diff(e):
    x = e._pool.symbol("x")
    return ak.diff(e, x)


def _worker_matrix(m):
    return m.transpose()


def test_multiprocessing_pool_map(pool):
    x = pool.symbol("x")
    inputs = [x**k * ak.sin(x) for k in range(1, 5)]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2) as mp_pool:
        results = mp_pool.map(_worker_diff, inputs)
        mats = mp_pool.map(_worker_matrix, [ak.Matrix([[x, 1], [2, 3]])])
    for e, r in zip(inputs, results):
        assert isinstance(r, ak.DerivedResult)
        # Results come home to the original pool and combine with local work.
        assert r.value == ak.diff(e, x).value
        assert r.steps == ak.diff(e, x).steps
    assert mats[0].to_list()[1][0] == pool.integer(1)
    assert mats[0].to_list()[0][1]._pool is pool
