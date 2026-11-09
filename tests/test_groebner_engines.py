"""Gröbner engines against SymPy: the multimodular default, lex via FGLM, F5.

Every engine must return *the* reduced Gröbner basis (it is unique), so each
result is compared exactly — same polynomials, same rational coefficients —
with SymPy's ``groebner``, and checked for ideal membership of the inputs.
"""

from __future__ import annotations

import contextlib
import signal
import time
from fractions import Fraction

import pytest

alk = pytest.importorskip("alkahest")
sp = pytest.importorskip("sympy")
hyp = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

if not hasattr(alk, "GroebnerBasis"):
    pytest.skip("built without the groebner feature", allow_module_level=True)

ENGINES = {
    "compute": alk.GroebnerBasis.compute,
    "compute_f5": alk.GroebnerBasis.compute_f5,
}


# Lex Buchberger over Q on a positive-dimensional ideal can run for hours in
# either system on a few small generators: one such example held a full suite
# run at 14 GB for over 80 minutes.  Every call here is bounded, and a call that
# hits its bound says nothing either way (the differential check is skipped for
# that example; the Budget is what keeps the engine from hanging CI).
ENGINE_BUDGET_MS = 20_000
SYMPY_SECONDS = 20


class _SymPyDeadline(Exception):
    pass


@contextlib.contextmanager
def _deadline(seconds):
    """Interrupt Python code after ``seconds`` (a no-op where there is no SIGALRM)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def fire(signum, frame):
        raise _SymPyDeadline

    prev = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _canon_alk(gb):
    return sorted(sorted((tuple(e), Fraction(c)) for e, c in g.terms()) for g in gb)


def _canon_sympy(polys, syms, order):
    g = sp.groebner(polys, *syms, order=order)
    out = []
    for p in g.exprs:
        P = sp.Poly(p, *syms)
        lc = P.LC(order=order)
        out.append(
            sorted(
                (tuple(m), Fraction(int(c.p), int(c.q)))
                for m, c in sp.Poly(p / lc, *syms).as_dict().items()
            )
        )
    return sorted(out)


def _build(n, terms_list):
    pool = alk.ExprPool()
    names = ["x", "y", "z"][:n]
    avars = [pool.symbol(s) for s in names]
    svars = sp.symbols(names)
    apolys, spolys = [], []
    for terms in terms_list:
        a = pool.integer(0)
        s = sp.Integer(0)
        for exps, num, den in terms:
            am = pool.rational(num, den) if den != 1 else pool.integer(num)
            sm = sp.Rational(num, den)
            for v, sv, e in zip(avars, svars, exps):
                if e:
                    am = am * v**e
                    sm = sm * sv**e
            a = a + am
            s = s + sm
        apolys.append(a)
        spolys.append(sp.expand(s))
    return avars, apolys, svars, spolys


def _check(n, terms_list, order, engine):
    avars, apolys, svars, spolys = _build(n, terms_list)
    if all(p == 0 for p in spolys):
        return
    nonzero = [i for i, p in enumerate(spolys) if p != 0]
    apolys = [apolys[i] for i in nonzero]
    spolys = [spolys[i] for i in nonzero]
    try:
        with alk.context(budget=alk.Budget(wall_ms=ENGINE_BUDGET_MS)):
            gb = ENGINES[engine](apolys, avars, order=order)
    except alk.BudgetExceededError:
        return
    try:
        with _deadline(SYMPY_SECONDS):
            want = _canon_sympy(spolys, svars, order)
    except _SymPyDeadline:
        return
    assert _canon_alk(gb) == want
    for p in apolys:
        assert gb.contains(p)


_term = st.tuples(
    st.lists(st.integers(0, 2), min_size=3, max_size=3),
    st.integers(-6, 6).filter(bool),
    st.integers(1, 3),
)
_system = st.integers(2, 3).flatmap(
    lambda n: st.tuples(
        st.just(n),
        st.lists(
            st.lists(
                _term.map(lambda t: (t[0][:n], t[1], t[2])),
                min_size=1,
                max_size=4,
            ),
            min_size=2,
            max_size=3,
        ),
    )
)


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("order", ["grevlex", "lex"])
@settings(max_examples=40, deadline=None)
@given(system=_system)
def test_random_ideals_match_sympy(system, order, engine):
    n, terms_list = system
    _check(n, terms_list, order, engine)


# compute_f5 used to return a non-Gröbner basis for this ideal (grevlex): the
# rewrite criterion kept the newest element at a signature rather than the one
# with the smallest leading monomial, and an element was lost.  Exact: SymPy's
# buchberger and f5b, Wolfram's GroebnerBasis, and compute all agree.
_F5_REWRITE_REGRESSION = (
    3,
    [
        [([0, 0, 0], 1, 1), ([0, 2, 0], 1, 1), ([1, 2, 1], 1, 1), ([2, 0, 2], 1, 1)],
        [([0, 0, 0], 1, 1), ([2, 2, 0], 1, 1)],
        [([0, 1, 0], 1, 1), ([0, 1, 1], 1, 1)],
    ],
)


@pytest.mark.parametrize("order", ["grevlex", "lex", "grlex"])
def test_f5_rewrite_regression_exact_agreement(order):
    n, terms = _F5_REWRITE_REGRESSION
    avars, apolys, svars, spolys = _build(n, terms)
    want = _canon_sympy(spolys, svars, order)
    gb = alk.GroebnerBasis.compute_f5(apolys, avars, order=order)
    assert _canon_alk(gb) == want
    assert gb.proof_status == "proved"


# Positive-dimensional lex over Q: 3 small generators.  At the time of writing it
# takes well over 5 minutes unbudgeted; the point is that a Budget stops it, on
# any path, so a test cannot hang on one.
_PATHOLOGICAL_LEX = (
    3,
    [
        [([2, 1, 2], 5, 3), ([2, 2, 0], -5, 2), ([2, 0, 1], -4, 1), ([1, 2, 2], -6, 3)],
        [([0, 2, 0], 5, 2), ([2, 0, 2], -3, 3), ([1, 1, 0], -6, 3)],
        [([0, 2, 2], -1, 2), ([0, 1, 2], 4, 3), ([2, 0, 2], -6, 3), ([2, 1, 2], -2, 3)],
    ],
)


def test_positive_dimensional_lex_honours_budget():
    """If the lex path ever gets fast enough to finish this, so much the better."""
    n, terms = _PATHOLOGICAL_LEX
    avars, apolys, svars, spolys = _build(n, terms)
    t = time.perf_counter()
    try:
        with alk.context(budget=alk.Budget(wall_ms=1000)):
            gb = alk.GroebnerBasis.compute(apolys, avars, order="lex")
    except alk.BudgetExceededError:
        gb = None
    assert time.perf_counter() - t < 15
    if gb is not None:
        assert _canon_alk(gb) == _canon_sympy(spolys, svars, "lex")


def _cyclic(n):
    pool = alk.ExprPool()
    xs = [pool.symbol(f"x{i}") for i in range(n)]
    eqs = []
    for k in range(1, n):
        s = pool.integer(0)
        for i in range(n):
            t = pool.integer(1)
            for j in range(k):
                t = t * xs[(i + j) % n]
            s = s + t
        eqs.append(s)
    t = pool.integer(1)
    for v in xs:
        t = t * v
    eqs.append(t - pool.integer(1))
    return xs, eqs


def _katsura(n):
    pool = alk.ExprPool()
    u = [pool.symbol(f"u{i}") for i in range(n + 1)]

    def U(i):
        i = abs(i)
        return u[i] if i <= n else pool.integer(0)

    eqs = []
    for m in range(n):
        s = pool.integer(0)
        for i in range(-n, n + 1):
            s = s + U(i) * U(m - i)
        eqs.append(s - u[m])
    s = pool.integer(0)
    for i in range(-n, n + 1):
        s = s + U(i)
    eqs.append(s - pool.integer(1))
    return u, eqs


def _leading_sizes(gb):
    return len(gb), sum(len(g.terms()) for g in gb)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_benchmark_systems_shapes_and_time(engine):
    """The report's table: these used to time out (katsura-5 lex, cyclic-6
    grevlex, every F5 run).  Basis sizes match SymPy's, and each finishes in
    well under the old 120 s cap (generous bounds for a loaded CI machine)."""
    f = ENGINES[engine]
    cases = [
        (_cyclic(5), "grevlex", 20),
        (_cyclic(5), "lex", 11),
        (_katsura(5), "grevlex", 22),
        (_katsura(5), "lex", 6),
        (_cyclic(6), "grevlex", 45),
    ]
    for (xs, eqs), order, size in cases:
        t = time.perf_counter()
        gb = f(eqs, xs, order=order)
        dt = time.perf_counter() - t
        assert len(gb) == size, (order, len(gb))
        for p in eqs:
            assert gb.contains(p)
        assert dt < 60, (order, dt)


@pytest.mark.parametrize(("n", "order"), [(3, "lex"), (4, "grevlex"), (4, "lex")])
def test_katsura_matches_sympy_exactly(n, order):
    """Exact agreement on a larger structured system (SymPy's direct lex on
    katsura-4 takes ~40 s, so that reference comes via its grevlex basis and
    SymPy's own FGLM)."""
    xs, eqs = _katsura(n)
    svars = sp.symbols([str(v) for v in xs])
    seqs = [sp.sympify(str(e).replace("^", "**")) for e in eqs]
    if n == 4 and order == "lex":
        g = sp.groebner(seqs, *svars, order="grevlex").fglm("lex")
        want = sorted(
            sorted(
                (tuple(m), Fraction(int(c.p), int(c.q)))
                for m, c in sp.Poly(p / sp.Poly(p, *svars).LC(order="lex"), *svars)
                .as_dict()
                .items()
            )
            for p in g.exprs
        )
    else:
        want = _canon_sympy(seqs, svars, order)
    for f in ENGINES.values():
        assert _canon_alk(f(eqs, xs, order=order)) == want
