"""N8: ArbBall mid/rad must be readable at full precision (mid +- rad encloses)."""

import random
from fractions import Fraction

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")
from mpmath import mp, mpf  # noqa: E402


def _frac(x):
    """Exact Fraction of an mpf."""
    s = mpmath.mpmathify(x)
    man, exp = s.man, s.exp
    return Fraction(int(man)) * (Fraction(2) ** int(exp))


def _cases(pool):
    mk = pool.integer
    pi = ak.parse("pi", pool)
    return [
        ("pi", pi, lambda: mp.pi),
        ("e", ak.exp(mk(1)), lambda: mp.e),
        ("sqrt2", ak.sqrt(mk(2)), lambda: mp.sqrt(2)),
        ("zeta3", ak.zeta(mk(3)), lambda: mp.zeta(3)),
        ("exp(1/3)", ak.exp(mk(1) / mk(3)), lambda: mp.exp(mpf(1) / 3)),
    ]


@pytest.mark.parametrize("digits", [50, 100, 300, 1000])
def test_full_precision_readout_encloses(digits):
    pool = ak.ExprPool()
    bits = int(digits * 3.33) + 64
    old = mp.dps
    mp.dps = digits + 30
    try:
        for name, e, ref in _cases(pool):
            b = ak.interval_eval(e, {}, bits)
            mid, rad = b.mid_exact(), b.rad_exact()
            truth = _frac(ref())
            assert abs(truth - mid) <= rad, name
            # the readout carries the requested digits
            assert rad < Fraction(1, 10**digits), (name, b.rad_str()[:30])
            m, r = b.to_mpf()
            assert abs(ref() - m) <= r + mpf(10) ** (-(digits + 25)), name
            assert abs(mpf(b.mid_str()) - m) < mpf(10) ** (-(digits + 20))
            assert b.prec == bits
    finally:
        mp.dps = old


def test_randomised_balls():
    rnd = random.Random(10)
    pool = ak.ExprPool()
    old = mp.dps
    mp.dps = 400
    for _ in range(25):
        n = rnd.randint(2, 500)
        bits = rnd.choice([128, 300, 700, 1200])
        for e, ref in [
            (ak.sqrt(pool.integer(n)), mp.sqrt(n)),
            (ak.log(pool.integer(n)), mp.log(n)),
        ]:
            b = ak.interval_eval(e, {}, bits)
            assert abs(_frac(ref) - b.mid_exact()) <= b.rad_exact()
    mp.dps = old


def test_repr_shows_digits_and_radius():
    pool = ak.ExprPool()
    b = ak.interval_eval(ak.zeta(pool.integer(3)), {}, 3400)
    s = str(b)
    assert "± 0.00e0" not in s
    assert s.startswith("ArbBall(1.2020569031595942853997381615114499907649862923404988817922715")
    assert float(b.rad) > 0


def test_f64_accessors_still_enclose():
    pool = ak.ExprPool()
    b = ak.interval_eval(ak.zeta(pool.integer(3)), {}, 3400)
    assert b.lo <= float(mpmath.zeta(3)) <= b.hi
