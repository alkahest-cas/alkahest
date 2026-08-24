"""Python surface for the Risch-Norman (parallel Risch) heuristic integrator.

The contract this file pins is narrow and deliberate:

* the function returns a result object and never raises on a failed attempt;
* a declined result carries no verdict — nothing on the object can be read as
  "no elementary antiderivative exists";
* every solved result really does differentiate back to the integrand, checked
  here independently of the kernel's own gate.
"""

import alkahest as ak
import pytest
from alkahest.experimental import ParallelRischResult, integrate_parallel_risch

SAMPLES = (0.3719, 0.9137, 1.4231, 2.1719)


@pytest.fixture
def env():
    pool = ak.ExprPool()
    with ak.context(pool=pool):
        yield pool, ak.symbol("x")


def _parse(pool, src):
    return ak.parse(src, pool)


def _differentiates_back(x, cand, integrand):
    """Independent numeric check, outside the kernel's gate."""
    d = ak.diff(cand, x)
    d = d.value if hasattr(d, "value") else d
    agreed = 0
    for xv in SAMPLES:
        try:
            a = float(ak.eval_expr(d, {x: xv}))
            b = float(ak.eval_expr(integrand, {x: xv}))
        except Exception:
            continue
        if a != a or b != b:
            continue
        assert abs(a - b) <= 1e-6 * max(1.0, abs(b)), f"d/dx F != f at x={xv}"
        agreed += 1
    return agreed


# --- solves -----------------------------------------------------------------

SOLVED = [
    "x^2*exp(x)",
    "x*log(x)",
    "exp(x)/(exp(x)+1)",
    "exp(2*x)/(exp(x)+1)",
    "1/(1+exp(-x))",
    "(x*log(x))^(-1)",
    "1/(x+1)^2",
    "x^3*exp(-x^2)",
]


@pytest.mark.parametrize("src", SOLVED)
def test_solves_and_verifies(env, src):
    pool, x = env
    f = _parse(pool, src)
    res = integrate_parallel_risch(f, x)
    assert isinstance(res, ParallelRischResult)
    assert res.solved, f"expected a solution for {src}, got: {res.reason}"
    assert res.verification in ("exact", "numeric")
    assert res.reason is None
    cand = res.antiderivative()
    assert cand is not None
    assert _differentiates_back(x, cand, f) > 0


def test_new_coverage_beyond_the_default_engine(env):
    """`exp(2x)/(exp(x)+1)` is `E-INT-001` from `integrate`, solved here."""
    pool, x = env
    f = _parse(pool, "exp(2*x)/(exp(x)+1)")
    with pytest.raises(Exception) as excinfo:
        ak.integrate(f, x)
    assert "E-INT-001" in str(excinfo.value)

    res = integrate_parallel_risch(f, x)
    assert res.solved
    assert _differentiates_back(x, res.antiderivative(), f) > 0


# --- declines ---------------------------------------------------------------

DECLINED = [
    "exp(x^2)",  # genuinely non-elementary
    "exp(x)/x",  # Ei
    "sin(x)/x",  # Si — and out of ring besides
    "x/(exp(x)+1)",  # polylogarithmic
    "log(log(x))",  # x*log(log x) - li(x)
    "log(x)/(1+x)",  # dilogarithmic
    "sqrt(tan(x))",  # elementary, but outside the ring
    "1/(x^2+1)",  # elementary (atan), but needs a field larger than Q
]


@pytest.mark.parametrize("src", DECLINED)
def test_declines_without_a_verdict(env, src):
    pool, x = env
    res = integrate_parallel_risch(_parse(pool, src), x)
    assert not res.solved
    assert res.antiderivative() is None
    assert res.verification is None
    assert isinstance(res.reason, str)
    assert res.reason


def test_a_decline_is_not_a_non_elementarity_claim(env):
    """The two declines below are mathematically opposite; the API cannot tell
    them apart, and must not pretend to."""
    pool, x = env
    non_elementary = integrate_parallel_risch(_parse(pool, "exp(x^2)"), x)
    elementary = integrate_parallel_risch(_parse(pool, "1/(x^2+1)"), x)
    assert not non_elementary.solved
    assert not elementary.solved
    # `1/(x^2+1)` is `atan(x)`; the default engine finds it.
    assert ak.integrate(_parse(pool, "1/(x^2+1)"), x) is not None
    # Nothing on the declined result distinguishes the two cases, and no
    # attribute of it mentions a verdict.
    for res in (non_elementary, elementary):
        assert "non-elementary" not in res.reason.lower().replace("not a non-elementarity", "")
        assert not hasattr(res, "non_elementary")
        assert not hasattr(res, "certificate")


def test_never_raises_on_a_hopeless_integrand(env):
    pool, x = env
    for src in ("sin(x^2)", "exp(exp(exp(x)))", "x^(1/3)", "1/log(x)"):
        res = integrate_parallel_risch(_parse(pool, src), x)
        assert isinstance(res, ParallelRischResult)
        assert res.solved in (True, False)


def test_repr_states_the_outcome(env):
    pool, x = env
    solved = integrate_parallel_risch(_parse(pool, "x*exp(x)"), x)
    declined = integrate_parallel_risch(_parse(pool, "exp(x^2)"), x)
    assert "solved=True" in repr(solved)
    assert "solved=False" in repr(declined)
