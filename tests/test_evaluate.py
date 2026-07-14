from fractions import Fraction

import alkahest as ak
from alkahest.experimental import evaluate


def test_auto_evaluate_prefers_exact_rational_results():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    result = evaluate(x + pool.rational(1, 3), {x: Fraction(1, 6)})

    assert result.status == "ok"
    assert result.backend == "exact_rational"
    assert result.value == Fraction(1, 2)
    assert result.enclosure is None


def test_evaluate_f64_reports_backend():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    result = evaluate(ak.sin(x), {x: 0.0}, mode="f64")

    assert result.status == "ok"
    assert result.backend == "interpreter_f64"
    assert result.value == 0.0


def test_evaluate_interval_returns_enclosure():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    result = evaluate(x * x, {x: ak.ArbBall(2.0, 0.1)}, mode="interval", precision_bits=128)

    assert result.status == "ok"
    assert result.is_enclosure
    assert result.value.contains(4.0)
    assert result.requested_precision_bits == 128


def test_unsupported_evaluation_returns_stable_status():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    result = evaluate(ak.log(x), {}, mode="exact")

    assert result.status == "unsupported"
    assert result.value is None
    assert result.reason == "E-EVAL-001"
