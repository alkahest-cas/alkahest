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

    result = evaluate(x, {}, mode="exact")

    assert result.status == "unsupported"
    assert result.value is None
    assert result.reason == "E-EVAL-001"


def test_interval_evaluation_covers_every_primitive_that_advertises_numeric_ball():
    """The interval evaluator dispatches through the primitive registry, so the
    set of functions it accepts *is* the set the registry reports a
    ``numeric_ball`` kernel for. It used to carry its own hand-written list,
    which had silently dropped ``bessel_j0`` / ``bessel_j1``: both had rigorous
    ball kernels and both were refused with ``E-EVAL-010``.

    No single probe point is inside every kernel's domain (``acosh`` needs
    x >= 1, ``atanh`` needs |x| < 1), so a primitive counts as covered if some
    probe evaluates — a domain refusal is a different answer from "no rule for
    this name", and only the second is a coverage gap.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    registry = ak.PrimitiveRegistry.default_registry()

    refused = []
    for row in registry.coverage_report():
        if not row["numeric_ball"]:
            continue
        call = pool.func(row["name"], [x])
        if all(
            evaluate(call, {x: ak.ArbBall(probe, 0.0)}, mode="interval").status != "ok"
            for probe in (1.5, 0.5)
        ):
            refused.append(row["name"])

    assert not refused, f"advertised numeric_ball but interval evaluation refuses: {refused}"


def _at(build, xi):
    """`build(y)` at the exact point `xi`, as a tight point enclosure."""
    pool = ak.ExprPool()
    y = pool.symbol("y")
    result = evaluate(build(y), {y: ak.ArbBall(xi, 0.0)}, mode="interval")
    assert result.status == "ok"
    return result.value.mid


def test_interval_evaluation_of_bessel():
    """The two the coverage audit was opened for."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    for build, expected in ((ak.bessel_j0, 0.7651976865579665), (ak.bessel_j1, 0.4400505857449335)):
        result = evaluate(build(x), {x: ak.ArbBall(1.0, 0.0)}, mode="interval")

        assert result.status == "ok"
        assert abs(result.value.mid - expected) < 1e-15

        # J0 and J1 oscillate, so the enclosure has to come from a Lipschitz
        # bound around the midpoint, not from hulling the two endpoints; on a
        # wide interval an endpoint hull misses the function's own extrema.
        wide = evaluate(build(x), {x: ak.ArbBall(0.0, 1.0)}, mode="interval")
        assert wide.status == "ok"
        for i in range(21):
            xi = -1.0 + 2.0 * i / 20
            assert wide.value.contains(_at(build, xi)), f"J({xi}) escaped {wide.value}"
