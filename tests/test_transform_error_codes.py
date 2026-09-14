"""The transform / ODE / series engines report *why*, not just *that*.

Eight conversion functions in ``alkahest-py/src/lib.rs`` used to end at

    pyo3::exceptions::PyValueError::new_err(e.to_string())

— a bare ``ValueError`` with no ``.code``, no ``.remediation`` and no ``.span``,
for ``dsolve``, ``dsolve_system``, ``series_solve``, ``asymptotic_expand``,
``Fps`` and all three transform tables. Everything else in the library carries a
stable ``E-SUBSYSTEM-NNN``, and the whole point of that guarantee is that a
caller can branch on the reason instead of matching English prose.

The distinction that was being thrown away is not a nicety. "This Laplace
transform's causality hypothesis is refuted" and "this integrand is not in the
table yet" are opposite facts: the first says no implementation will ever answer
(``L{theta(t+1)} = 1/s``, not ``e**s/s`` — the edge is outside the range the
unilateral integral sees), the second says rewrite the input or wait for a wider
table. Flattened to one ``ValueError`` they are indistinguishable, and an
unattended loop that reads the first as the second retries forever.

Each test below provokes one code and asserts three things: the code is the
expected string, the remediation is non-empty, and the exception is still a
``ValueError`` so that pre-existing ``except ValueError`` handlers are unbroken.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from alkahest import experimental as ex


@pytest.fixture
def pool():
    return ak.ExprPool()


def assert_structured(exc, code, cls=None):
    """Every structured error carries all three, and stays a ``ValueError``."""
    assert exc.code == code, f"expected {code}, got {exc.code}"
    assert exc.remediation, f"{code} has no remediation"
    assert isinstance(exc, ValueError), (
        f"{type(exc).__name__} is not a ValueError subclass; "
        "`except ValueError` around this call would stop catching it"
    )
    assert isinstance(exc, ak.AlkahestError)
    if cls is not None:
        assert isinstance(exc, cls), f"{code} arrived as {type(exc).__name__}, not {cls.__name__}"
    # The code is greppable from the message too, for logs that only keep str().
    assert code in str(exc)


# ---------------------------------------------------------------------------
# Laplace — E-TRANSFORM-001 … 004
# ---------------------------------------------------------------------------


def test_laplace_table_miss_is_e_transform_001(pool):
    t, s = pool.symbol("t"), pool.symbol("s")
    with pytest.raises(ak.TransformError) as ei:
        ex.laplace_transform(ak.log(t), t, s)
    assert_structured(ei.value, "E-TRANSFORM-001", ak.TransformError)


def test_inverse_laplace_table_miss_is_e_transform_002(pool):
    t, s = pool.symbol("t"), pool.symbol("s")
    with pytest.raises(ak.TransformError) as ei:
        ex.inverse_laplace_transform(ak.log(s), s, t)
    assert_structured(ei.value, "E-TRANSFORM-002", ak.TransformError)


def test_laplace_same_variable_is_e_transform_003(pool):
    t = pool.symbol("t")
    with pytest.raises(ak.TransformError) as ei:
        ex.laplace_transform(t, t, t)
    assert_structured(ei.value, "E-TRANSFORM-003", ak.TransformError)


def test_laplace_negative_step_edge_is_a_refutation_not_a_table_gap(pool):
    """``theta(t + 1)`` is a *refutation*: ``E-TRANSFORM-004``, not ``001``.

    The unilateral transform integrates over ``t >= 0``, so an edge at
    ``a = -1`` is invisible to it — ``L{theta(t+1)} = 1/s``, while the shift
    rule would emit ``e**s/s``. No wider table fixes that, which is exactly why
    it must not share a code with the table misses above.
    """
    t, s = pool.symbol("t"), pool.symbol("s")
    theta = pool.func("heaviside", [pool.add([t, pool.integer(1)])])
    with pytest.raises(ak.TransformError) as ei:
        ex.laplace_transform(theta, t, s)
    assert_structured(ei.value, "E-TRANSFORM-004", ak.TransformError)


def test_inverse_laplace_advance_factor_is_a_refutation(pool):
    """``e**(+s)/s`` is the transform of no causal ``f``."""
    t, s = pool.symbol("t"), pool.symbol("s")
    with pytest.raises(ak.TransformError) as ei:
        ex.inverse_laplace_transform(ak.exp(s) / s, s, t)
    assert_structured(ei.value, "E-TRANSFORM-004", ak.TransformError)


def test_laplace_nonnegative_step_edge_still_transforms(pool):
    """The control for the two refusals above.

    A gate made only of refusals is passed by an engine that refuses
    everything. ``theta(t - 1)`` is the nearest neighbour that must still be
    answered: ``L{theta(t-1)} = e**(-s)/s``.
    """
    t, s = pool.symbol("t"), pool.symbol("s")
    theta = pool.func("heaviside", [t - pool.integer(1)])
    out = ex.laplace_transform(theta, t, s)
    assert out is not None


# ---------------------------------------------------------------------------
# Fourier — E-TRANSFORM-011 … 013
# ---------------------------------------------------------------------------


def test_fourier_table_miss_is_e_transform_011(pool):
    x, xi = pool.symbol("x"), pool.symbol("xi")
    with pytest.raises(ak.TransformError) as ei:
        ex.fourier_transform(ak.log(x), x, xi)
    assert_structured(ei.value, "E-TRANSFORM-011", ak.TransformError)


def test_fourier_same_variable_is_e_transform_012(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.TransformError) as ei:
        ex.fourier_transform(x, x, x)
    assert_structured(ei.value, "E-TRANSFORM-012", ak.TransformError)


def test_fourier_divergent_rate_is_a_refutation_not_a_table_gap(pool):
    """``theta(x)*e**(2x)`` has no Fourier transform: the integral diverges.

    The one-sided-exponential entry is ``theta(x)*e**(-a x)`` with ``a > 0``.
    At ``a = -2`` there is nothing to tabulate, so this is ``E-TRANSFORM-013``
    and not the ``011`` that a missing table entry would get.
    """
    x, xi = pool.symbol("x"), pool.symbol("xi")
    f = pool.func("heaviside", [x]) * ak.exp(pool.integer(2) * x)
    with pytest.raises(ak.TransformError) as ei:
        ex.fourier_transform(f, x, xi)
    assert_structured(ei.value, "E-TRANSFORM-013", ak.TransformError)


def test_fourier_convergent_rate_still_transforms(pool):
    """The control: flip the sign and the same shape must be answered."""
    x, xi = pool.symbol("x"), pool.symbol("xi")
    f = pool.func("heaviside", [x]) * ak.exp(pool.integer(-2) * x)
    assert ex.fourier_transform(f, x, xi) is not None


# ---------------------------------------------------------------------------
# Z transform — E-TRANSFORM-101 … 103
# ---------------------------------------------------------------------------


def test_z_transform_table_miss_is_e_transform_101(pool):
    n, z = pool.symbol("n"), pool.symbol("z")
    with pytest.raises(ak.TransformError) as ei:
        ex.z_transform(ak.log(n), n, z)
    assert_structured(ei.value, "E-TRANSFORM-101", ak.TransformError)


def test_inverse_z_transform_delta_term_is_e_transform_102(pool):
    """A constant term would invert to the Kronecker delta, which has no
    primitive here — declined rather than fabricated."""
    n, z = pool.symbol("n"), pool.symbol("z")
    with pytest.raises(ak.TransformError) as ei:
        ex.inverse_z_transform(pool.integer(1), z, n)
    assert_structured(ei.value, "E-TRANSFORM-102", ak.TransformError)


def test_z_transform_same_variable_is_e_transform_103(pool):
    n = pool.symbol("n")
    with pytest.raises(ak.TransformError) as ei:
        ex.z_transform(n, n, n)
    assert_structured(ei.value, "E-TRANSFORM-103", ak.TransformError)


# ---------------------------------------------------------------------------
# dsolve / dsolve_system / series_solve — E-ODE-*
# ---------------------------------------------------------------------------


def test_dsolve_unsupported_class_is_e_ode_010(pool):
    x, y, yp = pool.symbol("x"), pool.symbol("y"), pool.symbol("yp")
    with pytest.raises(ak.OdeError) as ei:
        ex.dsolve(yp - ak.exp(y * x), x, y, [yp])
    assert_structured(ei.value, "E-ODE-010", ak.OdeError)


def test_dsolve_riccati_without_a_seed_is_e_ode_014(pool):
    """``y' = y**2 + x`` is recognised as Riccati and declined for a *reason*.

    ``E-ODE-014`` says the class matched and the method needs a particular
    solution; ``E-ODE-010`` would have said no class matched at all. Supplying a
    seed is a next step only the first of those suggests.
    """
    x, y, yp = pool.symbol("x"), pool.symbol("y"), pool.symbol("yp")
    with pytest.raises(ak.OdeError) as ei:
        ex.dsolve(yp - (y * y + x), x, y, [yp])
    assert_structured(ei.value, "E-ODE-014", ak.OdeError)


def test_dsolve_separable_control_still_solves(pool):
    """The control for the two dsolve refusals: ``y' = y`` must still solve."""
    x, y, yp = pool.symbol("x"), pool.symbol("y"), pool.symbol("yp")
    assert ex.dsolve(yp - y, x, y, [yp])


def test_dsolve_system_nonlinear_is_e_ode_030(pool):
    t, y1, y2 = pool.symbol("t"), pool.symbol("y1"), pool.symbol("y2")
    ode = ak.ODE([y1, y2], [y1 * y1, y2], t)
    with pytest.raises(ak.OdeError) as ei:
        ex.dsolve_system(ode)
    assert_structured(ei.value, "E-ODE-030", ak.OdeError)


def test_dsolve_system_time_varying_is_e_ode_031(pool):
    t, y1, y2 = pool.symbol("t"), pool.symbol("y1"), pool.symbol("y2")
    ode = ak.ODE([y1, y2], [t * y1, y2], t)
    with pytest.raises(ak.OdeError) as ei:
        ex.dsolve_system(ode)
    assert_structured(ei.value, "E-ODE-031", ak.OdeError)


def test_series_solve_irregular_singular_is_e_ode_041(pool):
    """``x**3 y'' + y = 0`` has an irregular singular point at 0.

    A fact about the equation: no Frobenius series exists there, so no wider
    implementation would find one.
    """
    x = pool.symbol("x")
    with pytest.raises(ak.OdeError) as ei:
        ex.series_solve(
            x, x ** pool.integer(3), pool.integer(0), pool.integer(1), pool.integer(0), 5
        )
    assert_structured(ei.value, "E-ODE-041", ak.OdeError)


def test_series_solve_degenerate_leading_coefficient_is_e_ode_043(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.OdeError) as ei:
        ex.series_solve(x, pool.integer(0), pool.integer(1), pool.integer(1), pool.integer(0), 5)
    assert_structured(ei.value, "E-ODE-043", ak.OdeError)


def test_series_solve_ordinary_point_control_still_solves(pool):
    """The control: ``y'' + y = 0`` at an ordinary point must still expand."""
    x = pool.symbol("x")
    out = ex.series_solve(x, pool.integer(1), pool.integer(0), pool.integer(1), pool.integer(0), 6)
    assert out


def test_series_solve_codes_no_longer_collide_with_the_numeric_integrators(pool):
    """``E-ODE-021`` used to mean two incompatible things.

    ``ode::numeric::NumericOdeError::StepSizeTooSmall`` and
    ``ode::series_solve::SeriesError::IrregularSingular`` both returned it, so a
    caller branching on the code could read "the adaptive integrator gave up" as
    "this equation has no Frobenius series". The series block moved to
    ``E-ODE-04x``; this pins that it stays moved.
    """
    x = pool.symbol("x")
    with pytest.raises(ak.OdeError) as ei:
        ex.series_solve(
            x, x ** pool.integer(3), pool.integer(0), pool.integer(1), pool.integer(0), 5
        )
    assert not ei.value.code.startswith("E-ODE-02")


# ---------------------------------------------------------------------------
# asymptotic_expand — E-ASYMPT-*
# ---------------------------------------------------------------------------


def test_asymptotic_invalid_term_count_is_e_asympt_001(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.AsymptoticError) as ei:
        ex.asymptotic_expand(pool.integer(1) / x, x, 0)
    assert_structured(ei.value, "E-ASYMPT-001", ak.AsymptoticError)


def test_asymptotic_unsupported_scale_is_e_asympt_005(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.AsymptoticError) as ei:
        ex.asymptotic_expand(ak.exp(ak.exp(x)), x, 3)
    assert_structured(ei.value, "E-ASYMPT-005", ak.AsymptoticError)


def test_asymptotic_power_scale_control_still_expands(pool):
    """The control: a power scale is in range and must still be answered."""
    x = pool.symbol("x")
    assert ex.asymptotic_expand(pool.integer(1) / (x + pool.integer(1)), x, 3) is not None


# ---------------------------------------------------------------------------
# Fps — E-FPS-*
# ---------------------------------------------------------------------------


def test_fps_denominator_vanishing_at_zero_is_e_fps_001():
    with pytest.raises(ak.FpsError) as ei:
        ex.Fps.from_rational([0, 1], [0, 1])
    assert_structured(ei.value, "E-FPS-001", ak.FpsError)


def test_fps_polar_part_is_e_fps_002(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.FpsError) as ei:
        ex.Fps.from_expr(pool.integer(1) / x, x, 5)
    assert_structured(ei.value, "E-FPS-002", ak.FpsError)


def test_fps_non_rational_coefficient_is_e_fps_003(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.FpsError) as ei:
        ex.Fps.from_expr(ak.log(x), x, 5)
    assert_structured(ei.value, "E-FPS-003", ak.FpsError)


def test_fps_composition_needs_zero_constant_term_e_fps_004():
    f = ex.Fps.from_rational([1, 1], [1])
    with pytest.raises(ak.FpsError) as ei:
        f.compose(ex.Fps.from_rational([1, 1], [1]))
    assert_structured(ei.value, "E-FPS-004", ak.FpsError)


def test_fps_inverse_needs_nonzero_constant_term_e_fps_006():
    with pytest.raises(ak.FpsError) as ei:
        ex.Fps.from_rational([0, 1], [1]).inverse()
    assert_structured(ei.value, "E-FPS-006", ak.FpsError)


def test_fps_control_still_builds():
    """The control: ``1/(1 - x)`` is the textbook case and must still work."""
    assert ex.Fps.from_rational([1], [1, -1]) is not None


# ---------------------------------------------------------------------------
# The whole surface
# ---------------------------------------------------------------------------

_NEW_CODES = [
    "E-TRANSFORM-001",
    "E-TRANSFORM-002",
    "E-TRANSFORM-003",
    "E-TRANSFORM-004",
    "E-TRANSFORM-011",
    "E-TRANSFORM-012",
    "E-TRANSFORM-013",
    "E-TRANSFORM-101",
    "E-TRANSFORM-102",
    "E-TRANSFORM-103",
    "E-ODE-040",
    "E-ODE-041",
    "E-ODE-042",
    "E-ODE-043",
    "E-ODE-044",
    "E-ODE-045",
]


def test_every_new_code_is_in_the_rust_registry():
    """A code that is raisable and unregistered is an undocumented contract.

    ``tests/test_error_code_registry.py`` runs the full gate; this narrower
    check names the codes this change introduced, so a failure points at them
    rather than at a count.
    """
    from pathlib import Path

    codes_rs = Path(__file__).resolve().parents[1] / "alkahest-core/src/errors/codes.rs"
    if not codes_rs.is_file():  # pragma: no cover - wheel install, no checkout
        pytest.skip("checkout-only source (absent in a wheel)")
    text = codes_rs.read_text(encoding="utf-8")
    missing = [c for c in _NEW_CODES if f'"{c}"' not in text]
    assert not missing, f"raisable but unregistered: {missing}"


def test_the_new_exception_classes_are_importable_and_value_errors():
    """``except ValueError`` is what existing callers wrote, and must keep working."""
    from alkahest import exceptions as exc_mod

    for name in ("TransformError", "AsymptoticError", "FpsError"):
        assert name in ak.__all__, f"{name} is missing from alkahest.__all__"
        cls = getattr(ak, name)
        assert issubclass(cls, ak.AlkahestError)
        assert issubclass(cls, ValueError)
        # The name resolves to the *native* class, not the pure-Python stub:
        # the engines raise the native one, so a caller who caught the stub
        # would catch nothing. (See the CudaError note in ``__init__.py``.)
        assert cls is not getattr(exc_mod, name)
        assert cls.__module__ == "alkahest"
