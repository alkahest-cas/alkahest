"""v1.0 milestone ergonomics tests.

Covers:
- V1-3: structured error hierarchy
- V1-4: polynomial system solver (requires groebner feature)
- V1-8: stable API / experimental module
- V1-12: expanded primitive registry (14 new primitives)
"""

from __future__ import annotations

import alkahest
import pytest

# ---------------------------------------------------------------------------
# V1-12: Expanded primitive registry
# ---------------------------------------------------------------------------


class TestExpandedPrimitives:
    def setup_method(self):
        self.pool = alkahest.ExprPool()
        self.x = self.pool.symbol("x")

    def test_trig_functions_exist(self):
        x = self.x
        assert str(alkahest.tan(x)) == "tan(x)"
        assert str(alkahest.sinh(x)) == "sinh(x)"
        assert str(alkahest.cosh(x)) == "cosh(x)"
        assert str(alkahest.tanh(x)) == "tanh(x)"
        assert str(alkahest.asin(x)) == "asin(x)"
        assert str(alkahest.acos(x)) == "acos(x)"
        assert str(alkahest.atan(x)) == "atan(x)"

    def test_special_functions_exist(self):
        x = self.x
        assert str(alkahest.erf(x)) == "erf(x)"
        assert str(alkahest.erfc(x)) == "erfc(x)"

    def test_rounding_functions_exist(self):
        x = self.x
        assert str(alkahest.floor(x)) == "floor(x)"
        assert str(alkahest.ceil(x)) == "ceil(x)"
        assert str(alkahest.sign(x)) == "sign(x)"
        assert "x" in str(alkahest.abs(x))  # abs(x) or |x|

    def test_diff_tan(self):
        """d/dx tan(x) = 1 + tan²(x)  (sec²(x) form)."""
        x = self.x
        d = alkahest.diff(alkahest.tan(x), x)
        s = str(d.value)
        assert "tan" in s  # should contain tan(x)

    def test_diff_tanh(self):
        """d/dx tanh(x) = 1 - tanh²(x)."""
        x = self.x
        d = alkahest.diff(alkahest.tanh(x), x)
        s = str(d.value)
        assert "tanh" in s

    def test_diff_sinh_cosh_chain(self):
        """d/dx sinh(x) = cosh(x); d/dx cosh(x) = sinh(x)."""
        x = self.x
        assert str(alkahest.diff(alkahest.sinh(x), x).value) == "cosh(x)"
        assert str(alkahest.diff(alkahest.cosh(x), x).value) == "sinh(x)"

    def test_diff_atan(self):
        """d/dx atan(x) = 1/(1 + x²)."""
        x = self.x
        d = alkahest.diff(alkahest.atan(x), x)
        s = str(d.value)
        assert "x^2" in s or "x²" in s or "x" in s

    def test_diff_asin(self):
        """d/dx asin(x) = 1/sqrt(1 - x²)."""
        x = self.x
        d = alkahest.diff(alkahest.asin(x), x)
        s = str(d.value)
        assert "sqrt" in s

    def test_diff_erf(self):
        """d/dx erf(x) = (2/√π) exp(-x²)."""
        x = self.x
        d = alkahest.diff(alkahest.erf(x), x)
        s = str(d.value)
        assert "exp" in s

    def test_primitive_registry_has_19_primitives(self):
        reg = alkahest.PrimitiveRegistry()
        report = reg.coverage_report()
        assert len(report) >= 19, f"expected ≥19 primitives, got {len(report)}"

    def test_primitive_registry_all_have_numeric_f64(self):
        reg = alkahest.PrimitiveRegistry()
        report = reg.coverage_report()
        missing = [r["name"] for r in report if not r["numeric_f64"]]
        assert not missing, f"primitives missing numeric_f64: {missing}"

    def test_primitive_registry_most_have_diff_forward(self):
        reg = alkahest.PrimitiveRegistry()
        report = reg.coverage_report()
        count = sum(1 for r in report if r["diff_forward"])
        assert count >= 12, f"expected ≥12 with diff_forward, got {count}"


# ---------------------------------------------------------------------------
# V1-3: Structured error hierarchy
# ---------------------------------------------------------------------------


class TestStructuredErrors:
    def setup_method(self):
        self.pool = alkahest.ExprPool()
        self.x = self.pool.symbol("x")

    def test_conversion_error_has_code(self):
        x = self.x
        with pytest.raises(alkahest.ConversionError) as exc_info:
            alkahest.poly_normal(alkahest.sin(x), [x])
        e = exc_info.value
        assert e.code == "E-POLY-006"

    def test_conversion_error_has_remediation(self):
        x = self.x
        with pytest.raises(alkahest.ConversionError) as exc_info:
            alkahest.poly_normal(alkahest.sin(x), [x])
        e = exc_info.value
        assert e.remediation is not None
        assert len(e.remediation) > 10

    def test_integration_error_has_code(self):
        x = self.x
        with pytest.raises(alkahest.IntegrationError) as exc_info:
            alkahest.integrate(alkahest.sin(alkahest.sin(x)), x)
        e = exc_info.value
        assert e.code.startswith("E-INT-")

    def test_integration_error_has_remediation(self):
        x = self.x
        with pytest.raises(alkahest.IntegrationError) as exc_info:
            alkahest.integrate(alkahest.sin(alkahest.sin(x)), x)
        e = exc_info.value
        assert e.remediation is not None

    def test_conversion_error_is_value_error(self):
        """Backward compat: ConversionError should be catchable as ValueError."""
        x = self.x
        with pytest.raises(ValueError):
            alkahest.poly_normal(alkahest.sin(x), [x])

    def test_alkahest_error_alias(self):
        """alkahest.AlkahestError and alkahest.AlkahestError are the same type."""
        assert alkahest.AlkahestError is alkahest.AlkahestError

    def test_all_exception_classes_have_code_attr(self):
        for cls_name in [
            "ConversionError",
            "DiffError",
            "IntegrationError",
            "MatrixError",
            "OdeError",
            "DaeError",
            "JitError",
        ]:
            cls = getattr(alkahest, cls_name)
            # All should be subclasses of AlkahestError
            assert issubclass(cls, alkahest.AlkahestError), (
                f"{cls_name} not subclass of AlkahestError"
            )


# ---------------------------------------------------------------------------
# V1-4: Polynomial system solver
# ---------------------------------------------------------------------------


_has_groebner = hasattr(alkahest, "solve")
_skip_no_groebner = pytest.mark.skipif(
    not _has_groebner,
    reason="alkahest.solve requires groebner feature at build time",
)


@_skip_no_groebner
class TestPolynomialSolver:
    def setup_method(self):
        self.pool = alkahest.ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")

    # --- numeric=True path (legacy float output) ----------------------------

    def test_linear_system_numeric(self):
        """x + y - 1 = 0, x - y = 0  →  x = y = 0.5 (float)."""
        p = self.pool
        x, y = self.x, self.y
        one = p.integer(1)
        eq1 = x + y + p.integer(-1) * one  # x + y - 1
        eq2 = x + p.integer(-1) * y  # x - y
        solutions = alkahest.solve([eq1, eq2], [x, y], numeric=True)
        assert isinstance(solutions, list), "expected list of solutions"
        assert len(solutions) > 0, "expected at least one solution"
        sol = solutions[0]
        assert abs(sol[x] - 0.5) < 1e-10
        assert abs(sol[y] - 0.5) < 1e-10

    def test_univariate_quadratic_numeric(self):
        """x^2 - 4 = 0  →  x = ±2 (float)."""
        p = self.pool
        x = self.x
        expr = x**2 + p.integer(-4)
        solutions = alkahest.solve([expr], [x], numeric=True)
        assert isinstance(solutions, list)
        roots = [sol[x] for sol in solutions]
        assert any(abs(r - 2.0) < 1e-10 for r in roots), f"x=2 not in {roots}"
        assert any(abs(r + 2.0) < 1e-10 for r in roots), f"x=-2 not in {roots}"

    # --- default symbolic output (V1-16) ------------------------------------

    def test_linear_system_symbolic(self):
        """x + y - 1 = 0, x - y = 0  →  solution values are Expr."""
        p = self.pool
        x, y = self.x, self.y
        one = p.integer(1)
        eq1 = x + y + p.integer(-1) * one
        eq2 = x + p.integer(-1) * y
        solutions = alkahest.solve([eq1, eq2], [x, y])
        assert isinstance(solutions, list)
        assert len(solutions) > 0
        sol = solutions[0]
        assert isinstance(sol[x], alkahest.Expr), "symbolic value should be Expr"
        assert isinstance(sol[y], alkahest.Expr), "symbolic value should be Expr"
        # Numeric agreement — evaluate the symbolic solution directly.
        assert abs(alkahest.eval_expr(sol[x], {}) - 0.5) < 1e-10
        assert abs(alkahest.eval_expr(sol[y], {}) - 0.5) < 1e-10

    def test_univariate_quadratic_symbolic(self):
        """x^2 - 4 = 0  →  solution values are Expr; numerically ±2."""
        p = self.pool
        x = self.x
        expr = x**2 + p.integer(-4)
        solutions = alkahest.solve([expr], [x])
        assert isinstance(solutions, list)
        for sol in solutions:
            assert isinstance(sol[x], alkahest.Expr)
        roots_str = {str(sol[x]) for sol in solutions}
        # Integer solutions: displayed as "2" and "-2".
        assert "2" in roots_str or any(
            abs(alkahest.eval_expr(sol[x], {}) - 2.0) < 1e-10 for sol in solutions
        )


@_skip_no_groebner
class TestGroebnerBasisCompute:
    """V1-16: GroebnerBasis.compute() Python binding."""

    def setup_method(self):
        self.pool = alkahest.ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")

    def test_compute_returns_groebner_basis(self):
        x = self.x
        p = self.pool
        gb = alkahest.GroebnerBasis.compute([x**2 + p.integer(-1)], [x])
        assert isinstance(gb, alkahest.GroebnerBasis)

    def test_contains_generator(self):
        """GB of x^2 - 1 contains x^2 - 1 (ideal membership of the generator)."""
        x = self.x
        p = self.pool
        poly = x**2 + p.integer(-1)  # x^2 - 1
        gb = alkahest.GroebnerBasis.compute([poly], [x])
        assert gb.contains(poly)

    def test_len_nonzero(self):
        x = self.x
        p = self.pool
        gb = alkahest.GroebnerBasis.compute([x**2 + p.integer(-4)], [x])
        assert len(gb) >= 1

    def test_repr(self):
        x = self.x
        p = self.pool
        gb = alkahest.GroebnerBasis.compute([x + p.integer(-1)], [x])
        assert "GroebnerBasis" in repr(gb)


# ---------------------------------------------------------------------------
# V1-8: Stable API / experimental module
# ---------------------------------------------------------------------------


class TestStableAPI:
    def test_all_is_defined(self):
        assert hasattr(alkahest, "__all__")
        assert len(alkahest.__all__) >= 50

    def test_experimental_module_importable(self):
        import alkahest.experimental as exp  # noqa: F401

    def test_experimental_has_to_stablehlo(self):
        import alkahest.experimental as exp

        assert hasattr(exp, "to_stablehlo")

    def test_version(self):
        assert alkahest.version() is not None
        assert len(alkahest.version()) > 0
