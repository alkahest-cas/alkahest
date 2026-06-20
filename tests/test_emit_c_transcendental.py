"""Tests for emit_c_expr and emit_c_vec — transcendental C code emission.

Issue #7b: extend emit_c to handle transcendental functions (sin, cos, exp, log,
tan, sqrt, atan2, erf, …) and add a vector-output compile path.
"""

import os
import shutil
import subprocess
import tempfile

import pytest
from alkahest import (
    ExprPool,
    cos,
    emit_c_expr,
    emit_c_vec,
    exp,
    log,
    sin,
    sqrt,
)

# ===========================================================================
# Helpers
# ===========================================================================


def pool():
    return ExprPool()


# ===========================================================================
# Basic emit_c_expr tests
# ===========================================================================


class TestEmitCExpr:
    """Tests for emit_c_expr: transcendental + general expression emission."""

    def test_sin_plus_x_squared(self):
        """The original failing case: sin(x) + x² must succeed and contain sin(."""
        p = pool()
        x = p.symbol("x")
        expr = sin(x) + x**2
        code = emit_c_expr(expr, x)
        assert "sin(" in code, f"expected sin( in:\n{code}"
        assert "double f(double x)" in code, f"expected signature:\n{code}"
        assert "return " in code, f"expected return:\n{code}"

    def test_cos_emitted(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_expr(cos(x), x)
        assert "cos(" in code

    def test_exp_emitted(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_expr(exp(x), x)
        assert "exp(" in code

    def test_log_emitted(self):
        p = pool()
        x = p.symbol("x")
        expr = log(x * x + p.integer(1))
        code = emit_c_expr(expr, x)
        assert "log(" in code

    def test_sqrt_emitted(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = sqrt(x**2 + y**2)
        code = emit_c_expr(expr, [x, y], var_names=["x", "y"])
        assert "sqrt(" in code
        assert "double x" in code
        assert "double y" in code

    def test_custom_fn_name(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_expr(sin(x), x, fn_name="my_func")
        assert "my_func" in code

    def test_custom_var_name(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_expr(sin(x), x, var_names=["theta"])
        assert "theta" in code
        assert "sin(theta)" in code

    def test_multivar(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = sin(x) * cos(y)
        code = emit_c_expr(expr, [x, y], var_names=["x", "y"])
        assert "sin(" in code
        assert "cos(" in code

    def test_polynomial_still_works(self):
        """Pure polynomial: x² + 2x + 1 — must still emit valid C."""
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2) * x + p.integer(1)
        code = emit_c_expr(expr, x)
        assert "double f(double x)" in code
        assert "return " in code

    def test_unsupported_function_raises(self):
        """diracdelta has no C math.h equivalent — should raise ValueError."""
        p = pool()
        x = p.symbol("x")
        # Build diracdelta(x) directly via pool.func
        expr = p.func("diracdelta", [x])
        with pytest.raises(ValueError, match="diracdelta"):
            emit_c_expr(expr, x)

    def test_missing_variable_raises(self):
        """Referencing an unlisted symbol raises ValueError."""
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = x + y
        with pytest.raises(ValueError):
            emit_c_expr(expr, x)  # y not listed

    def test_single_var_as_list(self):
        """vars may be given as a one-element list."""
        p = pool()
        x = p.symbol("x")
        expr = sin(x) + x**2
        code_scalar = emit_c_expr(expr, x)
        code_list = emit_c_expr(expr, [x])
        # Both should contain sin(
        assert "sin(" in code_scalar
        assert "sin(" in code_list

    def test_default_fn_name_is_f(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_expr(sin(x), x)
        assert "double f(" in code

    def test_uses_symbolic_name_by_default(self):
        """When var_names is omitted, the symbolic name is used."""
        p = pool()
        theta = p.symbol("theta")
        code = emit_c_expr(sin(theta), theta)
        assert "theta" in code


# ===========================================================================
# emit_c_vec tests
# ===========================================================================


class TestEmitCVec:
    """Tests for emit_c_vec: vector-output compile path."""

    def test_basic_two_outputs(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_vec([sin(x), cos(x)], x)
        assert "double *out" in code, f"expected out pointer:\n{code}"
        assert "out[0]" in code, f"expected out[0]:\n{code}"
        assert "out[1]" in code, f"expected out[1]:\n{code}"
        assert "sin(" in code
        assert "cos(" in code

    def test_void_return_type(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_vec([sin(x), cos(x)], x)
        assert code.strip().startswith("void "), f"expected void return:\n{code}"

    def test_custom_fn_name(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_vec([sin(x)], x, fn_name="my_vec")
        assert "my_vec" in code

    def test_multivar_vector(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        f0 = sin(x)
        f1 = x**2 + y
        code = emit_c_vec([f0, f1], [x, y], var_names=["x", "y"])
        assert "double *out" in code
        assert "out[0]" in code
        assert "out[1]" in code

    def test_single_output(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_vec([sin(x)], x)
        assert "out[0]" in code
        assert "out[1]" not in code

    def test_three_outputs(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_vec([sin(x), cos(x), exp(x)], x)
        assert "out[0]" in code
        assert "out[1]" in code
        assert "out[2]" in code

    def test_empty_exprs_raises(self):
        p = pool()
        x = p.symbol("x")
        with pytest.raises(ValueError):
            emit_c_vec([], x)

    def test_unsupported_function_raises(self):
        p = pool()
        x = p.symbol("x")
        expr = p.func("diracdelta", [x])
        with pytest.raises(ValueError, match="diracdelta"):
            emit_c_vec([expr], x)

    def test_default_fn_name_is_eval_vec(self):
        p = pool()
        x = p.symbol("x")
        code = emit_c_vec([sin(x)], x)
        assert "eval_vec" in code


# ===========================================================================
# Round-trip numeric tests (require a C compiler)
# ===========================================================================


# Prefer the system gcc/cc (not a custom wrapper that may lack system headers).
# `/usr/bin/gcc` is tried first; then any `cc` / `gcc` on PATH.
def _find_cc():
    for candidate in ["/usr/bin/gcc", "/usr/bin/cc"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("cc") or shutil.which("gcc")


CC = _find_cc()
requires_cc = pytest.mark.skipif(CC is None, reason="no C compiler found")


def _compile_and_run(c_source: str, x_val: float, n_outputs: int = 1) -> list:
    """Compile c_source + a main() that calls the function at x_val.

    For scalar functions the expected signature is:
        double f(double x);
    For vector functions:
        void eval_vec(double x, double *out);
    """
    if n_outputs == 1:
        main = f"""
#include <math.h>
#include <stdio.h>
{c_source}
int main(void) {{
    printf("%.15f\\n", f({x_val!r}));
    return 0;
}}
"""
    else:
        assignments = "\n".join(f'    printf("%.15f\\n", out[{i}]);' for i in range(n_outputs))
        main = f"""
#include <math.h>
#include <stdio.h>
{c_source}
int main(void) {{
    double out[{n_outputs}];
    eval_vec({x_val!r}, out);
{assignments}
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "test.c")
        exe = os.path.join(tmp, "test")
        with open(src, "w") as fh:
            fh.write(main)
        # Put -lm after the source for linkers that require ordering.
        result = subprocess.run(
            [CC, "-O2", "-I/usr/include/x86_64-linux-gnu", src, "-o", exe, "-lm"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"C compile failed:\n{result.stderr}"
        out = subprocess.check_output([exe], text=True).strip().split()
        return [float(v) for v in out]


@requires_cc
class TestRoundTrip:
    """Compile emitted C and verify numeric correctness."""

    def test_sin_plus_x_squared_at_1(self):
        import math

        p = pool()
        x = p.symbol("x")
        expr = sin(x) + x**2
        code = emit_c_expr(expr, x)
        vals = _compile_and_run(code, 1.0)
        expected = math.sin(1.0) + 1.0
        assert abs(vals[0] - expected) < 1e-12, f"got {vals[0]}, expected {expected}"

    def test_exp_cos_at_pi_over_4(self):
        import math

        p = pool()
        x = p.symbol("x")
        expr = exp(x) * cos(x)
        code = emit_c_expr(expr, x)
        xv = math.pi / 4
        vals = _compile_and_run(code, xv)
        expected = math.exp(xv) * math.cos(xv)
        assert abs(vals[0] - expected) < 1e-12, f"got {vals[0]}, expected {expected}"

    def test_sqrt_sum_of_squares(self):
        import math

        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = sqrt(x**2 + y**2)
        code = emit_c_expr(expr, [x, y], var_names=["x", "y"])
        # Inline y=3.0 into the wrapper manually — compile_and_run only supports one arg
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "test.c")
            exe = os.path.join(tmp, "test")
            main = f"""
#include <math.h>
#include <stdio.h>
{code}
int main(void) {{
    printf("%.15f\\n", norm(3.0, 4.0));
    return 0;
}}
"""
            with open(src, "w") as fh:
                fh.write(main.replace("double f(", "double norm("))
            result = subprocess.run(
                [CC, "-O2", "-I/usr/include/x86_64-linux-gnu", src, "-o", exe, "-lm"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            val = float(subprocess.check_output([exe], text=True).strip())
        expected = math.sqrt(3.0**2 + 4.0**2)
        assert abs(val - expected) < 1e-12

    def test_vec_sin_cos_at_1(self):
        import math

        p = pool()
        x = p.symbol("x")
        code = emit_c_vec([sin(x), cos(x)], x)
        vals = _compile_and_run(code, 1.0, n_outputs=2)
        assert abs(vals[0] - math.sin(1.0)) < 1e-12
        assert abs(vals[1] - math.cos(1.0)) < 1e-12

    def test_polynomial_numeric(self):
        """x² + 2x + 1 at x=3 should be 16."""
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2) * x + p.integer(1)
        code = emit_c_expr(expr, x)
        vals = _compile_and_run(code, 3.0)
        assert abs(vals[0] - 16.0) < 1e-12
