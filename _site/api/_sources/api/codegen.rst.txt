Code generation API
===================

.. currentmodule:: alkahest

MLIR dialect emission and C code export.

MLIR / StableHLO
----------------

.. function:: to_stablehlo(expr: Expr, vars: list[Expr], fn_name: str = "alkahest_fn") -> str

   Emit the expression as textual MLIR in the StableHLO dialect.

   The output is valid input to ``mlir-opt`` and the XLA compiler.
   Supported ops: ``Add``, ``Mul``, ``Pow``, ``sin``, ``cos``, ``exp``,
   ``log``, ``sqrt`` → mapped to ``stablehlo.*`` ops.

   :param expr: The expression to lower.
   :param vars: Input variables, in order.
   :param fn_name: Name of the emitted function.
   :returns: Textual MLIR string.

   Example::

      mlir = to_stablehlo(sin(x) * exp(y), [x, y], fn_name="my_fn")
      with open("kernel.mlir", "w") as f:
          f.write(mlir)

C code emission
---------------

.. function:: emit_c(expr: Expr, vars: list[Expr], var_name: str, fn_name: str) -> str

   Emit a standalone C function using only ``<math.h>``::

      c = emit_c(sin(x) * exp(pool.integer(-1) * x), [x],
                 var_name="x", fn_name="damped_sin")
      # "double damped_sin(double x) { return sin(x) * exp(-x); }"

   The emitted function has no Alkahest dependency.

Horner form
-----------

.. function:: horner(expr: Expr, var: Expr) -> Expr

   Rewrite a polynomial in Horner's form.

   Horner's form minimizes multiplications and improves numerical conditioning::

      h = horner(x**3 + pool.integer(2)*x**2 + pool.integer(3)*x + pool.integer(4), x)
      # x*(x*(x + 2) + 3) + 4

Primitive registry
------------------

.. class:: PrimitiveRegistry

   The global registry of mathematical primitives.

   Each primitive registers: a simplification rule, forward- and
   reverse-mode differentiation, an MLIR lowering, a Lean theorem tag,
   and a numerical evaluator.

   .. method:: coverage_report() -> str

      Print a table showing which capabilities are registered for each
      primitive. Useful for auditing coverage::

         PrimitiveRegistry.coverage_report()
         # sin: NUMERIC_F64=✓ DIFF_FORWARD=✓ NUMERIC_BALL=✓ LEAN=✓ ...

   .. method:: list_primitives() -> list[str]

      Return the names of all registered primitives.
