Code generation API
===================

.. currentmodule:: alkahest

MLIR dialect emission and C code export.

MLIR / StableHLO
----------------

.. function:: to_stablehlo(expr: Expr, vars: list[Expr], fn_name: str = "alkahest_fn") -> str

   Emit the expression as textual MLIR in the StableHLO dialect.

   The output is valid input to ``mlir-opt`` and the XLA compiler, and uses
   only ``stablehlo.*`` ops. Lowered: ``+``, ``*``, ``**``, ``sin``, ``cos``,
   ``tan``, ``exp``, ``log``, ``sqrt``, ``sinh``, ``cosh``, ``tanh``, ``abs``,
   ``sign``, ``floor``, ``ceil``, ``round``, ``heaviside``, ``atan2``,
   ``atan``, ``asin``, ``acos``, ``atanh``, ``max``, ``min``, ``Piecewise``
   and predicates (via ``select``/``compare``), and an unbound ``pi``.
   Anything else — ``erf``, ``gamma``, ``lambert_w``, ``asinh``, ``acosh``,
   user functions — raises :class:`AlkahestError` with ``E-STABLEHLO-001``
   naming the function; a free symbol not in ``vars`` raises
   ``E-STABLEHLO-003``. No partial or empty module is ever returned.

   :param expr: The expression to lower.
   :param vars: Input variables, in order.
   :param fn_name: Name of the emitted function.
   :returns: Textual MLIR string.
   :raises AlkahestError: ``E-STABLEHLO-001`` … ``E-STABLEHLO-005`` when the
      expression cannot be exported exactly.

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

.. function:: emit_c_expr(expr: Expr, vars: Expr | list[Expr], var_names=None, fn_name="f") -> str
.. function:: emit_c_vec(exprs: list[Expr], vars: Expr | list[Expr], var_names=None, fn_name="eval_vec") -> str

   Emit ISO C99 for an arbitrary expression (``double f(double x, …)``) or for
   several outputs at once (``void eval_vec(double x, …, double *out)``).

   * **Common sub-expressions** are computed once into ``const double``
     temporaries, across all outputs of ``emit_c_vec`` jointly.  Integer powers
     up to 4, their reciprocals and half-integer powers become products,
     divisions and a single shared ``sqrt``; ``a·b⁻¹`` becomes ``a / b``.
     The n = 4 pendulum's equations of motion emit ~25 kB (was 320 kB); a
     Black–Scholes price + gradient + Hessian emits one ``log`` and one
     ``sqrt`` (was 131 and 664).
   * **Constants**: ``pi`` (unless listed in *vars*) and ``exp(1)`` are
     emitted as the nearest ``double`` (``3.141592653589793``); rationals as
     the correctly rounded ``double`` the interpreter uses.
   * **Functions**: the C99 ``<math.h>`` set (``sin`` … ``atanh``, ``erf``,
     ``erfc``, ``gamma`` → ``tgamma``, ``floor``, ``ceil``, ``round``,
     ``atan2``, ``min``/``max`` → ``fmin``/``fmax``); ``sign``, ``heaviside``
     and ``Piecewise`` as ternaries; ``lambert_w`` through an emitted
     ``static double alkahest_lambert_w0`` helper (Halley iteration, relative
     error ≤ 1e-15 for ``x ≥ −1/e + 10⁻³``, ``#ifndef``-guarded).

   :raises AlkahestError: ``E-CODEGEN-001`` a function with no C lowering
      (``diracdelta``, Bessel, elliptic, digamma, …), ``E-CODEGEN-002`` a node
      with no straight-line C form (``O(x)``, quantifiers, root sums),
      ``E-CODEGEN-003`` a free symbol not listed in *vars*.  ``AlkahestError``
      subclasses ``ValueError``.

   Example::

      print(emit_c_vec([sin(x) * cos(y), sin(x) + cos(y)], [x, y]))
      # void eval_vec(double x, double y, double *out) {
      #     const double _t0 = sin(x);
      #     const double _t1 = cos(y);
      #     out[0] = (_t0 * _t1);
      #     out[1] = (_t0 + _t1);
      # }

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
