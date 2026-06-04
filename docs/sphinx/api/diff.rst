Calculus API
============

.. currentmodule:: alkahest

Symbolic differentiation and integration with full derivation logging.

Differentiation
---------------

.. function:: diff(expr: Expr, var: Expr) -> DerivedResult

   Compute the symbolic derivative of *expr* with respect to *var*.

   Applies chain rule, product rule, and the differentiation rule for each
   registered primitive. Returns a :class:`DerivedResult` with the full
   derivation log.

   :param expr: The expression to differentiate.
   :param var: The differentiation variable (must be a ``Symbol`` node).
   :returns: ``DerivedResult`` with ``.value = d(expr)/d(var)``.

   Example::

      pool = ExprPool()
      x = pool.symbol("x")

      dr = diff(sin(x**2), x)
      print(dr.value)   # 2*x*cos(x^2)

      dr = diff(x**3, x)
      print(dr.value)   # 3*x^2

.. function:: diff_forward(expr: Expr, var: Expr) -> DerivedResult

   Compute the derivative using forward-mode automatic differentiation
   (dual numbers). Produces the same result as ``diff`` through a
   different path; useful for cross-validation.

.. function:: symbolic_grad(expr: Expr, vars: list[Expr]) -> list[Expr]

   Differentiate *expr* with respect to each variable in *vars*.

   Returns a list of ``Expr`` objects (not ``DerivedResult``).

   **Not** the same name as :func:`grad` in :doc:`transform` — that function
   differentiates a :class:`TracedFn` and returns a :class:`GradTracedFn` for
   numeric evaluation (often with :func:`jit`). Use ``symbolic_grad`` for bare
   expressions; use ``grad`` only after ``@trace``.

   Example::

      grads = symbolic_grad(x**2 * y + sin(x * y), [x, y])
      # grads[0] = 2*x*y + y*cos(x*y)
      # grads[1] = x**2 + x*cos(x*y)

Registered primitives
~~~~~~~~~~~~~~~~~~~~~

The differentiation rules cover all 23 registered primitives:

``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``atan2``,
``sinh``, ``cosh``, ``tanh``, ``exp``, ``log``, ``sqrt``, ``abs``,
``sign``, ``erf``, ``erfc``, ``gamma``, ``floor``, ``ceil``, ``round``,
``min``, ``max``

Integration
-----------

.. function:: integrate(expr: Expr, var: Expr) -> DerivedResult

   Compute the symbolic antiderivative of *expr* with respect to *var*.

   Applies a rule table for common forms, then escalates to the full
   transcendental Risch algorithm:

   - Power rule (integer exponents)
   - Logarithm: ``∫ 1/x dx = log(x)``
   - Exponential tower: ``∫ exp(a*x + b) dx``, ``∫ xⁿ·exp(x) dx``
   - Linear substitution: ``∫ f(a*x + b) dx``
   - Trigonometric: sin, cos, and standard compositions
   - **Rational functions** ``A(x)/D(x)``: Hermite reduction (repeated factors),
     Rothstein–Trager (rational residues → log), irreducible quadratics
     (negative discriminant → arctan; positive discriminant → log with √Δ
     coefficients), degree-≥3 irreducible factors → ``RootSum`` node
     (Lazard–Rioboo–Trager).
   - **Rational coefficient × exp**: ``∫ f(x)·exp(η) dx`` for ``f ∈ ℚ(x)``
     via the rational Risch DE (Bronstein §6.1).

   Raises :exc:`IntegrationError` (``E-INT-001``) for integrands outside
   the supported classes (e.g. mixed algebraic+transcendental).
   Raises :exc:`IntegrationError` (``E-INT-004``) when the integrand is
   provably non-elementary (``sin(x)/x``, ``exp(x)/x``, ``exp(x²)``, etc.).

   :param expr: The integrand.
   :param var: The variable of integration.
   :returns: ``DerivedResult`` with ``.value = ∫ expr dvar``
      (no constant of integration).

   Example::

      r = integrate(x**3, x)
      print(r.value)   # 1/4 * x^4

      r = integrate(sin(x), x)
      print(r.value)   # -cos(x)

      r = integrate(exp(x), x)
      print(r.value)   # exp(x)

Primitives (math functions)
---------------------------

All of these accept ``Expr`` arguments and return ``Expr`` objects.
They are thin wrappers that create ``Call(primitive, args)`` nodes.

.. function:: sin(x: Expr) -> Expr
.. function:: cos(x: Expr) -> Expr
.. function:: tan(x: Expr) -> Expr
.. function:: asin(x: Expr) -> Expr
.. function:: acos(x: Expr) -> Expr
.. function:: atan(x: Expr) -> Expr
.. function:: atan2(y: Expr, x: Expr) -> Expr
.. function:: sinh(x: Expr) -> Expr
.. function:: cosh(x: Expr) -> Expr
.. function:: tanh(x: Expr) -> Expr
.. function:: exp(x: Expr) -> Expr
.. function:: log(x: Expr) -> Expr
.. function:: sqrt(x: Expr) -> Expr
.. function:: abs(x: Expr) -> Expr
.. function:: sign(x: Expr) -> Expr
.. function:: erf(x: Expr) -> Expr
.. function:: erfc(x: Expr) -> Expr
.. function:: gamma(x: Expr) -> Expr
.. function:: floor(x: Expr) -> Expr
.. function:: ceil(x: Expr) -> Expr
.. function:: round(x: Expr) -> Expr
.. function:: min(a: Expr, b: Expr) -> Expr
.. function:: max(a: Expr, b: Expr) -> Expr

Piecewise
---------

.. function:: piecewise(cases: list[tuple[Expr, Expr]], default: Expr) -> Expr

   Create a piecewise expression.

   Each case is a ``(value, condition)`` pair where *condition* is a
   :class:`Predicate` expression. The *default* is used when no condition
   holds::

      expr = piecewise(
          [(x, x > pool.integer(0)),
           (pool.integer(-1) * x, x < pool.integer(0))],
          default=pool.integer(0),  # x == 0
      )
