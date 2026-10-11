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

Undefined functions
~~~~~~~~~~~~~~~~~~~

A function name with no built-in, registered or reserved meaning —
``pool.func("q", [t])`` — is an *undefined function*: an arbitrary smooth
function. Its derivative is the canonical node
``Derivative(q(u…), k…)`` (orders per argument slot), printed as SymPy prints
it: ``Derivative(q(t), t)``, ``Derivative(u(x, t), x, t)``, and
``Subs(Derivative(f(_xi_1), _xi_1), _xi_1, t^2)`` for ``f'(t²)``. ``diff``,
``diff_forward`` and ``symbolic_grad`` all differentiate it; ``subs`` with the
key ``f(t)`` differentiates its value into the derivative nodes; ``parse``
reads the printed form back once ``f`` is declared (``{"f": f}`` in the
symbol table); ``eval_expr`` refuses it with ``E-EVAL-005``.

.. function:: euler_lagrange(L: Expr, coords: list[Expr], t: Expr) -> list[Expr]

   Euler–Lagrange equations ``d/dt(∂L/∂q̇ᵢ) − ∂L/∂qᵢ`` (each ``= 0``) for the
   coordinates ``coords = [q1(t), …]``; higher derivatives in ``L`` add the
   Euler–Poisson terms. Raises ``DiffError`` ``E-DIFF-005`` (not an undefined
   function) or ``E-DIFF-007`` (not a function of ``t`` alone).

   Example::

      t = pool.symbol("t"); th = pool.func("theta", [t])
      L = m*l**2*diff(th, t).value**2/2 + m*g*l*cos(th)
      euler_lagrange(L, [th], t)   # [g*l*m*sin(theta(t)) + l^2*m*Derivative(theta(t), t, t)]

.. function:: functions_to_symbols(exprs, funcs: list[Expr], t: Expr, domain=None)

   Replace ``q(t)``, ``q'(t)``, ``q''(t)``, … by symbols ``q``, ``q_dot``,
   ``q_ddot``, ``q_d3`` … (a trailing ``_`` keeps them fresh). Returns
   ``(new_exprs, {node: symbol})``; use it to hand equations of motion to
   ``solve`` or ``lambdify``.

.. function:: replace_function(expr: Expr, pattern: Expr, body: Expr) -> Expr

   Substitute the definition ``pattern = f(x1, …)`` ↦ ``body`` for every
   application and every derivative of ``f``, at any argument (SymPy's
   ``subs(f, Lambda(x, body)).doit()``).

``dsolve(eq, t, y)`` with ``y = pool.func("y", [t])`` and no ``derivs``
accepts an equation written with ``y(t)`` and its derivatives.

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

   Each case is a ``(condition, value)`` pair where *condition* is a
   predicate built with ``pool.gt`` / ``pool.lt`` / ``pool.ge`` / ``pool.le`` /
   ``pool.eq`` / ``pool.ne`` (Python's ``>`` is not overloaded on
   expressions). SymPy's ``(value, condition)`` order is accepted too; a case
   with no predicate in it raises ``TypeError``. The first case whose
   condition holds wins, and *default* is used when none does::

      expr = piecewise(
          [(pool.gt(x, pool.integer(0)), x),
           (pool.lt(x, pool.integer(0)), pool.integer(-1) * x)],
          pool.integer(0),  # x == 0
      )

   ``diff`` differentiates each branch and keeps the conditions, giving the
   derivative on the interior of every piece; ``eval_expr``, ``interval_eval``
   and ``compile_expr`` evaluate it numerically.
