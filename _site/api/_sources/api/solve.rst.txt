Solver API
==========

.. currentmodule:: alkahest

Polynomial system solving via Gröbner bases. Requires ``--features groebner``.

.. function:: solve(equations: list[Expr], variables: list[Expr]) -> list[dict] or GroebnerBasis

   Solve a system of polynomial equations symbolically.

   Uses Lex Gröbner basis computation followed by triangular
   back-substitution. Quadratic factors are solved exactly with
   symbolic square roots.

   :param equations: List of polynomial expressions (each equal to zero).
   :param variables: Variables to solve for.
   :returns:

      - ``list[dict[Expr, Expr]]`` — one dict per solution, mapping
        variable → value (symbolic, e.g. ``sqrt(2)/2``).
      - ``[]`` (empty list) — if the system is inconsistent.
      - :class:`GroebnerBasis` — for parametric (infinite) solution sets.

   Example::

      pool = ExprPool()
      x = pool.symbol("x")
      y = pool.symbol("y")

      # Linear system
      sols = solve([x + y - pool.integer(1), x - y], [x, y])
      # → [{x: 1/2, y: 1/2}]

      # Circle ∩ line (irrational)
      sols = solve([x**2 + y**2 - pool.integer(1), y - x], [x, y])
      # → [{x: sqrt(2)/2, y: sqrt(2)/2},
      #    {x: -sqrt(2)/2, y: -sqrt(2)/2}]

   .. note::

      Solutions are symbolic by default. Evaluate numerically with
      :func:`eval_expr` when needed::

         from alkahest import eval_expr
         for sol in sols:
             for var, val in sol.items():
                 print(f"{var} ≈ {eval_expr(val, {}):.6f}")

   **Upcoming (v1.1):** A ``numeric=True`` keyword argument will return
   float values directly.

GroebnerBasis
-------------

.. class:: GroebnerBasis

   A Gröbner basis for a polynomial ideal.

   .. classmethod:: compute(polys: list[Expr], vars: list[Expr], order: str = "GRevLex") -> GroebnerBasis

      Compute a Gröbner basis using the F4 algorithm.

      :param order: Monomial order — ``"Lex"``, ``"GrLex"``, or ``"GRevLex"``.
         Use ``"Lex"`` for elimination; ``"GRevLex"`` is generally fastest.

      **Upcoming (v1.1):** The Python binding for this method is being added.

   .. method:: reduce(expr: Expr) -> Expr

      Reduce *expr* modulo the ideal.

   .. method:: contains(expr: Expr) -> bool

      Test ideal membership.

   .. method:: eliminate(vars: list[Expr]) -> GroebnerBasis

      Compute the elimination ideal by removing generators that involve
      the specified variables.

      Useful for implicitization of parametric curves and surfaces.

.. class:: GbPoly

   A polynomial element of a Gröbner basis computation, with rational
   coefficients represented as FLINT rationals.
