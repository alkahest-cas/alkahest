Solver API
==========

.. currentmodule:: alkahest

Polynomial system solving via Gröbner bases. The ``groebner`` Cargo feature is **included in all PyPI wheels** by default since 2.3.1 — no special build flag needed.

.. function:: solve(equations: list[Expr], variables: list[Expr]) -> list[dict] or GroebnerBasis

   Solve a system of polynomial equations symbolically.

   Uses Lex Gröbner basis computation followed by triangular
   back-substitution. Quadratic factors are solved exactly with
   symbolic square roots.

   :param equations: List of polynomial expressions (each equal to zero).
   :param variables: Variables to solve for. Free symbols that appear in
      *equations* but are omitted here are treated as parameters.
   :returns:

      - ``list[dict[Expr, Expr]]`` — one dict per solution, mapping
        variable → value (symbolic, e.g. ``sqrt(2)/2``).
      - ``[]`` (empty list) — if the system is inconsistent.
      - :class:`GroebnerBasis` — for underdetermined (infinite) solution sets.

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

      # Parametric: omit free symbols from variables
      sols = solve([x**2 - y], [x])
      # → [{x: sqrt(y)}, {x: -sqrt(y)}]

   .. note::

      Solutions are symbolic by default. Evaluate numerically with
      :func:`eval_expr` when needed::

         from alkahest import eval_expr
         for sol in sols:
             for var, val in sol.items():
                 print(f"{var} ≈ {eval_expr(val, {}):.6f}")

   Pass ``numeric=True`` to return float values directly instead of
   symbolic ``Expr`` solutions.

GroebnerBasis
-------------

.. class:: GroebnerBasis

   A Gröbner basis for a polynomial ideal.

   A basis is a sequence: ``len(gb)``, ``gb[i]`` and iteration all yield its
   generators as :class:`GbPoly`.

   .. classmethod:: compute(polys: list[Expr], vars: list[Expr], order: str = "lex") -> GroebnerBasis

      Compute a Gröbner basis using the F4 algorithm.

      :param order: Monomial order — ``"lex"`` (default), ``"grlex"``, or
         ``"grevlex"``. Use ``"lex"`` for elimination; ``"grevlex"`` is
         generally fastest. For ``"lex"`` on a zero-dimensional ideal the
         grevlex-then-FGLM strategy is used automatically.

   .. attribute:: order

      The monomial order the generators are reduced under, as a string.

   .. method:: variables() -> list[Expr]

      The variables naming exponent slots 0, 1, … of the generators.

   .. method:: polynomials() -> list[GbPoly]

      The generators. Equivalent to ``list(gb)``.

   .. method:: to_exprs(vars: list[Expr] | None = None) -> list[Expr]

      The generators as expressions, each meaning ``g = 0``. This is the read
      path for elimination results.

   .. method:: reduce(p: GbPoly | Expr) -> GbPoly

      Reduce *p* modulo the ideal and return the remainder. Call
      :meth:`GbPoly.to_expr` on it to read it back; it is zero exactly when
      :meth:`contains` is true.

   .. method:: contains(p: GbPoly | Expr) -> bool

      Test ideal membership.

   .. method:: eliminate(vars: list[Expr]) -> GroebnerBasis

      The elimination ideal ``I ∩ k[remaining vars]``: drops every generator
      whose support mentions one of *vars*. Under a ``"lex"`` basis with the
      eliminated variables ordered **first**, what is left is a Gröbner basis
      for that ideal.

      Useful for implicitization of parametric curves and surfaces.

.. class:: GbPoly

   A sparse multivariate polynomial over ℚ, as used by the Gröbner machinery.
   It stores exponent *vectors*, so reading one back needs the variable list
   its slots refer to — polynomials Alkahest hands out carry that list.

   .. attribute:: is_zero
   .. attribute:: n_vars
   .. attribute:: n_terms

   .. method:: variables() -> list[Expr]

      The variables naming this polynomial's exponent slots, in order.

   .. method:: terms() -> list[tuple[tuple[int, ...], int | Fraction]]

      ``(exponents, coefficient)`` pairs, in ascending exponent-vector order.
      Coefficients are exact.

   .. method:: to_expr(vars: list[Expr] | None = None) -> Expr

      Convert back to an :class:`Expr`. Defaults to :meth:`variables`; raises
      ``ValueError`` if *vars* names fewer variables than the polynomial uses.

.. function:: expr_to_gbpoly(expr: Expr, vars: list[Expr]) -> GbPoly

   Convert a polynomial expression into the :class:`GbPoly` representation —
   the inverse of :meth:`GbPoly.to_expr`. Exponent slot ``i`` refers to
   ``vars[i]``, and the result remembers *vars*, so it can be passed straight
   to :meth:`GroebnerBasis.reduce`, :meth:`GroebnerBasis.contains` or
   ``GroebnerBasis.compute_raw``.

   Raises ``ValueError`` if *expr* is not polynomial in *vars*.
