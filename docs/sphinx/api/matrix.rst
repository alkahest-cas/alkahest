Matrix API
==========

.. currentmodule:: alkahest

Symbolic matrices and linear algebra operations.

Matrix
------

.. class:: Matrix

   A symbolic matrix with :class:`Expr` entries.

   Constructed from a list of rows::

      pool = ExprPool()
      x = pool.symbol("x")
      y = pool.symbol("y")

      M = Matrix([[x, pool.integer(1)],
                  [y, x**2]])

   **Arithmetic**

   ``+``, ``-``, ``*`` (matrix multiply), ``**`` (integer powers) are
   supported::

      A = Matrix([[x, pool.integer(0)], [pool.integer(0), y]])
      B = Matrix([[pool.integer(1), x], [y, pool.integer(1)]])
      C = A * B   # matrix multiply

   .. method:: det() -> Expr

      Compute the determinant symbolically.

   .. method:: inverse() -> Matrix

      Compute the matrix inverse symbolically. Three outcomes, kept distinct on
      purpose:

      - ``E-MAT-002`` — the matrix is not square.
      - ``E-MAT-003`` — the determinant is **proven** zero (singular).
      - ``E-MAT-004`` — the determinant's vanishing could be decided **neither
        way**. Refusing rather than returning an inverse that silently assumes
        ``det ≠ 0``.

   .. method:: rank() -> int
   .. method:: rref() -> list[list[Expr]]
   .. method:: nullspace() -> list[Matrix]
   .. method:: eigenvects() -> list
   .. method:: jordan_form() -> Matrix

      All five run the same elimination, which uses a **three-valued** zero
      test. An entry whose vanishing can be proven neither zero nor non-zero
      raises ``E-LINALG-010`` (as :exc:`LinearAlgebraError`, or
      :exc:`EigenError` from ``eigenvects``) rather than picking a branch.
      Substituting concrete values for the parameters is the remedy.

   .. method:: eigenvals() -> dict[Expr, int]

      Eigenvalue → algebraic multiplicity. For an irreducible cubic with three
      real roots the Cardano form is returned, whose cube roots are meant under
      the **real** branch; :func:`eval_expr` refuses such a value with
      ``E-EVAL-009``. Do not export it to a principal-branch evaluator — see
      `Interoperability <../interop.html>`_.

   .. method:: transpose() -> Matrix

      Return the transpose.

   .. method:: shape() -> tuple[int, int]

      Return ``(nrows, ncols)``. ``rows`` and ``cols`` are also available as
      attributes.

   .. method:: get(i, j) -> Expr

      Access entry ``(i, j)``. ``Matrix`` is **not** subscriptable —
      ``M[i, j]`` raises ``TypeError``.

.. function:: jacobian(exprs: list[Expr], vars: list[Expr]) -> Matrix

   Compute the Jacobian matrix of a vector-valued function.

   :param exprs: The component expressions ``[f0, f1, ..., fm]``.
   :param vars: The variables ``[x0, x1, ..., xn]``.
   :returns: The ``m × n`` Jacobian matrix with entry ``(i, j) = ∂fᵢ/∂xⱼ``.

   Example::

      pool = ExprPool()
      x = pool.symbol("x")
      y = pool.symbol("y")

      J = jacobian([x**2 + y, x * sin(y)], [x, y])
      # J = [[2*x, 1],
      #      [sin(y), x*cos(y)]]
