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

   .. method:: inv() -> Matrix

      Compute the matrix inverse symbolically.
      Raises ``MatrixError`` (``E-MAT-001``) if the matrix is
      symbolically singular (zero determinant).

   .. method:: transpose() -> Matrix

      Return the transpose.

   .. method:: shape() -> tuple[int, int]

      Return ``(nrows, ncols)``.

   .. method:: __getitem__(i, j) -> Expr

      Access entry ``(i, j)``.

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
