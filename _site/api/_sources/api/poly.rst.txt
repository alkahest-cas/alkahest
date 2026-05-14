Polynomial API
==============

.. currentmodule:: alkahest

Explicit polynomial representation types backed by FLINT.

UniPoly
-------

.. class:: UniPoly

   Dense univariate polynomial with exact integer or rational coefficients,
   backed by FLINT's ``fmpq_poly_t``.

   .. classmethod:: from_symbolic(expr: Expr, var: Expr) -> UniPoly

      Convert a symbolic expression to a univariate polynomial in *var*.

      :raises ConversionError: If *expr* contains non-polynomial terms.

      Example::

         p = UniPoly.from_symbolic(x**3 + pool.integer(-2) * x + pool.integer(1), x)

   .. method:: degree() -> int

      Return the degree of the polynomial.

   .. method:: coefficients() -> list

      Return the coefficient list, constant term first.

   .. method:: leading_coeff() -> object

      Return the leading coefficient.

   .. method:: to_symbolic(pool: ExprPool) -> Expr

      Convert back to a symbolic expression.

   **Arithmetic**

   All arithmetic operators are supported: ``+``, ``-``, ``*``, ``**``
   (integer exponent), ``//`` (exact quotient), ``%`` (remainder)::

      p = UniPoly.from_symbolic(x**2 + pool.integer(-1), x)
      q = UniPoly.from_symbolic(x + pool.integer(-1), x)
      print(p * q)   # x^3 - x^2 - x + 1
      print(p // q)  # x + 1
      print(p % q)   # 0

   .. method:: gcd(other: UniPoly) -> UniPoly

      Compute the GCD via FLINT's subresultant PRS.

   .. method:: __pow__(n: int) -> UniPoly

      Raise to a non-negative integer power.

MultiPoly
---------

.. class:: MultiPoly

   Sparse multivariate polynomial over ℤ. Terms are stored as a map from
   exponent vectors to coefficients.

   .. classmethod:: from_symbolic(expr: Expr, vars: list[Expr]) -> MultiPoly

      Convert a symbolic expression to a multivariate polynomial.

      :param vars: Variables, in the order used for exponent vectors.
      :raises ConversionError: If *expr* is not polynomial.

   .. method:: total_degree() -> int

      Return the total degree (maximum sum of exponents across all terms).

   .. method:: integer_content() -> int

      Return the GCD of all coefficients.

   .. method:: to_symbolic(pool: ExprPool) -> Expr

      Convert back to a symbolic expression.

   **Arithmetic**

   ``+``, ``-``, ``*`` are supported between ``MultiPoly`` objects with
   the same variable list.

   .. method:: gcd(other: MultiPoly) -> MultiPoly

      Compute the GCD via multivariate FLINT algorithms.

RationalFunction
----------------

.. class:: RationalFunction

   A quotient of two ``MultiPoly`` objects, automatically reduced to lowest
   terms by their GCD.

   .. classmethod:: from_symbolic(numer: Expr, denom: Expr, vars: list[Expr]) -> RationalFunction

      Construct a rational function and normalize by GCD.

      Example::

         # (x^2 - 1) / (x - 1) → normalized to x + 1
         rf = RationalFunction.from_symbolic(
             x**2 + pool.integer(-1),
             x + pool.integer(-1),
             [x]
         )
         print(rf)  # x + 1

   **Arithmetic**

   ``+``, ``-``, ``*``, ``/`` are all supported.
   Results are automatically normalized::

      rf_x = RationalFunction.from_symbolic(x, pool.integer(1), [x])
      rf_inv = RationalFunction.from_symbolic(pool.integer(1), x, [x])
      print(rf_x + rf_inv)   # (x^2 + 1) / x

   .. method:: numerator() -> MultiPoly

      Return the numerator after normalization.

   .. method:: denominator() -> MultiPoly

      Return the denominator after normalization.

Polynomial utilities
--------------------

.. function:: horner(expr: Expr, var: Expr) -> Expr
   :no-index:

   Rewrite a polynomial expression in Horner's form::

      # x^3 + 2x^2 + 3x + 4 → x*(x*(x + 2) + 3) + 4
      h = horner(x**3 + pool.integer(2)*x**2 + pool.integer(3)*x + pool.integer(4), x)

   Horner's form requires fewer multiplications and is numerically
   better conditioned than the expanded form.

.. function:: collect_like_terms(expr: Expr) -> Expr

   Collect monomials with the same base::

      result = collect_like_terms(pool.integer(2) * x + pool.integer(3) * x)
      # result == 5*x

.. function:: poly_normal(expr: Expr, vars: list[Expr]) -> Expr

   Normalize to canonical polynomial form.
   Raises ``ConversionError`` if *expr* is not polynomial in *vars*.
