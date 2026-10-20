Orthogonal polynomials and angular momentum
===========================================

.. currentmodule:: alkahest

Classical orthogonal polynomials (DLMF chapter 18), the complex spherical
harmonics, and exact Wigner 3j / 6j symbols and Clebsch–Gordan coefficients.
See the *Orthogonal polynomials and angular momentum* chapter of the user guide
for conventions and examples.

Polynomials
-----------

With an integer degree (``int`` or integer ``Expr``) and ``expand=True`` (the
default) each returns the explicit polynomial with exact rational coefficients;
a symbolic or non-integer degree, or ``expand=False``, returns the unevaluated
node, which differentiates in ``x``, evaluates numerically (``f64`` and
rigorous balls) and simplifies at special values. Family parameters may be
symbolic. An expansion past degree 1000 (64 with a symbolic parameter) raises
:exc:`SpecialFunctionError` ``E-SPECFN-004``.

.. function:: legendre_p(n, x, *, expand=True)

   Legendre ``P_n(x)``; ``P_{-n-1} = P_n``.

.. function:: assoc_legendre_p(n, m, x, *, expand=True)

   Associated Legendre (Ferrers) ``P_n^m(x)`` **with the Condon–Shortley
   phase**: ``(-1)^m (1-x^2)^{m/2} d^m P_n/dx^m``;
   ``P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m``; ``|m| > n`` gives ``0``.

.. function:: hermite_h(n, x, *, expand=True)

   Physicists' Hermite ``H_n(x)`` (weight ``exp(-x^2)``).

.. function:: hermite_he(n, x, *, expand=True)

   Probabilists' Hermite ``He_n(x)`` (monic, weight ``exp(-x^2/2)``).

.. function:: laguerre_l(n, x, *, expand=True)

   Laguerre ``L_n(x)``.

.. function:: assoc_laguerre_l(n, alpha, x, *, expand=True)

   Generalised Laguerre ``L_n^{(alpha)}(x)``.

.. function:: chebyshev_t(n, x, *, expand=True)

   Chebyshev ``T_n(x)``; ``T_{-n} = T_n``.

.. function:: chebyshev_u(n, x, *, expand=True)

   Chebyshev ``U_n(x)``; ``U_{-1} = 0``, ``U_{-n} = -U_{n-2}``.

.. function:: gegenbauer_c(n, a, x, *, expand=True)

   Gegenbauer ``C_n^{(a)}(x)``.

.. function:: jacobi_p(n, a, b, x, *, expand=True)

   Jacobi ``P_n^{(a,b)}(x)``.

.. function:: expand_orthopoly(expr)

   Expand every node whose degree (and Legendre order) is now an integer
   literal — e.g. after :func:`subs`.

Spherical harmonics
-------------------

.. function:: spherical_harmonic_y(l, m, theta, phi)

   ``Y_l^m(theta, phi) = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) P_l^m(cos theta)
   exp(i m phi)``, Condon–Shortley phase, orthonormal on the sphere (SymPy's
   ``Ynm``). Integer ``l >= 0`` and ``m`` only (``E-SPECFN-003``);
   ``(1-cos^2 theta)^{|m|/2}`` is written ``sin(theta)^|m|`` (exact for
   ``0 <= theta <= pi``).

.. function:: real_spherical_harmonic_y(l, m, theta, phi)

   The real (tesseral) harmonic ``Y_{l,m}``: ``sqrt(2) N P_l^{|m|}(cos theta)
   cos(m phi)`` for ``m > 0``, ``sqrt(2) N P_l^{|m|}(cos theta) sin(|m| phi)``
   for ``m < 0`` and ``Y_l^0`` for ``m = 0``, with ``N = sqrt((2l+1)/(4 pi)
   (l-|m|)!/(l+|m|)!)`` and ``P_l^{|m|}`` *without* the Condon–Shortley phase
   (Wikipedia's convention, ``Y_{l,m} = sqrt(2) (-1)^m Re/Im Y_l^{|m|}``), so
   ``Y_{1,1} = sqrt(3/(4 pi)) x/r``. Real and orthonormal on the sphere.
   Same error codes as :func:`spherical_harmonic_y`.

Coupling coefficients
---------------------

Arguments are integers or half-integers (``int``, ``Fraction``, a float that is
exactly ``k/2``, or a numeric ``Expr``). Results are exact ``Expr`` values
``q`` or ``q*r**(1/2)`` (``q`` rational, ``r`` square-free); selection-rule
violations give ``0``. The result lives in the pool of an ``Expr`` argument,
else ``pool``, else a module-wide default pool.

.. function:: wigner_3j(j1, j2, j3, m1, m2, m3, *, pool=None)

.. function:: wigner_6j(j1, j2, j3, j4, j5, j6, *, pool=None)

.. function:: clebsch_gordan(j1, j2, j3, m1, m2, m3, *, pool=None)

   ``<j1 m1; j2 m2 | j3 m3>`` — SymPy's argument order.

.. exception:: SpecialFunctionError
   :no-index:

   ``E-SPECFN-001`` … ``E-SPECFN-005``; see :doc:`errors`.
