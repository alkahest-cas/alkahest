Bessel, Airy, zeta and polylogarithm functions
===============================================

.. currentmodule:: alkahest

Cylinder and spherical Bessel functions (DLMF chapter 10), Airy functions
(chapter 9), the Riemann and Hurwitz zeta functions (chapter 25) and the
polylogarithm ``Li_s`` (§25.12). Conventions are DLMF's, which are also SymPy's,
mpmath's and Arb's. See the *Bessel, Airy, zeta and polylogarithm functions*
chapter of the user guide for examples.

Every function is a registered primitive: the node differentiates in its
variable, evaluates in ``f64`` (:func:`eval_expr`), in rigorous balls
(:func:`interval_eval`) and at complex points (``evaluate(e, {x: 1+2j})``) —
all through FLINT's Arb layer — and simplifies at special values. A
derivative in an order (``nu``, ``n``, ``s``), and any derivative of
:func:`zeta`, is refused with ``DiffError`` ``E-DIFF-001`` rather than dropped.
A value that is not real (``Y_nu``, ``K_nu`` at ``x < 0``, ``Li_s(x)`` for
``x > 1``) makes ``eval_expr`` / ``interval_eval`` decline; the complex
evaluator has it.

Bessel functions
----------------

A half-integer order (``Fraction(1, 2)``, ``pool.rational(3, 2)``) with
``expand=True`` (the default) gives the elementary form; otherwise the node,
canonicalised (``bessel_j(0, x)`` is :func:`bessel_j0`, ``J_{-n} = (-1)^n J_n``,
``K_{-nu} = K_nu``, ``J_nu(0)``/``I_nu(0)``). An expansion past order 200
raises :exc:`SpecialFunctionError` ``E-SPECFN-004``.

.. function:: bessel_j(nu, x, *, expand=True)

   ``J_nu(x)``; ``bessel_j(1/2, x) = sqrt(2)*sin(x)/(sqrt(pi)*sqrt(x))``;
   ``d/dx J_nu = (J_{nu-1} - J_{nu+1})/2``.

.. function:: bessel_y(nu, x, *, expand=True)

   ``Y_nu(x)``; ``d/dx Y_nu = (Y_{nu-1} - Y_{nu+1})/2``.

.. function:: bessel_i(nu, x, *, expand=True)

   ``I_nu(x)``; half-integer orders in ``sinh``/``cosh``;
   ``d/dx I_nu = (I_{nu-1} + I_{nu+1})/2``.

.. function:: bessel_k(nu, x, *, expand=True)

   ``K_nu(x)``; ``K_{1/2}(x) = sqrt(pi/(2x)) exp(-x)``;
   ``d/dx K_nu = -(K_{nu-1} + K_{nu+1})/2``.

.. function:: spherical_jn(n, x, *, expand=True)

   ``j_n(x) = sqrt(pi/(2x)) J_{n+1/2}(x)``; an integer ``n`` expands
   (``j_1 = sin(x)/x**2 - cos(x)/x``). A non-integer literal order raises
   ``E-SPECFN-007``.

.. function:: spherical_yn(n, x, *, expand=True)

   ``y_n(x) = sqrt(pi/(2x)) Y_{n+1/2}(x)``.

:func:`bessel_j0` and :func:`bessel_j1` remain, and are the spellings
``bessel_j(0, x)`` and ``bessel_j(1, x)`` produce. ``integrate`` knows
``∫x^nu F_{nu-1} = ±x^nu F_nu`` and ``∫x^{-nu} F_{nu+1} = ±x^{-nu} F_nu``
(``∫x J_0 = x J_1``, ``∫J_1 = -J_0``).

Airy functions
--------------

.. function:: airy_ai(x)
.. function:: airy_bi(x)
.. function:: airy_ai_prime(x)
.. function:: airy_bi_prime(x)

   ``Ai``, ``Bi`` and their derivatives; ``Ai'' = x Ai``. Exact at 0:
   ``Ai(0) = 3**(-2/3)/gamma(2/3)``, ``Ai'(0) = -3**(-1/3)/gamma(1/3)``,
   ``Bi(0) = 3**(-1/6)/gamma(2/3)``, ``Bi'(0) = 3**(1/6)/gamma(1/3)``.

Zeta functions and polylogarithms
---------------------------------

.. function:: zeta(s)

   Riemann ``zeta(s)``. Exact at even positive integers
   (``(-1)^{n+1} B_{2n} (2 pi)^{2n} / (2 (2n)!)``) and at non-positive integers
   (``zeta(-n) = (-1)^n B_{n+1}/(n+1)``, so ``zeta(0) = -1/2``); ``zeta(3)``
   stays a node. ``E-SPECFN-006`` at the pole ``s = 1``. No derivative rule.
   :func:`sum_definite` names odd zeta values: ``Σ 1/k³ = zeta(3)``.

.. function:: hurwitz_zeta(s, a)

   ``zeta(s, a) = Σ_{k≥0} (k + a)^{-s}``; ``hurwitz_zeta(s, 1) = zeta(s)``,
   ``hurwitz_zeta(s, 1/2) = (2^s - 1) zeta(s)``,
   ``hurwitz_zeta(-n, a) = -B_{n+1}(a)/(n+1)``;
   ``d/da = -s hurwitz_zeta(s + 1, a)``. ``E-SPECFN-006`` at ``s = 1``.

.. function:: polylog(s, z)

   ``Li_s(z)``, principal branch, cut ``[1, oo)``. ``polylog(2, z)`` is
   :func:`dilog`; ``Li_1 = -log(1 - z)``, ``Li_0 = z/(1 - z)``, ``Li_{-n}`` the
   Eulerian rational function; ``Li_s(1) = zeta(s)`` (``s > 1``),
   ``Li_s(-1) = -(1 - 2^{1-s}) zeta(s)``; ``d/dz Li_s = Li_{s-1}(z)/z``.
   ``E-SPECFN-006`` at ``z = 1`` for ``s <= 1``.

.. function:: expand_special(expr)

   Expand every Bessel node whose order has become a half-integer literal (and
   spherical nodes with an integer order) into elementary functions and
   re-canonicalise ``zeta``/``hurwitz_zeta``/``polylog`` nodes — SymPy's
   ``expand_func`` for these families, e.g. after :func:`subs`.

Gamma, digamma and polygamma; named constants
---------------------------------------------

The Euler–Mascheroni constant, Catalan's constant and Glaisher's constant are
exact symbols (``EulerGamma``, ``Catalan``, ``Glaisher``; LaTeX ``\gamma``,
``G``, ``A``). They print, parse and substitute like ``pi``; every numeric
evaluator (:func:`eval_expr`, :func:`interval_eval` at any precision, the
JITs, :func:`emit_c_expr`) resolves them, and no symbolic operation folds them
into a float.

.. function:: euler_gamma(pool)

   ``EulerGamma`` (``0.5772…``) in ``pool``.

.. function:: catalan(pool)

   ``Catalan`` (``0.9159…``) in ``pool``; ``trigamma(1/4) = pi**2 + 8*Catalan``.

.. function:: glaisher(pool)

   ``Glaisher`` (``1.2824…``) in ``pool``; its ball needs Arb.

.. function:: polygamma(n, x)

   ``psi^(n)(x)`` (DLMF 5.15). ``polygamma(0, x)`` is :func:`digamma`,
   ``polygamma(1, x)`` is :func:`trigamma`. Exact values (also applied by
   :func:`simplify` to ``gamma``/``digamma``/``trigamma`` nodes):
   ``gamma(n) = (n-1)!``, ``gamma(m + 1/2)`` a rational multiple of
   ``sqrt(pi)``; ``digamma`` at every integer shift of ``1, 1/2, 1/3, 2/3,
   1/4, 3/4, 1/6, 5/6`` (``digamma(1) = -EulerGamma``,
   ``digamma(1/2) = -EulerGamma - 2*log(2)``); ``polygamma(n, ·)`` at integer
   and half-integer points via ``(-1)^{n+1} n! zeta(n+1, a)``
   (``trigamma(1) = pi**2/6``, ``polygamma(2, 1) = -2*zeta(3)``).
   ``d/dx polygamma(n, x) = polygamma(n + 1, x)``; ``d/dn`` is refused
   (``E-DIFF-001``). ``E-SPECFN-008`` for an order that is not an integer
   ``>= 0``, ``E-SPECFN-006`` at a pole.

:func:`series` expands ``gamma``, ``digamma``, ``trigamma`` and ``polygamma``
at their poles ``0, -1, -2, …`` with exact coefficients:
``series(gamma(eps), eps, 0, 2)`` is
``1/eps - EulerGamma + (EulerGamma**2/2 + pi**2/12)*eps + O(eps**2)``, and
products such as ``m**(-2*eps)*gamma(eps)`` (dimensional regularisation)
expand the same way.

Generalized hypergeometric function
-----------------------------------

.. function:: hyper(a, b, z)

   ``pFq(a; b; z)`` (DLMF 16.2; SymPy's ``hyper``). The node prints as
   ``hyper(p, q, a1, …, ap, b1, …, bq, z)`` and parses back; common
   parameters cancel, an upper ``0`` or ``z = 0`` gives ``1``.
   ``d/dz = prod(a)/prod(b) * hyper(a + 1, b + 1, z)``; a derivative in a
   parameter is refused. ``f64`` and ball values from Arb (real values only;
   principal branch for ``p = q + 1``). ``E-SPECFN-006`` for a lower
   parameter that is a non-positive integer (unless the series terminates
   first).

.. function:: expand_hyper(expr)

   SymPy's ``hyperexpand``: terminating series become polynomials (degree
   ``<= 200``, else ``E-SPECFN-004``), and ``0F0 = exp``,
   ``1F0(a;;z) = (1 - z)**(-a)``, ``2F1(1, 1; 2; z) = -log(1 - z)/z``,
   ``2F1(1/2, 1; 3/2; z) = atanh(sqrt(z))/sqrt(z)``,
   ``2F1(1/2, 1/2; 3/2; z) = asin(sqrt(z))/sqrt(z)``, Gauss's
   ``2F1(a, b; c; 1)`` when ``c - a - b`` is a positive literal,
   ``1F1(1; 2; z) = (exp(z) - 1)/z``,
   ``1F1(1/2; 3/2; -w) = sqrt(pi)*erf(sqrt(w))/(2*sqrt(w))``, and ``0F1`` as
   ``cosh``/``sinh`` or Bessel ``J``/``I``. Other nodes are kept.
