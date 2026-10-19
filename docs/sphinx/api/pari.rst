PARI/GP interop API
===================

.. currentmodule:: alkahest.pari

Optional exact-object interop with PARI/GP (:mod:`alkahest.pari`). Pure
Python; needs a separately installed ``cypari`` or ``cypari2`` because PARI is
GPL and is never bundled or imported by ``import alkahest``. Every result is
re-verified with Alkahest's own exact arithmetic where that is cheap; see the
*PARI/GP interop* chapter of the user guide.

.. function:: available() -> bool
.. function:: backend() -> str | None
.. function:: pari_version() -> str | None

.. function:: number_field_invariants(field_or_poly, *, primes=(), class_group=True, galois=True, certify="auto") -> NumberFieldData

   Ring of integers, discriminant, class group, units, prime-ideal
   factorisation and Galois group of a monic number field. ``certification``
   is ``"certified"`` or ``"grh_conditional"``.

.. function:: elliptic_curve_invariants(ainvs, *, ap_bound=100, rank=True, l_value=True) -> EllipticCurveData

   Conductor, ``a_p``, torsion (checked in ``Pic⁰``), root number, analytic
   rank, ``L(E,1)`` / ``L'(E,1)`` as an ``ArbBall`` and 2-descent rank bounds
   with Mordell–Weil points.

.. function:: modular_forms(N, k, *, nmax=20) -> ModularFormsData

   q-expansion basis of ``S_k(Γ0(N))`` and the new subspace's Hecke
   eigenform orbits with exact (number-field) eigenvalues.

.. function:: ramanujan_tau(n) -> int

.. class:: PrimeIdeal
.. class:: NumberFieldData
.. class:: EllipticCurveData
.. class:: EigenformOrbit
.. class:: ModularFormsData

Errors: :class:`PariError` (``E-PARI-002/003/004``) and
:class:`PariUnavailable` (``E-PARI-001``).
