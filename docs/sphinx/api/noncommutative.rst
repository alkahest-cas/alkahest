Noncommutative algebra API
==========================

.. currentmodule:: alkahest

Expressions over noncommutative symbols (``pool.symbol(name,
commutative=False)``) read as noncommutative polynomials, with word-rewriting
normal forms. See the *Noncommutative algebra* chapter of the user guide.

.. function:: nc_expand(expr) -> Expr

   Distribute products keeping factor order, unroll non-negative integer
   powers of noncommutative factors, collect equal words.

.. function:: nc_terms(expr) -> list[tuple[Expr, list[Expr]]]

   The ``(coefficient, word)`` terms of :func:`nc_expand`.

.. function:: nc_rewrite(expr, rules, *, max_steps=200000) -> Expr

   Rewrite to a normal form with literal word rules from :func:`make_rule`,
   matching inside longer words. Raises :class:`NcError` ``E-NC-001`` when no
   normal form is reached within ``max_steps``, ``E-NC-003`` for a left side
   that is not a single word.

.. function:: commutator(a, b, rules=None) -> Expr
.. function:: anticommutator(a, b, rules=None) -> Expr

   ``a*b ∓ b*a`` expanded; with ``rules``, rewritten by :func:`nc_rewrite`.

.. function:: normal_order(expr, bosons=None, fermions=None) -> Expr

   Normal-order with ``[a, a†] = 1`` (bosons) and ``{c, c†} = 1``, ``c² = 0``
   (fermions). ``bosons`` / ``fermions`` are ``(annihilator, creator)`` pairs
   or lists of them. Raises ``E-NC-004`` for a malformed mode list.

.. function:: vacuum_expectation(expr, bosons=None, fermions=None) -> Expr

   ``⟨0|expr|0⟩``, the scalar part of :func:`normal_order`. Raises
   ``E-NC-007`` for a factor that is not a listed ladder operator.

.. function:: gamma_trace(slots, pool=None, *, dim=None, indices=None) -> Expr

   Four-dimensional Dirac trace, metric ``diag(+1,−1,−1,−1)``,
   ``γ⁵ = iγ⁰γ¹γ²γ³``. Slots are ``0``–``3``, ``5`` or a commutative ``Expr``
   (slashed vector). The result is a polynomial in ``dot(p, q)`` and
   ``levi_civita(a, b, c, d)`` (``ε^{0123} = +1``). Raises ``E-NC-005`` for a
   bad slot and ``E-NC-006`` past 14 slots.

   With ``dim`` (an integer or a symbolic ``d``) the trace is taken in ``dim``
   dimensions: slots are commutative labels, ``indices`` lists the labels that
   are Lorentz indices, an index occurring twice is summed with
   ``g^μ_μ = dim``, and ``tr 1 = 4``. ``γ⁵`` and explicit components are
   refused (``E-NC-005``).

.. function:: lorentz_contract(expr, indices, dim) -> Expr

   Contract dummy indices of a polynomial in ``dot(a, b)`` (``dot(μ, μ) → d``,
   ``dot(μ, a) dot(μ, b) → dot(a, b)``), e.g. a product of traces.

.. function:: gamma_contract(slots, indices, dim) -> list[tuple[Expr, list[Expr]]]

   Reduce a gamma string with contracted index pairs to terms
   ``(coefficient, labels)`` (``γ^μγ^νγ_μ = (2 − d)γ^ν``).

.. function:: lie_algebra(generators, structure_constants, *, check_jacobi=True) -> LieAlgebra
.. function:: lie_su2(pool, generators=None, *, prefix="J") -> LieAlgebra
.. function:: lie_su3(pool, generators=None, *, prefix="T") -> LieAlgebra
.. function:: su3_f(pool) -> dict
.. function:: su3_d(pool) -> dict

   Lie algebras with ``[T_a, T_b] = i f_{abc} T_c`` (1-based tables). The
   :class:`LieAlgebra` has ``generators``, ``rules``, ``commutator``,
   ``anticommutator``, ``normal_order``, ``casimir(inverse_metric=None)`` and
   ``verify_jacobi``. Raises ``E-NC-008`` for a malformed or non-Jacobi table.

``make_rule(lhs, rhs, sequence=[X, ...])`` declares noncommutative symbols of
``lhs`` to be sequence wildcards matching a contiguous, possibly empty run of
factors (leftmost start, shortest run), usable through ``simplify_with``;
``E-NC-009`` for a malformed pattern.

``simplify_pauli`` reduces any word over ``sx``, ``sy``, ``sz`` with the full
Pauli algebra; ``make_rule`` / ``simplify_with`` match product patterns inside
longer products (contiguous runs for noncommutative products, sub-multisets for
commutative ones).

Refusals raise :exc:`NcError` (``E-NC-001`` … ``E-NC-009``); see :doc:`errors`.
