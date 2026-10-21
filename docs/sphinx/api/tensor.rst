Tensor calculus API
===================

.. currentmodule:: alkahest

Metric → Christoffel → Riemann / Ricci / Einstein / Weyl on a coordinate chart,
and an abstract-index layer with Einstein summation (:mod:`alkahest.tensor`).

Every component is computed in an **exact** field: integer polynomials over the
coordinates, the parameters and one generator per transcendental atom
(``sin``/``cos``, ``sinh``/``cosh``, ``exp``, ``log``, radicals, undefined
functions such as ``a(t)`` and their ``Derivative`` nodes), with
``sin² = 1 − cos²``, ``cosh² − sinh² = 1``, ``i² = −1`` and ``(√b)² = b``
applied and every denominator made free of ``sin``, radicals and ``i``. That
normal form is canonical, so a component that vanishes identically is the
literal ``0`` and a non-zero one is never reported as zero — no sampling, no
floats. Only independent components are computed; the rest follow by symmetry.

Conventions (MTW / Carroll / Wald):
``Γ^a_{bc} = ½ g^{ad}(∂_b g_{dc} + ∂_c g_{db} − ∂_d g_{bc})``,
``R^a_{bcd} = ∂_c Γ^a_{db} − ∂_d Γ^a_{cb} + Γ^a_{ce}Γ^e_{db} − Γ^a_{de}Γ^e_{cb}``,
``R_{bd} = R^a_{bad}``, ``G_{ab} = R_{ab} − ½ R g_{ab}``,
``K = R_{abcd}R^{abcd}``; the 2-sphere of radius ``ρ`` has ``R = 2/ρ²``.

Metric
------

.. class:: Metric(g, coords)

   ``g`` is an ``n x n`` symmetric list of lists (or a ``Matrix``) of ``Expr``,
   ``int`` or ``Fraction``; ``coords`` are ``n`` distinct symbols. Raises
   :class:`TensorError` ``E-TENSOR-001`` (shape / coordinates), ``002`` (not
   symmetric), ``003`` (determinant identically zero), ``004`` (a float entry),
   ``005`` (``Piecewise`` or another node the field cannot hold).

   Results are cached. Tensor-valued methods return a :class:`Tensor`:

   * ``as_tensor()`` — ``g_{ab}`` (``"dd"``); ``inverse()`` — ``g^{ab}`` (``"uu"``);
     ``determinant()``.
   * ``christoffel()`` — nested list ``[a][b][c] = Γ^a_{bc}``;
     ``christoffel_first_kind()`` — ``Γ_{abc}``.
   * ``riemann()`` — ``R^a_{bcd}`` (``"uddd"``); ``riemann_lower()`` — ``R_{abcd}``.
   * ``ricci()`` (``"dd"``), ``ricci_scalar()``, ``einstein()`` (``"dd"``),
     ``weyl()`` (``"dddd"``, ``E-TENSOR-008`` for ``n < 3``), ``kretschmann()``.
   * ``is_ricci_flat()`` — ``True`` iff every ``R_{ab}`` is exactly zero (a proof).
   * ``geodesic_equations(velocities=None)`` — ``d²x^a/dτ² = −Γ^a_{bc}ẋ^bẋ^c``,
     one ``Expr`` per coordinate (velocities default to ``<coord>_dot``).
   * ``simplify(expr)``, ``is_zero(expr)``, ``diff(expr, coord)`` — the exact
     field's canonical form, zero test and partial derivative.
   * ``timings()`` — wall time per stage (``convert``, ``inverse``,
     ``christoffel``, ``riemann``, ``ricci``, …).

   A tripped ``Budget`` raises :class:`BudgetExceededError`.

   Example::

      t, r, th, ph, M, a = (p.symbol(s) for s in ("t", "r", "theta", "phi", "M", "a"))
      sigma = r**2 + a**2*ak.cos(th)**2; delta = r**2 - 2*M*r + a**2
      ...                                   # Boyer–Lindquist Kerr components
      ak.Metric(g, [t, r, th, ph]).is_ricci_flat()   # True, in well under a second

Abstract indices
----------------

.. function:: tensor_indices(names) -> tuple[TensorIndex, ...]

   ``a, b = tensor_indices("a b")``. ``+a`` (or ``a``) is an upper index,
   ``-a`` a lower one.

.. class:: Tensor(components, variance, metric=None)

   Components (nested lists) with a variance string (``"u"``/``"d"`` per
   slot). ``T[0, 1]`` is a component; ``T[-a, -b]`` is an
   :class:`IndexedTensor`. A slot indexed in the other variance is raised or
   lowered with the metric; a repeated index (once up, once down) is summed.
   ``nonzero()`` returns ``{indices: component}``.

.. class:: IndexedTensor

   ``+``, ``-``, ``*`` (outer product with Einstein summation), scalar ``*``
   and ``/``; ``symmetrize(*idx)``, ``antisymmetrize(*idx)``, ``is_zero()``,
   ``components``, ``component(*ints)``, ``tensor()``. Index misuse — an index
   three times, twice in the same position, free indices that differ between
   summands — is :class:`TensorError` ``E-TENSOR-009``.

   Example::

      a, b = ak.tensor_indices("a b")
      Ric, g = m.ricci(), m.as_tensor()
      R = Ric[a, -a]                         # Ricci scalar (rank 0)
      G = Ric[-a, -b] - g[-a, -b] * R / 2    # == m.einstein()

Refusals are :class:`TensorError` (``E-TENSOR-001`` … ``E-TENSOR-010``); see
:doc:`errors`.
