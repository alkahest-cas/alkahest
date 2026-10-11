Linear and integer programming
==============================

.. currentmodule:: alkahest

Exact linear and mixed-integer programming over the rationals. Every status is
returned with a certificate that :meth:`LPResult.verify` re-checks exactly; no
step uses floating point. See the user-guide chapter *Exact linear and integer
programming* for the certificate conventions.

.. function:: linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense="min") -> LPResult

   Minimise (``sense="max"``: maximise) ``c @ x`` subject to
   ``A_ub @ x <= b_ub``, ``A_eq @ x == b_eq`` and ``bounds`` — SciPy's calling
   convention. Coefficients are ``int``, ``Fraction``, ``"p/q"`` strings or
   rational ``Expr``; a float raises ``E-LP-002``. ``bounds`` is one
   ``(lower, upper)`` pair for all variables or one per variable, ``None`` /
   ``±inf`` for no bound, default ``(0, None)``.

   Two-phase dense simplex with Bland's rule (cannot cycle); the active budget
   is checked at every pivot.

   :returns: :class:`LPResult`
   :raises LinprogError: ``E-LP-001``, ``E-LP-002``, ``E-LP-006``
   :raises BudgetExceededError: the active budget tripped

   Example::

      r = linprog([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], sense="max")
      r.x, r.fun, r.ineqlin      # ([2, 6], 36, [0, Fraction(3, 2), 1])

.. function:: milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, integrality=None, sense="min", max_nodes=100000) -> MILPResult

   Branch and bound on top of :func:`linprog` (depth-first to a first
   incumbent, then best-bound). ``integrality`` is one ``0``/``1`` flag per
   variable or a single bool; the default makes every variable an integer.

   :returns: :class:`MILPResult` — ``"optimal"`` (closed tree, ``gap == 0``),
      ``"infeasible"``, or ``"unbounded"`` (integer point and integral ray)
   :raises LinprogError: ``E-LP-005`` at ``max_nodes`` (carries ``incumbent``,
      ``incumbent_fun``, ``bound``, ``nodes``), plus the codes of :func:`linprog`
   :raises BudgetExceededError: the budget tripped (same attachments)

   Example::

      r = milp([10, 13, 7, 8], A_ub=[[4, 6, 3, 5]], b_ub=[12], bounds=(0, 1), sense="max")
      r.x, r.fun                 # ([1, 0, 1, 1], 25)

.. function:: linprog_expr(objective, constraints, vars, sense="min", integer=None, max_nodes=100000) -> LPResult | MILPResult

   The same, stated with expressions: an affine ``objective`` and a list of
   ``pool.le`` / ``pool.ge`` / ``pool.pred_eq`` relations (or an ``And`` of
   them). Variables are free unless constrained. ``integer=True`` or a list
   of variables solves a MILP. ``result.solution`` maps variables to values;
   ``result.program()`` shows the rows the relations became.

   :raises LinprogError: ``E-LP-003`` (not affine / unknown symbol),
      ``E-LP-002`` (float or irrational coefficient), ``E-LP-004`` (strict
      inequality or non-relation)

.. class:: LPResult

   ``status`` (``"optimal"`` / ``"infeasible"`` / ``"unbounded"``),
   ``success``, ``x``, ``fun``; dual multipliers ``ineqlin``, ``eqlin``,
   ``lower``, ``upper`` (optimal, SciPy ``marginals`` convention); ``farkas``
   (infeasible: dict of row and bound weights); ``ray`` (unbounded);
   ``solution``; ``program()``; ``verify()``.

.. class:: MILPResult

   ``status``, ``success``, ``x``, ``fun``, ``bound``, ``gap`` (``0`` when
   optimal), ``nodes``, ``ray``, ``solution``, ``program()``, ``verify()``.
