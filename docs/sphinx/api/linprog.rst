Linear and integer programming
==============================

.. currentmodule:: alkahest

Exact linear and mixed-integer programming over the rationals. Every status is
returned with a certificate that :meth:`LPResult.verify` re-checks exactly; no
step uses floating point. See the user-guide chapter *Exact linear and integer
programming* for the certificate conventions.

.. function:: linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense="min", method="auto", basis=None) -> LPResult

   Minimise (``sense="max"``: maximise) ``c @ x`` subject to
   ``A_ub @ x <= b_ub``, ``A_eq @ x == b_eq`` and ``bounds`` — SciPy's calling
   convention. Coefficients are ``int``, ``Fraction``, ``"p/q"`` strings or
   rational ``Expr``; a float raises ``E-LP-002``. ``bounds`` is one
   ``(lower, upper)`` pair for all variables or one per variable, ``None`` /
   ``±inf`` for no bound, default ``(0, None)``.

   ``method``: ``"dense"`` (two-phase tableau, Bland's rule), ``"revised"``
   (sparse revised simplex with an exact product-form basis factorisation,
   Dantzig pricing with a Bland fallback — neither can cycle) or ``"auto"``
   (dense for at most 12 rows). ``basis``: a previous result's
   :attr:`LPResult.basis` (or the result) to warm-start from — primal
   feasible goes to phase 2, dual feasible runs the dual simplex, anything
   else cold-starts. The active budget is checked at every pivot.

   :returns: :class:`LPResult`
   :raises LinprogError: ``E-LP-001``, ``E-LP-002``, ``E-LP-006``
   :raises BudgetExceededError: the active budget tripped

   Example::

      r = linprog([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], sense="max")
      r.x, r.fun, r.ineqlin      # ([2, 6], 36, [0, Fraction(3, 2), 1])

.. function:: milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, integrality=None, sense="min", max_nodes=100000, cut_rounds=8, max_cuts=200, warm_start=True) -> MILPResult

   Cut-and-branch on top of :func:`linprog`: up to ``cut_rounds`` rounds of
   Gomory mixed-integer cuts (at most ``max_cuts``) at the root, each valid
   for every integer-feasible point and listed in :attr:`MILPResult.cuts`,
   then branch and bound (depth-first to a first incumbent, then best-bound)
   with node LPs warm-started from the parent's basis. ``integrality`` is one ``0``/``1`` flag per
   variable or a single bool; the default makes every variable an integer.

   :returns: :class:`MILPResult` — ``"optimal"`` (closed tree, ``gap == 0``),
      ``"infeasible"``, or ``"unbounded"`` (integer point and integral ray)
   :raises LinprogError: ``E-LP-005`` at ``max_nodes`` (carries ``incumbent``,
      ``incumbent_fun``, ``bound``, ``nodes``), plus the codes of :func:`linprog`
   :raises BudgetExceededError: the budget tripped (same attachments)

   Example::

      r = milp([10, 13, 7, 8], A_ub=[[4, 6, 3, 5]], b_ub=[12], bounds=(0, 1), sense="max")
      r.x, r.fun                 # ([1, 0, 1, 1], 25)

.. function:: linprog_expr(objective, constraints, vars, sense="min", integer=None, max_nodes=100000, parameter=None, parameter_range=None) -> LPResult | MILPResult | ParametricLPResult

   The same, stated with expressions: an affine ``objective`` and a list of
   ``pool.le`` / ``pool.ge`` / ``pool.pred_eq`` relations (or an ``And`` of
   them). Variables are free unless constrained. ``integer=True`` or a list
   of variables solves a MILP. ``result.solution`` maps variables to values;
   ``result.program()`` shows the rows the relations became.

   With ``parameter=t`` the objective's coefficients **or** the constraints'
   constant terms may be affine in ``t``; the result is a
   :class:`ParametricLPResult` over ``parameter_range``.

   :raises LinprogError: ``E-LP-003`` (not affine / unknown symbol),
      ``E-LP-002`` (float or irrational coefficient), ``E-LP-004`` (strict
      inequality or non-relation), ``E-LP-007`` (unsupported parametric form:
      ``t`` in the constraint matrix, non-linear, in both objective and
      right-hand side, or with ``integer``)

.. function:: linprog_parametric(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense="min", *, c_theta=None, b_ub_theta=None, b_eq_theta=None, theta_range=None) -> ParametricLPResult

   Parametric LP: the objective ``(c + θ·c_theta) @ x`` **or** the right-hand
   sides ``b + θ·b_theta`` move with ``θ ∈ theta_range``. Returns the exact
   breakpoints and every critical region with its optimal basis and affine
   value, point and multipliers, each region's certificate verified exactly
   for all ``θ`` in it.

   :raises LinprogError: ``E-LP-007`` (both ``c_theta`` and a right-hand-side
      direction), ``E-LP-001`` (wrong length, empty range), and the codes of
      :func:`linprog`

   Example::

      r = linprog_parametric([1, 1], A_ub=[[1, 2], [3, 1]], b_ub=[4, 6],
                             b_ub_theta=[1, 0], sense="max")
      r.breakpoints, r.value_at(0)   # ([-4, -2, 8], Fraction(14, 5))

.. class:: LPResult

   ``status`` (``"optimal"`` / ``"infeasible"`` / ``"unbounded"``),
   ``success``, ``x``, ``fun``; dual multipliers ``ineqlin``, ``eqlin``,
   ``lower``, ``upper`` (optimal, SciPy ``marginals`` convention); ``farkas``
   (infeasible: dict of row and bound weights); ``ray`` (unbounded);
   ``solution``; ``program()``; ``verify()``; ``basis`` (``(kind, index)``
   pairs, for a warm start), ``iterations``, ``method``, ``warm_started``.

.. class:: MILPResult

   ``status``, ``success``, ``x``, ``fun``, ``bound``, ``gap`` (``0`` when
   optimal), ``nodes``, ``ray``, ``cuts`` (root Gomory cuts as
   ``(row, rhs)``), ``solution``, ``program()``, ``verify()``.

.. class:: ParametricLPResult

   ``kind`` (``"rhs"`` / ``"objective"``), ``breakpoints``, ``regions`` (dicts
   with ``theta``, ``status``, affine ``value``, ``x``, ``ineqlin``,
   ``eqlin``, ``lower``, ``upper`` as ``(a, b)`` = ``a + θ·b``, and
   ``basis``), ``value_at(θ)``, ``status_at(θ)``, ``x_at(θ)``,
   ``solution_at(θ)``, ``program()``, ``verify()``.
