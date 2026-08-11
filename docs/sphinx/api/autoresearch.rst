Autoresearch modules API
========================

.. currentmodule:: alkahest

Three submodules aimed at unattended math-search loops. All are reachable as
``alkahest.ansatz`` / ``alkahest.crosscheck`` / ``alkahest.smt`` without a
separate import, and all are listed in ``alkahest.__all__``.

Conceptual guides: `Ansatz families <../ansatz.html>`_,
`Cross-CAS testing <../crosscheck.html>`_, `SMT bridge <../smt.html>`_.

.. note::

   Each module has one error code that means *undecided by this route*, not
   *false*: ``E-ANSATZ-003`` (no member of this family fits), ``E-XCHECK-002``
   (no oracle installed), ``E-SMT-003`` (algebraic witness that cannot be
   lifted exactly). A search loop that records any of them as a negative result
   closes a branch it never explored.

alkahest.ansatz
---------------

Parametric families with named unknown coefficients, plus the fitting step.

.. function:: ansatz.polynomial(pool, vars, degree, *, name="c", min_degree=0, max_terms=256, reserved=()) -> Ansatz
.. function:: ansatz.rational(pool, vars, num_degree, den_degree, *, name="a", den_name="b", monic_denominator=True, max_terms=256, reserved=()) -> Ansatz
.. function:: ansatz.exponential_polynomial(pool, var, rates, *, degree=0, name="c", max_terms=256, reserved=()) -> Ansatz
.. function:: ansatz.linear_combination(pool, basis, *, vars=None, name="c", max_terms=256, reserved=()) -> Ansatz
.. function:: ansatz.quadratic_form(pool, vars, *, name="q", max_terms=256, reserved=()) -> Ansatz

   Family constructors. Each returns an ``Ansatz``, which is an object rather
   than a bare ``Expr`` because a bare expression loses the distinction between
   an *unknown coefficient* and an *independent variable*.

.. function:: ansatz.fit(ansatz, residual, *, certify="residual", seed=None, oversample=None, max_points=None, degree_bound=None, tolerance=1e-08, samples=5) -> AnsatzSolution

   Solve for the coefficients that make ``residual`` vanish. ``certify`` is one
   of ``"residual"``, ``"exact"``, ``"none"``.

   The returned ``AnsatzSolution`` carries ``expr``, ``assignment``, ``free``,
   ``rank``, ``status``, ``verification``, ``steps``, ``residual``, ``check``,
   ``points``, ``ansatz`` and ``certificate``. ``status`` is
   ``"exactly_verified"`` only when the residual is symbolically zero — never
   on the strength of the collocation points alone.

   Raises :exc:`AnsatzError`: ``E-ANSATZ-003`` when no member of the family
   satisfies the constraints, ``E-ANSATZ-004`` when the residual is genuinely
   nonlinear in the unknowns.

   Because it goes through ``Matrix.rref``, a coefficient matrix containing an
   entry whose vanishing cannot be decided refuses with ``E-LINALG-010``.

.. function:: ansatz.enumerate_family(ansatz, coeffs=(-1, 0, 1), *, max_members=100000)

   Iterate concrete members over a coefficient grid, for conjecture generation.

.. function:: ansatz.certify_nonneg(candidate, vars=None, *, constraints=(), **kwargs)

   Hand a fitted candidate to :func:`sos_decompose` / :func:`prove_nonneg`.

alkahest.crosscheck
-------------------

Differential testing against an external CAS oracle (SymPy today).

.. function:: crosscheck.check(operation, *args, oracle=None, assumptions=None, points=5, seed=None, pool=None, **kwargs) -> CrossCheck

   Run one comparison. ``CrossCheck.outcome`` is one of ``"agree"``,
   ``"diverge"``, ``"incomparable"``, ``"unavailable"``, settled by the lowest
   rung that can settle it: 1 syntactic, 2 symbolic, 3 rigorous-numeric,
   4 invariant. Operations with no invariant (``diff``, ``limit``, ``series``)
   stop at rung 3 rather than pretending to one.

   **A missing oracle reports** ``"unavailable"`` (``E-XCHECK-002``) — never
   ``"agree"``.

.. function:: crosscheck.sweep(*, seed=None, cases=40, operations=("diff", "integrate", "simplify"), oracle=None, pool=None, points=5) -> SweepReport

   Generate and run a seeded corpus. ``SweepReport.summary()`` always prints the
   seed; ``to_dict()`` is JSON-serialisable. ``seed`` defaults to
   :func:`budget_seed`, then to ``DEFAULT_SEED``.

.. function:: crosscheck.run_frozen_corpus(*, oracle=None, cases=FROZEN_CORPUS)

   Replay the pinned cases, each recording its expected outcome and why.

.. function:: crosscheck.to_sympy(expr, *, assumptions=None)
.. function:: crosscheck.register_oracle(oracle_cls)
.. function:: crosscheck.oracles() -> dict[str, str | None]

   Installed oracles and their versions.

alkahest.smt
------------

SMT-LIB 2 export and a bridge to an external solver (``z3``, ``cvc5``).

.. function:: to_smtlib(formula, logic="auto", *, check_sat=True, get_model=True) -> str

   Emit a complete, runnable SMT-LIB 2 script. Works with no solver installed,
   and accepts quantified formulas.

.. function:: smt.solve(formula, *, solver="auto", logic="auto", budget=None, pool=None) -> SmtResult

   Run an installed solver on a **quantifier-free** formula.

   ``SmtResult`` carries ``status`` (``'sat'`` / ``'unsat'`` / ``'unknown'``),
   ``model`` (exact ``Fraction`` values), ``model_exprs``, ``engine``,
   ``logic``, ``smtlib``, ``verification``, ``reason_unknown``, ``elapsed_ms``,
   ``raw_output`` and ``steps``.

   Trust model, which is deliberately asymmetric:

   - ``sat`` — the model is lifted to exact rationals, substituted back, and
     checked **in this process**; ``verification["status"]`` is
     ``"exactly_verified"``, and a model that fails raises ``E-SMT-004``.
   - ``unsat`` — reported as ``"externally_asserted"`` and excluded from
     ``alkahest.research.MACHINE_CHECKED_STATUSES``. Nothing checked it.
   - ``unknown`` — ``"unverified"``, with ``reason_unknown`` set. A budget trip
     raises :exc:`BudgetExceededError` instead, so "hard" stays distinct from
     "hung".

.. function:: smt.supported(formula, *, solver="auto") -> SmtSupport

   Ask whether this route applies **before** paying for a solver run.
   ``SmtSupport`` carries ``supported``, ``exportable``, ``quantified``,
   ``solver``, ``logic``, ``reason``, ``detail``, ``recommendation``, ``script``
   and ``error``. ``recommendation`` is ``'smt'`` or ``'prefer_in_tree'``.

.. function:: smt.solvers() -> dict[str, str | None]

   Which of ``SOLVERS`` (``'z3'``, ``'cvc5'``) are installed, and their
   versions.
