Error handling API
==================

.. currentmodule:: alkahest

Structured exception hierarchy with stable diagnostic codes.

Base class
----------

.. exception:: AlkahestError

   Base class for all Alkahest exceptions.

   All subclasses expose:

   .. attribute:: code

      Stable diagnostic code string, e.g. ``"E-POLY-001"``.

   .. attribute:: message

      Human-readable description.

   .. attribute:: remediation

      Suggested remediation, or ``None``.

   .. attribute:: span

      Character offset range ``(start, end)`` in the source expression,
      or ``None``.

   Example::

      try:
          UniPoly.from_symbolic(sin(x), x)
      except AlkahestError as e:
          print(e.code)          # E-POLY-001
          print(e.remediation)   # "Use Expr directly, or expand sin(x) as a series first"

Refusals versus verdicts
------------------------

Some codes are **refusals**: Alkahest could not establish the answer, and the
only alternative to saying so was a confident wrong one. They mean *undecided*,
never *false*, and an unattended loop that records one as a negative result
closes a branch it never explored.

Refusals: ``E-CAD-001``, ``E-LINALG-010``, ``E-MAT-004``, ``E-SOS-002``,
``E-ANSATZ-003``, ``E-SMT-003``, ``E-INT-001``, ``E-BUDGET-001..003``.

Verdicts: ``E-INT-004`` (proven non-elementary), ``E-MAT-003`` (proven
singular), ``E-EVAL-009`` (undefined at this point).

Exception subclasses
--------------------

.. exception:: ConversionError

   Code prefix ``E-POLY-*``. Expression → polynomial or rational function
   conversion failed (non-polynomial terms, non-integer exponents, etc.).

.. exception:: DomainError

   Code prefix ``E-DOMAIN-*``. Mathematical side condition violated:
   division by zero, log of non-positive, sqrt of negative.

.. exception:: DiffError

   Code prefix ``E-DIFF-*``. Symbolic differentiation failed (unknown
   derivative, unsupported expression form).

.. exception:: IntegrationError

   Code prefix ``E-INT-*``. Symbolic integration failed.

   Common codes:

   - ``E-INT-001`` — integrand outside supported classes (NotImplemented)
   - ``E-INT-002`` — division by zero during integration
   - ``E-INT-003`` — unsupported extension degree
   - ``E-INT-004`` — provably non-elementary antiderivative (Liouville's theorem)

.. exception:: MatrixError

   Code prefix ``E-MAT-*``. Matrix errors.

   Common codes:

   - ``E-MAT-001`` — shape mismatch
   - ``E-MAT-002`` — operation requires a square matrix
   - ``E-MAT-003`` — matrix is **proven** singular
   - ``E-MAT-004`` — the determinant's vanishing could not be decided; the
     inverse is refused rather than computed on an unproven assumption

.. exception:: LinearAlgebraError

   Code prefix ``E-LINALG-*``. Subclass of :exc:`MatrixError`. Elimination,
   decompositions, and canonical forms.

   Common codes:

   - ``E-LINALG-002`` — nullspace elimination failed
   - ``E-LINALG-004`` — ``minimal_polynomial`` needs symbol-free entries
   - ``E-LINALG-009`` — ``rational_canonical_form`` needs rational constants
   - ``E-LINALG-010`` — an entry's vanishing could be proven neither zero nor
     non-zero, so ``rank`` / ``rref`` / ``nullspace`` / ``eigenvects`` /
     ``jordan_form`` refused. **This is "undecided", not "singular".**

.. exception:: EigenError

   Code prefix ``E-EIGEN-*``. Subclass of :exc:`MatrixError`. Eigenvalues,
   eigenvectors, Jordan form. ``E-EIGEN-005`` is a defective matrix passed to
   ``diagonalize``. Note that ``eigenvects`` surfaces an undecidable entry as
   an :exc:`EigenError` carrying code ``E-LINALG-010``: the code names what
   could not be decided, not the wrapper it arrived in.

.. exception:: CadError

   Code prefix ``E-CAD-*``. Real quantifier elimination (:func:`decide`).

   ``E-CAD-001`` is raised when the sentence is outside the supported fragment
   (polynomial bodies over ℚ, at most two real variables, quantifier prefix of
   at most two) **or** when the only candidate solutions lie at an irrational
   boundary point that rational CAD sampling cannot test exactly. It means
   *undecided*, never *false* — reporting it as a disproof would turn a refusal
   into a fabricated theorem.

.. exception:: AnsatzError

   Code prefix ``E-ANSATZ-*``. Ansatz family construction and fitting
   (``alkahest.ansatz``). ``E-ANSATZ-003`` means no member of *this* family
   satisfies the constraints — a closed branch for that family, not a proof
   that no such object exists. ``E-ANSATZ-004`` is a residual genuinely
   nonlinear in the unknowns, which needs the ``groebner`` route.

.. exception:: CrossCheckError

   Code prefix ``E-XCHECK-*``. Cross-CAS differential testing
   (``alkahest.crosscheck``). ``E-XCHECK-002`` means no oracle is installed —
   it exists so that a missing oracle can never be mistaken for agreement.

.. exception:: SmtError

   Code prefix ``E-SMT-*``. SMT-LIB export, solver invocation, and model lift
   (``alkahest.smt``). ``E-SMT-003`` refuses a model containing an algebraic
   number that cannot be lifted exactly, rather than truncating it to a float;
   ``E-SMT-004`` means the returned model failed the in-process substitution
   check.

.. exception:: OdeError

   Code prefix ``E-ODE-*``. ODE construction or lowering error.

.. exception:: DaeError

   Code prefix ``E-DAE-*``. DAE structural analysis error (Pantelides
   algorithm failure, inconsistent system).

.. exception:: SolverError

   Code prefix ``E-SOLVE-*``. Polynomial system solving error.

   Common codes:

   - ``E-SOLVE-001`` — inconsistent system (no solutions)
   - ``E-SOLVE-002`` — high-degree factor (degree > 2, no symbolic solution)
   - ``E-SOLVE-003`` — Gröbner basis did not converge

.. exception:: JitError

   Code prefix ``E-JIT-*``. LLVM/JIT compilation or linking error.

.. exception:: CudaError

   Code prefix ``E-CUDA-*``. CUDA device, compilation, or kernel launch
   error.

.. exception:: SparseInterpError

   Code prefix ``E-INTERP-00*``. Sparse interpolation failed: oracle
   inconsistency, term bound exceeded, or discrete-log resolution failure.

.. exception:: SparseGcdError

   Code prefix ``E-INTERP-01*``. Sparse modular GCD failed.

   Common codes:

   - ``E-INTERP-010`` — incompatible polynomials (different variable lists)
   - ``E-INTERP-011`` — underlying sparse interpolation step failed
   - ``E-INTERP-012`` — CRT lifting arithmetic error

.. exception:: ParseError

   Code prefix ``E-PARSE-*``. A lexical or syntax error was encountered
   while parsing an expression string via :func:`~alkahest.parse`.

   The ``.span`` attribute holds the ``(start, end)`` byte range of the
   offending token in the source string; ``.remediation`` holds a
   human-readable hint (e.g. a list of known function names when an
   unknown identifier is used as a function)::

      from alkahest import parse, ParseError, ExprPool

      pool = ExprPool()
      try:
          parse("zeta(x)", pool)
      except ParseError as e:
          print(e.code)          # E-PARSE-001
          print(e.span)          # (0, 4)
          print(e.remediation)   # known functions: abs, acos, asin, ...

.. exception:: PoolError

   Code prefix ``E-POOL-*``. ``ExprPool`` misuse: closed pool, cross-pool
   expression mixing, persisted-handle mismatch.

.. exception:: CertificateUnavailableError

   Code prefix ``E-CERT-*``. A Lean certificate was required but the emitter
   withheld one. Unusually among these, the computation *succeeded* — what is
   missing is the machine-checkable evidence, so this is a policy failure
   rather than a mathematical one. Raised only when you ask for it, by
   :func:`~alkahest.require_certificate` or ambiently inside
   ``with alkahest.context(require_certificate=True):``.

   - ``E-CERT-001`` — no certificate available for this result

   ``.remediation`` names the blocking rewrite rules where they can be
   identified. Use :func:`~alkahest.certifiable` to test a route before
   committing to it, and :func:`~alkahest.certificate_coverage` for the whole
   boundary::

      import alkahest as ak

      pool = ak.ExprPool()
      x = pool.symbol("x")

      answer = ak.certifiable("integrate", ak.log(x), x)
      print(bool(answer), answer.reason)   # False class_withheld

      with ak.context(require_certificate=True):
          ak.diff(ak.sin(x), x)            # fine — certifies
          ak.integrate(ak.log(x), x)       # raises E-CERT-001

.. exception:: BudgetExceededError

   Code prefix ``E-BUDGET-*``. A cooperative budget or cancellation trip —
   not a mathematical failure. Raised when an active
   :class:`~alkahest.Budget` is exceeded (or :func:`~alkahest.request_cancel`
   was called) at a checkpoint inside an engine that honors budgets
   (notably :func:`~alkahest.integrate`). See the
   `budgets guide <../budgets.html>`_ and the
   `workload API <workload.html>`_.

   - ``E-BUDGET-001`` — wall-clock limit elapsed
   - ``E-BUDGET-002`` — step limit exceeded
   - ``E-BUDGET-003`` — cancellation requested

   Example::

      import alkahest as ak

      pool = ak.ExprPool()
      x = pool.symbol("x")
      try:
          with ak.context(pool=pool, budget=ak.Budget(max_steps=0)):
              ak.integrate(x**2, x)
      except ak.BudgetExceededError as e:
          print(e.code)  # E-BUDGET-002

Catching errors by subsystem
----------------------------

Match on the base class and filter by code prefix::

   import alkahest

   try:
       result = alkahest.integrate(expr, x)
   except alkahest.AlkahestError as e:
       if e.code.startswith("E-INT-"):
           # integration failed
           print(f"Integration failed ({e.code}): {e.remediation}")
       else:
           raise

For the full error taxonomy (cause classification, rules for adding codes)
see ``docs/error-taxonomy.md`` in the repository.
