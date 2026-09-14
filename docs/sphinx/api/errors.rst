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
``E-ANSATZ-003``, ``E-SMT-003``, ``E-INT-001``, ``E-BUDGET-001..003``,
``E-ASYMPT-004``, ``E-ODE-011``, ``E-ODE-044``.

Verdicts: ``E-INT-004`` (proven non-elementary), ``E-MAT-003`` (proven
singular), ``E-EVAL-009`` (undefined at this point), ``E-TRANSFORM-004`` and
``E-TRANSFORM-013`` (the rule's own hypothesis is refuted — no transform of
this shape exists), ``E-ODE-041`` (an irregular singular point has no Frobenius
series).

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

   ``E-EIGEN-008`` arrives the same way. Closed-form eigenvalues are radicals,
   and a radical is not a number until a branch is chosen, so every list is
   checked against ``det(zI - A)`` at sampled ``z`` before it is returned. When
   the two disagree the radicals are not the spectrum on the branches anything
   reads them on, and ``eigenvals`` / ``eigenvects`` / ``diagonalize`` /
   ``jordan_form`` / ``matrix_exp`` refuse. **This is "these are not the
   eigenvalues", not "this matrix has none".**

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

   Code prefix ``E-ODE-*``. One class for every ODE engine, because the prefix
   is the subsystem; the number says which engine declined and why.

   - ``E-ODE-001`` … ``003`` — construction or lowering
   - ``E-ODE-010`` … ``014`` — :func:`~alkahest.experimental.dsolve`.
     ``013`` is a recognised class whose quadrature did not close, ``014`` a
     Riccati equation with no particular solution to seed it; both are strictly
     more informative than ``010`` ("no implemented class matched"), and
     ``011`` is a candidate that failed substitution verification and is
     withheld rather than returned
   - ``E-ODE-020`` … ``026`` — the numeric integrators
   - ``E-ODE-030`` … ``034`` — :func:`~alkahest.experimental.dsolve_system`
   - ``E-ODE-040`` … ``045`` — :func:`~alkahest.experimental.series_solve`
     (Frobenius). ``041`` is a fact about the equation: an irregular singular
     point has no Frobenius series, so no wider implementation would find one.
     ``042`` (irrational indicial roots) is a limit of *this* solver's rational
     recurrence instead.

   .. versionchanged:: 3.10
      The ``series_solve`` block moved from ``E-ODE-020`` … ``025`` to
      ``E-ODE-040`` … ``045``. It had collided with the numeric integrators'
      block since both were written — ``E-ODE-021`` meant both "the adaptive
      step size fell below the floor" and "the point is irregular singular" —
      and the series codes had never reached Python at all, because
      ``series_solve`` raised an uncoded :class:`ValueError`.

.. exception:: TransformError

   Code prefix ``E-TRANSFORM-*``. Laplace, Fourier and Z transforms and their
   inverses. One class for the whole ``transform`` module, as
   :exc:`OdeError` is one class for the whole of ``E-ODE-*``; the number says
   which table (``00x`` Laplace, ``01x`` Fourier, ``10x`` Z) and which failure.

   - ``E-TRANSFORM-001`` / ``011`` / ``101`` — no forward rule matched
   - ``E-TRANSFORM-002`` / ``102`` — the inverse table does not reach this form
   - ``E-TRANSFORM-003`` / ``012`` / ``103`` — the two variables are the same
     symbol
   - ``E-TRANSFORM-004`` — **refusal.** The unilateral hypothesis is *refuted*:
     the Laplace integral runs over ``t ≥ 0``, so a Heaviside/Dirac edge at
     ``a < 0`` is invisible to it (``L{θ(t+1)} = 1/s``, not ``e^s/s``), and an
     advance factor ``e^{+as}`` on the inverse is the transform of no causal
     function
   - ``E-TRANSFORM-013`` — **refusal.** The Fourier table entry's decay
     hypothesis is refuted: at a non-positive rate the defining integral
     diverges, and a negative Lorentzian amplitude transforms to the opposite
     sign and the opposite direction of growth from the tabulated form

   The split matters more than the numbers. A table miss is a fact about *this
   implementation* — rewrite the input, or wait for a wider table. ``004`` and
   ``013`` are facts about *the mathematics*: there is nothing to find, and a
   loop that retries them retries forever. A **symbolic** parameter is reported
   as neither: it cannot be decided, so it is carried as a side condition on
   ``side_conditions`` instead.

   .. versionadded:: 3.10
      These codes existed in the transform modules' message text from the
      start, but arrived in Python as a bare :class:`ValueError` with no
      ``.code``.

.. exception:: AsymptoticError

   Code prefix ``E-ASYMPT-*``. :func:`~alkahest.experimental.asymptotic_expand`.
   ``E-ASYMPT-004`` is an expansion that **was** computed and then withheld,
   because the numeric ``o()``-gate could not confirm a term of it at large
   ``x``; ``E-ASYMPT-005`` is a scale outside the implemented rules.

.. exception:: FpsError

   Code prefix ``E-FPS-*``. Formal power series
   (:class:`~alkahest.experimental.Fps`). ``E-FPS-001`` and ``002`` are the same
   mathematical fact reached two ways — a pole at the origin makes the object a
   Laurent series, not a formal *power* series. ``004`` / ``005`` / ``006`` are
   the constant-term hypotheses the operations need to be well defined
   (``f(0) = 0`` for composition/``exp``/reversion, ``f(0) = 1`` for ``log``,
   ``f(0) ≠ 0`` for the inverse), and each ``.remediation`` names the rewrite
   that removes the obstruction.

.. exception:: ProbabilityError

   Code prefix ``E-PROB-*``. The :mod:`alkahest.experimental` probability
   surface — distributions, expectations, moments, characteristic and
   generating functions, entropy and KL divergence. The six codes are kept
   apart because they call for four different next steps:

   - ``E-PROB-001`` — a parameter that is a **number** violates the law's own
     constraint (``sigma <= 0``, ``a >= b``, ``p`` outside ``[0, 1]``). A
     *symbolic* parameter is never reported here: it cannot be decided, so it
     is carried on :meth:`Distribution.constraints` instead.
   - ``E-PROB-002`` — outside the modelled class: a product of two variates
     (there are no joint laws here), a probability generating function for a
     law that is not on the non-negative integers, a cross-family KL
     divergence.
   - ``E-PROB-003`` — the reduction integral was built and the symbolic
     integrator declined it. The message names the integral.
   - ``E-PROB-004`` — no closed form exists inside this library's primitive
     set: the ``Gamma`` CDF at non-integer shape, the ``Beta`` CDF, the normal
     quantile (``erf**-1``), the log-normal characteristic function.
   - ``E-PROB-005`` — a closed form **was** computed and is being **withheld**,
     because quadrature of its own defining integral could not confirm it.
   - ``E-PROB-006`` — the quantity **does not exist**: a divergent expectation,
     a moment generating function outside its convergence strip, an infinite KL
     divergence. A verdict, not a refusal.

.. exception:: VectorError

   Code prefix ``E-VEC-*``. Vector calculus over an orthogonal chart
   (:class:`alkahest.experimental.Coordinates`). ``E-VEC-004`` is the one to
   read twice: every ``grad``/``div``/``curl``/``laplacian`` formula in the
   module is derived for an orthogonal frame, and on a skew chart they still
   evaluate — to an expression that looks like a divergence and is not one. So
   the tangent inner products must be *proven* to vanish, and an undecided one
   is a refusal rather than an assumption. ``E-VEC-005`` is a scale factor that
   is zero, or whose non-vanishing could not be established; every operator
   divides by it.

   Reached as ``alkahest.experimental.VectorError``; it is not on the
   top-level namespace.

.. exception:: QuaternionError

   Code prefix ``E-QUAT-*``. Quaternion algebra and rotations
   (:class:`alkahest.experimental.Quaternion`). ``E-QUAT-001`` is a zero or
   undecided norm. ``E-QUAT-002`` refuses the axis of the identity rotation,
   which does not exist — *every* unit vector is one, and the conventional
   ``(0, 0, 1)`` is a stated answer to a question with no answer.
   ``E-QUAT-003`` refuses a matrix that could not be checked to be a proper
   rotation, including any symbolic matrix, because Shepperd's branch
   selection is a comparison between entries and there is none to make on a
   symbol.

   Reached as ``alkahest.experimental.QuaternionError``.

.. exception:: DaeError

   Code prefix ``E-DAE-*``. DAE structural analysis error (Pantelides
   algorithm failure, inconsistent system).

.. exception:: SolverError

   Code prefix ``E-SOLVE-*``. Polynomial system solving error.

   Common codes:

   - ``E-SOLVE-001`` — inconsistent system (no solutions)
   - ``E-SOLVE-002`` — high-degree factor (degree > 2, no symbolic solution)
   - ``E-SOLVE-003`` — Gröbner basis did not converge
   - ``E-SOLVE-004`` — ``triangularize`` could not extract a triangular set whose
     ideal contains the input, so it refused rather than return a chain cutting
     out a larger variety. Travels inside ``E-SOLVE-001`` until the bindings
     read ``solver::regular_chains::take_triangularize_refusal()``.

.. note::

   ``radical`` and ``primary_decomposition`` refuse rather than return an ideal
   they cannot certify: ``E-IDEAL-005`` (no certified radical) and
   ``E-IDEAL-006`` (no certified primary decomposition). Both currently arrive
   as a plain :class:`ValueError` whose message states the reason; the stable
   code is available to Rust callers from
   ``alkahest_cas::ideal::take_ideal_refusal()``.

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

      answer = ak.certifiable("diff", ak.log(ak.sin(x)), x)
      print(bool(answer), answer.reason)   # False withheld_uncertifiable_step

      with ak.context(require_certificate=True):
          ak.diff(ak.sin(x), x)            # fine — certifies
          ak.diff(ak.log(ak.sin(x)), x)  # raises E-CERT-001

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
   - ``E-BUDGET-004`` — the declared memory budget was exceeded
   - ``E-BUDGET-005`` — the process address-space limit was about to be
     exhausted; this refusal replaces the uncatchable abort that would follow

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

For the full error taxonomy (cause classification, refusal vs verdict, rules for
adding codes) see the `Error handling <../errors.html>`_ chapter of the user
guide, whose source is ``docs/mdbook/src/errors.md``.
