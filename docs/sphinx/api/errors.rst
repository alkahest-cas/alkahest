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
``E-ASYMPT-004``, ``E-ODE-011``, ``E-ODE-044``, ``E-EVAL-012``.

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

.. exception:: LogicError

   Code prefix ``E-LOGIC-*``. The propositional layer (:func:`simplify_logic`,
   :func:`minimize_logic`, :func:`to_cnf`, :func:`to_dnf`, :func:`to_nnf`,
   :func:`bdd`, :func:`equivalent`, :func:`truth_table`) refused:
   ``E-LOGIC-002`` not a propositional formula, ``E-LOGIC-003`` a size ceiling
   (BDD nodes, normal-form terms, SAT decisions) — never a truncated answer,
   ``E-LOGIC-004`` an exact minimum demanded with ``exact=True`` could not be
   certified, ``E-LOGIC-005`` a malformed argument, ``E-LOGIC-006`` a failed
   internal self-check.  A trip of the active budget is ``BudgetExceededError``.

.. exception:: DiffError

   Code prefix ``E-DIFF-*``. Symbolic differentiation failed (unknown
   derivative, unsupported expression form).
   ``E-DIFF-005``–``E-DIFF-007`` come from the undefined-function helpers
   (:func:`euler_lagrange`, :func:`functions_to_symbols`,
   :func:`replace_function`): not an undefined function applied to
   arguments, a derivative order list of the wrong length, a coordinate that
   is not a function of the time variable alone.

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
   - ``E-TRANSFORM-021`` … ``E-TRANSFORM-024`` — the Fourier **sine** and
     **cosine** transforms (``alkahest.experimental.fourier_sine_transform``
     …): no table entry (``021``), the same variable twice (``022``), a rate
     known not to be positive — the half-line integral diverges (``023``), an
     unknown with an odd ``x``-derivative or a non-bare argument (``024``).
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

.. exception:: PdeError

   Code prefix ``E-PDE-*``. The :mod:`alkahest.experimental` PDE foundation
   layer: unknowns and the jet space, problem objects, ``classify_pde``,
   ``verify_pde_solution``, the ``Sum`` / ``Indexed`` / ``RootOfFamily`` heads
   and transforms of unknowns. Milestone M0 owns ``001``–``049``; later PDE
   milestones own ``050``–``099``, ``100``–``149``, and so on.

   - ``E-PDE-001`` — not an unknown: an unknown is an undefined function
     applied to distinct plain symbols (``u(x, t)``); built-in and reserved
     names (``sin``, ``Derivative``, ``Sum``) are not.
   - ``E-PDE-002`` — inconsistent variables: an argument that is not an
     independent variable, a repeated variable or unknown, a multi-index of the
     wrong length, a derivative in a variable the unknown does not depend on.
   - ``E-PDE-003`` — an unknown evaluated off its declared arguments
     (``u(x, 0)``) where a jet coordinate is needed; such evaluations belong in
     a boundary or initial condition.
   - ``E-PDE-004`` — a derivative above the jet space's maximum order.
   - ``E-PDE-005`` — a jet coordinate's name is taken by an input symbol.
   - ``E-PDE-006`` — not rational in the jets (``sin(u_x)``), so there is no
     polynomial system for a Gröbner engine.
   - ``E-PDE-007`` — not a differential equation in the declared unknowns.
   - ``E-PDE-008`` — wrong system shape (empty, or several where one is
     needed).
   - ``E-PDE-009`` — an ODE naming convention (``y'``, ``q_dot``, ``dy/dt``)
     for a jet space with several independent variables.
   - ``E-PDE-010`` — a malformed domain (empty interval, non-positive radius,
     a coordinate that is not an independent variable).
   - ``E-PDE-011`` — a boundary or initial condition that mentions no unknown.
   - ``E-PDE-012`` — classification of a system (only scalar equations are
     classified).
   - ``E-PDE-013`` — a candidate solution with the wrong number of components.
   - ``E-PDE-020`` — a malformed ``Sum`` (the bound variable is not a symbol,
     or a bound mentions it).
   - ``E-PDE-021`` — a ``Sum`` that cannot be expanded or evaluated: a bound is
     not an integer after substitution, or the range exceeds the term cap.
   - ``E-PDE-022`` — a ``Sum`` whose bounds depend on the differentiation
     variable (``diff`` itself raises ``DiffError`` ``E-DIFF-001`` there).
   - ``E-PDE-023`` — a malformed ``Indexed`` (the base is not a symbol).
   - ``E-PDE-030`` — a malformed ``RootOfFamily``, or one whose parameters are
     not numbers when an enclosure is asked for.
   - ``E-PDE-031`` — a ``RootOfFamily`` root that could not be **certified**:
     no isolated sign change, a pole (write ``tan(λL) = −λ/h`` as
     ``h·sin(λL) + λ·cos(λL) = 0``), a multiple root, or the search budget ran
     out. A refusal, never a guess.
   - ``E-PDE-040`` — a transform cannot act on an unknown because the transform
     variable is not a single bare argument (``u(x, 2t)``, ``u(t²)``); the
     transform raises its own ``TransformError`` carrying this message.
   - ``E-PDE-050`` — ``pdsolve`` has no method for the problem: a system, an
     equation of order other than one (milestone M1 covers first order), a
     single independent variable (use ``dsolve``), or an unknown ``hint``.
   - ``E-PDE-051`` — every applicable method ran and **no candidate verified**;
     the message names each discarded candidate and why. Never an unverified
     answer.
   - ``E-PDE-052`` — Cauchy data on a **characteristic** curve: the Jacobian of
     the data curve against the characteristic direction vanishes identically.
   - ``E-PDE-053`` — side conditions ``pdsolve`` cannot impose on a first-order
     problem: data on a derivative, several conditions, a data manifold of the
     wrong dimension, or parameters that could not be eliminated.
   - ``E-PDE-054`` — a malformed ``Integral(f, t, lo, hi)`` / ``Integral(f, x)``
     (the integration variable is not a symbol, or a bound mentions it).

   Second-order linear equations in two variables (milestone M2,
   ``canonical_form`` / ``solve_second_order`` / ``laplace_cascade``):

   - ``E-PDE-100`` — not a linear equation of order two in an unknown of
     exactly two variables.
   - ``E-PDE-101`` — a characteristic ODE ``dy/dx = λ(x, y)`` has no
     first integral found in closed form, or the coordinates did not reduce
     the principal part.
   - ``E-PDE-102`` — the type is not the same on the whole plane; use
     ``canonical_forms`` for one form per region.
   - ``E-PDE-103`` — the Laplace cascade spent its step budget (or its
     coefficients swelled) with no invariant vanishing. The chain may be
     infinite: ``u_xy = u`` never terminates.
   - ``E-PDE-104`` — no implemented method gives a closed form.
   - ``E-PDE-105`` — a candidate was produced but not proved by substitution,
     so it is withheld.
   - ``E-PDE-106`` — the solution needs ``∫ W(x, y) G(y) dy`` of an
     arbitrary ``G`` with no integral-free form, or a quadrature that does
     not close.
   - ``E-PDE-107`` — conditions other than Cauchy data on a line, a
     characteristic data line, or ``∫ g`` not in closed form.
   - ``E-PDE-108`` — ``laplace_cascade`` needs the principal part to be a
     multiple of ``u_xy``.

   Milestone M3 (the Sturm–Liouville solver and the transform methods) owns
   ``150``–``199``:

   - ``E-PDE-150`` — an eigenproblem outside the implemented classes
     (variable coefficients that ``dsolve`` does not solve, periodic
     conditions with variable coefficients, an unrecognised singular problem).
   - ``E-PDE-151`` — a malformed end condition: ``αX + βX' = 0`` with
     ``α = β = 0``, a regular condition at an infinite end, or ``bounded`` /
     ``decay`` at an end of a problem that is not one of the classical
     singular ones.
   - ``E-PDE-152`` — the eigenvalue condition vanishes identically (every
     ``λ`` would be an eigenvalue).
   - ``E-PDE-153`` — an eigenpair was **refuted** by its exact check and
     withheld; a bug to report, never an answer.
   - ``E-PDE-160`` — the transform method does not apply: not one linear
     homogeneous constant-coefficient equation in two variables, or a domain
     other than ``Line × HalfLine`` (Fourier) or ``HalfLine × HalfLine``
     (Laplace / sine / cosine).
   - ``E-PDE-161`` — initial or boundary data missing, duplicated or of the
     wrong shape (the Laplace method needs zero initial data; the image method
     a homogeneous Dirichlet or Neumann end).
   - ``E-PDE-162`` — ``dsolve`` did not solve the transformed ODE, or the
     bounded solution could not be selected.
   - ``E-PDE-163`` — the candidate built by a transform method was **refuted**
     by substitution and withheld.

   Milestone M5 (Green's functions) owns ``250``–``299``:

   - ``E-PDE-250`` — no fundamental solution / Green's function here: a
     Klein–Gordon or telegraph kernel in more than one dimension, a symbolic
     dimension for an operator other than Laplace and heat, or ``dsolve`` does
     not solve the homogeneous two-point equation.
   - ``E-PDE-251`` — malformed input: the wrong number of space variables, a
     non-symbol coordinate, a source point of the wrong length, a missing time
     variable.
   - ``E-PDE-252`` — the method of images does not apply to the region,
     operator and wall conditions (Neumann disk, a strip for Helmholtz, …).
   - ``E-PDE-253`` — the eigenfunction series cannot be assembled (a norm did
     not close, eigenvalues below the threshold are present or conditional).
   - ``E-PDE-254`` — **no Green's function exists**: the homogeneous problem
     has a nontrivial solution (``λ = 0`` is an eigenvalue).  The modified
     Green's function of the eigenfunction series applies.
   - ``E-PDE-255`` — a constructed Green's function or representation was
     **refuted** by its exact check and withheld.
   - ``E-PDE-256`` — ``green_solve`` does not recognise the problem.

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

.. exception:: UnitError

   Code prefix ``E-UNIT-*``. Physical units and dimensional analysis
   (:mod:`alkahest.units`). ``E-UNIT-001`` is an operation that needs equal
   dimensions — a sum, a conversion — and got two different ones; the message
   names both. ``E-UNIT-002`` is a malformed or unknown unit string.
   ``E-UNIT-003`` refuses an affine temperature (``degC``/``degF``) anywhere but
   on its own in a conversion: ``2 · 20 °C`` has no meaning. ``E-UNIT-004`` is
   a function argument or symbolic exponent that is not a pure number (an
   angle must be in ``rad``). ``E-UNIT-005`` is a symbol or function with no
   known unit; ``E-UNIT-006`` a sum whose terms share a dimension but not a
   unit (its value depends on which number is in which unit); ``E-UNIT-007``
   a fractional power whose scale has no exact rational root.

.. exception:: TensorError

   Code prefix ``E-TENSOR-*``. Tensor calculus on a coordinate chart
   (:mod:`alkahest.tensor`). ``E-TENSOR-001`` a metric that is not ``n x n`` for
   ``n`` distinct coordinate symbols; ``002`` a metric that is not symmetric;
   ``003`` a determinant that is identically zero; ``004`` a floating-point
   entry (the layer is exact); ``005`` a node the component field cannot hold
   (``Piecewise``, a predicate); ``006`` a partial derivative that is not
   available; ``007`` a denominator that vanishes identically; ``008`` the Weyl
   tensor below dimension 3; ``009`` an abstract-index error (an index three
   times, twice up or twice down, mismatched free indices); ``010`` an internal
   invariant, refused rather than answered.

.. exception:: ControlsError

   Code prefix ``E-CTRL-*``. The controls layer (:mod:`alkahest.controls`),
   raised from Python. ``E-CTRL-001`` malformed input (not a rational function
   of the transfer variable, shapes that do not conform, a MIMO system where a
   SISO one is needed); ``E-CTRL-002`` an improper transfer function where a
   proper one is needed; ``E-CTRL-003`` a numeric routine on data with free
   parameters; ``E-CTRL-004`` the requested object does not exist (singular
   Lyapunov operator, no stabilising Riccati solution, infinite DC gain);
   ``E-CTRL-005`` beyond the exact engine — use the numeric route;
   ``E-CTRL-006`` numpy is required and missing.

.. exception:: SpecialFunctionError

   Code prefix ``E-SPECFN-*``. Orthogonal polynomials, spherical harmonics and
   angular-momentum coupling (:func:`alkahest.legendre_p`,
   :func:`alkahest.wigner_3j`, …). ``E-SPECFN-001`` is an angular-momentum
   argument that is not an integer or half-integer (``1/3``, ``0.3``, a
   symbol). ``E-SPECFN-002`` is a negative ``j`` or ``l``. ``E-SPECFN-003`` is a
   non-integer ``l`` or ``m`` for :func:`alkahest.spherical_harmonic_y`.
   ``E-SPECFN-004`` is an explicit polynomial past degree 1000 (64 with a
   symbolic parameter) or a coupling coefficient past ``2j = 20000`` — build the
   node with ``expand=False`` instead. ``E-SPECFN-005`` is the wrong number of
   family parameters (Rust API). A selection-rule violation is not an error: the
   coefficient is an exact ``0``. ``E-SPECFN-006`` is a pole of a Bessel/zeta
   family function (:func:`alkahest.zeta` at ``1``, :func:`alkahest.polylog` at
   ``z = 1`` for ``s <= 1``); ``E-SPECFN-007`` a non-integer order for
   :func:`alkahest.spherical_jn` / :func:`alkahest.spherical_yn`; the Bessel
   expansion cap (order 200) is ``E-SPECFN-004``. ``E-SPECFN-008`` is a
   :func:`alkahest.polygamma` order that is not a non-negative integer; a pole of
   :func:`alkahest.polygamma` or a non-positive-integer lower parameter of
   :func:`alkahest.hyper` is ``E-SPECFN-006``. ``E-SPECFN-009`` is a
   :func:`alkahest.hyper_transform` that does not apply (the logarithmic case
   of a connection formula, a literal ``z`` outside the identity's region,
   parameters in no quadratic family, or a node of the wrong ``p, q``).

.. exception:: NcError

   Code prefix ``E-NC-*``. Noncommutative algebra. ``E-NC-001``: word
   rewriting did not reach a normal form within ``max_steps`` (the rules may
   not terminate). ``E-NC-002``: an expansion past the term or power cap.
   ``E-NC-003``: a rule whose left side is not a single word with coefficient
   1. ``E-NC-004``: a malformed ladder-mode list. ``E-NC-005``: a γ slot that
   is not ``0``–``3``, ``5`` or a commutative label. ``E-NC-006``: a γ trace
   past 14 slots. ``E-NC-007``: a vacuum expectation over a factor that is not
   a listed ladder operator. ``E-NC-008``: a malformed Lie-algebra table (not
   distinct noncommutative generators, bad or inconsistent structure
   constants, failed Jacobi identity). ``E-NC-009``: a malformed
   sequence-wildcard pattern in ``make_rule(..., sequence=[...])``.

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
   - ``E-SOLVE-002`` — an irreducible factor of degree ≥ 3 with symbolic or algebraic coefficients (rational-coefficient univariates of any degree are solved, using ``RootOf`` where needed)
   - ``E-SOLVE-008`` — ``root_of`` was given a non-polynomial / constant input or an out-of-range index
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

.. exception:: LinprogError

   Code prefix ``E-LP-*``. Exact linear / mixed-integer programming
   (:func:`linprog`, :func:`milp`, :func:`linprog_expr`):

   - ``E-LP-001`` — malformed shapes (row length, right-hand-side count,
     bounds, integrality, sense)
   - ``E-LP-002`` — a float or irrational coefficient; refused, never rounded
   - ``E-LP-003`` — not affine in the decision variables
   - ``E-LP-004`` — a strict inequality or a non-relation constraint
   - ``E-LP-005`` — branch and bound reached ``max_nodes``. Not an answer: the
     exception carries ``incumbent`` (best integer point, unproven),
     ``incumbent_fun``, ``bound`` (best open bound) and ``nodes``
   - ``E-LP-006`` — a certificate failed its own exact check; the result was
     withheld
   - ``E-LP-007`` — an unsupported parametric programme
     (:func:`linprog_parametric`, :func:`linprog_expr` with ``parameter``):
     the parameter multiplies a variable in a constraint, enters
     non-linearly, moves both the objective and the right-hand side, or is
     combined with integer variables

   A budget trip during either solver raises :exc:`BudgetExceededError` with
   the same ``incumbent`` / ``incumbent_fun`` / ``bound`` attributes.

.. exception:: FexError

   Code prefix ``E-FEX-*``. The experimental Finite Expression Method engine
   (:mod:`alkahest.fex`). All are refusals of malformed input; a candidate
   that overflows, divides by zero or takes ``log`` of a non-positive number
   is never an error — it is scored with loss ``+inf``.

   - ``E-FEX-001`` — the tree shape does not parse, or is too large
   - ``E-FEX-002`` — an operator name is not in the unary or binary table
   - ``E-FEX-003`` — an opcode vector has the wrong length or an opcode is
     out of range for its slot
   - ``E-FEX-004`` — a parameter vector has the wrong length
   - ``E-FEX-005`` — a point, opcode or parameter array has the wrong shape
   - ``E-FEX-006`` — the PDE operator mentions a symbol that is not a jet
     symbol or coordinate, or cannot be differentiated or compiled
   - ``E-FEX-007`` — the template and problem dimensions disagree, or the
     time coordinate is out of range
   - ``E-FEX-008`` — a residual vector (or Levenberg–Marquardt) was asked of
     a Rayleigh-quotient or normalised loss
   - ``E-FEX-009`` — :func:`alkahest.fex.finalize` was given a candidate
     without its template, or a candidate and a problem that do not belong
     together
   - ``E-FEX-010`` — a ``family=`` for :func:`alkahest.fex.finalize` does not
     reduce to the problem at the problem's own dimension
   - ``E-FEX-011`` — :func:`alkahest.fex.recovery_metric` was not given two
     expressions
   - ``E-FEX-012`` — :func:`alkahest.fex.transform_problem` /
     :func:`alkahest.fex.reduce_problem` were asked for a substitution or a
     variable reduction that does not apply (``u = e^v`` of an
     inhomogeneous problem, a dropped coordinate the operator still reads)
   - ``E-FEX-013`` — malformed controller options for
     :func:`alkahest.fex.search` / :func:`alkahest.fex.solve` (an operator
     outside the template's tables in ``slot_ops``, a slot with nothing
     allowed, an unknown baseline)
   - ``E-FEX-014`` — :class:`alkahest.fex.TorchEvaluator` is unavailable
     (no PyTorch or no CUDA) or does not cover the problem (an expression
     operator outside ``F₀(u, Δu, x) + Σ aᵢ(x) ∂ᵢu + b·|∇u|²``, a
     non-diagonal second-order form)
   - ``E-FEX-015`` — a malformed partial integro-differential problem
     (:meth:`alkahest.fex.Problem.pide`): a nonlocal term or second-order
     form of the wrong size or with non-finite entries, a jump law with a
     negative variance, or nonlocal terms with a loss that has no residual
     vector

.. exception:: JitError

   Code prefix ``E-JIT-*``. LLVM/JIT compilation or linking error.
   ``E-JIT-005`` is raised by :func:`compile_expr` and
   ``CompileCache.compile`` when the expression has a free symbol the input
   list does not bind; the compiled function would have no value at any point.
   ``E-JIT-006`` is raised by the same entry points when the expression applies
   a function with no numeric implementation (an undefined ``f(x)``, a
   ``Derivative`` of one); it used to compile and return NaN at every point.

.. note::

   :func:`to_stablehlo` raises the base :class:`AlkahestError` with code
   ``E-STABLEHLO-001`` (no exact StableHLO lowering for a function, named in
   the message), ``-002`` (a non-numeric node such as ``O(x)``), ``-003`` (a
   free symbol not in the input list), ``-004`` (an infinite/NaN constant) or
   ``-005`` (an invalid MLIR function name).

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

   - ``E-POOL-001`` — arguments from different pools were passed to one
     call: to an operator, to a pool constructor (``pool.add`` /
     ``pool.func`` / ``pool.forall`` / ``pool.gt`` …), or to any function
     or method taking more than one expression (``diff(expr, var)``,
     ``eval_expr(expr, {var: …})``, ``simplify_with(expr, rules)``,
     ``Matrix + Matrix`` …). The message names the argument. An expression's
     id only has meaning in the pool that interned it; build every argument
     from one pool. A ``CompileCache`` is tied to the pool of its first
     ``compile`` until ``clear()``.
   - ``E-POOL-002`` — a built-in function name was given the wrong number of
     arguments (``pool.func("sqrt", [])``, ``pool.func("EllipticPi", [x])``).
     Every consumer of a built-in reads its arguments by position, so such a
     node is refused at construction; a name that is not built in (a
     user-defined ``f``) takes any number of arguments.

.. exception:: CertificateUnavailableError

   Code prefix ``E-CERT-*``. A Lean certificate was required but the emitter
   withheld one. Unusually among these, the computation *succeeded* — what is
   missing is the machine-checkable evidence, so this is a policy failure
   rather than a mathematical one. Raised only when you ask for it, by
   :func:`~alkahest.require_certificate`, by :func:`~alkahest.to_lean` (which
   never returns an empty string), or ambiently inside
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
   - ``E-BUDGET-006`` — a single result would be larger than physical memory,
     or than GMP can represent as one integer; raised with no budget active,
     in place of the abort (or ``SIGFPE``) GMP and FLINT would otherwise
     produce

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
