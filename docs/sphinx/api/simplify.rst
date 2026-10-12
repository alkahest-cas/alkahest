Simplification API
==================

.. currentmodule:: alkahest

Alkahest provides two simplification engines: a fast rule-based engine and an
e-graph (equality saturation) engine. Domain-specific rule sets are also available.

Rule-based simplification
-------------------------

.. function:: simplify(expr: Expr) -> DerivedResult

   Apply the default arithmetic rule set to *expr* until no more rules
   fire (fixpoint).

   Rules include: identity elements, constant folding, polynomial
   normalization, and associativity/commutativity normalization.

   :returns: ``DerivedResult`` with ``.value`` (simplified) and ``.steps``.

   Example::

      r = simplify(x + pool.integer(0))
      print(r.value)   # x
      print(len(r.steps))  # number of rewrites applied

.. function:: simplify_with(expr: Expr, rules: list[RewriteRule]) -> DerivedResult

   Apply a custom rule set. The rules are applied in the order given until
   fixpoint.

.. function:: simplify_trig(expr: Expr) -> DerivedResult

   Apply trigonometric identities: Pythagorean identity, double-angle and
   half-angle formulas.

   Example::

      r = simplify_trig(sin(x)**2 + cos(x)**2)
      print(r.value)   # 1

   Ends with the default rule set, so the coefficients the identities leave
   behind fold: ``m*c*sin(t)**2 + m*c*cos(t)**2 - m*c`` simplifies to ``0``.

.. function:: simplify_log_exp(expr: Expr) -> DerivedResult

   Apply logarithm and exponential identities. Only includes branch-cut-safe
   rewrites: ``exp(log(x)) → x`` is included only when *x* has a positive
   domain.

.. function:: simplify_expanded(expr: Expr) -> DerivedResult

   Expand products and collect like terms. Useful for canonicalizing
   polynomial expressions before further processing.

Constant folding, special values and zero recognition
------------------------------------------------------

Every simplifier entry point (:func:`simplify`, :func:`simplify_trig`,
:func:`simplify_egraph`, :func:`simplify_log_exp`,
:func:`simplify_trig_normal_form`, :func:`simplify_expanded`) folds exact
special values — ``sin``/``cos``/``tan`` at rational multiples of ``pi`` with
denominator 1, 2, 3, 4 or 6, ``sin(n*pi) = 0`` and ``cos(n*pi) = (-1)**n`` for
an ``Integer``-domain ``n``, ``exp(I*pi*q)``, inverse trig at the standard
values, ``gamma`` at half-integers, ``erf(0)``, ``dilog(0, 1, -1, 1/2)`` —
radical powers (``sqrt(2)**2 = 2``, ``sqrt(8) = 2*sqrt(2)``, ``sqrt(x)**2 = x``
for a provably non-negative ``x``), ``I**2 = -1`` inside products,
cancelling exponentials (``exp(-x)*exp(x) = 1``), and replaces any sum that is
identically zero by ``0`` (see :func:`is_zero`; the step records ``d != 0``
for every denominator ``d`` it cleared).

.. function:: is_zero(expr, *, assumptions=None) -> bool | None

   ``True`` only when *expr* is **proved** identically zero by exact symbolic
   reasoning: it simplifies to ``0``, or its rational normal form over atoms
   (``sin``/``cos``/``tan`` and the hyperbolic functions rewritten through
   ``exp``, radicals and ``I`` reduced by their defining relations, ``log``
   split where every factor is provably positive) has a zero numerator.
   ``False`` when proved non-zero — an exact non-zero normal form of
   independent variables, or a rigorous Arb ball excluding zero at an exact
   rational point inside every symbol's domain and every assumption.
   ``None`` otherwise.  Numerical sampling alone never returns ``True``.
   Floating-point literals in *expr* never produce a proof.

   Example::

      is_zero(sin(x)**2 + cos(x)**2 - 1)   # True
      is_zero(asin(sin(x)) - x)            # False (disproved at a sample)

.. function:: equals(a, b, *, assumptions=None) -> ZeroTest

   :func:`is_zero` of ``a - b`` with its provenance.  The returned
   :class:`ZeroTest` has ``value`` (``True``/``False``/``None``), ``status``
   (``"proved_zero"``, ``"proved_nonzero"``, ``"probably_zero"``,
   ``"unknown"``), ``method`` (``"simplify"``, ``"rational_normal_form"``,
   ``"arb_ball"``, ``"arb_sampling"``, ``"complex_f64_sampling"``, ``"none"``)
   and ``evidence``.  ``"probably_zero"`` means every rigorous ball contained
   zero with radius below ``1e-30``: evidence, not proof, so ``value`` is
   ``None``.  ``bool(test)`` raises ``ValueError`` when ``value`` is ``None``.

.. function:: to_real_form(expr, assumptions=None) -> Expr

   Rewrite *expr* into an equal expression with no imaginary unit and no
   square root of a negative quantity.  ``exp(a ± iθ)`` becomes
   ``e^a(cos θ ± i sin θ)``; ``cos``/``sin``/``cosh``/``sinh`` of a complex
   argument go through the addition formulas (``cosh(I*x)`` → ``cos(x)``);
   ``sqrt(X)`` with ``X <= 0`` proved becomes ``I*sqrt(-X)``; perfect-square
   factors leave a square root (``sqrt(c**2*k)`` → ``abs(c)*sqrt(k)``, →
   ``c*sqrt(k)`` once ``c >= 0`` is known); and ``abs(u)`` is replaced by
   ``u`` wherever the result is an even function of it.  The real part is
   returned only when the imaginary part is **proved** identically zero by
   :func:`is_zero`'s exact methods; otherwise *expr* is returned unchanged,
   so the result is always equal to *expr* on the set the assumptions
   describe.  Free symbols must be real (any domain but ``Complex``).

   Signs are decided from symbol domains, *assumptions* (and the active
   ``context(assumptions=...)``), sign algebra, and a polynomial certificate:
   monomials of uncertain sign are absorbed by AM–GM pairs, and failing that
   the polynomial is factored over ℤ and signed factor by factor.

   ``Matrix.matrix_exp``, ``experimental.dsolve`` and
   ``experimental.inverse_laplace_transform`` (and through it
   ``TransferFunction.step_response`` / ``impulse_response``) apply the
   rewrite to their results.

   Example::

      to_real_form(sqrt(4*a**2))       # 2*abs(a)
      to_real_form(cosh(I*x))          # cos(x)
      to_real_form(exp(I*x))           # exp(I*x), unchanged

Parallel simplification
-----------------------

All three take a single expression and return the same result as
:func:`simplify`; only the schedule differs. Each requires ``--features
parallel`` at build time and falls back to sequential :func:`simplify`
without it, so the calls are always available. Check
``capabilities()["features"]["parallel"]`` to tell which you have.

.. function:: simplify_par(expr: Expr) -> DerivedResult

   Recursive fork-join traversal. Keeps each subtree on one worker, which
   suits **wide** expressions — a large sum or product of independent terms.

   The derivation log may vary in order between runs, because two workers can
   reach the same node concurrently.

.. function:: simplify_redex(expr: Expr) -> DerivedResult

   Level-scheduled traversal: nodes are bucketed by height and every node at a
   given height is rewritten concurrently, whatever its type. This suits
   **deep** expressions, where :func:`simplify_par` finds no wide node to fork
   on and ends up running essentially sequentially.

   Each node is visited exactly once, so the derivation log is
   **deterministic** — identical across runs and across CPU counts.

.. function:: simplify_auto(expr: Expr) -> DerivedResult

   Dispatch to :func:`simplify_par` or :func:`simplify_redex` based on the
   expression's shape and the number of available cores. Reach for this if you
   do not want to think about shape.

.. function:: simplify_strategy(expr: Expr) -> str

   Report which strategy :func:`simplify_auto` would use, without running it:
   ``"fork_join"``, ``"level_scheduled"``, or ``"sequential"`` when the
   extension was built without ``--features parallel``.

   The answer depends on the worker count as well as the expression, so it can
   differ between machines.

Utility passes
--------------

.. function:: collect_like_terms(expr: Expr) -> Expr
   :no-index:

   Collect terms with the same monomial factor::

      r = collect_like_terms(pool.integer(2) * x + pool.integer(3) * x)
      # r == 5*x

.. function:: poly_normal(expr: Expr, vars: list[Expr]) -> Expr
   :no-index:

   Normalize *expr* to canonical polynomial form over the given variables.
   Raises ``ConversionError`` if the expression is not polynomial.

.. function:: subs(expr: Expr, mapping: dict[Expr, Expr]) -> Expr

   Substitute expressions for variables::

      expr = x**2 + y
      result = subs(expr, {x: pool.integer(3), y: pool.integer(1)})
      # result == 10

E-graph simplification
----------------------

.. function:: simplify_egraph(expr: Expr) -> DerivedResult

   Apply equality saturation using the egglog backend.

   Explores many equivalent forms simultaneously and extracts the
   cheapest one according to the active cost function (default:
   ``SizeCost``). More powerful than ``simplify`` for non-obvious
   equivalences; slower and less predictable in performance.

   Requires ``--features egraph``.

.. function:: simplify_egraph_with(expr: Expr, config: dict) -> DerivedResult

   Like ``simplify_egraph`` with explicit configuration.

   Config keys:

   - ``node_limit`` (int) — stop after this many e-nodes
   - ``iter_limit`` (int) — cap every saturation phase at this many rounds
     (``0`` runs no rewrite rules)
   - ``cost`` (str) — cost function: ``"size"``, ``"depth"``, ``"op"``, ``"stability"``

   Example::

      r = simplify_egraph_with(expr, {"node_limit": 5000, "cost": "stability"})

Pattern matching and rules
--------------------------

.. function:: make_rule(name: str, lhs: Expr, rhs: Expr, condition: str = None) -> RewriteRule

   Create a rewrite rule.

   :param name: Stable rule identifier (appears in derivation logs).
   :param lhs: Pattern expression. Variables starting with ``?`` are
      pattern wildcards.
   :param rhs: Replacement template.
   :param condition: Optional domain condition on matched variables
      (``"nonnegative"``, ``"positive"``, etc.).

   Example::

      pv = pool.symbol("?a")
      rule = make_rule("add_zero", lhs=pv + pool.integer(0), rhs=pv)

.. function:: match_pattern(expr: Expr, pattern: Expr) -> list[dict]

   Find all matches of *pattern* in *expr*.

   Returns a list of substitution dicts mapping pattern variable → matched
   expression.

.. class:: RewriteRule

   Opaque handle to a rewrite rule created by :func:`make_rule`.
