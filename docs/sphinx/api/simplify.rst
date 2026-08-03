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

.. function:: simplify_log_exp(expr: Expr) -> DerivedResult

   Apply logarithm and exponential identities. Only includes branch-cut-safe
   rewrites: ``exp(log(x)) → x`` is included only when *x* has a positive
   domain.

.. function:: simplify_expanded(expr: Expr) -> DerivedResult

   Expand products and collect like terms. Useful for canonicalizing
   polynomial expressions before further processing.

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
   - ``iter_limit`` (int) — stop after this many saturation rounds
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
