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

.. function:: simplify_par(exprs: list[Expr]) -> list[DerivedResult]

   Simplify a list of expressions concurrently using Rayon.
   Requires ``--features parallel``.

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
