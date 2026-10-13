Propositional logic
===================

.. currentmodule:: alkahest

Boolean connectives, normal forms, exact two-level minimisation, reduced
ordered binary decision diagrams (ROBDDs) and equivalence checking.  Atoms are
Boolean symbols; the transformations that hold under any interpretation of
their atoms (:func:`to_nnf`, :func:`to_cnf`, :func:`to_dnf`,
:func:`simplify_logic`) also accept arithmetic relations such as
``pool.gt(x, 0)`` as opaque atoms.  Refusals raise :exc:`LogicError`
(``E-LOGIC-002`` … ``E-LOGIC-006``); an active ``alkahest.Budget`` is honoured
everywhere (:exc:`BudgetExceededError`).

Connectives
-----------

.. function:: And(*args) -> Expr
.. function:: Or(*args) -> Expr

   n-ary conjunction / disjunction.  Nested arguments of the same kind are
   flattened, the identity (``True`` / ``False``) is dropped, the absorbing
   constant or a complementary pair ``x, Not(x)`` collapses the result,
   duplicates are removed, and the arguments are put in a canonical order, so
   ``And(c, And(b, a)) == And(a, b, c)``.  Python bools are accepted alongside
   expressions; at least one ``Expr`` is required (it names the pool).

.. function:: Not(a) -> Expr

   Negation, with ``Not(Not(x)) == x``.

.. function:: Xor(*args) -> Expr
.. function:: Implies(a, b) -> Expr
.. function:: Equivalent(*args) -> Expr
.. function:: ITE(c, t, e) -> Expr

   Odd parity, material implication, all-equal, and if-then-else.  The kernel
   has three connectives, so these return their definitions written with
   ``And``/``Or``/``Not``; ``Xor`` carries the running parity and its
   complement together, so its size is linear in the number of arguments.

Normal forms and minimisation
-----------------------------

.. function:: to_nnf(expr) -> Expr

   Negation normal form.  ``Not(x < 0)`` stays a negated atom.

.. function:: to_cnf(expr, simplify=False) -> Expr
.. function:: to_dnf(expr, simplify=False) -> Expr

   Normal forms by distribution (contradictory/tautological, duplicate and
   absorbed terms dropped).  Exponential in the worst case and refused past
   100 000 terms (``E-LOGIC-003``).  ``simplify=True`` returns the minimum form
   of :func:`simplify_logic` instead.

.. function:: simplify_logic(expr, form="auto", dontcare=None, exact=None) -> Expr

   Two-level minimisation: ``form`` is ``"dnf"``, ``"cnf"`` or ``"auto"`` (the
   one with fewer literals).  ``dontcare`` is a formula marking assignments
   whose value is free.  Up to 12 variables (16 with ``exact=True``) the result
   is a proven minimum — fewest terms, then fewest literals — from
   Quine–McCluskey prime implicants and a branch-and-bound Petrick cover.
   Beyond that, an espresso-style expand/irredundant pass on the BDD of the
   function returns an equivalent cover that is not guaranteed minimum
   (``exact=True`` refuses with ``E-LOGIC-004`` instead).  Every result is
   checked against the input before it is returned.

   Example::

      p = ExprPool(); a, b = p.symbol("a"), p.symbol("b")
      simplify_logic(Or(And(a, b), And(a, Not(b))))   # a

.. function:: minimize_logic(expr, form="dnf", dontcare=None, exact=None) -> MinimizedLogic

   As :func:`simplify_logic`, returning a :class:`MinimizedLogic` with
   ``expr``, ``form``, ``exact`` (minimality proven), ``terms`` and
   ``literals``.

   Pass a **list or tuple of formulas** to minimise several outputs jointly:
   the result is a :class:`MultiMinimizedLogic` whose ``exprs`` are built from
   product terms (clauses for ``"cnf"``) shared between outputs, with
   ``terms`` / ``literals`` counting each distinct term once, ``shared_terms``
   (terms feeding more than one output) and ``per_output_terms``.  Up to 12
   variables (16 with ``exact=True``) the cover is a proven minimum
   (multi-output Quine–McCluskey primes tagged with the outputs they lie
   inside, then a minimum cover of the (output, minterm) pairs); beyond that
   the espresso-style pass runs per output and the pooled cubes are reduced to
   an irredundant shared set, flagged ``exact=False``.  ``dontcare`` applies to
   every output.  The shared cover never has more distinct terms than the sum
   of the per-output covers, every output is verified against its input, and
   more than 64 outputs is refused (``E-LOGIC-003``).

   Example::

      p = ExprPool(); a, b = p.symbol("a"), p.symbol("b")
      r = minimize_logic([Xor(a, b), And(a, b), Or(a, b)])
      r.terms, r.exact      # (3, True): a·b is used by two outputs

.. class:: MultiMinimizedLogic

   ``exprs`` (one formula per input), ``form``, ``exact``, ``terms``,
   ``literals``, ``per_output_terms``, ``shared_terms``; ``len()`` is the
   number of outputs.

Decision diagrams
-----------------

.. function:: bdd(expr, order=None, max_nodes=None, auto_reorder=None) -> BDD

   The ROBDD of a formula over Boolean symbols.  ``order`` (symbols or names,
   root first) fixes the variable order; unlisted symbols are appended.  The
   default order is depth-first first-visit over the canonical formula, which
   interleaves the operand bits of bit-sliced arithmetic.  A BDD past
   ``max_nodes`` (default 2 000 000) is refused with ``E-LOGIC-003`` — the
   middle output bit of an ``n``-bit multiplier is exponential under every
   order, so this is a real limit, not a formality.

   ``auto_reorder`` turns on dynamic variable reordering (Rudell sifting)
   once the manager holds that many live nodes (``True`` = 50 000, ``False`` =
   off).  ``None`` (default) means on at 50 000 nodes unless an explicit
   ``order`` was given.  Reordering never changes what a BDD means — only its
   order and size — and is checked after every operation that produces a BDD
   (and between the gates of a formula being built).

.. class:: BDD

   ``&``, ``|``, ``^``, ``~``; ``implies``, ``iff``, ``ite(g, h)``;
   ``restrict({x: True})``; ``exists([x])`` / ``forall([x])``;
   ``sat_count(variables=None)`` (exact ``int``; by default over every
   variable of the BDD's manager); ``node_count()``; ``support()``;
   ``order``; ``satisfy_one()``; ``equivalent(other)``; ``to_expr()``.
   ``reorder(max_growth=1.2, max_swaps=200000)`` sifts on demand and returns
   ``{"nodes_before", "nodes_after", "swaps", "complete"}``: each variable is
   moved through the whole order by adjacent swaps (abandoning a direction
   once the manager exceeds ``max_growth`` times the best size seen) and left
   at its best position; ``sat_count``, ``equivalent`` and ``to_expr`` are
   unchanged, ``order`` and ``node_count`` change, and every BDD sharing the
   manager is kept.  The active ``Budget`` is honoured between swaps.
   ``set_auto_reorder(threshold)`` / ``auto_reorder`` configure the
   automatic trigger.  Example: ``x0·y0 + … + x7·y7`` under the order
   ``x0…x7, y0…y7`` has 510 nodes, 16 after ``reorder()``.
   Operands from separate :func:`bdd` calls (or plain ``Expr`` formulas) are
   merged by variable name.

Equivalence and truth tables
----------------------------

.. function:: equivalent(f, g, method="auto", counterexample=False) -> bool

   ``method``: ``"truth_table"`` (≤ 22 variables), ``"bdd"``, ``"sat"`` (DPLL
   on the miter ``f ⊕ g``), or ``"auto"`` (truth table up to 16 variables,
   then a bounded BDD, then SAT).  ``counterexample=True`` returns
   ``(verdict, assignment_or_None)``.  With relations present a ``True``
   verdict is still a proof, but a propositional difference is refused
   (``E-LOGIC-002``): ``x > 0`` and ``Not(x <= 0)`` are propositionally
   independent and equal over ℝ.

.. function:: truth_table(expr, variables=None) -> list[tuple[tuple[bool, ...], bool]]

   Rows from all-``False`` to all-``True``, the last variable varying fastest;
   ``variables=None`` sorts the symbols by name.

:func:`satisfiable` decides any purely propositional formula, including ones
wider than 64 symbols or with heavily shared sub-formulas (parity chains,
carry chains), on the formula's DAG.
