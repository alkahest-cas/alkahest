Core API
========

.. currentmodule:: alkahest

The core API provides the expression pool, expression type, and result object.

ExprPool
--------

.. class:: ExprPool()

   An intern table that owns all expressions in a session.

   Every symbol, integer, and composite expression is stored in an
   ``ExprPool``. Expressions from different pools must not be mixed.

   **Creating symbols and constants**

   .. method:: symbol(name: str, domain: str = "real") -> Expr

      Intern a symbolic variable. Two calls with the same ``name`` and
      ``domain`` return the same ``ExprId``.

      :param name: Variable name (displayed in output).
      :param domain: One of ``"real"``, ``"positive"``, ``"nonnegative"``,
         ``"integer"``, ``"complex"``.

      Example::

         pool = ExprPool()
         x = pool.symbol("x")
         x_pos = pool.symbol("x", "positive")  # distinct from x

   .. method:: integer(n: int) -> Expr

      Intern an exact integer constant.

   .. method:: rational(p: int, q: int) -> Expr

      Intern an exact rational constant ``p/q``.

   **Persistence (V1-14)**

   .. method:: save_to(path: str) -> None

      Serialize the pool to disk. All ``ExprId``\ s remain valid after
      reloading.

   .. classmethod:: load_from(path: str) -> ExprPool

      Reload a previously saved pool.

   .. method:: checkpoint() -> None

      Flush and fsync the pool to its backing file (only valid for
      pools opened with ``open_persistent``).

   .. classmethod:: open_persistent(path: str) -> ExprPool

      Open a mmap-backed persistent pool. Changes are flushed on
      ``checkpoint()`` or when the pool is dropped.

   **Pool information**

   .. method:: len() -> int

      Number of interned expressions.

   .. method:: version() -> str

      Version string of the underlying serialization format.

Expr
----

.. class:: Expr

   A handle to an interned expression node. ``Expr`` objects support
   Python arithmetic operators which build new expressions in the same pool.

   **Operator overloading**

   ``+``, ``-``, ``*``, ``**``, ``/`` all produce new ``Expr`` objects::

      pool = ExprPool()
      x = pool.symbol("x")
      expr = x**2 + pool.integer(3) * x - pool.integer(1)

   **String representation**

   .. method:: __str__() -> str

      Returns the expression as an infix string.

   **Note on domains**

   ``Symbol("x", "real")`` and ``Symbol("x", "complex")`` are
   distinct expressions with different ``ExprId``\ s. Domains are part of
   structural identity, not a global assumption.

DerivedResult
-------------

.. class:: DerivedResult

   The return type of ``diff``, ``simplify``, ``integrate``, and all
   top-level operations.

   .. attribute:: value

      The result expression (``Expr``).

   .. attribute:: steps

      List of rewrite steps. Each step is a ``dict`` with keys:

      - ``rule`` (str) — rule name
      - ``before`` (str) — expression before the rewrite
      - ``after`` (str) — expression after the rewrite
      - ``subst`` (dict, optional) — variable substitution
      - ``side_condition`` (str, optional) — side condition checked

   .. attribute:: certificate

      Lean 4 proof term as a string, or ``None`` if not exported.

   .. attribute:: assumptions

      List of side conditions that were verified during derivation.

   .. attribute:: warnings

      List of non-fatal warning strings (e.g. branch-cut warnings).

   Example::

      dr = diff(sin(x**2), x)
      print(dr.value)   # 2*x*cos(x^2)
      for step in dr.steps:
          print(step['rule'], step['before'], "→", step['after'])

Context manager
---------------

.. function:: context(pool=None, domain="real", simplify=False)

   Context manager that sets a default pool and configuration.

   Inside the context, :func:`symbol` creates symbols in the active pool
   without passing it explicitly::

      with alkahest.context(pool=pool, simplify=True):
          z = alkahest.symbol("z")
          expr = z**2 + alkahest.sin(z)

.. function:: symbol(name: str, domain: str = "real") -> Expr

   Create a symbol in the active context pool. Raises ``RuntimeError``
   if called outside a ``context`` block.

.. function:: active_pool() -> ExprPool

   Return the pool active in the current context.

.. function:: simplify_enabled() -> bool

   Return whether auto-simplification is enabled in the current context.

Parsing
-------

.. function:: parse(source: str, pool: ExprPool, symbols: dict[str, Expr] | None = None) -> Expr

   Parse a mathematical expression string into an :class:`Expr` using a
   Pratt recursive-descent parser.

   :param source: Expression string, e.g. ``"sin(x)^2 + cos(x)^2"``.
   :param pool: Pool used to intern new symbols and constants.
   :param symbols: Optional pre-bound symbol map.  Identifiers not present
      are created via ``pool.symbol(name)`` and added to the map for reuse
      within the same call.
   :raises ParseError: On any lexical or syntax error. ``.span`` holds the
      ``(start, end)`` byte range of the offending token; ``.remediation``
      holds a hint.
   :returns: The parsed :class:`Expr`.

   **Supported syntax**

   - Integer and float literals: ``42``, ``3.14``, ``1e-5``
   - Identifiers (symbols): ``x``, ``alpha``, ``x_1``
   - Operators: ``+`` ``-`` ``*`` ``/`` ``^`` ``**`` and unary ``-`` ``+``
   - Grouping: ``(expr)``
   - Function calls: ``sin(x)``, ``atan2(y, x)`` — all 20 registered
     primitives are supported

   Operator precedence (lowest to highest): ``+`` ``-`` → ``*`` ``/`` →
   unary ``-`` → ``^`` ``**`` (right-associative).

   Example::

      pool = ExprPool()
      x = pool.symbol("x")

      # Parse and differentiate
      e = parse("x^3 - 2*x + 1", pool, {"x": x})
      dr = diff(e, x)
      print(dr.value)   # 3*x^2 - 2

      # Auto-create symbols
      sym_map = {}
      e2 = parse("a*x + b", pool, sym_map)
      print(sorted(sym_map))   # ['a', 'b', 'x']

      # Error handling
      from alkahest import ParseError
      try:
          parse("x @ y", pool, {"x": x})
      except ParseError as err:
          print(err.span)          # (2, 3)
          print(err.remediation)   # only ASCII arithmetic expressions ...
