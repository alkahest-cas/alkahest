Transformation API
==================

.. currentmodule:: alkahest

JAX-style composable transformations over the Alkahest kernel.

Tracing
-------

.. function:: trace(pool: ExprPool, *, names: list[str] = None, domain: str = "real")

   Decorator that symbolically traces a Python function.

   The decorated function is invoked once with symbolic :class:`Expr`
   arguments to capture its computation graph. The result is a
   :class:`TracedFn`.

   :param pool: The expression pool to use.
   :param names: Variable names. Inferred from the function signature
      when omitted.
   :param domain: Domain for all symbolic variables.

   Example::

      pool = ExprPool()

      @alkahest.trace(pool)
      def f(x, y):
          return x**2 + alkahest.sin(y)

      print(f.expr)     # x^2 + sin(y)
      print(f(3.0, 0.0))  # 9.0

.. function:: trace_fn(fn: callable, pool: ExprPool, *, names: list[str] = None, domain: str = "real") -> TracedFn

   Functional (non-decorator) version of :func:`trace`::

      fn = alkahest.trace_fn(lambda x, y: x * alkahest.exp(y), pool)

Gradient
--------

.. function:: grad(fn: TracedFn, *, wrt: list[Expr] = None) -> GradTracedFn

   Return a new callable that computes the gradient of *fn*.

   :param fn: A :class:`TracedFn` (from ``@trace``).
   :param wrt: Variables to differentiate with respect to. Defaults to
      all input variables.
   :returns: :class:`GradTracedFn` returning ``[∂f/∂x0, ∂f/∂x1, ...]``.

   Example::

      df = alkahest.grad(f)
      print(df(1.0, 0.0))   # [2.0, 1.0]

JIT compilation
---------------

.. function:: jit(fn: TracedFn) -> CompiledTracedFn

   JIT-compile a :class:`TracedFn` using the LLVM backend.

   Falls back to the Rust interpreter if ``--features jit`` was not
   enabled at build time.

   :param fn: A :class:`TracedFn`.
   :returns: :class:`CompiledTracedFn` backed by LLVM (or interpreter).

   Example::

      fast_f = alkahest.jit(f)
      print(fast_f(3.0, 0.0))   # compiled evaluation

Composing
~~~~~~~~~

Transformations stack freely::

   # Compiled gradient
   fast_df = alkahest.jit(alkahest.grad(f))

   # Second derivative
   d2f = alkahest.grad(alkahest.grad(f))

   # Vectorised compiled gradient
   import numpy as np
   xs = np.linspace(0, 1, 1_000_000)
   result = alkahest.jit(alkahest.grad(f))(xs, xs)

TracedFn
--------

.. class:: TracedFn

   A symbolic function produced by :func:`trace`.

   .. attribute:: expr

      The symbolic output expression.

   .. attribute:: pool

      The expression pool that owns all nodes.

   .. attribute:: symbols

      The symbolic input variables, one per positional argument.

   .. method:: __call__(*values) -> float or ndarray

      Evaluate numerically. Automatically uses the batch path for
      array inputs.

.. class:: CompiledTracedFn

   A :class:`TracedFn` whose evaluation is backed by LLVM JIT.
   Created by :func:`jit`.

   Has the same ``expr``, ``pool``, ``symbols`` attributes and the
   same call signature as :class:`TracedFn`.

.. class:: GradTracedFn

   The gradient of a :class:`TracedFn`. Created by :func:`grad`.

   .. method:: __call__(*values) -> list

      Return the gradient as a list, one value per ``wrt`` variable.

PyTree utilities
----------------

.. function:: flatten_exprs(tree) -> tuple[list[Expr], TreeDef]

   Flatten a nested structure of expressions into a flat list and a
   :class:`TreeDef` describing the structure.

.. function:: unflatten_exprs(flat: list[Expr], treedef: TreeDef)

   Reconstruct a nested structure from a flat list and a :class:`TreeDef`.

.. function:: map_exprs(fn: callable, tree) -> object

   Apply *fn* to every :class:`Expr` leaf in *tree*, preserving structure::

      simplified = map_exprs(simplify, [expr1, expr2, expr3])

.. class:: TreeDef

   Opaque descriptor of a nested structure's shape, for use with
   :func:`flatten_exprs` / :func:`unflatten_exprs`.
