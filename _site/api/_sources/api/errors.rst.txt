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

   - ``E-INT-001`` — no matching integration rule
   - ``E-INT-002`` — algebraic extension required (v1.1)

.. exception:: MatrixError

   Code prefix ``E-MAT-*``. Linear algebra errors (shape mismatch,
   singular matrix, non-invertible).

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
