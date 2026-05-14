Numerics and evaluation API
===========================

.. currentmodule:: alkahest

Compiled evaluation, batch evaluation, and ball arithmetic.

Compiled evaluation
-------------------

.. function:: compile_expr(expr: Expr, vars: list[Expr]) -> CompiledFn

   Compile *expr* to a fast callable.

   With ``--features jit``, uses the LLVM backend. Without it, uses the
   Rust tree-walking interpreter. The API is the same either way.

   :param expr: The expression to compile.
   :param vars: Input variables in order; the compiled function takes a
      list of floats in this order.
   :returns: A :class:`CompiledFn` callable.

   Example::

      pool = ExprPool()
      x = pool.symbol("x")
      y = pool.symbol("y")

      f = compile_expr(x**2 + sin(y), [x, y])
      print(f([3.0, 0.0]))   # 9.0

.. function:: eval_expr(expr: Expr, env: dict[Expr, float]) -> float

   Evaluate *expr* by substituting numeric values from *env*.

   Slower than a compiled function but has zero compilation overhead.
   Use for one-off evaluations::

      result = eval_expr(x**2 + y, {x: 3.0, y: 1.0})
      print(result)   # 10.0

.. function:: numpy_eval(compiled_fn: CompiledFn, *arrays) -> numpy.ndarray

   Vectorise *compiled_fn* over NumPy-compatible arrays.

   Each argument corresponds to one input variable. All arrays must
   have the same number of elements. Accepts NumPy arrays, PyTorch CPU
   tensors, JAX arrays, or anything with ``__dlpack__`` or ``__array__``.

   :returns: ``numpy.ndarray`` of float64 with the same shape as the
      first input.

   Example::

      import numpy as np
      f = compile_expr(sin(x) * exp(pool.integer(-1) * x), [x])
      xs = np.linspace(0, 10, 1_000_000)
      ys = numpy_eval(f, xs)   # vectorised, no Python loop

.. class:: CompiledFn

   A compiled function returned by :func:`compile_expr`.

   .. method:: __call__(inputs: list[float]) -> float

      Evaluate at a single point.

   .. method:: call_batch_raw(flat: list[float], n_vars: int, n_pts: int) -> list[float]

      Low-level batch evaluation. *flat* contains all input values
      concatenated: ``[x0_pt0, x0_pt1, ..., x1_pt0, x1_pt1, ...]``.

   .. attribute:: n_inputs

      Number of input variables.

.. function:: emit_c(expr: Expr, vars: list[Expr], var_name: str, fn_name: str) -> str
   :no-index:

   Emit a standalone C function string::

      c_code = emit_c(sin(x) * exp(pool.integer(-1) * x), [x],
                      var_name="x", fn_name="damped_sin")
      # "double damped_sin(double x) { return sin(x) * exp(-x); }"

Ball arithmetic
---------------

.. class:: ArbBall(mid: float, rad: float, prec: int = 53)

   A real interval ``[mid ± rad]`` backed by FLINT/Arb.

   All arithmetic on ``ArbBall`` values produces a guaranteed enclosure
   of the true result.

   :param mid: Midpoint.
   :param rad: Radius (must be ≥ 0).
   :param prec: Precision of the midpoint in bits (default: 53 = double).

   .. attribute:: mid

      Midpoint of the interval.

   .. attribute:: rad

      Radius of the interval.

   .. attribute:: lo

      Lower bound: ``mid - rad``.

   .. attribute:: hi

      Upper bound: ``mid + rad``.

   **Arithmetic operators**

   ``+``, ``-``, ``*``, ``/``, ``**`` are all supported and produce
   guaranteed enclosures::

      a = ArbBall(2.0, 0.1)
      b = ArbBall(3.0, 0.1)
      print(a + b)    # [4.8, 5.2]

.. function:: interval_eval(expr: Expr, env: dict[Expr, ArbBall]) -> ArbBall

   Evaluate *expr* using ball arithmetic.

   The output is a guaranteed enclosure of the true value for any input
   in the given input balls::

      pool = ExprPool()
      x = pool.symbol("x")

      result = interval_eval(sin(x), {x: ArbBall(1.0, 1e-10)})
      # result contains sin(1) to within the specified precision

GPU evaluation
--------------

.. function:: compile_cuda(expr: Expr, vars: list[Expr]) -> CudaCompiledFn

   Compile *expr* to NVPTX machine code for ``sm_86`` (Ampere).

   Requires ``--features cuda``, an LLVM installation with NVPTX support,
   and CUDA ``libdevice.10.bc``.

   .. class:: CudaCompiledFn

      .. method:: call_batch(inputs: list[list[float]]) -> list[float]

         Evaluate on the default CUDA device.

      .. method:: call_batch_on(device: int, inputs: list[list[float]]) -> list[float]

         Evaluate on a specific CUDA device ordinal.

      .. method:: call_device_ptrs(out_ptr, in_ptr, n_pts)

         Low-level DLPack entry — caller owns the device pointers.
