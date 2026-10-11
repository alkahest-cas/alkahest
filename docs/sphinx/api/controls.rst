Controls API
============

.. currentmodule:: alkahest

Linear time-invariant systems, Lyapunov and Riccati equations, and frequency
response (:mod:`alkahest.controls`). Exact for symbolic or rational data,
numeric (numpy) where stated. Hypotheses a result rests on are read with
:func:`controls_side_conditions` right after the call. See the *Controls*
chapter of the user guide.

Transfer functions and state space
----------------------------------

.. class:: TransferFunction(num, den=None, var=None, *, dt=None)

   SISO ``num(var)/den(var)``; coefficients may be symbolic. Numerator and
   denominator are kept as given (no silent pole–zero cancellation).

   Properties: ``num``, ``den``, ``num_coeffs``, ``den_coeffs`` (highest power
   first), ``expr``, ``var``, ``dt``, ``is_discrete``. Methods:
   ``from_coeffs(num, den, var)`` (classmethod), ``poles()``, ``zeros()``,
   ``dc_gain()``, ``series``, ``parallel``, ``feedback(other=1, sign=-1)``,
   ``minreal()``, ``to_ss()``, ``step_response(t)``, ``impulse_response(t)``,
   ``discretize(T, method="tustin"|"zoh")``, ``subs(mapping)``,
   ``equals(other)``, ``freqresp(w)``, ``bode(w)``, ``margins()``; operators
   ``* + - /`` and integer ``**``.

.. class:: StateSpace(A, B, C, D=None, *, dt=None)

   MIMO ``ẋ = Ax + Bu``, ``y = Cx + Du``. Methods: ``to_tf(var)``,
   ``characteristic_polynomial(var)``, ``poles()``, ``zeros()`` (SISO),
   ``dc_gain()``, ``controllability_matrix()``, ``observability_matrix()``,
   ``controllability_rank()``, ``observability_rank()``,
   ``is_controllable()``, ``is_observable()`` (generic ranks, pivot hypotheses
   stated), ``series``, ``parallel``, ``feedback(other=None, sign=-1)``,
   ``step_response(t)``, ``impulse_response(t)``,
   ``discretize(T, method="zoh"|"tustin")``, ``minreal()`` (SISO),
   ``subs(mapping)``.

Matrix equations
----------------

.. function:: kronecker(A, B) -> Matrix

   The Kronecker product ``A ⊗ B``.

.. function:: lyapunov(A, Q, method="auto")

   Solve ``A X + X Aᵀ + Q = 0``. Exact by vectorisation (residual verified)
   for rational data up to 8×8 and symbolic up to 3×3; ``method="numeric"``
   (or float entries) returns a ``numpy.ndarray``. ``E-CTRL-004`` when the
   operator is singular (``λᵢ + λⱼ = 0``).

.. function:: care(A, B, Q, R, method="auto", sample=None)

   Stabilising solution of ``AᵀX + XA − XBR⁻¹BᵀX + Q = 0``. Exact for
   ``n ≤ 2`` (the returned branch's Hurwitz conditions are stated), numeric
   via the Hamiltonian otherwise. ``E-CTRL-004`` when no stabilising solution
   exists.

.. function:: lqr(A, B, Q, R, method="auto", sample=None)

   ``(K, X)`` with ``K = R⁻¹BᵀX`` and ``X`` from :func:`care`.

Frequency response
------------------

.. function:: bode(sys, w)

   ``(magnitude_dB, phase_deg)`` at the frequencies ``w`` (unwrapped phase, as
   :func:`scipy.signal.bode`).

.. function:: margins(sys)

   ``Margins(gain_margin, phase_margin, phase_crossover, gain_crossover)``, the
   conventions of ``control.margin``. Crossovers are isolated exactly for
   rational coefficients.

Side conditions and errors
--------------------------

.. function:: controls_side_conditions() -> list[str]

   Hypotheses of the most recent controls call (``"m ≠ 0"``, ``"… > 0"``).

.. function:: matrix_rank_side_conditions() -> list[str]

   Pivot hypotheses of the most recent symbolic :meth:`Matrix.rank` /
   ``rref``: the generic rank is exact where they hold.

:class:`ControlsError` codes ``E-CTRL-001`` … ``E-CTRL-006`` are listed in
:doc:`errors`.
