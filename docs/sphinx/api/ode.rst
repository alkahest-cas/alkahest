ODE and DAE API
===============

.. currentmodule:: alkahest

Symbolic ordinary and differential-algebraic equation systems.

ODE
---

.. class:: ODE

   An ordinary differential equation system.

   Stores state variables, their time derivatives as symbolic expressions,
   and the independent variable::

      pool = ExprPool()
      t = pool.symbol("t")
      x = pool.symbol("x")
      v = pool.symbol("v")

      # Harmonic oscillator: x'' + ω²x = 0 → first-order system
      ode = ODE(
          state=[x, v],
          derivatives=[v, pool.integer(-1) * x],
          independent=t,
      )

.. function:: lower_to_first_order(ode: ODE) -> ODE

   Convert a higher-order ODE to first-order form by introducing new
   state variables for each derivative.

DAE
---

.. class:: DAE

   A differential-algebraic equation system mixing differential equations
   with algebraic constraints.

.. function:: pantelides(dae: DAE) -> DAE

   Apply the Pantelides algorithm for structural index reduction.

   Identifies which equations must be differentiated to make the system
   structurally regular (index 1). Returns the reduced DAE with the
   differentiated equations added as symbolic expressions.

   :returns: A reduced :class:`DAE` with ``.index`` and
      ``.differentiated`` attributes.

Sensitivity analysis
--------------------

.. function:: sensitivity_system(ode: ODE, params: list[Expr]) -> SensitivitySystem

   Generate the forward sensitivity equations alongside the ODE.

   For each parameter ``p`` in *params*, adds equations for ``∂x/∂p``
   to the system.

.. function:: adjoint_system(ode: ODE, output: Expr, params: list[Expr]) -> object

   Generate the adjoint sensitivity equations (reverse-mode AD for ODEs).

   More efficient than forward sensitivity when there are many parameters
   and a scalar output.

.. class:: SensitivitySystem

   The extended ODE with sensitivity equations attached.

Acausal modeling
----------------

.. class:: AcausalSystem(pool: ExprPool)

   An acausal component model. Components are added via
   :meth:`add_component` and connected via :meth:`connect`. The system is
   compiled to a :class:`DAE` via :meth:`flatten`.

   .. method:: add_component(component: Component) -> None

      Add a component to the system.

   .. method:: connect(port_a: Port, port_b: Port) -> None

      Connect two component ports (equates potentials, balances flows).

   .. method:: flatten(time_var: Expr) -> DAE

      Assemble the component and connection equations into a DAE.

.. class:: Component

   A physical component (resistor, capacitor, voltage source, …) with named
   :class:`Port` connectors and constitutive equations. Returned inside the
   ``"component"`` key of the dicts produced by :func:`resistor`,
   :func:`capacitor`, and :func:`voltage_source`.

   .. method:: port(name: str) -> Port | None

      Look up a port by its full name (e.g. ``"R1.p"``).

   .. method:: ports() -> list[Port]

      All ports, in declaration order.

.. class:: Port

   A connection port on a component (e.g. positive/negative terminals
   of a resistor), exposing ``potential`` and ``flow`` :class:`Expr`
   attributes.

.. function:: resistor(name: str, resistance: Expr) -> dict

   Create a resistor component (``v - R*i = 0``). Returns a dict with keys
   ``"name"``, ``"n_equations"``, ``"n_ports"``, and ``"component"`` (a
   :class:`Component`).

.. function:: capacitor(name: str, capacitance: Expr) -> dict

   Create an ideal capacitor component (``C * dv/dt - i = 0``). Same return
   shape as :func:`resistor`.

.. function:: voltage_source(name: str, voltage: Expr) -> dict

   Create an ideal voltage source component (``v_p - v_n = V``). Same return
   shape as :func:`resistor`.

Hybrid systems
--------------

.. class:: HybridODE

   An ODE with discrete events. At each event crossing, the state is
   reset and integration resumes.

.. class:: Event

   A hybrid system event.

   :param condition: Expression; event fires when this crosses zero.
   :param reset: Dict mapping state variables to their post-event values.
