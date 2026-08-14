# ODE and DAE modeling

Alkahest provides symbolic infrastructure for ordinary differential equations (ODEs) and differential-algebraic equations (DAEs), including structural analysis and automatic index reduction.

Every snippet on this page is executed by `tests/test_docs_ode_dae.py`, so it runs against the version of Alkahest it ships with.

## ODE

`ODE` represents an explicit first-order system `d(state_vars)/dt = rhs`. The constructor takes three **positional** arguments — state variables, right-hand sides, and the independent variable — and `ODE(...)` and `ODE.new(...)` are the same thing:

```python
from alkahest import ExprPool, ODE

pool = ExprPool()
t = pool.symbol("t")
x = pool.symbol("x")
v = pool.symbol("v")

# Simple harmonic oscillator x'' + x = 0, written as a first-order system:
#   x' = v,  v' = -x
ode = ODE.new([x, v], [v, pool.integer(-1) * x], t)

ode.order            # 2 — number of state variables
ode.state_vars()     # [x, v]
ode.rhs()            # [v, (x * -1)]
ode.is_autonomous()  # True — t does not appear on the right
```

Initial conditions are attached one at a time, and `with_ic` returns a new `ODE` rather than mutating:

```python
ode_with_ic = ode.with_ic(x, pool.integer(1)).with_ic(v, pool.integer(0))
```

### Lowering to first order

`lower_to_first_order` takes the *pieces* of a scalar higher-order equation — the unknown, the right-hand side, the order, and the independent variable — not an existing `ODE`:

```python
from alkahest import lower_to_first_order

# x'' = -4x  →  [x' = x_1, x_1' = -4x]
ode = lower_to_first_order(x, pool.integer(-4) * x, 2, t)

ode.state_vars()   # [x, x_1]
ode.rhs()          # [x_1, (x * -4)]
```

The auxiliary states are named `x_1`, `x_2`, … for the successive derivatives.

## DAE

`DAE` represents a system of implicit equations mixing differential and algebraic constraints, `F(t, variables, derivatives) = 0`. Build one with `DAE.new(equations, variables, derivatives, time_var)` — four positional arguments:

- `equations` — each expression means `g = 0`. Write `x' = f` as `dx - f`.
- `variables` — the dependent variables.
- `derivatives` — a **separate symbol** standing for the time derivative of `variables[i]`, e.g. `pool.symbol("dx/dt")`. Alkahest does not parse the name; the positional pairing is what makes it a derivative.
- `time_var` — the independent variable.

A variable with **no** entry in `derivatives` is purely algebraic. That is how you declare a Lagrange multiplier, and it is what makes the system high-index:

```python
from alkahest import DAE, ExprPool

pool = ExprPool()
t = pool.symbol("t")
x, y, u, w = (pool.symbol(n) for n in ("x", "y", "u", "w"))
lam = pool.symbol("lam")               # Lagrange multiplier — algebraic
dx, dy, du, dw = (pool.symbol(n) for n in ("dx/dt", "dy/dt", "du/dt", "dw/dt"))
one, two = pool.integer(1), pool.integer(2)

# Cartesian pendulum (index 3): x' = u, y' = w, u' = -lam*x, w' = -lam*y - 1,
# subject to x**2 + y**2 = 1.
dae = DAE.new(
    [dx - u, dy - w, du + lam * x, dw + lam * y + one, x**two + y**two - one],
    [x, y, u, w, lam],   # five variables …
    [dx, dy, du, dw],    # … but only four have derivatives
    t,
)

dae.n_equations   # 5
dae.n_variables   # 5
dae.equations()   # the five expressions, each meaning "= 0"
dae.derivatives() # [dx/dt, dy/dt, du/dt, dw/dt]
dae.time_var      # t
```

## Pantelides algorithm

The Pantelides algorithm performs *structural* index reduction: it repeatedly differentiates the equations that a maximum bipartite matching leaves unmatched, until every equation is matched to a variable. It returns the **reduced `DAE`** — not a separate report object:

```python
from alkahest import pantelides

reduced = pantelides(dae)

reduced.index        # 1 — differentiation rounds used
reduced.n_equations  # 6 — one more than the input
reduced.equations()[-1]
# ((x * dx/dt * 2) + (y * dy/dt * 2)) — the differentiated constraint
```

`reduced.index` is the number of rounds, so `0` means the input already had a perfect structural matching and nothing was differentiated. The equations appended by differentiation are visible in `reduced.equations()`, and the higher jets they introduce (`ddx/dt/dt`, …) in `reduced.derivatives()`.

`pantelides` raises `ValueError` (`E-DAE-002`) above index 10. `dae_index_reduce(dae)` runs Pantelides first and falls back to `rosenfeld_groebner` when it hits that cap.

Because the algorithm looks only at *which* variables occur in *which* equations, it can reduce a system whose coefficients make it unsolvable; structural regularity is not numerical regularity.

## Differential elimination

`rosenfeld_groebner` prolongs the system — differentiating each equation and treating the new jets as fresh indeterminates — and computes a Gröbner basis after each round. The basis holds the algebraic consequences of the differential system, which is what elimination-based work (input–output equations, structural identifiability) needs:

```python
from alkahest import DAE, ExprPool, rosenfeld_groebner

pool = ExprPool()
t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")

dae = DAE.new([dx - x], [x], [dx], t)          # x' = x
result = rosenfeld_groebner(dae, max_prolong_rounds=1)

result.consistent    # True — the unit ideal was never reached
result.truncated     # True — stopped at the round budget, not at saturation
result.variables()   # [t, x, dx/dt, ddx/dt/dt] — jets, in exponent-slot order

[str(e) for e in result.final_basis().to_exprs()]
# ['(x + (-1 * ddx/dt/dt))', '(dx/dt + (-1 * ddx/dt/dt))']
```

`final_basis()` returns a `GroebnerBasis` that knows its variable ordering, so `to_exprs()` reads the relations back as `Expr` (each meaning `= 0`). See [Solving](./solving.md) for the rest of the `GroebnerBasis` surface.

`truncated=True` means prolongation stopped because `max_prolong_rounds` ran out, not because the chain saturated. A truncated basis is a *sound* set of consequences but need not be complete — "not in the basis" then does not mean "not a consequence". Nonlinear jets often do not saturate in finitely many algebraic steps, so this is the common case.

`final_basis()` returns `None` when `consistent` is `False`.

## Sensitivity analysis

Sensitivity analysis computes how solutions depend on parameters. `sensitivity_system(ode, params)` augments the state with `∂x/∂p`; `adjoint_system(ode, objective_grad)` takes the gradient of the objective with respect to the state, as a list parallel to `ode.state_vars()`:

```python
from alkahest import ExprPool, ODE, adjoint_system, sensitivity_system

pool = ExprPool()
t, y, k = pool.symbol("t"), pool.symbol("y"), pool.symbol("k")

ode = ODE.new([y], [pool.integer(-1) * k * y], t)   # y' = -k*y

sens = sensitivity_system(ode, [k])
sens.original_dim                # 1
sens.n_params                    # 1
sens.extended_ode.state_vars()   # [y, dS_y_k]

# Objective J = y(T)**2  →  dJ/dy = 2y
adj = adjoint_system(ode, [pool.integer(2) * y])
adj.state_vars()                 # [lambda_y]
adj.rhs()                        # [(k * lambda_y)]
```

## Acausal modeling

Acausal component modeling lets you describe physical systems by their component equations without manually choosing which direction information flows:

```python
from alkahest import AcausalSystem, ExprPool, capacitor, resistor, voltage_source

pool = ExprPool()
t = pool.symbol("t")

# Component constructors return {"name", "n_equations", "n_ports", "component"}.
src = voltage_source("V1", pool.symbol("Vs"))["component"]
res = resistor("R1", pool.symbol("R"))["component"]
cap = capacitor("C1", pool.symbol("C"))["component"]

circuit = AcausalSystem(pool)
circuit.add_component(src)
circuit.add_component(res)
circuit.add_component(cap)

# Wire the loop: Vs.p -> R.p, R.n -> C.p, C.n -> Vs.n
circuit.connect(src.port("V1.p"), res.port("R1.p"))
circuit.connect(res.port("R1.n"), cap.port("C1.p"))
circuit.connect(cap.port("C1.n"), src.port("V1.n"))

# Flatten into a DAE
dae = circuit.flatten(t)
dae.n_equations   # 10
```

Built-in components (`resistor`, `capacitor`, `voltage_source`, and others registered via the component API) generate their constitutive equations automatically. `AcausalSystem.flatten` assembles them, plus the Kirchhoff-style connection equations, into a `DAE` that Pantelides can reduce. See `examples/acausal_and_laplace.py` for a runnable end-to-end example.

## Laplace transform

The Laplace transform lives in `alkahest.experimental` (the calculus/ODE/transform surface is not yet semver-frozen):

```python
from alkahest import ExprPool
from alkahest.experimental import inverse_laplace_transform, laplace_transform

pool = ExprPool()
s, t = pool.symbol("s"), pool.symbol("t")
F = laplace_transform(pool.integer(1), t, s)     # s^-1
f = inverse_laplace_transform(F, s, t)           # back to 1
```

## Hybrid systems

`HybridODE` adds event handling to an ODE: at a crossing event, the state is reset and integration resumes. `Event.new(name, condition, reset_map)` takes the reset map as a list of `(variable, new_value)` pairs, and both `HybridODE.new` and `add_event` return new objects:

```python
from alkahest import Event, ExprPool, HybridODE, ODE

pool = ExprPool()
t, x, v = pool.symbol("t"), pool.symbol("x"), pool.symbol("v")

# Bouncing ball: x' = v, v' = -1; velocity reverses at floor contact
base_ode = ODE.new([x, v], [v, pool.integer(-1)], t)
bounce = Event.new("bounce", x, [(v, pool.integer(-1) * v)])

hybrid = HybridODE.new(base_ode).add_event(bounce)
hybrid.n_events   # 1
hybrid.guards()   # [x] — fires when x crosses 0
```

`Event.rising()` and `Event.falling()` restrict an event to one crossing direction; by default it fires in both.
