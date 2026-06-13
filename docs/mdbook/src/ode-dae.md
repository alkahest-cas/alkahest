# ODE and DAE modeling

Alkahest provides symbolic infrastructure for ordinary differential equations (ODEs) and differential-algebraic equations (DAEs), including structural analysis and automatic index reduction.

## ODE

`ODE` represents an ordinary differential equation system. Build one from symbolic expressions:

```python
from alkahest import ExprPool, ODE, lower_to_first_order, sin

pool = ExprPool()
t = pool.symbol("t")
x = pool.symbol("x")
v = pool.symbol("v")

# Simple harmonic oscillator: x'' + ω²x = 0
# Represented as a first-order system: [x' = v, v' = -ω²x]
omega = pool.integer(1)
ode = ODE(
    state=[x, v],
    derivatives=[v, pool.integer(-1) * omega**pool.integer(2) * x],
    independent=t,
)
```

### Lowering to first order

Higher-order ODEs are automatically lowered to first-order form:

```python
from alkahest import lower_to_first_order

first_order_system = lower_to_first_order(higher_order_ode)
```

## DAE

`DAE` represents a differential-algebraic system where some equations are algebraic constraints rather than differential equations.

```python
from alkahest import DAE, pantelides

pool = ExprPool()
t = pool.symbol("t")
x = pool.symbol("x")   # differential variable
y = pool.symbol("y")   # algebraic variable (constrained)
lam = pool.symbol("lam")  # Lagrange multiplier

# Pendulum: differential equations + constraint
dae = DAE(
    equations=[...],  # system of equations
    variables=[x, y, lam],
    independent=t,
)
```

## Pantelides algorithm

The Pantelides algorithm performs structural index reduction on DAEs. It identifies which equations need to be differentiated to make the system structurally regular:

```python
from alkahest import pantelides

reduced = pantelides(dae)
print(reduced.index)          # structural index of the reduced system
print(reduced.differentiated)  # which equations were differentiated
```

Index reduction converts a high-index DAE (index > 1) into an index-1 system that ODE solvers can handle. The result includes the differentiated equations as symbolic expressions.

## Sensitivity analysis

Sensitivity analysis computes how solutions depend on parameters:

```python
from alkahest import sensitivity_system, adjoint_system

pool = ExprPool()
p = pool.symbol("p")   # parameter

# Forward sensitivity: generates ∂x/∂p equations alongside the ODE
sens = sensitivity_system(ode, [p])

# Adjoint method: more efficient for many parameters, one output
adj = adjoint_system(ode, output_expr, [p])
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
```

Built-in components (`resistor`, `capacitor`, `voltage_source`, and others registered via the component API) generate their constitutive equations automatically. `AcausalSystem.flatten` assembles them, plus the Kirchhoff-style connection equations, into a `DAE` that Pantelides can reduce. See `examples/acausal_and_laplace.py` for a runnable end-to-end example.

## Laplace transform

The Laplace transform lives in `alkahest.experimental` (the calculus/ODE/transform surface is not yet semver-frozen):

```python
from alkahest.experimental import inverse_laplace_transform, laplace_transform

s, t = pool.symbol("s"), pool.symbol("t")
F = laplace_transform(pool.integer(1), t, s)            # 1/s
f = inverse_laplace_transform(F, s, t)                   # back to 1 (Heaviside(t))
```

## Hybrid systems

`HybridODE` adds event handling to an ODE: at a crossing event, the state is reset and integration resumes with a new ODE:

```python
from alkahest import HybridODE, Event

# Bouncing ball: velocity reverses at floor contact
bounce_event = Event(
    condition=x,             # fires when x = 0
    reset={v: pool.integer(-1) * v},   # reverse velocity
)
hybrid = HybridODE(ode=base_ode, events=[bounce_event])
```
