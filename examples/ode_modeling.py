"""examples/ode_modeling.py — ODE modeling, sensitivity, and adjoints.

Covers the dynamical-systems surface: ODE construction, initial conditions,
first-order lowering, forward sensitivity augmentation, and adjoint systems
for gradient-based parameter estimation.

Run with:
    python examples/ode_modeling.py
"""

import alkahest as ak
from alkahest import (
    ExprPool,
    ODE,
    lower_to_first_order,
    sensitivity_system,
    adjoint_system,
    diff,
    simplify,
    jacobian,
    Matrix,
    sin, cos, exp, log,
)

pool = ExprPool()
t = pool.symbol("t")

# ---------------------------------------------------------------------------
# 1. First-order scalar ODE
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. First-order Scalar ODE: dy/dt = -y  (exponential decay)")
print("=" * 60)

y = pool.symbol("y")

# ODE.new(state_vars, rhs, time_var)
ode_decay = ODE.new([y], [pool.integer(-1) * y], t)

print(f"ODE        : {ode_decay}")
print(f"order      : {ode_decay.order()}")
print(f"autonomous : {ode_decay.is_autonomous()}")
print(f"state_vars : {ode_decay.state_vars()}")
print(f"rhs        : {ode_decay.rhs()}")

# Attach initial condition y(0) = 1
ode_ic = ode_decay.with_ic(y, pool.integer(1))
print(f"\nWith IC:\n{ode_ic}")

# ---------------------------------------------------------------------------
# 2. Second-order ODE → first-order system (harmonic oscillator)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Harmonic Oscillator: x'' = -ω²·x  (2nd → 1st order)")
print("=" * 60)

x = pool.symbol("x")
omega2 = pool.rational(4, 1)  # ω² = 4  →  ω = 2

# lower_to_first_order(var, rhs, order, time_var)
# x'' = -4x  becomes  [x' = v, v' = -4x]
ode_harmonic = lower_to_first_order(x, pool.integer(-1) * omega2 * x, 2, t)

print(f"Lowered ODE:")
print(f"  state_vars  : {ode_harmonic.state_vars()}")
print(f"  rhs         : {ode_harmonic.rhs()}")
print(f"  order       : {ode_harmonic.order()}")

# ---------------------------------------------------------------------------
# 3. Parametric ODE + forward sensitivity analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Parametric ODE: dy/dt = -k·y  (forward sensitivity w.r.t. k)")
print("=" * 60)

k = pool.symbol("k")  # rate parameter

ode_param = ODE.new([y], [pool.integer(-1) * k * y], t)
print(f"Parametric ODE:\n{ode_param}")

# Augment with sensitivity equations dS/dt where S = dy/dk
ss = sensitivity_system(ode_param, [k])
print(f"\nSensitivity system:")
print(f"  original_dim : {ss.original_dim}")
print(f"  n_params     : {ss.n_params}")
print(f"  augmented state_vars : {ss.extended_ode.state_vars()}")
print(f"  augmented rhs        : {ss.extended_ode.rhs()}")
print()

# Interpretation: dS_y_k/dt = -y - k*S  (= ∂(-k·y)/∂y · S + ∂(-k·y)/∂k)
s_var = ss.extended_ode.state_vars()[1]
s_rhs = ss.extended_ode.rhs()[1]
print(f"  Sensitivity variable : {s_var}")
print(f"  d(S)/dt              : {s_rhs}")
print(f"  (S encodes how y changes as k is perturbed)")

# ---------------------------------------------------------------------------
# 4. Adjoint-based gradient computation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Adjoint System for gradient ∂J/∂k, J = y(T)²")
print("=" * 60)

# Objective: J = y(T)²  →  ∂J/∂y = 2y
obj_grad = [pool.integer(2) * y]

adj = adjoint_system(ode_param, obj_grad)
print(f"Adjoint ODE state_vars : {adj.state_vars()}")
print(f"Adjoint ODE rhs        : {adj.rhs()}")
print()
# Adjoint λ satisfies dλ/dt = -k·λ  (backward in time)
# Gradient ∂J/∂k = ∫ λ · ∂f/∂k dt = ∫ -λ·y dt

# ---------------------------------------------------------------------------
# 5. Jacobian of ODE vector field
# ---------------------------------------------------------------------------

print("=" * 60)
print("5. Symbolic Jacobian of a 2D ODE vector field")
print("=" * 60)

# Lotka–Volterra: dx/dt = αx - βxy, dy/dt = δxy - γy
alpha = pool.rational(2, 3)
beta  = pool.rational(4, 3)
delta = pool.integer(1)
gamma = pool.integer(1)

f1 = alpha * x - beta * x * y
f2 = delta * x * y - gamma * y

print(f"dx/dt = {f1}")
print(f"dy/dt = {f2}")

J = jacobian([f1, f2], [x, y])
print(f"\nJacobian ∂(f₁,f₂)/∂(x,y):")
for r in range(J.rows):
    row = [str(J.get(r, c)) for c in range(J.cols)]
    print(f"  [{', '.join(row)}]")

# Equilibrium: (γ/δ, α/β) = (1, 1/2)  →  verify J there
x_eq = pool.integer(1)
y_eq = pool.rational(1, 2)

from alkahest import subs
J00_eq = simplify(subs(J.get(0, 0), {x: x_eq, y: y_eq})).value
J01_eq = simplify(subs(J.get(0, 1), {x: x_eq, y: y_eq})).value
J10_eq = simplify(subs(J.get(1, 0), {x: x_eq, y: y_eq})).value
J11_eq = simplify(subs(J.get(1, 1), {x: x_eq, y: y_eq})).value

print(f"\nJ at equilibrium (x=1, y=1/2):")
print(f"  [[{J00_eq}, {J01_eq}],")
print(f"   [{J10_eq}, {J11_eq}]]")

# ---------------------------------------------------------------------------
# 6. Stability analysis via the symbolic determinant
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. Stability via symbolic determinant and trace")
print("=" * 60)

# Build the 2x2 Jacobian as a Matrix and compute det (for eigenvalue check)
J_eq = Matrix.from_rows([
    [J00_eq, J01_eq],
    [J10_eq, J11_eq],
])
det_J = J_eq.det()
print(f"det(J_eq) = {det_J}")
det_simplified = simplify(det_J).value
print(f"det simplified = {det_simplified}")
print(f"(>0 means saddle-free; irrational eigenvalues → limit cycle)")

print("\nDone.")
