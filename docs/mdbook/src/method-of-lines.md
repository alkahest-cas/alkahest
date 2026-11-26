# Method of lines: from a symbolic PDE to a compiled ODE/DAE

`alkahest.experimental.method_of_lines` takes a PDE problem written with the
[PDE foundation layer](./numeric-pde.md) — `PDE(...).on(domain).bc(...).ic(...)`
— discretises it in space, and returns a `MolSystem`: a compiled semi-discrete
system `M y' = f(t, y)` with an exact sparse Jacobian, native time
integrators, JAX/StableHLO export, and a von Neumann stability check. The API
is experimental.

```python
import numpy as np
import alkahest as ak
import alkahest.experimental as akx

P = ak.ExprPool()
x, t = P.symbol("x"), P.symbol("t")
u = akx.Function("u")
U = u(x, t)

heat = (
    akx.PDE(akx.D(U, t) - akx.D(U, x, 2), U)
    .on(akx.Interval(x, 0, 1))
    .bc(u(0, t), 0)
    .bc(u(1, t), 0)
    .ic(u(x, 0), ak.sin(P.pi() * x))
)
ms = akx.method_of_lines(heat, {x: 41})          # 41 nodes, 39 states
res = ms.integrate(0.1)                           # BDF with the exact Jacobian
exact = np.exp(-np.pi**2 * 0.1) * np.sin(np.pi * ms.grid["x"])
assert np.max(np.abs(res.field("u") - exact)) < 3e-4
```

## How the right-hand side is generated

Nothing is generated per grid point. Points that share a stencil — the
interior bulk, each layer next to a boundary, each boundary face — form a
*class*. For each class one scalar kernel is built in the values at fixed
offsets (plus coordinates and `t`), compiled once by the JIT
(`compile_expr` on a list, Cranelift), and applied by the native runtime
over the index box of the class, reading neighbours through shifted slices of
the field arrays. The number of kernels depends on the stencils, not on the
grid, so a 10⁶-point grid compiles as fast as a 10-point one:

```python
big = akx.method_of_lines(heat, {x: 1_000_001})
assert big.n_kernels == akx.method_of_lines(heat, {x: 101}).n_kernels
f = big.rhs(0.0, big.y0)                          # one native pass
print(big.n_states, big.n_kernels, f"{big.compile_time:.3f} s")
```

The **exact sparse Jacobian** is built the same way: per class, one compiled
kernel of the symbolic partial derivatives with respect to every neighbour,
composed with the boundary closures (chain rule through eliminated boundary
values, periodic halos and spectral differentiation). `ms.jacobian(t, y)`
returns a SciPy CSR matrix (`format="dense"` / `"raw"` without SciPy).

## Discretisations

* **Finite differences of any order** (`order=p`): exact Fornberg weights
  from [`fd_weights`](./numeric-pde.md); centred in the interior, and near a
  boundary the most centred window of `m + p` points that fits (stable at
  least to order 8 for diffusion).
* **Upwinding** (`scheme="upwind"`): every advective first-derivative term
  `a·u_x` is written `½(a − |a|)·D⁻u + ½(a + |a|)·D⁺u`, so the bias follows
  the *sign of the coefficient at each point* — also for nonlinear
  coefficients such as Burgers' `u·u_x`. `scheme="weno5"` uses the Jiang–Peng
  WENO-5 derivatives instead (uniform grids).
* **Spectral collocation**: `scheme="chebyshev"` (Gauss–Lobatto nodes) and
  `scheme="fourier"` (periodic). Derivatives are dense differentiation
  matrices applied along each axis.
* **Stretched grids**: give the nodes, `{x: nodes}`; weights are computed per
  point.
* **1–3 space dimensions** on boxes; **systems**; equations of first or second
  order in time (`u_tt = …` becomes `u' = v, v' = …`); unknowns with no time
  derivative are algebraic fields (a DAE).

```python
nu = P.rational(1, 20)
burgers = (
    akx.PDE(akx.D(U, t) + U * akx.D(U, x) - nu * akx.D(U, x, 2), U)
    .on(akx.Interval(x, 0, 1))
    .bc(u(0, t), 0)
    .bc(u(1, t), 0)
    .ic(u(x, 0), 2 * nu * P.pi() * ak.sin(P.pi() * x) / (2 + ak.cos(P.pi() * x)))
)
ms = akx.method_of_lines(burgers, {x: 161}, scheme="weno5")
res = ms.integrate(1.0, rtol=1e-9, atol=1e-11)
xs = ms.grid["x"]
e = np.exp(-np.pi**2 * 0.05)                       # Cole–Hopf solution at t = 1
exact = 2 * 0.05 * np.pi * e * np.sin(np.pi * xs) / (2 + e * np.cos(np.pi * xs))
assert np.max(np.abs(res.field("u") - exact)) < 1e-4
```

## Boundary conditions

A condition is an equation on one face — `u(0, t) = g(t)` (Dirichlet),
`D(u, x)` at the face (Neumann), any combination (Robin), with time-dependent
data — or `u(lo, t) = u(hi, t)`, which makes the variable periodic. Normal
derivatives in a condition are one-sided stencils into the domain.

* `bc_mode="eliminate"` (default): the boundary value is solved from the
  discretised condition (it must be linear in that value) and filled in
  before each evaluation; the state holds interior nodes only.
* `bc_mode="algebraic"`: boundary nodes are states and the condition is an
  algebraic equation — the system is a semi-explicit DAE, integrated by the
  BDF method (which first makes the initial boundary values consistent). If
  the constraints do not determine the boundary values (index > 1), the
  integration is refused with `E-PDE-369` and the result of
  `dae_index_reduce` on `ms.to_dae()` is reported.

```python
ux = lambda v: ak.subs(akx.D(U, x), {x: P.integer(v)})
robin = (
    akx.PDE(akx.D(U, t) - akx.D(U, x, 2), U)
    .on(akx.Interval(x, 0, 1))
    .bc(u(0, t), ak.sin(t))
    .bc(ux(1) + u(1, t), 0)
    .ic(u(x, 0), 0)
)
dae = akx.method_of_lines(robin, {x: 41}, bc_mode="algebraic")
ode = akx.method_of_lines(robin, {x: 41})
print(dae)                                       # ... DAE)
a = dae.integrate(1.0, rtol=1e-9, atol=1e-11).field("u")
b = ode.integrate(1.0, rtol=1e-9, atol=1e-11).field("u")
assert np.max(np.abs(a - b)) < 1e-6
```

## Time integration

`ms.integrate(t_end, method=..., t_eval=...)`:

| method | |
|---|---|
| `"bdf"` (default) | native variable-order BDF 1–5 (Shampine–Reichelt form) with the exact sparse Jacobian; banded LU after reverse Cuthill–McKee ordering; DAEs with algebraic boundary rows |
| `"rk45"` | native Dormand–Prince 5(4) |
| `"euler"`, `"rk4"` | fixed step `dt`, checked against the stability condition |
| `"alkahest_rk45"` | `ode_integrate_rk45` on `ms.to_ode()` (one expression per state — small grids) |
| `"scipy_bdf"`, `"scipy_radau"`, … | `scipy.integrate.solve_ivp` with `ms.rhs` and the sparse `ms.jacobian` |

The native integrators run entirely in Rust: every right-hand side and
Jacobian evaluation is one pass of the compiled kernels.

## Stability

`ms.stability_report()` takes the linear constant-coefficient part of each
scalar equation (its linearisation at `u = 0`, kept when every coefficient
is an exact constant), discretises it with the interior stencil the system
uses, and states the step-size condition for the fixed-step methods:

* explicit Euler on a three-point stencil is passed to `von_neumann` as a
  two-level scheme, and the condition is **proved** (`dt ≤ h²/(2ν)` for the
  heat equation);
* for a symmetric stencil the symbol is real, and the condition is the
  method's real stability interval (`[−2, 0]` for Euler,
  `[−2.785…, 0]` for RK4) over the symbol's range, computed exactly for
  stencils up to five points (`proved`) and numerically beyond;
* for a non-symmetric stencil (advection) the symbol is sampled and the
  status is `numeric` — evidence, not a proof;
* in 2-D and 3-D the diagonal Fourier modes are analysed, which gives a
  **necessary** condition (`proof_status = "necessary"`).

`integrate(method="euler" | "rk4", dt=...)` warns with
`MolStabilityWarning` when `dt` violates the condition, and names the stable
step.

```python
import warnings

heat_nu = (
    akx.PDE(akx.D(U, t) - P.rational(1, 10) * akx.D(U, x, 2), U)
    .on(akx.Interval(x, 0, 1))
    .bc(u(0, t), 0)
    .bc(u(1, t), 0)
    .ic(u(x, 0), ak.sin(P.pi() * x))
)
ms = akx.method_of_lines(heat_nu, {x: 21})
eu = ms.stability_report()["u"]["methods"]["euler"]
print(eu["condition"], eu["proof_status"])       # ((dt ≥ 0) ∧ (dt ≤ 1/80)) proved
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ms.integrate(0.05, method="euler", dt=0.02)
assert "dt <= 0.0125" in str(w[0].message)
```

## Export

`ms.to_jax()` returns a pure `jax.numpy` function `f(t, y)` — the same
gathers and kernels — so `jax.jit`, `jax.jacfwd` and `jax.vmap` apply, and
`ms.to_stablehlo()` lowers it to StableHLO text. `ms.to_ode()` and
`ms.to_dae()` give an `alkahest.ODE` / `alkahest.DAE` with one expression per
state, for small grids.

<!-- doc-check: needs jax -->
```python
import jax

jax.config.update("jax_enable_x64", True)
ms = akx.method_of_lines(burgers, {x: 33})
f = jax.jit(ms.to_jax())
y = ms.y0 + 0.01
assert np.allclose(np.asarray(f(0.0, y)), ms.rhs(0.0, y))
assert "stablehlo" in ms.to_stablehlo()
```

## Errors

`NumericPdeError`, codes `E-PDE-359 … E-PDE-372`:

| code | meaning |
|---|---|
| `E-PDE-359` | unsupported problem shape: no domain, not 1–3 intervals, no time variable, symbolic bounds |
| `E-PDE-360` | an equation cannot be solved for one highest time derivative (nonlinear in it, coupled, order > 2), or the system is not square |
| `E-PDE-361` | invalid grid: too few points, nodes not increasing or not spanning the interval, grid too coarse for the stencil |
| `E-PDE-362` | boundary condition not on a face, involving no unknown, fixing the time argument, too many for a face, or conflicting with periodicity |
| `E-PDE-363` | a boundary condition cannot be eliminated (nonlinear in the boundary value, tangential derivatives) — use `bc_mode="algebraic"` |
| `E-PDE-364` | missing or malformed initial condition |
| `E-PDE-365` | invalid option: unknown scheme, Fourier without periodicity, Chebyshev on a periodic variable, WENO on a stretched grid, algebraic BCs for second order in time |
| `E-PDE-366` | free symbols without values (pass `params=`) |
| `E-PDE-367` | building or compiling the kernels failed |
| `E-PDE-368` | the time integration failed (step size underflow, non-finite values) |
| `E-PDE-369` | a DAE given to an explicit or SciPy method, or a DAE of index > 1 |
| `E-PDE-370` | export needs JAX (or a function has no JAX translation) |
| `E-PDE-371` | a SciPy-backed method or format without SciPy |
| `E-PDE-372` | invalid integration arguments (`dt` missing, unknown method, `to_ode` on a large grid) |
