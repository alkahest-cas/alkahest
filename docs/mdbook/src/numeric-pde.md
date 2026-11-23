# Numerical PDE analysis: stencils, spectral matrices, von Neumann

`alkahest.experimental.numeric_pde` covers the symbolic side of discretising a
PDE: exact finite-difference weights and their truncation errors,
multi-dimensional stencils, spectral differentiation matrices, and the
analysis of linear constant-coefficient schemes (von Neumann stability,
modified equation, consistency order). It is pure Python on exact rationals,
and the API is experimental (it may change between minor versions).

**Every stability verdict is either proved or reported as `"unknown"`.** No
floating-point sample ever decides a sign. A condition that could not be
decided exactly comes back with `proof_status="unknown"`, and only the part
that was proved stable is stated.

## Finite-difference weights

```python
import alkahest as ak
from alkahest.experimental import numeric_pde as N

st = N.fd_weights(2, [-1, 0, 1])      # u'' ≈ (u₋₁ − 2u₀ + u₁)/h²
st.weights                           # (1, -2, 1)
st.error_order, st.error_constant    # (2, Fraction(-1, 12))
st.truncation_error()                # -1/12 · h² · Derivative(u(x), x, x, x, x)
```

`fd_weights(order, offsets, h=None, x0=0)` runs Fornberg's algorithm over
exact rationals for any derivative order and any offsets:

* uniform offsets, such as `[-2, -1, 0, 1, 2]`;
* staggered offsets, such as `[-1/2, 1/2]` (pass `Fraction` or strings);
* one-sided offsets, such as `[0, 1, 2]`;
* symbolic offsets, for a non-uniform grid. For example, `[-1, 0, s]` with a
  symbol `s` gives weights that are rational functions of `s`, and the error
  `(1 − s)/3 · h · u'''`.

The truncation error is stated as **`τ = exact − approximation`**:
`τ = C h^p u^{(m+p)} + O(h^{p+1})`. For the central second difference
`C = −1/12`. Floating-point offsets are refused (`E-PDE-351`), because the
weights would no longer be exact.

`stencil(m, accuracy=p, kind="central" | "forward" | "backward")` picks the
smallest stencil of that kind with order of accuracy `p`.

## Multi-dimensional stencils

`stencil((a1, a2, …))` builds the tensor-product stencil for the mixed
partial `∂^α`. `laplacian_stencil(dim, points)` provides:

| stencil | leading error (`exact − approx`) |
|---|---|
| 2-D 5-point, 3-D 7-point | `−h²/12 Σ ∂⁴u/∂xᵢ⁴` |
| 2-D 9-point, 3-D 19-point (Mehrstellen), 3-D 27-point | `−h²/12 Δ²u` |

The compact stencils are second order for `Δu`, but their error is a
multiple of `Δ²u`. With the right-hand side corrected to `f + h²/12 Δf`, they
therefore solve Poisson's equation to fourth order.

Each stencil object has `.error_terms` (a map from multi-indices to exact
coefficients) and `.truncation_error(u, vars)`, which returns the error as an
expression in `Derivative` nodes.

### Kernels

`st.apply("u", j)` returns the stencil applied to grid values as an
expression, for example `(u(j − 1) − 2u(j) + u(j + 1))/h²`.
`st.kernel()` compiles the stencil through `compile_expr` and applies it to
every interior point of a NumPy array. `st.emit_c("name")` returns C99 source
through `emit_c_expr`. Assembling a method-of-lines system is milestone M7b.

## Spectral differentiation matrices

`cheb_diff_matrix(N)` returns the matrix on the Chebyshev–Gauss–Lobatto points
`cos(jπ/N)`. `fourier_diff_matrix(N, order=1|2)` returns the periodic matrix
on `2πj/N`. Both return `(D, x)`.

* **Exact mode** (the default) returns an `alkahest.Matrix`. Entries are
  rational or lie in ℚ(√2), ℚ(√3) or ℚ(√5) when the nodes do (`N ≤ 6` for
  Chebyshev). Otherwise they are exact expressions in `cos(kπ/N)`.
* **Numeric mode** (`numeric=True`) uses Trefethen's formulas and returns
  NumPy arrays. The Chebyshev diagonal is computed by the negative-sum trick.

## Scheme analysis

A scheme is `Σ a_{l,k} u_{j+k}^{n+l} = 0`. You can build it from a
coefficient map, or read it from an expression in applied grid functions:

```python
p = ak.ExprPool()
n, j, dt, dx, nu = (p.symbol(s) for s in ("n", "j", "dt", "dx", "nu"))
u = lambda l, k: p.func("u", [n + l, j + k])
ftcs = N.Scheme.from_expr((u(1,0) - u(0,0))/dt - nu*(u(0,1) - 2*u(0,0) + u(0,-1))/dx**2,
                          "u", n, j, dt=dt, dx=dx)
res = N.von_neumann(ftcs)
assert res["proof_status"] == "proved"
print(res["stability_condition"])   # 0 ≤ dt·nu/dx² ≤ 1/2
```

### Von Neumann

`von_neumann(scheme, subs=None)` returns a mapping-like result with these
keys:

* `G`: the amplification factor, for two-level schemes;
* `amplification_polynomial`: `Φ(g, θ)`;
* `abs_G_squared`: `|G(θ)|²`;
* `stability_condition`: a predicate expression;
* `proof_status`: `"proved"` or `"unknown"`;
* `intervals`: the stable parameter intervals, with exact endpoints.

For the scheme above it returns `0 ≤ ν·dt/dx² ≤ 1/2`. Here the parameters
`dt`, `dx` and `ν` enter only through the monomial `ν dt/dx²`, so the
analysis merges them into one parameter automatically.

The criterion is the von Neumann condition: for every θ, `Φ(·, θ)` keeps its
degree and is a *simple von Neumann polynomial*. That means all roots lie in
the closed unit disc, and roots on the unit circle are simple. For a
two-level scheme this is `|G| ≤ 1`. Schemes for equations of second order in
time, such as leapfrog for `u_tt = c²u_xx`, are detected by a double root
`g = 1` at θ = 0. For those the criterion of Strikwerda's Theorem 8.2.1
applies instead: roots on the unit circle may be double.

How the condition is decided:

1. **Miller's recursion** turns the root condition into a Boolean formula of
   polynomial sign conditions.
2. **The half-angle substitution** `t = tan(θ/2)` makes those conditions
   polynomial in `t`. The point θ = π is checked separately.
3. **The sentence `∀θ` is decided exactly** for one free parameter. A
   one-parameter CAD cuts the parameter line at the real roots of the
   leading coefficients and subresultant coefficients. The condition is then
   decided at a rational sample in every cell, and at every boundary point.
   An irrational boundary is decided with exact arithmetic in ℚ(α), so
   `c² ≤ 1/2` comes back with closed endpoints `±RootOf(2x² − 1)`.

If two or more independent parameters remain, the result is `"unknown"`. Fix
all but one of them with `subs=`.

| scheme | proved condition |
|---|---|
| FTCS heat | `0 ≤ r ≤ 1/2` |
| BTCS, Crank–Nicolson, DuFort–Frankel | `r ≥ 0` |
| upwind | `0 ≤ c ≤ 1` |
| Lax–Friedrichs, Lax–Wendroff | `−1 ≤ c ≤ 1` |
| Beam–Warming | `0 ≤ c ≤ 2` |
| leapfrog (advection) | `−1 < c < 1` |
| leapfrog (wave equation) | `−1 ≤ c ≤ 1` |
| FTCS advection, Richardson | parameter `= 0` only |

### Modified equation and consistency

`modified_equation(scheme, order, dt, dx, subs)` expands the principal root
`g(θ)` exactly and takes `log g = Σ γ_q (iθ)^q`. This gives
`u_t = Σ_q γ_q (Δx^q/Δt) ∂_x^q u`, the Warming–Hyett modified equation,
without eliminating time derivatives by hand. The result reports:

* the physical and numerical terms, split apart;
* `leading_dissipation`, the first even numerical term;
* `leading_dispersion`, the first odd numerical term.

For example, upwind has numerical diffusion `aΔx/2 (1 − c)`, and Lax–Wendroff
has dispersion `−aΔx²/6 (1 − c²)`.

`consistency(scheme, pde, dt, dx, subs)` returns three things:

* the local truncation error: the scheme's residual on an exact solution of
  `u_t = Σ c_q ∂_x^q u`, normalised to approximate `u_t − Lu`;
* the orders `p, q` in `O(Δt^p + Δx^q)`;
* whether the scheme is consistent.

For example, FTCS is `O(Δt + Δx²)` and Crank–Nicolson is `O(Δt² + Δx²)`.
Lax–Friedrichs comes back as `consistent=None`, because its error term
`Δx²/Δt` vanishes only on some refinement paths.

## Errors

`NumericPdeError`, with codes `E-PDE-350` to `E-PDE-358` (see
[Error codes](./errors.md)). The rest of the block up to `E-PDE-399` is
reserved for this layer.
