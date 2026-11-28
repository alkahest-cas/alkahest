# Partial differential equations

> **Experimental.** Everything on this page lives in `alkahest.experimental`
> (Rust: `alkahest_cas::pde`) and may change between minor releases.

The PDE layer has one rule throughout: **a symbolic answer is returned only
after it has been substituted back.** A candidate that does not verify is
discarded and named in the refusal, and refusals carry `E-PDE-*` codes (see
[Error handling](./errors.md)). This page covers the foundation and the exact
solvers. The rest of the stack has its own pages:

| topic | page |
|---|---|
| fundamental solutions, Green's functions by images, eigenfunction series and variation of parameters | [Green's functions](./green-functions.md) |
| Lie point symmetries, conservation laws, traveling-wave and similarity reductions, Cole–Hopf, Miura, hodograph, Hopf–Lax | [Nonlinear PDEs](./pde-nonlinear.md) |
| exact finite-difference weights, spectral matrices, von Neumann stability, modified equations | [Numerical PDE analysis](./numeric-pde.md) |
| a symbolic PDE problem compiled to an ODE/DAE system, with Krylov solves and WENO-5 fluxes | [Method of lines](./method-of-lines.md) |
| closed-form solutions of high-dimensional PDEs found by expression search, then proved for every dimension | [Finite Expression Method](./fex.md) |

## Unknowns, problems and classification

An unknown is an applied undefined function, `Function("u")(x, t)`, and
`D(f, x, t)` / `D(f, x, 3)` take mixed and higher partials. `PDE(eq, u)` builds
a problem; `.on(domain)`, `.bc(lhs, rhs)` and `.ic(lhs, rhs)` attach a domain
(`Interval`, `HalfLine`, `Line`, `Box`, `Disk`, `Ball` and products such as
`Interval(x, 0, L) * HalfLine(t, 0)`) and conditions.

`classify_pde` reports the order, the linearity class, homogeneity, constant
coefficients and the type from the principal symbol. Named equations are
recognised however they are scaled or written (heat, wave, Laplace, Poisson,
Helmholtz, transport, Burgers, KdV, Fisher–KPP, sine-Gordon, Klein–Gordon,
telegraph). For variable coefficients the type comes with exact regions:

```python
import alkahest as ak
import alkahest.experimental as akx
from alkahest.experimental import D, PDE, Function

p = ak.ExprPool()
x, y, t = p.symbol("x"), p.symbol("y"), p.symbol("t")
U = Function("u")

akx.classify_pde(D(U(x, t), t) - D(U(x, t), x, 2), U(x, t))
# PdeInfo(order=2, linearity='linear', type='parabolic', named=['heat'])

tricomi = akx.classify_pde(y * D(U(x, y), x, 2) + D(U(x, y), y, 2), U(x, y))
[r["type"] for r in tricomi.regions]   # ['hyperbolic', 'parabolic', 'elliptic']
```

A list of equations classifies a system (`PdeSystemInfo`).

`verify_pde_solution(pde, u, bcs, ics)` substitutes a candidate into every
equation and condition and reports each residual. `exactly_verified` means
every residual was proved zero symbolically; numeric evidence can refute a
candidate but never certify one. Arbitrary functions `F1(x − ct)` stay
symbolic. This is the check every solver below runs before it returns.

## `pdsolve`

`pdsolve(problem_or_eq, u, hint=None, ics=..., solvefun="F", all=False)` is a
dispatcher over method tiers. `pdsolve_hints()` lists them:

* **First order** (any number of variables): linear and semilinear equations
  by characteristics; quasilinear equations by Lagrange's method, with an
  implicit general solution when `u` does not isolate; Clairaut (complete and
  singular integrals), `f(p, q) = 0`, `f(u, p, q) = 0`, separable /
  Hamilton–Jacobi equations, and Lagrange–Charpit.
* **Second order, linear, two variables**: an ODE in one variable, a factored
  symbol (d'Alembert's solution falls out of the factoring step), an
  exponential shift, the Laplace cascade, or the canonical form.
* **Nonlinear**: Cole–Hopf, traveling waves and similarity solutions (see
  [Nonlinear PDEs](./pde-nonlinear.md)).

```python
u = U(x, y)
sol = akx.pdsolve(2 * D(u, x) + 3 * D(u, y), u)
sol.kind, sol.method                   # ('general', 'first_order_linear')
sol.value                              # F1(y - 3x/2)
sol.verification["status"]             # 'exactly_verified'

c = p.symbol("c", "positive")
wave = akx.pdsolve(D(U(x, t), t, 2) - c**2 * D(U(x, t), x, 2), U(x, t))
wave.method, wave.value                # ('second_order', F1(x - c t) + F2(t + x/c))
```

Cauchy data is passed with `.ic(...)` or `ics=`. The data curve must be
non-characteristic, and conditions the answer needs come back in
`side_conditions`. For Burgers' equation with `u(x, 0) = sin x` the solution
is implicit and holds until the breaking time:

```python
v = U(x, t)
s = akx.pdsolve(PDE(D(v, t) + v * D(v, x), v).ic(U(x, 0), ak.sin(x)))
s.kind                                 # 'implicit': u = sin(x - t u)
s.side_conditions[0][2]                # '(1 + t cos(x - t u)) ≠ 0'
```

When a characteristic quadrature has no closed form, the answer keeps an
unevaluated `Integral(f, t, lo, hi)` (`alkahest.experimental.Integral`), and is
still verified exactly through the fundamental theorem of calculus.

`canonical_form(eq, u)` returns the characteristic coordinates and the
hyperbolic `w_ξη`, parabolic `w_ηη` or elliptic `w_αα + w_ββ` form, one per
region for a mixed-type equation, and `solve_second_order` and
`laplace_cascade` expose the second-order tier directly.

```python
cf = akx.canonical_form(D(u, x, 2) - 4 * D(u, y, 2), u)
cf.type, cf.equation                   # ('hyperbolic', Derivative(w(xi, eta), xi, eta))
```

## Sturm–Liouville problems

`sturm_liouville(x, (a, b), bc, p=, q=, w=)` solves `(pX')' + qX + λwX = 0`
with Dirichlet, Neumann, Robin (`("robin", α, β)` for `αX + βX' = 0`) or
periodic conditions. Eigenvalues come as a family in an integer index, in
closed form or as a `RootOfFamily` of a transcendental equation, with
certified enclosures when the data is numeric:

```python
L = p.symbol("L", "positive")
es = akx.sturm_liouville(x, (0, L), ("dirichlet", "dirichlet"))
es.families[0].eigenvalue              # (n π / L)^2

h = p.symbol("h", "positive")
robin = akx.sturm_liouville(x, (0, 1), ("dirichlet", ("robin", h, 1))).families[0]
robin.defining_equation                # h sin(ν) + ν cos(ν)
[round(lo, 4) for _, lo, _ in robin.enclosures_with(3, {h: 1})]   # [4.1159, 24.1393, 63.6591]
```

The singular classics are built in: Bessel on a disk, Legendre, Chebyshev,
Hermite and Laguerre. Every eigen-family is checked exactly against the ODE and
both conditions.

## Separation of variables

`separate(eq, u)` tests whether a product (or, with `kind="additive"`, a sum)
ansatz separates the equation. `separation_solve(pde)` solves linear heat,
wave, Laplace, Poisson, Helmholtz, Schrödinger, telegraph and Klein–Gordon
problems on intervals and boxes and on polar, cylindrical and spherical
regions (`PolarDisk`, `Annulus`, `Sector`, `Cylinder`, `SphericalBall`). The
answer is a `Sum` over the modes, with each direction solved as a
Sturm–Liouville problem:

```python
k = p.symbol("k", "positive")
w = U(x, t)
heat = (
    PDE(D(w, t) - k * D(w, x, 2), w)
    .on(akx.Interval(x, 0, L) * akx.HalfLine(t, 0))
    .bc(ak.subs(w, {x: p.integer(0)}), 0)
    .bc(ak.subs(w, {x: L}), 0)
    .ic(ak.subs(w, {t: p.integer(0)}), x * (L - x))
)
series = akx.separation_solve(heat)
series.kind, series.convergence, series.verified   # ('series', 'uniform', True)
first = series.truncate(2)             # the n = 1 term; even modes vanish
```

Inhomogeneous boundary data is lifted, and sources are handled by Duhamel's
principle. `truncate(N)`, `numeric(N, params)` and `compile(N, inputs,
params)` evaluate partial sums. Every mode is verified symbolically. The
projection of the data is checked numerically, which can refute an answer but
never certifies one.

## Transform methods

`transform_solve(pde)` uses the Fourier transform in `x` on the line, the
Laplace transform in `t` on the half-line, and the image method through the
Fourier sine and cosine transforms. It returns a closed form where one exists
and otherwise a kernel representation `∫ G(x − s, t) f(s) ds` with the kernel
`G` verified:

```python
gauss = (
    PDE(D(w, t) - D(w, x, 2), w)
    .on(akx.Line(x) * akx.HalfLine(t, 0))
    .ic(ak.subs(w, {t: p.integer(0)}), ak.exp(-x**2))
)
g = akx.transform_solve(gauss)
g.kind, g.verification["status"]       # ('closed_form', 'exactly_verified')
```

`laplace_transform` and `fourier_transform` also act on unknowns and their
derivatives (`L{u_t} = s·U(x, s) − u(x, 0)`), and
`fourier_sine_transform` / `fourier_cosine_transform` and their inverses are
available on their own.

## Corpora

Each solver has a corpus in `tests/pde_corpus/` with Mathematica 14.3
references; the CHANGELOG for 3.15.0 has the head-to-head tables. On 400
problems of the first-order chapter of Abbasi's PDE benchmark, `pdsolve`
answers 262 with no unverified answer, against 188 for Mathematica 14.3.
