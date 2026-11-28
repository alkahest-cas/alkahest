# Finite Expression Method (`alkahest.fex`)

> **Experimental.** `alkahest.fex` may change between minor releases.

> **Requires NumPy.** `alkahest.fex` imports NumPy when it is imported, and
> NumPy is not installed with `alkahest`. Run `pip install numpy` first;
> otherwise `import alkahest.fex` fails with `ModuleNotFoundError: No module
> named 'numpy'`. `fex.TorchEvaluator` additionally needs PyTorch.

The Finite Expression Method (FEX; Liang & Yang, arXiv 2206.10121) looks for a
closed-form solution of a PDE inside a fixed binary tree of operator slots.
A candidate is an opcode vector `e` (one operator per slot) plus parameters
`θ`. Leaves are `Σᵢ αᵢ φ(xᵢ) + β` over all `d` coordinates. Search and fitting
run at a concrete dimension, and `finalize` turns a fitted candidate into an
exact closed form, lifts it to a symbolic dimension `d`, and proves it.

## The pieces

| name | what it does |
|---|---|
| `Template` | the tree shape, the operator tables (`0 1 id sq cube quartic exp sin cos tanh log abs`, `+ * - /`) and the dimension; `Template.instantiate(e, θ)` gives an ordinary `Expr` |
| `Problem` | the PDE: `poisson`, the linear conservation law, cubic Schrödinger, the Rayleigh-quotient eigenproblem, `dirichlet` (Ritz energy), `pide` (integro-differential equations with Gaussian jumps), or any operator written in `Jet` symbols (`Problem.from_operator`) |
| `Evaluator` | computes `u`, `∇u`, `Δu`, the residuals and the full Jacobian in `θ` for a batch of candidates and points in one parallel float64 pass (a fused Forward-Laplacian, O(d) per node); scoring a new candidate never compiles anything |
| `fit` | Levenberg–Marquardt on the exact Jacobian, or Adam / BFGS |
| `search`, `solve` | a reference controller (risk-seeking policy gradient), and a driver that searches the restricted proposals of `analyze` in turn |
| `analyze`, `transform_problem`, `reduce_problem` | symbolic pruning: the data's operator vocabulary, separability, radial symmetry, dispersion relations, and the substitutions `u = e^v` / `u = log w` |
| `finalize` | exact closed form, symbolic-`d` lift and proof (below) |
| `TorchEvaluator` | a PyTorch (CUDA) evaluator for the same problems |

Candidates that overflow, divide by zero or take `log` of a non-positive number
score `+inf`, never NaN. Malformed input raises `FexError` (`E-FEX-001` …
`E-FEX-015`).

## Fit, then finalize

<!-- doc-check: needs numpy -->
```python
import numpy as np
from alkahest import fex

d = 5
tpl = fex.Template("depth2_sub", dim=d)
prob = fex.Problem.poisson(d, rhs=-d, bc=lambda x: 0.5 * sum(xi**2 for xi in x))
rng = np.random.default_rng(0)
pts = fex.sample_box(500, d, rng=rng)
bd = fex.sample_box_boundary(200, d, rng=rng)

ev = fex.Evaluator(tpl, prob, pts, bd)
res = fex.fit(ev, [3, 0, 0, 2], rng=rng)   # id(L[sq] + L[0]) = a·Σxᵢ² + b
res.loss < 1e-20                            # True

out = fex.finalize(res, prob, template=tpl, points=pts, boundary=bd)
out["exact"]                                # True
out["expr_symbolic_d"]                      # 1/2 * Sum(Indexed(x, i)^2, i, 1, d)
out["verification"]["symbolic_d"]["status"] # 'exactly_verified'
out["certificate"]["path"]                  # 'kernel'
```

`finalize` pins redundant parameters, filters small coefficients, ties equal
leaf weights, and snaps each value to an exact constant (a rational, a
rational multiple of `π`, `√2`, `√3`, `e`, `log 2`, `log 3`, or a relation found by
PSLQ or LLL). A constant is accepted only within a tolerance that shrinks with
its description cost, and each step is kept only if a refit keeps the loss
inside the gate. Tied leaves then lift to `Sum(φ(Indexed(x, i)), i, 1, d)`.

## What "proved" means here

`out["verification"]` has three parts, in the statuses of
`verify_pde_solution`:

* `concrete`: the PDE residual and every boundary face at the fitted `d`,
  checked exactly;
* `symbolic_d`: the PDE residual and the boundary condition on every face of
  `[lo, hi]^d` for **every** `d`. A zero is a proof; a nonzero is never
  reported as a refutation;
* `interval`: Arb-ball enclosures of the residual at the sample points when no
  exact form exists. These are evidence, not a proof.

For Poisson, the cubic Schrödinger equation, the eigenproblem and linear
transport, `out["certificate"]` carries a Lean 4 theorem about the PDE residual
at every point of `ℝ^d` (at the fitted `d`), checked by proof by reflection
through `Alkahest.Kernel.Pde`, with a tactic `HasDerivAt` fallback for Poisson
and transport. Boundary conditions are not part of the theorem. See
[PDE residuals](./lean-kernel.md#pde-residuals-fex-solutions).

On the FEX paper's problems (from the 3.15.0 CHANGELOG): Poisson at
`d = 10, 20, 50` and the conservation law at `d = 5, 20` are recovered by
`search` and proved for every `d`; with symbolic pruning (`u = e^v`), the cubic Schrödinger
equation is solved at every `d` from 6 to 48 (24 of 24 runs, 7 of 24 without
pruning) and proved for every `d`; the harmonic eigenproblem at `d = 10 … 18`
gives `λ = d` exactly. Benchmarks are in `benchmarks/fex/`.

## Integro-differential equations

`fex.Problem.pide(dim, time=, drift=, diffusion=, jump=, rhs=, terminal=, T=)`
builds `u_t + b·∇u + ½Tr(σσᵀ∇²u) + λ∫(u(x+G) − u − G·∇u)φ(z)dz = rhs` with
Gaussian jumps (`fex.GaussianJump`), following Hardwick–Liang–Yang (arXiv
2410.00835). The search uses the Taylor-moment replacement of the integral,
but `finalize` verifies a solution against the exact integral
(`fex.gaussian_expectation`), and reports whether the replacement was exact on
it. On the paper's two high-dimensional examples `u = |x|²/d` is recovered at
every `d` from 2 to 100 (eq. 32) and from 25 to 100 (eq. 33), and proved for
every `d`.
