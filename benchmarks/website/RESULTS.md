# Landing-page benchmarks: full results

Generated 2026-10-25 by `benchmarks/website/run_all.py` against Alkahest 3.12.0 built from commit `07061b446227`; benchmark scripts at commit `258acb44a331e7d08b0184207819128f2f1bd659`.

These are the numbers shown on the landing page (`website/src/data/benchmarks.json`), with every timed run, the check applied to each answer, and the losses.

## Machine and versions

- CPU: AMD Ryzen Threadripper PRO 3955WX 16-Cores (16 cores, 32 threads)
- RAM: 252 GiB; OS: Ubuntu 22.04.5 LTS, Linux 6.8.0-138-generic; Python 3.10.12
- Resource cap: 8 CPUs (systemd CPUQuota=800%), MemoryMax=24G, `MALLOC_ARENA_MAX=8`
- Alkahest: 3.12.0
- Mathematica: 14.3.0
- mpmath: 1.3.0
- NumPy: 2.2.6
- SymEngine: 0.14.1
- SymPy: 1.14.0

## Method

Median of 5 timed runs after 1 discarded warm-up (3 runs if a run takes > 5 s, 1 if > 20 s); inputs rebuilt and caches cleared before every run; each system in its own process with a per-run timeout. Mathematica timed with AbsoluteTiming inside the kernel (start-up excluded). Every answer is checked against an independent reference; wrong or incomplete answers are marked and never count as a win.

- Each (case, system) pair runs in a fresh process; inputs are generated from fixed seeds in `benchmarks/website/_inputs.py` and parsed by every system from the same string (parsing is not timed).
- Timing covers only the operation named in the table, not parsing, printing or the correctness check.
- `status` is `ok` (answer matches the reference), `wrong` (does not), `incomplete` (some sub-problems unsolved), `timeout`, or `error`.
- The machine is a shared multi-user server; the runs were serialised with `flock` and capped with `systemd-run`, but other users' load adds noise. Outliers were re-run and the min/max columns below show the spread.

## Core CAS

Polynomial arithmetic, factorisation, GCD, integration, Gröbner bases and real roots — the kernels every other workload sits on.

### Expand f·(f+1), f = (1+x+y+z+w)^20 (`expand_fateman`)

Fateman's polynomial-expansion benchmark: 4 variables, 135,751 terms in the result. Checked by term count and the exact value at (2,-3,5,7).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 329.1 ms | 308.9 ms | 332.2 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | timeout > 300 s | — | — | 0 | timeout |  |
| SymEngine | 0.14.1 | 122.16 s | 122.16 s | 122.16 s | 1 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 872.1 ms | 833.9 ms | 936.5 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 371.2× faster than SymEngine; 2.7× faster than Mathematica.

Note: SymPy is timed with sympy.expand, its standard expansion entry point. SymPy did not finish within 300 s.

Script: `benchmarks/website/core.py`

### Factor a dense degree-200 polynomial over Z (`factor_univariate`)

Product of three random dense polynomials of degree 60, 60, 80 with 20-bit coefficients, expanded (degree 200). Checked: the irreducible factor degrees are recovered.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 14.3 ms | 14.2 ms | 14.5 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 3.02 s | 3.01 s | 3.04 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 24.7 ms | 24.6 ms | 25.0 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 211.7× faster than SymPy; 1.7× faster than Mathematica.

Script: `benchmarks/website/core.py`

### Factor a 4-variable polynomial over Z (`factor_multivariate`)

Product of three random sparse 4-variable polynomials (25 terms, total degree <= 10 each), expanded to 5,069 terms. Checked: the three irreducible factors and their degrees are recovered.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 16.9 ms | 16.6 ms | 17.1 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 4.41 s | 4.30 s | 4.50 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 44.4 ms | 42.8 ms | 47.4 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 261.3× faster than SymPy; 2.6× faster than Mathematica.

Script: `benchmarks/website/core.py`

### GCD of two 4-variable polynomials (`gcd_multivariate`)

g·a and g·b with g, a, b random sparse 4-variable polynomials (61 terms, total degree <= 12 each); the inputs have ~2,100 terms each. Checked: the GCD has g's degree and term count.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 5.26 ms | 5.23 ms | 5.56 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 73.7 ms | 73.2 ms | 74.4 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 26.1 ms | 25.7 ms | 26.4 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 14.0× faster than SymPy; 5.0× faster than Mathematica.

Script: `benchmarks/website/core.py`

### Indefinite integrals: batch of 16 (`integrate_batch`)

16 rational and elementary integrands (partial fractions with repeated and irreducible quadratic factors, x^n·exp·sin, log powers, atan, ...). Each antiderivative is checked by F(2.1) - F(1.3) against mpmath quadrature (rel. tol 1e-9).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 6.70 ms | 6.62 ms | 6.75 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 1.02 s | 1.00 s | 1.06 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 285.2 ms | 272.1 ms | 301.5 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 152.8× faster than SymPy; 42.6× faster than Mathematica.

Script: `benchmarks/website/core.py`

### Gröbner basis: cyclic-5 (`groebner_cyclic5`)

Reduced Gröbner basis, grevlex, 5 variables, over Q. Checked: number of basis elements and the sum of their total degrees (the reduced basis is unique).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 5.30 ms | 5.27 ms | 5.31 ms | 3 | ok | matches reference (consensus of 3 systems) |
| SymPy | 1.14.0 | 424.9 ms | 422.4 ms | 433.1 ms | 3 | ok | matches reference (consensus of 3 systems) |
| Mathematica | 14.3.0 | 9.69 ms | 9.66 ms | 9.73 ms | 3 | ok | matches reference (consensus of 3 systems) |

Alkahest is 80.2× faster than SymPy; 1.8× faster than Mathematica.

Script: `benchmarks/website/core.py`

### Gröbner basis: katsura-5 (`groebner_katsura5`)

Reduced Gröbner basis, grevlex, 6 variables, over Q. Checked: number of basis elements and the sum of their total degrees (the reduced basis is unique).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 17.1 ms | 17.1 ms | 17.2 ms | 3 | ok | matches reference (consensus of 3 systems) |
| SymPy | 1.14.0 | 998.4 ms | 992.8 ms | 1.00 s | 3 | ok | matches reference (consensus of 3 systems) |
| Mathematica | 14.3.0 | 26.7 ms | 26.5 ms | 26.7 ms | 3 | ok | matches reference (consensus of 3 systems) |

Alkahest is 58.3× faster than SymPy; 1.6× faster than Mathematica.

Script: `benchmarks/website/core.py`

### Gröbner basis: katsura-6 (`groebner_katsura6`)

Reduced Gröbner basis, grevlex, 7 variables, over Q. Checked: number of basis elements and the sum of their total degrees (the reduced basis is unique).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 195.9 ms | 195.3 ms | 197.8 ms | 3 | ok | matches reference (consensus of 3 systems) |
| SymPy | 1.14.0 | 11.95 s | 11.94 s | 11.98 s | 3 | ok | matches reference (consensus of 3 systems) |
| Mathematica | 14.3.0 | 277.9 ms | 277.9 ms | 279.3 ms | 3 | ok | matches reference (consensus of 3 systems) |

Alkahest is 61.0× faster than SymPy; 1.4× faster than Mathematica.

Script: `benchmarks/website/core.py`

### Gröbner basis: cyclic-6 (`groebner_cyclic6`)

Reduced Gröbner basis, grevlex, 6 variables, over Q. Checked: number of basis elements and the sum of their total degrees (the reduced basis is unique).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | timeout > 120 s | — | — | 0 | timeout |  |
| SymPy | 1.14.0 | 47.96 s | 47.96 s | 47.96 s | 1 | ok | matches reference (consensus of 2 systems) |
| Mathematica | 14.3.0 | 185.7 ms | 184.5 ms | 189.1 ms | 3 | ok | matches reference (consensus of 2 systems) |

Note: Alkahest did not finish within 120 s.

Script: `benchmarks/website/core.py`

### Isolate all real roots, degree 200 (`real_roots_legendre`)

2^200·P_200(x) (Legendre; its roots are the Gauss–Legendre nodes, clustered near ±1). Exact isolation into disjoint rational intervals. Checked: all 200 roots found.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 171.9 ms | 171.2 ms | 176.0 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 3.57 s | 3.56 s | 3.58 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 43.3 ms | 43.1 ms | 44.1 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 20.7× faster than SymPy; 4.0× slower than Mathematica.

Script: `benchmarks/website/core.py`

## Pharma / PK-PD

Linear compartment models solved in closed form, exposure metrics, parameter sensitivities, and the compiled model evaluated across a virtual population.

### Closed-form exp(A·t), 2-compartment oral model (`pk_matrix_exp`)

Symbolic 3×3 matrix exponential of the depot/central/peripheral rate matrix with symbolic ka, k10, k12, k21 and t (the full analytic solution of the linear PK model). Checked: all nine entries at a parameter point against scipy.linalg.expm.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 33.5 ms | 33.3 ms | 33.9 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 3.31 s | 3.28 s | 3.35 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 27.2 ms | 27.1 ms | 27.6 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 98.8× faster than SymPy; 1.2× slower than Mathematica.

Script: `benchmarks/website/pharma.py`

### AUC₀^∞ and AUMC₀^∞, 1-compartment oral (`pk_auc_aumc`)

Exact improper integrals ∫₀^∞ C(t) dt and ∫₀^∞ t·C(t) dt of the Bateman function with symbolic positive F, D, ka, CL, V. Checked against mpmath quadrature at a parameter point.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 311 µs | 299 µs | 320 µs | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 450.5 ms | 447.1 ms | 454.6 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 1.56 s | 1.54 s | 1.59 s | 5 | ok | matches reference (independent reference) |

Alkahest is 1447.1× faster than SymPy; 5003.8× faster than Mathematica.

Note: A small problem (two closed-form integrals); the differences are integrator overhead, not algorithmic scaling. Mathematica is timed with Integrate[..., Assumptions -> ...].

Script: `benchmarks/website/pharma.py`

### Gradient + Hessian of a 2-compartment model (`pk_sensitivities`)

All 5 first and 15 second partial derivatives of the closed-form 2-compartment oral concentration (eigenvalues written with the square root, ~120 nodes) with respect to ka, k10, k12, k21, V. Checked: all 20 values against 40-digit mpmath numerical differentiation.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 6.10 ms | 6.05 ms | 6.20 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 657.6 ms | 653.2 ms | 663.9 ms | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 1000 µs | 997 µs | 1.15 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 5.86 ms | 5.75 ms | 6.06 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 107.8× faster than SymPy; 6.1× slower than SymEngine; 1.0× slower than Mathematica.

Note: Raw (unsimplified) derivatives in every system; timing is differentiation only.

Script: `benchmarks/website/pharma.py`

### Evaluate a 2-compartment model on 1M virtual subjects (`pk_population_1e6`)

The closed-form 2-compartment oral concentration with per-subject ka, k10, k12, k21, V and sampling time t (1,000,000 subjects, float64). Alkahest: compile_expr + numpy_eval (one thread); NumPy: hand-written vectorised code; SymPy: lambdify(numpy, cse=True); SymEngine: Lambdify(LLVM, cse=True); Mathematica: vectorised packed-array evaluation. Compilation is not timed. Checked: mean and four sampled subjects against 30-digit mpmath.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 30.0 ms | 29.7 ms | 31.9 ms | 5 | ok | matches reference (independent reference) |
| NumPy | 2.2.6 | 56.3 ms | 55.6 ms | 61.8 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 64.8 ms | 62.3 ms | 71.9 ms | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 25.5 ms | 25.3 ms | 25.9 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 332.2 ms | 258.5 ms | 369.1 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 1.9× faster than NumPy; 2.2× faster than SymPy; 1.2× slower than SymEngine; 11.1× faster than Mathematica.

Script: `benchmarks/website/pharma.py`

## Engineering / robotics

Equations of motion of multibody systems, manipulator kinematics, and the compiled kinematics evaluated over a million configurations.

### Euler–Lagrange equations, 4-link pendulum (`eom_4link`)

Derive the 4 equations of motion d/dt(∂L/∂q̇ᵢ) − ∂L/∂qᵢ of a planar 4-link pendulum with symbolic masses, lengths and g, from its Lagrangian (built untimed). SymPy uses sympy.physics.mechanics.LagrangesMethod. Checked: each equation at a numeric state against the closed-form M(q)q̈ + c(q,q̇) + g(q).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 1.99 ms | 1.98 ms | 2.03 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 676.7 ms | 672.1 ms | 703.6 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 1.15 ms | 1.14 ms | 1.16 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 340.0× faster than SymPy; 1.7× slower than Mathematica.

Script: `benchmarks/website/engineering.py`

### Euler–Lagrange equations, 6-link pendulum (`eom_6link`)

Derive the 6 equations of motion d/dt(∂L/∂q̇ᵢ) − ∂L/∂qᵢ of a planar 6-link pendulum with symbolic masses, lengths and g, from its Lagrangian (built untimed). SymPy uses sympy.physics.mechanics.LagrangesMethod. Checked: each equation at a numeric state against the closed-form M(q)q̈ + c(q,q̇) + g(q).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 4.14 ms | 4.12 ms | 4.17 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 1.99 s | 1.95 s | 2.00 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 2.71 ms | 2.68 ms | 2.97 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 481.5× faster than SymPy; 1.5× slower than Mathematica.

Script: `benchmarks/website/engineering.py`

### Forward kinematics + Jacobian of a 6R arm (`robot_6r_jacobian`)

PUMA-560 Denavit–Hartenberg chain with symbolic joint angles and link parameters (a2, a3, d3, d4, d6): product of six 4×4 transforms, then the 3×6 position Jacobian. Checked: position and all 18 Jacobian entries against 30-digit mpmath differentiation of the numeric chain.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 3.70 ms | 3.69 ms | 3.92 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 117.8 ms | 117.1 ms | 117.9 ms | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 352 µs | 343 µs | 354 µs | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 775 µs | 763 µs | 812 µs | 5 | ok | matches reference (independent reference) |

Alkahest is 31.8× faster than SymPy; 10.5× slower than SymEngine; 4.8× slower than Mathematica.

Script: `benchmarks/website/engineering.py`

### Evaluate a 6R Jacobian at 1M configurations (`robot_jacobian_eval_1e6`)

The 18 entries of the PUMA-560 position Jacobian (derived once with SymPy, so every system compiles identical expressions) evaluated at 1,000,000 random joint configurations. Alkahest: compile_expr (one fused multi-output kernel) + numpy_eval, one thread; SymPy: lambdify(numpy, cse=True); SymEngine: Lambdify(LLVM, cse=True); Mathematica: packed-array evaluation. Compilation not timed. Checked: mean and sampled entries against mpmath.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 147.9 ms | 146.5 ms | 162.6 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 294.7 ms | 293.6 ms | 305.7 ms | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 131.4 ms | 129.4 ms | 132.0 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 1.07 s | 1.02 s | 1.41 s | 5 | ok | matches reference (independent reference) |

Alkahest is 2.0× faster than SymPy; 1.1× slower than SymEngine; 7.2× faster than Mathematica.

Script: `benchmarks/website/engineering.py`

## Physics

Curvature invariants of black-hole metrics, angular-momentum coupling coefficients, and special functions evaluated to a thousand digits.

### Kretschmann scalar, Schwarzschild metric (`kretschmann_schwarzschild`)

R_abcd R^abcd from the Schwarzschild metric in Boyer–Lindquist/Schwarzschild coordinates with symbolic M: inverse metric, Christoffel symbols, Riemann tensor, full contraction, simplified result. SymPy and Mathematica run the textbook component algorithm (Christoffel symbols, Riemann components and the scalar each simplified); Alkahest calls Metric.kretschmann(). Checked against the closed form 48M²/r⁶.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 1.12 ms | 1.12 ms | 1.16 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 489.1 ms | 482.9 ms | 494.8 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 63.3 ms | 62.4 ms | 63.8 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 436.7× faster than SymPy; 56.6× faster than Mathematica.

Script: `benchmarks/website/physics.py`

### Kretschmann scalar, Kerr metric (`kretschmann_kerr`)

R_abcd R^abcd from the Kerr metric in Boyer–Lindquist/Schwarzschild coordinates with symbolic M, a: inverse metric, Christoffel symbols, Riemann tensor, full contraction, simplified result. SymPy and Mathematica run the textbook component algorithm (Christoffel symbols, Riemann components and the scalar each simplified); Alkahest calls Metric.kretschmann(). Checked against the closed form of Henry (2000).

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 34.2 ms | 34.0 ms | 34.9 ms | 3 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | timeout > 120 s | — | — | 0 | timeout |  |
| Mathematica | 14.3.0 | timeout > 120 s | — | — | 0 | timeout |  |

Note: In a separate run with a 900 s limit, Mathematica took 316 s for this case (one Simplify call hit its internal 300 s TimeConstraint; the value was still correct). SymPy did not finish within 120 s. Mathematica did not finish within 120 s.

Script: `benchmarks/website/physics.py`

### All Clebsch–Gordan coefficients up to j = 3 (`clebsch_gordan_batch`)

Every ⟨j1 m1; j2 m2 | J M⟩ with j1, j2 ∈ {0, 1/2, …, 3} (2,408 coefficients), exactly (square roots of rationals). Checked: count, sum, sum of squares and a weighted sum against Racah's formula evaluated independently.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 11.1 ms | 11.0 ms | 11.1 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 322.9 ms | 322.1 ms | 324.3 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 799.0 ms | 784.5 ms | 819.0 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 29.2× faster than SymPy; 72.3× faster than Mathematica.

Script: `benchmarks/website/physics.py`

### 16 special-function values to 1000 digits (`special_functions_1000_digits`)

Γ(1/3), ζ(3), ζ(1/2), erf(2), erfc(5), J₀(10), J₃(7), Y₀(3), K₁(2), Ai(2), Ei(1), W(1), Li₂(1/3), ψ(1/4), Si(3), exp(π√163) at 1000 significant digits. Alkahest: rigorous Arb balls via interval_eval (radius checked to certify the digits); mpmath at dps=1000 (the engine behind SymPy's evalf); SymPy N(·, 1000); Mathematica N[·, 1000]. Checked: the first 990 digits of every value against mpmath at 1200 digits.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 59.5 ms | 59.4 ms | 60.5 ms | 5 | ok | matches reference (independent reference) |
| mpmath | 1.3.0 | 5.03 s | 5.01 s | 5.05 s | 3 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 5.02 s | 5.01 s | 5.03 s | 3 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 240.1 ms | 236.2 ms | 241.7 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 84.6× faster than mpmath; 84.3× faster than SymPy; 4.0× faster than Mathematica.

Note: mpmath and Arb cache constants such as π across calls; the warm-up run populates those caches for every system alike.

Script: `benchmarks/website/physics.py`

## Finance

Closed-form option greeks and SABR calibration derivatives derived symbolically, then compiled and evaluated over a million options.

### Black–Scholes: all greeks to 2nd order (`bs_greeks`)

European call written with erf; all 4 first (delta, vega, theta, rho) and 10 second-order partials (gamma, vanna, volga, charm, ...) in S, σ, T, r, unsimplified. Checked against 40-digit mpmath numerical differentiation.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 2.90 ms | 2.86 ms | 2.91 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 227.3 ms | 216.1 ms | 231.4 ms | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 460 µs | 457 µs | 463 µs | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 2.01 ms | 1.97 ms | 2.20 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 78.4× faster than SymPy; 6.3× slower than SymEngine; 1.4× slower than Mathematica.

Note: A small expression: this measures per-derivative overhead more than algorithmic scaling.

Script: `benchmarks/website/finance.py`

### SABR implied vol: gradient + Hessian (`sabr_hessian`)

Hagan et al. (2002) lognormal SABR implied-volatility formula; all 4 first and 10 second partials in α, β, ρ, ν (the model-calibration Jacobian and Hessian), unsimplified. Checked against 40-digit mpmath numerical differentiation.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 4.01 ms | 4.00 ms | 4.17 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 352.9 ms | 352.3 ms | 353.4 ms | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 704 µs | 696 µs | 732 µs | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 3.42 ms | 3.40 ms | 3.63 ms | 5 | ok | matches reference (independent reference) |

Alkahest is 88.0× faster than SymPy; 5.7× slower than SymEngine; 1.2× slower than Mathematica.

Script: `benchmarks/website/finance.py`

### Price + delta + gamma + vega for 1M options (`bs_grid_1e6`)

Four closed-form Black–Scholes outputs evaluated for 1,000,000 options with per-option S, K, T, σ (float64). Alkahest: compile_expr (one fused 4-output kernel) + numpy_eval, one thread; NumPy: hand-written vectorised code with scipy.special.erf; SymPy: lambdify(scipy/numpy, cse=True); SymEngine: Lambdify(LLVM, cse=True); Mathematica: packed-array evaluation. Compilation not timed. Checked: means and sampled values against mpmath.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 67.4 ms | 67.3 ms | 69.5 ms | 5 | ok | matches reference (independent reference) |
| NumPy | 2.2.6 | 63.3 ms | 62.9 ms | 63.5 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 76.5 ms | 73.3 ms | 84.2 ms | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 51.5 ms | 51.3 ms | 52.1 ms | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 1.55 s | 1.50 s | 1.62 s | 5 | ok | matches reference (independent reference) |

Alkahest is 1.1× slower than NumPy; 1.1× faster than SymPy; 1.3× slower than SymEngine; 23.0× faster than Mathematica.

Script: `benchmarks/website/finance.py`

## JIT / numeric

Symbolic expressions compiled to native code and evaluated over millions of points, against NumPy, SymPy's lambdify, SymEngine's LLVM backend and Mathematica.

### Evaluate a 24-term expression at 10M points (`jit_eval_1e7`)

A sum of 24 damped sin/cos/sqrt·log terms in x, y, z (~250 nodes) at 10,000,000 float64 points, single thread. Alkahest: compile_expr + numpy_eval; NumPy: the same expression as NumPy ufunc calls; SymPy: lambdify(numpy, cse=True); SymEngine: Lambdify(LLVM, cse=True); Mathematica: packed-array evaluation. Compilation not timed. Checked: mean and sampled values against mpmath.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 2.32 s | 2.28 s | 2.34 s | 5 | ok | matches reference (independent reference) |
| NumPy | 2.2.6 | 4.39 s | 4.32 s | 4.41 s | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 3.85 s | 3.83 s | 3.88 s | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 2.30 s | 2.25 s | 2.31 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 4.40 s | 4.26 s | 4.46 s | 5 | ok | matches reference (independent reference) |

Alkahest is 1.9× faster than NumPy; 1.7× faster than SymPy; 1.0× slower than SymEngine; 1.9× faster than Mathematica.

Script: `benchmarks/website/numeric.py`

### Same 10M-point evaluation, multithreaded Alkahest (`jit_eval_1e7_parallel`)

As above, but Alkahest uses numpy_eval_par (Rayon, 8 threads under the 8-CPU cap). NumPy and SymEngine have no built-in multithreaded path for this and run one thread; Mathematica's packed-array arithmetic uses its own threading where it applies.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 298.1 ms | 296.8 ms | 299.0 ms | 5 | ok | matches reference (independent reference) |
| NumPy | 2.2.6 | 4.43 s | 4.37 s | 4.46 s | 5 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 2.32 s | 2.31 s | 2.32 s | 5 | ok | matches reference (independent reference) |
| Mathematica | 14.3.0 | 4.50 s | 4.44 s | 4.57 s | 5 | ok | matches reference (independent reference) |

Alkahest is 14.9× faster than NumPy; 7.8× faster than SymEngine; 15.1× faster than Mathematica.

Note: Not a like-for-like thread count: this row shows what the parallel path buys.

Script: `benchmarks/website/numeric.py`

### Compile a 2000-term expression (`jit_compile_large`)

Time to turn a ~20,000-node expression (2000 damped oscillatory terms in x, y, z) into a native vectorised function, from an already-built expression. Alkahest: compile_expr (Cranelift); SymPy: lambdify(numpy, cse=True); SymEngine: Lambdify(LLVM, cse=True). Checked: the compiled function's value at one point against mpmath.

| System | Version | Median | Min | Max | Runs | Status | Check |
|---|---|---|---|---|---|---|---|
| Alkahest | 3.12.0 | 253.5 ms | 252.9 ms | 257.8 ms | 5 | ok | matches reference (independent reference) |
| SymPy | 1.14.0 | 6.87 s | 6.81 s | 6.96 s | 3 | ok | matches reference (independent reference) |
| SymEngine | 0.14.1 | 2.40 s | 2.36 s | 2.43 s | 5 | ok | matches reference (independent reference) |

Alkahest is 27.1× faster than SymPy; 9.5× faster than SymEngine.

Script: `benchmarks/website/numeric.py`

## Reproduce

```bash
# 1. Build/install alkahest with the features used here
maturin build --release --manifest-path alkahest-py/Cargo.toml \
  --features "egraph groebner parallel cranelift system-gmp"
pip install target/wheels/alkahest-*.whl sympy==1.14.0 symengine==0.14.1 mpmath numpy scipy
# 2. (optional) Wolfram Engine / Mathematica 14.3 with WolframKernel on PATH
# 3. Run everything (one benchmark process at a time), then render
MALLOC_ARENA_MAX=8 RAYON_NUM_THREADS=8 python benchmarks/website/run_all.py
# re-measure one case or one system and merge:
python benchmarks/website/run_all.py --cases expand_fateman --systems sympy
python benchmarks/website/run_all.py --render-only
```

On the reference machine every invocation was wrapped as `flock ~/.cache/alkahest-build/heavy.lock systemd-run --user --scope -q -p MemoryMax=24G -p CPUQuota=800% …` so that only one benchmark ran at a time.

