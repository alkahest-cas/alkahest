# Bessel, Airy, zeta and polylogarithm functions

Cylinder and spherical Bessel functions, the Airy functions, the Riemann and
Hurwitz zeta functions and the polylogarithm are registered primitives
(`alkahest_cas::specfun`). Conventions are DLMF's — also SymPy's, mpmath's and
Arb's.

| Python | node | DLMF |
|---|---|---|
| `bessel_j(nu, x)`, `bessel_y(nu, x)` | `J_ν`, `Y_ν` | 10.2 |
| `bessel_i(nu, x)`, `bessel_k(nu, x)` | `I_ν`, `K_ν` | 10.25 |
| `spherical_jn(n, x)`, `spherical_yn(n, x)` | `j_n`, `y_n` | 10.47 |
| `airy_ai`, `airy_bi`, `airy_ai_prime`, `airy_bi_prime` | `Ai`, `Bi`, `Ai′`, `Bi′` | 9.2 |
| `zeta(s)`, `hurwitz_zeta(s, a)` | `ζ(s)`, `ζ(s, a)` | 25.2, 25.11 |
| `polylog(s, z)` | `Li_s(z)` | 25.12 |

```python
import alkahest as ak
from fractions import Fraction
p = ak.ExprPool(); x = p.symbol("x"); nu = p.symbol("nu"); k = p.symbol("k")

ak.bessel_j(Fraction(1, 2), x)     # sqrt(2)*pi^(-1/2)*x^(-1/2)*sin(x)
ak.spherical_jn(1, x)              # x^-2*sin(x) - x^-1*cos(x)
ak.bessel_j(0, x)                  # bessel_j0(x) — one spelling per function
ak.diff(ak.bessel_j(nu, x), x)     # 1/2*(bessel_j(-1 + nu, x) - bessel_j(1 + nu, x))
ak.zeta(p.integer(4))              # 1/90*pi^4
ak.zeta(p.integer(-1))             # -1/12
ak.polylog(p.integer(-1), x)       # x*(1 - x)^-2
ak.sum_definite(k**-3, k, p.integer(1), p.pos_infinity())   # zeta(3)
ak.sum_definite(x**k/k**2, k, p.integer(1), p.pos_infinity())  # dilog(x), |x| < 1 stated
ak.sum_definite(x**k/ak.gamma(k + 1), k, p.integer(0), p.pos_infinity())  # exp(x)
ak.integrate(x*ak.bessel_j0(x), x) # x*bessel_j1(x)
ak.evaluate(ak.bessel_k(nu, x), {nu: 0.5 + 0j, x: 1 + 2j})  # complex value via Arb
```

## Exact forms

- **Half-integer order** Bessel functions are elementary; the constructors
  expand them by default (`expand=False` keeps the node; `expand_special`
  expands later, e.g. after `subs`). Coefficients come from the three-term
  recurrences over the rationals. Past order 200: `E-SPECFN-004`.
- **Integer order** nodes are canonicalised: `J_{−n} = (−1)ⁿJ_n`,
  `Y_{−n} = (−1)ⁿY_n`, `I_{−n} = I_n`, `K_{−ν} = K_ν`, `J_n(−x) = (−1)ⁿJ_n(x)`,
  `J_0(0) = I_0(0) = 1`, `J_ν(0) = I_ν(0) = 0` for `ν > 0`; `bessel_j(0, x)` and
  `bessel_j(1, x)` are `bessel_j0(x)` / `bessel_j1(x)`.
- **ζ**: `ζ(2n)` is a rational multiple of `π^{2n}` (Bernoulli numbers),
  `ζ(−n) = (−1)ⁿB_{n+1}/(n+1)`; odd `ζ(2n+1)` is kept as `zeta(2n+1)`.
  `ζ(1)` is refused (`E-SPECFN-006`).
- **Li_s**: `Li_2 = dilog`, `Li_1(z) = −log(1−z)`, `Li_0(z) = z/(1−z)`,
  `Li_{−n}` the Eulerian rational function, `Li_s(1) = ζ(s)`,
  `Li_s(−1) = −(1−2^{1−s})ζ(s)`.

## Derivatives

`d/dx` only: `J′_ν = (J_{ν−1} − J_{ν+1})/2` (same for `Y`),
`I′_ν = (I_{ν−1} + I_{ν+1})/2`, `K′_ν = −(K_{ν−1} + K_{ν+1})/2`,
`j′_n = −j_{n+1} + (n/x)j_n`, `Ai″ = x·Ai`, `∂_a ζ(s, a) = −s·ζ(s+1, a)`,
`Li′_s(z) = Li_{s−1}(z)/z`, `ζ′(s) = zeta_prime(s)` (exact at `0, −1, 2` and
the negative even integers; numeric through Arb's `acb_dirichlet_zeta_jet`),
`H′_ν = (H_{ν−1} − H_{ν+1})/2` for `hankel1`/`hankel2` (`J ± iY`, complex-valued;
half-integer orders expand to `exp(±ix)` forms). A derivative in the order or
in `s`, and the second derivative of `zeta`, raises `E-DIFF-001` — it has no
closed form, and is never silently dropped.

## Numerics

`f64` values are certified to 53 bits by Arb (`arb_hypgeom_bessel_*`,
`arb_hypgeom_airy`, `arb_zeta`, `arb_hurwitz_zeta`, `arb_polylog`); balls are
rigorous at the requested precision; complex points go through the `acb_*`
kernels. A non-real value (`Y_ν`/`K_ν` at `x < 0`, `Li_s(x > 1)`) or a pole is
declined by the real evaluators rather than replaced by a real part.

## Γ, ψ, polygamma and the named constants

`EulerGamma` (γ), `Catalan` (G) and `Glaisher` (A) are exact symbols
(`ak.euler_gamma(pool)`, `ak.catalan(pool)`, `ak.glaisher(pool)`): they print,
parse and render (`\gamma`, `G`, `A`), every numeric evaluator resolves them,
and no symbolic path folds them into a float.

```python
ak.simplify(ak.digamma(p.integer(1)))       # -EulerGamma (was the float -0.5772…)
ak.simplify(ak.digamma(p.rational(1, 2))) # -EulerGamma - 2*log(2)
ak.simplify(ak.trigamma(p.integer(1)))      # pi^2/6
ak.simplify(ak.trigamma(p.rational(1, 4)))  # pi^2 + 8*Catalan
ak.polygamma(2, p.integer(1))               # -2*zeta(3)
ak.diff(ak.trigamma(x), x)                  # polygamma(2, x)
eps = p.symbol("eps"); m = p.symbol("m")
ak.series(ak.gamma(eps), eps, p.integer(0), 2)
# 1/eps - EulerGamma + (EulerGamma^2/2 + pi^2/12)*eps + O(eps^2)
ak.series(m**(-2*eps) * ak.gamma(eps), eps, p.integer(0), 1)
# 1/eps - EulerGamma - 2*log(m) + O(eps)
```

Exact values: `Γ(n) = (n−1)!`, `Γ(m+½) = r·√π`; `ψ` at every integer shift of
`1, ½, ⅓, ⅔, ¼, ¾, ⅙, ⅚` (Gauss's digamma theorem); `ψ⁽ⁿ⁾` at integer and
half-integer points through `(−1)ⁿ⁺¹ n! ζ(n+1, a)`. `series` expands `Γ`, `ψ`
and `ψ⁽ⁿ⁾` at the poles `0, −1, −2, …` by shifting them off the pole
(`Γ(u) = Γ(u+k+1)/(u(u+1)⋯(u+k))`), so the coefficients are exact in γ, π and
`ζ(3)`, `ζ(5)`, …. `polygamma(1/2, x)` is `E-SPECFN-008`.

## Generalized hypergeometric `ₚFq`

```python
z = p.symbol("z")
F = ak.hyper([1, 1], [2], z)        # hyper(2, 1, 1, 1, 2, z); LaTeX {}_{2}F_{1}(1, 1; 2; z)
ak.expand_hyper(F)                  # -log(1 - z)/z
ak.expand_hyper(ak.hyper([-3, x], [Fraction(1, 2)], z))   # a cubic in z
ak.expand_hyper(ak.hyper([Fraction(1, 2), Fraction(1, 3)], [2], p.integer(1)))  # Gauss: Γ…/Γ…
ak.diff(F, z)                       # 1/2*hyper(2, 1, 2, 2, 3, z)
ak.eval_expr(F, {z: 0.5})           # 1.386294361… (Arb)
```

The node is `hyper(p, q, a1..ap, b1..bq, z)`; `d/dz` shifts every parameter by
one, a parameter derivative is refused, values come from Arb
(`arb_hypgeom_0f1/1f1/2f1/pfq`, real only). `expand_hyper` knows terminating
series, `₀F₀`, `₁F₀`, the `log`/`atanh`/`asin` cases of `₂F₁`, Gauss's sum at
`z = 1` (when `c − a − b` is a positive literal), `₁F₁(1; 2; z)`,
`₁F₁(½; 3/2; −w)` (erf) and `₀F₁` (cosh/sinh, Bessel `J`/`I`); other nodes
are kept.

## Not yet

The large-`x` asymptotic expansion of `J_ν` through `asymptotic_expand`,
and Taylor models for non-integer Bessel orders, `hurwitz_zeta`, `hyper` and
the spherical Bessel functions (integer-order `J/Y/I/K`, Airy, `zeta`,
`polylog`, `polygamma` and the orthogonal polynomials have them). For G11c: Stirling's series through
`asymptotic_expand`, `loggamma`, Γ duplication/reflection rewriting, ψ at
other rationals (needs Clausen values), complex `hyper` values, contiguous
relations and `₂F₁` transformations (Kummer's 24 solutions), Meijer G.
