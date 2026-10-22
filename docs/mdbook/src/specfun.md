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

The Stieltjes constants `γₙ` (`ak.stieltjes_constant(pool, n)`, `n ≤ 30`;
`γ₀` is `EulerGamma`) are the same kind of symbol (`StieltjesGamma1`, …), and
`series(ak.zeta(s), s, p.integer(1), n)` is the Laurent series
`1/(s−1) + γ − γ₁(s−1) + γ₂/2 (s−1)² − …` in them. A series coefficient that
would contain a special function *at its pole* (`ζ(1)`, `Γ(0)`, `Y₀(0)`, …) is
refused, never returned.

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
`1, ½, ⅓, ⅔, ¼, ¾, ⅙, ⅚` in classical form, and at every other rational `p/q`
with `q ≤ 100` by Gauss's digamma theorem
(`−γ − log(2q) − (π/2)cot(πp/q) + 2Σ cos(2πkp/q) log sin(πk/q)`, with exact
trigonometric nodes); `ψ⁽ⁿ⁾` at every rational through
`(−1)ⁿ⁺¹ n! ζ(n+1, a)` — in `πˢ`/`ζ(s)`/Catalan/`√3` form at integers,
half-integers, quarters (`ψ″(¼) = −2π³ − 56ζ(3)`) and, for even `n`, thirds
and sixths; otherwise the exact `hurwitz_zeta(n+1, f)` value (odd `n` at
thirds needs a Clausen value, at quarters `β(n+1)`). `series` expands `Γ`, `ψ`
and `ψ⁽ⁿ⁾` at the poles `0, −1, −2, …` by shifting them off the pole
(`Γ(u) = Γ(u+k+1)/(u(u+1)⋯(u+k))`), so the coefficients are exact in γ, π and
`ζ(3)`, `ζ(5)`, …. `polygamma(1/2, x)` is `E-SPECFN-008`.

## `loggamma`, `gammasimp` and asymptotics at infinity (F25)

`loggamma(x)` is `log Γ(x)`: real for `x > 0`, and for complex `x` the
principal branch (the continuation from the positive axis, Arb's
`acb_lgamma`; at `x = −1/2` it is `log(2√π) − 2πi`, not `log(Γ(−1/2))`).
It is exact at integers up to 100 and at positive half-integers,
differentiates to `digamma`, and has f64, ball, complex and Taylor-model
values. A literal pole raises `E-SPECFN-006`.

`gammasimp` rewrites products of `Γ` with three identities (DLMF 5.5.1, 5.5.3
and 5.5.5). The recurrence turns `Γ(x+n)/Γ(x)` into a rising factorial for an
integer `|n| ≤ 64`, but only when the two exponents have opposite signs.
Reflection turns `Γ(a)Γ(m−a)` into `π/sin(πa)` times a rising factorial.
Duplication turns `Γ(a)Γ(a+½+k)` into `2^{1−2a}√π Γ(2a)` times a rising
factorial. Each rewrite is an identity of meromorphic functions. When the
`Γ` factors cancel completely (`Γ(x+3)/Γ(x) → x(x+1)(x+2)`), the result is the
continuous extension across `x ∈ {0, −1, …}`, where the quotient itself is
undefined. `beta(a, b)` is first written `Γ(a)Γ(b)/Γ(a+b)` (so `beta(x, 1)`
becomes `1/x`); `uppergamma`/`lowergamma` are left alone. `is_zero` tries `gammasimp`, so `Γ(x)Γ(1−x) − π/sin(πx)` is proved
zero.

```python
ak.loggamma(p.integer(5))                       # log(24)
ak.diff(ak.loggamma(x), x).value                # digamma(x)
ak.gammasimp(ak.gamma(x) * ak.gamma(1 - x))     # pi/sin(pi*x)
ak.gammasimp(ak.gamma(x) * ak.gamma(x + p.rational(1, 2)))  # 2^(1-2x)*sqrt(pi)*gamma(2x)
ak.gammasimp(ak.gamma(x + 3) / ak.gamma(x))     # x*(x+1)*(x+2)

from alkahest.experimental import asymptotic_expand
asymptotic_expand(ak.loggamma(x), x, 6)
# [x*log(x), -x, -log(x)/2, log(2*pi)/2, 1/(12*x), -1/(360*x^3)]
asymptotic_expand(ak.gamma(x), x, 3)
# [sqrt(2*pi)*x^(x-1/2)*exp(-x), …/(12*x), …/(288*x^2)]
asymptotic_expand(ak.bessel_j(0, x), x, 2)
# [sqrt(2)*cos(x - pi/4)/(sqrt(pi)*sqrt(x)), sqrt(2)*sin(x - pi/4)/(8*sqrt(pi)*x^(3/2))]
asymptotic_expand(ak.erfc(x), x, 2)      # [exp(-x^2)/(sqrt(pi)*x), -exp(-x^2)/(2*sqrt(pi)*x^3)]
```

`asymptotic_expand` builds these expansions at `+∞` from exact coefficients:

| function (argument `a·x + b`, rational `a > 0`) | expansion | DLMF |
|---|---|---|
| `loggamma`, `gamma^e`, `factorial`, `binomial`, products of `Γ` | Stirling, `B_{k+1}(b)` | 5.11.3, 5.11.8 |
| `digamma`, `trigamma`, `polygamma(n, ·)` | `log x + Σ (−1)^{m+1} B_m(b)/(m x^m)`, differentiated | 5.15.8 |
| `bessel_j/y(ν, a·x)`, rational `ν` | Hankel, `cos ω`/`sin ω` with `ω = ax − νπ/2 − π/4` | 10.17.3–4 |
| `bessel_i/k(ν, a·x)` | `e^{±ax}` times Hankel's `a_k(ν)` | 10.40.1–2 |
| `erfc(a·x)`, `Ei(a·x)` | `e^{−a²x²}`, `e^{ax}` times the classical series | 7.12.1, 6.12.2 |
| `uppergamma(s, a·x)`, rational `s` | `e^{−ax}(ax)^{s−1} Σ (s−1)⋯(s−k)/(ax)^k`; finite at integer `s > 0` | 8.11.2 |
| Airy `Ai`, `Bi`, `Ai′`, `Bi′` of `a·x` | `e^{∓ζ}`, `ζ = (2/3)(ax)^{3/2}` | 9.7.5–8 |

These also combine. Products with constants, with `log(x)^k`, with `x^q` and
with power-scale cofactors are expanded, and so are sums whose summands share
one exponential scale (for example `loggamma(x) − x·log(x)` or
`√x·J₀(x)`). Every expansion is checked numerically before it is returned:
at four large points, using 640-bit balls, the residual must be within 8
times the first omitted term. A symbolic order, a nonlinear or negative-slope
argument, a reciprocal such as `1/erfc(x)`, or a sum of two different scales
is refused with `E-ASYMPT-005`.

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
`₁F₁(½; 3/2; −w)` (erf) and `₀F₁` (cosh/sinh, Bessel `J`/`I`), the
`₂F₁(a, a + ½; c; z)` families for `c = ½, 3/2, 2a, 2a + 1` (A&S 15.1.9–14),
`₁F₁(a; 2a; z)` as Bessel `I`, `₁F₁(a; a+1; −w) = a w^{−a} lowergamma(a, w)`,
and any node one Euler, Pfaff or Kummer
transformation away from one of these (`₂F₁(1, 1; 3/2; z)`,
`₁F₁(1; 3/2; z) = eᶻ√π erf(√z)/(2√z)`); other nodes are kept.

Complex values (F26): `evaluate(F, {z: 3+2j}, mode="complex")` evaluates
every `ₚFq` through Arb's `acb_hypgeom_*`. `₂F₁` is continued analytically to
the whole cut plane; on the cut `[1, ∞)` itself the value is the limit from
below (`z − i0`, mpmath's convention: `₂F₁(1, 1; 2; 2) = −iπ/2`). Other
`p = q + 1` families (`₃F₂`, …) are summed as series only, so `|z| ≥ 1`
declines rather than guessing a continuation.

```python
ak.hyper_transform(ak.hyper([a, b], [c], z), "euler")   # (1-z)^(c-a-b) * 2F1(c-a, c-b; c; z)
ak.hyper_transform(F, "1/z")        # Γ-coefficient connection formula (A&S 15.3.7)
ak.hyper_transform(ak.hyper([a], [b], z), "kummer")     # exp(z) * 1F1(b-a; b; -z)
```

`hyper_transform(expr, kind)` returns the exact right-hand side of `"euler"`,
`"pfaff"` (`"z/(z-1)"`), `"pfaff_b"`, the two-term connection formulas
`"1-z"`, `"1/z"`, `"1/(1-z)"`, `"1-1/z"` (A&S 15.3.6–9, Gamma prefactors),
`"quadratic"` (DLMF 15.8.13, 15.8.15, 15.8.18 — whichever family `c = 2b`,
`c = a − b + 1`, `2c = a + b + 1` the parameters are in) and `"kummer"`
(`₁F₁`). Each holds on a stated region — `z ∉ [1, ∞)` for Euler/Pfaff,
`z ∉ (−∞, 0] ∪ [1, ∞)` for `1−z` / `1−1/z`, `z ∉ [0, ∞)` for `1/z` /
`1/(1−z)`, `|z| < 1` / `Re z < ½` for two of the quadratic ones; a literal
`z` outside it, the logarithmic case (`a − b` or `c − a − b` an integer),
and parameters in no quadratic family are `E-SPECFN-009`.

## Beta and incomplete gamma functions

```python
s, z, a = p.symbol("s"), p.symbol("z"), p.symbol("a")
ak.beta(p.integer(2), p.integer(3))     # 1/12; beta(a, 2) = 1/(a*(a + 1)); node otherwise
ak.uppergamma(s, z)                     # Γ(s, z) = ∫_z^∞ t^{s−1}e^{−t} dt; LaTeX \Gamma(s, z)
ak.uppergamma(p.rational(1, 2), z)      # sqrt(pi)*erfc(sqrt(z))
ak.lowergamma(p.integer(2), z)          # 1 - (z + 1)*exp(-z)
ak.diff(ak.uppergamma(s, z), z)         # -z^(s - 1)*exp(-z); d/ds refused (E-DIFF-001)
ak.eval_expr(ak.uppergamma(s, z), {s: 2.5, z: 1.5})   # 0.9305194427… (Arb)
```

`Γ(s, z) + γ(s, z) = Γ(s)`; integer orders are elementary, half-integer orders
reduce to `erfc`/`erf` by `Γ(s+1, z) = sΓ(s, z) + z^s e^{−z}`. Values come from
Arb (`arb_hypgeom_gamma_upper/_lower`, `arb_gamma`·`arb_rgamma` for `B`), real
only. The integrator uses them for Beta-family and incomplete-gamma definite
integrals (see *Calculus*).

## Not yet

Taylor models for non-integer Bessel orders, `hurwitz_zeta`, `hyper` and
the spherical Bessel functions (integer-order `J/Y/I/K`, Airy, `zeta`,
`polylog`, `polygamma` and the orthogonal polynomials have them). For G11c:
contiguous relations, the logarithmic cases of the `₂F₁` connection
formulas, `U(a, b, z)` ↔ upper incomplete gamma (no `U` primitive yet),
Meijer G.
