# Probability and information theory

**Experimental.** Everything on this page lives in `alkahest.experimental` and may
change in a minor release — see [Stability policy](./stability.md).

```python
import alkahest as ak
from alkahest import experimental as ex
```

## A law is a name plus symbolic parameters

A `Distribution` is not a sampler and not a table lookup. It carries its density,
its support, and the constraints that make it a distribution at all:

```python
pool = ak.ExprPool()
mu, sigma, x, t = (pool.symbol(n) for n in ("mu", "sigma", "x", "t"))

N = ex.Normal(mu, sigma)
N.kind            # 'Normal'
N.params()        # the symbolic parameters, in constructor order
N.support()       # ('real', None)
N.constraints()   # [sigma > 0]
```

The nine constructors are `Normal`, `LogNormal`, `Uniform`, `Exponential`,
`Gamma`, `Beta` (continuous) and `Bernoulli`, `Binomial`, `Poisson` (discrete).

A parameter that is a **number** outside its constraint raises `E-PROB-001`
immediately. A **symbolic** one cannot be decided, so it is not decided: it stays
on `constraints()` for the caller to discharge. That split — decide what is
decidable, publish what is not, assume nothing — is the pattern for the whole
module.

## Everything is a method on the law

There is one way to ask each question, and it is `dist.method(...)`, never
`method(dist, ...)`:

| Method | Returns |
|---|---|
| `mean()`, `variance()` | `E[X]` (the first *raw* moment — the first central moment is identically zero) and the second *central* moment |
| `moment(k)` | raw `E[X^k]` |
| `pdf(x)`, `cdf(x)`, `quantile(p)` | density, distribution function, inverse |
| `characteristic_function(t)` | `E[e^{itX}]` — **complex-valued**, see below |
| `moment_generating_function(t)` | `M_X(t) = E[e^{tX}]` |
| `cumulant_generating_function(t)` | `K_X = log M_X` |
| `probability_generating_function(z)` | `G_X(z) = E[z^X]` — lattice laws only |
| `cumulant(n)`, `factorial_moment(n)` | `κ_n = K^{(n)}(0)`, `E[X(X−1)…]` |
| `skewness()`, `excess_kurtosis()` | from **central** moments; the kurtosis is the *excess* one (`0` for a normal, not `3`) |
| `entropy(base=None)` | Shannon or differential, see below |

**Every derived quantity is checked against numerical quadrature of its own
defining integral before it is returned.** One the checker cannot confirm raises
`E-PROB-005` rather than arriving with a caveat. See
[Error handling](./errors.md#probability-four-ways-not-to-answer) for the four
distinct ways this surface declines.

### The characteristic function is complex

```python
phi = N.characteristic_function(t)          # exp(i·mu·t − sigma²t²/2)
ak.evaluate(phi, {mu: 0.0, sigma: 1.0, t: 1.0}, mode="complex")
```

The real evaluation modes return `value=None` with `status="unsupported"` rather
than dropping the imaginary part.

## Generating functions report their convergence strip

`M_X(t) = lambda/(lambda − t)` for an `Exponential` is true **only** on
`t < lambda`. Outside that strip the expression is still finite, still plausible
and still wrong. So:

* a decidable argument outside the strip raises `E-PROB-006`;
* an undecidable one is answered, with the strip published on
  `ex.prob_side_conditions()`;
* a law whose MGF is entire leaves that list empty — so **empty means checked**.

```python
lam = pool.symbol("lam")
ex.Exponential(lam).moment_generating_function(t)
ex.prob_side_conditions()      # ['lam - t > 0', 'lam > 0']
```

A `LogNormal` has no MGF at all: `E[e^{tX}]` is `+inf` for every `t > 0`. Asking
raises `E-PROB-006`, because a CAS that completes the square anyway returns a
clean closed form that is the value of no integral. `cumulant(n)` refuses there
for the same reason — `κ_n = K^{(n)}(0)` and there is no `K` — while `skewness()`
and `excess_kurtosis()`, which are built from *central* moments, are returned as
usual.

`G_X(z) = Σ_k z^k P(X = k)` is defined only for a law on the non-negative
integers. Asking a `Normal` for one raises `E-PROB-002` rather than returning the
formal `E[exp(X log z)]`, which is the *moment* generating function at `log z`
and says nothing about any `P(X = k)`.

## Expectations

```python
S, K = pool.symbol("S"), pool.symbol("K")

ex.expectation(ak.exp(S), S, ex.Normal(mu, sigma))
# exp(mu + sigma²/2)

ex.expectation(ak.max(S - K, pool.integer(0)), S, ex.LogNormal(mu, sigma))
# Black–Scholes: e^{mu+sigma²/2}·Φ(d₁) − K·Φ(d₂)
```

`expectation(f, var, dist)` reduces `E[f(X)]` to integrals the existing
integrator attempts. A divergence the convergence gate can see is reported as
`E-PROB-006` *before* the symbolic work, so the answer is "no value exists"
rather than "the integrator declined"; an integral that does not close raises
`E-PROB-003` **naming it**, never an unevaluated object dressed as an answer.

Two combination rules, and their names say what they need:

* `expectation_affine(expr, [(var, dist), …])` — linearity, which holds for any
  joint law, dependent or not;
* `variance_affine_independent(expr, [(var, dist), …])` — which does need
  independence, and says so.

`expectation_affine` still raises `E-PROB-002` on a product of two variates: a
list of marginals says nothing about their dependence. A joint law is declared
explicitly instead — see [Joint laws](#joint-laws) below.

### Closed-form expectations of common functions

Before the integrator runs, `expectation` tries a table of textbook
expectations, term by term after expansion. Every term must be in the table or
the general route runs instead; either way the answer is gated by quadrature.

| law | term | `E[term]` |
|---|---|---|
| any | `x**k`, `k` a non-negative integer | the raw-moment table |
| `Normal(mu, sigma)` | `x**k * exp(a*x)` | `exp(a*mu + a²sigma²/2)·E[Y**k]`, `Y ~ Normal(mu + a·sigma², sigma)` |
| `Normal(mu, sigma)` | `exp(a*x) * cos(w*x + phi)`, `sin` | `exp(a*mu + (a² − w²)sigma²/2)·cos(w(mu + a·sigma²) + phi)` |
| `Gamma(k, theta)`, `Exponential` | `x**s * exp(a*x) * log(x)**{0,1}` | `theta**s Γ(k+s)/Γ(k)·(1 − a·theta)**−(k+s)`, times `digamma(k+s) + log theta − log(1 − a·theta)` |
| `Beta(alpha, beta)` | `x**s * log(x)**{0,1}` | `Γ(alpha+s)Γ(alpha+beta)/(Γ(alpha)Γ(alpha+beta+s))`, times `digamma(alpha+s) − digamma(alpha+beta+s)` |
| `LogNormal(mu, sigma)` | `x**s * log(x)**{0,1}` | `exp(s·mu + s²sigma²/2)`, times `mu + s·sigma²` |

Each entry has a hypothesis (`k + s > 0`, `1 − a·theta > 0`, `alpha + s > 0`).
Decidably false is `E-PROB-006` — `E[1/X]` under `Gamma(1, theta)` is `+inf`;
undecidable is published on `prob_side_conditions()`.

### Binomial with a symbolic `n`, and the negative binomial

`Binomial(n, p)` accepts a symbolic `n`. It carries `n ≥ 0` on `constraints()`
and `n` integer on `prob_side_conditions()`; moments, generating functions,
cumulants, skewness and kurtosis come from closed forms in `n`, checked against
the finite defining sum at integer `n`. What needs to enumerate `{0…n}` — the
CDF, the entropy, a non-polynomial `expectation` — refuses.

`NegativeBinomial(r, p)` counts failures before the `r`-th success (SciPy's
`nbinom(r, p)`), `r > 0` real, `0 < p ≤ 1`; mean `r(1−p)/p`. Its MGF converges
for `t < −log(1 − p)` and its PGF for `|z| < 1/(1 − p)`, both reported like the
Exponential's strip.

## Joint laws

A joint law is a list of **mutually independent blocks**: `(var, Distribution)`
or `([vars], MultivariateNormal)`. Dependence exists only inside a
multivariate-normal block, through its covariance matrix.

```python
X, Y, Z = (pool.symbol(n) for n in "XYZ")
m1, m2, s11, s12, s22, x1, x2, t1, t2, a = (
    pool.symbol(n) for n in ("m1", "m2", "s11", "s12", "s22", "x1", "x2", "t1", "t2", "a"))
A, b = [[1, 1], [1, -1]], [0, 1]
mvn = ex.MultivariateNormal([m1, m2], [[s11, s12], [s12, s22]])

mvn.pdf([x1, x2])                       # (2π)^{-1} det(Σ)^{-1/2} exp(-½ qᵀΣ⁻¹q)
mvn.moment_generating_function([t1, t2])
mvn.linear_transform(A, b)              # N(Aμ + b, AΣAᵀ)
mvn.marginal([1])
mvn.conditional([1], [a]).mean()        # [m1 + s12*(a - m2)/s22]
mvn.quadratic_form_expectation(A)       # tr(AΣ) + μᵀAμ

blocks = [([X, Y], mvn), (Z, ex.Exponential(lam))]
ex.joint_expectation(X**2 * Y**2 * Z, blocks)
ex.covariance(X + Y, X - Y, blocks)     # s11 - s22
ex.joint_variance(2*X - 3*Y + Z, blocks)
```

* `MultivariateNormal(mean, cov)` checks that `cov` is symmetric and, by
  Sylvester's criterion, that no leading principal minor is a number `≤ 0`
  (`E-PROB-001`); a symbolic minor is carried as a side condition. A linear map
  whose image is degenerate (`A` without full row rank) is refused the same way.
* `joint_expectation` expands the expression and splits each term into
  per-block factors (`E[g(X)h(Z)] = E[g(X)]E[h(Z)]`). A univariate factor goes
  through `expectation`; a Gaussian factor must be a polynomial times `exp` of
  an affine form and is computed exactly by **Stein's identity**
  `E[X_i g(X)] = μ_i E[g] + Σ_j Σ_ij E[∂_j g]` and exponential tilting. A factor
  coupling two independent blocks non-separably (`sin(X + Z)`) is `E-PROB-002`.
* `covariance(e1, e2, blocks)` is `E[e1·e2] − E[e1]E[e2]`, and `joint_variance`
  is its diagonal — the correlated counterpart of `variance_affine_independent`.

`sum_independent([d1, d2, …])` returns the law of a sum of **independent**
variates where it stays in a closed family: Normals (means and variances add),
Gammas/Exponentials with a common scale, Poissons, Binomials/Bernoullis with a
common `p` (symbolic `n` allowed), negative binomials with a common `p`. A
"common" parameter is the same expression; anything else is `E-PROB-002`.

`order_statistic(dist, n, kind="max")` is the law of the maximum (or minimum)
of `n` iid copies, `n` literal or symbolic: `cdf`, `pdf`, a verified `mean` and
`variance` (tabled in closed form for `Uniform` and `Exponential`:
`E[max]` of exponentials is `(digamma(n+1) − digamma(1))/lam = H_n/lam`), and
`distribution()` where it is a named law (`min` of `Exponential(lam)` is
`Exponential(n·lam)`; the extremes of `Uniform(0, 1)` are `Beta(n, 1)`,
`Beta(1, n)`).

`order_statistic(dist, n, k=k)` is the **`k`-th smallest** (1-based) for *any*
continuous law with a closed-form CDF — `Normal`, `LogNormal`, `Uniform`,
`Exponential`, `Gamma` with an integer shape, `Beta` with integer parameters:

```text
f_(k)(x) = n!/((k−1)!(n−k)!) · F(x)^{k−1} (1−F(x))^{n−k} · f(x)
F_(k)(x) = Σ_{j=k}^{n} C(n,j) F(x)^j (1−F(x))^{n−j}        (literal n)
```

`n` and `k` may be symbolic for the density (a Gamma ratio, with `n > 0`,
`k`, `n` integer and `n − k + 1 > 0` published on `prob_side_conditions()`);
the CDF needs a literal `n` (otherwise it is an incomplete beta: `E-PROB-004`).
`U_(k:n)` of `Uniform(0, 1)` is `Beta(k, n − k + 1)`. The **moments** are the
expectation of `x · f_(k)(x)/f(x)` under the base law, computed by the
integrator and checked against quadrature: exact rationals for integer `Beta`
and `Gamma`, Rényi-type closed forms for `Exponential` (`E[X_(2:3)] = 5/(6λ)`),
`μ + σ·3/(2√π)` for the maximum of three Normals. Where the integrator declines
(a `Normal` with `n ≥ 4`, whose mean involves `arctan` constants) the moment is
`E-PROB-003`, never a guess; a symbolic `n` outside the `Uniform`/`Exponential`
extremes is `E-PROB-002`. A **discrete** base law is refused (`E-PROB-002`):
ties change the density formula.

## Entropy, KL divergence, cross-entropy

`Distribution.entropy()` is Shannon `H = −Σ p log p` on a discrete support and
**differential** `h = −∫ f log f` on a continuous one. They are not the same
quantity and are not unified behind one formula:

* `h` is not the limit of `H`;
* `h` is **not non-negative** — `Uniform(0, 1/2).entropy()` is `log(1/2) < 0`,
  returned rather than clamped;
* `h` is **not invariant under a change of variables** — it shifts by
  `E[log|dx/dy|]`, which is exactly the `mu` by which a `LogNormal`'s entropy
  exceeds the underlying `Normal`'s.

`base=None` is nats, the right default for symbolic work; `base=2` is bits.

`kl_divergence(p, q, base=None)` is `D(P‖Q)`, and it is **not a distance**: not
symmetric, no triangle inequality, and `+inf` whenever `P` charges a set `Q`
gives probability zero. That last case is *gated*, because the closed form does
not fail there — substituting `Uniform(0,1)` against `Uniform(0,1/2)` into
`log((b₂−a₂)/(b₁−a₁))` gives `−0.693`, a finite **negative** KL divergence, which
Gibbs' inequality forbids. Decidably non-nested supports raise `E-PROB-006`
naming the value as `+inf`; an undecidable containment is published on
`prob_side_conditions()` instead of assumed. `P` and `Q` must be the same family.

`cross_entropy(p, q)` is `H(P, Q)`, assembled as `H(P) + D(P‖Q)` from the two
tables and then checked against `−E_P[log q]`, which is neither of them — so the
identity is falsifiable rather than definitional.

`mutual_information_independent(p, q)` is `0`, under an independence the caller
asserts by choosing the name. The dependent case lives on the joint law:

* `MultivariateNormal.mutual_information(a, b)` is
  `−½ log det(I − Σ_AA⁻¹Σ_AB Σ_BB⁻¹Σ_BA)` between two disjoint index groups;
  for two scalar components with correlation `ρ` it is `−½ log(1 − ρ²)`, and
  the standard deviations cancel symbolically.
* `MultivariateNormal.entropy(base=None)` is `½ log((2πe)^d det Σ)` (differential,
  mean-free), and `MultivariateNormal.kl_divergence(other)` is
  `½[tr(Σ₁⁻¹Σ₀) + (μ₁−μ₀)ᵀΣ₁⁻¹(μ₁−μ₀) − d + log(det Σ₁/det Σ₀)]`.

These are exact algebra (a `d`-dimensional integral is outside the one-variable
quadrature gate); the tests check them against numpy/scipy, Monte-Carlo and a
2-D numerical integral, and against the quadrature-gated univariate
`kl_divergence` at `d = 1`.

### Entropy with no closed form: `entropy_asymptotic`

`Binomial(n, p).entropy()` with a symbolic `n`, `NegativeBinomial` and `Poisson`
have no closed form and `entropy` refuses them (`E-PROB-004`). `entropy_asymptotic(dist)`
returns the **labelled approximation** `½ log(2πe σ²) − γ₁²/12` (variance `σ²`,
skewness `γ₁`; error `O(σ⁻⁴)`), recorded under the derivation rule
`prob_entropy_asymptotic_edgeworth`. It is a separate function on purpose, so an
approximation is never returned in place of an exact value; use it where
`np(1−p)`, `r(1−p)/p²` or `λ` is large. Any other law is `E-PROB-002`.

`expectation` also closes `E[xᵏ e^{ax} cos(ωx+φ)]` and the `sin` version under a
`Normal` for `k ≤ 12`: the raw moment polynomial (the Hermite expansion) at the
complex tilted mean `μ + (a+iω)σ²`, split into real and imaginary parts.

## See also

* [Error handling](./errors.md#probability-four-ways-not-to-answer) — the six
  `E-PROB-*` codes and which of them are verdicts
* [Vector calculus and quaternions](./vector-quaternion.md) — the other surface
  added in the same cycle
* [Stability policy](./stability.md) — what `experimental` commits to
