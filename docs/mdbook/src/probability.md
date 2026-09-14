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

There are **no joint distributions, no conditioning and no covariance** here. A
product of two variates raises `E-PROB-002` rather than guessing at a joint law.

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
asserts by choosing the name. There is no dependent case: the bivariate-normal
`−log(1 − rho²)/2` would need a joint-distribution type this library does not
have.

## See also

* [Error handling](./errors.md#probability-four-ways-not-to-answer) — the six
  `E-PROB-*` codes and which of them are verdicts
* [Vector calculus and quaternions](./vector-quaternion.md) — the other surface
  added in the same cycle
* [Stability policy](./stability.md) — what `experimental` commits to
