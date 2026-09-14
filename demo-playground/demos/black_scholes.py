## Black–Scholes, derived

The Black–Scholes formula is usually *quoted*. Here it is **derived** — from the
risk-neutral expectation, in one call — and then checked three independent ways.

# ---

# Recorded with:
#   npx tsx cli/src/index.ts record --code demos/black_scholes.py \
#       --output demo-videos/black-scholes-derivation.webm --pace 2.5
import math

import alkahest as ak
import alkahest.experimental as ex
from alkahest import latex

pool = ak.ExprPool()
S, K, mu, Sigma = (pool.symbol(n) for n in ("S", "K", "mu", "Sigma"))
S0, r, sigma, T = (pool.symbol(n) for n in ("S0", "r", "sigma", "T"))
PI, half = pool.symbol("pi"), pool.rational(1, 2)


def value(expr, **at):
    """Evaluate a symbolic expression at a parameter point."""
    binding = {pool.symbol(k): float(v) for k, v in at.items()}
    binding[PI] = math.pi
    return ak.eval_expr(expr, binding)

# ---

## 1 — The model

Under the risk-neutral measure the terminal price is

$$S_T = S_0\,e^{(r-\sigma^2/2)T\;+\;\sigma\sqrt{T}\,Z},\qquad Z\sim\mathcal N(0,1)$$

so $S_T$ is log-normal. That is the *only* modelling assumption below.

# ---

lognormal = ex.LogNormal(mu, Sigma)
print("density of a log-normal:")
print(f"$${latex(lognormal.pdf(S))}$$")
print(f"support: {lognormal.support()}")
print(f"E[S] = $${latex(lognormal.mean())}$$")

# ---

## 2 — The price is an expectation

A European call pays $(S_T-K)^+$, so its price is the discounted expectation

$$C = e^{-rT}\,\mathbb E\big[(S_T-K)^+\big].$$

No formula is typed in. We ask for the expectation and see what comes back.

# ---

payoff = ak.max(S - K, pool.integer(0))
lemma = ex.expectation(payoff, S, lognormal)

print("E[(S-K)^+] for S ~ LogNormal(mu, Sigma):")
print(f"$${latex(lemma)}$$")

# ---

## 3 — What it will not do

Substituting the risk-neutral parameters *before* integrating leaves every
constant symbolic, and alkahest can then find no admissible parameter point at
which to check its own answer. It withholds the closed form rather than
returning one it could not verify.

# ---

try:
    ex.expectation(
        payoff, S,
        ex.LogNormal(ak.log(S0) + (r - half * sigma**2) * T, sigma * ak.sqrt(T)),
    )
except Exception as exc:                      # noqa: BLE001 — showing the refusal
    print(f"{type(exc).__name__}: {exc}")

# ---

## 4 — Specialise

Matching moments, $\mu=\log S_0+(r-\tfrac12\sigma^2)T$ and
$\Sigma=\sigma\sqrt T$. Substitute, discount, and the call price falls out.

# ---

call = ak.exp(-r * T) * ak.subs(
    lemma,
    {mu: ak.log(S0) + (r - half * sigma**2) * T, Sigma: sigma * ak.sqrt(T)},
)
print(f"$$C = {latex(call)}$$")

# ---

## 5 — Is it the textbook formula?

$C=S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2)$ — written out independently, from
`math.erf`, and compared. Alkahest never sees this formula.

# ---

def textbook(S0v, Kv, rv, sv, Tv):
    Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    d1 = (math.log(S0v / Kv) + (rv + sv * sv / 2) * Tv) / (sv * math.sqrt(Tv))
    return S0v * Phi(d1) - Kv * math.exp(-rv * Tv) * Phi(d1 - sv * math.sqrt(Tv))


POINTS = [(100, 100, 0.05, 0.20, 1.00), (80, 120, 0.01, 0.35, 0.50),
          (150, 90, 0.03, 0.15, 2.00), (42, 50, 0.07, 0.60, 0.25),
          (7, 7, 0.00, 0.90, 3.00)]

print(f"{'S0':>5}{'K':>5}{'r':>6}{'sig':>6}{'T':>6}{'derived':>16}{'textbook':>16}{'rel':>10}")
for pt in POINTS:
    got, want = value(call, S0=pt[0], K=pt[1], r=pt[2], sigma=pt[3], T=pt[4]), textbook(*pt)
    print(f"{pt[0]:>5}{pt[1]:>5}{pt[2]:>6}{pt[3]:>6}{pt[4]:>6}"
          f"{got:>16.10f}{want:>16.10f}{abs(got - want) / abs(want):>10.1e}")

# ---

## 6 — The Greeks are derivatives, not a second formula

$\Delta=\partial C/\partial S_0$. Differentiate the expression we just derived
and compare against $\Phi(d_1)$ — which, again, alkahest is never told.

# ---

delta = ak.diff(call, S0).value
gamma = ak.diff(delta, S0).value
theta = ak.diff(call, T).value

Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
print(f"{'S0':>5}{'K':>5}{'d(C)/d(S0)':>18}{'Phi(d1)':>18}")
for S0v, Kv, rv, sv, Tv in POINTS:
    d1 = (math.log(S0v / Kv) + (rv + sv * sv / 2) * Tv) / (sv * math.sqrt(Tv))
    got = value(delta, S0=S0v, K=Kv, r=rv, sigma=sv, T=Tv)
    print(f"{S0v:>5}{Kv:>5}{got:>18.12f}{Phi(d1):>18.12f}")

# ---

## 7 — And it solves the Black–Scholes PDE

The price came from an expectation. The PDE is a *separate* fact about it:

$$\frac{\partial C}{\partial T}=\tfrac12\sigma^2S_0^2\frac{\partial^2C}{\partial S_0^2}+rS_0\frac{\partial C}{\partial S_0}-rC$$

Every term here is a symbolic derivative of the derived price.

# ---

print(f"{'S0':>5}{'K':>5}{'call':>14}{'PDE residual':>16}")
for S0v, Kv, rv, sv, Tv in POINTS:
    at = dict(S0=S0v, K=Kv, r=rv, sigma=sv, T=Tv)
    residual = (value(theta, **at)
                - 0.5 * sv * sv * S0v * S0v * value(gamma, **at)
                - rv * S0v * value(delta, **at)
                + rv * value(call, **at))
    print(f"{S0v:>5}{Kv:>5}{value(call, **at):>14.8f}{residual:>16.2e}")

# ---

## 8 — Put–call parity

The put is the same construction with the payoff reversed. Parity,
$C-P=S_0-Ke^{-rT}$, is then a consequence, not an input.

# ---

put = ak.exp(-r * T) * ak.subs(
    ex.expectation(ak.max(K - S, pool.integer(0)), S, lognormal),
    {mu: ak.log(S0) + (r - half * sigma**2) * T, Sigma: sigma * ak.sqrt(T)},
)
print(f"$$P = {latex(put)}$$")
print()
print(f"{'S0':>5}{'K':>5}{'C - P':>16}{'S0 - K e^-rT':>16}{'diff':>10}")
for S0v, Kv, rv, sv, Tv in POINTS:
    at = dict(S0=S0v, K=Kv, r=rv, sigma=sv, T=Tv)
    lhs = value(call, **at) - value(put, **at)
    rhs = S0v - Kv * math.exp(-rv * Tv)
    print(f"{S0v:>5}{Kv:>5}{lhs:>16.10f}{rhs:>16.10f}{abs(lhs - rhs):>10.1e}")

# ---

## What just happened

One modelling assumption — log-normal terminal price — and one call to
`expectation`. Everything after it is a consequence: the closed form, the
Greeks, the PDE, parity. Each checked against a source that is not alkahest.
