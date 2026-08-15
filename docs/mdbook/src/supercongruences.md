# Supercongruences: sequences modulo `p^k`

A supercongruence is a claim about a P-recursive sequence at one index per
prime — Beukers' `A(p−1) ≡ 1 (mod p³)` for the Apéry numbers, or any of the
several hundred open ones that OEIS records as "checked up to p = 499".
Producing evidence for one means evaluating the sequence at that index for
every prime in a range.

Done the obvious way, that is expensive for a reason that has nothing to do
with the mathematics: `A(p−1)` is an integer with `Θ(p)` digits, and the
recurrence touches it `Θ(p)` times, so the cost is quadratic in `p` and the
answer — a residue mod `p⁴` — throws almost all of it away.

`ModularRecurrence` runs the recurrence in `ℤ/p^K` instead. Same relation, same
arithmetic, machine words throughout, `O(1)` memory:

```python
import alkahest as ak

# (n+2)³A(n+2) = (34n³+153n²+231n+117)A(n+1) − (n+1)³A(n)
apery = ak.ModularRecurrence(
    [[1, 3, 3, 1], [-117, -231, -153, -34], [8, 12, 6, 1]],
    [1, 5],
)

apery.value_mod(12, 13, 3)      # A(12) mod 13³
# 1
```

Coefficients are given lowest-degree first, one list per shift — the convention
[`guess_holonomic`](./guessing.md) already returns, so a fitted recurrence goes
straight in:

```python
motzkin = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511,
           41835, 113634, 310572, 853467, 2356779, 6536382, 18199284, 50852019]
guess = ak.guess_holonomic(motzkin)
rec = ak.ModularRecurrence(list(guess.coeffs), motzkin[: guess.order], start=guess.start)
rec.value_mod(200, 10007, 3)
```

## Sweeping

`supercongruence_sweep` is the loop, with the verdict bookkeeping attached:

```python
primes = [p for p in range(5, 400) if all(p % q for q in range(2, int(p**0.5) + 1))]
sweep = ak.supercongruence_sweep(apery, primes, k=3, expect=1)

sweep.holds          # True  — no counterexample in the range
sweep.n_tested       # 76
sweep.n_skipped      # 0     — every prime produced a residue
sweep.valuations()   # {3: 76}
sweep.sharp          # True
```

`holds` is falsification failing, not a proof, and the documentation says so in
those words. The one thing a sweep can *settle* is sharpness: `valuations()` is
the histogram of `v_p(LHS − RHS)`, and `sharp` is `True` when some prime hits
exactly the claimed exponent — so here `A(p−1) ≡ 1 (mod p⁴)` is **false**, and
the `p³` in Beukers' theorem is best possible rather than merely cautious.

`index` and `expect` are callables of `p`, so the shifted statements work too:

```python
ak.supercongruence_sweep(apery, primes, k=3, index=lambda p: p, expect=lambda p: 5)
```

## Singular indices

Stepping forward solves for the top term,

```text
S(n+J) = ( b(n) − Σ_{i<J} a_i(n)·S(n+i) ) / a_J(n),
```

and `a_J(n)` need not be invertible mod `p`. For Apéry, `a_2(n) = (n+2)³`
vanishes to order three at every `n ≡ −2 (mod p)` — which is exactly the index
the sweep above crosses when it asks for `A(p)` rather than `A(p−1)`. This is
where a naive implementation goes quietly wrong: `pow(a, -1, m)` raises for a
non-unit if you are lucky, and a hand-rolled inverse returns *something* if you
are not.

Alkahest measures the loss before it computes anything. A first pass evaluates
`v_p(a_J(n))` at every step — dropping to exact integer arithmetic at the rare
index a residue cannot decide — so the total loss `L` is known up front, and the
forward pass runs at working precision `k + L`. Each singular step then divides
numerator and denominator by `p^v` (checking, not assuming, that the numerator
is divisible), spending exactly the `v` digits the budget already bought.
`ModularEvaluation` reports the whole account:

```python
report = apery.evaluate([13], 13, 3)
report.residues()           # [5]
report.singular_indices()   # [11]
report.n_singular           # 1
report.working_precision    # 6  — three digits lost at n = 11, three requested
```

Three cases never produce a residue:

| Code | Cause |
|---|---|
| `E-HOLO-006` | the modulus is not a prime power the machine-word backend supports (`p` composite, `k = 0`, or `p**k >= 2**62`) |
| `E-HOLO-007` | a step does not determine its next term as a `p`-adic integer: `a_J(n) = 0` exactly there, or the sequence leaves `ℤ_p` — the harmonic numbers do, at `H_p = H_{p−1} + 1/p` |
| `E-HOLO-008` | `k + L` needs a modulus past `2**62` |

The last is a real limit, not a formality. Reaching `A(199)` at `p = 5` crosses
39 singular steps costing 141 digits between them, and 141 digits of `5` does
not fit a 64-bit word, so that call refuses rather than answering:

```python
apery.value_mod(199, 5, 1)
# HolonomicError: E-HOLO-008 — the 39 singular step(s) cost 141 digits of
# p-adic precision, so answering to p^1 needs a working modulus of 5^142,
# which is past the machine-word backend's ceiling of 2^62
```

The loss is intrinsic to running the recurrence over residues, not an artefact:
at a singular index the residues of the earlier terms genuinely do not
determine the next one, and only more precision recovers it. In the regime
these sweeps live in — one index per prime, at or near `p` — there are at most
one or two singular steps and the headroom is free.

`supercongruence_sweep` records `E-HOLO-007` and `E-HOLO-008` in `skipped()` and
carries on, because those are facts about one prime. `E-HOLO-006` is a fact
about the *call*, so it propagates — a list of composites must not come back
`holds=True` over zero primes.

## Binomial coefficients

`binomial_mod(a, b, p, k)` is the same workload from the other side, and is what
a closed form is spot-checked against:

```python
ak.binomial_mod(2 * 11 - 1, 10, 11, 3)   # Wolstenholme: 1
ak.binomial_mod(1_000_000, 3, 7, 4)      # 2261
ak.binomial_mod(5, 9, 7, 4)              # 0 — b > a
```

At `k = 1` this *is* Lucas' theorem; for prime powers it is the Andrew
Granville / Davis–Webb factorisation of `n!` into its `p`-free part. The
`p`-free factorial is taken by a product tree over blocks of `p` consecutive
integers rather than term by term, so the cost is `O(p·k³ + log_p(a)·p·k)` and
`a` far larger than `p` is the ordinary case rather than the hard one.
