# Guessing recurrences

`guess_holonomic` fits a P-recursive (holonomic) recurrence to the first terms
of a sequence, in exact rational arithmetic. It is the *guess* in guess-then-
prove: fit a recurrence to the terms you can compute, then certify it with
[`zeilberger`](telescoping.md) when the sequence has a hypergeometric summand.

```python
import alkahest as ak

motzkin = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511,
           41835, 113634, 310572, 853467, 2356779, 6536382, 18199284, 50852019]

guess = ak.guess_holonomic(motzkin)
guess.order, guess.degree   # (2, 1)
guess.coeffs                # ((-3, -3), (-5, -2), (4, 1))
guess.confirmed             # True
guess.surplus_terms         # 14
```

The coefficients read lowest-degree-first, so that answer is

```text
−(3n + 3)·M(n) − (2n + 5)·M(n+1) + (n + 4)·M(n+2) = 0
```

which is Motzkin's recurrence.

## The guard is the feature

An unguarded fitter is worse than no fitter. A recurrence of order `J` whose
coefficients have degree `D` has `U = (J+1)(D+1)` unknowns, and a homogeneous
linear system in `U` unknowns has a nonzero solution the moment it has fewer
than `U` independent equations — *whatever the numbers are*. So some recurrence
always fits, and a fit that used up all its data is not evidence, it is
interpolation.

`guess_holonomic` therefore fits a candidate only when the terms
**over-determine** it. A candidate needing `U` unknowns is tried only where the
terms supply at least `U + min_surplus` equations, and `min_surplus` defaults to
`U` itself — the data has to be twice what the ansatz needs. Everything below
that is skipped rather than fitted.

What survives is then reported with the evidence attached:

| Attribute | Meaning |
|---|---|
| `n_equations` | equations the terms provided |
| `equations_used` | independent equations the fit consumed (the matrix rank) |
| `surplus_terms` | equations that were *not* needed and agreed anyway |
| `dimension` | dimension of the solution space; `1` for a genuine fit |
| `untested_candidates` | lower `(order, degree)` candidates the terms could not test |
| `confirmed` | enough surplus **and** dimension exactly 1 |

`untested_candidates` is the minimality caveat, and it is the same discipline as
`ZeilbergerCertificate.order_is_minimal`. `0` means the returned order is the
smallest that fits anywhere in the bounds; anything higher means it is the
smallest among the candidates the data could *decide*, and a shorter relation
may be hiding in the ones it could not.

`surplus_terms` is the number to judge a guess by, and `evidence()` returns all
of it as a dict for logging next to the result. This is `relation_confidence`'s
discipline applied to sequences: a fit is judged against what the data can
actually support, rather than endorsed because the arithmetic came out even.

## What it refuses, and what `None` means

Two different negative answers, kept apart on purpose:

```python
ak.guess_holonomic(motzkin[:7])
# HolonomicError: E-HOLO-005 — 7 terms are not enough to test every recurrence
# in bounds …
```

Seven Motzkin terms give exactly the five equations needed to pin down the six
unknowns of an order-2, degree-1 ansatz. The fit would be exact, and the same
fit exists for *any* seven numbers. That is refused.

```python
ak.guess_holonomic(first_sixty_primes) is None   # True
```

The primes are not P-recursive. With sixty of them every `(order, degree)`
candidate inside the default bounds is over-determined and was actually tested,
so `None` here is a genuine negative that a search loop may record as one.

**`None` is returned only when the whole grid was swept with adequate surplus.**
If some candidates had to be skipped for lack of terms, the call raises
`E-HOLO-005` instead — including the message of how many terms the cheapest
skipped candidate needs. A loop that reads "not holonomic" off a grid it never
swept has closed a branch it never explored, and that failure has no symptom
later.

## Knobs

```python
ak.guess_holonomic(terms, max_order=4, max_degree=4, *,
                   start=0, min_surplus=None, check_evidence=True)
```

- `max_order`, `max_degree` bound the search. The sweep is order-major, so the
  order returned is the smallest one that fits within the bounds *and* that the
  terms were able to test.
- `start` is the index `n` that `terms[0]` stands for; the coefficient
  polynomials are polynomials in that `n`.
- `min_surplus` overrides the surplus demanded. `0` turns the requirement off
  while leaving the reporting intact.
- `check_evidence=False` fits every candidate regardless of surplus and returns
  the first fit with `confirmed` set honestly. It is the escape hatch, in the
  same role `check_precision=False` plays on `guess_relation` — useful when the
  candidate is going somewhere else to be checked, never a way to make a weak
  fit look strong.

Terms must be exact: Python `int` of any size, or `fractions.Fraction`. A
`float` is refused rather than converted, because every step after this one is
exact and would happily certify a recurrence for the sequence you rounded to.

## Guess, then prove

```python
guess = ak.guess_holonomic(terms)
if guess is not None and guess.confirmed:
    assert guess.holds_for(more_terms)          # exact, on data it never saw
    cert = ak.zeilberger(F, n, k, minimal=True) # …and now prove it
    cert.order == guess.order
    cert.order_is_minimal
```

`holds_for` re-checks the recurrence exactly against a longer list, and
`to_exprs(pool, n)` hands the coefficient polynomials to the rest of the library
— most usefully to compare against `ZeilbergerCertificate.coeffs` once the same
recurrence has been certified. A guessed order agreeing with a certified
*minimal* order is the pair of facts worth reporting.
