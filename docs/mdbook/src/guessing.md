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
| `basis` | every independent relation the terms admit; `basis[0]` is `coeffs` |
| `singular_indices` | indices inside the data where the leading coefficient vanishes |
| `untested_candidates` | lower `(order, degree)` candidates the terms could not test |
| `status` | one of `GUESS_STATUSES`; `means` glosses it |
| `confirmed` | `True` / `False` / `None` — see below |

`untested_candidates` is the minimality caveat, and it is the same discipline as
`ZeilbergerCertificate.order_is_minimal`. `0` means the returned order is the
smallest that fits anywhere in the bounds; anything higher means it is the
smallest among the candidates the data could *decide*, and a shorter relation
may be hiding in the ones it could not.

`surplus_terms` is the number to judge a guess by, and `evidence()` returns all
of it as a dict for logging next to the result. This is `relation_confidence`'s
discipline applied to sequences: a fit is judged against what the data can
actually support, rather than endorsed because the arithmetic came out even.

## The verdict is three-valued

`confirmed` is `True`, `False`, or `None`, and `status` names which:

| `status` | `confirmed` | what it means |
|---|---|---|
| `confirmed` | `True` | over-determined, unique, and non-singular in the data |
| `singular` | `None` | the operator vanishes identically at `singular_indices` |
| `underdetermined` | `None` | several independent relations; read `basis` |
| `unconfirmed` | `False` | the fit consumed the equations that would have confirmed it |

Neither `False` nor `None` is a pass, and they are different: `False` is *the
data says nothing about this fit*, `None` is *the relation holds and is still
not the sequence's recurrence*. This is `relation_confidence`'s `credible` and
`NoveltyVerdict.found` for sequences — the same three values and the same rule
that only the first is a result.

`GUESS_STATUSES` is the closed vocabulary and `GUESS_STATUS_MEANINGS` glosses
each entry; `guess.means` is the gloss for the one at hand.

## Singular indices, and the corrupted term

This is the failure `surplus_terms` and `dimension` do not catch. A single
wrong term in an otherwise clean sequence does not stop a fit — it is absorbed:

```python
spoiled = motzkin_71_terms.copy()
spoiled[30] += 1                       # one typo

fit = ak.guess_holonomic(spoiled)      # default max_degree = 4
fit.order, fit.degree                  # (2, 4) — one degree up from the truth
fit.dimension, fit.surplus_terms       # (1, 55)  — every number still perfect
fit.singular_indices                   # (28, 29, 30)
fit.status, fit.confirmed              # ('singular', None)
```

The fit is the true operator multiplied by the cubic `(n−28)(n−29)(n−30)`,
which vanishes at exactly the three indices whose equations the typo breaks.
Every coefficient polynomial vanishes there at once, so those three equations
read `0 = 0` and constrained nothing — and the relation that comes back
satisfies every equation the terms supplied. **No re-check can catch this**: it
holds on the clean sequence too, being a left multiple of the true operator.
The roots inside the data are the only tell, which is why they are a field.

Two typos need `max_degree=8` and produce six roots — the count scales with the
corruption, so the field is a diagnostic and not just a flag. The first move on
a non-empty `singular_indices` is to recompute the terms at those indices.

`ModularRecurrence.value_mod` meets the same phenomenon and *refuses*
(`E-HOLO-007`), because a modular evaluation genuinely cannot step through a
singular index. A fit can be returned and flagged, because the relation is true
on the data — it is only untrue that it is the sequence's recurrence.

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
  the first fit with `status` set honestly. It is the escape hatch, in the
  same role `check_precision=False` plays on `guess_relation` — useful when the
  candidate is going somewhere else to be checked, never a way to make a weak
  fit look strong.

Only `status == "unconfirmed"` — a fit with no surplus left — is refused.
`"singular"` and `"underdetermined"` are *returned*, carrying the reason: in
both the relation genuinely holds on the terms, so there is something for the
caller to act on. `dimension > 1` used to raise, which made the whole
`(order, degree)` cell unusable on sequences whose annihilator is narrower than
the probe that reached them first.

Terms must be exact: Python `int` of any size, or `fractions.Fraction`. A
`float` is refused rather than converted, because every step after this one is
exact and would happily certify a recurrence for the sequence you rounded to.

## Guess, then prove

```python
guess = ak.guess_holonomic(terms)
if guess is not None and guess.confirmed is True:
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
