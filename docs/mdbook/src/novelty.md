# Novelty filtering against OEIS

A search loop over this library can rediscover a known identity within the
hour — the mathematics is not the hard part. The difference between "produced
400 certified recurrences" and "produced three that nobody had" is a filter
that puts every claim into a canonical form, hashes it, and asks whether it is
already written down somewhere *before* anything calls it a finding.
`alkahest.experimental.novelty` is that filter, for P-recursive recurrences
checked against [OEIS](https://oeis.org).

## Normal form and the hash

`RecurrenceClaim` takes a recurrence `Σ_i p_i(n)·u(n+i) = 0` and quotients out
everything that is presentation rather than content:

```python
from alkahest.experimental.novelty import RecurrenceClaim

# (n+1)·u(n+1) − (4n+2)·u(n) = 0 — central binomial coefficients
a = RecurrenceClaim([(-2, -4), (1, 1)])

# the same relation, scaled by −2, stated about u(n+7) and u(n+8)
b = RecurrenceClaim([(-60, -8), (16, 2)], offset=7)

a.claim_hash == b.claim_hash   # True
```

Four things are quotiented out: **scale** (multiplying every coefficient by a
nonzero rational — denominators are cleared and the integer content divided
out, sign fixed by making the first nonzero coefficient positive), **index
shift** (the window is moved to start at `u(n)`), **a common polynomial
factor** (`(n+1)·L` and `L` are the same recurrence up to the finitely many
`n` where the factor vanishes), and **zero-padding** at either end of the
window. What is *not* quotiented out: a genuinely different relation, a
different-order operator that happens to be a left multiple, or the range of
`n` a source claims the relation holds on — two sources stating the same
recurrence from different starting indices agree here, which is what a
novelty filter wants and is not a claim that the statements are
interchangeable at small `n`.

Build a claim straight from what [`zeilberger`](./telescoping.md) or
[`guess_holonomic`](./guessing.md) already produced:

```python
guess = ak.guess_holonomic(terms, max_order=3, max_degree=4)
claim = RecurrenceClaim.from_recurrence(guess)
```

`RecurrenceClaim.from_text` reads OEIS's own `a(n) = …` formula lines by
recursive descent over `+ - * / ^ ( )`, `n` and `a(n±k)`. It refuses — returns
`None`, never a guess — anything outside that shape: a reference to another
sequence (`a(n) = a(n-1) + A002026(n-1)`), a sum, a generating function, an
inhomogeneous relation, a nonlinear one. A parser that guesses at prose
invents claims nobody made, so a line the parser does not fully cover is
counted as unusable rather than truncated into a shorter claim that happens to
parse.

### `holds_for` / `confirmations`, and what `start` means

Both exactly re-check a claim's normal form against concrete terms —
`fractions.Fraction` arithmetic throughout, so `True` is a fact about the
terms given, not a tolerance. `holds_for` requires every window to check out;
`confirmations` counts only the **trailing** run, because a recurrence is
routinely stated only for `n` past some initial segment and a mismatch at
`n = 0` says nothing about whether it is the relation that was meant.

`start` is the true index of `terms[0]` — nothing more forgiving than that.
Because coefficients here are genuine polynomials in `n` (this is
P-recursive, not constant-coefficient), a wrong `start` is not a small numeric
error that shifts a few results: it evaluates every coefficient at the wrong
point and generically fails the *entire* array, trailing windows included,
even ones that never touch whatever made `start` wrong. If you slice, drop, or
prepend elements relative to some original indexing, adjust `start` by the
same amount — `[junk, *real]` needs `start=-1`, not the default `0`, because
`junk` sits where `u(-1)` would.

## Checking a claim

```python
from alkahest.experimental.novelty import OeisCache, OeisEntry, check_novelty

cache = OeisCache("my_oeis_cache.json")
verdict = check_novelty(claim, [cache], terms=terms[:12])
```

`sources` has **no default** — a check with an empty list, or no source able
to answer, is `unavailable`, and nothing here reaches for the network on your
behalf. Two source types:

- **`OeisCache`** — file-backed, offline. Holds entries *and* the queries
  already put to OEIS, keyed by what was asked — the second is what makes an
  honest negative possible: a cache that only stores hits can never
  distinguish "asked and OEIS had nothing" from "never asked", and reporting
  the second as the first is exactly the overclaim this module exists to
  prevent.
- **`OeisWeb`** — live lookup, **opt-in**: nothing constructs one for you. It
  serves from its own `OeisCache` before touching the network, sleeps between
  requests, sends an identifying User-Agent, and **returns `unavailable`
  rather than raising** when the network is not there.

```python
web = OeisWeb(cache=OeisCache())
web.lookup(ids=["A005259"])
web.cache.save("tests/data/oeis_novelty_fixture.json")
```

records a fixture once so later runs — and CI, which has no network guarantee
— never need to touch oeis.org at all.

## Reading the verdict

`NoveltyVerdict.found` is **three-valued**, in the manner of
[`relation_confidence`](./guessing.md)'s tri-state `credible` and
`GuessedRecurrence.confirmed`:

| `found` | `status` | Means |
|---|---|---|
| `True` | `"recorded"` / `"recorded_conjecturally"` | a source states this claim; `hedged` says whether as a theorem or a conjecture |
| `False` | `"not_found"` | the sources searched do not state it — **not** "novel" |
| `None` | `"unavailable"` | no source could answer; nothing was established either way |

There is deliberately no `novel` attribute anywhere on `NoveltyVerdict`, and
`bool(verdict)` **raises** rather than silently reading `True`, because
`if check_novelty(...):` is the exact sentence this module exists to prevent:

```python
bool(verdict)
# TypeError: a NoveltyVerdict has no truth value: `if verdict:` would read as
# 'is this novel?' and there is no such answer here. Test verdict.status (...)
# or verdict.found, which is True/False/None and whose False means 'not in
# the sources searched', not 'new'
```

`verdict.hedged` is the distinction the whole filter exists for. OEIS marks a
formula `Conjecture` or `Empirical` when it was fitted rather than proved —
restating a hedged recurrence is not a result, *proving* it is:

```python
verdict = check_novelty(recorded_claim, [cache], terms=terms[:12])
verdict.status    # "recorded_conjecturally"
verdict.hedged    # True — OEIS has this, but never proved it
```

`verdict.report()` carries the scope of the search — `entries_examined`,
`statements_compared`, `statements_unusable` — so the size of a negative is
visible next to it: a `"not_found"` against zero entries examined means
something quite different from one against fifty.

## Testing without the network

`tests/test_novelty.py` never constructs `OeisWeb`; every OEIS-backed test
runs against `tests/data/oeis_novelty_fixture.json`, a cache recorded once
from oeis.org (© The OEIS Foundation Inc., licensed CC BY-NC-SA 4.0 — the
license travels with every cache this module saves) and committed. The
fixture carries the sequences this project already certifies recurrences for
— Apéry (A005259), Motzkin (A001006), Catalan (A000108), central binomial
coefficients (A000984) — plus A359643, a result this project's own search
found and which OEIS records only as an unproved `Conjecture`: the recorded
statement is `verdict.hedged is True`, and a claim one order lower that OEIS
does not have at all comes back `"not_found"`.
