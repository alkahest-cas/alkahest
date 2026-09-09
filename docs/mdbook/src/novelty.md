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

`RecurrenceClaim.from_text` reads OEIS's own formula lines by recursive
descent over `+ - * / ^ ( )`, `n` and a shifted sequence term. It refuses —
returns `None`, never a guess — anything outside that shape: a sum, a
generating function, an inhomogeneous relation, a nonlinear one, or a relation
between *two* sequences (`a(n) = a(n-1) + A002026(n-1)` is a statement about
two of them and a recurrence for neither). A parser that guesses at prose
invents claims nobody made, so a line the parser does not fully cover is
counted as unusable rather than truncated into a shorter claim that happens to
parse.

The sequence need not be spelled `a(n)`. OEIS names a sequence after what it
counts, and an entry's **name** is where the recurrence lives for the entries
that are defined by one — A000045's whole name is *"Fibonacci numbers: F(n) =
F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1"*, and a filter reading only the
formula lines could not find the Fibonacci recurrence in the Fibonacci entry.
So `OeisEntry.candidate_lines()` puts the name first, any single letter may be
the sequence, an identifier passed as `names=` (the entry's own A-number) may
be too, and juxtaposition is read as multiplication (`2a(n-2)`). One line may
still only name **one** sequence, and — as always — a parsed line is only
indexed once it reproduces the entry's own terms, which is what stops a
comment's auxiliary `b(n)` from becoming a claim the entry never made.

Over a 377-entry live sample (970 → 1276 candidate lines, since the name and
the other notations are now candidates) that widening takes the parser from
156 lines read to 252, from 121 usable statements to 195, and from 114 entries
with at least one usable statement to 174.

### `q`-recurrences

`QRecurrenceClaim` is the same three things — normal form, hash, equality —
for `Σ_i c_i(q, q^n)·u(n+i) = 0`, what
[`q_zeilberger`](./telescoping.md) produces. Its coefficients are Laurent
polynomials in `q` and `q^n` over `ℚ` (rational functions are accepted and
cleared), so they are not polynomials in `n` at all and `RecurrenceClaim`
refuses them outright. The same four things are quotiented out, read over
`ℚ[q^±1, (q^n)^±1]`; note that the index shift now *acts* on the
coefficients, because `n → n+1` sends `q^n` to `q·q^n`. The normal form is
tagged `q-recurrence/1` where the ordinary one is tagged `recurrence/1`, so
the two hash spaces cannot collide.

```python
from alkahest.experimental.novelty import QRecurrenceClaim

claim = QRecurrenceClaim.from_recurrence(certificate, var=n, q=q)
claim.normal_form   # 'q-recurrence/1 (q^n - 1)*u(n+0) + (1)*u(n+1)'
```

**No source in this module can state a `q`-recurrence** — OEIS indexes integer
sequences — so `check_novelty` reports every OEIS source as `unavailable` for
a claim of this kind rather than manufacturing a `not_found` out of a search
that could not have matched. What it is good for today is the other half of
the job: a stable content address a loop can dedupe its own `q`-output with.

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

A `terms=` search is **paged**; an `ids=` lookup is not. `fmt=json` answers a
search with a bare list of at most ten results and no total count, so a single
full page is not evidence that there is nothing else: `OeisWeb` keeps asking
at `&start=` until a short page comes back (the search is over — the answer is
exhaustive, and is recorded in the cache as a complete answer) or until
`max_results` is reached (there may well be more — `exhaustive=False`, the
query is *not* recorded, and `check_novelty` reports `unavailable` rather than
`not_found`). An `id:A…` query asks for named entries and gets exactly them,
so it is exhaustive after one request.

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

`verdict.terms_check` is the other half of that honesty. `terms=` is used
twice: to identify the sequence to a source, and — since the two are supposed
to be about the same sequence — to re-check the claim itself, on the same
lenient trailing-window rule a source's own formula line has to pass. It reads
`"holds"`, `"fails"` or `"not_checked"`, and a `"fails"` means the lookup was
about a different sequence from the claim, so nothing it returned bears on the
claim: either the claim is wrong, the terms are, or `start` is (pass
`check_novelty(..., start=…)`, which means exactly what `holds_for`'s `start`
means and is never sent to a source).

## Testing without the network

`tests/test_novelty.py` never constructs `OeisWeb`; every OEIS-backed test
runs against `tests/data/oeis_novelty_fixture.json`, a cache recorded once
from oeis.org (© The OEIS Foundation Inc., licensed CC BY-NC-SA 4.0 — the
license travels with every cache this module saves) and committed. The
fixture carries the sequences this project already certifies recurrences for
— Apéry (A005259), Motzkin (A001006), Catalan (A000108), central binomial
coefficients (A000984) — plus A000045, where the recurrence is in the name,
and A359643, a result this project's own search found and which OEIS records
only as an unproved `Conjecture`: the recorded statement is
`verdict.hedged is True`, and a claim one order lower that OEIS does not have
at all comes back `"not_found"`.

The paging tests need raw HTTP pages rather than a cache, so
`tests/data/oeis_paging_fixture.json` holds recorded
`search?…&fmt=json` responses keyed `"query|start"` and the tests serve them
through a fake transport. `OeisWeb` is still never pointed at the network.
