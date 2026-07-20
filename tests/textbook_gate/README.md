# The textbook gate

A curated suite of first-course calculus/algebra identities — the things a
student would type in week one of a calculus or algebra class: `d/dx x^n`,
`∫ 1/x dx`, `Σk from 1 to n`, `sin²x+cos²x`, factoring a quadratic. Advanced
subsystems (Risch, Gröbner, e-graphs, CAD) can and should keep growing, but
none of that matters to a user if `sum_definite(k, k, 1, 10)` can't compute
55. This suite exists to make base-case regressions loud instead of silent,
and to make it visible when a previously-broken base case gets fixed.

Background: `temp-alkahest/planning/report7-20.md` (2026-07-20 usage eval)
found several base-case failures — a fresh-interpreter `parse()` crash, a
false "no elementary antiderivative" verdict on an elementary integrand,
`sum_definite` rejecting `Σk`, invalid Lean certificates, `solve(numeric=True)`
not falling back past its degree-2 limit — while advanced features (WZ pairs,
CAD, gamma-hypergeometric sums, homotopy continuation) worked. That
mismatch — breadth outrunning the base — is what this gate is meant to catch
going forward.

## How it's organized

One file per topic, collected automatically by `pytest` (`testpaths = tests`
in `pytest.ini`). Shared verification helpers live in `tests/_tg_helpers.py`
(one level up, alongside `_step_logs.py`) and are importable as
`from _tg_helpers import ...` because the root `tests/conftest.py` puts
`tests/` on `sys.path`.

| File | Covers |
|---|---|
| `test_tg_derivatives.py` | Power/product/quotient/chain rule, standard functions |
| `test_tg_integrals_indefinite.py` | Standard antiderivatives, substitution, by parts, partial fractions |
| `test_tg_integrals_definite.py` | Definite integrals with known closed-form values |
| `test_tg_limits.py` | Standard limits (0/0, ∞/∞, one-sided, at infinity) |
| `test_tg_series.py` | Taylor/Laurent expansions of standard functions |
| `test_tg_sums.py` | Faulhaber sums, geometric series, telescoping |
| `test_tg_solve.py` | Linear/quadratic/cubic systems, both exact and numeric solving |
| `test_tg_polynomials.py` | Expand, factor, gcd, roots, division |
| `test_tg_simplify_trig.py` | Pythagorean/double-angle/reciprocal trig identities |
| `test_tg_algebra_identities.py` | log/exp inverses, `cancel`/`together`, exponent rules |

## Verification philosophy

**Never assert against alkahest's printed normal form.** `((x^2 + -1) *
(x + -1)^-1)` and `(x + 1)` are the same value in different clothes, and the
simplifier's choice of clothes is an implementation detail that will drift.
Instead:

- **Derivatives / limits / series**: evaluate numerically at several sample
  points and compare to a hand-computed closed-form reference (plain
  `math`/`cmath`, no SymPy dependency).
- **Indefinite integrals**: don't check the antiderivative's shape at all —
  differentiate what alkahest returns and check *that* matches the original
  integrand numerically. This is immune to `+C`, `log(x-2)` vs `-log(2-x)`,
  factoring order, etc., and is exactly the fundamental theorem of calculus.
- **Definite integrals / sums to a concrete bound**: the result is a single
  number — compare it directly to the known constant.
- **Symbolic-`n` sums**: substitute several concrete integer values of `n`
  and compare against a direct Python summation (`sum(range(...))`), same
  pattern the repo's own `tests/test_sum_v210.py` already uses.
- **`solve`**: substitute each returned solution back into the original
  equations and check the residual is ~0. Order- and form-independent.
- **Simplification identities**: check that simplifying preserves the
  expression's numeric value, and — for cases that fully collapse to a
  constant (`sin²x+cos²x → 1`) — assert that constant directly.

See `tests/_tg_helpers.py` for the concrete assertion helpers
(`assert_derivative_matches`, `assert_integral_self_consistent`,
`assert_series_matches_reference`, `assert_sum_closed_form`,
`assert_solutions_satisfy`, ...).

## Known-broken cases: `xfail(strict=True)`, not deletion

A first-course identity that alkahest currently gets wrong is not excluded
from the suite — it's marked:

```python
@pytest.mark.xfail(
    strict=True,
    reason="B4 (report7-20.md): sum_definite rejects Σk as 'not Gosper-summable'",
)
def test_sum_k_faulhaber():
    ...
```

`strict=True` is load-bearing: if the bug is later fixed, this test flips
from `xfail` to an *unexpected pass*, which pytest reports as a failure. That
failure is the signal — remove the `xfail` marker and the case becomes a
normal regression test. Never leave a case out of the suite just because it
currently fails; an absent case can't catch a fix or a further regression.

## Running it

```bash
pytest tests/textbook_gate/ -v
```

It's collected as part of the normal `pytest tests/` run — no special
markers or opt-in required (unlike `@pytest.mark.slow`).
