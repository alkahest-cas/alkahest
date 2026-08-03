# Simplification

Alkahest provides two complementary simplification engines that operate on the same expression pool.

## Rule-based simplification

`simplify` applies a fixed set of algebraic rewrite rules until no more apply (fixpoint). It is fast, predictable, and always terminates.

```python
from alkahest import simplify

r = simplify(x + pool.integer(0))   # → x
r = simplify(x * pool.integer(1))   # → x
r = simplify(pool.integer(2) * pool.integer(3))  # → 6  (constant folding)
```

The default rule set covers:
- Identity and absorbing elements (`x + 0 → x`, `x * 1 → x`, `x * 0 → 0`)
- Constant folding (integer and rational arithmetic)
- Basic polynomial simplification (`x + x → 2*x`, `x² * x → x³`)
- Commutativity and associativity (normalized at construction)

### Domain-specific rule sets

```python
from alkahest import simplify_trig, simplify_log_exp, simplify_expanded

# Pythagorean identity and double-angle formulas
r = simplify_trig(sin(x)**2 + cos(x)**2)  # → 1

# Conservatively leaves branch-sensitive identities unchanged
r = simplify_log_exp(exp(log(x)))

# Expand products and collect like terms
r = simplify_expanded((x + pool.integer(1))**3)
```

### Customizing the rule set

```python
from alkahest import simplify_with, make_rule

# Add a custom rule: sin²(x) → 1 - cos²(x)
my_rule = make_rule("sin_sq_to_cos", lhs=sin(x)**2, rhs=pool.integer(1) - cos(x)**2)
r = simplify_with(expr, rules=[my_rule])
```

### Conditional simplification (colored e-graphs)

Branch-sensitive rewrites are opt-in. In Python, create an explicit experimental
context tied to one expression pool; `x > 0` then enables `sqrt(x²) → x`:

```python
from alkahest.experimental import Assumptions

assumptions = Assumptions(pool)
assumptions.refine(pool.gt(x, pool.integer(0)))
r = assumptions.simplify(sqrt(x**2))  # → x
```

The current fact language recognizes conjunctions of positive and non-zero
predicates. Unsupported predicates are retained for contradiction detection but
do not authorize a rewrite; a definitive contradiction raises
`AssumptionError` with code `E-SIMPLIFY-001`. Contexts never modify the
thread-local `context()` helper or global simplifier state.

Without a proven fact, branch-cut identities such as `sqrt(x²) → x`,
`exp(log(x)) → x`, and `log(a*b) → log(a) + log(b)` remain unchanged. Algebraic
cancelations such as `x/x → 1` and `x^0 → 1` still fire in the default
simplifier and record `NonZero` side conditions in the derivation log. The
colored pass runs after ordinary rule simplification and preserves repeated
terms and factors.

### Parallel simplification

```python
from alkahest import simplify_auto, simplify_par, simplify_redex, simplify_strategy

simplify_par(expr)      # fork-join: best on wide expressions
simplify_redex(expr)    # level-scheduled: best on deep ones, deterministic log
simplify_auto(expr)     # picks one of the two from the expression's shape
simplify_strategy(expr) # "fork_join" | "level_scheduled" | "sequential"
```

Each takes a single expression and returns the same result as `simplify`; only
the schedule differs. Builds without `--features parallel` — including the
default PyPI wheel — fall back to the sequential path, so all four are always
callable, and `simplify_strategy` then reports `"sequential"`.

Neither parallel scheduler dominates:

| | `simplify_par` | `simplify_redex` |
|---|---|---|
| Strategy | fork-join mirroring the sequential traversal | expression bucketed by height, one `par_iter` per level |
| Forks on | `Add`/`Mul` with ≥ 4 children | every node, regardless of type |
| Memo | sharded `DashMap` | flat `Vec<AtomicU32>` indexed by `ExprId` |
| Traversal | recursive (stack-refill trampoline for deep inputs) | iterative |
| Derivation log | order varies with thread count | deterministic |

Measured on 32 cores, best time over 1–32 threads:

| shape | sequential | `simplify_par` | `simplify_redex` |
|---|---|---|---|
| deep chain (2000 levels, width 1) | 38.7 ms | 23.1 ms | **5.5 ms** |
| wide sum, independent terms | 28.4 ms | **5.1 ms** | 10.3 ms |
| many medium chains | 115.6 ms | **19.2 ms** | 36.6 ms |
| wide sum over a shared DAG | 2.48 ms | 0.89 ms | **0.83 ms** |

Fork-join keeps each chain on one worker and wins on wide expressions through
cache locality. Level scheduling wins on deep ones, where fork-join finds no
wide node to fork on and runs essentially sequentially. At one thread the level
scheduler is faster on every shape measured. Reproduce with
`cargo run --release --features parallel --example simplify_three_way`.

Both are `experimental`: `alkahest_core::experimental::{simplify_par, simplify_redex}`.

## E-graph simplification

`simplify_egraph` uses equality saturation via [egglog](https://github.com/egraphs-good/egglog) to explore many equivalent forms simultaneously before committing to the best one via a cost function.

```python
from alkahest import simplify_egraph

# The e-graph can discover non-obvious equivalences
r = simplify_egraph(x * x - pool.integer(1))  # may factor or simplify
```

E-graph saturation is more powerful than rule-based simplification for some inputs but slower and has non-deterministic performance for complex expressions. See [E-graph saturation](./egraph.md) for configuration options (`EgraphConfig`, including `disjoint_schedule` for match-disjoint rule groups).

User-defined `PatternRule` sets can use discrimination-net indexing on the Rust side (`PatternRuleSet`, `simplify_with_pattern_rules`); the built-in `RewriteRule` engine still scans rules linearly.

## Choosing between the two

| Criterion | `simplify` | `simplify_egraph` |
|---|---|---|
| Speed | Fast, predictable | Slower, variable |
| Completeness | Fixed rule set | Equality saturation |
| Termination | Always | Configurable limits |
| Side conditions | Respected | Respected |
| Best for | Hot paths, cleanup | Difficult equalities |

For most workflows: use `simplify` (or a domain-specific variant) first. Reach for `simplify_egraph` when you need the system to discover a non-obvious equivalence.

## Collect and normalize

Two utility passes that sit between the two engines:

```python
from alkahest import collect_like_terms, poly_normal

# 2*x + 3*x → 5*x
r = collect_like_terms(pool.integer(2) * x + pool.integer(3) * x)

# Normalize to canonical polynomial form over given variables
r = poly_normal(x**2 + pool.integer(2) * x * y + y**2, [x, y])
```

## Complex constructors (experimental)

`alkahest.experimental` exposes symbolic `conjugate`, `re`, `im`, and principal
`arg`. These are symbolic-only: they are not registered for f64, ball, or JIT
evaluation.

Safe simplifications include involution of `conjugate`, real/integer literals
for `re`/`im`, `arg` of a strictly positive literal or `Domain.Positive`
symbol, and exact `arg(I)` / `arg(-I)`. Branch-sensitive cases such as
`conjugate(log(z))`, `arg(0)`, negative reals, and generic complex inputs stay
unevaluated. Principal Arg uses the conventional range `(−π, π]` with a cut on
the negative real axis; do not rewrite through `atan2`, `log`, or `sqrt` yet.
