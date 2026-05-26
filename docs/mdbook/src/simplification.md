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

# Logarithm and exponential cancellation (branch-cut safe)
r = simplify_log_exp(exp(log(x)))   # → x  (with positive domain side condition)

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

When domain assumptions are known, pass them via `SimplifyConfig` (Rust) so conditional rewrites apply — e.g. `x > 0` enables `sqrt(x²) → x` instead of `|x|`:

```rust
// Rust API (experimental re-export)
use alkahest_cas::{simplify_with, SimplifyConfig, Predicate};

let mut config = SimplifyConfig::default();
config.assumptions = vec![Predicate::Gt(x, pool.integer(0))];
let r = simplify_with(expr, &pool, &rules, config);
```

The colored pass (`simplify/colored_egraph.rs`) runs a native layered union-find e-graph before the rule engine when assumptions are non-empty. The egglog backend is unchanged.

### Parallel simplification

```python
from alkahest import simplify_par

# Simplify a list of expressions concurrently (requires --features parallel)
exprs = [x**i for i in range(100)]
results = simplify_par(exprs)
```

## E-graph simplification

`simplify_egraph` uses equality saturation via [egglog](https://github.com/egraphs-good/egglog) to explore many equivalent forms simultaneously before committing to the best one via a cost function.

```python
from alkahest import simplify_egraph

# The e-graph can discover non-obvious equivalences
r = simplify_egraph(x * x - pool.integer(1))  # may factor or simplify
```

E-graph saturation is more powerful than rule-based simplification for some inputs but slower and has non-deterministic performance for complex expressions. See [E-graph saturation](./egraph.md) for configuration options.

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
