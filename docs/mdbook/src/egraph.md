# E-graph saturation

The e-graph backend exposes a fundamentally different approach to simplification: rather than applying rules one at a time in a fixed order, it builds a structure that represents many equivalent expressions simultaneously, then extracts the best one.

## What is an e-graph?

An e-graph partitions expressions into *equivalence classes* (e-classes). When a rewrite rule fires, it does not replace the LHS — it adds the RHS to the same e-class as the LHS. At the end of saturation, an extraction step picks the cheapest representative from each e-class according to a cost function.

This eliminates the phase-ordering problem: rules can fire in any order without risk of committing to a suboptimal form. The e-graph remembers all explored forms and chooses among them at the end.

## Using the e-graph

```python
from alkahest import simplify_egraph, simplify_egraph_with

# Default configuration
r = simplify_egraph(expr)

# With explicit config
from alkahest import EgraphConfig, simplify_egraph_with

cfg = EgraphConfig(node_limit=10_000, iter_limit=20)
r = simplify_egraph_with(expr, cfg)
```

## Cost functions

The extraction step minimizes a cost function over e-class representatives. Three built-in cost functions:

| Name | Behavior |
|---|---|
| `SizeCost` | Prefers the expression with the fewest AST nodes |
| `DepthCost` | Prefers the shallowest expression tree |
| `OpCost` | Assigns per-operation costs; penalizes expensive ops |
| `StabilityCost` | Penalizes patterns that cause catastrophic cancellation |

`StabilityCost` is aware of numerical stability issues: it penalizes subtractive cancellation patterns and prefers numerically stable rearrangements.

## Configuration

The e-graph runs until saturation (no new e-class merges) or until a limit is hit:

- **`node_limit`** — maximum number of e-nodes. Once reached, saturation stops and extraction runs on the current state.
- **`iter_limit`** — maximum number of saturation rounds.

For large or complex expressions, saturation can be expensive. The rule-based `simplify` is often sufficient and should be preferred on hot paths.

## Rule sets in the e-graph

The e-graph uses the same `RewriteRule` objects as the rule-based engine. By default it loads the arithmetic rules. Domain-specific rules (trig, log/exp) are kept separate to avoid e-class explosions on expressions that do not involve those operations.

The default e-graph rule set includes trig identities (`sin²+cos²→1`) and safe log/exp cancellation. Disable per domain via `EgraphConfig(include_trig_rules=False)` or `include_log_exp_rules=False`.

### Match-disjoint scheduling

By default (`disjoint_schedule=True`), shrink and explore rules are split into match-disjoint egglog rulesets (`shrink-add`, `shrink-mul`, `shrink-pow`, `explore-trig`, `explore-log`, `explore-mul`) and run as separate `(run …)` steps within each phase. This reduces cross-rule interference during saturation. Set `disjoint_schedule=False` to use the legacy single-ruleset schedule.

## When e-graphs help

The e-graph is especially powerful when:

- Multiple non-obvious rewrites must be combined in a specific order that is hard to predict.
- The "right" form is not syntactically similar to the input (e.g. factoring followed by cancellation).
- You want the globally cheapest form under a custom cost function, not just any simplified form.

It is less useful when:

- The expression is already in near-canonical form and only identity cleanup is needed.
- You need predictable performance on a hot path.
- The expression is large and associative-commutative, where the e-graph can grow combinatorially.

## Colored e-graphs (conditional rewrites)

Separate from the egglog saturation backend, Alkahest implements a **native colored e-graph** (`simplify/colored_egraph.rs`) for conditional simplification under explicit assumptions (e.g. `x > 0 ⊢ sqrt(x²) → x`). When `SimplifyConfig::assumptions` is non-empty, `simplify_with` runs this pass before the rule engine. See [Simplification — conditional simplification](./simplification.md#conditional-simplification-colored-e-graphs).

## AC matching in the e-graph

The egglog backend handles associativity and commutativity structurally: `Add` and `Mul` children are sorted at pool-insertion time, so there is a single canonical ordering. The e-graph does not need to enumerate permutations.

This is more efficient than classical AC-completion but requires that the canonical ordering is established at construction, which the kernel enforces.
