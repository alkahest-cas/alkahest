# Rule engine

The rule engine underlies both `simplify` and the e-graph backend. Rules are the atomic units of algebraic knowledge.

## Anatomy of a rule

A `RewriteRule` has:
- A **name** — stable string identifier (used in derivation logs and Lean certificate output)
- A **LHS pattern** — an expression template with pattern variables
- A **RHS template** — the replacement
- Optional **side conditions** — predicates that must hold for the rule to fire

## Pattern syntax

Patterns are regular expressions with a subset of `ExprData` nodes used as wildcards. From the Python side, pattern variables are `Expr` objects whose names start with `?`:

```python
from alkahest import make_rule, match_pattern

pool = ExprPool()
x = pool.symbol("x")

# Pattern variable — matches any subexpression
pv = pool.symbol("?a")

# Rule: ?a + 0 → ?a
add_zero = make_rule("add_zero", lhs=pv + pool.integer(0), rhs=pv)
```

Pattern variables capture any subexpression and must bind consistently: if `?a` appears twice in the LHS it must match the same expression in both positions.

## Matching

`match_pattern` applies a pattern to an expression and returns all match substitutions:

```python
matches = match_pattern(sin(x)**2 + cos(x)**2, pattern)
for subst in matches:
    print(subst)  # dict mapping pattern variable → matched expr
```

The matcher is associative-commutative (AC): `a + b` matches `b + a`, and `a + b + c` matches any ordering.

## Built-in rule sets

The rule sets loaded by `simplify` and the domain-specific simplifiers are:

| Function | Rules |
|---|---|
| `simplify` | Arithmetic identities, constant folding, polynomial normalization — with one carve-out: no rule folds a product that contains a **literal zero raised to a negative power**, because `0 · 0⁻¹` has no value ([details](./simplification.md#the-literal-zero-carve-out)) |
| `simplify_trig` | Pythagorean identity, double-angle and half-angle formulas |
| `simplify_log_exp` | Log/exp cancellation (branch-cut safe subset) |
| `simplify_expanded` | Distributive expansion, like-term collection |

## Defining custom rules

```python
from alkahest import make_rule, simplify_with

pool = ExprPool()
x = pool.symbol("x")
a = pool.symbol("?a")
b = pool.symbol("?b")

# Commutativity of subtraction rewrite: a - a → 0
self_cancel = make_rule(
    "self_cancel",
    lhs=a + pool.integer(-1) * a,
    rhs=pool.integer(0),
)

# Apply the custom rule alongside the default set
r = simplify_with(expr, rules=[self_cancel])
```

Custom rules are recorded in derivation logs with the name you provide. If you tag the rule with a Lean theorem name (via the Rust `PrimitiveRegistry` API), the corresponding step can be exported as a Lean proof term.

## Rule execution model

`simplify` applies rules in a fixpoint loop:

1. For each node in the expression (post-order traversal):
   - Try each rule in the rule set.
   - If a rule matches and its side conditions are satisfied, apply it, emit a `RewriteStep`, and restart the loop for the modified subtree.
2. Repeat until no rule fires in a full pass.

This is an inner-outer loop strategy rather than exhaustive bottom-up application. It is fast but not complete — some sequences of rewrites require rules to be applied in a specific order. The e-graph engine removes this ordering dependency.

## Side conditions

A rule can carry a side condition checked against the matched substitution:

```python
# sqrt(x^2) → x  only when x is non-negative
sqrt_sq = make_rule(
    "sqrt_sq_nonneg",
    lhs=sqrt(a**2),
    rhs=a,
    condition="nonnegative",   # checked against the domain of ?a
)
```

Side conditions that reference symbol domains are sound: `sqrt_sq_nonneg` will only fire when `?a` is bound to a symbol with domain `positive` or `nonneg`. They propagate into the derivation log as `SideCondition` entries and into Lean output as assumptions.
