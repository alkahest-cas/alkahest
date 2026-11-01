# Rule engine

The rule engine underlies both `simplify` and the e-graph backend. Rules are the atomic units of algebraic knowledge.

## Anatomy of a rule

A `RewriteRule` has:
- A **name** — stable string identifier (used in derivation logs and Lean certificate output)
- A **LHS pattern** — an expression template with pattern variables
- A **RHS template** — the replacement
- Optional **side conditions** — predicates that must hold for the rule to fire

## Pattern syntax

Patterns are regular expressions with a subset of `ExprData` nodes used as wildcards. From the Python side, pattern variables are symbols whose names start with a **lower-case letter** (so `x` in a pattern is a wildcard too; `match_pattern(..., wildcards=False)` treats every symbol as a literal). `make_rule(lhs, rhs)` builds a rule from two expressions:

```python
from alkahest import ExprPool, make_rule, match_pattern, simplify_with, sin, cos, sqrt

pool = ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")

# Pattern variable — matches any subexpression
a = pool.symbol("a")

# Rule: sin(a)^2 + cos(a)^2 → 1
pythagoras = make_rule(sin(a)**2 + cos(a)**2, pool.integer(1))
```

Pattern variables capture any subexpression and must bind consistently: if `a` appears twice in the LHS it must match the same expression in both positions.

## Matching

`match_pattern` applies a pattern to an expression and returns all match substitutions:

```python
pattern = sin(a)**2 + cos(a)**2
matches = match_pattern(pattern, sin(y)**2 + cos(y)**2)
for subst in matches:
    print(subst)  # {'a': y} — dict mapping pattern-variable name → matched expr
```

The matcher is associative-commutative (AC): `a + b` matches `b + a`, and `a + b + c` matches any ordering. A pattern sum matches a whole sum node, not a subset of a longer one: the pattern above finds nothing in `sin(y)**2 + cos(y)**2 + x`.

## Built-in rule sets

The rule sets loaded by `simplify` and the domain-specific simplifiers are:

| Function | Rules |
|---|---|
| `simplify` | Arithmetic identities, constant folding, polynomial normalization — with one carve-out: no rule folds a product that contains a **literal zero raised to a negative power**, because `0 · 0⁻¹` has no value ([details](./simplification.md#the-literal-zero-carve-out)) |
| `simplify_trig` | Pythagorean identity, double-angle and half-angle formulas |
| `simplify_log_exp` | Log/exp cancellation (branch-cut safe subset) |
| `simplify_expanded` | Distributive expansion, like-term collection. A polynomial part the rules would expand completely is expanded in one sparse-polynomial product instead, logged as a single `expand_polynomial` step |

## Defining custom rules

```python
b, c = pool.symbol("b"), pool.symbol("c")

# Factoring rule: a*b + a*c → a*(b + c)
factor_out = make_rule(a*b + a*c, a*(b + c))

# simplify_with applies only the rules you pass (the default set is not included)
r = simplify_with(x*y + x*sin(y), [factor_out])
print(r.value)              # (x * (y + sin(y)))
print(r.steps[0]["rule"])   # pattern_rule
```

Custom rules from Python are recorded in derivation logs as `pattern_rule` steps. If you tag the rule with a Lean theorem name (via the Rust `PrimitiveRegistry` API), the corresponding step can be exported as a Lean proof term.

## Rule execution model

`simplify` applies rules in a fixpoint loop:

1. For each node in the expression (post-order traversal):
   - Try each rule in the rule set.
   - If a rule matches and its side conditions are satisfied, apply it, emit a `RewriteStep`, and restart the loop for the modified subtree.
2. Repeat until no rule fires in a full pass.

This is an inner-outer loop strategy rather than exhaustive bottom-up application. It is fast but not complete — some sequences of rewrites require rules to be applied in a specific order. The e-graph engine removes this ordering dependency.

## Side conditions

A built-in rule can carry a side condition checked against the matched substitution. `sqrt(x^2) → x` fires only when `x` is known to be non-negative:

```python
from alkahest import simplify

xp = pool.symbol("t", "positive")
print(simplify(sqrt(x**2)).value)    # sqrt(x^2)   — x is real, could be negative
print(simplify(sqrt(xp**2)).value)   # t
```

`make_rule` from Python takes no condition: a custom `make_rule(sqrt(a**2), a)` fires on every match, so its soundness is up to you. Side conditions that reference symbol domains are sound. They propagate into the derivation log as `SideCondition` entries and into Lean output as assumptions.
