# Propositional logic, minimisation and BDDs

Alkahest's propositional layer covers the Boolean side of circuit and EDA work:
n-ary connectives, normal forms, exact two-level minimisation, reduced ordered
binary decision diagrams (ROBDDs) and equivalence checking by truth table, BDD
or SAT.

```python
import alkahest as ak

p = ak.ExprPool()
a, b, c = p.symbol("a"), p.symbol("b"), p.symbol("c")

ak.And(a, b, c)                                  # n-ary, flattened, canonical order
ak.simplify_logic(ak.Or(ak.And(a, b), ak.And(a, ak.Not(b))))   # a
ak.to_cnf(ak.Or(a, ak.And(b, c)))                # (a ∨ b) ∧ (a ∨ c)

m = ak.minimize_logic(ak.Xor(a, b, c))
m.exact, m.terms, m.literals                     # (True, 4, 12)

f = ak.bdd(ak.Or(ak.And(a, b), ak.And(b, c), ak.And(a, c)))   # majority
f.sat_count(), f.node_count()                    # (4, 4)
f.exists([a]).to_expr()                          # b ∨ c

ak.equivalent(ak.Not(ak.And(a, b)), ak.Or(ak.Not(a), ak.Not(b)))   # True
ak.equivalent(ak.Implies(a, b), ak.Implies(b, a), counterexample=True)
# (False, {'a': ..., 'b': ...})
```

## Connectives

`And`, `Or` and `Xor` take any number of arguments; `Not`, `Implies(a, b)`,
`Equivalent(*args)` and `ITE(c, t, e)` complete the set.  `And`/`Or` flatten
nested arguments, fold constants and complementary pairs, deduplicate and sort,
so equal formulas built in different orders are the same `Expr`.  The kernel has
only `And`/`Or`/`Not` nodes, so the derived connectives return their definitions;
`Xor` keeps its size linear by sharing the running parity.

## Minimisation

`simplify_logic(expr, form="auto"|"dnf"|"cnf", dontcare=None, exact=None)`:

- **Exact** up to 12 variables (16 when `exact=True`): Quine–McCluskey prime
  implicants, then a minimum cover — fewest terms, then fewest literals — by
  Petrick's method run as branch-and-bound.  `minimize_logic(...).exact` is
  `True`.
- **Heuristic** beyond: an espresso-style expand/irredundant pass over the BDD of
  the function.  The cover is always equivalent, but not guaranteed minimum, and
  `exact` is `False`.  Ask for `exact=True` to get a refusal (`E-LOGIC-004`)
  instead of a heuristic answer.
- **Don't-cares**: pass a formula; the result agrees with `expr` wherever it is
  false.

Every result is checked against the function (by truth table or BDD) before it
is returned.  Arithmetic relations are accepted as opaque atoms, so the minimum
is propositional: `x > 0 ∨ x ≤ 0` is not reduced to `True`.

## BDDs

`bdd(expr, order=None, max_nodes=None)` builds the ROBDD; the default order is
depth-first over the canonical formula, which interleaves bit-sliced operands
(`a0 b0 a1 b1 …`).  BDDs combine with `& | ^ ~`, `implies`, `iff`, `ite`;
support `restrict`, `exists`, `forall`, `sat_count` (exact integer),
`satisfy_one`, `support`, `node_count` and `to_expr`.

Some functions have no small BDD under any order — the middle bit of an `n`-bit
multiplier grows exponentially.  The node ceiling (`max_nodes`, default
2 000 000) and the active `Budget` are both honoured: exceeding them raises
`LogicError` `E-LOGIC-003` or `BudgetExceededError`, never a partial diagram.
For such miters use `equivalent(f, g, method="sat")`.

## Equivalence

`equivalent(f, g, method="auto")` uses a truth table up to 16 variables, then a
BDD capped at 200 000 nodes, then DPLL on the miter `f ⊕ g`.  A counterexample is
checked to separate the two formulas.  With relations present, `True` is still a
proof, but a propositional difference is refused (`E-LOGIC-002`) because the
relations' truth values may be dependent.  `satisfiable` decides propositional
formulas of any width on their DAG.
