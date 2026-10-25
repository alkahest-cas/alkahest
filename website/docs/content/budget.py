import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

# Bound the work a call may do; a trip raises BudgetExceededError
with ak.context(budget=ak.Budget(wall_ms=1000, max_steps=100_000)):
    r = ak.diff(ak.sin(x) * ak.cos(x), x)
print(r.value)  # (cos(x)^2 + (-1 * sin(x)^2))

# Export the derivation as Lean 4 source
print(ak.to_lean(r).splitlines()[0])  # import Mathlib.Tactic
