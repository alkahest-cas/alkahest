"""Phase 23 — Parallel simplification.

Demonstrates `simplify_par`, which simplifies expressions using a parallel
bottom-up traversal (Rayon) when the `parallel` feature is enabled, or falls
back to the sequential simplifier otherwise.

The output of `simplify_par` is identical to `simplify`; only performance
differs for large expressions.
"""

import time
import alkahest as ak
from alkahest import ExprPool, simplify, simplify_par

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

print("=" * 60)
print("Phase 23 — Parallel simplification demo")
print("=" * 60)

# ── 1. Correctness: par matches seq ──────────────────────────────────────
print("\n1. Correctness: simplify_par matches simplify")

test_cases = [
    ("x + 0", x + pool.integer(0)),
    ("x * 1", x * pool.integer(1)),
    ("0 * x", pool.integer(0) * x),
    ("x - x", x - x),
    ("x^0",   x ** 0),
    ("x^1",   x ** 1),
    ("2 + 3", pool.integer(2) + pool.integer(3)),
    ("(2 + 3) * x", (pool.integer(2) + pool.integer(3)) * x),
]

all_ok = True
for name, expr in test_cases:
    seq_result = simplify(expr).value
    par_result = simplify_par(expr).value
    ok = str(seq_result) == str(par_result)
    all_ok = all_ok and ok
    status = "✓" if ok else f"✗  seq={seq_result}  par={par_result}"
    print(f"  {name:30s}: {status}")

print(f"\n  All match: {all_ok}")

# ── 2. Large constant folding ─────────────────────────────────────────────
print("\n2. Large sum: fold 1 + 2 + … + 50 in one pass")
args = [pool.integer(i) for i in range(1, 51)]
big_sum = args[0]
for a in args[1:]:
    big_sum = big_sum + a

t0 = time.perf_counter()
seq = simplify(big_sum)
t_seq = time.perf_counter() - t0

t0 = time.perf_counter()
par = simplify_par(big_sum)
t_par = time.perf_counter() - t0

expected = 1275  # sum(1..50)
print(f"  seq result = {seq.value}  ({t_seq*1000:.2f} ms)")
print(f"  par result = {par.value}  ({t_par*1000:.2f} ms)")
print(f"  Both = {expected}: seq={str(seq.value)==str(pool.integer(expected))} par={str(par.value)==str(pool.integer(expected))}")

# ── 3. Large product with ones ────────────────────────────────────────────
print("\n3. x * 1 * 1 * … * 1  (20 ones) simplifies to x")
ones = [pool.integer(1)] * 20
factors = [x] + ones
prod = factors[0]
for f in factors[1:]:
    prod = prod * f

seq = simplify(prod)
par = simplify_par(prod)
print(f"  seq = {seq.value}")
print(f"  par = {par.value}")
print(f"  Both simplify to x: {str(seq.value) == str(x) and str(par.value) == str(x)}")

# ── 4. Mixed tree ──────────────────────────────────────────────────────────
print("\n4. Mixed tree: (x + 0) * (y * 1) + (2 * 3)")
zero = pool.integer(0)
one = pool.integer(1)
expr = (x + zero) * (y * one) + (pool.integer(2) * pool.integer(3))

seq = simplify(expr)
par = simplify_par(expr)
print(f"  seq = {seq.value}")
print(f"  par = {par.value}")
print(f"  Match: {str(seq.value) == str(par.value)}")

print("\nDone.")
