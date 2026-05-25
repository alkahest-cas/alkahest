# Gröbner Basis Demo — Alkahest vs SymPy
# System: Cyclic-5 (5 variables, lex order)
# A standard computer algebra benchmark from the PoSSo test suite.

import time
from alkahest import GroebnerBasis, ExprPool, symbol
import importlib.metadata

pool = ExprPool()
n = 5
vs = [symbol(f"x{i}", pool=pool) for i in range(n)]

# Build the cyclic-5 polynomial system
# Equations: sum of products of consecutive variables (cyclic shifts)
polys = []
for k in range(1, n + 1):
    acc = vs[0] - vs[0]         # start at zero
    for start in range(n):
        m = vs[start % n]
        for d in range(1, k):
            m = m * vs[(start + d) % n]
        acc = acc + m
    if k == n:
        acc = acc - 1           # last equation: product - 1 = 0
    polys.append(acc)

print(f"Cyclic-{n} polynomial system:")
print(f"  {len(polys)} equations in {len(vs)} variables")
print(f"  Monomial order: lex")

# ---

print()
print("Computing Gröbner basis with Alkahest...")
t0 = time.perf_counter()
gb = GroebnerBasis.compute(polys, vs, order="lex")
elapsed_ms = (time.perf_counter() - t0) * 1000

alk_ver = importlib.metadata.version("alkahest")
print(f"  ✓ Done in {elapsed_ms:.1f} ms  (alkahest {alk_ver})")
print(f"  Basis has {len(gb)} generators")
print()
print("Verification:")
print(f"  f₁ in ideal: {gb.contains(polys[0])}")
print(f"  f₂ in ideal: {gb.contains(polys[1])}")
print(f"  f₅ in ideal: {gb.contains(polys[4])}")
