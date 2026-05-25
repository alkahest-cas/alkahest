# Gröbner Basis Demo — SymPy
# System: Cyclic-5 (5 variables, lex order)
# Same benchmark used in the Alkahest demo for a fair comparison.

import time
import sympy as sp
from sympy import groebner

n = 5
vs = sp.symbols(f"x0:{n}")

# Build the cyclic-5 polynomial system
polys = []
for k in range(1, n + 1):
    p = sp.Integer(0)
    for start in range(n):
        m = sp.Integer(1)
        for d in range(k):
            m = m * vs[(start + d) % n]
        p = p + m
    if k == n:
        p = p - 1               # last equation: product - 1 = 0
    polys.append(p)

print(f"Cyclic-{n} polynomial system:")
print(f"  {len(polys)} equations in {len(vs)} variables")
print(f"  Monomial order: lex")

# ---

print()
print("Computing Gröbner basis with SymPy...")
t0 = time.perf_counter()
gb = groebner(polys, *vs, order="lex")
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"  ✓ Done in {elapsed_ms:.1f} ms  (sympy {sp.__version__})")
print(f"  Basis has {len(gb)} generators")
print()
print("Verification:")
print(f"  Ideal contains input polynomials: {all(p in gb for p in polys)}")
