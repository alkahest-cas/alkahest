## Groebner Basis: Katsura-4

A benchmark from **laser physics** modeling (Katsura, 1990).
Five quadratic equations in five unknowns, describing binary-spin configurations at equilibrium:

$$u_0 + 2\sum_{i=1}^{4} u_i = 1, \qquad \sum_{j=-4}^{4} u_{|j-\ell|}\,u_{|j|} = u_\ell \quad \ell = 1,\ldots,4$$

Computing the **lex-order Groebner basis** -- a standard PoSSo benchmark.

# ---

import time
import importlib.metadata
import alkahest as ak
from alkahest import GroebnerBasis, ExprPool

pool = ExprPool()
n = 4
vs = [pool.symbol(f"u{i}") for i in range(n + 1)]

def u(k):
    k = abs(k)
    return vs[k] if k <= n else None

# Build Katsura-4
polys = []
eq0 = vs[0]
for i in range(1, n + 1):
    eq0 = eq0 + pool.integer(2) * vs[i]
polys.append(eq0 - pool.integer(1))

for l in range(1, n + 1):
    eq = pool.integer(0) - vs[l]
    for j in range(-n, n + 1):
        a, b = u(j - l), u(j)
        if a is not None and b is not None:
            eq = eq + a * b
    polys.append(ak.simplify(eq).value)

ver = importlib.metadata.version("alkahest")
print(f"Katsura-{n}  |  {len(polys)} equations  {len(vs)} variables  lex order")
print()
print("Input system (first two equations):")
print(f"$$f_1 = {ak.latex(polys[0])}$$")
print(f"$$f_2 = {ak.latex(polys[1])}$$")
print()

t0 = time.perf_counter()
gb = GroebnerBasis.compute(polys, vs, order="lex")
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"[done]  alkahest {ver}")
print(f"        {elapsed_ms:.1f} ms   ({len(gb)} basis elements)")
print()
print("Membership check:")
for i, p in enumerate(polys, 1):
    print(f"   f{i} in ideal : {gb.contains(p)}")
