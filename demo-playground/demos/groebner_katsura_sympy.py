## Groebner Basis: Katsura-4

A benchmark from **laser physics** modeling (Katsura, 1990).
Five quadratic equations in five unknowns, describing binary-spin configurations at equilibrium:

$$u_0 + 2\sum_{i=1}^{4} u_i = 1, \qquad \sum_{j=-4}^{4} u_{|j-\ell|}\,u_{|j|} = u_\ell \quad \ell = 1,\ldots,4$$

Computing the **lex-order Groebner basis** -- a standard PoSSo benchmark.

# ---

import time
import sympy as sp
from sympy import groebner, reduced

n = 4
vs = sp.symbols(" ".join(f"u{i}" for i in range(n + 1)))

def u(k):
    k = abs(k)
    return vs[k] if k <= n else None

# Build Katsura-4
polys = []
eq0 = vs[0] + 2 * sum(vs[1:]) - 1
polys.append(eq0)
for l in range(1, n + 1):
    terms = [u(j - l) * u(j) for j in range(-n, n + 1)
             if u(j - l) is not None and u(j) is not None]
    polys.append(sum(terms) - vs[l])

print(f"Katsura-{n}  |  {len(polys)} equations  {len(vs)} variables  lex order")
print()
print("Input system (first two equations):")
print(f"$$f_1 = {sp.latex(polys[0])}$$")
print(f"$$f_2 = {sp.latex(polys[1])}$$")
print()
print(f"Computing Groebner basis with SymPy {sp.__version__}...")

t0 = time.perf_counter()
gb = groebner(polys, *vs, order="lex")
elapsed_s = time.perf_counter() - t0

print(f"[done]  sympy {sp.__version__}")
print(f"        {elapsed_s:.2f} s   ({len(gb)} basis elements)")
print()
print("Membership check:")
for i, p in enumerate(polys, 1):
    _, rem = reduced(p, list(gb), *vs, order="lex")
    print(f"   f{i} in ideal : {rem == 0}")
