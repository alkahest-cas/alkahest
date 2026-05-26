## Groebner basis: Cyclic-5 (lex)

Variables $x_0,\ldots,x_4$. For $k = 1,\ldots,5$, the $k$-th equation is the sum of all cyclic products of $k$ variables; the fifth equation is that product minus $1$:

$$
\sum_{i=0}^{4}\ \prod_{d=0}^{k-1} x_{(i+d)\bmod 5}
\;=\;
\begin{cases}
0 & k < 5 \\
1 & k = 5
\end{cases}
$$

Monomial order: **lex** ($x_0 > x_1 > \cdots > x_4$).

# ---

import time
import sympy as sp
from sympy import groebner

n = 5
xs = sp.symbols("x0:5")

polys = []
for k in range(1, n + 1):
    p = sp.Integer(0)
    for start in range(n):
        m = sp.Integer(1)
        for d in range(k):
            m = m * xs[(start + d) % n]
        p = p + m
    if k == n:
        p = p - 1
    polys.append(p)

print("Computing Groebner basis (SymPy)...")
t0 = time.perf_counter()
gb = groebner(polys, *xs, order="lex")
ms = (time.perf_counter() - t0) * 1000

print(f"Done in {ms:.1f} ms  (sympy {sp.__version__})")
print(f"Basis size: {len(gb)} generators")
print("Ideal membership checks:")
print(f"  f1 reduces to 0: {polys[0].reduce(gb)[1] == 0}")
print(f"  f5 reduces to 0: {polys[4].reduce(gb)[1] == 0}")
