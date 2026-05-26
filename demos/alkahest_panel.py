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
import importlib.metadata
from alkahest import GroebnerBasis, ExprPool, symbol

pool = ExprPool()
n = 5
xs = [symbol(f"x{i}", pool=pool) for i in range(n)]

polys = []
for k in range(1, n + 1):
    acc = xs[0] - xs[0]
    for start in range(n):
        m = xs[start % n]
        for d in range(1, k):
            m = m * xs[(start + d) % n]
        acc = acc + m
    if k == n:
        acc = acc - 1
    polys.append(acc)

print("Computing Groebner basis (Alkahest)...")
t0 = time.perf_counter()
gb = GroebnerBasis.compute(polys, xs, order="lex")
ms = (time.perf_counter() - t0) * 1000

ver = importlib.metadata.version("alkahest")
print(f"Done in {ms:.1f} ms  (alkahest {ver})")
print(f"Basis size: {len(gb)} generators")
print("Ideal membership checks:")
print(f"  f1 in ideal: {gb.contains(polys[0])}")
print(f"  f5 in ideal: {gb.contains(polys[4])}")
