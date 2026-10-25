import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")

# Exact solutions of a polynomial system (Gröbner bases)
print(ak.solve([x**2 + y**2 - 1, x - y], [x, y]))
# [{x: (sqrt(2) * (1/2)), y: (sqrt(2) * (1/2))},
#  {x: (-1/2 * sqrt(2)), y: (-1/2 * sqrt(2))}]
