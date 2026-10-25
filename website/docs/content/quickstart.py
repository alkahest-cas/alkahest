import alkahest as ak
from alkahest import ExprPool, cos, diff, exp, integrate, simplify_trig, sin

pool = ExprPool()
x = pool.symbol("x")

# Differentiate. Every result carries a derivation log in .steps
d = diff(sin(x**2), x)
print(d.value)  # (x * 2 * cos(x^2))

# Integrate, and ask how the answer was checked
r = integrate(x * exp(x), x)
print(r.value)  # (exp(x) * (x + -1))
print(r.verification["status"])  # exactly_verified

# Simplify a trigonometric identity
print(simplify_trig(sin(x) ** 2 + cos(x) ** 2).value)  # 1

# Parse from a string, then take a limit
e = ak.parse("sin(x)/x", pool, {"x": x})
print(ak.limit(e, x, pool.integer(0)).value)  # 1

# When there is no closed form, it refuses with a stable error code
try:
    integrate(exp(x**2), x)
except ak.IntegrationError as err:
    print(err.code)  # E-INT-004: proven non-elementary
