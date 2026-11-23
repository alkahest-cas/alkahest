"""Every ``pdsolve`` call in SymPy 1.14's ``sympy/solvers/tests/test_pde.py``
(and the equations of its ``classify_pde`` / ``checkpdesol`` tests), as
first-order problems in ``u(x, y)``.  SymPy's unknown ``f(x, y)`` is ``u``."""

SYMPY = [
    {"id": "sympy-solvefun", "eq": "u + ux + uy"},
    {"id": "sympy-ccoeff-hom-1", "eq": "2*u + ux + uy"},
    {"id": "sympy-ccoeff-hom-2", "eq": "4 + 3*ux/u + 2*uy/u"},
    {"id": "sympy-ccoeff-hom-3", "eq": "u + 6*ux + 7*uy"},
    {"id": "sympy-ccoeff-hom-4", "eq": "a*u + b*ux + c*uy", "params": "a b c"},
    {"id": "sympy-check-1", "eq": "a*u + b*ux + c*uy", "params": "a b c"},
    {"id": "sympy-check-2", "eq": "3*u + 2*ux + uy"},
    {"id": "sympy-check-3", "eq": "a*u + b*ux + 2*uy", "params": "a b"},
    {"id": "sympy-check-4", "eq": "x*u + ux + 3*uy"},
    {"id": "sympy-check-5", "eq": "2*u + ux + 3*uy"},
    {"id": "sympy-check-6", "eq": "u + ux + 3*uy"},
    {"id": "sympy-ccoeff-1", "eq": "-2*ux + 4*uy + 5*u - exp(x + 3*y)"},
    {"id": "sympy-ccoeff-2", "eq": "ux/u + uy/u + 1 - exp(x + y)/u"},
    {"id": "sympy-ccoeff-3", "eq": "2*u - ux + 3*uy + sin(x)"},
    {"id": "sympy-ccoeff-4", "eq": "u + ux + uy + x*y"},
    {"id": "sympy-ccoeff-5", "eq": "u + ux + uy + log(x)"},
    {"id": "sympy-all", "eq": "u + ux + uy + x**2*y"},
    {"id": "sympy-classify-4", "eq": "x*u + ux + 3*uy"},
    {"id": "sympy-classify-5", "eq": "x**2*u + x*ux + x*y*uy"},
    {"id": "sympy-classify-6", "eq": "y*x**2*u + y*ux + uy"},
    {"id": "sympy-varcoeff-1", "eq": "x*ux - y*uy + y**2*u - y**2"},
    {"id": "sympy-varcoeff-2", "eq": "x**2*u + x*ux + x*y*uy"},
    {"id": "sympy-varcoeff-3", "eq": "y*x**2*u + y*ux + uy"},
    {"id": "sympy-varcoeff-4", "eq": "exp(x)**2*ux + y"},
    {"id": "sympy-varcoeff-5", "eq": "exp(2*x)*uy + y*u - u"},
]

for _e in SYMPY:
    _e.setdefault("vars", "x y")
    _e.setdefault("source", "sympy-1.14 test_pde.py")
