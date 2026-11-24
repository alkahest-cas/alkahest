"""Textbook and Kamke-style scalar ODEs for the ``dsolve`` regression suite.

Each entry is ``(id, category, equation, order, params)``. The equation is
``F(x, y, yp, ypp, yppp, ypppp) = 0`` in SymPy-compatible syntax (it parses
with both ``sympy.sympify`` and ``alkahest.parse``); ``params`` binds any
symbolic parameter to the value the numeric verification uses.

``test_ode_corpus.py`` runs the suite. The comparison script that scores
Alkahest against SymPy (and Mathematica) on the same list lives outside the
repository; its numbers are quoted in the pull request that added this file.
"""

from __future__ import annotations

CORPUS: list[tuple[str, str, str, int, dict[str, float]]] = [
    # -- constant coefficients: exact characteristic roots ---------------------
    ("cc01", "const-coeff", "yppp - y", 3, {}),
    ("cc02", "const-coeff", "ypppp + y", 4, {}),
    ("cc03", "const-coeff", "ypppp - y", 4, {}),
    ("cc04", "const-coeff", "yppp - 3*ypp + 3*yp - y", 3, {}),
    ("cc05", "const-coeff", "ypppp + 2*ypp + y", 4, {}),
    ("cc06", "const-coeff", "yppp + y", 3, {}),
    ("cc07", "const-coeff", "ypp - 2*y", 2, {}),
    ("cc08", "const-coeff", "ypp + yp + y", 2, {}),
    ("cc09", "const-coeff", "yppp - 2*y", 3, {}),
    ("cc10", "const-coeff", "ypppp + 4*y", 4, {}),
    ("cc11", "const-coeff", "yppp - 3*yp + y", 3, {}),
    ("cc12", "const-coeff", "ypppp - yp - y", 4, {}),
    ("cc13", "const-coeff", "yppp - yp", 3, {}),
    ("cc14", "const-coeff", "ypp - 4*yp + 13*y", 2, {}),
    # -- constant coefficients, forced -----------------------------------------
    ("cf01", "const-coeff-forced", "ypp + 4*y - sin(2*x)", 2, {}),
    ("cf02", "const-coeff-forced", "ypp - y - x*exp(x)", 2, {}),
    ("cf03", "const-coeff-forced", "yppp - y - exp(x)", 3, {}),
    ("cf04", "const-coeff-forced", "ypp + 2*yp + 5*y - exp(-x)*cos(2*x)", 2, {}),
    ("cf05", "const-coeff-forced", "ypp + y - x**2", 2, {}),
    ("cf06", "const-coeff-forced", "ypp - 4*yp + 4*y - x*exp(2*x)", 2, {}),
    ("cf07", "const-coeff-forced", "ypppp - y - cos(x)", 4, {}),
    ("cf08", "const-coeff-forced", "ypp + y - 1/cos(x)", 2, {}),
    ("cf09", "const-coeff-forced", "ypp + 9*y - x*sin(3*x)", 2, {}),
    ("cf10", "const-coeff-forced", "ypppp + 2*ypp + y - sin(x)", 4, {}),
    ("cf11", "const-coeff-forced", "ypp - 3*yp + 2*y - exp(x) - x", 2, {}),
    ("cf12", "const-coeff-forced", "ypp + y - exp(x)/(1 + exp(x))", 2, {}),
    # -- Euler–Cauchy -------------------------------------------------------------
    ("eu01", "euler", "x**2*ypp - 2*y", 2, {}),
    ("eu02", "euler", "x**2*ypp + x*yp + y", 2, {}),
    ("eu03", "euler", "x**2*ypp - 3*x*yp + 4*y", 2, {}),
    ("eu04", "euler", "x**3*yppp + x*yp - y", 3, {}),
    ("eu05", "euler", "x**2*ypp + x*yp - y - x", 2, {}),
    ("eu06", "euler", "x**2*ypp - 2*y - x**3", 2, {}),
    ("eu07", "euler", "x**2*ypp + 3*x*yp + y - log(x)", 2, {}),
    ("eu08", "euler", "x**2*ypp - x*yp + 2*y", 2, {}),
    ("eu09", "euler", "x**2*ypp + x*yp - 2*y", 2, {}),
    ("eu10", "euler", "x**3*yppp + 3*x**2*ypp - 2*x*yp + 2*y", 3, {}),
    # -- first order ----------------------------------------------------------------
    ("fo01", "first-linear", "yp + y*cos(x) - exp(-sin(x))", 1, {}),
    ("fo02", "first-linear", "yp - y/x - x", 1, {}),
    ("fo03", "first-linear", "yp - 2*x*y - x", 1, {}),
    ("fo04", "first-linear", "yp*cos(x) + y*sin(x) - 1", 1, {}),
    ("fo05", "first-homogeneous", "yp - (x + y)/x", 1, {}),
    ("fo06", "first-homogeneous", "yp - (x**2 + y**2)/(x*y)", 1, {}),
    ("fo07", "first-homogeneous", "yp - y/x - tan(y/x)", 1, {}),
    ("fo08", "first-homogeneous", "x*yp - y - sqrt(x**2 + y**2)", 1, {}),
    ("fo09", "first-homogeneous", "yp - (y/x)**2 - y/x", 1, {}),
    ("fo10", "first-exact", "2*x*y + (x**2 + 3*y**2)*yp", 1, {}),
    ("fo11", "first-exact", "exp(x)*sin(y) + exp(x)*cos(y)*yp", 1, {}),
    ("fo12", "first-separable", "yp - x*y**2", 1, {}),
    ("fo13", "first-separable", "yp - exp(x - y)", 1, {}),
    ("fo14", "first-separable", "yp - (1 + y**2)/(1 + x**2)", 1, {}),
    ("fo15", "riccati", "yp + y**2 - 2/x**2", 1, {}),
    ("fo16", "riccati", "yp - (y - x)**2 - 1", 1, {}),
    ("fo17", "riccati", "yp - y**2 + y/x + 1/x**2", 1, {}),
    ("fo18", "bernoulli", "yp + y - y**n", 1, {"n": 1.5}),
    ("fo19", "bernoulli", "yp - y - x*y**3", 1, {}),
    ("fo20", "bernoulli", "x*yp + y - x**2*y**2", 1, {}),
    ("fo21", "bernoulli", "yp + y/x - y**n/x", 1, {"n": 2.5}),
    ("fo22", "clairaut", "y - x*yp - yp**2", 1, {}),
    ("fo23", "clairaut", "y - x*yp - sqrt(1 + yp**2)", 1, {}),
    ("fo24", "lagrange", "y - 2*x*yp - yp**2", 1, {}),
    ("fo25", "first-integrating-factor", "y + (2*x - y**2)*yp", 1, {}),
    # -- second order, nonlinear -------------------------------------------------
    ("nl01", "y-absent", "ypp - yp**2", 2, {}),
    ("nl02", "y-absent", "ypp - x*yp**2", 2, {}),
    ("nl03", "y-absent", "x*ypp - yp - yp**2", 2, {}),
    ("nl04", "autonomous", "ypp - y*yp", 2, {}),
    ("nl05", "autonomous", "ypp - 2*y**3", 2, {}),
    ("nl06", "autonomous", "ypp - exp(y)", 2, {}),
    ("nl07", "autonomous", "y*ypp - yp**2", 2, {}),
    ("nl08", "autonomous", "y*ypp + yp**2", 2, {}),
    ("nl09", "autonomous", "ypp - 1/y**3", 2, {}),
    ("nl10", "autonomous", "y*ypp - 2*yp**2", 2, {}),
    ("nl11", "autonomous", "ypp + sin(y)", 2, {}),
    ("nl12", "autonomous", "ypp - 6*y**2", 2, {}),
    ("nl13", "autonomous", "2*y*ypp - yp**2 - 1", 2, {}),
    # -- second order, exact / reducible -------------------------------------------
    ("ex01", "second-exact", "x*ypp + 2*yp", 2, {}),
    ("ex02", "second-exact", "(1 - x**2)*ypp - 2*x*yp", 2, {}),
    ("ex03", "second-exact", "sin(x)*ypp + cos(x)*yp", 2, {}),
    ("ex04", "second-exact", "ypp + x*yp + y", 2, {}),
    ("ex05", "second-exact", "x*ypp + (x + 2)*yp + y", 2, {}),
    # -- classical polynomial families --------------------------------------------
    ("sp01", "legendre", "(1 - x**2)*ypp - 2*x*yp + 2*y", 2, {}),
    ("sp02", "legendre", "(1 - x**2)*ypp - 2*x*yp + 6*y", 2, {}),
    ("sp03", "legendre", "(1 - x**2)*ypp - 2*x*yp + 12*y", 2, {}),
    ("sp04", "hermite", "ypp - 2*x*yp + 4*y", 2, {}),
    ("sp05", "hermite", "ypp - 2*x*yp + 6*y", 2, {}),
    ("sp06", "laguerre", "x*ypp + (1 - x)*yp + y", 2, {}),
    ("sp07", "laguerre", "x*ypp + (1 - x)*yp + 2*y", 2, {}),
    ("sp08", "chebyshev", "(1 - x**2)*ypp - x*yp + 4*y", 2, {}),
    ("sp09", "chebyshev", "(1 - x**2)*ypp - x*yp + 9*y", 2, {}),
    # -- Bessel and Airy ---------------------------------------------------------------
    ("ba01", "airy", "ypp - x*y", 2, {}),
    ("ba02", "airy", "ypp + x*y", 2, {}),
    ("ba03", "airy", "ypp - (2*x + 1)*y", 2, {}),
    ("ba04", "bessel", "x**2*ypp + x*yp + (x**2 - 1)*y", 2, {}),
    ("ba05", "bessel", "x**2*ypp + x*yp + (x**2 - 1/4)*y", 2, {}),
    ("ba06", "bessel", "x**2*ypp + x*yp - (x**2 + 4)*y", 2, {}),
    ("ba07", "bessel", "x*ypp + yp + x*y", 2, {}),
    ("ba08", "bessel", "x**2*ypp + x*yp + (4*x**2 - 9)*y", 2, {}),
    ("ba09", "bessel", "x*ypp + y", 2, {}),
    ("ba10", "bessel", "x*ypp - yp + x*y", 2, {}),
    # -- added with the elliptic / special-quadrature / Lagrange round ------------
    ("nl14", "autonomous", "ypp - y + y**3", 2, {}),
    ("nl15", "autonomous", "ypp + y + y**3", 2, {}),
    ("nl16", "autonomous", "ypp - 3*y**2 + 2*y - 1", 2, {}),
    ("nl17", "autonomous", "ypp + 4*sin(y)", 2, {}),
    ("nl18", "autonomous", "ypp + 2*exp(-y)", 2, {}),
    ("nl19", "autonomous", "ypp + exp(2*y)", 2, {}),
    ("fo26", "lagrange", "y - x*yp**2 - yp", 1, {}),
    ("fo27", "lagrange", "y - x*(1 + yp) - yp**2", 1, {}),
    ("sp10", "hermite", "ypp - 2*x*yp + 2*y", 2, {}),
    ("sp11", "laguerre", "x*ypp + (1 - x)*yp + 3*y", 2, {}),
    ("ex06", "second-exact", "ypp - x*yp - y", 2, {}),
]

# Linear constant-coefficient systems ``d(vars)/dt = rhs`` for
# ``experimental.dsolve_system``: ``(id, vars, rhs)``.
SYSTEMS: list[tuple[str, list[str], list[str]]] = [
    ("sy01", ["u", "v"], ["v", "-u"]),
    ("sy02", ["u", "v"], ["u + v", "4*u + v"]),
    ("sy03", ["u", "v"], ["3*u - 4*v", "u - v"]),
    ("sy04", ["u", "v", "w"], ["v", "w", "u"]),
    ("sy05", ["u", "v"], ["v + t", "-u"]),
    ("sy06", ["u", "v"], ["-2*u + v", "u - 2*v"]),
    ("sy07", ["u", "v"], ["u - 5*v", "2*u - v"]),
]
