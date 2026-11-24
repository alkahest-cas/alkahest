"""Textbook first-order problems: Debnath (*Nonlinear PDEs for Scientists and
Engineers*, ch. 2), Sneddon (*Elements of PDEs*, ch. 2) and Evans (*PDE*,
ch. 3) — linear, semilinear and quasilinear equations, Cauchy problems
(including Burgers with breaking), and fully nonlinear equations (complete
and singular integrals, Hamilton–Jacobi)."""

TEXTBOOK = [
    # --- linear, homogeneous -------------------------------------------
    {"id": "tb-transport", "eq": "ux + uy", "source": "Evans 2.1"},
    {"id": "tb-transport-ab", "eq": "a*ux + b*uy", "params": "a b", "source": "Debnath 2.3"},
    {"id": "tb-rotation", "eq": "y*ux - x*uy", "source": "Sneddon II.4"},
    {"id": "tb-euler-0", "eq": "x*ux + y*uy", "source": "Sneddon II.4"},
    {"id": "tb-euler-1", "eq": "x*ux + y*uy - u", "source": "Sneddon II.4"},
    {"id": "tb-euler-2", "eq": "x*ux + y*uy - 2*u", "source": "Debnath 2.4"},
    {"id": "tb-x2y2", "eq": "x**2*ux + y**2*uy", "source": "Sneddon II.4"},
    {"id": "tb-hyperbolic", "eq": "y*ux + x*uy", "source": "Debnath 2.4"},
    {"id": "tb-3d-transport", "eq": "ux + uy + uz", "vars": "x y z", "source": "Evans 3.2"},
    {"id": "tb-3d-euler", "eq": "x*ux + y*uy + z*uz", "vars": "x y z", "source": "Sneddon II.4"},
    {
        "id": "tb-3d-euler-3u",
        "eq": "x*ux + y*uy + z*uz - 3*u",
        "vars": "x y z",
        "source": "Sneddon II.4",
    },
    # --- linear, inhomogeneous / variable coefficients -------------------
    {"id": "tb-2x-y", "eq": "ux + 2*x*uy - y", "source": "Debnath 2.4"},
    {"id": "tb-damped", "eq": "ux + uy - u", "source": "Debnath 2.4"},
    {"id": "tb-forced-exp", "eq": "ux + 2*uy - u - exp(x)", "source": "Debnath 2.4"},
    {"id": "tb-xy-source", "eq": "x*ux + y*uy - x*y", "source": "Sneddon II.4"},
    {"id": "tb-3d-inhom", "eq": "ux + uy + uz - u", "vars": "x y z", "source": "Evans 3.2"},
    # --- semilinear ------------------------------------------------------
    {"id": "tb-riccati", "eq": "ux + uy - u**2", "source": "Debnath 2.5"},
    {"id": "tb-euler-u2", "eq": "x*ux + y*uy - u**2", "source": "Debnath 2.5"},
    # --- quasilinear (Lagrange) -------------------------------------------
    {"id": "tb-lagrange-sym", "eq": "(y - u)*ux + (u - x)*uy - (x - y)", "source": "Sneddon II.4"},
    {"id": "tb-lagrange-u-y", "eq": "u*ux + y*uy - x", "source": "Sneddon II.4"},
    {"id": "tb-burgers-general", "eq": "uy + u*ux", "source": "Debnath 2.6"},
    {"id": "tb-quasi-1", "eq": "ux + u*uy - 1", "source": "Debnath 2.6"},
    {"id": "tb-quasi-xyu", "eq": "x*ux + y*uy - x*exp(-u)", "source": "Sneddon II.4"},
    # --- Cauchy problems ----------------------------------------------------
    {
        "id": "tb-cauchy-advection",
        "eq": "ut + 2*ux",
        "vars": "x t",
        "ics": [("U(x, 0)", "exp(-x**2)")],
        "source": "Evans 2.1",
    },
    {
        "id": "tb-cauchy-stretch",
        "eq": "ut + x*ux",
        "vars": "x t",
        "ics": [("U(x, 0)", "sin(x)")],
        "source": "Debnath 2.4",
    },
    {
        "id": "tb-cauchy-growth",
        "eq": "ux + uy - u",
        "ics": [("U(x, 0)", "cos(x)")],
        "source": "Debnath 2.4",
    },
    {
        "id": "tb-cauchy-decay-t",
        "eq": "ut + ux + u",
        "vars": "x t",
        "ics": [("U(x, 0)", "x**2")],
        "source": "Debnath 2.4",
    },
    {
        "id": "tb-cauchy-source",
        "eq": "ux - uy - 1",
        "ics": [("U(x, 0)", "x**2")],
        "source": "Sneddon II.4",
    },
    {
        "id": "tb-cauchy-euler",
        "eq": "x*ux + y*uy - 2*u",
        "ics": [("U(x, 1)", "x**2 + 1")],
        "source": "Evans 3.2",
    },
    {
        "id": "tb-cauchy-3d",
        "eq": "ut + ux + 2*uy",
        "vars": "x y t",
        "ics": [("U(x, y, 0)", "x*y")],
        "source": "Evans 2.1",
    },
    {
        "id": "tb-burgers-linear",
        "eq": "ut + u*ux",
        "vars": "x t",
        "ics": [("U(x, 0)", "x")],
        "source": "Debnath 2.6",
    },
    {
        "id": "tb-burgers-breaking",
        "eq": "ut + u*ux",
        "vars": "x t",
        "ics": [("U(x, 0)", "-x")],
        "source": "Debnath 2.6",
    },
    {
        "id": "tb-burgers-sine",
        "eq": "ut + u*ux",
        "vars": "x t",
        "ics": [("U(x, 0)", "sin(x)")],
        "source": "Debnath 2.6",
    },
    {
        "id": "tb-burgers-u2",
        "eq": "ut + u**2*ux",
        "vars": "x t",
        "ics": [("U(x, 0)", "x")],
        "source": "Debnath 2.6",
    },
    {
        "id": "tb-cauchy-semilinear",
        "eq": "ux + uy - u**2",
        "ics": [("U(x, 0)", "1")],
        "source": "Debnath 2.5",
    },
    {
        "id": "tb-characteristic-refusal",
        "eq": "ux",
        "ics": [("U(x, 0)", "x")],
        "source": "Evans 3.2 (characteristic data: must refuse)",
        "expect": "refused",
    },
    # --- fully nonlinear ------------------------------------------------------
    {"id": "tb-pq1", "eq": "ux*uy - 1", "source": "Sneddon II.13"},
    {"id": "tb-eikonal", "eq": "ux**2 + uy**2 - 1", "source": "Evans 3.2"},
    {"id": "tb-clairaut-pq", "eq": "u - x*ux - y*uy - ux*uy", "source": "Sneddon II.13"},
    {
        "id": "tb-clairaut-sphere",
        "eq": "u - x*ux - y*uy - sqrt(1 + ux**2 + uy**2)",
        "source": "Sneddon II.13",
    },
    {"id": "tb-f-upq-1", "eq": "ux**2 + uy**2 - u", "source": "Sneddon II.13"},
    {"id": "tb-f-upq-2", "eq": "ux*uy - u", "source": "Evans 3.2"},
    {"id": "tb-separable-1", "eq": "ux - x**2 - uy - y**2", "source": "Sneddon II.13"},
    {"id": "tb-separable-2", "eq": "ux**2 + uy**2 - x - y", "source": "Sneddon II.13"},
    {"id": "tb-pq-xy", "eq": "ux*uy - x*y", "source": "Sneddon II.13 (Charpit)"},
    {"id": "tb-hj-free", "eq": "ut + ux**2", "vars": "x t", "source": "Evans 3.3"},
    {"id": "tb-hj-potential", "eq": "ut + ux**2/2 + x", "vars": "x t", "source": "Evans 3.3"},
]

for _e in TEXTBOOK:
    _e.setdefault("vars", "x y")
