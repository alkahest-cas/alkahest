"""Nonlinear PDE corpus (milestone M6).

``ABBASI_NONLINEAR`` is the whole nonlinear chapter (ch. 7, Polyanin & Zaitsev,
*Handbook of Nonlinear PDEs*) of Nasser Abbasi's benchmark,
https://www.12000.org/my_notes/pde_in_CAS/maple_2026_1_and_mma_15/ — problems
2001–2012, with the benchmark's published verdicts for Mathematica 15 and
Maple 2026.1 (a solution returned, not verified): Maple 5/12, Mathematica 1/12.

``SOLITONS`` is a table of standard nonlinear equations with known exact
traveling-wave / soliton / kink families (Ablowitz–Clarkson, Polyanin–Zaitsev,
Wazwaz).  ``SYMMETRIES`` lists the dimension of the point-symmetry algebra of
standard equations (Olver, *Applications of Lie Groups to Differential
Equations*; Bluman, Cheviakov & Anco 2010; Ibragimov's CRC handbook), and
``CONSERVATION`` the number of conservation laws with multipliers of order
``≤ 2``.

Entries use the harness format of ``pde_harness.py`` (unknown ``u(x, t)``).
"""

ABBASI_NONLINEAR = [
    {
        "id": "abbasi-2001",
        "vars": "x t",
        "eq": "D(u, t) - (a*D(u, x, 2) + b*u**2)",
        "mma_published": False,
        "maple_published": False,
    },
    {
        "id": "abbasi-2002",
        "vars": "x t",
        "eq": "D(u, t) - (D(u, x, 2) + a*u*(1 - u))",
        "mma_published": True,
        "maple_published": True,
    },
    {
        "id": "abbasi-2003",
        "vars": "x t",
        "eq": "D(u, t) - (a*D(u, x, 2) - b*u**3)",
        "mma_published": False,
        "maple_published": False,
    },
    {
        "id": "abbasi-2004",
        "vars": "x t",
        "eq": "D(u, t) - (D(u, x, 2) + a*u - b*u**3)",
        "mma_published": False,
        "maple_published": True,
    },
    {
        "id": "abbasi-2005",
        "vars": "x t",
        "eq": "D(u, t) - (a*D(u, x, 2) - b*u**3 - c*u**2)",
        "mma_published": False,
        "maple_published": True,
    },
    {
        "id": "abbasi-2006",
        "vars": "x t",
        "eq": "D(u, t) - (D(u, x, 2) - u*(1 - u)*(a - u))",
        "mma_published": False,
        "maple_published": True,
    },
    {
        "id": "abbasi-2007",
        "vars": "x t",
        "eq": "D(u, t) - (a*D(u, x, 2) + b0 + b1*u + b2*u**2 + b3*u**3)",
        "mma_published": False,
        "maple_published": True,
    },
    {
        "id": "abbasi-2008",
        "vars": "x t",
        "eq": "D(u, t) - (a*D(u, x, 2) + b*u**k)",
        "mma_published": False,
        "maple_published": False,
    },
    {
        "id": "abbasi-2009",
        "vars": "x t",
        "eq": "D(u, t) - (D(u, x, 2) + a*u + b*u**m)",
        "mma_published": False,
        "maple_published": False,
    },
    {
        "id": "abbasi-2010",
        "vars": "x t",
        "eq": "D(u, t) - (D(u, x, 2) + a*u + b*u**m + c*u**(2*m - 1))",
        "mma_published": False,
        "maple_published": False,
    },
    {
        "id": "abbasi-2011",
        "vars": "x t",
        "eq": "D(u, t) - (D(u, x, 2) + a*u**(m - 1) + b*m*u**m - m*b**2*u**(2*m - 1))",
        "mma_published": False,
        "maple_published": False,
    },
    {
        "id": "abbasi-2012",
        "vars": "x t",
        "eq": "D(u, t) - (a*D(u, x, 2) + s1*(b*x + c*t)**k + s2*u**m)",
        "mma_published": False,
        "maple_published": False,
    },
]

SOLITONS = [
    {"id": "kdv", "vars": "x t", "eq": "D(u, t) + 6*u*D(u, x) + D(u, x, 3)"},
    {"id": "kdv-1", "vars": "x t", "eq": "D(u, t) + u*D(u, x) + D(u, x, 3)"},
    {"id": "mkdv", "vars": "x t", "eq": "D(u, t) + 6*u**2*D(u, x) + D(u, x, 3)"},
    {"id": "gardner", "vars": "x t", "eq": "D(u, t) + 6*u*D(u, x) + 6*u**2*D(u, x) + D(u, x, 3)"},
    {"id": "burgers", "vars": "x t", "eq": "D(u, t) + u*D(u, x) - D(u, x, 2)"},
    {"id": "burgers-nu", "vars": "x t", "eq": "D(u, t) + u*D(u, x) - nu*D(u, x, 2)"},
    {"id": "kdv-burgers", "vars": "x t", "eq": "D(u, t) + u*D(u, x) - D(u, x, 2) + D(u, x, 3)"},
    {"id": "fisher", "vars": "x t", "eq": "D(u, t) - D(u, x, 2) - u*(1 - u)"},
    {"id": "newell-whitehead", "vars": "x t", "eq": "D(u, t) - D(u, x, 2) - u + u**3"},
    {"id": "allen-cahn", "vars": "x t", "eq": "D(u, t) - D(u, x, 2) + u**3 - u"},
    {"id": "zeldovich", "vars": "x t", "eq": "D(u, t) - D(u, x, 2) - u**2 + u**3"},
    {"id": "huxley", "vars": "x t", "eq": "D(u, t) - D(u, x, 2) + u*(u - 1)*(u - R(1, 2))"},
    {"id": "sine-gordon", "vars": "x t", "eq": "D(u, x, t) - sin(u)"},
    {"id": "sine-gordon-wave", "vars": "x t", "eq": "D(u, t, 2) - D(u, x, 2) + sin(u)"},
    {"id": "phi4", "vars": "x t", "eq": "D(u, t, 2) - D(u, x, 2) - u + u**3"},
    {"id": "klein-gordon-cubic", "vars": "x t", "eq": "D(u, t, 2) - D(u, x, 2) + u - u**3"},
    {"id": "bbm", "vars": "x t", "eq": "D(u, t) + D(u, x) + u*D(u, x) - D(D(u, x, 2), t)"},
    {
        "id": "boussinesq",
        "vars": "x t",
        "eq": "D(u, t, 2) - D(u, x, 2) - 3*D(u**2, x, 2) - D(u, x, 4)",
    },
    {"id": "kawahara", "vars": "x t", "eq": "D(u, t) + u*D(u, x) + D(u, x, 3) - D(u, x, 5)"},
    {
        "id": "fitzhugh-nagumo",
        "vars": "x t",
        "eq": "D(u, t) - D(u, x, 2) + u*(1 - u)*(R(1, 3) - u)",
    },
    {"id": "porous-medium", "vars": "x t", "eq": "D(u, t) - D(u*D(u, x), x)"},
    {"id": "liouville", "vars": "x t", "eq": "D(u, x, t) - exp(u)"},
]

# Dimension of the finite part of the point-symmetry algebra and whether an
# infinite-dimensional part exists (generic parameter values).
SYMMETRIES = [
    {"id": "heat", "eq": "D(u, t) - D(u, x, 2)", "dim": 6, "infinite": True},
    {"id": "burgers", "eq": "D(u, t) + u*D(u, x) - D(u, x, 2)", "dim": 5, "infinite": False},
    {"id": "kdv", "eq": "D(u, t) + u*D(u, x) + D(u, x, 3)", "dim": 4, "infinite": False},
    {"id": "mkdv", "eq": "D(u, t) + u**2*D(u, x) + D(u, x, 3)", "dim": 3, "infinite": False},
    {"id": "fisher", "eq": "D(u, t) - D(u, x, 2) - u*(1 - u)", "dim": 2, "infinite": False},
    {"id": "sine-gordon", "eq": "D(u, x, t) - sin(u)", "dim": 3, "infinite": False},
    {"id": "liouville", "eq": "D(u, x, t) - exp(u)", "dim": 0, "infinite": True},
    {"id": "wave", "eq": "D(u, t, 2) - D(u, x, 2)", "dim": 1, "infinite": True},
    {"id": "laplace", "eq": "D(u, x, 2) + D(u, t, 2)", "dim": 1, "infinite": True},
    {"id": "inviscid-burgers", "eq": "D(u, t) + u*D(u, x)", "dim": 0, "infinite": True},
    {
        "id": "nonlinear-diffusion-m",
        "eq": "D(u, t) - D(u**m*D(u, x), x)",
        "dim": 4,
        "infinite": False,
    },
    {
        "id": "nonlinear-diffusion-exp",
        "eq": "D(u, t) - D(exp(u)*D(u, x), x)",
        "dim": 4,
        "infinite": False,
    },
    {"id": "porous-medium", "eq": "D(u, t) - D(u*D(u, x), x)", "dim": 4, "infinite": False},
    {
        "id": "fast-diffusion-43",
        "eq": "D(u, t) - D(u**R(-4, 3)*D(u, x), x)",
        "dim": 5,
        "infinite": False,
    },
    {
        "id": "klein-gordon-cubic",
        "eq": "D(u, t, 2) - D(u, x, 2) + u**3",
        "dim": 4,
        "infinite": False,
    },
    {"id": "phi4", "eq": "D(u, t, 2) - D(u, x, 2) - u + u**3", "dim": 3, "infinite": False},
    {
        "id": "boussinesq",
        "eq": "D(u, t, 2) - D(u, x, 2) - 3*D(u**2, x, 2) - D(u, x, 4)",
        "dim": 3,
        "infinite": False,
    },
    {
        "id": "potential-burgers",
        "eq": "D(u, t) - D(u, x, 2) - D(u, x)**2",
        "dim": 6,
        "infinite": True,
    },
    {"id": "linear-transport", "eq": "D(u, t) + D(u, x)", "dim": 0, "infinite": True},
    {"id": "allen-cahn", "eq": "D(u, t) - D(u, x, 2) + u**3 - u", "dim": 2, "infinite": False},
]

# Number of independent conservation laws with multipliers of order <= 2
# (Bluman, Cheviakov & Anco 2010; Olver 1993, ch. 4-5).
# ``None``: infinitely many (every solution of the adjoint linear equation).
CONSERVATION = [
    {"id": "kdv", "eq": "D(u, t) + u*D(u, x) + D(u, x, 3)", "laws": 4},
    {"id": "burgers", "eq": "D(u, t) + u*D(u, x) - D(u, x, 2)", "laws": 1},
    # Mass, momentum, energy, and the explicitly x,t-dependent law with
    # multiplier t*(3*u_xx + u**3) - x*u (density t*H - x*u**2/2), which the
    # test checks as an exact divergence identity.
    {"id": "mkdv", "eq": "D(u, t) + u**2*D(u, x) + D(u, x, 3)", "laws": 4},
    {"id": "heat", "eq": "D(u, t) - D(u, x, 2)", "laws": None},
    {"id": "fisher", "eq": "D(u, t) - D(u, x, 2) - u*(1 - u)", "laws": 0},
    {"id": "porous-medium", "eq": "D(u, t) - D(u*D(u, x), x)", "laws": 2},
]
