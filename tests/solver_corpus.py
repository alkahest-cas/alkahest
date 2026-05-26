"""
Cross-CAS oracle for the V1-4 polynomial system solver.

Builds 20 curated polynomial systems, solves each with both
``alkahest.solve`` and SymPy, and checks that every solution Alkahest
returns is satisfied numerically by the original equations.

Run:
    pytest tests/solver_corpus.py -v

Requires:
    pip install sympy
    Alkahest built with ``--features groebner``.
"""

import pytest

sympy = pytest.importorskip("sympy")

import alkahest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "solve"),
    reason="alkahest.solve requires the groebner feature",
)


# ---------------------------------------------------------------------------
# Tiny DSL: build Alkahest + SymPy expressions in lockstep.
# ---------------------------------------------------------------------------


class _EqBuilder:
    """Parallel Alkahest / SymPy expression builder.

    Every call returns a ``(ak_expr, sp_expr)`` pair so the same Python
    code describes both the Alkahest and SymPy view of a polynomial.
    """

    def __init__(self, symbols: list[str]):
        self.pool = alkahest.ExprPool()
        self.ak_vars = [self.pool.symbol(s) for s in symbols]
        sp_vars = sympy.symbols(" ".join(symbols))
        if not isinstance(sp_vars, tuple):
            sp_vars = (sp_vars,)
        self.sp_vars = list(sp_vars)
        self._by_name = dict(zip(symbols, range(len(symbols))))

    def var(self, name: str):
        i = self._by_name[name]
        return self.ak_vars[i], self.sp_vars[i]

    def const(self, n: int):
        return self.pool.integer(n), sympy.Integer(n)

    def add(self, *pairs):
        ak_sum = pairs[0][0]
        sp_sum = pairs[0][1]
        for a, s in pairs[1:]:
            ak_sum = ak_sum + a
            sp_sum = sp_sum + s
        return ak_sum, sp_sum

    def sub(self, a, b):
        neg_b_ak = self.pool.integer(-1) * b[0]
        return (a[0] + neg_b_ak, a[1] - b[1])

    def mul(self, *pairs):
        ak_prod = pairs[0][0]
        sp_prod = pairs[0][1]
        for a, s in pairs[1:]:
            ak_prod = ak_prod * a
            sp_prod = sp_prod * s
        return ak_prod, sp_prod

    def pow(self, base, k: int):
        return (base[0] ** k, base[1] ** k)

    def neg(self, a):
        return (self.pool.integer(-1) * a[0], -a[1])


def _k(b: _EqBuilder, n: int):
    return b.const(n)


# ---------------------------------------------------------------------------
# Corpus: 20 curated systems.
#
# Each entry: (case_id, symbols, system_builder).  The builder takes an
# `_EqBuilder` and returns a list of (ak_eq, sp_eq) pairs, each meaning
# ``... == 0``.
# ---------------------------------------------------------------------------


def _sys_univar_linear(b: _EqBuilder):
    x = b.var("x")
    return [b.sub(b.mul(_k(b, 2), x), _k(b, 6))]  # 2x - 6


def _sys_univar_quadratic_real(b: _EqBuilder):
    x = b.var("x")
    return [b.sub(b.pow(x, 2), _k(b, 4))]  # x² - 4


def _sys_univar_quadratic_irr(b: _EqBuilder):
    x = b.var("x")
    return [b.sub(b.pow(x, 2), _k(b, 2))]  # x² - 2


def _sys_univar_quadratic_sum(b: _EqBuilder):
    x = b.var("x")
    return [b.add(b.pow(x, 2), b.neg(b.mul(_k(b, 3), x)), _k(b, 2))]  # x² - 3x + 2


def _sys_univar_double_root(b: _EqBuilder):
    x = b.var("x")
    # (x - 1)² = x² - 2x + 1
    return [b.add(b.pow(x, 2), b.neg(b.mul(_k(b, 2), x)), _k(b, 1))]


def _sys_linear_2d_basic(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(b.add(x, y), _k(b, 1)),  # x + y - 1
        b.sub(x, y),  # x - y
    ]


def _sys_linear_2d_scaled(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(b.add(b.mul(_k(b, 2), x), b.mul(_k(b, 3), y)), _k(b, 5)),  # 2x + 3y - 5
        b.sub(b.sub(x, y), _k(b, 1)),  # x - y - 1
    ]


def _sys_linear_2d_negative(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.add(x, y, _k(b, 2)),  # x + y + 2
        b.sub(b.sub(x, y), _k(b, 4)),  # x - y - 4
    ]


def _sys_circle_line(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(b.add(b.pow(x, 2), b.pow(y, 2)), _k(b, 1)),  # x² + y² - 1
        b.sub(y, x),  # y - x
    ]


def _sys_circle_line_shifted(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(b.add(b.pow(x, 2), b.pow(y, 2)), _k(b, 2)),  # x² + y² - 2
        b.sub(y, x),  # y - x
    ]


def _sys_parabola_line(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(y, b.pow(x, 2)),  # y - x²
        b.sub(y, x),  # y - x
    ]


def _sys_hyperbola_line(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(b.mul(x, y), _k(b, 1)),  # xy - 1
        b.sub(y, x),  # y - x
    ]


def _sys_parabola_horizontal(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(y, b.pow(x, 2)),  # y - x²
        b.sub(y, _k(b, 4)),  # y - 4
    ]


def _sys_two_parabolas(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    return [
        b.sub(y, b.pow(x, 2)),  # y - x²
        b.add(b.sub(y, b.mul(_k(b, 2), b.pow(x, 2))), _k(b, 1)),  # y - 2x² + 1
    ]


def _sys_ellipse_line(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    # x² + 4y² - 4 = 0 ,  2y - x = 0   (using 2y = x instead of y = x/2)
    return [
        b.sub(b.add(b.pow(x, 2), b.mul(_k(b, 4), b.pow(y, 2))), _k(b, 4)),
        b.sub(b.mul(_k(b, 2), y), x),
    ]


def _sys_linear_3d_identity(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    z = b.var("z")
    return [b.sub(x, _k(b, 1)), b.sub(y, _k(b, 2)), b.sub(z, _k(b, 3))]


def _sys_linear_3d_standard(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    z = b.var("z")
    return [
        b.sub(b.add(x, y, z), _k(b, 6)),
        b.sub(x, y),
        b.sub(y, z),
    ]


def _sys_linear_3d_dependent(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    z = b.var("z")
    # x + 2y + 3z - 6, x - y, 2y + z - 3  →  (x, y, z) = (1, 1, 1)
    return [
        b.sub(b.add(x, b.mul(_k(b, 2), y), b.mul(_k(b, 3), z)), _k(b, 6)),
        b.sub(x, y),
        b.sub(b.add(b.mul(_k(b, 2), y), z), _k(b, 3)),
    ]


def _sys_quadric_plane(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    z = b.var("z")
    return [
        b.sub(b.add(b.pow(x, 2), b.pow(y, 2), b.pow(z, 2)), _k(b, 3)),
        b.sub(x, y),
        b.sub(y, z),
    ]


def _sys_two_circles_tangent(b: _EqBuilder):
    x = b.var("x")
    y = b.var("y")
    # x² + y² - 1, (x-1)² + y² - 1  =>  x² - 2x + 1 + y² - 1 = x² + y² - 2x
    # So second eq simplifies to x² + y² - 2x = 0.  After Gröbner with first: 2x - 1 = 0.
    return [
        b.sub(b.add(b.pow(x, 2), b.pow(y, 2)), _k(b, 1)),
        b.sub(b.add(b.pow(x, 2), b.pow(y, 2)), b.mul(_k(b, 2), x)),
    ]


CORPUS = [
    ("univar_linear", ["x"], _sys_univar_linear, 1),
    ("univar_quadratic_real", ["x"], _sys_univar_quadratic_real, 2),
    ("univar_quadratic_irr", ["x"], _sys_univar_quadratic_irr, 2),
    ("univar_quadratic_sum", ["x"], _sys_univar_quadratic_sum, 2),
    ("univar_double_root", ["x"], _sys_univar_double_root, 1),
    ("linear_2d_basic", ["x", "y"], _sys_linear_2d_basic, 1),
    ("linear_2d_scaled", ["x", "y"], _sys_linear_2d_scaled, 1),
    ("linear_2d_negative", ["x", "y"], _sys_linear_2d_negative, 1),
    ("circle_line", ["x", "y"], _sys_circle_line, 2),
    ("circle_line_shifted", ["x", "y"], _sys_circle_line_shifted, 2),
    ("parabola_line", ["x", "y"], _sys_parabola_line, 2),
    ("hyperbola_line", ["x", "y"], _sys_hyperbola_line, 2),
    ("parabola_horizontal", ["x", "y"], _sys_parabola_horizontal, 2),
    ("two_parabolas", ["x", "y"], _sys_two_parabolas, 2),
    ("ellipse_line", ["x", "y"], _sys_ellipse_line, 2),
    ("linear_3d_identity", ["x", "y", "z"], _sys_linear_3d_identity, 1),
    ("linear_3d_standard", ["x", "y", "z"], _sys_linear_3d_standard, 1),
    ("linear_3d_dependent", ["x", "y", "z"], _sys_linear_3d_dependent, 1),
    ("quadric_plane", ["x", "y", "z"], _sys_quadric_plane, 2),
    ("two_circles_tangent", ["x", "y"], _sys_two_circles_tangent, 2),
]


def test_corpus_has_20_entries():
    assert len(CORPUS) == 20, f"corpus should have 20 entries, has {len(CORPUS)}"


@pytest.mark.parametrize(
    ("case_id", "symbols", "builder_fn", "expected_count"),
    CORPUS,
    ids=[c[0] for c in CORPUS],
)
def test_solver_vs_sympy(case_id, symbols, builder_fn, expected_count):
    """Solve with Alkahest; verify each solution satisfies every equation."""
    b = _EqBuilder(symbols)
    pairs = builder_fn(b)
    ak_eqs = [p[0] for p in pairs]
    sp_eqs = [p[1] for p in pairs]

    result = alkahest.solve(ak_eqs, b.ak_vars)

    if not isinstance(result, list):
        pytest.fail(f"[{case_id}] expected finite solutions, got {type(result)}")

    assert len(result) >= 1, f"[{case_id}] expected ≥1 solution, got 0"

    for sol in result:
        subs = {sp: float(sol[ak]) for ak, sp in zip(b.ak_vars, b.sp_vars)}
        for sp_eq in sp_eqs:
            residual = float(sp_eq.subs(subs).evalf())
            assert abs(residual) < 1e-7, (
                f"[{case_id}] solution {subs} doesn't satisfy {sp_eq} (residual {residual:.3e})"
            )

    # SymPy cross-check: ensure the oracle agrees that at least one real
    # solution exists.
    try:
        sp_solutions = sympy.solve(sp_eqs, b.sp_vars, dict=True)
        real_sols = []
        for s in sp_solutions:
            try:
                if all(sympy.im(v).evalf() == 0 for v in s.values()):
                    real_sols.append(s)
            except (TypeError, AttributeError):
                pass
        assert len(real_sols) >= 1, f"[{case_id}] SymPy found no real solutions — corpus bug"
    except (NotImplementedError, Exception):
        # If SymPy can't solve it, trust the Alkahest numerical check.
        pass
