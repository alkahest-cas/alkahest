"""V2-14 — Numerical homotopy (`solve(..., method="homotopy")`, `solve_numerical`)."""

import alkahest
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "solve_numerical"),
    reason="native module built without groebner feature",
)


def test_solve_homotopy_product_quadratics():
    import alkahest

    p = alkahest.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    neg1 = p.integer(-1)
    sols = alkahest.solve(
        [x**2 + neg1, y**2 + neg1],
        [x, y],
        method="homotopy",
    )
    assert len(sols) == 4
    tol = 1e-8
    assert all(isinstance(s[x], float) for s in sols)
    roots = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    found = {(round(s[x], 6), round(s[y], 6)) for s in sols}
    want_r = {(round(ax, 6), round(by, 6)) for ax, by in roots}
    assert found == want_r, (found, want_r)
    for s in sols:
        assert abs(s[x] ** 2 - 1.0) < tol
        assert abs(s[y] ** 2 - 1.0) < tol


def test_solve_numerical_certified_solution_api():
    import alkahest

    p = alkahest.ExprPool()
    x = p.symbol("x")
    neg1 = p.integer(-1)
    pts = alkahest.solve_numerical([x**2 + neg1], [x])
    assert len(pts) >= 1
    assert hasattr(pts[0], "coordinates")
    assert hasattr(pts[0], "smale_certified")
    d = pts[0].to_dict()
    assert len(d) == 1
    assert abs(list(d.values())[0] ** 2 - 1.0) < 1e-10
