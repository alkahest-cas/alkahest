"""Trigonometric normal form: ``simplify_trig_normal_form``.

Headline probe: orthogonality of a 3-2-1 (yaw-pitch-roll) direction-cosine
matrix.  Every entry of ``R.T @ R - I`` must collapse to ``0`` in a single
call.  A guard test confirms a genuinely non-identity product does *not*
collapse to ``0`` (no false simplification).
"""

import alkahest
from alkahest import ExprPool, cos, simplify_trig_normal_form, sin


def _pool():
    return ExprPool()


def _neg(p, e):
    return p.integer(-1) * e


def _dcm_321(p, phi, theta, psi):
    """3-2-1 Euler-angle DCM: R = Rx(phi) @ Ry(theta) @ Rz(psi)."""
    zero = p.integer(0)
    one = p.integer(1)

    rz = [
        [cos(psi), sin(psi), zero],
        [_neg(p, sin(psi)), cos(psi), zero],
        [zero, zero, one],
    ]
    ry = [
        [cos(theta), zero, _neg(p, sin(theta))],
        [zero, one, zero],
        [sin(theta), zero, cos(theta)],
    ]
    rx = [
        [one, zero, zero],
        [zero, cos(phi), sin(phi)],
        [zero, _neg(p, sin(phi)), cos(phi)],
    ]
    return _matmul(p, _matmul(p, rx, ry), rz)


def _matmul(p, a, b):
    out = [[p.integer(0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            acc = p.integer(0)
            for k in range(3):
                acc = acc + a[i][k] * b[k][j]
            out[i][j] = acc
    return out


def _transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def test_module_exports_simplify_trig_normal_form():
    assert hasattr(alkahest, "simplify_trig_normal_form")
    assert "simplify_trig_normal_form" in alkahest.__all__


def test_pythagorean_single_angle():
    p = _pool()
    x = p.symbol("x")
    r = simplify_trig_normal_form(sin(x) ** 2 + cos(x) ** 2)
    assert r.value == p.integer(1)


def test_pythagorean_multi_angle():
    # cos^2(theta)*sin^2(phi) + cos^2(theta)*cos^2(phi) -> cos^2(theta)
    p = _pool()
    theta = p.symbol("theta")
    phi = p.symbol("phi")
    expr = cos(theta) ** 2 * sin(phi) ** 2 + cos(theta) ** 2 * cos(phi) ** 2
    r = simplify_trig_normal_form(expr)
    assert r.value == cos(theta) ** 2


def test_dcm_rtr_minus_identity_is_zero():
    p = _pool()
    phi = p.symbol("phi")
    theta = p.symbol("theta")
    psi = p.symbol("psi")
    r_mat = _dcm_321(p, phi, theta, psi)
    rt = _transpose(r_mat)
    rtr = _matmul(p, rt, r_mat)
    zero = p.integer(0)
    for i in range(3):
        for j in range(3):
            ident = p.integer(1) if i == j else p.integer(0)
            diff = rtr[i][j] + _neg(p, ident)
            res = simplify_trig_normal_form(diff)
            assert res.value == zero, (
                f"R.T@R - I entry [{i}][{j}] did not collapse to 0: {res.value}"
            )


def test_dcm_diagonal_entry_is_one():
    p = _pool()
    phi = p.symbol("phi")
    theta = p.symbol("theta")
    psi = p.symbol("psi")
    r_mat = _dcm_321(p, phi, theta, psi)
    rtr = _matmul(p, _transpose(r_mat), r_mat)
    res = simplify_trig_normal_form(rtr[2][2])
    assert res.value == p.integer(1)


def test_non_orthogonal_product_does_not_collapse():
    # (2R).T @ (2R) = 4*I, so the [0][0] entry minus 1 must reduce to 3, not 0.
    p = _pool()
    phi = p.symbol("phi")
    theta = p.symbol("theta")
    psi = p.symbol("psi")
    r_mat = _dcm_321(p, phi, theta, psi)
    two = p.integer(2)
    m = [[two * r_mat[i][j] for j in range(3)] for i in range(3)]
    mtm = _matmul(p, _transpose(m), m)
    diff = mtm[0][0] + p.integer(-1)
    res = simplify_trig_normal_form(diff)
    assert res.value != p.integer(0)
    assert res.value == p.integer(3)
