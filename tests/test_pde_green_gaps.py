"""Green's-function gap round: Klein–Gordon and telegraph kernels in 2 and 3
space dimensions, the Neumann problem on a disk, Helmholtz waveguides
(strips and slabs) and the certified count of eigenvalues below the threshold
of a Robin Sturm–Liouville problem.

Every answer must be exactly verified; a refusal carries an ``E-PDE-25x/260``
code.  Numeric spot checks compare with published closed forms (Duffy §4.3–4.4,
Morse & Feshbach §7.2, Strauss §6.4).
"""

from __future__ import annotations

import math

import alkahest as ak
import alkahest.experimental as akx
import pytest


def _num(e, env: dict) -> complex:
    v = ak.evaluate(e, env).value
    return complex(v)


# ---------------------------------------------------------------------------
# Light-cone kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["klein_gordon", "telegraph"])
@pytest.mark.parametrize("dim", [2, 3])
def test_light_cone_kernels_verified(which, dim):
    p = ak.ExprPool()
    space = [p.symbol(n) for n in ("x", "y", "z")[:dim]]
    t = p.symbol("t")
    c = p.symbol("c", "positive")
    par = p.symbol("m" if which == "klein_gordon" else "a", "positive")
    kw = {"speed": c, ("mass" if which == "klein_gordon" else "damping"): par}
    fs = akx.fundamental_solution(which, space, t, **kw)
    assert fs.verified, fs.failed_checks()
    whats = {ch["what"] for ch in fs.checks}
    if dim == 2:
        assert {"descent", "mass", "homogeneous"} <= whats
        assert ("cosh" if which == "telegraph" else "cos") in str(fs.kernel)
    else:
        assert {"dimension_walk", "commutation", "homogeneous"} <= whats
        assert "diracdelta" in str(fs.kernel)
        assert ("bessel_i" if which == "telegraph" else "bessel_j") in str(fs.kernel)
    assert "heaviside" in str(fs.kernel)
    assert fs.support is not None


def test_klein_gordon_2d_matches_published_form():
    # c = 1, m = 2: G = cos(2√(t² − r²)) / (2π√(t² − r²)) inside the cone.
    p = ak.ExprPool()
    x, y, t = p.symbol("x"), p.symbol("y"), p.symbol("t", "positive")
    fs = akx.fundamental_solution("klein_gordon", [x, y], t, speed=p.integer(1), mass=p.integer(2))
    assert fs.verified
    env = {x: 0.3, y: 0.2, t: 0.9}
    s = math.sqrt(0.81 - 0.09 - 0.04)
    want = math.cos(2 * s) / (2 * math.pi * s)
    assert abs(_num(fs.smooth, env) - want) < 1e-10


def test_klein_gordon_3d_tail_shape():
    # c = 1, m = 1: tail −J₁(√(t² − r²)) / (4π√(t² − r²)).
    p = ak.ExprPool()
    x, y, z = p.symbol("x"), p.symbol("y"), p.symbol("z")
    t = p.symbol("t", "positive")
    fs = akx.fundamental_solution(
        "klein_gordon", [x, y, z], t, speed=p.integer(1), mass=p.integer(1)
    )
    assert fs.verified
    # `evaluate` has no J₁ kernel: compare the shape −J₁(σ)/(4πσ), σ = √(t² − r²).
    shown = str(fs.smooth)
    assert "bessel_j1" in shown or "bessel_j(1" in shown
    assert "-1/4" in shown
    assert "pi" in shown
    assert "diracdelta" in str(fs.kernel)


# ---------------------------------------------------------------------------
# Neumann problem on a disk
# ---------------------------------------------------------------------------


def test_neumann_function_of_the_disk():
    p = ak.ExprPool()
    x, y, xi, eta = (p.symbol(n) for n in ("x", "y", "xi", "eta"))
    R = p.symbol("R", "positive")
    g = akx.image_green_function("laplace", [x, y], [xi, eta], ("disk", R), "neumann")
    assert g.verified, g.failed_checks()
    assert g.method == "neumann_function"
    assert {"pde", "bc", "symmetry", "boundary_kernel"} <= {c["what"] for c in g.checks}
    assert g.poisson_kernel is not None


def test_neumann_disk_solve_harmonic_and_poisson():
    p = ak.ExprPool()
    x, y, th = p.symbol("x"), p.symbol("y"), p.symbol("theta")
    R = p.symbol("R", "positive")
    s = akx.neumann_disk_solve(x, y, R, 0, ak.cos(th) + 2 * ak.sin(2 * th), th)
    assert s.verified, s.failed_checks()
    assert {"pde", "bc", "compatibility"} <= {
        c["what"] for c in s.checks if c["status"] == "proved_zero"
    }
    # u = C + x + (2R/2)(2xy)/R² = C + x + 2xy/R.
    env = {x: 0.3, y: -0.4, R: 1.7, s.constant: 0.0}
    assert abs(_num(s.value, env) - (0.3 + 2 * 0.3 * -0.4 / 1.7)) < 1e-12
    # Poisson: Δu = 1 with ∂u/∂r = R/2.
    s = akx.neumann_disk_solve(x, y, R, 1, R / 2, th)
    assert s.verified
    env = {x: 0.3, y: -0.4, R: 1.7, s.constant: 0.0}
    assert abs(_num(s.value, env) - (0.09 + 0.16) / 4) < 1e-12
    assert s.green_verified
    assert any("unique" in n for n in s.notes)


def test_neumann_disk_compatibility_refusals():
    p = ak.ExprPool()
    x, y, th = p.symbol("x"), p.symbol("y"), p.symbol("theta")
    R = p.symbol("R", "positive")
    with pytest.raises(Exception, match="E-PDE-257"):
        akx.neumann_disk_solve(x, y, R, 0, p.integer(1), th)
    with pytest.raises(Exception, match="E-PDE-257"):
        akx.neumann_disk_solve(x, y, p.integer(1), 1, ak.cos(th), th)
    b = p.symbol("b")
    with pytest.raises(Exception, match="E-PDE-258"):
        akx.neumann_disk_solve(x, y, R, 0, ak.cos(th) + b, th)
    with pytest.raises(Exception, match="E-PDE-258"):
        akx.neumann_disk_solve(x, y, R, ak.exp(x), ak.cos(th), th)


# ---------------------------------------------------------------------------
# Helmholtz waveguides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "walls",
    [["dirichlet", "dirichlet"], ["neumann", "neumann"], ["dirichlet", "neumann"]],
)
def test_helmholtz_strip_mode_series(walls):
    p = ak.ExprPool()
    x, y, xi, eta = (p.symbol(n) for n in ("x", "y", "xi", "eta"))
    k, L = p.symbol("k", "positive"), p.symbol("L", "positive")
    g = akx.image_green_function("helmholtz", [x, y], [xi, eta], ("strip", 1, L), walls, k=k)
    assert g.verified, g.failed_checks()
    assert g.method == "mode_series"
    assert "Sum" in str(g.value)
    assert any("radiation" in n for n in g.notes)


def test_helmholtz_slab_and_cutoff():
    p = ak.ExprPool()
    x, y, z, xi, eta, zeta = (p.symbol(n) for n in ("x", "y", "z", "xi", "eta", "zeta"))
    k, L = p.symbol("k", "positive"), p.symbol("L", "positive")
    g = akx.image_green_function(
        "helmholtz", [x, y, z], [xi, eta, zeta], ("slab", 2, L), ["dirichlet", "dirichlet"], k=k
    )
    assert g.verified
    assert "hankel1" in str(g.value)
    pi = ak.parse("pi", p)
    with pytest.raises(Exception, match="E-PDE-260"):
        akx.image_green_function(
            "helmholtz",
            [x, y],
            [xi, eta],
            ("strip", 1, p.integer(1)),
            ["dirichlet", "dirichlet"],
            k=3 * pi,
        )


# ---------------------------------------------------------------------------
# Certified count of eigenvalues below the threshold
# ---------------------------------------------------------------------------


def test_count_lower_robin():
    p = ak.ExprPool()
    x = p.symbol("x")
    one = akx.SturmLiouville(x, (0, 1), ("dirichlet", ("robin", -2, 1)))
    n = akx.count_lower_eigenvalues(one)
    assert n.verified
    assert n.count == 1
    assert n.negative == 1
    _, lo, hi = n.enclosures[0]
    assert hi - lo < 1e-9
    assert abs((lo + hi) / 2 + 1.915008048154**2) < 1e-9
    two = akx.SturmLiouville(x, (0, 1), (("robin", 3, 1), ("robin", -3, 1)))
    assert akx.count_lower_eigenvalues(two).count == 2
    none = akx.SturmLiouville(x, (0, 1), ("dirichlet", ("robin", p.rational(-1, 2), 1)))
    n = akx.count_lower_eigenvalues(none)
    assert n.count == 0
    assert n.method == "interval_cover"
    nn = akx.SturmLiouville(x, (0, 1), ("neumann", "neumann"))
    n = akx.count_lower_eigenvalues(nn)
    assert n.method == "rayleigh"
    assert n.threshold_is_eigenvalue


def test_count_lower_params_and_refusal():
    p = ak.ExprPool()
    x, h = p.symbol("x"), p.symbol("h")
    sl = akx.SturmLiouville(x, (0, 1), ("dirichlet", ("robin", -h, 1)))
    with pytest.raises(Exception, match="E-PDE-259"):
        akx.count_lower_eigenvalues(sl)
    assert akx.count_lower_eigenvalues(sl, {h: p.integer(2)}).count == 1
    assert akx.count_lower_eigenvalues(sl, {h: p.rational(1, 2)}).count == 0
