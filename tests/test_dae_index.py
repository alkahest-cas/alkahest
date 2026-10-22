"""DAE index reduction (N9) and the DAE panic boundary (C3).

``pantelides`` used to match an equation to a variable whenever the equation
mentioned the variable *or any derivative of it*, and reported the number of
matching rounds as the index.  The Cartesian pendulum came back as index 1
(the textbook value is 3) with a "reduced" system that left the Lagrange
multiplier undetermined, and ``x' = y, x = sin t`` came back as index 0 (it is
2).  ``rosenfeld_groebner`` on a DAE with an algebraic variable (five variables,
four derivatives — the documented way to declare one) panicked with an index
out of bounds, which reached Python as a ``PanicException``.

The oracle used throughout: after reduction, the Jacobian of the last
derivative of each equation with respect to the highest derivative of each
unknown must be non-singular (SymPy, at random rational points), which is
what "the reduced system determines the highest derivatives" means.
"""

from __future__ import annotations

import random

import alkahest as ak
import pytest

sympy = pytest.importorskip("sympy")


def _top_jet(name: str, order: int) -> str:
    for _ in range(order):
        name = f"d{name}/dt"
    return name


def assert_reduced_jacobian_nonsingular(reduced: ak.DAE, unknowns: dict[str, int]) -> None:
    """SymPy oracle: the reduced equations determine the highest derivatives.

    ``unknowns`` maps each base variable to the order of its highest jet in the
    reduced system (textbook values, not read back from alkahest).  The matrix
    of every reduced equation differentiated by those top jets must have full
    column rank at a random rational point.
    """
    from alkahest.crosscheck import to_sympy

    eqs = [to_sympy(e) for e in reduced.equations()]
    by_name = {str(s): s for e in eqs for s in e.free_symbols}
    tops = [by_name[_top_jet(v, k)] for v, k in unknowns.items()]
    m = sympy.Matrix([[sympy.diff(e, s) for s in tops] for e in eqs])
    rng = random.Random(7)
    free = sorted(m.free_symbols, key=str)
    ranks = []
    for _ in range(3):
        pt = {s: sympy.Rational(rng.randint(2, 40), rng.randint(1, 9)) for s in free}
        ranks.append(m.subs(pt).rank())
    assert max(ranks) == len(unknowns), (ranks, len(unknowns))


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


def pendulum(pool: ak.ExprPool) -> ak.DAE:
    t = pool.symbol("t")
    x, y, u, w, lam = (pool.symbol(n) for n in ("x", "y", "u", "w", "lam"))
    dx, dy, du, dw = (pool.symbol(n) for n in ("dx/dt", "dy/dt", "du/dt", "dw/dt"))
    one = pool.integer(1)
    return ak.DAE.new(
        [dx - u, dy - w, du + lam * x, dw + lam * y + one, x**2 + y**2 - one],
        [x, y, u, w, lam],
        [dx, dy, du, dw],
        t,
    )


# ---------------------------------------------------------------------------
# N9 — the index, and the reduced system it implies
# ---------------------------------------------------------------------------


def test_pendulum_is_index_3(pool):
    reduced = ak.pantelides(pendulum(pool))
    assert reduced.index == 3
    # 5 originals + x', y' equations once + the constraint twice.
    assert reduced.n_equations == 9
    # λ appears in the reduced system's top-jet Jacobian: rank 5, not 4.
    assert_reduced_jacobian_nonsingular(reduced, {"x": 2, "y": 2, "u": 1, "w": 1, "lam": 0})


def test_pendulum_with_lambda_derivative_listed_is_still_index_3(pool):
    t = pool.symbol("t")
    x, y, u, w, lam = (pool.symbol(n) for n in ("x", "y", "u", "w", "lam"))
    d = [pool.symbol(f"d{n}/dt") for n in ("x", "y", "u", "w", "lam")]
    one = pool.integer(1)
    dae = ak.DAE.new(
        [d[0] - u, d[1] - w, d[2] + lam * x, d[3] + lam * y + one, x**2 + y**2 - one],
        [x, y, u, w, lam],
        d,
        t,
    )
    assert ak.pantelides(dae).index == 3


def test_x_prime_y_x_sin_t_is_index_2(pool):
    t = pool.symbol("t")
    x, y, dx, dy = (pool.symbol(n) for n in ("x", "y", "dx/dt", "dy/dt"))
    reduced = ak.pantelides(ak.DAE.new([dx - y, x - ak.sin(t)], [x, y], [dx, dy], t))
    assert reduced.index == 2
    assert reduced.n_equations == 3
    assert_reduced_jacobian_nonsingular(reduced, {"x": 1, "y": 0})


def test_hairer_wanner_hessenberg_index_2_and_3(pool):
    t = pool.symbol("t")
    x1, x2, y = (pool.symbol(n) for n in ("x1", "x2", "y"))
    dx1, dx2 = pool.symbol("dx1/dt"), pool.symbol("dx2/dt")
    # Index 2: x1' = x2 + y, x2' = -x1, 0 = x1 - sin t.
    dae = ak.DAE.new([dx1 - x2 - y, dx2 + x1, x1 - ak.sin(t)], [x1, x2, y], [dx1, dx2], t)
    assert ak.pantelides(dae).index == 2
    # Index 3: x' = v, v' = y, 0 = x - t**3.
    x, v = pool.symbol("x"), pool.symbol("v")
    dx, dv = pool.symbol("dx/dt"), pool.symbol("dv/dt")
    dae = ak.DAE.new([dx - v, dv - y, x - t**3], [x, v, y], [dx, dv], t)
    reduced = ak.pantelides(dae)
    assert reduced.index == 3
    assert_reduced_jacobian_nonsingular(reduced, {"x": 2, "v": 1, "y": 0})


def test_ode_is_index_0_and_algebraic_is_index_1(pool):
    t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")
    assert ak.pantelides(ak.DAE.new([dx - x], [x], [dx], t)).index == 0
    z = pool.symbol("z")
    reduced = ak.pantelides(ak.DAE.new([dx + x - z, z - x**2], [x, z], [dx], t))
    assert reduced.index == 1
    assert reduced.n_equations == 2


@pytest.mark.parametrize("seed", range(6))
def test_random_semi_explicit_index_1(pool, seed):
    """x' = A x + B z, 0 = C x + D z with D non-singular: index 1, nothing differentiated."""
    rng = random.Random(seed)
    n, m = rng.randint(1, 3), rng.randint(1, 3)
    t = pool.symbol("t")
    xs = [pool.symbol(f"x{i}") for i in range(n)]
    zs = [pool.symbol(f"z{i}") for i in range(m)]
    dxs = [pool.symbol(f"dx{i}/dt") for i in range(n)]

    def lin(coeffs, syms):
        acc = pool.integer(0)
        for c, s in zip(coeffs, syms):
            acc = acc + pool.integer(c) * s
        return acc

    while True:
        dmat = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(m)]
        if sympy.Matrix(dmat).det() != 0:
            break
    eqs = [
        dxs[i]
        - lin([rng.randint(-3, 3) for _ in range(n)], xs)
        - lin([rng.randint(-3, 3) for _ in range(m)], zs)
        for i in range(n)
    ]
    eqs += [lin([rng.randint(-3, 3) for _ in range(n)], xs) + lin(dmat[k], zs) for k in range(m)]
    reduced = ak.pantelides(ak.DAE.new(eqs, xs + zs, dxs, t))
    assert reduced.index == 1
    assert reduced.n_equations == n + m


@pytest.mark.parametrize("seed", range(6))
def test_random_hessenberg_index_2(pool, seed):
    """x' = A x + B y, 0 = C x - g(t) with C·B non-singular: index 2.

    Dense non-zero coefficients keep the structure generic; the SymPy oracle
    then checks that the reduced system determines x' and y.
    """
    rng = random.Random(100 + seed)
    n = rng.randint(1, 3)
    m = rng.randint(1, n)

    def nz() -> int:
        return rng.choice([-3, -2, -1, 1, 2, 3])

    while True:
        b = [[nz() for _ in range(m)] for _ in range(n)]
        c = [[nz() for _ in range(n)] for _ in range(m)]
        if (sympy.Matrix(c) * sympy.Matrix(b)).det() != 0:
            break
    t = pool.symbol("t")
    xs = [pool.symbol(f"x{i}") for i in range(n)]
    ys = [pool.symbol(f"y{k}") for k in range(m)]
    dxs = [pool.symbol(f"dx{i}/dt") for i in range(n)]

    def lin(coeffs, syms):
        acc = pool.integer(0)
        for co, s in zip(coeffs, syms):
            acc = acc + pool.integer(co) * s
        return acc

    eqs = [dxs[i] - lin([nz() for _ in range(n)], xs) - lin(b[i], ys) for i in range(n)]
    eqs += [lin(c[k], xs) - ak.sin(pool.integer(k + 1) * t) for k in range(m)]
    reduced = ak.pantelides(ak.DAE.new(eqs, xs + ys, dxs, t))
    assert reduced.index == 2
    # Only the m constraints are differentiated, once each.
    assert reduced.n_equations == n + 2 * m
    unknowns = {f"x{i}": 1 for i in range(n)} | {f"y{k}": 0 for k in range(m)}
    assert_reduced_jacobian_nonsingular(reduced, unknowns)


def test_singular_reduced_jacobian_is_refused(pool):
    """x' + y' = 1, x + y = t: dependent equations, x' and y' never determined."""
    t, x, y = pool.symbol("t"), pool.symbol("x"), pool.symbol("y")
    dx, dy = pool.symbol("dx/dt"), pool.symbol("dy/dt")
    with pytest.raises(ak.DaeError) as ei:
        ak.pantelides(ak.DAE.new([dx + dy - pool.integer(1), x + y - t], [x, y], [dx, dy], t))
    assert ei.value.code == "E-DAE-005"


def test_non_square_is_refused_with_code(pool):
    t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")
    with pytest.raises(ak.DaeError) as ei:
        ak.pantelides(ak.DAE.new([dx - x, x - t], [x], [dx], t))
    assert ei.value.code == "E-DAE-003"
    assert isinstance(ei.value, ValueError)  # unchanged for existing callers


# ---------------------------------------------------------------------------
# DAE.new shape validation (E-DAE-004)
# ---------------------------------------------------------------------------


def test_dae_new_refuses_more_derivatives_than_variables(pool):
    t, x, dx, dy = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt"), pool.symbol("dy/dt")
    with pytest.raises(ak.DaeError) as ei:
        ak.DAE.new([dx - x], [x], [dx, dy], t)
    assert ei.value.code == "E-DAE-004"


@pytest.mark.parametrize(
    "case", ["non_symbol_derivative", "duplicate_variable", "time_as_variable"]
)
def test_dae_new_refuses_malformed(pool, case):
    t, x, y, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("y"), pool.symbol("dx/dt")
    args = {
        "non_symbol_derivative": ([dx - x], [x], [x + y]),
        "duplicate_variable": ([dx - x, y], [x, x], [dx]),
        "time_as_variable": ([dx - x, t], [x, t], [dx]),
    }[case]
    with pytest.raises(ak.DaeError) as ei:
        ak.DAE.new(*args, t)
    assert ei.value.code == "E-DAE-004"


def test_dae_new_accepts_fewer_derivatives(pool):
    dae = pendulum(pool)
    assert dae.n_variables == 5
    assert len(dae.derivatives()) == 4


# ---------------------------------------------------------------------------
# C3 — rosenfeld_groebner with an algebraic variable; no panics reach Python
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(ak, "rosenfeld_groebner"), reason="needs the groebner feature")
def test_rosenfeld_groebner_with_algebraic_variable_does_not_panic(pool):
    dae = pendulum(pool)
    try:
        r = ak.rosenfeld_groebner(dae, max_prolong_rounds=1)
    except BaseException as e:  # pragma: no cover - the regression (a PanicException)
        pytest.fail(f"escaped as {type(e).__name__}: {e}")
    assert r.consistent is True
    # λ got a synthesised derivative, so the working DAE is parallel again.
    work = r.working_dae()
    assert len(work.variables()) == len(work.derivatives())
    # The hidden velocity constraint x*u + y*w (half the derivative of the
    # position constraint, after x' = u, y' = w) is in the ideal: checked with
    # SymPy's own Groebner basis over the returned generators.
    from alkahest.crosscheck import to_sympy

    gens = [to_sympy(e) for e in r.final_basis().to_exprs()]
    syms = sorted({s for g in gens for s in g.free_symbols}, key=str)
    by_name = {str(s): s for s in syms}
    vel = by_name["x"] * by_name["u"] + by_name["y"] * by_name["w"]
    gb = sympy.groebner(gens, *syms, order="grevlex")
    assert gb.contains(vel)


@pytest.mark.skipif(not hasattr(ak, "dae_index_reduce"), reason="needs the groebner feature")
def test_dae_index_reduce_pendulum_uses_pantelides(pool):
    out = ak.dae_index_reduce(pendulum(pool))
    assert out.used_pantelides is True
    assert out.dae().index == 3


def test_panic_boundary_raises_coded_alkahest_error():
    """A Rust panic inside a guarded entry point is an ``AlkahestError``
    (``E-INTERNAL-001``), so ``except Exception`` catches it."""
    from alkahest import alkahest as native

    # A PanicException is a BaseException: it would escape this `raises` and
    # fail the test, which is the regression being guarded.
    with pytest.raises(ak.AlkahestError) as ei:
        native._panic_boundary_selftest()
    assert isinstance(ei.value, Exception)
    assert ei.value.code == "E-INTERNAL-001"
    assert "deliberate panic" in str(ei.value)
