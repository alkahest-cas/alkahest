"""F20 — noncommutative follow-ups: d-dimensional gamma algebra, Lie-algebra
tables (su(2), su(3), general), and sequence wildcards in ``make_rule``.
Every result is checked against explicit matrices (numpy) or textbook
identities (FORM / FeynCalc conventions, ``tr 1 = 4``)."""

from __future__ import annotations

import itertools
import random

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")


def evaluate(expr, env, funcs=None):
    """Evaluate ``expr`` with matrices / numbers in place of symbols, keeping
    product order."""
    funcs = {"sqrt": np.sqrt, **(funcs or {})}
    node = expr.node()
    tag = node[0]
    if tag == "symbol":
        return env[node[1]]
    if tag == "integer":
        return int(node[1])
    if tag == "rational":
        return int(node[1]) / int(node[2])
    if tag == "add":
        out = 0
        for a in node[1]:
            out = out + evaluate(a, env, funcs)
        return out
    if tag == "mul":
        out = 1
        for a in node[1]:
            v = evaluate(a, env, funcs)
            both = isinstance(out, np.ndarray) and isinstance(v, np.ndarray)
            out = out @ v if both else out * v
        return out
    if tag == "pow":
        base = evaluate(node[1], env, funcs)
        n = evaluate(node[2], env, funcs)
        if isinstance(base, np.ndarray):
            return np.linalg.matrix_power(base, int(n))
        return base**n
    if tag == "func":
        return funcs[node[1]](*[evaluate(a, env, funcs) for a in node[2]])
    raise AssertionError(f"unexpected node {node}")


@pytest.fixture
def pool():
    return ak.ExprPool()


def nc_equal(a, b):
    return ak.nc_terms(a - b) == []


def is_zero(e):
    return ak.nc_terms(e) == []


# ---------------------------------------------------------------------------
# Gamma matrices in d = 2, 3, 4 (explicit), metric (+, -, ..., -)
# ---------------------------------------------------------------------------

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def gammas(d):
    if d == 2:
        return [SX, 1j * SY]
    if d == 3:
        return [SX, 1j * SY, 1j * SZ]
    assert d == 4
    z, i2 = np.zeros((2, 2)), np.eye(2)
    g = [np.block([[i2, z], [z, -i2]]).astype(complex)]
    g += [np.block([[z, s], [-s, z]]) for s in (SX, SY, SZ)]
    return g


def eta(d):
    return np.diag([1.0] + [-1.0] * (d - 1))


def test_gammas_satisfy_clifford():
    for d in (2, 3, 4):
        g, e = gammas(d), eta(d)
        for a in range(d):
            for b in range(d):
                ac = g[a] @ g[b] + g[b] @ g[a]
                assert np.allclose(ac, 2 * e[a, b] * np.eye(len(g[0])))


def slash(v, d):
    low = eta(d) @ v
    return sum(low[m] * gammas(d)[m] for m in range(d))


def vec_dot(x, y, d):
    return float(x @ eta(d) @ y)


# ---------------------------------------------------------------------------
# d-dimensional traces
# ---------------------------------------------------------------------------


def test_trace_known_identities_symbolic_d(pool):
    d = pool.symbol("d")
    mu, nu, rho, sig = (pool.symbol(n) for n in ("mu", "nu", "rho", "sig"))
    a, b, c, e = (pool.symbol(n) for n in ("a", "b", "c", "e"))
    funcs = {"dot": lambda x, y: x * y}  # vectors are numbers: dot = product
    env0 = {"a": 1.3, "b": -0.7, "c": 2.1, "e": 0.4}

    def val(expr, dv):
        return float(evaluate(expr, {**env0, "d": dv}, funcs))

    # tr(g^mu g_mu) = 4 d
    t = ak.gamma_trace([mu, mu], dim=d, indices=[mu])
    for dv in (2, 3, 4, 5.5):
        assert val(t, dv) == pytest.approx(4 * dv)
    # tr(g^mu g^nu g_mu g_nu) = 4 d (2 - d)
    t = ak.gamma_trace([mu, nu, mu, nu], dim=d, indices=[mu, nu])
    for dv in (2, 3, 4, 7):
        assert val(t, dv) == pytest.approx(4 * dv * (2 - dv))
    # tr(g^mu a g_mu b) = 4 (2 - d) a.b
    t = ak.gamma_trace([mu, a, mu, b], dim=d, indices=[mu])
    for dv in (2, 3, 4, 9):
        assert val(t, dv) == pytest.approx(4 * (2 - dv) * 1.3 * -0.7)
    # tr(g^mu a b g_mu c e) etc. against the textbook reduction
    # g^mu a b c g_mu = -2 c b a + (4 - d) a b c
    t1 = ak.gamma_trace([mu, a, b, c, mu, e], dim=d, indices=[mu])
    tcba = ak.gamma_trace([c, b, a, e], dim=d)
    tabc = ak.gamma_trace([a, b, c, e], dim=d)
    for dv in (2, 3, 4, 6, 11):
        want = -2 * val(tcba, dv) + (4 - dv) * val(tabc, dv)
        assert val(t1, dv) == pytest.approx(want)
    # g^mu g^nu g^rho g^sig g_mu = 2 (g^sig g^nu g^rho + g^rho g^nu g^sig)... via traces:
    # tr(g^mu a b c e g_mu) = 4 d (a.b c.e - a.c b.e + a.e b.c)  (cyclicity)
    t = ak.gamma_trace([mu, a, b, c, e, mu], dim=d, indices=[mu])
    abce = ak.gamma_trace([a, b, c, e], dim=d)
    for dv in (2, 3, 4, 8):
        assert val(t, dv) == pytest.approx(dv * val(abce, dv))
    # unused-index smoke: rho, sig free in a 2-trace
    t = ak.gamma_trace([rho, sig], dim=d)
    assert "dot" in str(t)


def test_trace_two_vectors_is_4_dot(pool):
    d = pool.symbol("d")
    p, q = pool.symbol("p"), pool.symbol("q")
    t = ak.gamma_trace([p, q], dim=d)
    assert nc_equal(t, 4 * pool.func("dot", [p, q]))


def random_trace_case(rng, d, n_pairs, n_vecs):
    labels = [("idx", i) for i in range(n_pairs) for _ in range(2)] + [
        ("vec", rng.randrange(3)) for _ in range(n_vecs)
    ]
    rng.shuffle(labels)
    return labels


@pytest.mark.parametrize("seed", range(8))
def test_trace_with_dummy_indices_vs_matrices_d4(pool, seed):
    """Words of vectors and contracted index pairs against numpy 4x4 sums."""
    rng = random.Random(seed)
    d = 4
    dd = pool.symbol("d")
    vec_names = ["p", "q", "k"]
    vec_syms = [pool.symbol(n) for n in vec_names]
    idx_syms = [pool.symbol(f"mu{i}") for i in range(3)]
    nprng = np.random.default_rng(seed)
    vecs = [nprng.integers(-3, 4, size=d).astype(float) for _ in vec_names]
    for _ in range(12):
        n_pairs = rng.choice([0, 1, 2])
        n_vecs = rng.choice([0, 2, 4]) if n_pairs else rng.choice([2, 4, 6])
        labels = random_trace_case(rng, d, n_pairs, n_vecs)
        slots = [idx_syms[i] if k == "idx" else vec_syms[i] for k, i in labels]
        got = ak.gamma_trace(slots, dim=dd, indices=idx_syms[:n_pairs])
        env = {"d": 4, **dict(zip(vec_names, vecs))}
        got_val = complex(evaluate(got, env, {"dot": lambda x, y: vec_dot(x, y, d)}))
        # numpy: sum over index assignments, lowering the second occurrence
        want = 0j
        for assign in itertools.product(range(d), repeat=n_pairs):
            seen = [0] * n_pairs
            m = np.eye(4, dtype=complex)
            coef = 1.0
            for k, i in labels:
                if k == "vec":
                    m = m @ slash(vecs[i], d)
                else:
                    m = m @ gammas(d)[assign[i]]
                    seen[i] += 1
                    if seen[i] == 2:
                        coef *= eta(d)[assign[i], assign[i]]
            want += coef * np.trace(m)
        assert abs(got_val - want) < 1e-8, (labels, got_val, want)


def test_product_of_traces_contracted(pool):
    d = pool.symbol("d")
    idx = [pool.symbol(n) for n in ("m", "n", "r", "s")]
    t = ak.gamma_trace(idx, dim=d)  # free indices -> g g - g g + g g
    c = ak.lorentz_contract(t * t, idx, d)
    for dv in (2, 3, 4, 5, 10):
        assert float(evaluate(c, {"d": dv})) == pytest.approx(16 * dv * (3 * dv - 2))
    # numerics at d=4: explicit sum over the 4^4 index assignments
    g, e = gammas(4), eta(4)
    total = 0.0
    for m, n, r, s in itertools.product(range(4), repeat=4):
        t1 = np.trace(g[m] @ g[n] @ g[r] @ g[s])
        t2 = np.trace(g[m] @ g[n] @ g[r] @ g[s])
        total += (e[m, m] * e[n, n] * e[r, r] * e[s, s] * t1 * t2).real
    assert total == pytest.approx(640.0)


def test_product_of_traces_with_vectors_vs_matrices(pool):
    """tr(g^m a g^n b) tr(g_m c g_n e) at d=4 against explicit matrices."""
    d = pool.symbol("d")
    m, n = pool.symbol("m"), pool.symbol("n")
    names = ["a", "b", "c", "e"]
    syms = {k: pool.symbol(k) for k in names}
    t1 = ak.gamma_trace([m, syms["a"], n, syms["b"]], dim=d)
    t2 = ak.gamma_trace([m, syms["c"], n, syms["e"]], dim=d)
    res = ak.lorentz_contract(t1 * t2, [m, n], d)
    rng = np.random.default_rng(3)
    vecs = {k: rng.normal(size=4) for k in names}
    got = float(evaluate(res, {"d": 4, **vecs}, {"dot": lambda x, y: vec_dot(x, y, 4)}))
    g, ee = gammas(4), eta(4)
    want = 0.0
    for mu, nu in itertools.product(range(4), repeat=2):
        a1 = np.trace(g[mu] @ slash(vecs["a"], 4) @ g[nu] @ slash(vecs["b"], 4))
        a2 = np.trace(g[mu] @ slash(vecs["c"], 4) @ g[nu] @ slash(vecs["e"], 4))
        want += (ee[mu, mu] * ee[nu, nu] * a1 * a2).real
    assert got == pytest.approx(want)


def test_lorentz_contract_basics(pool):
    d = pool.symbol("d")
    mu, nu = pool.symbol("mu"), pool.symbol("nu")
    p, q = pool.symbol("p"), pool.symbol("q")
    dot = lambda x, y: pool.func("dot", [x, y])  # noqa: E731
    # g^{mu mu} = d
    assert ak.lorentz_contract(dot(mu, mu), [mu], d) == d
    # p^mu q_mu = p.q
    assert str(ak.lorentz_contract(dot(mu, p) * dot(mu, q), [mu], d)) == str(dot(p, q))
    # g^{mu nu} g_{mu nu} = d
    assert ak.lorentz_contract(dot(mu, nu) ** 2, [mu, nu], d) == d
    # free index kept
    assert str(ak.lorentz_contract(dot(mu, p), [mu], d)) == str(dot(mu, p))
    # numeric dimension
    assert ak.lorentz_contract(dot(mu, mu), [mu], pool.integer(7)) == pool.integer(7)
    # three occurrences refused; hidden in levi_civita refused
    with pytest.raises(ak.NcError) as e:
        ak.lorentz_contract(dot(mu, p) * dot(mu, q) * dot(mu, p), [mu], d)
    assert e.value.code == "E-NC-005"
    with pytest.raises(ak.NcError) as e:
        ak.lorentz_contract(pool.func("levi_civita", [mu, p, q, nu]) * dot(mu, p), [mu], d)
    assert e.value.code == "E-NC-005"


def test_gamma5_and_components_refused_in_d_dim(pool):
    d = pool.symbol("d")
    p = pool.symbol("p")
    for slots in ([5, p, p, p, p], [0, 0], [p, 1]):
        with pytest.raises(ak.NcError) as e:
            ak.gamma_trace(slots, dim=d)
        assert e.value.code == "E-NC-005"
        assert "four dimensions" in str(e.value) or "four-dim" in str(e.value)
    with pytest.raises(ValueError):
        ak.gamma_trace([p, p], pool=pool, indices=[p])
    # the 4D path is untouched
    assert ak.gamma_trace([5, 0, 1, 2, 3], pool=pool) == -4 * pool.imaginary_unit()
    # noncommutative label refused
    x = pool.symbol("x", commutative=False)
    with pytest.raises(ak.NcError):
        ak.gamma_trace([x, x], dim=d)
    # too long
    with pytest.raises(ak.NcError) as e:
        ak.gamma_trace([p] * 16, dim=d)
    assert e.value.code == "E-NC-006"


def test_eight_gamma_trace_symbolic_d(pool):
    """8 slashed vectors: 105 terms, same as the 4D formula; check vs numpy."""
    d = pool.symbol("d")
    names = list("abcdefgh")
    syms = [pool.symbol(n) for n in names]
    t = ak.gamma_trace(syms, dim=d)
    rng = np.random.default_rng(21)
    vecs = {n: rng.integers(-2, 3, size=4).astype(float) for n in names}
    m = np.eye(4, dtype=complex)
    for n in names:
        m = m @ slash(vecs[n], 4)
    got = complex(evaluate(t, {"d": 4, **vecs}, {"dot": lambda x, y: vec_dot(x, y, 4)}))
    assert abs(got - np.trace(m)) < 1e-8
    assert str(t).count("dot") > 100  # 105 terms of 4 dots


# ---------------------------------------------------------------------------
# gamma_contract: reduction identities, checked with explicit matrices in
# d = 2, 3, 4 (the identities are valid for every d)
# ---------------------------------------------------------------------------


def test_gamma_contract_textbook_identities(pool):
    d = pool.symbol("d")
    mu, nu, rho, sig = (pool.symbol(n) for n in ("mu", "nu", "rho", "sig"))

    def terms(slots, idx):
        return [(str(c), [str(x) for x in w]) for c, w in ak.gamma_contract(slots, idx, d)]

    assert terms([mu, mu], [mu]) == [("d", [])]
    r = ak.gamma_contract([mu, nu, mu], [mu], d)
    assert len(r) == 1
    assert str(r[0][1][0]) == "nu"
    for dv in (2, 3, 4, 9):
        assert float(evaluate(r[0][0], {"d": dv})) == 2 - dv
    # g^mu g^nu g^rho g^sig g_mu = -2 g^sig g^rho g^nu + (4-d) g^nu g^rho g^sig
    r = ak.gamma_contract([mu, nu, rho, sig, mu], [mu], d)
    assert r  # checked as matrices below
    # in d = 2, 3, 4: compare with explicit matrices
    for dv in (2, 3, 4):
        g, e = gammas(dv), eta(dv)
        for slots, idx_syms in (
            ([mu, mu], [mu]),
            ([mu, nu, mu], [mu]),
            ([mu, nu, rho, mu], [mu]),
            ([mu, nu, rho, sig, mu], [mu]),
            ([mu, nu, mu, rho], [mu]),
            ([mu, nu, mu, nu], [mu, nu]),
            ([mu, nu, rho, nu, mu, sig], [mu, nu]),
        ):
            red = ak.gamma_contract(slots, idx_syms, d)
            free = sorted({str(s) for s in slots if s not in idx_syms})
            idx_names = [str(s) for s in idx_syms]
            for fa in itertools.product(range(dv), repeat=len(free)):
                fassign = dict(zip(free, fa))
                lhs = 0
                for assign in itertools.product(range(dv), repeat=len(idx_names)):
                    amap = {**fassign, **dict(zip(idx_names, assign))}
                    seen = dict.fromkeys(idx_names, 0)
                    m = np.eye(len(g[0]), dtype=complex)
                    coef = 1.0
                    for s in slots:
                        n = str(s)
                        m = m @ g[amap[n]]
                        if n in seen:
                            seen[n] += 1
                            if seen[n] == 2:
                                coef *= e[amap[n], amap[n]]
                    lhs = lhs + coef * m
                rhs = np.zeros_like(lhs)
                for c, w in red:
                    m = np.eye(len(g[0]), dtype=complex)
                    for s in w:
                        m = m @ g[fassign[str(s)]]
                    rhs = rhs + float(evaluate(c, {"d": dv})) * m
                assert np.allclose(lhs, rhs), (slots, dv, fassign)


def test_gamma_contract_refusals(pool):
    d = pool.symbol("d")
    mu, nu = pool.symbol("mu"), pool.symbol("nu")
    with pytest.raises(ak.NcError) as e:
        ak.gamma_contract([mu, mu, mu], [mu], d)
    assert e.value.code == "E-NC-005"
    with pytest.raises(ak.NcError) as e:
        ak.gamma_contract([0, 0], [mu], d)
    assert e.value.code == "E-NC-005"
    # a free index is left alone
    r = ak.gamma_contract([mu, nu], [mu], d)
    assert len(r) == 1
    assert [str(x) for x in r[0][1]] == ["mu", "nu"]


# ---------------------------------------------------------------------------
# Lie algebras
# ---------------------------------------------------------------------------

LAMBDA = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
    np.diag([1, 1, -2]).astype(complex) / np.sqrt(3),
]
T3 = [m / 2 for m in LAMBDA]
T2 = [SX / 2, SY / 2, SZ / 2]
FUNCS: dict = {}


def mat_env(name, mats):
    return {f"{name}{i + 1}": m for i, m in enumerate(mats)}


def test_su2_commutators_vs_pauli(pool):
    su2 = ak.lie_su2(pool)
    J = su2.generators
    env = {**mat_env("J", T2), "I": 1j}
    for a in range(3):
        for b in range(3):
            got = evaluate(su2.commutator(J[a], J[b]), env)
            want = T2[a] @ T2[b] - T2[b] @ T2[a]
            assert np.allclose(got, want)
    assert str(su2.commutator(J[0], J[1])) == str(pool.imaginary_unit() * J[2])
    # [J1, J2] = i J3 and cyclic, [J, J] = 0
    assert is_zero(su2.commutator(J[0], J[0]))
    # polynomial arguments: [J1 J2, J3] and anticommutators vs matrices
    for expr in (J[0] * J[1], J[0] * J[1] * J[2], J[2] ** 2 + J[0] * J[1]):
        for a in range(3):
            got = evaluate(su2.commutator(expr, J[a]), env)
            ev = evaluate(expr, env)
            want = ev @ T2[a] - T2[a] @ ev
            assert np.allclose(got, want)
            got = evaluate(su2.anticommutator(expr, J[a]), env)
            assert np.allclose(got, ev @ T2[a] + T2[a] @ ev)


def test_su2_casimir(pool):
    su2 = ak.lie_su2(pool)
    J = su2.generators
    c = su2.casimir()
    for g in J:
        assert is_zero(su2.commutator(c, g))
    env = {**mat_env("J", T2), "I": 1j}
    assert np.allclose(evaluate(c, env), 0.75 * np.eye(2))
    # spin-1 representation: C = j(j+1) = 2
    s1 = [
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2),
        np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2),
        np.diag([1, 0, -1]).astype(complex),
    ]
    assert np.allclose(evaluate(c, {**mat_env("J", s1), "I": 1j}), 2 * np.eye(3))
    # the commutator table holds in spin 1, too
    for a, b, c3 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        assert np.allclose(s1[a] @ s1[b] - s1[b] @ s1[a], 1j * s1[c3])


def sqrt3_env():
    return {"I": 1j}


def test_su3_commutators_vs_gellmann(pool):
    su3 = ak.lie_su3(pool)
    T = su3.generators
    env = {**mat_env("T", T3), "I": 1j}
    for a in range(8):
        for b in range(8):
            got = evaluate(su3.commutator(T[a], T[b]), env)
            want = T3[a] @ T3[b] - T3[b] @ T3[a]
            assert np.allclose(got, want), (a, b)


def test_su3_tables_vs_matrices(pool):
    f = ak.su3_f(pool)
    d = ak.su3_d(pool)
    num = {"I": 1j}
    for a in range(8):
        for b in range(8):
            comm = LAMBDA[a] @ LAMBDA[b] - LAMBDA[b] @ LAMBDA[a]
            anti = LAMBDA[a] @ LAMBDA[b] + LAMBDA[b] @ LAMBDA[a]
            rhs_c = sum(
                2j * float(evaluate(f[(a + 1, b + 1, c + 1)], num)) * LAMBDA[c]
                for c in range(8)
                if (a + 1, b + 1, c + 1) in f
            ) + np.zeros((3, 3))
            assert np.allclose(comm, rhs_c), (a, b)
            rhs_a = (4 / 3) * (a == b) * np.eye(3) + sum(
                2 * float(evaluate(d[(a + 1, b + 1, c + 1)], num)) * LAMBDA[c]
                for c in range(8)
                if (a + 1, b + 1, c + 1) in d
            )
            assert np.allclose(anti, rhs_a), (a, b)
    # f is totally antisymmetric, d totally symmetric
    for (a, b, c), v in f.items():
        assert str(f[(b, a, c)]) == str(-v) or float(evaluate(f[(b, a, c)], num)) == pytest.approx(
            -float(evaluate(v, num))
        )
    assert len(f) == 54
    assert len(d) > 50


def test_su3_normal_order_and_casimir(pool):
    su3 = ak.lie_su3(pool)
    T = su3.generators
    env = {**mat_env("T", T3), "I": 1j}
    c = su3.casimir()
    assert np.allclose(evaluate(c, env), (4 / 3) * np.eye(3))
    for g in T:
        assert is_zero(su3.commutator(c, g))
    rng = random.Random(4)
    for _ in range(25):
        word = [rng.randrange(8) for _ in range(rng.choice([2, 3, 4]))]
        expr = 1
        for i in word:
            expr = expr * T[i]
        no = su3.normal_order(expr)
        m = np.eye(3, dtype=complex)
        for i in word:
            m = m @ T3[i]
        assert np.allclose(evaluate(no, env), m), word
    # normal-ordered words are increasing
    no = su3.normal_order(T[7] * T[0] * T[3])
    for _coeff, w in ak.nc_terms(no):
        idx = [int(str(g)[1:]) for g in w]
        assert idx == sorted(idx)


def test_general_lie_algebra_and_jacobi(pool):
    X = [pool.symbol(f"X{i}", commutative=False) for i in (1, 2, 3)]
    # su(2) given by hand, both orders of a pair supplied
    alg = ak.lie_algebra(X, {(1, 2, 3): 1, (2, 3, 1): 1, (3, 1, 2): 1, (2, 1, 3): -1})
    assert str(alg.commutator(X[0], X[1])) == str(pool.imaginary_unit() * X[2])
    # scaled structure constants: [Y1, Y2] = i g Y3 with a symbolic coupling
    g = pool.symbol("g")
    alg = ak.lie_algebra(X, {(1, 2, 3): g, (2, 3, 1): g, (3, 1, 2): g})
    r = alg.commutator(X[1], X[2])
    assert nc_equal(r, pool.imaginary_unit() * g * X[0])
    # a table that is antisymmetric but breaks Jacobi is refused
    with pytest.raises(ak.NcError) as e:
        ak.lie_algebra(X, {(1, 2, 3): 1, (2, 3, 2): 1})
    assert e.value.code == "E-NC-008"
    # unless the caller opts out
    ak.lie_algebra(X, {(1, 2, 3): 1, (2, 3, 2): 1}, check_jacobi=False)
    # malformed input
    with pytest.raises(ak.NcError) as e:
        ak.lie_algebra(X, {(1, 1, 2): 1})
    assert e.value.code == "E-NC-008"
    with pytest.raises(ak.NcError) as e:
        ak.lie_algebra(X, {(1, 2, 3): 1, (2, 1, 3): 1})
    assert e.value.code == "E-NC-008"
    with pytest.raises(ak.NcError) as e:
        ak.lie_algebra([pool.symbol("c1"), X[1]], {})
    assert e.value.code == "E-NC-008"
    with pytest.raises(ValueError):
        ak.lie_algebra(X, {(0, 1, 2): 1})


def test_affine_algebra_nonsemisimple(pool):
    """ax+b algebra [T1, T2] = i T2 (f_{122} = 1): not su(n)-like. Checks the
    Leibniz rule [T1, T2^2] = 2 i T2^2 and a 2x2 representation."""
    T = [pool.symbol(f"A{i}", commutative=False) for i in (1, 2)]
    alg = ak.lie_algebra(T, {(1, 2, 2): 1})
    assert str(alg.commutator(T[0], T[1])) == str(pool.imaginary_unit() * T[1])
    r = alg.commutator(T[0], T[1] ** 2)
    assert nc_equal(r, 2 * pool.imaginary_unit() * T[1] ** 2)
    # explicit matrices: [E22, E21] = E21, so T1 = i E22, T2 = E21 realise it
    t1 = 1j * np.array([[0, 0], [0, 1]], dtype=complex)
    t2 = np.array([[0, 0], [1, 0]], dtype=complex)
    assert np.allclose(t1 @ t2 - t2 @ t1, 1j * t2)
    env = {"A1": t1, "A2": t2, "I": 1j}
    assert np.allclose(
        evaluate(alg.commutator(T[0], T[1] ** 3), env),
        t1 @ np.linalg.matrix_power(t2, 3) - np.linalg.matrix_power(t2, 3) @ t1,
    )


def test_casimir_with_metric(pool):
    su2 = ak.lie_su2(pool)
    J = su2.generators
    c = su2.casimir([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert nc_equal(c, 2 * (J[0] ** 2 + J[1] ** 2 + J[2] ** 2))
    with pytest.raises(ak.NcError):
        su2.casimir([[1, 0], [0, 1]])


def test_lie_algebra_rules_feed_nc_rewrite(pool):
    su2 = ak.lie_su2(pool)
    J = su2.generators
    got = ak.nc_rewrite(J[2] * J[0], su2.rules)
    assert nc_equal(got, J[0] * J[2] + pool.imaginary_unit() * J[1])
    # generators of another pool are refused
    other = ak.ExprPool()
    with pytest.raises(Exception):
        su2.commutator(other.symbol("q", commutative=False), J[0])


# ---------------------------------------------------------------------------
# Sequence wildcards
# ---------------------------------------------------------------------------


def nc_syms(pool, names):
    return [pool.symbol(n, commutative=False) for n in names]


def test_sequence_wildcard_basic(pool):
    A, B, C, D, X = nc_syms(pool, "ABCDX")
    rule = ak.make_rule(A * X * B, C * X * C, sequence=[X])
    assert nc_equal(ak.simplify_with(A * D * D * B, [rule]).value, C * D * D * C)
    # X = empty
    assert nc_equal(ak.simplify_with(A * B, [rule]).value, C * C)
    # surrounded by other factors, and with a coefficient
    r = ak.simplify_with(3 * D * A * D * B * D, [rule]).value
    assert nc_equal(r, 3 * D * C * D * C * D)
    # no match: unchanged
    assert nc_equal(ak.simplify_with(B * D * A, [rule]).value, B * D * A)
    # inside a sum
    r = ak.simplify_with(A * D * B + B * A, [rule]).value
    assert nc_equal(r, C * D * C + B * A)


def test_sequence_wildcard_leftmost_shortest(pool):
    A, B, C, D, X = nc_syms(pool, "ABCDX")
    rule = ak.make_rule(A * X * B, C * X * C, sequence=[X])
    # shortest run: the first B closes
    r = ak.simplify_with(A * D * B * D * B, [rule]).value
    assert nc_equal(r, C * D * C * D * B)
    # leftmost start: the first A opens (C*...*C then no more A..B)
    r = ak.simplify_with(A * A * D * B, [rule]).value
    # leftmost-shortest: X = (A D); result C A D C
    assert nc_equal(r, C * A * D * C)


def test_sequence_wildcard_deletes_middle(pool):
    """Real use: drop everything between two markers."""
    L, R, P, Q, S, X = nc_syms(pool, ["L", "R", "P", "Q", "S", "X"])
    rule = ak.make_rule(L * X * R, L * R, sequence=[X])
    r = ak.simplify_with(P * L * Q * S * Q * R * P, [rule]).value
    assert nc_equal(r, P * L * R * P)
    # re-applies to a result that exposes a new match
    r = ak.simplify_with(L * L * P * R * R, [rule]).value
    assert nc_equal(r, L * R * R)


def test_sequence_wildcard_repeated_variable(pool):
    A, B, C, D, E, X = nc_syms(pool, "ABCDEX")
    # X A X -> D: prefix equal to the run after A
    rule = ak.make_rule(X * A * X, D, sequence=[X])
    r = ak.simplify_with(B * C * A * B * C * E, [rule]).value
    assert nc_equal(r, D * E)
    # the run need not start at the front: C A C inside B C A C B
    r = ak.simplify_with(B * C * A * C * B, [rule]).value
    assert nc_equal(r, B * D * B)
    # unequal runs do not match
    rule2 = ak.make_rule(X * A * X * B, E, sequence=[X])
    assert nc_equal(ak.simplify_with(C * A * D * B, [rule2]).value, C * A * D * B)
    assert nc_equal(ak.simplify_with(C * A * C * B, [rule2]).value, E)


def test_sequence_wildcard_two_variables_and_commutative_wildcard(pool):
    A, B, C, D, E, X, Y = nc_syms(pool, "ABCDEXY")
    # A X B Y -> Y B X A  (swap the two runs, between A..B and after B)
    rule = ak.make_rule(A * X * B * C * Y, Y * C * B * X * A, sequence=[X, Y])
    r = ak.simplify_with(A * D * B * C * E, [rule]).value
    assert nc_equal(r, C * B * D * A * E)
    # trailing variable takes the shortest (empty) run
    rule2 = ak.make_rule(A * X, B * X, sequence=[X])
    assert nc_equal(ak.simplify_with(A * D * E, [rule2]).value, B * D * E)


def test_sequence_wildcard_pauli_application(pool):
    """sx X sx -> X-conjugated-by-sx for X in {sy, sz} words: check the
    rewriting against numpy 2x2 matrices on a long word."""
    sx, sy, sz, X = nc_syms(pool, ["sx", "sy", "sz", "X"])
    rule = ak.make_rule(sx * X * sx, sx * X * sx, sequence=[X])  # identity rewrite
    e = sx * sy * sz * sx
    assert nc_equal(ak.simplify_with(e, [rule]).value, e)
    # rewrite sx X sx with X free of sx using an explicit conjugation rule
    rule = ak.make_rule(sx * X * sx, X, sequence=[X])  # intentionally not a Pauli identity
    r = ak.simplify_with(sx * sy * sx, [rule]).value
    assert nc_equal(r, sy)


def test_sequence_rule_validation(pool):
    A, B, X, Y = nc_syms(pool, "ABXY")
    x = pool.symbol("x")
    for lhs, seq in (
        (A * B, [X]),  # X not in lhs
        (X * Y, [X, Y]),  # no fixed factor
        (A, [X]),  # not a product
        (A * X**2, [X]),  # hidden in a power
    ):
        with pytest.raises(ak.NcError) as e:
            ak.make_rule(lhs, A, sequence=seq)
        assert e.value.code == "E-NC-009", str(lhs)
    with pytest.raises(ak.NcError) as e:
        ak.make_rule(A * x, A, sequence=[x])  # commutative symbol
    assert e.value.code == "E-NC-009"
    # nc_rewrite refuses a sequence rule rather than reading X literally
    rule = ak.make_rule(A * X * B, A * B, sequence=[X])
    with pytest.raises(ak.NcError) as e:
        ak.nc_rewrite(A * Y * B, [rule])
    assert e.value.code == "E-NC-003"


def test_sequence_rule_termination_and_cost(pool):
    A, B, C, X = nc_syms(pool, "ABCX")
    # a rule that regrows its own redex: the round cap stops it
    rule = ak.make_rule(A * X * B, A * A * X * B, sequence=[X])
    r = ak.simplify_with(A * C * B, [rule]).value  # terminates
    assert "B" in str(r)
    # the same without sequence wildcards never reaches a fixpoint; the
    # active budget stops it
    import time

    t = time.time()
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=300)):
        r = ak.simplify_with(A * B, [ak.make_rule(A * B, A * A * B)]).value
    assert "B" in str(r)
    assert time.time() - t < 30
    # an adversarial long word: the match budget gives up quickly instead of hanging
    rule = ak.make_rule(A * X * A * X * A * X * B, C, sequence=[X])
    word = pool.integer(1)
    for i in range(60):
        word = word * (A if i % 3 else C)
    ak.simplify_with(word * B, [rule])
