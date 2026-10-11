"""G10c — noncommutative algebra: rules inside products, Pauli words, ladder
operators, Dirac traces. Every result is checked against an independent oracle:
explicit matrices (numpy), differential operators (SymPy) or SymPy's own
``normal_ordered_form``."""

from __future__ import annotations

import itertools
import random

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")
sympy = pytest.importorskip("sympy")


# ---------------------------------------------------------------------------
# Evaluating an Expr with matrices / operators in place of symbols
# ---------------------------------------------------------------------------


def evaluate(expr, env, funcs=None):
    """Evaluate ``expr`` with ``env[name]`` for symbols (numbers, matrices or
    any objects supporting ``+``/``*``/``@``), keeping product order."""
    funcs = funcs or {}
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
            return np.linalg.matrix_power(base, n)
        return base**n
    if tag == "func":
        return funcs[node[1]](*[evaluate(a, env, funcs) for a in node[2]])
    raise AssertionError(f"unexpected node {node}")


@pytest.fixture
def pool():
    return ak.ExprPool()


def nc(pool, *names):
    return [pool.symbol(n, commutative=False) for n in names]


def nc_equal(a, b):
    """Equal as noncommutative polynomials."""
    return ak.nc_terms(a - b) == []


# ---------------------------------------------------------------------------
# Report repro: make_rule inside n-ary products
# ---------------------------------------------------------------------------


def apply_ops(expr, f, x, hbar):
    """Apply an expression in X (multiply by x) and P (-i hbar d/dx) to f(x)."""
    node = expr.node()
    tag = node[0]
    if tag == "symbol" and node[1] == "X":
        return x * f
    if tag == "symbol" and node[1] == "P":
        return -sympy.I * hbar * sympy.diff(f, x)
    if tag == "add":
        return sum(apply_ops(a, f, x, hbar) for a in node[1])
    if tag == "mul":
        out = f
        for a in reversed(node[1]):  # rightmost operator acts first
            out = apply_ops(a, out, x, hbar)
        return out
    if tag == "pow":
        out = f
        for _ in range(int(node[2].node()[1])):
            out = apply_ops(node[1], out, x, hbar)
        return out
    # commutative scalar
    scal = sympy.sympify(str(expr).replace("hbar", "hb"), locals={"hb": hbar, "I": sympy.I})
    return scal * f


def test_qm2_rule_fires_inside_longer_product(pool):
    X, P = nc(pool, "X", "P")
    hbar = pool.symbol("hbar")
    i_ = pool.imaginary_unit()
    rule = ak.make_rule(X * P, P * X + i_ * hbar)
    # bare X*P (worked before) and X*P*P (did not)
    assert nc_equal(ak.simplify_with(X * P, [rule]).value, P * X + i_ * hbar)
    out = ak.simplify_with(X * P * P - P * P * X, [rule]).value
    assert nc_equal(out, 2 * i_ * hbar * P), str(out)

    # Oracle: X = x·, P = -iħ d/dx acting on a test function.
    x, hb = sympy.symbols("x hb")
    f = sympy.exp(-(x**2)) * (1 + x + x**3)
    lhs = apply_ops(X * P * P - P * P * X, f, x, hb)
    rhs = apply_ops(out, f, x, hb)
    assert sympy.simplify(lhs - rhs) == 0


def test_commutator_with_rules_matches_differential_operators(pool):
    X, P = nc(pool, "X", "P")
    hbar = pool.symbol("hbar")
    i_ = pool.imaginary_unit()
    rule = ak.make_rule(X * P, P * X + i_ * hbar)
    x, hb = sympy.symbols("x hb")
    f = sympy.exp(-(x**2)) * (2 + x**2)
    for a, b in [(X, P**2), (X**2, P**2), (X * P, P * X), (X**3, P)]:
        c = ak.commutator(a, b, [rule])
        direct = apply_ops(a * b - b * a, f, x, hb)
        assert sympy.simplify(apply_ops(c, f, x, hb) - direct) == 0, str(c)
        # normal form: every word is P^m X^n
        for _, word in ak.nc_terms(c):
            names = [w.node()[1] for w in word]
            assert names == sorted(names, key=lambda s: s != "P"), names
    assert nc_equal(ak.commutator(X, P**2, [rule]), 2 * i_ * hbar * P)


def test_lowercase_noncommutative_symbols_are_literal_in_rules(pool):
    # The QM3 probe: names starting with a lower-case letter used to be wildcards.
    a, ad = nc(pool, "a", "ad")
    rule = ak.make_rule(a * ad, ad * a + 1)
    out = ak.simplify_with(a * ad * ad * ad - ad * ad * ad * a, [rule]).value
    assert nc_equal(out, 3 * ad**2), str(out)


def test_commutative_rule_fires_on_sub_multiset(pool):
    A, B, C, D = (pool.symbol(n) for n in "ABCD")
    rule = ak.make_rule(A * B, C)
    assert ak.simplify_with(A * B * D, [rule]).value == C * D
    # the QM2b control from the report
    rule2 = ak.make_rule(A * B, B + 1)
    out = ak.simplify_with(A * B * B, [rule2]).value
    assert ak.simplify(out - (B + 1) * B).value == pool.integer(0)


def test_match_pattern_respects_order(pool):
    X, P = nc(pool, "X", "P")
    assert ak.match_pattern(X * P, X * P * P) != []
    assert ak.match_pattern(X * P, P * X) == []
    A, B, D = (pool.symbol(n) for n in "ABD")
    assert ak.match_pattern(A * B, A * B * D) != []


def test_nc_rewrite_refuses_non_terminating_rules(pool):
    A, B = nc(pool, "A", "B")
    rules = [ak.make_rule(A * B, B * A), ak.make_rule(B * A, A * B)]
    with pytest.raises(ak.NcError) as e:
        ak.nc_rewrite(A * B, rules, max_steps=500)
    assert e.value.code == "E-NC-001"
    with pytest.raises(ak.NcError) as e:
        ak.nc_rewrite(A, [ak.make_rule(A + B, A)])
    assert e.value.code == "E-NC-003"


def test_nc_expand_against_sympy(pool):
    X, Y, Z = nc(pool, "X", "Y", "Z")
    sX, sY, sZ = sympy.symbols("X Y Z", commutative=False)
    c = pool.symbol("c")
    sc = sympy.Symbol("c")
    cases = [
        ((X + Y) ** 3, (sX + sY) ** 3),
        ((X + c * Y) * (Z - X) * (Y + 2), (sX + sc * sY) * (sZ - sX) * (sY + 2)),
        ((X * Y - Y * X) ** 2, (sX * sY - sY * sX) ** 2),
    ]
    for ours, theirs in cases:
        terms = ak.nc_terms(ours)
        got = sum(
            sympy.sympify(str(co))
            * sympy.Mul(*[{"X": sX, "Y": sY, "Z": sZ}[w.node()[1]] for w in word])
            for co, word in terms
        )
        assert sympy.expand(got - sympy.expand(theirs)) == 0


# ---------------------------------------------------------------------------
# Pauli algebra vs explicit 2x2 matrices
# ---------------------------------------------------------------------------

SIGMA = {
    "sx": np.array([[0, 1], [1, 0]], dtype=complex),
    "sy": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "sz": np.array([[1, 0], [0, -1]], dtype=complex),
    "I": 1j,
}


def test_report_pauli_repros(pool):
    sx, sy, sz = nc(pool, "sx", "sy", "sz")
    i_ = pool.imaginary_unit()
    assert ak.simplify_pauli(sz * sz).value == pool.integer(1)
    assert ak.simplify_pauli(sz**2).value == pool.integer(1)
    assert nc_equal(ak.simplify_pauli(sx * sy * sx).value, -sy)
    assert ak.simplify_pauli((sx + sy) ** 2).value == pool.integer(2)
    assert nc_equal(ak.simplify_pauli(sx * sy - sy * sx).value, 2 * i_ * sz)
    assert nc_equal(ak.simplify_pauli(sx * sy).value, i_ * sz)


def test_pauli_words_against_matrices(pool):
    sx, sy, sz = nc(pool, "sx", "sy", "sz")
    syms = {"sx": sx, "sy": sy, "sz": sz}
    rng = random.Random(7)
    exprs = []
    for n in range(1, 8):
        for _ in range(12):
            w = [rng.choice("xyz") for _ in range(n)]
            e = syms["s" + w[0]]
            for k in w[1:]:
                e = e * syms["s" + k]
            exprs.append(e)
    c = pool.symbol("c")
    exprs += [(sx + 2 * sy - sz) ** 3, (sx * sy + sz) ** 2 - sx, sz**5 * sx**3, (sx + sy) ** 4]
    for e in exprs:
        r = ak.simplify_pauli(e).value
        for _, word in ak.nc_terms(r):
            assert len(word) <= 1, str(r)
        assert np.allclose(evaluate(r, SIGMA) * np.eye(2), evaluate(e, SIGMA) * np.eye(2)), (
            str(e),
            str(r),
        )
    # symbolic coefficients ride along
    r = ak.simplify_pauli((c * sx + sy) ** 2).value
    assert ak.simplify(r - (c**2 + 1)).value == pool.integer(0)


# ---------------------------------------------------------------------------
# Ladder operators
# ---------------------------------------------------------------------------


def to_sympy_boson(expr, ops):
    node = expr.node()
    tag = node[0]
    if tag == "symbol":
        if node[1] in ops:
            return ops[node[1]]
        if node[1] == "I":
            return sympy.I
        return sympy.Symbol(node[1])
    if tag == "integer":
        return sympy.Integer(int(node[1]))
    if tag == "add":
        return sympy.Add(*[to_sympy_boson(a, ops) for a in node[1]])
    if tag == "mul":
        return sympy.Mul(*[to_sympy_boson(a, ops) for a in node[1]])
    if tag == "pow":
        return to_sympy_boson(node[1], ops) ** int(node[2].node()[1])
    raise AssertionError(node)


def test_boson_normal_order_against_sympy(pool):
    from sympy.physics.quantum import Dagger
    from sympy.physics.quantum.boson import BosonOp
    from sympy.physics.quantum.operatorordering import normal_ordered_form

    a, ad, b, bd = nc(pool, "a", "ad", "b", "bd")
    A, B = BosonOp("A"), BosonOp("B")
    ops = {"a": A, "ad": Dagger(A), "b": B, "bd": Dagger(B)}
    cases = [
        a * ad,
        a * ad**3,
        a**2 * ad**2,
        a * ad * a * ad,
        (a + ad) ** 4,
        a * b * ad * bd,
        (a + bd) * (ad + b) * a,
        ak.commutator(a, ad**3),
    ]
    for e in cases:
        ours = ak.normal_order(e, bosons=[(a, ad), (b, bd)])
        theirs = normal_ordered_form(sympy.expand(to_sympy_boson(e, ops)), independent=True)
        diff = sympy.expand(to_sympy_boson(ours, ops) - theirs)
        diff = normal_ordered_form(diff, independent=True)
        assert sympy.expand(diff) == 0, (str(e), str(ours), theirs)
    # number operator: [N, a] = -a, [N, a†] = a†
    N = ad * a
    assert nc_equal(ak.normal_order(ak.commutator(N, a), bosons=(a, ad)), -a)
    assert nc_equal(ak.normal_order(ak.commutator(N, ad), bosons=(a, ad)), ad)


def truncated_boson(dim):
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    return a, a.T.copy()


def test_boson_vacuum_expectation_against_matrices(pool):
    a, ad = nc(pool, "a", "ad")
    A, Ad = truncated_boson(12)
    env = {"a": A, "ad": Ad}
    for word in itertools.product("ab", repeat=6):
        e = pool.integer(1)
        for ch in word:
            e = e * (a if ch == "a" else ad)
        v = ak.vacuum_expectation(e, bosons=(a, ad))
        expect = (evaluate(e, env) * np.eye(12))[0, 0]
        assert abs(int(v.node()[1]) - expect) < 1e-9, (word, str(v))
    assert ak.vacuum_expectation(a * a * ad * ad, bosons=(a, ad)) == pool.integer(2)


def jordan_wigner():
    sm = np.array([[0, 1], [0, 0]], dtype=float)  # annihilates |1> -> |0>
    sz = np.diag([1.0, -1.0])
    i2 = np.eye(2)
    c1 = np.kron(sm, i2)
    c2 = np.kron(sz, sm)
    return c1, c1.T.copy(), c2, c2.T.copy()


def test_fermion_normal_order_against_matrices(pool):
    c, cd, d, dd = nc(pool, "c", "cd", "d", "dd")
    C1, C1d, C2, C2d = jordan_wigner()
    env = {"c": C1, "cd": C1d, "d": C2, "dd": C2d}
    syms = {"c": c, "cd": cd, "d": d, "dd": dd}
    rng = random.Random(3)
    for n in range(1, 7):
        for _ in range(25):
            names = [rng.choice(list(syms)) for _ in range(n)]
            e = syms[names[0]]
            for k in names[1:]:
                e = e * syms[k]
            ours = ak.normal_order(e, fermions=[(c, cd), (d, dd)])
            assert np.allclose(evaluate(ours, env) * np.eye(4), evaluate(e, env) * np.eye(4)), (
                names,
                str(ours),
            )
            # normal order: no annihilator left of a creator
            for _, word in ak.nc_terms(ours):
                kinds = [w.node()[1].endswith("d") and w.node()[1] != "d" for w in word]
                assert kinds == sorted(kinds, reverse=True), (names, str(ours))
            vac = ak.vacuum_expectation(e, fermions=[(c, cd), (d, dd)])
            expect = (evaluate(e, env) * np.eye(4))[0, 0]
            assert abs(float(vac.node()[1]) - expect) < 1e-12
    assert ak.normal_order(c * c, fermions=(c, cd)) == pool.integer(0)
    assert nc_equal(ak.normal_order(c * cd, fermions=(c, cd)), 1 - cd * c)


def test_ladder_refusals(pool):
    a, ad, X = nc(pool, "a", "ad", "X")
    with pytest.raises(ak.NcError) as e:
        ak.normal_order(a, bosons=(pool.symbol("b"), ad))
    assert e.value.code == "E-NC-004"
    with pytest.raises(ak.NcError) as e:
        ak.vacuum_expectation(a * X * ad, bosons=(a, ad))
    assert e.value.code == "E-NC-007"


# ---------------------------------------------------------------------------
# Dirac traces vs explicit 4x4 matrices
# ---------------------------------------------------------------------------

ETA = np.diag([1.0, -1.0, -1.0, -1.0])


def dirac_gammas():
    s = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    z = np.zeros((2, 2))
    i2 = np.eye(2)
    g = [np.block([[i2, z], [z, -i2]]).astype(complex)]
    g += [np.block([[z, sk], [-sk, z]]) for sk in s]
    g5 = 1j * g[0] @ g[1] @ g[2] @ g[3]
    return g, g5


GAMMA, GAMMA5 = dirac_gammas()


def levi_civita_upper(i, j, k, m):
    perm = [i, j, k, m]
    if len(set(perm)) < 4:
        return 0
    sign = 1
    for a in range(4):
        for b in range(a + 1, 4):
            if perm[a] > perm[b]:
                sign = -sign
    return sign  # eps^{0123} = +1


def test_explicit_index_traces(pool):
    rng = random.Random(11)
    words = [list(w) for n in range(0, 5) for w in itertools.product([0, 1, 2, 3, 5], repeat=n)]
    words += [[rng.choice([0, 1, 2, 3, 5]) for _ in range(n)] for n in (6, 8) for _ in range(150)]
    for w in words:
        m = np.eye(4, dtype=complex)
        for k in w:
            m = m @ (GAMMA5 if k == 5 else GAMMA[k])
        got = ak.gamma_trace(w, pool=pool)
        val = complex(evaluate(got, {"I": 1j}))
        assert abs(val - np.trace(m)) < 1e-9, (w, str(got))
    assert ak.gamma_trace([5, 0, 1, 2, 3], pool=pool) == -4 * pool.imaginary_unit()


def test_symbolic_traces_against_matrices(pool):
    rng = np.random.default_rng(5)
    names = ["p", "q", "k", "l"]
    syms = {n: pool.symbol(n) for n in names}
    vecs = {n: rng.integers(-3, 4, size=4).astype(float) for n in names}  # contravariant

    def slash(v):
        low = ETA @ v
        return sum(low[m] * GAMMA[m] for m in range(4))

    def comp(x):  # a dot()/levi_civita() argument: index int or vector name
        return x

    def dot(x, y):
        def vec(z):
            if isinstance(z, str):
                return vecs[z]
            e = np.zeros(4)  # explicit index k stands for gamma^k: covariant unit vector
            e[int(z)] = 1.0
            return ETA @ e  # contravariant components of e^k

        return vec(x) @ ETA @ vec(y)

    def eps(*args):
        lows = []
        for z in args:
            if isinstance(z, str):
                lows.append(ETA @ vecs[z])
            else:
                e = np.zeros(4)
                e[int(z)] = 1.0
                lows.append(e)
        return sum(
            levi_civita_upper(i, j, k, m) * lows[0][i] * lows[1][j] * lows[2][k] * lows[3][m]
            for i in range(4)
            for j in range(4)
            for k in range(4)
            for m in range(4)
        )

    env = {"I": 1j, **{n: n for n in names}}
    funcs = {"dot": dot, "levi_civita": eps}
    slots_pool = [*names, 0, 1, 2, 3, 5]
    r = random.Random(9)
    cases = [
        ["p", "q"],
        ["p", "q", "k", "l"],
        [5, "p", "q", "k", "l"],
        ["p", 5, "q", "k", 0, "l", 2],
    ]
    cases += [[r.choice(slots_pool) for _ in range(n)] for n in (4, 5, 6) for _ in range(40)]
    for w in cases:
        m = np.eye(4, dtype=complex)
        for s in w:
            if s == 5:
                m = m @ GAMMA5
            elif isinstance(s, int):
                m = m @ GAMMA[s]
            else:
                m = m @ slash(vecs[s])
        slots = [syms[s] if isinstance(s, str) else s for s in w]
        got = ak.gamma_trace(slots, pool=pool)
        val = complex(evaluate(got, env, funcs))
        assert abs(val - np.trace(m)) < 1e-8, (w, str(got))
    t = ak.gamma_trace([5, *(syms[n] for n in names)])
    assert "levi_civita" in str(t)
    assert "I" in str(t)


def test_gamma_trace_refusals(pool):
    p = pool.symbol("p")
    with pytest.raises(ak.NcError) as e:
        ak.gamma_trace([4], pool=pool)
    assert e.value.code == "E-NC-005"
    with pytest.raises(ak.NcError) as e:
        ak.gamma_trace([p] * 16)
    assert e.value.code == "E-NC-006"
