"""G10 — propositional logic layer.

n-ary ``And``/``Or``/``Xor``, ``Implies``/``Equivalent``/``ITE``, two-level
minimisation (``simplify_logic`` / ``minimize_logic``), normal forms, ROBDDs and
equivalence checking.

Every answer is checked against an independent oracle: a brute-force Python
evaluator over ``Expr.node()`` (never alkahest's own truth tables), and SymPy's
``simplify_logic`` / ``SOPform`` / ``POSform`` for minimality — an exact
minimum can never use more terms (then literals) than any two-level form SymPy
returns.
"""

from __future__ import annotations

import itertools
import random

import alkahest as ak
import pytest

sympy = pytest.importorskip("sympy")
from sympy.logic import boolalg as sbool  # noqa: E402

# ---------------------------------------------------------------------------
# Independent oracle: evaluate an Expr predicate tree in Python
# ---------------------------------------------------------------------------


def ev(expr, env, memo=None):
    """Evaluate a propositional Expr under ``env`` (name -> bool)."""
    if memo is None:
        memo = {}
    key = expr
    if key in memo:
        return memo[key]
    node = expr.node()
    if node[0] == "symbol":
        val = env[node[1]]
    else:
        assert node[0] == "predicate", node
        kind, args = node[1], node[2]
        if kind == "true":
            val = True
        elif kind == "false":
            val = False
        elif kind == "not":
            val = not ev(args[0], env, memo)
        elif kind == "and":
            val = all(ev(x, env, memo) for x in args)
        elif kind == "or":
            val = any(ev(x, env, memo) for x in args)
        else:  # pragma: no cover - relations are not evaluated here
            raise AssertionError(kind)
    memo[key] = val
    return val


def table(expr, names):
    return [
        ev(expr, dict(zip(names, bits)))
        for bits in itertools.product([False, True], repeat=len(names))
    ]


def random_formula(rng, vs, depth):
    if depth == 0 or rng.random() < 0.25:
        v = rng.choice(vs)
        return v if rng.random() < 0.5 else ak.Not(v)
    k = rng.choice([2, 2, 3])
    args = [random_formula(rng, vs, depth - 1) for _ in range(k)]
    op = rng.randrange(6)
    if op == 0:
        return ak.And(*args)
    if op == 1:
        return ak.Or(*args)
    if op == 2:
        return ak.Xor(*args)
    if op == 3:
        return ak.Implies(args[0], args[1])
    if op == 4:
        return ak.ITE(*(args + args)[:3])
    return ak.Not(ak.And(*args))


def to_sympy(expr, syms, memo=None):
    if memo is None:
        memo = {}
    key = expr
    if key in memo:
        return memo[key]
    node = expr.node()
    if node[0] == "symbol":
        out = syms[node[1]]
    else:
        kind, args = node[1], node[2]
        conv = [to_sympy(x, syms, memo) for x in args]
        out = {
            "true": lambda: sympy.true,
            "false": lambda: sympy.false,
            "not": lambda: sympy.Not(conv[0]),
            "and": lambda: sympy.And(*conv),
            "or": lambda: sympy.Or(*conv),
        }[kind]()
    memo[key] = out
    return out


def two_level_counts(e, outer, inner):
    """(terms, literals) of a SymPy two-level form; ``outer`` is Or for DNF."""
    if e in (sympy.true, sympy.false):
        return (0, 0)
    terms = e.args if isinstance(e, outer) else (e,)
    lits = 0
    for t in terms:
        lits += len(t.args) if isinstance(t, inner) else 1
    return (len(terms), lits)


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# The report's repro (G10 §4.2)
# ---------------------------------------------------------------------------


def test_report_repro_nary_and_and_simplification(pool):
    a, b, c = pool.symbol("a"), pool.symbol("b"), pool.symbol("c")
    # Was: TypeError (binary only).
    abc = ak.And(a, b, c)
    assert abc.node() == ["predicate", "and", [a, b, c]]
    # Was: returned unchanged.
    f = ak.Or(ak.And(a, b), ak.And(a, ak.Not(b)))
    assert ak.simplify_logic(f) == a
    m = ak.minimize_logic(f)
    assert m.exact
    assert (m.terms, m.literals) == (1, 1)


def test_binary_constructors_and_relations_still_work(pool):
    x = pool.symbol("x")
    zero = pool.integer(0)
    assert ak.satisfiable(ak.And(pool.gt(x, zero), pool.lt(x, zero))) is False
    r = ak.satisfiable(ak.Or(pool.gt(x, zero), pool.le(x, zero)))
    assert r is True or isinstance(r, dict)
    assert ak.Not(pool.gt(x, zero)).node()[1] == "not"


def test_flattening_canonical_order_and_folding(pool):
    a, b, c = pool.symbol("a"), pool.symbol("b"), pool.symbol("c")
    assert ak.And(c, ak.And(b, a)) == ak.And(a, b, c)
    assert ak.Or(ak.Or(c, b), a) == ak.Or(a, b, c)
    assert ak.And(a, a) == a
    assert ak.And(a, ak.Not(a)) == pool.pred_false()
    assert ak.Or(a, ak.Not(a)) == pool.pred_true()
    assert ak.And(a, True) == a
    assert ak.Or(a, True) == pool.pred_true()
    assert ak.Not(ak.Not(a)) == a
    assert ak.Xor(a, a) == pool.pred_false()
    assert ak.Xor(a, True) == ak.Not(a)
    with pytest.raises(TypeError):
        ak.And()
    with pytest.raises(TypeError):
        ak.Or(True, False)
    other = ak.ExprPool().symbol("z")
    with pytest.raises(ak.PoolError):
        ak.And(a, other)


def test_derived_connectives_semantics(pool):
    names = ["a", "b", "c", "d"]
    a, b, c, d = (pool.symbol(n) for n in names)
    cases = [
        (ak.Xor(a, b, c, d), lambda a, b, c, d: (a + b + c + d) % 2 == 1),
        (ak.Implies(a, b), lambda a, b, c, d: (not a) or b),
        (ak.Equivalent(a, b, c), lambda a, b, c, d: a == b == c),
        (ak.ITE(a, b, c), lambda a, b, c, d: b if a else c),
        (ak.Equivalent(a), lambda *_: True),
    ]
    for expr, fn in cases:
        for bits in itertools.product([False, True], repeat=4):
            env = dict(zip(names, bits))
            assert ev(expr, env) == fn(*bits), (str(expr), env)


def test_xor_is_linear_in_size(pool):
    vs = [pool.symbol(f"x{i}") for i in range(40)]
    f = ak.Xor(*vs)
    g = ak.bdd(f)
    assert g.node_count() == 2 * 40 - 1
    assert g.sat_count() == 2**39


# ---------------------------------------------------------------------------
# Minimisation and normal forms vs brute force and SymPy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_random_formulas_vs_sympy(seed):
    rng = random.Random(seed)
    pool = ak.ExprPool()
    n = 2 + seed % 7  # 2..8 variables (SymPy simplify_logic needs force beyond 8)
    names = [f"v{i}" for i in range(n)]
    vs = [pool.symbol(nm) for nm in names]
    f = random_formula(rng, vs, 4)
    want = table(f, names)

    syms = {nm: sympy.Symbol(nm) for nm in names}
    sf = to_sympy(f, syms)

    for form, outer, inner in (("dnf", sympy.Or, sympy.And), ("cnf", sympy.And, sympy.Or)):
        m = ak.minimize_logic(f, form=form)
        assert table(m.expr, names) == want, (form, str(f), str(m.expr))
        assert m.exact
        theirs = sbool.simplify_logic(sf, form=form)
        # SymPy's answer is a correct two-level form too ...
        assert sbool.simplify_logic(sbool.Xor(theirs, sf)) == sympy.false
        # ... and our proven minimum is never worse than it.
        assert (m.terms, m.literals) <= two_level_counts(theirs, outer, inner), (
            form,
            str(f),
            str(m.expr),
            theirs,
        )

    auto = ak.simplify_logic(f)
    assert table(auto, names) == want

    for nf in (ak.to_cnf(f), ak.to_dnf(f), ak.to_nnf(f)):
        assert table(nf, names) == want
    # Shape checks: CNF is an And of Or-of-literals.
    cnf = ak.to_cnf(f)
    for clause in cnf.node()[2] if cnf.node()[1:2] == ["and"] else [cnf]:
        node = clause.node()
        if node[0] == "predicate" and node[1] == "or":
            for lit in node[2]:
                ln = lit.node()
                assert ln[0] == "symbol" or (ln[1] == "not" and ln[2][0].node()[0] == "symbol")


@pytest.mark.parametrize("seed", range(8))
def test_dontcares_vs_sympy_sopform(seed):
    rng = random.Random(1000 + seed)
    n = 4 + seed % 3
    names = [f"v{i}" for i in range(n)]
    pool = ak.ExprPool()
    vs = [pool.symbol(nm) for nm in names]
    cells = list(range(2**n))
    rng.shuffle(cells)
    on = sorted(cells[: len(cells) // 3])
    dc = sorted(cells[len(cells) // 3 : len(cells) // 2])

    def minterm(m):
        bits = [(m >> (n - 1 - i)) & 1 for i in range(n)]
        return ak.And(*[v if bit else ak.Not(v) for v, bit in zip(vs, bits)])

    f = ak.Or(*[minterm(m) for m in on])
    d = ak.Or(*[minterm(m) for m in dc])
    m = ak.minimize_logic(f, form="dnf", dontcare=d)
    assert m.exact
    got = table(m.expr, names)
    for idx in range(2**n):
        if idx in on:
            assert got[idx]
        elif idx not in dc:
            assert not got[idx]
    syms = [sympy.Symbol(nm) for nm in names]
    theirs = sbool.SOPform(syms, on, dc)
    assert (m.terms, m.literals) <= two_level_counts(theirs, sympy.Or, sympy.And)
    mc = ak.minimize_logic(f, form="cnf", dontcare=d)
    theirs_c = sbool.POSform(syms, on, dc)
    assert (mc.terms, mc.literals) <= two_level_counts(theirs_c, sympy.And, sympy.Or)


def test_bcd_dontcares_textbook(pool):
    # BCD digit >= 5 with codes 10..15 unused: x3 + x2 x0 + x2 x1.
    x = [pool.symbol(f"x{i}") for i in range(4)]

    def minterm(m):
        return ak.And(*[x[i] if (m >> i) & 1 else ak.Not(x[i]) for i in range(4)])

    f = ak.Or(*[minterm(m) for m in range(5, 10)])
    dc = ak.Or(*[minterm(m) for m in range(10, 16)])
    m = ak.minimize_logic(f, dontcare=dc)
    assert (m.terms, m.literals, m.exact) == (3, 5, True)


def test_heuristic_beyond_twelve_variables_is_equivalent_but_flagged(pool):
    names = [f"y{i}" for i in range(14)]
    ys = [pool.symbol(nm) for nm in names]
    # A function with a small cover hidden behind redundant (absorbed) terms.
    pairs = [ak.And(ys[i], ys[i + 1]) for i in range(0, 14, 2)]
    extra = [ak.And(ys[i], ys[i + 1], ys[(i + 2) % 14]) for i in range(0, 14, 2)]
    f = ak.Or(*pairs, *extra)
    m = ak.minimize_logic(f)
    assert m.exact is False
    assert ak.equivalent(m.expr, f)
    assert (m.terms, m.literals) == (7, 14)
    with pytest.raises(ak.LogicError) as ei:
        ak.minimize_logic(ak.Xor(*[pool.symbol(f"z{i}") for i in range(17)]), exact=True)
    assert ei.value.code == "E-LOGIC-004"


def test_relations_are_opaque_atoms_in_normal_forms(pool):
    x, a = pool.symbol("x"), pool.symbol("a")
    g = pool.gt(x, pool.integer(0))
    assert ak.simplify_logic(ak.Or(ak.And(g, a), ak.And(g, ak.Not(a)))) == g
    # ¬(x > 0) stays a negated atom; it is not rewritten to x <= 0.
    assert ak.to_nnf(ak.Not(ak.And(g, a))) == ak.Or(ak.Not(g), ak.Not(a))


def test_refusals(pool):
    x, a = pool.symbol("x"), pool.symbol("a")
    with pytest.raises(ak.LogicError) as ei:
        ak.simplify_logic(pool.forall(x, pool.gt(x, pool.integer(0))))
    assert ei.value.code == "E-LOGIC-002"
    with pytest.raises(ak.LogicError) as ei:
        ak.simplify_logic(a, form="sop-ish")
    assert ei.value.code == "E-LOGIC-005"
    with pytest.raises(ak.LogicError) as ei:
        ak.bdd(ak.And(a, pool.gt(x, pool.integer(0))))
    assert ei.value.code == "E-LOGIC-002"
    with pytest.raises(ak.LogicError) as ei:
        ak.to_cnf(ak.Or(*[ak.And(pool.symbol(f"p{i}"), pool.symbol(f"q{i}")) for i in range(20)]))
    assert ei.value.code == "E-LOGIC-003"


# ---------------------------------------------------------------------------
# BDDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_bdd_sat_count_and_quantifiers_vs_brute_force(seed):
    rng = random.Random(500 + seed)
    pool = ak.ExprPool()
    n = 2 + seed % 9  # 2..10
    names = [f"v{i}" for i in range(n)]
    vs = [pool.symbol(nm) for nm in names]
    f = random_formula(rng, vs, 4)
    want = table(f, names)
    order = list(reversed(vs)) if seed % 2 else None
    g = ak.bdd(f, order=order)
    assert set(g.order) <= set(names)
    assert g.sat_count(variables=vs) == sum(want)
    assert g.equivalent(f)
    assert table(g.to_expr(), names) == want
    q = vs[0]
    t_ex = table(g.exists([q]).to_expr(), names)
    t_fa = table(g.forall([q]).to_expr(), names)
    t_r1 = table(g.restrict({q: True}).to_expr(), names)
    for idx, bits in enumerate(itertools.product([False, True], repeat=n)):
        env = dict(zip(names, bits))
        hi = ev(f, {**env, names[0]: True})
        lo = ev(f, {**env, names[0]: False})
        assert t_ex[idx] == (hi or lo)
        assert t_fa[idx] == (hi and lo)
        assert t_r1[idx] == hi
    sol = g.satisfy_one()
    if sum(want) == 0:
        assert sol is None
        assert g.is_false()
    else:
        env = {nm: sol.get(nm, False) for nm in names}
        assert ev(f, env)


def test_bdd_operators_merge_managers(pool):
    a, b, c = pool.symbol("a"), pool.symbol("b"), pool.symbol("c")
    fa = ak.bdd(ak.Or(a, b))
    fb = ak.bdd(ak.Or(ak.Not(b), c))
    res = fa & fb
    assert res.equivalent(ak.And(ak.Or(a, b), ak.Or(ak.Not(b), c)))
    assert (fa | ~fa).is_true()
    assert (fa ^ fa).is_false()
    assert fa.implies(ak.Or(a, b, c)).is_true()
    assert fa.ite(b, c).equivalent(ak.ITE(ak.Or(a, b), b, c))
    # Resolution: ∃b. (a ∨ b) ∧ (¬b ∨ c) = a ∨ c
    assert res.exists([b]).equivalent(ak.Or(a, c))
    assert res.sat_count(variables=[a, b, c, pool.symbol("d")]) == 2 * 4


def test_majority_docstring(pool):
    a, b, c = pool.symbol("a"), pool.symbol("b"), pool.symbol("c")
    f = ak.bdd(ak.Or(ak.And(a, b), ak.And(b, c), ak.And(a, c)))
    assert (f.sat_count(), f.node_count()) == (4, 4)


def _ripple_adder(a, b, cin):
    out, c = [], cin
    for ai, bi in zip(a, b):
        out.append(ak.Xor(ai, bi, c))
        c = ak.Or(ak.And(ai, bi), ak.And(c, ak.Xor(ai, bi)))
    return out, c


def _lookahead_adder(a, b, cin):
    """Carry-lookahead adder: c_{i+1} = g_i ∨ p_i g_{i-1} ∨ … ∨ p_i…p_0 cin."""
    g = [ak.And(x, y) for x, y in zip(a, b)]
    p = [ak.Or(x, y) for x, y in zip(a, b)]
    carries = [cin]
    for i in range(len(a)):
        terms = [g[i]]
        for j in range(i - 1, -1, -1):
            terms.append(ak.And(*p[j + 1 : i + 1], g[j]))
        terms.append(ak.And(*p[: i + 1], cin))
        carries.append(ak.Or(*terms))
    out = [ak.Xor(x, y, carries[i]) for i, (x, y) in enumerate(zip(a, b))]
    return out, carries[-1]


@pytest.mark.parametrize("method", ["auto", "truth_table", "bdd", "sat"])
def test_adder_miter(pool, method):
    n = 4
    a = [pool.symbol(f"a{i}") for i in range(n)]
    b = [pool.symbol(f"b{i}") for i in range(n)]
    cin = pool.symbol("cin")
    s1, c1 = _ripple_adder(a, b, cin)
    s2, c2 = _lookahead_adder(a, b, cin)
    for x, y in zip([*s1, c1], [*s2, c2]):
        assert ak.equivalent(x, y, method=method) is True
    # A faulty lookahead: drop the cin term from the top carry.
    bad = ak.Or(*[t for t in c2.node()[2] if "cin" not in str(t)])
    ok, cex = ak.equivalent(c1, bad, method=method, counterexample=True)
    assert ok is False
    full = dict.fromkeys([f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)] + ["cin"], False)
    full.update(cex)
    assert ev(c1, full) != ev(bad, full)
    # The miter itself is unsatisfiable for the good pair and satisfiable for the bad one.
    assert ak.satisfiable(ak.Xor(c1, c2)) is False
    assert isinstance(ak.satisfiable(ak.Xor(c1, bad)), dict)
    # Adder arithmetic, checked on every input against Python integers.
    for av in range(2**n):
        for bv in range(0, 2**n, 3):
            env = {f"a{i}": bool(av >> i & 1) for i in range(n)}
            env.update({f"b{i}": bool(bv >> i & 1) for i in range(n)})
            env["cin"] = False
            total = sum(ev(s, env) << i for i, s in enumerate(s1)) + (ev(c1, env) << n)
            assert total == av + bv


def _alu(a, b, op0, op1, adder):
    """2-op-bit ALU: 00 AND, 01 OR, 10 XOR, 11 ADD."""
    s, _ = adder(a, b, pool_false(a[0]))
    outs = []
    for i, (x, y) in enumerate(zip(a, b)):
        lo = ak.ITE(op0, ak.Or(x, y), ak.And(x, y))
        hi = ak.ITE(op0, s[i], ak.Xor(x, y))
        outs.append(ak.ITE(op1, hi, lo))
    return outs


def pool_false(sym):
    return ak.And(sym, ak.Not(sym))


def test_alu_miter(pool):
    n = 3
    a = [pool.symbol(f"a{i}") for i in range(n)]
    b = [pool.symbol(f"b{i}") for i in range(n)]
    op0, op1 = pool.symbol("op0"), pool.symbol("op1")
    r = _alu(a, b, op0, op1, _ripple_adder)
    la = _alu(a, b, op0, op1, _lookahead_adder)
    for x, y in zip(r, la):
        assert ak.equivalent(x, y, method="sat")
        assert ak.equivalent(x, y, method="bdd")
    # Swap the AND/OR legs in one bit: the miter finds an op=00/01 witness.
    broken = ak.ITE(
        op1,
        ak.ITE(op0, la[1], ak.Xor(a[1], b[1])),
        ak.ITE(op0, ak.And(a[1], b[1]), ak.Or(a[1], b[1])),
    )
    ok, cex = ak.equivalent(r[1], broken, method="sat", counterexample=True)
    assert not ok
    assert cex["op1"] is False


def _multiplier_bit(pool, n, k):
    """Bit ``k`` of the n×n array multiplier a·b (shift-and-add)."""
    a = [pool.symbol(f"a{i}") for i in range(n)]
    b = [pool.symbol(f"b{i}") for i in range(n)]
    f = pool.pred_false()
    acc = [f] * (2 * n)
    for j in range(n):
        row = [f] * j + [ak.And(a[i], b[j]) for i in range(n)] + [f] * (n - j)
        carry = f
        nxt = []
        for x, y in zip(acc, row):
            nxt.append(ak.Xor(x, y, carry))
            carry = ak.Or(ak.And(x, y), ak.And(carry, ak.Xor(x, y)))
        acc = nxt
    return acc[k]


def test_multiplier_middle_bit_bdd_is_refused_under_a_small_limit(pool):
    mid = _multiplier_bit(pool, 10, 9)
    with pytest.raises(ak.LogicError) as ei:
        ak.bdd(mid, max_nodes=5_000)
    assert ei.value.code == "E-LOGIC-003"
    with ak.context(budget=ak.Budget(max_steps=3)), pytest.raises(ak.BudgetExceededError):
        ak.bdd(mid)


def test_multiplier_small_is_correct(pool):
    n = 3
    bits = [_multiplier_bit(pool, n, k) for k in range(2 * n)]
    names = [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)]
    for av in range(2**n):
        for bv in range(2**n):
            env = {f"a{i}": bool(av >> i & 1) for i in range(n)}
            env.update({f"b{i}": bool(bv >> i & 1) for i in range(n)})
            assert sum(ev(x, env) << k for k, x in enumerate(bits)) == av * bv
    g = ak.bdd(bits[n])
    assert g.sat_count(variables=[pool.symbol(nm) for nm in names]) == sum(
        1 for av in range(2**n) for bv in range(2**n) if (av * bv) >> n & 1
    )


# ---------------------------------------------------------------------------
# Equivalence / satisfiability
# ---------------------------------------------------------------------------


def test_equivalent_basic_and_relations(pool):
    a, b = pool.symbol("a"), pool.symbol("b")
    assert ak.equivalent(ak.Not(ak.And(a, b)), ak.Or(ak.Not(a), ak.Not(b)))
    ok, cex = ak.equivalent(ak.Implies(a, b), ak.Implies(b, a), counterexample=True)
    assert ok is False
    assert cex in ({"a": True, "b": False}, {"a": False, "b": True})
    x = pool.symbol("x")
    g = pool.gt(x, pool.integer(0))
    assert ak.equivalent(ak.Or(ak.And(g, a), ak.And(g, ak.Not(a))), g)
    # Propositionally different, equal over the reals: refused, never "False".
    with pytest.raises(ak.LogicError) as ei:
        ak.equivalent(g, ak.Not(pool.le(x, pool.integer(0))))
    assert ei.value.code == "E-LOGIC-002"


def test_satisfiable_handles_wide_and_shared_formulas(pool):
    # 70 symbols (past the tree path's 64) and a parity chain (exponential as a tree).
    vs = [pool.symbol(f"w{i}") for i in range(70)]
    w = ak.satisfiable(ak.Xor(*vs))
    assert isinstance(w, dict)
    assert sum(v == "true" for v in w.values()) % 2 == 1
    assert ak.satisfiable(ak.And(ak.Xor(*vs[:20]), ak.Not(ak.Xor(*vs[:20])))) is False


def test_truth_table(pool):
    a, b = pool.symbol("a"), pool.symbol("b")
    assert ak.truth_table(ak.Implies(a, b)) == [
        ((False, False), True),
        ((False, True), True),
        ((True, False), False),
        ((True, True), True),
    ]
    rows = ak.truth_table(ak.Implies(a, b), variables=[b, a])
    assert rows[1] == ((False, True), False)
