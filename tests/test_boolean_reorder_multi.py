"""F22 — logic follow-ups: dynamic BDD variable reordering and multi-output
two-level minimisation.

Oracles are independent of alkahest's own truth tables: a brute-force Python
evaluator over ``Expr.node()``, SymPy's two-level minimisers for the cost
bound, and per-output ``simplify_logic`` as the baseline a shared cover must
never exceed.
"""

from __future__ import annotations

import itertools
import random

import alkahest as ak
import pytest


def ev(expr, env, memo=None):
    if memo is None:
        memo = {}
    if expr in memo:
        return memo[expr]
    node = expr.node()
    if node[0] == "symbol":
        val = env[node[1]]
    else:
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
        else:  # pragma: no cover
            raise AssertionError(kind)
    memo[expr] = val
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


def inner_product(pool, n):
    """x0·y0 + ... + x(n-1)·y(n-1) and its worst variable order."""
    xs = [pool.symbol(f"x{i}") for i in range(n)]
    ys = [pool.symbol(f"y{i}") for i in range(n)]
    f = ak.Or(*[ak.And(x, y) for x, y in zip(xs, ys)])
    return f, xs + ys


def multiplier_bits(pool, n):
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
    return acc, a, b


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# 1. Dynamic variable reordering (Rudell sifting)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8, 10])
def test_sifting_repairs_the_inner_product_order(pool, n):
    f, bad = inner_product(pool, n)
    g = ak.bdd(f, order=bad, auto_reorder=False)
    assert g.order == [str(v) for v in bad]
    before = g.node_count()
    assert before >= 2 ** (n // 2 + 1)  # exponential in the bad order
    count = g.sat_count()
    st = g.reorder()
    assert st["complete"]
    assert st["nodes_after"] == g.node_count()
    assert g.node_count() == 2 * n  # the interleaved optimum
    assert g.node_count() * 3 < before
    # Meaning unchanged.
    assert g.sat_count() == count
    assert g.equivalent(f)
    names = [str(v) for v in bad]
    assert ak.equivalent(g.to_expr(), f)
    if n <= 8:
        assert table(g.to_expr(), names) == table(f, names)
    assert sorted(g.order) == sorted(names)


def test_inner_product_count_is_exact(pool):
    n = 7
    f, bad = inner_product(pool, n)
    g = ak.bdd(f, order=bad, auto_reorder=False)
    g.reorder()
    # P(no pair both true) = (3/4)^n.
    assert g.sat_count() == 4**n - 3**n


def test_multiplier_bits_shrink_and_stay_correct(pool):
    n = 3
    bits, a, b = multiplier_bits(pool, n)
    order = a + b  # operands apart: the unfavourable layout
    gs = [ak.bdd(x, order=order, auto_reorder=False) for x in bits]
    names = [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)]
    want = [table(x, names) for x in bits]
    counts = [g.sat_count() for g in gs]
    # Bring every product bit into the first one's manager (x ^ (x ^ g) == g).
    ms = [gs[0]] + [gs[0] ^ (gs[0] ^ g) for g in gs[1:]]
    before = ms[0].node_count()
    st = ms[0].reorder()
    assert st["nodes_after"] <= st["nodes_before"]
    assert ms[0].node_count() <= before
    for m, x, w, c in zip(ms, bits, want, counts):
        assert m.equivalent(x)
        assert m.sat_count() == c
        assert table(m.to_expr(), names) == w

    # Reordering the middle product bit alone (a bigger multiplier) helps.
    n = 4
    bits, a, b = multiplier_bits(pool, n)
    mid = ak.bdd(bits[n], order=a + b, auto_reorder=False)
    before = mid.node_count()
    mid.reorder()
    assert mid.node_count() < before
    assert mid.equivalent(bits[n])


@pytest.mark.parametrize("seed", range(25))
def test_reorder_preserves_random_functions(seed):
    rng = random.Random(7000 + seed)
    pool = ak.ExprPool()
    n = 3 + seed % 7
    names = [f"v{i}" for i in range(n)]
    vs = [pool.symbol(nm) for nm in names]
    f = random_formula(rng, vs, 4)
    h = random_formula(rng, vs, 4)
    want_f, want_h = table(f, names), table(h, names)
    g = ak.bdd(f, order=vs, auto_reorder=False)
    k = g | h  # h and f share one manager; both stay referenced
    scratch = g ^ h  # garbage once dropped
    del scratch
    st = g.reorder(max_growth=1.0 + (seed % 3) * 0.2)
    assert st["nodes_after"] <= st["nodes_before"]
    assert sorted(g.order) == sorted(set(g.order))
    assert set(g.order) >= set(g.support())
    assert table(g.to_expr(), names) == want_f
    assert g.sat_count(variables=vs) == sum(want_f)
    assert table(k.to_expr(), names) == [a or b for a, b in zip(want_f, want_h)]
    assert g.equivalent(f)
    assert k.equivalent(ak.Or(f, h))
    # Quantification/restriction after the order changed.
    q = vs[seed % n]
    ex = table(g.exists([q]).to_expr(), names)
    fa = table(g.forall([q]).to_expr(), names)
    pos = names.index(str(q))
    for idx in range(2**n):
        hi = want_f[idx | (1 << (n - 1 - pos))]
        lo = want_f[idx & ~(1 << (n - 1 - pos))]
        assert ex[idx] == (hi or lo)
        assert fa[idx] == (hi and lo)
    sol = g.satisfy_one()
    if sum(want_f) == 0:
        assert sol is None
    else:
        assert ev(f, {nm: sol.get(nm, False) for nm in names})
    # A second pass never grows the diagram.
    again = g.reorder()
    assert again["nodes_after"] <= again["nodes_before"]


def test_automatic_reordering_threshold_and_off_switch(pool):
    f, bad = inner_product(pool, 10)
    off = ak.bdd(f, order=bad, auto_reorder=False)
    assert off.auto_reorder is None
    assert off.order == [str(v) for v in bad]  # nothing moved it
    on = ak.bdd(f, order=bad, auto_reorder=200)
    assert on.auto_reorder == 200
    assert on.order != [str(v) for v in bad]
    assert on.node_count() < off.node_count() // 4
    assert on.sat_count() == off.sat_count() == 4**10 - 3**10
    assert on.equivalent(f)
    assert ak.equivalent(on.to_expr(), f)
    # An explicit order disables the default trigger; no order enables it.
    assert ak.bdd(f, order=bad).auto_reorder is None
    assert ak.bdd(f).auto_reorder == 50_000
    assert ak.bdd(f, auto_reorder=True).auto_reorder == 50_000
    on.set_auto_reorder(False)
    assert on.auto_reorder is None
    on.set_auto_reorder(1234)
    assert on.auto_reorder == 1234


def test_automatic_reordering_during_operations(pool):
    n = 8
    xs = [pool.symbol(f"x{i}") for i in range(n)]
    ys = [pool.symbol(f"y{i}") for i in range(n)]
    g = ak.bdd(ak.And(xs[0], ys[0]), order=xs + ys, auto_reorder=60)
    want = ak.And(xs[0], ys[0])
    for x, y in zip(xs[1:], ys[1:]):
        g = g | ak.And(x, y)  # grows past the threshold midway
        want = ak.Or(want, ak.And(x, y))
        assert g.equivalent(want)
    assert g.node_count() <= 4 * n
    assert g.sat_count() == 4**n - 3**n


def test_reorder_refusals_and_bounds(pool):
    f, bad = inner_product(pool, 8)
    g = ak.bdd(f, order=bad, auto_reorder=False)
    with pytest.raises(ak.LogicError) as ei:
        g.reorder(max_growth=0.5)
    assert ei.value.code == "E-LOGIC-005"
    # A swap limit stops early, reports it, and leaves a valid diagram.
    st = g.reorder(max_swaps=5)
    assert not st["complete"]
    assert st["swaps"] <= 5 + 2 * 16
    assert g.equivalent(f)
    # The Budget is honoured between swaps; the order reached stays valid.
    h = ak.bdd(f, order=bad, auto_reorder=False)
    with ak.context(budget=ak.Budget(max_steps=3)), pytest.raises(ak.BudgetExceededError):
        h.reorder()
    assert h.equivalent(f)
    assert h.sat_count() == 4**8 - 3**8
    for bad_arg in (0, "yes", -1):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            ak.bdd(f, auto_reorder=bad_arg)


# ---------------------------------------------------------------------------
# 2. Multi-output minimisation
# ---------------------------------------------------------------------------


def per_output_total(outs, **kw):
    return sum(ak.minimize_logic(f, **kw).terms for f in outs)


def check_outputs(m, outs, names, dc=None):
    assert len(m.exprs) == len(outs)
    for got, f in zip(m.exprs, outs):
        if dc is None:
            assert table(got, names) == table(f, names)
            assert ak.bdd(got).equivalent(f)
            assert ak.equivalent(got, f)
        else:
            tg, tf, td = table(got, names), table(f, names), table(dc, names)
            assert all(d or a == b for a, b, d in zip(tg, tf, td))


def adder(pool, width):
    a = [pool.symbol(f"a{i}") for i in range(width)]
    b = [pool.symbol(f"b{i}") for i in range(width)]
    carry = pool.pred_false()
    outs = []
    for x, y in zip(a, b):
        outs.append(ak.Xor(x, y, carry))
        carry = ak.Or(ak.And(x, y), ak.And(carry, ak.Xor(x, y)))
    outs.append(carry)
    return outs, [f"a{i}" for i in range(width)] + [f"b{i}" for i in range(width)]


def test_shared_terms_across_outputs(pool):
    a, b = pool.symbol("a"), pool.symbol("b")
    outs = [ak.Xor(a, b), ak.And(a, b), ak.Or(a, b)]
    m = ak.minimize_logic(outs)
    assert isinstance(m, ak.MultiMinimizedLogic)
    assert m.exact
    assert m.form == "dnf"
    assert len(m) == 3
    check_outputs(m, outs, ["a", "b"])
    # Separately 2 + 1 + 2 terms; jointly a·¬b, ¬a·b, a·b serve all three.
    assert per_output_total(outs) == 5
    assert m.terms == 3
    assert m.shared_terms == 3
    assert m.per_output_terms == [2, 1, 3]
    assert m.literals == 6
    assert "MultiMinimizedLogic(" in repr(m)


def test_two_bit_adder_is_correct_and_no_worse(pool):
    outs, names = adder(pool, 2)
    m = ak.minimize_logic(outs)
    assert m.exact
    check_outputs(m, outs, names)
    assert m.terms <= per_output_total(outs)


def test_multi_output_cost_never_exceeds_sympy_per_output(pool):
    sympy = pytest.importorskip("sympy")
    outs, names = adder(pool, 2)
    m = ak.minimize_logic(outs)
    syms = {nm: sympy.Symbol(nm) for nm in names}
    total = 0
    for f in outs:
        minterms = [i for i, v in enumerate(table(f, names)) if v]
        e = sympy.SOPform([syms[nm] for nm in names], minterms)
        total += 0 if e in (sympy.true, sympy.false) else len(sympy.Or.make_args(e))
    assert m.terms <= total


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("form", ["dnf", "cnf"])
def test_multi_vs_per_output_random(seed, form):
    rng = random.Random(900 + seed)
    pool = ak.ExprPool()
    n = 3 + seed % 5
    names = [f"v{i}" for i in range(n)]
    vs = [pool.symbol(nm) for nm in names]
    k = 2 + seed % 3
    outs = [random_formula(rng, vs, 4) for _ in range(k)]
    m = ak.minimize_logic(outs, form=form)
    assert m.exact
    assert m.form == form
    check_outputs(m, outs, names)
    assert m.terms <= per_output_total(outs, form=form)
    # Heuristic sharing is also never worse than the heuristic per-output pass.
    h = ak.minimize_logic(outs, form=form, exact=False)
    assert not h.exact
    check_outputs(h, outs, names)
    assert h.terms <= per_output_total(outs, form=form, exact=False)
    assert m.terms <= h.terms  # exact is a lower bound


def test_seven_segment_decoder_with_dontcares(pool):
    bits = [pool.symbol(f"d{i}") for i in range(4)]
    names = [f"d{i}" for i in range(4)]
    segs = {
        "a": [0, 2, 3, 5, 6, 7, 8, 9],
        "b": [0, 1, 2, 3, 4, 7, 8, 9],
        "c": [0, 1, 3, 4, 5, 6, 7, 8, 9],
        "d": [0, 2, 3, 5, 6, 8, 9],
        "e": [0, 2, 6, 8],
        "f": [0, 4, 5, 6, 8, 9],
        "g": [2, 3, 4, 5, 6, 8, 9],
    }

    def minterm(v):
        return ak.And(*[bits[i] if (v >> (3 - i)) & 1 else ak.Not(bits[i]) for i in range(4)])

    outs = [ak.Or(*[minterm(v) for v in on]) for on in segs.values()]
    dc = ak.Or(*[minterm(v) for v in range(10, 16)])
    m = ak.minimize_logic(outs, dontcare=dc)
    assert m.exact
    assert len(m) == 7
    check_outputs(m, outs, names, dc=dc)
    separate = per_output_total(outs, dontcare=dc)
    assert m.terms <= separate
    assert m.shared_terms >= 1
    assert m.terms < separate


def test_heuristic_beyond_twelve_variables_is_flagged_and_equivalent(pool):
    rng = random.Random(5)
    n = 14
    names = [f"w{i}" for i in range(n)]
    vs = [pool.symbol(nm) for nm in names]
    common = ak.Or(
        ak.And(vs[0], vs[1], vs[2]),
        ak.And(vs[3], ak.Not(vs[4])),
        ak.And(vs[11], vs[12], ak.Not(vs[13])),
    )
    outs = [ak.Or(common, ak.And(vs[5 + i], vs[6 + i], vs[10 - i])) for i in range(3)] + [
        random_formula(rng, vs[:6], 3)
    ]
    m = ak.minimize_logic(outs)
    assert not m.exact
    for got, f in zip(m.exprs, outs):
        assert ak.equivalent(got, f)
    assert m.terms <= per_output_total(outs)
    assert m.shared_terms >= 1


def test_auto_form_and_single_element_list(pool):
    outs, names = adder(pool, 2)
    a = ak.minimize_logic(outs, form="auto")
    d = ak.minimize_logic(outs, form="dnf")
    c = ak.minimize_logic(outs, form="cnf")
    assert (a.literals, a.terms) == min((d.literals, d.terms), (c.literals, c.terms))
    check_outputs(a, outs, names)
    one = ak.minimize_logic([outs[2]])
    single = ak.minimize_logic(outs[2])
    assert (one.terms, one.literals) == (single.terms, single.literals)
    assert table(one.exprs[0], names) == table(outs[2], names)
    # Constants and duplicated outputs.
    t, f = pool.pred_true(), pool.pred_false()
    x = pool.symbol("x")
    m = ak.minimize_logic([t, f, x, x])
    assert m.per_output_terms[:2] == [0, 0]
    assert m.terms == 1
    assert m.shared_terms == 1
    assert ak.equivalent(m.exprs[0], t)
    assert ak.equivalent(m.exprs[1], f)


def test_multi_output_refusals(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.LogicError) as ei:
        ak.minimize_logic([])
    assert ei.value.code == "E-LOGIC-005"
    with pytest.raises(ak.LogicError) as ei:
        ak.minimize_logic([x] * 65)
    assert ei.value.code == "E-LOGIC-003"
    vs = [pool.symbol(f"z{i}") for i in range(17)]
    with pytest.raises(ak.LogicError) as ei:
        ak.minimize_logic([ak.And(*vs), ak.Or(vs[0], vs[1])], exact=True)
    assert ei.value.code == "E-LOGIC-004"
    with pytest.raises(TypeError):
        ak.minimize_logic([x, 3])
    with pytest.raises(TypeError):
        ak.minimize_logic(5)
    other = ak.ExprPool().symbol("x")
    with pytest.raises(Exception, match="E-POOL-001"):
        ak.minimize_logic([x, other])
    with pytest.raises(ak.LogicError) as ei:
        ak.minimize_logic([x, x], form="bogus")
    assert ei.value.code == "E-LOGIC-005"
    # The Budget is honoured.
    outs, _ = adder(pool, 3)
    with ak.context(budget=ak.Budget(max_steps=3)), pytest.raises(ak.BudgetExceededError):
        ak.minimize_logic(outs)
