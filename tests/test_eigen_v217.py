"""V2-17 — Matrix eigenvals / eigenvects / diagonalize and SymPy cross-checks."""

from __future__ import annotations

from collections import Counter

import pytest

import alkahest

sympy = pytest.importorskip("sympy")


def _expr_to_sympy(e):
    n = e.node()
    tag = n[0]
    if tag == "symbol":
        return sympy.Symbol(n[1])
    if tag == "integer":
        return sympy.Integer(int(n[1]))
    if tag == "rational":
        return sympy.Rational(int(n[1]), int(n[2]))
    if tag == "add":
        return sympy.Add(*[_expr_to_sympy(c) for c in n[1]])
    if tag == "mul":
        return sympy.Mul(*[_expr_to_sympy(c) for c in n[1]])
    if tag == "pow":
        return sympy.Pow(_expr_to_sympy(n[1]), _expr_to_sympy(n[2]))
    if tag == "func":
        name, args = n[1], n[2]
        av = [_expr_to_sympy(a) for a in args]
        return getattr(sympy, name)(*av)
    raise AssertionError(f"unsupported expr tag {tag!r}")


def _counter_eigenvals(m: alkahest.Matrix) -> Counter:
    d = m.eigenvals()
    out = Counter()
    for ev, mult in d.items():
        out[_expr_to_sympy(ev)] += mult
    return out


def _counter_sympy(sm: sympy.Matrix) -> Counter:
    return Counter(sympy.Matrix(sm).eigenvals())


def _matrix_from_sympy(sm: sympy.Matrix, pool):
    rows = []
    for i in range(sm.rows):
        row = []
        for j in range(sm.cols):
            ex = sympy.simplify(sympy.sympify(sm[i, j]))
            assert ex.is_rational, (i, j, ex)
            if ex.is_Integer:
                row.append(pool.integer(int(ex)))
            else:
                frac = sympy.fraction(ex)
                row.append(pool.rational(int(frac[0]), int(frac[1])))
        rows.append(row)
    return alkahest.Matrix(rows)


def test_jordan_block_eigenvals_and_diagonalize_error():
    p = alkahest.ExprPool()
    two = p.integer(2)
    one = p.integer(1)
    z = p.integer(0)
    m = alkahest.Matrix([[two, one], [z, two]])
    d = m.eigenvals()
    assert len(d) == 1
    ev, mult = next(iter(d.items()))
    assert _expr_to_sympy(ev) == 2 and mult == 2

    triples = m.eigenvects()
    assert len(triples) == 1 and triples[0][1] == 2 and len(triples[0][2]) == 1

    with pytest.raises(alkahest.EigenError):
        m.diagonalize()


def test_similar_integer_diagonal_random3x3():
    pool = alkahest.ExprPool()
    rng = pytest.importorskip("random").Random(17)
    for trial in range(50):
        for _attempt in range(400):
            ent = [rng.randint(-3, 3) for _ in range(9)]
            pm = sympy.Matrix(3, 3, ent)
            if pm.det() == 0:
                continue
            break
        else:
            pytest.fail("no invertible random P")

        vals = sorted(rng.sample(range(-4, 5), k=3))
        d_diag = sympy.diag(*[sympy.Integer(v) for v in vals])
        sm = sympy.simplify(pm.inv() * d_diag * pm)

        alk = _matrix_from_sympy(sm, pool)

        ck_alk = _counter_eigenvals(alk)
        ck_sp = _counter_sympy(sm)
        assert ck_alk == ck_sp, f"trial={trial}"

        triples = alk.eigenvects()
        assert all(len(vs) == mult for (_, mult, vs) in triples)
        assert sum(len(vs) for (_, _, vs) in triples) == 3


def test_rotation_diagonalizes():
    pool = alkahest.ExprPool()
    z = pool.integer(0)
    one = pool.integer(1)
    neg_one = pool.integer(-1)
    m = alkahest.Matrix([[z, neg_one], [one, z]])
    assert len(m.eigenvals()) == 2
    p_mat, dd = m.diagonalize()
    mp = (m @ p_mat).simplify()
    pd = (p_mat @ dd).simplify()
    lists_m = mp.to_list()
    lists_p = pd.to_list()
    for r in range(2):
        for c in range(2):
            assert sympy.simplify(_expr_to_sympy(lists_m[r][c]) - _expr_to_sympy(lists_p[r][c])) == 0
