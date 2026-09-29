"""Python-boundary correctness and cost: pool identity, n-ary parsing, lazy jax.

* ``Expr`` equality and hashing are by (pool, id), not id alone.
* ``pool.add`` / ``pool.mul`` / ``pool.func`` refuse an expression from another
  pool (``E-POOL-001``) instead of reading its id as a local one.
* The parser interns a run of ``+``/``-`` (or ``*``) with one constructor call
  and produces exactly the node the left fold did.
* ``import alkahest`` does not import jax; ``alkahest.to_jax`` still resolves.
* Research dependency lookup matches partial sums without interning them.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap
import time

import alkahest as ak
import pytest
from alkahest import research as R

# ---------------------------------------------------------------------------
# Pool identity in __eq__ / __hash__
# ---------------------------------------------------------------------------


def test_same_index_in_two_pools_is_not_equal():
    p, q = ak.ExprPool(), ak.ExprPool()
    # First symbol interned in each pool: same ExprId, different expressions.
    a, b = p.symbol("x"), q.symbol("zzz")
    assert (a == b) is False
    assert a != b
    assert len({a, b}) == 2
    d = {a: "p", b: "q"}
    assert d[a] == "p"
    assert d[b] == "q"


def test_same_structure_in_two_pools_is_not_equal():
    p, q = ak.ExprPool(), ak.ExprPool()
    assert p.symbol("x") != q.symbol("x")


def test_equality_within_a_pool_unchanged():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    assert p.symbol("x") == x
    assert hash(p.symbol("x")) == hash(x)
    assert x + y == y + x
    assert hash(x + y) == hash(y + x)
    assert x != y
    assert (x == 3) is False


# ---------------------------------------------------------------------------
# Cross-pool construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctor", ["add", "mul"])
def test_bulk_constructor_rejects_foreign_expr(ctor):
    p, q = ak.ExprPool(), ak.ExprPool()
    x = p.symbol("x")
    foreign = q.symbol("y")
    with pytest.raises(ak.PoolError) as info:
        getattr(p, ctor)([x, foreign])
    assert info.value.code == "E-POOL-001"
    assert "argument 1" in str(info.value)
    assert info.value.remediation


def test_func_rejects_foreign_expr():
    p, q = ak.ExprPool(), ak.ExprPool()
    with pytest.raises(ak.PoolError) as info:
        p.func("f", [q.symbol("y")])
    assert info.value.code == "E-POOL-001"


def test_bulk_constructors_accept_own_pool():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    assert p.add([x, y]) == x + y
    assert p.mul([x, y]) == x * y
    assert str(p.func("f", [x])) == "f(x)"


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: a**b,
    ],
)
def test_binary_operators_reject_foreign_expr(op):
    p, q = ak.ExprPool(), ak.ExprPool()
    with pytest.raises(ak.PoolError):
        op(p.symbol("x"), q.symbol("y"))


# ---------------------------------------------------------------------------
# Parser: n-ary runs build exactly the left fold's node
# ---------------------------------------------------------------------------


def _random_ast(rng: random.Random, depth: int):
    """A random expression as ``(source, build)``; ``build(env)`` folds with
    the operators, left to right — the parser's old construction."""
    if depth == 0 or rng.random() < 0.3:
        choice = rng.random()
        if choice < 0.5:
            name = rng.choice(["x", "y", "z", "A", "B"])
            return name, lambda env, n=name: env[n]
        value = rng.randint(0, 9)
        return str(value), lambda env, v=value: env["pool"].integer(v)
    kind = rng.random()
    if kind < 0.1:
        base_src, base = _random_ast(rng, depth - 1)
        exp = rng.randint(2, 4)
        return (
            f"({base_src})^{exp}",
            lambda env, b=base, e=exp: b(env).pow_expr(env["pool"].integer(e)),
        )
    if kind < 0.15:
        inner_src, inner = _random_ast(rng, depth - 1)
        return f"sin({inner_src})", lambda env, i=inner: ak.sin(i(env))
    ops = ["+", "-"] if kind < 0.6 else ["*", "/"]
    width = rng.randint(2, 7)
    parts = [_random_ast(rng, depth - 1) for _ in range(width)]
    chosen = [rng.choice(ops) for _ in range(width - 1)]
    src = f"({parts[0][0]})"
    for o, (s, _b) in zip(chosen, parts[1:]):
        src += f" {o} ({s})"

    def build(env, parts=parts, chosen=chosen):
        acc = parts[0][1](env)
        for o, (_s, b) in zip(chosen, parts[1:]):
            rhs = b(env)
            if o == "+":
                acc = acc + rhs
            elif o == "-":
                acc = acc - rhs
            elif o == "*":
                acc = acc * rhs
            else:
                if str(rhs) == "0":
                    raise ZeroDivisionError
                acc = acc / rhs
        return acc

    return src, build


@pytest.mark.parametrize("parse_first", [False, True])
def test_parse_runs_match_left_fold(parse_first):
    # Both orders: the canonical sort is by interned id, so whichever side
    # runs second reuses the first side's children; either way the two must
    # land on the same node.
    rng = random.Random(20260928)
    checked = 0
    for _ in range(400):
        src, build = _random_ast(rng, 3)
        pool = ak.ExprPool()
        env = {
            "pool": pool,
            "x": pool.symbol("x"),
            "y": pool.symbol("y"),
            "z": pool.symbol("z"),
            # Non-commutative symbols: `mul` must keep factor order.
            "A": pool.symbol("A", commutative=False),
            "B": pool.symbol("B", commutative=False),
        }
        symbols = {k: v for k, v in env.items() if k != "pool"}
        try:
            if parse_first:
                got = ak.parse(src, pool, symbols)
                expected = build(env)
            else:
                expected = build(env)
                got = ak.parse(src, pool, symbols)
        except ZeroDivisionError:
            with pytest.raises(ZeroDivisionError):
                build(env)
            with pytest.raises(ZeroDivisionError):
                ak.parse(src, pool, symbols)
            continue
        assert got == expected, src
        checked += 1
    assert checked > 300


@pytest.mark.parametrize(
    "src",
    [
        "a + b - c + d",
        "a - b - c",
        "-a + b",
        "a * b * c / d * e",
        "a / b / c",
        "2*a*b + 3*c - a*b*c",
        "(a + b) + (c + d)",
        "a + (b - c) * d - e",
    ],
)
def test_parse_runs_match_left_fold_examples(src):
    pool = ak.ExprPool()
    syms = {n: pool.symbol(n) for n in "abcde"}
    got = ak.parse(src, pool, dict(syms))
    # The same text through Python's own left-associative operators — the
    # construction the parser used to perform.  (`^` is not in these strings:
    # Python's `**` coerces integer exponents the way `pow_expr` does not.)
    expected = eval(src, {"__builtins__": {}}, dict(syms))
    assert got == expected


def test_parse_division_by_literal_zero_still_raises():
    pool = ak.ExprPool()
    with pytest.raises(ZeroDivisionError):
        ak.parse("x * y / 0 * z", pool)


def test_parse_with_symbols_from_another_pool_keeps_old_behaviour():
    p, q = ak.ExprPool(), ak.ExprPool()
    qx, qy = q.symbol("x"), q.symbol("y")
    # Every operand lives in `q`: the fold builds there, as it always did.
    assert ak.parse("x + y - x*y", p, {"x": qx, "y": qy}) == qx + qy - qx * qy
    # Mixed pools still raise the mismatch.
    with pytest.raises(ak.PoolError):
        ak.parse("x + t", p, {"x": qx})


def test_parse_long_sum_is_not_quadratic():
    n = 20_000
    src = " + ".join(f"x^{i}" for i in range(n))
    pool = ak.ExprPool()
    x = pool.symbol("x")
    t0 = time.perf_counter()
    got = ak.parse(src, pool, {"x": x})
    elapsed = time.perf_counter() - t0
    assert got == pool.add([x.pow_expr(pool.integer(i)) for i in range(n)])
    # The left fold took ~7 s and 1.6 GB here; the single `add` well under 1 s.
    assert elapsed < 5.0, elapsed


# ---------------------------------------------------------------------------
# Lazy jax
# ---------------------------------------------------------------------------


def _run(code: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_import_alkahest_does_not_import_jax():
    out = _run(
        """
        import sys
        import alkahest
        print("jax" in sys.modules, "alkahest._jax" in sys.modules)
        """
    )
    assert out.split() == ["False", "False"]


def test_to_jax_still_resolves_lazily():
    out = _run(
        """
        import sys
        import alkahest
        print("to_jax" in dir(alkahest))
        from alkahest import to_jax
        print(to_jax is alkahest.to_jax, callable(to_jax))
        """
    )
    assert out.split() == ["True", "True", "True"]


def test_to_jax_without_jax_names_the_install():
    out = _run(
        """
        import sys
        sys.modules["jax"] = None  # make `import jax` fail
        import alkahest
        p = alkahest.ExprPool()
        x = p.symbol("x")
        try:
            alkahest.to_jax(x, [x])
        except ImportError as exc:
            print("pip install jax" in str(exc))
        """
    )
    assert out.split() == ["True"]


# ---------------------------------------------------------------------------
# Certificate gate fast path
# ---------------------------------------------------------------------------


def test_certificate_required_tracks_nested_contexts():
    from alkahest._certificates import certificate_required

    assert certificate_required() is False
    with ak.context(require_certificate=True):
        assert certificate_required() is True
        with ak.context(require_certificate=False):
            assert certificate_required() is False
        with ak.context(precision=64):  # a frame without the key shadows it
            assert certificate_required() is False
        assert certificate_required() is True
    assert certificate_required() is False


def test_certificate_required_is_per_thread():
    import threading

    from alkahest._certificates import certificate_required

    entered, release = threading.Event(), threading.Event()
    seen = []

    def other():
        with ak.context(require_certificate=True):
            seen.append(certificate_required())
            entered.set()
            release.wait(10)

    t = threading.Thread(target=other)
    t.start()
    assert entered.wait(10)
    try:
        # A frame is open on another thread: this one still has none.
        assert certificate_required() is False
    finally:
        release.set()
        t.join(10)
    assert seen == [True]
    assert certificate_required() is False


# ---------------------------------------------------------------------------
# Research dependency keys
# ---------------------------------------------------------------------------


def test_expr_key_separates_pools():
    p, q = ak.ExprPool(), ak.ExprPool()
    assert R._expr_key(p.symbol("x")) != R._expr_key(q.symbol("zzz"))
    assert R._expr_key(p.symbol("x")) == R._expr_key(p.symbol("x"))


def test_partial_product_is_matched_without_being_interned(tmp_path):
    pool = ak.ExprPool()
    a, b, c = pool.symbol("a"), pool.symbol("b"), pool.symbol("c")
    wide = pool.integer(2) * a * b * c
    a_plus_c, ab, bc = a + c, a * b, b * c
    before = tmp_path / "before.alkp"
    pool.save_to(str(before))

    # Nothing matching is registered: the lookup used to intern every
    # partial product of `wide` (2a, 2b, ab, abc, ...) into the pool anyway.
    session = R.ResearchSession(pool=pool)
    session._register_origin(a_plus_c, "unrelated")
    assert session._origins_in(wide) == []

    session2 = R.ResearchSession(pool=pool)
    session2._register_origin(ab, "ab")
    session2._register_origin(bc, "bc")
    session2._register_origin(a, "a")
    assert set(session2._origins_in(wide)) == {"ab", "bc", "a"}

    after = tmp_path / "after.alkp"
    pool.save_to(str(after))
    assert before.stat().st_size == after.stat().st_size


def test_partial_product_keeps_noncommutative_order():
    pool = ak.ExprPool()
    A = pool.symbol("A", commutative=False)
    B = pool.symbol("B", commutative=False)
    C = pool.symbol("C", commutative=False)
    session = R.ResearchSession(pool=pool)
    session._register_origin(A * C, "AC")
    session._register_origin(C * A, "CA")
    assert session._origins_in(A * B * C) == ["AC"]


def test_subexpressions_walk_is_linear():
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    n = 400
    e = pool.add([pool.mul([pool.integer(i + 1), x**i, y ** (n - i)]) for i in range(n)])
    subs = list(R._subexpressions(e))
    assert len(subs) == len(set(subs))
    session = R.ResearchSession(pool=pool)
    session._register_origin(x**7, "x7")
    t0 = time.perf_counter()
    assert session._origins_in(e) == ["x7"]
    assert time.perf_counter() - t0 < 2.0
