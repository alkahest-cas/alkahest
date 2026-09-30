"""Every malformed input to ``alkahest.parse`` raises ``ParseError`` (audit D).

The parser used to let other exception types escape: ``RecursionError`` for
deep nesting, ``ValueError`` for an integer literal past Python's
``int``-string digit limit, ``ZeroDivisionError`` for ``2/0`` — none of them an
``AlkahestError``, so a caller catching ``ParseError`` (or
``AlkahestError``) around untrusted text did not catch them.  ``1e999999``
parsed silently as ``inf`` and ``1e-999999`` as ``0``.

The same audit found a handful of other entry points that raised
``OverflowError`` or ``TypeError`` for an out-of-range or empty argument where
their neighbours raise ``ValueError`` / ``IndexError``; those are pinned here
too.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from alkahest import ExprPool
from alkahest.exceptions import AlkahestError, ParseError


@pytest.fixture
def pool():
    return ExprPool()


# -- nesting ---------------------------------------------------------------

DEEP = 100_000


@pytest.mark.parametrize(
    "src",
    [
        "(" * DEEP + "x" + ")" * DEEP,
        "x" + "^x" * DEEP,
        "-" * DEEP + "x",
        "+" * DEEP + "x",
        "sin(" * DEEP + "x" + ")" * DEEP,
        "x" + "**(x" * DEEP + ")" * DEEP,
    ],
    ids=["parens", "pow-chain", "minus", "plus", "sin", "pow-parens"],
)
def test_deep_nesting_is_a_parse_error_not_recursion_error(pool, src):
    with pytest.raises(ParseError) as info:
        ak.parse(src, pool)
    assert "nesting too deep" in str(info.value)
    assert info.value.code == "E-PARSE-004"
    assert info.value.span is not None


def test_nesting_at_the_limit_still_parses(pool):
    # The limit matches the Rust parser's (MAX_EXPR_DEPTH = 2048): the
    # top-level expression is one level, so 2047 parentheses fit.
    n = 2047
    e = ak.parse("(" * n + "x" + ")" * n, pool)
    assert e == pool.symbol("x")
    with pytest.raises(ParseError):
        ak.parse("(" * (n + 1) + "x" + ")" * (n + 1), pool)


def test_nesting_near_the_limit_works_in_a_thread(pool):
    # No interpreter frames per level: a thread with a small stack copes too.
    import threading

    out = {}

    def run():
        try:
            out["e"] = ak.parse("(" * 2000 + "x" + ")" * 2000, pool)
        except BaseException as exc:  # pragma: no cover - reported below
            out["exc"] = exc

    old = threading.stack_size()
    threading.stack_size(256 * 1024)
    try:
        t = threading.Thread(target=run)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    assert "exc" not in out, out.get("exc")
    assert out["e"] == pool.symbol("x")


def test_long_flat_input_is_not_deep(pool):
    # Length is not depth: a 20 000-term sum is two levels.
    e = ak.parse(" + ".join(f"x{i}" for i in range(20_000)), pool)
    assert "x19999" in str(e)


# -- literals --------------------------------------------------------------


def test_huge_integer_literal_is_exact(pool):
    # 100 000 digits: past Python's default int-string limit (4300), which
    # ``int(text)`` enforced with a ValueError.  Built here without any
    # decimal conversion, so the comparison does not hit the limit either.
    e = ak.parse("9" * 100_000, pool)
    assert e == pool.integer(10**100_000 - 1)


@pytest.mark.parametrize("src", ["1.5e99999999999999", "-1e999999999", "1e-999999999"])
def test_float_literal_past_every_exponent_is_refused(pool, src):
    # Past even MPFR's exponent range (about ±3·10^8 decimal digits): refused,
    # never read as ±inf or as 0.
    with pytest.raises(ParseError, match="out of range") as info:
        ak.parse(src, pool)
    assert info.value.code == "E-PARSE-001"


def test_float_literal_within_the_big_exponent_range_is_not_inf_or_zero(pool):
    # Past f64 but inside MPFR's range: the value at 53 bits.
    big = ak.parse("1e999999", pool)
    assert "inf" not in str(big).lower()
    assert big != ak.parse("1e999998", pool)
    tiny = ak.parse("1e-999999", pool)
    assert tiny != pool.float(0.0)
    assert str(tiny) not in ("0", "0.0", "-0")


def test_ordinary_float_literals_unchanged(pool):
    assert ak.parse("1.5", pool) == pool.float(1.5)
    assert ak.parse("1e300", pool) == pool.float(1e300)
    assert ak.parse("2.5e-3", pool) == pool.float(2.5e-3)
    assert ak.parse("0.0", pool) == pool.float(0.0)
    assert ak.parse("0e5", pool) == pool.float(0.0)


# -- division by a literal zero -------------------------------------------


@pytest.mark.parametrize(("src", "col"), [("2/0", 1), ("x/0", 1), ("0/0", 1), ("1 + x*y / 0", 8)])
def test_division_by_literal_zero_is_a_parse_error(pool, src, col):
    with pytest.raises(ParseError) as info:
        ak.parse(src, pool)
    assert "division by zero" in str(info.value)
    assert info.value.span == (col, col + 1)
    assert info.value.code == "E-PARSE-002"
    # Still the ZeroDivisionError parse raised before, for existing handlers.
    assert isinstance(info.value, ZeroDivisionError)


def test_parse_errors_are_alkahest_errors(pool):
    for src in ["2/0", "(" * 5000 + "x" + ")" * 5000, "sin()", "x +", "1e999999999"]:
        with pytest.raises(AlkahestError):
            ak.parse(src, pool)


def test_parse_error_default_code_unchanged():
    assert ParseError("boom").code == "E-PARSE-001"


# -- other out-of-range arguments ------------------------------------------


def test_interval_eval_precision_out_of_range_is_value_error(pool):
    x = pool.symbol("x")
    ball = ak.ArbBall(1.0, 0.0, 64)
    for prec in (0, 2**31, 2**40, 2**70):
        with pytest.raises(ValueError, match="precision"):
            ak.interval_eval(x, {x: ball}, prec)
    with pytest.raises(ValueError):
        ak.interval_eval(x, {x: ball}, -1)


def test_matrix_with_empty_row_is_value_error():
    with pytest.raises(ValueError, match="column"):
        ak.Matrix([[]])
    with pytest.raises(ValueError):
        ak.Matrix([])


def test_matrix_get_negative_index_is_index_error(pool):
    x = pool.symbol("x")
    m = ak.Matrix([[x, x], [x, x]])
    for i, j in [(-1, 0), (0, -1), (2**70, 0), (5, 0)]:
        with pytest.raises(IndexError, match="out of range"):
            m.get(i, j)


def test_binomial_mod_negative_arguments_are_typed_errors():
    # A bad modulus p^k is E-HOLO-006, as for a composite p; a bad a or b is
    # a malformed call, E-HOLO-004.
    cases = [
        ((-5, 2, 7, 1), "E-HOLO-004"),
        ((2**70, 2, 7, 1), "E-HOLO-004"),
        ((5, 2**200, 7, 1), "E-HOLO-004"),
        ((5, 2, -7, 1), "E-HOLO-006"),
        ((5, 2, 7, -1), "E-HOLO-006"),
        ((5, 2, 2**70, 1), "E-HOLO-006"),
    ]
    for args, code in cases:
        with pytest.raises(ak.HolonomicError) as info:
            ak.binomial_mod(*args)
        assert info.value.code == code, (args, info.value.code)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shrink_iters": -1},
        {"explore_iters": -1},
        {"const_fold_iters": -1},
        {"node_limit": -1},
        {"iter_limit": -1},
        {"node_limit": 2**70},
    ],
)
def test_egraph_config_negative_is_value_error(kwargs):
    with pytest.raises(ValueError, match="non-negative"):
        ak.EgraphConfig(**kwargs)


def test_egraph_config_positional_negative_is_value_error():
    with pytest.raises(ValueError):
        ak.EgraphConfig(-1)
