"""Parametric Routh–Hurwitz stability conditions (pytest)."""


def test_quadratic_parametric_condition():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    s = pool.symbol("s")
    a = pool.symbol("a")
    b = pool.symbol("b")
    # s^2 + a*s + b
    poly = s**2 + a * s + b
    res = alkahest.routh_hurwitz(poly, s)

    assert res["degree"] == 2
    # first column: [1, a, b]
    assert len(res["first_column"]) == 3
    cond = str(res["condition"])
    # Condition is a > 0 ∧ b > 0
    assert "a" in cond
    assert "b" in cond
    assert ">" in cond


def test_cubic_parametric_condition():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    s = pool.symbol("s")
    a = pool.symbol("a")
    b = pool.symbol("b")
    c = pool.symbol("c")
    # s^3 + a*s^2 + b*s + c -> a>0, c>0, a*b - c > 0
    poly = s**3 + a * s**2 + b * s + c
    res = alkahest.routh_hurwitz(poly, s)

    assert res["degree"] == 3
    cols = [str(e) for e in res["first_column"]]
    # The middle first-column entry should encode a*b - c (textbook condition).
    joined = " ".join(cols)
    assert "a" in joined
    assert "b" in joined
    assert "c" in joined
    cond = str(res["condition"])
    assert cond.count(">") == 3


def test_numeric_stable_instance_first_column_positive():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    s = pool.symbol("s")
    # s^3 + 2 s^2 + 3 s + 1: a=2,b=3,c=1 -> a*b - c = 5 > 0 => STABLE.
    poly = s**3 + pool.integer(2) * s**2 + pool.integer(3) * s + pool.integer(1)
    res = alkahest.routh_hurwitz(poly, s)
    assert res["degree"] == 3
    # All first-column entries are positive numeric constants.
    vals = [float(str(e)) for e in res["first_column"]]
    assert all(v > 0 for v in vals)


def test_numeric_unstable_instance_violates_condition():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    s = pool.symbol("s")
    # s^3 + s^2 + s + 6: a*b - c = 1 - 6 = -5 < 0 => UNSTABLE.
    poly = s**3 + s**2 + s + pool.integer(6)
    res = alkahest.routh_hurwitz(poly, s)
    vals = [float(str(e)) for e in res["first_column"]]
    # At least one first-column entry is non-positive.
    assert any(v <= 0 for v in vals)
