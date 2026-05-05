"""V2-9 CAD / real QE bindings (pytest)."""


def test_decide_forall_x_squared_plus_one_positive():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    z = pool.integer(0)
    one = pool.integer(1)
    body = pool.gt(x**2 + one, z)
    phi = alkahest.Forall(x, body)
    truth, wit = alkahest.decide(phi)
    assert truth is True
    assert wit is None


def test_decide_exists_x_squared_equals_two():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    two = pool.integer(2)
    body = pool.pred_eq(x**2, two)
    phi = alkahest.Exists(x, body)
    truth, wit = alkahest.decide(phi)
    assert truth is True
    assert isinstance(wit, dict)
    assert "x" in wit


def test_cad_lift_quadratic_roots():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    p = x**2 + pool.integer(-2)
    intervals = alkahest.cad_lift([p], x)
    assert len(intervals) == 2
