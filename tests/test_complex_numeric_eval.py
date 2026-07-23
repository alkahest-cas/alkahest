import alkahest as ak
from alkahest.experimental import arg, evaluate, im, re


def test_complex_auto_mode_with_complex_binding():
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)

    result = evaluate(z, {z: 2 + 3j})

    assert result.status == "ok"
    assert result.backend == "interpreter_complex_f64"
    assert result.value == 2 + 3j


def test_complex_re_im_parts():
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)

    assert evaluate(re(z), {z: 2 + 3j}).value == 2 + 0j
    assert evaluate(im(z), {z: 2 + 3j}).value == 3 + 0j


def test_arg_branch_cut_declines():
    pool = ak.ExprPool()

    result = evaluate(arg(pool.integer(-1)), {}, mode="complex")

    assert result.status == "unsupported"
    assert result.reason == "E-EVAL-011"


def test_imaginary_unit_auto_binds_in_complex_mode():
    pool = ak.ExprPool()
    i = pool.imaginary_unit()
    assert evaluate(i, {}, mode="complex").value == 1j
    assert evaluate(i * i, {}, mode="complex").value == -1 + 0j


def test_principal_half_power_of_negative():
    pool = ak.ExprPool()
    result = evaluate(pool.integer(-1) ** pool.rational(1, 2), {}, mode="complex")
    assert result.status == "ok"
    assert abs(result.value - 1j) < 1e-9
