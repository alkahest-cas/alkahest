import alkahest as ak
from alkahest.experimental import arg, conjugate, im, re


def test_symbolic_complex_constructors_and_safe_constants():
    p = ak.ExprPool()
    z = p.symbol("z", ak.Domain.Complex)
    assert str(ak.simplify(conjugate(conjugate(z))).value) == "z"
    assert str(ak.simplify(re(p.integer(2))).value) == "2"
    assert str(ak.simplify(im(p.integer(2))).value) == "0"


def test_branch_sensitive_conjugation_remains_symbolic():
    p = ak.ExprPool()
    z = p.symbol("z", ak.Domain.Complex)
    assert str(conjugate(ak.log(z))) == "conjugate(log(z))"
    assert str(conjugate(ak.sqrt(z))) == "conjugate(sqrt(z))"


def test_principal_arg_safe_cases():
    p = ak.ExprPool()
    x = p.symbol("x", ak.Domain.Positive)
    i = p.symbol("I", ak.Domain.Complex)

    assert str(ak.simplify(arg(p.integer(3))).value) == "0"
    assert str(ak.simplify(arg(x)).value) == "0"
    assert "pi" in str(ak.simplify(arg(i)).value)
    assert str(ak.simplify(arg(p.integer(0))).value) == "arg(0)"
    assert str(ak.simplify(arg(p.integer(-1))).value) == "arg(-1)"


def test_principal_arg_leaves_branch_cut_and_generic_complex():
    p = ak.ExprPool()
    z = p.symbol("z", ak.Domain.Complex)
    assert str(arg(z)) == "arg(z)"
    assert str(arg(ak.log(z))) == "arg(log(z))"
    assert str(arg(ak.sqrt(z))) == "arg(sqrt(z))"
