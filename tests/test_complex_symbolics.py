import alkahest as ak
from alkahest.experimental import conjugate, im, re


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
