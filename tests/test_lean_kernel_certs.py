"""Reflective-kernel (``Alkahest.Kernel``) Lean certificates.

Integration certificates are, by default, data checked by the verified Lean
library in ``lean/Alkahest/``: ``integral_sound _ (by decide +kernel)``. These
tests check the Python surface (``certificate_status["path"]`` /
``["kernel_detail"]``, :func:`alkahest.set_lean_certificate_mode`), the
fallback to the tactic emitter, and the ODE client
:func:`alkahest.experimental.dsolve_certificate`. The Lean CI job compiles the
certificates themselves (``tests/lean_integrate_corpus.py``).
"""

from __future__ import annotations

import alkahest as ak
import pytest
from alkahest import experimental as ex

KERNEL_INTEGRANDS = [
    "x^2*sin(x)",
    "1/x",
    "1/(x^2+1)",
    "sin(x)*cos(x)",
    "x*sqrt(x^2+1)",
    "exp(x)/(1+exp(x))",
    "log(x)^3",
    "atan(x)",
    "tan(x)",
    "sinh(x)^2*cosh(x)",
]


@pytest.fixture(autouse=True)
def _default_mode():
    ak.set_lean_certificate_mode("auto")
    yield
    ak.set_lean_certificate_mode("auto")


def _integrate(src):
    x = ak.symbol("x")
    return ak.integrate(ak.parse(src), x)


@pytest.mark.parametrize("src", KERNEL_INTEGRANDS)
def test_default_certificate_is_kernel(src):
    r = _integrate(src)
    cert = r.certificate
    st = r.certificate_status
    assert cert is not None, st
    assert st["path"] == "kernel", st
    assert st["kernel_detail"] is None
    assert cert.startswith("import Alkahest.Kernel\n")
    assert "integral_sound" in cert
    assert "by decide +kernel" in cert
    for token in ("sorry", "admit", "axiom", "native_decide"):
        assert token not in cert


def test_special_function_falls_back_to_tactic():
    # ∫ exp(-x²) = √π/2 · erf x: the kernel has no `erf`, the tactic emitter defines it.
    r = _integrate("exp(-x^2)")
    st = r.certificate_status
    assert st["path"] == "tactic"
    assert st["kernel_detail"] == "withheld_special_function"
    assert "alkahestErf" in r.certificate


def test_tactic_mode_uses_the_tactic_emitter():
    r = _integrate("x^2*sin(x)")
    ak.set_lean_certificate_mode("tactic")
    assert ak.lean_certificate_mode() == "tactic"
    st = r.certificate_status
    assert st["path"] == "tactic"
    assert st["kernel_detail"] is None
    assert "linear_combination" in r.certificate


def test_native_mode_is_opt_in_and_reported():
    r = _integrate("x*exp(x)")
    assert "native_decide" not in r.certificate
    ak.set_lean_certificate_mode("kernel_native")
    assert r.certificate_status["path"] == "kernel_native"
    assert "by native_decide" in r.certificate


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        ak.set_lean_certificate_mode("fast")
    assert ak.lean_certificate_mode() == "auto"


def test_domain_hypotheses_and_derived_positivity():
    # log needs x ≠ 0 (a hypothesis); √(x²+1) needs x²+1 > 0 (derived by the kernel).
    cert = _integrate("1/x").certificate
    assert "(h1 : (x : ℝ) ≠ 0)" in cert
    cert = _integrate("x*sqrt(x^2+1)").certificate
    assert "(h1 :" not in cert


def test_status_keys_for_non_integration_results():
    x = ak.symbol("x")
    st = ak.diff(ak.sin(x), x).certificate_status
    assert "path" in st
    assert st["kernel_detail"] is None


def test_dsolve_solution_certificate():
    x, y, yp, ypp = (ak.symbol(n) for n in ("x", "y", "yp", "ypp"))
    eq = ak.parse("ypp + y")
    sol = ex.dsolve(eq, x, y, [yp, ypp])[0]["y_of_x"]
    out = ex.dsolve_certificate(eq, x, y, [yp, ypp], sol)
    assert out["path"] == "kernel", out
    assert out["detail"] is None
    cert = out["certificate"]
    assert "ode_sound" in cert
    assert "by decide +kernel" in cert
    assert "(C1 : ℝ)" in cert
    for token in ("sorry", "admit", "axiom"):
        assert token not in cert
    native = ex.dsolve_certificate(eq, x, y, [yp, ypp], sol, native=True)
    assert native["path"] == "kernel_native"
    assert "by native_decide" in native["certificate"]


def test_dsolve_certificate_refuses_a_non_solution():
    x, y, yp = (ak.symbol(n) for n in ("x", "y", "yp"))
    out = ex.dsolve_certificate(ak.parse("yp - y"), x, y, [yp], ak.parse("sin(x)"))
    assert out["certificate"] is None
    assert out["detail"] == "withheld_identity_not_reduced"


def test_first_order_dsolve_answers_certify():
    x, y, yp = (ak.symbol(n) for n in ("x", "y", "yp"))
    eq = ak.parse("yp - 2*x*y")
    sols = ex.dsolve(eq, x, y, [yp])
    explicit = [s for s in sols if s["form"] == "explicit"]
    assert explicit
    out = ex.dsolve_certificate(eq, x, y, [yp], explicit[0]["y_of_x"])
    assert out["path"] == "kernel", out
