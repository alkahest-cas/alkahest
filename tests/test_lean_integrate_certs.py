"""``HasDerivAt`` certificates for indefinite integrals.

``alkahest.integrate`` certifies ``∫ f dx = F`` with a Lean theorem

    theorem alkahest_integral (x : ℝ) (h₁ : …) … :
        HasDerivAt (fun (x : ℝ) => F) (f) x

These tests check, independently of Lean, that the statement is about the
``F`` and ``f`` the API returned (numerically, at points satisfying the
hypotheses), that the derivative claim is true (central differences), that no
admission appears, and that withheld integrals carry a specific reason code.
When the Lean toolchain with a built Mathlib is present the whole corpus is
compiled in one file (Mathlib loads once).
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys

import alkahest as ak
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lean_integrate_corpus import INTEGRANDS, certificates  # noqa: E402
from test_lean_faithful import (  # noqa: E402
    close,
    parse_hasderiv,
    py_eval,
    sample_env,
    statements,
)

_RESULTS = list(certificates())


def _value(expr, env):
    x = ak.symbol("x")
    return float(ak.evaluate(expr, {x: env["x"]}, mode="f64").value)


@pytest.mark.parametrize(("src", "result"), _RESULTS, ids=[s for s, _ in _RESULTS])
def test_statement_is_the_returned_antiderivative(src, result):
    cert = result.certificate
    if cert is None:
        st = result.certificate_status
        assert st["certifiable"] is False
        assert st["detail"], f"{src}: withheld without a reason code"
        pytest.skip(f"withheld: {st['detail']}")
    for token in ("sorry", "admit", "axiom"):
        assert token not in cert
    assert result.certificate_status["certifiable"] is True
    stmts = statements(cert)
    assert len(stmts) == 1, cert
    binders, goal = stmts[0]
    parsed = parse_hasderiv(goal)
    assert parsed, goal
    var, body, deriv, point = parsed
    assert var == point == "x"
    rng = random.Random(7)
    for _ in range(4):
        env = sample_env(binders, rng)
        # The function is the returned F, the derivative the integrand.
        assert close(py_eval(body, env), _value(result.value, env), 1e-8), (src, body)
        h = 1e-6
        num = (py_eval(body, {**env, "x": env["x"] + h}) - py_eval(body, {**env, "x": env["x"] - h})) / (
            2 * h
        )
        assert close(num, py_eval(deriv, env), 1e-4), (src, goal)


def test_most_of_the_corpus_is_certified():
    certified = [s for s, r in _RESULTS if r.certificate is not None]
    assert len(certified) >= 0.75 * len(INTEGRANDS), [
        (s, r.certificate_status) for s, r in _RESULTS if r.certificate is None
    ]


def test_integrand_is_stated_verbatim():
    x = ak.symbol("x")
    r = ak.integrate(1 / (x**2 + 1), x)
    goal = statements(r.certificate)[0][1]
    _, body, deriv, _ = parse_hasderiv(goal)
    assert "Real.arctan" in body
    assert close(py_eval(deriv, {"x": 0.5}), 1 / 1.25)


def test_special_function_answer_is_withheld_with_reason():
    x = ak.symbol("x")
    r = ak.integrate(ak.exp(-(x**2)), x)
    st = r.certificate_status
    assert st["certifiable"] is False
    assert st["reason"] == "withheld_integration_shape"
    assert st["detail"] == "withheld_special_function"


def test_domain_hypotheses_are_explicit():
    x = ak.symbol("x")
    r = ak.integrate(1 / x, x)
    binders, _ = statements(r.certificate)[0]
    assert "≠ 0" in binders
    r = ak.integrate(ak.sqrt(1 - x**2), x)
    binders, _ = statements(r.certificate)[0]
    assert "-1 < " in binders and " < 1" in binders


_LEAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lean")


def _lake():
    lake = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
    built = os.path.isdir(os.path.join(_LEAN_DIR, ".lake", "packages", "mathlib", ".lake", "build"))
    return lake if os.path.exists(lake) and built else None


@pytest.mark.skipif(_lake() is None, reason="Lean toolchain / built Mathlib not available")
def test_corpus_compiles(tmp_path):
    imports, bodies = [], []
    for i, (_, r) in enumerate(_RESULTS):
        if r.certificate is None:
            continue
        body = []
        for line in r.certificate.splitlines():
            if line.startswith("import "):
                if line not in imports:
                    imports.append(line)
            else:
                body.append(line)
        bodies.append(f"namespace I{i}\n" + "\n".join(body) + f"\nend I{i}\n")
    path = tmp_path / "integrals.lean"
    path.write_text("\n".join(imports) + "\n\n" + "\n".join(bodies))
    proc = subprocess.run(
        [_lake(), "env", "lean", "-DwarningAsError=true", "-DautoImplicit=false", str(path)],
        cwd=_LEAN_DIR,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, f"{proc.stdout}{proc.stderr}"
