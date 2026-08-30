"""Smoke tests: one test per major subsystem so a broken build fails fast."""

import warnings

import alkahest
import pytest
from alkahest import (
    HAS_EGRAPH,
    ArbBall,
    ExprPool,
    MultiPoly,
    RationalFunction,
    UniPoly,
    compile_expr,
    cos,
    diff,
    eval_expr,
    exp,
    integrate,
    interval_eval,
    jit_is_available,
    latex,
    limit,
    log,
    parse,
    simplify,
    sin,
    to_lean,
    unicode_str,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pool():
    return ExprPool()


# ---------------------------------------------------------------------------
# Core import and version
# ---------------------------------------------------------------------------


def test_import():
    assert alkahest is not None


def test_version():
    v = alkahest.version()
    assert isinstance(v, str)
    assert len(v) > 0


# ---------------------------------------------------------------------------
# Differentiation
# ---------------------------------------------------------------------------


def test_diff_sin():
    p = pool()
    x = p.symbol("x")
    r = diff(sin(x), x)
    # d/dx sin(x) = cos(x)
    assert r is not None
    assert r.value is not None
    assert len(r.steps) > 0


def test_diff_polynomial():
    p = pool()
    x = p.symbol("x")
    # d/dx x^3 = 3*x^2
    r = diff(x**3, x)
    assert r is not None
    assert r.value is not None


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_integrate_exp():
    p = pool()
    x = p.symbol("x")
    r = integrate(exp(x), x)
    assert r is not None
    assert r.value is not None


def test_integrate_constant():
    p = pool()
    x = p.symbol("x")
    five = p.integer(5)
    r = integrate(five, x)
    assert r is not None
    assert r.value is not None


def test_integrate_exact_verification_metadata():
    p = pool()
    x = p.symbol("x")
    r = integrate(x**2, x)

    assert r.verification["status"] == "exactly_verified"
    assert r.verification["evidence"] == "antiderivative_derivative_identity"
    assert r.verification["method"] == "in_kernel_symbolic_residual"
    assert r.verification["externally_verified"] is False


def test_non_integration_retains_derivation_verification_metadata():
    p = pool()
    x = p.symbol("x")

    verification = diff(x**2, x).verification
    assert verification["status"] == "certificate_available"
    assert verification["method"] == "derivation_log"


# ---------------------------------------------------------------------------
# Simplification (rule-based)
# ---------------------------------------------------------------------------


def test_simplify_zero():
    p = pool()
    x = p.symbol("x")
    zero = p.integer(0)
    r = simplify(x + zero)
    assert r.value == x


def test_simplify_trig_identity():
    """sin^2 + cos^2 = 1 via rule-based simplifier (or e-graph if available)."""
    p = pool()
    x = p.symbol("x")
    expr = sin(x) ** 2 + cos(x) ** 2
    if HAS_EGRAPH:
        from alkahest import simplify_egraph

        r = simplify_egraph(expr)
        # The e-graph should collapse this to 1
        assert r.value == p.integer(1)
    else:
        # Without e-graph, at least check that simplify runs without error
        r = simplify(expr)
        assert r.value is not None


# ---------------------------------------------------------------------------
# Expression parsing and pretty-printing
# ---------------------------------------------------------------------------


def test_parse_roundtrip():
    p = pool()
    x = p.symbol("x")
    e = parse("x^2 + 2*x + 1", p, {"x": x})
    assert e is not None


def test_latex_output():
    p = pool()
    x = p.symbol("x")
    out = latex(sin(x))
    assert "sin" in out.lower()


def test_unicode_output():
    p = pool()
    x = p.symbol("x")
    out = unicode_str(x**2)
    assert out is not None
    assert len(out) > 0


# ---------------------------------------------------------------------------
# JIT compiled evaluation / interpreter
# ---------------------------------------------------------------------------


def test_jit_is_available_returns_bool():
    assert isinstance(jit_is_available(), bool)


def test_compile_expr_constant():
    p = pool()
    five = p.integer(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = compile_expr(five, [])
    assert abs(f([]) - 5.0) < 1e-10


def test_compile_expr_variable():
    p = pool()
    x = p.symbol("x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = compile_expr(x**2, [x])
    assert abs(f([3.0]) - 9.0) < 1e-10


def test_eval_expr_basic():
    p = pool()
    x = p.symbol("x")
    result = eval_expr(x**2 + p.integer(1), {x: 3.0})
    assert abs(result - 10.0) < 1e-10


def test_compile_warns_when_jit_unavailable():
    """When JIT is not compiled in, compile_expr should emit a RuntimeWarning."""
    if jit_is_available():
        pytest.skip("JIT is available in this build — no fallback warning expected")
    p = pool()
    x = p.symbol("x")
    with pytest.warns(RuntimeWarning, match="not available"):
        compile_expr(x, [x])


# ---------------------------------------------------------------------------
# Polynomial subsystem
# ---------------------------------------------------------------------------


def test_unipoly_degree():
    p = pool()
    x = p.symbol("x")
    poly = UniPoly.from_symbolic(x**3 + p.integer(-2) * x + p.integer(1), x)
    assert poly.degree == 3


def test_unipoly_gcd():
    p = pool()
    x = p.symbol("x")
    a = UniPoly.from_symbolic(x**2 + p.integer(-1), x)
    b = UniPoly.from_symbolic(x + p.integer(-1), x)
    g = a.gcd(b)
    assert g is not None


def test_multipoly_total_degree():
    p = pool()
    x = p.symbol("x")
    y = p.symbol("y")
    mp = MultiPoly.from_symbolic(x**2 * y + x * y**2, [x, y])
    assert mp.total_degree == 3


def test_rational_function_normalization():
    p = pool()
    x = p.symbol("x")
    rf = RationalFunction.from_symbolic(x**2 + p.integer(-1), x + p.integer(-1), [x])
    # (x^2 - 1)/(x - 1) = x + 1
    assert rf is not None


# ---------------------------------------------------------------------------
# Ball arithmetic (Arb)
# ---------------------------------------------------------------------------


def test_arb_ball_construction():
    b = ArbBall(1.0, 1e-10)
    s = str(b)
    assert s is not None


def test_interval_eval_sin():
    p = pool()
    x = p.symbol("x")
    result = interval_eval(sin(x), {x: ArbBall(1.0, 1e-10)})
    assert result is not None


# ---------------------------------------------------------------------------
# Polynomial system solver (Gröbner; optional)
# ---------------------------------------------------------------------------


def test_solver_linear_system():
    try:
        from alkahest import solve
    except ImportError:
        pytest.skip("groebner feature not available")
    p = pool()
    x = p.symbol("x")
    y = p.symbol("y")
    neg_one = p.integer(-1)
    # x + y - 1 = 0,  x - y = 0  →  x = y = 0.5
    r = solve([x + y + neg_one, x + p.integer(-1) * y], [x, y])
    assert r is not None


# ---------------------------------------------------------------------------
# Lean 4 certificate export
# ---------------------------------------------------------------------------


def test_lean_export():
    p = pool()
    x = p.symbol("x")
    # to_lean takes an Expr, not a DerivedResult. A bare Expr whose default
    # simplification does real work (here `x + 0 → x` via add_zero) emits a
    # certificate of that derivation.
    lean = to_lean(x + p.integer(0))
    assert isinstance(lean, str)
    assert len(lean) > 0
    assert "add_zero" in lean or "simp" in lean


def test_lean_export_withholds_vacuous_identity():
    """Part C: a bare Expr the default simplifier leaves untouched has no
    derivation, so `to_lean` must not present `example : e = e := rfl` (which
    reads as a proven theorem). It withholds (returns "") instead."""
    p = pool()
    x = p.symbol("x")
    # `x**2` is already canonical (empty derivation log). The rewrite that would
    # touch `exp(log(x))` lives in `simplify_log_exp`, not the default set, so
    # the default simplifier also leaves it untouched.
    for expr in (x**2, exp(log(x))):
        src = to_lean(expr)
        assert src == "", f"vacuous identity must withhold, got: {src!r}"
        assert "rfl" not in src


def test_lean_diff_export():
    p = pool()
    x = p.symbol("x")
    result = diff(x**3, x)
    lean = to_lean(result)
    assert "deriv (fun" in lean
    assert "deriv_pow" in lean
    assert "MeasureTheory" not in lean
    assert "sorry" not in lean
    assert "admit" not in lean
    assert any(step["rule"] == "diff_univariate_poly" for step in result.steps)


def test_lean_integrate_certifies_via_ftc_derivative():
    """Part A: ∫ f dx = F certifies via the FTC derivative relation
    `deriv (fun x => F) x = f`, never the false unwrapped equality `f = F`
    (e.g. `sin x = -cos x`)."""
    p = pool()
    x = p.symbol("x")
    r = integrate(sin(x), x)
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert to_lean(r) == cert
    # The sound statement is the derivative relation, not `sin x = -cos x`.
    assert "deriv (fun" in cert
    assert "sorry" not in cert
    assert "admit" not in cert
    # Must never assert the false unwrapped integration equality.
    assert "Real.sin ((x : ℝ)) = " not in cert
    # Residual check still labels the antiderivative as exact.
    assert r.verification["status"] == "exactly_verified"


def test_lean_withholds_non_elementary_integrate_certificate():
    """An antiderivative whose derivative escapes the certifiable diff fragment
    (here `∫ log x dx = x log x − x`, whose product_rule step cannot close)
    stays withheld rather than emitting an admission."""
    p = pool()
    x = p.symbol("x")
    r = integrate(log(x), x)
    assert r.certificate is None
    assert to_lean(r) == ""


def test_lean_integrate_inv_x_via_log():
    """∫ x⁻¹ dx = log x certifies via FTC reuse of Real.deriv_log."""
    p = pool()
    x = p.symbol("x")
    r = integrate(x**-1, x)
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "Real.deriv_log" in cert
    assert to_lean(r) == cert


def test_lean_integrate_x_neg_two_withheld():
    """∫ x⁻² intern-equals d/dx(-x⁻¹) but product_rule cannot close — withhold."""
    p = pool()
    x = p.symbol("x")
    r = integrate(x**-2, x)
    assert r.certificate is None
    assert to_lean(r) == ""


def test_lean_diff_sin_certificate_has_no_sorry():
    """B3: simple d/dx sin(x) emits Real.deriv_sin, not sorry."""
    p = pool()
    x = p.symbol("x")
    r = diff(sin(x), x)
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "Real.deriv_sin" in cert
    assert r.verification["status"] == "certificate_available"


def test_lean_emits_chain_rule_composite_certificate():
    """Chain rule for f(x^n), f in {sin, cos, exp}, now emits a compiling cert."""
    p = pool()
    x = p.symbol("x")
    r = diff(sin(x**2), x)
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "hasDerivAt_pow" in cert
    assert "(hg.sin).deriv" in cert
    assert r.verification["status"] == "certificate_available"


def test_lean_withholds_chain_rule_log_composite():
    """Composite log(x^n) still routes through diff_log and stays withheld."""
    p = pool()
    x = p.symbol("x")
    r = diff(log(x**2), x)
    assert r.certificate is None
    assert r.verification["status"] == "unverified"


def test_lean_diff_x_squared_certificate():
    """d/dx x² must close coeff order (`2*x` vs `x*2`) without sorry."""
    p = pool()
    x = p.symbol("x")
    r = diff(x**2, x)
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "deriv (fun" in cert


def test_lean_diff_x_inv_certificate():
    """d/dx x⁻¹ emits hasDerivAt_inv with an explicit x ≠ 0 binder."""
    p = pool()
    x = p.symbol("x")
    r = diff(x**-1, x)
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "(hx : x ≠ 0)" in cert
    assert "hasDerivAt_inv" in cert
    assert r.verification["status"] == "certificate_available"


def test_lean_diff_x_neg_two_certificate():
    """d/dx x⁻² emits hasDerivAt_inv + HasDerivAt.pow with x ≠ 0."""
    p = pool()
    x = p.symbol("x")
    r = diff(x**-2, x)
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "(hx : x ≠ 0)" in cert
    assert "hasDerivAt_inv" in cert
    assert "hinv.pow 2" in cert
    assert r.verification["status"] == "certificate_available"


def test_lean_sum_and_product_rule_certificates():
    """Sum/product of unary trig/exp diffs should emit, not withhold."""
    p = pool()
    x = p.symbol("x")
    sum_r = diff(sin(x) + cos(x), x)
    assert sum_r.certificate
    assert "sorry" not in sum_r.certificate
    assert any(s["rule"] == "sum_rule" for s in sum_r.steps)

    prod_r = diff(sin(x) * exp(x), x)
    assert prod_r.certificate
    assert "sorry" not in prod_r.certificate
    assert any(s["rule"] == "product_rule" for s in prod_r.steps)


def test_lean_log_exp_parens_and_exp_log_certifies_with_positivity_hyp():
    """Nested log/exp must parenthesize; exp∘log certifies under x > 0."""
    from alkahest import Assumptions, simplify_log_exp

    p = pool()
    x = p.symbol("x")
    log_exp = simplify_log_exp(log(exp(x)))
    assert log_exp.certificate
    assert "Real.log (Real.exp" in log_exp.certificate
    assert "sorry" not in log_exp.certificate

    # exp(log(x)) needs `0 < x`; gate via Assumptions so the colored rewrite
    # records positivity and the Lean exporter emits `(hx : 0 < x)`.
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))
    exp_log = simplify_log_exp(exp(log(x)), assumptions)
    assert exp_log.certificate is not None
    assert "(hx : 0 < x)" in exp_log.certificate
    assert "Real.exp_log hx" in exp_log.certificate
    assert "sorry" not in exp_log.certificate
    assert to_lean(exp_log) != ""


def test_lean_tan_expand_certificate():
    from alkahest import simplify_trig, tan

    p = pool()
    x = p.symbol("x")
    r = simplify_trig(tan(x))
    assert r.certificate
    assert "div_eq_mul_inv" in r.certificate
    assert "Real.tan" in r.certificate
    assert "sorry" not in r.certificate


def test_lean_definite_integral_sum_certificate():
    """∫ (sin x + cos x) certifies via HasDerivAt.add / IntervalIntegrable.add
    composed on top of the interval FTC — the linear-combination fragment."""
    p = pool()
    x = p.symbol("x")
    r = integrate(sin(x) + cos(x), x, p.integer(0), p.integer(1))
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "admit" not in cert
    assert "intervalIntegral.integral_eq_sub_of_hasDerivAt" in cert
    assert ".add" in cert
    assert to_lean(r) == cert


def test_lean_definite_integral_constant_multiple_certificate():
    """∫ 3*cos(x) certifies via HasDerivAt.const_mul / .mul_const."""
    p = pool()
    x = p.symbol("x")
    r = integrate(3 * cos(x), x, p.integer(0), p.integer(1))
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "admit" not in cert
    assert "const_mul" in cert or "mul_const" in cert


def test_lean_definite_integral_negative_coefficient_certificate():
    """∫ -exp(x) (i.e. -1 * exp(x)) also certifies via the same scaling path."""
    p = pool()
    x = p.symbol("x")
    r = integrate(-exp(x), x, p.integer(0), p.integer(1))
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "admit" not in cert


def test_lean_definite_integral_rational_coefficient_certificate():
    """A Rational-literal coefficient (not just Integer) must also certify."""
    p = pool()
    x = p.symbol("x")
    half = p.rational(1, 2)
    r = integrate(half * x**2, x, p.integer(0), p.integer(1))
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "admit" not in cert


def test_lean_definite_integral_three_term_linear_combination_certificate():
    """A three-term sum mixing a bare power, a bare trig term, and a scaled
    trig term must certify as a single linear-combination certificate."""
    p = pool()
    x = p.symbol("x")
    r = integrate(x**2 + sin(x) + 3 * cos(x), x, p.integer(0), p.integer(1))
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "admit" not in cert


def test_lean_definite_integral_withholds_sum_with_unsupported_term():
    """One non-certifiable addend (log x) must withhold the WHOLE certificate,
    not silently drop that term."""
    p = pool()
    x = p.symbol("x")
    r = integrate(cos(x) + log(x), x, p.integer(1), p.integer(2))
    assert r.certificate is None
    assert to_lean(r) == ""


def test_lean_definite_integral_withholds_symbolic_coefficient():
    """A non-literal (symbolic) coefficient must not be mistaken for a
    constant multiple: `y * cos(x)` withholds rather than fabricating an
    unsound `HasDerivAt.const_mul` proof."""
    p = pool()
    x = p.symbol("x")
    y = p.symbol("y")
    r = integrate(y * cos(x), x, p.integer(0), p.integer(1))
    assert r.certificate is None
    assert to_lean(r) == ""


def test_lean_tendsto_exp_neg_x_certificate():
    """lim_{x→+∞} exp(-x) = 0 emits Filter.Tendsto via tendsto_exp_neg_atTop_nhds_zero."""
    p = pool()
    x = p.symbol("x")
    r = limit(exp(-x), x, p.pos_infinity())
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "admit" not in cert
    assert "tendsto_exp_neg_atTop_nhds_zero" in cert
    assert to_lean(r) == cert


def test_lean_tendsto_exp_x_certificate():
    """lim_{x→+∞} exp(x) = +∞ emits Filter.Tendsto via tendsto_exp_atTop."""
    p = pool()
    x = p.symbol("x")
    r = limit(exp(x), x, p.pos_infinity())
    cert = r.certificate
    assert isinstance(cert, str)
    assert cert
    assert "sorry" not in cert
    assert "admit" not in cert
    assert "tendsto_exp_atTop" in cert
    assert to_lean(r) == cert


def test_lean_tendsto_withholds_finite_sinc():
    """Finite limits (sin x / x as x → 0) stay withheld — no invented Tendsto proof."""
    p = pool()
    x = p.symbol("x")
    r = limit(sin(x) / x, x, p.integer(0))
    assert r.certificate is None
    assert to_lean(r) == ""


def test_lean_diff_sinh_cosh_atan_asin_certificates():
    """Pointwise hyperbolic / inverse-trig diffs emit, with asin carrying
    the |x| < 1 binder. Composites and tanh stay withheld."""
    from alkahest import asin, atan, cosh, sinh, tanh

    p = pool()
    x = p.symbol("x")

    sinh_r = diff(sinh(x), x)
    assert sinh_r.certificate
    assert "sorry" not in sinh_r.certificate
    assert "Real.deriv_sinh" in sinh_r.certificate

    cosh_r = diff(cosh(x), x)
    assert cosh_r.certificate
    assert "Real.deriv_cosh" in cosh_r.certificate

    sum_r = diff(sinh(x) + cosh(x), x)
    assert sum_r.certificate
    assert "sorry" not in sum_r.certificate

    prod_r = diff(exp(x) * sinh(x), x)
    assert prod_r.certificate
    assert "sorry" not in prod_r.certificate

    atan_r = diff(atan(x), x)
    assert atan_r.certificate
    assert "sorry" not in atan_r.certificate
    assert "hasDerivAt_arctan'" in atan_r.certificate
    assert "Real.arctan" in atan_r.certificate

    asin_r = diff(asin(x), x)
    assert asin_r.certificate
    assert "sorry" not in asin_r.certificate
    assert "(hx : -1 < x ∧ x < 1)" in asin_r.certificate
    assert "hasDerivAt_arcsin" in asin_r.certificate

    assert diff(asin(x**2), x).certificate is None
    assert diff(tanh(x), x).certificate is None


def test_lean_integrate_inv_one_plus_x_squared():
    """∫ (1+x²)⁻¹ dx certifies via d/dx atan(x). The definite form proves
    F(b)−F(a) = arctan 1 − arctan 0, not π/4."""
    p = pool()
    x = p.symbol("x")
    integrand = 1 / (1 + x**2)

    r = integrate(integrand, x)
    assert r.certificate
    assert "sorry" not in r.certificate
    assert "deriv (fun" in r.certificate
    assert "Real.arctan" in r.certificate or "hasDerivAt_arctan" in r.certificate
    # Residual check still labels the antiderivative as exact.
    assert "atan" in str(r.value)

    definite = integrate(integrand, x, p.integer(0), p.integer(1))
    assert definite.certificate
    assert "sorry" not in definite.certificate
    assert "intervalIntegral.integral_eq_sub_of_hasDerivAt" in definite.certificate
    assert "Real.arctan" in definite.certificate
    assert "π / 4" not in definite.certificate
    assert "Real.pi" not in definite.certificate


# ---------------------------------------------------------------------------
# StableHLO / XLA bridge
# ---------------------------------------------------------------------------


def test_stablehlo_export():
    p = pool()
    x = p.symbol("x")
    mlir = alkahest.to_stablehlo(x**2 + p.integer(1), [x])
    assert isinstance(mlir, str)
    assert "stablehlo" in mlir.lower() or "func" in mlir.lower()
