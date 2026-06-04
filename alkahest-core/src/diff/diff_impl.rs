use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::UniPoly;
use crate::simplify::engine::simplify;
use std::collections::HashMap;
use std::fmt;

/// Build a canonical constant node for a rational `r`: an `Integer` when `r` is
/// integer-valued, otherwise a `Rational`.  Shared by all three diff modes for
/// the constant-exponent power rule.
pub(crate) fn const_node(pool: &ExprPool, r: rug::Rational) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        let (n, d) = r.into_numer_denom();
        pool.rational(n, d)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffError {
    /// An unknown function was encountered; differentiation is not defined.
    UnknownFunction(String),
    /// A `Pow` node whose exponent is not a constant integer.
    NonIntegerExponent,
    /// Forward-mode: unknown function (folded from the former `ForwardDiffError`).
    ForwardUnknownFunction(String),
    /// Forward-mode: non-integer exponent (folded from the former `ForwardDiffError`).
    ForwardNonIntegerExponent,
}

impl fmt::Display for DiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffError::UnknownFunction(name) => {
                write!(f, "cannot differentiate unknown function '{name}'")
            }
            DiffError::NonIntegerExponent => {
                write!(f, "cannot differentiate power with non-integer exponent")
            }
            DiffError::ForwardUnknownFunction(name) => {
                write!(f, "diff_forward: unknown function '{name}'")
            }
            DiffError::ForwardNonIntegerExponent => {
                write!(f, "diff_forward: non-integer exponent")
            }
        }
    }
}

impl std::error::Error for DiffError {}

impl crate::errors::AlkahestError for DiffError {
    fn code(&self) -> &'static str {
        match self {
            DiffError::UnknownFunction(_) => "E-DIFF-001",
            DiffError::NonIntegerExponent => "E-DIFF-002",
            DiffError::ForwardUnknownFunction(_) => "E-DIFF-003",
            DiffError::ForwardNonIntegerExponent => "E-DIFF-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DiffError::UnknownFunction(_) => Some(
                "register the function in PrimitiveRegistry, or use diff_forward with a custom rule",
            ),
            DiffError::NonIntegerExponent => Some(
                "symbolic exponents require the chain rule; use diff_forward for non-integer powers",
            ),
            DiffError::ForwardUnknownFunction(_) => Some(
                "register the function in PrimitiveRegistry with diff_forward implemented",
            ),
            DiffError::ForwardNonIntegerExponent => Some(
                "substitute concrete values first; diff_forward requires integer exponents",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Symbolically differentiate `expr` with respect to `var`.
///
/// The returned log records every rule applied, including post-differentiation
/// simplification steps appended at the end.
pub fn diff(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, DiffError> {
    // One memo table per top-level diff call.  Maps ExprId → derivative ExprId
    // so shared subexpressions are differentiated at most once.
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    let result = diff_raw(expr, var, pool, &mut memo)?;
    Ok(result.and_then(|v| simplify(v, pool)))
}

// ---------------------------------------------------------------------------
// Core recursive differentiation (no simplification)
// ---------------------------------------------------------------------------

#[inline]
fn diff_poly_try_univariate_fastpath(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    // Skip atoms so simple cases keep their dedicated log rules (`diff_identity`, `diff_const`, …).
    if matches!(
        pool.get(expr),
        ExprData::Symbol { .. } | ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
    ) {
        return None;
    }
    let poly = UniPoly::from_symbolic(expr, var, pool).ok()?;
    let der = poly.derivative();
    let result = der.to_symbolic_expr(pool);
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("diff_univariate_poly", expr, result));
    Some(DerivedExpr::with_log(result, log))
}

/// Memoised differentiation worker.
///
/// `memo` maps `ExprId → ExprId` (derivative value).  Shared subexpressions
/// are differentiated once; subsequent occurrences return the cached result
/// with an empty derivation log to avoid duplicate log entries.
fn diff_raw(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<DerivedExpr<ExprId>, DiffError> {
    // Return cached derivative for shared subexpressions.
    if let Some(&cached) = memo.get(&expr) {
        return Ok(DerivedExpr::new(cached));
    }

    if let Some(hit) = diff_poly_try_univariate_fastpath(expr, var, pool) {
        memo.insert(expr, hit.value);
        return Ok(hit);
    }

    // Extract only what we need from the pool in a single lock acquisition,
    // then release the lock before any recursive diff_raw calls.
    enum Node {
        IdentVar,
        Const,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow {
            base: ExprId,
            exp: ExprId,
        },
        Func {
            name: String,
            args: Vec<ExprId>,
        },
        Piecewise {
            branches: Vec<(ExprId, ExprId)>,
            default: ExprId,
        },
        RootSum {
            poly: ExprId,
            rvar: ExprId,
            body: ExprId,
        },
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } if expr == var => Node::IdentVar,
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Node::Const,
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow {
            base: *base,
            exp: *exp,
        },
        ExprData::Func { name, args } => Node::Func {
            name: name.clone(),
            args: args.clone(),
        },
        ExprData::Piecewise { branches, default } => Node::Piecewise {
            branches: branches.clone(),
            default: *default,
        },
        // Predicates have no algebraic derivative.
        ExprData::Predicate { .. } => Node::Const,
        ExprData::Forall { .. } | ExprData::Exists { .. } => Node::Const,
        ExprData::BigO(_) => Node::Const,
        ExprData::RootSum { poly, var, body } => Node::RootSum {
            poly: *poly,
            rvar: *var,
            body: *body,
        },
    });

    match node {
        // d/dx x = 1
        Node::IdentVar => {
            let one = pool.integer(1_i32);
            memo.insert(expr, one);
            Ok(DerivedExpr::with_step(
                one,
                RewriteStep::simple("diff_identity", expr, one),
            ))
        }
        // d/dx c = 0  (any atom that is not the target variable)
        Node::Const => {
            let zero = pool.integer(0_i32);
            memo.insert(expr, zero);
            Ok(DerivedExpr::with_step(
                zero,
                RewriteStep::simple("diff_const", expr, zero),
            ))
        }
        // Sum rule: d/dx (f₁ + f₂ + …) = f₁' + f₂' + …
        Node::Add(args) => {
            let mut log = DerivationLog::new();
            let mut dargs: Vec<ExprId> = Vec::with_capacity(args.len());
            for a in args {
                let da = diff_raw(a, var, pool, memo)?;
                log = log.merge(da.log);
                dargs.push(da.value);
            }
            let sum = pool.add(dargs);
            log.push(RewriteStep::simple("sum_rule", expr, sum));
            let result = DerivedExpr::with_log(sum, log);
            memo.insert(expr, result.value);
            Ok(result)
        }
        // Product rule (n-ary Leibniz): d/dx (∏ᵢ fᵢ) = Σᵢ (fᵢ' · ∏_{j≠i} fⱼ)
        Node::Mul(args) => {
            let mut log = DerivationLog::new();
            let dargs: Vec<DerivedExpr<ExprId>> = args
                .iter()
                .map(|&a| diff_raw(a, var, pool, memo))
                .collect::<Result<_, _>>()?;
            for da in &dargs {
                log = log.merge(da.log.clone());
            }
            let mut terms: Vec<ExprId> = Vec::with_capacity(args.len());
            for (i, da) in dargs.iter().enumerate() {
                let di = da.value;
                let rest: Vec<ExprId> = args
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &a)| a)
                    .collect();
                let term = if rest.is_empty() {
                    di
                } else if rest.len() == 1 {
                    pool.mul(vec![di, rest[0]])
                } else {
                    let prod = pool.mul(rest);
                    pool.mul(vec![di, prod])
                };
                terms.push(term);
            }
            let result_id = match terms.len() {
                0 => pool.integer(0_i32),
                1 => terms[0],
                _ => pool.add(terms),
            };
            log.push(RewriteStep::simple("product_rule", expr, result_id));
            let result = DerivedExpr::with_log(result_id, log);
            memo.insert(expr, result.value);
            Ok(result)
        }
        // Power rule, constant exponent (integer or rational):
        //   d/dx f^r = r · f^(r-1) · f'.
        // A var-dependent / non-constant exponent (e.g. x^y, x^x) is a different
        // rule (logarithmic differentiation) and remains unsupported.
        Node::Pow { base, exp } => {
            // Read the exponent without holding the pool lock during recursion.
            let r = pool
                .with(exp, |data| match data {
                    ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
                    ExprData::Rational(q) => Some(q.0.clone()),
                    _ => None,
                })
                .ok_or(DiffError::NonIntegerExponent)?;

            // Special case r=0: d/dx f^0 = 0
            if r == 0 {
                let zero = pool.integer(0_i32);
                let mut log = DerivationLog::new();
                log.push(RewriteStep::simple("power_rule_n0", expr, zero));
                memo.insert(expr, zero);
                return Ok(DerivedExpr::with_log(zero, log));
            }
            // Special case r=1: d/dx f^1 = f'
            if r == 1 {
                let mut result = diff_raw(base, var, pool, memo)?;
                result
                    .log
                    .push(RewriteStep::simple("power_rule_n1", expr, result.value));
                memo.insert(expr, result.value);
                return Ok(result);
            }

            let mut log = DerivationLog::new();
            let df = diff_raw(base, var, pool, memo)?;
            log = log.merge(df.log);
            let r_id = const_node(pool, r.clone());
            let r_minus_1 = const_node(pool, r - 1);
            let base_pow = pool.pow(base, r_minus_1);
            let result_id = pool.mul(vec![r_id, base_pow, df.value]);
            log.push(RewriteStep::simple("power_rule", expr, result_id));
            memo.insert(expr, result_id);
            Ok(DerivedExpr::with_log(result_id, log))
        }
        // Chain rules for single-argument named functions
        Node::Func { name, args } if args.len() == 1 => {
            let f = args[0];
            let mut log = DerivationLog::new();
            let df = diff_raw(f, var, pool, memo)?;
            log = log.merge(df.log);
            let result = match name.as_str() {
                "sin" => {
                    let cos_f = pool.func("cos", vec![f]);
                    let r = pool.mul(vec![cos_f, df.value]);
                    log.push(RewriteStep::simple("diff_sin", expr, r));
                    r
                }
                "cos" => {
                    let sin_f = pool.func("sin", vec![f]);
                    let neg_one = pool.integer(-1_i32);
                    let r = pool.mul(vec![neg_one, sin_f, df.value]);
                    log.push(RewriteStep::simple("diff_cos", expr, r));
                    r
                }
                "exp" => {
                    let exp_f = pool.func("exp", vec![f]);
                    let r = pool.mul(vec![exp_f, df.value]);
                    log.push(RewriteStep::simple("diff_exp", expr, r));
                    r
                }
                "log" => {
                    let f_inv = pool.pow(f, pool.integer(-1_i32));
                    let r = pool.mul(vec![df.value, f_inv]);
                    log.push(RewriteStep::simple("diff_log", expr, r));
                    r
                }
                "sqrt" => {
                    let sqrt_f = pool.func("sqrt", vec![f]);
                    let two_sqrt = pool.mul(vec![pool.integer(2_i32), sqrt_f]);
                    let denom_inv = pool.pow(two_sqrt, pool.integer(-1_i32));
                    let r = pool.mul(vec![df.value, denom_inv]);
                    log.push(RewriteStep::simple("diff_sqrt", expr, r));
                    r
                }
                other => {
                    // Fall back to PrimitiveRegistry for V1-12 primitives
                    let reg = crate::primitive::PrimitiveRegistry::default_registry();
                    if let Some(d) = reg.diff_forward(other, &[f], var, pool) {
                        log.push(RewriteStep::simple("diff_primitive_registry", expr, d));
                        d
                    } else {
                        return Err(DiffError::UnknownFunction(other.to_string()));
                    }
                }
            };
            memo.insert(expr, result);
            Ok(DerivedExpr::with_log(result, log))
        }
        Node::Func { name, .. } => Err(DiffError::UnknownFunction(name)),
        // PA-9: Piecewise diff distributes into branches.
        // d/dx Piecewise([(c₁,v₁), …], d) = Piecewise([(c₁, d/dx v₁), …], d/dx d)
        Node::Piecewise { branches, default } => {
            let mut log = DerivationLog::new();
            let mut new_branches = Vec::with_capacity(branches.len());
            for (cond, val) in branches {
                let dval = diff_raw(val, var, pool, memo)?;
                log = log.merge(dval.log);
                new_branches.push((cond, dval.value));
            }
            let ddefault = diff_raw(default, var, pool, memo)?;
            log = log.merge(ddefault.log);
            let result = pool.piecewise(new_branches, ddefault.value);
            log.push(RewriteStep::simple("diff_piecewise", expr, result));
            memo.insert(expr, result);
            Ok(DerivedExpr::with_log(result, log))
        }
        // d/dx Σ_{c:P(c)=0} body(c,x) = Σ_{c:P(c)=0} ∂body/∂x.
        // The root `c` (rvar) is constant in `x`; `poly` is free of `x`.
        Node::RootSum { poly, rvar, body } => {
            let dbody = diff_raw(body, var, pool, memo)?;
            let result = pool.root_sum(poly, rvar, dbody.value);
            let mut log = dbody.log;
            log.push(RewriteStep::simple("diff_root_sum", expr, result));
            memo.insert(expr, result);
            Ok(DerivedExpr::with_log(result, log))
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::poly::UniPoly;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn diff_constant() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.integer(5_i32), x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_const"));
    }

    #[test]
    fn diff_identity() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(x, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_identity"));
    }

    #[test]
    fn diff_other_variable() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let r = diff(y, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn diff_linear() {
        // d/dx (3x) = 3
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.integer(3_i32), x]);
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(3_i32));
    }

    #[test]
    fn diff_quadratic() {
        // d/dx x² = 2x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.pow(x, pool.integer(2_i32)), x, &pool).unwrap();
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![0, 2]);
    }

    #[test]
    fn diff_cubic() {
        // d/dx x³ = 3x²
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.pow(x, pool.integer(3_i32)), x, &pool).unwrap();
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![0, 0, 3]);
    }

    #[test]
    fn diff_polynomial() {
        // d/dx (x³ + 2x² + x + 1) = 3x² + 4x + 1
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(2_i32), pool.pow(x, pool.integer(2_i32))]),
            x,
            pool.integer(1_i32),
        ]);
        let r = diff(expr, x, &pool).unwrap();
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![1, 4, 3]);
    }

    #[test]
    fn diff_sum_rule_logged() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let r = diff(pool.add(vec![x, y]), x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "sum_rule"));
    }

    #[test]
    fn diff_product_rule_logged() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let r = diff(pool.mul(vec![x, y]), x, &pool).unwrap();
        assert_eq!(r.value, y);
        assert!(r.log.steps().iter().any(|s| s.rule_name == "product_rule"));
    }

    #[test]
    fn diff_sin() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.func("sin", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.func("cos", vec![x]));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_sin"));
    }

    #[test]
    fn diff_cos() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.func("cos", vec![x]), x, &pool).unwrap();
        // d/dx cos(x) = -sin(x) = Mul([-1, sin(x)]) in canonical arg order
        let sin_x = pool.func("sin", vec![x]);
        let neg_one = pool.integer(-1_i32);
        match pool.get(r.value) {
            ExprData::Mul(ref args) => {
                assert_eq!(args.len(), 2);
                assert!(args.contains(&neg_one) && args.contains(&sin_x));
            }
            _ => panic!("expected Mul, got {:?}", pool.display(r.value)),
        }
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_cos"));
    }

    #[test]
    fn diff_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let r = diff(exp_x, x, &pool).unwrap();
        assert_eq!(r.value, exp_x);
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_exp"));
    }

    #[test]
    fn diff_log() {
        // d/dx log(x) = x^(-1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.func("log", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.pow(x, pool.integer(-1_i32)));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_log"));
    }

    #[test]
    fn diff_chain_rule_sin() {
        // d/dx sin(x²): cos inner uses ℤ-polynomial fast-path for x² → 2x (not the granular power_rule).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(
            pool.func("sin", vec![pool.pow(x, pool.integer(2_i32))]),
            x,
            &pool,
        )
        .unwrap();
        assert_ne!(r.value, pool.integer(0_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_sin"));
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "diff_univariate_poly"));
    }

    #[test]
    fn diff_pow_n0() {
        // d/dx f^0 = 0 — x^0 ≅ 1 is read as a ℤ-poly constant, so the dense derivative path applies.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(0_i32));
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "diff_univariate_poly"));
    }

    #[test]
    fn diff_pow_n1() {
        // d/dx x^1 — same fast-path as other pure ℤ-polynomials.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(1_i32));
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "diff_univariate_poly"));
    }

    #[test]
    fn diff_unknown_function_error() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let err = diff(pool.func("zeta", vec![x]), x, &pool);
        assert!(matches!(err, Err(DiffError::UnknownFunction(_))));
    }

    #[test]
    fn diff_non_integer_exponent_error() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // A *var-dependent* exponent still needs logarithmic differentiation and
        // remains unsupported.
        let err = diff(pool.pow(x, y), x, &pool);
        assert!(matches!(err, Err(DiffError::NonIntegerExponent)));
    }

    #[test]
    fn diff_fractional_power() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // d/dx x^{1/2} = (1/2) x^{-1/2}.
        let half = pool.pow(x, pool.rational(1_i32, 2_i32));
        let d = diff(half, x, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.rational(1_i32, 2_i32),
            pool.pow(x, pool.rational(-1_i32, 2_i32)),
        ]);
        assert_eq!(
            simplify(d.value, &pool).value,
            simplify(expected, &pool).value
        );

        // d/dx x^{2/3} = (2/3) x^{-1/3}.
        let two_thirds = pool.pow(x, pool.rational(2_i32, 3_i32));
        let d = diff(two_thirds, x, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.rational(2_i32, 3_i32),
            pool.pow(x, pool.rational(-1_i32, 3_i32)),
        ]);
        assert_eq!(
            simplify(d.value, &pool).value,
            simplify(expected, &pool).value
        );
    }

    #[test]
    fn diff_fractional_power_chain_rule() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // d/dx (x²+1)^{3/2} = (3/2)(x²+1)^{1/2}·2x = 3x·(x²+1)^{1/2}.
        let base = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let expr = pool.pow(base, pool.rational(3_i32, 2_i32));
        let d = diff(expr, x, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.integer(3_i32),
            x,
            pool.pow(base, pool.rational(1_i32, 2_i32)),
        ]);
        assert_eq!(
            simplify(d.value, &pool).value,
            simplify(expected, &pool).value
        );
    }

    #[test]
    fn diff_balanced_geom_series_univariate_fastpath() {
        fn balanced_sum(pool: &ExprPool, terms: &[ExprId]) -> ExprId {
            match terms.len() {
                0 => pool.integer(0_i32),
                1 => terms[0],
                _ => {
                    let mid = terms.len() / 2;
                    pool.add(vec![
                        balanced_sum(pool, &terms[..mid]),
                        balanced_sum(pool, &terms[mid..]),
                    ])
                }
            }
        }
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let n = 80i32;
        let mut terms = vec![pool.integer(1_i32)];
        for k in 1..=n {
            terms.push(pool.pow(x, pool.integer(k)));
        }
        let expr = balanced_sum(&pool, &terms);
        let r = diff(expr, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "diff_univariate_poly"),
            "expected dense ℤ-poly fast-path for balanced sum"
        );
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.degree(), i64::from(n) - 1);
        let coeffs = poly.coefficients_i64();
        assert_eq!(coeffs.first().copied(), Some(1));
        assert_eq!(coeffs.last().copied(), Some(n as i64));
    }

    #[test]
    fn diff_log_has_both_diff_and_simplify_steps() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            y,
            pool.integer(0_i32),
        ]);
        let r = diff(expr, x, &pool).unwrap();
        let rules: Vec<&str> = r.log.steps().iter().map(|s| s.rule_name).collect();
        assert!(
            rules.contains(&"sum_rule"),
            "should have sum_rule: {rules:?}"
        );
        assert!(
            rules.contains(&"diff_univariate_poly"),
            "x² term differentiates via ℤ-polynomial fast-path: {rules:?}"
        );
        assert!(rules.len() > 1, "log should have multiple steps: {rules:?}");
    }
}
