use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::fmt;

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
    let result = diff_raw(expr, var, pool)?;
    Ok(result.and_then(|v| simplify(v, pool)))
}

// ---------------------------------------------------------------------------
// Core recursive differentiation (no simplification)
// ---------------------------------------------------------------------------

fn diff_raw(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, DiffError> {
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
    });

    match node {
        // d/dx x = 1
        Node::IdentVar => {
            let one = pool.integer(1_i32);
            Ok(DerivedExpr::with_step(
                one,
                RewriteStep::simple("diff_identity", expr, one),
            ))
        }
        // d/dx c = 0  (any atom that is not the target variable)
        Node::Const => {
            let zero = pool.integer(0_i32);
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
                let da = diff_raw(a, var, pool)?;
                log = log.merge(da.log);
                dargs.push(da.value);
            }
            let sum = pool.add(dargs);
            log.push(RewriteStep::simple("sum_rule", expr, sum));
            Ok(DerivedExpr::with_log(sum, log))
        }
        // Product rule (n-ary Leibniz): d/dx (∏ᵢ fᵢ) = Σᵢ (fᵢ' · ∏_{j≠i} fⱼ)
        Node::Mul(args) => {
            let mut log = DerivationLog::new();
            let dargs: Vec<DerivedExpr<ExprId>> = args
                .iter()
                .map(|&a| diff_raw(a, var, pool))
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
            let result = match terms.len() {
                0 => pool.integer(0_i32),
                1 => terms[0],
                _ => pool.add(terms),
            };
            log.push(RewriteStep::simple("product_rule", expr, result));
            Ok(DerivedExpr::with_log(result, log))
        }
        // Power rule (integer exponent): d/dx f^n = n · f^(n-1) · f'
        Node::Pow { base, exp } => {
            // Read the exponent without holding the pool lock during recursion.
            let n = pool
                .with(exp, |data| match data {
                    ExprData::Integer(n) => Some(n.0.clone()),
                    _ => None,
                })
                .ok_or(DiffError::NonIntegerExponent)?;

            // Special case n=0: d/dx f^0 = 0
            if n == 0 {
                let zero = pool.integer(0_i32);
                let mut log = DerivationLog::new();
                log.push(RewriteStep::simple("power_rule_n0", expr, zero));
                return Ok(DerivedExpr::with_log(zero, log));
            }
            // Special case n=1: d/dx f^1 = f'
            if n == 1 {
                let mut result = diff_raw(base, var, pool)?;
                result
                    .log
                    .push(RewriteStep::simple("power_rule_n1", expr, result.value));
                return Ok(result);
            }

            let mut log = DerivationLog::new();
            let df = diff_raw(base, var, pool)?;
            log = log.merge(df.log);
            let n_id = pool.integer(n.clone());
            let n_minus_1 = pool.integer(n - 1);
            let base_pow = pool.pow(base, n_minus_1);
            let result = pool.mul(vec![n_id, base_pow, df.value]);
            log.push(RewriteStep::simple("power_rule", expr, result));
            Ok(DerivedExpr::with_log(result, log))
        }
        // Chain rules for single-argument named functions
        Node::Func { name, args } if args.len() == 1 => {
            let f = args[0];
            let mut log = DerivationLog::new();
            let df = diff_raw(f, var, pool)?;
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
            Ok(DerivedExpr::with_log(result, log))
        }
        Node::Func { name, .. } => Err(DiffError::UnknownFunction(name)),
        // PA-9: Piecewise diff distributes into branches.
        // d/dx Piecewise([(c₁,v₁), …], d) = Piecewise([(c₁, d/dx v₁), …], d/dx d)
        Node::Piecewise { branches, default } => {
            let mut log = DerivationLog::new();
            let mut new_branches = Vec::with_capacity(branches.len());
            for (cond, val) in branches {
                let dval = diff_raw(val, var, pool)?;
                log = log.merge(dval.log);
                new_branches.push((cond, dval.value));
            }
            let ddefault = diff_raw(default, var, pool)?;
            log = log.merge(ddefault.log);
            let result = pool.piecewise(new_branches, ddefault.value);
            log.push(RewriteStep::simple("diff_piecewise", expr, result));
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
        let r = diff(pool.mul(vec![x, x]), x, &pool).unwrap();
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![0, 2]);
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
        // d/dx sin(x²): should involve cos and power_rule
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
        assert!(r.log.steps().iter().any(|s| s.rule_name == "power_rule"));
    }

    #[test]
    fn diff_pow_n0() {
        // d/dx f^0 = 0 (regardless of f)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(0_i32));
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "power_rule_n0"));
    }

    #[test]
    fn diff_pow_n1() {
        // d/dx f^1 = f' = 1 for f=x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(1_i32));
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "power_rule_n1"));
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
        let err = diff(pool.pow(x, y), x, &pool);
        assert!(matches!(err, Err(DiffError::NonIntegerExponent)));
    }

    #[test]
    fn diff_log_has_both_diff_and_simplify_steps() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(0_i32)]);
        let r = diff(expr, x, &pool).unwrap();
        let rules: Vec<&str> = r.log.steps().iter().map(|s| s.rule_name).collect();
        assert!(
            rules.contains(&"power_rule"),
            "should have power_rule: {rules:?}"
        );
        assert!(rules.len() > 1, "log should have multiple steps: {rules:?}");
    }
}
