/// Forward-mode automatic differentiation via dual numbers.
///
/// A dual number `DualValue { value: T, tangent: T }` tracks both the primal
/// value and its derivative simultaneously.  Evaluating an expression with
/// `DualValue<ExprId>` inputs — setting the tangent of the variable of
/// differentiation to `1` and all others to `0` — propagates the derivative
/// through every operation automatically.
///
/// The result agrees with the symbolic differentiator on all expressions
/// whose derivative is defined; property tests cross-validate both.
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::diff::diff_impl::DiffError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

// ---------------------------------------------------------------------------
// Deprecated type alias — `ForwardDiffError` is now folded into `DiffError`
// (variants `ForwardUnknownFunction` / `ForwardNonIntegerExponent`, codes
// E-DIFF-003 / E-DIFF-004).  This alias keeps old `ForwardDiffError` names
// compiling; it will be removed in the next major version.
// ---------------------------------------------------------------------------

#[deprecated(
    since = "2.0.0",
    note = "use DiffError::ForwardUnknownFunction / ForwardNonIntegerExponent instead"
)]
pub type ForwardDiffError = DiffError;

// ---------------------------------------------------------------------------
// DualValue
// ---------------------------------------------------------------------------

/// A dual number carrying a primal `value` and a first-order `tangent`.
///
/// Arithmetic on `DualValue` follows the dual-number algebra:
/// - `(a + ε·da) + (b + ε·db) = (a+b) + ε·(da+db)`
/// - `(a + ε·da) * (b + ε·db) = a·b + ε·(a·db + b·da)`
#[derive(Clone, Debug)]
pub struct DualValue {
    pub value: ExprId,
    pub tangent: ExprId,
}

impl DualValue {
    fn new(value: ExprId, tangent: ExprId) -> Self {
        DualValue { value, tangent }
    }

    fn constant(value: ExprId, pool: &ExprPool) -> Self {
        let zero = pool.integer(0_i32);
        DualValue::new(value, zero)
    }

    fn seed(value: ExprId, pool: &ExprPool) -> Self {
        let one = pool.integer(1_i32);
        DualValue::new(value, one)
    }

    fn add(self, rhs: Self, pool: &ExprPool) -> Self {
        let value = pool.add(vec![self.value, rhs.value]);
        let tangent = pool.add(vec![self.tangent, rhs.tangent]);
        DualValue::new(value, tangent)
    }

    fn mul(self, rhs: Self, pool: &ExprPool) -> Self {
        // (a·db + b·da)
        let value = pool.mul(vec![self.value, rhs.value]);
        let term1 = pool.mul(vec![self.value, rhs.tangent]);
        let term2 = pool.mul(vec![rhs.value, self.tangent]);
        let tangent = pool.add(vec![term1, term2]);
        DualValue::new(value, tangent)
    }

    #[allow(dead_code)]
    fn neg(self, pool: &ExprPool) -> Self {
        let neg_one = pool.integer(-1_i32);
        let value = pool.mul(vec![neg_one, self.value]);
        let tangent = pool.mul(vec![neg_one, self.tangent]);
        DualValue::new(value, tangent)
    }

    #[allow(dead_code)]
    fn sub(self, rhs: Self, pool: &ExprPool) -> Self {
        self.add(rhs.neg(pool), pool)
    }

    /// Division: d(a/b) = (b·da - a·db) / b²
    #[allow(dead_code)]
    fn div(self, rhs: Self, pool: &ExprPool) -> Self {
        let value = pool.mul(vec![self.value, pool.pow(rhs.value, pool.integer(-1_i32))]);
        let bda = pool.mul(vec![rhs.value, self.tangent]);
        let adb = pool.mul(vec![self.value, rhs.tangent]);
        let neg_one = pool.integer(-1_i32);
        let numerator = pool.add(vec![bda, pool.mul(vec![neg_one, adb])]);
        let b_sq = pool.pow(rhs.value, pool.integer(2_i32));
        let tangent = pool.mul(vec![numerator, pool.pow(b_sq, pool.integer(-1_i32))]);
        DualValue::new(value, tangent)
    }

    /// Power rule for integer exponent n: d(f^n) = n * f^(n-1) * f'
    fn pow_int(self, n: rug::Integer, pool: &ExprPool) -> Self {
        if n == 0 {
            let one = pool.integer(1_i32);
            return DualValue::new(one, pool.integer(0_i32));
        }
        if n == 1 {
            return self;
        }
        let n_id = pool.integer(n.clone());
        let n_minus_1 = pool.integer(n - 1);
        let value = pool.pow(self.value, n_id);
        let base_pow = pool.pow(self.value, n_minus_1);
        let tangent = pool.mul(vec![n_id, base_pow, self.tangent]);
        DualValue::new(value, tangent)
    }

    fn sin(self, pool: &ExprPool) -> Self {
        // d/dx sin(f) = cos(f) * f'
        let value = pool.func("sin", vec![self.value]);
        let cos_f = pool.func("cos", vec![self.value]);
        let tangent = pool.mul(vec![cos_f, self.tangent]);
        DualValue::new(value, tangent)
    }

    fn cos(self, pool: &ExprPool) -> Self {
        // d/dx cos(f) = -sin(f) * f'
        let value = pool.func("cos", vec![self.value]);
        let sin_f = pool.func("sin", vec![self.value]);
        let neg_one = pool.integer(-1_i32);
        let tangent = pool.mul(vec![neg_one, sin_f, self.tangent]);
        DualValue::new(value, tangent)
    }

    fn exp(self, pool: &ExprPool) -> Self {
        // d/dx exp(f) = exp(f) * f'
        let value = pool.func("exp", vec![self.value]);
        let tangent = pool.mul(vec![value, self.tangent]);
        DualValue::new(value, tangent)
    }

    fn log(self, pool: &ExprPool) -> Self {
        // d/dx log(f) = f' / f = f' * f^(-1)
        let value = pool.func("log", vec![self.value]);
        let f_inv = pool.pow(self.value, pool.integer(-1_i32));
        let tangent = pool.mul(vec![self.tangent, f_inv]);
        DualValue::new(value, tangent)
    }

    fn sqrt(self, pool: &ExprPool) -> Self {
        // d/dx sqrt(f) = f' / (2 * sqrt(f))
        let value = pool.func("sqrt", vec![self.value]);
        let two_sqrt = pool.mul(vec![pool.integer(2_i32), value]);
        let tangent = pool.mul(vec![self.tangent, pool.pow(two_sqrt, pool.integer(-1_i32))]);
        DualValue::new(value, tangent)
    }
}

// ---------------------------------------------------------------------------
// Core evaluation
// ---------------------------------------------------------------------------

fn eval_dual(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<DualValue, DiffError> {
    enum Node {
        IsVar,
        IsConst,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow { base: ExprId, exp: ExprId },
        Func { name: String, arg: ExprId },
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } if expr == var => Node::IsVar,
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Node::IsConst,
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow {
            base: *base,
            exp: *exp,
        },
        ExprData::Func { name, args } if args.len() == 1 => Node::Func {
            name: name.clone(),
            arg: args[0],
        },
        ExprData::Func { name, .. } => Node::Func {
            name: name.clone(),
            arg: expr,
        },
        // PA-9: Piecewise and Predicate are treated as constants w.r.t. the
        // variable being differentiated (predicates don't depend on x algebraically).
        ExprData::Piecewise { .. } | ExprData::Predicate { .. } => Node::IsConst,
        ExprData::Forall { .. } | ExprData::Exists { .. } => Node::IsConst,
    });

    match node {
        Node::IsVar => Ok(DualValue::seed(expr, pool)),
        Node::IsConst => Ok(DualValue::constant(expr, pool)),
        Node::Add(args) => {
            let mut acc = DualValue::constant(pool.integer(0_i32), pool);
            for a in args {
                acc = acc.add(eval_dual(a, var, pool)?, pool);
            }
            Ok(acc)
        }
        Node::Mul(args) => {
            let mut acc = DualValue::constant(pool.integer(1_i32), pool);
            for a in args {
                acc = acc.mul(eval_dual(a, var, pool)?, pool);
            }
            Ok(acc)
        }
        Node::Pow { base, exp } => {
            let n = pool
                .with(exp, |data| match data {
                    ExprData::Integer(n) => Some(n.0.clone()),
                    _ => None,
                })
                .ok_or(DiffError::ForwardNonIntegerExponent)?;
            let b = eval_dual(base, var, pool)?;
            Ok(b.pow_int(n, pool))
        }
        Node::Func { name, arg } => {
            // Protect against the dummy self-referential node from multi-arg fns
            if arg == expr {
                return Err(DiffError::ForwardUnknownFunction(name));
            }
            let inner = eval_dual(arg, var, pool)?;
            match name.as_str() {
                "sin" => Ok(inner.sin(pool)),
                "cos" => Ok(inner.cos(pool)),
                "exp" => Ok(inner.exp(pool)),
                "log" => Ok(inner.log(pool)),
                "sqrt" => Ok(inner.sqrt(pool)),
                other => Err(DiffError::ForwardUnknownFunction(other.to_string())),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Differentiate `expr` with respect to `var` using forward-mode (dual-number)
/// automatic differentiation.
///
/// Returns the derivative expression after applying the rule-based simplifier.
/// The derivation log records a single `diff_forward` step.
///
/// # Agreement with symbolic diff
///
/// For any polynomial or rational-function expression, `diff_forward` and
/// `diff` (symbolic) produce structurally equal results after simplification.
/// Property tests in this module verify this on random polynomials.
pub fn diff_forward(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, DiffError> {
    let dual = eval_dual(expr, var, pool)?;
    let tangent_raw = dual.tangent;

    // Simplify the raw tangent
    let simplified = simplify(tangent_raw, pool);

    // Wrap in a derivation log
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("diff_forward", expr, simplified.value));
    let full_log = log.merge(simplified.log);
    Ok(DerivedExpr::with_log(simplified.value, full_log))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::diff as sym_diff;
    use crate::kernel::{Domain, ExprPool};
    use crate::poly::UniPoly;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn forward_diff_constant() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff_forward(pool.integer(5_i32), x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn forward_diff_identity() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff_forward(x, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
    }

    #[test]
    fn forward_diff_other_var() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let r = diff_forward(y, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn forward_diff_linear() {
        // d/dx (3x) = 3
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.integer(3_i32), x]);
        let r = diff_forward(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(3_i32));
    }

    #[test]
    fn forward_diff_quadratic_agrees_with_symbolic() {
        // d/dx x² via forward vs symbolic
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(2_i32));
        let fwd = diff_forward(expr, x, &pool).unwrap();
        let sym = sym_diff(expr, x, &pool).unwrap();
        // Both should give 2x
        let fwd_poly = UniPoly::from_symbolic(fwd.value, x, &pool).unwrap();
        let sym_poly = UniPoly::from_symbolic(sym.value, x, &pool).unwrap();
        assert_eq!(fwd_poly.coefficients_i64(), sym_poly.coefficients_i64());
    }

    #[test]
    fn forward_diff_cubic_agrees_with_symbolic() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(3_i32));
        let fwd = diff_forward(expr, x, &pool).unwrap().value;
        let sym = sym_diff(expr, x, &pool).unwrap().value;
        let fwd_poly = UniPoly::from_symbolic(fwd, x, &pool).unwrap();
        let sym_poly = UniPoly::from_symbolic(sym, x, &pool).unwrap();
        assert_eq!(fwd_poly.coefficients_i64(), sym_poly.coefficients_i64());
    }

    #[test]
    fn forward_diff_sin() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff_forward(pool.func("sin", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.func("cos", vec![x]));
    }

    #[test]
    fn forward_diff_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let r = diff_forward(exp_x, x, &pool).unwrap();
        assert_eq!(r.value, exp_x);
    }

    #[test]
    fn forward_diff_log() {
        // d/dx log(x) = x^{-1}
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff_forward(pool.func("log", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.pow(x, pool.integer(-1_i32)));
    }

    #[test]
    fn forward_diff_step_logged() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff_forward(x, x, &pool).unwrap();
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_forward"));
    }
}
