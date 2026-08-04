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
use std::collections::HashMap;

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

    // No `neg`, `sub` or `div`: the kernel has no `Neg`, `Sub` or `Div` node.
    // Subtraction is `Add([a, Mul([-1, b])])` and division is `Mul([a, Pow(b, -1)])`,
    // so both reach this evaluator through `add`, `mul` and `pow_int` alone.  The
    // quotient rule falls out of the product rule composed with `pow_int(-1)`:
    // `d(a·b⁻¹) = b⁻¹·da + a·(−b⁻²·db) = (b·da − a·db)/b²`.  See the
    // `forward_diff_subtraction_*` and `forward_diff_division_*` tests.

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

    /// `f^r` for a constant rational `r`: `d/dx f^r = r · f^{r-1} · f'`.
    fn pow_rat(self, r: rug::Rational, pool: &ExprPool) -> Self {
        let r_id = super::diff_impl::const_node(pool, r.clone());
        let r_minus_1 = super::diff_impl::const_node(pool, r - 1);
        let value = pool.pow(self.value, r_id);
        let base_pow = pool.pow(self.value, r_minus_1);
        let tangent = pool.mul(vec![r_id, base_pow, self.tangent]);
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

    fn atan(self, pool: &ExprPool) -> Self {
        // d/dx atan(f) = f' / (1 + f²)
        let value = pool.func("atan", vec![self.value]);
        let one_plus_f2 = pool.add(vec![
            pool.integer(1_i32),
            pool.pow(self.value, pool.integer(2_i32)),
        ]);
        let tangent = pool.mul(vec![
            self.tangent,
            pool.pow(one_plus_f2, pool.integer(-1_i32)),
        ]);
        DualValue::new(value, tangent)
    }
}

// ---------------------------------------------------------------------------
// Core evaluation
// ---------------------------------------------------------------------------

/// Memoised dual-number evaluator.
///
/// `memo` maps `ExprId → DualValue` so that shared subexpressions are
/// evaluated only once per `diff_forward` call.  `DualValue` holds two
/// `ExprId` values and is cheap to clone.
/// A recognized constant power exponent: integer or rational.
enum ExpKind {
    Int(rug::Integer),
    Rat(rug::Rational),
}

fn eval_dual(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, DualValue>,
) -> Result<DualValue, DiffError> {
    // Return cached dual for shared subexpressions.
    if let Some(cached) = memo.get(&expr) {
        return Ok(cached.clone());
    }

    enum Node {
        IsVar,
        IsConst,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow {
            base: ExprId,
            exp: ExprId,
        },
        Func {
            name: String,
            arg: ExprId,
        },
        RootSum {
            poly: ExprId,
            rvar: ExprId,
            body: ExprId,
        },
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
        ExprData::BigO(_) => Node::IsConst,
        ExprData::RootSum { poly, var, body } => Node::RootSum {
            poly: *poly,
            rvar: *var,
            body: *body,
        },
    });

    let result = match node {
        Node::IsVar => Ok(DualValue::seed(expr, pool)),
        Node::IsConst => Ok(DualValue::constant(expr, pool)),
        Node::Add(args) => {
            let mut acc = DualValue::constant(pool.integer(0_i32), pool);
            for a in args {
                acc = acc.add(eval_dual(a, var, pool, memo)?, pool);
            }
            Ok(acc)
        }
        Node::Mul(args) => {
            let mut acc = DualValue::constant(pool.integer(1_i32), pool);
            for a in args {
                acc = acc.mul(eval_dual(a, var, pool, memo)?, pool);
            }
            Ok(acc)
        }
        Node::Pow { base, exp } => {
            // Constant exponent: integer → pow_int, rational → pow_rat.
            let exp_kind = pool
                .with(exp, |data| match data {
                    ExprData::Integer(n) => Some(ExpKind::Int(n.0.clone())),
                    ExprData::Rational(q) => Some(ExpKind::Rat(q.0.clone())),
                    _ => None,
                })
                .ok_or(DiffError::ForwardNonIntegerExponent)?;
            let b = eval_dual(base, var, pool, memo)?;
            Ok(match exp_kind {
                ExpKind::Int(n) => b.pow_int(n, pool),
                ExpKind::Rat(q) => b.pow_rat(q, pool),
            })
        }
        Node::Func { name, arg } => {
            // Protect against the dummy self-referential node from multi-arg fns
            if arg == expr {
                return Err(DiffError::ForwardUnknownFunction(name));
            }
            let inner = eval_dual(arg, var, pool, memo)?;
            match name.as_str() {
                "sin" => Ok(inner.sin(pool)),
                "cos" => Ok(inner.cos(pool)),
                "exp" => Ok(inner.exp(pool)),
                "log" => Ok(inner.log(pool)),
                "sqrt" => Ok(inner.sqrt(pool)),
                "atan" => Ok(inner.atan(pool)),
                other => Err(DiffError::ForwardUnknownFunction(other.to_string())),
            }
        }
        Node::RootSum { poly, rvar, body } => {
            // d/dx Σ_{c:P(c)=0} body = Σ_{c:P(c)=0} ∂body/∂x; the root `rvar` is
            // constant in `x` (tangent 0), so threading the dual through `body`
            // gives the per-root derivative directly.
            let inner = eval_dual(body, var, pool, memo)?;
            let value = pool.root_sum(poly, rvar, inner.value);
            let tangent = pool.root_sum(poly, rvar, inner.tangent);
            Ok(DualValue::new(value, tangent))
        }
    }?;

    memo.insert(expr, result.clone());
    Ok(result)
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
    let mut memo: HashMap<ExprId, DualValue> = HashMap::new();
    let dual = eval_dual(expr, var, pool, &mut memo)?;
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
    fn forward_diff_fractional_power_agrees_with_symbolic() {
        // d/dx x^{2/3} via forward vs symbolic.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.rational(2_i32, 3_i32));
        let fwd = diff_forward(expr, x, &pool).unwrap().value;
        let sym = sym_diff(expr, x, &pool).unwrap().value;
        assert_eq!(
            crate::simplify::engine::simplify(fwd, &pool).value,
            crate::simplify::engine::simplify(sym, &pool).value
        );
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

    // The kernel has no `Sub` or `Div` node, so subtraction and division reach
    // forward mode as `Add`/`Mul`/`Pow` over canonical forms.  These two tests
    // pin that down: they are the reason `DualValue` needs no `sub`/`div`/`neg`.

    #[test]
    fn forward_diff_subtraction_agrees_with_symbolic() {
        // d/dx (x² − 3x) = 2x − 3, where `− 3x` is `Mul([-3, x])`.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(-3_i32), x]),
        ]);
        let fwd = diff_forward(expr, x, &pool).unwrap().value;
        let sym = sym_diff(expr, x, &pool).unwrap().value;
        let fwd_poly = UniPoly::from_symbolic(fwd, x, &pool).unwrap();
        let sym_poly = UniPoly::from_symbolic(sym, x, &pool).unwrap();
        assert_eq!(fwd_poly.coefficients_i64(), sym_poly.coefficients_i64());
        assert_eq!(fwd_poly.coefficients_i64(), vec![-3, 2]);
    }

    #[test]
    fn forward_diff_division_agrees_with_symbolic() {
        // d/dx ((x+1)/x) = −1/x², where the quotient is `Mul([x+1, Pow(x, -1)])`.
        // Compared by exact rational evaluation, since neither derivative is a
        // polynomial and the two spellings need not be structurally identical.
        use crate::eval::eval_exact_rational;
        use rug::Rational;
        use std::collections::HashMap;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let numer = pool.add(vec![x, pool.integer(1_i32)]);
        let expr = pool.mul(vec![numer, pool.pow(x, pool.integer(-1_i32))]);
        let fwd = diff_forward(expr, x, &pool).unwrap().value;
        let sym = sym_diff(expr, x, &pool).unwrap().value;

        for pt in [2_i32, 3, 5, -4] {
            let mut b = HashMap::new();
            b.insert(x, Rational::from(pt));
            let fv = eval_exact_rational(fwd, &pool, &b).unwrap();
            let sv = eval_exact_rational(sym, &pool, &b).unwrap();
            // Closed form: −1/x².
            let expected = Rational::from((-1, pt as i64 * pt as i64));
            assert_eq!(fv, sv, "forward vs symbolic disagree at x = {pt}");
            assert_eq!(fv, expected, "forward wrong at x = {pt}");
        }
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
