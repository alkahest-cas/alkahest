//! Limits of expressions that contain **undefined functions** (`f(t)`) and
//! their `Derivative` nodes.
//!
//! Nothing is known about an undefined function, so no limit involving one is
//! computed unless the caller *states* the one fact that makes it meaningful:
//! that the node is continuous at the limit point.  [`limit_with_continuous`]
//! then returns the value by substitution — `f(t) → f(a)` — but only when the
//! expression is built from operations that are jointly continuous at that
//! point (sums, products, non-negative integer powers, entire functions
//! `exp`/`sin`/`cos`/`sinh`/`cosh`/`atan`/`tanh`, and negative powers of a
//! factor whose value at the point is a non-zero rational constant).  A `0/0`
//! form such as `(f(t) − f(a))/(t − a)` is the *definition of a derivative*,
//! which continuity does not give, so it is refused rather than guessed.
//!
//! Without the declaration the call is refused (`E-LIMIT-007`).

use crate::calculus::limits::{limit, LimitDirection, LimitError};
use crate::diff::undefined::var_dependent_nodes;
use crate::kernel::subs::subs;
use crate::kernel::{contains_non_finite_atom, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::collections::HashMap;
use std::fmt;

/// Why a limit involving undefined functions was refused (`E-LIMIT-007`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueLimitRefusal {
    /// What was missing or unsupported, with the remedy.
    pub reason: String,
}

impl fmt::Display for OpaqueLimitRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "limit: {}", self.reason)
    }
}

impl std::error::Error for OpaqueLimitRefusal {}

impl crate::errors::AlkahestError for OpaqueLimitRefusal {
    fn code(&self) -> &'static str {
        "E-LIMIT-007"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "an undefined function is opaque: declare the nodes that are continuous at the \
             point with `continuous=[f(t), …]`, and keep the expression jointly continuous \
             there (no 0/0 or ∞−∞ forms); a derivative quotient needs the function's \
             differentiability, which is never assumed",
        )
    }
}

/// Failure of [`limit_with_continuous`].
#[derive(Debug)]
pub enum OpaqueLimitError {
    /// Refused because of an undeclared or unsupported undefined function.
    Refused(OpaqueLimitRefusal),
    /// The ordinary limit engine failed (expression without undefined
    /// functions).
    Limit(LimitError),
}

impl fmt::Display for OpaqueLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpaqueLimitError::Refused(e) => write!(f, "{e}"),
            OpaqueLimitError::Limit(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for OpaqueLimitError {}

fn refused(reason: String) -> OpaqueLimitError {
    OpaqueLimitError::Refused(OpaqueLimitRefusal { reason })
}

/// Entire (everywhere continuous) elementary functions of one argument.
const CONTINUOUS_FUNCS: &[&str] = &["exp", "sin", "cos", "sinh", "cosh", "atan", "tanh", "asinh"];

/// `lim_{var → point} expr`, with `continuous` the undefined-function nodes
/// (`f(t)`, `Derivative(f(t), t)`, …) that the caller asserts are continuous
/// at `point`.
///
/// Expressions with no undefined function of `var` go to [`limit`] unchanged.
///
/// # Errors
///
/// [`OpaqueLimitError::Refused`] (`E-LIMIT-007`) when a node is not declared,
/// the point is not finite, or the expression is not jointly continuous at the
/// point by the structural rule above.
pub fn limit_with_continuous(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    continuous: &[ExprId],
    pool: &ExprPool,
) -> Result<ExprId, OpaqueLimitError> {
    let nodes = var_dependent_nodes(expr, var, pool);
    if nodes.is_empty() {
        return limit(expr, var, point, direction, pool).map_err(OpaqueLimitError::Limit);
    }
    if let Some(&n) = nodes.iter().find(|n| !continuous.contains(n)) {
        return Err(refused(format!(
            "{} is an undefined function (or derivative) of {}: nothing is known about it, so \
             its limit is not assumed. State that it is continuous at the point with \
             continuous=[{}]",
            pool.display(n),
            pool.display(var),
            pool.display(n)
        )));
    }
    if contains_non_finite_atom(pool, point) {
        return Err(refused(
            "an undefined function has no stated behaviour at infinity; only a finite point is \
             supported"
                .to_string(),
        ));
    }
    let mut at = HashMap::new();
    at.insert(var, point);
    // The declared nodes must themselves be defined at the point.
    for &n in &nodes {
        if contains_non_finite_atom(pool, subs(n, &at, pool)) {
            return Err(refused(format!(
                "{} is not defined at {}",
                pool.display(n),
                pool.display(point)
            )));
        }
    }
    if !jointly_continuous(expr, var, point, &nodes, pool) {
        return Err(refused(format!(
            "{} is not provably continuous at {} from continuity of the function(s) alone \
             (a quotient with a vanishing denominator, a non-integer power, a logarithm, or a \
             derivative quotient needs more than continuity); refusing rather than guessing",
            pool.display(expr),
            pool.display(point)
        )));
    }
    let v = simplify(subs(expr, &at, pool), pool).value;
    if contains_non_finite_atom(pool, v) {
        return Err(refused(
            "substituting the point gave a non-finite value; refusing".to_string(),
        ));
    }
    Ok(v)
}

/// Is `e` built only from operations continuous at `var = point` (with the
/// declared `nodes` continuous there)?
fn jointly_continuous(
    e: ExprId,
    var: ExprId,
    point: ExprId,
    nodes: &[ExprId],
    pool: &ExprPool,
) -> bool {
    if nodes.contains(&e) {
        return true;
    }
    match pool.get(e) {
        ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_)
        | ExprData::Symbol { .. } => true,
        ExprData::Add(a) | ExprData::Mul(a) => a
            .iter()
            .all(|&x| jointly_continuous(x, var, point, nodes, pool)),
        ExprData::Pow { base, exp } => {
            let ExprData::Integer(n) = pool.get(exp) else {
                return false;
            };
            if !jointly_continuous(base, var, point, nodes, pool) {
                return false;
            }
            if n.0 >= 0 {
                return true;
            }
            // Negative power: the base must be a non-zero rational constant at
            // the point, and free of the opaque nodes (their value is unknown).
            if nodes
                .iter()
                .any(|&n| crate::ode::dsolve::contains(base, n, pool))
            {
                return false;
            }
            let mut at = HashMap::new();
            at.insert(var, point);
            let b = simplify(subs(base, &at, pool), pool).value;
            match pool.get(b) {
                ExprData::Integer(k) => k.0 != 0,
                ExprData::Rational(r) => r.0 != 0,
                _ => false,
            }
        }
        ExprData::Func { name, args } => {
            CONTINUOUS_FUNCS.contains(&name.as_str())
                && args.len() == 1
                && jointly_continuous(args[0], var, point, nodes, pool)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn setup() -> (ExprPool, ExprId, ExprId, ExprId) {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let f = pool.func("f", vec![t]);
        let a = pool.integer(2);
        (pool, t, f, a)
    }

    #[test]
    fn undeclared_function_is_refused() {
        let (pool, t, f, a) = setup();
        let r = limit_with_continuous(f, t, a, LimitDirection::Bidirectional, &[], &pool);
        assert!(matches!(r, Err(OpaqueLimitError::Refused(_))));
    }

    #[test]
    fn declared_continuous_function_substitutes() {
        let (pool, t, f, a) = setup();
        let e = pool.add(vec![
            pool.mul(vec![f, f]),
            pool.mul(vec![pool.integer(3), t]),
        ]);
        let v = limit_with_continuous(e, t, a, LimitDirection::Bidirectional, &[f], &pool).unwrap();
        let f2 = pool.func("f", vec![a]);
        let want = simplify(
            pool.add(vec![pool.pow(f2, pool.integer(2)), pool.integer(6)]),
            &pool,
        )
        .value;
        assert_eq!(v, want);
    }

    #[test]
    fn derivative_quotient_is_refused() {
        let (pool, t, f, a) = setup();
        let fa = pool.func("f", vec![a]);
        let num = pool.add(vec![f, pool.mul(vec![pool.integer(-1), fa])]);
        let den = pool.add(vec![t, pool.integer(-2)]);
        let e = pool.mul(vec![num, pool.pow(den, pool.integer(-1))]);
        let r = limit_with_continuous(e, t, a, LimitDirection::Bidirectional, &[f], &pool);
        assert!(matches!(r, Err(OpaqueLimitError::Refused(_))));
    }
}
