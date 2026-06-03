//! Constant folding helpers for predicates and numeric evaluation.

use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};

/// If *expr* is a numeric constant, return its `f64` value.
pub fn try_expr_f64(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (num, den) = r.0.clone().into_numer_denom();
            Some(num.to_f64() / den.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        _ => None,
    }
}

/// Evaluate a predicate when all arguments are numeric constants.
pub fn try_predicate_bool(kind: &PredicateKind, args: &[ExprId], pool: &ExprPool) -> Option<bool> {
    match kind {
        PredicateKind::True => Some(true),
        PredicateKind::False => Some(false),
        PredicateKind::Not => {
            let inner = try_predicate_bool_from_expr(args[0], pool)?;
            Some(!inner)
        }
        PredicateKind::And => {
            for &a in args {
                if !try_predicate_bool_from_expr(a, pool)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        PredicateKind::Or => {
            for &a in args {
                if try_predicate_bool_from_expr(a, pool)? {
                    return Some(true);
                }
            }
            Some(false)
        }
        PredicateKind::Lt => Some(try_expr_f64(args[0], pool)? < try_expr_f64(args[1], pool)?),
        PredicateKind::Le => Some(try_expr_f64(args[0], pool)? <= try_expr_f64(args[1], pool)?),
        PredicateKind::Gt => Some(try_expr_f64(args[0], pool)? > try_expr_f64(args[1], pool)?),
        PredicateKind::Ge => Some(try_expr_f64(args[0], pool)? >= try_expr_f64(args[1], pool)?),
        PredicateKind::Eq => Some(try_expr_f64(args[0], pool)? == try_expr_f64(args[1], pool)?),
        PredicateKind::Ne => Some(try_expr_f64(args[0], pool)? != try_expr_f64(args[1], pool)?),
    }
}

/// Evaluate a predicate expression node (may be nested `And`/`Or` trees).
pub fn try_predicate_bool_from_expr(expr: ExprId, pool: &ExprPool) -> Option<bool> {
    match pool.get(expr) {
        ExprData::Predicate { kind, args } => try_predicate_bool(&kind, &args, pool),
        _ => None,
    }
}
