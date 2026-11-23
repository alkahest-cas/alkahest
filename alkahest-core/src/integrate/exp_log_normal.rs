//! Remove the algebraic dependencies `exp(c·log u) = u^c` and
//! `log(exp u) = u` from an integrand before the Risch tower is built.
//!
//! The Risch structure theorem requires every adjoined monomial to be
//! transcendental over the field built so far.  `exp(log y)` is not: it *is*
//! `y`.  Handed to the exponential tower as a fresh monomial `θ = exp(η)` with
//! `η = log y`, the Risch DE for `∫ θ⁻¹ dy` has no rational solution and the
//! integrator certified `∫ 1/exp(log y) dy` non-elementary (`E-INT-004`) —
//! a false theorem, the answer being `log y`.
//!
//! [`normalize_exp_log`] rewrites, recursively:
//!
//! * `exp(Σ cᵢ·log uᵢ + r) → Π uᵢ^{cᵢ} · exp(r)` for every term with a
//!   **rational literal** coefficient `cᵢ`.  This is an identity on the
//!   principal branch everywhere: `u^c` *is* `exp(c·log u)` by definition.
//! * `log(exp u) → u` when `u` is real-valued (no imaginary unit), where
//!   `log(exp u) = u` holds for every real `u`.
//!
//! The Risch dispatcher additionally refuses to *certify* when a remaining
//! generator is not provably transcendental (see
//! `risch::tower_is_certifiably_transcendental`); this pass is what lets the
//! common disguised cases be answered rather than merely not mis-certified.

use crate::kernel::{ExprData, ExprId, ExprPool};

fn rational_literal(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(_) | ExprData::Rational(_))
}

/// `Some((c, u))` when `term` is `c·log(u)` with `c` a rational literal
/// (`c = 1` for a bare `log(u)`).
fn rational_multiple_of_log(term: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    match pool.get(term) {
        ExprData::Func { name, args } if name == "log" && args.len() == 1 => {
            Some((pool.integer(1_i32), args[0]))
        }
        ExprData::Mul(factors) => {
            let mut coeff = Vec::new();
            let mut log_arg = None;
            for f in factors {
                match pool.get(f) {
                    ExprData::Func { name, args }
                        if name == "log" && args.len() == 1 && log_arg.is_none() =>
                    {
                        log_arg = Some(args[0]);
                    }
                    _ if rational_literal(f, pool) => coeff.push(f),
                    _ => return None,
                }
            }
            let c = match coeff.len() {
                0 => pool.integer(1_i32),
                1 => coeff[0],
                _ => pool.mul(coeff),
            };
            Some((c, log_arg?))
        }
        _ => None,
    }
}

fn contains_imaginary(e: ExprId, pool: &ExprPool) -> bool {
    if pool.is_imaginary_unit(e) {
        return true;
    }
    match pool.get(e) {
        ExprData::Func { args, .. } | ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_imaginary(a, pool))
        }
        ExprData::Pow { base, exp } => {
            contains_imaginary(base, pool) || contains_imaginary(exp, pool)
        }
        _ => false,
    }
}

/// See the module docs.  Returns `expr` itself when nothing changes.
pub(crate) fn normalize_exp_log(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            let args: Vec<ExprId> = args.iter().map(|&a| normalize_exp_log(a, pool)).collect();
            if args.len() == 1 && name == "exp" {
                let terms = match pool.get(args[0]) {
                    ExprData::Add(ts) => ts,
                    _ => vec![args[0]],
                };
                let mut powers = Vec::new();
                let mut rest = Vec::new();
                for t in terms {
                    match rational_multiple_of_log(t, pool) {
                        Some((c, u)) => powers.push(pool.pow(u, c)),
                        None => rest.push(t),
                    }
                }
                if !powers.is_empty() {
                    match rest.len() {
                        0 => {}
                        1 => powers.push(pool.func("exp", vec![rest[0]])),
                        _ => powers.push(pool.func("exp", vec![pool.add(rest)])),
                    }
                    return if powers.len() == 1 {
                        powers[0]
                    } else {
                        pool.mul(powers)
                    };
                }
            }
            if args.len() == 1 && name == "log" {
                if let ExprData::Func {
                    name: inner,
                    args: inner_args,
                } = pool.get(args[0])
                {
                    if inner == "exp"
                        && inner_args.len() == 1
                        && !contains_imaginary(inner_args[0], pool)
                    {
                        return inner_args[0];
                    }
                }
            }
            pool.func(name, args)
        }
        ExprData::Add(args) => pool.add(args.iter().map(|&a| normalize_exp_log(a, pool)).collect()),
        ExprData::Mul(args) => pool.mul(args.iter().map(|&a| normalize_exp_log(a, pool)).collect()),
        ExprData::Pow { base, exp } => {
            pool.pow(normalize_exp_log(base, pool), normalize_exp_log(exp, pool))
        }
        _ => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    fn norm(src: &str) -> String {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        let e = crate::parse::parse(src, &pool, &mut syms).unwrap();
        let n = crate::simplify::engine::simplify(normalize_exp_log(e, &pool), &pool).value;
        pool.display(n).to_string()
    }

    #[test]
    fn exp_of_rational_multiples_of_logs() {
        assert_eq!(norm("exp(log(x))"), "x");
        assert!(!norm("exp(-log(x))").contains("exp"));
        assert!(!norm("exp(2*log(x) + 3*log(x+1))").contains("exp"));
        assert!(norm("exp(x + log(x))").contains("exp(x)"));
        assert_eq!(norm("log(exp(x))"), "x");
    }

    #[test]
    fn symbolic_and_complex_cases_are_left_alone() {
        // `a·log x` with a symbolic `a` is `x^a` only formally; leave it.
        assert!(norm("exp(a*log(x))").contains("exp"));
        assert!(norm("log(exp(I*x))").contains("log"));
    }
}
