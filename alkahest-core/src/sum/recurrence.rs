//! Homogeneous linear recurrences with constant coefficients (order ≤ 2).

use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;
use rug::Rational;
use std::fmt;

fn simp(pool: &ExprPool, e: ExprId) -> ExprId {
    simplify(e, pool).value
}

/// Errors from [`solve_linear_recurrence_homogeneous`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearRecurrenceError {
    OrderUnsupported(usize),
    InitialLength { expected: usize, got: usize },
}

impl fmt::Display for LinearRecurrenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearRecurrenceError::OrderUnsupported(o) => {
                write!(f, "recurrence order {o} is not supported (max 2)")
            }
            LinearRecurrenceError::InitialLength { expected, got } => {
                write!(f, "expected {expected} initial value(s), got {got}")
            }
        }
    }
}

impl std::error::Error for LinearRecurrenceError {}

impl crate::errors::AlkahestError for LinearRecurrenceError {
    fn code(&self) -> &'static str {
        match self {
            LinearRecurrenceError::OrderUnsupported(_) => "E-REC-001",
            LinearRecurrenceError::InitialLength { .. } => "E-REC-002",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some("use order 1 or 2 with rational coefficients; initials must match order")
    }
}

#[derive(Debug, Clone)]
pub struct RecurrenceSolution {
    pub n: ExprId,
    pub closed_form: ExprId,
}

/// Solve `∑_{i=0}^d c_i · f(n+i) = 0` with rational coefficients (`c_d ≠ 0`).
pub fn solve_linear_recurrence_homogeneous(
    pool: &ExprPool,
    n: ExprId,
    coeffs: &[Rational],
    initials: &[ExprId],
) -> Result<RecurrenceSolution, LinearRecurrenceError> {
    if coeffs.len() < 2 {
        return Err(LinearRecurrenceError::OrderUnsupported(0));
    }
    let d = coeffs.len() - 1;
    match d {
        1 => solve_order1(pool, n, coeffs, initials),
        2 => solve_order2(pool, n, coeffs, initials),
        _ => Err(LinearRecurrenceError::OrderUnsupported(d)),
    }
}

fn rational_atom(pool: &ExprPool, r: &Rational) -> ExprId {
    let numer = r.numer().clone();
    let denom = r.denom().clone();
    if denom == 1 {
        pool.integer(numer)
    } else {
        pool.rational(numer, denom)
    }
}

fn expr_div(pool: &ExprPool, num: ExprId, den: ExprId) -> ExprId {
    pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
}

fn sqrt_disc_expr(pool: &ExprPool, disc: &Rational) -> ExprId {
    let num = disc.numer().clone();
    let den = disc.denom().clone();
    let prod = num * den.clone();
    let inner = pool.integer(prod);
    let sqrt_e = pool.func("sqrt", vec![inner]);
    let den_e = pool.integer(den);
    expr_div(pool, sqrt_e, den_e)
}

fn solve_order1(
    pool: &ExprPool,
    n: ExprId,
    coeffs: &[Rational],
    initials: &[ExprId],
) -> Result<RecurrenceSolution, LinearRecurrenceError> {
    if coeffs.len() != 2 {
        return Err(LinearRecurrenceError::OrderUnsupported(1));
    }
    if initials.len() != 1 {
        return Err(LinearRecurrenceError::InitialLength {
            expected: 1,
            got: initials.len(),
        });
    }
    let r = (Rational::from(0) - coeffs[0].clone()) / coeffs[1].clone();
    let r_expr = rational_atom(pool, &r);
    let closed = simp(pool, pool.mul(vec![initials[0], pool.pow(r_expr, n)]));
    Ok(RecurrenceSolution {
        n,
        closed_form: closed,
    })
}

fn solve_order2(
    pool: &ExprPool,
    n: ExprId,
    coeffs: &[Rational],
    initials: &[ExprId],
) -> Result<RecurrenceSolution, LinearRecurrenceError> {
    if coeffs.len() != 3 {
        return Err(LinearRecurrenceError::OrderUnsupported(2));
    }
    if initials.len() != 2 {
        return Err(LinearRecurrenceError::InitialLength {
            expected: 2,
            got: initials.len(),
        });
    }
    let c0 = &coeffs[0];
    let c1 = &coeffs[1];
    let c2 = &coeffs[2];
    if c2.is_zero() {
        return Err(LinearRecurrenceError::OrderUnsupported(2));
    }

    let b = c1.clone() / c2.clone();
    let c = c0.clone() / c2.clone();
    let disc = b.clone() * b.clone() - Rational::from(4) * c.clone();
    if disc < 0 {
        return Err(LinearRecurrenceError::OrderUnsupported(2));
    }

    let sqrt_e = sqrt_disc_expr(pool, &disc);
    let neg_b = rational_atom(pool, &(-b.clone()));
    let half = rational_atom(pool, &Rational::from((1, 2)));

    let inner1 = simp(pool, pool.add(vec![neg_b, sqrt_e]));
    let r1 = simp(pool, pool.mul(vec![half, inner1]));
    let inner2 = simp(
        pool,
        pool.add(vec![neg_b, pool.mul(vec![sqrt_e, pool.integer(-1_i32)])]),
    );
    let r2 = simp(pool, pool.mul(vec![half, inner2]));

    let denom_e = simp(
        pool,
        pool.add(vec![r1, pool.mul(vec![r2, pool.integer(-1_i32)])]),
    );

    let r2_u0 = simp(pool, pool.mul(vec![initials[0], r2]));
    let num_a = simp(
        pool,
        pool.add(vec![
            initials[1],
            pool.mul(vec![r2_u0, pool.integer(-1_i32)]),
        ]),
    );

    let r1_u0 = simp(pool, pool.mul(vec![initials[0], r1]));
    let num_b = simp(
        pool,
        pool.add(vec![
            r1_u0,
            pool.mul(vec![initials[1], pool.integer(-1_i32)]),
        ]),
    );

    let big_a = expr_div(pool, num_a, denom_e);
    let big_b = expr_div(pool, num_b, denom_e);

    let closed = simp(
        pool,
        pool.add(vec![
            simp(pool, pool.mul(vec![big_a, pool.pow(r1, n)])),
            simp(pool, pool.mul(vec![big_b, pool.pow(r2, n)])),
        ]),
    );

    Ok(RecurrenceSolution {
        n,
        closed_form: closed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    #[test]
    fn fibonacci_numeric_check() {
        let pool = ExprPool::new();
        let n_sym = pool.symbol("n", Domain::Real);
        let coeffs = vec![Rational::from(-1), Rational::from(-1), Rational::from(1)];
        let initials = vec![pool.integer(0_i32), pool.integer(1_i32)];
        let sol =
            solve_linear_recurrence_homogeneous(&pool, n_sym, &coeffs, &initials).expect("solve");

        let mut fib = vec![0_i64, 1_i64];
        for _ in 2..=12 {
            let l = fib.len();
            fib.push(fib[l - 1] + fib[l - 2]);
        }

        for (ni, &expected) in fib.iter().enumerate() {
            let mut env = HashMap::new();
            env.insert(n_sym, ni as f64);
            let v = eval_interp(sol.closed_form, &env, &pool).expect("eval");
            assert!((v - expected as f64).abs() < 1e-6);
        }
    }
}
