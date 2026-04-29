//! Decompose a symbolic expression into `A(x) + B(x) * sqrt(P(x))`.
//!
//! Given the ExprId of the sqrt generator (either `sqrt(P)` or `P^(1/2)`)
//! and the expression to decompose, this module returns `(A, B)` such that
//! `expr == A + B * sqrt_id` where A and B are free of the sqrt generator.
//!
//! Arithmetic in the algebraic field K = Q(x)\[y\]/(y² - P):
//!   - Addition:       (a,b) + (c,d) = (a+c, b+d)
//!   - Multiplication: (a,b)·(c,d)   = (a·c + b·d·P, a·d + b·c)
//!   - Inversion:      (a,b)⁻¹       = (a/(a²−b²P), −b/(a²−b²P))
//!   - Integer power:  (a,b)^n via squaring

use crate::kernel::{ExprData, ExprId, ExprPool};
use super::poly_utils::{is_free_of_subexpr, as_integer};

// ---------------------------------------------------------------------------
// Field element: a + b*sqrt(P)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct FieldElem {
    pub a: ExprId, // rational part (free of sqrt)
    pub b: ExprId, // coefficient of sqrt (free of sqrt)
}

impl FieldElem {
    pub fn pure_rational(a: ExprId, pool: &ExprPool) -> Self {
        FieldElem { a, b: pool.integer(0_i32) }
    }
    pub fn pure_sqrt(b: ExprId, pool: &ExprPool) -> Self {
        FieldElem { a: pool.integer(0_i32), b }
    }
    pub fn one(pool: &ExprPool) -> Self {
        FieldElem { a: pool.integer(1_i32), b: pool.integer(0_i32) }
    }
    pub fn zero(pool: &ExprPool) -> Self {
        FieldElem { a: pool.integer(0_i32), b: pool.integer(0_i32) }
    }

    pub fn add(self, other: FieldElem, pool: &ExprPool) -> FieldElem {
        let a = pool.add(vec![self.a, other.a]);
        let b = pool.add(vec![self.b, other.b]);
        FieldElem { a, b }
    }

    #[allow(dead_code)]
    pub fn neg(self, pool: &ExprPool) -> FieldElem {
        let neg1 = pool.integer(-1_i32);
        let a = pool.mul(vec![neg1, self.a]);
        let b = pool.mul(vec![neg1, self.b]);
        FieldElem { a, b }
    }

    /// Multiply two field elements: (a+b·y)·(c+d·y) = (a·c + b·d·P) + (a·d + b·c)·y
    pub fn mul(self, other: FieldElem, p: ExprId, pool: &ExprPool) -> FieldElem {
        // new_a = self.a * other.a + self.b * other.b * P
        let ac = pool.mul(vec![self.a, other.a]);
        let bd_p = pool.mul(vec![self.b, other.b, p]);
        let new_a = pool.add(vec![ac, bd_p]);
        // new_b = self.a * other.b + self.b * other.a
        let ad = pool.mul(vec![self.a, other.b]);
        let bc = pool.mul(vec![self.b, other.a]);
        let new_b = pool.add(vec![ad, bc]);
        FieldElem { a: new_a, b: new_b }
    }

    /// Invert: 1/(a+b·y) = (a−b·y) / (a²−b²·P) = conj/norm
    pub fn inv(self, p: ExprId, pool: &ExprPool) -> FieldElem {
        use super::poly_utils::is_zero_expr;
        // Special case: inv(0, b) = (0, (b·P)^{-1})
        // This avoids the messy (-b^2·P)^{-1} form that confuses later pattern matching.
        if is_zero_expr(self.a, pool) {
            let bp = pool.mul(vec![self.b, p]);
            let new_b = pool.pow(bp, pool.integer(-1_i32));
            return FieldElem { a: pool.integer(0_i32), b: new_b };
        }
        // General case: norm = a^2 - b^2 * P
        let a2 = pool.pow(self.a, pool.integer(2_i32));
        let b2_p = pool.mul(vec![pool.pow(self.b, pool.integer(2_i32)), p]);
        let neg1 = pool.integer(-1_i32);
        let norm = pool.add(vec![a2, pool.mul(vec![neg1, b2_p])]);
        let norm_inv = pool.pow(norm, pool.integer(-1_i32));
        let new_a = pool.mul(vec![self.a, norm_inv]);
        let new_b = pool.mul(vec![neg1, self.b, norm_inv]);
        FieldElem { a: new_a, b: new_b }
    }

    /// Integer power (positive, negative, or zero)
    pub fn powi(self, n: i64, p: ExprId, pool: &ExprPool) -> FieldElem {
        if n == 0 {
            return FieldElem::one(pool);
        }
        if n < 0 {
            return self.inv(p, pool).powi(-n, p, pool);
        }
        if n == 1 {
            return self;
        }
        // Fast exponentiation by squaring
        let half = self.powi(n / 2, p, pool);
        let sq = half.mul(half, p, pool);
        if n % 2 == 0 {
            sq
        } else {
            sq.mul(self, p, pool)
        }
    }
}

// ---------------------------------------------------------------------------
// Main decomposition
// ---------------------------------------------------------------------------

/// Decompose `expr` into `(A, B)` such that `expr = A + B * sqrt_id`
/// where A and B are free of `sqrt_id`.
///
/// Returns `None` if the decomposition cannot be performed (e.g., sqrt appears
/// with a different argument than `p_expr`, or the expression is structurally
/// incompatible).
pub fn decompose_sqrt(
    expr: ExprId,
    sqrt_id: ExprId,
    p_expr: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let elem = decompose_elem(expr, sqrt_id, p_expr, pool)?;
    Some((elem.a, elem.b))
}

/// Recursive decomposition returning a `FieldElem`.
fn decompose_elem(
    expr: ExprId,
    sqrt_id: ExprId,
    p_expr: ExprId,
    pool: &ExprPool,
) -> Option<FieldElem> {
    // Base case: expr is the sqrt generator itself
    if expr == sqrt_id {
        return Some(FieldElem::pure_sqrt(pool.integer(1_i32), pool));
    }

    // Base case: expr is free of the sqrt generator
    if is_free_of_subexpr(expr, sqrt_id, pool) {
        return Some(FieldElem::pure_rational(expr, pool));
    }

    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut acc = FieldElem::zero(pool);
            for a in &args {
                let elem = decompose_elem(*a, sqrt_id, p_expr, pool)?;
                acc = acc.add(elem, pool);
            }
            Some(acc)
        }

        ExprData::Mul(args) => {
            let mut acc = FieldElem::one(pool);
            for a in &args {
                let elem = decompose_elem(*a, sqrt_id, p_expr, pool)?;
                acc = acc.mul(elem, p_expr, pool);
            }
            Some(acc)
        }

        ExprData::Pow { base, exp } => {
            // Special case: sqrt_id^n or (p_expr)^(1/2) patterns
            if base == sqrt_id {
                // sqrt(P)^n
                let n = as_integer(exp, pool)?;
                // sqrt(P)^n = P^(n/2) for n even, P^((n-1)/2) * sqrt(P) for n odd
                if n == 0 {
                    return Some(FieldElem::one(pool));
                }
                if n > 0 {
                    let n_u = n as u32;
                    if n_u % 2 == 0 {
                        // P^(n/2) — fully rational
                        let p_pow = pool.pow(p_expr, pool.integer(n_u / 2));
                        return Some(FieldElem::pure_rational(p_pow, pool));
                    } else {
                        // P^((n-1)/2) * sqrt(P)
                        let p_pow = pool.pow(p_expr, pool.integer((n_u - 1) / 2));
                        return Some(FieldElem::pure_sqrt(p_pow, pool));
                    }
                } else {
                    // Negative power of sqrt(P): sqrt(P)^(-n) = 1/sqrt(P)^n
                    let base_elem = FieldElem::pure_sqrt(pool.integer(1_i32), pool);
                    return Some(base_elem.powi(n, p_expr, pool));
                }
            }

            // Fractional power: base^(p/q) where this is the sqrt generator
            // We handle Pow(p_expr, Rational(1/2)) → same as sqrt_id, already handled above
            // For Pow(base, Integer(n)) where base contains sqrt:
            if let Some(n) = as_integer(exp, pool) {
                let base_elem = decompose_elem(base, sqrt_id, p_expr, pool)?;
                return Some(base_elem.powi(n, p_expr, pool));
            }

            // Pow with non-integer exponent that isn't our sqrt generator: give up
            None
        }

        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            // This is a different sqrt — only allowed if it matches our generator
            if expr == sqrt_id {
                Some(FieldElem::pure_sqrt(pool.integer(1_i32), pool))
            } else {
                // Different algebraic generator — we don't support multiple generators
                None
            }
        }

        _ => {
            // Any other expression: if free of sqrt_id it's rational, else unsupported
            if is_free_of_subexpr(expr, sqrt_id, pool) {
                Some(FieldElem::pure_rational(expr, pool))
            } else {
                None
            }
        }
    }
}
