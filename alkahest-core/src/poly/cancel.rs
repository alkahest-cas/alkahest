//! Rational-function `cancel` / `together` normalization.
//!
//! These routines take an expression built from `+`, `-`, `*`, `/` and
//! **integer**-power nodes of polynomials in `vars`, combine it over a common
//! denominator, and divide the numerator and denominator by their polynomial
//! GCD (via [`RationalFunction::new`], which performs the GCD reduction).
//!
//! Any sub-expression that is *not* a polynomial in `vars` — a function call
//! such as `sin(x)`, a symbol that is not in `vars`, or a base raised to a
//! non-integer / negative exponent that is not otherwise reducible — is treated
//! as an **opaque generator**: a fresh polynomial variable. This lets `cancel`
//! operate on expressions like `(sin(x)**2 - 1) / (sin(x) - 1)` and on matrix
//! entries containing unrelated symbols, collapsing common factors structurally.
//!
//! ## Limitations
//! * Generators are matched **structurally**: `sin(x)` and `sin(2*x/2)` are
//!   distinct generators (no simplification of arguments is performed first).
//! * A base raised to a *symbolic* exponent (e.g. `x**n`) is opaque as a whole.
//! * Negative integer powers of opaque generators are supported (they go into
//!   the denominator), but a negative power of a *non-atomic* opaque base
//!   (e.g. `sin(x)**-2`) is handled by treating `sin(x)` as a generator and
//!   inverting it.

use super::error::ConversionError;
use super::multipoly::MultiPoly;
use super::rational::RationalFunction;
use crate::kernel::{ExprData, ExprId, ExprPool};

/// Convert *expr* into a single normalized rational function over *vars*
/// (plus any opaque generators discovered inside it), returning the cancelled
/// `(numerator, denominator)` expression pair.
///
/// The numerator and denominator are returned as separate `ExprId`s so callers
/// can decide how to present the quotient; see [`cancel`] for the combined form.
pub fn together_parts(
    expr: ExprId,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<(ExprId, ExprId), ConversionError> {
    // Discover opaque generators and build the full generator list.
    let mut gens: Vec<ExprId> = vars.clone();
    collect_generators(expr, &vars, pool, &mut gens);

    let rf = expr_to_rational(expr, &gens, pool)?;
    let numer = rf.numer.to_expr(pool);
    let denom = rf.denom.to_expr(pool);
    Ok((numer, denom))
}

/// Combine *expr* over a common denominator and cancel common polynomial
/// factors, returning a single symbolic expression.
///
/// If the resulting denominator is the constant `1`, the bare numerator is
/// returned; otherwise `numerator / denominator` (i.e. `numerator *
/// denominator**-1`) is returned.
///
/// Because [`RationalFunction::new`] already divides out the numerator/
/// denominator GCD, `cancel` and [`together`] share this implementation; they
/// differ only in intent (`cancel` emphasises factor removal). `together`
/// is provided as a thin alias.
pub fn cancel(expr: ExprId, vars: Vec<ExprId>, pool: &ExprPool) -> Result<ExprId, ConversionError> {
    let (numer, denom) = together_parts(expr, vars, pool)?;
    Ok(combine(numer, denom, pool))
}

/// Alias of [`cancel`]: combine over a common denominator with GCD reduction.
pub fn together(
    expr: ExprId,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<ExprId, ConversionError> {
    cancel(expr, vars, pool)
}

/// Build `numer / denom` as a symbolic expression, simplifying `denom == 1`.
fn combine(numer: ExprId, denom: ExprId, pool: &ExprPool) -> ExprId {
    // A zero numerator collapses the whole quotient to 0 (denom is non-zero by
    // RationalFunction's invariant).
    let numer_is_zero = pool.with(numer, |n| matches!(n, ExprData::Integer(z) if z.0 == 0));
    if numer_is_zero {
        return pool.integer(0_i32);
    }
    let denom_is_one = pool.with(denom, |d| matches!(d, ExprData::Integer(n) if n.0 == 1));
    if denom_is_one {
        return numer;
    }
    let inv = pool.pow(denom, pool.integer(-1_i32));
    pool.mul(vec![numer, inv])
}

/// Walk *expr*, appending any opaque generator atoms (not already present in
/// `acc`) to `acc`. A node is opaque when it is a symbol outside `vars`, a
/// function call, or a power whose exponent is not a constant integer.
fn collect_generators(expr: ExprId, vars: &[ExprId], pool: &ExprPool, acc: &mut Vec<ExprId>) {
    enum Kind {
        Var,
        Opaque,
        Recurse(Vec<ExprId>),
        PowIntExp(ExprId),
        Skip,
    }

    let kind = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } => {
            if vars.contains(&expr) {
                Kind::Var
            } else {
                Kind::Opaque
            }
        }
        ExprData::Integer(_) => Kind::Skip,
        ExprData::Rational(_) | ExprData::Float(_) => Kind::Skip,
        ExprData::Add(args) | ExprData::Mul(args) => Kind::Recurse(args.clone()),
        ExprData::Pow { base, exp } => {
            let int_exp = pool.with(*exp, |e| matches!(e, ExprData::Integer(_)));
            if int_exp {
                Kind::PowIntExp(*base)
            } else {
                Kind::Opaque
            }
        }
        ExprData::Func { .. } => Kind::Opaque,
        _ => Kind::Opaque,
    });

    match kind {
        Kind::Var | Kind::Skip => {}
        Kind::Opaque => {
            if !acc.contains(&expr) {
                acc.push(expr);
            }
        }
        Kind::Recurse(args) => {
            for a in args {
                collect_generators(a, vars, pool, acc);
            }
        }
        Kind::PowIntExp(base) => collect_generators(base, vars, pool, acc),
    }
}

/// Recursively convert *expr* into a [`RationalFunction`] over generator list
/// `gens`. Division and negative integer powers are honoured; opaque atoms in
/// `gens` are treated as polynomial variables.
fn expr_to_rational(
    expr: ExprId,
    gens: &[ExprId],
    pool: &ExprPool,
) -> Result<RationalFunction, ConversionError> {
    // If this whole node is one of our generators, emit it directly. This
    // short-circuits opaque atoms (functions, foreign symbols, symbolic
    // powers) before we try to interpret their internal structure.
    if gens.contains(&expr) {
        return generator_rf(expr, gens, pool);
    }

    enum Node {
        Poly,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow { base: ExprId, exp_i64: Option<i64> },
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Symbol { .. } => Node::Poly,
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => {
            let exp_i64 = pool.with(*exp, |e| match e {
                ExprData::Integer(n) => n.0.to_i64(),
                _ => None,
            });
            Node::Pow {
                base: *base,
                exp_i64,
            }
        }
        _ => Node::Poly,
    });

    match node {
        // A polynomial leaf/atom: convert directly to a poly-over-1 rational.
        Node::Poly => poly_rf(expr, gens, pool),
        Node::Add(args) => {
            let mut acc = const_rf(0, gens);
            for a in args {
                let r = expr_to_rational(a, gens, pool)?;
                acc = (acc + r)?;
            }
            Ok(acc)
        }
        Node::Mul(args) => {
            let mut acc = const_rf(1, gens);
            for a in args {
                let r = expr_to_rational(a, gens, pool)?;
                acc = (acc * r)?;
            }
            Ok(acc)
        }
        Node::Pow { base, exp_i64 } => {
            let n = exp_i64.ok_or(ConversionError::NonConstantExponent)?;
            let base_rf = expr_to_rational(base, gens, pool)?;
            pow_rf(base_rf, n, gens)
        }
    }
}

/// Build the rational function `g / 1` for a single generator `g`.
fn generator_rf(
    g: ExprId,
    gens: &[ExprId],
    _pool: &ExprPool,
) -> Result<RationalFunction, ConversionError> {
    let idx = gens
        .iter()
        .position(|&v| v == g)
        .expect("generator present");
    let mut exp = vec![0u32; idx + 1];
    exp[idx] = 1;
    let mut terms = std::collections::BTreeMap::new();
    terms.insert(exp, rug::Integer::from(1));
    let numer = MultiPoly {
        vars: gens.to_vec(),
        terms,
    };
    let denom = MultiPoly::constant(gens.to_vec(), 1);
    RationalFunction::new(numer, denom)
}

/// Convert a polynomial expression (no division, only integer powers) over
/// `gens` into a `poly / 1` rational function.
fn poly_rf(
    expr: ExprId,
    gens: &[ExprId],
    pool: &ExprPool,
) -> Result<RationalFunction, ConversionError> {
    let numer = MultiPoly::from_symbolic(expr, gens.to_vec(), pool)?;
    let denom = MultiPoly::constant(gens.to_vec(), 1);
    RationalFunction::new(numer, denom)
}

/// The constant rational function `c / 1`.
fn const_rf(c: i64, gens: &[ExprId]) -> RationalFunction {
    RationalFunction {
        numer: MultiPoly::constant(gens.to_vec(), c),
        denom: MultiPoly::constant(gens.to_vec(), 1),
    }
}

/// Raise a rational function to an integer power `n` (negative inverts).
fn pow_rf(
    base: RationalFunction,
    n: i64,
    gens: &[ExprId],
) -> Result<RationalFunction, ConversionError> {
    if n == 0 {
        return Ok(const_rf(1, gens));
    }
    let (b, exp) = if n < 0 {
        // Invert: (num/den)^-1 = den/num.
        if base.numer.is_zero() {
            return Err(ConversionError::ZeroDenominator);
        }
        (
            RationalFunction::new(base.denom.clone(), base.numer.clone())?,
            (-n) as u64,
        )
    } else {
        (base, n as u64)
    };
    let mut acc = const_rf(1, gens);
    for _ in 0..exp {
        acc = (acc * b.clone())?;
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::matrix::Matrix;

    fn is_int(pool: &ExprPool, e: ExprId, val: i64) -> bool {
        pool.with(e, |d| matches!(d, ExprData::Integer(n) if n.0 == val))
    }

    #[test]
    fn cancels_difference_of_squares() {
        // (x^2 - 1)/(x - 1) -> x + 1
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let num = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-1_i32)]);
        let den = p.add(vec![x, p.integer(-1_i32)]);
        let expr = p.mul(vec![num, p.pow(den, p.integer(-1_i32))]);
        let out = cancel(expr, vec![x], &p).unwrap();
        // Expect x + 1 (denominator gone). Normalize via poly_normal-style check.
        let mp = MultiPoly::from_symbolic(out, vec![x], &p).unwrap();
        assert_eq!(mp.terms.get(&vec![1]).cloned(), Some(rug::Integer::from(1)));
        assert_eq!(mp.terms.get(&vec![]).cloned(), Some(rug::Integer::from(1)));
        assert_eq!(mp.terms.len(), 2, "should be exactly x + 1: {mp:?}");
    }

    #[test]
    fn sum_of_two_fractions() {
        // 1/x + 1/(x+1) -> (2x + 1) / (x^2 + x)
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inv_x = p.pow(x, p.integer(-1_i32));
        let xp1 = p.add(vec![x, p.integer(1_i32)]);
        let inv_xp1 = p.pow(xp1, p.integer(-1_i32));
        let expr = p.add(vec![inv_x, inv_xp1]);
        let (numer, denom) = together_parts(expr, vec![x], &p).unwrap();
        let n = MultiPoly::from_symbolic(numer, vec![x], &p).unwrap();
        let d = MultiPoly::from_symbolic(denom, vec![x], &p).unwrap();
        // numerator 2x + 1
        assert_eq!(n.terms.get(&vec![1]).cloned(), Some(rug::Integer::from(2)));
        assert_eq!(n.terms.get(&vec![]).cloned(), Some(rug::Integer::from(1)));
        // denominator x^2 + x
        assert_eq!(d.terms.get(&vec![2]).cloned(), Some(rug::Integer::from(1)));
        assert_eq!(d.terms.get(&vec![1]).cloned(), Some(rug::Integer::from(1)));
        assert_eq!(d.terms.len(), 2, "denominator should be x^2 + x: {d:?}");
    }

    #[test]
    fn constant_quotient_collapses() {
        // x / x -> 1
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let expr = p.mul(vec![x, p.pow(x, p.integer(-1_i32))]);
        let out = cancel(expr, vec![x], &p).unwrap();
        assert!(is_int(&p, out, 1), "x/x should cancel to 1");
    }

    #[test]
    fn opaque_generator_cancellation() {
        // (sin(x)^2 - 1)/(sin(x) - 1) -> sin(x) + 1, with vars = [x].
        // sin(x) is opaque (a function), so it becomes a generator.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let s = p.func("sin", vec![x]);
        let num = p.add(vec![p.pow(s, p.integer(2_i32)), p.integer(-1_i32)]);
        let den = p.add(vec![s, p.integer(-1_i32)]);
        let expr = p.mul(vec![num, p.pow(den, p.integer(-1_i32))]);
        let out = cancel(expr, vec![x], &p).unwrap();
        // Result should be sin(x) + 1: a poly in the opaque generator sin(x).
        // Re-run cancel treating sin(x) as a generator to extract numer/denom,
        // since MultiPoly::from_symbolic does not recognise composite generators.
        let (numer, denom) = together_parts(out, vec![x], &p).unwrap();
        // denominator should be 1
        assert!(
            is_int(&p, denom, 1),
            "denominator should collapse to 1: {}",
            p.display(denom)
        );
        // numerator is sin(x) + 1 — confirm structurally via display.
        let s_disp = p.display(numer).to_string();
        assert!(
            s_disp.contains("sin") && s_disp.contains("1"),
            "numerator should be sin(x) + 1, got {s_disp}"
        );
    }

    #[test]
    fn foreign_symbol_is_opaque() {
        // (y^2 - 1)/(y - 1) with vars = [x] -> y + 1 (y opaque).
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let num = p.add(vec![p.pow(y, p.integer(2_i32)), p.integer(-1_i32)]);
        let den = p.add(vec![y, p.integer(-1_i32)]);
        let expr = p.mul(vec![num, p.pow(den, p.integer(-1_i32))]);
        let out = cancel(expr, vec![x], &p).unwrap();
        let mp = MultiPoly::from_symbolic(out, vec![y], &p).unwrap();
        assert_eq!(mp.terms.get(&vec![1]).cloned(), Some(rug::Integer::from(1)));
        assert_eq!(mp.terms.get(&vec![]).cloned(), Some(rug::Integer::from(1)));
    }

    #[test]
    fn matrix_inverse_product_is_identity() {
        // For a symbolic 2x2 A, cancel applied entrywise to A * A^-1 yields I.
        let p = ExprPool::new();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let c = p.symbol("c", Domain::Real);
        let d = p.symbol("d", Domain::Real);
        let m = Matrix::new(vec![vec![a, b], vec![c, d]]).unwrap();
        let inv = m.inverse(&p).unwrap();
        let prod = m.mul(&inv, &p).unwrap();
        let vars = vec![a, b, c, d];
        for r in 0..2 {
            for col in 0..2 {
                let entry = prod.get(r, col);
                let reduced = cancel(entry, vars.clone(), &p).unwrap();
                let expect = if r == col { 1 } else { 0 };
                assert!(
                    is_int(&p, reduced, expect),
                    "entry ({r},{col}) should cancel to {expect}, got {}",
                    p.display(reduced)
                );
            }
        }
    }
}
