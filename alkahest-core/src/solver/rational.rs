//! Clearing denominators: turning a **rational**-function equation into the
//! polynomial numerator the Gröbner solver can take, plus the condition that
//! says where that numerator is allowed to speak for it.
//!
//! Every admittance equation in circuit analysis has the shape `(V₁ − V₂)/R`
//! or `V/(s·L)`, so refusing anything with a negative exponent refuses symbolic
//! nodal analysis outright.  Multiplying up is easy; the part that has to be
//! got right is that **it is not an equivalence**:
//!
//! ```text
//!     N(x)/D(x) = 0     ⟺     N(x) = 0  ∧  D(x) ≠ 0
//! ```
//!
//! The right-hand conjunct is dropped by the multiplication, and a root of `N`
//! at which the equation is undefined is *not* a solution of it.
//! `x/(x−1) = 1/(x−1)` clears to `(x−1)² = 0`, whose only root `x = 1` is
//! precisely the point where the original equation reads `1/0 = 1/0`.
//! Returning it would be a wrong answer, not a cosmetic one, so
//! [`super::exclude_pole_roots`] removes it again downstream, using the
//! `domain` polynomial this module returns alongside the numerator.
//!
//! # The domain polynomial, and why it is not "the denominator"
//!
//! The obvious candidate — the product of the denominators multiplied through —
//! is **wrong**, and wrong in the direction that returns a spurious root.  A
//! reciprocal swaps the two halves: `(n/d)⁻¹ = d/n`, so an inner *denominator*
//! becomes an outer *numerator* and drops out of the product entirely.  In
//! `1/(1/x − 1)` the requirement `x ≠ 0` disappears exactly that way, leaving
//! `1 − x` as the only recorded condition and `x = 0` — a point where the
//! expression has no value — looking like a solution.
//!
//! So what is carried is the **domain**: the product of everything that has to
//! be non-zero for the expression to denote a number at all, accumulated
//! through the recursion with
//!
//! ```text
//!     rec(e) = (n, d, dom)   such that   dom(p) ≠ 0  ⟹  e is defined at p,
//!                                                       d(p) ≠ 0,
//!                                                       and e(p) = n(p)/d(p)
//! ```
//!
//! and the one rule that adds to `dom` is the negative power: `(n/d)^−k`
//! requires `n ≠ 0` on top of whatever the base already required.  For
//! `1/(1/x − 1)` that yields `dom = x·(1 − x)`, which is precisely the pair of
//! points the expression is undefined at.
//!
//! # Why nothing is reduced to lowest terms
//!
//! Cancelling `N/D` is tempting and is *not* sound on its own.  Writing
//! `D = g·D′` and `N = g·N′`, the original solution set is
//! `{N′ = 0} ∩ {g ≠ 0} ∩ {D′ ≠ 0}`, while the reduced one is
//! `{N′ = 0} ∩ {D′ ≠ 0}` — larger by exactly the removable singularities that
//! happen to be zeros of `N′`.  `x²/x = 0` is the smallest example: cancelling
//! gives `x = 0`, but the original expression is `0/0` there and the equation
//! has no solution at all.  Keeping `dom` uncancelled keeps that conjunct
//! available.

use super::SolverError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::groebner::GbPoly;
use rug::Rational;
use std::collections::BTreeMap;

/// One equation rewritten as `numer = 0`, valid where `domain ≠ 0`.
///
/// Both polynomials are over the solver's full indeterminate list (unknowns
/// followed by parameters).  `domain` is never the zero polynomial: an equation
/// that is undefined everywhere is refused before this is built.
#[derive(Debug, Clone)]
pub(crate) struct ClearedEquation {
    /// The numerator — what the Gröbner solver is actually given.
    pub numer: GbPoly,
    /// Everything that has to be non-zero for the equation to have a value,
    /// as one product and **not** reduced against `numer`.  A candidate root at
    /// which this vanishes is not a solution of the equation the caller wrote.
    pub domain: GbPoly,
}

impl ClearedEquation {
    /// True when nothing was multiplied through — the input was already a
    /// polynomial, is defined everywhere, and no exclusion step is owed.
    pub fn is_polynomial(&self) -> bool {
        is_one(&self.domain)
    }
}

/// Is `p` the constant polynomial `1`?
fn is_one(p: &GbPoly) -> bool {
    p.terms.len() == 1
        && p.terms
            .iter()
            .all(|(exp, c)| exp.iter().all(|&e| e == 0) && *c == 1)
}

/// Put `expr` over a common denominator as polynomials in `vars`.
///
/// `vars` is the solver's full indeterminate list: unknowns first, then the
/// free parameters.  Anything that is still not polynomial once the division
/// nodes have been accounted for — `exp(x)`, a symbolic exponent, a symbol
/// absent from `vars` — is refused with the same
/// [`SolverError::NotPolynomial`] the polynomial-only conversion raised, so a
/// transcendental input keeps its existing verdict and its existing code.
pub(crate) fn clear_denominators(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<ClearedEquation, SolverError> {
    let n = vars.len();
    let (numer, _denom, domain) = rec(expr, vars, n, pool)?;
    Ok(ClearedEquation { numer, domain })
}

/// `(numerator, denominator, domain)` of `expr`, maintaining the invariant
/// stated in the module documentation: wherever `domain` is non-zero, `expr` is
/// defined, `denominator` is non-zero, and `expr = numerator / denominator`.
///
/// Neither `denominator` nor `domain` is ever the zero polynomial — the only
/// way either could become one is a negative power of an identically-zero base,
/// which is refused.
fn rec(
    expr: ExprId,
    vars: &[ExprId],
    n_vars: usize,
    pool: &ExprPool,
) -> Result<(GbPoly, GbPoly, GbPoly), SolverError> {
    let one = || GbPoly::constant(Rational::from(1), n_vars);

    if let Some(idx) = vars.iter().position(|&v| v == expr) {
        let mut exp = vec![0u32; n_vars];
        exp[idx] = 1;
        let mut terms = BTreeMap::new();
        terms.insert(exp, Rational::from(1));
        return Ok((GbPoly { terms, n_vars }, one(), one()));
    }

    enum Node {
        IntConst(rug::Integer),
        RatConst(Rational),
        FloatConst(f64),
        FreeSymbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String),
        Other,
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Integer(n) => Node::IntConst(n.0.clone()),
        ExprData::Rational(r) => Node::RatConst(r.0.clone()),
        ExprData::Float(f) => Node::FloatConst(f.inner.to_f64()),
        ExprData::Symbol { name, .. } => Node::FreeSymbol(name.clone()),
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        ExprData::Func { name, .. } => Node::Func(name.clone()),
        _ => Node::Other,
    });

    match node {
        Node::IntConst(n) => Ok((GbPoly::constant(Rational::from(n), n_vars), one(), one())),
        Node::RatConst(r) => Ok((GbPoly::constant(r, n_vars), one(), one())),
        Node::FloatConst(v) => {
            let r = Rational::from_f64(v).unwrap_or_else(|| Rational::from(0));
            Ok((GbPoly::constant(r, n_vars), one(), one()))
        }
        Node::FreeSymbol(name) => Err(SolverError::NotPolynomial(format!(
            "free symbol '{name}' not in variable list"
        ))),
        Node::Add(args) => {
            // `n₁/d₁ + n₂/d₂ = (n₁·d₂ + n₂·d₁)/(d₁·d₂)`, left-folded.  A sum is
            // defined exactly where both summands are, so the domains multiply
            // and nothing new is required.
            //
            // The `is_one` guards are not micro-optimisation: without them a
            // purely polynomial equation would pay several full polynomial
            // multiplications by the constant `1` per summand, which is a cost
            // the polynomial path never used to have.
            let mut num = GbPoly::zero(n_vars);
            let mut den = one();
            let mut dom = one();
            for a in args {
                let (n2, d2, dom2) = rec(a, vars, n_vars, pool)?;
                let scaled = if is_one(&den) { n2 } else { n2.mul(&den) };
                num = if is_one(&d2) { num } else { num.mul(&d2) }.add(&scaled);
                if !is_one(&d2) {
                    den = den.mul(&d2);
                }
                dom = combine_domains(dom, dom2);
            }
            Ok((num, den, dom))
        }
        Node::Mul(args) => {
            let mut num = one();
            let mut den = one();
            let mut dom = one();
            for a in args {
                let (n2, d2, dom2) = rec(a, vars, n_vars, pool)?;
                num = num.mul(&n2);
                if !is_one(&d2) {
                    den = den.mul(&d2);
                }
                dom = combine_domains(dom, dom2);
            }
            Ok((num, den, dom))
        }
        Node::Pow(base, exp_id) => {
            let exp_node = pool.with(exp_id, |d| match d {
                ExprData::Integer(n) => Some(n.0.clone()),
                _ => None,
            });
            let Some(n) = exp_node else {
                return Err(SolverError::NotPolynomial(
                    "symbolic or non-integer exponent".to_string(),
                ));
            };
            // Exponent vectors are `u32`. Expanding past that range would wrap
            // in a release build — silently returning a polynomial that is not
            // the input's — so refuse instead. Nothing computable is excluded:
            // a single term of degree 2³² is already far past what any Gröbner
            // basis over these polynomials could be asked to do.
            let too_wide = "integer exponent too large to expand";
            let Some(n_val) = n.to_i64() else {
                return Err(SolverError::NotPolynomial(too_wide.to_string()));
            };
            if u32::try_from(n_val.unsigned_abs()).is_err() {
                return Err(SolverError::NotPolynomial(too_wide.to_string()));
            }
            let (bn, bd, bdom) = rec(base, vars, n_vars, pool)?;
            if n_val >= 0 {
                // A non-negative power is defined wherever its base is.
                let k = n_val as u64;
                let den = if is_one(&bd) { bd } else { pow_poly(&bd, k) };
                Ok((pow_poly(&bn, k), den, bdom))
            } else {
                // `(n/d)^−k = d^k / n^k`: the base's numerator becomes the
                // denominator, so `n ≠ 0` joins the domain — this is the only
                // rule that adds to it, and the one the "product of the
                // denominators" reading gets wrong.  A base that is identically
                // zero makes the equation undefined everywhere rather than
                // merely non-polynomial.
                if bn.is_zero() {
                    return Err(super::refuse_undefined_equation(
                        "a denominator is identically zero",
                    ));
                }
                let k = n_val.unsigned_abs();
                let dom = combine_domains(bdom, bn.clone());
                Ok((pow_poly(&bd, k), pow_poly(&bn, k), dom))
            }
        }
        Node::Func(name) => Err(SolverError::NotPolynomial(format!(
            "function '{name}' is not a polynomial"
        ))),
        Node::Other => Err(SolverError::NotPolynomial(
            "unsupported expression node".to_string(),
        )),
    }
}

/// `a·b`, skipping the multiplication when either side is the constant `1`.
///
/// Two conditions hold together exactly when their product is non-zero, so a
/// domain is one polynomial rather than a list — there is no way for a nested
/// requirement to be dropped on the way up.
fn combine_domains(a: GbPoly, b: GbPoly) -> GbPoly {
    if is_one(&b) {
        a
    } else if is_one(&a) {
        b
    } else {
        a.mul(&b)
    }
}

/// `p^k` by binary exponentiation.
fn pow_poly(p: &GbPoly, k: u64) -> GbPoly {
    let mut result = GbPoly::constant(Rational::from(1), p.n_vars);
    let mut cur = p.clone();
    let mut rem = k;
    while rem > 0 {
        if rem & 1 == 1 {
            result = result.mul(&cur);
        }
        rem >>= 1;
        if rem > 0 {
            let cur2 = cur.clone();
            cur = cur.mul(&cur2);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::solver::expr_to_gbpoly;

    /// `terms` of the polynomial `expr` denotes, for comparing against a
    /// numerator or a domain without going through a display form.
    fn poly(expr: ExprId, vars: &[ExprId], pool: &ExprPool) -> BTreeMap<Vec<u32>, Rational> {
        expr_to_gbpoly(expr, vars, pool).unwrap().terms
    }

    /// `clear_denominators` agrees with the polynomial-only conversion on
    /// polynomial input, and reports that nothing was cleared.
    #[test]
    fn a_polynomial_clears_to_itself_over_one() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let c = clear_denominators(e, &[x], &pool).unwrap();
        assert!(c.is_polynomial(), "no denominator was introduced");
        assert_eq!(c.numer.terms, poly(e, &[x], &pool));
    }

    #[test]
    fn reciprocal_moves_to_the_denominator() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.pow(x, pool.integer(-1_i32));
        let c = clear_denominators(e, &[x], &pool).unwrap();
        assert!(!c.is_polynomial());
        assert_eq!(c.numer.terms, poly(pool.integer(1_i32), &[x], &pool));
        assert_eq!(c.domain.terms, poly(x, &[x], &pool));
    }

    /// A reciprocal swaps numerator and denominator, so an inner denominator
    /// becomes an outer numerator and would vanish from any "product of the
    /// denominators" bookkeeping.  `1/(1/x − 1)` is undefined at `x = 0` *and*
    /// at `x = 1`, and the domain has to name both — the first is the one a
    /// denominator product loses, and losing it hands back `x = 0` as a root.
    #[test]
    fn a_reciprocal_of_a_quotient_keeps_the_inner_condition() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let inner = pool.add(vec![
            pool.pow(x, pool.integer(-1_i32)),
            pool.integer(-1_i32),
        ]);
        let c = clear_denominators(pool.pow(inner, pool.integer(-1_i32)), &[x], &pool).unwrap();
        // numerator x, domain x·(1 − x)
        assert_eq!(c.numer.terms, poly(x, &[x], &pool));
        let one_minus_x = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let expected = pool.mul(vec![x, one_minus_x]);
        assert_eq!(c.domain.terms, poly(expected, &[x], &pool));
    }

    /// A zeroth power of an undefined base is still undefined: `(1/x)^0` says
    /// nothing about `x = 0`, but the expression does not denote a number there.
    #[test]
    fn a_zeroth_power_does_not_erase_its_bases_domain() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let recip = pool.pow(x, pool.integer(-1_i32));
        let c = clear_denominators(pool.pow(recip, pool.integer(0_i32)), &[x], &pool).unwrap();
        assert!(!c.is_polynomial(), "x ≠ 0 is still required");
        assert_eq!(c.domain.terms, poly(x, &[x], &pool));
    }

    /// The domain is deliberately **not** reduced against the numerator:
    /// `x²/x` must keep `x` as an exclusion condition, or `x = 0` comes back as
    /// a solution of an equation that reads `0/0` there.
    #[test]
    fn a_removable_singularity_is_still_a_domain_condition() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let c = clear_denominators(e, &[x], &pool).unwrap();
        let x2 = pool.pow(x, pool.integer(2_i32));
        assert_eq!(c.numer.terms, poly(x2, &[x], &pool));
        assert_eq!(c.domain.terms, poly(x, &[x], &pool));
    }

    /// `x/(x−1) − 1/(x−1)` clears to the numerator `(x−1)²` with the domain
    /// `(x−1)²`: both the spurious root and the condition that refutes it are
    /// present.
    #[test]
    fn a_common_denominator_survives_in_both_halves() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let xm1 = pool.add(vec![x, pool.integer(-1_i32)]);
        let inv = pool.pow(xm1, pool.integer(-1_i32));
        let e = pool.add(vec![
            pool.mul(vec![x, inv]),
            pool.mul(vec![pool.integer(-1_i32), inv]),
        ]);
        let c = clear_denominators(e, &[x], &pool).unwrap();
        let sq = pool.pow(xm1, pool.integer(2_i32));
        let expected = poly(sq, &[x], &pool);
        assert_eq!(c.numer.terms, expected);
        assert_eq!(c.domain.terms, expected);
    }

    #[test]
    fn a_transcendental_keeps_its_refusal() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![pool.func("exp", vec![x]), pool.integer(-2_i32)]);
        let err = clear_denominators(e, &[x], &pool).unwrap_err();
        assert!(matches!(err, SolverError::NotPolynomial(_)), "{err}");
        assert_eq!(
            err.to_string(),
            "not a polynomial: function 'exp' is not a polynomial"
        );
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-SOLVE-001");
    }

    /// An exponent past the `u32` exponent-vector range must refuse, not wrap:
    /// a wrapped exponent is a different polynomial wearing the input's name.
    #[test]
    fn an_exponent_wider_than_the_exponent_vector_is_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for e in [5_000_000_000_i64, -5_000_000_000_i64] {
            let expr = pool.pow(x, pool.integer(e));
            let err = clear_denominators(expr, &[x], &pool).unwrap_err();
            assert!(matches!(err, SolverError::NotPolynomial(_)), "{err}");
            assert!(err.to_string().contains("too large"), "{err}");
        }
        // The ordinary range is untouched.
        let ok = pool.pow(x, pool.integer(-3_i32));
        assert!(clear_denominators(ok, &[x], &pool).is_ok());
    }

    #[test]
    fn an_identically_zero_denominator_is_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 1/(x − x)
        let zero = pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), x])]);
        let e = pool.pow(zero, pool.integer(-1_i32));
        let err = clear_denominators(e, &[x], &pool).unwrap_err();
        // The refusal travels inside `NotPolynomial` (the enum is public and
        // exhaustive) and carries its own code out of band.
        assert!(matches!(err, SolverError::NotPolynomial(_)), "{err}");
        let refusal = crate::solver::take_undefined_equation().expect("refusal recorded");
        assert_eq!(crate::errors::AlkahestError::code(&refusal), "E-SOLVE-005");
        assert!(
            crate::solver::take_undefined_equation().is_none(),
            "consuming"
        );
    }
}
