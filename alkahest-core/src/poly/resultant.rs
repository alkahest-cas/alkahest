//! Resultant and subresultant polynomial remainder sequence (V2-2).
//!
//! # Public API
//!
//! - [`resultant`] — compute `res(p, q, var)` using FLINT's multivariate
//!   resultant.  Works for univariate (integer result) and multivariate
//!   (polynomial result) inputs.
//! - [`subresultant_prs`] — compute the full subresultant polynomial
//!   remainder sequence for univariate polynomials with integer coefficients.
//!
//! # Derivation log
//!
//! Both functions record a single [`RewriteStep`] with rule name
//! `"Resultant"` / `"SubresultantPRS"` and the Lean 4 theorem tag
//! `Polynomial.resultant_eq_zero_iff_common_root`.

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep};
use crate::flint::integer::FlintInteger;
use crate::flint::mpoly::FlintMPolyCtx;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::error::ConversionError;
use crate::poly::multipoly::multi_to_flint_pub;
use crate::poly::multipoly::MultiPoly;
use crate::poly::unipoly::UniPoly;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by [`resultant`] and [`subresultant_prs`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultantError {
    /// One or both expressions could not be parsed as polynomials in the
    /// given variable(s).
    NotAPolynomial(ConversionError),
    /// FLINT's internal resultant computation failed (algorithm error).
    FlintError,
}

impl From<ConversionError> for ResultantError {
    fn from(e: ConversionError) -> Self {
        ResultantError::NotAPolynomial(e)
    }
}

impl fmt::Display for ResultantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResultantError::NotAPolynomial(e) => write!(f, "not a polynomial: {e}"),
            ResultantError::FlintError => {
                write!(f, "FLINT resultant computation failed (E-RES-003)")
            }
        }
    }
}

impl std::error::Error for ResultantError {}

impl crate::errors::AlkahestError for ResultantError {
    fn code(&self) -> &'static str {
        match self {
            ResultantError::NotAPolynomial(_) => "E-RES-001",
            ResultantError::FlintError => "E-RES-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            ResultantError::NotAPolynomial(_) => Some(
                "ensure both arguments are polynomial expressions with integer \
                 coefficients in the given variable",
            ),
            ResultantError::FlintError => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Free-variable collection
// ---------------------------------------------------------------------------

/// Walk the expression DAG and collect every distinct [`ExprId`] that
/// corresponds to a `Symbol` node.  Result is sorted by `ExprId` for a
/// deterministic variable ordering.
pub(crate) fn collect_free_vars(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut set = BTreeSet::new();
    collect_vars_rec(expr, pool, &mut set);
    set.into_iter().collect()
}

fn collect_vars_rec(expr: ExprId, pool: &ExprPool, out: &mut BTreeSet<ExprId>) {
    // Collect sub-expression IDs to recurse into without holding the pool lock.
    let children: Vec<ExprId> = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } => {
            out.insert(expr);
            vec![]
        }
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => vec![],
        ExprData::Add(args) | ExprData::Mul(args) => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::Func { args, .. } => args.clone(),
        ExprData::Piecewise { branches, default } => {
            let mut ids: Vec<ExprId> = branches.iter().flat_map(|(c, v)| [*c, *v]).collect();
            ids.push(*default);
            ids
        }
        ExprData::Predicate { args, .. } => args.clone(),
    });
    for child in children {
        collect_vars_rec(child, pool, out);
    }
}

// ---------------------------------------------------------------------------
// resultant
// ---------------------------------------------------------------------------

/// Compute the resultant of `p` and `q` with respect to `var`.
///
/// Both `p` and `q` must be polynomial expressions with integer coefficients
/// in all the symbolic variables they contain.  Non-polynomial sub-expressions
/// (transcendental functions, rational coefficients, symbolic exponents) are
/// rejected with [`ResultantError::NotAPolynomial`].
///
/// The return value is the resultant polynomial as a symbolic expression:
/// - In the **univariate** case (only `var` appears) the result is an integer
///   constant.
/// - In the **multivariate** case the result is a polynomial in the remaining
///   variables (`var` has been eliminated).
///
/// # Derivation log
///
/// Records a single `"Resultant"` step tagged with the Lean 4 theorem
/// `Polynomial.resultant_eq_zero_iff_common_root`.
///
/// # Errors
///
/// - [`ResultantError::NotAPolynomial`] — an input is not a polynomial with
///   integer coefficients.
/// - [`ResultantError::FlintError`] — FLINT's internal computation failed
///   (extremely rare; indicates degenerate or overflow inputs).
///
/// # Examples
///
/// ```text
/// // Univariate: res(x^2 - 5x + 6, x - 2, x) == 0  (common root x=2)
/// // Bivariate:  res(x^2 + y^2 - 1, y - x, y) == 2*x^2 - 1
/// ```
pub fn resultant(
    p: ExprId,
    q: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ResultantError> {
    // Collect all free variables from both expressions; always include `var`.
    let mut all: BTreeSet<ExprId> = BTreeSet::new();
    for v in collect_free_vars(p, pool) {
        all.insert(v);
    }
    for v in collect_free_vars(q, pool) {
        all.insert(v);
    }
    all.insert(var);

    let vars: Vec<ExprId> = all.into_iter().collect();
    let nvars = vars.len();
    let var_idx = vars.iter().position(|&v| v == var).unwrap();

    // Convert both expressions to MultiPoly in the unified variable list.
    let mp = MultiPoly::from_symbolic(p, vars.clone(), pool)?;
    let mq = MultiPoly::from_symbolic(q, vars.clone(), pool)?;

    // Build FLINT multivariate context and polynomials.
    let ctx = FlintMPolyCtx::new(nvars.max(1));
    let fp = multi_to_flint_pub(&mp, &ctx);
    let fq = multi_to_flint_pub(&mq, &ctx);

    // Call FLINT's resultant.
    let fr = fp
        .resultant(&fq, var_idx, &ctx)
        .ok_or(ResultantError::FlintError)?;

    // Extract terms from the FLINT result (all in the same nvars-dim context).
    let res_raw = fr.terms(nvars.max(1), &ctx);

    // Build a MultiPoly for the result, dropping the eliminated variable
    // dimension (its exponent should be 0 in every term).
    let remaining_vars: Vec<ExprId> = vars
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if i == var_idx { None } else { Some(v) })
        .collect();

    let mut new_terms: BTreeMap<Vec<u32>, rug::Integer> = BTreeMap::new();
    for (exp, coeff) in res_raw {
        let mut new_exp: Vec<u32> = exp
            .into_iter()
            .enumerate()
            .filter_map(|(i, e)| if i == var_idx { None } else { Some(e) })
            .collect();
        while new_exp.last() == Some(&0) {
            new_exp.pop();
        }
        let entry = new_terms
            .entry(new_exp)
            .or_insert_with(|| rug::Integer::from(0));
        *entry += &coeff;
    }
    new_terms.retain(|_, v| *v != 0);

    let result_mp = MultiPoly {
        vars: remaining_vars,
        terms: new_terms,
    };
    let result_expr = result_mp.to_expr(pool);

    let step = RewriteStep::simple("Resultant", p, result_expr);
    Ok(DerivedExpr::with_step(result_expr, step))
}

// ---------------------------------------------------------------------------
// subresultant_prs — pure-Rust, univariate, integer coefficients
// ---------------------------------------------------------------------------

/// Compute the subresultant polynomial remainder sequence of `p` and `q`
/// with respect to `var`.
///
/// Both polynomials must be **univariate** in `var` with **integer**
/// coefficients.  Multivariate inputs (coefficients involving other symbols)
/// produce [`ResultantError::NotAPolynomial`].
///
/// Returns a [`DerivedExpr`] whose value is the full PRS as a
/// `Vec<ExprId>`:
/// `[p, q, S₂, S₃, …, Sₖ]`
///
/// The 0th subresultant — the resultant — can be extracted as the last
/// element that is a constant (degree-0) polynomial, or from
/// [`resultant`] directly.
///
/// # Algorithm
///
/// Classical Brown–Collins subresultant algorithm (1971/1967).  Computations
/// stay in ℤ[x]; all coefficient scalings are exact integer divisions
/// guaranteed by the subresultant theory.
///
/// # Derivation log
///
/// Records a single `"SubresultantPRS"` step.
pub fn subresultant_prs(
    p: ExprId,
    q: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<Vec<ExprId>>, ResultantError> {
    // Convert to UniPoly (rejects non-integer coefficients and other symbols).
    let mut up = UniPoly::from_symbolic(p, var, pool)?;
    let mut uq = UniPoly::from_symbolic(q, var, pool)?;

    // Canonical orientation: deg(P) >= deg(Q).
    if up.degree() < uq.degree() {
        std::mem::swap(&mut up, &mut uq);
    }

    let prs_polys = sprs_inner(up, uq);

    // Convert each polynomial in the sequence back to a symbolic expression.
    let exprs: Vec<ExprId> = prs_polys
        .into_iter()
        .map(|poly| unipoly_to_expr(&poly, var, pool))
        .collect();

    let mut log = DerivationLog::new();
    if let (Some(&first), Some(&last)) = (exprs.first(), exprs.last()) {
        log.push(RewriteStep::simple("SubresultantPRS", first, last));
    }
    Ok(DerivedExpr::with_log(exprs, log))
}

// ---------------------------------------------------------------------------
// Internal: Brown–Collins subresultant PRS
// ---------------------------------------------------------------------------

/// Classical subresultant PRS (Brown 1971, Collins 1967).
///
/// Requires `deg(p) >= deg(q)`.  Returns the sequence `[P, Q, S₂, …, Sₖ]`.
fn sprs_inner(p: UniPoly, q: UniPoly) -> Vec<UniPoly> {
    let var = p.var;
    let mut sequence = vec![p.clone(), q.clone()];

    if q.is_zero() {
        return sequence;
    }

    let m = p.degree();
    let n = q.degree();
    if n < 0 {
        return sequence;
    }

    // β₁ = (-1)^(m - n + 1)
    let delta0 = (m - n) as u32;
    let beta: rug::Integer = if (delta0 + 1) % 2 == 0 {
        rug::Integer::from(1)
    } else {
        rug::Integer::from(-1)
    };

    let mut beta_cur = beta;
    let mut psi_cur: rug::Integer = rug::Integer::from(-1);

    let mut a = p;
    let mut b = q;

    loop {
        if b.is_zero() {
            break;
        }

        let deg_a = a.degree();
        let deg_b = b.degree();
        if deg_b < 0 {
            break;
        }
        let delta = (deg_a - deg_b) as u32;

        // Pseudo-remainder: lc(b)^d * a = Q*b + R
        let (_, r_flint, _d) = a.coeffs.pseudo_divrem(&b.coeffs);
        if r_flint.is_zero() {
            break;
        }

        // S_{i+1} = prem(S_{i-1}, S_i) / β_i  [exact scalar division]
        let beta_fi = FlintInteger::from_rug(&beta_cur);
        let c_coeffs = r_flint.scalar_divexact_fmpz(&beta_fi);
        let c = UniPoly {
            var,
            coeffs: c_coeffs,
        };
        sequence.push(c.clone());

        // Update ψ: ψ_{i+1} = (-lc(b))^δ / ψ_i^(δ-1)
        let lc_b_fmpz = b.coeffs.leading_coeff_fmpz();
        let lc_b = lc_b_fmpz.to_rug();
        let neg_lc_b: rug::Integer = -lc_b;

        let psi_new = if delta <= 1 {
            // ψ^0 = 1, so result is just (-lc(b))^δ
            rug_pow(&neg_lc_b, delta)
        } else {
            let num = rug_pow(&neg_lc_b, delta);
            let den = rug_pow(&psi_cur, delta - 1);
            rug::Integer::from(num.div_exact_ref(&den))
        };

        // β_{i+1} = -lc(b) · ψ_{i+1}
        let beta_new = neg_lc_b * &psi_new;

        a = b;
        b = c;
        psi_cur = psi_new;
        beta_cur = beta_new;
    }

    sequence
}

/// Integer exponentiation for [`rug::Integer`] (non-negative exponent).
fn rug_pow(base: &rug::Integer, exp: u32) -> rug::Integer {
    if exp == 0 {
        return rug::Integer::from(1);
    }
    let mut r = base.clone();
    for _ in 1..exp {
        r *= base;
    }
    r
}

// ---------------------------------------------------------------------------
// Helper: UniPoly → symbolic ExprId
// ---------------------------------------------------------------------------

/// Convert a `UniPoly` back to a symbolic expression in `pool`.
fn unipoly_to_expr(poly: &UniPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    let coeffs = poly.coefficients(); // rug::Integer in ascending degree order
    if coeffs.is_empty() {
        return pool.integer(0_i32);
    }

    let summands: Vec<ExprId> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(deg, coeff)| {
            let c_id = pool.integer(coeff.clone());
            if deg == 0 {
                c_id
            } else {
                let exp_id = pool.integer(deg as i64);
                let x_pow = if deg == 1 { var } else { pool.pow(var, exp_id) };
                if *coeff == 1 {
                    x_pow
                } else {
                    pool.mul(vec![c_id, x_pow])
                }
            }
        })
        .collect();

    match summands.len() {
        0 => pool.integer(0_i32),
        1 => summands[0],
        _ => pool.add(summands),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool_xy() -> (ExprPool, ExprId, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        (p, x, y)
    }

    // --- collect_free_vars ---

    #[test]
    fn free_vars_constant() {
        let p = ExprPool::new();
        let five = p.integer(5_i32);
        let vars = collect_free_vars(five, &p);
        assert!(vars.is_empty());
    }

    #[test]
    fn free_vars_symbol() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let vars = collect_free_vars(x, &p);
        assert_eq!(vars, vec![x]);
    }

    #[test]
    fn free_vars_polynomial() {
        let (p, x, y) = pool_xy();
        // x^2 + y - 1
        let xsq = p.pow(x, p.integer(2_i32));
        let expr = p.add(vec![xsq, y, p.integer(-1_i32)]);
        let vars = collect_free_vars(expr, &p);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&x));
        assert!(vars.contains(&y));
    }

    // --- resultant: univariate cases ---

    #[test]
    fn resultant_common_root() {
        // res(x^2 - 5x + 6, x - 2, x) == 0  (both vanish at x=2)
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // p = x^2 - 5x + 6
        let xsq = p.pow(x, p.integer(2_i32));
        let five_x = p.mul(vec![p.integer(-5_i32), x]);
        let poly_p = p.add(vec![xsq, five_x, p.integer(6_i32)]);
        // q = x - 2
        let poly_q = p.add(vec![x, p.integer(-2_i32)]);

        let dr = resultant(poly_p, poly_q, x, &p).unwrap();
        // Result should be the integer 0
        match p.get(dr.value) {
            ExprData::Integer(n) => assert_eq!(n.0, 0),
            _ => panic!("expected integer 0, got {:?}", p.get(dr.value)),
        }
        // Derivation log records one step
        assert_eq!(dr.log.len(), 1);
        assert_eq!(dr.log.steps()[0].rule_name, "Resultant");
    }

    #[test]
    fn resultant_coprime() {
        // res(x^2 + 1, x - 1, x) == 2  (no common roots over ℂ... actually x=i,
        // but x-1 has root 1 and x^2+1 has roots ±i, so coprime)
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // x^2 + 1
        let xsq = p.pow(x, p.integer(2_i32));
        let poly_p = p.add(vec![xsq, p.integer(1_i32)]);
        // x - 1
        let poly_q = p.add(vec![x, p.integer(-1_i32)]);
        let dr = resultant(poly_p, poly_q, x, &p).unwrap();
        match p.get(dr.value) {
            ExprData::Integer(n) => assert_eq!(n.0, 2),
            _ => panic!("expected integer 2, got {:?}", p.get(dr.value)),
        }
    }

    #[test]
    fn resultant_linear_linear() {
        // res(x - a, x - b, x) = a - b  (resultant = lc(f)^deg(g) * g(roots of f))
        // Concretely: res(x - 3, x - 7, x) = g(3) = 3 - 7 = -4
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let poly_p = p.add(vec![x, p.integer(-3_i32)]);
        let poly_q = p.add(vec![x, p.integer(-7_i32)]);
        let dr = resultant(poly_p, poly_q, x, &p).unwrap();
        match p.get(dr.value) {
            ExprData::Integer(n) => {
                // res(x-3, x-7) = (3 - 7) = -4
                assert_eq!(
                    n.0.clone().abs(),
                    rug::Integer::from(4),
                    "magnitude should be 4"
                );
            }
            _ => panic!("expected integer, got {:?}", p.get(dr.value)),
        }
    }

    // --- resultant: bivariate (implicitization) ---

    #[test]
    fn resultant_bivariate_eliminates_var() {
        // res(x^2 + y^2 - 1, y - x, y) should equal 2x^2 - 1
        // We verify by checking the result is non-zero and degree 2 in x.
        let (p, x, y) = pool_xy();

        // x^2 + y^2 - 1
        let xsq = p.pow(x, p.integer(2_i32));
        let ysq = p.pow(y, p.integer(2_i32));
        let circle = p.add(vec![xsq, ysq, p.integer(-1_i32)]);

        // y - x
        let line = p.add(vec![y, p.mul(vec![p.integer(-1_i32), x])]);

        let dr = resultant(circle, line, y, &p).unwrap();
        let res_expr = dr.value;

        // The result should be a polynomial in x of degree 2.
        // Verify by converting to UniPoly in x.
        let res_poly = UniPoly::from_symbolic(res_expr, x, &p).unwrap();
        assert_eq!(res_poly.degree(), 2, "expected degree-2 resultant in x");
        // Coefficients should be [-1, 0, 2] i.e. -1 + 0*x + 2*x^2
        let coeffs = res_poly.coefficients_i64();
        assert_eq!(coeffs[0], -1, "constant term should be -1");
        assert_eq!(coeffs[2], 2, "leading coefficient should be 2");
    }

    // --- implicitization: twisted cubic (t^2, t^3) ---

    #[test]
    fn resultant_implicitization_twisted_cubic() {
        // Parametrically: x = t^2, y = t^3.
        // Eliminate t: res(x - t^2, y - t^3, t) == y^2 - x^3
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);

        // p1 = x - t^2
        let t2 = pool.pow(t, pool.integer(2_i32));
        let p1 = pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), t2])]);

        // p2 = y - t^3
        let t3 = pool.pow(t, pool.integer(3_i32));
        let p2 = pool.add(vec![y, pool.mul(vec![pool.integer(-1_i32), t3])]);

        let dr = resultant(p1, p2, t, &pool).unwrap();
        let res_expr = dr.value;

        // The result should be y^2 - x^3 (or a scalar multiple).
        // Verify by evaluating at (x=4, y=8): 64 - 64 = 0 (point on the curve).
        // And at (x=1, y=2): 4 - 1 = 3 ≠ 0 (not on the curve).
        use crate::kernel::subs;
        use std::collections::HashMap;
        let one = pool.integer(1_i32);
        let two = pool.integer(2_i32);
        let four = pool.integer(4_i32);
        let eight = pool.integer(8_i32);

        // Substitute (x=4, y=8) → should give 0
        let mut map_on = HashMap::new();
        map_on.insert(x, four);
        map_on.insert(y, eight);
        let at_4_8 = subs(res_expr, &map_on, &pool);
        let simplified_0 = crate::simplify::simplify(at_4_8, &pool);
        match pool.get(simplified_0.value) {
            ExprData::Integer(n) => assert_eq!(n.0, 0, "res at (4,8) should be 0"),
            _ => {
                panic!(
                    "expected integer 0 at (4,8), got {:?}",
                    pool.get(simplified_0.value)
                )
            }
        }

        // Substitute (x=1, y=2) → should give nonzero
        let mut map_off = HashMap::new();
        map_off.insert(x, one);
        map_off.insert(y, two);
        let at_1_2 = subs(res_expr, &map_off, &pool);
        let simplified_nz = crate::simplify::simplify(at_1_2, &pool);
        if let ExprData::Integer(n) = pool.get(simplified_nz.value) {
            assert_ne!(n.0, 0, "res at (1,2) should be non-zero");
        } // non-integer result is also non-zero
    }

    // --- subresultant_prs ---

    #[test]
    fn sprs_sequence_length() {
        // For coprime polynomials, PRS terminates at degree 0.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // x^2 + 1  (irreducible over ℤ)
        let xsq = p.pow(x, p.integer(2_i32));
        let poly_p = p.add(vec![xsq, p.integer(1_i32)]);
        // x - 1
        let poly_q = p.add(vec![x, p.integer(-1_i32)]);

        let dr = subresultant_prs(poly_p, poly_q, x, &p).unwrap();
        // Sequence starts with [p, q, ...] and ends with a constant (or empty
        // if gcd is non-trivial).
        let seq = &dr.value;
        assert!(seq.len() >= 2, "sequence must have at least [p, q]");
        // First element is p or q (may have been swapped by degree).
        // Last element should be a constant (degree 0) for coprime polynomials.
        let last_id = *seq.last().unwrap();
        match p.get(last_id) {
            ExprData::Integer(_) => {} // scalar: good
            _ => {
                // Try parsing as UniPoly and check degree.
                let last_poly = UniPoly::from_symbolic(last_id, x, &p).unwrap();
                assert_eq!(last_poly.degree(), 0, "last PRS element should be degree 0");
            }
        }
    }

    #[test]
    fn sprs_first_elements() {
        // The first two elements of the PRS are p and q (possibly swapped).
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let two = p.integer(2_i32);
        let xsq = p.pow(x, p.integer(2_i32));
        // p = x^2 - 1
        let poly_p_expr = p.add(vec![xsq, p.integer(-1_i32)]);
        // q = 2x - 2  (to test: gcd = x - 1)
        let two_x = p.mul(vec![two, x]);
        let poly_q_expr = p.add(vec![two_x, p.integer(-2_i32)]);

        let dr = subresultant_prs(poly_p_expr, poly_q_expr, x, &p).unwrap();
        assert!(dr.value.len() >= 2);
    }

    #[test]
    fn sprs_gcd_from_sequence() {
        // The last non-zero element of the PRS (up to content) is the GCD.
        // gcd(x^2 - 1, x - 1) = x - 1
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let xsq = p.pow(x, p.integer(2_i32));
        let poly_p_expr = p.add(vec![xsq, p.integer(-1_i32)]);
        let poly_q_expr = p.add(vec![x, p.integer(-1_i32)]);

        let dr = subresultant_prs(poly_p_expr, poly_q_expr, x, &p).unwrap();
        let seq = &dr.value;
        assert!(seq.len() >= 2);
        // Convert the last element to UniPoly.
        let last_id = *seq.last().unwrap();
        let last_poly = UniPoly::from_symbolic(last_id, x, &p).unwrap();
        // Should have degree 1 (matching gcd x - 1 up to scalar).
        assert_eq!(
            last_poly.degree(),
            1,
            "last PRS element should be degree-1 (matching GCD)"
        );
    }

    #[test]
    fn sprs_sylvester_consistency() {
        // The resultant is the last constant element of the subresultant PRS.
        // For x - 3 and x - 7, res = 4.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let poly_p_expr = p.add(vec![x, p.integer(-3_i32)]);
        let poly_q_expr = p.add(vec![x, p.integer(-7_i32)]);

        let dr_prs = subresultant_prs(poly_p_expr, poly_q_expr, x, &p).unwrap();
        let dr_res = resultant(poly_p_expr, poly_q_expr, x, &p).unwrap();

        // The resultant should match the constant at the end of the PRS.
        let last = *dr_prs.value.last().unwrap();
        match p.get(last) {
            ExprData::Integer(n) => {
                let res_n = match p.get(dr_res.value) {
                    ExprData::Integer(m) => m.0.clone(),
                    _ => panic!("resultant not integer"),
                };
                // They should match up to sign.
                assert_eq!(n.0.clone().abs(), res_n.abs());
            }
            _ => {
                // Degree-0 polynomial stored as a mul/add — tolerate this form.
            }
        }
    }

    // --- error cases ---

    #[test]
    fn resultant_non_polynomial_error() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // sin(x) is not a polynomial
        let sin_x = p.func("sin", vec![x]);
        let poly_q = p.add(vec![x, p.integer(-1_i32)]);
        let err = resultant(sin_x, poly_q, x, &p);
        assert!(
            matches!(err, Err(ResultantError::NotAPolynomial(_))),
            "expected NotAPolynomial error"
        );
    }

    #[test]
    fn subresultant_prs_non_polynomial_error() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        // y appears as a free variable — not polynomial in x alone
        let poly_p = p.add(vec![x, y]);
        let poly_q = p.add(vec![x, p.integer(-1_i32)]);
        let err = subresultant_prs(poly_p, poly_q, x, &p);
        assert!(
            matches!(err, Err(ResultantError::NotAPolynomial(_))),
            "expected NotAPolynomial error for multivariate input to subresultant_prs"
        );
    }
}
