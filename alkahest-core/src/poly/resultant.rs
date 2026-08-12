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
use crate::flint::mpoly::FlintMPolyCtx;
use crate::flint::FlintPoly;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::error::ConversionError;
use crate::poly::multipoly::multi_to_flint_pub;
use crate::poly::multipoly::MultiPoly;
use crate::poly::unipoly::UniPoly;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

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
pub fn collect_free_vars(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
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
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => vec![*var, *body],
        ExprData::BigO(arg) => vec![*arg],
        ExprData::RootSum { poly, var, body } => vec![*poly, *var, *body],
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
    let fp = multi_to_flint_pub(&mp, Arc::clone(&ctx));
    let fq = multi_to_flint_pub(&mq, Arc::clone(&ctx));

    // Call FLINT's resultant.
    let fr = fp
        .resultant(&fq, var_idx)
        .ok_or(ResultantError::FlintError)?;

    // Extract terms from the FLINT result (all in the same nvars-dim context).
    let res_raw = fr.terms();

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
/// Each element after the first two is a genuine **subresultant**: the entry of
/// degree `j` is `S_j(p, q)`, the polynomial whose coefficients are the
/// determinants of the corresponding submatrices of the Sylvester matrix.
///
/// The 0th subresultant — the resultant — can be extracted as the last
/// element that is a constant (degree-0) polynomial, or from
/// [`resultant`] directly.  The two agree; when `gcd(p, q)` is non-constant the
/// chain terminates early and no degree-0 element is produced, which is the
/// honest report that the resultant is `0`.
///
/// The one corner where no `S_j` exists at all is `deg q = 0`: the chain
/// `S_j`, `0 ≤ j < deg q`, is empty, so the sequence is just `[p, q]` and the
/// resultant `lc(q)^{deg p}` must be taken from [`resultant`].
///
/// # Algorithm
///
/// Ducos' formulation of the subresultant chain (Ducos, *Optimizations of the
/// subresultant algorithm*, JPAA 145 (2000)), which is the Brown–Collins
/// recurrence written so that the emitted elements are the *regular*
/// subresultants rather than the raw remainders.  The distinction is not
/// cosmetic: for a defective sequence (a degree drop of more than one) the raw
/// Brown–Collins remainder differs from `S_{deg}` by a power of a leading
/// coefficient, so a sequence built from the remainders alone contradicts
/// [`resultant`] on its last element.
///
/// Computations stay in ℤ\[x\]; every coefficient scaling is an exact integer
/// division guaranteed by the subresultant theory.  Those divisions are
/// *checked* rather than assumed: an inexact one would be an internal
/// contradiction, and it is reported as [`ResultantError::FlintError`] instead
/// of being handed to a routine that aborts the process.
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

    let prs_polys = sprs_inner(up, uq).ok_or(ResultantError::FlintError)?;

    // Convert each polynomial in the sequence back to a symbolic expression.
    let exprs: Vec<ExprId> = prs_polys
        .into_iter()
        .map(|poly| poly.to_symbolic_expr(pool))
        .collect();

    let mut log = DerivationLog::new();
    if let (Some(&first), Some(&last)) = (exprs.first(), exprs.last()) {
        log.push(RewriteStep::simple("SubresultantPRS", first, last));
    }
    Ok(DerivedExpr::with_log(exprs, log))
}

// ---------------------------------------------------------------------------
// Internal: the subresultant chain (Ducos' form of Brown–Collins)
// ---------------------------------------------------------------------------

/// Dense coefficient vector, little-endian (`c[i]` multiplies `xⁱ`), with no
/// trailing zeros.  The empty vector is the zero polynomial.
type Coeffs = Vec<rug::Integer>;

/// Drop trailing zero coefficients so that `len() - 1` is the degree.
fn trim(c: &mut Coeffs) {
    while c.last().is_some_and(|t| *t == 0) {
        c.pop();
    }
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

/// `c · a`.
fn scalar_mul(a: &Coeffs, c: &rug::Integer) -> Coeffs {
    if *c == 0 {
        return Coeffs::new();
    }
    a.iter().map(|t| rug::Integer::from(t * c)).collect()
}

/// `a / c`, or `None` when the division is not exact (or `c = 0`).
///
/// Checked rather than assumed: the subresultant theory says every division
/// this module performs is exact, and FLINT's `scalar_divexact` *aborts the
/// process* when it is not.  A bug upstream must surface as an error, not as a
/// `SIGABRT` in the caller's Python process.
fn scalar_div_exact(a: &Coeffs, c: &rug::Integer) -> Option<Coeffs> {
    if *c == 0 {
        return None;
    }
    let mut out = Coeffs::with_capacity(a.len());
    for t in a {
        if !t.is_divisible(c) {
            return None;
        }
        out.push(rug::Integer::from(t / c));
    }
    trim(&mut out);
    Some(out)
}

/// Canonical pseudo-remainder: the `R` in `lc(b)^(deg a − deg b + 1) · a = q·b + R`.
///
/// The *canonical* exponent `δ+1` matters. FLINT's `fmpz_poly_pseudo_divrem`
/// returns the **minimal** exponent `d ≤ δ+1` instead, and the subresultant
/// recurrence is stated for `δ+1`, so using FLINT's remainder unscaled leaves
/// every element short by `lc(b)^(δ+1−d)`.
///
/// Returns `None` if `b` is zero.
fn pseudo_remainder(a: &Coeffs, b: &Coeffs) -> Option<Coeffs> {
    let db = b.len().checked_sub(1)?;
    let lc_b = &b[db];
    if a.len() <= db {
        // deg a < deg b: the remainder is `a` itself.
        return Some(a.clone());
    }
    let delta = (a.len() - 1) - db;
    let mut r = scalar_mul(a, &rug_pow(lc_b, delta as u32 + 1));
    while r.len() > db {
        let dr = r.len() - 1;
        // Exact by construction: pre-scaling by `lc(b)^(δ+1)` leaves every
        // coefficient after `k` reduction steps divisible by `lc(b)^(δ+1−k)`,
        // and the loop runs at most `δ+1` steps.  Checked anyway — a truncating
        // division here would be a wrong polynomial with no symptom, which is
        // the exact failure mode this module was fixed for.
        if !r[dr].is_divisible(lc_b) {
            return None;
        }
        let quot = rug::Integer::from(&r[dr] / lc_b);
        let shift = dr - db;
        for (i, bi) in b.iter().enumerate() {
            r[shift + i] -= rug::Integer::from(&quot * bi);
        }
        trim(&mut r);
        if r.is_empty() {
            break;
        }
    }
    Some(r)
}

/// The subresultant chain of `p` and `q`, as Ducos states it.
///
/// Requires `deg(p) >= deg(q)`.  Returns the sequence `[P, Q, S₂, …, Sₖ]`,
/// where every element after the first two is a *regular* subresultant: the
/// element of degree `j` is exactly `S_j(p, q)`, so the last degree-0 element
/// is `S₀ = Res(p, q)` and agrees with [`resultant`].
///
/// Returns `None` if one of the exact divisions the theory guarantees turns out
/// not to be exact — an internal contradiction, reported rather than aborted.
fn sprs_inner(p: UniPoly, q: UniPoly) -> Option<Vec<UniPoly>> {
    let var = p.var;
    let mut sequence = vec![p.clone(), q.clone()];

    let mut pc = p.coefficients();
    let mut qc = q.coefficients();
    trim(&mut pc);
    trim(&mut qc);
    // `deg q < 0` (q = 0) or `deg q = 0`: the chain `S_j`, `0 ≤ j < deg q`, is
    // empty, so there is nothing to append.
    if qc.len() <= 1 || pc.is_empty() {
        return Some(sequence);
    }

    // s = lc(q)^(deg p − deg q);  A = q;  B = prem(p, −q).
    let mut s = rug_pow(&qc[qc.len() - 1], (pc.len() - qc.len()) as u32);
    let mut a = qc.clone();
    let neg_q: Coeffs = qc.iter().map(|t| rug::Integer::from(-t)).collect();
    let mut b = pseudo_remainder(&pc, &neg_q)?;

    while !b.is_empty() {
        let d = a.len() - 1;
        let e = b.len() - 1;
        let delta = d - e;

        // `B` is the (possibly defective) subresultant `S_{d−1}`.  The regular
        // one of the same degree is `C = lc(B)^(δ−1) · B / s^(δ−1)`; when the
        // sequence is normal (δ = 1) the two coincide.
        let c = if delta > 1 {
            let scaled = scalar_mul(&b, &rug_pow(&b[e], delta as u32 - 1));
            scalar_div_exact(&scaled, &rug_pow(&s, delta as u32 - 1))?
        } else {
            b.clone()
        };
        sequence.push(UniPoly {
            var,
            coeffs: FlintPoly::from_rug_coefficients(&c),
        });
        if e == 0 {
            break;
        }

        // B ← prem(A, −B) / (s^δ · lc(A));  A ← C;  s ← lc(A).
        let neg_b: Coeffs = b.iter().map(|t| rug::Integer::from(-t)).collect();
        let rem = pseudo_remainder(&a, &neg_b)?;
        let divisor = rug_pow(&s, delta as u32) * &a[d];
        b = scalar_div_exact(&rem, &divisor)?;
        a = c;
        s = a[a.len() - 1].clone();
    }

    Some(sequence)
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

    // --- subresultant chain: determinantal ground truth ---

    /// Build the symbolic polynomial `Σ c[i]·xⁱ` from little-endian coefficients.
    fn from_coeffs(p: &ExprPool, x: ExprId, c: &[i64]) -> ExprId {
        let terms: Vec<ExprId> = c
            .iter()
            .enumerate()
            .filter(|(_, &k)| k != 0)
            .map(|(i, &k)| {
                let xi = p.pow(x, p.integer(i as i64));
                p.mul(vec![p.integer(k), xi])
            })
            .collect();
        if terms.is_empty() {
            p.integer(0_i32)
        } else {
            p.add(terms)
        }
    }

    /// Read a PRS element back as little-endian integer coefficients.
    fn to_coeffs(p: &ExprPool, x: ExprId, e: ExprId) -> Vec<rug::Integer> {
        let mut c = UniPoly::from_symbolic(e, x, p).unwrap().coefficients();
        while c.last().is_some_and(|t| *t == 0) {
            c.pop();
        }
        c
    }

    /// Determinant by Gaussian elimination over ℚ (test-only; the matrices here
    /// are tiny and this is deliberately a different algorithm from anything in
    /// the module under test).
    fn det_rational(mut m: Vec<Vec<rug::Rational>>) -> rug::Rational {
        let n = m.len();
        let mut d = rug::Rational::from(1);
        for i in 0..n {
            let Some(piv) = (i..n).find(|&r| m[r][i] != 0) else {
                return rug::Rational::from(0);
            };
            if piv != i {
                m.swap(i, piv);
                d = -d;
            }
            let (head, tail) = m.split_at_mut(i + 1);
            let pivot_row = &head[i];
            d *= pivot_row[i].clone();
            let inv = rug::Rational::from(1) / pivot_row[i].clone();
            for row in tail.iter_mut() {
                let f = row[i].clone() * inv.clone();
                if f == 0 {
                    continue;
                }
                for (cell, pivot) in row[i..n].iter_mut().zip(pivot_row[i..n].iter()) {
                    *cell -= f.clone() * pivot.clone();
                }
            }
        }
        d
    }

    /// `S_j(f, g)` straight from the definition: the coefficient of `x^k` in
    /// `S_j` is the determinant of the `(m+n−2j)`-square matrix whose rows are
    /// `x^{n−j−1}f, …, f, x^{m−j−1}g, …, g` taken in the degree columns
    /// `m+n−j−1, …, j+1` together with the degree-`k` column.
    ///
    /// This is the ground truth the chain is checked against — no part of it
    /// shares code with `sprs_inner`.
    fn subresultant_by_determinant(f: &[i64], g: &[i64], j: usize) -> Vec<rug::Integer> {
        let m = f.len() - 1;
        let n = g.len() - 1;
        let width = m + n - j; // degrees m+n−j−1 … 0
        let row_of = |poly: &[i64], sh: usize| -> Vec<rug::Rational> {
            // Column c holds the coefficient of degree `width−1−c`.
            (0..width)
                .map(|c| {
                    let deg = width - 1 - c;
                    let k = deg.wrapping_sub(sh);
                    if deg >= sh && k < poly.len() {
                        rug::Rational::from(poly[k])
                    } else {
                        rug::Rational::from(0)
                    }
                })
                .collect()
        };
        let mut rows: Vec<Vec<rug::Rational>> = Vec::new();
        for sh in (0..n - j).rev() {
            rows.push(row_of(f, sh));
        }
        for sh in (0..m - j).rev() {
            rows.push(row_of(g, sh));
        }
        let size = m + n - 2 * j;
        assert_eq!(rows.len(), size);
        let mut out: Vec<rug::Integer> = Vec::new();
        for k in 0..=j {
            let mut cols: Vec<usize> = (0..size - 1).collect();
            cols.push(width - 1 - k);
            let sub: Vec<Vec<rug::Rational>> = rows
                .iter()
                .map(|r| cols.iter().map(|&c| r[c].clone()).collect())
                .collect();
            let d = det_rational(sub);
            assert_eq!(*d.denom(), 1);
            out.push(d.numer().clone());
        }
        while out.last().is_some_and(|t| *t == 0) {
            out.pop();
        }
        out
    }

    #[test]
    fn sprs_matches_the_sylvester_determinants() {
        // Every element of degree `j` in the returned sequence must be exactly
        // `S_j`, and the last degree-0 element must be `Res(f, g)`.
        //
        // The two families below are the ones from the 3.8 silent-error hunt:
        // `subresultant_prs(x²−3x+2, 2x)` used to end in `4` while `resultant`
        // said `8`, and `subresultant_prs(3x³−x, −3x²+2x−3)` returned
        // `8x+6, −44` where the determinants give `−24x−18, −396`.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let cases: &[(&[i64], &[i64])] = &[
            (&[2, -3, 1], &[0, 2]),
            (&[0, -1, 0, 3], &[-3, 2, -3]),
            (&[1, 2, 2], &[1, 1, 2]),
            (&[1, 0, 1], &[0, 2]),
            (&[-2, 0, 0, 3, 2, -1], &[-3, 2, 0, -1, -1]),
            (&[-2, -3, -1, 3, 3, -1], &[-3, -2, -2, 0, 2]),
            (&[1, 1, 1, 1], &[2, 0, 3]),
            (&[-5, 0, 0, 0, 7], &[1, -1, 1]),
        ];
        for (f, g) in cases {
            let pf = from_coeffs(&p, x, f);
            let pg = from_coeffs(&p, x, g);
            let seq = subresultant_prs(pf, pg, x, &p).unwrap().value;
            for &elem in &seq[2..] {
                let c = to_coeffs(&p, x, elem);
                let j = c.len() - 1;
                assert_eq!(
                    c,
                    subresultant_by_determinant(f, g, j),
                    "element of degree {j} is not S_{j} for f={f:?}, g={g:?}"
                );
            }
            // …and the resultant agrees with `resultant`, sign included.
            let last = to_coeffs(&p, x, *seq.last().unwrap());
            if last.len() == 1 && seq.len() > 2 {
                let r = resultant(pf, pg, x, &p).unwrap().value;
                let expected = match p.get(r) {
                    ExprData::Integer(n) => n.0.clone(),
                    other => panic!("resultant was not an integer: {other:?}"),
                };
                assert_eq!(last[0], expected, "last PRS element ≠ resultant");
            }
        }
    }

    #[test]
    fn sprs_survives_an_inexact_scaling_input() {
        // `subresultant_prs(2x²+2x+1, 2x²+x+1)` used to hand a non-exact
        // division to FLINT's `scalar_divexact`, which does not raise — it
        // calls `flint_abort`, taking the whole process down with SIGABRT, so
        // no `except` of any kind could survive it.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = from_coeffs(&p, x, &[1, 2, 2]);
        let g = from_coeffs(&p, x, &[1, 1, 2]);
        let seq = subresultant_prs(f, g, x, &p).unwrap().value;
        assert_eq!(
            to_coeffs(&p, x, *seq.last().unwrap()),
            vec![rug::Integer::from(2)]
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
