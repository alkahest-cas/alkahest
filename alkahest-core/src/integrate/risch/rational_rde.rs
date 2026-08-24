//! Rational Risch Differential Equation (RDE) solver over ℚ(x) (Risch Gap 1).
//!
//! Extends [`super::poly_rde`] from polynomial coefficients to **rational**
//! coefficients.  Solves
//! ```text
//!   v'(x) + f(x)·v(x) = c(x)
//! ```
//! where `f ∈ ℚ[x]` (in the exp tower, `f = k·η'`, always a polynomial) and
//! `c ∈ ℚ(x)` is a rational function, returning `v ∈ ℚ(x)` when one exists.
//!
//! ## Algorithm (Bronstein 2005, §6.1)
//!
//! Because `f` is a polynomial (no poles), a Laurent-series analysis at any pole
//! `α` of `v` shows that `v'` raises the pole order by one while `f·v` keeps it,
//! so the pole order of `v` at `α` is exactly one less than that of `c`.  Hence
//! the denominator of any rational solution divides
//! ```text
//!   E = gcd(B, B')
//! ```
//! where `B` is the (reduced, monic) denominator of `c`.  Writing `v = N/E` for
//! an unknown polynomial `N`, substituting, and clearing denominators yields a
//! **polynomial identity** that is *linear* in the coefficients of `N`:
//! ```text
//!   G·E·N' − G·E'·N + G·E·f·N − C·E = 0,     G = B/E,  c = C/B.
//! ```
//! Equating coefficients gives a linear system over ℚ.  A consistent system
//! certifies an elementary antiderivative `v·exp(kη)`; an inconsistent one
//! certifies that the integral is **non-elementary** (the leftover simple-pole
//! residues are exactly the Ei/Li-type logarithmic part the exp tower cannot
//! express).
//!
//! A final substitution check guards the positive direction; the negative
//! direction is guarded by the type — see below.
//!
//! ## When `f` has poles: the resonance, and why the answer is three-valued
//!
//! The `E = gcd(B, B')` bound above is exact **only because `f` is a
//! polynomial**.  For [`solve_rational_rde_generalized_checked`], where
//! `f ∈ ℚ(x)` may itself have poles, it is not: at a *simple* pole of `f` with
//! residue `ρ ∈ ℤ_{>0}`, the leading terms of `v'` and `f·v` cancel and `v`
//! acquires a pole of order `ρ` at a point `c` is regular at.  `v' + (2/x+1)·v = 1`
//! (which is `∫x²eˣ`) has `c = 1`, so `B = E = 1`, yet its solution
//! `v = (x²−2x+2)/x²` has a double pole at `0`.  The fix is
//! `resonant_denominator` (below), which carries the pole-order argument in full.
//!
//! Because a *missing* solution and a *nonexistent* solution are different
//! claims — and only the second may license a non-elementarity certificate —
//! the solvers return the three-valued [`RdeOutcome`] rather than an `Option`.
//! The `Option`-returning entry points are retained as shims and documented as
//! unsafe to conclude from.
//!
//! References:
//!   - Bronstein (2005). *Symbolic Integration I*, §6.1 (RischDE, normal part;
//!     `WeakNormalizer`, `RdeNormalDenominator`).
//!   - SymPy `sympy/integrals/rde.py` (`bound_degree`, `spde`, `no_cancel_*`).

use rug::Rational;

use super::number_field::{KElem, KPoly, NumberField};
use super::poly_rde::{
    degree, poly_add, poly_deriv, poly_mul, poly_one, poly_scale, poly_zero, trim, QPoly,
};

// ---------------------------------------------------------------------------
// Polynomial arithmetic over ℚ not already provided by `poly_rde`
// ---------------------------------------------------------------------------

/// Subtract `b` from `a`.
pub fn poly_sub(a: &QPoly, b: &QPoly) -> QPoly {
    poly_add(a, &poly_scale(b, &Rational::from(-1)))
}

/// Coefficient of `x^i` (0 outside the stored range or for negative `i`).
fn coeff(p: &QPoly, i: i64) -> Rational {
    if i < 0 {
        return Rational::from(0);
    }
    p.get(i as usize)
        .cloned()
        .unwrap_or_else(|| Rational::from(0))
}

/// Long division over ℚ: returns `(q, r)` with `a = q·b + r`, `deg r < deg b`.
/// `b` must be nonzero.
pub fn poly_divrem(a: &QPoly, b: &QPoly) -> (QPoly, QPoly) {
    let b = trim(b.clone());
    let bd = degree(&b);
    debug_assert!(bd >= 0, "poly_divrem: division by zero polynomial");
    let lcb = b[bd as usize].clone();

    let mut r = trim(a.clone());
    let ad = degree(&r);
    if ad < bd {
        return (poly_zero(), r);
    }
    let mut q = vec![Rational::from(0); (ad - bd + 1) as usize];

    loop {
        let rd = degree(&r);
        if rd < bd {
            break;
        }
        let shift = (rd - bd) as usize;
        let factor = r[rd as usize].clone() / lcb.clone();
        q[shift] += factor.clone();
        for (i, bc) in b.iter().enumerate() {
            r[shift + i] -= factor.clone() * bc.clone();
        }
        r = trim(r);
        if r.is_empty() {
            break;
        }
    }
    (trim(q), trim(r))
}

/// Make a polynomial monic (leading coefficient 1).  The zero polynomial is
/// returned unchanged.
pub fn poly_monic(p: &QPoly) -> QPoly {
    let p = trim(p.clone());
    let d = degree(&p);
    if d < 0 {
        return p;
    }
    let lc = p[d as usize].clone();
    poly_scale(&p, &(Rational::from(1) / lc))
}

/// Clear denominators: the primitive integer associate of a `ℚ`-polynomial,
/// as a FLINT `fmpz_poly`.  Scaling by a nonzero rational does not change a
/// *monic* GCD, so this is lossless for [`poly_gcd`]'s purposes.
fn qpoly_to_fmpz(p: &[Rational]) -> crate::flint::FlintPoly {
    let mut l = rug::Integer::from(1);
    for c in p.iter().filter(|c| **c != 0) {
        l.lcm_mut(c.denom());
    }
    let ints: Vec<rug::Integer> = p
        .iter()
        .map(|c| c.numer() * rug::Integer::from(&l / c.denom()))
        .collect();
    crate::flint::FlintPoly::from_rug_coefficients(&ints)
}

/// Monic GCD of `a` and `b` over ℚ.
///
/// # Why this goes through FLINT
///
/// The obvious implementation — the Euclidean algorithm over `ℚ` — is
/// quadratic in the *number* of coefficient operations but its coefficients
/// blow up through the remainder sequence, and every one of those operations is
/// a canonicalising `rug::Rational` multiply that pays a bignum GCD. Measured
/// on the degree-80 image that `∫ cos x·sin¹²x/(sin¹⁷x + sin x + 1) dx`
/// produces after the Weierstrass substitution, the ℚ-Euclid took **11.5 s**;
/// clearing denominators and handing the integer problem to FLINT's modular
/// `fmpz_poly_gcd` takes **0.3 s** for a bit-identical answer.
///
/// The result is unchanged, not merely equivalent: the monic GCD of two
/// polynomials over a field is unique, and clearing denominators multiplies
/// each input by a nonzero rational, which cannot change it.
/// `poly_gcd_euclid` (crate-internal) remains as the reference implementation and is what the
/// two agree-on-random-input property tests compare against.
pub fn poly_gcd(a: &QPoly, b: &QPoly) -> QPoly {
    let a = trim(a.clone());
    let b = trim(b.clone());
    if a.is_empty() {
        return poly_monic(&b);
    }
    if b.is_empty() {
        return poly_monic(&a);
    }
    let g = qpoly_to_fmpz(&a).gcd(&qpoly_to_fmpz(&b));
    let coeffs: Vec<Rational> = (0..g.length())
        .map(|i| Rational::from(g.get_coeff_flint(i).to_rug()))
        .collect();
    poly_monic(&coeffs)
}

/// The textbook Euclidean algorithm over `ℚ` — the reference [`poly_gcd`] is
/// checked against, kept because it needs no FFI and no integer conversion.
#[cfg(test)]
pub(crate) fn poly_gcd_euclid(a: &QPoly, b: &QPoly) -> QPoly {
    let mut a = trim(a.clone());
    let mut b = trim(b.clone());
    while !b.is_empty() {
        let (_, r) = poly_divrem(&a, &b);
        a = b;
        b = r;
    }
    poly_monic(&a)
}

/// [`poly_gcd`], but gives up when [`crate::budget`] trips.
///
/// # Why a second function instead of a check inside `poly_gcd`
///
/// A GCD has no error channel — it returns the polynomial — so a checkpoint
/// inside it could only *stop early*, and stopping the Euclidean algorithm early
/// returns a **wrong** GCD. Downstream that is a wrong antiderivative, which is
/// the one outcome worse than being slow. `poly_gcd` is also public API, so it
/// cannot grow a `Result` without a major semver break. This variant is
/// crate-internal, returns `None` rather than a truncated answer, and leaves
/// every existing caller of `poly_gcd` untouched.
///
/// # Why it is worth having
///
/// The Euclidean algorithm over ℚ has no modular or fraction-free strategy here,
/// so the coefficients grow through the remainder sequence. Normalising `A/D` to
/// lowest terms in [`super::rational_integrate::try_integrate_rational`]
/// measured **480 ms of a 482 ms call** on the degree-80 image that
/// `∫ cos x·sin⁴⁰x/(sin¹⁷x + sin x + 1) dx` produces after the Weierstrass
/// substitution — the last uninterruptible block on that route once the other
/// checkpoints were in, and the reason a 50 ms budget still took 400 ms.
/// Checking once per Euclidean step makes the granularity one `poly_divrem`.
pub(crate) fn poly_gcd_budgeted(a: &QPoly, b: &QPoly) -> Option<QPoly> {
    // `poly_gcd` now delegates to FLINT, which is a single uninterruptible call
    // — but one that is orders of magnitude shorter than the ℚ-Euclid it
    // replaced, so the checkpoint before it is the granularity that matters.
    crate::budget::check().ok()?;
    Some(poly_gcd(a, b))
}

/// Exact division `a / b` (panics in debug if the remainder is nonzero).
pub fn poly_div_exact(a: &QPoly, b: &QPoly) -> QPoly {
    let (q, r) = poly_divrem(a, b);
    debug_assert!(trim(r).is_empty(), "poly_div_exact: nonzero remainder");
    q
}

/// `p^n` for `n ≥ 0`.
pub fn poly_pow(p: &QPoly, n: u32) -> QPoly {
    let mut acc = poly_one();
    for _ in 0..n {
        acc = poly_mul(&acc, p);
    }
    acc
}

fn polys_equal(a: &QPoly, b: &QPoly) -> bool {
    trim(a.clone()) == trim(b.clone())
}

// ---------------------------------------------------------------------------
// Exact linear system solver over ℚ
// ---------------------------------------------------------------------------

/// Solve `mat · x = rhs` over ℚ by Gauss–Jordan elimination.
///
/// Returns a particular solution (free variables set to 0), or `None` if the
/// system is inconsistent.  `mat` is `rows × cols`.
fn solve_linear_system(
    mut mat: Vec<Vec<Rational>>,
    mut rhs: Vec<Rational>,
    cols: usize,
) -> Option<Vec<Rational>> {
    let rows = mat.len();
    let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; cols];
    let mut row = 0usize;

    for col in 0..cols {
        if row >= rows {
            break;
        }
        // Find a pivot in this column at or below `row`.
        let Some(sel) = (row..rows).find(|&r| mat[r][col] != 0) else {
            continue;
        };
        mat.swap(row, sel);
        rhs.swap(row, sel);

        // Normalise the pivot row.
        let piv = mat[row][col].clone();
        for cell in mat[row][col..cols].iter_mut() {
            *cell /= piv.clone();
        }
        rhs[row] /= piv.clone();

        // Eliminate the column from every other row.
        let pivot_row = mat[row].clone();
        let pivot_rhs = rhs[row].clone();
        for r in 0..rows {
            if r != row && mat[r][col] != 0 {
                let factor = mat[r][col].clone();
                for (cell, pv) in mat[r][col..cols]
                    .iter_mut()
                    .zip(pivot_row[col..cols].iter())
                {
                    *cell -= factor.clone() * pv.clone();
                }
                rhs[r] -= factor.clone() * pivot_rhs.clone();
            }
        }
        pivot_row_of_col[col] = Some(row);
        row += 1;
    }

    // Consistency: an all-zero row in `mat` with nonzero `rhs` has no solution.
    for r in 0..rows {
        if mat[r].iter().all(|v| *v == 0) && rhs[r] != 0 {
            return None;
        }
    }

    let mut x = vec![Rational::from(0); cols];
    for (col, pr) in pivot_row_of_col.iter().enumerate() {
        if let Some(pr) = pr {
            x[col] = rhs[*pr].clone();
        }
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Rational RDE solver
// ---------------------------------------------------------------------------

/// The canonical Bronstein §6.5 base-case bound on the degree of the numerator
/// `N` of a rational solution `v = N/E` of `v' + f·v = c`, where `c = C/B`
/// (reduced, `B` monic) and `E = gcd(B, B')`.
///
/// Concretely `dbound = deg E + max(deg C − deg B, deg f) + 2` (clamped at 0).
/// This is the single source of truth for the base bound; the polymorphic
/// [`DifferentialField::rde_degree_bound`](super::diff_field::DifferentialField::rde_degree_bound)
/// for `ℚ(x)` mirrors it so the ansatz solvers can use it as a search ceiling.
pub(crate) fn numerator_degree_bound(deg_b: i64, deg_c: i64, deg_e: i64, deg_f: i64) -> usize {
    let poly_part = (deg_c - deg_b).max(0);
    (deg_e.max(0) + poly_part.max(deg_f.max(0)) + 2).max(0) as usize
}

/// Reason a rational Risch DE solve **declined** to reach a verdict.
///
/// A decline is never a mathematical statement about the equation: it records
/// only that the method's ansatz, denominator bound or search budget was not
/// provably sufficient.  Callers must map it to
/// `IntegrationError::NotImplemented` (`E-INT-001`) and **never** to
/// `NonElementary` (`E-INT-004`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdeDecline {
    /// A supplied denominator was the zero polynomial.
    MalformedInput,
    /// Locating the positive-integer residues of `f` — the resonance analysis
    /// that fixes the denominator bound — would need a search past the
    /// internal cap `MAX_RESONANCE_SEARCH`.
    ResonanceSearchTooLarge,
    /// The linear system implied by the bounds would exceed the internal cap
    /// `MAX_RDE_UNKNOWNS` on the number of unknowns.
    AnsatzTooLarge,
    /// The linear system was consistent but the reconstructed `v` failed the
    /// exact substitution check.  Unreachable barring a bug in the setup; it is
    /// surfaced as a decline so an internal inconsistency can never be
    /// laundered into a certificate.
    VerificationFailed,
}

impl core::fmt::Display for RdeDecline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RdeDecline::MalformedInput => write!(f, "malformed input (zero denominator)"),
            RdeDecline::ResonanceSearchTooLarge => write!(
                f,
                "the integer-residue search needed for the denominator bound exceeds \
                 the internal cap"
            ),
            RdeDecline::AnsatzTooLarge => {
                write!(f, "the RDE ansatz exceeds the internal size cap")
            }
            RdeDecline::VerificationFailed => write!(
                f,
                "the candidate solution failed the exact substitution check"
            ),
        }
    }
}

/// Three-valued outcome of a rational Risch DE solve.
///
/// The two-valued `Option` this replaces conflated *"proved there is no
/// rational solution"* with *"my ansatz did not find one"*.  Only the first of
/// those may license a `NonElementary` certificate; callers that turn the
/// second into one emit a wrong theorem.  The distinction is carried in the
/// type so that it cannot be dropped by accident.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdeOutcome {
    /// A rational solution `v = num/den`, reduced and verified by exact
    /// substitution.
    Solved {
        /// Numerator of `v` (lowest terms).
        num: QPoly,
        /// Denominator of `v` (lowest terms, monic).
        den: QPoly,
    },
    /// **Proved**: no `v ∈ ℚ(x)` satisfies the equation.  The denominator bound
    /// and the degree bound used were both complete, and the resulting linear
    /// system over ℚ is inconsistent.  This — and only this — may be reported
    /// as `NonElementary`.
    NoRationalSolution,
    /// Nothing may be concluded; see [`RdeDecline`].
    Declined(RdeDecline),
}

impl RdeOutcome {
    /// The solution, if one was found and verified.
    pub fn solution(self) -> Option<(QPoly, QPoly)> {
        match self {
            RdeOutcome::Solved { num, den } => Some((num, den)),
            _ => None,
        }
    }

    /// Whether this outcome is a *proof* that no rational solution exists.
    ///
    /// The only predicate a caller may use to license a non-elementarity
    /// certificate.
    pub fn proves_no_solution(&self) -> bool {
        matches!(self, RdeOutcome::NoRationalSolution)
    }

    /// Whether this outcome is a decline (nothing may be concluded).
    pub fn is_declined(&self) -> bool {
        matches!(self, RdeOutcome::Declined(_))
    }
}

/// Cap on the positive-integer residue search in [`resonant_denominator`].
///
/// The search is `O(k_max)` monic GCDs; a caller cannot make it unbounded by
/// handing over an `f` with a huge residue — past this the solver declines.
const MAX_RESONANCE_SEARCH: i64 = 1024;

/// Cap on the number of unknowns in the linear system (`deg N + 1`).
///
/// Gaussian elimination over `rug::Rational` is `O(rows·cols²)` with bignum
/// coefficients, so this is the knob that bounds worst-case latency.  Measured
/// on `∫ exp(x + n·log x) dx` (i.e. `∫ xⁿ·eˣ dx`, whose resonant denominator has
/// degree `n`): `n = 120` → 0.3 s, `n = 250` → 2 s, `n = 500` → 22 s, and
/// `n = 900` declines here in 0.5 s instead of running for minutes.  Every
/// worked example in the integrator is under 100 unknowns.
const MAX_RDE_UNKNOWNS: usize = 512;

/// Solve `v' + f·v = c_num/c_den` for `v ∈ ℚ(x)`, `f` a **polynomial**.
///
/// `f` is a polynomial (the exp-tower coefficient `k·η'`), so `f` has no finite
/// poles and the classical Bronstein §6.1 denominator bound `E = gcd(B, B')`
/// (`B` = denominator of `c`) is *complete*: see [`solve_rational_rde_checked`]
/// for the pole-order argument.  Both bounds used here are therefore provable,
/// and an inconsistent linear system is a genuine non-existence proof.
pub fn solve_rational_rde_checked(f: &QPoly, c_num: &QPoly, c_den: &QPoly) -> RdeOutcome {
    let c_num = trim(c_num.clone());
    let c_den = trim(c_den.clone());

    // c = 0 → v = 0.
    if c_num.is_empty() {
        return RdeOutcome::Solved {
            num: poly_zero(),
            den: poly_one(),
        };
    }
    if c_den.is_empty() {
        return RdeOutcome::Declined(RdeDecline::MalformedInput);
    }

    // Reduce c = C/B to lowest terms with B monic.
    let g = poly_gcd(&c_num, &c_den);
    let big_c = poly_div_exact(&c_num, &g);
    let b_raw = trim(poly_div_exact(&c_den, &g));
    // Scale so that B is monic, applying the same scale to C.
    let bd = degree(&b_raw);
    if bd < 0 {
        return RdeOutcome::Declined(RdeDecline::MalformedInput);
    }
    let scale = Rational::from(1) / b_raw[bd as usize].clone();
    let big_b = poly_scale(&b_raw, &scale);
    let big_c = poly_scale(&big_c, &scale);

    // Denominator bound for v: E = gcd(B, B').  G = B / E.
    let bprime = poly_deriv(&big_b);
    let e_poly = poly_gcd(&big_b, &bprime);
    let g_poly = poly_div_exact(&big_b, &e_poly);
    let eprime = poly_deriv(&e_poly);

    // Precompute the polynomial multipliers of the linear identity
    //   Σ_j n_j · P_j(x) = C·E,   P_j = G·E·(j x^{j-1}) − G·E'·x^j + G·E·f·x^j.
    let ge = poly_mul(&g_poly, &e_poly); // G·E
    let gep = poly_mul(&g_poly, &eprime); // G·E'
    let gef = poly_mul(&ge, f); // G·E·f
    let target = poly_mul(&big_c, &e_poly); // C·E

    // Degree bound for N (= numerator of v = N/E).  See [`numerator_degree_bound`].
    let deg_b = degree(&big_b);
    let deg_c = degree(&big_c);
    let deg_e = degree(&e_poly).max(0);
    let deg_f = degree(f).max(0);
    let dbound = numerator_degree_bound(deg_b, deg_c, deg_e, deg_f);
    let cols = dbound + 1; // unknowns n_0..n_dbound
    if cols > MAX_RDE_UNKNOWNS {
        return RdeOutcome::Declined(RdeDecline::AnsatzTooLarge);
    }

    // Maximum degree appearing in the identity.
    let max_deg = (degree(&gef) + dbound as i64)
        .max(degree(&ge) + dbound as i64)
        .max(degree(&gep) + dbound as i64)
        .max(degree(&target))
        .max(0) as usize;
    let n_rows = max_deg + 1;

    // Assemble the linear system M·n = target.
    let mut mat = vec![vec![Rational::from(0); cols]; n_rows];
    for (d, row) in mat.iter_mut().enumerate() {
        let d = d as i64;
        for (j, cell) in row.iter_mut().enumerate() {
            let jj = j as i64;
            // [G·E·(j x^{j-1})]_d = j · (G·E)[d-j+1]
            let mut v = Rational::from(jj) * coeff(&ge, d - jj + 1);
            // − [G·E'·x^j]_d = −(G·E')[d-j]
            v -= coeff(&gep, d - jj);
            // + [G·E·f·x^j]_d = (G·E·f)[d-j]
            v += coeff(&gef, d - jj);
            *cell = v;
        }
    }
    let rhs: Vec<Rational> = (0..n_rows).map(|d| coeff(&target, d as i64)).collect();

    // `solve_linear_system` returns `None` **only** for an inconsistent system
    // (free variables are set to zero, so rank deficiency is not a refusal).
    // With both bounds proved complete, that is a non-existence proof.
    let Some(solution) = solve_linear_system(mat, rhs, cols) else {
        return RdeOutcome::NoRationalSolution;
    };
    let n_poly = trim(solution);

    // Verify: (N'E − N E' + f N E)·B == C·E²   (i.e. v'+f v == C/B with v=N/E).
    let np = poly_deriv(&n_poly);
    let lhs = poly_mul(
        &poly_add(
            &poly_sub(&poly_mul(&np, &e_poly), &poly_mul(&n_poly, &eprime)),
            &poly_mul(&poly_mul(f, &n_poly), &e_poly),
        ),
        &big_b,
    );
    let rhs_check = poly_mul(&big_c, &poly_mul(&e_poly, &e_poly));
    if !polys_equal(&lhs, &rhs_check) {
        return RdeOutcome::Declined(RdeDecline::VerificationFailed);
    }

    // Reduce v = N/E to lowest terms.
    if n_poly.is_empty() {
        return RdeOutcome::Solved {
            num: poly_zero(),
            den: poly_one(),
        };
    }
    let gve = poly_gcd(&n_poly, &e_poly);
    let num = poly_div_exact(&n_poly, &gve);
    let den = poly_monic(&poly_div_exact(&e_poly, &gve));
    RdeOutcome::Solved { num, den }
}

/// Solve `v' + f·v = c_num/c_den` for `v ∈ ℚ(x)`.
///
/// `f` is a polynomial (the exp-tower coefficient `k·η'`).  Returns the solution
/// as a reduced `(numerator, denominator)` pair, or `None`.
///
/// # `None` is not a proof
///
/// This two-valued shim cannot distinguish *"no rational solution exists"* from
/// *"the solver declined"*.  Use [`solve_rational_rde_checked`] — which returns
/// [`RdeOutcome`] — whenever the answer feeds a non-elementarity certificate.
pub fn solve_rational_rde(f: &QPoly, c_num: &QPoly, c_den: &QPoly) -> Option<(QPoly, QPoly)> {
    solve_rational_rde_checked(f, c_num, c_den).solution()
}

// ---------------------------------------------------------------------------
// Generalized rational RDE: f ∈ ℚ(x)  (Risch Gap F — rational exponents)
// ---------------------------------------------------------------------------

/// Extended Euclid over ℚ: `(g, s, t)` with `s·a + t·b = g` and `g` monic.
///
/// Returns all-zero for `a = b = 0`.
fn poly_ext_gcd(a: &QPoly, b: &QPoly) -> (QPoly, QPoly, QPoly) {
    let mut r0 = trim(a.clone());
    let mut r1 = trim(b.clone());
    let mut s0 = poly_one();
    let mut s1 = poly_zero();
    let mut t0 = poly_zero();
    let mut t1 = poly_one();
    while !r1.is_empty() {
        let (q, r) = poly_divrem(&r0, &r1);
        let s2 = poly_sub(&s0, &poly_mul(&q, &s1));
        let t2 = poly_sub(&t0, &poly_mul(&q, &t1));
        r0 = r1;
        r1 = trim(r);
        s0 = s1;
        s1 = s2;
        t0 = t1;
        t1 = t2;
    }
    let d = degree(&r0);
    if d < 0 {
        return (poly_zero(), poly_zero(), poly_zero());
    }
    let inv = Rational::from(1) / r0[d as usize].clone();
    (
        poly_scale(&r0, &inv),
        poly_scale(&s0, &inv),
        poly_scale(&t0, &inv),
    )
}

/// The product of the irreducible factors of `d` occurring with multiplicity
/// **exactly one** — Bronstein §6.1's `d₁` in `WeakNormalizer`.
///
/// Its roots are precisely the *simple* poles of a reduced `a/d`; every other
/// pole has order `≥ 2`.
fn multiplicity_one_part(d: &QPoly) -> QPoly {
    let g = poly_gcd(d, &poly_deriv(d)); // Π p_i^{m_i − 1}
    let d_star = poly_div_exact(d, &g); // Π p_i         (the radical)
    poly_div_exact(&d_star, &poly_gcd(&d_star, &g)) // Π_{m_i = 1} p_i
}

/// A bound on `|ρ|` over all residues `ρ` of `f = A/B` at the *simple* poles
/// enumerated by `d1`, given `β ≡ A·(d₁'·W)⁻¹ (mod d₁)` (`W = B/d₁`).
///
/// The residue at a root `α` of `d₁` is `β(α)`, and those values are exactly
/// the eigenvalues of multiplication-by-`β` on the ℚ-algebra `ℚ[x]/(d₁)`.  The
/// spectral radius of a matrix is bounded by its maximum absolute row sum, so
/// building that matrix gives a certified ceiling for the integer search.
fn residue_magnitude_bound(beta: &QPoly, d1: &QPoly) -> Rational {
    let dd = degree(d1).max(0) as usize;
    let mut rows = vec![Rational::from(0); dd];
    let mut cols = vec![Rational::from(0); dd];
    // Column j is (β·x^j) mod d₁; accumulate |entry| into its row and column.
    let mut col = poly_divrem(beta, d1).1;
    for c in cols.iter_mut() {
        for (i, cell) in col.iter().enumerate() {
            if i < dd {
                let a = cell.clone().abs();
                rows[i] += a.clone();
                *c += a;
            }
        }
        // Advance to the next column: multiply by x, reduce mod d₁.
        let mut shifted = vec![Rational::from(0)];
        shifted.extend(col.iter().cloned());
        col = poly_divrem(&trim(shifted), d1).1;
    }
    let max = |v: Vec<Rational>| {
        v.into_iter()
            .fold(Rational::from(0), |m, r| if r > m { r } else { m })
    };
    // ρ(M) ≤ ‖M‖_∞ and ρ(M) ≤ ‖M‖_1, so the smaller is still a certified bound.
    let (r, c) = (max(rows), max(cols));
    if r < c {
        r
    } else {
        c
    }
}

/// The **resonant denominator** of `f = A/B` (lowest terms, `B` monic): the
/// extra denominator a rational solution of `v' + f·v = c` may acquire from the
/// poles of `f` alone, independently of `c`.
///
/// # The pole-order argument
///
/// Let `α` be a point, `m = ord` of the pole of `f` there, `p` the pole order
/// of `c`, and `ν = ord_α(v)` for a rational solution `v`.
///
/// * `m = 0` (`f` regular): `ord(v') = ν−1` and `ord(f·v) ≥ ν`, so no
///   cancellation — `ord(c) = ν−1`, i.e. `v` has pole order `max(p−1, 0)`.
/// * `m ≥ 2`: `ord(f·v) = ν−m < ν−1`, again no cancellation — `ν = ord(c)+m`,
///   i.e. `v` has pole order `max(p−m, 0) ≤ max(p−1, 0)`.
/// * `m = 1` with residue `ρ`: `v'` and `f·v` both have order `ν−1` and their
///   leading coefficients sum to `(ν+ρ)·a`.  Unless `ν = −ρ` the previous case
///   applies; when `ν = −ρ` they cancel and `v` may have a pole of order `ρ` at
///   a point where `c` is *regular*.  A pole needs `ν < 0`, so this **resonance**
///   requires `ρ ∈ ℤ_{>0}`.
///
/// The first two bullets are exactly what `E = gcd(D, D')` (`D` = denominator of
/// `c`) already covers; the third is what this function supplies.  Multiplying
/// the two gives a *complete* denominator bound — over-sized in general (it
/// allows `p−1+ρ` where `max(p−1, ρ)` would do), which costs unknowns but never
/// solutions.
///
/// # Finding the resonances
///
/// The simple poles of `f` are the roots of `d₁`, the multiplicity-one part of
/// `B` ([`multiplicity_one_part`]); with `W = B/d₁` the residue at a root `α`
/// of `d₁` is `A(α)/(d₁'(α)·W(α))`.  So the points with residue exactly `k` are
/// the roots of `gcd(d₁, A − k·d₁'·W)`, and the answer is
/// `∏_k gcd(d₁, A − k·d₁'·W)^k` over the positive integers `k`.  This is
/// Bronstein §6.1's `WeakNormalizer` with the Rothstein–Trager resultant
/// replaced by an eigenvalue bound (see [`residue_magnitude_bound`]) plus a
/// direct GCD test — the same set of `k`, without a bivariate resultant.
fn resonant_denominator(big_a: &QPoly, big_b: &QPoly) -> Result<QPoly, RdeDecline> {
    let d1 = multiplicity_one_part(big_b);
    if degree(&d1) < 1 {
        return Ok(poly_one()); // no simple poles → no resonance
    }
    let w = poly_div_exact(big_b, &d1);
    let d1p = poly_deriv(&d1);
    let denom = poly_mul(&d1p, &w); // d₁'·W, invertible mod d₁

    // β ≡ A·(d₁'·W)⁻¹ (mod d₁): the residue element of ℚ[x]/(d₁).
    let (g, s, _) = poly_ext_gcd(&denom, &d1);
    if degree(&g) != 0 {
        // d₁ is squarefree and coprime to W, so d₁'·W is a unit mod d₁ and this
        // is unreachable.  Decline rather than assume.
        return Err(RdeDecline::VerificationFailed);
    }
    let beta = poly_divrem(&poly_mul(big_a, &s), &d1).1;

    let bound = residue_magnitude_bound(&beta, &d1);
    let k_max = bound.floor().numer().to_i64().unwrap_or(i64::MAX);
    if k_max > MAX_RESONANCE_SEARCH {
        return Err(RdeDecline::ResonanceSearchTooLarge);
    }

    let mut q = poly_one();
    for k in 1..=k_max.max(0) {
        let probe = poly_sub(big_a, &poly_scale(&denom, &Rational::from(k)));
        let gk = poly_gcd(&d1, &probe);
        if degree(&gk) >= 1 {
            q = poly_mul(&q, &poly_pow(&gk, k as u32));
        }
    }
    Ok(poly_monic(&q))
}

/// Bound on `deg_∞ v = deg(num v) − deg(den v)` for a solution of
/// `v' + f·v = c`, from the same valuation argument applied at infinity.
///
/// With `v ~ a·x^δ`, `deg(v') = δ−1` (`δ ≠ 0`) or `≤ δ−2` (`δ = 0`), and
/// `deg(f·v) = φ + δ` where `φ = deg f`:
///
/// * `φ ≥ 0`: `f·v` strictly dominates `v'`, so `γ = φ + δ` exactly.
/// * `φ ≤ −2`: `v'` strictly dominates, so `δ = γ+1` (or `δ = 0`).
/// * `φ = −1`: both have degree `δ−1` with leading coefficients summing to
///   `(δ + ρ_∞)·a`, `ρ_∞ = lc(A)/lc(B)`.  Non-resonant gives `δ = γ+1`; the
///   resonance `δ = −ρ_∞` is the extra candidate, and needs `ρ_∞ ∈ ℤ`.
///
/// `None` when the resonant candidate does not fit in an `i64` (the caller
/// declines).
fn infinity_degree_bound(deg_a: i64, deg_b: i64, lc_a: &Rational, gamma: i64) -> Option<i64> {
    if deg_a < 0 {
        return Some((gamma + 1).max(0)); // f = 0
    }
    let phi = deg_a - deg_b;
    let delta = if phi >= 0 {
        gamma - phi
    } else if phi == -1 {
        // ρ_∞ = lc(A)/lc(B) = lc(A) since B is monic here.
        let resonant = if lc_a.is_integer() {
            -lc_a.numer().to_i64()?
        } else {
            i64::MIN
        };
        (gamma + 1).max(resonant)
    } else {
        gamma + 1
    };
    Some(delta.max(0))
}

/// Solve `v' + f·v = c_num/c_den` for `v ∈ ℚ(x)` where `f = f_num/f_den` is a
/// **rational** function (not necessarily a polynomial), returning the
/// three-valued [`RdeOutcome`].
///
/// # Bounds
///
/// Write `f = A/B` and `c = C/D` in lowest terms with `B`, `D` monic.  The
/// ansatz is `v = N/Q` with
///
/// ```text
///   Q = E · q,     E = gcd(D, D'),     q = resonant denominator of f
/// ```
///
/// where `q` comes from `resonant_denominator` — the poles `f` can force into
/// `v` at points `c` knows nothing about.  `E` alone (the classical Bronstein
/// §6.1 bound, correct when `f` is a *polynomial*) is **not** complete for
/// rational `f`: `v' + (2/x + 1)·v = 1` has `c = 1`, hence `E = 1`, yet the
/// solution `v = (x²−2x+2)/x²` has a double pole at `0`.
///
/// The degree of `N` is bounded by `deg Q + δ` with `δ` from
/// `infinity_degree_bound`.  Both bounds are proved complete, so an
/// inconsistent linear system is [`RdeOutcome::NoRationalSolution`] — a genuine
/// non-existence proof.  Everything the analysis cannot cover (a residue search
/// or an ansatz past the internal caps) is [`RdeOutcome::Declined`].
///
/// Substituting `v = N/Q` and clearing `B·D·Q²` gives the identity that is
/// linear in the coefficients of `N`:
///
/// ```text
///   (B·D·Q)·N' + (A·D·Q − B·D·Q')·N = C·B·Q².
/// ```
///
/// References: Bronstein (2005) §5.4, §6.1 (`WeakNormalizer`,
/// `RdeNormalDenominator`).
pub fn solve_rational_rde_generalized_checked(
    f_num: &QPoly,
    f_den: &QPoly,
    c_num: &QPoly,
    c_den: &QPoly,
) -> RdeOutcome {
    let f_den_t = trim(f_den.clone());
    let f_num_t = trim(f_num.clone());

    // Degenerate: f_den = 0 is undefined input.
    if f_den_t.is_empty() {
        return RdeOutcome::Declined(RdeDecline::MalformedInput);
    }

    // f = 0: no poles at all, straight to the polynomial-coefficient path.
    // (Handled before the constant-denominator test, whose `f_den_t[0]` would
    // be the zero coefficient of e.g. `f_den = x`.)
    if f_num_t.is_empty() {
        return solve_rational_rde_checked(&poly_zero(), c_num, c_den);
    }

    // f is a polynomial (constant denominator): the no-finite-poles path, where
    // E = gcd(B, B') is already complete.
    if degree(&f_den_t) == 0 {
        let scale = Rational::from(1) / f_den_t[0].clone();
        let f_poly = poly_scale(&f_num_t, &scale);
        return solve_rational_rde_checked(&f_poly, c_num, c_den);
    }

    // Reduce f = A / B (lowest terms, B monic).
    let gf = poly_gcd(&f_num_t, &f_den_t);
    let a_raw = poly_div_exact(&f_num_t, &gf);
    let b_raw = poly_div_exact(&f_den_t, &gf);
    let b_raw = trim(b_raw);
    let a_raw = trim(a_raw);
    let b_d = degree(&b_raw);
    if b_d <= 0 {
        // f collapsed to a polynomial (or a malformed zero) after reduction.
        if b_d < 0 {
            return RdeOutcome::Declined(RdeDecline::MalformedInput);
        }
        let scale = Rational::from(1) / b_raw[0].clone();
        let f_poly = poly_scale(&a_raw, &scale);
        return solve_rational_rde_checked(&f_poly, c_num, c_den);
    }
    let b_lc_inv = Rational::from(1) / b_raw[b_d as usize].clone();
    let big_a = poly_scale(&a_raw, &b_lc_inv);
    let big_b = poly_scale(&b_raw, &b_lc_inv);

    // Reduce c = C / D (lowest terms, D monic).
    let c_num_t = trim(c_num.clone());
    let c_den_t = trim(c_den.clone());
    if c_num_t.is_empty() {
        return RdeOutcome::Solved {
            num: poly_zero(),
            den: poly_one(),
        };
    }
    if c_den_t.is_empty() {
        return RdeOutcome::Declined(RdeDecline::MalformedInput);
    }
    let gc = poly_gcd(&c_num_t, &c_den_t);
    let big_c_raw = poly_div_exact(&c_num_t, &gc);
    let d_raw = trim(poly_div_exact(&c_den_t, &gc));
    let d_d = degree(&d_raw);
    if d_d < 0 {
        return RdeOutcome::Declined(RdeDecline::MalformedInput);
    }
    let d_lc_inv = Rational::from(1) / d_raw[d_d as usize].clone();
    let big_d = poly_scale(&d_raw, &d_lc_inv);
    let big_c = poly_scale(&big_c_raw, &d_lc_inv);

    // Complete denominator bound: Q = gcd(D, D') · (resonant part of f).
    let e_poly = poly_gcd(&big_d, &poly_deriv(&big_d));
    let q_res = match resonant_denominator(&big_a, &big_b) {
        Ok(q) => q,
        Err(reason) => return RdeOutcome::Declined(reason),
    };
    let big_q = poly_monic(&poly_mul(&e_poly, &q_res));
    let q_prime = poly_deriv(&big_q);

    // Degree bound for N: deg Q + deg_∞ v.
    let gamma = degree(&big_c) - degree(&big_d);
    let deg_a = degree(&big_a);
    debug_assert!(deg_a >= 0, "A is nonzero: f_num was checked nonempty");
    let lc_a = big_a
        .get(deg_a.max(0) as usize)
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    let Some(delta) = infinity_degree_bound(deg_a, degree(&big_b), &lc_a, gamma) else {
        return RdeOutcome::Declined(RdeDecline::AnsatzTooLarge);
    };
    let dbound = (degree(&big_q) + delta).max(0) as usize;
    let cols = dbound + 1;
    if cols > MAX_RDE_UNKNOWNS {
        return RdeOutcome::Declined(RdeDecline::AnsatzTooLarge);
    }

    // The identity  (B·D·Q)·N' + (A·D·Q − B·D·Q')·N = C·B·Q².
    let bd = poly_mul(&big_b, &big_d);
    let u_mul = poly_mul(&bd, &big_q); // multiplier of N'
    let v_mul = poly_sub(
        &poly_mul(&poly_mul(&big_a, &big_d), &big_q),
        &poly_mul(&bd, &q_prime),
    ); // multiplier of N
    let target = poly_mul(&poly_mul(&big_c, &big_b), &poly_mul(&big_q, &big_q));

    let max_deg = (degree(&u_mul) + dbound as i64 - 1)
        .max(degree(&v_mul) + dbound as i64)
        .max(degree(&target))
        .max(0) as usize;
    let n_rows = max_deg + 1;

    // Assemble the linear system M · n = rhs, one equation per degree.
    let mut mat = vec![vec![Rational::from(0); cols]; n_rows];
    for (d, row) in mat.iter_mut().enumerate() {
        let d = d as i64;
        for (j, cell) in row.iter_mut().enumerate() {
            let jj = j as i64;
            // [U · N']_d = j · U[d−j+1];   [V · N]_d = V[d−j].
            *cell = Rational::from(jj) * coeff(&u_mul, d - jj + 1) + coeff(&v_mul, d - jj);
        }
    }
    let rhs: Vec<Rational> = (0..n_rows).map(|d| coeff(&target, d as i64)).collect();

    // Inconsistent system ⇒ no `N` of the bounded shape ⇒ (both bounds being
    // complete) no rational solution at all.
    let Some(solution) = solve_linear_system(mat, rhs, cols) else {
        return RdeOutcome::NoRationalSolution;
    };
    let n_poly = trim(solution);

    // Verify by exact substitution.
    let lhs = poly_add(
        &poly_mul(&u_mul, &poly_deriv(&n_poly)),
        &poly_mul(&v_mul, &n_poly),
    );
    if !polys_equal(&lhs, &target) {
        return RdeOutcome::Declined(RdeDecline::VerificationFailed);
    }

    // Reduce v = N / Q to lowest terms.
    if n_poly.is_empty() {
        return RdeOutcome::Solved {
            num: poly_zero(),
            den: poly_one(),
        };
    }
    let gvq = poly_gcd(&n_poly, &big_q);
    let num = poly_div_exact(&n_poly, &gvq);
    let den = poly_monic(&poly_div_exact(&big_q, &gvq));
    RdeOutcome::Solved { num, den }
}

/// Solve `v' + f·v = c_num/c_den` for `v ∈ ℚ(x)` with rational `f = f_num/f_den`.
///
/// Returns the reduced `(numerator, denominator)` of `v`, or `None`.
///
/// # `None` is not a proof
///
/// This two-valued shim collapses *"proved there is no rational solution"* and
/// *"the solver declined"* into the same `None`.  Turning that `None` into a
/// non-elementarity certificate is a soundness bug; use
/// [`solve_rational_rde_generalized_checked`] and match on [`RdeOutcome`].
pub fn solve_rational_rde_generalized(
    f_num: &QPoly,
    f_den: &QPoly,
    c_num: &QPoly,
    c_den: &QPoly,
) -> Option<(QPoly, QPoly)> {
    solve_rational_rde_generalized_checked(f_num, f_den, c_num, c_den).solution()
}

// ---------------------------------------------------------------------------
// Rational RDE over a number field K = ℚ(α)  (Risch Gap E, rational case)
// ---------------------------------------------------------------------------

/// Solve `v' + f·v = c_num/c_den` for `v ∈ K(x)`, `K = ℚ(α)`.
///
/// This is the number-field analogue of [`solve_rational_rde`]: identical
/// algorithm — denominator bound `E = gcd(B, B')`, ansatz `v = N/E`, the linear
/// identity `Σ_j n_j·P_j = C·E`, and the final substitution check — with every
/// coefficient operation routed through `field` instead of ℚ.  `f`, `c_num`,
/// `c_den` are `K`-polynomials in `x`.
///
/// In the exp tower `f = k·η'` is a polynomial (no poles), so the `E = gcd(B,B')`
/// bound is exact and an inconsistent/over-determined system correctly certifies
/// a non-elementary integral over `K` (the residual simple poles are the Ei/Li
/// part the exp tower cannot express).
pub fn solve_rational_rde_k(
    field: &NumberField,
    f: &KPoly,
    c_num: &KPoly,
    c_den: &KPoly,
) -> Option<(KPoly, KPoly)> {
    let c_num = NumberField::kpoly_trim(c_num.clone());
    let c_den = NumberField::kpoly_trim(c_den.clone());
    let one: KPoly = vec![field.from_int(1)];

    // c = 0 → v = 0.
    if c_num.is_empty() {
        return Some((Vec::new(), one));
    }
    if c_den.is_empty() {
        return None; // division by zero — malformed input
    }

    // Reduce c = C/B to lowest terms with B monic.
    let g = field.kpoly_gcd(&c_num, &c_den)?;
    let big_c = field.kpoly_div_exact(&c_num, &g)?;
    let b_raw = field.kpoly_div_exact(&c_den, &g)?;
    let bd = NumberField::kdeg(&b_raw);
    let lead_inv = field.inv(&b_raw[bd as usize])?;
    let big_b = field.kpoly_scale(&b_raw, &lead_inv);
    let big_c = field.kpoly_scale(&big_c, &lead_inv);

    // Denominator bound for v: E = gcd(B, B'). G = B / E.
    let bprime = field.kpoly_deriv(&big_b);
    let e_poly = field.kpoly_gcd(&big_b, &bprime)?;
    let g_poly = field.kpoly_div_exact(&big_b, &e_poly)?;
    let eprime = field.kpoly_deriv(&e_poly);

    // Polynomial multipliers of the identity  Σ_j n_j·P_j = C·E,
    //   P_j = G·E·(j x^{j-1}) − G·E'·x^j + G·E·f·x^j.
    let ge = field.kpoly_mul(&g_poly, &e_poly);
    let gep = field.kpoly_mul(&g_poly, &eprime);
    let gef = field.kpoly_mul(&ge, f);
    let target = field.kpoly_mul(&big_c, &e_poly);

    // Degree bound for N (= numerator of v = N/E).
    let deg_b = NumberField::kdeg(&big_b);
    let deg_c = NumberField::kdeg(&big_c);
    let deg_e = NumberField::kdeg(&e_poly).max(0);
    let deg_f = NumberField::kdeg(f).max(0);
    let poly_part = (deg_c - deg_b).max(0);
    let dbound = (deg_e + poly_part.max(deg_f) + 2).max(0) as usize;
    let cols = dbound + 1;

    let max_deg = (NumberField::kdeg(&gef) + dbound as i64)
        .max(NumberField::kdeg(&ge) + dbound as i64)
        .max(NumberField::kdeg(&gep) + dbound as i64)
        .max(NumberField::kdeg(&target))
        .max(0) as usize;
    let n_rows = max_deg + 1;

    // Assemble M·n = target over K.
    let mut mat = vec![vec![NumberField::k_zero(); cols]; n_rows];
    for (d, row) in mat.iter_mut().enumerate() {
        let d = d as i64;
        for (j, cell) in row.iter_mut().enumerate() {
            let jj = j as i64;
            // [G·E·(j x^{j-1})]_d = j · (G·E)[d-j+1]
            let mut v = field.mul(&field.from_int(jj), &NumberField::kcoeff(&ge, d - jj + 1));
            // − [G·E'·x^j]_d = −(G·E')[d-j]
            v = field.sub(&v, &NumberField::kcoeff(&gep, d - jj));
            // + [G·E·f·x^j]_d = (G·E·f)[d-j]
            v = field.add(&v, &NumberField::kcoeff(&gef, d - jj));
            *cell = v;
        }
    }
    let rhs: Vec<KElem> = (0..n_rows)
        .map(|d| NumberField::kcoeff(&target, d as i64))
        .collect();

    let solution = solve_linear_system_k(field, mat, rhs, cols)?;
    let n_poly = NumberField::kpoly_trim(solution);

    // Verify (N'E − N E' + f N E)·B == C·E².
    let np = field.kpoly_deriv(&n_poly);
    let lhs = field.kpoly_mul(
        &field.kpoly_add(
            &field.kpoly_sub(
                &field.kpoly_mul(&np, &e_poly),
                &field.kpoly_mul(&n_poly, &eprime),
            ),
            &field.kpoly_mul(&field.kpoly_mul(f, &n_poly), &e_poly),
        ),
        &big_b,
    );
    let rhs_check = field.kpoly_mul(&big_c, &field.kpoly_mul(&e_poly, &e_poly));
    if !NumberField::kpoly_eq(&lhs, &rhs_check) {
        return None;
    }

    // Reduce v = N/E to lowest terms.
    if n_poly.is_empty() {
        return Some((Vec::new(), one));
    }
    let gve = field.kpoly_gcd(&n_poly, &e_poly)?;
    let num = field.kpoly_div_exact(&n_poly, &gve)?;
    let den = field.kpoly_monic(&field.kpoly_div_exact(&e_poly, &gve)?)?;
    Some((num, den))
}

/// Decide Bronstein eq (18) for the primitive (log) case top coefficient.
///
/// For a logarithmic monomial `t = log(b)` over `K(x)` and integrand
/// `f = Σ fᵢ tⁱ`, the structure theorem (Bronstein 2005, §5.10; *Symbolic
/// Integration Tutorial* §3.5, eq (18)) requires that the **top** coefficient
/// `f_d` satisfy
///
/// ```text
///     f_d = v_d' + (d+1)·e·(b'/b),     v_d ∈ K(x),  e ∈ Const(K).
/// ```
///
/// i.e. `∫ f_d dx` is elementary **and** its only new logarithm is a constant
/// multiple of the tower generator `t = log(b)` itself.  Equation (18) has a
/// solution iff the affine family `{ f_d − e·g_drift : e ∈ K }` (with
/// `g_drift = (d+1)·b'/b`) contains a `K(x)`-rational antiderivative — a single
/// free scalar `e` that can cancel the residue at the zeros/poles of `b`, but
/// **nothing else**.
///
/// This routine returns:
/// - `Some(Some(e))` — eq (18) is solvable with that constant `e` (so `f_d` does
///   not by itself obstruct elementarity; the lower coefficients decide the rest);
/// - `Some(None)`    — `g_drift` is identically zero (degenerate `b' = 0`); falls
///   back to the plain `e = 0` antidifferentiation decision by the caller;
/// - `None`          — eq (18) has **no** solution for any `e`, which by the
///   structure theorem **proves `∫ f dx` is non-elementary** (the obstruction is
///   a residue at a pole that is not a zero of `b`, e.g. the dilogarithm pole of
///   `1/(x+√2)·log(x)` at `x = −√2`).
///
/// The decision is exact linear algebra over `K`: the ansatz `v_d = N/E` with
/// `E = gcd(B, B')` reduces eq (18) to a `K`-linear system in the coefficients of
/// `N` **and** the single unknown `e`; a verified solution of that system is
/// returned, an inconsistent system is `None`.  Soundness: a `None` is only
/// emitted after the exact system proves no `(N, e)` exists, exactly mirroring
/// the `f = 0` solver's certificate that the residual simple poles cannot be
/// absorbed.
///
/// `c_num/c_den = f_d` and `gd_num/gd_den = b'/b`, all `K`-polynomials in `x`.
#[allow(clippy::type_complexity)]
pub fn solve_primitive_top_rde_k(
    field: &NumberField,
    c_num: &KPoly,
    c_den: &KPoly,
    gd_num: &KPoly,
    gd_den: &KPoly,
    d: i64,
) -> Option<Option<KElem>> {
    let c_num = NumberField::kpoly_trim(c_num.clone());
    let c_den = NumberField::kpoly_trim(c_den.clone());
    if c_den.is_empty() {
        return None; // malformed input (division by zero)
    }
    // f_d = 0 ⇒ trivially solvable with e = 0.
    if c_num.is_empty() {
        return Some(Some(NumberField::k_zero()));
    }

    let gd_num = NumberField::kpoly_trim(gd_num.clone());
    let gd_den = NumberField::kpoly_trim(gd_den.clone());

    // (d+1)·(b'/b).  If b'/b is zero (b' = 0), there is no drift direction.
    let dp1 = field.from_int(d + 1);
    let drift_num = field.kpoly_scale(&gd_num, &dp1);
    if drift_num.is_empty() || gd_den.is_empty() {
        return Some(None);
    }

    // Put f_d and g_drift over a common denominator Bc (monic).
    //   f_d      = c_num/c_den
    //   g_drift  = drift_num/gd_den
    // Common denominator Bc = lcm(c_den, gd_den); numerators rescaled.
    let g = field.kpoly_gcd(&c_den, &gd_den)?;
    let cd_over_g = field.kpoly_div_exact(&c_den, &g)?;
    let gdd_over_g = field.kpoly_div_exact(&gd_den, &g)?;
    let bc_raw = field.kpoly_mul(&c_den, &gdd_over_g); // = lcm(c_den, gd_den)
    let bcd = NumberField::kdeg(&bc_raw);
    let lead_inv = field.inv(&bc_raw[bcd as usize])?;
    let bc = field.kpoly_scale(&bc_raw, &lead_inv);
    // f_d  = (c_num·gdd_over_g)/(Bc·lead)   ⇒ scale numerators by lead_inv too.
    let cf = field.kpoly_scale(&field.kpoly_mul(&c_num, &gdd_over_g), &lead_inv);
    // g_drift = (drift_num·cd_over_g)/(Bc·lead)
    let cg = field.kpoly_scale(&field.kpoly_mul(&drift_num, &cd_over_g), &lead_inv);

    // v_d = N/E with E = gcd(Bc, Bc'),  G = Bc/E.
    let bcp = field.kpoly_deriv(&bc);
    let e_poly = field.kpoly_gcd(&bc, &bcp)?;
    let g_poly = field.kpoly_div_exact(&bc, &e_poly)?;
    let eprime = field.kpoly_deriv(&e_poly);
    let ge = field.kpoly_mul(&g_poly, &e_poly);
    let gep = field.kpoly_mul(&g_poly, &eprime);
    // RHS target: (Cf − e·Cg)·E, with e unknown ⇒ split into Cf·E and (Cg·E).
    let target = field.kpoly_mul(&cf, &e_poly);
    let cg_e = field.kpoly_mul(&cg, &e_poly);

    // Degree bound for N (numerator of v_d = N/E), f = 0 here.
    let deg_bc = NumberField::kdeg(&bc);
    let deg_cf = NumberField::kdeg(&cf);
    let deg_e = NumberField::kdeg(&e_poly).max(0);
    let poly_part = (deg_cf - deg_bc).max(0);
    let dbound = (deg_e + poly_part + 2).max(0) as usize;
    let n_cols = dbound + 1; // coefficients of N
    let cols = n_cols + 1; // + the unknown constant e (last column)

    let max_deg = (NumberField::kdeg(&ge) + dbound as i64)
        .max(NumberField::kdeg(&gep) + dbound as i64)
        .max(NumberField::kdeg(&target))
        .max(NumberField::kdeg(&cg_e))
        .max(0) as usize;
    let n_rows = max_deg + 1;

    // Assemble M·[n; e] = target.
    //   Σ_j n_j·(G·E·j x^{j-1} − G·E'·x^j)  +  e·(Cg·E)  =  Cf·E
    // (no f·v term: f = 0 for the primitive antidifferentiation).
    let mut mat = vec![vec![NumberField::k_zero(); cols]; n_rows];
    for (deg, row) in mat.iter_mut().enumerate() {
        let deg = deg as i64;
        for (j, cell) in row.iter_mut().take(n_cols).enumerate() {
            let jj = j as i64;
            // [G·E·(j x^{j-1})]_deg = j · (G·E)[deg-j+1]
            let mut v = field.mul(&field.from_int(jj), &NumberField::kcoeff(&ge, deg - jj + 1));
            // − [G·E'·x^j]_deg = −(G·E')[deg-j]
            v = field.sub(&v, &NumberField::kcoeff(&gep, deg - jj));
            *cell = v;
        }
        // e-column: + (Cg·E)[deg]
        row[n_cols] = NumberField::kcoeff(&cg_e, deg);
    }
    let rhs: Vec<KElem> = (0..n_rows)
        .map(|deg| NumberField::kcoeff(&target, deg as i64))
        .collect();

    let solution = solve_linear_system_k(field, mat, rhs, cols)?;

    // Verify: with N = solution[..n_cols], e = solution[n_cols],
    //   (N'·E − N·E')·Bc  ==  (Cf − e·Cg)·E².
    let n_poly = NumberField::kpoly_trim(solution[..n_cols].to_vec());
    let e_val = solution[n_cols].clone();
    let np = field.kpoly_deriv(&n_poly);
    let lhs = field.kpoly_mul(
        &field.kpoly_sub(
            &field.kpoly_mul(&np, &e_poly),
            &field.kpoly_mul(&n_poly, &eprime),
        ),
        &bc,
    );
    let rhs_num = field.kpoly_sub(&cf, &field.kpoly_scale(&cg, &e_val));
    let rhs_check = field.kpoly_mul(&rhs_num, &field.kpoly_mul(&e_poly, &e_poly));
    if !NumberField::kpoly_eq(&lhs, &rhs_check) {
        return None;
    }

    Some(Some(e_val))
}

/// Solve `mat · x = rhs` over a number field `K` by Gauss–Jordan elimination.
/// Returns a particular solution (free variables 0), or `None` if inconsistent.
fn solve_linear_system_k(
    field: &NumberField,
    mut mat: Vec<Vec<KElem>>,
    mut rhs: Vec<KElem>,
    cols: usize,
) -> Option<Vec<KElem>> {
    let rows = mat.len();
    let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; cols];
    let mut row = 0usize;

    for col in 0..cols {
        if row >= rows {
            break;
        }
        let Some(sel) = (row..rows).find(|&r| !NumberField::is_zero(&mat[r][col])) else {
            continue;
        };
        mat.swap(row, sel);
        rhs.swap(row, sel);

        // Normalise the pivot row.
        let piv_inv = field.inv(&mat[row][col])?;
        for cell in mat[row][col..cols].iter_mut() {
            *cell = field.mul(cell, &piv_inv);
        }
        rhs[row] = field.mul(&rhs[row], &piv_inv);

        // Eliminate the column from every other row.
        let pivot_row = mat[row].clone();
        let pivot_rhs = rhs[row].clone();
        for r in 0..rows {
            if r != row && !NumberField::is_zero(&mat[r][col]) {
                let factor = mat[r][col].clone();
                for (cell, pv) in mat[r][col..cols]
                    .iter_mut()
                    .zip(pivot_row[col..cols].iter())
                {
                    *cell = field.sub(cell, &field.mul(&factor, pv));
                }
                rhs[r] = field.sub(&rhs[r], &field.mul(&factor, &pivot_rhs));
            }
        }
        pivot_row_of_col[col] = Some(row);
        row += 1;
    }

    // Consistency: an all-zero row with nonzero rhs has no solution.
    for r in 0..rows {
        if mat[r].iter().all(NumberField::is_zero) && !NumberField::is_zero(&rhs[r]) {
            return None;
        }
    }

    let mut x = vec![NumberField::k_zero(); cols];
    for (col, pr) in pivot_row_of_col.iter().enumerate() {
        if let Some(pr) = pr {
            x[col] = rhs[*pr].clone();
        }
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Conversion: ExprId → rational function (numerator, denominator) over ℚ
// ---------------------------------------------------------------------------

use crate::kernel::{ExprData, ExprId, ExprPool};

/// Parse `expr` as a rational function in `var` over ℚ, returning
/// `(numerator, denominator)` as `QPoly`s, or `None` if it is not a rational
/// function (e.g. contains a transcendental generator or a foreign symbol).
pub fn expr_to_qrational(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(QPoly, QPoly)> {
    if expr == var {
        return Some((vec![Rational::from(0), Rational::from(1)], poly_one()));
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some((vec![Rational::from(n.0.to_i64()?)], poly_one())),
        ExprData::Rational(r) => Some((vec![r.0.clone()], poly_one())),
        ExprData::Add(args) => {
            let mut acc = (poly_zero(), poly_one());
            for a in &args {
                let term = expr_to_qrational(*a, var, pool)?;
                acc = rat_add(&acc, &term);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = (poly_one(), poly_one());
            for a in &args {
                let factor = expr_to_qrational(*a, var, pool)?;
                acc = rat_mul(&acc, &factor);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            // Fold the exponent rather than requiring a bare `Integer` node: the
            // parser turns `x^(-1)` into `x^(1 · -1)`, and reading only `Integer`
            // made `a·b^(-1)` fail to parse as a rational function while the
            // identical `a/b` succeeded.
            let n = super::tower::literal_integer(exp, pool)?;
            let (bn, bd) = expr_to_qrational(base, var, pool)?;
            if n >= 0 {
                Some((poly_pow(&bn, n as u32), poly_pow(&bd, n as u32)))
            } else {
                let m = (-n) as u32;
                if trim(bn.clone()).is_empty() {
                    return None; // 1 / 0
                }
                Some((poly_pow(&bd, m), poly_pow(&bn, m)))
            }
        }
        _ => None,
    }
}

fn rat_add(a: &(QPoly, QPoly), b: &(QPoly, QPoly)) -> (QPoly, QPoly) {
    // a.0/a.1 + b.0/b.1 = (a.0·b.1 + b.0·a.1) / (a.1·b.1)
    let num = poly_add(&poly_mul(&a.0, &b.1), &poly_mul(&b.0, &a.1));
    let den = poly_mul(&a.1, &b.1);
    (num, den)
}

fn rat_mul(a: &(QPoly, QPoly), b: &(QPoly, QPoly)) -> (QPoly, QPoly) {
    (poly_mul(&a.0, &b.0), poly_mul(&a.1, &b.1))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    // -- poly_gcd: the FLINT route must agree with the reference ℚ-Euclid ----

    /// A cheap deterministic PRNG so this stays a unit test, not a proptest.
    fn lcg(state: &mut u64) -> i64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((*state >> 33) % 21) as i64 - 10
    }

    #[test]
    fn poly_gcd_flint_agrees_with_euclid_on_random_rationals() {
        let mut s = 0x2545_F491_4F6C_DD1D_u64;
        for trial in 0..400 {
            let da = (trial % 7) + 1;
            let db = (trial % 5) + 1;
            let mk = |n: usize, s: &mut u64| -> QPoly {
                let mut p: QPoly = (0..=n)
                    .map(|_| {
                        let num = lcg(s);
                        let den = lcg(s).unsigned_abs().max(1) as i64;
                        Rational::from((num, den))
                    })
                    .collect();
                if trim(p.clone()).is_empty() {
                    p = vec![rat(1)];
                }
                p
            };
            let a = mk(da, &mut s);
            let b = mk(db, &mut s);
            // Force a nontrivial common factor half the time.
            let (a, b) = if trial % 2 == 0 {
                let c = mk(2, &mut s);
                (poly_mul(&a, &c), poly_mul(&b, &c))
            } else {
                (a, b)
            };
            assert_eq!(
                poly_gcd(&a, &b),
                poly_gcd_euclid(&a, &b),
                "trial {trial}: FLINT gcd disagrees with ℚ-Euclid on {a:?}, {b:?}"
            );
        }
    }

    #[test]
    fn poly_gcd_flint_agrees_with_euclid_on_degenerate_inputs() {
        let zero: QPoly = Vec::new();
        let cases: Vec<(QPoly, QPoly)> = vec![
            (zero.clone(), zero.clone()),
            (vec![rat(0), rat(0)], zero.clone()),
            (zero.clone(), vec![rat(3), rat(6)]),
            (vec![rat(3), rat(6)], zero.clone()),
            (vec![rat(5)], vec![rat(7)]),
            (vec![rat(0), rat(1)], vec![rat(0), rat(0), rat(1)]),
            (
                vec![Rational::from((1, 3)), Rational::from((2, 5))],
                vec![Rational::from((7, 11))],
            ),
        ];
        for (a, b) in cases {
            assert_eq!(
                poly_gcd(&a, &b),
                poly_gcd_euclid(&a, &b),
                "FLINT gcd disagrees with ℚ-Euclid on {a:?}, {b:?}"
            );
        }
    }

    // ∫ (x-1)/x² · exp(x) dx = exp(x)/x.
    // RDE: v' + v = (x-1)/x²  →  v = 1/x.
    #[test]
    fn rational_elementary_exp_x() {
        let f = vec![rat(1)]; // f = 1 (η = x, k = 1)
        let c_num = vec![rat(-1), rat(1)]; // x - 1
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        let sol = solve_rational_rde(&f, &c_num, &c_den).expect("elementary");
        // v = 1/x  ⇒  num = 1, den = x.
        assert_eq!(trim(sol.0.clone()), vec![rat(1)], "numerator should be 1");
        assert_eq!(trim(sol.1.clone()), vec![rat(0), rat(1)], "denominator x");
    }

    // ∫ x²/(x+1) · exp(x) dx is NON-elementary (leaves an Ei term).
    #[test]
    fn rational_nonelementary_x2_over_x_plus_1() {
        let f = vec![rat(1)];
        let c_num = vec![rat(0), rat(0), rat(1)]; // x²
        let c_den = vec![rat(1), rat(1)]; // x + 1
        assert!(
            solve_rational_rde(&f, &c_num, &c_den).is_none(),
            "x²/(x+1)·exp(x) must be certified non-elementary"
        );
    }

    // ∫ exp(x)/x dx = Ei(x): RDE v' + v = 1/x has no rational solution.
    #[test]
    fn rational_nonelementary_one_over_x() {
        let f = vec![rat(1)];
        let c_num = vec![rat(1)]; // 1
        let c_den = vec![rat(0), rat(1)]; // x
        assert!(solve_rational_rde(&f, &c_num, &c_den).is_none());
    }

    // exp(x)/x² is non-elementary (residual 1/x simple pole → Ei).
    #[test]
    fn rational_nonelementary_one_over_x2() {
        let f = vec![rat(1)];
        let c_num = vec![rat(1)];
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        assert!(solve_rational_rde(&f, &c_num, &c_den).is_none());
    }

    // Polynomial RHS still works through the rational solver (E = 1).
    // ∫ x·exp(x²) dx: f = 2x, c = x  →  v = 1/2.
    #[test]
    fn rational_reduces_to_polynomial_case() {
        let f = vec![rat(0), rat(2)]; // 2x
        let c_num = vec![rat(0), rat(1)]; // x
        let c_den = poly_one();
        let sol = solve_rational_rde(&f, &c_num, &c_den).expect("elementary");
        assert_eq!(trim(sol.0), vec![Rational::from((1, 2))]);
        assert_eq!(trim(sol.1), vec![rat(1)]);
    }

    // gcd / divrem sanity.
    #[test]
    fn divrem_gcd_basic() {
        // (x² − 1) = (x + 1)(x − 1) + 0
        let a = vec![rat(-1), rat(0), rat(1)];
        let b = vec![rat(1), rat(1)];
        let (q, r) = poly_divrem(&a, &b);
        assert_eq!(trim(q), vec![rat(-1), rat(1)]); // x − 1
        assert!(trim(r).is_empty());
        // gcd(x²−1, x²−2x+1) = x − 1 (monic)
        let c = vec![rat(1), rat(-2), rat(1)];
        let g = poly_gcd(&a, &c);
        assert_eq!(trim(g), vec![rat(-1), rat(1)]);
    }

    #[test]
    fn qrational_parse() {
        use crate::kernel::{Domain, ExprPool};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // (x - 1)/x²
        let num = pool.add(vec![x, pool.integer(-1_i32)]);
        let den = pool.pow(x, pool.integer(-2_i32));
        let expr = pool.mul(vec![num, den]);
        let (n, d) = expr_to_qrational(expr, x, &pool).expect("parse");
        // Should equal (x-1)/x² up to a common factor.
        // Cross-check: n · x² == d · (x-1).
        let lhs = poly_mul(&n, &vec![rat(0), rat(0), rat(1)]);
        let rhs = poly_mul(&d, &vec![rat(-1), rat(1)]);
        assert!(polys_equal(&lhs, &rhs), "n={n:?} d={d:?}");
    }

    // -----------------------------------------------------------------------
    // Rational RDE over a number field K = ℚ(√d)  (Gap E, rational case)
    // -----------------------------------------------------------------------

    /// ℚ(√2) = ℚ[t]/(t²−2).
    fn field_sqrt2() -> NumberField {
        NumberField::new(vec![rat(-2), rat(0), rat(1)])
    }

    /// A K-constant from a rational.
    fn kc(field: &NumberField, n: i64) -> KElem {
        field.from_int(n)
    }

    // ∫ (x − √2 − 1)/(x − √2)² · exp(x) dx = exp(x)/(x − √2).
    // RDE: v' + v = (x−√2−1)/(x−√2)²  →  v = 1/(x − √2).
    #[test]
    fn rational_rde_k_elementary_sqrt2() {
        let field = field_sqrt2();
        let f: KPoly = vec![kc(&field, 1)]; // f = 1 (η = x, k = 1)
        let sqrt2 = vec![rat(0), rat(1)]; // √2 as a K-element
                                          // c_num = x − √2 − 1: x^0 = −√2 − 1, x^1 = 1.
        let c0 = field.sub(&field.neg(&sqrt2), &kc(&field, 1));
        let c_num: KPoly = vec![c0, kc(&field, 1)];
        // c_den = (x − √2)² = x² − 2√2·x + 2.
        let base: KPoly = vec![field.neg(&sqrt2), kc(&field, 1)]; // x − √2
        let c_den = field.kpoly_mul(&base, &base);

        let (vn, vd) = solve_rational_rde_k(&field, &f, &c_num, &c_den).expect("elementary");
        // v = 1/(x − √2): num = 1, den = x − √2 (monic).
        assert_eq!(NumberField::kdeg(&vn), 0);
        assert_eq!(trim(vn[0].clone()), vec![rat(1)]);
        assert!(NumberField::kpoly_eq(&vd, &base));
    }

    // ∫ x²/(x − √2) · exp(x) dx is non-elementary (Ei term: simple pole, residue 2).
    #[test]
    fn rational_rde_k_nonelementary_sqrt2() {
        let field = field_sqrt2();
        let f: KPoly = vec![kc(&field, 1)];
        let sqrt2 = vec![rat(0), rat(1)];
        let c_num: KPoly = vec![NumberField::k_zero(), NumberField::k_zero(), kc(&field, 1)]; // x²
        let c_den: KPoly = vec![field.neg(&sqrt2), kc(&field, 1)]; // x − √2
        assert!(solve_rational_rde_k(&field, &f, &c_num, &c_den).is_none());
    }

    // A polynomial RHS still works through the K rational solver (E = 1).
    // ∫ x·exp(x²) dx: f = 2x, c = x  →  v = 1/2 (a K-constant).
    #[test]
    fn rational_rde_k_reduces_to_polynomial() {
        let field = field_sqrt2();
        let f: KPoly = vec![NumberField::k_zero(), kc(&field, 2)]; // 2x
        let c_num: KPoly = vec![NumberField::k_zero(), kc(&field, 1)]; // x
        let c_den: KPoly = vec![kc(&field, 1)]; // 1
        let (vn, vd) = solve_rational_rde_k(&field, &f, &c_num, &c_den).expect("elementary");
        assert_eq!(trim(vn[0].clone()), vec![Rational::from((1, 2))]);
        assert_eq!(trim(vd[0].clone()), vec![rat(1)]);
    }

    // -----------------------------------------------------------------------
    // Primitive (log) case eq (18) — solve_primitive_top_rde_k
    // -----------------------------------------------------------------------

    /// ∫ 1/(x+√2)·log(x) dx: top coefficient f_1 = 1/(x+√2), tower arg b = x so
    /// b'/b = 1/x.  eq (18): 1/(x+√2) = v' + 2·e·(1/x) has NO solution (residue 1
    /// at x=−√2 cannot be absorbed by any e acting at x=0).  Must return None →
    /// certify NonElementary.
    #[test]
    fn primitive_top_inv_x_plus_sqrt2_over_log_x_none() {
        let field = field_sqrt2();
        let sqrt2 = vec![rat(0), rat(1)];
        // f_1 = 1/(x+√2):  c_num = 1, c_den = x + √2.
        let c_num: KPoly = vec![kc(&field, 1)];
        let c_den: KPoly = vec![sqrt2.clone(), kc(&field, 1)];
        // b'/b = 1/x: gd_num = 1, gd_den = x.
        let gd_num: KPoly = vec![kc(&field, 1)];
        let gd_den: KPoly = vec![NumberField::k_zero(), kc(&field, 1)];
        let r = solve_primitive_top_rde_k(&field, &c_num, &c_den, &gd_num, &gd_den, 1);
        assert!(
            r.is_none(),
            "eq (18) for 1/(x+√2) with b=x must be unsolvable (None); got {r:?}"
        );
    }

    /// ∫ 1/(x+√2)²·log(x) dx: f_1 = 1/(x+√2)² is K-rationally integrable on its
    /// own (P = −1/(x+√2), zero simple residue), so eq (18) is solvable with e=0.
    /// Must return Some(..) → certificate declines.
    #[test]
    fn primitive_top_inv_x_plus_sqrt2_sq_over_log_x_some() {
        let field = field_sqrt2();
        let sqrt2 = vec![rat(0), rat(1)];
        let base: KPoly = vec![sqrt2.clone(), kc(&field, 1)]; // x + √2
        let c_num: KPoly = vec![kc(&field, 1)]; // 1
        let c_den = field.kpoly_mul(&base, &base); // (x+√2)²
        let gd_num: KPoly = vec![kc(&field, 1)];
        let gd_den: KPoly = vec![NumberField::k_zero(), kc(&field, 1)]; // x
        let r = solve_primitive_top_rde_k(&field, &c_num, &c_den, &gd_num, &gd_den, 1);
        assert!(
            r.is_some(),
            "eq (18) for 1/(x+√2)² with b=x must be solvable (Some); got {r:?}"
        );
    }

    /// ∫ 1/(x+√2)·log(x+√2) dx: f_1 = 1/(x+√2) = b'/b with b = x+√2.  eq (18) is
    /// solvable with e = 1/2 (v=0): the residue at x=−√2 IS the drift pole.  Must
    /// return Some(..) → certificate declines (the log-derivative shortcut covers
    /// this elsewhere, but the obstruction test must agree it is solvable).
    #[test]
    fn primitive_top_log_derivative_same_arg_some() {
        let field = field_sqrt2();
        let sqrt2 = vec![rat(0), rat(1)];
        let base: KPoly = vec![sqrt2.clone(), kc(&field, 1)]; // x + √2
        let c_num: KPoly = vec![kc(&field, 1)]; // 1
        let c_den = base.clone(); // x + √2
                                  // b = x+√2 ⇒ b'/b = 1/(x+√2).
        let gd_num: KPoly = vec![kc(&field, 1)];
        let gd_den = base.clone();
        let r = solve_primitive_top_rde_k(&field, &c_num, &c_den, &gd_num, &gd_den, 1);
        assert!(
            r.is_some(),
            "eq (18) for 1/(x+√2) with b=x+√2 must be solvable (Some); got {r:?}"
        );
    }

    /// Soundness probe for the degree bound: a *polynomial* top coefficient
    /// f_1 = x has the trivial K-rational antiderivative v = x²/2 (e = 0), so
    /// eq (18) must be solvable.  This guards against a too-small `dbound`
    /// spuriously rejecting a genuine polynomial-part solution (a false None
    /// here would be a wrong NonElementary).  b = x ⇒ b'/b = 1/x.
    #[test]
    fn primitive_top_polynomial_coeff_some() {
        let field = field_sqrt2();
        // f_1 = x:  c_num = x, c_den = 1.
        let c_num: KPoly = vec![NumberField::k_zero(), kc(&field, 1)];
        let c_den: KPoly = vec![kc(&field, 1)];
        // b'/b = 1/x: gd_num = 1, gd_den = x.
        let gd_num: KPoly = vec![kc(&field, 1)];
        let gd_den: KPoly = vec![NumberField::k_zero(), kc(&field, 1)];
        let r = solve_primitive_top_rde_k(&field, &c_num, &c_den, &gd_num, &gd_den, 1);
        assert!(
            r.is_some(),
            "eq (18) for the polynomial coeff f_1=x (v=x²/2, e=0) must be solvable; got {r:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Generalized rational RDE: f ∈ ℚ(x)  (Gap F — rational exponents)
    // -----------------------------------------------------------------------

    // ∫ exp(1/x) dx: RDE v' − (1/x²)·v = 1, no rational solution.
    // f = −1/x² (f_num = −1, f_den = x²), c = 1.
    #[test]
    fn gen_rde_exp_inv_x_nonelementary() {
        let f_num = vec![rat(-1)];
        let f_den = vec![rat(0), rat(0), rat(1)]; // x²
        let c_num = poly_one();
        let c_den = poly_one();
        assert!(
            solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den).is_none(),
            "∫ exp(1/x) dx must be certified non-elementary"
        );
    }

    // ∫ (1/x²)·exp(1/x) dx = −exp(1/x).
    // f = −1/x², c = 1/x².  Solution v = −1 = N/E with N = −x, E = x.
    #[test]
    fn gen_rde_inv_x2_exp_inv_x_elementary() {
        let f_num = vec![rat(-1)];
        let f_den = vec![rat(0), rat(0), rat(1)]; // x²
        let c_num = poly_one(); // 1
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        let (vn, vd) = solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den)
            .expect("∫ (1/x²)·exp(1/x) dx must be elementary");
        // v = −1: num = −1, den = 1.
        assert_eq!(
            trim(vn.clone()),
            vec![rat(-1)],
            "numerator should be −1, got {vn:?}"
        );
        assert_eq!(
            trim(vd.clone()),
            poly_one(),
            "denominator should be 1, got {vd:?}"
        );
    }

    // ∫ (2/x³)·exp(−1/x²) dx = exp(−1/x²).
    // η = −1/x², η' = 2/x³.  f = 2/x³, c = 2/x³.  Solution v = 1.
    #[test]
    fn gen_rde_exp_neg_inv_x2_elementary() {
        let f_num = vec![rat(2)];
        let f_den = vec![rat(0), rat(0), rat(0), rat(1)]; // x³
        let c_num = vec![rat(2)];
        let c_den = vec![rat(0), rat(0), rat(0), rat(1)]; // x³
        let (vn, vd) = solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den)
            .expect("∫ (2/x³)·exp(−1/x²) dx must be elementary");
        // v = 1: num = 1, den = 1.
        assert_eq!(
            trim(vn.clone()),
            poly_one(),
            "numerator should be 1, got {vn:?}"
        );
        assert_eq!(
            trim(vd.clone()),
            poly_one(),
            "denominator should be 1, got {vd:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Resonant poles of `f`: the denominator of `v` is not bounded by `c` alone
    // -----------------------------------------------------------------------

    /// Assert `v = num/den` really solves `v' + (f_num/f_den)·v = c_num/c_den`,
    /// by exact polynomial identity — not by comparing against an expected
    /// display form.
    ///
    /// `v' + f·v = c`  ⟺  `f_den·c_den·(num'·den − num·den') + f_num·c_den·num·den
    ///                      = c_num·f_den·den²`.
    fn assert_solves(
        f_num: &QPoly,
        f_den: &QPoly,
        c_num: &QPoly,
        c_den: &QPoly,
        num: &QPoly,
        den: &QPoly,
    ) {
        let lhs = poly_add(
            &poly_mul(
                &poly_mul(f_den, c_den),
                &poly_sub(
                    &poly_mul(&poly_deriv(num), den),
                    &poly_mul(num, &poly_deriv(den)),
                ),
            ),
            &poly_mul(&poly_mul(f_num, c_den), &poly_mul(num, den)),
        );
        let rhs = poly_mul(&poly_mul(c_num, f_den), &poly_mul(den, den));
        assert!(
            polys_equal(&lhs, &rhs),
            "v = ({num:?})/({den:?}) does not satisfy the RDE"
        );
    }

    /// `v' + (1/x + 1)·v = 1` — the RDE behind `∫ x·eˣ dx = (x−1)·eˣ`.
    ///
    /// `c = 1`, so the classical `E = gcd(D, D')` bound admits only *polynomial*
    /// `v`; the true solution `v = (x−1)/x` has a simple pole at `0`, forced by
    /// `f`'s residue `1` there.  Before the resonance analysis this returned
    /// `None`, which the exp-tower caller turned into a false `E-INT-004`.
    #[test]
    fn gen_rde_resonant_simple_pole_kappa1() {
        let f_num = vec![rat(1), rat(1)]; // 1 + x   (f = 1/x + 1)
        let f_den = vec![rat(0), rat(1)]; // x
        let c_num = poly_one();
        let c_den = poly_one();
        let (vn, vd) = match solve_rational_rde_generalized_checked(&f_num, &f_den, &c_num, &c_den)
        {
            RdeOutcome::Solved { num, den } => (num, den),
            other => panic!("expected v = (x−1)/x, got {other:?}"),
        };
        assert_eq!(trim(vn.clone()), vec![rat(-1), rat(1)], "numerator ≠ x−1");
        assert_eq!(trim(vd.clone()), vec![rat(0), rat(1)], "denominator ≠ x");
        assert_solves(&f_num, &f_den, &c_num, &c_den, &vn, &vd);
    }

    /// `v' + (2/x + 1)·v = 1` — the RDE behind `∫ x²·eˣ dx = (x²−2x+2)·eˣ`.
    /// Residue `2` at `0` ⇒ `v` has a *double* pole there.
    #[test]
    fn gen_rde_resonant_simple_pole_kappa2() {
        let f_num = vec![rat(2), rat(1)]; // 2 + x   (f = 2/x + 1)
        let f_den = vec![rat(0), rat(1)]; // x
        let c_num = poly_one();
        let c_den = poly_one();
        let (vn, vd) = match solve_rational_rde_generalized_checked(&f_num, &f_den, &c_num, &c_den)
        {
            RdeOutcome::Solved { num, den } => (num, den),
            other => panic!("expected v = (x²−2x+2)/x², got {other:?}"),
        };
        assert_eq!(
            trim(vn.clone()),
            vec![rat(2), rat(-2), rat(1)],
            "numerator ≠ x²−2x+2"
        );
        assert_eq!(
            trim(vd.clone()),
            vec![rat(0), rat(0), rat(1)],
            "denominator ≠ x²"
        );
        assert_solves(&f_num, &f_den, &c_num, &c_den, &vn, &vd);
    }

    /// The resonant poles need not be rational: `f = 2x/(x²+1) + 1` has residue
    /// `1` at each of `±i`, and `∫ (x²+1)·eˣ dx = (x²−2x+3)·eˣ` needs
    /// `v = (x²−2x+3)/(x²+1)`.  This is the case a rational-root test on the
    /// residue would miss and the GCD formulation catches.
    #[test]
    fn gen_rde_resonant_pole_irrational() {
        // f = (x² + 2x + 1)/(x² + 1).
        let f_num = vec![rat(1), rat(2), rat(1)];
        let f_den = vec![rat(1), rat(0), rat(1)];
        let c_num = poly_one();
        let c_den = poly_one();
        let (vn, vd) = match solve_rational_rde_generalized_checked(&f_num, &f_den, &c_num, &c_den)
        {
            RdeOutcome::Solved { num, den } => (num, den),
            other => panic!("expected v = (x²−2x+3)/(x²+1), got {other:?}"),
        };
        assert_eq!(trim(vn.clone()), vec![rat(3), rat(-2), rat(1)]);
        assert_eq!(trim(vd.clone()), vec![rat(1), rat(0), rat(1)]);
        assert_solves(&f_num, &f_den, &c_num, &c_den, &vn, &vd);
    }

    /// A *negative* residue is not a resonance (a pole of `v` needs `ν < 0`,
    /// i.e. `ρ = −ν > 0`), so `∫ eˣ/x dx = Ei(x)` stays proved non-elementary.
    /// This is the guard that the wider denominator bound has not turned an
    /// honest `NoRationalSolution` into a decline.
    #[test]
    fn gen_rde_negative_residue_still_proves_no_solution() {
        // f = 1 − 1/x = (x − 1)/x.
        let f_num = vec![rat(-1), rat(1)];
        let f_den = vec![rat(0), rat(1)];
        let out = solve_rational_rde_generalized_checked(&f_num, &f_den, &poly_one(), &poly_one());
        assert_eq!(
            out,
            RdeOutcome::NoRationalSolution,
            "∫ eˣ/x dx must stay *proved* non-elementary"
        );
        assert!(out.proves_no_solution());
    }

    /// A residue past the internal search cap must **decline**, never certify.
    /// `f = 1 + 4000/x` has residue `4000` at `0`; the honest answer is
    /// `Declined`, and `proves_no_solution()` must be false.
    #[test]
    fn gen_rde_huge_residue_declines_rather_than_certifies() {
        let f_num = vec![rat(4000), rat(1)]; // 4000 + x
        let f_den = vec![rat(0), rat(1)]; // x
        let out = solve_rational_rde_generalized_checked(&f_num, &f_den, &poly_one(), &poly_one());
        assert!(
            out.is_declined(),
            "a residue past the cap must decline; got {out:?}"
        );
        assert!(
            !out.proves_no_solution(),
            "a decline must never license a NonElementary certificate"
        );
        // The two-valued shim collapses this to `None` — which is exactly why
        // callers must not read `None` as a proof.
        assert!(solve_rational_rde_generalized(&f_num, &f_den, &poly_one(), &poly_one()).is_none());
    }

    /// The polynomial-`f` entry point keeps its (already sound) proof strength:
    /// `∫ eˣ/x` restated as `v' + v = 1/x`.
    #[test]
    fn poly_rde_checked_proves_no_solution() {
        let f = poly_one(); // f = 1
        let c_num = poly_one();
        let c_den = vec![rat(0), rat(1)]; // 1/x
        assert_eq!(
            solve_rational_rde_checked(&f, &c_num, &c_den),
            RdeOutcome::NoRationalSolution
        );
    }

    // Polynomial f falls back to the existing solver correctly.
    // ∫ (x−1)/x²·exp(x) dx = exp(x)/x.  f = 1 (constant den), c = (x−1)/x².
    #[test]
    fn gen_rde_falls_back_to_polynomial_f() {
        let f_num = vec![rat(1)];
        let f_den = poly_one(); // constant denominator → delegate to solve_rational_rde
        let c_num = vec![rat(-1), rat(1)]; // x − 1
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        let (vn, vd) = solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den)
            .expect("fallback must succeed");
        // v = 1/x.
        assert_eq!(trim(vn), vec![rat(1)]);
        assert_eq!(trim(vd), vec![rat(0), rat(1)]);
    }
}
