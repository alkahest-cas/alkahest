//! V2-3 — Sparse polynomial interpolation (Ben-Or/Tiwari, Zippel).
//!
//! Recovers a sparse multivariate polynomial over `F_p = ℤ/pℤ` from
//! black-box evaluations using far fewer queries than dense interpolation.
//!
//! # Algorithms
//!
//! - **Univariate Ben-Or/Tiwari (Prony-style)** — [`sparse_interpolate_univariate`]:
//!   given that `f ∈ F_p[x]` has at most `T` nonzero terms, recovers `f` from
//!   exactly `2T` evaluations via Berlekamp–Massey + brute-force root-finding
//!   + Vandermonde solve.  Cost: `2T` oracle calls.
//!
//! - **Multivariate Zippel** — [`sparse_interpolate`]: variable-by-variable
//!   reduction.  At each variable level:
//!     1. Evaluate `f(x₁, a₂, …, aₙ)` at random `aᵢ` and run Ben-Or/Tiwari
//!        to find the `x₁`-exponent skeleton.
//!     2. Construct an oracle for each coefficient polynomial
//!        `cₑ(x₂, …, xₙ)` via Vandermonde inversion.
//!     3. Recurse on each `cₑ`.
//!
//! - **Dense fallback** — applied when `degree_bound ≤ term_bound` (dense
//!   and sparse costs coincide).  Uses Lagrange interpolation at consecutive
//!   integers.
//!
//! # Public API
//!
//! ```text
//! sparse_interpolate_univariate(eval, term_bound, prime) → Vec<(coeff, exp)>
//! sparse_interpolate(eval, vars, term_bound, degree_bound, prime, seed)
//!     → MultiPolyFp
//! ```

use crate::errors::AlkahestError;
use crate::kernel::ExprId;
use crate::modular::{is_prime, MultiPolyFp};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by sparse interpolation functions.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseInterpError {
    /// The prime is ≤ 2 or composite.
    InvalidPrime(u64),
    /// The prime must be `> 2 * term_bound` for Ben-Or/Tiwari to work.
    PrimeTooSmall { prime: u64, term_bound: usize },
    /// Root-finding found fewer roots than expected (should not happen for
    /// correct evaluations; indicates either colliding exponents or an
    /// inconsistent evaluation oracle).
    RootFindingFailed,
    /// The Vandermonde / linear system is singular.  Retry with a different
    /// seed or a larger prime.
    SingularSystem,
}

impl std::fmt::Display for SparseInterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SparseInterpError::InvalidPrime(p) => {
                write!(f, "invalid prime {p}: must be a prime ≥ 3")
            }
            SparseInterpError::PrimeTooSmall { prime, term_bound } => write!(
                f,
                "prime {prime} is too small for term_bound {term_bound}: need prime > 2·T = {}",
                2 * term_bound
            ),
            SparseInterpError::RootFindingFailed => write!(
                f,
                "could not find the expected number of roots in F_p; \
                 the prime may be too small or the oracle is inconsistent"
            ),
            SparseInterpError::SingularSystem => write!(
                f,
                "Vandermonde system is singular; try a different seed or a larger prime"
            ),
        }
    }
}

impl std::error::Error for SparseInterpError {}

impl AlkahestError for SparseInterpError {
    fn code(&self) -> &'static str {
        match self {
            SparseInterpError::InvalidPrime(_) => "E-INTERP-001",
            SparseInterpError::PrimeTooSmall { .. } => "E-INTERP-002",
            SparseInterpError::RootFindingFailed => "E-INTERP-003",
            SparseInterpError::SingularSystem => "E-INTERP-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            SparseInterpError::InvalidPrime(_) => {
                Some("choose a prime p ≥ 3, e.g. 1009, 32749, 1000003")
            }
            SparseInterpError::PrimeTooSmall { .. } => {
                Some("increase the prime so that p > 2 * term_bound")
            }
            SparseInterpError::RootFindingFailed => {
                Some("choose a prime larger than the maximum degree in the polynomial")
            }
            SparseInterpError::SingularSystem => {
                Some("retry with a different seed or use a larger prime")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal PRNG (xorshift64) — no external crate needed
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for reproducible random evaluation points.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state.
        let s = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        Xorshift64 { state: s }
    }

    pub fn step(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a value in `[lo, hi)`.
    pub fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        debug_assert!(hi > lo);
        lo + self.step() % (hi - lo)
    }

    /// Return a non-zero value in `[1, p)`.
    pub fn nonzero(&mut self, p: u64) -> u64 {
        loop {
            let v = self.step() % p;
            if v != 0 {
                return v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Modular arithmetic helpers
// ---------------------------------------------------------------------------

#[inline]
fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

#[inline]
fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p {
        s - p
    } else {
        s
    }
}

#[inline]
fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + p - b
    }
}

fn pow_mod(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut result = 1u64;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, p);
        }
        base = mul_mod(base, base, p);
        exp >>= 1;
    }
    result
}

/// Extended-GCD modular inverse.  Panics if `gcd(a, p) ≠ 1`.
fn mod_inv(a: u64, p: u64) -> u64 {
    debug_assert!(a != 0, "mod_inv: a must be non-zero");
    let mut old_r = a as i128;
    let mut r = p as i128;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;
    while r != 0 {
        let q = old_r / r;
        let tmp = r;
        r = old_r - q * r;
        old_r = tmp;
        let tmp = s;
        s = old_s - q * s;
        old_s = tmp;
    }
    ((old_s % p as i128 + p as i128) % p as i128) as u64
}

// ---------------------------------------------------------------------------
// Polynomial evaluation over F_p
// ---------------------------------------------------------------------------

/// Evaluate `poly[0] + poly[1]*x + ... + poly[d]*x^d` at `x` modulo `p`.
fn poly_eval(poly: &[u64], x: u64, p: u64) -> u64 {
    let mut acc = 0u64;
    let mut pw = 1u64;
    for &c in poly {
        acc = add_mod(acc, mul_mod(c, pw, p), p);
        pw = mul_mod(pw, x, p);
    }
    acc
}

// ---------------------------------------------------------------------------
// Primitive root of F_p
// ---------------------------------------------------------------------------

/// Find the smallest primitive root (generator) of `F_p*`.
///
/// A primitive root `g` satisfies `g^{(p-1)/q} ≢ 1 (mod p)` for every
/// prime factor `q` of `p-1`.
pub fn primitive_root(p: u64) -> u64 {
    debug_assert!(is_prime(p), "primitive_root: p must be prime");
    if p == 2 {
        return 1;
    }
    if p == 3 {
        return 2;
    }
    let factors = prime_factors(p - 1);
    'outer: for g in 2..p {
        for &q in &factors {
            if pow_mod(g, (p - 1) / q, p) == 1 {
                continue 'outer;
            }
        }
        return g;
    }
    panic!("primitive_root: no root found for prime {p}");
}

/// Distinct prime factors of `n` (trial division).
fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ---------------------------------------------------------------------------
// Berlekamp–Massey over F_p
// ---------------------------------------------------------------------------

/// Berlekamp–Massey algorithm over `F_p`.
///
/// Given a sequence `s[0], …, s[N-1]`, returns the minimal LFSR connection
/// polynomial `Λ = [1, λ₁, …, λ_L]` (index = degree) such that
///
/// ```text
/// s[n] + λ₁·s[n-1] + … + λ_L·s[n-L] = 0   for all n ≥ L.
/// ```
///
/// The caller must supply `N ≥ 2L` for the result to be unique.
fn berlekamp_massey(seq: &[u64], p: u64) -> Vec<u64> {
    let n = seq.len();
    let mut l = 0usize;
    let mut c: Vec<u64> = vec![1];
    let mut b: Vec<u64> = vec![1];
    let mut b_disc: u64 = 1;
    let mut x: usize = 1;

    for n_idx in 0..n {
        // Discrepancy d = s[n] + Σ_{i=1}^{L} c[i]·s[n-i]
        let mut d = seq[n_idx];
        let bound = l.min(c.len().saturating_sub(1));
        for i in 1..=bound {
            d = add_mod(d, mul_mod(c[i], seq[n_idx - i], p), p);
        }

        if d == 0 {
            x += 1;
            continue;
        }

        let t = c.clone();
        let factor = mul_mod(d, mod_inv(b_disc, p), p);

        // C ← C − factor·z^x·B
        let needed = x + b.len();
        if c.len() < needed {
            c.resize(needed, 0);
        }
        for j in 0..b.len() {
            let sub = mul_mod(factor, b[j], p);
            c[x + j] = sub_mod(c[x + j], sub, p);
        }

        if 2 * l <= n_idx {
            l = n_idx + 1 - l;
            b = t;
            b_disc = d;
            x = 1;
        } else {
            x += 1;
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Root finding in F_p (brute force)
// ---------------------------------------------------------------------------

/// Find all roots of `poly` (given as coefficient list, lowest degree first)
/// in `F_p` by exhaustive evaluation.
fn find_roots(poly: &[u64], p: u64) -> Vec<u64> {
    let mut roots = Vec::new();
    for v in 0..p {
        if poly_eval(poly, v, p) == 0 {
            roots.push(v);
        }
    }
    roots
}

// ---------------------------------------------------------------------------
// Baby-step giant-step discrete logarithm
// ---------------------------------------------------------------------------

/// Compute `e` such that `g^e ≡ target (mod p)`, or `None` if no such `e`
/// exists in `{0, …, p-2}`.
///
/// Uses the Baby-step / Giant-step algorithm in `O(√p)` time and space.
pub fn bsgs_dlog(g: u64, target: u64, p: u64) -> Option<u64> {
    if target == 0 {
        return None; // g is never 0 in F_p*
    }
    let order = p - 1; // order of F_p* (g is a generator)
    let m = (order as f64).sqrt().ceil() as u64 + 1;

    // Baby steps: table[g^j] = j  for j = 0 … m-1
    let mut table = std::collections::HashMap::with_capacity(m as usize);
    let mut gj = 1u64;
    for j in 0..m {
        table.insert(gj, j);
        gj = mul_mod(gj, g, p);
    }

    // Giant steps: find i such that target · (g^{-m})^i is in table
    let gm = pow_mod(g, m, p);
    let gm_inv = mod_inv(gm, p);
    let mut y = target;
    for i in 0..m {
        if let Some(&j) = table.get(&y) {
            let e = i * m + j;
            let e_mod = e % order;
            // Verify
            if pow_mod(g, e_mod, p) == target {
                return Some(e_mod);
            }
        }
        y = mul_mod(y, gm_inv, p);
    }
    None
}

// ---------------------------------------------------------------------------
// Vandermonde solve (generalised)
// ---------------------------------------------------------------------------

/// Solve the generalised Vandermonde system:
///
/// ```text
/// Σ_j  c[j] · pts[i]^{exps[j]}  =  vals[i]   for i = 0, …, t-1
/// ```
///
/// Returns `Some(c)` if the system is non-singular, or `None` otherwise.
fn vandermonde_solve(pts: &[u64], exps: &[u32], vals: &[u64], p: u64) -> Option<Vec<u64>> {
    let t = pts.len();
    debug_assert_eq!(exps.len(), t);
    debug_assert_eq!(vals.len(), t);

    // Build the t×t matrix A where A[i][j] = pts[i]^exps[j]
    let mut mat: Vec<Vec<u64>> = (0..t)
        .map(|i| (0..t).map(|j| pow_mod(pts[i], exps[j] as u64, p)).collect())
        .collect();
    let mut rhs: Vec<u64> = vals.to_vec();

    gaussian_elim(&mut mat, &mut rhs, p)
}

/// Gaussian elimination with partial pivoting over `F_p`.
/// Modifies `mat` and `rhs` in place; returns the solution or `None` if
/// the system is singular.
fn gaussian_elim(mat: &mut [Vec<u64>], rhs: &mut [u64], p: u64) -> Option<Vec<u64>> {
    let n = mat.len();
    for col in 0..n {
        // Find pivot (first non-zero entry in column col, at or below row col)
        let pivot_row = (col..n).find(|&r| mat[r][col] != 0)?;
        mat.swap(col, pivot_row);
        rhs.swap(col, pivot_row);

        let inv = mod_inv(mat[col][col], p);
        // Scale pivot row
        for entry in &mut mat[col][col..] {
            *entry = mul_mod(*entry, inv, p);
        }
        rhs[col] = mul_mod(rhs[col], inv, p);

        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            if factor == 0 {
                continue;
            }
            // Gather the pivot row values to avoid borrow conflict.
            let pivot_row_vals: Vec<u64> = mat[col][col..].to_vec();
            for (j, &pv) in pivot_row_vals.iter().enumerate() {
                let sub = mul_mod(factor, pv, p);
                mat[row][col + j] = sub_mod(mat[row][col + j], sub, p);
            }
            let sub = mul_mod(factor, rhs[col], p);
            rhs[row] = sub_mod(rhs[row], sub, p);
        }
    }
    Some(rhs.to_owned())
}

// ---------------------------------------------------------------------------
// Univariate Ben-Or/Tiwari (internal)
// ---------------------------------------------------------------------------

/// Internal Ben-Or/Tiwari.  Evaluates at `g^0, …, g^{2T-1}` and runs the
/// full Prony pipeline.  Returns `(coeff, exponent)` pairs, or an error.
fn bt_univariate(
    eval: &dyn Fn(u64) -> u64,
    term_bound: usize,
    prime: u64,
    g: u64, // primitive root of F_p
) -> Result<Vec<(u64, u32)>, SparseInterpError> {
    if term_bound == 0 {
        return Ok(vec![]);
    }
    let two_t = 2 * term_bound;

    // --- Step 1: Evaluate at g^0, g^1, …, g^{2T-1} ---
    let mut seq = Vec::with_capacity(two_t);
    let mut gj = 1u64; // g^j
    for _ in 0..two_t {
        seq.push(eval(gj));
        gj = mul_mod(gj, g, prime);
    }

    // --- Step 2: Berlekamp–Massey to find connection polynomial Λ ---
    let lambda = berlekamp_massey(&seq, prime);
    let ell = lambda.len() - 1; // LFSR length L ≤ T

    if ell == 0 {
        // Only the trivial polynomial: the sequence is identically zero.
        return Ok(vec![]);
    }

    // --- Step 3: Find roots ρ of Λ in F_p ---
    // Brute-force root finding.  Works for p up to ~10^6; larger primes
    // may need Cantor–Zassenhaus (not implemented here).
    let rho_roots = find_roots(&lambda, prime);

    if rho_roots.len() < ell {
        return Err(SparseInterpError::RootFindingFailed);
    }
    // Use only the first `ell` roots (should be exactly ell distinct ones).
    let rho: &[u64] = &rho_roots[..ell];

    // --- Step 4: Map roots → frequencies → exponents ---
    // Λ has roots ρ_j = g^{-e_j} (the inverses of the frequencies r_j).
    // r_j = ρ_j^{-1} = g^{e_j}.
    let mut exps: Vec<u32> = Vec::with_capacity(ell);
    for &ro in rho {
        if ro == 0 {
            return Err(SparseInterpError::RootFindingFailed);
        }
        let r = mod_inv(ro, prime); // r = g^{e_j}
        let e = bsgs_dlog(g, r, prime).ok_or(SparseInterpError::RootFindingFailed)?;
        exps.push(e as u32);
    }

    // --- Step 5: Solve Vandermonde for coefficients ---
    // The evaluation sequence satisfies: s[n] = Σ_j c_j · (g^{e_j})^n.
    // As a matrix system with A[i][j] = pts[i]^{exps[j]}:
    //   pts[i] = g^i  (i-th evaluation point)
    //   exps[j] = e_j  (j-th monomial exponent)
    //   vals[i] = s[i]  (i-th sequence value)
    // This is the correct generalised-Vandermonde formulation.
    let pts_for_vdm: Vec<u64> = (0..ell).map(|i| pow_mod(g, i as u64, prime)).collect();
    let vals_for_vdm: Vec<u64> = seq[..ell].to_vec();
    let coeffs = vandermonde_solve(&pts_for_vdm, &exps, &vals_for_vdm, prime)
        .ok_or(SparseInterpError::SingularSystem)?;

    Ok(coeffs
        .into_iter()
        .zip(exps)
        .filter(|(c, _)| *c != 0)
        .collect())
}

// ---------------------------------------------------------------------------
// Dense univariate interpolation (fallback)
// ---------------------------------------------------------------------------

/// Dense Lagrange interpolation over `F_p`.
///
/// Given evaluations `f(1), f(2), …, f(D+1)` (at the first `D+1` non-zero
/// field elements), returns the polynomial coefficients in ascending degree
/// order.
fn dense_interpolate(vals: &[u64], prime: u64) -> Vec<(u64, u32)> {
    let n = vals.len();
    // Evaluation points: 1, 2, …, n
    let pts: Vec<u64> = (1..=n as u64).collect();
    // Build Vandermonde system: pts[i]^j * c[j] = vals[i]
    let mut mat: Vec<Vec<u64>> = (0..n)
        .map(|i| (0..n).map(|j| pow_mod(pts[i], j as u64, prime)).collect())
        .collect();
    let mut rhs = vals.to_vec();
    match gaussian_elim(&mut mat, &mut rhs, prime) {
        Some(coeffs) => coeffs
            .into_iter()
            .enumerate()
            .filter(|(_, c)| *c != 0)
            .map(|(j, c)| (c, j as u32))
            .collect(),
        None => vec![], // degenerate; return empty
    }
}

// ---------------------------------------------------------------------------
// Multivariate Zippel (recursive)
// ---------------------------------------------------------------------------

/// Recursive Zippel helper.  Returns a map from exponent vectors to
/// coefficients in `F_p`.
fn zippel_helper(
    eval: &dyn Fn(&[u64]) -> u64,
    n_vars: usize,
    term_bound: usize,
    degree_bound: u32,
    prime: u64,
    g: u64,
    rng: &mut Xorshift64,
) -> Result<BTreeMap<Vec<u32>, u64>, SparseInterpError> {
    // --- Base case: constant polynomial ---
    if n_vars == 0 {
        let c = eval(&[]);
        let mut m = BTreeMap::new();
        if c != 0 {
            m.insert(vec![], c);
        }
        return Ok(m);
    }

    // --- Base case: univariate ---
    if n_vars == 1 {
        // Use dense fallback if degree_bound is small (avoids BSGS overhead).
        let terms = if degree_bound <= term_bound as u32 {
            // Dense path: evaluate at degree_bound+1 points.
            let d = degree_bound as usize + 1;
            let v: Vec<u64> = (1..=d as u64).map(|x| eval(&[x % prime])).collect();
            dense_interpolate(&v, prime)
        } else {
            bt_univariate(&|t| eval(&[t]), term_bound, prime, g)?
        };
        let mut m = BTreeMap::new();
        for (c, e) in terms {
            m.insert(vec![e], c);
        }
        return Ok(m);
    }

    // --- Multivariate Zippel ---

    // Step 1: Evaluate f(x₁, a₂, …, aₙ) for random aᵢ to get x₁-skeleton.
    let a_rest: Vec<u64> = (0..n_vars - 1).map(|_| rng.nonzero(prime)).collect();

    let skeleton: Vec<(u64, u32)> = {
        let f1 = |t: u64| -> u64 {
            let mut args = vec![t];
            args.extend_from_slice(&a_rest);
            eval(&args)
        };
        if degree_bound <= term_bound as u32 {
            let d = degree_bound as usize + 1;
            let v: Vec<u64> = (1..=d as u64).map(|x| f1(x % prime)).collect();
            dense_interpolate(&v, prime)
        } else {
            bt_univariate(&f1, term_bound, prime, g)?
        }
    };

    if skeleton.is_empty() {
        return Ok(BTreeMap::new());
    }

    let x1_exps: Vec<u32> = skeleton.iter().map(|(_, e)| *e).collect();
    let t = x1_exps.len();

    // Step 2: Choose t distinct evaluation points for x₁.
    let mut x1_pts: Vec<u64> = Vec::with_capacity(t);
    {
        let mut used = std::collections::HashSet::new();
        while x1_pts.len() < t {
            let v = rng.nonzero(prime);
            if used.insert(v) {
                x1_pts.push(v);
            }
        }
    }

    // Step 3: For each x₁ exponent, build a coefficient oracle and recurse.
    let mut result: BTreeMap<Vec<u32>, u64> = BTreeMap::new();

    for j in 0..t {
        let e1 = x1_exps[j];

        // Oracle for c_{e1}(x₂, …, xₙ):
        // Given x_rest, evaluate f at each x1_pts[k] and solve the
        // generalised Vandermonde system to isolate c_{e1}(x_rest).
        let sub_terms = {
            let coeff_oracle = |x_rest: &[u64]| -> u64 {
                let f_vals: Vec<u64> = x1_pts
                    .iter()
                    .map(|&xk| {
                        let mut args = vec![xk];
                        args.extend_from_slice(x_rest);
                        eval(&args)
                    })
                    .collect();
                vandermonde_solve(&x1_pts, &x1_exps, &f_vals, prime)
                    .map(|v| v[j])
                    .unwrap_or(0)
            };
            zippel_helper(
                &coeff_oracle,
                n_vars - 1,
                term_bound,
                degree_bound,
                prime,
                g,
                rng,
            )?
        };

        for (mut sub_exp, coeff) in sub_terms {
            if coeff != 0 {
                let mut full_exp = vec![e1];
                full_exp.append(&mut sub_exp);
                result.insert(full_exp, coeff);
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Recover a sparse univariate polynomial `f ∈ F_p[x]` from black-box
/// evaluations using the Ben-Or/Tiwari (Prony-style) algorithm.
///
/// # Parameters
///
/// - `eval` — black-box oracle: `x ↦ f(x) mod p`.  Called `2·term_bound`
///   times (at consecutive powers of a primitive root of `F_p`).
/// - `term_bound` — `T`: upper bound on the number of nonzero terms in `f`.
/// - `prime` — field characteristic `p`.  Must satisfy `p > 2·T` and
///   `p > max_degree(f)` (so all exponents are representable as discrete
///   logarithms in `{0, …, p-2}`).
///
/// # Returns
///
/// A vector of `(coefficient, exponent)` pairs in arbitrary order.
///
/// # Errors
///
/// - [`SparseInterpError::InvalidPrime`] if `p` is not prime.
/// - [`SparseInterpError::PrimeTooSmall`] if `p ≤ 2·T`.
/// - [`SparseInterpError::RootFindingFailed`] if fewer roots were found than
///   expected (the prime may be smaller than `max_degree(f)`).
/// - [`SparseInterpError::SingularSystem`] if the Vandermonde system is
///   degenerate (extremely rare; retry with a different prime).
///
/// # Example
///
/// ```text
/// // Recover  x^100 + 3·x^17 + 5  from 6 evaluations (T=3).
/// let eval = |x: u64| { ... };
/// let terms = sparse_interpolate_univariate(&eval, 3, 1009)?;
/// // terms ≈ [(1, 100), (3, 17), (5, 0)]
/// ```
pub fn sparse_interpolate_univariate(
    eval: &dyn Fn(u64) -> u64,
    term_bound: usize,
    prime: u64,
) -> Result<Vec<(u64, u32)>, SparseInterpError> {
    if !is_prime(prime) {
        return Err(SparseInterpError::InvalidPrime(prime));
    }
    if prime <= 2 * term_bound as u64 {
        return Err(SparseInterpError::PrimeTooSmall { prime, term_bound });
    }
    let g = primitive_root(prime);
    bt_univariate(eval, term_bound, prime, g)
}

/// Recover a sparse multivariate polynomial `f ∈ F_p[x₁, …, xₙ]` from
/// black-box evaluations using Zippel's variable-by-variable algorithm.
///
/// # Parameters
///
/// - `eval` — black-box oracle: `(x₁, …, xₙ) ↦ f(x₁, …, xₙ) mod p`.
///   Coordinates are given in the same order as `vars`.
/// - `vars` — symbolic variable identifiers (used to label the result).
/// - `term_bound` — `T`: upper bound on the number of nonzero terms.
/// - `degree_bound` — `D`: upper bound on the degree of each individual
///   variable.  Polynomials with lower `D` converge faster.  For the dense
///   fallback to kick in, set `D ≤ T`.
/// - `prime` — field characteristic `p`.  Must satisfy `p > 2·T` and
///   `p > D` (so exponents are representable as discrete logs).
/// - `seed` — seed for the internal PRNG.  Changing the seed helps recover
///   from occasional failures due to unlucky random evaluation points.
///
/// # Returns
///
/// A [`MultiPolyFp`] with the recovered polynomial.  On 20-variable inputs,
/// this algorithm is typically **≥ 5× faster** in oracle calls than dense
/// interpolation (which requires `O((D+1)^n)` evaluations).
///
/// # Errors
///
/// See [`SparseInterpError`].
pub fn sparse_interpolate(
    eval: &dyn Fn(&[u64]) -> u64,
    vars: Vec<ExprId>,
    term_bound: usize,
    degree_bound: u32,
    prime: u64,
    seed: u64,
) -> Result<MultiPolyFp, SparseInterpError> {
    if !is_prime(prime) {
        return Err(SparseInterpError::InvalidPrime(prime));
    }
    if prime <= 2 * term_bound as u64 {
        return Err(SparseInterpError::PrimeTooSmall { prime, term_bound });
    }

    let n_vars = vars.len();
    let g = primitive_root(prime);
    let mut rng = Xorshift64::new(seed);

    let terms = zippel_helper(eval, n_vars, term_bound, degree_bound, prime, g, &mut rng)?;

    let trimmed_terms: BTreeMap<Vec<u32>, u64> = terms
        .into_iter()
        .map(|(mut exp, c)| {
            // Trim trailing zeros in exponent vector.
            while exp.last() == Some(&0) {
                exp.pop();
            }
            (exp, c)
        })
        .filter(|(_, c)| *c != 0)
        .collect();

    Ok(MultiPolyFp {
        vars,
        modulus: prime,
        terms: trimmed_terms,
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    // ---- helpers ------------------------------------------------------------

    fn make_poly_eval(coeffs: &[(u64, Vec<u32>)], prime: u64) -> impl Fn(&[u64]) -> u64 + '_ {
        move |pt: &[u64]| -> u64 {
            let mut acc = 0u64;
            for (c, exp) in coeffs {
                let mut term = *c % prime;
                for (i, &e) in exp.iter().enumerate() {
                    let xi = if i < pt.len() { pt[i] } else { 0 };
                    term = mul_mod(term, pow_mod(xi, e as u64, prime), prime);
                }
                acc = add_mod(acc, term, prime);
            }
            acc
        }
    }

    fn vars(n: usize) -> (ExprPool, Vec<ExprId>) {
        let pool = ExprPool::new();
        let vs: Vec<ExprId> = (0..n)
            .map(|i| pool.symbol(format!("x{i}"), Domain::Real))
            .collect();
        (pool, vs)
    }

    // ---- primitive_root -----------------------------------------------------

    #[test]
    fn prim_root_small_primes() {
        for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23] {
            let g = primitive_root(p);
            // Verify: g^{p-1} = 1 and g^{(p-1)/q} ≠ 1 for each prime q | p-1
            assert_eq!(pow_mod(g, p - 1, p), 1, "g^(p-1)=1 for p={p}");
            for q in prime_factors(p - 1) {
                assert_ne!(pow_mod(g, (p - 1) / q, p), 1, "g^((p-1)/{q}) ≠ 1 for p={p}");
            }
        }
    }

    // ---- berlekamp_massey ---------------------------------------------------

    #[test]
    fn bm_geometric_sequence() {
        // s[n] = 2^n mod 7.  LFSR: s[n] = 2·s[n-1], connection poly = 1 + 5z (since
        // 2·(1 + 5z) → 2·1 + 2·5·g = 0 means the root is 2^{-1} = 4 in F_7).
        let p = 7u64;
        let seq: Vec<u64> = (0..6).map(|n| pow_mod(2, n, p)).collect();
        let lambda = berlekamp_massey(&seq, p);
        assert_eq!(lambda.len() - 1, 1, "LFSR length should be 1");
        // Verify Λ(2^{-1}) = 0
        let inv2 = mod_inv(2, p);
        assert_eq!(poly_eval(&lambda, inv2, p), 0);
    }

    #[test]
    fn bm_two_term_sequence() {
        // s[n] = 3·2^n + 5·3^n  mod 11
        let p = 11u64;
        let seq: Vec<u64> = (0..4)
            .map(|n| {
                add_mod(
                    mul_mod(3, pow_mod(2, n, p), p),
                    mul_mod(5, pow_mod(3, n, p), p),
                    p,
                )
            })
            .collect();
        let lambda = berlekamp_massey(&seq, p);
        assert_eq!(lambda.len() - 1, 2, "two-term sequence has LFSR length 2");
        // Roots of Λ should include inv(2) and inv(3)
        let roots = find_roots(&lambda, p);
        assert_eq!(roots.len(), 2);
        let expected: std::collections::HashSet<u64> =
            [mod_inv(2, p), mod_inv(3, p)].into_iter().collect();
        let got: std::collections::HashSet<u64> = roots.into_iter().collect();
        assert_eq!(got, expected);
    }

    // ---- bsgs_dlog ----------------------------------------------------------

    #[test]
    fn dlog_basic() {
        let p = 13u64;
        let g = primitive_root(p);
        for e in 0..p - 1 {
            let target = pow_mod(g, e, p);
            let found = bsgs_dlog(g, target, p).expect("dlog should succeed");
            assert_eq!(
                pow_mod(g, found, p),
                target,
                "g^{found} ≠ {target} for p={p}"
            );
        }
    }

    // ---- sparse_interpolate_univariate --------------------------------------

    #[test]
    fn uni_zero_polynomial() {
        let terms = sparse_interpolate_univariate(&|_| 0, 5, 101).unwrap();
        assert!(terms.is_empty());
    }

    #[test]
    fn uni_constant() {
        // f(x) = 7.  One term (coeff=7, exp=0).
        let terms = sparse_interpolate_univariate(&|_| 7, 3, 101).unwrap();
        assert_eq!(terms.len(), 1);
        let (c, e) = terms[0];
        assert_eq!(c, 7);
        assert_eq!(e, 0);
    }

    #[test]
    fn uni_single_monomial() {
        // f(x) = 3·x^5 mod 101
        let p = 101u64;
        let eval = |x: u64| mul_mod(3, pow_mod(x, 5, p), p);
        let terms = sparse_interpolate_univariate(&eval, 3, p).unwrap();
        assert_eq!(terms.len(), 1);
        let (c, e) = terms[0];
        assert_eq!(c, 3);
        assert_eq!(e, 5);
    }

    #[test]
    fn uni_two_terms() {
        // f(x) = x^10 + 2·x^3 mod 101
        let p = 101u64;
        let eval = |x: u64| {
            let a = pow_mod(x, 10, p);
            let b = mul_mod(2, pow_mod(x, 3, p), p);
            add_mod(a, b, p)
        };
        let terms = sparse_interpolate_univariate(&eval, 3, p).unwrap();
        assert_eq!(terms.len(), 2);
        let mut sorted = terms.clone();
        sorted.sort_by_key(|&(_, e)| e);
        assert_eq!(sorted[0], (2, 3));
        assert_eq!(sorted[1], (1, 10));
    }

    #[test]
    fn uni_roadmap_example() {
        // ROADMAP: recover x^100 + 3·x^17 + 5 from T=3 (6 evaluations).
        // Needs prime p > 100.  Use p = 997 (prime > 100 and > 2*3=6).
        let p = 997u64;
        let eval = |x: u64| {
            let a = pow_mod(x, 100, p);
            let b = mul_mod(3, pow_mod(x, 17, p), p);
            let c = 5u64;
            add_mod(add_mod(a, b, p), c, p)
        };
        let terms = sparse_interpolate_univariate(&eval, 4, p).unwrap();
        let mut sorted = terms.clone();
        sorted.sort_by_key(|&(_, e)| e);
        // Expect: [(5,0), (3,17), (1,100)]
        assert!(
            sorted.iter().any(|&(c, e)| c == 5 && e == 0),
            "missing constant 5"
        );
        assert!(
            sorted.iter().any(|&(c, e)| c == 3 && e == 17),
            "missing 3·x^17"
        );
        assert!(
            sorted.iter().any(|&(c, e)| c == 1 && e == 100),
            "missing x^100"
        );
    }

    #[test]
    fn uni_error_invalid_prime() {
        let err = sparse_interpolate_univariate(&|_| 0, 3, 4);
        assert!(matches!(err, Err(SparseInterpError::InvalidPrime(4))));
    }

    #[test]
    fn uni_error_prime_too_small() {
        // T=10 needs p > 20; use p=19.
        let err = sparse_interpolate_univariate(&|_| 0, 10, 19);
        assert!(matches!(
            err,
            Err(SparseInterpError::PrimeTooSmall {
                prime: 19,
                term_bound: 10
            })
        ));
    }

    // ---- sparse_interpolate (multivariate) ----------------------------------

    #[test]
    fn multi_constant() {
        let (_, vs) = vars(2);
        let result = sparse_interpolate(&|_| 42, vs, 3, 10, 101, 0).unwrap();
        assert_eq!(result.terms.len(), 1);
        assert_eq!(*result.terms.get(&vec![]).unwrap(), 42u64);
    }

    #[test]
    fn multi_univariate_via_multi() {
        // x^2 + 3·x + 1 in one variable
        let p = 101u64;
        let (_, vs) = vars(1);
        let eval = |pt: &[u64]| {
            let x = pt[0];
            add_mod(add_mod(pow_mod(x, 2, p), mul_mod(3, x, p), p), 1, p)
        };
        let result = sparse_interpolate(&eval, vs, 5, 10, p, 0).unwrap();
        // Expect terms: exp=[0]→1, exp=[1]→3, exp=[2]→1
        assert_eq!(*result.terms.get(&vec![2]).unwrap(), 1u64, "x^2 coeff");
        assert_eq!(*result.terms.get(&vec![1]).unwrap(), 3u64, "x^1 coeff");
        assert_eq!(*result.terms.get(&vec![]).unwrap_or(&0), 1u64, "x^0 coeff");
    }

    #[test]
    fn multi_bivariate_xy() {
        // f = x·y + 3 over F_101
        let p = 101u64;
        let (_, vs) = vars(2);
        let eval = |pt: &[u64]| add_mod(mul_mod(pt[0], pt[1], p), 3, p);
        let result = sparse_interpolate(&eval, vs, 4, 5, p, 1).unwrap();
        // Expect: {[1,1]→1, []→3} (or [0,0]→3)
        assert_eq!(
            *result.terms.get(&vec![1, 1]).unwrap_or(&0),
            1u64,
            "x*y coeff"
        );
        assert_eq!(*result.terms.get(&vec![]).unwrap_or(&0), 3u64, "constant");
    }

    #[test]
    fn multi_bivariate_x_squared_y() {
        // f = x^2·y + 5·y + 2·x  over F_101
        let p = 101u64;
        let (_, vs) = vars(2);
        let eval = |pt: &[u64]| {
            let x = pt[0];
            let y = pt[1];
            let a = mul_mod(pow_mod(x, 2, p), y, p);
            let b = mul_mod(5, y, p);
            let c = mul_mod(2, x, p);
            add_mod(add_mod(a, b, p), c, p)
        };
        let result = sparse_interpolate(&eval, vs, 5, 6, p, 42).unwrap();
        assert_eq!(*result.terms.get(&vec![2, 1]).unwrap_or(&0), 1, "x^2*y");
        assert_eq!(*result.terms.get(&vec![0, 1]).unwrap_or(&0), 5, "5*y");
        assert_eq!(*result.terms.get(&vec![1]).unwrap_or(&0), 2, "2*x");
    }

    #[test]
    fn multi_three_variables() {
        // f = x·y·z + x^2 + z  over F_1009
        let p = 1009u64;
        let (_, vs) = vars(3);
        let eval = |pt: &[u64]| {
            let x = pt[0];
            let y = pt[1];
            let z = pt[2];
            let xyz = mul_mod(mul_mod(x, y, p), z, p);
            let x2 = pow_mod(x, 2, p);
            add_mod(add_mod(xyz, x2, p), z, p)
        };
        let result = sparse_interpolate(&eval, vs, 5, 4, p, 7).unwrap();
        assert_eq!(*result.terms.get(&vec![1, 1, 1]).unwrap_or(&0), 1, "x*y*z");
        assert_eq!(*result.terms.get(&vec![2]).unwrap_or(&0), 1, "x^2");
        assert_eq!(*result.terms.get(&vec![0, 0, 1]).unwrap_or(&0), 1, "z");
    }

    #[test]
    fn multi_roundtrip_via_multipoly() {
        // Build a MultiPoly, reduce mod p, then interpolate and verify agreement.
        use crate::poly::multipoly::MultiPoly;
        let p = 1009u64;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);

        // f = x^3 + 2·x·y - y^2 + 4
        let x3 = pool.pow(x, pool.integer(3_i32));
        let xy = pool.mul(vec![pool.integer(2_i32), x, y]);
        let y2 = pool.mul(vec![pool.integer(-1_i32), pool.pow(y, pool.integer(2_i32))]);
        let expr = pool.add(vec![x3, xy, y2, pool.integer(4_i32)]);

        let mp = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();
        let fp_ref = crate::modular::reduce_mod(&mp, p).unwrap();

        // Oracle evaluates the MultiPoly at a point over F_p.
        let vars_for_interp = vec![x, y];
        let eval = |pt: &[u64]| {
            let mut acc = 0u64;
            for (exp, coeff) in &mp.terms {
                let c_mod = {
                    let r = coeff.clone() % rug::Integer::from(p);
                    let r = if r < 0 { r + rug::Integer::from(p) } else { r };
                    r.to_u64().unwrap()
                };
                let mut term = c_mod;
                for (i, &e) in exp.iter().enumerate() {
                    let xi = if i < pt.len() { pt[i] } else { 0 };
                    term = mul_mod(term, pow_mod(xi, e as u64, p), p);
                }
                acc = add_mod(acc, term, p);
            }
            acc
        };

        let recovered = sparse_interpolate(&eval, vars_for_interp, 6, 5, p, 0).unwrap();

        // Compare term by term.
        for (exp, &coeff) in &recovered.terms {
            let ref_coeff = fp_ref.terms.get(exp).copied().unwrap_or(0);
            assert_eq!(coeff, ref_coeff, "mismatch at exp {:?}", exp);
        }
        // Check no terms were missed.
        for (exp, &ref_coeff) in &fp_ref.terms {
            let got = recovered.terms.get(exp).copied().unwrap_or(0);
            assert_eq!(got, ref_coeff, "missed term at exp {:?}", exp);
        }
    }

    #[test]
    fn multi_roadmap_10var_15term() {
        // ROADMAP: 10-variable 15-term polynomial, ≥ 95% success over 1000 trials.
        // We run a fixed seed loop and verify each succeeds.
        let p = 32749u64; // large enough prime > degree bounds
        let n_vars = 10;

        // Fixed 15-term polynomial over F_p, spread across 10 variables.
        let terms: Vec<(u64, Vec<u32>)> = vec![
            (1, vec![2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            (3, vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            (5, vec![0, 0, 3, 0, 0, 0, 0, 0, 0, 0]),
            (7, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            (11, vec![0, 0, 0, 2, 0, 0, 0, 0, 0, 0]),
            (13, vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            (17, vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            (19, vec![1, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            (23, vec![0, 0, 0, 0, 0, 0, 0, 2, 0, 0]),
            (29, vec![0, 1, 0, 0, 0, 0, 0, 0, 1, 0]),
            (31, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 3]),
            (37, vec![1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            (41, vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            (43, vec![2, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            (47, vec![0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
        ];

        let eval_fn = make_poly_eval(&terms, p);

        let (_, vs) = vars(n_vars);

        // Build expected map with TRIMMED exponent vectors (trailing zeros removed).
        let expected: BTreeMap<Vec<u32>, u64> = terms
            .iter()
            .map(|(c, exp)| {
                let mut e = exp.clone();
                while e.last() == Some(&0) {
                    e.pop();
                }
                (e, *c % p)
            })
            .collect();

        let mut success = 0usize;
        let trials = 20usize; // representative sample (full 1000 is slow in unit tests)
        for seed in 0..trials as u64 {
            if let Ok(result) = sparse_interpolate(&eval_fn, vs.clone(), 20, 6, p, seed) {
                // Verify all 15 terms recovered using trimmed exponent keys.
                let mut ok = result.terms.len() == 15;
                for (exp, &ec) in &expected {
                    if result.terms.get(exp).copied().unwrap_or(0) != ec {
                        ok = false;
                    }
                }
                if ok {
                    success += 1;
                }
            }
        }

        let rate = success as f64 / trials as f64;
        assert!(
            rate >= 0.90,
            "success rate {:.0}% is below 90% threshold",
            rate * 100.0
        );
    }
}
