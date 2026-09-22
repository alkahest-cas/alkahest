//! Krawtchouk polynomials `K_k(x; n, q)` in exact arithmetic.
//!
//! These are the eigenvalues of the Hamming association scheme `H(n, q)`: the
//! `k`-th eigenvalue of the adjacency matrix of "distance `k`" on `GF(q)^n`,
//! evaluated at `x`, is `K_k(x; n, q)`. Everything the Delsarte linear
//! programme and the MacWilliams identity do is expressed in them, so they are
//! computed exactly and never in floating point — a rounded eigenvalue turns a
//! *bound* into an estimate.
//!
//! The definition used here is
//!
//! ```text
//!                k
//!   K_k(x; n,q) = Σ  (-1)^j (q-1)^(k-j) C(x, j) C(n-x, k-j)
//!               j=0
//! ```
//!
//! with `C(a, j)` the generalised binomial `a(a-1)…(a-j+1) / j!`, so `x` may be
//! any integer (values outside `0..=n` are meaningful and are used when
//! checking the recurrence at its ends). Equivalently `K_k(x; n, q)` is the
//! coefficient of `z^k` in `(1 + (q-1)z)^(n-x) (1 - z)^x`, which is where the
//! identities below come from.
//!
//! # What is here
//!
//! * [`krawtchouk`] — the value at an integer point, as a [`rug::Integer`].
//! * [`krawtchouk_poly`] — the coefficients of `K_k` as a polynomial in `x`,
//!   ascending, as [`rug::Rational`]s. The polynomial has rational (not
//!   integer) coefficients in general: `K_2(x; 2, 2) = 1 - 4x + 2x^2` is
//!   integral, but `K_2(x; 3, 2) = 1 - 4x + 2x^2` scaled by the `1/k!` in the
//!   definition is not in general, and the leading coefficient is
//!   `(-q)^k / k!`.
//!
//! # Identities this module is tested against
//!
//! * `K_k(0) = C(n, k) (q-1)^k`.
//! * `K_0(x) = 1` and `K_1(x) = (q-1)n - qx`.
//! * The three-term recurrence
//!   `(k+1) K_{k+1}(x) = [k + (q-1)(n-k) - qx] K_k(x) - (q-1)(n-k+1) K_{k-1}(x)`,
//!   which is also how [`krawtchouk_poly`] is built, so testing the two against
//!   each other tests the recurrence against the closed form.
//! * Orthogonality:
//!   `Σ_i C(n,i)(q-1)^i K_k(i) K_l(i) = q^n C(n,k) (q-1)^k δ_{kl}`.
//! * Reciprocity: `C(n,i)(q-1)^i K_k(i) = C(n,k)(q-1)^k K_i(k)`.
//! * `Σ_{k=0}^{n} K_k(i) = q^n [i = 0]` — the identity that makes the Delsarte
//!   programme bounded.
//!
//! # Scope
//!
//! `q` is only an *alphabet size* here, not a field order: the Hamming scheme
//! `H(n, q)` is defined for any `q ≥ 2`, and so is the Delsarte bound
//! `A_q(n, d)`. Nothing in this file requires `q` to be a prime power. Values
//! for `k > n` are zero, which the definition already gives.

use rug::ops::Pow;
use rug::{Integer, Rational};

/// The generalised binomial coefficient `C(a, j) = a(a-1)…(a-j+1) / j!`.
///
/// Defined for any integer `a`, including negative ones; agrees with the usual
/// binomial when `0 ≤ j ≤ a` and is zero when `a` is a non-negative integer
/// smaller than `j`.
pub fn binomial_generalised(a: &Integer, j: usize) -> Integer {
    let mut num = Integer::from(1);
    let mut term = a.clone();
    for _ in 0..j {
        num *= &term;
        term -= 1;
    }
    if num == 0 {
        return num;
    }
    let mut fact = Integer::from(1);
    for t in 2..=j {
        fact *= t as u32;
    }
    // Exact by construction: a product of j consecutive integers is divisible
    // by j!.
    num / fact
}

/// The ordinary binomial coefficient `C(n, k)`, zero when `k > n`.
pub fn binomial(n: usize, k: usize) -> Integer {
    if k > n {
        return Integer::from(0);
    }
    Integer::from(n).binomial(k as u32)
}

/// `K_k(x; n, q)`, the Krawtchouk polynomial evaluated at the integer `x`.
///
/// Exact: every intermediate is a [`rug::Integer`]. `x` may lie outside
/// `0..=n`.
///
/// ```
/// use alkahest_cas::experimental::krawtchouk;
/// use rug::Integer;
///
/// // K_k(0) = C(n, k) (q-1)^k
/// assert_eq!(krawtchouk(3, 0, 7, 2), Integer::from(35));
/// // The [7,4] Hamming code's dual has a weight-4 word in every position:
/// // K_1(x; 7, 2) = 7 - 2x
/// assert_eq!(krawtchouk(1, 3, 7, 2), Integer::from(1));
/// ```
pub fn krawtchouk(k: usize, x: i64, n: usize, q: u64) -> Integer {
    let xi = Integer::from(x);
    let n_minus_x = Integer::from(n) - &xi;
    let qm1 = Integer::from(q) - 1u32;

    let mut total = Integer::from(0);
    for j in 0..=k {
        let left = binomial_generalised(&xi, j);
        if left == 0 {
            continue;
        }
        let right = binomial_generalised(&n_minus_x, k - j);
        if right == 0 {
            continue;
        }
        let mut term = left * right;
        term *= qm1.clone().pow((k - j) as u32);
        if j % 2 == 1 {
            total -= term;
        } else {
            total += term;
        }
    }
    total
}

/// `K_k(· ; n, q)` as a polynomial in `x`: coefficients ascending, length
/// `min(k, n) + 1` (or `[0]` when `k > n`).
///
/// Built from the three-term recurrence
/// `(k+1) K_{k+1} = [k + (q-1)(n-k) - qx] K_k - (q-1)(n-k+1) K_{k-1}`,
/// which is why comparing it against [`krawtchouk`] — built from the closed
/// form — is a real test of both.
pub fn krawtchouk_poly(k: usize, n: usize, q: u64) -> Vec<Rational> {
    if k > n {
        return vec![Rational::from(0)];
    }
    let qm1 = Rational::from(q) - 1u32;
    let qr = Rational::from(q);

    // K_{-1} = 0, K_0 = 1.
    let mut prev: Vec<Rational> = vec![Rational::from(0)];
    let mut cur: Vec<Rational> = vec![Rational::from(1)];
    for kk in 0..k {
        // linear = [kk + (q-1)(n-kk)] - q x
        let c0 = Rational::from(kk) + qm1.clone() * Rational::from(n - kk);
        let mut next: Vec<Rational> = vec![Rational::from(0); cur.len() + 1];
        for (i, ci) in cur.iter().enumerate() {
            next[i] += Rational::from(&c0 * ci);
            next[i + 1] -= Rational::from(&qr * ci);
        }
        let tail = qm1.clone() * Rational::from(n - kk + 1);
        for (i, pi) in prev.iter().enumerate() {
            next[i] -= Rational::from(&tail * pi);
        }
        let scale = Rational::from(kk + 1);
        for c in next.iter_mut() {
            *c /= &scale;
        }
        prev = cur;
        cur = next;
    }
    cur
}

/// `Σ_{i=0}^{n} A_i K_k(i; n, q)` — one Delsarte dual-feasibility form,
/// evaluated exactly.
///
/// This is the quantity that must be non-negative for every `k` when `A` is the
/// distance distribution of *any* code over an alphabet of size `q`, linear or
/// not. It is the only place the Hamming scheme's eigenvalues meet a code in
/// this module, so both the MacWilliams transform and the linear programme go
/// through it.
pub fn krawtchouk_pairing(a: &[Integer], k: usize, n: usize, q: u64) -> Integer {
    let mut total = Integer::from(0);
    for (i, ai) in a.iter().enumerate() {
        if *ai == 0 {
            continue;
        }
        total += ai * krawtchouk(k, i as i64, n, q);
    }
    total
}
