//! The Delsarte linear-programming bound on `A_q(n, d)`, solved exactly.
//!
//! `A_q(n, d)` is the largest number of words in *any* code (linear or not) of
//! length `n` over an alphabet of size `q` whose pairwise Hamming distances are
//! all at least `d`. Delsarte's observation is that the **distance
//! distribution** `A_0 … A_n` of such a code — `A_i` the average number of
//! codewords at distance `i` from a codeword — satisfies a system of linear
//! inequalities, so maximising `Σ A_i` over that system bounds `A_q(n, d)` from
//! above:
//!
//! ```text
//!   maximise   Σ_i A_i
//!   subject to A_0 = 1
//!              A_i ≥ 0                       for all i
//!              A_i = 0                       for 1 ≤ i < d
//!              Σ_i A_i K_k(i; n, q) ≥ 0      for k = 1 … n
//! ```
//!
//! The last family is dual feasibility in the Hamming association scheme: the
//! `K_k` are its eigenvalues ([`super::krawtchouk`]), and the inequality says a
//! certain Gram matrix has a non-negative entry.
//!
//! # Why this is a *certified* bound and not an estimate
//!
//! Every coefficient is a `rug::Rational`, the simplex
//! ([`crate::real::sos::lp`]) pivots in exact arithmetic under Bland's rule, and
//! nothing is rounded anywhere. That alone would only mean the arithmetic was
//! exact — it would not prove the number *is* an upper bound, because a buggy
//! simplex that stopped early would report something too small, and too small
//! is the dangerous direction.
//!
//! So the bound returned here is read off the **dual** programme, which carries
//! its own proof. If `y_1 … y_n ≥ 0` satisfy
//!
//! ```text
//!   Σ_k y_k K_k(i; n, q) ≤ −1        for every i in d … n
//! ```
//!
//! then for any code `C` of length `n` and minimum distance ≥ `d`,
//!
//! ```text
//!   0 ≤ Σ_k y_k (Σ_i A_i K_k(i))              (each inner sum is ≥ 0, y ≥ 0)
//!     = Σ_k y_k K_k(0) + Σ_{i ≥ d} A_i (Σ_k y_k K_k(i))
//!     ≤ Σ_k y_k K_k(0) − Σ_{i ≥ d} A_i
//! ```
//!
//! so `|C| = 1 + Σ_{i ≥ d} A_i ≤ 1 + Σ_k y_k K_k(0)`. That derivation needs
//! nothing from the solver: it is a valid bound for *any* non-negative `y`
//! meeting the inequalities. [`delsarte_lp_bound`] solves the dual, checks the
//! returned `y` against those inequalities entry by entry in exact arithmetic,
//! and refuses (`E-CODE-007`) if the check fails. The `y` is kept on the result
//! as [`DelsarteBound::certificate`] so a caller can re-verify it — or hand it
//! to a referee — without trusting this crate at all.
//!
//! The primal is solved too, and its optimum must equal the dual's. That does
//! not affect soundness; it establishes that the bound is the *best* the linear
//! programme can give, and it is a strong cross-check on the simplex.
//!
//! # Feasibility and boundedness, which are not assumed
//!
//! The primal is always feasible: `A_i = 0` for `i ≥ d` satisfies every
//! constraint, since `Σ_i A_i K_k(i) = K_k(0) = C(n,k)(q−1)^k ≥ 0`. It is
//! always bounded, because `Σ_{k=0}^{n} K_k(i) = q^n [i = 0]` (set `z = 1` in
//! the generating function `(1+(q−1)z)^{n−i}(1−z)^i`), so summing the `n`
//! constraints gives `Σ_{i ≥ d} A_i ≤ q^n − 1`. An `Infeasible` or `Unbounded`
//! status from the simplex is therefore a bug, and is reported as one rather
//! than folded into a number.
//!
//! # Scope
//!
//! * `q` need not be a prime power — the Hamming scheme `H(n, q)` is defined
//!   for any alphabet size, and so is `A_q(n, d)`.
//! * This is the **plain** Delsarte programme. The strengthenings that give the
//!   best known bounds for specific `(n, d)` — Schrijver's semidefinite
//!   programme, the parity and Lloyd-type extra constraints, the
//!   Mallows–Odlyzko–Sloane inequalities — are not here. Nothing returned is
//!   ever *smaller* than the true `A_q(n, d)`; it is simply not always the
//!   tightest bound in the literature.
//! * Length is capped at [`MAX_LP_LENGTH`]. The programme is `n × (n−d+1)` in
//!   exact rationals whose entries reach `C(n, n/2)(q−1)^{n/2}`, and the cost
//!   is empirical rather than mathematical.

use rug::ops::Pow;
use rug::{Integer, Rational};

use crate::real::sos::lp::{Lp, LpStatus, Rel};

use super::error::CodingError;
use super::krawtchouk::krawtchouk;

/// Cap on `n` for the Delsarte programme.
///
/// Empirical, not mathematical: the simplex is dense, exact and uses Bland's
/// rule, so the cost grows steeply in `n`, and the Krawtchouk entries grow with
/// `C(n, n/2)`. Measured on one unoptimised (debug) build, solving both sides
/// of `A_2(n, ⌊n/3⌋)`:
///
/// | `n` | time |
/// |----:|-----:|
/// |  24 | 0.2 s |
/// |  32 | 0.4 s |
/// |  48 | 4.6 s |
/// |  64 |  21 s |
/// |  96 | 3.2 min |
/// | 128 | did not finish in 25 min |
///
/// The cap sits where the curve does: `n = 96` is already a coffee break and
/// `n = 128` is not a computation anyone is waiting on interactively. Asking
/// past it is `E-CODE-005` rather than a hang.
pub const MAX_LP_LENGTH: usize = 96;

/// An exact Delsarte bound on `A_q(n, d)`, together with the dual certificate
/// that proves it.
///
/// [`DelsarteBound::bound`] is an upper bound on the size of **every** code of
/// length `n` over an alphabet of size `q` with minimum distance at least `d`,
/// linear or not. It is certified in the sense of the module docs: the
/// non-negative multipliers in [`DelsarteBound::certificate`] have been checked
/// against the dual inequalities in exact rational arithmetic, and that check
/// is what makes the number a bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelsarteBound {
    n: usize,
    d: usize,
    q: u64,
    optimum: Rational,
    bound: Integer,
    distribution: Vec<Rational>,
    certificate: Vec<Rational>,
}

impl DelsarteBound {
    /// The length.
    pub fn length(&self) -> usize {
        self.n
    }

    /// The minimum distance.
    pub fn distance(&self) -> usize {
        self.d
    }

    /// The alphabet size.
    pub fn alphabet_size(&self) -> u64 {
        self.q
    }

    /// The exact optimum of the linear programme, as a rational.
    ///
    /// Usually not an integer — `A_2(11, 3)` optimises at `2048/11` — which is
    /// why [`DelsarteBound::bound`] floors it.
    pub fn optimum(&self) -> &Rational {
        &self.optimum
    }

    /// `⌊optimum⌋`: the certified upper bound on `A_q(n, d)`.
    ///
    /// A code has an integer number of words, so flooring is sound.
    pub fn bound(&self) -> &Integer {
        &self.bound
    }

    /// The optimal distance distribution `A_0 … A_n` the programme found.
    ///
    /// `A_0 = 1` and `A_i = 0` for `1 ≤ i < d` by construction. This is a
    /// *vertex of the relaxation*, not the distance distribution of any code:
    /// entries are generally fractional and no code need attain them.
    pub fn distribution(&self) -> &[Rational] {
        &self.distribution
    }

    /// The dual certificate `y_1 … y_n ≥ 0` with `Σ_k y_k K_k(i) ≤ −1` for
    /// `d ≤ i ≤ n`, indexed from `k = 1`.
    ///
    /// Together with the derivation in the module docs this is a
    /// self-contained proof of [`DelsarteBound::bound`]; see
    /// [`DelsarteBound::verify_certificate`].
    pub fn certificate(&self) -> &[Rational] {
        &self.certificate
    }

    /// Re-check the dual certificate from scratch, in exact arithmetic.
    ///
    /// Recomputes every Krawtchouk value and re-tests `y ≥ 0`,
    /// `Σ_k y_k K_k(i) ≤ −1` for `d ≤ i ≤ n`, and
    /// `1 + Σ_k y_k K_k(0) = optimum`. [`delsarte_lp_bound`] already refuses
    /// unless this passes, so it returning `false` would mean the struct was
    /// mutated or built by other means; it is public because a caller who wants
    /// to audit the bound should not have to take this crate's word for it.
    pub fn verify_certificate(&self) -> bool {
        if self.certificate.len() != self.n {
            return false;
        }
        if self.certificate.iter().any(|y| *y < 0) {
            return false;
        }
        for i in self.d..=self.n {
            let mut acc = Rational::from(0);
            for (idx, y) in self.certificate.iter().enumerate() {
                if *y == 0 {
                    continue;
                }
                let kk = Rational::from(krawtchouk(idx + 1, i as i64, self.n, self.q));
                acc += Rational::from(y * &kk);
            }
            if acc > -1 {
                return false;
            }
        }
        let mut value = Rational::from(1);
        for (idx, y) in self.certificate.iter().enumerate() {
            if *y == 0 {
                continue;
            }
            let kk = Rational::from(krawtchouk(idx + 1, 0, self.n, self.q));
            value += Rational::from(y * &kk);
        }
        value == self.optimum
    }

    /// Re-check that the reported distribution is feasible for the primal.
    ///
    /// `A_0 = 1`, `A_i ≥ 0`, `A_i = 0` below `d`, and every Krawtchouk pairing
    /// non-negative. A feasible primal point of the same value as the dual
    /// certificate is what makes the bound *tight for this programme* rather
    /// than merely valid.
    pub fn verify_distribution(&self) -> bool {
        if self.distribution.len() != self.n + 1 || self.distribution[0] != 1 {
            return false;
        }
        if self.distribution.iter().any(|a| *a < 0) {
            return false;
        }
        if self.distribution[1..self.d.min(self.n + 1)]
            .iter()
            .any(|a| *a != 0)
        {
            return false;
        }
        for k in 1..=self.n {
            let mut acc = Rational::from(0);
            for (i, ai) in self.distribution.iter().enumerate() {
                if *ai == 0 {
                    continue;
                }
                let kk = Rational::from(krawtchouk(k, i as i64, self.n, self.q));
                acc += Rational::from(ai * &kk);
            }
            if acc < 0 {
                return false;
            }
        }
        let total = self
            .distribution
            .iter()
            .fold(Rational::from(0), |a, v| a + v.clone());
        total == self.optimum
    }
}

/// The Delsarte linear-programming upper bound on `A_q(n, d)`.
///
/// ```
/// use alkahest_cas::experimental::delsarte_lp_bound;
/// use rug::Integer;
///
/// // The [7,4,3] Hamming code is perfect, and the LP knows it.
/// let b = delsarte_lp_bound(7, 3, 2).unwrap();
/// assert_eq!(*b.bound(), Integer::from(16));
/// assert!(b.verify_certificate());
/// ```
///
/// # Errors
///
/// * `E-CODE-001` — `n = 0`.
/// * `E-CODE-002` — `d` outside `1 ..= n`.
/// * `E-CODE-005` — `n` past [`MAX_LP_LENGTH`].
/// * `E-CODE-008` — `q < 2`.
/// * `E-CODE-007` — the simplex did not return an optimal vertex, the primal
///   and dual optima disagreed, or a returned vertex failed its exact
///   feasibility check. All three are bugs; the uncertified number is withheld.
pub fn delsarte_lp_bound(n: usize, d: usize, q: u64) -> Result<DelsarteBound, CodingError> {
    if n == 0 {
        return Err(CodingError::InvalidLength { n });
    }
    if d == 0 || d > n {
        return Err(CodingError::InvalidDistance { d, n });
    }
    if q < 2 {
        return Err(CodingError::InvalidAlphabet {
            q: q.to_string(),
            reason: "an alphabet needs at least two symbols".to_string(),
        });
    }
    if n > MAX_LP_LENGTH {
        return Err(CodingError::LengthTooLarge {
            n,
            cap: MAX_LP_LENGTH,
        });
    }

    // kvals[k][i] = K_k(i; n, q) for k = 0..=n, i = 0..=n.
    let kvals: Vec<Vec<Rational>> = (0..=n)
        .map(|k| {
            (0..=n)
                .map(|i| Rational::from(krawtchouk(k, i as i64, n, q)))
                .collect()
        })
        .collect();

    // ---------------------------------------------------------------------
    // Primal: variables A_d … A_n, objective max Σ A_i (minimise −Σ A_i).
    // ---------------------------------------------------------------------
    let free: Vec<usize> = (d..=n).collect();
    let mut primal = Lp::new(free.len());
    for (v, _) in free.iter().enumerate() {
        primal.set_objective(v, Rational::from(-1));
    }
    for row in kvals.iter().take(n + 1).skip(1) {
        let coeffs: Vec<Rational> = free.iter().map(|i| row[*i].clone()).collect();
        primal.constrain(coeffs, Rel::Ge, -row[0].clone());
    }
    let primal_x = match primal.solve() {
        LpStatus::Optimal(x) => x,
        other => {
            return Err(CodingError::LpFailure {
                detail: format!(
                    "the primal programme for A_{q}({n}, {d}) returned {other:?}, but it is \
                     feasible (A = 0 works) and bounded (Σ A_i ≤ q^n − 1)"
                ),
            })
        }
    };
    let primal_value = Rational::from(1)
        + primal_x
            .iter()
            .fold(Rational::from(0), |a, v| a + v.clone());

    // ---------------------------------------------------------------------
    // Dual: variables y_1 … y_n ≥ 0, minimise Σ_k K_k(0) y_k subject to
    // Σ_k y_k K_k(i) ≤ −1 for every i in d … n.
    // ---------------------------------------------------------------------
    let mut dual = Lp::new(n);
    for (k, row) in kvals.iter().enumerate().take(n + 1).skip(1) {
        dual.set_objective(k - 1, row[0].clone());
    }
    for i in d..=n {
        let coeffs: Vec<Rational> = kvals
            .iter()
            .take(n + 1)
            .skip(1)
            .map(|row| row[i].clone())
            .collect();
        dual.constrain(coeffs, Rel::Le, Rational::from(-1));
    }
    let dual_y = match dual.solve() {
        LpStatus::Optimal(y) => y,
        other => {
            return Err(CodingError::LpFailure {
                detail: format!(
                    "the dual programme for A_{q}({n}, {d}) returned {other:?}; by LP duality \
                     it must have an optimum whenever the primal does"
                ),
            })
        }
    };
    let dual_value = Rational::from(1)
        + (1..=n).fold(Rational::from(0), |a, k| {
            a + Rational::from(&dual_y[k - 1] * &kvals[k][0])
        });

    if primal_value != dual_value {
        return Err(CodingError::LpFailure {
            detail: format!(
                "strong duality failed for A_{q}({n}, {d}): the primal optimum is \
                 {primal_value} and the dual optimum is {dual_value}"
            ),
        });
    }

    let mut distribution = vec![Rational::from(0); n + 1];
    distribution[0] = Rational::from(1);
    for (v, i) in free.iter().enumerate() {
        distribution[*i] = primal_x[v].clone();
    }

    let bound = dual_value
        .numer()
        .clone()
        .div_rem_floor(dual_value.denom().clone())
        .0;
    let out = DelsarteBound {
        n,
        d,
        q,
        optimum: dual_value,
        bound,
        distribution,
        certificate: dual_y,
    };

    // The certificate is what makes this a bound rather than a number the
    // solver liked. Re-check it here rather than trusting the simplex.
    if !out.verify_certificate() {
        return Err(CodingError::LpFailure {
            detail: format!(
                "the dual certificate for A_{q}({n}, {d}) failed its own exact feasibility \
                 check; the bound it would have supported is not proved and is withheld"
            ),
        });
    }
    if !out.verify_distribution() {
        return Err(CodingError::LpFailure {
            detail: format!(
                "the primal vertex for A_{q}({n}, {d}) is not feasible for the programme it \
                 was supposed to optimise"
            ),
        });
    }
    Ok(out)
}

/// The Singleton bound `A_q(n, d) ≤ q^(n−d+1)`.
///
/// Elementary and always valid: deleting `d − 1` coordinates from a
/// distance-`d` code keeps all its words distinct.
///
/// # Errors
///
/// `E-CODE-001` for `n = 0`, `E-CODE-002` for `d` outside `1 ..= n`,
/// `E-CODE-008` for `q < 2`.
pub fn singleton_bound(n: usize, d: usize, q: u64) -> Result<Integer, CodingError> {
    check_ndq(n, d, q)?;
    Ok(Integer::from(q).pow((n - d + 1) as u32))
}

/// The Hamming (sphere-packing) bound
/// `A_q(n, d) ≤ q^n / Σ_{i=0}^{t} C(n,i)(q−1)^i`, `t = ⌊(d−1)/2⌋`, floored.
///
/// # Errors
///
/// `E-CODE-001` for `n = 0`, `E-CODE-002` for `d` outside `1 ..= n`,
/// `E-CODE-008` for `q < 2`.
pub fn hamming_bound(n: usize, d: usize, q: u64) -> Result<Integer, CodingError> {
    check_ndq(n, d, q)?;
    let t = (d - 1) / 2;
    let qm1 = Integer::from(q) - 1u32;
    let mut volume = Integer::from(0);
    for i in 0..=t {
        volume += super::krawtchouk::binomial(n, i) * qm1.clone().pow(i as u32);
    }
    let total = Integer::from(q).pow(n as u32);
    Ok(total.div_rem_floor(volume).0)
}

fn check_ndq(n: usize, d: usize, q: u64) -> Result<(), CodingError> {
    if n == 0 {
        return Err(CodingError::InvalidLength { n });
    }
    if d == 0 || d > n {
        return Err(CodingError::InvalidDistance { d, n });
    }
    if q < 2 {
        return Err(CodingError::InvalidAlphabet {
            q: q.to_string(),
            reason: "an alphabet needs at least two symbols".to_string(),
        });
    }
    Ok(())
}
