//! Evaluation of holonomic sequences and binomial coefficients modulo `p^k`.
//!
//! # What this is for
//!
//! A supercongruence is a statement like `A(p−1) ≡ 1 (mod p⁴)` about a
//! P-recursive sequence — Apéry numbers, Franel numbers, Domb numbers, the
//! Almkvist–Zudilin family. Checking one at a single prime by computing
//! `A(p−1)` over `ℤ` and reducing is quadratic in `p`: the integers involved
//! have `Θ(p)` digits. Checking a *range* of primes that way is what turns a
//! millisecond of arithmetic into a minute of bignum work.
//!
//! Everything here works the other way round: reduce first, then iterate. The
//! recurrence
//!
//! ```text
//! Σ_{i=0}^{J} a_i(n) · S(n+i) = b(n)
//! ```
//!
//! is a statement over `ℤ`, so it is also a statement over `ℤ/p^K`. Running it
//! forward in `ℤ/p^K` costs `O(N)` machine-word multiplications and `O(1)`
//! memory, whatever the size of `S(N)` over `ℤ`.
//!
//! # The pitfall: singular indices
//!
//! Stepping the recurrence forward means solving for the top term,
//!
//! ```text
//! S(n+J) = ( b(n) − Σ_{i<J} a_i(n)·S(n+i) ) / a_J(n),
//! ```
//!
//! and `a_J(n)` need not be invertible mod `p`. For the Apéry recurrence
//! `a_2(n) = (n+2)³`, every `n ≡ −2 (mod p)` is such an index — and that is
//! exactly the index a supercongruence sweep walks through when it asks for
//! `A(p)` rather than `A(p−1)`. A naive implementation either divides by a
//! non-unit (garbage) or, worse, computes `a_J(n)⁻¹` by a routine that silently
//! returns something for a non-unit.
//!
//! This module handles them by *lifting*, not by ignoring:
//!
//! 1. A first pass computes `v = v_p(a_J(n))` for every step, using exact
//!    integer arithmetic at the (rare) indices where the residue alone cannot
//!    decide `v`. If `a_J(n) = 0` exactly, the recurrence does not determine
//!    `S(n+J)` at all and the call refuses
//!    ([`ModularError::PAdicallyUndetermined`], `E-HOLO-007`).
//! 2. The total loss `L = Σ v` is known before a single sequence value is
//!    computed, so the forward pass runs at *working* precision `K = k + L`.
//!    Each singular step divides numerator and denominator by `p^v`, costing
//!    exactly `v` digits of precision — and the budget was raised by `L` up
//!    front, so the value at the target index is still known to the full `p^k`
//!    that was asked for.
//! 3. The division by `p^v` is *checked*: if the numerator is not divisible by
//!    `p^v` then `S(n+J)` is not a `p`-adic integer and the call refuses
//!    (`E-HOLO-007`) rather than producing a residue for a value that has none.
//!    This is not hypothetical — the harmonic numbers are holonomic and
//!    `H_p = H_{p−1} + 1/p` has `v_p = −1`.
//! 4. If `k + L` needs a modulus past the machine-word backend, or `p` is not
//!    prime, the call refuses ([`ModularError::ModulusUnsupported`],
//!    `E-HOLO-006`, and [`ModularError::WorkLimitExceeded`], `E-HOLO-008`).
//!
//! The one thing that never happens is a returned residue that is wrong.
//!
//! # Why `binomial_mod` lives here
//!
//! `binomial(a, b) mod p^k` is the same workload seen from the other side: it
//! is how a sweep spot-checks a closed form against the recurrence, and how the
//! `k = 1` case of a supercongruence is read off directly. The algorithm is the
//! Andrew Granville / Davis–Webb factorisation of `n!` into its `p`-free part,
//! which for `k = 1` degenerates to Lucas' theorem exactly. See
//! [`binomial_mod`].

use std::fmt;

use rug::ops::RemRounding;
use rug::Integer;

use super::HolonomicError;

/// Errors specific to modular evaluation.
///
/// A separate enum rather than three more variants on [`HolonomicError`],
/// which is public and *exhaustive*: adding to it is a major-version break
/// (`cargo semver-checks` fails on `enum_variant_added`, and a downstream
/// `match` without a wildcard stops compiling). [`super::qzeil::QHolonomicError`]
/// is the same pattern. Both surface to Python as `HolonomicError` carrying
/// their own `E-HOLO-*` code, so a Python caller sees no difference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModularError {
    /// The modulus is not a prime power this subsystem can work over: the base
    /// is composite, the exponent is zero, or `p^k` is past the machine-word
    /// backend's ceiling.
    ModulusUnsupported(String),
    /// A step of the recurrence does not determine the next term as a `p`-adic
    /// integer: the leading coefficient vanishes identically at that index, or
    /// the numerator's `p`-adic valuation is below the leading coefficient's.
    /// See the module docs for how singular indices are handled when they
    /// *are* determined.
    PAdicallyUndetermined(String),
    /// The computation is well posed but past a resource budget — a working
    /// precision the machine-word modulus cannot hold, or a `binomial_mod`
    /// whose cost is dominated by a pass over `1 … p−1`.
    WorkLimitExceeded(String),
    /// Malformed call, forwarded from [`ModularError::InvalidInput`] so the
    /// modular entry points have a single error type.
    InvalidInput(String),
}

impl fmt::Display for ModularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModularError::ModulusUnsupported(s) => write!(f, "holonomic: unsupported modulus: {s}"),
            ModularError::PAdicallyUndetermined(s) => {
                write!(f, "holonomic: not determined p-adically: {s}")
            }
            ModularError::WorkLimitExceeded(s) => {
                write!(f, "holonomic: work limit exceeded: {s}")
            }
            ModularError::InvalidInput(s) => write!(f, "holonomic: invalid input: {s}"),
        }
    }
}

impl std::error::Error for ModularError {}

impl From<HolonomicError> for ModularError {
    fn from(e: HolonomicError) -> Self {
        match e {
            HolonomicError::InvalidInput(s) => ModularError::InvalidInput(s),
            other => ModularError::InvalidInput(other.to_string()),
        }
    }
}

impl crate::errors::AlkahestError for ModularError {
    fn code(&self) -> &'static str {
        match self {
            ModularError::ModulusUnsupported(_) => "E-HOLO-006",
            ModularError::PAdicallyUndetermined(_) => "E-HOLO-007",
            ModularError::WorkLimitExceeded(_) => "E-HOLO-008",
            ModularError::InvalidInput(_) => "E-HOLO-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            ModularError::ModulusUnsupported(_) => {
                "the modulus must be p**k with p prime, k >= 1 and p**k < 2**62; for a \
                 composite modulus, evaluate at each prime power and recombine by CRT"
            }
            ModularError::PAdicallyUndetermined(_) => {
                "no modulus repairs this: the recurrence itself leaves Z_p at that index. \
                 Supply more initial terms so the evaluation starts past it, use a \
                 recurrence whose leading coefficient does not vanish there, or accept \
                 that the sequence is not p-integral and rescale it"
            }
            ModularError::WorkLimitExceeded(_) => {
                "lower k, use a smaller prime, or ask for an index the recurrence reaches \
                 without crossing so many singular steps"
            }
            ModularError::InvalidInput(_) => {
                "n and k must be distinct symbols; max_order and max_degree must be positive"
            }
        })
    }
}

/// Largest modulus the machine-word backend accepts.
///
/// Products are formed in `u128`, so any `m < 2⁶⁴` would be safe; the limit is
/// set at `2⁶²` so that the *working* modulus `p^(k+L)` has room to grow past
/// the requested `p^k` without the ceiling being hit by rounding alone.
const MAX_MODULUS: u64 = 1 << 62;

/// Extra precision the leading-coefficient scan runs at, above what was asked.
///
/// The scan only needs `v_p(a_J(n))`, which a residue mod `p^K` decides unless
/// the residue is zero — and then the exact integer value has to be formed,
/// which is a bignum evaluation. Scanning with headroom makes that fallback
/// rare rather than routine (for `k = 1` it would otherwise fire at every
/// singular index).
const SCAN_HEADROOM: u32 = 16;

/// Steps whose leading coefficients are inverted in one batch.
///
/// Montgomery's trick turns `c` inversions into one inversion and `3c`
/// multiplications. The chunk is bounded so that memory stays `O(1)` in the
/// target index — a sweep may run to `N = 10⁷`.
const INVERSION_CHUNK: usize = 1024;

/// How many singular indices are reported back before the list is truncated.
const MAX_REPORTED_SINGULAR: usize = 64;

/// Work units [`binomial_mod`] will spend before refusing.
///
/// The cost is `O(p·k³ + log_p(a)·p·k)`, dominated by the one pass over
/// `1 … p−1` that builds the block polynomial. The budget exists for the `p^k`
/// a caller can write down but nobody can afford, not for a realistic call.
const BINOMIAL_WORK_BUDGET: u128 = 1 << 31;

// ---------------------------------------------------------------------------
// Machine-word modular arithmetic
// ---------------------------------------------------------------------------

#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// `a + b mod m` for `a, b < m <= MAX_MODULUS`.
///
/// No `u128` here: `MAX_MODULUS` is `2⁶²`, so `a + b < 2⁶³` cannot overflow and
/// one conditional subtraction is exact. This is the hottest line in the
/// forward pass — a `u128` remainder costs about as much as the multiply it
/// follows, and there are three of them per coefficient.
#[inline]
fn add_mod(a: u64, b: u64, m: u64) -> u64 {
    let s = a + b;
    if s >= m {
        s - m
    } else {
        s
    }
}

#[inline]
fn sub_mod(a: u64, b: u64, m: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        m - (b - a)
    }
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// Modular inverse by the extended Euclidean algorithm, or `None` for a
/// non-unit.
///
/// `m` is a prime power, not a prime, so Fermat's little theorem does not
/// apply and a `pow_mod(a, m-2, m)` shortcut would return a plausible-looking
/// wrong answer. Returning `None` is the point: every caller here has a real
/// decision to make when the leading coefficient is not invertible.
fn inv_mod(a: u64, m: u64) -> Option<u64> {
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    if old_r != 1 {
        return None;
    }
    Some(old_s.rem_euclid(m as i128) as u64)
}

/// `p^e` as a `u64`, or `None` on overflow past [`MAX_MODULUS`].
fn prime_power(p: u64, e: u32) -> Option<u64> {
    let mut acc: u128 = 1;
    for _ in 0..e {
        acc = acc.checked_mul(p as u128)?;
        if acc > MAX_MODULUS as u128 {
            return None;
        }
    }
    Some(acc as u64)
}

/// `v_p(x)` for a residue `x` mod `p^cap`, saturating at `cap`.
///
/// A return of `cap` means "at least `cap`" and nothing more; every caller
/// treats that as undecided rather than as a valuation.
fn valuation(mut x: u64, p: u64, cap: u32) -> u32 {
    if x == 0 {
        return cap;
    }
    let mut v = 0;
    while v < cap && x % p == 0 {
        x /= p;
        v += 1;
    }
    v
}

/// `n` reduced into `[0, m)`, for a signed sequence index.
#[inline]
fn index_mod(n: i64, m: u64) -> u64 {
    // `m <= MAX_MODULUS < 2^63`, so the cast is lossless and the remainder is
    // representable.
    n.rem_euclid(m as i64) as u64
}

fn reduce_integer(z: &Integer, m: u64) -> u64 {
    let modulus = Integer::from(m);
    let r = z.clone().rem_euc(modulus);
    r.to_u64().expect("residue of a positive modulus fits u64")
}

// ---------------------------------------------------------------------------
// The recurrence
// ---------------------------------------------------------------------------

/// A P-recursive recurrence prepared for evaluation modulo prime powers.
///
/// Holds `Σ_{i=0}^{J} a_i(n)·S(n+i) = b(n)` with integer polynomial
/// coefficients, together with the `J` initial values `S(start), …,
/// S(start+J−1)` as exact rationals. Nothing about `p` is fixed at
/// construction: the same object is evaluated at every prime of a sweep.
///
/// The recurrence is a *hypothesis about the caller's sequence*. This type
/// verifies that it is well formed and that each forward step is determined
/// `p`-adically; it cannot verify that the sequence satisfies it. Certify with
/// [`super::zeilberger()`], or fit and confirm with `alkahest.guess_holonomic`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularRecurrence {
    /// `coeffs[i]` is `a_i`, lowest-degree coefficient first. Length `J+1`.
    coeffs: Vec<Vec<Integer>>,
    /// `b`, lowest-degree first; empty for the homogeneous recurrence.
    rhs: Vec<Integer>,
    /// `(numerator, denominator)` of `S(start+j)`, denominator positive.
    initial: Vec<(Integer, Integer)>,
    start: i64,
}

/// One evaluation of a [`ModularRecurrence`], with the evidence that makes the
/// residues trustworthy.
///
/// [`ModularEvaluation::singular_indices`] is the field to read when a result
/// is surprising: it lists the steps where `a_J(n)` was not a unit and the
/// working precision had to absorb the loss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularEvaluation {
    residues: Vec<u64>,
    prime: u64,
    precision: u32,
    working_precision: u32,
    modulus: u64,
    singular_indices: Vec<i64>,
    n_singular: u64,
    steps: u64,
}

impl ModularEvaluation {
    /// The residues, one per requested index, each in `[0, p^k)`.
    pub fn residues(&self) -> &[u64] {
        &self.residues
    }

    /// The prime the evaluation ran at.
    pub fn prime(&self) -> u64 {
        self.prime
    }

    /// `k`, the precision that was asked for and delivered.
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// `K ≥ k`, the precision the forward pass actually ran at.
    ///
    /// `K − k` is the total `p`-adic precision lost to singular steps. It is
    /// `0` for a recurrence whose leading coefficient is a unit throughout.
    pub fn working_precision(&self) -> u32 {
        self.working_precision
    }

    /// `p^k`.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Indices `n` where `p | a_J(n)`, truncated to the first 64.
    pub fn singular_indices(&self) -> &[i64] {
        &self.singular_indices
    }

    /// How many singular steps there were in total (the untruncated count).
    pub fn n_singular(&self) -> u64 {
        self.n_singular
    }

    /// How many forward steps the evaluation took.
    pub fn steps(&self) -> u64 {
        self.steps
    }
}

/// Per-step data the chunk planner produces and the forward pass consumes.
struct StepPlan {
    /// `v_p(a_J(n))` for each `n` in the chunk.
    valuations: Vec<u32>,
    /// `p^v` for the same `v`, so the forward pass never re-derives it.
    prime_powers: Vec<u64>,
    /// `(a_J(n)/p^v)⁻¹ mod p^K` for each `n` in the chunk.
    unit_inverses: Vec<u64>,
}

impl ModularRecurrence {
    /// Build a recurrence, or refuse a malformed one.
    ///
    /// * `coeffs[i]` is the polynomial `a_i(n)`, lowest-degree coefficient
    ///   first; `coeffs` has length `J+1` and `a_J` must not be the zero
    ///   polynomial.
    /// * `rhs` is `b(n)` in the same form; pass an empty slice for the
    ///   homogeneous recurrence. A *rational* `b(n)` is out of scope — clear
    ///   its denominator through the whole relation first.
    /// * `initial` is `S(start), …, S(start+J−1)` as `(numerator,
    ///   denominator)` pairs with non-zero denominators.
    pub fn new(
        coeffs: Vec<Vec<Integer>>,
        rhs: Vec<Integer>,
        initial: Vec<(Integer, Integer)>,
        start: i64,
    ) -> Result<Self, ModularError> {
        if coeffs.len() < 2 {
            return Err(ModularError::InvalidInput(format!(
                "a recurrence needs at least one shift: got {} coefficient \
                 polynomials, so the order would be {}",
                coeffs.len(),
                coeffs.len().saturating_sub(1)
            )));
        }
        let order = coeffs.len() - 1;
        if coeffs.iter().any(|c| c.is_empty()) {
            return Err(ModularError::InvalidInput(
                "every coefficient polynomial needs at least one coefficient; \
                 use [0] for a coefficient that is identically zero"
                    .into(),
            ));
        }
        if coeffs[order].iter().all(|c| *c == 0) {
            return Err(ModularError::InvalidInput(format!(
                "the leading coefficient a_{order}(n) is the zero polynomial, so \
                 the relation never determines S(n+{order}); drop the trailing \
                 coefficient and use the order-{} recurrence it really is",
                order - 1
            )));
        }
        if initial.len() != order {
            return Err(ModularError::InvalidInput(format!(
                "an order-{order} recurrence needs exactly {order} initial values, \
                 got {}",
                initial.len()
            )));
        }
        let mut normalised = Vec::with_capacity(order);
        for (j, (num, den)) in initial.into_iter().enumerate() {
            if den == 0 {
                return Err(ModularError::InvalidInput(format!(
                    "initial value {j} has a zero denominator"
                )));
            }
            let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
            normalised.push((num, den));
        }
        Ok(Self {
            coeffs,
            rhs,
            initial: normalised,
            start,
        })
    }

    /// Recurrence order `J`.
    pub fn order(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Largest degree of any coefficient polynomial (the right-hand side
    /// included).
    pub fn degree(&self) -> usize {
        self.coeffs
            .iter()
            .chain(std::iter::once(&self.rhs))
            .map(|c| c.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
    }

    /// Index that the first initial value belongs to.
    pub fn start(&self) -> i64 {
        self.start
    }

    /// Whether `b(n)` is identically zero.
    pub fn is_homogeneous(&self) -> bool {
        self.rhs.iter().all(|c| *c == 0)
    }

    /// `a_i(n)` as exact integers, `i = 0..=J`, lowest degree first.
    pub fn coefficients(&self) -> &[Vec<Integer>] {
        &self.coeffs
    }

    /// `b(n)` as exact integers, lowest degree first; empty when homogeneous.
    pub fn inhomogeneity(&self) -> &[Integer] {
        &self.rhs
    }

    /// `S(start), …, S(start+J−1)` as `(numerator, denominator)`.
    pub fn initial_values(&self) -> &[(Integer, Integer)] {
        &self.initial
    }

    /// `S(target) mod p^k`.
    ///
    /// Shorthand for [`ModularRecurrence::evaluate`] at a single index.
    ///
    /// ```
    /// use alkahest_cas::holonomic::modular::ModularRecurrence;
    /// use rug::Integer;
    ///
    /// // Apéry: (n+2)³·A(n+2) = (34n³+153n²+231n+117)·A(n+1) − (n+1)³·A(n),
    /// // i.e. a_0 = (n+1)³, a_1 = −(34n³+153n²+231n+117), a_2 = (n+2)³, each
    /// // written lowest-degree coefficient first.
    /// let z = |v: i64| Integer::from(v);
    /// let rec = ModularRecurrence::new(
    ///     vec![
    ///         vec![z(1), z(3), z(3), z(1)],
    ///         vec![z(-117), z(-231), z(-153), z(-34)],
    ///         vec![z(8), z(12), z(6), z(1)],
    ///     ],
    ///     vec![],
    ///     vec![(z(1), z(1)), (z(5), z(1))],
    ///     0,
    /// )
    /// .unwrap();
    ///
    /// // A(p−1) ≡ 1 (mod p³) for p = 13 — the Apéry supercongruence.
    /// assert_eq!(rec.value_mod(12, 13, 3).unwrap(), 1);
    /// ```
    pub fn value_mod(&self, target: i64, p: u64, k: u32) -> Result<u64, ModularError> {
        Ok(self.evaluate(&[target], p, k)?.residues[0])
    }

    /// `S(n) mod p^k` at every index in `targets`, in one forward pass.
    ///
    /// `targets` must be strictly increasing and at least `start`; the caller
    /// is expected to sort and de-duplicate, because a silently reordered
    /// result is a bug waiting to be read off in the wrong order.
    pub fn evaluate(
        &self,
        targets: &[i64],
        p: u64,
        k: u32,
    ) -> Result<ModularEvaluation, ModularError> {
        self.check_modulus(p, k)?;
        if targets.is_empty() {
            return Err(ModularError::InvalidInput(
                "no target indices were given".into(),
            ));
        }
        for w in targets.windows(2) {
            if w[1] <= w[0] {
                return Err(ModularError::InvalidInput(format!(
                    "target indices must be strictly increasing, got {} after {}",
                    w[1], w[0]
                )));
            }
        }
        if targets[0] < self.start {
            return Err(ModularError::InvalidInput(format!(
                "target index {} is below start = {}; the recurrence is only run \
                 forwards",
                targets[0], self.start
            )));
        }

        let order = self.order() as i64;
        let last = *targets.last().expect("non-empty");
        // Steps produce S(start+J), …, S(last); the step producing S(n+J) is
        // indexed by n.
        let n_steps = (last - self.start - order + 1).max(0) as u64;

        let (loss, singular_indices, n_singular) = self.scan_losses(n_steps, p, k)?;
        let working = k
            .checked_add(loss)
            .ok_or_else(|| precision_overflow(p, k, loss))?;
        let modulus_k = prime_power(p, k).ok_or_else(|| unsupported_modulus(p, k))?;
        let working_modulus = prime_power(p, working).ok_or_else(|| {
            ModularError::WorkLimitExceeded(format!(
                "the {n_singular} singular step(s) cost {loss} digits of p-adic \
                 precision, so answering to p^{k} needs a working modulus of \
                 {p}^{working}, which is past the machine-word backend's ceiling \
                 of 2^62"
            ))
        })?;

        let residues = self.forward(targets, p, k, working, working_modulus, modulus_k)?;

        Ok(ModularEvaluation {
            residues,
            prime: p,
            precision: k,
            working_precision: working,
            modulus: modulus_k,
            singular_indices,
            n_singular,
            steps: n_steps,
        })
    }

    fn check_modulus(&self, p: u64, k: u32) -> Result<(), ModularError> {
        if k == 0 {
            return Err(ModularError::ModulusUnsupported(
                "precision k must be at least 1; p^0 = 1 has one residue and \
                 says nothing"
                    .into(),
            ));
        }
        if p < 2 || !crate::modular::is_prime(p) {
            return Err(ModularError::ModulusUnsupported(format!(
                "{p} is not prime; the lifting argument this module rests on \
                 needs a prime power modulus, and v_p is not defined otherwise"
            )));
        }
        if prime_power(p, k).is_none() {
            return Err(unsupported_modulus(p, k));
        }
        Ok(())
    }

    /// Pass one: `v_p(a_J(n))` at every step, without touching the sequence.
    ///
    /// The leading coefficient does not depend on `S`, so the entire precision
    /// budget can be settled before the first sequence value exists. That is
    /// what makes the forward pass a single deterministic run instead of a
    /// retry loop that has to guess how much headroom to add.
    fn scan_losses(
        &self,
        n_steps: u64,
        p: u64,
        k: u32,
    ) -> Result<(u32, Vec<i64>, u64), ModularError> {
        if n_steps == 0 {
            return Ok((0, Vec::new(), 0));
        }
        let scan_precision = scan_precision(p, k);
        let scan_modulus =
            prime_power(p, scan_precision).ok_or_else(|| unsupported_modulus(p, k))?;
        let lead = reduce_poly(&self.coeffs[self.order()], scan_modulus);

        let mut loss: u64 = 0;
        let mut reported = Vec::new();
        let mut n_singular = 0u64;
        for step in 0..n_steps {
            let n = self.start + step as i64;
            let d = eval_poly(&lead, index_mod(n, scan_modulus), scan_modulus);
            let mut v = valuation(d, p, scan_precision);
            if v == scan_precision {
                // The residue cannot decide the valuation. This is rare by
                // construction (`scan_precision` carries headroom), so paying
                // for one exact bignum evaluation here is cheaper than raising
                // the precision of the whole scan.
                v = self.exact_leading_valuation(n, p)?;
            }
            if v > 0 {
                n_singular += 1;
                if reported.len() < MAX_REPORTED_SINGULAR {
                    reported.push(n);
                }
                loss += v as u64;
            }
        }
        let loss = u32::try_from(loss).map_err(|_| {
            ModularError::WorkLimitExceeded(format!(
                "the singular steps of this recurrence cost {loss} digits of \
                 p-adic precision at p = {p}, far past any workable modulus"
            ))
        })?;
        Ok((loss, reported, n_singular))
    }

    /// `v_p(a_J(n))` computed over `ℤ`, for the indices a residue cannot decide.
    fn exact_leading_valuation(&self, n: i64, p: u64) -> Result<u32, ModularError> {
        let value = horner_exact(&self.coeffs[self.order()], n);
        if value == 0 {
            return Err(ModularError::PAdicallyUndetermined(format!(
                "the leading coefficient a_{}(n) vanishes at n = {n}, so the \
                 recurrence does not determine S({}) from the terms before it — \
                 no modulus can repair that",
                self.order(),
                n + self.order() as i64
            )));
        }
        let mut v = 0u32;
        let mut z = value.abs();
        let prime = Integer::from(p);
        while z.is_divisible(&prime) {
            z /= &prime;
            v += 1;
        }
        Ok(v)
    }

    /// Pass two: run the recurrence forward at working precision `K`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        targets: &[i64],
        p: u64,
        k: u32,
        working: u32,
        modulus: u64,
        modulus_k: u64,
    ) -> Result<Vec<u64>, ModularError> {
        let order = self.order();
        let reduced: Vec<Vec<u64>> = self
            .coeffs
            .iter()
            .map(|c| reduce_poly(c, modulus))
            .collect();
        let rhs = reduce_poly(&self.rhs, modulus);
        let lead = &reduced[order];

        // A ring buffer of the last `order` values, `window[j]` holding
        // S(base + j). Residues are carried at the full working modulus; only
        // the *claimed* precision shrinks, and reduction mod p^c commutes with
        // every operation below, so the low `c` digits stay exact.
        let mut window: Vec<u64> = Vec::with_capacity(order);
        for (num, den) in &self.initial {
            let d = reduce_integer(den, modulus);
            let inv = inv_mod(d, modulus).ok_or_else(|| {
                ModularError::PAdicallyUndetermined(format!(
                    "initial value {num}/{den} has a denominator divisible by \
                     {p}, so it is not a p-adic integer and has no residue mod \
                     {p}^{working}"
                ))
            })?;
            window.push(mul_mod(reduce_integer(num, modulus), inv, modulus));
        }

        let mut out = vec![0u64; targets.len()];
        let mut next_target = 0usize;
        let mut precision = working;

        // Targets that land on an initial value need no stepping at all.
        while next_target < targets.len() && targets[next_target] < self.start + order as i64 {
            let j = (targets[next_target] - self.start) as usize;
            out[next_target] = window[j] % modulus_k;
            next_target += 1;
        }
        if next_target == targets.len() {
            return Ok(out);
        }

        let last = *targets.last().expect("non-empty");
        let n_steps = (last - self.start - order as i64 + 1).max(0) as u64;

        let mut step = 0u64;
        while step < n_steps {
            let chunk = INVERSION_CHUNK.min((n_steps - step) as usize);
            let plan =
                self.plan_chunk(lead, self.start + step as i64, chunk, p, working, modulus)?;
            for slot in 0..chunk {
                let n = self.start + (step + slot as u64) as i64;
                let x = index_mod(n, modulus);

                // numerator = b(n) − Σ_{i<J} a_i(n)·S(n+i)
                let mut numerator = eval_poly(&rhs, x, modulus);
                for (i, window_value) in window.iter().enumerate() {
                    let a = eval_poly(&reduced[i], x, modulus);
                    numerator = sub_mod(numerator, mul_mod(a, *window_value, modulus), modulus);
                }

                let v = plan.valuations[slot];
                let next = if v == 0 {
                    mul_mod(numerator, plan.unit_inverses[slot], modulus)
                } else {
                    let pv = plan.prime_powers[slot];
                    if numerator % pv != 0 {
                        return Err(ModularError::PAdicallyUndetermined(format!(
                            "at n = {n} the leading coefficient a_{order}(n) has \
                             v_{p} = {v} but the numerator has v_{p} = {}, so \
                             S({}) is not a p-adic integer and has no residue mod \
                             a power of {p}",
                            valuation(numerator, p, v),
                            n + order as i64,
                        )));
                    }
                    // The numerator is trustworthy to `precision` digits and
                    // divisible by `p^v` there, so the quotient is trustworthy
                    // to `precision − v`. The scan sized `working` so that this
                    // never falls below the `k` that was asked for.
                    precision -= v;
                    mul_mod(numerator / pv, plan.unit_inverses[slot], modulus)
                };

                window.rotate_left(1);
                window[order - 1] = next;

                let produced = n + order as i64;
                if next_target < targets.len() && targets[next_target] == produced {
                    out[next_target] = next % modulus_k;
                    next_target += 1;
                }
            }
            step += chunk as u64;
        }

        // The scan computed the total loss up front, so this cannot fail — but
        // it is the invariant the whole design rests on, so it is checked
        // rather than assumed, and a violation refuses instead of returning a
        // residue that is short of the precision it claims.
        if precision < k {
            return Err(ModularError::PAdicallyUndetermined(format!(
                "the forward pass ended with {precision} digits of p-adic \
                 precision at p = {p}, below the {k} the leading-coefficient \
                 scan budgeted for; please report this recurrence"
            )));
        }
        Ok(out)
    }

    /// Valuations and unit inverses for one chunk of steps.
    ///
    /// The inverses are taken in a single batch (Montgomery's trick): the
    /// leading coefficient is the only thing that has to be inverted and it
    /// does not depend on the sequence, so `c` extended-Euclid runs collapse
    /// into one plus `3c` multiplications.
    fn plan_chunk(
        &self,
        lead: &[u64],
        first_index: i64,
        chunk: usize,
        p: u64,
        working: u32,
        modulus: u64,
    ) -> Result<StepPlan, ModularError> {
        let mut valuations = Vec::with_capacity(chunk);
        let mut prime_powers = Vec::with_capacity(chunk);
        let mut units = Vec::with_capacity(chunk);
        for slot in 0..chunk {
            let n = first_index + slot as i64;
            let d = eval_poly(lead, index_mod(n, modulus), modulus);
            let v = valuation(d, p, working);
            if v == working {
                // Unreachable: the scan set `working = k + Σ v`, so every
                // individual `v` is strictly below it. Refuse rather than
                // divide by a `p^v` that the modulus cannot represent, which is
                // exactly the silent-garbage path this module exists to close.
                return Err(ModularError::PAdicallyUndetermined(format!(
                    "the leading coefficient at n = {n} is 0 mod {p}^{working} \
                     even though the scan budgeted a finite valuation for it; \
                     please report this recurrence"
                )));
            }
            let pv = prime_power(p, v).expect("v < working and p^working fits");
            valuations.push(v);
            prime_powers.push(pv);
            units.push(d / pv);
        }

        // prefix[i] = units[0] · … · units[i-1]
        let mut prefix = Vec::with_capacity(chunk + 1);
        prefix.push(1 % modulus);
        for u in &units {
            let acc = mul_mod(*prefix.last().expect("seeded"), *u, modulus);
            prefix.push(acc);
        }
        let mut running = inv_mod(prefix[chunk], modulus).ok_or_else(|| {
            ModularError::PAdicallyUndetermined(format!(
                "the unit part of a leading coefficient near n = {first_index} is \
                 not invertible mod {p}^{working}; this is an internal invariant \
                 violation, please report it"
            ))
        })?;
        let mut unit_inverses = vec![0u64; chunk];
        for slot in (0..chunk).rev() {
            unit_inverses[slot] = mul_mod(running, prefix[slot], modulus);
            running = mul_mod(running, units[slot], modulus);
        }

        Ok(StepPlan {
            valuations,
            prime_powers,
            unit_inverses,
        })
    }
}

fn scan_precision(p: u64, k: u32) -> u32 {
    let mut e = k;
    while e < k + SCAN_HEADROOM {
        match prime_power(p, e + 1) {
            Some(_) => e += 1,
            None => break,
        }
    }
    e
}

fn unsupported_modulus(p: u64, k: u32) -> ModularError {
    ModularError::ModulusUnsupported(format!(
        "{p}^{k} does not fit the machine-word backend, whose ceiling is 2^62; \
         reduce k, or use a smaller prime"
    ))
}

fn precision_overflow(p: u64, k: u32, loss: u32) -> ModularError {
    ModularError::WorkLimitExceeded(format!(
        "the singular steps cost {loss} digits of p-adic precision, so answering \
         to {p}^{k} would need a working precision of {k} + {loss}, which \
         overflows"
    ))
}

fn reduce_poly(coeffs: &[Integer], m: u64) -> Vec<u64> {
    coeffs.iter().map(|c| reduce_integer(c, m)).collect()
}

/// Horner evaluation of a polynomial given lowest-degree-first, mod `m`.
#[inline]
fn eval_poly(coeffs: &[u64], x: u64, m: u64) -> u64 {
    let mut acc = 0u64;
    for c in coeffs.iter().rev() {
        acc = add_mod(mul_mod(acc, x, m), *c, m);
    }
    acc
}

fn horner_exact(coeffs: &[Integer], x: i64) -> Integer {
    let x = Integer::from(x);
    let mut acc = Integer::new();
    for c in coeffs.iter().rev() {
        acc *= &x;
        acc += c;
    }
    acc
}

// ---------------------------------------------------------------------------
// binomial(a, b) mod p^k
// ---------------------------------------------------------------------------

/// `binomial(a, b) mod p^k`, exactly, for `p` prime.
///
/// # Method
///
/// Every factorial splits as `n! = p^⌊n/p⌋ · (n!)_p · ⌊n/p⌋!`, where `(n!)_p`
/// is the product of the integers up to `n` that `p` does not divide. Unrolling
/// that gives, exactly over `ℚ`,
///
/// ```text
/// binomial(a, b) = p^e · Π_{j≥0} (n_j!)_p / ( (m_j!)_p · (r_j!)_p ),
/// n_j = ⌊a/p^j⌋,  m_j = ⌊b/p^j⌋,  r_j = ⌊(a−b)/p^j⌋,
/// ```
///
/// with `e = v_p(binomial(a,b))` by Legendre's formula. Every `(·!)_p` is a
/// unit mod `p^k`, so the quotient is taken there directly. This is the
/// Andrew Granville / Davis–Webb prime-power generalisation of Lucas; at
/// `k = 1` it *is* Lucas, since `(n!)_p ≡ (−1)^⌊n/p⌋·(n mod p)! (mod p)` turns
/// the product into `Π binomial(a_j, b_j)` over base-`p` digits.
///
/// `(r!)_p` for `r < p^k` is computed by a product tree over blocks of `p`
/// consecutive integers rather than term by term, which is what keeps the cost
/// `O(p·k³)` instead of `O(p^k)`.
///
/// # Refusals
///
/// * `E-HOLO-006` — `p` is not prime, `k = 0`, or `p^k` is past `2^62`.
/// * `E-HOLO-008` — the work budget would be exceeded.
///
/// `b > a` and `b < 0` are not errors: the binomial coefficient is `0` and the
/// residue is `0 mod p^k`.
///
/// ```
/// use alkahest_cas::holonomic::modular::binomial_mod;
///
/// // Wolstenholme: binomial(2p−1, p−1) ≡ 1 (mod p³) for p ≥ 5.
/// assert_eq!(binomial_mod(2 * 11 - 1, 10, 11, 3).unwrap(), 1);
/// // A binomial far larger than the prime; one whose p-adic valuation is at
/// // least k, so the residue is 0; and one that is 0 because b > a.
/// assert_eq!(binomial_mod(1_000_000, 3, 7, 4).unwrap(), 2261);
/// assert_eq!(binomial_mod(1_000_000, 500_000, 7, 4).unwrap(), 0);
/// assert_eq!(binomial_mod(5, 9, 7, 4).unwrap(), 0);
/// ```
pub fn binomial_mod(a: u64, b: i128, p: u64, k: u32) -> Result<u64, ModularError> {
    if k == 0 {
        return Err(ModularError::ModulusUnsupported(
            "precision k must be at least 1; p^0 = 1 has one residue and says \
             nothing"
                .into(),
        ));
    }
    if p < 2 || !crate::modular::is_prime(p) {
        return Err(ModularError::ModulusUnsupported(format!(
            "{p} is not prime; binomial_mod needs a prime power modulus"
        )));
    }
    let m = prime_power(p, k).ok_or_else(|| unsupported_modulus(p, k))?;
    if b < 0 || b > a as i128 {
        return Ok(0);
    }
    let b = b as u64;
    let c = a - b;

    // Legendre: e = Σ_{j≥1} (⌊a/p^j⌋ − ⌊b/p^j⌋ − ⌊c/p^j⌋).
    let mut e: u32 = 0;
    let mut pj: u128 = p as u128;
    while pj <= a as u128 {
        let d = (a as u128 / pj) - (b as u128 / pj) - (c as u128 / pj);
        e = e.saturating_add(d as u32);
        if e >= k {
            return Ok(0);
        }
        pj *= p as u128;
    }

    let levels = {
        let mut l = 1u64;
        let mut q = a;
        while q >= p {
            q /= p;
            l += 1;
        }
        l
    };
    let kk = k as u128;
    let work = (p as u128) * (kk * kk * kk + kk * levels as u128);
    if work > BINOMIAL_WORK_BUDGET {
        return Err(ModularError::WorkLimitExceeded(format!(
            "binomial({a}, {b}) mod {p}^{k} needs about {work} unit operations, \
             past the budget of {BINOMIAL_WORK_BUDGET}; the cost is O(p·k³), so \
             lower k or p"
        )));
    }

    let ctx = UnitFactorial::new(p, k, m);
    let mut numerator = 1 % m;
    let mut denominator = 1 % m;
    let (mut aj, mut bj, mut cj) = (a, b, c);
    loop {
        numerator = mul_mod(numerator, ctx.unit_factorial(aj), m);
        denominator = mul_mod(denominator, ctx.unit_factorial(bj), m);
        denominator = mul_mod(denominator, ctx.unit_factorial(cj), m);
        if aj == 0 {
            break;
        }
        aj /= p;
        bj /= p;
        cj /= p;
    }
    let inverse = inv_mod(denominator, m).ok_or_else(|| {
        ModularError::PAdicallyUndetermined(format!(
            "the p-free part of a factorial came out non-invertible mod {p}^{k}; \
             this is an internal invariant violation, please report \
             binomial({a}, {b}) mod {p}^{k}"
        ))
    })?;
    Ok(mul_mod(
        mul_mod(numerator, inverse, m),
        pow_mod(p, e as u64, m),
        m,
    ))
}

/// `(n!)_p mod p^k` — the `p`-free part of a factorial.
struct UnitFactorial {
    p: u64,
    k: u32,
    m: u64,
    /// `Π_{t=1}^{p−1} (x+t)` truncated to degree `k−1`, then evaluated along
    /// `x = j·p`: coefficient `block[i]` is that of `j^i`, divisible by `p^i`.
    block: Vec<u64>,
    /// `Π_{0<i<p^k, p∤i} i mod p^k`, which is `±1`.
    wilson: u64,
}

impl UnitFactorial {
    fn new(p: u64, k: u32, m: u64) -> Self {
        let width = k as usize;
        // Π_{t=1}^{p−1} (x + t), truncated to degree k−1.
        let mut poly = vec![0u64; width];
        poly[0] = 1 % m;
        for t in 1..p {
            let t = t % m;
            // poly ← poly · (x + t), truncated.
            for i in (0..width).rev() {
                let shifted = if i == 0 { 0 } else { poly[i - 1] };
                poly[i] = add_mod(mul_mod(poly[i], t, m), shifted, m);
            }
        }
        // Substitute x = j·p: coefficient of j^i picks up p^i.
        let mut block = vec![0u64; width];
        let mut power = 1 % m;
        for i in 0..width {
            block[i] = mul_mod(poly[i], power, m);
            power = mul_mod(power, p % m, m);
        }
        // The units mod p^k form a cyclic group for odd p (and for p^k ∈ {2,4}),
        // so their product is the unique element of order 2, namely −1. For
        // p = 2, k ≥ 3 the group is (Z/2) × (Z/2^{k−2}) with three involutions
        // whose product is 1.
        let wilson = if p == 2 && k >= 3 { 1 % m } else { m - (1 % m) };
        Self {
            p,
            k,
            m,
            block,
            wilson,
        }
    }

    /// `Π_{1≤i≤n, p∤i} i mod p^k`.
    fn unit_factorial(&self, n: u64) -> u64 {
        let m = self.m;
        let full = n / m;
        let rest = n % m;
        let mut acc = if full % 2 == 1 { self.wilson } else { 1 % m };
        acc = mul_mod(acc, self.unit_prefix(rest), m);
        acc
    }

    /// `Π_{1≤i≤r, p∤i} i mod p^k`, for `r < p^k`.
    fn unit_prefix(&self, r: u64) -> u64 {
        let (p, m) = (self.p, self.m);
        let blocks = r / p;
        let tail = r % p;
        let mut acc = product_over_range(&self.block, blocks, p, m, self.k);
        // The leftover `blocks·p + 1 … blocks·p + tail` are all coprime to p.
        let base = mul_mod(blocks % m, p % m, m);
        for t in 1..=tail {
            acc = mul_mod(acc, add_mod(base, t % m, m), m);
        }
        acc
    }
}

/// `Π_{j=0}^{count−1} P(j) mod m`, for a polynomial `P` whose `x^i`
/// coefficient is divisible by `p^i`.
///
/// That divisibility is what makes the recursion work: substituting `x → x·p+t`
/// preserves it, so the product of the `p` shifted copies can be truncated back
/// to `k` coefficients without losing anything mod `p^k`. Each level of the
/// recursion therefore costs `O(p·k²)` and divides `count` by `p`, which is how
/// a product of up to `p^(k−1)` terms is taken in `O(p·k³)`.
fn product_over_range(poly: &[u64], count: u64, p: u64, m: u64, k: u32) -> u64 {
    if count == 0 {
        return 1 % m;
    }
    if count <= p || count <= 64 {
        let mut acc = 1 % m;
        for j in 0..count {
            acc = mul_mod(acc, eval_poly(poly, j % m, m), m);
        }
        return acc;
    }
    let outer = count / p;
    let leftover = count % p;
    // Q(x) = Π_{t=0}^{p−1} P(x·p + t), truncated to degree k−1.
    let width = k as usize;
    let mut q = vec![0u64; width];
    q[0] = 1 % m;
    let binom = pascal(width, m);
    for t in 0..p {
        let shifted = shift_poly(poly, p, t % m, m, &binom);
        q = mul_trunc(&q, &shifted, m);
    }
    let mut acc = product_over_range(&q, outer, p, m, k);
    // The `leftover` values j = outer·p … outer·p + leftover − 1 are left over.
    let base = mul_mod(outer % m, p % m, m);
    for t in 0..leftover {
        acc = mul_mod(acc, eval_poly(poly, add_mod(base, t % m, m), m), m);
    }
    acc
}

/// `P(x·p + t)` truncated to `poly.len()` coefficients.
fn shift_poly(poly: &[u64], p: u64, t: u64, m: u64, binom: &[Vec<u64>]) -> Vec<u64> {
    let width = poly.len();
    // t^0, t^1, …
    let mut t_pow = Vec::with_capacity(width);
    let mut acc = 1 % m;
    for _ in 0..width {
        t_pow.push(acc);
        acc = mul_mod(acc, t, m);
    }
    let mut p_pow = Vec::with_capacity(width);
    let mut acc = 1 % m;
    for _ in 0..width {
        p_pow.push(acc);
        acc = mul_mod(acc, p % m, m);
    }
    let mut out = vec![0u64; width];
    for (s, &a_s) in poly.iter().enumerate() {
        if a_s == 0 {
            continue;
        }
        for u in 0..=s {
            let term = mul_mod(
                mul_mod(a_s, binom[s][u], m),
                mul_mod(p_pow[u], t_pow[s - u], m),
                m,
            );
            out[u] = add_mod(out[u], term, m);
        }
    }
    out
}

fn mul_trunc(a: &[u64], b: &[u64], m: u64) -> Vec<u64> {
    let width = a.len();
    let mut out = vec![0u64; width];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if i + j >= width {
                break;
            }
            out[i + j] = add_mod(out[i + j], mul_mod(ai, bj, m), m);
        }
    }
    out
}

fn pascal(width: usize, m: u64) -> Vec<Vec<u64>> {
    let mut rows: Vec<Vec<u64>> = Vec::with_capacity(width);
    for i in 0..width {
        let mut row = vec![0u64; i + 1];
        row[0] = 1 % m;
        for j in 1..=i {
            let up = if j < rows[i - 1].len() {
                rows[i - 1][j]
            } else {
                0
            };
            row[j] = add_mod(rows[i - 1][j - 1], up, m);
        }
        rows.push(row);
    }
    rows
}

impl fmt::Display for ModularEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} residue(s) mod {}^{} ({} step(s), {} singular, working precision {})",
            self.residues.len(),
            self.prime,
            self.precision,
            self.steps,
            self.n_singular,
            self.working_precision
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use rug::ops::Pow;

    fn z(v: i64) -> Integer {
        Integer::from(v)
    }

    /// Apéry numbers A005259, index-shifted to `Σ_i a_i(n)·A(n+i) = 0`:
    /// `(n+2)³A(n+2) − (34n³+153n²+231n+117)A(n+1) + (n+1)³A(n) = 0`.
    fn apery() -> ModularRecurrence {
        ModularRecurrence::new(
            vec![
                vec![z(1), z(3), z(3), z(1)],
                vec![z(-117), z(-231), z(-153), z(-34)],
                vec![z(8), z(12), z(6), z(1)],
            ],
            vec![],
            vec![(z(1), z(1)), (z(5), z(1))],
            0,
        )
        .unwrap()
    }

    fn apery_exact(limit: usize) -> Vec<Integer> {
        let mut a = vec![Integer::from(1), Integer::from(5)];
        for n in 1..limit {
            let n_i = Integer::from(n);
            let poly = Integer::from(34) * n_i.clone().pow(3)
                + Integer::from(51) * n_i.clone().pow(2)
                + Integer::from(27) * n_i.clone()
                + 5;
            let next: Integer = (poly * &a[n]) - n_i.clone().pow(3) * &a[n - 1];
            let denominator: Integer = (n_i + 1u32).pow(3);
            let (q, r) = next.div_rem(denominator);
            assert_eq!(r, 0);
            a.push(q);
        }
        a
    }

    #[test]
    fn apery_matches_exact_arithmetic() {
        let rec = apery();
        let exact = apery_exact(200);
        for &p in &[5u64, 7, 11, 13, 101, 199] {
            // `n` beyond a few multiples of `p` crosses enough singular steps
            // to exhaust a 64-bit modulus at the small primes; that case is
            // `a_long_run_of_singular_steps_refuses` below.
            let indices: &[usize] = if p >= 101 {
                &[0, 1, 2, 3, 10, 37, 100, 199]
            } else {
                // n = 25 makes p = 5 cross n = 23, where (n+2)³ = 5⁶: six
                // digits lost in a single step, not the three the other
                // crossings cost.
                &[0, 1, 2, 3, 10, 25]
            };
            for &k in &[1u32, 2, 3, 4] {
                let m = Integer::from(p).pow(k);
                for &n in indices {
                    let want = Integer::from(&exact[n] % &m).to_u64().unwrap();
                    let got = rec.value_mod(n as i64, p, k).unwrap();
                    assert_eq!(got, want, "A({n}) mod {p}^{k}");
                }
            }
        }
    }

    /// Precision loss is real, cumulative, and refused rather than absorbed.
    ///
    /// Reaching `A(199)` at `p = 5` crosses 40 indices where `(n+2)³ ≡ 0`, each
    /// costing three `p`-adic digits. 120 digits of `5` is far past a 64-bit
    /// modulus, so there is no honest answer and the call says so — rather than
    /// dividing by a non-unit and returning the residue that comes out.
    #[test]
    fn a_long_run_of_singular_steps_refuses() {
        let rec = apery();
        let err = rec.value_mod(199, 5, 1).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-008");
        let text = format!("{err}");
        assert!(text.contains("digits of p-adic precision"), "{text}");
        // The same index at a prime it does not cross is answered normally.
        assert!(rec.value_mod(199, 199, 4).is_ok());
    }

    #[test]
    fn apery_supercongruence() {
        let rec = apery();
        // A(p−1) ≡ 1 (mod p³) — Beukers. It is *not* a mod-p⁴ congruence, and
        // asking for p⁴ is how the sharpness of a modulus gets measured.
        for &p in &[5u64, 7, 11, 13, 17, 19, 23, 29, 31, 101, 211] {
            assert_eq!(rec.value_mod(p as i64 - 1, p, 3).unwrap(), 1 % p.pow(3));
        }
        assert_ne!(rec.value_mod(12, 13, 4).unwrap(), 1);
    }

    #[test]
    fn many_targets_in_one_pass() {
        let rec = apery();
        let exact = apery_exact(60);
        let targets: Vec<i64> = (0..60).step_by(7).collect();
        let ev = rec.evaluate(&targets, 13, 3).unwrap();
        let m = Integer::from(13u32).pow(3);
        for (slot, &t) in targets.iter().enumerate() {
            let want = Integer::from(&exact[t as usize] % &m).to_u64().unwrap();
            assert_eq!(ev.residues()[slot], want);
        }
        // n ≡ −2 (mod 13) below 55: 11, 24, 37, 50 — three digits lost each.
        assert_eq!(ev.n_singular(), 4);
        assert_eq!(ev.singular_indices(), &[11, 24, 37, 50]);
        assert_eq!(ev.working_precision(), 3 + 12);
        assert_eq!(ev.steps(), 55);

        // A window entirely below the first singular index loses nothing.
        let clean = rec.evaluate(&[0, 1, 5, 10, 12], 13, 3).unwrap();
        assert_eq!(clean.n_singular(), 0);
        assert_eq!(clean.working_precision(), 3);
    }

    /// The singular case. `A(p)` steps through `n = p−2`, where the leading
    /// coefficient `(n+2)³` is divisible by `p³`.
    #[test]
    fn singular_index_is_lifted_not_ignored() {
        let rec = apery();
        let exact = apery_exact(40);
        for &p in &[5u64, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            for &k in &[1u32, 2, 3, 4, 5] {
                let target = p as i64;
                if target >= 40 {
                    continue;
                }
                let ev = rec.evaluate(&[target], p, k).unwrap();
                let m = Integer::from(p).pow(k);
                let want = Integer::from(&exact[target as usize] % &m)
                    .to_u64()
                    .unwrap();
                assert_eq!(ev.residues()[0], want, "A({p}) mod {p}^{k}");
                assert_eq!(ev.n_singular(), 1, "one singular step at n = p−2");
                assert_eq!(ev.singular_indices(), &[p as i64 - 2]);
                assert_eq!(ev.working_precision(), k + 3, "(n+2)³ costs three digits");
            }
        }
    }

    #[test]
    fn singular_index_over_a_long_run() {
        // Several singular steps at once: A(3p) crosses n = p−1, 2p−1, 3p−1.
        let rec = apery();
        let exact = apery_exact(40);
        let p = 11u64;
        let ev = rec.evaluate(&[33], p, 3).unwrap();
        let m = Integer::from(p).pow(3u32);
        assert_eq!(
            ev.residues()[0],
            Integer::from(&exact[33] % &m).to_u64().unwrap()
        );
        assert_eq!(ev.n_singular(), 3);
        assert_eq!(ev.working_precision(), 3 + 9);
    }

    /// A recurrence engineered so that the leading coefficient vanishes
    /// *identically* at one index. No modulus repairs that, so it must refuse.
    #[test]
    fn leading_coefficient_zero_refuses() {
        // (n − 4)·S(n+1) − S(n) = 0, S(0) = 1. At n = 4 the step is undefined.
        let rec = ModularRecurrence::new(
            vec![vec![z(-1)], vec![z(-4), z(1)]],
            vec![],
            vec![(z(1), z(1))],
            0,
        )
        .unwrap();
        assert!(rec.value_mod(4, 7, 3).is_ok(), "n = 3 step is fine");
        let err = rec.value_mod(5, 7, 3).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-007");
        assert!(format!("{err}").contains("vanishes at n = 4"), "{err}");
    }

    /// A sequence that leaves ℤ_p: `p·S(n+1) = S(n)` has `v_p(S(n)) = −n`.
    #[test]
    fn non_p_integral_step_refuses() {
        let rec =
            ModularRecurrence::new(vec![vec![z(-1)], vec![z(7)]], vec![], vec![(z(1), z(1))], 0)
                .unwrap();
        let err = rec.value_mod(1, 7, 3).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-007");
        assert!(format!("{err}").contains("not a p-adic integer"), "{err}");
    }

    #[test]
    fn initial_value_with_p_in_the_denominator_refuses() {
        let rec =
            ModularRecurrence::new(vec![vec![z(-1)], vec![z(1)]], vec![], vec![(z(1), z(7))], 0)
                .unwrap();
        let err = rec.value_mod(3, 7, 2).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-007");
    }

    /// Inhomogeneous: `(n+2)·S(n+1) − (2n+2)·S(n) = 1` is the true relation for
    /// `Σ_{k=0}^{n} binomial(n,k)/(k+1) = (2^{n+1} − 1)/(n+1)`.
    #[test]
    fn inhomogeneous_recurrence() {
        let rec = ModularRecurrence::new(
            vec![vec![z(-2), z(-2)], vec![z(2), z(1)]],
            vec![z(1)],
            vec![(z(1), z(1))],
            0,
        )
        .unwrap();
        for &p in &[11u64, 13, 101] {
            for n in [1i64, 2, 3, 5, 8] {
                let k = 3;
                let m = Integer::from(p).pow(k);
                let want = {
                    let num = Integer::from(2).pow(n as u32 + 1) - 1u32;
                    let den = Integer::from(n + 1);
                    let inv = den.invert(&m).unwrap();
                    Integer::from(&(num * inv) % &m).to_u64().unwrap()
                };
                assert_eq!(rec.value_mod(n, p, k).unwrap(), want, "n = {n}, p = {p}");
            }
        }
    }

    #[test]
    fn composite_and_oversized_moduli_refuse() {
        let rec = apery();
        let err = rec.value_mod(5, 9, 2).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-006");
        let err = rec.value_mod(5, 1_000_003, 4).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-006");
        let err = rec.value_mod(5, 7, 0).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-006");
    }

    #[test]
    fn malformed_recurrences_refuse() {
        assert!(ModularRecurrence::new(vec![vec![z(1)]], vec![], vec![], 0).is_err());
        assert!(ModularRecurrence::new(
            vec![vec![z(1)], vec![z(0), z(0)]],
            vec![],
            vec![(z(1), z(1))],
            0
        )
        .is_err());
        assert!(ModularRecurrence::new(
            vec![vec![z(1)], vec![z(1)]],
            vec![],
            vec![(z(1), z(1)), (z(2), z(1))],
            0
        )
        .is_err());
        let rec = apery();
        assert!(rec.evaluate(&[3, 3], 7, 2).is_err());
        assert!(rec.evaluate(&[], 7, 2).is_err());
        assert!(rec.evaluate(&[-1], 7, 2).is_err());
    }

    // -- binomial_mod ------------------------------------------------------

    fn binomial_exact(a: u64, b: u64) -> Integer {
        let mut acc = Integer::from(1);
        for i in 0..b {
            acc *= Integer::from(a - i);
            acc /= Integer::from(i + 1);
        }
        acc
    }

    #[test]
    fn binomial_matches_exact_arithmetic() {
        for &p in &[2u64, 3, 5, 7, 11, 13, 97] {
            for k in 1..=4u32 {
                if prime_power(p, k).is_none() {
                    continue;
                }
                let m = Integer::from(p).pow(k);
                for a in [0u64, 1, 5, 12, 40, 97, 200, 1000, 5000] {
                    for b in [0u64, 1, 2, 7, 13, 40, 99, 501] {
                        if b > a {
                            assert_eq!(binomial_mod(a, b as i128, p, k).unwrap(), 0);
                            continue;
                        }
                        let want = Integer::from(&binomial_exact(a, b) % &m).to_u64().unwrap();
                        assert_eq!(
                            binomial_mod(a, b as i128, p, k).unwrap(),
                            want,
                            "C({a},{b}) mod {p}^{k}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn binomial_lucas_agrees_with_the_prime_power_path() {
        // k = 1 is Lucas; check it against a digitwise Lucas product.
        for &p in &[5u64, 7, 13] {
            for a in [10u64, 99, 512, 4321, 99_999] {
                for b in [3u64, 17, 100, 4000] {
                    if b > a {
                        continue;
                    }
                    let mut lucas = 1u64 % p;
                    let (mut x, mut y) = (a, b);
                    while x > 0 || y > 0 {
                        let (dx, dy) = (x % p, y % p);
                        if dy > dx {
                            lucas = 0;
                            break;
                        }
                        let mut c = 1u64;
                        for i in 0..dy {
                            c = mul_mod(c, (dx - i) % p, p);
                            c = mul_mod(c, inv_mod((i + 1) % p, p).unwrap(), p);
                        }
                        lucas = mul_mod(lucas, c, p);
                        x /= p;
                        y /= p;
                    }
                    assert_eq!(binomial_mod(a, b as i128, p, 1).unwrap(), lucas);
                }
            }
        }
    }

    #[test]
    fn binomial_wilson_constant_is_right() {
        // Π_{0<i<p^k, p∤i} i mod p^k, by brute force.
        for &(p, k) in &[
            (2u64, 1u32),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 1),
            (3, 2),
            (5, 2),
            (7, 2),
        ] {
            let m = prime_power(p, k).unwrap();
            let mut want = 1u64 % m;
            for i in 1..m {
                if i % p != 0 {
                    want = mul_mod(want, i, m);
                }
            }
            assert_eq!(UnitFactorial::new(p, k, m).wilson, want, "p={p} k={k}");
        }
    }

    #[test]
    fn binomial_unit_prefix_matches_brute_force() {
        for &(p, k) in &[(2u64, 3u32), (3, 3), (5, 3), (7, 2), (11, 2)] {
            let m = prime_power(p, k).unwrap();
            let ctx = UnitFactorial::new(p, k, m);
            for n in 0..(3 * m).min(4000) {
                let mut want = 1u64 % m;
                for i in 1..=n {
                    if i % p != 0 {
                        want = mul_mod(want, i % m, m);
                    }
                }
                assert_eq!(ctx.unit_factorial(n), want, "p={p} k={k} n={n}");
            }
        }
    }

    #[test]
    fn binomial_refusals() {
        let err = binomial_mod(10, 3, 9, 2).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-006");
        let err = binomial_mod(10, 3, 7, 0).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-006");
        let err = binomial_mod(10, 3, 1_000_003, 4).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-006");
        // p² fits the backend, but a pass over 1 … p−1 does not fit the budget.
        let err = binomial_mod(10, 3, 1_000_000_007, 2).unwrap_err();
        assert_eq!(err.code(), "E-HOLO-008");
        assert_eq!(binomial_mod(10, -1, 7, 2).unwrap(), 0);
    }
}
