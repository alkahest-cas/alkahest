//! Angular-momentum coupling: Wigner 3j and 6j symbols, Clebsch–Gordan
//! coefficients, and the complex spherical harmonics `Y_l^m(θ, φ)`.
//!
//! # Exactness
//!
//! Every coupling coefficient is `±√(rational)` times a rational sum, so it is
//! returned as a [`Surd`] `q·√r` with `q` rational and `r` a square-free
//! positive integer.  Racah's single-sum formulas are evaluated over big
//! integers; the square-root factor is a ratio of factorials whose prime
//! factorisation is read off with Legendre's formula, so making `r`
//! square-free never factors a large integer.
//!
//! # Conventions
//!
//! * 3j symbol: Racah's formula (Edmonds 3.7.3; DLMF 34.2.4), the same
//!   normalisation and phase as SymPy's `wigner_3j` and Mathematica's
//!   `ThreeJSymbol`.
//! * Clebsch–Gordan: Condon–Shortley phase,
//!   `⟨j₁ m₁ j₂ m₂ | J M⟩ = (−1)^{j₁−j₂+M} √(2J+1) (j₁ j₂ J; m₁ m₂ −M)`
//!   (DLMF 34.1.1), as SymPy's `clebsch_gordan` and Mathematica's
//!   `ClebschGordan`.
//! * 6j symbol: Racah's formula (DLMF 34.4.1 / Edmonds 6.3.7).
//! * `Y_l^m`: `√((2l+1)/(4π) · (l−m)!/(l+m)!) · P_l^m(cos θ) · e^{imφ}` with
//!   the Condon–Shortley phase inside `P_l^m` (DLMF 14.30.1), orthonormal on
//!   the unit sphere; `Y_l^{−m} = (−1)^m conj(Y_l^m)`.  Same as SymPy's `Ynm`
//!   and Mathematica's `SphericalHarmonicY`.
//!
//! # Selection rules and refusals
//!
//! A coefficient that vanishes by a selection rule (triangle inequality,
//! `|m| > j`, `m₁ + m₂ + m₃ ≠ 0`, `j − m` not an integer) is an exact `0`.  An
//! argument that is not an integer or half-integer is refused with
//! `E-SPECFN-001`, a negative `j` with `E-SPECFN-002`.

use std::collections::BTreeMap;
use std::fmt;

use rug::{Integer, Rational};

use crate::kernel::{Domain, ExprId, ExprPool};

use super::error::SpecialFunctionError;
use super::{mk_add, mk_mul, pow_int, rat_node};

/// Largest `2j` (twice an angular momentum) accepted by the coupling
/// coefficients.  The factorials involved grow like `(4j)!`; past this the
/// exact answer is megabytes long.
pub const MAX_TWICE_J: i64 = 20_000;

/// Largest `l` for which [`spherical_harmonic_y`] builds the explicit form.
pub const MAX_SPHERICAL_L: i64 = 1000;

/// An integer or half-integer, stored as twice its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HalfInt {
    twice: i64,
}

impl HalfInt {
    /// The half-integer `t / 2`.
    pub const fn from_twice(t: i64) -> Self {
        HalfInt { twice: t }
    }

    /// The integer `n`.
    pub const fn from_int(n: i64) -> Self {
        HalfInt { twice: 2 * n }
    }

    /// `q` when `2q` is an integer that fits in an `i64`; otherwise
    /// `E-SPECFN-001`, naming `what`.
    pub fn from_rational(q: &Rational, what: &str) -> Result<Self, SpecialFunctionError> {
        let t = Rational::from(q * 2u32);
        if *t.denom() != 1 {
            return Err(SpecialFunctionError::NotHalfInteger {
                what: what.to_string(),
                value: q.to_string(),
            });
        }
        let twice = t
            .numer()
            .to_i64()
            .ok_or_else(|| SpecialFunctionError::TooLarge {
                what: what.to_string(),
                value: i64::MAX,
                max: MAX_TWICE_J / 2,
            })?;
        Ok(HalfInt { twice })
    }

    /// Twice the value.
    pub const fn twice(self) -> i64 {
        self.twice
    }

    /// The value as a rational.
    pub fn to_rational(self) -> Rational {
        Rational::from((self.twice, 2))
    }
}

impl fmt::Display for HalfInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.twice % 2 == 0 {
            write!(f, "{}", self.twice / 2)
        } else {
            write!(f, "{}/2", self.twice)
        }
    }
}

/// The exact number `coeff · √radicand`, `radicand ≥ 1` square-free.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Surd {
    /// The rational factor.
    pub coeff: Rational,
    /// The square-free radicand (`1` for a rational value).
    pub radicand: Integer,
}

impl Surd {
    /// Exact zero.
    pub fn zero() -> Self {
        Surd {
            coeff: Rational::new(),
            radicand: Integer::from(1),
        }
    }

    /// Is this exactly zero?
    pub fn is_zero(&self) -> bool {
        self.coeff == 0
    }

    /// The square of the value, exactly (`coeff² · radicand`).
    pub fn squared(&self) -> Rational {
        Rational::from(&self.coeff * &self.coeff) * &self.radicand
    }

    /// Nearest `f64`.
    pub fn to_f64(&self) -> f64 {
        let r = rug::Float::with_val(128, &self.radicand).sqrt();
        let v = r * rug::Float::with_val(128, &self.coeff);
        v.to_f64()
    }

    /// The value as a pool expression: `coeff` or `coeff · radicand^(1/2)`.
    pub fn to_expr(&self, pool: &ExprPool) -> ExprId {
        let c = rat_node(pool, &self.coeff);
        if self.radicand == 1 || self.coeff == 0 {
            return c;
        }
        let root = pool.pow(pool.integer(self.radicand.clone()), pool.rational(1, 2));
        mk_mul(pool, vec![c, root])
    }
}

impl fmt::Display for Surd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.radicand == 1 {
            write!(f, "{}", self.coeff)
        } else {
            write!(f, "{}*sqrt({})", self.coeff, self.radicand)
        }
    }
}

// ---------------------------------------------------------------------------
// Prime-exponent bookkeeping for the square-root factor
// ---------------------------------------------------------------------------

/// A positive rational as `Π pᵉ`, built from factorials and small integers.
struct PrimePower {
    primes: Vec<u64>,
    exps: BTreeMap<u64, i64>,
}

impl PrimePower {
    fn new(bound: u64) -> Self {
        // Sieve of Eratosthenes up to `bound`.
        let n = bound.max(2) as usize;
        let mut is = vec![true; n + 1];
        is[0] = false;
        is[1] = false;
        let mut i = 2;
        while i * i <= n {
            if is[i] {
                let mut j = i * i;
                while j <= n {
                    is[j] = false;
                    j += i;
                }
            }
            i += 1;
        }
        let primes = (2..=n).filter(|&k| is[k]).map(|k| k as u64).collect();
        PrimePower {
            primes,
            exps: BTreeMap::new(),
        }
    }

    /// Multiply by `(k!)^sign`.
    fn factorial(&mut self, k: u64, sign: i64) {
        for &p in &self.primes {
            if p > k {
                break;
            }
            // Legendre: v_p(k!) = Σ ⌊k/pⁱ⌋
            let mut v = 0i64;
            let mut q = k;
            while q > 0 {
                q /= p;
                v += q as i64;
            }
            *self.exps.entry(p).or_insert(0) += sign * v;
        }
    }

    /// Multiply by `k^sign` for `1 ≤ k ≤ bound`.
    fn small(&mut self, mut k: u64, sign: i64) {
        for &p in &self.primes {
            if p * p > k {
                break;
            }
            while k % p == 0 {
                k /= p;
                *self.exps.entry(p).or_insert(0) += sign;
            }
        }
        if k > 1 {
            *self.exps.entry(k).or_insert(0) += sign;
        }
    }

    /// `√(Π pᵉ) = coeff · √radicand`.
    fn sqrt(&self) -> (Rational, Integer) {
        let mut num = Integer::from(1);
        let mut den = Integer::from(1);
        let mut rad = Integer::from(1);
        for (&p, &e) in &self.exps {
            let q = e.div_euclid(2);
            let r = e.rem_euclid(2);
            if q > 0 {
                num *= Integer::from(Integer::u_pow_u(p as u32, q as u32));
            } else if q < 0 {
                den *= Integer::from(Integer::u_pow_u(p as u32, (-q) as u32));
            }
            if r == 1 {
                rad *= p;
            }
        }
        (Rational::from((num, den)), rad)
    }
}

fn fact(k: i64) -> Integer {
    Integer::from(Integer::factorial(k as u32))
}

fn check_j(j: HalfInt, what: &str) -> Result<(), SpecialFunctionError> {
    if j.twice < 0 {
        return Err(SpecialFunctionError::NegativeAngularMomentum {
            what: what.to_string(),
            value: j.to_string(),
        });
    }
    if j.twice > MAX_TWICE_J {
        return Err(SpecialFunctionError::TooLarge {
            what: format!("2·{what}"),
            value: j.twice,
            max: MAX_TWICE_J,
        });
    }
    Ok(())
}

fn check_m(m: HalfInt, what: &str) -> Result<(), SpecialFunctionError> {
    if m.twice.abs() > MAX_TWICE_J {
        return Err(SpecialFunctionError::TooLarge {
            what: format!("2·|{what}|"),
            value: m.twice.abs(),
            max: MAX_TWICE_J,
        });
    }
    Ok(())
}

/// Triangle condition with integer perimeter, on twice-values.
fn triangle(a: i64, b: i64, c: i64) -> bool {
    a + b - c >= 0 && a - b + c >= 0 && -a + b + c >= 0 && (a + b + c) % 2 == 0
}

/// `Δ(a, b, c)` into `pp` (factorials of the integer quantities; twice-values in).
fn add_delta(pp: &mut PrimePower, a: i64, b: i64, c: i64) {
    pp.factorial(((a + b - c) / 2) as u64, 1);
    pp.factorial(((a - b + c) / 2) as u64, 1);
    pp.factorial(((-a + b + c) / 2) as u64, 1);
    pp.factorial(((a + b + c) / 2 + 1) as u64, -1);
}

/// The 3j symbol as `(sign·S, √R)` pieces, or zero.  Shared by
/// [`wigner_3j`] and [`clebsch_gordan`]; `extra` multiplies `R` by a small
/// integer (the `2J+1` of a Clebsch–Gordan coefficient).
fn three_j_impl(
    j: [HalfInt; 3],
    m: [HalfInt; 3],
    extra: u64,
    extra_sign: i64,
) -> Result<Surd, SpecialFunctionError> {
    let [tj1, tj2, tj3] = j.map(|v| v.twice);
    let [tm1, tm2, tm3] = m.map(|v| v.twice);
    if tm1 + tm2 + tm3 != 0 {
        return Ok(Surd::zero());
    }
    for (tj, tm) in [(tj1, tm1), (tj2, tm2), (tj3, tm3)] {
        if tm.abs() > tj || (tj - tm) % 2 != 0 {
            return Ok(Surd::zero());
        }
    }
    if !triangle(tj1, tj2, tj3) {
        return Ok(Surd::zero());
    }
    // Integer quantities.
    let h = |t: i64| t / 2; // exact: every use below is of an even number
    let (j1m1m, j1m1p) = (h(tj1 - tm1), h(tj1 + tm1));
    let (j2m2m, j2m2p) = (h(tj2 - tm2), h(tj2 + tm2));
    let (j3m3m, j3m3p) = (h(tj3 - tm3), h(tj3 + tm3));
    let a = h(tj1 + tj2 - tj3); // j1 + j2 − j3
    let b1 = h(tj3 - tj2 + tm1); // j3 − j2 + m1
    let b2 = h(tj3 - tj1 - tm2); // j3 − j1 − m2
    let bound = (h(tj1 + tj2 + tj3) + 1).max(extra as i64).max(2) as u64;
    let mut pp = PrimePower::new(bound);
    add_delta(&mut pp, tj1, tj2, tj3);
    for k in [j1m1m, j1m1p, j2m2m, j2m2p, j3m3m, j3m3p] {
        pp.factorial(k as u64, 1);
    }
    if extra > 1 {
        pp.small(extra, 1);
    }
    let kmin = 0.max(-b1).max(-b2);
    let kmax = a.min(j1m1m).min(j2m2p);
    let mut s = Rational::new();
    for k in kmin..=kmax {
        if (k - kmin) % 16 == 15 {
            crate::budget::check()?;
        }
        let den =
            fact(k) * fact(b1 + k) * fact(b2 + k) * fact(a - k) * fact(j1m1m - k) * fact(j2m2p - k);
        let term = Rational::from((Integer::from(1), den));
        if k % 2 == 0 {
            s += term;
        } else {
            s -= term;
        }
    }
    // (−1)^{j1 − j2 − m3}
    let phase = h(tj1 - tj2 - tm3);
    let mut sign = if phase.rem_euclid(2) == 1 { -1 } else { 1 };
    sign *= extra_sign;
    let (c, r) = pp.sqrt();
    Ok(Surd {
        coeff: s * c * sign,
        radicand: r,
    })
}

/// The Wigner 3j symbol `(j₁ j₂ j₃; m₁ m₂ m₃)`, exactly.
///
/// # Errors
///
/// `E-SPECFN-002` for a negative `j`, `E-SPECFN-004` past [`MAX_TWICE_J`],
/// `E-BUDGET-*` if the active budget runs out.  Arguments violating a
/// selection rule give `0`.
pub fn wigner_3j(
    j1: HalfInt,
    j2: HalfInt,
    j3: HalfInt,
    m1: HalfInt,
    m2: HalfInt,
    m3: HalfInt,
) -> Result<Surd, SpecialFunctionError> {
    check_j(j1, "j1")?;
    check_j(j2, "j2")?;
    check_j(j3, "j3")?;
    check_m(m1, "m1")?;
    check_m(m2, "m2")?;
    check_m(m3, "m3")?;
    three_j_impl([j1, j2, j3], [m1, m2, m3], 1, 1)
}

/// The Clebsch–Gordan coefficient `⟨j₁ m₁ j₂ m₂ | j₃ m₃⟩` (Condon–Shortley
/// phase), exactly.  Argument order is SymPy's:
/// `clebsch_gordan(j1, j2, j3, m1, m2, m3)`.
///
/// # Errors
///
/// As [`wigner_3j`].
pub fn clebsch_gordan(
    j1: HalfInt,
    j2: HalfInt,
    j3: HalfInt,
    m1: HalfInt,
    m2: HalfInt,
    m3: HalfInt,
) -> Result<Surd, SpecialFunctionError> {
    check_j(j1, "j1")?;
    check_j(j2, "j2")?;
    check_j(j3, "j3")?;
    check_m(m1, "m1")?;
    check_m(m2, "m2")?;
    check_m(m3, "m3")?;
    // (−1)^{j1 − j2 + m3} √(2 j3 + 1) (j1 j2 j3; m1 m2 −m3)
    let t = j1.twice - j2.twice + m3.twice;
    if t % 2 != 0 {
        return Ok(Surd::zero()); // j − m parity already fails
    }
    let sign = if (t / 2).rem_euclid(2) == 1 { -1 } else { 1 };
    three_j_impl(
        [j1, j2, j3],
        [m1, m2, HalfInt::from_twice(-m3.twice)],
        (j3.twice + 1) as u64,
        sign,
    )
}

/// The Wigner 6j symbol `{j₁ j₂ j₃; j₄ j₅ j₆}`, exactly.
///
/// # Errors
///
/// As [`wigner_3j`].  A triad that fails the triangle condition gives `0`.
pub fn wigner_6j(
    j1: HalfInt,
    j2: HalfInt,
    j3: HalfInt,
    j4: HalfInt,
    j5: HalfInt,
    j6: HalfInt,
) -> Result<Surd, SpecialFunctionError> {
    for (j, w) in [
        (j1, "j1"),
        (j2, "j2"),
        (j3, "j3"),
        (j4, "j4"),
        (j5, "j5"),
        (j6, "j6"),
    ] {
        check_j(j, w)?;
    }
    let [t1, t2, t3, t4, t5, t6] = [j1, j2, j3, j4, j5, j6].map(|v| v.twice);
    let triads = [(t1, t2, t3), (t1, t5, t6), (t4, t2, t6), (t4, t5, t3)];
    if triads.iter().any(|&(a, b, c)| !triangle(a, b, c)) {
        return Ok(Surd::zero());
    }
    let h = |t: i64| t / 2;
    let a1 = h(t1 + t2 + t3);
    let a2 = h(t1 + t5 + t6);
    let a3 = h(t4 + t2 + t6);
    let a4 = h(t4 + t5 + t3);
    let b1 = h(t1 + t2 + t4 + t5);
    let b2 = h(t2 + t3 + t5 + t6);
    let b3 = h(t3 + t1 + t6 + t4);
    let bound = (a1.max(a2).max(a3).max(a4) + 1).max(2) as u64;
    let mut pp = PrimePower::new(bound);
    for &(a, b, c) in &triads {
        add_delta(&mut pp, a, b, c);
    }
    let tmin = a1.max(a2).max(a3).max(a4);
    let tmax = b1.min(b2).min(b3);
    let mut s = Rational::new();
    for t in tmin..=tmax {
        if (t - tmin) % 16 == 15 {
            crate::budget::check()?;
        }
        let num = fact(t + 1);
        let den = fact(t - a1)
            * fact(t - a2)
            * fact(t - a3)
            * fact(t - a4)
            * fact(b1 - t)
            * fact(b2 - t)
            * fact(b3 - t);
        let term = Rational::from((num, den));
        if t % 2 == 0 {
            s += term;
        } else {
            s -= term;
        }
    }
    let (c, r) = pp.sqrt();
    Ok(Surd {
        coeff: s * c,
        radicand: r,
    })
}

/// The complex spherical harmonic `Y_l^m(θ, φ)` as an explicit expression,
///
/// ```text
/// Y_l^m = c·√r·π^{−1/2} · sin(θ)^{|m|} · Q(cos θ) · exp(i m φ),
/// ```
///
/// `c` rational, `r` square-free, `Q` a polynomial with rational
/// coefficients — `√((2l+1)/(4π) (l−m)!/(l+m)!) P_l^m(cos θ) e^{imφ}` with the
/// Condon–Shortley phase.  `(1 − cos²θ)^{|m|/2}` is written `sin(θ)^{|m|}`,
/// which is exact for the polar angle's range `0 ≤ θ ≤ π`.  `|m| > l` gives
/// `0`.
///
/// # Errors
///
/// `E-SPECFN-003` when `l` or `m` is not an integer, `E-SPECFN-002` for
/// `l < 0`, `E-SPECFN-004` for `l` past [`MAX_SPHERICAL_L`].
pub fn spherical_harmonic_y(
    l: &Rational,
    m: &Rational,
    theta: ExprId,
    phi: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    for (v, w) in [(l, "l"), (m, "m")] {
        if *v.denom() != 1 {
            return Err(SpecialFunctionError::NotAnInteger {
                what: w.to_string(),
                value: v.to_string(),
            });
        }
    }
    let l = l.numer().clone();
    if l < 0 {
        return Err(SpecialFunctionError::NegativeAngularMomentum {
            what: "l".into(),
            value: l.to_string(),
        });
    }
    if l > MAX_SPHERICAL_L {
        return Err(SpecialFunctionError::TooLarge {
            what: "l".into(),
            value: l.to_i64().unwrap_or(i64::MAX),
            max: MAX_SPHERICAL_L,
        });
    }
    let l = l.to_i64().expect("bounded");
    let m_int = m.numer().clone();
    if m_int.clone().abs() > l {
        return Ok(pool.integer(0));
    }
    let m = m_int.to_i64().expect("|m| ≤ l");
    let am = m.unsigned_abs() as i64;
    // N² = (2l+1)(l−|m|)! / (4 (l+|m|)!)  (the π goes in separately).
    let mut pp = PrimePower::new((l + am + 2).max(2 * l + 1) as u64);
    pp.small((2 * l + 1) as u64, 1);
    pp.factorial((l - am) as u64, 1);
    pp.factorial((l + am) as u64, -1);
    pp.small(4, -1);
    let (c, r) = pp.sqrt();
    // Q = (−1)^m d^{|m|}P_l/dx^{|m|} for m ≥ 0; the plain derivative for m < 0.
    let d = super::coeffs::derivative(&super::coeffs::legendre(l as u32)?, am as u32);
    let sign = if m > 0 && am % 2 == 1 { -1 } else { 1 };
    let cos_t = pool.func("cos", vec![theta]);
    let nonzero: Vec<(usize, Rational)> = d
        .iter()
        .enumerate()
        .filter(|(_, ck)| **ck != 0)
        .map(|(k, ck)| (k, Rational::from(ck * sign)))
        .collect();
    // A single-term Q folds its coefficient into the normalisation.
    let (c, q) = if let [(k, ck)] = nonzero.as_slice() {
        (c * ck, pow_int(pool, cos_t, *k as u32))
    } else {
        let terms = nonzero
            .iter()
            .map(|(k, ck)| {
                mk_mul(
                    pool,
                    vec![rat_node(pool, ck), pow_int(pool, cos_t, *k as u32)],
                )
            })
            .collect();
        (c, mk_add(pool, terms))
    };
    let norm = Surd {
        coeff: c,
        radicand: r,
    }
    .to_expr(pool);
    let pi = pool.symbol("pi", Domain::Real);
    let mut factors = vec![norm, pool.pow(pi, pool.rational(-1, 2))];
    if am > 0 {
        factors.push(pow_int(pool, pool.func("sin", vec![theta]), am as u32));
    }
    factors.push(q);
    if m != 0 {
        let i = pool.imaginary_unit();
        factors.push(pool.func("exp", vec![mk_mul(pool, vec![pool.integer(m), i, phi])]));
    }
    Ok(mk_mul(pool, factors))
}

/// The **real** spherical harmonic `Y_{l,m}(θ, φ)` (the tesseral basis of
/// quantum chemistry and geophysics) as an explicit expression.
///
/// Convention (Wikipedia, "Spherical harmonics § Real form"; the Condon–Shortley
/// phase of the complex `Y_l^m` cancels against the sign of the definition):
///
/// ```text
/// Y_{l,m} = √2 (−1)^m Im[Y_l^{|m|}]    (m < 0)    =  √2·N·P_l^{|m|}(cos θ)·sin(|m|φ)
/// Y_{l,0} = Y_l^0
/// Y_{l,m} = √2 (−1)^m Re[Y_l^m]        (m > 0)    =  √2·N·P_l^{m}(cos θ)·cos(mφ)
/// ```
///
/// with `N = √((2l+1)/(4π)·(l−|m|)!/(l+|m|)!)` and `P_l^{|m|}` the Legendre
/// function **without** the Condon–Shortley phase (so `Y_{1,1} = √(3/4π)·x/r`,
/// `Y_{1,−1} = √(3/4π)·y/r`).  The set `{Y_{l,m}}` is orthonormal on the unit
/// sphere and real.  `(1 − cos²θ)^{|m|/2}` is written `sin(θ)^{|m|}`, exact
/// for `0 ≤ θ ≤ π`; `|m| > l` gives `0`.
///
/// # Errors
///
/// As [`spherical_harmonic_y`].
pub fn real_spherical_harmonic_y(
    l: &Rational,
    m: &Rational,
    theta: ExprId,
    phi: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    for (v, w) in [(l, "l"), (m, "m")] {
        if *v.denom() != 1 {
            return Err(SpecialFunctionError::NotAnInteger {
                what: w.to_string(),
                value: v.to_string(),
            });
        }
    }
    let l = l.numer().clone();
    if l < 0 {
        return Err(SpecialFunctionError::NegativeAngularMomentum {
            what: "l".into(),
            value: l.to_string(),
        });
    }
    if l > MAX_SPHERICAL_L {
        return Err(SpecialFunctionError::TooLarge {
            what: "l".into(),
            value: l.to_i64().unwrap_or(i64::MAX),
            max: MAX_SPHERICAL_L,
        });
    }
    let l = l.to_i64().expect("bounded");
    let m_int = m.numer().clone();
    if m_int.clone().abs() > l {
        return Ok(pool.integer(0));
    }
    let m = m_int.to_i64().expect("|m| ≤ l");
    if m == 0 {
        return spherical_harmonic_y(&Rational::from(l), &Rational::from(0), theta, phi, pool);
    }
    let am = m.unsigned_abs() as i64;
    // (√2 N)² = (2l+1)(l−|m|)! / (2π (l+|m|)!)  (the π goes in separately).
    let mut pp = PrimePower::new((l + am + 2).max(2 * l + 1) as u64);
    pp.small((2 * l + 1) as u64, 1);
    pp.factorial((l - am) as u64, 1);
    pp.factorial((l + am) as u64, -1);
    pp.small(2, -1);
    let (c, r) = pp.sqrt();
    // P_l^{|m|} without the Condon–Shortley phase: the plain derivative.
    let d = super::coeffs::derivative(&super::coeffs::legendre(l as u32)?, am as u32);
    let cos_t = pool.func("cos", vec![theta]);
    let nonzero: Vec<(usize, Rational)> = d
        .iter()
        .enumerate()
        .filter(|(_, ck)| **ck != 0)
        .map(|(k, ck)| (k, Rational::from(ck)))
        .collect();
    let (c, q) = if let [(k, ck)] = nonzero.as_slice() {
        (c * ck, pow_int(pool, cos_t, *k as u32))
    } else {
        let terms = nonzero
            .iter()
            .map(|(k, ck)| {
                mk_mul(
                    pool,
                    vec![rat_node(pool, ck), pow_int(pool, cos_t, *k as u32)],
                )
            })
            .collect();
        (c, mk_add(pool, terms))
    };
    let norm = Surd {
        coeff: c,
        radicand: r,
    }
    .to_expr(pool);
    let pi = pool.symbol("pi", Domain::Real);
    let trig = if m > 0 { "cos" } else { "sin" };
    Ok(mk_mul(
        pool,
        vec![
            norm,
            pool.pow(pi, pool.rational(-1, 2)),
            pow_int(pool, pool.func("sin", vec![theta]), am as u32),
            q,
            pool.func(trig, vec![mk_mul(pool, vec![pool.integer(am), phi])]),
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hi(t: i64) -> HalfInt {
        HalfInt::from_twice(t)
    }

    #[test]
    fn three_j_known_values() {
        // (1 1 0; 0 0 0) = −1/√3 = −(1/3)√3
        let v = wigner_3j(hi(2), hi(2), hi(0), hi(0), hi(0), hi(0)).unwrap();
        assert_eq!(v.coeff, Rational::from((-1, 3)));
        assert_eq!(v.radicand, 3);
        // (1/2 1/2 1; 1/2 −1/2 0) = 1/√6
        let v = wigner_3j(hi(1), hi(1), hi(2), hi(1), hi(-1), hi(0)).unwrap();
        assert_eq!(v.squared(), Rational::from((1, 6)));
        assert!(v.coeff > 0);
        // selection rules
        assert!(wigner_3j(hi(2), hi(2), hi(6), hi(0), hi(0), hi(0))
            .unwrap()
            .is_zero());
        assert!(wigner_3j(hi(2), hi(2), hi(2), hi(2), hi(0), hi(0))
            .unwrap()
            .is_zero());
        // (1 1 1; 0 0 0) = 0 by parity of the sum
        assert!(wigner_3j(hi(2), hi(2), hi(2), hi(0), hi(0), hi(0))
            .unwrap()
            .is_zero());
    }

    #[test]
    fn clebsch_gordan_spin_half() {
        // ⟨½ ½ ½ −½ | 1 0⟩ = 1/√2,  ⟨½ ½ ½ −½ | 0 0⟩ = 1/√2,
        // ⟨½ −½ ½ ½ | 0 0⟩ = −1/√2
        let a = clebsch_gordan(hi(1), hi(1), hi(2), hi(1), hi(-1), hi(0)).unwrap();
        assert_eq!(a.squared(), Rational::from((1, 2)));
        assert!(a.coeff > 0);
        let b = clebsch_gordan(hi(1), hi(1), hi(0), hi(1), hi(-1), hi(0)).unwrap();
        assert_eq!(b.squared(), Rational::from((1, 2)));
        assert!(b.coeff > 0);
        let c = clebsch_gordan(hi(1), hi(1), hi(0), hi(-1), hi(1), hi(0)).unwrap();
        assert!(c.coeff < 0);
    }

    #[test]
    fn six_j_known_value() {
        // {1 1 1; 1 1 1} = 1/6
        let v = wigner_6j(hi(2), hi(2), hi(2), hi(2), hi(2), hi(2)).unwrap();
        assert_eq!(v.coeff, Rational::from((1, 6)));
        assert_eq!(v.radicand, 1);
        // {1/2 1/2 1; 1/2 1/2 0} = 1/2
        let v = wigner_6j(hi(1), hi(1), hi(2), hi(1), hi(1), hi(0)).unwrap();
        assert_eq!(v.squared(), Rational::from((1, 4)));
        assert!(v.coeff > 0);
    }

    #[test]
    fn refusals() {
        let e = HalfInt::from_rational(&Rational::from((1, 3)), "j1").unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&e), "E-SPECFN-001");
        let e = wigner_3j(hi(-2), hi(2), hi(0), hi(0), hi(0), hi(0)).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&e), "E-SPECFN-002");
    }

    /// Σ_{m1,m2} ⟨j1 m1 j2 m2|J M⟩⟨j1 m1 j2 m2|J' M'⟩ = δ_{JJ'}δ_{MM'}, exactly.
    #[test]
    fn clebsch_gordan_orthogonality() {
        let (tj1, tj2) = (3_i64, 2_i64); // j1 = 3/2, j2 = 1
        let js: Vec<i64> = ((tj1 - tj2).abs()..=tj1 + tj2).step_by(2).collect();
        for &tj in &js {
            for &tjp in &js {
                for tm in (-tj..=tj).step_by(2) {
                    let mut total = Rational::new();
                    for tm1 in (-tj1..=tj1).step_by(2) {
                        let tm2 = tm - tm1;
                        if tm2.abs() > tj2 {
                            continue;
                        }
                        let a = clebsch_gordan(hi(tj1), hi(tj2), hi(tj), hi(tm1), hi(tm2), hi(tm))
                            .unwrap();
                        let b = clebsch_gordan(hi(tj1), hi(tj2), hi(tjp), hi(tm1), hi(tm2), hi(tm))
                            .unwrap();
                        if a.is_zero() || b.is_zero() {
                            continue;
                        }
                        // a·b = ca·cb·√(ra·rb); for the sum to be rational both
                        // radicands agree whenever the product is non-zero.
                        if a.radicand == b.radicand {
                            total += Rational::from(&a.coeff * &b.coeff) * &a.radicand;
                        } else {
                            // Different square-free radicands: the product is
                            // irrational, which must cancel in the sum; fall back
                            // to floating point for this pair.
                            total += Rational::from_f64(a.to_f64() * b.to_f64()).unwrap();
                        }
                    }
                    let want = if tj == tjp { 1 } else { 0 };
                    let diff = (total - Rational::from(want)).abs();
                    assert!(diff < Rational::from((1, 1_000_000_000)), "{tj} {tjp} {tm}");
                }
            }
        }
    }
}
