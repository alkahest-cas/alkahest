//! Discrete logarithms in `(ℤ/pℤ)^×`: Pohlig–Hellman over the factorisation of
//! the order of the base, with baby-step giant-step or Pollard's rho in each
//! prime-order subgroup.
//!
//! # Why not the linear sweep it replaces
//!
//! The sweep was `O(p)` multiplications with no budget checkpoint: `p = 2³¹−1`
//! under `Budget(wall_ms=2000)` ran for 55 s, and `2¹²⁷−1` never finished.
//! Pohlig–Hellman reduces the problem to one discrete log per prime `q`
//! dividing `ord(base)`, each solved in `O(√q)` group operations, so the cost
//! is governed by the **largest prime factor** of the order — smooth orders of
//! any size are cheap, and a 64-bit prime factor is genuinely hard (it is for
//! every generic algorithm).
//!
//! # What is exact here
//!
//! Every exponent the subgroup solvers return is checked by a modular
//! exponentiation before it is used, and the final answer is checked once
//! more against the input, so a wrong exponent can only ever surface as an
//! `Internal`-style refusal, never as a value. The answer is reduced modulo
//! `ord(base)`, which makes it the *smallest* non-negative exponent — the
//! contract the sweep had.
//!
//! # Budget
//!
//! The factorisation of `p − 1` goes through [`super::factorint_bounded`], and
//! every subgroup solver polls [`crate::budget::check_all`] every
//! [`POLL_EVERY`] group operations. A trip is
//! [`super::arith::ArithmeticError::Resource`], never a partial exponent.

use super::arith::ArithmeticError;
use super::NumberTheoryError;
use crate::budget::{self, BudgetTrip};
use rug::ops::RemRounding;
use rug::Integer;
use std::collections::HashMap;
use std::str::FromStr;

/// Group operations between two budget polls. A poll is a thread-local read
/// plus `Instant::now()`; a group operation is a bignum multiply and reduce,
/// so this keeps polling well under one percent of the work while bounding
/// the overshoot of a wall budget to microseconds.
const POLL_EVERY: u64 = 4096;

/// Prime-order subgroups of order at most this are solved by trying every
/// exponent; below it a hash table is pure overhead.
const BRUTE_MAX: u64 = 256;

/// Largest baby-step table, in entries. `2²¹` entries is a few tens of MB and
/// covers prime orders up to `2⁴²`; above that Pollard's rho, which needs no
/// memory, takes over.
const BSGS_MAX_TABLE: u64 = 1 << 21;

fn poll(counter: &mut u64) -> Result<(), BudgetTrip> {
    *counter += 1;
    if *counter % POLL_EVERY == 0 {
        budget::check_all()?;
    }
    Ok(())
}

fn powm(b: &Integer, e: &Integer, p: &Integer) -> Integer {
    // `e ≥ 0` at every call site, so `pow_mod_ref` cannot fail.
    match b.pow_mod_ref(e, p) {
        Some(v) => Integer::from(v),
        None => Integer::new(),
    }
}

fn mulm(a: &mut Integer, b: &Integer, p: &Integer) {
    *a *= b;
    *a %= p;
}

fn internal(msg: &'static str) -> ArithmeticError {
    ArithmeticError::Input(NumberTheoryError::Domain { msg })
}

/// Smallest `e ≥ 0` with `base^e ≡ residue (mod p)`, for `p` prime,
/// `base, residue` already reduced into `[0, p)` with `base ≠ 0`.
pub(super) fn discrete_log_prime(
    r: &Integer,
    b: &Integer,
    p: &Integer,
) -> Result<Integer, ArithmeticError> {
    if r.is_zero() {
        // A unit never powers to zero.
        return Err(NumberTheoryError::NoSolution.into());
    }
    if *r == 1 {
        return Ok(Integer::new());
    }
    let group_order = Integer::from(p - 1u32);

    // ord(b): strip every prime from p − 1 that b's order does not need.
    let (_, fac) = super::factorint_bounded(&group_order.to_string())?;
    let mut n = group_order.clone();
    let mut order_fac: Vec<(Integer, u32)> = Vec::new();
    for (q_str, e) in fac {
        let q =
            Integer::from_str(&q_str).map_err(|_| internal("factorint returned a non-integer"))?;
        let mut kept = e as u32;
        while kept > 0 {
            let cofactor = Integer::from(&n / &q);
            if powm(b, &cofactor, p) == 1 {
                n = cofactor;
                kept -= 1;
            } else {
                break;
            }
        }
        if kept > 0 {
            order_fac.push((q, kept));
        }
        budget::check_all()?;
    }

    // `(ℤ/pℤ)^×` is cyclic, so `r ∈ ⟨b⟩` iff `r^ord(b) = 1`.
    if powm(r, &n, p) != 1 {
        return Err(NumberTheoryError::NoSolution.into());
    }

    let b_inv = Integer::from(
        b.invert_ref(p)
            .ok_or_else(|| internal("base is not a unit modulo a prime"))?,
    );

    // Pohlig–Hellman: x mod q^e for each prime power, then CRT.
    let mut x_all = Integer::new();
    let mut modulus = Integer::from(1);
    for (q, e) in &order_fac {
        let g0 = powm(b, &Integer::from(&n / q), p); // order exactly q
        let mut x = Integer::new();
        let mut qk = Integer::from(1);
        for _ in 0..*e {
            // h = (r · b^{−x})^{n / q^{k+1}} lies in ⟨g0⟩.
            let mut h = powm(&b_inv, &x, p);
            mulm(&mut h, r, p);
            let qk1 = Integer::from(&qk * q);
            let h = powm(&h, &Integer::from(&n / &qk1), p);
            let d = dlog_prime_order(&g0, &h, q, p)?;
            x += Integer::from(&d * &qk);
            qk = qk1;
        }
        // CRT: x_all ≡ x_all (mod modulus), ≡ x (mod qk).
        let m_inv = Integer::from(
            modulus
                .invert_ref(&qk)
                .ok_or_else(|| internal("prime powers of one order were not coprime"))?,
        );
        let mut t = Integer::from(&x - &x_all);
        t *= &m_inv;
        t = t.rem_euc(&qk);
        x_all += Integer::from(&modulus * &t);
        modulus *= &qk;
    }
    x_all = x_all.rem_euc(&n);

    if powm(b, &x_all, p) != *r {
        return Err(internal(
            "discrete_log: the Pohlig–Hellman exponent failed its final check",
        ));
    }
    Ok(x_all)
}

/// `d ∈ [0, q)` with `g^d = h`, where `g` has prime order `q` and `h ∈ ⟨g⟩`.
fn dlog_prime_order(
    g: &Integer,
    h: &Integer,
    q: &Integer,
    p: &Integer,
) -> Result<Integer, ArithmeticError> {
    if *h == 1 {
        return Ok(Integer::new());
    }
    let d = match q.to_u64() {
        Some(qs) if qs <= BRUTE_MAX => brute(g, h, qs, p)?,
        Some(qs) if isqrt_ceil(qs) <= BSGS_MAX_TABLE => bsgs(g, h, qs, p)?,
        _ => rho(g, h, q, p)?,
    };
    match d {
        Some(d) if powm(g, &d, p) == *h => Ok(d),
        _ => Err(internal(
            "discrete_log: a prime-order subgroup solver returned an unverified exponent",
        )),
    }
}

fn isqrt_ceil(n: u64) -> u64 {
    let mut s = (n as f64).sqrt() as u64;
    while s.saturating_mul(s) > n {
        s -= 1;
    }
    while s.saturating_mul(s) < n {
        s += 1;
    }
    s
}

fn brute(g: &Integer, h: &Integer, q: u64, p: &Integer) -> Result<Option<Integer>, BudgetTrip> {
    let mut cur = Integer::from(1);
    let mut polls = 0;
    for e in 0..q {
        if cur == *h {
            return Ok(Some(Integer::from(e)));
        }
        mulm(&mut cur, g, p);
        poll(&mut polls)?;
    }
    Ok(None)
}

/// A 64-bit key for a residue: the residue itself when `p < 2⁶⁴` (so keys are
/// exact), otherwise its low 64 bits, with every hit re-verified by the caller.
fn key(x: &Integer) -> u64 {
    x.to_u64_wrapping()
}

fn bsgs(g: &Integer, h: &Integer, q: u64, p: &Integer) -> Result<Option<Integer>, BudgetTrip> {
    let m = isqrt_ceil(q).max(1);
    let exact_keys = p.significant_bits() <= 64;
    let mut table: HashMap<u64, u32> = HashMap::with_capacity(m as usize);
    // Distinct baby steps whose (truncated) keys collided; scanned on a hit.
    let mut overflow: Vec<(u64, u32)> = Vec::new();
    let mut polls = 0;
    let mut cur = Integer::from(1);
    for j in 0..m {
        let k = key(&cur);
        if table.contains_key(&k) {
            // With exact keys a repeat would mean g^j = g^i for i < j < q,
            // impossible in a group of order q; with truncated keys it is a
            // collision of distinct residues and both must be kept.
            if !exact_keys {
                overflow.push((k, j as u32));
            }
        } else {
            table.insert(k, j as u32);
        }
        mulm(&mut cur, g, p);
        poll(&mut polls)?;
    }
    // giant = g^{−m}
    let g_inv = match g.invert_ref(p) {
        Some(v) => Integer::from(v),
        None => return Ok(None),
    };
    let giant = powm(&g_inv, &Integer::from(m), p);
    let mut gamma = h.clone();
    for i in 0..=m {
        let k = key(&gamma);
        let mut candidates: Vec<u32> = Vec::new();
        if let Some(&j) = table.get(&k) {
            candidates.push(j);
            for &(ok, oj) in &overflow {
                if ok == k {
                    candidates.push(oj);
                }
            }
        }
        for j in candidates {
            let e = Integer::from(i) * Integer::from(m) + Integer::from(j);
            if exact_keys || powm(g, &e, p) == *h {
                return Ok(Some(Integer::from(&e % Integer::from(q))));
            }
        }
        mulm(&mut gamma, &giant, p);
        poll(&mut polls)?;
    }
    Ok(None)
}

/// SplitMix64, for the rho walk's multipliers and restarts. Deterministic
/// (seeded from the active budget's seed, or a constant) so runs reproduce.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, q: &Integer) -> Integer {
        // Two words of randomness reduced mod q: bias below 2^-64 for q < 2^64,
        // and irrelevant to correctness either way (every result is verified).
        let hi = Integer::from(self.next()) << 64;
        let v = hi + Integer::from(self.next());
        v % q
    }
}

/// Number of multipliers in Teske's r-adding walk; 20 gives a walk whose
/// collision time is within a few percent of a truly random map.
const RHO_ADDERS: usize = 20;

/// Pollard's rho with an r-adding walk and Brent's cycle detection: `O(√q)`
/// group operations, `O(1)` memory.
fn rho(g: &Integer, h: &Integer, q: &Integer, p: &Integer) -> Result<Option<Integer>, BudgetTrip> {
    let mut rng = SplitMix64(budget::seed().unwrap_or(0x5EED_D106_0000_0001));
    let mut polls = 0;
    // A handful of restarts covers the (probability ~1/q) degenerate
    // collisions; the budget, not this count, bounds the total work.
    for _attempt in 0..64 {
        let mut mult: Vec<(Integer, Integer, Integer)> = Vec::with_capacity(RHO_ADDERS);
        for _ in 0..RHO_ADDERS {
            let a = rng.below(q);
            let b = rng.below(q);
            let mut m = powm(g, &a, p);
            mulm(&mut m, &powm(h, &b, p), p);
            mult.push((m, a, b));
        }
        let step = |x: &mut Integer, a: &mut Integer, b: &mut Integer| {
            let s = (x.to_u64_wrapping() % RHO_ADDERS as u64) as usize;
            let (m, ma, mb) = &mult[s];
            mulm(x, m, p);
            *a += ma;
            if *a >= *q {
                *a -= q;
            }
            *b += mb;
            if *b >= *q {
                *b -= q;
            }
        };
        let a0 = rng.below(q);
        let b0 = rng.below(q);
        let mut x = powm(g, &a0, p);
        mulm(&mut x, &powm(h, &b0, p), p);
        let (mut a, mut b) = (a0, b0);
        // Brent: compare against a saved point, doubling the lap length.
        let (mut sx, mut sa, mut sb) = (x.clone(), a.clone(), b.clone());
        let mut power: u64 = 1;
        let mut lam: u64 = 0;
        loop {
            step(&mut x, &mut a, &mut b);
            lam += 1;
            poll(&mut polls)?;
            if x == sx {
                break;
            }
            if lam == power {
                sx = x.clone();
                sa = a.clone();
                sb = b.clone();
                power <<= 1;
                lam = 0;
            }
        }
        // g^a h^b = g^sa h^sb  ⇒  d·(b − sb) ≡ sa − a (mod q).
        let db = Integer::from(&b - &sb).rem_euc(q);
        if db.is_zero() {
            continue;
        }
        let Some(inv) = db.invert_ref(q).map(Integer::from) else {
            continue;
        };
        let d = (Integer::from(&sa - &a) * inv).rem_euc(q);
        if powm(g, &d, p) == *h {
            return Ok(Some(d));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::super::{discrete_log, discrete_log_bounded};
    use super::*;
    use std::time::{Duration, Instant};

    fn z(s: &str) -> Integer {
        Integer::from_str(s).unwrap()
    }

    /// Exhaustive against the definition on every (residue, base) pair for a
    /// few small primes: the answer is the *smallest* exponent, and NoSolution
    /// exactly when none exists.
    #[test]
    fn matches_brute_force_on_small_primes() {
        for p in [2u64, 3, 5, 7, 11, 13, 17, 31, 61, 97, 101, 127] {
            for b in 0..p {
                for r in 0..p {
                    let mut cur = 1 % p;
                    let mut want = None;
                    for e in 0..p {
                        if cur == r {
                            want = Some(e);
                            break;
                        }
                        cur = cur * b % p;
                    }
                    let got = discrete_log(&r.to_string(), &b.to_string(), &p.to_string());
                    match want {
                        Some(e) => assert_eq!(got.unwrap(), e.to_string(), "p={p} b={b} r={r}"),
                        None => assert!(
                            matches!(got, Err(NumberTheoryError::NoSolution)),
                            "p={p} b={b} r={r}: {got:?}"
                        ),
                    }
                }
            }
        }
    }

    fn check_large(p: &str, b: &str, e: &str) {
        let pm = z(p);
        let r = powm(&z(b), &z(e), &pm);
        let x = z(&discrete_log(&r.to_string(), b, p).unwrap());
        assert_eq!(powm(&z(b), &x, &pm), r, "p={p}");
        assert!(x < pm, "p={p}");
    }

    /// The report's repro: `p = 2³¹ − 1` took 55 s, `2¹²⁷ − 1` never finished.
    #[test]
    fn large_primes_with_smooth_orders_are_fast() {
        let t = Instant::now();
        check_large("2147483647", "7", "987654321");
        check_large(
            "170141183460469231731687303715884105727",
            "3",
            "123456789012345678901234567890123",
        );
        check_large("2305843009213693951", "37", "1234567890123456789");
        check_large("1000000000039", "5", "777777777777");
        assert!(t.elapsed() < Duration::from_secs(20), "{:?}", t.elapsed());
    }

    /// A prime order between 2^42 and 2^64 goes through Pollard's rho.
    /// `p = 2q + 1` with `q ≈ 2^45` prime; 4 is a square, so it has order `q`.
    #[test]
    fn rho_path_is_exercised_and_verified() {
        let q = z("35184372090953");
        let p = z("70368744181907");
        let g = Integer::from(4);
        let h = powm(&g, &z("12345678901234"), &p);
        let d = rho(&g, &h, &q, &p).unwrap().unwrap();
        assert_eq!(powm(&g, &d, &p), h);
        // And end to end through the public entry point.
        let x = discrete_log(&h.to_string(), "4", "70368744181907").unwrap();
        assert_eq!(x, "12345678901234");
    }

    /// The BSGS path with truncated (low-64-bit) keys: `p > 2^64`, smooth
    /// enough order but one factor above the brute-force range.
    #[test]
    fn bsgs_with_truncated_keys() {
        let p = "170141183460469231731687303715884105727";
        check_large(p, "5", "99999999999999999999999");
        check_large(p, "2", "77");
    }

    /// A safe prime near 2^62: the order of 4 is a 61-bit prime, which no
    /// generic algorithm finishes quickly. The budget must stop it.
    #[test]
    fn budget_trips_on_a_hard_prime_order() {
        let p = "4611686018427394499";
        let r = powm(&z("4"), &z("987654321987654321"), &z(p));
        let wall = Duration::from_millis(300);
        let t = Instant::now();
        let out = {
            let _g = budget::enter(budget::Budget::new().with_wall(wall));
            discrete_log_bounded(&r.to_string(), "4", p)
        };
        let el = t.elapsed();
        assert!(matches!(out, Err(ArithmeticError::Resource(_))), "{out:?}");
        assert!(el < wall * 3 + Duration::from_millis(500), "{el:?}");
        // The legacy entry point reports Domain and leaves the trip recorded.
        let out = {
            let _g = budget::enter(budget::Budget::new().with_wall(wall));
            discrete_log(&r.to_string(), "4", p)
        };
        assert!(matches!(out, Err(NumberTheoryError::Domain { .. })));
        assert!(budget::take_trip().is_some());
    }
}
