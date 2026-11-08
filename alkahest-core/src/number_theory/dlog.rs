//! Discrete logarithms modulo any `m ≥ 2`: Pohlig–Hellman over the
//! factorisation of the order of the base, with baby-step giant-step or
//! Pollard's rho in each prime-order subgroup; a composite modulus is split
//! into prime powers and recombined by the CRT ([`discrete_log_composite`]).
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
use rug::ops::{Pow, RemRounding};
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
    let fac = factor_list(&group_order)?;
    let (x, _) = dlog_unit(r, b, p, &group_order, fac, true)?;
    Ok(x)
}

/// `n`'s factorisation as `(prime, exponent)` pairs, through the budgeted
/// factoriser.
fn factor_list(n: &Integer) -> Result<Vec<(Integer, u32)>, ArithmeticError> {
    if *n <= 1 {
        return Ok(Vec::new());
    }
    let (_, fac) = super::factorint_bounded(&n.to_string())?;
    fac.into_iter()
        .map(|(q, e)| {
            Integer::from_str(&q)
                .map(|q| (q, e as u32))
                .map_err(|_| internal("factorint returned a non-integer"))
        })
        .collect()
}

/// `(x, ord(b))` with `x` the smallest exponent such that `b^x ≡ r (mod m)`,
/// for units `b`, `r` of a unit group `(ℤ/mℤ)^×` of order `group_order`
/// (factorised as `fac`).
///
/// `cyclic` says whether the group is cyclic (`m` a prime, an odd prime power,
/// 2 or 4): then `r ∈ ⟨b⟩ ⟺ r^ord(b) = 1` decides membership up front and a
/// later miss is an internal failure. In the non-cyclic `(ℤ/2^kℤ)^×`
/// (`k ≥ 3`) every prime of the order is 2, so each subgroup step is an
/// exhaustive two-element search: a miss there, or a failed final check, is a
/// proof that `r ∉ ⟨b⟩` and comes back as [`NumberTheoryError::NoSolution`].
fn dlog_unit(
    r: &Integer,
    b: &Integer,
    m: &Integer,
    group_order: &Integer,
    fac: Vec<(Integer, u32)>,
    cyclic: bool,
) -> Result<(Integer, Integer), ArithmeticError> {
    // ord(b): strip every prime from the group order that b's order does not need.
    let one_mod_m = Integer::from(1) % m;
    let mut n = group_order.clone();
    let mut order_fac: Vec<(Integer, u32)> = Vec::new();
    for (q, e) in fac {
        let mut kept = e;
        while kept > 0 {
            let cofactor = Integer::from(&n / &q);
            if powm(b, &cofactor, m) == one_mod_m {
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
    if *r == one_mod_m {
        return Ok((Integer::new(), n));
    }

    // Necessary in every group; sufficient in a cyclic one.
    if powm(r, &n, m) != one_mod_m {
        return Err(NumberTheoryError::NoSolution.into());
    }

    let b_inv = Integer::from(
        b.invert_ref(m)
            .ok_or_else(|| internal("discrete_log: the base is not a unit"))?,
    );

    // Pohlig–Hellman: x mod q^e for each prime power, then CRT.
    let mut x_all = Integer::new();
    let mut modulus = Integer::from(1);
    for (q, e) in &order_fac {
        let g0 = powm(b, &Integer::from(&n / q), m); // order exactly q
        let mut x = Integer::new();
        let mut qk = Integer::from(1);
        for _ in 0..*e {
            // h = (r · b^{−x})^{n / q^{k+1}} lies in ⟨g0⟩ when r ∈ ⟨b⟩.
            let mut h = powm(&b_inv, &x, m);
            mulm(&mut h, r, m);
            let qk1 = Integer::from(&qk * q);
            let h = powm(&h, &Integer::from(&n / &qk1), m);
            let d = match dlog_prime_order(&g0, &h, q, m) {
                Ok(d) => d,
                // q = 2 is solved exhaustively, so a miss is a proof.
                Err(_) if !cyclic && *q == 2 => {
                    return Err(NumberTheoryError::NoSolution.into());
                }
                Err(e) => return Err(e),
            };
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

    if powm(b, &x_all, m) != *r {
        if !cyclic {
            return Err(NumberTheoryError::NoSolution.into());
        }
        return Err(internal(
            "discrete_log: the Pohlig–Hellman exponent failed its final check",
        ));
    }
    Ok((x_all, n))
}

/// `v_p(b)` for `b ≠ 0`.
fn valuation(b: &Integer, p: &Integer) -> u32 {
    let mut v = 0u32;
    let mut bb = b.clone();
    while bb.is_divisible(p) {
        bb /= p;
        v += 1;
    }
    v
}

/// Smallest `e ≥ 0` with `b^e ≡ r (mod m)` for a composite modulus `m`, with
/// `b` and `r` already reduced into `[0, m)`. Neither needs to be a unit.
///
/// Factor `m = ∏ pᵢ^kᵢ`. For a prime `p | b`, `b^e ≡ 0 (mod p^k)` as soon as
/// `e · v_p(b) ≥ k`; for a prime `p ∤ b`, `b` is a unit modulo `p^k` and the
/// exponents that work there form one residue class modulo `ord_{p^k}(b)`
/// (Pohlig–Hellman in `(ℤ/p^kℤ)^×`, cyclic of order `p^{k−1}(p − 1)` for odd
/// `p`). So:
///
/// 1. every `e < T`, where `T = max ⌈k / v_p(b)⌉ ≤ log₂ m` over the primes
///    dividing `b`, is tried directly — this is where a non-unit base reaches
///    residues that no large power does (`2¹ ≡ 2 (mod 12)`);
/// 2. for `e ≥ T` the condition is `r ≡ 0` modulo every `p^k` with `p | b`,
///    and `e ≡ xᵢ (mod nᵢ)` for every unit prime power; the classes are merged
///    by the CRT for non-coprime moduli (an incompatible pair proves there is
///    no solution), and the smallest `e ≥ T` in the merged class is returned.
///
/// The answer is checked by one modular exponentiation before it is returned.
pub(super) fn discrete_log_composite(
    r: &Integer,
    b: &Integer,
    m: &Integer,
) -> Result<Integer, ArithmeticError> {
    let fac = factor_list(m)?;

    // T: from it on, every prime dividing b has driven its prime power to 0.
    let mut t_sat: u32 = 0;
    for (p, k) in &fac {
        if b.is_divisible(p) {
            let need = if b.is_zero() {
                1
            } else {
                k.div_ceil(valuation(b, p))
            };
            t_sat = t_sat.max(need);
        }
    }

    // Step 1: small exponents, directly modulo m.
    let mut cur = Integer::from(1) % m;
    for e in 0..t_sat {
        if cur == *r {
            return Ok(Integer::from(e));
        }
        mulm(&mut cur, b, m);
    }
    budget::check_all()?;

    // Step 2: e ≥ T, one residue class per unit prime power.
    let mut x_all = Integer::new();
    let mut l_all = Integer::from(1);
    for (p, k) in &fac {
        let pk = Integer::from(Pow::pow(p, *k));
        let rk = Integer::from(r % &pk);
        if b.is_divisible(p) {
            if !rk.is_zero() {
                return Err(NumberTheoryError::NoSolution.into());
            }
            continue;
        }
        if Integer::from(rk.gcd_ref(&pk)) != 1 {
            // A unit never powers to a non-unit.
            return Err(NumberTheoryError::NoSolution.into());
        }
        let bk = Integer::from(b % &pk);
        let (group_order, gfac, cyclic) = if *p == 2 {
            let gfac = if *k >= 2 {
                vec![(Integer::from(2), k - 1)]
            } else {
                Vec::new()
            };
            (Integer::from(1) << (k - 1), gfac, *k <= 2)
        } else {
            let pm1 = Integer::from(p - 1u32);
            let mut gfac = factor_list(&pm1)?;
            if *k >= 2 {
                gfac.push((p.clone(), k - 1));
            }
            (Integer::from(Pow::pow(p, k - 1)) * &pm1, gfac, true)
        };
        let (x, n) = dlog_unit(&rk, &bk, &pk, &group_order, gfac, cyclic)?;
        // Merge e ≡ x_all (mod l_all) with e ≡ x (mod n).
        let g = Integer::from(l_all.gcd_ref(&n));
        let diff = Integer::from(&x - &x_all);
        if !diff.is_divisible(&g) {
            return Err(NumberTheoryError::NoSolution.into());
        }
        let l_g = Integer::from(&l_all / &g);
        let n_g = Integer::from(&n / &g);
        let inv = if n_g == 1 {
            Integer::new()
        } else {
            Integer::from(
                l_g.invert_ref(&n_g)
                    .ok_or_else(|| internal("discrete_log: CRT moduli were not reduced"))?,
            )
        };
        let tt = (Integer::from(&diff / &g) * inv).rem_euc(&n_g);
        x_all += Integer::from(&l_all * &tt);
        l_all *= &n_g;
        x_all = x_all.rem_euc(&l_all);
        budget::check_all()?;
    }

    // Smallest e ≥ T with e ≡ x_all (mod l_all).
    let t = Integer::from(t_sat);
    let mut e = x_all;
    if e < t {
        let steps = (Integer::from(&t - &e) + &l_all - 1u32) / &l_all;
        e += steps * &l_all;
    }
    if powm(b, &e, m) != *r {
        return Err(internal(
            "discrete_log: the composite-modulus exponent failed its final check",
        ));
    }
    Ok(e)
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
        match table.entry(k) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(j as u32);
            }
            // With exact keys a repeat would mean g^j = g^i for i < j < q,
            // impossible in a group of order q; with truncated keys it is a
            // collision of distinct residues and both must be kept.
            std::collections::hash_map::Entry::Occupied(_) => {
                if !exact_keys {
                    overflow.push((k, j as u32));
                }
            }
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
                return Ok(Some(e % Integer::from(q)));
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

    /// First exponent at which each residue appears in `1, b, b², …` mod `m`
    /// (the sequence is eventually periodic with preperiod + period ≤ m).
    fn first_hits(b: u64, m: u64) -> Vec<Option<u64>> {
        let mut first = vec![None; m as usize];
        let mut cur = 1 % m;
        for e in 0..=m {
            if first[cur as usize].is_none() {
                first[cur as usize] = Some(e);
            }
            cur = cur * b % m;
        }
        first
    }

    fn check_against_brute(r: u64, b: u64, m: u64, want: Option<u64>) {
        let got = discrete_log(&r.to_string(), &b.to_string(), &m.to_string());
        match want {
            Some(e) => assert_eq!(got.unwrap(), e.to_string(), "m={m} b={b} r={r}"),
            None => assert!(
                matches!(got, Err(NumberTheoryError::NoSolution)),
                "m={m} b={b} r={r}: {got:?}"
            ),
        }
    }

    /// Every (residue, base) pair for every modulus up to 130 — composite
    /// moduli, non-unit bases and residues, powers of two (non-cyclic unit
    /// groups) — against the definition: the smallest exponent, and
    /// NoSolution exactly when there is none.
    #[test]
    fn composite_moduli_exhaustive_up_to_130() {
        for m in 2u64..=130 {
            for b in 0..m {
                let first = first_hits(b, m);
                for r in 0..m {
                    check_against_brute(r, b, m, first[r as usize]);
                }
            }
        }
    }

    /// Every modulus up to 2000 with a spread of bases (units, zero divisors,
    /// prime-power multiples) and residues (reachable and not).
    #[test]
    fn composite_moduli_sampled_up_to_2000() {
        let mut rng = SplitMix64(0xD106_C0DE);
        for m in 131u64..=2000 {
            let mut bases = vec![2, 3, 6, 10, 12, m - 1, m / 2, (m / 3).max(2)];
            bases.extend((0..4).map(|_| rng.next() % m));
            for b in bases {
                let b = b % m;
                let first = first_hits(b, m);
                let mut rs: Vec<u64> = (0..m).filter(|&r| first[r as usize].is_some()).collect();
                // Every reachable residue when there are few, a sample otherwise.
                if rs.len() > 24 {
                    let len = rs.len() as u64;
                    rs = (0..24).map(|_| rs[(rng.next() % len) as usize]).collect();
                }
                rs.extend((0..8).map(|_| rng.next() % m));
                rs.extend([0, 1]);
                for r in rs {
                    check_against_brute(r, b, m, first[r as usize]);
                }
            }
        }
    }

    /// The reported repro and some structured composite cases.
    #[test]
    fn composite_modulus_examples() {
        // 13 generates the index-2 subgroup of (ℤ/100)^× (20 elements); 3 is
        // outside it, so `13^e ≡ 3 (mod 100)` has no solution.
        assert!(matches!(
            discrete_log("3", "13", "100"),
            Err(NumberTheoryError::NoSolution)
        ));
        assert_eq!(discrete_log("17", "13", "100").unwrap(), "7");
        // Non-unit base: 2^1 = 2 (mod 12) is reached once, never again.
        assert_eq!(discrete_log("2", "2", "12").unwrap(), "1");
        assert_eq!(discrete_log("4", "2", "12").unwrap(), "2");
        assert_eq!(discrete_log("8", "2", "12").unwrap(), "3");
        assert!(matches!(
            discrete_log("6", "2", "12"),
            Err(NumberTheoryError::NoSolution)
        ));
        // (ℤ/2^k)^× is not cyclic: 3 and 5 generate different subgroups mod 64.
        assert!(matches!(
            discrete_log("5", "3", "64"),
            Err(NumberTheoryError::NoSolution)
        ));
        assert_eq!(discrete_log("9", "3", "64").unwrap(), "2");
    }

    /// Large composite moduli with smooth unit-group orders, checked by
    /// exponentiation (the answer must also be the minimal one modulo the
    /// order of the base, which `b^e ≠ r` for every smaller `e` would need a
    /// sweep to confirm; minimality is covered by the exhaustive tests).
    #[test]
    fn large_composite_moduli() {
        let p1 = z("2147483647");
        let p2 = z("1000000000039");
        let cases = [
            (Integer::from(&p1 * &p2), "7", "123456789012345"),
            (Integer::from(&p1 * &p1) * 12u32, "5", "98765432101234"),
            (Integer::from(1) << 100u32, "3", "123456789123456789"),
        ];
        for (m, b, e) in cases {
            let r = powm(&z(b), &z(e), &m);
            let x = z(&discrete_log(&r.to_string(), b, &m.to_string()).unwrap());
            assert_eq!(powm(&z(b), &x, &m), r, "m={m}");
            assert!(x <= z(e), "m={m}: {x} is not minimal");
        }
    }

    /// A composite modulus whose unit group has a 61-bit prime order factor
    /// still honours the budget (#518): a typed trip, never a partial value.
    #[test]
    fn budget_trips_on_a_hard_composite_modulus() {
        let p = z("4611686018427394499");
        let m = Integer::from(&p * 15u32);
        let r = powm(&z("4"), &z("987654321987654321"), &m);
        let wall = Duration::from_millis(300);
        let t = Instant::now();
        let out = {
            let _g = budget::enter(budget::Budget::new().with_wall(wall));
            discrete_log_bounded(&r.to_string(), "4", &m.to_string())
        };
        assert!(matches!(out, Err(ArithmeticError::Resource(_))), "{out:?}");
        assert!(t.elapsed() < wall * 3 + Duration::from_millis(500));
    }
}
