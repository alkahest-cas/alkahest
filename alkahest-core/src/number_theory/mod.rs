//! V3-1 — Integer number theory via FLINT `fmpz`.
//!
//! Wraps proven primality, integer factorisation, totients, Jacobi symbols,
//! modular square roots (`fmpz_sqrtmod`), nth roots modulo primes when Coprime holds,
//! brute-force discrete logs, and quadratic Dirichlet characters (odd square-free conductor).

use crate::errors::AlkahestError;
use crate::flint::ffi::{self as ffi, FmpzFactorStruct};
use crate::flint::FlintInteger;
use rug::Complete;
use rug::Integer;
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// NumberTheoryError
// ---------------------------------------------------------------------------

/// Failed integer number-theory primitive (`E-NT-*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberTheoryError {
    InvalidInput { msg: &'static str },
    Domain { msg: &'static str },
    NoSolution,
    CompositeModulus,
    UnsupportedNthRoot,
}

impl fmt::Display for NumberTheoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumberTheoryError::InvalidInput { msg } => write!(f, "{msg}"),
            NumberTheoryError::Domain { msg } => write!(f, "{msg}"),
            NumberTheoryError::NoSolution => {
                write!(f, "no discrete logarithm or modular root exists")
            }
            NumberTheoryError::CompositeModulus => write!(f, "operation requires a prime modulus"),
            NumberTheoryError::UnsupportedNthRoot => {
                write!(f, "nth root modulo p requires k=2 or gcd(k,p−1)=1")
            }
        }
    }
}

impl std::error::Error for NumberTheoryError {}

impl AlkahestError for NumberTheoryError {
    fn code(&self) -> &'static str {
        match self {
            NumberTheoryError::InvalidInput { .. } => "E-NT-001",
            NumberTheoryError::Domain { .. } => "E-NT-002",
            NumberTheoryError::NoSolution => "E-NT-003",
            NumberTheoryError::CompositeModulus => "E-NT-004",
            NumberTheoryError::UnsupportedNthRoot => "E-NT-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            NumberTheoryError::InvalidInput { .. } => {
                Some("pass arbitrary-precision integers as decimal strings without spaces")
            }
            NumberTheoryError::Domain { .. } => {
                Some("check parity, positivity, and defined ranges")
            }
            NumberTheoryError::NoSolution => {
                Some("verify solvability: residue in ⟨base⟩, or quadratic residue for k=2")
            }
            NumberTheoryError::CompositeModulus => {
                Some("use a prime field modulus where the FLINT primitives apply")
            }
            NumberTheoryError::UnsupportedNthRoot => Some(
                "use sqrt (k=2) or primes with gcd(k,p−1)=1; Tonelli–Shanks chains are deferred",
            ),
        }
    }
}

fn parse_int(s: &str) -> Result<Integer, NumberTheoryError> {
    Integer::from_str(s.trim()).map_err(|_| NumberTheoryError::InvalidInput {
        msg: "invalid decimal integer string",
    })
}

fn parse_nonnegative(s: &str) -> Result<Integer, NumberTheoryError> {
    let z = parse_int(s)?;
    if z.cmp0() == Ordering::Less {
        Err(NumberTheoryError::Domain {
            msg: "expected a non-negative integer",
        })
    } else {
        Ok(z)
    }
}

/// Multiplicative inverse of `a` modulo `m` when `gcd(a,m)=1`.
fn mod_inverse(mut a: Integer, m: &Integer) -> Option<Integer> {
    if m.cmp0() != Ordering::Greater {
        return None;
    }
    if m == &Integer::from(1) {
        return Some(Integer::from(0));
    }
    a %= m;
    let (g, s, _) = a.extended_gcd(m.clone(), Integer::new());
    if g != Integer::from(1) && g != Integer::from(-1) {
        return None;
    }
    let mut inv = if g == Integer::from(-1) { -s } else { s };
    inv %= m;
    if inv.cmp0() == Ordering::Less {
        inv += m;
    }
    Some(inv)
}

fn integer_is_odd(n: &Integer) -> bool {
    (n.clone() % Integer::from(2_u32)).cmp0() != Ordering::Equal
}

/// Positive integer parser (strictly \(> 0\) when required by the caller).
fn parse_positive(s: &str) -> Result<Integer, NumberTheoryError> {
    let z = parse_nonnegative(s)?;
    if z.is_zero() {
        Err(NumberTheoryError::Domain {
            msg: "expected a positive integer",
        })
    } else {
        Ok(z)
    }
}

/// Exact primality (`fmpz_is_prime`).
pub fn isprime(n: &str) -> Result<bool, NumberTheoryError> {
    let z = parse_int(n)?;
    if z.cmp0() != Ordering::Greater || z < 2 {
        return Ok(false);
    }
    let fz = FlintInteger::from_rug(&z);
    let r = unsafe { ffi::fmpz_is_prime(fz.inner_ptr()) };
    Ok(r != 0)
}

/// Full factorisation: `(sign, list of (prime, exponent))` for \(\prod p^e\cdot \mathrm{sign}\).
pub fn factorint(n: &str) -> Result<(i32, Vec<(String, u64)>), NumberTheoryError> {
    let z = parse_int(n)?;
    let fz = FlintInteger::from_rug(&z);
    unsafe {
        let mut fac = std::mem::MaybeUninit::<FmpzFactorStruct>::uninit();
        ffi::fmpz_factor_init(fac.as_mut_ptr());
        let mut fac = fac.assume_init();
        ffi::fmpz_factor(&mut fac, fz.inner_ptr());
        let mut out = Vec::with_capacity(fac.num.max(0) as usize);
        for i in 0..fac.num {
            let mut base = FlintInteger::new();
            ffi::fmpz_set(base.inner_mut_ptr(), fac.p.add(i as usize));
            let exp = *fac.exp.add(i as usize);
            out.push((base.to_string(), exp));
        }
        let sign = fac.sign;
        ffi::fmpz_factor_clear(&mut fac);
        Ok((sign, out))
    }
}

/// Next prime strictly after `n` (`fmpz_nextprime`).
pub fn nextprime(n: &str, proved: bool) -> Result<String, NumberTheoryError> {
    let z = parse_int(n)?;
    let fz = FlintInteger::from_rug(&z);
    let mut res = FlintInteger::new();
    unsafe {
        ffi::fmpz_nextprime(
            res.inner_mut_ptr(),
            fz.inner_ptr(),
            if proved { 1 } else { 0 },
        );
    }
    Ok(res.to_string())
}

/// Euler totient \(\varphi(n)\) (`fmpz_euler_phi`).
pub fn totient(n: &str) -> Result<String, NumberTheoryError> {
    let z = parse_positive(n)?;
    let fz = FlintInteger::from_rug(&z);
    let mut out = FlintInteger::new();
    unsafe {
        ffi::fmpz_euler_phi(out.inner_mut_ptr(), fz.inner_ptr());
    }
    Ok(out.to_string())
}

/// Jacobi symbol \((a | n)\) for odd \(n > 1\) (`fmpz_jacobi`).
pub fn jacobi_symbol(a: &str, n: &str) -> Result<i32, NumberTheoryError> {
    let na = parse_int(a)?;
    let nn = parse_positive(n)?;
    if nn <= Integer::from(1) || !integer_is_odd(&nn) {
        return Err(NumberTheoryError::Domain {
            msg: "Jacobi denominator must be odd and greater than 1",
        });
    }
    let fa = FlintInteger::from_rug(&na);
    let fn_ = FlintInteger::from_rug(&nn);
    let j = unsafe { ffi::fmpz_jacobi(fa.inner_ptr(), fn_.inner_ptr()) };
    Ok(j as i32)
}

/// Modular \(k\)th root: some \(x\) with \(x^k \equiv a \pmod p\) for prime \(p\).
///
/// Implemented for `k == 2` via `fmpz_sqrtmod`, or when \(\gcd(k, p{-}1)=1\) via exponent inversion.
pub fn nthroot_mod(a: &str, k: u64, p: &str) -> Result<String, NumberTheoryError> {
    if k == 0 {
        return Err(NumberTheoryError::InvalidInput {
            msg: "root degree must be ≥ 1",
        });
    }
    let pm = parse_positive(p)?;
    let fp = FlintInteger::from_rug(&pm);
    if unsafe { ffi::fmpz_is_prime(fp.inner_ptr()) } == 0 {
        return Err(NumberTheoryError::CompositeModulus);
    }

    let mut ared = parse_int(a)?;
    ared %= &pm;

    let mut out = FlintInteger::new();

    if k == 2 {
        let fa = FlintInteger::from_rug(&ared);
        let ok = unsafe { ffi::fmpz_sqrtmod(out.inner_mut_ptr(), fa.inner_ptr(), fp.inner_ptr()) };
        if ok == 0 {
            return Err(NumberTheoryError::NoSolution);
        }
        return Ok(out.to_string());
    }

    let ord = (&pm).clone() - 1;
    let kk = Integer::from(k);
    if kk.clone().gcd(&ord) != Integer::from(1) {
        return Err(NumberTheoryError::UnsupportedNthRoot);
    }
    let mut inv_e = mod_inverse(kk.clone(), &ord).ok_or(NumberTheoryError::UnsupportedNthRoot)?;
    inv_e %= &ord;
    let fa = FlintInteger::from_rug(&ared);
    let fe = FlintInteger::from_rug(&inv_e);
    unsafe {
        ffi::fmpz_powm(
            out.inner_mut_ptr(),
            fa.inner_ptr(),
            fe.inner_ptr(),
            fp.inner_ptr(),
        );
    }
    Ok(out.to_string())
}

/// Smallest exponent \(e \geq 0\) with \(\mathit{base}^e \equiv \mathit{residue}\pmod{p}\) (`p` prime).
///
/// This uses a deterministic linear sweep over exponents bounded by \(p{-}1\); it is tuned for API
/// parity and moderate primes, not large-field cryptography.
pub fn discrete_log(residue: &str, base: &str, p: &str) -> Result<String, NumberTheoryError> {
    let pm = parse_positive(p)?;
    if pm < Integer::from(2) {
        return Err(NumberTheoryError::Domain {
            msg: "modulus must be at least 2",
        });
    }
    let fp = FlintInteger::from_rug(&pm);
    if unsafe { ffi::fmpz_is_prime(fp.inner_ptr()) } == 0 {
        return Err(NumberTheoryError::CompositeModulus);
    }

    let ord = (&pm).clone() - Integer::from(1);
    let mut b = parse_int(base)?;
    let mut r = parse_int(residue)?;
    r %= &pm;
    b %= &pm;

    if b.is_zero() {
        return if r.is_zero() {
            Ok("1".into())
        } else {
            Err(NumberTheoryError::NoSolution)
        };
    }

    let mut cur = Integer::from(1);
    let mut exp = Integer::from(0);
    while exp < ord {
        if cur == r {
            return Ok(exp.to_string());
        }
        cur = (&cur * &b).complete();
        cur %= &pm;
        exp += 1;
    }
    Err(NumberTheoryError::NoSolution)
}

/// Quadratic Dirichlet character: Jacobi symbol \((· | q)\) for odd square-free \(q≥3\).
#[derive(Clone, Debug)]
pub struct QuadraticDirichlet {
    modulus: Integer,
}

impl QuadraticDirichlet {
    pub fn new(conductor: &str) -> Result<Self, NumberTheoryError> {
        let q = parse_positive(conductor)?;
        if q <= Integer::from(2) || !integer_is_odd(&q) {
            return Err(NumberTheoryError::Domain {
                msg: "quadratic Dirichlet conductor must be odd and ≥ 3",
            });
        }
        let (_sign, fac) = factorint(conductor)?;
        for (_, e) in &fac {
            if *e != 1 {
                return Err(NumberTheoryError::Domain {
                    msg: "conductor must be square-free",
                });
            }
        }
        Ok(QuadraticDirichlet { modulus: q })
    }

    pub fn conductor(&self) -> String {
        self.modulus.to_string()
    }

    /// \(\chi_q(n)\) as `−1`, `0`, or `1`.
    pub fn eval(&self, n: &str) -> Result<i32, NumberTheoryError> {
        jacobi_symbol(n, &self.modulus.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;
    use std::collections::HashMap;

    #[test]
    fn mersenne_m127_prime() {
        let m = Integer::from(2u32).pow(127_u32) - 1_u32;
        assert!(isprime(&m.to_string()).unwrap());
    }

    #[test]
    fn factorint_f5() {
        let n = &(1u128 << 32) - 1;
        let (sign, pairs) = factorint(&n.to_string()).unwrap();
        assert_eq!(sign, 1);
        let m: HashMap<_, _> = pairs.into_iter().collect();
        assert_eq!(m.get("65537").copied(), Some(1));
    }

    #[test]
    fn nextprime_gap() {
        assert_eq!(nextprime("13", true).unwrap(), "17");
    }

    #[test]
    fn totient_twelve() {
        assert_eq!(totient("12").unwrap(), "4");
    }

    #[test]
    fn jacobi_two_fifteen() {
        assert_eq!(jacobi_symbol("2", "15").unwrap(), 1);
    }

    #[test]
    fn sqrt_mod_prime() {
        let x_str = nthroot_mod("144", 2, "401").unwrap();
        let x: u64 = x_str.parse().unwrap();
        assert_eq!((x * x) % 401, 144);
    }

    #[test]
    fn nth_root_via_coprime_exponent() {
        let pm = Integer::from(10007);
        let a = Integer::from(42);
        let k = 5u64;
        let kk = Integer::from(k);
        let ord = pm.clone() - Integer::from(1);
        assert_eq!(kk.clone().gcd(&ord), Integer::from(1));

        let x_str = nthroot_mod(&a.to_string(), k, &pm.to_string()).unwrap();
        let x = Integer::from_str(&x_str).unwrap();
        let chk = x.clone().pow(k as u32) % &pm;
        assert_eq!(chk, a % &pm);
    }

    #[test]
    fn discrete_log_three_mod_seventeen() {
        assert_eq!(discrete_log("13", "3", "17").unwrap(), "4",);
    }

    #[test]
    fn dirichlet_phi_fifteen() {
        let chi = QuadraticDirichlet::new("15").unwrap();
        assert_eq!(chi.eval("14").unwrap(), -1);
        assert_eq!(chi.eval("3").unwrap(), 0);
    }
}
