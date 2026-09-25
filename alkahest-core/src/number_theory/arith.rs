//! The classical arithmetic functions, on FLINT's `arith`, `bernoulli` and
//! `partitions`.
//!
//! Everything here is FLINT's implementation, not a reimplementation: the
//! partition function is the Hardy–Ramanujan–Rademacher formula with rigorous
//! error bounds, the Bernoulli numbers come from `arith_bernoulli_number`, and
//! the Stirling and Euler numbers from `arith_stirling_*` / `arith_euler_number`.
//! Each is pinned against published values in this module's tests.
//!
//! # The Bernoulli convention
//!
//! **FLINT uses `B₁ = −1/2`** — the "first Bernoulli numbers" of DLMF,
//! Abramowitz & Stegun, Mathematica's `BernoulliB` and SymPy's `bernoulli`,
//! from the generating function `t/(eᵗ − 1)`. The other convention, from
//! `t·eᵗ/(eᵗ − 1)`, gives `B₁ = +1/2`. The two differ in **exactly that one
//! value** and nowhere else, which is what makes choosing the wrong one so
//! quiet a bug. See `bernoulli_number()`.
//!
//! # Why the refusals are not [`NumberTheoryError`]
//!
//! [`NumberTheoryError`] is an exhaustive public enum re-exported from
//! `alkahest_cas::stable`, so adding a variant to it is a semver-major change:
//! `cargo semver-checks` reports `enum_variant_added` and demands a new major
//! version. Forcing a major bump for a new experimental surface is the wrong
//! trade, so the work-cap refusal lives on `ArithmeticError` instead.
//!
//! That enum is **`#[non_exhaustive]` from birth**, which is the point: every
//! future arithmetic-function refusal can be added without another break. The
//! two share the `E-NT-NNN` code space, and
//! `ArithmeticError::Input` *delegates* its code, message and remediation to
//! the [`NumberTheoryError`] it wraps — so a domain violation reaching a caller
//! of `moebius_mu()` still reads `E-NT-002` rather than acquiring a second
//! identity. The two can be consolidated at a deliberate major bump. In Python
//! the split is invisible: `ArithmeticError` is created as a **subclass** of
//! `NumberTheoryError`, so one `except NumberTheoryError` still catches
//! everything.

use std::fmt;

use rug::Rational;

use super::{parse_nonnegative, parse_positive, NumberTheoryError};
use crate::errors::AlkahestError;
use crate::flint::ffi;
use crate::flint::rational::FlintRational;
use crate::flint::FlintInteger;

// ---------------------------------------------------------------------------
// ArithmeticError
// ---------------------------------------------------------------------------

/// Refusals from the classical arithmetic functions.
///
/// `#[non_exhaustive]`: match with a `_` arm. New variants will be added.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArithmeticError {
    /// The argument is past this module's work cap for that function.
    ///
    /// Not a statement that the value does not exist — `p(10⁹)` is a perfectly
    /// good integer. It is a refusal to disappear into FLINT for an unbounded
    /// time on a call that looked cheap. `E-NT-006`.
    WorkLimitExceeded {
        /// Which function refused, e.g. `"partition_number"`.
        function: &'static str,
        /// The cap that was exceeded.
        limit: u64,
    },
    /// The argument is outside the function's domain, or is not a decimal
    /// integer at all. Carries `E-NT-001` / `E-NT-002` unchanged.
    ///
    /// Present because these functions parse arbitrary-precision arguments with
    /// the same helpers the rest of [`crate::number_theory`] uses: `μ(0)` is
    /// undefined for the same reason and with the same remediation whether it
    /// is reached through `moebius_mu()` or through anything else, and
    /// re-coding it here would have given one condition two codes.
    Input(NumberTheoryError),
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithmeticError::WorkLimitExceeded { function, limit } => write!(
                f,
                "{function} refuses arguments above {limit} here; the value exists, \
                 this module just will not spend unbounded time computing it"
            ),
            ArithmeticError::Input(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ArithmeticError {}

impl From<NumberTheoryError> for ArithmeticError {
    fn from(e: NumberTheoryError) -> Self {
        ArithmeticError::Input(e)
    }
}

impl AlkahestError for ArithmeticError {
    fn code(&self) -> &'static str {
        match self {
            ArithmeticError::WorkLimitExceeded { .. } => "E-NT-006",
            ArithmeticError::Input(e) => e.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            ArithmeticError::WorkLimitExceeded { .. } => {
                Some("reduce the argument, or call FLINT directly if the wait is acceptable")
            }
            ArithmeticError::Input(e) => e.remediation(),
        }
    }
}

// ---------------------------------------------------------------------------
// Work caps
// ---------------------------------------------------------------------------

/// Largest `n` accepted by [`partition_number`].
pub const MAX_PARTITION_N: u64 = 1_000_000;
/// Largest `n` accepted by `bernoulli_number()`.
pub const MAX_BERNOULLI_N: u64 = 50_000;
/// Largest `n` accepted by [`euler_number`].
pub const MAX_EULER_N: u64 = 20_000;
/// Largest `n` accepted by [`harmonic_number`].
pub const MAX_HARMONIC_N: u64 = 1_000_000;
/// Largest `n` accepted by the Stirling-number entry points.
pub const MAX_STIRLING_N: u64 = 10_000;

fn work_cap(function: &'static str, n: u64, limit: u64) -> Result<(), ArithmeticError> {
    if n > limit {
        Err(ArithmeticError::WorkLimitExceeded { function, limit })
    } else {
        Ok(())
    }
}

/// The partition function \(p(n)\) — the number of ways of writing \(n\) as an
/// unordered sum of positive integers. Exact and arbitrary-precision.
///
/// `p(0) = 1` by the empty partition. `p(100) = 190569292`.
///
/// # Errors
///
/// `E-NT-006` for `n` above [`MAX_PARTITION_N`].
pub fn partition_number(n: u64) -> Result<String, ArithmeticError> {
    work_cap("partition_number", n, MAX_PARTITION_N)?;
    let mut out = FlintInteger::new();
    // SAFETY: `out` is a live `fmpz`; `arith_number_of_partitions` takes a
    // machine word and writes the result.
    unsafe { ffi::arith_number_of_partitions(out.inner_mut_ptr(), n) };
    Ok(out.to_string())
}

/// The Bernoulli number \(B_n\), exact.
///
/// # Which convention
///
/// **FLINT uses \(B_1 = -1/2\)** — the "first Bernoulli numbers", the
/// convention of DLMF, Abramowitz & Stegun, Mathematica's `BernoulliB` and
/// SymPy's `bernoulli`. The generating function is
/// \(\frac{t}{e^{t}-1} = \sum B_n t^n / n!\), **not** \(\frac{t e^{t}}{e^{t}-1}\),
/// which would give \(B_1 = +1/2\) and is what Knuth's `B_n^+`, Concrete
/// Mathematics' later editions, and some combinatorics texts use.
///
/// This matters and is a classic source of silent disagreement, because the two
/// conventions differ in **exactly one value**: \(B_1\). Every other Bernoulli
/// number is identical (all odd-index ones past 1 are zero), so a program that
/// picks the wrong convention agrees with the right one everywhere except the
/// single place it does not — and produces, for instance, an Euler–Maclaurin
/// correction term of the wrong sign with nothing else looking amiss. This
/// function returns `-1/2`; if the caller wants `B_1^+`, negate that one value.
///
/// `B_0 = 1`, `B_2 = 1/6`, `B_4 = -1/30`, `B_12 = -691/2730`.
///
/// # Errors
///
/// `E-NT-006` for `n` above [`MAX_BERNOULLI_N`].
pub fn bernoulli_number(n: u64) -> Result<Rational, ArithmeticError> {
    work_cap("bernoulli_number", n, MAX_BERNOULLI_N)?;
    let mut out = FlintRational::new();
    // SAFETY: `out` is a live, initialised `fmpq`.
    unsafe { ffi::arith_bernoulli_number(out.as_mut_ptr(), n) };
    Ok(out.to_rug())
}

/// The Euler number \(E_n\) (the secant numbers): `1, 0, -1, 0, 5, 0, -61, …`.
///
/// These are the coefficients of \(\operatorname{sech} t = \sum E_n t^n/n!\).
/// Odd-index Euler numbers are zero. They are **not** the Eulerian numbers, and
/// not the Euler polynomials evaluated anywhere.
///
/// # Errors
///
/// `E-NT-006` for `n` above [`MAX_EULER_N`].
pub fn euler_number(n: u64) -> Result<String, ArithmeticError> {
    work_cap("euler_number", n, MAX_EULER_N)?;
    let mut out = FlintInteger::new();
    // SAFETY: `out` is a live `fmpz`.
    unsafe { ffi::arith_euler_number(out.inner_mut_ptr(), n) };
    Ok(out.to_string())
}

/// The harmonic number \(H_n = \sum_{k=1}^{n} 1/k\) as an exact rational.
///
/// `H_0 = 0`, `H_4 = 25/12`.
///
/// # Errors
///
/// `E-NT-006` for `n` above [`MAX_HARMONIC_N`].
pub fn harmonic_number(n: u64) -> Result<Rational, ArithmeticError> {
    work_cap("harmonic_number", n, MAX_HARMONIC_N)?;
    let mut out = FlintRational::new();
    // SAFETY: `out` is a live, initialised `fmpq`; `n` is within `i64` because
    // `MAX_HARMONIC_N` is.
    unsafe { ffi::arith_harmonic_number(out.as_mut_ptr(), n as ffi::slong) };
    Ok(out.to_rug())
}

/// The **signed** Stirling number of the first kind \(s(n, k)\).
///
/// These are the coefficients of the falling factorial:
/// \(x(x-1)\cdots(x-n+1) = \sum_k s(n,k)\,x^k\). `s(4, 2) = 11`,
/// `s(3, 2) = -3`. Use [`stirling_first_unsigned`] for the cycle counts
/// \(c(n,k) = |s(n,k)|\) — the two differ by \((-1)^{n-k}\), and mixing them up
/// is a sign error that hides on every input with `n - k` even.
///
/// `s(n, k) = 0` for `k > n`, and `s(0, 0) = 1`.
///
/// # Errors
///
/// `E-NT-006` for `n` above [`MAX_STIRLING_N`].
pub fn stirling_first(n: u64, k: u64) -> Result<String, ArithmeticError> {
    stirling(ffi::arith_stirling_number_1, "stirling_first", n, k)
}

/// The **unsigned** Stirling number of the first kind \(c(n, k) = |s(n,k)|\) —
/// the number of permutations of `n` elements with exactly `k` cycles.
///
/// # Errors
///
/// `E-NT-006` for `n` above [`MAX_STIRLING_N`].
pub fn stirling_first_unsigned(n: u64, k: u64) -> Result<String, ArithmeticError> {
    stirling(
        ffi::arith_stirling_number_1u,
        "stirling_first_unsigned",
        n,
        k,
    )
}

/// The Stirling number of the second kind \(S(n, k)\) — the number of ways of
/// partitioning `n` labelled elements into `k` non-empty unlabelled blocks.
///
/// `S(n, 1) = S(n, n) = 1` for `n >= 1`, and `S(4, 2) = 7`.
///
/// # Errors
///
/// `E-NT-006` for `n` above [`MAX_STIRLING_N`].
pub fn stirling_second(n: u64, k: u64) -> Result<String, ArithmeticError> {
    stirling(ffi::arith_stirling_number_2, "stirling_second", n, k)
}

fn stirling(
    f: unsafe extern "C" fn(*mut ffi::fmpz, ffi::slong, ffi::slong),
    name: &'static str,
    n: u64,
    k: u64,
) -> Result<String, ArithmeticError> {
    work_cap(name, n, MAX_STIRLING_N)?;
    work_cap(name, k, MAX_STIRLING_N)?;
    let mut out = FlintInteger::new();
    // SAFETY: `out` is a live `fmpz`; both indices are non-negative and below
    // `MAX_STIRLING_N`, so they fit `slong`.
    unsafe { f(out.inner_mut_ptr(), n as ffi::slong, k as ffi::slong) };
    Ok(out.to_string())
}

/// The Möbius function \(\mu(n)\) for \(n \geq 1\): `1` if `n` is a square-free
/// product of an even number of primes, `-1` for an odd number, `0` if `n` has
/// a squared prime factor.
///
/// # Errors
///
/// `E-NT-002` for `n < 1` — \(\mu\) is defined on the positive integers, and
/// extending it by \(\mu(|n|)\) is a convention this module does not pick for
/// the caller.
pub fn moebius_mu(n: &str) -> Result<i32, ArithmeticError> {
    let z = parse_positive(n).map_err(ArithmeticError::Input)?;
    let fz = FlintInteger::from_rug(&z);
    // SAFETY: `fz` is a live `fmpz`; `fmpz_moebius_mu` is a pure read.
    Ok(unsafe { ffi::fmpz_moebius_mu(fz.inner_ptr()) } as i32)
}

/// The divisor-sum function \(\sigma_k(n) = \sum_{d \mid n} d^{k}\) for
/// \(n \geq 1\).
///
/// `k = 0` counts the divisors, `k = 1` sums them: \(\sigma_0(12) = 6\),
/// \(\sigma_1(12) = 28\).
///
/// # Errors
///
/// `E-NT-002` for `n < 1`.
pub fn divisor_sigma(k: u64, n: &str) -> Result<String, ArithmeticError> {
    let z = parse_positive(n).map_err(ArithmeticError::Input)?;
    let fz = FlintInteger::from_rug(&z);
    let mut out = FlintInteger::new();
    // SAFETY: `out` and `fz` are live `fmpz`. Note the argument order —
    // `fmpz_divisor_sigma` takes `(result, k, n)` in FLINT 3, with the exponent
    // *before* the argument; it was `(result, n, k)` in FLINT 2.
    unsafe { ffi::fmpz_divisor_sigma(out.inner_mut_ptr(), k, fz.inner_ptr()) };
    Ok(out.to_string())
}

/// \(r_k(n)\), the number of representations of \(n\) as an **ordered** sum of
/// \(k\) squares of integers, counting signs.
///
/// `r_2(5) = 8` — the eight points `(±1, ±2)` and `(±2, ±1)` — and
/// `r_4(1) = 8`. `r_k(0) = 1`.
///
/// # Errors
///
/// `E-NT-002` for `n < 0` or `k == 0`.
pub fn sum_of_squares(k: u64, n: &str) -> Result<String, ArithmeticError> {
    if k == 0 {
        return Err(ArithmeticError::Input(NumberTheoryError::Domain {
            msg: "the number of squares k must be at least 1",
        }));
    }
    let z = parse_nonnegative(n).map_err(ArithmeticError::Input)?;
    let fz = FlintInteger::from_rug(&z);
    let mut out = FlintInteger::new();
    // SAFETY: `out` and `fz` are live `fmpz`. Argument order `(result, k, n)`
    // as for `fmpz_divisor_sigma`.
    unsafe { ffi::arith_sum_of_squares(out.inner_mut_ptr(), k, fz.inner_ptr()) };
    Ok(out.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::number_theory::{isprime, nextprime};
    use rug::ops::Pow;
    use rug::Integer;
    use std::str::FromStr;

    /// The wrapped domain errors must keep their own codes rather than
    /// acquiring a new one — otherwise the split would have given one condition
    /// two identities.
    #[test]
    fn wrapped_input_errors_delegate_their_code() {
        let inner = moebius_mu_inner_error();
        assert_eq!(inner.code(), "E-NT-002");
        let wrapped: ArithmeticError = inner.clone().into();
        assert_eq!(wrapped.code(), inner.code());
        assert_eq!(wrapped.remediation(), inner.remediation());
        assert_eq!(wrapped.to_string(), inner.to_string());

        // Every entry point that parses its argument delegates the same way.
        for e in [
            moebius_mu("0").unwrap_err(),
            moebius_mu("-5").unwrap_err(),
            divisor_sigma(1, "0").unwrap_err(),
            sum_of_squares(2, "-1").unwrap_err(),
            sum_of_squares(0, "5").unwrap_err(),
        ] {
            assert_eq!(e.code(), "E-NT-002", "{e}");
            assert!(matches!(e, ArithmeticError::Input(_)));
        }
        assert_eq!(
            moebius_mu("not a number").unwrap_err().code(),
            "E-NT-001",
            "a parse failure keeps its own code too"
        );

        // The work cap is this enum's own, and the only code it introduces.
        let capped = partition_number(MAX_PARTITION_N + 1).unwrap_err();
        assert_eq!(capped.code(), "E-NT-006");
        assert!(matches!(capped, ArithmeticError::WorkLimitExceeded { .. }));
        assert!(capped.remediation().is_some());
    }

    /// The `NumberTheoryError` that `μ(0)` produces, fetched through the public
    /// surface so the test cannot drift from what the parser actually returns.
    fn moebius_mu_inner_error() -> NumberTheoryError {
        match moebius_mu("0").unwrap_err() {
            ArithmeticError::Input(e) => e,
            other => panic!("expected a wrapped input error, got {other:?}"),
        }
    }

    #[test]
    fn partition_function_anchors() {
        assert_eq!(partition_number(0).unwrap(), "1");
        assert_eq!(partition_number(1).unwrap(), "1");
        assert_eq!(partition_number(5).unwrap(), "7");
        assert_eq!(partition_number(100).unwrap(), "190569292");
        assert_eq!(
            partition_number(1000).unwrap(),
            "24061467864032622473692149727991"
        );
    }

    #[test]
    fn partition_function_refuses_past_the_cap() {
        let e = partition_number(MAX_PARTITION_N + 1).unwrap_err();
        assert_eq!(AlkahestError::code(&e), "E-NT-006");
    }

    /// FLINT's Bernoulli convention is `B_1 = -1/2`. This test is the pin: if
    /// a FLINT upgrade ever switched to `B_1 = +1/2`, every other value here
    /// would still pass and only this line would fail — which is precisely the
    /// failure mode the docs on `bernoulli_number` warn about.
    #[test]
    fn bernoulli_convention_is_b1_minus_one_half() {
        assert_eq!(bernoulli_number(1).unwrap(), Rational::from((-1, 2)));
    }

    #[test]
    fn bernoulli_anchors() {
        assert_eq!(bernoulli_number(0).unwrap(), Rational::from(1));
        assert_eq!(bernoulli_number(2).unwrap(), Rational::from((1, 6)));
        assert_eq!(bernoulli_number(4).unwrap(), Rational::from((-1, 30)));
        assert_eq!(bernoulli_number(6).unwrap(), Rational::from((1, 42)));
        assert_eq!(bernoulli_number(10).unwrap(), Rational::from((5, 66)));
        // The famous 691, the first irregular numerator.
        assert_eq!(bernoulli_number(12).unwrap(), Rational::from((-691, 2730)));
        for n in (3..=21).step_by(2) {
            assert_eq!(
                bernoulli_number(n).unwrap(),
                Rational::new(),
                "B_{n} should vanish"
            );
        }
    }

    /// Von Staudt–Clausen: the denominator of `B_{2n}` is the product of the
    /// primes `p` with `(p-1) | 2n`. An independent check of the whole value,
    /// not just of a table lookup.
    #[test]
    fn bernoulli_denominators_satisfy_von_staudt_clausen() {
        for n in 1..=20u64 {
            let two_n = 2 * n;
            let mut expected = Integer::from(1);
            let mut p = Integer::from(2);
            while p <= two_n + 1 {
                if isprime(&p.to_string()).unwrap()
                    && two_n % (p.clone() - 1u32).to_u64().unwrap() == 0
                {
                    expected *= &p;
                }
                p = Integer::from_str(&nextprime(&p.to_string(), true).unwrap()).unwrap();
            }
            assert_eq!(
                *bernoulli_number(two_n).unwrap().denom(),
                expected,
                "denominator of B_{two_n}"
            );
        }
    }

    #[test]
    fn euler_number_anchors() {
        assert_eq!(euler_number(0).unwrap(), "1");
        assert_eq!(euler_number(2).unwrap(), "-1");
        assert_eq!(euler_number(4).unwrap(), "5");
        assert_eq!(euler_number(6).unwrap(), "-61");
        assert_eq!(euler_number(8).unwrap(), "1385");
        assert_eq!(euler_number(10).unwrap(), "-50521");
        for n in (1..=11).step_by(2) {
            assert_eq!(euler_number(n).unwrap(), "0", "E_{n} should vanish");
        }
    }

    #[test]
    fn harmonic_number_anchors() {
        assert_eq!(harmonic_number(0).unwrap(), Rational::new());
        assert_eq!(harmonic_number(1).unwrap(), Rational::from(1));
        assert_eq!(harmonic_number(4).unwrap(), Rational::from((25, 12)));
        assert_eq!(harmonic_number(5).unwrap(), Rational::from((137, 60)));
        // H_n - H_{n-1} = 1/n, all the way up.
        for n in 1..=40u64 {
            let d = harmonic_number(n).unwrap() - harmonic_number(n - 1).unwrap();
            assert_eq!(d, Rational::from((1, n as i64)), "H_{n} step");
        }
    }

    #[test]
    fn stirling_anchors() {
        assert_eq!(stirling_first(4, 2).unwrap(), "11");
        assert_eq!(stirling_first_unsigned(4, 2).unwrap(), "11");
        // n - k odd, so the two kinds differ in sign here.
        assert_eq!(stirling_first(3, 2).unwrap(), "-3");
        assert_eq!(stirling_first_unsigned(3, 2).unwrap(), "3");
        assert_eq!(stirling_first(0, 0).unwrap(), "1");
        assert_eq!(stirling_first(5, 6).unwrap(), "0");

        assert_eq!(stirling_second(4, 2).unwrap(), "7");
        assert_eq!(stirling_second(5, 3).unwrap(), "25");
        for n in 1..=12u64 {
            assert_eq!(stirling_second(n, 1).unwrap(), "1", "S({n},1)");
            assert_eq!(stirling_second(n, n).unwrap(), "1", "S({n},{n})");
            assert_eq!(stirling_second(n, n + 1).unwrap(), "0");
        }
    }

    /// `sum_k S(n, k)` is the Bell number. An independent cross-check that the
    /// second-kind numbers are the ones FLINT says they are.
    #[test]
    fn stirling_second_sums_to_bell_numbers() {
        let bell = [
            "1", "1", "2", "5", "15", "52", "203", "877", "4140", "21147",
        ];
        for (n, want) in bell.iter().enumerate().skip(1) {
            let mut total = Integer::new();
            for k in 1..=n as u64 {
                total += Integer::from_str(&stirling_second(n as u64, k).unwrap()).unwrap();
            }
            assert_eq!(total.to_string(), *want, "Bell({n})");
        }
    }

    /// The falling factorial expanded by the signed first-kind numbers must
    /// reproduce `x(x-1)...(x-n+1)` at a concrete `x`.
    #[test]
    fn stirling_first_expands_the_falling_factorial() {
        for n in 1..=8u64 {
            for x in -3i64..=6 {
                let mut lhs = Integer::from(1);
                for i in 0..n as i64 {
                    lhs *= Integer::from(x - i);
                }
                let mut rhs = Integer::new();
                for k in 0..=n {
                    let s = Integer::from_str(&stirling_first(n, k).unwrap()).unwrap();
                    rhs += s * Integer::from(x).pow(k as u32);
                }
                assert_eq!(lhs, rhs, "falling factorial n={n}, x={x}");
            }
        }
    }

    #[test]
    fn moebius_anchors() {
        assert_eq!(moebius_mu("1").unwrap(), 1);
        assert_eq!(moebius_mu("2").unwrap(), -1);
        assert_eq!(moebius_mu("6").unwrap(), 1);
        assert_eq!(moebius_mu("12").unwrap(), 0);
        assert_eq!(moebius_mu("30").unwrap(), -1);
        assert!(moebius_mu("0").is_err());
        assert!(moebius_mu("-5").is_err());
        // sum_{d | n} mu(d) = [n == 1].
        for n in 1..=60u64 {
            let mut total = 0i32;
            for d in 1..=n {
                if n % d == 0 {
                    total += moebius_mu(&d.to_string()).unwrap();
                }
            }
            assert_eq!(total, i32::from(n == 1), "Mobius sum at {n}");
        }
    }

    #[test]
    fn divisor_sigma_anchors() {
        assert_eq!(divisor_sigma(0, "12").unwrap(), "6");
        assert_eq!(divisor_sigma(1, "12").unwrap(), "28");
        assert_eq!(divisor_sigma(2, "12").unwrap(), "210");
        assert_eq!(divisor_sigma(1, "6").unwrap(), "12", "6 is perfect");
        assert_eq!(divisor_sigma(1, "1").unwrap(), "1");
        assert!(divisor_sigma(1, "0").is_err());
        // Brute-force cross-check on a range.
        for n in 1..=50u64 {
            for k in 0..=3u32 {
                let want: u64 = (1..=n).filter(|d| n % d == 0).map(|d| d.pow(k)).sum();
                assert_eq!(
                    divisor_sigma(u64::from(k), &n.to_string()).unwrap(),
                    want.to_string(),
                    "sigma_{k}({n})"
                );
            }
        }
    }

    #[test]
    fn sum_of_squares_anchors() {
        assert_eq!(sum_of_squares(2, "0").unwrap(), "1");
        assert_eq!(sum_of_squares(2, "5").unwrap(), "8");
        assert_eq!(sum_of_squares(4, "1").unwrap(), "8");
        assert!(sum_of_squares(0, "5").is_err());
        assert!(sum_of_squares(2, "-1").is_err());
        // Jacobi: r_4(n) = 8 * sum of the divisors of n not divisible by 4.
        for n in 1..=40u64 {
            let want: u64 = 8 * (1..=n).filter(|d| n % d == 0 && d % 4 != 0).sum::<u64>();
            assert_eq!(
                sum_of_squares(4, &n.to_string()).unwrap(),
                want.to_string(),
                "r_4({n})"
            );
        }
        // Brute force r_2 on a small range.
        for n in 0..=25i64 {
            let mut count = 0u64;
            for a in -6i64..=6 {
                for b in -6i64..=6 {
                    if a * a + b * b == n {
                        count += 1;
                    }
                }
            }
            assert_eq!(
                sum_of_squares(2, &n.to_string()).unwrap(),
                count.to_string(),
                "r_2({n})"
            );
        }
    }

    #[test]
    fn work_caps_refuse_rather_than_hang() {
        for e in [
            bernoulli_number(MAX_BERNOULLI_N + 1).unwrap_err(),
            euler_number(MAX_EULER_N + 1).unwrap_err(),
            harmonic_number(MAX_HARMONIC_N + 1).unwrap_err(),
            stirling_first(MAX_STIRLING_N + 1, 2).unwrap_err(),
            stirling_second(2, MAX_STIRLING_N + 1).unwrap_err(),
        ] {
            assert_eq!(AlkahestError::code(&e), "E-NT-006");
            assert!(AlkahestError::remediation(&e).is_some());
            assert!(!e.to_string().is_empty());
        }
    }
}
