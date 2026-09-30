//! Size pre-flight for polynomial powers.
//!
//! `p^n` is computed by FLINT (`fmpz_poly_pow`, `fmpz_mpoly_pow_ui`) or by a
//! schoolbook loop over GMP integers, and none of them can fail gracefully:
//! when the result does not fit, the allocation aborts the process or the OOM
//! killer takes it. The degree ceilings stop a power whose *degree* is absurd,
//! but a modest degree with a genuine binomial base is just as fatal —
//! `(x + 1)^(2^21)` has two million coefficients of up to two million bits
//! each, half a terabyte in all.
//!
//! So a power is sized before it is computed, from two cheap bounds on the
//! result:
//!
//! * **coefficient size** — every coefficient of `p^n` is at most `‖p‖₁ⁿ` in
//!   absolute value, so it has at most `n·log₂‖p‖₁` bits;
//! * **term count** — at most the number of monomials of degree `n` in
//!   `len(p)` unknowns, `C(n + len − 1, len − 1)`, and never more than the
//!   caller's dense bound (`deg·n + 1` for a univariate result).
//!
//! Their product is an upper bound on the result's size (within a small
//! factor for the binomial case), checked against the machine's physical
//! memory, the active `Budget(max_bytes=…)` and `RLIMIT_AS` by
//! [`crate::budget::preflight_bytes`]. A refusal is the polynomial layer's
//! existing size error, [`ConversionError::ExponentTooLarge`] (`E-POLY-004`).

use super::error::ConversionError;
use rug::Integer;

/// Results estimated below this many bytes skip the memory probes, which
/// cost a `/proc` read.
const PROBE_BYTES: f64 = (1u64 << 20) as f64;

/// `log₂ Σ|cᵢ|` — the bit size of the ℓ₁ norm of a coefficient list, as a
/// real number (`0` for an all-zero or empty list).
pub(crate) fn log2_l1_norm<'a>(coeffs: impl IntoIterator<Item = &'a Integer>) -> f64 {
    let mut l1 = Integer::new();
    for c in coeffs {
        if *c < 0 {
            l1 -= c;
        } else {
            l1 += c;
        }
    }
    log2_int(&l1)
}

/// `log₂ |v|` for a non-zero integer, `0` for zero.
fn log2_int(v: &Integer) -> f64 {
    if v.is_zero() {
        return 0.0;
    }
    let bits = v.significant_bits();
    if bits <= 1000 {
        v.to_f64().abs().log2()
    } else {
        // Keep the top 64 bits for the mantissa.
        let shift = bits - 64;
        let top = Integer::from(v.abs_ref()) >> shift;
        top.to_f64().log2() + f64::from(shift)
    }
}

/// `C(n + k − 1, k − 1)`: the number of monomials of degree `n` in `k`
/// unknowns, which bounds the number of terms of a `k`-term polynomial raised
/// to the `n`-th power. Saturates at `cap`.
pub(crate) fn power_term_bound(k: usize, n: u32, cap: f64) -> f64 {
    if k <= 1 {
        return 1.0_f64.min(cap);
    }
    let n = f64::from(n);
    let mut acc = 1.0_f64;
    for i in 1..k {
        acc = acc * (n + i as f64) / i as f64;
        if acc >= cap {
            return cap;
        }
    }
    acc
}

/// Refuse a power whose result would hold about `terms` coefficients of at
/// most `coeff_bits` bits each, plus `per_term` bytes of bookkeeping per
/// term, when that cannot fit the machine or the active memory budget.
pub(crate) fn check_power_size(
    terms: f64,
    coeff_bits: f64,
    per_term: u64,
) -> Result<(), ConversionError> {
    // A single coefficient past the GMP sanity ceiling is refused outright,
    // however few there are.
    if coeff_bits > crate::budget::MAX_INTEGER_BITS as f64 {
        return Err(ConversionError::ExponentTooLarge);
    }
    // A limb-rounded integer plus the per-term overhead.
    let per_coeff = (coeff_bits / 64.0).ceil() * 8.0 + per_term as f64;
    let bytes = terms.max(1.0) * per_coeff;
    if bytes < PROBE_BYTES {
        return Ok(());
    }
    // `as u64` saturates for anything past u64::MAX (and for infinity).
    crate::budget::preflight_bytes(bytes as u64).map_err(|_| ConversionError::ExponentTooLarge)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_norm_bits() {
        let cs = [Integer::from(1), Integer::from(-1), Integer::from(2)];
        assert!((log2_l1_norm(&cs) - 2.0).abs() < 1e-12);
        assert_eq!(log2_l1_norm(&[] as &[Integer]), 0.0);
        let big = Integer::from(1) << 5000;
        assert!((log2_l1_norm([&big]) - 5000.0).abs() < 1e-9);
    }

    #[test]
    fn term_bound_is_the_multinomial_count() {
        // (a + b)^n has n + 1 terms; (a + b + c)^2 has 6.
        assert_eq!(power_term_bound(2, 10, f64::INFINITY), 11.0);
        assert_eq!(power_term_bound(3, 2, f64::INFINITY), 6.0);
        assert_eq!(power_term_bound(1, 1 << 30, f64::INFINITY), 1.0);
        assert_eq!(power_term_bound(50, u32::MAX, 1e12), 1e12);
    }

    #[test]
    fn small_powers_pass_and_absurd_ones_do_not() {
        assert!(check_power_size(1000.0, 1000.0, 8).is_ok());
        // (x + 1)^(2^21): 2^21 coefficients of up to 2^21 bits — 512 GiB.
        let n = f64::from(1u32 << 21);
        assert_eq!(
            check_power_size(n + 1.0, n, 8),
            Err(ConversionError::ExponentTooLarge)
        );
        // A single coefficient past the GMP sanity ceiling.
        assert_eq!(
            check_power_size(1.0, 2.0 * crate::budget::MAX_INTEGER_BITS as f64, 8),
            Err(ConversionError::ExponentTooLarge)
        );
    }

    #[test]
    fn the_active_budget_is_honoured() {
        // 4 MiB of result: fine unbudgeted, refused under a 1 MiB budget.
        assert!(check_power_size(4096.0, 8192.0, 0).is_ok());
        let _g = crate::budget::enter_with_memory(crate::budget::Budget::default(), Some(1 << 20));
        assert_eq!(
            check_power_size(4096.0, 8192.0, 0),
            Err(ConversionError::ExponentTooLarge)
        );
    }
}
