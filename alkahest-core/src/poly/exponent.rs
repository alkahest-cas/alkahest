//! Checked exponent arithmetic shared by the sparse polynomial types.
//!
//! Every sparse representation in [`crate::poly`] keys its terms by `u32`
//! exponents (`MultiPoly`, `GbPoly`, the univariate coefficient maps). Those
//! key types are part of the public API, so they stay `u32`; what must never
//! happen is that an exponent or a total degree which does not fit is
//! *wrapped* into one that does — `x^(2^31)·x^(2^31)` silently becoming `1`.
//!
//! The invariant the helpers here maintain: **every exponent and every
//! monomial's total degree fits in `u32`.** Conversions from expressions check
//! it and refuse with [`ConversionError::ExponentTooLarge`] (`E-POLY-004`);
//! arithmetic on values that satisfy it either stays inside it or reports the
//! overflow. Summing the exponents of a monomial that satisfies it can
//! therefore never overflow, which is what lets the monomial orders and
//! `total_degree` keep their `u32` results.

use super::error::ConversionError;

/// A non-negative integer exponent from an expression, as `u32`.
pub(crate) fn exponent_u32(n: &rug::Integer) -> Result<u32, ConversionError> {
    if *n < 0 {
        return Err(ConversionError::NegativeExponent);
    }
    n.to_u32().ok_or(ConversionError::ExponentTooLarge)
}

/// `a + b` for two exponents, or `ExponentTooLarge`.
#[inline]
pub(crate) fn add(a: u32, b: u32) -> Result<u32, ConversionError> {
    a.checked_add(b).ok_or(ConversionError::ExponentTooLarge)
}

/// `a + b` for two exponents, panicking (never wrapping) on overflow. For the
/// infallible arithmetic of public types whose operands came through a
/// checked conversion.
#[inline]
pub(crate) fn add_or_panic(a: u32, b: u32) -> u32 {
    a.checked_add(b).unwrap_or_else(|| overflow_panic())
}

/// Total degree of an exponent vector, or `None` if it exceeds `u32::MAX`.
#[inline]
pub(crate) fn total_degree(e: &[u32]) -> Option<u32> {
    u32::try_from(e.iter().map(|&x| u64::from(x)).sum::<u64>()).ok()
}

/// Total degree of an exponent vector that satisfies the module invariant.
///
/// # Panics
///
/// If the total degree exceeds `u32::MAX`, i.e. the vector was built without
/// going through a checked conversion. Panicking is the only honest option for
/// an infallible signature: the alternative is a wrapped, wrong degree.
#[inline]
pub(crate) fn total_degree_or_panic(e: &[u32]) -> u32 {
    total_degree(e).unwrap_or_else(|| overflow_panic())
}

/// Componentwise `a + b` over the common prefix of two exponent vectors
/// (longer vector's tail kept), or `ExponentTooLarge` if any component or the
/// total degree of the result exceeds `u32::MAX`.
pub(crate) fn add_vecs(a: &[u32], b: &[u32]) -> Result<Vec<u32>, ConversionError> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = long.to_vec();
    for (slot, &e) in out.iter_mut().zip(short) {
        *slot = add(*slot, e)?;
    }
    if total_degree(&out).is_none() {
        return Err(ConversionError::ExponentTooLarge);
    }
    Ok(out)
}

/// Componentwise `a + b` for two exponent vectors of equal length, panicking
/// (never wrapping) on overflow. For the infallible arithmetic of public types
/// whose operands came through a checked conversion.
#[inline]
pub(crate) fn add_vecs_or_panic(a: &[u32], b: &[u32]) -> Vec<u32> {
    add_vecs(a, b).unwrap_or_else(|_| overflow_panic())
}

#[cold]
#[inline(never)]
pub(crate) fn overflow_panic() -> ! {
    panic!(
        "polynomial exponent overflow: an exponent or total degree exceeds u32::MAX \
         (E-POLY-004); refusing to wrap it"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const B31: u32 = 1 << 31;

    #[test]
    fn exponent_u32_boundaries() {
        use rug::Integer;
        assert_eq!(exponent_u32(&Integer::from(u32::MAX)), Ok(u32::MAX));
        assert_eq!(exponent_u32(&Integer::from(B31)), Ok(B31));
        for big in [
            Integer::from(1u64 << 32),
            Integer::from(1u64 << 63),
            Integer::from(u64::MAX),
            Integer::from(u64::MAX) + 1u32,
        ] {
            assert_eq!(exponent_u32(&big), Err(ConversionError::ExponentTooLarge));
        }
        assert_eq!(
            exponent_u32(&Integer::from(-1)),
            Err(ConversionError::NegativeExponent)
        );
    }

    #[test]
    fn add_never_wraps() {
        assert_eq!(add(B31, B31), Err(ConversionError::ExponentTooLarge));
        assert_eq!(add(u32::MAX, 1), Err(ConversionError::ExponentTooLarge));
        assert_eq!(add(u32::MAX - 1, 1), Ok(u32::MAX));
    }

    #[test]
    fn total_degree_boundaries() {
        assert_eq!(total_degree(&[B31 - 1, B31]), Some(u32::MAX));
        assert_eq!(total_degree(&[B31, B31]), None);
        assert_eq!(total_degree(&[u32::MAX, u32::MAX, u32::MAX]), None);
        assert_eq!(total_degree(&[]), Some(0));
    }

    #[test]
    fn add_vecs_checks_components_and_total() {
        assert_eq!(add_vecs(&[1, 2], &[3]), Ok(vec![4, 2]));
        assert_eq!(
            add_vecs(&[B31], &[B31]),
            Err(ConversionError::ExponentTooLarge)
        );
        // Each component fits, the total does not.
        assert_eq!(
            add_vecs(&[B31], &[0, B31]),
            Err(ConversionError::ExponentTooLarge)
        );
    }

    #[test]
    #[should_panic(expected = "polynomial exponent overflow")]
    fn add_vecs_or_panic_panics_instead_of_wrapping() {
        let _ = add_vecs_or_panic(&[B31], &[B31]);
    }
}
