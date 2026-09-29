//! Safe wrapper over FLINT's `fmpq_t` — an exact rational number.
//!
//! Only two things happen to an `fmpq` in this crate: FLINT writes one (a
//! Bernoulli number, a harmonic number, a field norm) and we read it out as a
//! [`rug::Rational`]. Both directions copy the numerator and denominator limb
//! by limb (`integer::fmpz_set_rug` / `fmpz_get_rug`) through the mirrored
//! [`ffi::Fmpq`] layout, which `numfield::tests::fmpq_layout_matches_flint`
//! checks against FLINT's own printer at run time.

use rug::{Integer, Rational};

use super::ffi;
use super::integer::{fmpz_get_rug, fmpz_set_rug};

/// Owned `fmpq_t`. `Drop` calls `fmpq_clear`.
pub(crate) struct FlintRational {
    inner: ffi::Fmpq,
}

// SAFETY: an `fmpq` owns its two `fmpz`, which own their GMP memory. No
// shared or thread-local state.
unsafe impl Send for FlintRational {}
unsafe impl Sync for FlintRational {}

impl FlintRational {
    pub(crate) fn new() -> Self {
        let mut inner = ffi::Fmpq { num: 0, den: 0 };
        // SAFETY: `fmpq_init` writes both fields of a live `fmpq`.
        unsafe { ffi::fmpq_init(&mut inner) };
        Self { inner }
    }

    /// Build from a [`rug::Rational`] by copying the numerator's and
    /// denominator's limbs.
    ///
    /// A `rug::Rational` is always in lowest terms with a positive
    /// denominator, which is exactly FLINT's canonical form, so no
    /// `fmpq_canonicalise` (a gcd) is needed afterwards.
    pub(crate) fn from_rug(r: &Rational) -> Self {
        let mut q = Self::new();
        // SAFETY: `q.inner` is initialised by `new`, and `num` / `den` are two
        // distinct `fmpz` owned by it.
        unsafe {
            fmpz_set_rug(&mut q.inner.num, r.numer());
            fmpz_set_rug(&mut q.inner.den, r.denom());
        }
        q
    }

    /// Read the numerator and denominator out limb by limb.
    ///
    /// The pair goes through `Rational::from`, which canonicalises, so a
    /// non-canonical `fmpq` left by a raw FLINT write still yields a valid
    /// `Rational` — the same guarantee the former string parse gave.
    pub(crate) fn to_rug(&self) -> Rational {
        // SAFETY: both fields are initialised `fmpz` (see `new`).
        let (num, den): (Integer, Integer) =
            unsafe { (fmpz_get_rug(&self.inner.num), fmpz_get_rug(&self.inner.den)) };
        Rational::from((num, den))
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::Fmpq {
        &self.inner
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::Fmpq {
        &mut self.inner
    }
}

impl Drop for FlintRational {
    fn drop(&mut self) {
        // SAFETY: initialised in `new`.
        unsafe { ffi::fmpq_clear(&mut self.inner) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// FLINT's own printer — the conversion `to_rug` used before limb copying.
    fn flint_printed(q: &FlintRational) -> String {
        unsafe {
            let raw = ffi::fmpq_get_str(std::ptr::null_mut(), 10, q.as_ptr());
            let s = std::ffi::CStr::from_ptr(raw).to_string_lossy().into_owned();
            ffi::flint_free(raw.cast());
            s
        }
    }

    fn check(r: &Rational) {
        let q = FlintRational::from_rug(r);
        // FLINT agrees on the value it now holds (so the `fmpq` is canonical
        // and the mirrored field layout is right) ...
        assert_eq!(flint_printed(&q), r.to_string());
        // ... and it reads back unchanged.
        assert_eq!(&q.to_rug(), r);
    }

    fn big(negative: bool, limbs: &[u64]) -> Integer {
        let m = Integer::from_digits(limbs, rug::integer::Order::Lsf);
        if negative {
            -m
        } else {
            m
        }
    }

    #[test]
    fn edge_values() {
        let two64 = Integer::from(1) << 64u32;
        for (n, d) in [
            (Integer::new(), Integer::from(1)),
            (Integer::from(-691), Integer::from(2730)),
            (Integer::from(i64::MIN), Integer::from(u64::MAX)),
            (-two64.clone(), two64.clone() + 1u32),
            (
                (Integer::from(1) << 5000u32) + 1u32,
                Integer::from(3) << 700u32,
            ),
        ] {
            check(&Rational::from((n, d)));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_roundtrip(
            n_neg in any::<bool>(),
            n in prop::collection::vec(any::<u64>(), 0..60),
            d in prop::collection::vec(any::<u64>(), 0..60),
        ) {
            let den = big(false, &d);
            prop_assume!(den != 0);
            check(&Rational::from((big(n_neg, &n), den)));
        }
    }
}
