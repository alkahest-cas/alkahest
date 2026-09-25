//! Drop-safe Rust wrappers over the `arb_t` / `acb_t` FFI in
//! [`crate::flint::arb`].
//!
//! Everything in this file is `pub(crate)`-shaped in spirit: it is the layer
//! [`crate::theta`] is written against, and it is what keeps raw FLINT pointers
//! out of the rest of the crate. Each type owns exactly one FLINT object and
//! clears it on drop.
//!
//! | FLINT C type | wrapper      | `Drop` calls       |
//! |--------------|--------------|--------------------|
//! | `arb_t`      | `Arb`      | `arb_clear`        |
//! | `acb_t`      | `Acb`      | `acb_clear`        |
//! | `acb_ptr`    | `AcbVec`   | `_acb_vec_clear`   |
//! | `arb_mat_t`  | `ArbMat`   | `arb_mat_clear`    |
//! | `acb_mat_t`  | `AcbMat`   | `acb_mat_clear`    |
//! | `fmpz_mat_t` | `IntMat`   | `fmpz_mat_clear`   |
//!
//! # Exact interchange with `rug::Float`
//!
//! A ball's midpoint and radius are binary floating-point numbers, so they
//! cross into `rug::Float` **exactly**, by way of the integer mantissa and
//! exponent (`arf_get_fmpz_2exp` / `arf_set_fmpz_2exp`). No decimal rounding
//! happens anywhere on that path, and no MPFR object is ever handed to FLINT:
//! FLINT here is linked against its own MPFR and `rug` against `gmp-mpfr-sys`'s,
//! and passing an `mpfr_t` between them would make the two libraries' exponent
//! ranges and allocators meet. The mantissa-and-exponent route avoids the
//! question entirely.
//!
//! Exponents outside `±2^28` are refused rather than converted, because MPFR's
//! default exponent range is narrower than `fmpz`'s and a silent overflow to
//! infinity would turn a finite enclosure into a meaningless one.

use rug::Float;

use super::arb::{self, AcbMatBuf, AcbStruct, ArbMatBuf, ArbStruct, ArfStruct, FmpzMatBuf};
use super::ffi;
use super::integer::FlintInteger;

/// Largest `|exponent|` this module will move between FLINT and MPFR. MPFR's
/// default exponent range is roughly `±2^30`; staying an order of magnitude
/// inside it leaves room for the mantissa length.
const MAX_ABS_EXP: i64 = 1 << 28;

/// Largest mantissa this module will move, in bits. A ball whose midpoint needs
/// more than this many bits is past any precision this crate offers.
const MAX_MANTISSA_BITS: u32 = 1 << 24;

// ---------------------------------------------------------------------------
// arf <-> rug::Float, exactly
// ---------------------------------------------------------------------------

/// Read an `arf_t` as an exact `rug::Float`.
///
/// Returns `None` when the value is not finite, or when its exponent or
/// mantissa is outside the range MPFR can hold — never a rounded answer.
///
/// # Safety
/// `a` must point to an initialised, **finite** `arf_t`.
unsafe fn arf_to_rug(a: *const ArfStruct) -> Option<Float> {
    let mut man = FlintInteger::new();
    let mut exp = FlintInteger::new();
    arb::arf_get_fmpz_2exp(man.inner_mut_ptr(), exp.inner_mut_ptr(), a);
    let exp_big = exp.to_rug();
    let e = exp_big.to_i64()?;
    if !(-MAX_ABS_EXP..=MAX_ABS_EXP).contains(&e) {
        return None;
    }
    let m = man.to_rug();
    let bits = m.significant_bits().max(2);
    if bits > MAX_MANTISSA_BITS {
        return None;
    }
    // `with_val(bits, &m)` is exact because `bits` is the mantissa's own width,
    // and `<<` is `mul_2exp`, which is exact for an in-range exponent.
    let f = Float::with_val(bits, &m);
    Some(f << (e as i32))
}

/// Write an exact `rug::Float` into an initialised `arf_t`.
///
/// Returns `false` (leaving `out` untouched) when `f` is not finite or falls
/// outside the interchange range.
///
/// # Safety
/// `out` must point to an initialised `arf_t`.
unsafe fn rug_to_arf(f: &Float, out: *mut ArfStruct) -> bool {
    if !f.is_finite() {
        return false;
    }
    if f.is_zero() {
        // `to_integer_exp` reports MPFR's *minimum* exponent for a zero, which
        // is far outside the interchange window below and would otherwise turn
        // "exactly zero" into a refusal. `arf_set_d(_, 0.0)` is exact and
        // carries no sign surprises: FLINT's arf has one zero.
        arb::arf_set_d(out, 0.0);
        return true;
    }
    let Some((m, e)) = f.to_integer_exp() else {
        return false;
    };
    if m.significant_bits() > MAX_MANTISSA_BITS {
        return false;
    }
    let e64 = i64::from(e);
    if !(-MAX_ABS_EXP..=MAX_ABS_EXP).contains(&e64) {
        return false;
    }
    let fm = FlintInteger::from_rug(&m);
    let fe = FlintInteger::from_i64(e64);
    arb::arf_set_fmpz_2exp(out, fm.inner_ptr(), fe.inner_ptr());
    true
}

// ---------------------------------------------------------------------------
// Arb — a real ball
// ---------------------------------------------------------------------------

/// An owned `arb_t`: a real number as `[mid ± rad]`.
pub struct Arb {
    raw: ArbStruct,
}

// SAFETY: an `arb_t` owns its heap allocations and holds no pointer back into
// itself, so the bytes may be moved and the value may cross threads. It has no
// interior mutability, so `&Arb` is likewise safe to share.
unsafe impl Send for Arb {}
unsafe impl Sync for Arb {}

impl Arb {
    pub fn new() -> Self {
        let mut raw = ArbStruct::zeroed();
        unsafe { arb::arb_init(&mut raw) };
        Arb { raw }
    }

    /// The ball `[0 ± ∞]` — a true but useless enclosure of anything.
    pub fn indeterminate() -> Self {
        let mut x = Self::new();
        unsafe { arb::arb_indeterminate(&mut x.raw) };
        x
    }

    /// Exact: an `f64` is a binary float and crosses without rounding.
    pub fn from_f64(v: f64) -> Self {
        let mut x = Self::new();
        unsafe { arb::arb_set_d(&mut x.raw, v) };
        x
    }

    pub fn from_i64(v: i64) -> Self {
        let mut x = Self::new();
        unsafe { arb::arb_set_si(&mut x.raw, v) };
        x
    }

    /// Build `[mid ± rad]` from two `rug::Float`s, exactly.
    ///
    /// Returns `None` if either falls outside the interchange range; the caller
    /// must treat that as a refusal rather than substituting an approximation.
    pub fn from_mid_rad(mid: &Float, rad: &Float) -> Option<Self> {
        let mut x = Self::new();
        unsafe {
            if !rug_to_arf(mid, &mut x.raw.mid) {
                return None;
            }
            arb::mag_zero(&mut x.raw.rad);
            if rad.is_infinite() {
                arb::arb_indeterminate(&mut x.raw);
                return Some(x);
            }
            if !rad.is_zero() {
                let mut err = ArfStruct::zeroed();
                arb::arf_init(&mut err);
                let ok = rug_to_arf(rad, &mut err);
                if ok {
                    arb::arb_add_error_arf(&mut x.raw, &err);
                }
                arb::arf_clear(&mut err);
                if !ok {
                    return None;
                }
            }
        }
        Some(x)
    }

    /// The midpoint as an exact `rug::Float`, or `None` for a ball that is not
    /// finite or whose exponent is out of interchange range.
    pub fn midpoint_rug(&self) -> Option<Float> {
        if !self.is_finite() {
            return None;
        }
        unsafe { arf_to_rug(&self.raw.mid) }
    }

    /// The radius as an exact `rug::Float`, or `None` under the same conditions
    /// as [`Arb::midpoint_rug`].
    pub fn radius_rug(&self) -> Option<Float> {
        if !self.is_finite() {
            return None;
        }
        // `arb_get_rad_arb` hands back the radius as an exact ball, whose
        // midpoint is the value wanted. Going through it avoids reading the
        // `mag`'s 30-bit mantissa layout directly.
        let mut t = Arb::new();
        unsafe {
            arb::arb_get_rad_arb(&mut t.raw, &self.raw);
            arf_to_rug(&t.raw.mid)
        }
    }

    /// Midpoint as `f64`, rounded to nearest. **Not** an enclosure on its own —
    /// pair it with [`Arb::radius_f64`], which rounds outward.
    pub fn midpoint_f64(&self) -> f64 {
        unsafe { arb::arf_get_d(&self.raw.mid, arb::ARF_RND_NEAR) }
    }

    /// Radius as `f64`, rounded **up**, and inflated by the error the midpoint
    /// picked up on its way to `f64`. `[midpoint_f64 ± radius_f64]` is a true
    /// enclosure of the ball.
    pub fn radius_f64(&self) -> f64 {
        if !self.is_finite() {
            return f64::INFINITY;
        }
        let mid_d = self.midpoint_f64();
        if !mid_d.is_finite() {
            return f64::INFINITY;
        }
        // err = |self - mid_d|, bounded above.
        let approx = Arb::from_f64(mid_d);
        let mut diff = Arb::new();
        let mut out;
        unsafe {
            arb::arb_sub(&mut diff.raw, &self.raw, &approx.raw, 64);
            let mut m = arb::MagStruct::zeroed();
            arb::mag_init(&mut m);
            arb::arb_get_mag(&mut m, &diff.raw);
            let mut a = ArfStruct::zeroed();
            arb::arf_init(&mut a);
            arb::arf_set_mag(&mut a, &m);
            out = arb::arf_get_d(&a, arb::ARF_RND_CEIL);
            arb::arf_clear(&mut a);
            arb::mag_clear(&mut m);
        }
        if !out.is_finite() || out < 0.0 {
            out = f64::INFINITY;
        }
        out
    }

    /// `floor(log2(|mid| / rad))`: how many bits of the midpoint are justified.
    ///
    /// FLINT saturates at `±ARF_PREC_EXACT` (`2^62 - 1`), which is passed
    /// through unchanged — an exact ball really does have unlimited relative
    /// accuracy, and a ball whose radius swamps its midpoint really has none.
    pub fn rel_accuracy_bits(&self) -> i64 {
        unsafe { arb::arb_rel_accuracy_bits(&self.raw) }
    }

    pub fn is_finite(&self) -> bool {
        unsafe { arb::arb_is_finite(&self.raw) != 0 }
    }

    pub fn is_exact(&self) -> bool {
        unsafe { arb::arb_is_exact(&self.raw) != 0 }
    }

    /// Certainly `> 0`.
    pub fn is_positive(&self) -> bool {
        unsafe { arb::arb_is_positive(&self.raw) != 0 }
    }

    pub fn contains_zero(&self) -> bool {
        unsafe { arb::arb_contains_zero(&self.raw) != 0 }
    }

    pub fn contains(&self, other: &Arb) -> bool {
        unsafe { arb::arb_contains(&self.raw, &other.raw) != 0 }
    }

    pub fn overlaps(&self, other: &Arb) -> bool {
        unsafe { arb::arb_overlaps(&self.raw, &other.raw) != 0 }
    }

    /// Byte-for-byte equality of midpoint and radius — *not* a mathematical
    /// equality test between the numbers the two balls enclose.
    pub fn same_ball(&self, other: &Arb) -> bool {
        unsafe { arb::arb_equal(&self.raw, &other.raw) != 0 }
    }

    /// FLINT's own rendering, e.g. `[1.0864348112133080145 +/- 4.72e-20]`.
    pub fn to_flint_string(&self, digits: usize) -> String {
        unsafe {
            let p = arb::arb_get_str(&self.raw, digits as ffi::slong, 0);
            if p.is_null() {
                return String::new();
            }
            let s = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            ffi::flint_free(p.cast());
            s
        }
    }

    pub(crate) fn as_ptr(&self) -> *const ArbStruct {
        &self.raw
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut ArbStruct {
        &mut self.raw
    }
}

macro_rules! arb_binop {
    ($name:ident, $ffi:ident) => {
        impl Arb {
            pub fn $name(&self, other: &Arb, prec: i64) -> Arb {
                let mut out = Arb::new();
                unsafe { arb::$ffi(&mut out.raw, &self.raw, &other.raw, prec) };
                out
            }
        }
    };
}
arb_binop!(add, arb_add);
arb_binop!(sub, arb_sub);
arb_binop!(mul, arb_mul);
arb_binop!(div, arb_div);
arb_binop!(pow, arb_pow);

macro_rules! arb_unop {
    ($name:ident, $ffi:ident) => {
        impl Arb {
            pub fn $name(&self, prec: i64) -> Arb {
                let mut out = Arb::new();
                unsafe { arb::$ffi(&mut out.raw, &self.raw, prec) };
                out
            }
        }
    };
}
arb_unop!(sqrt, arb_sqrt);
arb_unop!(exp, arb_exp);
arb_unop!(log, arb_log);
arb_unop!(gamma, arb_gamma);

impl Arb {
    pub fn neg(&self) -> Arb {
        let mut out = Arb::new();
        unsafe { arb::arb_neg(&mut out.raw, &self.raw) };
        out
    }

    pub fn abs(&self) -> Arb {
        let mut out = Arb::new();
        unsafe { arb::arb_abs(&mut out.raw, &self.raw) };
        out
    }

    /// `self * 2^e`, exactly.
    pub fn mul_2exp(&self, e: i64) -> Arb {
        let mut out = Arb::new();
        unsafe { arb::arb_mul_2exp_si(&mut out.raw, &self.raw, e) };
        out
    }

    pub fn pi(prec: i64) -> Arb {
        let mut out = Arb::new();
        unsafe { arb::arb_const_pi(&mut out.raw, prec) };
        out
    }
}

impl Default for Arb {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Arb {
    fn clone(&self) -> Self {
        let mut out = Arb::new();
        unsafe { arb::arb_set(&mut out.raw, &self.raw) };
        out
    }
}

impl Drop for Arb {
    fn drop(&mut self) {
        unsafe { arb::arb_clear(&mut self.raw) };
    }
}

impl std::fmt::Debug for Arb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_flint_string(20))
    }
}

// ---------------------------------------------------------------------------
// Acb — a complex ball
// ---------------------------------------------------------------------------

/// An owned `acb_t`: a complex number as a pair of real balls.
pub struct Acb {
    raw: AcbStruct,
}

// SAFETY: as for `Arb`; an `acb_t` is two `arb_t`s side by side.
unsafe impl Send for Acb {}
unsafe impl Sync for Acb {}

impl Acb {
    pub fn new() -> Self {
        let mut raw = AcbStruct::zeroed();
        unsafe { arb::acb_init(&mut raw) };
        Acb { raw }
    }

    pub fn indeterminate() -> Self {
        let mut z = Self::new();
        unsafe { arb::acb_indeterminate(&mut z.raw) };
        z
    }

    pub fn from_f64(re: f64, im: f64) -> Self {
        let mut z = Self::new();
        unsafe { arb::acb_set_d_d(&mut z.raw, re, im) };
        z
    }

    pub fn from_parts(re: &Arb, im: &Arb) -> Self {
        let mut z = Self::new();
        unsafe { arb::acb_set_arb_arb(&mut z.raw, re.as_ptr(), im.as_ptr()) };
        z
    }

    pub fn real(&self) -> Arb {
        let mut out = Arb::new();
        unsafe { arb::acb_get_real(out.as_mut_ptr(), &self.raw) };
        out
    }

    pub fn imag(&self) -> Arb {
        let mut out = Arb::new();
        unsafe { arb::acb_get_imag(out.as_mut_ptr(), &self.raw) };
        out
    }

    /// The smaller of the two components' relative accuracies — the honest
    /// summary for a complex ball, and what `acb_rel_accuracy_bits` returns.
    pub fn rel_accuracy_bits(&self) -> i64 {
        unsafe { arb::acb_rel_accuracy_bits(&self.raw) }
    }

    pub fn is_finite(&self) -> bool {
        unsafe { arb::acb_is_finite(&self.raw) != 0 }
    }

    pub fn contains_zero(&self) -> bool {
        unsafe { arb::acb_contains_zero(&self.raw) != 0 }
    }

    pub fn contains(&self, other: &Acb) -> bool {
        unsafe { arb::acb_contains(&self.raw, &other.raw) != 0 }
    }

    pub fn overlaps(&self, other: &Acb) -> bool {
        unsafe { arb::acb_overlaps(&self.raw, &other.raw) != 0 }
    }

    pub fn same_ball(&self, other: &Acb) -> bool {
        unsafe { arb::acb_equal(&self.raw, &other.raw) != 0 }
    }

    pub fn abs(&self, prec: i64) -> Arb {
        let mut out = Arb::new();
        unsafe { arb::acb_abs(out.as_mut_ptr(), &self.raw, prec) };
        out
    }

    pub fn neg(&self) -> Acb {
        let mut out = Acb::new();
        unsafe { arb::acb_neg(&mut out.raw, &self.raw) };
        out
    }

    pub fn conj(&self) -> Acb {
        let mut out = Acb::new();
        unsafe { arb::acb_conj(&mut out.raw, &self.raw) };
        out
    }

    pub fn mul_2exp(&self, e: i64) -> Acb {
        let mut out = Acb::new();
        unsafe { arb::acb_mul_2exp_si(&mut out.raw, &self.raw, e) };
        out
    }

    pub fn pi(prec: i64) -> Acb {
        let mut out = Acb::new();
        unsafe { arb::acb_const_pi(&mut out.raw, prec) };
        out
    }

    pub fn i() -> Acb {
        let mut out = Acb::new();
        unsafe { arb::acb_onei(&mut out.raw) };
        out
    }

    pub(crate) fn as_ptr(&self) -> *const AcbStruct {
        &self.raw
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut AcbStruct {
        &mut self.raw
    }

    /// Copy the value out of a raw `acb_struct` this crate does not own — used
    /// to read a matrix entry or a vector slot back.
    ///
    /// # Safety
    /// `p` must point to an initialised `acb_struct`.
    pub(crate) unsafe fn from_raw(p: *const AcbStruct) -> Acb {
        let mut out = Acb::new();
        arb::acb_set(&mut out.raw, p);
        out
    }
}

macro_rules! acb_binop {
    ($name:ident, $ffi:ident) => {
        impl Acb {
            pub fn $name(&self, other: &Acb, prec: i64) -> Acb {
                let mut out = Acb::new();
                unsafe { arb::$ffi(&mut out.raw, &self.raw, &other.raw, prec) };
                out
            }
        }
    };
}
acb_binop!(add, acb_add);
acb_binop!(sub, acb_sub);
acb_binop!(mul, acb_mul);
acb_binop!(div, acb_div);
acb_binop!(pow, acb_pow);

macro_rules! acb_unop {
    ($name:ident, $ffi:ident) => {
        impl Acb {
            pub fn $name(&self, prec: i64) -> Acb {
                let mut out = Acb::new();
                unsafe { arb::$ffi(&mut out.raw, &self.raw, prec) };
                out
            }
        }
    };
}
acb_unop!(sqrt, acb_sqrt);
acb_unop!(exp, acb_exp);
acb_unop!(log, acb_log);

impl Default for Acb {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Acb {
    fn clone(&self) -> Self {
        let mut out = Acb::new();
        unsafe { arb::acb_set(&mut out.raw, &self.raw) };
        out
    }
}

impl Drop for Acb {
    fn drop(&mut self) {
        unsafe { arb::acb_clear(&mut self.raw) };
    }
}

impl std::fmt::Debug for Acb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} + {:?}i", self.real(), self.imag())
    }
}

// ---------------------------------------------------------------------------
// AcbVec — a contiguous `acb_ptr`
// ---------------------------------------------------------------------------

/// `n` contiguous `acb_struct`s, the shape every `acb_theta` entry point wants
/// for its `z` argument and its output.
pub struct AcbVec {
    ptr: *mut AcbStruct,
    len: usize,
}

// SAFETY: the buffer is owned exclusively by this value.
unsafe impl Send for AcbVec {}

impl AcbVec {
    pub fn new(len: usize) -> Self {
        assert!(
            len <= i64::MAX as usize,
            "acb vector length overflows slong"
        );
        let ptr = if len == 0 {
            std::ptr::null_mut()
        } else {
            unsafe { arb::_acb_vec_init(len as ffi::slong) }
        };
        AcbVec { ptr, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Overwrite slot `i`.
    pub fn set(&mut self, i: usize, v: &Acb) {
        assert!(
            i < self.len,
            "acb vector index {i} out of range {}",
            self.len
        );
        // SAFETY: `i < len` and the buffer holds `len` initialised structs.
        unsafe { arb::acb_set(self.ptr.add(i), v.as_ptr()) };
    }

    /// Copy slot `i` out.
    pub fn get(&self, i: usize) -> Acb {
        assert!(
            i < self.len,
            "acb vector index {i} out of range {}",
            self.len
        );
        // SAFETY: as for `set`.
        unsafe { Acb::from_raw(self.ptr.add(i)) }
    }

    pub(crate) fn as_ptr(&self) -> *const AcbStruct {
        self.ptr
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut AcbStruct {
        self.ptr
    }
}

impl Drop for AcbVec {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { arb::_acb_vec_clear(self.ptr, self.len as ffi::slong) };
        }
    }
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

/// An owned `acb_mat_t`. Dimensions are carried here rather than read back out
/// of the C struct — see the layout note in [`crate::flint::arb`].
pub struct AcbMat {
    buf: AcbMatBuf,
    rows: usize,
    cols: usize,
}

// SAFETY: the matrix owns its entries and holds no pointer into itself.
unsafe impl Send for AcbMat {}

impl AcbMat {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows <= i64::MAX as usize && cols <= i64::MAX as usize);
        let mut buf = AcbMatBuf::sentinel();
        unsafe { arb::acb_mat_init(&mut buf, rows as ffi::slong, cols as ffi::slong) };
        AcbMat { buf, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn set_entry(&mut self, i: usize, j: usize, v: &Acb) {
        assert!(i < self.rows && j < self.cols);
        // SAFETY: bounds checked; the address comes from FLINT's own accessor,
        // so no layout assumption is made here.
        unsafe {
            let p = arb::acb_mat_entry_ptr(&mut self.buf, i as ffi::slong, j as ffi::slong);
            arb::acb_set(p, v.as_ptr());
        }
    }

    pub fn get_entry(&mut self, i: usize, j: usize) -> Acb {
        assert!(i < self.rows && j < self.cols);
        // SAFETY: as for `set_entry`.
        unsafe {
            let p = arb::acb_mat_entry_ptr(&mut self.buf, i as ffi::slong, j as ffi::slong);
            Acb::from_raw(p)
        }
    }

    /// Imaginary part, as an `arb_mat_t` of the same shape.
    pub fn imag(&self) -> ArbMat {
        let mut out = ArbMat::new(self.rows, self.cols);
        unsafe { arb::acb_mat_get_imag(out.as_mut_ptr(), &self.buf) };
        out
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut AcbMatBuf {
        &mut self.buf
    }
}

impl Drop for AcbMat {
    fn drop(&mut self) {
        unsafe { arb::acb_mat_clear(&mut self.buf) };
    }
}

/// An owned `arb_mat_t`.
pub struct ArbMat {
    buf: ArbMatBuf,
    rows: usize,
    cols: usize,
}

// SAFETY: as for `AcbMat`.
unsafe impl Send for ArbMat {}

impl ArbMat {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows <= i64::MAX as usize && cols <= i64::MAX as usize);
        let mut buf = ArbMatBuf::sentinel();
        unsafe { arb::arb_mat_init(&mut buf, rows as ffi::slong, cols as ffi::slong) };
        ArbMat { buf, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get_entry(&mut self, i: usize, j: usize) -> Arb {
        assert!(i < self.rows && j < self.cols);
        // SAFETY: bounds checked; address from FLINT's accessor.
        unsafe {
            let p = arb::arb_mat_entry_ptr(&mut self.buf, i as ffi::slong, j as ffi::slong);
            let mut out = Arb::new();
            arb::arb_set(out.as_mut_ptr(), p);
            out
        }
    }

    pub fn set_entry(&mut self, i: usize, j: usize, v: &Arb) {
        assert!(i < self.rows && j < self.cols);
        // SAFETY: bounds checked; address from FLINT's accessor.
        unsafe {
            let p = arb::arb_mat_entry_ptr(&mut self.buf, i as ffi::slong, j as ffi::slong);
            arb::arb_set(p, v.as_ptr());
        }
    }

    /// Is this matrix **certainly** symmetric positive definite?
    ///
    /// Answered by asking FLINT for a Cholesky factor: `arb_mat_cho` succeeds
    /// only when positive-definiteness is established at the working precision,
    /// so a `false` means "not proved", never "proved false".
    pub fn is_certainly_positive_definite(&self, prec: i64) -> bool {
        let mut l = ArbMat::new(self.rows, self.cols);
        unsafe { arb::arb_mat_cho(l.as_mut_ptr(), &self.buf, prec) != 0 }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut ArbMatBuf {
        &mut self.buf
    }
}

impl Drop for ArbMat {
    fn drop(&mut self) {
        unsafe { arb::arb_mat_clear(&mut self.buf) };
    }
}

/// An owned `fmpz_mat_t`, used here only to carry the symplectic matrix
/// produced by Siegel reduction back to Rust.
pub struct IntMat {
    buf: FmpzMatBuf,
    rows: usize,
    cols: usize,
}

// SAFETY: as for `AcbMat`.
unsafe impl Send for IntMat {}

impl IntMat {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows <= i64::MAX as usize && cols <= i64::MAX as usize);
        let mut buf = FmpzMatBuf::sentinel();
        // SAFETY: `FmpzMatBuf` is 64 bytes at 8-byte alignment; FLINT writes
        // the 32 bytes of `fmpz_mat_struct` into the front of it and this
        // crate never reads a field back out. The cast exists only so that the
        // single `fmpz_mat_init` declaration in `flint::ffi` can be reused —
        // see the note on `fmpz_mat_entry` in `flint::arb`.
        unsafe {
            ffi::fmpz_mat_init(
                (&mut buf as *mut FmpzMatBuf).cast(),
                rows as ffi::slong,
                cols as ffi::slong,
            )
        };
        IntMat { buf, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Entry `(i, j)` as a `rug::Integer`. Exact at any size.
    pub fn get_entry(&self, i: usize, j: usize) -> rug::Integer {
        assert!(i < self.rows && j < self.cols);
        // SAFETY: bounds checked; `fmpz_mat_entry` is a real exported function
        // in FLINT 3.1+, so the row offset is computed by the library.
        unsafe {
            let p = ffi::fmpz_mat_entry(
                (&self.buf as *const arb::FmpzMatBuf).cast::<ffi::FmpzMatStruct>(),
                i as ffi::slong,
                j as ffi::slong,
            );
            let mut tmp = FlintInteger::new();
            ffi::fmpz_set(tmp.inner_mut_ptr(), p);
            tmp.to_rug()
        }
    }

    pub(crate) fn as_ptr(&self) -> *const FmpzMatBuf {
        &self.buf
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut FmpzMatBuf {
        &mut self.buf
    }
}

impl Drop for IntMat {
    fn drop(&mut self) {
        // SAFETY: see `IntMat::new` for why the cast is sound.
        unsafe { ffi::fmpz_mat_clear((&mut self.buf as *mut FmpzMatBuf).cast()) };
    }
}
