//! Dense `fmpz_mat_t` wrapper for Hermite / Smith normal forms.

use super::ffi;
use super::integer::FlintInteger;

/// Owned `m × n` integer matrix backed by FLINT.
pub(crate) struct FlintMat {
    inner: ffi::FmpzMatStruct,
}

impl FlintMat {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        assert!(rows <= i64::MAX as usize && cols <= i64::MAX as usize);
        let mut inner = ffi::FmpzMatStruct {
            entries: std::ptr::null_mut(),
            r: 0,
            c: 0,
            rows: std::ptr::null_mut(),
        };
        unsafe { ffi::fmpz_mat_init(&mut inner, rows as ffi::slong, cols as ffi::slong) };
        Self { inner }
    }

    pub(crate) fn rows(&self) -> usize {
        self.inner.r as usize
    }

    pub(crate) fn cols(&self) -> usize {
        self.inner.c as usize
    }

    unsafe fn entry_ptr(&mut self, i: usize, j: usize) -> *mut ffi::fmpz {
        debug_assert!(i < self.rows() && j < self.cols());
        let row_start = *self.inner.rows.add(i);
        row_start.add(j)
    }

    pub(crate) unsafe fn entry_const(&self, i: usize, j: usize) -> *const ffi::fmpz {
        debug_assert!(i < self.rows() && j < self.cols());
        let row_start = *self.inner.rows.add(i);
        row_start.add(j)
    }

    pub(crate) fn set_entry(&mut self, i: usize, j: usize, v: &FlintInteger) {
        unsafe {
            let e = self.entry_ptr(i, j);
            ffi::fmpz_set(e, v.inner_ptr());
        }
    }

    pub(crate) fn get_flint(&self, i: usize, j: usize) -> FlintInteger {
        let mut out = FlintInteger::new();
        unsafe {
            ffi::fmpz_set(out.inner_mut_ptr(), self.entry_const(i, j));
        }
        out
    }

    pub(crate) fn hnf_transform(&self, h: &mut FlintMat, u: &mut FlintMat) {
        unsafe {
            ffi::fmpz_mat_hnf_transform(&mut h.inner, &mut u.inner, &self.inner);
        }
    }
}

/// SNF oracle and shape checks used only from `matrix::normal_form` unit tests.
#[cfg(test)]
impl FlintMat {
    pub(crate) fn snf_diagonal(&self, s: &mut FlintMat) {
        unsafe {
            ffi::fmpz_mat_snf(&mut s.inner, &self.inner);
        }
    }

    pub(crate) fn is_in_hnf(&self) -> bool {
        unsafe { ffi::fmpz_mat_is_in_hnf(&self.inner) != 0 }
    }

    pub(crate) fn is_in_snf(&self) -> bool {
        unsafe { ffi::fmpz_mat_is_in_snf(&self.inner) != 0 }
    }

    pub(crate) fn equals(&self, other: &FlintMat) -> bool {
        unsafe { ffi::fmpz_mat_equal(&self.inner, &other.inner) != 0 }
    }
}

impl Drop for FlintMat {
    fn drop(&mut self) {
        unsafe { ffi::fmpz_mat_clear(&mut self.inner) };
    }
}
