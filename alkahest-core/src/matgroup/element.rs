//! Small helpers over [`GfMatrix`] that the matrix-group algorithms need and
//! [`crate::ffield`] does not provide.
//!
//! Three kinds of thing live here.
//!
//! * **Hashable keys.** [`GfMatrix`] is FLINT-backed and implements neither
//!   `Hash` nor `Eq`, but orbit computation needs a hash map keyed by a vector
//!   and normal closure needs a set keyed by a matrix. Both keys are the
//!   entries' coordinates over the prime subfield, padded to `k` per entry and
//!   flattened — one allocation per key, and equal keys mean equal matrices
//!   because [`crate::ffield::FieldElement`] is stored in canonical form.
//! * **Scalar arithmetic.** `FieldElement` carries no field and exposes no
//!   `+`/`*`, so single-element arithmetic is done through `1 × 1` matrices.
//!   That is clumsy and allocates, and it is used only off the hot paths
//!   (building generators, normalising projective points, finding a primitive
//!   element) — the orbit and Schreier–Sims loops multiply whole matrices.
//! * **Row-vector plumbing.** The action throughout this module is on **row**
//!   vectors, `v ↦ v·M`, so `eᵢ·M` is simply row `i` of `M` and needs no
//!   multiplication at all. That is why the stabilizer chain's base points are
//!   standard basis vectors: it makes the innermost step of `strip` a row read.

use super::error::MatGroupError;
use super::MAX_MATGROUP_FIELD_ORDER;
use crate::ffield::{FieldElement, FiniteField, GfMatrix};

/// A hashable key for a matrix or row vector over GF(q).
///
/// The entries' prime-subfield coordinates, each padded to `k`, flattened in
/// row-major order.
pub(crate) type EntryKey = Vec<u64>;

/// The hashable key of any matrix (or `1 × d` row vector).
pub(crate) fn entry_key(field: &FiniteField, m: &GfMatrix) -> EntryKey {
    let k = field.degree();
    let elements = m.to_elements();
    let mut out = Vec::with_capacity(elements.len() * k);
    for e in &elements {
        out.extend(e.coefficients_padded(k));
    }
    out
}

/// `q` as a `u64`, refusing above [`MAX_MATGROUP_FIELD_ORDER`].
///
/// Used only by the operations that enumerate GF(q) itself.
pub(crate) fn field_order_u64(field: &FiniteField) -> Result<u64, MatGroupError> {
    let q = field.order().ok_or_else(|| MatGroupError::FieldTooLarge {
        order: format!("{}^{}", field.characteristic(), field.degree()),
        max: MAX_MATGROUP_FIELD_ORDER,
    })?;
    if q > u128::from(MAX_MATGROUP_FIELD_ORDER) {
        return Err(MatGroupError::FieldTooLarge {
            order: q.to_string(),
            max: MAX_MATGROUP_FIELD_ORDER,
        });
    }
    Ok(q as u64)
}

/// Every element of GF(q), in a fixed order: the base-`p` digits of the index
/// are the element's coordinates in the polynomial basis, least significant
/// first. Index `0` is zero and index `1` is one.
pub(crate) fn field_elements(field: &FiniteField) -> Result<Vec<FieldElement>, MatGroupError> {
    let q = field_order_u64(field)?;
    let p = field.characteristic();
    let k = field.degree();
    let mut out = Vec::with_capacity(q as usize);
    for code in 0..q {
        let mut rest = code;
        let mut coeffs = Vec::with_capacity(k);
        for _ in 0..k {
            coeffs.push(rest % p);
            rest /= p;
        }
        out.push(field.element(&coeffs)?);
    }
    Ok(out)
}

/// A `1 × 1` matrix holding `a`, so that FLINT can do scalar arithmetic.
fn scalar_matrix(field: &FiniteField, a: &FieldElement) -> Result<GfMatrix, MatGroupError> {
    Ok(GfMatrix::from_elements(
        field,
        1,
        1,
        std::slice::from_ref(a),
    )?)
}

fn only_element(m: &GfMatrix) -> Result<FieldElement, MatGroupError> {
    m.to_elements()
        .into_iter()
        .next()
        .ok_or(MatGroupError::Internal {
            detail: "a 1x1 matrix with no entry",
        })
}

/// `a * b` in GF(q).
pub(crate) fn fe_mul(
    field: &FiniteField,
    a: &FieldElement,
    b: &FieldElement,
) -> Result<FieldElement, MatGroupError> {
    let prod = scalar_matrix(field, a)?.mul(&scalar_matrix(field, b)?)?;
    only_element(&prod)
}

/// `a − b` in GF(q).
pub(crate) fn fe_sub(
    field: &FiniteField,
    a: &FieldElement,
    b: &FieldElement,
) -> Result<FieldElement, MatGroupError> {
    let diff = scalar_matrix(field, a)?.sub(&scalar_matrix(field, b)?)?;
    only_element(&diff)
}

/// `a⁻¹` in GF(q).
///
/// # Errors
///
/// `E-GFQ-009` for `a = 0`, passed through from the GF(q) layer.
pub(crate) fn fe_inverse(
    field: &FiniteField,
    a: &FieldElement,
) -> Result<FieldElement, MatGroupError> {
    let inv = scalar_matrix(field, a)?.inverse()?;
    only_element(&inv)
}

/// A generator of the multiplicative group GF(q)*, found by taking the first
/// element of the enumeration whose multiplicative order is `q − 1`.
///
/// [`FiniteField::generator`] is **not** used: it returns the polynomial-basis
/// generator `a = x mod f`, which is primitive exactly when `f` is a primitive
/// polynomial. `FiniteField::extension` uses FLINT's Conway polynomials, which
/// are primitive, but `FiniteField::with_defining_polynomial` accepts any
/// irreducible `f`, and over such a field `a` can have order strictly less than
/// `q − 1`. Trusting it would make `GL(n, q)`'s generating set silently
/// generate a proper subgroup.
pub(crate) fn primitive_element(field: &FiniteField) -> Result<FieldElement, MatGroupError> {
    let q = field_order_u64(field)?;
    let one = field.one();
    for cand in field_elements(field)? {
        if cand.is_zero() {
            continue;
        }
        let mut x = cand.clone();
        let mut order = 1u64;
        while x != one {
            x = fe_mul(field, &x, &cand)?;
            order += 1;
            if order > q {
                return Err(MatGroupError::Internal {
                    detail: "an element of GF(q)* has multiplicative order above q",
                });
            }
        }
        if order == q - 1 {
            return Ok(cand);
        }
    }
    Err(MatGroupError::Internal {
        detail: "GF(q)* is cyclic but no generator was found",
    })
}

/// The `i`-th standard basis vector, as a `1 × d` row vector.
pub(crate) fn basis_vector(
    field: &FiniteField,
    degree: usize,
    i: usize,
) -> Result<GfMatrix, MatGroupError> {
    if i >= degree {
        return Err(MatGroupError::DegreeMismatch {
            expected: degree,
            got: i + 1,
        });
    }
    let mut entries = vec![field.zero(); degree];
    entries[i] = field.one();
    Ok(GfMatrix::from_elements(field, 1, degree, &entries)?)
}

/// Row `i` of `m`, as a `1 × d` row vector — this is `eᵢ·m`.
pub(crate) fn row_vector(
    field: &FiniteField,
    m: &GfMatrix,
    i: usize,
) -> Result<GfMatrix, MatGroupError> {
    let cols = m.ncols();
    let elements = m.to_elements();
    let start = i * cols;
    let slice = elements
        .get(start..start + cols)
        .ok_or(MatGroupError::Internal {
            detail: "row index outside a matrix whose shape was already checked",
        })?;
    Ok(GfMatrix::from_elements(field, 1, cols, slice)?)
}

/// Is `m` the identity matrix?
///
/// Compares entry by entry rather than allocating an identity and calling
/// `equals`, because the Schreier–Sims inner loop asks this once per Schreier
/// generator.
pub(crate) fn is_identity(field: &FiniteField, m: &GfMatrix) -> bool {
    let (rows, cols) = m.shape();
    if rows != cols {
        return false;
    }
    let one = field.one();
    let elements = m.to_elements();
    for i in 0..rows {
        for j in 0..cols {
            let e = &elements[i * cols + j];
            if i == j {
                if *e != one {
                    return false;
                }
            } else if !e.is_zero() {
                return false;
            }
        }
    }
    true
}

/// The first standard basis vector `m` does not fix, i.e. the first `i` with
/// row `i` of `m` different from `eᵢ`.
///
/// `None` exactly when `m` is the identity, which is why a stabilizer chain's
/// base can always be taken inside the standard basis: a matrix fixing every
/// `eᵢ` *is* the identity, so the chain has at most `d` levels.
pub(crate) fn first_moved_basis_vector(field: &FiniteField, m: &GfMatrix) -> Option<usize> {
    let (rows, cols) = m.shape();
    if rows != cols {
        return None;
    }
    let one = field.one();
    let elements = m.to_elements();
    for i in 0..rows {
        for j in 0..cols {
            let e = &elements[i * cols + j];
            let expected_one = i == j;
            if (expected_one && *e != one) || (!expected_one && !e.is_zero()) {
                return Some(i);
            }
        }
    }
    None
}

/// The canonical representative of the projective point `[v]`: `v` scaled so
/// that its first non-zero coordinate is `1`.
///
/// # Errors
///
/// `E-MATGRP-012` for the zero vector, which spans no line.
pub(crate) fn projective_normalise(
    field: &FiniteField,
    v: &GfMatrix,
) -> Result<GfMatrix, MatGroupError> {
    let elements = v.to_elements();
    let pivot = elements
        .iter()
        .find(|e| !e.is_zero())
        .ok_or(MatGroupError::ZeroVector)?;
    if *pivot == field.one() {
        return Ok(v.clone());
    }
    let scale = fe_inverse(field, pivot)?;
    Ok(v.scalar_mul(&scale)?)
}

/// `g⁻¹ h g`.
pub(crate) fn conjugate(g: &GfMatrix, h: &GfMatrix) -> Result<GfMatrix, MatGroupError> {
    Ok(g.inverse()?.mul(h)?.mul(g)?)
}

/// `[g, h] = g⁻¹ h⁻¹ g h`.
pub(crate) fn commutator(g: &GfMatrix, h: &GfMatrix) -> Result<GfMatrix, MatGroupError> {
    Ok(g.inverse()?.mul(&h.inverse()?)?.mul(g)?.mul(h)?)
}

/// Do `a` and `b` commute?
pub(crate) fn commutes(a: &GfMatrix, b: &GfMatrix) -> Result<bool, MatGroupError> {
    Ok(a.mul(b)?.equals(&b.mul(a)?))
}
