//! Generating sets for the classical families `GL`, `SL` and `Sp`, and for a
//! Singer cycle.
//!
//! Every generating set here is **theory-backed and complete**, not pruned
//! against a known order. That distinction matters for the tests: the orders in
//! `matgroup::tests` are computed by Schreier–Sims from these generators and
//! compared with the closed-form product formulas, so a generating set that had
//! been trimmed until the order matched the formula would make that test
//! vacuous.
//!
//! * `SL(n, q) = ⟨E_ij(λ)⟩` for `i ≠ j` and `λ` running over the polynomial
//!   basis `1, a, …, a^{k−1}` of GF(q) over GF(p). Elementary transvections
//!   generate `SL` for `n ≥ 2`, and `E_ij(λ)·E_ij(μ) = E_ij(λ+μ)` (because
//!   `e_ij·e_ij = 0` for `i ≠ j`), so the GF(p)-basis values generate all of
//!   them.
//! * `GL(n, q) = ⟨SL(n, q), diag(a, 1, …, 1)⟩` for a **primitive** `a`. The
//!   determinant of the extra generator generates GF(q)*, so the subgroup
//!   surjects onto GF(q)* with kernel containing `SL`, hence is everything.
//! * `Sp(2n, q) = ⟨T_{v,λ}⟩` over symplectic transvections
//!   `T_{v,λ}: x ↦ x + λ⟨x,v⟩v`, i.e. `T = I + λ·Ω·vᵀ·v`. Transvections
//!   generate `Sp` (Dieudonné); `T_{v,λ}T_{v,μ} = T_{v,λ+μ}` because
//!   `⟨v,v⟩ = 0`, and `T_{cv,λ} = T_{v,λc²}`, so `v` running over *projective*
//!   representatives with `λ` over the GF(p)-basis is already the full set.
//! * A Singer cycle is the companion matrix of a primitive polynomial of degree
//!   `n` over GF(q): it is multiplication by a generator of GF(qⁿ)* written in
//!   the basis `1, α, …, α^{n−1}`, so it is cyclic of order `qⁿ − 1`.
//!
//! # What is not here
//!
//! `GU`, `SU`, `SO`, `O`, `Ω` and the twisted types. None of them "come
//! cheaply": each needs its own canonical form for the relevant sesquilinear or
//! quadratic form (and, for the orthogonal groups in odd dimension and in
//! characteristic 2, several inequivalent ones), plus a generating set that is
//! correct in every degenerate small case. Those are a separate piece of work
//! and are refused with `E-MATGRP-010` rather than approximated. The generic
//! algorithms are the point of this module: an orthogonal group can be built
//! today with [`MatGroup::new`] from generators the caller supplies, and
//! everything else here then applies to it.

use rug::ops::Pow;
use rug::Integer;

use super::element::{
    entry_key, field_elements, field_order_u64, is_identity, primitive_element,
    projective_normalise,
};
use super::error::MatGroupError;
use super::group::MatGroup;
use super::MAX_MATGROUP_DEGREE;
use crate::ffield::{FieldElement, FiniteField, GfMatrix};

/// `∏_{i=0}^{n−1} (qⁿ − qⁱ)` — the closed form for `|GL(n, q)|`.
///
/// Provided so that a caller (and this module's tests) can compare the
/// Schreier–Sims answer with the formula. No enumeration and no cap.
pub fn gl_order(q: &Integer, n: usize) -> Integer {
    let qn = q.clone().pow(u32::try_from(n).unwrap_or(u32::MAX));
    let mut acc = Integer::from(1);
    for i in 0..n {
        acc *= qn.clone() - q.clone().pow(u32::try_from(i).unwrap_or(u32::MAX));
    }
    acc
}

/// `|GL(n, q)| / (q − 1)` — the closed form for `|SL(n, q)|`.
pub fn sl_order(q: &Integer, n: usize) -> Integer {
    gl_order(q, n) / (q.clone() - Integer::from(1))
}

/// `q^{n²} · ∏_{i=1}^{n} (q^{2i} − 1)` — the closed form for `|Sp(2n, q)|`.
///
/// Note the argument is `n`, so the matrices are `2n × 2n`.
pub fn sp_order(q: &Integer, n: usize) -> Integer {
    let mut acc = q.clone().pow(u32::try_from(n * n).unwrap_or(u32::MAX));
    for i in 1..=n {
        acc *= q.clone().pow(u32::try_from(2 * i).unwrap_or(u32::MAX)) - Integer::from(1);
    }
    acc
}

/// The polynomial basis `1, a, a², …, a^{k−1}` of GF(q) over GF(p).
///
/// Every element of GF(q) is a GF(p)-combination of these, which is what makes
/// them enough for the transvection arguments in the module docs.
fn prime_field_basis(field: &FiniteField) -> Result<Vec<FieldElement>, MatGroupError> {
    let k = field.degree();
    let mut out = Vec::with_capacity(k);
    for m in 0..k {
        let mut coeffs = vec![0u64; k];
        coeffs[m] = 1;
        out.push(field.element(&coeffs)?);
    }
    Ok(out)
}

fn check_degree(family: &'static str, n: usize, max: usize) -> Result<(), MatGroupError> {
    if n == 0 {
        return Err(MatGroupError::UnsupportedConstruction {
            family,
            reason: "n must be at least 1",
        });
    }
    if n > max {
        return Err(MatGroupError::DegreeTooLarge { degree: n, max });
    }
    Ok(())
}

/// `E_ij(λ) = I + λ·e_ij`, the elementary transvection.
fn transvection(
    field: &FiniteField,
    n: usize,
    i: usize,
    j: usize,
    lambda: &FieldElement,
) -> Result<GfMatrix, MatGroupError> {
    let mut entries = vec![field.zero(); n * n];
    for d in 0..n {
        entries[d * n + d] = field.one();
    }
    entries[i * n + j] = lambda.clone();
    Ok(GfMatrix::from_elements(field, n, n, &entries)?)
}

impl MatGroup {
    /// `SL(n, q)`, from elementary transvections.
    ///
    /// `SL(1, q)` is the trivial group.
    ///
    /// # Errors
    ///
    /// `E-MATGRP-010` for `n = 0`; `E-MATGRP-005` above
    /// [`MAX_MATGROUP_DEGREE`].
    pub fn special_linear(field: &FiniteField, n: usize) -> Result<MatGroup, MatGroupError> {
        check_degree("SL", n, MAX_MATGROUP_DEGREE)?;
        if n == 1 {
            return MatGroup::trivial(field, 1);
        }
        let basis = prime_field_basis(field)?;
        let mut generators = Vec::with_capacity(n * (n - 1) * basis.len());
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                for lambda in &basis {
                    generators.push(transvection(field, n, i, j, lambda)?);
                }
            }
        }
        MatGroup::new(field, n, generators)
    }

    /// `GL(n, q)`: `SL(n, q)` together with `diag(a, 1, …, 1)` for a primitive
    /// `a`.
    ///
    /// # Errors
    ///
    /// `E-MATGRP-010` for `n = 0`; `E-MATGRP-005` above
    /// [`MAX_MATGROUP_DEGREE`]; `E-MATGRP-009` when `q` is too large to
    /// enumerate in the search for a primitive element.
    pub fn general_linear(field: &FiniteField, n: usize) -> Result<MatGroup, MatGroupError> {
        check_degree("GL", n, MAX_MATGROUP_DEGREE)?;
        let a = primitive_element(field)?;
        let mut entries = vec![field.zero(); n * n];
        for d in 0..n {
            entries[d * n + d] = field.one();
        }
        entries[0] = a;
        let scaling = GfMatrix::from_elements(field, n, n, &entries)?;

        let mut generators = if n == 1 {
            Vec::new()
        } else {
            MatGroup::special_linear(field, n)?.generators().to_vec()
        };
        if !is_identity(field, &scaling) {
            generators.push(scaling);
        }
        MatGroup::new(field, n, generators)
    }

    /// `Sp(2n, q)`, from symplectic transvections.
    ///
    /// The form is `Ω = [[0, I], [−I, 0]]`, the same one
    /// [`crate::stabilizer::is_symplectic`] tests against, so membership in this
    /// group and that predicate agree — asserted in `matgroup::tests` rather
    /// than left as an argument on paper.
    ///
    /// Note the argument is `n`: the matrices are `2n × 2n`.
    ///
    /// # Errors
    ///
    /// `E-MATGRP-010` for `n = 0`; `E-MATGRP-005` when `2n` is above
    /// [`MAX_MATGROUP_DEGREE`]; `E-MATGRP-009` for a `q` too large to
    /// enumerate.
    pub fn symplectic(field: &FiniteField, n: usize) -> Result<MatGroup, MatGroupError> {
        if n == 0 {
            return Err(MatGroupError::UnsupportedConstruction {
                family: "Sp",
                reason: "n must be at least 1, giving 2n x 2n matrices",
            });
        }
        let d = 2 * n;
        if d > MAX_MATGROUP_DEGREE {
            return Err(MatGroupError::DegreeTooLarge {
                degree: d,
                max: MAX_MATGROUP_DEGREE,
            });
        }
        let omega = symplectic_form_matrix(field, n)?;
        let identity = GfMatrix::identity(field, d)?;
        let basis = prime_field_basis(field)?;

        // Projective representatives of the non-zero vectors of GF(q)^{2n}.
        let alphabet = field_elements(field)?;
        let q = alphabet.len();
        let total = q.checked_pow(u32::try_from(d).unwrap_or(u32::MAX)).ok_or(
            MatGroupError::UnsupportedConstruction {
                family: "Sp",
                reason: "q^{2n} does not fit in a machine word",
            },
        )?;
        let mut seen = std::collections::HashSet::new();
        let mut generators = Vec::new();
        for code in 1..total {
            let mut rest = code;
            let mut entries = Vec::with_capacity(d);
            for _ in 0..d {
                entries.push(alphabet[rest % q].clone());
                rest /= q;
            }
            let v = GfMatrix::from_elements(field, 1, d, &entries)?;
            let v = projective_normalise(field, &v)?;
            if !seen.insert(entry_key(field, &v)) {
                continue;
            }
            let outer = omega.mul(&v.transpose())?.mul(&v)?;
            for lambda in &basis {
                let t = identity.add(&outer.scalar_mul(lambda)?)?;
                if !is_identity(field, &t) {
                    generators.push(t);
                }
            }
        }
        MatGroup::new(field, d, generators)
    }

    /// A **Singer cycle** in `GL(n, q)`: a cyclic subgroup of order `qⁿ − 1`.
    ///
    /// The multiplicative group of GF(qⁿ) acts on GF(qⁿ) as an `n`-dimensional
    /// GF(q)-vector space, and a generator of it becomes the companion matrix of
    /// a primitive polynomial of degree `n`. Nothing about its order resembles
    /// the product formulas of the classical families, which is what makes it a
    /// useful independent check on the Schreier–Sims order.
    ///
    /// # Errors
    ///
    /// `E-MATGRP-010` for `n = 0`, or for a non-prime base field: finding a
    /// primitive polynomial of degree `n` over GF(p^k) with `k > 1` needs
    /// irreducibility testing over GF(p^k), which [`crate::ffield`] does not
    /// offer. `E-MATGRP-009` when `qⁿ` is above
    /// [`MAX_MATGROUP_FIELD_ORDER`](super::MAX_MATGROUP_FIELD_ORDER).
    pub fn singer_cycle(field: &FiniteField, n: usize) -> Result<MatGroup, MatGroupError> {
        check_degree("Singer", n, MAX_MATGROUP_DEGREE)?;
        let q = field_order_u64(field)?;
        if n == 1 {
            // GL(1, q) is already cyclic of order q - 1.
            let a = primitive_element(field)?;
            let m = GfMatrix::from_elements(field, 1, 1, std::slice::from_ref(&a))?;
            return MatGroup::new(field, 1, vec![m]);
        }
        if !field.is_prime_field() {
            return Err(MatGroupError::UnsupportedConstruction {
                family: "Singer",
                reason: "a primitive polynomial of degree n is searched for over GF(p) only; \
                         over a proper extension GF(p^k) this needs irreducibility testing \
                         that the GF(q) layer does not expose",
            });
        }
        let p = field.characteristic();
        let big_q = Integer::from(q).pow(u32::try_from(n).unwrap_or(u32::MAX));
        // `q^n` bounds both the search over monic degree-n polynomials over
        // GF(p) (their coefficient vectors are exactly the base-q codes below)
        // and the order `q^n - 1` whose prime factors the primitivity test
        // needs, so one cap covers both.
        let candidates = big_q
            .to_u64()
            .filter(|c| *c <= super::MAX_MATGROUP_FIELD_ORDER)
            .ok_or_else(|| MatGroupError::FieldTooLarge {
                order: big_q.to_string(),
                max: super::MAX_MATGROUP_FIELD_ORDER,
            })?;
        let order_u64 = candidates - 1;
        let primes = distinct_prime_factors(order_u64);
        for code in 0..candidates {
            let mut rest = code;
            let mut coeffs = Vec::with_capacity(n + 1);
            for _ in 0..n {
                coeffs.push(rest % q);
                rest /= q;
            }
            if coeffs[0] == 0 {
                continue;
            }
            coeffs.push(1);
            // `with_defining_polynomial` refuses a reducible modulus with
            // E-GFQ-004, which is exactly the irreducibility test wanted here.
            if FiniteField::with_defining_polynomial(p, &coeffs).is_err() {
                continue;
            }
            let companion = companion_matrix(field, &coeffs)?;
            if is_primitive(field, &companion, order_u64, &primes)? {
                return MatGroup::new(field, n, vec![companion]);
            }
        }
        Err(MatGroupError::Internal {
            detail: "no primitive polynomial of degree n over GF(p) was found, but one exists",
        })
    }
}

/// `Ω = [[0, I], [−I, 0]]`, `2n × 2n`.
///
/// Built here rather than taken from [`crate::stabilizer`] so that this module
/// depends only on [`crate::ffield`] and [`crate::group`]. The two are checked
/// against each other in `matgroup::tests`.
fn symplectic_form_matrix(field: &FiniteField, n: usize) -> Result<GfMatrix, MatGroupError> {
    let d = 2 * n;
    let minus_one = field.characteristic() - 1;
    let mut entries = vec![0u64; d * d];
    for i in 0..n {
        entries[i * d + (n + i)] = 1;
        entries[(n + i) * d + i] = minus_one;
    }
    Ok(GfMatrix::from_u64(field, d, d, &entries)?)
}

/// The companion matrix of the monic polynomial with ascending coefficients
/// `coeffs`, in the **row-vector** convention: row `i` is the image of `eᵢ`.
///
/// With the basis `1, α, …, α^{n−1}` of `GF(q)[x]/(f)`, multiplication by `α`
/// sends `α^i ↦ α^{i+1}` for `i < n−1` and `α^{n−1} ↦ −c₀ − c₁α − ⋯`.
fn companion_matrix(field: &FiniteField, coeffs: &[u64]) -> Result<GfMatrix, MatGroupError> {
    let n = coeffs.len() - 1;
    let p = field.characteristic();
    let mut entries = vec![0u64; n * n];
    for i in 0..n - 1 {
        entries[i * n + (i + 1)] = 1;
    }
    for (j, &c) in coeffs.iter().take(n).enumerate() {
        entries[(n - 1) * n + j] = (p - c % p) % p;
    }
    Ok(GfMatrix::from_u64(field, n, n, &entries)?)
}

/// `m^exponent`, by binary exponentiation.
fn matrix_pow(field: &FiniteField, m: &GfMatrix, exponent: u64) -> Result<GfMatrix, MatGroupError> {
    let mut result = GfMatrix::identity(field, m.nrows())?;
    let mut base = m.clone();
    let mut e = exponent;
    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&base)?;
        }
        e >>= 1;
        if e > 0 {
            base = base.mul(&base)?;
        }
    }
    Ok(result)
}

/// Does `m` have multiplicative order exactly `order`?
///
/// `m^order = I` is assumed (it holds for the companion matrix of an
/// irreducible polynomial) and is checked anyway; the test is then that
/// `m^{order/r} ≠ I` for every prime `r | order`.
fn is_primitive(
    field: &FiniteField,
    m: &GfMatrix,
    order: u64,
    primes: &[u64],
) -> Result<bool, MatGroupError> {
    if !is_identity(field, &matrix_pow(field, m, order)?) {
        return Ok(false);
    }
    for &r in primes {
        if is_identity(field, &matrix_pow(field, m, order / r)?) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The distinct prime factors of `n`, by trial division. `n` here is at most
/// `q^d − 1` with `q^d` small, so trial division is the right tool.
fn distinct_prime_factors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            out.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}
