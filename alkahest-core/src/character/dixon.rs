//! The modular half of Dixon's algorithm: a working prime, a primitive root of
//! unity in GF(p), and the simultaneous diagonalisation of the class
//! multiplication matrices over GF(p).
//!
//! Nothing in here knows what a character is. It takes integer matrices that
//! are known to commute and to be diagonalisable over GF(p), and returns their
//! common eigenvectors — one line per conjugacy class. The lifting back to
//! `ℤ[ζ_e]` is [`super::table`]'s job.
//!
//! The linear algebra is [`GfMatrix`]: `charpoly` and `nullspace` are FLINT's
//! `nmod_mat` routines, and the eigenvalues are the linear factors of the
//! characteristic polynomial from FLINT's `nmod_poly_factor`. Only the scalar
//! arithmetic (`pow_mod`, `inv_mod`) is done here, in `u128`.

use super::error::CharacterError;
use crate::ffield::{FiniteField, GfMatrix};
use crate::flint::nmod::{FlintNmodPoly, FlintNmodPolyFactor};

/// `base^exp mod p`, with the products taken in `u128` so any word-sized `p`
/// is safe.
pub(super) fn pow_mod(base: u64, exp: u64, p: u64) -> u64 {
    let mut result: u64 = 1 % p;
    let mut b = base % p;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = ((result as u128 * b as u128) % p as u128) as u64;
        }
        b = ((b as u128 * b as u128) % p as u128) as u64;
        e >>= 1;
    }
    result
}

/// `a·b mod p`.
pub(super) fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

/// `a⁻¹ mod p` for prime `p`, by Fermat. Zero has no inverse and is a typed
/// refusal rather than a wrong answer.
pub(super) fn inv_mod(a: u64, p: u64) -> Result<u64, CharacterError> {
    let a = a % p;
    if a == 0 {
        return Err(CharacterError::Internal {
            detail: format!("tried to invert 0 modulo {p}"),
        });
    }
    Ok(pow_mod(a, p - 2, p))
}

/// The distinct prime factors of `n ≥ 1`, by trial division.
///
/// `n` here is `exp G`, at most the group order, so trial division to `√n` is
/// far cheaper than the group enumeration that produced it.
pub(super) fn distinct_prime_factors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
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

/// How many candidates `k·e + 1` [`working_prime`] will try before refusing.
///
/// Dirichlet guarantees infinitely many primes in the progression, and the
/// first one is in practice a handful of steps away; the bound exists only so
/// that a bug cannot turn into a hang.
const PRIME_SEARCH_CANDIDATES: u64 = 200_000;

/// The smallest prime `p ≡ 1 (mod exponent)` with `p > floor`.
///
/// Both conditions earn their keep:
///
/// * `p ≡ 1 (mod exp G)` puts a full set of `exp G`-th roots of unity in GF(p),
///   which is what makes every eigenvalue of a class multiplication matrix lie
///   in GF(p) rather than in an extension.
/// * `p > |G|` makes the lift back to `ℤ` unambiguous. Every quantity lifted in
///   [`super::table`] is a non-negative integer bounded by `χ(1) ≤ √|G|` or by
///   `χ(1)² ≤ |G|`, so its residue in `0..p` *is* its value. It also forces
///   `p ∤ |G|`, without which the class algebra would not be semisimple mod `p`
///   and the matrices would not be diagonalisable at all.
pub(super) fn working_prime(exponent: u64, floor: u64) -> Result<u64, CharacterError> {
    if exponent == 0 {
        return Err(CharacterError::Internal {
            detail: "exp G came out as 0".to_string(),
        });
    }
    let start = floor / exponent + 1;
    let mut last = 0u64;
    for k in start..start.saturating_add(PRIME_SEARCH_CANDIDATES) {
        let candidate = match k.checked_mul(exponent).and_then(|v| v.checked_add(1)) {
            Some(v) => v,
            None => break,
        };
        last = candidate;
        if candidate > floor && crate::modular::is_prime(candidate) {
            return Ok(candidate);
        }
    }
    Err(CharacterError::NoSuitablePrime {
        exponent,
        order: floor.to_string(),
        searched_to: last,
    })
}

/// An element of GF(p) of order **exactly** `e`, given `e | p − 1`.
///
/// This is the `w` that fixes the ring homomorphism `ℤ[ζ_e] → GF(p)`,
/// `ζ_e ↦ w`, used for the whole table. Any generator of the order-`e` subgroup
/// would do — a different choice permutes the rows by a Galois action and gives
/// an equally valid table — but it must be the *same* one for every class and
/// every character, or the values lifted for one row would belong to different
/// embeddings.
///
/// # The order is computed here and then checked, not assumed
///
/// An element of order a proper *divisor* of `e` would be a silent wrong
/// answer: the inverse DFT in [`super::table`] would reconstruct multiplicities
/// against the wrong roots of unity and the character values would be wrong
/// rather than absent. So `w = base^((p−1)/e)` is accepted only after
/// `w^(e/q) ≠ 1` has been confirmed for every prime `q | e` — which is exactly
/// the statement that the order is `e` and not a proper divisor of it — and
/// `w^e = 1` is asserted on the way out as well.
///
/// This is also why the root is built here in the prime field by hand rather
/// than taken from [`FiniteField`]: `FiniteField::generator` returns the
/// polynomial-basis generator (the class of `x`), which is primitive only when
/// the defining polynomial is, and nothing in this module would notice if it
/// were not.
pub(super) fn root_of_unity(p: u64, e: u64) -> Result<u64, CharacterError> {
    if e == 0 || (p - 1) % e != 0 {
        return Err(CharacterError::Internal {
            detail: format!("{e} does not divide p - 1 = {}", p - 1),
        });
    }
    if e == 1 {
        return Ok(1 % p);
    }
    let cofactor = (p - 1) / e;
    let primes = distinct_prime_factors(e);
    for base in 2..p {
        let w = pow_mod(base, cofactor, p);
        if w <= 1 {
            continue;
        }
        if primes.iter().all(|&q| pow_mod(w, e / q, p) != 1) {
            // Belt and braces: the loop above establishes that the order is not
            // a proper divisor of `e`; this establishes that it divides `e`.
            // Together they say it *is* `e`.
            if pow_mod(w, e, p) != 1 {
                return Err(CharacterError::Internal {
                    detail: format!("{w} has order not dividing {e} in GF({p})"),
                });
            }
            return Ok(w);
        }
    }
    Err(CharacterError::Internal {
        detail: format!("GF({p}) has no element of order {e} despite {e} | p - 1"),
    })
}

/// A row-major `u64` block as a matrix over GF(p).
fn matrix(
    field: &FiniteField,
    rows: usize,
    cols: usize,
    data: &[u64],
) -> Result<GfMatrix, CharacterError> {
    GfMatrix::from_u64(field, rows, cols, data).map_err(CharacterError::from)
}

/// A matrix over a prime field back to row-major `u64`.
fn entries(m: &GfMatrix) -> Result<Vec<u64>, CharacterError> {
    m.to_u64().ok_or_else(|| CharacterError::Internal {
        detail: "a matrix over a prime field did not render as u64 entries".to_string(),
    })
}

/// The roots in GF(p) of a polynomial given by ascending coefficients.
///
/// Via FLINT's `nmod_poly_factor`: the roots are the linear factors. Returned
/// sorted and without repetition, since an eigenvalue's multiplicity is not
/// wanted here — the eigenspace is computed from the kernel instead.
fn roots(p: u64, ascending: &[u64]) -> Vec<u64> {
    let degree = match ascending.iter().rposition(|&c| c % p != 0) {
        Some(d) => d,
        None => return Vec::new(),
    };
    if degree == 0 {
        return Vec::new();
    }
    let mut poly = FlintNmodPoly::new(p);
    for (i, &c) in ascending.iter().take(degree + 1).enumerate() {
        poly.set_coeff(i, c % p);
    }
    let mut factorisation = FlintNmodPolyFactor::new();
    factorisation.factor(&poly);
    let mut out = Vec::new();
    for i in 0..factorisation.len() {
        let factor = factorisation.poly_at(p, i);
        if factor.degree() != 1 {
            continue;
        }
        let a1 = factor.get_coeff(1);
        let a0 = factor.get_coeff(0);
        // x = -a0 / a1. `a1 != 0` because the factor has degree 1.
        if let Ok(inv) = inv_mod(a1, p) {
            let neg = (p - a0 % p) % p;
            out.push(mul_mod(neg, inv, p));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The intersection of two column spans, as a basis of columns.
///
/// `ker [U | V]` pairs the coefficient vectors `(x, y)` with `U x = −V y`, so
/// `U` applied to the top block of that kernel spans `U ∩ V`. The resulting
/// columns are independent: if `Σ c_j x_j = 0` then `Σ c_j (x_j, y_j)` is a
/// kernel vector of the form `(0, z)` with `V z = 0`, so `z = 0` too, and the
/// kernel basis was independent.
fn intersect(u: &GfMatrix, v: &GfMatrix) -> Result<GfMatrix, CharacterError> {
    let rows = u.nrows();
    let (a, b) = (u.ncols(), v.ncols());
    let field = u.field();
    if a == 0 || b == 0 {
        return GfMatrix::zeros(field, rows, 0).map_err(CharacterError::from);
    }
    let ud = entries(u)?;
    let vd = entries(v)?;
    let mut stacked = vec![0u64; rows * (a + b)];
    for i in 0..rows {
        for j in 0..a {
            stacked[i * (a + b) + j] = ud[i * a + j];
        }
        for j in 0..b {
            stacked[i * (a + b) + a + j] = vd[i * b + j];
        }
    }
    let kernel = matrix(field, rows, a + b, &stacked)?.nullspace()?;
    let t = kernel.ncols();
    if t == 0 {
        return GfMatrix::zeros(field, rows, 0).map_err(CharacterError::from);
    }
    let kd = entries(&kernel)?;
    let mut top = vec![0u64; a * t];
    for i in 0..a {
        for j in 0..t {
            top[i * t + j] = kd[i * t + j];
        }
    }
    let top = matrix(field, a, t, &top)?;
    u.mul(&top).map_err(CharacterError::from)
}

/// The common eigenvectors of a commuting family of matrices over GF(p), one
/// per conjugacy class.
///
/// `matrices` are the class multiplication matrices `M_k` as integer blocks
/// reduced mod `p`; `classes` is `r`, their common dimension. Because the `M_k`
/// commute and each is diagonalisable over GF(p), refining an invariant
/// subspace by the eigenspaces of the next matrix decomposes it completely, and
/// the `r` distinct algebra homomorphisms `Z(F_p G) → F_p` guarantee that the
/// family as a whole separates every line.
///
/// Returns `r` vectors of length `r`. Refuses with
/// [`CharacterError::SplittingIncomplete`] if the refinement stalls, which for
/// a `p` prime to `|G|` cannot happen.
pub(super) fn common_eigenvectors(
    p: u64,
    classes: usize,
    matrices: &[Vec<Vec<u64>>],
) -> Result<Vec<Vec<u64>>, CharacterError> {
    let field = FiniteField::prime(p)?;
    if classes == 0 {
        return Ok(Vec::new());
    }
    let mut parts = vec![GfMatrix::identity(&field, classes)?];

    for m in matrices {
        if parts.len() >= classes {
            break;
        }
        if m.len() != classes {
            return Err(CharacterError::Internal {
                detail: format!(
                    "a class multiplication matrix has {} rows, not {classes}",
                    m.len()
                ),
            });
        }
        let mut flat = Vec::with_capacity(classes * classes);
        for row in m {
            if row.len() != classes {
                return Err(CharacterError::Internal {
                    detail: format!(
                        "a class multiplication matrix row has {} entries, not {classes}",
                        row.len()
                    ),
                });
            }
            flat.extend(row.iter().map(|&v| v % p));
        }
        let mk = matrix(&field, classes, classes, &flat)?;

        let mut cp = Vec::new();
        for c in mk.charpoly()? {
            cp.push(c.as_u64().ok_or_else(|| CharacterError::Internal {
                detail: "a charpoly coefficient over a prime field was not a scalar".to_string(),
            })?);
        }
        let eigenvalues = roots(p, &cp);
        let mut eigenspaces = Vec::with_capacity(eigenvalues.len());
        let identity = GfMatrix::identity(&field, classes)?;
        for &lambda in &eigenvalues {
            let shifted = mk.sub(&identity.scalar_mul(&field.scalar(lambda))?)?;
            let kernel = shifted.nullspace()?;
            if kernel.ncols() > 0 {
                eigenspaces.push(kernel);
            }
        }

        let mut refined = Vec::with_capacity(parts.len());
        for part in &parts {
            if part.ncols() <= 1 {
                refined.push(part.clone());
                continue;
            }
            let mut covered = 0usize;
            for space in &eigenspaces {
                let piece = intersect(part, space)?;
                if piece.ncols() > 0 {
                    covered += piece.ncols();
                    refined.push(piece);
                }
            }
            // `M_k` preserves `part` (the matrices commute) and is
            // diagonalisable, so the eigenspace intersections must exhaust it.
            if covered != part.ncols() {
                return Err(CharacterError::Internal {
                    detail: format!(
                        "the eigenspaces of a class multiplication matrix cover {covered} of \
                         {} dimensions of an invariant subspace",
                        part.ncols()
                    ),
                });
            }
        }
        parts = refined;
    }

    if parts.len() != classes || parts.iter().any(|b| b.ncols() != 1) {
        return Err(CharacterError::SplittingIncomplete {
            found: parts.len(),
            classes,
        });
    }

    let mut out = Vec::with_capacity(classes);
    for part in &parts {
        out.push(entries(part)?);
    }
    Ok(out)
}
