//! Detect near-linear dependencies among floating scalars via an augmented lattice + LLL.
//!
//! Rows `eᵢ ⊕ ⌊β·xᵢ⌋` spanning ℤⁿ⁺¹ are reduced with [`crate::lattice::lattice_reduce_rows`]; short
//! vectors correlate with approximate integer relations \(\sumᵢ aᵢ xᵢ ≈ 0\).

use crate::errors::AlkahestError;
use crate::lattice::{lattice_reduce_rows, LatticeError};
use rug::ops::PowAssign;
use rug::{Assign, Float, Integer};
use std::cmp::Ordering;
use std::fmt;

/// Errors from [`guess_integer_relation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PslqError {
    TooFewCoordinates,
    AllZeroMagnitudes,
    PrecisionTooThin { bits: u32 },
    Lattice(LatticeError),
}

impl fmt::Display for PslqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PslqError::TooFewCoordinates => write!(
                f,
                "guess_integer_relation needs at least two floating scalars"
            ),
            PslqError::AllZeroMagnitudes => write!(
                f,
                "scaled magnitudes vanished (check precision or literals)"
            ),
            PslqError::PrecisionTooThin { bits } => {
                write!(f, "precision_bits ({bits}); require ≥64 MPFR bits")
            }
            PslqError::Lattice(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PslqError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PslqError::Lattice(e) => Some(e),
            _ => None,
        }
    }
}

impl AlkahestError for PslqError {
    fn code(&self) -> &'static str {
        match self {
            PslqError::TooFewCoordinates => "E-PSLQ-001",
            PslqError::AllZeroMagnitudes => "E-PSLQ-002",
            PslqError::PrecisionTooThin { .. } => "E-PSLQ-003",
            PslqError::Lattice(e) => e.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            PslqError::TooFewCoordinates => Some("pass [x₀,…,x_{n−1}] with n ≥ 2"),
            PslqError::AllZeroMagnitudes => {
                Some("use higher-precision inputs (strings or MPFR literals)")
            }
            PslqError::PrecisionTooThin { .. } => {
                Some("raise precision_bits — ≈664 bits ≈ 200 decimal digits")
            }
            PslqError::Lattice(e) => e.remediation(),
        }
    }
}

fn lin_residual(bits: u32, coeffs: &[Integer], xs: &[Float]) -> Float {
    let mut acc = Float::with_val(bits, 0);
    for (c, xv) in coeffs.iter().zip(xs.iter()) {
        let mut term = Float::with_val(bits, c);
        term *= Float::with_val(bits, xv);
        acc += term;
    }
    acc.abs_mut();
    acc
}

/// Search for integers `(a₀,…,a_{n−1})` with \(|\sum_i a_i x_i|\) below a precision-derived threshold.
///
/// * `max_abs_coeff` — optional filter rejecting candidates with any `|a_i|` above the bound.
pub fn guess_integer_relation(
    xs: &[Float],
    precision_bits: u32,
    max_abs_coeff: Option<u128>,
) -> Result<Option<Vec<Integer>>, PslqError> {
    let n = xs.len();
    if n < 2 {
        return Err(PslqError::TooFewCoordinates);
    }
    if precision_bits < 64 {
        return Err(PslqError::PrecisionTooThin {
            bits: precision_bits,
        });
    }
    let bits = precision_bits.min(16_384);

    let mut normed: Vec<Float> = xs.iter().map(|xv| Float::with_val(bits, xv)).collect();
    let mut ymax = Float::with_val(bits, 0);
    for v in &normed {
        let mut cp = Float::with_val(bits, v);
        cp.abs_mut();
        if cp.partial_cmp(&ymax) == Some(Ordering::Greater) {
            ymax.assign(&cp);
        }
    }
    let zero = Float::with_val(bits, 0);
    if ymax.partial_cmp(&zero) == Some(Ordering::Equal) {
        return Err(PslqError::AllZeroMagnitudes);
    }

    for v in &mut normed {
        let mut quot = Float::with_val(bits, &*v);
        quot /= &ymax;
        v.assign(&quot);
    }

    let shift_amt = (bits * 3 / 4).min(1536);
    let mut scale = Integer::from(1u32);
    scale <<= shift_amt;

    let mut augmented: Vec<Vec<Integer>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = vec![Integer::from(0); n + 1];
        row[i] = Integer::from(1);
        let mut xf = Float::with_val(bits, &normed[i]);
        xf *= Float::with_val(bits, &scale);
        let tail = xf.to_integer().ok_or(PslqError::AllZeroMagnitudes)?;
        row[n].assign(&tail);
        augmented.push(row);
    }

    let reduced = lattice_reduce_rows(&augmented).map_err(PslqError::Lattice)?;

    let mut tol = Float::with_val(bits, 2);
    let exp_lim = ((-((bits as f64) * 0.75).floor()) as i32).min(-1);
    tol.pow_assign(exp_lim);
    tol *= Float::with_val(bits, (n.max(1)) as i32);

    let mut best: Option<(Vec<Integer>, Float)> = None;
    for row in &reduced {
        let coeffs: Vec<Integer> = row.iter().take(n).cloned().collect();
        if coeffs.iter().all(Integer::is_zero) {
            continue;
        }
        if let Some(limit) = max_abs_coeff {
            let lim = Integer::from(limit);
            let mut ok = true;
            for z in &coeffs {
                let mut a = z.clone();
                a.abs_mut();
                if a.cmp(&lim) == Ordering::Greater {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
        }
        let resid = lin_residual(bits, &coeffs, &normed);
        let take = match &best {
            None => true,
            Some((_, r0)) => resid.partial_cmp(r0) == Some(Ordering::Less),
        };
        if take {
            best = Some((coeffs, resid));
        }
    }

    Ok(best.and_then(|(v, r)| {
        if r.partial_cmp(&tol) != Some(Ordering::Greater) {
            Some(v)
        } else {
            None
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_on_1_2_3() {
        let bits = 256u32;
        let xs = vec![
            Float::with_val(bits, 1),
            Float::with_val(bits, 2),
            Float::with_val(bits, 3),
        ];
        let rel = guess_integer_relation(&xs, bits, Some(10_000))
            .unwrap()
            .unwrap();
        let r = lin_residual(bits, &rel, &xs);
        let mut tol = Float::with_val(bits, 2);
        tol.pow_assign(-((bits as f64 * 0.75).floor() as i32));
        tol *= Float::with_val(bits, 3);
        assert!(
            r.partial_cmp(&tol) != Some(Ordering::Greater),
            "residual {r:?} tol {tol:?}"
        );
    }
}
