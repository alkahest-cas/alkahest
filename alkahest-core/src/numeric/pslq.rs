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

/// The width the search may honestly run at: no wider than the request, and no
/// wider than the narrowest input actually carries. Never below the 64-bit
/// engine minimum (the request has already been checked against it).
fn effective_search_bits(xs: &[Float], precision_bits: u32) -> u32 {
    let narrowest = xs.iter().map(Float::prec).min().unwrap_or(precision_bits);
    precision_bits.min(narrowest).max(64).min(16_384)
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
/// * `precision_bits` — the requested search width. The search never runs wider
///   than the inputs themselves: the effective width is the smaller of
///   `precision_bits` and the narrowest input's MPFR precision (floored at the
///   64-bit engine minimum), and the detection threshold is derived from that
///   effective width. Searching wider than the data zero-pads the inputs into
///   exact rationals, which both hides true relations (their residual sits at the
///   inputs' precision, far above a threshold set by the wider search) and
///   manufactures spurious ones among the padded values.
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
    let bits = effective_search_bits(xs, precision_bits);

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
    use rug::ops::Pow;

    fn zeta4_and_pi4(prec: u32) -> Vec<Float> {
        use rug::float::Constant;
        let pi = Float::with_val(prec, Constant::Pi);
        let pi4 = Float::with_val(prec, pi.clone().pow(4u32));
        // ζ(4) = π⁴/90, rounded to `prec` independently of π⁴.
        let z4 = Float::with_val(
            prec,
            Float::with_val(prec + 64, Constant::Pi).pow(4u32) / 90u32,
        );
        vec![z4, pi4]
    }

    #[test]
    fn search_is_capped_at_the_input_precision() {
        // 128-bit inputs searched at a requested 664 bits: the true relation's
        // residual is ~2⁻¹²⁸, far above a threshold derived from 664 bits, so
        // the uncapped search answered `None` (a false negative).
        let xs = zeta4_and_pi4(128);
        let rel = guess_integer_relation(&xs, 664, None).unwrap().unwrap();
        let mut rel: Vec<i64> = rel.iter().map(|z| z.to_i64().unwrap()).collect();
        if rel[1] < 0 {
            rel = rel.iter().map(|a| -a).collect();
        }
        assert_eq!(rel, vec![-90, 1]);
        assert_eq!(effective_search_bits(&xs, 664), 128);
    }

    #[test]
    fn effective_bits_never_drop_below_the_engine_minimum() {
        let xs = vec![Float::with_val(53, 0.5), Float::with_val(53, 0.25)];
        assert_eq!(effective_search_bits(&xs, 664), 64);
        assert_eq!(effective_search_bits(&zeta4_and_pi4(400), 300), 300);
    }

    #[test]
    fn minimal_polynomial_of_cbrt2_plus_sqrt3_at_input_precision() {
        // α = 2^(1/3) + 3^(1/2) has minimal polynomial
        // x⁶ − 9x⁴ − 4x³ + 27x² − 36x − 23.
        for prec in [200u32, 400, 830] {
            let a =
                Float::with_val(prec + 64, 2u32).cbrt() + Float::with_val(prec + 64, 3u32).sqrt();
            let xs: Vec<Float> = (0..7u32)
                .map(|k| Float::with_val(prec, a.clone().pow(k)))
                .collect();
            let rel = guess_integer_relation(&xs, 664, None).unwrap().unwrap();
            let mut rel: Vec<i64> = rel.iter().map(|z| z.to_i64().unwrap()).collect();
            if rel[6] < 0 {
                rel = rel.iter().map(|c| -c).collect();
            }
            assert_eq!(rel, vec![-23, -36, 27, -4, -9, 0, 1], "prec {prec}");
        }
    }

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
