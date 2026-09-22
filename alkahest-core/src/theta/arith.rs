//! Ball arithmetic on [`ComplexBall`] and [`RealBall`], evaluated by FLINT.
//!
//! This exists so that an input which is not a Gaussian rational can be built
//! *rigorously*. `rho = exp(2 pi i / 3)` is the motivating case: it is the
//! second classical checkpoint for the j-invariant, it is irrational, and
//! writing it as a decimal literal would quietly turn "j vanishes at rho" into
//! "j is small near something close to rho".
//!
//! Every operation returns a `Result`, because every operation needs the FLINT
//! backend and because a midpoint or radius can fall outside the range that
//! moves between FLINT and MPFR without rounding. Both of those are refusals,
//! not approximations.

use super::ball::{ComplexBall, RealBall};
use super::error::ThetaError;
use super::{backend, CBinop, CUnop, RBinop, RUnop, MAX_PRECISION_BITS, MIN_PRECISION_BITS};

pub(crate) fn check_precision(bits: u32) -> Result<(), ThetaError> {
    if !(MIN_PRECISION_BITS..=MAX_PRECISION_BITS).contains(&bits) {
        return Err(ThetaError::PrecisionOutOfRange {
            bits: u64::from(bits),
            max: MAX_PRECISION_BITS,
        });
    }
    Ok(())
}

macro_rules! real_binop {
    ($($name:ident => $op:ident),* $(,)?) => {$(
        /// Ball arithmetic at `prec` bits of working precision.
        pub fn $name(&self, other: &RealBall, prec: u32) -> Result<RealBall, ThetaError> {
            check_precision(prec)?;
            backend::rb_binop(RBinop::$op, self, other, prec)
        }
    )*};
}

macro_rules! real_unop {
    ($($name:ident => $op:ident),* $(,)?) => {$(
        /// Ball arithmetic at `prec` bits of working precision.
        pub fn $name(&self, prec: u32) -> Result<RealBall, ThetaError> {
            check_precision(prec)?;
            backend::rb_unop(RUnop::$op, self, prec)
        }
    )*};
}

impl RealBall {
    real_binop! {
        add => Add,
        sub => Sub,
        mul => Mul,
        div => Div,
        pow => Pow,
    }

    real_unop! {
        neg => Neg,
        abs => Abs,
        sqrt => Sqrt,
        exp => Exp,
        log => Log,
        gamma => Gamma,
    }

    /// An enclosure of `pi` at `prec` bits.
    pub fn pi(prec: u32) -> Result<RealBall, ThetaError> {
        check_precision(prec)?;
        backend::real_pi(prec)
    }
}

macro_rules! complex_binop {
    ($($name:ident => $op:ident),* $(,)?) => {$(
        /// Ball arithmetic at `prec` bits of working precision.
        pub fn $name(&self, other: &ComplexBall, prec: u32) -> Result<ComplexBall, ThetaError> {
            check_precision(prec)?;
            backend::cb_binop(CBinop::$op, self, other, prec)
        }
    )*};
}

macro_rules! complex_unop {
    ($($name:ident => $op:ident),* $(,)?) => {$(
        /// Ball arithmetic at `prec` bits of working precision.
        pub fn $name(&self, prec: u32) -> Result<ComplexBall, ThetaError> {
            check_precision(prec)?;
            backend::cb_unop(CUnop::$op, self, prec)
        }
    )*};
}

impl ComplexBall {
    complex_binop! {
        add => Add,
        sub => Sub,
        mul => Mul,
        div => Div,
        pow => Pow,
    }

    complex_unop! {
        neg => Neg,
        conj => Conj,
        sqrt => Sqrt,
        exp => Exp,
        log => Log,
    }

    /// `|self|`, as a real ball.
    pub fn abs(&self, prec: u32) -> Result<RealBall, ThetaError> {
        check_precision(prec)?;
        backend::cb_abs(self, prec)
    }

    /// An enclosure of `pi` on the real axis.
    pub fn pi(prec: u32) -> Result<ComplexBall, ThetaError> {
        Ok(ComplexBall::from_parts(
            RealBall::pi(prec)?,
            RealBall::exact_f64(0.0, prec),
        ))
    }

    /// The exact ball at `i`.
    pub fn imaginary_unit(prec: u32) -> ComplexBall {
        ComplexBall::exact_i64(0, 1, prec)
    }

    /// `exp(2 pi i / n)`, a primitive `n`-th root of unity, as an enclosure.
    ///
    /// Provided because the classical checkpoints for this module live at
    /// `i` and at `rho = exp(2 pi i / 3)`, and building `rho` out of decimal
    /// literals would be the one place a normalisation test could be fooled.
    pub fn root_of_unity(n: u32, prec: u32) -> Result<ComplexBall, ThetaError> {
        check_precision(prec)?;
        if n == 0 {
            return Err(ThetaError::DimensionMismatch {
                what: "root of unity order",
                expected: 1,
                got: 0,
            });
        }
        // Work a little above the requested precision so that the division and
        // the exponential do not eat into the caller's budget.
        let w = prec.saturating_add(32).min(MAX_PRECISION_BITS);
        let two_pi = RealBall::pi(w)?.mul(&RealBall::exact_i64(2, w), w)?;
        let angle = two_pi.div(&RealBall::exact_i64(i64::from(n), w), w)?;
        let arg = ComplexBall::from_parts(RealBall::exact_f64(0.0, w), angle);
        arg.exp(w)
    }
}
