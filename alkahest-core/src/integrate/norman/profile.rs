//! Opt-in phase timing for the Risch–Norman pipeline.
//!
//! The module is compiled unconditionally but is inert unless [`enable`] has
//! been called: every probe first reads one relaxed [`AtomicBool`], and only
//! then samples the clock.  That keeps the instrumentation out of the measured
//! path (a disabled probe is a load and a predictable branch) while still
//! letting the benchmark in `tests.rs` attribute runtime to a phase without a
//! separate build.
//!
//! It exists because there is no `perf` on the machines this is developed on,
//! and "where does the time actually go" is the question every optimisation
//! here has to answer first.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The phases the pipeline is broken into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Phase {
    /// `simplify` on the incoming integrand.
    Normalize,
    /// [`super::ring::build`] — collection, lattice reduction, structure checks.
    RingBuild,
    /// Reading the integrand into `ℚ(x, θ)`.
    ToRf,
    /// Choosing the monomial box and the logarithm candidates.
    AnsatzSetup,
    /// Differentiating every ansatz atom.
    AtomDerivs,
    /// The common denominator and the cleared columns.
    ClearDenoms,
    /// Building the coefficient matrix.
    MatrixBuild,
    /// The linear solve itself.
    Solve,
    /// Rebuilding the antiderivative expression.
    Rebuild,
    /// The `d/dx F = f` gate.
    Verify,
}

/// Number of variants of [`Phase`].
pub(crate) const N_PHASES: usize = 10;

impl Phase {
    /// Every phase, in pipeline order.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const ALL: [Phase; N_PHASES] = [
        Phase::Normalize,
        Phase::RingBuild,
        Phase::ToRf,
        Phase::AnsatzSetup,
        Phase::AtomDerivs,
        Phase::ClearDenoms,
        Phase::MatrixBuild,
        Phase::Solve,
        Phase::Rebuild,
        Phase::Verify,
    ];

    /// A short label for reports.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Phase::Normalize => "normalize",
            Phase::RingBuild => "ring_build",
            Phase::ToRf => "to_rf",
            Phase::AnsatzSetup => "ansatz_setup",
            Phase::AtomDerivs => "atom_derivs",
            Phase::ClearDenoms => "clear_denoms",
            Phase::MatrixBuild => "matrix_build",
            Phase::Solve => "solve",
            Phase::Rebuild => "rebuild",
            Phase::Verify => "verify",
        }
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TOTALS: RefCell<[Duration; N_PHASES]> =
        const { RefCell::new([Duration::ZERO; N_PHASES]) };
}

/// Turn instrumentation on for this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Discard everything accumulated so far on this thread.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn reset() {
    TOTALS.with(|t| *t.borrow_mut() = [Duration::ZERO; N_PHASES]);
    SHAPE.with(|s| *s.borrow_mut() = Shape::default());
}

/// Accumulated time per phase on this thread, in pipeline order.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn totals() -> [Duration; N_PHASES] {
    TOTALS.with(|t| *t.borrow())
}

/// The sizes each cap in this module is meant to bound.
///
/// Recorded so that `MAX_UNKNOWNS`, `MAX_EQUATIONS`, `MAX_DENOM_TERMS`,
/// `MAX_GENERATORS` and `MAX_CELLS` can be documented against a measured
/// maximum rather than a guess.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Shape {
    /// Distinct monomials in the cleared identity — rows of the system.
    pub equations: usize,
    /// Ansatz atoms — columns of the system.
    pub unknowns: usize,
    /// Terms in the cleared common denominator.
    pub denom_terms: usize,
    /// Tower generators, including the integration variable.
    pub generators: usize,
    /// Non-zero cells the sparse elimination held at its peak.
    pub peak_cells: usize,
}

thread_local! {
    static SHAPE: RefCell<Shape> = const { RefCell::new(Shape {
        equations: 0,
        unknowns: 0,
        denom_terms: 0,
        generators: 0,
        peak_cells: 0,
    }) };
}

/// Update one field of the recorded shape.
#[inline]
pub(crate) fn record(update: impl FnOnce(&mut Shape)) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    SHAPE.with(|s| update(&mut s.borrow_mut()));
}

/// The last recorded shape.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn shape() -> Shape {
    SHAPE.with(|s| *s.borrow())
}

/// Time `body`, charging it to `phase`.
///
/// When instrumentation is off this is `body()` plus one relaxed atomic load.
#[inline]
pub(crate) fn timed<T>(phase: Phase, body: impl FnOnce() -> T) -> T {
    if !ENABLED.load(Ordering::Relaxed) {
        return body();
    }
    let start = Instant::now();
    let out = body();
    let dt = start.elapsed();
    TOTALS.with(|t| t.borrow_mut()[phase as usize] += dt);
    out
}
