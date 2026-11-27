//! Time integrators driving a [`MolProgram`] natively: no Python, no
//! per-point expressions — every right-hand side and Jacobian evaluation is
//! one pass of the program's stencil kernels.
//!
//! * [`MolMethod::Rk45`] — Dormand–Prince 5(4) with the usual step-size
//!   controller (the same tableau and controller as SciPy's `RK45`).
//! * [`MolMethod::Bdf`] — variable-order (1–5), quasi-constant-step BDF in
//!   Nordsieck-like backward-difference form (Shampine & Reichelt's `ode15s`
//!   scheme, as in SciPy's `BDF`), with the program's exact sparse Jacobian,
//!   a banded LU after reverse Cuthill–McKee ordering, and a diagonal mass
//!   matrix so semi-explicit index-1 DAEs (boundary conditions kept as
//!   algebraic constraints) integrate directly.
//! * [`MolMethod::Euler`], [`MolMethod::Rk4`] — fixed step, for stability
//!   studies.
//!
//! The BDF Newton iterations solve with the iteration matrix `M - c J`
//! either directly (the banded LU) or iteratively (ILU(0)-preconditioned
//! restarted GMRES on the same exact sparse matrix, [`super::krylov`]);
//! [`integrate_with`] chooses by size ([`MolLinearSolver::Auto`]) unless told.

#![allow(clippy::needless_range_loop)]

use super::banded::{rcm_order, BandedLu};
use super::krylov::{gmres, Ilu0};
use super::{CsrMatrix, MolError, MolProgram};

/// Integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolMethod {
    Rk45,
    Bdf,
    Euler,
    Rk4,
}

/// Options for [`integrate`].
#[derive(Debug, Clone)]
pub struct MolIntegrateOptions {
    pub method: MolMethod,
    pub rtol: f64,
    pub atol: f64,
    /// Initial step (adaptive methods; `None` selects one), or the fixed step.
    pub h0: Option<f64>,
    pub max_step: f64,
    pub max_steps: usize,
    /// BDF: largest order (1–5).
    pub max_order: usize,
    /// BDF: refuse a factorisation whose band storage exceeds this.
    pub max_band_bytes: usize,
}

impl Default for MolIntegrateOptions {
    fn default() -> Self {
        MolIntegrateOptions {
            method: MolMethod::Rk45,
            rtol: 1e-6,
            atol: 1e-9,
            h0: None,
            max_step: f64::INFINITY,
            max_steps: 10_000_000,
            max_order: 5,
            max_band_bytes: 2 << 30,
        }
    }
}

/// Work counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MolStats {
    pub steps: usize,
    pub rejected: usize,
    pub rhs_evals: usize,
    pub jac_evals: usize,
    pub factorizations: usize,
    pub newton_iters: usize,
}

/// Why an integration stopped early.
#[derive(Debug, Clone, PartialEq)]
pub struct StepFailure {
    pub t: f64,
    pub message: String,
}

/// States at the requested output times.
#[derive(Debug, Clone)]
pub struct MolSolution {
    pub t: Vec<f64>,
    /// Row-major `t.len() × n_states`.
    pub y: Vec<f64>,
    pub stats: MolStats,
    /// Band dimensions of the BDF iteration matrix `(n, kl, ku)`.
    pub band: Option<(usize, usize, usize)>,
}

/// Linear solver for the Newton iterations of the BDF method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MolLinearSolver {
    /// Choose by size: the banded direct solver while its factorisation is
    /// cheap, GMRES + ILU(0) once the band is wide (large 2-D and 3-D grids)
    /// or its storage would exceed `max_band_bytes`.
    #[default]
    Auto,
    /// Banded LU after reverse Cuthill–McKee ordering.
    Direct,
    /// Restarted GMRES, left-preconditioned by ILU(0) of the iteration
    /// matrix, in the integrator's weighted RMS norm.
    Krylov,
}

/// Linear-solver options for [`integrate_with`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MolSolverOptions {
    pub linear_solver: MolLinearSolver,
    /// GMRES restart length.
    pub gmres_restart: usize,
    /// GMRES iterations allowed per linear solve.
    pub gmres_max_iters: usize,
    /// The GMRES tolerance as a fraction of the Newton tolerance (in the
    /// same weighted RMS norm; SUNDIALS CVODE's `eplifac`).
    pub krylov_tol_factor: f64,
    /// [`MolLinearSolver::Auto`]: use GMRES when the system has at least
    /// this many states ...
    pub auto_min_states: usize,
    /// ... and the reordered band has at least this lower bandwidth.
    pub auto_min_bandwidth: usize,
}

impl Default for MolSolverOptions {
    fn default() -> Self {
        MolSolverOptions {
            linear_solver: MolLinearSolver::Auto,
            gmres_restart: 30,
            gmres_max_iters: 120,
            krylov_tol_factor: 0.05,
            auto_min_states: AUTO_MIN_STATES,
            auto_min_bandwidth: AUTO_MIN_BANDWIDTH,
        }
    }
}

/// Default [`MolSolverOptions::auto_min_states`].
pub const AUTO_MIN_STATES: usize = 2000;
/// Default [`MolSolverOptions::auto_min_bandwidth`].
pub const AUTO_MIN_BANDWIDTH: usize = 32;

/// Linear-solver work counters of a BDF integration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct MolLinearStats {
    /// The solver the integration started with (`Direct` or `Krylov`);
    /// `None` for the explicit methods.
    pub solver: Option<MolLinearSolver>,
    /// Linear solves (one per Newton iteration).
    pub linear_solves: usize,
    /// GMRES iterations, summed over all solves.
    pub krylov_iters: usize,
    /// GMRES solves that missed their tolerance (each one counts as a
    /// failed Newton iteration, and the step is retried exactly as after a
    /// singular factorisation).
    pub krylov_failures: usize,
    /// ILU(0) factorisations.
    pub preconditioner_setups: usize,
    /// Times the integration switched from GMRES to the direct solver
    /// (ILU breakdown, or repeated GMRES failures with the band affordable).
    pub direct_fallbacks: usize,
}

fn rms(v: impl Iterator<Item = f64>, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    (v.map(|x| x * x).sum::<f64>() / n as f64).sqrt()
}

struct Ctx<'a> {
    prog: &'a MolProgram,
    stats: MolStats,
    lin: MolLinearStats,
}

impl Ctx<'_> {
    fn f(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), MolError> {
        self.stats.rhs_evals += 1;
        self.prog.rhs(t, y, out)
    }
}

fn fail(t: f64, m: impl Into<String>) -> MolError {
    MolError::Integration(format!("at t = {t}: {}", m.into()))
}

/// Integrate `M y' = f(t, y)` from `(t0, y0)`, returning the state at each
/// time of `t_eval` (increasing, all `≥ t0`).
pub fn integrate(
    prog: &MolProgram,
    t0: f64,
    y0: &[f64],
    t_eval: &[f64],
    opts: &MolIntegrateOptions,
) -> Result<MolSolution, MolError> {
    integrate_with(prog, t0, y0, t_eval, opts, &MolSolverOptions::default()).map(|r| r.0)
}

/// [`integrate`] with explicit linear-solver options; also returns the
/// linear-solver work counters.
pub fn integrate_with(
    prog: &MolProgram,
    t0: f64,
    y0: &[f64],
    t_eval: &[f64],
    opts: &MolIntegrateOptions,
    solver: &MolSolverOptions,
) -> Result<(MolSolution, MolLinearStats), MolError> {
    prog.check_y(y0)?;
    if t_eval.is_empty() {
        return Err(MolError::BadArgument("t_eval is empty".into()));
    }
    if t_eval.windows(2).any(|w| w[1] < w[0]) || t_eval[0] < t0 || !t0.is_finite() {
        return Err(MolError::BadArgument(
            "t_eval must be increasing and start at or after t0".into(),
        ));
    }
    if !(opts.rtol > 0.0 && opts.atol >= 0.0) {
        return Err(MolError::BadArgument(
            "rtol must be > 0 and atol >= 0".into(),
        ));
    }
    if !(solver.krylov_tol_factor > 0.0 && solver.krylov_tol_factor.is_finite())
        || solver.gmres_restart == 0
        || solver.gmres_max_iters == 0
    {
        return Err(MolError::BadArgument(
            "krylov_tol_factor must be finite and > 0, and gmres_restart and \
             gmres_max_iters at least 1"
                .into(),
        ));
    }
    let has_alg = prog.algebraic().iter().any(|&a| a);
    if has_alg && opts.method != MolMethod::Bdf {
        return Err(MolError::BadArgument(
            "the system has algebraic components (a DAE); use the BDF method".into(),
        ));
    }
    let mut ctx = Ctx {
        prog,
        stats: MolStats::default(),
        lin: MolLinearStats::default(),
    };
    let mut sol = MolSolution {
        t: Vec::with_capacity(t_eval.len()),
        y: Vec::with_capacity(t_eval.len() * y0.len()),
        stats: MolStats::default(),
        band: None,
    };
    match opts.method {
        MolMethod::Rk45 => rk45(&mut ctx, t0, y0, t_eval, opts, &mut sol)?,
        MolMethod::Euler | MolMethod::Rk4 => fixed(&mut ctx, t0, y0, t_eval, opts, &mut sol)?,
        MolMethod::Bdf => bdf(&mut ctx, t0, y0, t_eval, opts, solver, &mut sol)?,
    }
    sol.stats = ctx.stats;
    Ok((sol, ctx.lin))
}

#[allow(clippy::too_many_arguments)]
fn select_initial_step(
    ctx: &mut Ctx<'_>,
    t0: f64,
    y0: &[f64],
    f0: &[f64],
    order: f64,
    rtol: f64,
    atol: f64,
    mask: Option<&[bool]>,
) -> Result<f64, MolError> {
    let n = y0.len();
    if n == 0 {
        return Ok(f64::INFINITY);
    }
    let keep = |i: usize| mask.map_or(true, |m| !m[i]);
    let nn = (0..n).filter(|&i| keep(i)).count().max(1);
    let scale: Vec<f64> = y0.iter().map(|y| atol + y.abs() * rtol).collect();
    let d0 = rms((0..n).filter(|&i| keep(i)).map(|i| y0[i] / scale[i]), nn);
    let d1 = rms((0..n).filter(|&i| keep(i)).map(|i| f0[i] / scale[i]), nn);
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    let y1: Vec<f64> = y0
        .iter()
        .zip(f0)
        .enumerate()
        .map(|(i, (y, f))| if keep(i) { y + h0 * f } else { *y })
        .collect();
    let mut f1 = vec![0.0; n];
    ctx.f(t0 + h0, &y1, &mut f1)?;
    let d2 = rms(
        (0..n)
            .filter(|&i| keep(i))
            .map(|i| (f1[i] - f0[i]) / scale[i]),
        nn,
    ) / h0;
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (order + 1.0))
    };
    Ok((100.0 * h0).min(h1))
}

fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

// ---------------------------------------------------------------------------
// Dormand–Prince 5(4)
// ---------------------------------------------------------------------------

const DP_C: [f64; 6] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0];
const DP_A: [[f64; 5]; 6] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
];
const DP_B: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];
const DP_E: [f64; 7] = [
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0,
];

fn rk45(
    ctx: &mut Ctx<'_>,
    t0: f64,
    y0: &[f64],
    t_eval: &[f64],
    opts: &MolIntegrateOptions,
    sol: &mut MolSolution,
) -> Result<(), MolError> {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut f = vec![0.0; n];
    ctx.f(t, &y, &mut f)?;
    let mut h_abs = match opts.h0 {
        Some(h) => h,
        None => select_initial_step(ctx, t, &y, &f, 4.0, opts.rtol, opts.atol, None)?,
    }
    .min(opts.max_step);
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; 7];
    let mut ytmp = vec![0.0; n];
    let mut ynew = vec![0.0; n];
    let mut next = 0;
    while next < t_eval.len() && t_eval[next] <= t {
        sol.t.push(t_eval[next]);
        sol.y.extend_from_slice(&y);
        next += 1;
    }
    let exponent = -1.0 / 5.0;
    while next < t_eval.len() {
        let target = t_eval[next];
        let min_step = 10.0 * (next_up(t) - t).abs();
        let mut rejected = false;
        loop {
            if h_abs < min_step {
                return Err(fail(t, "step size underflow"));
            }
            let mut h = h_abs;
            let mut t_new = t + h;
            if t_new >= target {
                t_new = target;
                h = t_new - t;
            }
            k[0].copy_from_slice(&f);
            for s in 1..6 {
                for i in 0..n {
                    let mut acc = 0.0;
                    for (j, kj) in k.iter().enumerate().take(s) {
                        acc += DP_A[s][j] * kj[i];
                    }
                    ytmp[i] = y[i] + h * acc;
                }
                ctx.f(t + DP_C[s] * h, &ytmp, &mut k[s])?;
            }
            for i in 0..n {
                let mut acc = 0.0;
                for (j, kj) in k.iter().enumerate().take(6) {
                    acc += DP_B[j] * kj[i];
                }
                ynew[i] = y[i] + h * acc;
            }
            ctx.f(t_new, &ynew, &mut k[6])?;
            let err = rms(
                (0..n).map(|i| {
                    let mut e = 0.0;
                    for (j, kj) in k.iter().enumerate() {
                        e += DP_E[j] * kj[i];
                    }
                    let sc = opts.atol + y[i].abs().max(ynew[i].abs()) * opts.rtol;
                    e * h / sc
                }),
                n,
            );
            if !err.is_finite() || !finite(&ynew) {
                ctx.stats.rejected += 1;
                h_abs *= 0.2;
                rejected = true;
                continue;
            }
            if err < 1.0 {
                let mut factor = if err == 0.0 {
                    10.0
                } else {
                    (0.9 * err.powf(exponent)).min(10.0)
                };
                if rejected {
                    factor = factor.min(1.0);
                }
                // Keep the controller's step, not the one shortened to land
                // on an output time.
                if h < h_abs {
                    h_abs = h_abs.max(h * factor);
                } else {
                    h_abs *= factor;
                }
                h_abs = h_abs.min(opts.max_step);
                t = t_new;
                std::mem::swap(&mut y, &mut ynew);
                f.copy_from_slice(&k[6]);
                ctx.stats.steps += 1;
                break;
            }
            ctx.stats.rejected += 1;
            h_abs *= (0.9 * err.powf(exponent)).max(0.2);
            rejected = true;
        }
        if ctx.stats.steps > opts.max_steps {
            return Err(fail(t, format!("more than {} steps", opts.max_steps)));
        }
        while next < t_eval.len() && t_eval[next] <= t {
            sol.t.push(t_eval[next]);
            sol.y.extend_from_slice(&y);
            next += 1;
        }
    }
    Ok(())
}

fn next_up(t: f64) -> f64 {
    if t.is_nan() || t == f64::INFINITY {
        return t;
    }
    if t == 0.0 {
        return f64::from_bits(1);
    }
    let b = t.to_bits();
    f64::from_bits(if t > 0.0 { b + 1 } else { b - 1 })
}

// ---------------------------------------------------------------------------
// Fixed step
// ---------------------------------------------------------------------------

fn fixed(
    ctx: &mut Ctx<'_>,
    t0: f64,
    y0: &[f64],
    t_eval: &[f64],
    opts: &MolIntegrateOptions,
    sol: &mut MolSolution,
) -> Result<(), MolError> {
    let dt = opts
        .h0
        .filter(|h| *h > 0.0 && h.is_finite())
        .ok_or_else(|| MolError::BadArgument("a fixed-step method needs dt > 0".into()))?;
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    for &target in t_eval {
        while target - t > 1e-12 * dt.max(t.abs() * 1e-4) {
            let h = dt.min(target - t);
            match opts.method {
                MolMethod::Euler => {
                    ctx.f(t, &y, &mut k1)?;
                    for i in 0..n {
                        y[i] += h * k1[i];
                    }
                }
                _ => {
                    ctx.f(t, &y, &mut k1)?;
                    for i in 0..n {
                        tmp[i] = y[i] + 0.5 * h * k1[i];
                    }
                    ctx.f(t + 0.5 * h, &tmp, &mut k2)?;
                    for i in 0..n {
                        tmp[i] = y[i] + 0.5 * h * k2[i];
                    }
                    ctx.f(t + 0.5 * h, &tmp, &mut k3)?;
                    for i in 0..n {
                        tmp[i] = y[i] + h * k3[i];
                    }
                    ctx.f(t + h, &tmp, &mut k4)?;
                    for i in 0..n {
                        y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                    }
                }
            }
            t += h;
            ctx.stats.steps += 1;
            if ctx.stats.steps > opts.max_steps {
                return Err(fail(t, format!("more than {} steps", opts.max_steps)));
            }
        }
        t = t.max(target);
        sol.t.push(target);
        sol.y.extend_from_slice(&y);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// BDF (variable order 1-5, quasi-constant step, backward differences)
// ---------------------------------------------------------------------------

const MAX_ORDER: usize = 5;
const NEWTON_MAXITER: usize = 4;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// `R(order, factor)` of Shampine & Reichelt: the change-of-step matrix of
/// the backward-difference array.
fn compute_r(order: usize, factor: f64) -> Vec<Vec<f64>> {
    let m = order + 1;
    let mut mm = vec![vec![0.0; m]; m];
    for (i, row) in mm.iter_mut().enumerate().skip(1) {
        for (j, v) in row.iter_mut().enumerate().skip(1) {
            *v = (i as f64 - 1.0 - factor * j as f64) / i as f64;
        }
    }
    mm[0].iter_mut().for_each(|v| *v = 1.0);
    // Cumulative product down the columns.
    for i in 1..m {
        for j in 0..m {
            mm[i][j] *= mm[i - 1][j];
        }
    }
    mm
}

fn change_d(d: &mut [Vec<f64>], order: usize, factor: f64) {
    let r = compute_r(order, factor);
    let u = compute_r(order, 1.0);
    let m = order + 1;
    // RU = R · U ; D[:m] = RUᵀ · D[:m]
    let mut ru = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..m {
                s += r[i][k] * u[k][j];
            }
            ru[i][j] = s;
        }
    }
    let n = d[0].len();
    let mut newd = vec![vec![0.0; n]; m];
    for (a, row) in newd.iter_mut().enumerate() {
        for (b, db) in d.iter().enumerate().take(m) {
            let c = ru[b][a];
            if c != 0.0 {
                for (x, y) in row.iter_mut().zip(db) {
                    *x += c * y;
                }
            }
        }
    }
    for (a, row) in newd.into_iter().enumerate() {
        d[a] = row;
    }
}

struct Iteration {
    perm: Vec<usize>,
    kl: usize,
    ku: usize,
}

/// A factored (or preconditioned) iteration matrix.
enum Lin {
    Banded(BandedLu),
    Krylov { a: CsrMatrix, ilu: Ilu0 },
}

/// GMRES failures in a row (with no accepted step between) after which the
/// integration switches to the direct solver, when the band fits.
const KRYLOV_FAILS_BEFORE_FALLBACK: usize = 3;

#[allow(clippy::too_many_arguments)]
fn bdf(
    ctx: &mut Ctx<'_>,
    t0: f64,
    y0: &[f64],
    t_eval: &[f64],
    opts: &MolIntegrateOptions,
    solver: &MolSolverOptions,
    sol: &mut MolSolution,
) -> Result<(), MolError> {
    let n = y0.len();
    let max_order = opts.max_order.clamp(1, MAX_ORDER);
    let alg: Vec<bool> = ctx.prog.algebraic().to_vec();
    let has_alg = alg.iter().any(|&a| a);
    let n_diff = alg.iter().filter(|&&a| !a).count().max(1);
    let mass: Vec<f64> = alg.iter().map(|&a| if a { 0.0 } else { 1.0 }).collect();
    let eps = f64::EPSILON;
    let rtol = opts.rtol.max(100.0 * eps);
    let atol = opts.atol;
    let newton_tol = (10.0 * eps / rtol).max(0.03f64.min(rtol.sqrt()));

    let kappa = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];
    let mut gamma = [0.0f64; MAX_ORDER + 1];
    for k in 1..=MAX_ORDER {
        gamma[k] = gamma[k - 1] + 1.0 / k as f64;
    }
    let alpha: Vec<f64> = (0..=MAX_ORDER)
        .map(|k| (1.0 - kappa[k]) * gamma[k])
        .collect();
    let error_const: Vec<f64> = (0..=MAX_ORDER)
        .map(|k| kappa[k] * gamma[k] + 1.0 / (k as f64 + 1.0))
        .collect();

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut f = vec![0.0; n];
    if has_alg {
        consistent_init(ctx, t, &mut y, &alg, newton_tol * 1e-3)?;
    }
    ctx.f(t, &y, &mut f)?;
    let mut h_abs = match opts.h0 {
        Some(h) => h,
        None => select_initial_step(ctx, t, &y, &f, 1.0, rtol, atol, Some(&alg))?,
    }
    .min(opts.max_step);

    let mut d: Vec<Vec<f64>> = vec![vec![0.0; n]; MAX_ORDER + 3];
    d[0].copy_from_slice(&y);
    for i in 0..n {
        d[1][i] = if alg[i] { 0.0 } else { f[i] * h_abs };
    }
    let mut order = 1usize;
    let mut n_equal_steps = 0usize;

    let mut jac = ctx.prog.jacobian(t, &y)?;
    ctx.stats.jac_evals += 1;
    // The iteration matrix pattern is the Jacobian's plus the diagonal.
    let iter_pattern = with_diagonal(&jac);
    let perm = rcm_order(&iter_pattern);
    let (kl, ku) = BandedLu::bandwidths(&iter_pattern, &perm);
    let band_fits = BandedLu::storage_bytes(n, kl, ku) <= opts.max_band_bytes;
    let mut use_krylov = match solver.linear_solver {
        MolLinearSolver::Direct => false,
        MolLinearSolver::Krylov => true,
        MolLinearSolver::Auto => {
            !band_fits || (n >= solver.auto_min_states && kl >= solver.auto_min_bandwidth)
        }
    };
    if !use_krylov && !band_fits {
        return Err(MolError::Integration(format!(
            "the BDF iteration matrix needs {} MB of band storage (n = {n}, kl = {kl}, ku = {ku}); \
             refine less, use the Krylov linear solver or an explicit method",
            BandedLu::storage_bytes(n, kl, ku) >> 20
        )));
    }
    ctx.lin.solver = Some(if use_krylov {
        MolLinearSolver::Krylov
    } else {
        MolLinearSolver::Direct
    });
    sol.band = Some((n, kl, ku));
    let it = Iteration { perm, kl, ku };
    let mut lu: Option<Lin> = None;
    let mut current_jac = true;
    let mut krylov_fails_in_row = 0usize;
    let lin_tol = solver.krylov_tol_factor * newton_tol;

    let mut next = 0;
    while next < t_eval.len() && t_eval[next] <= t {
        sol.t.push(t_eval[next]);
        sol.y.extend_from_slice(&y);
        next += 1;
    }
    let t_bound = *t_eval.last().expect("non-empty");
    let mut fbuf = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    let mut ypred = vec![0.0; n];
    let mut psi = vec![0.0; n];
    let mut ynew = vec![0.0; n];
    let mut dcorr = vec![0.0; n];
    let mut scale = vec![0.0; n];
    let mut weight = vec![0.0; n];

    while next < t_eval.len() {
        let min_step = 10.0 * (next_up(t) - t).abs();
        if h_abs > opts.max_step {
            change_d(&mut d, order, opts.max_step / h_abs);
            h_abs = opts.max_step;
            n_equal_steps = 0;
            lu = None;
        } else if h_abs < min_step {
            change_d(&mut d, order, min_step / h_abs);
            h_abs = min_step;
            n_equal_steps = 0;
            lu = None;
        }
        let mut accepted = false;
        let mut n_iter = 0usize;
        let mut error_norm = 0.0;
        let mut t_new = t;
        while !accepted {
            if h_abs < min_step {
                return Err(fail(t, "step size underflow (BDF)"));
            }
            let mut h = h_abs;
            t_new = t + h;
            if t_new > t_bound {
                t_new = t_bound;
                change_d(&mut d, order, (t_new - t).abs() / h_abs);
                n_equal_steps = 0;
                lu = None;
            }
            h = t_new - t;
            h_abs = h.abs();
            for i in 0..n {
                let mut s = 0.0;
                for dk in d.iter().take(order + 1) {
                    s += dk[i];
                }
                ypred[i] = s;
                scale[i] = atol + rtol * s.abs();
            }
            for i in 0..n {
                let mut s = 0.0;
                for k in 1..=order {
                    s += d[k][i] * gamma[k];
                }
                psi[i] = s / alpha[order];
            }
            let c = h / alpha[order];
            if use_krylov {
                for i in 0..n {
                    weight[i] = 1.0 / scale[i];
                }
            }
            let mut converged = false;
            loop {
                if lu.is_none() {
                    let a = iteration_matrix(&jac, &mass, c);
                    ctx.stats.factorizations += 1;
                    if use_krylov {
                        ctx.lin.preconditioner_setups += 1;
                        match Ilu0::factor(&a) {
                            Some(ilu) => lu = Some(Lin::Krylov { a, ilu }),
                            None if band_fits => {
                                // ILU breakdown (a zero pivot): go direct.
                                use_krylov = false;
                                ctx.lin.direct_fallbacks += 1;
                                lu = BandedLu::factor(&a, &it.perm, it.kl, it.ku).map(Lin::Banded);
                            }
                            None => {}
                        }
                    } else {
                        lu = BandedLu::factor(&a, &it.perm, it.kl, it.ku).map(Lin::Banded);
                    }
                    if lu.is_none() {
                        // Singular: shrink the step.
                        break;
                    }
                }
                let lu_ref = lu.as_ref().expect("factored");
                // Newton iteration.
                ynew.copy_from_slice(&ypred);
                dcorr.iter_mut().for_each(|v| *v = 0.0);
                let mut dy_norm_old: Option<f64> = None;
                converged = false;
                n_iter = 0;
                for kk in 0..NEWTON_MAXITER {
                    n_iter = kk + 1;
                    ctx.f(t_new, &ynew, &mut fbuf)?;
                    ctx.stats.newton_iters += 1;
                    if !finite(&fbuf) {
                        break;
                    }
                    for i in 0..n {
                        rhs[i] = c * fbuf[i] - mass[i] * (psi[i] + dcorr[i]);
                    }
                    ctx.lin.linear_solves += 1;
                    match lu_ref {
                        Lin::Banded(b) => b.solve(&mut rhs),
                        Lin::Krylov { a, ilu } => {
                            let out = gmres(
                                a,
                                ilu,
                                &weight,
                                &mut rhs,
                                lin_tol,
                                solver.gmres_restart,
                                solver.gmres_max_iters,
                            );
                            ctx.lin.krylov_iters += out.iterations;
                            if !out.converged {
                                ctx.lin.krylov_failures += 1;
                                krylov_fails_in_row += 1;
                                break;
                            }
                        }
                    }
                    let dy_norm = rms((0..n).map(|i| rhs[i] / scale[i]), n);
                    let rate = dy_norm_old.map(|o| dy_norm / o);
                    if let Some(r) = rate {
                        if r >= 1.0
                            || r.powi((NEWTON_MAXITER - kk) as i32) / (1.0 - r) * dy_norm
                                > newton_tol
                        {
                            break;
                        }
                    }
                    for i in 0..n {
                        ynew[i] += rhs[i];
                        dcorr[i] += rhs[i];
                    }
                    if dy_norm == 0.0 || rate.is_some_and(|r| r / (1.0 - r) * dy_norm < newton_tol)
                    {
                        converged = true;
                        break;
                    }
                    dy_norm_old = Some(dy_norm);
                }
                if converged || current_jac {
                    break;
                }
                jac = ctx.prog.jacobian(t_new, &ypred)?;
                ctx.stats.jac_evals += 1;
                current_jac = true;
                lu = None;
            }
            if !converged {
                ctx.stats.rejected += 1;
                h_abs *= 0.5;
                change_d(&mut d, order, 0.5);
                n_equal_steps = 0;
                lu = None;
                if use_krylov && band_fits && krylov_fails_in_row >= KRYLOV_FAILS_BEFORE_FALLBACK {
                    use_krylov = false;
                    ctx.lin.direct_fallbacks += 1;
                }
                continue;
            }
            let safety =
                0.9 * (2 * NEWTON_MAXITER + 1) as f64 / (2 * NEWTON_MAXITER + n_iter) as f64;
            for i in 0..n {
                scale[i] = atol + rtol * ynew[i].abs();
            }
            error_norm = rms(
                (0..n)
                    .filter(|&i| !alg[i])
                    .map(|i| error_const[order] * dcorr[i] / scale[i]),
                n_diff,
            );
            if error_norm > 1.0 {
                ctx.stats.rejected += 1;
                let factor = MIN_FACTOR.max(safety * error_norm.powf(-1.0 / (order as f64 + 1.0)));
                h_abs *= factor;
                change_d(&mut d, order, factor);
                n_equal_steps = 0;
            } else {
                accepted = true;
            }
        }
        ctx.stats.steps += 1;
        if ctx.stats.steps > opts.max_steps {
            return Err(fail(t, format!("more than {} steps", opts.max_steps)));
        }
        n_equal_steps += 1;
        t = t_new;
        std::mem::swap(&mut y, &mut ynew);
        current_jac = false;
        krylov_fails_in_row = 0;
        // Update the differences.
        for i in 0..n {
            d[order + 2][i] = dcorr[i] - d[order + 1][i];
            d[order + 1][i] = dcorr[i];
        }
        for k in (0..=order).rev() {
            let (lo, hi) = d.split_at_mut(k + 1);
            for (a, b) in lo[k].iter_mut().zip(&hi[0]) {
                *a += b;
            }
        }
        // Dense output for the requested times in (t - h, t].
        while next < t_eval.len() && t_eval[next] <= t {
            let te = t_eval[next];
            sol.t.push(te);
            if te == t {
                sol.y.extend_from_slice(&y);
            } else {
                // Interpolate with the current (accepted-step) differences.
                let mut p = 1.0;
                let mut out = d[0].clone();
                for j in 0..order {
                    let shift = t - h_abs * j as f64;
                    let denom = h_abs * (1 + j) as f64;
                    p *= (te - shift) / denom;
                    for (o, dv) in out.iter_mut().zip(&d[j + 1]) {
                        *o += p * dv;
                    }
                }
                sol.y.extend_from_slice(&out);
            }
            next += 1;
        }
        if n_equal_steps < order + 1 {
            continue;
        }
        let scale_n = |v: &[f64]| {
            rms(
                (0..n)
                    .filter(|&i| !alg[i])
                    .map(|i| v[i] / (atol + rtol * y[i].abs())),
                n_diff,
            )
        };
        let error_m_norm = if order > 1 {
            let v: Vec<f64> = d[order]
                .iter()
                .map(|x| error_const[order - 1] * x)
                .collect();
            scale_n(&v)
        } else {
            f64::INFINITY
        };
        let error_p_norm = if order < max_order {
            let v: Vec<f64> = d[order + 2]
                .iter()
                .map(|x| error_const[order + 1] * x)
                .collect();
            scale_n(&v)
        } else {
            f64::INFINITY
        };
        let norms = [error_m_norm, error_norm, error_p_norm];
        let mut best = 0usize;
        let mut best_f = f64::NEG_INFINITY;
        for (k, &e) in norms.iter().enumerate() {
            let fct = if e == 0.0 {
                f64::INFINITY
            } else {
                e.powf(-1.0 / (order + k) as f64)
            };
            if fct > best_f {
                best_f = fct;
                best = k;
            }
        }
        let safety = 0.9 * (2 * NEWTON_MAXITER + 1) as f64 / (2 * NEWTON_MAXITER + n_iter) as f64;
        order = (order as isize + best as isize - 1).clamp(1, max_order as isize) as usize;
        let factor = MAX_FACTOR.min(safety * best_f);
        h_abs *= factor;
        change_d(&mut d, order, factor);
        n_equal_steps = 0;
        lu = None;
    }
    Ok(())
}

fn with_diagonal(a: &CsrMatrix) -> CsrMatrix {
    let n = a.n_rows;
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    for r in 0..n {
        let seg = &a.indices[a.indptr[r]..a.indptr[r + 1]];
        let mut row: Vec<usize> = seg.to_vec();
        if seg.binary_search(&r).is_err() {
            row.push(r);
            row.sort_unstable();
        }
        indices.extend(row);
        indptr.push(indices.len());
    }
    let nnz = indices.len();
    CsrMatrix {
        n_rows: n,
        n_cols: n,
        indptr,
        indices,
        data: vec![1.0; nnz],
    }
}

/// `M - c J` (with an explicit diagonal).
fn iteration_matrix(j: &CsrMatrix, mass: &[f64], c: f64) -> CsrMatrix {
    let n = j.n_rows;
    let mut indptr = vec![0usize];
    let mut indices = Vec::with_capacity(j.indices.len() + n);
    let mut data = Vec::with_capacity(j.indices.len() + n);
    for r in 0..n {
        let mut diag_done = false;
        for k in j.indptr[r]..j.indptr[r + 1] {
            let col = j.indices[k];
            if col > r && !diag_done {
                indices.push(r);
                data.push(mass[r]);
                diag_done = true;
            }
            let mut v = -c * j.data[k];
            if col == r {
                v += mass[r];
                diag_done = true;
            }
            indices.push(col);
            data.push(v);
        }
        if !diag_done {
            indices.push(r);
            data.push(mass[r]);
        }
        indptr.push(indices.len());
    }
    CsrMatrix {
        n_rows: n,
        n_cols: n,
        indptr,
        indices,
        data,
    }
}

/// Newton on the algebraic components (differential ones held fixed) so the
/// constraints hold at `t0`.
fn consistent_init(
    ctx: &mut Ctx<'_>,
    t: f64,
    y: &mut [f64],
    alg: &[bool],
    tol: f64,
) -> Result<(), MolError> {
    let n = y.len();
    let idx: Vec<usize> = (0..n).filter(|&i| alg[i]).collect();
    if idx.is_empty() {
        return Ok(());
    }
    let mut pos = vec![usize::MAX; n];
    for (k, &i) in idx.iter().enumerate() {
        pos[i] = k;
    }
    let mut f = vec![0.0; n];
    for _ in 0..20 {
        ctx.f(t, y, &mut f)?;
        let res = rms(idx.iter().map(|&i| f[i]), idx.len());
        // The algebraic block must be nonsingular (index 1) even when the
        // initial state happens to satisfy the constraints already.
        let jac = ctx.prog.jacobian(t, y)?;
        ctx.stats.jac_evals += 1;
        // The algebraic block G_z.
        let mut indptr = vec![0usize];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for &r in &idx {
            for k in jac.indptr[r]..jac.indptr[r + 1] {
                let c = jac.indices[k];
                if pos[c] != usize::MAX {
                    indices.push(pos[c]);
                    data.push(jac.data[k]);
                }
            }
            indptr.push(indices.len());
        }
        let g = CsrMatrix {
            n_rows: idx.len(),
            n_cols: idx.len(),
            indptr,
            indices,
            data,
        };
        let pat = with_diagonal(&g);
        let perm = rcm_order(&pat);
        let (kl, ku) = BandedLu::bandwidths(&pat, &perm);
        let gz = add_zero_diagonal(&g);
        let lu = BandedLu::factor(&gz, &perm, kl, ku).ok_or_else(|| {
            fail(
                t,
                "the algebraic constraints are singular in the algebraic variables \
                 (differentiation index > 1)",
            )
        })?;
        if res <= tol.max(1e-14) {
            return Ok(());
        }
        let mut b: Vec<f64> = idx.iter().map(|&i| -f[i]).collect();
        lu.solve(&mut b);
        for (k, &i) in idx.iter().enumerate() {
            y[i] += b[k];
        }
    }
    Err(fail(
        t,
        "could not make the initial state consistent with the constraints",
    ))
}

fn add_zero_diagonal(a: &CsrMatrix) -> CsrMatrix {
    iteration_matrix(a, &vec![0.0; a.n_rows], -1.0)
}
