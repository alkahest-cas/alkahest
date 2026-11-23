//! Parameter fitting for FEX candidates.
//!
//! * **Levenberg–Marquardt** on the least-squares residual, using the exact
//!   Jacobian from the fused evaluator — the default, and far more effective
//!   per step than first-order methods on these small (|θ| ≤ a few hundred)
//!   problems.
//! * **Adam** and **BFGS** on the scalar loss and its exact gradient, to
//!   mirror the FEX paper's "20 Adam + 20 L-BFGS" coarse fit and its long
//!   Adam fine-tune, and for losses that are not least squares (Rayleigh
//!   quotient, normalised residual).
//!
//! Every fitter returns the best parameters it saw, never a worse iterate.
//! A candidate that cannot be evaluated anywhere (overflow, zero
//! denominator) comes back with loss `+∞`.

#![allow(clippy::needless_range_loop)]

use super::error::FexError;
use super::eval::{Evaluator, NodeOp};

/// Which optimiser to run.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum FitMethod {
    /// Levenberg–Marquardt with `max_iter` accepted or rejected steps.
    Lm {
        /// Iteration cap.
        max_iter: usize,
    },
    /// Adam with a fixed learning rate.
    Adam {
        /// Steps.
        iters: usize,
        /// Learning rate.
        lr: f64,
    },
    /// Dense BFGS with a backtracking (Armijo) line search.
    Bfgs {
        /// Iteration cap.
        iters: usize,
    },
    /// Adam then BFGS — the FEX paper's coarse scoring fit (20 + 20).
    AdamBfgs {
        /// Adam steps.
        adam: usize,
        /// Adam learning rate.
        lr: f64,
        /// BFGS iterations.
        bfgs: usize,
    },
}

impl Default for FitMethod {
    fn default() -> Self {
        FitMethod::Lm { max_iter: 50 }
    }
}

/// Fitting options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitOptions {
    /// The optimiser.
    pub method: FitMethod,
    /// Stop once the loss is below this.
    pub tol: f64,
    /// Pin the parameters [`super::redundant_params`] marks as redundant to
    /// their neutral values (better conditioning, same function family).
    pub freeze_redundant: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            method: FitMethod::default(),
            tol: 1e-30,
            freeze_redundant: true,
        }
    }
}

/// The outcome of a fit.
#[derive(Clone, Debug, PartialEq)]
pub struct FitResult {
    /// Best parameters found.
    pub theta: Vec<f64>,
    /// Their loss (`+∞` when the candidate cannot be evaluated).
    pub loss: f64,
    /// Optimiser iterations taken.
    pub iterations: usize,
    /// Loss (or residual) evaluations.
    pub evaluations: usize,
}

impl FitResult {
    /// The FEX reward `1 / (1 + √loss)`.
    pub fn score(&self) -> f64 {
        if self.loss.is_finite() {
            1.0 / (1.0 + self.loss.max(0.0).sqrt())
        } else {
            0.0
        }
    }
}

/// Fit one candidate starting from `theta0`.
pub fn fit(
    ev: &Evaluator,
    e: &[usize],
    theta0: &[f64],
    opts: &FitOptions,
) -> Result<FitResult, FexError> {
    let ops = ev.resolve(e)?;
    ev.template().check_params(theta0)?;
    if matches!(opts.method, FitMethod::Lm { .. })
        && ev.problem().loss() != super::problem::LossKind::LeastSquares
    {
        return Err(FexError::LossKind {
            detail: "Levenberg-Marquardt needs a least-squares problem".into(),
        });
    }
    let mask = freeze_mask(ev, e, opts)?;
    Ok(fit_ops(ev, &ops, theta0, opts, mask.as_deref()))
}

/// Pinned values per parameter (`None` = free), when freezing is on.
fn freeze_mask(
    ev: &Evaluator,
    e: &[usize],
    opts: &FitOptions,
) -> Result<Option<Vec<Option<f64>>>, FexError> {
    if !opts.freeze_redundant {
        return Ok(None);
    }
    let m = super::canon::redundant_params(ev.template(), e)?;
    Ok(m.iter().any(Option::is_some).then_some(m))
}

/// Fit a batch of candidates (`es` is `B × n_slots`, `theta0` is
/// `B × n_params`), parallel over candidates.
pub fn fit_batch(
    ev: &Evaluator,
    es: &[usize],
    theta0: &[f64],
    opts: &FitOptions,
) -> Result<Vec<FitResult>, FexError> {
    let ns = ev.template().n_slots();
    let np = ev.template().n_params();
    if ns == 0 || es.len() % ns != 0 {
        return Err(FexError::Opcode {
            detail: format!("opcode batch length {} is not a multiple of {ns}", es.len()),
        });
    }
    let b = es.len() / ns;
    if theta0.len() != b * np {
        return Err(FexError::Params {
            expected: b * np,
            got: theta0.len(),
        });
    }
    if matches!(opts.method, FitMethod::Lm { .. })
        && ev.problem().loss() != super::problem::LossKind::LeastSquares
    {
        return Err(FexError::LossKind {
            detail: "Levenberg-Marquardt needs a least-squares problem".into(),
        });
    }
    let ops: Vec<Vec<NodeOp>> = (0..b)
        .map(|c| ev.resolve(&es[c * ns..(c + 1) * ns]))
        .collect::<Result<_, _>>()?;
    let masks: Vec<Option<Vec<Option<f64>>>> = (0..b)
        .map(|c| freeze_mask(ev, &es[c * ns..(c + 1) * ns], opts))
        .collect::<Result<_, _>>()?;
    let run = |c: usize| {
        fit_ops(
            ev,
            &ops[c],
            &theta0[c * np..(c + 1) * np],
            opts,
            masks[c].as_deref(),
        )
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        Ok((0..b).into_par_iter().map(run).collect())
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok((0..b).map(run).collect())
    }
}

pub(crate) fn fit_ops(
    ev: &Evaluator,
    ops: &[NodeOp],
    theta0: &[f64],
    opts: &FitOptions,
    pin: Option<&[Option<f64>]>,
) -> FitResult {
    let mut start = theta0.to_vec();
    let free: Option<Vec<bool>> = pin.map(|p| {
        for (t, v) in start.iter_mut().zip(p) {
            if let Some(v) = v {
                *t = *v;
            }
        }
        p.iter().map(Option::is_none).collect()
    });
    let free = free.as_deref();
    let theta0 = &start[..];
    match opts.method {
        FitMethod::Lm { max_iter } => lm(ev, ops, theta0, max_iter, opts.tol, free),
        FitMethod::Adam { iters, lr } => adam(ev, ops, theta0, iters, lr, opts.tol, free),
        FitMethod::Bfgs { iters } => bfgs(ev, ops, theta0, iters, opts.tol, free),
        FitMethod::AdamBfgs { adam: a, lr, bfgs: b } => {
            let r1 = adam(ev, ops, theta0, a, lr, opts.tol, free);
            let start = if r1.loss.is_finite() {
                r1.theta.clone()
            } else {
                theta0.to_vec()
            };
            let mut r2 = bfgs(ev, ops, &start, b, opts.tol, free);
            r2.iterations += r1.iterations;
            r2.evaluations += r1.evaluations;
            if r1.loss < r2.loss {
                FitResult {
                    theta: r1.theta,
                    loss: r1.loss,
                    iterations: r2.iterations,
                    evaluations: r2.evaluations,
                }
            } else {
                r2
            }
        }
    }
}

/// Solve `A x = b` for symmetric positive (semi)definite `A` (`n × n`,
/// row-major) by Cholesky; `None` if not positive definite.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

/// `JᵀJ` (upper triangle mirrored) and `Jᵀr`.
fn normal_equations(j: &[f64], r: &[f64], np: usize) -> (Vec<f64>, Vec<f64>) {
    let m = r.len();
    let work = |rows: std::ops::Range<usize>| {
        let mut a = vec![0.0; np * np];
        let mut g = vec![0.0; np];
        for row in rows {
            let jr = &j[row * np..(row + 1) * np];
            let rv = r[row];
            for p in 0..np {
                let jp = jr[p];
                if jp == 0.0 {
                    continue;
                }
                g[p] += jp * rv;
                let ap = &mut a[p * np..(p + 1) * np];
                for q in p..np {
                    ap[q] += jp * jr[q];
                }
            }
        }
        (a, g)
    };
    const ROWS: usize = 512;
    let chunks: Vec<std::ops::Range<usize>> = (0..m)
        .step_by(ROWS)
        .map(|r0| r0..(r0 + ROWS).min(m))
        .collect();
    #[cfg(feature = "parallel")]
    let parts: Vec<(Vec<f64>, Vec<f64>)> = {
        use rayon::prelude::*;
        chunks.into_par_iter().map(work).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let parts: Vec<(Vec<f64>, Vec<f64>)> = chunks.into_iter().map(work).collect();
    let mut a = vec![0.0; np * np];
    let mut g = vec![0.0; np];
    for (pa, pg) in parts {
        for (x, y) in a.iter_mut().zip(&pa) {
            *x += y;
        }
        for (x, y) in g.iter_mut().zip(&pg) {
            *x += y;
        }
    }
    for p in 0..np {
        for q in 0..p {
            a[p * np + q] = a[q * np + p];
        }
    }
    (a, g)
}

/// Zero the entries of `g` (or the columns of a row-major `rows × n`
/// matrix) belonging to pinned parameters.
fn mask_cols(v: &mut [f64], n: usize, free: Option<&[bool]>) {
    if let Some(f) = free {
        for row in v.chunks_mut(n) {
            for (x, &ok) in row.iter_mut().zip(f) {
                if !ok {
                    *x = 0.0;
                }
            }
        }
    }
}

fn lm(
    ev: &Evaluator,
    ops: &[NodeOp],
    theta0: &[f64],
    max_iter: usize,
    tol: f64,
    free: Option<&[bool]>,
) -> FitResult {
    let np = theta0.len();
    let m = ev.n_residuals();
    let mut theta = theta0.to_vec();
    let mut r = vec![0.0; m];
    let mut jac = vec![0.0; m * np];
    let mut evals = 1;
    let ok = ev_rows(ev, ops, &theta, &mut r, Some(&mut jac));
    if !ok {
        return FitResult {
            theta,
            loss: f64::INFINITY,
            iterations: 0,
            evaluations: evals,
        };
    }
    mask_cols(&mut jac, np, free);
    let mut loss: f64 = r.iter().map(|v| v * v).sum();
    let mut mu = 1e-3;
    let mut iters = 0;
    let mut r_try = vec![0.0; m];
    let (mut a, mut g) = normal_equations(&jac, &r, np);
    while iters < max_iter && loss > tol {
        iters += 1;
        // (A + μ·diag(A) + ε I) δ = −g
        let maxdiag = (0..np).map(|p| a[p * np + p]).fold(0.0f64, f64::max);
        let mut damped = a.clone();
        for p in 0..np {
            damped[p * np + p] += mu * a[p * np + p] + 1e-12 * maxdiag.max(1e-300);
        }
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let Some(delta) = cholesky_solve(&damped, &neg_g, np) else {
            mu *= 10.0;
            if mu > 1e16 {
                break;
            }
            continue;
        };
        let trial: Vec<f64> = theta.iter().zip(&delta).map(|(t, d)| t + d).collect();
        evals += 1;
        let ok = ev_rows(ev, ops, &trial, &mut r_try, None);
        let new_loss: f64 = if ok {
            r_try.iter().map(|v| v * v).sum()
        } else {
            f64::INFINITY
        };
        if new_loss < loss {
            let rel = (loss - new_loss) / loss.max(1e-300);
            theta = trial;
            loss = new_loss;
            mu = (mu / 3.0).max(1e-15);
            evals += 1;
            if !ev_rows(ev, ops, &theta, &mut r, Some(&mut jac)) {
                // Cannot happen for a point whose residuals were finite
                // unless the Jacobian alone overflows; stop at this iterate.
                break;
            }
            mask_cols(&mut jac, np, free);
            (a, g) = normal_equations(&jac, &r, np);
            if rel < 1e-15 {
                break;
            }
        } else {
            mu *= 4.0;
            if mu > 1e16 {
                break;
            }
        }
    }
    FitResult {
        theta,
        loss,
        iterations: iters,
        evaluations: evals,
    }
}

fn ev_rows(ev: &Evaluator, ops: &[NodeOp], th: &[f64], r: &mut [f64], j: Option<&mut [f64]>) -> bool {
    ev.cand_rows(ops, th, r, j)
}

fn loss_grad(ev: &Evaluator, ops: &[NodeOp], th: &[f64], g: &mut [f64], free: Option<&[bool]>) -> f64 {
    let l = ev.loss_impl(ops, th, Some(g));
    let n = g.len();
    mask_cols(g, n, free);
    l
}

fn adam(
    ev: &Evaluator,
    ops: &[NodeOp],
    theta0: &[f64],
    iters: usize,
    lr: f64,
    tol: f64,
    free: Option<&[bool]>,
) -> FitResult {
    let np = theta0.len();
    let (b1, b2, eps) = (0.9, 0.999, 1e-8);
    let mut th = theta0.to_vec();
    let mut mom = vec![0.0; np];
    let mut vel = vec![0.0; np];
    let mut g = vec![0.0; np];
    let mut best = (f64::INFINITY, th.clone());
    let mut it = 0;
    for t in 1..=iters {
        it = t;
        let l = loss_grad(ev, ops, &th, &mut g, free);
        if l < best.0 {
            best = (l, th.clone());
        }
        if !l.is_finite() || l <= tol {
            break;
        }
        let c1 = 1.0 - f64::powi(b1, t as i32);
        let c2 = 1.0 - f64::powi(b2, t as i32);
        for k in 0..np {
            mom[k] = b1 * mom[k] + (1.0 - b1) * g[k];
            vel[k] = b2 * vel[k] + (1.0 - b2) * g[k] * g[k];
            th[k] -= lr * (mom[k] / c1) / ((vel[k] / c2).sqrt() + eps);
        }
    }
    let l = ev.loss_impl(ops, &th, None);
    if l < best.0 {
        best = (l, th);
    }
    FitResult {
        theta: best.1,
        loss: best.0,
        iterations: it,
        evaluations: it + 1,
    }
}

fn bfgs(
    ev: &Evaluator,
    ops: &[NodeOp],
    theta0: &[f64],
    iters: usize,
    tol: f64,
    free: Option<&[bool]>,
) -> FitResult {
    let n = theta0.len();
    let mut x = theta0.to_vec();
    let mut g = vec![0.0; n];
    let mut f = loss_grad(ev, ops, &x, &mut g, free);
    let mut evals = 1;
    if !f.is_finite() {
        return FitResult {
            theta: x,
            loss: f64::INFINITY,
            iterations: 0,
            evaluations: evals,
        };
    }
    // Inverse Hessian approximation, started as a scaled identity.
    let mut h = vec![0.0; n * n];
    let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let h0 = if gn > 0.0 { (1.0 / gn).min(1.0) } else { 1.0 };
    for i in 0..n {
        h[i * n + i] = h0;
    }
    let mut it = 0;
    let mut gnew = vec![0.0; n];
    while it < iters && f > tol {
        it += 1;
        let mut p = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s -= h[i * n + j] * g[j];
            }
            p[i] = s;
        }
        let mut slope: f64 = p.iter().zip(&g).map(|(a, b)| a * b).sum();
        if slope >= 0.0 {
            // Not a descent direction: reset to steepest descent.
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] = if i == j { h0 } else { 0.0 };
                }
                p[i] = -h0 * g[i];
            }
            slope = p.iter().zip(&g).map(|(a, b)| a * b).sum();
            if slope >= 0.0 {
                break;
            }
        }
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let xt: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + step * b).collect();
            let ft = loss_grad(ev, ops, &xt, &mut gnew, free);
            evals += 1;
            if ft.is_finite() && ft <= f + 1e-4 * step * slope {
                accepted = Some((xt, ft));
                break;
            }
            step *= 0.5;
        }
        let Some((xt, ft)) = accepted else { break };
        let s: Vec<f64> = xt.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = gnew.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        let rel = (f - ft) / f.abs().max(1e-300);
        x = xt;
        f = ft;
        g.copy_from_slice(&gnew);
        if sy > 1e-300 {
            // H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
            let rho = 1.0 / sy;
            let mut hy = vec![0.0; n];
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += h[i * n + j] * y[j];
                }
                hy[i] = acc;
            }
            let yhy: f64 = y.iter().zip(&hy).map(|(a, b)| a * b).sum();
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] += -rho * (hy[i] * s[j] + s[i] * hy[j])
                        + (rho * rho * yhy + rho) * s[i] * s[j];
                }
            }
        }
        if rel < 1e-16 {
            break;
        }
    }
    FitResult {
        theta: x,
        loss: f,
        iterations: it,
        evaluations: evals,
    }
}
