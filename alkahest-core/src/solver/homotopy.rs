//! Numerical algebraic geometry — total-degree and polyhedral homotopy continuation (V2-14, V2-17).
//!
//! **Total-degree (Bézout):** tracks `(1−t)·γ·G(Z) + t·F̂(Z) = 0` from `t=0→1` in
//! projective space — each equation homogenised with `Z₀`, on a random affine chart
//! `a·Z = 1` — with decoupled start `G_i(Z) = Z_i^{d_i} − Z₀^{d_i}`.  A path to a
//! solution at infinity ends at an ordinary point with `Z₀ = 0` instead of diverging.
//! The Bézout count ∏ d_i is tight only for generic dense systems.  Path jumps are
//! detected (two paths at one regular root, or a non-real root without its conjugate)
//! and the run is repeated with a fresh γ; see [`solve_numerical_complex`].
//!
//! **Polyhedral (BKK):** for 2-variable systems where the mixed volume is below the
//! Bézout bound (e.g. Katsura family), [`polyhedral`] is meant to supply binomial start
//! systems and exact start points, tracked by `H = (1−t)·G_cell(z) + t·F(z)` with the
//! same Euler-Newton predictor-corrector.  Its mixed-cell enumeration is currently
//! broken (see the TODO in [`polyhedral`]) and yields no start points, so
//! [`solve_numerical`] checks the supplied path count against the mixed volume and
//! falls back to the Bézout start whenever it comes up short.
//!
//! Endpoints are Newton-polished in ℝⁿ and checked with a conservative Smale
//! heuristic plus `ArbBall` enclosures.

#![allow(clippy::needless_range_loop)]

use crate::ball::ArbBall;
use crate::eval::ComplexF64;
use crate::kernel::{ExprId, ExprPool};
use crate::poly::groebner::GbPoly;
use crate::solver::{expr_to_gbpoly, polyhedral, SolverError};
use rug::Rational;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub(crate) struct C64 {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

impl C64 {
    const ZERO: C64 = C64 { re: 0.0, im: 0.0 };

    fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }

    fn from_f64(re: f64) -> Self {
        C64 { re, im: 0.0 }
    }

    fn norm2(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn add(a: C64, b: C64) -> C64 {
        C64 {
            re: a.re + b.re,
            im: a.im + b.im,
        }
    }

    fn sub(a: C64, b: C64) -> C64 {
        C64 {
            re: a.re - b.re,
            im: a.im - b.im,
        }
    }

    fn mul(a: C64, b: C64) -> C64 {
        C64 {
            re: a.re * b.re - a.im * b.im,
            im: a.re * b.im + a.im * b.re,
        }
    }

    fn scale(s: f64, a: C64) -> C64 {
        C64 {
            re: s * a.re,
            im: s * a.im,
        }
    }

    fn neg(a: C64) -> C64 {
        C64 {
            re: -a.re,
            im: -a.im,
        }
    }

    fn div(a: C64, b: C64) -> Option<C64> {
        let d = b.re * b.re + b.im * b.im;
        if d < 1e-30 {
            return None;
        }
        Some(C64 {
            re: (a.re * b.re + a.im * b.im) / d,
            im: (a.im * b.re - a.re * b.im) / d,
        })
    }

    fn pow_int(base: C64, exp: u32) -> C64 {
        if exp == 0 {
            return C64::new(1.0, 0.0);
        }
        let mut e = exp;
        let mut acc = C64::new(1.0, 0.0);
        let mut cur = base;
        while e > 0 {
            if e & 1 == 1 {
                acc = C64::mul(acc, cur);
            }
            cur = C64::mul(cur, cur);
            e >>= 1;
        }
        acc
    }
}

/// Controls for [`solve_numerical`].
#[derive(Debug, Clone)]
pub struct HomotopyOpts {
    pub max_tracker_steps: usize,
    pub dt_initial: f64,
    pub dt_min: f64,
    pub homotopy_tol: f64,
    pub newton_tol: f64,
    pub newton_cap: usize,
    pub dedup_tol: f64,
    pub gamma_angle_seed: Option<u64>,
    pub certify_prec_bits: u32,
    /// Abort if Bézout path budget (= ∏ total degrees) exceeds this cap.
    pub max_bezout_paths: usize,
}

impl Default for HomotopyOpts {
    fn default() -> Self {
        Self {
            max_tracker_steps: 50_000,
            dt_initial: 0.02,
            dt_min: 1e-8,
            homotopy_tol: 1e-10,
            newton_tol: 1e-12,
            newton_cap: 48,
            dedup_tol: 1e-5,
            gamma_angle_seed: Some(31415926535897),
            certify_prec_bits: 128,
            max_bezout_paths: 20_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CertifiedPoint {
    pub coordinates: Vec<f64>,
    pub max_residual_f64: f64,
    pub smale_alpha: Option<f64>,
    pub smale_certified: bool,
    pub enclosure: Vec<ArbBall>,
}

#[derive(Debug, Clone)]
pub enum HomotopyError {
    Algebraic(SolverError),
    BezoutTooLarge(usize),
    SingularJacobian,
    TrackerFailed(&'static str),
}

impl std::fmt::Display for HomotopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HomotopyError::Algebraic(e) => write!(f, "{e}"),
            HomotopyError::BezoutTooLarge(n) => {
                write!(
                    f,
                    "Bézout path budget {n} exceeds HomotopyOpts::max_bezout_paths — \
                     use mixed-volume starts for large sparse systems",
                )
            }
            HomotopyError::SingularJacobian => write!(f, "singular Jacobian"),
            HomotopyError::TrackerFailed(s) => write!(f, "path tracker failed: {s}"),
        }
    }
}

impl std::error::Error for HomotopyError {}

impl crate::errors::AlkahestError for HomotopyError {
    fn code(&self) -> &'static str {
        match self {
            HomotopyError::Algebraic(inner) => inner.code(),
            HomotopyError::BezoutTooLarge(_) => "E-HOMOTOPY-002",
            HomotopyError::SingularJacobian => "E-HOMOTOPY-003",
            HomotopyError::TrackerFailed(_) => "E-HOMOTOPY-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            HomotopyError::Algebraic(inner) => inner.remediation(),
            HomotopyError::BezoutTooLarge(_) => {
                Some("raise HomotopyOpts::max_bezout_paths or switch to polyhedral continuation")
            }
            HomotopyError::SingularJacobian => {
                Some("try HomotopyOpts::gamma_angle_seed or rescale equations")
            }
            HomotopyError::TrackerFailed(_) => {
                Some("adjust dt_initial, relax tolerances, or increase max_tracker_steps")
            }
        }
    }
}

fn rat_to_f64(r: &Rational) -> f64 {
    r.numer().to_f64() / r.denom().to_f64()
}

fn total_degree(p: &GbPoly) -> u32 {
    // `expr_to_gbpoly` bounds every total degree by u32::MAX, so this cannot
    // wrap (it used to: x^(2^31)·y^(2^31) had degree 0).
    p.terms
        .keys()
        .map(|e| crate::poly::exponent::total_degree_or_panic(e))
        .max()
        .unwrap_or(0)
}

fn gbpoly_eval_c(p: &GbPoly, z: &[C64]) -> C64 {
    let mut acc = C64::ZERO;
    for (exp, coeff) in &p.terms {
        let c = rat_to_f64(coeff);
        let mut mono = C64::new(c, 0.0);
        for (i, &e) in exp.iter().enumerate() {
            if e != 0 {
                mono = C64::mul(mono, C64::pow_int(z[i], e));
            }
        }
        acc = C64::add(acc, mono);
    }
    acc
}

fn gbpoly_derive_var(p: &GbPoly, var: usize) -> GbPoly {
    let nv = p.n_vars;
    let mut out = GbPoly::zero(nv);
    for (exp, coeff) in &p.terms {
        let e = exp.get(var).copied().unwrap_or(0);
        if e == 0 {
            continue;
        }
        let mut new_exp = exp.clone();
        new_exp[var] = e - 1;
        let scale = coeff * Rational::from(e);
        out = out.add(&GbPoly::monomial(new_exp, scale));
    }
    out
}

fn jacobian_c(sys: &[GbPoly], z: &[C64]) -> Vec<Vec<C64>> {
    let n = sys.len();
    let mut j = vec![vec![C64::ZERO; n]; n];
    for i in 0..n {
        for k in 0..n {
            let di = gbpoly_derive_var(&sys[i], k);
            j[i][k] = gbpoly_eval_c(&di, z);
        }
    }
    j
}

fn hessian_ij_c(poly: &GbPoly, row_var: usize, col_var: usize, z: &[C64]) -> C64 {
    let d_row = gbpoly_derive_var(poly, row_var);
    gbpoly_eval_c(&gbpoly_derive_var(&d_row, col_var), z)
}

fn start_system_roots(degrees: &[u32]) -> Vec<Vec<C64>> {
    let mut curves: Vec<Vec<C64>> = Vec::with_capacity(degrees.len());
    for &d in degrees {
        assert!(d > 0);
        let mut roots = Vec::with_capacity(d as usize);
        for k in 0..d {
            let ang = PI * (2.0 * (k as f64) / (d as f64));
            roots.push(C64::new(ang.cos(), ang.sin()));
        }
        curves.push(roots);
    }
    let mut out = curves[0]
        .iter()
        .cloned()
        .map(|c| vec![c])
        .collect::<Vec<_>>();
    for tier in curves.iter().skip(1) {
        let mut next = Vec::with_capacity(out.len() * tier.len());
        for prefix in &out {
            for r in tier {
                let mut v = prefix.clone();
                v.push(*r);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

fn complex_linsolve(mut a: Vec<Vec<C64>>, mut b: Vec<C64>) -> Option<Vec<C64>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = None;
        let mut best = -1.0_f64;
        for row in col..n {
            let nm = C64::norm2(a[row][col]);
            if nm > best {
                best = nm;
                piv = Some(row);
            }
        }
        let prow = piv?;
        if best < 1e-18 {
            return None;
        }
        if prow != col {
            a.swap(prow, col);
            b.swap(prow, col);
        }
        let div = a[col][col];
        for j in col..n {
            a[col][j] = C64::div(a[col][j], div)?;
        }
        b[col] = C64::div(b[col], div)?;
        for row in (0..n).filter(|&r| r != col) {
            let fac = a[row][col];
            if fac.re.abs() < 1e-30 && fac.im.abs() < 1e-30 {
                continue;
            }
            for j in col..n {
                a[row][j] = C64::sub(a[row][j], C64::mul(fac, a[col][j]));
            }
            b[row] = C64::sub(b[row], C64::mul(fac, b[col]));
        }
    }
    Some(b)
}

fn fv_linf(vals: &[C64]) -> f64 {
    vals.iter().map(|v| v.norm2()).fold(0.0_f64, f64::max)
}

fn jacobian_real(sys: &[GbPoly], x: &[f64]) -> Vec<Vec<f64>> {
    let z: Vec<C64> = x.iter().map(|&r| C64::from_f64(r)).collect();
    let jc = jacobian_c(sys, &z);
    let n = x.len();
    let mut jr = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            jr[i][j] = jc[i][j].re;
        }
    }
    jr
}

fn fv_real(sys: &[GbPoly], x: &[f64]) -> Vec<f64> {
    let z: Vec<C64> = x.iter().map(|&r| C64::from_f64(r)).collect();
    (0..sys.len())
        .map(|i| gbpoly_eval_c(&sys[i], &z).re)
        .collect()
}

fn real_gaussian_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in i + 1..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if best < 1e-18 {
            return None;
        }
        if piv != i {
            a.swap(piv, i);
            b.swap(piv, i);
        }
        let div = a[i][i];
        for j in i..n {
            a[i][j] /= div;
        }
        b[i] /= div;
        for r in 0..n {
            if r == i {
                continue;
            }
            let fac = a[r][i];
            if fac.abs() < 1e-28 {
                continue;
            }
            for j in i..n {
                a[r][j] -= fac * a[i][j];
            }
            b[r] -= fac * b[i];
        }
    }
    Some(b)
}

// ---------------------------------------------------------------------------
// Polyhedral homotopy helpers (V2-17)
//
// These track from a GbPoly start system G (mixed-cell starts) instead of
// the degree-based Bézout start.
// The homotopy is H(z,t) = (1−t)·G(z) + t·F(z).
// ---------------------------------------------------------------------------

fn hv_sys(target: &[GbPoly], start: &[GbPoly], z: &[C64], t: f64) -> Vec<C64> {
    let mt = C64::new(1.0 - t, 0.0);
    let tt = C64::new(t, 0.0);
    (0..z.len())
        .map(|i| {
            let f = gbpoly_eval_c(&target[i], z);
            let g = gbpoly_eval_c(&start[i], z);
            C64::add(C64::mul(mt, g), C64::mul(tt, f))
        })
        .collect()
}

fn dh_dt_sys(target: &[GbPoly], start: &[GbPoly], z: &[C64]) -> Vec<C64> {
    (0..z.len())
        .map(|i| {
            let f = gbpoly_eval_c(&target[i], z);
            let g = gbpoly_eval_c(&start[i], z);
            C64::sub(f, g)
        })
        .collect()
}

fn jh_sys(target: &[GbPoly], start: &[GbPoly], z: &[C64], t: f64) -> Vec<Vec<C64>> {
    let n = z.len();
    let j_f = jacobian_c(target, z);
    let j_g = jacobian_c(start, z);
    let mt = C64::new(1.0 - t, 0.0);
    let tt = C64::new(t, 0.0);
    let mut jac = vec![vec![C64::ZERO; n]; n];
    for i in 0..n {
        for k in 0..n {
            jac[i][k] = C64::add(C64::mul(tt, j_f[i][k]), C64::mul(mt, j_g[i][k]));
        }
    }
    jac
}

fn damped_correct_sys(
    target: &[GbPoly],
    start: &[GbPoly],
    z0: &[C64],
    t_tgt: f64,
    opts: &HomotopyOpts,
) -> Option<Vec<C64>> {
    let mut z = z0.to_vec();
    for _ in 0..opts.newton_cap {
        let fv = hv_sys(target, start, &z, t_tgt);
        let res = fv_linf(&fv);
        if res < opts.homotopy_tol {
            return Some(z);
        }
        let jac = jh_sys(target, start, &z, t_tgt);
        let neg_f: Vec<C64> = fv.iter().map(|c| C64::neg(*c)).collect();
        let step = complex_linsolve(jac, neg_f)?;
        let mut lm = 1.0_f64;
        loop {
            let trial: Vec<C64> = z
                .iter()
                .zip(step.iter())
                .map(|(zi, s)| C64::add(*zi, C64::scale(lm, *s)))
                .collect();
            let new_res = fv_linf(&hv_sys(target, start, &trial, t_tgt));
            if new_res < res || new_res < opts.homotopy_tol {
                z = trial;
                break;
            }
            lm *= 0.5;
            if lm < 1e-12 {
                return None;
            }
        }
    }
    fv_linf(&hv_sys(target, start, &z, t_tgt))
        .le(&(opts.homotopy_tol * 8.0))
        .then_some(z)
}

fn track_path_sys(
    target: &[GbPoly],
    start: &[GbPoly],
    z_start: Vec<C64>,
    opts: &HomotopyOpts,
) -> Result<Vec<C64>, HomotopyError> {
    let mut z = z_start;
    let mut t = 0.0_f64;
    let mut dt = opts.dt_initial;
    let mut steps_total = 0usize;
    while t < 1.0 - 1e-15 {
        if steps_total > opts.max_tracker_steps {
            return Err(HomotopyError::TrackerFailed("max_tracker_steps"));
        }
        let t_next = (t + dt).min(1.0);
        let jac = jh_sys(target, start, &z, t);
        let htd = dh_dt_sys(target, start, &z);
        let dt_c = C64::new(t_next - t, 0.0);
        let rhs: Vec<C64> = htd
            .into_iter()
            .map(|h| C64::neg(C64::mul(dt_c, h)))
            .collect();
        let step = match complex_linsolve(jac, rhs) {
            Some(s) => s,
            None => {
                dt *= 0.5;
                if dt < opts.dt_min {
                    return Err(HomotopyError::SingularJacobian);
                }
                continue;
            }
        };
        steps_total += 1;
        let zp: Vec<C64> = z
            .iter()
            .zip(step.iter())
            .map(|(zi, dsi)| C64::add(*zi, *dsi))
            .collect();
        match damped_correct_sys(target, start, &zp, t_next, opts) {
            Some(zn) => {
                z = zn;
                t = t_next;
                dt = (dt * 1.15_f64).min(opts.dt_initial);
            }
            None => {
                dt *= 0.5_f64;
                if dt < opts.dt_min {
                    return Err(HomotopyError::TrackerFailed("corrector"));
                }
            }
        }
    }
    Ok(z)
}

fn newton_terminal(target: &[GbPoly], mut x: Vec<f64>, opts: &HomotopyOpts) -> Option<Vec<f64>> {
    for _ in 0..opts.newton_cap {
        let f = fv_real(target, &x);
        let res = f.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if res < opts.newton_tol {
            return Some(x);
        }
        let j = jacobian_real(target, &x);
        let neg_f = f.iter().map(|v| -*v).collect();
        let step = real_gaussian_solve(j.clone(), neg_f)?;
        let mut lm = 1.0_f64;
        loop {
            let trial: Vec<f64> = x
                .iter()
                .zip(step.iter())
                .map(|(&xi, &s)| xi + lm * s)
                .collect();
            let tres = fv_real(target, &trial)
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            if tres < res {
                x = trial;
                break;
            }
            lm *= 0.5;
            if lm < 1e-14 {
                return None;
            }
        }
    }
    Some(x)
}

fn smale_estimate(target: &[GbPoly], x: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    let f = fv_real(target, x);
    let jac = jacobian_real(target, x);
    let step = real_gaussian_solve(jac.clone(), f.iter().map(|v| -*v).collect())?;
    let beta = step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let mut j_inv_inf = 0.0_f64;
    for i in 0..n {
        let mut ej = vec![0.0_f64; n];
        ej[i] = 1.0;
        let col = real_gaussian_solve(jac.clone(), ej)?;
        let s = col.iter().map(|v| v.abs()).sum::<f64>();
        j_inv_inf = j_inv_inf.max(s);
    }
    let z: Vec<C64> = x.iter().map(|&r| C64::from_f64(r)).collect();
    let mut hmax = 0.0_f64;
    for poly in target {
        for j in 0..n {
            for k in 0..n {
                let h = hessian_ij_c(poly, j, k, &z);
                hmax = hmax.max(h.re.abs().max(h.im.abs()));
            }
        }
    }
    let gamma_tilde = j_inv_inf * hmax * (n as f64).sqrt().max(1.0);
    Some((beta, beta * gamma_tilde))
}

fn random_gamma(seed: Option<u64>) -> C64 {
    let mut x = seed
        .unwrap_or(31415926535897_u64)
        .wrapping_add(1469580727_u64);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let frac = ((x >> 11) & ((1_u64 << 53) - 1)) as f64 / ((1_u64 << 53) as f64);
    let ang = 2.0 * PI * frac;
    C64::new(ang.cos(), ang.sin())
}

fn dedup(points: &[Vec<f64>], tol: f64) -> Vec<Vec<f64>> {
    let mut uniq: Vec<Vec<f64>> = Vec::new();
    'outer: for p in points {
        for u in &uniq {
            let d2: f64 = u
                .iter()
                .zip(p.iter())
                .map(|(&a, &b)| {
                    let s = (a - b).abs();
                    s * s
                })
                .sum();
            if d2.sqrt() < tol {
                continue 'outer;
            }
        }
        uniq.push(p.clone());
    }
    uniq
}

/// Total-degree or polyhedral-BKK continuation + polishing + Smale / ArbBall packaging.
///
/// For 2-variable systems where the BKK mixed volume is strictly below the Bézout bound
/// (e.g. Katsura family), polyhedral homotopy is attempted first; if the mixed-cell
/// decomposition supplies fewer start points than the mixed volume — as the
/// [`polyhedral`] module's cell enumeration does today, where it supplies none — the
/// run falls back to the total-degree (Bézout) start rather than reporting the paths
/// it could not track as an absence of solutions.  The path budget is checked against
/// whichever count is used.
///
/// Returns the **real** solutions only; non-real roots are discarded (use
/// [`solve_numerical_complex`] for all of them).  The real points are taken from the
/// complete complex solution set, so a lost or jumped continuation path raises
/// `E-HOMOTOPY-004` instead of silently dropping a real root, and an empty result
/// means the system has no real solution.
pub fn solve_numerical(
    equations: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
    opts: &HomotopyOpts,
) -> Result<Vec<CertifiedPoint>, HomotopyError> {
    if equations.len() != vars.len() {
        return Err(HomotopyError::Algebraic(SolverError::ShapeMismatch));
    }
    let mut sys: Vec<GbPoly> = Vec::with_capacity(vars.len());
    for &eq in equations {
        sys.push(expr_to_gbpoly(eq, vars, pool).map_err(HomotopyError::Algebraic)?);
    }
    let mut degs: Vec<u32> = sys.iter().map(total_degree).collect();
    for d in &mut degs {
        if *d == 0 {
            *d = 1;
        }
    }
    let mut bez = 1usize;
    for &d in &degs {
        bez = bez
            .checked_mul(d as usize)
            .ok_or(HomotopyError::BezoutTooLarge(usize::MAX))?;
    }

    let prec = opts.certify_prec_bits;
    const SMALE_THRESH: f64 = 0.125;
    let mut raw: Vec<Vec<f64>> = Vec::new();
    // A path that never reached `t = 1` yields nothing, and "nothing" is
    // indistinguishable from "no solution on this path".  Counting the paths
    // that did arrive is what lets an empty result be told apart from a
    // tracker that failed everywhere.
    let mut paths_completed = 0usize;
    let mut paths_started = 0usize;
    let mut used_polyhedral = false;

    if polyhedral::should_use_polyhedral(&sys) {
        // BKK bound is strictly below Bézout — try polyhedral mixed-cell starts.
        let mv = polyhedral::mixed_volume(&sys).unwrap_or(bez);
        if mv > opts.max_bezout_paths {
            return Err(HomotopyError::BezoutTooLarge(mv));
        }
        let cells = polyhedral::polyhedral_cell_iter(&sys[0], &sys[1]);
        let n_starts: usize = cells.iter().map(|(_, s)| s.len()).sum();
        // A polyhedral run is only a valid substitute for the Bézout run when
        // it supplies at least `mv` paths; fewer start points cannot reach
        // every isolated root, and the missing ones would be reported as an
        // empty solution set — a mathematical claim, not a diagnostic.
        // `polyhedral_cell_iter` currently supplies *none* (its mixed-cell
        // criterion selects exactly the edge pairs its binomial solver
        // rejects; see the module TODO), so this fallback fires every time.
        if n_starts >= mv && mv > 0 {
            used_polyhedral = true;
            for (start_sys, cell_starts) in cells {
                for z0 in cell_starts {
                    paths_started += 1;
                    let z_end = match track_path_sys(&sys, &start_sys, z0, opts) {
                        Ok(z) => z,
                        Err(_) => continue,
                    };
                    paths_completed += 1;
                    if z_end.iter().all(|c| c.im.abs() < 1e-6) {
                        let xr: Vec<f64> = z_end.iter().map(|c| c.re).collect();
                        if let Some(xp) = newton_terminal(&sys, xr, opts) {
                            raw.push(xp);
                        }
                    }
                }
            }
        }
    }
    if !used_polyhedral {
        // The real points of the complete complex solution set.  Tracking the
        // Bézout paths here directly, as this used to, skipped every path that
        // failed and kept whatever a jumped path landed on — so a real root
        // could go missing with nothing to say so.  `solve_numerical_complex`
        // accounts for every path (or refuses), and detects jumps.
        for p in solve_numerical_complex(equations, vars, pool, opts)? {
            if p.iter().all(|c| c.im == 0.0) {
                raw.push(p.iter().map(|c| c.re).collect());
            }
        }
    }
    if paths_started > 0 && paths_completed == 0 {
        return Err(HomotopyError::TrackerFailed(
            "no continuation path reached t = 1",
        ));
    }
    let uniq = dedup(&raw, opts.dedup_tol);
    let mut out = Vec::new();
    for x in uniq {
        let resv = fv_real(&sys, &x);
        let max_r = resv.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let (smale_alpha, smale_certified, rad) = match smale_estimate(&sys, &x) {
            Some((beta, alpha)) => {
                let cert = alpha < SMALE_THRESH;
                let r = if cert {
                    beta.clamp(1e-12, 0.05)
                } else {
                    1e-6_f64
                };
                (Some(alpha), cert, r)
            }
            None => (None, false, 1e-6_f64),
        };
        let enclosure = x
            .iter()
            .map(|&v| ArbBall::from_midpoint_radius(v, rad, prec))
            .collect();
        out.push(CertifiedPoint {
            coordinates: x,
            max_residual_f64: max_r,
            smale_alpha,
            smale_certified,
            enclosure,
        });
    }
    out.sort_by(|p, q| {
        let a = &p.coordinates;
        let b = &q.coordinates;
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// Projective total-degree homotopy (used by the complex solver)
//
// The affine tracker above follows a path that runs off to a solution at
// infinity into ever larger `z`, where its steps lose accuracy and it can land
// on a *finite* root that another path already owns — a jump that hides the
// root the other path should have reached.  Here every equation is
// homogenised with `Z₀` and tracked on a random affine chart `a·Z = 1` of ℙⁿ,
// so a solution at infinity is an ordinary point with `Z₀ = 0`: no path
// diverges, and none has to be guessed at.
// ---------------------------------------------------------------------------

/// `F̂ᵢ(Z₀, Z₁…Zₙ) = Z₀^{dᵢ} Fᵢ(Z/Z₀)`.
fn homogenize(p: &GbPoly, degree: u32) -> GbPoly {
    let mut terms = std::collections::BTreeMap::new();
    for (exp, coeff) in &p.terms {
        let k: u32 = exp.iter().sum();
        let mut e = Vec::with_capacity(exp.len() + 1);
        e.push(degree - k);
        e.extend_from_slice(exp);
        terms.insert(e, coeff.clone());
    }
    GbPoly {
        terms,
        n_vars: p.n_vars + 1,
    }
}

struct ProjectiveHomotopy {
    /// Homogenised target, one per affine equation (n of them, n + 1 vars).
    target: Vec<GbPoly>,
    /// ∂F̂ᵢ/∂Zⱼ, precomputed.
    target_jac: Vec<Vec<GbPoly>>,
    degs: Vec<u32>,
    gamma: C64,
    /// The chart `a·Z = 1`.
    chart: Vec<C64>,
}

impl ProjectiveHomotopy {
    fn new(sys: &[GbPoly], degs: &[u32], gamma: C64, chart: Vec<C64>) -> Self {
        let target: Vec<GbPoly> = sys
            .iter()
            .zip(degs.iter())
            .map(|(p, &d)| homogenize(p, d))
            .collect();
        let nv = sys.len() + 1;
        let target_jac = target
            .iter()
            .map(|p| (0..nv).map(|j| gbpoly_derive_var(p, j)).collect())
            .collect();
        ProjectiveHomotopy {
            target,
            target_jac,
            degs: degs.to_vec(),
            gamma,
            chart,
        }
    }

    /// `Gᵢ = γ(Zᵢ^{dᵢ} − Z₀^{dᵢ})`.
    fn start_value(&self, i: usize, z: &[C64]) -> C64 {
        let d = self.degs[i];
        C64::mul(
            self.gamma,
            C64::sub(C64::pow_int(z[i + 1], d), C64::pow_int(z[0], d)),
        )
    }

    fn chart_value(&self, z: &[C64]) -> C64 {
        let mut acc = C64::new(-1.0, 0.0);
        for (a, zi) in self.chart.iter().zip(z.iter()) {
            acc = C64::add(acc, C64::mul(*a, *zi));
        }
        acc
    }

    fn h(&self, z: &[C64], t: f64) -> Vec<C64> {
        let mut out: Vec<C64> = (0..self.target.len())
            .map(|i| {
                C64::add(
                    C64::scale(1.0 - t, self.start_value(i, z)),
                    C64::scale(t, gbpoly_eval_c(&self.target[i], z)),
                )
            })
            .collect();
        out.push(self.chart_value(z));
        out
    }

    fn dh_dt(&self, z: &[C64]) -> Vec<C64> {
        let mut out: Vec<C64> = (0..self.target.len())
            .map(|i| C64::sub(gbpoly_eval_c(&self.target[i], z), self.start_value(i, z)))
            .collect();
        out.push(C64::ZERO);
        out
    }

    fn jac(&self, z: &[C64], t: f64) -> Vec<Vec<C64>> {
        let n = self.target.len();
        let mut j = vec![vec![C64::ZERO; n + 1]; n + 1];
        for i in 0..n {
            for k in 0..=n {
                j[i][k] = C64::scale(t, gbpoly_eval_c(&self.target_jac[i][k], z));
            }
            let d = self.degs[i];
            let dg = |w: C64| C64::scale(d as f64, C64::pow_int(w, d - 1));
            let g_i = C64::scale(1.0 - t, C64::mul(self.gamma, dg(z[i + 1])));
            let g_0 = C64::scale(1.0 - t, C64::mul(self.gamma, dg(z[0])));
            j[i][i + 1] = C64::add(j[i][i + 1], g_i);
            j[i][0] = C64::sub(j[i][0], g_0);
        }
        j[n] = self.chart.clone();
        j
    }

    /// Newton on `H(·, t)`: at most a few plain steps, each required to
    /// contract.  No line search — a corrector that has to be coaxed is a
    /// corrector that may be converging onto a neighbouring path.
    fn correct(&self, z0: &[C64], t: f64, tol: f64) -> Option<Vec<C64>> {
        let mut z = z0.to_vec();
        let mut last_step = f64::INFINITY;
        for _ in 0..4 {
            let hv = self.h(&z, t);
            let neg: Vec<C64> = hv.iter().map(|c| C64::neg(*c)).collect();
            let step = complex_linsolve(self.jac(&z, t), neg)?;
            let size = step.iter().map(|c| c.norm2()).fold(0.0_f64, f64::max);
            if size > 0.5 * last_step {
                return None;
            }
            last_step = size;
            for (zi, s) in z.iter_mut().zip(step.iter()) {
                *zi = C64::add(*zi, *s);
            }
            let scale = z.iter().map(|c| c.norm2()).fold(1.0_f64, f64::max);
            if size <= tol * scale {
                return Some(z);
            }
        }
        None
    }

    /// Track from `z` at `t = 0` to `t = 1`; `Err` carries the last point.
    fn track(&self, mut z: Vec<C64>, opts: &HomotopyOpts) -> Result<Vec<C64>, Vec<C64>> {
        let mut t = 0.0_f64;
        let mut dt = opts.dt_initial;
        let mut steps = 0usize;
        while t < 1.0 {
            steps += 1;
            if steps > opts.max_tracker_steps {
                return Err(z);
            }
            let t_next = (t + dt).min(1.0);
            let rhs: Vec<C64> = self
                .dh_dt(&z)
                .into_iter()
                .map(|c| C64::scale(-(t_next - t), c))
                .collect();
            let Some(pred) = complex_linsolve(self.jac(&z, t), rhs) else {
                dt *= 0.5;
                if dt < opts.dt_min {
                    return Err(z);
                }
                continue;
            };
            let zp: Vec<C64> = z
                .iter()
                .zip(pred.iter())
                .map(|(a, b)| C64::add(*a, *b))
                .collect();
            match self.correct(&zp, t_next, 1e-11) {
                Some(zn) => {
                    // Reject a corrector that moved further than the predictor
                    // step itself — the signature of switching paths.
                    let pred_size = pred.iter().map(|c| c.norm2()).fold(0.0_f64, f64::max);
                    let corr = zn
                        .iter()
                        .zip(zp.iter())
                        .map(|(a, b)| C64::sub(*a, *b).norm2())
                        .fold(0.0_f64, f64::max);
                    if corr > 0.5 * pred_size.max(1e-12) {
                        dt *= 0.5;
                        if dt < opts.dt_min {
                            return Err(z);
                        }
                        continue;
                    }
                    z = zn;
                    t = t_next;
                    dt = (dt * 1.5).min(opts.dt_initial);
                }
                None => {
                    dt *= 0.5;
                    if dt < opts.dt_min {
                        return Err(z);
                    }
                }
            }
        }
        Ok(z)
    }

    /// Start point on the chart for the affine start root `w`.
    fn start_point(&self, w: &[C64]) -> Vec<C64> {
        let mut z = Vec::with_capacity(w.len() + 1);
        z.push(C64::new(1.0, 0.0));
        z.extend_from_slice(w);
        let mut s = C64::ZERO;
        for (a, zi) in self.chart.iter().zip(z.iter()) {
            s = C64::add(s, C64::mul(*a, *zi));
        }
        z.iter()
            .map(|zi| C64::div(*zi, s).unwrap_or(C64::new(f64::NAN, f64::NAN)))
            .collect()
    }
}

/// A deterministic pseudo-random point on the unit circle, from a seed and an
/// index (for the chart coefficients).
fn unit_from(seed: u64, k: u64) -> C64 {
    // splitmix64: neighbouring `k` give unrelated outputs.
    let mut x = seed.wrapping_add(k.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let ang = 2.0 * PI * ((x >> 11) as f64 / (1u64 << 53) as f64);
    C64::new(ang.cos(), ang.sin())
}

// ---------------------------------------------------------------------------
// Complex solution set
// ---------------------------------------------------------------------------

/// Complex Newton polish of a path endpoint.  Unlike [`newton_terminal`] it
/// never discards the point: at a multiple root Newton stalls or the Jacobian
/// goes singular, and the best iterate so far is still the root — whether it
/// is good enough is decided by [`accept_root`], not here.
fn newton_terminal_c(target: &[GbPoly], mut z: Vec<C64>, opts: &HomotopyOpts) -> Vec<C64> {
    let eval = |z: &[C64]| -> Vec<C64> { target.iter().map(|p| gbpoly_eval_c(p, z)).collect() };
    for _ in 0..opts.newton_cap {
        let f = eval(&z);
        let res = fv_linf(&f);
        if res == 0.0 {
            break;
        }
        let neg_f: Vec<C64> = f.iter().map(|c| C64::neg(*c)).collect();
        let Some(step) = complex_linsolve(jacobian_c(target, &z), neg_f) else {
            break;
        };
        let mut lm = 1.0_f64;
        let mut improved = false;
        while lm >= 1e-14 {
            let trial: Vec<C64> = z
                .iter()
                .zip(step.iter())
                .map(|(zi, s)| C64::add(*zi, C64::scale(lm, *s)))
                .collect();
            if fv_linf(&eval(&trial)) < res {
                z = trial;
                improved = true;
                break;
            }
            lm *= 0.5;
        }
        if !improved {
            break;
        }
    }
    z
}

/// Whether `z` is a root of `target` to double precision: every residual is
/// small next to the size of the terms that produced it (the rounding that
/// evaluating the polynomial in `f64` cannot avoid).
fn accept_root(target: &[GbPoly], z: &[C64]) -> bool {
    target.iter().all(|p| {
        let mut scale = 0.0_f64;
        for (exp, coeff) in &p.terms {
            let mut m = rat_to_f64(coeff).abs();
            for (i, &e) in exp.iter().enumerate() {
                if e != 0 {
                    m *= z[i].norm2().powi(e as i32);
                }
            }
            scale += m;
        }
        let r = gbpoly_eval_c(p, z).norm2();
        r.is_finite() && r <= 1e-7 * scale.max(1e-300)
    })
}

/// Every isolated **complex** solution of a square polynomial system, by
/// total-degree homotopy continuation.
///
/// [`solve_numerical`] reports real projections only (and documents it); this
/// is the function to use when the domain is ℂ — `solve(..., numeric=True)`
/// falls back to it when exact back-substitution meets a degree above 2.
///
/// It never quietly loses a root.  Every Bézout path must either arrive at a
/// point that is a root to double precision, or — for a system of two or more
/// equations, which may have solutions at infinity — be seen running off to
/// infinity.  A path that fails anywhere else raises `E-HOMOTOPY-004`, because
/// the set it would otherwise return could be missing that path's root.  A
/// single univariate equation has no roots at infinity, so there every path
/// must arrive.
///
/// Coordinates whose imaginary parts are negligible and that polish to a real
/// root are returned with `im == 0.0` exactly.  Multiple roots are returned
/// once (the result is a set), sorted by real then imaginary part.
pub fn solve_numerical_complex(
    equations: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
    opts: &HomotopyOpts,
) -> Result<Vec<Vec<ComplexF64>>, HomotopyError> {
    if equations.len() != vars.len() {
        return Err(HomotopyError::Algebraic(SolverError::ShapeMismatch));
    }
    let mut sys: Vec<GbPoly> = Vec::with_capacity(vars.len());
    for &eq in equations {
        sys.push(expr_to_gbpoly(eq, vars, pool).map_err(HomotopyError::Algebraic)?);
    }
    // A nonzero constant equation has no solutions; `0 = 0` constrains nothing,
    // so the solution set is not zero-dimensional.
    if sys.iter().any(|p| p.terms.is_empty()) {
        return Err(HomotopyError::TrackerFailed(
            "an equation is identically zero, so the solution set is not zero-dimensional",
        ));
    }
    if sys.iter().any(|p| total_degree(p) == 0) {
        return Ok(Vec::new());
    }
    let degs: Vec<u32> = sys.iter().map(total_degree).collect();
    let mut bez = 1usize;
    for &d in &degs {
        bez = bez
            .checked_mul(d as usize)
            .ok_or(HomotopyError::BezoutTooLarge(usize::MAX))?;
    }
    if bez > opts.max_bezout_paths {
        return Err(HomotopyError::BezoutTooLarge(bez));
    }

    if sys.len() == 1 {
        if let Some(roots) = univariate_complex_roots(equations[0], vars[0], pool) {
            let mut out: Vec<Vec<ComplexF64>> = roots.into_iter().map(|z| vec![z]).collect();
            sort_points(&mut out);
            return Ok(out);
        }
    }

    // Continuation.  A path can jump onto another path's root, and the root
    // it should have reached is then silently missing (dedup would hide the
    // duplicate).  [`track_all_complex`] detects the two signatures of a jump
    // — two paths ending at the same *regular* root (a simple root absorbs
    // exactly one path), or a non-real root whose complex conjugate never
    // turned up (the coefficients are rational, so the solution set is closed
    // under conjugation) — and the run is repeated with a new random γ and a
    // smaller step.  If every attempt shows a jump, the call refuses.
    let base_seed = opts.gamma_angle_seed.unwrap_or(31415926535897);
    let attempts: [(u64, f64); 4] = [
        (base_seed, 1.0),
        (base_seed ^ 0x9E37_79B9_7F4A_7C15, 1.0),
        (base_seed.wrapping_add(7919), 0.25),
        (
            base_seed.wrapping_mul(6364136223846793005).wrapping_add(1),
            0.0625,
        ),
    ];
    let mut chosen = None;
    for (seed, step_scale) in attempts {
        let run_opts = HomotopyOpts {
            dt_initial: opts.dt_initial * step_scale,
            max_tracker_steps: opts
                .max_tracker_steps
                .saturating_mul((1.0 / step_scale) as usize),
            ..opts.clone()
        };
        if let Some(points) = track_all_complex(&sys, &degs, Some(seed), &run_opts)? {
            chosen = Some(points);
            break;
        }
    }
    let Some(chosen) = chosen else {
        return Err(HomotopyError::TrackerFailed(
            "continuation paths kept jumping between roots, so no complete complex solution \
             set can be reported",
        ));
    };

    let mut out: Vec<Vec<ComplexF64>> = chosen
        .into_iter()
        .map(|z| {
            let nearly_real = z.iter().all(|c| c.im.abs() <= 1e-8 * (1.0 + c.re.abs()));
            if nearly_real {
                let xr: Vec<f64> = z.iter().map(|c| c.re).collect();
                if let Some(x) = newton_terminal(&sys, xr, opts) {
                    let zr: Vec<C64> = x.iter().map(|&r| C64::from_f64(r)).collect();
                    if accept_root(&sys, &zr) {
                        return x.into_iter().map(|r| ComplexF64::new(r, 0.0)).collect();
                    }
                }
            }
            z.into_iter().map(|c| ComplexF64::new(c.re, c.im)).collect()
        })
        .collect();
    sort_points(&mut out);
    Ok(out)
}

fn close_points(a: &[C64], b: &[C64], tol: f64) -> bool {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| C64::sub(*x, *y).norm2() / (1.0 + x.norm2()))
        .fold(0.0_f64, f64::max)
        < tol
}

/// Track every Bézout path for one random γ and chart; return the distinct
/// finite roots, or `None` when a path is seen to have jumped.
///
/// Paths are tracked in ℙⁿ (see [`ProjectiveHomotopy`]), so a path to a
/// solution at infinity simply ends with `Z₀ = 0` and is set aside.  Every
/// other path must end at a point that polishes to a root to double
/// precision; anything else is `E-HOMOTOPY-004`, since that path's root would
/// otherwise be missing from the answer.
///
/// A jump shows up as two paths ending at the same root while the Jacobian
/// there is regular (a simple root is the end of exactly one path), or as a
/// non-real root without its complex conjugate (the coefficients are
/// rational, so the solution set is closed under conjugation).
fn track_all_complex(
    sys: &[GbPoly],
    degs: &[u32],
    seed: Option<u64>,
    opts: &HomotopyOpts,
) -> Result<Option<Vec<Vec<C64>>>, HomotopyError> {
    let lost = "a continuation path failed before reaching a root, so the complex solution \
                set would be incomplete";
    let n = sys.len();
    let seed = seed.unwrap_or(31415926535897);
    let chart: Vec<C64> = (0..=n).map(|k| unit_from(seed, k as u64)).collect();
    let hom = ProjectiveHomotopy::new(sys, degs, random_gamma(Some(seed)), chart);
    // |Z₀| relative to the largest |Zᵢ|: how close to the hyperplane at infinity.
    let rel_z0 = |z: &[C64]| {
        let big = z[1..].iter().map(|c| c.norm2()).fold(0.0_f64, f64::max);
        if big == 0.0 {
            f64::INFINITY
        } else {
            z[0].norm2() / big
        }
    };
    // A univariate equation of degree d has no solution at infinity.
    let can_be_at_infinity = n > 1;

    let mut uniq: Vec<Vec<C64>> = Vec::new();
    for w in start_system_roots(degs) {
        let z_end = match hom.track(hom.start_point(&w), opts) {
            Ok(z) => z,
            // A path into a *singular* solution at infinity may stall just
            // short of `t = 1`, already next to the hyperplane.
            Err(z_last) if can_be_at_infinity && rel_z0(&z_last) < 1e-4 => continue,
            Err(_) => return Err(HomotopyError::TrackerFailed(lost)),
        };
        if can_be_at_infinity && rel_z0(&z_end) < 1e-10 {
            continue;
        }
        let affine: Vec<C64> = z_end[1..]
            .iter()
            .map(|c| C64::div(*c, z_end[0]).unwrap_or(C64::new(f64::NAN, f64::NAN)))
            .collect();
        let z = newton_terminal_c(sys, affine, opts);
        if !accept_root(sys, &z) {
            if can_be_at_infinity && rel_z0(&z_end) < 1e-4 {
                continue;
            }
            return Err(HomotopyError::TrackerFailed(lost));
        }
        // A root of multiplicity m is reached by m paths; a simple one by one.
        if uniq.iter().any(|u| close_points(u, &z, opts.dedup_tol)) {
            if jacobian_is_regular(sys, &z) {
                return Ok(None);
            }
        } else {
            uniq.push(z);
        }
    }
    let conj = |p: &[C64]| -> Vec<C64> { p.iter().map(|c| C64::new(c.re, -c.im)).collect() };
    for p in &uniq {
        let off_axis = p.iter().any(|c| c.im.abs() > 1e-8 * (1.0 + c.re.abs()));
        if off_axis
            && !uniq
                .iter()
                .any(|q| close_points(&conj(p), q, opts.dedup_tol))
        {
            return Ok(None);
        }
    }
    Ok(Some(uniq))
}

/// Whether the Jacobian of `sys` at `z` is comfortably nonsingular: every
/// pivot of a partially pivoted elimination is above `1e-6` of the largest
/// entry.  At a multiple root the computed point is only `√ε`-accurate and
/// the Jacobian is (numerically) singular, so this separates "two paths met at
/// a multiple root" from "a path jumped onto a simple root".
fn jacobian_is_regular(sys: &[GbPoly], z: &[C64]) -> bool {
    let mut a = jacobian_c(sys, z);
    let n = a.len();
    let scale = a
        .iter()
        .flatten()
        .map(|c| c.norm2())
        .fold(0.0_f64, f64::max);
    if !(scale > 0.0 && scale.is_finite()) {
        return false;
    }
    for col in 0..n {
        let (prow, best) = (col..n)
            .map(|r| (r, a[r][col].norm2()))
            .fold((col, -1.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        if best <= 1e-6 * scale {
            return false;
        }
        a.swap(prow, col);
        let piv = a[col][col];
        for row in col + 1..n {
            let Some(f) = C64::div(a[row][col], piv) else {
                return false;
            };
            for j in col..n {
                a[row][j] = C64::sub(a[row][j], C64::mul(f, a[col][j]));
            }
        }
    }
    true
}

fn sort_points(out: &mut [Vec<ComplexF64>]) {
    out.sort_by(|p, q| {
        for (a, b) in p.iter().zip(q.iter()) {
            let o =
                a.re.partial_cmp(&b.re)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal));
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
}

/// Complex roots of one univariate equation, without path tracking.
///
/// Path tracking can jump: two paths land on one root and the other root is
/// never reached, which dedup then hides.  For one variable there is a sharper
/// tool.  The exact squarefree part `q` of the numerator is formed over ℤ, its
/// roots are found by (rescaled) Durand–Kerner and Newton-polished, and the
/// answer is only accepted when
///
/// * there are exactly `deg q` of them, pairwise distinct,
/// * each passes a relative backward-error test, and
/// * the number snapped to the real axis is the **exact** real-root count of
///   `q` (Descartes/Vincent isolation in [`crate::poly::real_roots`]).
///
/// `None` means "not decided here" — the caller falls back to continuation.
fn univariate_complex_roots(eq: ExprId, var: ExprId, pool: &ExprPool) -> Option<Vec<ComplexF64>> {
    use crate::eval::root_sum::{horner_with_magnitude, polynomial_roots};
    let up = crate::poly::UniPoly::from_symbolic_clear_denoms(eq, var, pool).ok()?;
    if up.is_zero() {
        return None;
    }
    let q = up.squarefree_part();
    let degree = usize::try_from(q.degree()).ok()?;
    if degree == 0 {
        return Some(Vec::new());
    }
    let coeffs: Vec<f64> = q.coefficients().iter().map(rug::Integer::to_f64).collect();
    if coeffs.iter().any(|c| !c.is_finite()) {
        return None;
    }
    let n_real = crate::poly::real_roots(&q).ok()?.len();

    let deriv: Vec<f64> = coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, &c)| c * k as f64)
        .collect();
    let polish = |mut z: ComplexF64| -> ComplexF64 {
        for _ in 0..8 {
            let (v, _) = horner_with_magnitude(&coeffs, z);
            let (d, _) = horner_with_magnitude(&deriv, z);
            let dd = d.re * d.re + d.im * d.im;
            if dd == 0.0 || !dd.is_finite() {
                break;
            }
            let step = ComplexF64::new(
                (v.re * d.re + v.im * d.im) / dd,
                (v.im * d.re - v.re * d.im) / dd,
            );
            let next = ComplexF64::new(z.re - step.re, z.im - step.im);
            if !next.re.is_finite() || !next.im.is_finite() {
                break;
            }
            let (vn, _) = horner_with_magnitude(&coeffs, next);
            if vn.re.hypot(vn.im) > v.re.hypot(v.im) {
                break;
            }
            z = next;
        }
        z
    };

    let mut roots: Vec<ComplexF64> = polynomial_roots(&coeffs)?.into_iter().map(polish).collect();
    if roots.len() != degree {
        return None;
    }
    for &z in &roots {
        let (v, mag) = horner_with_magnitude(&coeffs, z);
        if !(v.re.hypot(v.im) <= 1e-9 * mag.max(f64::MIN_POSITIVE)) {
            return None;
        }
    }
    for i in 0..degree {
        for j in i + 1..degree {
            let (a, b) = (roots[i], roots[j]);
            let gap = (a.re - b.re).hypot(a.im - b.im);
            if gap <= 1e-9 * (1.0 + a.re.hypot(a.im)) {
                return None;
            }
        }
    }

    // Snap exactly `n_real` roots — the ones nearest the axis — to ℝ, and
    // insist the next one is visibly off it.
    roots.sort_by(|a, b| {
        a.im.abs()
            .partial_cmp(&b.im.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if n_real < degree {
        let z = roots[n_real];
        if z.im.abs() <= 1e-10 * (1.0 + z.re.abs()) {
            return None;
        }
    }
    for z in roots.iter_mut().take(n_real) {
        if z.im.abs() > 1e-6 * (1.0 + z.re.abs()) {
            return None;
        }
        *z = polish(ComplexF64::new(z.re, 0.0));
        z.im = 0.0;
    }
    Some(roots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn product_quadratics_four_real_roots() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let eq1 = pool.add(vec![pool.pow(x, pool.integer(2)), pool.integer(-1)]);
        let eq2 = pool.add(vec![pool.pow(y, pool.integer(2)), pool.integer(-1)]);
        let opts = HomotopyOpts {
            dedup_tol: 1e-4,
            ..Default::default()
        };
        let sols = solve_numerical(&[eq1, eq2], &[x, y], &pool, &opts).expect("solve");
        assert_eq!(sols.len(), 4, "±1 ⊗ ±1");
        assert!(sols.iter().all(|s| s.max_residual_f64 < 1e-8));
    }

    /// Systems the mixed volume routes away from the Bézout start must still
    /// produce their solutions.  `x²y − 1, xy² − 2` has MV 3 against a Bézout
    /// bound of 9, so it takes the polyhedral branch — which supplies no start
    /// points at all and used to hand back an empty list, i.e. the claim that
    /// a system with an obvious real solution has none.
    #[test]
    fn polyhedral_routed_system_still_finds_its_root() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // x²y − 1
        let eq1 = pool.add(vec![
            pool.mul(vec![pool.pow(x, pool.integer(2)), y]),
            pool.integer(-1),
        ]);
        // xy² − 2
        let eq2 = pool.add(vec![
            pool.mul(vec![x, pool.pow(y, pool.integer(2))]),
            pool.integer(-2),
        ]);
        let sys = [
            expr_to_gbpoly(eq1, &[x, y], &pool).unwrap(),
            expr_to_gbpoly(eq2, &[x, y], &pool).unwrap(),
        ];
        assert!(
            polyhedral::should_use_polyhedral(&sys),
            "this system is the polyhedral-routed case the test is about",
        );
        let opts = HomotopyOpts::default();
        let sols = solve_numerical(&[eq1, eq2], &[x, y], &pool, &opts).expect("solve");
        // x²y = 1 and xy² = 2 ⇒ (x²y)(xy²) = x³y³ = 2 ⇒ xy = 2^{1/3};
        // dividing xy² by x²y gives y/x = 2, so x = 2^{-1/3}, y = 2^{2/3}.
        let x0 = 2.0_f64.powf(-1.0 / 3.0);
        let y0 = 2.0_f64.powf(2.0 / 3.0);
        assert!(
            sols.iter()
                .any(|s| (s.coordinates[0] - x0).abs() < 1e-6
                    && (s.coordinates[1] - y0).abs() < 1e-6),
            "expected ({x0}, {y0}) among {sols:?}",
        );
    }

    #[test]
    fn circle_line_two_real_roots() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let eq1 = pool.add(vec![
            pool.pow(x, pool.integer(2)),
            pool.pow(y, pool.integer(2)),
            pool.integer(-1),
        ]);
        let neg_one = pool.integer(-1);
        let eq2 = pool.add(vec![y, pool.mul(vec![neg_one, x])]);
        let opts = HomotopyOpts {
            dedup_tol: 1e-4,
            ..Default::default()
        };
        let sols = solve_numerical(&[eq1, eq2], &[x, y], &pool, &opts).expect("solve");
        let r = 0.5_f64.sqrt();
        let mut matched = 0_usize;
        for s in &sols {
            let (xv, yv) = (s.coordinates[0], s.coordinates[1]);
            let ok = ((xv - r).abs() < 5e-3 && (yv - r).abs() < 5e-3)
                || ((xv + r).abs() < 5e-3 && (yv + r).abs() < 5e-3);
            if ok {
                matched += 1;
            }
        }
        assert_eq!(matched, 2, "{sols:?}");
    }

    fn univariate(pool: &ExprPool, x: ExprId, coeffs: &[i64]) -> ExprId {
        let terms: Vec<ExprId> = coeffs
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 0)
            .map(|(k, &c)| pool.mul(vec![pool.integer(c), pool.pow(x, pool.integer(k as i64))]))
            .collect();
        pool.add(terms)
    }

    fn close(a: ComplexF64, re: f64, im: f64) -> bool {
        (a.re - re).abs() < 1e-9 && (a.im - im).abs() < 1e-9
    }

    /// `s⁴ + 1` has no real root; the real-only solver returns nothing, which
    /// `solve(numeric=True)` used to pass on as "no solutions".
    #[test]
    fn complex_solver_finds_all_fourth_roots_of_minus_one() {
        let pool = ExprPool::new();
        let s = pool.symbol("s", Domain::Complex);
        let eq = univariate(&pool, s, &[1, 0, 0, 0, 1]);
        let sols = solve_numerical_complex(&[eq], &[s], &pool, &HomotopyOpts::default()).unwrap();
        assert_eq!(sols.len(), 4, "{sols:?}");
        let h = std::f64::consts::FRAC_1_SQRT_2;
        for (re, im) in [(h, h), (h, -h), (-h, h), (-h, -h)] {
            assert!(
                sols.iter().any(|p| close(p[0], re, im)),
                "{re}+{im}i missing"
            );
        }
        // And the real-only solver really does see none — the documented contract.
        assert!(
            solve_numerical(&[eq], &[s], &pool, &HomotopyOpts::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn complex_solver_keeps_the_real_root_real() {
        let pool = ExprPool::new();
        let s = pool.symbol("s", Domain::Complex);
        let eq = univariate(&pool, s, &[1, 0, 0, 1]); // s³ + 1
        let sols = solve_numerical_complex(&[eq], &[s], &pool, &HomotopyOpts::default()).unwrap();
        assert_eq!(sols.len(), 3, "{sols:?}");
        assert!(sols.iter().any(|p| p[0].re == -1.0 && p[0].im == 0.0));
        let r3 = 3f64.sqrt() / 2.0;
        assert!(sols.iter().any(|p| close(p[0], 0.5, r3)));
        assert!(sols.iter().any(|p| close(p[0], 0.5, -r3)));
    }

    /// `(x − 1)²(x + 2)`: two paths meet at the double root, reported once.
    #[test]
    fn complex_solver_reports_a_double_root_once() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Complex);
        let eq = univariate(&pool, x, &[2, -3, 0, 1]);
        let sols = solve_numerical_complex(&[eq], &[x], &pool, &HomotopyOpts::default()).unwrap();
        assert_eq!(sols.len(), 2, "{sols:?}");
        assert!(sols.iter().any(|p| (p[0].re + 2.0).abs() < 1e-9));
        assert!(sols.iter().any(|p| (p[0].re - 1.0).abs() < 1e-6));
    }

    /// `x²y − 1, xy² − 2`: Bézout 9, but only 3 finite solutions — the other
    /// six paths go to infinity, which is not a lost root.
    #[test]
    fn complex_solver_tells_roots_at_infinity_from_lost_roots() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Complex);
        let y = pool.symbol("y", Domain::Complex);
        let eq1 = pool.add(vec![
            pool.mul(vec![pool.pow(x, pool.integer(2)), y]),
            pool.integer(-1),
        ]);
        let eq2 = pool.add(vec![
            pool.mul(vec![x, pool.pow(y, pool.integer(2))]),
            pool.integer(-2),
        ]);
        let sols =
            solve_numerical_complex(&[eq1, eq2], &[x, y], &pool, &HomotopyOpts::default()).unwrap();
        // x³ = 1/2, y = 2x: the three cube roots.
        assert_eq!(sols.len(), 3, "{sols:?}");
        for p in &sols {
            let (zx, zy) = (C64::new(p[0].re, p[0].im), C64::new(p[1].re, p[1].im));
            let r1 = C64::sub(C64::mul(C64::mul(zx, zx), zy), C64::from_f64(1.0));
            assert!(r1.norm2() < 1e-9, "{p:?}");
        }
    }
}
