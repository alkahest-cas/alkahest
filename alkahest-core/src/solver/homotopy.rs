//! Numerical algebraic geometry — total-degree homotopy continuation (V2-14).
//!
//! We track `(1−t)·γ·G(z) + t·F(z) = 0` from `t=0→1`, with decoupled start
//! `G_i(z) = z_i^{d_i} − 1` and `d_i` the total degree of `F_i`.  The Bézout
//! start count ∏ d_i reaches the affine root count only for sufficiently generic
//! **dense** systems; **deficient** families (fewer finite roots than the
//! Bézout bound — e.g. Katsura) need a polyhedral / mixed-volume start; that
//! is out of scope here.
//!
//! Endpoints are Newton-polished in ℝⁿ and checked with a conservative Smale
//! heuristic plus `ArbBall` enclosures.

#![allow(clippy::needless_range_loop)]

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use crate::poly::groebner::GbPoly;
use crate::solver::{expr_to_gbpoly, SolverError};
use rug::Rational;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
struct C64 {
    re: f64,
    im: f64,
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
    p.terms
        .keys()
        .map(|e| e.iter().sum::<u32>())
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

fn hv(gamma: C64, target: &[GbPoly], start_degrees: &[u32], z: &[C64], t: f64) -> Vec<C64> {
    let one = C64::new(1.0, 0.0);
    let mt = C64::new(1.0 - t, 0.0);
    let tt = C64::new(t, 0.0);
    let mut h = Vec::with_capacity(z.len());
    for i in 0..z.len() {
        let f = gbpoly_eval_c(&target[i], z);
        let d = start_degrees[i];
        let mon = C64::sub(C64::pow_int(z[i], d), one);
        let g = C64::mul(gamma, mon);
        h.push(C64::add(C64::mul(mt, g), C64::mul(tt, f)));
    }
    h
}

fn dh_dt(gamma: C64, target: &[GbPoly], start_degrees: &[u32], z: &[C64]) -> Vec<C64> {
    let one = C64::new(1.0, 0.0);
    let mut out = Vec::with_capacity(z.len());
    for i in 0..z.len() {
        let f = gbpoly_eval_c(&target[i], z);
        let d = start_degrees[i];
        let mon = C64::sub(C64::pow_int(z[i], d), one);
        out.push(C64::sub(f, C64::mul(gamma, mon)));
    }
    out
}

fn jh(gamma: C64, target: &[GbPoly], start_degrees: &[u32], z: &[C64], t: f64) -> Vec<Vec<C64>> {
    let n = z.len();
    let j_f = jacobian_c(target, z);
    let mt = C64::new(1.0 - t, 0.0);
    let tt = C64::new(t, 0.0);
    let mut jac = vec![vec![C64::ZERO; n]; n];
    for i in 0..n {
        for k in 0..n {
            let mut elt = C64::mul(tt, j_f[i][k]);
            if i == k {
                let di = start_degrees[i];
                let deriv_g = if di >= 1 {
                    C64::scale(di as f64, C64::pow_int(z[i], di - 1))
                } else {
                    C64::ZERO
                };
                elt = C64::add(elt, C64::mul(C64::mul(mt, gamma), deriv_g));
            }
            jac[i][k] = elt;
        }
    }
    jac
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

fn damped_correct(
    gamma: C64,
    target: &[GbPoly],
    degs: &[u32],
    z0: &[C64],
    t_tgt: f64,
    opts: &HomotopyOpts,
) -> Option<Vec<C64>> {
    let mut z = z0.to_vec();
    for _ in 0..opts.newton_cap {
        let fv = hv(gamma, target, degs, &z, t_tgt);
        let res = fv_linf(&fv);
        if res < opts.homotopy_tol {
            return Some(z);
        }
        let jac = jh(gamma, target, degs, &z, t_tgt);
        let neg_f: Vec<C64> = fv.iter().map(|c| C64::neg(*c)).collect();
        let step = complex_linsolve(jac, neg_f)?;
        let mut lm = 1.0_f64;
        loop {
            let trial: Vec<C64> = z
                .iter()
                .zip(step.iter())
                .map(|(zi, s)| C64::add(*zi, C64::scale(lm, *s)))
                .collect();
            let new_res = fv_linf(&hv(gamma, target, degs, &trial, t_tgt));
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
    fv_linf(&hv(gamma, target, degs, &z, t_tgt))
        .le(&(opts.homotopy_tol * 8.0))
        .then_some(z)
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

fn track_path(
    gamma: C64,
    target: &[GbPoly],
    degs: &[u32],
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
        let jac = jh(gamma, target, degs, &z, t);
        let htd = dh_dt(gamma, target, degs, &z);
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
        match damped_correct(gamma, target, degs, &zp, t_next, opts) {
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

/// Total-degree continuation + polishing + Smale / ArbBall packaging.
///
/// Returns **real projections** whose imaginary tails were negligible; complex
/// roots with large imaginary part are discarded.
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
    if bez > opts.max_bezout_paths {
        return Err(HomotopyError::BezoutTooLarge(bez));
    }
    let starts = start_system_roots(&degs);
    let gamma = random_gamma(opts.gamma_angle_seed);
    let prec = opts.certify_prec_bits;
    const SMALE_THRESH: f64 = 0.125;
    let mut raw: Vec<Vec<f64>> = Vec::new();
    for z0 in starts {
        let z_end = match track_path(gamma, &sys, &degs, z0, opts) {
            Ok(z) => z,
            Err(_) => continue,
        };
        if z_end.iter().all(|c| c.im.abs() < 1e-6) {
            let xr: Vec<f64> = z_end.iter().map(|c| c.re).collect();
            if let Some(xp) = newton_terminal(&sys, xr, opts) {
                raw.push(xp);
            }
        }
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
}
