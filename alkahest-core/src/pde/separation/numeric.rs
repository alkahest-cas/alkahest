//! Numeric companions of the series solution: the projection check (which
//! may only refute), and numeric truncations for fast evaluation.

use std::collections::HashMap;

use crate::ball::IntervalEval;
use crate::kernel::subs::subs;
use crate::kernel::{ExprData, ExprId, ExprPool};

use super::super::error::PdeError;
use super::super::jet::{push_children, symbols_of};
use super::super::root_family::{root_family_enclosure, root_family_parts};
use super::super::sturm_liouville::simp;
use super::super::verify::VerificationStatus;
use super::project::{has_integral, integral_parts};
use super::solve::ProjectionCheck;

/// Gauss–Legendre nodes and weights on `[−1, 1]`.
pub(crate) fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < 1e-15 {
                w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
                break;
            }
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        }
        x[i] = z;
    }
    (x, w)
}

fn eval(e: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
    crate::jit::eval_interp_real(e, env, pool)
        .ok()
        .or_else(|| crate::eval::eval_f64(e, pool, env).ok())
        .filter(|v| v.is_finite())
}

/// `∫ f` over the box `ranges` by composite Gauss–Legendre (`panels ×
/// nodes` per dimension).
fn quad(
    f: ExprId,
    ranges: &[(ExprId, f64, f64)],
    env: &HashMap<ExprId, f64>,
    panels: usize,
    nodes: usize,
    pool: &ExprPool,
) -> Option<f64> {
    let (gx, gw) = gauss_legendre(nodes);
    fn rec(
        f: ExprId,
        ranges: &[(ExprId, f64, f64)],
        env: &mut HashMap<ExprId, f64>,
        panels: usize,
        gx: &[f64],
        gw: &[f64],
        pool: &ExprPool,
    ) -> Option<f64> {
        let Some(&(v, a, b)) = ranges.first() else {
            return eval(f, env, pool);
        };
        let h = (b - a) / panels as f64;
        let mut total = 0.0;
        for p in 0..panels {
            let lo = a + p as f64 * h;
            for (xi, wi) in gx.iter().zip(gw) {
                let x = lo + 0.5 * h * (xi + 1.0);
                env.insert(v, x);
                total += 0.5 * h * wi * rec(f, &ranges[1..], env, panels, gx, gw, pool)?;
            }
        }
        env.remove(&v);
        Some(total)
    }
    let mut env = env.clone();
    rec(f, ranges, &mut env, panels, &gx, &gw, pool)
}

fn is_named_constant(s: ExprId, pool: &ExprPool) -> bool {
    pool.with(s, |d| match d {
        ExprData::Symbol { name, .. } => crate::eval::symbols::constant_value_f64(name).is_some(),
        _ => false,
    })
}

const SAMPLE: [f64; 8] = [1.3, 0.7, 1.9, 1.1, 2.3, 0.9, 1.7, 0.6];

/// Compare the closed-form coefficients of the first members against
/// quadratures of the projections.  `None` when nothing applies.
#[allow(clippy::too_many_arguments)]
pub(crate) fn projection_check(
    indices: &[(ExprId, ExprId)],
    uppers: &[ExprId],
    ranges: &[(ExprId, ExprId, ExprId)],
    weight: ExprId,
    phis: &[ExprId],
    norms: &[ExprId],
    coefficients: &[Vec<(String, ExprId)>],
    data: &[(String, ExprId)],
    count: usize,
    pool: &ExprPool,
) -> Option<Vec<ProjectionCheck>> {
    if ranges.is_empty() || data.is_empty() {
        return None;
    }
    let range_vars: Vec<ExprId> = ranges.iter().map(|r| r.0).collect();
    let index_syms: Vec<ExprId> = indices.iter().map(|r| r.0).collect();
    let mut out = Vec::new();
    for (name, h) in data {
        if crate::diff::undefined::has_undefined_function(*h, pool) {
            out.push(ProjectionCheck {
                data: static_name(name),
                status: VerificationStatus::Unverified,
                terms: 0,
                max_error: 0.0,
                evidence: vec!["symbolic data (an arbitrary function): nothing to evaluate".into()],
            });
            continue;
        }
        // Parameters → sample values.
        let mut params: Vec<ExprId> = Vec::new();
        let mut collect = |e: ExprId| {
            for s in symbols_of(e, pool) {
                if !range_vars.contains(&s)
                    && !index_syms.contains(&s)
                    && !is_named_constant(s, pool)
                    && !params.contains(&s)
                {
                    params.push(s);
                }
            }
        };
        collect(*h);
        collect(weight);
        for &p in phis {
            collect(p);
        }
        for &n in norms {
            collect(n);
        }
        for cs in coefficients {
            for (n, c) in cs {
                if n == name {
                    collect(*c);
                }
            }
        }
        // Bound variables of RootOfFamily / Sum nodes are not parameters.
        params.retain(|&s| !bound_somewhere(s, weight, phis, pool));
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        let mut evidence = Vec::new();
        for (k, &s) in params.iter().enumerate() {
            let v = SAMPLE[k % SAMPLE.len()];
            env.insert(s, v);
            evidence.push(format!("{} = {v}", pool.display(s)));
        }
        let exact_env: HashMap<ExprId, ExprId> =
            env.iter().map(|(&k, &v)| (k, pool.float(v, 53))).collect();
        // Index tuples.
        let mut tuples: Vec<Vec<(ExprId, i64)>> = vec![vec![]];
        for (ii, &(idx, first)) in indices.iter().enumerate() {
            let last = uppers.get(ii).and_then(|&u| {
                pool.with(u, |d| match d {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                })
            });
            let mut next = Vec::new();
            for t in &tuples {
                let m: HashMap<ExprId, ExprId> =
                    t.iter().map(|&(i, v)| (i, pool.integer(v))).collect();
                let f0 = simp(subs(first, &m, pool), pool);
                let Some(f0) = pool.with(f0, |d| match d {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                }) else {
                    return None;
                };
                let hi = match last {
                    Some(l) => (f0 + count as i64).min(l + 1),
                    None => f0 + count as i64,
                };
                for k in f0..hi {
                    let mut tt = t.clone();
                    tt.push((idx, k));
                    next.push(tt);
                }
            }
            tuples = next;
            if tuples.len() > 24 {
                tuples.truncate(24);
            }
        }
        let mut max_err = 0.0f64;
        let mut terms = 0;
        let mut refuted = false;
        let mut ball_used = false;
        let mut scale = 0.0f64;
        let mut rows = Vec::new();
        for t in &tuples {
            let mut m: HashMap<ExprId, ExprId> = exact_env.clone();
            for &(i, v) in t {
                m.insert(i, pool.integer(v));
            }
            let mut fenv = env.clone();
            for &(i, v) in t {
                fenv.insert(i, v as f64);
            }
            for (vi, cs) in coefficients.iter().enumerate() {
                let Some((_, c)) = cs.iter().find(|(n, _)| n == name) else {
                    continue;
                };
                if has_integral(*c, pool) {
                    continue;
                }
                let cnum = simp(subs(*c, &m, pool), pool);
                let cnum = numeric_root_families(cnum, pool).unwrap_or(cnum);
                let undefined = super::project::has_singular_value(cnum, pool);
                let cv = match IntervalEval::new(96).eval(cnum, pool) {
                    Some(b)
                        if b.rad.is_finite()
                            && b.rad.to_f64() < 1e-12 * (1.0 + b.mid.to_f64().abs()) =>
                    {
                        ball_used = true;
                        b.mid.to_f64()
                    }
                    _ => match eval(cnum, &HashMap::new(), pool) {
                        Some(v) => v,
                        None if undefined => f64::NAN,
                        None => continue,
                    },
                };
                let f = pool.mul(vec![weight, *h, phis[vi]]);
                let f = simp(subs(f, &m, pool), pool);
                let f = numeric_root_families(f, pool).unwrap_or(f);
                let nr = simp(subs(norms[vi], &m, pool), pool);
                let nr = numeric_root_families(nr, pool).unwrap_or(nr);
                let Some(nv) = eval(nr, &HashMap::new(), pool) else {
                    continue;
                };
                let mut rr = Vec::new();
                let mut ok = true;
                for &(v, lo, hi) in ranges {
                    let lo = eval(simp(subs(lo, &m, pool), pool), &HashMap::new(), pool);
                    let hi = eval(simp(subs(hi, &m, pool), pool), &HashMap::new(), pool);
                    match (lo, hi) {
                        (Some(a), Some(b)) => rr.push((v, a, b)),
                        _ => ok = false,
                    }
                }
                if !ok {
                    continue;
                }
                let (p1, n1) = match ranges.len() {
                    1 => (8, 16),
                    2 => (4, 12),
                    _ => (2, 8),
                };
                let q1 = quad(f, &rr, &HashMap::new(), p1, n1, pool);
                let q2 = quad(f, &rr, &HashMap::new(), 2 * p1, n1, pool);
                let (Some(q1), Some(q2)) = (q1, q2) else {
                    continue;
                };
                let (q1, q2) = (q1 / nv, q2 / nv);
                scale = scale
                    .max(q2.abs())
                    .max(if cv.is_finite() { cv.abs() } else { 0.0 });
                let err = if cv.is_finite() {
                    (cv - q2).abs()
                } else {
                    f64::INFINITY
                };
                let qerr = (q2 - q1).abs();
                terms += 1;
                max_err = max_err.max(err / (1.0 + scale));
                rows.push((t.clone(), cv, q2, err, qerr));
            }
        }
        for (t, cv, q2, err, qerr) in &rows {
            let tol = 10.0 * qerr + 1e-8 * (1.0 + scale);
            let idx: Vec<String> = t
                .iter()
                .map(|&(i, v)| format!("{}={v}", pool.display(i)))
                .collect();
            if *err > tol {
                refuted = true;
                evidence.push(format!(
                    "{name}[{}]: formula {cv:.12e}, quadrature {q2:.12e} (quadrature error ≈ {qerr:.1e})",
                    idx.join(",")
                ));
            } else if evidence.len() < 12 {
                evidence.push(format!("{name}[{}]: {cv:.10e} ≈ {q2:.10e}", idx.join(",")));
            }
        }
        if terms == 0 {
            continue;
        }
        evidence.push(if ball_used {
            "coefficients evaluated in ball arithmetic (Arb); projections by composite \
             Gauss–Legendre at two resolutions"
                .into()
        } else {
            "coefficients evaluated in f64; projections by composite Gauss–Legendre at two \
             resolutions"
                .into()
        });
        out.push(ProjectionCheck {
            data: static_name(name),
            status: if refuted {
                VerificationStatus::Refuted
            } else {
                VerificationStatus::NumericallyChecked
            },
            terms,
            max_error: max_err,
            evidence,
        });
    }
    Some(out)
}

fn static_name(n: &str) -> &'static str {
    match n {
        "ic" => "ic",
        "ic1" => "ic1",
        "left" => "left",
        "right" => "right",
        _ => "source",
    }
}

fn bound_somewhere(s: ExprId, weight: ExprId, phis: &[ExprId], pool: &ExprPool) -> bool {
    let mut stack: Vec<ExprId> = phis.to_vec();
    stack.push(weight);
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some((_, lam, _, _)) = root_family_parts(n, pool) {
            if lam == s {
                return true;
            }
        }
        if let Some((_, k, _, _)) = super::super::sum::sum_parts(n, pool) {
            if k == s {
                return true;
            }
        }
        if let Some((_, v, _, _)) = integral_parts(n, pool) {
            if v == s {
                return true;
            }
        }
        push_children(n, pool, &mut stack);
    }
    false
}

/// Replace every `RootOfFamily` member with numeric data by a float (the
/// midpoint of its certified enclosure).
pub(crate) fn numeric_root_families(e: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let mut map = HashMap::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if root_family_parts(n, pool).is_some() {
            let (lo, hi) = root_family_enclosure(n, 64, pool)?;
            let mid = ((lo + hi) / 2u32).to_f64();
            map.insert(n, pool.float(mid, 53));
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    if map.is_empty() {
        return Ok(e);
    }
    Ok(simp(super::separate::replace_nodes(e, &map, pool), pool))
}

/// Replace every `Integral` placeholder whose integrand and bounds are
/// numeric (after substitution) by its quadrature.
fn numeric_integrals(e: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let mut map = HashMap::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some((body, v, lo, hi)) = integral_parts(n, pool) {
            let body = numeric_integrals(body, pool)?;
            let lo_v = eval(lo, &HashMap::new(), pool);
            let hi_v = eval(hi, &HashMap::new(), pool);
            let (Some(a), Some(b)) = (lo_v, hi_v) else {
                return Err(PdeError::SeparationTruncation(format!(
                    "the bounds of {} are not numeric",
                    pool.display(n)
                )));
            };
            let val = quad(body, &[(v, a, b)], &HashMap::new(), 16, 20, pool).ok_or_else(|| {
                PdeError::SeparationTruncation(format!(
                    "cannot evaluate {} numerically (free symbols?)",
                    pool.display(n)
                ))
            })?;
            map.insert(n, pool.float(val, 53));
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    if map.is_empty() {
        return Ok(e);
    }
    Ok(simp(super::separate::replace_nodes(e, &map, pool), pool))
}

/// The truncation of a series solution to indices `≤ n`, with `params`
/// substituted, every `RootOfFamily` member replaced by a float (its certified
/// enclosure's midpoint) and every unevaluated coefficient integral by its
/// quadrature — an explicit expression ready for `compile`.
///
/// # Errors
///
/// `E-PDE-209` when a `RootOfFamily` or `Integral` is not numeric after the
/// substitution; `E-PDE-021` from the truncation.
pub fn numeric_truncation(
    expr: ExprId,
    n: i64,
    params: &HashMap<ExprId, ExprId>,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let e = simp(subs(expr, params, pool), pool);
    let t = super::super::sum::truncate(e, n, pool)?;
    let t = numeric_root_families(t, pool).map_err(|err| {
        PdeError::SeparationTruncation(format!(
            "an eigenvalue family is not numeric (give values for its parameters): {err}"
        ))
    })?;
    let t = numeric_integrals(t, pool)?;
    Ok(simp(t, pool))
}
