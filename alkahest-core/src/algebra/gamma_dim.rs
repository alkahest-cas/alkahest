//! d-dimensional Dirac-gamma algebra (F20).
//!
//! The four-dimensional [`gamma_trace`](super::nc_poly::gamma_trace) works with
//! explicit components `0…3`. Dimensional regularisation needs the algebra
//! with a *symbolic* dimension `d`, where only the Clifford relation
//! `{γ^μ, γ^ν} = 2 g^{μν}`, the cyclic trace with `tr 1 = 4`, and the metric
//! contraction `g^{μν} g_{μν} = d` are available:
//!
//! * [`gamma_trace_dim`]: the trace of a product of slashed vectors / Lorentz
//!   indices, with the repeated (dummy) indices contracted;
//! * [`lorentz_contract`]: contract dummy indices in any polynomial of
//!   `dot(a, b)` factors — e.g. a *product of traces*;
//! * [`gamma_contract`]: remove contracted pairs from a gamma string
//!   (`γ^μ γ_μ = d`, `γ^μ γ^ν γ_μ = (2 − d) γ^ν`, …).
//!
//! γ⁵ has no consistent anticommuting d-dimensional extension, so it (and any
//! explicit component index) is refused (`E-NC-005`); the four-dimensional
//! `gamma_trace` is the place for it.
//!
//! Every identity is derived from the Clifford relation alone, so the results
//! are valid for any `d`; at `d = 4` they agree with explicit 4×4 matrices.

use super::nc_poly::NcError;
use crate::budget;
use crate::kernel::subs::subs;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// Cap on the number of labels in one trace / gamma string.
const MAX_SLOTS: usize = 14;
/// Cap on terms held while expanding a product for contraction.
const MAX_TERMS: usize = 200_000;

fn bad(reason: String) -> NcError {
    NcError::InvalidGamma { reason }
}

fn too_large(what: String) -> NcError {
    NcError::TooLarge { what }
}

fn is_dot(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    pool.with(expr, |d| match d {
        ExprData::Func { name, args } if name == "dot" && args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    })
}

fn dot_pair(a: ExprId, b: ExprId) -> (ExprId, ExprId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn mk_dot(p: (ExprId, ExprId), pool: &ExprPool) -> ExprId {
    pool.func("dot", vec![p.0, p.1])
}

fn small_int(expr: ExprId, pool: &ExprPool) -> Option<i64> {
    pool.with(expr, |d| match d {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    })
}

fn check_labels(
    slots: &[ExprId],
    indices: &[ExprId],
    pool: &ExprPool,
    what: &str,
) -> Result<(), NcError> {
    for &s in slots {
        let ok = pool.is_mult_commutative(s)
            && pool.with(s, |d| {
                matches!(d, ExprData::Symbol { .. } | ExprData::Func { .. })
            });
        if !ok {
            return Err(bad(format!(
                "{what}: label `{}` must be a commutative symbol (explicit components and γ⁵ are four-dimensional only)",
                pool.display(s)
            )));
        }
    }
    for (i, &x) in indices.iter().enumerate() {
        if !pool.with(x, |d| matches!(d, ExprData::Symbol { .. })) {
            return Err(bad(format!(
                "{what}: Lorentz index `{}` must be a symbol",
                pool.display(x)
            )));
        }
        if indices[..i].contains(&x) {
            return Err(bad(format!(
                "{what}: Lorentz index `{}` listed twice",
                pool.display(x)
            )));
        }
    }
    Ok(())
}

fn dim_value(dim: ExprId, pool: &ExprPool) -> Result<Option<i64>, NcError> {
    match small_int(dim, pool) {
        Some(n) if n >= 1 => Ok(Some(n)),
        Some(n) => Err(bad(format!("dimension {n} must be a positive integer"))),
        None => {
            if pool.with(dim, |d| {
                matches!(
                    d,
                    ExprData::Symbol { .. }
                        | ExprData::Add(_)
                        | ExprData::Mul(_)
                        | ExprData::Pow { .. }
                )
            }) {
                Ok(None)
            } else {
                Err(bad(format!(
                    "dimension `{}` must be a positive integer or a symbolic expression",
                    pool.display(dim)
                )))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Contraction of dot-polynomials
// ---------------------------------------------------------------------------

/// One monomial: `coef · d^dpow · Π other · Π dot(a, b)`.
#[derive(Clone)]
struct CTerm {
    coef: i64,
    dpow: u32,
    other: Vec<ExprId>,
    dots: Vec<(ExprId, ExprId)>,
}

fn mul_terms(a: &[CTerm], b: &[CTerm]) -> Result<Vec<CTerm>, NcError> {
    if a.len().saturating_mul(b.len()) > MAX_TERMS {
        return Err(too_large(format!(
            "{} terms while contracting (cap {MAX_TERMS})",
            a.len().saturating_mul(b.len())
        )));
    }
    let mut out = Vec::with_capacity(a.len() * b.len());
    for x in a {
        for y in b {
            let coef = x
                .coef
                .checked_mul(y.coef)
                .ok_or_else(|| too_large("coefficient overflow while contracting".into()))?;
            let mut other = x.other.clone();
            other.extend_from_slice(&y.other);
            let mut dots = x.dots.clone();
            dots.extend_from_slice(&y.dots);
            out.push(CTerm {
                coef,
                dpow: x.dpow + y.dpow,
                other,
                dots,
            });
        }
    }
    budget::check()?;
    Ok(out)
}

fn unit_term(coef: i64) -> CTerm {
    CTerm {
        coef,
        dpow: 0,
        other: Vec::new(),
        dots: Vec::new(),
    }
}

fn expand(expr: ExprId, pool: &ExprPool) -> Result<Vec<CTerm>, NcError> {
    enum Sh {
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Other,
    }
    if let Some(n) = small_int(expr, pool) {
        return Ok(if n == 0 { vec![] } else { vec![unit_term(n)] });
    }
    if let Some(p) = is_dot(expr, pool) {
        let mut t = unit_term(1);
        t.dots.push(dot_pair(p.0, p.1));
        return Ok(vec![t]);
    }
    let sh = pool.with(expr, |d| match d {
        ExprData::Add(v) => Sh::Add(v.clone()),
        ExprData::Mul(v) => Sh::Mul(v.clone()),
        ExprData::Pow { base, exp } => Sh::Pow(*base, *exp),
        _ => Sh::Other,
    });
    match sh {
        Sh::Add(v) => {
            let mut out = Vec::new();
            for a in v {
                out.extend(expand(a, pool)?);
                if out.len() > MAX_TERMS {
                    return Err(too_large(format!("more than {MAX_TERMS} terms")));
                }
            }
            Ok(out)
        }
        Sh::Mul(v) => {
            let mut acc = vec![unit_term(1)];
            for a in v {
                let e = expand(a, pool)?;
                acc = mul_terms(&acc, &e)?;
            }
            Ok(acc)
        }
        Sh::Pow(b, e) => match small_int(e, pool) {
            Some(n) if (1..=64).contains(&n) => {
                let base = expand(b, pool)?;
                let mut acc = base.clone();
                for _ in 1..n {
                    acc = mul_terms(&acc, &base)?;
                }
                Ok(acc)
            }
            _ => {
                let mut t = unit_term(1);
                t.other.push(expr);
                Ok(vec![t])
            }
        },
        Sh::Other => {
            let mut t = unit_term(1);
            t.other.push(expr);
            Ok(vec![t])
        }
    }
}

/// Contract every dummy index of `t`.
fn contract_term(t: &mut CTerm, indices: &[ExprId], dim_val: Option<i64>) -> Result<(), NcError> {
    loop {
        let mut changed = false;
        for &mu in indices {
            let count: usize = t
                .dots
                .iter()
                .map(|&(a, b)| (a == mu) as usize + (b == mu) as usize)
                .sum();
            if count < 2 {
                continue;
            }
            if count > 2 {
                return Err(bad(format!(
                    "Lorentz index occurs {count} times in one term (a dummy index appears exactly twice)"
                )));
            }
            if let Some(pos) = t.dots.iter().position(|&(a, b)| a == mu && b == mu) {
                t.dots.swap_remove(pos);
                match dim_val {
                    Some(n) => {
                        t.coef = t
                            .coef
                            .checked_mul(n)
                            .ok_or_else(|| too_large("coefficient overflow".into()))?
                    }
                    None => t.dpow += 1,
                }
            } else {
                let mut others: Vec<ExprId> = Vec::with_capacity(2);
                let mut k = 0;
                while k < t.dots.len() {
                    let (a, b) = t.dots[k];
                    if a == mu || b == mu {
                        others.push(if a == mu { b } else { a });
                        t.dots.swap_remove(k);
                    } else {
                        k += 1;
                    }
                }
                t.dots.push(dot_pair(others[0], others[1]));
            }
            changed = true;
        }
        if !changed {
            return Ok(());
        }
    }
}

type MonoKey = (u32, Vec<ExprId>, Vec<(ExprId, ExprId)>);

fn build(terms: Vec<CTerm>, dim: ExprId, pool: &ExprPool) -> ExprId {
    // Collect equal monomials.
    let mut idx: HashMap<MonoKey, usize> = HashMap::new();
    let mut acc: Vec<(MonoKey, i64)> = Vec::new();
    for mut t in terms {
        t.other.sort_unstable();
        t.dots.sort_unstable();
        let key = (t.dpow, t.other, t.dots);
        match idx.get(&key) {
            Some(&i) => acc[i].1 += t.coef,
            None => {
                idx.insert(key.clone(), acc.len());
                acc.push((key, t.coef));
            }
        }
    }
    let zero = pool.integer(0_i32);
    let parts: Vec<ExprId> = acc
        .into_iter()
        .filter(|(_, c)| *c != 0)
        .map(|((dpow, other, dots), c)| {
            let mut f = vec![pool.integer(c)];
            if dpow > 0 {
                f.push(pool.pow(dim, pool.integer(dpow as i64)));
            }
            f.extend(other);
            f.extend(dots.into_iter().map(|p| mk_dot(p, pool)));
            if f.len() == 1 {
                f[0]
            } else {
                pool.mul(f)
            }
        })
        .collect();
    let sum = match parts.len() {
        0 => zero,
        1 => parts[0],
        _ => pool.add(parts),
    };
    crate::simplify::engine::simplify_expanded(sum, pool).value
}

fn refuse_hidden_indices(
    terms: &[CTerm],
    indices: &[ExprId],
    pool: &ExprPool,
) -> Result<(), NcError> {
    let probe = pool.symbol("__lorentz_probe__", Domain::Real);
    for t in terms {
        for &o in &t.other {
            for &mu in indices {
                let m: HashMap<ExprId, ExprId> = [(mu, probe)].into_iter().collect();
                if subs(o, &m, pool) != o {
                    return Err(bad(format!(
                        "index `{}` appears inside `{}`, which is not a dot(·,·) factor; \
                         only metric contractions are supported (no levi_civita in d dimensions)",
                        pool.display(mu),
                        pool.display(o)
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Contract the dummy indices `indices` of a polynomial in `dot(a, b)`
/// factors, in dimension `dim`.
///
/// `dot(μ, μ) → d`, `dot(μ, a)·dot(μ, b) → dot(a, b)`, repeated until no
/// index with two occurrences is left. An index that occurs once is *free*
/// and kept. This is what turns a product of traces such as
/// `tr(γ^μ γ^ν γ^ρ γ^σ) · tr(γ_μ γ_ν γ_ρ γ_σ)` into a polynomial in `d`.
///
/// # Errors
/// `E-NC-005` for an index that occurs more than twice in a term or inside a
/// factor that is not a `dot`; `E-NC-002` past the size caps.
pub fn lorentz_contract(
    expr: ExprId,
    indices: &[ExprId],
    dim: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, NcError> {
    check_labels(&[], indices, pool, "lorentz_contract")?;
    let dv = dim_value(dim, pool)?;
    let mut terms = expand(expr, pool)?;
    refuse_hidden_indices(&terms, indices, pool)?;
    for (n, t) in terms.iter_mut().enumerate() {
        if n % 256 == 0 {
            budget::check()?;
        }
        contract_term(t, indices, dv)?;
    }
    Ok(build(terms, dim, pool))
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

fn trace_rec(
    slots: &[ExprId],
    coef: i64,
    dots: &mut Vec<(ExprId, ExprId)>,
    out: &mut Vec<CTerm>,
    counter: &mut usize,
) -> Result<(), NcError> {
    *counter += 1;
    if *counter % 256 == 0 {
        budget::check()?;
    }
    if slots.is_empty() {
        out.push(CTerm {
            coef: 4 * coef,
            dpow: 0,
            other: Vec::new(),
            dots: dots.clone(),
        });
        return Ok(());
    }
    let first = slots[0];
    for k in 1..slots.len() {
        let sign = if k % 2 == 1 { 1 } else { -1 };
        let mut rest = Vec::with_capacity(slots.len() - 2);
        rest.extend_from_slice(&slots[1..k]);
        rest.extend_from_slice(&slots[k + 1..]);
        dots.push(dot_pair(first, slots[k]));
        trace_rec(&rest, coef * sign, dots, out, counter)?;
        dots.pop();
    }
    Ok(())
}

/// Trace of a product of gamma matrices in `dim` dimensions (`tr 1 = 4`).
///
/// `slots` are commutative labels: a label listed in `indices` is a Lorentz
/// index `γ^μ`; any other label is a slashed vector `p̸`. Indices that occur
/// twice in `slots` are summed (`γ^μ … γ_μ`) with `g^μ_μ = dim`; an index
/// that occurs once stays free. The result is a polynomial in `dot(a, b)`
/// (`dot(p, q) = p·q`, `dot(μ, ν) = g^{μν}`, `dot(μ, p) = p^μ`).
///
/// Examples: `tr(γ^μ γ_μ) = 4d`, `tr(γ^μ γ^ν γ_μ γ_ν) = 4d(2 − d)`.
///
/// # Errors
/// `E-NC-005` for a non-commutative label or an index repeated more than
/// twice; `E-NC-006` past 14 slots.
pub fn gamma_trace_dim(
    slots: &[ExprId],
    indices: &[ExprId],
    dim: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, NcError> {
    check_labels(slots, indices, pool, "gamma_trace")?;
    let dv = dim_value(dim, pool)?;
    if slots.len() > MAX_SLOTS {
        return Err(NcError::GammaTraceTooLong {
            slots: slots.len(),
            max: MAX_SLOTS,
        });
    }
    if slots.len() % 2 == 1 {
        return Ok(pool.integer(0_i32));
    }
    let mut terms = Vec::new();
    trace_rec(slots, 1, &mut Vec::new(), &mut terms, &mut 0)?;
    for (n, t) in terms.iter_mut().enumerate() {
        if n % 256 == 0 {
            budget::check()?;
        }
        contract_term(t, indices, dv)?;
    }
    Ok(build(terms, dim, pool))
}

// ---------------------------------------------------------------------------
// Gamma strings
// ---------------------------------------------------------------------------

/// `Σ_k c_k d^k` as a coefficient vector.
type DPoly = Vec<i64>;

fn dpoly_add(a: &DPoly, b: &DPoly) -> Result<DPoly, NcError> {
    let n = a.len().max(b.len());
    let mut out = vec![0i64; n];
    for (i, o) in out.iter_mut().enumerate() {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        *o = x
            .checked_add(y)
            .ok_or_else(|| too_large("coefficient overflow".into()))?;
    }
    Ok(out)
}

fn dpoly_scale(a: &DPoly, c: i64) -> Result<DPoly, NcError> {
    a.iter()
        .map(|&x| {
            x.checked_mul(c)
                .ok_or_else(|| too_large("coefficient overflow".into()))
        })
        .collect()
}

fn dpoly_times_d(a: &DPoly) -> DPoly {
    let mut out = vec![0];
    out.extend_from_slice(a);
    out
}

type GString = Vec<ExprId>;
type GSum = Vec<(DPoly, GString)>;

fn push_term(sum: &mut GSum, c: DPoly, w: GString) -> Result<(), NcError> {
    if let Some(e) = sum.iter_mut().find(|(_, x)| *x == w) {
        e.0 = dpoly_add(&e.0, &c)?;
    } else {
        sum.push((c, w));
    }
    Ok(())
}

/// Position pair of a contracted index in `w`, if any.
fn find_pair(w: &[ExprId], indices: &[ExprId]) -> Result<Option<(usize, usize)>, NcError> {
    for &mu in indices {
        let pos: Vec<usize> = (0..w.len()).filter(|&i| w[i] == mu).collect();
        match pos.len() {
            0 | 1 => {}
            2 => return Ok(Some((pos[0], pos[1]))),
            n => {
                return Err(bad(format!(
                    "Lorentz index occurs {n} times in the gamma string (a dummy index appears exactly twice)"
                )))
            }
        }
    }
    Ok(None)
}

/// Contract the dummy indices of a product of gamma matrices.
///
/// `slots` are labels as in [`gamma_trace_dim`]; indices in `indices` that
/// occur twice are summed. Returns terms `(coefficient, labels)` meaning
/// `coefficient · Π γ(label)`, where the coefficient is a polynomial in
/// `dim` and no index is repeated any more. Uses only
/// `γ^μ γ^a S γ_μ = 2 S γ^a − γ^a (γ^μ S γ_μ)` and `γ^μ γ_μ = d`:
///
/// * `γ^μ γ_μ = d`
/// * `γ^μ γ^ν γ_μ = (2 − d) γ^ν`
/// * `γ^μ γ^ν γ^ρ γ_μ = 2 γ^ρ γ^ν + (d − 2) γ^ν γ^ρ`
///   (`= 4 g^{νρ} − (4 − d) γ^ν γ^ρ`)
///
/// # Errors
/// `E-NC-005` for bad labels or an index repeated more than twice;
/// `E-NC-006` past 14 slots; `E-NC-002` when the expansion exceeds the
/// term cap.
pub fn gamma_contract(
    slots: &[ExprId],
    indices: &[ExprId],
    dim: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, Vec<ExprId>)>, NcError> {
    check_labels(slots, indices, pool, "gamma_contract")?;
    let dv = dim_value(dim, pool)?;
    if slots.len() > MAX_SLOTS {
        return Err(NcError::GammaTraceTooLong {
            slots: slots.len(),
            max: MAX_SLOTS,
        });
    }
    let mut done: GSum = Vec::new();
    let mut work: GSum = vec![(vec![1], slots.to_vec())];
    let mut steps = 0usize;
    while let Some((c, w)) = work.pop() {
        steps += 1;
        if steps > 2_000_000 {
            return Err(too_large("gamma-string contraction".into()));
        }
        if steps % 256 == 0 {
            budget::check()?;
        }
        let Some((i, j)) = find_pair(&w, indices)? else {
            push_term(&mut done, c, w)?;
            continue;
        };
        let (pre, mid, post) = (&w[..i], &w[i + 1..j], &w[j + 1..]);
        let join = |m: &[ExprId]| -> GString {
            let mut v = pre.to_vec();
            v.extend_from_slice(m);
            v.extend_from_slice(post);
            v
        };
        if mid.is_empty() {
            work.push((dpoly_times_d(&c), join(&[])));
        } else {
            // γ^μ a S' γ_μ = 2 S' a − a (γ^μ S' γ_μ)
            let a = mid[0];
            let s = &mid[1..];
            let mut m1 = s.to_vec();
            m1.push(a);
            work.push((dpoly_scale(&c, 2)?, join(&m1)));
            let mut m2 = vec![a, w[i]];
            m2.extend_from_slice(s);
            m2.push(w[j]);
            work.push((dpoly_scale(&c, -1)?, join(&m2)));
        }
    }
    let mut out = Vec::new();
    for (c, w) in done {
        let coeff = dpoly_to_expr(&c, dim, dv, pool);
        let coeff = crate::simplify::engine::simplify_expanded(coeff, pool).value;
        if coeff != pool.integer(0_i32) {
            out.push((coeff, w));
        }
    }
    Ok(out)
}

fn dpoly_to_expr(c: &DPoly, dim: ExprId, dv: Option<i64>, pool: &ExprPool) -> ExprId {
    let mut parts = Vec::new();
    for (k, &ck) in c.iter().enumerate() {
        if ck == 0 {
            continue;
        }
        let term = match dv {
            Some(n) => {
                let mut v = rug::Integer::from(ck);
                v *= rug::ops::Pow::pow(rug::Integer::from(n), k as u32);
                pool.integer(v)
            }
            None => {
                if k == 0 {
                    pool.integer(ck)
                } else {
                    pool.mul(vec![
                        pool.integer(ck),
                        pool.pow(dim, pool.integer(k as i64)),
                    ])
                }
            }
        };
        parts.push(term);
    }
    match parts.len() {
        0 => pool.integer(0_i32),
        1 => parts[0],
        _ => pool.add(parts),
    }
}
