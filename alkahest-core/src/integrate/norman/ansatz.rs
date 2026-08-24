//! The Risch–Norman ansatz and the single linear system it reduces to.
//!
//! Given `f = N/Q` in the differential ring built by [`super::ring`], we posit
//!
//! ```text
//!     F  =  P/Q  +  Σⱼ dⱼ·log(pⱼ)
//! ```
//!
//! where `P` ranges over a bounded box of monomials in `ℚ[x, θ₁, …, θₙ]` and
//! the `pⱼ` are the irreducible factors of `Q` (plus `x` and the tower's own
//! logarithm generators, so that `log(log x)`-shaped answers are reachable).
//! Differentiating each ansatz atom gives a *known* rational function; the
//! unknowns enter only linearly.  Clearing a common denominator turns
//! `D(F) − f = 0` into one polynomial identity, and equating the coefficient
//! of every monomial gives a linear system over `ℚ`.
//!
//! Two things are worth being explicit about.
//!
//! * The construction is sound *by design* — we differentiate the candidate to
//!   build the system — but it is only *complete* to the extent that the
//!   monomial box and the logarithm candidate set happen to contain the
//!   answer.  A failure to solve is a failure of the ansatz, never a proof.
//! * Equating coefficients presumes the generators are algebraically
//!   independent.  [`super::ring`] establishes that before we get here.

use std::collections::BTreeSet;

use rug::Rational;

use crate::kernel::{ExprId, ExprPool};
use crate::poly::multipoly::MultiPoly;
use crate::poly::rational::{mpoly_exact_div, RationalFunction};

use super::ring::{GenKind, NormanRing};
use super::DeclineReason;

/// Largest number of unknowns (monomials plus logarithm candidates).
const MAX_UNKNOWNS: usize = 240;
/// Largest number of equations (distinct monomials in the cleared identity).
const MAX_EQUATIONS: usize = 6000;
/// Largest number of terms tolerated in the cleared common denominator.
const MAX_DENOM_TERMS: usize = 2000;

/// One term of the ansatz, together with its derivative.
struct Atom {
    /// The term itself, as an expression builder input.
    kind: AtomKind,
    /// `D(term)`, cleared against the common denominator later.
    deriv: RationalFunction,
}

enum AtomKind {
    /// `monomial / Q`.
    Rational(Vec<u32>),
    /// `log(p)`.
    Log(MultiPoly),
}

/// Build and solve the ansatz.  Returns the candidate antiderivative
/// expression, *unverified*.
pub(super) fn solve(
    ring: &NormanRing,
    f: &RationalFunction,
    pool: &ExprPool,
) -> Result<ExprId, DeclineReason> {
    let q = f.denom.clone();
    let n = f.numer.clone();

    // ---- monomial box for the numerator of the rational part.
    let nv = ring.nvars();
    let mut bounds = Vec::with_capacity(nv);
    let mut count: usize = 1;
    for i in 0..nv {
        let b = (n.degree_in(i) + q.degree_in(i) + 1) as usize;
        count = count.saturating_mul(b + 1);
        if count > MAX_UNKNOWNS {
            return Err(DeclineReason::TooLarge("ansatz monomial box"));
        }
        bounds.push(b);
    }
    let monomials = monomial_box(&bounds);

    // ---- logarithm candidates.
    let logs = log_candidates(ring, &q)?;

    if monomials.len() + logs.len() > MAX_UNKNOWNS {
        return Err(DeclineReason::TooLarge("ansatz size"));
    }
    if monomials.is_empty() && logs.is_empty() {
        return Err(DeclineReason::NoSolution);
    }

    // ---- derivatives of every ansatz atom.
    let q_rf = ring.rf(q.clone())?;
    let mut atoms: Vec<Atom> = Vec::with_capacity(monomials.len() + logs.len());
    for m in &monomials {
        let mono = ring.monomial(m);
        let term = (ring.rf(mono)? / q_rf.clone()).map_err(|_| DeclineReason::RingArithmetic)?;
        let deriv = ring.deriv_rf(&term)?;
        atoms.push(Atom {
            kind: AtomKind::Rational(m.clone()),
            deriv,
        });
    }
    for p in &logs {
        // D(log p) = D(p)/p
        let dp = ring.deriv_poly(p)?;
        let p_rf = ring.rf(p.clone())?;
        let deriv = (dp / p_rf).map_err(|_| DeclineReason::RingArithmetic)?;
        atoms.push(Atom {
            kind: AtomKind::Log(p.clone()),
            deriv,
        });
    }

    // ---- common denominator.
    let mut common = ring.constant_poly(1);
    for a in &atoms {
        common = lcm(&common, &a.deriv.denom).ok_or(DeclineReason::RingArithmetic)?;
        if common.terms.len() > MAX_DENOM_TERMS {
            return Err(DeclineReason::TooLarge("common denominator"));
        }
    }
    common = lcm(&common, &f.denom).ok_or(DeclineReason::RingArithmetic)?;
    if common.terms.len() > MAX_DENOM_TERMS {
        return Err(DeclineReason::TooLarge("common denominator"));
    }

    // ---- clear denominators.
    let mut columns: Vec<MultiPoly> = Vec::with_capacity(atoms.len());
    for a in &atoms {
        let scale =
            mpoly_exact_div(&common, &a.deriv.denom).ok_or(DeclineReason::RingArithmetic)?;
        columns.push(a.deriv.numer.clone() * scale);
    }
    let rhs_scale = mpoly_exact_div(&common, &f.denom).ok_or(DeclineReason::RingArithmetic)?;
    let rhs_poly = f.numer.clone() * rhs_scale;

    // ---- one equation per monomial of the cleared identity.
    let mut keys: BTreeSet<&Vec<u32>> = BTreeSet::new();
    for c in &columns {
        keys.extend(c.terms.keys());
    }
    keys.extend(rhs_poly.terms.keys());
    if keys.len() > MAX_EQUATIONS {
        return Err(DeclineReason::TooLarge("linear system"));
    }

    let mut mat: Vec<Vec<Rational>> = Vec::with_capacity(keys.len());
    let mut rhs: Vec<Rational> = Vec::with_capacity(keys.len());
    let zero = || Rational::from(0);
    for k in &keys {
        let row = columns
            .iter()
            .map(|c| {
                c.terms
                    .get(*k)
                    .map(|v| Rational::from(v.clone()))
                    .unwrap_or_else(zero)
            })
            .collect();
        mat.push(row);
        rhs.push(
            rhs_poly
                .terms
                .get(*k)
                .map(|v| Rational::from(v.clone()))
                .unwrap_or_else(zero),
        );
    }

    // A degenerate (empty) system with a non-zero right-hand side cannot be
    // solved; an empty system with a zero right-hand side means `f = 0`.
    let sol =
        crate::sum::gosper::rational_gaussian_solve(mat, rhs).ok_or(DeclineReason::NoSolution)?;
    if sol.len() != atoms.len() {
        return Err(DeclineReason::NoSolution);
    }

    Ok(build_expression(ring, &atoms, &sol, &q, pool))
}

/// Every exponent vector in the box `0 ≤ eᵢ ≤ bounds[i]`.
fn monomial_box(bounds: &[usize]) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![Vec::new()];
    for &b in bounds {
        let mut next = Vec::with_capacity(out.len() * (b + 1));
        for base in &out {
            for e in 0..=b {
                let mut v = base.clone();
                v.push(e as u32);
                next.push(v);
            }
        }
        out = next;
    }
    for v in out.iter_mut() {
        while v.last() == Some(&0) {
            v.pop();
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Logarithm arguments the ansatz may use.
///
/// The irreducible factors of the denominator (Rothstein–Trager's candidate
/// set, generalised to the tower), plus `x` and the tower's own logarithm
/// generators so that answers of the shape `log(log x)` are reachable.
fn log_candidates(ring: &NormanRing, q: &MultiPoly) -> Result<Vec<MultiPoly>, DeclineReason> {
    let mut out: Vec<MultiPoly> = Vec::new();
    let push = |p: MultiPoly, out: &mut Vec<MultiPoly>| {
        if p.total_degree() == 0 {
            return;
        }
        let neg = -p.clone();
        if !out.iter().any(|g| *g == p || *g == neg) {
            out.push(p);
        }
    };

    let (_unit, factors) = q
        .factor_irreducible()
        .ok_or(DeclineReason::RingArithmetic)?;
    for (f, _m) in factors {
        // An exponential generator is a unit: `log(exp η) = η` is already in
        // the field and adds nothing but noise to the output.
        if is_exp_monomial(ring, &f) {
            continue;
        }
        push(f, &mut out);
    }

    // `x` itself.
    push(ring.monomial(&[1]), &mut out);
    // Each logarithm generator, so that `log(log x)` is reachable.
    for (i, k) in ring.kinds.iter().enumerate() {
        if *k == GenKind::Log {
            let idx = i + 1;
            let mut e = vec![0u32; idx + 1];
            e[idx] = 1;
            push(ring.monomial(&e), &mut out);
        }
    }
    Ok(out)
}

/// `true` when `p` is a single monomial in exponential generators only.
fn is_exp_monomial(ring: &NormanRing, p: &MultiPoly) -> bool {
    if p.terms.len() != 1 {
        return false;
    }
    let (exp, _) = p.terms.iter().next().expect("exactly one term");
    let mut saw = false;
    for (i, &e) in exp.iter().enumerate() {
        if e == 0 {
            continue;
        }
        if i == 0 || ring.kinds.get(i - 1) != Some(&GenKind::Exp) {
            return false;
        }
        saw = true;
    }
    saw
}

/// `true` for the constant polynomial `1`.
fn is_unit(p: &MultiPoly) -> bool {
    p.terms.len() == 1 && p.terms.get(&Vec::new()).is_some_and(|c| *c == 1)
}

/// `lcm(a, b) = a·(b / gcd(a, b))`.
///
/// Only the literal unit is short-circuited.  A *constant* denominator is not
/// a unit here: `RationalFunction` keeps `3/5` as `3/5`, and clearing against
/// `lcm(5, x) = x` would leave an inexact division and a spurious decline.
fn lcm(a: &MultiPoly, b: &MultiPoly) -> Option<MultiPoly> {
    if is_unit(a) {
        return Some(b.clone());
    }
    if is_unit(b) {
        return Some(a.clone());
    }
    match a.gcd(b) {
        Some(g) => {
            let q = mpoly_exact_div(b, &g)?;
            Some(a.clone() * q)
        }
        None => Some(a.clone() * b.clone()),
    }
}

/// A scalar multiple `c·e`, with `c = 1` elided and integral `c` emitted as an
/// integer rather than a rational node (which keeps `simplify` on the fast
/// path, and matters because the exactness of the `d/dx F = f` gate depends on
/// the answer normalising cleanly).
fn scale(c: &Rational, e: ExprId, pool: &ExprPool) -> ExprId {
    if *c == 1 {
        return e;
    }
    let coeff = if *c.denom() == 1 {
        pool.integer(c.numer().clone())
    } else {
        pool.rational(c.numer().clone(), c.denom().clone())
    };
    pool.mul(vec![coeff, e])
}

/// Assemble `P/Q + Σ dⱼ log(pⱼ)` from the solved coefficients.
///
/// `P` carries rational coefficients, so it is cleared to `ℤ` by the lcm of
/// their denominators and then reduced against `Q` through
/// [`RationalFunction::new`], which cancels the polynomial gcd.  Without that
/// reduction `∫dx/(x+1)²` would come back as `−(x+1)/(x+1)²`.
fn build_expression(
    ring: &NormanRing,
    atoms: &[Atom],
    sol: &[Rational],
    q: &MultiPoly,
    pool: &ExprPool,
) -> ExprId {
    let mut log_terms: Vec<ExprId> = Vec::new();

    // Common denominator of the rational part's coefficients.
    let mut lcm_den = rug::Integer::from(1);
    for (a, c) in atoms.iter().zip(sol.iter()) {
        if matches!(a.kind, AtomKind::Rational(_)) && *c != 0 {
            lcm_den = lcm_den.lcm(c.denom());
        }
    }

    let mut p_int = MultiPoly::zero(ring.vars.clone());
    for (a, c) in atoms.iter().zip(sol.iter()) {
        if *c == 0 {
            continue;
        }
        match &a.kind {
            AtomKind::Rational(exp) => {
                let scaled = c.clone() * Rational::from(lcm_den.clone());
                debug_assert_eq!(*scaled.denom(), 1);
                let mut term = ring.monomial(exp);
                for v in term.terms.values_mut() {
                    *v *= scaled.numer().clone();
                }
                p_int = p_int + term;
            }
            AtomKind::Log(p) => {
                let arg = p.to_expr(pool);
                let l = pool.func("log", vec![arg]);
                log_terms.push(scale(c, l, pool));
            }
        }
    }

    let mut parts: Vec<ExprId> = Vec::new();
    if !p_int.is_zero() {
        let reduced = RationalFunction::new(p_int, q.clone());
        let (n, d) = match reduced {
            Ok(rf) => (rf.numer, rf.denom),
            Err(_) => (MultiPoly::zero(ring.vars.clone()), ring.constant_poly(1)),
        };
        let n_expr = n.to_expr(pool);
        let body = if is_unit(&d) {
            n_expr
        } else {
            let inv = pool.pow(d.to_expr(pool), pool.integer(-1_i32));
            pool.mul(vec![n_expr, inv])
        };
        let s = Rational::from((rug::Integer::from(1), lcm_den));
        parts.push(scale(&s, body, pool));
    }
    parts.extend(log_terms);

    let raw = match parts.len() {
        0 => pool.integer(0_i32),
        1 => parts[0],
        _ => pool.add(parts),
    };
    crate::simplify::engine::simplify(raw, pool).value
}
