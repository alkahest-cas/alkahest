//! Canonicalisation and deduplication of FEX skeletons (R5).
//!
//! Two opcode vectors are *skeleton-equivalent* when they describe the same
//! family of functions as `θ` ranges over all real values. Scoring both is
//! wasted work, so before a candidate is fitted its skeleton is rewritten to a
//! normal form with a slot-algebra rewrite table and keyed by that form.
//!
//! Every rule below preserves the function family **exactly** (both
//! inclusions hold for all `θ`), using only two closure properties of a
//! family `T`: *scale-closed* (`c·t ∈ T` for all real `c`, including `0`) and
//! *shift-closed* (`t + c ∈ T`). Leaves `Σαᵢφ(xᵢ)+β` and unary nodes
//! `α·g(s)+β` are both.
//!
//! | rule | why |
//! |---|---|
//! | `L[0]`, `L[1]`, `α·0+β`, `α·1+β`, `α·g(c)+β` → `c` | each is a free constant |
//! | `id(T)` → `T` for `T` scale- and shift-closed | `αT+β = T` (id chains, `x + 0·x²` padding) |
//! | `cos(T)` → `sin(T)` for `T` shift-closed | `cos t = sin(t + π/2)` |
//! | `A − B` → `A + B` for `B` scale-closed | `−B = B` |
//! | `+` and `×` flattened and sorted | commutativity, associativity |
//! | `c + T` → `T` for `T` shift-closed; `c·T` → `T` for `T` scale-closed | the constant is absorbed |
//! | `L[φ] + L[φ]` → `L[φ]` | a leaf is linear in its parameters |
//! | `T / c` → `T` for `T` scale-closed | |
//!
//! The rewrite table is not complete — two equivalent skeletons may still
//! get different keys — but it never merges two inequivalent ones. For
//! fitted candidates [`same_function`] decides equality of the actual
//! functions with Alkahest's zero test after snapping the parameters to
//! nearby rationals; only a *proof* of zero counts as a duplicate.
//!
//! [`redundant_params`] lists parameters that can be pinned to a neutral
//! value without shrinking the family (an `α` multiplying a constant, the
//! `α, β` of an `id` slot over a leaf, the second of two free constants in
//! a sum, …), which improves the conditioning of a fit.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::kernel::{ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::error::FexError;
use super::ops::{BinaryOp, UnaryOp};
use super::template::{SlotKind, Template};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Term {
    /// A free constant.
    C,
    /// A leaf with a non-constant `φ`.
    Leaf(UnaryOp),
    /// `α·op(T) + β`.
    Un(UnaryOp, Box<Term>),
    /// A sum, flattened and sorted.
    Sum(Vec<Term>),
    /// `A − B` (kept only when it is not a sum).
    Diff(Box<Term>, Box<Term>),
    /// A product, flattened and sorted.
    Prod(Vec<Term>),
    /// `A / B`.
    Quot(Box<Term>, Box<Term>),
}

impl Term {
    fn scale_closed(&self) -> bool {
        match self {
            Term::C | Term::Leaf(_) | Term::Un(..) => true,
            Term::Sum(v) => v.iter().all(Term::scale_closed),
            Term::Diff(a, b) => a.scale_closed() && b.scale_closed(),
            Term::Prod(v) => v.iter().any(Term::scale_closed),
            Term::Quot(a, _) => a.scale_closed(),
        }
    }

    fn shift_closed(&self) -> bool {
        match self {
            Term::C | Term::Leaf(_) | Term::Un(..) => true,
            Term::Sum(v) => v.iter().any(Term::shift_closed),
            Term::Diff(a, b) => a.shift_closed() || b.shift_closed(),
            Term::Prod(_) | Term::Quot(..) => false,
        }
    }

    fn write(&self, out: &mut String) {
        match self {
            Term::C => out.push('c'),
            Term::Leaf(phi) => {
                let _ = write!(out, "L[{}]", phi.name());
            }
            Term::Un(op, t) => {
                out.push_str(op.name());
                out.push('(');
                t.write(out);
                out.push(')');
            }
            Term::Sum(v) | Term::Prod(v) => {
                let sep = if matches!(self, Term::Sum(_)) {
                    " + "
                } else {
                    " * "
                };
                out.push('(');
                for (i, t) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(sep);
                    }
                    t.write(out);
                }
                out.push(')');
            }
            Term::Diff(a, b) | Term::Quot(a, b) => {
                let sep = if matches!(self, Term::Diff(..)) {
                    " - "
                } else {
                    " / "
                };
                out.push('(');
                a.write(out);
                out.push_str(sep);
                b.write(out);
                out.push(')');
            }
        }
    }
}

fn leaf(phi: UnaryOp) -> Term {
    if phi.is_constant() {
        Term::C
    } else {
        Term::Leaf(phi)
    }
}

fn unary(op: UnaryOp, t: Term) -> Term {
    if op.is_constant() || t == Term::C {
        return Term::C;
    }
    if op == UnaryOp::Id && t.scale_closed() && t.shift_closed() {
        return t;
    }
    if op == UnaryOp::Cos && t.shift_closed() {
        return Term::Un(UnaryOp::Sin, Box::new(t));
    }
    Term::Un(op, Box::new(t))
}

fn sum(terms: Vec<Term>) -> Term {
    let mut flat: Vec<Term> = Vec::new();
    for t in terms {
        match t {
            Term::Sum(v) => flat.extend(v),
            other => flat.push(other),
        }
    }
    let any_shift = flat.iter().any(|t| *t != Term::C && t.shift_closed());
    let mut out: Vec<Term> = Vec::new();
    let mut have_c = false;
    for t in flat {
        if t == Term::C {
            if any_shift || have_c {
                continue;
            }
            have_c = true;
            out.push(t);
            continue;
        }
        if matches!(t, Term::Leaf(_)) && out.contains(&t) {
            continue;
        }
        out.push(t);
    }
    out.sort();
    if out.len() == 1 {
        out.pop().expect("one element")
    } else {
        Term::Sum(out)
    }
}

fn diff(a: Term, b: Term) -> Term {
    if b.scale_closed() {
        sum(vec![a, b])
    } else {
        Term::Diff(Box::new(a), Box::new(b))
    }
}

fn prod(terms: Vec<Term>) -> Term {
    let mut flat: Vec<Term> = Vec::new();
    for t in terms {
        match t {
            Term::Prod(v) => flat.extend(v),
            other => flat.push(other),
        }
    }
    let any_scale = flat.iter().any(|t| *t != Term::C && t.scale_closed());
    let mut out: Vec<Term> = Vec::new();
    let mut have_c = false;
    for t in flat {
        if t == Term::C {
            if any_scale || have_c {
                continue;
            }
            have_c = true;
        }
        out.push(t);
    }
    out.sort();
    if out.len() == 1 {
        out.pop().expect("one element")
    } else {
        Term::Prod(out)
    }
}

fn quot(a: Term, b: Term) -> Term {
    if b == Term::C && a.scale_closed() {
        return a;
    }
    if a == Term::C && b == Term::C {
        return Term::C;
    }
    Term::Quot(Box::new(a), Box::new(b))
}

fn term_of(t: &Template, e: &[usize]) -> Vec<Term> {
    let mut vals: Vec<Term> = Vec::with_capacity(t.plan.len());
    for n in &t.plan {
        let v = match n.kind {
            SlotKind::Leaf => leaf(t.unary_at(e, n.slot)),
            SlotKind::Unary => unary(t.unary_at(e, n.slot), vals[n.a].clone()),
            SlotKind::Binary => {
                let (a, b) = (vals[n.a].clone(), vals[n.b].clone());
                match t.binary_at(e, n.slot) {
                    BinaryOp::Add => sum(vec![a, b]),
                    BinaryOp::Sub => diff(a, b),
                    BinaryOp::Mul => prod(vec![a, b]),
                    BinaryOp::Div => quot(a, b),
                }
            }
        };
        vals.push(v);
    }
    vals
}

/// The canonical key of skeleton `e`: equal keys mean equal function
/// families (see the module documentation).
pub fn canonical_key(t: &Template, e: &[usize]) -> Result<String, FexError> {
    t.check_opcodes(e)?;
    let vals = term_of(t, e);
    let mut s = String::new();
    vals.last().expect("non-empty template").write(&mut s);
    Ok(s)
}

/// Parameters that can be pinned without shrinking the family: entry `k` is
/// `Some(v)` when `θ_k` may be fixed at `v`.
pub fn redundant_params(t: &Template, e: &[usize]) -> Result<Vec<Option<f64>>, FexError> {
    t.check_opcodes(e)?;
    let dim = t.dim();
    let mut fixed: Vec<Option<f64>> = vec![None; t.n_params()];
    let vals = term_of(t, e);
    // Is plan node k (as built) a bare leaf or unary node, i.e. an affine
    // node with its own free β?
    let affine = |k: usize| matches!(t.plan[k].kind, SlotKind::Leaf | SlotKind::Unary);
    for n in t.plan.iter() {
        match n.kind {
            SlotKind::Leaf => {
                let phi = t.unary_at(e, n.slot);
                if phi.is_constant() {
                    // Σαᵢ·c + β: only β matters.
                    for i in 0..dim {
                        fixed[n.poff + i] = Some(0.0);
                    }
                }
            }
            SlotKind::Unary => {
                let op = t.unary_at(e, n.slot);
                if op.is_constant() || vals[n.a] == Term::C {
                    // α·0 + β, α·1 + β and α·g(c) + β are constants: β alone
                    // carries them.
                    fixed[n.poff] = Some(0.0);
                } else if op == UnaryOp::Id && vals[n.a].scale_closed() && vals[n.a].shift_closed()
                {
                    // α·T + β over a scale- and shift-closed family T: the
                    // child's own parameters absorb these.
                    fixed[n.poff] = Some(1.0);
                    fixed[n.poff + 1] = Some(0.0);
                }
            }
            SlotKind::Binary => {
                let op = t.binary_at(e, n.slot);
                if matches!(op, BinaryOp::Add | BinaryOp::Sub) && affine(n.a) && affine(n.b) {
                    // Two free constants in a sum: pin the right one's β —
                    // unless the right child is a constant leaf/node, whose
                    // β is its only parameter (then pin the left one's).
                    let rb = &t.plan[n.b];
                    let lb = &t.plan[n.a];
                    let beta = |m: &super::template::PlanNode| match m.kind {
                        SlotKind::Leaf => m.poff + dim,
                        _ => m.poff + 1,
                    };
                    let right_const = vals[n.b] == Term::C;
                    let left_const = vals[n.a] == Term::C;
                    if !right_const {
                        fixed[beta(rb)] = Some(0.0);
                    } else if !left_const {
                        fixed[beta(lb)] = Some(0.0);
                    }
                }
                if op == BinaryOp::Mul
                    && t.plan[n.b].kind == SlotKind::Unary
                    && vals[n.a].scale_closed()
                    && vals[n.b] != Term::C
                {
                    // Two free scales in a product: pin the right unary's α…
                    // only when its β is already pinned (otherwise the
                    // product (αg+β)·A is not a pure scaling of g·A).
                    let rb = &t.plan[n.b];
                    if fixed[rb.poff + 1] == Some(0.0) {
                        fixed[rb.poff] = Some(1.0);
                    }
                }
            }
        }
    }
    Ok(fixed)
}

/// A cache from canonical skeleton keys to the best fit seen for that
/// family.
#[derive(Clone, Debug, Default)]
pub struct DedupCache {
    map: HashMap<String, DedupEntry>,
    hits: usize,
}

/// One entry of a [`DedupCache`].
#[derive(Clone, Debug, PartialEq)]
pub struct DedupEntry {
    /// The first opcode vector seen with this key.
    pub e: Vec<usize>,
    /// Best parameters seen.
    pub theta: Vec<f64>,
    /// Best loss seen.
    pub loss: f64,
}

impl DedupCache {
    /// An empty cache.
    pub fn new() -> DedupCache {
        DedupCache::default()
    }

    /// The entry for `key`, if any (a hit is counted).
    pub fn lookup(&mut self, key: &str) -> Option<&DedupEntry> {
        let r = self.map.get(key);
        if r.is_some() {
            self.hits += 1;
        }
        r
    }

    /// Record a fit; returns `true` if it is new or better than the stored
    /// one.
    pub fn insert(&mut self, key: String, e: &[usize], theta: &[f64], loss: f64) -> bool {
        match self.map.get_mut(&key) {
            Some(ent) => {
                if loss < ent.loss {
                    ent.loss = loss;
                    ent.theta = theta.to_vec();
                    true
                } else {
                    false
                }
            }
            None => {
                self.map.insert(
                    key,
                    DedupEntry {
                        e: e.to_vec(),
                        theta: theta.to_vec(),
                        loss,
                    },
                );
                true
            }
        }
    }

    /// Number of distinct keys.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Lookups that found an entry.
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// Entries sorted by loss, best first.
    pub fn best(&self, k: usize) -> Vec<(&str, &DedupEntry)> {
        let mut v: Vec<(&str, &DedupEntry)> =
            self.map.iter().map(|(k, e)| (k.as_str(), e)).collect();
        v.sort_by(|a, b| a.1.loss.total_cmp(&b.1.loss));
        v.truncate(k);
        v
    }
}

/// One representative opcode vector per canonical key, over every skeleton
/// of the template (in lexicographic opcode order, first one wins).
/// Refuses templates with more than `limit` skeletons.
pub fn canonical_skeletons(
    t: &Template,
    limit: u128,
) -> Result<Vec<(String, Vec<usize>)>, FexError> {
    let total = t.n_skeletons();
    if total > limit {
        return Err(FexError::Opcode {
            detail: format!("{total} skeletons is more than the limit {limit}"),
        });
    }
    let ar = t.slot_arities();
    let mut e = vec![0usize; ar.len()];
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    loop {
        let key = canonical_key(t, &e)?;
        if !seen.contains_key(&key) {
            seen.insert(key.clone(), out.len());
            out.push((key, e.clone()));
        }
        // Odometer increment, last slot fastest.
        let mut i = ar.len();
        loop {
            if i == 0 {
                return Ok(out);
            }
            i -= 1;
            e[i] += 1;
            if e[i] < ar[i] {
                break;
            }
            e[i] = 0;
        }
    }
}

/// The best rational `p/q` with `q ≤ max_den` within `tol` of `x`, if any.
fn snap(x: f64, max_den: i64, tol: f64) -> Option<(i64, i64)> {
    if !x.is_finite() {
        return None;
    }
    // Continued-fraction convergents.
    let (mut h0, mut h1) = (0i64, 1i64);
    let (mut k0, mut k1) = (1i64, 0i64);
    let mut y = x;
    for _ in 0..40 {
        let a = y.floor();
        if a.abs() > 1e12 {
            break;
        }
        let ai = a as i64;
        let h2 = ai.checked_mul(h1)?.checked_add(h0)?;
        let k2 = ai.checked_mul(k1)?.checked_add(k0)?;
        if k2 > max_den {
            break;
        }
        (h0, h1, k0, k1) = (h1, h2, k1, k2);
        if (x - h1 as f64 / k1 as f64).abs() <= tol {
            return Some((h1, k1));
        }
        let frac = y - a;
        if frac.abs() < 1e-300 {
            break;
        }
        y = 1.0 / frac;
    }
    None
}

/// Whether two fitted candidates are the same function, decided exactly.
///
/// Each parameter is snapped to the nearest rational with denominator at
/// most `max_den` within `tol` (a parameter that does not snap is kept as a
/// float, which can never yield a proof). The two candidates are then
/// instantiated over exact rationals and their difference handed to
/// Alkahest's zero test. Returns `Some(true)` only for a **proof** that they
/// agree identically, `Some(false)` for a proof that they differ, `None`
/// otherwise.
#[allow(clippy::too_many_arguments)]
pub fn same_function(
    t: &Template,
    e1: &[usize],
    theta1: &[f64],
    e2: &[usize],
    theta2: &[f64],
    coords: &[ExprId],
    pool: &ExprPool,
    max_den: i64,
    tol: f64,
) -> Result<Option<bool>, FexError> {
    t.check_params(theta1)?;
    t.check_params(theta2)?;
    let exact = |th: &[f64]| -> Vec<ExprId> {
        th.iter()
            .map(|&v| match snap(v, max_den, tol) {
                Some((p, q)) => pool.rational(p, q),
                None => pool.float(v, 53),
            })
            .collect()
    };
    let a = t.instantiate_with(e1, &exact(theta1), coords, pool)?;
    let b = t.instantiate_with(e2, &exact(theta2), coords, pool)?;
    let d = pool.add(vec![a, pool.mul(vec![pool.integer(-1), b])]);
    let z = zero_test(d, pool, None);
    Ok(match z.status {
        ZeroStatus::ProvedZero => Some(true),
        ZeroStatus::ProvedNonzero => Some(false),
        _ => None,
    })
}
