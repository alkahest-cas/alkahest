//! Newton–Puiseux expansion of a plane algebraic curve `F(x, y) = 0` — the
//! local fractional-power series of the `y`-branches at `x = 0`.
//!
//! A **Puiseux series** is a Laurent series in `x^{1/e}` for some ramification
//! index `e ≥ 1`: `y(x) = Σ_k c_k x^{k/e}`.  Each branch of the curve through a
//! point over `x = 0` has such an expansion; ramified places are exactly those
//! with `e > 1`.  These local expansions are the substrate for residue
//! computation at ramified/infinite places in the Trager algebraic-integration
//! algorithm (Risch milestone M3 / van Hoeij integral bases).
//!
//! ## Algorithm (classical Newton–Puiseux)
//!
//! For `F = Σ a_{ij} x^i y^j` we seek `y → c₀` as `x → 0`:
//! 1. the constant `c₀` is a root of `F(0, y)`; substitute `y = c₀ + w` so `w → 0`;
//! 2. on the **Newton polygon** of the shifted polynomial (lower hull of the
//!    points `(j, i)`), each edge of slope `−q` (`q > 0`) gives a leading term
//!    `w ≈ c·x^q`, with `c` a nonzero root of the edge's **characteristic
//!    polynomial** `φ(c) = Σ_{(i,j)∈edge} a_{ij} c^j`;
//! 3. substitute `w = x^q(c + w₁)` and recurse on the resulting polynomial,
//!    accumulating `q` into the exponents, until the target precision is reached.
//!
//! Keeping `x` itself (with *rational* exponents) throughout — rather than the
//! usual `x = τ^e` rescaling — lets one polynomial type carry every level.
//!
//! ## Scope
//!
//! [`puiseux_at_zero`] returns the branches with **rational** coefficients —
//! sound and complete for that class.  [`puiseux_at`] expands at an arbitrary
//! rational base point `x = α` (exponents in `(x − α)`).  [`puiseux_at_zero_algebraic`]
//! returns **all** branches up to a *single* algebraic extension per branch: the
//! characteristic polynomial is factored over `ℚ`, a root `θ` is adjoined, and the
//! branch continued over `ℚ(θ)` (see [`AlgPuiseuxSeries`]).  This is complete for
//! radical / superelliptic curves (`yⁿ = p`, roots of unity) and constant
//! algebraic branches; a branch needing a *further* extension (non-linear
//! characteristic over `ℚ(θ)`) is skipped, never mis-reported — the summed
//! `conjugates` reveal whether every sheet was recovered.  Every rational branch
//! is back-substitution-checked in the test suite.

use rug::{Integer, Rational};
use std::collections::BTreeMap;

use crate::flint::FlintPoly;
use crate::integrate::risch::number_field::{KPoly, NumberField};

/// A coefficient in a number field `ℚ[θ]/(m)`: a polynomial in `θ` (ascending),
/// reduced mod `m`.  For the base field `ℚ` this is a constant `[r]`.
type KElem = Vec<Rational>;
/// Bivariate polynomial with **number-field** coefficients: `(x-exp, y-exp) → KElem`.
type KBi = BTreeMap<(Rational, u32), KElem>;

/// A truncated Puiseux series `Σ c_k x^{e_k}` with rational exponents `e_k`
/// (ascending) and ramification index `e` (the lcm of the exponent denominators).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PuiseuxSeries {
    /// Ramification index `e`: every exponent has denominator dividing `e`.
    pub ramification: u64,
    /// `(exponent, coefficient)` pairs, strictly ascending in exponent.
    pub terms: Vec<(Rational, Rational)>,
    /// Terms with exponent `≥ order` are unknown (truncated).  `None` means the
    /// branch is *exact* (a terminating, polynomial-in-`x^{1/e}` branch).
    pub order: Option<Rational>,
}

/// Bivariate polynomial as `(x-exponent, y-exponent) → coefficient`, with
/// **rational** `x`-exponents (produced by the `w = x^q(c+w₁)` substitutions).
type Bi = BTreeMap<(Rational, u32), Rational>;

/// A Newton-polygon edge: `(q, monomials)` where `q = −slope` and `monomials`
/// are the `((x-exp, y-exp), coeff)` triples lying on it.
type Edge = (Rational, Vec<((Rational, u32), Rational)>);

fn rzero() -> Rational {
    Rational::from(0)
}

/// Puiseux expansions of the rational branches of `F(x, y) = 0` at `x = 0`, each
/// to precision `prec` (terms with `x`-exponent `< prec` are returned).
///
/// `coeffs` lists the monomials `(i, j, a_{ij})` of `F = Σ a_{ij} x^i y^j`.
pub fn puiseux_at_zero(coeffs: &[(u32, u32, Rational)], prec: u32) -> Vec<PuiseuxSeries> {
    let mut f: Bi = BTreeMap::new();
    for (i, j, a) in coeffs {
        if *a != 0 {
            *f.entry((Rational::from(*i), *j)).or_insert_with(rzero) += a;
        }
    }
    f.retain(|_, a| *a != 0);
    if f.is_empty() {
        return Vec::new();
    }
    factor_min_x(&mut f); // F = x^v · F'; branches unaffected

    let prec_r = Rational::from(prec);
    let mut out = Vec::new();

    // Constant terms c₀ are the rational roots of F(0, y).
    let mut f0: BTreeMap<u32, Rational> = BTreeMap::new();
    for ((xe, ye), a) in &f {
        if *xe == 0 {
            *f0.entry(*ye).or_insert_with(rzero) += a;
        }
    }
    let f0_dense = dense(&f0);
    for c0 in rational_roots(&f0_dense) {
        let g = shift_y(&f, &c0);
        for (mut terms, exact) in lift(&g, &prec_r, 0) {
            if c0 != 0 {
                terms.insert(0, (rzero(), c0.clone()));
            }
            terms.retain(|(e, _)| exact || *e < prec_r);
            terms.sort_by(|a, b| a.0.cmp(&b.0));
            let e = terms.iter().fold(1u64, |acc, (ex, _)| {
                lcm_u64(acc, ex.denom().to_u64().unwrap_or(1))
            });
            out.push(PuiseuxSeries {
                ramification: e,
                terms,
                order: if exact { None } else { Some(prec_r.clone()) },
            });
        }
    }
    out
}

/// Lift the `w → 0` branches of `g(x, w) = 0` as series in `x`, to relative
/// precision `prec`.  Returns `(terms, exact)` where `exact` marks a terminating
/// branch (`w ≡ 0` after some point).
fn lift(g: &Bi, prec: &Rational, depth: u32) -> Vec<(Vec<(Rational, Rational)>, bool)> {
    const MAX_DEPTH: u32 = 64;
    let mut g = g.clone();
    g.retain(|_, a| *a != 0);
    if g.is_empty() {
        return vec![(Vec::new(), true)]; // g ≡ 0: w can be 0 (exact)
    }
    if depth > MAX_DEPTH {
        return vec![(Vec::new(), false)];
    }

    let mut result: Vec<(Vec<(Rational, Rational)>, bool)> = Vec::new();

    // Factor w^{m0}: w = 0 is an exact branch.
    let m0 = g.keys().map(|(_, j)| *j).min().unwrap_or(0);
    if m0 > 0 {
        result.push((Vec::new(), true));
        let shifted: Bi = g
            .into_iter()
            .map(|((xe, ye), a)| ((xe, ye - m0), a))
            .collect();
        g = shifted;
    }

    for (q, edge) in newton_edges(&g) {
        if q <= 0 {
            continue; // q ≤ 0 is not a w → 0 branch
        }
        // Characteristic polynomial φ(c) = Σ_{(i,j)∈edge} a_{ij} c^j.
        let mut phi: BTreeMap<u32, Rational> = BTreeMap::new();
        for ((_, j), a) in &edge {
            *phi.entry(*j).or_insert_with(rzero) += a;
        }
        for c in rational_roots(&dense(&phi)) {
            if c == 0 {
                continue;
            }
            if prec.clone() - &q <= 0 {
                result.push((vec![(q.clone(), c.clone())], false));
                continue;
            }
            let g1 = substitute(&g, &q, &c);
            for (sub, exact) in lift(&g1, &(prec.clone() - &q), depth + 1) {
                let mut terms = vec![(q.clone(), c.clone())];
                for (gamma, b) in sub {
                    terms.push((q.clone() + &gamma, b));
                }
                result.push((terms, exact));
            }
        }
    }
    result
}

/// Edges `(q, keys-on-edge)` of the lower convex hull of the points `(j, i)`
/// (`i` the x-exponent, `j` the y-exponent) with **positive** `q = −slope` — the
/// `w → 0` Newton-polygon edges.  Pure geometry on the monomial *keys*, so it is
/// shared by the rational and number-field coefficient layers.
pub(crate) fn newton_edges_keys(keys: &[(Rational, u32)]) -> Vec<(Rational, Vec<(Rational, u32)>)> {
    // For each y-exponent j keep the minimal x-exponent i (lower envelope).
    let mut lo: BTreeMap<u32, Rational> = BTreeMap::new();
    for (xe, ye) in keys {
        lo.entry(*ye)
            .and_modify(|m| {
                if xe < m {
                    *m = xe.clone();
                }
            })
            .or_insert_with(|| xe.clone());
    }
    let pts: Vec<(u32, Rational)> = lo.into_iter().collect(); // ascending j
    if pts.len() < 2 {
        return Vec::new();
    }
    let mut hull: Vec<(u32, Rational)> = Vec::new();
    for p in pts {
        while hull.len() >= 2 {
            let a = &hull[hull.len() - 2];
            let b = &hull[hull.len() - 1];
            let lhs = Rational::from(b.0 as i64 - a.0 as i64) * (p.1.clone() - &b.1);
            let rhs = Rational::from(p.0 as i64 - b.0 as i64) * (b.1.clone() - &a.1);
            if lhs - rhs <= 0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    let mut edges = Vec::new();
    for w in hull.windows(2) {
        let (j1, i1) = (&w[0].0, &w[0].1);
        let (j2, i2) = (&w[1].0, &w[1].1);
        let dj = Rational::from(*j2 as i64 - *j1 as i64);
        let q = (i1.clone() - i2) / dj; // −slope
        if q <= 0 {
            continue;
        }
        let val = i1.clone() + q.clone() * Rational::from(*j1 as i64);
        let on_edge: Vec<(Rational, u32)> = keys
            .iter()
            .filter(|(xe, ye)| xe.clone() + q.clone() * Rational::from(*ye as i64) == val)
            .cloned()
            .collect();
        edges.push((q, on_edge));
    }
    edges
}

/// Edges `(q, monomials)` of the Newton polygon of `g` (rational coefficients).
fn newton_edges(g: &Bi) -> Vec<Edge> {
    let keys: Vec<(Rational, u32)> = g.keys().cloned().collect();
    let mut edges = Vec::new();
    for (q, on_edge) in newton_edges_keys(&keys) {
        let monos: Vec<((Rational, u32), Rational)> = on_edge
            .into_iter()
            .map(|k| {
                let a = g[&k].clone();
                (k, a)
            })
            .collect();
        edges.push((q, monos));
    }
    edges
}

/// Substitute `w = x^q (c + w₁)` into `g`, divide by `x^ν` (ν = min x-exponent),
/// and return the polynomial in `(x, w₁)`.
fn substitute(g: &Bi, q: &Rational, c: &Rational) -> Bi {
    let mut g1: Bi = BTreeMap::new();
    for ((xe, ye), a) in g {
        let j = *ye;
        let new_xe = xe.clone() + q.clone() * Rational::from(j as i64);
        // (c + w₁)^j = Σ_l C(j,l) c^{j−l} w₁^l
        for l in 0..=j {
            let binom = Rational::from(binomial(j, l));
            let cpow = rat_pow(c, j - l);
            let coeff = a.clone() * &binom * &cpow;
            if coeff != 0 {
                *g1.entry((new_xe.clone(), l)).or_insert_with(rzero) += &coeff;
            }
        }
    }
    g1.retain(|_, a| *a != 0);
    factor_min_x(&mut g1);
    g1
}

/// Divide out the largest power `x^ν` (ν = minimal x-exponent).
fn factor_min_x(g: &mut Bi) {
    let Some(v) = g.keys().map(|(xe, _)| xe.clone()).min() else {
        return;
    };
    if v == 0 {
        return;
    }
    *g = std::mem::take(g)
        .into_iter()
        .map(|((xe, ye), a)| ((xe - &v, ye), a))
        .collect();
}

/// `F(x, c₀ + w)` as a polynomial in `(x, w)`.
fn shift_y(f: &Bi, c0: &Rational) -> Bi {
    if *c0 == 0 {
        return f.clone();
    }
    let mut g: Bi = BTreeMap::new();
    for ((xe, ye), a) in f {
        let j = *ye;
        for l in 0..=j {
            let binom = Rational::from(binomial(j, l));
            let cpow = rat_pow(c0, j - l);
            let coeff = a.clone() * &binom * &cpow;
            if coeff != 0 {
                *g.entry((xe.clone(), l)).or_insert_with(rzero) += &coeff;
            }
        }
    }
    g.retain(|_, a| *a != 0);
    g
}

/// Dense coefficient vector (index = degree) from a sparse `degree → coeff` map.
fn dense(m: &BTreeMap<u32, Rational>) -> Vec<Rational> {
    let Some(&maxd) = m.keys().max() else {
        return Vec::new();
    };
    let mut v = vec![rzero(); maxd as usize + 1];
    for (d, c) in m {
        v[*d as usize] = c.clone();
    }
    v
}

/// All distinct rational roots of `Σ p[k] c^k` (including `0`), via the rational
/// root theorem.  Empty for the zero polynomial.
fn rational_roots(p: &[Rational]) -> Vec<Rational> {
    // Trim trailing zeros.
    let mut hi = p.len();
    while hi > 0 && p[hi - 1] == 0 {
        hi -= 1;
    }
    let p = &p[..hi];
    if p.is_empty() {
        return Vec::new();
    }
    let mut roots = Vec::new();
    // Factor out c^t (low-order zeros) ⇒ root 0.
    let mut lo = 0usize;
    while lo < p.len() && p[lo] == 0 {
        lo += 1;
    }
    if lo > 0 {
        roots.push(rzero());
    }
    let p = &p[lo..];
    if p.len() <= 1 {
        return roots; // constant (after factoring) — no further roots
    }
    // Clear denominators → integer coefficients.
    let mut den_lcm = Integer::from(1);
    for c in p {
        den_lcm = den_lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = p
        .iter()
        .map(|c| {
            (c.clone() * Rational::from(den_lcm.clone()))
                .numer()
                .clone()
        })
        .collect();
    let a0 = ints[0].clone().abs();
    let an = ints[ints.len() - 1].clone().abs();
    let pdiv = divisors(&a0);
    let qdiv = divisors(&an);
    let mut seen: Vec<Rational> = Vec::new();
    for pn in &pdiv {
        for qn in &qdiv {
            for sign in [1i32, -1] {
                let cand = Rational::from((Integer::from(sign) * pn.clone(), qn.clone()));
                if seen.contains(&cand) {
                    continue;
                }
                if eval_int_poly(&ints, &cand) == 0 {
                    seen.push(cand);
                }
            }
        }
    }
    roots.extend(seen);
    roots
}

fn eval_int_poly(coeffs: &[Integer], c: &Rational) -> Rational {
    let mut acc = rzero();
    for a in coeffs.iter().rev() {
        acc = acc * c + Rational::from(a.clone());
    }
    acc
}

/// Positive divisors of `|n|` (with `n ≠ 0`); `{1}` for `n = 0`.
fn divisors(n: &Integer) -> Vec<Integer> {
    let n = n.clone().abs();
    if n == 0 {
        return vec![Integer::from(1)];
    }
    let mut ds = Vec::new();
    let mut d = Integer::from(1);
    while Integer::from(&d * &d) <= n {
        if n.is_divisible(&d) {
            ds.push(d.clone());
            let other = n.clone() / &d;
            if other != d {
                ds.push(other);
            }
        }
        d += 1;
    }
    ds
}

fn binomial(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::from(0);
    }
    let mut num = Integer::from(1);
    for t in 0..k {
        num *= Integer::from(n - t);
    }
    let mut den = Integer::from(1);
    for t in 1..=k {
        den *= Integer::from(t);
    }
    num / den
}

fn rat_pow(c: &Rational, e: u32) -> Rational {
    let mut acc = Rational::from(1);
    for _ in 0..e {
        acc *= c;
    }
    acc
}

fn lcm_u64(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd_u64(a, b) * b
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ---------------------------------------------------------------------------
// Expansion at an arbitrary rational base point
// ---------------------------------------------------------------------------

/// Puiseux expansions of `F(x, y) = 0` at `x = α` (`α ∈ ℚ`): the returned
/// exponents are powers of `(x − α)`.  Implemented by the shift `x ↦ x + α` and
/// [`puiseux_at_zero`].  (Rational branches only — see [`puiseux_at_zero`].)
pub fn puiseux_at(
    coeffs: &[(u32, u32, Rational)],
    alpha: &Rational,
    prec: u32,
) -> Vec<PuiseuxSeries> {
    if *alpha == 0 {
        return puiseux_at_zero(coeffs, prec);
    }
    // x^i ↦ (x+α)^i = Σ_m C(i,m) α^{i−m} x^m.
    let mut shifted: BTreeMap<(u32, u32), Rational> = BTreeMap::new();
    for (i, j, a) in coeffs {
        for m in 0..=*i {
            let binom = Rational::from(binomial(*i, m));
            let apow = rat_pow(alpha, *i - m);
            let c = a.clone() * &binom * &apow;
            if c != 0 {
                *shifted.entry((m, *j)).or_insert_with(rzero) += &c;
            }
        }
    }
    let flat: Vec<(u32, u32, Rational)> =
        shifted.into_iter().map(|((i, j), a)| (i, j, a)).collect();
    puiseux_at_zero(&flat, prec)
}

// ---------------------------------------------------------------------------
// Algebraic-coefficient branches
// ---------------------------------------------------------------------------

/// A Puiseux branch whose coefficients live in a number field `ℚ[θ]/(minpoly)`.
/// When `minpoly` is `None` the field is `ℚ` and each coefficient is a constant
/// `[c]`.  A branch over a degree-`d` field represents `conjugates = d` concrete
/// (conjugate) branches of the curve.
#[derive(Clone, Debug)]
pub struct AlgPuiseuxSeries {
    /// Monic minimal polynomial of `θ` (ascending); `None` ⇒ base field `ℚ`.
    pub minpoly: Option<Vec<Rational>>,
    /// Number of conjugate branches this class represents (`= deg(minpoly)`).
    pub conjugates: usize,
    /// Ramification index `e`.
    pub ramification: u64,
    /// `(exponent, coefficient ∈ ℚ[θ])` pairs, ascending.
    pub terms: Vec<(Rational, KElem)>,
    /// Truncation order (`None` ⇒ exact).
    pub order: Option<Rational>,
}

/// Puiseux expansions of `F(x,y)=0` at `x=0` over `ℚ̄`, returning **all** branches
/// up to a single algebraic extension per branch.  Rational branches come from
/// [`puiseux_at_zero`]; the remaining branches are those whose characteristic
/// root is algebraic — handled by factoring the characteristic polynomial over
/// `ℚ`, adjoining a root `θ`, and continuing over `ℚ(θ)`.
///
/// Scope: a *single* extension per branch with **smooth** continuation (deeper
/// characteristic polynomials linear over `ℚ(θ)`) — complete for radical /
/// superelliptic curves (`yⁿ=p`, `∛x`, roots of unity) and constant algebraic
/// branches.  Branches needing a *further* extension are skipped (documented,
/// never mis-reported); the total `conjugates` of all branches indicates whether
/// every sheet was recovered.
pub fn puiseux_at_zero_algebraic(
    coeffs: &[(u32, u32, Rational)],
    prec: u32,
) -> Vec<AlgPuiseuxSeries> {
    // Rational branches.
    let mut out: Vec<AlgPuiseuxSeries> = puiseux_at_zero(coeffs, prec)
        .into_iter()
        .map(|s| AlgPuiseuxSeries {
            minpoly: None,
            conjugates: 1,
            ramification: s.ramification,
            terms: s.terms.into_iter().map(|(e, c)| (e, vec![c])).collect(),
            order: s.order,
        })
        .collect();

    let mut f: Bi = BTreeMap::new();
    for (i, j, a) in coeffs {
        if *a != 0 {
            *f.entry((Rational::from(*i), *j)).or_insert_with(rzero) += a;
        }
    }
    f.retain(|_, a| *a != 0);
    if f.is_empty() {
        return out;
    }
    factor_min_x(&mut f);
    let prec_r = Rational::from(prec);

    // Constant terms: factor F(0, y) over ℚ.
    let mut f0: BTreeMap<u32, Rational> = BTreeMap::new();
    for ((xe, ye), a) in &f {
        if *xe == 0 {
            *f0.entry(*ye).or_insert_with(rzero) += a;
        }
    }
    let f0_dense = dense(&f0);
    // The constant root c₀ = 0 (when y | F(0,y)) is stripped by `factor_over_q`;
    // explore that stem explicitly so origin branches (e.g. yⁿ = x) are found.
    if f0_dense.first().map(|c| *c == 0).unwrap_or(true) {
        out.extend(collect_algebraic(&f, &prec_r, &[]));
    }
    for (fac, deg) in factor_over_q(&f0_dense) {
        if deg == 1 {
            // Nonzero rational constant c₀ = −fac[0]; explore deeper for spawns.
            let c0 = -fac[0].clone();
            out.extend(collect_algebraic(
                &shift_y(&f, &c0),
                &prec_r,
                &[(rzero(), c0.clone())],
            ));
        } else {
            // Algebraic constant c₀ = θ over K = ℚ[t]/(fac).  A deeper non-linear
            // characteristic over ℚ(θ) collapses into a compositum (tower).
            let nf = NumberField::new(fac.clone());
            let theta = nf.reduce(&vec![Rational::from(0), Rational::from(1)]);
            let gk = substitute_k(&nf, &embed(&nf, &f), &rzero(), &theta);
            let (subs, _missed) = lift_tower(&nf, &gk, &prec_r, 0, deg);
            for sub in subs {
                let kfinal = NumberField::new(sub.mp.clone());
                let theta_f = embed_elem(&kfinal, &theta, &sub.theta_in_f);
                let mut full = vec![(rzero(), theta_f)];
                full.extend(sub.terms);
                out.push(make_alg_series(
                    Some(sub.mp.clone()),
                    sub.conjugates,
                    full,
                    sub.exact,
                    &prec_r,
                ));
            }
        }
    }
    out
}

/// Walk the Newton polygon of `g` (`w → 0`, rational coefficients) and emit only
/// the **algebraic** (number-field) branches: at each edge, factor the
/// characteristic polynomial over `ℚ`; degree-1 factors continue rationally
/// (recurse, to reach deeper spawns), degree-≥2 factors spawn an extension and
/// continue over it via `lift_tower`.
fn collect_algebraic(
    g: &Bi,
    prec: &Rational,
    prefix: &[(Rational, Rational)],
) -> Vec<AlgPuiseuxSeries> {
    let mut g = g.clone();
    g.retain(|_, a| *a != 0);
    if g.is_empty() {
        return Vec::new();
    }
    let m0 = g.keys().map(|(_, j)| *j).min().unwrap_or(0);
    if m0 > 0 {
        g = g
            .into_iter()
            .map(|((xe, ye), a)| ((xe, ye - m0), a))
            .collect();
    }
    let mut out = Vec::new();
    let keys: Vec<(Rational, u32)> = g.keys().cloned().collect();
    for (q, on_edge) in newton_edges_keys(&keys) {
        // Characteristic polynomial φ(c) = Σ_{(i,j)∈edge} a_{ij} c^j (over ℚ).
        let mut phi: BTreeMap<u32, Rational> = BTreeMap::new();
        for k in &on_edge {
            *phi.entry(k.1).or_insert_with(rzero) += &g[k];
        }
        for (fac, deg) in factor_over_q(&dense(&phi)) {
            if deg == 1 {
                let c = -fac[0].clone();
                if c == 0 || prec.clone() - &q <= 0 {
                    continue;
                }
                let mut np = prefix.to_vec();
                np.push((q.clone(), c.clone()));
                out.extend(collect_algebraic(
                    &substitute(&g, &q, &c),
                    &(prec.clone() - &q),
                    &np,
                ));
            } else {
                let nf = NumberField::new(fac.clone());
                let theta = nf.reduce(&vec![Rational::from(0), Rational::from(1)]);
                if prec.clone() - &q <= 0 {
                    let mut full = embed_prefix(&nf, prefix);
                    full.push((q.clone(), theta));
                    out.push(make_alg_series(Some(fac.clone()), deg, full, false, prec));
                    continue;
                }
                let gk = substitute_k(&nf, &embed(&nf, &g), &q, &theta);
                // Continue over ℚ(θ); a *deeper* non-linear characteristic over
                // ℚ(θ) collapses into a compositum via the tower machinery, so the
                // returned sub-branch may live in a field F ⊇ ℚ(θ).
                let (subs, _missed) = lift_tower(&nf, &gk, &(prec.clone() - &q), 0, deg);
                for sub in subs {
                    let kfinal = NumberField::new(sub.mp.clone());
                    // Embed the rational prefix and the θ head into F.
                    let mut full: Vec<(Rational, KElem)> = prefix
                        .iter()
                        .map(|(e, c)| (e.clone(), kfinal.from_rational(c)))
                        .collect();
                    let theta_f = embed_elem(&kfinal, &theta, &sub.theta_in_f);
                    full.push((q.clone(), theta_f));
                    full.extend(sub.terms);
                    out.push(make_alg_series(
                        Some(sub.mp.clone()),
                        sub.conjugates,
                        full,
                        sub.exact,
                        prec,
                    ));
                }
            }
        }
    }
    out
}

/// `w = x^q (c + w₁)` substitution over a number field, then divide by `x^ν`.
fn substitute_k(nf: &NumberField, g: &KBi, q: &Rational, c: &KElem) -> KBi {
    let mut g1: KBi = BTreeMap::new();
    for ((xe, ye), a) in g {
        let j = *ye;
        let new_xe = xe.clone() + q.clone() * Rational::from(j as i64);
        for l in 0..=j {
            let binom = k_from_int(nf, &binomial(j, l));
            let cpow = k_pow(nf, c, j - l);
            let coeff = nf.mul(&nf.mul(a, &binom), &cpow);
            if !NumberField::is_zero(&coeff) {
                let e = g1.entry((new_xe.clone(), l)).or_default();
                *e = nf.add(e, &coeff);
            }
        }
    }
    g1.retain(|_, a| !NumberField::is_zero(a));
    factor_min_x_k(&mut g1);
    g1
}

fn factor_min_x_k(g: &mut KBi) {
    let Some(v) = g.keys().map(|(xe, _)| xe.clone()).min() else {
        return;
    };
    if v == 0 {
        return;
    }
    *g = std::mem::take(g)
        .into_iter()
        .map(|((xe, ye), a)| ((xe - &v, ye), a))
        .collect();
}

fn embed(nf: &NumberField, g: &Bi) -> KBi {
    g.iter()
        .map(|((xe, ye), a)| ((xe.clone(), *ye), nf.reduce(&vec![a.clone()])))
        .collect()
}

fn embed_prefix(nf: &NumberField, prefix: &[(Rational, Rational)]) -> Vec<(Rational, KElem)> {
    prefix
        .iter()
        .map(|(e, c)| (e.clone(), nf.reduce(&vec![c.clone()])))
        .collect()
}

fn k_from_int(nf: &NumberField, n: &Integer) -> KElem {
    nf.reduce(&vec![Rational::from(n.clone())])
}

fn k_pow(nf: &NumberField, c: &KElem, e: u32) -> KElem {
    let mut acc = nf.reduce(&vec![Rational::from(1)]);
    for _ in 0..e {
        acc = nf.mul(&acc, c);
    }
    acc
}

fn make_alg_series(
    minpoly: Option<Vec<Rational>>,
    conjugates: usize,
    mut terms: Vec<(Rational, KElem)>,
    exact: bool,
    prec: &Rational,
) -> AlgPuiseuxSeries {
    terms.retain(|(e, c)| (exact || *e < *prec) && !NumberField::is_zero(c));
    terms.sort_by(|a, b| a.0.cmp(&b.0));
    let e = terms.iter().fold(1u64, |acc, (ex, _)| {
        lcm_u64(acc, ex.denom().to_u64().unwrap_or(1))
    });
    AlgPuiseuxSeries {
        minpoly,
        conjugates,
        ramification: e,
        terms,
        order: if exact { None } else { Some(prec.clone()) },
    }
}

/// Factor `φ(c) = Σ p[k] c^k` over `ℚ` into monic irreducible factors of degree
/// `≥ 1`, after dividing out the largest `c`-power (the root `c = 0`, not a
/// branch).  Returns `(monic factor, degree)`.
pub(crate) fn factor_over_q(p: &[Rational]) -> Vec<(Vec<Rational>, usize)> {
    let p = {
        let mut hi = p.len();
        while hi > 0 && p[hi - 1] == 0 {
            hi -= 1;
        }
        p[..hi].to_vec()
    };
    // Divide out low-order zeros (root c = 0).
    let lo = p.iter().position(|c| *c != 0).unwrap_or(p.len());
    let psi = &p[lo..];
    if psi.len() <= 1 {
        return Vec::new();
    }
    // Clear denominators → integer coefficients (ascending).
    let mut den_lcm = Integer::from(1);
    for c in psi {
        den_lcm = den_lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = psi
        .iter()
        .map(|c| {
            (c.clone() * Rational::from(den_lcm.clone()))
                .numer()
                .clone()
        })
        .collect();
    let fp = FlintPoly::from_rug_coefficients(&ints);
    let Ok((_unit, facs)) = fp.factor_over_z() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (fpoly, _mult) in facs {
        let deg = fpoly.degree();
        if deg < 1 {
            continue;
        }
        let icoeffs = fpoly.coefficients(); // ascending i64
        let qcoeffs: Vec<Rational> = icoeffs.iter().map(|&c| Rational::from(c)).collect();
        let lead = qcoeffs[deg as usize].clone();
        let monic: Vec<Rational> = qcoeffs.iter().map(|c| c.clone() / &lead).collect();
        out.push((monic, deg as usize));
    }
    out
}

// ---------------------------------------------------------------------------
// Expansion at an ALGEBRAIC base point  x = α  (α irrational, given by minpoly)
// ---------------------------------------------------------------------------

/// A Puiseux branch at an **algebraic** base point `x = α`, with coefficients in
/// the base number field `K = ℚ[t]/(α_minpoly)` (where the generator `t = α`).
///
/// The branch is `y(x) = Σ c_k (x − α)^{k/e}` with each `c_k ∈ K` (a `KElem`,
/// a ℚ-polynomial in `α`).  A class over a degree-`d` base field represents
/// `conjugates = d` concrete conjugate branches (one per embedding of `α`).
#[derive(Clone, Debug)]
pub struct AlgBasePuiseuxSeries {
    /// Minimal polynomial of the base point `α` over `ℚ` (monic, ascending) —
    /// the modulus of the **base** field `ℚ(α)`.
    pub alpha_minpoly: Vec<Rational>,
    /// Minimal polynomial (over `ℚ`, monic, ascending) of the field the branch
    /// **coefficients** actually live in.  Equals [`Self::alpha_minpoly`] for a
    /// branch that stays in `K = ℚ(α)`; for a branch whose continuation required a
    /// **further extension** (Trager + primitive-element collapse, risch.md §D),
    /// it is the minimal polynomial of the compositum `K' = ℚ(α)(θ') ⊋ K`.  A
    /// consumer that only handles `K = ℚ(α)` coefficients must restrict to
    /// branches with `coeff_minpoly == alpha_minpoly`.
    pub coeff_minpoly: Vec<Rational>,
    /// Number of conjugate base points the class stands for `= deg(coeff_minpoly)`
    /// — the number of concrete branches it represents (one per embedding of the
    /// coefficient field into `ℚ̄`).
    pub conjugates: usize,
    /// Ramification index `e`: every exponent has denominator dividing `e`.
    pub ramification: u64,
    /// `(exponent, coefficient ∈ K)` pairs in `(x − α)^{·}`, ascending.
    pub terms: Vec<(Rational, KElem)>,
    /// Truncation order (`None` ⇒ exact).
    pub order: Option<Rational>,
}

/// Branches of `F(x, y) = 0` at an **algebraic** base point `x = α`, where `α`
/// is an irrational algebraic number given by its minimal polynomial
/// `alpha_minpoly` over `ℚ`.
///
/// The natural construction: shift `x ↦ x + α` so the base point moves to
/// `0`, but now over the number field `K = ℚ(α)` — the curve coefficients
/// (polynomials in `ℚ[x]`) become `K[x]`-polynomials.  The classical
/// Newton–Puiseux recursion then runs with **coefficient arithmetic in `K`**
/// instead of `ℚ`, reusing the same generic `lift_tower` / `substitute_k` core that
/// backs [`puiseux_at_zero_algebraic`].  Exponents are powers of `(x − α)`.
///
/// Return type: a [`Vec`] of [`AlgBasePuiseuxSeries`] (each over `K`) plus a
/// `skipped` count of branches whose continuation needs a **further** extension
/// beyond `K` (a non-`K` characteristic root) — never mis-reported, only
/// counted, exactly as [`puiseux_at_zero_algebraic`] does for `ℚ(θ)`.
///
/// ## Scope
///
/// Complete for every branch whose Puiseux coefficients lie in `K = ℚ(α)`
/// itself (the constant root and every characteristic root stays in `K`).  This
/// covers: unramified / nodal places where the branch slopes are `K`-rational,
/// the degenerate case `α ∈ ℚ` (which agrees with [`puiseux_at`]), and radical
/// continuations whose characteristic factor is a binomial over `K`.  A branch
/// requiring a root in a proper extension of `K` (e.g. a ramified place whose
/// leading coefficient is `√(·)` of a non-square of `K`) is **skipped-but-
/// counted** — the `skipped` total plus the summed `conjugates` reveal whether
/// every sheet over `α` was recovered.  Lifting those branches needs a Puiseux
/// tower `ℚ(α)(θ)` (risch.md §D "Puiseux over a tower of extensions").
///
/// Every returned branch is back-substitution checked in the test suite:
/// `F(α + t^e, y(t)) ≡ 0 (mod t^N)` with **exact** arithmetic in `K[t]`.
pub fn puiseux_at_algebraic(
    coeffs: &[(u32, u32, Rational)],
    alpha_minpoly: &[Rational],
    prec: u32,
) -> (Vec<AlgBasePuiseuxSeries>, usize) {
    let deg_alpha = {
        let mut d = alpha_minpoly.len();
        while d > 0 && alpha_minpoly[d - 1] == 0 {
            d -= 1;
        }
        d.saturating_sub(1)
    };
    // Degenerate base field ℚ (deg ≤ 1): the rational path is sound and complete
    // for the rational branches; fold it through `puiseux_at` so the algebraic
    // entry agrees with the rational one on a rational α.
    if deg_alpha <= 1 {
        let alpha = if deg_alpha == 0 {
            rzero()
        } else {
            // minpoly = a₀ + a₁ t  ⇒  α = −a₀/a₁.
            -alpha_minpoly[0].clone() / alpha_minpoly[1].clone()
        };
        let branches = puiseux_at(coeffs, &alpha, prec);
        let out = branches
            .into_iter()
            .map(|s| AlgBasePuiseuxSeries {
                alpha_minpoly: vec![Rational::from(1)],
                coeff_minpoly: vec![Rational::from(1)],
                conjugates: 1,
                ramification: s.ramification,
                terms: s.terms.into_iter().map(|(e, c)| (e, vec![c])).collect(),
                order: s.order,
            })
            .collect();
        return (out, 0);
    }

    let nf = NumberField::new(alpha_minpoly.to_vec());
    let alpha = nf.reduce(&vec![rzero(), Rational::from(1)]); // α = t
    let prec_r = Rational::from(prec);

    // Shift x ↦ x + α over K: F(x+α, y) as a K-bivariate `(x-exp, y-exp) → K`.
    let mut f = shift_x_alpha(&nf, coeffs, &alpha);

    // Strip a common x-power (does not change the branches).
    factor_min_x_k(&mut f);
    if f.is_empty() {
        return (Vec::new(), 0);
    }

    // F(α, y): the y-fibre over the base point, a univariate over K.
    let mut f0: BTreeMap<u32, KElem> = BTreeMap::new();
    for ((xe, ye), a) in &f {
        if *xe == 0 {
            let e = f0.entry(*ye).or_default();
            *e = nf.add(e, a);
        }
    }
    let f0_dense = k_dense(&nf, &f0);

    let mut out: Vec<AlgBasePuiseuxSeries> = Vec::new();
    let mut skipped = 0usize;

    // Constant roots c₀ of F(α, y): the y-values of the branches at x = α.  Roots
    // in K continue directly; an irreducible-over-K factor of degree ≥ 2 collapses
    // to a compositum (tower continuation) when within the degree cap.
    let (roots, kfactors) = k_roots_and_factors(&nf, &f0_dense);
    for c0 in roots {
        let g = if NumberField::is_zero(&c0) {
            f.clone()
        } else {
            shift_y_k(&nf, &f, &c0)
        };
        let in_k_prefix = !NumberField::is_zero(&c0);
        let (lifted, missed) = lift_tower(&nf, &g, &prec_r, 0, deg_alpha);
        skipped += missed;
        for sub in lifted {
            let kfinal = NumberField::new(sub.mp.clone());
            let mut full = Vec::new();
            if in_k_prefix {
                // Embed the in-K constant root into the branch's final field.
                let c0_f = embed_elem(&kfinal, &c0, &sub.theta_in_f);
                full.push((rzero(), c0_f));
            }
            full.extend(sub.terms);
            out.push(make_alg_base_series_field(
                alpha_minpoly,
                &sub.mp,
                sub.conjugates,
                full,
                sub.exact,
                &prec_r,
            ));
        }
    }
    // Constant roots that escape K (an irreducible degree-≥2 K-factor of F(α,y)):
    // collapse to a compositum K' and continue the branch there.
    for (chi, fdeg) in kfactors {
        let Some((kp, theta_img, z_img)) = build_compositum(&nf, &chi) else {
            skipped += fdeg; // past the cap (or non-primitive): skip-count
            continue;
        };
        // z = c₀ ∈ K'; shift y ↦ z_img + w over K' and continue over K'.
        let f_kp = embed_bivariate(&kp, &f, &theta_img);
        let g = shift_y_k(&kp, &f_kp, &z_img);
        let conj_kp = kp.degree().max(0) as usize;
        let (lifted, missed) = lift_tower(&kp, &g, &prec_r, 0, conj_kp);
        skipped += missed;
        for sub in lifted {
            let kfinal = NumberField::new(sub.mp.clone());
            let c0_f = embed_elem(&kfinal, &z_img, &sub.theta_in_f);
            let mut full = vec![(rzero(), c0_f)];
            full.extend(sub.terms);
            out.push(make_alg_base_series_field(
                alpha_minpoly,
                &sub.mp,
                sub.conjugates,
                full,
                sub.exact,
                &prec_r,
            ));
        }
    }
    (out, skipped)
}

/// `F(x + α, y)` over `K = ℚ(α)`: the ℚ[x,y]-coefficients are embedded into `K`
/// and `x ↦ x + α` is expanded via the binomial theorem with `α ∈ K`.
fn shift_x_alpha(nf: &NumberField, coeffs: &[(u32, u32, Rational)], alpha: &KElem) -> KBi {
    let mut f: KBi = BTreeMap::new();
    for (i, j, a) in coeffs {
        if *a == 0 {
            continue;
        }
        let ak = nf.reduce(&vec![a.clone()]);
        // (x+α)^i = Σ_m C(i,m) α^{i−m} x^m.
        for m in 0..=*i {
            let binom = k_from_int(nf, &binomial(*i, m));
            let apow = k_pow(nf, alpha, *i - m);
            let coeff = nf.mul(&nf.mul(&ak, &binom), &apow);
            if !NumberField::is_zero(&coeff) {
                let e = f.entry((Rational::from(m), *j)).or_default();
                *e = nf.add(e, &coeff);
            }
        }
    }
    f.retain(|_, a| !NumberField::is_zero(a));
    f
}

/// `F(x, c₀ + w)` over `K` — the `y ↦ c₀ + w` shift with `c₀ ∈ K`.
fn shift_y_k(nf: &NumberField, f: &KBi, c0: &KElem) -> KBi {
    let mut g: KBi = BTreeMap::new();
    for ((xe, ye), a) in f {
        let j = *ye;
        for l in 0..=j {
            let binom = k_from_int(nf, &binomial(j, l));
            let cpow = k_pow(nf, c0, j - l);
            let coeff = nf.mul(&nf.mul(a, &binom), &cpow);
            if !NumberField::is_zero(&coeff) {
                let e = g.entry((xe.clone(), l)).or_default();
                *e = nf.add(e, &coeff);
            }
        }
    }
    g.retain(|_, a| !NumberField::is_zero(a));
    g
}

/// Dense coefficient vector (index = `c`-degree) of a sparse `degree → K` map.
fn k_dense(nf: &NumberField, m: &BTreeMap<u32, KElem>) -> Vec<KElem> {
    let Some(&maxd) = m.keys().max() else {
        return Vec::new();
    };
    let mut v = vec![NumberField::k_zero(); maxd as usize + 1];
    for (d, c) in m {
        v[*d as usize] = nf.reduce(c);
    }
    v
}

/// Roots **in `K`** of a univariate `φ(c) = Σ p[k] c^k` over `K`, together with the
/// `K`-irreducible factors of degree `≥ 2` (whose roots lie in a *proper*
/// extension of `K`).
///
/// Method (Trager, no global `K`-factoring needed beyond [`kfactor_over_k`]):
/// factor `φ` over `K` into monic `K`-irreducibles; a **degree-1** factor
/// `h₀ + h₁ c` contributes the `K`-root `−h₀/h₁`; a **degree-≥2** factor collects
/// roots conjugate over `K` and is returned so the caller can build a compositum
/// and collapse the tower (or skip-but-count past the degree cap).  The root
/// `c = 0` (when `c | φ`) is included among the roots.
fn k_roots_and_factors(nf: &NumberField, p: &[KElem]) -> (Vec<KElem>, Vec<KFactor>) {
    let mut hi = p.len();
    while hi > 0 && NumberField::is_zero(&p[hi - 1]) {
        hi -= 1;
    }
    let p = &p[..hi];
    if p.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut roots: Vec<KElem> = Vec::new();
    let mut lo = 0usize;
    while lo < p.len() && NumberField::is_zero(&p[lo]) {
        lo += 1;
    }
    if lo > 0 {
        roots.push(NumberField::k_zero());
    }
    let work: Vec<KElem> = p[lo..].iter().map(|c| nf.reduce(c)).collect();
    if work.len() <= 1 {
        return (roots, Vec::new());
    }
    let mut factors: Vec<KFactor> = Vec::new();
    for (h, hdeg) in kfactor_over_k(nf, &work) {
        if hdeg == 1 {
            if let Some(inv) = nf.inv(&h[1]) {
                roots.push(nf.neg(&nf.mul(&h[0], &inv)));
            }
        } else {
            factors.push((h, hdeg));
        }
    }
    (roots, factors)
}

/// The `ℚ`-norm `N_{K/ℚ}(φ)` of a polynomial `φ(c)` over `K` — a `ℚ`-polynomial
/// in `c` whose roots include every root of `φ` (together with the `α`-conjugate
/// shifts).  Computed as `Res_t(m_α(t), Φ(c, t))`, the resultant eliminating the
/// `α`-variable `t` from the bivariate lift `Φ(c, t)` of `φ` (each `K`-coefficient
/// `p[k]` is a `ℚ`-polynomial in `t = α`).  Evaluated at enough integer points
/// `c = s` and Lagrange-interpolated, the resultant being a `ℚ`-poly of bounded
/// `c`-degree.
fn k_norm_poly(nf: &NumberField, p: &[KElem]) -> Vec<Rational> {
    let m_alpha = nf.modulus().clone();
    let d_alpha = (m_alpha.len() as i64 - 1).max(0) as usize;
    // deg_c Res ≤ deg_c(Φ) · deg_t(m_α) = (len(p)-1) · d_alpha.
    let cdeg_bound = p.len().saturating_sub(1) * d_alpha;
    let n_pts = cdeg_bound + 1;
    let mut xs: Vec<Rational> = Vec::with_capacity(n_pts);
    let mut ys: Vec<Rational> = Vec::with_capacity(n_pts);
    let mut s: i64 = 0;
    while xs.len() < n_pts {
        let cs = Rational::from(s);
        // Φ(s, t): a ℚ-poly in t = Σ_k p[k](t) · s^k.
        let mut phi_t: Vec<Rational> = Vec::new();
        let mut spow = Rational::from(1);
        for pk in p {
            for (i, coeff) in pk.iter().enumerate() {
                if i >= phi_t.len() {
                    phi_t.resize(i + 1, rzero());
                }
                phi_t[i] += coeff.clone() * &spow;
            }
            spow *= &cs;
        }
        let r = q_resultant(&m_alpha, &phi_t);
        xs.push(cs);
        ys.push(r);
        s += 1;
    }
    lagrange_interpolate(&xs, &ys)
}

/// Resultant `Res(a, b)` of two `ℚ`-polynomials (ascending) via the Euclidean
/// remainder sequence with the standard degree / leading-coefficient
/// bookkeeping.  Returns a rational scalar.
fn q_resultant(a: &[Rational], b: &[Rational]) -> Rational {
    let mut a = trim_q(a.to_vec());
    let mut b = trim_q(b.to_vec());
    if a.is_empty() || b.is_empty() {
        return rzero();
    }
    let mut res = Rational::from(1);
    loop {
        let da = a.len() - 1;
        let db = b.len() - 1;
        if db == 0 {
            // Res(a, const) = const^{deg a}.
            res *= rat_pow_q(&b[0], da as u32);
            return res;
        }
        let rem = q_rem(&a, &b);
        let drem = if rem.is_empty() { 0 } else { rem.len() - 1 };
        // Res(a,b) = (-1)^{da·db} · lc(b)^{da−drem} · Res(b, rem).
        let sign = if (da * db) % 2 == 0 {
            Rational::from(1)
        } else {
            Rational::from(-1)
        };
        let lc_b = b[db].clone();
        res *= sign * rat_pow_q(&lc_b, (da - drem) as u32);
        if rem.is_empty() {
            return rzero();
        }
        a = b;
        b = rem;
    }
}

/// Remainder of `a mod b` over `ℚ` (ascending coefficient vectors).
fn q_rem(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let mut r = trim_q(a.to_vec());
    let b = trim_q(b.to_vec());
    let db = b.len() - 1;
    let lc_inv = Rational::from(1) / b[db].clone();
    loop {
        let dr = if r.is_empty() { 0 } else { r.len() - 1 };
        if r.is_empty() || dr < db {
            break;
        }
        let factor = r[dr].clone() * &lc_inv;
        let shift = dr - db;
        for (i, bc) in b.iter().enumerate() {
            r[shift + i] -= factor.clone() * bc;
        }
        r = trim_q(r);
    }
    r
}

fn trim_q(mut p: Vec<Rational>) -> Vec<Rational> {
    while p.last().is_some_and(|c| *c == 0) {
        p.pop();
    }
    p
}

fn rat_pow_q(c: &Rational, e: u32) -> Rational {
    let mut acc = Rational::from(1);
    for _ in 0..e {
        acc *= c;
    }
    acc
}

/// Lagrange interpolation through `(xs, ys)` (distinct `xs`) → ℚ-polynomial
/// (ascending).
fn lagrange_interpolate(xs: &[Rational], ys: &[Rational]) -> Vec<Rational> {
    let n = xs.len();
    let mut acc: Vec<Rational> = vec![rzero(); n];
    for i in 0..n {
        // Basis poly Lᵢ = ∏_{j≠i} (x − xⱼ)/(xᵢ − xⱼ).
        let mut basis = vec![Rational::from(1)];
        let mut denom = Rational::from(1);
        for j in 0..n {
            if i == j {
                continue;
            }
            // multiply basis by (x − xⱼ).
            let mut nb = vec![rzero(); basis.len() + 1];
            for (k, c) in basis.iter().enumerate() {
                nb[k] += -xs[j].clone() * c;
                nb[k + 1] += c.clone();
            }
            basis = nb;
            denom *= xs[i].clone() - &xs[j];
        }
        let scale = ys[i].clone() / denom;
        for (k, c) in basis.iter().enumerate() {
            acc[k] += c.clone() * &scale;
        }
    }
    trim_q(acc)
}

/// A lifted branch, tagged with the field its coefficients live in.  Fields:
/// * `mp` — minimal polynomial over `ℚ` of the branch's coefficient field `F`;
/// * `conjugates` — `[F:ℚ]`;
/// * `theta_in_f` — the image in `F` of the **current** recursion level's
///   generator `θ` (identity `[0,1]` when `F` is the current field, otherwise a
///   `ℚ`-polynomial in the compositum generator), so a caller can embed its own
///   head coefficient `c ∈ K_current` into `F` via [`embed_elem`];
/// * `terms` — `(exponent, coefficient ∈ F)` pairs;
/// * `exact` — terminating-branch flag.
struct TowerBranch {
    mp: Vec<Rational>,
    conjugates: usize,
    theta_in_f: KElem,
    terms: Vec<(Rational, KElem)>,
    exact: bool,
}

/// Newton-recursion over `K = ℚ[t]/(nf)` that, at every edge, finds the
/// characteristic roots **in `K`** (continue directly) *and* collapses any
/// `K`-irreducible factor of degree `≥ 2` into a compositum `K' ⊋ K` via Trager
/// factorization + primitive element, then continues over `K'` (risch.md §D
/// "Puiseux over a tower of extensions").  Returns the lifted branches (each
/// tagged with its coefficient field) and the count of sheets still skipped (past
/// the degree cap or non-primitive).  `conj_base = [K:ℚ]` is the conjugate
/// multiplier carried for branches that stay in `K`.
fn lift_tower(
    nf: &NumberField,
    g: &KBi,
    prec: &Rational,
    depth: u32,
    conj_base: usize,
) -> (Vec<TowerBranch>, usize) {
    const MAX_DEPTH: u32 = 48;
    let mut g = g.clone();
    g.retain(|_, a| !NumberField::is_zero(a));
    let here_mp = nf.modulus().clone();
    let ident: KElem = nf.reduce(&vec![rzero(), Rational::from(1)]); // θ = t
    let mk_leaf = |exact: bool| TowerBranch {
        mp: here_mp.clone(),
        conjugates: conj_base,
        theta_in_f: ident.clone(),
        terms: Vec::new(),
        exact,
    };
    if g.is_empty() {
        return (vec![mk_leaf(true)], 0);
    }
    if depth > MAX_DEPTH {
        return (vec![mk_leaf(false)], 0);
    }
    let mut result: Vec<TowerBranch> = Vec::new();
    let mut skipped = 0usize;
    let m0 = g.keys().map(|(_, j)| *j).min().unwrap_or(0);
    if m0 > 0 {
        result.push(mk_leaf(true));
        g = g
            .into_iter()
            .map(|((xe, ye), a)| ((xe, ye - m0), a))
            .collect();
    }
    let keys: Vec<(Rational, u32)> = g.keys().cloned().collect();
    for (q, on_edge) in newton_edges_keys(&keys) {
        let mut phi: BTreeMap<u32, KElem> = BTreeMap::new();
        for k in &on_edge {
            let e = phi.entry(k.1).or_default();
            *e = nf.add(e, &g[k]);
        }
        let phi_dense = k_dense(nf, &phi);
        let (roots, kfactors) = k_roots_and_factors(nf, &phi_dense);
        // Roots in K: continue over K (no field change at this step).
        for c in roots {
            if NumberField::is_zero(&c) {
                continue;
            }
            if prec.clone() - &q <= 0 {
                result.push(TowerBranch {
                    mp: here_mp.clone(),
                    conjugates: conj_base,
                    theta_in_f: ident.clone(),
                    terms: vec![(q.clone(), c)],
                    exact: false,
                });
                continue;
            }
            let gk = substitute_k(nf, &g, &q, &c);
            let (subs, sub_missed) =
                lift_tower(nf, &gk, &(prec.clone() - &q), depth + 1, conj_base);
            skipped += sub_missed;
            for sub in subs {
                // The sub-branch's field F may extend K (a deeper tower spawn);
                // embed this head (q, c) into F via the recorded θ-image.
                let kfinal = NumberField::new(sub.mp.clone());
                let head_c = embed_elem(&kfinal, &c, &sub.theta_in_f);
                let mut terms = vec![(q.clone(), head_c)];
                for (gamma, b) in sub.terms {
                    terms.push((q.clone() + &gamma, b));
                }
                result.push(TowerBranch {
                    mp: sub.mp,
                    conjugates: sub.conjugates,
                    theta_in_f: sub.theta_in_f,
                    terms,
                    exact: sub.exact,
                });
            }
        }
        // K-irreducible factors of degree ≥ 2: collapse to a compositum K' ⊋ K.
        for (chi, fdeg) in kfactors {
            let Some((kp, theta_img, z_img)) = build_compositum(nf, &chi) else {
                skipped += fdeg; // past the cap / non-primitive: skip-count
                continue;
            };
            let conj_kp = kp.degree().max(0) as usize;
            // Re-express g in K', then substitute w = x^q (z_img + w₁) and recurse.
            let g_kp = embed_bivariate(&kp, &g, &theta_img);
            if prec.clone() - &q <= 0 {
                result.push(TowerBranch {
                    mp: kp.modulus().clone(),
                    conjugates: conj_kp,
                    theta_in_f: theta_img.clone(),
                    terms: vec![(q.clone(), z_img)],
                    exact: false,
                });
                continue;
            }
            let gk = substitute_k(&kp, &g_kp, &q, &z_img);
            let (subs, sub_missed) = lift_tower(&kp, &gk, &(prec.clone() - &q), depth + 1, conj_kp);
            skipped += sub_missed;
            for sub in subs {
                let kfinal = NumberField::new(sub.mp.clone());
                // Head root z ∈ K' embedded into the (possibly deeper) final field.
                let head_c = embed_elem(&kfinal, &z_img, &sub.theta_in_f);
                // Compose the θ-images: θ(old K) ↦ theta_img(K') ↦ final.
                let theta_old_in_final = embed_elem(&kfinal, &theta_img, &sub.theta_in_f);
                let mut terms = vec![(q.clone(), head_c)];
                for (gamma, b) in sub.terms {
                    terms.push((q.clone() + &gamma, b));
                }
                result.push(TowerBranch {
                    mp: sub.mp,
                    conjugates: sub.conjugates,
                    theta_in_f: theta_old_in_final,
                    terms,
                    exact: sub.exact,
                });
            }
        }
    }
    (result, skipped)
}

fn make_alg_base_series_field(
    alpha_minpoly: &[Rational],
    coeff_minpoly: &[Rational],
    conjugates: usize,
    mut terms: Vec<(Rational, KElem)>,
    exact: bool,
    prec: &Rational,
) -> AlgBasePuiseuxSeries {
    terms.retain(|(e, c)| (exact || *e < *prec) && !NumberField::is_zero(c));
    terms.sort_by(|a, b| a.0.cmp(&b.0));
    let e = terms.iter().fold(1u64, |acc, (ex, _)| {
        lcm_u64(acc, ex.denom().to_u64().unwrap_or(1))
    });
    AlgBasePuiseuxSeries {
        alpha_minpoly: alpha_minpoly.to_vec(),
        coeff_minpoly: coeff_minpoly.to_vec(),
        conjugates,
        ramification: e,
        terms,
        order: if exact { None } else { Some(prec.clone()) },
    }
}

// ---------------------------------------------------------------------------
// Tower continuation: primitive-element collapse  (risch.md §D item 2)
// ---------------------------------------------------------------------------
//
// A Newton-polygon characteristic polynomial `χ(z)` may be **irreducible of
// degree d ≥ 2 over the current field K = ℚ(θ)** — its roots generate a *proper*
// extension of `K`, so the branch cannot continue with `K`-coefficients.  Rather
// than build a genuine tower `K(z)` and carry two generators, we collapse to a
// single primitive element: `K' = ℚ(θ')` with `θ' = θ + λ·z` for a small integer
// `λ`, and continue the recursion over `K'` (where `z` — hence the root — is now a
// `K'`-element).  This is **Trager's classical method**:
//
// 1. **`K`-factorization of `χ`** (`kfactor_over_k`): factor the `ℚ`-norm
//    `N_{K/ℚ}(χ)` over `ℚ` (FLINT) for a shift that makes the norm squarefree,
//    then recover the `K`-irreducible factors as `gcd_K(χ, gᵢ(z))` for each
//    `ℚ`-irreducible norm factor `gᵢ`.
// 2. **Primitive-element compositum** (`build_compositum`): for a `K`-irreducible
//    factor `χ₁` of degree `d`, build `K' = ℚ[t']/(m')` with `m'` the minimal
//    polynomial of `θ' = θ + λ·z` over `ℚ` (`deg m' = [K:ℚ]·d`), together with the
//    **exact embeddings** `θ ↦ ι_θ(t')`, `z ↦ ι_z(t')` recovered by linear algebra
//    in the basis `{θⁱ zʲ}`.
// 3. Re-express the partial branch (its `K`-coefficients and the working
//    `K`-bivariate `g`) in `K'` via the embedding and **continue the recursion**.
//
// The total field degree is **capped** at [`TOWER_DEGREE_CAP`]; past the cap the
// branch is left **skip-but-count** (sound).  Every returned branch is exact
// back-substitution-checked (the test convention is unchanged).

/// Maximum `[K':ℚ]` the tower collapse will build.  Beyond it, a branch needing a
/// further extension is skip-but-counted (sound) rather than collapsed.
const TOWER_DEGREE_CAP: usize = 16;

/// A `K`-irreducible factor of a characteristic polynomial, with its degree.
type KFactor = (KPoly, usize);

/// Factor a univariate `φ(z) = Σ p[k] zᵏ` over `K = ℚ[t]/(nf.modulus())` into
/// monic-in-`z` `K`-irreducible factors, after dividing out the largest `z`-power
/// (the root `z = 0`, handled separately).  Returns the factors with their
/// `z`-degrees.
///
/// **Trager's algorithm.**  Pick a small integer shift `s` so the `ℚ`-norm
/// `N_{K/ℚ}(φ(z − s·θ))` is **squarefree** (separable); factor that norm over `ℚ`
/// (FLINT, via [`factor_over_q`]); each `ℚ`-irreducible norm factor `g(z)` yields a
/// `K`-irreducible factor of `φ` as `gcd_K(φ_shifted, g)` (then shift back
/// `z ↦ z + s·θ`).  Squarefreeness of the norm is exactly the condition that this
/// gcd-recovery is unambiguous.
fn kfactor_over_k(nf: &NumberField, p: &[KElem]) -> Vec<KFactor> {
    // Trim trailing zeros; strip leading z-power (root 0 handled by caller).
    let mut hi = p.len();
    while hi > 0 && NumberField::is_zero(&p[hi - 1]) {
        hi -= 1;
    }
    let p = &p[..hi];
    let mut lo = 0usize;
    while lo < p.len() && NumberField::is_zero(&p[lo]) {
        lo += 1;
    }
    let work: KPoly = p[lo..].iter().map(|c| nf.reduce(c)).collect();
    if NumberField::kdeg(&work) < 1 {
        return Vec::new();
    }
    // Make φ monic in z over K.
    let Some(phi) = nf.kpoly_monic(&work) else {
        return Vec::new();
    };
    let theta: KElem = nf.reduce(&vec![rzero(), Rational::from(1)]); // θ = t

    // Find a shift s with squarefree norm.
    for s in 0..=8i64 {
        // φ_s(z) = φ(z − s·θ): substitute z ↦ z − s·θ.
        let phi_s = if s == 0 {
            phi.clone()
        } else {
            let shift = nf.mul(&nf.from_int(-s), &theta); // −s·θ ∈ K
            kpoly_shift_z(nf, &phi, &shift)
        };
        let norm = k_norm_poly(nf, &phi_s);
        if norm.len() <= 1 || !is_squarefree_q(&norm) {
            continue;
        }
        let mut out: Vec<KFactor> = Vec::new();
        let mut ok = true;
        for (g, _deg) in factor_over_q(&norm) {
            let gk: KPoly = g.iter().map(|c| nf.reduce(&vec![c.clone()])).collect();
            let Some(h) = nf.kpoly_gcd(&phi_s, &gk) else {
                ok = false;
                break;
            };
            let hdeg = NumberField::kdeg(&h);
            if hdeg < 1 {
                ok = false; // norm factor with no K-common root: norm not separable
                break;
            }
            // Shift back: z ↦ z + s·θ.
            let h_back = if s == 0 {
                h
            } else {
                let shift = nf.mul(&nf.from_int(s), &theta);
                kpoly_shift_z(nf, &h, &shift)
            };
            let Some(h_monic) = nf.kpoly_monic(&h_back) else {
                ok = false;
                break;
            };
            let d = NumberField::kdeg(&h_monic) as usize;
            out.push((h_monic, d));
        }
        if ok && !out.is_empty() {
            return out;
        }
    }
    // No good shift found within the search window: report the whole polynomial as
    // one factor (its degree is the count of escaping sheets — sound skip-count).
    let d = NumberField::kdeg(&phi) as usize;
    vec![(phi, d)]
}

/// `φ(z + c)` for `c ∈ K`: substitute `z ↦ z + c` in a monic `K`-polynomial via
/// repeated Horner-style expansion (`Σ pₖ (z+c)ᵏ`).
fn kpoly_shift_z(nf: &NumberField, phi: &[KElem], c: &KElem) -> KPoly {
    // (z + c)ᵏ accumulated incrementally.
    let mut acc: KPoly = Vec::new();
    let mut zpc_pow: KPoly = vec![nf.from_int(1)]; // (z+c)^0
    let zpc: KPoly = vec![c.clone(), nf.from_int(1)]; // z + c
    for (k, pk) in phi.iter().enumerate() {
        if k > 0 {
            zpc_pow = nf.kpoly_mul(&zpc_pow, &zpc);
        }
        if !NumberField::is_zero(pk) {
            let term = nf.kpoly_scale(&zpc_pow, pk);
            acc = nf.kpoly_add(&acc, &term);
        }
    }
    NumberField::kpoly_trim(acc)
}

/// Is a `ℚ`-polynomial (ascending) squarefree?  `gcd(f, f') = const`.
fn is_squarefree_q(f: &[Rational]) -> bool {
    let f = trim_q(f.to_vec());
    if f.len() <= 1 {
        return true;
    }
    // f' (ascending).
    let mut df: Vec<Rational> = Vec::with_capacity(f.len().saturating_sub(1));
    for (k, c) in f.iter().enumerate().skip(1) {
        df.push(Rational::from(k as i64) * c);
    }
    let g = q_gcd(&f, &df);
    g.len() <= 1
}

/// Monic `ℚ`-GCD (ascending) via the Euclidean algorithm.
fn q_gcd(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let mut a = trim_q(a.to_vec());
    let mut b = trim_q(b.to_vec());
    while !b.is_empty() {
        let r = q_rem(&a, &b);
        a = b;
        b = trim_q(r);
    }
    if let Some(lc) = a.last().cloned() {
        if lc != 0 {
            for c in a.iter_mut() {
                *c /= &lc;
            }
        }
    }
    a
}

/// The compositum `K' = ℚ(θ')`, `θ' = θ + λ·z`, of `K = ℚ[t]/(nf.modulus())` and a
/// `K`-irreducible `χ₁(z)` of degree `d ≥ 2`.  Returns
/// `(K', ι_θ, ι_z)` where `K'` is the number field `ℚ[t']/(m')`,
/// `ι_θ ∈ K'` is the image of the old generator `θ`, and `ι_z ∈ K'` is the image
/// of the adjoined root `z` (a root of `χ₁`).  Both images are **exact** (verified
/// by reconstruction in the basis `{θⁱ zʲ}`).  `None` if no small `λ` is primitive
/// or `[K':ℚ] > TOWER_DEGREE_CAP`.
#[allow(clippy::type_complexity)]
fn build_compositum(nf: &NumberField, chi: &[KElem]) -> Option<(NumberField, KElem, KElem)> {
    let n = nf.degree().max(0) as usize; // [K:ℚ]
    let chi_monic = nf.kpoly_monic(chi)?;
    let d = NumberField::kdeg(&chi_monic).max(0) as usize; // deg χ₁
    if d < 2 || n == 0 {
        return None;
    }
    let big = n * d; // [K':ℚ]
    if big > TOWER_DEGREE_CAP {
        return None;
    }

    // Arithmetic in T = K[z]/(χ₁): an element is a KPoly of z-degree < d.
    // Basis of T over ℚ: {θⁱ zʲ : 0≤i<n, 0≤j<d}, flattened index = j·n + i.
    let tred = |a: &[KElem]| -> KPoly {
        // reduce a (z-poly over K) mod χ₁.
        match nf.kpoly_divrem(a, &chi_monic) {
            Some((_q, rem)) => NumberField::kpoly_trim(rem),
            None => NumberField::kpoly_trim(a.to_vec()),
        }
    };
    let tmul = |a: &[KElem], b: &[KElem]| -> KPoly { tred(&nf.kpoly_mul(a, b)) };
    let flat = |a: &[KElem]| -> Vec<Rational> {
        let mut v = vec![rzero(); big];
        for (j, cj) in a.iter().enumerate().take(d) {
            // cj ∈ K is a ℚ-poly in θ of degree < n.
            let cjr = nf.reduce(cj);
            for (i, ci) in cjr.iter().enumerate().take(n) {
                v[j * n + i] = ci.clone();
            }
        }
        v
    };

    let z_elem: KPoly = vec![NumberField::k_zero(), nf.from_int(1)]; // z
    let theta_elem: KPoly = vec![nf.reduce(&vec![rzero(), Rational::from(1)])]; // θ (in K, z-deg 0)

    for lambda in 1..=8i64 {
        // θ' = θ + λ·z  in T.
        let theta_p =
            tred(&nf.kpoly_add(&theta_elem, &nf.kpoly_scale(&z_elem, &nf.from_int(lambda))));
        // Powers θ'^0 … θ'^big as ℚ^big columns.
        let one_t: KPoly = vec![nf.from_int(1)];
        let mut powers: Vec<KPoly> = vec![one_t.clone()];
        let mut cur = one_t.clone();
        for _ in 0..big {
            cur = tmul(&cur, &theta_p);
            powers.push(cur.clone());
        }
        let cols: Vec<Vec<Rational>> = (0..big).map(|k| flat(&powers[k])).collect();
        // Solve θ'^big = Σ mₖ θ'^k  ⇒ m' = t^big − Σ mₖ t^k.
        let Some(mcoef) = solve_q_columns(&cols, &flat(&powers[big]), big) else {
            continue; // θ' not primitive at this λ
        };
        let mut mprime = vec![rzero(); big + 1];
        mprime[big] = Rational::from(1);
        for (k, mk) in mcoef.iter().enumerate() {
            mprime[k] = -mk.clone();
        }
        // Images of θ and z in K' = ℚ[t']/(m').
        let Some(theta_in) = solve_q_columns(&cols, &flat(&theta_elem), big) else {
            continue;
        };
        let Some(z_in) = solve_q_columns(&cols, &flat(&z_elem), big) else {
            continue;
        };
        let kprime = NumberField::new(trim_q(mprime));
        return Some((
            kprime.clone(),
            kprime.reduce(&trim_q(theta_in)),
            kprime.reduce(&trim_q(z_in)),
        ));
    }
    None
}

/// Solve `A·x = rhs` over `ℚ` (columns `cols`, each length `n`).  `None` if `A` is
/// singular.  (A local copy mirroring `alg_tower::solve_columns`, kept here to keep
/// `poly` free of an `integrate` dependency.)
fn solve_q_columns(cols: &[Vec<Rational>], rhs: &[Rational], n: usize) -> Option<Vec<Rational>> {
    let mut a: Vec<Vec<Rational>> = (0..n)
        .map(|i| {
            let mut row: Vec<Rational> = (0..n).map(|j| cols[j][i].clone()).collect();
            row.push(rhs[i].clone());
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n).find(|&rr| a[rr][col] != 0)?;
        a.swap(col, piv);
        let inv = Rational::from(1) / a[col][col].clone();
        for v in a[col].iter_mut() {
            *v *= &inv;
        }
        for rr in 0..n {
            if rr != col && a[rr][col] != 0 {
                let f = a[rr][col].clone();
                #[allow(clippy::needless_range_loop)]
                for k in col..=n {
                    let s = f.clone() * &a[col][k];
                    a[rr][k] -= s;
                }
            }
        }
    }
    Some((0..n).map(|i| a[i][n].clone()).collect())
}

/// Re-express a `K`-element `c` (ℚ-poly in `θ`) as a `K'`-element, given the image
/// `ι_θ ∈ K'` of `θ`: evaluate the ℚ-polynomial `c` at `t = ι_θ` in `K'` (Horner).
fn embed_elem(kp: &NumberField, c: &KElem, theta_img: &KElem) -> KElem {
    let mut acc = NumberField::k_zero();
    for ci in c.iter().rev() {
        acc = kp.mul(&acc, theta_img);
        if *ci != 0 {
            acc = kp.add(&acc, &kp.from_rational(ci));
        }
    }
    acc
}

/// Re-express a `K`-bivariate `g` (`(x-exp, y-exp) → KElem` over `K`) in `K'` via
/// the embedding `θ ↦ theta_img`.
fn embed_bivariate(kp: &NumberField, g: &KBi, theta_img: &KElem) -> KBi {
    let mut out: KBi = BTreeMap::new();
    for (k, a) in g {
        let e = embed_elem(kp, a, theta_img);
        if !NumberField::is_zero(&e) {
            out.insert(k.clone(), e);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rational {
        Rational::from(n)
    }
    fn rr(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    /// Find the branch whose leading (exponent, coeff) matches, for assertions.
    fn has_term(s: &PuiseuxSeries, exp: Rational, coeff: Rational) -> bool {
        s.terms.iter().any(|(e, c)| *e == exp && *c == coeff)
    }

    /// Back-substitution soundness check: every returned branch must satisfy
    /// `F(x, y(x)) = O(x^prec)` numerically at a few small `x` samples.
    fn verify_branches(f: &[(u32, u32, Rational)], prec: u32) {
        let br = puiseux_at_zero(f, prec);
        assert!(!br.is_empty(), "expected at least one branch");
        for s in &br {
            for &x0 in &[0.01_f64, 0.03, 0.07] {
                // y(x0) = Σ c_k x0^{e_k}.
                let y: f64 = s
                    .terms
                    .iter()
                    .map(|(e, c)| c.to_f64() * x0.powf(e.to_f64()))
                    .sum();
                let fval: f64 = f
                    .iter()
                    .map(|(i, j, a)| a.to_f64() * x0.powi(*i as i32) * y.powi(*j as i32))
                    .sum();
                // F vanishes to order ~ prec along the branch.
                let tol = 1e-6 + 50.0 * x0.powf(prec as f64);
                assert!(
                    fval.abs() < tol,
                    "branch {s:?}: F({x0}, y)={fval} not O(x^{prec})"
                );
            }
        }
    }

    #[test]
    fn back_substitution_soundness() {
        verify_branches(&[(0, 2, r(1)), (1, 0, r(-1))], 4); // y²−x
        verify_branches(&[(0, 2, r(1)), (3, 0, r(-1))], 5); // y²−x³
        verify_branches(&[(0, 3, r(1)), (1, 0, r(-1))], 4); // y³−x
        verify_branches(&[(0, 2, r(1)), (2, 0, r(-1)), (3, 0, r(-1))], 5); // y²−x²−x³
        verify_branches(
            &[(0, 2, r(1)), (1, 1, r(-2)), (2, 0, r(1)), (3, 0, r(-1))],
            4,
        ); // (y−x)²−x³
    }

    #[test]
    fn puiseux_at_base_point() {
        // y² − (x−1) = 0 ⇒ y = ±(x−1)^{1/2}; expand at x=1.
        let f = [(0, 2, r(1)), (1, 0, r(-1)), (0, 0, r(1))]; // y² − x + 1
        let br = puiseux_at(&f, &r(1), 3);
        assert_eq!(br.len(), 2);
        for s in &br {
            assert_eq!(s.ramification, 2);
            assert!(has_term(s, rr(1, 2), r(1)) || has_term(s, rr(1, 2), r(-1)));
        }
    }

    #[test]
    fn algebraic_cube_root_of_unity() {
        // y³ − x: rational branch x^{1/3} (c=1) + an algebraic class (c=ω) over
        // ℚ[t]/(t²+t+1) with conjugates=2.  Total sheets 1+2 = 3 = deg_y.
        let br = puiseux_at_zero_algebraic(&[(0, 3, r(1)), (1, 0, r(-1))], 2);
        let total: usize = br.iter().map(|s| s.conjugates).sum();
        assert_eq!(total, 3, "branches: {br:?}");
        let alg = br
            .iter()
            .find(|s| s.minpoly.is_some())
            .expect("an algebraic branch");
        assert_eq!(alg.conjugates, 2);
        assert_eq!(alg.minpoly.as_ref().unwrap(), &vec![r(1), r(1), r(1)]);
        assert_eq!(alg.ramification, 3);
        assert_eq!(alg.terms[0].0, rr(1, 3));
        assert_eq!(alg.terms[0].1, vec![r(0), r(1)]); // coefficient θ
                                                      // Soundness of the leading term: θ³ = 1 in ℚ(ω), so (θ x^{1/3})³ = x.
        let nf = NumberField::new(vec![r(1), r(1), r(1)]);
        let theta = vec![r(0), r(1)];
        let theta3 = nf.mul(&nf.mul(&theta, &theta), &theta);
        assert_eq!(nf.reduce(&theta3), vec![r(1)]);
    }

    #[test]
    fn algebraic_constant_branches() {
        // y² − 2 ⇒ y = ±√2: a constant algebraic class over ℚ[t]/(t²−2).
        let br = puiseux_at_zero_algebraic(&[(0, 2, r(1)), (0, 0, r(-2))], 2);
        let alg = br
            .iter()
            .find(|s| s.minpoly.is_some())
            .expect("an algebraic branch");
        assert_eq!(alg.conjugates, 2);
        assert_eq!(alg.minpoly.as_ref().unwrap(), &vec![r(-2), r(0), r(1)]);
        assert_eq!(alg.terms.len(), 1);
        assert_eq!(alg.terms[0], (r(0), vec![r(0), r(1)]));
    }

    #[test]
    fn algebraic_includes_rational_branches() {
        // y² − x still yields its two rational branches via the algebraic entry.
        let br = puiseux_at_zero_algebraic(&[(0, 2, r(1)), (1, 0, r(-1))], 3);
        let total: usize = br.iter().map(|s| s.conjugates).sum();
        assert_eq!(total, 2);
        assert!(br.iter().all(|s| s.minpoly.is_none()));
    }

    #[test]
    fn sqrt_x() {
        // y² − x = 0  ⇒  y = ± x^{1/2}.
        let f = [(0, 2, r(1)), (1, 0, r(-1))];
        let br = puiseux_at_zero(&f, 3);
        assert_eq!(br.len(), 2);
        for s in &br {
            assert_eq!(s.ramification, 2);
            assert!(has_term(s, rr(1, 2), r(1)) || has_term(s, rr(1, 2), r(-1)));
        }
    }

    #[test]
    fn cusp_y2_eq_x3() {
        // y² − x³ = 0  ⇒  y = ± x^{3/2}.
        let f = [(0, 2, r(1)), (3, 0, r(-1))];
        let br = puiseux_at_zero(&f, 4);
        assert_eq!(br.len(), 2);
        for s in &br {
            assert_eq!(s.ramification, 2);
            assert!(has_term(s, rr(3, 2), r(1)) || has_term(s, rr(3, 2), r(-1)));
        }
    }

    #[test]
    fn cbrt_x_principal_branch() {
        // y³ − x = 0: the only rational branch is y = x^{1/3} (others need ω).
        let f = [(0, 3, r(1)), (1, 0, r(-1))];
        let br = puiseux_at_zero(&f, 2);
        assert_eq!(br.len(), 1);
        assert_eq!(br[0].ramification, 3);
        assert!(has_term(&br[0], rr(1, 3), r(1)));
    }

    #[test]
    fn double_root_recursion() {
        // (y − x)² − x³ = 0  ⇒  y = x ± x^{3/2}.  Characteristic at q=1 is (c−1)²
        // (a double root), exercising the recursion.
        // (y−x)² − x³ = y² − 2xy + x² − x³.
        let f = [(0, 2, r(1)), (1, 1, r(-2)), (2, 0, r(1)), (3, 0, r(-1))];
        let br = puiseux_at_zero(&f, 3);
        assert_eq!(br.len(), 2, "branches: {br:?}");
        for s in &br {
            assert!(has_term(s, r(1), r(1)), "leading x term: {s:?}");
            assert!(has_term(s, rr(3, 2), r(1)) || has_term(s, rr(3, 2), r(-1)));
        }
    }

    #[test]
    fn multi_term_taylor_branch() {
        // y² − x²(1+x) = y² − x² − x³ = 0  ⇒  y = ± x·√(1+x)
        //   = ±(x + x²/2 − x³/8 + …).  Ramification 1 (integer powers).
        let f = [(0, 2, r(1)), (2, 0, r(-1)), (3, 0, r(-1))];
        let br = puiseux_at_zero(&f, 4);
        assert_eq!(br.len(), 2);
        // The +branch: x + ½x² − ⅛x³.
        let plus = br
            .iter()
            .find(|s| has_term(s, r(1), r(1)))
            .expect("a +x branch");
        assert_eq!(plus.ramification, 1);
        assert!(has_term(plus, r(2), rr(1, 2)), "x² coeff ½: {plus:?}");
        assert!(has_term(plus, r(3), rr(-1, 8)), "x³ coeff −⅛: {plus:?}");
    }

    #[test]
    fn nonzero_constant_branch() {
        // (y−1)(y−x) = y² − (1+x)y + x = 0 ⇒ branches y = 1 + … and y = x + ….
        // y² − y − xy + x.
        let f = [(0, 2, r(1)), (0, 1, r(-1)), (1, 1, r(-1)), (1, 0, r(1))];
        let br = puiseux_at_zero(&f, 3);
        // Expect a branch with constant term 1 and a branch with leading x.
        assert!(br.iter().any(|s| has_term(s, r(0), r(1))));
        assert!(br.iter().any(|s| has_term(s, r(1), r(1))));
    }

    // -----------------------------------------------------------------------
    // Puiseux at an ALGEBRAIC base point  x = α
    // -----------------------------------------------------------------------

    /// Multiply two `K`-polys in `t` (ascending), truncated to degree `< n`.
    fn kt_mul_trunc(nf: &NumberField, a: &[KElem], b: &[KElem], n: usize) -> Vec<KElem> {
        let mut r = vec![NumberField::k_zero(); n];
        for (i, ca) in a.iter().enumerate() {
            if i >= n || NumberField::is_zero(ca) {
                continue;
            }
            for (j, cb) in b.iter().enumerate() {
                if i + j >= n {
                    break;
                }
                let p = nf.mul(ca, cb);
                r[i + j] = nf.add(&r[i + j], &p);
            }
        }
        r
    }

    /// `p^e` of a `K`-poly in `t`, truncated to degree `< n`.
    fn kt_pow_trunc(nf: &NumberField, p: &[KElem], e: u32, n: usize) -> Vec<KElem> {
        let mut acc = vec![nf.reduce(&vec![r(1)])];
        for _ in 0..e {
            acc = kt_mul_trunc(nf, &acc, p, n);
        }
        acc
    }

    /// EXACT back-substitution check of an algebraic-base branch: build
    /// `F(α + t^e, y(t))` as a `K`-poly in `t` and assert it `≡ 0 (mod t^N)`,
    /// where `N = e · prec`.  All arithmetic is exact in `K = ℚ(α)[t]`.
    fn verify_alg_branch(
        coeffs: &[(u32, u32, Rational)],
        alpha_minpoly: &[Rational],
        s: &AlgBasePuiseuxSeries,
        prec: u32,
    ) {
        let nf = NumberField::new(alpha_minpoly.to_vec());
        let alpha = nf.reduce(&vec![r(0), r(1)]); // α = t-generator
        let e = s.ramification;
        // Truncate to t-degree < N (the branch is known to relative order prec
        // in (x−α), i.e. t-order N = e·prec).
        let n = (e * prec as u64) as usize + 1;

        // x = α + t^e  as a K-poly in t.
        let mut xpoly = vec![NumberField::k_zero(); e as usize + 1];
        xpoly[0] = alpha.clone();
        xpoly[e as usize] = nf.reduce(&vec![r(1)]);

        // y(t) = Σ c_k t^{(num/den)·e}; each exponent·e is an integer ≤ N.
        let mut ypoly = vec![NumberField::k_zero(); n];
        for (exp, c) in &s.terms {
            let te = exp.clone() * Rational::from(e as i64);
            assert!(*te.denom() == 1, "exponent·e must be integral: {te}");
            let idx = te.numer().to_i64().unwrap();
            assert!(idx >= 0);
            let idx = idx as usize;
            if idx < n {
                ypoly[idx] = nf.add(&ypoly[idx], c);
            }
        }

        // F(x, y) = Σ a_ij x^i y^j, truncated to t-degree < N.
        let mut fpoly = vec![NumberField::k_zero(); n];
        for (i, j, a) in coeffs {
            if *a == 0 {
                continue;
            }
            let xi = kt_pow_trunc(&nf, &xpoly, *i, n);
            let yj = kt_pow_trunc(&nf, &ypoly, *j, n);
            let term = kt_mul_trunc(&nf, &xi, &yj, n);
            let ak = nf.reduce(&vec![a.clone()]);
            for (idx, tc) in term.iter().enumerate() {
                let scaled = nf.mul(&ak, tc);
                fpoly[idx] = nf.add(&fpoly[idx], &scaled);
            }
        }

        for (idx, c) in fpoly.iter().enumerate() {
            assert!(
                NumberField::is_zero(c),
                "alg branch {s:?}: F(α+t^{e}, y)[t^{idx}] = {c:?} ≠ 0 (not ≡0 mod t^{n})"
            );
        }
    }

    #[test]
    fn alg_base_degenerate_matches_rational() {
        // Algebraic path with a *rational* α (minpoly t − 2) must agree with the
        // existing rational `puiseux_at`.  Use  y² − (x−2) = y² − x + 2, whose
        // branch at α = 2 is the ramified, *rational*-coefficient y = ±(x−2)^{1/2}.
        let f = [(0, 2, r(1)), (1, 0, r(-1)), (0, 0, r(2))]; // y² − x + 2
        let alpha_minpoly = vec![r(-2), r(1)]; // t − 2  ⇒  α = 2
        let (br, skipped) = puiseux_at_algebraic(&f, &alpha_minpoly, 3);
        assert_eq!(skipped, 0);
        // Same as the rational expansion at α = 2: y = ±(x−2)^{1/2}.
        let rat = puiseux_at(&f, &r(2), 3);
        assert_eq!(br.len(), rat.len());
        assert_eq!(br.len(), 2);
        for s in &br {
            assert_eq!(s.conjugates, 1);
            assert_eq!(s.ramification, 2);
            // Leading exponent 1/2, coefficient ±1 (a constant in ℚ ⊆ K).
            assert_eq!(s.terms[0].0, rr(1, 2));
            let lead = &s.terms[0].1;
            assert!(lead == &vec![r(1)] || lead == &vec![r(-1)]);
            verify_alg_branch(&f, &alpha_minpoly, s, 3);
        }
    }

    #[test]
    fn alg_base_rational_branches_at_sqrt2() {
        // F = (y−x)(y−x²) = y² − (x+x²)·y + x³ has the two RATIONAL branches
        // y = x and y = x²; both stay in any base field, including ℚ(√2).  At the
        // (smooth) algebraic place x = √2 the branches are
        //   y = √2 + (x−√2) + …         (from y = x)
        //   y = 2 + 2√2·(x−√2) + …       (from y = x²)
        // — coefficients genuinely in ℚ(√2).  All recovered, none skipped.
        // y² − xy − x²y + x³:
        let f = [(0, 2, r(1)), (1, 1, r(-1)), (2, 1, r(-1)), (3, 0, r(1))];
        let mp = vec![r(-2), r(0), r(1)]; // t² − 2  ⇒  α = √2
        let (br, skipped) = puiseux_at_algebraic(&f, &mp, 4);
        assert_eq!(skipped, 0, "branches: {br:?}");
        assert_eq!(br.len(), 2, "branches: {br:?}");
        let total: usize = br.iter().map(|s| s.conjugates).sum();
        assert_eq!(total, 4); // 2 classes × conjugates-2 = 4 concrete sheets
        for s in &br {
            assert_eq!(s.conjugates, 2);
            assert_eq!(s.ramification, 1); // unramified place
            verify_alg_branch(&f, &mp, s, 4);
        }
        // One class has constant term √2 (= [0,1]); the other has 2 (= [2]).
        let consts: Vec<KElem> = br
            .iter()
            .map(|s| {
                s.terms
                    .iter()
                    .find(|(e, _)| *e == r(0))
                    .map(|(_, c)| c.clone())
                    .unwrap_or_default()
            })
            .collect();
        assert!(
            consts.iter().any(|c| *c == vec![r(0), r(1)]),
            "√2 const: {consts:?}"
        );
        assert!(
            consts.iter().any(|c| *c == vec![r(2)]),
            "2 const: {consts:?}"
        );
    }

    #[test]
    fn alg_base_constant_algebraic_node_in_field() {
        // Exercise the in-field constant-root path on a curve whose y-fibre over
        // α = √2 factors inside ℚ(√2).  F = y² + x²·y − 2y = y·(y + x² − 2):
        // F(√2, y) = y·(y + 0) = y², so the constant root c₀ = 0 ∈ ℚ(√2); the
        // second branch is the global rational y = 2 − x² = −(x−√2)(x+√2), whose
        // value and slope at √2 are ℚ(√2)-rational.  Nothing skipped.
        let f = [(0, 2, r(1)), (2, 1, r(1)), (0, 1, r(-2))]; // y² + x²y − 2y
        let mp = vec![r(-2), r(0), r(1)]; // √2
        let (br, skipped) = puiseux_at_algebraic(&f, &mp, 4);
        assert_eq!(skipped, 0, "branches: {br:?}");
        assert!(!br.is_empty(), "branches: {br:?}");
        for s in &br {
            assert_eq!(s.conjugates, 2);
            assert_eq!(s.ramification, 1); // both branches unramified at √2
            verify_alg_branch(&f, &mp, s, 4);
        }
    }

    /// Numeric back-substitution check for a **tower** branch (coefficients in a
    /// compositum `K' = ℚ[t']/coeff_minpoly` strictly larger than the base
    /// `K = ℚ(α)`).  For each numeric root `θ'₀` of `coeff_minpoly` there is an
    /// induced base point `α₀` (a root of `alpha_minpoly`); the branch must satisfy
    /// `F(α₀ + δ, y) = O(δ^prec)` for *some* such `α₀` and small real `δ` (we test a
    /// real `θ'₀` to keep the check real-valued).
    fn verify_alg_branch_tower(
        coeffs: &[(u32, u32, Rational)],
        alpha_minpoly: &[Rational],
        s: &AlgBasePuiseuxSeries,
        prec: u32,
    ) {
        // Real roots of coeff_minpoly (coarse bisection-free scan + refine).
        let eval = |p: &[Rational], x: f64| -> f64 {
            p.iter().rev().fold(0.0, |acc, c| acc * x + c.to_f64())
        };
        let mut tprimes = Vec::new();
        let (lo, hi, steps) = (-6.0_f64, 6.0_f64, 24000);
        let dx = (hi - lo) / steps as f64;
        let mut prev = eval(&s.coeff_minpoly, lo);
        for k in 1..=steps {
            let x = lo + dx * k as f64;
            let cur = eval(&s.coeff_minpoly, x);
            if prev == 0.0 || (prev < 0.0) != (cur < 0.0) {
                // Refine by a few bisection steps.
                let (mut a, mut b) = (x - dx, x);
                for _ in 0..60 {
                    let mid = 0.5 * (a + b);
                    if (eval(&s.coeff_minpoly, a) < 0.0) != (eval(&s.coeff_minpoly, mid) < 0.0) {
                        b = mid;
                    } else {
                        a = mid;
                    }
                }
                tprimes.push(0.5 * (a + b));
            }
            prev = cur;
        }
        assert!(
            !tprimes.is_empty(),
            "coeff_minpoly should have a real root for the numeric check: {s:?}"
        );
        let alpha_roots: Vec<f64> = {
            // Real roots of alpha_minpoly by the same scan.
            let mut rts = Vec::new();
            let mut prev = eval(alpha_minpoly, lo);
            for k in 1..=steps {
                let x = lo + dx * k as f64;
                let cur = eval(alpha_minpoly, x);
                if prev == 0.0 || (prev < 0.0) != (cur < 0.0) {
                    let (mut a, mut b) = (x - dx, x);
                    for _ in 0..60 {
                        let mid = 0.5 * (a + b);
                        if (eval(alpha_minpoly, a) < 0.0) != (eval(alpha_minpoly, mid) < 0.0) {
                            b = mid;
                        } else {
                            a = mid;
                        }
                    }
                    rts.push(0.5 * (a + b));
                }
                prev = cur;
            }
            rts
        };
        for tp in &tprimes {
            // Try each numeric α₀; the branch must vanish for the consistent one.
            let mut best = f64::INFINITY;
            for a0 in &alpha_roots {
                for &delta in &[0.002_f64, 0.005, 0.011] {
                    let x = a0 + delta;
                    let y: f64 = s
                        .terms
                        .iter()
                        .map(|(exp, c)| eval(c, *tp) * delta.powf(exp.to_f64()))
                        .sum();
                    let fval: f64 = coeffs
                        .iter()
                        .map(|(i, j, a)| a.to_f64() * x.powi(*i as i32) * y.powi(*j as i32))
                        .sum();
                    let tol = 1e-6 + 200.0 * delta.powf(prec as f64);
                    best = best.min(fval.abs() / tol);
                }
            }
            assert!(
                best < 1.0,
                "tower branch {s:?} at θ'₀={tp}: F not O(δ^{prec}) for any α₀ ({alpha_roots:?})"
            );
        }
    }

    #[test]
    fn alg_base_ramified_tower_collapse_sqrt2_to_quartic() {
        // F = y² − (x²−2) at α=√2.  Shift x→x+√2: x²−2 = x² + 2√2·x, so near the
        // place F = y² − 2√2·x − x²: a ramified branch y = c·x^{1/2}+… with leading
        // coefficient c = ±(2√2)^{1/2} = ±2^{3/4} ∈ ℚ(2^{3/4}) ⊋ ℚ(√2).  The tower
        // collapse builds the degree-4 compositum K' = ℚ(√2)(2^{3/4}) = ℚ(2^{1/4})
        // and RETURNS the branch (previously skip-but-counted).
        let f = [(0, 2, r(1)), (2, 0, r(-1)), (0, 0, r(2))]; // y² − x² + 2
        let mp = vec![r(-2), r(0), r(1)]; // √2
        let (br, skipped) = puiseux_at_algebraic(&f, &mp, 3);
        assert_eq!(skipped, 0, "tower closes — nothing skipped: {br:?}");
        // The ramified x^{1/2} branch is now present, over a degree-4 field.
        let ram: Vec<&AlgBasePuiseuxSeries> = br
            .iter()
            .filter(|s| s.terms.iter().any(|(e, _)| *e == rr(1, 2)))
            .collect();
        assert!(!ram.is_empty(), "ramified branch must be returned: {br:?}");
        for s in &ram {
            assert_eq!(s.ramification, 2);
            // Compositum is a proper extension of ℚ(√2): degree 4 = [ℚ(√2):ℚ]·2.
            assert_ne!(s.coeff_minpoly, mp, "coeff field must extend ℚ(√2)");
            assert_eq!(s.coeff_minpoly.len() - 1, 4, "deg K' = 4: {s:?}");
            assert_eq!(s.conjugates, 4);
            verify_alg_branch_tower(&f, &mp, s, 3);
        }
    }

    #[test]
    fn alg_base_node_tower_collapse_sqrt2() {
        // F = y² − (x²−2)²·(x+1) has an irrational double point at x = ±√2.
        // F = y² − x⁵ − x⁴ + 4x³ + 4x² − 4x − 4.  At α=√2 the node has two branches
        // y ≈ ±x·√(8(√2+1)) with √(8(√2+1)) ∉ ℚ(√2): the tower collapse closes them
        // over the degree-4 compositum (previously skip-but-counted).
        let f = [
            (0, 2, r(1)),
            (5, 0, r(-1)),
            (4, 0, r(-1)),
            (3, 0, r(4)),
            (2, 0, r(4)),
            (1, 0, r(-4)),
            (0, 0, r(-4)),
        ];
        let mp = vec![r(-2), r(0), r(1)]; // √2
        let (br, skipped) = puiseux_at_algebraic(&f, &mp, 3);
        assert_eq!(skipped, 0, "tower closes — nothing skipped: {br:?}");
        // The two escaping node sheets are returned over a proper extension of ℚ(√2).
        let tower: Vec<&AlgBasePuiseuxSeries> =
            br.iter().filter(|s| s.coeff_minpoly != mp).collect();
        assert!(!tower.is_empty(), "node tower branches returned: {br:?}");
        for s in &tower {
            assert_eq!(s.coeff_minpoly.len() - 1, 4, "deg K' = 4: {s:?}");
            assert_eq!(s.conjugates, 4);
            verify_alg_branch_tower(&f, &mp, s, 3);
        }
    }

    #[test]
    fn alg_base_constant_tower_sqrt2_over_sqrt3() {
        // F = y² − 2 at the algebraic BASE point α = √3 (minpoly t² − 3).  The
        // y-fibre F(√3, y) = y² − 2 is irreducible over K = ℚ(√3) (2 is not a
        // square there), so the constant root √2 escapes ℚ(√3): the tower collapse
        // builds the compositum ℚ(√3, √2) (degree 4) and RETURNS the branch.
        let f = [(0, 2, r(1)), (0, 0, r(-2))]; // y² − 2
        let mp = vec![r(-3), r(0), r(1)]; // √3
        let (br, skipped) = puiseux_at_algebraic(&f, &mp, 2);
        assert_eq!(skipped, 0, "tower closes: {br:?}");
        assert_eq!(
            br.len(),
            1,
            "one class (±√2 are conjugate over ℚ(√3)): {br:?}"
        );
        let s = &br[0];
        assert_ne!(s.coeff_minpoly, mp, "coeff field must extend ℚ(√3)");
        assert_eq!(s.coeff_minpoly.len() - 1, 4, "deg ℚ(√3,√2) = 4: {s:?}");
        assert_eq!(s.conjugates, 4);
        // Constant branch: a single (exponent 0) term, the value √2 in the
        // compositum; squares to 2.  Verify numerically.
        verify_alg_branch_tower(&f, &mp, s, 2);
    }

    #[test]
    fn zero_genuine_two_level_tower_y4_minus_2x2() {
        // F = y⁴ − 2x² at x = 0.  Newton edge slope 1/2: characteristic y⁴ = 2x²,
        // i.e. y² = ±√2·x  (first extension ℚ(√2)); then y = ±2^{1/4}·√(±x)
        // (second extension ℚ(2^{1/4})).  A GENUINE tower over ℚ — the deeper
        // characteristic is non-linear over ℚ(√2) and collapses to ℚ(2^{1/4}).
        let f = [(0, 4, r(1)), (2, 0, r(-2))]; // y⁴ − 2x²
        let br = puiseux_at_zero_algebraic(&f, 2);
        // Total sheets = deg_y = 4.
        let total: usize = br.iter().map(|s| s.conjugates).sum();
        assert_eq!(total, 4, "all four sheets recovered: {br:?}");
        // Every returned class has a leading x^{1/2} term and lives over a field of
        // degree ≥ 2 (ℚ(2^{1/4}) for the genuine tower sheets).
        for s in &br {
            assert!(
                s.terms.iter().any(|(e, _)| *e == rr(1, 2)),
                "leading x^{{1/2}}: {s:?}"
            );
        }
        // At least one class is over the degree-4 compositum ℚ(2^{1/4}).
        assert!(
            br.iter().any(|s| s
                .minpoly
                .as_ref()
                .map(|m| m.len() - 1 == 4)
                .unwrap_or(false)),
            "a degree-4 tower class expected: {br:?}"
        );
    }

    #[test]
    fn alg_base_conjugate_count_bookkeeping() {
        // Degree-3 base field ℚ(α), α = ∛2 (minpoly t³−2).  The smooth rational
        // branch y = x of F = (y−x)(y−x²) = y² − (x+x²)y + x³ at x = ∛2 is a
        // single class standing for conjugates = 3 concrete branches (one per
        // conjugate of ∛2).  Summed conjugates over the two classes = 6 = 2·3.
        let f = [(0, 2, r(1)), (1, 1, r(-1)), (2, 1, r(-1)), (3, 0, r(1))];
        let mp = vec![r(-2), r(0), r(0), r(1)]; // t³ − 2  ⇒  α = ∛2, degree 3
        let (br, skipped) = puiseux_at_algebraic(&f, &mp, 3);
        assert_eq!(skipped, 0, "branches: {br:?}");
        let total: usize = br.iter().map(|s| s.conjugates).sum();
        assert_eq!(total, 6, "two classes × 3 conjugates: {br:?}");
        for s in &br {
            assert_eq!(s.conjugates, 3);
            verify_alg_branch(&f, &mp, s, 3);
        }
    }
}
