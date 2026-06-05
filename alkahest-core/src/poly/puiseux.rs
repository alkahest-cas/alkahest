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
use crate::integrate::risch::number_field::NumberField;

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
            // Algebraic constant c₀ = θ over K = ℚ[t]/(fac).
            let nf = NumberField::new(fac.clone());
            let theta = nf.reduce(&vec![Rational::from(0), Rational::from(1)]);
            let gk = substitute_k(&nf, &embed(&nf, &f), &rzero(), &theta);
            for (terms, exact) in lift_k(&nf, &gk, &prec_r, 0) {
                let mut full = vec![(rzero(), theta.clone())];
                full.extend(terms);
                out.push(make_alg_series(
                    Some(fac.clone()),
                    deg,
                    full,
                    exact,
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
/// continue over it via [`lift_k`].
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
                for (sub, exact) in lift_k(&nf, &gk, &(prec.clone() - &q), 0) {
                    let mut full = embed_prefix(&nf, prefix);
                    full.push((q.clone(), theta.clone()));
                    for (gamma, b) in sub {
                        full.push((q.clone() + &gamma, b));
                    }
                    out.push(make_alg_series(Some(fac.clone()), deg, full, exact, prec));
                }
            }
        }
    }
    out
}

/// `lift` over a fixed number field `nf`: find the `w → 0` branches of `g(x,w)=0`,
/// taking only roots that lie in `nf` (linear characteristic factors — sufficient
/// for smooth radical continuations; non-linear cases are skipped).
fn lift_k(
    nf: &NumberField,
    g: &KBi,
    prec: &Rational,
    depth: u32,
) -> Vec<(Vec<(Rational, KElem)>, bool)> {
    const MAX_DEPTH: u32 = 48;
    let mut g = g.clone();
    g.retain(|_, a| !NumberField::is_zero(a));
    if g.is_empty() {
        return vec![(Vec::new(), true)];
    }
    if depth > MAX_DEPTH {
        return vec![(Vec::new(), false)];
    }
    let mut result = Vec::new();
    let m0 = g.keys().map(|(_, j)| *j).min().unwrap_or(0);
    if m0 > 0 {
        result.push((Vec::new(), true));
        g = g
            .into_iter()
            .map(|((xe, ye), a)| ((xe, ye - m0), a))
            .collect();
    }
    let keys: Vec<(Rational, u32)> = g.keys().cloned().collect();
    for (q, on_edge) in newton_edges_keys(&keys) {
        // Characteristic polynomial over nf, indexed by y-exponent.
        let mut phi: BTreeMap<u32, KElem> = BTreeMap::new();
        for k in &on_edge {
            let e = phi.entry(k.1).or_default();
            *e = nf.add(e, &g[k]);
        }
        let Some(c) = k_linear_root(nf, &phi) else {
            continue; // non-linear over nf: would need a further extension
        };
        if prec.clone() - &q <= 0 {
            result.push((vec![(q.clone(), c)], false));
            continue;
        }
        let gk = substitute_k(nf, &g, &q, &c);
        for (sub, exact) in lift_k(nf, &gk, &(prec.clone() - &q), depth + 1) {
            let mut terms = vec![(q.clone(), c.clone())];
            for (gamma, b) in sub {
                terms.push((q.clone() + &gamma, b));
            }
            result.push((terms, exact));
        }
    }
    result
}

/// The unique root in `nf` of a characteristic polynomial `Σ_j a_j c^j` that is a
/// **binomial of consecutive degrees** `a_{j} c^{j} + a_{j+1} c^{j+1}` (root
/// `−a_j/a_{j+1}`).  `None` otherwise (no nonzero root, or degree ≥ 2 ⇒ would need
/// a further extension).
fn k_linear_root(nf: &NumberField, phi: &BTreeMap<u32, KElem>) -> Option<KElem> {
    let nz: Vec<(&u32, &KElem)> = phi
        .iter()
        .filter(|(_, c)| !NumberField::is_zero(c))
        .collect();
    if nz.len() != 2 {
        return None;
    }
    let (j1, a1) = nz[0];
    let (j2, a2) = nz[1];
    if *j2 != *j1 + 1 {
        return None; // c^{j2−j1} = … : a higher root, not in nf in general
    }
    let inv = nf.inv(a2)?;
    Some(nf.neg(&nf.mul(a1, &inv)))
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
fn factor_over_q(p: &[Rational]) -> Vec<(Vec<Rational>, usize)> {
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
}
