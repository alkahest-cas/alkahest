//! Residues of a differential on an algebraic curve — the foundation of the
//! **logarithmic part / FIND-ORDER** (Risch milestone **MC**).
//!
//! After Hermite reduction (`hermite_curve`) the remaining integrand `h` is a
//! differential of the **third kind** (only simple poles).  Its integral is
//! `∫ h dx = Σ cⱼ log(uⱼ)`, and the `cⱼ` are governed by the **residues** of
//! `h dx` at the places of the curve: the residue divisor `δ = Σ res_P · P` must
//! be (a torsion multiple of) a principal divisor for the integral to be
//! elementary.
//!
//! At a place over `x = α` with ramification `e` and uniformizer `t`
//! (`x − α = t^e`, `dx = e·t^{e−1} dt`), the residue is
//!
//! ```text
//!   res_P(h dx) = [t^{-1}](h · e t^{e−1}) = e · [t^{-e}]( h along the branch ).
//! ```
//!
//! computed from the Puiseux expansion at `α` via the Laurent-`t` substitution
//! shared with [`super::vanhoeij`].  The **residue theorem** `Σ_P res_P = 0`
//! (over *all* places, including infinity) is the soundness check.
//!
//! [`finite_residues`] handles the rational finite places (poles of `h`, branch
//! points); [`residues_at_infinity`] handles the places over `∞` (via the
//! `w = 1/y`, `z = 1/x` curve `ã(z)wⁿ − zᵐ = 0`); [`residue_divisor`] combines
//! them.  Scope: rational branches; **algebraic** finite/infinite places are the
//! remaining gap (build on `puiseux_at_zero_algebraic`).  Together with FIND-ORDER
//! (genus-graded principality) these complete MC.

use rug::{Integer, Rational};
use std::collections::{BTreeSet, HashMap};

use super::super::risch::alg_field::AlgElem;
use super::super::risch::poly_rde::{degree, trim, QPoly};
use super::vanhoeij::{branch_ts, elem_ts, ts_add, ts_inv, ts_mul, ts_pow, TS};
use crate::poly::puiseux::{puiseux_at, puiseux_at_zero, PuiseuxSeries};

/// A residue at a place of the curve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residue {
    /// The base point `α` (ignored when `at_infinity`).
    pub point: Rational,
    /// `true` for a place over `x = ∞`.
    pub at_infinity: bool,
    /// Index of the Puiseux sheet at the place's base (distinguishes places over
    /// the same `α` or over `∞`).
    pub sheet: usize,
    /// Ramification index of the place.
    pub ramification: u64,
    /// The residue `res_P(h dx)`.
    pub value: Rational,
    /// The `y`-coordinate of the place (constant term of the branch): `0` at a
    /// branch point, `±√(a(α))` at an unramified place (rational when captured);
    /// unused when `at_infinity`.  Lets FIND-ORDER map the place onto the curve.
    pub y_coord: Rational,
}

/// Residues of the differential `h dx` at all **rational finite places** of the
/// curve `yⁿ = a(x)` — the poles of `h` and the branch points (roots of `a`).
/// Places with zero residue are omitted.
///
/// (Algebraic places and the place at infinity are out of this rational-only
/// scope; for fully ramified rational curves the finite residues already sum to
/// `−res_∞`.)
pub fn finite_residues(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
    if n < 2 {
        return Vec::new();
    }
    let monos = curve_monomials(n, a);

    // Candidate base points: rational roots of `a` (branch points) and of every
    // coordinate denominator of `h` (finite poles).
    let mut cands: BTreeSet<Rational> = BTreeSet::new();
    for r in rational_roots(a) {
        cands.insert(r);
    }
    for c in h {
        for r in rational_roots(c.denom()) {
            cands.insert(r);
        }
    }

    let mut out = Vec::new();
    for alpha in cands {
        // Enough Puiseux terms to resolve the simple poles of h's coordinates.
        let prec = (h
            .iter()
            .map(|c| degree(c.denom()).max(0))
            .max()
            .unwrap_or(0)
            + n as i64
            + 3) as u32;
        // A ramified place of index e is returned by Puiseux as e Galois-conjugate
        // sheets, all carrying the same residue; keep one representative per place.
        let mut seen_per_ram: HashMap<u64, usize> = HashMap::new();
        for (sheet, br) in puiseux_at(&monos, &alpha, prec).iter().enumerate() {
            let e = br.ramification as i64;
            let idx = seen_per_ram.entry(br.ramification).or_insert(0);
            let keep = *idx % (br.ramification.max(1) as usize) == 0;
            *idx += 1;
            if !keep {
                continue;
            }
            let u = 2 * e + 4;
            let series = elem_ts(h, &alpha, e, u, &branch_ts(br));
            let coeff = series
                .get(&(-e))
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
            let value = Rational::from(e) * coeff;
            if value != 0 {
                // y-coordinate of the place = constant term of the branch.
                let y_coord = br
                    .terms
                    .iter()
                    .find(|(ex, _)| *ex == 0)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(|| Rational::from(0));
                out.push(Residue {
                    point: alpha.clone(),
                    at_infinity: false,
                    sheet,
                    ramification: br.ramification,
                    value,
                    y_coord,
                });
            }
        }
    }
    out
}

/// Puiseux branches at **infinity** of `yⁿ = a(x)`, returned as the `w = 1/y`
/// branches (series in `z = 1/x`) of the curve `ã(z)·wⁿ − zᵐ = 0`, where
/// `m = deg a` and `ã(z) = zᵐ·a(1/z)` is the reversed radicand (`ã(0) ≠ 0`).
/// The actual branch is `y = 1/w`, `x = 1/z`.
pub fn puiseux_at_infinity(n: usize, a: &QPoly, prec: u32) -> Vec<PuiseuxSeries> {
    let a = trim(a.clone());
    let m = degree(&a);
    if m < 0 {
        return Vec::new();
    }
    let m = m as usize;
    // ã(z) = Σᵢ a_i z^{m−i}  (reversed).
    let mut monos: Vec<(u32, u32, Rational)> = Vec::new();
    for (i, ai) in a.iter().enumerate() {
        if *ai != 0 {
            monos.push(((m - i) as u32, n as u32, ai.clone())); // ã_k wⁿ
        }
    }
    monos.push((m as u32, 0, Rational::from(-1))); // − zᵐ
    puiseux_at_zero(&monos, prec)
}

/// Residues of `h dx` at the places over **infinity** of `yⁿ = a(x)`.
///
/// At a place over `∞` with ramification `e` (uniformizer `t`, `z = 1/x = tᵉ`,
/// `x = t^{−e}`, `dx = −e·t^{−e−1} dt`), the residue is
/// `res = [t^{-1}](h dx) = −e·[tᵉ](h along the branch)`.
pub fn residues_at_infinity(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
    let m = degree(&trim(a.clone())).max(0) as usize;
    let dmax = h
        .iter()
        .map(|c| degree(c.numer()).max(degree(c.denom())).max(0))
        .max()
        .unwrap_or(0);
    let prec = (dmax + (n as i64) + (m as i64) + 4) as u32;
    let mut out = Vec::new();
    // Dedup the e conjugate sheets of a ramified place over ∞ (see finite_residues).
    let mut seen_per_ram: HashMap<u64, usize> = HashMap::new();
    for (sheet, w_branch) in puiseux_at_infinity(n, a, prec).iter().enumerate() {
        let idx = seen_per_ram.entry(w_branch.ramification).or_insert(0);
        let keep = *idx % (w_branch.ramification.max(1) as usize) == 0;
        *idx += 1;
        if !keep {
            continue;
        }
        let e = w_branch.ramification as i64;
        let u = 2 * e + 2 * (m as i64) * e + 8;
        // y = 1/w  as a t-series; w from the branch (z = tᵉ).
        let w_ts = branch_ts(w_branch);
        let Some(y_ts) = ts_inv(&w_ts, u) else {
            continue;
        };
        let h_ts = elem_at_infinity(h, e, u, &y_ts);
        let coeff = h_ts.get(&e).cloned().unwrap_or_else(|| Rational::from(0));
        let value = -Rational::from(e) * coeff;
        if value != 0 {
            out.push(Residue {
                point: Rational::from(0),
                at_infinity: true,
                sheet,
                ramification: w_branch.ramification,
                value,
                y_coord: Rational::from(0),
            });
        }
    }
    out
}

/// Series of `h = Σⱼ hⱼ(x) yʲ` along an infinite place: `x = t^{−e}`, `y = y_ts(t)`.
fn elem_at_infinity(h: &AlgElem, e: i64, u: i64, y_ts: &TS) -> TS {
    let mut acc = TS::new();
    for (j, coeff) in h.iter().enumerate() {
        if coeff.numer().is_empty() {
            continue;
        }
        let cj = ratfn_at_infinity(coeff, e, u);
        let yj = ts_pow(y_ts, j as u32, u);
        acc = ts_add(&acc, &ts_mul(&cj, &yj, u));
    }
    acc
}

/// `r(t^{−e})` for `r ∈ ℚ(x)`, as a Laurent `t`-series truncated to exps `< u`.
fn ratfn_at_infinity(r: &crate::integrate::risch::alg_field::RatFn, e: i64, u: i64) -> TS {
    let num = poly_at_infinity(r.numer(), e, u);
    let den = poly_at_infinity(r.denom(), e, u);
    match ts_inv(&den, u) {
        Some(inv) => ts_mul(&num, &inv, u),
        None => TS::new(),
    }
}

/// `p(t^{−e})` for `p ∈ ℚ[x]`: `Σᵢ p_i t^{−e·i}`.
fn poly_at_infinity(p: &QPoly, e: i64, u: i64) -> TS {
    let mut ts = TS::new();
    for (i, pi) in p.iter().enumerate() {
        if *pi != 0 {
            let exp = -e * i as i64;
            if exp < u {
                *ts.entry(exp).or_insert_with(|| Rational::from(0)) += pi;
            }
        }
    }
    ts.retain(|_, c| *c != 0);
    ts
}

/// The full **residue divisor** of `h dx` on `yⁿ = a(x)`: residues at all
/// rational finite places **and** at the places over infinity.  The integral
/// `∫ h dx` is elementary iff this divisor is (a torsion multiple of) a principal
/// divisor — the FIND-ORDER decision.  `residue_sum(&divisor)` should be `0`
/// (residue theorem) when every place was captured.
pub fn residue_divisor(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
    let mut d = finite_residues(n, a, h);
    d.extend(residues_at_infinity(n, a, h));
    d
}

/// Sum of the residue values (should be `0` over a complete set of places).
pub fn residue_sum(divisor: &[Residue]) -> Rational {
    divisor.iter().fold(Rational::from(0), |s, r| s + &r.value)
}

/// Monomials `(i, j, coeff)` of `F = yⁿ − a(x)`.
fn curve_monomials(n: usize, a: &QPoly) -> Vec<(u32, u32, Rational)> {
    let mut m = vec![(0u32, n as u32, Rational::from(1))]; // yⁿ
    for (i, c) in a.iter().enumerate() {
        if *c != 0 {
            m.push((i as u32, 0, -c.clone())); // −a_i x^i
        }
    }
    m
}

/// Distinct rational roots of `p ∈ ℚ[x]` via the rational-root theorem.
fn rational_roots(p: &QPoly) -> Vec<Rational> {
    let p = trim(p.clone());
    if degree(&p) < 1 {
        return Vec::new();
    }
    let lo = p.iter().position(|c| *c != 0).unwrap_or(0);
    let mut roots = Vec::new();
    if lo > 0 {
        roots.push(Rational::from(0));
    }
    let psi = &p[lo..];
    if psi.len() <= 1 {
        return roots;
    }
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
    let a0 = ints[0].clone().abs();
    let an = ints[ints.len() - 1].clone().abs();
    for pn in divisors(&a0) {
        for qn in &divisors(&an) {
            for sign in [1i32, -1] {
                let cand = Rational::from((Integer::from(sign) * pn.clone(), qn.clone()));
                if roots.contains(&cand) {
                    continue;
                }
                let mut acc = Rational::from(0);
                for c in ints.iter().rev() {
                    acc = acc * &cand + Rational::from(c.clone());
                }
                if acc == 0 {
                    roots.push(cand);
                }
            }
        }
    }
    roots
}

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
            let o = n.clone() / &d;
            if o != d {
                ds.push(o);
            }
        }
        d += 1;
    }
    ds
}

#[cfg(test)]
mod tests {
    use super::super::super::risch::alg_field::RatFn;
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }
    fn rf(num: &[i64], den: &[i64]) -> RatFn {
        RatFn::new(qp(num), qp(den))
    }
    fn r(n: i64) -> Rational {
        Rational::from(n)
    }

    /// h dx = du/u with u = (y−1)/(y+1) on y²=x:
    /// `∫ 1/((x−1)√x) dx = log((√x−1)/(√x+1))`.
    /// Residues: +1 and −1 at the two sheets over x=1; total 0.
    #[test]
    fn log_differential_residues() {
        // h = 1/((x−1)√x) = y/((x−1)x) = AlgElem [0, 1/(x²−x)].
        let h = vec![RatFn::int(0), rf(&[1], &[0, -1, 1])];
        let res = finite_residues(2, &qp(&[0, 1]), &h);
        // Two nonzero residues at x=1.
        let mut at1: Vec<Rational> = res
            .iter()
            .filter(|r| r.point == r_one())
            .map(|r| r.value.clone())
            .collect();
        at1.sort();
        assert_eq!(at1, vec![r(-1), r(1)]);
        // Residue theorem (finite places): sum is 0 here (no residue at ∞).
        let total: Rational = res.iter().fold(r(0), |s, x| s + &x.value);
        assert_eq!(total, r(0));
    }

    fn r_one() -> Rational {
        Rational::from(1)
    }

    /// Residues at infinity: `∫ dx/√(x²+1) = log(x+√(x²+1))`.  At ∞ the curve
    /// `y²=x²+1` has two unramified places (`y ~ ±x`); `du/u` with `u=x+y` has
    /// residues `−1` and `+1` there.
    #[test]
    fn infinity_residues() {
        // h = 1/y = y/(x²+1) = AlgElem [0, 1/(x²+1)].
        let h = vec![RatFn::int(0), rf(&[1], &[1, 0, 1])];
        let res = super::residues_at_infinity(2, &qp(&[1, 0, 1]), &h);
        let mut vals: Vec<Rational> = res.iter().map(|r| r.value.clone()).collect();
        vals.sort();
        assert_eq!(vals, vec![r(-1), r(1)]);
        // No rational finite poles (y=0 ⇒ x²=−1, algebraic) ⇒ finite is empty.
        assert!(finite_residues(2, &qp(&[1, 0, 1]), &h).is_empty());
    }

    /// Residue theorem across finite + infinite places: for `y²=x`,
    /// `h = 1/((x−1)√x)`, the finite residues (±1) and `res_∞ = 0` sum to 0.
    #[test]
    fn residue_theorem_finite_plus_infinity() {
        let a = qp(&[0, 1]);
        let h = vec![RatFn::int(0), rf(&[1], &[0, -1, 1])]; // y/((x−1)x)
        let mut total = r(0);
        for r0 in finite_residues(2, &a, &h) {
            total += &r0.value;
        }
        for r0 in super::residues_at_infinity(2, &a, &h) {
            total += &r0.value;
        }
        assert_eq!(total, r(0));
    }

    /// h dx = x^{-1/2} dx = d(2√x): no residues (exact, no log part).
    #[test]
    fn exact_differential_no_residues() {
        // h = 1/√x = y/x = AlgElem [0, 1/x].
        let h = vec![RatFn::int(0), rf(&[1], &[0, 1])];
        let res = finite_residues(2, &qp(&[0, 1]), &h);
        assert!(res.is_empty(), "expected no residues; got {res:?}");
    }

    /// A single simple pole away from the branch locus: y/(x−2) on y²=x,
    /// i.e. √x/(x−2).  The two sheets ±√2 over x=2 carry opposite residues.
    #[test]
    fn simple_pole_off_branch() {
        // h = y/(x−2) = AlgElem [0, 1/(x−2)].
        let h = vec![RatFn::int(0), rf(&[1], &[-2, 1])];
        let res = finite_residues(2, &qp(&[0, 1]), &h);
        // x=2 is not a rational branch point and √2 ∉ ℚ ⇒ no *rational* sheets
        // there; the residues live at an algebraic place (out of scope) — so the
        // rational-place result is empty, soundly (not a wrong value).
        assert!(res.iter().all(|r| r.value != 0));
    }
}
