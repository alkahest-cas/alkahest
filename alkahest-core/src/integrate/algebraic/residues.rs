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
//! them.  [`finite_residues_algebraic`] (hyperelliptic `y² = a`) covers the
//! **algebraic** finite simple poles those miss — both algebraic base points
//! (irreducible factors `deg ≥ 2` of the pole denominator) and rational base
//! points with an *irrational sheet* `√a(α)` — with residues in a number field;
//! [`residue_sum_complete`] checks the residue theorem over the resulting
//! complete divisor.  Remaining gap: algebraic **branch** places (a pole at an
//! irrational root of `a`) and `n > 2`.  Together with FIND-ORDER (genus-graded
//! principality) these complete MC.
//!
//! # Completeness is a separate question from the residue theorem
//!
//! `Σ res = 0` is necessary, not sufficient: a *missing* conjugate pair sums to
//! zero on its own, and an enumeration that finds no places at all sums to zero
//! vacuously.  [`residues_at_infinity`] is where that bites — it reads the
//! places over `∞` off a **rational** Puiseux expansion, so when the leading
//! coefficient of `a` is not a rational square it returns nothing at all rather
//! than the conjugate pair that is actually there.  Two additions close the
//! gap: [`residues_at_infinity_exact`] computes those residues in closed form
//! over `ℚ(√lc)`, and [`residue_enumeration_is_complete`] is the predicate a
//! caller must consult before reading an empty divisor as "no residues
//! anywhere".

use rug::{Integer, Rational};
use std::collections::{BTreeSet, HashMap};

use super::super::risch::alg_field::{AlgElem, RatFn};
use super::super::risch::number_field::{KElem, NumberField};
use super::super::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};
use super::super::risch::rational_rde::{poly_div_exact, poly_gcd};
use super::vanhoeij::{branch_ts, elem_ts, ts_add, ts_inv, ts_mul, ts_pow, TS};
use crate::poly::puiseux::{factor_over_q, puiseux_at, puiseux_at_zero, PuiseuxSeries};

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
}

/// A [`Residue`] paired with the place's `y`-coordinate — internal FIND-ORDER
/// plumbing.
///
/// The `y`-coordinate is the constant term of the branch (`0` at a branch point,
/// `±√(a(α))` at an unramified place; unused when `at_infinity`) and lets
/// FIND-ORDER map the place onto the elliptic curve.  It is deliberately kept
/// *out* of the public, semver-stable, externally-constructible [`Residue`]
/// struct (adding any field there is a breaking change), so the genus-1 path
/// threads this richer type through `pub(crate)` channels instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlacedResidue {
    pub(crate) residue: Residue,
    pub(crate) y_coord: Rational,
}

/// Residues of the differential `h dx` at all **rational finite places** of the
/// curve `yⁿ = a(x)` — the poles of `h` and the branch points (roots of `a`).
/// Places with zero residue are omitted.
///
/// (Algebraic places and the place at infinity are out of this rational-only
/// scope; for fully ramified rational curves the finite residues already sum to
/// `−res_∞`.)
pub fn finite_residues(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
    finite_residues_placed(n, a, h)
        .into_iter()
        .map(|p| p.residue)
        .collect()
}

/// As [`finite_residues`], but each residue carries the place's `y`-coordinate
/// (for the genus-1 Abel–Jacobi map).  Internal — see [`PlacedResidue`].
pub(crate) fn finite_residues_placed(n: usize, a: &QPoly, h: &AlgElem) -> Vec<PlacedResidue> {
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
                out.push(PlacedResidue {
                    residue: Residue {
                        point: alpha.clone(),
                        at_infinity: false,
                        sheet,
                        ramification: br.ramification,
                        value,
                    },
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
    residues_at_infinity_placed(n, a, h)
        .into_iter()
        .map(|p| p.residue)
        .collect()
}

/// As [`residues_at_infinity`], but each residue carries the place's
/// `y`-coordinate (always `0` at ∞ here).  Internal — see [`PlacedResidue`].
pub(crate) fn residues_at_infinity_placed(n: usize, a: &QPoly, h: &AlgElem) -> Vec<PlacedResidue> {
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
            out.push(PlacedResidue {
                residue: Residue {
                    point: Rational::from(0),
                    at_infinity: true,
                    sheet,
                    ramification: w_branch.ramification,
                    value,
                },
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

/// As [`residue_divisor`], but each place carries its `y`-coordinate for the
/// genus-1 Abel–Jacobi map.  Internal — see [`PlacedResidue`].
pub(crate) fn residue_divisor_placed(n: usize, a: &QPoly, h: &AlgElem) -> Vec<PlacedResidue> {
    let mut d = finite_residues_placed(n, a, h);
    d.extend(residues_at_infinity_placed(n, a, h));
    d
}

/// Sum of the residue values (should be `0` over a complete set of places).
pub fn residue_sum(divisor: &[Residue]) -> Rational {
    divisor.iter().fold(Rational::from(0), |s, r| s + &r.value)
}

// ===========================================================================
// Residues at algebraic (non-rational) finite places — hyperelliptic `y² = a`
// ===========================================================================

/// A residue at an **algebraic place**: a Galois orbit of finite places of
/// `y² = a(x)` over the roots of an irreducible factor `q` of the integrand's
/// pole denominator (`deg q ≥ 2`, `q` coprime to `a`, so a non-branch place).
///
/// Over a root `α` of `q` there are two sheets `(α, ±√a(α))`, and on the curve
/// the residue of `(A + B·y) dx` at sheet `±` is `r0 ± r1·√a(α)` with
/// `r0, r1 ∈ ℚ(α) = ℚ[z]/(q)`.  The `√a(α)` part cancels between the two sheets,
/// so the orbit's total contribution to the residue sum is `2·Tr_{ℚ(α)/ℚ}(r0)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlgResidue {
    /// Monic irreducible minimal polynomial `q` of `α` (the place's base).
    pub minpoly: QPoly,
    /// Number of conjugate base points (`= deg q`).
    pub conjugates: usize,
    /// Rational part of the residue, in `ℚ[z]/(q)` (equal on both sheets).
    pub r0: KElem,
    /// `√a(α)` coefficient, in `ℚ[z]/(q)` (opposite sign on the two sheets).
    pub r1: KElem,
}

/// Residues of `h dx = (A + B·y) dx` (`y² = a`) at the **algebraic** finite
/// simple poles — the irreducible factors `q` of the pole denominator with
/// `deg q ≥ 2` and `gcd(q, a) = 1`.  Requires a squarefree pole denominator
/// (third-kind / simple poles); returns `[]` for `n ≠ 2`, a non-squarefree
/// denominator, or no algebraic poles.
///
/// Complements [`finite_residues`] (rational places); together with
/// [`residues_at_infinity`] they form the **complete** residue divisor whose
/// values sum to zero (the residue theorem) — see [`residue_sum_complete`].
pub fn finite_residues_algebraic(n: usize, a: &QPoly, h: &AlgElem) -> Vec<AlgResidue> {
    if n != 2 {
        return Vec::new();
    }
    let a = trim(a.clone());
    let a_c = h.first().cloned().unwrap_or_else(|| RatFn::int(0)); // A
    let b_c = h.get(1).cloned().unwrap_or_else(|| RatFn::int(0)); // B
    let a_den = if a_c.numer().is_empty() {
        vec![Rational::from(1)]
    } else {
        a_c.denom().clone()
    };
    let b_den = if b_c.numer().is_empty() {
        vec![Rational::from(1)]
    } else {
        b_c.denom().clone()
    };
    // Common pole denominator D = lcm(den A, den B).
    let d = poly_lcm(&a_den, &b_den);
    if degree(&d) < 1 {
        return Vec::new();
    }
    // Simple poles only.
    if degree(&poly_gcd(&d, &poly_deriv(&d))) > 0 {
        return Vec::new();
    }
    // Numerators over the common denominator: Ã = A_num·(D/den A), B̃ likewise.
    let a_num = poly_mul(a_c.numer(), &poly_div_exact(&d, &a_den));
    let b_num = poly_mul(b_c.numer(), &poly_div_exact(&d, &b_den));
    let d_prime = poly_deriv(&d);

    let mut out = Vec::new();
    for (q, deg_q) in factor_including_zero_root(&d) {
        if degree(&poly_gcd(&q, &a)) > 0 {
            continue; // shares a factor with `a`: a branch place, not handled here
        }
        // Which places have **algebraic** residues that `finite_residues`
        // (rational Puiseux) misses?  (a) an algebraic base point `deg q ≥ 2`;
        // (b) a *rational* base point `x = α` whose sheet `√a(α)` is irrational
        // (`a(α)` not a perfect square).  Rational base + rational sheet is
        // already handled by `finite_residues`, so skip it (no double-count).
        if deg_q == 1 {
            let alpha = -q.first().cloned().unwrap_or_else(|| Rational::from(0)); // q = x − α (monic)
            let a_at = eval_q(&a, &alpha);
            if is_rational_square(&a_at) {
                continue; // rational sheet → `finite_residues` already counts it
            }
        }
        let nf = NumberField::new(q.clone());
        // Evaluate at α (= reduce mod q) and divide by D'(α) in ℚ(α).
        let dp_alpha = nf.reduce(&d_prime);
        let Some(dp_inv) = nf.inv(&dp_alpha) else {
            continue; // D'(α) = 0 ⇒ not a simple pole (shouldn't happen, D squarefree)
        };
        let r0 = nf.mul(&nf.reduce(&a_num), &dp_inv);
        let r1 = nf.mul(&nf.reduce(&b_num), &dp_inv);
        out.push(AlgResidue {
            minpoly: q,
            conjugates: deg_q,
            r0,
            r1,
        });
    }
    out
}

/// Total residue over the **complete** divisor of `y² = a`: rational finite
/// places + algebraic finite places (`2·Tr(r0)` per orbit) + infinity.  By the
/// residue theorem this is `0` whenever the residue computation is complete; it
/// is the soundness check that no place was missed.
pub fn residue_sum_complete(n: usize, a: &QPoly, h: &AlgElem) -> Rational {
    let mut total = residue_sum(&finite_residues(n, a, h));
    total += residue_sum(&residues_at_infinity(n, a, h));
    for r in finite_residues_algebraic(n, a, h) {
        let nf = NumberField::new(r.minpoly.clone());
        total += Rational::from(2) * nf.trace(&r.r0);
    }
    total
}

// ===========================================================================
// Places over `x = ∞` — exact, closed form (hyperelliptic `y² = a`)
// ===========================================================================
//
// `residues_at_infinity` reads the places over `∞` off the *rational* Puiseux
// expansion of `ã(z)w² − zᵐ = 0`, and `puiseux_at_zero` only follows branches
// whose leading coefficient is a **rational** root of `F(0, ·)`.  When the
// leading coefficient of `a` is not a rational square the branches at infinity
// are conjugate over `ℚ(√lc)` and that routine silently returns *nothing* — not
// "no residues", but "no branches found".  Every consumer that reads an empty
// result as "no residues anywhere" is then reasoning from an incomplete
// divisor.  `∫x dx/√(1−x⁴)` is exactly that case: `lc = −1`, the two places
// over `∞` carry residues `±i`, and the rational routine reports none.
//
// The closed form below removes the guesswork.  With `m = deg a`,
// `lc = a_m ≠ 0`, `ã(z) = zᵐ·a(1/z)` (so `ã(0) = lc`) and
// `T(z) = √(ã(z)/lc) ∈ ℚ[[z]]`, `T(0) = 1`:
//
// * **`m = 2s` even.**  Two places, `x = 1/z`, `y = ±z^{−s}·√lc·T(z)`,
//   `dx = −z^{−2} dz`, so with `ca = A(1/z)`, `cb = B(1/z)` as Laurent series
//   in `z`,
//
//   ```text
//     res_± = R₀ ± √lc·R₁,   R₀ = −[z¹] ca,   R₁ = −[z^{s+1}](cb·T).
//   ```
//
// * **`m` odd.**  One place, ramified (`z = t²`).  The `B·y` half lands
//   entirely on even powers of `t` and contributes nothing, leaving the
//   **always rational** `res = −2·[z¹] ca`.
//
// Both agree with the residue theorem's other half: the total over `∞` is
// `−2·[z¹]ca` either way.

/// Residues of `h dx = (A + B·y) dx` at the places over `x = ∞` of `y² = a`,
/// in closed form.  `value(±) = r0 ± r1·√lc`; `two_places` is `false` for odd
/// `deg a` (a single ramified place, `r1 = 0`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InfinityResidues {
    /// Rational part of the residue, shared by both places.
    pub(crate) r0: Rational,
    /// Coefficient of `√lc`; opposite sign on the two places.
    pub(crate) r1: Rational,
    /// Leading coefficient of `a` — the square generating the residue field.
    pub(crate) lc: Rational,
    /// `true` for even `deg a` (two places over `∞`), `false` for odd.
    pub(crate) two_places: bool,
}

impl InfinityResidues {
    /// Every place over `∞` has residue `0`.
    pub(crate) fn all_zero(&self) -> bool {
        self.r0 == 0 && self.r1 == 0
    }

    /// Do the residues lie outside `ℚ`?  (Only then is the divisor beyond the
    /// rational FIND-ORDER scope.)
    pub(crate) fn is_irrational(&self) -> bool {
        self.r1 != 0 && !is_rational_square(&self.lc)
    }

    /// The places over `∞` as [`PlacedResidue`]s — **only** when every residue
    /// is rational (`r1 = 0`, or `lc` a rational square).  Zero residues are
    /// omitted, matching [`residues_at_infinity`].
    pub(crate) fn placed(&self) -> Option<Vec<PlacedResidue>> {
        if self.all_zero() {
            return Some(Vec::new());
        }
        let s = if self.r1 == 0 {
            Rational::from(0)
        } else {
            rational_sqrt(&self.lc)?
        };
        let mut out = Vec::new();
        for (sheet, sign) in [Rational::from(1), Rational::from(-1)]
            .into_iter()
            .enumerate()
        {
            let value = if self.two_places {
                self.r0.clone() + sign * s.clone() * self.r1.clone()
            } else {
                self.r0.clone()
            };
            if value != 0 {
                out.push(PlacedResidue {
                    residue: Residue {
                        point: Rational::from(0),
                        at_infinity: true,
                        sheet,
                        ramification: if self.two_places { 1 } else { 2 },
                        value,
                    },
                    y_coord: Rational::from(0),
                });
            }
            if !self.two_places {
                break;
            }
        }
        Some(out)
    }
}

/// Closed-form residues at the places over `x = ∞` of `y² = a(x)` for
/// `h = A + B·y`.  Returns `None` outside the hyperelliptic scope (`n ≠ 2`,
/// `deg a < 1`, or a `y`-degree above 1 in `h`).
///
/// Unlike [`residues_at_infinity`] this never silently drops a place: the two
/// (or one) places over `∞` are enumerated by degree parity, and their residues
/// are computed in `ℚ(√lc)` rather than searched for over `ℚ`.
pub(crate) fn residues_at_infinity_exact(
    n: usize,
    a: &QPoly,
    h: &AlgElem,
) -> Option<InfinityResidues> {
    if n != 2 {
        return None;
    }
    let a = trim(a.clone());
    let m = degree(&a);
    if m < 1 {
        return None;
    }
    if h.len() > 2 && h[2..].iter().any(|c| !trim(c.numer().clone()).is_empty()) {
        return None; // `h` is not `A + B·y` on this curve
    }
    let m = m as usize;
    let lc = a[m].clone();
    let a_c = h.first().cloned().unwrap_or_else(|| RatFn::int(0));
    let b_c = h.get(1).cloned().unwrap_or_else(|| RatFn::int(0));

    // Truncation bound.  `ratfn_at_infinity` inverts the denominator series,
    // whose lowest exponent is `+deg den`, so reaching exponent `k` of the
    // quotient needs headroom for the numerator's `−deg num` shift as well.
    let dmax = h
        .iter()
        .map(|c| degree(c.numer()).max(degree(c.denom())).max(0))
        .max()
        .unwrap_or(0);
    let s = (m / 2) as i64;
    let u = s + 2 * dmax + 4;

    // R₀ = −[z¹] A(1/z) — the same for both places; doubled at a ramified one.
    let ca = ratfn_at_infinity(&a_c, 1, u);
    let c1 = ca.get(&1).cloned().unwrap_or_else(|| Rational::from(0));

    if m % 2 == 1 {
        return Some(InfinityResidues {
            r0: -Rational::from(2) * c1,
            r1: Rational::from(0),
            lc,
            two_places: false,
        });
    }

    // T(z) = √(ã(z)/lc) ∈ ℚ[[z]], T(0) = 1.  `cb` starts at exponent
    // `−deg(num B)`, so reaching `[z^{s+1}]` of the product needs `T` that much
    // further out than `s+1` — truncating at `s+2` silently drops the term that
    // makes `res(x√(1+x⁴) dx) = ∓½` nonzero.
    let a_tilde_over_lc: Vec<Rational> = (0..=m)
        .map(|k| a[m - k].clone() / lc.clone()) // ã_k = a_{m−k}
        .collect();
    let t = series_sqrt_unit(&a_tilde_over_lc, u as usize)?;
    let mut t_ts = TS::new();
    for (k, c) in t.iter().enumerate() {
        if *c != 0 {
            t_ts.insert(k as i64, c.clone());
        }
    }
    let cb = ratfn_at_infinity(&b_c, 1, u);
    let prod = ts_mul(&cb, &t_ts, s + 2);
    let r1 = -prod
        .get(&(s + 1))
        .cloned()
        .unwrap_or_else(|| Rational::from(0));

    Some(InfinityResidues {
        r0: -c1,
        r1,
        lc,
        two_places: true,
    })
}

/// Power series `T` with `T² = s` and `T(0) = 1`, to `prec` coefficients.
/// `s[0]` must be `1`.
fn series_sqrt_unit(s: &[Rational], prec: usize) -> Option<Vec<Rational>> {
    if s.first().map(|c| *c != 1).unwrap_or(true) {
        return None;
    }
    let mut t = vec![Rational::from(0); prec.max(1)];
    t[0] = Rational::from(1);
    for k in 1..prec {
        let mut acc = s.get(k).cloned().unwrap_or_else(|| Rational::from(0));
        for i in 1..k {
            acc -= t[i].clone() * &t[k - i];
        }
        t[k] = acc / Rational::from(2);
    }
    Some(t)
}

/// The rational square root of `r`, when it has one.
fn rational_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    let (n, d) = (r.numer().clone(), r.denom().clone());
    let (ns, ds) = (n.clone().sqrt(), d.clone().sqrt());
    if Integer::from(&ns * &ns) == n && Integer::from(&ds * &ds) == d {
        Some(Rational::from((ns, ds)))
    } else {
        None
    }
}

// ===========================================================================
// Completeness certificate
// ===========================================================================

/// Does the residue enumeration for `h dx` on `y² = a` provably cover **every**
/// place of the curve?
///
/// The residue theorem (`Σ res = 0`) is *not* a completeness check: a missing
/// conjugate pair sums to zero on its own, and an enumeration that finds no
/// places at all sums to zero vacuously.  Reading either as "no residues
/// anywhere" is how `∫x dx/√(1−x⁴)` came to be certified non-elementary.  This
/// predicate is the actual check, and it is deliberately conservative: `false`
/// means "cannot certify", never "incomplete".
///
/// It rests on three facts about `y² = a` with `a` **squarefree**:
///
/// 1. **Branch places carry `res = 2·Res_α(A)`.**  At a root `α` of `a` the
///    place is simply ramified (`x − α = t²`, `y = t·V(t²)`), so `B·y·dx`
///    expands in *even* powers of `t` alone and contributes nothing to
///    `[t^{−1}]`, whatever the order of `B`'s pole there.  Only `A` can leave a
///    residue at a branch place — which is why `∫B√P dx` (where `A ≡ 0`) is
///    unaffected by poles of `B` on the branch locus.
/// 2. **Finite non-branch places** are the roots of the `a`-coprime part `Dc`
///    of `D = lcm(den A, den B)`.  [`finite_residues`] covers those with a
///    rational base point *and* a rational sheet at any pole order;
///    [`finite_residues_algebraic`] covers the rest, but only for simple poles
///    and only when the whole of `D` is squarefree.
/// 3. **Places over `∞`** are enumerated exactly by
///    [`residues_at_infinity_exact`].
pub(crate) fn residue_enumeration_is_complete(n: usize, a: &QPoly, h: &AlgElem) -> bool {
    if n != 2 {
        return false;
    }
    let a = trim(a.clone());
    if degree(&a) < 1 {
        return false;
    }
    // Squarefree `a`: every branch place is simply ramified (fact 1).
    if degree(&poly_gcd(&a, &poly_deriv(&a))) > 0 {
        return false;
    }
    if residues_at_infinity_exact(2, &a, h).is_none() {
        return false; // fact 3 unavailable
    }

    let a_c = h.first().cloned().unwrap_or_else(|| RatFn::int(0));
    let b_c = h.get(1).cloned().unwrap_or_else(|| RatFn::int(0));
    let den_of = |c: &RatFn| -> QPoly {
        if trim(c.numer().clone()).is_empty() {
            vec![Rational::from(1)]
        } else {
            c.denom().clone()
        }
    };
    let a_den = den_of(&a_c);
    let b_den = den_of(&b_c);

    // Fact 1: a residue at a branch place needs a pole of `A` there, and only
    // *rational* branch points are enumerated.  Requiring `A` to be regular on
    // the branch locus is the conservative reading (and is free when `A ≡ 0`).
    if degree(&poly_gcd(&a_den, &a)) > 0 {
        return false;
    }

    // Fact 2: the `a`-coprime part of the common pole denominator.
    let d = poly_lcm(&a_den, &b_den);
    let mut dc = d.clone();
    loop {
        let g = poly_gcd(&dc, &a);
        if degree(&g) < 1 {
            break;
        }
        dc = poly_div_exact(&dc, &g);
    }
    if degree(&dc) < 1 {
        return true; // no finite non-branch poles at all
    }
    // Which factors does the rational-Puiseux routine already own?
    let all_rational_places = factor_including_zero_root(&dc)
        .into_iter()
        .all(|(q, deg_q)| {
            deg_q == 1 && {
                let alpha = -q.first().cloned().unwrap_or_else(|| Rational::from(0));
                is_rational_square(&eval_q(&a, &alpha))
            }
        });
    if all_rational_places {
        return true; // `finite_residues` handles these at any pole order
    }
    // Algebraic places are present, so `finite_residues_algebraic` must be live:
    // it needs simple poles throughout `D`.
    degree(&poly_gcd(&d, &poly_deriv(&d))) == 0
}

/// A residue divisor that is **certified complete**: every place of the curve
/// has been enumerated, and every nonzero residue appears in exactly one of the
/// two lists.  An empty [`CertifiedDivisor`] therefore genuinely means "no
/// residues anywhere" — which is the only thing a non-elementarity certificate
/// may rest on.
#[derive(Clone, Debug)]
pub(crate) struct CertifiedDivisor {
    /// Places with a **rational** residue: the finite rational places plus the
    /// places over `∞`.  In the FIND-ORDER representation.
    pub(crate) rational: Vec<PlacedResidue>,
    /// Finite places whose residues live in a number field (Trager's route).
    pub(crate) algebraic: Vec<AlgResidue>,
}

/// The certified-complete residue divisor of `h dx` on `yⁿ = a`, or `None` when
/// completeness cannot be established.
///
/// `None` covers three refusals, all of which must make the caller decline
/// rather than pronounce:
///
/// * the enumeration is not certifiable ([`residue_enumeration_is_complete`]);
/// * a residue over `∞` lies outside `ℚ` — `±i` on `y² = 1−x⁴`, say — so it
///   cannot be represented in the rational [`PlacedResidue`] form FIND-ORDER
///   consumes, and the rational part alone is *not* the whole divisor;
/// * the residue theorem fails on the assembled divisor, which would mean a bug
///   here rather than an out-of-scope input.
pub(crate) fn certified_residue_divisor(
    n: usize,
    a: &QPoly,
    h: &AlgElem,
) -> Option<CertifiedDivisor> {
    if !residue_enumeration_is_complete(n, a, h) {
        return None;
    }
    let inf = residues_at_infinity_exact(n, a, h)?;
    if inf.is_irrational() {
        return None;
    }
    let mut rational = finite_residues_placed(n, a, h);
    rational.extend(inf.placed()?);
    let algebraic = finite_residues_algebraic(n, a, h);

    // Residue theorem on the now-complete divisor — a self-check, not the
    // completeness argument (see the module header).
    let mut total = rational
        .iter()
        .fold(Rational::from(0), |s, r| s + &r.residue.value);
    for r in &algebraic {
        let nf = NumberField::new(r.minpoly.clone());
        total += Rational::from(2) * nf.trace(&r.r0);
    }
    if total != 0 {
        return None;
    }
    Some(CertifiedDivisor {
        rational,
        algebraic,
    })
}

/// Monic irreducible factors of `p` over `ℚ`, **including `x` itself** when `0`
/// is a root.
///
/// [`factor_over_q`] deliberately divides out the largest power of `x` first —
/// for its Puiseux callers the constant root `c = 0` is not a branch and must
/// not appear.  Here it is an ordinary place like any other, and dropping it is
/// how `∫√(x⁴−1)/x dx` came to be certified non-elementary: the pole at `x = 0`
/// has an irrational sheet (`a(0) = −1`), so the rational-Puiseux routine cannot
/// see it either, and between the two omissions the residue divisor looked
/// empty.  Multiplicity is not returned (matching `factor_over_q`), so `x`
/// appears once.
fn factor_including_zero_root(p: &QPoly) -> Vec<(QPoly, usize)> {
    let mut out = factor_over_q(p);
    if trim(p.clone()).first().map(|c0| *c0 == 0) == Some(true) {
        out.push((vec![Rational::from(0), Rational::from(1)], 1));
    }
    out
}

/// Least common multiple `a·b/gcd(a,b)` over `ℚ[x]`.
fn poly_lcm(a: &QPoly, b: &QPoly) -> QPoly {
    if degree(a) < 0 || degree(b) < 0 {
        return vec![Rational::from(1)];
    }
    poly_div_exact(&poly_mul(a, b), &poly_gcd(a, b))
}

/// Horner evaluation of `p` at a rational point.
fn eval_q(p: &QPoly, x: &Rational) -> Rational {
    p.iter().rev().fold(Rational::from(0), |acc, c| acc * x + c)
}

/// Is the rational `r` a perfect square in `ℚ` (`r = (s)²`, `s ∈ ℚ`)?
fn is_rational_square(r: &Rational) -> bool {
    rational_sqrt(r).is_some()
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

    /// Algebraic place over `x² − 2` (`α = ±√2`) on `y² = x`, differential
    /// `(x + y)/(x²−2) dx`.  Direct residue values: `r0 = ½`, `r1 = ¼·√2`
    /// (so res on sheet ± = ½ ± ¼√2·√(√2)), conjugates = 2.
    #[test]
    fn algebraic_residue_values() {
        // h = (x + y)/(x²−2) = AlgElem [x/(x²−2), 1/(x²−2)].
        let h = vec![rf(&[0, 1], &[-2, 0, 1]), rf(&[1], &[-2, 0, 1])];
        let res = finite_residues_algebraic(2, &qp(&[0, 1]), &h);
        assert_eq!(res.len(), 1);
        let ar = &res[0];
        assert_eq!(ar.minpoly, qp(&[-2, 0, 1])); // x²−2
        assert_eq!(ar.conjugates, 2);
        assert_eq!(ar.r0, vec![Rational::from((1, 2))]); // ½
        assert_eq!(ar.r1, vec![r(0), Rational::from((1, 4))]); // ¼·√2
    }

    /// Residue theorem over the **complete** divisor (rational + algebraic +
    /// infinity) must sum to zero — the soundness check for algebraic places.
    #[test]
    fn residue_theorem_with_algebraic_places() {
        // (x + y)/(x²−2) on y²=x: poles only at the algebraic place ±√2.
        let h = vec![rf(&[0, 1], &[-2, 0, 1]), rf(&[1], &[-2, 0, 1])];
        assert_eq!(residue_sum_complete(2, &qp(&[0, 1]), &h), r(0));

        // A mixed case: 1/((x−1)(x²−3)) · (1 + y) on y²=x+1, rational pole at
        // x=1 plus an algebraic place over x²−3.
        let den = qp(&[3, -3, -1, 1]); // (x−1)(x²−3) = x³ − x² − 3x + 3
        let h2 = vec![
            RatFn::new(qp(&[1]), den.clone()),
            RatFn::new(qp(&[1]), den.clone()),
        ];
        assert_eq!(residue_sum_complete(2, &qp(&[1, 1]), &h2), r(0));
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

    // -----------------------------------------------------------------------
    // Places over ∞: the closed form vs. the rational-Puiseux routine
    // -----------------------------------------------------------------------

    /// `x dx/√(1−x⁴)`: the two places over `∞` carry residues `±i`, which the
    /// rational-Puiseux routine cannot see (it finds **no branches at all**,
    /// because `w² = z⁴/(z⁴−1)` has leading coefficient `±i ∉ ℚ`).  This is the
    /// exact hole that certified `∫x dx/√(1−x⁴)` non-elementary.
    #[test]
    fn infinity_residues_are_algebraic_for_one_minus_x4() {
        let a = qp(&[1, 0, 0, 0, -1]);
        let h = vec![RatFn::int(0), rf(&[0, 1], &[1, 0, 0, 0, -1])];

        assert!(
            super::residues_at_infinity(2, &a, &h).is_empty(),
            "rational Puiseux sees nothing over ∞ here"
        );
        assert!(
            super::puiseux_at_infinity(2, &a, 8).is_empty(),
            "…because it finds no branches at all"
        );

        let inf = super::residues_at_infinity_exact(2, &a, &h).expect("hyperelliptic");
        // res = 0 ± 1·√(−1) = ±i.
        assert_eq!(inf.r0, r(0));
        assert_eq!(inf.r1, r(1));
        assert_eq!(inf.lc, r(-1));
        assert!(inf.two_places);
        assert!(!inf.all_zero(), "there *are* residues over ∞");
        assert!(inf.is_irrational());
        assert!(inf.placed().is_none(), "±i is not representable in ℚ");

        // The *places* are all enumerated — it is their residues that cannot be
        // written down over ℚ, so no certified divisor is available and the
        // empty rational one must not be read as "no residues anywhere".
        assert!(super::residue_divisor(2, &a, &h).is_empty());
        assert_eq!(super::residue_sum_complete(2, &a, &h), r(0)); // vacuously!
        assert!(super::residue_enumeration_is_complete(2, &a, &h));
        assert!(super::certified_residue_divisor(2, &a, &h).is_none());
    }

    /// `x dx/√(1−x⁶)` (genus 2): `x dx/y` is *holomorphic* at `∞` too, so the
    /// empty divisor here is real — and certifiable.
    #[test]
    fn infinity_residues_vanish_for_one_minus_x6() {
        let a = qp(&[1, 0, 0, 0, 0, 0, -1]);
        let h = vec![RatFn::int(0), rf(&[0, 1], &[1, 0, 0, 0, 0, 0, -1])];
        let inf = super::residues_at_infinity_exact(2, &a, &h).expect("hyperelliptic");
        assert!(inf.all_zero());
        assert_eq!(inf.placed().map(|p| p.len()), Some(0));
        assert!(super::residue_enumeration_is_complete(2, &a, &h));
        let cert = super::certified_residue_divisor(2, &a, &h).expect("certifiable");
        assert!(cert.rational.is_empty() && cert.algebraic.is_empty());
    }

    /// `x dx/√(1+x⁴)`: `lc = 1` is a square, the places over `∞` are rational
    /// (`±1`), and the closed form reproduces what the Puiseux routine finds.
    #[test]
    fn infinity_residues_agree_when_rational() {
        let a = qp(&[1, 0, 0, 0, 1]);
        let h = vec![RatFn::int(0), rf(&[0, 1], &[1, 0, 0, 0, 1])];
        let inf = super::residues_at_infinity_exact(2, &a, &h).expect("hyperelliptic");
        assert!(!inf.is_irrational());
        let mut exact: Vec<Rational> = inf
            .placed()
            .unwrap()
            .iter()
            .map(|p| p.residue.value.clone())
            .collect();
        let mut puiseux: Vec<Rational> = super::residues_at_infinity(2, &a, &h)
            .iter()
            .map(|p| p.value.clone())
            .collect();
        exact.sort();
        puiseux.sort();
        assert_eq!(exact, puiseux);
        assert_eq!(exact, vec![r(-1), r(1)]);
    }

    /// `x√(1+x⁴) dx` — a *polynomial* weight, so the only poles are over `∞`
    /// and the residues there are `∓½`.  The series square root has to be
    /// carried past `z^{s+1}` to see them, because `B(1/z)` starts at `z^{−1}`.
    #[test]
    fn infinity_residues_of_a_polynomial_weight() {
        let a = qp(&[1, 0, 0, 0, 1]);
        let h = vec![RatFn::int(0), rf(&[0, 1], &[1])]; // B = x
        let inf = super::residues_at_infinity_exact(2, &a, &h).expect("hyperelliptic");
        assert_eq!(inf.r0, r(0));
        assert_eq!(inf.r1, Rational::from((-1, 2)));
        let vals: Vec<Rational> = inf
            .placed()
            .unwrap()
            .iter()
            .map(|p| p.residue.value.clone())
            .collect();
        assert_eq!(vals.len(), 2, "two nonzero places over ∞");
        assert_eq!(vals[0].clone() + vals[1].clone(), r(0));
    }

    /// `∫dx/√(x⁵+1)` — odd degree, a single ramified place over `∞` with
    /// residue `−2·[z¹]A(1/z) = 0` (here `A ≡ 0`).  The whole divisor is
    /// certifiably empty, which is what licenses that famous certificate.
    #[test]
    fn odd_degree_infinity_place_is_rational_and_certified() {
        let a = qp(&[1, 0, 0, 0, 0, 1]);
        let h = vec![RatFn::int(0), rf(&[1], &[1, 0, 0, 0, 0, 1])];
        let inf = super::residues_at_infinity_exact(2, &a, &h).expect("hyperelliptic");
        assert!(!inf.two_places);
        assert_eq!(inf.r1, r(0));
        assert!(inf.all_zero());
        // The pole denominator is `a` itself: every finite pole sits on the
        // branch locus, where `A ≡ 0` leaves no residue.
        assert!(super::residue_enumeration_is_complete(2, &a, &h));
        let cert = super::certified_residue_divisor(2, &a, &h).expect("certifiable");
        assert!(cert.rational.is_empty() && cert.algebraic.is_empty());
    }

    /// A residue at `∞` that is rational but that the Puiseux routine misses
    /// (odd degree, non-square leading coefficient): `A = 1/x`, `a = 2x³+1`.
    /// The closed form gets it; `residues_at_infinity` returns nothing.
    #[test]
    fn odd_degree_nonsquare_lc_infinity_residue() {
        let a = qp(&[1, 0, 0, 2]);
        let h = vec![rf(&[1], &[0, 1]), RatFn::int(0)];
        let inf = super::residues_at_infinity_exact(2, &a, &h).expect("hyperelliptic");
        // A(1/z) = z, so [z¹] = 1 and res = −2.
        assert_eq!(inf.r0, r(-2));
        assert_eq!(inf.r1, r(0));
    }

    /// `√(x⁴−1)/x dx` — the pole at `x = 0` sits on an *irrational* sheet
    /// (`a(0) = −1`), so the rational-Puiseux routine cannot see it, and
    /// `factor_over_q` used to drop the factor `x` before the algebraic routine
    /// got a chance.  Between the two the divisor looked empty and the integral
    /// — which is elementary — was certified non-elementary.
    #[test]
    fn pole_at_the_origin_is_not_dropped() {
        let a = qp(&[-1, 0, 0, 0, 1]);
        let h = vec![RatFn::int(0), rf(&[1], &[0, 1])]; // B = 1/x
        let alg = super::finite_residues_algebraic(2, &a, &h);
        assert_eq!(alg.len(), 1, "the place over x = 0 must be enumerated");
        assert_eq!(alg[0].minpoly, qp(&[0, 1]));
        assert!(alg[0].r0.iter().all(|c| *c == 0)); // rational part zero…
        assert_eq!(alg[0].r1, vec![r(1)]); // …residues are ±√(a(0)) = ±i
        assert!(super::finite_residues(2, &a, &h).is_empty());

        let cert = super::certified_residue_divisor(2, &a, &h).expect("certifiable");
        assert!(
            !cert.algebraic.is_empty(),
            "the certified divisor must not be empty here"
        );
    }

    /// The certificate refuses a non-squarefree radicand and `n ≠ 2`.
    #[test]
    fn completeness_certificate_refuses_out_of_scope() {
        let sq = qp(&[0, 0, 1]); // x² — not squarefree
        let h = vec![RatFn::int(0), rf(&[1], &[1])];
        assert!(!super::residue_enumeration_is_complete(2, &sq, &h));
        assert!(!super::residue_enumeration_is_complete(
            3,
            &qp(&[1, 0, 0, 1]),
            &h
        ));
    }
}
