//! Lean 4 certificates for Zeilberger (creative telescoping) and WZ certificates.
//!
//! A [`ZeilbergerResult`] for a proper hypergeometric term `F(n, k)` claims
//!
//! ```text
//! Σ_{j=0}^{J} a_j(n)·F(n+j, k) = G(n, k+1) − G(n, k),    G(n, k) = R(n, k)·F(n, k)
//! ```
//!
//! (a WZ pair is the order-1 case `a = (−c, c)`). Dividing by `F(n, k)` turns
//! that into a pure rational-function identity in `n, k`, because every ratio
//! `F(n+j, k)/F(n, k)` and `F(n, k+1)/F(n, k)` is rational for a proper
//! hypergeometric term. The proof clears every denominator with an explicit
//! `linear_combination` of the shift relations (cofactors computed here from
//! the factorisations, so `ring` only checks a polynomial identity) and divides
//! back out with `div_sub_div` / `eq_div_iff`.
//!
//! The emitted theorem is stated for an arbitrary `F : ℝ → ℝ → ℝ` that
//! satisfies the term's shift relations, written with denominators cleared
//! (`D·F(n+1, k) = N·F(n, k)`), plus explicit `≠ 0` hypotheses for every
//! denominator that is divided by. Its conclusion is the telescoping identity
//! itself, with `G` written as the certificate `R` times `F`. The relations
//! hold for the actual `F` wherever it is defined — the Python test checks them
//! numerically against the Γ-function form of the term — so the theorem
//! applies to it; it is never weakened to "the rational identity holds".
//!
//! What is *not* stated: the recurrence for the sum `Σ_k F(n, k)`. Summing the
//! identity over a range telescopes to a boundary difference that is a separate
//! claim (`ZeilbergerCertificate.boundary`); the certificate stops at the
//! term-level identity, which is exactly what `zeilberger` verified.

use super::{contains_admission, is_lean_binder_name, NON_VARIABLE_SYMBOLS};
use crate::holonomic::hyperterm::{as_ratk, ProperTerm};
use crate::holonomic::qfield::{PolyK, RatK};
use crate::holonomic::zeilberger::ZeilbergerResult;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;
use rug::{Integer, Rational};
use std::collections::BTreeMap;

/// A polynomial in `n, k` over `ℚ`: `(deg_n, deg_k) ↦ coefficient`, no zeros.
type Bi = BTreeMap<(u32, u32), Rational>;

/// Largest recurrence order emitted (one hypothesis per shift in `n`).
const MAX_ORDER: usize = 6;

/// Largest total degree of any printed polynomial; `ring` on the cleared
/// identity grows with it.
const MAX_DEGREE: u32 = 24;

fn bi_add_term(b: &mut Bi, key: (u32, u32), c: Rational) {
    if c == 0 {
        return;
    }
    let slot = b.entry(key).or_insert_with(|| Rational::from(0));
    *slot += c;
    if *slot == 0 {
        b.remove(&key);
    }
}

/// `u(n)·b(n, k)`.
fn uni_mul_bi(u: &RatUniPoly, b: &Bi) -> Bi {
    let mut out = Bi::new();
    for (i, c) in u.coeffs.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        for (&(dn, dk), bc) in b {
            bi_add_term(&mut out, (dn + i as u32, dk), Rational::from(c * bc));
        }
    }
    out
}

/// `p = B / L` with `B ∈ ℚ[n, k]` and `L ∈ ℚ[n]` (the lcm of the coefficient
/// denominators of `p ∈ ℚ(n)[k]`).
fn polyk_cleared(p: &PolyK) -> Option<(Bi, RatUniPoly)> {
    let den_of = |d: &RatUniPoly| {
        if d.is_zero() {
            RatUniPoly::one()
        } else {
            d.clone()
        }
    };
    let mut l = RatUniPoly::one();
    for c in &p.coeffs {
        let d = den_of(&c.den);
        let g = l.gcd(&d);
        if g.is_zero() {
            return None;
        }
        let (q, r) = RatUniPoly::div_rem(&d, &g);
        if !r.is_zero() {
            return None;
        }
        l = &l * &q;
    }
    let mut out = Bi::new();
    for (dk, c) in p.coeffs.iter().enumerate() {
        if c.num.is_zero() {
            continue;
        }
        let (q, r) = RatUniPoly::div_rem(&l, &den_of(&c.den));
        if !r.is_zero() {
            return None;
        }
        let scaled = &c.num * &q;
        for (dn, v) in scaled.coeffs.iter().enumerate() {
            bi_add_term(&mut out, (dn as u32, dk as u32), v.clone());
        }
    }
    Some((out, l))
}

/// Slices of `b` by `k`-degree, each a polynomial in `n`.
fn k_slices(b: &Bi) -> Vec<RatUniPoly> {
    let mut by_k: BTreeMap<u32, Vec<Rational>> = BTreeMap::new();
    for (&(dn, dk), c) in b {
        let v = by_k.entry(dk).or_default();
        if v.len() <= dn as usize {
            v.resize(dn as usize + 1, Rational::from(0));
        }
        v[dn as usize] = c.clone();
    }
    by_k.into_values()
        .map(|coeffs| RatUniPoly { coeffs }.trim())
        .collect()
}

/// `b / u` for a polynomial `u(n)` dividing every `k`-slice of `b`.
fn bi_div_uni(b: &Bi, u: &RatUniPoly) -> Option<Bi> {
    let mut by_k: BTreeMap<u32, Vec<Rational>> = BTreeMap::new();
    for (&(dn, dk), c) in b {
        let v = by_k.entry(dk).or_default();
        if v.len() <= dn as usize {
            v.resize(dn as usize + 1, Rational::from(0));
        }
        v[dn as usize] = c.clone();
    }
    let mut out = Bi::new();
    for (dk, coeffs) in by_k {
        let (q, r) = RatUniPoly::div_rem(&RatUniPoly { coeffs }.trim(), u);
        if !r.is_zero() {
            return None;
        }
        for (dn, c) in q.coeffs.into_iter().enumerate() {
            bi_add_term(&mut out, (dn as u32, dk), c);
        }
    }
    Some(out)
}

/// Leading term of a printed polynomial: highest total degree, then highest
/// `n`-degree (the order [`print_bi`] writes terms in).
fn print_order(b: &Bi) -> Vec<((u32, u32), Rational)> {
    let mut terms: Vec<((u32, u32), Rational)> =
        b.iter().map(|(&key, c)| (key, c.clone())).collect();
    terms.sort_by_key(|&((dn, dk), _)| std::cmp::Reverse((dn + dk, dn)));
    terms
}

/// Scale `(num, den)` by one rational so that every coefficient is an integer
/// with no common factor and `den`'s leading printed coefficient is positive.
fn integerize(num: &mut Bi, den: &mut Bi) {
    let mut lcm = Integer::from(1);
    for c in num.values().chain(den.values()) {
        lcm.lcm_mut(c.denom());
    }
    let mut g = Integer::from(0);
    for c in num.values().chain(den.values()) {
        let v = Integer::from(c.numer() * &lcm) / c.denom();
        g.gcd_mut(&v);
    }
    if g == 0 {
        return;
    }
    let mut factor = Rational::from((lcm, g));
    if let Some((_, lc)) = print_order(den).first() {
        if *lc < 0 {
            factor = -factor;
        }
    }
    for c in num.values_mut().chain(den.values_mut()) {
        *c *= &factor;
    }
}

/// `r ∈ ℚ(n, k)` as `num / den`, both in `ℤ[n, k]`, with the common
/// `n`-content cancelled and `den` sign-normalised.
fn ratk_frac(r: &RatK) -> Option<(Bi, Bi)> {
    let (bn, ln) = polyk_cleared(&r.num)?;
    let (bd, ld) = polyk_cleared(&r.den)?;
    if bd.is_empty() {
        return None;
    }
    let mut num = uni_mul_bi(&ld, &bn);
    let mut den = uni_mul_bi(&ln, &bd);
    // Cancel a common factor in `n` (the `k`-content gcd of both sides).
    let mut g = RatUniPoly::zero();
    for s in k_slices(&num).iter().chain(k_slices(&den).iter()) {
        g = if g.is_zero() { s.clone() } else { g.gcd(s) };
    }
    if g.degree() > 0 {
        num = bi_div_uni(&num, &g)?;
        den = bi_div_uni(&den, &g)?;
    }
    integerize(&mut num, &mut den);
    Some((num, den))
}

fn bi_total_degree(b: &Bi) -> u32 {
    b.keys().map(|(a, c)| a + c).max().unwrap_or(0)
}

fn bi_is_constant(b: &Bi) -> bool {
    b.keys().all(|&(a, c)| a == 0 && c == 0)
}

/// Print a polynomial with integer coefficients in a shape `field_simp`'s simp
/// set leaves alone (so a `≠ 0` hypothesis printed the same way discharges the
/// side goal by assumption): expanded, highest degree first, no `1 *`, no
/// `^ 1`, subtraction for negative coefficients, and — for denominators — a
/// positive leading coefficient. `None` if a coefficient is not an integer.
fn print_bi(b: &Bi, n: &str, k: &str) -> Option<String> {
    if b.is_empty() {
        return Some("0".to_string());
    }
    let mut out = String::new();
    for (i, ((dn, dk), c)) in print_order(b).into_iter().enumerate() {
        if *c.denom() != 1 {
            return None;
        }
        let neg = c < 0;
        let mag = Integer::from(c.numer().abs_ref());
        let mut factors: Vec<String> = Vec::new();
        for (name, d) in [(n, dn), (k, dk)] {
            match d {
                0 => {}
                1 => factors.push(name.to_string()),
                _ => factors.push(format!("{name} ^ {d}")),
            }
        }
        let mono = if factors.is_empty() {
            mag.to_string()
        } else if mag == 1 {
            factors.join(" * ")
        } else {
            format!("{mag} * {}", factors.join(" * "))
        };
        if i == 0 {
            if neg {
                out.push('-');
            }
            out.push_str(&mono);
        } else {
            out.push_str(if neg { " - " } else { " + " });
            out.push_str(&mono);
        }
    }
    Some(out)
}

fn symbol_name(id: ExprId, pool: &ExprPool) -> Option<String> {
    pool.with(id, |d| match d {
        ExprData::Symbol { name, .. } => Some(name.clone()),
        _ => None,
    })
}

/// A polynomial in `n, k` as `unit · Π factorᵉ`, every factor irreducible
/// over `ℤ`, non-constant, primitive and with a positive leading printed
/// coefficient — so equal factors are equal maps and exponents can be
/// compared, added and subtracted.
#[derive(Clone, Debug)]
struct Fp {
    unit: Rational,
    pows: BTreeMap<Bi, u32>,
}

impl Fp {
    fn constant(c: Rational) -> Fp {
        Fp {
            unit: c,
            pows: BTreeMap::new(),
        }
    }

    fn is_zero(&self) -> bool {
        self.unit == 0
    }

    fn is_one(&self) -> bool {
        self.unit == 1 && self.pows.is_empty()
    }

    fn mul(&self, other: &Fp) -> Fp {
        if self.is_zero() || other.is_zero() {
            return Fp::constant(Rational::from(0));
        }
        let mut pows = self.pows.clone();
        for (f, e) in &other.pows {
            *pows.entry(f.clone()).or_insert(0) += e;
        }
        Fp {
            unit: Rational::from(&self.unit * &other.unit),
            pows,
        }
    }

    /// `self / other`, when `other`'s factors divide `self`'s.
    fn div(&self, other: &Fp) -> Option<Fp> {
        if other.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(self.clone());
        }
        let mut pows = self.pows.clone();
        for (f, e) in &other.pows {
            let have = pows.get_mut(f)?;
            if *have < *e {
                return None;
            }
            *have -= e;
            if *have == 0 {
                pows.remove(f);
            }
        }
        Some(Fp {
            unit: Rational::from(&self.unit / &other.unit),
            pows,
        })
    }

    fn neg(&self) -> Fp {
        Fp {
            unit: Rational::from(-&self.unit),
            pows: self.pows.clone(),
        }
    }
}

/// Factor `b` over `ℤ[n, k]`. Falls back to the whole (primitive,
/// sign-normalised) polynomial as one "factor" when FLINT declines, so
/// printing never fails on account of the factoriser.
fn factor_bi(b: &Bi, n: ExprId, k: ExprId) -> Option<Fp> {
    if b.is_empty() {
        return Some(Fp::constant(Rational::from(0)));
    }
    // Integer, primitive copy: `b = scale · prim`.
    let mut lcm = Integer::from(1);
    for c in b.values() {
        lcm.lcm_mut(c.denom());
    }
    let mut g = Integer::from(0);
    for c in b.values() {
        g.gcd_mut(&(Integer::from(c.numer() * &lcm) / c.denom()));
    }
    let scale = Rational::from((g, lcm));
    let prim: BTreeMap<(u32, u32), Integer> = b
        .iter()
        .map(|(&key, c)| {
            let q = Rational::from(c / &scale);
            (key, q.numer().clone())
        })
        .collect();
    let as_mp = crate::poly::MultiPoly {
        vars: vec![n, k],
        terms: prim
            .iter()
            .map(|(&(dn, dk), c)| (vec![dn, dk], c.clone()))
            .collect(),
    };
    let from_terms = |terms: &crate::poly::MultiPoly| -> Bi {
        let mut out = Bi::new();
        for (e, c) in &terms.terms {
            let dn = e.first().copied().unwrap_or(0);
            let dk = e.get(1).copied().unwrap_or(0);
            bi_add_term(&mut out, (dn, dk), Rational::from(c.clone()));
        }
        out
    };
    let (unit, raw): (Integer, Vec<(Bi, u32)>) = match as_mp.factor_irreducible() {
        Some((u, fs)) => (u, fs.iter().map(|(f, e)| (from_terms(f), *e)).collect()),
        None => (
            Integer::from(1),
            vec![(
                prim.iter()
                    .map(|(&key, c)| (key, Rational::from(c.clone())))
                    .collect(),
                1_u32,
            )],
        ),
    };
    let mut out = Fp::constant(Rational::from(unit) * scale);
    for (mut f, e) in raw {
        if bi_is_constant(&f) {
            let c = f.get(&(0, 0)).cloned().unwrap_or_else(|| Rational::from(0));
            for _ in 0..e {
                out.unit *= &c;
            }
            continue;
        }
        if print_order(&f).first().is_some_and(|(_, c)| *c < 0) {
            for c in f.values_mut() {
                *c = Rational::from(-&*c);
            }
            if e % 2 == 1 {
                out.unit = Rational::from(-&out.unit);
            }
        }
        *out.pows.entry(f).or_insert(0) += e;
    }
    Some(out)
}

/// `true` when `b` is a single variable with coefficient 1 (printed bare).
fn is_bare_variable(b: &Bi) -> bool {
    b.len() == 1
        && b.iter()
            .next()
            .is_some_and(|(&(dn, dk), c)| *c == 1 && dn + dk == 1)
}

fn rational_text(q: &Rational) -> String {
    if *q.denom() == 1 {
        q.numer().to_string()
    } else {
        format!("({} / {} : ℝ)", q.numer(), q.denom())
    }
}

/// `unit * (f₁) ^ e₁ * (f₂) * …`, left-associated, the unit omitted when it
/// is `±1` (a `-` prefix for a negative unit).
fn fp_text(f: &Fp, n: &str, k: &str) -> Option<String> {
    if f.pows.is_empty() {
        return Some(if f.unit < 0 && *f.unit.denom() == 1 {
            f.unit.to_string()
        } else if f.unit < 0 {
            format!("-{}", rational_text(&Rational::from(-&f.unit)))
        } else {
            rational_text(&f.unit)
        });
    }
    let mut parts = Vec::new();
    let mag = Rational::from(f.unit.abs_ref());
    if mag != 1 {
        parts.push(rational_text(&mag));
    }
    for (b, e) in &f.pows {
        let body = print_bi(b, n, k)?;
        let base = if is_bare_variable(b) {
            body
        } else {
            format!("({body})")
        };
        parts.push(if *e == 1 {
            base
        } else {
            format!("{base} ^ {e}")
        });
    }
    let text = parts.join(" * ");
    Some(if f.unit < 0 { format!("-{text}") } else { text })
}

/// The distinct non-constant denominator factors, each with a `≠ 0`
/// hypothesis `hN1, hN2, …`.
#[derive(Default)]
struct NonzeroFactors {
    /// `(printed factor without outer parentheses, hypothesis name)`.
    seen: Vec<(String, String)>,
}

impl NonzeroFactors {
    fn name_for(&mut self, text: String) -> String {
        if let Some((_, h)) = self.seen.iter().find(|(t, _)| *t == text) {
            return h.clone();
        }
        let h = format!("hN{}", self.seen.len() + 1);
        self.seen.push((text, h.clone()));
        h
    }

    /// A proof term of `f ≠ 0` matching the left-associated product
    /// [`fp_text`] prints, for a positive integer unit.
    fn proof(&mut self, f: &Fp, n: &str, k: &str) -> Option<String> {
        if f.unit <= 0 || *f.unit.denom() != 1 {
            return None;
        }
        let mut pieces: Vec<String> = Vec::new();
        if f.unit != 1 || f.pows.is_empty() {
            pieces.push(format!("(by norm_num : ({} : ℝ) ≠ 0)", f.unit));
        }
        for (b, e) in &f.pows {
            let h = self.name_for(print_bi(b, n, k)?);
            pieces.push(if *e == 1 {
                h
            } else {
                format!("(pow_ne_zero {e} {h})")
            });
        }
        let mut it = pieces.into_iter();
        let mut acc = it.next()?;
        for p in it {
            acc = format!("(mul_ne_zero {acc} {p})");
        }
        Some(acc)
    }
}

/// `r ∈ ℚ(n, k)` as factored `num / den`, `den` with a positive integer unit.
fn frac_fp(r: &RatK, n: ExprId, k: ExprId) -> Option<(Fp, Fp, u32)> {
    let (num, den) = ratk_frac(r)?;
    let degree = bi_total_degree(&num).max(bi_total_degree(&den));
    if degree > MAX_DEGREE {
        return None;
    }
    let fnum = factor_bi(&num, n, k)?;
    let fden = factor_bi(&den, n, k)?;
    // Keep the denominator's unit a positive integer (it is the product of
    // integer factors with a positive leading coefficient); move any rational
    // part to the numerator.
    let u = fden.unit.clone();
    if u == 0 {
        return None;
    }
    let mut den_int = fden.clone();
    let mut num_adj = fnum.clone();
    if u < 0 || *u.denom() != 1 {
        let target = Rational::from(Integer::from(u.numer().abs_ref()));
        let ratio = Rational::from(&target / &u);
        den_int.unit = target;
        num_adj.unit = Rational::from(&num_adj.unit * &ratio);
    }
    Some((num_adj, den_int, degree))
}

/// Generate a Lean 4 certificate for a Zeilberger / WZ telescoping identity.
///
/// `term` is the summand `F(n, k)` that was passed to
/// [`crate::holonomic::zeilberger::zeilberger`], `result` its certificate. The
/// emitted `example` binds a function `F : ℝ → ℝ → ℝ` and reals `n k`, assumes
/// `F`'s hypergeometric shift relations with denominators cleared
/// (`D(n,k)·F(n+1, k) = N(n,k)·F(n, k)`, every polynomial factored over `ℤ`)
/// and that each irreducible factor of a denominator is nonzero, and proves
///
/// `Σ_j a_j(n)·F(n+j, k) = P(n,k+1)/Q(n,k+1)·F(n, k+1) − P(n,k)/Q(n,k)·F(n, k)`
///
/// where `R = P/Q` is the certificate (so the right-hand side is
/// `G(n, k+1) − G(n, k)`, `G = R·F`). The proof first clears every
/// denominator with an explicit `linear_combination` of the shift relations —
/// its cofactors are computed here from the factorisations — and then divides
/// back out with `div_sub_div` / `eq_div_iff`; no `field_simp` search runs, so
/// the check stays fast on the Apéry and Franel certificates.
///
/// Before anything is printed, the identity is re-checked exactly in
/// `ℚ(n, k)`; a mismatch, a term outside the proper hypergeometric class, a
/// coefficient that is not a polynomial, an order above 6, a polynomial of
/// total degree above 24, or a variable name that cannot be a Lean binder
/// returns `""` (no certificate).
pub fn emit_zeilberger_cert(
    term: ExprId,
    n: ExprId,
    k: ExprId,
    result: &ZeilbergerResult,
    pool: &ExprPool,
) -> String {
    emit_zeilberger_cert_opt(term, n, k, result, pool).unwrap_or_default()
}

fn emit_zeilberger_cert_opt(
    term: ExprId,
    n: ExprId,
    k: ExprId,
    result: &ZeilbergerResult,
    pool: &ExprPool,
) -> Option<String> {
    let order = result.order;
    if order == 0 || order > MAX_ORDER || result.coeffs.len() != order + 1 {
        return None;
    }
    let n_name = symbol_name(n, pool)?;
    let k_name = symbol_name(k, pool)?;
    if n_name == k_name {
        return None;
    }
    // The names the proof introduces: `hF1 … hFk`, `hN1 …`, `key`. A variable
    // spelled like one of them would be shadowed.
    let generated = |name: &str| {
        let tail = |p: &str| {
            name.strip_prefix(p)
                .is_some_and(|t| t == "k" || t.chars().all(|c| c.is_ascii_digit()))
        };
        tail("hF") || tail("hN") || name == "key"
    };
    for name in [&n_name, &k_name] {
        if !is_lean_binder_name(name)
            || NON_VARIABLE_SYMBOLS.contains(&name.as_str())
            || generated(name)
        {
            return None;
        }
    }
    let f_name = ["F", "T", "W"]
        .into_iter()
        .find(|c| *c != n_name && *c != k_name)?;

    // Exact data: shift ratios of F, the coefficients, R at k and k + 1.
    let pt = ProperTerm::parse(term, n, k, pool).ok()?;
    let ratio_k = pt.ratio_k().ok()?;
    let mut ratio_n = vec![RatK::one()];
    for j in 1..=order {
        ratio_n.push(pt.ratio_n(j as i64).ok()?);
    }
    let r0 = as_ratk(result.certificate, n, k, pool, 0)?;
    let r1 = RatK {
        num: r0.num.shift_k(1),
        den: r0.den.shift_k(1),
    }
    .normalize();
    let mut coeffs = Vec::with_capacity(order + 1);
    for &a in &result.coeffs {
        coeffs.push(as_ratk(a, n, k, pool, 0)?);
    }

    // Defence in depth: the identity divided by F(n, k), exactly.
    let mut lhs = RatK::zero();
    for (a, r) in coeffs.iter().zip(&ratio_n) {
        lhs = lhs.add(&a.mul(r));
    }
    let rhs = r1.mul(&ratio_k).sub(&r0);
    if !lhs.sub(&rhs).is_zero() {
        return None;
    }

    let (n_s, k_s) = (n_name.as_str(), k_name.as_str());
    let txt = |f: &Fp| fp_text(f, n_s, k_s);

    // a_j(n): polynomials.
    let mut a_fp = Vec::with_capacity(order + 1);
    for a in &coeffs {
        let (num, den, _) = frac_fp(a, n, k)?;
        if !den.is_one() {
            return None;
        }
        a_fp.push(num);
    }
    let (p0, q0, _) = frac_fp(&r0, n, k)?;
    let (p1, q1, _) = frac_fp(&r1, n, k)?;

    // Shift relations: (tag, application, N, D, Lean cofactor's numerator X)
    // with `D · app = N · F n k`; X/D is the cofactor before the common
    // multiplier M.
    let base = format!("{f_name} {n_s} {k_s}");
    // `lq = lcm(Q0, Q1)`, `gq = Q0·Q1 / lq`: `key` is cleared by `lq` only.
    let mut lq = Fp::constant(Rational::from(1));
    for f in [&q0, &q1] {
        for (b, e) in &f.pows {
            let slot = lq.pows.entry(b.clone()).or_insert(0);
            *slot = (*slot).max(*e);
        }
    }
    // Denominator units are positive integers (see `frac_fp`).
    lq.unit = Rational::from(Integer::from(q0.unit.numer().lcm_ref(q1.unit.numer())));
    let gq = q0.mul(&q1).div(&lq)?;
    let p0x = p0.mul(&lq.div(&q0)?);
    let p1x = p1.mul(&lq.div(&q1)?);
    let mut shifts: Vec<(String, String, Fp, Fp, Fp)> = Vec::new();
    for (j, r) in ratio_n.iter().enumerate().skip(1) {
        let (num, den, _) = frac_fp(r, n, k)?;
        shifts.push((
            format!("{j}"),
            format!("{f_name} ({n_s} + {j}) {k_s}"),
            num,
            den,
            lq.mul(&a_fp[j]),
        ));
    }
    {
        let (num, den, _) = frac_fp(&ratio_k, n, k)?;
        shifts.push((
            "k".to_string(),
            format!("{f_name} {n_s} ({k_s} + 1)"),
            num,
            den,
            p1x.neg(),
        ));
    }

    // Smallest M (a product of denominator factors) with D | M·X for every
    // relation, so every cofactor M·X/D is a polynomial.
    let mut m = Fp::constant(Rational::from(1));
    for (_, _, _, d, x) in &shifts {
        for (f, e) in &d.pows {
            let have = x.pows.get(f).copied().unwrap_or(0) + m.pows.get(f).copied().unwrap_or(0);
            if !x.is_zero() && have < *e {
                *m.pows.entry(f.clone()).or_insert(0) += e - have;
            }
        }
    }

    let mut nz = NonzeroFactors::default();
    let mut combo = Vec::new();
    for (tag, _, _, d, x) in &shifts {
        if x.is_zero() {
            continue;
        }
        let c = m.mul(x).div(d)?;
        combo.push(format!("({}) * hF{tag}", txt(&c)?));
    }
    let q0_ne = nz.proof(&q0, n_s, k_s)?;
    let q1_ne = nz.proof(&q1, n_s, k_s)?;
    let m_ne = if m.is_one() {
        None
    } else {
        Some(nz.proof(&m, n_s, k_s)?)
    };
    // Only the factors the proof divides by get a hypothesis: the relations
    // are stated with denominators cleared, so their own denominators need
    // none, and every extra hypothesis would only weaken the theorem.

    let mut binders = vec![
        format!("({f_name} : ℝ → ℝ → ℝ)"),
        format!("({n_s} {k_s} : ℝ)"),
    ];
    for (tag, app, num, den, _) in &shifts {
        binders.push(format!(
            "(hF{tag} : ({}) * {app} = ({}) * {base})",
            txt(den)?,
            txt(num)?
        ));
    }
    for (text, h) in &nz.seen {
        binders.push(format!("({h} : {text} ≠ 0)"));
    }

    let mut lhs_terms = Vec::with_capacity(order + 1);
    for (j, a) in a_fp.iter().enumerate() {
        let app = if j == 0 {
            base.clone()
        } else {
            format!("{f_name} ({n_s} + {j}) {k_s}")
        };
        lhs_terms.push(format!("({}) * {app}", txt(a)?));
    }
    let lhs_text = lhs_terms.join(" + ");
    let k1 = format!("{f_name} {n_s} ({k_s} + 1)");
    let (p0_t, q0_t, p1_t, q1_t) = (txt(&p0)?, txt(&q0)?, txt(&p1)?, txt(&q1)?);
    let rhs_text = format!("({p1_t}) / ({q1_t}) * {k1} - ({p0_t}) / ({q0_t}) * {base}");

    let coeff_display: Vec<String> = result
        .coeffs
        .iter()
        .map(|&a| pool.display(a).to_string())
        .collect();
    let mut out = String::from("import Mathlib.Tactic\n\n");
    out.push_str(&format!(
        "-- Zeilberger certificate (creative telescoping), order {order}:\n\
         --   F({n_s}, {k_s}) = {}\n\
         --   a = [{}]\n\
         --   R({n_s}, {k_s}) = {},  G({n_s}, {k_s}) = R({n_s}, {k_s}) · F({n_s}, {k_s})\n\
         -- Statement: for every {f_name} satisfying the shift relations of F above\n\
         -- (hF*: denominators cleared; they hold wherever F is defined),\n\
         --   Σ_j a_j({n_s}) · {f_name}({n_s} + j, {k_s}) = G({n_s}, {k_s} + 1) − G({n_s}, {k_s}),\n\
         -- away from the zeros of the denominator factors (hN*). The recurrence for\n\
         -- the sum over {k_s} additionally needs the boundary terms; it is not claimed here.\n",
        pool.display(term),
        coeff_display.join(", "),
        pool.display(result.certificate),
    ));
    out.push_str(&format!(
        "set_option maxHeartbeats 4000000 in\n\
         example {} :\n    {lhs_text} =\n      {rhs_text} := by\n",
        binders.join("\n    "),
    ));
    out.push_str(&format!(
        "  have key : ({}) * ({lhs_text}) =\n      ({}) * {k1} - ({}) * {base} := by\n",
        txt(&lq)?,
        txt(&p1x)?,
        txt(&p0x)?,
    ));
    if let Some(m_ne) = m_ne {
        out.push_str(&format!("    apply mul_left_cancel₀ {m_ne}\n"));
    }
    let combo_text = if combo.is_empty() {
        "0".to_string()
    } else {
        combo.join(" + ")
    };
    out.push_str(&format!("    linear_combination {combo_text}\n"));
    out.push_str(&format!(
        "  rw [div_mul_eq_mul_div, div_mul_eq_mul_div, div_sub_div _ _ {q1_ne} {q0_ne},\n    \
         eq_div_iff (mul_ne_zero {q1_ne} {q0_ne})]\n  \
         linear_combination ({}) * key\n",
        txt(&gq)?,
    ));
    if contains_admission(&out) {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomic::zeilberger::{zeilberger, ZeilbergerOpts};
    use crate::kernel::Domain;

    fn binom(pool: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        let one = pool.integer(1_i32);
        let g = |e: ExprId| pool.func("gamma", vec![pool.add(vec![e, one])]);
        let a_minus_b = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        pool.mul(vec![
            g(a),
            pool.pow(g(b), pool.integer(-1_i32)),
            pool.pow(g(a_minus_b), pool.integer(-1_i32)),
        ])
    }

    fn cert_for(term: ExprId, n: ExprId, k: ExprId, pool: &ExprPool) -> String {
        let z = zeilberger(term, n, k, pool, &ZeilbergerOpts::default()).expect("zeilberger");
        emit_zeilberger_cert(term, n, k, &z.value, pool)
    }

    #[test]
    fn binomial_squared_certificate_states_the_telescoping_identity() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let b = binom(&pool, n, k);
        let term = pool.pow(b, pool.integer(2_i32));
        let src = cert_for(term, n, k, &pool);
        assert!(
            src.contains("example (F : ℝ → ℝ → ℝ)\n    (n k : ℝ)"),
            "{src}"
        );
        assert!(src.contains("(hF1 : "), "{src}");
        assert!(src.contains("(hFk : "), "{src}");
        assert!(src.contains("F (n + 1) k"), "{src}");
        assert!(src.contains("F n (k + 1)"), "{src}");
        assert!(src.contains(") * key\n"), "{src}");
        // Denominators are printed factored, one `≠ 0` hypothesis per factor.
        assert!(src.contains("(hN1 : "), "{src}");
        assert!(src.contains("(n - k + 1) ^ 2"), "{src}");
        assert!(!contains_admission(&src));
    }

    #[test]
    fn second_order_certificate_has_two_shift_hypotheses() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let b = binom(&pool, n, k);
        let term = pool.pow(b, pool.integer(3_i32)); // Franel
        let src = cert_for(term, n, k, &pool);
        assert!(src.contains("(hF2 : "), "{src}");
        assert!(src.contains("F (n + 2) k"), "{src}");
    }

    #[test]
    fn a_wrong_certificate_is_not_emitted() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let b = binom(&pool, n, k);
        let term = pool.pow(b, pool.integer(2_i32));
        let mut z = zeilberger(term, n, k, &pool, &ZeilbergerOpts::default())
            .expect("zeilberger")
            .value;
        assert!(!emit_zeilberger_cert(term, n, k, &z, &pool).is_empty());
        // Tamper with R: the exact re-check must refuse it.
        z.certificate = pool.add(vec![z.certificate, pool.integer(1_i32)]);
        assert_eq!(emit_zeilberger_cert(term, n, k, &z, &pool), "");
    }

    #[test]
    fn print_bi_is_expanded_and_sign_normalised() {
        let mut num = Bi::new();
        let mut den = Bi::new();
        num.insert((1, 0), Rational::from(1));
        num.insert((0, 0), Rational::from(1));
        den.insert((1, 0), Rational::from(-2));
        den.insert((0, 1), Rational::from(2));
        den.insert((0, 0), Rational::from(-2));
        integerize(&mut num, &mut den);
        assert_eq!(print_bi(&den, "n", "k").unwrap(), "2 * n - 2 * k + 2");
        assert_eq!(print_bi(&num, "n", "k").unwrap(), "-n - 1");
    }
}
