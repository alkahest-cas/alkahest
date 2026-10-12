//! Real-form rewriting: `exp(±i·θ)` pairs to `cos`/`sin`, `√(negative)` to
//! `i·√(positive)`, `cosh(i·x)` to `cos x`, `√(c²·k)` to `|c|·√k` — and back to
//! an expression with **no** imaginary unit when, and only when, the imaginary
//! part is proved to vanish.
//!
//! # Why this exists
//!
//! The closed forms that `matrix_exp`, `dsolve` and the inverse Laplace
//! transform build for an oscillatory system are correct as complex-analytic
//! expressions, but they are written through the eigenvalues, and for an
//! under-damped system those are complex: `exp(√(−4w²t²)/2)`,
//! `cos(√(k₁₀k₂₁ − (k₁₀+k₁₂+k₂₁)²/4)·t)`. Real-mode evaluation refuses them
//! (`E-EVAL-009`), and the JIT and the emitted C return NaN, although the
//! function they denote is real.
//!
//! # How it stays correct
//!
//! The rewrite never *drops* anything. It splits an expression `e` into
//! `re + i·im` using identities that hold for every complex value of their
//! arguments (`e^{a+ib} = e^a(cos b + i sin b)`, the addition formulas,
//! `1/(a+ib) = (a−ib)/(a²+b²)`) plus exactly one branch fact — `√X = i·√(−X)`
//! for a real `X ≤ 0` — which is used only when the sign of `X` is **proved**
//! from the symbols' domains and the stated assumptions. `re` is returned in
//! place of `e` only when `im` is proved identically zero (simplifier or the
//! exact rational normal form of [`super::zero_test`]); evidence is not
//! enough. Anything else returns `e` unchanged.
//!
//! Every free symbol other than the imaginary unit must be real (any domain
//! but [`Domain::Complex`]); an expression mentioning a complex symbol is
//! returned unchanged.
//!
//! # Signs
//!
//! [`real_sign`] decides the sign of a real expression from structure, symbol
//! domains and the ambient assumptions, and — for a polynomial in such atoms —
//! from a small certificate: each monomial that may be negative is absorbed by
//! two non-negative ones by AM–GM (`c₁m₁ + c₂m₂ ≥ 2√(c₁c₂)|m|` when
//! `m₁m₂ = m²`), and failing that the polynomial is factored over ℤ and the
//! factors are signed one at a time. It is what decides that the two-compartment
//! discriminant `(k₁₀+k₁₂+k₂₁)² − 4k₁₀k₂₁ = (k₁₀−k₂₁)² + k₁₂² + 2k₁₂(k₁₀+k₂₁)`
//! is positive, and that `4ω²(ζ²−1)` is negative under `0 < ζ < 1`.
//!
//! # Cost
//!
//! Nothing here runs inside the default simplifier. It is called explicitly
//! ([`to_real_form`]) or at the exit of the transforms named above, and there
//! only when [`looks_complex`] finds an imaginary unit or a fractional power.

use std::collections::{BTreeMap, HashMap};

use super::assumptions::{assumed_sign, assumed_weak_sign, enter_assumptions, AssumptionContext};
use super::Sign;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Sign sets
// ---------------------------------------------------------------------------

/// The set of signs a real quantity may take. Never empty in practice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SignSet {
    neg: bool,
    zero: bool,
    pos: bool,
}

impl SignSet {
    pub(crate) const POS: SignSet = SignSet {
        neg: false,
        zero: false,
        pos: true,
    };
    pub(crate) const NEG: SignSet = SignSet {
        neg: true,
        zero: false,
        pos: false,
    };
    pub(crate) const ZERO: SignSet = SignSet {
        neg: false,
        zero: true,
        pos: false,
    };
    pub(crate) const NONNEG: SignSet = SignSet {
        neg: false,
        zero: true,
        pos: true,
    };
    pub(crate) const NONPOS: SignSet = SignSet {
        neg: true,
        zero: true,
        pos: false,
    };
    pub(crate) const NONZERO: SignSet = SignSet {
        neg: true,
        zero: false,
        pos: true,
    };
    pub(crate) const ANY: SignSet = SignSet {
        neg: true,
        zero: true,
        pos: true,
    };

    fn of_sign(s: Sign) -> SignSet {
        match s {
            Sign::Negative => SignSet::NEG,
            Sign::Zero => SignSet::ZERO,
            Sign::Positive => SignSet::POS,
        }
    }

    fn of_ordering(o: std::cmp::Ordering) -> SignSet {
        match o {
            std::cmp::Ordering::Less => SignSet::NEG,
            std::cmp::Ordering::Equal => SignSet::ZERO,
            std::cmp::Ordering::Greater => SignSet::POS,
        }
    }

    /// `true` when every value in the set is `≥ 0`.
    pub(crate) fn is_nonneg(self) -> bool {
        !self.neg
    }

    /// `true` when every value in the set is `≤ 0`.
    pub(crate) fn is_nonpos(self) -> bool {
        !self.pos
    }

    /// `true` when the quantity is strictly positive.
    pub(crate) fn is_pos(self) -> bool {
        self == SignSet::POS
    }

    /// `true` when the quantity is strictly negative.
    pub(crate) fn is_neg(self) -> bool {
        self == SignSet::NEG
    }

    /// `true` when zero is excluded.
    pub(crate) fn is_nonzero(self) -> bool {
        !self.zero
    }

    /// The exact sign when the set is a singleton.
    pub(crate) fn to_sign(self) -> Option<Sign> {
        match (self.neg, self.zero, self.pos) {
            (true, false, false) => Some(Sign::Negative),
            (false, true, false) => Some(Sign::Zero),
            (false, false, true) => Some(Sign::Positive),
            _ => None,
        }
    }

    fn negate(self) -> SignSet {
        SignSet {
            neg: self.pos,
            zero: self.zero,
            pos: self.neg,
        }
    }

    /// Intersection of two independent facts about the same quantity. An
    /// empty intersection means the facts contradict each other; the first
    /// is kept rather than inventing a sign.
    fn meet(self, other: SignSet) -> SignSet {
        let m = SignSet {
            neg: self.neg && other.neg,
            zero: self.zero && other.zero,
            pos: self.pos && other.pos,
        };
        if m.neg || m.zero || m.pos {
            m
        } else {
            self
        }
    }

    fn mul(self, other: SignSet) -> SignSet {
        SignSet {
            neg: (self.neg && other.pos) || (self.pos && other.neg),
            zero: self.zero || other.zero,
            pos: (self.pos && other.pos) || (self.neg && other.neg),
        }
    }

    fn add(self, other: SignSet) -> SignSet {
        SignSet {
            neg: self.neg || other.neg,
            pos: self.pos || other.pos,
            zero: (self.zero && other.zero) || (self.pos && other.neg) || (self.neg && other.pos),
        }
    }

    /// Sign set of `x^n` for an integer `n` (`x ≠ 0` implied when `n < 0`).
    fn pow_int(self, n: i64) -> SignSet {
        if n == 0 {
            return SignSet::POS;
        }
        let zero = self.zero && n > 0;
        if n % 2 == 0 {
            SignSet {
                neg: false,
                zero,
                pos: self.pos || self.neg,
            }
        } else {
            SignSet {
                neg: self.neg,
                zero,
                pos: self.pos,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Real sign prover
// ---------------------------------------------------------------------------

const MAX_SIGN_DEPTH: u32 = 32;
/// Polynomials with more terms than this are not certified.
const MAX_CERT_TERMS: usize = 64;
/// Polynomial expansion cap (terms) for conversion.
const MAX_POLY_TERMS: usize = 512;
/// Largest exponent expanded when converting to a polynomial.
const MAX_POLY_EXP: i64 = 16;

/// The sign of a **real** expression, as far as structure, symbol domains,
/// the explicit `assumptions` and the ambient assumption scope establish it.
///
/// `None` when the expression is not known to be real-valued (it mentions a
/// complex symbol, the imaginary unit, or a function this prover does not
/// model) or when no sign is established.
pub fn real_sign(
    expr: ExprId,
    pool: &ExprPool,
    assumptions: Option<&AssumptionContext>,
) -> Option<Sign> {
    let _scope = assumptions
        .filter(|a| !a.predicates().is_empty())
        .map(|a| enter_assumptions(a, pool));
    sign_set(expr, pool).and_then(SignSet::to_sign)
}

/// Sign set of a real expression under the ambient scope; `None` when the
/// expression is not known to be real.
pub(crate) fn sign_set(expr: ExprId, pool: &ExprPool) -> Option<SignSet> {
    let mut memo = HashMap::new();
    sign_set_at(expr, pool, 0, &mut memo)
}

fn stated(expr: ExprId, pool: &ExprPool) -> SignSet {
    let mut s = SignSet::ANY;
    // Without an ambient frame `assumed_sign` only re-derives what the
    // structural rules below already know; skip its (recursive) walk.
    if !super::assumptions::ambient_assumptions_active(pool) {
        return s;
    }
    if let Some(sign) = assumed_sign(expr, pool) {
        s = s.meet(SignSet::of_sign(sign));
    }
    let (nonneg, nonpos) = assumed_weak_sign(expr, pool);
    if nonneg {
        s = s.meet(SignSet::NONNEG);
    }
    if nonpos {
        s = s.meet(SignSet::NONPOS);
    }
    s
}

fn literal_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

fn small_int(e: ExprId, pool: &ExprPool) -> Option<i64> {
    match pool.get(e) {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    }
}

fn is_half(e: ExprId, pool: &ExprPool) -> bool {
    literal_rational(e, pool).is_some_and(|r| r == rug::Rational::from((1, 2)))
}

fn sign_set_at(
    expr: ExprId,
    pool: &ExprPool,
    depth: u32,
    memo: &mut HashMap<ExprId, Option<SignSet>>,
) -> Option<SignSet> {
    if depth > MAX_SIGN_DEPTH {
        return None;
    }
    if let Some(v) = memo.get(&expr) {
        return *v;
    }
    let v = sign_set_uncached(expr, pool, depth, memo);
    memo.insert(expr, v);
    v
}

fn sign_set_uncached(
    expr: ExprId,
    pool: &ExprPool,
    depth: u32,
    memo: &mut HashMap<ExprId, Option<SignSet>>,
) -> Option<SignSet> {
    let structural = match pool.get(expr) {
        ExprData::Integer(n) => return Some(SignSet::of_ordering(n.0.cmp0())),
        ExprData::Rational(r) => return Some(SignSet::of_ordering(r.0.cmp0())),
        ExprData::Float(f) => {
            if !f.inner.is_finite() {
                return None;
            }
            return Some(SignSet::of_ordering(f.inner.cmp0()?));
        }
        ExprData::Symbol { domain, .. } => {
            if pool.is_imaginary_unit(expr) || domain == Domain::Complex {
                return None;
            }
            if crate::eval::symbols::is_pi(expr, pool) {
                return Some(SignSet::POS);
            }
            match domain {
                Domain::Positive => SignSet::POS,
                Domain::NonNegative => SignSet::NONNEG,
                Domain::NonZero => SignSet::NONZERO,
                _ => SignSet::ANY,
            }
        }
        ExprData::Mul(args) => {
            let mut acc = SignSet::POS;
            for a in args {
                acc = acc.mul(sign_set_at(a, pool, depth + 1, memo)?);
            }
            acc
        }
        ExprData::Add(args) => {
            let mut acc = SignSet::ZERO;
            for &a in &args {
                acc = acc.add(sign_set_at(a, pool, depth + 1, memo)?);
            }
            if acc.neg && acc.pos {
                if let Some(c) = poly_certificate(expr, pool, depth, memo) {
                    acc = acc.meet(c);
                }
            }
            acc
        }
        ExprData::Pow { base, exp } => {
            let b = sign_set_at(base, pool, depth + 1, memo)?;
            if let Some(n) = small_int(exp, pool) {
                b.pow_int(n)
            } else if let Some(r) = literal_rational(exp, pool) {
                // A fractional power is real only on a non-negative base.
                if !b.is_nonneg() {
                    return None;
                }
                SignSet {
                    neg: false,
                    zero: b.zero && r > 0,
                    pos: true,
                }
            } else {
                // `b^e` with a positive base and a real exponent is positive.
                sign_set_at(exp, pool, depth + 1, memo)?;
                if !b.is_pos() {
                    return None;
                }
                SignSet::POS
            }
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = sign_set_at(args[0], pool, depth + 1, memo)?;
            match name.as_str() {
                "exp" | "cosh" => SignSet::POS,
                "sinh" | "tanh" | "atan" | "asinh" | "erf" => a,
                "abs" => SignSet {
                    neg: false,
                    zero: a.zero,
                    pos: a.pos || a.neg,
                },
                "sqrt" => {
                    if !a.is_nonneg() {
                        return None;
                    }
                    a
                }
                "sign" => a,
                "heaviside" => SignSet::NONNEG,
                "sin" | "cos" | "tan" | "floor" | "ceil" => SignSet::ANY,
                "log" | "ln" => {
                    if !a.is_pos() {
                        return None;
                    }
                    SignSet::ANY
                }
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(structural.meet(stated(expr, pool)))
}

// ---------------------------------------------------------------------------
// Polynomials over ℚ in real atoms
// ---------------------------------------------------------------------------

type Exps = Vec<u32>;

/// Sparse polynomial over ℚ in an atom list owned by the caller.
#[derive(Clone, Debug, Default)]
struct QPoly {
    terms: BTreeMap<Exps, rug::Rational>,
}

fn trim(mut e: Exps) -> Exps {
    while e.last() == Some(&0) {
        e.pop();
    }
    e
}

fn add_exps(a: &Exps, b: &Exps) -> Exps {
    let n = a.len().max(b.len());
    let mut out = vec![0u32; n];
    for (i, o) in out.iter_mut().enumerate() {
        *o = a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0);
    }
    trim(out)
}

impl QPoly {
    fn constant(c: rug::Rational) -> QPoly {
        let mut terms = BTreeMap::new();
        if c != 0 {
            terms.insert(Vec::new(), c);
        }
        QPoly { terms }
    }

    fn atom(i: usize) -> QPoly {
        let mut e = vec![0u32; i + 1];
        e[i] = 1;
        let mut terms = BTreeMap::new();
        terms.insert(e, rug::Rational::from(1));
        QPoly { terms }
    }

    fn add(&self, other: &QPoly) -> QPoly {
        let mut terms = self.terms.clone();
        for (k, v) in &other.terms {
            let entry = terms.entry(k.clone()).or_default();
            *entry += v;
            if *entry == 0 {
                terms.remove(k);
            }
        }
        QPoly { terms }
    }

    fn mul(&self, other: &QPoly) -> Option<QPoly> {
        if self.terms.len() * other.terms.len() > MAX_POLY_TERMS * 8 {
            return None;
        }
        let mut terms: BTreeMap<Exps, rug::Rational> = BTreeMap::new();
        for (ka, va) in &self.terms {
            for (kb, vb) in &other.terms {
                let k = add_exps(ka, kb);
                let entry = terms.entry(k.clone()).or_default();
                *entry += rug::Rational::from(va * vb);
                if *entry == 0 {
                    terms.remove(&k);
                }
            }
        }
        if terms.len() > MAX_POLY_TERMS {
            return None;
        }
        Some(QPoly { terms })
    }

    fn pow(&self, n: u32) -> Option<QPoly> {
        let mut acc = QPoly::constant(rug::Rational::from(1));
        for _ in 0..n {
            acc = acc.mul(self)?;
        }
        Some(acc)
    }

    fn neg(&self) -> QPoly {
        QPoly {
            terms: self
                .terms
                .iter()
                .map(|(k, v)| (k.clone(), rug::Rational::from(-v)))
                .collect(),
        }
    }

    fn is_constant(&self) -> Option<rug::Rational> {
        match self.terms.len() {
            0 => Some(rug::Rational::new()),
            1 => self.terms.get(&Exps::new()).cloned(),
            _ => None,
        }
    }
}

/// Atoms of a polynomial conversion: the expressions standing for variables.
#[derive(Default)]
struct Atoms {
    list: Vec<ExprId>,
}

impl Atoms {
    fn index(&mut self, e: ExprId) -> usize {
        if let Some(i) = self.list.iter().position(|&a| a == e) {
            return i;
        }
        self.list.push(e);
        self.list.len() - 1
    }
}

/// Convert `expr` to a polynomial over ℚ in real atoms. Negative integer
/// powers become the atom `u^{-1}`; anything that is not `+`, `·`, an integer
/// power or a number becomes an atom.
fn to_qpoly(expr: ExprId, pool: &ExprPool, atoms: &mut Atoms, depth: u32) -> Option<QPoly> {
    if depth > MAX_SIGN_DEPTH {
        return None;
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(QPoly::constant(rug::Rational::from(n.0.clone()))),
        ExprData::Rational(r) => Some(QPoly::constant(r.0.clone())),
        ExprData::Float(_) => None,
        ExprData::Add(args) => {
            let mut acc = QPoly::default();
            for a in args {
                acc = acc.add(&to_qpoly(a, pool, atoms, depth + 1)?);
                if acc.terms.len() > MAX_POLY_TERMS {
                    return None;
                }
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = QPoly::constant(rug::Rational::from(1));
            for a in args {
                acc = acc.mul(&to_qpoly(a, pool, atoms, depth + 1)?)?;
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => match small_int(exp, pool) {
            Some(n) if n > 0 && n <= MAX_POLY_EXP => {
                to_qpoly(base, pool, atoms, depth + 1)?.pow(n as u32)
            }
            Some(n) if (-MAX_POLY_EXP..0).contains(&n) => {
                let inv = if n == -1 {
                    expr
                } else {
                    pool.pow(base, pool.integer(-1_i32))
                };
                QPoly::atom(atoms.index(inv)).pow((-n) as u32)
            }
            _ => Some(QPoly::atom(atoms.index(expr))),
        },
        _ => Some(QPoly::atom(atoms.index(expr))),
    }
}

fn monomial_expr(exps: &Exps, atoms: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut factors = Vec::new();
    for (i, &e) in exps.iter().enumerate() {
        if e == 0 {
            continue;
        }
        factors.push(if e == 1 {
            atoms[i]
        } else {
            pool.pow(atoms[i], pool.integer(e))
        });
    }
    match factors.len() {
        0 => pool.integer(1_i32),
        1 => factors[0],
        _ => pool.mul(factors),
    }
}

fn rational_expr(r: &rug::Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

fn qpoly_expr(p: &QPoly, atoms: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut summands = Vec::new();
    for (k, c) in &p.terms {
        if k.is_empty() {
            summands.push(rational_expr(c, pool));
            continue;
        }
        let m = monomial_expr(k, atoms, pool);
        summands.push(if *c == 1 {
            m
        } else {
            pool.mul(vec![rational_expr(c, pool), m])
        });
    }
    match summands.len() {
        0 => pool.integer(0_i32),
        1 => summands[0],
        _ => pool.add(summands),
    }
}

fn monomial_sign(exps: &Exps, signs: &[SignSet]) -> SignSet {
    let mut s = SignSet::POS;
    for (i, &e) in exps.iter().enumerate() {
        if e > 0 {
            s = s.mul(signs[i].pow_int(e as i64));
        }
    }
    s
}

/// AM–GM certificate that `p ≥ 0` (and whether `p > 0`).
///
/// Every term whose sign set is not `≥ 0` must be absorbed by a pair of
/// non-negative terms `c₁m₁`, `c₂m₂` with `m₁m₂ = m²`, using
/// `a₁m₁ + a₂m₂ ≥ 2√(a₁a₂)·|m|`, so it is enough that `c² ≤ 4a₁a₂`.
fn nonneg_certificate(p: &QPoly, signs: &[SignSet]) -> Option<SignSet> {
    if p.terms.is_empty() {
        return Some(SignSet::ZERO);
    }
    if p.terms.len() > MAX_CERT_TERMS {
        return None;
    }
    // (exponents, remaining coefficient, monomial sign set)
    let mut good: Vec<(Exps, rug::Rational, SignSet)> = Vec::new();
    let mut bad: Vec<(Exps, rug::Rational)> = Vec::new();
    for (k, c) in &p.terms {
        let ms = monomial_sign(k, signs);
        let ts = if *c > 0 { ms } else { ms.negate() };
        if ts.is_nonneg() {
            let s = if *c > 0 { ms } else { ms.negate() };
            good.push((k.clone(), rug::Rational::from(c.abs_ref()), s));
        } else {
            bad.push((k.clone(), rug::Rational::from(c.abs_ref())));
        }
    }
    for (m, c) in bad {
        let two_m: Exps = trim(m.iter().map(|&e| 2 * e).collect());
        let mut absorbed = false;
        'outer: for i in 0..good.len() {
            for j in (i + 1)..good.len() {
                if add_exps(&good[i].0, &good[j].0) != two_m {
                    continue;
                }
                let half = rug::Rational::from(&c / 2u32);
                let c2q = rug::Rational::from(&c * &c) / 4u32;
                let (c1, c2) = (good[i].1.clone(), good[j].1.clone());
                let alloc = if c1 >= half && c2 >= half {
                    Some((half.clone(), half))
                } else if c1 > 0 && rug::Rational::from(&c2q / &c1) <= c2 {
                    Some((c1.clone(), rug::Rational::from(&c2q / &c1)))
                } else if c2 > 0 && rug::Rational::from(&c2q / &c2) <= c1 {
                    Some((rug::Rational::from(&c2q / &c2), c2.clone()))
                } else {
                    None
                };
                if let Some((a1, a2)) = alloc {
                    good[i].1 -= a1;
                    good[j].1 -= a2;
                    absorbed = true;
                    break 'outer;
                }
            }
        }
        if !absorbed {
            return None;
        }
    }
    let strict = good.iter().any(|(_, c, s)| *c > 0 && s.is_pos());
    Some(if strict {
        SignSet::POS
    } else {
        SignSet::NONNEG
    })
}

/// Sign of a polynomial-shaped `expr` from the AM–GM certificate or, failing
/// that, from its factorisation over ℤ.
fn poly_certificate(
    expr: ExprId,
    pool: &ExprPool,
    depth: u32,
    memo: &mut HashMap<ExprId, Option<SignSet>>,
) -> Option<SignSet> {
    if crate::budget::check().is_err() {
        return None;
    }
    let mut atoms = Atoms::default();
    let p = to_qpoly(expr, pool, &mut atoms, 0)?;
    if p.terms.len() > MAX_CERT_TERMS {
        return None;
    }
    let mut signs = Vec::with_capacity(atoms.list.len());
    for &a in &atoms.list {
        // An atom that is itself a sum would recurse back here; a sum inside
        // an atom (`exp(a+b)`) is fine.
        signs.push(sign_set_at(a, pool, depth + 1, memo)?);
    }
    if let Some(c) = p.is_constant() {
        return Some(SignSet::of_ordering(c.cmp0()));
    }
    if let Some(s) = nonneg_certificate(&p, &signs) {
        return Some(s);
    }
    if let Some(s) = nonneg_certificate(&p.neg(), &signs) {
        return Some(s.negate());
    }
    factor_sign(&p, &atoms.list, &signs, pool)
}

/// Clear denominators: `p = q / d` with `q` over ℤ and `d > 0`.
fn to_integer_poly(p: &QPoly, atoms: &[ExprId]) -> crate::poly::MultiPoly {
    let mut lcm = rug::Integer::from(1);
    for c in p.terms.values() {
        lcm = lcm.lcm(c.denom());
    }
    let mut terms = BTreeMap::new();
    for (k, c) in &p.terms {
        let v = rug::Rational::from(c * &lcm);
        terms.insert(k.clone(), v.numer().clone());
    }
    crate::poly::MultiPoly {
        vars: atoms.to_vec(),
        terms,
    }
}

fn factor_sign(p: &QPoly, atoms: &[ExprId], signs: &[SignSet], pool: &ExprPool) -> Option<SignSet> {
    let zp = to_integer_poly(p, atoms);
    let (unit, factors) = zp.factor_irreducible()?;
    if factors.len() == 1 && factors[0].1 == 1 {
        return None;
    }
    let mut acc = SignSet::of_ordering(unit.cmp0());
    for (f, mult) in factors {
        let fq = QPoly {
            terms: f
                .terms
                .iter()
                .map(|(k, v)| (k.clone(), rug::Rational::from(v.clone())))
                .collect(),
        };
        let mut s = SignSet::ANY;
        let fe = qpoly_expr(&fq, atoms, pool);
        if fq.terms.len() == 1 {
            let (k, c) = fq.terms.iter().next()?;
            let ms = monomial_sign(k, signs);
            s = if *c > 0 { ms } else { ms.negate() };
        } else {
            if let Some(c) = nonneg_certificate(&fq, signs) {
                s = s.meet(c);
            } else if let Some(c) = nonneg_certificate(&fq.neg(), signs) {
                s = s.meet(c.negate());
            }
            if let Some(st) = assumed_sign(fe, pool) {
                s = s.meet(SignSet::of_sign(st));
            }
        }
        acc = acc.mul(s.pow_int(mult as i64));
    }
    Some(acc)
}

// ---------------------------------------------------------------------------
// Square-root extraction
// ---------------------------------------------------------------------------

/// `√X = outside · √inside` with every perfect-square factor of `X` pulled
/// out: `√(9m²k² − 4k²m²) = √5·|k|·|m|`, `√(4a²) = 2|a|` (`2a` when `a ≥ 0`).
///
/// Sound for every real `X`: each extracted factor `f^{2j}` is `≥ 0`, and
/// `√(A·B) = √A·√B` holds for `A ≥ 0` and any `B`. `None` when nothing can be
/// extracted or `X` is not a polynomial in real atoms.
fn sqrt_extract(x: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let mut atoms = Atoms::default();
    let p = to_qpoly(x, pool, &mut atoms, 0)?;
    if p.terms.is_empty() {
        return None;
    }
    let mut signs = Vec::with_capacity(atoms.list.len());
    for &a in &atoms.list {
        signs.push(sign_set(a, pool)?);
    }
    // Rational content: p = (n/d)·q with q primitive over ℤ.
    let zp = to_integer_poly(&p, &atoms.list);
    let mut lcm = rug::Integer::from(1);
    for c in p.terms.values() {
        lcm = lcm.lcm(c.denom());
    }
    let (unit, factors) = zp.factor_irreducible()?;
    // `p = unit · Π fᵢ^{eᵢ} / lcm`.
    let mut outside: Vec<ExprId> = Vec::new();
    // Constant: |unit|/lcm = (num/den); √(num/den) = √(num·den)/den.
    let num = rug::Integer::from(unit.abs_ref());
    let prod = num * &lcm;
    let (sq, rest) = square_part(&prod);
    let coeff_out = rug::Rational::from((sq, lcm.clone()));
    if coeff_out != 1 {
        outside.push(rational_expr(&coeff_out, pool));
    }
    let mut inside_const = rest;
    if unit < 0 {
        inside_const = -inside_const;
    }
    // The part left under the root, kept expanded: `√(1 − z²)`, not
    // `√(−(z − 1)(z + 1))`.
    let mut inside = QPoly::constant(rug::Rational::from(inside_const));
    let mut extracted_symbolic = false;
    for (f, mult) in factors {
        let fe = f.to_expr(pool);
        let half = mult / 2;
        let fq = QPoly {
            terms: f
                .terms
                .iter()
                .map(|(k, v)| (k.clone(), rug::Rational::from(v.clone())))
                .collect(),
        };
        if half > 0 {
            extracted_symbolic = true;
            let fs = if fq.terms.len() == 1 {
                let (k, c) = fq.terms.iter().next()?;
                let ms = monomial_sign(k, &signs);
                if *c > 0 {
                    ms
                } else {
                    ms.negate()
                }
            } else {
                sign_set(fe, pool).unwrap_or(SignSet::ANY)
            };
            // √(f^{2h}) = |f|^h; the absolute value goes when the sign of f
            // is known, or when h is even.
            let base = if fs.is_nonneg() || half % 2 == 0 {
                fe
            } else if fs.is_nonpos() {
                pool.mul(vec![pool.integer(-1_i32), fe])
            } else {
                pool.func("abs", vec![fe])
            };
            outside.push(if half == 1 {
                base
            } else {
                pool.pow(base, pool.integer(half))
            });
        }
        if mult % 2 == 1 {
            inside = inside.mul(&fq)?;
        }
    }
    if outside.is_empty() || (!extracted_symbolic && coeff_out == 1) {
        return None;
    }
    let out = match outside.len() {
        1 => outside[0],
        _ => pool.mul(outside),
    };
    Some((out, qpoly_expr(&inside, &atoms.list, pool)))
}

/// `n = s²·r` with `r` square-free (trial division up to a small bound; a
/// large cofactor stays in `r`, which is still correct, just less reduced).
fn square_part(n: &rug::Integer) -> (rug::Integer, rug::Integer) {
    let mut s = rug::Integer::from(1);
    let mut r = n.clone();
    if r == 0 {
        return (rug::Integer::from(0), rug::Integer::from(1));
    }
    if r.is_perfect_square() {
        return (r.sqrt(), rug::Integer::from(1));
    }
    let mut p = 2u32;
    while p < 1000 {
        let pp = rug::Integer::from(p) * p;
        while r.is_divisible(&pp) {
            r /= &pp;
            s *= p;
        }
        p += 1;
    }
    (s, r)
}

// ---------------------------------------------------------------------------
// Complex split
// ---------------------------------------------------------------------------

/// Quick scan: does `expr` contain the imaginary unit, a fractional power, a
/// `sqrt` or an `abs`? The transforms skip the rewrite when it does not.
pub fn looks_complex(expr: ExprId, pool: &ExprPool) -> bool {
    let mut seen = std::collections::HashSet::new();
    looks_complex_at(expr, pool, &mut seen)
}

fn looks_complex_at(
    expr: ExprId,
    pool: &ExprPool,
    seen: &mut std::collections::HashSet<ExprId>,
) -> bool {
    if !seen.insert(expr) {
        return false;
    }
    match pool.get(expr) {
        ExprData::Symbol { .. } => pool.is_imaginary_unit(expr),
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| looks_complex_at(a, pool, seen))
        }
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Rational(_))
                || looks_complex_at(base, pool, seen)
                || looks_complex_at(exp, pool, seen)
        }
        ExprData::Func { name, args } => {
            name == "sqrt" || name == "abs" || args.iter().any(|&a| looks_complex_at(a, pool, seen))
        }
        _ => false,
    }
}

/// `true` when every free symbol except the imaginary unit is real.
fn all_symbols_real(expr: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![expr];
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        match pool.get(e) {
            ExprData::Symbol { domain, .. } => {
                if domain == Domain::Complex && !pool.is_imaginary_unit(e) {
                    return false;
                }
            }
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
                stack.extend(args)
            }
            ExprData::Pow { base, exp } => {
                stack.push(base);
                stack.push(exp);
            }
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => {}
            // Binders, predicates, piecewise: left to the caller unchanged.
            _ => return false,
        }
    }
    true
}

/// `(re, im)` with `im = None` meaning structurally zero.
type Parts = (ExprId, Option<ExprId>);

struct Splitter<'a> {
    pool: &'a ExprPool,
    changed: bool,
    memo: HashMap<ExprId, Parts>,
    nodes: usize,
}

const MAX_SPLIT_NODES: usize = 20_000;

impl<'a> Splitter<'a> {
    fn zero(&self) -> ExprId {
        self.pool.integer(0_i32)
    }

    fn neg(&self, e: ExprId) -> ExprId {
        self.pool.mul(vec![self.pool.integer(-1_i32), e])
    }

    fn mul_parts(&self, a: Parts, b: Parts) -> Parts {
        let p = self.pool;
        match (a.1, b.1) {
            (None, None) => (p.mul(vec![a.0, b.0]), None),
            (Some(ai), None) => (p.mul(vec![a.0, b.0]), Some(p.mul(vec![ai, b.0]))),
            (None, Some(bi)) => (p.mul(vec![a.0, b.0]), Some(p.mul(vec![a.0, bi]))),
            (Some(ai), Some(bi)) => (
                p.add(vec![p.mul(vec![a.0, b.0]), self.neg(p.mul(vec![ai, bi]))]),
                Some(p.add(vec![p.mul(vec![a.0, bi]), p.mul(vec![ai, b.0])])),
            ),
        }
    }

    /// `1/(a + ib) = (a − ib)/(a² + b²)`.
    fn recip_parts(&self, z: Parts) -> Parts {
        let p = self.pool;
        match z.1 {
            None => (p.pow(z.0, p.integer(-1_i32)), None),
            Some(b) => {
                let two = p.integer(2_i32);
                let den = p.add(vec![p.pow(z.0, two), p.pow(b, two)]);
                let inv = p.pow(den, p.integer(-1_i32));
                (p.mul(vec![z.0, inv]), Some(self.neg(p.mul(vec![b, inv]))))
            }
        }
    }

    fn simp(&self, e: ExprId) -> ExprId {
        super::engine::simplify(e, self.pool).value
    }

    /// `f(arg)` for `exp` or a trig/hyperbolic `f`, with a simplified `arg`;
    /// for the even (`cos`, `cosh`) and odd (`sin`, `sinh`) ones a negative
    /// literal coefficient is pulled out of the argument.
    fn fn_of(&self, name: &str, arg: ExprId) -> ExprId {
        let p = self.pool;
        let arg = self.simp(arg);
        let has_parity = matches!(name, "cos" | "cosh" | "sin" | "sinh");
        if let (true, ExprData::Mul(xs)) = (has_parity, p.get(arg)) {
            let lit = xs
                .iter()
                .enumerate()
                .find_map(|(i, &c)| literal_rational(c, p).map(|v| (i, v)));
            if let Some((i, c)) = lit {
                if c < 0 {
                    let mut rest = xs.clone();
                    rest[i] = rational_expr(&(-c), p);
                    let pos = self.simp(p.mul(rest));
                    let inner = p.func(name, vec![pos]);
                    return match name {
                        "cos" | "cosh" => inner,
                        _ => self.neg(inner),
                    };
                }
            }
        }
        if let Some(c) = literal_rational(arg, p) {
            if c == 0 {
                return match name {
                    "cos" | "cosh" | "exp" => p.integer(1_i32),
                    _ => p.integer(0_i32),
                };
            }
        }
        p.func(name, vec![arg])
    }

    fn split(&mut self, e: ExprId) -> Parts {
        if let Some(&v) = self.memo.get(&e) {
            return v;
        }
        self.nodes += 1;
        let v = if self.nodes > MAX_SPLIT_NODES || crate::budget::check().is_err() {
            (e, None)
        } else {
            self.split_uncached(e)
        };
        self.memo.insert(e, v);
        v
    }

    fn split_uncached(&mut self, e: ExprId) -> Parts {
        let p = self.pool;
        match p.get(e) {
            ExprData::Symbol { .. } if p.is_imaginary_unit(e) => {
                self.changed = true;
                (self.zero(), Some(p.integer(1_i32)))
            }
            ExprData::Add(args) => {
                let parts: Vec<Parts> = args.iter().map(|&a| self.split(a)).collect();
                let re = p.add(parts.iter().map(|x| x.0).collect());
                let ims: Vec<ExprId> = parts.iter().filter_map(|x| x.1).collect();
                let im = match ims.len() {
                    0 => None,
                    1 => Some(ims[0]),
                    _ => Some(p.add(ims)),
                };
                (re, im)
            }
            ExprData::Mul(args) => {
                let mut acc: Parts = (p.integer(1_i32), None);
                for a in args {
                    let x = self.split(a);
                    acc = self.mul_parts(acc, x);
                }
                acc
            }
            ExprData::Pow { base, exp } => self.split_pow(e, base, exp),
            ExprData::Func { name, args } if args.len() == 1 => {
                let a = self.split(args[0]);
                match (name.as_str(), a.1) {
                    ("sqrt", None) => self.split_sqrt(e, args[0], a.0),
                    ("exp", Some(b)) => {
                        self.changed = true;
                        let ea = self.fn_of("exp", a.0);
                        (
                            p.mul(vec![ea, self.fn_of("cos", b)]),
                            Some(p.mul(vec![ea, self.fn_of("sin", b)])),
                        )
                    }
                    ("cos", Some(b)) => {
                        // cos(a+ib) = cos a cosh b − i sin a sinh b
                        self.changed = true;
                        (
                            p.mul(vec![self.fn_of("cos", a.0), self.fn_of("cosh", b)]),
                            Some(
                                self.neg(
                                    p.mul(vec![self.fn_of("sin", a.0), self.fn_of("sinh", b)]),
                                ),
                            ),
                        )
                    }
                    ("sin", Some(b)) => {
                        // sin(a+ib) = sin a cosh b + i cos a sinh b
                        self.changed = true;
                        (
                            p.mul(vec![self.fn_of("sin", a.0), self.fn_of("cosh", b)]),
                            Some(p.mul(vec![self.fn_of("cos", a.0), self.fn_of("sinh", b)])),
                        )
                    }
                    ("cosh", Some(b)) => {
                        // cosh(a+ib) = cosh a cos b + i sinh a sin b
                        self.changed = true;
                        (
                            p.mul(vec![self.fn_of("cosh", a.0), self.fn_of("cos", b)]),
                            Some(p.mul(vec![self.fn_of("sinh", a.0), self.fn_of("sin", b)])),
                        )
                    }
                    ("sinh", Some(b)) => {
                        // sinh(a+ib) = sinh a cos b + i cosh a sin b
                        self.changed = true;
                        (
                            p.mul(vec![self.fn_of("sinh", a.0), self.fn_of("cos", b)]),
                            Some(p.mul(vec![self.fn_of("cosh", a.0), self.fn_of("sin", b)])),
                        )
                    }
                    (_, None) => {
                        if a.0 == args[0] {
                            (e, None)
                        } else {
                            (p.func(name.as_str(), vec![a.0]), None)
                        }
                    }
                    // Anything else with a complex argument stays as it was.
                    (_, Some(_)) => (e, None),
                }
            }
            ExprData::Func { name, args } => {
                let parts: Vec<Parts> = args.iter().map(|&a| self.split(a)).collect();
                if parts.iter().all(|x| x.1.is_none()) {
                    let new: Vec<ExprId> = parts.iter().map(|x| x.0).collect();
                    if new == args {
                        (e, None)
                    } else {
                        (p.func(name.as_str(), new), None)
                    }
                } else {
                    (e, None)
                }
            }
            _ => (e, None),
        }
    }

    fn split_pow(&mut self, e: ExprId, base: ExprId, exp: ExprId) -> Parts {
        let p = self.pool;
        if let Some(n) = small_int(exp, p) {
            let b = self.split(base);
            if b.1.is_none() {
                return if b.0 == base {
                    (e, None)
                } else {
                    (p.pow(b.0, exp), None)
                };
            }
            if n.unsigned_abs() > 8 {
                return (e, None);
            }
            let z = if n < 0 { self.recip_parts(b) } else { b };
            let mut acc: Parts = (p.integer(1_i32), None);
            for _ in 0..n.unsigned_abs() {
                acc = self.mul_parts(acc, z);
            }
            return acc;
        }
        if let Some(r) = literal_rational(exp, p) {
            // X^{k/2} = (√X)^k for an odd k.
            if *r.denom() == 2 {
                let k = r.numer().to_i64().unwrap_or(i64::MAX);
                if k.unsigned_abs() <= 8 {
                    let b = self.split(base);
                    if b.1.is_some() {
                        return (e, None);
                    }
                    let root = if is_half(exp, p) {
                        self.split_sqrt(e, base, b.0)
                    } else {
                        let sq = p.pow(b.0, p.rational(1_i32, 2_i32));
                        self.split_sqrt(sq, b.0, b.0)
                    };
                    if k == 1 {
                        return root;
                    }
                    let z = if k < 0 { self.recip_parts(root) } else { root };
                    let mut acc: Parts = (p.integer(1_i32), None);
                    for _ in 0..k.unsigned_abs() {
                        acc = self.mul_parts(acc, z);
                    }
                    return acc;
                }
            }
            let b = self.split(base);
            return if b.1.is_none() && b.0 != base {
                (p.pow(b.0, exp), None)
            } else {
                (e, None)
            };
        }
        // Symbolic exponent: `b^{c+id} = b^c (cos(d ln b) + i sin(d ln b))`
        // for a positive real base.
        let b = self.split(base);
        let x = self.split(exp);
        match (b.1, x.1) {
            (None, Some(d)) if sign_set(b.0, p).is_some_and(SignSet::is_pos) => {
                self.changed = true;
                let mag = p.pow(b.0, x.0);
                let ang = p.mul(vec![d, p.func("log", vec![b.0])]);
                (
                    p.mul(vec![mag, self.fn_of("cos", ang)]),
                    Some(p.mul(vec![mag, self.fn_of("sin", ang)])),
                )
            }
            (None, None) if b.0 != base || x.0 != exp => (p.pow(b.0, x.0), None),
            _ => (e, None),
        }
    }

    /// `√X` for a real `X`: perfect squares out, and `i·√(−X)` when `X ≤ 0`
    /// is proved. `orig` is returned when nothing applies.
    fn split_sqrt(&mut self, orig: ExprId, orig_radicand: ExprId, x: ExprId) -> Parts {
        let p = self.pool;
        let (out, inside, extracted) = match sqrt_extract(x, p) {
            Some((o, i)) => (o, i, true),
            None => (p.integer(1_i32), x, false),
        };
        let inside_sign = sign_set(inside, p);
        let mk = |q: ExprId| -> ExprId {
            if literal_rational(q, p).is_some_and(|r| r == 1) {
                p.integer(1_i32)
            } else {
                p.func("sqrt", vec![q])
            }
        };
        match inside_sign {
            Some(s) if s.is_nonpos() && !s.is_nonneg() => {
                self.changed = true;
                let neg_inside = self.simp(self.neg(inside));
                (self.zero(), Some(p.mul(vec![out, mk(neg_inside)])))
            }
            Some(s) if s == SignSet::ZERO => {
                self.changed = true;
                (self.zero(), None)
            }
            _ if extracted => {
                self.changed = true;
                (p.mul(vec![out, mk(inside)]), None)
            }
            _ => {
                if x == orig_radicand {
                    (orig, None)
                } else {
                    (p.func("sqrt", vec![x]), None)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Rewrite `expr` into an equal expression free of the imaginary unit and of
/// square roots of negative quantities, under `assumptions`.
///
/// * `exp(a ± i·θ)` → `e^a·(cos θ ± i·sin θ)`; `cos`/`sin`/`cosh`/`sinh` of a
///   complex argument by the addition formulas (`cosh(i·x)` → `cos x`);
/// * `√X` with `X ≤ 0` proved → `i·√(−X)`, after perfect-square factors are
///   taken out: `√(c²·k)` → `|c|·√k`, and `→ c·√k` once `c ≥ 0` is known;
/// * the imaginary parts so produced are collected, and the real part is
///   returned **only when the imaginary part is proved identically zero**.
/// * finally `cos(|u|)` → `cos u`, `sin(|u|)/|u|` → `sin u/u`, … — every
///   `|u|` the result is an even function of is replaced by `u`.
///
/// Every free symbol must be real (any domain but [`Domain::Complex`]); with
/// a complex symbol, or when no imaginary part could be proved to vanish,
/// `expr` is returned unchanged. The result is never a different function
/// of the symbols: on the set the assumptions describe it is equal to `expr`.
pub fn to_real_form(expr: ExprId, pool: &ExprPool, assumptions: &AssumptionContext) -> ExprId {
    let _scope = if assumptions.predicates().is_empty() {
        None
    } else {
        Some(enter_assumptions(assumptions, pool))
    };
    to_real_form_ambient(expr, pool)
}

/// [`to_real_form`] under the ambient assumption scope only — the form the
/// transforms call at their exit.
pub(crate) fn to_real_form_ambient(expr: ExprId, pool: &ExprPool) -> ExprId {
    if !looks_complex(expr, pool) || !all_symbols_real(expr, pool) {
        return expr;
    }
    let mut sp = Splitter {
        pool,
        changed: false,
        memo: HashMap::new(),
        nodes: 0,
    };
    let (re, im) = sp.split(expr);
    if !sp.changed {
        return expr;
    }
    if let Some(im) = im {
        if !imaginary_part_vanishes(im, pool) {
            return expr;
        }
    }
    let re = simplify_ambient(re, pool);
    let out = drop_even_abs(re, pool);
    if out == expr {
        expr
    } else {
        out
    }
}

fn simplify_ambient(e: ExprId, pool: &ExprPool) -> ExprId {
    super::engine::simplify(e, pool).value
}

fn imaginary_part_vanishes(im: ExprId, pool: &ExprPool) -> bool {
    let s = simplify_ambient(im, pool);
    if literal_rational(s, pool).is_some_and(|r| r == 0) {
        return true;
    }
    matches!(
        super::zero::zero_test(im, pool, None).status,
        super::zero::ZeroStatus::ProvedZero
    )
}

/// Replace `|u|` by `u` wherever the expression is an even function of `|u|`.
fn drop_even_abs(expr: ExprId, pool: &ExprPool) -> ExprId {
    let mut abs_nodes: Vec<ExprId> = Vec::new();
    collect_abs(
        expr,
        pool,
        &mut abs_nodes,
        &mut std::collections::HashSet::new(),
    );
    if abs_nodes.is_empty() || abs_nodes.len() > 4 {
        return expr;
    }
    let mut cur = expr;
    for (i, node) in abs_nodes.into_iter().enumerate() {
        let ExprData::Func { args, .. } = pool.get(node) else {
            continue;
        };
        let u = args[0];
        // `|u| = ±u` needs a real `u`.
        if sign_set(u, pool).is_none() {
            continue;
        }
        let z = pool.symbol(format!("__realform_abs_{i}"), Domain::Real);
        let mut m = HashMap::new();
        m.insert(node, z);
        let ez = crate::kernel::subs(cur, &m, pool);
        if crate::budget::check().is_err() {
            return cur;
        }
        let mut flip = HashMap::new();
        flip.insert(z, pool.mul(vec![pool.integer(-1_i32), z]));
        let emz = crate::kernel::subs(ez, &flip, pool);
        let diff = pool.add(vec![ez, pool.mul(vec![pool.integer(-1_i32), emz])]);
        if !imaginary_part_vanishes(diff, pool) {
            continue;
        }
        let mut back = HashMap::new();
        back.insert(z, u);
        cur = simplify_ambient(crate::kernel::subs(ez, &back, pool), pool);
    }
    cur
}

fn collect_abs(
    e: ExprId,
    pool: &ExprPool,
    out: &mut Vec<ExprId>,
    seen: &mut std::collections::HashSet<ExprId>,
) {
    if !seen.insert(e) {
        return;
    }
    match pool.get(e) {
        ExprData::Func { name, args } => {
            if name == "abs" && args.len() == 1 && !out.contains(&e) {
                out.push(e);
            }
            for a in args {
                collect_abs(a, pool, out, seen);
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect_abs(a, pool, out, seen);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_abs(base, pool, out, seen);
            collect_abs(exp, pool, out, seen);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
