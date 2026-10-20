//! The exact differential field the tensor layer computes in.
//!
//! Every metric component is converted once into a quotient `N/D` of integer
//! polynomials (FLINT `fmpz_mpoly`) over a fixed list of *generators*:
//!
//! * the coordinates, and every other free symbol (a parameter);
//! * for each trigonometric argument `v`, the pair `s = sin v`, `c = cos v`
//!   with the relation `s² = 1 − c²` (hyperbolic: `sh² = ch² − 1`).  `sin(n v)`
//!   for an integer `n` is expanded by the multiple-angle recurrence, and
//!   `tan`/`cot`/`sec`/`csc` become quotients;
//! * `exp(v)` (with `exp(n v) = exp(v)ⁿ` for integer `n`), `log(v)`, and
//!   radicals `b^(1/q)`; a square root `w = √b` whose radicand is free of the
//!   other algebraic generators carries the relation `w² = b`;
//! * the imaginary unit, with `i² = −1`;
//! * anything else (an undefined function `a(t)`, its `Derivative` nodes, …)
//!   as an opaque atom whose partial derivatives come from [`crate::diff::diff`].
//!
//! Partial derivatives are a derivation on this field: `∂/∂x` acts on each
//! generator through its own rule (`∂s/∂x = c·∂v/∂x`, `∂w/∂x = w·∂b/(2b)`, …),
//! and those rules respect every relation (`∂(s² + c² − 1) = 0`), so the
//! derivation is well defined on the quotient ring.
//!
//! # Canonical form
//!
//! After every operation the numerator is reduced modulo the quadratic
//! relations (so it is at most linear in each of `s`, `w`, `i`), the
//! denominator is made free of those generators by multiplying through by its
//! conjugate (`(A + sB)(A − sB) = A² − (1 − c²)B²`), and the common factor is
//! cancelled with a FLINT gcd; the denominator's leading coefficient is made
//! positive.  The polynomials linear in the algebraic generators are a basis
//! of the quotient ring over the transcendental ones, so when that ring is an
//! integral domain (always, for the trigonometric, hyperbolic and `i`
//! relations and for a non-square radicand), **two field elements are equal
//! if and only if their canonical forms are identical** — in particular an
//! element is zero exactly when its numerator is the zero polynomial.  This is
//! the exact zero test that proves, e.g., that the Kerr metric is Ricci-flat.
//!
//! Soundness never depends on that domain condition: every rewrite applied is
//! an identity of the functions involved, so a value computed here is always
//! the right value.  What could fail on a square radicand is only the
//! *completeness* of the zero test (a zero not recognised).

use super::TensorError;
use crate::diff::diff;
use crate::flint::mpoly::{FlintMPoly, FlintMPolyCtx, FlintMPolyFactor};
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Integer;
use std::collections::HashMap;
use std::sync::Arc;

/// Above this many terms an output polynomial is printed expanded rather
/// than factored (multivariate factoring is not free).
const FACTOR_TERM_LIMIT: usize = 4000;

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// What a field generator stands for.
#[derive(Clone, Debug)]
pub(crate) enum VarKind {
    /// The `i`-th coordinate.
    Coord(usize),
    /// A free symbol other than a coordinate (constant under `∂`).
    Param,
    /// The imaginary unit, `i² = −1`.
    Imag,
    /// `sin v` (or `sinh v`); `partner` is the matching cosine.
    Sin {
        arg: ExprId,
        hyperbolic: bool,
        partner: usize,
    },
    /// `cos v` (or `cosh v`); `partner` is the matching sine.
    Cos {
        arg: ExprId,
        hyperbolic: bool,
        partner: usize,
    },
    /// `b^(1/q)`.
    Root { radicand: ExprId, q: u32 },
    /// `exp v`.
    Exp { arg: ExprId },
    /// `log v`.
    Log { arg: ExprId },
    /// Opaque: partial derivatives from `diff`, one per coordinate.
    Atom,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    Sym(ExprId),
    Sin(ExprId, bool),
    Cos(ExprId, bool),
    Root(ExprId, u32),
    Exp(ExprId),
    Log(ExprId),
    Atom(ExprId),
}

#[derive(Clone, Debug)]
struct Var {
    kind: VarKind,
    /// The expression this generator prints as.
    expr: ExprId,
    /// For [`VarKind::Atom`]: `∂atom/∂x_j`, or the reason it is unavailable.
    dexprs: Option<Vec<Result<ExprId, String>>>,
}

/// A field element `num / den` in canonical form (see the module docs).
#[derive(Clone)]
pub(crate) struct Rf {
    pub(crate) num: FlintMPoly,
    pub(crate) den: FlintMPoly,
}

impl Rf {
    pub(crate) fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
}

/// How a generator differentiates with respect to one coordinate.
#[derive(Clone)]
enum DVal {
    Zero,
    One,
    Val(Rf),
    Missing(String),
}

// ---------------------------------------------------------------------------
// Scanning (phase 1): discover every generator
// ---------------------------------------------------------------------------

/// Collects the generators of the field before its FLINT context is created.
pub(crate) struct FieldBuilder<'p> {
    pool: &'p ExprPool,
    coords: Vec<ExprId>,
    vars: Vec<Var>,
    index: HashMap<Key, usize>,
    /// Expression → the derivative depth it has been scanned to.
    seen: HashMap<ExprId, usize>,
}

const TRIG: &[(&str, bool)] = &[
    ("sin", false),
    ("cos", false),
    ("tan", false),
    ("cot", false),
    ("sec", false),
    ("csc", false),
    ("sinh", true),
    ("cosh", true),
    ("tanh", true),
    ("coth", true),
    ("sech", true),
    ("csch", true),
];

fn trig_kind(name: &str) -> Option<bool> {
    TRIG.iter().find(|(n, _)| *n == name).map(|(_, h)| *h)
}

/// `u = n·v` with an integer `n ≠ 0`: returns `(n, v)` (`(1, u)` otherwise).
fn split_int_multiple(u: ExprId, pool: &ExprPool) -> (i64, ExprId) {
    if let ExprData::Mul(args) = pool.get(u) {
        let ints: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| matches!(pool.get(**a), ExprData::Integer(_)))
            .map(|(i, _)| i)
            .collect();
        if ints.len() == 1 {
            if let ExprData::Integer(n) = pool.get(args[ints[0]]) {
                if let Some(k) = n.0.to_i64() {
                    if k != 0 && k.unsigned_abs() <= 64 {
                        let rest: Vec<ExprId> = args
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != ints[0])
                            .map(|(_, a)| *a)
                            .collect();
                        let v = if rest.len() == 1 {
                            rest[0]
                        } else {
                            pool.mul(rest)
                        };
                        return (k, v);
                    }
                }
            }
        }
    }
    (1, u)
}

impl<'p> FieldBuilder<'p> {
    /// Start a builder whose first `coords.len()` generators are the coordinates.
    pub(crate) fn new(coords: &[ExprId], pool: &'p ExprPool) -> Self {
        let mut b = FieldBuilder {
            pool,
            coords: coords.to_vec(),
            vars: Vec::new(),
            index: HashMap::new(),
            seen: HashMap::new(),
        };
        for (i, &c) in coords.iter().enumerate() {
            b.index.insert(Key::Sym(c), i);
            b.vars.push(Var {
                kind: VarKind::Coord(i),
                expr: c,
                dexprs: None,
            });
        }
        b
    }

    fn register(&mut self, key: Key, kind: VarKind, expr: ExprId) -> usize {
        if let Some(&i) = self.index.get(&key) {
            return i;
        }
        let i = self.vars.len();
        self.vars.push(Var {
            kind,
            expr,
            dexprs: None,
        });
        self.index.insert(key, i);
        i
    }

    fn register_trig(&mut self, v: ExprId, hyperbolic: bool) {
        if self.index.contains_key(&Key::Sin(v, hyperbolic)) {
            return;
        }
        let pool = self.pool;
        let (sn, cn) = if hyperbolic {
            ("sinh", "cosh")
        } else {
            ("sin", "cos")
        };
        let s = self.vars.len();
        let c = s + 1;
        self.register(
            Key::Sin(v, hyperbolic),
            VarKind::Sin {
                arg: v,
                hyperbolic,
                partner: c,
            },
            pool.func(sn, vec![v]),
        );
        self.register(
            Key::Cos(v, hyperbolic),
            VarKind::Cos {
                arg: v,
                hyperbolic,
                partner: s,
            },
            pool.func(cn, vec![v]),
        );
    }

    /// Register every generator `e` needs, with atoms differentiable to `depth`.
    pub(crate) fn scan(&mut self, e: ExprId, depth: usize) -> Result<(), TensorError> {
        if let Some(&d) = self.seen.get(&e) {
            if d >= depth {
                return Ok(());
            }
        }
        self.seen.insert(e, depth);
        let pool = self.pool;
        match pool.get(e) {
            ExprData::Symbol { .. } => {
                if self.coords.contains(&e) {
                    return Ok(());
                }
                if pool.is_imaginary_unit(e) {
                    self.register(Key::Sym(e), VarKind::Imag, e);
                } else {
                    self.register(Key::Sym(e), VarKind::Param, e);
                }
                Ok(())
            }
            ExprData::Integer(_) | ExprData::Rational(_) => Ok(()),
            ExprData::Float(_) => Err(TensorError::InexactInput {
                expr: pool.display(e).to_string(),
            }),
            ExprData::Add(args) | ExprData::Mul(args) => {
                for a in args {
                    self.scan(a, depth)?;
                }
                Ok(())
            }
            ExprData::Pow { base, exp } => match pool.get(exp) {
                ExprData::Integer(_) => self.scan(base, depth),
                ExprData::Rational(r) => {
                    let q = r.0.denom().to_u32().filter(|&q| (2..=64).contains(&q));
                    match q {
                        Some(q) => {
                            self.scan(base, depth)?;
                            let ex = pool.pow(base, pool.rational(1, q));
                            self.register(
                                Key::Root(base, q),
                                VarKind::Root { radicand: base, q },
                                ex,
                            );
                            Ok(())
                        }
                        None => self.scan_atom(e, depth),
                    }
                }
                _ => self.scan_atom(e, depth),
            },
            ExprData::Func { name, args } if args.len() == 1 => {
                let a = args[0];
                if let Some(h) = trig_kind(&name) {
                    let (_, v) = split_int_multiple(a, pool);
                    self.scan(v, depth)?;
                    self.register_trig(v, h);
                    return Ok(());
                }
                match name.as_str() {
                    "exp" => {
                        let (_, v) = split_int_multiple(a, pool);
                        self.scan(v, depth)?;
                        let ex = pool.func("exp", vec![v]);
                        self.register(Key::Exp(v), VarKind::Exp { arg: v }, ex);
                        Ok(())
                    }
                    "log" => {
                        self.scan(a, depth)?;
                        self.register(Key::Log(a), VarKind::Log { arg: a }, e);
                        Ok(())
                    }
                    "sqrt" => {
                        self.scan(a, depth)?;
                        let ex = pool.pow(a, pool.rational(1, 2));
                        self.register(Key::Root(a, 2), VarKind::Root { radicand: a, q: 2 }, ex);
                        Ok(())
                    }
                    _ => self.scan_atom(e, depth),
                }
            }
            ExprData::Func { .. } => self.scan_atom(e, depth),
            _ => Err(TensorError::UnsupportedExpression {
                expr: pool.display(e).to_string(),
            }),
        }
    }

    /// An opaque generator; its partial derivatives come from `diff`.
    fn scan_atom(&mut self, e: ExprId, depth: usize) -> Result<(), TensorError> {
        let pool = self.pool;
        let i = self.register(Key::Atom(e), VarKind::Atom, e);
        if depth == 0 || self.vars[i].dexprs.is_some() {
            if let Some(ds) = self.vars[i].dexprs.clone() {
                for d in ds.into_iter().flatten() {
                    self.scan(d, depth.saturating_sub(1))?;
                }
            }
            return Ok(());
        }
        let mut ds = Vec::with_capacity(self.coords.len());
        for &x in &self.coords.clone() {
            match diff(e, x, pool) {
                Ok(d) => ds.push(Ok(d.value)),
                Err(err) => ds.push(Err(err.to_string())),
            }
        }
        self.vars[i].dexprs = Some(ds.clone());
        for d in ds.into_iter().flatten() {
            self.scan(d, depth - 1)?;
        }
        Ok(())
    }

    /// Create the FLINT context and the derivative tables.
    pub(crate) fn build(self) -> Result<DiffField, TensorError> {
        let pool = self.pool;
        let nvars = self.vars.len().max(1);
        let ctx = FlintMPolyCtx::new(nvars);
        let mut field = DiffField {
            ctx,
            nvars,
            ncoords: self.coords.len(),
            vars: self.vars,
            index: self.index,
            relations: Vec::new(),
            dtab: Vec::new(),
            memo: HashMap::new(),
        };
        field.init_tables(pool)?;
        Ok(field)
    }
}

// ---------------------------------------------------------------------------
// The field (phase 2)
// ---------------------------------------------------------------------------

/// The differential field over the scanned generators.
pub(crate) struct DiffField {
    ctx: Arc<FlintMPolyCtx>,
    nvars: usize,
    ncoords: usize,
    vars: Vec<Var>,
    index: HashMap<Key, usize>,
    /// `(q, rhs)`: the relation `q² = rhs`, `rhs` free of every relation generator.
    relations: Vec<(usize, Rf)>,
    /// `dtab[k][j] = ∂(generator k)/∂(coordinate j)`.
    dtab: Vec<Vec<DVal>>,
    memo: HashMap<ExprId, Rf>,
}

fn budget() -> Result<(), TensorError> {
    crate::budget::check_all().map_err(TensorError::Budget)
}

impl DiffField {
    // ----- polynomial helpers -------------------------------------------

    fn poly_const(&self, c: &Integer) -> FlintMPoly {
        let mut p = FlintMPoly::new(Arc::clone(&self.ctx));
        if *c != 0 {
            p.push_term(c, &vec![0u64; self.nvars]);
            p.finish();
        }
        p
    }

    fn poly_var(&self, k: usize) -> FlintMPoly {
        let mut e = vec![0u64; self.nvars];
        e[k] = 1;
        let mut p = FlintMPoly::new(Arc::clone(&self.ctx));
        p.push_term(&Integer::from(1), &e);
        p.finish();
        p
    }

    fn zero_poly(&self) -> FlintMPoly {
        FlintMPoly::new(Arc::clone(&self.ctx))
    }

    fn one_poly(&self) -> FlintMPoly {
        self.poly_const(&Integer::from(1))
    }

    fn neg_poly(&self, p: &FlintMPoly) -> FlintMPoly {
        self.zero_poly().sub(p)
    }

    fn is_one_poly(&self, p: &FlintMPoly) -> bool {
        p.length() == 1 && p.equal(&self.one_poly())
    }

    /// `p = Σ_e q^e P_e`: the coefficients `P_e` (index = exponent).
    fn split_by_var(&self, p: &FlintMPoly, q: usize) -> Result<Vec<FlintMPoly>, TensorError> {
        let deg = p.degree_in(q).max(0) as usize;
        let terms = p.terms_u64().ok_or(TensorError::Internal(
            "polynomial exponent overflow".to_string(),
        ))?;
        let mut parts: Vec<FlintMPoly> = (0..=deg).map(|_| self.zero_poly()).collect();
        for (mut exp, coeff) in terms {
            exp.resize(self.nvars, 0);
            let e = exp[q] as usize;
            exp[q] = 0;
            parts[e].push_term(&coeff, &exp);
        }
        for part in &mut parts {
            part.finish();
        }
        Ok(parts)
    }

    /// Reduce `p` modulo every relation: returns `(r, s)` with `p = r / s`
    /// on the variety, `r` of degree ≤ 1 in every relation generator and `s`
    /// a product of relation denominators.
    fn reduce_poly(&self, p: &FlintMPoly) -> Result<(FlintMPoly, FlintMPoly), TensorError> {
        let mut cur = p.clone();
        let mut mult = self.one_poly();
        for (q, rhs) in &self.relations {
            if cur.degree_in(*q) < 2 {
                continue;
            }
            let parts = self.split_by_var(&cur, *q)?;
            let big_m = (parts.len() - 1) / 2;
            let qp = self.poly_var(*q);
            let sden_trivial = self.is_one_poly(&rhs.den);
            // Powers of R and S.
            let mut rpow = vec![self.one_poly()];
            let mut spow = vec![self.one_poly()];
            for i in 1..=big_m {
                rpow.push(rpow[i - 1].mul(&rhs.num));
                if !sden_trivial {
                    spow.push(spow[i - 1].mul(&rhs.den));
                }
            }
            let mut acc = self.zero_poly();
            for (e, pe) in parts.iter().enumerate() {
                if pe.is_zero() {
                    continue;
                }
                let m = e / 2;
                let mut t = pe.mul(&rpow[m]);
                if !sden_trivial {
                    t = t.mul(&spow[big_m - m]);
                }
                if e % 2 == 1 {
                    t = t.mul(&qp);
                }
                acc = acc.add(&t);
            }
            cur = acc;
            if !sden_trivial {
                mult = mult.mul(&spow[big_m]);
            }
        }
        Ok((cur, mult))
    }

    /// gcd-cancel and make the denominator's leading coefficient positive.
    fn light(&self, num: FlintMPoly, den: FlintMPoly) -> Result<Rf, TensorError> {
        if den.is_zero() {
            return Err(TensorError::DivisionByZero);
        }
        if num.is_zero() {
            return Ok(Rf {
                num,
                den: self.one_poly(),
            });
        }
        let (mut num, mut den) = (num, den);
        if !self.is_one_poly(&den) {
            if let Some(g) = num.gcd(&den) {
                if !g.is_zero() && !self.is_one_poly(&g) {
                    if let (Some(n2), Some(d2)) = (num.divides(&g), den.divides(&g)) {
                        num = n2;
                        den = d2;
                    }
                }
            }
        }
        if den.leading_coeff() < 0 {
            num = self.neg_poly(&num);
            den = self.neg_poly(&den);
        }
        Ok(Rf { num, den })
    }

    /// Full canonicalisation: reduce, rationalise the denominator, cancel.
    fn normalize(&self, num: FlintMPoly, den: FlintMPoly) -> Result<Rf, TensorError> {
        if den.is_zero() {
            return Err(TensorError::DivisionByZero);
        }
        if self.relations.is_empty() || num.is_zero() {
            return self.light(num, den);
        }
        let (n1, sn) = self.reduce_poly(&num)?;
        let (d1, sd) = self.reduce_poly(&den)?;
        let mut num = n1.mul(&sd);
        let mut den = d1.mul(&sn);
        if den.is_zero() {
            return Err(TensorError::DivisionByZero);
        }
        for (q, rhs) in &self.relations {
            if den.degree_in(*q) < 1 {
                continue;
            }
            let parts = self.split_by_var(&den, *q)?;
            let a = &parts[0];
            let b = &parts[1];
            let qp = self.poly_var(*q);
            let conj = a.sub(&b.mul(&qp));
            // den·conj = A² − (R/S)·B²  ⇒  num/den = num·conj·S / (A²S − R B²).
            let new_den = a.mul(a).mul(&rhs.den).sub(&rhs.num.mul(&b.mul(b)));
            let new_num = num.mul(&conj).mul(&rhs.den);
            let (rn, sn) = self.reduce_poly(&new_num)?;
            let (rd, sd) = self.reduce_poly(&new_den)?;
            num = rn.mul(&sd);
            den = rd.mul(&sn);
            if den.is_zero() {
                return Err(TensorError::DivisionByZero);
            }
        }
        self.light(num, den)
    }

    // ----- field arithmetic ---------------------------------------------

    pub(crate) fn zero(&self) -> Rf {
        Rf {
            num: self.zero_poly(),
            den: self.one_poly(),
        }
    }

    pub(crate) fn int(&self, c: i64) -> Rf {
        Rf {
            num: self.poly_const(&Integer::from(c)),
            den: self.one_poly(),
        }
    }

    pub(crate) fn rational(&self, n: &Integer, d: &Integer) -> Result<Rf, TensorError> {
        self.light(self.poly_const(n), self.poly_const(d))
    }

    fn var(&self, k: usize) -> Rf {
        Rf {
            num: self.poly_var(k),
            den: self.one_poly(),
        }
    }

    pub(crate) fn add(&self, a: &Rf, b: &Rf) -> Result<Rf, TensorError> {
        if a.is_zero() {
            return Ok(b.clone());
        }
        if b.is_zero() {
            return Ok(a.clone());
        }
        if a.den.equal(&b.den) {
            return self.light(a.num.add(&b.num), a.den.clone());
        }
        let g = a.den.gcd(&b.den).unwrap_or_else(|| self.one_poly());
        let (ad, bd) = match (a.den.divides(&g), b.den.divides(&g)) {
            (Some(x), Some(y)) => (x, y),
            _ => (a.den.clone(), b.den.clone()),
        };
        let num = a.num.mul(&bd).add(&b.num.mul(&ad));
        let den = a.den.mul(&bd);
        self.light(num, den)
    }

    pub(crate) fn neg(&self, a: &Rf) -> Rf {
        Rf {
            num: self.neg_poly(&a.num),
            den: a.den.clone(),
        }
    }

    pub(crate) fn sub(&self, a: &Rf, b: &Rf) -> Result<Rf, TensorError> {
        self.add(a, &self.neg(b))
    }

    pub(crate) fn mul(&self, a: &Rf, b: &Rf) -> Result<Rf, TensorError> {
        if a.is_zero() || b.is_zero() {
            return Ok(self.zero());
        }
        let cancel = |n: &FlintMPoly, d: &FlintMPoly| -> (FlintMPoly, FlintMPoly) {
            if self.is_one_poly(d) {
                return (n.clone(), d.clone());
            }
            match n.gcd(d) {
                Some(g) if !self.is_one_poly(&g) && !g.is_zero() => {
                    match (n.divides(&g), d.divides(&g)) {
                        (Some(x), Some(y)) => (x, y),
                        _ => (n.clone(), d.clone()),
                    }
                }
                _ => (n.clone(), d.clone()),
            }
        };
        let (an, bd) = cancel(&a.num, &b.den);
        let (bn, ad) = cancel(&b.num, &a.den);
        let num = an.mul(&bn);
        let den = ad.mul(&bd);
        let needs_reduce = self.relations.iter().any(|(q, _)| num.degree_in(*q) >= 2);
        if needs_reduce {
            self.normalize(num, den)
        } else {
            self.light(num, den)
        }
    }

    pub(crate) fn inv(&self, a: &Rf) -> Result<Rf, TensorError> {
        if a.is_zero() {
            return Err(TensorError::DivisionByZero);
        }
        self.normalize(a.den.clone(), a.num.clone())
    }

    pub(crate) fn div(&self, a: &Rf, b: &Rf) -> Result<Rf, TensorError> {
        let bi = self.inv(b)?;
        self.mul(a, &bi)
    }

    pub(crate) fn scale(&self, a: &Rf, n: i64, d: i64) -> Result<Rf, TensorError> {
        let r = self.rational(&Integer::from(n), &Integer::from(d))?;
        self.mul(a, &r)
    }

    fn pow(&self, a: &Rf, k: i64) -> Result<Rf, TensorError> {
        if k == 0 {
            return Ok(self.int(1));
        }
        let base = if k < 0 { self.inv(a)? } else { a.clone() };
        let mut e = k.unsigned_abs();
        let mut acc = self.int(1);
        let mut sq = base;
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul(&acc, &sq)?;
            }
            e >>= 1;
            if e > 0 {
                sq = self.mul(&sq, &sq)?;
            }
        }
        Ok(acc)
    }

    /// `∂a/∂x_j`.
    pub(crate) fn derivative(&self, a: &Rf, j: usize) -> Result<Rf, TensorError> {
        if a.is_zero() {
            return Ok(self.zero());
        }
        let n = &a.num;
        let d = &a.den;
        let d_const = d.length() == 1 && (0..self.nvars).all(|k| d.degree_in(k) <= 0);
        let mut acc = self.zero_poly();
        let mut extra: Vec<(FlintMPoly, Rf)> = Vec::new();
        for k in 0..self.nvars {
            let in_n = n.degree_in(k) > 0;
            let in_d = !d_const && d.degree_in(k) > 0;
            if !in_n && !in_d {
                continue;
            }
            let dv = self.dtab.get(k).map(|row| &row[j]).ok_or_else(|| {
                TensorError::Internal("derivative of a generator not yet tabulated".to_string())
            })?;
            if matches!(dv, DVal::Zero) {
                continue;
            }
            // (∂_k N)·D − N·(∂_k D)
            let mut t = if in_n {
                n.derivative(k).mul(d)
            } else {
                self.zero_poly()
            };
            if in_d {
                t = t.sub(&n.mul(&d.derivative(k)));
            }
            if t.is_zero() {
                continue;
            }
            match dv {
                DVal::Zero => {}
                DVal::One => acc = acc.add(&t),
                DVal::Val(r) if self.is_one_poly(&r.den) => acc = acc.add(&t.mul(&r.num)),
                DVal::Val(r) => extra.push((t, r.clone())),
                DVal::Missing(why) => {
                    return Err(TensorError::Differentiation { expr: why.clone() })
                }
            }
        }
        let d2 = d.mul(d);
        let mut out = self.normalize(acc, d2.clone())?;
        for (t, r) in extra {
            let part = self.normalize(t, d2.clone())?;
            out = self.add(&out, &self.mul(&part, &r)?)?;
        }
        Ok(out)
    }

    // ----- tables ---------------------------------------------------------

    fn uses_relation_var(&self, r: &Rf) -> bool {
        self.relations
            .iter()
            .any(|(q, _)| r.num.degree_in(*q) > 0 || r.den.degree_in(*q) > 0)
    }

    fn init_tables(&mut self, pool: &ExprPool) -> Result<(), TensorError> {
        // Pass 1: the relations, in generator order (a radicand only uses
        // earlier generators, whose relations are then already known).
        for k in 0..self.vars.len() {
            budget()?;
            let kind = self.vars[k].kind.clone();
            match &kind {
                VarKind::Imag => {
                    let rhs = self.int(-1);
                    self.relations.push((k, rhs));
                }
                VarKind::Sin {
                    hyperbolic,
                    partner,
                    ..
                } => {
                    // s² = 1 − c² (trig) / c² − 1 (hyperbolic)
                    let c = self.poly_var(*partner);
                    let c2 = c.mul(&c);
                    let one = self.one_poly();
                    let num = if *hyperbolic {
                        c2.sub(&one)
                    } else {
                        one.sub(&c2)
                    };
                    self.relations.push((
                        k,
                        Rf {
                            num,
                            den: self.one_poly(),
                        },
                    ));
                }
                VarKind::Root { radicand, q: 2 } => {
                    let b = self.convert(*radicand, pool)?;
                    if !b.is_zero() && !self.uses_relation_var(&b) {
                        self.relations.push((k, b));
                    }
                }
                _ => {}
            }
        }
        // Conversions made above saw only a prefix of the relations.
        self.memo.clear();
        // Pass 2: the derivative table, in generator order (a rule only
        // differentiates earlier generators).
        self.dtab = Vec::with_capacity(self.vars.len());
        for k in 0..self.vars.len() {
            budget()?;
            let kind = self.vars[k].kind.clone();
            let mut row = Vec::with_capacity(self.ncoords);
            for j in 0..self.ncoords {
                // A rule that needs a derivative the scan did not prepare is
                // recorded as missing and only refused if it is actually used.
                let dv = match self.rule(k, &kind, j, pool) {
                    Ok(dv) => dv,
                    Err(TensorError::Differentiation { expr }) => DVal::Missing(expr),
                    Err(e) => return Err(e),
                };
                row.push(dv);
            }
            self.dtab.push(row);
        }
        Ok(())
    }

    /// `∂(generator k)/∂x_j` by the generator's own rule.
    fn rule(
        &mut self,
        k: usize,
        kind: &VarKind,
        j: usize,
        pool: &ExprPool,
    ) -> Result<DVal, TensorError> {
        Ok(match kind {
            VarKind::Coord(i) => {
                if *i == j {
                    DVal::One
                } else {
                    DVal::Zero
                }
            }
            VarKind::Param | VarKind::Imag => DVal::Zero,
            VarKind::Sin {
                arg,
                partner,
                hyperbolic: _,
            } => {
                let dv = self.dconv(*arg, j, pool)?;
                self.wrap(self.mul(&self.var(*partner), &dv)?)
            }
            VarKind::Cos {
                arg,
                partner,
                hyperbolic,
            } => {
                let dv = self.dconv(*arg, j, pool)?;
                let t = self.mul(&self.var(*partner), &dv)?;
                self.wrap(if *hyperbolic { t } else { self.neg(&t) })
            }
            VarKind::Root { radicand, q } => {
                let b = self.convert(*radicand, pool)?;
                let db = self.derivative(&b, j)?;
                if db.is_zero() {
                    DVal::Zero
                } else {
                    let qb = self.scale(&b, i64::from(*q), 1)?;
                    let t = self.mul(&self.var(k), &self.div(&db, &qb)?)?;
                    self.wrap(t)
                }
            }
            VarKind::Exp { arg } => {
                let dv = self.dconv(*arg, j, pool)?;
                self.wrap(self.mul(&self.var(k), &dv)?)
            }
            VarKind::Log { arg } => {
                let v = self.convert(*arg, pool)?;
                let dv = self.derivative(&v, j)?;
                if dv.is_zero() {
                    DVal::Zero
                } else {
                    self.wrap(self.div(&dv, &v)?)
                }
            }
            VarKind::Atom => match self.vars[k].dexprs.as_ref().map(|ds| ds[j].clone()) {
                None => DVal::Missing(format!(
                    "a derivative of `{}` beyond the scanned order",
                    pool.display(self.vars[k].expr)
                )),
                Some(Ok(de)) => {
                    let v = self.convert(de, pool)?;
                    self.wrap(v)
                }
                Some(Err(msg)) => DVal::Missing(msg),
            },
        })
    }

    fn wrap(&self, r: Rf) -> DVal {
        if r.is_zero() {
            DVal::Zero
        } else if self.is_one_poly(&r.num) && self.is_one_poly(&r.den) {
            DVal::One
        } else {
            DVal::Val(r)
        }
    }

    /// `∂(convert(e))/∂x_j`, using only the generators already tabulated.
    fn dconv(&mut self, e: ExprId, j: usize, pool: &ExprPool) -> Result<Rf, TensorError> {
        let v = self.convert(e, pool)?;
        self.derivative(&v, j)
    }

    // ----- conversion ------------------------------------------------------

    fn lookup(&self, key: &Key) -> Result<usize, TensorError> {
        self.index
            .get(key)
            .copied()
            .ok_or_else(|| TensorError::Internal(format!("unscanned generator {key:?}")))
    }

    /// Convert an expression whose generators were all scanned.
    pub(crate) fn convert(&mut self, e: ExprId, pool: &ExprPool) -> Result<Rf, TensorError> {
        if let Some(r) = self.memo.get(&e) {
            return Ok(r.clone());
        }
        let r = self.convert_uncached(e, pool)?;
        self.memo.insert(e, r.clone());
        Ok(r)
    }

    fn convert_uncached(&mut self, e: ExprId, pool: &ExprPool) -> Result<Rf, TensorError> {
        match pool.get(e) {
            ExprData::Symbol { .. } => Ok(self.var(self.lookup(&Key::Sym(e))?)),
            ExprData::Integer(n) => Ok(Rf {
                num: self.poly_const(&n.0),
                den: self.one_poly(),
            }),
            ExprData::Rational(r) => {
                let (n, d) = r.0.clone().into_numer_denom();
                self.rational(&n, &d)
            }
            ExprData::Float(_) => Err(TensorError::InexactInput {
                expr: pool.display(e).to_string(),
            }),
            ExprData::Add(args) => {
                let mut acc = self.zero();
                for a in args {
                    let x = self.convert(a, pool)?;
                    acc = self.add(&acc, &x)?;
                }
                Ok(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = self.int(1);
                for a in args {
                    let x = self.convert(a, pool)?;
                    acc = self.mul(&acc, &x)?;
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => match pool.get(exp) {
                ExprData::Integer(n) => {
                    let k =
                        n.0.to_i64()
                            .ok_or_else(|| TensorError::UnsupportedExpression {
                                expr: pool.display(e).to_string(),
                            })?;
                    let b = self.convert(base, pool)?;
                    self.pow(&b, k)
                }
                ExprData::Rational(r) => {
                    let q = r.0.denom().to_u32().filter(|&q| (2..=64).contains(&q));
                    match q {
                        Some(q) => {
                            let p = r.0.numer().to_i64().ok_or_else(|| {
                                TensorError::UnsupportedExpression {
                                    expr: pool.display(e).to_string(),
                                }
                            })?;
                            let k = self.lookup(&Key::Root(base, q))?;
                            let w = self.var(k);
                            self.pow(&w, p)
                        }
                        None => Ok(self.var(self.lookup(&Key::Atom(e))?)),
                    }
                }
                _ => Ok(self.var(self.lookup(&Key::Atom(e))?)),
            },
            ExprData::Func { name, args } if args.len() == 1 => {
                let a = args[0];
                if let Some(h) = trig_kind(&name) {
                    let (n, v) = split_int_multiple(a, pool);
                    let s = self.var(self.lookup(&Key::Sin(v, h))?);
                    let c = self.var(self.lookup(&Key::Cos(v, h))?);
                    let (sn, cn) = self.multiple_angle(&s, &c, n, h)?;
                    return match name.as_str() {
                        "sin" | "sinh" => Ok(sn),
                        "cos" | "cosh" => Ok(cn),
                        "tan" | "tanh" => self.div(&sn, &cn),
                        "cot" | "coth" => self.div(&cn, &sn),
                        "sec" | "sech" => self.inv(&cn),
                        _ => self.inv(&sn),
                    };
                }
                match name.as_str() {
                    "exp" => {
                        let (n, v) = split_int_multiple(a, pool);
                        let x = self.var(self.lookup(&Key::Exp(v))?);
                        self.pow(&x, n)
                    }
                    "log" => Ok(self.var(self.lookup(&Key::Log(a))?)),
                    "sqrt" => Ok(self.var(self.lookup(&Key::Root(a, 2))?)),
                    _ => Ok(self.var(self.lookup(&Key::Atom(e))?)),
                }
            }
            ExprData::Func { .. } => Ok(self.var(self.lookup(&Key::Atom(e))?)),
            _ => Err(TensorError::UnsupportedExpression {
                expr: pool.display(e).to_string(),
            }),
        }
    }

    /// `(sin(n v), cos(n v))` from `(s, c) = (sin v, cos v)` (or sinh/cosh).
    fn multiple_angle(
        &self,
        s: &Rf,
        c: &Rf,
        n: i64,
        hyperbolic: bool,
    ) -> Result<(Rf, Rf), TensorError> {
        let m = n.unsigned_abs();
        let (mut sk, mut ck) = (s.clone(), c.clone());
        for _ in 1..m {
            let ns = self.add(&self.mul(&sk, c)?, &self.mul(&ck, s)?)?;
            let cc = self.mul(&ck, c)?;
            let ss = self.mul(&sk, s)?;
            let nc = if hyperbolic {
                self.add(&cc, &ss)?
            } else {
                self.sub(&cc, &ss)?
            };
            sk = ns;
            ck = nc;
        }
        if n < 0 {
            sk = self.neg(&sk);
        }
        Ok((sk, ck))
    }

    // ----- back to expressions -----------------------------------------

    /// `generator_k ^ e`, with `exp(v)^e` written `exp(e·v)`.
    fn var_pow_expr(&self, k: usize, e: i64, pool: &ExprPool) -> ExprId {
        let base = self.vars[k].expr;
        if e == 1 {
            return base;
        }
        if let VarKind::Exp { arg } = self.vars[k].kind {
            return pool.func("exp", vec![pool.mul(vec![pool.integer(e), arg])]);
        }
        pool.pow(base, pool.integer(e))
    }

    fn poly_to_expr(&self, p: &FlintMPoly, pool: &ExprPool) -> Result<ExprId, TensorError> {
        let terms = p.terms_u64().ok_or(TensorError::Internal(
            "polynomial exponent overflow".to_string(),
        ))?;
        if terms.is_empty() {
            return Ok(pool.integer(0));
        }
        let mut out = Vec::with_capacity(terms.len());
        // Descending order reads better (FLINT's own order is lex-descending).
        for (exp, coeff) in terms.into_iter().rev() {
            let mut fs = Vec::new();
            if coeff != 1 {
                fs.push(pool.integer(coeff));
            }
            for (k, &e) in exp.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                fs.push(self.var_pow_expr(k, e as i64, pool));
            }
            out.push(match fs.len() {
                0 => pool.integer(1),
                1 => fs[0],
                _ => pool.mul(fs),
            });
        }
        Ok(if out.len() == 1 {
            out[0]
        } else {
            pool.add(out)
        })
    }

    /// `Some(k)` when `p` is exactly the generator `x_k`.
    fn single_generator(&self, p: &FlintMPoly) -> Option<usize> {
        if p.length() != 1 {
            return None;
        }
        (0..self.nvars).find(|&k| p.equal(&self.poly_var(k)))
    }

    /// Factor `p` (unit, `[(factor, multiplicity)]`); unfactored above the limit.
    fn factor_list(&self, p: &FlintMPoly) -> (Integer, Vec<(FlintMPoly, u32)>) {
        if p.length() <= FACTOR_TERM_LIMIT {
            let mut f = FlintMPolyFactor::new(Arc::clone(&self.ctx));
            if f.factor(p) && f.constant_den_is_one() {
                let unit = f.unit().to_rug();
                let order = f.canonical_order();
                let mut out = Vec::with_capacity(order.len());
                for i in order {
                    let e = f.exp_at(i);
                    out.push((f.base_at(i), e));
                }
                return (unit, out);
            }
        }
        // Constant polynomials have an empty factor list.
        if (0..self.nvars).all(|k| p.degree_in(k) <= 0) {
            return (p.leading_coeff(), Vec::new());
        }
        (Integer::from(1), vec![(p.clone(), 1)])
    }

    /// Fold `(c − 1)^k (c + 1)^k` into `s^{2k}` in a factor list; returns
    /// the sign `(−1)^k` picked up by the trigonometric pairs.
    fn fold_trig_pairs(&self, list: &mut Vec<(FlintMPoly, u32)>) -> bool {
        let mut negate = false;
        for (k, var) in self.vars.iter().enumerate() {
            let VarKind::Cos {
                hyperbolic,
                partner,
                ..
            } = var.kind
            else {
                continue;
            };
            let c = self.poly_var(k);
            let one = self.one_poly();
            let cm = c.sub(&one);
            let cp = c.add(&one);
            let im = list.iter().position(|(f, _)| f.equal(&cm));
            let ip = list.iter().position(|(f, _)| f.equal(&cp));
            if let (Some(im), Some(ip)) = (im, ip) {
                let kk = list[im].1.min(list[ip].1);
                list[im].1 -= kk;
                list[ip].1 -= kk;
                // (c − 1)(c + 1) = c² − 1 = −s² (trig) or +s² (hyperbolic).
                if !hyperbolic && kk % 2 == 1 {
                    negate = !negate;
                }
                let s = self.poly_var(partner);
                match list.iter().position(|(f, _)| f.equal(&s)) {
                    Some(is) => list[is].1 += 2 * kk,
                    None => list.push((s, 2 * kk)),
                }
                list.retain(|(_, e)| *e > 0);
            }
        }
        negate
    }

    /// The expression for `a`: numerator and denominator factored over ℤ,
    /// with `(cos v − 1)(cos v + 1)` folded back into `−sin² v` and powers of
    /// `sin v` cancelled between numerator and denominator.
    pub(crate) fn to_expr(&self, a: &Rf, pool: &ExprPool) -> Result<ExprId, TensorError> {
        if a.is_zero() {
            return Ok(pool.integer(0));
        }
        let (mut un, mut nf) = self.factor_list(&a.num);
        let (ud, mut df) = self.factor_list(&a.den);
        if self.fold_trig_pairs(&mut nf) {
            un = -un;
        }
        if self.fold_trig_pairs(&mut df) {
            un = -un;
        }
        for (k, var) in self.vars.iter().enumerate() {
            if !matches!(var.kind, VarKind::Sin { .. }) {
                continue;
            }
            let s = self.poly_var(k);
            let inum = nf.iter().position(|(f, _)| f.equal(&s));
            let iden = df.iter().position(|(f, _)| f.equal(&s));
            if let (Some(i), Some(j)) = (inum, iden) {
                let m = nf[i].1.min(df[j].1);
                nf[i].1 -= m;
                df[j].1 -= m;
            }
            nf.retain(|(_, e)| *e > 0);
            df.retain(|(_, e)| *e > 0);
        }
        let mut fs = Vec::new();
        let coeff = rug::Rational::from((un, ud));
        if coeff != 1 {
            let (n, d) = coeff.into_numer_denom();
            fs.push(if d == 1 {
                pool.integer(n)
            } else {
                pool.rational(n, d)
            });
        }
        for (f, e) in &nf {
            if let Some(k) = self.single_generator(f) {
                fs.push(self.var_pow_expr(k, i64::from(*e), pool));
                continue;
            }
            let fe = self.poly_to_expr(f, pool)?;
            fs.push(if *e == 1 {
                fe
            } else {
                pool.pow(fe, pool.integer(*e))
            });
        }
        for (f, e) in &df {
            if let Some(k) = self.single_generator(f) {
                fs.push(self.var_pow_expr(k, -i64::from(*e), pool));
                continue;
            }
            let fe = self.poly_to_expr(f, pool)?;
            fs.push(pool.pow(fe, pool.integer(-i64::from(*e))));
        }
        Ok(match fs.len() {
            0 => pool.integer(1),
            1 => fs[0],
            _ => pool.mul(fs),
        })
    }

    /// Number of generators (for diagnostics).
    #[allow(dead_code)]
    pub(crate) fn nvars(&self) -> usize {
        self.nvars
    }
}
