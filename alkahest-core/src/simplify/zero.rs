//! Zero recognition (G4): [`zero_test`] and the [`ZeroSum`] rewrite rule.
//!
//! # The exact half: a rational normal form over atoms
//!
//! An expression is converted to a quotient of two polynomials with integer
//! coefficients in a set of *atoms*.  Every step of the conversion is an
//! identity that holds wherever the expression is defined:
//!
//! * `sin`, `cos`, `tan`, `sec`, `csc`, `cot` become rational functions of
//!   `exp(i·u)` (`sin u = (e^{iu} − e^{−iu})/(2i)`), and the hyperbolic
//!   functions rational functions of `exp(u)`;
//! * `exp(a + b) = exp(a)·exp(b)`, and `exp(r·u)` for a rational `r` is a
//!   power of the single atom `exp(u/L)`, `L` being the common denominator of
//!   every coefficient `u` carries in the expression;
//! * `exp(r·log b) = b^r` and `b^y = exp(y·log b)` (the principal-branch
//!   definition of a power);
//! * `b^(p/q)` is a power of the atom `t = b^(1/L)`, reduced by `t^L = b`;
//!   `sqrt(b) = b^(1/2)`;
//! * the imaginary unit is an atom reduced by `i² = −1`;
//! * `log` of a product / power / positive rational splits into a sum only
//!   when every factor is provably positive (domain `Positive`, an assumption,
//!   `π`, a positive literal, or `exp` of a real argument);
//! * the golden ratio `φ` is rewritten as `(1 + 5^(1/2))/2`, so `φ² = φ + 1`
//!   is seen;
//! * every other function application, `π`, every other named constant and
//!   every symbol is an atom.
//!
//! If the numerator reduces to the zero polynomial, the expression is zero
//! wherever it is defined: the reduction is substitution of identities, and a
//! zero polynomial is zero for every value of its atoms, related or not.  So
//! [`ZeroStatus::ProvedZero`] never rests on sampling.  The converse needs
//! more: the atoms may be algebraically dependent (`√2·√3` and `√6`), so a
//! non-zero numerator proves nothing in general.  It is a proof of
//! non-zeroness only when every atom the numerator mentions is a free symbol,
//! `π`, `i`, or `exp` of a monomial in non-integer free symbols — a genuine
//! rational function with coefficients in `ℚ(i)` of algebraically independent
//! quantities (the exponentials by Ax's theorem), which is non-zero somewhere
//! on any product of infinite sets.  (That is the Schwartz–Zippel situation;
//! with exact arithmetic the normal form decides it directly, so no sampling
//! is needed.)
//!
//! # The numeric half: Arb balls
//!
//! Otherwise the expression is evaluated with rigorous ball arithmetic at
//! exact rational points chosen inside each symbol's domain (and checked
//! against the assumption facts).  A ball that excludes zero at an admissible
//! point *proves* the expression is not identically zero —
//! [`ZeroStatus::ProvedNonzero`].  Balls that contain zero at every point
//! with a tiny radius are only evidence: [`ZeroStatus::ProbablyZero`], whose
//! verdict is `None`, never `True`.

use super::idmap::IdMap;
use super::rules::{NodeKinds, RewriteRule};
use crate::ball::{ArbBall, IntervalEval};
use crate::deriv::log::{DerivationLog, RewriteStep, SideCondition};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::poly::multipoly::{termmap_add, termmap_mul, termmap_pow, TermMap};
use rug::{Integer, Rational};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Rational functions over atoms
// ---------------------------------------------------------------------------

/// `num / den`, both integer polynomials in the atoms; `den ≠ 0`.
#[derive(Clone, Debug)]
struct Rf {
    num: TermMap,
    den: TermMap,
}

fn tm_const(n: Integer) -> TermMap {
    let mut t = TermMap::new();
    if n != 0 {
        t.insert(vec![], n);
    }
    t
}

fn tm_monomial(idx: usize, k: u32) -> TermMap {
    let mut e = vec![0u32; idx + 1];
    e[idx] = k;
    while e.last() == Some(&0) {
        e.pop();
    }
    let mut t = TermMap::new();
    t.insert(e, Integer::from(1));
    t
}

fn tm_is_one(t: &TermMap) -> bool {
    t.len() == 1 && t.get(&Vec::new()).is_some_and(|c| *c == 1)
}

impl Rf {
    fn constant(r: &Rational) -> Self {
        Rf {
            num: tm_const(r.numer().clone()),
            den: tm_const(r.denom().clone()),
        }
    }
    fn is_zero(&self) -> bool {
        self.num.is_empty()
    }
}

/// Why a conversion gave up.  Every variant just means "no verdict".
#[derive(Debug)]
struct Bail;

type R<T> = Result<T, Bail>;

/// An atom's identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum AtomKey {
    /// An opaque subexpression (symbol, `π`, unknown function, …).
    Opaque(ExprId),
    /// `exp(core / L)`.
    Exp(ExprId),
    /// `base^(1 / L)`.
    Root(ExprId),
}

/// What kind of thing an atom is, for the non-zeroness argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AtomClass {
    /// A free symbol or `π`: an independent transcendental / indeterminate.
    Free,
    /// The imaginary unit.
    ImagUnit,
    /// Anything else.
    Other,
}

/// Limits for one conversion.
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    /// Largest number of terms in any intermediate numerator or denominator.
    pub max_terms: usize,
    /// Largest number of distinct DAG nodes converted.
    pub max_nodes: usize,
    /// Largest `|n|` for an integer power of a non-monomial.
    pub max_pow: u32,
    /// Real-domain discipline for the rewrite rule: identities that hold over
    /// ℂ but turn a non-real left side into a real right side (`(√a)² = a`
    /// for `a < 0`, `exp(log b) = b` for `b < 0`) are used only when the
    /// radicand / logarithm argument is provably non-negative / positive.
    pub strict_real: bool,
}

impl Limits {
    pub(crate) const SIMPLIFY: Limits = Limits {
        max_terms: 256,
        max_nodes: 400,
        max_pow: 12,
        strict_real: true,
    };
    pub(crate) const ZERO_TEST: Limits = Limits {
        max_terms: 20_000,
        max_nodes: 50_000,
        max_pow: 64,
        strict_real: false,
    };
}

/// The converter.  One per expression.
pub(crate) struct Normalizer<'p> {
    pool: &'p ExprPool,
    limits: Limits,
    atoms: HashMap<AtomKey, usize>,
    classes: Vec<AtomClass>,
    /// Per atom index: `t^deg = rhs`.
    relations: Vec<Option<(u32, Rf)>>,
    exp_lcm: HashMap<ExprId, Integer>,
    root_lcm: HashMap<ExprId, Integer>,
    memo: IdMap<Rf>,
    /// Extra symbols / subexpressions known to be positive.
    positive: HashSet<ExprId>,
    /// Bases divided by during conversion (for side conditions).
    pub(crate) divisors: Vec<ExprId>,
    nodes: usize,
}

fn lcm_into(map: &mut HashMap<ExprId, Integer>, key: ExprId, den: &Integer) {
    let e = map.entry(key).or_insert_with(|| Integer::from(1));
    let l = Integer::from(e.lcm_ref(den));
    *e = l;
}

impl<'p> Normalizer<'p> {
    pub(crate) fn new(pool: &'p ExprPool, limits: Limits) -> Self {
        Normalizer {
            pool,
            limits,
            atoms: HashMap::new(),
            classes: Vec::new(),
            relations: Vec::new(),
            exp_lcm: HashMap::new(),
            root_lcm: HashMap::new(),
            memo: IdMap::default(),
            positive: HashSet::new(),
            divisors: Vec::new(),
            nodes: 0,
        }
    }

    pub(crate) fn with_positive(mut self, positive: HashSet<ExprId>) -> Self {
        self.positive = positive;
        self
    }

    fn one(&self) -> ExprId {
        self.pool.integer(1_i32)
    }

    // -- term splitting ----------------------------------------------------

    /// `arg` as `Σ r_i · core_i` with rational `r_i`.
    fn linear_terms(&self, arg: ExprId) -> Vec<(Rational, ExprId)> {
        let pool = self.pool;
        let terms = pool.with(arg, |d| match d {
            ExprData::Add(a) => a.clone(),
            _ => vec![arg],
        });
        terms
            .into_iter()
            .map(|t| {
                if let Some(r) = rational_literal(t, pool) {
                    return (r, self.one());
                }
                let fs = pool.with(t, |d| match d {
                    ExprData::Mul(a) => Some(a.clone()),
                    _ => None,
                });
                match fs {
                    Some(fs) => {
                        let mut c = Rational::from(1);
                        let mut rest = Vec::new();
                        for f in fs {
                            match rational_literal(f, pool) {
                                Some(r) => c *= r,
                                None => rest.push(f),
                            }
                        }
                        let core = match rest.len() {
                            0 => self.one(),
                            1 => rest[0],
                            _ => pool.mul(rest),
                        };
                        (c, core)
                    }
                    None => (Rational::from(1), t),
                }
            })
            .collect()
    }

    /// `(r, core) ↦ (r', core')` with `r'·core' = i·r·core`.
    fn times_i(&self, r: Rational, core: ExprId) -> (Rational, ExprId) {
        let pool = self.pool;
        let i = pool.imaginary_unit();
        if core == self.one() {
            return (r, i);
        }
        if core == i {
            return (-r, self.one());
        }
        let fs = pool.with(core, |d| match d {
            ExprData::Mul(a) => Some(a.clone()),
            _ => None,
        });
        if let Some(fs) = fs {
            if let Some(pos) = fs.iter().position(|&f| f == i) {
                let rest: Vec<ExprId> = fs
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != pos)
                    .map(|(_, &f)| f)
                    .collect();
                let core = match rest.len() {
                    0 => self.one(),
                    1 => rest[0],
                    _ => pool.mul(rest),
                };
                return (-r, core);
            }
        }
        (r, pool.mul(vec![i, core]))
    }

    /// The `log` argument of `core`, if `core` is `log(b)`.
    fn log_arg(&self, core: ExprId) -> Option<ExprId> {
        self.pool.with(core, |d| match d {
            ExprData::Func { name, args } if (name == "log" || name == "ln") && args.len() == 1 => {
                Some(args[0])
            }
            _ => None,
        })
    }

    /// May `exp(r·log b)` be read as `b^r` (and `b^y` as `exp(y·log b)`)?
    fn log_ok(&self, b: ExprId) -> bool {
        !self.limits.strict_real || self.is_positive(b, 6)
    }

    /// May the radical `b^(1/L)` be reduced by `t^L = b`?
    fn root_relation_ok(&self, b: ExprId) -> bool {
        !self.limits.strict_real
            || rational_literal(b, self.pool).is_some()
            || super::special_values::is_nonneg(b, self.pool, 6)
            || self.is_positive(b, 6)
    }

    /// The exponent of `exp(·)` that `b^y` (non-rational `y`) stands for.
    fn power_as_exp_arg(&self, base: ExprId, exp: ExprId) -> ExprId {
        let pool = self.pool;
        let log_b = pool.func("log", vec![base]);
        let terms = pool.with(exp, |d| match d {
            ExprData::Add(a) => a.clone(),
            _ => vec![exp],
        });
        pool.add(
            terms
                .into_iter()
                .map(|t| pool.mul(vec![t, log_b]))
                .collect(),
        )
    }

    // -- pre-scan ----------------------------------------------------------

    /// Record the denominators each exponential core / radical base carries.
    fn scan(&mut self, root: ExprId) -> R<()> {
        let mut seen: HashSet<ExprId> = HashSet::new();
        let mut stack = vec![root];
        while let Some(e) = stack.pop() {
            if !seen.insert(e) {
                continue;
            }
            if seen.len() > self.limits.max_nodes * 4 {
                return Err(Bail);
            }
            let data = self.pool.get(e);
            match &data {
                ExprData::Pow { base, exp } => {
                    stack.push(*base);
                    stack.push(*exp);
                    if let Some(r) = rational_literal(*exp, self.pool) {
                        if *r.denom() != 1 {
                            lcm_into(&mut self.root_lcm, *base, r.denom());
                        }
                    } else if integer_literal(*exp, self.pool).is_none() && self.log_ok(*base) {
                        let arg = self.power_as_exp_arg(*base, *exp);
                        self.scan_exp_arg(arg, false);
                        stack.push(arg);
                    }
                }
                ExprData::Func { name, args } if args.len() == 1 => {
                    stack.push(args[0]);
                    match name.as_str() {
                        "sqrt" => lcm_into(&mut self.root_lcm, args[0], &Integer::from(2)),
                        "exp" => self.scan_exp_arg(args[0], false),
                        "sin" | "cos" | "tan" | "sec" | "csc" | "cot" => {
                            self.scan_exp_arg(args[0], true)
                        }
                        "sinh" | "cosh" | "tanh" | "sech" | "csch" | "coth" => {
                            self.scan_exp_arg(args[0], false)
                        }
                        _ => {}
                    }
                }
                ExprData::Func { args, .. } | ExprData::Add(args) | ExprData::Mul(args) => {
                    stack.extend(args.iter().copied());
                }
                // `convert_node` rewrites `φ` in radicals; scan that form so
                // its `5^(1/2)` shares one root atom with any other `√5`.
                ExprData::Symbol { .. } if crate::eval::symbols::is_golden_ratio(e, self.pool) => {
                    stack.push(crate::eval::symbols::golden_ratio_radical(self.pool));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn scan_exp_arg(&mut self, arg: ExprId, times_i: bool) {
        for (r, core) in self.linear_terms(arg) {
            let (r, core) = if times_i {
                self.times_i(r, core)
            } else {
                (r, core)
            };
            match self.log_arg(core) {
                Some(b) if self.log_ok(b) => {
                    if *r.denom() != 1 {
                        lcm_into(&mut self.root_lcm, b, r.denom());
                    }
                }
                _ => lcm_into(&mut self.exp_lcm, core, r.denom()),
            }
        }
    }

    // -- atoms -------------------------------------------------------------

    fn atom(&mut self, key: AtomKey, class: AtomClass) -> usize {
        if let Some(&i) = self.atoms.get(&key) {
            return i;
        }
        let i = self.classes.len();
        self.atoms.insert(key, i);
        self.classes.push(class);
        self.relations.push(None);
        i
    }

    fn atom_pow(&self, idx: usize, k: i64) -> R<Rf> {
        let ku = u32::try_from(k.unsigned_abs()).map_err(|_| Bail)?;
        let m = tm_monomial(idx, ku);
        Ok(if k >= 0 {
            Rf {
                num: m,
                den: tm_const(Integer::from(1)),
            }
        } else {
            Rf {
                num: tm_const(Integer::from(1)),
                den: m,
            }
        })
    }

    // -- arithmetic --------------------------------------------------------

    fn check(&self, r: &Rf) -> R<()> {
        if r.num.len() > self.limits.max_terms || r.den.len() > self.limits.max_terms {
            return Err(Bail);
        }
        if crate::budget::check().is_err() {
            return Err(Bail);
        }
        Ok(())
    }

    fn mul_tm(&self, a: &TermMap, b: &TermMap) -> R<TermMap> {
        if tm_is_one(a) {
            return Ok(b.clone());
        }
        if tm_is_one(b) {
            return Ok(a.clone());
        }
        if a.len().saturating_mul(b.len()) > self.limits.max_terms * 64 {
            return Err(Bail);
        }
        termmap_mul(a, b).map_err(|_| Bail)
    }

    fn add(&self, a: Rf, b: Rf) -> R<Rf> {
        if a.is_zero() {
            return Ok(b);
        }
        if b.is_zero() {
            return Ok(a);
        }
        let r = if a.den == b.den {
            Rf {
                num: termmap_add(a.num, b.num),
                den: a.den,
            }
        } else if a.den.len() == 1 && b.den.len() == 1 {
            // Monomial denominators: use their lcm.
            let (ea, ca) = a.den.iter().next().unwrap();
            let (eb, cb) = b.den.iter().next().unwrap();
            let n = ea.len().max(eb.len());
            let mut el = vec![0u32; n];
            for (k, slot) in el.iter_mut().enumerate() {
                *slot = (*ea.get(k).unwrap_or(&0)).max(*eb.get(k).unwrap_or(&0));
            }
            let cl = Integer::from(ca.lcm_ref(cb));
            let fa = monomial_quotient(&el, &cl, ea, ca);
            let fb = monomial_quotient(&el, &cl, eb, cb);
            let num = termmap_add(self.mul_tm(&a.num, &fa)?, self.mul_tm(&b.num, &fb)?);
            let mut den = TermMap::new();
            while el.last() == Some(&0) {
                el.pop();
            }
            den.insert(el, cl);
            Rf { num, den }
        } else {
            let num = termmap_add(self.mul_tm(&a.num, &b.den)?, self.mul_tm(&b.num, &a.den)?);
            let den = self.mul_tm(&a.den, &b.den)?;
            Rf { num, den }
        };
        self.check(&r)?;
        Ok(r)
    }

    fn mul(&self, a: &Rf, b: &Rf) -> R<Rf> {
        if a.is_zero() || b.is_zero() {
            return Ok(Rf::constant(&Rational::from(0)));
        }
        let r = Rf {
            num: self.mul_tm(&a.num, &b.num)?,
            den: self.mul_tm(&a.den, &b.den)?,
        };
        self.check(&r)?;
        self.reduce(r)
    }

    fn inv(&self, a: Rf) -> R<Rf> {
        let a = self.reduce(a)?;
        if a.is_zero() {
            return Err(Bail);
        }
        Ok(Rf {
            num: a.den,
            den: a.num,
        })
    }

    fn powi(&self, a: &Rf, n: i64) -> R<Rf> {
        if n == 0 {
            return Ok(Rf::constant(&Rational::from(1)));
        }
        let k = u32::try_from(n.unsigned_abs()).map_err(|_| Bail)?;
        let single = a.num.len() <= 1 && a.den.len() <= 1;
        if !single && k > self.limits.max_pow {
            return Err(Bail);
        }
        let r = if single {
            Rf {
                num: termmap_pow(&a.num, k).map_err(|_| Bail)?,
                den: termmap_pow(&a.den, k).map_err(|_| Bail)?,
            }
        } else {
            // Square-and-multiply through the size-capped product, so a
            // power whose expansion would be huge bails instead of running.
            let mut base = a.clone();
            let mut acc = Rf::constant(&Rational::from(1));
            let mut e = k;
            while e > 0 {
                if e & 1 == 1 {
                    acc = self.mul(&acc, &base)?;
                }
                e >>= 1;
                if e > 0 {
                    base = self.mul(&base, &base)?;
                }
            }
            acc
        };
        self.check(&r)?;
        let r = self.reduce(r)?;
        if n < 0 {
            self.inv(r)
        } else {
            Ok(r)
        }
    }

    /// Rewrite `t^k` (`k ≥ deg`) by `t^deg = rhs` until no relation applies,
    /// in numerator and denominator alike.
    fn reduce(&self, r: Rf) -> R<Rf> {
        if self.relations.iter().all(Option::is_none) {
            return Ok(r);
        }
        let (num, fnum) = self.reduce_tm(r.num)?;
        let (den, fden) = self.reduce_tm(r.den)?;
        // value = (num / fnum) / (den / fden)
        let out = Rf {
            num: self.mul_tm(&num, &fden)?,
            den: self.mul_tm(&den, &fnum)?,
        };
        self.check(&out)?;
        Ok(out)
    }

    /// `p = reduced / factor`.
    fn reduce_tm(&self, mut p: TermMap) -> R<(TermMap, TermMap)> {
        let mut factor = tm_const(Integer::from(1));
        for _ in 0..16 {
            let needs = p.keys().any(|e| {
                e.iter().enumerate().any(
                    |(v, &k)| matches!(self.relations.get(v), Some(Some((deg, _))) if k >= *deg),
                )
            });
            if !needs {
                return Ok((p, factor));
            }
            // Pick one variable to reduce this round.
            let v = p
                .keys()
                .find_map(|e| {
                    e.iter().enumerate().find_map(|(v, &k)| {
                        matches!(self.relations.get(v), Some(Some((deg, _))) if k >= *deg)
                            .then_some(v)
                    })
                })
                .unwrap();
            let (deg, rhs) = self.relations[v].clone().unwrap();
            let jmax = p
                .keys()
                .map(|e| e.get(v).copied().unwrap_or(0) / deg)
                .max()
                .unwrap_or(0);
            // p = Σ_j (Σ terms with ⌊k/deg⌋ = j) ; multiply by rhs.den^jmax.
            let mut out = TermMap::new();
            let mut num_pows: Vec<TermMap> = vec![tm_const(Integer::from(1))];
            let mut den_pows: Vec<TermMap> = vec![tm_const(Integer::from(1))];
            for _ in 0..jmax {
                let n = self.mul_tm(num_pows.last().unwrap(), &rhs.num)?;
                let d = self.mul_tm(den_pows.last().unwrap(), &rhs.den)?;
                num_pows.push(n);
                den_pows.push(d);
            }
            for (e, c) in p {
                let k = e.get(v).copied().unwrap_or(0);
                let j = (k / deg) as usize;
                let mut e2 = e.clone();
                if v < e2.len() {
                    e2[v] = k % deg;
                }
                while e2.last() == Some(&0) {
                    e2.pop();
                }
                let mut mono = TermMap::new();
                mono.insert(e2, c);
                let scale = self.mul_tm(&num_pows[j], &den_pows[jmax as usize - j])?;
                out = termmap_add(out, self.mul_tm(&mono, &scale)?);
                if out.len() > self.limits.max_terms {
                    return Err(Bail);
                }
            }
            p = out;
            factor = self.mul_tm(&factor, &den_pows[jmax as usize])?;
        }
        Err(Bail)
    }

    // -- conversion --------------------------------------------------------

    fn convert(&mut self, e: ExprId) -> R<Rf> {
        if let Some(r) = self.memo.get(&e) {
            return Ok(r.clone());
        }
        self.nodes += 1;
        if self.nodes > self.limits.max_nodes {
            return Err(Bail);
        }
        let r = self.convert_node(e)?;
        self.check(&r)?;
        self.memo.insert(e, r.clone());
        Ok(r)
    }

    fn convert_node(&mut self, e: ExprId) -> R<Rf> {
        let pool = self.pool;
        let data = pool.get(e);
        match data {
            ExprData::Integer(n) => Ok(Rf::constant(&Rational::from(n.0))),
            ExprData::Rational(r) => Ok(Rf::constant(&r.0)),
            ExprData::Float(_) => Err(Bail),
            ExprData::Symbol { commutative, .. } => {
                if !commutative {
                    return Err(Bail);
                }
                if pool.is_imaginary_unit(e) {
                    let i = self.atom(AtomKey::Opaque(e), AtomClass::ImagUnit);
                    if self.relations[i].is_none() {
                        self.relations[i] = Some((2, Rf::constant(&Rational::from(-1))));
                    }
                    return self.atom_pow(i, 1);
                }
                // `φ` is algebraic: `φ² = φ + 1`. As an independent atom
                // `φ² − φ − 1` was a non-zero polynomial and was "proved"
                // non-zero; in radicals it reduces to `0`.
                if crate::eval::symbols::is_golden_ratio(e, pool) {
                    return self.convert(crate::eval::symbols::golden_ratio_radical(pool));
                }
                // The other named constants (`γ`, Catalan's `G`, Glaisher's
                // `A`, the Stieltjes `γₙ`) are numbers, not indeterminates, and
                // no theorem makes them algebraically independent of each
                // other or of `π` — whether `γ` is even irrational is open. So
                // they are opaque atoms of class `Other`: an exact zero
                // numerator still proves zero, but a non-zero one proves
                // nothing and the verdict falls to the Arb ball. Only `π`
                // (transcendental, Lindemann) is safe as a `Free` atom.
                let class = if crate::eval::symbols::is_real_constant(e, pool)
                    && !crate::eval::symbols::is_pi(e, pool)
                {
                    AtomClass::Other
                } else {
                    AtomClass::Free
                };
                let i = self.atom(AtomKey::Opaque(e), class);
                self.atom_pow(i, 1)
            }
            ExprData::Add(args) => {
                let mut acc = Rf::constant(&Rational::from(0));
                for a in args {
                    let t = self.convert(a)?;
                    acc = self.add(acc, t)?;
                }
                self.reduce(acc)
            }
            ExprData::Mul(args) => {
                if !pool.is_mult_commutative(e) {
                    return Err(Bail);
                }
                let mut acc = Rf::constant(&Rational::from(1));
                for a in args {
                    let t = self.convert(a)?;
                    acc = self.mul(&acc, &t)?;
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => {
                if let Some(n) = integer_literal(exp, pool) {
                    let b = self.convert(base)?;
                    if n < 0 && !(b.num.len() == 1 && b.num.contains_key(&Vec::new())) {
                        self.divisors.push(base);
                    }
                    return self.powi(&b, n);
                }
                if let Some(r) = rational_literal(exp, pool) {
                    return self.root_power(base, &r);
                }
                if !self.log_ok(base) {
                    return self.opaque(e);
                }
                let arg = self.power_as_exp_arg(base, exp);
                self.exp_of(arg, false)
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let u = args[0];
                match name.as_str() {
                    "sqrt" => self.root_power(u, &Rational::from((1, 2))),
                    "exp" => self.exp_of(u, false),
                    "sin" | "cos" | "tan" | "sec" | "csc" | "cot" => {
                        // E = exp(i u)
                        let e1 = self.exp_of(u, true)?;
                        let einv = self.inv(e1.clone())?;
                        let i = self.convert(pool.imaginary_unit())?;
                        let half = Rf::constant(&Rational::from((1, 2)));
                        let minus_one = Rf::constant(&Rational::from(-1));
                        let diff = self.add(e1.clone(), self.mul(&einv, &minus_one)?)?;
                        let sum = self.add(e1, einv)?;
                        // sin = (E − 1/E)/(2i) = −(i/2)(E − 1/E)
                        let neg_i_half = self.mul(&self.mul(&i, &half)?, &minus_one)?;
                        let sin = self.mul(&neg_i_half, &diff)?;
                        let cos = self.mul(&half, &sum)?;
                        match name.as_str() {
                            "sin" => Ok(sin),
                            "cos" => Ok(cos),
                            "tan" => {
                                let c = self.inv(cos)?;
                                self.mul(&sin, &c)
                            }
                            "cot" => {
                                let s = self.inv(sin)?;
                                self.mul(&cos, &s)
                            }
                            "sec" => self.inv(cos),
                            _ => self.inv(sin),
                        }
                    }
                    "sinh" | "cosh" | "tanh" | "sech" | "csch" | "coth" => {
                        let e1 = self.exp_of(u, false)?;
                        let einv = self.inv(e1.clone())?;
                        let half = Rf::constant(&Rational::from((1, 2)));
                        let minus_one = Rf::constant(&Rational::from(-1));
                        let diff = self.add(e1.clone(), self.mul(&einv, &minus_one)?)?;
                        let sum = self.add(e1, einv)?;
                        let sh = self.mul(&half, &diff)?;
                        let ch = self.mul(&half, &sum)?;
                        match name.as_str() {
                            "sinh" => Ok(sh),
                            "cosh" => Ok(ch),
                            "tanh" => {
                                let c = self.inv(ch)?;
                                self.mul(&sh, &c)
                            }
                            "coth" => {
                                let s = self.inv(sh)?;
                                self.mul(&ch, &s)
                            }
                            "sech" => self.inv(ch),
                            _ => self.inv(sh),
                        }
                    }
                    "log" | "ln" => self.log_of(e, u),
                    _ => self.opaque(e),
                }
            }
            ExprData::Func { .. } | ExprData::Piecewise { .. } => self.opaque(e),
            _ => Err(Bail),
        }
    }

    fn opaque(&mut self, e: ExprId) -> R<Rf> {
        if contains_float(e, self.pool) {
            return Err(Bail);
        }
        // f(−u) = −f(u) / f(u) for the odd / even functions (on all of ℂ).
        let sym = self.pool.with(e, |d| match d {
            ExprData::Func { name, args } if args.len() == 1 => {
                super::special_values::parity(name).map(|odd| (name.clone(), odd, args[0]))
            }
            _ => None,
        });
        if let Some((name, odd, arg)) = sym {
            if let Some(inner) = super::special_values::strip_negative(arg, self.pool) {
                let f = self.pool.func(name, vec![inner]);
                let v = self.convert(f)?;
                return if odd {
                    self.mul(&Rf::constant(&Rational::from(-1)), &v)
                } else {
                    Ok(v)
                };
            }
        }
        let class = if crate::eval::symbols::is_pi(e, self.pool) {
            AtomClass::Free
        } else {
            AtomClass::Other
        };
        let i = self.atom(AtomKey::Opaque(e), class);
        self.atom_pow(i, 1)
    }

    /// `base^r` for a rational `r`.
    fn root_power(&mut self, base: ExprId, r: &Rational) -> R<Rf> {
        if *r.denom() == 1 {
            let n = r.numer().to_i64().ok_or(Bail)?;
            let b = self.convert(base)?;
            return self.powi(&b, n);
        }
        let l = self
            .root_lcm
            .get(&base)
            .cloned()
            .unwrap_or_else(|| r.denom().clone());
        let lu = l.to_u32().ok_or(Bail)?;
        if lu > 64 {
            return Err(Bail);
        }
        let k = Rational::from(r * &l);
        if *k.denom() != 1 {
            return Err(Bail);
        }
        let k = k.numer().to_i64().ok_or(Bail)?;
        let idx = self.atom(AtomKey::Root(base), AtomClass::Other);
        if self.relations[idx].is_none() && self.root_relation_ok(base) {
            let b = self.convert(base)?;
            if b.is_zero() {
                return Err(Bail);
            }
            self.relations[idx] = Some((lu, b));
        }
        if k < 0 {
            self.divisors.push(base);
        }
        let t = self.atom_pow(idx, k)?;
        self.reduce(t)
    }

    /// `exp(arg)`, or `exp(i·arg)` when `times_i`.
    fn exp_of(&mut self, arg: ExprId, times_i: bool) -> R<Rf> {
        let mut acc = Rf::constant(&Rational::from(1));
        for (r, core) in self.linear_terms(arg) {
            let (r, core) = if times_i {
                self.times_i(r, core)
            } else {
                (r, core)
            };
            let log_b = self.log_arg(core).filter(|&b| self.log_ok(b));
            let f = if let Some(b) = log_b {
                // exp(r·log b) = b^r.
                self.root_power(b, &r)?
            } else if let Some(v) = self.exp_i_pi(&r, core) {
                // exp(i·r·π) at a tabulated r: cos(rπ) + i·sin(rπ), exactly.
                self.convert(v)?
            } else {
                let l = self
                    .exp_lcm
                    .get(&core)
                    .cloned()
                    .unwrap_or_else(|| r.denom().clone());
                let k = Rational::from(&r * &l);
                if *k.denom() != 1 {
                    return Err(Bail);
                }
                let k = k.numer().to_i64().ok_or(Bail)?;
                let class = if self.exp_core_is_free(core) {
                    AtomClass::Free
                } else {
                    AtomClass::Other
                };
                let idx = self.atom(AtomKey::Exp(core), class);
                self.atom_pow(idx, k)?
            };
            acc = self.mul(&acc, &f)?;
        }
        Ok(acc)
    }

    /// Is `exp(core)` algebraically independent of the free symbols and of
    /// every other such exponential?
    ///
    /// True when `core` is a product of non-integer-domain free symbols
    /// (optionally times `i` and `π`) — distinct such monomials are
    /// ℚ-linearly independent modulo constants, so their exponentials are
    /// algebraically independent over ℂ(symbols) (the exponential case of
    /// Ax's theorem).  Integer-domain symbols are excluded: `exp(iπn)² = 1`
    /// for every integer `n`.
    fn exp_core_is_free(&self, core: ExprId) -> bool {
        let pool = self.pool;
        let factors = pool.with(core, |d| match d {
            ExprData::Mul(a) => a.clone(),
            _ => vec![core],
        });
        let mut has_var = false;
        for f in factors {
            let ok = pool.with(f, |d| match d {
                ExprData::Symbol {
                    name,
                    domain,
                    commutative: true,
                } => {
                    if pool.is_imaginary_unit(f) || name == "pi" {
                        true
                    } else if crate::eval::symbols::constant_value_f64(name).is_some() {
                        // `exp(γ)`, `exp(γ·x)`: a named constant is a number,
                        // not a variable, and `γ·x` vs `x` are ℚ-linearly
                        // independent only if `γ` is irrational — unknown.
                        false
                    } else if *domain == Domain::Integer {
                        false
                    } else {
                        has_var = true;
                        true
                    }
                }
                _ => false,
            });
            if !ok {
                return false;
            }
        }
        has_var
    }

    /// `cos(rπ) + i·sin(rπ)` when `core = i·π` and `r` is in the table.
    fn exp_i_pi(&self, r: &Rational, core: ExprId) -> Option<ExprId> {
        let pool = self.pool;
        let i = pool.imaginary_unit();
        let fs = pool.with(core, |d| match d {
            ExprData::Mul(a) if a.len() == 2 => Some(a.clone()),
            _ => None,
        })?;
        let ok = (fs[0] == i && crate::eval::symbols::is_pi(fs[1], pool))
            || (fs[1] == i && crate::eval::symbols::is_pi(fs[0], pool));
        if !ok {
            return None;
        }
        let c = super::special_values::cos_pi(r, pool)?;
        let s = super::special_values::sin_pi(r, pool)?;
        Some(pool.add(vec![c, pool.mul(vec![i, s])]))
    }

    fn log_of(&mut self, e: ExprId, u: ExprId) -> R<Rf> {
        let pool = self.pool;
        // log of a positive rational: split over small primes.
        if let Some(r) = rational_literal(u, pool) {
            if r > 0 {
                if let Some(parts) = factor_small(r.numer()).and_then(|mut n| {
                    let d = factor_small(r.denom())?;
                    n.extend(d.into_iter().map(|(p, k)| (p, -k)));
                    Some(n)
                }) {
                    let mut acc = Rf::constant(&Rational::from(0));
                    for (p, k) in parts {
                        let lp = pool.func("log", vec![pool.integer(p)]);
                        let i = self.atom(AtomKey::Opaque(lp), AtomClass::Other);
                        let t =
                            self.mul(&Rf::constant(&Rational::from(k)), &self.atom_pow(i, 1)?)?;
                        acc = self.add(acc, t)?;
                    }
                    return Ok(acc);
                }
            }
            return self.opaque(e);
        }
        let data = pool.get(u);
        match data {
            ExprData::Mul(fs) if fs.iter().all(|&f| self.is_positive(f, 6)) => {
                let mut acc = Rf::constant(&Rational::from(0));
                for f in fs {
                    let l = pool.func("log", vec![f]);
                    let t = self.convert(l)?;
                    acc = self.add(acc, t)?;
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp }
                if rational_literal(exp, pool).is_some() && self.is_positive(base, 6) =>
            {
                let r = rational_literal(exp, pool).unwrap();
                let l = pool.func("log", vec![base]);
                let t = self.convert(l)?;
                self.mul(&Rf::constant(&r), &t)
            }
            ExprData::Func { name, args }
                if name == "sqrt" && args.len() == 1 && self.is_positive(args[0], 6) =>
            {
                let l = pool.func("log", vec![args[0]]);
                let t = self.convert(l)?;
                self.mul(&Rf::constant(&Rational::from((1, 2))), &t)
            }
            ExprData::Func { name, args }
                if name == "exp" && args.len() == 1 && is_real(args[0], pool, 6) =>
            {
                self.convert(args[0])
            }
            _ => self.opaque(e),
        }
    }

    /// Conservative positivity, including the caller's positive facts.
    fn is_positive(&self, e: ExprId, depth: u32) -> bool {
        if depth == 0 {
            return false;
        }
        if self.positive.contains(&e) {
            return true;
        }
        let pool = self.pool;
        enum K {
            Yes,
            No,
            All(Vec<ExprId>),
            Pow(ExprId, ExprId),
            Func(String, ExprId),
        }
        let k = pool.with(e, |d| match d {
            ExprData::Integer(n) => {
                if n.0 > 0 {
                    K::Yes
                } else {
                    K::No
                }
            }
            ExprData::Rational(r) => {
                if r.0 > 0 {
                    K::Yes
                } else {
                    K::No
                }
            }
            ExprData::Symbol { name, domain, .. } => {
                if *domain == Domain::Positive
                    || crate::eval::symbols::constant_value_f64(name).is_some_and(|v| v > 0.0)
                {
                    K::Yes
                } else {
                    K::No
                }
            }
            ExprData::Add(a) | ExprData::Mul(a) => K::All(a.clone()),
            ExprData::Pow { base, exp } => K::Pow(*base, *exp),
            ExprData::Func { name, args } if args.len() == 1 => K::Func(name.clone(), args[0]),
            _ => K::No,
        });
        match k {
            K::Yes => true,
            K::No => false,
            K::All(a) => a.iter().all(|&x| self.is_positive(x, depth - 1)),
            K::Pow(b, x) => self.is_positive(b, depth - 1) && is_real(x, pool, depth - 1),
            K::Func(name, a) => match name.as_str() {
                "exp" | "cosh" => is_real(a, pool, depth - 1),
                "sqrt" => self.is_positive(a, depth - 1),
                _ => false,
            },
        }
    }

    /// Convert `expr` and test the numerator.
    ///
    /// `Some(true)`: zero wherever defined.  `Some(false)`: a non-zero
    /// rational function of independent atoms (proof of non-zeroness).
    /// `None`: no verdict.
    pub(crate) fn decide(&mut self, expr: ExprId) -> Option<bool> {
        // ∞ − ∞ is not 0: infinities are not values the identities cover.
        if self.pool.has_non_finite(expr) {
            return None;
        }
        self.scan(expr).ok()?;
        let r = self.convert(expr).ok()?;
        let r = self.reduce(r).ok()?;
        if r.den.is_empty() {
            return None;
        }
        if r.is_zero() {
            return Some(true);
        }
        // Non-zero numerator.  A non-zero *constant* numerator is non-zero
        // whatever the atoms are.
        if r.num.len() == 1 && r.num.contains_key(&Vec::new()) {
            return Some(false);
        }
        let independent = self.classes.iter().enumerate().all(|(i, c)| match c {
            AtomClass::Free => true,
            AtomClass::ImagUnit => true,
            AtomClass::Other => {
                // An atom the numerator does not mention is harmless.
                !r.num.keys().any(|e| e.get(i).copied().unwrap_or(0) > 0)
            }
        });
        if independent {
            return Some(false);
        }
        None
    }
}

/// `L / (e, c)` as a monomial, where `(e, c)` divides `(L_e, L_c)`.
fn monomial_quotient(le: &[u32], lc: &Integer, e: &[u32], c: &Integer) -> TermMap {
    let mut q: Vec<u32> = le
        .iter()
        .enumerate()
        .map(|(k, &x)| x - e.get(k).copied().unwrap_or(0))
        .collect();
    while q.last() == Some(&0) {
        q.pop();
    }
    let mut t = TermMap::new();
    t.insert(q, Integer::from(lc / c));
    t
}

fn rational_literal(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    })
}

fn integer_literal(e: ExprId, pool: &ExprPool) -> Option<i64> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    })
}

fn contains_float(e: ExprId, pool: &ExprPool) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if !seen.insert(x) {
            continue;
        }
        let found = pool.with(x, |d| match d {
            ExprData::Float(_) => true,
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
                stack.extend(a.iter().copied());
                false
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
                false
            }
            ExprData::Piecewise { branches, default } => {
                for (c, v) in branches {
                    stack.push(*c);
                    stack.push(*v);
                }
                stack.push(*default);
                false
            }
            ExprData::Predicate { args, .. } => {
                stack.extend(args.iter().copied());
                false
            }
            _ => false,
        });
        if found {
            return true;
        }
    }
    false
}

/// Free symbols of `e` (not `π`, not `i`), in first-seen order; a DAG walk,
/// linear in the number of distinct nodes.
fn free_symbols(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if !seen.insert(x) {
            continue;
        }
        pool.with(x, |d| match d {
            ExprData::Symbol { .. } if !crate::eval::symbols::is_named_constant(x, pool) => {
                out.push(x)
            }
            ExprData::Add(a)
            | ExprData::Mul(a)
            | ExprData::Func { args: a, .. }
            | ExprData::Predicate { args: a, .. } => stack.extend(a.iter().copied()),
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            ExprData::Piecewise { branches, default } => {
                for (c, v) in branches {
                    stack.push(*c);
                    stack.push(*v);
                }
                stack.push(*default);
            }
            ExprData::BigO(a) => stack.push(*a),
            _ => {}
        });
    }
    out
}

/// Structural reality (conservative).
fn is_real(e: ExprId, pool: &ExprPool, depth: u32) -> bool {
    if depth == 0 {
        return false;
    }
    enum K {
        Yes,
        No,
        All(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, ExprId),
    }
    let k = pool.with(e, |d| match d {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => K::Yes,
        ExprData::Symbol { domain, .. } => {
            if matches!(
                domain,
                Domain::Real | Domain::Integer | Domain::Positive | Domain::NonNegative
            ) {
                K::Yes
            } else {
                K::No
            }
        }
        ExprData::Add(a) | ExprData::Mul(a) => K::All(a.clone()),
        ExprData::Pow { base, exp } => K::Pow(*base, *exp),
        ExprData::Func { name, args } if args.len() == 1 => K::Func(name.clone(), args[0]),
        _ => K::No,
    });
    match k {
        K::Yes => true,
        K::No => false,
        K::All(a) => a.iter().all(|&x| is_real(x, pool, depth - 1)),
        K::Pow(b, x) => integer_literal(x, pool).is_some() && is_real(b, pool, depth - 1),
        K::Func(name, a) => {
            matches!(
                name.as_str(),
                "sin" | "cos" | "exp" | "sinh" | "cosh" | "tanh" | "atan" | "erf" | "abs"
            ) && is_real(a, pool, depth - 1)
        }
    }
}

/// `n = Π p^k` over primes below 10 000, or `None` if a larger prime divides.
fn factor_small(n: &Integer) -> Option<Vec<(u32, i64)>> {
    let mut m = n.clone();
    let mut out = Vec::new();
    if m <= 0 {
        return None;
    }
    let mut p: u32 = 2;
    while m > 1 && p < 10_000 {
        let mut k = 0i64;
        while m.is_divisible_u(p) {
            m /= p;
            k += 1;
        }
        if k > 0 {
            out.push((p, k));
        }
        p += if p == 2 { 1 } else { 2 };
    }
    (m == 1).then_some(out)
}

// ---------------------------------------------------------------------------
// ZeroSum: the rewrite rule
// ---------------------------------------------------------------------------

/// Marker rule: a rule list containing it ends every
/// [`simplify_with`](super::engine::simplify_with) fixed point with the
/// zero-sum pass (an `Add` that is identically zero becomes `0`).
///
/// The pass is not a per-node rule because the screen it needs — an `f64`
/// value for every subexpression — is linear when computed once for the whole
/// DAG and quadratic when recomputed at every `Add`.  It is in
/// [`rules_for_config`](super::engine::rules_for_config), so `simplify`,
/// `simplify_expanded`, `simplify_trig_normal_form` and every caller that
/// passes the default rules get it.  `apply` never fires.
///
/// The pass replaces an `Add` by `0` only when the rational normal form over
/// atoms (module docs) proves it zero, with the real-domain discipline of
/// the simplifier's limits; the step records `d ≠ 0` for every base the sum
/// divided by.  A cheap `f64` evaluation at a fixed point screens out the
/// (overwhelmingly common) visibly non-zero sums first.
pub struct ZeroSum;

impl ZeroSum {
    pub(crate) const NAME: &'static str = "zero_sum";
}

impl RewriteRule for ZeroSum {
    fn node_kinds(&self) -> NodeKinds {
        // Never fires; the narrowest mask keeps the engine from probing it.
        NodeKinds::OTHER
    }
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn apply(&self, _expr: ExprId, _pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        None
    }
}

/// Does `rules` ask for the zero-sum pass?
pub(crate) fn has_zero_marker(rules: &[Box<dyn RewriteRule>]) -> bool {
    // `rules_for_config` puts the marker last; check there before scanning.
    rules.last().is_some_and(|r| r.name() == ZeroSum::NAME)
        || rules.iter().any(|r| r.name() == ZeroSum::NAME)
}

/// Whether the DAG under `e` has at most `cap` nodes, all of kinds the
/// normal form handles exactly (no floats, no non-finite values).
fn small_dag(e: ExprId, pool: &ExprPool, cap: usize) -> bool {
    if pool.has_non_finite(e) {
        return false;
    }
    let mut seen = super::idmap::IdSet::default();
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if !seen.insert(x) {
            continue;
        }
        if seen.len() > cap {
            return false;
        }
        let ok = pool.with(x, |d| match d {
            ExprData::Float(_) => false,
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
                stack.extend(a.iter().copied());
                true
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
                true
            }
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Symbol { .. } => true,
            _ => false,
        });
        if !ok {
            return false;
        }
    }
    true
}

/// The splitmix64 finaliser: every input bit affects every output bit.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A fixed, irrational-looking sample value for a symbol.
fn screen_value(name: &str, domain: Domain) -> f64 {
    let mut h: u64 = 1469598103934665603;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(1099511628211);
    }
    // FNV alone barely moves the high bits for names that differ only in the
    // last byte (`a01` vs `a02`), and near-equal sample values make every
    // determinant of symbols screen as ≈ 0; finish with a full avalanche.
    let h = mix64(h);
    let frac = (h >> 11) as f64 / (1u64 << 53) as f64;
    match domain {
        Domain::Integer => 3.0 + (h % 1009) as f64,
        _ => 0.61 + 0.73 * frac,
    }
}

/// The derivation-log name of [`super::rules::collect_through_sum`].
const COLLECT_THROUGH_SUM: &str = "collect_through_sum";

/// At most this many exact attempts per pass: the screen lets through every
/// sum it cannot evaluate (complex, unsupported functions), and each attempt
/// may cost up to `Limits::SIMPLIFY`.
const MAX_EXACT_ATTEMPTS: usize = 32;

/// How the sweep combines a node's children: the node kinds it descends into.
#[derive(Clone, Copy)]
enum SweepKind {
    Add,
    Mul,
    Pow,
    Func(FuncCode),
}

/// The screening value of a leaf of the sweep, or the kind of an inner node
/// with its children pushed onto `children`.
fn sweep_node(
    e: ExprId,
    pool: &ExprPool,
    children: &mut Vec<ExprId>,
) -> Result<SweepKind, Option<f64>> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Err(Some(n.0.to_f64())),
        ExprData::Rational(r) => Err(Some(r.0.to_f64())),
        ExprData::Float(f) => Err(Some(f.inner.to_f64())),
        ExprData::Symbol { name, domain, .. } => Err(
            if let Some(c) = crate::eval::symbols::constant_value_f64(name) {
                Some(c)
            } else if pool.is_imaginary_unit(e) {
                None
            } else {
                Some(screen_value(name, *domain))
            },
        ),
        ExprData::Add(a) => {
            children.extend_from_slice(a);
            Ok(SweepKind::Add)
        }
        ExprData::Mul(a) => {
            children.extend_from_slice(a);
            Ok(SweepKind::Mul)
        }
        ExprData::Pow { base, exp } => {
            children.push(*base);
            children.push(*exp);
            Ok(SweepKind::Pow)
        }
        ExprData::Func { name, args } => {
            children.extend_from_slice(args);
            Ok(SweepKind::Func(FuncCode::of(name)))
        }
        _ => Err(None),
    })
}

/// Whether `Σ vals` is visibly non-zero next to the size of its terms (an
/// unknown value makes it unknown, so not visibly non-zero).
fn screened_nonzero_sum(vals: &[Option<f64>]) -> bool {
    let mut sum = 0.0f64;
    let mut scale = 0.0f64;
    for v in vals {
        let Some(v) = *v else {
            return false;
        };
        sum += v;
        scale += v.abs();
    }
    sum.abs() > 1e-9 * (1.0 + scale)
}

/// The screening value of an inner node from its children's values.  `exp`
/// is the node's exponent child for a `Pow` (its kind decides whether a NaN
/// power is screened as an opaque atom).
fn screen_combine(
    kind: SweepKind,
    vals: &[Option<f64>],
    exp: Option<ExprId>,
    pool: &ExprPool,
) -> Option<f64> {
    let v = match kind {
        SweepKind::Add => {
            let mut sum = 0.0f64;
            for v in vals {
                sum += (*v)?;
            }
            Some(sum)
        }
        SweepKind::Mul => vals.iter().try_fold(1.0f64, |acc, v| v.map(|v| acc * v)),
        SweepKind::Pow => match (vals[0], vals[1]) {
            (Some(b), Some(x)) => {
                let r = if x.fract() == 0.0 && x.abs() < 1e6 {
                    b.powi(x as i32)
                } else {
                    b.powf(x)
                };
                // `b^y` with `b < 0` and a symbolic `y` is an opaque atom to
                // the exact test (real-domain discipline), so an opaque
                // pseudo-value screens it honestly.  A rational exponent
                // keeps `None`: its radical relation is something the exact
                // test does use.
                let symbolic_exp =
                    exp.is_some_and(|x| !pool.with(x, |d| matches!(d, ExprData::Rational(_))));
                if r.is_nan() && symbolic_exp {
                    FuncCode::Opaque {
                        name_hash: 0x9e37_79b9_7f4a_7c15,
                        parity: None,
                    }
                    .screen(vals)
                } else {
                    Some(r)
                }
            }
            _ => None,
        },
        SweepKind::Func(code) => code.screen(vals),
    };
    v.filter(|v| v.is_finite())
}

/// The screening value of the subtree under `root`, memoised per node in the
/// pool (a pure function of the node, so it is computed once per pool).
/// Iterative post-order; `stack` and `vals` are scratch space.
fn screen_of(root: ExprId, pool: &ExprPool, scratch: &mut ScreenScratch) -> Option<f64> {
    if let Some(v) = pool.zero_screen(root) {
        return v;
    }
    let ScreenScratch {
        stack,
        children,
        vals,
    } = scratch;
    stack.clear();
    stack.push((root, false));
    while let Some((e, expanded)) = stack.pop() {
        if pool.zero_screen(e).is_some() {
            continue;
        }
        children.clear();
        let kind = match sweep_node(e, pool, children) {
            Err(v) => {
                pool.set_zero_screen(e, v);
                continue;
            }
            Ok(k) => k,
        };
        if !expanded {
            stack.push((e, true));
            for &c in children.iter().rev() {
                if pool.zero_screen(c).is_none() {
                    stack.push((c, false));
                }
            }
            continue;
        }
        vals.clear();
        vals.extend(children.iter().map(|&c| pool.zero_screen(c).flatten()));
        let exp = matches!(kind, SweepKind::Pow).then(|| children[1]);
        pool.set_zero_screen(e, screen_combine(kind, vals, exp, pool));
    }
    pool.zero_screen(root).flatten()
}

/// Reusable buffers for [`screen_of`].
#[derive(Default)]
struct ScreenScratch {
    stack: Vec<(ExprId, bool)>,
    children: Vec<ExprId>,
    vals: Vec<Option<f64>>,
}

/// One bottom-up sweep: every `Add` the exact normal form proves zero becomes
/// `0`, and one whose like terms meet only through a sum factor is collected
/// ([`super::rules::collect_through_sum`]).  `None` when nothing changed.
///
/// Only subtrees that can hide a zero sum are walked: the pool records, per
/// node at intern time, whether some `Add` below has a term that is not a
/// plain monomial (a sum of plain monomials is an expanded polynomial that
/// like-term collection has already made canonical), and marks a subtree
/// clean once a complete sweep has left it unchanged.  Everything else — the
/// root of `2·x·cos(x²)`, the `x + 1` of `(x+1)¹⁴`, the 2×2 minors of a
/// determinant — is not walked; where a parent sum needs its screening value,
/// that value is memoised in the pool too.  So a repeated sweep over a
/// settled expression is O(1), and a first one is linear in the candidate
/// part of the DAG.
#[inline(never)]
pub(crate) fn zero_sum_pass(root: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
    if !pool.zero_candidate(root) {
        return None;
    }

    // Post-order over the candidate nodes (always `Add`/`Mul`/`Pow`/`Func`);
    // every other child is unchanged and contributes its memoised value.
    let mut out: IdMap<(ExprId, Option<f64>)> = IdMap::default();
    let mut log = DerivationLog::new();
    let mut attempts = 0usize;
    let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];
    let mut children: Vec<ExprId> = Vec::new();
    let mut new_ids: Vec<ExprId> = Vec::new();
    let mut vals: Vec<Option<f64>> = Vec::new();
    let mut scratch = ScreenScratch::default();
    while let Some((e, expanded)) = stack.pop() {
        if out.contains_key(&e) {
            continue;
        }
        children.clear();
        let kind = match sweep_node(e, pool, &mut children) {
            Ok(k) => k,
            // A candidate is never a leaf; treat one defensively as clean.
            Err(v) => {
                out.insert(e, (e, v));
                continue;
            }
        };
        if !expanded {
            stack.push((e, true));
            for &c in children.iter().rev() {
                if pool.zero_candidate(c) && !out.contains_key(&c) {
                    stack.push((c, false));
                }
            }
            continue;
        }
        let mut changed = false;
        new_ids.clear();
        vals.clear();
        for &c in &children {
            let (n, v) = match out.get(&c) {
                Some(&nv) => nv,
                None => (c, screen_of(c, pool, &mut scratch)),
            };
            changed |= n != c;
            new_ids.push(n);
            vals.push(v);
        }
        let node = if !changed {
            e
        } else {
            match kind {
                SweepKind::Add => pool.add(new_ids.clone()),
                SweepKind::Mul => pool.mul(new_ids.clone()),
                SweepKind::Pow => pool.pow(new_ids[0], new_ids[1]),
                SweepKind::Func(_) => {
                    let name = pool.with(e, |d| match d {
                        ExprData::Func { name, .. } => name.clone(),
                        _ => String::new(),
                    });
                    pool.func(name, new_ids.clone())
                }
            }
        };
        if let SweepKind::Add = kind {
            let screened_nonzero = screened_nonzero_sum(&vals);
            let is_add = pool.with(node, |d| matches!(d, ExprData::Add(a) if a.len() >= 2));
            if !screened_nonzero
                && is_add
                && attempts < MAX_EXACT_ATTEMPTS
                && small_dag(node, pool, Limits::SIMPLIFY.max_nodes)
            {
                attempts += 1;
                let mut nz = Normalizer::new(pool, Limits::SIMPLIFY);
                if nz.decide(node) == Some(true) {
                    let zero = pool.integer(0_i32);
                    let mut conds: Vec<SideCondition> = Vec::new();
                    for d in nz.divisors {
                        let c = SideCondition::NonZero(d);
                        if !conds.contains(&c) {
                            conds.push(c);
                        }
                    }
                    log.push(RewriteStep::with_conditions(
                        ZeroSum::NAME,
                        node,
                        zero,
                        conds,
                    ));
                    out.insert(e, (zero, Some(0.0)));
                    continue;
                }
            }
            // Not zero: like terms may still meet through a sum factor
            // (`−a²(b−1) + a²b → a²`) without expanding anything.
            if let Some(c) = super::rules::collect_through_sum(node, pool) {
                log.push(RewriteStep::simple(COLLECT_THROUGH_SUM, node, c));
                out.insert(e, (c, screen_of(c, pool, &mut scratch)));
                continue;
            }
        }
        let exp = matches!(kind, SweepKind::Pow).then(|| new_ids[1]);
        let value = screen_combine(kind, &vals, exp, pool);
        // The value is `node`'s own screening value (same children, same
        // formula), so it is memoised for `node`.
        pool.set_zero_screen(node, value);
        out.insert(e, (node, value));
    }
    let new_root = out[&root].0;
    // Remember the subtrees the sweep left alone — interned nodes are
    // immutable, so they need not be swept again on this pool.  Only after a
    // complete sweep: a capped or budget-limited one may have skipped a zero.
    if attempts < MAX_EXACT_ATTEMPTS && crate::budget::check().is_ok() {
        for (k, (n, _)) in &out {
            if k == n {
                pool.mark_zero_clean(*k);
            }
        }
    }
    if new_root == root {
        return None;
    }
    Some((new_root, log))
}

/// How the zero-sum screen values a function application.
#[derive(Clone, Copy)]
enum FuncCode {
    /// An elementary function `std` evaluates cheaply (index into the match
    /// in [`FuncCode::screen`]).
    Elementary(u8),
    /// Anything else: a deterministic pseudo-value from the name and the
    /// argument values.  The exact test treats such a function as an opaque
    /// atom (up to odd/even symmetry), so any value consistent with "equal
    /// arguments give equal values" and that symmetry is an honest screen —
    /// and it costs a hash, where the function's own `f64` kernel can be
    /// arbitrarily slow (`EllipticPi` integrates adaptively).
    Opaque {
        name_hash: u64,
        parity: Option<bool>,
    },
}

impl FuncCode {
    fn of(name: &str) -> FuncCode {
        const ELEMENTARY: [&str; 17] = [
            "sin", "cos", "tan", "exp", "log", "ln", "sqrt", "sinh", "cosh", "tanh", "asin",
            "acos", "atan", "asinh", "acosh", "atanh", "abs",
        ];
        if let Some(i) = ELEMENTARY.iter().position(|&n| n == name) {
            return FuncCode::Elementary(i as u8);
        }
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        FuncCode::Opaque {
            name_hash: h,
            parity: super::special_values::parity(name),
        }
    }

    fn screen(self, vals: &[Option<f64>]) -> Option<f64> {
        match self {
            FuncCode::Elementary(i) => {
                let [Some(a)] = vals else {
                    return None;
                };
                let a = *a;
                Some(match i {
                    0 => a.sin(),
                    1 => a.cos(),
                    2 => a.tan(),
                    3 => a.exp(),
                    4 | 5 => a.ln(),
                    6 => a.sqrt(),
                    7 => a.sinh(),
                    8 => a.cosh(),
                    9 => a.tanh(),
                    10 => a.asin(),
                    11 => a.acos(),
                    12 => a.atan(),
                    13 => a.asinh(),
                    14 => a.acosh(),
                    15 => a.atanh(),
                    _ => a.abs(),
                })
            }
            FuncCode::Opaque { name_hash, parity } => {
                let mut h = name_hash;
                let mut sign = 1.0;
                for (k, v) in vals.iter().enumerate() {
                    let mut v = (*v)?;
                    if k == 0 && parity.is_some() && v < 0.0 {
                        v = -v;
                        if parity == Some(true) {
                            sign = -1.0;
                        }
                    }
                    if k == 0 && parity == Some(true) && v == 0.0 {
                        return Some(0.0);
                    }
                    // Ignore the last ~20 mantissa bits, so two spellings of
                    // one argument that round differently still agree.
                    h ^= v.to_bits() & !0xF_FFFF;
                    h = h.wrapping_mul(0x0100_0000_01b3).rotate_left(17);
                }
                Some(sign * (0.5 + (mix64(h) >> 11) as f64 / (1u64 << 53) as f64))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// zero_test: the public decision procedure
// ---------------------------------------------------------------------------

/// How a [`ZeroTest`] reached its verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZeroStatus {
    /// Proved identically zero (wherever defined) by exact symbolic means.
    ProvedZero,
    /// Proved not identically zero: an exact normal form, or a rigorous ball
    /// excluding zero at an admissible point.
    ProvedNonzero,
    /// Every rigorous ball evaluation contained zero with a tiny radius.
    /// Evidence, not proof — the verdict is `None`.
    ProbablyZero,
    /// Nothing conclusive.
    Unknown,
}

impl ZeroStatus {
    /// `Some(true)`, `Some(false)` or `None` (undetermined).
    pub fn verdict(self) -> Option<bool> {
        match self {
            ZeroStatus::ProvedZero => Some(true),
            ZeroStatus::ProvedNonzero => Some(false),
            _ => None,
        }
    }

    /// Stable lower-case name (`"proved_zero"`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            ZeroStatus::ProvedZero => "proved_zero",
            ZeroStatus::ProvedNonzero => "proved_nonzero",
            ZeroStatus::ProbablyZero => "probably_zero",
            ZeroStatus::Unknown => "unknown",
        }
    }
}

/// Result of [`zero_test`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ZeroTest {
    /// The verdict's provenance.
    pub status: ZeroStatus,
    /// `"simplify"`, `"rational_normal_form"`, `"gammasimp"`, `"arb_ball"`,
    /// `"arb_sampling"`, `"complex_f64_sampling"` or `"none"`.
    pub method: &'static str,
    /// Human-readable evidence lines (sample points and enclosures).
    pub evidence: Vec<String>,
}

impl ZeroTest {
    fn new(status: ZeroStatus, method: &'static str, evidence: Vec<String>) -> Self {
        ZeroTest {
            status,
            method,
            evidence,
        }
    }

    /// `Some(true)` / `Some(false)` when proved, `None` otherwise.
    pub fn verdict(&self) -> Option<bool> {
        self.status.verdict()
    }
}

/// Decide whether `expr` is identically zero on the domain of its symbols.
///
/// * **True** (`ProvedZero`) only from exact symbolic reasoning: the
///   simplifier reaches the literal `0`, or the rational normal form over
///   atoms (module docs) has a zero numerator.
/// * **False** (`ProvedNonzero`) from an exact non-zero normal form, or from
///   a rigorous Arb ball excluding zero at an exact rational point inside
///   every symbol's declared domain that also satisfies every assumption
///   fact.
/// * **None** otherwise, with status `ProbablyZero` when every rigorous ball
///   contained zero with radius below `2^-100` at several admissible points
///   (or, for complex-valued input, double-precision samples all vanished),
///   and `Unknown` when not even that was found.
///
/// `assumptions` adds `Positive` / `NonZero` facts and predicates; sample
/// points that cannot be shown to satisfy them are skipped.  Floats in the
/// input never produce a proof either way.
pub fn zero_test(
    expr: ExprId,
    pool: &ExprPool,
    assumptions: Option<&super::assumptions::AssumptionContext>,
) -> ZeroTest {
    let has_float = contains_float(expr, pool);
    // 1. Simplify (with the assumptions, when given).
    let simplified = match assumptions {
        Some(ctx) if !ctx.facts().is_empty() || !ctx.predicates().is_empty() => {
            super::assumptions::simplify_with_assumptions(expr, pool, ctx).value
        }
        _ => super::engine::simplify(expr, pool).value,
    };
    if !has_float {
        if let Some(r) = rational_literal(simplified, pool) {
            return if r == 0 {
                ZeroTest::new(ZeroStatus::ProvedZero, "simplify", vec![])
            } else {
                ZeroTest::new(
                    ZeroStatus::ProvedNonzero,
                    "simplify",
                    vec![format!("simplifies to the constant {r}")],
                )
            };
        }
        // 2. Rational normal form.
        let mut positive = HashSet::new();
        if let Some(ctx) = assumptions {
            for f in ctx.facts() {
                if let SideCondition::Positive(e) = f {
                    positive.insert(*e);
                }
            }
        }
        // F25: products of Γ that the reflection / duplication / recurrence
        // identities collapse (`Γ(x)Γ(1−x) − π/sin(πx)`) are invisible to the
        // normal form, which treats each `gamma(…)` as an independent atom.
        // `gammasimp` rewrites only by identities of meromorphic functions.
        let mut candidates = vec![simplified, expr];
        if let Ok(g) = crate::specfun::gammasimp(simplified, pool) {
            if g != simplified {
                if rational_literal(g, pool).is_some_and(|r| r == 0) {
                    return ZeroTest::new(ZeroStatus::ProvedZero, "gammasimp", vec![]);
                }
                candidates.insert(0, g);
            }
        }
        for candidate in candidates {
            let mut nz = Normalizer::new(pool, Limits::ZERO_TEST).with_positive(positive.clone());
            match nz.decide(candidate) {
                Some(true) => {
                    return ZeroTest::new(ZeroStatus::ProvedZero, "rational_normal_form", vec![])
                }
                Some(false) if assumptions.map_or(true, |c| c.predicates().is_empty()) => {
                    return ZeroTest::new(
                        ZeroStatus::ProvedNonzero,
                        "rational_normal_form",
                        vec!["non-zero rational function of independent variables".into()],
                    )
                }
                _ => {}
            }
        }
    }
    // 3. Numeric evidence.
    let mut t = numeric_test(simplified, pool, assumptions);
    if has_float && t.status == ZeroStatus::ProvedNonzero {
        t.status = ZeroStatus::Unknown;
        t.evidence
            .push("input contains floating-point literals: no proof either way".into());
    }
    t
}

/// Domain of a symbol.
fn symbol_domain(s: ExprId, pool: &ExprPool) -> Domain {
    pool.with(s, |d| match d {
        ExprData::Symbol { domain, .. } => *domain,
        _ => Domain::Complex,
    })
}

/// Deterministic pseudo-random generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// A rational in `(0, 3]` with a small odd-ish denominator.
    fn positive(&mut self) -> Rational {
        let den = 7 + (self.next() % 90) as i64;
        let num = 1 + (self.next() % (3 * den as u64)) as i64;
        Rational::from((num, den))
    }
}

/// Memoised rigorous ball evaluation (the shared [`IntervalEval`] walks the
/// expression as a tree).
fn ball_eval(
    expr: ExprId,
    pool: &ExprPool,
    bind: &HashMap<ExprId, ArbBall>,
    prec: u32,
) -> Option<ArbBall> {
    fn go(
        e: ExprId,
        pool: &ExprPool,
        bind: &HashMap<ExprId, ArbBall>,
        prec: u32,
        memo: &mut HashMap<ExprId, Option<ArbBall>>,
        reg: &crate::primitive::PrimitiveRegistry,
        depth: u32,
    ) -> Option<ArbBall> {
        if let Some(v) = memo.get(&e) {
            return v.clone();
        }
        if depth > 2000 {
            return None;
        }
        let v = match pool.get(e) {
            ExprData::Integer(n) => Some(ArbBall::from_integer(&n.0, prec)),
            ExprData::Rational(r) => Some(ArbBall::from_rational(&r.0, prec)),
            ExprData::Symbol { .. } => match bind.get(&e) {
                Some(b) => Some(b.clone()),
                None if crate::eval::symbols::is_real_constant(e, pool) => {
                    IntervalEval::new(prec).eval(e, pool)
                }
                None => None,
            },
            ExprData::Add(a) => {
                let mut acc = ArbBall::from_f64(0.0, prec);
                for x in a {
                    acc = acc + go(x, pool, bind, prec, memo, reg, depth + 1)?;
                }
                Some(acc)
            }
            ExprData::Mul(a) => {
                let mut acc = ArbBall::from_f64(1.0, prec);
                for x in a {
                    acc = acc * go(x, pool, bind, prec, memo, reg, depth + 1)?;
                }
                Some(acc)
            }
            ExprData::Pow { base, exp } => {
                let b = go(base, pool, bind, prec, memo, reg, depth + 1)?;
                match integer_literal(exp, pool) {
                    Some(n) => Some(b.powi(n)),
                    None => {
                        let x = go(exp, pool, bind, prec, memo, reg, depth + 1)?;
                        Some(b.pow_f(&x))
                    }
                }
            }
            ExprData::Func { name, args } if !args.is_empty() => {
                let mut vals = Vec::with_capacity(args.len());
                for x in args {
                    vals.push(go(x, pool, bind, prec, memo, reg, depth + 1)?);
                }
                reg.numeric_ball(&name, &vals)
            }
            _ => {
                // Piecewise and friends: the shared evaluator.
                let mut ev = IntervalEval::new(prec);
                for (k, v) in bind {
                    ev.bind(*k, v.clone());
                }
                ev.eval(e, pool)
            }
        };
        memo.insert(e, v.clone());
        v
    }
    use std::sync::OnceLock;
    static REG: OnceLock<crate::primitive::PrimitiveRegistry> = OnceLock::new();
    let reg = REG.get_or_init(crate::primitive::PrimitiveRegistry::dispatch_registry);
    let mut memo = HashMap::new();
    go(expr, pool, bind, prec, &mut memo, reg, 0)
}

fn excludes_zero(b: &ArbBall) -> bool {
    b.lo() > 0 || b.hi() < 0
}

fn tiny(b: &ArbBall) -> bool {
    b.contains(0.0) && b.rad_f64() < 1e-30 && b.mid_f64().abs() < 1e-30
}

/// Does the sample point satisfy every assumption fact and predicate?
fn admissible(
    pool: &ExprPool,
    bind: &HashMap<ExprId, ArbBall>,
    assumptions: Option<&super::assumptions::AssumptionContext>,
) -> bool {
    let Some(ctx) = assumptions else {
        return true;
    };
    for f in ctx.facts() {
        let ok = match f {
            SideCondition::Positive(e) => {
                ball_eval(*e, pool, bind, 128).is_some_and(|b| b.lo() > 0)
            }
            SideCondition::NonZero(e) => {
                ball_eval(*e, pool, bind, 128).is_some_and(|b| excludes_zero(&b))
            }
            SideCondition::InDomain(e, d) => match d {
                Domain::Positive => ball_eval(*e, pool, bind, 128).is_some_and(|b| b.lo() > 0),
                Domain::NonNegative => ball_eval(*e, pool, bind, 128).is_some_and(|b| b.lo() >= 0),
                Domain::NonZero => {
                    ball_eval(*e, pool, bind, 128).is_some_and(|b| excludes_zero(&b))
                }
                Domain::Real | Domain::Complex => true,
                Domain::Integer => false,
            },
        };
        if !ok {
            return false;
        }
    }
    for &p in ctx.predicates() {
        let mut ev = IntervalEval::new(128);
        for (k, v) in bind {
            ev.bind(*k, v.clone());
        }
        match ev.eval(p, pool) {
            Some(b) if b.contains(1.0) && !b.contains(0.0) => {}
            _ => return false,
        }
    }
    true
}

fn numeric_test(
    expr: ExprId,
    pool: &ExprPool,
    assumptions: Option<&super::assumptions::AssumptionContext>,
) -> ZeroTest {
    let syms = free_symbols(expr, pool);
    let mut evidence = Vec::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut near_zero = 0usize;
    let mut tried = 0usize;
    let points = if syms.is_empty() { 1 } else { 16 };
    for k in 0..points {
        let mut point: HashMap<ExprId, Rational> = HashMap::new();
        let mut desc = Vec::new();
        for &s in &syms {
            let d = symbol_domain(s, pool);
            let v = match d {
                Domain::Integer => {
                    let n = 1 + (rng.next() % 6) as i64;
                    let n = if k >= 8 && rng.next() % 2 == 0 { -n } else { n };
                    Rational::from(n)
                }
                Domain::Positive | Domain::NonNegative => rng.positive(),
                _ => {
                    let v = rng.positive();
                    if k >= 8 && rng.next() % 2 == 0 {
                        -v
                    } else {
                        v
                    }
                }
            };
            desc.push(format!("{}={}", pool.display(s), v));
            point.insert(s, v);
        }
        let ball_at = |prec: u32| -> HashMap<ExprId, ArbBall> {
            point
                .iter()
                .map(|(k, v)| (*k, ArbBall::from_rational(v, prec)))
                .collect()
        };
        if !admissible(pool, &ball_at(256), assumptions) {
            continue;
        }
        let mut last = None;
        for prec in [128u32, 512, 2048] {
            let b = ball_at(prec);
            let Some(val) = ball_eval(expr, pool, &b, prec) else {
                break;
            };
            if excludes_zero(&val) {
                evidence.push(format!(
                    "at {}: value in [{:.6e} ± {:.1e}]",
                    if desc.is_empty() {
                        "(constant)".into()
                    } else {
                        desc.join(", ")
                    },
                    val.mid_f64(),
                    val.rad_f64()
                ));
                return ZeroTest::new(
                    ZeroStatus::ProvedNonzero,
                    if syms.is_empty() {
                        "arb_ball"
                    } else {
                        "arb_sampling"
                    },
                    evidence,
                );
            }
            let small = tiny(&val);
            last = Some(val);
            if small {
                break;
            }
        }
        if let Some(val) = last {
            tried += 1;
            if tiny(&val) {
                near_zero += 1;
                if evidence.len() < 4 {
                    evidence.push(format!(
                        "at {}: |value| <= {:.1e}",
                        if desc.is_empty() {
                            "(constant)".into()
                        } else {
                            desc.join(", ")
                        },
                        val.mid_f64().abs() + val.rad_f64()
                    ));
                }
            }
        }
    }
    let needed = if syms.is_empty() { 1 } else { 3 };
    if tried > 0 && near_zero == tried && near_zero >= needed {
        return ZeroTest::new(ZeroStatus::ProbablyZero, "arb_sampling", evidence);
    }
    if tried == 0 {
        // Complex-valued (or not ball-evaluable): double-precision evidence
        // only, never a proof.
        if let Some(t) = complex_samples(expr, pool, &syms) {
            return t;
        }
    }
    ZeroTest::new(ZeroStatus::Unknown, "none", evidence)
}

fn complex_samples(expr: ExprId, pool: &ExprPool, syms: &[ExprId]) -> Option<ZeroTest> {
    use crate::eval::ComplexF64;
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut evidence = Vec::new();
    let mut ok = 0;
    for _ in 0..6 {
        let mut bind = HashMap::new();
        for &s in syms {
            let v = rng.positive().to_f64();
            bind.insert(s, ComplexF64::new(v, 0.0));
        }
        let v = crate::eval::eval_complex_f64(expr, pool, &bind).ok()?;
        let m = (v.re * v.re + v.im * v.im).sqrt();
        if !m.is_finite() {
            return None;
        }
        if m > 1e-9 {
            evidence.push(format!("double-precision sample |value| = {m:.3e}"));
            return Some(ZeroTest::new(
                ZeroStatus::Unknown,
                "complex_f64_sampling",
                evidence,
            ));
        }
        ok += 1;
    }
    evidence.push(format!("{ok} double-precision samples vanished"));
    Some(ZeroTest::new(
        ZeroStatus::ProbablyZero,
        "complex_f64_sampling",
        evidence,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zt(e: ExprId, pool: &ExprPool) -> ZeroStatus {
        zero_test(e, pool, None).status
    }

    #[test]
    fn exact_zeros() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let s = |e| pool.func("sin", vec![e]);
        let c = |e| pool.func("cos", vec![e]);
        let sq = |e| pool.pow(e, pool.integer(2_i32));
        let neg = |e| pool.mul(vec![pool.integer(-1_i32), e]);
        // sin² + cos² − 1
        let e = pool.add(vec![sq(s(x)), sq(c(x)), pool.integer(-1_i32)]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedZero);
        // sin(x+y) − sin x cos y − cos x sin y
        let e = pool.add(vec![
            s(pool.add(vec![x, y])),
            neg(pool.mul(vec![s(x), c(y)])),
            neg(pool.mul(vec![c(x), s(y)])),
        ]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedZero);
        // cos(2x) − 2cos²x + 1
        let e = pool.add(vec![
            c(pool.mul(vec![pool.integer(2_i32), x])),
            neg(pool.mul(vec![pool.integer(2_i32), sq(c(x))])),
            pool.integer(1_i32),
        ]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedZero);
        // −a²(b−1) + a²b − a²
        let e = pool.add(vec![
            neg(pool.mul(vec![sq(x), pool.add(vec![y, pool.integer(-1_i32)])])),
            pool.mul(vec![sq(x), y]),
            neg(sq(x)),
        ]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedZero);
    }

    #[test]
    fn non_zeros() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // sin² − cos² is not zero.
        let s2 = pool.pow(pool.func("sin", vec![x]), pool.integer(2_i32));
        let c2 = pool.pow(pool.func("cos", vec![x]), pool.integer(2_i32));
        let e = pool.add(vec![s2, pool.mul(vec![pool.integer(-1_i32), c2])]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedNonzero);
        // x² − 1
        let e = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedNonzero);
        // sqrt(x^2) − x is not identically zero over the reals.
        let e = pool.add(vec![
            pool.func("sqrt", vec![pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedNonzero);
    }

    /// The report's `simplify_trig` symptom: `m·c·sin²θ + m·c·cos²θ − m·c`
    /// stopped at `m·c + m·c·(−1)`.  Every entry point now reaches `0`.
    #[test]
    fn every_simplifier_reaches_zero_on_the_pythagorean_residue() {
        let pool = ExprPool::new();
        let m = pool.symbol("m", Domain::Real);
        let c = pool.symbol("c", Domain::Real);
        let th = pool.symbol("th", Domain::Real);
        let mc = pool.mul(vec![m, c]);
        let sq = |e| pool.pow(e, pool.integer(2_i32));
        let e = pool.add(vec![
            pool.mul(vec![mc, sq(pool.func("sin", vec![th]))]),
            pool.mul(vec![mc, sq(pool.func("cos", vec![th]))]),
            pool.mul(vec![pool.integer(-1_i32), mc]),
        ]);
        let zero = pool.integer(0_i32);
        assert_eq!(crate::simplify::simplify(e, &pool).value, zero);
        assert_eq!(crate::simplify::simplify_trig(e, &pool).value, zero);
        assert_eq!(crate::simplify::simplify_egraph(e, &pool).value, zero);
    }

    /// A zero sum nested under a function is found too, and the parent
    /// folds: `sin(x(y−1) − xy + x) → sin(0) → 0`.
    #[test]
    fn nested_zero_sum_folds_its_parent() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let inner = pool.add(vec![
            pool.mul(vec![x, pool.add(vec![y, pool.integer(-1_i32)])]),
            pool.mul(vec![pool.integer(-1_i32), x, y]),
            x,
        ]);
        let e = pool.func("sin", vec![inner]);
        let r = crate::simplify::simplify(e, &pool);
        assert_eq!(r.value, pool.integer(0_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "zero_sum"));
    }

    /// The sweep skips what cannot hide a zero sum: no `Add` at all, or only
    /// sums of plain monomials (`(x+1)¹⁴`); and a subtree a complete sweep
    /// found clean is not walked again.  A zero sum nested under plain
    /// structure is still found.
    #[test]
    fn zero_sweep_candidates_and_clean_memo() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let one = pool.integer(1_i32);
        let x1 = pool.add(vec![x, one]);
        // No `Add` / only plain sums: not candidates.
        let d = pool.mul(vec![
            pool.integer(2_i32),
            x,
            pool.func("cos", vec![pool.pow(x, pool.integer(2_i32))]),
        ]);
        assert!(!pool.zero_candidate(d));
        assert!(!pool.zero_candidate(pool.pow(x1, pool.integer(14_i32))));
        let poly = pool.add(vec![
            pool.mul(vec![x, y]),
            pool.mul(vec![pool.integer(-3_i32), x]),
        ]);
        assert!(!pool.zero_candidate(poly));
        // `i` and a symbolic exponent are not plain.
        let i_sum = pool.add(vec![pool.imaginary_unit(), x]);
        assert!(pool.zero_candidate(i_sum));
        assert!(pool.zero_candidate(pool.add(vec![pool.pow(x, y), one])));
        // A sum with a non-plain term is, and a clean sweep retires it.
        let s = pool.add(vec![pool.func("sin", vec![x]), x1]);
        let e = pool.mul(vec![y, s]);
        assert!(pool.zero_candidate(e));
        assert!(zero_sum_pass(e, &pool).is_none());
        assert!(!pool.zero_candidate(e) && !pool.zero_candidate(s));
        // The memoised screen value is the subtree's value.
        let v = pool.zero_screen(e).flatten().unwrap();
        let sx = screen_value("x", Domain::Real);
        let sy = screen_value("y", Domain::Real);
        assert!((v - sy * (sx.sin() + sx + 1.0)).abs() < 1e-12);
        // `sin(x)·(x+1) − sin(x)·x − sin(x)` is zero, below a plain product.
        let sinx = pool.func("sin", vec![x]);
        let z = pool.add(vec![
            pool.mul(vec![sinx, x1]),
            pool.mul(vec![pool.integer(-1_i32), sinx, x]),
            pool.mul(vec![pool.integer(-1_i32), sinx]),
        ]);
        let outer = pool.mul(vec![y, pool.func("exp", vec![z])]);
        let (r, log) = zero_sum_pass(outer, &pool).expect("nested zero found");
        assert_eq!(
            r,
            pool.mul(vec![y, pool.func("exp", vec![pool.integer(0_i32)])])
        );
        assert!(log.steps().iter().any(|s| s.rule_name == ZeroSum::NAME));
    }

    /// `(x+y)²/(x+y) − x − y` is zero where defined.
    #[test]
    fn zero_sum_through_a_quotient() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let s = pool.add(vec![x, y]);
        let e = pool.add(vec![
            pool.mul(vec![
                pool.pow(s, pool.integer(2_i32)),
                pool.pow(s, pool.integer(-1_i32)),
            ]),
            pool.mul(vec![pool.integer(-1_i32), x]),
            pool.mul(vec![pool.integer(-1_i32), y]),
        ]);
        let r = crate::simplify::simplify(e, &pool);
        assert_eq!(r.value, pool.integer(0_i32));
    }

    /// Infinities are not values the identities cover.
    #[test]
    fn zero_sum_leaves_infinities_alone() {
        let pool = ExprPool::new();
        let oo = pool.pos_infinity();
        let e = pool.add(vec![oo, pool.mul(vec![pool.integer(-1_i32), oo])]);
        assert_ne!(
            crate::simplify::simplify(e, &pool).value,
            pool.integer(0_i32)
        );
        assert_eq!(zero_test(e, &pool, None).verdict(), None);
    }

    /// Sampling alone never proves zero.
    #[test]
    fn sampling_is_never_a_proof_of_zero() {
        let pool = ExprPool::new();
        let p = pool.symbol("p", Domain::Positive);
        // W(p)·e^{W(p)} − p = 0, invisible to the normal form.
        let w = pool.func("lambert_w", vec![p]);
        let e = pool.add(vec![
            pool.mul(vec![w, pool.func("exp", vec![w])]),
            pool.mul(vec![pool.integer(-1_i32), p]),
        ]);
        let t = zero_test(e, &pool, None);
        assert_ne!(t.status, ZeroStatus::ProvedZero);
        assert_ne!(t.status, ZeroStatus::ProvedNonzero);
    }

    #[test]
    #[ignore]
    fn perf_zero_pass_on_cofactor_sums() {
        use std::time::Instant;
        let build = |pool: &ExprPool| {
            let a: Vec<Vec<ExprId>> = (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| pool.symbol(format!("a{i}{j}"), Domain::Real))
                        .collect()
                })
                .collect();
            let neg = |e| pool.mul(vec![pool.integer(-1_i32), e]);
            let m2 = |p: ExprId, q: ExprId, r: ExprId, s: ExprId| {
                pool.add(vec![pool.mul(vec![p, q]), neg(pool.mul(vec![r, s]))])
            };
            let t1 = m2(a[1][1], a[2][2], a[1][2], a[2][1]);
            let t2 = m2(a[1][0], a[2][2], a[1][2], a[2][0]);
            let t3 = m2(a[1][0], a[2][1], a[1][1], a[2][0]);
            pool.add(vec![
                pool.mul(vec![a[0][0], t1]),
                neg(pool.mul(vec![a[0][1], t2])),
                pool.mul(vec![a[0][2], t3]),
            ])
        };
        let n = 20_000;
        let mut rules = crate::simplify::engine::rules_for_config(&Default::default());
        rules.retain(|r| r.name() != ZeroSum::NAME);
        let mut base = crate::simplify::engine::rules_for_config(&Default::default());
        let special: Vec<&str> = crate::simplify::special_values::special_value_rules()
            .iter()
            .map(|r| r.name())
            .collect();
        base.retain(|r| r.name() != ZeroSum::NAME && !special.contains(&r.name()));
        for label in ["build", "base", "without", "with", "pass-only"] {
            let t0 = Instant::now();
            for _ in 0..n {
                let pool = ExprPool::new();
                let e = build(&pool);
                match label {
                    "build" => {}
                    "base" => {
                        crate::simplify::simplify_with(e, &pool, &base, Default::default());
                    }
                    "without" => {
                        crate::simplify::simplify_with(e, &pool, &rules, Default::default());
                    }
                    "with" => {
                        crate::simplify::simplify(e, &pool);
                    }
                    _ => {
                        zero_sum_pass(e, &pool);
                    }
                }
            }
            eprintln!(
                "{label}: {:.2} us",
                t0.elapsed().as_secs_f64() / n as f64 * 1e6
            );
        }
    }

    #[test]
    fn positive_domain_makes_sqrt_square_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Positive);
        let e = pool.add(vec![
            pool.func("sqrt", vec![pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        assert_ne!(zt(e, &pool), ZeroStatus::ProvedNonzero);
    }

    /// Launch report row 3: `φ² − φ − 1` was "proved" non-zero by the normal
    /// form, which treated `GoldenRatio` as an indeterminate.
    #[test]
    fn golden_ratio_is_algebraic() {
        let pool = ExprPool::new();
        let phi = pool.symbol("GoldenRatio", Domain::Real);
        let e = pool.add(vec![
            pool.pow(phi, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(-1_i32), phi]),
            pool.integer(-1_i32),
        ]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedZero);
        // φ³ = 2φ + 1, next to an explicit √5.
        let sqrt5 = pool.func("sqrt", vec![pool.integer(5_i32)]);
        let e = pool.add(vec![
            pool.pow(phi, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), sqrt5]),
            pool.integer(-2_i32),
        ]);
        assert_eq!(zt(e, &pool), ZeroStatus::ProvedZero);
    }

    /// No independence theorem is known for γ, G, A or γₙ: a non-zero verdict
    /// may come only from a ball, never from the normal form.
    #[test]
    fn other_named_constants_are_not_indeterminates() {
        let pool = ExprPool::new();
        for name in ["EulerGamma", "Catalan", "Glaisher", "StieltjesGamma1"] {
            let c = pool.symbol(name, Domain::Real);
            let e = pool.add(vec![
                pool.pow(c, pool.integer(2_i32)),
                pool.mul(vec![pool.integer(-1_i32), c]),
                pool.integer(-1_i32),
            ]);
            let t = zero_test(e, &pool, None);
            assert_eq!(t.status, ZeroStatus::ProvedNonzero, "{name}");
            assert_ne!(t.method, "rational_normal_form", "{name}");
            let mut nz = Normalizer::new(&pool, Limits::ZERO_TEST);
            assert_eq!(nz.decide(e), None, "{name}");
            let ex = pool.func("exp", vec![c]);
            let e = pool.add(vec![ex, pool.integer(-2_i32)]);
            let mut nz = Normalizer::new(&pool, Limits::ZERO_TEST);
            assert_eq!(nz.decide(e), None, "exp({name})");
        }
    }
}
