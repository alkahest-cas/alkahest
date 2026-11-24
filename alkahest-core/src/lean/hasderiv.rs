//! `HasDerivAt` certificates for indefinite integrals.
//!
//! For an antiderivative `F` of an integrand `f` this emits a Lean 4 theorem
//!
//! ```text
//! theorem alkahest_integral (x : ℝ) (h₁ : …) … :
//!     HasDerivAt (fun (x : ℝ) => F) (f) x
//! ```
//!
//! whose hypotheses carve out the domain on which `F` is differentiable and
//! the identity holds (`log` arguments `≠ 0`, `sqrt` radicands `> 0`,
//! denominators `≠ 0`, `arcsin` arguments in `(-1, 1)`, `cos ≠ 0` under
//! `tan`, …).
//!
//! The proof is built in three mechanical layers, never by search:
//!
//! 1. **Derivative term.** `F`'s expression tree is walked once and the
//!    matching Mathlib lemma is applied at every node (`HasDerivAt.add`,
//!    `.mul`, `.pow`, `.inv`, `.log`, `.exp`, `.sin`, `.cos`, `.sqrt`,
//!    `.arctan`, `.rpow_const`, `Real.hasDerivAt_tan` / `_arcsin` /
//!    `_arccos` via `.comp`). The emitter also prints the derivative those
//!    lemmas produce, *verbatim*, and ascribes it, so the shape Lean sees is
//!    known exactly.
//! 2. **Algebra, decided here.** The printed derivative `G` and the integrand
//!    `f` are mapped to polynomials over `ℚ` in *atoms* (`x`, `sin u`,
//!    `√v`, `(u)⁻¹`, …). The emitter computes, in Rust, explicit cofactors
//!    `cⱼ` with `G - f = Σ cⱼ·(Lⱼ - Rⱼ)` as a field identity, where each
//!    `Lⱼ = Rⱼ` is a Mathlib fact (`√v ^ 2 = v`, `sin² + cos² = 1`,
//!    `sin (a + b) = …`, `tan = sin / cos`, `exp (a + b) = …`,
//!    `v ^ (3/2) = √v ^ 3`, …). If the residual does not reduce to zero the
//!    certificate is **withheld** — the emitter never hands Lean an identity
//!    it has not already checked.
//!    The facts include `u * u⁻¹ = 1` for every denominator `u`, so the
//!    cofactors also clear denominators.
//! 3. **Closing.** `linear_combination Σ cⱼ * factⱼ` — a single `ring1`
//!    check of a polynomial identity the emitter already verified. No
//!    `field_simp` / `simp` whose normal forms could drift.
//!
//! No `sorry`, no `admit`: every failure is a withheld certificate with a
//! specific reason code (see [`IntegrationCertWithheld`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rug::{Integer, Rational};

use crate::kernel::{ExprData, ExprId, ExprPool};

/// Why an indefinite-integral `HasDerivAt` certificate was withheld.
///
/// The string is a stable reason code, reported by the Python
/// `certificate_status["detail"]`.
pub type IntegrationCertWithheld = &'static str;

/// The antiderivative or integrand uses a special function (`erf`, `Si`,
/// `Ei`, `li`, …) that Mathlib (at the pinned v4.36.0) does not define.
pub const WITHHELD_SPECIAL_FUNCTION: IntegrationCertWithheld = "withheld_special_function";
/// The antiderivative is a `RootSum` over the roots of a polynomial.
pub const WITHHELD_ROOTSUM: IntegrationCertWithheld = "withheld_rootsum";
/// A function with no encoded derivative lemma (`atanh`, `atan2`, …).
pub const WITHHELD_UNSUPPORTED_FUNCTION: IntegrationCertWithheld = "withheld_unsupported_function";
/// A node the printer cannot render faithfully (float, huge literal,
/// piecewise, a symbol that is not a Lean identifier, symbolic exponent …).
pub const WITHHELD_UNSUPPORTED_NODE: IntegrationCertWithheld = "withheld_unsupported_node";
/// A constant whose sign the emitter cannot prove (`√c` with `c` not
/// syntactically positive, a constant denominator of unknown sign).
pub const WITHHELD_CONSTANT_SIGN: IntegrationCertWithheld = "withheld_constant_sign";
/// `G - f` did not reduce to zero modulo the encoded identities: the
/// derivative identity needs a relation the emitter does not know
/// (`√(x²) = |x|`, `log` of a product, …).
pub const WITHHELD_IDENTITY: IntegrationCertWithheld = "withheld_identity_not_reduced";
/// The domain hypotheses have no common point among the sample points: the
/// theorem could be vacuous, so it is not emitted.
pub const WITHHELD_EMPTY_DOMAIN: IntegrationCertWithheld = "withheld_empty_domain";
/// The certificate (or an intermediate polynomial) exceeds the size budget.
pub const WITHHELD_TOO_LARGE: IntegrationCertWithheld = "withheld_certificate_too_large";

pub(super) const MAX_TERMS: usize = 6_000;
/// Size budget for one certificate. Under Mathlib v4.9.0 a `linear_combination`
/// above 36 KB took `ring1` minutes; under v4.36.0 the largest certificates the
/// calculus suites produce (~740 KB of combination) check in under 2.5 min on
/// one core including the Mathlib import, most in under a minute.
const MAX_SRC_BYTES: usize = 1_000_000;
const MAX_COMBINATION_BYTES: usize = 800_000;
pub(super) const MAX_ROUNDS: usize = 400;

// ===========================================================================
// Sparse polynomials over ℚ
// ===========================================================================

/// A monomial: `(variable, exponent)` pairs, sorted by variable, exponents > 0.
pub(super) type Mono = Vec<(u32, u32)>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct Poly(pub(super) BTreeMap<Mono, Rational>);

pub(super) fn mono_mul(a: &Mono, b: &Mono) -> Mono {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            out.push((a[i].0, a[i].1 + b[j].1));
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

pub(super) fn exp_of(m: &Mono, v: u32) -> u32 {
    m.iter().find(|&&(w, _)| w == v).map_or(0, |&(_, e)| e)
}

impl Poly {
    pub(super) fn zero() -> Self {
        Poly(BTreeMap::new())
    }
    pub(super) fn konst(c: Rational) -> Self {
        let mut m = BTreeMap::new();
        if c != 0 {
            m.insert(Vec::new(), c);
        }
        Poly(m)
    }
    pub(super) fn int(c: i64) -> Self {
        Self::konst(Rational::from(c))
    }
    pub(super) fn one() -> Self {
        Self::int(1)
    }
    pub(super) fn var(v: u32) -> Self {
        let mut m = BTreeMap::new();
        m.insert(vec![(v, 1)], Rational::from(1));
        Poly(m)
    }
    pub(super) fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
    pub(super) fn add_term(&mut self, m: Mono, c: Rational) {
        use std::collections::btree_map::Entry;
        match self.0.entry(m) {
            Entry::Vacant(e) => {
                if c != 0 {
                    e.insert(c);
                }
            }
            Entry::Occupied(mut e) => {
                *e.get_mut() += c;
                if *e.get() == 0 {
                    e.remove();
                }
            }
        }
    }
    pub(super) fn add(&self, o: &Poly) -> Poly {
        let mut r = self.clone();
        for (m, c) in &o.0 {
            r.add_term(m.clone(), c.clone());
        }
        r
    }
    pub(super) fn sub(&self, o: &Poly) -> Poly {
        let mut r = self.clone();
        for (m, c) in &o.0 {
            r.add_term(m.clone(), Rational::from(-c));
        }
        r
    }
    pub(super) fn neg(&self) -> Poly {
        Poly::zero().sub(self)
    }
    pub(super) fn scale(&self, k: &Rational) -> Poly {
        if *k == 0 {
            return Poly::zero();
        }
        Poly(
            self.0
                .iter()
                .map(|(m, c)| (m.clone(), Rational::from(c * k)))
                .collect(),
        )
    }
    pub(super) fn mul(&self, o: &Poly) -> Result<Poly, IntegrationCertWithheld> {
        if self.len().saturating_mul(o.len()) > MAX_TERMS * 40 {
            return Err(WITHHELD_TOO_LARGE);
        }
        let mut r = Poly::zero();
        for (ma, ca) in &self.0 {
            for (mb, cb) in &o.0 {
                r.add_term(mono_mul(ma, mb), Rational::from(ca * cb));
            }
        }
        if r.len() > MAX_TERMS {
            return Err(WITHHELD_TOO_LARGE);
        }
        Ok(r)
    }
    pub(super) fn pow(&self, n: u32) -> Result<Poly, IntegrationCertWithheld> {
        let mut r = Poly::one();
        for _ in 0..n {
            r = r.mul(self)?;
        }
        Ok(r)
    }
    pub(super) fn deg_in(&self, v: u32) -> u32 {
        self.0.keys().map(|m| exp_of(m, v)).max().unwrap_or(0)
    }
    pub(super) fn has_var(&self, v: u32) -> bool {
        self.0.keys().any(|m| exp_of(m, v) > 0)
    }
    /// `[c₀, c₁, …]` with `self = Σ cₖ vᵏ`.
    pub(super) fn split(&self, v: u32) -> Vec<Poly> {
        let d = self.deg_in(v) as usize;
        let mut out = vec![Poly::zero(); d + 1];
        for (m, c) in &self.0 {
            let k = exp_of(m, v) as usize;
            let rest: Mono = m.iter().copied().filter(|&(w, _)| w != v).collect();
            out[k].add_term(rest, c.clone());
        }
        out
    }
    pub(super) fn vars(&self) -> BTreeSet<u32> {
        self.0
            .keys()
            .flat_map(|m| m.iter().map(|&(v, _)| v))
            .collect()
    }
    /// `Some(q)` when `self = q · other` for a rational `q ≠ 0`.
    pub(super) fn ratio_to(&self, other: &Poly) -> Option<Rational> {
        let (m0, c0) = other.0.iter().next()?;
        let c = self.0.get(m0)?;
        let q = Rational::from(c / c0);
        (other.scale(&q) == *self).then_some(q)
    }
}

/// `p(v) - p(e) = (v - e)·Δ`; returns `(p(e), Δ)`.
pub(super) fn substitute(p: &Poly, v: u32, e: &Poly) -> Result<(Poly, Poly), IntegrationCertWithheld> {
    let cs = p.split(v);
    let vv = Poly::var(v);
    let mut epows = vec![Poly::one()];
    let mut vpows = vec![Poly::one()];
    for k in 1..cs.len() {
        let next_e = epows[k - 1].mul(e)?;
        epows.push(next_e);
        let next_v = vpows[k - 1].mul(&vv)?;
        vpows.push(next_v);
    }
    let mut pe = Poly::zero();
    let mut delta = Poly::zero();
    for (k, c) in cs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        pe = pe.add(&c.mul(&epows[k])?);
        if k >= 1 {
            let mut s = Poly::zero();
            for i in 0..k {
                s = s.add(&vpows[i].mul(&epows[k - 1 - i])?);
            }
            delta = delta.add(&c.mul(&s)?);
        }
    }
    Ok((pe, delta))
}

/// Reduce `p` by `v^d = tail`: returns `(r, q)` with `p = r + q·(v^d - tail)`
/// and `deg_v r < d`.
pub(super) fn reduce_power(
    p: &Poly,
    v: u32,
    d: u32,
    tail: &Poly,
) -> Result<(Poly, Poly), IntegrationCertWithheld> {
    let mut cur = p.clone();
    let mut q = Poly::zero();
    for _ in 0..64 {
        if cur.deg_in(v) < d {
            return Ok((cur, q));
        }
        let mut low = Poly::zero();
        let mut high = Poly::zero();
        for (m, c) in &cur.0 {
            let k = exp_of(m, v);
            if k >= d {
                let mut m2: Mono = m.iter().copied().filter(|&(w, _)| w != v).collect();
                if k > d {
                    m2.push((v, k - d));
                    m2.sort_unstable();
                }
                high.add_term(m2, c.clone());
            } else {
                low.add_term(m.clone(), c.clone());
            }
        }
        q = q.add(&high);
        cur = low.add(&high.mul(tail)?);
    }
    Err(WITHHELD_TOO_LARGE)
}

// ===========================================================================
// Atoms, facts, relations
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
enum Fn1 {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Atan,
    Asin,
    Acos,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    /// `erf`, by its defining integral (Mathlib has no `erf`; see [`SPECIAL_DEFS`]).
    Erf,
    /// `erfi`, by its defining integral.
    Erfi,
    /// The sine integral `Si`, by its defining integral `∫₀ˣ sinc`.
    Si,
    /// `Real.sinc` — only introduced by the derivative of `Si`.
    Sinc,
}

impl Fn1 {
    fn from_name(name: &str) -> Option<Fn1> {
        Some(match name {
            "sin" => Fn1::Sin,
            "cos" => Fn1::Cos,
            "tan" => Fn1::Tan,
            "exp" => Fn1::Exp,
            "log" => Fn1::Log,
            "sqrt" => Fn1::Sqrt,
            "atan" => Fn1::Atan,
            "asin" => Fn1::Asin,
            "acos" => Fn1::Acos,
            "sinh" => Fn1::Sinh,
            "cosh" => Fn1::Cosh,
            "tanh" => Fn1::Tanh,
            "asinh" => Fn1::Asinh,
            "acosh" => Fn1::Acosh,
            "erf" => Fn1::Erf,
            "erfi" => Fn1::Erfi,
            "Si" => Fn1::Si,
            _ => return None,
        })
    }
    fn lean(self) -> &'static str {
        match self {
            Fn1::Sin => "Real.sin",
            Fn1::Cos => "Real.cos",
            Fn1::Tan => "Real.tan",
            Fn1::Exp => "Real.exp",
            Fn1::Log => "Real.log",
            Fn1::Sqrt => "Real.sqrt",
            Fn1::Atan => "Real.arctan",
            Fn1::Asin => "Real.arcsin",
            Fn1::Acos => "Real.arccos",
            Fn1::Sinh => "Real.sinh",
            Fn1::Cosh => "Real.cosh",
            Fn1::Tanh => "Real.tanh",
            Fn1::Asinh => "Real.arsinh",
            Fn1::Acosh => "Real.arcosh",
            Fn1::Erf => "alkahestErf",
            Fn1::Erfi => "alkahestErfi",
            Fn1::Si => "alkahestSi",
            Fn1::Sinc => "Real.sinc",
        }
    }
}

pub(super) fn special_function_reason(name: &str) -> IntegrationCertWithheld {
    match name {
        "erf" | "erfc" | "erfi" | "Si" | "Ci" | "Shi" | "Chi" | "Ei" | "li" | "Li" | "polylog"
        | "fresnels" | "fresnelc" | "FresnelS" | "FresnelC" | "gamma" | "uppergamma"
        | "lowergamma" | "EllipticF" | "EllipticE" | "EllipticPi" | "elliptic_f" | "elliptic_e"
        | "hyper" | "besselj" | "bessely" | "airyai" | "airybi" | "lambertw" | "LambertW"
        | "zeta" | "dilog" => WITHHELD_SPECIAL_FUNCTION,
        _ => WITHHELD_UNSUPPORTED_FUNCTION,
    }
}

#[derive(Clone, Debug)]
enum AKind {
    /// The integration variable, a parameter, or `π`: no relations.
    Free,
    /// `f(arg)` for a unary elementary function.
    Fun { f: Fn1, arg: Poly, arg_lean: String },
    /// `(u)⁻¹`.
    Inv { u_lean: String },
    /// `base ^ ((1:ℝ) / b)`, `b ≥ 3` (the `b = 2` root is `Real.sqrt`); its
    /// `ρ ^ b = base` rule is registered when the atom is made.
    Root,
}

#[derive(Clone, Debug)]
struct Atom {
    lean: String,
    kind: AKind,
    depth: u32,
}

/// A numerically checkable domain condition (for the non-vacuity check).
#[derive(Clone, Copy, Debug)]
enum Cond {
    Ne0(ExprId),
    Pos(ExprId),
    GtM1(ExprId),
    Lt1(ExprId),
    Gt1(ExprId),
    CosNe0(ExprId),
}

#[derive(Clone, Debug)]
struct Fact {
    name: String,
    prop: String,
    /// `None` for a hypothesis (binder), `Some(proof)` for a `have`.
    proof: Option<String>,
    cond: Option<Cond>,
}

#[derive(Clone, Debug)]
enum RelKind {
    /// `v = e`.
    Subst { v: u32, e: Poly },
    /// `v ^ d = tail`.
    Power { v: u32, d: u32, tail: Poly },
    /// `u * w = 1` (`w` the inverse atom of `u`).
    Inverse { w: u32, u: Poly },
}

#[derive(Clone, Debug)]
struct Rel {
    name: String,
    lhs: String,
    rhs: String,
    proof: String,
    kind: RelKind,
    /// Emitted even with a zero cofactor (other facts' proofs cite it).
    always: bool,
}

/// One node of the derivative computation.
struct Node {
    /// The `HasDerivAt` proof term.
    proof: String,
    /// The derivative exactly as the lemmas state it.
    dlean: String,
    /// Its polynomial model.
    dpoly: Poly,
}

fn rat_lean(q: &Rational) -> Result<String, IntegrationCertWithheld> {
    let n = q.numer().to_i64().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
    let d = q.denom().to_i64().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
    Ok(if d == 1 {
        format!("({n} : ℝ)")
    } else {
        format!("({n} / {d} : ℝ)")
    })
}

struct Ctx<'a> {
    pool: &'a ExprPool,
    var: ExprId,
    var_name: String,
    params: BTreeSet<String>,
    atoms: Vec<Atom>,
    by_lean: HashMap<String, u32>,
    facts: Vec<Fact>,
    fact_by_prop: HashMap<String, String>,
    rels: Vec<Rel>,
    lean_memo: HashMap<ExprId, String>,
    poly_memo: HashMap<ExprId, Poly>,
    /// sqrt atom → proof of `0 ≤ radicand`.
    sqrt_nonneg: HashMap<u32, String>,
    /// Atom → an earlier atom with the same function and an equal argument.
    twin_of: HashMap<u32, u32>,
    /// `(twin atom, index of its congrArg fact)`.
    twin_rel: Vec<(u32, usize)>,
    /// `(name, lhs, rhs, proof)` rewrites applied to the goal `G = f`.
    goal_rw: Vec<(String, String, String, String)>,
}

impl<'a> Ctx<'a> {
    fn new(pool: &'a ExprPool, var: ExprId) -> Result<Self, IntegrationCertWithheld> {
        let var_name = pool.with(var, |d| match d {
            ExprData::Symbol { name, .. } => Some(name.clone()),
            _ => None,
        });
        let var_name = var_name.ok_or(WITHHELD_UNSUPPORTED_NODE)?;
        if !super::is_lean_binder_name(&var_name) {
            return Err(WITHHELD_UNSUPPORTED_NODE);
        }
        let mut c = Ctx {
            pool,
            var,
            var_name: var_name.clone(),
            params: BTreeSet::new(),
            atoms: Vec::new(),
            by_lean: HashMap::new(),
            facts: Vec::new(),
            fact_by_prop: HashMap::new(),
            rels: Vec::new(),
            lean_memo: HashMap::new(),
            poly_memo: HashMap::new(),
            sqrt_nonneg: HashMap::new(),
            twin_of: HashMap::new(),
            twin_rel: Vec::new(),
            goal_rw: Vec::new(),
        };
        // Atom 0 is the integration variable.
        c.new_atom(format!("({var_name} : ℝ)"), AKind::Free, 0);
        Ok(c)
    }

    fn new_atom(&mut self, lean: String, kind: AKind, depth: u32) -> u32 {
        let id = self.atoms.len() as u32;
        self.by_lean.insert(lean.clone(), id);
        self.atoms.push(Atom { lean, kind, depth });
        id
    }

    fn depth_of(&self, p: &Poly) -> u32 {
        p.vars()
            .iter()
            .map(|&v| self.atoms[v as usize].depth)
            .max()
            .unwrap_or(0)
    }

    // ------------------------------------------------------------------ facts

    fn hyp(&mut self, prop: String, cond: Cond) -> String {
        if let Some(n) = self.fact_by_prop.get(&prop) {
            return n.clone();
        }
        let name = format!(
            "h{}",
            self.facts.iter().filter(|f| f.proof.is_none()).count() + 1
        );
        self.fact_by_prop.insert(prop.clone(), name.clone());
        self.facts.push(Fact {
            name: name.clone(),
            prop,
            proof: None,
            cond: Some(cond),
        });
        name
    }

    fn have(&mut self, prop: String, proof: String) -> String {
        if let Some(n) = self.fact_by_prop.get(&prop) {
            return n.clone();
        }
        let name = format!(
            "n{}",
            self.facts.iter().filter(|f| f.proof.is_some()).count() + 1
        );
        self.fact_by_prop.insert(prop.clone(), name.clone());
        self.facts.push(Fact {
            name: name.clone(),
            prop,
            proof: Some(proof),
            cond: None,
        });
        name
    }

    fn is_const(&self, e: ExprId) -> bool {
        !self.depends_on_any_symbol(e)
    }

    fn depends_on_any_symbol(&self, e: ExprId) -> bool {
        self.pool.with(e, |d| match d {
            ExprData::Symbol { name, .. } => name != "pi",
            ExprData::Add(xs) | ExprData::Mul(xs) | ExprData::Func { args: xs, .. } => {
                xs.iter().any(|&c| self.depends_on_any_symbol(c))
            }
            ExprData::Pow { base, exp } => {
                self.depends_on_any_symbol(*base) || self.depends_on_any_symbol(*exp)
            }
            _ => false,
        })
    }

    fn depends_on_var(&self, e: ExprId) -> bool {
        if e == self.var {
            return true;
        }
        self.pool.with(e, |d| match d {
            ExprData::Add(xs) | ExprData::Mul(xs) | ExprData::Func { args: xs, .. } => {
                xs.iter().any(|&c| self.depends_on_var(c))
            }
            ExprData::Pow { base, exp } => self.depends_on_var(*base) || self.depends_on_var(*exp),
            _ => false,
        })
    }

    /// Name of a fact `0 < e`.
    fn pos_fact(&mut self, e: ExprId) -> Result<String, IntegrationCertWithheld> {
        let prop = format!("0 < {}", self.lean(e)?);
        if surely_positive(e, self.pool) {
            return Ok(self.have(prop, "by positivity".into()));
        }
        if self.is_const(e) {
            return Err(WITHHELD_CONSTANT_SIGN);
        }
        Ok(self.hyp(prop, Cond::Pos(e)))
    }

    /// Name of a fact `e ≠ 0`.
    fn ne0_fact(&mut self, e: ExprId) -> Result<String, IntegrationCertWithheld> {
        let prop = format!("{} ≠ 0", self.lean(e)?);
        if surely_positive(e, self.pool) {
            return Ok(self.have(prop, "by positivity".into()));
        }
        let literal = self.pool.with(e, |d| match d {
            ExprData::Integer(n) => Some(n.0 != 0),
            ExprData::Rational(r) => Some(r.0 != 0),
            _ => None,
        });
        if let Some(nonzero) = literal {
            return if nonzero {
                Ok(self.have(prop, "by norm_num".into()))
            } else {
                Err(WITHHELD_CONSTANT_SIGN)
            };
        }
        if self.is_const(e) {
            return Err(WITHHELD_CONSTANT_SIGN);
        }
        Ok(self.hyp(prop, Cond::Ne0(e)))
    }

    // ---------------------------------------------------------------- printer

    fn lean(&mut self, e: ExprId) -> Result<String, IntegrationCertWithheld> {
        if let Some(s) = self.lean_memo.get(&e) {
            return Ok(s.clone());
        }
        let s = self.lean_uncached(e)?;
        self.lean_memo.insert(e, s.clone());
        Ok(s)
    }

    fn lean_uncached(&mut self, e: ExprId) -> Result<String, IntegrationCertWithheld> {
        enum Shape {
            Int(Integer),
            Rat(Rational),
            Sym(String),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow(ExprId, ExprId),
            Func(String, Vec<ExprId>),
            RootSum,
            Other,
        }
        let shape = self.pool.with(e, |d| match d {
            ExprData::Integer(n) => Shape::Int(n.0.clone()),
            ExprData::Rational(r) => Shape::Rat(r.0.clone()),
            ExprData::Symbol { name, .. } => Shape::Sym(name.clone()),
            ExprData::Add(xs) => Shape::Add(xs.clone()),
            ExprData::Mul(xs) => Shape::Mul(xs.clone()),
            ExprData::Pow { base, exp } => Shape::Pow(*base, *exp),
            ExprData::Func { name, args } => Shape::Func(name.clone(), args.clone()),
            ExprData::RootSum { .. } => Shape::RootSum,
            _ => Shape::Other,
        });
        Ok(match shape {
            Shape::Int(n) => {
                let v = n.to_i64().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                format!("({v} : ℝ)")
            }
            Shape::Rat(r) => rat_lean(&r)?,
            Shape::Sym(name) => {
                if name == "pi" {
                    "Real.pi".to_string()
                } else if name == self.var_name {
                    format!("({name} : ℝ)")
                } else {
                    if !super::is_lean_binder_name(&name)
                        || name.starts_with('h') && name[1..].chars().all(|c| c.is_ascii_digit())
                        || name.starts_with('n') && name[1..].chars().all(|c| c.is_ascii_digit())
                        || name.starts_with('r') && name[1..].chars().all(|c| c.is_ascii_digit())
                        || matches!(name.as_str(), "hd" | "hd0" | "key" | "EulerGamma")
                    {
                        return Err(WITHHELD_UNSUPPORTED_NODE);
                    }
                    self.params.insert(name.clone());
                    format!("({name} : ℝ)")
                }
            }
            Shape::Add(xs) => {
                let parts: Result<Vec<String>, _> = xs.iter().map(|&a| self.lean(a)).collect();
                format!("({})", parts?.join(" + "))
            }
            Shape::Mul(xs) => {
                let parts: Result<Vec<String>, _> = xs.iter().map(|&a| self.lean(a)).collect();
                format!("({})", parts?.join(" * "))
            }
            Shape::Pow(b, x) => {
                let bs = self.lean(b)?;
                let ex = self.pool.with(x, |d| match d {
                    ExprData::Integer(n) => Some(Ok(n.0.clone())),
                    ExprData::Rational(r) => Some(Err(r.0.clone())),
                    _ => None,
                });
                match ex {
                    Some(Ok(n)) => {
                        let n = n.to_i64().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                        if n == -1 {
                            format!("({bs})⁻¹")
                        } else if n < 0 {
                            format!("({bs})⁻¹ ^ ({} : ℕ)", -n)
                        } else {
                            format!("({bs}) ^ ({n} : ℕ)")
                        }
                    }
                    Some(Err(r)) => format!("({bs}) ^ {}", rat_lean(&r)?),
                    None => return Err(WITHHELD_UNSUPPORTED_NODE),
                }
            }
            Shape::Func(name, args) => {
                if name == "abs" && args.len() == 1 {
                    // Printed for the statement only: `log |a|` is proved
                    // through `Real.log_abs`; any other `abs` is withheld by
                    // the derivative pass.
                    return Ok(format!("abs ({})", self.lean(args[0])?));
                }
                let f = Fn1::from_name(&name).ok_or_else(|| special_function_reason(&name))?;
                if args.len() != 1 {
                    return Err(WITHHELD_UNSUPPORTED_NODE);
                }
                format!("{} ({})", f.lean(), self.lean(args[0])?)
            }
            Shape::RootSum => return Err(WITHHELD_ROOTSUM),
            Shape::Other => return Err(WITHHELD_UNSUPPORTED_NODE),
        })
    }

    // ------------------------------------------------------------ atom makers

    fn fun_atom(&mut self, f: Fn1, arg_lean: &str, arg: Poly) -> u32 {
        let lean = format!("{} ({arg_lean})", f.lean());
        if let Some(&id) = self.by_lean.get(&lean) {
            return id;
        }
        let depth = 1 + self.depth_of(&arg);
        // An atom with the same function and an equal argument but a
        // different spelling is related by `congrArg`. Arguments are compared
        // through earlier twins, so `log (x + √(x² + 1·-1))` meets
        // `log (x + √(x² + -1))`.
        let norm = self.canon_poly(&arg);
        let twin = self.atoms.iter().position(|a| {
            matches!(&a.kind, AKind::Fun { f: g, arg: q, .. } if *g == f && self.canon_poly(q) == norm)
        });
        let id = self.new_atom(
            lean.clone(),
            AKind::Fun {
                f,
                arg: arg.clone(),
                arg_lean: arg_lean.to_string(),
            },
            depth,
        );
        if let Some(t) = twin {
            let t = self.canon(t as u32);
            let (t_lean, t_arg, t_poly) = match &self.atoms[t as usize].kind {
                AKind::Fun { arg_lean, arg, .. } => (
                    self.atoms[t as usize].lean.clone(),
                    arg_lean.clone(),
                    arg.clone(),
                ),
                _ => unreachable!(),
            };
            let arg_proof = if arg == t_poly {
                format!("(by ring : ({arg_lean}) = ({t_arg}))")
            } else {
                // `arg - t_arg = Σ cᵢ·(twinᵢ - canonᵢ)` over the inner twins.
                let mut diff = arg.sub(&t_poly);
                let mut terms = Vec::new();
                for (v, ri) in self.twin_rel.clone() {
                    if !diff.has_var(v) {
                        continue;
                    }
                    let Ok((rest, delta)) = substitute(&diff, v, &Poly::var(self.canon(v))) else {
                        break;
                    };
                    diff = rest;
                    let Ok(c) = self.print_poly(&delta) else {
                        break;
                    };
                    terms.push(format!("({c}) * {}", self.rels[ri].name));
                }
                format!(
                    "(show ({arg_lean}) = ({t_arg}) by linear_combination {})",
                    terms.join(" + ")
                )
            };
            self.twin_of.insert(id, t);
            let name = format!("r{}", self.rels.len() + 1);
            self.twin_rel.push((id, self.rels.len()));
            self.rels.push(Rel {
                name,
                lhs: lean,
                rhs: t_lean,
                proof: format!("congrArg {} {arg_proof}", f.lean()),
                kind: RelKind::Subst {
                    v: id,
                    e: Poly::var(t),
                },
                always: true,
            });
            if f == Fn1::Sqrt {
                if let Some(p) = self.sqrt_nonneg.get(&t).cloned() {
                    self.sqrt_nonneg.insert(id, p);
                }
            }
        }
        id
    }

    /// `p` with every twin atom replaced by its canonical atom.
    fn canon_poly(&self, p: &Poly) -> Poly {
        if !p.vars().iter().any(|v| self.twin_of.contains_key(v)) {
            return p.clone();
        }
        let mut out = Poly::zero();
        for (m, c) in &p.0 {
            let mut mono: Mono = m.iter().map(|&(v, e)| (self.canon(v), e)).collect();
            mono.sort_unstable();
            let mut merged: Mono = Vec::new();
            for (v, e) in mono {
                match merged.last_mut() {
                    Some(last) if last.0 == v => last.1 += e,
                    _ => merged.push((v, e)),
                }
            }
            out.add_term(merged, c.clone());
        }
        out
    }

    /// The polynomial of `1 / u`, where `u_lean` is a spelling of `u` with a
    /// `≠ 0` fact already in scope.
    ///
    /// This mirrors how `ring` sees an inverse: the inverse of a *monomial*
    /// distributes (`(c·a²)⁻¹ = c⁻¹·(a⁻¹)²`), while the inverse of any other
    /// polynomial is one atom per normal form — so inverse atoms are keyed by
    /// atom (monomial case) or by polynomial (otherwise), never by spelling.
    fn inv_poly(&mut self, u: &Poly, u_lean: &str) -> Result<Poly, IntegrationCertWithheld> {
        if let Some(c) = u_as_constant(u) {
            if c == 0 {
                return Err(WITHHELD_CONSTANT_SIGN);
            }
            return Ok(Poly::konst(Rational::from(c.recip_ref())));
        }
        let u_fact = self
            .fact_by_prop
            .get(&format!("{u_lean} ≠ 0"))
            .cloned()
            .ok_or(WITHHELD_UNSUPPORTED_NODE)?;
        if u.len() == 1 {
            let (m, c) =
                u.0.iter()
                    .next()
                    .map(|(m, c)| (m.clone(), c.clone()))
                    .unwrap();
            let mut out = Poly::konst(Rational::from(c.recip_ref()));
            for &(v, e) in &m {
                if let AKind::Inv { u_lean: inner_lean } = self.atoms[v as usize].kind.clone() {
                    // `(u⁻¹)⁻¹`: `ring` keeps it as an atom, so state `inv_inv`.
                    let inner = self.rels.iter().find_map(|r| match &r.kind {
                        RelKind::Inverse { w, u } if *w == v => Some(u.clone()),
                        _ => None,
                    });
                    let inner = inner.ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                    let lean = format!("({})⁻¹", self.atoms[v as usize].lean);
                    let ww = match self.by_lean.get(&lean) {
                        Some(&id) => id,
                        None => {
                            let depth = self.atoms[v as usize].depth + 1;
                            let id = self.new_atom(lean.clone(), AKind::Free, depth);
                            self.push_rel(
                                lean,
                                format!("({inner_lean})"),
                                "inv_inv _".into(),
                                RelKind::Subst {
                                    v: id,
                                    e: inner.clone(),
                                },
                            );
                            id
                        }
                    };
                    out = out.mul(&Poly::var(ww).pow(e)?)?;
                    continue;
                }
                let a_lean = self.atoms[v as usize].lean.clone();
                let prop = format!("{a_lean} ≠ 0");
                // A factor of a nonzero monomial is nonzero: rewriting it to 0
                // turns the monomial into a product with a zero factor.
                let fact = match self.fact_by_prop.get(&prop) {
                    Some(f) => f.clone(),
                    None => self.have(prop, format!("fun h => {u_fact} (by rw [h]; norm_num)")),
                };
                let w = self.inverse_atom(&Poly::var(v), &a_lean, &fact);
                self.rationalize_constant_root(v, w, &fact)?;
                out = out.mul(&Poly::var(w).pow(e)?)?;
            }
            return Ok(out);
        }
        Ok(Poly::var(self.inverse_atom(u, u_lean, &u_fact)))
    }

    /// `(√c)⁻¹ = √c · (1/c)` for a positive rational `c`: removes the
    /// inverse atom before any denominator is cleared, which keeps the
    /// cofactors of `atan`-style answers (`(√3)⁻¹ · atan((2x - 1)/√3)`) small.
    fn rationalize_constant_root(
        &mut self,
        r: u32,
        w: u32,
        fact: &str,
    ) -> Result<(), IntegrationCertWithheld> {
        let c = match &self.atoms[r as usize].kind {
            AKind::Fun {
                f: Fn1::Sqrt, arg, ..
            } => match u_as_constant(arg) {
                Some(c) if c > 0 => c,
                _ => return Ok(()),
            },
            _ => return Ok(()),
        };
        if self.has_rule(w) {
            return Ok(());
        }
        let inv_c = Rational::from(c.recip_ref());
        let (rl, wl) = (
            self.atoms[r as usize].lean.clone(),
            self.atoms[w as usize].lean.clone(),
        );
        let icl = rat_lean(&inv_c)?;
        let radicand = self.arg_lean_of(r);
        self.push_rel(
            wl.clone(),
            format!("{rl} * {icl}"),
            format!(
                "by linear_combination (-({wl}) * {icl}) * (Real.sq_sqrt (show (0:ℝ) ≤ {radicand} by norm_num)) + (({rl}) * {icl}) * (mul_inv_cancel₀ {fact})"
            ),
            RelKind::Subst {
                v: w,
                e: Poly::var(r).scale(&inv_c),
            },
        );
        Ok(())
    }

    /// The atom `(u)⁻¹` with its `u * u⁻¹ = 1` fact (one per polynomial `u`).
    fn inverse_atom(&mut self, u: &Poly, u_lean: &str, fact: &str) -> u32 {
        let existing = self.rels.iter().find_map(|r| match &r.kind {
            RelKind::Inverse { w, u: q } if q == u => Some(*w),
            _ => None,
        });
        if let Some(w) = existing {
            return w;
        }
        let lean = format!("({u_lean})⁻¹");
        let depth = 1 + self.depth_of(u);
        let id = self.new_atom(
            lean.clone(),
            AKind::Inv {
                u_lean: u_lean.to_string(),
            },
            depth,
        );
        self.push_rel(
            format!("({u_lean}) * {lean}"),
            "1".into(),
            format!("mul_inv_cancel₀ {fact}"),
            RelKind::Inverse {
                w: id,
                u: u.clone(),
            },
        );
        id
    }

    fn sqrt_atom(&mut self, arg_lean: &str, arg: Poly, nonneg: Option<String>) -> u32 {
        let id = self.fun_atom(Fn1::Sqrt, arg_lean, arg);
        if let Some(p) = nonneg {
            self.sqrt_nonneg.entry(id).or_insert(p);
        }
        id
    }

    // ------------------------------------------------------------- values

    /// Polynomial model of the *value* of `e`, registering the facts its
    /// Lean spelling needs (denominators `≠ 0`, radicands `> 0`).
    fn vpoly(&mut self, e: ExprId) -> Result<Poly, IntegrationCertWithheld> {
        if let Some(p) = self.poly_memo.get(&e) {
            return Ok(p.clone());
        }
        let p = self.vpoly_uncached(e)?;
        self.poly_memo.insert(e, p.clone());
        Ok(p)
    }

    fn vpoly_uncached(&mut self, e: ExprId) -> Result<Poly, IntegrationCertWithheld> {
        let lean = self.lean(e)?;
        enum Shape {
            Int(Integer),
            Rat(Rational),
            Sym,
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow(ExprId, ExprId),
            Func(String, ExprId),
        }
        let shape = self.pool.with(e, |d| match d {
            ExprData::Integer(n) => Some(Shape::Int(n.0.clone())),
            ExprData::Rational(r) => Some(Shape::Rat(r.0.clone())),
            ExprData::Symbol { .. } => Some(Shape::Sym),
            ExprData::Add(xs) => Some(Shape::Add(xs.clone())),
            ExprData::Mul(xs) => Some(Shape::Mul(xs.clone())),
            ExprData::Pow { base, exp } => Some(Shape::Pow(*base, *exp)),
            ExprData::Func { name, args } if args.len() == 1 => {
                Some(Shape::Func(name.clone(), args[0]))
            }
            _ => None,
        });
        match shape.ok_or(WITHHELD_UNSUPPORTED_NODE)? {
            Shape::Int(n) => Ok(Poly::konst(Rational::from(n))),
            Shape::Rat(r) => Ok(Poly::konst(r)),
            Shape::Sym => {
                if e == self.var {
                    return Ok(Poly::var(0));
                }
                if let Some(&id) = self.by_lean.get(&lean) {
                    return Ok(Poly::var(id));
                }
                Ok(Poly::var(self.new_atom(lean, AKind::Free, 0)))
            }
            Shape::Add(xs) => {
                let mut acc = Poly::zero();
                for x in xs {
                    acc = acc.add(&self.vpoly(x)?);
                }
                Ok(acc)
            }
            Shape::Mul(xs) => {
                let mut acc = Poly::one();
                for x in xs {
                    let p = self.vpoly(x)?;
                    acc = acc.mul(&p)?;
                }
                Ok(acc)
            }
            Shape::Pow(b, x) => {
                let bp = self.vpoly(b)?;
                let ex = self.pool.with(x, |d| match d {
                    ExprData::Integer(n) => Some(Ok(n.0.clone())),
                    ExprData::Rational(r) => Some(Err(r.0.clone())),
                    _ => None,
                });
                match ex.ok_or(WITHHELD_UNSUPPORTED_NODE)? {
                    Ok(n) => {
                        let n = n.to_i64().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                        if n.unsigned_abs() > 64 {
                            return Err(WITHHELD_TOO_LARGE);
                        }
                        if n >= 0 {
                            bp.pow(n as u32)
                        } else {
                            self.ne0_fact(b)?;
                            let bl = self.lean(b)?;
                            let w = self.inv_poly(&bp, &bl)?;
                            w.pow((-n) as u32)
                        }
                    }
                    Err(r) => self.rpow_poly(b, &bp, &r),
                }
            }
            Shape::Func(name, a) => {
                let f = Fn1::from_name(&name).ok_or_else(|| special_function_reason(&name))?;
                let ap = self.vpoly(a)?;
                let al = self.lean(a)?;
                match f {
                    Fn1::Sqrt => {
                        let h = self.pos_fact(a)?;
                        let id = self.sqrt_atom(&al, ap, Some(format!("({h}).le")));
                        Ok(Poly::var(id))
                    }
                    Fn1::Tan => {
                        if !self.is_const(a) {
                            self.hyp(format!("Real.cos ({al}) ≠ 0"), Cond::CosNe0(a));
                        } else {
                            return Err(WITHHELD_CONSTANT_SIGN);
                        }
                        Ok(Poly::var(self.fun_atom(f, &al, ap)))
                    }
                    Fn1::Asin | Fn1::Acos | Fn1::Acosh if self.is_const(a) => {
                        Err(WITHHELD_CONSTANT_SIGN)
                    }
                    Fn1::Acosh => {
                        // `Real.arcosh` agrees with the principal `acosh` only on `[1, ∞)`.
                        self.hyp(format!("1 < {al}"), Cond::Gt1(a));
                        Ok(Poly::var(self.fun_atom(f, &al, ap)))
                    }
                    Fn1::Log if self.is_const(a) => {
                        // `Real.log c` agrees with the complex log only for c > 0.
                        if !surely_positive(a, self.pool) {
                            return Err(WITHHELD_CONSTANT_SIGN);
                        }
                        Ok(Poly::var(self.fun_atom(f, &al, ap)))
                    }
                    _ => Ok(Poly::var(self.fun_atom(f, &al, ap))),
                }
            }
        }
    }

    /// `b ^ (1/q)` (`q ≥ 3`; every other rational power was respelled by
    /// [`respell_rational_powers`]) — a root atom with its `ρ ^ q = b` fact.
    fn rpow_poly(
        &mut self,
        b: ExprId,
        bp: &Poly,
        r: &Rational,
    ) -> Result<Poly, IntegrationCertWithheld> {
        if *r.numer() != 1 || *r.denom() <= 2 {
            return Err(WITHHELD_UNSUPPORTED_NODE);
        }
        let q = r.denom().to_u32().ok_or(WITHHELD_TOO_LARGE)?;
        if q > 12 {
            return Err(WITHHELD_TOO_LARGE);
        }
        let pos = self.pos_fact(b)?;
        let bl = self.lean(b)?;
        let lean = format!("({bl}) ^ {}", rat_lean(r)?);
        if let Some(&id) = self.by_lean.get(&lean) {
            return Ok(Poly::var(id));
        }
        let depth = 1 + self.depth_of(bp);
        let id = self.new_atom(lean.clone(), AKind::Root, depth);
        self.push_rel(
            format!("({lean}) ^ {q}"),
            bl.clone(),
            format!(
                "by\n    rw [← Real.rpow_natCast ({lean}) {q}, ← Real.rpow_mul ({pos}).le]\n    norm_num"
            ),
            RelKind::Power {
                v: id,
                d: q,
                tail: bp.clone(),
            },
        );
        self.have(
            format!("{lean} ≠ 0"),
            format!("(Real.rpow_pos_of_pos {pos} _).ne'"),
        );
        Ok(Poly::var(id))
    }

    // -------------------------------------------------------- derivatives

    fn deriv(&mut self, e: ExprId) -> Result<Node, IntegrationCertWithheld> {
        let x = self.var_name.clone();
        let val = self.lean(e)?;
        if !self.depends_on_var(e) {
            self.vpoly(e)?;
            return Ok(Node {
                proof: format!("(hasDerivAt_const {x} ({val}))"),
                dlean: "0".into(),
                dpoly: Poly::zero(),
            });
        }
        if e == self.var {
            return Ok(Node {
                proof: format!("(hasDerivAt_id' {x})"),
                dlean: "1".into(),
                dpoly: Poly::one(),
            });
        }
        enum Shape {
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow(ExprId, ExprId),
            Func(String, Vec<ExprId>),
            Other,
        }
        let shape = self.pool.with(e, |d| match d {
            ExprData::Add(xs) => Shape::Add(xs.clone()),
            ExprData::Mul(xs) => Shape::Mul(xs.clone()),
            ExprData::Pow { base, exp } => Shape::Pow(*base, *exp),
            ExprData::Func { name, args } => Shape::Func(name.clone(), args.clone()),
            _ => Shape::Other,
        });
        match shape {
            Shape::Add(xs) => {
                let mut it = xs.into_iter();
                let first = it.next().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                let mut acc = self.deriv(first)?;
                for c in it {
                    let n = self.deriv(c)?;
                    acc = Node {
                        proof: format!("(({}).add {})", acc.proof, n.proof),
                        dlean: format!("({}) + ({})", acc.dlean, n.dlean),
                        dpoly: acc.dpoly.add(&n.dpoly),
                    };
                }
                Ok(acc)
            }
            Shape::Mul(xs) => {
                let mut it = xs.into_iter();
                let first = it.next().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                let mut acc = self.deriv(first)?;
                let mut acc_val = self.lean(first)?;
                let mut acc_poly = self.vpoly(first)?;
                for c in it {
                    let n = self.deriv(c)?;
                    let cv = self.lean(c)?;
                    let cp = self.vpoly(c)?;
                    let dpoly = acc.dpoly.mul(&cp)?.add(&acc_poly.mul(&n.dpoly)?);
                    acc = Node {
                        proof: format!("(({}).mul {})", acc.proof, n.proof),
                        dlean: format!("({}) * ({cv}) + ({acc_val}) * ({})", acc.dlean, n.dlean),
                        dpoly,
                    };
                    acc_val = format!("{acc_val} * {cv}");
                    acc_poly = acc_poly.mul(&cp)?;
                }
                Ok(acc)
            }
            Shape::Pow(b, ex) => {
                if self.depends_on_var(ex) {
                    return Err(WITHHELD_UNSUPPORTED_NODE);
                }
                let exv = self.pool.with(ex, |d| match d {
                    ExprData::Integer(n) => Some(Ok(n.0.clone())),
                    ExprData::Rational(r) => Some(Err(r.0.clone())),
                    _ => None,
                });
                let nb = self.deriv(b)?;
                let bl = self.lean(b)?;
                let bp = self.vpoly(b)?;
                match exv.ok_or(WITHHELD_UNSUPPORTED_NODE)? {
                    Ok(n) => {
                        let n = n.to_i64().ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                        if n.unsigned_abs() > 64 {
                            return Err(WITHHELD_TOO_LARGE);
                        }
                        if n >= 1 {
                            let dpoly = bp
                                .pow((n - 1) as u32)?
                                .mul(&nb.dpoly)?
                                .scale(&Rational::from(n));
                            Ok(Node {
                                proof: format!("(({}).pow {n})", nb.proof),
                                dlean: format!(
                                    "(({n} : ℕ) : ℝ) * ({bl}) ^ (({n} : ℕ) - 1) * ({})",
                                    nb.dlean
                                ),
                                dpoly,
                            })
                        } else if n == 0 {
                            Err(WITHHELD_UNSUPPORTED_NODE)
                        } else {
                            let ne = self.ne0_fact(b)?;
                            let w = self.inv_poly(&bp, &bl)?;
                            let sq_lean = format!("({bl}) ^ (2 : ℕ)");
                            self.have(format!("{sq_lean} ≠ 0"), format!("pow_ne_zero 2 {ne}"));
                            let w2 = self.inv_poly(&bp.pow(2)?, &sq_lean)?;
                            let inv_d = nb.dpoly.neg().mul(&w2)?;
                            let inv_proof = format!("(({}).inv {ne})", nb.proof);
                            let inv_dlean = format!("(-({})) / (({bl}) ^ (2 : ℕ))", nb.dlean);
                            if n == -1 {
                                Ok(Node {
                                    proof: inv_proof,
                                    dlean: inv_dlean,
                                    dpoly: inv_d,
                                })
                            } else {
                                let k = -n;
                                let dpoly = w
                                    .pow((k - 1) as u32)?
                                    .mul(&inv_d)?
                                    .scale(&Rational::from(k));
                                Ok(Node {
                                    proof: format!("(({inv_proof}).pow {k})"),
                                    dlean: format!(
                                        "(({k} : ℕ) : ℝ) * (({bl})⁻¹) ^ (({k} : ℕ) - 1) * ({inv_dlean})"
                                    ),
                                    dpoly,
                                })
                            }
                        }
                    }
                    Err(r) => {
                        // Only `b ^ (1/q)`, q ≥ 3, survives respelling. Its
                        // derivative carries `b ^ (1/q - 1)`, which the proof
                        // rewrites to `(b ^ (1/q))⁻¹ ^ (q-1)` before `ring`
                        // sees it (two rational powers of one base would make
                        // `ring` compare real exponents by unfolding them).
                        let rho = self.rpow_poly(b, &bp, &r)?;
                        let pos = self.pos_fact(b)?;
                        let p_lean = rat_lean(&r)?;
                        let rho_lean = format!("({bl}) ^ {p_lean}");
                        let k = r.denom().to_u32().ok_or(WITHHELD_TOO_LARGE)? - 1;
                        let w = self.inv_poly(&rho, &rho_lean)?;
                        let atom_lean = format!("({bl}) ^ ({p_lean} - 1)");
                        if !self.goal_rw.iter().any(|g| g.1 == atom_lean) {
                            let name = format!("g{}", self.goal_rw.len() + 1);
                            self.goal_rw.push((
                                name,
                                atom_lean.clone(),
                                format!("({rho_lean})⁻¹ ^ {k}"),
                                format!(
                                    "by\n    rw [inv_pow ({rho_lean}) {k}, ← Real.rpow_natCast ({rho_lean}) {k}, ← Real.rpow_mul ({pos}).le, ← Real.rpow_neg ({pos}).le]\n    norm_num"
                                ),
                            ));
                        }
                        let dpoly = nb.dpoly.mul(&w.pow(k)?)?.scale(&r);
                        Ok(Node {
                            proof: format!(
                                "(({}).rpow_const (p := {p_lean}) (Or.inl ({pos}).ne'))",
                                nb.proof
                            ),
                            dlean: format!("({}) * {p_lean} * {atom_lean}", nb.dlean),
                            dpoly,
                        })
                    }
                }
            }
            Shape::Func(name, args) => {
                let f = Fn1::from_name(&name).ok_or_else(|| special_function_reason(&name))?;
                if args.len() != 1 {
                    return Err(WITHHELD_UNSUPPORTED_NODE);
                }
                let a = args[0];
                let na = self.deriv(a)?;
                let al = self.lean(a)?;
                let ap = self.vpoly(a)?;
                let da = na.dlean.clone();
                let dp = na.dpoly.clone();
                let pf = na.proof.clone();
                let node = |proof: String, dlean: String, dpoly: Poly| Node {
                    proof,
                    dlean,
                    dpoly,
                };
                Ok(match f {
                    Fn1::Exp => {
                        let e_atom = self.fun_atom(Fn1::Exp, &al, ap);
                        node(
                            format!("(({pf}).exp)"),
                            format!("Real.exp ({al}) * ({da})"),
                            Poly::var(e_atom).mul(&dp)?,
                        )
                    }
                    Fn1::Log => {
                        let ne = self.ne0_fact(a)?;
                        let w = self.inv_poly(&ap, &al)?;
                        self.fun_atom(Fn1::Log, &al, ap.clone());
                        node(
                            format!("(({pf}).log {ne})"),
                            format!("({da}) / ({al})"),
                            dp.mul(&w)?,
                        )
                    }
                    Fn1::Sin => {
                        let c = self.fun_atom(Fn1::Cos, &al, ap.clone());
                        self.fun_atom(Fn1::Sin, &al, ap);
                        node(
                            format!("(({pf}).sin)"),
                            format!("Real.cos ({al}) * ({da})"),
                            Poly::var(c).mul(&dp)?,
                        )
                    }
                    Fn1::Cos => {
                        let s = self.fun_atom(Fn1::Sin, &al, ap.clone());
                        self.fun_atom(Fn1::Cos, &al, ap);
                        node(
                            format!("(({pf}).cos)"),
                            format!("(-Real.sin ({al})) * ({da})"),
                            Poly::var(s).mul(&dp)?.neg(),
                        )
                    }
                    Fn1::Tan => {
                        let hc = self.hyp(format!("Real.cos ({al}) ≠ 0"), Cond::CosNe0(a));
                        let c = self.fun_atom(Fn1::Cos, &al, ap.clone());
                        self.fun_atom(Fn1::Tan, &al, ap);
                        let cl = self.atoms[c as usize].lean.clone();
                        let w = self.inv_poly(&Poly::var(c), &cl)?;
                        node(
                            format!("(HasDerivAt.comp (h := fun ({x} : ℝ) => {al}) {x} (Real.hasDerivAt_tan {hc}) {pf})"),
                            format!("(1 / Real.cos ({al}) ^ 2) * ({da})"),
                            w.pow(2)?.mul(&dp)?,
                        )
                    }
                    Fn1::Atan => {
                        self.fun_atom(Fn1::Atan, &al, ap.clone());
                        let den = format!("1 + ({al}) ^ 2");
                        self.have(format!("{den} ≠ 0"), "by positivity".into());
                        let w = self.inv_poly(&Poly::one().add(&ap.pow(2)?), &den)?;
                        node(
                            format!("(({pf}).arctan)"),
                            format!("(1 / ({den})) * ({da})"),
                            w.mul(&dp)?,
                        )
                    }
                    Fn1::Asin | Fn1::Acos => {
                        let h1 = self.hyp(format!("-1 < {al}"), Cond::GtM1(a));
                        let h2 = self.hyp(format!("{al} < 1"), Cond::Lt1(a));
                        self.fun_atom(f, &al, ap.clone());
                        let rad = format!("1 - ({al}) ^ 2");
                        let hp = self.have(format!("0 < {rad}"), "by nlinarith".into());
                        let rad_poly = Poly::one().sub(&ap.pow(2)?);
                        let r = self.sqrt_atom(&rad, rad_poly, Some(format!("({hp}).le")));
                        let rl = self.atoms[r as usize].lean.clone();
                        self.have(format!("{rl} ≠ 0"), format!("(Real.sqrt_pos.mpr {hp}).ne'"));
                        let w = self.inv_poly(&Poly::var(r), &rl)?;
                        let lemma = if f == Fn1::Asin {
                            "Real.hasDerivAt_arcsin"
                        } else {
                            "Real.hasDerivAt_arccos"
                        };
                        let core = format!("1 / Real.sqrt ({rad})");
                        let (dl, sign) = if f == Fn1::Asin {
                            (format!("({core}) * ({da})"), 1)
                        } else {
                            (format!("(-({core})) * ({da})"), -1)
                        };
                        node(
                            format!("(HasDerivAt.comp (h := fun ({x} : ℝ) => {al}) {x} ({lemma} (ne_of_gt {h1}) (ne_of_lt {h2})) {pf})"),
                            dl,
                            w.mul(&dp)?.scale(&Rational::from(sign)),
                        )
                    }
                    Fn1::Sinh => {
                        let c = self.fun_atom(Fn1::Cosh, &al, ap.clone());
                        self.fun_atom(Fn1::Sinh, &al, ap);
                        node(
                            format!("(({pf}).sinh)"),
                            format!("Real.cosh ({al}) * ({da})"),
                            Poly::var(c).mul(&dp)?,
                        )
                    }
                    Fn1::Cosh => {
                        let s = self.fun_atom(Fn1::Sinh, &al, ap.clone());
                        self.fun_atom(Fn1::Cosh, &al, ap);
                        node(
                            format!("(({pf}).cosh)"),
                            format!("Real.sinh ({al}) * ({da})"),
                            Poly::var(s).mul(&dp)?,
                        )
                    }
                    Fn1::Sqrt => {
                        let hp = self.pos_fact(a)?;
                        let r = self.sqrt_atom(&al, ap, Some(format!("({hp}).le")));
                        let rl = self.atoms[r as usize].lean.clone();
                        self.have(format!("{rl} ≠ 0"), format!("(Real.sqrt_pos.mpr {hp}).ne'"));
                        let w = self.inv_poly(&Poly::var(r), &rl)?;
                        node(
                            format!("(({pf}).sqrt ({hp}).ne')"),
                            format!("({da}) / (2 * Real.sqrt ({al}))"),
                            dp.mul(&w)?.scale(&Rational::from((1, 2))),
                        )
                    }
                    Fn1::Erf | Fn1::Erfi => {
                        // `erf' u = 2/√π · exp(-u²)`, `erfi' u = 2/√π · exp(u²)`
                        // (the lemmas in [`SPECIAL_DEFS`]). `π` and `√π` are
                        // spelled as the printer spells them, so they meet the
                        // answer's own `√π` atom.
                        self.fun_atom(f, &al, ap.clone());
                        let pi = match self.by_lean.get("Real.pi") {
                            Some(&id) => id,
                            None => self.new_atom("Real.pi".into(), AKind::Free, 0),
                        };
                        let r =
                            self.sqrt_atom("Real.pi", Poly::var(pi), Some("Real.pi_pos.le".into()));
                        let rl = self.atoms[r as usize].lean.clone();
                        self.have(format!("{rl} ≠ 0"), "by positivity".into());
                        let w = self.inv_poly(&Poly::var(r), &rl)?;
                        let (arg_l, arg_p, lemma) = if f == Fn1::Erf {
                            (
                                format!("-({al}) ^ 2"),
                                ap.pow(2)?.neg(),
                                "alkahestErf_hasDerivAt",
                            )
                        } else {
                            (format!("({al}) ^ 2"), ap.pow(2)?, "alkahestErfi_hasDerivAt")
                        };
                        let e = self.fun_atom(Fn1::Exp, &arg_l, arg_p);
                        let el = self.atoms[e as usize].lean.clone();
                        node(
                            format!("(HasDerivAt.comp (h := fun ({x} : ℝ) => {al}) {x} ({lemma} ({al})) {pf})"),
                            format!("(2 / {rl} * {el}) * ({da})"),
                            Poly::var(e)
                                .mul(&w)?
                                .mul(&dp)?
                                .scale(&Rational::from(2)),
                        )
                    }
                    Fn1::Si => {
                        // `Si' u = sinc u`, and `sinc u = sin u / u` for `u ≠ 0`.
                        self.fun_atom(Fn1::Si, &al, ap.clone());
                        let ne = self.ne0_fact(a)?;
                        let w = self.inv_poly(&ap, &al)?;
                        let sn = self.fun_atom(Fn1::Sin, &al, ap.clone());
                        let sc = self.fun_atom(Fn1::Sinc, &al, ap.clone());
                        if !self.has_rule(sc) {
                            let snl = self.atoms[sn as usize].lean.clone();
                            let scl = self.atoms[sc as usize].lean.clone();
                            self.push_rel(
                                scl,
                                format!("{snl} / ({al})"),
                                format!("Real.sinc_of_ne_zero {ne}"),
                                RelKind::Subst {
                                    v: sc,
                                    e: Poly::var(sn).mul(&w)?,
                                },
                            );
                        }
                        node(
                            format!("(HasDerivAt.comp (h := fun ({x} : ℝ) => {al}) {x} (alkahestSi_hasDerivAt ({al})) {pf})"),
                            format!("Real.sinc ({al}) * ({da})"),
                            Poly::var(sc).mul(&dp)?,
                        )
                    }
                    Fn1::Sinc => return Err(WITHHELD_UNSUPPORTED_FUNCTION),
                    Fn1::Tanh => {
                        // Mathlib has no `HasDerivAt.tanh`: differentiate
                        // `sinh u / cosh u` and move to `tanh u` along
                        // `Real.tanh_eq_sinh_div_cosh` (an everywhere-equality,
                        // so `congr_of_eventuallyEq` applies).
                        let s = self.fun_atom(Fn1::Sinh, &al, ap.clone());
                        let c = self.fun_atom(Fn1::Cosh, &al, ap.clone());
                        self.fun_atom(Fn1::Tanh, &al, ap);
                        let cl = self.atoms[c as usize].lean.clone();
                        self.have(format!("{cl} ≠ 0"), format!("(Real.cosh_pos ({al})).ne'"));
                        let w = self.inv_poly(&Poly::var(c), &cl)?;
                        let dpoly = Poly::var(c)
                            .pow(2)?
                            .sub(&Poly::var(s).pow(2)?)
                            .mul(&dp)?
                            .mul(&w.pow(2)?)?;
                        node(
                            format!(
                                "(HasDerivAt.congr_of_eventuallyEq (f₁ := fun ({x} : ℝ) => Real.tanh ({al})) \
                                 ((({pf}).sinh).fun_div (({pf}).cosh) (Real.cosh_pos _).ne') \
                                 (Filter.Eventually.of_forall fun ({x} : ℝ) => Real.tanh_eq_sinh_div_cosh ({al})))"
                            ),
                            format!(
                                "((Real.cosh ({al}) * ({da})) * Real.cosh ({al}) - Real.sinh ({al}) * (Real.sinh ({al}) * ({da}))) / Real.cosh ({al}) ^ 2"
                            ),
                            dpoly,
                        )
                    }
                    Fn1::Asinh => {
                        self.fun_atom(Fn1::Asinh, &al, ap.clone());
                        let rad = format!("1 + ({al}) ^ 2");
                        let hp = self.have(format!("0 < {rad}"), "by positivity".into());
                        let rad_poly = Poly::one().add(&ap.pow(2)?);
                        let r = self.sqrt_atom(&rad, rad_poly, Some(format!("({hp}).le")));
                        let rl = self.atoms[r as usize].lean.clone();
                        self.have(format!("{rl} ≠ 0"), format!("(Real.sqrt_pos.mpr {hp}).ne'"));
                        let w = self.inv_poly(&Poly::var(r), &rl)?;
                        node(
                            format!("(({pf}).arsinh)"),
                            format!("(Real.sqrt ({rad}))⁻¹ * ({da})"),
                            w.mul(&dp)?,
                        )
                    }
                    Fn1::Acosh => {
                        let h1 = self.hyp(format!("1 < {al}"), Cond::Gt1(a));
                        self.fun_atom(Fn1::Acosh, &al, ap.clone());
                        let rad = format!("({al}) ^ 2 - 1");
                        let hp = self.have(format!("0 < {rad}"), "by nlinarith".into());
                        let rad_poly = ap.pow(2)?.sub(&Poly::one());
                        let r = self.sqrt_atom(&rad, rad_poly, Some(format!("({hp}).le")));
                        let rl = self.atoms[r as usize].lean.clone();
                        self.have(format!("{rl} ≠ 0"), format!("(Real.sqrt_pos.mpr {hp}).ne'"));
                        let w = self.inv_poly(&Poly::var(r), &rl)?;
                        node(
                            format!("(HasDerivAt.comp (h := fun ({x} : ℝ) => {al}) {x} (Real.hasDerivAt_arcosh (show ({al}) ∈ Set.Ioi 1 from {h1})) {pf})"),
                            format!("(Real.sqrt ({rad}))⁻¹ * ({da})"),
                            w.mul(&dp)?,
                        )
                    }
                })
            }
            Shape::Other => Err(self.pool.with(e, |d| match d {
                ExprData::RootSum { .. } => WITHHELD_ROOTSUM,
                _ => WITHHELD_UNSUPPORTED_NODE,
            })),
        }
    }

    // ------------------------------------------------------------ relations

    fn push_rel(&mut self, lhs: String, rhs: String, proof: String, kind: RelKind) {
        let name = format!("r{}", self.rels.len() + 1);
        self.rels.push(Rel {
            name,
            lhs,
            rhs,
            proof,
            kind,
            always: false,
        });
    }

    /// Fun atoms of kind `f` (canonical ones: those without a `congrArg` twin rule).
    fn fun_atoms(&self, fs: &[Fn1]) -> Vec<(u32, Fn1, Poly, String)> {
        self.atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| match &a.kind {
                AKind::Fun { f, arg, arg_lean }
                    if fs.contains(f) && !self.twin_of.contains_key(&(i as u32)) =>
                {
                    Some((i as u32, *f, arg.clone(), arg_lean.clone()))
                }
                _ => None,
            })
            .collect()
    }

    fn canon(&self, mut id: u32) -> u32 {
        while let Some(&t) = self.twin_of.get(&id) {
            id = t;
        }
        id
    }

    fn arg_lean_of(&self, id: u32) -> String {
        match &self.atoms[id as usize].kind {
            AKind::Fun { arg_lean, .. } => arg_lean.clone(),
            _ => self.atoms[id as usize].lean.clone(),
        }
    }

    fn has_rule(&self, v: u32) -> bool {
        self.rels.iter().any(|r| match r.kind {
            RelKind::Subst { v: w, .. } | RelKind::Power { v: w, .. } => w == v,
            RelKind::Inverse { .. } => false,
        })
    }

    /// Group arguments into classes of rational multiples of a common base.
    /// Returns, per class, `(base_poly, base_lean, [(atom, f, k)])`.
    #[allow(clippy::type_complexity)]
    fn multiple_classes(
        &self,
        items: &[(u32, Fn1, Poly, String)],
    ) -> Result<Vec<(Poly, String, Vec<(u32, Fn1, i64, String)>)>, IntegrationCertWithheld> {
        // (representative arg, members with ratio to the representative)
        let mut classes: Vec<(Poly, String, Vec<(u32, Fn1, Rational, String)>)> = Vec::new();
        for (id, f, arg, arg_lean) in items {
            if arg.is_zero() {
                continue;
            }
            let mut placed = false;
            for (rep, _, members) in classes.iter_mut() {
                if let Some(q) = arg.ratio_to(rep) {
                    members.push((*id, *f, q, arg_lean.clone()));
                    placed = true;
                    break;
                }
            }
            if !placed {
                classes.push((
                    arg.clone(),
                    arg_lean.clone(),
                    vec![(*id, *f, Rational::from(1), arg_lean.clone())],
                ));
            }
        }
        let mut out = Vec::new();
        for (rep, rep_lean, members) in classes {
            // g = gcd of the ratios (positive).
            let mut num_g = Integer::new();
            let mut den_l = Integer::from(1);
            for (_, _, q, _) in &members {
                num_g = num_g.gcd(q.numer());
                den_l = den_l.lcm(q.denom());
            }
            let g = Rational::from((num_g, den_l));
            let mut ms = Vec::new();
            for (id, f, q, l) in members {
                let k = Rational::from(&q / &g);
                if *k.denom() != 1 {
                    return Err(WITHHELD_IDENTITY);
                }
                let k = k.numer().to_i64().ok_or(WITHHELD_TOO_LARGE)?;
                if k.unsigned_abs() > 24 {
                    return Err(WITHHELD_TOO_LARGE);
                }
                ms.push((id, f, k, l));
            }
            let base = rep.scale(&g);
            let base_lean = if let Some((_, _, _, l)) = ms.iter().find(|m| m.2 == 1) {
                l.clone()
            } else if g == 1 {
                rep_lean
            } else {
                format!("({rep_lean}) * {}", rat_lean(&g)?)
            };
            out.push((base, base_lean, ms));
        }
        Ok(out)
    }

    fn build_relations(&mut self) -> Result<(), IntegrationCertWithheld> {
        // --- trig families --------------------------------------------------
        let trig = self.fun_atoms(&[Fn1::Sin, Fn1::Cos, Fn1::Tan]);
        for (base, base_lean, members) in self.multiple_classes(&trig)? {
            let kmax = members
                .iter()
                .map(|m| m.2.unsigned_abs())
                .max()
                .unwrap_or(1) as i64;
            let mut spell_of: BTreeMap<i64, String> = BTreeMap::new();
            spell_of.insert(1, base_lean.clone());
            for (_, _, k, l) in &members {
                spell_of.entry(*k).or_insert_with(|| l.clone());
            }
            let spell = |k: i64, m: &mut BTreeMap<i64, String>| -> String {
                m.entry(k)
                    .or_insert_with(|| format!("({base_lean}) * ({k} : ℝ)"))
                    .clone()
            };
            let at = |k: i64| base.scale(&Rational::from(k));
            let l1 = spell(1, &mut spell_of);
            let s1 = self.fun_atom(Fn1::Sin, &l1, at(1));
            let s1 = self.canon(s1);
            let s1l = self.arg_lean_of(s1);
            let c1 = self.fun_atom(Fn1::Cos, &s1l, at(1));
            if !self.has_rule(s1) {
                let (s1_lean, c1_lean) = (
                    self.atoms[s1 as usize].lean.clone(),
                    self.atoms[c1 as usize].lean.clone(),
                );
                self.push_rel(
                    format!("{s1_lean} ^ 2 + {c1_lean} ^ 2"),
                    "1".into(),
                    format!("Real.sin_sq_add_cos_sq ({s1l})"),
                    RelKind::Power {
                        v: s1,
                        d: 2,
                        tail: Poly::one().sub(&Poly::var(c1).pow(2)?),
                    },
                );
            }
            for k in 2..=kmax {
                let lk = spell(k, &mut spell_of);
                let pl = spell(k - 1, &mut spell_of);
                let sp = self.fun_atom(Fn1::Sin, &pl, at(k - 1));
                let cp = self.fun_atom(Fn1::Cos, &pl, at(k - 1));
                let sb = self.fun_atom(Fn1::Sin, &s1l, at(1));
                let cb = self.fun_atom(Fn1::Cos, &s1l, at(1));
                let (spl, cpl, sbl, cbl) = (
                    self.atoms[sp as usize].lean.clone(),
                    self.atoms[cp as usize].lean.clone(),
                    self.atoms[sb as usize].lean.clone(),
                    self.atoms[cb as usize].lean.clone(),
                );
                for f in [Fn1::Sin, Fn1::Cos] {
                    let id = self.fun_atom(f, &lk, at(k));
                    let id = self.canon(id);
                    if self.has_rule(id) {
                        continue;
                    }
                    let kl = self.arg_lean_of(id);
                    let (lemma, rhs, e) = if f == Fn1::Sin {
                        (
                            "Real.sin_add",
                            format!("{spl} * {cbl} + {cpl} * {sbl}"),
                            Poly::var(sp)
                                .mul(&Poly::var(cb))?
                                .add(&Poly::var(cp).mul(&Poly::var(sb))?),
                        )
                    } else {
                        (
                            "Real.cos_add",
                            format!("{cpl} * {cbl} - {spl} * {sbl}"),
                            Poly::var(cp)
                                .mul(&Poly::var(cb))?
                                .sub(&Poly::var(sp).mul(&Poly::var(sb))?),
                        )
                    };
                    self.push_rel(
                        self.atoms[id as usize].lean.clone(),
                        rhs,
                        format!(
                            "by\n    have h := {lemma} ({pl}) ({s1l})\n    rw [show ({pl}) + ({s1l}) = ({kl}) by ring] at h\n    exact h"
                        ),
                        RelKind::Subst { v: id, e },
                    );
                }
            }
            for (id, f, k, l) in members.clone() {
                if k < 0 {
                    let ml = spell(-k, &mut spell_of);
                    for g in [Fn1::Sin, Fn1::Cos] {
                        let gid = self.fun_atom(g, &l, at(k));
                        let gid = self.canon(gid);
                        if self.has_rule(gid) {
                            continue;
                        }
                        let kl = self.arg_lean_of(gid);
                        let m_atom = self.fun_atom(g, &ml, at(-k));
                        let m_lean = self.atoms[m_atom as usize].lean.clone();
                        let (lemma, rhs, e) = if g == Fn1::Sin {
                            (
                                "Real.sin_neg",
                                format!("-{m_lean}"),
                                Poly::var(m_atom).neg(),
                            )
                        } else {
                            ("Real.cos_neg", m_lean.clone(), Poly::var(m_atom))
                        };
                        self.push_rel(
                            self.atoms[gid as usize].lean.clone(),
                            rhs,
                            format!(
                                "by\n    have h := {lemma} ({ml})\n    rw [show -({ml}) = ({kl}) by ring] at h\n    exact h"
                            ),
                            RelKind::Subst { v: gid, e },
                        );
                    }
                }
                if f == Fn1::Tan && !self.has_rule(id) {
                    let sk = self.fun_atom(Fn1::Sin, &l, at(k));
                    let ck = self.fun_atom(Fn1::Cos, &l, at(k));
                    let skl = self.atoms[sk as usize].lean.clone();
                    let ckl = self.atoms[ck as usize].lean.clone();
                    let w = self.inv_poly(&Poly::var(ck), &ckl)?;
                    self.push_rel(
                        self.atoms[id as usize].lean.clone(),
                        format!("{skl} / {ckl}"),
                        format!("Real.tan_eq_sin_div_cos ({l})"),
                        RelKind::Subst {
                            v: id,
                            e: Poly::var(sk).mul(&w)?,
                        },
                    );
                }
            }
        }

        // --- exp families ---------------------------------------------------
        let exps = self.fun_atoms(&[Fn1::Exp]);
        for (base, base_lean, members) in self.multiple_classes(&exps)? {
            let kmax = members
                .iter()
                .map(|m| m.2.unsigned_abs())
                .max()
                .unwrap_or(1) as i64;
            if kmax == 1 && members.iter().all(|m| m.2 == 1) {
                continue;
            }
            let mut spell_of: BTreeMap<i64, String> = BTreeMap::new();
            spell_of.insert(1, base_lean.clone());
            for (_, _, k, l) in &members {
                spell_of.entry(*k).or_insert_with(|| l.clone());
            }
            let spell = |k: i64, m: &mut BTreeMap<i64, String>| -> String {
                m.entry(k)
                    .or_insert_with(|| format!("({base_lean}) * ({k} : ℝ)"))
                    .clone()
            };
            let at = |k: i64| base.scale(&Rational::from(k));
            let l1 = spell(1, &mut spell_of);
            let e1 = self.fun_atom(Fn1::Exp, &l1, at(1));
            let e1 = self.canon(e1);
            let e1l = self.arg_lean_of(e1);
            let e1_lean = self.atoms[e1 as usize].lean.clone();
            for k in 2..=kmax {
                let lk = spell(k, &mut spell_of);
                let pl = spell(k - 1, &mut spell_of);
                let ep = self.fun_atom(Fn1::Exp, &pl, at(k - 1));
                let epl = self.atoms[ep as usize].lean.clone();
                let id = self.fun_atom(Fn1::Exp, &lk, at(k));
                let id = self.canon(id);
                if self.has_rule(id) {
                    continue;
                }
                let kl = self.arg_lean_of(id);
                self.push_rel(
                    self.atoms[id as usize].lean.clone(),
                    format!("{epl} * {e1_lean}"),
                    format!(
                        "by\n    have h := Real.exp_add ({pl}) ({e1l})\n    rw [show ({pl}) + ({e1l}) = ({kl}) by ring] at h\n    exact h"
                    ),
                    RelKind::Subst {
                        v: id,
                        e: Poly::var(ep).mul(&Poly::var(e1))?,
                    },
                );
            }
            for (_, _, k, l) in members.clone() {
                if k >= 0 {
                    continue;
                }
                let id = self.fun_atom(Fn1::Exp, &l, at(k));
                let id = self.canon(id);
                if self.has_rule(id) {
                    continue;
                }
                let kl = self.arg_lean_of(id);
                let ml = spell(-k, &mut spell_of);
                let em = self.fun_atom(Fn1::Exp, &ml, at(-k));
                let eml = self.atoms[em as usize].lean.clone();
                self.have(format!("{eml} ≠ 0"), "(Real.exp_pos _).ne'".into());
                let w = self.inv_poly(&Poly::var(em), &eml)?;
                self.push_rel(
                    self.atoms[id as usize].lean.clone(),
                    format!("({eml})⁻¹"),
                    format!(
                        "by\n    have h := Real.exp_neg ({ml})\n    rw [show -({ml}) = ({kl}) by ring] at h\n    exact h"
                    ),
                    RelKind::Subst { v: id, e: w },
                );
            }
        }

        // --- log ∘ exp ------------------------------------------------------
        for (id, _, arg, _) in self.fun_atoms(&[Fn1::Log]) {
            let vars = arg.vars();
            if vars.len() != 1 || arg != Poly::var(*vars.iter().next().unwrap()) {
                continue;
            }
            let inner = *vars.iter().next().unwrap();
            if let AKind::Fun {
                f: Fn1::Exp,
                arg: e_arg,
                arg_lean: e_lean,
            } = self.atoms[inner as usize].kind.clone()
            {
                let lhs = self.atoms[id as usize].lean.clone();
                if lhs == format!("Real.log (Real.exp ({e_lean}))") && !self.has_rule(id) {
                    self.push_rel(
                        lhs,
                        e_lean.clone(),
                        format!("Real.log_exp ({e_lean})"),
                        RelKind::Subst { v: id, e: e_arg },
                    );
                }
            }
        }

        // --- tanh = sinh / cosh ----------------------------------------------
        for (id, _, arg, al) in self.fun_atoms(&[Fn1::Tanh]) {
            if self.has_rule(id) {
                continue;
            }
            let s = self.fun_atom(Fn1::Sinh, &al, arg.clone());
            let c = self.fun_atom(Fn1::Cosh, &al, arg.clone());
            let sl = self.atoms[s as usize].lean.clone();
            let cl = self.atoms[c as usize].lean.clone();
            self.have(format!("{cl} ≠ 0"), format!("(Real.cosh_pos ({al})).ne'"));
            let w = self.inv_poly(&Poly::var(c), &cl)?;
            self.push_rel(
                self.atoms[id as usize].lean.clone(),
                format!("{sl} / {cl}"),
                format!("Real.tanh_eq_sinh_div_cosh ({al})"),
                RelKind::Subst {
                    v: id,
                    e: Poly::var(s).mul(&w)?,
                },
            );
        }

        // --- cosh² - sinh² = 1 ---------------------------------------------
        let hyps = self.fun_atoms(&[Fn1::Sinh, Fn1::Cosh]);
        let mut done: Vec<Poly> = Vec::new();
        for (_, _, arg, al) in &hyps {
            if done.contains(arg) {
                continue;
            }
            done.push(arg.clone());
            let sh = self.fun_atom(Fn1::Sinh, al, arg.clone());
            let ch = self.fun_atom(Fn1::Cosh, al, arg.clone());
            if self.has_rule(sh) || self.has_rule(ch) {
                continue;
            }
            let shl = self.atoms[sh as usize].lean.clone();
            let chl = self.atoms[ch as usize].lean.clone();
            self.push_rel(
                format!("{chl} ^ 2 - {shl} ^ 2"),
                "1".into(),
                format!("Real.cosh_sq_sub_sinh_sq ({al})"),
                RelKind::Power {
                    v: ch,
                    d: 2,
                    tail: Poly::one().add(&Poly::var(sh).pow(2)?),
                },
            );
        }

        // --- √v ² = v ----------------------------------------------------------
        let sqrts = self.fun_atoms(&[Fn1::Sqrt]);
        for (id, _, arg, al) in sqrts {
            let Some(nonneg) = self.sqrt_nonneg.get(&id).cloned() else {
                continue;
            };
            let lean = self.atoms[id as usize].lean.clone();
            self.push_rel(
                format!("{lean} ^ 2"),
                al,
                format!("Real.sq_sqrt {nonneg}"),
                RelKind::Power {
                    v: id,
                    d: 2,
                    tail: arg,
                },
            );
        }
        Ok(())
    }

    // ------------------------------------------------------------ reduction

    /// Cofactors `cⱼ` with `p = Σ cⱼ·(Lⱼ - Rⱼ)` as a field identity, or
    /// `Err(WITHHELD_IDENTITY)`.
    fn certify_zero(&self, p: Poly) -> Result<Vec<Poly>, IntegrationCertWithheld> {
        let mut cof = vec![Poly::zero(); self.rels.len()];
        let mut cur = p;
        let mut mult = Poly::one();
        let inv_rels: Vec<(usize, u32, Poly)> = self
            .rels
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match &r.kind {
                RelKind::Inverse { w, u } => Some((i, *w, u.clone())),
                _ => None,
            })
            .collect();
        let mut powers: Vec<(usize, u32)> = self
            .rels
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.kind {
                RelKind::Power { v, .. } => Some((i, self.atoms[v as usize].depth)),
                _ => None,
            })
            .collect();
        powers.sort_by_key(|a| std::cmp::Reverse(a.1));
        for _round in 0..MAX_ROUNDS {
            if cur.is_zero() {
                return Ok(cof);
            }
            // 1. substitutions
            let mut progressed = false;
            for (i, r) in self.rels.iter().enumerate() {
                if let RelKind::Subst { v, e } = &r.kind {
                    if cur.has_var(*v) {
                        let (pe, delta) = substitute(&cur, *v, e)?;
                        cof[i] = cof[i].add(&mult.mul(&delta)?);
                        cur = pe;
                        progressed = true;
                    }
                }
            }
            if progressed {
                continue;
            }
            // 1b. cancel `a·a⁻¹` inside monomials: `c·m''·(a·w)ᵗ = c·m'' +
            //     c·m''·(1 + a·w + … + (a·w)^(t-1))·(a·w - 1)`.
            for (ri, w, u) in &inv_rels {
                let Some(a) = single_var(u) else { continue };
                if !cur.has_var(*w) || !cur.has_var(a) {
                    continue;
                }
                let aw = Poly::var(a).mul(&Poly::var(*w))?;
                if !cur.0.keys().any(|m| exp_of(m, a) > 0 && exp_of(m, *w) > 0) {
                    continue;
                }
                let mut next = Poly::zero();
                let mut q = Poly::zero();
                for (m, c) in &cur.0 {
                    let t = exp_of(m, a).min(exp_of(m, *w));
                    if t == 0 {
                        next.add_term(m.clone(), c.clone());
                        continue;
                    }
                    let mut m2: Mono = Vec::new();
                    for &(v, e) in m {
                        let e2 = if v == a || v == *w { e - t } else { e };
                        if e2 > 0 {
                            m2.push((v, e2));
                        }
                    }
                    next.add_term(m2.clone(), c.clone());
                    let mut s_geo = Poly::zero();
                    let mut tp = Poly::one();
                    for _ in 0..t {
                        s_geo = s_geo.add(&tp);
                        tp = tp.mul(&aw)?;
                    }
                    let mut mono = Poly::zero();
                    mono.add_term(m2, c.clone());
                    q = q.add(&mono.mul(&s_geo)?);
                }
                cof[*ri] = cof[*ri].add(&mult.mul(&q)?);
                cur = next;
                progressed = true;
            }
            if progressed {
                continue;
            }
            // 2. clear the outermost inverse: with t = u·w,
            //    p = w^K·Σ cₖ u^(K-k) - (t - 1)·Σ cₖ wᵏ (1 + t + … + t^(K-k-1)).
            let vars = cur.vars();
            if let Some((ri, w, u)) = inv_rels
                .iter()
                .filter(|(_, w, _)| vars.contains(w))
                .max_by_key(|(_, w, _)| self.atoms[*w as usize].depth)
            {
                let cs = cur.split(*w);
                let kmax = (cs.len() - 1) as u32;
                let t = u.mul(&Poly::var(*w))?;
                let mut next = Poly::zero();
                let mut a = Poly::zero();
                for (k, c) in cs.iter().enumerate() {
                    if c.is_zero() {
                        continue;
                    }
                    let k = k as u32;
                    next = next.add(&c.mul(&u.pow(kmax - k)?)?);
                    let mut s_geo = Poly::zero();
                    let mut tp = Poly::one();
                    for _ in 0..(kmax - k) {
                        s_geo = s_geo.add(&tp);
                        tp = tp.mul(&t)?;
                    }
                    if !s_geo.is_zero() {
                        a = a.add(&c.mul(&Poly::var(*w).pow(k)?)?.mul(&s_geo)?);
                    }
                }
                cof[*ri] = cof[*ri].sub(&mult.mul(&a)?);
                cur = next;
                mult = mult.mul(&Poly::var(*w).pow(kmax)?)?;
                continue;
            }
            // 3. power relations
            let mut reduced = false;
            for &(i, _) in &powers {
                if let RelKind::Power { v, d, tail } = &self.rels[i].kind {
                    if cur.deg_in(*v) >= *d {
                        let (r, q) = reduce_power(&cur, *v, *d, tail)?;
                        cof[i] = cof[i].add(&mult.mul(&q)?);
                        cur = r;
                        reduced = true;
                    }
                }
            }
            if !reduced {
                return Err(WITHHELD_IDENTITY);
            }
        }
        Err(WITHHELD_TOO_LARGE)
    }

    // ------------------------------------------------------------- printing

    fn print_poly(&self, p: &Poly) -> Result<String, IntegrationCertWithheld> {
        if p.is_zero() {
            return Ok("0".into());
        }
        let mut terms = Vec::new();
        for (m, c) in &p.0 {
            let mut parts = vec![rat_lean(c)?];
            for &(v, e) in m {
                let a = &self.atoms[v as usize].lean;
                parts.push(if e == 1 {
                    a.to_string()
                } else {
                    format!("({a}) ^ ({e} : ℕ)")
                });
            }
            terms.push(parts.join(" * "));
        }
        Ok(terms.join(" + "))
    }
}

/// `Some(v)` when `p` is exactly the variable `v`.
pub(super) fn single_var(p: &Poly) -> Option<u32> {
    if p.len() != 1 {
        return None;
    }
    let (m, c) = p.0.iter().next()?;
    (m.len() == 1 && m[0].1 == 1 && *c == 1).then_some(m[0].0)
}

pub(super) fn u_as_constant(u: &Poly) -> Option<Rational> {
    if u.is_zero() {
        return Some(Rational::new());
    }
    if u.len() == 1 {
        if let Some(c) = u.0.get(&Vec::new()) {
            return Some(c.clone());
        }
    }
    None
}

/// Conservative syntactic positivity — the cases Mathlib's `positivity`
/// proves without hypotheses.
fn surely_positive(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => n.0 > 0,
        ExprData::Rational(r) => r.0 > 0,
        ExprData::Symbol { name, .. } => name == "pi",
        ExprData::Func { name, args } if args.len() == 1 => match name.as_str() {
            "exp" | "cosh" => true,
            "sqrt" => surely_positive(args[0], pool),
            _ => false,
        },
        ExprData::Add(xs) => {
            xs.iter().all(|&c| surely_nonneg(c, pool))
                && xs.iter().any(|&c| surely_positive(c, pool))
        }
        ExprData::Mul(xs) => xs.iter().all(|&c| surely_positive(c, pool)),
        ExprData::Pow { base, exp } => {
            let int_or_rat = pool.with(*exp, |x| {
                matches!(x, ExprData::Integer(_) | ExprData::Rational(_))
            });
            int_or_rat && surely_positive(*base, pool)
        }
        _ => false,
    })
}

fn surely_nonneg(e: ExprId, pool: &ExprPool) -> bool {
    if surely_positive(e, pool) {
        return true;
    }
    pool.with(e, |d| match d {
        ExprData::Func { name, args } if args.len() == 1 => name == "sqrt",
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&c| surely_nonneg(c, pool)),
        ExprData::Pow { base, exp } => pool.with(*exp, |x| match x {
            ExprData::Integer(n) => n.0 >= 0 && (n.0.is_even() || surely_nonneg(*base, pool)),
            _ => false,
        }),
        _ => false,
    })
}

// ===========================================================================
// Non-vacuity: numeric evaluation of the hypotheses
// ===========================================================================

fn eval_f64(e: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> f64 {
    if let Some(v) = env.get(&e) {
        return *v;
    }
    pool.with(e, |d| match d {
        ExprData::Integer(n) => n.0.to_f64(),
        ExprData::Rational(r) => r.0.to_f64(),
        ExprData::Symbol { name, .. } if name == "pi" => std::f64::consts::PI,
        ExprData::Add(xs) => xs.iter().map(|&c| eval_f64(c, env, pool)).sum(),
        ExprData::Mul(xs) => xs.iter().map(|&c| eval_f64(c, env, pool)).product(),
        ExprData::Pow { base, exp } => {
            let b = eval_f64(*base, env, pool);
            let x = eval_f64(*exp, env, pool);
            if x.fract() == 0.0 && x.abs() < 1e6 {
                b.powi(x as i32)
            } else {
                b.powf(x)
            }
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = eval_f64(args[0], env, pool);
            match name.as_str() {
                "sin" => a.sin(),
                "cos" => a.cos(),
                "tan" => a.tan(),
                "exp" => a.exp(),
                "log" => a.abs().ln(),
                "sqrt" => a.sqrt(),
                "atan" => a.atan(),
                "asin" => a.asin(),
                "acos" => a.acos(),
                "sinh" => a.sinh(),
                "cosh" => a.cosh(),
                _ => f64::NAN,
            }
        }
        _ => f64::NAN,
    })
}

fn cond_holds(c: Cond, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> bool {
    let ok = |v: f64, f: &dyn Fn(f64) -> bool| v.is_finite() && f(v);
    match c {
        Cond::Ne0(e) => ok(eval_f64(e, env, pool), &|v| v.abs() > 1e-9),
        Cond::Pos(e) => ok(eval_f64(e, env, pool), &|v| v > 1e-12),
        Cond::GtM1(e) => ok(eval_f64(e, env, pool), &|v| v > -1.0 + 1e-12),
        Cond::Lt1(e) => ok(eval_f64(e, env, pool), &|v| v < 1.0 - 1e-12),
        Cond::Gt1(e) => ok(eval_f64(e, env, pool), &|v| v > 1.0 + 1e-12),
        Cond::CosNe0(e) => ok(eval_f64(e, env, pool), &|v| v.cos().abs() > 1e-9),
    }
}

fn collect_symbols(e: ExprId, pool: &ExprPool, out: &mut BTreeSet<ExprId>) {
    pool.with(e, |d| match d {
        ExprData::Symbol { name, .. } if name != "pi" => {
            out.insert(e);
        }
        ExprData::Add(xs) | ExprData::Mul(xs) | ExprData::Func { args: xs, .. } => {
            for &c in xs {
                collect_symbols(c, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_symbols(*base, pool, out);
            collect_symbols(*exp, pool, out);
        }
        _ => {}
    })
}

// ===========================================================================
// Entry point
// ===========================================================================

/// Respell every rational power `b ^ (a/q)` as `√b ^ a` (`q = 2`) or
/// `(b ^ (1/q)) ^ a` (`q ≥ 3`), negative `a` through `⁻¹`.
///
/// Equal to the original wherever `b > 0`, which the certificate assumes (or
/// proves) for every such base. Afterwards a base carries at most one real
/// exponent, `1/q`: `ring` compares atoms up to definitional unfolding, and
/// two real exponents of one base (`b ^ (7/2)` against `b ^ (1/2)`) send it
/// into unfolding real-number literals until the heartbeat limit.
pub(super) fn respell_rational_powers(
    e: ExprId,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, IntegrationCertWithheld> {
    if let Some(&r) = memo.get(&e) {
        return Ok(r);
    }
    enum Shape {
        Leaf,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
    }
    let shape = pool.with(e, |d| match d {
        ExprData::Add(xs) => Shape::Add(xs.clone()),
        ExprData::Mul(xs) => Shape::Mul(xs.clone()),
        ExprData::Pow { base, exp } => Shape::Pow(*base, *exp),
        ExprData::Func { name, args } => Shape::Func(name.clone(), args.clone()),
        _ => Shape::Leaf,
    });
    let out = match shape {
        Shape::Leaf => e,
        Shape::Add(xs) => {
            let ys: Result<Vec<_>, _> = xs
                .iter()
                .map(|&c| respell_rational_powers(c, pool, memo))
                .collect();
            pool.add(ys?)
        }
        Shape::Mul(xs) => {
            let ys: Result<Vec<_>, _> = xs
                .iter()
                .map(|&c| respell_rational_powers(c, pool, memo))
                .collect();
            pool.mul(ys?)
        }
        Shape::Func(name, xs) => {
            let ys: Result<Vec<_>, _> = xs
                .iter()
                .map(|&c| respell_rational_powers(c, pool, memo))
                .collect();
            pool.func(name, ys?)
        }
        Shape::Pow(b, x) => {
            let b2 = respell_rational_powers(b, pool, memo)?;
            let r = pool.with(x, |d| match d {
                ExprData::Rational(r) => Some(r.0.clone()),
                _ => None,
            });
            match r {
                None => pool.pow(b2, x),
                Some(r) => {
                    let a = r.numer().to_i64().ok_or(WITHHELD_TOO_LARGE)?;
                    let q = r.denom().to_i64().ok_or(WITHHELD_TOO_LARGE)?;
                    let root = if q == 2 {
                        pool.func("sqrt", vec![b2])
                    } else {
                        pool.pow(b2, pool.rational(1, q))
                    };
                    if a == 1 {
                        root
                    } else {
                        pool.pow(root, pool.integer(a))
                    }
                }
            }
        }
    };
    memo.insert(e, out);
    Ok(out)
}

/// `log(abs(a))` → `log(a)` throughout (other `abs` nodes are kept).
pub(super) fn strip_log_abs(e: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
    if let Some(&r) = memo.get(&e) {
        return r;
    }
    enum Shape {
        Leaf,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
    }
    let shape = pool.with(e, |d| match d {
        ExprData::Add(xs) => Shape::Add(xs.clone()),
        ExprData::Mul(xs) => Shape::Mul(xs.clone()),
        ExprData::Pow { base, exp } => Shape::Pow(*base, *exp),
        ExprData::Func { name, args } => Shape::Func(name.clone(), args.clone()),
        _ => Shape::Leaf,
    });
    let out = match shape {
        Shape::Leaf => e,
        Shape::Add(xs) => {
            let ys = xs.iter().map(|&c| strip_log_abs(c, pool, memo)).collect();
            pool.add(ys)
        }
        Shape::Mul(xs) => {
            let ys = xs.iter().map(|&c| strip_log_abs(c, pool, memo)).collect();
            pool.mul(ys)
        }
        Shape::Pow(b, x) => {
            let b2 = strip_log_abs(b, pool, memo);
            pool.pow(b2, x)
        }
        Shape::Func(name, xs) => {
            let inner_abs = (name == "log" && xs.len() == 1)
                .then(|| {
                    pool.with(xs[0], |d| match d {
                        ExprData::Func { name, args } if name == "abs" && args.len() == 1 => {
                            Some(args[0])
                        }
                        _ => None,
                    })
                })
                .flatten();
            match inner_abs {
                Some(a) => {
                    let a2 = strip_log_abs(a, pool, memo);
                    pool.func("log", vec![a2])
                }
                None => {
                    let ys = xs.iter().map(|&c| strip_log_abs(c, pool, memo)).collect();
                    pool.func(name, ys)
                }
            }
        }
    };
    memo.insert(e, out);
    out
}

/// Definitions for special functions Mathlib does not define, each by its
/// standard defining integral, with the derivative lemma the certificate uses
/// (`(name, Lean source)`; a certificate mentions `name (`).
/// The definitions are the textbook ones — `erf x = 2/√π ∫₀ˣ e^(-t²) dt`,
/// `erfi x = 2/√π ∫₀ˣ e^(t²) dt`, `Si x = ∫₀ˣ sin t / t dt` (with Mathlib's
/// `Real.sinc` as the continuous integrand) — and the lemmas are proved from
/// Mathlib's FTC, so a certificate using them assumes nothing new.
const SPECIAL_DEFS: [(&str, &str); 3] = [
    (
        "alkahestErf",
        "/-- `erf x = 2/√π · ∫₀ˣ exp (-t²) dt` (its defining integral; Mathlib has no `erf`). -/\n\
         noncomputable def alkahestErf (x : ℝ) : ℝ :=\n  \
         2 / Real.sqrt Real.pi * ∫ t in (0 : ℝ)..x, Real.exp (-t ^ 2)\n\n\
         theorem alkahestErf_hasDerivAt (x : ℝ) :\n    \
         HasDerivAt alkahestErf (2 / Real.sqrt Real.pi * Real.exp (-x ^ 2)) x :=\n  \
         ((Continuous.integral_hasStrictDerivAt (by fun_prop) 0 x).hasDerivAt).const_mul _\n\n",
    ),
    (
        "alkahestErfi",
        "/-- `erfi x = 2/√π · ∫₀ˣ exp (t²) dt` (its defining integral). -/\n\
         noncomputable def alkahestErfi (x : ℝ) : ℝ :=\n  \
         2 / Real.sqrt Real.pi * ∫ t in (0 : ℝ)..x, Real.exp (t ^ 2)\n\n\
         theorem alkahestErfi_hasDerivAt (x : ℝ) :\n    \
         HasDerivAt alkahestErfi (2 / Real.sqrt Real.pi * Real.exp (x ^ 2)) x :=\n  \
         ((Continuous.integral_hasStrictDerivAt (by fun_prop) 0 x).hasDerivAt).const_mul _\n\n",
    ),
    (
        "alkahestSi",
        "/-- The sine integral `Si x = ∫₀ˣ sin t / t dt`, with `Real.sinc` as the integrand. -/\n\
         noncomputable def alkahestSi (x : ℝ) : ℝ :=\n  \
         ∫ t in (0 : ℝ)..x, Real.sinc t\n\n\
         theorem alkahestSi_hasDerivAt (x : ℝ) : HasDerivAt alkahestSi (Real.sinc x) x :=\n  \
         (Real.continuous_sinc.integral_hasStrictDerivAt 0 x).hasDerivAt\n\n",
    ),
];

/// Lean imports for `HasDerivAt` integration certificates, plus the FTC and
/// `sinc` modules when a definition from [`SPECIAL_DEFS`] is used.
fn hasderiv_header_with(special: bool) -> String {
    let h = hasderiv_header();
    if !special {
        return h;
    }
    h.replacen(
        "\n\nset_option",
        "\nimport Mathlib.Analysis.SpecialFunctions.Trigonometric.Sinc\n\
         import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus\n\nset_option",
        1,
    )
}

/// Lean imports for `HasDerivAt` integration certificates.
pub(crate) fn hasderiv_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.Calculus.Deriv.Basic\n\
     import Mathlib.Analysis.Calculus.Deriv.Add\n\
     import Mathlib.Analysis.Calculus.Deriv.Pow\n\
     import Mathlib.Analysis.Calculus.Deriv.Mul\n\
     import Mathlib.Analysis.Calculus.Deriv.Inv\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.InverseDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.Log.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.Sqrt\n\
     import Mathlib.Analysis.SpecialFunctions.Pow.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.Arsinh\n\
     import Mathlib.Analysis.SpecialFunctions.Arcosh\n\
     \n\
     set_option linter.unusedSimpArgs false\n\
     set_option linter.unusedTactic false\n\
     set_option linter.unreachableTactic false\n\
     \n"
    .to_string()
}

/// Emit a `HasDerivAt` certificate for `∫ integrand d(var) = antiderivative`.
///
/// See the module documentation. Returns the `.lean` source, or the reason
/// the certificate is withheld. The result never contains `sorry`/`admit`.
pub fn emit_integration_hasderiv_cert(
    antiderivative: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<String, IntegrationCertWithheld> {
    let mut memo = HashMap::new();
    let antiderivative = respell_rational_powers(antiderivative, pool, &mut memo)?;
    let integrand = respell_rational_powers(integrand, pool, &mut memo)?;
    // `Real.log |a| = Real.log a` holds for every real `a` (`Real.log_abs`),
    // so `log |a|` is certified as `log a`: same derivative `a'/a`, same
    // domain `a ≠ 0`. The statement keeps the `abs`; one `simp only` step
    // removes it from the goal.
    let stated_big = antiderivative;
    let stated_small = integrand;
    let mut abs_memo = HashMap::new();
    let antiderivative = strip_log_abs(antiderivative, pool, &mut abs_memo);
    let integrand = strip_log_abs(integrand, pool, &mut abs_memo);
    let has_log_abs = antiderivative != stated_big || integrand != stated_small;
    let mut cx = Ctx::new(pool, var)?;
    let stated_big_lean = cx.lean(stated_big)?;
    let stated_small_lean = cx.lean(stated_small)?;
    let big_f = cx.lean(antiderivative)?;
    let node = cx.deriv(antiderivative)?;
    let fpoly = cx.vpoly(integrand)?;
    cx.build_relations()?;
    let resid = node.dpoly.sub(&fpoly);
    let cof = cx.certify_zero(resid)?;

    // Non-vacuity: some sample point satisfies every hypothesis.
    let mut syms = BTreeSet::new();
    collect_symbols(antiderivative, pool, &mut syms);
    collect_symbols(integrand, pool, &mut syms);
    let conds: Vec<Cond> = cx.facts.iter().filter_map(|f| f.cond).collect();
    if !conds.is_empty() {
        const XS: [f64; 16] = [
            0.37, 0.61, 0.83, 1.37, 2.3, -0.42, 0.13, 3.7, -1.7, 5.3, 0.05, 0.95, -3.1, 9.1, -0.77,
            1.9,
        ];
        let params: Vec<ExprId> = syms.iter().copied().filter(|&s| s != var).collect();
        let found = XS.iter().enumerate().any(|(i, &xv)| {
            let mut env = HashMap::new();
            env.insert(var, xv);
            for (j, &p) in params.iter().enumerate() {
                env.insert(p, 0.7 + 0.37 * ((i + j) % 5) as f64);
            }
            conds.iter().all(|&c| cond_holds(c, &env, pool))
        });
        if !found {
            return Err(WITHHELD_EMPTY_DOMAIN);
        }
    }

    // ---- assemble ---------------------------------------------------------
    let x = cx.var_name.clone();
    let mut binders = format!("({x} : ℝ)");
    for p in &cx.params {
        binders.push_str(&format!(" ({p} : ℝ)"));
    }
    for f in cx.facts.iter().filter(|f| f.proof.is_none()) {
        binders.push_str(&format!(" ({} : {})", f.name, f.prop));
    }
    let mut body = String::new();
    if has_log_abs {
        body.push_str("  simp only [Real.log_abs]\n");
    }
    for f in cx.facts.iter().filter(|f| f.proof.is_some()) {
        body.push_str(&format!(
            "  have {} : {} := {}\n",
            f.name,
            f.prop,
            f.proof.as_ref().unwrap()
        ));
    }
    body.push_str(&format!("  have hd0 := {}\n", node.proof));
    body.push_str(&format!(
        "  have hd : HasDerivAt (fun ({x} : ℝ) => {big_f}) ({}) {x} := hd0\n",
        node.dlean
    ));
    let mut lc_terms = Vec::new();
    for (i, r) in cx.rels.iter().enumerate() {
        if cof[i].is_zero() && !r.always {
            continue;
        }
        body.push_str(&format!(
            "  have {} : {} = {} := {}\n",
            r.name, r.lhs, r.rhs, r.proof
        ));
        if !cof[i].is_zero() {
            let c = cx.print_poly(&cof[i])?;
            lc_terms.push(format!("({c}) * {}", r.name));
        }
    }
    // `G = f` is a polynomial identity in the atoms modulo the facts above:
    // `ring1` checks `G - f - Σ cⱼ·(Lⱼ - Rⱼ) = 0` with the cofactors computed here.
    for (name, lhs, rhs, proof) in &cx.goal_rw {
        body.push_str(&format!("  have {name} : {lhs} = {rhs} := {proof}\n"));
    }
    let combo = if lc_terms.is_empty() {
        String::new()
    } else {
        format!(" {}", lc_terms.join(" + "))
    };
    // `ring1`'s cost grows with the combination; past this size it can run
    // for many minutes, so withhold rather than hand out a slow certificate.
    if combo.len() > MAX_COMBINATION_BYTES {
        return Err(WITHHELD_TOO_LARGE);
    }
    body.push_str("  refine hd.congr_deriv ?_\n");
    if !cx.goal_rw.is_empty() {
        let names: Vec<&str> = cx.goal_rw.iter().map(|g| g.0.as_str()).collect();
        body.push_str(&format!("  rw [{}]\n", names.join(", ")));
    }
    body.push_str(&format!("  linear_combination{combo}\n"));

    let display_f = pool.display(integrand).to_string();
    let display_big = pool.display(antiderivative).to_string();
    let mut defs = String::new();
    for (name, def) in SPECIAL_DEFS {
        let needle = format!("{name} (");
        if stated_big_lean.contains(&needle)
            || stated_small_lean.contains(&needle)
            || body.contains(&needle)
        {
            defs.push_str(def);
        }
    }
    let src = format!(
        "{}{defs}-- ∫ {display_f} d{x} = {display_big}\n\
         -- Certified via the derivative relation on the domain given by the hypotheses:\n\
         -- HasDerivAt (fun {x} => F) f {x}.\n\
         set_option linter.unusedVariables false in\n\
         set_option maxHeartbeats 4000000 in\n\
         theorem alkahest_integral {binders} :\n    \
         HasDerivAt (fun ({x} : ℝ) => {stated_big_lean}) ({stated_small_lean}) {x} := by\n{body}",
        hasderiv_header_with(!defs.is_empty())
    );
    if src.len() > MAX_SRC_BYTES {
        return Err(WITHHELD_TOO_LARGE);
    }
    if super::contains_admission(&src) {
        return Err(WITHHELD_UNSUPPORTED_NODE);
    }
    Ok(src)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn emit(f: impl Fn(&ExprPool, ExprId) -> (ExprId, ExprId)) -> Result<String, &'static str> {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let (big_f, small_f) = f(&pool, x);
        emit_integration_hasderiv_cert(big_f, small_f, x, &pool)
    }

    #[test]
    fn polynomial_times_trig_certifies() {
        // ∫ x² sin x = -x² cos x + 2x sin x + 2 cos x
        let src = emit(|p, x| {
            let two = p.integer(2);
            let m1 = p.integer(-1);
            let x2 = p.pow(x, two);
            let s = p.func("sin", vec![x]);
            let c = p.func("cos", vec![x]);
            let big = p.add(vec![
                p.mul(vec![x2, m1, c]),
                p.mul(vec![x, two, s]),
                p.mul(vec![two, c]),
            ]);
            (big, p.mul(vec![x2, s]))
        })
        .expect("certificate");
        assert!(src.contains("HasDerivAt (fun (x : ℝ) =>"), "{src}");
        assert!(src.contains("linear_combination"), "{src}");
        assert!(!src.contains("sorry") && !src.contains("admit"), "{src}");
    }

    #[test]
    fn log_answer_states_its_domain() {
        // ∫ 1/x = log x on x ≠ 0
        let src = emit(|p, x| (p.func("log", vec![x]), p.pow(x, p.integer(-1)))).expect("cert");
        assert!(src.contains("(h1 : (x : ℝ) ≠ 0)"), "{src}");
        assert!(src.contains(".log h1"), "{src}");
    }

    #[test]
    fn rational_power_answer_is_respelled() {
        // ∫ x·√(x²+1) = (x²+1)^(3/2) / 3
        let src = emit(|p, x| {
            let v = p.add(vec![p.pow(x, p.integer(2)), p.integer(1)]);
            let big = p.mul(vec![p.rational(1, 3), p.pow(v, p.rational(3, 2))]);
            (big, p.mul(vec![x, p.func("sqrt", vec![v])]))
        })
        .expect("cert");
        assert!(src.contains("Real.sqrt"), "{src}");
        // `x² + 1 > 0` is discharged by `positivity`, not assumed.
        assert!(src.contains("by positivity"), "{src}");
        assert!(!src.contains("(h1 :"), "{src}");
    }

    #[test]
    fn double_angle_uses_cos_add() {
        // ∫ sin x cos x = -cos(2x)/4
        let src = emit(|p, x| {
            let c2 = p.func("cos", vec![p.mul(vec![x, p.integer(2)])]);
            (
                p.mul(vec![c2, p.rational(-1, 4)]),
                p.mul(vec![p.func("sin", vec![x]), p.func("cos", vec![x])]),
            )
        })
        .expect("cert");
        assert!(src.contains("Real.sin_add"), "{src}");
    }

    #[test]
    fn log_of_algebraic_argument_certifies() {
        // ∫ log(x + √(x²-1)) = x·log(x + √(x²-1)) - √(x²-1)
        let r = emit(|p, x| {
            let r = p.func(
                "sqrt",
                vec![p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)])],
            );
            let l = p.func("log", vec![p.add(vec![x, r])]);
            (
                p.add(vec![p.mul(vec![x, l]), p.mul(vec![p.integer(-1), r])]),
                l,
            )
        });
        assert!(r.is_ok(), "{r:?}");
    }

    #[test]
    fn log_abs_and_log_of_negative_certify() {
        // ∫ 1/x = log|x| on x ≠ 0
        let src = emit(|p, x| {
            (
                p.func("log", vec![p.func("abs", vec![x])]),
                p.pow(x, p.integer(-1)),
            )
        })
        .expect("cert");
        assert!(src.contains("Real.log (abs ((x : ℝ)))"), "{src}");
        assert!(src.contains("simp only [Real.log_abs]"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
        // ∫ 1/(x - 1) = log(1 - x) on 1 - x ≠ 0
        let src = emit(|p, x| {
            let one_minus = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), x])]);
            let x_minus = p.add(vec![x, p.integer(-1)]);
            (
                p.func("log", vec![one_minus]),
                p.pow(x_minus, p.integer(-1)),
            )
        })
        .expect("cert");
        assert!(src.contains("≠ 0"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
    }

    #[test]
    fn wrong_antiderivative_is_withheld() {
        // d/dx sin x ≠ sin x: the residual does not reduce.
        let r = emit(|p, x| (p.func("sin", vec![x]), p.func("sin", vec![x])));
        assert_eq!(r, Err(WITHHELD_IDENTITY));
    }

    #[test]
    fn special_functions_have_reasons() {
        let r = emit(|p, x| (p.func("Ei", vec![x]), p.func("exp", vec![x])));
        assert_eq!(r, Err(WITHHELD_SPECIAL_FUNCTION));
        let r = emit(|p, x| (p.func("atan2", vec![x, x]), x));
        assert_eq!(r, Err(WITHHELD_UNSUPPORTED_FUNCTION));
    }

    #[test]
    fn erf_and_si_certify_through_their_defining_integrals() {
        // ∫ exp(-x²) = √π/2 · erf x
        let src = emit(|p, x| {
            let sqrt_pi = p.func("sqrt", vec![p.symbol("pi", Domain::Real)]);
            let big = p.mul(vec![p.func("erf", vec![x]), sqrt_pi, p.rational(1, 2)]);
            let small = p.func(
                "exp",
                vec![p.mul(vec![p.pow(x, p.integer(2)), p.integer(-1)])],
            );
            (big, small)
        })
        .expect("erf cert");
        assert!(src.contains("noncomputable def alkahestErf"), "{src}");
        assert!(src.contains("alkahestErf_hasDerivAt"), "{src}");
        assert!(!src.contains("def alkahestSi"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
        // ∫ sin x / x = Si x on x ≠ 0
        let src = emit(|p, x| {
            (
                p.func("Si", vec![x]),
                p.mul(vec![p.func("sin", vec![x]), p.pow(x, p.integer(-1))]),
            )
        })
        .expect("Si cert");
        assert!(src.contains("Real.sinc_of_ne_zero"), "{src}");
        assert!(src.contains("(h1 : (x : ℝ) ≠ 0)"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
        // A wrong special-function answer still does not reduce.
        let r = emit(|p, x| (p.func("erf", vec![x]), p.func("exp", vec![x])));
        assert_eq!(r, Err(WITHHELD_IDENTITY));
    }

    #[test]
    fn hyperbolic_answers_certify() {
        // ∫ tanh x = log(cosh x): tanh is related to sinh/cosh, no hypothesis.
        let src = emit(|p, x| {
            (
                p.func("log", vec![p.func("cosh", vec![x])]),
                p.func("tanh", vec![x]),
            )
        })
        .expect("tanh cert");
        assert!(src.contains("Real.tanh_eq_sinh_div_cosh"), "{src}");
        assert!(!src.contains("(h1 :"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
        // ∫ sech² x = tanh x: the derivative goes through sinh / cosh.
        let src = emit(|p, x| {
            let c = p.func("cosh", vec![x]);
            (p.func("tanh", vec![x]), p.pow(c, p.integer(-2)))
        })
        .expect("d tanh cert");
        assert!(src.contains("congr_of_eventuallyEq"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
        // ∫ 1/√(1+x²) = asinh x
        let src = emit(|p, x| {
            let r = p.func(
                "sqrt",
                vec![p.add(vec![p.integer(1), p.pow(x, p.integer(2))])],
            );
            (p.func("asinh", vec![x]), p.pow(r, p.integer(-1)))
        })
        .expect("asinh cert");
        assert!(src.contains(".arsinh"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
        // ∫ 1/√(x²-1) = acosh x on x > 1
        let src = emit(|p, x| {
            let r = p.func(
                "sqrt",
                vec![p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)])],
            );
            (p.func("acosh", vec![x]), p.pow(r, p.integer(-1)))
        })
        .expect("acosh cert");
        assert!(src.contains("Real.hasDerivAt_arcosh"), "{src}");
        assert!(src.contains("(h1 : 1 < (x : ℝ))"), "{src}");
        println!("=====CERT=====\n{src}=====END=====");
    }

    #[test]
    fn empty_domain_is_withheld() {
        // asin(x + 3): no sample point has -1 < x + 3 < 1 together with x > 0.
        let r = emit(|p, x| {
            let a = p.func("asin", vec![p.add(vec![x, p.integer(3)])]);
            let s = p.func("sqrt", vec![x]);
            let big = p.add(vec![a, s]);
            (big, big)
        });
        assert!(
            r == Err(WITHHELD_EMPTY_DOMAIN) || r == Err(WITHHELD_IDENTITY),
            "{r:?}"
        );
    }

    #[test]
    fn negative_constant_radicand_is_withheld() {
        let r = emit(|p, x| {
            let s = p.func("sqrt", vec![p.integer(-3)]);
            (p.mul(vec![s, x]), s)
        });
        assert_eq!(r, Err(WITHHELD_CONSTANT_SIGN));
    }
}
