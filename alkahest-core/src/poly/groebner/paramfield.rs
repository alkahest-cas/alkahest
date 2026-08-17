//! Exact arithmetic in the coefficient field `Q(p₁, …, p_m)`.
//!
//! A parametric Gröbner basis wants the parameters in the *coefficient field*,
//! not as extra ring variables: eliminating states from an ODE model in
//! `Q(params)[states, Y]` is a much smaller computation than the same
//! elimination in `Q[states, Y, params]`, because the parameters never enter
//! the monomial order and never generate S-pairs of their own.
//!
//! Two types live here:
//!
//! * [`ParamPoly`] — a sparse multivariate polynomial in the parameters over
//!   ℤ.  This is the *denominator ring*; it is also the type in which the
//!   degeneracy conditions of a parametric basis are reported.
//! * [`QParam`] — an element of `Q(p₁, …, p_m)`, stored as a reduced pair of
//!   [`ParamPoly`]s.
//!
//! # Why this is not a naive fraction type
//!
//! `holonomic::qfield` exists because a rational-function coefficient field
//! implemented with textbook Euclidean gcd swells catastrophically: every
//! division leaves a fresh quotient whose numerator and denominator degrees
//! add, and nothing removes the content that would let them cancel again.  The
//! same lesson applies here, one variable further out, so the same two defences
//! are used:
//!
//! 1. **Every value is kept in canonical reduced form** — `gcd(num, den) = 1`
//!    in `ℤ[p]` (content included) and `lc(den) > 0`.  Canonical form makes
//!    structural equality decide field equality, and it is the only thing that
//!    stops denominators from compounding.
//! 2. **Cancellation happens before multiplication, not after.**  Addition goes
//!    through the *lcm* of the two denominators rather than their product, and
//!    multiplication cancels crosswise first.  Both are exactly the moves
//!    `qfield::rn_add` / `rn_mul` make, and both are proved below to leave the
//!    result already reduced, so the common case costs no extra gcd at all.
//!
//! The gcd itself is FLINT's multivariate gcd (`fmpz_mpoly_gcd`, a Hensel /
//! Zippel hybrid), which is the part `qfield` had to hand-roll a subresultant
//! PRS for because its coefficients were univariate `Q[n]` and `RatUniPoly::gcd`
//! was the naive Euclidean sequence.

use crate::flint::mpoly::{FlintMPoly, FlintMPolyCtx, FlintMPolyFactor};
use rug::{Integer, Rational};
use std::collections::BTreeMap;
use std::sync::Arc;

/// An exponent vector over the parameters; always exactly `n_params` long.
pub type ParamExp = Vec<u32>;

// ---------------------------------------------------------------------------
// ParamPoly — sparse ℤ[p₁, …, p_m]
// ---------------------------------------------------------------------------

/// A sparse multivariate polynomial in the parameters, over ℤ.
///
/// Exponent keys always have length `n_params` (no trailing-zero stripping), so
/// the `BTreeMap` order is plain lexicographic and the last entry is the
/// lex-leading term.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParamPoly {
    /// Integer coefficients keyed by exponent vector.
    pub terms: BTreeMap<ParamExp, Integer>,
    /// Number of parameters.
    pub n_params: usize,
}

impl ParamPoly {
    /// The zero polynomial.
    pub fn zero(n_params: usize) -> Self {
        ParamPoly {
            terms: BTreeMap::new(),
            n_params,
        }
    }

    /// The constant `c`.
    pub fn constant(c: Integer, n_params: usize) -> Self {
        let mut terms = BTreeMap::new();
        if c != 0 {
            terms.insert(vec![0u32; n_params], c);
        }
        ParamPoly { terms, n_params }
    }

    /// The constant `1`.
    pub fn one(n_params: usize) -> Self {
        ParamPoly::constant(Integer::from(1), n_params)
    }

    /// The parameter `p_i`.
    pub fn var(i: usize, n_params: usize) -> Self {
        let mut exp = vec![0u32; n_params];
        if i < n_params {
            exp[i] = 1;
        }
        let mut terms = BTreeMap::new();
        terms.insert(exp, Integer::from(1));
        ParamPoly { terms, n_params }
    }

    /// True for the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// True for the constant `1`.
    pub fn is_one(&self) -> bool {
        matches!(self.as_constant(), Some(c) if *c == 1)
    }

    /// The constant value, or `None` if any parameter actually occurs.
    pub fn as_constant(&self) -> Option<&Integer> {
        match self.terms.len() {
            0 => None,
            1 => {
                let (exp, c) = self.terms.iter().next()?;
                exp.iter().all(|&e| e == 0).then_some(c)
            }
            _ => None,
        }
    }

    /// True when this is a non-zero constant — the case that carries no
    /// information as a degeneracy condition.
    pub fn is_nonzero_constant(&self) -> bool {
        self.as_constant().is_some()
    }

    /// Number of non-zero terms.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Maximum total degree over all terms (`0` for the zero polynomial).
    pub fn total_degree(&self) -> u32 {
        self.terms
            .keys()
            .map(|e| e.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// The lex-leading coefficient, or `0`.
    fn leading_coeff(&self) -> Integer {
        self.terms
            .iter()
            .next_back()
            .map(|(_, c)| c.clone())
            .unwrap_or_else(|| Integer::from(0))
    }

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Self {
        let mut terms = self.terms.clone();
        for (e, c) in &other.terms {
            let slot = terms.entry(e.clone()).or_insert_with(|| Integer::from(0));
            *slot += c;
            if *slot == 0 {
                terms.remove(e);
            }
        }
        ParamPoly {
            terms,
            n_params: self.n_params,
        }
    }

    /// `-self`.
    pub fn neg(&self) -> Self {
        ParamPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), -c.clone()))
                .collect(),
            n_params: self.n_params,
        }
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// `self · other`.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return ParamPoly::zero(self.n_params);
        }
        if let Some(c) = self.as_constant() {
            return other.scale_int(c);
        }
        if let Some(c) = other.as_constant() {
            return self.scale_int(c);
        }
        let mut terms: BTreeMap<ParamExp, Integer> = BTreeMap::new();
        for (ea, ca) in &self.terms {
            for (eb, cb) in &other.terms {
                let e: ParamExp = ea.iter().zip(eb.iter()).map(|(a, b)| a + b).collect();
                let slot = terms.entry(e).or_insert_with(|| Integer::from(0));
                *slot += Integer::from(ca * cb);
            }
        }
        terms.retain(|_, c| *c != 0);
        ParamPoly {
            terms,
            n_params: self.n_params,
        }
    }

    /// `self · z` for an integer `z`.
    pub fn scale_int(&self, z: &Integer) -> Self {
        if *z == 0 {
            return ParamPoly::zero(self.n_params);
        }
        ParamPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), Integer::from(c * z)))
                .collect(),
            n_params: self.n_params,
        }
    }

    /// Non-negative gcd of the integer coefficients (`0` for zero).
    pub fn content(&self) -> Integer {
        let mut g = Integer::from(0);
        for c in self.terms.values() {
            g = Integer::from(g.gcd_ref(c));
            if g == 1 {
                break;
            }
        }
        g
    }

    /// Primitive part with a positive lex-leading coefficient.
    ///
    /// This is the canonical representative of the hypersurface `{self = 0}`,
    /// which is what makes it the right normalisation for a reported
    /// degeneracy condition.
    pub fn primitive_part(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        let mut cont = self.content();
        if self.leading_coeff() < 0 {
            cont = -cont;
        }
        ParamPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), Integer::from(c / &cont)))
                .collect(),
            n_params: self.n_params,
        }
    }

    /// `gcd(self, other)`, normalised to a positive leading coefficient.
    ///
    /// Integer content is included, so `gcd(2x, 4) = 2`.  Falls back to `1`
    /// when FLINT declines (which would only cost efficiency, never
    /// correctness, since an unreduced fraction is still the same field
    /// element).
    pub fn gcd(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.primitive_part();
        }
        if other.is_zero() {
            return self.primitive_part();
        }
        // Constant-only fast path: FLINT is not needed to gcd two integers.
        if let (Some(a), Some(b)) = (self.as_constant(), other.as_constant()) {
            return ParamPoly::constant(Integer::from(a.gcd_ref(b)).abs(), self.n_params);
        }
        if self.n_params == 0 {
            let g = self.content().gcd(&other.content());
            return ParamPoly::constant(g.abs(), 0);
        }
        let ctx = FlintMPolyCtx::new(self.n_params);
        let a = self.to_flint(Arc::clone(&ctx));
        let b = other.to_flint(Arc::clone(&ctx));
        match a.gcd(&b) {
            Some(g) => {
                let g = ParamPoly::from_flint(&g, self.n_params);
                if g.is_zero() {
                    ParamPoly::one(self.n_params)
                } else if g.leading_coeff() < 0 {
                    g.neg()
                } else {
                    g
                }
            }
            None => ParamPoly::one(self.n_params),
        }
    }

    /// `Some(self / divisor)` when the division is exact, else `None`.
    pub fn exact_div(&self, divisor: &Self) -> Option<Self> {
        if divisor.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(ParamPoly::zero(self.n_params));
        }
        if let Some(d) = divisor.as_constant() {
            if *d == 1 {
                return Some(self.clone());
            }
            let mut terms = BTreeMap::new();
            for (e, c) in &self.terms {
                if !c.is_divisible(d) {
                    return None;
                }
                terms.insert(e.clone(), Integer::from(c.div_exact_ref(d)));
            }
            return Some(ParamPoly {
                terms,
                n_params: self.n_params,
            });
        }
        if self.n_params == 0 {
            return None;
        }
        let ctx = FlintMPolyCtx::new(self.n_params);
        let a = self.to_flint(Arc::clone(&ctx));
        let b = divisor.to_flint(Arc::clone(&ctx));
        a.divides(&b)
            .map(|q| ParamPoly::from_flint(&q, self.n_params))
    }

    /// Value at a rational point.
    ///
    /// `values` must have one entry per parameter; missing entries are read as
    /// zero, which keeps the function total.
    pub fn eval(&self, values: &[Rational]) -> Rational {
        let mut acc = Rational::from(0);
        for (exp, c) in &self.terms {
            let mut term = Rational::from(c.clone());
            for (i, &e) in exp.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                let v = values.get(i).cloned().unwrap_or_else(|| Rational::from(0));
                if v == 0 {
                    term = Rational::from(0);
                    break;
                }
                for _ in 0..e {
                    term *= v.clone();
                }
            }
            acc += term;
        }
        acc
    }

    /// The distinct non-constant irreducible factors, each primitive with a
    /// positive leading coefficient.
    ///
    /// The hypersurface `{self = 0}` is the union of the `{f = 0}` over these,
    /// so reporting the factors instead of the product is what turns "the
    /// basis is wrong somewhere on this degree-12 surface" into a list of
    /// conditions a caller can actually read.  Multiplicities are dropped —
    /// they do not change the zero set.
    ///
    /// Falls back to `[primitive_part]` when FLINT declines to factor.
    pub fn irreducible_factors(&self) -> Vec<Self> {
        if self.is_zero() || self.is_nonzero_constant() {
            return vec![];
        }
        if self.n_params == 0 {
            return vec![];
        }
        let ctx = FlintMPolyCtx::new(self.n_params);
        let f = self.to_flint(Arc::clone(&ctx));
        let mut fac = FlintMPolyFactor::new(Arc::clone(&ctx));
        if !fac.factor(&f) {
            return vec![self.primitive_part()];
        }
        let mut out = Vec::with_capacity(fac.len());
        for i in 0..fac.len() {
            let base = ParamPoly::from_flint(&fac.base_at(i), self.n_params);
            if base.is_zero() || base.is_nonzero_constant() {
                continue;
            }
            out.push(base.primitive_part());
        }
        if out.is_empty() {
            vec![self.primitive_part()]
        } else {
            out
        }
    }

    fn to_flint(&self, ctx: Arc<FlintMPolyCtx>) -> FlintMPoly {
        let nvars = ctx.nvars();
        let mut fp = FlintMPoly::new(ctx);
        for (exp, c) in &self.terms {
            let mut e = vec![0u64; nvars];
            for (i, &v) in exp.iter().enumerate() {
                if i < nvars {
                    e[i] = v as u64;
                }
            }
            fp.push_term(c, &e);
        }
        fp.finish();
        fp
    }

    fn from_flint(f: &FlintMPoly, n_params: usize) -> Self {
        // `FlintMPoly::terms` strips trailing zeros; re-pad so every key has
        // the same length and the map order stays lexicographic.
        let terms = f
            .terms()
            .into_iter()
            .map(|(mut e, c)| {
                e.resize(n_params, 0);
                (e, c)
            })
            .collect();
        ParamPoly { terms, n_params }
    }
}

// ---------------------------------------------------------------------------
// QParam — an element of Q(p₁, …, p_m)
// ---------------------------------------------------------------------------

/// An element of `Q(p₁, …, p_m)`, stored as a reduced fraction of
/// [`ParamPoly`]s.
///
/// Invariants, maintained by every constructor and operation:
///
/// * `den` is non-zero and `gcd(num, den) = 1` in `ℤ[p]` (integer content
///   included);
/// * the lex-leading coefficient of `den` is positive;
/// * zero is exactly `0 / 1`.
///
/// Together these make the representation canonical, so `==` decides equality
/// in the field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QParam {
    num: ParamPoly,
    den: ParamPoly,
}

impl QParam {
    /// Zero.
    pub fn zero(n_params: usize) -> Self {
        QParam {
            num: ParamPoly::zero(n_params),
            den: ParamPoly::one(n_params),
        }
    }

    /// One.
    pub fn one(n_params: usize) -> Self {
        QParam {
            num: ParamPoly::one(n_params),
            den: ParamPoly::one(n_params),
        }
    }

    /// A rational constant.
    pub fn from_rational(r: &Rational, n_params: usize) -> Self {
        let (n, d) = r.clone().into_numer_denom();
        QParam {
            num: ParamPoly::constant(n, n_params),
            den: ParamPoly::constant(d, n_params),
        }
    }

    /// A polynomial in the parameters, as an element of the field.
    pub fn from_poly(p: ParamPoly) -> Self {
        let n_params = p.n_params;
        if p.is_zero() {
            return QParam::zero(n_params);
        }
        QParam {
            num: p,
            den: ParamPoly::one(n_params),
        }
    }

    /// `num / den`, reduced.  `None` when `den` is zero.
    pub fn from_ratio(num: ParamPoly, den: ParamPoly) -> Option<Self> {
        if den.is_zero() {
            return None;
        }
        Some(Self::reduced(num, den))
    }

    /// The numerator in canonical form.
    pub fn numerator(&self) -> &ParamPoly {
        &self.num
    }

    /// The denominator in canonical form.
    pub fn denominator(&self) -> &ParamPoly {
        &self.den
    }

    /// Number of parameters in the ambient field.
    pub fn n_params(&self) -> usize {
        self.num.n_params
    }

    /// True for zero.
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    /// True for one.
    pub fn is_one(&self) -> bool {
        self.num.is_one() && self.den.is_one()
    }

    /// True when this element is a rational number (no parameter occurs).
    pub fn is_rational(&self) -> bool {
        (self.num.is_zero() || self.num.is_nonzero_constant()) && self.den.is_nonzero_constant()
    }

    /// The rational value, when [`Self::is_rational`].
    pub fn as_rational(&self) -> Option<Rational> {
        let d = self.den.as_constant()?;
        if self.num.is_zero() {
            return Some(Rational::from(0));
        }
        let n = self.num.as_constant()?;
        Some(Rational::from((n.clone(), d.clone())))
    }

    /// Build the canonical form of `num / den` (`den ≠ 0` assumed).
    fn reduced(num: ParamPoly, den: ParamPoly) -> Self {
        let n_params = den.n_params;
        if num.is_zero() {
            return QParam::zero(n_params);
        }
        let g = num.gcd(&den);
        let (num, den) = if g.is_one() {
            (num, den)
        } else {
            match (num.exact_div(&g), den.exact_div(&g)) {
                (Some(n), Some(d)) => (n, d),
                _ => (num, den),
            }
        };
        Self::fix_sign(num, den)
    }

    /// Force a positive leading denominator coefficient.
    fn fix_sign(num: ParamPoly, den: ParamPoly) -> Self {
        if den.leading_coeff() < 0 {
            QParam {
                num: num.neg(),
                den: den.neg(),
            }
        } else {
            QParam { num, den }
        }
    }

    /// `self + other`.
    ///
    /// Goes through `lcm(den₁, den₂)` rather than the product.  Two facts make
    /// the result reduced without a further gcd in the common case:
    ///
    /// * with `gcd(d₁, d₂) = 1`, `gcd(n₁d₂ + n₂d₁, d₁) = gcd(n₁d₂, d₁) = 1` and
    ///   symmetrically for `d₂`, and in a UFD coprimality to both factors is
    ///   coprimality to the product — so the cross-multiplied form is already
    ///   in lowest terms;
    /// * otherwise every common factor of the sum and the lcm divides
    ///   `gcd(d₁, d₂)`, which is the smaller polynomial to cancel against.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        if self.den.is_one() && other.den.is_one() {
            return QParam::from_poly(self.num.add(&other.num));
        }
        if self.den == other.den {
            let num = self.num.add(&other.num);
            return Self::reduced(num, self.den.clone());
        }
        let g = self.den.gcd(&other.den);
        if g.is_one() {
            let num = self.num.mul(&other.den).add(&other.num.mul(&self.den));
            if num.is_zero() {
                return QParam::zero(self.n_params());
            }
            return Self::fix_sign(num, self.den.mul(&other.den));
        }
        let (Some(a1), Some(b1)) = (self.den.exact_div(&g), other.den.exact_div(&g)) else {
            // gcd claimed a factor it cannot divide out — fall back to the
            // cross-multiplied form and reduce it in full.
            let num = self.num.mul(&other.den).add(&other.num.mul(&self.den));
            return Self::reduced(num, self.den.mul(&other.den));
        };
        let num = self.num.mul(&b1).add(&other.num.mul(&a1));
        if num.is_zero() {
            return QParam::zero(self.n_params());
        }
        let den = self.den.mul(&b1);
        // Only factors of `g` can survive, so cancel against `g`, not `den`.
        let h = num.gcd(&g);
        if !h.is_one() {
            if let (Some(n), Some(d)) = (num.exact_div(&h), den.exact_div(&h)) {
                return Self::reduced(n, d);
            }
        }
        Self::fix_sign(num, den)
    }

    /// `-self`.
    pub fn neg(&self) -> Self {
        QParam {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// `self · other`.
    ///
    /// Cancels crosswise first: with both operands reduced,
    /// `gcd(n₁n₂, d₁d₂) = gcd(n₁, d₂)·gcd(n₂, d₁)`, so removing those two gcds
    /// before multiplying leaves the product already in lowest terms.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return QParam::zero(self.n_params());
        }
        if self.den.is_one() && other.den.is_one() {
            return QParam::from_poly(self.num.mul(&other.num));
        }
        let (n1, d2) = cross_cancel(&self.num, &other.den);
        let (n2, d1) = cross_cancel(&other.num, &self.den);
        Self::fix_sign(n1.mul(&n2), d1.mul(&d2))
    }

    /// `1 / self`, or `None` for zero.
    pub fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(Self::fix_sign(self.den.clone(), self.num.clone()))
    }

    /// `self / other`, or `None` when `other` is zero.
    pub fn div(&self, other: &Self) -> Option<Self> {
        Some(self.mul(&other.inv()?))
    }

    /// Value at a rational parameter point; `None` when the denominator
    /// vanishes there.
    pub fn eval(&self, values: &[Rational]) -> Option<Rational> {
        let d = self.den.eval(values);
        if d == 0 {
            return None;
        }
        Some(self.num.eval(values) / d)
    }
}

/// Divide `gcd(x, y)` out of both, leaving them untouched when it is a unit.
fn cross_cancel(x: &ParamPoly, y: &ParamPoly) -> (ParamPoly, ParamPoly) {
    if x.is_one() || y.is_one() {
        return (x.clone(), y.clone());
    }
    let g = x.gcd(y);
    if g.is_one() {
        return (x.clone(), y.clone());
    }
    match (x.exact_div(&g), y.exact_div(&g)) {
        (Some(a), Some(b)) => (a, b),
        _ => (x.clone(), y.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(n: usize) -> usize {
        n
    }

    fn a(n_params: usize) -> ParamPoly {
        ParamPoly::var(0, n_params)
    }

    fn b(n_params: usize) -> ParamPoly {
        ParamPoly::var(1, n_params)
    }

    #[test]
    fn poly_arithmetic() {
        let n = p(2);
        let f = a(n).add(&ParamPoly::one(n)); // a + 1
        let g = a(n).sub(&ParamPoly::one(n)); // a - 1
        let prod = f.mul(&g); // a² - 1
        assert_eq!(prod.terms.len(), 2);
        assert_eq!(prod.total_degree(), 2);
        assert_eq!(prod.eval(&[Rational::from(3), Rational::from(0)]), 8);
    }

    #[test]
    fn gcd_and_exact_div() {
        let n = p(2);
        let f = a(n).mul(&a(n)).sub(&ParamPoly::one(n)); // a² - 1
        let g = a(n).sub(&ParamPoly::one(n)); // a - 1
        let d = f.gcd(&g);
        assert_eq!(d, g, "gcd(a²-1, a-1) should be a-1");
        let q = f.exact_div(&g).expect("exact division");
        assert_eq!(q, a(n).add(&ParamPoly::one(n)));
    }

    #[test]
    fn integer_content_is_part_of_the_gcd() {
        let n = p(1);
        let two_a = a(n).scale_int(&Integer::from(2));
        let four = ParamPoly::constant(Integer::from(4), n);
        assert_eq!(two_a.gcd(&four), ParamPoly::constant(Integer::from(2), n));
    }

    #[test]
    fn field_is_canonical() {
        let n = p(2);
        // (a² - 1)/(a - 1) must reduce to (a + 1)/1.
        let f = QParam::from_ratio(
            a(n).mul(&a(n)).sub(&ParamPoly::one(n)),
            a(n).sub(&ParamPoly::one(n)),
        )
        .unwrap();
        assert_eq!(f, QParam::from_poly(a(n).add(&ParamPoly::one(n))));
        assert!(f.denominator().is_one());
    }

    #[test]
    fn sign_is_normalised() {
        let n = p(1);
        let neg_den = ParamPoly::one(n).sub(&a(n)); // 1 - a, lex lc is -1
        let f = QParam::from_ratio(ParamPoly::one(n), neg_den).unwrap();
        assert!(
            f.denominator().leading_coeff() > 0,
            "denominator must be sign-normalised"
        );
        // 1/(1-a) + 1/(a-1) = 0 requires the two to be recognised as negatives.
        let g = QParam::from_ratio(ParamPoly::one(n), a(n).sub(&ParamPoly::one(n))).unwrap();
        assert!(f.add(&g).is_zero());
    }

    #[test]
    fn addition_uses_the_lcm() {
        let n = p(2);
        // 1/(ab) + 1/(a) = (1 + b)/(ab): denominator degree stays 2, not 3.
        let ab = a(n).mul(&b(n));
        let x = QParam::from_ratio(ParamPoly::one(n), ab.clone()).unwrap();
        let y = QParam::from_ratio(ParamPoly::one(n), a(n)).unwrap();
        let s = x.add(&y);
        assert_eq!(s.denominator().total_degree(), 2);
        assert_eq!(*s.denominator(), ab);
    }

    #[test]
    fn field_axioms_on_a_sample() {
        let n = p(2);
        let x = QParam::from_ratio(a(n).add(&b(n)), a(n).sub(&b(n))).unwrap();
        let y = QParam::from_ratio(b(n), a(n).mul(&a(n)).add(&ParamPoly::one(n))).unwrap();
        assert!(x.sub(&x).is_zero());
        assert!(x.mul(&x.inv().unwrap()).is_one());
        assert_eq!(x.add(&y), y.add(&x));
        assert_eq!(x.mul(&y), y.mul(&x));
        assert_eq!(x.mul(&y.add(&x)), x.mul(&y).add(&x.mul(&x)));
    }

    #[test]
    fn eval_reports_poles() {
        let n = p(1);
        let f = QParam::from_ratio(ParamPoly::one(n), a(n).sub(&ParamPoly::one(n))).unwrap();
        assert_eq!(f.eval(&[Rational::from(3)]), Some(Rational::from((1, 2))));
        assert_eq!(
            f.eval(&[Rational::from(1)]),
            None,
            "1/(a-1) has a pole at 1"
        );
    }

    #[test]
    fn irreducible_factors_split_the_locus() {
        let n = p(2);
        // a(a - b)²  →  {a, a - b}, multiplicity dropped.
        let f = a(n)
            .mul(&a(n).sub(&b(n)))
            .mul(&a(n).sub(&b(n)))
            .scale_int(&Integer::from(-6));
        let mut facs = f.irreducible_factors();
        facs.sort();
        assert_eq!(facs.len(), 2, "got {facs:?}");
        assert!(facs.contains(&a(n)));
        assert!(facs
            .iter()
            .any(|g| *g == a(n).sub(&b(n)) || *g == b(n).sub(&a(n))));
    }

    #[test]
    fn constants_are_not_conditions() {
        let n = p(2);
        assert!(ParamPoly::constant(Integer::from(7), n)
            .irreducible_factors()
            .is_empty());
    }
}
