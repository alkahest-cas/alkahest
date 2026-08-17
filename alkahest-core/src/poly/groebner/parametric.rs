//! Gröbner bases over the coefficient field `Q(p₁, …, p_m)` (M9).
//!
//! The engine is the same Buchberger loop as [`super::buchberger`] — same
//! Gebauer–Möller pair management from `super::pairs` (private — internal to
//! this module), same sugar selection,
//! same interreduction — with `rug::Rational` replaced by
//! [`QParam`], an element of `Q(params)`.  What changes is the *ring*: the
//! parameters are no longer variables, so they do not appear in the monomial
//! order, do not generate S-pairs, and do not enlarge the staircase.  For a
//! differential-elimination problem that is the difference between eliminating
//! states from `Q[states, Y, params]` and from `Q(params)[states, Y]`.
//!
//! # The specialisation hypothesis, and why it is reported
//!
//! A basis over `Q(params)` is a basis for *generic* parameter values.  A
//! leading coefficient that is a non-zero element of `Q(params)` can still
//! vanish at particular parameter values, and there the computation the
//! algorithm performed is not the computation it would have performed over ℚ.
//! That locus is information the caller needs, so it is computed and returned
//! rather than assumed away — see [`ParamGroebnerBasis::conditions`].
//!
//! Exactly two things can go wrong under a specialisation `σ: params ↦ p̄`:
//!
//! 1. a coefficient's denominator vanishes at `p̄`, so the coefficient has no
//!    value there at all; or
//! 2. a leading coefficient that the algorithm inverted vanishes at `p̄`, so
//!    over ℚ the leading monomial would have been a different one and the whole
//!    pair schedule downstream would have differed.
//!
//! Both are recorded as they happen: every inversion contributes its numerator
//! *and* its denominator to the condition set, and the input coefficients
//! contribute their denominators.  Nothing else can introduce a denominator —
//! addition and multiplication in `Q(params)` stay inside the local ring of
//! functions regular at `p̄`, and division only ever happens by a recorded
//! element — so the recorded set is closed:
//!
//! > **Specialisation.** Let `G` be the returned basis and `C` the returned
//! > conditions.  For every `p̄` with `f(p̄) ≠ 0` for all `f ∈ C`, every
//! > coefficient of `G` is regular at `p̄`, and `σ(G)` is precisely the basis
//! > this same algorithm computes over ℚ from `σ(F)` — every leading monomial,
//! > every pair, every reduction agrees step for step.
//!
//! The conditions are **sufficient, not necessary**: the true bad locus can be
//! smaller, because a specialisation can be harmless in a way this bookkeeping
//! cannot see. Erring in that direction is deliberate — the alternative is a
//! basis that is silently wrong on a set of measure zero.

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use rug::Rational;

use crate::errors::AlkahestError;
use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::pairs::{lcm_exp, update_pairs, CriticalPair};
use crate::poly::groebner::paramfield::{ParamPoly, QParam};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failures specific to the `Q(params)` Gröbner engine.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamGroebnerError {
    /// `compute` was called with no generators.
    NoGenerators,
    /// Generators disagree on the number of variables or parameters.
    ShapeMismatch {
        /// Shape of the first generator, as `(n_vars, n_params)`.
        expected: (usize, usize),
        /// Shape of the offending generator.
        got: (usize, usize),
    },
    /// A specialisation was requested with the wrong number of values.
    WrongArity {
        /// Number of parameters the basis is written over.
        expected: usize,
        /// Number of values supplied.
        got: usize,
    },
    /// The requested parameter point lies on the degeneracy locus: at least one
    /// of the basis's conditions vanishes there, so the basis says nothing
    /// about that point.
    Degenerate {
        /// The conditions that vanish at the requested point.
        vanishing: Vec<ParamPoly>,
    },
}

impl std::fmt::Display for ParamGroebnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamGroebnerError::NoGenerators => {
                write!(f, "a parametric Gröbner basis needs at least one generator")
            }
            ParamGroebnerError::ShapeMismatch { expected, got } => write!(
                f,
                "generator shape mismatch: expected {} variables and {} parameters, got {} and {}",
                expected.0, expected.1, got.0, got.1
            ),
            ParamGroebnerError::WrongArity { expected, got } => write!(
                f,
                "specialisation needs one value per parameter: expected {expected}, got {got}"
            ),
            ParamGroebnerError::Degenerate { vanishing } => write!(
                f,
                "parameter point is on the degeneracy locus: {} of the basis's conditions vanish \
                 there, so this basis does not describe that point",
                vanishing.len()
            ),
        }
    }
}

impl std::error::Error for ParamGroebnerError {}

impl AlkahestError for ParamGroebnerError {
    fn code(&self) -> &'static str {
        match self {
            ParamGroebnerError::NoGenerators => "E-PARAMGB-001",
            ParamGroebnerError::ShapeMismatch { .. } => "E-PARAMGB-002",
            ParamGroebnerError::WrongArity { .. } => "E-PARAMGB-003",
            ParamGroebnerError::Degenerate { .. } => "E-PARAMGB-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            ParamGroebnerError::NoGenerators => Some("pass at least one polynomial"),
            ParamGroebnerError::ShapeMismatch { .. } => {
                Some("build every generator against the same variable and parameter lists")
            }
            ParamGroebnerError::WrongArity { .. } => {
                Some("supply exactly one value per parameter, in the parameter list's order")
            }
            ParamGroebnerError::Degenerate { .. } => Some(
                "compute the basis directly over ℚ at that parameter point, or move the vanishing \
                 factors into the generators and recompute",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// ParamGbPoly — a polynomial in the variables with Q(params) coefficients
// ---------------------------------------------------------------------------

/// A sparse polynomial in the ring variables with coefficients in
/// `Q(p₁, …, p_m)`.
///
/// The exponent keys index the *variables* only — the parameters live in the
/// coefficients, which is the whole point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamGbPoly {
    /// Coefficients keyed by exponent vector over the variables.
    pub terms: BTreeMap<Vec<u32>, QParam>,
    /// Number of ring variables.
    pub n_vars: usize,
    /// Number of parameters in the coefficient field.
    pub n_params: usize,
}

impl ParamGbPoly {
    /// The zero polynomial.
    pub fn zero(n_vars: usize, n_params: usize) -> Self {
        ParamGbPoly {
            terms: BTreeMap::new(),
            n_vars,
            n_params,
        }
    }

    /// Reinterpret a `GbPoly` over `vars ++ params` as a polynomial in `vars`
    /// with coefficients in `Q(params)`.
    ///
    /// This is the bridge from the existing `Expr → GbPoly` conversion: build
    /// the polynomial over the concatenated variable list, then move the
    /// trailing `n_params` exponent slots into the coefficient field.
    ///
    /// Returns `None` when `p.n_vars ≠ n_vars + n_params`.
    pub fn from_gbpoly(p: &GbPoly, n_vars: usize, n_params: usize) -> Option<Self> {
        if p.n_vars != n_vars + n_params {
            return None;
        }
        let mut out = ParamGbPoly::zero(n_vars, n_params);
        for (exp, coeff) in &p.terms {
            if *coeff == 0 {
                continue;
            }
            let (var_exp, par_exp) = exp.split_at(n_vars);
            let mut mono = ParamPoly::zero(n_params);
            mono.terms.insert(par_exp.to_vec(), rug::Integer::from(1));
            let c = QParam::from_rational(coeff, n_params).mul(&QParam::from_poly(mono));
            let slot = out
                .terms
                .entry(var_exp.to_vec())
                .or_insert_with(|| QParam::zero(n_params));
            *slot = slot.add(&c);
        }
        out.terms.retain(|_, c| !c.is_zero());
        Some(out)
    }

    /// True for the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of non-zero terms.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Leading term under `order`.
    pub fn leading_term(&self, order: MonomialOrder) -> Option<(&Vec<u32>, &QParam)> {
        self.terms
            .iter()
            .max_by(|(ea, _), (eb, _)| order.cmp(ea, eb))
    }

    /// Leading exponent under `order`.
    pub fn leading_exp(&self, order: MonomialOrder) -> Option<Vec<u32>> {
        self.leading_term(order).map(|(e, _)| e.clone())
    }

    /// Leading coefficient under `order`.
    pub fn leading_coeff(&self, order: MonomialOrder) -> Option<QParam> {
        self.leading_term(order).map(|(_, c)| c.clone())
    }

    /// Max total degree over all terms (the "sugar" of the polynomial).
    pub fn sugar(&self) -> u32 {
        self.terms
            .keys()
            .map(|e| e.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Self {
        let mut terms = self.terms.clone();
        for (e, c) in &other.terms {
            let slot = terms
                .entry(e.clone())
                .or_insert_with(|| QParam::zero(self.n_params));
            *slot = slot.add(c);
            if slot.is_zero() {
                terms.remove(e);
            }
        }
        ParamGbPoly {
            terms,
            n_vars: self.n_vars,
            n_params: self.n_params,
        }
    }

    /// `-self`.
    pub fn neg(&self) -> Self {
        ParamGbPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), c.neg()))
                .collect(),
            n_vars: self.n_vars,
            n_params: self.n_params,
        }
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// `self · c` for a field element `c`.
    pub fn scale(&self, c: &QParam) -> Self {
        if c.is_zero() {
            return ParamGbPoly::zero(self.n_vars, self.n_params);
        }
        ParamGbPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, v)| (e.clone(), v.mul(c)))
                .collect(),
            n_vars: self.n_vars,
            n_params: self.n_params,
        }
    }

    /// `self · c · x^shift`.
    pub fn mul_monomial(&self, shift: &[u32], c: &QParam) -> Self {
        if c.is_zero() {
            return ParamGbPoly::zero(self.n_vars, self.n_params);
        }
        ParamGbPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, v)| {
                    let ne: Vec<u32> = e.iter().zip(shift.iter()).map(|(a, b)| a + b).collect();
                    (ne, v.mul(c))
                })
                .collect(),
            n_vars: self.n_vars,
            n_params: self.n_params,
        }
    }

    /// Divide through by the leading coefficient, recording the inversion.
    fn make_monic(&self, order: MonomialOrder, conds: &mut ConditionLog) -> Self {
        let Some(lc) = self.leading_coeff(order) else {
            return self.clone();
        };
        if lc.is_one() {
            return self.clone();
        }
        conds.record_inversion(&lc);
        match lc.inv() {
            Some(inv) => self.scale(&inv),
            None => self.clone(),
        }
    }

    /// Specialise every coefficient at a rational parameter point.
    ///
    /// `None` if any coefficient has a pole there.
    pub fn specialize(&self, values: &[Rational]) -> Option<GbPoly> {
        let mut terms = BTreeMap::new();
        for (e, c) in &self.terms {
            let v = c.eval(values)?;
            if v != 0 {
                terms.insert(e.clone(), v);
            }
        }
        Some(GbPoly {
            terms,
            n_vars: self.n_vars,
        })
    }
}

// ---------------------------------------------------------------------------
// Condition log
// ---------------------------------------------------------------------------

/// Accumulates the polynomials in the parameters whose non-vanishing the
/// computation assumed.
#[derive(Debug, Default)]
struct ConditionLog {
    raw: Vec<ParamPoly>,
}

impl ConditionLog {
    /// Record that `c` was inverted: both its numerator (which must stay
    /// non-zero for the leading monomial to survive) and its denominator
    /// (which must stay non-zero for `c` to have a value at all).
    fn record_inversion(&mut self, c: &QParam) {
        self.push(c.numerator());
        self.push(c.denominator());
    }

    /// Record that `c` must be regular at the parameter point.
    fn record_regular(&mut self, c: &QParam) {
        self.push(c.denominator());
    }

    fn push(&mut self, p: &ParamPoly) {
        if p.is_zero() || p.is_nonzero_constant() {
            return; // carries no condition
        }
        self.raw.push(p.clone());
    }

    /// Split every recorded polynomial into irreducible factors and dedup, so
    /// the reported locus is a list of distinct hypersurfaces rather than a
    /// pile of products.
    fn finish(self) -> Vec<ParamPoly> {
        let mut set: BTreeSet<ParamPoly> = BTreeSet::new();
        for p in &self.raw {
            for f in p.irreducible_factors() {
                set.insert(f);
            }
        }
        set.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Reduction and S-polynomials over Q(params)
// ---------------------------------------------------------------------------

/// Multivariate division of `f` by `gs`, returning the remainder.
fn reduce_param(
    f: &ParamGbPoly,
    gs: &[ParamGbPoly],
    order: MonomialOrder,
    conds: &mut ConditionLog,
) -> ParamGbPoly {
    let mut p = f.clone();
    let mut r = ParamGbPoly::zero(f.n_vars, f.n_params);
    let mut last_divisor: usize = 0;
    let is_graded = order.is_graded();

    'outer: while !p.is_zero() {
        let (lt_exp, lt_coeff) = match p.leading_term(order) {
            Some((e, c)) => (e.clone(), c.clone()),
            None => break,
        };
        let lt_deg: u32 = if is_graded { lt_exp.iter().sum() } else { 0 };

        for offset in 0..gs.len() {
            let idx = (last_divisor + offset) % gs.len();
            let g = &gs[idx];
            if let Some((lg_exp, lg_coeff)) = g.leading_term(order) {
                if is_graded && lg_exp.iter().sum::<u32>() > lt_deg {
                    continue;
                }
                if lt_exp.len() == lg_exp.len()
                    && lt_exp.iter().zip(lg_exp.iter()).all(|(a, b)| a >= b)
                {
                    let shift: Vec<u32> = lt_exp
                        .iter()
                        .zip(lg_exp.iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    if !lg_coeff.is_one() {
                        conds.record_inversion(lg_coeff);
                    }
                    let Some(coeff) = lt_coeff.div(lg_coeff) else {
                        continue;
                    };
                    p = p.sub(&g.mul_monomial(&shift, &coeff));
                    last_divisor = idx;
                    continue 'outer;
                }
            }
        }

        // No divisor found — move the leading term to the remainder.
        let mut lt = ParamGbPoly::zero(f.n_vars, f.n_params);
        lt.terms.insert(lt_exp.clone(), lt_coeff);
        r = r.add(&lt);
        p.terms.remove(&lt_exp);
    }

    r
}

/// The S-polynomial of `f` and `g` under `order`.
fn s_polynomial_param(
    f: &ParamGbPoly,
    g: &ParamGbPoly,
    order: MonomialOrder,
    conds: &mut ConditionLog,
) -> ParamGbPoly {
    let (Some((lf_exp, lf_coeff)), Some((lg_exp, lg_coeff))) =
        (f.leading_term(order), g.leading_term(order))
    else {
        return ParamGbPoly::zero(f.n_vars, f.n_params);
    };
    let lcm = lcm_exp(lf_exp, lg_exp);
    let shift_f: Vec<u32> = lcm.iter().zip(lf_exp.iter()).map(|(l, a)| l - a).collect();
    let shift_g: Vec<u32> = lcm.iter().zip(lg_exp.iter()).map(|(l, b)| l - b).collect();

    let one = QParam::one(f.n_params);
    if !lf_coeff.is_one() {
        conds.record_inversion(lf_coeff);
    }
    if !lg_coeff.is_one() {
        conds.record_inversion(lg_coeff);
    }
    let (Some(cf), Some(cg)) = (one.div(lf_coeff), one.div(lg_coeff)) else {
        return ParamGbPoly::zero(f.n_vars, f.n_params);
    };

    f.mul_monomial(&shift_f, &cf)
        .sub(&g.mul_monomial(&shift_g, &cg))
}

/// Reduce each basis element by the others and drop the redundant ones.
fn interreduce(
    mut basis: Vec<ParamGbPoly>,
    order: MonomialOrder,
    conds: &mut ConditionLog,
) -> Vec<ParamGbPoly> {
    let mut i = 0;
    while i < basis.len() {
        let others: Vec<ParamGbPoly> = basis
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, g)| g.clone())
            .collect();
        let reduced = reduce_param(&basis[i], &others, order, conds);
        if reduced.is_zero() {
            basis.remove(i);
        } else {
            basis[i] = reduced.make_monic(order, conds);
            i += 1;
        }
    }
    basis
}

// ---------------------------------------------------------------------------
// The basis
// ---------------------------------------------------------------------------

/// A Gröbner basis computed over the coefficient field `Q(p₁, …, p_m)`.
///
/// Read the generators back with [`Self::generators`], the hypotheses the
/// computation made with [`Self::conditions`], and check or apply a parameter
/// point with [`Self::vanishing_conditions`] / [`Self::specialize`].
#[derive(Clone, Debug)]
pub struct ParamGroebnerBasis {
    generators: Vec<ParamGbPoly>,
    order: MonomialOrder,
    n_vars: usize,
    n_params: usize,
    conditions: Vec<ParamPoly>,
}

impl ParamGroebnerBasis {
    /// Compute a Gröbner basis of `⟨gens⟩ ⊆ Q(params)[vars]` under `order`.
    pub fn compute(
        gens: Vec<ParamGbPoly>,
        order: MonomialOrder,
    ) -> Result<Self, ParamGroebnerError> {
        let first = gens.first().ok_or(ParamGroebnerError::NoGenerators)?;
        let (n_vars, n_params) = (first.n_vars, first.n_params);
        for g in &gens {
            if (g.n_vars, g.n_params) != (n_vars, n_params) {
                return Err(ParamGroebnerError::ShapeMismatch {
                    expected: (n_vars, n_params),
                    got: (g.n_vars, g.n_params),
                });
            }
        }

        let mut conds = ConditionLog::default();
        // The input coefficients have to be regular at the parameter point
        // before anything else can be said about them.
        for g in &gens {
            for c in g.terms.values() {
                conds.record_regular(c);
            }
        }

        let initial: Vec<ParamGbPoly> = gens
            .into_iter()
            .filter(|g| !g.is_zero())
            .map(|g| g.make_monic(order, &mut conds))
            .collect();

        if initial.is_empty() {
            return Ok(ParamGroebnerBasis {
                generators: vec![],
                order,
                n_vars,
                n_params,
                conditions: conds.finish(),
            });
        }

        let mut basis: Vec<ParamGbPoly> = Vec::with_capacity(initial.len() * 2);
        let mut basis_sugar: Vec<u32> = Vec::with_capacity(initial.len() * 2);
        let mut basis_lead: Vec<Vec<u32>> = Vec::with_capacity(initial.len() * 2);
        let mut pair_vec: Vec<CriticalPair> = Vec::new();

        for gen in initial {
            let sugar = gen.sugar();
            let Some(lead) = gen.leading_exp(order) else {
                continue;
            };
            let new_idx = basis.len();
            basis.push(gen);
            basis_sugar.push(sugar);
            basis_lead.push(lead);
            update_pairs(&basis_lead, &basis_sugar, &mut pair_vec, new_idx);
        }

        let mut heap: BinaryHeap<CriticalPair> = BinaryHeap::from(pair_vec);

        while let Some(pair) = heap.pop() {
            let sp = s_polynomial_param(&basis[pair.i], &basis[pair.j], order, &mut conds);
            let r = reduce_param(&sp, &basis, order, &mut conds);

            if !r.is_zero() {
                let r = r.make_monic(order, &mut conds);
                let sugar = r.sugar();
                let Some(lead) = r.leading_exp(order) else {
                    continue;
                };
                let new_idx = basis.len();
                basis.push(r);
                basis_sugar.push(sugar);
                basis_lead.push(lead);

                let mut pv: Vec<CriticalPair> = heap.into_vec();
                update_pairs(&basis_lead, &basis_sugar, &mut pv, new_idx);
                heap = BinaryHeap::from(pv);
            }
        }

        let generators = interreduce(basis, order, &mut conds);
        Ok(ParamGroebnerBasis {
            generators,
            order,
            n_vars,
            n_params,
            conditions: conds.finish(),
        })
    }

    /// The basis generators, interreduced and monic.
    pub fn generators(&self) -> &[ParamGbPoly] {
        &self.generators
    }

    /// The monomial order the generators are reduced under.
    pub fn order(&self) -> MonomialOrder {
        self.order
    }

    /// Number of ring variables.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Number of parameters in the coefficient field.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Number of generators.
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    /// True when the basis has no generators.
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    /// The polynomials in the parameters whose non-vanishing this basis
    /// assumed, each irreducible, primitive and with a positive leading
    /// coefficient.
    ///
    /// The basis describes exactly those parameter points at which none of
    /// them vanishes; the degeneracy locus is the union of the hypersurfaces
    /// they cut out.  An empty list means the basis holds at every rational
    /// parameter point.
    ///
    /// The list is sufficient, not necessary — see the module documentation.
    pub fn conditions(&self) -> &[ParamPoly] {
        &self.conditions
    }

    /// The conditions that vanish at `values` — empty exactly when the basis
    /// applies at that parameter point.
    pub fn vanishing_conditions(&self, values: &[Rational]) -> Vec<ParamPoly> {
        self.conditions
            .iter()
            .filter(|c| c.eval(values) == 0)
            .cloned()
            .collect()
    }

    /// True when `values` is off the degeneracy locus.
    pub fn is_regular_at(&self, values: &[Rational]) -> bool {
        self.conditions.iter().all(|c| c.eval(values) != 0)
    }

    /// Substitute rational values for the parameters.
    ///
    /// Off the degeneracy locus the result is the reduced Gröbner basis of the
    /// specialised ideal under the same order — the same thing computing over
    /// ℚ from the specialised generators would produce.  On the locus this
    /// refuses with [`ParamGroebnerError::Degenerate`] rather than returning a
    /// basis that is not one.
    pub fn specialize(&self, values: &[Rational]) -> Result<Vec<GbPoly>, ParamGroebnerError> {
        if values.len() != self.n_params {
            return Err(ParamGroebnerError::WrongArity {
                expected: self.n_params,
                got: values.len(),
            });
        }
        let vanishing = self.vanishing_conditions(values);
        if !vanishing.is_empty() {
            return Err(ParamGroebnerError::Degenerate { vanishing });
        }
        let mut out = Vec::with_capacity(self.generators.len());
        for g in &self.generators {
            match g.specialize(values) {
                Some(p) => out.push(p),
                // Unreachable while the condition set is closed under the
                // argument in the module docs; reported rather than papered
                // over if it ever is not.
                None => {
                    return Err(ParamGroebnerError::Degenerate {
                        vanishing: g
                            .terms
                            .values()
                            .filter(|c| c.denominator().eval(values) == 0)
                            .map(|c| c.denominator().clone())
                            .collect(),
                    })
                }
            }
        }
        Ok(out)
    }

    /// Reduce a polynomial modulo this basis and return the remainder.
    pub fn reduce(&self, p: &ParamGbPoly) -> ParamGbPoly {
        let mut sink = ConditionLog::default();
        reduce_param(p, &self.generators, self.order, &mut sink)
    }

    /// Ideal membership: true when [`Self::reduce`] gives zero.
    pub fn contains(&self, p: &ParamGbPoly) -> bool {
        self.reduce(p).is_zero()
    }

    /// The elimination ideal `I ∩ Q(params)[remaining vars]`.
    ///
    /// Drops every generator whose support mentions one of `vars`, exactly as
    /// [`super::GroebnerBasis::eliminate`] does.  Under a `Lex` basis with the
    /// eliminated variables ordered first, what is left generates the
    /// elimination ideal.
    pub fn eliminate(&self, vars: &[usize]) -> ParamGroebnerBasis {
        let generators: Vec<ParamGbPoly> = self
            .generators
            .iter()
            .filter(|g| {
                !g.terms
                    .keys()
                    .any(|e| vars.iter().any(|&i| e.get(i).copied().unwrap_or(0) > 0))
            })
            .cloned()
            .collect();
        ParamGroebnerBasis {
            generators,
            order: self.order,
            n_vars: self.n_vars,
            n_params: self.n_params,
            conditions: self.conditions.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::groebner::GroebnerBasis;

    /// `c · x^var_exp · p^par_exp` as a one-term parametric polynomial.
    fn term(
        n_vars: usize,
        n_params: usize,
        var_exp: &[u32],
        par_exp: &[u32],
        c: i64,
    ) -> ParamGbPoly {
        let mut mono = ParamPoly::zero(n_params);
        mono.terms.insert(par_exp.to_vec(), rug::Integer::from(1));
        let coeff =
            QParam::from_rational(&Rational::from(c), n_params).mul(&QParam::from_poly(mono));
        let mut p = ParamGbPoly::zero(n_vars, n_params);
        if !coeff.is_zero() {
            p.terms.insert(var_exp.to_vec(), coeff);
        }
        p
    }

    fn sum(parts: Vec<ParamGbPoly>) -> ParamGbPoly {
        let mut it = parts.into_iter();
        let first = it.next().expect("non-empty");
        it.fold(first, |a, b| a.add(&b))
    }

    fn rat(v: i64) -> Rational {
        Rational::from(v)
    }

    /// The same system written over ℚ with the parameters substituted, as a
    /// plain `GbPoly` — the oracle for the specialisation tests.
    fn gb_over_q(polys: &[Vec<(Vec<u32>, Rational)>], n_vars: usize) -> GroebnerBasis {
        let gens: Vec<GbPoly> = polys
            .iter()
            .map(|terms| GbPoly {
                terms: terms.iter().cloned().collect(),
                n_vars,
            })
            .collect();
        GroebnerBasis::compute(gens, MonomialOrder::Lex)
    }

    #[test]
    fn linear_system_with_a_parametric_coefficient() {
        // { a·x - y, x + y - 1 } over Q(a)[x, y], lex with x > y.
        let f = term(2, 1, &[1, 0], &[1], 1).sub(&term(2, 1, &[0, 1], &[0], 1));
        let g = sum(vec![
            term(2, 1, &[1, 0], &[0], 1),
            term(2, 1, &[0, 1], &[0], 1),
            term(2, 1, &[0, 0], &[0], -1),
        ]);
        let gb = ParamGroebnerBasis::compute(vec![f, g], MonomialOrder::Lex).unwrap();
        assert_eq!(gb.len(), 2, "expected a triangular basis");

        // y = a/(a+1) is the solution, so (a + 1) must be a reported condition.
        assert!(
            gb.conditions()
                .iter()
                .any(|c| *c == ParamPoly::var(0, 1).add(&ParamPoly::one(1))),
            "a + 1 must be reported: the system is degenerate at a = -1, got {:?}",
            gb.conditions()
        );
        assert!(!gb.is_regular_at(&[rat(-1)]));
        assert!(gb.is_regular_at(&[rat(3)]));
    }

    #[test]
    fn specialisation_agrees_with_computing_over_q() {
        // { a·x - y, x + y - 1 }, specialised at several values of a.
        let f = term(2, 1, &[1, 0], &[1], 1).sub(&term(2, 1, &[0, 1], &[0], 1));
        let g = sum(vec![
            term(2, 1, &[1, 0], &[0], 1),
            term(2, 1, &[0, 1], &[0], 1),
            term(2, 1, &[0, 0], &[0], -1),
        ]);
        let gb = ParamGroebnerBasis::compute(vec![f, g], MonomialOrder::Lex).unwrap();

        for a in [2i64, 3, -5, 7] {
            let spec = gb.specialize(&[rat(a)]).expect("regular point");
            let direct = gb_over_q(
                &[
                    vec![
                        (vec![1, 0], Rational::from(a)),
                        (vec![0, 1], Rational::from(-1)),
                    ],
                    vec![
                        (vec![1, 0], Rational::from(1)),
                        (vec![0, 1], Rational::from(1)),
                        (vec![0, 0], Rational::from(-1)),
                    ],
                ],
                2,
            );
            let mut got: Vec<_> = spec.iter().map(|p| p.terms.clone()).collect();
            let mut want: Vec<_> = direct
                .generators()
                .iter()
                .map(|p| p.terms.clone())
                .collect();
            got.sort();
            want.sort();
            assert_eq!(got, want, "specialisation at a = {a} disagrees with ℚ");
        }
    }

    #[test]
    fn degenerate_point_is_refused_and_is_genuinely_different() {
        let f = term(2, 1, &[1, 0], &[1], 1).sub(&term(2, 1, &[0, 1], &[0], 1));
        let g = sum(vec![
            term(2, 1, &[1, 0], &[0], 1),
            term(2, 1, &[0, 1], &[0], 1),
            term(2, 1, &[0, 0], &[0], -1),
        ]);
        let gb = ParamGroebnerBasis::compute(vec![f, g], MonomialOrder::Lex).unwrap();

        let err = gb.specialize(&[rat(-1)]).unwrap_err();
        assert_eq!(err.code(), "E-PARAMGB-004");
        let ParamGroebnerError::Degenerate { vanishing } = &err else {
            panic!("expected a degeneracy report, got {err}");
        };
        assert!(!vanishing.is_empty());

        // And the point really is special: at a = -1 the system {-x - y, x + y - 1}
        // is inconsistent, so its basis over ℚ is {1}, which is not the
        // specialisation of the generic basis.
        let direct = gb_over_q(
            &[
                vec![
                    (vec![1, 0], Rational::from(-1)),
                    (vec![0, 1], Rational::from(-1)),
                ],
                vec![
                    (vec![1, 0], Rational::from(1)),
                    (vec![0, 1], Rational::from(1)),
                    (vec![0, 0], Rational::from(-1)),
                ],
            ],
            2,
        );
        assert_eq!(direct.len(), 1);
        assert!(direct.generators()[0].terms.contains_key(&vec![0, 0]));
    }

    #[test]
    fn no_parameters_reproduces_the_rational_engine() {
        // x² - 1, x - 1 with an empty parameter list.
        let f = term(1, 0, &[2], &[], 1).sub(&term(1, 0, &[0], &[], 1));
        let g = term(1, 0, &[1], &[], 1).sub(&term(1, 0, &[0], &[], 1));
        let gb = ParamGroebnerBasis::compute(vec![f, g], MonomialOrder::Lex).unwrap();
        assert_eq!(gb.len(), 1);
        assert!(gb.conditions().is_empty());
        let spec = gb.specialize(&[]).unwrap();
        assert_eq!(spec.len(), 1);
        assert!(spec[0].terms.contains_key(&vec![1]));
    }

    #[test]
    fn from_gbpoly_moves_the_trailing_slots_into_the_field() {
        // a·x + a²·y over vars [x, y] and params [a].
        let src = GbPoly {
            terms: [
                (vec![1u32, 0, 1], Rational::from(1)),
                (vec![0, 1, 2], Rational::from(1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 3,
        };
        let p = ParamGbPoly::from_gbpoly(&src, 2, 1).unwrap();
        assert_eq!(p.n_terms(), 2);
        let cx = p.terms.get(&vec![1u32, 0]).unwrap();
        assert_eq!(*cx, QParam::from_poly(ParamPoly::var(0, 1)));
        let cy = p.terms.get(&vec![0u32, 1]).unwrap();
        assert_eq!(
            *cy,
            QParam::from_poly(ParamPoly::var(0, 1).mul(&ParamPoly::var(0, 1)))
        );
        assert!(ParamGbPoly::from_gbpoly(&src, 1, 1).is_none());
    }

    #[test]
    fn membership_and_elimination() {
        // { x - t, y - t² } in Q(a)[t, x, y] — implicitisation, no parameters
        // actually used, so the elimination ideal is ⟨y - x²⟩.
        let n = 3;
        let f = term(n, 1, &[1, 0, 0], &[0], -1).add(&term(n, 1, &[0, 1, 0], &[0], 1));
        let g = term(n, 1, &[2, 0, 0], &[0], -1).add(&term(n, 1, &[0, 0, 1], &[0], 1));
        let gb = ParamGroebnerBasis::compute(vec![f.clone(), g], MonomialOrder::Lex).unwrap();
        assert!(gb.contains(&f));
        let el = gb.eliminate(&[0]);
        assert_eq!(el.len(), 1);
        let rel = &el.generators()[0];
        assert!(rel.terms.contains_key(&vec![0, 2, 0]));
        assert!(rel.terms.contains_key(&vec![0, 0, 1]));
    }

    #[test]
    fn wrong_arity_is_reported() {
        let f = term(1, 2, &[1], &[0, 0], 1);
        let gb = ParamGroebnerBasis::compute(vec![f], MonomialOrder::Lex).unwrap();
        let err = gb.specialize(&[rat(1)]).unwrap_err();
        assert_eq!(err.code(), "E-PARAMGB-003");
    }

    #[test]
    fn no_generators_is_reported() {
        let err = ParamGroebnerBasis::compute(vec![], MonomialOrder::Lex).unwrap_err();
        assert_eq!(err.code(), "E-PARAMGB-001");
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let f = term(2, 1, &[1, 0], &[0], 1);
        let g = term(3, 1, &[1, 0, 0], &[0], 1);
        let err = ParamGroebnerBasis::compute(vec![f, g], MonomialOrder::Lex).unwrap_err();
        assert_eq!(err.code(), "E-PARAMGB-002");
    }
}
