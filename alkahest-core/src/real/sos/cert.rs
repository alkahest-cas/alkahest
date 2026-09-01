//! Positivity certificate objects, their exact verifier, and their renderings.
//!
//! A certificate is a *claim plus its proof*, and the proof is a polynomial
//! identity that can be re-expanded from scratch.  [`PositivityCertificate::verify`]
//! does exactly that, in ℚ, and nothing in this subsystem ever returns a
//! certificate that has not been through it.

use super::ratpoly::{format_rational, lean_rational, rational_expr, RatPoly};
use crate::kernel::{ExprId, ExprPool};
use rug::Rational;
use std::fmt;

/// One `σ · q²` summand of a sum-of-squares polynomial.  `coeff` is always
/// strictly positive (zero terms are dropped at construction).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SosTerm {
    pub coeff: Rational,
    pub square: RatPoly,
}

/// A sum of squares with non-negative rational weights: `Σ_j σ_j q_j²`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SosPoly {
    pub terms: Vec<SosTerm>,
}

impl SosPoly {
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn push(&mut self, coeff: Rational, square: RatPoly) {
        if coeff > 0 && !square.is_zero() {
            self.terms.push(SosTerm { coeff, square });
        }
    }

    /// Expand to an ordinary polynomial.  Exact.
    pub fn to_poly(&self, nvars: usize) -> RatPoly {
        let mut acc = RatPoly::zero(nvars);
        for t in &self.terms {
            acc = acc.add(&t.square.square().scale(&t.coeff));
        }
        acc
    }

    /// True iff every weight is non-negative — the structural half of the
    /// soundness argument (the other half is the identity check).
    pub fn weights_nonnegative(&self) -> bool {
        self.terms.iter().all(|t| t.coeff >= 0)
    }
}

/// One `(Π_{i ∈ constraints} g_i) · σ` term of a certificate.
///
/// An empty `constraints` list is the unconstrained `σ_0` term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Multiplier {
    /// Multiset of constraint indices whose product forms the weight.
    pub constraints: Vec<usize>,
    pub sos: SosPoly,
}

/// Which Positivstellensatz shape the certificate has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertificateKind {
    /// `p = Σ σ_j q_j²` — unconstrained sum of squares.
    Sos,
    /// `p = Σ_α c_α Π g_i^{α_i}` with `c_α ≥ 0` rational constants.
    Handelman,
    /// `p = σ_0 + Σ σ_i g_i` with every `σ` a sum of squares.
    Putinar,
}

impl CertificateKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CertificateKind::Sos => "sos",
            CertificateKind::Handelman => "handelman",
            CertificateKind::Putinar => "putinar",
        }
    }
}

impl fmt::Display for CertificateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An exact, re-checkable proof that `p ≥ 0` on `{x : g_i(x) ≥ 0}`.
#[derive(Clone, Debug)]
pub struct PositivityCertificate {
    pub vars: Vec<ExprId>,
    pub var_names: Vec<String>,
    pub target: RatPoly,
    pub constraints: Vec<RatPoly>,
    pub kind: CertificateKind,
    /// The degree / level bound that was actually used to find this certificate.
    pub degree: u32,
    /// Terms of the identity; `terms[k]` is `(Π g_i) · σ_k`.
    pub terms: Vec<Multiplier>,
    /// Human-readable audit trail: how the search proceeded.
    pub log: Vec<String>,
}

impl PositivityCertificate {
    pub fn nvars(&self) -> usize {
        self.vars.len()
    }

    /// A Reznick-style multiplier `σ = (x_1² + … + x_n²)^N`, present exactly
    /// when the certificate proves `target ≥ 0` indirectly via `target·σ =
    /// Σ terms` rather than directly via `target = Σ terms`. `None` for a
    /// direct certificate (the only kind before 3.10.0).
    ///
    /// Not stored: derived here by literally re-deriving `N` (trying every
    /// `N` up to the search's own budget and checking `target·(Σxᵢ²)^N ==`
    /// the re-expanded identity) rather than trusted from whatever the search
    /// that produced this certificate happened to remember — the same
    /// "recompute, never trust the search" discipline [`Self::verify`]
    /// already applies to everything else in this type, extended one step
    /// further so this struct's field set never needs to record which route
    /// a certificate took, only the identity it proves.
    pub fn multiplier(&self) -> Option<RatPoly> {
        let rhs = self.expand();
        if self.target == rhs {
            return None;
        }
        let sigma_base = RatPoly::sum_of_squares(self.nvars());
        for n in 1..=super::MAX_MULTIPLIER_POWER {
            let sigma = sigma_base.pow(n);
            if self.target.mul(&sigma) == rhs {
                return Some(sigma);
            }
        }
        None
    }

    /// Expand the right-hand side of the certificate identity, exactly.
    pub fn expand(&self) -> RatPoly {
        let n = self.vars.len();
        let mut acc = RatPoly::zero(n);
        for m in &self.terms {
            let mut weight = RatPoly::one(n);
            for &i in &m.constraints {
                weight = weight.mul(&self.constraints[i]);
            }
            acc = acc.add(&weight.mul(&m.sos.to_poly(n)));
        }
        acc
    }

    /// Exact verification.  Returns `Ok(())` only if
    ///
    /// 1. every constraint index is in range,
    /// 2. every SOS weight is non-negative,
    /// 3. the re-expanded right-hand side is *identically* `target` (direct
    ///    certificates) or *identically* `target · multiplier` (multiplier
    ///    certificates — see [`Self::multiplier`]), and
    /// 4. for a multiplier certificate, `multiplier` really is `(x_1² + … +
    ///    x_n²)^N` for some `N` (recomputed here, not trusted from the
    ///    search) and `target` is non-negative at the origin, the one point
    ///    `multiplier` cannot rule out.
    ///
    /// This is called on every path that returns a certificate; a failure here
    /// is a bug in the search, never something the caller sees as a success.
    pub fn verify(&self) -> Result<(), String> {
        for m in &self.terms {
            for &i in &m.constraints {
                if i >= self.constraints.len() {
                    return Err(format!(
                        "certificate references constraint #{i} but only {} were given",
                        self.constraints.len()
                    ));
                }
            }
            if !m.sos.weights_nonnegative() {
                return Err("certificate contains a negative sum-of-squares weight".to_string());
            }
        }
        let rhs = self.expand();
        match self.multiplier() {
            None => {
                if self.target == rhs {
                    Ok(())
                } else {
                    let diff = self.target.sub(&rhs);
                    Err(format!(
                        "certificate does not re-expand to the target; residual = {}",
                        diff.display(&self.var_names)
                    ))
                }
            }
            Some(sigma) => {
                let lhs = self.target.mul(&sigma);
                if lhs != rhs {
                    let diff = lhs.sub(&rhs);
                    return Err(format!(
                        "certificate does not re-expand to target·multiplier; residual = {}",
                        diff.display(&self.var_names)
                    ));
                }
                let deg = sigma.total_degree();
                if deg % 2 != 0 {
                    return Err(
                        "multiplier has odd degree, so it cannot be a power of a sum \
                                 of squares"
                            .to_string(),
                    );
                }
                let n = deg / 2;
                let expected = RatPoly::sum_of_squares(self.nvars()).pow(n);
                if sigma != expected {
                    return Err(
                        "multiplier is not recognised as (x_1^2 + ... + x_n^2)^N for \
                                 any N, so its non-negativity is not established by this \
                                 checker"
                            .to_string(),
                    );
                }
                let origin = vec![Rational::from(0); self.nvars()];
                if self.target.eval(&origin) < 0 {
                    return Err(
                        "target is negative at the origin, the one point the multiplier \
                         (x_1^2 + ... + x_n^2)^N cannot rule out"
                            .to_string(),
                    );
                }
                Ok(())
            }
        }
    }

    /// Total number of squares across all multipliers.
    pub fn num_squares(&self) -> usize {
        self.terms.iter().map(|m| m.sos.terms.len()).sum()
    }

    /// The right-hand side of the identity as a symbolic expression.
    pub fn to_expr(&self, pool: &ExprPool) -> ExprId {
        let mut summands: Vec<ExprId> = Vec::new();
        for m in &self.terms {
            for t in &m.sos.terms {
                let mut factors: Vec<ExprId> = Vec::new();
                if t.coeff != 1 {
                    factors.push(rational_expr(&t.coeff, pool));
                }
                for &i in &m.constraints {
                    factors.push(self.constraints[i].to_expr(&self.vars, pool));
                }
                let two = pool.integer(2);
                let base = t.square.to_expr(&self.vars, pool);
                factors.push(pool.pow(base, two));
                summands.push(match factors.len() {
                    1 => factors[0],
                    _ => pool.mul(factors),
                });
            }
        }
        match summands.len() {
            0 => pool.integer(0),
            1 => summands[0],
            _ => pool.add(summands),
        }
    }

    /// A one-line-per-term human rendering of the identity.
    pub fn identity_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for m in &self.terms {
            for t in &m.sos.terms {
                let sq = format!("({})^2", t.square.display(&self.var_names));
                let mut s = String::new();
                if t.coeff != 1 {
                    s.push_str(&format_rational(&t.coeff));
                    s.push('*');
                }
                for &i in &m.constraints {
                    s.push_str(&format!(
                        "({})*",
                        self.constraints[i].display(&self.var_names)
                    ));
                }
                s.push_str(&sq);
                parts.push(s);
            }
        }
        let rhs = if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        };
        // Render the whole identity, not just its right-hand side: the point of
        // a certificate is that a reader can check `target = rhs` (or
        // `target * multiplier = rhs`, for a multiplier certificate) by
        // expanding.
        match self.multiplier() {
            None => format!("{} = {}", self.target.display(&self.var_names), rhs),
            Some(sigma) => format!(
                "({}) * ({}) = {}",
                self.target.display(&self.var_names),
                sigma.display(&self.var_names),
                rhs
            ),
        }
    }

    /// Description of the statement the certificate proves.
    pub fn claim_string(&self) -> String {
        let p = self.target.display(&self.var_names);
        if self.constraints.is_empty() {
            format!("0 <= {p}  for all real {}", self.var_names.join(", "))
        } else {
            let hyps: Vec<String> = self
                .constraints
                .iter()
                .map(|g| format!("0 <= {}", g.display(&self.var_names)))
                .collect();
            format!("{}  ==>  0 <= {p}", hyps.join(" and "))
        }
    }

    // -----------------------------------------------------------------------
    // Lean export
    // -----------------------------------------------------------------------

    /// Emit a self-contained Lean 4 source snippet proving the claim.
    ///
    /// Returns `None` when the certificate has not been verified or when the
    /// statement cannot be rendered soundly — an unsound or `sorry`-bearing
    /// certificate is never emitted.  The unconstrained shape is proved by
    /// rewriting with the (`ring`-checked) identity and closing with
    /// `positivity`; the constrained shape hands `nlinarith` the identity plus
    /// one non-negativity fact per product term.
    pub fn to_lean(&self) -> Option<String> {
        if self.verify().is_err() {
            return None;
        }
        if self.terms.iter().all(|m| m.sos.is_empty()) && !self.target.is_zero() {
            return None;
        }
        let names = &self.var_names;
        if names.iter().any(|n| !is_lean_ident(n)) {
            return None;
        }
        // Every listed variable, for the multiplier block (whose `σ` mentions
        // them all); only the ones the statement actually mentions, for the
        // direct blocks — see [`Self::lean_bound_vars`].
        let all_binders: String = names
            .iter()
            .map(|n| format!("({n} : ℝ) "))
            .collect::<String>();
        let binders = self.lean_binders(&self.lean_bound_vars());

        let rhs = self.lean_rhs();
        let target = self.target.to_lean(names);

        let mut out = String::new();
        out.push_str("import Mathlib.Tactic\n\n");
        out.push_str(&format!(
            "-- Alkahest positivity certificate ({}, degree {})\n",
            self.kind, self.degree
        ));
        out.push_str(&format!("-- {}\n\n", self.claim_string()));

        if let Some(sigma) = self.multiplier() {
            out.push_str(&self.lean_multiplier_block(&all_binders, &target, &rhs, &sigma));
        } else if self.constraints.is_empty() {
            out.push_str(&format!(
                "theorem alkahest_sos_identity {binders}:\n    {target} = {rhs} := by\n  ring\n\n"
            ));
            out.push_str(&format!(
                "theorem alkahest_nonneg {binders}:\n    (0 : ℝ) ≤ {target} := by\n\
                 \x20 rw [alkahest_sos_identity]\n  positivity\n"
            ));
        } else {
            let hyp_binders: String = self
                .constraints
                .iter()
                .enumerate()
                .map(|(i, g)| format!("(hg{i} : (0 : ℝ) ≤ {}) ", g.to_lean(names)))
                .collect::<String>();
            out.push_str(&format!(
                "theorem alkahest_positivstellensatz_identity {binders}:\n\
                 \x20   {target} = {rhs} := by\n  ring\n\n"
            ));
            let hints = self.lean_hints();
            out.push_str(&format!(
                "theorem alkahest_nonneg {binders}{hyp_binders}:\n\
                 \x20   (0 : ℝ) ≤ {target} := by\n\
                 \x20 rw [alkahest_positivstellensatz_identity]\n\
                 \x20 nlinarith [{hints}]\n"
            ));
        }
        // Defense in depth, matching every emitter in `crate::lean`: an
        // admission anywhere in the rendered file (goal text included) means
        // withhold, not warn.
        if out.contains("sorry") || out.contains("admit") {
            return None;
        }
        Some(out)
    }

    /// Lean rendering for a multiplier certificate: `target ≥ 0` follows
    /// from `σ·target = rhs ≥ 0` away from the origin (where `σ > 0`), and
    /// from a direct numeric check at the origin (where `σ` vanishes).
    /// `σ = (x_1² + … + x_n²)^power` by construction — [`Self::verify`] has
    /// already confirmed this exactly, so the factored form used here is
    /// not an extra trust assumption.
    fn lean_multiplier_block(
        &self,
        binders: &str,
        target: &str,
        rhs: &str,
        sigma: &RatPoly,
    ) -> String {
        let names = &self.var_names;
        let n = names.len();
        let power = sigma.total_degree() / 2;
        let sum_sq: String = names
            .iter()
            .map(|nm| format!("{nm} ^ 2"))
            .collect::<Vec<_>>()
            .join(" + ");
        let sigma_base = format!("({sum_sq})");
        let sigma_factored = if power == 1 {
            sigma_base.clone()
        } else {
            format!("({sum_sq}) ^ {power}")
        };
        let args = names.join(" ");

        let mut out = String::new();
        out.push_str(&format!(
            "theorem alkahest_multiplier_factor {binders}:\n    {sigma_factored} * ({target}) = {rhs} := by\n  ring\n\n"
        ));
        // `positivity` closes `0 ≤ rhs` because `rhs` is a sum of squares with
        // positive rational weights; `nlinarith` cannot see that on its own.
        out.push_str(&format!(
            "theorem alkahest_rhs_nonneg {binders}:\n    (0 : ℝ) ≤ {rhs} := by\n  positivity\n\n"
        ));
        out.push_str(&format!(
            "theorem alkahest_nonneg {binders}:\n    (0 : ℝ) ≤ {target} := by\n"
        ));
        out.push_str(&format!(
            "  have hid := alkahest_multiplier_factor {args}\n"
        ));
        out.push_str(&format!("  have hrhs := alkahest_rhs_nonneg {args}\n"));

        if n == 1 {
            let x = &names[0];
            out.push_str(&format!("  by_cases hz : {x} = 0\n"));
            out.push_str("  · subst hz\n    norm_num\n");
            out.push_str(&format!(
                "  · have hs0 : (0 : ℝ) < {sigma_base} := by\n      nlinarith [sq_pos_of_ne_zero hz]\n"
            ));
        } else {
            let conj: String = names
                .iter()
                .map(|nm| format!("{nm} = 0"))
                .collect::<Vec<_>>()
                .join(" ∧ ");
            let obtain: String = (1..=n)
                .map(|i| format!("h{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let substs: String = (1..=n)
                .map(|i| format!("    subst h{i}"))
                .collect::<Vec<_>>()
                .join("\n");
            out.push_str(&format!("  by_cases hz : {conj}\n"));
            out.push_str(&format!(
                "  · obtain ⟨{obtain}⟩ := hz\n{substs}\n    norm_num\n"
            ));
            out.push_str(&format!(
                "  · have hs0 : (0 : ℝ) < {sigma_base} := by\n{}",
                lean_sigma_pos_proof(names, 0, "hz", 6)
            ));
        }

        if power == 1 {
            out.push_str("    nlinarith [hid, hrhs, hs0]\n");
        } else {
            out.push_str(&format!(
                "    have hs : (0 : ℝ) < {sigma_factored} := pow_pos hs0 {power}\n    nlinarith [hid, hrhs, hs]\n"
            ));
        }
        out
    }

    /// Which of `var_names` actually occur in the terms the direct (non-
    /// multiplier) Lean blocks render: the target, the constraints, and every
    /// square of the identity.
    ///
    /// Binding a variable the statement never mentions is not merely untidy.
    /// Under `-DwarningAsError=true` Mathlib's `unusedVariables` linter makes
    /// it a hard error, and — worse — `rw [alkahest_sos_identity]` cannot
    /// instantiate a binder that does not appear in the goal, so it leaves a
    /// stray `⊢ ℝ` side goal. `prove_nonneg(x**2, [x, y])` emitted a file that
    /// failed on both counts. The multiplier block is exempt: its
    /// `σ = (x₁² + … + xₙ²)` mentions every name by construction.
    fn lean_bound_vars(&self) -> Vec<bool> {
        let mut used = vec![false; self.var_names.len()];
        let mark = |p: &RatPoly, used: &mut Vec<bool>| {
            for (i, u) in used.iter_mut().enumerate() {
                if p.degree_in(i) > 0 {
                    *u = true;
                }
            }
        };
        mark(&self.target, &mut used);
        for g in &self.constraints {
            mark(g, &mut used);
        }
        for m in &self.terms {
            for t in &m.sos.terms {
                mark(&t.square, &mut used);
            }
        }
        used
    }

    /// `(x : ℝ) ` binders for exactly [`Self::lean_bound_vars`].
    fn lean_binders(&self, used: &[bool]) -> String {
        self.var_names
            .iter()
            .zip(used)
            .filter(|(_, &u)| u)
            .map(|(n, _)| format!("({n} : ℝ) "))
            .collect()
    }

    fn lean_rhs(&self) -> String {
        let names = &self.var_names;
        let mut parts: Vec<String> = Vec::new();
        for m in &self.terms {
            for t in &m.sos.terms {
                let mut factors: Vec<String> = Vec::new();
                if t.coeff != 1 {
                    factors.push(lean_rational(&t.coeff));
                }
                for &i in &m.constraints {
                    factors.push(self.constraints[i].to_lean(names));
                }
                factors.push(format!("{} ^ (2 : ℕ)", t.square.to_lean(names)));
                parts.push(factors.join(" * "));
            }
        }
        if parts.is_empty() {
            "(0 : ℝ)".to_string()
        } else {
            format!("({})", parts.join(" + "))
        }
    }

    /// `nlinarith` hint terms: every square is non-negative, and every product
    /// of hypothesis constraints with a square is non-negative.
    fn lean_hints(&self) -> String {
        let names = &self.var_names;
        let mut hints: Vec<String> = Vec::new();
        for m in &self.terms {
            for t in &m.sos.terms {
                let sq = format!("sq_nonneg {}", t.square.to_lean(names));
                if m.constraints.is_empty() {
                    hints.push(sq);
                } else {
                    let mut acc = format!("hg{}", m.constraints[0]);
                    for &i in &m.constraints[1..] {
                        acc = format!("mul_nonneg {acc} hg{i}");
                    }
                    hints.push(format!("mul_nonneg {acc} ({sq})"));
                }
            }
        }
        hints.sort();
        hints.dedup();
        hints.join(", ")
    }
}

/// Nested `rcases … with … | …` on `¬(x_1 = 0 ∧ … ∧ x_n = 0)`, closing
/// every branch with `nlinarith [sq_pos_of_ne_zero h, sq_nonneg …]`.
/// `positivity` does not close `0 < x² + y²` from a single `x ≠ 0`.
fn lean_sigma_pos_proof(names: &[String], start: usize, hz: &str, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let n = names.len();
    let others = |nonzero: usize| -> String {
        names
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != nonzero)
            .map(|(_, nm)| format!("sq_nonneg {nm}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let nlin = |hyp: &str, nonzero: usize| -> String {
        let extra = others(nonzero);
        if extra.is_empty() {
            format!("nlinarith [sq_pos_of_ne_zero {hyp}]")
        } else {
            format!("nlinarith [sq_pos_of_ne_zero {hyp}, {extra}]")
        }
    };

    if start + 1 >= n {
        format!("{}{}\n", pad, nlin(hz, start))
    } else {
        let first = nlin("h0", start);
        let rest = start + 1;
        if rest + 1 >= n {
            format!(
                "{pad}rcases not_and_or.mp {hz} with h0 | hz'\n\
                 {pad}· {first}\n\
                 {pad}· {}\n",
                nlin("hz'", rest)
            )
        } else {
            format!(
                "{pad}rcases not_and_or.mp {hz} with h0 | hz'\n\
                 {pad}· {first}\n\
                 {pad}·\n{}",
                lean_sigma_pos_proof(names, rest, "hz'", indent + 2)
            )
        }
    }
}

fn is_lean_ident(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
}

impl fmt::Display for PositivityCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} certificate (degree {}): {}",
            self.kind,
            self.degree,
            self.identity_string()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn setup() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        (pool, x)
    }

    /// `x² + 2x + 1 = (x + 1)²`
    fn simple_cert() -> PositivityCertificate {
        let (_pool, x) = setup();
        let mut target = RatPoly::monomial(1, vec![2], Rational::from(1));
        target = target.add(&RatPoly::monomial(1, vec![1], Rational::from(2)));
        target = target.add(&RatPoly::constant(1, Rational::from(1)));
        let mut sq = RatPoly::monomial(1, vec![1], Rational::from(1));
        sq = sq.add(&RatPoly::constant(1, Rational::from(1)));
        let mut sos = SosPoly::default();
        sos.push(Rational::from(1), sq);
        PositivityCertificate {
            vars: vec![x],
            var_names: vec!["x".to_string()],
            target,
            constraints: vec![],
            kind: CertificateKind::Sos,
            degree: 2,
            terms: vec![Multiplier {
                constraints: vec![],
                sos,
            }],
            log: vec![],
        }
    }

    #[test]
    fn verify_accepts_a_true_identity() {
        assert!(simple_cert().verify().is_ok());
    }

    #[test]
    fn verify_rejects_a_perturbed_identity() {
        let mut c = simple_cert();
        c.target = c.target.add(&RatPoly::constant(1, Rational::from(1)));
        let err = c.verify().unwrap_err();
        assert!(err.contains("residual"), "unexpected message: {err}");
    }

    #[test]
    fn verify_rejects_negative_weights() {
        let mut c = simple_cert();
        c.terms[0].sos.terms[0].coeff = Rational::from(-1);
        assert!(c.verify().is_err());
    }

    #[test]
    fn lean_emission_is_gated_on_verification() {
        let mut c = simple_cert();
        assert!(c.to_lean().is_some());
        c.target = c.target.add(&RatPoly::constant(1, Rational::from(7)));
        assert!(c.to_lean().is_none());
    }

    #[test]
    fn lean_output_has_no_admissions() {
        let lean = simple_cert().to_lean().unwrap();
        assert!(!lean.contains("sorry"));
        assert!(!lean.contains("admit"));
        assert!(lean.contains("positivity"));
    }

    /// `x² + 2x + 1 ≥ 0` proved over the variable list `[x, y]`. The property:
    /// every binder the emitted theorems introduce must occur in the statement
    /// they introduce it for. `y` did not, which is a hard `unusedVariables`
    /// error under `-DwarningAsError=true` *and* leaves `rw
    /// [alkahest_sos_identity]` with an uninstantiated `⊢ ℝ` side goal.
    #[test]
    fn lean_binds_only_variables_the_statement_mentions() {
        let (pool, x) = setup();
        let y = pool.symbol("y", Domain::Real);
        let mut c = simple_cert();
        // Re-embed the same univariate identity in a two-variable ambient
        // space; `y` occurs in nothing.
        let lift = |p: &RatPoly| {
            let mut q = RatPoly::zero(2);
            for (e, coeff) in p.terms() {
                q = q.add(&RatPoly::monomial(2, vec![e[0], 0], coeff.clone()));
            }
            q
        };
        c.target = lift(&c.target);
        c.terms[0].sos.terms[0].square = lift(&c.terms[0].sos.terms[0].square);
        c.vars = vec![x, y];
        c.var_names = vec!["x".to_string(), "y".to_string()];
        assert!(c.verify().is_ok());

        let lean = c.to_lean().expect("a verified certificate must render");
        for line in lean.lines().filter(|l| l.starts_with("theorem ")) {
            let (binders, statement) = line.split_once(':').expect("a typed theorem line");
            for name in &c.var_names {
                let bound = binders.contains(&format!("({name} : ℝ)"));
                let mentioned = statement.contains(name.as_str());
                assert!(
                    !bound || mentioned,
                    "{name} is bound but never mentioned in: {line}"
                );
            }
        }
    }
}
