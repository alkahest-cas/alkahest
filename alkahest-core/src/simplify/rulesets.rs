/// Opt-in rule bundles for algebraic and transcendental identities.
///
/// These rules are **not** included in the default simplifier; include them via
/// [`simplify_with`](super::engine::simplify_with) when the target domain is known.
///
/// # Example
///
/// ```
/// # use alkahest_cas::kernel::{Domain, ExprPool};
/// # use alkahest_cas::simplify::{simplify_with, SimplifyConfig, rulesets};
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let rules = rulesets::trig_rules();
/// let tan_x = pool.func("tan", vec![x]);
/// let r = simplify_with(tan_x, &pool, &rules, SimplifyConfig::default());
/// // tan(x) → sin(x) * cos(x)^(-1)
/// ```
use crate::deriv::log::{DerivationLog, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::pattern::{Pattern, Substitution};
use crate::simplify::discrimination_net::{pattern_head, DiscriminationIndex};
use crate::simplify::rules::{FlattenAdd, FlattenMul, RewriteRule};

fn one_step(name: &'static str, before: ExprId, after: ExprId) -> DerivationLog {
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(name, before, after));
    log
}

// ---------------------------------------------------------------------------
// Trigonometric identity rules
// ---------------------------------------------------------------------------

/// `sin(-x) → -sin(x)` where `-x = (-1)*x`.
pub struct SinNeg;

impl RewriteRule for SinNeg {
    fn name(&self) -> &'static str {
        "sin_neg"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("sin", expr, pool)?;
        let inner = neg_inner(arg, pool)?;
        let after_inner = pool.func("sin", vec![inner]);
        let neg_one = pool.integer(-1_i32);
        let after = pool.mul(vec![neg_one, after_inner]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// `cos(-x) → cos(x)`.
pub struct CosNeg;

impl RewriteRule for CosNeg {
    fn name(&self) -> &'static str {
        "cos_neg"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("cos", expr, pool)?;
        let inner = neg_inner(arg, pool)?;
        let after = pool.func("cos", vec![inner]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// `tan(x) → sin(x) * cos(x)^(-1)`.
pub struct TanExpand;

impl RewriteRule for TanExpand {
    fn name(&self) -> &'static str {
        "tan_expand"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("tan", expr, pool)?;
        let sin_x = pool.func("sin", vec![arg]);
        let cos_x = pool.func("cos", vec![arg]);
        let cos_inv = pool.pow(cos_x, pool.integer(-1_i32));
        let after = pool.mul(vec![sin_x, cos_inv]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Coefficient-aware Pythagorean identity: `a·sin²(u) + a·cos²(u) → a`.
///
/// Matches `Add([…, c₁·sin²(u), …, c₂·cos²(u), …])` where `u` is any
/// sub-expression appearing identically in both terms and the *coefficients*
/// `c₁`, `c₂` (the remaining multiplicative factors of each term) are
/// structurally equal.  The matched pair is replaced by that common
/// coefficient `a`, so:
///
/// - `sin²(u) + cos²(u) → 1`            (the original bare case, `a = 1`),
/// - `2·sin²(u) + 2·cos²(u) → 2`,
/// - `a·sin²(u) + a·cos²(u) → a`        (symbolic `a`),
/// - `3 + 2·sin²(u) + 2·cos²(u) → 5`    (embedded in a larger sum; the
///   leftover numeric terms are folded so the result is fully reduced even
///   though `trig_rules` carries no general constant-folder).
///
/// The shared coefficient is matched on the canonically sorted factor lists,
/// so factor order is irrelevant.  Only a *single* `sin²`/`cos²` factor per
/// term is considered (terms like `sin²(u)·cos²(u)` are left untouched).
pub struct SinCosIdentity;

impl RewriteRule for SinCosIdentity {
    fn name(&self) -> &'static str {
        "sin_sq_plus_cos_sq"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // Find a term `c₁·sin²(u)` …
        let mut sin_pos = None;
        for (i, &a) in args.iter().enumerate() {
            if let Some((u, coeff)) = split_trig_sq("sin", a, pool) {
                sin_pos = Some((i, u, coeff));
                break;
            }
        }
        let (sin_idx, u, sin_coeff) = sin_pos?;

        // … and a matching `c₂·cos²(u)` with the same `u` and the same
        // coefficient factor multiset.
        let mut cos_idx = None;
        for (i, &a) in args.iter().enumerate() {
            if i == sin_idx {
                continue;
            }
            if let Some((cu, cos_coeff)) = split_trig_sq("cos", a, pool) {
                if cu == u && cos_coeff == sin_coeff {
                    cos_idx = Some(i);
                    break;
                }
            }
        }
        let cos_idx = cos_idx?;

        // The shared coefficient `a` (product of the leftover factors; empty → 1).
        let coeff_expr = match sin_coeff.len() {
            0 => pool.integer(1_i32),
            1 => sin_coeff[0],
            _ => pool.mul(sin_coeff.clone()),
        };

        // Replace the matched pair with `a` in the term list, then fold any
        // resulting numeric literals together (e.g. `3 + 2 → 5`).
        let mut new_args: Vec<ExprId> = args
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != sin_idx && i != cos_idx)
            .map(|(_, a)| a)
            .collect();
        new_args.push(coeff_expr);
        new_args = fold_numeric_terms(new_args, pool);

        let after = match new_args.len() {
            0 => pool.integer(0_i32),
            1 => new_args[0],
            _ => pool.add(new_args),
        };

        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Hyperbolic Pythagorean identity: `a·cosh²(u) − a·sinh²(u) → a`.
///
/// Unlike [`SinCosIdentity`] (same-sign coefficients), the Python surface builds
/// subtraction as `Add([cosh², Mul([-1, sinh²])])`, so the coefficients are
/// opposite in sign: empty/`a` on the cosh term and `[-1]`/`[-1, …a…]` on the
/// sinh term.  Matched pairs collapse to the shared magnitude `a`:
///
/// - `cosh²(u) − sinh²(u) → 1`,
/// - `2·cosh²(u) − 2·sinh²(u) → 2`,
/// - `a·cosh²(u) − a·sinh²(u) → a`.
///
/// The cosh coefficient must be sign-positive (no literal `-1` factor); the
/// sinh coefficient must carry exactly one `-1` whose remainder matches the
/// cosh coefficient as a multiset.
pub struct CoshSinhIdentity;

impl RewriteRule for CoshSinhIdentity {
    fn name(&self) -> &'static str {
        "cosh_sq_minus_sinh_sq"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // Positive term `a·cosh²(u)` (no `-1` factor in the coefficient).
        for (ci, &cosh_term) in args.iter().enumerate() {
            let Some((u, cosh_coeff)) = split_trig_sq("cosh", cosh_term, pool) else {
                continue;
            };
            if coeff_has_neg_one(&cosh_coeff, pool) {
                continue;
            }
            // Negative term `(-1)·a·sinh²(u)` with the same `u` and magnitude `a`.
            for (si, &sinh_term) in args.iter().enumerate() {
                if si == ci {
                    continue;
                }
                let Some((su, sinh_coeff)) = split_trig_sq("sinh", sinh_term, pool) else {
                    continue;
                };
                if su != u {
                    continue;
                }
                let Some(rest) = strip_one_neg_one(&sinh_coeff, pool) else {
                    continue;
                };
                if !coeff_multiset_eq(&cosh_coeff, &rest, pool) {
                    continue;
                }

                let coeff_expr = match cosh_coeff.len() {
                    0 => pool.integer(1_i32),
                    1 => cosh_coeff[0],
                    _ => pool.mul(cosh_coeff.clone()),
                };

                let mut new_args: Vec<ExprId> = args
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != ci && i != si)
                    .map(|(_, &a)| a)
                    .collect();
                new_args.push(coeff_expr);
                new_args = fold_numeric_terms(new_args, pool);

                let after = match new_args.len() {
                    0 => pool.integer(0_i32),
                    1 => new_args[0],
                    _ => pool.add(new_args),
                };
                return Some((after, one_step(self.name(), expr, after)));
            }
        }
        None
    }
}

/// Multi-angle Pythagorean identity: `c·sin²(u) + c·cos²(u) → c` where the
/// shared coefficient `c` may itself contain *other* trig-squared factors with
/// **different** arguments.
///
/// This generalizes [`SinCosIdentity`], whose `split_trig_sq` bails as soon as
/// a term holds more than one trig-squared factor.  When two angles interleave —
/// as in a direction-cosine matrix — entries look like
///
/// ```text
/// cos²(θ)·sin²(φ) + cos²(θ)·cos²(φ)
/// ```
///
/// Here each term carries *two* trig-squared factors (`cos²(θ)` and the
/// `sin²/cos²(φ)` pair).  Keyed on the inner angle `φ`, the matching `cos²(θ)`
/// is just part of the coefficient, so the pair collapses to `cos²(θ)`.  Fed
/// back through the same rule (now with `sin²(θ) + cos²(θ)`) the whole `RᵀR`
/// diagonal entry reaches `1`.
///
/// The coefficient match is a multiset comparison, so factor order is
/// irrelevant and a single `sin²`/`cos²(u)` factor per term is required (terms
/// with two `sin²(u)` for the *same* `u` are left untouched).
pub struct PythagoreanMultiAngle;

impl RewriteRule for PythagoreanMultiAngle {
    fn name(&self) -> &'static str {
        "pythagorean_multi_angle"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };
        if args.len() < 2 {
            return None;
        }

        // For each term, enumerate every angle `u` that appears as a sin²(u)
        // factor, and try to find a partner cos²(u) term with the same
        // coefficient multiset (where the partner's coefficient excludes its
        // own cos²(u) factor).
        for (sin_i, &sin_term) in args.iter().enumerate() {
            for u in trig_sq_angles("sin", sin_term, pool) {
                let Some(sin_coeff) = split_trig_sq_for_angle("sin", u, sin_term, pool) else {
                    continue;
                };
                for (cos_i, &cos_term) in args.iter().enumerate() {
                    if cos_i == sin_i {
                        continue;
                    }
                    let Some(cos_coeff) = split_trig_sq_for_angle("cos", u, cos_term, pool) else {
                        continue;
                    };
                    if !coeff_multiset_eq(&sin_coeff, &cos_coeff, pool) {
                        continue;
                    }
                    // Collapse: replace both terms with the shared coefficient.
                    let coeff_expr = match sin_coeff.len() {
                        0 => pool.integer(1_i32),
                        1 => sin_coeff[0],
                        _ => pool.mul(sin_coeff.clone()),
                    };
                    let mut new_args: Vec<ExprId> = args
                        .iter()
                        .enumerate()
                        .filter(|&(k, _)| k != sin_i && k != cos_i)
                        .map(|(_, &a)| a)
                        .collect();
                    new_args.push(coeff_expr);
                    new_args = fold_numeric_terms(new_args, pool);
                    let after = match new_args.len() {
                        0 => pool.integer(0_i32),
                        1 => new_args[0],
                        _ => pool.add(new_args),
                    };
                    if after == expr {
                        return None;
                    }
                    return Some((after, one_step(self.name(), expr, after)));
                }
            }
        }
        None
    }
}

/// Double-angle for sine: `2·sin(u)·cos(u) → sin(2u)`.
///
/// Fires on a `Mul` containing the literal factor `2`, a `sin(u)` and a
/// `cos(u)` with the *same* argument `u`.  Any further factors are preserved,
/// so `k·2·sin(u)·cos(u) → k·sin(2u)`.
pub struct SinDoubleAngle;

impl RewriteRule for SinDoubleAngle {
    fn name(&self) -> &'static str {
        "sin_double_angle"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let factors = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };

        let two_pos = factors
            .iter()
            .position(|&f| pool.with(f, |d| matches!(d, ExprData::Integer(n) if n.0 == 2)))?;
        let sin_pos = factors
            .iter()
            .position(|&f| func_arg("sin", f, pool).is_some())?;
        let u = func_arg("sin", factors[sin_pos], pool).unwrap();
        let cos_pos = factors
            .iter()
            .enumerate()
            .position(|(i, &f)| i != sin_pos && func_arg("cos", f, pool) == Some(u))?;

        if two_pos == sin_pos || two_pos == cos_pos {
            return None;
        }

        let two = pool.integer(2_i32);
        let double_u = pool.mul(vec![two, u]);
        let sin_2u = pool.func("sin", vec![double_u]);

        let mut rest: Vec<ExprId> = factors
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != two_pos && i != sin_pos && i != cos_pos)
            .map(|(_, f)| f)
            .collect();
        rest.push(sin_2u);
        let after = match rest.len() {
            1 => rest[0],
            _ => pool.mul(rest),
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Double-angle for cosine: `cos²(u) − sin²(u) → cos(2u)`.
///
/// Fires on an `Add` containing `cos²(u)` and `(-1)·sin²(u)` with the same
/// argument `u`.  Remaining terms are preserved.
pub struct CosDoubleAngle;

impl RewriteRule for CosDoubleAngle {
    fn name(&self) -> &'static str {
        "cos_double_angle"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // `cos²(u)` term (coefficient must be +1, i.e. no leftover factors).
        let mut cos_hit = None;
        for (i, &a) in args.iter().enumerate() {
            if let Some((u, coeff)) = split_trig_sq("cos", a, pool) {
                if coeff.is_empty() {
                    cos_hit = Some((i, u));
                    break;
                }
            }
        }
        let (cos_idx, u) = cos_hit?;

        // `(-1)·sin²(u)` term.
        let mut sin_idx = None;
        for (i, &a) in args.iter().enumerate() {
            if i == cos_idx {
                continue;
            }
            if let Some((su, coeff)) = split_trig_sq("sin", a, pool) {
                if su == u
                    && coeff.len() == 1
                    && pool.with(coeff[0], |d| matches!(d, ExprData::Integer(n) if n.0 == -1))
                {
                    sin_idx = Some(i);
                    break;
                }
            }
        }
        let sin_idx = sin_idx?;

        let two = pool.integer(2_i32);
        let double_u = pool.mul(vec![two, u]);
        let cos_2u = pool.func("cos", vec![double_u]);

        let mut rest: Vec<ExprId> = args
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != cos_idx && i != sin_idx)
            .map(|(_, a)| a)
            .collect();
        rest.push(cos_2u);
        let after = match rest.len() {
            1 => rest[0],
            _ => pool.add(rest),
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Angle-subtraction for sine: `c·sin(a)·cos(b) − c·cos(a)·sin(b) → c·sin(a−b)`.
///
/// Fires on an `Add` containing a positive product `c·sin(a)·cos(b)` and a
/// negative product `(-1)·c·cos(a)·sin(b)` that share the **same** coefficient
/// multiset `c` (numeric or symbolic, possibly empty — the bare
/// `sin(a)cos(b) − cos(a)sin(b)` case).  The matched pair is replaced by
/// `c·sin(a−b)`; unrelated terms are preserved.
///
/// The coefficient match is the angle-identity analogue of the coefficient-aware
/// Pythagorean rule: it lets the 2-link Jacobian determinant
/// `l1·l2·cos(θ1)·sin(θ1+θ2) − l1·l2·sin(θ1)·cos(θ1+θ2)` collapse to
/// `l1·l2·sin(θ2)` even though `l1·l2` is shared across both terms.
pub struct SinAngleSub;

impl RewriteRule for SinAngleSub {
    fn name(&self) -> &'static str {
        "sin_angle_sub"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // Positive term `c·sin(a)·cos(b)` (coefficient must be sign-positive,
        // i.e. carry no `-1` factor).
        for (pi, &pos) in args.iter().enumerate() {
            let Some((a, b, pos_coeff)) = split_trig_pair("sin", "cos", pos, pool) else {
                continue;
            };
            if coeff_has_neg_one(&pos_coeff, pool) {
                continue;
            }
            // Negative term `(-1)·c·cos(a)·sin(b)` with the same (a, b) and the
            // same coefficient multiset `c`.
            for (ni, &neg) in args.iter().enumerate() {
                if ni == pi {
                    continue;
                }
                let Some((na, nb, neg_coeff)) = split_trig_pair("cos", "sin", neg, pool) else {
                    continue;
                };
                if (na, nb) != (a, b) {
                    continue;
                }
                let Some(rest) = strip_one_neg_one(&neg_coeff, pool) else {
                    continue;
                };
                if !coeff_multiset_eq(&pos_coeff, &rest, pool) {
                    continue;
                }
                let diff = sub(a, b, pool);
                let sin_diff = pool.func("sin", vec![diff]);
                let replacement = attach_coeff(&pos_coeff, sin_diff, pool);
                let after = rebuild_add_replacing(&args, pi, ni, replacement, pool);
                return Some((after, one_step(self.name(), expr, after)));
            }
        }
        None
    }
}

/// Angle-subtraction for cosine: `c·cos(a)·cos(b) + c·sin(a)·sin(b) → c·cos(a−b)`.
///
/// Coefficient-aware in the same way as [`SinAngleSub`]: both products must
/// share the same coefficient multiset `c` (numeric or symbolic, possibly
/// empty).
pub struct CosAngleSub;

impl RewriteRule for CosAngleSub {
    fn name(&self) -> &'static str {
        "cos_angle_sub"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        for (ci, &cc) in args.iter().enumerate() {
            let Some((a, b, cos_coeff)) = split_trig_pair("cos", "cos", cc, pool) else {
                continue;
            };
            if coeff_has_neg_one(&cos_coeff, pool) {
                continue;
            }
            for (si, &ss) in args.iter().enumerate() {
                if si == ci {
                    continue;
                }
                // `c·sin(a)·sin(b)` with the same (a, b) — order-insensitive —
                // and the same coefficient multiset.
                let Some((sa, sb, sin_coeff)) = split_trig_pair("sin", "sin", ss, pool) else {
                    continue;
                };
                if !((sa == a && sb == b) || (sa == b && sb == a)) {
                    continue;
                }
                if !coeff_multiset_eq(&cos_coeff, &sin_coeff, pool) {
                    continue;
                }
                let diff = sub(a, b, pool);
                let cos_diff = pool.func("cos", vec![diff]);
                let replacement = attach_coeff(&cos_coeff, cos_diff, pool);
                let after = rebuild_add_replacing(&args, ci, si, replacement, pool);
                return Some((after, one_step(self.name(), expr, after)));
            }
        }
        None
    }
}

/// Return all trigonometric identity rules.
///
/// The set leads with the structural normalizers [`FlattenMul`]/[`FlattenAdd`]
/// so the AC-sensitive identity rules below (Pythagorean, double-angle,
/// angle-subtraction) see fully flattened `Add`/`Mul` nodes even when the input
/// arrives as nested binary trees (as it does from the Python surface, which
/// builds `a*b*c` as `a*(b*c)`).  Both normalizers only restructure nested
/// `Add`/`Mul`; they perform no arithmetic and so cannot introduce regressions.
pub fn trig_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![
        Box::new(FlattenMul),
        Box::new(FlattenAdd),
        Box::new(SinNeg),
        Box::new(CosNeg),
        Box::new(TanExpand),
        Box::new(SinCosIdentity),
        Box::new(CoshSinhIdentity),
        Box::new(SinDoubleAngle),
        Box::new(CosDoubleAngle),
        Box::new(SinAngleSub),
        Box::new(CosAngleSub),
    ]
}

/// Trigonometric **normal-form** rule set: the full algebraic core *with
/// bounded polynomial expansion* (`ExpandPow` + `ExpandMul`) plus the
/// sin/cos-polynomial trig identities — argument-sign normalization
/// ([`SinNeg`]/[`CosNeg`]) and the Pythagorean identities
/// ([`SinCosIdentity`] and its multi-angle generalization
/// [`PythagoreanMultiAngle`]).
///
/// Unlike [`trig_rules`] — which carries only the trig identities and so cannot
/// even multiply out a product of rotations — this bundle composes
/// expansion → constant folding → like-term collection → Pythagorean
/// reduction into a single fixed-point run.  It is what
/// [`simplify_trig_normal_form`](super::engine::simplify_trig_normal_form) uses
/// to collapse `Rᵀ·R − I → 0` for a direction-cosine matrix in one call.
///
/// # Why no double-angle / angle-sum rules
///
/// The reduction works by driving everything into a canonical **sin/cos
/// monomial polynomial** where like terms cancel.  Folding rules such as
/// `2·sin·cos → sin(2u)` or `sin·cos ± cos·sin → sin(u±v)`
/// ([`SinDoubleAngle`], [`CosDoubleAngle`], [`SinAngleSub`], [`CosAngleSub`])
/// are deliberately **excluded**: they pull terms *out* of that monomial basis
/// into compound-angle functions, and because they only fire when the
/// coefficient is isolable, they collapse one term of a cancelling pair but not
/// its partner — leaving a non-zero `sin(2u)·X − 2·sin·cos·X` residue that can
/// never close.  Pythagorean reduction in the pure monomial basis is both
/// complete for orthogonality probes and cheap.
///
/// # Scope
///
/// The expansion rules are bounded (`ExpandPow` caps the literal exponent it
/// unfolds and distributes straight to a flat sum, so it never oscillates with
/// the factor-collecting rule), so this set is heavier than the default
/// simplifier and is **opt-in** — it is never wired into
/// [`simplify`](super::engine::simplify).  It targets real-argument sin/cos
/// polynomials (DCM / rotation entries); it is not a complete decision
/// procedure for arbitrary trigonometric expressions and does not introduce
/// compound-angle (`sin(2u)`, `sin(u+v)`, …) forms.
pub fn trig_normal_form_rules() -> Vec<Box<dyn RewriteRule>> {
    let cfg = super::engine::SimplifyConfig {
        expand: true,
        ..Default::default()
    };
    let mut rules = super::engine::rules_for_config(&cfg);
    // Append the sin/cos-polynomial identities only.  Compound-angle folding
    // rules are intentionally omitted — see the doc comment above.
    rules.push(Box::new(SinNeg));
    rules.push(Box::new(CosNeg));
    rules.push(Box::new(SinCosIdentity));
    rules.push(Box::new(PythagoreanMultiAngle));
    rules
}

// ---------------------------------------------------------------------------
// log / exp identity rules
// ---------------------------------------------------------------------------

/// `log(exp(x)) → x`.
pub struct LogOfExp;

impl RewriteRule for LogOfExp {
    fn name(&self) -> &'static str {
        "log_of_exp"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("log", expr, pool)?;
        let inner = func_arg("exp", arg, pool)?;
        Some((inner, one_step(self.name(), expr, inner)))
    }
}

/// `exp(log(x)) → x` (domain: x > 0 assumed).
pub struct ExpOfLog;

impl RewriteRule for ExpOfLog {
    fn name(&self) -> &'static str {
        "exp_of_log"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("exp", expr, pool)?;
        let inner = func_arg("log", arg, pool)?;
        Some((inner, one_step(self.name(), expr, inner)))
    }
}

/// `log(a * b) → log(a) + log(b)`.
///
/// **Branch-cut caveat**: this identity is only valid when all factors are
/// positive reals.  The rule still fires, but each factor is recorded as a
/// [`SideCondition::Positive`] in the derivation log so callers can audit the
/// assumptions made.  Use [`log_exp_rules_safe`] to obtain a rule set that
/// excludes this rule entirely.
pub struct LogOfProduct;

impl RewriteRule for LogOfProduct {
    fn name(&self) -> &'static str {
        "log_of_product"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("log", expr, pool)?;
        let factors = match pool.get(arg) {
            ExprData::Mul(v) if v.len() >= 2 => v,
            _ => return None,
        };
        let logs: Vec<ExprId> = factors.iter().map(|&f| pool.func("log", vec![f])).collect();
        let after = pool.add(logs);
        let conds: Vec<SideCondition> = factors
            .iter()
            .map(|&f| SideCondition::Positive(f))
            .collect();
        let mut log = DerivationLog::new();
        log.push(RewriteStep::with_conditions(
            "log_of_product",
            expr,
            after,
            conds,
        ));
        Some((after, log))
    }
}

/// `log(a^n) → n * log(a)`.
pub struct LogOfPow;

impl RewriteRule for LogOfPow {
    fn name(&self) -> &'static str {
        "log_of_pow"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("log", expr, pool)?;
        let (base, exp) = match pool.get(arg) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        let log_base = pool.func("log", vec![base]);
        let after = pool.mul(vec![exp, log_base]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Return the conservative log/exp identity rules.
///
/// Symbolic log/exp identities need real-domain or positivity facts. Use an
/// [`crate::simplify::AssumptionContext`] to enable the condition-gated
/// identities; this standalone ruleset intentionally keeps no such facts.
pub fn log_exp_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![]
}

/// Log/exp rules that are safe for complex numbers (no branch-cut rewrites).
///
/// No nontrivial symbolic log/exp identity is valid over all principal-complex
/// branches, so this is intentionally empty.
pub fn log_exp_rules_safe() -> Vec<Box<dyn RewriteRule>> {
    vec![]
}

// ---------------------------------------------------------------------------
// R-5: Pattern-driven user rewrite rules
// ---------------------------------------------------------------------------

/// A rewrite rule specified as a (lhs pattern, rhs template) pair.
///
/// When the rule fires, all wildcards bound by matching `lhs` against the
/// current expression are substituted into `rhs`.
///
/// # Wildcard convention
///
/// Any `Symbol` whose name starts with a lower-case letter is a wildcard.
///
/// # Example
///
/// ```
/// # use alkahest_cas::kernel::{Domain, ExprPool};
/// # use alkahest_cas::simplify::{simplify_with, SimplifyConfig};
/// # use alkahest_cas::simplify::rulesets::PatternRule;
/// # use alkahest_cas::pattern::Pattern;
/// # use alkahest_cas::simplify::rules::RewriteRule;
/// let pool = ExprPool::new();
/// let a = pool.symbol("a", Domain::Real);  // wildcard
/// let b = pool.symbol("b", Domain::Real);  // wildcard
/// // Rule: a*b + a*c → a*(b+c)  (factoring)
/// // lhs pattern: a*b  (simplified, as a*b is the structure)
/// // Here we demonstrate a simpler identity: a + a → 2*a
/// let lhs = pool.add(vec![a, a]);
/// let two_a = pool.mul(vec![pool.integer(2_i32), a]);
/// let rule = PatternRule::new(Pattern::from_expr(lhs), two_a);
/// let x = pool.symbol("x", Domain::Real);
/// let expr = pool.add(vec![x, x]);
/// let r = simplify_with(expr, &pool, &[Box::new(rule)], SimplifyConfig::default());
/// // x + x → 2*x
/// ```
#[derive(Clone)]
pub struct PatternRule {
    pub lhs: Pattern,
    pub rhs: ExprId,
    name: &'static str,
}

/// Pattern rules plus a discrimination-net index for O(1) head lookup.
pub struct PatternRuleSet {
    rules: Vec<PatternRule>,
    index: DiscriminationIndex,
}

impl PatternRuleSet {
    pub fn new(rules: Vec<PatternRule>, pool: &ExprPool) -> Self {
        let heads = rules.iter().map(|r| pattern_head(r.lhs.root, pool));
        let index = DiscriminationIndex::build(heads);
        PatternRuleSet { rules, index }
    }

    pub fn rules(&self) -> &[PatternRule] {
        &self.rules
    }

    pub fn index(&self) -> &DiscriminationIndex {
        &self.index
    }

    pub fn as_dyn_rules(&self) -> Vec<Box<dyn RewriteRule>> {
        self.rules
            .iter()
            .map(|r| Box::new(r.clone()) as Box<dyn RewriteRule>)
            .collect()
    }
}

impl PatternRule {
    pub fn new(lhs: Pattern, rhs: ExprId) -> Self {
        PatternRule {
            lhs,
            rhs,
            name: "pattern_rule",
        }
    }

    pub fn named(lhs: Pattern, rhs: ExprId, name: &'static str) -> Self {
        PatternRule { lhs, rhs, name }
    }
}

impl RewriteRule for PatternRule {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        // Try to match the pattern at the root only (engine does bottom-up traversal)
        let subst = match_at_root(&self.lhs, expr, pool)?;
        let after = subst.apply(self.rhs, pool);
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name, expr, after)))
    }
}

/// Match `pattern` at the root of `expr` (no recursion into children).
fn match_at_root(pattern: &Pattern, expr: ExprId, pool: &ExprPool) -> Option<Substitution> {
    let empty = Substitution {
        bindings: std::collections::HashMap::new(),
    };
    match_root_node(pattern.root, expr, empty, pool)
}

fn match_root_node(
    pat: ExprId,
    expr: ExprId,
    subst: Substitution,
    pool: &ExprPool,
) -> Option<Substitution> {
    use crate::kernel::expr::ExprData as ED;

    enum PN {
        Wildcard(String),
        Integer(i64),
        Symbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Literal,
    }
    enum EN {
        Integer(i64),
        Symbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Other,
    }

    let pn = pool.with(pat, |d| match d {
        ED::Symbol { name, .. } if name.starts_with(|c: char| c.is_lowercase()) => {
            PN::Wildcard(name.clone())
        }
        ED::Symbol { name, .. } => PN::Symbol(name.clone()),
        ED::Integer(n) => PN::Integer(n.0.to_i64().unwrap_or(i64::MIN)),
        ED::Add(v) => PN::Add(v.clone()),
        ED::Mul(v) => PN::Mul(v.clone()),
        ED::Pow { base, exp } => PN::Pow(*base, *exp),
        ED::Func { name, args } => PN::Func(name.clone(), args.clone()),
        _ => PN::Literal,
    });

    let en = pool.with(expr, |d| match d {
        ED::Symbol { name, .. } => EN::Symbol(name.clone()),
        ED::Integer(n) => EN::Integer(n.0.to_i64().unwrap_or(i64::MIN)),
        ED::Add(v) => EN::Add(v.clone()),
        ED::Mul(v) => EN::Mul(v.clone()),
        ED::Pow { base, exp } => EN::Pow(*base, *exp),
        ED::Func { name, args } => EN::Func(name.clone(), args.clone()),
        _ => EN::Other,
    });

    match pn {
        PN::Wildcard(name) => {
            let mut s = subst;
            match s.bindings.get(&name) {
                Some(&existing) if existing != expr => return None,
                _ => {
                    s.bindings.insert(name, expr);
                }
            }
            Some(s)
        }
        PN::Integer(pv) => {
            if matches!(en, EN::Integer(ev) if ev == pv) {
                Some(subst)
            } else {
                None
            }
        }
        PN::Symbol(pname) => {
            if matches!(en, EN::Symbol(ref ename) if *ename == pname) {
                Some(subst)
            } else {
                None
            }
        }
        PN::Add(pargs) => {
            let EN::Add(eargs) = en else { return None };
            match_args_exact(&pargs, &eargs, subst, pool)
        }
        PN::Mul(pargs) => {
            let EN::Mul(eargs) = en else { return None };
            match_args_exact(&pargs, &eargs, subst, pool)
        }
        PN::Pow(pb, pe) => {
            let EN::Pow(eb, ee) = en else { return None };
            let s = match_root_node(pb, eb, subst, pool)?;
            match_root_node(pe, ee, s, pool)
        }
        PN::Func(pname, pargs) => {
            let EN::Func(ename, eargs) = en else {
                return None;
            };
            if pname != ename {
                return None;
            }
            match_args_exact(&pargs, &eargs, subst, pool)
        }
        PN::Literal => {
            if pat == expr {
                Some(subst)
            } else {
                None
            }
        }
    }
}

fn match_args_exact(
    pat_args: &[ExprId],
    expr_args: &[ExprId],
    subst: Substitution,
    pool: &ExprPool,
) -> Option<Substitution> {
    if pat_args.len() != expr_args.len() {
        return None;
    }
    let mut s = subst;
    for (&p, &e) in pat_args.iter().zip(expr_args.iter()) {
        s = match_root_node(p, e, s, pool)?;
    }
    Some(s)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn func_arg(name: &str, expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    pool.with(expr, |data| match data {
        ExprData::Func { name: n, args } if n == name && args.len() == 1 => Some(args[0]),
        _ => None,
    })
}

/// If `expr` is `(-1) * inner` or `inner * (-1)`, return `inner`.
fn neg_inner(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let args = match pool.get(expr) {
        ExprData::Mul(v) => v,
        _ => return None,
    };
    let neg1_pos = args
        .iter()
        .position(|&a| pool.with(a, |d| matches!(d, ExprData::Integer(n) if n.0 == -1)))?;
    let others: Vec<ExprId> = args
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| i != neg1_pos)
        .map(|(_, a)| a)
        .collect();
    Some(match others.len() {
        0 => pool.integer(1_i32),
        1 => others[0],
        _ => pool.mul(others),
    })
}

/// If `expr` is `Pow(Func(`name`, [arg]), 2)`, return `arg`.
fn trig_sq_inner(name: &str, expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let is_two = pool.with(exp, |d| matches!(d, ExprData::Integer(n) if n.0 == 2));
            if !is_two {
                return None;
            }
            func_arg(name, base, pool)
        }
        _ => None,
    }
}

/// View a single Add-term as a multiset of multiplicative factors.
///
/// A bare (non-`Mul`) term is treated as a one-element factor list; a `Mul`
/// returns its (already canonically sorted) factor vector.  Used to peel a
/// shared coefficient off a `c · sin²(u)` term.
fn factor_list(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(expr) {
        ExprData::Mul(v) => v,
        _ => vec![expr],
    }
}

/// If exactly one factor of `term` is `Pow(`name`(u), 2)`, return
/// `(u, remaining_factors)` where `remaining_factors` is the coefficient.
///
/// `remaining_factors` may be empty (meaning coefficient `1`).
fn split_trig_sq(name: &str, term: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<ExprId>)> {
    let factors = factor_list(term, pool);
    let mut inner = None;
    let mut rest = Vec::with_capacity(factors.len());
    let mut matched = 0usize;
    for &f in &factors {
        if let Some(u) = trig_sq_inner(name, f, pool) {
            if matched == 0 {
                inner = Some(u);
            }
            matched += 1;
            if matched > 1 {
                // Two trig-squared factors in one term — ambiguous; bail.
                return None;
            }
        } else {
            rest.push(f);
        }
    }
    inner.map(|u| (u, rest))
}

/// List the distinct angles `u` for which `term` contains a `name²(u)` factor,
/// in canonical factor order, de-duplicated.
fn trig_sq_angles(name: &str, term: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut angles = Vec::new();
    for &f in &factor_list(term, pool) {
        if let Some(u) = trig_sq_inner(name, f, pool) {
            if !angles.contains(&u) {
                angles.push(u);
            }
        }
    }
    angles
}

/// Like [`split_trig_sq`] but targets a **specific** angle `u`.
///
/// Returns the coefficient factor list (everything except the single
/// `name²(u)` factor) when the term contains exactly one `name²(u)` factor for
/// that *specific* `u`.  Unlike [`split_trig_sq`], other trig-squared factors
/// with *different* arguments (e.g. `cos²(θ)` inside `cos²(θ)·cos²(φ)`) are
/// tolerated — they become part of the coefficient.  This is what lets the
/// multi-angle Pythagorean rule collapse
/// `cos²(θ)·sin²(φ) + cos²(θ)·cos²(φ) → cos²(θ)` even though each term carries a
/// second trig-squared factor.
fn split_trig_sq_for_angle(
    name: &str,
    u: ExprId,
    term: ExprId,
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    let factors = factor_list(term, pool);
    let mut rest = Vec::with_capacity(factors.len());
    let mut matched = 0usize;
    for &f in &factors {
        if trig_sq_inner(name, f, pool) == Some(u) {
            matched += 1;
            if matched > 1 {
                // Two `name²(u)` factors for the same angle — ambiguous; bail.
                return None;
            }
        } else {
            rest.push(f);
        }
    }
    if matched == 1 {
        Some(rest)
    } else {
        None
    }
}

/// Split an `Add`-term `c · f(a) · g(b)` into the trig argument pair plus the
/// surrounding (possibly empty) coefficient factor list.
///
/// Allows an arbitrary number of *extra*
/// multiplicative factors (a shared coefficient, numeric or symbolic) around
/// exactly one `f(·)` and exactly one `g(·)`:
///
/// ```text
/// l1·l2·cos(θ1)·sin(θ1+θ2)  →  (θ1, θ1+θ2, [l1, l2])   for f=cos, g=sin
/// sin(a)·cos(b)             →  (a,  b,     [])
/// ```
///
/// Returns `(a, b, coeff)` where `a` is the argument of the `f`-named factor and
/// `b` that of the `g`-named factor.  When `f_name == g_name` the two arguments
/// are returned in the canonical factor order in which they appear.  Any factor
/// that is itself an `f`/`g` application beyond the first match makes the term
/// ambiguous and yields `None`.  The coefficient list is whatever remains; it is
/// returned in the term's canonical factor order so two terms sharing the same
/// coefficient multiset compare equal.
fn split_trig_pair(
    f_name: &str,
    g_name: &str,
    term: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId, Vec<ExprId>)> {
    let factors = factor_list(term, pool);
    let mut a = None;
    let mut b = None;
    let mut coeff = Vec::with_capacity(factors.len());

    if f_name == g_name {
        // Need exactly two `f(·)` factors; everything else is coefficient.
        for &f in &factors {
            if let Some(arg) = func_arg(f_name, f, pool) {
                if a.is_none() {
                    a = Some(arg);
                } else if b.is_none() {
                    b = Some(arg);
                } else {
                    return None; // three+ matching factors — ambiguous
                }
            } else {
                coeff.push(f);
            }
        }
        return Some((a?, b?, coeff));
    }

    for &f in &factors {
        if a.is_none() {
            if let Some(arg) = func_arg(f_name, f, pool) {
                a = Some(arg);
                continue;
            }
        } else if func_arg(f_name, f, pool).is_some() {
            return None; // second `f` factor — ambiguous
        }
        if b.is_none() {
            if let Some(arg) = func_arg(g_name, f, pool) {
                b = Some(arg);
                continue;
            }
        } else if func_arg(g_name, f, pool).is_some() {
            return None; // second `g` factor — ambiguous
        }
        coeff.push(f);
    }
    Some((a?, b?, coeff))
}

/// Does the coefficient factor list contain a literal `-1`?
fn coeff_has_neg_one(coeff: &[ExprId], pool: &ExprPool) -> bool {
    coeff
        .iter()
        .any(|&f| pool.with(f, |d| matches!(d, ExprData::Integer(n) if n.0 == -1)))
}

/// Remove exactly one literal `-1` factor from `coeff`, returning the rest.
/// `None` if there is no `-1` factor (so the term is not sign-negative).
fn strip_one_neg_one(coeff: &[ExprId], pool: &ExprPool) -> Option<Vec<ExprId>> {
    let pos = coeff
        .iter()
        .position(|&f| pool.with(f, |d| matches!(d, ExprData::Integer(n) if n.0 == -1)))?;
    let rest: Vec<ExprId> = coeff
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pos)
        .map(|(_, &f)| f)
        .collect();
    Some(rest)
}

/// Compare two coefficient factor lists as multisets (order-independent).
fn coeff_multiset_eq(a: &[ExprId], b: &[ExprId], _pool: &ExprPool) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_sorted: Vec<ExprId> = a.to_vec();
    let mut b_sorted: Vec<ExprId> = b.to_vec();
    a_sorted.sort_unstable();
    b_sorted.sort_unstable();
    a_sorted == b_sorted
}

/// Multiply `inner` by the coefficient factors (empty → `inner` unchanged).
fn attach_coeff(coeff: &[ExprId], inner: ExprId, pool: &ExprPool) -> ExprId {
    match coeff.len() {
        0 => inner,
        _ => {
            let mut factors = coeff.to_vec();
            factors.push(inner);
            pool.mul(factors)
        }
    }
}

/// Build `a − b` as `Add([a, (-1)·b])`.
fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let neg_one = pool.integer(-1_i32);
    let neg_b = pool.mul(vec![neg_one, b]);
    pool.add(vec![a, neg_b])
}

/// Rebuild an `Add` from `args`, dropping positions `i` and `j` and appending
/// `replacement`.
fn rebuild_add_replacing(
    args: &[ExprId],
    i: usize,
    j: usize,
    replacement: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let mut rest: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != i && k != j)
        .map(|(_, &a)| a)
        .collect();
    rest.push(replacement);
    match rest.len() {
        1 => rest[0],
        _ => pool.add(rest),
    }
}

/// Numeric value of `expr` as an exact `rug::Rational`, if it is an integer
/// or rational literal.  `Float` is intentionally excluded (folding floats
/// would change the result's exactness).
fn as_exact_rational(expr: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0)),
        ExprData::Rational(r) => Some(r.0),
        _ => None,
    }
}

/// Intern an exact rational, collapsing `n/1` to an `Integer`.
fn intern_rational(value: rug::Rational, pool: &ExprPool) -> ExprId {
    if value.denom() == &rug::Integer::from(1) {
        pool.integer(value.numer().clone())
    } else {
        let (num, den) = value.into_numer_denom();
        pool.rational(num, den)
    }
}

/// Combine any exact numeric literals appearing as top-level terms of an `Add`
/// argument list into a single literal, leaving non-numeric terms untouched.
///
/// Returns the rebuilt term list.  This lets the coefficient-aware Pythagorean
/// rule reduce `3 + 2` (produced after collapsing `2·sin²+2·cos² → 2`) to `5`
/// even though the bare `trig_rules` set has no general constant-folding rule.
fn fold_numeric_terms(terms: Vec<ExprId>, pool: &ExprPool) -> Vec<ExprId> {
    let mut acc: Option<rug::Rational> = None;
    let mut others = Vec::with_capacity(terms.len());
    for t in terms {
        if let Some(r) = as_exact_rational(t, pool) {
            acc = Some(match acc {
                Some(a) => a + r,
                None => r,
            });
        } else {
            others.push(t);
        }
    }
    if let Some(sum) = acc {
        // Drop an exact zero so it does not survive as `… + 0`.
        if sum.cmp0() != std::cmp::Ordering::Equal || others.is_empty() {
            others.push(intern_rational(sum, pool));
        }
    }
    others
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::pattern::Pattern;
    use crate::simplify::engine::{simplify_with, simplify_with_pattern_rules, SimplifyConfig};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn sin_neg_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.func("sin", vec![neg_x]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        // sin(-x) → -sin(x)
        let expected = pool.mul(vec![pool.integer(-1_i32), pool.func("sin", vec![x])]);
        assert_eq!(r.value, expected);
    }

    #[test]
    fn cos_neg_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.func("cos", vec![neg_x]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, pool.func("cos", vec![x]));
    }

    #[test]
    fn tan_expand_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("tan", vec![x]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let cos_inv = pool.pow(cos_x, pool.integer(-1_i32));
        let expected = pool.mul(vec![sin_x, cos_inv]);
        assert_eq!(r.value, expected);
    }

    #[test]
    fn sin_cos_identity_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let two = pool.integer(2_i32);
        let sin_sq = pool.pow(sin_x, two);
        let cos_sq = pool.pow(cos_x, two);
        let expr = pool.add(vec![sin_sq, cos_sq]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, pool.integer(1_i32));
    }

    #[test]
    fn cosh_sinh_identity_fires() {
        // cosh²(x) − sinh²(x) → 1
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let cosh_sq = pool.pow(pool.func("cosh", vec![x]), two);
        let sinh_sq = pool.pow(pool.func("sinh", vec![x]), two);
        let neg_sinh_sq = pool.mul(vec![pool.integer(-1_i32), sinh_sq]);
        let expr = pool.add(vec![cosh_sq, neg_sinh_sq]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(
            r.value,
            pool.integer(1_i32),
            "got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn cosh_sinh_identity_coeff() {
        // 2·cosh²(x) − 2·sinh²(x) → 2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let cosh_sq = pool.pow(pool.func("cosh", vec![x]), pool.integer(2_i32));
        let sinh_sq = pool.pow(pool.func("sinh", vec![x]), pool.integer(2_i32));
        let pos = pool.mul(vec![two, cosh_sq]);
        let neg = pool.mul(vec![pool.integer(-1_i32), two, sinh_sq]);
        let expr = pool.add(vec![pos, neg]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(
            r.value,
            pool.integer(2_i32),
            "got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn cosh_sinh_identity_different_angles_untouched() {
        // cosh²(x) − sinh²(y) must not fold when angles differ.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let two = pool.integer(2_i32);
        let cosh_sq = pool.pow(pool.func("cosh", vec![x]), two);
        let sinh_sq = pool.pow(pool.func("sinh", vec![y]), two);
        let neg_sinh_sq = pool.mul(vec![pool.integer(-1_i32), sinh_sq]);
        let expr = pool.add(vec![cosh_sq, neg_sinh_sq]);
        assert!(
            CoshSinhIdentity.apply(expr, &pool).is_none(),
            "must not collapse when hyperbolic angles differ"
        );
    }

    // -------------------------------------------------------------------
    // Multi-angle Pythagorean + trig normal form (RᵀR = I for a DCM)
    // -------------------------------------------------------------------

    /// 3×3 matrix multiply over `ExprId` cells (no simplification).
    fn mat3_mul(pool: &ExprPool, a: &[[ExprId; 3]; 3], b: &[[ExprId; 3]; 3]) -> [[ExprId; 3]; 3] {
        let mut out = [[pool.integer(0_i32); 3]; 3];
        for (i, orow) in out.iter_mut().enumerate() {
            for (j, ocell) in orow.iter_mut().enumerate() {
                let terms: Vec<ExprId> = (0..3).map(|k| pool.mul(vec![a[i][k], b[k][j]])).collect();
                *ocell = pool.add(terms);
            }
        }
        out
    }

    fn mat3_transpose(pool: &ExprPool, a: &[[ExprId; 3]; 3]) -> [[ExprId; 3]; 3] {
        let mut out = [[pool.integer(0_i32); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[j][i];
            }
        }
        out
    }

    /// Build a 3-2-1 (yaw-pitch-roll) direction-cosine matrix
    /// `R = Rx(φ)·Ry(θ)·Rz(ψ)` with the given Euler-angle symbols.
    fn dcm_321(pool: &ExprPool, phi: ExprId, theta: ExprId, psi: ExprId) -> [[ExprId; 3]; 3] {
        let s = |a: ExprId| pool.func("sin", vec![a]);
        let c = |a: ExprId| pool.func("cos", vec![a]);
        let neg = |a: ExprId| pool.mul(vec![pool.integer(-1_i32), a]);
        let zero = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let rz = [
            [c(psi), s(psi), zero],
            [neg(s(psi)), c(psi), zero],
            [zero, zero, one],
        ];
        let ry = [
            [c(theta), zero, neg(s(theta))],
            [zero, one, zero],
            [s(theta), zero, c(theta)],
        ];
        let rx = [
            [one, zero, zero],
            [zero, c(phi), s(phi)],
            [zero, neg(s(phi)), c(phi)],
        ];
        let rxy = mat3_mul(pool, &rx, &ry);
        mat3_mul(pool, &rxy, &rz)
    }

    #[test]
    fn dcm_rtr_minus_identity_is_zero() {
        // For a 3-2-1 Euler-angle direction-cosine matrix R, every entry of
        // Rᵀ·R − I must collapse to 0 under a single `simplify_trig_normal_form`
        // call.  This is the headline orthogonality probe: it requires
        // expansion, constant folding, like-term collection, and the
        // multi-angle Pythagorean identity to all compose in one pass.
        let pool = p();
        let phi = pool.symbol("phi", Domain::Real);
        let theta = pool.symbol("theta", Domain::Real);
        let psi = pool.symbol("psi", Domain::Real);
        let r = dcm_321(&pool, phi, theta, psi);
        let rt = mat3_transpose(&pool, &r);
        let rtr = mat3_mul(&pool, &rt, &r);
        let zero = pool.integer(0_i32);
        for (i, row) in rtr.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                let identity = if i == j {
                    pool.integer(1_i32)
                } else {
                    pool.integer(0_i32)
                };
                // entry − identity
                let neg_id = pool.mul(vec![pool.integer(-1_i32), identity]);
                let diff = pool.add(vec![entry, neg_id]);
                let res = crate::simplify::simplify_trig_normal_form(diff, &pool);
                assert_eq!(
                    res.value,
                    zero,
                    "RᵀR − I entry [{}][{}] did not collapse to 0: got {}",
                    i,
                    j,
                    pool.display(res.value)
                );
            }
        }
    }

    #[test]
    fn dcm_rtr_diagonal_entry_is_one() {
        // A direct check that a diagonal entry of Rᵀ·R itself reduces to 1.
        let pool = p();
        let phi = pool.symbol("phi", Domain::Real);
        let theta = pool.symbol("theta", Domain::Real);
        let psi = pool.symbol("psi", Domain::Real);
        let r = dcm_321(&pool, phi, theta, psi);
        let rt = mat3_transpose(&pool, &r);
        let rtr = mat3_mul(&pool, &rt, &r);
        let res = crate::simplify::simplify_trig_normal_form(rtr[2][2], &pool);
        assert_eq!(
            res.value,
            pool.integer(1_i32),
            "got {}",
            pool.display(res.value)
        );
    }

    #[test]
    fn non_orthogonal_product_does_not_collapse() {
        // Guard: a genuinely non-identity product must NOT be simplified to 0.
        // Take M = R·R (not Rᵀ·R) for a 3-2-1 DCM; MᵀM is still I (R·R is a
        // rotation), so instead scale one rotation to break orthogonality.
        //
        // Concretely, perturb R by replacing R with 2·R (no longer a rotation);
        // then (2R)ᵀ(2R) = 4·I, whose [0][0] entry − 1 = 3 ≠ 0, and whose
        // off-diagonal [0][1] entry is 0.  We assert the diagonal does NOT
        // collapse to 0 and in fact reduces to the correct nonzero constant 3.
        let pool = p();
        let phi = pool.symbol("phi", Domain::Real);
        let theta = pool.symbol("theta", Domain::Real);
        let psi = pool.symbol("psi", Domain::Real);
        let r = dcm_321(&pool, phi, theta, psi);
        let two = pool.integer(2_i32);
        // scale every entry of R by 2
        let mut m = [[pool.integer(0_i32); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = pool.mul(vec![two, r[i][j]]);
            }
        }
        let mt = mat3_transpose(&pool, &m);
        let mtm = mat3_mul(&pool, &mt, &m);
        // (2R)ᵀ(2R) [0][0] − 1 should reduce to 3, never 0.
        let neg_one = pool.integer(-1_i32);
        let diff = pool.add(vec![mtm[0][0], neg_one]);
        let res = crate::simplify::simplify_trig_normal_form(diff, &pool);
        let zero = pool.integer(0_i32);
        assert_ne!(
            res.value, zero,
            "non-orthogonal (2R)ᵀ(2R)−I diagonal must not collapse to 0"
        );
        assert_eq!(
            res.value,
            pool.integer(3_i32),
            "expected (2R)ᵀ(2R)[0][0]−1 = 3, got {}",
            pool.display(res.value)
        );
    }

    #[test]
    fn pythagorean_multi_angle_basic() {
        // cos²(θ)·sin²(φ) + cos²(θ)·cos²(φ) → cos²(θ)
        let pool = p();
        let theta = pool.symbol("theta", Domain::Real);
        let phi = pool.symbol("phi", Domain::Real);
        let two = pool.integer(2_i32);
        let cos_th_sq = pool.pow(pool.func("cos", vec![theta]), two);
        let sin_ph_sq = pool.pow(pool.func("sin", vec![phi]), two);
        let cos_ph_sq = pool.pow(pool.func("cos", vec![phi]), two);
        let t1 = pool.mul(vec![cos_th_sq, sin_ph_sq]);
        let t2 = pool.mul(vec![cos_th_sq, cos_ph_sq]);
        let expr = pool.add(vec![t1, t2]);
        let (after, _) = PythagoreanMultiAngle.apply(expr, &pool).unwrap();
        assert_eq!(after, cos_th_sq, "got {}", pool.display(after));
    }

    #[test]
    fn pythagorean_multi_angle_mismatched_coeff_no_fire() {
        // cos²(θ)·sin²(φ) + sin²(θ)·cos²(φ): coefficients differ
        // (cos²θ vs sin²θ), so no Pythagorean collapse on φ may fire.
        let pool = p();
        let theta = pool.symbol("theta", Domain::Real);
        let phi = pool.symbol("phi", Domain::Real);
        let two = pool.integer(2_i32);
        let cos_th_sq = pool.pow(pool.func("cos", vec![theta]), two);
        let sin_th_sq = pool.pow(pool.func("sin", vec![theta]), two);
        let sin_ph_sq = pool.pow(pool.func("sin", vec![phi]), two);
        let cos_ph_sq = pool.pow(pool.func("cos", vec![phi]), two);
        let t1 = pool.mul(vec![cos_th_sq, sin_ph_sq]);
        let t2 = pool.mul(vec![sin_th_sq, cos_ph_sq]);
        let expr = pool.add(vec![t1, t2]);
        assert!(
            PythagoreanMultiAngle.apply(expr, &pool).is_none(),
            "must not collapse when coefficient multisets differ"
        );
    }

    /// Helper: `c · sin²(u)` (or any single-arg trig) built as a Mul.
    fn coeff_trig_sq(pool: &ExprPool, coeff: ExprId, fname: &str, u: ExprId) -> ExprId {
        let f = pool.func(fname, vec![u]);
        let sq = pool.pow(f, pool.integer(2_i32));
        pool.mul(vec![coeff, sq])
    }

    #[test]
    fn coeff_pythagorean_two() {
        // 2·sin²(x) + 2·cos²(x) → 2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let s = coeff_trig_sq(&pool, two, "sin", x);
        let c = coeff_trig_sq(&pool, two, "cos", x);
        let expr = pool.add(vec![s, c]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(
            r.value,
            pool.integer(2_i32),
            "got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn coeff_pythagorean_symbolic_compound_arg() {
        // a·sin²(θ1+θ2) + a·cos²(θ1+θ2) → a   (symbolic a, compound u)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let t1 = pool.symbol("theta1", Domain::Real);
        let t2 = pool.symbol("theta2", Domain::Real);
        let u = pool.add(vec![t1, t2]);
        let s = coeff_trig_sq(&pool, a, "sin", u);
        let c = coeff_trig_sq(&pool, a, "cos", u);
        let expr = pool.add(vec![s, c]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(r.value, a, "got {}", pool.display(r.value));
    }

    #[test]
    fn coeff_pythagorean_embedded_constant_fold() {
        // 3 + 2·sin²(x) + 2·cos²(x) → 5
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let three = pool.integer(3_i32);
        let two = pool.integer(2_i32);
        let s = coeff_trig_sq(&pool, two, "sin", x);
        let c = coeff_trig_sq(&pool, two, "cos", x);
        let expr = pool.add(vec![three, s, c]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(
            r.value,
            pool.integer(5_i32),
            "got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn sin_double_angle_fires() {
        // 2·sin(x)·cos(x) → sin(2x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let expr = pool.mul(vec![two, sin_x, cos_x]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let expected = pool.func("sin", vec![two_x]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn cos_double_angle_fires() {
        // cos²(x) − sin²(x) → cos(2x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cos_sq = pool.pow(pool.func("cos", vec![x]), pool.integer(2_i32));
        let sin_sq = pool.pow(pool.func("sin", vec![x]), pool.integer(2_i32));
        let neg_sin_sq = pool.mul(vec![pool.integer(-1_i32), sin_sq]);
        let expr = pool.add(vec![cos_sq, neg_sin_sq]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let expected = pool.func("cos", vec![two_x]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn sin_angle_sub_fires() {
        // sin(a)·cos(b) − cos(a)·sin(b) → sin(a−b)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let pos = pool.mul(vec![pool.func("sin", vec![a]), pool.func("cos", vec![b])]);
        let neg = pool.mul(vec![
            pool.integer(-1_i32),
            pool.func("cos", vec![a]),
            pool.func("sin", vec![b]),
        ]);
        let expr = pool.add(vec![pos, neg]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let expected = pool.func("sin", vec![diff]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn cos_angle_sub_fires() {
        // cos(a)·cos(b) + sin(a)·sin(b) → cos(a−b)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let cc = pool.mul(vec![pool.func("cos", vec![a]), pool.func("cos", vec![b])]);
        let ss = pool.mul(vec![pool.func("sin", vec![a]), pool.func("sin", vec![b])]);
        let expr = pool.add(vec![cc, ss]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let expected = pool.func("cos", vec![diff]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn sin_angle_sub_coefficient_aware_symbolic() {
        // l1·l2·sin(a)·cos(b) − l1·l2·cos(a)·sin(b) → l1·l2·sin(a−b)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let l1 = pool.symbol("l1", Domain::Real);
        let l2 = pool.symbol("l2", Domain::Real);
        let pos = pool.mul(vec![
            l1,
            l2,
            pool.func("sin", vec![a]),
            pool.func("cos", vec![b]),
        ]);
        let neg = pool.mul(vec![
            pool.integer(-1_i32),
            l1,
            l2,
            pool.func("cos", vec![a]),
            pool.func("sin", vec![b]),
        ]);
        let expr = pool.add(vec![pos, neg]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let sin_diff = pool.func("sin", vec![diff]);
        let expected = pool.mul(vec![l1, l2, sin_diff]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn cos_angle_sub_coefficient_aware_symbolic() {
        // l1·l2·cos(a)·cos(b) + l1·l2·sin(a)·sin(b) → l1·l2·cos(a−b)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let l1 = pool.symbol("l1", Domain::Real);
        let l2 = pool.symbol("l2", Domain::Real);
        let cc = pool.mul(vec![
            l1,
            l2,
            pool.func("cos", vec![a]),
            pool.func("cos", vec![b]),
        ]);
        let ss = pool.mul(vec![
            l1,
            l2,
            pool.func("sin", vec![a]),
            pool.func("sin", vec![b]),
        ]);
        let expr = pool.add(vec![cc, ss]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let cos_diff = pool.func("cos", vec![diff]);
        let expected = pool.mul(vec![l1, l2, cos_diff]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn sin_angle_sub_mismatched_coeff_does_not_collapse() {
        // 2·sin(a)·cos(b) − 3·cos(a)·sin(b) must NOT become sin(a−b): the
        // coefficients differ, so no angle identity applies.
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let pos = pool.mul(vec![
            pool.integer(2_i32),
            pool.func("sin", vec![a]),
            pool.func("cos", vec![b]),
        ]);
        let neg = pool.mul(vec![
            pool.integer(-3_i32),
            pool.func("cos", vec![a]),
            pool.func("sin", vec![b]),
        ]);
        let expr = pool.add(vec![pos, neg]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let sin_diff = pool.func("sin", vec![diff]);
        assert_ne!(
            r.value, sin_diff,
            "mismatched coefficients must not collapse to sin(a-b)"
        );
        // It must also not become c·sin(a−b) for any single coefficient.
        for c in [
            pool.integer(2_i32),
            pool.integer(3_i32),
            pool.integer(-3_i32),
        ] {
            let bogus = pool.mul(vec![c, sin_diff]);
            assert_ne!(r.value, bogus, "got {}", pool.display(r.value));
        }
    }

    #[test]
    fn sin_angle_sub_symbolic_coeff_mismatch_does_not_collapse() {
        // l1·sin(a)·cos(b) − l2·cos(a)·sin(b): different symbolic coefficients
        // (l1 vs l2) must NOT collapse.
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let l1 = pool.symbol("l1", Domain::Real);
        let l2 = pool.symbol("l2", Domain::Real);
        let pos = pool.mul(vec![
            l1,
            pool.func("sin", vec![a]),
            pool.func("cos", vec![b]),
        ]);
        let neg = pool.mul(vec![
            pool.integer(-1_i32),
            l2,
            pool.func("cos", vec![a]),
            pool.func("sin", vec![b]),
        ]);
        let expr = pool.add(vec![pos, neg]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let sin_diff = pool.func("sin", vec![diff]);
        for c in [l1, l2] {
            let bogus = pool.mul(vec![c, sin_diff]);
            assert_ne!(r.value, bogus, "got {}", pool.display(r.value));
        }
        assert_ne!(r.value, sin_diff, "got {}", pool.display(r.value));
    }

    #[test]
    fn two_link_jacobian_determinant_collapses() {
        // 2-link planar arm Jacobian determinant:
        //   det = l1·l2·[ cos(θ1)·sin(θ1+θ2) − sin(θ1)·cos(θ1+θ2) ]
        //       = l1·l2·sin((θ1+θ2) − θ1) = l1·l2·sin(θ2)
        // We feed the bracket (the angle-difference part) and check it collapses
        // to sin(θ1+θ2 − θ1); the l1·l2 factor is carried verbatim.
        let pool = p();
        let t1 = pool.symbol("theta1", Domain::Real);
        let t2 = pool.symbol("theta2", Domain::Real);
        let sum = pool.add(vec![t1, t2]); // θ1+θ2
                                          // cos(θ1)·sin(θ1+θ2)
        let pos = pool.mul(vec![
            pool.func("cos", vec![t1]),
            pool.func("sin", vec![sum]),
        ]);
        // −sin(θ1)·cos(θ1+θ2)
        let neg = pool.mul(vec![
            pool.integer(-1_i32),
            pool.func("sin", vec![t1]),
            pool.func("cos", vec![sum]),
        ]);
        let bracket = pool.add(vec![pos, neg]);
        let r = simplify_with(bracket, &pool, &trig_rules(), SimplifyConfig::default());
        // sin((θ1+θ2) − θ1) = sin(θ2) up to the flattened, un-cancelled Add the
        // trig ruleset produces: θ1 + θ2 + (-1)·θ1 (no like-term collection in
        // the bare trig set — that is the default simplifier's job).
        let arg = pool.add(vec![t1, t2, pool.mul(vec![pool.integer(-1_i32), t1])]);
        let expected = pool.func("sin", vec![arg]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
        // And under the *default* simplifier the inner sum cancels to sin(θ2).
        let collapsed = crate::simplify::simplify(r.value, &pool);
        let want = pool.func("sin", vec![t2]);
        assert_eq!(
            collapsed.value,
            want,
            "got {}",
            pool.display(collapsed.value)
        );
    }

    #[test]
    fn log_of_exp_stays_conservative_without_context() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("log", vec![pool.func("exp", vec![x])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, expr);
    }

    #[test]
    fn exp_of_log_stays_conservative_without_context() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, expr);
    }

    #[test]
    fn log_of_product_stays_conservative_without_context() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.func("log", vec![pool.mul(vec![x, y])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, expr);
    }

    #[test]
    fn log_of_product_does_not_emit_unproven_side_conditions() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.func("log", vec![pool.mul(vec![x, y])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, expr);
        assert!(r.log.is_empty());
    }

    #[test]
    fn log_of_product_safe_does_not_fire() {
        // log_exp_rules_safe() excludes LogOfProduct — log(x*y) should not expand.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.func("log", vec![pool.mul(vec![x, y])]);
        let rules = log_exp_rules_safe();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(
            r.value, expr,
            "log(x*y) should NOT be split with log_exp_rules_safe"
        );
    }

    #[test]
    fn log_of_pow_stays_conservative_without_context() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let n = pool.integer(3_i32);
        let expr = pool.func("log", vec![pool.pow(x, n)]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, expr);
    }

    #[test]
    fn pattern_rule_simple() {
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let lhs = pool.add(vec![a, a]);
        let rhs = pool.mul(vec![pool.integer(2_i32), a]);
        let rule = PatternRule::new(Pattern::from_expr(lhs), rhs);
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, x]);
        let rule_set = PatternRuleSet::new(vec![rule], &pool);
        let r = simplify_with_pattern_rules(expr, &pool, &rule_set, SimplifyConfig::default());
        let expected = pool.mul(vec![pool.integer(2_i32), x]);
        assert_eq!(r.value, expected);
    }
}
