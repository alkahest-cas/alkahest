//! Singularity analysis of rational generating functions (P1 item 10).
//!
//! For a rational `f(z) = N(z)/D(z)`, the growth of `[zⁿ] f(z)` is governed
//! entirely by the singularity of smallest modulus. If that dominant pole `ρ`
//! is unique and has multiplicity `m`, then
//!
//! ```text
//! [zⁿ] f(z)  ~  C · n^{m-1} · ρ^{-n},     C = (−1)^m · N(ρ) / (ρ^m · (m−1)! · Dₘ(ρ))
//! ```
//!
//! where `Dₘ` is the `m`-th derivative factor left after dividing out the pole.
//! This is the standard route by which "how fast does this sequence grow?"
//! becomes a closed form — it is what turns the Fibonacci recurrence into
//! `φⁿ/√5`, and it is the workhorse of analytic combinatorics.
//!
//! # What is refused
//!
//! The transfer theorem needs a *unique* dominant singularity. When several
//! poles share the smallest modulus the coefficients carry an oscillating
//! factor and no single power-law term describes them — the classic trap, and
//! this module declines rather than reporting one of the poles as if it won.
//! A complex dominant pole (necessarily one of a conjugate pair) is declined
//! for the same reason.
//!
//! Poles are located numerically, so "unique" is decided against a relative
//! separation tolerance; the resulting expansion is then put through the same
//! numeric gate as every other route in this family, against coefficients
//! obtained by exact power-series division.

use super::asymptotic::AsymptoticError;
use super::asymptotic_common::{
    as_rational_function, complex_roots, gate_accept, qp_degree, qp_eval, qp_is_zero,
    rational_to_expr, verification_points, AsymptoticReport, Hypothesis, QPoly, Rigor,
    DEFAULT_SLACK,
};
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::simplify;
use rug::Rational;

/// Relative separation required before a smallest-modulus pole counts as the
/// *unique* dominant one.
const DOMINANCE_MARGIN: f64 = 1e-6;

/// Largest pole multiplicity handled.
const MAX_MULTIPLICITY: usize = 8;

/// Series coefficients used by the numeric gate.
const GATE_INDICES: [usize; 4] = [24, 32, 40, 48];

/// Asymptotics of `[zⁿ] f(z)` for a rational generating function `f`.
///
/// `gf` must be a rational function of `z` with rational coefficients whose
/// denominator does not vanish at the origin (so the series exists).
///
/// Refuses ([`AsymptoticError`]) rather than guessing when `f` is not rational,
/// when the dominant singularity is not unique (equal-modulus poles), when it
/// is complex, or when the resulting expansion fails the numeric gate.
pub fn coefficient_asymptotics(
    gf: ExprId,
    z: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<AsymptoticReport, AsymptoticError> {
    if z == n {
        return Err(AsymptoticError::InvalidTermCount);
    }
    let rf = as_rational_function(gf, z, pool).ok_or(AsymptoticError::UnsupportedScale)?;
    let num = rf.num.clone();
    let den = rf.den.clone();
    if qp_is_zero(&den) {
        return Err(AsymptoticError::UnsupportedScale);
    }
    // A pole at the origin means there is no ordinary power series to expand.
    if qp_eval(&den, &Rational::from(0)) == 0 {
        return Err(AsymptoticError::UnsupportedScale);
    }
    if qp_is_zero(&num) {
        return Err(AsymptoticError::GateFailed);
    }
    if qp_degree(&den) == 0 {
        // A polynomial has finitely many nonzero coefficients; no growth law.
        return Err(AsymptoticError::UnsupportedScale);
    }

    let mut derivation = Vec::new();

    // --- Locate the dominant pole ---
    let den_f64: Vec<f64> = den.iter().map(|c| c.to_f64()).collect();
    let roots = complex_roots(&den_f64).ok_or(AsymptoticError::UnsupportedScale)?;
    if roots.is_empty() {
        return Err(AsymptoticError::UnsupportedScale);
    }
    let mut moduli: Vec<(f64, usize)> = roots
        .iter()
        .enumerate()
        .map(|(i, r)| (r.abs(), i))
        .collect();
    moduli.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (rho_mod, rho_idx) = moduli[0];
    if !(rho_mod.is_finite() && rho_mod > 0.0) {
        return Err(AsymptoticError::UnsupportedScale);
    }

    // Group roots of (numerically) equal modulus: those are the competing
    // dominant singularities. More than one and the transfer theorem does not
    // give a single power-law term.
    let equal_modulus: Vec<usize> = moduli
        .iter()
        .filter(|(m, _)| (m - rho_mod).abs() <= DOMINANCE_MARGIN * rho_mod.max(1.0))
        .map(|(_, i)| *i)
        .collect();

    let rho = roots[rho_idx];
    if rho.im.abs() > DOMINANCE_MARGIN * rho_mod.max(1.0) {
        return Err(AsymptoticError::UnsupportedScale);
    }

    // Multiplicity: how many of the equal-modulus roots coincide with rho.
    let multiplicity = equal_modulus
        .iter()
        .filter(|&&i| {
            let r = roots[i];
            (r.re - rho.re).abs() <= DOMINANCE_MARGIN * rho_mod.max(1.0)
                && (r.im - rho.im).abs() <= DOMINANCE_MARGIN * rho_mod.max(1.0)
        })
        .count();
    if multiplicity == 0 || multiplicity > MAX_MULTIPLICITY {
        return Err(AsymptoticError::UnsupportedScale);
    }
    // Any equal-modulus root that is *not* rho is a competing singularity.
    if equal_modulus.len() > multiplicity {
        return Err(AsymptoticError::UnsupportedScale);
    }
    derivation.push(format!(
        "dominant pole at z ≈ {:.12} with multiplicity {multiplicity}",
        rho.re
    ));

    // --- Build the leading term  C · n^{m-1} · rho^{-n} ---
    //
    // The constant is obtained numerically from the exact series: dividing the
    // closed-form residue formula out symbolically would need algebraic-number
    // arithmetic when rho is irrational, whereas the *shape* (rho^{-n} and the
    // power of n) is exact, and one scalar is all that is left.
    let want = GATE_INDICES[GATE_INDICES.len() - 1] + 1;
    let coeffs = series_coefficients(&num, &den, want).ok_or(AsymptoticError::UnsupportedScale)?;

    let growth = 1.0 / rho.re;
    let shape = |k: usize| -> f64 {
        let nn = k as f64;
        nn.powi(multiplicity as i32 - 1) * growth.powi(k as i32)
    };
    // Fitting the constant at a single finite index silently absorbs the
    // *subleading* term. For `1/(1-z)^2`, where `[z^n] = n + 1` exactly and the
    // leading law is `C·n`, taking `C = a_48/48 = 49/48` leaves a permanent 2%
    // bias: the relative error stops shrinking with `n` and settles on `C - 1`,
    // which is precisely what an asymptotic statement must not do. Ratios of
    // this kind behave as `C_k = C + d/k`, so one Richardson step on two
    // indices cancels the `1/k` term and recovers `C` (exactly 1 in that case).
    let k_hi = GATE_INDICES[GATE_INDICES.len() - 1];
    let k_lo = GATE_INDICES[GATE_INDICES.len() / 2];
    if k_hi == k_lo {
        return Err(AsymptoticError::GateFailed);
    }
    let ratio_at = |k: usize| -> Option<f64> {
        let sh = shape(k);
        if !sh.is_finite() || sh == 0.0 {
            return None;
        }
        let v = coeffs[k].to_f64() / sh;
        if v.is_finite() {
            Some(v)
        } else {
            None
        }
    };
    let c_hi = ratio_at(k_hi).ok_or(AsymptoticError::GateFailed)?;
    let c_lo = ratio_at(k_lo).ok_or(AsymptoticError::GateFailed)?;
    let c_const = (c_hi * k_hi as f64 - c_lo * k_lo as f64) / (k_hi as f64 - k_lo as f64);
    if !c_const.is_finite() || c_const == 0.0 {
        return Err(AsymptoticError::GateFailed);
    }
    derivation.push(format!(
        "leading constant by Richardson extrapolation of a_k/shape(k) at k = {k_lo}, {k_hi}: \
         {c_lo} , {c_hi} -> {c_const}"
    ));

    // --- Numeric gate against the exact coefficients ---
    let points: Vec<f64> = GATE_INDICES.iter().map(|&k| k as f64).collect();
    let oracle: Vec<f64> = GATE_INDICES.iter().map(|&k| coeffs[k].to_f64()).collect();
    if oracle.iter().any(|v| !v.is_finite()) {
        return Err(AsymptoticError::GateFailed);
    }
    let term_vals: Vec<Vec<f64>> = vec![GATE_INDICES.iter().map(|&k| c_const * shape(k)).collect()];
    let accepted = gate_accept(&oracle, &term_vals, DEFAULT_SLACK);
    if accepted == 0 {
        return Err(AsymptoticError::GateFailed);
    }
    let verification = verification_points(&points, &oracle, &term_vals, accepted);

    // --- Assemble  C · n^{m-1} · (1/rho)^n  as an expression ---
    let c_expr = float_to_expr(c_const, pool);
    let growth_expr = float_to_expr(growth, pool);
    let mut factors = vec![c_expr];
    if multiplicity > 1 {
        factors.push(pool.pow(n, pool.integer((multiplicity - 1) as i32)));
    }
    factors.push(pool.pow(growth_expr, n));
    let term = simplify(pool.mul(factors), pool).value;

    Ok(AsymptoticReport {
        method: "singularity-analysis",
        var: n,
        terms: vec![term],
        rigor: Rigor::NumericallyConsistent,
        hypotheses: vec![
            Hypothesis::checked(
                "the generating function is rational and regular at the origin, so the \
                 coefficient sequence exists",
            ),
            Hypothesis::checked(
                "the singularity of smallest modulus is unique and real, so the transfer \
                 theorem yields a single power-law term",
            ),
            Hypothesis::assumed(
                "the poles were located numerically, so uniqueness is decided against a \
                 relative separation tolerance rather than proved",
            ),
            Hypothesis::assumed(
                "the leading constant was fitted from the exact series, not derived in \
                 closed form",
            ),
        ],
        verification,
        derivation,
    })
}

/// First `count` coefficients of `num/den` as an exact power series about 0.
fn series_coefficients(num: &QPoly, den: &QPoly, count: usize) -> Option<Vec<Rational>> {
    let d0 = den.first()?.clone();
    if d0 == 0 {
        return None;
    }
    let mut out: Vec<Rational> = Vec::with_capacity(count);
    for k in 0..count {
        // a_k = (num_k - Σ_{j=1..k} den_j · a_{k-j}) / den_0
        let mut acc = num.get(k).cloned().unwrap_or_else(|| Rational::from(0));
        for j in 1..=k {
            if let Some(dj) = den.get(j) {
                if *dj != 0 {
                    acc -= Rational::from(dj * &out[k - j]);
                }
            }
        }
        out.push(Rational::from(&acc / &d0));
    }
    Some(out)
}

/// A float as a rational expression, rounded to a manageable denominator.
fn float_to_expr(v: f64, pool: &ExprPool) -> ExprId {
    match Rational::from_f64(v) {
        Some(q) => {
            let scale = rug::Integer::from(1_000_000_000_000_i64);
            let scaled = (q * Rational::from(scale.clone())).round();
            rational_to_expr(&Rational::from((scaled.numer().clone(), scale)), pool)
        }
        None => pool.integer(0_i32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn env() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let z = pool.symbol("z", Domain::Real);
        let n = pool.symbol("n", Domain::Real);
        (pool, z, n)
    }

    /// `1/(1 - z - z²)` generates the Fibonacci numbers; `[zⁿ] ~ φⁿ/√5`.
    #[test]
    fn fibonacci_growth_is_the_golden_ratio() {
        let (pool, z, n) = env();
        let one = pool.integer(1_i32);
        let den = pool.add(vec![
            one,
            pool.mul(vec![pool.integer(-1_i32), z]),
            pool.mul(vec![pool.integer(-1_i32), z, z]),
        ]);
        let gf = pool.mul(vec![one, pool.pow(den, pool.integer(-1_i32))]);

        let r = coefficient_asymptotics(gf, z, n, &pool).expect("expansion");
        assert_eq!(r.method, "singularity-analysis");
        assert_eq!(r.terms.len(), 1);

        // Compare against the true Fibonacci numbers at a large index.
        let mut env_map = std::collections::HashMap::new();
        env_map.insert(n, 40.0);
        let approx = crate::jit::eval_interp(r.terms[0], &env_map, &pool).expect("evaluates");
        // [z^n] 1/(1 - z - z^2) is F_{n+1} (the series starts 1, 1, 2, 3, ...),
        // so index 40 of the series is the 41st Fibonacci number.
        let (mut a, mut b) = (0u64, 1u64);
        for _ in 0..41 {
            let t = a + b;
            a = b;
            b = t;
        }
        let truth = a as f64;
        assert!(
            (approx - truth).abs() / truth < 1e-6,
            "[z^40]: expansion {approx} vs truth {truth}"
        );
        assert!(r.max_relative_error().unwrap() < 1e-6);
    }

    /// A double pole gives an `n·ρ^{-n}` law: `1/(1-z)² → [zⁿ] = n+1`.
    #[test]
    fn double_pole_gives_linear_factor() {
        let (pool, z, n) = env();
        let one = pool.integer(1_i32);
        let base = pool.add(vec![one, pool.mul(vec![pool.integer(-1_i32), z])]);
        let gf = pool.pow(base, pool.integer(-2_i32));

        let r = coefficient_asymptotics(gf, z, n, &pool).expect("expansion");

        // This is a *leading-order* law: [z^n] = n + 1 exactly, and the reported
        // term is C·n, so the relative error is O(1/n) by construction. Assert
        // the asymptotic claim -- the ratio tends to 1 as n grows -- rather than
        // pointwise equality, which a one-term expansion does not promise.
        let rel = |ni: f64| -> f64 {
            let mut env_map = std::collections::HashMap::new();
            env_map.insert(n, ni);
            let approx = crate::jit::eval_interp(r.terms[0], &env_map, &pool).expect("evaluates");
            let truth = ni + 1.0;
            (approx - truth).abs() / truth
        };
        let (e100, e1000) = (rel(100.0), rel(1000.0));
        assert!(e100 < 0.05, "relative error at n=100 too large: {e100}");
        assert!(
            e1000 < e100 / 2.0,
            "relative error must shrink with n: {e100} -> {e1000}"
        );
    }

    /// Equal-modulus poles (`1/(1-z²)` has ±1) make the coefficients oscillate;
    /// there is no single power-law term and the routine must decline.
    #[test]
    fn refuses_competing_dominant_singularities() {
        let (pool, z, n) = env();
        let one = pool.integer(1_i32);
        let den = pool.add(vec![one, pool.mul(vec![pool.integer(-1_i32), z, z])]);
        let gf = pool.mul(vec![one, pool.pow(den, pool.integer(-1_i32))]);

        let err = coefficient_asymptotics(gf, z, n, &pool).expect_err("must decline");
        assert!(matches!(err, AsymptoticError::UnsupportedScale));
    }

    /// A pole at the origin means there is no ordinary power series at all.
    #[test]
    fn refuses_pole_at_the_origin() {
        let (pool, z, n) = env();
        let gf = pool.pow(z, pool.integer(-1_i32));
        assert!(coefficient_asymptotics(gf, z, n, &pool).is_err());
    }

    /// Non-rational input is out of scope for this route.
    #[test]
    fn refuses_non_rational_input() {
        let (pool, z, n) = env();
        let gf = pool.func("exp", vec![z]);
        let err = coefficient_asymptotics(gf, z, n, &pool).expect_err("must decline");
        assert!(matches!(err, AsymptoticError::UnsupportedScale));
    }

    /// A polynomial has finitely many coefficients — no growth law to report.
    #[test]
    fn refuses_polynomial_input() {
        let (pool, z, n) = env();
        let gf = pool.add(vec![pool.integer(1_i32), z]);
        assert!(coefficient_asymptotics(gf, z, n, &pool).is_err());
    }
}
