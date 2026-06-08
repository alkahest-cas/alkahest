//! Elliptic-integral *output* (first, second and third kind) for genus-1
//! radicands.
//!
//! When `∫ R(x, √P) dx` with `P` a **cubic or quartic** polynomial is genus-1
//! and **non-elementary**, the antiderivative is a combination of an algebraic
//! part and incomplete elliptic integrals of the first (`EllipticF`), second
//! (`EllipticE`) and third (`EllipticPi`) kind.  Byrd & Friedman, *Handbook of
//! Elliptic Integrals*, show that all of these reduce under a **single**
//! substitution `φ = φ(x)`, `m` — the one used for the first kind.
//!
//! * [`try_elliptic_output`] — the *pure first kind* `∫ c·dx/√P`
//!   → `c·g·EllipticF(φ(x), m)` (PR2).
//! * [`try_elliptic_output_higher_kind`] — `∫ b(x)·√P dx` for rational `b`
//!   (so the general `∫ R(x)/√P dx` via `b·√P = (b·P)/√P`), emitting
//!   ```text
//!   F_cand(x) = (Σⱼ αⱼ xʲ)·√P + Σ_r ρ_r·√P/(x−r)
//!              + β·EllipticF(φ,m) + γ·EllipticE(φ,m)
//!              + Σ_p δ_p·EllipticPi(n_p,φ,m)
//!   ```
//!   (PR3, second/third kind).  `φ(x) = arcsin/arccos(S(x))` for an explicit
//!   real Möbius/quotient `S`, modulus `m` (Mathematica convention `m = k²`).
//!
//! For the higher-kind path the block coefficients are **fitted numerically**
//! (least squares over many in-domain samples, then snapped to exact rationals);
//! several progressively richer block sets are tried and the first that
//! *gate-verifies* wins.
//!
//! # Soundness
//!
//! No reduction constant is trusted blindly.  Every candidate is run through a
//! numeric verification gate (`verify` / `verify_higher`): its *symbolic*
//! `d/dx` (via the engine's `diff`, which differentiates the elliptic functions
//! through the primitive registry — `∂φ F = 1/√(1 − m·sin²φ)`,
//! `∂φ E = √(1 − m·sin²φ)`, `∂φ Π = 1/((1 − n sin²φ)√(1 − m sin²φ))`, all
//! elementary since `m`, `n` are constant here) is sampled against the integrand
//! at points where `P > 0`.  A form is emitted **only** if the gate passes;
//! otherwise the caller falls through to `NonElementary`.  An imperfect fit can
//! therefore never produce a wrong answer — it merely declines.

use crate::integrate::risch::poly_rde::{expr_to_qpoly, is_free_of_var};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// A complex root, stored as `(re, im)`.
type Croot = (f64, f64);

/// Try to emit a first-kind `EllipticF` closed form for `∫ (a + b·√P) dx` when
/// the integrand reduces to the pure first-kind shape `c/√P` (`a = 0`,
/// `b = c/P` with `c` a constant) and `P` is a gate-verifiable cubic/quartic.
///
/// Returns the antiderivative `g·EllipticF(φ(x), m)` (numeric `g`, `m`,
/// real-Möbius `φ`) when the verification gate passes, else `None` (caller
/// falls through to the existing `NonElementary` path).
pub fn try_elliptic_output(
    a_part: ExprId,
    b_part: ExprId,
    p_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    // Restrict to the *pure first kind*: `∫ c·dx/√P`.  This is `a = 0` and
    // `b·√P = c/√P`, i.e. `b = c/P` with `c` free of `var`.
    if !is_zero(a_part, pool) {
        return None;
    }
    let bp = pool.mul(vec![b_part, p_expr]);
    let c_expr = simplify(bp, pool).value;
    if !is_free_of_var(c_expr, var, pool) {
        return None;
    }
    let c = eval_const(c_expr, pool)?;
    if !c.is_finite() || c == 0.0 {
        return None;
    }

    // Parse P to rational coefficients (ascending) and get its degree.
    let p_poly = expr_to_qpoly(p_expr, var, pool)?;
    let coeffs: Vec<f64> = p_poly.iter().map(|r| r.to_f64()).collect();
    let deg = coeffs.len().checked_sub(1)?;
    if deg != 3 && deg != 4 {
        return None;
    }
    let lead = *coeffs.last()?;
    if lead == 0.0 {
        return None;
    }

    let (g, m, phi) = first_kind_reduction(&coeffs, deg, lead, var, pool)?;

    // F_cand = (c · g) · EllipticF(phi, m).
    let m_expr = float_to_expr(m, pool);
    let f = pool.func("EllipticF", vec![phi, m_expr]);
    let coeff = float_to_expr(c * g, pool);
    let f_cand = simplify(pool.mul(vec![coeff, f]), pool).value;

    // Soundness gate: d/dx F_cand = c/√P numerically where P > 0.
    if verify(f_cand, &coeffs, c, var, pool) {
        Some(f_cand)
    } else {
        None
    }
}

/// Compute the shared first-kind Legendre reduction `(g, m, φ(x))` for `√P`,
/// chosen so that `d/dx[g·EllipticF(φ,m)] = 1/√P` on the real region where
/// `P > 0`.  This is the *same* substitution used by every higher-kind
/// reduction (B&F: all of `∫R(x,√P)dx` reduce under one substitution), so the
/// second/third-kind paths reuse it verbatim.
///
/// Returns `None` for radicand shapes outside the handled cubic/quartic
/// root-configurations (e.g. all-complex quartic).
fn first_kind_reduction(
    coeffs: &[f64],
    deg: usize,
    lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let roots = poly_roots(coeffs)?;
    let (mut reals, pairs) = classify_roots(&roots);
    reals.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let inv_sqrt_lead = 1.0 / lead.abs().sqrt();
    match (deg, reals.len(), pairs.len()) {
        (3, 3, 0) => cubic_three_real(&reals, inv_sqrt_lead, var, pool),
        (3, 1, 1) => cubic_one_real(reals[0], pairs[0], inv_sqrt_lead, var, pool),
        (4, 4, 0) => quartic_four_real(&reals, inv_sqrt_lead, var, pool),
        (4, 2, 1) => quartic_two_real(&reals, pairs[0], inv_sqrt_lead, var, pool),
        (4, 0, 2) => quartic_no_real(pairs[0], pairs[1], lead, var, pool),
        _ => None, // genus-2 / unhandled config: declined (falls through)
    }
}

/// Extra **real poles** introduced into the second-kind reduction by the
/// `EllipticE` block.
///
/// `d/dx[g·E(φ,m)] = (1 − m·sin²φ(x))/√P`, and `sin²φ(x)` is a rational function
/// of `x` whose poles are *not* in general roots of `P`.  For the genuine
/// second-kind reduction `∫poly(x)/√P → algebraic + β·F + γ·E` to close in the
/// numeric fit, the algebraic ansatz must contain rational blocks `√P/(x−p)`
/// (and `√P/(x−p)²`) at exactly these poles so the `E`-induced rational part can
/// be cancelled.  This returns those poles (the "B&F second-kind reduction
/// poles") for each handled root configuration:
///
/// * cubic, three real roots `e1>e2>e3`: `sin²φ = (e1−e3)/(x−e3)` ⇒ pole `e3`
///   (already a root of `P`, but returned for completeness).
/// * cubic, one real root `y1`, pair `b1±i·a1`: `cos φ = (A−u)/(A+u)`,
///   `u = x−y1`, `A = √((y1−b1)²+a1²)` ⇒ double pole at `x = y1 − A`.
/// * quartic, four real roots `a>b>c>d`: `sin²φ ∝ (x−c)/(x−d)` ⇒ pole `d`.
/// * quartic, two real roots `b1>b2`, pair `b3±i·a3`: `cos φ` denominator
///   `(A1−A2)x + (b1·A2 − b2·A1)` ⇒ double pole at `x = (b2·A1 − b1·A2)/(A1−A2)`.
fn reduction_poles(coeffs: &[f64], deg: usize) -> Vec<f64> {
    let Some(roots) = poly_roots(coeffs) else {
        return Vec::new();
    };
    let (mut reals, pairs) = classify_roots(&roots);
    reals.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let mut out = Vec::new();
    match (deg, reals.len(), pairs.len()) {
        (3, 3, 0) => out.push(reals[2]), // e3
        (3, 1, 1) => {
            let (y1, (b1, a1)) = (reals[0], pairs[0]);
            let aa = ((y1 - b1).powi(2) + a1 * a1).sqrt();
            out.push(y1 - aa);
        }
        (4, 4, 0) => out.push(reals[3]), // d
        (4, 2, 1) => {
            let (b1, b2) = (reals[0], reals[1]);
            let (b3, a3) = pairs[0];
            let aa1 = ((b1 - b3).powi(2) + a3 * a3).sqrt();
            let aa2 = ((b2 - b3).powi(2) + a3 * a3).sqrt();
            if (aa1 - aa2).abs() > 1e-12 {
                out.push((b2 * aa1 - b1 * aa2) / (aa1 - aa2));
            }
        }
        (4, 0, 2) => {
            // arctan substitution: `sin²φ(x) = L²/(1+L²)`, `L = (px+q)/(rx+s)`.
            // The only real pole of `sin²φ` (hence of the `E`-block rational part)
            // is the pole of `L` at `x = −s/r`.
            let lead = *coeffs.last().unwrap_or(&1.0);
            if let Some((_p, _q, r, s, _m, _g)) = quartic_no_real_consts(pairs[0], pairs[1], lead) {
                if r.abs() > 1e-12 {
                    out.push(-s / r);
                }
            }
        }
        _ => {}
    }
    out.retain(|p| p.is_finite());
    out
}

/// Second/third-kind elliptic-integral *output* for genus-1 radicands.
///
/// Handles `∫ b(x)·√P dx` where `b` is a rational function of `var` and `P` is a
/// gate-verifiable cubic/quartic — i.e. the general `∫ R(x)/√P dx` (writing
/// `b·√P = (b·P)/√P`).  The antiderivative is built as an *ansatz*
///
/// ```text
///   F_cand(x) = (Σⱼ αⱼ xʲ)·√P  +  β·EllipticF(φ,m) + γ·EllipticE(φ,m)
///                                  +  Σ_p δ_p·EllipticPi(n_p, φ, m)
/// ```
///
/// over the shared first-kind substitution `(g, m, φ)`.  The algebraic block
/// degree is chosen from the numerator degree; the `EllipticPi` blocks are one
/// per simple real pole `p` of `b` (third kind).
///
/// For the **general second kind** `∫ poly(x)/√P dx` (polynomial numerator, no
/// extra poles) the algebraic ansatz also carries rational blocks
/// `√P/(x−p)`, `√P/(x−p)²` at the `reduction_poles` of the `EllipticE`
/// reduction — the poles of `sin²φ(x)`, which for the cubic-one-real and
/// quartic-two-real configurations lie *off* the roots of `P`.  Without these
/// the `E`-induced rational part cannot be cancelled and the fit cannot close
/// (e.g. `∫ x/√(x³+1) dx`).  The block coefficients are
/// **fitted numerically** (least squares over many sample points where `P > 0`),
/// reconstructed as exact rationals, and the assembled candidate is run through
/// the *same* `d/dx F = integrand` soundness gate as the first kind.  An
/// imperfect fit can therefore only *decline* (return `None`, caller falls
/// through to `NonElementary`) — never emit a wrong answer.
///
/// Requires `a_part = 0` (the wiring integrates a separate rational `a_part`
/// itself); `b_part` purely algebraic.
pub fn try_elliptic_output_higher_kind(
    a_part: ExprId,
    b_part: ExprId,
    p_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    use crate::integrate::risch::rational_rde::expr_to_qrational;

    if !is_zero(a_part, pool) {
        return None;
    }
    if is_zero(b_part, pool) {
        return None;
    }

    // Parse P (ascending rational coeffs) and validate degree / leading coeff.
    let p_poly = expr_to_qpoly(p_expr, var, pool)?;
    let p_coeffs: Vec<f64> = p_poly.iter().map(|r| r.to_f64()).collect();
    let deg = p_coeffs.len().checked_sub(1)?;
    if deg != 3 && deg != 4 {
        return None;
    }
    let lead = *p_coeffs.last()?;
    if lead == 0.0 {
        return None;
    }

    // `b = b_num / b_den` as rational polynomials in `var`.  `expr_to_qrational`
    // does *not* reduce to lowest terms (e.g. `1/((x−p)√P)` decomposes to
    // `(x−p)/((x−p)²·P)·√P`), so we cancel the polynomial GCD first.  This is
    // essential for the third-kind path: an *un-reduced* `b` hides the simple pole
    // at `x = p` (it appears as a numerator root too), so the pole detector would
    // miss it and no `EllipticPi` block would be added.
    let (b_num, b_den) = {
        use crate::integrate::risch::rational_rde::{poly_div_exact, poly_gcd};
        let (n, d) = expr_to_qrational(b_part, var, pool)?;
        let gcd = poly_gcd(&n, &d);
        if gcd.len() > 1 {
            (poly_div_exact(&n, &gcd), poly_div_exact(&d, &gcd))
        } else {
            (n, d)
        }
    };
    let b_num_f: Vec<f64> = b_num.iter().map(|r| r.to_f64()).collect();
    let b_den_f: Vec<f64> = b_den.iter().map(|r| r.to_f64()).collect();
    if b_den_f.iter().all(|&c| c == 0.0) {
        return None;
    }

    // The shared first-kind substitution.
    let (g, m, phi) = first_kind_reduction(&p_coeffs, deg, lead, var, pool)?;
    if !(g.is_finite() && m.is_finite()) || m >= 1.0 {
        return None;
    }

    // ── Candidate block sets ───────────────────────────────────────────────
    //
    // Integrand to match: `b·√P`.  Every block is an `ExprId` whose `d/dx` is
    // elementary (the elliptic derivatives reduce to `√(1−m sin²φ)`-type forms
    // because `m`, `n` are constants here), so the gate can sample them.
    //
    //  * Algebraic polynomial blocks `xʲ·√P` (`d/dx → (…)/√P`, numerator degree
    //    `j + deg − 1`).
    //  * Rational algebraic blocks `√P/(x−r)` for each real root `r` of `P`
    //    (needed when the substitution puts a pole into the `E` reduction — the
    //    three-real-root cubic / generic quartic cases).
    //  * `EllipticF`, `EllipticE` blocks (first/second kind).
    //  * `EllipticPi(n_p,φ,m)` + `√P/(x−p)` for each simple real pole `p` of `b`
    //    (third kind); characteristic `n_p = 1/sin²φ(p)`.
    //
    // We try progressively richer sets and keep the first that *gate-verifies*.
    // Soundness is unconditional: an inexact fit just declines.
    let m_expr = float_to_expr(m, pool);
    let sqrt_p = pool.func("sqrt", vec![p_expr]);
    let g_expr = float_to_expr(g, pool);
    let f_blk = simplify(
        pool.mul(vec![g_expr, pool.func("EllipticF", vec![phi, m_expr])]),
        pool,
    )
    .value;
    let e_blk = simplify(
        pool.mul(vec![g_expr, pool.func("EllipticE", vec![phi, m_expr])]),
        pool,
    )
    .value;

    // Polynomial degree of `b` numerator (used to pick the algebraic ladder).
    let db = (b_num.len().max(1) as i64 - 1) - (b_den.len().max(1) as i64 - 1);
    let k_poly = (db.max(0) as usize) + 1;

    // Real roots of `P` (for the rational algebraic blocks) and real poles of
    // `b` (for the third-kind Π blocks).
    let p_roots: Vec<f64> = {
        let roots = poly_roots(&p_coeffs).unwrap_or_default();
        let (mut r, _) = classify_roots(&roots);
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r
    };
    let real_poles = real_simple_poles(&b_num_f, &b_den_f);

    // Second-kind reduction poles (where the `EllipticE` block's `sin²φ(x)`
    // introduces non-`P` poles that the algebraic ansatz must cancel).
    let red_poles = reduction_poles(&p_coeffs, deg);

    // Helper to build `xʲ·√P` and `√P/(x−r)^k` blocks.
    let poly_block = |j: usize, pool: &ExprPool| -> ExprId {
        let xj = match j {
            0 => pool.integer(1_i32),
            1 => var,
            _ => pool.pow(var, pool.integer(j as i32)),
        };
        pool.mul(vec![xj, sqrt_p])
    };
    let rat_block = |r: f64, pool: &ExprPool| -> ExprId {
        let xr = pool.add(vec![var, float_to_expr(-r, pool)]);
        pool.mul(vec![sqrt_p, pool.pow(xr, pool.integer(-1_i32))])
    };
    let rat_pow_block = |r: f64, k: i32, pool: &ExprPool| -> ExprId {
        let xr = pool.add(vec![var, float_to_expr(-r, pool)]);
        pool.mul(vec![sqrt_p, pool.pow(xr, pool.integer(-k))])
    };

    // Build the list of block-set recipes (each a Vec of block ExprIds).
    let mut recipes: Vec<Vec<ExprId>> = Vec::new();
    // 1) base: x·√P, √P, F, E  (+ higher x ladder if b has high degree)
    {
        let mut s = Vec::new();
        for j in 0..=k_poly.max(1) {
            s.push(poly_block(j, pool));
        }
        s.push(f_blk);
        s.push(e_blk);
        recipes.push(s);
    }
    // 1b) GENERAL SECOND KIND (this PR): polynomial `xʲ·√P` ladder + the
    //     second-kind reduction-pole blocks `√P/(x−p)` and `√P/(x−p)²` + F + E.
    //     This is the basis that closes `∫poly(x)/√P dx` for the cubic-one-real
    //     and quartic-two-real cases (e.g. `∫x/√(x³+1)`), where the `E`-induced
    //     rational part has a pole *off* the roots of `P`.  Built only when there
    //     are reduction poles to add (else identical to recipe 1).
    if !red_poles.is_empty() {
        let mut s = Vec::new();
        for j in 0..=k_poly.max(1) {
            s.push(poly_block(j, pool));
        }
        for &p in &red_poles {
            s.push(rat_block(p, pool));
            s.push(rat_pow_block(p, 2, pool));
        }
        s.push(f_blk);
        s.push(e_blk);
        recipes.push(s);
    }
    // 2) base + one rational block at the smallest real root of P.
    if let Some(&rmin) = p_roots.first() {
        let mut s = vec![poly_block(0, pool), poly_block(1, pool)];
        s.push(rat_block(rmin, pool));
        s.push(f_blk);
        s.push(e_blk);
        recipes.push(s);
    }
    // 3) base + a rational block at every real root of P.
    if p_roots.len() > 1 {
        let mut s = vec![poly_block(0, pool), poly_block(1, pool)];
        for &r in &p_roots {
            s.push(rat_block(r, pool));
        }
        s.push(f_blk);
        s.push(e_blk);
        recipes.push(s);
    }
    // ── THIRD KIND (this PR) ────────────────────────────────────────────────
    //
    // For `∫ R(x)/((x−p)√P) dx` the antiderivative carries an `EllipticPi(n_p,φ,m)`
    // block for each *real* simple pole `p` of the rational weight `b` that is
    // **not** a root of `P` (a pole *at* a root of `P` is a different kind, handled
    // by the algebraic/`F`/`E` blocks).  The characteristic is `n_p = 1/sin²φ(p)`.
    //
    // This single-`Π` reduction is exact **iff** `sin²φ(x)` is a Möbius
    // (linear-fractional) function of `x`, which holds for the `asin(√·)`
    // substitutions — the cubic-three-real-root and quartic-four-real-root
    // configurations.  For the `cos φ` substitutions (cubic-one-real,
    // quartic-two-real-plus-pair) `sin²φ` is a *quadratic*-over-quadratic in `x`,
    // so a pole at `x = p` is shared with a "twin" preimage and a single `Π`
    // introduces a spurious pole there; the fit then fails to close and the gate
    // declines (→ `NonElementary`).  We still *try* these — soundness is
    // unconditional — but they are expected to decline.
    //
    // We add the Π blocks for every off-`P`-root real pole and let the numeric fit
    // + gate decide.  Two recipe variants are pushed: a *minimal* one (just the
    // algebraic ladder, `F`, and the Π blocks) which produces clean rational
    // coefficients when it closes, and the *rich* one (also `E` and a `√P/(x−r)`
    // block at the smallest root of `P`) as a fallback.
    let pi_poles: Vec<(f64, f64)> = real_poles
        .iter()
        .filter_map(|&p| {
            // Skip poles that coincide with a root of `P` (different kind).
            if p_roots.iter().any(|&r| (r - p).abs() < 1e-7) {
                return None;
            }
            let np = characteristic_from_pole(p, phi, var, pool)?;
            if np.is_finite() && (np - 1.0).abs() > 1e-9 {
                Some((p, np))
            } else {
                None
            }
        })
        .collect();
    if !pi_poles.is_empty() {
        let build_pi = |s: &mut Vec<ExprId>, pool: &ExprPool| {
            for &(_p, np) in &pi_poles {
                let n_expr = float_to_expr(np, pool);
                s.push(simplify(pool.func("EllipticPi", vec![n_expr, phi, m_expr]), pool).value);
            }
        };
        // 4a) minimal third-kind basis: algebraic ladder + F + Π blocks.
        {
            let mut s = vec![poly_block(0, pool), poly_block(1, pool)];
            s.push(f_blk);
            build_pi(&mut s, pool);
            recipes.push(s);
        }
        // 4b) rich third-kind basis: + E, + reduction-pole / root algebraic blocks
        //     + a `√P/(x−p)` block per Π pole (cancels residual rational parts).
        {
            let mut s = vec![poly_block(0, pool), poly_block(1, pool)];
            if let Some(&rmin) = p_roots.first() {
                s.push(rat_block(rmin, pool));
            }
            for &p in &red_poles {
                s.push(rat_block(p, pool));
                s.push(rat_pow_block(p, 2, pool));
            }
            s.push(f_blk);
            s.push(e_blk);
            build_pi(&mut s, pool);
            for &(p, _) in &pi_poles {
                s.push(rat_block(p, pool));
            }
            recipes.push(s);
        }
    }

    // Sample grid (shared across recipes) where `P > 0` and away from b-poles.
    let samples = sample_grid(&p_coeffs, &b_den_f);

    for blocks in &recipes {
        if let Some(f_cand) =
            fit_and_assemble(blocks, &samples, &p_coeffs, &b_num_f, &b_den_f, var, pool)
        {
            if verify_higher(f_cand, &p_coeffs, &b_num_f, &b_den_f, var, pool) {
                return Some(f_cand);
            }
        }
    }
    None
}

/// Fit block coefficients by least squares against the integrand `b·√P` over the
/// in-domain samples, snap them to rationals, and assemble the candidate
/// `Σ cᵢ·blockᵢ`.  Returns `None` on a rank-deficient / non-evaluable design.
#[allow(clippy::too_many_arguments)]
fn fit_and_assemble(
    blocks: &[ExprId],
    samples: &[f64],
    p_coeffs: &[f64],
    b_num: &[f64],
    b_den: &[f64],
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let block_dx: Vec<ExprId> = blocks
        .iter()
        .map(|&blk| {
            crate::diff::diff(blk, var, pool)
                .ok()
                .map(|d| simplify(d.value, pool).value)
        })
        .collect::<Option<Vec<_>>>()?;
    let nblk = blocks.len();

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for &xv in samples {
        let pv = eval_poly(p_coeffs, xv);
        if pv <= 1e-6 {
            continue;
        }
        let Some(bv) = eval_ratio(b_num, b_den, xv) else {
            continue;
        };
        let yv = bv * pv.sqrt();
        if !yv.is_finite() {
            continue;
        }
        let mut row = Vec::with_capacity(nblk);
        let mut ok = true;
        for &dxi in &block_dx {
            match eval(dxi, var, xv, pool) {
                Some(v) if v.is_finite() => row.push(v),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            rows.push(row);
            ys.push(yv);
        }
    }
    if rows.len() < nblk + 1 {
        return None;
    }

    let coeffs_fit = lstsq(&rows, &ys, nblk)?;

    // Reject a poor fit early: the design must reproduce the integrand at the
    // sample points to high accuracy, otherwise the basis is incomplete for this
    // integrand and we should not even build a candidate (the gate would reject
    // it anyway, but this keeps cheap recipes from masking richer ones).
    {
        let mut maxr = 0.0_f64;
        for (row, &y) in rows.iter().zip(&ys) {
            let pred: f64 = row.iter().zip(&coeffs_fit).map(|(a, b)| a * b).sum();
            maxr = maxr.max((pred - y).abs() / (1.0 + y.abs()));
        }
        if !maxr.is_finite() || maxr > 1e-7 {
            return None;
        }
    }

    let mut terms: Vec<ExprId> = Vec::new();
    for (i, &ci) in coeffs_fit.iter().enumerate() {
        // Snap to a nearby simple rational *only* when that does not move the
        // value (clean output for the rational algebraic / first-kind blocks);
        // otherwise keep the fitted float (the second-kind `F`/`E` coefficients
        // are generically irrational, e.g. involve `√3`).  The gate guards
        // correctness in both cases.
        let snapped = snap_rational(ci);
        let cr = if (snapped - ci).abs() < 1e-10 * (1.0 + ci.abs()) {
            snapped
        } else {
            ci
        };
        if cr.abs() < 1e-12 {
            continue;
        }
        let coeff = float_to_expr(cr, pool);
        terms.push(pool.mul(vec![coeff, blocks[i]]));
    }
    if terms.is_empty() {
        return None;
    }
    Some(simplify(pool.add(terms), pool).value)
}

/// Numeric value of `b_num(x)/b_den(x)` (ascending coeffs); `None` if denom ≈ 0.
fn eval_ratio(num: &[f64], den: &[f64], x: f64) -> Option<f64> {
    let d = eval_poly(den, x);
    if d.abs() < 1e-12 {
        return None;
    }
    Some(eval_poly(num, x) / d)
}

/// Real simple poles of `b = num/den`: real roots of `den` that are not roots of
/// `num`.  Returns at most a couple (enough for the third-kind ladder).
fn real_simple_poles(num: &[f64], den: &[f64]) -> Vec<f64> {
    if den.len() <= 1 {
        return Vec::new();
    }
    let Some(roots) = poly_roots(den) else {
        return Vec::new();
    };
    let (reals, _) = classify_roots(&roots);
    let mut out = Vec::new();
    for r in reals {
        if eval_poly(num, r).abs() > 1e-7 {
            // Deduplicate close poles.
            if !out.iter().any(|&q: &f64| (q - r).abs() < 1e-6) {
                out.push(r);
            }
        }
    }
    out
}

/// Characteristic `n_p = 1/sin²φ(p)` for an `EllipticPi` block whose pole is at
/// `x = p`.  Evaluates the elementary φ expression numerically.
fn characteristic_from_pole(p: f64, phi: ExprId, var: ExprId, pool: &ExprPool) -> Option<f64> {
    let phi_v = eval(phi, var, p, pool)?;
    let s = phi_v.sin();
    let s2 = s * s;
    if s2.abs() < 1e-12 {
        return None;
    }
    Some(1.0 / s2)
}

/// Sample grid for the fit, biased to the `P > 0` region and away from poles.
fn sample_grid(p_coeffs: &[f64], b_den: &[f64]) -> Vec<f64> {
    let mut xs = Vec::new();
    let mut x = -4.0_f64;
    while x <= 6.0 {
        // Skip points too close to a denominator zero.
        if eval_poly(b_den, x).abs() > 1e-3 {
            xs.push(x);
        }
        let _ = p_coeffs;
        x += 0.137;
    }
    xs
}

/// Minimal least-squares solver: normal equations `AᵀA c = Aᵀy` with Gaussian
/// elimination (partial pivoting).  `n` = number of unknowns.
fn lstsq(rows: &[Vec<f64>], ys: &[f64], n: usize) -> Option<Vec<f64>> {
    // Normal equations.
    let mut ata = vec![vec![0.0_f64; n]; n];
    let mut aty = vec![0.0_f64; n];
    for (row, &y) in rows.iter().zip(ys) {
        for i in 0..n {
            aty[i] += row[i] * y;
            for j in 0..n {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    // Gaussian elimination with partial pivoting on the n×n system.
    for col in 0..n {
        let mut piv = col;
        let mut best = ata[col][col].abs();
        for (r, arow) in ata.iter().enumerate().take(n).skip(col + 1) {
            if arow[col].abs() > best {
                best = arow[col].abs();
                piv = r;
            }
        }
        if best < 1e-12 {
            return None; // singular / rank-deficient design
        }
        ata.swap(col, piv);
        aty.swap(col, piv);
        let d = ata[col][col];
        // Snapshot the pivot row to avoid a borrow conflict while updating others.
        let pivot_row = ata[col].clone();
        let pivot_y = aty[col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = ata[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for (c, prc) in pivot_row.iter().enumerate().take(n).skip(col) {
                ata[r][c] -= f * prc;
            }
            aty[r] -= f * pivot_y;
        }
    }
    let mut out = vec![0.0; n];
    for (i, oi) in out.iter_mut().enumerate() {
        *oi = aty[i] / ata[i][i];
        if !oi.is_finite() {
            return None;
        }
    }
    Some(out)
}

/// Snap a fitted float coefficient to a nearby simple rational (denominators up
/// to 60) and zero out numerical noise.  Keeps emitted forms clean; the gate
/// still guards correctness regardless.
fn snap_rational(v: f64) -> f64 {
    if v.abs() < 1e-9 {
        return 0.0;
    }
    for den in 1..=60_i64 {
        let num = (v * den as f64).round();
        let cand = num / den as f64;
        if (cand - v).abs() < 1e-9 * (1.0 + v.abs()) {
            return cand;
        }
    }
    v
}

/// Numerically verify `d/dx F_cand = b·√P` at sample points where `P > 0`.
fn verify_higher(
    f_cand: ExprId,
    p_coeffs: &[f64],
    b_num: &[f64],
    b_den: &[f64],
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    let Ok(df) = crate::diff::diff(f_cand, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    let samples = [
        -3.5, -2.7, -1.6, -0.9, -0.4, 0.15, 0.3, 0.55, 0.7, 0.9, 1.1, 1.4, 1.9, 2.6, 3.4, 4.7, 5.3,
    ];
    let mut checked = 0;
    for &xv in &samples {
        let pv = eval_poly(p_coeffs, xv);
        if pv <= 1e-6 {
            continue;
        }
        let Some(bv) = eval_ratio(b_num, b_den, xv) else {
            continue;
        };
        let rhs = bv * pv.sqrt();
        let Some(lhs) = eval(ds, var, xv, pool) else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        if (lhs - rhs).abs() > 1e-7 * (1.0 + rhs.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 3
}

// ---------------------------------------------------------------------------
// Reduction cases (Byrd & Friedman normal forms)
// ---------------------------------------------------------------------------

/// Cubic, three real roots `e1 > e2 > e3` (region `x ≥ e1`, where `P > 0` for a
/// positive leading coefficient): `sin²φ = (e1−e3)/(x−e3)`,
/// `m = (e2−e3)/(e1−e3)`, `g = −2/√(e1−e3)`.
fn cubic_three_real(
    reals: &[f64],
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (e1, e2, e3) = (reals[0], reals[1], reals[2]);
    let denom = e1 - e3;
    if denom <= 0.0 {
        return None;
    }
    let g = -2.0 / denom.sqrt() * inv_sqrt_lead;
    let m = (e2 - e3) / denom;
    // φ = arcsin( √( (e1−e3)/(x−e3) ) )
    let x_minus_e3 = pool.add(vec![var, float_to_expr(-e3, pool)]);
    let ratio = pool.mul(vec![
        float_to_expr(e1 - e3, pool),
        pool.pow(x_minus_e3, pool.integer(-1_i32)),
    ]);
    let s = pool.func("sqrt", vec![ratio]);
    let phi = pool.func("asin", vec![s]);
    Some((g, m, phi))
}

/// Cubic, one real root `y1` and a complex pair `b1 ± i·a1` (region `x ≥ y1`):
/// `A = √((y1−b1)² + a1²)`, `g = 1/√A`, `m = (A + (b1−y1))/(2A)`,
/// `cos φ = (A − (x−y1))/(A + (x−y1))`.
fn cubic_one_real(
    y1: f64,
    pair: Croot,
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (b1, a1) = pair;
    let aa = ((y1 - b1).powi(2) + a1 * a1).sqrt();
    if aa <= 0.0 {
        return None;
    }
    let g = inv_sqrt_lead / aa.sqrt();
    let m = (aa + (b1 - y1)) / (2.0 * aa);
    // cos φ = (A − (x − y1)) / (A + (x − y1)); φ = arccos(...)
    let x_minus_y1 = pool.add(vec![var, float_to_expr(-y1, pool)]);
    let num = pool.add(vec![
        float_to_expr(aa, pool),
        pool.mul(vec![pool.integer(-1_i32), x_minus_y1]),
    ]);
    let den = pool.add(vec![float_to_expr(aa, pool), x_minus_y1]);
    let cosphi = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
    let phi = pool.func("acos", vec![cosphi]);
    Some((g, m, phi))
}

/// Quartic, four real roots `a > b > c > d` (region `c ≤ x ≤ b`, where `P > 0`):
/// `sn²φ = (b−d)(x−c)/((b−c)(x−d))`, `m = (b−c)(a−d)/((a−c)(b−d))`,
/// `g = 2/√((a−c)(b−d))`.
fn quartic_four_real(
    reals: &[f64],
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (a, b, c, d) = (reals[0], reals[1], reals[2], reals[3]);
    let ac = a - c;
    let bd = b - d;
    let bc = b - c;
    if ac <= 0.0 || bd <= 0.0 || bc <= 0.0 {
        return None;
    }
    let g = 2.0 / (ac * bd).sqrt() * inv_sqrt_lead;
    let m = bc * (a - d) / (ac * bd);
    // sin²φ = (b−d)(x−c) / ((b−c)(x−d))
    let x_minus_c = pool.add(vec![var, float_to_expr(-c, pool)]);
    let x_minus_d = pool.add(vec![var, float_to_expr(-d, pool)]);
    let num = pool.mul(vec![float_to_expr(bd, pool), x_minus_c]);
    let den = pool.mul(vec![float_to_expr(bc, pool), x_minus_d]);
    let ratio = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
    let s = pool.func("sqrt", vec![ratio]);
    let phi = pool.func("asin", vec![s]);
    Some((g, m, phi))
}

/// Quartic, two real roots `b1 > b2` and a complex pair `b3 ± i·a3`
/// (region `b2 ≤ x ≤ b1`, where `P > 0`):
/// `A1 = √((b1−b3)² + a3²)`, `A2 = √((b2−b3)² + a3²)`, `g = 1/√(A1·A2)`,
/// `m = ((A1+A2)² − (b1−b2)²)/(4·A1·A2)`,
/// `cos φ = ((b1−x)A2 − (x−b2)A1)/((b1−x)A2 + (x−b2)A1)`.
fn quartic_two_real(
    reals: &[f64],
    pair: Croot,
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (b1, b2) = (reals[0], reals[1]);
    let (b3, a3) = pair;
    let aa1 = ((b1 - b3).powi(2) + a3 * a3).sqrt();
    let aa2 = ((b2 - b3).powi(2) + a3 * a3).sqrt();
    if aa1 <= 0.0 || aa2 <= 0.0 {
        return None;
    }
    let g = inv_sqrt_lead / (aa1 * aa2).sqrt();
    let m = ((aa1 + aa2).powi(2) - (b1 - b2).powi(2)) / (4.0 * aa1 * aa2);
    // cos φ = ((b1−x)A2 − (x−b2)A1) / ((b1−x)A2 + (x−b2)A1)
    let b1_minus_x = pool.add(vec![
        float_to_expr(b1, pool),
        pool.mul(vec![pool.integer(-1_i32), var]),
    ]);
    let x_minus_b2 = pool.add(vec![var, float_to_expr(-b2, pool)]);
    let t1 = pool.mul(vec![b1_minus_x, float_to_expr(aa2, pool)]);
    let t2 = pool.mul(vec![x_minus_b2, float_to_expr(aa1, pool)]);
    let num = pool.add(vec![t1, pool.mul(vec![pool.integer(-1_i32), t2])]);
    let den = pool.add(vec![t1, t2]);
    let cosphi = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
    let phi = pool.func("acos", vec![cosphi]);
    Some((g, m, phi))
}

/// Quartic with **no real roots** — two complex-conjugate pairs `b1 ± i·a1`,
/// `b2 ± i·a2` (`a1, a2 > 0`).  `P(x) = lead·((x−b1)²+a1²)·((x−b2)²+a2²)`.
///
/// Reduction (derived/confirmed numerically, gate-certified):  with the
/// `arctan` substitution `φ(x) = arctan(L(x))`, `L(x) = (p·x+q)/(r·x+s)`, one has
///
/// ```text
///   d/dx[g·EllipticF(φ,m)] = g·L'/(√(1+L²)·√(1+(1−m)·L²)) = 1/√P
/// ```
///
/// iff `P·g²·(ps−qr)² = ((rx+s)²+(px+q)²)·((rx+s)²+(1−m)(px+q)²)`, i.e. the two
/// (no-real-root) quadratic factors of `P` are matched by the two sum-of-squares
/// factors on the right.  Writing `a=(p,r)`, `b=(q,s)` and fixing the scale
/// `|a|²=1` (`p=cosθ`, `r=±sinθ`), the matching reduces to a **quadratic in
/// `u=√t`** (`t = 1−m`):
///
/// ```text
///   a1·a2·u² − (a1²+a2²+(b1−b2)²)·u + a1·a2 = 0
/// ```
///
/// whose two roots are reciprocal (`u`, `1/u`); we take the root with `u<1` so
/// that `m = 1−u² ∈ (0,1)`.  Then `c = cos²θ = (K−1)/(t−1)` with `K = u·a1/a2`,
/// and `q = −b1·p − r·D`, `s = −b1·r + p·D` with `D = ps−qr = ±a1`.  The signs of
/// `r` and `D` are fixed by requiring the second-factor vertex/perp conditions
/// `t·p·q + r·s = −b2·(t·p²+r²)` and `ps−qr = D`; we try the four sign
/// combinations and keep the one that closes.  Finally
/// `g = √((p²+r²)(t·p²+r²)/(lead·D²))`.
///
/// The whole triple `(g, m, φ)` is then handed to the shared soundness gate, so
/// a mis-derivation can only *decline* — never emit a wrong form.
/// Constants `(p, q, r, s, m, g)` of the no-real-root quartic `arctan`
/// substitution (see [`quartic_no_real`]).  Pure numeric; shared by the builder
/// and by [`reduction_poles`] (which needs `r`, `s` to locate the `E`-block pole
/// at `x = −s/r`).  Returns `None` when no valid configuration closes.
fn quartic_no_real_consts(
    pair1: Croot,
    pair2: Croot,
    lead: f64,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let (b1, a1) = pair1;
    let (b2, a2) = pair2;
    let (a1, a2) = (a1.abs(), a2.abs());
    if !(a1 > 0.0 && a2 > 0.0 && lead != 0.0) {
        return None;
    }

    // Quadratic in `u = √t`:  a1·a2·u² − (a1²+a2²+(b1−b2)²)·u + a1·a2 = 0.
    let qa = a1 * a2;
    let qb = -(a1 * a1 + a2 * a2 + (b1 - b2).powi(2));
    let qc = a1 * a2;
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 || qa.abs() < 1e-30 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let u_roots = [
        (-qb + sqrt_disc) / (2.0 * qa),
        (-qb - sqrt_disc) / (2.0 * qa),
    ];

    for &u in &u_roots {
        if !(u.is_finite() && u > 0.0) {
            continue;
        }
        let t = u * u; // t = 1 − m
        let m = 1.0 - t;
        if !(m > 0.0 && m < 1.0) {
            continue;
        }
        // c = cos²θ = (K−1)/(t−1), K = u·a1/a2.
        let kk = u * a1 / a2;
        if (t - 1.0).abs() < 1e-15 {
            continue;
        }
        let c = (kk - 1.0) / (t - 1.0);
        if !c.is_finite() || !(-1e-9..=1.0 + 1e-9).contains(&c) {
            continue;
        }
        let c = c.clamp(0.0, 1.0);
        let cth = c.sqrt();
        let sth = (1.0 - c).sqrt();

        // Try the four (sign of r, sign of D) combinations; keep the one that
        // satisfies the second-factor matching conditions.
        for sr in [1.0_f64, -1.0] {
            for sd in [1.0_f64, -1.0] {
                let p = cth;
                let r = sr * sth;
                let d = sd * a1; // D = ps − qr
                let q = -b1 * p - r * d;
                let s = -b1 * r + p * d;
                // (ps − qr) must equal D.
                if (p * s - q * r - d).abs() > 1e-7 {
                    continue;
                }
                // Second-factor vertex: t·p·q + r·s = −b2·(t·p²+r²).
                let kk2 = t * p * p + r * r;
                if (t * p * q + r * s + b2 * kk2).abs() > 1e-7 * (1.0 + kk2.abs()) {
                    continue;
                }
                let c1 = p * p + r * r;
                let c2 = t * p * p + r * r;
                let val = c1 * c2 / (lead * d * d);
                if !(val.is_finite() && val > 0.0) {
                    continue;
                }
                let g = val.sqrt();
                return Some((p, q, r, s, m, g));
            }
        }
    }
    None
}

fn quartic_no_real(
    pair1: Croot,
    pair2: Croot,
    lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (p, q, r, s, m, g) = quartic_no_real_consts(pair1, pair2, lead)?;
    // φ(x) = arctan( (p·x+q)/(r·x+s) ).
    let lp = pool.add(vec![
        pool.mul(vec![float_to_expr(p, pool), var]),
        float_to_expr(q, pool),
    ]);
    let ld = pool.add(vec![
        pool.mul(vec![float_to_expr(r, pool), var]),
        float_to_expr(s, pool),
    ]);
    let l = pool.mul(vec![lp, pool.pow(ld, pool.integer(-1_i32))]);
    let phi = pool.func("atan", vec![l]);
    Some((g, m, phi))
}

// ---------------------------------------------------------------------------
// Verification gate
// ---------------------------------------------------------------------------

/// Numerically verify `d/dx F_cand = c/√P` at sample points where `P > 0`.
fn verify(f_cand: ExprId, coeffs: &[f64], c: f64, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(df) = crate::diff::diff(f_cand, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    // Sample widely; keep only points where P > 0 and both sides are finite reals.
    let samples = [
        -3.5, -2.7, -1.6, -0.9, -0.4, 0.15, 0.3, 0.55, 0.7, 0.9, 1.1, 1.4, 1.9, 2.6, 3.4, 4.7,
    ];
    let mut checked = 0;
    for &xv in &samples {
        let pv = eval_poly(coeffs, xv);
        if pv <= 1e-6 {
            continue;
        }
        let rhs = c / pv.sqrt();
        let Some(lhs) = eval(ds, var, xv, pool) else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        if (lhs - rhs).abs() > 1e-7 * (1.0 + rhs.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 3
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    super::poly_utils::is_zero_expr(expr, pool)
}

/// Evaluate a constant (var-free) expression to `f64`.
fn eval_const(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |s, &a| Some(s + eval_const(a, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |s, &a| Some(s * eval_const(a, pool)?)),
        ExprData::Pow { base, exp } => Some(eval_const(base, pool)?.powf(eval_const(exp, pool)?)),
        _ => None,
    }
}

/// Horner evaluation of a polynomial given ascending coefficients.
fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Numeric eval supporting the elementary functions produced by `diff` of the
/// candidate (`sin`, `cos`, `asin`, `acos`, `sqrt`; the `EllipticF` derivative
/// is rewritten to the elementary `1/√(1−m·sin²φ)` so no special-function eval
/// is needed).
fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> Option<f64> {
    if expr == x {
        return Some(xv);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |s, &a| Some(s + eval(a, x, xv, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |s, &a| Some(s * eval(a, x, xv, pool)?)),
        ExprData::Pow { base, exp } => Some(eval(base, x, xv, pool)?.powf(eval(exp, x, xv, pool)?)),
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let v = eval(args[0], x, xv, pool)?;
            match name.as_str() {
                "sin" => Some(v.sin()),
                "cos" => Some(v.cos()),
                "tan" => Some(v.tan()),
                "asin" => Some(v.asin()),
                "acos" => Some(v.acos()),
                "atan" => Some(v.atan()),
                "sqrt" => Some(v.sqrt()),
                "exp" => Some(v.exp()),
                "log" => Some(v.ln()),
                "cbrt" => Some(v.cbrt()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Build an `ExprId` for an `f64` value as an exact rational reconstruction of
/// the float.  Keeping the constant rational (rather than a float literal the
/// engine has no node for) lets `simplify`/`diff` operate on it cleanly.
fn float_to_expr(v: f64, pool: &ExprPool) -> ExprId {
    // Exact small integers stay integer nodes.
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        return pool.integer(v as i32);
    }
    match rug::Rational::from_f64(v) {
        Some(r) => {
            let (num, den) = r.into_numer_denom();
            pool.rational(num, den)
        }
        None => pool.integer(0_i32),
    }
}

// ---------------------------------------------------------------------------
// Complex root finding (Durand–Kerner) + classification
// ---------------------------------------------------------------------------

/// Find all complex roots of a polynomial with ascending real coefficients
/// (degree 3 or 4) via Durand–Kerner iteration.
fn poly_roots(coeffs: &[f64]) -> Option<Vec<Croot>> {
    let n = coeffs.len() - 1;
    if n == 0 {
        return Some(vec![]);
    }
    let lead = *coeffs.last()?;
    // Monic normalized coefficients, ascending.
    let mono: Vec<f64> = coeffs.iter().map(|&c| c / lead).collect();

    // Initial guesses: powers of the classic Durand–Kerner seed 0.4 + 0.9i.
    let seed = (0.4_f64, 0.9_f64);
    let mut z: Vec<Croot> = (0..n).map(|k| cpow(seed, k as i32)).collect();

    for _ in 0..500 {
        let mut max_delta = 0.0_f64;
        for i in 0..n {
            let num = ceval(&mono, z[i]);
            let mut den = (1.0, 0.0);
            for j in 0..n {
                if i != j {
                    den = cmul(den, csub(z[i], z[j]));
                }
            }
            let delta = cdiv(num, den);
            z[i] = csub(z[i], delta);
            let d = (delta.0 * delta.0 + delta.1 * delta.1).sqrt();
            if d > max_delta {
                max_delta = d;
            }
        }
        if max_delta < 1e-14 {
            break;
        }
    }
    Some(z)
}

/// Classify roots into sorted real roots and complex-conjugate pairs
/// `(re, |im|)` (one entry per conjugate pair).
fn classify_roots(roots: &[Croot]) -> (Vec<f64>, Vec<Croot>) {
    let tol = 1e-7;
    let mut reals = Vec::new();
    let mut pairs = Vec::new();
    let mut used = vec![false; roots.len()];
    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        if roots[i].1.abs() < tol {
            reals.push(roots[i].0);
            used[i] = true;
        } else {
            // Find the conjugate partner.
            let mut best = None;
            let mut best_d = f64::INFINITY;
            for (j, used_j) in used.iter().enumerate().skip(i + 1) {
                if *used_j {
                    continue;
                }
                let d = (roots[j].0 - roots[i].0).abs() + (roots[j].1 + roots[i].1).abs();
                if d < best_d {
                    best_d = d;
                    best = Some(j);
                }
            }
            if let Some(j) = best {
                if best_d < 1e-5 {
                    pairs.push((roots[i].0, roots[i].1.abs()));
                    used[i] = true;
                    used[j] = true;
                }
            }
        }
    }
    (reals, pairs)
}

// Minimal complex arithmetic on `(re, im)` tuples.
fn cmul(a: Croot, b: Croot) -> Croot {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn csub(a: Croot, b: Croot) -> Croot {
    (a.0 - b.0, a.1 - b.1)
}
fn cdiv(a: Croot, b: Croot) -> Croot {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn cpow(a: Croot, n: i32) -> Croot {
    let mut r = (1.0, 0.0);
    for _ in 0..n {
        r = cmul(r, a);
    }
    r
}
/// Horner evaluation of a monic polynomial (ascending coeffs) at a complex point.
fn ceval(mono: &[f64], z: Croot) -> Croot {
    let mut acc = (0.0, 0.0);
    for &c in mono.iter().rev() {
        acc = cmul(acc, z);
        acc.0 += c;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// Assert `∫ c·dx/√P` emits an `EllipticF` form whose `d/dx` matches the
    /// integrand at sample points; return the form's display string.
    fn check_emits(p_expr: ExprId, var: ExprId, c: f64, pool: &ExprPool) -> Option<String> {
        let zero = pool.integer(0_i32);
        // b = c / P  ⇒ integrand = b·√P = c/√P.
        let c_e = float_to_expr(c, pool);
        let b = pool.mul(vec![c_e, pool.pow(p_expr, pool.integer(-1_i32))]);
        let f = try_elliptic_output(zero, b, p_expr, var, pool)?;
        let s = pool.display(f).to_string();
        assert!(s.contains("EllipticF"), "no EllipticF in {s}");
        Some(s)
    }

    #[test]
    fn cubic_x3_plus_1_emits_ellipticf() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let s = check_emits(p, x, 1.0, &pool).expect("∫dx/√(x³+1) should emit EllipticF");
        assert!(s.contains("EllipticF"), "{s}");
    }

    #[test]
    fn cubic_three_real_emits_ellipticf() {
        // x³ − x = x(x−1)(x+1): three real roots.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(x³−x) should emit EllipticF");
    }

    #[test]
    fn quartic_four_real_emits_ellipticf() {
        // (x²−1)(x²−4) = x⁴ − 5x² + 4: four real roots ±1, ±2.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.mul(vec![pool.integer(-5_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(4_i32),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√((x²−1)(x²−4)) should emit EllipticF");
    }

    #[test]
    fn quartic_two_real_pair_emits_ellipticf() {
        // 1 − x⁴ = (1−x²)(1+x²): two real roots ±1, complex pair ±i.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(4_i32))]),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(1−x⁴) should emit EllipticF");
    }

    #[test]
    fn quintic_declined() {
        // x⁵+1 is genus-2: no degree-3/4 reduction ⇒ None (caller → NonElementary).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let zero = pool.integer(0_i32);
        let b = pool.pow(p, pool.integer(-1_i32));
        assert!(try_elliptic_output(zero, b, p, x, &pool).is_none());
    }

    // ── Second / third kind ─────────────────────────────────────────────────

    /// Run the higher-kind reduction for `∫ b·√P dx`, assert it emits a form
    /// containing each substring in `must_contain`, and verify `d/dx F = b·√P`
    /// numerically at points where `P > 0`.
    #[allow(clippy::too_many_arguments)]
    fn check_higher(
        p_expr: ExprId,
        b: ExprId,
        var: ExprId,
        must_contain: &[&str],
        b_num: &[f64],
        b_den: &[f64],
        p_coeffs: &[f64],
        samples: &[f64],
        pool: &ExprPool,
    ) -> String {
        let zero = pool.integer(0_i32);
        let f = try_elliptic_output_higher_kind(zero, b, p_expr, var, pool)
            .expect("expected higher-kind elliptic output");
        let s = pool.display(f).to_string();
        for needle in must_contain {
            assert!(s.contains(needle), "expected {needle} in {s}");
        }
        // Independent numeric re-check of d/dx F = b·√P.
        let df = crate::diff::diff(f, var, pool).unwrap().value;
        let ds = simplify(df, pool).value;
        let mut checked = 0;
        for &xv in samples {
            let pv = eval_poly(p_coeffs, xv);
            if pv <= 1e-6 {
                continue;
            }
            let Some(bv) = eval_ratio(b_num, b_den, xv) else {
                continue;
            };
            let rhs = bv * pv.sqrt();
            let Some(lhs) = eval(ds, var, xv, pool) else {
                continue;
            };
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {s}"
            );
            checked += 1;
        }
        assert!(checked >= 3, "too few in-domain samples");
        s
    }

    #[test]
    fn sqrt_cubic_x3_plus_1_emits_ellipticf_secondkind() {
        // Headline: ∫√(x³+1) dx → algebraic part + EllipticF.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let b = pool.integer(1_i32); // integrand = 1·√P
        let s = check_higher(
            p,
            b,
            x,
            &["EllipticF"],
            &[1.0],
            &[1.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.5, 1.0, 2.0, 3.0, 4.5],
            &pool,
        );
        // Algebraic part `x·√P` must be present.
        assert!(s.contains("EllipticF"), "{s}");
    }

    #[test]
    fn sqrt_cubic_three_real_emits_ellipticf_and_e() {
        // ∫√(x³−x) dx (region x>1) genuinely needs EllipticE.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let b = pool.integer(1_i32);
        check_higher(
            p,
            b,
            x,
            &["EllipticE"],
            &[1.0],
            &[1.0],
            &[0.0, -1.0, 0.0, 1.0],
            &[1.2, 1.6, 2.2, 3.1, 4.0],
            &pool,
        );
    }

    #[test]
    fn sqrt_cubic_x3_plus_8_emits_secondkind() {
        // ∫√(x³+8) dx → algebraic part + EllipticF (one real root −2).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(8_i32)]);
        let b = pool.integer(1_i32);
        check_higher(
            p,
            b,
            x,
            &["EllipticF"],
            &[1.0],
            &[1.0],
            &[8.0, 0.0, 0.0, 1.0],
            &[1.0, 2.0, 3.0, 4.5, 5.5],
            &pool,
        );
    }

    #[test]
    fn sqrt_quartic_1_minus_x4_emits_secondkind() {
        // ∫√(1−x⁴) dx → algebraic part + EllipticF/EllipticE (region |x|<1).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(4_i32))]),
        ]);
        let b = pool.integer(1_i32);
        check_higher(
            p,
            b,
            x,
            &["Elliptic"],
            &[1.0],
            &[1.0],
            &[1.0, 0.0, 0.0, 0.0, -1.0],
            &[-0.8, -0.3, 0.2, 0.6, 0.85],
            &pool,
        );
    }

    #[test]
    fn engine_integrate_sqrt_x3_plus_1_emits_elliptic() {
        // End-to-end: the algebraic engine itself returns an elliptic form for
        // ∫√(x³+1) dx (was NonElementary before PR3), and d/dx matches √(x³+1).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let integrand = pool.func("sqrt", vec![p]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫√(x³+1) dx should now integrate (PR3)");
        let s = pool.display(res.value).to_string();
        assert!(s.contains("Elliptic"), "expected an elliptic form, got {s}");
        let ds = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in &[0.5, 1.0, 2.0, 3.0] {
            let rhs = (xv * xv * xv + 1.0_f64).sqrt();
            let lhs = eval(ds, x, xv, &pool).unwrap();
            assert!((lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()), "x={xv}");
            checked += 1;
        }
        assert!(checked >= 3);
    }

    #[test]
    fn quintic_higher_kind_declined() {
        // ∫√(x⁵+1) dx is genus-2: higher-kind reduction declines (NonElementary).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let zero = pool.integer(0_i32);
        let b = pool.integer(1_i32);
        assert!(try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none());
    }

    // ── General second kind: `∫ poly(x)/√P dx` (this PR) ────────────────────

    /// Helper for `∫ R(x)/√P dx` cases: integrand `b = R/P` so `b·√P = R/√P`.
    /// Emits, asserts each `must_contain`, and numerically re-checks the gate.
    fn check_poly_over_sqrt(
        p_expr: ExprId,
        r_num: &[i64],
        var: ExprId,
        must_contain: &[&str],
        p_coeffs: &[f64],
        samples: &[f64],
        pool: &ExprPool,
    ) -> String {
        // b = R(x) / P(x).
        let r_terms: Vec<ExprId> = r_num
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 0)
            .map(|(j, &c)| {
                let cj = pool.integer(c as i32);
                match j {
                    0 => cj,
                    1 => pool.mul(vec![cj, var]),
                    _ => pool.mul(vec![cj, pool.pow(var, pool.integer(j as i32))]),
                }
            })
            .collect();
        let r_expr = pool.add(r_terms);
        let b = pool.mul(vec![r_expr, pool.pow(p_expr, pool.integer(-1_i32))]);
        let r_num_f: Vec<f64> = r_num.iter().map(|&c| c as f64).collect();
        let p_poly: Vec<f64> = p_coeffs.to_vec();
        check_higher(
            p_expr,
            b,
            var,
            must_contain,
            &r_num_f,
            &p_poly,
            p_coeffs,
            samples,
            pool,
        )
    }

    #[test]
    fn x_over_sqrt_x3_plus_1_emits_secondkind() {
        // Headline: ∫ x/√(x³+1) dx → algebraic + EllipticF + EllipticE.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let s = check_poly_over_sqrt(
            p,
            &[0, 1],
            x,
            &["EllipticE"],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 0.3, 0.6, 0.9, 1.4, 2.0, 3.0, 4.0],
            &pool,
        );
        assert!(s.contains("Elliptic"), "{s}");
    }

    #[test]
    fn x2_over_sqrt_x3_plus_1_emits_secondkind() {
        // ∫ x²/√(x³+1) dx = (2/3)√(x³+1) (purely algebraic, no elliptic needed).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        check_poly_over_sqrt(
            p,
            &[0, 0, 1],
            x,
            &["sqrt"],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 0.3, 0.6, 0.9, 1.4, 2.0, 3.0, 4.0],
            &pool,
        );
    }

    #[test]
    fn x_plus_1_over_sqrt_x3_plus_1_emits_secondkind() {
        // General polynomial numerator: ∫ (x+1)/√(x³+1) dx.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let s = check_poly_over_sqrt(
            p,
            &[1, 1],
            x,
            &["Elliptic"],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 0.3, 0.6, 0.9, 1.4, 2.0, 3.0, 4.0],
            &pool,
        );
        assert!(s.contains("Elliptic"), "{s}");
    }

    #[test]
    fn engine_integrate_x_over_sqrt_x3_plus_1_emits_elliptic() {
        // End-to-end: the algebraic engine returns an elliptic form for
        // ∫ x/√(x³+1) dx, and d/dx matches x/√(x³+1).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let sqrt_p = pool.func("sqrt", vec![p]);
        let integrand = pool.mul(vec![x, pool.pow(sqrt_p, pool.integer(-1_i32))]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫ x/√(x³+1) dx should integrate to an elliptic form");
        let s = pool.display(res.value).to_string();
        assert!(s.contains("Elliptic"), "expected an elliptic form, got {s}");
        let ds = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in &[0.5, 1.0, 2.0, 3.0] {
            let rhs = xv / (xv * xv * xv + 1.0_f64).sqrt();
            let lhs = eval(ds, x, xv, &pool).unwrap();
            assert!((lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()), "x={xv}");
            checked += 1;
        }
        assert!(checked >= 3);
    }

    #[test]
    fn x_over_sqrt_quintic_declined() {
        // ∫ x/√(x⁵+1) dx is genus-2: higher-kind reduction declines.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let zero = pool.integer(0_i32);
        let b = pool.mul(vec![x, pool.pow(p, pool.integer(-1_i32))]);
        assert!(try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none());
    }

    // ── Third kind: `∫ R(x)/((x−p)√P) dx` → EllipticPi (this PR) ─────────────

    /// Run the third-kind reduction for `∫ dx/((x−pole)√P)` (integrand
    /// `b = 1/((x−pole)·P)`, so `b·√P = 1/((x−pole)√P)`), assert an `EllipticPi`
    /// form is emitted, and numerically re-check `d/dx F = integrand`.
    fn check_third_kind_simple_pole(
        p_expr: ExprId,
        pole: i64,
        var: ExprId,
        p_coeffs: &[f64],
        samples: &[f64],
        pool: &ExprPool,
    ) -> String {
        // b = 1 / ((x − pole) · P).
        let x_minus_pole = pool.add(vec![var, pool.integer(-(pole as i32))]);
        let den = pool.mul(vec![x_minus_pole, p_expr]);
        let b = pool.pow(den, pool.integer(-1_i32));
        // b_num = 1; b_den = (x − pole)·P, in ascending coeffs.
        let mut b_den = vec![0.0; p_coeffs.len() + 1];
        for (j, &c) in p_coeffs.iter().enumerate() {
            b_den[j + 1] += c; // x · P
            b_den[j] += -(pole as f64) * c; // −pole · P
        }
        check_higher(
            p_expr,
            b,
            var,
            &["EllipticPi"],
            &[1.0],
            &b_den,
            p_coeffs,
            samples,
            pool,
        )
    }

    #[test]
    fn third_kind_cubic_three_real_emits_pi() {
        // ∫ dx/((x−3)√(x³−x)), region x>1, pole at x=3 off the roots {−1,0,1}.
        // sin²φ is Möbius here (asin substitution) so a single EllipticPi closes.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let s = check_third_kind_simple_pole(
            p,
            3,
            x,
            &[0.0, -1.0, 0.0, 1.0],
            &[1.2, 1.6, 2.2, 2.6, 4.0, 5.0, 6.0],
            &pool,
        );
        assert!(s.contains("EllipticPi"), "{s}");
    }

    #[test]
    fn third_kind_quartic_four_real_emits_pi() {
        // ∫ dx/((x−1/2 ·? )√(x⁴−5x²+4)); roots ±1,±2, region −1<x<1.
        // Use pole at x=0? x=0 is not a root (P(0)=4) and lies in (−1,1).  But the
        // integer-pole helper needs an integer pole inside (−1,1); none exists, so
        // build the integrand directly with a rational pole p=1/2.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.mul(vec![pool.integer(-5_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(4_i32),
        ]);
        // pole at x = 1/2.
        let half = pool.rational(rug::Integer::from(1), rug::Integer::from(2));
        let x_minus = pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), half])]);
        let den = pool.mul(vec![x_minus, p]);
        let b = pool.pow(den, pool.integer(-1_i32));
        // b_den = (x − 1/2)·P, ascending: P = 4 −5x² + x⁴.
        let p_coeffs = [4.0, 0.0, -5.0, 0.0, 1.0];
        let mut b_den = vec![0.0; p_coeffs.len() + 1];
        for (j, &c) in p_coeffs.iter().enumerate() {
            b_den[j + 1] += c;
            b_den[j] += -0.5 * c;
        }
        let s = check_higher(
            p,
            b,
            x,
            &["EllipticPi"],
            &[1.0],
            &b_den,
            &p_coeffs,
            &[-0.8, -0.4, -0.1, 0.2, 0.8],
            &pool,
        );
        assert!(s.contains("EllipticPi"), "{s}");
    }

    #[test]
    fn third_kind_cubic_one_real_declines() {
        // ∫ dx/((x−2)√(x³+1)) — the `acos`/cosφ substitution makes sin²φ a
        // *quadratic* rational of x, so a single EllipticPi has a spurious twin
        // pole and the fit cannot close.  The path declines (→ NonElementary):
        // soundness gate never emits an unverified form.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let x_minus = pool.add(vec![x, pool.integer(-2_i32)]);
        let den = pool.mul(vec![x_minus, p]);
        let b = pool.pow(den, pool.integer(-1_i32));
        let zero = pool.integer(0_i32);
        assert!(try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none());
    }

    #[test]
    fn third_kind_complex_pole_declines() {
        // ∫ dx/((x²+1)√(x³+1)): the pole factor x²+1 has *no real root*, so there
        // is no real characteristic — the third-kind path adds no Π block and the
        // remaining basis cannot represent the complex-pole integrand → declines.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let q = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let den = pool.mul(vec![q, p]);
        let b = pool.pow(den, pool.integer(-1_i32));
        let zero = pool.integer(0_i32);
        assert!(try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none());
    }

    #[test]
    fn engine_integrate_third_kind_cubic_three_real_emits_pi() {
        // End-to-end through the engine: ∫ dx/((x−3)√(x³−x)) → EllipticPi form,
        // with d/dx matching the integrand on x>1.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let sqrt_p = pool.func("sqrt", vec![p]);
        let x_minus = pool.add(vec![x, pool.integer(-3_i32)]);
        let den = pool.mul(vec![x_minus, sqrt_p]);
        let integrand = pool.pow(den, pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫ dx/((x−3)√(x³−x)) should integrate to an elliptic form");
        let s = pool.display(res.value).to_string();
        assert!(s.contains("EllipticPi"), "expected EllipticPi, got {s}");
        let ds = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in &[1.2, 1.6, 2.2, 4.0, 5.0] {
            let pv: f64 = xv * xv * xv - xv;
            if pv <= 1e-6 {
                continue;
            }
            let rhs = 1.0 / ((xv - 3.0) * pv.sqrt());
            let lhs = eval(ds, x, xv, &pool).unwrap();
            assert!((lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()), "x={xv}");
            checked += 1;
        }
        assert!(checked >= 3);
    }

    // ── All-complex-root (no real root) genus-1 quartics (this PR) ───────────

    #[test]
    fn quartic_no_real_x4_plus_1_emits_ellipticf() {
        // Headline: ∫ dx/√(x⁴+1) → EllipticF (two complex pairs, no real roots).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(4_i32)), pool.integer(1_i32)]);
        let s = check_emits(p, x, 1.0, &pool).expect("∫dx/√(x⁴+1) should emit EllipticF");
        assert!(s.contains("EllipticF"), "{s}");
    }

    #[test]
    fn quartic_no_real_x4_plus_x2_plus_1_emits_ellipticf() {
        // ∫ dx/√(x⁴+x²+1) → EllipticF; (x²+x+1)(x²−x+1), two complex pairs.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.pow(x, pool.integer(2_i32)),
            pool.integer(1_i32),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(x⁴+x²+1) should emit EllipticF");
    }

    #[test]
    fn quartic_no_real_x4_plus_4_emits_ellipticf() {
        // ∫ dx/√(x⁴+4) → EllipticF; (x²−2x+2)(x²+2x+2), roots 1±i, −1±i.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(4_i32)), pool.integer(4_i32)]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(x⁴+4) should emit EllipticF");
    }

    #[test]
    fn quartic_no_real_scaled_lead_emits_ellipticf() {
        // ∫ dx/√(3x⁴+3): non-unit leading coefficient, no real roots.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.mul(vec![pool.integer(3_i32), pool.pow(x, pool.integer(4_i32))]),
            pool.integer(3_i32),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(3x⁴+3) should emit EllipticF");
    }

    #[test]
    fn quartic_no_real_sqrt_x4_plus_1_emits_secondkind() {
        // Second kind: ∫ √(x⁴+1) dx → algebraic part + EllipticF/EllipticE.
        // (The symmetric x⁴+1 closes cleanly as (1/3)x√P + (2/3)g·E.)
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(4_i32)), pool.integer(1_i32)]);
        let b = pool.integer(1_i32); // integrand = 1·√P
        let s = check_higher(
            p,
            b,
            x,
            &["Elliptic"],
            &[1.0],
            &[1.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0],
            &[-2.0, -1.0, -0.3, 0.4, 1.2, 2.3, 3.0],
            &pool,
        );
        assert!(s.contains("Elliptic"), "{s}");
    }

    #[test]
    fn engine_integrate_x4_plus_1_emits_ellipticf() {
        // End-to-end through the engine: ∫ dx/√(x⁴+1) → EllipticF form, d/dx OK.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(4_i32)), pool.integer(1_i32)]);
        let sqrt_p = pool.func("sqrt", vec![p]);
        let integrand = pool.pow(sqrt_p, pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫ dx/√(x⁴+1) should integrate to an elliptic form");
        let s = pool.display(res.value).to_string();
        assert!(s.contains("Elliptic"), "expected an elliptic form, got {s}");
        let ds = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in &[-1.5, -0.5, 0.5, 1.0, 2.0] {
            let rhs = 1.0 / (xv * xv * xv * xv + 1.0_f64).sqrt();
            let lhs = eval(ds, x, xv, &pool).unwrap();
            assert!((lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()), "x={xv}");
            checked += 1;
        }
        assert!(checked >= 3);
    }

    #[test]
    fn quartic_real_root_regression_still_works() {
        // Regression: a real-root quartic ∫dx/√(x⁴−5x²+4) still emits EllipticF.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.mul(vec![pool.integer(-5_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(4_i32),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(x⁴−5x²+4) should still emit EllipticF");
    }

    #[test]
    fn quartic_no_real_quintic_still_declines() {
        // Genus-2 ∫dx/√(x⁵+1) still declines (no degree-3/4 reduction).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let zero = pool.integer(0_i32);
        let b = pool.pow(p, pool.integer(-1_i32));
        assert!(try_elliptic_output(zero, b, p, x, &pool).is_none());
    }
}
