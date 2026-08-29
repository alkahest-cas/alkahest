//! `∫_{-∞}^{∞} P(x)/Q(x) dx` by the residue theorem — exactly, or not at all.
//!
//! # What this computes
//!
//! For a rational integrand the residue theorem gives
//!
//! ```text
//!   ∫_{-∞}^{∞} P(x)/Q(x) dx = 2πi · Σ_{α : Im α > 0} Res(P/Q, α)
//! ```
//!
//! subject to two hypotheses that this module **checks rather than assumes**:
//!
//! 1. `deg Q ≥ deg P + 2` — otherwise the integrand decays no faster than
//!    `c/x` and the integral diverges (`∫ x dx/(x²+1)` has a Cauchy principal
//!    value of `0` but is *not* convergent).
//! 2. `Q` has **no real root** — otherwise the integrand has a pole on the
//!    contour and the integral diverges or exists only as a principal value.
//!
//! Both are decided exactly: the first on the reduced degrees, the second by
//! [`crate::poly::real_roots`] (VAS isolation over ℤ, complete). A failure of
//! either is reported as *divergent*, never as a number.
//!
//! # Why not the existing `poly::residue` primitive
//!
//! [`crate::poly::residue::residue`] computes residues at a point of **ℚ(i)**.
//! The poles that matter here are usually not in ℚ(i): those of `1/(x⁴+1)` are
//! primitive 8th roots of unity. A per-pole loop over that primitive would
//! handle `1/(x²+1)` and miss `1/(x⁴+1)` — precisely the integral that was
//! returning `0`. The route below never enumerates individual poles.
//!
//! # The route actually taken
//!
//! Selecting "the poles in the upper half-plane" is a *semi-algebraic*
//! condition on the roots, so no purely rational symmetric-function identity
//! can express the answer. (Concretely: `Σ_{Im α>0} Res` is `iπ⁻¹` times
//! `Σ_α sgn(Im α)·Res_α`, and the `sgn` cannot be removed.) Certified complex
//! root isolation does not exist in this crate — `real/` isolates real roots
//! and `ball/` does interval arithmetic, neither separates a complex root set
//! by half-plane — so a "isolate the roots, assign half-planes, sum over a
//! number field" design would have meant writing that machinery first.
//!
//! Instead the half-plane split is pushed into a **spectral (Hurwitz)
//! factorisation**, where it is a *polynomial* factorisation problem and
//! sometimes solvable over ℚ:
//!
//! * **Normalise to an even denominator.** If `Q` is not even, multiply top and
//!   bottom by `Q(-x)`: `D = Q(x)·Q(-x)` is even and still has no real root,
//!   `N = P(x)·Q(-x)`, and the odd part of `N` integrates to `0` over a
//!   symmetric interval (the integral converges absolutely), so only the even
//!   part `M` survives. This step is exact and costs only degree.
//!   Evenness is what makes the spectral factor `A` **real**: `A`'s roots are
//!   `{iα : Im α > 0}`, and that set is closed under conjugation exactly when
//!   `D(-x) = D(x)`.
//! * **Rotate to the imaginary axis.** `Ď(s) := D(s/i)`, a rational polynomial,
//!   even, with no root on the imaginary axis. Its roots in the open *left*
//!   half-plane correspond one-to-one with the poles of the integrand in the
//!   *upper* half-plane.
//! * **Factor `Ď = A(s)·A(-s)` with `A` monic Hurwitz**, then solve the small
//!   rational linear system `M̌(s) = d·[G(s)A(-s) + G(-s)A(s)]` for `G` of
//!   degree `< deg A`. Contour-closing in the right half-plane gives
//!
//!   ```text
//!       ∫_{-∞}^{∞} M/D dx = 2π · (leading coefficient of G)
//!   ```
//!
//!   which is an element of `ℚ·π`.
//!
//! `A` is found by factoring `Ď` over ℚ and sorting the irreducible factors
//! into Hurwitz / anti-Hurwitz by an exact Routh array. That works whenever `A`
//! itself is rational (`x²+1`, `x²+4`, `(x²+1)²`, `x⁶+1`, every quadratic
//! denominator after even-normalisation, …) and provably cannot when it is not
//! — `x⁴+1` needs `A = s² + √2·s + 1`.
//!
//! For `deg D ≤ 4` that gap is closed by a closed form in radicals, obtained by
//! the substitution `u = x²`:
//!
//! ```text
//!   ∫_{-∞}^{∞} M(x)/D(x) dx = π · Σ_β Res_{u=β}[ M̃(u)/D̃(u) · (-u)^{-1/2} ]
//! ```
//!
//! (`D̃(x²) = D(x)`, principal branch — legitimate because `D̃` has no root on
//! `[0, ∞)`). Evaluating that for `deg D̃ ≤ 2` gives, with
//! `W = √(d₀/d₄)` and `S = √(d₂/d₄ + 2W)`,
//!
//! ```text
//!   ∫ (m₂x² + m₀)/(d₄x⁴ + d₂x² + d₀) dx = π·(m₀ + m₂W) / (d₄·S·W)
//!   ∫  m₀/(d₂x² + d₀) dx                = π·m₀ / √(d₂·d₀)
//! ```
//!
//! which covers `1/(x⁴+1)`, `x²/(x⁴+1)`, `1/(x⁴+x²+1)`, `1/(x²+1)²` and every
//! quadratic denominator, and reaches into `ℚ(√·)·π`.
//!
//! Everything else declines, explicitly.
//!
//! # Nothing is returned unverified
//!
//! Every value produced above is cross-checked against a **rigorous enclosure
//! of the true integral** before it is returned ([`enclose_line_integral`]).
//! The whole real line is covered without any truncation error: `[-1, 1]`
//! directly, and each tail through the exact change of variable `x = ±1/t`,
//! which turns `∫_1^{∞} P/Q dx` into `∫_0^1 t^{q-p-2}·P*(t)/Q*(t) dt` — again a
//! rational integrand, regular on `[0, 1]` under the two hypotheses above.
//! All three pieces go through [`crate::validated::bounds::verified_integral`]
//! (adaptive Taylor models in outward-rounded ball arithmetic), so their sum is
//! a sound outer bound. A candidate outside that bound is a **bug**, and is
//! reported as a decline rather than returned.
//!
//! If the enclosure cannot be established at all, the candidate is declined
//! too. There is no path through this module that emits an unchecked number.

use rug::{Integer, Rational};

use crate::integrate::risch::poly_rde::{
    degree, poly_add, poly_mul, poly_one, poly_scale, qpoly_to_expr, rational_to_expr, trim, QPoly,
};
use crate::integrate::risch::rational_rde::{expr_to_qrational, poly_div_exact, poly_gcd};
use crate::kernel::{ExprId, ExprPool};

/// Largest denominator degree accepted. The spectral route factors a
/// polynomial of twice this degree and solves a dense rational system of that
/// size; the cap keeps a pathological input from turning into a hang.
const MAX_DENOMINATOR_DEGREE: i64 = 24;

/// Relative slack allowed between the exact candidate's `f64` evaluation and
/// the rigorous enclosure. The enclosure is sound; this only absorbs the
/// rounding of evaluating the closed form in double precision, which is
/// ~1e-16 relative. Anything larger is a genuine disagreement.
const VERIFY_SLACK: f64 = 1e-9;

/// Outcome of the residue route.
#[derive(Debug, Clone)]
pub(crate) enum LineIntegral {
    /// An exact closed form, already cross-checked against a rigorous
    /// enclosure of the integral.
    Value {
        /// The exact value.
        value: ExprId,
        /// The rigorous enclosure it was checked against, for the log.
        enclosure: (f64, f64),
    },
    /// The integral provably does not converge. Never a number.
    Divergent(String),
    /// Outside this route's scope; the caller should try something else.
    OutOfScope(String),
}

/// `∫_{-∞}^{∞} integrand d(var)` by the residue theorem, when the integrand is
/// a rational function of `var`.
pub(crate) fn integrate_rational_over_real_line(
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> LineIntegral {
    // Read the integrand as `P/Q` over ℚ. `expr_to_qrational` is
    // spelling-robust but not simplification-robust, so a form it cannot parse
    // is retried once through the simplifier rather than declined outright.
    let Some((num, den)) = expr_to_qrational(integrand, var, pool).or_else(|| {
        let normalised = crate::simplify::engine::simplify(integrand, pool).value;
        (normalised != integrand)
            .then(|| expr_to_qrational(normalised, var, pool))
            .flatten()
    }) else {
        return LineIntegral::OutOfScope(
            "the integrand is not a rational function of the integration variable".into(),
        );
    };
    let mut num = trim(num);
    let mut den = trim(den);
    if den.is_empty() {
        return LineIntegral::OutOfScope("the denominator is identically zero".into());
    }
    if num.is_empty() {
        return LineIntegral::Value {
            value: pool.integer(0_i32),
            enclosure: (0.0, 0.0),
        };
    }
    // Cancel common factors first: the convergence conditions below are about
    // the *reduced* fraction. `(x²-1)/((x-1)(x⁴+1))` has no real pole once the
    // removable one is divided out.
    let g = poly_gcd(&num, &den);
    if degree(&g) > 0 {
        num = poly_div_exact(&num, &g);
        den = poly_div_exact(&den, &g);
    }

    let p = degree(&num);
    let q = degree(&den);
    if q > MAX_DENOMINATOR_DEGREE {
        return LineIntegral::OutOfScope(format!(
            "denominator degree {q} exceeds the residue route's cap of {MAX_DENOMINATOR_DEGREE}"
        ));
    }

    // --- Convergence condition 1: decay. -----------------------------------
    // `|P/Q| ~ c·|x|^{p-q}` at infinity, so the integral converges absolutely
    // iff `q - p ≥ 2`. `q - p == 1` diverges logarithmically (a principal value
    // may still exist); `q - p ≤ 0` diverges outright.
    if q - p < 2 {
        return LineIntegral::Divergent(format!(
            "∫_{{-∞}}^{{∞}} of a rational function converges only when \
             deg(denominator) ≥ deg(numerator) + 2, but here deg = {q} and {p}: the integrand \
             decays like x^({}) at infinity, so the integral diverges (a Cauchy principal value \
             may still exist, but it is not the integral)",
            p - q
        ));
    }

    // --- Convergence condition 2: no pole on the contour. ------------------
    let den_expr = qpoly_to_expr(&den, var, pool);
    let Ok(den_uni) = crate::poly::UniPoly::from_symbolic_clear_denoms(den_expr, var, pool) else {
        return LineIntegral::OutOfScope(
            "the denominator could not be converted to an integer polynomial for root isolation"
                .into(),
        );
    };
    let Ok(roots) = crate::poly::real_roots(&den_uni) else {
        return LineIntegral::OutOfScope("real-root isolation of the denominator failed".into());
    };
    if let Some(first) = roots.first() {
        return LineIntegral::Divergent(format!(
            "the denominator has a real root (isolated in [{}, {}]{}), so the integrand has a \
             pole on the integration contour: the integral diverges, or exists only as a Cauchy \
             principal value, which is not the same number",
            first.lo_f64(),
            first.hi_f64(),
            if roots.len() > 1 {
                format!(" and {} more", roots.len() - 1)
            } else {
                String::new()
            },
        ));
    }

    // A real polynomial with no real root has even degree and constant sign.
    // Normalise that sign to `+` so `D > 0` on ℝ below.
    let (num, den) = if den[den.len() - 1] < 0 {
        (
            poly_scale(&num, &Rational::from(-1)),
            poly_scale(&den, &Rational::from(-1)),
        )
    } else {
        (num, den)
    };

    // --- Normalise to an even denominator. ---------------------------------
    let (big_d, big_n) = if is_even_poly(&den) {
        (den.clone(), num.clone())
    } else {
        let den_reflected = reflect(&den);
        (
            poly_mul(&den, &den_reflected),
            poly_mul(&num, &den_reflected),
        )
    };
    // The odd part of the numerator integrates to zero against an even
    // denominator over a symmetric interval; the integral converges absolutely
    // so the split is legitimate.
    let big_m = even_part(&big_n);

    let two_m = degree(&big_d);
    if two_m % 2 != 0 {
        return LineIntegral::OutOfScope(
            "internal: the normalised denominator is not of even degree".into(),
        );
    }
    let m = (two_m / 2) as usize;
    if degree(&big_m) > two_m - 2 {
        return LineIntegral::OutOfScope(
            "internal: the normalised numerator degree exceeds deg(denominator) - 2".into(),
        );
    }

    let candidate = match m {
        0 => {
            return LineIntegral::OutOfScope("internal: normalised denominator is constant".into())
        }
        1 => quadratic_closed_form(&big_m, &big_d, pool),
        2 => quartic_closed_form(&big_m, &big_d, pool),
        _ => spectral_closed_form(&big_m, &big_d, m, var, pool),
    };
    let value = match candidate {
        Ok(v) => v,
        Err(why) => return LineIntegral::OutOfScope(why),
    };

    // --- Verification. Nothing below this line may be skipped. -------------
    let Some((lo, hi)) = enclose_line_integral(&num, &den, var, pool) else {
        return LineIntegral::OutOfScope(
            "the residue-theorem value could not be cross-checked against a rigorous numeric \
             enclosure of the integral, so it is not returned"
                .into(),
        );
    };
    let Some(numeric) = numeric_value(value, pool) else {
        return LineIntegral::OutOfScope(
            "the residue-theorem value could not be evaluated numerically for cross-checking"
                .into(),
        );
    };
    let slack = VERIFY_SLACK * (1.0 + numeric.abs());
    if !numeric.is_finite() || numeric < lo - slack || numeric > hi + slack {
        return LineIntegral::OutOfScope(format!(
            "the residue-theorem value {} ≈ {numeric} disagrees with the rigorous enclosure \
             [{lo}, {hi}] of the same integral. That is a bug in the residue route, not a \
             rounding artefact; declining rather than returning either number",
            pool.display(value),
        ));
    }
    LineIntegral::Value {
        value,
        enclosure: (lo, hi),
    }
}

// ---------------------------------------------------------------------------
// Polynomial helpers (ascending-degree `QPoly` over ℚ)
// ---------------------------------------------------------------------------

/// `p(-x)`.
fn reflect(p: &QPoly) -> QPoly {
    trim(
        p.iter()
            .enumerate()
            .map(|(i, c)| if i % 2 == 0 { c.clone() } else { -c.clone() })
            .collect(),
    )
}

/// True when every odd-degree coefficient vanishes.
fn is_even_poly(p: &QPoly) -> bool {
    p.iter().enumerate().all(|(i, c)| i % 2 == 0 || *c == 0)
}

/// The even-degree part of `p` (odd coefficients zeroed).
fn even_part(p: &QPoly) -> QPoly {
    trim(
        p.iter()
            .enumerate()
            .map(|(i, c)| {
                if i % 2 == 0 {
                    c.clone()
                } else {
                    Rational::from(0)
                }
            })
            .collect(),
    )
}

/// Coefficient of `x^k`, or zero.
fn coeff(p: &QPoly, k: usize) -> Rational {
    p.get(k).cloned().unwrap_or_else(|| Rational::from(0))
}

/// `f(s/i)` for an **even** polynomial `f`: `f(s/i) = Σ f_{2k}·(-1)^k·s^{2k}`.
///
/// This is the rotation that maps the real axis of `x` onto the imaginary axis
/// of `s`, so that "pole in the upper half `x`-plane" becomes "root in the open
/// left half `s`-plane".
fn rotate_even(p: &QPoly) -> QPoly {
    trim(
        p.iter()
            .enumerate()
            .map(|(i, c)| {
                if i % 2 != 0 {
                    Rational::from(0)
                } else if (i / 2) % 2 == 0 {
                    c.clone()
                } else {
                    -c.clone()
                }
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Exact square roots of rationals, kept pretty
// ---------------------------------------------------------------------------

/// `√r` for `r > 0`, as an exact expression with the largest recognisable
/// square factor pulled out (`√(9/2)` becomes `3·2^{-1/2}`, `√2` stays `2^{1/2}`).
///
/// The extraction is best-effort — it only removes square factors found by
/// trial division — which affects how the answer *prints*, never whether it is
/// correct.
fn sqrt_rational(r: &Rational, pool: &ExprPool) -> Option<ExprId> {
    if *r <= 0 {
        return None;
    }
    let n = r.numer().clone();
    let d = r.denom().clone();
    // √(n/d) = √(n·d)/d.
    let radicand = n * d.clone();
    let (outside, inside) = split_square_factor(&radicand);
    let outside = Rational::from((outside, d));
    let outside_expr = rational_to_expr(&outside, pool);
    if inside == 1 {
        return Some(outside_expr);
    }
    let root = pool.pow(pool.integer(inside), pool.rational(1, 2));
    Some(if outside == 1 {
        root
    } else {
        pool.mul(vec![outside_expr, root])
    })
}

/// Write `k = a²·b` with `b` as small as trial division can make it, and
/// return `(a, b)`. Correctness of the caller does not depend on `b` being
/// square-free — only the printed form does.
fn split_square_factor(k: &Integer) -> (Integer, Integer) {
    let mut rest = k.clone();
    let mut outside = Integer::from(1);
    let mut f = Integer::from(2);
    while f.clone() * f.clone() <= rest {
        let sq = f.clone() * f.clone();
        while rest.clone() % sq.clone() == 0 {
            rest /= sq.clone();
            outside *= f.clone();
        }
        f += 1;
        // Trial division is only a beautifier; stop before it costs anything.
        if f > 100_000 {
            break;
        }
    }
    (outside, rest)
}

// ---------------------------------------------------------------------------
// deg D = 2: ∫ m₀/(d₂x² + d₀) dx = π·m₀/√(d₂d₀)
// ---------------------------------------------------------------------------

fn quadratic_closed_form(m: &QPoly, d: &QPoly, pool: &ExprPool) -> Result<ExprId, String> {
    let m0 = coeff(m, 0);
    let d0 = coeff(d, 0);
    let d2 = coeff(d, 2);
    let radicand = d2 * d0;
    if radicand <= 0 {
        return Err(
            "internal: the normalised quadratic denominator is not positive definite".into(),
        );
    }
    let root = sqrt_rational(&radicand, pool)
        .ok_or_else(|| "internal: non-positive radicand".to_string())?;
    Ok(times_pi(
        pool.mul(vec![
            rational_to_expr(&m0, pool),
            pool.pow(root, pool.integer(-1_i32)),
        ]),
        pool,
    ))
}

// ---------------------------------------------------------------------------
// deg D = 4: ∫ (m₂x²+m₀)/(d₄x⁴+d₂x²+d₀) dx = π(m₀ + m₂W)/(d₄·S·W)
// ---------------------------------------------------------------------------

fn quartic_closed_form(m: &QPoly, d: &QPoly, pool: &ExprPool) -> Result<ExprId, String> {
    let m0 = coeff(m, 0);
    let m2 = coeff(m, 2);
    let d0 = coeff(d, 0);
    let d2 = coeff(d, 2);
    let d4 = coeff(d, 4);
    if d4 <= 0 || d0 <= 0 {
        return Err("internal: the normalised quartic denominator is not positive definite".into());
    }
    // W = √(d₀/d₄) — the product of the two principal roots `√(-β_j)`.
    // S = W₁ + W₂, with S² = d₂/d₄ + 2W; positive because both `W_j` have
    // positive real part, but checked rather than assumed.
    let w_sq = d0.clone() / d4.clone();
    // When `W` is itself rational the whole answer lives in `ℚ(√·)·π` with a
    // *single* radical, which is worth the special case: it is the difference
    // between printing `π/2` and printing `π·(4^{1/2})^{-1}`.
    if let Some(w) = exact_sqrt(&w_sq) {
        let s_sq = d2.clone() / d4.clone() + Rational::from(2) * w.clone();
        if s_sq <= 0 {
            return Err(format!(
                "internal: S² = {s_sq} is not positive, so the quartic spectral factor does not \
                 exist as expected"
            ));
        }
        let s =
            sqrt_rational(&s_sq, pool).ok_or_else(|| "internal: non-positive S²".to_string())?;
        // (m₀ + m₂W) / (d₄·W) — exactly rational here.
        let scale = (m0 + m2 * w.clone()) / (d4 * w);
        return Ok(times_pi(
            pool.mul(vec![
                rational_to_expr(&scale, pool),
                pool.pow(s, pool.integer(-1_i32)),
            ]),
            pool,
        ));
    }
    let w = sqrt_rational(&w_sq, pool).ok_or_else(|| "internal: non-positive W²".to_string())?;
    let s_sq_rat_part = d2.clone() / d4.clone();
    let s_sq = pool.add(vec![
        rational_to_expr(&s_sq_rat_part, pool),
        pool.mul(vec![pool.integer(2_i32), w]),
    ]);
    let s_sq_num = numeric_value(s_sq, pool)
        .ok_or_else(|| "internal: S² is not numerically evaluable".to_string())?;
    if s_sq_num <= 0.0 {
        return Err(format!(
            "internal: S² = {s_sq_num} is not positive, so the quartic spectral factor does not \
             exist as expected"
        ));
    }
    let s = pool.pow(s_sq, pool.rational(1, 2));
    // π·(m₀ + m₂W) / (d₄·S·W)
    let numerator = pool.add(vec![
        rational_to_expr(&m0, pool),
        pool.mul(vec![rational_to_expr(&m2, pool), w]),
    ]);
    let denominator = pool.mul(vec![rational_to_expr(&d4, pool), s, w]);
    Ok(times_pi(
        pool.mul(vec![numerator, pool.pow(denominator, pool.integer(-1_i32))]),
        pool,
    ))
}

/// `√r` when it is exactly rational, else `None`.
fn exact_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    let n = r.numer().clone();
    let d = r.denom().clone();
    let (rn, ok_n) = n.sqrt_rem(Integer::new());
    if ok_n != 0 {
        return None;
    }
    let (rd, ok_d) = d.sqrt_rem(Integer::new());
    if ok_d != 0 {
        return None;
    }
    Some(Rational::from((rn, rd)))
}

// ---------------------------------------------------------------------------
// deg D ≥ 6: rational spectral factorisation
// ---------------------------------------------------------------------------

fn spectral_closed_form(
    m: &QPoly,
    d: &QPoly,
    half_degree: usize,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, String> {
    // Ď(s) = D(s/i), M̌(s) = M(s/i). Both rational, both even in `s`.
    let d_hat = rotate_even(d);
    let m_hat = rotate_even(m);
    let lead = coeff(d, 2 * half_degree);

    // Target for the spectral factor: A(s)·A(-s) = Ď(s)/d_{2m}, `A` monic.
    let inv_lead = Rational::from(1) / lead.clone();
    let target = poly_scale(&d_hat, &inv_lead);
    let a = hurwitz_spectral_factor(&target, half_degree, var, pool)?;

    // Solve M̌(s) = d_{2m}·[G(s)A(-s) + G(-s)A(s)] = 2·d_{2m}·(Gₑ·Aₑ − G₀·A₀)
    // for `G` of degree ≤ m-1.
    let a_even = even_part(&a);
    let a_odd = trim(poly_sub_qq(&a, &a_even));
    let mut columns: Vec<QPoly> = Vec::with_capacity(half_degree);
    let two_lead = Rational::from(2) * lead.clone();
    for j in 0..half_degree {
        let mut shift = vec![Rational::from(0); j];
        shift.push(Rational::from(1));
        let base = if j % 2 == 0 {
            poly_mul(&shift, &a_even)
        } else {
            poly_scale(&poly_mul(&shift, &a_odd), &Rational::from(-1))
        };
        columns.push(poly_scale(&base, &two_lead));
    }
    // Match coefficients at s^0, s^2, …, s^{2m-2}.
    let mut mat: Vec<Vec<Rational>> = Vec::with_capacity(half_degree);
    let mut rhs: Vec<Rational> = Vec::with_capacity(half_degree);
    for row in 0..half_degree {
        let deg = 2 * row;
        mat.push((0..half_degree).map(|j| coeff(&columns[j], deg)).collect());
        rhs.push(coeff(&m_hat, deg));
    }
    let g = solve_exact(mat, rhs).ok_or_else(|| {
        "the spectral partial-fraction system has no rational solution".to_string()
    })?;

    // ∫ M/D dx = 2π · (leading coefficient of G).
    let g_lead = g[half_degree - 1].clone();
    let coefficient = Rational::from(2) * g_lead;
    Ok(times_pi(rational_to_expr(&coefficient, pool), pool))
}

/// The monic real Hurwitz `A` with `A(s)·A(-s) = target(s)`, when `A` is
/// **rational** — found by factoring `target` over ℚ and sorting the
/// irreducible factors by an exact Routh test.
///
/// A ℚ-irreducible factor of `target` divides `A` or `A(-s)` whole (ℚ[s] is a
/// UFD), so if `A` is rational this classification finds it; if some factor is
/// neither Hurwitz nor anti-Hurwitz then `A ∉ ℚ[s]` and there is nothing to
/// find. The reconstructed `A` is checked against `target` **exactly** before
/// it is used.
fn hurwitz_spectral_factor(
    target: &QPoly,
    half_degree: usize,
    var: ExprId,
    pool: &ExprPool,
) -> Result<QPoly, String> {
    let target_expr = qpoly_to_expr(target, var, pool);
    let uni = crate::poly::UniPoly::from_symbolic_clear_denoms(target_expr, var, pool)
        .map_err(|e| format!("the rotated denominator is not an integer polynomial: {e}"))?;
    let factorization = uni
        .factor_z()
        .map_err(|e| format!("the rotated denominator could not be factored over ℚ: {e}"))?;

    let mut a = poly_one();
    for (factor, multiplicity) in &factorization.factors {
        let coeffs: Vec<Rational> = factor
            .coefficients()
            .into_iter()
            .map(Rational::from)
            .collect();
        let f = trim(coeffs);
        if degree(&f) == 0 {
            continue;
        }
        match classify_half_plane(&f) {
            HalfPlane::Hurwitz => {
                for _ in 0..*multiplicity {
                    a = poly_mul(&a, &f);
                }
            }
            HalfPlane::AntiHurwitz => {}
            HalfPlane::Mixed => {
                return Err(format!(
                    "the spectral factorisation of the rotated denominator is not rational: the \
                     irreducible factor of degree {} has roots in both half-planes, so the \
                     Hurwitz factor needs an algebraic extension (deg ≤ 4 denominators are \
                     handled in radicals instead; this one is degree {})",
                    degree(&f),
                    2 * half_degree
                ));
            }
        }
    }
    // Make monic.
    let lead = coeff(&a, degree(&a).max(0) as usize);
    if lead == 0 {
        return Err("internal: degenerate spectral factor".into());
    }
    let a = poly_scale(&a, &(Rational::from(1) / lead));
    if degree(&a) != half_degree as i64 {
        return Err(format!(
            "the Hurwitz part has degree {} but half the rotated denominator's degree is \
             {half_degree}; the spectral factorisation is not rational",
            degree(&a)
        ));
    }
    // Exact check: A(s)·A(-s) must reproduce the target on the nose.
    let product = trim(poly_mul(&a, &reflect(&a)));
    if product != trim(target.clone()) {
        return Err(
            "internal: the reconstructed spectral factor does not reproduce the denominator".into(),
        );
    }
    Ok(a)
}

/// Which open half-plane an irreducible real polynomial's roots lie in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HalfPlane {
    /// Every root has `Re < 0`.
    Hurwitz,
    /// Every root has `Re > 0`.
    AntiHurwitz,
    /// Neither — roots straddle the axis, sit on it, or the exact Routh test
    /// hit a degenerate row and could not decide.
    Mixed,
}

fn classify_half_plane(p: &QPoly) -> HalfPlane {
    if routh_is_hurwitz(p) == Some(true) {
        return HalfPlane::Hurwitz;
    }
    if routh_is_hurwitz(&reflect(p)) == Some(true) {
        return HalfPlane::AntiHurwitz;
    }
    HalfPlane::Mixed
}

/// Exact Routh–Hurwitz test over ℚ.
///
/// `Some(true)` when every root of `p` has strictly negative real part,
/// `Some(false)` when some root provably does not, and `None` when the array
/// degenerates (a zero pivot) and the test cannot decide. `None` is treated as
/// "not Hurwitz" by [`classify_half_plane`], so an undecidable factor makes
/// the whole route decline rather than guess.
fn routh_is_hurwitz(p: &QPoly) -> Option<bool> {
    let p = trim(p.clone());
    let n = degree(&p);
    if n < 0 {
        return None;
    }
    if n == 0 {
        return Some(true); // a nonzero constant has no roots
    }
    let n = n as usize;
    // Descending coefficients a₀sⁿ + a₁sⁿ⁻¹ + … + aₙ, normalised to a₀ > 0.
    let mut desc: Vec<Rational> = (0..=n).map(|i| coeff(&p, n - i)).collect();
    if desc[0] < 0 {
        for c in desc.iter_mut() {
            *c = -c.clone();
        }
    }
    // Necessary condition: all coefficients strictly positive.
    if desc.iter().any(|c| *c <= 0) {
        return Some(false);
    }
    let width = n / 2 + 1;
    let mut prev: Vec<Rational> = (0..width)
        .map(|j| {
            desc.get(2 * j)
                .cloned()
                .unwrap_or_else(|| Rational::from(0))
        })
        .collect();
    let mut cur: Vec<Rational> = (0..width)
        .map(|j| {
            desc.get(2 * j + 1)
                .cloned()
                .unwrap_or_else(|| Rational::from(0))
        })
        .collect();
    for _ in 0..n.saturating_sub(1) {
        if cur[0] == 0 {
            return None;
        }
        if cur[0] < 0 {
            return Some(false);
        }
        let mut next = vec![Rational::from(0); width];
        for j in 0..width - 1 {
            next[j] = prev[j + 1].clone() - (prev[0].clone() / cur[0].clone()) * cur[j + 1].clone();
        }
        prev = cur;
        cur = next;
    }
    if cur[0] == 0 {
        return None;
    }
    Some(cur[0] > 0)
}

/// `a - b` for ascending-degree ℚ polynomials.
fn poly_sub_qq(a: &QPoly, b: &QPoly) -> QPoly {
    poly_add(a, &poly_scale(b, &Rational::from(-1)))
}

/// Gauss–Jordan over ℚ for a square system with a unique solution.
fn solve_exact(mut mat: Vec<Vec<Rational>>, mut rhs: Vec<Rational>) -> Option<Vec<Rational>> {
    let n = mat.len();
    if n == 0 || mat.iter().any(|r| r.len() != n) || rhs.len() != n {
        return None;
    }
    for col in 0..n {
        let pivot = (col..n).find(|&r| mat[r][col] != 0)?;
        mat.swap(col, pivot);
        rhs.swap(col, pivot);
        let inv = Rational::from(1) / mat[col][col].clone();
        for entry in mat[col][col..n].iter_mut() {
            *entry *= inv.clone();
        }
        rhs[col] = rhs[col].clone() * inv;
        let pivot_row = mat[col].clone();
        let pivot_rhs = rhs[col].clone();
        for r in 0..n {
            if r == col || mat[r][col] == 0 {
                continue;
            }
            let factor = mat[r][col].clone();
            for (entry, p) in mat[r][col..n].iter_mut().zip(&pivot_row[col..n]) {
                *entry -= factor.clone() * p.clone();
            }
            rhs[r] = rhs[r].clone() - factor * pivot_rhs.clone();
        }
    }
    Some(rhs)
}

/// The `π` symbol, in the crate-wide convention (a `Domain::Real` symbol named
/// `pi`, which is what the Python binding binds and what `sum::special` emits).
fn pi_symbol(pool: &ExprPool) -> ExprId {
    pool.symbol("pi", crate::kernel::Domain::Real)
}

/// `π · e`.
fn times_pi(e: ExprId, pool: &ExprPool) -> ExprId {
    let pi = pi_symbol(pool);
    crate::simplify::engine::simplify(pool.mul(vec![pi, e]), pool).value
}

/// Evaluate `expr` in `f64`, binding `π` — the plain evaluator treats `pi` as
/// an unbound symbol, and every value this module produces contains one.
fn numeric_value(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(pi_symbol(pool), std::f64::consts::PI);
    let v = crate::eval::eval_f64(expr, pool, &bindings).ok()?;
    v.is_finite().then_some(v)
}

// ---------------------------------------------------------------------------
// Rigorous enclosure of the true integral
// ---------------------------------------------------------------------------

/// A rigorous outer bound for `∫_{-∞}^{∞} num/den dx`, or `None` if one could
/// not be established.
///
/// The real line is covered exactly, with no truncation:
/// `[-1, 1]` directly, and each infinite tail through the change of variable
/// `x = ±1/t`, which maps it onto `[0, 1]` and turns the integrand into
/// another rational function — regular there because `den` has no real root and
/// `deg den ≥ deg num + 2` (so no `1/t` blow-up at `t = 0` survives).
///
/// Callers must have established both of those hypotheses first; this function
/// assumes them.
fn enclose_line_integral(
    num: &QPoly,
    den: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64)> {
    let p = degree(num);
    let q = degree(den);
    // The tail change of variable is only regular when the integrand decays at
    // least like `x^{-2}`. Refuse rather than build a nonsense integrand — a
    // caller that skipped the degree check gets a decline, not a number.
    if q - p < 2 {
        return None;
    }
    let pad = (q - p - 2) as usize;

    // `x = 1/t` tail: t^{q-p-2}·P*(t) / Q*(t)  on [0, 1].
    let plus_num = shift_up(&reverse(num, p), pad);
    let plus_den = reverse(den, q);
    // `x = -1/t` tail: same with alternating signs before reversal.
    let minus_num = shift_up(&reverse(&reflect(num), p), pad);
    let minus_den = reverse(&reflect(den), q);

    let core = ratio_expr(num, den, var, pool);
    let tail_plus = ratio_expr(&plus_num, &plus_den, var, pool);
    let tail_minus = ratio_expr(&minus_num, &minus_den, var, pool);

    let opts = crate::validated::bounds::IntegralOptions {
        order: 8,
        prec: 128,
        tol: 1e-10,
        max_subdivisions: 4096,
    };
    let mut lo = 0.0_f64;
    let mut hi = 0.0_f64;
    for (expr, a, b) in [
        (core, -1.0, 1.0),
        (tail_plus, 0.0, 1.0),
        (tail_minus, 0.0, 1.0),
    ] {
        let piece =
            crate::validated::bounds::verified_integral(expr, pool, var, a, b, &opts).ok()?;
        lo += piece.lower();
        hi += piece.upper();
    }
    (lo.is_finite() && hi.is_finite() && lo <= hi).then_some((lo, hi))
}

/// `t^deg · p(1/t)` — the coefficient list reversed within degree `deg`.
fn reverse(p: &QPoly, deg: i64) -> QPoly {
    let deg = deg.max(0) as usize;
    trim((0..=deg).map(|i| coeff(p, deg - i)).collect())
}

/// Multiply by `t^k`.
fn shift_up(p: &QPoly, k: usize) -> QPoly {
    if k == 0 {
        return p.clone();
    }
    let mut out = vec![Rational::from(0); k];
    out.extend(p.iter().cloned());
    trim(out)
}

fn ratio_expr(num: &QPoly, den: &QPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    let n = qpoly_to_expr(num, var, pool);
    let d = qpoly_to_expr(den, var, pool);
    if trim(den.clone()) == poly_one() {
        return n;
    }
    crate::simplify::engine::simplify(pool.mul(vec![n, pool.pow(d, pool.integer(-1_i32))]), pool)
        .value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    fn q(v: &[i64]) -> QPoly {
        trim(v.iter().map(|c| Rational::from(*c)).collect())
    }

    fn value_of(src: &str) -> (f64, String) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_owned(), x)]);
        let f = crate::parse::parse(src, &pool, &mut syms).unwrap();
        match integrate_rational_over_real_line(f, x, &pool) {
            LineIntegral::Value { value, .. } => (
                numeric_value(value, &pool).unwrap(),
                pool.display(value).to_string(),
            ),
            other => panic!("{src}: expected a value, got {other:?}"),
        }
    }

    fn assert_close(src: &str, expected: f64) {
        let (got, shown) = value_of(src);
        assert!(
            (got - expected).abs() < 1e-9 * (1.0 + expected.abs()),
            "∫_{{-∞}}^{{∞}} {src} dx: expected {expected}, got {got} (as {shown})"
        );
    }

    #[test]
    fn quadratic_denominators() {
        assert_close("1/(x^2+1)", std::f64::consts::PI);
        assert_close("1/(x^2+4)", std::f64::consts::FRAC_PI_2);
        assert_close("1/(x^2+2*x+2)", std::f64::consts::PI);
        assert_close("1/(2*x^2+3)", std::f64::consts::PI / 6.0_f64.sqrt());
    }

    #[test]
    fn quartic_denominators() {
        assert_close("1/(x^4+1)", std::f64::consts::PI / 2.0_f64.sqrt());
        assert_close("x^2/(x^4+1)", std::f64::consts::PI / 2.0_f64.sqrt());
        assert_close("1/(x^4+x^2+1)", std::f64::consts::PI / 3.0_f64.sqrt());
        assert_close("1/(x^2+1)^2", std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn nested_radical_branch_when_the_spectral_factor_is_irrational() {
        // `W = √(d₀/d₄) = √3` is not rational here, so this exercises the
        // nested-radical arm: ∫ dx/(x⁴+3) = 3^{-3/4}·π/√2.
        assert_close(
            "1/(x^4+3)",
            3.0_f64.powf(-0.75) * std::f64::consts::PI / 2.0_f64.sqrt(),
        );
    }

    #[test]
    fn odd_denominator_is_normalised_by_multiplying_by_q_of_minus_x() {
        // `(x²+1)(x²+2x+2)` is not even; `D = Q(x)·Q(−x) = (x²+1)²(x⁴+4)` is,
        // and its rotation factors over ℚ (Sophie Germain on `s⁴+4`).
        // ∫ dx/((x²+1)(x²+2x+2)) = 2π/5.
        assert_close("1/((x^2+1)*(x^2+2*x+2))", 2.0 * std::f64::consts::PI / 5.0);
    }

    #[test]
    fn out_of_scope_denominator_declines_rather_than_guessing() {
        // `x⁸+1` is ℚ-irreducible with roots in both half-planes, so its
        // Hurwitz spectral factor is not rational; and `deg D = 8` is past the
        // radical case. The route must say so, not guess.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_owned(), x)]);
        let f = crate::parse::parse("1/(x^8+1)", &pool, &mut syms).unwrap();
        match integrate_rational_over_real_line(f, x, &pool) {
            LineIntegral::OutOfScope(why) => {
                assert!(why.contains("not rational"), "unexpected reason: {why}")
            }
            other => panic!("expected a decline, got {other:?}"),
        }
    }

    #[test]
    fn higher_degree_via_rational_spectral_factor() {
        assert_close("1/(x^6+1)", 2.0 * std::f64::consts::PI / 3.0);
        // ∫ dx/(x²+1)³ = 3π/8.
        assert_close("1/(x^2+1)^3", 3.0 * std::f64::consts::PI / 8.0);
        // ∫ x²dx/(x²+1)³ = π/8.
        assert_close("x^2/(x^2+1)^3", std::f64::consts::PI / 8.0);
        // ∫ dx/(x²+1)⁴ = 5π/16 — a fourth-order pole.
        assert_close("1/(x^2+1)^4", 5.0 * std::f64::consts::PI / 16.0);
    }

    #[test]
    fn real_pole_is_divergent_not_a_number() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_owned(), x)]);
        let f = crate::parse::parse("1/(x^2-1)", &pool, &mut syms).unwrap();
        assert!(matches!(
            integrate_rational_over_real_line(f, x, &pool),
            LineIntegral::Divergent(_)
        ));
    }

    #[test]
    fn degree_condition_failure_is_divergent_not_a_principal_value() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_owned(), x)]);
        for src in ["x/(x^2+1)", "1", "x^2/(x^2+1)"] {
            let f = crate::parse::parse(src, &pool, &mut syms).unwrap();
            assert!(
                matches!(
                    integrate_rational_over_real_line(f, x, &pool),
                    LineIntegral::Divergent(_)
                ),
                "{src} must be reported divergent"
            );
        }
    }

    #[test]
    fn removable_singularity_does_not_look_like_a_real_pole() {
        // (x²-1)/((x-1)(x⁴+1)) reduces to (x+1)/(x⁴+1); no real pole remains.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_owned(), x)]);
        let f = crate::parse::parse("(x^2-1)/((x-1)*(x^4+1))", &pool, &mut syms).unwrap();
        match integrate_rational_over_real_line(f, x, &pool) {
            LineIntegral::Value { value, .. } => {
                let got = numeric_value(value, &pool).unwrap();
                // ∫(x+1)/(x⁴+1) = ∫1/(x⁴+1) (the odd part cancels) = π/√2.
                let want = std::f64::consts::PI / 2.0_f64.sqrt();
                assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
            }
            other => panic!("expected a value, got {other:?}"),
        }
    }

    #[test]
    fn non_rational_integrand_is_out_of_scope() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("sin", vec![x]);
        assert!(matches!(
            integrate_rational_over_real_line(f, x, &pool),
            LineIntegral::OutOfScope(_)
        ));
    }

    #[test]
    fn routh_matches_textbook_cases() {
        // s + 1 — Hurwitz. s - 1 — anti-Hurwitz.
        assert_eq!(routh_is_hurwitz(&q(&[1, 1])), Some(true));
        assert_eq!(routh_is_hurwitz(&q(&[-1, 1])), Some(false));
        // s³ + 2s² + 2s + 1 — the spectral factor of 1 - s⁶.
        assert_eq!(routh_is_hurwitz(&q(&[1, 2, 2, 1])), Some(true));
        // s³ + s² + 2s + 8 — fails the Routh inequality (a·b < c).
        assert_eq!(routh_is_hurwitz(&q(&[8, 2, 1, 1])), Some(false));
        // s⁴ + 1 — roots in both half-planes.
        assert_ne!(routh_is_hurwitz(&q(&[1, 0, 0, 0, 1])), Some(true));
    }

    #[test]
    fn spectral_factor_of_one_minus_s_sixth() {
        // Ď(s) = 1 - s⁶ for D = x⁶ + 1; A = s³ + 2s² + 2s + 1.
        let pool = ExprPool::new();
        let target = q(&[1, 0, 0, 0, 0, 0, -1]);
        let x = pool.symbol("s", Domain::Complex);
        let a = hurwitz_spectral_factor(&target, 3, x, &pool).unwrap();
        assert_eq!(a, q(&[1, 2, 2, 1]));
    }

    #[test]
    fn enclosure_brackets_the_known_value() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let (lo, hi) = enclose_line_integral(&q(&[1]), &q(&[1, 0, 0, 0, 1]), x, &pool).unwrap();
        let want = std::f64::consts::PI / 2.0_f64.sqrt();
        assert!(lo <= want && want <= hi, "π/√2 ∉ [{lo}, {hi}]");
        assert!(
            hi - lo < 1e-6,
            "enclosure [{lo}, {hi}] is too wide to be useful"
        );
    }
}
