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
//! No reduction constant is trusted blindly.  Every candidate is run through
//! the shared [`crate::integrate::gate`] — the reusable *propose → fit →
//! verify → emit-or-decline* facility this module's pattern was extracted
//! into.  The candidate's *symbolic* `d/dx` (via the engine's `diff`, which
//! differentiates the elliptic functions through the primitive registry —
//! `∂φ F = 1/√(1 − m·sin²φ)`, `∂φ E = √(1 − m·sin²φ)`,
//! `∂φ Π = 1/((1 − n sin²φ)√(1 − m sin²φ))`, all elementary since `m`, `n` are
//! constant here) is checked against the integrand on `R ∩ {P > 0}`, where `R`
//! is the [`Region`] the reduction claims — **not** on all of `{P > 0}`, which
//! is a strictly larger set for most root configurations:
//!
//! * first, symbolically — `simplify(d/dx F − f) == 0` gives
//!   [`gate::Verdict::Proven`];
//! * then by `f64` sampling at the `gate_samples` grid, which is drawn from
//!   `R`, at a `1e-7` relative tolerance over at least three points
//!   ([`gate::Verdict::SampledOnly`]) — this tier is the **acceptance
//!   decision**;
//! * then, on the candidate that already survived that screen, by a rigorous
//!   Taylor-model enclosure of `d/dx F − f` over a closed box strictly inside
//!   `R` ([`gate::Verdict::EnclosureVerified`]).  This tier is *additive*: it
//!   can only strengthen the recorded evidence, never widen what is accepted.
//!
//! A form is emitted **only** if the gate passes; otherwise the caller falls
//! through to `NonElementary`.  An imperfect fit can therefore never produce a
//! wrong answer — it merely declines.
//!
//! The region is load-bearing rather than an optimisation, because the gate
//! treats "the candidate is undefined where the integrand is an ordinary finite
//! real" as a **disagreement**.  Under that rule a sample set wider than the
//! claim is not extra caution; it refutes correct answers.  See [`Region`].
//!
//! What the enclosure tier does *not* cover is stated where it belongs, in the
//! gate's own [honest-limitations list](crate::integrate::gate): neighbourhoods
//! of the roots of `P` (where `1/√P` is unbounded and no finite bound exists),
//! the points `R` cuts out, and the unbounded tails are never inside a box.

use crate::integrate::gate::{self, eval_at as eval};
use crate::integrate::risch::poly_rde::{expr_to_qpoly, is_free_of_var};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// A complex root, stored as `(re, im)`.
pub(super) type Croot = (f64, f64);

/// The real set on which a Byrd–Friedman reduction **claims** `d/dx F = f`.
///
/// This is deliberately *not* `{P > 0}`.  Each reduction below is a
/// substitution `φ(x)` that is real on only part of `{P > 0}`: the
/// three-real-root cubic needs `x` beyond the largest root, the four-real-root
/// quartic needs `x` between the two middle roots, and so on.  `∫dx/√(x³−x)`
/// is the standard illustration — `P > 0` on `(−1, 0) ∪ (1, ∞)`, the reduction
/// holds only on `(1, ∞)`, and on `(−1, 0)` the integrand is an ordinary finite
/// real while the candidate's derivative is `NaN`.
///
/// Sampling the whole of `{P > 0}` therefore asks the candidate about points it
/// never claimed, and a gate that reads "candidate undefined where the
/// integrand is finite" as a disagreement (which is the right reading — see
/// [`crate::integrate::gate`]) refuses a correct answer.  So every reduction
/// states its region here, and the gate's sample grid, its in-domain predicate
/// and its enclosure boxes are all built from it.
///
/// The region is an **open** interval: the finite endpoints are roots of `P`,
/// where `1/√P` is unbounded and there is nothing to compare.  `cuts` removes
/// finitely many interior points, of which there are two kinds and both are
/// facts about the *written* candidate, not about the mathematics:
///
/// * the pole of the `arctan` substitution's Möbius argument (the no-real-root
///   quartic), where `φ` jumps by `π`; and
/// * the poles the fitted higher-kind block set is written with — `log|x−t|`
///   at a twin preimage, `√P/(x−p)` at a reduction pole, `EllipticPi`'s
///   spurious twin pole.  Those cancel in the sum, so the residual's *limit*
///   there is finite, but the expression as written evaluates `∞ − ∞`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Region {
    lo: f64,
    hi: f64,
    cuts: Vec<f64>,
}

impl Region {
    /// The open interval `(lo, hi)`; either endpoint may be infinite.
    fn open(lo: f64, hi: f64) -> Self {
        Region {
            lo,
            hi,
            cuts: Vec::new(),
        }
    }

    /// The whole real line.
    fn all() -> Self {
        Region::open(f64::NEG_INFINITY, f64::INFINITY)
    }

    /// Remove the interior points `xs` (see the type docs for what qualifies).
    fn cut_all(mut self, xs: impl IntoIterator<Item = f64>) -> Self {
        for x in xs {
            if x.is_finite() && x > self.lo && x < self.hi && !self.cuts.contains(&x) {
                self.cuts.push(x);
            }
        }
        self.cuts
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self
    }

    /// Is `x` a point this reduction claims?
    ///
    /// A cut is a single real number, but `f64` cannot see it that way: the
    /// blocks whose poles cancel there are `O(1/(x−c))`, so evaluating within
    /// `~ε/tol` of `c` loses the cancellation to rounding.  With `ε ≈ 2.2e−16`
    /// and the gate's `1e−7` tolerance that boundary sits near `1e−9`; the
    /// `1e−6` used here is three decades clear of it and still far narrower
    /// than any domain hole the gate exists to catch.
    fn contains(&self, x: f64) -> bool {
        x.is_finite()
            && x > self.lo
            && x < self.hi
            && !self
                .cuts
                .iter()
                .any(|&c| (x - c).abs() <= 1e-6 * (1.0 + c.abs()))
    }

    /// The claimed set as disjoint open intervals, clipped to `[−window,
    /// window]`.  Sub-intervals narrower than `min_width` are dropped.
    fn components(&self, window: f64, min_width: f64) -> Vec<(f64, f64)> {
        let lo = self.lo.max(-window);
        let hi = self.hi.min(window);
        if hi <= lo || hi.is_nan() || lo.is_nan() {
            return Vec::new();
        }
        let mut cuts: Vec<f64> = vec![lo];
        cuts.extend(self.cuts.iter().copied().filter(|&c| c > lo && c < hi));
        cuts.push(hi);
        cuts.windows(2)
            .map(|w| (w[0], w[1]))
            .filter(|&(a, b)| b - a >= min_width)
            .collect()
    }
}

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
    try_elliptic_output_with(a_part, b_part, p_expr, var, pool, &gate_options())
}

/// [`try_elliptic_output`] with an explicit gate configuration.
///
/// Pass `gate::GateOptions::rigorous(..)` to demand a
/// [`gate::Verdict::EnclosureVerified`] and decline anything weaker.  Note
/// that this can only ever *narrow* what is emitted: the default options are
/// already the historical acceptance rule.
pub fn try_elliptic_output_with(
    a_part: ExprId,
    b_part: ExprId,
    p_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    gate_opts: &gate::GateOptions,
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

    let (g, m, phi, region) = first_kind_reduction(&coeffs, deg, lead, var, pool)?;

    // F_cand = (c · g) · EllipticF(phi, m).
    let m_expr = float_to_expr(m, pool);
    let f = pool.func("EllipticF", vec![phi, m_expr]);
    let coeff = float_to_expr(c * g, pool);
    let f_cand = simplify(pool.mul(vec![coeff, f]), pool).value;

    // Soundness gate: d/dx F_cand = c/√P on the region this reduction claims,
    // intersected with `P > 0`.
    if verify(
        f_cand, &coeffs, c, c_expr, p_expr, var, region, pool, gate_opts,
    )
    .is_verified()
    {
        Some(f_cand)
    } else {
        None
    }
}

/// Compute the shared first-kind Legendre reduction `(g, m, φ(x), R)` for
/// `√P`, chosen so that `d/dx[g·EllipticF(φ,m)] = 1/√P` on the region `R`.
/// This is the *same* substitution used by every higher-kind reduction (B&F:
/// all of `∫R(x,√P)dx` reduce under one substitution), so the second/third-kind
/// paths reuse it verbatim — region included.
///
/// `R` is **not** `{P > 0}`.  Each B&F normal form is real on one component of
/// `{P > 0}` (or, for the four-real-root quartic, on the bounded middle one),
/// and every caller must verify only there; see [`Region`].  Each case returns
/// its own region next to its own constants, so the two cannot drift apart.
///
/// Returns `None` for radicand shapes outside the handled cubic/quartic
/// root-configurations (e.g. all-complex quartic).
fn first_kind_reduction(
    coeffs: &[f64],
    deg: usize,
    lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId, Region)> {
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
        // Not the genus-≥2 case, whatever the arms above look like: every
        // caller has already required `deg ∈ {3, 4}`, so a quintic never
        // reaches here.  Since `reals + 2·pairs` is the degree, the arms above
        // exhaust both degrees, and this one is left for a root split the
        // numeric solver produced that does not add up — a lost or spurious
        // root.  Declining falls through to the caller's own analysis.
        _ => None,
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
    try_elliptic_output_higher_kind_with(a_part, b_part, p_expr, var, pool, &gate_options())
}

/// [`try_elliptic_output_higher_kind`] with an explicit gate configuration.
pub fn try_elliptic_output_higher_kind_with(
    a_part: ExprId,
    b_part: ExprId,
    p_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    gate_opts: &gate::GateOptions,
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

    // The shared first-kind substitution, and the region it claims.
    let (g, m, phi, region) = first_kind_reduction(&p_coeffs, deg, lead, var, pool)?;
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

    // Points at which some block below is *written* with a pole that the fitted
    // combination cancels: `√P/(x−p)` and `√P/(x−p)²` at the reduction poles and
    // at the roots of `P`, `log|x−t|` and `EllipticPi`'s spurious twin pole at a
    // twin preimage.  The residual's limit at such a point is finite — that is
    // the whole reason the block is in the ansatz — but the expression as
    // written evaluates `∞ − ∞`, so `d/dx F` comes back non-finite where the
    // integrand is an ordinary real.  Under the gate's rule that is a
    // disagreement, and it would be a false one: the candidate does not claim
    // these isolated points.  They are cut from the region, so the claim and
    // the sample set stay the same set.
    //
    // The union over *all* recipes is cut, not just the winning one's, because
    // one domain serves the whole `propose_fit_verify` search.  That is a few
    // isolated points wider than strictly necessary; each is named and finite,
    // and none of them is an interval, which is the thing the rule exists to
    // catch.
    let mut written_poles: Vec<f64> = Vec::new();
    written_poles.extend(red_poles.iter().copied());
    written_poles.extend(p_roots.iter().copied());

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
    // so a pole at `x = p` is shared with a "twin" preimage `t` and a single `Π`
    // introduces a **spurious pole at `t`**: `d/dx[Π] = N(x)/((x−p)(x−t)√P)`, so
    // the genuine `1/((x−p)√P)` is accompanied by a `1/((x−t)√P)` part the
    // `Π`/`F`/`E`/algebraic basis alone cannot match (the fit closes only to
    // ~1e-5 and the gate declines).
    //
    // PR7 adds the missing *elementary* block for the **cubic** one-real config:
    // when the twin `t` lies in the real region (`P(t) > 0`), the twin third-kind
    // integral `∫dx/((x−t)√P)` is **elementary for a *cubic* `P`** — a combination
    // of `log|x − t|` and `log(√P + √P(t))` (see [`elem_log_blocks`]) — whose
    // derivative supplies exactly the twin part.  With it the cubic-one-real third
    // kind closes, e.g. `∫dx/((x−2)√(x³+1))` → `δ·Π + β·F + ε·log(√P+1) + ζ·log|x|`
    // (gate-verified).
    //
    // **The `quartic` two-real cos φ config does NOT close this way** (diagnosed
    // 2026-06-10, `risch/elliptic-output-remaining`): for a *quartic* `P` the twin
    // third-kind integral `∫dx/((x−t)√P)` is itself **non-elementary** (a genuine
    // third-kind elliptic integral — numerically, the best elementary-log fit of
    // its antiderivative stalls at residual ~1.6e-2, never closing).  Because the
    // pole `p` and its twin `t` share the *same* characteristic
    // `n = 1/sin²φ(p) = 1/sin²φ(t)`, a single real `EllipticPi(n,φ,m)` carries both
    // poles and a *second* real `Π` would be the identical block, so the genuine
    // `1/((x−p)√P)` part cannot be isolated within the real
    // `F`/`E`/`Π`/algebraic/elementary-log basis (the derivative-gate residual
    // stays ≳ 3.6 with the full basis).  The `twin_log`/`elem_log_blocks` recipes
    // are still *offered* (they are correct for the cubic case and harmless here —
    // the gate just declines), so the quartic-two-real third kind falls through to
    // `NonElementary`.  Soundness is unconditional: an incomplete basis only
    // declines, never emits a wrong answer.
    //
    // We add the Π blocks for every off-`P`-root real pole and let the numeric fit
    // + gate decide.  Recipe variants are pushed: a *minimal* one (algebraic
    // ladder, `F`, Π), the *rich* one (also `E` + reduction-pole blocks), and the
    // *elementary-augmented* ones (adding the twin log blocks) for the cos φ case.
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
    written_poles.extend(pi_poles.iter().map(|&(p, _)| p));
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
        // 4c/4d) ELEMENTARY-AUGMENTED third-kind basis (this PR) for the cos φ
        //     configurations.  For each Π pole add the twin's elementary log
        //     blocks (`log|x−t|`, `log(√P+√P(t))`) so the spurious twin-pole part
        //     of the Π derivative can be cancelled and the fit can close.  Two
        //     variants: minimal (ladder + F + Π + logs) for clean coefficients,
        //     and rich (also E) as a fallback.
        let twins: Vec<f64> = pi_poles
            .iter()
            .filter_map(|&(p, _)| twin_pole(p, phi, var, pool))
            .collect();
        written_poles.extend(twins.iter().copied());
        let twin_logs: Vec<ExprId> = twins
            .iter()
            .flat_map(|&t| elem_log_blocks(t, p_expr, sqrt_p, var, pool))
            .collect();
        if !twin_logs.is_empty() {
            // 4c) minimal + twin logs.
            {
                let mut s = vec![poly_block(0, pool), poly_block(1, pool)];
                s.push(f_blk);
                build_pi(&mut s, pool);
                s.extend(twin_logs.iter().copied());
                recipes.push(s);
            }
            // 4d) + E (and the smallest-root algebraic block) + twin logs.
            {
                let mut s = vec![poly_block(0, pool), poly_block(1, pool)];
                if let Some(&rmin) = p_roots.first() {
                    s.push(rat_block(rmin, pool));
                }
                s.push(f_blk);
                s.push(e_blk);
                build_pi(&mut s, pool);
                s.extend(twin_logs.iter().copied());
                recipes.push(s);
            }
        }
    }

    let region = region.cut_all(written_poles);

    // Sample grid (shared across recipes) inside the claimed region and away
    // from b-poles.
    let samples = sample_grid(&p_coeffs, &b_den_f, &region);

    // `b·√P` in `f64`, restricted to the in-domain points.  The *same* closure
    // drives the least-squares fit and the gate's `f64` screen, so the two can
    // never drift apart about what "in domain" means.
    let rhs = |xv: f64| -> Option<f64> {
        let pv = eval_poly(&p_coeffs, xv);
        if pv <= 1e-6 {
            return None;
        }
        let bv = eval_ratio(&b_num_f, &b_den_f, xv)?;
        Some(bv * pv.sqrt())
    };
    let integrand = simplify(pool.mul(vec![b_part, sqrt_p]), pool).value;
    let domain = elliptic_domain(&p_coeffs, region);
    let target = gate::Target::symbolic(integrand).with_numeric(&rhs);
    let to_expr = |v: f64, p: &ExprPool| float_to_expr(v, p);

    gate::propose_fit_verify(
        &recipes,
        &samples,
        &rhs,
        &to_expr,
        &target,
        var,
        &domain,
        &gate::FitOptions::default(),
        gate_opts,
        pool,
    )
    .map(|a| a.antiderivative)
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

/// For a `cos φ` substitution `sin²φ(x)` is *quadratic*-over-quadratic in `x`, so
/// the value `sin²φ(p)` at a pole `p` is shared by a second "twin" preimage
/// `x = t ≠ p`.  An `EllipticPi(n_p, φ, m)` block (characteristic `n_p =
/// 1/sin²φ(p)`) consequently has a *spurious* pole at the twin `t` in addition to
/// the genuine pole at `p`; the twin contribution must be cancelled by an extra
/// elementary block for the third-kind fit to close (see [`elem_log_blocks`]).
///
/// Returns the twin `t` (the real `x ≠ p` with `sin²φ(x) = sin²φ(p)`), located by
/// a coarse sign-change scan of `sin²φ(x) − sin²φ(p)` followed by bisection.
/// `None` if no distinct twin is found in the scanned window.
fn twin_pole(p: f64, phi: ExprId, var: ExprId, pool: &ExprPool) -> Option<f64> {
    let target = {
        let v = eval(phi, var, p, pool)?;
        let s = v.sin();
        s * s
    };
    let f = |x: f64| -> Option<f64> {
        let v = eval(phi, var, x, pool)?;
        let s = v.sin();
        Some(s * s - target)
    };
    // Coarse scan for a sign change away from `p`.
    let (lo, hi, step) = (-40.0_f64, 40.0_f64, 0.05_f64);
    let mut x0 = lo;
    let mut f0 = f(x0);
    let mut x = lo + step;
    while x <= hi {
        let f1 = f(x);
        if let (Some(a), Some(b)) = (f0, f1) {
            if a.is_finite() && b.is_finite() && a * b <= 0.0 && (x - p).abs() > 1e-3 {
                // Bisect on [x0, x].
                let (mut l, mut r) = (x0, x);
                let (mut fl, _fr) = (a, b);
                for _ in 0..80 {
                    let mid = 0.5 * (l + r);
                    let Some(fm) = f(mid) else { break };
                    if !fm.is_finite() {
                        break;
                    }
                    if fl * fm <= 0.0 {
                        r = mid;
                    } else {
                        l = mid;
                        fl = fm;
                    }
                }
                let root = 0.5 * (l + r);
                if (root - p).abs() > 1e-4 && root.is_finite() {
                    return Some(root);
                }
            }
        }
        x0 = x;
        f0 = f1;
        x += step;
    }
    None
}

/// Elementary log blocks that cancel the **twin-pole** contribution of an
/// `EllipticPi` block in the `cos φ` third-kind configurations (cubic-one-real,
/// quartic-two-real).
///
/// When the twin preimage `t` of a pole `p` lies in the real region where
/// `P(t) > 0`, the twin third-kind integral `∫dx/((x−t)√P)` is *elementary* for
/// these configurations and its closed form is a combination of
/// `log|x − t|` and `log(√P(x) + √P(t))`.  Adding both as candidate blocks lets
/// the otherwise-stuck fit close (and the soundness gate certifies it); when the
/// twin integral is *not* elementary the fit simply fails and the path declines.
///
/// Returns the (possibly empty) list of block `ExprId`s; the numeric fit assigns
/// their coefficients.
fn elem_log_blocks(
    t: f64,
    p_expr: ExprId,
    sqrt_p: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Vec<ExprId> {
    let mut blocks = Vec::new();
    // `√P(t)` must be a positive real for the second block to be well defined.
    let pt = {
        let v = eval(sqrt_p, var, t, pool);
        v.filter(|w| w.is_finite() && *w > 0.0)
    };
    // Block 1: log|x − t|.
    let xt = pool.add(vec![var, float_to_expr(-t, pool)]);
    blocks.push(pool.func("log", vec![xt]));
    // Block 2: log(√P + √P(t)).
    if let Some(spt) = pt {
        let _ = p_expr;
        let arg = pool.add(vec![sqrt_p, float_to_expr(spt, pool)]);
        blocks.push(pool.func("log", vec![arg]));
    }
    blocks
}

/// Sample grid for the least-squares *fit*, restricted to the region the
/// reduction claims and kept away from the weight's poles.
///
/// The region filter does not change which rows the fit sees — every recipe
/// contains an `EllipticF` block, whose derivative is `NaN` outside the region,
/// so [`gate::fit_blocks`] already dropped those rows.  It is applied because
/// the design matrix should be built from the set the answer is about, rather
/// than from a wider set that happens to self-filter.
fn sample_grid(p_coeffs: &[f64], b_den: &[f64], region: &Region) -> Vec<f64> {
    let mut xs = Vec::new();
    let mut x = -4.0_f64;
    while x <= 6.0 {
        // Skip points outside the claim, and points too close to a denominator
        // zero.
        if region.contains(x) && eval_poly(p_coeffs, x) > 1e-6 && eval_poly(b_den, x).abs() > 1e-3 {
            xs.push(x);
        }
        x += 0.137;
    }
    xs
}

// ---------------------------------------------------------------------------
// Reduction cases (Byrd & Friedman normal forms)
// ---------------------------------------------------------------------------

/// Cubic, three real roots `e1 > e2 > e3`: `sin²φ = (e1−e3)/(x−e3)`,
/// `m = (e2−e3)/(e1−e3)`, `g = −2/√(e1−e3)`.
///
/// **Region `(e1, ∞)`.**  `sin²φ = (e1−e3)/(x−e3)` lies in `[0, 1]` exactly for
/// `x ≥ e1`: below `e1` (but above `e3`) the ratio exceeds `1`, and below `e3`
/// it is negative, so `asin(√·)` is not real either way.  For a positive
/// leading coefficient `{P > 0}` is `(e3, e2) ∪ (e1, ∞)` — strictly wider — and
/// `∫dx/√(x³−x)` on `(−1, 0)` is the case that makes the difference visible.
/// The `P > 0` half of the claim is *not* assumed here: it is re-checked
/// pointwise by [`elliptic_domain`]'s predicate, which is what makes a negative
/// leading coefficient (where this region and `{P > 0}` are disjoint) decline
/// rather than misreport.
fn cubic_three_real(
    reals: &[f64],
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId, Region)> {
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
    Some((g, m, phi, Region::open(e1, f64::INFINITY)))
}

/// Cubic, one real root `y1` and a complex pair `b1 ± i·a1`:
/// `A = √((y1−b1)² + a1²)`, `g = 1/√A`, `m = (A + (b1−y1))/(2A)`,
/// `cos φ = (A − (x−y1))/(A + (x−y1))`.
///
/// **Region `(y1, ∞)`.**  With `u = x − y1 > 0` the quotient `(A−u)/(A+u)` runs
/// over `(−1, 1)`; for `u < 0` it leaves `[−1, 1]` (and passes through a pole at
/// `u = −A`), so `acos` is not real.  Here the region coincides with `{P > 0}`
/// for a positive leading coefficient — the narrowing is a no-op for this
/// case — and it is stated anyway so that every reduction makes the same kind
/// of claim.
fn cubic_one_real(
    y1: f64,
    pair: Croot,
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId, Region)> {
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
    Some((g, m, phi, Region::open(y1, f64::INFINITY)))
}

/// Quartic, four real roots `a > b > c > d`:
/// `sn²φ = (b−d)(x−c)/((b−c)(x−d))`, `m = (b−c)(a−d)/((a−c)(b−d))`,
/// `g = 2/√((a−c)(b−d))`.
///
/// **Region `(c, b)`.**  Write `h(x) = K(x−c)/(x−d)` with `K = (b−d)/(b−c) > 1`.
/// `h` is increasing on each side of its pole at `d`, `h(c) = 0` and `h(b) = 1`,
/// so `h ∈ [0, 1]` exactly on `[c, b]`: above `b` it exceeds `1`, on `(d, c)` it
/// is negative, and below `d` it exceeds `1` again.  For a positive leading
/// coefficient `{P > 0}` is `(−∞, d) ∪ (c, b) ∪ (a, ∞)`, so this reduction
/// claims one bounded component out of three.
fn quartic_four_real(
    reals: &[f64],
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId, Region)> {
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
    Some((g, m, phi, Region::open(c, b)))
}

/// Quartic, two real roots `b1 > b2` and a complex pair `b3 ± i·a3`:
/// `A1 = √((b1−b3)² + a3²)`, `A2 = √((b2−b3)² + a3²)`, `g = 1/√(A1·A2)`,
/// `m = ((A1+A2)² − (b1−b2)²)/(4·A1·A2)`,
/// `cos φ = ((b1−x)A2 − (x−b2)A1)/((b1−x)A2 + (x−b2)A1)`.
///
/// **Region `(b2, b1)`.**  Writing `u = (b1−x)A2` and `v = (x−b2)A1`,
/// `cos φ = (u−v)/(u+v)` has modulus `< 1` exactly when `u` and `v` are both
/// positive, i.e. strictly between the two real roots; it is `+1` at `b2`, `−1`
/// at `b1`, and of modulus `> 1` immediately outside.  `P > 0` on this interval
/// requires a *negative* leading coefficient (`1 − x⁴` is the usual instance);
/// with a positive one the region and `{P > 0}` are disjoint and the gate
/// declines for want of in-domain points, which is the honest outcome rather
/// than a mis-stated one.
fn quartic_two_real(
    reals: &[f64],
    pair: Croot,
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId, Region)> {
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
    Some((g, m, phi, Region::open(b2, b1)))
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
) -> Option<(f64, f64, ExprId, Region)> {
    let (p, q, r, s, m, g) = quartic_no_real_consts(pair1, pair2, lead)?;
    // φ(x) = arctan( L(x) ),  L = (p·x+q)/(r·x+s).  The raw `(p,q,r,s)` are
    // `cos/sin θ`-scaled (θ a fixed angle of the substitution), so individually
    // they are nested-radical floats — but `atan(L)` is invariant under scaling
    // the numerator and denominator of `L` by the *same* constant.  Divide all
    // four by their largest magnitude so the shared `cos/sin θ` factor cancels and
    // `float_to_expr` sees simple `a+b√n` coefficients (e.g. `∫dx/√(x⁴+1)` →
    // `L = (1+√2)(x−1)/(x+1)`) instead of `2⁵³`-scale reconstructions.
    let nrm = [p, q, r, s]
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let (p, q, r, s) = if nrm > 1e-300 {
        (p / nrm, q / nrm, r / nrm, s / nrm)
    } else {
        (p, q, r, s)
    };
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
    // Region: the whole real line — `P > 0` everywhere and `atan` is real for
    // every argument — **minus the pole of `L` at `x = −s/r`**.  There `φ` jumps
    // from `+π/2` to `−π/2`, so the candidate is discontinuous across it (the
    // one-sided derivatives are still right; the jump is the additive constant a
    // branch change contributes).  The written derivative is `0/0` exactly at
    // the pole, so the point is cut rather than sampled.  Rescaling
    // `(p, q, r, s)` above does not move `−s/r`.
    let region = if r.abs() > 1e-12 {
        Region::all().cut_all([-s / r])
    } else {
        Region::all()
    };
    Some((g, m, phi, region))
}

// ---------------------------------------------------------------------------
// Verification gate
// ---------------------------------------------------------------------------

/// Sample points for the soundness gate, drawn from the region the reduction
/// **claims** — and from nowhere else.
///
/// This used to put points in *every* `P > 0` interval and say so, on the
/// reasoning that "points where the substitution is invalid simply evaluate
/// non-finite and are skipped".  That is true only of a gate that skips them.
/// A gate that reads "the candidate is undefined where the integrand is an
/// ordinary finite real" as a **disagreement** — which is the right reading,
/// and is what [`crate::integrate::verify_antiderivative_status`] has done
/// since PR #344 — refuses a correct answer instead: for `∫dx/√(x³−x)` the
/// integrand is finite all over `(−1, 0)` while the cubic-three-real
/// candidate's derivative is `NaN` there, because that component is not part
/// of the claim.  The sample set was wider than the claim, so the fix belongs
/// here and not in the rule.
///
/// So: points come from `region` only.  A wide fixed grid supplies the ordinary
/// case, and region-derived points guarantee the gate's three-point minimum is
/// reachable however narrow or far away the region is (`∫dx/√(x³−7x−6)`, region
/// `(3, ∞)`, was the case that motivated deriving them at all).  Every point is
/// still re-checked against `P > 0`: a region and `{P > 0}` can be disjoint
/// when the leading coefficient has the wrong sign for the normal form, and
/// then the gate declines for want of points rather than reporting anything.
///
/// The offsets and fractions are deliberately not round numbers.  That is not
/// superstition: a fitted higher-kind block set carries `log|x−t|` and
/// `√P/(x−p)` written singularities, and those *are* cut from the region by the
/// caller — but only the ones the caller knows about, so a grid that avoids
/// landing on tidy values is one less way to trip over one it does not.
fn gate_samples(p_coeffs: &[f64], region: &Region) -> Vec<f64> {
    /// Offsets inward from a finite endpoint of an unbounded component.
    const OFFSETS: [f64; 9] = [
        0.0613, 0.1373, 0.2917, 0.6131, 1.2437, 2.5391, 5.1173, 10.241, 20.487,
    ];
    /// Fractions along a bounded component.
    const FRACTIONS: [f64; 11] = [
        0.0137, 0.0731, 0.1523, 0.2417, 0.3313, 0.4909, 0.6121, 0.7213, 0.8317, 0.9109, 0.9767,
    ];
    let pos = |x: f64| eval_poly(p_coeffs, x) > 1e-6;
    let mut xs: Vec<f64> = Vec::new();
    let push = |x: f64, xs: &mut Vec<f64>| {
        if region.contains(x) && pos(x) {
            xs.push(x);
        }
    };
    for &x in &[
        -3.5_f64, -2.7, -1.6, -0.9, -0.4, 0.15, 0.3, 0.55, 0.7, 0.9, 1.1, 1.4, 1.9, 2.6, 3.4, 4.7,
        5.3,
    ] {
        push(x, &mut xs);
    }
    for (lo, hi) in region.components(f64::INFINITY, 0.0) {
        match (lo.is_finite(), hi.is_finite()) {
            (true, true) => {
                for f in FRACTIONS {
                    push(lo + (hi - lo) * f, &mut xs);
                }
            }
            (true, false) => {
                for o in OFFSETS {
                    push(lo + o, &mut xs);
                }
            }
            (false, true) => {
                for o in OFFSETS {
                    push(hi - o, &mut xs);
                }
            }
            // Doubly unbounded: the fixed grid above already covers it.
            (false, false) => {}
        }
    }
    xs
}

/// Closed boxes **strictly inside the region the reduction claims**, for the
/// gate's rigorous enclosure tier.
///
/// The residual `d/dx F − f` carries a `1/√P` factor and is therefore unbounded
/// at every root of `P`: no rigorous bound over a box that touches a root can
/// exist, and asking for one only wastes the subdivision budget.  The same
/// goes for the interior points `region` cuts out.  So each component of the
/// region is clipped to a finite window, inset away from its finite endpoints,
/// and the widest survivor comes first (the gate takes them in order, up to its
/// `max_boxes`).
///
/// This is the domain-awareness of [`gate_samples`], expressed as a box instead
/// of a point set — both are built from the same [`Region`], because an
/// enclosure over a wider set would certify something the reduction never
/// claimed.
fn gate_boxes(p_coeffs: &[f64], region: &Region) -> Vec<(f64, f64)> {
    // A finite window to clip an unbounded component to.  It only has to be
    // wide enough to hold the interesting part of the curve; the region's own
    // endpoints, not this, are what the boxes are inset from.
    let window = poly_roots(p_coeffs)
        .map(|roots| classify_roots(&roots).0)
        .unwrap_or_default()
        .iter()
        .fold(1.0_f64, |m, r| m.max(r.abs()))
        + 3.0;

    let mut boxes: Vec<(f64, f64)> = Vec::new();
    for (mut lo, mut hi) in region.components(window, 0.3) {
        // Inset away from an endpoint the region itself put there (a root of
        // `P`, or a cut point); the artificial `±window` clips are ordinary
        // interior points and need no inset.
        let inset = (0.12 * (hi - lo)).clamp(0.05, 1.0);
        if lo > -window {
            lo += inset;
        }
        if hi < window {
            hi -= inset;
        }
        if hi - lo < 0.2 {
            continue;
        }
        // Keep only components on which `P` is actually positive; sample a few
        // interior points rather than only the midpoint.
        if ![0.15_f64, 0.5, 0.85]
            .iter()
            .all(|f| eval_poly(p_coeffs, lo + (hi - lo) * f) > 1e-3)
        {
            continue;
        }
        // A wide box is cheap to state and expensive to certify: the Taylor
        // remainder grows with the box, so the branch-and-bound budget can run
        // out before the tolerance is met.  Offer a *narrower concentric
        // fallback* as well.  The gate takes boxes in order and keeps the ones
        // it can actually certify, so this trades coverage for a verdict only
        // when the wide box does not work out.
        let mid = 0.5 * (lo + hi);
        if hi - lo > 4.4 {
            boxes.push((mid - 2.2, mid + 2.2));
        } else {
            boxes.push((lo, hi));
        }
        if hi - lo > 2.6 {
            boxes.push((mid - 1.0, mid + 1.0));
        }
    }
    boxes.sort_by(|a, b| {
        (b.1 - b.0)
            .partial_cmp(&(a.1 - a.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    boxes
}

/// The gate's view of "where this reduction is claimed to hold": the sample
/// grid drawn from `region`, the conjunction of `region` and `P > 0` as the
/// in-domain predicate, and enclosure boxes inside the same set.
///
/// Both halves of the predicate are load-bearing and neither implies the other.
/// `region` is where the substitution is real; `P > 0` is where the integrand
/// is.  They coincide for some root configurations and are disjoint for others
/// (a normal form applied with the wrong sign of leading coefficient), and the
/// gate must sample the intersection — which is exactly the set on which
/// `d/dx F = f` is being asserted.
fn elliptic_domain(p_coeffs: &[f64], region: Region) -> gate::Domain<'_> {
    let samples = gate_samples(p_coeffs, &region);
    let boxes = gate_boxes(p_coeffs, &region);
    gate::Domain::from_samples(samples)
        .with_predicate(move |x: f64| region.contains(x) && eval_poly(p_coeffs, x) > 1e-6)
        .with_boxes(boxes)
}

/// Default gate configuration for this route.
///
/// The acceptance decision is the `f64` screen at `1e-7` relative over ≥ 3
/// in-domain points — bit-for-bit the historical gate.  The symbolic tier is
/// on: it can only *strengthen* the verdict (a syntactic zero residual is a
/// proof of the identity), never widen what is emitted.
///
/// The rigorous enclosure tier is **off by default**, and that is a cost
/// decision, not a soundness one.  Measured on this machine, in a release
/// build, on the first-kind candidate for `∫dx/√(x³+1)`:
///
/// | tier | wall time |
/// |---|---|
/// | symbolic + `f64` screen (≈ 20 in-domain points) | ~0.4 ms |
/// | + [`gate::EnclosureBudget::cheap`] | ~0.9 s, and it does **not** reach the tolerance |
/// | + [`gate::EnclosureBudget::thorough`] | 1.3 s – 9.9 s, and it does |
///
/// A three-order-of-magnitude tax on every elliptic emission, for evidence
/// that cannot change the answer, is the wrong default.  Callers who want it
/// ask for it: [`try_elliptic_output_with`] and
/// [`try_elliptic_output_higher_kind_with`] take a [`gate::GateOptions`], and
/// `first_kind_candidates_reach_a_rigorous_enclosure` in this module's tests
/// exercises the rigorous path on real reductions and records the boxes and
/// residual bounds it certifies.
fn gate_options() -> gate::GateOptions {
    gate::GateOptions {
        tolerance: 1e-7,
        min_points: 3,
        symbolic: true,
        egraph: false,
        enclosure: gate::EnclosurePolicy::Skip,
        min_strength: gate::Strength::Sampled,
    }
}

/// Verify `d/dx F_cand = c/√P` on `region ∩ {P > 0}` — the set this reduction
/// claims, not the whole of `{P > 0}`.
///
/// `c_expr` is the exact symbolic constant (`c` is its `f64` value), so the
/// symbolic and enclosure tiers see the true integrand rather than a float
/// reconstruction of it.
#[allow(clippy::too_many_arguments)]
fn verify(
    f_cand: ExprId,
    coeffs: &[f64],
    c: f64,
    c_expr: ExprId,
    p_expr: ExprId,
    var: ExprId,
    region: Region,
    pool: &ExprPool,
    gate_opts: &gate::GateOptions,
) -> gate::Verdict {
    let sqrt_p = pool.func("sqrt", vec![p_expr]);
    let integrand = simplify(
        pool.mul(vec![c_expr, pool.pow(sqrt_p, pool.integer(-1_i32))]),
        pool,
    )
    .value;
    let rhs = |xv: f64| -> Option<f64> {
        let pv = eval_poly(coeffs, xv);
        if pv <= 1e-6 {
            return None;
        }
        Some(c / pv.sqrt())
    };
    let domain = elliptic_domain(coeffs, region);
    let target = gate::Target::symbolic(integrand).with_numeric(&rhs);
    gate::verify(f_cand, &target, var, &domain, gate_opts, pool)
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

/// Build an `ExprId` for an `f64` constant.
///
/// The reduction constants `g`, `m`, the Legendre substitution's root offsets and
/// the fitted block coefficients are computed numerically, but they are almost
/// always **simple algebraic numbers** — `√3`, `3^(-1/4)`, `(2+√3)/4`, … — not
/// arbitrary floats.  Reconstructing them with `rug::Rational::from_f64` is
/// *exact for the float* but yields ugly `…/2⁵³` denominators that merely
/// approximate the true constant (e.g. `∫dx/√(x³+1)` printed `√3` as
/// `3900231685776981/2251799813685248`).
///
/// [`pretty_const`] first tries to recognize `v` as one of those simple closed
/// forms and emit it symbolically; only when nothing matches do we fall back to
/// the exact float→rational reconstruction (preserving the previous behaviour for
/// genuinely irrational-with-no-simple-form constants).  This is purely a
/// *display* improvement: the soundness gate (`verify` / `verify_higher`)
/// re-checks `d/dx F = integrand` numerically afterwards, so a mis-recognition
/// can only make the path *decline*, never emit a wrong answer.
pub(super) fn float_to_expr(v: f64, pool: &ExprPool) -> ExprId {
    // Exact small integers stay integer nodes.
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        return pool.integer(v as i32);
    }
    if let Some(e) = pretty_const(v, pool) {
        return e;
    }
    match rug::Rational::from_f64(v) {
        Some(r) => {
            let (num, den) = r.into_numer_denom();
            pool.rational(num, den)
        }
        None => pool.integer(0_i32),
    }
}

/// Tolerance for accepting a recognized closed form (relative).  Kept tight: the
/// reduction constants carry only root-finder / float round-off (≈1e-13), so a
/// genuine simple form matches to well under this, while an unrelated float will
/// not coincide with a low-height algebraic number to this precision.
const PRETTY_TOL: f64 = 1e-11;

/// Emit `num/den` as a reduced integer or rational `ExprId` (`den` may be any
/// non-zero sign; `rug::Rational` canonicalizes).
fn rat_expr(num: i64, den: i64, pool: &ExprPool) -> ExprId {
    let r = rug::Rational::from((rug::Integer::from(num), rug::Integer::from(den)));
    if r.is_integer() {
        return pool.integer(r.numer().clone());
    }
    let (n, d) = r.into_numer_denom();
    pool.rational(n, d)
}

/// `coeff · factor`, collapsing the trivial `coeff = ±1` cases for clean display.
fn scale(coeff: (i64, i64), factor: ExprId, pool: &ExprPool) -> ExprId {
    if coeff == (1, 1) {
        return factor;
    }
    if coeff == (-1, 1) {
        return pool.mul(vec![pool.integer(-1_i32), factor]);
    }
    pool.mul(vec![rat_expr(coeff.0, coeff.1, pool), factor])
}

/// Best simple rational `p/q` (reduced, `q ≤ max_den`) within `PRETTY_TOL` of `v`,
/// via continued-fraction convergents.  `None` if no such rational is that close.
fn as_rational(v: f64, max_den: i64) -> Option<(i64, i64)> {
    if !v.is_finite() {
        return None;
    }
    let sign = if v < 0.0 { -1 } else { 1 };
    let x = v.abs();
    let (mut h0, mut k0, mut h1, mut k1) = (0i64, 1i64, 1i64, 0i64);
    let mut b = x;
    for _ in 0..48 {
        let a = b.floor();
        if !a.is_finite() || a.abs() > 1e15 {
            break;
        }
        let ai = a as i64;
        let h2 = ai.checked_mul(h1)?.checked_add(h0)?;
        let k2 = ai.checked_mul(k1)?.checked_add(k0)?;
        if k2 <= 0 || k2 > max_den {
            break;
        }
        h0 = h1;
        k0 = k1;
        h1 = h2;
        k1 = k2;
        if (h1 as f64 / k1 as f64 - x).abs() <= PRETTY_TOL * (1.0 + x) {
            return Some((sign * h1, k1));
        }
        let frac = b - a;
        if frac.abs() < 1e-15 {
            break;
        }
        b = 1.0 / frac;
    }
    None
}

/// Whether `n` is squarefree (so `√n` is genuinely irrational and not reducible
/// to a smaller radical).
fn is_squarefree(mut n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2i64;
    while d * d <= n {
        if n % (d * d) == 0 {
            return false;
        }
        if n % d == 0 {
            n /= d;
        } else {
            d += 1;
        }
    }
    true
}

/// Whether `n^{1/4}` is a sensible canonical radical to emit: `n` must be a
/// non-square (else `n^{1/4} = √(√n)` reduces to a `√` form) and **4th-power-free**
/// (no `d⁴ ∣ n`, else `n^{1/4}` pulls out an integer factor).  This *includes*
/// non-squarefree `n` like `12` — `12^{-1/4}` is exactly the `∫dx/√(x³+8)`
/// coefficient `(2√3)^{-1/2}`, which the squarefree `√`/`n^{1/4}` forms miss.
fn is_quartic_radical(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let r = (n as f64).sqrt().round() as i64;
    if r * r == n {
        return false; // perfect square → use the √ form instead
    }
    let mut d = 2i64;
    while d * d * d * d <= n {
        if n % (d * d * d * d) == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// Recognize `v` as a simple algebraic constant and build it symbolically, else
/// `None` (caller falls back to exact float→rational reconstruction).
///
/// Forms tried, in increasing complexity (first match wins):
///   1. simple rational `p/q`;
///   2. `(p/q)·√n`            (`n` squarefree);
///   3. `(p/q)·n^{±1/4}`      (`n` a 4th-power-free non-square);
///   4. `a/q + (b/q)·√n`      (`a + b√n` over a common denominator);
///   5. `(a/q)·√m + (b/q)·√n` (two distinct `√`, e.g. `2(√3−√2)`, `(√2+√3)/2`).
fn pretty_const(v: f64, pool: &ExprPool) -> Option<ExprId> {
    if !v.is_finite() || v == 0.0 {
        return None;
    }

    // 1) simple rational.
    if let Some(pq) = as_rational(v, 4096) {
        return Some(rat_expr(pq.0, pq.1, pool));
    }

    let squarefree: Vec<i64> = (2..=50).filter(|&n| is_squarefree(n)).collect();

    // 2) (p/q)·√n.
    for &n in &squarefree {
        let sn = (n as f64).sqrt();
        if let Some(pq) = as_rational(v / sn, 256) {
            let sqrt_n = pool.func("sqrt", vec![pool.integer(n as i32)]);
            return Some(scale(pq, sqrt_n, pool));
        }
    }

    // 3) (p/q)·n^{±1/4}.
    for n in 2..=50i64 {
        if !is_quartic_radical(n) {
            continue;
        }
        let q4 = (n as f64).powf(0.25);
        if let Some(pq) = as_rational(v / q4, 64) {
            let r = pool.pow(pool.integer(n as i32), rat_expr(1, 4, pool));
            return Some(scale(pq, r, pool));
        }
        if let Some(pq) = as_rational(v * q4, 64) {
            let r = pool.pow(pool.integer(n as i32), rat_expr(-1, 4, pool));
            return Some(scale(pq, r, pool));
        }
    }

    // 4) a + b·√n over a common denominator q (catches e.g. `(2+√3)/4`).
    for q in 1..=24i64 {
        let w = v * q as f64;
        for &n in &squarefree {
            let sn = (n as f64).sqrt();
            for bnum in -32..=32i64 {
                if bnum == 0 {
                    continue;
                }
                let a = w - bnum as f64 * sn;
                let ar = a.round();
                if ar.abs() <= 1.0e9 && (a - ar).abs() <= PRETTY_TOL * (1.0 + w.abs()) {
                    let a_e = rat_expr(ar as i64, q, pool);
                    let sqrt_n = pool.func("sqrt", vec![pool.integer(n as i32)]);
                    let b_e = scale((bnum, q), sqrt_n, pool);
                    return Some(pool.add(vec![a_e, b_e]));
                }
            }
        }
    }

    // 5) (a/q)·√m + (b/q)·√n with distinct squarefree m < n, over a common
    //    denominator q.  Catches constants in `ℚ(√m, √n)` that the single-radical
    //    forms miss — e.g. the four-real-root quartic with roots `±√2, ±√3`, whose
    //    `g = 2(√3−√2)` and `sin²φ` coefficient `(√2+√3)/2` are otherwise floats.
    for q in 1..=16i64 {
        let w = v * q as f64;
        for (i, &m) in squarefree.iter().enumerate() {
            if m > 30 {
                break;
            }
            let sm = (m as f64).sqrt();
            for &n in &squarefree[i + 1..] {
                if n > 30 {
                    break;
                }
                let sn = (n as f64).sqrt();
                for bnum in -24..=24i64 {
                    if bnum == 0 {
                        continue;
                    }
                    let af = (w - bnum as f64 * sn) / sm;
                    let ar = af.round();
                    if ar != 0.0
                        && ar.abs() <= 1.0e9
                        && (af - ar).abs() <= PRETTY_TOL * (1.0 + w.abs())
                    {
                        let sqrt_m = pool.func("sqrt", vec![pool.integer(m as i32)]);
                        let sqrt_n = pool.func("sqrt", vec![pool.integer(n as i32)]);
                        let a_e = scale((ar as i64, q), sqrt_m, pool);
                        let b_e = scale((bnum, q), sqrt_n, pool);
                        return Some(pool.add(vec![a_e, b_e]));
                    }
                }
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Complex root finding (Durand–Kerner) + classification
// ---------------------------------------------------------------------------

/// Find all complex roots of a polynomial with ascending real coefficients
/// (degree 3 or 4) via Durand–Kerner iteration.
pub(super) fn poly_roots(coeffs: &[f64]) -> Option<Vec<Croot>> {
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
pub(super) fn classify_roots(roots: &[Croot]) -> (Vec<f64>, Vec<Croot>) {
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
        // The reduction constants are √3 / 3^(-1/4) / (2+√3)/4, *not* float
        // reconstructions: the output must be free of the giant 2⁵³-scale
        // denominators that the old float→rational path produced.
        assert!(
            !s.contains("9007199254740992") && !s.contains("2251799813685248"),
            "elliptic constants leaked a float reconstruction: {s}"
        );
        assert!(
            s.contains("sqrt(3)") || s.contains('√'),
            "expected an exact √3: {s}"
        );
    }

    #[test]
    fn pretty_const_recognizes_simple_algebraic_numbers() {
        let pool = ExprPool::new();
        // √3, 3^(-1/4), (2+√3)/4, 2/3 — the constants that show up in the
        // ∫dx/√(x³+1) reduction — must round-trip to a clean symbolic form whose
        // value matches and whose printout carries no float-reconstruction junk.
        let cases = [
            (3.0_f64.sqrt(), "sqrt(3)"),
            (3.0_f64.powf(-0.25), ""),
            ((2.0 + 3.0_f64.sqrt()) / 4.0, "sqrt(3)"),
            (2.0 / 3.0, ""),
            // 12^(-1/4) = (2√3)^(-1/2): the ∫dx/√(x³+8) coefficient (non-squarefree
            // 4th-power-free radicand).
            (12.0_f64.powf(-0.25), ""),
            // 1+√2: the normalized ∫dx/√(x⁴+1) atan Möbius coefficient.
            (1.0 + 2.0_f64.sqrt(), "sqrt(2)"),
            // 2√3−2√2 and (√2+√3)/2: ℚ(√2,√3) constants from the four-real-root
            // quartic with roots ±√2, ±√3 (∫dx/√(x⁴−5x²+6)).
            (2.0 * 3.0_f64.sqrt() - 2.0 * 2.0_f64.sqrt(), "sqrt(3)"),
            ((2.0_f64.sqrt() + 3.0_f64.sqrt()) / 2.0, "sqrt(2)"),
        ];
        for (v, needle) in cases {
            let e = float_to_expr(v, &pool);
            let got = eval(e, x_dummy(&pool), 0.0, &pool).expect("evaluable");
            assert!(
                (got - v).abs() <= 1e-10 * (1.0 + v.abs()),
                "value drift for {v}"
            );
            let s = pool.display(e).to_string();
            assert!(
                !s.contains("9007199254740992") && !s.contains("2251799813685248"),
                "float reconstruction leaked for {v}: {s}"
            );
            if !needle.is_empty() {
                assert!(s.contains(needle), "expected {needle} in {s}");
            }
        }
    }

    /// A throwaway symbol so constant-only expressions can be fed to `eval`.
    /// Ascending-`f64`-coefficient polynomial as an `ExprId`.
    fn poly_expr(coeffs: &[f64], var: ExprId, pool: &ExprPool) -> ExprId {
        let terms: Vec<ExprId> = coeffs
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(j, &v)| {
                let xj = match j {
                    0 => pool.integer(1_i32),
                    1 => var,
                    _ => pool.pow(var, pool.integer(j as i32)),
                };
                pool.mul(vec![float_to_expr(v, pool), xj])
            })
            .collect();
        if terms.is_empty() {
            pool.integer(0_i32)
        } else {
            pool.add(terms)
        }
    }

    /// The pre-refactor boolean `verify_higher`, rebuilt on the shared gate so
    /// the soundness assertions in this module keep reading the same.  Same
    /// domain, same tolerance, same minimum point count.
    fn verify_higher(
        f_cand: ExprId,
        p_coeffs: &[f64],
        b_num: &[f64],
        b_den: &[f64],
        var: ExprId,
        pool: &ExprPool,
    ) -> bool {
        let p_expr = poly_expr(p_coeffs, var, pool);
        let sqrt_p = pool.func("sqrt", vec![p_expr]);
        let integrand = simplify(
            pool.mul(vec![
                poly_expr(b_num, var, pool),
                pool.pow(poly_expr(b_den, var, pool), pool.integer(-1_i32)),
                sqrt_p,
            ]),
            pool,
        )
        .value;
        let rhs = |xv: f64| -> Option<f64> {
            let pv = eval_poly(p_coeffs, xv);
            if pv <= 1e-6 {
                return None;
            }
            Some(eval_ratio(b_num, b_den, xv)? * pv.sqrt())
        };
        let domain = elliptic_domain(p_coeffs, region_of(p_coeffs, var, pool));
        let target = gate::Target::symbolic(integrand).with_numeric(&rhs);
        gate::verify(f_cand, &target, var, &domain, &gate_options(), pool).is_verified()
    }

    fn x_dummy(pool: &ExprPool) -> ExprId {
        pool.symbol("__unused__", Domain::Real)
    }

    /// The region the first-kind reduction for `p_coeffs` claims, taken from
    /// [`first_kind_reduction`] itself rather than restated here — a test that
    /// hard-coded the interval would keep passing after the reduction changed.
    fn region_of(p_coeffs: &[f64], var: ExprId, pool: &ExprPool) -> Region {
        let deg = p_coeffs.len() - 1;
        let lead = *p_coeffs.last().expect("non-empty coefficients");
        first_kind_reduction(p_coeffs, deg, lead, var, pool)
            .expect("a handled cubic/quartic root configuration")
            .3
    }

    /// The gate's domain for a three-real-root cubic is the component the
    /// reduction claims, and nothing else.
    ///
    /// `P = x³ − x` is positive on `(−1, 0)` and on `(1, ∞)`; the B&F
    /// `sin²φ = (e1−e3)/(x−e3)` normal form is real only on the second.  The
    /// grid used to cover both and rely on the gate skipping the `NaN`s, which
    /// is exactly what stopped the gate from being allowed to treat a `NaN`
    /// derivative as a disagreement.  Nothing may be sampled in `(−1, 0)`, and
    /// the surviving points must still clear the gate's three-point minimum.
    #[test]
    fn gate_domain_is_the_claimed_component_only() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let coeffs = [0.0, -1.0, 0.0, 1.0]; // x³ − x
        let region = region_of(&coeffs, x, &pool);
        assert_eq!(region, Region::open(1.0, f64::INFINITY));

        let domain = elliptic_domain(&coeffs, region);
        let live: Vec<f64> = domain
            .samples()
            .iter()
            .copied()
            .filter(|&v| domain.contains(v))
            .collect();
        assert!(
            live.len() >= 3,
            "narrowing must not starve the gate: {live:?}"
        );
        for &v in &live {
            assert!(
                v > 1.0,
                "sampled {v}, which is outside the claimed region (1, ∞)"
            );
        }
        // The other `P > 0` component is positive and *not* claimed.
        for &v in &[-0.9_f64, -0.5, -0.1] {
            assert!(eval_poly(&coeffs, v) > 0.0, "{v} should have P > 0");
            assert!(!domain.contains(v), "{v} is in the domain but not claimed");
        }
        for b in domain.boxes() {
            assert!(b.0 > 1.0, "enclosure box {b:?} leaves the claimed region");
        }
    }

    /// The no-real-root quartic cuts the pole of its `arctan` substitution.
    ///
    /// `φ = atan(L)` with `L` a Möbius function has one real pole; `φ` jumps by
    /// `π` across it and the written derivative is `0/0` there.  The region
    /// removes it, so the two components either side are claimed and the point
    /// itself is not — a `Region` fact, not a budget accident of the enclosure
    /// tier.
    #[test]
    fn quartic_no_real_region_cuts_the_substitution_pole() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let coeffs = [1.0, 0.0, 0.0, 0.0, 1.0]; // x⁴ + 1
        let region = region_of(&coeffs, x, &pool);
        assert_eq!(region.cuts.len(), 1, "expected exactly one cut: {region:?}");
        let pole = region.cuts[0];
        assert!(!region.contains(pole));
        assert!(region.contains(pole - 0.5) && region.contains(pole + 0.5));
        let domain = elliptic_domain(&coeffs, region);
        for b in domain.boxes() {
            assert!(
                pole <= b.0 || pole >= b.1,
                "enclosure box {b:?} straddles the substitution pole {pole}"
            );
        }
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
    fn cubic_three_real_narrow_region_emits_ellipticf() {
        // Regression for the gate's region-aware sampling: (x+1)(x+2)(x−3) =
        // x³ − 7x − 6 has its valid reduction region at x ≥ 3, far from the old
        // fixed sample grid's center.  Before `gate_samples` this *spuriously
        // declined* (only 2 fixed grid points fell in x ≥ 3, below the 3 required);
        // now it emits a gate-verified EllipticF.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-7_i32), x]),
            pool.integer(-6_i32),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(x³−7x−6) should emit EllipticF (region x ≥ 3)");
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

    /// The graded gate on real elliptic candidates: for each first-kind
    /// reduction, report the strongest verdict the tiered gate reaches with a
    /// generous enclosure budget.
    ///
    /// This is the test that proves the rigorous tier is not decorative — the
    /// residual `d/dx[g·EllipticF(φ(x),m)] − c/√P` is bounded *over a whole
    /// interval* by Taylor models in outward-rounded ball arithmetic, which
    /// point sampling can never do.  It is also the honest record of where the
    /// rigorous tier does **not** reach: any case that comes back
    /// `SampledOnly` is listed as such rather than being quietly dropped.
    #[test]
    fn first_kind_candidates_reach_a_rigorous_enclosure() {
        let cases: [(&str, &[f64]); 4] = [
            // x³ + 1  (cubic, one real root)
            ("x^3+1", &[1.0, 0.0, 0.0, 1.0]),
            // x³ − x  (cubic, three real roots)
            ("x^3-x", &[0.0, -1.0, 0.0, 1.0]),
            // x⁴ − 5x² + 4 = (x²−1)(x²−4)  (quartic, four real roots)
            ("x^4-5x^2+4", &[4.0, 0.0, -5.0, 0.0, 1.0]),
            // x⁴ + 1  (quartic, no real root)
            ("x^4+1", &[1.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let mut enclosed = 0;
        for (name, coeffs) in cases {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let p = poly_expr(coeffs, x, &pool);
            let zero = pool.integer(0_i32);
            let b = pool.pow(p, pool.integer(-1_i32));
            let Some(f) = try_elliptic_output(zero, b, p, x, &pool) else {
                panic!("{name}: expected an EllipticF emission");
            };
            let opts = gate::GateOptions {
                enclosure: gate::EnclosurePolicy::BestEffort(gate::EnclosureBudget {
                    order: 6,
                    prec: 96,
                    tol: 1e-7,
                    max_subdivisions: 96,
                    max_boxes: 8,
                }),
                ..gate_options()
            };
            let sqrt_p = pool.func("sqrt", vec![p]);
            let integrand = simplify(pool.pow(sqrt_p, pool.integer(-1_i32)), &pool).value;
            let rhs = |xv: f64| -> Option<f64> {
                let pv = eval_poly(coeffs, xv);
                if pv <= 1e-6 {
                    return None;
                }
                Some(1.0 / pv.sqrt())
            };
            let domain = elliptic_domain(coeffs, region_of(coeffs, x, &pool));
            assert!(
                !domain.boxes().is_empty(),
                "{name}: no in-domain box was produced"
            );
            let target = gate::Target::symbolic(integrand).with_numeric(&rhs);
            let verdict = gate::verify(f, &target, x, &domain, &opts, &pool);
            assert!(
                verdict.is_verified(),
                "{name}: the emitted form must verify, got {verdict:?}"
            );
            if let gate::Verdict::EnclosureVerified {
                boxes,
                residual_bound,
                ..
            } = &verdict
            {
                enclosed += 1;
                assert!(*residual_bound <= 1e-7, "{name}: bound {residual_bound:e}");
                assert!(!boxes.is_empty());
                // Every certified box must lie strictly inside `P > 0`.
                for b in boxes {
                    for f in [0.0_f64, 0.25, 0.5, 0.75, 1.0] {
                        let xv = b.lo + (b.hi - b.lo) * f;
                        assert!(
                            eval_poly(coeffs, xv) > 0.0,
                            "{name}: box [{}, {}] leaves the domain at {xv}",
                            b.lo,
                            b.hi
                        );
                    }
                }
            }
        }
        assert!(
            enclosed >= 3,
            "expected the rigorous tier to certify at least three of the four \
             first-kind reductions, it certified {enclosed}"
        );
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
            // Skip removable singularities of the *derivative representation*
            // (e.g. the `atan` Möbius pole at `x = −1` for `√(x⁴+1)`, where the
            // exact `(−1+√2)(x+1)` denominator vanishes and `L'/(1+L²)` evaluates
            // to `∞/∞`).  The antiderivative is fine there; the production gate
            // `verify_higher` skips such points the same way.
            if !lhs.is_finite() || !rhs.is_finite() {
                continue;
            }
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
    fn third_kind_cubic_one_real_emits_pi_and_log() {
        // Headline (this PR): ∫ dx/((x−2)√(x³+1)).  The `acos`/cosφ substitution
        // makes sin²φ a *quadratic* rational of x, so a single EllipticPi has a
        // spurious twin pole (here at x=0).  Adding the twin's elementary log
        // blocks (`log|x|`, `log(√P+1)`) lets the fit close:
        //   F = δ·Π + β·F + ε·log(√P+1) + ζ·log|x|  (gate-verified).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let x_minus = pool.add(vec![x, pool.integer(-2_i32)]);
        let den = pool.mul(vec![x_minus, p]);
        let b = pool.pow(den, pool.integer(-1_i32));
        // b_den = (x−2)·P, ascending.
        let p_coeffs = [1.0, 0.0, 0.0, 1.0];
        let mut b_den = vec![0.0; p_coeffs.len() + 1];
        for (j, &c) in p_coeffs.iter().enumerate() {
            b_den[j + 1] += c;
            b_den[j] += -2.0 * c;
        }
        let s = check_higher(
            p,
            b,
            x,
            &["EllipticPi", "log"],
            &[1.0],
            &b_den,
            &p_coeffs,
            &[1.2, 1.6, 2.4, 2.8, 3.5, 4.0, 5.0],
            &pool,
        );
        assert!(s.contains("EllipticPi"), "{s}");
        assert!(s.contains("log"), "{s}");
    }

    #[test]
    fn engine_integrate_third_kind_cubic_one_real_emits_pi() {
        // End-to-end through the engine: ∫ dx/((x−2)√(x³+1)) → EllipticPi + log
        // form, with d/dx matching the integrand on x>−1, x≠2.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let sqrt_p = pool.func("sqrt", vec![p]);
        let x_minus = pool.add(vec![x, pool.integer(-2_i32)]);
        let den = pool.mul(vec![x_minus, sqrt_p]);
        let integrand = pool.pow(den, pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫ dx/((x−2)√(x³+1)) should integrate to an elliptic form");
        let s = pool.display(res.value).to_string();
        assert!(s.contains("EllipticPi"), "expected EllipticPi, got {s}");
        let ds = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in &[1.2_f64, 1.6, 2.4, 2.8, 3.5, 4.0] {
            let pv: f64 = xv * xv * xv + 1.0;
            if pv <= 1e-6 {
                continue;
            }
            let rhs = 1.0 / ((xv - 2.0) * pv.sqrt());
            let lhs = eval(ds, x, xv, &pool).unwrap();
            assert!((lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()), "x={xv}");
            checked += 1;
        }
        assert!(checked >= 3);
    }

    #[test]
    fn third_kind_cubic_one_real_plus2_emits_or_declines_soundly() {
        // ∫ dx/((x+2)√(x³+1)): the pole x=−2 lies where P(−2)=−7<0, outside the
        // φ domain — `characteristic_from_pole` returns NaN so no Π block is added
        // and the path declines.  (Kept as a soundness assertion: never emits an
        // unverified form.  If a future reduction handles it the form must still
        // gate-verify.)
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let x_plus = pool.add(vec![x, pool.integer(2_i32)]);
        let den = pool.mul(vec![x_plus, p]);
        let b = pool.pow(den, pool.integer(-1_i32));
        let zero = pool.integer(0_i32);
        if let Some(f) = try_elliptic_output_higher_kind(zero, b, p, x, &pool) {
            // If something is emitted it must be gate-correct.
            let b_num = [1.0];
            let mut b_den = vec![0.0; 5];
            for (j, &c) in [1.0, 0.0, 0.0, 1.0].iter().enumerate() {
                b_den[j + 1] += c;
                b_den[j] += 2.0 * c;
            }
            assert!(verify_higher(
                f,
                &[1.0, 0.0, 0.0, 1.0],
                &b_num,
                &b_den,
                x,
                &pool
            ));
        }
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
        // The `atan` substitution's Möbius coefficients are normalized so they
        // print as exact `1±√2` constants, not `2⁵³`-scale float reconstructions.
        assert!(
            !s.contains("9007199254740992")
                && !s.contains("2251799813685248")
                && !s.contains("4503599627370496")
                && !s.contains("1125899906842624"),
            "atan Möbius coefficients leaked a float reconstruction: {s}"
        );
        assert!(
            s.contains("sqrt(2)") || s.contains('√'),
            "expected an exact √2: {s}"
        );
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
    fn quartic_four_real_irrational_roots_emits_clean() {
        // ∫dx/√(x⁴−5x²+6), P = (x²−2)(x²−3): four irrational real roots ±√2, ±√3.
        // The substitution constants live in ℚ(√2,√3) — `g = 2(√3−√2)`, `sin²φ`
        // coefficient `(√2+√3)/2` — so they exercise the two-radical recognizer and
        // must print exactly, with no float-reconstruction denominators.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.mul(vec![pool.integer(-5_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(6_i32),
        ]);
        let s = check_emits(p, x, 1.0, &pool).expect("∫dx/√(x⁴−5x²+6) should emit EllipticF");
        assert!(
            !s.contains("9007199254740992")
                && !s.contains("4503599627370496")
                && !s.contains("1125899906842624"),
            "ℚ(√2,√3) constants leaked a float reconstruction: {s}"
        );
        assert!(
            s.contains("sqrt(2)") && s.contains("sqrt(3)"),
            "expected √2 and √3: {s}"
        );
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

    // ── Decline-stability: genus-1 configs that remain NonElementary ─────────
    //
    // The following integrals are gate-safe *declines* — the available real
    // `F`/`E`/`Π`/algebraic/elementary-log basis cannot represent them (see the
    // diagnosis in `try_elliptic_output_higher_kind`'s THIRD KIND comment and the
    // `(4,0,2)` arm of `reduction_poles`), so the path returns `None` and the
    // caller falls through to `NonElementary`.  These tests pin that the path
    // never *emits* a (necessarily wrong) closed form, guarding the soundness gate
    // against future basis changes that might fit numerically but mis-verify.

    #[test]
    fn x2_over_sqrt_x4_plus_1_declines() {
        // ∫ x²/√(x⁴+1) dx.  The (4,0,2) arctan config's `sin²φ = L²/(1+L²)` has its
        // EllipticPi characteristic tied to the (complex-rooted) `den_E` quadratic,
        // so no *real* `Π` aligns; F/E/algebraic alone is insufficient (derivative-
        // gate residual ≳ 0.16 even with two `Π` + the rich algebraic ladder).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(4_i32)), pool.integer(1_i32)]);
        let zero = pool.integer(0_i32);
        // integrand x²/√P = b·√P with b = x²/P.
        let b = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(p, pool.integer(-1_i32)),
        ]);
        assert!(
            try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none(),
            "∫x²/√(x⁴+1) must decline (no real Π characteristic for the arctan config)"
        );
    }

    #[test]
    fn sqrt_x4_plus_x2_plus_1_declines() {
        // ∫ √(x⁴+x²+1) dx — non-symmetric all-complex quartic.  The fixed
        // first-kind (g,m,φ) does not also linearize this second-kind integrand;
        // the F/E/algebraic basis is insufficient and there is no aligned real `Π`.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.pow(x, pool.integer(2_i32)),
            pool.integer(1_i32),
        ]);
        let zero = pool.integer(0_i32);
        let b = pool.integer(1_i32); // integrand = 1·√P
        assert!(
            try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none(),
            "∫√(x⁴+x²+1) must decline (non-symmetric quartic, basis insufficient)"
        );
    }

    #[test]
    fn quartic_two_real_third_kind_declines() {
        // ∫ dx/((x−½)√P), P = −x⁴−x³+x+1 = (1−x²)(x²+x+1): two real roots ±1 + a
        // complex pair (the quartic two-real cos φ config).  The pole p=½ and its
        // twin t=−⅘ share the same characteristic, and the twin third-kind integral
        // ∫dx/((x−t)√P) is itself NON-elementary for a quartic (unlike the cubic
        // one-real case PR7 closed), so the single `Π` cannot be isolated — declines.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.integer(1_i32),
            x,
            pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(3_i32))]),
            pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(4_i32))]),
        ]);
        let zero = pool.integer(0_i32);
        // b = 1/(x−½); integrand = √P/(x−½).
        let xp = pool.add(vec![x, pool.rational(-1, 2)]);
        let b = pool.pow(xp, pool.integer(-1_i32));
        assert!(
            try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none(),
            "quartic two-real third kind must decline (twin integral non-elementary)"
        );
    }

    #[test]
    fn cubic_one_real_nonelementary_twin_declines() {
        // ∫ dx/((x−3)√(x³+1)) — cubic one-real cos φ config whose twin t=−¼ has a
        // twin third-kind integral ∫dx/((x−t)√P) that is NOT elementary, so PR7's
        // elementary-log augmented basis (`twin_log`/`elem_log_blocks`) still cannot
        // close it (it closes only when that twin integral *is* elementary, e.g. the
        // headline ∫dx/((x−2)√(x³+1)), twin t=0).  Gate-safe decline (roadmap item 3).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let zero = pool.integer(0_i32);
        let xp = pool.add(vec![x, pool.integer(-3_i32)]);
        let b = pool.pow(xp, pool.integer(-1_i32));
        assert!(
            try_elliptic_output_higher_kind(zero, b, p, x, &pool).is_none(),
            "∫dx/((x−3)√(x³+1)) must decline (non-elementary twin)"
        );
    }
}
