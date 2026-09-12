//! The numerical half: high-precision quadrature, and the evaluator the
//! quadrature calls.
//!
//! This is the instrument the whole module's trustworthiness rests on, so it
//! is built to be able to **fail**. Two things follow from that:
//!
//! * It works at [`VERIFY_PREC`] bits through MPFR, not in `f64`. A closed
//!   form and a quadrature that agree to 15 digits agree for one of two
//!   reasons, and at `f64` precision they are indistinguishable; at 160 bits a
//!   real agreement keeps going and a coincidental one does not.
//! * It **locates the mass before it integrates**. A double-exponential rule
//!   applied blind to `Normal(100, 1)` puts almost no nodes where the density
//!   is, converges to something small, and would then "confirm" any mean at
//!   all. So the integrand is probed on a geometric ladder first, and the rule
//!   is shifted and scaled onto where the integrand actually lives. The probe
//!   reads the integrand, never the formula being checked, so it cannot
//!   launder an error in that formula.
//!
//! Convergence is decided by comparing successive halvings of the step, and a
//! rule that has not settled reports [`QuadOutcome::Inconclusive`] rather than
//! its last iterate — a check that cannot fail is not a check.
//!
//! # The limit of the probe, stated plainly
//!
//! [`locate`] samples a half-octave ladder. A feature narrower than the gap
//! between two rungs — a density concentrated in a width-`10⁻³` spike at
//! `x = 40`, say — is not seen, and the rule then converges, cleanly and
//! quickly, to the integral of the empty part of the line. That is the one
//! failure mode of a located rule that is worse than inaccuracy, because it
//! looks exactly like success.
//!
//! Nothing in *this* file can close that hole; the caller can. `verify` never
//! uses a rule it has not first watched reproduce `∫p = 1`, and a rule that
//! cannot see the density cannot produce `1`. See `verify::prepare_rule`.

use rug::Float;

use crate::ball::{ArbBall, IntervalEval};
use crate::kernel::{ExprId, ExprPool};

/// Working precision, in bits, for every verification quadrature.
///
/// Two things set it, and the larger one wins:
///
/// * the quadrature's own error has to be negligible against the `1e-12`
///   relative agreement the gate demands, which `160` bits already covers;
/// * a tanh–sinh rule on a compact interval places nodes within `e^{-2u}` of
///   the endpoints, and an integrand with an endpoint singularity — the
///   `Beta(½, ½)` arcsine density — reads `1 - x` there. At `192` bits and the
///   [`T_MAX`] below, that difference is around `10⁻⁵⁵` against an absolute
///   resolution of `10⁻⁵⁸`, so it survives; push `T_MAX` out or the precision
///   down and it rounds to zero, the density becomes `0^{-1/2}`, and the rule
///   reports a divergence that is an artefact of its own arithmetic.
pub(crate) const VERIFY_PREC: u32 = 192;

/// Largest `|t|` the double-exponential rule samples.
///
/// Two-sided: past this the weight is below `10⁻⁵⁵` and the node contributes
/// nothing, but going *further* out pushes a compact interval's abscissae
/// closer to its endpoints than [`VERIFY_PREC`] can represent the distance —
/// see the note there.
const T_MAX: f64 = 4.4;

/// Step halvings. Level `j` uses `h = 2⁻¹⁻ʲ`, so the finest rule is
/// `h = 1/1024` — around 9800 nodes.
///
/// The levels **nest**: level `j`'s abscissae contain level `j-1`'s, so each
/// level only evaluates the odd-index nodes and the figure above is the total
/// cost of reaching the *last* level, not the sum over all of them. A run that
/// settles at level four has cost 154 evaluations. Only an integrand that
/// genuinely needs the resolution pays for it, which is what makes it
/// affordable to keep halving until the rule has settled rather than stopping
/// early and calling the last iterate an answer.
const LEVELS: u32 = 10;

/// Relative change between successive levels below which the rule is called
/// converged.
///
/// Eight orders below [`super::verify::REL_TOL`], so the quadrature's own
/// uncertainty is never what the comparison is actually measuring.
const CONVERGED: f64 = 1e-20;

/// How far the integrand must have fallen below its peak for the window to be
/// closed.
///
/// Not "as far as possible": the window sets the *node density* near the mass,
/// while the double-exponential map reaches the whole line whatever the scale
/// is. A window stretched out to where the integrand is `1e-12` of its peak
/// therefore buys nothing in coverage and spends a factor of three in
/// resolution, which is the difference between the rule converging and the
/// rule reporting inconclusive.
const DECAY_FLOOR: f64 = 1e-8;

pub(crate) fn fl(prec: u32, v: f64) -> Float {
    Float::with_val(prec, v)
}

/// `π` as a rigorous ball at `prec` bits.
///
/// The crate spells `π` as an ordinary symbol with no numeric kernel, so every
/// gate that evaluates an expression containing it has to bind it — and has to
/// bind it *as a ball*, not as an `f64` literal whose last bit is a silent
/// approximation. MPFR's constant is correctly rounded, so half an ulp bounds
/// the error; the radius here is four ulps, which is cheap insurance.
pub(crate) fn pi_ball(prec: u32) -> ArbBall {
    let mid = Float::with_val(prec, rug::float::Constant::Pi);
    let rad = Float::with_val(prec, 16u32) >> prec;
    ArbBall { mid, rad, prec }
}

/// Evaluate `expr` with `π` bound and nothing else — `None` when a free symbol
/// or an unsupported head stops the walk.
pub(crate) fn numeric_ball(expr: ExprId, pool: &ExprPool) -> Option<ArbBall> {
    eval_ball(expr, &[], pool, VERIFY_PREC)
}

/// Evaluate `expr` at `bindings`, with `π` bound, as a rigorous ball.
pub(crate) fn eval_ball(
    expr: ExprId,
    bindings: &[(ExprId, Float)],
    pool: &ExprPool,
    prec: u32,
) -> Option<ArbBall> {
    let mut ev = IntervalEval::new(prec);
    ev.bind(super::pi(pool), pi_ball(prec));
    for (sym, val) in bindings {
        ev.bind(
            *sym,
            ArbBall {
                mid: val.clone(),
                rad: fl(prec, 0.0),
                prec,
            },
        );
    }
    let ball = ev.eval(expr, pool)?;
    // An indeterminate ball is the whole real line: it "contains" every value,
    // so letting one through would turn every comparison against it into a
    // pass. Treat it as no answer at all.
    if !ball.mid.is_finite() || !ball.rad.is_finite() {
        return None;
    }
    Some(ball)
}

/// The outcome of evaluating an integrand at one node.
pub(crate) enum Sample {
    /// A usable value.
    Value(Float),
    /// Nothing usable here — an unbound symbol, an unregistered head, or a
    /// value that overflowed. Deliberately *not* read as evidence of
    /// divergence: the outermost abscissae of a double-exponential rule sit
    /// where any density overflows, and treating that as a divergent integral
    /// would refuse every convergent one. Divergence is detected instead by the
    /// `w·f` product overflowing at nodes that are inside the representable
    /// range, and by the edge test in `de_levels`.
    Unevaluable,
}

/// Where to integrate. Endpoints are *values*, already reduced.
#[derive(Clone)]
pub(crate) enum Region {
    /// `(-∞, ∞)`.
    Real,
    /// `(0, ∞)`, integrated on a logarithmic scale.
    Positive,
    /// `[lo, hi]`, integrated by tanh–sinh so an endpoint singularity (the
    /// `Beta(½, ½)` arcsine density, say) is still resolved.
    Interval(Float, Float),
}

/// What a quadrature concluded.
pub(crate) enum QuadOutcome {
    /// A value, with the change between the last two levels as the error
    /// estimate.
    Value { value: Float, est_err: Float },
    /// The integrand blew up where the rule needs it to decay: the integral
    /// does not converge.
    Divergent,
    /// The rule did not settle, or the integrand could not be evaluated. This
    /// is **not** a pass — see the module docs.
    Inconclusive,
}

/// Where the mass of an integrand sits, in the rule's own variable: `x` for
/// [`Region::Real`], `s = log x` for [`Region::Positive`].
#[derive(Clone, Debug)]
pub(crate) struct Window {
    pub centre: Float,
    pub scale: Float,
}

impl Window {
    /// The smallest window containing both — used to make the rule for
    /// `f·p` cover the density's own mass as well as its own, so a `locate`
    /// that found only one of the two cannot silently drop the other.
    pub fn union(&self, other: &Window, prec: u32) -> Window {
        let lo_a = Float::with_val(prec, &self.centre - &self.scale);
        let hi_a = Float::with_val(prec, &self.centre + &self.scale);
        let lo_b = Float::with_val(prec, &other.centre - &other.scale);
        let hi_b = Float::with_val(prec, &other.centre + &other.scale);
        let lo = if lo_a < lo_b { lo_a } else { lo_b };
        let hi = if hi_a > hi_b { hi_a } else { hi_b };
        Window {
            centre: mul(&Float::with_val(prec, &lo + &hi), &fl(prec, 0.5), prec),
            scale: mul(&Float::with_val(prec, &hi - &lo), &fl(prec, 0.5), prec),
        }
    }
}

/// What the probe concluded about where an integrand lives.
pub(crate) enum Located {
    /// Mass found, here.
    At(Window),
    /// The integrand was **exactly** zero at every point probed. Kept apart
    /// from [`Located::NotFound`] because the two need opposite answers: a
    /// vanishing integrand has integral zero, and refusing it would make every
    /// option payoff unverifiable — the half of the support on the wrong side
    /// of the strike is identically zero by construction.
    Vanishes,
    /// Nothing could be evaluated, or the integrand never fell below its own
    /// peak. Not an answer.
    NotFound,
}

/// Integrate `g` over `region`, locating the mass first.
pub(crate) fn quadrature(
    g: &mut impl FnMut(&Float) -> Sample,
    region: Region,
    prec: u32,
) -> QuadOutcome {
    let window = match region {
        Region::Interval(_, _) => None,
        Region::Real | Region::Positive => {
            let log_scale = matches!(region, Region::Positive);
            match locate_where(g, prec, log_scale) {
                Located::At(w) => Some(w),
                Located::Vanishes => {
                    return QuadOutcome::Value {
                        value: fl(prec, 0.0),
                        est_err: fl(prec, 0.0),
                    }
                }
                Located::NotFound => return QuadOutcome::Inconclusive,
            }
        }
    };
    quadrature_at(g, region, window.as_ref(), prec)
}

/// Integrate `g` over `region` using a window the caller has already located.
///
/// Splitting this from [`quadrature`] is what lets the verifier hand the rule
/// a window it has *validated* — see `verify`'s normalisation check — rather
/// than one this file guessed at.
pub(crate) fn quadrature_at(
    g: &mut impl FnMut(&Float) -> Sample,
    region: Region,
    window: Option<&Window>,
    prec: u32,
) -> QuadOutcome {
    match region {
        Region::Interval(lo, hi) => tanh_sinh(g, &lo, &hi, prec),
        Region::Real => match window {
            Some(w) => sinh_sinh(g, &w.centre, &w.scale, prec, false),
            None => QuadOutcome::Inconclusive,
        },
        // s = log x. The Jacobian e^s is folded into the integrand, so the
        // rule sees a function on ℝ. This substitution carries no distribution
        // parameter of any kind, which is what keeps it from agreeing with a
        // parameter mistake in the thing being checked.
        Region::Positive => match window {
            Some(w) => sinh_sinh(g, &w.centre, &w.scale, prec, true),
            None => QuadOutcome::Inconclusive,
        },
    }
}

/// Probe the integrand on a geometric ladder and report where its mass is, in
/// the integration variable (`x`, or `s = log x` when `log_scale`).
///
/// Returns `None` when the integrand is zero or unevaluable everywhere probed:
/// there is then nothing to centre on, and guessing would produce a rule that
/// integrates the wrong part of the line — which is *worse* than refusing,
/// because it converges cleanly to a small wrong number.
///
/// The width is found by growing from below rather than shrinking from above:
/// starting tiny and stopping at the first radius where the integrand has
/// fallen twelve orders below its peak gives the *smallest* such radius, so a
/// narrow feature is resolved instead of being averaged into a wide window.
pub(crate) fn locate(
    g: &mut impl FnMut(&Float) -> Sample,
    prec: u32,
    log_scale: bool,
) -> Option<Window> {
    match locate_where(g, prec, log_scale) {
        Located::At(w) => Some(w),
        _ => None,
    }
}

/// [`locate`], keeping apart "nowhere" and "nothing there to find".
pub(crate) fn locate_where(
    g: &mut impl FnMut(&Float) -> Sample,
    prec: u32,
    log_scale: bool,
) -> Located {
    let mut probe = |s: f64| -> Option<Float> {
        let arg = if log_scale {
            Float::with_val(prec, fl(prec, s).exp())
        } else {
            fl(prec, s)
        };
        match g(&arg) {
            Sample::Value(v) if v.is_finite() => {
                let mut a = v;
                a.abs_mut();
                if log_scale {
                    // the e^s Jacobian
                    a *= &arg;
                }
                Some(a)
            }
            _ => None,
        }
    };

    // Half-octave ladder: 0, ±2^{k/2} from 2⁻¹² to 2²⁴. In log-scale the
    // variable is `s`, so this covers `x` from `e^{-4096}` to `e^{16777216}` —
    // every scale a density can plausibly live at.
    let mut grid: Vec<f64> = vec![0.0];
    for k in -24i32..=48 {
        let v = (2f64).powf(f64::from(k) / 2.0);
        grid.push(v);
        grid.push(-v);
    }

    let mut best: Option<(f64, Float)> = None;
    let mut evaluated = 0usize;
    for &s in &grid {
        if let Some(v) = probe(s) {
            evaluated += 1;
            let better = match &best {
                None => v > 0,
                Some((_, bv)) => &v > bv,
            };
            if better {
                best = Some((s, v));
            }
        }
    }
    let Some((centre, peak)) = best else {
        // Every probe returned an exact zero (rather than failing): the
        // integrand vanishes on this region.
        return if evaluated >= grid.len() / 2 {
            Located::Vanishes
        } else {
            Located::NotFound
        };
    };
    if peak == 0 {
        return Located::Vanishes;
    }

    // Grow each side independently. A symmetric radius is set by whichever
    // tail is longer, and an integrand with one long tail and one short one —
    // `e^{s}e^{-e^{s}}`, which is what `∫₀^∞ e^{-x}dx` becomes on a log scale —
    // then gets a window several times wider than its own feature, and the
    // rule runs out of resolution before it runs out of levels.
    let floor = peak * fl(prec, DECAY_FLOOR);
    let start = centre.abs().max(1.0) * 1e-9;
    let mut edges = [0.0f64; 2];
    for (i, sign) in [-1.0f64, 1.0].iter().enumerate() {
        let mut r = start;
        let mut settled = false;
        for _ in 0..140 {
            let v = probe(centre + sign * r).unwrap_or_else(|| fl(prec, 0.0));
            if v < floor {
                settled = true;
                break;
            }
            r *= 2.0;
        }
        if !settled {
            return Located::NotFound;
        }
        edges[i] = centre + sign * r;
    }
    Located::At(Window {
        centre: fl(prec, (edges[0] + edges[1]) / 2.0),
        scale: fl(prec, (edges[1] - edges[0]) / 2.0),
    })
}

/// tanh–sinh on `[lo, hi]`.
fn tanh_sinh(
    g: &mut impl FnMut(&Float) -> Sample,
    lo: &Float,
    hi: &Float,
    prec: u32,
) -> QuadOutcome {
    let half_width = mul(&Float::with_val(prec, hi - lo), &fl(prec, 0.5), prec);

    let pi_half = mul(&pi_f(prec), &fl(prec, 0.5), prec);

    let (lo, hi) = (lo.clone(), hi.clone());
    let node = move |t: &Float| -> (Float, Float) {
        let u = mul(&pi_half, &Float::with_val(prec, t.clone().sinh()), prec);
        // `lo + w(1 + tanh u)` and `hi - w(1 - tanh u)` rather than
        // `c + w·tanh u`. At the outermost node `tanh u` is `1 - 6·10⁻⁵¹`, and
        // `c + w·tanh u` computes the distance to the endpoint as the
        // difference of two numbers of size `w` — every significant digit of
        // it is lost, and an integrand that reads `1 - x` there (a `Beta`
        // density with a parameter below one) gets `0^{-1/2}`. The forms below
        // build that distance as `2w/(1 + e^{∓2u})`, with no subtraction in
        // sight.
        let two_u = mul(&u, &fl(prec, 2.0), prec);
        let x = if u <= 0 {
            let d = Float::with_val(
                prec,
                mul(&half_width, &fl(prec, 2.0), prec)
                    / Float::with_val(prec, 1.0 + Float::with_val(prec, (-two_u.clone()).exp())),
            );
            Float::with_val(prec, &lo + &d)
        } else {
            let d = Float::with_val(
                prec,
                mul(&half_width, &fl(prec, 2.0), prec)
                    / Float::with_val(prec, 1.0 + Float::with_val(prec, two_u.clone().exp())),
            );
            Float::with_val(prec, &hi - &d)
        };
        let ch = Float::with_val(prec, u.clone().cosh());
        let num = mul(
            &mul(&half_width, &pi_half, prec),
            &Float::with_val(prec, t.clone().cosh()),
            prec,
        );
        let w = Float::with_val(prec, &num / mul(&ch, &ch, prec));
        (x, w)
    };
    de_levels(g, &node, prec)
}

fn mul(a: &Float, b: &Float, prec: u32) -> Float {
    Float::with_val(prec, a * b)
}

fn pi_f(prec: u32) -> Float {
    Float::with_val(prec, rug::float::Constant::Pi)
}

/// sinh–sinh on `ℝ`, shifted to `centre` and scaled by `scale`; when
/// `log_scale` the variable is `s = log x` and the `e^s` Jacobian is applied.
fn sinh_sinh(
    g: &mut impl FnMut(&Float) -> Sample,
    centre: &Float,
    scale: &Float,
    prec: u32,
    log_scale: bool,
) -> QuadOutcome {
    let pi_half = mul(&pi_f(prec), &fl(prec, 0.5), prec);
    let node = |t: &Float| -> (Float, Float) {
        let u = mul(&pi_half, &Float::with_val(prec, t.clone().sinh()), prec);
        let s = Float::with_val(
            prec,
            centre + mul(scale, &Float::with_val(prec, u.clone().sinh()), prec),
        );
        let w = mul(
            &mul(scale, &pi_half, prec),
            &mul(
                &Float::with_val(prec, t.clone().cosh()),
                &Float::with_val(prec, u.clone().cosh()),
                prec,
            ),
            prec,
        );
        if log_scale {
            let x = Float::with_val(prec, s.clone().exp());
            let w = mul(&w, &x, prec);
            (x, w)
        } else {
            (s, w)
        }
    };
    de_levels(g, &node, prec)
}

/// Run the rule at successively halved steps until two levels agree.
///
/// The node sets nest — level `j`'s abscissae at step `h/2ʲ` contain level
/// `j-1`'s — so each level only evaluates the *new* (odd-index) nodes and adds
/// them to a running sum. Total work is therefore the finest level's node
/// count rather than the sum over levels, which is what makes it affordable to
/// keep halving until the rule has genuinely settled instead of stopping early
/// and calling the last iterate an answer.
/// Binary-exponent bound on an abscissa, `|x| ∈ (2⁻⁴⁰⁹⁶, 2⁴⁰⁹⁶)`.
///
/// Not a tolerance — a **termination** guard. A sinh–sinh rule on a
/// logarithmic region places its outer abscissae at `e^{10⁴³}`, and asking
/// MPFR for `sin` of a number that size means reducing the argument modulo
/// `2π` to `10⁴³` bits: it does not return, and it takes the machine's memory
/// with it on the way. Characteristic-function integrands (`p(x)cos(tx)`) are
/// the ones that hit this, which is why it did not show up until `φ` arrived.
fn within_magnitude(x: &Float) -> bool {
    match x.get_exp() {
        None => true, // zero, or a special value already excluded above
        Some(e) => e.abs() <= 4096,
    }
}

fn de_levels(
    g: &mut impl FnMut(&Float) -> Sample,
    node: &impl Fn(&Float) -> (Float, Float),
    prec: u32,
) -> QuadOutcome {
    let mut acc = fl(prec, 0.0);
    let mut max_term = fl(prec, 0.0);
    let mut edge_term = fl(prec, 0.0);
    let mut outermost = 0.0f64;
    let mut any_value = false;
    let mut prev: Option<Float> = None;

    for level in 0..LEVELS {
        let h = 0.5f64 / f64::from(1u32 << level);
        let n = (T_MAX / h) as i64;
        let step = if level == 0 { 1 } else { 2 };
        let start = if level == 0 { -n } else { -n + (n + 1) % 2 };
        let mut k = start;
        while k <= n {
            let t = fl(prec, k as f64 * h);
            k += step;
            let (x, w) = node(&t);
            if !x.is_finite() || !w.is_finite() || !within_magnitude(&x) {
                // Either the node ran past MPFR's exponent range, or it ran
                // past the range any density has mass in. Both are skipped, and
                // the edge test below is what makes that safe: if the integrand
                // were still significant out there, the outermost node the rule
                // *did* evaluate would not be negligible and the whole
                // quadrature is reported divergent rather than truncated.
                continue;
            }
            match g(&x) {
                Sample::Value(v) => {
                    any_value = true;
                    let term = Float::with_val(prec, &w * &v);
                    if !term.is_finite() {
                        return QuadOutcome::Divergent;
                    }
                    let mut mag = term.clone();
                    mag.abs_mut();
                    if mag > max_term {
                        max_term = mag.clone();
                    }
                    // Track the outermost node the rule actually managed to
                    // evaluate, not the outermost node in principle: on a
                    // logarithmic region the extreme abscissae overflow and are
                    // skipped, and an edge test anchored to them would never
                    // fire at all.
                    let abs_t = t.to_f64().abs();
                    if abs_t > outermost {
                        outermost = abs_t;
                        edge_term = mag.clone();
                    } else if abs_t >= outermost && mag > edge_term {
                        edge_term = mag.clone();
                    }
                    acc += term;
                }
                Sample::Unevaluable => {}
            }
        }
        if !any_value {
            return QuadOutcome::Inconclusive;
        }
        let value = Float::with_val(prec, &acc * fl(prec, h));
        // The rule's whole premise is that the transformed integrand decays at
        // the ends of the `t` range. If it has not, the truncation at `T_MAX`
        // is throwing away real mass and the "value" is a fabrication.
        if max_term > 0 && edge_term > Float::with_val(prec, &max_term * fl(prec, 1e-20)) {
            return QuadOutcome::Divergent;
        }
        if let Some(p) = &prev {
            let diff = Float::with_val(prec, &value - p).abs();
            let denom = {
                let mut a = value.clone();
                a.abs_mut();
                if a < 1 {
                    a = fl(prec, 1.0);
                }
                a
            };
            if diff <= Float::with_val(prec, &denom * fl(prec, CONVERGED)) {
                return QuadOutcome::Value {
                    value,
                    est_err: diff,
                };
            }
        }
        prev = Some(value);
    }
    QuadOutcome::Inconclusive
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quadrature is the instrument every other claim in this module is
    /// measured against, so it is itself measured against integrals whose
    /// values are known exactly.
    fn run(f: impl Fn(&Float) -> Float, region: Region) -> Option<Float> {
        let mut g = |v: &Float| Sample::Value(f(v));
        match quadrature(&mut g, region, VERIFY_PREC) {
            QuadOutcome::Value { value, .. } => Some(value),
            QuadOutcome::Divergent => {
                println!("DIVERGENT");
                None
            }
            QuadOutcome::Inconclusive => {
                println!("INCONCLUSIVE");
                None
            }
        }
    }

    fn agrees(v: &Float, expected: &Float) -> bool {
        let d = Float::with_val(VERIFY_PREC, v - expected).abs();
        d < Float::with_val(VERIFY_PREC, 1e-30)
    }

    fn gauss(x: &Float) -> Float {
        let e = mul(&mul(x, x, VERIFY_PREC), &fl(VERIFY_PREC, -0.5), VERIFY_PREC);
        Float::with_val(VERIFY_PREC, e.exp())
    }

    #[test]
    fn gaussian_over_the_line_is_root_two_pi() {
        let v = run(gauss, Region::Real).expect("should converge");
        let expected = Float::with_val(
            VERIFY_PREC,
            mul(&pi_f(VERIFY_PREC), &fl(VERIFY_PREC, 2.0), VERIFY_PREC).sqrt(),
        );
        assert!(agrees(&v, &expected), "{} vs {}", v, expected);
    }

    #[test]
    fn a_shifted_gaussian_is_found_and_integrated() {
        // The rule has to *locate* this: centred at 40, nothing at the origin.
        // A double-exponential rule applied blind returns essentially zero,
        // and would then "confirm" any claim at all.
        let v = run(
            |x| {
                let d = Float::with_val(VERIFY_PREC, x - fl(VERIFY_PREC, 40.0));
                let e = mul(
                    &mul(&d, &d, VERIFY_PREC),
                    &fl(VERIFY_PREC, -0.125),
                    VERIFY_PREC,
                );
                Float::with_val(VERIFY_PREC, e.exp())
            },
            Region::Real,
        )
        .expect("should converge");
        let expected = Float::with_val(
            VERIFY_PREC,
            mul(&pi_f(VERIFY_PREC), &fl(VERIFY_PREC, 8.0), VERIFY_PREC).sqrt(),
        );
        assert!(agrees(&v, &expected), "{} vs {}", v, expected);
    }

    #[test]
    fn a_divergent_integrand_is_reported_divergent_not_valued() {
        let mut g = |v: &Float| {
            let e = mul(&mul(v, v, VERIFY_PREC), &fl(VERIFY_PREC, 0.5), VERIFY_PREC);
            Sample::Value(Float::with_val(VERIFY_PREC, e.exp()))
        };
        assert!(matches!(
            quadrature(&mut g, Region::Real, VERIFY_PREC),
            QuadOutcome::Divergent
        ));
    }
}
