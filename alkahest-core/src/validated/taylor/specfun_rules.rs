//! Taylor-model rules for the orthogonal-polynomial and special-function
//! primitives of `crate::orthopoly` and `crate::specfun` (F24).
//!
//! Every rule here is an instance of one scheme, `TaylorModel::compose_enclosed`:
//!
//! 1. split the argument model `X = m₀ + Δ` at the exact midpoint `m₀` of its
//!    constant term;
//! 2. take the Taylor coefficients `aₖ = f⁽ᵏ⁾(m₀)/k!` for `k ≤ p` from a
//!    rigorous ball evaluation **at the point** `m₀`;
//! 3. bound the Lagrange remainder by a rigorous ball enclosure of
//!    `f⁽ᵖ⁺¹⁾(ξ)/(p+1)!` over the **whole** argument enclosure
//!    `m₀ + range(Δ)`, times `|Δ|^{p+1}`.
//!
//! Step 3 is where soundness lives, and it never uses monotonicity or the
//! shape of the graph: a ball kernel (Arb) returns an enclosure of the
//! `(p+1)`-th derivative over a ball, and that is by definition an upper bound
//! of `|f⁽ᵖ⁺¹⁾(ξ)|` for every `ξ` in it.  The derivatives themselves come
//! from exact identities, each proved in the item that uses it:
//!
//! | family | `f⁽ⁿ⁾` |
//! |---|---|
//! | `Y_ν`, `I_ν`, `K_ν` (integer `ν`) | `2⁻ⁿ Σ_j (±1)ʲ C(n,j) C_{ν−n+2j}` (DLMF 10.6.1, 10.29.1) |
//! | `Ai`, `Bi`, `Ai′`, `Bi′` | `Pₙ(x) F + Qₙ(x) F′` from `F″ = xF` |
//! | `ζ(s)` | `acb_dirichlet_zeta_jet` — the Taylor coefficients, directly |
//! | `ψ⁽ᵐ⁾(x)` | `ψ⁽ᵐ⁺ⁿ⁾(x)` |
//! | `Li_s(z)` | `z⁻ⁿ Σ_j s(n,j) Li_{s−j}(z)` (`s(n,j)` signed Stirling numbers of the first kind) |
//!
//! The orthogonal polynomials need no derivative at all: for an integer
//! literal degree they are *polynomials*, expanded exactly by
//! `crate::orthopoly::explicit` and evaluated with the model's own
//! (exact-up-to-rounding) arithmetic.
//!
//! Orders, degrees and the `ψ` order must be **literals**: a symbolic order is
//! refused with `E-VALIDATED-005`, a non-integer literal Bessel order with
//! `E-VALIDATED-001`.

use super::*;
use crate::orthopoly::{int_literal, rational_literal, OrthoFamily};
use crate::specfun::numeric::{eval_ball, zeta_jet_ball, Kernel};

/// Does the context have a rule for `name` at this arity?
pub(super) fn is_specfun_call(name: &str, arity: usize) -> bool {
    if let Some(f) = OrthoFamily::from_name(name) {
        return arity == f.arity();
    }
    match name {
        "bessel_j" | "bessel_y" | "bessel_i" | "bessel_k" | "polylog" | "polygamma" => arity == 2,
        "airy_ai" | "airy_bi" | "airy_ai_prime" | "airy_bi_prime" | "zeta" => arity == 1,
        _ => false,
    }
}

fn invalid(what: String) -> ValidatedError {
    ValidatedError::InvalidInput { what }
}

fn domain(what: &str) -> ValidatedError {
    ValidatedError::DomainViolation {
        what: what.to_string(),
    }
}

/// Re-round a ball computed at higher precision down to `prec` bits, folding
/// the midpoint rounding into the radius.
fn round_ball(b: &ArbBall, prec: u32) -> ArbBall {
    let mid = Float::with_val(prec, &b.mid);
    let err = Float::with_val(b.prec.max(prec) + 16, &b.mid - &mid).abs();
    super::super::inflate(&ArbBall {
        mid,
        rad: Float::with_val(b.prec.max(prec) + 16, &b.rad + &err),
        prec,
    })
}

/// Evaluate `f` on `b`, and where Arb gives up on a wide ball (a non-finite
/// result) bisect and take the hull of the pieces: the union of enclosures over
/// a cover of `b` encloses `f` over `b`.  A point ball is never split.
fn split_hull(b: &ArbBall, depth: u32, f: &dyn Fn(&ArbBall) -> Option<ArbBall>) -> Option<ArbBall> {
    if let Some(v) = f(b) {
        if is_finite(&v) {
            return Some(v);
        }
    }
    if depth == 0 || b.rad == 0 {
        return None;
    }
    let (lo, hi) = (b.lo(), b.hi());
    let mid = Float::with_val(b.prec + 8, Float::with_val(b.prec + 8, &lo + &hi) / 2u32);
    let left = split_hull(&from_bounds(&lo, &mid, b.prec), depth - 1, f)?;
    let right = split_hull(&from_bounds(&mid, &hi, b.prec), depth - 1, f)?;
    Some(hull(&left, &right))
}

/// Bisection depth for [`split_hull`] (at most `2^depth` Arb calls).
const SPLIT_DEPTH: u32 = 7;

/// An exact point as a ball at `prec` bits (radius zero).
fn point(v: &Float, prec: u32) -> ArbBall {
    ArbBall {
        mid: Float::with_val(prec, v),
        rad: Float::new(prec),
        prec,
    }
}

fn int_ball(n: i64, prec: u32) -> ArbBall {
    ArbBall::from_integer(&Integer::from(n), prec)
}

fn factorial_ball(k: usize, prec: u32) -> ArbBall {
    ArbBall::from_integer(&Integer::factorial(k as u32).complete(), prec)
}

/// Signed Stirling numbers of the first kind `s(n, 0..=n)`.
pub(super) fn stirling_first(n: usize) -> Vec<Integer> {
    let mut row = vec![Integer::from(1)];
    for m in 0..n {
        let mut next = vec![Integer::new(); m + 2];
        for (j, c) in row.iter().enumerate() {
            // s(m+1, j+1) += s(m, j);  s(m+1, j) -= m·s(m, j)
            next[j + 1] += c;
            next[j] -= Integer::from(c * Integer::from(m));
        }
        row = next;
    }
    row
}

/// Polynomials `(Pₙ, Qₙ)` with `F⁽ⁿ⁾ = Pₙ(x)·F + Qₙ(x)·F′` for any solution
/// `F″ = xF` of Airy's equation.  Coefficients in increasing degree.
///
/// *Induction.*  `n = 0`: `(1, 0)`.  Differentiating,
/// `(P F + Q F′)′ = (P′ + xQ)·F + (P + Q′)·F′`, using `F″ = xF`.
pub(super) fn airy_polys(n: usize) -> (Vec<Integer>, Vec<Integer>) {
    let mut p = vec![Integer::from(1)];
    let mut q: Vec<Integer> = vec![];
    let deriv = |v: &[Integer]| -> Vec<Integer> {
        v.iter()
            .enumerate()
            .skip(1)
            .map(|(k, c)| Integer::from(c * Integer::from(k)))
            .collect()
    };
    let add = |a: &[Integer], b: &[Integer]| -> Vec<Integer> {
        let len = a.len().max(b.len());
        (0..len)
            .map(|i| {
                Integer::from(a.get(i).cloned().unwrap_or_default())
                    + b.get(i).cloned().unwrap_or_default()
            })
            .collect()
    };
    let times_x = |v: &[Integer]| -> Vec<Integer> {
        if v.is_empty() {
            return vec![];
        }
        let mut out = vec![Integer::new()];
        out.extend(v.iter().cloned());
        out
    };
    for _ in 0..n {
        let np = add(&deriv(&p), &times_x(&q));
        let nq = add(&p, &deriv(&q));
        p = np;
        q = nq;
    }
    (p, q)
}

fn horner(coeffs: &[Integer], x: &ArbBall) -> ArbBall {
    let prec = x.prec;
    let mut acc = ArbBall::from_f64(0.0, prec);
    for c in coeffs.iter().rev() {
        acc = acc * x.clone() + ArbBall::from_integer(c, prec);
    }
    acc
}

/// `C_m(x)` for an integer order `m` of one cylinder kernel, with the
/// negative-order reflections applied here so Arb only sees `m ≥ 0`:
/// `J_{−n} = (−1)ⁿ Jₙ`, `Y_{−n} = (−1)ⁿ Yₙ`, `I_{−n} = Iₙ`, `K_{−n} = Kₙ`.
fn cyl(kernel: Kernel, m: i64, x: &ArbBall) -> Result<ArbBall> {
    let nu = int_ball(m.abs(), x.prec);
    let v = split_hull(x, SPLIT_DEPTH, &|xs| {
        eval_ball(kernel, &[nu.clone(), xs.clone()])
    })
    .ok_or_else(|| ValidatedError::NotFinite {
        what: format!("{} order {m}", kernel.name()),
    })?;
    let odd = m < 0 && m % 2 != 0;
    Ok(
        if odd && matches!(kernel, Kernel::BesselJ | Kernel::BesselY) {
            -v
        } else {
            v
        },
    )
}

impl TaylorModel {
    /// The shared scheme (module docs).  `coeffs(c, p)` returns `a₀..a_p` at
    /// the exact point ball `c`; `remainder(arg, p)` returns an enclosure of
    /// `f⁽ᵖ⁺¹⁾(ξ)/(p+1)!` for all `ξ` in the ball `arg`.  `check(arg)` is the
    /// domain guard on the argument enclosure.
    fn compose_enclosed(
        &self,
        what: &str,
        check: impl FnOnce(&ArbBall) -> Result<()>,
        coeffs: impl FnOnce(&ArbBall, usize) -> Result<Vec<ArbBall>>,
        remainder: impl FnOnce(&ArbBall, usize) -> Result<ArbBall>,
    ) -> Result<Self> {
        self.check_finite(what)?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: format!("{what} argument"),
            });
        }
        check(&arg)?;
        let p = self.order;
        // Extra bits for the coefficients: the derivative identities are
        // alternating sums, so a few bits are lost to cancellation.
        let work = self.prec + 64 + 4 * p as u32;
        let c = point(&m0, work);
        let a: Vec<ArbBall> = coeffs(&c, p)?
            .iter()
            .map(|b| round_ball(b, self.prec))
            .collect();
        if a.len() != p + 1 || a.iter().any(|b| !is_finite(b)) {
            return Err(ValidatedError::NotFinite {
                what: format!("{what} Taylor coefficients"),
            });
        }
        let rem = remainder(&arg, p)?;
        if !is_finite(&rem) {
            return Err(ValidatedError::NotFinite {
                what: format!("{what} remainder"),
            });
        }
        let dp = from_float(&delta.delta_pow(&d), self.prec);
        let radius = ub(&(from_float(&mag(&rem), self.prec) * dp));
        let out = delta.compose(&a, &radius);
        out.check_finite(&format!("{what} result"))?;
        Ok(out)
    }

    /// A function whose `n`-th Taylor coefficient `f⁽ⁿ⁾(x)/n!` is `d(n, ball)`
    /// for every ball (point or interval).
    fn compose_by_derivatives(
        &self,
        what: &str,
        check: impl FnOnce(&ArbBall) -> Result<()>,
        d: impl Fn(usize, &ArbBall) -> Result<ArbBall>,
    ) -> Result<Self> {
        self.compose_enclosed(
            what,
            check,
            |c, p| (0..=p).map(|k| d(k, c)).collect(),
            |arg, p| d(p + 1, arg),
        )
    }

    /// `C_ν(self)` for `C ∈ {Y, I, K}` and an integer literal `ν`.
    ///
    /// **Derivatives.**  `2·J′_μ = J_{μ−1} − J_{μ+1}` (and the same for `Y`)
    /// iterates by Pascal's rule to
    /// `C_ν⁽ⁿ⁾ = 2⁻ⁿ Σ_{j=0}^{n} (−1)ʲ C(n,j) C_{ν−n+2j}` — the induction is
    /// that of [`TaylorModel::bessel_j`].  For `I` (`2·I′_μ = I_{μ−1} + I_{μ+1}`)
    /// every sign is `+`; for `K` (`2·K′_μ = −K_{μ−1} − K_{μ+1}`) the extra
    /// `(−1)ⁿ` is applied.  Each `C_m` is an Arb ball kernel call, so the
    /// remainder is an honest enclosure over the whole argument range.
    ///
    /// **Domain.**  `Y` and `K` are real only for `x > 0` and singular at 0:
    /// the argument enclosure must lie strictly in `(0, ∞)`.  `I` is entire.
    fn bessel_cylinder(&self, kernel: Kernel, nu: i64) -> Result<Self> {
        let needs_positive = matches!(kernel, Kernel::BesselY | Kernel::BesselK);
        self.compose_by_derivatives(
            kernel.name(),
            |arg| {
                if needs_positive && !strictly_positive(&lb(arg)) {
                    return Err(domain(&format!(
                        "{} of an argument whose enclosure reaches 0 or below",
                        kernel.name()
                    )));
                }
                Ok(())
            },
            |k, x| {
                let prec = x.prec;
                let mut acc = ArbBall::from_f64(0.0, prec);
                for j in 0..=k {
                    let binom = Integer::from(k).binomial(j as u32);
                    let term = ArbBall::from_integer(&binom, prec)
                        * cyl(kernel, nu - k as i64 + 2 * j as i64, x)?;
                    let negative = match kernel {
                        Kernel::BesselI | Kernel::BesselK => false,
                        _ => j % 2 == 1,
                    };
                    acc = if negative { acc - term } else { acc + term };
                }
                if kernel == Kernel::BesselK && k % 2 == 1 {
                    acc = -acc;
                }
                // 1 / (2ᵏ · k!)
                let scale = ArbBall::from_integer(
                    &((Integer::from(1) << (k as u32)) * Integer::factorial(k as u32).complete()),
                    prec,
                );
                Self::div_ball(&acc, &scale)
            },
        )
    }

    /// `F(self)` for `F ∈ {Ai, Bi, Ai′, Bi′}`.
    ///
    /// **Derivatives.**  Every solution of `F″ = xF` has
    /// `F⁽ⁿ⁾ = Pₙ(x)·F + Qₙ(x)·F′` with the integer polynomials of
    /// `airy_polys`; `Ai′` and `Bi′` are the same family shifted by one
    /// derivative.  Entire, so no domain guard.
    fn airy_function(&self, name: &str) -> Result<Self> {
        let (value_kernel, deriv_kernel, shift) = match name {
            "airy_ai" => (Kernel::AiryAi, Kernel::AiryAiPrime, 0usize),
            "airy_ai_prime" => (Kernel::AiryAi, Kernel::AiryAiPrime, 1),
            "airy_bi" => (Kernel::AiryBi, Kernel::AiryBiPrime, 0),
            _ => (Kernel::AiryBi, Kernel::AiryBiPrime, 1),
        };
        self.compose_by_derivatives(
            name,
            |_| Ok(()),
            |k, x| {
                let prec = x.prec;
                let (p, q) = airy_polys(k + shift);
                let f = split_hull(x, SPLIT_DEPTH, &|xs| {
                    eval_ball(value_kernel, std::slice::from_ref(xs))
                })
                .ok_or_else(|| ValidatedError::NotFinite {
                    what: format!("{name} value"),
                })?;
                let fp = split_hull(x, SPLIT_DEPTH, &|xs| {
                    eval_ball(deriv_kernel, std::slice::from_ref(xs))
                })
                .ok_or_else(|| ValidatedError::NotFinite {
                    what: format!("{name} derivative"),
                })?;
                let val = horner(&p, x) * f + horner(&q, x) * fp;
                Self::div_ball(&val, &factorial_ball(k, prec))
            },
        )
    }

    /// `ζ(self)`.  The pole `s = 1` must lie outside the argument enclosure.
    ///
    /// Taylor coefficients are the jet `ζ⁽ᵏ⁾(s)/k!` itself
    /// (`acb_dirichlet_zeta_jet`); the remainder is the `(p+1)`-th jet entry
    /// evaluated on the whole argument ball.
    fn zeta_function(&self) -> Result<Self> {
        self.compose_enclosed(
            "zeta",
            |arg| {
                if arg.lo() <= 1 && arg.hi() >= 1 {
                    return Err(domain(
                        "zeta of an argument whose enclosure contains the pole s = 1",
                    ));
                }
                Ok(())
            },
            |c, p| {
                zeta_jet_ball(c, p + 1).ok_or_else(|| ValidatedError::NotFinite {
                    what: "zeta jet".into(),
                })
            },
            |arg, p| {
                split_hull(arg, SPLIT_DEPTH, &|a| {
                    zeta_jet_ball(a, p + 2).map(|mut jet| jet.swap_remove(p + 1))
                })
                .ok_or_else(|| ValidatedError::NotFinite {
                    what: "zeta jet over the argument enclosure".into(),
                })
            },
        )
    }

    /// `ψ⁽ᵐ⁾(self)` for an integer literal order `m ≥ 0`: `f⁽ⁿ⁾ = ψ⁽ᵐ⁺ⁿ⁾`.
    /// The argument enclosure must lie strictly in `(0, ∞)` (poles at every
    /// non-positive integer).
    fn polygamma_function(&self, m: u32) -> Result<Self> {
        self.compose_by_derivatives(
            "polygamma",
            |arg| {
                if !strictly_positive(&lb(arg)) {
                    return Err(domain(
                        "polygamma of an argument whose enclosure reaches 0 or below (poles sit at every non-positive integer)",
                    ));
                }
                Ok(())
            },
            |k, x| {
                let prec = x.prec;
                let order = int_ball(i64::from(m) + k as i64, prec);
                let v = split_hull(x, SPLIT_DEPTH, &|xs| {
                    crate::specfun::gamma::polygamma_ball(&order, xs)
                })
                .ok_or_else(|| ValidatedError::NotFinite {
                    what: "polygamma value".into(),
                })?;
                Self::div_ball(&v, &factorial_ball(k, prec))
            },
        )
    }

    /// `Li_s(self)` for a literal order `s` and an argument enclosure inside
    /// the open unit interval `(-1, 1)`.
    ///
    /// **Coefficients.**  At a point `m₀ ≠ 0`,
    /// `f⁽ⁿ⁾(m₀)/n! = m₀⁻ⁿ/n! · Σ_{j=1}^{n} s(n,j)·Li_{s−j}(m₀)` from
    /// `(z d/dz) Li_s = Li_{s−1}` and `zⁿ f⁽ⁿ⁾ = θ(θ−1)⋯(θ−n+1) f` with
    /// `θ = z d/dz` (`s(n,j)` signed Stirling numbers of the first kind); at
    /// `m₀ = 0` the series itself, `aₖ = k⁻ˢ`.
    ///
    /// **Remainder — a Cauchy estimate, not a ball derivative.**  Arb's real
    /// `polylog` is a poor performer on wide balls (hundreds of times too
    /// wide), so the Lagrange term uses the series' absolute convergence
    /// instead.  Let `ζ = max |z|` over the argument enclosure, `r = (1−ζ)/2`
    /// and `ρ = (1+ζ)/2 < 1`.  Every circle `|w − ξ| = r` around a point `ξ`
    /// of the enclosure lies in `|w| ≤ ζ + r = ρ`, where
    /// `|Li_s(w)| ≤ Σ_k ρᵏ k⁻ˢ = Li_s(ρ)` (all coefficients `k⁻ˢ` are
    /// positive), so Cauchy's inequality gives
    /// `|f⁽ᵖ⁺¹⁾(ξ)|/(p+1)! ≤ Li_s(ρ)/r^{p+1}` for every real `s`.
    fn polylog_function(&self, s: &Rational) -> Result<Self> {
        self.compose_enclosed(
            "polylog",
            |arg| {
                if arg.lo() <= -1 || arg.hi() >= 1 {
                    return Err(domain(
                        "polylog of an argument whose enclosure is not inside (-1, 1)",
                    ));
                }
                Ok(())
            },
            |z, p| {
                let prec = z.prec;
                let sb = ArbBall::from_rational(s, prec);
                let mut out = Vec::with_capacity(p + 1);
                if z.mid == 0 {
                    out.push(ArbBall::from_f64(0.0, prec));
                    for k in 1..=p {
                        out.push(int_ball(k as i64, prec).pow_f(&(-sb.clone())));
                    }
                    return Ok(out);
                }
                let value = |order: ArbBall| {
                    eval_ball(Kernel::Polylog, &[order, z.clone()]).ok_or_else(|| {
                        ValidatedError::NotFinite {
                            what: "polylog value".into(),
                        }
                    })
                };
                // Li_{s-j}(m0) for j = 0..=p, once.
                // The order `s - j` is formed exactly in ℚ, not by ball
                // subtraction: Arb's polylog is indeterminate for an *inexact*
                // order next to a positive integer.
                let li: Vec<ArbBall> = (0..=p)
                    .map(|j| {
                        value(ArbBall::from_rational(
                            &Rational::from(s - Rational::from(j as u32)),
                            prec,
                        ))
                    })
                    .collect::<Result<_>>()?;
                out.push(li[0].clone());
                for k in 1..=p {
                    let st = stirling_first(k);
                    let mut acc = ArbBall::from_f64(0.0, prec);
                    for (j, c) in st.iter().enumerate().skip(1) {
                        acc = acc + ArbBall::from_integer(c, prec) * li[j].clone();
                    }
                    let denom = z.powi(k as i64) * factorial_ball(k, prec);
                    out.push(Self::div_ball(&acc, &denom)?);
                }
                Ok(out)
            },
            |arg, p| {
                let prec = arg.prec;
                let sb = ArbBall::from_rational(s, prec);
                let zeta = from_float(&mag(arg), prec);
                let one = ArbBall::from_f64(1.0, prec);
                let r = (one.clone() - zeta.clone()) * ArbBall::from_f64(0.5, prec);
                let rho = (one + zeta) * ArbBall::from_f64(0.5, prec);
                if !strictly_positive(&lb(&r)) {
                    return Err(domain(
                        "polylog: no disc of analyticity around the argument",
                    ));
                }
                let li = eval_ball(Kernel::Polylog, &[sb, rho]).ok_or_else(|| {
                    ValidatedError::NotFinite {
                        what: "polylog majorant".into(),
                    }
                })?;
                Self::div_ball(&li, &r.powi(p as i64 + 1))
            },
        )
    }
}

impl TaylorContext<'_> {
    /// Evaluate a call recognised by [`is_specfun_call`].
    pub(super) fn eval_specfun(&mut self, name: &str, args: &[ExprId]) -> Result<TaylorModel> {
        if let Some(family) = OrthoFamily::from_name(name) {
            let (x, head) = args.split_last().expect("arity checked");
            let (n, params) = head.split_first().expect("arity checked");
            return match crate::orthopoly::explicit(family, *n, params, *x, self.pool) {
                Ok(Some(e)) => self.eval(e),
                Ok(None) => Err(invalid(format!(
                    "{name}: a Taylor model needs a non-negative integer literal degree \
                     (and, for assoc_legendre_p, an integer literal order); a symbolic or \
                     non-integer one has no polynomial form"
                ))),
                Err(e) => Err(invalid(format!("{name}: {e}"))),
            };
        }
        match name {
            "bessel_j" | "bessel_y" | "bessel_i" | "bessel_k" => {
                let Some(q) = rational_literal(self.pool, args[0]) else {
                    return Err(invalid(format!(
                        "{name}: a Taylor model needs a literal order, not a symbolic one"
                    )));
                };
                if *q.denom() != 1 {
                    return Err(ValidatedError::Unsupported {
                        what: format!("`{name}` of non-integer order {q}"),
                    });
                }
                let nu = q
                    .numer()
                    .to_i64()
                    .filter(|v| v.abs() <= 1000)
                    .ok_or_else(|| ValidatedError::Unsupported {
                        what: format!("`{name}` of order {q} (|order| > 1000)"),
                    })?;
                let x = self.eval(args[1])?;
                match name {
                    "bessel_j" => x.bessel_j(nu as i32),
                    "bessel_y" => x.bessel_cylinder(Kernel::BesselY, nu),
                    "bessel_i" => x.bessel_cylinder(Kernel::BesselI, nu),
                    _ => x.bessel_cylinder(Kernel::BesselK, nu),
                }
            }
            "airy_ai" | "airy_bi" | "airy_ai_prime" | "airy_bi_prime" => {
                self.eval(args[0])?.airy_function(name)
            }
            "zeta" => self.eval(args[0])?.zeta_function(),
            "polygamma" => {
                let Some(m) = int_literal(self.pool, args[0]).and_then(|v| v.to_u32()) else {
                    return Err(invalid(
                        "polygamma: a Taylor model needs a non-negative integer literal order"
                            .into(),
                    ));
                };
                self.eval(args[1])?.polygamma_function(m)
            }
            "polylog" => {
                let Some(s) = rational_literal(self.pool, args[0]) else {
                    return Err(invalid(
                        "polylog: a Taylor model needs a literal order s, not a symbolic one"
                            .into(),
                    ));
                };
                self.eval(args[1])?.polylog_function(&s)
            }
            other => Err(ValidatedError::Unsupported {
                what: format!("function `{other}`"),
            }),
        }
    }
}
