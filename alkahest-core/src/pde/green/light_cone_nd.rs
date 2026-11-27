//! Klein–Gordon and telegraph kernels in two and three space dimensions.
//!
//! With `μ = m/c` (Klein–Gordon) or `λ = a/c` (telegraph, after
//! `u = e^{−at} v`, which turns it into Klein–Gordon with `m² = −a²`) and
//! `σ = √(c²t² − r²)`:
//!
//! * **2-D** — `G = H(ct − r) cos(μσ) / (2πcσ)`, resp.
//!   `G = H(ct − r) e^{−at} cosh(λσ) / (2πcσ)`.  Proved by **descent from the
//!   3-D wave kernel** `W = δ(t − ρ/c)/(4πc²ρ)`, `ρ = √(r² + z²)`: the
//!   Fourier transform in the extra coordinate, `∫ W cos(μz) dz` (resp.
//!   `∫ W cosh(λz) dz` — `W` has compact support in `z`, so the exponential
//!   weight is admissible), turns `∂ₜₜ − c²Δ₃` into `∂ₜₜ − c²Δ₂ ± c²μ²`.  The
//!   integral is a sifting integral, `δ(g(z)) = Σ δ(z − zₖ)/|g'(zₖ)|` over the
//!   roots `z = ±σ`, computed and checked exactly.
//! * **3-D** — `G = δ(t − r/c)/(4πc²r) − H(ct − r) μ J₁(μσ)/(4πcσ)`, resp.
//!   `e^{−at}[δ(t − r/c)/(4πc²r) + H(ct − r) λ I₁(λσ)/(4πcσ)]`.  Proved by the
//!   **dimension walk** `E₃ = −(2πr)^{−1} ∂ᵣ E₁` from the (verified) 1-D
//!   light-cone kernel `E₁ = H(ct − |x|) R(r, t)/(2c)`: the derivative of the
//!   Heaviside factor is the light-cone `δ` with coefficient
//!   `R(ct, t)/(2πcr)`, the derivative of `R` is the Bessel tail.  Both parts
//!   are checked by substitution, and so is the commutation
//!   `L₃[(1/r)∂ᵣf] = (1/r)∂ᵣ(L₁f)` for an arbitrary radial `f`.
//!
//! In both dimensions the smooth part is also checked to solve the
//! homogeneous equation inside the cone, and to reduce to the wave kernel when
//! the mass (damping) vanishes; in 2-D the mass `∫ G dx` is checked against
//! the `ξ = 0` transfer function (`sin(mt)/m`, resp. `e^{−at} sinh(at)/a`)
//! through an explicit antiderivative.

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::sturm_liouville::{replace_node, simp, subs1};
use super::super::verify::VerificationStatus;
use super::{
    at, check, combine, dd, div, fresh, fundamental_solution, neg, pi, refuse_refuted, sq, sqrt,
    sub, zstatus, FundamentalSolution, GreenCheck, GreenOperator,
};

fn int(n: i64, pool: &ExprPool) -> ExprId {
    pool.integer(n)
}

/// The data of one light-cone operator.
struct Cone {
    c: ExprId,
    /// `μ` (Klein–Gordon) or `λ` (telegraph).
    lam: ExprId,
    /// The parameter itself (`m` or `a`).
    param: ExprId,
    /// Telegraph: `cosh` / `I` instead of `cos` / `J`, and the factor `e^{−at}`.
    hyperbolic: bool,
    /// `e^{−at}` or `1`.
    pre: ExprId,
    /// The damping in the radial operator (`a` or `0`).
    damping: ExprId,
    /// The zeroth-order coefficient (`m²` or `0`).
    m2: ExprId,
}

/// `f_tt + 2a f_t − c²(f_rr + (n−1) f_r / r) + m² f`.
fn radial_op(
    f: ExprId,
    r: ExprId,
    t: ExprId,
    n: i64,
    cone: &Cone,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let fr = dd(f, r, pool)?;
    let lap = pool.add(vec![
        dd(fr, r, pool)?,
        pool.mul(vec![int(n - 1, pool), pool.pow(r, int(-1, pool)), fr]),
    ]);
    Ok(pool.add(vec![
        dd(dd(f, t, pool)?, t, pool)?,
        pool.mul(vec![int(2, pool), cone.damping, dd(f, t, pool)?]),
        neg(pool.mul(vec![sq(cone.c, pool), lap]), pool),
        pool.mul(vec![cone.m2, f]),
    ]))
}

/// The light-cone kernel of a Klein–Gordon or telegraph operator in 2 or 3
/// space dimensions (validated by the caller).  The kernel lives on `t > 0`,
/// so it is built with a positive time symbol and the caller's time
/// substituted back.
pub(super) fn light_cone_nd(
    op: GreenOperator,
    space: &[ExprId],
    time: ExprId,
    pool: &ExprPool,
) -> Result<FundamentalSolution, PdeError> {
    let positive = pool.with(
        time,
        |d| matches!(d, ExprData::Symbol { domain, .. } if *domain == Domain::Positive),
    );
    if positive {
        return light_cone_nd_pos(op, space, time, pool);
    }
    let mut avoid: Vec<ExprId> = space.to_vec();
    avoid.push(time);
    avoid.extend(op.parameters().iter().map(|&(_, v)| v));
    let tp = fresh("t", Domain::Positive, &avoid, pool);
    let mut fs = light_cone_nd_pos(op, space, tp, pool)?;
    let back = |e: ExprId| simp(subs1(e, tp, time, pool), pool);
    fs.time = Some(time);
    fs.kernel = back(fs.kernel);
    fs.radial = back(fs.radial);
    fs.smooth = back(fs.smooth);
    fs.support = fs.support.map(back);
    fs.side_conditions = fs.side_conditions.into_iter().map(back).collect();
    Ok(fs)
}

fn light_cone_nd_pos(
    op: GreenOperator,
    space: &[ExprId],
    time: ExprId,
    pool: &ExprPool,
) -> Result<FundamentalSolution, PdeError> {
    let n = space.len();
    let t = time;
    let cone = match op {
        GreenOperator::KleinGordon { speed, mass } => Cone {
            c: speed,
            lam: div(mass, speed, pool),
            param: mass,
            hyperbolic: false,
            pre: int(1, pool),
            damping: int(0, pool),
            m2: sq(mass, pool),
        },
        GreenOperator::Telegraph { speed, damping } => Cone {
            c: speed,
            lam: div(damping, speed, pool),
            param: damping,
            hyperbolic: true,
            pre: pool.func("exp", vec![neg(pool.mul(vec![damping, t]), pool)]),
            damping,
            m2: int(0, pool),
        },
        _ => unreachable!("light_cone_nd on a non-light-cone operator"),
    };
    let c = cone.c;
    let mut avoid: Vec<ExprId> = space.to_vec();
    avoid.push(t);
    avoid.extend(op.parameters().iter().map(|&(_, v)| v));
    let r = fresh("r", Domain::Positive, &avoid, pool);
    avoid.push(r);
    // On the support t > 0: checks that need √(c²t²) = ct run with t → T > 0.
    let tp = fresh("T", Domain::Positive, &avoid, pool);
    let on_support = |e: ExprId| simp(subs1(e, t, tp, pool), pool);
    let ct = pool.mul(vec![c, t]);
    let sigma = sqrt(sub(sq(ct, pool), sq(r, pool), pool), pool);
    let two = int(2, pool);
    let four = int(4, pool);
    let r_cart = sqrt(pool.add(space.iter().map(|&v| sq(v, pool)).collect()), pool);
    let cart = |e: ExprId| simp(subs1(e, r, r_cart, pool), pool);
    let param_zero = simp(cone.param, pool) == int(0, pool);
    let mut checks: Vec<GreenCheck> = Vec::new();
    let mut derivation: Vec<String> = Vec::new();
    let mut references: Vec<String> = Vec::new();
    let name = op.name();
    let (radial, kernel, identity, proof) = if n == 2 {
        // ------------------------------------------------------------ 2-D
        let (osc, osc_name) = if cone.hyperbolic {
            (
                pool.func("cosh", vec![pool.mul(vec![cone.lam, sigma])]),
                "cosh",
            )
        } else {
            (
                pool.func("cos", vec![pool.mul(vec![cone.lam, sigma])]),
                "cos",
            )
        };
        let radial = simp(
            div(
                pool.mul(vec![cone.pre, osc]),
                pool.mul(vec![two, pi(pool), c, sigma]),
                pool,
            ),
            pool,
        );
        // Descent from the 3-D wave kernel of the table, W = W(ρ, t) with
        // ρ = √(r² + z²): δ(g(ρ)) sifts at ρ = ct, i.e. z = ±σ, with
        // |∂_z g| = |g'(ρ)| · |z|/ρ = |g'(ct)| σ/(ct).
        let z = fresh("z", Domain::Real, &avoid, pool);
        let wave = fundamental_solution(
            GreenOperator::Wave { speed: c },
            &[space[0], space[1], z],
            Some(t),
            pool,
        )?;
        let rho = wave.radius;
        let w = wave.radial;
        let (dnode, g) = find_delta(w, pool).ok_or_else(|| {
            PdeError::GreenVerificationFailed(
                "the 3-D wave kernel has no δ factor to sift".to_string(),
            )
        })?;
        let phi0 = simp(replace_node(w, dnode, int(1, pool), pool), pool);
        let weight = |zz: ExprId| {
            if cone.hyperbolic {
                pool.func("cosh", vec![pool.mul(vec![cone.lam, zz])])
            } else {
                pool.func("cos", vec![pool.mul(vec![cone.lam, zz])])
            }
        };
        let wave_ok = wave.status == VerificationStatus::ExactlyVerified;
        checks.push(GreenCheck {
            what: "descent",
            status: if wave_ok {
                ZeroStatus::ProvedZero
            } else {
                ZeroStatus::Unknown
            },
            method: "fundamental_solution(wave, 3-D)".into(),
            detail: "the 3-D wave kernel W = δ(t − ρ/c)/(4πc²ρ) is exactly verified".into(),
        });
        if mentions_sym(g, z, pool) || mentions_sym(phi0, z, pool) {
            return Err(PdeError::GreenVerificationFailed(
                "the 3-D wave kernel is not radial".to_string(),
            ));
        }
        checks.push(check(
            "descent",
            at(g, rho, ct, pool),
            "substitution",
            "the δ argument g(ρ) = t − ρ/c vanishes at ρ = ct, i.e. at z = ±σ (its only roots \
             for r < ct)",
            pool,
        ));
        let gp = at(dd(g, rho, pool)?, rho, ct, pool);
        checks.push(check(
            "descent",
            pool.add(vec![gp, div(int(1, pool), c, pool)]),
            "substitution",
            "g'(ct) = −1/c, so |∂_z g(±σ)| = (1/c)·σ/(ct) = σ/(c²t)",
            pool,
        ));
        checks.push(check(
            "descent",
            sub(weight(neg(z, pool)), weight(z), pool),
            "substitution",
            "the weight is even in z (both roots contribute equally)",
            pool,
        ));
        let slope = div(sigma, pool.mul(vec![sq(c, pool), t]), pool);
        let sifted = pool.mul(vec![
            two,
            cone.pre,
            at(phi0, rho, ct, pool),
            weight(sigma),
            pool.pow(slope, int(-1, pool)),
        ]);
        checks.push(check(
            "descent",
            sub(sifted, radial, pool),
            "sifting: δ(g(z)) = Σ δ(z − zₖ)/|g'(zₖ)|",
            format!(
                "G = {}∫ W {osc_name}({}z) dz: the 3-D wave kernel transformed in the third \
                 coordinate",
                if cone.hyperbolic { "e^{−at} " } else { "" },
                if cone.hyperbolic { "λ" } else { "μ" }
            ),
            pool,
        ));
        // Mass: ∫ G dx = sin(mt)/m (resp. e^{−at} sinh(at)/a), by the
        // antiderivative F(r) of 2πr G.
        if !param_zero {
            let (f, want) = if cone.hyperbolic {
                (
                    neg(
                        div(
                            pool.mul(vec![
                                cone.pre,
                                pool.func("sinh", vec![pool.mul(vec![cone.lam, sigma])]),
                            ]),
                            pool.mul(vec![cone.lam, c]),
                            pool,
                        ),
                        pool,
                    ),
                    div(
                        pool.mul(vec![
                            cone.pre,
                            pool.func("sinh", vec![pool.mul(vec![cone.param, t])]),
                        ]),
                        cone.param,
                        pool,
                    ),
                )
            } else {
                (
                    neg(
                        div(
                            pool.func("sin", vec![pool.mul(vec![cone.lam, sigma])]),
                            pool.mul(vec![cone.lam, c]),
                            pool,
                        ),
                        pool,
                    ),
                    div(
                        pool.func("sin", vec![pool.mul(vec![cone.param, t])]),
                        cone.param,
                        pool,
                    ),
                )
            };
            let integrand = pool.mul(vec![two, pi(pool), r, radial]);
            checks.push(check(
                "mass",
                sub(dd(f, r, pool)?, integrand, pool),
                "substitution",
                "F(r) is an antiderivative of 2πr G(r, t) on (0, ct)",
                pool,
            ));
            let total = sub(at(f, r, ct, pool), at(f, r, int(0, pool), pool), pool);
            let mut ck = check(
                "mass",
                on_support(sub(total, want, pool)),
                "fundamental theorem of calculus (F continuous on [0, ct])",
                "∫ G(x, t) dx = F(ct) − F(0) equals the ξ = 0 transfer function",
                pool,
            );
            if ck.status != ZeroStatus::ProvedZero {
                ck.status = zstatus(
                    super::trig_to_exp(on_support(sub(total, want, pool)), pool),
                    pool,
                );
            }
            checks.push(ck);
        }
        derivation.push(format!(
            "method of descent: the Fourier transform in a third coordinate z of \
             (∂ₜₜ − c²Δ₃)W = δ(x)δ(y)δ(z)δ(t) at frequency {} gives the {name} operator in the \
             plane, so G = {}∫ δ(t − ρ/c)/(4πc²ρ) {osc_name}({}z) dz with ρ = √(r² + z²); the \
             δ sifts at z = ±σ, σ = √(c²t² − r²), with |∂_z(t − ρ/c)| = σ/(c²t)",
            if cone.hyperbolic {
                "iλ (λ = a/c, after u = e^{−at}v)"
            } else {
                "μ = m/c"
            },
            if cone.hyperbolic { "e^{−at} " } else { "" },
            if cone.hyperbolic { "λ" } else { "μ" },
        ));
        references.push("Duffy, Green's Functions with Applications §4.3–4.4".into());
        references.push("Morse & Feshbach, Methods of Theoretical Physics §7.4".into());
        references.push("Courant & Hilbert, Methods II §VI.12 (method of descent)".into());
        let smooth_cart = cart(radial);
        let kernel = pool.mul(vec![
            pool.func("heaviside", vec![sub(ct, r_cart, pool)]),
            smooth_cart,
        ]);
        let identity = if cone.hyperbolic {
            "(∂ₜₜ + 2a∂ₜ − c²Δ) G = δ(x) δ(t), G = H(ct − r) e^{−at} cosh((a/c)√(c²t² − r²)) \
             / (2πc√(c²t² − r²))"
                .to_string()
        } else {
            "(∂ₜₜ − c²Δ + m²) G = δ(x) δ(t), G = H(ct − r) cos((m/c)√(c²t² − r²)) \
             / (2πc√(c²t² − r²))"
                .to_string()
        };
        let proof = "method of descent: the 3-D wave kernel (exactly verified) transformed in \
                     the third coordinate solves the transformed equation with the transformed \
                     source δ(x)δ(y)δ(t); the transform is a sifting integral, evaluated and \
                     checked exactly (Courant–Hilbert II §VI.12; Duffy §4.4)"
            .to_string();
        (radial, kernel, identity, proof)
    } else {
        // ------------------------------------------------------------ 3-D
        let one_d = fundamental_solution(op, &space[..1], Some(t), pool)?;
        let r1 = simp(subs1(one_d.radial, one_d.radius, r, pool), pool);
        let (bessel, sign) = if cone.hyperbolic {
            ("bessel_i", 1)
        } else {
            ("bessel_j", -1)
        };
        let tail = simp(
            pool.mul(vec![
                int(sign, pool),
                cone.pre,
                cone.lam,
                div(
                    pool.func(bessel, vec![int(1, pool), pool.mul(vec![cone.lam, sigma])]),
                    pool.mul(vec![four, pi(pool), c, sigma]),
                    pool,
                ),
            ]),
            pool,
        );
        let delta_coef = div(
            cone.pre,
            pool.mul(vec![four, pi(pool), sq(c, pool), r]),
            pool,
        );
        let one_ok = one_d.status == VerificationStatus::ExactlyVerified;
        checks.push(GreenCheck {
            what: "dimension_walk",
            status: if one_ok {
                ZeroStatus::ProvedZero
            } else {
                ZeroStatus::Unknown
            },
            method: format!("fundamental_solution({name}, 1-D)"),
            detail: "the 1-D light-cone kernel E₁ = H(ct − |x|) R/(2c) is exactly verified".into(),
        });
        let walk = neg(
            div(dd(r1, r, pool)?, pool.mul(vec![two, pi(pool), r]), pool),
            pool,
        );
        let mut ck = check(
            "dimension_walk",
            sub(tail, walk, pool),
            "substitution (Bessel derivative J₀' = −J₁, I₀' = I₁)",
            "the smooth part is −(2πr)^{−1} ∂ᵣ of the 1-D kernel",
            pool,
        );
        if ck.status != ZeroStatus::ProvedZero {
            ck.status = zstatus(super::expand_powers_pos(sub(tail, walk, pool), pool), pool);
        }
        checks.push(ck);
        let jump = div(
            at(r1, r, ct, pool),
            pool.mul(vec![two, pi(pool), c, r]),
            pool,
        );
        checks.push(check(
            "dimension_walk",
            sub(delta_coef, jump, pool),
            "substitution (the Riemann function at r = ct)",
            "−(2πr)^{−1} ∂ᵣ H(ct − r) · R₁ = δ(t − r/c) R₁(ct, t)/(2πcr): the light-cone δ \
             has coefficient R₁(ct, t)/(2πcr)",
            pool,
        ));
        let fname = "__green_F";
        let f = pool.func(fname, vec![r, t]);
        let lhs = radial_op(div(dd(f, r, pool)?, r, pool), r, t, 3, &cone, pool)?;
        let rhs = div(dd(radial_op(f, r, t, 1, &cone, pool)?, r, pool)?, r, pool);
        checks.push(check(
            "commutation",
            sub(lhs, rhs, pool),
            "substitution (arbitrary radial f)",
            "L₃[(1/r)∂ᵣ f] = (1/r)∂ᵣ(L₁ f): the dimension walk maps solutions to solutions",
            pool,
        ));
        if !param_zero && super::is_symbol(cone.param, pool) {
            checks.push(check(
                "reduction",
                at(tail, cone.param, int(0, pool), pool),
                "substitution",
                format!(
                    "{} = 0 removes the Bessel tail, leaving the Kirchhoff kernel \
                     δ(t − r/c)/(4πc²r)",
                    if cone.hyperbolic { "a" } else { "m" }
                ),
                pool,
            ));
        }
        derivation.push(format!(
            "dimension walk: for a radial fundamental solution E_n of P(∂ₜ) − c²Δ, \
             E_{{n+2}} = −(2πr)^{{−1}} ∂ᵣ E_n; from the 1-D kernel H(ct − |x|) R(r, t)/(2c), \
             R = {}, the Heaviside factor gives the light-cone δ(t − r/c)/(4πc²r){} and R gives \
             the tail {}",
            if cone.hyperbolic {
                "e^{−at} I₀((a/c)√(c²t² − r²))"
            } else {
                "J₀((m/c)√(c²t² − r²))"
            },
            if cone.hyperbolic { " e^{−at}" } else { "" },
            if cone.hyperbolic {
                "+e^{−at} (a/c) I₁((a/c)σ)/(4πcσ)"
            } else {
                "−(m/c) J₁((m/c)σ)/(4πcσ)"
            },
        ));
        references.push(
            "Grafakos, Classical Fourier Analysis, Appendix B.5 (the dimension walk for radial \
             Fourier transforms)"
                .into(),
        );
        references.push("Duffy, Green's Functions with Applications §4.3–4.4".into());
        references.push("Morse & Feshbach, Methods of Theoretical Physics §7.4".into());
        let delta_cart = pool.mul(vec![
            cart(delta_coef),
            pool.func("diracdelta", vec![sub(t, div(r_cart, c, pool), pool)]),
        ]);
        let kernel = pool.add(vec![
            delta_cart,
            pool.mul(vec![
                pool.func("heaviside", vec![sub(ct, r_cart, pool)]),
                cart(tail),
            ]),
        ]);
        let identity = if cone.hyperbolic {
            "(∂ₜₜ + 2a∂ₜ − c²Δ) G = δ(x) δ(t), G = e^{−at}[δ(t − r/c)/(4πc²r) + H(ct − r) \
             (a/c) I₁((a/c)σ)/(4πcσ)], σ = √(c²t² − r²)"
                .to_string()
        } else {
            "(∂ₜₜ − c²Δ + m²) G = δ(x) δ(t), G = δ(t − r/c)/(4πc²r) − H(ct − r) \
             (m/c) J₁((m/c)σ)/(4πcσ), σ = √(c²t² − r²)"
                .to_string()
        };
        let proof = "dimension walk: E₃ = −(2πr)^{−1} ∂ᵣ E₁ has the spatial Fourier transform of \
                     E₁ (Grafakos App. B.5), which is the transfer function sin(ωt)/ω of the \
                     operator in every dimension; E₁ is exactly verified, and both parts of E₃ \
                     are checked to be the walk of E₁"
            .to_string();
        (tail, kernel, identity, proof)
    };
    // L G = 0 inside the cone: Cartesian substitution, else the radial form.
    let smooth = cart(radial);
    let res = op.apply(smooth, space, Some(t), pool)?;
    let mut hom = check(
        "homogeneous",
        res,
        "substitution",
        "L G = 0 inside the light cone (r < ct)",
        pool,
    );
    if hom.status != ZeroStatus::ProvedZero {
        let rr = radial_op(radial, r, t, n as i64, &cone, pool)?;
        let mut st = zstatus(rr, pool);
        if st != ZeroStatus::ProvedZero {
            st = zstatus(super::expand_powers_pos(simp(rr, pool), pool), pool);
        }
        if st == ZeroStatus::ProvedZero || hom.status == ZeroStatus::Unknown {
            hom.status = st;
            hom.method = "substitution (radial form: Δ = ∂ᵣᵣ + (n−1)/r ∂ᵣ for r > 0)".into();
        }
    }
    checks.push(hom);
    if cone.hyperbolic {
        // L_tel[e^{−at} V] = e^{−at}(V_tt − c²ΔV − a²V) for any V.
        let mut vargs = space.to_vec();
        vargs.push(t);
        let v = pool.func("__green_V", vargs);
        let lhs = op.apply(pool.mul(vec![cone.pre, v]), space, Some(t), pool)?;
        let kg = GreenOperator::KleinGordon {
            speed: c,
            mass: pool.mul(vec![pool.imaginary_unit(), cone.param]),
        };
        let rhs = pool.mul(vec![cone.pre, kg.apply(v, space, Some(t), pool)?]);
        checks.push(check(
            "reduction",
            sub(lhs, rhs, pool),
            "substitution (arbitrary V)",
            "(∂ₜₜ + 2a∂ₜ − c²Δ)(e^{−at}V) = e^{−at}(V_tt − c²ΔV − a²V)",
            pool,
        ));
    }
    refuse_refuted(&format!("the {n}-D {name} kernel"), &checks)?;
    let status = combine(&checks);
    let mut side = Vec::new();
    for (pname, v) in op.parameters() {
        if pname == "damping" {
            side.push(pool.pred_ge(v, int(0, pool)));
        } else {
            side.push(pool.pred_gt(v, int(0, pool)));
        }
    }
    Ok(FundamentalSolution {
        operator: op,
        dimension: int(n as i64, pool),
        space: space.to_vec(),
        time: Some(t),
        radius: r,
        kernel: simp(kernel, pool),
        radial,
        smooth,
        support: Some(pool.pred_gt(sub(ct, r_cart, pool), int(0, pool))),
        identity,
        identity_proof: proof,
        derivation,
        checks,
        status,
        side_conditions: side,
        references,
    })
}

/// The (first) `diracdelta(g)` node of `e` and its argument.
fn find_delta(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if let ExprData::Func { name, args } = pool.get(n) {
            if name == "diracdelta" && args.len() == 1 {
                return Some((n, args[0]));
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    None
}

fn mentions_sym(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    super::mentions(e, v, pool)
}
