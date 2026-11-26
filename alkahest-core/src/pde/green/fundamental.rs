//! The fundamental-solution table (`L G = δ`, causal / outgoing).

use crate::calculus::{limit, LimitDirection};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::sturm_liouville::{replace_node, simp, subs1};
use super::{
    at, branch_abs, check, combine, dd, div, fresh, is_symbol, neg, pi, refuse_refuted, sq, sqrt,
    sub, zstatus, FundamentalSolution, GreenCheck, GreenOperator,
};

fn int(n: i64, pool: &ExprPool) -> ExprId {
    pool.integer(n)
}

fn i_unit(pool: &ExprPool) -> ExprId {
    pool.imaginary_unit()
}

/// `|S^{n−1}|` for `n = 1, 2, 3`: `2`, `2π`, `4π`.
fn sphere_area(n: usize, pool: &ExprPool) -> ExprId {
    match n {
        1 => int(2, pool),
        2 => pool.mul(vec![int(2, pool), pi(pool)]),
        _ => pool.mul(vec![int(4, pool), pi(pool)]),
    }
}

/// `|S^{n−1}| = 2 π^{n/2} / Γ(n/2)` for a symbolic `n`.
fn sphere_area_sym(n: ExprId, pool: &ExprPool) -> ExprId {
    let half_n = pool.mul(vec![pool.rational(1, 2), n]);
    div(
        pool.mul(vec![int(2, pool), pool.pow(pi(pool), half_n)]),
        pool.func("gamma", vec![half_n]),
        pool,
    )
}

/// The Klein–Gordon Riemann kernel `J₀((m/c)√(c²t² − x²)) / (2c)` (valid for
/// `|x| < ct`).
pub(crate) fn klein_gordon_kernel(
    x: ExprId,
    t: ExprId,
    c: ExprId,
    m: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let s = sqrt(sub(sq(pool.mul(vec![c, t]), pool), sq(x, pool), pool), pool);
    div(
        pool.func(
            "bessel_j",
            vec![int(0, pool), pool.mul(vec![div(m, c, pool), s])],
        ),
        pool.mul(vec![int(2, pool), c]),
        pool,
    )
}

/// The telegraph kernel `e^{−at} I₀((a/c)√(c²t² − x²)) / (2c)` of
/// `u_tt + 2a u_t − c² u_xx` (valid for `|x| < ct`).
pub(crate) fn telegraph_kernel(
    x: ExprId,
    t: ExprId,
    c: ExprId,
    a: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let s = sqrt(sub(sq(pool.mul(vec![c, t]), pool), sq(x, pool), pool), pool);
    pool.mul(vec![
        pool.func("exp", vec![neg(pool.mul(vec![a, t]), pool)]),
        div(
            pool.func(
                "bessel_i",
                vec![int(0, pool), pool.mul(vec![div(a, c, pool), s])],
            ),
            pool.mul(vec![int(2, pool), c]),
            pool,
        ),
    ])
}

/// Replace the Bessel functions of positive order by their leading term at
/// `z → 0` (DLMF 10.30.2 for `K_ν`, 10.7.7 for `H_ν⁽¹⁾`): `K₁(z) ~ 1/z`,
/// `H₁⁽¹⁾(z) ~ −2i/(πz)`.  Returns `None` when a Bessel function remains
/// that has no rule here.
fn leading_terms(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let e = super::bessel_normal(e, pool).unwrap_or(e);
    let mut out = e;
    let mut stack = vec![e];
    let mut nodes = Vec::new();
    while let Some(n) = stack.pop() {
        if let ExprData::Func { name, args } = pool.get(n) {
            if name.starts_with("bessel") || name.starts_with("hankel") {
                nodes.push((name, args, n));
                continue;
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    for (name, args, n) in nodes {
        let one = args.len() == 2 && args[0] == int(1, pool);
        let rep = match (name.as_str(), one) {
            ("bessel_k", true) => pool.pow(args[1], int(-1, pool)),
            ("hankel1", true) => div(
                pool.mul(vec![int(-2, pool), i_unit(pool)]),
                pool.mul(vec![pi(pool), args[1]]),
                pool,
            ),
            _ => return None,
        };
        out = replace_node(out, n, rep, pool);
    }
    Some(out)
}

/// The flux `lim_{r→0⁺} F(r)` compared with 1.
fn flux_check(f: ExprId, r: ExprId, detail: &str, pool: &ExprPool) -> GreenCheck {
    let f = simp(f, pool);
    let has_bessel = {
        let mut stack = vec![f];
        let mut hit = false;
        while let Some(n) = stack.pop() {
            if let ExprData::Func { name, .. } = pool.get(n) {
                if name.starts_with("bessel") || name.starts_with("hankel") {
                    hit = true;
                    break;
                }
            }
            super::super::jet::push_children(n, pool, &mut stack);
        }
        hit
    };
    // Exact without a limit (symbolic dimension, power kernels).
    let direct = sub(f, int(1, pool), pool);
    if zstatus(direct, pool) == ZeroStatus::ProvedZero {
        return GreenCheck {
            what: "flux",
            status: ZeroStatus::ProvedZero,
            method: "identity".into(),
            detail: detail.into(),
        };
    }
    let (g, method) = if has_bessel {
        match leading_terms(f, pool) {
            Some(g) => (
                simp(g, pool),
                "limit of the leading term (DLMF 10.30.2 / 10.7.7)",
            ),
            None => {
                return GreenCheck {
                    what: "flux",
                    status: ZeroStatus::Unknown,
                    method: "no leading-term rule".into(),
                    detail: detail.into(),
                }
            }
        }
    } else {
        (f, "limit")
    };
    let status = match limit(g, r, pool.integer(0), LimitDirection::Plus, pool) {
        Ok(v) => zstatus(sub(v, int(1, pool), pool), pool),
        Err(_) => ZeroStatus::Unknown,
    };
    GreenCheck {
        what: "flux",
        status,
        method: method.into(),
        detail: detail.into(),
    }
}

/// `∫ φ(r) δ(α r + β) dr` over `(0, ∞)` with `α`, `β` free of `r` and the
/// root `−β/α > 0` (the parameters are positive symbols).  `None` when the
/// integrand is not of that shape.
fn sift(integrand: ExprId, r: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let factors: Vec<ExprId> = match pool.get(integrand) {
        ExprData::Mul(xs) => xs,
        _ => vec![integrand],
    };
    let pos = factors.iter().position(|&f| {
        pool.with(
            f,
            |d| matches!(d, ExprData::Func { name, args } if name == "diracdelta" && args.len() == 1),
        )
    })?;
    let arg = pool.with(factors[pos], |d| match d {
        ExprData::Func { args, .. } => Some(args[0]),
        _ => None,
    })?;
    let alpha = simp(dd(arg, r, pool).ok()?, pool);
    if super::mentions(alpha, r, pool) {
        return None;
    }
    let beta = simp(subs1(arg, r, int(0, pool), pool), pool);
    let root = simp(neg(div(beta, alpha, pool), pool), pool);
    if super::super::sturm_liouville::sign(root, pool) != super::super::sturm_liouville::Sgn::Pos {
        return None;
    }
    let abs_alpha = match super::super::sturm_liouville::sign(alpha, pool) {
        super::super::sturm_liouville::Sgn::Pos => alpha,
        super::super::sturm_liouville::Sgn::Neg => neg(alpha, pool),
        _ => return None,
    };
    let rest: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pos)
        .map(|(_, &f)| f)
        .collect();
    let rest = pool.mul(rest);
    Some(simp(div(subs1(rest, r, root, pool), abs_alpha, pool), pool))
}

fn validate(
    op: &GreenOperator,
    space: &[ExprId],
    time: Option<ExprId>,
    pool: &ExprPool,
) -> Result<(), PdeError> {
    if space.is_empty() || space.len() > 3 {
        return Err(PdeError::GreenInvalidInput(format!(
            "{} space variables: the table covers dimensions 1, 2 and 3 \
             (fundamental_solution_nd takes a symbolic dimension)",
            space.len()
        )));
    }
    let mut all = space.to_vec();
    if let Some(t) = time {
        all.push(t);
    }
    for (i, &v) in all.iter().enumerate() {
        if !is_symbol(v, pool) {
            return Err(PdeError::GreenInvalidInput(format!(
                "{} is not a symbol",
                pool.display(v)
            )));
        }
        if all[..i].contains(&v) {
            return Err(PdeError::GreenInvalidInput(format!(
                "{} is repeated",
                pool.display(v)
            )));
        }
    }
    match (op.is_time_dependent(), time) {
        (true, None) => {
            return Err(PdeError::GreenInvalidInput(format!(
                "the {} operator needs a time variable",
                op.name()
            )))
        }
        (false, Some(_)) => {
            return Err(PdeError::GreenInvalidInput(format!(
                "the {} operator has no time variable",
                op.name()
            )))
        }
        _ => {}
    }
    if matches!(
        op,
        GreenOperator::KleinGordon { .. } | GreenOperator::Telegraph { .. }
    ) && space.len() != 1
    {
        return Err(PdeError::GreenUnsupported(format!(
            "the {} kernel is implemented in one space dimension",
            op.name()
        )));
    }
    Ok(())
}

/// The fundamental solution of `op` in the Cartesian variables `space`
/// (dimension `1 ≤ n ≤ 3`) and, for an evolution operator, the time `time`:
/// the causal / outgoing kernel with `L G = δ`, its derivation, references
/// and exact checks (see the [module docs](super)).
///
/// The variables stand for `x − ξ` and `t − τ`: the source sits at the
/// origin.  Parameters should be positive symbols (or positive numbers): the
/// side conditions record `κ > 0`, `c > 0`, `k > 0`, `m > 0`.
///
/// # Errors
///
/// * `E-PDE-250` — Klein–Gordon or telegraph in more than one dimension;
/// * `E-PDE-251` — no / too many variables, a repeated or non-symbol
///   variable, a missing or spurious time variable;
/// * `E-PDE-255` — a check refuted the kernel (a bug: never returned).
pub fn fundamental_solution(
    op: GreenOperator,
    space: &[ExprId],
    time: Option<ExprId>,
    pool: &ExprPool,
) -> Result<FundamentalSolution, PdeError> {
    validate(&op, space, time, pool)?;
    let n = space.len();
    let mut avoid: Vec<ExprId> = space.to_vec();
    avoid.extend(time);
    avoid.extend(op.parameters().iter().map(|&(_, v)| v));
    let r = fresh("r", Domain::Positive, &avoid, pool);
    let x0 = space[0];
    let r_cart = if n == 1 {
        pool.func("abs", vec![x0])
    } else {
        sqrt(pool.add(space.iter().map(|&v| sq(v, pool)).collect()), pool)
    };
    let two = int(2, pool);
    let four = int(4, pool);
    let half = pool.rational(1, 2);
    let mut derivation: Vec<String> = Vec::new();
    let mut references: Vec<String> = Vec::new();
    let mut side: Vec<ExprId> = Vec::new();
    for (name, v) in op.parameters() {
        if !matches!(op, GreenOperator::Telegraph { .. }) || name != "damping" {
            side.push(pool.pred_gt(v, int(0, pool)));
        } else {
            side.push(pool.pred_ge(v, int(0, pool)));
        }
    }
    let t_or = |pool: &ExprPool| time.unwrap_or_else(|| int(0, pool));
    // radial smooth part, Cartesian kernel (with heaviside / delta), support
    let (radial, support, identity, proof): (ExprId, Option<ExprId>, String, String) = match op {
        GreenOperator::Laplace => {
            let g = match n {
                1 => pool.mul(vec![half, r]),
                2 => div(
                    pool.func("log", vec![r]),
                    pool.mul(vec![two, pi(pool)]),
                    pool,
                ),
                _ => neg(
                    div(int(1, pool), pool.mul(vec![four, pi(pool), r]), pool),
                    pool,
                ),
            };
            derivation.push(format!(
                "radial symmetry: G = G(r) with Δ G = r^{{1−n}} (r^{{n−1}} G')' = 0 for r > 0, \
                 so G' = C r^{{1−n}}; the flux condition |S^{{n−1}}| r^{{n−1}} G'(r) = 1 \
                 fixes C = 1/|S^{{n−1}}| (n = {n})"
            ));
            references.push("Evans, PDE §2.2.1 (with the sign of Δ G = δ)".into());
            references.push("Strauss, PDE §7.4; Haberman §9.5".into());
            (
                g,
                None,
                "Δ G = δ(x)".into(),
                if n == 1 {
                    "jump_formula".into()
                } else {
                    "divergence theorem: Δ G = 0 for x ≠ 0 and unit flux through every sphere \
                     (Evans §2.2.1, Theorem 1)"
                        .into()
                },
            )
        }
        GreenOperator::Biharmonic => {
            let g = match n {
                1 => div(pool.pow(r, int(3, pool)), int(12, pool), pool),
                2 => div(
                    pool.mul(vec![sq(r, pool), pool.func("log", vec![r])]),
                    pool.mul(vec![int(8, pool), pi(pool)]),
                    pool,
                ),
                _ => neg(div(r, pool.mul(vec![int(8, pool), pi(pool)]), pool), pool),
            };
            derivation.push(
                "Δ G is the Laplace fundamental solution up to a harmonic constant, so \
                 Δ² G = Δ(Δ G) = δ"
                    .into(),
            );
            references.push("Duffy, Green's Functions with Applications §5.6".into());
            references.push("Gel'fand & Shilov, Generalized Functions I, §III.2".into());
            (
                g,
                None,
                "Δ² G = δ(x)".into(),
                if n == 1 {
                    "jump_formula".into()
                } else {
                    "Δ G − (Laplace kernel) is constant (checked), and the Laplace kernel \
                     has unit flux"
                        .into()
                },
            )
        }
        GreenOperator::Helmholtz { k } => {
            let ikr = pool.mul(vec![i_unit(pool), k, r]);
            let g = match n {
                1 => neg(
                    div(
                        pool.mul(vec![i_unit(pool), pool.func("exp", vec![ikr])]),
                        pool.mul(vec![two, k]),
                        pool,
                    ),
                    pool,
                ),
                2 => pool.mul(vec![
                    neg(pool.mul(vec![pool.rational(1, 4), i_unit(pool)]), pool),
                    pool.func("hankel1", vec![int(0, pool), pool.mul(vec![k, r])]),
                ]),
                _ => neg(
                    div(
                        pool.func("exp", vec![ikr]),
                        pool.mul(vec![four, pi(pool), r]),
                        pool,
                    ),
                    pool,
                ),
            };
            derivation.push(
                "radial solutions of G'' + (n−1)G'/r + k²G = 0 are e^{±ikr}/r (3-D), \
                 H₀^{(1,2)}(kr) (2-D), e^{±ik|x|} (1-D); the outgoing one (time factor \
                 e^{−iωt}) is selected, and normalised by the unit flux at r → 0 where it \
                 behaves like the Laplace kernel"
                    .into(),
            );
            references.push("Duffy, Green's Functions with Applications §5.3".into());
            references.push("DLMF 10.7.7 (H_ν^{(1)} near 0), 10.2.5 (outgoing asymptotics)".into());
            (
                g,
                None,
                "(Δ + k²) G = δ(x), outgoing (Sommerfeld radiation condition)".into(),
                if n == 1 {
                    "jump_formula".into()
                } else {
                    "divergence theorem: (Δ + k²) G = 0 for x ≠ 0, G − (Laplace kernel) is \
                     integrable with no flux at 0, unit flux (Duffy §5.3)"
                        .into()
                },
            )
        }
        GreenOperator::ModifiedHelmholtz { k } => {
            let kr = pool.mul(vec![k, r]);
            let g = match n {
                1 => neg(
                    div(
                        pool.func("exp", vec![neg(kr, pool)]),
                        pool.mul(vec![two, k]),
                        pool,
                    ),
                    pool,
                ),
                2 => neg(
                    div(
                        pool.func("bessel_k", vec![int(0, pool), kr]),
                        pool.mul(vec![two, pi(pool)]),
                        pool,
                    ),
                    pool,
                ),
                _ => neg(
                    div(
                        pool.func("exp", vec![neg(kr, pool)]),
                        pool.mul(vec![four, pi(pool), r]),
                        pool,
                    ),
                    pool,
                ),
            };
            derivation.push(
                "radial solutions decaying at infinity: e^{−kr}/r (3-D), K₀(kr) (2-D), \
                 e^{−k|x|} (1-D), normalised by the unit flux at r → 0"
                    .into(),
            );
            references.push("Duffy, Green's Functions with Applications §5.4".into());
            references.push("DLMF 10.30.2 (K_ν near 0), 10.25.3 (decay)".into());
            (
                g,
                None,
                "(Δ − k²) G = δ(x), decaying at infinity".into(),
                if n == 1 {
                    "jump_formula".into()
                } else {
                    "divergence theorem: (Δ − k²) G = 0 for x ≠ 0, unit flux at 0".into()
                },
            )
        }
        GreenOperator::Heat { diffusivity } => {
            let t = t_or(pool);
            let kt = pool.mul(vec![diffusivity, t]);
            let g = pool.mul(vec![
                pool.pow(
                    pool.mul(vec![four, pi(pool), kt]),
                    pool.rational(-(n as i64), 2),
                ),
                pool.func(
                    "exp",
                    vec![neg(div(sq(r, pool), pool.mul(vec![four, kt]), pool), pool)],
                ),
            ]);
            derivation.push(
                "Fourier transform in x: ĝ_t + 4π²κ|ξ|²ĝ = δ(t) gives ĝ = H(t) e^{−4π²κ|ξ|²t}; \
                 its inverse is the Gaussian (4πκt)^{−n/2} e^{−|x|²/(4κt)}, a product of \
                 one-dimensional kernels"
                    .into(),
            );
            references.push("Evans, PDE §2.3.1; Strauss, PDE §2.4".into());
            references.push("Duffy, Green's Functions with Applications ch. 3".into());
            (
                g,
                Some(pool.pred_gt(t, int(0, pool))),
                "(∂ₜ − κΔ) G = δ(x) δ(t), G = 0 for t < 0".into(),
                "approximate identity: (∂ₜ − κΔ)G = 0 for t > 0, G ≥ 0 with unit mass for every \
                 t > 0 and G → 0 off x = 0 as t → 0⁺, so G(·, t) → δ weakly (Evans §2.3.1, \
                 Theorem 1)"
                    .into(),
            )
        }
        GreenOperator::Wave { speed: c } => {
            let t = t_or(pool);
            let ct = pool.mul(vec![c, t]);
            match n {
                1 => {
                    let g = div(int(1, pool), pool.mul(vec![two, c]), pool);
                    derivation.push(
                        "d'Alembert: G = H(ct − |x|)/(2c), the response to u(·,0) = 0, \
                         u_t(·,0) = δ"
                            .into(),
                    );
                    references.push("Strauss, PDE §2.1 and §12.2; Evans §2.4.1".into());
                    (
                        g,
                        Some(pool.pred_gt(sub(ct, r, pool), int(0, pool))),
                        "(∂ₜₜ − c²∂ₓₓ) G = δ(x) δ(t), G = H(ct − |x|)/(2c)".into(),
                        "transfer function: the spatial Fourier transform of G equals \
                         sin(2πcξt)/(2πcξ), the solution of ĝ'' + 4π²c²ξ²ĝ = 0, ĝ(0) = 0, \
                         ĝ'(0) = 1 (checked), so G(·,0⁺) = 0 and G_t(·,0⁺) = δ"
                            .into(),
                    )
                }
                2 => {
                    let g = div(
                        int(1, pool),
                        pool.mul(vec![
                            two,
                            pi(pool),
                            c,
                            sqrt(sub(sq(ct, pool), sq(r, pool), pool), pool),
                        ]),
                        pool,
                    );
                    derivation.push(
                        "Hadamard's method of descent from the Kirchhoff kernel: integrating \
                         δ(ct − ρ)/(4πcρ) over the third coordinate gives \
                         H(ct − r)/(2πc√(c²t² − r²)) (Poisson's formula)"
                            .into(),
                    );
                    references.push("Evans, PDE §2.4.1(b) (Poisson's formula)".into());
                    references.push("Duffy, Green's Functions with Applications §4.4".into());
                    (
                        g,
                        Some(pool.pred_gt(sub(ct, r, pool), int(0, pool))),
                        "(∂ₜₜ − c²Δ) G = δ(x) δ(t), G = H(ct − r)/(2πc√(c²t² − r²))".into(),
                        "G solves the equation inside the light cone, vanishes outside it, \
                         and ∫G dx = t (checked), so G(·,0⁺) = 0, G_t(·,0⁺) = δ \
                         (Evans §2.4.1(b))"
                            .into(),
                    )
                }
                _ => {
                    let g = div(
                        pool.func("diracdelta", vec![sub(t, div(r, c, pool), pool)]),
                        pool.mul(vec![four, pi(pool), sq(c, pool), r]),
                        pool,
                    );
                    derivation.push(
                        "spherical means: r G = F(t − r/c) solves the 1-D wave equation; the \
                         outgoing solution with unit strength is δ(t − r/c)/(4πc²r) \
                         (Kirchhoff's formula)"
                            .into(),
                    );
                    references.push("Evans, PDE §2.4.1(c) (Kirchhoff's formula)".into());
                    references.push("Duffy, Green's Functions with Applications §4.4".into());
                    (
                        g,
                        None,
                        "(∂ₜₜ − c²Δ) G = δ(x) δ(t), G = δ(t − r/c)/(4πc²r)".into(),
                        "r G is a function of t − r/c (checked), and ∫G dx = t (sifting, \
                         checked), so G(·,0⁺) = 0, G_t(·,0⁺) = δ (Evans §2.4.1(c))"
                            .into(),
                    )
                }
            }
        }
        GreenOperator::KleinGordon { speed: c, mass: m } => {
            let t = t_or(pool);
            let g = klein_gordon_kernel(r, t, c, m, pool);
            derivation.push(
                "Riemann's method: the Riemann function R = J₀((m/c)√(c²t² − x²)) solves the \
                 equation and equals 1 on the characteristics x = ±ct, so \
                 G = H(ct − |x|) R/(2c) — the d'Alembert kernel times R"
                    .into(),
            );
            references.push("Courant & Hilbert, Methods II §V.5 (Riemann's method)".into());
            references
                .push("Duffy, Green's Functions with Applications §4.3 (Klein–Gordon)".into());
            (
                g,
                Some(pool.pred_gt(sub(pool.mul(vec![c, t]), r, pool), int(0, pool))),
                "(∂ₜₜ − c²∂ₓₓ + m²) G = δ(x) δ(t), G = H(ct − |x|) J₀((m/c)√(c²t² − x²))/(2c)"
                    .into(),
                "Riemann's method: R solves the equation inside the cone (checked) and R = 1 on \
                 x = ±ct (checked), so H(ct − |x|) R/(2c) has the jump of the d'Alembert kernel \
                 across the characteristics (Courant–Hilbert II §V.5)"
                    .into(),
            )
        }
        GreenOperator::Telegraph {
            speed: c,
            damping: a,
        } => {
            let t = t_or(pool);
            let g = telegraph_kernel(r, t, c, a, pool);
            derivation.push(
                "u = e^{−at} v turns u_tt + 2a u_t − c²u_xx into v_tt − c²v_xx − a²v, the \
                 Klein–Gordon operator with m² = −a²; J₀(i z) = I₀(z), so \
                 G = H(ct − |x|) e^{−at} I₀((a/c)√(c²t² − x²))/(2c)"
                    .into(),
            );
            references.push("Courant & Hilbert, Methods II §V.5".into());
            references.push("Duffy, Green's Functions with Applications §4.2 (telegraph)".into());
            (
                g,
                Some(pool.pred_gt(sub(pool.mul(vec![c, t]), r, pool), int(0, pool))),
                "(∂ₜₜ + 2a∂ₜ − c²∂ₓₓ) G = δ(x) δ(t), \
                 G = H(ct − |x|) e^{−at} I₀((a/c)√(c²t² − x²))/(2c)"
                    .into(),
                "the reduction u = e^{−at}v (checked) and Riemann's method for v (solution inside \
                 the cone and value 1 on the characteristics, checked)"
                    .into(),
            )
        }
    };
    // Cartesian smooth part and the kernel as a distribution.
    let smooth = simp(subs1(radial, r, r_cart, pool), pool);
    let smooth = if n == 1 {
        // |x|² = x²
        simp(
            replace_node(smooth, sq(r_cart, pool), sq(x0, pool), pool),
            pool,
        )
    } else {
        smooth
    };
    let kernel = match op {
        GreenOperator::Heat { .. } => {
            pool.mul(vec![pool.func("heaviside", vec![t_or(pool)]), smooth])
        }
        GreenOperator::Wave { speed: c } if n <= 2 => pool.mul(vec![
            pool.func(
                "heaviside",
                vec![sub(pool.mul(vec![c, t_or(pool)]), r_cart, pool)],
            ),
            smooth,
        ]),
        GreenOperator::KleinGordon { speed: c, .. } | GreenOperator::Telegraph { speed: c, .. } => {
            pool.mul(vec![
                pool.func(
                    "heaviside",
                    vec![sub(pool.mul(vec![c, t_or(pool)]), r_cart, pool)],
                ),
                smooth,
            ])
        }
        _ => smooth,
    };
    // ---------------------------------------------------------------- checks
    let mut checks: Vec<GreenCheck> = Vec::new();
    let stationary_1d = n == 1 && !op.is_time_dependent();
    if stationary_1d {
        // Jump formula.
        let order = if matches!(op, GreenOperator::Biharmonic) {
            4
        } else {
            2
        };
        let gp = branch_abs(smooth, x0, 1, pool);
        let gm = branch_abs(smooth, x0, -1, pool);
        for (side_name, g) in [("x > 0", gp), ("x < 0", gm)] {
            let res = op.apply(g, space, time, pool)?;
            checks.push(check(
                "homogeneous",
                res,
                "substitution",
                format!("L G = 0 for {side_name}"),
                pool,
            ));
        }
        let (mut dp, mut dm) = (gp, gm);
        for j in 0..order {
            let jump = sub(
                at(dp, x0, int(0, pool), pool),
                at(dm, x0, int(0, pool), pool),
                pool,
            );
            let (res, what) = if j + 1 == order {
                (sub(jump, int(1, pool), pool), format!("[G^({j})](0) = 1"))
            } else {
                (jump, format!("[G^({j})](0) = 0"))
            };
            checks.push(check("distributional", res, "jump_formula", what, pool));
            dp = dd(dp, x0, pool)?;
            dm = dd(dm, x0, pool)?;
        }
    } else {
        match op {
            GreenOperator::Wave { speed: c } if n == 3 => {
                // r G = F(t − r/c)/(4πc²): any F solves the radial equation.
                let t = t_or(pool);
                let fname = "__green_F";
                let f = pool.func(fname, vec![sub(t, div(r, c, pool), pool)]);
                let res = sub(
                    dd(dd(f, t, pool)?, t, pool)?,
                    pool.mul(vec![sq(c, pool), dd(dd(f, r, pool)?, r, pool)?]),
                    pool,
                );
                checks.push(check(
                    "homogeneous",
                    res,
                    "substitution (arbitrary profile F)",
                    "(∂ₜₜ − c²∂ᵣᵣ)(r G) = 0 for r > 0: r G = δ(t − r/c)/(4πc²) is a function \
                     of t − r/c",
                    pool,
                ));
            }
            _ => {
                let res = op.apply(smooth, space, time, pool)?;
                checks.push(check(
                    "homogeneous",
                    res,
                    "substitution",
                    if support.is_some() {
                        "L G = 0 on the support (t > 0 / inside the light cone)".to_string()
                    } else {
                        "L G = 0 for x ≠ 0".to_string()
                    },
                    pool,
                ));
            }
        }
    }
    // Normalisation.
    match op {
        GreenOperator::Laplace
        | GreenOperator::Helmholtz { .. }
        | GreenOperator::ModifiedHelmholtz { .. }
            if n >= 2 =>
        {
            let f = pool.mul(vec![
                sphere_area(n, pool),
                pool.pow(r, int(n as i64 - 1, pool)),
                dd(radial, r, pool)?,
            ]);
            checks.push(flux_check(
                f,
                r,
                "lim_{r→0} |S^{n−1}| r^{n−1} ∂ᵣG = 1",
                pool,
            ));
            if let GreenOperator::Helmholtz { k } = op {
                if n == 3 {
                    // r²(∂ᵣG − ikG) e^{−ikr} is constant: O(1/r²) decay.
                    let rad = pool.mul(vec![
                        sq(r, pool),
                        sub(
                            dd(radial, r, pool)?,
                            pool.mul(vec![i_unit(pool), k, radial]),
                            pool,
                        ),
                        pool.func("exp", vec![neg(pool.mul(vec![i_unit(pool), k, r]), pool)]),
                    ]);
                    checks.push(check(
                        "radiation",
                        dd(rad, r, pool)?,
                        "substitution",
                        "r²(∂ᵣG − ikG) e^{−ikr} is constant, so r(∂ᵣG − ikG) → 0 (Sommerfeld)",
                        pool,
                    ));
                }
            }
        }
        GreenOperator::Biharmonic if n >= 2 => {
            let lap_g = op_laplace_radial(radial, r, int(n as i64, pool), pool)?;
            let lap_kernel = fundamental_solution(GreenOperator::Laplace, space, None, pool)?;
            let diff = sub(lap_g, lap_kernel.radial, pool);
            checks.push(check(
                "laplacian",
                dd(diff, r, pool)?,
                "substitution",
                "Δ G − (Laplace kernel) is constant",
                pool,
            ));
        }
        GreenOperator::Heat { diffusivity } => {
            let t = t_or(pool);
            let mass = if n == 1 {
                crate::integrate::integrate_definite(
                    smooth,
                    x0,
                    neg(pool.pos_infinity(), pool),
                    pool.pos_infinity(),
                    pool,
                )
                .ok()
                .map(|v| v.value)
            } else {
                crate::integrate::integrate_definite(
                    pool.mul(vec![
                        sphere_area(n, pool),
                        pool.pow(r, int(n as i64 - 1, pool)),
                        radial,
                    ]),
                    r,
                    int(0, pool),
                    pool.pos_infinity(),
                    pool,
                )
                .ok()
                .map(|v| v.value)
            };
            checks.push(match mass {
                Some(v) => check(
                    "mass",
                    sub(v, int(1, pool), pool),
                    "integrate",
                    "∫ G(x, t) dx = 1 for t > 0",
                    pool,
                ),
                None => GreenCheck {
                    what: "mass",
                    status: ZeroStatus::Unknown,
                    method: "integrate did not close".into(),
                    detail: "∫ G(x, t) dx = 1 for t > 0".into(),
                },
            });
            checks.push(concentration_check(radial, r, t, pool));
            if n == 1 {
                let xi = fresh("xi", Domain::Real, &avoid, pool);
                let want = pool.func(
                    "exp",
                    vec![neg(
                        pool.mul(vec![four, sq(pi(pool), pool), diffusivity, sq(xi, pool), t]),
                        pool,
                    )],
                );
                checks.push(
                    match crate::transform::fourier_transform(smooth, x0, xi, pool) {
                        Ok(v) => check(
                            "transfer",
                            sub(v, want, pool),
                            "fourier_transform",
                            "Ĝ(ξ, t) = e^{−4π²κξ²t}, the solution of ĝ' + 4π²κξ²ĝ = 0, ĝ(0) = 1",
                            pool,
                        ),
                        Err(_) => GreenCheck {
                            what: "transfer",
                            status: ZeroStatus::Unknown,
                            method: "fourier_transform did not close".into(),
                            detail: "Ĝ(ξ, t) = e^{−4π²κξ²t}".into(),
                        },
                    },
                );
            }
        }
        GreenOperator::Wave { speed: c } => {
            let t = t_or(pool);
            let ct = pool.mul(vec![c, t]);
            let mass = match n {
                1 => crate::integrate::integrate_definite(smooth, x0, neg(ct, pool), ct, pool)
                    .ok()
                    .map(|v| (v.value, "integrate")),
                2 => crate::integrate::integrate_definite(
                    pool.mul(vec![sphere_area(2, pool), r, radial]),
                    r,
                    int(0, pool),
                    ct,
                    pool,
                )
                .ok()
                .map(|v| (v.value, "integrate")),
                _ => sift(
                    simp(
                        pool.mul(vec![sphere_area(3, pool), sq(r, pool), radial]),
                        pool,
                    ),
                    r,
                    pool,
                )
                .map(|v| (v, "sifting: δ(αr + β) = δ(r − r₀)/|α|")),
            };
            checks.push(match mass {
                Some((v, how)) => check(
                    "mass",
                    sub(v, t, pool),
                    how,
                    "∫ G(x, t) dx = t (the ξ = 0 transfer function)",
                    pool,
                ),
                None => GreenCheck {
                    what: "mass",
                    status: ZeroStatus::Unknown,
                    method: "integral did not close".into(),
                    detail: "∫ G(x, t) dx = t".into(),
                },
            });
            if n == 1 {
                let xi = fresh("xi", Domain::Real, &avoid, pool);
                let e = pool.func(
                    "exp",
                    vec![pool.mul(vec![int(-2, pool), pi(pool), i_unit(pool), xi, x0])],
                );
                let tf = crate::integrate::integrate_definite(
                    pool.mul(vec![e, smooth]),
                    x0,
                    neg(ct, pool),
                    ct,
                    pool,
                )
                .ok()
                .map(|v| v.value);
                let want = div(
                    pool.func("sin", vec![pool.mul(vec![two, pi(pool), c, xi, t])]),
                    pool.mul(vec![two, pi(pool), c, xi]),
                    pool,
                );
                checks.push(match tf {
                    Some(v) => {
                        let res = sub(v, want, pool);
                        let mut ck = check(
                            "transfer",
                            res,
                            "integrate",
                            "Ĝ(ξ, t) = sin(2πcξt)/(2πcξ)",
                            pool,
                        );
                        if ck.status != ZeroStatus::ProvedZero {
                            let res2 = super::trig_to_exp(res, pool);
                            ck.status = zstatus(res2, pool);
                        }
                        ck
                    }
                    None => GreenCheck {
                        what: "transfer",
                        status: ZeroStatus::Unknown,
                        method: "integrate did not close".into(),
                        detail: "Ĝ(ξ, t) = sin(2πcξt)/(2πcξ)".into(),
                    },
                });
            }
        }
        GreenOperator::KleinGordon { speed: c, mass: m } => {
            let t = t_or(pool);
            let ct = pool.mul(vec![c, t]);
            for (sgn, label) in [(1, "x = ct"), (-1, "x = −ct")] {
                let on = at(
                    pool.mul(vec![two, c, smooth]),
                    x0,
                    pool.mul(vec![int(sgn, pool), ct]),
                    pool,
                );
                checks.push(check(
                    "characteristic",
                    sub(on, int(1, pool), pool),
                    "substitution",
                    format!("R = 2c G = 1 on {label}"),
                    pool,
                ));
            }
            let m0 = at(smooth, m, int(0, pool), pool);
            checks.push(check(
                "reduction",
                sub(m0, div(int(1, pool), pool.mul(vec![two, c]), pool), pool),
                "substitution",
                "m = 0 gives the d'Alembert kernel 1/(2c)",
                pool,
            ));
        }
        GreenOperator::Telegraph {
            speed: c,
            damping: a,
        } => {
            let t = t_or(pool);
            let ct = pool.mul(vec![c, t]);
            let eat = pool.func("exp", vec![pool.mul(vec![a, t])]);
            for (sgn, label) in [(1, "x = ct"), (-1, "x = −ct")] {
                let on = at(
                    pool.mul(vec![two, c, eat, smooth]),
                    x0,
                    pool.mul(vec![int(sgn, pool), ct]),
                    pool,
                );
                checks.push(check(
                    "characteristic",
                    sub(on, int(1, pool), pool),
                    "substitution",
                    format!("R = 2c e^{{at}} G = 1 on {label}"),
                    pool,
                ));
            }
            // L_tel[e^{−at} V] = e^{−at} (V_tt − c²V_xx − a²V) for any V.
            let v = pool.func("__green_V", vec![x0, t]);
            let emat = pool.func("exp", vec![neg(pool.mul(vec![a, t]), pool)]);
            let lhs = op.apply(pool.mul(vec![emat, v]), space, time, pool)?;
            let kg = GreenOperator::KleinGordon {
                speed: c,
                mass: pool.mul(vec![i_unit(pool), a]),
            };
            let rhs = pool.mul(vec![emat, kg.apply(v, space, time, pool)?]);
            checks.push(check(
                "reduction",
                sub(lhs, rhs, pool),
                "substitution (arbitrary V)",
                "(∂ₜₜ + 2a∂ₜ − c²∂ₓₓ)(e^{−at}V) = e^{−at}(V_tt − c²V_xx − a²V)",
                pool,
            ));
        }
        _ => {}
    }
    refuse_refuted(&format!("the {} fundamental solution", op.name()), &checks)?;
    let status = combine(&checks);
    let mut identity_proof = proof;
    if stationary_1d {
        identity_proof =
            "jump formula: for G smooth on each side of 0, L G = Σ_j c_j δ^{(j)} with \
             c_j built from the jumps [G^{(i)}](0); the checked jumps give L G = δ \
             (Strauss §12.1; Gel'fand–Shilov I §I.2)"
                .into();
    }
    Ok(FundamentalSolution {
        operator: op,
        dimension: int(n as i64, pool),
        space: space.to_vec(),
        time,
        radius: r,
        kernel: simp(kernel, pool),
        radial: simp(radial, pool),
        smooth,
        support,
        identity,
        identity_proof,
        derivation,
        checks,
        status,
        side_conditions: side,
        references,
    })
}

fn op_laplace_radial(e: ExprId, r: ExprId, n: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    GreenOperator::Laplace.apply_radial(e, r, n, None, pool)
}

/// `lim_{t→0⁺} G(r, t) = 0` for `r > 0`.
fn concentration_check(radial: ExprId, r: ExprId, t: ExprId, pool: &ExprPool) -> GreenCheck {
    // t = 1/s, s → +∞: a polynomial in s times e^{−c s}.
    let s = fresh("s", Domain::Positive, &[radial, r, t], pool);
    let g = simp(subs1(radial, t, pool.pow(s, pool.integer(-1)), pool), pool);
    let g = simp(super::expand_powers_pos(g, pool), pool);
    let status = match limit(
        g,
        s,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    ) {
        Ok(v) => zstatus(v, pool),
        Err(_) => ZeroStatus::Unknown,
    };
    GreenCheck {
        what: "concentration",
        status,
        method: "limit".into(),
        detail: "G(x, t) → 0 as t → 0⁺ for x ≠ 0".into(),
    }
}

/// The fundamental solution in dimension `n` given as a **symbol** (Laplace
/// for `n ≥ 3`, heat for `n ≥ 1`), as a radial function of `r` (and `t`).
///
/// Laplace: `G = −Γ(n/2) / (2(n − 2) π^{n/2}) · r^{2−n}`; heat:
/// `G = H(t) (4πκt)^{−n/2} e^{−r²/(4κt)}`.  Checks: the radial equation
/// `G'' + (n − 1)G'/r = 0` (resp. `G_t = κ(G'' + (n−1)G'/r)`) identically in
/// `n`, the flux `|S^{n−1}| r^{n−1} G' = 1` with `|S^{n−1}| = 2π^{n/2}/Γ(n/2)`,
/// and for the heat kernel the unit mass through the Gaussian moment
/// `∫₀^∞ r^{n−1} e^{−a r²} dr = Γ(n/2)/(2a^{n/2})` (DLMF 5.9.1).
///
/// # Errors
///
/// `E-PDE-250` for other operators; `E-PDE-251` when `n`, `r` or `t` is not
/// a symbol, or `t` is missing for the heat operator.
pub fn fundamental_solution_nd(
    op: GreenOperator,
    n: ExprId,
    r: ExprId,
    time: Option<ExprId>,
    pool: &ExprPool,
) -> Result<FundamentalSolution, PdeError> {
    for v in [Some(n), Some(r), time].into_iter().flatten() {
        if !is_symbol(v, pool) {
            return Err(PdeError::GreenInvalidInput(format!(
                "{} is not a symbol",
                pool.display(v)
            )));
        }
    }
    let half_n = pool.mul(vec![pool.rational(1, 2), n]);
    let area = sphere_area_sym(n, pool);
    let mut checks = Vec::new();
    let mut side = Vec::new();
    let mut derivation = Vec::new();
    let mut references = Vec::new();
    let (g, kernel, support, identity, proof) = match op {
        GreenOperator::Laplace => {
            if time.is_some() {
                return Err(PdeError::GreenInvalidInput(
                    "the Laplace operator has no time variable".into(),
                ));
            }
            let g = neg(
                div(
                    pool.mul(vec![
                        pool.func("gamma", vec![half_n]),
                        pool.pow(r, sub(int(2, pool), n, pool)),
                    ]),
                    pool.mul(vec![
                        int(2, pool),
                        sub(n, int(2, pool), pool),
                        pool.pow(pi(pool), half_n),
                    ]),
                    pool,
                ),
                pool,
            );
            side.push(pool.pred_ge(n, int(3, pool)));
            derivation.push(
                "G = C r^{2−n} solves r^{1−n}(r^{n−1}G')' = 0; the flux \
                 |S^{n−1}| r^{n−1} G' = 1 with |S^{n−1}| = 2π^{n/2}/Γ(n/2) gives \
                 C = −1/((n − 2)|S^{n−1}|)"
                    .into(),
            );
            references.push("Evans, PDE §2.2.1 (α(n) = π^{n/2}/Γ(n/2 + 1))".into());
            (
                g,
                g,
                None,
                "Δ G = δ(x) in ℝⁿ, n ≥ 3".to_string(),
                "divergence theorem: Δ G = 0 for r > 0 and unit flux (Evans §2.2.1)".to_string(),
            )
        }
        GreenOperator::Heat { diffusivity } => {
            let t = time.ok_or_else(|| {
                PdeError::GreenInvalidInput("the heat operator needs a time variable".into())
            })?;
            let kt = pool.mul(vec![diffusivity, t]);
            let g = pool.mul(vec![
                pool.pow(
                    pool.mul(vec![int(4, pool), pi(pool), kt]),
                    neg(half_n, pool),
                ),
                pool.func(
                    "exp",
                    vec![neg(
                        div(sq(r, pool), pool.mul(vec![int(4, pool), kt]), pool),
                        pool,
                    )],
                ),
            ]);
            side.push(pool.pred_gt(diffusivity, int(0, pool)));
            side.push(pool.pred_ge(n, int(1, pool)));
            derivation.push(
                "the n-dimensional kernel is the product of n one-dimensional Gaussians \
                 (4πκt)^{−1/2} e^{−xᵢ²/(4κt)}"
                    .into(),
            );
            references.push("Evans, PDE §2.3.1".into());
            let kernel = pool.mul(vec![pool.func("heaviside", vec![t]), g]);
            (
                g,
                kernel,
                Some(pool.pred_gt(t, int(0, pool))),
                "(∂ₜ − κΔ) G = δ(x) δ(t) in ℝⁿ".to_string(),
                "approximate identity (Evans §2.3.1, Theorem 1)".to_string(),
            )
        }
        _ => {
            return Err(PdeError::GreenUnsupported(format!(
                "a symbolic dimension is implemented for the Laplace and heat operators, not {}",
                op.name()
            )))
        }
    };
    let res = op.apply_radial(g, r, n, time, pool)?;
    checks.push(check(
        "homogeneous_radial",
        res,
        "substitution",
        "L G = 0 for r > 0, identically in n",
        pool,
    ));
    match op {
        GreenOperator::Laplace => {
            let f = pool.mul(vec![
                area,
                pool.pow(r, sub(n, int(1, pool), pool)),
                dd(g, r, pool)?,
            ]);
            checks.push(flux_check(f, r, "|S^{n−1}| r^{n−1} ∂ᵣG = 1", pool));
        }
        GreenOperator::Heat { diffusivity } => {
            let t = time.expect("checked");
            // ∫₀^∞ r^{n−1} e^{−a r²} dr = Γ(n/2) / (2 a^{n/2}), a = 1/(4κt).
            let a = div(
                int(1, pool),
                pool.mul(vec![int(4, pool), diffusivity, t]),
                pool,
            );
            let moment = div(
                pool.func("gamma", vec![half_n]),
                pool.mul(vec![int(2, pool), pool.pow(a, half_n)]),
                pool,
            );
            let prefactor = pool.pow(
                pool.mul(vec![int(4, pool), pi(pool), diffusivity, t]),
                neg(half_n, pool),
            );
            let mass = pool.mul(vec![area, prefactor, moment]);
            checks.push(check(
                "mass",
                sub(mass, int(1, pool), pool),
                "Gaussian moment (DLMF 5.9.1)",
                "|S^{n−1}| ∫₀^∞ r^{n−1} G dr = 1",
                pool,
            ));
            checks.push(concentration_check(g, r, t, pool));
        }
        _ => {}
    }
    refuse_refuted(
        &format!("the n-D {} fundamental solution", op.name()),
        &checks,
    )?;
    let status = combine(&checks);
    Ok(FundamentalSolution {
        operator: op,
        dimension: n,
        space: Vec::new(),
        time,
        radius: r,
        kernel: simp(kernel, pool),
        radial: simp(g, pool),
        smooth: simp(g, pool),
        support,
        identity,
        identity_proof: proof,
        derivation,
        checks,
        status,
        side_conditions: side,
        references,
    })
}

/// The light-cone kernels of `u_tt + D u_t + R u − c² u_xx = 0` (M0's named
/// `klein_gordon` / `telegraph` equations), for the transform method's
/// representation `u = e^{−Dt/2}(f(x − ct) + f(x + ct))/2
/// + ∫_{x−ct}^{x+ct} K_f(x − s, t) f(s) ds + ∫_{x−ct}^{x+ct} K_g(x − s, t) g(s) ds`.
///
/// With `M² = R − D²/4`, `u = e^{−Dt/2} v` reduces the equation to
/// Klein–Gordon with mass `M`, whose Riemann function is
/// `J₀((M/c)√(c²t² − x²))` (`I₀((|M|/c)√(…))` for `M² < 0`); then
/// `K_g = e^{−Dt/2} R/(2c)` and `K_f = (∂ₜ + D) K_g`.  `None` when the
/// equation is not of that shape, `M² = 0`, or the sign of `M²` is unknown.
pub(crate) struct LightCone {
    pub(crate) k_f: ExprId,
    pub(crate) k_g: ExprId,
    pub(crate) decay: ExprId,
    pub(crate) speed: ExprId,
    pub(crate) riemann: ExprId,
}

pub(crate) fn light_cone(
    eq: ExprId,
    u: ExprId,
    x: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Option<LightCone> {
    use super::super::sturm_liouville::{sign, Sgn};
    let info = super::super::classify::classify_pde(eq, u, pool).ok()?;
    let named = info
        .named
        .iter()
        .find(|n| n.name == "klein_gordon" || n.name == "telegraph")?;
    if named.time != Some(t) {
        return None;
    }
    let get = |k: &str| {
        named
            .parameters
            .iter()
            .find(|(n, _)| n == k)
            .map(|&(_, v)| v)
    };
    if get("source").is_some() {
        return None;
    }
    let c2 = get("speed_squared")?;
    let dmp = get("damping").unwrap_or_else(|| int(0, pool));
    let r = get("mass_squared")
        .or_else(|| get("reaction"))
        .unwrap_or_else(|| int(0, pool));
    let c = simp(super::expand_powers_pos(sqrt(c2, pool), pool), pool);
    if sign(c, pool) != Sgn::Pos {
        return None;
    }
    let m2 = simp(
        sub(r, pool.mul(vec![pool.rational(1, 4), sq(dmp, pool)]), pool),
        pool,
    );
    let s = sqrt(sub(sq(pool.mul(vec![c, t]), pool), sq(x, pool), pool), pool);
    let riemann = match sign(m2, pool) {
        Sgn::Pos => {
            let m = simp(super::expand_powers_pos(sqrt(m2, pool), pool), pool);
            pool.func(
                "bessel_j",
                vec![int(0, pool), pool.mul(vec![div(m, c, pool), s])],
            )
        }
        Sgn::Neg => {
            let m = simp(
                super::expand_powers_pos(sqrt(neg(m2, pool), pool), pool),
                pool,
            );
            pool.func(
                "bessel_i",
                vec![int(0, pool), pool.mul(vec![div(m, c, pool), s])],
            )
        }
        _ => return None,
    };
    let decay = simp(
        pool.func(
            "exp",
            vec![neg(pool.mul(vec![pool.rational(1, 2), dmp, t]), pool)],
        ),
        pool,
    );
    let k_g = simp(
        div(
            pool.mul(vec![decay, riemann]),
            pool.mul(vec![int(2, pool), c]),
            pool,
        ),
        pool,
    );
    let k_f = simp(
        pool.add(vec![dd(k_g, t, pool).ok()?, pool.mul(vec![dmp, k_g])]),
        pool,
    );
    Some(LightCone {
        k_f,
        k_g,
        decay,
        speed: c,
        riemann,
    })
}
