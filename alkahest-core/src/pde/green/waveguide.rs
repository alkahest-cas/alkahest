//! Helmholtz Green's functions of strips and slabs (waveguides): the
//! transverse eigenfunction (mode) series with outgoing longitudinal kernels.
//!
//! On `0 < y < L` with a Dirichlet or Neumann condition on each wall, the
//! transverse problem `−φ'' = λφ` (M3's [`sturm_liouville`]) has modes
//! `φₙ` with squared norms `‖φₙ‖²`, and
//!
//! `G(x, ξ) = Σₙ φₙ(y) φₙ(η) / ‖φₙ‖² · gₙ(x∥ − ξ∥)`,
//!
//! where `gₙ` is the outgoing Helmholtz kernel of the longitudinal
//! coordinates with wavenumber `κₙ = √(k² − λₙ)` on the principal branch:
//! `gₙ = e^{iκₙ|X|}/(2iκₙ)` for a strip, `gₙ = −(i/4) H₀⁽¹⁾(κₙ ρ)` for a slab.
//! Propagating modes (`λₙ < k²`, `κₙ > 0`) are outgoing waves; evanescent
//! modes (`λₙ > k²`, `κₙ = i√(λₙ − k²)`) decay exponentially.  That choice is
//! the radiation condition: as `|x∥| → ∞`, `G` is a finite sum of outgoing
//! propagating modes plus exponentially small terms.  At a cutoff
//! (`k² = λₙ` for some `n`) the Green's function does not exist.

use crate::ball::IntervalEval;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::sturm_liouville::{
    simp, sturm_liouville, subs1, EndCondition, LowerSpectrum, SlBoundary, SturmLiouvilleProblem,
};
use super::super::sum::sum;
use super::super::verify::VerificationStatus;
use super::{
    branch_abs, check, combine, div, fresh, fundamental_solution, refuse_refuted, sq, sqrt, sub,
    zstatus, GreenCheck, GreenFunction, GreenOperator, WallCondition,
};

fn end(w: WallCondition) -> EndCondition {
    match w {
        WallCondition::Dirichlet => EndCondition::Dirichlet,
        WallCondition::Neumann => EndCondition::Neumann,
    }
}

/// `e` with every `space[i] → space[i] − source[i]` for the listed indices.
fn shift(e: ExprId, vars: &[ExprId], pts: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut m = std::collections::HashMap::new();
    for (&v, &p) in vars.iter().zip(pts) {
        m.insert(v, sub(v, p, pool));
    }
    crate::kernel::subs::subs(e, &m, pool)
}

fn is_numeric(e: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        match pool.get(n) {
            ExprData::Symbol { name, .. } => {
                if name != "pi" {
                    return false;
                }
            }
            _ => super::super::jet::push_children(n, pool, &mut stack),
        }
    }
    true
}

/// The Helmholtz Green's function of the strip / slab `0 < x_axis < width`.
#[allow(clippy::too_many_arguments)]
pub(super) fn helmholtz_waveguide(
    k: ExprId,
    space: &[ExprId],
    source: &[ExprId],
    axis: usize,
    width: ExprId,
    walls: &[WallCondition],
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    let n = space.len();
    if n != 2 && n != 3 {
        return Err(PdeError::GreenRegionUnsupported(format!(
            "the Helmholtz waveguide is implemented for strips (2 variables) and slabs \
             (3 variables), not {n} variables"
        )));
    }
    let y = space[axis];
    let eta = source[axis];
    let long: Vec<usize> = (0..n).filter(|&i| i != axis).collect();
    let long_vars: Vec<ExprId> = long.iter().map(|&i| space[i]).collect();
    let long_src: Vec<ExprId> = long.iter().map(|&i| source[i]).collect();
    let mut avoid: Vec<ExprId> = space.to_vec();
    avoid.extend(source);
    avoid.push(width);
    avoid.push(k);
    // Transverse modes (M3).
    let problem = SturmLiouvilleProblem::interval(
        y,
        pool.integer(0),
        width,
        SlBoundary::Separated {
            left: end(walls[0]),
            right: end(walls[1]),
        },
        pool,
    )?;
    let sys = sturm_liouville(&problem, pool)?;
    if !matches!(sys.lower, LowerSpectrum::None) {
        return Err(PdeError::GreenSeriesUnsupported(
            "the transverse problem has (possibly) eigenvalues below 0".into(),
        ));
    }
    // The longitudinal kernel for a generic wavenumber κ > 0 (the table).
    let kap = fresh("kappa", Domain::Positive, &avoid, pool);
    let table = fundamental_solution(GreenOperator::Helmholtz { k: kap }, &long_vars, None, pool)?;
    let mut checks: Vec<GreenCheck> = Vec::new();
    let mut notes: Vec<String> = sys.notes.clone();
    checks.push(GreenCheck {
        what: "longitudinal",
        status: if table.status == VerificationStatus::ExactlyVerified {
            ZeroStatus::ProvedZero
        } else {
            ZeroStatus::Unknown
        },
        method: "fundamental_solution(helmholtz)".into(),
        detail: format!(
            "(Δ∥ + κ²) g = δ for the outgoing {}-D kernel g = {}",
            n - 1,
            pool.display(table.kernel)
        ),
    });
    notes.push(
        "the table's checks of the outgoing kernel (the homogeneous equation by substitution, \
         the jump / the DLMF 10.7.7 leading term) are identities in the wavenumber valid for \
         every κ ≠ 0 with Im κ ≥ 0, so they hold for κₙ = √(k² − λₙ) on the principal branch, \
         evanescent modes included"
            .into(),
    );
    // g(κ) and g with κ → iγ (evanescent), compared with the modified
    // Helmholtz kernel in a strip.
    let g_of = |kappa: ExprId| simp(subs1(table.kernel, kap, kappa, pool), pool);
    if n == 2 {
        let gam = fresh("gamma", Domain::Positive, &avoid, pool);
        let modk = fundamental_solution(
            GreenOperator::ModifiedHelmholtz { k: gam },
            &long_vars,
            None,
            pool,
        )?;
        let ev = sub(
            g_of(pool.mul(vec![pool.imaginary_unit(), gam])),
            modk.kernel,
            pool,
        );
        let mut ck = check(
            "radiation",
            ev,
            "substitution κ = iγ",
            "an evanescent mode (κ = i√(λₙ − k²)) is the decaying modified-Helmholtz kernel \
             −e^{−γ|X|}/(2γ)",
            pool,
        );
        if ck.status != ZeroStatus::ProvedZero {
            ck.status = zstatus(super::expand_func_args(ev, pool), pool);
        }
        if modk.status != VerificationStatus::ExactlyVerified {
            ck.status = ZeroStatus::Unknown;
        }
        checks.push(ck);
        // Outgoing: ∂_X g = iκ g for X > 0.
        let x0 = long_vars[0];
        let gp = branch_abs(table.smooth, x0, 1, pool);
        checks.push(check(
            "radiation",
            sub(
                super::dd(gp, x0, pool)?,
                pool.mul(vec![pool.imaginary_unit(), kap, gp]),
                pool,
            ),
            "substitution",
            "each propagating mode is outgoing: ∂_X g = iκ g for X > 0",
            pool,
        ));
    } else {
        notes.push(
            "each propagating mode is the outgoing Hankel wave H₀⁽¹⁾(κₙρ) ~ √(2/(πκₙρ)) \
             e^{i(κₙρ − π/4)} (DLMF 10.2.5); with κₙ = iγₙ, −(i/4)H₀⁽¹⁾(iγₙρ) = −K₀(γₙρ)/(2π) \
             (DLMF 10.27.8) decays"
                .into(),
        );
    }
    let mut terms: Vec<ExprId> = Vec::new();
    let mut side: Vec<ExprId> = sys.side_conditions.clone();
    side.push(pool.pred_gt(k, pool.integer(0)));
    side.push(pool.pred_gt(eta, pool.integer(0)));
    side.push(pool.pred_lt(eta, width));
    side.push(pool.pred_gt(width, pool.integer(0)));
    let numeric = is_numeric(k, pool) && is_numeric(width, pool);
    let k2_ball = IntervalEval::new(128).eval(sq(k, pool), pool);
    // One term φ(y)φ(η)/N · g(√(k² − λ), X) with its PDE check.
    let term = |phi: ExprId,
                lam: ExprId,
                norm: ExprId,
                label: &str,
                checks: &mut Vec<GreenCheck>|
     -> Result<ExprId, PdeError> {
        let kappa = sqrt(sub(sq(k, pool), lam, pool), pool);
        let phi_eta = subs1(phi, y, eta, pool);
        let body = pool.mul(vec![phi, phi_eta, g_of(kappa)]);
        // (Δ + k²) T = 0 off the source line (smooth part, X > 0 branch).
        let gs = simp(subs1(table.smooth, kap, kappa, pool), pool);
        let gs = if n == 2 {
            branch_abs(gs, long_vars[0], 1, pool)
        } else {
            gs
        };
        let t_mode = pool.mul(vec![phi, phi_eta, gs]);
        let res = GreenOperator::Helmholtz { k }.apply(t_mode, space, None, pool)?;
        let mut ck = check(
            "term_pde",
            res,
            "substitution",
            format!("{label}: (Δ + k²) T = 0 off the source line (κ² = k² − λ)"),
            pool,
        );
        if ck.status != ZeroStatus::ProvedZero {
            ck.status = zstatus(super::expand_powers_pos(simp(res, pool), pool), pool);
        }
        checks.push(ck);
        Ok(div(body, norm, pool))
    };
    for (mi, mode) in sys.modes.iter().enumerate() {
        if mode.condition.is_some() {
            continue;
        }
        if mode.verification != VerificationStatus::ExactlyVerified {
            return Err(PdeError::GreenSeriesUnsupported(format!(
                "the transverse mode λ = {} is not exactly verified",
                pool.display(mode.eigenvalue)
            )));
        }
        let norm = mode.norm_squared.ok_or_else(|| {
            PdeError::GreenSeriesUnsupported("a transverse squared norm did not close".into())
        })?;
        let lam = simp(mode.eigenvalue, pool);
        if numeric && zstatus(sub(sq(k, pool), lam, pool), pool) == ZeroStatus::ProvedZero {
            return Err(PdeError::GreenResonance(format!(
                "k² = {} is the cutoff of the transverse mode {mi}",
                pool.display(lam)
            )));
        }
        terms.push(term(
            mode.eigenfunction,
            lam,
            norm,
            &format!("mode {mi}"),
            &mut checks,
        )?);
        checks.push(GreenCheck {
            what: "term_bc",
            status: ZeroStatus::ProvedZero,
            method: "M3 eigenpair verification".into(),
            detail: format!("mode {mi}: both wall conditions"),
        });
    }
    for (fi, fam) in sys.families.iter().enumerate() {
        if fam.verification.status != VerificationStatus::ExactlyVerified || !fam.closed_form {
            return Err(PdeError::GreenSeriesUnsupported(format!(
                "transverse family {fi} is not exactly verified in closed form"
            )));
        }
        let norm = fam.norms_squared[0].ok_or_else(|| {
            PdeError::GreenSeriesUnsupported("a transverse squared norm did not close".into())
        })?;
        // Term check identically in the frequency ν.
        let _ = term(
            fam.eigenfunctions_of_frequency[0],
            fam.eigenvalue_of_frequency,
            pool.integer(1),
            &format!("family {fi}"),
            &mut checks,
        )?;
        let body = term(
            fam.eigenfunctions[0],
            fam.eigenvalue,
            norm,
            &format!("family {fi} (member)"),
            &mut Vec::new(),
        )?;
        terms.push(sum(
            simp(body, pool),
            fam.index,
            pool.integer(fam.first),
            pool.pos_infinity(),
            pool,
        )?);
        let bc_ok = fam.verification.left == ZeroStatus::ProvedZero
            && fam.verification.right == ZeroStatus::ProvedZero;
        checks.push(GreenCheck {
            what: "term_bc",
            status: if bc_ok {
                ZeroStatus::ProvedZero
            } else {
                ZeroStatus::Unknown
            },
            method: format!("M3 eigenpair verification ({})", fam.verification.method),
            detail: format!("family {fi}: every term meets both wall conditions"),
        });
        // Cutoffs: k² = λₙ for some n.
        let mut decided = false;
        if let (true, Some(k2)) = (numeric, k2_ball.as_ref()) {
            let khi = k2.hi().to_f64().max(0.0).sqrt();
            let count = ((khi * width_f64(width, pool).unwrap_or(f64::NAN) / std::f64::consts::PI)
                .ceil()
                .max(0.0) as usize)
                + 3;
            if count < 100_000 {
                if let Ok(encl) = fam.enclosures(count, &Default::default(), 64, pool) {
                    let mut all_apart = true;
                    for e in &encl {
                        let overlap = !(e.hi < k2.lo() || e.lo > k2.hi());
                        if overlap {
                            let lam = fam.eigenvalue_at(e.index, pool)?;
                            match zstatus(sub(sq(k, pool), lam, pool), pool) {
                                ZeroStatus::ProvedZero => {
                                    return Err(PdeError::GreenResonance(format!(
                                        "k² = λ_{} = {} is a cutoff of the waveguide",
                                        e.index,
                                        pool.display(lam)
                                    )))
                                }
                                ZeroStatus::ProvedNonzero => {}
                                _ => all_apart = false,
                            }
                        }
                    }
                    // Enclosures are increasing: the last one above k² covers
                    // every later eigenvalue.
                    let last_above = encl.last().map(|e| e.lo > k2.hi()).unwrap_or(false);
                    decided = all_apart && last_above;
                }
            }
        }
        if decided {
            notes.push(format!(
                "family {fi}: k² is certified to differ from every transverse eigenvalue \
                 (interval enclosures, and an exact test where they overlap)"
            ));
        } else {
            side.push(pool.pred_ne(sq(k, pool), fam.eigenvalue));
            notes.push(format!(
                "family {fi}: k² must not be a transverse eigenvalue λₙ (a cutoff, where the \
                 waveguide has no Green's function)"
            ));
        }
    }
    if terms.is_empty() {
        return Err(PdeError::GreenSeriesUnsupported(
            "the transverse problem has no modes".into(),
        ));
    }
    refuse_refuted("the Helmholtz waveguide Green's function", &checks)?;
    let status = combine(&checks);
    let value = simp(shift(pool.add(terms), &long_vars, &long_src, pool), pool);
    notes.push(
        "mode series: the transverse eigenfunctions are complete in L²(0, L) (Sturm–Liouville \
         theory), so Σ φₙ(y)φₙ(η)/‖φₙ‖² = δ(y − η) and each longitudinal kernel carries the \
         unit source of its mode; the series converges absolutely and locally uniformly with \
         all derivatives for x∥ ≠ ξ∥ (the evanescent terms decay like e^{−√λₙ |x∥ − ξ∥|})"
            .into(),
    );
    notes.push(
        "radiation condition: κₙ = √(k² − λₙ) on the principal branch, so the finitely many \
         propagating modes (λₙ < k²) are outgoing for the time factor e^{−iωt} and the others \
         decay; this selects the unique Green's function bounded at infinity with outgoing \
         propagating modes"
            .into(),
    );
    Ok(GreenFunction {
        operator: "helmholtz".into(),
        method: "mode_series".into(),
        space: space.to_vec(),
        source: source.to_vec(),
        time: None,
        value,
        series: None,
        branches: None,
        poisson_kernel: None,
        boundary_vars: Vec::new(),
        checks,
        status,
        side_conditions: side,
        notes,
        references: vec![
            "Morse & Feshbach, Methods of Theoretical Physics §7.2 (Green's functions of \
             waveguides by mode expansion)"
                .into(),
            "Duffy, Green's Functions with Applications §6.3 (Helmholtz equation in a strip)"
                .into(),
            "Jackson, Classical Electrodynamics §8.3–8.4 (cutoff, propagating and evanescent \
             modes)"
                .into(),
        ],
    })
}

fn width_f64(width: ExprId, pool: &ExprPool) -> Option<f64> {
    IntervalEval::new(64)
        .eval(width, pool)
        .map(|b| b.hi().to_f64())
}
