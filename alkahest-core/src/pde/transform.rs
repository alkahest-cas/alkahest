//! Transforms of unknowns and their derivatives (plan §5 gap list): what lets
//! `laplace_transform` and `fourier_transform` act on a PDE.
//!
//! The transformed unknown is another undefined function of the same
//! arguments with the transform variable replaced — `U(x, s)` for the
//! Laplace transform of `u(x, t)` in `t`, `u_hat(ξ, t)` for the Fourier
//! transform of `u(x, t)` in `x` (names from [`transformed_name`]) — so the
//! transformed equation is again an ordinary expression with unknowns that
//! `dsolve`, the jet layer and verification understand.
//!
//! * Laplace, `t ↦ s` (one-sided):
//!   `L{∂ₓᵒ ∂ₜᵐ u}(x, s) = ∂ₓᵒ [ sᵐ U − Σ_{i<m} s^{m−1−i} ∂ₜⁱu(x, 0) ]`,
//!   the initial-value terms written as derivatives of `u(x, 0)`.
//! * Fourier, `x ↦ ξ` (the module's convention `F(ξ) = ∫ f e^{−2πiξx} dx`):
//!   `F{∂ₓᵐ ∂ₜᵒ u}(ξ, t) = (2πiξ)ᵐ ∂ₜᵒ û(ξ, t)`, for `u` decaying at `±∞`.
//!
//! The transform variable must occupy exactly one argument slot as a bare
//! symbol (`u(x, t)`, not `u(x, 2t)` or `u(t²)`); otherwise `E-PDE-040`.
//! Both rules assume what every transform method assumes of the unknown —
//! exponential order in `t`, decay in `x` — and that is not checkable here;
//! the solver milestones verify their final answers by substitution.

use std::collections::HashMap;

use crate::diff::undefined::{build, view};
use crate::kernel::{ExprId, ExprPool};

use super::error::PdeError;

/// Which transform a transformed unknown comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransformKind {
    /// One-sided Laplace transform.
    Laplace,
    /// Fourier transform.
    Fourier,
}

/// The name of the transformed unknown: Laplace — the name with its first
/// letter upper-cased (`u → U`, `phi → Phi`), or `{name}_L` when it already
/// starts with an upper-case letter; Fourier — `{name}_hat`.
pub fn transformed_name(name: &str, kind: TransformKind) -> String {
    match kind {
        TransformKind::Laplace => {
            let mut c = name.chars();
            match c.next() {
                Some(f) if f.is_lowercase() => f.to_uppercase().chain(c).collect(),
                _ => format!("{name}_L"),
            }
        }
        TransformKind::Fourier => format!("{name}_hat"),
    }
}

/// The slot of `var` among `args`, when it is exactly one bare slot and no
/// other argument mentions it.
fn slot_of(args: &[ExprId], var: ExprId, pool: &ExprPool) -> Result<usize, ()> {
    let mut slot = None;
    for (j, &a) in args.iter().enumerate() {
        if a == var {
            if slot.is_some() {
                return Err(());
            }
            slot = Some(j);
        } else if crate::kernel::subs::mentions_var(a, var, pool) {
            return Err(());
        }
    }
    slot.ok_or(())
}

/// `L{f}(s)` when `f` is an unknown or a derivative of one depending on `t`;
/// `None` for any other node (the table transforms handle those).
pub(crate) fn laplace_unknown(
    f: ExprId,
    t: ExprId,
    s: ExprId,
    pool: &ExprPool,
) -> Option<Result<ExprId, PdeError>> {
    let v = view(f, pool)?;
    if !v
        .args
        .iter()
        .any(|&a| crate::kernel::subs::mentions_var(a, t, pool))
    {
        return None;
    }
    let Ok(j) = slot_of(&v.args, t, pool) else {
        return Some(Err(PdeError::TransformUnknown(format!(
            "{}: the transform variable {} is not a single bare argument",
            pool.display(f),
            pool.display(t)
        ))));
    };
    let m = v.orders[j];
    let mut big_args = v.args.clone();
    big_args[j] = s;
    let big = pool.func(transformed_name(&v.name, TransformKind::Laplace), big_args);
    let mut other = v.orders.clone();
    other[j] = 0;
    let mut terms = vec![pool.mul(vec![pool.pow(s, pool.integer(m)), build(big, &other, pool)])];
    let mut at0 = HashMap::new();
    at0.insert(t, pool.integer(0));
    for i in 0..m {
        let mut ord = other.clone();
        ord[j] = i;
        let node = crate::kernel::subs::subs(build(v.inner, &ord, pool), &at0, pool);
        let p = (m - 1 - i) as i64;
        terms.push(pool.mul(vec![pool.integer(-1), pool.pow(s, pool.integer(p)), node]));
    }
    let out = if terms.len() == 1 {
        terms[0]
    } else {
        pool.add(terms)
    };
    Some(Ok(out))
}

/// `F{f}(ξ)` when `f` is an unknown or a derivative of one depending on `x`;
/// `None` otherwise.
pub(crate) fn fourier_unknown(
    f: ExprId,
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
) -> Option<Result<ExprId, PdeError>> {
    let v = view(f, pool)?;
    if !v
        .args
        .iter()
        .any(|&a| crate::kernel::subs::mentions_var(a, x, pool))
    {
        return None;
    }
    let Ok(j) = slot_of(&v.args, x, pool) else {
        return Some(Err(PdeError::TransformUnknown(format!(
            "{}: the transform variable {} is not a single bare argument",
            pool.display(f),
            pool.display(x)
        ))));
    };
    let m = v.orders[j];
    let mut hat_args = v.args.clone();
    hat_args[j] = xi;
    let hat = pool.func(transformed_name(&v.name, TransformKind::Fourier), hat_args);
    let mut other = v.orders.clone();
    other[j] = 0;
    let body = build(hat, &other, pool);
    if m == 0 {
        return Some(Ok(body));
    }
    let two_pi_i_xi = pool.mul(vec![
        pool.integer(2),
        pool.symbol("pi", crate::kernel::Domain::Real),
        pool.imaginary_unit(),
        xi,
    ]);
    Some(Ok(
        pool.mul(vec![pool.pow(two_pi_i_xi, pool.integer(m)), body])
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn laplace_of_heat_equation() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let s = pool.symbol("s", Domain::Real);
        let u = pool.func("u", vec![x, t]);
        let d = |e, v| crate::diff::diff(e, v, &pool).unwrap().value;
        let heat = pool.add(vec![
            d(u, t),
            pool.mul(vec![pool.integer(-1), d(d(u, x), x)]),
        ]);
        let lt = crate::transform::laplace_transform(heat, t, s, &pool).unwrap();
        let shown = pool.display(lt).to_string();
        assert!(shown.contains("U(x, s)"), "{shown}");
        assert!(shown.contains("u(x, 0)"), "{shown}");
        assert!(shown.contains("Derivative(U(x, s), x, x)"), "{shown}");
        // second order in t: s² U − s u(x,0) − u_t(x,0)
        let utt = d(d(u, t), t);
        let l2 = crate::transform::laplace_transform(utt, t, s, &pool).unwrap();
        let shown = pool.display(l2).to_string();
        assert!(
            shown.contains("Subs(Derivative") || shown.contains("Derivative(u(x, 0)"),
            "{shown}"
        );
    }

    #[test]
    fn fourier_of_derivatives() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let xi = pool.symbol("xi", Domain::Real);
        let u = pool.func("u", vec![x, t]);
        let d = |e, v| crate::diff::diff(e, v, &pool).unwrap().value;
        let e = pool.add(vec![
            d(u, t),
            pool.mul(vec![pool.integer(-3), d(d(u, x), x)]),
        ]);
        let ft = crate::transform::fourier_transform(e, x, xi, &pool).unwrap();
        let shown = pool.display(ft).to_string();
        assert!(shown.contains("u_hat(xi, t)"), "{shown}");
        assert!(shown.contains("Derivative(u_hat(xi, t), t)"), "{shown}");
        // a non-slot occurrence is refused, not mis-transformed
        let bad = pool.func("u", vec![pool.mul(vec![pool.integer(2), x]), t]);
        assert!(crate::transform::fourier_transform(bad, x, xi, &pool).is_err());
    }
}
