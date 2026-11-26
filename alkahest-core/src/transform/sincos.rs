//! Fourier **sine** and **cosine** transforms on the half-line (PDE plan M3).
//!
//! ```text
//! F_s{f}(ω) = ∫₀^∞ f(x) sin(ωx) dx,     f(x) = (2/π) ∫₀^∞ F_s(ω) sin(ωx) dω
//! F_c{f}(ω) = ∫₀^∞ f(x) cos(ωx) dx,     f(x) = (2/π) ∫₀^∞ F_c(ω) cos(ωx) dω
//! ```
//!
//! They turn `∂ₓ²` on `x > 0` into multiplication with the boundary value
//! the method needs — the sine transform for a Dirichlet end, the cosine
//! transform for a Neumann end:
//!
//! ```text
//! F_s{f''}(ω) = −ω² F_s{f}(ω) + ω f(0)
//! F_c{f''}(ω) = −ω² F_c{f}(ω) − f'(0)
//! ```
//!
//! Both rules apply to an unknown `u(x, t)` and its derivatives (the
//! transformed unknown is `u_hat_s(ω, t)` / `u_hat_c(ω, t)`, the boundary
//! values `u(0, t)` / `Derivative(u(0, t), 1, 0)`), so a PDE on the
//! half-line transforms term by term.  Each inverse is the forward transform
//! times `2/π` (the transforms are self-reciprocal), so one table serves both
//! directions.
//!
//! The table is small and every entry is classical (Erdélyi, *Tables of
//! Integral Transforms* I, §1–2): `e^{−ax}`, `x e^{−ax}`, `e^{−ax}/x`,
//! `x/(x² + a²)`, `1/(x² + a²)`, `e^{−ax²}`, `x e^{−ax²}`, `1/x`, with
//! linearity and constant factors.  The rate `a` must be positive — a literal
//! non-positive rate refutes the entry (`E-TRANSFORM-023`), a symbolic one is
//! accepted only when its sign is established (a `Positive` symbol, a positive
//! number); otherwise the entry does not match (`E-TRANSFORM-021`).

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

/// Which half-line transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SinCosKind {
    /// The Fourier sine transform.
    Sine,
    /// The Fourier cosine transform.
    Cosine,
}

/// Errors from the sine / cosine transforms.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SinCosError {
    /// No table entry matched (`E-TRANSFORM-021`).
    NoRule(String),
    /// The space and frequency variables coincide (`E-TRANSFORM-022`).
    SameVariable,
    /// A rate is known not to be positive, so the integral diverges
    /// (`E-TRANSFORM-023`).
    Divergent(String),
    /// An unknown occurs other than as `u(…, x, …)` with `x` a bare slot,
    /// or with an odd number of `x`-derivatives (`E-TRANSFORM-024`).
    Unknown(String),
}

impl std::fmt::Display for SinCosError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SinCosError::NoRule(m) => write!(f, "sine/cosine transform: no rule for {m}"),
            SinCosError::SameVariable => {
                write!(f, "sine/cosine transform: the variables must be distinct")
            }
            SinCosError::Divergent(m) => write!(f, "sine/cosine transform: {m}"),
            SinCosError::Unknown(m) => write!(f, "sine/cosine transform of an unknown: {m}"),
        }
    }
}

impl std::error::Error for SinCosError {}

impl crate::errors::AlkahestError for SinCosError {
    fn code(&self) -> &'static str {
        match self {
            SinCosError::NoRule(_) => "E-TRANSFORM-021",
            SinCosError::SameVariable => "E-TRANSFORM-022",
            SinCosError::Divergent(_) => "E-TRANSFORM-023",
            SinCosError::Unknown(_) => "E-TRANSFORM-024",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        crate::errors::codes::REGISTRY
            .iter()
            .find(|s| s.code == self.code())
            .and_then(|s| s.remediation)
    }
}

fn pi(pool: &ExprPool) -> ExprId {
    pool.symbol("pi", Domain::Real)
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

fn recip(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.integer(-1))
}

fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::engine::simplify(e, pool).value
}

fn mentions(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    crate::kernel::subs::mentions_var(e, v, pool)
}

/// `Pos`/`NonPos`/`Unknown` for a rate.
fn rate_sign(a: ExprId, pool: &ExprPool) -> Option<bool> {
    match crate::pde::sturm_liouville::sign(a, pool) {
        crate::pde::sturm_liouville::Sgn::Pos => Some(true),
        crate::pde::sturm_liouville::Sgn::Neg | crate::pde::sturm_liouville::Sgn::Zero => {
            Some(false)
        }
        crate::pde::sturm_liouville::Sgn::Unknown => None,
    }
}

/// The forward sine (`Sine`) or cosine (`Cosine`) transform of `f(x)`.
///
/// # Errors
///
/// [`SinCosError`] as documented on its variants.
pub fn sincos_transform(
    kind: SinCosKind,
    f: ExprId,
    x: ExprId,
    w: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SinCosError> {
    if x == w {
        return Err(SinCosError::SameVariable);
    }
    Ok(simp(forward(kind, simp(f, pool), x, w, pool)?, pool))
}

/// `F_s{f}(ω) = ∫₀^∞ f(x) sin(ωx) dx`.
///
/// # Errors
///
/// See [`SinCosError`].
pub fn fourier_sine_transform(
    f: ExprId,
    x: ExprId,
    w: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SinCosError> {
    sincos_transform(SinCosKind::Sine, f, x, w, pool)
}

/// `F_c{f}(ω) = ∫₀^∞ f(x) cos(ωx) dx`.
///
/// # Errors
///
/// See [`SinCosError`].
pub fn fourier_cosine_transform(
    f: ExprId,
    x: ExprId,
    w: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SinCosError> {
    sincos_transform(SinCosKind::Cosine, f, x, w, pool)
}

/// `f(x) = (2/π) ∫₀^∞ F(ω) sin(ωx) dω`.
///
/// # Errors
///
/// See [`SinCosError`].
pub fn inverse_fourier_sine_transform(
    big_f: ExprId,
    w: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SinCosError> {
    let g = sincos_transform(SinCosKind::Sine, big_f, w, x, pool)?;
    Ok(simp(
        pool.mul(vec![pool.integer(2), recip(pi(pool), pool), g]),
        pool,
    ))
}

/// `f(x) = (2/π) ∫₀^∞ F(ω) cos(ωx) dω`.
///
/// # Errors
///
/// See [`SinCosError`].
pub fn inverse_fourier_cosine_transform(
    big_f: ExprId,
    w: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SinCosError> {
    let g = sincos_transform(SinCosKind::Cosine, big_f, w, x, pool)?;
    Ok(simp(
        pool.mul(vec![pool.integer(2), recip(pi(pool), pool), g]),
        pool,
    ))
}

/// The name of the transformed unknown: `{name}_hat_s` / `{name}_hat_c`.
pub fn sincos_transformed_name(name: &str, kind: SinCosKind) -> String {
    match kind {
        SinCosKind::Sine => format!("{name}_hat_s"),
        SinCosKind::Cosine => format!("{name}_hat_c"),
    }
}

fn forward(
    kind: SinCosKind,
    f: ExprId,
    x: ExprId,
    w: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SinCosError> {
    if let Some(r) = unknown_rule(kind, f, x, w, pool) {
        return r;
    }
    match pool.get(f) {
        ExprData::Add(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for t in xs {
                out.push(forward(kind, t, x, w, pool)?);
            }
            return Ok(pool.add(out));
        }
        ExprData::Mul(xs) => {
            let (consts, rest): (Vec<ExprId>, Vec<ExprId>) =
                xs.iter().partition(|&&t| !mentions(t, x, pool));
            if !consts.is_empty() && !rest.is_empty() {
                let body = if rest.len() == 1 {
                    rest[0]
                } else {
                    pool.mul(rest)
                };
                let mut out = consts;
                out.push(forward(kind, body, x, w, pool)?);
                return Ok(pool.mul(out));
            }
        }
        _ => {}
    }
    table(kind, f, x, w, pool)
}

/// `f = x^p · e^{E} · (x² + b)^q` with the pieces classified.
struct Shape {
    p: i64,
    /// Linear rate `a` of `e^{−ax}`.
    lin: Option<ExprId>,
    /// Quadratic rate `a` of `e^{−ax²}`.
    quad: Option<ExprId>,
    /// `b` of `(x² + b)^{-1}`.
    lorentz: Option<ExprId>,
}

fn shape(f: ExprId, x: ExprId, pool: &ExprPool) -> Option<Shape> {
    let factors = match pool.get(f) {
        ExprData::Mul(xs) => xs,
        _ => vec![f],
    };
    let mut s = Shape {
        p: 0,
        lin: None,
        quad: None,
        lorentz: None,
    };
    for fa in factors {
        if fa == x {
            s.p += 1;
            continue;
        }
        match pool.get(fa) {
            ExprData::Pow { base, exp } if base == x => {
                let k = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                })?;
                s.p += k;
            }
            ExprData::Pow { base, exp } if exp == pool.integer(-1) => {
                let b = simp(
                    pool.add(vec![base, neg(pool.pow(x, pool.integer(2)), pool)]),
                    pool,
                );
                if mentions(b, x, pool) || s.lorentz.is_some() {
                    return None;
                }
                s.lorentz = Some(b);
            }
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                let arg = args[0];
                let at0 = simp(
                    crate::pde::sturm_liouville::subs1(arg, x, pool.integer(0), pool),
                    pool,
                );
                if at0 != pool.integer(0) {
                    return None;
                }
                let lin = simp(neg(pool.mul(vec![arg, recip(x, pool)]), pool), pool);
                if !mentions(lin, x, pool) {
                    if s.lin.is_some() {
                        return None;
                    }
                    s.lin = Some(lin);
                    continue;
                }
                let quad = simp(
                    neg(pool.mul(vec![arg, pool.pow(x, pool.integer(-2))]), pool),
                    pool,
                );
                if mentions(quad, x, pool) || s.quad.is_some() {
                    return None;
                }
                s.quad = Some(quad);
            }
            _ => return None,
        }
    }
    Some(s)
}

fn positive(a: ExprId, what: &str, pool: &ExprPool) -> Result<(), SinCosError> {
    match rate_sign(a, pool) {
        Some(true) => Ok(()),
        Some(false) => Err(SinCosError::Divergent(format!(
            "the {what} {} is not positive",
            pool.display(a)
        ))),
        None => Err(SinCosError::NoRule(format!(
            "the {what} {} is not known to be positive",
            pool.display(a)
        ))),
    }
}

fn table(
    kind: SinCosKind,
    f: ExprId,
    x: ExprId,
    w: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SinCosError> {
    let no = || SinCosError::NoRule(pool.display(f).to_string());
    let s = shape(f, x, pool).ok_or_else(no)?;
    let w2 = pool.pow(w, pool.integer(2));
    let half = pool.rational(1, 2);
    use SinCosKind::*;
    let out = match (kind, s.p, s.lin, s.quad, s.lorentz) {
        // e^{−ax}
        (_, 0, Some(a), None, None) => {
            positive(a, "rate", pool)?;
            let den = recip(pool.add(vec![pool.pow(a, pool.integer(2)), w2]), pool);
            match kind {
                Sine => pool.mul(vec![w, den]),
                _ => pool.mul(vec![a, den]),
            }
        }
        // x e^{−ax}
        (_, 1, Some(a), None, None) => {
            positive(a, "rate", pool)?;
            let a2 = pool.pow(a, pool.integer(2));
            let den = pool.pow(pool.add(vec![a2, w2]), pool.integer(-2));
            match kind {
                Sine => pool.mul(vec![pool.integer(2), a, w, den]),
                _ => pool.mul(vec![pool.add(vec![a2, neg(w2, pool)]), den]),
            }
        }
        // e^{−ax}/x (sine only)
        (Sine, -1, Some(a), None, None) => {
            positive(a, "rate", pool)?;
            pool.func("atan", vec![pool.mul(vec![w, recip(a, pool)])])
        }
        // 1/x (sine only), ω > 0
        (Sine, -1, None, None, None) => pool.mul(vec![pi(pool), half]),
        // x/(x² + a²) (sine), 1/(x² + a²) (cosine); ω > 0
        (Sine, 1, None, None, Some(b)) | (Cosine, 0, None, None, Some(b)) => {
            positive(b, "constant a² of x² + a²", pool)?;
            let a = simp(pool.func("sqrt", vec![b]), pool);
            let decay = pool.func("exp", vec![neg(pool.mul(vec![a, w]), pool)]);
            match kind {
                Sine => pool.mul(vec![pi(pool), half, decay]),
                _ => pool.mul(vec![pi(pool), half, recip(a, pool), decay]),
            }
        }
        // e^{−ax²} (cosine), x e^{−ax²} (sine)
        (Cosine, 0, None, Some(a), None) | (Sine, 1, None, Some(a), None) => {
            positive(a, "Gaussian rate", pool)?;
            let g = pool.func(
                "exp",
                vec![neg(
                    pool.mul(vec![w2, recip(pool.mul(vec![pool.integer(4), a]), pool)]),
                    pool,
                )],
            );
            let sqrt_pi_a = pool.func("sqrt", vec![pool.mul(vec![pi(pool), recip(a, pool)])]);
            match kind {
                Cosine => pool.mul(vec![half, sqrt_pi_a, g]),
                _ => pool.mul(vec![
                    w,
                    recip(pool.mul(vec![pool.integer(4), a]), pool),
                    sqrt_pi_a,
                    g,
                ]),
            }
        }
        _ => return Err(no()),
    };
    Ok(out)
}

/// The transform of an unknown `u(…, x, …)` or a derivative of it.
fn unknown_rule(
    kind: SinCosKind,
    f: ExprId,
    x: ExprId,
    w: ExprId,
    pool: &ExprPool,
) -> Option<Result<ExprId, SinCosError>> {
    let v = crate::diff::undefined::view(f, pool)?;
    if !v.args.iter().any(|&a| mentions(a, x, pool)) {
        return None;
    }
    let slots: Vec<usize> = (0..v.args.len()).filter(|&j| v.args[j] == x).collect();
    let others = v
        .args
        .iter()
        .enumerate()
        .any(|(j, &a)| !slots.contains(&j) && mentions(a, x, pool));
    if slots.len() != 1 || others {
        return Some(Err(SinCosError::Unknown(format!(
            "{}: {} is not a single bare argument",
            pool.display(f),
            pool.display(x)
        ))));
    }
    let j = slots[0];
    let m = v.orders[j];
    if m % 2 == 1 {
        return Some(Err(SinCosError::Unknown(format!(
            "{}: an odd x-derivative maps the sine transform to the cosine one",
            pool.display(f)
        ))));
    }
    let mut hat_args = v.args.clone();
    hat_args[j] = w;
    let hat = pool.func(sincos_transformed_name(&v.name, kind), hat_args);
    let mut base = v.orders.clone();
    base[j] = 0;
    // F{f^{(2m)}} = −ω² F{f^{(2m−2)}} + (ω f^{(2m−2)}(0)  |  −f^{(2m−1)}(0))
    let mut acc = crate::diff::undefined::build(hat, &base, pool);
    let mut at0 = std::collections::HashMap::new();
    at0.insert(x, pool.integer(0));
    let mut k = 0;
    while k < m {
        let mut ord = base.clone();
        let boundary = match kind {
            SinCosKind::Sine => {
                ord[j] = k;
                pool.mul(vec![
                    w,
                    crate::kernel::subs::subs(
                        crate::diff::undefined::build(v.inner, &ord, pool),
                        &at0,
                        pool,
                    ),
                ])
            }
            SinCosKind::Cosine => {
                ord[j] = k + 1;
                neg(
                    crate::kernel::subs::subs(
                        crate::diff::undefined::build(v.inner, &ord, pool),
                        &at0,
                        pool,
                    ),
                    pool,
                )
            }
        };
        acc = pool.add(vec![
            neg(pool.mul(vec![pool.pow(w, pool.integer(2)), acc]), pool),
            boundary,
        ]);
        k += 2;
    }
    Some(Ok(acc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;

    #[test]
    fn exponential_pairs_and_inverses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let w = pool.symbol("w", Domain::Positive);
        let a = pool.symbol("a", Domain::Positive);
        let f = pool.func("exp", vec![neg(pool.mul(vec![a, x]), &pool)]);
        let s = fourier_sine_transform(f, x, w, &pool).unwrap();
        let c = fourier_cosine_transform(f, x, w, &pool).unwrap();
        let want_s = pool.mul(vec![
            w,
            recip(
                pool.add(vec![
                    pool.pow(a, pool.integer(2)),
                    pool.pow(w, pool.integer(2)),
                ]),
                &pool,
            ),
        ]);
        assert_eq!(simp(want_s, &pool), s);
        // inverse cosine of a/(a²+ω²) is e^{−ax}
        let back = inverse_fourier_cosine_transform(c, w, x, &pool).unwrap();
        let z = crate::simplify::zero::zero_test(pool.add(vec![back, neg(f, &pool)]), &pool, None);
        assert_eq!(
            z.status,
            crate::simplify::zero::ZeroStatus::ProvedZero,
            "{}",
            pool.display(back)
        );
        let back = inverse_fourier_sine_transform(s, w, x, &pool).unwrap();
        let z = crate::simplify::zero::zero_test(pool.add(vec![back, neg(f, &pool)]), &pool, None);
        assert_eq!(
            z.status,
            crate::simplify::zero::ZeroStatus::ProvedZero,
            "{}",
            pool.display(back)
        );
    }

    #[test]
    fn gaussian_and_refusals() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let w = pool.symbol("w", Domain::Positive);
        let g = pool.func("exp", vec![neg(pool.pow(x, pool.integer(2)), &pool)]);
        let c = fourier_cosine_transform(g, x, w, &pool).unwrap();
        assert!(pool.display(c).to_string().contains("exp"));
        let bad = pool.func("exp", vec![x]);
        assert_eq!(
            fourier_sine_transform(bad, x, w, &pool).unwrap_err().code(),
            "E-TRANSFORM-023"
        );
        let other = pool.func("sin", vec![x]);
        assert_eq!(
            fourier_sine_transform(other, x, w, &pool)
                .unwrap_err()
                .code(),
            "E-TRANSFORM-021"
        );
        assert_eq!(
            fourier_sine_transform(other, x, x, &pool)
                .unwrap_err()
                .code(),
            "E-TRANSFORM-022"
        );
    }

    #[test]
    fn unknown_second_derivative() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let w = pool.symbol("w", Domain::Positive);
        let u = pool.func("u", vec![x, t]);
        let uxx = crate::diff::diff(crate::diff::diff(u, x, &pool).unwrap().value, x, &pool)
            .unwrap()
            .value;
        let s = fourier_sine_transform(uxx, x, w, &pool).unwrap();
        let shown = pool.display(s).to_string();
        assert!(
            shown.contains("u_hat_s(w, t)") && shown.contains("u(0, t)"),
            "{shown}"
        );
        let c = fourier_cosine_transform(uxx, x, w, &pool).unwrap();
        let shown = pool.display(c).to_string();
        assert!(
            shown.contains("u_hat_c(w, t)") && shown.contains("Derivative"),
            "{shown}"
        );
        let ux = crate::diff::diff(u, x, &pool).unwrap().value;
        assert_eq!(
            fourier_sine_transform(ux, x, w, &pool).unwrap_err().code(),
            "E-TRANSFORM-024"
        );
    }
}
