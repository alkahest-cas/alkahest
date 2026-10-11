//! Classical orthogonal polynomials, spherical harmonics and angular-momentum
//! coupling coefficients (G11a).
//!
//! # Families and conventions
//!
//! Every family is a registered primitive whose **first** argument is the
//! degree `n` and whose **last** is the variable `x`; family parameters sit in
//! between.  Conventions follow DLMF chapter 18 (and §14 for the associated
//! Legendre functions), which is also what SymPy, mpmath and Arb use:
//!
//! | node | DLMF | notes |
//! |---|---|---|
//! | `legendre_p(n, x)` | 18.3 | `P_n(1) = 1` |
//! | `assoc_legendre_p(n, m, x)` | 14.6.1 | **Condon–Shortley phase included**: `P_n^m(x) = (−1)^m (1−x²)^{m/2} dᵐP_n/dxᵐ`; `P_n^{−m} = (−1)^m (n−m)!/(n+m)! P_n^m`; Ferrers function, real on `[−1, 1]` |
//! | `hermite_h(n, x)` | 18.3 | physicists', `H_n = 2ⁿxⁿ + …`, weight `e^{−x²}` |
//! | `hermite_he(n, x)` | 18.3 | probabilists', monic, weight `e^{−x²/2}` |
//! | `laguerre_l(n, x)` | 18.3 | `L_n = L_n^{(0)}` |
//! | `assoc_laguerre_l(n, α, x)` | 18.3 | generalised Laguerre `L_n^{(α)}` |
//! | `chebyshev_t(n, x)`, `chebyshev_u(n, x)` | 18.3 | first and second kind |
//! | `gegenbauer_c(n, a, x)` | 18.3 | ultraspherical `C_n^{(a)}`; `C_n^{(0)} ≡ 0` for `n ≥ 1` |
//! | `jacobi_p(n, a, b, x)` | 18.3 | `P_n^{(a,b)}` |
//!
//! # Explicit versus symbolic degree
//!
//! [`orthopoly`] with an **integer literal** degree (and, for
//! `assoc_legendre_p`, an integer literal order) returns the explicit
//! polynomial with exact rational coefficients, generated from the closed-form
//! sums of DLMF §18.5 — never from floating point.  A family parameter (`α`,
//! `a`, `b`) may be any expression; it then appears polynomially in the
//! coefficients.  Negative integer degrees use the reflections that keep the
//! result a polynomial (`P_{−n−1} = P_n`, `T_{−n} = T_n`,
//! `U_{−n} = −U_{n−2}`); for the other families a negative or non-integer
//! degree has no polynomial form and the call returns the unevaluated node,
//! which still differentiates and evaluates numerically.
//!
//! A symbolic degree (or `expand = false`) keeps the node.  The node
//! differentiates in `x` by the standard identities (DLMF 18.9.15–18.9.20,
//! 14.10.5), evaluates in `f64` and in rigorous ball arithmetic (Arb's
//! `arb_hypgeom_*` family when this build carries FLINT's Arb layer), and
//! simplifies at the special values that hold for every degree.
//! [`expand_orthopoly`] rewrites the nodes whose degree has since become an
//! integer literal (e.g. after `subs`).
//!
//! # Angular momentum
//!
//! [`angular`] has exact Wigner 3j / 6j symbols, Clebsch–Gordan coefficients
//! (Racah's formulas over big integers; the result is `q·√r` with `q`
//! rational and `r` square-free) and the complex spherical harmonics
//! `Y_l^m(θ, φ)`.

pub mod angular;
pub(crate) mod coeffs;
mod error;
pub(crate) mod numeric;
pub(crate) mod primitives;

#[cfg(test)]
mod tests;

pub use angular::{clebsch_gordan, spherical_harmonic_y, wigner_3j, wigner_6j, HalfInt, Surd};
pub use error::SpecialFunctionError;

use rug::{Integer, Rational};

use crate::kernel::{ExprData, ExprId, ExprPool};

use coeffs::{ParamPoly, ParamVal};

/// Largest degree [`orthopoly`] expands when every family parameter is a
/// rational literal.
pub const MAX_EXPAND_DEGREE: i64 = 1000;

/// Largest degree [`orthopoly`] expands when a family parameter is symbolic
/// (the coefficients are then polynomials in that parameter, and a Jacobi
/// polynomial in two symbolic parameters has `O(n²)` terms per coefficient).
pub const MAX_EXPAND_DEGREE_SYMBOLIC: i64 = 64;

/// A family of classical orthogonal polynomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OrthoFamily {
    /// `legendre_p(n, x)`
    Legendre,
    /// `assoc_legendre_p(n, m, x)` — Condon–Shortley phase included.
    AssocLegendre,
    /// `hermite_h(n, x)` — physicists'.
    HermiteH,
    /// `hermite_he(n, x)` — probabilists'.
    HermiteHe,
    /// `laguerre_l(n, x)`
    Laguerre,
    /// `assoc_laguerre_l(n, α, x)`
    AssocLaguerre,
    /// `chebyshev_t(n, x)`
    ChebyshevT,
    /// `chebyshev_u(n, x)`
    ChebyshevU,
    /// `gegenbauer_c(n, a, x)`
    Gegenbauer,
    /// `jacobi_p(n, a, b, x)`
    Jacobi,
}

impl OrthoFamily {
    /// Every family, in a fixed order.
    pub const ALL: [OrthoFamily; 10] = [
        OrthoFamily::Legendre,
        OrthoFamily::AssocLegendre,
        OrthoFamily::HermiteH,
        OrthoFamily::HermiteHe,
        OrthoFamily::Laguerre,
        OrthoFamily::AssocLaguerre,
        OrthoFamily::ChebyshevT,
        OrthoFamily::ChebyshevU,
        OrthoFamily::Gegenbauer,
        OrthoFamily::Jacobi,
    ];

    /// The `Func` node name.
    pub fn name(self) -> &'static str {
        match self {
            OrthoFamily::Legendre => "legendre_p",
            OrthoFamily::AssocLegendre => "assoc_legendre_p",
            OrthoFamily::HermiteH => "hermite_h",
            OrthoFamily::HermiteHe => "hermite_he",
            OrthoFamily::Laguerre => "laguerre_l",
            OrthoFamily::AssocLaguerre => "assoc_laguerre_l",
            OrthoFamily::ChebyshevT => "chebyshev_t",
            OrthoFamily::ChebyshevU => "chebyshev_u",
            OrthoFamily::Gegenbauer => "gegenbauer_c",
            OrthoFamily::Jacobi => "jacobi_p",
        }
    }

    /// The family whose node is called `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        OrthoFamily::ALL.into_iter().find(|f| f.name() == name)
    }

    /// Number of parameters between the degree and the variable.
    pub fn n_params(self) -> usize {
        match self {
            OrthoFamily::AssocLegendre | OrthoFamily::AssocLaguerre | OrthoFamily::Gegenbauer => 1,
            OrthoFamily::Jacobi => 2,
            _ => 0,
        }
    }

    /// Total argument count of the node: degree, parameters, variable.
    pub fn arity(self) -> usize {
        self.n_params() + 2
    }
}

// ---------------------------------------------------------------------------
// Small expression helpers
// ---------------------------------------------------------------------------

pub(crate) fn int_literal(pool: &ExprPool, id: ExprId) -> Option<Integer> {
    pool.with(id, |d| match d {
        ExprData::Integer(n) => Some(n.0.clone()),
        ExprData::Rational(q) if *q.0.denom() == 1 => Some(q.0.numer().clone()),
        _ => None,
    })
}

pub(crate) fn rational_literal(pool: &ExprPool, id: ExprId) -> Option<Rational> {
    pool.with(id, |d| match d {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(q) => Some(q.0.clone()),
        _ => None,
    })
}

pub(crate) fn rat_node(pool: &ExprPool, q: &Rational) -> ExprId {
    if *q.denom() == 1 {
        pool.integer(q.numer().clone())
    } else {
        pool.rational(q.numer().clone(), q.denom().clone())
    }
}

pub(crate) fn mk_add(pool: &ExprPool, mut terms: Vec<ExprId>) -> ExprId {
    match terms.len() {
        0 => pool.integer(0),
        1 => terms.pop().unwrap(),
        _ => pool.add(terms),
    }
}

pub(crate) fn mk_mul(pool: &ExprPool, factors: Vec<ExprId>) -> ExprId {
    let one = pool.integer(1);
    let mut f: Vec<ExprId> = factors.into_iter().filter(|&a| a != one).collect();
    match f.len() {
        0 => one,
        1 => f.pop().unwrap(),
        _ => pool.mul(f),
    }
}

pub(crate) fn pow_int(pool: &ExprPool, base: ExprId, k: u32) -> ExprId {
    match k {
        0 => pool.integer(1),
        1 => base,
        _ => pool.pow(base, pool.integer(k)),
    }
}

fn param_poly_expr(pool: &ExprPool, p: &ParamPoly, params: &[ExprId]) -> ExprId {
    let mut terms = Vec::with_capacity(p.0.len());
    for (&(i, j), c) in &p.0 {
        let mut f = vec![rat_node(pool, c)];
        if i > 0 {
            f.push(pow_int(pool, params[0], i));
        }
        if j > 0 {
            f.push(pow_int(pool, params[1], j));
        }
        terms.push(mk_mul(pool, f));
    }
    mk_add(pool, terms)
}

/// `Σ_k c_k xᵏ`, skipping zero coefficients. `params` are the expressions the
/// symbolic parameters `p₀`, `p₁` of the coefficients stand for.
fn poly_expr(pool: &ExprPool, c: &[ParamPoly], params: &[ExprId], x: ExprId) -> ExprId {
    let mut terms = Vec::new();
    for (k, ck) in c.iter().enumerate() {
        if ck.is_zero() {
            continue;
        }
        let coeff = match ck.as_constant() {
            Some(q) => rat_node(pool, &q),
            None => param_poly_expr(pool, ck, params),
        };
        terms.push(mk_mul(pool, vec![coeff, pow_int(pool, x, k as u32)]));
    }
    mk_add(pool, terms)
}

fn rational_poly_expr(pool: &ExprPool, c: &[Rational], x: ExprId) -> ExprId {
    let pp: Vec<ParamPoly> = c.iter().cloned().map(ParamPoly::constant).collect();
    poly_expr(pool, &pp, &[], x)
}

// ---------------------------------------------------------------------------
// Explicit polynomials
// ---------------------------------------------------------------------------

/// The data of an explicit associated Legendre function:
/// `scale · (1 − x²)^{m/2} · Σ_k poly_k xᵏ` with `m = |order|`.
pub(crate) struct AssocLegendreForm {
    pub(crate) abs_m: u32,
    pub(crate) poly: Vec<Rational>,
}

/// `P_n^m` for integer `n ≥ 0`, `m`, as `(1−x²)^{|m|/2} · Σ poly_k xᵏ`
/// (Condon–Shortley phase folded into `poly`).  `None` when `|m| > n`
/// (the function is identically zero).
pub(crate) fn assoc_legendre_form(
    n: u32,
    m: i64,
) -> Result<Option<AssocLegendreForm>, SpecialFunctionError> {
    let abs_m = m.unsigned_abs();
    if abs_m > n as u64 {
        return Ok(None);
    }
    let abs_m = abs_m as u32;
    let d = coeffs::derivative(&coeffs::legendre(n)?, abs_m);
    // m ≥ 0: (−1)^m (1−x²)^{m/2} P_n^{(m)}.
    // m < 0: P_n^{−|m|} = (−1)^{|m|} (n−|m|)!/(n+|m|)! P_n^{|m|}
    //                  = (n−|m|)!/(n+|m|)! (1−x²)^{|m|/2} P_n^{(|m|)}.
    let scale = if m >= 0 {
        Rational::from(if abs_m % 2 == 1 { -1 } else { 1 })
    } else {
        Rational::from((
            Integer::from(Integer::factorial(n - abs_m)),
            Integer::from(Integer::factorial(n + abs_m)),
        ))
    };
    let poly = d.into_iter().map(|c| c * &scale).collect();
    Ok(Some(AssocLegendreForm { abs_m, poly }))
}

/// The exact value of an explicit associated Legendre function at a rational
/// point `q`: `P(q)·(1−q²)^{m/2}`.  The square root is taken exactly when
/// `1 − q²` is the square of a rational, and otherwise left as `w^{1/2}` with
/// the integer part of the power pulled out — never rounded.
fn assoc_legendre_at_rational(pool: &ExprPool, form: &AssocLegendreForm, q: &Rational) -> ExprId {
    let mut p = Rational::new();
    for c in form.poly.iter().rev() {
        p *= q;
        p += c;
    }
    if form.abs_m == 0 || p == 0 {
        return rat_node(pool, &p);
    }
    let w = Rational::from(1) - Rational::from(q * q);
    let half = form.abs_m / 2;
    let mut coeff = p * rug::ops::Pow::pow(w.clone(), half);
    if form.abs_m % 2 == 0 {
        return rat_node(pool, &coeff);
    }
    // (1 − q²)^{1/2} = √(a·b)/b for w = a/b; the square part of a·b (by
    // trial division, then a perfect-square test of what is left) comes out
    // of the root. w < 0 (|q| > 1) leaves the principal power symbolic.
    if w < 0 {
        return mk_mul(
            pool,
            vec![
                rat_node(pool, &coeff),
                pool.pow(rat_node(pool, &w), pool.rational(1, 2)),
            ],
        );
    }
    let mut k = Integer::from(w.numer() * w.denom());
    coeff /= Rational::from(w.denom().clone());
    let mut out = Integer::from(1);
    let mut p = 2u32;
    while p < 10_000 && Integer::from(p) * p <= k {
        let pp = Integer::from(p) * p;
        while k.is_divisible(&pp) {
            k /= &pp;
            out *= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if k.is_perfect_square() {
        out *= Integer::from(k.sqrt_ref());
        k = Integer::from(1);
    }
    coeff *= Rational::from(out);
    Surd { coeff, radicand: k }.to_expr(pool)
}

/// Map a (possibly negative) integer degree to the non-negative one with the
/// same polynomial, and the sign to apply; `None` when the family has no
/// polynomial at this degree; `Some((_, 0))` when the polynomial is `0`.
fn normalise_degree(family: OrthoFamily, n: &Integer) -> Option<(Integer, i32)> {
    if *n >= 0 {
        return Some((n.clone(), 1));
    }
    match family {
        // P_{−n−1} = P_n for the Legendre functions (DLMF 14.9.5 / 14.9.11).
        OrthoFamily::Legendre | OrthoFamily::AssocLegendre => Some((Integer::from(-n) - 1u32, 1)),
        // T_{−n} = T_n.
        OrthoFamily::ChebyshevT => Some((Integer::from(-n), 1)),
        // U_{−1} = 0, U_{−n} = −U_{n−2}.
        OrthoFamily::ChebyshevU => {
            if *n == -1 {
                Some((Integer::new(), 0))
            } else {
                Some((Integer::from(-n) - 2u32, -1))
            }
        }
        _ => None,
    }
}

/// The explicit form of `family(n, params…, x)`, or `None` when it has none
/// (symbolic or non-integer degree, symbolic or non-integer Legendre order,
/// negative degree of a family without a reflection).
pub(crate) fn explicit(
    family: OrthoFamily,
    n: ExprId,
    params: &[ExprId],
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    if params.len() != family.n_params() {
        return Err(SpecialFunctionError::WrongArity {
            family: family.name(),
            expected: family.n_params(),
            got: params.len(),
        });
    }
    let Some(n_raw) = int_literal(pool, n) else {
        return Ok(None);
    };
    let Some((n_int, sign)) = normalise_degree(family, &n_raw) else {
        return Ok(None);
    };
    if sign == 0 {
        return Ok(Some(pool.integer(0)));
    }
    // Family parameters: rational literals become constants; anything else
    // stays symbolic (except the Legendre order, which must be an integer).
    let mut pvals = Vec::with_capacity(params.len());
    let mut psyms = Vec::new();
    if family != OrthoFamily::AssocLegendre {
        for &p in params {
            match rational_literal(pool, p) {
                Some(q) => pvals.push(ParamVal::Const(q)),
                None => {
                    pvals.push(ParamVal::Sym(psyms.len()));
                    psyms.push(p);
                }
            }
        }
    }
    let cap = if psyms.is_empty() {
        MAX_EXPAND_DEGREE
    } else {
        MAX_EXPAND_DEGREE_SYMBOLIC
    };
    if n_int > cap {
        return Err(SpecialFunctionError::TooLarge {
            what: format!("{} degree n", family.name()),
            value: n_int.to_i64().unwrap_or(i64::MAX),
            max: cap,
        });
    }
    let nn = n_int.to_u32().expect("bounded by the cap");
    let out = match family {
        OrthoFamily::Legendre => rational_poly_expr(pool, &coeffs::legendre(nn)?, x),
        OrthoFamily::HermiteH => rational_poly_expr(pool, &coeffs::hermite_h(nn)?, x),
        OrthoFamily::HermiteHe => rational_poly_expr(pool, &coeffs::hermite_he(nn)?, x),
        OrthoFamily::ChebyshevT => rational_poly_expr(pool, &coeffs::chebyshev_t(nn)?, x),
        OrthoFamily::ChebyshevU => rational_poly_expr(pool, &coeffs::chebyshev_u(nn)?, x),
        OrthoFamily::Laguerre => poly_expr(
            pool,
            &coeffs::laguerre(nn, &ParamVal::Const(Rational::new()))?,
            &[],
            x,
        ),
        OrthoFamily::AssocLaguerre => poly_expr(pool, &coeffs::laguerre(nn, &pvals[0])?, &psyms, x),
        OrthoFamily::Gegenbauer => poly_expr(pool, &coeffs::gegenbauer(nn, &pvals[0])?, &psyms, x),
        OrthoFamily::Jacobi => {
            poly_expr(pool, &coeffs::jacobi(nn, &pvals[0], &pvals[1])?, &psyms, x)
        }
        OrthoFamily::AssocLegendre => {
            let Some(m) = int_literal(pool, params[0]) else {
                return Ok(None);
            };
            let Some(m) = m.to_i64() else {
                return Ok(Some(pool.integer(0))); // |m| > any admissible n
            };
            match assoc_legendre_form(nn, m)? {
                None => pool.integer(0),
                Some(form) => {
                    if let Some(q) = rational_literal(pool, x) {
                        return Ok(Some(assoc_legendre_at_rational(pool, &form, &q)));
                    }
                    let poly = rational_poly_expr(pool, &form.poly, x);
                    if form.abs_m == 0 {
                        poly
                    } else {
                        let one_minus_x2 = mk_add(
                            pool,
                            vec![
                                pool.integer(1),
                                pool.mul(vec![pool.integer(-1), pow_int(pool, x, 2)]),
                            ],
                        );
                        let e = if form.abs_m % 2 == 0 {
                            pool.integer(form.abs_m / 2)
                        } else {
                            pool.rational(form.abs_m, 2)
                        };
                        mk_mul(pool, vec![pool.pow(one_minus_x2, e), poly])
                    }
                }
            }
        }
    };
    Ok(Some(if sign < 0 {
        mk_mul(pool, vec![pool.integer(-1), out])
    } else {
        out
    }))
}

/// `family(n, params…, x)`.
///
/// With `expand = true` and an integer literal degree (and, for
/// `assoc_legendre_p`, an integer literal order) the explicit polynomial is
/// returned, with exact rational coefficients; otherwise the unevaluated node.
/// See the [module documentation](self) for conventions and reflections.
///
/// # Errors
///
/// * `E-SPECFN-004` — an explicit expansion past [`MAX_EXPAND_DEGREE`]
///   ([`MAX_EXPAND_DEGREE_SYMBOLIC`] with a symbolic family parameter);
///   build the node with `expand = false` instead.
/// * `E-SPECFN-005` — `params` has the wrong length for the family.
/// * `E-BUDGET-*` — the active budget ran out.
pub fn orthopoly(
    family: OrthoFamily,
    n: ExprId,
    params: &[ExprId],
    x: ExprId,
    expand: bool,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    if params.len() != family.n_params() {
        return Err(SpecialFunctionError::WrongArity {
            family: family.name(),
            expected: family.n_params(),
            got: params.len(),
        });
    }
    if expand {
        if let Some(e) = explicit(family, n, params, x, pool)? {
            return Ok(e);
        }
    }
    let mut args = Vec::with_capacity(family.arity());
    args.push(n);
    args.extend_from_slice(params);
    args.push(x);
    Ok(pool.func(family.name(), args))
}

/// Rewrite every orthogonal-polynomial node in `expr` whose degree (and
/// Legendre order) is an integer literal into its explicit polynomial.
///
/// Nodes with a symbolic degree are left alone, so this is safe to call on
/// any expression; it is what `subs(legendre_p(n, x), {n: 5})` needs to
/// become a polynomial.
///
/// # Errors
///
/// As [`orthopoly`]: `E-SPECFN-004` past the size cap, `E-BUDGET-*`.
pub fn expand_orthopoly(expr: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    let mut memo = std::collections::HashMap::new();
    expand_rec(expr, pool, &mut memo)
}

fn expand_rec(
    expr: ExprId,
    pool: &ExprPool,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> Result<ExprId, SpecialFunctionError> {
    if let Some(&r) = memo.get(&expr) {
        return Ok(r);
    }
    let data = pool.get(expr);
    let out = match data {
        ExprData::Add(args) => {
            let a = args
                .iter()
                .map(|&a| expand_rec(a, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            if a == args {
                expr
            } else {
                pool.add(a)
            }
        }
        ExprData::Mul(args) => {
            let a = args
                .iter()
                .map(|&a| expand_rec(a, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            if a == args {
                expr
            } else {
                pool.mul(a)
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (expand_rec(base, pool, memo)?, expand_rec(exp, pool, memo)?);
            if (b, e) == (base, exp) {
                expr
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => {
            let a = args
                .iter()
                .map(|&a| expand_rec(a, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            let rebuilt = if a == args {
                expr
            } else {
                pool.func(name.clone(), a.clone())
            };
            match OrthoFamily::from_name(&name) {
                Some(f) if a.len() == f.arity() => {
                    let (n, rest) = a.split_first().expect("arity ≥ 2");
                    let (x, params) = rest.split_last().expect("arity ≥ 2");
                    explicit(f, *n, params, *x, pool)?.unwrap_or(rebuilt)
                }
                _ => rebuilt,
            }
        }
        ExprData::Piecewise { branches, default } => {
            let mut nb = Vec::with_capacity(branches.len());
            for (c, v) in &branches {
                nb.push((expand_rec(*c, pool, memo)?, expand_rec(*v, pool, memo)?));
            }
            let d = expand_rec(default, pool, memo)?;
            if nb == branches && d == default {
                expr
            } else {
                pool.piecewise(nb, d)
            }
        }
        ExprData::Predicate { kind, args } => {
            let a = args
                .iter()
                .map(|&a| expand_rec(a, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            if a == args {
                expr
            } else {
                pool.predicate(kind, a)
            }
        }
        ExprData::BigO(arg) => {
            let a = expand_rec(arg, pool, memo)?;
            if a == arg {
                expr
            } else {
                pool.big_o(a)
            }
        }
        _ => expr,
    };
    memo.insert(expr, out);
    Ok(out)
}
