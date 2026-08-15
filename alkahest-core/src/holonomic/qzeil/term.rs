//! Recognising *`q`-proper hypergeometric terms* and computing their exact
//! shift quotients and their support in `k`.
//!
//! # The class
//!
//! ```text
//! F(n, k) = R(x, y) · z^k · w^n · q^{A·k² + B·n·k + C·n² + D·k + E·n}
//!           · ∏_j (q^{a_j·n + b_j·k + c_j}; q^{d_j})_{p_j·n + r_j·k + s_j}^{e_j}
//! ```
//!
//! with `x = qⁿ`, `y = q^k`, `R ∈ Q(q)(x, y)`, `z, w ∈ Q(q)\{0}`, integer
//! `a, b, c, p, r, s, e` and `d ≥ 1`, and rational `A, B, C, D, E` (an overall
//! constant power of `q` is irrelevant — it cancels out of every shift
//! quotient, which is all the algorithm ever uses).
//!
//! This is the `q`-analogue of [`super::super::hyperterm`]'s proper
//! hypergeometric class: `Γ(a·n + b·k + c)` becomes the `q`-Pochhammer symbol
//! `(a; q^d)_m = ∏_{t=0}^{m−1}(1 − a·q^{d·t})`, extended to every integer `m`
//! by its own recurrence `(a;q^d)_{m+1} = (a;q^d)_m·(1 − a·q^{d·m})`, and the
//! quotient of two `Γ`s at arguments differing by an integer becomes a quotient
//! of two Pochhammers whose lengths differ by an integer.
//!
//! # Two restrictions that are enforced, not assumed
//!
//! 1. **The base must divide the shift of the first argument.**
//!    `(q^{u}; q^{d})_v` shifted in `k` becomes `(q^{u + b}; q^{d})_{v + r}`,
//!    and the two are related by a *finite* product only when `d | b` — for
//!    `(q; q²)_v` under `k ↦ k+1` with `b = 1` the quotient is an infinite
//!    product and no algorithm in this family applies. It is refused
//!    ([`QHolonomicError::Unsupported`]), not approximated.
//! 2. **The quadratic exponent must give integer shift quotients.**
//!    `q^{k(k−1)/2}` is *not* rational in `y`, but its quotients are
//!    (`q^{k}` under `k ↦ k+1`), so half-integer `A`, `D` are accepted exactly
//!    when every quotient the search will form lands back in `Q(q)(x)(y)`.
//!    Anything else is refused.
//!
//! # Support
//!
//! [`QProperTerm::support`] decides, structurally, for which integers `k` the
//! term is **exactly zero** and whether it is ever **infinite** — the two facts
//! the boundary verdict in [`super`] is built from. A Pochhammer is zero
//! exactly when one of its factors `1 − q^{u + d·t}` is `1 − q⁰`, and infinite
//! exactly when the same happens in the reciprocal product a negative length
//! denotes; both are linear conditions on `(n, k)` plus a divisibility, and
//! both are decided over the rationals by Fourier–Motzkin, which is complete
//! (so a *proved empty* region really is empty over the integers too).

use super::field::{q_monomial, PolyX, PolyY, Qq, RatX, RatY};
use super::QHolonomicError;
use crate::holonomic::qfield::{rn_inv, rn_is_zero, rn_mul, rn_one, rn_rat, rn_var};
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;

/// Largest integer exponent accepted on a sub-term.
const MAX_POW: i32 = 32;
/// Largest number of explicit `1 − q^…` factors a single quotient may expand to.
const MAX_SPAN: i64 = 64;
/// Largest magnitude accepted for a linear-form coefficient.
const MAX_COEFF: i64 = 1 << 20;
/// Recursion guard for the parser.
const MAX_PARSE_DEPTH: usize = 64;
/// Cap on the constraint count during Fourier–Motzkin elimination.
const MAX_FM_CONSTRAINTS: usize = 64;

/// An integer-affine form `cn·n + ck·k + c0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Affine {
    pub cn: i64,
    pub ck: i64,
    pub c0: i64,
}

impl Affine {
    fn checked(cn: i64, ck: i64, c0: i64) -> Option<Affine> {
        (cn.abs() <= MAX_COEFF && ck.abs() <= MAX_COEFF && c0.abs() <= MAX_COEFF)
            .then_some(Affine { cn, ck, c0 })
    }

    fn add(&self, other: &Affine) -> Option<Affine> {
        Affine::checked(
            self.cn.checked_add(other.cn)?,
            self.ck.checked_add(other.ck)?,
            self.c0.checked_add(other.c0)?,
        )
    }

    fn scale(&self, m: i64) -> Option<Affine> {
        Affine::checked(
            self.cn.checked_mul(m)?,
            self.ck.checked_mul(m)?,
            self.c0.checked_mul(m)?,
        )
    }

    /// The form after `n ↦ n + i`.
    fn shift_n(&self, i: i64) -> Option<Affine> {
        Affine::checked(
            self.cn,
            self.ck,
            self.c0.checked_add(self.cn.checked_mul(i)?)?,
        )
    }

    /// `q^{form}` as an element of `Q(q)(x)(y)`.
    fn monomial(&self) -> RatY {
        q_monomial(self.cn, self.ck, self.c0)
    }
}

/// One `(q^{u}; q^{d})_{v}^{e}` factor, `u` and `v` integer-affine in `(n, k)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QPochFactor {
    /// Exponent of the first argument: `(q^u; q^d)_v`.
    pub u: Affine,
    /// Base step `d ≥ 1`: the base is `q^d`.
    pub d: i64,
    /// Length.
    pub v: Affine,
    /// Integer exponent on the whole symbol.
    pub e: i32,
}

/// The quadratic exponent `q^{A·k² + B·n·k + C·n² + D·k + E·n + F}`.
///
/// Rational coefficients are allowed because only *quotients* are ever formed;
/// see the module docs. The constant term `F` is carried (the parser needs it
/// to read affine forms out of the same routine) but never used by a quotient,
/// which is exactly why a half-integer `F` is harmless.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuadExp {
    pub a_kk: Rational,
    pub b_nk: Rational,
    pub c_nn: Rational,
    pub d_k: Rational,
    pub e_n: Rational,
    pub konst: Rational,
}

impl QuadExp {
    fn add(&self, other: &QuadExp) -> QuadExp {
        QuadExp {
            a_kk: (self.a_kk.clone() + other.a_kk.clone()),
            b_nk: (self.b_nk.clone() + other.b_nk.clone()),
            c_nn: (self.c_nn.clone() + other.c_nn.clone()),
            d_k: (self.d_k.clone() + other.d_k.clone()),
            e_n: (self.e_n.clone() + other.e_n.clone()),
            konst: (self.konst.clone() + other.konst.clone()),
        }
    }

    fn scale(&self, m: &Rational) -> QuadExp {
        QuadExp {
            a_kk: self.a_kk.clone() * m,
            b_nk: self.b_nk.clone() * m,
            c_nn: self.c_nn.clone() * m,
            d_k: self.d_k.clone() * m,
            e_n: self.e_n.clone() * m,
            konst: self.konst.clone() * m,
        }
    }

    /// Whether the exponent contributes nothing to any shift quotient.
    fn is_trivial(&self) -> bool {
        self.a_kk == 0 && self.b_nk == 0 && self.c_nn == 0 && self.d_k == 0 && self.e_n == 0
    }
}

/// A parsed `q`-proper hypergeometric term.
#[derive(Clone, Debug)]
pub struct QProperTerm {
    /// The rational prefactor `R(x, y) ∈ Q(q)(x)(y)`.
    pub rat: RatY,
    /// Base of `z^k`.
    pub z: Qq,
    /// Base of `w^n`.
    pub w: Qq,
    /// The quadratic exponent of `q`.
    pub quad: QuadExp,
    /// The `q`-Pochhammer factors.
    pub poch: Vec<QPochFactor>,
}

impl QProperTerm {
    fn one() -> Self {
        QProperTerm {
            rat: RatY::one(),
            z: rn_one(),
            w: rn_one(),
            quad: QuadExp::default(),
            poch: Vec::new(),
        }
    }

    fn mul(&self, other: &QProperTerm) -> QProperTerm {
        let mut poch = self.poch.clone();
        poch.extend(other.poch.iter().copied());
        QProperTerm {
            rat: self.rat.mul(&other.rat),
            z: rn_mul(&self.z, &other.z),
            w: rn_mul(&self.w, &other.w),
            quad: self.quad.add(&other.quad),
            poch,
        }
    }

    fn pow(&self, e: i32) -> Option<QProperTerm> {
        if e.unsigned_abs() > MAX_POW as u32 {
            return None;
        }
        let poch = self
            .poch
            .iter()
            .map(|f| {
                Some(QPochFactor {
                    e: f.e.checked_mul(e)?,
                    ..*f
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(QProperTerm {
            rat: self.rat.pow_i32(e)?,
            z: qq_pow_of(&self.z, e as i64)?,
            w: qq_pow_of(&self.w, e as i64)?,
            quad: self.quad.scale(&Rational::from(e)),
            poch,
        })
    }

    /// `F(n, k+1) / F(n, k)` as an exact element of `Q(q)(x)(y)`.
    pub fn ratio_k(&self) -> Result<RatY, QHolonomicError> {
        let mut acc = self.rat.qshift_y(1).div(&self.rat).ok_or_else(|| {
            QHolonomicError::NotQHypergeometric("term vanishes identically".into())
        })?;
        acc = acc.mul(&RatY::from_ratx(RatX::from_rn(self.z.clone())));
        if !self.quad.is_trivial() {
            // The exponent gains A·(2k+1) + B·n + D.
            let two_a = self.quad.a_kk.clone() * Rational::from(2);
            let konst = self.quad.a_kk.clone() + self.quad.d_k.clone();
            let form = quad_shift_form(&two_a, &self.quad.b_nk, &konst, "k")?;
            acc = acc.mul(&form.monomial());
        }
        for f in &self.poch {
            let step = self.poch_ratio(f, f.u.ck, f.v.ck)?;
            acc = acc.mul(&step.pow_i32(f.e).ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "q-Pochhammer factor is identically zero".into(),
                )
            })?);
        }
        Ok(acc)
    }

    /// `F(n+i, k) / F(n, k)` as an exact element of `Q(q)(x)(y)`.
    pub fn ratio_n(&self, i: i64) -> Result<RatY, QHolonomicError> {
        if i == 0 {
            return Ok(RatY::one());
        }
        let mut acc = self.rat.qshift_x(i).div(&self.rat).ok_or_else(|| {
            QHolonomicError::NotQHypergeometric("term vanishes identically".into())
        })?;
        let wi = qq_pow_of(&self.w, i)
            .ok_or_else(|| QHolonomicError::NotQHypergeometric("w^n has a zero base".into()))?;
        acc = acc.mul(&RatY::from_ratx(RatX::from_rn(wi)));
        if !self.quad.is_trivial() {
            // The exponent gains B·i·k + 2C·i·n + (C·i² + E·i).
            let ri = Rational::from(i);
            let k_coeff = self.quad.b_nk.clone() * ri.clone();
            let n_coeff = self.quad.c_nn.clone() * Rational::from(2) * ri.clone();
            let konst =
                self.quad.c_nn.clone() * Rational::from(i * i) + self.quad.e_n.clone() * ri.clone();
            let form = quad_shift_form(&k_coeff, &n_coeff, &konst, "n")?;
            acc = acc.mul(&form.monomial());
        }
        for f in &self.poch {
            let du =
                f.u.cn
                    .checked_mul(i)
                    .ok_or_else(|| QHolonomicError::Unsupported("shift overflow".into()))?;
            let dv =
                f.v.cn
                    .checked_mul(i)
                    .ok_or_else(|| QHolonomicError::Unsupported("shift overflow".into()))?;
            let step = self.poch_ratio(f, du, dv)?;
            acc = acc.mul(&step.pow_i32(f.e).ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "q-Pochhammer factor is identically zero".into(),
                )
            })?);
        }
        Ok(acc)
    }

    /// `(q^{u+δu}; q^d)_{v+δv} / (q^u; q^d)_v`, exactly.
    ///
    /// With `m = δu/d` (an integer, or the input is out of class),
    /// `(a·q^{d·m}; q^d)_L = (a;q^d)_{L+m} / (a;q^d)_m`, so the quotient is
    /// `(a;q^d)_{v+δv+m} / [(a;q^d)_m · (a;q^d)_v]` — two products of *constant*
    /// length, which is what makes it a rational function at all.
    fn poch_ratio(&self, f: &QPochFactor, du: i64, dv: i64) -> Result<RatY, QHolonomicError> {
        if f.d <= 0 {
            return Err(QHolonomicError::InvalidInput(
                "the q-Pochhammer base step must be a positive integer".into(),
            ));
        }
        if du % f.d != 0 {
            return Err(QHolonomicError::Unsupported(format!(
                "(q^u; q^{d})_v shifts its first argument by {du}, which q^{d} does not divide: \
                 the quotient is an infinite product and is outside the class this module supports",
                d = f.d
            )));
        }
        let m = du / f.d;
        let delta = dv
            .checked_add(m)
            .ok_or_else(|| QHolonomicError::Unsupported("shift overflow".into()))?;
        let grow = self.poch_len_ratio(f, delta)?;
        let fix = poch_const_len(&f.u, f.d, m)?;
        grow.div(&fix).ok_or_else(|| {
            QHolonomicError::NotQHypergeometric("q-Pochhammer factor is identically zero".into())
        })
    }

    /// `(q^u; q^d)_{v+c} / (q^u; q^d)_v` for a constant integer `c`.
    fn poch_len_ratio(&self, f: &QPochFactor, c: i64) -> Result<RatY, QHolonomicError> {
        if c.abs() > MAX_SPAN {
            return Err(QHolonomicError::Unsupported(format!(
                "a shift quotient would expand to {} explicit factors (limit {MAX_SPAN})",
                c.abs()
            )));
        }
        // u + d·v, the exponent at `t = 0`.
        let base =
            f.v.scale(f.d)
                .and_then(|dv| f.u.add(&dv))
                .ok_or_else(|| QHolonomicError::Unsupported("linear form overflow".into()))?;
        let range: Vec<i64> = if c >= 0 {
            (0..c).collect()
        } else {
            (c..0).collect()
        };
        let mut prod = RatY::one();
        for t in range {
            let step = base
                .add(&Affine {
                    cn: 0,
                    ck: 0,
                    c0: f.d.checked_mul(t).unwrap_or(i64::MAX),
                })
                .ok_or_else(|| QHolonomicError::Unsupported("linear form overflow".into()))?;
            prod = prod.mul(&one_minus(&step)?);
        }
        if c >= 0 {
            Ok(prod)
        } else {
            prod.inv().ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "q-Pochhammer factor is identically zero".into(),
                )
            })
        }
    }

    /// Parse an expression into the `q`-proper hypergeometric class.
    pub fn parse(
        expr: ExprId,
        q: ExprId,
        n: ExprId,
        k: ExprId,
        pool: &ExprPool,
    ) -> Result<QProperTerm, QHolonomicError> {
        let ctx = Ctx { q, n, k, pool };
        parse_rec(expr, &ctx, 0)
    }
}

/// `(q^u; q^d)_m` for a constant integer length `m`.
fn poch_const_len(u: &Affine, d: i64, m: i64) -> Result<RatY, QHolonomicError> {
    if m == 0 {
        return Ok(RatY::one());
    }
    if m.abs() > MAX_SPAN {
        return Err(QHolonomicError::Unsupported(format!(
            "a q-Pochhammer of constant length {m} exceeds the limit of {MAX_SPAN} factors"
        )));
    }
    let mut prod = RatY::one();
    if m > 0 {
        for t in 0..m {
            let step = u
                .add(&Affine {
                    cn: 0,
                    ck: 0,
                    c0: d.checked_mul(t).unwrap_or(i64::MAX),
                })
                .ok_or_else(|| QHolonomicError::Unsupported("linear form overflow".into()))?;
            prod = prod.mul(&one_minus(&step)?);
        }
        Ok(prod)
    } else {
        for t in 1..=(-m) {
            let step = u
                .add(&Affine {
                    cn: 0,
                    ck: 0,
                    c0: -d.checked_mul(t).unwrap_or(i64::MAX),
                })
                .ok_or_else(|| QHolonomicError::Unsupported("linear form overflow".into()))?;
            prod = prod.mul(&one_minus(&step)?);
        }
        prod.inv().ok_or_else(|| {
            QHolonomicError::NotQHypergeometric("q-Pochhammer factor is identically zero".into())
        })
    }
}

/// `1 − q^{form}`, refusing the identically-zero case `form ≡ 0`.
fn one_minus(form: &Affine) -> Result<RatY, QHolonomicError> {
    if form.cn == 0 && form.ck == 0 && form.c0 == 0 {
        return Err(QHolonomicError::NotQHypergeometric(
            "a q-Pochhammer factor 1 - q^0 is identically zero, so the term is not a well-defined \
             q-hypergeometric term"
                .into(),
        ));
    }
    Ok(RatY::one().sub(&form.monomial()))
}

/// A quadratic exponent's contribution to a shift quotient, as an integer form.
fn quad_shift_form(
    k_coeff: &Rational,
    n_coeff: &Rational,
    konst: &Rational,
    which: &str,
) -> Result<Affine, QHolonomicError> {
    let int = |r: &Rational| -> Option<i64> { (*r.denom() == 1).then(|| r.numer().to_i64())? };
    match (int(k_coeff), int(n_coeff), int(konst)) {
        (Some(ck), Some(cn), Some(c0)) => Affine::checked(cn, ck, c0)
            .ok_or_else(|| QHolonomicError::Unsupported("linear form overflow".into())),
        _ => Err(QHolonomicError::Unsupported(format!(
            "the quadratic exponent of q leaves Q(q)(q^n)(q^k) under the {which}-shift: its \
             quotient exponent {k_coeff}·k + {n_coeff}·n + {konst} is not integral"
        ))),
    }
}

/// `base^e` in `Q(q)`.
fn qq_pow_of(base: &Qq, e: i64) -> Option<Qq> {
    if e == 0 {
        return Some(rn_one());
    }
    if rn_is_zero(base) {
        return None;
    }
    if e.unsigned_abs() > 1024 {
        return None;
    }
    let b = if e < 0 { rn_inv(base)? } else { base.clone() };
    let mut acc = rn_one();
    for _ in 0..e.unsigned_abs() {
        acc = rn_mul(&acc, &b);
    }
    Some(acc)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Ctx<'a> {
    q: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &'a ExprPool,
}

fn parse_rec(expr: ExprId, ctx: &Ctx<'_>, depth: usize) -> Result<QProperTerm, QHolonomicError> {
    if depth > MAX_PARSE_DEPTH {
        return Err(QHolonomicError::NotQHypergeometric(
            "expression nests deeper than the parser supports".into(),
        ));
    }
    // Fast path: a sub-expression that is already rational in q, x and y.
    if let Some(r) = as_raty(expr, ctx, 0) {
        return Ok(QProperTerm {
            rat: r,
            ..QProperTerm::one()
        });
    }
    match ctx.pool.get(expr) {
        ExprData::Mul(args) => {
            let mut acc = QProperTerm::one();
            for a in args {
                acc = acc.mul(&parse_rec(a, ctx, depth + 1)?);
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => parse_pow(base, exp, ctx, depth),
        ExprData::Func { name, args } => parse_func(&name, &args, ctx),
        _ => Err(QHolonomicError::NotQHypergeometric(format!(
            "{} is not a q-hypergeometric factor",
            ctx.pool.display(expr)
        ))),
    }
}

fn parse_pow(
    base: ExprId,
    exp: ExprId,
    ctx: &Ctx<'_>,
    depth: usize,
) -> Result<QProperTerm, QHolonomicError> {
    if let Some(e) = as_i32(exp, ctx.pool) {
        let b = parse_rec(base, ctx, depth + 1)?;
        return b.pow(e).ok_or_else(|| {
            QHolonomicError::NotQHypergeometric(format!(
                "exponent {e} is outside the supported range (|e| <= {MAX_POW})"
            ))
        });
    }
    // `q^{quadratic in n, k}` — the only way out of the rational class that the
    // algorithm can still use, because its quotients come back into it.
    if base == ctx.q {
        let quad = as_quadratic(exp, ctx, 0).ok_or_else(|| {
            QHolonomicError::NotQHypergeometric(format!(
                "q^({}) needs an exponent that is a polynomial of degree <= 2 in n and k",
                ctx.pool.display(exp)
            ))
        })?;
        return Ok(QProperTerm {
            quad,
            ..QProperTerm::one()
        });
    }
    // `c^{α·n + β·k + γ}` with `c ∈ Q(q)` — a `z^k·w^n` factor.
    let Some(c) = as_qq(base, ctx, 0) else {
        return Err(QHolonomicError::NotQHypergeometric(format!(
            "a power with a symbolic exponent needs a base in Q(q), got {}",
            ctx.pool.display(base)
        )));
    };
    if rn_is_zero(&c) {
        return Err(QHolonomicError::NotQHypergeometric(
            "zero raised to a symbolic power".into(),
        ));
    }
    let form = as_affine(exp, ctx, 0).ok_or_else(|| {
        QHolonomicError::NotQHypergeometric(format!(
            "the exponent {} is not integer-affine in n and k",
            ctx.pool.display(exp)
        ))
    })?;
    let z = qq_pow_of(&c, form.ck)
        .ok_or_else(|| QHolonomicError::NotQHypergeometric("exponential base overflow".into()))?;
    let w = qq_pow_of(&c, form.cn)
        .ok_or_else(|| QHolonomicError::NotQHypergeometric("exponential base overflow".into()))?;
    let konst = qq_pow_of(&c, form.c0)
        .ok_or_else(|| QHolonomicError::NotQHypergeometric("exponential base overflow".into()))?;
    Ok(QProperTerm {
        rat: RatY::from_ratx(RatX::from_rn(konst)),
        z,
        w,
        ..QProperTerm::one()
    })
}

fn parse_func(name: &str, args: &[ExprId], ctx: &Ctx<'_>) -> Result<QProperTerm, QHolonomicError> {
    match (name, args.len()) {
        // `qpochhammer(u, d, v)` = (q^u; q^d)_v.
        ("qpochhammer", 3) => {
            let u = as_affine(args[0], ctx, 0).ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "qpochhammer(u, d, v): u must be integer-affine in n and k".into(),
                )
            })?;
            let d = as_i32(args[1], ctx.pool).map(i64::from).ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "qpochhammer(u, d, v): d must be a positive integer literal".into(),
                )
            })?;
            if d <= 0 {
                return Err(QHolonomicError::InvalidInput(
                    "qpochhammer(u, d, v): the base step d must be at least 1".into(),
                ));
            }
            let v = as_affine(args[2], ctx, 0).ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "qpochhammer(u, d, v): v must be integer-affine in n and k".into(),
                )
            })?;
            Ok(QProperTerm {
                poch: vec![QPochFactor { u, d, v, e: 1 }],
                ..QProperTerm::one()
            })
        }
        // `qbinomial(N, K)` — the Gaussian binomial coefficient.
        ("qbinomial", 2) => {
            let top = as_affine(args[0], ctx, 0).ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "qbinomial(N, K): N must be integer-affine in n and k".into(),
                )
            })?;
            let bot = as_affine(args[1], ctx, 0).ok_or_else(|| {
                QHolonomicError::NotQHypergeometric(
                    "qbinomial(N, K): K must be integer-affine in n and k".into(),
                )
            })?;
            let diff = bot
                .scale(-1)
                .and_then(|neg| top.add(&neg))
                .ok_or_else(|| QHolonomicError::Unsupported("linear form overflow".into()))?;
            let one = Affine {
                cn: 0,
                ck: 0,
                c0: 1,
            };
            Ok(QProperTerm {
                poch: vec![
                    QPochFactor {
                        u: one,
                        d: 1,
                        v: top,
                        e: 1,
                    },
                    QPochFactor {
                        u: one,
                        d: 1,
                        v: bot,
                        e: -1,
                    },
                    QPochFactor {
                        u: one,
                        d: 1,
                        v: diff,
                        e: -1,
                    },
                ],
                ..QProperTerm::one()
            })
        }
        _ => Err(QHolonomicError::NotQHypergeometric(format!(
            "{name}/{} is not a q-hypergeometric factor; supported heads are qpochhammer(u, d, v) \
             and qbinomial(N, K)",
            args.len()
        ))),
    }
}

/// An expression in `q` alone, as an element of `Q(q)`.
fn as_qq(expr: ExprId, ctx: &Ctx<'_>, depth: usize) -> Option<Qq> {
    if depth > MAX_PARSE_DEPTH {
        return None;
    }
    if expr == ctx.q {
        return Some(rn_var());
    }
    if expr == ctx.n || expr == ctx.k {
        return None;
    }
    match ctx.pool.get(expr) {
        ExprData::Integer(i) => Some(rn_rat(Rational::from(i.0.clone()))),
        ExprData::Rational(r) => Some(rn_rat(r.0.clone())),
        ExprData::Add(args) => {
            args.iter()
                .try_fold(crate::holonomic::qfield::rn_zero(), |acc, &a| {
                    Some(crate::holonomic::qfield::rn_add(
                        &acc,
                        &as_qq(a, ctx, depth + 1)?,
                    ))
                })
        }
        ExprData::Mul(args) => args.iter().try_fold(rn_one(), |acc, &a| {
            Some(rn_mul(&acc, &as_qq(a, ctx, depth + 1)?))
        }),
        ExprData::Pow { base, exp } => {
            let e = as_i32(exp, ctx.pool)?;
            if e.unsigned_abs() > MAX_POW as u32 {
                return None;
            }
            qq_pow_of(&as_qq(base, ctx, depth + 1)?, e as i64)
        }
        _ => None,
    }
}

/// An expression as an element of `Q(q)(x)(y)`, with `x = q^n`, `y = q^k`.
fn as_raty(expr: ExprId, ctx: &Ctx<'_>, depth: usize) -> Option<RatY> {
    if depth > MAX_PARSE_DEPTH {
        return None;
    }
    if expr == ctx.q {
        return Some(RatY::from_ratx(RatX::from_rn(rn_var())));
    }
    if expr == ctx.n || expr == ctx.k {
        return None;
    }
    match ctx.pool.get(expr) {
        ExprData::Integer(i) => Some(RatY::from_ratx(RatX::from_rn(rn_rat(Rational::from(
            i.0.clone(),
        ))))),
        ExprData::Rational(r) => Some(RatY::from_ratx(RatX::from_rn(rn_rat(r.0.clone())))),
        ExprData::Add(args) => args.iter().try_fold(RatY::zero(), |acc, &a| {
            Some(acc.add(&as_raty(a, ctx, depth + 1)?))
        }),
        ExprData::Mul(args) => args.iter().try_fold(RatY::one(), |acc, &a| {
            Some(acc.mul(&as_raty(a, ctx, depth + 1)?))
        }),
        ExprData::Pow { base, exp } => {
            if let Some(e) = as_i32(exp, ctx.pool) {
                if e.unsigned_abs() > MAX_POW as u32 {
                    return None;
                }
                return as_raty(base, ctx, depth + 1)?.pow_i32(e);
            }
            // `q^{affine in n, k}` is the monomial `x^α·y^β·q^γ`.
            if base == ctx.q {
                return Some(as_affine(exp, ctx, 0)?.monomial());
            }
            None
        }
        _ => None,
    }
}

/// An integer-affine form in `n` and `k`.
fn as_affine(expr: ExprId, ctx: &Ctx<'_>, depth: usize) -> Option<Affine> {
    let quad = as_quadratic(expr, ctx, depth)?;
    let int = |r: &Rational| -> Option<i64> {
        if *r.denom() != 1 {
            return None;
        }
        r.numer().to_i64()
    };
    if quad.a_kk != 0 || quad.b_nk != 0 || quad.c_nn != 0 {
        return None;
    }
    Affine::checked(int(&quad.e_n)?, int(&quad.d_k)?, int(&quad.konst)?)
}

/// An expression as a polynomial of degree ≤ 2 in `n` and `k` with rational
/// coefficients.
fn as_quadratic(expr: ExprId, ctx: &Ctx<'_>, depth: usize) -> Option<QuadExp> {
    if depth > MAX_PARSE_DEPTH {
        return None;
    }
    if expr == ctx.n {
        return Some(QuadExp {
            e_n: Rational::from(1),
            ..QuadExp::default()
        });
    }
    if expr == ctx.k {
        return Some(QuadExp {
            d_k: Rational::from(1),
            ..QuadExp::default()
        });
    }
    if expr == ctx.q {
        return None;
    }
    match ctx.pool.get(expr) {
        ExprData::Integer(i) => Some(QuadExp {
            konst: Rational::from(i.0.clone()),
            ..QuadExp::default()
        }),
        ExprData::Rational(r) => Some(QuadExp {
            konst: r.0.clone(),
            ..QuadExp::default()
        }),
        ExprData::Add(args) => args.iter().try_fold(QuadExp::default(), |acc, &a| {
            Some(acc.add(&as_quadratic(a, ctx, depth + 1)?))
        }),
        ExprData::Mul(args) => {
            let mut acc = QuadExp {
                konst: Rational::from(1),
                ..QuadExp::default()
            };
            for &a in args.iter() {
                acc = quad_mul(&acc, &as_quadratic(a, ctx, depth + 1)?)?;
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let e = as_i32(exp, ctx.pool)?;
            if !(0..=2).contains(&e) {
                return None;
            }
            let b = as_quadratic(base, ctx, depth + 1)?;
            let mut acc = QuadExp {
                konst: Rational::from(1),
                ..QuadExp::default()
            };
            for _ in 0..e {
                acc = quad_mul(&acc, &b)?;
            }
            Some(acc)
        }
        _ => None,
    }
}

/// Product of two quadratics, when it stays quadratic.
fn quad_mul(a: &QuadExp, b: &QuadExp) -> Option<QuadExp> {
    let a_lin = a.a_kk != 0 || a.b_nk != 0 || a.c_nn != 0;
    let b_lin = b.a_kk != 0 || b.b_nk != 0 || b.c_nn != 0;
    let a_deg1 = a.d_k != 0 || a.e_n != 0;
    let b_deg1 = b.d_k != 0 || b.e_n != 0;
    if (a_lin && (b_deg1 || b_lin)) || (b_lin && (a_deg1 || a_lin)) {
        return None; // degree would exceed 2
    }
    let mut out = QuadExp {
        a_kk: a.a_kk.clone() * b.konst.clone() + b.a_kk.clone() * a.konst.clone(),
        b_nk: a.b_nk.clone() * b.konst.clone() + b.b_nk.clone() * a.konst.clone(),
        c_nn: a.c_nn.clone() * b.konst.clone() + b.c_nn.clone() * a.konst.clone(),
        d_k: a.d_k.clone() * b.konst.clone() + b.d_k.clone() * a.konst.clone(),
        e_n: a.e_n.clone() * b.konst.clone() + b.e_n.clone() * a.konst.clone(),
        konst: a.konst.clone() * b.konst.clone(),
    };
    // The degree-1 × degree-1 cross terms.
    out.a_kk += a.d_k.clone() * b.d_k.clone();
    out.c_nn += a.e_n.clone() * b.e_n.clone();
    out.b_nk += a.d_k.clone() * b.e_n.clone() + a.e_n.clone() * b.d_k.clone();
    Some(out)
}

fn as_i32(expr: ExprId, pool: &ExprPool) -> Option<i32> {
    match pool.get(expr) {
        ExprData::Integer(i) => i.0.to_i32(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Support analysis
// ---------------------------------------------------------------------------

/// A rational linear constraint `cn·n + ck·k + c0 ≤ 0`.
#[derive(Clone, Debug)]
struct Lin {
    cn: Rational,
    ck: Rational,
    c0: Rational,
}

/// Where a factor is exactly `0`, or exactly `∞`.
#[derive(Clone, Debug)]
enum Region {
    /// Provably empty.
    Empty,
    /// Exactly the points satisfying every constraint.
    Constraints(Vec<Lin>),
    /// Not characterised — treated as possibly non-empty, never as a proof.
    Opaque,
}

/// What [`QProperTerm::support`] establishes about the summand.
#[derive(Clone, Debug)]
pub struct QSupport {
    /// Proved: the term vanishes at every sufficiently large integer `k`.
    pub bounded_above: bool,
    /// Proved: the term vanishes at every sufficiently negative integer `k`.
    pub bounded_below: bool,
    /// A witness bound when one is expressible as a single affine form:
    /// `F(n+i, k) = 0` for every integer `k > hi`.
    pub hi: Option<Rational2>,
    /// Likewise: `F(n+i, k) = 0` for every integer `k < lo`.
    pub lo: Option<Rational2>,
    /// Whether no factor can be infinite anywhere in `{n ≥ n_min, k ∈ Z}`.
    pub finite: bool,
    /// Why the analysis stopped short, when it did.
    pub reason: String,
}

/// An affine bound `a·n + b` with rational coefficients.
#[derive(Clone, Debug)]
pub struct Rational2 {
    pub a: Rational,
    pub b: Rational,
}

impl QProperTerm {
    /// Decide the structural support of `F(n+i, k)` in `k`, for `n ≥ n_min`.
    ///
    /// Returns bounds `lo`, `hi` such that the term is **exactly zero** at every
    /// integer `k` outside `[lo, hi]`, and whether the term is finite at every
    /// integer `k` at all. Both are what [`super::q_boundary_status`] needs;
    /// neither is guessed — a bound is reported only when the linear conditions
    /// for it hold on the whole half-line.
    pub fn support(&self, i: i64, n_min: i64) -> QSupport {
        let mut reason = String::new();
        // 1. Nothing may be infinite: a `0·∞` would make the value undefined,
        //    and an infinite summand breaks the telescoping argument outright.
        let mut finite = self.rat_is_everywhere_finite();
        if !finite {
            reason = "the rational prefactor may be singular at an integer k (its denominator is \
                      not a monomial in q^n, q^k)"
                .to_string();
        }
        if finite {
            for f in &self.poch {
                let Some(shifted) = shift_factor(f, i) else {
                    finite = false;
                    reason = "a q-Pochhammer factor overflowed under the n-shift".to_string();
                    break;
                };
                let region = if shifted.e > 0 {
                    infinite_region(&shifted)
                } else {
                    zero_region(&shifted)
                };
                if !region_is_empty(&region, n_min) {
                    finite = false;
                    reason = format!(
                        "the factor (q^({}n+{}k+{}); q^{})_({}n+{}k+{})^{} may be infinite at an \
                         integer k, so the summand is not everywhere finite",
                        shifted.u.cn,
                        shifted.u.ck,
                        shifted.u.c0,
                        shifted.d,
                        shifted.v.cn,
                        shifted.v.ck,
                        shifted.v.c0,
                        shifted.e
                    );
                    break;
                }
            }
        }

        // 2. Support bounds: a factor that is exactly zero on a whole half-line
        //    in `k` bounds the support on that side.
        let mut hi: Option<Rational2> = None;
        let mut lo: Option<Rational2> = None;
        let mut bounded_above = false;
        let mut bounded_below = false;
        for f in &self.poch {
            let Some(shifted) = shift_factor(f, i) else {
                continue;
            };
            let region = if shifted.e > 0 {
                zero_region(&shifted)
            } else {
                infinite_region(&shifted)
            };
            let Region::Constraints(cons) = &region else {
                continue;
            };
            // `covers_*` returns the thresholds that must *all* be met; the
            // effective one is their max (resp. min), which is an affine form
            // only when they are comparable. Coverage is what the boundary
            // proof needs; the bound is reporting, so an incomparable family
            // still proves the support is bounded and simply reports no number.
            if let Some(ts) = covers_k_large(cons, n_min) {
                bounded_above = true;
                if let Some(t) = fold_bound(&ts, Extreme::Max) {
                    hi = tighten(hi, sub_one(&t), Extreme::Min);
                }
            }
            if let Some(ts) = covers_k_small(cons, n_min) {
                bounded_below = true;
                if let Some(t) = fold_bound(&ts, Extreme::Min) {
                    lo = tighten(lo, add_one(&t), Extreme::Max);
                }
            }
        }
        if !bounded_above && reason.is_empty() {
            reason = "no factor forces the summand to vanish for all large k, so its support in k \
                      was not established"
                .to_string();
        }
        if !bounded_below && reason.is_empty() {
            reason = "no factor forces the summand to vanish for all sufficiently negative k, so \
                      its support in k was not established"
                .to_string();
        }
        QSupport {
            bounded_above,
            bounded_below,
            hi,
            lo,
            finite,
            reason,
        }
    }

    /// Whether the rational prefactor is finite at every `x = qⁿ`, `y = q^k`.
    ///
    /// Sufficient, deliberately: a denominator that is a monomial in `x` and
    /// `y` never vanishes there, and anything else is left undecided rather
    /// than analysed for integer roots.
    fn rat_is_everywhere_finite(&self) -> bool {
        polyy_is_monomial(&self.rat.den)
            && self
                .rat
                .num
                .coeffs
                .iter()
                .chain(self.rat.den.coeffs.iter())
                .all(|c| polyx_is_monomial(&c.den))
    }
}

fn polyy_is_monomial(p: &PolyY) -> bool {
    p.coeffs.iter().filter(|c| !c.is_zero()).count() == 1
}

fn polyx_is_monomial(p: &PolyX) -> bool {
    p.coeffs.iter().filter(|c| !rn_is_zero(c)).count() == 1
}

fn shift_factor(f: &QPochFactor, i: i64) -> Option<QPochFactor> {
    Some(QPochFactor {
        u: f.u.shift_n(i)?,
        d: f.d,
        v: f.v.shift_n(i)?,
        e: f.e,
    })
}

/// Where `(q^u; q^d)_v = 0`: some `t ∈ [0, v−1]` has `u + d·t = 0`, i.e.
/// `d | u`, `U = u/d ≤ 0` and `v + U ≥ 1`.
fn zero_region(f: &QPochFactor) -> Region {
    let Some(uu) = divide_form(&f.u, f.d) else {
        return match divisibility(&f.u, f.d) {
            Divisibility::Never => Region::Empty,
            _ => Region::Opaque,
        };
    };
    Region::Constraints(vec![
        // U ≤ 0
        uu.clone(),
        // 1 − v − U ≤ 0
        lin_sub(&lin_const(1), &lin_add(&affine_lin(&f.v), &uu)),
    ])
}

/// Where `(q^u; q^d)_v = ∞`: some `t ∈ [1, −v]` has `u − d·t = 0`, i.e.
/// `d | u`, `U = u/d ≥ 1` and `v + U ≤ 0`.
fn infinite_region(f: &QPochFactor) -> Region {
    let Some(uu) = divide_form(&f.u, f.d) else {
        return match divisibility(&f.u, f.d) {
            Divisibility::Never => Region::Empty,
            _ => Region::Opaque,
        };
    };
    Region::Constraints(vec![
        // 1 − U ≤ 0
        lin_sub(&lin_const(1), &uu),
        // v + U ≤ 0
        lin_add(&affine_lin(&f.v), &uu),
    ])
}

enum Divisibility {
    Always,
    Never,
    Sometimes,
}

fn divisibility(u: &Affine, d: i64) -> Divisibility {
    if d == 0 {
        return Divisibility::Never;
    }
    if u.cn % d == 0 && u.ck % d == 0 {
        return if u.c0 % d == 0 {
            Divisibility::Always
        } else {
            Divisibility::Never
        };
    }
    let g = gcd_i64(gcd_i64(u.cn.abs(), u.ck.abs()), d.abs());
    if g != 0 && u.c0 % g != 0 {
        Divisibility::Never
    } else {
        Divisibility::Sometimes
    }
}

/// `u/d` as a linear form, when `d | u` for *every* integer `(n, k)`.
fn divide_form(u: &Affine, d: i64) -> Option<Lin> {
    match divisibility(u, d) {
        Divisibility::Always => Some(Lin {
            cn: Rational::from((u.cn, d)),
            ck: Rational::from((u.ck, d)),
            c0: Rational::from((u.c0, d)),
        }),
        _ => None,
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn affine_lin(a: &Affine) -> Lin {
    Lin {
        cn: Rational::from(a.cn),
        ck: Rational::from(a.ck),
        c0: Rational::from(a.c0),
    }
}

fn lin_const(c: i64) -> Lin {
    Lin {
        cn: Rational::new(),
        ck: Rational::new(),
        c0: Rational::from(c),
    }
}

fn lin_add(a: &Lin, b: &Lin) -> Lin {
    Lin {
        cn: a.cn.clone() + b.cn.clone(),
        ck: a.ck.clone() + b.ck.clone(),
        c0: a.c0.clone() + b.c0.clone(),
    }
}

fn lin_sub(a: &Lin, b: &Lin) -> Lin {
    Lin {
        cn: a.cn.clone() - b.cn.clone(),
        ck: a.ck.clone() - b.ck.clone(),
        c0: a.c0.clone() - b.c0.clone(),
    }
}

/// Whether the region is **provably** empty over `{n ≥ n_min, k ∈ R}`.
///
/// Fourier–Motzkin over the rationals is complete for the relaxation, and a
/// rational-empty region is integer-empty, so a `true` here is a proof. A cap
/// on the constraint count makes this return `false` rather than take
/// exponential time — the conservative direction.
fn region_is_empty(region: &Region, n_min: i64) -> bool {
    match region {
        Region::Empty => true,
        Region::Opaque => false,
        Region::Constraints(cons) => {
            let mut all = cons.clone();
            // n_min − n ≤ 0
            all.push(Lin {
                cn: Rational::from(-1),
                ck: Rational::new(),
                c0: Rational::from(n_min),
            });
            let Some(after_k) = fm_eliminate(all, Var::K) else {
                return false;
            };
            let Some(after_n) = fm_eliminate(after_k, Var::N) else {
                return false;
            };
            after_n.iter().any(|l| l.c0 > 0)
        }
    }
}

#[derive(Clone, Copy)]
enum Var {
    N,
    K,
}

fn coeff_of(l: &Lin, v: Var) -> &Rational {
    match v {
        Var::N => &l.cn,
        Var::K => &l.ck,
    }
}

/// One Fourier–Motzkin elimination step; `None` when the constraint cap blows.
fn fm_eliminate(cons: Vec<Lin>, v: Var) -> Option<Vec<Lin>> {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    let mut out = Vec::new();
    for l in cons {
        let c = coeff_of(&l, v).clone();
        if c > 0 {
            pos.push(l);
        } else if c < 0 {
            neg.push(l);
        } else {
            out.push(l);
        }
    }
    if out.len() + pos.len() * neg.len() > MAX_FM_CONSTRAINTS {
        return None;
    }
    for p in &pos {
        for m in &neg {
            let pc = coeff_of(p, v).clone();
            let mc = -coeff_of(m, v).clone();
            // p·mc + m·pc has a zero coefficient on `v`, and both scales are > 0.
            out.push(Lin {
                cn: p.cn.clone() * mc.clone() + m.cn.clone() * pc.clone(),
                ck: p.ck.clone() * mc.clone() + m.ck.clone() * pc.clone(),
                c0: p.c0.clone() * mc.clone() + m.c0.clone() * pc.clone(),
            });
        }
    }
    Some(out)
}

/// If every constraint holds for all large `k` (given `n ≥ n_min`), the
/// thresholds whose maximum the region starts at: it contains every
/// `k ≥ max_j t_j(n)`. `None` means the region does **not** cover large `k`.
fn covers_k_large(cons: &[Lin], n_min: i64) -> Option<Vec<Rational2>> {
    let mut bounds = Vec::new();
    for l in cons {
        if l.ck < 0 {
            // cn·n + ck·k + c0 ≤ 0  ⟺  k ≥ (cn·n + c0)/(−ck)
            let s = -l.ck.clone();
            bounds.push(Rational2 {
                a: l.cn.clone() / s.clone(),
                b: l.c0.clone() / s,
            });
        } else if l.ck == 0 {
            // Must hold for every n ≥ n_min on its own.
            if l.cn > 0 || l.cn.clone() * Rational::from(n_min) + l.c0.clone() > 0 {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(bounds)
}

/// The mirror of [`covers_k_large`]: the region contains every
/// `k ≤ min_j t_j(n)`.
fn covers_k_small(cons: &[Lin], n_min: i64) -> Option<Vec<Rational2>> {
    let mut bounds = Vec::new();
    for l in cons {
        if l.ck > 0 {
            // cn·n + ck·k + c0 ≤ 0  ⟺  k ≤ −(cn·n + c0)/ck
            let s = l.ck.clone();
            bounds.push(Rational2 {
                a: -l.cn.clone() / s.clone(),
                b: -l.c0.clone() / s,
            });
        } else if l.ck == 0 {
            if l.cn > 0 || l.cn.clone() * Rational::from(n_min) + l.c0.clone() > 0 {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(bounds)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Extreme {
    Min,
    Max,
}

/// The extremum of a family of affine bounds, when it *is* affine — i.e. when
/// they share a slope. Incomparable slopes give `None`: no number is reported
/// rather than a wrong one.
fn fold_bound(ts: &[Rational2], which: Extreme) -> Option<Rational2> {
    let first = ts.first()?;
    let mut best = first.clone();
    for t in &ts[1..] {
        if t.a != best.a {
            return None;
        }
        let take = match which {
            Extreme::Min => t.b < best.b,
            Extreme::Max => t.b > best.b,
        };
        if take {
            best = t.clone();
        }
    }
    Some(best)
}

/// Keep the tighter of two affine bounds; an incomparable pair keeps the one
/// already held, which is sound because each is independently valid.
fn tighten(cur: Option<Rational2>, t: Rational2, which: Extreme) -> Option<Rational2> {
    match cur {
        None => Some(t),
        Some(c) => {
            if t.a != c.a {
                return Some(c);
            }
            let take = match which {
                Extreme::Min => t.b < c.b,
                Extreme::Max => t.b > c.b,
            };
            Some(if take { t } else { c })
        }
    }
}

fn sub_one(t: &Rational2) -> Rational2 {
    Rational2 {
        a: t.a.clone(),
        b: t.b.clone() - Rational::from(1),
    }
}

fn add_one(t: &Rational2) -> Rational2 {
    Rational2 {
        a: t.a.clone(),
        b: t.b.clone() + Rational::from(1),
    }
}
