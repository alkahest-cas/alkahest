//! Registry entries for the Bessel, Airy, zeta and polylogarithm functions.
//!
//! One [`SpecFunPrimitive`] per [`Kernel`]. Capability bundle:
//!
//! * `simplify` — [`canonical`]: reflections in the order, the `bessel_j0` /
//!   `bessel_j1` / `dilog` spellings, special values (module docs of
//!   [`crate::specfun`]).
//! * `diff_forward` — in the variable only:
//!
//!   | node | `d/dx` |
//!   |---|---|
//!   | `J_ν`, `Y_ν` | `(F_{ν−1} − F_{ν+1})/2` (DLMF 10.6.1) |
//!   | `I_ν` | `(I_{ν−1} + I_{ν+1})/2` (10.29.1) |
//!   | `K_ν` | `−(K_{ν−1} + K_{ν+1})/2` (10.29.1) |
//!   | `j_n`, `y_n` | `−f_{n+1} + (n/x) f_n` (10.51.2) |
//!   | `Ai`, `Bi` | `Ai′`, `Bi′`; and `Ai″ = x·Ai` (9.2.1) |
//!   | `ζ(s, a)` | `∂/∂a = −s·ζ(s+1, a)` (25.11.17) |
//!   | `Li_s(z)` | `Li_{s−1}(z)/z` |
//!
//!   An order, `s`, or the argument of `zeta` that depends on the variable
//!   makes the rule decline, so `diff` refuses (`E-DIFF-001`).
//! * `numeric_f64` / `numeric_ball` — [`super::numeric`] (Arb).
//! * `zeta_prime` (`ζ′`) and the Hankel functions `hankel1`/`hankel2` follow the
//!   same bundle; the Hankel pair is complex-valued, so it carries no real
//!   `numeric_f64` / `numeric_ball` slot (the complex evaluator handles it).
//! * No reverse rule (an order has no cotangent; the generic fallback asks
//!   `diff_forward`), no Lean theorem, no Taylor model, no LLVM lowering.

use rug::ops::Pow;
use rug::{Complete, Integer, Rational};

use crate::ball::ArbBall;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::orthopoly::{int_literal, mk_add, mk_mul, rat_node, rational_literal};
use crate::primitive::Primitive;

use super::closed;
use super::node;
use super::numeric::Kernel;

/// Largest `|s|` for which `ζ(s)` at an integer is rewritten exactly.
const MAX_EXACT_ZETA: i64 = 400;
/// Largest `n` for which `ζ(−n, a)` becomes its Bernoulli polynomial.
const MAX_BERNOULLI_POLY: i64 = 60;
/// Largest `n` for which `Li_{−n}` becomes its rational function.
const MAX_POLYLOG_NEG: i64 = 60;
/// Largest positive integer `a` for which `ζ(s, a)` peels off `a − 1` terms.
const MAX_HURWITZ_SHIFT: i64 = 100;

/// A registry primitive for one kernel.
pub struct SpecFunPrimitive(pub(crate) Kernel);

fn is_zero_lit(pool: &ExprPool, e: ExprId) -> bool {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(q) => q.0 == 0,
        _ => false,
    })
}

fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
    mk_mul(pool, vec![pool.integer(-1), e])
}

/// `y` when `x` is syntactically `−y`.
fn negated(pool: &ExprPool, x: ExprId) -> Option<ExprId> {
    let args = pool.with(x, |d| match d {
        ExprData::Mul(a) => Some(a.clone()),
        _ => None,
    })?;
    let minus = pool.integer(-1);
    let pos = args.iter().position(|&a| a == minus)?;
    let rest: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pos)
        .map(|(_, &a)| a)
        .collect();
    Some(mk_mul(pool, rest))
}

/// `a + k` (folding a literal `a`).
fn shift(pool: &ExprPool, a: ExprId, k: i64) -> ExprId {
    match rational_literal(pool, a) {
        Some(q) => rat_node(pool, &(q + Rational::from(k))),
        None => pool.add(vec![a, pool.integer(k)]),
    }
}

fn pi_pow(pool: &ExprPool, k: i64) -> ExprId {
    pool.pow(crate::eval::symbols::pi_symbol(pool), pool.integer(k))
}

fn gamma_of(pool: &ExprPool, num: i64, den: i64) -> ExprId {
    pool.func("gamma", vec![pool.rational(num, den)])
}

/// `ζ(n)` for an integer `n` when it has a closed form here.
fn zeta_at_integer(pool: &ExprPool, n: &Integer) -> Option<ExprId> {
    let n = n.to_i64()?;
    if n.abs() > MAX_EXACT_ZETA {
        return None;
    }
    if n <= 0 {
        let v = closed::zeta_nonpositive((-n) as usize).ok()?;
        return Some(rat_node(pool, &v));
    }
    if n % 2 == 0 {
        let c = closed::zeta_even_coefficient((n / 2) as usize).ok()?;
        return Some(mk_mul(pool, vec![rat_node(pool, &c), pi_pow(pool, n)]));
    }
    None
}

fn bessel_canonical(k: Kernel, nu: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let q = rational_literal(pool, nu);
    let x_zero = is_zero_lit(pool, x);
    if let Some(q) = &q {
        let is_int = *q.denom() == 1;
        if *q < 0 {
            if k == Kernel::BesselK {
                return Some(node(k, vec![rat_node(pool, &Rational::from(-q)), x], pool));
            }
            if is_int {
                let n = Rational::from(-q);
                let odd = n.numer().is_odd();
                let base = node(k, vec![rat_node(pool, &n), x], pool);
                return Some(if odd && k != Kernel::BesselI {
                    neg(pool, base)
                } else {
                    base
                });
            }
        }
        if k == Kernel::BesselJ && is_int && (*q == 0 || *q == 1) {
            let name = if *q == 0 { "bessel_j0" } else { "bessel_j1" };
            return Some(j01_canonical(name, x, pool).unwrap_or_else(|| pool.func(name, vec![x])));
        }
        if x_zero && matches!(k, Kernel::BesselJ | Kernel::BesselI) {
            if *q == 0 {
                return Some(pool.integer(1));
            }
            if *q > 0 {
                return Some(pool.integer(0));
            }
        }
        if is_int && matches!(k, Kernel::BesselJ | Kernel::BesselI) {
            if let Some(y) = negated(pool, x) {
                let base = node(k, vec![nu, y], pool);
                return Some(if q.numer().is_odd() {
                    neg(pool, base)
                } else {
                    base
                });
            }
        }
    }
    None
}

/// Special values and parity of the unary `bessel_j0` / `bessel_j1`.
pub(crate) fn j01_canonical(name: &str, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if is_zero_lit(pool, x) {
        return Some(pool.integer(if name == "bessel_j0" { 1 } else { 0 }));
    }
    let y = negated(pool, x)?;
    let base = pool.func(name, vec![y]);
    Some(if name == "bessel_j1" {
        neg(pool, base)
    } else {
        base
    })
}

fn spherical_canonical(k: Kernel, n: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if k == Kernel::SphericalJ && is_zero_lit(pool, x) {
        let n = int_literal(pool, n)?;
        if n == 0 {
            return Some(pool.integer(1));
        }
        if n > 0 {
            return Some(pool.integer(0));
        }
    }
    None
}

fn airy_canonical(k: Kernel, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if !is_zero_lit(pool, x) {
        return None;
    }
    let three = pool.integer(3);
    let inv = |g: ExprId| pool.pow(g, pool.integer(-1));
    // DLMF 9.2.3–9.2.6.
    Some(match k {
        Kernel::AiryAi => mk_mul(
            pool,
            vec![
                pool.pow(three, pool.rational(-2, 3)),
                inv(gamma_of(pool, 2, 3)),
            ],
        ),
        Kernel::AiryAiPrime => mk_mul(
            pool,
            vec![
                pool.integer(-1),
                pool.pow(three, pool.rational(-1, 3)),
                inv(gamma_of(pool, 1, 3)),
            ],
        ),
        Kernel::AiryBi => mk_mul(
            pool,
            vec![
                pool.pow(three, pool.rational(-1, 6)),
                inv(gamma_of(pool, 2, 3)),
            ],
        ),
        Kernel::AiryBiPrime => mk_mul(
            pool,
            vec![
                pool.pow(three, pool.rational(1, 6)),
                inv(gamma_of(pool, 1, 3)),
            ],
        ),
        _ => return None,
    })
}

fn hurwitz_canonical(s: ExprId, a: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let one = pool.integer(1);
    if a == one {
        return Some(node(Kernel::Zeta, vec![s], pool));
    }
    let s_int = int_literal(pool, s).and_then(|v| v.to_i64());
    // ζ(−n, a) = −B_{n+1}(a)/(n+1), a polynomial in any a (DLMF 25.11.14).
    if let Some(sv) = s_int {
        if sv <= 0 && -sv <= MAX_BERNOULLI_POLY {
            let n = (-sv) as usize;
            let c = closed::bernoulli_poly(n + 1).ok()?;
            let scale = Rational::from((-1, n as i64 + 1));
            let terms: Vec<ExprId> = c
                .iter()
                .enumerate()
                .filter(|(_, ck)| **ck != 0)
                .map(|(k, ck)| {
                    let coeff = rat_node(pool, &Rational::from(ck * &scale));
                    let ak = match k {
                        0 => one,
                        1 => a,
                        _ => pool.pow(a, pool.integer(k as i64)),
                    };
                    mk_mul(pool, vec![coeff, ak])
                })
                .collect();
            return Some(mk_add(pool, terms));
        }
    }
    let aq = rational_literal(pool, a)?;
    let sv = s_int?;
    if sv == 1 {
        return None;
    }
    // ζ(s, 1/2) = (2^s − 1) ζ(s) (DLMF 25.11.11).
    if aq == Rational::from((1, 2)) && sv.abs() <= MAX_EXACT_ZETA {
        let two_s = if sv >= 0 {
            Rational::from(Integer::from(1) << sv as u32)
        } else {
            Rational::from((Integer::from(1), Integer::from(1) << (-sv) as u32))
        };
        let factor = two_s - Rational::from(1);
        return Some(mk_mul(
            pool,
            vec![rat_node(pool, &factor), node(Kernel::Zeta, vec![s], pool)],
        ));
    }
    // ζ(s, k) = ζ(s) − Σ_{j<k} j^{−s} for an integer k ≥ 2, s ≥ 2.
    if *aq.denom() == 1 && sv >= 2 {
        let kv = aq.numer().to_i64()?;
        if (2..=MAX_HURWITZ_SHIFT).contains(&kv) {
            let mut corr = Rational::new();
            for j in 1..kv {
                corr += Rational::from((Integer::from(1), Integer::from(j).pow(sv as u32)));
            }
            return Some(mk_add(
                pool,
                vec![node(Kernel::Zeta, vec![s], pool), rat_node(pool, &-corr)],
            ));
        }
    }
    None
}

fn polylog_canonical(s: ExprId, z: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if is_zero_lit(pool, z) {
        return Some(pool.integer(0));
    }
    let one = pool.integer(1);
    let one_minus_z = || pool.add(vec![one, neg(pool, z)]);
    if let Some(n) = int_literal(pool, s).and_then(|v| v.to_i64()) {
        match n {
            2 => return Some(pool.func("dilog", vec![z])),
            1 => return Some(neg(pool, pool.func("log", vec![one_minus_z()]))),
            0 => {
                return Some(mk_mul(
                    pool,
                    vec![z, pool.pow(one_minus_z(), pool.integer(-1))],
                ))
            }
            n if n < 0 && -n <= MAX_POLYLOG_NEG => {
                // Li_{−m}(z) = z·A_m(z)/(1 − z)^{m+1}
                let m = (-n) as usize;
                let row = closed::eulerian_row(m);
                let num: Vec<ExprId> = row
                    .iter()
                    .enumerate()
                    .map(|(k, c)| {
                        let zk = match k {
                            0 => one,
                            1 => z,
                            _ => pool.pow(z, pool.integer(k as i64)),
                        };
                        mk_mul(pool, vec![pool.integer(c.clone()), zk])
                    })
                    .collect();
                return Some(mk_mul(
                    pool,
                    vec![
                        z,
                        mk_add(pool, num),
                        pool.pow(one_minus_z(), pool.integer(-(m as i64) - 1)),
                    ],
                ));
            }
            _ => {}
        }
    }
    let sq = rational_literal(pool, s)?;
    if z == one && sq > 1 {
        // Li_s(1) = ζ(s) for s > 1 (DLMF 25.12.12).
        return Some(node(Kernel::Zeta, vec![s], pool));
    }
    if is_literal_minus_one(pool, z) && *sq.denom() == 1 && sq >= 2 {
        // Li_s(−1) = −(1 − 2^{1−s}) ζ(s) = −η(s) (DLMF 25.12.13 via 25.2.3).
        let sv = sq.numer().to_u32()?;
        if i64::from(sv) > MAX_EXACT_ZETA {
            return None;
        }
        let f =
            Rational::from((Integer::from(1), Integer::from(1) << (sv - 1))) - Rational::from(1);
        return Some(mk_mul(
            pool,
            vec![rat_node(pool, &f), node(Kernel::Zeta, vec![s], pool)],
        ));
    }
    None
}

fn is_literal_minus_one(pool: &ExprPool, e: ExprId) -> bool {
    rational_literal(pool, e).is_some_and(|q| q == -1)
}

/// The canonical form of `kernel(args)`, or `None` when the node is already
/// canonical.
pub(crate) fn canonical(kernel: Kernel, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    if args.len() != kernel.arity() {
        return None;
    }
    match kernel {
        Kernel::BesselJ | Kernel::BesselY | Kernel::BesselI | Kernel::BesselK => {
            bessel_canonical(kernel, args[0], args[1], pool)
        }
        Kernel::SphericalJ | Kernel::SphericalY => {
            spherical_canonical(kernel, args[0], args[1], pool)
        }
        Kernel::AiryAi | Kernel::AiryBi | Kernel::AiryAiPrime | Kernel::AiryBiPrime => {
            airy_canonical(kernel, args[0], pool)
        }
        Kernel::Zeta => zeta_at_integer(pool, &int_literal(pool, args[0])?),
        Kernel::HurwitzZeta => hurwitz_canonical(args[0], args[1], pool),
        Kernel::Polylog => polylog_canonical(args[0], args[1], pool),
        Kernel::ZetaPrime => zeta_prime_canonical(args[0], pool),
        Kernel::HankelH1 | Kernel::HankelH2 => hankel_canonical(kernel, args[0], args[1], pool),
        Kernel::LogGamma => loggamma_canonical(args[0], pool),
    }
}

/// Largest positive integer `n` at which `loggamma(n)` becomes `log((n−1)!)`.
const MAX_EXACT_LOGGAMMA: i64 = 100;

/// Exact values of `log Γ(x)` (F25) where `Γ(x) > 0` is known exactly:
/// `loggamma(1) = loggamma(2) = 0`, `loggamma(n) = log((n−1)!)` for an integer
/// `3 ≤ n ≤ 100`, and `loggamma(m + ½) = log(Γ(m + ½))` for `m ≥ 0`
/// (`loggamma(1/2) = log(π)/2`). On the positive axis the principal branch
/// *is* the real logarithm of `Γ`, so these are exact; at a negative
/// half-integer `Γ < 0` and the principal branch carries an imaginary part,
/// so the node is kept. The poles (`x ∈ {0, −1, …}`) are kept too — the
/// constructor refuses them.
fn loggamma_canonical(x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let q = rational_literal(pool, x)?;
    if q <= 0 {
        return None;
    }
    if *q.denom() == 1 {
        let n = q.numer().to_i64()?;
        if n <= 2 {
            return Some(pool.integer(0));
        }
        if n > MAX_EXACT_LOGGAMMA {
            return None;
        }
        let f = Integer::factorial((n - 1) as u32).complete();
        return Some(pool.func("log", vec![pool.integer(f)]));
    }
    if *q.denom() == 2 {
        let g = super::gamma::gamma_value(x, pool)?;
        return Some(pool.func("log", vec![g]));
    }
    None
}

/// `H^{(1,2)}_{−n} = (−1)ⁿ H^{(1,2)}_n` for an integer `n` (DLMF 10.4.6).
/// (Half-integer orders are rewritten by [`super::hankel`], not here.)
fn hankel_canonical(k: Kernel, nu: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let q = rational_literal(pool, nu)?;
    if *q.denom() != 1 || q >= 0 {
        return None;
    }
    let n = -q;
    let odd = n.numer().is_odd();
    let base = node(k, vec![rat_node(pool, &n), x], pool);
    Some(if odd { neg(pool, base) } else { base })
}

/// Exact values of `ζ′(s)` (DLMF 25.6.9–25.6.12 and the functional equation):
///
/// * `ζ′(0) = −½ ln(2π)`,
/// * `ζ′(−1) = 1/12 − ln A` (`A` the Glaisher–Kinkelin constant),
/// * `ζ′(2) = (π²/6)(γ + ln(2π) − 12 ln A)`,
/// * `ζ′(−2n) = (−1)ⁿ (2n)! ζ(2n+1) / (2 (2π)^{2n})` for `n ≥ 1`.
fn zeta_prime_canonical(s: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let n = int_literal(pool, s)?.to_i64()?;
    let two_pi = || {
        mk_mul(
            pool,
            vec![pool.integer(2), crate::eval::symbols::pi_symbol(pool)],
        )
    };
    let ln = |e: ExprId| pool.func("log", vec![e]);
    match n {
        0 => Some(mk_mul(pool, vec![pool.rational(-1, 2), ln(two_pi())])),
        -1 => Some(mk_add(
            pool,
            vec![pool.rational(1, 12), neg(pool, ln(super::glaisher(pool)))],
        )),
        2 => Some(mk_mul(
            pool,
            vec![
                pool.rational(1, 6),
                pi_pow(pool, 2),
                mk_add(
                    pool,
                    vec![
                        super::euler_gamma(pool),
                        ln(two_pi()),
                        mk_mul(pool, vec![pool.integer(-12), ln(super::glaisher(pool))]),
                    ],
                ),
            ],
        )),
        n if n < 0 && n % 2 == 0 && -n <= MAX_EXACT_ZETA => {
            let m = (-n / 2) as u32; // ζ′(−2m)
            let fact = Integer::factorial(2 * m).complete();
            let sign = if m % 2 == 0 { 1 } else { -1 };
            Some(mk_mul(
                pool,
                vec![
                    rat_node(pool, &Rational::from((fact * sign, Integer::from(2)))),
                    node(Kernel::Zeta, vec![pool.integer(2 * i64::from(m) + 1)], pool),
                    pool.pow(two_pi(), pool.integer(-2 * i64::from(m))),
                ],
            ))
        }
        _ => None,
    }
}

/// `∂/∂x kernel(order, x)` (or `∂/∂x kernel(x)`), the order constant.
pub(crate) fn local_derivative(kernel: Kernel, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let half = pool.rational(1, 2);
    Some(match kernel {
        Kernel::BesselJ
        | Kernel::BesselY
        | Kernel::BesselI
        | Kernel::BesselK
        | Kernel::HankelH1
        | Kernel::HankelH2 => {
            let (nu, x) = (args[0], args[1]);
            let lo = node(kernel, vec![shift(pool, nu, -1), x], pool);
            let hi = node(kernel, vec![shift(pool, nu, 1), x], pool);
            match kernel {
                // H = J ± iY obeys the same recurrence (DLMF 10.6.1).
                Kernel::BesselJ | Kernel::BesselY | Kernel::HankelH1 | Kernel::HankelH2 => {
                    mk_mul(pool, vec![half, mk_add(pool, vec![lo, neg(pool, hi)])])
                }
                Kernel::BesselI => mk_mul(pool, vec![half, mk_add(pool, vec![lo, hi])]),
                _ => mk_mul(pool, vec![pool.rational(-1, 2), mk_add(pool, vec![lo, hi])]),
            }
        }
        Kernel::SphericalJ | Kernel::SphericalY => {
            let (n, x) = (args[0], args[1]);
            let up = node(kernel, vec![shift(pool, n, 1), x], pool);
            let same = pool.func(kernel.name(), vec![n, x]);
            mk_add(
                pool,
                vec![
                    neg(pool, up),
                    mk_mul(pool, vec![n, pool.pow(x, pool.integer(-1)), same]),
                ],
            )
        }
        Kernel::AiryAi => pool.func(Kernel::AiryAiPrime.name(), vec![args[0]]),
        Kernel::AiryBi => pool.func(Kernel::AiryBiPrime.name(), vec![args[0]]),
        Kernel::AiryAiPrime => mk_mul(
            pool,
            vec![args[0], pool.func(Kernel::AiryAi.name(), vec![args[0]])],
        ),
        Kernel::AiryBiPrime => mk_mul(
            pool,
            vec![args[0], pool.func(Kernel::AiryBi.name(), vec![args[0]])],
        ),
        Kernel::HurwitzZeta => {
            let (s, a) = (args[0], args[1]);
            mk_mul(
                pool,
                vec![
                    neg(pool, s),
                    node(Kernel::HurwitzZeta, vec![shift(pool, s, 1), a], pool),
                ],
            )
        }
        Kernel::Polylog => {
            let (s, z) = (args[0], args[1]);
            mk_mul(
                pool,
                vec![
                    node(Kernel::Polylog, vec![shift(pool, s, -1), z], pool),
                    pool.pow(z, pool.integer(-1)),
                ],
            )
        }
        // ζ′(s): the derivative in s of ζ is the node ζ′.
        Kernel::Zeta => pool.func(Kernel::ZetaPrime.name(), vec![args[0]]),
        // ζ″ has no node (a higher derivative is refused, not dropped).
        Kernel::ZetaPrime => return None,
        // d/dx log Γ(x) = ψ(x) (DLMF 5.2.2), on every branch.
        Kernel::LogGamma => pool.func("digamma", vec![args[0]]),
    })
}

impl Primitive for SpecFunPrimitive {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        let a: Vec<String> = args.iter().map(|&a| pool.display(a).to_string()).collect();
        format!("{}({})", self.0.name(), a.join(", "))
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        canonical(self.0, args, pool)
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let k = self.0;
        if args.len() != k.arity() {
            return None;
        }
        let (consts, x) = args.split_at(args.len() - 1);
        let x = x[0];
        if consts
            .iter()
            .any(|&c| crate::kernel::subs::mentions_var(c, wrt, pool))
        {
            // ∂/∂ν, ∂/∂s have no closed form: decline rather than drop.
            return None;
        }
        let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
        if is_zero_lit(pool, dx) {
            return Some(pool.integer(0));
        }
        // `zeta` has no derivative rule (ζ′ has no closed form): `None` here.
        let local = local_derivative(k, args, pool)?;
        Some(mk_mul(pool, vec![local, dx]))
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        super::numeric::eval_f64(self.0, args)
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        super::numeric::eval_ball(self.0, args)
    }
}
