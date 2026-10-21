//! `gammasimp` (F25): rewrite products and quotients of `Γ` with the exact
//! identities of DLMF 5.5.
//!
//! | rule | identity | DLMF |
//! |---|---|---|
//! | recurrence | `Γ(b + n) = Γ(b)·b(b+1)⋯(b+n−1)` for an integer `n` | 5.5.1 |
//! | reflection | `Γ(a)·Γ(1 − a) = π / sin(πa)` | 5.5.3 |
//! | duplication | `Γ(a)·Γ(a + ½) = 2^{1−2a} √π Γ(2a)` | 5.5.5 |
//!
//! The rules fire on the `Γ` factors of one product (a `Mul`, or a lone
//! `Pow`), each factor `Γ(arg)^e` with an integer `e`:
//!
//! * **recurrence** pairs two factors whose arguments differ by an integer
//!   `n` (`|n| ≤` [`MAX_SHIFT`]) and whose exponents have opposite signs —
//!   `Γ(x+3)/Γ(x) → x(x+1)(x+2)`, `Γ(x)/Γ(x+2) → 1/(x(x+1))`. Factors of the
//!   same sign are left alone (SymPy's `x·Γ(x)²` for `Γ(x)Γ(x+1)` is not a
//!   simplification);
//! * **reflection** pairs two factors with the same exponent whose arguments
//!   sum to an integer `m`: the second is first moved to `Γ(1 − a)` by the
//!   recurrence, then `Γ(a)Γ(1−a)` becomes `π/sin(πa)`;
//! * **duplication** pairs two factors with the same exponent whose arguments
//!   differ by `½ + k` for an integer `k`, likewise shifted first.
//!
//! # Where the identities hold
//!
//! Reflection and duplication are identities of meromorphic functions: both
//! sides have the same poles (`a ∈ ℤ`, resp. `2a ∈ ℤ_{≤0}`) and agree
//! everywhere else. The recurrence is too, *as written with the `Γ`
//! factor kept*; once the `Γ(b)` factors cancel (`Γ(x+n)/Γ(x)`) the result
//! is the polynomial that continuously extends the quotient across
//! `b ∈ {0, −1, …}`, where the input itself is `∞/∞` and undefined. That is
//! the usual convention (SymPy's `gammasimp`, Mathematica's
//! `FunctionExpand`); a caller that needs the poles excluded must state
//! `b ∉ ℤ_{≤0}` itself.
//!
//! `beta(a, b)` is first written `Γ(a)Γ(b)/Γ(a+b)` (DLMF 5.12.1), as
//! SymPy's `gammasimp` does, so `beta(x, 1)` becomes `1/x`.
//!
//! Nothing else is touched: no numeric evaluation, no float, no change to
//! `Γ` factors with a non-integer exponent (`uppergamma`/`lowergamma` are
//! left alone).

use rug::Rational;

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::orthopoly::{mk_add, mk_mul, rational_literal};

use super::SpecialFunctionError;

/// Largest integer shift the recurrence expands into a product of linear
/// factors (`Γ(x + 64)/Γ(x)` is a degree-64 polynomial; larger shifts are
/// left alone).
pub const MAX_SHIFT: i64 = 64;

/// Upper bound on rule applications in one product (each application removes
/// at least one `Γ` factor, so this is never reached on a real input).
const MAX_STEPS: usize = 256;

/// Rewrite products of `Γ` with the recurrence, reflection and duplication
/// formulas (module docs). Returns the input unchanged when no rule applies.
///
/// # Errors
///
/// `E-BUDGET-*` when the active budget runs out.
pub fn gammasimp(expr: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    let mut memo = std::collections::HashMap::new();
    let out = walk(expr, pool, &mut memo)?;
    if out == expr {
        return Ok(expr);
    }
    Ok(crate::simplify::simplify(out, pool).value)
}

fn walk(
    expr: ExprId,
    pool: &ExprPool,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> Result<ExprId, SpecialFunctionError> {
    if let Some(&r) = memo.get(&expr) {
        return Ok(r);
    }
    crate::budget::check()?;
    let out = match pool.get(expr) {
        ExprData::Add(args) => {
            let a = args
                .iter()
                .map(|&x| walk(x, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            match common_base(&a, pool) {
                Some(e) => e,
                None if a == args => expr,
                None => pool.add(a),
            }
        }
        ExprData::Mul(args) => {
            let mut a = Vec::with_capacity(args.len());
            for &x in &args {
                let w = walk(x, pool, memo)?;
                // A rewritten `beta` is itself a product: flatten it so its
                // Γ factors meet the others.
                match pool.get(w) {
                    ExprData::Mul(inner) if w != x => a.extend(inner),
                    _ => a.push(w),
                }
            }
            match combine_product(&a, pool)? {
                Some(e) => e,
                None if a == args => expr,
                None => pool.mul(a),
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (walk(base, pool, memo)?, walk(exp, pool, memo)?);
            if (b, e) == (base, exp) {
                expr
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => {
            let a = args
                .iter()
                .map(|&x| walk(x, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            if name == "beta" && a.len() == 2 {
                // B(a, b) = Γ(a)Γ(b)/Γ(a+b) (DLMF 5.12.1), an identity of
                // meromorphic functions; the product rules then see it.
                let sum = pool.add(vec![a[0], a[1]]);
                let ratio = pool.mul(vec![
                    pool.func("gamma", vec![a[0]]),
                    pool.func("gamma", vec![a[1]]),
                    pool.pow(pool.func("gamma", vec![sum]), pool.integer(-1)),
                ]);
                let r = crate::simplify::simplify(ratio, pool).value;
                let out = match pool.get(r) {
                    ExprData::Mul(fs) => combine_product(&fs, pool)?.unwrap_or(r),
                    _ => r,
                };
                memo.insert(expr, out);
                return Ok(out);
            }
            if a == args {
                expr
            } else {
                pool.func(name, a)
            }
        }
        _ => expr,
    };
    memo.insert(expr, out);
    Ok(out)
}

/// In a sum, write every `Γ(b + k)` (`k` a positive integer `≤` [`MAX_SHIFT`])
/// through the smallest argument `b` of its integer class that occurs in the
/// sum, `Γ(b + k) = Γ(b)·b(b+1)⋯(b+k−1)`, so that like terms meet:
/// `Γ(x+1) − x·Γ(x)` becomes `x·Γ(x) − x·Γ(x) = 0`. `None` when no class has
/// two distinct arguments.
fn common_base(summands: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let mut args: Vec<ExprId> = Vec::new();
    for &s in summands {
        let factors = match pool.get(s) {
            ExprData::Mul(xs) => xs,
            _ => vec![s],
        };
        for f in factors {
            if let Some((a, _)) = gamma_factor(f, pool) {
                if !args.contains(&a) {
                    args.push(a);
                }
            }
        }
    }
    if args.len() < 2 {
        return None;
    }
    // For each argument, the smallest argument it differs from by a positive
    // integer.
    let mut map = std::collections::HashMap::new();
    for &a in &args {
        let mut best: Option<(i64, ExprId)> = None;
        for &b in &args {
            if a == b {
                continue;
            }
            let Some(d) = rational_difference(a, b, pool) else {
                continue;
            };
            if *d.denom() != 1 {
                continue;
            }
            let Some(d) = d.numer().to_i64().filter(|d| (1..=MAX_SHIFT).contains(d)) else {
                continue;
            };
            if best.map_or(true, |(bd, _)| d > bd) {
                best = Some((d, b));
            }
        }
        if let Some((d, b)) = best {
            map.insert(
                pool.func("gamma", vec![a]),
                mk_mul(
                    pool,
                    vec![pool.func("gamma", vec![b]), pochhammer(b, d, pool)],
                ),
            );
        }
    }
    if map.is_empty() {
        return None;
    }
    let sum = crate::kernel::subs(pool.add(summands.to_vec()), &map, pool);
    Some(sum)
}

/// `Γ(arg)^e` with an integer `e ≠ 0`, if `f` is one.
fn gamma_factor(f: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
    match pool.get(f) {
        ExprData::Func { name, args } if name == "gamma" && args.len() == 1 => Some((args[0], 1)),
        ExprData::Pow { base, exp } => {
            let e = rational_literal(pool, exp)?;
            if *e.denom() != 1 {
                return None;
            }
            let e = e.numer().to_i64().filter(|&v| v != 0)?;
            match pool.get(base) {
                ExprData::Func { name, args } if name == "gamma" && args.len() == 1 => {
                    Some((args[0], e))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// `a − b` as a rational literal, when it is one.
fn rational_difference(a: ExprId, b: ExprId, pool: &ExprPool) -> Option<Rational> {
    let d = mk_add(pool, vec![a, mk_mul(pool, vec![pool.integer(-1), b])]);
    rational_literal(pool, crate::simplify::simplify(d, pool).value)
}

fn shifted(a: ExprId, k: i64, pool: &ExprPool) -> ExprId {
    if k == 0 {
        return a;
    }
    crate::simplify::simplify(pool.add(vec![a, pool.integer(k)]), pool).value
}

/// `Γ(b + n)/Γ(b)` for an integer `n`: `b(b+1)⋯(b+n−1)` for `n > 0`,
/// `1/((b−1)(b−2)⋯(b+n))` for `n < 0`.
fn pochhammer(b: ExprId, n: i64, pool: &ExprPool) -> ExprId {
    if n >= 0 {
        mk_mul(pool, (0..n).map(|k| shifted(b, k, pool)).collect())
    } else {
        let den = mk_mul(pool, (1..=-n).map(|k| shifted(b, -k, pool)).collect());
        pool.pow(den, pool.integer(-1))
    }
}

fn pow_i(base: ExprId, e: i64, pool: &ExprPool) -> ExprId {
    if e == 1 {
        base
    } else {
        pool.pow(base, pool.integer(e))
    }
}

/// Size of an expression tree (a tie-breaker for which argument to keep).
fn weight(e: ExprId, pool: &ExprPool) -> usize {
    match pool.get(e) {
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            1 + xs.iter().map(|&x| weight(x, pool)).sum::<usize>()
        }
        ExprData::Pow { base, exp } => 1 + weight(base, pool) + weight(exp, pool),
        ExprData::Func { args, .. } => 1 + args.iter().map(|&x| weight(x, pool)).sum::<usize>(),
        _ => 1,
    }
}

/// Apply the rules to the factors of one product; `None` if nothing fired.
fn combine_product(
    factors: &[ExprId],
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    let mut gammas: Vec<(ExprId, i64)> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for &f in factors {
        match gamma_factor(f, pool) {
            Some((arg, e)) => match gammas.iter_mut().find(|(a, _)| *a == arg) {
                Some(slot) => slot.1 += e,
                None => gammas.push((arg, e)),
            },
            None => rest.push(f),
        }
    }
    if gammas.len() < 2 {
        return Ok(None);
    }
    let mut fired = false;
    for _ in 0..MAX_STEPS {
        crate::budget::check()?;
        gammas.retain(|&(_, e)| e != 0);
        match step(&gammas, pool) {
            Some((i, j, new_gammas, extra)) => {
                fired = true;
                let (hi, lo) = if i > j { (i, j) } else { (j, i) };
                gammas.remove(hi);
                gammas.remove(lo);
                rest.extend(extra);
                for (arg, e) in new_gammas {
                    match gammas.iter_mut().find(|(a, _)| *a == arg) {
                        Some(slot) => slot.1 += e,
                        None => gammas.push((arg, e)),
                    }
                }
            }
            None => break,
        }
    }
    if !fired {
        return Ok(None);
    }
    for (arg, e) in gammas {
        if e != 0 {
            rest.push(pow_i(pool.func("gamma", vec![arg]), e, pool));
        }
    }
    Ok(Some(pool.mul(rest)))
}

type Step = (usize, usize, Vec<(ExprId, i64)>, Vec<ExprId>);

/// One rule application on the `Γ` factor list: the indices of the two
/// factors consumed, the `Γ` factors produced, and the other factors
/// produced.
fn step(gammas: &[(ExprId, i64)], pool: &ExprPool) -> Option<Step> {
    let pi = crate::eval::symbols::pi_symbol(pool);
    let n = gammas.len();
    // Reflection and duplication first (they remove two Γ's at once).
    for i in 0..n {
        for j in (i + 1)..n {
            let ((a, ea), (b, eb)) = (gammas[i], gammas[j]);
            if ea != eb {
                continue;
            }
            let e = ea;
            // Keep the simpler argument as `a`.
            let (a, b) = if weight(b, pool) < weight(a, pool) {
                (b, a)
            } else {
                (a, b)
            };
            // Reflection: a + b = m ∈ ℤ.
            let sum = mk_add(pool, vec![a, b]);
            if let Some(m) = rational_literal(pool, crate::simplify::simplify(sum, pool).value) {
                if *m.denom() == 1 {
                    if let Some(m) = m.numer().to_i64().filter(|m| (m - 1).abs() <= MAX_SHIFT) {
                        // Γ(b) = Γ(1 − a + (m − 1)) = Γ(1 − a)·(1 − a)_{m−1}.
                        let one_minus_a = crate::simplify::simplify(
                            mk_add(
                                pool,
                                vec![pool.integer(1), mk_mul(pool, vec![pool.integer(-1), a])],
                            ),
                            pool,
                        )
                        .value;
                        let poch = pochhammer(one_minus_a, m - 1, pool);
                        let pi_a = mk_mul(pool, vec![pi, a]);
                        let refl = mk_mul(
                            pool,
                            vec![pi, pool.pow(pool.func("sin", vec![pi_a]), pool.integer(-1))],
                        );
                        return Some((
                            i,
                            j,
                            vec![],
                            vec![pow_i(refl, e, pool), pow_i(poch, e, pool)],
                        ));
                    }
                }
            }
            // Duplication: b − a = ½ + k.
            for (lo, hi) in [(a, b), (b, a)] {
                let Some(d) = rational_difference(hi, lo, pool) else {
                    continue;
                };
                if *d.denom() != 2 {
                    continue;
                }
                let k = &d - Rational::from((1, 2));
                let Some(k) = k.numer().to_i64().filter(|k| k.abs() <= MAX_SHIFT) else {
                    continue;
                };
                // Γ(hi) = Γ(lo + ½ + k) = Γ(lo + ½)·(lo + ½)_k.
                let lo_half =
                    crate::simplify::simplify(pool.add(vec![lo, pool.rational(1, 2)]), pool).value;
                let poch = pochhammer(lo_half, k, pool);
                // Γ(lo)Γ(lo + ½) = 2^{1 − 2 lo} √π Γ(2 lo).
                let two_lo =
                    crate::simplify::simplify(mk_mul(pool, vec![pool.integer(2), lo]), pool).value;
                let power = mk_add(
                    pool,
                    vec![pool.integer(1), mk_mul(pool, vec![pool.integer(-2), lo])],
                );
                let factor = mk_mul(
                    pool,
                    vec![
                        pool.pow(pool.integer(2), power),
                        pool.pow(pi, pool.rational(1, 2)),
                    ],
                );
                return Some((
                    i,
                    j,
                    vec![(two_lo, e)],
                    vec![pow_i(factor, e, pool), pow_i(poch, e, pool)],
                ));
            }
        }
    }
    // Recurrence: integer difference, opposite signs.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let ((a, ea), (b, eb)) = (gammas[i], gammas[j]);
            if ea.signum() == eb.signum() || ea.abs() > eb.abs() {
                continue;
            }
            // Rewrite Γ(a) (the factor with the smaller |exponent|) through Γ(b).
            let Some(d) = rational_difference(a, b, pool) else {
                continue;
            };
            if *d.denom() != 1 {
                continue;
            }
            let Some(d) = d.numer().to_i64().filter(|d| d.abs() <= MAX_SHIFT) else {
                continue;
            };
            let poch = pochhammer(b, d, pool);
            return Some((i, j, vec![(b, ea + eb)], vec![pow_i(poch, ea, pool)]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::simplify::zero::zero_test;

    fn g(p: &ExprPool, a: ExprId) -> ExprId {
        p.func("gamma", vec![a])
    }

    fn proved_zero(e: ExprId, p: &ExprPool) -> bool {
        zero_test(e, p, None).status == crate::simplify::zero::ZeroStatus::ProvedZero
    }

    #[test]
    fn reflection() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let pi = crate::eval::symbols::pi_symbol(&p);
        let one_minus_x = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), x])]);
        let e = p.mul(vec![g(&p, x), g(&p, one_minus_x)]);
        let out = gammasimp(e, &p).unwrap();
        let want = p.mul(vec![
            pi,
            p.pow(p.func("sin", vec![p.mul(vec![pi, x])]), p.integer(-1)),
        ]);
        let diff = p.add(vec![out, p.mul(vec![p.integer(-1), want])]);
        assert!(proved_zero(diff, &p), "{}", p.display(out));
        // And the zero test sees it from the raw identity.
        let raw = p.add(vec![e, p.mul(vec![p.integer(-1), want])]);
        assert!(proved_zero(raw, &p));
    }

    #[test]
    fn recurrence_ratio_is_a_rising_factorial() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let x3 = p.add(vec![x, p.integer(3)]);
        let e = p.mul(vec![g(&p, x3), p.pow(g(&p, x), p.integer(-1))]);
        let out = gammasimp(e, &p).unwrap();
        let want = p.mul(vec![
            x,
            p.add(vec![x, p.integer(1)]),
            p.add(vec![x, p.integer(2)]),
        ]);
        let diff = p.add(vec![out, p.mul(vec![p.integer(-1), want])]);
        assert!(proved_zero(diff, &p), "{}", p.display(out));
        assert!(
            !p.display(out).to_string().contains("gamma"),
            "{}",
            p.display(out)
        );
    }

    #[test]
    fn duplication() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let xh = p.add(vec![x, p.rational(1, 2)]);
        let e = p.mul(vec![g(&p, x), g(&p, xh)]);
        let out = gammasimp(e, &p).unwrap();
        // One Γ left, of the doubled argument.
        let shown = p.display(out).to_string();
        assert_eq!(shown.matches("gamma").count(), 1, "{shown}");
        for v in [0.3, 1.7, 4.25] {
            let mut env = std::collections::HashMap::new();
            env.insert(x, v);
            let a = crate::jit::eval_interp(e, &env, &p).unwrap();
            let b = crate::jit::eval_interp(out, &env, &p).unwrap();
            assert!((a - b).abs() <= 1e-12 * a.abs(), "{v}: {a} vs {b}");
        }
    }

    #[test]
    fn beta_is_written_through_gamma() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let y = p.symbol("y", Domain::Positive);
        // B(x, 1) = 1/x
        let out = gammasimp(p.func("beta", vec![x, p.integer(1)]), &p).unwrap();
        let diff = p.add(vec![
            out,
            p.mul(vec![p.integer(-1), p.pow(x, p.integer(-1))]),
        ]);
        assert!(proved_zero(diff, &p), "{}", p.display(out));
        // B(x, y)·Γ(x+y) − Γ(x)Γ(y) = 0, and the zero test sees it.
        let lhs = p.mul(vec![p.func("beta", vec![x, y]), g(&p, p.add(vec![x, y]))]);
        let raw = p.add(vec![lhs, p.mul(vec![p.integer(-1), g(&p, x), g(&p, y)])]);
        assert!(proved_zero(raw, &p));
        // uppergamma is not a Γ product: untouched.
        let ug = p.func("uppergamma", vec![x, y]);
        assert_eq!(gammasimp(ug, &p).unwrap(), ug);
    }

    #[test]
    fn same_sign_products_and_non_gamma_are_unchanged() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.mul(vec![g(&p, x), g(&p, p.add(vec![x, p.integer(1)]))]);
        assert_eq!(gammasimp(e, &p).unwrap(), e);
        let s = p.func("sin", vec![x]);
        assert_eq!(gammasimp(s, &p).unwrap(), s);
    }
}
