//! `RootOfFamily(F, λ, n, branch)` — the `n`-th root of a transcendental
//! equation `F(λ, n) = 0`, with certified enclosures (plan decision D5).
//!
//! Eigenvalue problems with Robin conditions produce families with no closed
//! form: `tan(λL) = −λ/h`, `J₀(λ) = 0`.  `RootOf` (an algebraic number with a
//! fixed integer index) cannot express them.  This head can:
//!
//! ```text
//! Func("RootOfFamily", [F, λ, n, branch])
//! ```
//!
//! * `λ` is a symbol **bound** in `F` (substitution never reaches it);
//! * `n` is the index — a symbol for the whole family (`λₙ`) or a positive
//!   integer for one member; `F` may itself depend on `n`;
//! * `branch` selects the counting convention: `0` — the positive roots of
//!   `F(·, n)` in increasing order, `n = 1, 2, …`; `1` — the negative roots in
//!   decreasing order.  Other values are reserved (`E-PDE-030`).
//!
//! # Certified value
//!
//! [`root_family_enclosure`] isolates the roots on `(0, ∞)` left to right with
//! ball arithmetic: an interval is root-free when the enclosure of `F` on it
//! excludes 0; it holds exactly one root when the enclosure of `∂F/∂λ`
//! excludes 0 (so `F` is monotone there) and `F` has opposite certified signs
//! at its ends; otherwise it is bisected.  Every piece of `(0, λₙ]` is so
//! certified, so the count is exact, and the `n`-th root is then refined by
//! bisection to the requested precision.  Anything not certifiable within the
//! budget — a pole of `F` (write `tan(λL) = −λ/h` as
//! `h·sin(λL) + λ·cos(λL) = 0`), a double root, a root at an interval end that
//! never separates — is refused with `E-PDE-031`; nothing is ever guessed.
//!
//! A root at `λ = 0` (common: `sin(λ) = 0`) is excluded from the positive
//! count when `F(0, n)` simplifies to exactly 0 and the first derivative
//! `∂ᵐF/∂λᵐ` that does not vanish identically at 0 is certified nonzero on a
//! neighbourhood `[0, δ]` (`J_ν(λ)` has a root of order `ν` there).
//!
//! `diff` treats the node by implicit differentiation,
//! `∂λₙ/∂p = −F_p/F_λ` at `λ = λₙ`, so a family depending on a parameter
//! (`L`, `h`) differentiates exactly.

use std::collections::HashMap;

use rug::Float;

use crate::ball::{ArbBall, IntervalEval};
use crate::kernel::{ExprData, ExprId, ExprPool};

use super::error::PdeError;
use super::jet::symbols_of;

/// Head of a `RootOfFamily` node.
pub const ROOT_OF_FAMILY: &str = "RootOfFamily";

/// Largest number of interval evaluations one enclosure may spend.
const MAX_EVALS: usize = 400_000;
/// The search for the `n`-th positive root gives up past this abscissa.
const MAX_ABSCISSA: f64 = 1.0e6;
/// Deepest bisection of one scanning step.
const MAX_DEPTH: u32 = 48;
/// Highest multiplicity of a root at `λ = 0` the scanner steps over.
const MAX_ROOT_ORDER_AT_ZERO: usize = 12;

fn is_symbol(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Symbol { .. }))
}

fn int_literal(e: ExprId, pool: &ExprPool) -> Option<rug::Integer> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(n.0.clone()),
        _ => None,
    })
}

/// `RootOfFamily(F, λ, n, branch)`.
///
/// # Errors
///
/// `E-PDE-030` when `λ` is not a symbol, `n` is neither a symbol nor a
/// positive integer, or `branch` is not `0` or `1`.
pub fn root_of_family(
    f: ExprId,
    lam: ExprId,
    n: ExprId,
    branch: i64,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    if !is_symbol(lam, pool) {
        return Err(PdeError::MalformedRootFamily(format!(
            "the root variable {} is not a symbol",
            pool.display(lam)
        )));
    }
    let n_ok = is_symbol(n, pool) || int_literal(n, pool).is_some_and(|k| k >= 1);
    if !n_ok || n == lam {
        return Err(PdeError::MalformedRootFamily(format!(
            "the index {} must be a symbol other than the root variable, or an integer ≥ 1",
            pool.display(n)
        )));
    }
    if !(0..=1).contains(&branch) {
        return Err(PdeError::MalformedRootFamily(format!(
            "branch {branch}: 0 (positive roots, ascending) and 1 (negative roots, descending) are defined"
        )));
    }
    Ok(pool.func(ROOT_OF_FAMILY, vec![f, lam, n, pool.integer(branch)]))
}

/// `(F, λ, n, branch)` of a `RootOfFamily` node.
pub fn root_family_parts(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId, i64)> {
    let args = pool.with(e, |d| match d {
        ExprData::Func { name, args } if name == ROOT_OF_FAMILY && args.len() == 4 => {
            Some(args.clone())
        }
        _ => None,
    })?;
    if !is_symbol(args[1], pool) {
        return None;
    }
    let b = int_literal(args[3], pool)?.to_i64()?;
    Some((args[0], args[1], args[2], b))
}

/// The family member `RootOfFamily(F, λ, k, branch)` for a concrete index.
///
/// # Errors
///
/// `E-PDE-030` when `e` is not a `RootOfFamily` whose index is a symbol, or
/// `k < 1`.
pub fn member(e: ExprId, k: i64, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let (f, lam, n, b) = root_family_parts(e, pool).ok_or_else(|| {
        PdeError::MalformedRootFamily(format!("{} is not a RootOfFamily", pool.display(e)))
    })?;
    if !is_symbol(n, pool) || k < 1 {
        return Err(PdeError::MalformedRootFamily(
            "member() needs a family with a symbolic index and k ≥ 1".into(),
        ));
    }
    let mut m = HashMap::new();
    m.insert(n, pool.integer(k));
    let fk = crate::kernel::subs::subs(f, &m, pool);
    root_of_family(fk, lam, pool.integer(k), b, pool)
}

struct Scanner<'a> {
    f: ExprId,
    df: ExprId,
    lam: ExprId,
    pool: &'a ExprPool,
    prec: u32,
    evals: usize,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Sgn {
    Neg,
    Pos,
    Unknown,
}

impl Scanner<'_> {
    fn ball(&self, lo: &Float, hi: &Float) -> ArbBall {
        let p = self.prec;
        let mid = Float::with_val(p + 8, lo + hi) / 2u32;
        let rad = Float::with_val_round(p + 8, hi - &mid, rug::float::Round::Up).0;
        ArbBall {
            mid: Float::with_val(p + 8, &mid),
            rad: Float::with_val_round(p, &rad, rug::float::Round::Up).0
                + Float::with_val(p, Float::i_exp(1, -(p as i32))),
            prec: p,
        }
    }

    fn eval(&mut self, e: ExprId, b: ArbBall) -> Result<Option<ArbBall>, PdeError> {
        self.evals += 1;
        if self.evals > MAX_EVALS {
            return Err(PdeError::RootFamilyNotCertified(
                "the isolation budget was exhausted".into(),
            ));
        }
        let mut ie = IntervalEval::new(self.prec);
        ie.bind(self.lam, b);
        Ok(ie.eval(e, self.pool))
    }

    fn excludes_zero(b: &Option<ArbBall>) -> bool {
        match b {
            Some(b) => {
                let (lo, hi) = (b.lo(), b.hi());
                lo.is_finite() && hi.is_finite() && (lo > 0 || hi < 0)
            }
            None => false,
        }
    }

    fn sign_at(&mut self, x: &Float) -> Result<Sgn, PdeError> {
        let b = ArbBall {
            mid: Float::with_val(self.prec + 8, x),
            rad: Float::new(self.prec),
            prec: self.prec,
        };
        Ok(match self.eval(self.f, b)? {
            Some(v) if v.lo().is_finite() && v.lo() > 0 => Sgn::Pos,
            Some(v) if v.hi().is_finite() && v.hi() < 0 => Sgn::Neg,
            _ => Sgn::Unknown,
        })
    }

    /// Isolate the roots in `[lo, hi]`, appending isolating intervals in
    /// increasing order, until `want` roots are known in total.
    fn isolate(
        &mut self,
        lo: Float,
        hi: Float,
        depth: u32,
        out: &mut Vec<(Float, Float)>,
        want: usize,
    ) -> Result<(), PdeError> {
        if out.len() >= want {
            return Ok(());
        }
        let b = self.ball(&lo, &hi);
        let fv = self.eval(self.f, b.clone())?;
        if Self::excludes_zero(&fv) {
            return Ok(());
        }
        let dv = self.eval(self.df, b)?;
        if Self::excludes_zero(&dv) {
            let (sl, sh) = (self.sign_at(&lo)?, self.sign_at(&hi)?);
            match (sl, sh) {
                (Sgn::Pos, Sgn::Neg) | (Sgn::Neg, Sgn::Pos) => {
                    out.push((lo, hi));
                    return Ok(());
                }
                (Sgn::Pos, Sgn::Pos) | (Sgn::Neg, Sgn::Neg) => return Ok(()),
                _ => {}
            }
        }
        if depth >= MAX_DEPTH {
            return Err(PdeError::RootFamilyNotCertified(format!(
                "could not certify [{}, {}] (a pole, a multiple root, or a root at a split point)",
                lo.to_f64(),
                hi.to_f64()
            )));
        }
        let mid = Float::with_val(lo.prec().max(hi.prec()) + 2, &lo + &hi) / 2u32;
        self.isolate(lo, mid.clone(), depth + 1, out, want)?;
        self.isolate(mid, hi, depth + 1, out, want)
    }
}

/// A certified enclosure `[lo, hi]` of a `RootOfFamily` with a concrete
/// index, of width at most `2^-bits` relative to the root (absolute near 0).
///
/// # Errors
///
/// `E-PDE-030` when `e` is not a `RootOfFamily`, its index is not an integer,
/// or `F` has free symbols other than `λ` (bind parameters first);
/// `E-PDE-031` when the root cannot be certified.
pub fn root_family_enclosure(
    e: ExprId,
    bits: u32,
    pool: &ExprPool,
) -> Result<(Float, Float), PdeError> {
    let (f, lam, n, branch) = root_family_parts(e, pool).ok_or_else(|| {
        PdeError::MalformedRootFamily(format!("{} is not a RootOfFamily", pool.display(e)))
    })?;
    let k = int_literal(n, pool)
        .and_then(|k| k.to_usize())
        .filter(|&k| k >= 1)
        .ok_or_else(|| {
            PdeError::MalformedRootFamily(format!(
                "the index {} is not a positive integer; take a member first",
                pool.display(n)
            ))
        })?;
    // λ → −λ for the negative branch.
    let f = if branch == 1 {
        let mut m = HashMap::new();
        m.insert(lam, pool.mul(vec![pool.integer(-1), lam]));
        crate::kernel::subs::subs(f, &m, pool)
    } else {
        f
    };
    let free: Vec<ExprId> = symbols_of(f, pool)
        .into_iter()
        .filter(|&s| s != lam && !is_constant_symbol(s, pool))
        .collect();
    if !free.is_empty() {
        return Err(PdeError::MalformedRootFamily(format!(
            "F has free symbols {}; substitute numbers for the parameters first",
            free.iter()
                .map(|&s| pool.display(s).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let f = regular_bessel_form(f, pool);
    let df = regular_bessel_form(
        crate::diff::diff(f, lam, pool)
            .map_err(|e| PdeError::RootFamilyNotCertified(format!("∂F/∂λ: {e}")))?
            .value,
        pool,
    );
    let prec = bits.max(64) + 64;
    let mut sc = Scanner {
        f,
        df,
        lam,
        pool,
        prec,
        evals: 0,
    };
    let fl = |v: f64| Float::with_val(prec, v);
    // The left end: start at 0, or just right of a simple root at 0.
    let mut start = fl(0.0);
    let mut m0 = HashMap::new();
    m0.insert(lam, pool.integer(0));
    let f0 = crate::simplify::engine::simplify(crate::kernel::subs::subs(f, &m0, pool), pool).value;
    if f0 == pool.integer(0) {
        // A root of multiplicity m at 0 (J_ν(λ) has one of order ν): find
        // the first derivative F^(m) not vanishing identically at 0 and
        // certify it nonzero on [0, δ]; with F^(j)(0) = 0 for j < m, each
        // lower derivative is then strictly monotone from 0, so F has no
        // root in (0, δ].
        let mut dm = df;
        let mut ok = false;
        let mut delta = fl(0.25);
        'order: for _m in 1..=MAX_ROOT_ORDER_AT_ZERO {
            delta = fl(0.25);
            for _ in 0..24 {
                let b = sc.ball(&fl(0.0), &delta);
                let dv = sc.eval(dm, b)?;
                if Scanner::excludes_zero(&dv) {
                    ok = true;
                    break 'order;
                }
                delta /= 2u32;
            }
            let at0 =
                crate::simplify::engine::simplify(crate::kernel::subs::subs(dm, &m0, pool), pool)
                    .value;
            // Exactly zero at 0: symbolically, or as an exact (radius-0)
            // Arb ball — `J₃(0)` is returned as the exact 0.
            let exact_zero = at0 == pool.integer(0) || {
                let z = ArbBall {
                    mid: Float::with_val(sc.prec + 8, 0),
                    rad: Float::new(sc.prec),
                    prec: sc.prec,
                };
                matches!(sc.eval(dm, z)?, Some(v) if v.is_exact() && v.mid == 0)
            };
            if !exact_zero {
                break;
            }
            dm = regular_bessel_form(
                crate::diff::diff(dm, lam, pool)
                    .map_err(|e| PdeError::RootFamilyNotCertified(format!("∂ᵐF/∂λᵐ: {e}")))?
                    .value,
                pool,
            );
        }
        if !ok {
            return Err(PdeError::RootFamilyNotCertified(
                "F vanishes at λ = 0 and no derivative could be certified nonzero near 0 \
                 (a root of high multiplicity at 0)"
                    .into(),
            ));
        }
        start = delta;
    }
    let mut roots: Vec<(Float, Float)> = Vec::new();
    let mut step = fl(0.5);
    while roots.len() < k {
        if start.to_f64() > MAX_ABSCISSA {
            return Err(PdeError::RootFamilyNotCertified(format!(
                "found {} of {k} roots below {MAX_ABSCISSA}",
                roots.len()
            )));
        }
        let end = Float::with_val(prec, &start + &step);
        sc.isolate(start.clone(), end.clone(), 0, &mut roots, k)?;
        start = end;
        // Widen the scanning step slowly so far roots stay cheap.
        if step.to_f64() < 64.0 {
            step *= 2u32;
        }
    }
    let (mut lo, mut hi) = roots[k - 1].clone();
    let s_lo = sc.sign_at(&lo)?;
    // Refine by bisection on certified signs.
    let target = bits.max(8);
    for _ in 0..(4 * target + 200) {
        let width = Float::with_val(prec, &hi - &lo);
        let scale = hi.clone().abs().max(&Float::with_val(prec, 1));
        if width <= scale * Float::with_val(prec, Float::i_exp(1, -(target as i32))) {
            break;
        }
        let mid = Float::with_val(prec, &lo + &hi) / 2u32;
        match sc.sign_at(&mid)? {
            s if s == s_lo => lo = mid,
            Sgn::Unknown => {
                // The midpoint ball straddles the root: shrink around it.
                let q = Float::with_val(prec, &hi - &lo) / 4u32;
                let a = Float::with_val(prec, &mid - &q);
                let b = Float::with_val(prec, &mid + &q);
                if sc.sign_at(&a)? == s_lo
                    && sc.sign_at(&b)? != s_lo
                    && sc.sign_at(&b)? != Sgn::Unknown
                {
                    lo = a;
                    hi = b;
                } else {
                    break;
                }
            }
            _ => hi = mid,
        }
    }
    if branch == 1 {
        Ok((-hi, -lo))
    } else {
        Ok((lo, hi))
    }
}

/// `bessel_j1(u)` → `bessel_j(1, u)`: the same function, but the unary
/// node differentiates to `J₀(u) − J₁(u)/u`, which no ball containing
/// `u = 0` can evaluate, while the binary node's rule `(J₀ − J₂)/2` is
/// regular there.  Used only for the derivatives the scanner certifies.
fn regular_bessel_form(e: ExprId, pool: &ExprPool) -> ExprId {
    fn go(e: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
        if let Some(&r) = memo.get(&e) {
            return r;
        }
        let r = match pool.get(e) {
            ExprData::Func { name, args } if name == "bessel_j1" && args.len() == 1 => {
                let u = go(args[0], pool, memo);
                pool.func("bessel_j", vec![pool.integer(1), u])
            }
            ExprData::Func { name, args } => {
                pool.func(name, args.iter().map(|&a| go(a, pool, memo)).collect())
            }
            ExprData::Add(xs) => pool.add(xs.iter().map(|&a| go(a, pool, memo)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&a| go(a, pool, memo)).collect()),
            ExprData::Pow { base, exp } => pool.pow(go(base, pool, memo), go(exp, pool, memo)),
            _ => e,
        };
        memo.insert(e, r);
        r
    }
    go(e, pool, &mut HashMap::new())
}

fn is_constant_symbol(s: ExprId, pool: &ExprPool) -> bool {
    pool.with(s, |d| match d {
        ExprData::Symbol { name, .. } => crate::eval::symbols::constant_value_f64(name).is_some(),
        _ => false,
    })
}

/// `f64` value of a `RootOfFamily` whose index and parameters are bound by
/// `env` (numeric evaluators' hook); `None` when it cannot be certified.
pub(crate) fn root_family_f64(
    e: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> Option<f64> {
    let (f, lam, n, b) = root_family_parts(e, pool)?;
    let mut m = HashMap::new();
    for s in symbols_of(f, pool).into_iter().chain(symbols_of(n, pool)) {
        if s == lam {
            continue;
        }
        if let Some(&v) = env.get(&s) {
            let val = if v.fract() == 0.0 && v.abs() < 1e15 {
                pool.integer(v as i64)
            } else {
                pool.float(v, 53)
            };
            m.insert(s, val);
        }
    }
    let (f2, n2) = if m.is_empty() {
        (f, n)
    } else {
        (
            crate::kernel::subs::subs(f, &m, pool),
            crate::kernel::subs::subs(n, &m, pool),
        )
    };
    let node = pool.func(ROOT_OF_FAMILY, vec![f2, lam, n2, pool.integer(b)]);
    let (lo, hi) = root_family_enclosure(node, 60, pool).ok()?;
    Some(((lo + hi) / 2u32).to_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::kernel::Domain;

    #[test]
    fn sine_roots_are_multiples_of_pi() {
        let pool = ExprPool::new();
        let lam = pool.symbol("lam", Domain::Real);
        let n = pool.symbol("n", Domain::Integer);
        let fam = root_of_family(pool.func("sin", vec![lam]), lam, n, 0, &pool).unwrap();
        assert_eq!(
            pool.display(fam).to_string(),
            "RootOfFamily(sin(lam), lam, n, 0)"
        );
        for k in 1..=4 {
            let m = member(fam, k, &pool).unwrap();
            let (lo, hi) = root_family_enclosure(m, 80, &pool).unwrap();
            let want = std::f64::consts::PI * k as f64;
            assert!(
                lo.to_f64() <= want + 1e-12 && want - 1e-12 <= hi.to_f64(),
                "k={k}"
            );
            assert!((hi.to_f64() - lo.to_f64()).abs() < 1e-15 * want.max(1.0) * 10.0);
        }
        // negative branch
        let neg =
            root_of_family(pool.func("sin", vec![lam]), lam, pool.integer(2), 1, &pool).unwrap();
        let (lo, _) = root_family_enclosure(neg, 50, &pool).unwrap();
        assert!((lo.to_f64() + 2.0 * std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn robin_family_tan_form() {
        // h sin(λ) + λ cos(λ) = 0 with h = 1: first root ≈ 2.0287578381
        let pool = ExprPool::new();
        let lam = pool.symbol("lam", Domain::Real);
        let f = pool.add(vec![
            pool.func("sin", vec![lam]),
            pool.mul(vec![lam, pool.func("cos", vec![lam])]),
        ]);
        let r = root_of_family(f, lam, pool.integer(1), 0, &pool).unwrap();
        let (lo, hi) = root_family_enclosure(r, 64, &pool).unwrap();
        assert!(
            (lo.to_f64() - 2.028_757_838_1).abs() < 1e-9,
            "{}",
            lo.to_f64()
        );
        assert!(hi >= lo);
        let mut env = HashMap::new();
        env.insert(lam, 0.0);
        assert!((root_family_f64(r, &env, &pool).unwrap() - 2.028_757_838_1).abs() < 1e-9);
    }

    #[test]
    fn refusals() {
        let pool = ExprPool::new();
        let lam = pool.symbol("lam", Domain::Real);
        let h = pool.symbol("h", Domain::Positive);
        let x = pool.integer(3);
        assert_eq!(
            root_of_family(lam, x, lam, 0, &pool).unwrap_err().code(),
            "E-PDE-030"
        );
        assert_eq!(
            root_of_family(lam, lam, x, 7, &pool).unwrap_err().code(),
            "E-PDE-030"
        );
        // free parameter
        let f = pool.add(vec![lam, pool.mul(vec![pool.integer(-1), h])]);
        let r = root_of_family(f, lam, pool.integer(1), 0, &pool).unwrap();
        assert_eq!(
            root_family_enclosure(r, 53, &pool).unwrap_err().code(),
            "E-PDE-030"
        );
        // no positive root at all: λ² + 1
        let g = pool.add(vec![pool.pow(lam, pool.integer(2)), pool.integer(1)]);
        let r = root_of_family(g, lam, pool.integer(1), 0, &pool).unwrap();
        assert_eq!(
            root_family_enclosure(r, 53, &pool).unwrap_err().code(),
            "E-PDE-031"
        );
    }

    #[test]
    fn implicit_derivative_in_a_parameter() {
        // F = λ − h  ⇒ λ₁ = h, dλ/dh = 1.
        let pool = ExprPool::new();
        let lam = pool.symbol("lam", Domain::Real);
        let h = pool.symbol("h", Domain::Positive);
        let f = pool.add(vec![lam, pool.mul(vec![pool.integer(-1), h])]);
        let r = root_of_family(f, lam, pool.integer(1), 0, &pool).unwrap();
        assert_eq!(
            crate::diff::diff(r, h, &pool).unwrap().value,
            pool.integer(1)
        );
        assert_eq!(
            crate::diff::diff(r, lam, &pool).unwrap().value,
            pool.integer(0)
        );
    }
}
