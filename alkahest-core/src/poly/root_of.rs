//! `RootOf(p, k)` — the `k`-th root of a squarefree integer polynomial.
//!
//! An irreducible factor of degree ≥ 3 usually has no useful radical form
//! (degree ≥ 5 in general has none at all; a cubic with three real roots needs
//! complex cube roots — casus irreducibilis). Such a root is represented
//! exactly by its polynomial and an index instead, the way SymPy's `CRootOf`
//! does it.
//!
//! # Encoding
//!
//! A `RootOf` is the ordinary function node
//!
//! ```text
//! Func("RootOf", [k, aₙ, aₙ₋₁, …, a₀])
//! ```
//!
//! with every argument an integer literal: the index `k`, then the
//! coefficients of `p(x) = aₙxⁿ + … + a₀` from the leading one down. `p` is
//! primitive with a positive leading coefficient, squarefree and of degree
//! ≥ 2, and `0 ≤ k < n`. The polynomial is stored by its coefficients rather
//! than as an expression in a variable, so a `RootOf` has **no free symbols**:
//! substituting for the variable that was solved for cannot reach inside it, and
//! it is a constant to `diff`, `subs` and the free-variable walkers. It prints
//! as `RootOf(x^3 - x - 1, 0)`.
//!
//! # Root order
//!
//! The index refers to a fixed, documented order of the `n` distinct roots:
//!
//! 1. the real roots, ascending;
//! 2. then the non-real roots, by ascending real part, ties broken by ascending
//!    imaginary part (so a conjugate pair is listed lower half-plane first).
//!
//! Real roots are certified real (FLINT isolates every root and proves which
//! are real), so the split in step 1 is exact. The comparison in step 2 is made
//! on enclosures computed at [`ORDER_PREC`] bits: two real parts whose
//! enclosures overlap there are treated as equal. That is exact for conjugate
//! pairs and for every polynomial whose distinct real parts differ by more than
//! about `2^-256` relative — which is every polynomial of practical size — and
//! in all cases the order is a deterministic function of the polynomial, so the
//! index a `solve` hands out and the index an evaluator reads always agree.
//!
//! # Numeric value
//!
//! [`root_enclosure`] returns rigorous complex enclosures at any precision. The
//! roots are isolated with FLINT's `arb_fmpz_poly_complex_roots`; a higher
//! precision is matched back to the order fixed at [`ORDER_PREC`] by
//! overlap, never re-sorted. On a FLINT without the Arb layer (no `flint_arb`
//! cfg) there is no root isolation, every enclosure is `None`, and the solver
//! does not emit `RootOf`.

use std::collections::HashMap;

use rug::{Float, Integer};

use crate::kernel::{ExprData, ExprId, ExprPool};

/// Head of a `RootOf` node.
pub const ROOT_OF: &str = "RootOf";

/// Precision, in bits, at which the root order is decided.
pub const ORDER_PREC: u32 = 256;

/// Largest polynomial degree a `RootOf` is evaluated for. Root isolation is
/// polynomial-time, but a pathological input should not stall an evaluator.
pub(crate) const MAX_ROOT_OF_DEGREE: usize = 400;

/// A rigorous enclosure of one root: `re_mid ± re_rad + (im_mid ± im_rad)·i`.
#[derive(Clone, Debug)]
pub(crate) struct RootEnclosure {
    pub(crate) re_mid: Float,
    pub(crate) re_rad: Float,
    pub(crate) im_mid: Float,
    pub(crate) im_rad: Float,
    /// Certified real: the imaginary part is exactly zero.
    pub(crate) is_real: bool,
}

impl RootEnclosure {
    fn re_overlaps(&self, other: &RootEnclosure) -> bool {
        intervals_overlap(&self.re_mid, &self.re_rad, &other.re_mid, &other.re_rad)
    }

    fn overlaps(&self, other: &RootEnclosure) -> bool {
        self.re_overlaps(other)
            && intervals_overlap(&self.im_mid, &self.im_rad, &other.im_mid, &other.im_rad)
    }

    /// Midpoint as an `f64` pair (nearest rounding).
    pub(crate) fn to_c64(&self) -> (f64, f64) {
        (self.re_mid.to_f64(), self.im_mid.to_f64())
    }
}

fn intervals_overlap(m1: &Float, r1: &Float, m2: &Float, r2: &Float) -> bool {
    // |m1 − m2| ≤ r1 + r2, evaluated with enough precision to be exact for the
    // difference and rounded up for the radius sum.
    let p = m1.prec().max(m2.prec()).max(r1.prec()).max(r2.prec()) + 8;
    let d = Float::with_val(p, m1 - m2).abs();
    let r = Float::with_val_round(p, r1 + r2, rug::float::Round::Up).0;
    d <= r
}

/// The canonical order of two roots: real before non-real, reals ascending,
/// non-reals by real part (overlap counts as a tie) then imaginary part.
fn cmp_roots(a: &RootEnclosure, b: &RootEnclosure) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_real, b.is_real) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => a.re_mid.partial_cmp(&b.re_mid).unwrap_or(Ordering::Equal),
        (false, false) => {
            if a.re_overlaps(b) {
                a.im_mid.partial_cmp(&b.im_mid).unwrap_or(Ordering::Equal)
            } else {
                a.re_mid.partial_cmp(&b.re_mid).unwrap_or(Ordering::Equal)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Read a `RootOf` node: its coefficients in **ascending** order (constant
/// term first) and its index. `None` when `expr` is not a well-formed
/// `RootOf` (see the module docs for the shape).
///
/// Squarefreeness is not checked here — it is checked before any root is
/// computed.
pub fn root_of_parts(expr: ExprId, pool: &ExprPool) -> Option<(Vec<Integer>, usize)> {
    let args = pool.with(expr, |data| match data {
        ExprData::Func { name, args } if name == ROOT_OF => Some(args.clone()),
        _ => None,
    })?;
    parts_from_args(&args, pool)
}

/// [`root_of_parts`] on the argument list of a `RootOf` node.
pub(crate) fn parts_from_args(args: &[ExprId], pool: &ExprPool) -> Option<(Vec<Integer>, usize)> {
    if args.len() < 4 {
        return None;
    }
    let mut ints = Vec::with_capacity(args.len());
    for &a in args {
        match pool.get(a) {
            ExprData::Integer(n) => ints.push(n.0),
            _ => return None,
        }
    }
    let k = ints[0].to_usize()?;
    let mut coeffs: Vec<Integer> = ints[1..].to_vec();
    coeffs.reverse();
    let degree = coeffs.len() - 1;
    if *coeffs.last()? <= 0 || k >= degree {
        return None;
    }
    Some((coeffs, k))
}

/// Is `expr` a well-formed `RootOf` node?
pub fn is_root_of(expr: ExprId, pool: &ExprPool) -> bool {
    root_of_parts(expr, pool).is_some()
}

/// Degree of the largest `RootOf` occurring anywhere in `expr`, if any.
pub(crate) fn max_root_of_degree(expr: ExprId, pool: &ExprPool) -> Option<usize> {
    let mut stack = vec![expr];
    let mut seen = std::collections::HashSet::new();
    let mut best: Option<usize> = None;
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        if let Some((c, _)) = root_of_parts(e, pool) {
            best = Some(best.map_or(c.len() - 1, |b| b.max(c.len() - 1)));
            continue;
        }
        pool.with(e, |d| match d {
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
                stack.extend(a.iter().copied())
            }
            ExprData::Pow { base, exp } => stack.extend([*base, *exp]),
            _ => {}
        });
    }
    best
}

/// Build the `RootOf` node for `coeffs` (ascending, already primitive with a
/// positive leading coefficient, squarefree, degree ≥ 2) and index `k`.
pub(crate) fn make_root_of(coeffs: &[Integer], k: usize, pool: &ExprPool) -> ExprId {
    debug_assert!(coeffs.len() >= 3 && k + 1 < coeffs.len());
    let mut args = Vec::with_capacity(coeffs.len() + 1);
    args.push(pool.integer(Integer::from(k)));
    for c in coeffs.iter().rev() {
        args.push(pool.integer(c.clone()));
    }
    pool.func(ROOT_OF, args)
}

/// `p` as text in a placeholder variable `x`, leading term first, with the
/// power rendered by `pow` (`x^3`, `x³`, `x^{3}`).
pub(crate) fn poly_text(coeffs: &[Integer], pow: impl Fn(usize) -> String) -> String {
    let mut out = String::new();
    for (k, c) in coeffs.iter().enumerate().rev() {
        if *c == 0 {
            continue;
        }
        let neg = *c < 0;
        let mag = Integer::from(c.abs_ref());
        if out.is_empty() {
            if neg {
                out.push('-');
            }
        } else {
            out.push_str(if neg { " - " } else { " + " });
        }
        let var = match k {
            0 => String::new(),
            1 => "x".to_string(),
            _ => pow(k),
        };
        if k == 0 {
            out.push_str(&mag.to_string());
        } else if mag == 1 {
            out.push_str(&var);
        } else {
            out.push_str(&format!("{mag}*{var}"));
        }
    }
    out
}

/// `RootOf(x^3 - x - 1, 0)` for the arguments of a well-formed node, `None`
/// otherwise; `pow` renders `x` to a power.
pub(crate) fn display_root_of_args(
    args: &[ExprId],
    pool: &ExprPool,
    pow: impl Fn(usize) -> String,
) -> Option<String> {
    let (coeffs, k) = parts_from_args(args, pool)?;
    Some(format!("{ROOT_OF}({}, {k})", poly_text(&coeffs, pow)))
}

// ---------------------------------------------------------------------------
// Construction from a polynomial
// ---------------------------------------------------------------------------

/// Why [`root_of`] could not build a root.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootOfError {
    /// `poly` is not a polynomial in `var` with rational coefficients.
    NotPolynomial(String),
    /// `index` is not below the number of distinct roots.
    IndexOutOfRange { index: usize, count: usize },
    /// The polynomial is too large, or this build of FLINT has no root
    /// isolation (the Arb layer).
    Unavailable(String),
}

impl std::fmt::Display for RootOfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootOfError::NotPolynomial(s) => {
                write!(
                    f,
                    "RootOf needs a polynomial with rational coefficients: {s}"
                )
            }
            RootOfError::IndexOutOfRange { index, count } => write!(
                f,
                "RootOf index {index} is out of range: the polynomial has {count} distinct root(s)"
            ),
            RootOfError::Unavailable(s) => write!(f, "RootOf is unavailable: {s}"),
        }
    }
}

impl std::error::Error for RootOfError {}

impl crate::errors::AlkahestError for RootOfError {
    fn code(&self) -> &'static str {
        "E-SOLVE-008"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "pass a non-constant polynomial with rational coefficients in the given variable \
             and an index 0 <= k < number of distinct roots (real roots ascending, then \
             complex roots by real part, then imaginary part)",
        )
    }
}

/// The `index`-th distinct root of `poly` (a polynomial in `var` over ℚ), in
/// the order of the module docs.
///
/// The result is exact: a rational number when the root is rational, otherwise
/// the `RootOf` of the irreducible factor of `poly` it is a root of, with that
/// factor's own index. Repeated roots count once.
///
/// ```text
/// root_of(x³ − x − 1, x, 0)  →  RootOf(x^3 - x - 1, 0)    (≈ 1.3247)
/// root_of((x − 2)(x² − 2), x, 2)  →  2
/// ```
pub fn root_of(
    poly: ExprId,
    var: ExprId,
    index: usize,
    pool: &ExprPool,
) -> Result<ExprId, RootOfError> {
    let up = crate::poly::UniPoly::from_symbolic_clear_denoms(poly, var, pool)
        .map_err(|e| RootOfError::NotPolynomial(e.to_string()))?;
    if up.is_zero() {
        return Err(RootOfError::NotPolynomial(
            "the zero polynomial has every number as a root".to_string(),
        ));
    }
    let fac = up
        .factor_z()
        .map_err(|e| RootOfError::NotPolynomial(e.to_string()))?;
    // Every root of every irreducible factor, tagged with (factor, index).
    let mut all: Vec<(RootEnclosure, usize, usize)> = Vec::new();
    let mut factors: Vec<Vec<Integer>> = Vec::new();
    for (f, _) in &fac.factors {
        let c = f.coefficients();
        if c.len() < 2 {
            continue;
        }
        let roots = if c.len() == 2 {
            vec![rational_enclosure(&c)]
        } else {
            ordered_roots(&c, ORDER_PREC).ok_or_else(|| {
                RootOfError::Unavailable(
                    "the roots could not be isolated (polynomial too large, or FLINT \
                     without the Arb layer)"
                        .to_string(),
                )
            })?
        };
        let fi = factors.len();
        factors.push(c);
        for (j, r) in roots.into_iter().enumerate() {
            all.push((r, fi, j));
        }
    }
    let count = all.len();
    if index >= count {
        return Err(RootOfError::IndexOutOfRange { index, count });
    }
    all.sort_by(|a, b| cmp_roots(&a.0, &b.0));
    let (_, fi, j) = &all[index];
    let c = &factors[*fi];
    if c.len() == 2 {
        return Ok(rational_expr(
            &rug::Rational::from((-c[0].clone(), c[1].clone())),
            pool,
        ));
    }
    Ok(make_root_of(c, *j, pool))
}

/// `r` as an integer node when it is one, a rational node otherwise.
pub(crate) fn rational_expr(r: &rug::Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

/// The exact enclosure of the root `−c₀/c₁` of a linear factor.
fn rational_enclosure(c: &[Integer]) -> RootEnclosure {
    let r = rug::Rational::from((-c[0].clone(), c[1].clone()));
    let b = crate::ball::ArbBall::from_rational(&r, ORDER_PREC);
    RootEnclosure {
        re_mid: b.mid,
        re_rad: b.rad,
        im_mid: Float::new(ORDER_PREC),
        im_rad: Float::new(ORDER_PREC),
        is_real: true,
    }
}

// ---------------------------------------------------------------------------
// Root isolation
// ---------------------------------------------------------------------------

thread_local! {
    /// Root sets already isolated on this thread, keyed by polynomial and
    /// precision. Evaluators reach the same `RootOf` many times (a solution
    /// tuple is evaluated by the solver's own check, then by `numeric=True`,
    /// then once per sample by a gate), and isolation is the costly step.
    static CACHE: std::cell::RefCell<HashMap<(Vec<Integer>, u32), Vec<RootEnclosure>>> =
        std::cell::RefCell::new(HashMap::new());
}

const CACHE_CAP: usize = 128;

/// Every root of the squarefree polynomial `coeffs` (ascending), in canonical
/// order, enclosed to about `prec` bits. `None` when the polynomial is not
/// squarefree, has degree < 1 or above [`MAX_ROOT_OF_DEGREE`], or this build
/// has no root isolation.
pub(crate) fn ordered_roots(coeffs: &[Integer], prec: u32) -> Option<Vec<RootEnclosure>> {
    let prec = prec.max(ORDER_PREC);
    let key = (coeffs.to_vec(), prec);
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return Some(hit);
    }
    let out = ordered_roots_uncached(coeffs, prec)?;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= CACHE_CAP {
            c.clear();
        }
        c.insert(key, out.clone());
    });
    Some(out)
}

fn ordered_roots_uncached(coeffs: &[Integer], prec: u32) -> Option<Vec<RootEnclosure>> {
    let degree = coeffs.len().checked_sub(1)?;
    if degree == 0 || degree > MAX_ROOT_OF_DEGREE || coeffs[degree] == 0 {
        return None;
    }
    if !is_squarefree(coeffs) {
        return None;
    }
    let mut ordered = isolate(coeffs, ORDER_PREC)?;
    ordered.sort_by(cmp_roots);
    if prec <= ORDER_PREC {
        return Some(ordered);
    }
    // Refine without re-sorting: each high-precision root is assigned to the
    // single low-precision enclosure it overlaps. Both sets are isolating, so
    // exactly one exists; anything else is refused rather than guessed.
    let fine = isolate(coeffs, prec)?;
    let mut out: Vec<Option<RootEnclosure>> = vec![None; ordered.len()];
    for r in fine {
        let mut hits = ordered
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_real == r.is_real && o.overlaps(&r));
        let (i, _) = hits.next()?;
        if hits.next().is_some() || out[i].is_some() {
            return None;
        }
        out[i] = Some(r);
    }
    out.into_iter().collect()
}

fn is_squarefree(coeffs: &[Integer]) -> bool {
    let p = crate::flint::FlintPoly::from_rug_coefficients(coeffs);
    let d = p.derivative();
    if d.is_zero() {
        return p.degree() == 0;
    }
    p.gcd(&d).degree() == 0
}

#[cfg(flint_arb)]
fn isolate(coeffs: &[Integer], prec: u32) -> Option<Vec<RootEnclosure>> {
    use crate::flint::acb::AcbVec;
    let degree = coeffs.len() - 1;
    let p = crate::flint::FlintPoly::from_rug_coefficients(coeffs);
    let mut v = AcbVec::new(degree);
    // SAFETY: `v` holds `degree` initialised `acb_struct`s, which is exactly
    // the number of roots FLINT writes for a degree-`degree` polynomial, and
    // `p` is an initialised `fmpz_poly_t`. The caller has checked that `p` is
    // squarefree, without which FLINT does not return.
    unsafe {
        crate::flint::arb::arb_fmpz_poly_complex_roots(
            v.as_mut_ptr(),
            p.inner_ptr(),
            0,
            i64::from(prec),
        );
    }
    let mut out = Vec::with_capacity(degree);
    for i in 0..degree {
        let z = v.get(i);
        let (re, im) = (z.real(), z.imag());
        let is_real = im.is_exact() && im.midpoint_rug()? == 0;
        out.push(RootEnclosure {
            re_mid: re.midpoint_rug()?,
            re_rad: re.radius_rug()?,
            im_mid: im.midpoint_rug()?,
            im_rad: im.radius_rug()?,
            is_real,
        });
    }
    Some(out)
}

#[cfg(not(flint_arb))]
fn isolate(_coeffs: &[Integer], _prec: u32) -> Option<Vec<RootEnclosure>> {
    None
}

/// Whether this build can isolate roots, i.e. whether `RootOf` values are
/// numerically available. The solver only emits `RootOf` when they are.
pub fn root_isolation_available() -> bool {
    cfg!(flint_arb)
}

// ---------------------------------------------------------------------------
// Evaluation entry points
// ---------------------------------------------------------------------------

/// A rigorous enclosure of the `RootOf` node `expr` at about `prec` bits.
pub(crate) fn root_enclosure(expr: ExprId, pool: &ExprPool, prec: u32) -> Option<RootEnclosure> {
    let (coeffs, k) = root_of_parts(expr, pool)?;
    ordered_roots(&coeffs, prec)?.into_iter().nth(k)
}

/// The value of a `RootOf` node as a complex `f64` pair.
pub(crate) fn root_of_c64(expr: ExprId, pool: &ExprPool) -> Option<(f64, f64)> {
    let r = root_enclosure(expr, pool, ORDER_PREC)?;
    let (re, im) = r.to_c64();
    Some((re, if r.is_real { 0.0 } else { im }))
}

/// The value of a real `RootOf` node as an `f64`; `None` for a non-real root.
pub(crate) fn root_of_real_f64(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    let r = root_enclosure(expr, pool, ORDER_PREC)?;
    r.is_real.then(|| r.re_mid.to_f64())
}

/// A rigorous real ball for a real `RootOf` node at `prec` bits; `None` for a
/// non-real root (it is not a real number).
pub(crate) fn root_of_real_ball(
    expr: ExprId,
    pool: &ExprPool,
    prec: u32,
) -> Option<crate::ball::ArbBall> {
    let r = root_enclosure(expr, pool, prec.saturating_add(16))?;
    if !r.is_real {
        return None;
    }
    Some(widen(&r.re_mid, &r.re_rad, prec))
}

/// Rigorous real and imaginary balls at `prec` bits.
pub(crate) fn root_of_complex_balls(
    expr: ExprId,
    pool: &ExprPool,
    prec: u32,
) -> Option<(crate::ball::ArbBall, crate::ball::ArbBall)> {
    let r = root_enclosure(expr, pool, prec.saturating_add(16))?;
    let re = widen(&r.re_mid, &r.re_rad, prec);
    let im = if r.is_real {
        crate::ball::ArbBall::from_f64(0.0, prec)
    } else {
        widen(&r.im_mid, &r.im_rad, prec)
    };
    Some((re, im))
}

/// The ball `mid ± rad`, re-rounded to `prec` bits without losing enclosure.
fn widen(mid: &Float, rad: &Float, prec: u32) -> crate::ball::ArbBall {
    let mut b = crate::ball::ArbBall::from_float(mid, prec);
    let r = Float::with_val_round(prec, rad, rug::float::Round::Up).0;
    b.rad = Float::with_val_round(prec, &b.rad + &r, rug::float::Round::Up).0;
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn encoding_round_trips() {
        let pool = ExprPool::new();
        let c = ints(&[-1, -1, 0, 1]); // x³ − x − 1
        let r = make_root_of(&c, 0, &pool);
        let (back, k) = root_of_parts(r, &pool).unwrap();
        assert_eq!(back, c);
        assert_eq!(k, 0);
        assert_eq!(pool.display(r).to_string(), "RootOf(x^3 - x - 1, 0)");
    }

    #[test]
    fn malformed_nodes_are_not_root_of() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // Symbolic coefficient, out-of-range index, too short.
        let a = pool.func(
            ROOT_OF,
            vec![pool.integer(0), pool.integer(1), x, pool.integer(1)],
        );
        let b = pool.func(
            ROOT_OF,
            vec![
                pool.integer(2),
                pool.integer(1),
                pool.integer(0),
                pool.integer(-2),
            ],
        );
        let c = pool.func(ROOT_OF, vec![pool.integer(0), pool.integer(1)]);
        for e in [a, b, c] {
            assert!(root_of_parts(e, &pool).is_none());
        }
    }

    #[cfg(flint_arb)]
    #[test]
    fn real_roots_come_first_and_ascend() {
        // (x² − 2)(x² + 1)... use x⁴ − x² − 2 = (x² − 2)(x² + 1): reducible but
        // squarefree, so the order is still defined.
        let roots = ordered_roots(&ints(&[-2, 0, -1, 0, 1]), ORDER_PREC).unwrap();
        assert_eq!(roots.len(), 4);
        let s2 = std::f64::consts::SQRT_2;
        let v: Vec<(f64, f64)> = roots.iter().map(|r| r.to_c64()).collect();
        assert!(roots[0].is_real && roots[1].is_real);
        assert!((v[0].0 + s2).abs() < 1e-15 && (v[1].0 - s2).abs() < 1e-15);
        assert!(!roots[2].is_real && !roots[3].is_real);
        assert!((v[2].1 + 1.0).abs() < 1e-15 && (v[3].1 - 1.0).abs() < 1e-15);
    }

    #[cfg(flint_arb)]
    #[test]
    fn equal_real_parts_order_by_imaginary_part() {
        // x⁴ + 3x² + 1: roots ±iφ, ±i/φ, all on the imaginary axis.
        let roots = ordered_roots(&ints(&[1, 0, 3, 0, 1]), ORDER_PREC).unwrap();
        let ims: Vec<f64> = roots.iter().map(|r| r.to_c64().1).collect();
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let want = [-phi, -1.0 / phi, 1.0 / phi, phi];
        for (g, w) in ims.iter().zip(want) {
            assert!((g - w).abs() < 1e-14, "{ims:?}");
        }
    }

    #[cfg(flint_arb)]
    #[test]
    fn refinement_keeps_the_order_and_tightens() {
        let c = ints(&[-1, -1, 0, 1]);
        let lo = ordered_roots(&c, ORDER_PREC).unwrap();
        let hi = ordered_roots(&c, 1024).unwrap();
        for (a, b) in lo.iter().zip(&hi) {
            assert!(a.overlaps(b));
            assert!(b.re_rad <= a.re_rad || b.re_rad.is_zero());
        }
        // The real root of x³ − x − 1 (the plastic number) to 300 digits.
        let r = &hi[0];
        assert!(r.is_real);
        let rad_bits = r.re_rad.get_exp().unwrap_or(i32::MIN);
        assert!(rad_bits < -1000, "radius 2^{rad_bits}");
    }

    #[test]
    fn a_repeated_root_is_refused_not_looped_on() {
        // (x − 1)²(x + 2): FLINT would never return on this.
        assert!(ordered_roots(&ints(&[2, -3, 0, 1]), ORDER_PREC).is_none());
    }

    #[cfg(flint_arb)]
    #[test]
    fn root_of_reduces_to_the_irreducible_factor() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // (x − 2)(x³ − x − 1) = x⁴ − 2x³ − x² + x + 2
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4)),
            pool.mul(vec![pool.integer(-2), pool.pow(x, pool.integer(3))]),
            pool.mul(vec![pool.integer(-1), pool.pow(x, pool.integer(2))]),
            x,
            pool.integer(2),
        ]);
        // Real roots: 1.3247…, 2.
        let r0 = root_of(p, x, 0, &pool).unwrap();
        assert_eq!(pool.display(r0).to_string(), "RootOf(x^3 - x - 1, 0)");
        let r1 = root_of(p, x, 1, &pool).unwrap();
        assert_eq!(r1, pool.integer(2));
        // Then the complex pair of the cubic, lower half-plane first.
        let r2 = root_of(p, x, 2, &pool).unwrap();
        assert_eq!(pool.display(r2).to_string(), "RootOf(x^3 - x - 1, 1)");
        assert!(matches!(
            root_of(p, x, 4, &pool),
            Err(RootOfError::IndexOutOfRange { index: 4, count: 4 })
        ));
    }
}
