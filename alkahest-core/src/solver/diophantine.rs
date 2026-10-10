//! Diophantine equations — linear parametric families and binary quadratics.
//!
//! ## Sum of two squares
//!
//! For `x² + y² = n` with `n ≥ 0`, factor `n` and use **Cornacchia** on primes `p ≡ 1 (mod 4)`,
//! then **compose** representations via the Brahmagupta–Fibonacci identity.
//! When factorization is impractical (very large `n`), falls back to scanning `x ≤ √n`.
//!
//! ## Generalized Pell
//!
//! `x² - D·y² = N` with `D > 0` non-square: one period of the **continued fraction** of `√D`
//! gives the fundamental unit (and decides `N = −1`: solvable iff the period is odd).  Other `N`
//! are decided by sweeping `y` up to **Nagell's bound** (complete), or — when that bound is too
//! large — by the convergents over two periods (complete for `|N| < √D`); otherwise the solver
//! refuses rather than report a false "no solution".  Solutions multiply by the unit
//! `u² - D·v² = 1`.  `D = s²` factors as `(x − s·y)(x + s·y) = N` (finite).
//! `N = 0`: trivial `(0,0)` unless `D` is a rational square, else a parametric line.
//!
//! All loops honour the active [`crate::budget`]; see [`last_budget_trip`].

use crate::budget::BudgetError;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::groebner::ideal::GbPoly;
use rug::ops::{DivRounding, Pow};
use rug::Integer;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;

use super::{expr_to_gbpoly, SolverError};

/// Errors from [`diophantine`].
#[derive(Debug, Clone)]
pub enum DiophantineError {
    /// Equation is not a polynomial in the listed variables.
    NotPolynomial(String),
    /// Coefficients are not rational integers (even after clearing denominators).
    NonIntegerCoefficients,
    /// Equation degree or term pattern is not handled.
    Unsupported(String),
    /// No integer solutions exist for this instance.
    NoSolution,
}

impl fmt::Display for DiophantineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiophantineError::NotPolynomial(s) => write!(f, "diophantine: {s}"),
            DiophantineError::NonIntegerCoefficients => {
                write!(f, "diophantine: coefficients must be rational integers")
            }
            DiophantineError::Unsupported(s) => write!(f, "diophantine: unsupported: {s}"),
            DiophantineError::NoSolution => write!(f, "diophantine: no integer solution"),
        }
    }
}

impl std::error::Error for DiophantineError {}

impl AlkahestError for DiophantineError {
    fn code(&self) -> &'static str {
        match self {
            DiophantineError::NotPolynomial(_) => "E-DIOPH-001",
            DiophantineError::NonIntegerCoefficients => "E-DIOPH-002",
            DiophantineError::Unsupported(_) => "E-DIOPH-003",
            DiophantineError::NoSolution => "E-DIOPH-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DiophantineError::NotPolynomial(_) => Some(
                "pass a single polynomial equation in the listed symbols with integer/rational coefficients",
            ),
            DiophantineError::NonIntegerCoefficients => Some(
                "rewrite so all coefficients are integers (no fractional parameters)",
            ),
            DiophantineError::Unsupported(_) => Some(
                "supported: linear two-variable, x²+y²=n, x²−D·y²=N (no xy term); huge integers may need a smaller instance",
            ),
            DiophantineError::NoSolution => Some(
                "check divisibility for linear equations; for quadratics verify solvability over ℤ",
            ),
        }
    }
}

impl From<SolverError> for DiophantineError {
    fn from(e: SolverError) -> Self {
        DiophantineError::NotPolynomial(e.to_string())
    }
}

/// Result of [`diophantine`].
#[derive(Debug, Clone)]
pub enum DiophantineSolution {
    /// `a·x + b·y + … = 0`: values are `x(t)`, `y(t)`, … in the same order as `vars`,
    /// with integer parameter `t`.
    ParametricLinear {
        parameter: ExprId,
        values: Vec<ExprId>,
    },
    /// Explicit list of integer tuples (each parallel to `vars`).
    Finite(Vec<Vec<ExprId>>),
    /// `x² - D·y² = 1`: fundamental unit `(x0, y0)`; all solutions via
    /// `(x0 + y0√D)^k`, `k ∈ ℤ`.
    PellFundamental { d: ExprId, x0: ExprId, y0: ExprId },
    /// `x² - D·y² = N` with `N ≠ 1`: minimal found pair `(x0, y0)` and unit `(ux, uy)` with
    /// `ux² - D·uy² = 1`.  All solutions satisfy
    /// `x + y√D = (x0 + y0√D)·(ux + uy√D)^k`, `k ∈ ℤ`.
    PellGeneralized {
        d: ExprId,
        n: ExprId,
        x0: ExprId,
        y0: ExprId,
        unit_x: ExprId,
        unit_y: ExprId,
    },
    /// No integer solutions.
    NoSolution,
}

fn lcm_rational_denominators(poly: &GbPoly) -> Integer {
    let mut l = Integer::from(1);
    for c in poly.terms.values() {
        let den: Integer = c.denom().into();
        l = l.lcm(&den);
    }
    l
}

fn gbpoly_integer_coeffs(poly: &GbPoly) -> Result<BTreeMap<Vec<u32>, Integer>, DiophantineError> {
    let scale = lcm_rational_denominators(poly);
    let mut out = BTreeMap::new();
    for (e, c) in &poly.terms {
        let num: Integer = c.numer().into();
        let den: Integer = c.denom().into();
        let prod = num * &scale;
        let scaled = div_exact(&prod, &den).ok_or(DiophantineError::NonIntegerCoefficients)?;
        if scaled != 0 {
            out.insert(e.clone(), scaled);
        }
    }
    Ok(out)
}

fn term_gcd(iv: &[Integer]) -> Integer {
    let mut g = iv.first().cloned().unwrap_or_else(|| Integer::from(0));
    for x in iv.iter().skip(1) {
        g = g.gcd(x);
    }
    g
}

fn div_exact(a: &Integer, g: &Integer) -> Option<Integer> {
    let (q, r) = a.clone().div_rem_euc_ref(g).into();
    if r == 0 {
        Some(q)
    } else {
        None
    }
}

/// Extended gcd: `(g, u, v)` with `u·a + v·b = g = gcd(a,b)`.
fn extended_gcd(a: &Integer, b: &Integer) -> (Integer, Integer, Integer) {
    let mut old_r = a.clone();
    let mut r = b.clone();
    let mut old_s = Integer::from(1);
    let mut s = Integer::from(0);
    let mut old_t = Integer::from(0);
    let mut t = Integer::from(1);
    while r != 0 {
        let q = old_r.clone() / &r;
        let mut tmp = old_r - &q * &r;
        old_r = r;
        r = tmp;
        tmp = old_s - &q * &s;
        old_s = s;
        s = tmp;
        tmp = old_t - &q * &t;
        old_t = t;
        t = tmp;
    }
    (old_r, old_s, old_t)
}

/// `(a²+b²)(c²+d²) = (ac−bd)² + (ad+bc)²`
fn compose_sum_sq(x: &Integer, y: &Integer, c: &Integer, d: &Integer) -> (Integer, Integer) {
    let nx: Integer = x.clone() * c - y.clone() * d;
    let ny: Integer = x.clone() * d + y.clone() * c;
    (nx, ny)
}

fn is_perfect_square(n: &Integer) -> bool {
    if n.cmp0().is_lt() {
        return false;
    }
    let (_, r) = n.clone().sqrt_rem(Integer::new());
    r == 0
}

/// Legendre symbol (a / p) for odd prime p, a not divisible by p → ±1.
fn legendre(a: &Integer, p: &Integer) -> i32 {
    let exp = (p.clone() - 1) / 2;
    let ls = a
        .clone()
        .pow_mod(&exp, p)
        .unwrap_or_else(|_| Integer::from(0));
    if ls == 1 {
        1
    } else if ls == p.clone() - 1 {
        -1
    } else {
        0
    }
}

/// Tonelli–Shanks: square root of `n` mod odd prime `p` (when it exists).
fn tonelli_shanks(n: &Integer, p: &Integer) -> Option<Integer> {
    let (_, rrem) = n.clone().div_rem_euc_ref(p).into();
    if rrem == 0 {
        return Some(Integer::from(0));
    }
    if legendre(n, p) != 1 {
        return None;
    }
    if p.clone() % 4u32 == 3 {
        let exp = (p.clone() + 1) / 4;
        return n.clone().pow_mod(&exp, p).ok();
    }

    let mut q: Integer = p.clone() - Integer::from(1);
    let mut s = 0u32;
    while q.clone() % 2u32 == 0 {
        q /= 2u32;
        s += 1;
    }

    let mut z = Integer::from(2);
    while legendre(&z, p) != -1 {
        z += 1;
        if z >= *p {
            return None;
        }
    }

    let mut m = s;
    let mut c = z.clone().pow_mod(&q, p).ok()?;
    let mut t = n.clone().pow_mod(&q, p).ok()?;
    let mut r = n.clone().pow_mod(&((q.clone() + 1) / 2), p).ok()?;

    while t != 1 {
        let mut i = 0u32;
        let mut tt = t.clone();
        while tt != 1 {
            tt = (tt.clone() * &tt) % p;
            i += 1;
            if i > m {
                return None;
            }
        }
        let exp = m - i - 1;
        let two_exp = Integer::from(1) << exp;
        let b = c.clone().pow_mod(&two_exp, p).ok()?;
        r = (r.clone() * &b) % p;
        t = (t * &b * &b) % p;
        c = (b.clone() * &b) % p;
        m = i;
    }
    Some(r)
}

/// Cornacchia: `x² + d·y² = p` for odd prime `p`, `gcd(d,p)=1`, `(−d/p)=1`.
/// Returns `(x, y)` with `x, y ≥ 0`.
fn cornacchia_prime(d: &Integer, p: &Integer) -> Option<(Integer, Integer)> {
    if *p == 2 {
        if *d == 1 {
            return Some((Integer::from(1), Integer::from(1)));
        }
        return None;
    }
    if p.clone() % 2 == 0 {
        return None;
    }

    // (−d / p) = 1
    let negd = (p.clone() - (d.clone() % p)) % p;
    if legendre(&negd, p) != 1 {
        return None;
    }

    let mut r0 = tonelli_shanks(&negd, p)?;
    if r0.clone() > p.clone() / 2 {
        r0 = p.clone() - &r0;
    }

    let mut r = p.clone();
    let mut s = r0;
    while s.clone() * &s > *p {
        let rem = r.clone() % &s;
        r = s;
        s = rem;
    }

    let diff = p.clone() - &s * &s;
    if diff.cmp0().is_lt() {
        return None;
    }
    let q = div_exact(&diff, d)?;
    let (_, rr) = q.clone().sqrt_rem(Integer::new());
    if rr != 0 {
        return None;
    }
    let y = q.sqrt();
    Some((s, y))
}

/// `x² + y² = p` for prime `p`.
fn prime_as_sum_two_squares(p: &Integer) -> Option<(Integer, Integer)> {
    cornacchia_prime(&Integer::from(1), p)
}

fn pollard_step(g: &Integer, c: &Integer, x: &Integer) -> Integer {
    (x.clone() * x + c) % g
}

/// One nontrivial factor of composite `n` (not necessarily prime).
fn pollard_rho_factor(n: &Integer) -> Option<Integer> {
    if n <= &Integer::from(3) || is_probable_prime(n) {
        return None;
    }
    let mut x = Integer::from(2);
    let mut y = Integer::from(2);
    let mut d = Integer::from(1);
    let c = Integer::from(1);
    while d == 1 {
        x = pollard_step(n, &c, &x);
        y = pollard_step(n, &c, &pollard_step(n, &c, &y));
        let diff = if x.clone() >= y {
            x.clone() - &y
        } else {
            y.clone() - &x
        };
        d = diff.gcd(n);
        if d == *n {
            return None;
        }
    }
    if d > 1 && d < *n {
        Some(d)
    } else {
        None
    }
}

/// Deterministic probable-prime (Miller–Rabin with small bases) for odd `n > 2`.
fn is_probable_prime(n: &Integer) -> bool {
    if n <= &Integer::from(1) {
        return false;
    }
    if n <= &Integer::from(3) {
        return true;
    }
    if n.clone() % 2u32 == 0 {
        return false;
    }
    n.is_probably_prime(40) != rug::integer::IsPrime::No
}

/// Distinct prime factors with multiplicity, `n ≥ 2`.
fn factor_positive(mut n: Integer) -> Vec<(Integer, u32)> {
    let mut fac: Vec<(Integer, u32)> = Vec::new();

    let push_pow = |fac: &mut Vec<(Integer, u32)>, p: Integer, e: u32| {
        if e > 0 {
            fac.push((p, e));
        }
    };

    let small: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &pr in &small {
        let p = Integer::from(pr);
        if n <= 1 {
            break;
        }
        let mut e = 0u32;
        while n.clone() % &p == 0 {
            n /= &p;
            e += 1;
        }
        push_pow(&mut fac, p, e);
    }

    let mut stack: Vec<Integer> = Vec::new();
    if n > 1 {
        stack.push(n);
    }
    let mut prime_parts: Vec<Integer> = Vec::new();
    while let Some(m) = stack.pop() {
        if m <= 1 {
            continue;
        }
        if is_probable_prime(&m) {
            prime_parts.push(m);
            continue;
        }
        let mut split = None;
        for _ in 0..16 {
            if let Some(d) = pollard_rho_factor(&m) {
                let other = m.clone() / &d;
                split = Some((d, other));
                break;
            }
        }
        if let Some((d, other)) = split {
            stack.push(d);
            stack.push(other);
        } else {
            prime_parts.push(m);
        }
    }

    prime_parts.sort();
    let mut i = 0usize;
    while i < prime_parts.len() {
        let p = prime_parts[i].clone();
        let mut e = 0u32;
        while i < prime_parts.len() && prime_parts[i] == p {
            e += 1;
            i += 1;
        }
        push_pow(&mut fac, p, e);
    }

    fac
}

fn scan_sum_two_squares_pairs(n: &Integer) -> Vec<(Integer, Integer)> {
    let mut pts: Vec<(Integer, Integer)> = Vec::new();
    let mut x = Integer::from(0);
    let max_x = n.clone().sqrt();
    while x <= max_x {
        let r = n.clone() - &x * &x;
        if is_perfect_square(&r) {
            let y = r.sqrt();
            if x <= y {
                pts.push((x.clone(), y.clone()));
                if x < y {
                    pts.push((y.clone(), x.clone()));
                }
            }
        }
        x += 1;
    }
    pts
}

fn merge_distinct_pairs(acc: &mut Vec<(Integer, Integer)>, more: Vec<(Integer, Integer)>) {
    use std::collections::BTreeSet;
    let mut seen: BTreeSet<String> = acc.iter().map(|(a, b)| format!("{a},{b}")).collect();
    for (x, y) in more {
        let k = format!("{x},{y}");
        if seen.insert(k) {
            acc.push((x, y));
        }
    }
}

/// Ordered pairs `(x,y)` with `x,y ≥ 0` and `x² + y² = n`: one orbit from Cornacchia composition,
/// plus any further orbits found by a bounded scan when `n` is moderate (bit size ≤ 256).
fn sum_two_squares_representatives(n: &Integer) -> Vec<(Integer, Integer)> {
    if n.cmp0().is_lt() {
        return vec![];
    }
    if *n == 0 {
        return vec![(Integer::from(0), Integer::from(0))];
    }

    if n.significant_bits() > 4000 {
        return vec![];
    }

    let mut rest = n.clone();
    let mut e2 = 0u32;
    while rest.clone() % 2u32 == 0 {
        rest /= 2u32;
        e2 += 1;
    }

    if rest == 1 {
        // n = 2^e2
        let mut x = Integer::from(1);
        let mut y = Integer::from(0);
        for _ in 0..e2 {
            let c = compose_sum_sq(&x, &y, &Integer::from(1), &Integer::from(1));
            x = c.0;
            y = c.1;
        }
        return canonical_pairs(x, y);
    }

    let facs = factor_positive(rest);
    for (p, e) in &facs {
        let m4 = p.clone() % 4;
        if m4 == 3 && e % 2 == 1 {
            return vec![];
        }
    }

    let mut xr = Integer::from(1);
    let mut yr = Integer::from(0);
    for (p, e) in facs {
        let m4 = p.clone() % 4;
        if m4 == 3 {
            debug_assert!(e % 2 == 0);
            let half = e / 2;
            let pk = p.clone().pow(half);
            xr *= &pk;
            yr *= &pk;
            continue;
        }
        if p == 2 {
            for _ in 0..e {
                let c = compose_sum_sq(&xr, &yr, &Integer::from(1), &Integer::from(1));
                xr = c.0;
                yr = c.1;
            }
            continue;
        }
        // p ≡ 1 (mod 4)
        let (up, vp) = match prime_as_sum_two_squares(&p) {
            Some(t) => t,
            None => return vec![],
        };
        let mut xq = Integer::from(1);
        let mut yq = Integer::from(0);
        for _ in 0..e {
            let c = compose_sum_sq(&xq, &yq, &up, &vp);
            xq = c.0;
            yq = c.1;
        }
        let c = compose_sum_sq(&xr, &yr, &xq, &yq);
        xr = c.0;
        yr = c.1;
    }

    for _ in 0..e2 {
        let c = compose_sum_sq(&xr, &yr, &Integer::from(1), &Integer::from(1));
        xr = c.0;
        yr = c.1;
    }

    let mut out = canonical_pairs(xr, yr);
    if n.significant_bits() <= 256 {
        merge_distinct_pairs(&mut out, scan_sum_two_squares_pairs(n));
    }
    out
}

fn canonical_pairs(x: Integer, y: Integer) -> Vec<(Integer, Integer)> {
    let x = x.abs();
    let y = y.abs();
    let mut pts = Vec::new();
    if x <= y {
        pts.push((x.clone(), y.clone()));
        if x < y {
            pts.push((y, x));
        }
    } else {
        pts.push((y.clone(), x.clone()));
        if y < x {
            pts.push((x, y));
        }
    }
    pts
}

fn solve_sum_two_squares_scan(pool: &ExprPool, n: &Integer) -> DiophantineSolution {
    let n = n.clone();
    if n < 0 {
        return DiophantineSolution::NoSolution;
    }
    if n == 0 {
        let z = pool.integer(0);
        return DiophantineSolution::Finite(vec![vec![z, z]]);
    }
    let mut pts: Vec<(Integer, Integer)> = Vec::new();
    let mut x = Integer::from(0);
    let max_x = n.clone().sqrt();
    while x <= max_x {
        let r = n.clone() - &x * &x;
        if is_perfect_square(&r) {
            let y = r.sqrt();
            if x <= y {
                pts.push((x.clone(), y.clone()));
                if x < y {
                    pts.push((y.clone(), x.clone()));
                }
            }
        }
        x += 1;
    }
    if pts.is_empty() {
        return DiophantineSolution::NoSolution;
    }
    let sols: Vec<Vec<ExprId>> = pts
        .into_iter()
        .map(|(xi, yi)| vec![pool.integer(xi), pool.integer(yi)])
        .collect();
    DiophantineSolution::Finite(sols)
}

fn solve_sum_two_squares(
    pool: &ExprPool,
    _a: &Integer,
    n: &Integer,
    _vx: ExprId,
    _vy: ExprId,
) -> DiophantineSolution {
    let rep = sum_two_squares_representatives(n);
    if !rep.is_empty() {
        let sols: Vec<Vec<ExprId>> = rep
            .into_iter()
            .map(|(xi, yi)| vec![pool.integer(xi), pool.integer(yi)])
            .collect();
        return DiophantineSolution::Finite(sols);
    }
    // Fallback when factorization failed or n has no two-square representation.
    solve_sum_two_squares_scan(pool, n)
}

// ---------------------------------------------------------------------------
// Pell machinery
// ---------------------------------------------------------------------------

/// Hard cap on the continued-fraction period of `√D` this solver will expand
/// when no [`crate::budget`] bounds the work.  The fundamental unit has
/// `Θ(L)` digits for a period of length `L`, and building it costs `Θ(L²)`, so
/// past this the solver refuses rather than run for hours.
const MAX_CF_PERIOD: usize = 200_000;

/// Hard cap on the `y`-sweep for `x² − D·y² = N` (see [`nagell_y_bound`]).
const MAX_Y_SWEEP: u64 = 5_000_000;

thread_local! {
    static BUDGET_TRIP: Cell<Option<BudgetError>> = const { Cell::new(None) };
}

/// The [`BudgetError`] that stopped the most recent [`diophantine`] call on this
/// thread, or `None` if that call was not stopped by a budget.
///
/// [`DiophantineError`] is an exhaustive public enum, so a budget trip is
/// reported as [`DiophantineError::Unsupported`] (an honest "gave up") and
/// this function tells bindings *why*, so they can raise the dedicated
/// budget-exceeded error carrying the `E-BUDGET-*` code.  Cleared at the start
/// of every [`diophantine`] call.
pub fn last_budget_trip() -> Option<BudgetError> {
    BUDGET_TRIP.with(|c| c.get())
}

/// Cooperative budget checkpoint, consulted every 64 iterations of a loop.
fn checkpoint(iter: u64) -> Result<(), DiophantineError> {
    if iter % 64 == 0 {
        if let Err(e) = crate::budget::check() {
            BUDGET_TRIP.with(|c| c.set(Some(e)));
            return Err(DiophantineError::Unsupported(format!(
                "stopped by the active budget: {e}"
            )));
        }
    }
    Ok(())
}

/// Continued fraction of `√d` for non-square `d > 0`: `(a0, [a1, …, aL])`,
/// where `a1 … aL` is one full period (so `aL = 2·a0`).
///
/// Standard recurrence: `m ← q·a − m`, `q ← (d − m²)/q` (always exact),
/// `a ← ⌊(a0 + m)/q⌋` (a *floor*, not an exact division).
fn sqrt_cf_period(d: &Integer) -> Result<(Integer, Vec<Integer>), DiophantineError> {
    let a0 = d.clone().sqrt();
    let two_a0 = Integer::from(&a0 * 2u32);
    let mut m = Integer::from(0);
    let mut q = Integer::from(1);
    let mut a = a0.clone();
    let mut period = Vec::new();
    loop {
        checkpoint(period.len() as u64)?;
        m = Integer::from(&q * &a) - &m;
        let num: Integer = d.clone() - Integer::from(&m * &m);
        q = num / &q; // exact by the theory of the expansion
        if q == 0 {
            return Err(DiophantineError::Unsupported(
                "D is a perfect square (no Pell unit)".into(),
            ));
        }
        a = Integer::from(&a0 + &m).div_floor(&q);
        period.push(a.clone());
        if a == two_a0 {
            return Ok((a0, period));
        }
        if period.len() > MAX_CF_PERIOD {
            return Err(DiophantineError::Unsupported(format!(
                "continued-fraction period of √{d} exceeds {MAX_CF_PERIOD} terms"
            )));
        }
    }
}

/// Visit the convergents `(p_k, q_k)` of `√d` for `k = 0 .. count-1` (with
/// their index), stopping early when `visit` returns `Some`.
fn walk_convergents<T>(
    a0: &Integer,
    period: &[Integer],
    count: usize,
    mut visit: impl FnMut(usize, &Integer, &Integer) -> Option<T>,
) -> Result<Option<T>, DiophantineError> {
    let mut p_prev = Integer::from(1);
    let mut q_prev = Integer::from(0);
    let mut p = a0.clone();
    let mut q = Integer::from(1);
    for k in 0..count {
        checkpoint(k as u64)?;
        if let Some(t) = visit(k, &p, &q) {
            return Ok(Some(t));
        }
        if k + 1 == count {
            break;
        }
        let a = &period[k % period.len()];
        let p_new = Integer::from(a * &p) + &p_prev;
        let q_new = Integer::from(a * &q) + &q_prev;
        p_prev = std::mem::replace(&mut p, p_new);
        q_prev = std::mem::replace(&mut q, q_new);
    }
    Ok(None)
}

fn pell_norm(h: &Integer, k: &Integer, d: &Integer) -> Integer {
    h.clone() * h - d.clone() * k * k
}

/// Units of `ℤ[√d]` for non-square `d > 0`, with the expansion they came from.
struct PellUnits {
    a0: Integer,
    period: Vec<Integer>,
    /// Fundamental solution of `x² − d·y² = 1`.
    plus: (Integer, Integer),
    /// Fundamental solution of `x² − d·y² = −1`; `Some` iff the period is odd.
    minus: Option<(Integer, Integer)>,
}

/// The convergent `p_{L−1}/q_{L−1}` (`L` = period length) has norm `(−1)^L`.
/// If `L` is even it is the fundamental `+1` unit and `x² − d·y² = −1` has no
/// solution; if `L` is odd it is the fundamental `−1` unit and its square is
/// the fundamental `+1` unit.
fn pell_units(d: &Integer) -> Result<PellUnits, DiophantineError> {
    let (a0, period) = sqrt_cf_period(d)?;
    let l = period.len();
    let (p, q) = walk_convergents(&a0, &period, l, |k, p, q| {
        (k + 1 == l).then(|| (p.clone(), q.clone()))
    })?
    .expect("the walk visits index L-1");
    if l % 2 == 0 {
        debug_assert_eq!(pell_norm(&p, &q, d), 1);
        Ok(PellUnits {
            a0,
            period,
            plus: (p, q),
            minus: None,
        })
    } else {
        debug_assert_eq!(pell_norm(&p, &q, d), -1);
        let x1 = Integer::from(&p * &p) + d.clone() * &q * &q;
        let y1 = Integer::from(&p * &q) * 2u32;
        Ok(PellUnits {
            a0,
            period,
            plus: (x1, y1),
            minus: Some((p, q)),
        })
    }
}

/// Nagell's bound: if `x² − d·y² = n` (`n ≠ 0`) is solvable, every solution
/// class contains a solution with `0 ≤ y ≤ Y`, where, with `(x1, y1)` the
/// fundamental `+1` unit,
/// `Y = y1·√(n / (2(x1+1)))` for `n > 0` and `Y = y1·√(|n| / (2(x1−1)))` for
/// `n < 0`.  Sweeping `y ∈ [0, Y]` is therefore a complete decision procedure.
fn nagell_y_bound(n: &Integer, x1: &Integer, y1: &Integer) -> Integer {
    let den: Integer = if n.cmp0().is_gt() {
        (x1.clone() + 1u32) * 2u32
    } else {
        (x1.clone() - 1u32) * 2u32
    };
    let num: Integer = Integer::from(y1 * y1) * n.clone().abs();
    (num / den).sqrt()
}

/// Smallest `y ∈ [0, bound]` with `n + d·y²` a perfect square.
fn pell_y_sweep(
    d: &Integer,
    n: &Integer,
    bound: u64,
) -> Result<Option<(Integer, Integer)>, DiophantineError> {
    for yi in 0..=bound {
        checkpoint(yi)?;
        let y = Integer::from(yi);
        let rhs: Integer = n.clone() + d.clone() * &y * &y;
        if rhs.cmp0().is_ge() && is_perfect_square(&rhs) {
            let x = rhs.sqrt();
            debug_assert_eq!(pell_norm(&x, &y, d), *n);
            return Ok(Some((x, y)));
        }
    }
    Ok(None)
}

/// Search the convergents of `√d` over two periods for `x² − d·y² = n/g²`
/// (`g² | n`), returning `g·(p, q)`.  When `|n| < √d` every primitive positive
/// solution is a convergent (Lagrange), and convergent norms repeat with the
/// period, so this search is complete in that regime.
fn pell_convergent_search(
    d: &Integer,
    n: &Integer,
    units: &PellUnits,
) -> Result<Option<(Integer, Integer)>, DiophantineError> {
    let n_abs = n.clone().abs();
    let mut targets: Vec<(Integer, Integer)> = Vec::new();
    let mut g = Integer::from(1);
    let mut iter = 0u64;
    while Integer::from(&g * &g) <= n_abs {
        checkpoint(iter)?;
        iter += 1;
        let g2 = Integer::from(&g * &g);
        if n.is_divisible(&g2) {
            targets.push((g.clone(), Integer::from(n / &g2)));
        }
        g += 1u32;
    }
    let count = 2 * units.period.len();
    walk_convergents(&units.a0, &units.period, count, |_, p, q| {
        let norm = pell_norm(p, q, d);
        targets
            .iter()
            .find(|(_, t)| *t == norm)
            .map(|(g, _)| (Integer::from(g * p), Integer::from(g * q)))
    })
}

/// Decide `x² − d·y² = n` (`d > 0` non-square, `n ∉ {0, 1}`), returning a
/// particular solution, `Ok(None)` when none exists (proved), or a refusal
/// when the instance is too large to decide.
fn pell_particular(
    d: &Integer,
    n: &Integer,
    units: &PellUnits,
) -> Result<Option<(Integer, Integer)>, DiophantineError> {
    if *n == -1 {
        return Ok(units.minus.clone());
    }
    let (x1, y1) = &units.plus;
    let bound = nagell_y_bound(n, x1, y1);
    if let Some(b) = bound.to_u64().filter(|b| *b <= MAX_Y_SWEEP) {
        return pell_y_sweep(d, n, b);
    }
    // The complete Nagell sweep is too long; the convergent search is complete
    // when |n| < √d and is otherwise still a sound way to find a solution.
    if let Some(s) = pell_convergent_search(d, n, units)? {
        return Ok(Some(s));
    }
    if Integer::from(n * n) < *d {
        return Ok(None);
    }
    if let Some(s) = pell_y_sweep(d, n, MAX_Y_SWEEP)? {
        return Ok(Some(s));
    }
    Err(DiophantineError::Unsupported(format!(
        "x² − {d}·y² = {n}: the complete search bound y ≤ {bound} is too large to decide solvability"
    )))
}

/// `x² − s²·y² = n` (`s ≥ 1`, `n ≠ 0`) factors as `(x − s·y)(x + s·y) = n`, so
/// the solutions are finite: for every factorisation `n = e·f`,
/// `x = (e + f)/2`, `y = (f − e)/(2s)` when both are integers.  Returns the
/// non-negative representatives (the same convention as `x² + y² = n`).
fn square_d_solutions(
    pool: &ExprPool,
    s: &Integer,
    n: &Integer,
) -> Result<DiophantineSolution, DiophantineError> {
    let n_abs = n.clone().abs();
    let mut divisors = vec![Integer::from(1)];
    if n_abs > 1 {
        for (p, e) in factor_positive(n_abs.clone()) {
            if !is_probable_prime(&p) {
                return Err(DiophantineError::Unsupported(format!(
                    "could not factor {n_abs} to enumerate x² − {}·y² = {n}",
                    Integer::from(s * s)
                )));
            }
            let mut next = Vec::with_capacity(divisors.len() * (e as usize + 1));
            for dv in &divisors {
                let mut pk = dv.clone();
                for _ in 0..=e {
                    next.push(pk.clone());
                    pk *= &p;
                }
            }
            divisors = next;
        }
    }
    let two_s = Integer::from(s * 2u32);
    let mut pts: std::collections::BTreeSet<(Integer, Integer)> = Default::default();
    for (i, e) in divisors.iter().enumerate() {
        checkpoint(i as u64)?;
        let f = Integer::from(n / e);
        let sum = Integer::from(e + &f);
        let diff = Integer::from(&f - e);
        if sum.is_even() && diff.is_divisible(&two_s) {
            pts.insert((sum.abs() / 2u32, diff.abs() / &two_s));
        }
    }
    if pts.is_empty() {
        return Ok(DiophantineSolution::NoSolution);
    }
    Ok(DiophantineSolution::Finite(
        pts.into_iter()
            .map(|(x, y)| vec![pool.integer(x), pool.integer(y)])
            .collect(),
    ))
}

fn solve_pell_like(
    pool: &ExprPool,
    pos: &Integer,
    neg: &Integer,
    rhs: &Integer,
) -> Result<DiophantineSolution, DiophantineError> {
    if *pos == 0 || *neg == 0 {
        return Err(DiophantineError::Unsupported("degenerate quadratic".into()));
    }
    let g = pos.clone().gcd(neg).gcd(&rhs.clone().abs());
    let p = div_exact(pos, &g).unwrap();
    let nn = div_exact(neg, &g).unwrap();
    let r = div_exact(rhs, &g).unwrap();
    // p·X² - nn·Y² = r

    if r == 0 {
        // p·X² = nn·Y²: if nn/p or p/nn is a perfect square, parametrize; else only (0,0).
        if let Some(s2) = div_exact(&nn, &p) {
            if is_perfect_square(&s2) {
                let s = s2.sqrt();
                let t = pool.symbol("_t", Domain::Integer);
                let x_e = pool.mul(vec![pool.integer(s), t]);
                return Ok(DiophantineSolution::ParametricLinear {
                    parameter: t,
                    values: vec![x_e, t],
                });
            }
        }
        if let Some(t2) = div_exact(&p, &nn) {
            if is_perfect_square(&t2) {
                let tc = t2.sqrt();
                let t = pool.symbol("_t", Domain::Integer);
                let y_e = pool.mul(vec![pool.integer(tc), t]);
                return Ok(DiophantineSolution::ParametricLinear {
                    parameter: t,
                    values: vec![t, y_e],
                });
            }
        }
        let z = pool.integer(0);
        return Ok(DiophantineSolution::Finite(vec![vec![z, z]]));
    }

    let g2 = p.clone().gcd(&nn);
    let (_, rem) = r.clone().div_rem_euc_ref(&g2).into();
    if rem != 0 {
        return Ok(DiophantineSolution::NoSolution);
    }
    let p2 = div_exact(&p, &g2).unwrap();
    let n2 = div_exact(&nn, &g2).unwrap();
    let r2 = div_exact(&r, &g2).unwrap();

    if p2 != 1 {
        return Err(DiophantineError::Unsupported(
            "Pell-type equation must reduce to x² - D·y² = N (leading x² coefficient 1 after gcd)"
                .into(),
        ));
    }

    if is_perfect_square(&n2) {
        return square_d_solutions(pool, &n2.sqrt(), &r2);
    }

    let units = pell_units(&n2)?;
    let (ux, uy) = units.plus.clone();

    if r2 == 1 {
        return Ok(DiophantineSolution::PellFundamental {
            d: pool.integer(n2),
            x0: pool.integer(ux),
            y0: pool.integer(uy),
        });
    }

    let Some(part) = pell_particular(&n2, &r2, &units)? else {
        return Ok(DiophantineSolution::NoSolution);
    };

    Ok(DiophantineSolution::PellGeneralized {
        d: pool.integer(n2.clone()),
        n: pool.integer(r2),
        x0: pool.integer(part.0),
        y0: pool.integer(part.1),
        unit_x: pool.integer(ux),
        unit_y: pool.integer(uy),
    })
}

fn solve_linear_two_var(
    pool: &ExprPool,
    a: &Integer,
    b: &Integer,
    c: &Integer,
    _vx: ExprId,
    _vy: ExprId,
) -> Result<DiophantineSolution, DiophantineError> {
    let rhs = -c.clone();
    let g = a.clone().gcd(b);
    let (_, rem) = rhs.clone().div_rem_euc_ref(&g).into();
    if rem != 0 {
        return Ok(DiophantineSolution::NoSolution);
    }
    let (g0, u, v) = extended_gcd(a, b);
    debug_assert_eq!(g0, g);
    let a1 = div_exact(a, &g).unwrap();
    let b1 = div_exact(b, &g).unwrap();
    let rhs1 = div_exact(&rhs, &g).unwrap();
    let x0 = &u * &rhs1;
    let y0 = &v * &rhs1;
    let t = pool.symbol("_t", Domain::Integer);
    let bt = pool.mul(vec![pool.integer(b1.clone()), t]);
    let neg_one = pool.integer(-1_i32);
    let neg_at = pool.mul(vec![neg_one, pool.integer(a1.clone()), t]);
    let xt = pool.add(vec![pool.integer(x0), bt]);
    let yt = pool.add(vec![pool.integer(y0), neg_at]);
    Ok(DiophantineSolution::ParametricLinear {
        parameter: t,
        values: vec![xt, yt],
    })
}

fn classify_and_solve(
    pool: &ExprPool,
    terms: &BTreeMap<Vec<u32>, Integer>,
    vars: &[ExprId],
) -> Result<DiophantineSolution, DiophantineError> {
    if vars.len() != 2 {
        return Err(DiophantineError::Unsupported(
            "exactly two variables are required".into(),
        ));
    }
    let vx = vars[0];
    let vy = vars[1];

    let mut max_deg = 0u32;
    for e in terms.keys() {
        // `expr_to_gbpoly` bounds every total degree by u32::MAX.
        let tdeg: u32 = crate::poly::exponent::total_degree_or_panic(e);
        max_deg = max_deg.max(tdeg);
    }

    if max_deg > 2 {
        return Err(DiophantineError::Unsupported(
            "degree > 2 is not supported".into(),
        ));
    }

    if max_deg <= 1 {
        let c00 = terms
            .get(&vec![0, 0])
            .cloned()
            .unwrap_or_else(|| Integer::from(0));
        let c10 = terms
            .get(&vec![1, 0])
            .cloned()
            .unwrap_or_else(|| Integer::from(0));
        let c01 = terms
            .get(&vec![0, 1])
            .cloned()
            .unwrap_or_else(|| Integer::from(0));
        if terms.len() > 3 {
            return Err(DiophantineError::Unsupported(
                "linear equation with unexpected monomials".into(),
            ));
        }
        for e in terms.keys() {
            let s: u32 = crate::poly::exponent::total_degree_or_panic(e);
            if s > 1 {
                return Err(DiophantineError::Unsupported(
                    "mixed-degree polynomial".into(),
                ));
            }
        }
        return solve_linear_two_var(pool, &c10, &c01, &c00, vx, vy);
    }

    let c20 = terms
        .get(&vec![2, 0])
        .cloned()
        .unwrap_or_else(|| Integer::from(0));
    let c11 = terms
        .get(&vec![1, 1])
        .cloned()
        .unwrap_or_else(|| Integer::from(0));
    let c02 = terms
        .get(&vec![0, 2])
        .cloned()
        .unwrap_or_else(|| Integer::from(0));
    let c10 = terms
        .get(&vec![1, 0])
        .cloned()
        .unwrap_or_else(|| Integer::from(0));
    let c01 = terms
        .get(&vec![0, 1])
        .cloned()
        .unwrap_or_else(|| Integer::from(0));
    let c00 = terms
        .get(&vec![0, 0])
        .cloned()
        .unwrap_or_else(|| Integer::from(0));

    if c10 != 0 || c01 != 0 || c11 != 0 {
        return Err(DiophantineError::Unsupported(
            "quadratic with linear or xy terms is not implemented".into(),
        ));
    }

    let g_content = term_gcd(&[c20.clone(), c02.clone(), c00.clone()]);
    if g_content == 0 {
        return Err(DiophantineError::Unsupported("zero polynomial".into()));
    }
    let a2 = div_exact(&c20, &g_content).unwrap();
    let b2 = div_exact(&c02, &g_content).unwrap();
    let cc = div_exact(&c00, &g_content).unwrap();

    if a2 == 0 && b2 == 0 {
        return Err(DiophantineError::Unsupported("no quadratic terms".into()));
    }

    // Normalise so the x² coefficient is positive: the solution set is
    // unchanged, and every branch below can assume `a2 > 0`.  (Taking `|a2|`
    // without negating `cc` flipped the sign of `n` in the ellipse case, and
    // swapping the roles of x and y in the hyperbola case reported the
    // solution of `y² − D·x² = N` as values for `(x, y)`.)
    let (a2, b2, cc) = if a2 < 0 {
        (-a2, -b2, -cc)
    } else {
        (a2, b2, cc)
    };

    if a2 > 0 && b2 > 0 {
        if a2 != b2 {
            return Err(DiophantineError::Unsupported(
                "x² and y² must have equal coefficients for the ellipse case".into(),
            ));
        }
        // a·(x² + y²) + cc = 0
        let (_, rem) = cc.clone().div_rem_euc_ref(&a2).into();
        if rem != 0 {
            return Ok(DiophantineSolution::NoSolution);
        }
        let n = -cc / &a2;
        if n < 0 {
            return Ok(DiophantineSolution::NoSolution);
        }
        return Ok(solve_sum_two_squares(pool, &a2, &n, vx, vy));
    }

    if a2 > 0 && b2 < 0 {
        // pos·x² − neg·y² = rhs
        let pos = a2;
        let neg = -b2;
        let rhs = -cc;

        if rhs == 0 {
            // pos·x² = neg·y² has non-zero solutions iff pos/neg is a rational
            // square: with g = gcd, pos/g = α², neg/g = β², the solutions are
            // α·x = ±β·y, i.e. (x, y) = (β·t, ±α·t).
            let g = pos.clone().gcd(&neg);
            let p1 = pos / &g;
            let n1 = neg / &g;
            if !is_perfect_square(&p1) || !is_perfect_square(&n1) {
                let z = pool.integer(0);
                return Ok(DiophantineSolution::Finite(vec![vec![z, z]]));
            }
            let alpha = p1.sqrt();
            let beta = n1.sqrt();
            let t = pool.symbol("_t", Domain::Integer);
            let scaled = |c: Integer| {
                if c == 1 {
                    t
                } else {
                    pool.mul(vec![pool.integer(c), t])
                }
            };
            return Ok(DiophantineSolution::ParametricLinear {
                parameter: t,
                values: vec![scaled(beta), scaled(alpha)],
            });
        }

        return solve_pell_like(pool, &pos, &neg, &rhs);
    }

    Err(DiophantineError::Unsupported(
        "unrecognized binary quadratic shape".into(),
    ))
}

/// Solve a single Diophantine equation in integer unknowns.
pub fn diophantine(
    pool: &ExprPool,
    equation: ExprId,
    vars: &[ExprId],
) -> Result<DiophantineSolution, DiophantineError> {
    if vars.len() != 2 {
        return Err(DiophantineError::Unsupported(
            "exactly two variables are required".into(),
        ));
    }
    BUDGET_TRIP.with(|c| c.set(None));
    let poly = expr_to_gbpoly(equation, vars, pool)?;
    let int_terms = gbpoly_integer_coeffs(&poly)?;
    for c in poly.terms.values() {
        if !c.is_integer() {
            return Err(DiophantineError::NonIntegerCoefficients);
        }
    }
    classify_and_solve(pool, &int_terms, vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{ExprData, ExprPool};

    #[test]
    fn linear_3x_5y_1() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Integer);
        let y = pool.symbol("y", Domain::Integer);
        let eq = pool.add(vec![
            pool.mul(vec![pool.integer(3), x]),
            pool.mul(vec![pool.integer(5), y]),
            pool.integer(-1),
        ]);
        let r = diophantine(&pool, eq, &[x, y]).unwrap();
        match r {
            DiophantineSolution::ParametricLinear { .. } => {}
            _ => panic!("expected parametric linear"),
        }
    }

    #[test]
    fn pell_x2_2y2_1() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Integer);
        let y = pool.symbol("y", Domain::Integer);
        let x2 = pool.pow(x, pool.integer(2));
        let y2 = pool.pow(y, pool.integer(2));
        let eq = pool.add(vec![
            x2,
            pool.mul(vec![pool.integer(-2), y2]),
            pool.integer(-1),
        ]);
        let r = diophantine(&pool, eq, &[x, y]).unwrap();
        match r {
            DiophantineSolution::PellFundamental { x0, y0, .. } => {
                assert!(pool.with(x0, |d| matches!(d, ExprData::Integer(n) if n.0 == 3)));
                assert!(pool.with(y0, |d| matches!(d, ExprData::Integer(n) if n.0 == 2)));
            }
            _ => panic!("expected Pell fundamental"),
        }
    }

    #[test]
    fn sum_squares_5() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Integer);
        let y = pool.symbol("y", Domain::Integer);
        let eq = pool.add(vec![
            pool.pow(x, pool.integer(2)),
            pool.pow(y, pool.integer(2)),
            pool.integer(-5),
        ]);
        let r = diophantine(&pool, eq, &[x, y]).unwrap();
        match r {
            DiophantineSolution::Finite(v) => {
                assert_eq!(v.len(), 2);
            }
            _ => panic!("expected finite set"),
        }
    }

    #[test]
    fn sum_squares_65_two_orbits() {
        // 65 = 1²+8² = 4²+7²
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Integer);
        let y = pool.symbol("y", Domain::Integer);
        let eq = pool.add(vec![
            pool.pow(x, pool.integer(2)),
            pool.pow(y, pool.integer(2)),
            pool.integer(-65),
        ]);
        let r = diophantine(&pool, eq, &[x, y]).unwrap();
        match r {
            DiophantineSolution::Finite(v) => {
                let sets: std::collections::HashSet<(i32, i32)> = v
                    .iter()
                    .map(|row| {
                        let xi = match pool.get(row[0]) {
                            ExprData::Integer(i) => i.0.to_i32().unwrap(),
                            _ => panic!(),
                        };
                        let yi = match pool.get(row[1]) {
                            ExprData::Integer(i) => i.0.to_i32().unwrap(),
                            _ => panic!(),
                        };
                        (xi, yi)
                    })
                    .collect();
                assert!(sets.contains(&(1, 8)));
                assert!(sets.contains(&(8, 1)));
                assert!(sets.contains(&(4, 7)));
                assert!(sets.contains(&(7, 4)));
            }
            _ => panic!("expected finite set"),
        }
    }

    #[test]
    fn pell_generalized_n_minus1() {
        // x² - 2 y² = -1  →  (1,1) fundamental for negative Pell
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Integer);
        let y = pool.symbol("y", Domain::Integer);
        let eq = pool.add(vec![
            pool.pow(x, pool.integer(2)),
            pool.mul(vec![pool.integer(-2), pool.pow(y, pool.integer(2))]),
            pool.integer(1),
        ]);
        let r = diophantine(&pool, eq, &[x, y]).unwrap();
        match r {
            DiophantineSolution::PellGeneralized { .. } => {}
            DiophantineSolution::PellFundamental { .. } => {
                // tolerate unit-path implementation detail
            }
            _ => panic!("expected Pell generalized or fundamental: {:?}", r),
        }
    }

    #[test]
    fn linear_no_solution() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Integer);
        let y = pool.symbol("y", Domain::Integer);
        let eq = pool.add(vec![
            pool.mul(vec![pool.integer(2), x]),
            pool.mul(vec![pool.integer(4), y]),
            pool.integer(1),
        ]);
        let r = diophantine(&pool, eq, &[x, y]).unwrap();
        assert!(matches!(r, DiophantineSolution::NoSolution));
    }

    fn int_of(pool: &ExprPool, e: ExprId) -> Integer {
        match pool.get(e) {
            ExprData::Integer(i) => i.0.clone(),
            other => panic!("expected integer, got {other:?}"),
        }
    }

    /// `c20·x² + c02·y² + c00`.
    fn quad(pool: &ExprPool, c20: i64, c02: i64, c00: i64) -> (ExprId, [ExprId; 2]) {
        let x = pool.symbol("x", Domain::Integer);
        let y = pool.symbol("y", Domain::Integer);
        let eq = pool.add(vec![
            pool.mul(vec![pool.integer(c20), pool.pow(x, pool.integer(2))]),
            pool.mul(vec![pool.integer(c02), pool.pow(y, pool.integer(2))]),
            pool.integer(c00),
        ]);
        (eq, [x, y])
    }

    /// Independent fundamental-unit oracle: smallest `y ≥ 1` with `1 + d·y²` square.
    fn brute_pell(d: u64) -> (u64, u64) {
        let mut y = 1u64;
        loop {
            let r = 1 + d * y * y;
            let x = (r as f64).sqrt() as u64;
            for cand in x.saturating_sub(1)..=x + 1 {
                if cand * cand == r {
                    return (cand, y);
                }
            }
            y += 1;
        }
    }

    #[test]
    fn pell_fundamental_matches_brute_force() {
        // Every non-square D ≤ 60 except those with huge units (brute force is slow).
        for d in 2u64..=60 {
            if Integer::from(d).is_perfect_square() || [46, 53, 58].contains(&d) {
                continue;
            }
            let (bx, by) = brute_pell(d);
            let units = pell_units(&Integer::from(d)).unwrap();
            assert_eq!(units.plus, (Integer::from(bx), Integer::from(by)), "D={d}");
        }
        // The report's failures, including the big ones.
        let pool = ExprPool::new();
        for (d, x0, y0) in [
            (13i64, "649", "180"),
            (61, "1766319049", "226153980"),
            (94, "2143295", "221064"),
            (46, "24335", "3588"),
        ] {
            let (eq, v) = quad(&pool, 1, -d, -1);
            match diophantine(&pool, eq, &v).unwrap() {
                DiophantineSolution::PellFundamental { x0: a, y0: b, .. } => {
                    assert_eq!(int_of(&pool, a), x0.parse::<Integer>().unwrap(), "D={d}");
                    assert_eq!(int_of(&pool, b), y0.parse::<Integer>().unwrap(), "D={d}");
                }
                r => panic!("D={d}: {r:?}"),
            }
        }
    }

    #[test]
    fn negative_pell_solvable_iff_period_odd() {
        let pool = ExprPool::new();
        // D ∈ {2,5,10,13,17,26,29,37,41,50,53,58,61,65,73,74,82,85,89,97} are solvable below 100.
        let solvable = [
            2, 5, 10, 13, 17, 26, 29, 37, 41, 50, 53, 58, 61, 65, 73, 74, 82, 85, 89, 97,
        ];
        for d in 2i64..100 {
            if Integer::from(d).is_perfect_square() {
                continue;
            }
            let (eq, v) = quad(&pool, 1, -d, 1);
            let r = diophantine(&pool, eq, &v).unwrap();
            match r {
                DiophantineSolution::PellGeneralized { x0, y0, .. } => {
                    assert!(solvable.contains(&d), "D={d} reported solvable");
                    let (x, y) = (int_of(&pool, x0), int_of(&pool, y0));
                    assert_eq!(pell_norm(&x, &y, &Integer::from(d)), -1, "D={d}");
                }
                DiophantineSolution::NoSolution => {
                    assert!(!solvable.contains(&d), "D={d} reported unsolvable")
                }
                r => panic!("D={d}: {r:?}"),
            }
        }
    }

    #[test]
    fn generalized_pell_matches_sweep_oracle() {
        // x² − D·y² = N against a direct y ≤ 2000 search (all such instances
        // here have a solution with y well below that if they have one at all,
        // by Nagell's bound for these small units).
        let pool = ExprPool::new();
        for (d, n) in [
            (7i64, 2i64),
            (7, 3),
            (13, 3),
            (13, -3),
            (21, 4),
            (6, -2),
            (3, -1),
            (11, 5),
        ] {
            let (eq, v) = quad(&pool, 1, -d, -n);
            let r = diophantine(&pool, eq, &v).unwrap();
            let oracle = (0i64..2000).any(|y| {
                let r = n + d * y * y;
                r >= 0 && Integer::from(r).is_perfect_square()
            });
            match r {
                DiophantineSolution::PellGeneralized { x0, y0, .. } => {
                    let (x, y) = (int_of(&pool, x0), int_of(&pool, y0));
                    assert_eq!(pell_norm(&x, &y, &Integer::from(d)), n, "D={d} N={n}");
                    assert!(oracle);
                }
                DiophantineSolution::NoSolution => assert!(!oracle, "D={d} N={n}"),
                r => panic!("D={d} N={n}: {r:?}"),
            }
        }
    }

    #[test]
    fn negated_and_swapped_quadratics() {
        let pool = ExprPool::new();
        // −x² − y² + 5 = 0  ⇔  x² + y² = 5
        let (eq, v) = quad(&pool, -1, -1, 5);
        assert!(matches!(
            diophantine(&pool, eq, &v).unwrap(),
            DiophantineSolution::Finite(ref s) if s.len() == 2
        ));
        // −x² + 2y² + 1 = 0  ⇔  x² − 2y² = 1  (x0 = 3, y0 = 2 in vars order)
        let (eq, v) = quad(&pool, -1, 2, 1);
        match diophantine(&pool, eq, &v).unwrap() {
            DiophantineSolution::PellFundamental { x0, y0, .. } => {
                assert_eq!(int_of(&pool, x0), 3);
                assert_eq!(int_of(&pool, y0), 2);
            }
            r => panic!("{r:?}"),
        }
        // y² − 2x² = 1 is not of the form x² − D·y² = N: refuse, not swap.
        let (eq, v) = quad(&pool, -2, 1, -1);
        assert!(matches!(
            diophantine(&pool, eq, &v),
            Err(DiophantineError::Unsupported(_))
        ));
        // 4x² − 9y² = 0  ⇒  (x, y) = (3t, 2t)
        let (eq, v) = quad(&pool, 4, -9, 0);
        match diophantine(&pool, eq, &v).unwrap() {
            DiophantineSolution::ParametricLinear { parameter, values } => {
                let t = parameter;
                assert_eq!(values[0], pool.mul(vec![pool.integer(3), t]));
                assert_eq!(values[1], pool.mul(vec![pool.integer(2), t]));
            }
            r => panic!("{r:?}"),
        }
        // x² − 4y² = 5  ⇒  (3, 1) only
        let (eq, v) = quad(&pool, 1, -4, -5);
        match diophantine(&pool, eq, &v).unwrap() {
            DiophantineSolution::Finite(s) => {
                assert_eq!(s.len(), 1);
                assert_eq!(int_of(&pool, s[0][0]), 3);
                assert_eq!(int_of(&pool, s[0][1]), 1);
            }
            r => panic!("{r:?}"),
        }
        // x² − 4y² = 2 has no solution
        let (eq, v) = quad(&pool, 1, -4, -2);
        assert!(matches!(
            diophantine(&pool, eq, &v).unwrap(),
            DiophantineSolution::NoSolution
        ));
    }

    #[test]
    fn budget_stops_the_expansion() {
        use crate::budget::{self, Budget};
        let pool = ExprPool::new();
        let (eq, v) = quad(&pool, 1, -13, -1);
        let r = {
            let _guard = budget::enter(Budget::new().with_max_steps(0));
            diophantine(&pool, eq, &v)
        };
        assert!(matches!(r, Err(DiophantineError::Unsupported(_))), "{r:?}");
        assert!(last_budget_trip().is_some());
    }

    #[test]
    fn cornacchia_prime_13() {
        let p = Integer::from(13);
        let r = prime_as_sum_two_squares(&p).unwrap();
        assert_eq!(r.0.clone() * &r.0 + r.1.clone() * &r.1, p);
    }
}
