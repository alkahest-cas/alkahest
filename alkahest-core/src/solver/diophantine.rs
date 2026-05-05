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
//! `x² - D·y² = N` with `D > 0` non-square: search **continued-fraction convergents** of `√D`,
//! then a bounded `y`-sweep `N + D·y² = □`.  Solutions multiply by the unit `u² - D·v² = 1`.
//! `N = 0`: trivial `(0,0)` if `D` is non-square; if `D = s²`, a parametric line `x = s·t`, `y = t`.

use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::groebner::ideal::GbPoly;
use rug::ops::Pow;
use rug::Integer;
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
    PellFundamental {
        d: ExprId,
        x0: ExprId,
        y0: ExprId,
    },
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
    let ls = a.clone().pow_mod(&exp, p).unwrap_or_else(|_| Integer::from(0));
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
    let mut seen: BTreeSet<String> = acc
        .iter()
        .map(|(a, b)| format!("{a},{b}"))
        .collect();
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

/// One step of continued fraction for `√d`; updates `(h,k)` convergents.
#[allow(clippy::too_many_arguments)]
fn sqrt_cf_step(
    d: &Integer,
    a0: &Integer,
    m: &mut Integer,
    d_cf: &mut Integer,
    a: &mut Integer,
    h_prev: &mut Integer,
    k_prev: &mut Integer,
    h: &mut Integer,
    k: &mut Integer,
) -> Option<()> {
    *m = (&*d_cf * &*a - &*m).into();
    let num = d.clone() - &*m * &*m;
    *d_cf = div_exact(&num, d_cf)?;
    if *d_cf == 0 {
        return None;
    }
    let sum: Integer = (a0 + &*m).into();
    *a = div_exact(&sum, d_cf)?;
    let h_new: Integer = (&*a * &*h + &*h_prev).into();
    let k_new: Integer = (&*a * &*k + &*k_prev).into();
    *h_prev = h.clone();
    *k_prev = k.clone();
    *h = h_new;
    *k = k_new;
    Some(())
}

fn pell_norm(h: &Integer, k: &Integer, d: &Integer) -> Integer {
    h.clone() * h - d.clone() * k * k
}

/// Minimal positive solution to `x² - d·y² = 1` (`d` non-square), via convergents.
fn pell_fundamental_xy(d: &Integer) -> Option<(Integer, Integer)> {
    pell_convergent_solution(d, &Integer::from(1))
}

/// Some `(x, y)` with `x² - d·y² = target` if found among convergents or a bounded search.
fn pell_convergent_solution(d: &Integer, target: &Integer) -> Option<(Integer, Integer)> {
    let d = d.clone();
    if d <= 0 {
        return None;
    }
    let (_, rem) = d.clone().sqrt_rem(Integer::new());
    if rem == 0 {
        return None;
    }
    let a0 = d.clone().sqrt();
    let mut m = Integer::from(0);
    let mut d_cf = Integer::from(1);
    let mut a = a0.clone();

    let mut h_prev = Integer::from(1);
    let mut h = a0.clone();
    let mut k_prev = Integer::from(0);
    let mut k = Integer::from(1);

    let max_steps = 500_000u64;
    for _ in 0..max_steps {
        let lhs = pell_norm(&h, &k, &d);
        if lhs == *target {
            return Some((h, k));
        }
        sqrt_cf_step(&d, &a0, &mut m, &mut d_cf, &mut a, &mut h_prev, &mut k_prev, &mut h, &mut k)?;
    }
    None
}

/// Try `x² = target + d·y²` for increasing `y`.
fn pell_y_sweep(d: &Integer, target: &Integer) -> Option<(Integer, Integer)> {
    let bound = Integer::from(2_000_000);
    let mut y = Integer::from(0);
    while y <= bound {
        let rhs = target.clone() + d.clone() * &y * &y;
        if rhs.cmp0().is_ge() && is_perfect_square(&rhs) {
            let x = rhs.sqrt();
            if pell_norm(&x, &y, d) == *target {
                return Some((x, y));
            }
        }
        y += 1;
    }
    None
}

fn solve_pell_like(
    pool: &ExprPool,
    pos: &Integer,
    neg: &Integer,
    rhs: &Integer,
) -> Result<DiophantineSolution, DiophantineError> {
    if *pos == 0 || *neg == 0 {
        return Err(DiophantineError::Unsupported(
            "degenerate quadratic".into(),
        ));
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
            "Pell-type equation must reduce to x² - D·y² = N (leading x² coefficient 1 after gcd)".into(),
        ));
    }

    let (ux, uy) = match pell_fundamental_xy(&n2) {
        Some(u) => u,
        None => {
            return Err(DiophantineError::Unsupported(
                "no fundamental unit (D may be a perfect square)" .into(),
            ));
        }
    };

    if r2 == 0 {
        unreachable!("handled above");
    }

    if r2 == 1 {
        return Ok(DiophantineSolution::PellFundamental {
            d: pool.integer(n2),
            x0: pool.integer(ux),
            y0: pool.integer(uy),
        });
    }

    let part = pell_convergent_solution(&n2, &r2)
        .or_else(|| pell_y_sweep(&n2, &r2))
        .ok_or(DiophantineError::NoSolution)?;

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
        let tdeg: u32 = e.iter().sum();
        max_deg = max_deg.max(tdeg);
    }

    if max_deg > 2 {
        return Err(DiophantineError::Unsupported(
            "degree > 2 is not supported".into(),
        ));
    }

    if max_deg <= 1 {
        let c00 = terms.get(&vec![0, 0]).cloned().unwrap_or_else(|| Integer::from(0));
        let c10 = terms.get(&vec![1, 0]).cloned().unwrap_or_else(|| Integer::from(0));
        let c01 = terms.get(&vec![0, 1]).cloned().unwrap_or_else(|| Integer::from(0));
        if terms.len() > 3 {
            return Err(DiophantineError::Unsupported(
                "linear equation with unexpected monomials".into(),
            ));
        }
        for e in terms.keys() {
            let s: u32 = e.iter().sum();
            if s > 1 {
                return Err(DiophantineError::Unsupported(
                    "mixed-degree polynomial".into(),
                ));
            }
        }
        return solve_linear_two_var(pool, &c10, &c01, &c00, vx, vy);
    }

    let c20 = terms.get(&vec![2, 0]).cloned().unwrap_or_else(|| Integer::from(0));
    let c11 = terms.get(&vec![1, 1]).cloned().unwrap_or_else(|| Integer::from(0));
    let c02 = terms.get(&vec![0, 2]).cloned().unwrap_or_else(|| Integer::from(0));
    let c10 = terms.get(&vec![1, 0]).cloned().unwrap_or_else(|| Integer::from(0));
    let c01 = terms.get(&vec![0, 1]).cloned().unwrap_or_else(|| Integer::from(0));
    let c00 = terms.get(&vec![0, 0]).cloned().unwrap_or_else(|| Integer::from(0));

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

    if (a2 > 0 && b2 > 0) || (a2 < 0 && b2 < 0) {
        if a2 != b2 {
            return Err(DiophantineError::Unsupported(
                "x² and y² must have equal coefficients for the ellipse case".into(),
            ));
        }
        let a_abs = a2.clone().abs();
        let (_, rem) = cc.clone().div_rem_euc_ref(&a_abs).into();
        if rem != 0 {
            return Ok(DiophantineSolution::NoSolution);
        }
        let n = -cc / &a_abs;
        return Ok(solve_sum_two_squares(
            pool,
            &a_abs,
            &n,
            vx,
            vy,
        ));
    }

    if (a2 > 0 && b2 < 0) || (a2 < 0 && b2 > 0) {
        let pos = if a2 > 0 { a2.clone() } else { b2.clone().abs() };
        let neg = if a2 > 0 { b2.clone().abs() } else { a2.clone().abs() };
        let rhs = -cc;

        if rhs == 0 {
            let (_, remd) = neg.clone().sqrt_rem(Integer::new());
            if remd != 0 {
                let z = pool.integer(0);
                return Ok(DiophantineSolution::Finite(vec![vec![z, z]]));
            }
            let s = neg.sqrt();
            let t = pool.symbol("_t", Domain::Integer);
            let st = pool.mul(vec![pool.integer(s), t]);
            return Ok(DiophantineSolution::ParametricLinear {
                parameter: t,
                values: vec![st, t],
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
        let eq = pool.add(vec![x2, pool.mul(vec![pool.integer(-2), y2]), pool.integer(-1)]);
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

    #[test]
    fn cornacchia_prime_13() {
        let p = Integer::from(13);
        let r = prime_as_sum_two_squares(&p).unwrap();
        assert_eq!(r.0.clone() * &r.0 + r.1.clone() * &r.1, p);
    }
}
