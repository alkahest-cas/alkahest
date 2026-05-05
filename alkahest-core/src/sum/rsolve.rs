//! Linear difference equations with constant coefficients (V2-18).
//!
//! `rsolve` accepts an equation linear in `seq_name(n + offset)` applications with
//! integer offsets, rational multipliers, and a right-hand side polynomial in `n`.

#![allow(clippy::needless_range_loop)]

use crate::kernel::subs::subs;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;
use crate::poly::unipoly::UniPoly;
use crate::simplify::engine::simplify;
use rug::{Integer, Rational};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

fn simp(pool: &ExprPool, e: ExprId) -> ExprId {
    simplify(e, pool).value
}

/// True when ``r`` is multiplicative unity (checks canonical `numer/denom`; also matches `±k/±k`).
#[inline]
fn rational_eq_one(r: &Rational) -> bool {
    !r.is_zero() && r.numer() == r.denom()
}

/// Errors from [`rsolve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RsolveError {
    /// Equation shape is not a supported linear recurrence in `seq_name`.
    NotLinearRecurrence(String),
    /// Non-constant-coefficient factor (e.g. `n*f(n)`).
    NonlinearTerm,
    /// Right-hand side is not a polynomial in `n`.
    NonPolynomialRhs(String),
    /// Order or characteristic factorization outside the supported fragment.
    Unsupported(String),
    /// Initial values do not fix the constants (singular or wrong count).
    InitialMismatch(String),
}

impl fmt::Display for RsolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RsolveError::NotLinearRecurrence(s) => write!(f, "rsolve: {s}"),
            RsolveError::NonlinearTerm => write!(f, "rsolve: nonlinear term in sequence variable"),
            RsolveError::NonPolynomialRhs(s) => write!(f, "rsolve: non-polynomial rhs: {s}"),
            RsolveError::Unsupported(s) => write!(f, "rsolve: unsupported: {s}"),
            RsolveError::InitialMismatch(s) => write!(f, "rsolve: initial values: {s}"),
        }
    }
}

impl std::error::Error for RsolveError {}

impl crate::errors::AlkahestError for RsolveError {
    fn code(&self) -> &'static str {
        match self {
            RsolveError::NotLinearRecurrence(_) => "E-RSOLVE-001",
            RsolveError::NonlinearTerm => "E-RSOLVE-002",
            RsolveError::NonPolynomialRhs(_) => "E-RSOLVE-003",
            RsolveError::Unsupported(_) => "E-RSOLVE-004",
            RsolveError::InitialMismatch(_) => "E-RSOLVE-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "use pool.func(name, [n + integer]) for shifts; keep coefficients rational and rhs polynomial in n",
        )
    }
}

fn rational_atom(pool: &ExprPool, r: &Rational) -> ExprId {
    let numer = r.numer().clone();
    let denom = r.denom().clone();
    if denom == 1 {
        pool.integer(numer)
    } else {
        pool.rational(numer, denom)
    }
}

fn expr_div(pool: &ExprPool, num: ExprId, den: ExprId) -> ExprId {
    pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
}

fn flatten_add(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(expr) {
        ExprData::Add(args) => args.iter().flat_map(|&x| flatten_add(x, pool)).collect(),
        _ => vec![expr],
    }
}

fn flatten_mul(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(expr) {
        ExprData::Mul(args) => args.iter().flat_map(|&x| flatten_mul(x, pool)).collect(),
        _ => vec![expr],
    }
}

fn linear_in_sym(expr: ExprId, sym: ExprId, pool: &ExprPool) -> Option<(Rational, Rational)> {
    let e = simp(pool, expr);
    if e == sym {
        return Some((Rational::from(1), Rational::from(0)));
    }
    match pool.get(e) {
        ExprData::Integer(n) => Some((Rational::from(0), Rational::from(n.0.clone()))),
        ExprData::Rational(r) => Some((Rational::from(0), r.0.clone())),
        ExprData::Add(args) => {
            let mut a = Rational::from(0);
            let mut b = Rational::from(0);
            for t in args {
                if t == sym {
                    a += Rational::from(1);
                } else if let Some((a0, b0)) = linear_in_sym(t, sym, pool) {
                    a += a0;
                    b += b0;
                } else {
                    return None;
                }
            }
            Some((a, b))
        }
        ExprData::Mul(args) => {
            if args.len() == 2 && args[0] == sym {
                match pool.get(args[1]) {
                    ExprData::Integer(n) => Some((Rational::from(n.0.clone()), Rational::from(0))),
                    ExprData::Rational(r) => Some((r.0.clone(), Rational::from(0))),
                    _ => None,
                }
            } else if args.len() == 2 && args[1] == sym {
                match pool.get(args[0]) {
                    ExprData::Integer(n) => Some((Rational::from(n.0.clone()), Rational::from(0))),
                    ExprData::Rational(r) => Some((r.0.clone(), Rational::from(0))),
                    _ => None,
                }
            } else {
                None
            }
        }
        ExprData::Pow { base, exp } => {
            if base == sym {
                match pool.get(exp) {
                    ExprData::Integer(n) if n.0 == 1 => {
                        Some((Rational::from(1), Rational::from(0)))
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn offset_in_n(arg: ExprId, n: ExprId, pool: &ExprPool) -> Result<i64, RsolveError> {
    let (coef, c) = linear_in_sym(arg, n, pool).ok_or_else(|| {
        RsolveError::NotLinearRecurrence(
            "sequence index must be an affine integer shift of the recurrence variable".into(),
        )
    })?;
    if coef != 1 {
        return Err(RsolveError::NotLinearRecurrence(
            "recurrence variable must appear with coefficient 1 in each index".into(),
        ));
    }
    let num = c.numer();
    let den = c.denom();
    if num.clone() % den.clone() == 0 {
        let q = Integer::from(num / den);
        Ok(q.to_i64().unwrap_or(i64::MIN))
    } else {
        Err(RsolveError::NotLinearRecurrence(
            "index shift must be an integer".into(),
        ))
    }
}

fn contains_seq(expr: ExprId, seq_name: &str, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            if name == seq_name {
                return true;
            }
            args.iter().any(|&a| contains_seq(a, seq_name, pool))
        }
        ExprData::Add(xs) => xs.iter().any(|&a| contains_seq(a, seq_name, pool)),
        ExprData::Mul(xs) => xs.iter().any(|&a| contains_seq(a, seq_name, pool)),
        ExprData::Pow { base, exp } => {
            contains_seq(base, seq_name, pool) || contains_seq(exp, seq_name, pool)
        }
        _ => false,
    }
}

enum Peeled {
    Seq { coeff: Rational, offset: i64 },
    Other(ExprId),
}

fn peel_addend(
    term: ExprId,
    seq_name: &str,
    n: ExprId,
    pool: &ExprPool,
) -> Result<Peeled, RsolveError> {
    let factors = flatten_mul(term, pool);
    let mut rat = Rational::from(1);
    let mut seq_off: Option<i64> = None;
    let mut rest: Vec<ExprId> = Vec::new();

    for g in factors {
        match pool.get(g) {
            ExprData::Integer(nn) => {
                rat *= Rational::from(nn.0.clone());
            }
            ExprData::Rational(rr) => {
                rat *= rr.0.clone();
            }
            ExprData::Func { name, args } if name == seq_name => {
                if args.len() != 1 {
                    return Err(RsolveError::NotLinearRecurrence(
                        "sequence applications must have exactly one index argument".into(),
                    ));
                }
                if seq_off.is_some() {
                    return Err(RsolveError::NonlinearTerm);
                }
                seq_off = Some(offset_in_n(args[0], n, pool)?);
            }
            _ => rest.push(g),
        }
    }

    match (seq_off, rest.is_empty()) {
        (Some(o), true) => Ok(Peeled::Seq {
            coeff: rat,
            offset: o,
        }),
        (None, _) => {
            let rhs = if rest.is_empty() {
                rational_atom(pool, &rat)
            } else if rest.len() == 1 {
                if rat == 1 {
                    rest[0]
                } else {
                    simp(pool, pool.mul(vec![rational_atom(pool, &rat), rest[0]]))
                }
            } else {
                let mut v = rest;
                if rat != 1 {
                    v.insert(0, rational_atom(pool, &rat));
                }
                simp(pool, pool.mul(v))
            };
            Ok(Peeled::Other(rhs))
        }
        (Some(_), false) => Err(RsolveError::NonlinearTerm),
    }
}

/// Returns `a[k]` for `∑_{k=0}^d a[k] f(n-k) = rhs` and polynomial `rhs`.
fn extract_recurrence(
    equation: ExprId,
    seq_name: &str,
    n: ExprId,
    pool: &ExprPool,
) -> Result<(Vec<Rational>, RatUniPoly), RsolveError> {
    let zero = simp(pool, equation);
    let parts = flatten_add(zero, pool);
    let mut by_shift: BTreeMap<i64, Rational> = BTreeMap::new();
    let mut rhs_terms: Vec<ExprId> = Vec::new();

    for p in parts {
        match peel_addend(p, seq_name, n, pool)? {
            Peeled::Seq { coeff, offset } => {
                *by_shift.entry(offset).or_insert(Rational::from(0)) += coeff;
            }
            Peeled::Other(e) => rhs_terms.push(e),
        }
    }

    if by_shift.is_empty() {
        return Err(RsolveError::NotLinearRecurrence(
            "no sequence term in equation".into(),
        ));
    }

    let max_o = *by_shift.keys().max().unwrap();
    let mut shifts: BTreeMap<i64, Rational> = BTreeMap::new();
    for (&o, c) in &by_shift {
        let lag = max_o - o;
        *shifts.entry(lag).or_insert(Rational::from(0)) += c;
    }

    let d = *shifts.keys().max().unwrap() as usize;
    let mut a = vec![Rational::from(0); d + 1];
    for (&k, v) in &shifts {
        a[k as usize] = v.clone();
    }

    if a[0] == 0 {
        return Err(RsolveError::NotLinearRecurrence(
            "leading coefficient of f(n) vanishes after normalization".into(),
        ));
    }

    let rhs_expr = if rhs_terms.is_empty() {
        pool.integer(0_i32)
    } else {
        let s = simp(pool, pool.add(rhs_terms));
        simp(pool, pool.mul(vec![s, pool.integer(-1_i32)]))
    };

    if contains_seq(rhs_expr, seq_name, pool) {
        return Err(RsolveError::NotLinearRecurrence(
            "right-hand side still references the sequence".into(),
        ));
    }

    let rhs_poly = match UniPoly::from_symbolic_clear_denoms(rhs_expr, n, pool) {
        Ok(p) => {
            let cs: Vec<Rational> = p.coefficients().into_iter().map(Rational::from).collect();
            RatUniPoly { coeffs: cs }.trim()
        }
        Err(e) => {
            return Err(RsolveError::NonPolynomialRhs(e.to_string()));
        }
    };

    Ok((a, rhs_poly))
}

fn binom(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::from(0);
    }
    let mut acc = Integer::from(1);
    for i in 0..k {
        acc *= Integer::from(n - i);
        acc /= Integer::from(i + 1);
    }
    acc
}

/// `(n - m)^deg` as a polynomial in `n` (ascending coefficients).
fn shift_x_sub_m(deg: u32, m: i64) -> RatUniPoly {
    if deg == 0 {
        return RatUniPoly::one();
    }
    let mut coeffs = vec![Rational::from(0); (deg + 1) as usize];
    let mm = Rational::from(m);
    for k in 0..=deg {
        let mut term = Rational::from(binom(deg, k));
        if (deg - k) % 2 == 1 {
            term = -term;
        }
        for _ in 0..(deg - k) {
            term *= mm.clone();
        }
        coeffs[k as usize] = term;
    }
    RatUniPoly { coeffs }.trim()
}

/// `L[p] = p(n) - r·p(n-1)` on polynomials in `n`, with `minus_r = -r` matching
/// `f(n) + r f(n-1)` normalization... Here `r_forward` is the root of `x - r = 0` in
/// `f(n) = r f(n-1) + h` i.e. apply `p - r * sub(p,n-1)`.
fn poly_apply_order1_shift(r: &Rational, p: &RatUniPoly) -> RatUniPoly {
    let mut out = RatUniPoly::zero();
    for (deg, c) in p.coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let mut mon = vec![Rational::from(0); deg + 1];
        mon[deg] = c.clone();
        let n_poly = RatUniPoly { coeffs: mon }.trim();
        let shifted = shift_x_sub_m(deg as u32, 1);
        let sub = &RatUniPoly::constant(r.clone()) * &shifted;
        out = &out + &(&n_poly - &sub);
    }
    out.trim()
}

fn poly_apply_order2(a0: &Rational, a1: &Rational, a2: &Rational, p: &RatUniPoly) -> RatUniPoly {
    let mut out = RatUniPoly::zero();
    for (deg, coeff) in p.coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let mut mon = vec![Rational::from(0); deg + 1];
        mon[deg] = coeff.clone();
        let n_poly = RatUniPoly { coeffs: mon }.trim();
        let p1 = shift_x_sub_m(deg as u32, 1);
        let p2 = shift_x_sub_m(deg as u32, 2);
        let term = &(&(&RatUniPoly::constant(a0.clone()) * &n_poly)
            + &(&RatUniPoly::constant(a1.clone()) * &p1))
            + &(&RatUniPoly::constant(a2.clone()) * &p2);
        out = &out + &term;
    }
    out.trim()
}

fn mono_n(j: usize) -> RatUniPoly {
    let mut c = vec![Rational::from(0); j + 1];
    c[j] = Rational::from(1);
    RatUniPoly { coeffs: c }.trim()
}

fn solve_rational_linear_system(
    mut a: Vec<Vec<Rational>>,
    mut b: Vec<Rational>,
) -> Option<Vec<Rational>> {
    let n = b.len();
    debug_assert_eq!(a.len(), n);
    for col in 0..n {
        let mut pivot = None;
        for row in col..n {
            if !a[row][col].is_zero() {
                pivot = Some(row);
                break;
            }
        }
        let pr = pivot?;
        if pr != col {
            a.swap(col, pr);
            b.swap(col, pr);
        }
        let div = a[col][col].clone();
        if div.is_zero() {
            return None;
        }
        let inv = Rational::from(1) / div.clone();
        for j in col..n {
            a[col][j] *= inv.clone();
        }
        b[col] *= inv;
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row][col].clone();
            if f.is_zero() {
                continue;
            }
            for j in col..n {
                let sub = f.clone() * a[col][j].clone();
                a[row][j] -= sub;
            }
            let bcol = b[col].clone();
            b[row] -= f * bcol;
        }
    }
    Some(b)
}

fn undetermined_order1(r: &Rational, h: &RatUniPoly) -> Option<RatUniPoly> {
    let dh = h.degree().max(0) as usize;
    let start_deg = if rational_eq_one(r) { 1 } else { 0 };
    for bump in 0..24 {
        let hi_deg = (dh + bump + usize::from(rational_eq_one(r))).max(start_deg);
        if hi_deg > 40 {
            break;
        }
        let u = hi_deg.saturating_sub(start_deg) + 1;
        let mut mat = vec![vec![Rational::from(0); u]; u];
        let mut rhs = vec![Rational::from(0); u];
        for row in 0..u {
            for j in 0..u {
                let pow = start_deg + j;
                let basis = mono_n(pow);
                let applied = poly_apply_order1_shift(r, &basis);
                mat[row][j] = applied
                    .coeffs
                    .get(row)
                    .cloned()
                    .unwrap_or_else(|| Rational::from(0));
            }
            rhs[row] = h
                .coeffs
                .get(row)
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
        }
        if let Some(x) = solve_rational_linear_system(mat, rhs) {
            let mut coeffs = vec![Rational::from(0); hi_deg + 1];
            for (j, coeff) in x.into_iter().enumerate() {
                coeffs[start_deg + j] = coeff;
            }
            return Some(RatUniPoly { coeffs }.trim());
        }
    }
    None
}

fn undetermined_order2(
    a0: &Rational,
    a1: &Rational,
    a2: &Rational,
    h: &RatUniPoly,
) -> Option<RatUniPoly> {
    let dh = h.degree().max(0) as usize;
    for bump in 0..24 {
        let trial_deg = (dh + 4 + bump).min(42);
        let u = trial_deg + 1;
        let mut mat = vec![vec![Rational::from(0); u]; u];
        let mut rhs = vec![Rational::from(0); u];
        for row in 0..u {
            for j in 0..u {
                let basis = mono_n(j);
                let applied = poly_apply_order2(a0, a1, a2, &basis);
                mat[row][j] = applied
                    .coeffs
                    .get(row)
                    .cloned()
                    .unwrap_or_else(|| Rational::from(0));
            }
            rhs[row] = h
                .coeffs
                .get(row)
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
        }
        if let Some(x) = solve_rational_linear_system(mat, rhs) {
            return Some(RatUniPoly { coeffs: x }.trim());
        }
    }
    None
}

fn rat_poly_to_expr(pool: &ExprPool, n_sym: ExprId, p: &RatUniPoly) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for (deg, coeff) in p.coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let coeff_q = coeff.clone();
        let numer = coeff_q.numer();
        let denom = coeff_q.denom();
        let coeff_expr = if *denom == 1 {
            pool.integer(numer.clone())
        } else {
            pool.rational(numer.clone(), denom.clone())
        };
        let pow_id = if deg == 0 {
            coeff_expr
        } else if deg == 1 {
            pool.mul(vec![coeff_expr, n_sym])
        } else {
            pool.mul(vec![coeff_expr, pool.pow(n_sym, pool.integer(deg as i64))])
        };
        terms.push(pow_id);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

fn sqrt_disc_expr(pool: &ExprPool, disc: &Rational) -> ExprId {
    let num = disc.numer().clone();
    let den = disc.denom().clone();
    let prod = num * den.clone();
    let inner = pool.integer(prod);
    let sqrt_e = pool.func("sqrt", vec![inner]);
    let den_e = pool.integer(den);
    expr_div(pool, sqrt_e, den_e)
}

/// `a[0] r^d + … + a[d]` in ascending powers of `r`.
fn char_poly_asc(a: &[Rational]) -> RatUniPoly {
    let d = a.len() - 1;
    let mut v = vec![Rational::from(0); d + 1];
    for i in 0..=d {
        v[i] = a[d - i].clone();
    }
    RatUniPoly { coeffs: v }.trim()
}

fn horner_rat(p: &RatUniPoly, x: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.coeffs.iter().rev() {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

fn divisors_int(mut n: Integer) -> Vec<Integer> {
    if n < 0 {
        n = -n;
    }
    if n == 0 {
        return vec![Integer::from(1)];
    }
    let mut out = vec![Integer::from(1)];
    let mut i = Integer::from(2);
    while i.clone() * i.clone() <= n {
        if n.clone() % i.clone() == 0 {
            let mut pws = vec![Integer::from(1)];
            let mut nn = n.clone();
            while nn.clone() % i.clone() == 0 {
                let last = pws.last().unwrap().clone();
                pws.push(last * i.clone());
                nn /= i.clone();
            }
            n = nn;
            let old = out.clone();
            out.clear();
            for base in old {
                for pw in &pws {
                    out.push(base.clone() * pw);
                }
            }
        }
        i += 1;
    }
    if n > 1 {
        let old = out.clone();
        out.clear();
        for base in old {
            out.push(base.clone());
            out.push(base * n.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn peel_rational_root(p: &RatUniPoly) -> Option<Rational> {
    if p.is_zero() {
        return None;
    }
    let mut z: Vec<Integer> = Vec::new();
    let mut lcm_den = Integer::from(1);
    for c in &p.coeffs {
        lcm_den = lcm_den.lcm(&c.denom().clone());
    }
    for c in &p.coeffs {
        let d = c.denom().clone();
        let scale = lcm_den.clone() / d;
        z.push(scale * c.numer().clone());
    }
    let lc = z.last().cloned().unwrap_or_else(|| Integer::from(0));
    let c0 = z.first().cloned().unwrap_or_else(|| Integer::from(0));
    if lc.is_zero() {
        return Some(Rational::from(0));
    }
    let mut try_vals: Vec<Rational> = Vec::new();
    for pd in divisors_int(lc.clone()) {
        for qd in divisors_int(c0.clone()) {
            try_vals.push(Rational::from((pd.clone(), qd.clone())));
            try_vals.push(-Rational::from((pd.clone(), qd)));
        }
    }
    try_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    try_vals.dedup();
    try_vals
        .into_iter()
        .find(|r| !p.coeffs.is_empty() && horner_rat(p, r).is_zero())
}

fn div_linear_factor(p: RatUniPoly, root: &Rational) -> RatUniPoly {
    let r = root.clone();
    let lin = RatUniPoly {
        coeffs: vec![-r, Rational::from(1)],
    }
    .trim();
    let (q, rem) = RatUniPoly::div_rem(&p, &lin);
    debug_assert!(rem.is_zero());
    q
}

fn factor_char_polynomial(mut p: RatUniPoly) -> Result<Vec<(Rational, usize)>, RsolveError> {
    let mut roots: Vec<(Rational, usize)> = Vec::new();
    let mut guard = 0usize;
    while p.degree() > 0 && guard < 64 {
        guard += 1;
        let Some(r0) = peel_rational_root(&p) else {
            break;
        };
        let mut m = 0usize;
        while p.degree() > 0 && horner_rat(&p, &r0).is_zero() {
            p = div_linear_factor(p, &r0);
            m += 1;
        }
        roots.push((r0, m));
    }
    match p.degree() {
        -1 | 0 => Ok(roots),
        1 => {
            let c0 = p.coeffs[0].clone();
            let c1 = p.coeffs[1].clone();
            if c1 == 0 {
                return Err(RsolveError::Unsupported("degenerate characteristic".into()));
            }
            roots.push((-c0 / c1, 1));
            Ok(roots)
        }
        2 => {
            let c0 = p.coeffs[0].clone();
            let c1 = p.coeffs[1].clone();
            let c2 = p.coeffs[2].clone();
            if c2 == 0 {
                return Err(RsolveError::Unsupported(
                    "characteristic degree mismatch".into(),
                ));
            }
            let disc = c1.clone() * c1.clone() - Rational::from(4) * c2.clone() * c0.clone();
            if disc == 0 {
                let r = -c1 / (Rational::from(2) * c2.clone());
                roots.push((r, 2));
            } else if disc > 0 {
                let disc_numer = disc.numer().clone();
                let disc_denom = disc.denom().clone();
                let (sn, rem_n) = disc_numer.sqrt_rem(Integer::new());
                let (sd, rem_d) = disc_denom.sqrt_rem(Integer::new());
                if rem_n != 0 || rem_d != 0 {
                    return Err(RsolveError::Unsupported(
                        "order-3+ with irreducible quadratic characteristic tail".into(),
                    ));
                }
                let sqrt_d = Rational::from((sn, sd));
                let r1 = (-c1.clone() + sqrt_d.clone()) / (Rational::from(2) * c2.clone());
                let r2 = (-c1 - sqrt_d) / (Rational::from(2) * c2.clone());
                roots.push((r1, 1));
                roots.push((r2, 1));
            } else {
                return Err(RsolveError::Unsupported(
                    "complex characteristic roots".into(),
                ));
            }
            Ok(roots)
        }
        d => Err(RsolveError::Unsupported(format!(
            "characteristic leftover degree {d}"
        ))),
    }
}

fn hom_solution_from_roots(
    pool: &ExprPool,
    n_sym: ExprId,
    root_facts: &[(Rational, usize)],
    c_syms: &[ExprId],
) -> Result<ExprId, RsolveError> {
    let mut terms: Vec<ExprId> = Vec::new();
    let mut idx = 0;
    for (r, mult) in root_facts {
        let re = rational_atom(pool, r);
        for p in 0..*mult {
            if idx >= c_syms.len() {
                return Err(RsolveError::Unsupported(
                    "internal: not enough constant symbols".into(),
                ));
            }
            let basis = if p == 0 {
                simp(pool, pool.pow(re, n_sym))
            } else {
                let np = pool.pow(n_sym, pool.integer(p as i64));
                simp(pool, pool.mul(vec![np, pool.pow(re, n_sym)]))
            };
            terms.push(simp(pool, pool.mul(vec![c_syms[idx], basis])));
            idx += 1;
        }
    }
    if idx != c_syms.len() {
        return Err(RsolveError::Unsupported(
            "internal: constant count mismatch".into(),
        ));
    }
    match terms.len() {
        0 => Ok(pool.integer(0_i32)),
        1 => Ok(terms[0]),
        _ => Ok(simp(pool, pool.add(terms))),
    }
}

fn order2_r_exprs(pool: &ExprPool, a_rec: &[Rational]) -> Result<(ExprId, ExprId), RsolveError> {
    let p = char_poly_asc(a_rec);
    if p.degree() != 2 {
        return Err(RsolveError::Unsupported(
            "expected order-2 characteristic".into(),
        ));
    }
    let p0 = p.coeffs[0].clone();
    let p1 = p.coeffs[1].clone();
    let p2 = p.coeffs[2].clone();
    if p2 == 0 {
        return Err(RsolveError::Unsupported("degenerate characteristic".into()));
    }
    let b = p1 / p2.clone();
    let c = p0 / p2.clone();
    let disc = b.clone() * b.clone() - Rational::from(4) * c.clone();
    if disc < 0 {
        return Err(RsolveError::Unsupported("complex roots".into()));
    }
    let sqrt_e = sqrt_disc_expr(pool, &disc);
    let neg_b = rational_atom(pool, &(-b.clone()));
    let half = rational_atom(pool, &Rational::from((1, 2)));
    let inner1 = simp(pool, pool.add(vec![neg_b, sqrt_e]));
    let r1 = simp(pool, pool.mul(vec![half, inner1]));
    let inner2 = simp(
        pool,
        pool.add(vec![neg_b, pool.mul(vec![sqrt_e, pool.integer(-1_i32)])]),
    );
    let r2 = simp(pool, pool.mul(vec![half, inner2]));
    Ok((r1, r2))
}

fn fresh_constants(pool: &ExprPool, k: usize) -> Vec<ExprId> {
    (0..k)
        .map(|i: usize| pool.symbol(format!("C{}", i), crate::kernel::Domain::Real))
        .collect()
}

fn subs_n_int(pool: &ExprPool, expr: ExprId, n_sym: ExprId, ni: i64) -> ExprId {
    let mut m = HashMap::new();
    m.insert(n_sym, pool.integer(ni));
    simp(pool, subs(expr, &m, pool))
}

#[allow(clippy::too_many_arguments)]
fn apply_init(
    pool: &ExprPool,
    general: ExprId,
    n_sym: ExprId,
    c_syms: &[ExprId],
    initials: &BTreeMap<i64, ExprId>,
    d: usize,
    a: &[Rational],
    particular: ExprId,
) -> Result<ExprId, RsolveError> {
    if initials.len() != d {
        return Err(RsolveError::InitialMismatch(format!(
            "need {d} initial values for order {d}, got {}",
            initials.len()
        )));
    }

    if d == 1 {
        let (&n0, v0) = initials.first_key_value().unwrap();
        let r = (-a[1].clone()) / a[0].clone();
        let r_e = rational_atom(pool, &r);
        let p0 = subs_n_int(pool, particular, n_sym, n0);
        let rpow = simp(pool, pool.pow(r_e, pool.integer(n0)));
        let rhs = simp(
            pool,
            pool.add(vec![*v0, pool.mul(vec![p0, pool.integer(-1_i32)])]),
        );
        let c0v = expr_div(pool, rhs, rpow);
        let mut m = HashMap::new();
        m.insert(c_syms[0], c0v);
        return Ok(simp(pool, subs(general, &m, pool)));
    }

    if d == 2 {
        let keys: Vec<i64> = initials.keys().copied().collect();
        if keys.len() != 2 {
            return Err(RsolveError::InitialMismatch("need two integers".into()));
        }
        let (n0, n1) = (keys[0], keys[1]);
        let (r1_e, r2_e) = order2_r_exprs(pool, a)?;
        let v0 = *initials.get(&n0).unwrap();
        let v1 = *initials.get(&n1).unwrap();
        let p0 = subs_n_int(pool, particular, n_sym, n0);
        let p1 = subs_n_int(pool, particular, n_sym, n1);
        let v0p = simp(
            pool,
            pool.add(vec![v0, pool.mul(vec![p0, pool.integer(-1_i32)])]),
        );
        let v1p = simp(
            pool,
            pool.add(vec![v1, pool.mul(vec![p1, pool.integer(-1_i32)])]),
        );
        let a00 = simp(pool, pool.pow(r1_e, pool.integer(n0)));
        let b00 = simp(pool, pool.pow(r2_e, pool.integer(n0)));
        let a10 = simp(pool, pool.pow(r1_e, pool.integer(n1)));
        let b10 = simp(pool, pool.pow(r2_e, pool.integer(n1)));
        let det = simp(
            pool,
            pool.add(vec![
                pool.mul(vec![a00, b10]),
                pool.mul(vec![a10, b00, pool.integer(-1_i32)]),
            ]),
        );
        let num_c0 = simp(
            pool,
            pool.add(vec![
                pool.mul(vec![v0p, b10]),
                pool.mul(vec![v1p, b00, pool.integer(-1_i32)]),
            ]),
        );
        let num_c1 = simp(
            pool,
            pool.add(vec![
                pool.mul(vec![a00, v1p]),
                pool.mul(vec![a10, v0p, pool.integer(-1_i32)]),
            ]),
        );
        let c0v = expr_div(pool, num_c0, det);
        let c1v = expr_div(pool, num_c1, det);
        let mut m = HashMap::new();
        m.insert(c_syms[0], c0v);
        m.insert(c_syms[1], c1v);
        return Ok(simp(pool, subs(general, &m, pool)));
    }

    Err(RsolveError::InitialMismatch(
        "initial values for order > 2 not implemented".into(),
    ))
}

/// Solve a linear recurrence coded as `equation == 0` in `n`:
///
/// Each sequence term is `pool.func(seq_name, [n + integer])`.  The general
/// solution introduces symbols `C0`, `C1`, …; pass `initials` to eliminate them.
pub fn rsolve(
    pool: &ExprPool,
    equation: ExprId,
    n: ExprId,
    seq_name: &str,
    initials: Option<&BTreeMap<i64, ExprId>>,
) -> Result<ExprId, RsolveError> {
    let (a, rhs_p) = extract_recurrence(equation, seq_name, n, pool)?;
    let d = a.len() - 1;

    let a0_lead = a[0].clone();
    let hom_norm: Vec<Rational> = a.iter().map(|x| x.clone() / a0_lead.clone()).collect();
    let rhs_norm = {
        let inv = Rational::from(1) / a0_lead.clone();
        RatUniPoly {
            coeffs: rhs_p
                .coeffs
                .iter()
                .map(|c| c.clone() * inv.clone())
                .collect(),
        }
        .trim()
    };

    let particular_p = if rhs_norm.is_zero() {
        RatUniPoly::zero()
    } else if d == 1 {
        let r = -hom_norm[1].clone();
        undetermined_order1(&r, &rhs_norm).ok_or_else(|| {
            RsolveError::Unsupported("particular solution (order 1) failed".into())
        })?
    } else if d == 2 {
        undetermined_order2(&hom_norm[0], &hom_norm[1], &hom_norm[2], &rhs_norm).ok_or_else(
            || RsolveError::Unsupported("particular solution (order 2) failed".into()),
        )?
    } else {
        if !rhs_norm.is_zero() {
            return Err(RsolveError::Unsupported(
                "non-homogeneous order > 2 is not implemented".into(),
            ));
        }
        RatUniPoly::zero()
    };

    let particular_e = if particular_p.is_zero() {
        pool.integer(0_i32)
    } else {
        rat_poly_to_expr(pool, n, &particular_p)
    };

    let (hom_e, c_syms): (ExprId, Vec<ExprId>) = match d {
        1 => {
            let r = -hom_norm[1].clone();
            let re = rational_atom(pool, &r);
            let c0 = pool.symbol("C0", crate::kernel::Domain::Real);
            let h = simp(pool, pool.mul(vec![c0, pool.pow(re, n)]));
            (h, vec![c0])
        }
        2 => {
            let c0 = pool.symbol("C0", crate::kernel::Domain::Real);
            let c1 = pool.symbol("C1", crate::kernel::Domain::Real);
            let (r1, r2) = order2_r_exprs(pool, &a)?;
            let h = simp(
                pool,
                pool.add(vec![
                    simp(pool, pool.mul(vec![c0, pool.pow(r1, n)])),
                    simp(pool, pool.mul(vec![c1, pool.pow(r2, n)])),
                ]),
            );
            (h, vec![c0, c1])
        }
        _ => {
            let facts = factor_char_polynomial(char_poly_asc(&a))?;
            let nconst: usize = facts.iter().map(|(_, m)| *m).sum();
            let cs = fresh_constants(pool, nconst);
            let h = hom_solution_from_roots(pool, n, &facts, &cs)?;
            (h, cs)
        }
    };

    let general = simp(pool, pool.add(vec![hom_e, particular_e]));

    if let Some(init) = initials {
        apply_init(pool, general, n, &c_syms, init, d, &a, particular_e)
    } else {
        Ok(general)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;
    use rug::Rational;
    use std::collections::HashMap;

    fn has_sym(expr: ExprId, name: &str, pool: &ExprPool) -> bool {
        match pool.get(expr) {
            ExprData::Symbol { name: n, .. } => n == name,
            ExprData::Add(xs) => xs.iter().any(|&x| has_sym(x, name, pool)),
            ExprData::Mul(xs) => xs.iter().any(|&x| has_sym(x, name, pool)),
            ExprData::Pow { base, exp } => has_sym(base, name, pool) || has_sym(exp, name, pool),
            ExprData::Func { args, .. } => args.iter().any(|&a| has_sym(a, name, pool)),
            _ => false,
        }
    }

    #[test]
    fn arithmetic_progression_general() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let f = |args: Vec<ExprId>| pool.func("f", args);
        let eq = simp(
            &pool,
            pool.add(vec![
                f(vec![n]),
                pool.mul(vec![
                    f(vec![pool.add(vec![n, pool.integer(-1_i32)])]),
                    pool.integer(-1_i32),
                ]),
                pool.integer(-1_i32),
            ]),
        );
        let sol = rsolve(&pool, eq, n, "f", None).expect("rsolve");
        assert!(has_sym(sol, "C0", &pool));
    }

    #[test]
    fn fibonacci_numeric_with_init() {
        use crate::sum::recurrence::solve_linear_recurrence_homogeneous;
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let f = |args: Vec<ExprId>| pool.func("f", args);
        let eq = simp(
            &pool,
            pool.add(vec![
                f(vec![n]),
                pool.mul(vec![
                    f(vec![pool.add(vec![n, pool.integer(-1_i32)])]),
                    pool.integer(-1_i32),
                ]),
                pool.mul(vec![
                    f(vec![pool.add(vec![n, pool.integer(-2_i32)])]),
                    pool.integer(-1_i32),
                ]),
            ]),
        );
        let mut init = BTreeMap::new();
        init.insert(0, pool.integer(0));
        init.insert(1, pool.integer(1));
        let sol = rsolve(&pool, eq, n, "f", Some(&init)).expect("rsolve");

        let ref_sol = solve_linear_recurrence_homogeneous(
            &pool,
            n,
            &[Rational::from(-1), Rational::from(-1), Rational::from(1)],
            &[pool.integer(0), pool.integer(1)],
        )
        .expect("ref");

        for ni in 0..=12 {
            let mut env = HashMap::new();
            env.insert(n, ni as f64);
            let v = eval_interp(sol, &env, &pool).expect("eval");
            let vr = eval_interp(ref_sol.closed_form, &env, &pool).expect("eref");
            assert!((v - vr).abs() < 1e-4, "n={ni} rsolve={v} ref={vr}",);
        }
    }
}
