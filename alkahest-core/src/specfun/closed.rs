//! Exact closed forms: Bernoulli numbers and polynomials, `ζ` at integers,
//! Eulerian numerators of `Li_{−n}`, and the elementary forms of the
//! half-integer-order Bessel functions.
//!
//! Everything here is rational arithmetic; nothing passes through `f64`.

use rug::{Integer, Rational};

use crate::kernel::{ExprId, ExprPool};
use crate::orthopoly::{mk_add, mk_mul, rat_node};

use super::SpecialFunctionError;

// ---------------------------------------------------------------------------
// Bernoulli numbers and polynomials
// ---------------------------------------------------------------------------

/// `B_0, …, B_n` with `B_1 = −1/2` (DLMF 24.2.1), from
/// `Σ_{k=0}^{m} C(m+1, k) B_k = 0`.
///
/// `O(n²)` rational operations; polls the active budget once per row.
pub(crate) fn bernoulli(n: usize) -> Result<Vec<Rational>, SpecialFunctionError> {
    let mut b = Vec::with_capacity(n + 1);
    b.push(Rational::from(1));
    for m in 1..=n {
        crate::budget::check()?;
        // Odd Bernoulli numbers past B_1 vanish (DLMF 24.2.4).
        if m >= 3 && m % 2 == 1 {
            b.push(Rational::new());
            continue;
        }
        let mut binom = Integer::from(1); // C(m+1, k)
        let mut s = Rational::new();
        for (k, bk) in b.iter().enumerate() {
            if *bk != 0 {
                s += Rational::from(&binom) * bk;
            }
            binom *= (m + 1 - k) as u64;
            binom /= (k + 1) as u64;
        }
        b.push(-s / Rational::from(m as u64 + 1));
    }
    Ok(b)
}

/// The coefficients of the Bernoulli polynomial
/// `B_m(a) = Σ_k C(m, k) B_k a^{m−k}` (DLMF 24.2.5), lowest power of `a`
/// first.
pub(crate) fn bernoulli_poly(m: usize) -> Result<Vec<Rational>, SpecialFunctionError> {
    let b = bernoulli(m)?;
    let mut out = vec![Rational::new(); m + 1];
    let mut binom = Integer::from(1); // C(m, k)
    for (k, bk) in b.iter().enumerate() {
        out[m - k] = Rational::from(&binom) * bk;
        binom *= (m - k) as u64;
        binom /= (k + 1) as u64;
    }
    Ok(out)
}

/// `c` with `ζ(2m) = c·π^{2m}` (DLMF 25.6.2), `m ≥ 1`.
pub(crate) fn zeta_even_coefficient(m: usize) -> Result<Rational, SpecialFunctionError> {
    let b = bernoulli(2 * m)?;
    let mut fact = Integer::from(1);
    for j in 2..=(2 * m as u64) {
        fact *= j;
    }
    let two_pow = Integer::from(1) << (2 * m as u32);
    let c = Rational::from((two_pow, Integer::from(2) * fact)) * &b[2 * m];
    Ok(if m % 2 == 1 { c } else { -c })
}

/// `ζ(−n) = (−1)ⁿ B_{n+1}/(n+1)` (DLMF 25.6.3–4), `n ≥ 0`; `ζ(0) = −1/2`.
pub(crate) fn zeta_nonpositive(n: usize) -> Result<Rational, SpecialFunctionError> {
    let b = bernoulli(n + 1)?;
    let v = b[n + 1].clone() / Rational::from(n as u64 + 1);
    Ok(if n % 2 == 1 { -v } else { v })
}

/// `A(n, k)`, `k = 0..n−1`: the Eulerian numbers, so that
/// `Li_{−n}(z) = z·Σ_k A(n,k) z^k / (1 − z)^{n+1}` for `n ≥ 1`, which is
/// `(z d/dz)ⁿ z/(1 − z)` (Lewin, *Polylogarithms*, §7.1).
pub(crate) fn eulerian_row(n: usize) -> Vec<Integer> {
    let mut row = vec![Integer::from(1)];
    for m in 2..=n {
        // A(m, k) = (k+1) A(m−1, k) + (m−k) A(m−1, k−1)
        let mut next = vec![Integer::new(); m];
        for k in 0..m {
            let mut v = Integer::new();
            if k < row.len() {
                v += Integer::from(&row[k] * (k as u64 + 1));
            }
            if k >= 1 {
                v += Integer::from(&row[k - 1] * (m - k) as u64);
            }
            next[k] = v;
        }
        row = next;
    }
    row
}

// ---------------------------------------------------------------------------
// Half-integer Bessel functions
// ---------------------------------------------------------------------------

/// A polynomial in `t = 1/x` with rational coefficients, lowest power first.
type TPoly = Vec<Rational>;

fn tp_add(a: &TPoly, b: &TPoly, sign_b: i32) -> TPoly {
    let n = a.len().max(b.len());
    (0..n)
        .map(|k| {
            let x = a.get(k).cloned().unwrap_or_default();
            let y = b.get(k).cloned().unwrap_or_default();
            if sign_b < 0 {
                x - y
            } else {
                x + y
            }
        })
        .collect()
}

/// `c·t·a`.
fn tp_mul_t(a: &TPoly, c: i64) -> TPoly {
    let mut out = Vec::with_capacity(a.len() + 1);
    out.push(Rational::new());
    out.extend(a.iter().map(|v| Rational::from(v * c)));
    out
}

/// `(sin-part, cos-part)` (or `(sinh, cosh)` for the modified functions).
type Pair = (TPoly, TPoly);

fn pair_lin(a: &Pair, ca: i64, b: &Pair, sign_b: i32) -> Pair {
    // ca·t·a ± b
    (
        tp_add(&tp_mul_t(&a.0, ca), &b.0, sign_b),
        tp_add(&tp_mul_t(&a.1, ca), &b.1, sign_b),
    )
}

/// Which three-term recurrence links neighbouring orders.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Spherical {
    /// `j_n`, `y_n`: `f_{n+1} = (2n+1) t f_n − f_{n−1}` (DLMF 10.51.1).
    Ordinary,
    /// `i_n^{(1)}`: `f_{n+1} = f_{n−1} − (2n+1) t f_n` (DLMF 10.51.4).
    Modified,
}

/// The `(sin, cos)` coefficient polynomials of `f_m` for every integer `m`,
/// from the seeds `f_0`, `f_{−1}`.
pub(crate) fn spherical_pair(
    rec: Spherical,
    f0: Pair,
    fm1: Pair,
    m: i64,
) -> Result<Pair, SpecialFunctionError> {
    if m == 0 {
        return Ok(f0);
    }
    if m == -1 {
        return Ok(fm1);
    }
    if m > 0 {
        let (mut prev, mut cur) = (fm1, f0);
        for n in 0..m {
            crate::budget::check()?;
            let c = 2 * n + 1;
            let next = match rec {
                Spherical::Ordinary => pair_lin(&cur, c, &prev, -1),
                // f_{n−1} − c t f_n
                Spherical::Modified => pair_lin(&cur, -c, &prev, 1),
            };
            prev = cur;
            cur = next;
        }
        Ok(cur)
    } else {
        // Downward from (f_0, f_{−1}): f_{n−1} from f_n and f_{n+1}.
        let (mut above, mut cur) = (f0, fm1); // f_{n+1}, f_n with n = −1
        let mut n = -1_i64;
        while n > m {
            crate::budget::check()?;
            let c = 2 * n + 1;
            let next = match rec {
                // f_{n−1} = c t f_n − f_{n+1}
                Spherical::Ordinary => pair_lin(&cur, c, &above, -1),
                // f_{n−1} = f_{n+1} + c t f_n
                Spherical::Modified => pair_lin(&cur, c, &above, 1),
            };
            above = cur;
            cur = next;
            n -= 1;
        }
        Ok(cur)
    }
}

/// `p_m(t)` with `K_{m+1/2}(x) = √(π/(2x)) e^{−x} p_m(1/x)`, `m ≥ 0`
/// (DLMF 10.49.12); `K_{−ν} = K_ν` covers `m < 0`.
pub(crate) fn k_half_poly(m: i64) -> Result<TPoly, SpecialFunctionError> {
    let m = if m < 0 { -m - 1 } else { m };
    // p_{n+1} = p_{n−1} + (2n+1) t p_n, p_{−1} = p_0 = 1.
    let (mut prev, mut cur): (TPoly, TPoly) = (vec![Rational::from(1)], vec![Rational::from(1)]);
    for n in 0..m {
        crate::budget::check()?;
        let next = tp_add(&tp_mul_t(&cur, 2 * n + 1), &prev, 1);
        prev = cur;
        cur = next;
    }
    Ok(cur)
}

/// `c·t`.
fn ct(c: i64) -> TPoly {
    vec![Rational::new(), Rational::from(c)]
}

/// Seeds `(f_0, f_{−1})` as `(sin, cos)` pairs.
pub(crate) fn seeds_j() -> (Pair, Pair) {
    ((ct(1), vec![]), (vec![], ct(1)))
}
pub(crate) fn seeds_y() -> (Pair, Pair) {
    // y_0 = −cos x / x, y_{−1} = sin x / x
    ((vec![], ct(-1)), (ct(1), vec![]))
}
/// `i_0^{(1)} = sinh x / x`, `i_{−1}^{(1)} = cosh x / x`.
pub(crate) fn seeds_i() -> (Pair, Pair) {
    seeds_j()
}

/// `Σ_k c_k x^{−k + offset}`, skipping zero coefficients.
pub(crate) fn laurent(pool: &ExprPool, c: &TPoly, x: ExprId, offset: &Rational) -> ExprId {
    let mut terms = Vec::new();
    for (k, ck) in c.iter().enumerate() {
        if *ck == 0 {
            continue;
        }
        let e = Rational::from(offset - Rational::from(k as u64));
        let xp = if e == 0 {
            pool.integer(1)
        } else if e == 1 {
            x
        } else {
            pool.pow(x, rat_node(pool, &e))
        };
        terms.push(mk_mul(pool, vec![rat_node(pool, ck), xp]));
    }
    mk_add(pool, terms)
}

/// `a(x)·S(x) + b(x)·C(x)` where `a`, `b` are the Laurent forms of a pair.
pub(crate) fn pair_expr(
    pool: &ExprPool,
    p: &Pair,
    x: ExprId,
    offset: &Rational,
    sin_name: &str,
    cos_name: &str,
) -> ExprId {
    let mut terms = Vec::new();
    if p.0.iter().any(|c| *c != 0) {
        terms.push(mk_mul(
            pool,
            vec![laurent(pool, &p.0, x, offset), pool.func(sin_name, vec![x])],
        ));
    }
    if p.1.iter().any(|c| *c != 0) {
        terms.push(mk_mul(
            pool,
            vec![laurent(pool, &p.1, x, offset), pool.func(cos_name, vec![x])],
        ));
    }
    mk_add(pool, terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_first_values() {
        let b = bernoulli(12).unwrap();
        let q = |n: i64, d: i64| Rational::from((n, d));
        assert_eq!(b[0], q(1, 1));
        assert_eq!(b[1], q(-1, 2));
        assert_eq!(b[2], q(1, 6));
        assert_eq!(b[3], q(0, 1));
        assert_eq!(b[4], q(-1, 30));
        assert_eq!(b[6], q(1, 42));
        assert_eq!(b[8], q(-1, 30));
        assert_eq!(b[10], q(5, 66));
        assert_eq!(b[12], q(-691, 2730));
    }

    #[test]
    fn zeta_values() {
        assert_eq!(zeta_even_coefficient(1).unwrap(), Rational::from((1, 6)));
        assert_eq!(zeta_even_coefficient(2).unwrap(), Rational::from((1, 90)));
        assert_eq!(zeta_even_coefficient(3).unwrap(), Rational::from((1, 945)));
        assert_eq!(zeta_nonpositive(0).unwrap(), Rational::from((-1, 2)));
        assert_eq!(zeta_nonpositive(1).unwrap(), Rational::from((-1, 12)));
        assert_eq!(zeta_nonpositive(2).unwrap(), Rational::new());
        assert_eq!(zeta_nonpositive(3).unwrap(), Rational::from((1, 120)));
    }

    #[test]
    fn eulerian_rows() {
        let r = |n| {
            eulerian_row(n)
                .into_iter()
                .map(|v| v.to_i64().unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(r(1), vec![1]);
        assert_eq!(r(2), vec![1, 1]);
        assert_eq!(r(3), vec![1, 4, 1]);
        assert_eq!(r(4), vec![1, 11, 11, 1]);
    }

    #[test]
    fn spherical_j1_and_y1() {
        let (f0, fm1) = seeds_j();
        let j1 = spherical_pair(Spherical::Ordinary, f0, fm1, 1).unwrap();
        // j_1 = sin/x² − cos/x
        assert_eq!(
            j1.0,
            vec![Rational::new(), Rational::new(), Rational::from(1)]
        );
        assert_eq!(j1.1, vec![Rational::new(), Rational::from(-1)]);
        let (f0, fm1) = seeds_y();
        let y1 = spherical_pair(Spherical::Ordinary, f0, fm1, 1).unwrap();
        // y_1 = −cos/x² − sin/x
        assert_eq!(
            y1.1,
            vec![Rational::new(), Rational::new(), Rational::from(-1)]
        );
        assert_eq!(y1.0, vec![Rational::new(), Rational::from(-1)]);
        // j_{−2} = y_1 · (−1)^{…}: j_{−n−1} = (−1)^{n+1} y_n, so j_{−2} = y_1.
        let (f0, fm1) = seeds_j();
        let jm2 = spherical_pair(Spherical::Ordinary, f0, fm1, -2).unwrap();
        let strip = |p: &TPoly| {
            let mut p = p.clone();
            while p.last().is_some_and(|v| *v == 0) {
                p.pop();
            }
            p
        };
        assert_eq!(strip(&jm2.0), strip(&y1.0));
        assert_eq!(strip(&jm2.1), strip(&y1.1));
    }

    #[test]
    fn k_half_polys() {
        assert_eq!(k_half_poly(0).unwrap(), vec![Rational::from(1)]);
        assert_eq!(
            k_half_poly(1).unwrap(),
            vec![Rational::from(1), Rational::from(1)]
        );
        // K_{5/2}: 1 + 3/x + 3/x²
        assert_eq!(
            k_half_poly(2).unwrap(),
            vec![Rational::from(1), Rational::from(3), Rational::from(3)]
        );
        assert_eq!(k_half_poly(-3).unwrap(), k_half_poly(2).unwrap());
    }
}
