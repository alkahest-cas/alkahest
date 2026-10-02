//! Fraction-free determinants and minors of polynomial matrices.
//!
//! `symbolic_inverse` needs `det(A)` and all `n²` first minors in expanded
//! form. Through the memoised cofactor expansion (`DetMemo`) that is
//! `O(n·2ⁿ)` nested sub-determinants which the simplifier then has to expand
//! term by term. When every entry is a polynomial with rational coefficients
//! in commutative symbols, [`PolyMatrix::det_and_minors_expanded`] instead runs
//! one fraction-free Gauss–Jordan elimination of `[A | I]` over
//! `ℤ[x₁,…,x_k]` with FLINT's `fmpz_mpoly` — `O(n³)` polynomial operations,
//! whose divisions are exact (`fmpz_mpoly_divides`) — and reads the
//! determinant off the diagonal and the adjugate off the right block.
//!
//! Rational coefficients are cleared per row: row `i` is multiplied by the lcm
//! `dᵢ` of its entries' denominators, and a determinant of the scaled rows is
//! divided by the product of the `dᵢ` of the rows it keeps.
//!
//! Each polynomial is returned as a sum of monomials passed through
//! `simplify_expanded`, which gives the same canonical expression, `ExprId`
//! for `ExprId`, as `simplify_expanded` of the cofactor expansion — the
//! differential tests here and in `linear_algebra::tests` compare the two
//! routes. The one place they could differ is a power the expanding
//! simplifier declines to distribute (`ExpandPow`'s size bound); matrices
//! where the cofactor route could meet one are declined up front
//! (`cofactor_route_expands_fully`).
//!
//! Anything that is not such a polynomial — a function application, a float,
//! a negative or symbolic exponent, a non-commutative symbol — makes
//! [`PolyMatrix::from_matrix`] decline, and the caller keeps the cofactor
//! route. `Matrix::det` itself is unchanged: it returns the unexpanded
//! cofactor expression, whose shape a polynomial elimination cannot produce.

use super::Matrix;
use crate::flint::mpoly::{FlintMPoly, FlintMPolyCtx};
use crate::kernel::{ExprData, ExprId, ExprPool, IdMap};
use crate::poly::multipoly::MultiPoly;
use crate::simplify::engine::simplify_expanded;
use rug::{Integer, Rational};
use std::sync::Arc;

/// A square matrix of polynomials over `ℤ[vars]`, each row scaled by the lcm
/// of its denominators.
pub(crate) struct PolyMatrix {
    n: usize,
    vars: Vec<ExprId>,
    ctx: Arc<FlintMPolyCtx>,
    /// Row-major scaled entries.
    entries: Vec<FlintMPoly>,
    /// Per row, the positive integer the row was multiplied by.
    row_scale: Vec<Integer>,
}

/// A polynomial with rational coefficients as `num / den` (`den > 0`).
struct QPoly {
    num: MultiPoly,
    den: Integer,
}

/// Why an entry is not taken by the polynomial route. Only used to decline.
struct NotPolynomial;

impl PolyMatrix {
    /// Convert `m` (square) entry by entry, or `None` if some entry is not a
    /// polynomial with rational coefficients in commutative symbols.
    pub(crate) fn from_matrix(m: &Matrix, pool: &ExprPool) -> Option<Self> {
        debug_assert_eq!(m.rows, m.cols);
        let n = m.rows;
        let mut vars: Vec<ExprId> = Vec::new();
        let mut seen: IdMap<()> = IdMap::default();
        for &e in &m.data {
            collect_vars(e, pool, &mut vars, &mut seen).ok()?;
        }
        if vars.is_empty() {
            // All-numeric: the existing FLINT `fmpz_mat` route is the fast one.
            return None;
        }
        if !cofactor_route_expands_fully(m, pool) {
            return None;
        }
        let mut memo: IdMap<(MultiPoly, Integer)> = IdMap::default();
        let mut q: Vec<QPoly> = Vec::with_capacity(n * n);
        for &e in &m.data {
            let (num, den) = to_qpoly(e, &vars, pool, &mut memo).ok()?;
            q.push(QPoly { num, den });
        }
        let ctx = FlintMPolyCtx::new(vars.len());
        let mut entries = Vec::with_capacity(n * n);
        let mut row_scale = Vec::with_capacity(n);
        for r in 0..n {
            let row = &q[r * n..(r + 1) * n];
            let mut l = Integer::from(1);
            for e in row {
                l.lcm_mut(&e.den);
            }
            for e in row {
                let f = Integer::from(&l / &e.den);
                entries.push(multipoly_to_flint_scaled(&e.num, &f, &ctx));
            }
            row_scale.push(l);
        }
        Some(PolyMatrix {
            n,
            vars,
            ctx,
            entries,
            row_scale,
        })
    }

    /// Determinant of the submatrix without row `skip.0` and column `skip.1`
    /// (or of the whole matrix for `None`), as `(numerator, denominator)`:
    /// the fraction-free determinant of the scaled rows, and the product of
    /// the scales of the rows kept. `None` if the budget ran out.
    #[cfg(test)]
    fn det_scaled(&self, skip: Option<(usize, usize)>) -> Option<(FlintMPoly, Integer)> {
        let rows: Vec<usize> = (0..self.n)
            .filter(|&r| skip.is_none_or(|(sr, _)| r != sr))
            .collect();
        let cols: Vec<usize> = (0..self.n)
            .filter(|&c| skip.is_none_or(|(_, sc)| c != sc))
            .collect();
        let k = rows.len();
        let mut den = Integer::from(1);
        for &r in &rows {
            den *= &self.row_scale[r];
        }
        if k == 0 {
            return Some((self.constant_one(), den));
        }
        let mut a: Vec<FlintMPoly> = Vec::with_capacity(k * k);
        for &r in &rows {
            for &c in &cols {
                a.push(self.entries[r * self.n + c].clone());
            }
        }
        let d = bareiss(&mut a, k, &self.ctx)?;
        Some((d, den))
    }

    fn constant_one(&self) -> FlintMPoly {
        let mut p = FlintMPoly::new(Arc::clone(&self.ctx));
        p.push_term(&Integer::from(1), &vec![0u64; self.ctx.nvars()]);
        p.finish();
        p
    }

    /// `det` of the whole matrix, expanded and canonicalised.
    #[cfg(test)]
    pub(crate) fn det_expanded(&self, pool: &ExprPool) -> Option<ExprId> {
        let (num, den) = self.det_scaled(None)?;
        self.to_expr(&num, &den, pool)
    }

    /// `det` of the minor without row `skip_row` and column `skip_col`,
    /// expanded and canonicalised.
    #[cfg(test)]
    pub(crate) fn minor_det_expanded(
        &self,
        skip_row: usize,
        skip_col: usize,
        pool: &ExprPool,
    ) -> Option<ExprId> {
        let (num, den) = self.det_scaled(Some((skip_row, skip_col)))?;
        self.to_expr(&num, &den, pool)
    }

    /// The determinant and, when it is not zero, every first minor, all
    /// expanded and canonicalised: `(det, Some(minors))` with
    /// `minors[j * n + i]` the determinant of the matrix without row `j` and
    /// column `i` — exactly `minor_det_expanded(j, i)` — or `(0, None)` for a
    /// singular matrix. `None` if the budget ran out.
    ///
    /// One fraction-free Gauss–Jordan elimination of `[A | I]` (`2n³`
    /// polynomial operations) yields `[det(PA)·I | det(PA)·A⁻¹]` for the row
    /// permutation `P` it pivots with, i.e. the adjugate up to the sign of
    /// `P`, instead of `n²` separate eliminations.
    pub(crate) fn det_and_minors_expanded(
        &self,
        pool: &ExprPool,
    ) -> Option<(ExprId, Option<Vec<ExprId>>)> {
        let n = self.n;
        let Some((det, adj)) = self.gauss_jordan()? else {
            return Some((pool.integer(0_i32), None));
        };
        let mut all_scale = Integer::from(1);
        for d in &self.row_scale {
            all_scale *= d;
        }
        let det_expr = self.to_expr(&det, &all_scale, pool)?;
        let zero = FlintMPoly::new(Arc::clone(&self.ctx));
        let mut minors = Vec::with_capacity(n * n);
        for j in 0..n {
            // Kept rows: all but `j`.
            let den = Integer::from(&all_scale / &self.row_scale[j]);
            for i in 0..n {
                // minor(j, i) = (−1)^(i+j) · adj(A)[i][j].
                let a = &adj[i * n + j];
                let signed = if (i + j) % 2 == 0 {
                    a.clone()
                } else {
                    zero.sub(a)
                };
                minors.push(self.to_expr(&signed, &den, pool)?);
            }
        }
        Some((det_expr, Some(minors)))
    }

    /// Fraction-free Gauss–Jordan on the scaled rows: `Some(None)` if the
    /// matrix is singular, else `Some(Some((det, adj)))` with `adj` the
    /// row-major adjugate of the scaled matrix. `None` if the budget ran out
    /// or a division was inexact (which the theory rules out).
    #[allow(clippy::type_complexity)]
    fn gauss_jordan(&self) -> Option<Option<(FlintMPoly, Vec<FlintMPoly>)>> {
        let n = self.n;
        let w = 2 * n;
        let mut a: Vec<FlintMPoly> = Vec::with_capacity(n * w);
        for r in 0..n {
            for c in 0..n {
                a.push(self.entries[r * n + c].clone());
            }
            for c in 0..n {
                a.push(if c == r {
                    self.constant_one()
                } else {
                    FlintMPoly::new(Arc::clone(&self.ctx))
                });
            }
        }
        let mut negate = false;
        let mut prev: Option<FlintMPoly> = None;
        for k in 0..n {
            crate::budget::check().ok()?;
            if a[k * w + k].is_zero() {
                let Some(r) = (k + 1..n).find(|&r| !a[r * w + k].is_zero()) else {
                    return Some(None);
                };
                for c in 0..w {
                    a.swap(k * w + c, r * w + c);
                }
                negate = !negate;
            }
            let p = a[k * w + k].clone();
            for i in (0..n).filter(|&i| i != k) {
                let f = a[i * w + k].clone();
                for j in (0..w).filter(|&j| j != k) {
                    let akj_zero = a[k * w + j].is_zero();
                    if akj_zero && a[i * w + j].is_zero() {
                        continue;
                    }
                    let mut t = p.mul(&a[i * w + j]);
                    if !akj_zero && !f.is_zero() {
                        t = t.sub(&f.mul(&a[k * w + j]));
                    }
                    a[i * w + j] = match &prev {
                        None => t,
                        Some(q) => t.divides(q)?,
                    };
                }
                a[i * w + k] = FlintMPoly::new(Arc::clone(&self.ctx));
            }
            prev = Some(p);
        }
        // Every diagonal entry is now det(PA); the right block is
        // det(PA)·A⁻¹ = ±adj(A).
        let zero = FlintMPoly::new(Arc::clone(&self.ctx));
        let fix = |x: &FlintMPoly| if negate { zero.sub(x) } else { x.clone() };
        let det = fix(&a[(n - 1) * w + (n - 1)]);
        let mut adj = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                adj.push(fix(&a[r * w + n + c]));
            }
        }
        Some(Some((det, adj)))
    }

    /// `simplify_expanded(num / den)` written as a sum of monomials with
    /// reduced rational coefficients.
    fn to_expr(&self, num: &FlintMPoly, den: &Integer, pool: &ExprPool) -> Option<ExprId> {
        let raw = self.to_expr_raw(num, den, pool)?;
        Some(simplify_expanded(raw, pool).value)
    }

    /// `num / den` as a sum of monomials, before canonicalisation.
    fn to_expr_raw(&self, num: &FlintMPoly, den: &Integer, pool: &ExprPool) -> Option<ExprId> {
        let terms = num.try_terms()?;
        if terms.is_empty() {
            return Some(pool.integer(0_i32));
        }
        let summands: Vec<ExprId> = terms
            .iter()
            .map(|(exps, coeff)| {
                let c = Rational::from((coeff.clone(), den.clone()));
                let mut factors = Vec::with_capacity(exps.len() + 1);
                if c != 1 {
                    factors.push(if *c.denom() == 1 {
                        pool.integer(c.numer().clone())
                    } else {
                        pool.rational(c.numer().clone(), c.denom().clone())
                    });
                }
                for (i, &e) in exps.iter().enumerate() {
                    if e == 0 {
                        continue;
                    }
                    let v = self.vars[i];
                    factors.push(if e == 1 {
                        v
                    } else {
                        pool.pow(v, pool.integer(e))
                    });
                }
                match factors.len() {
                    0 => pool.integer(1_i32),
                    1 => factors[0],
                    _ => pool.mul(factors),
                }
            })
            .collect();
        Some(if summands.len() == 1 {
            summands[0]
        } else {
            pool.add(summands)
        })
    }
}

/// Whether `simplify_expanded` of the cofactor expansion would expand fully.
///
/// `ExpandPow` declines `(a₁+…+a_m)^k` when `k > 4` and `mᵏ > 4096`, leaving
/// that power unexpanded, and the simplifier collects a product of equal sums
/// into such a power before it distributes it. A determinant term multiplies
/// one entry from each row, so a sum `B` can reach at most the exponent
/// `S(B) = Σ_rows max_cols e(entry, B)`, where `e` over-counts the occurrences
/// of `B` in an entry (as an entry, a factor, or a power's base, nested sums
/// included). When some `S(B)` could cross the bound the cofactor route's
/// answer is not the expanded polynomial, so the Bareiss route — whose answer
/// always is — must not stand in for it.
fn cofactor_route_expands_fully(m: &Matrix, pool: &ExprPool) -> bool {
    const MAX_EXP: u64 = 4;
    const MAX_PRODUCTS: u64 = 4096;
    let n = m.rows;
    // Per sum (keyed by its simplified form): (summand bound, Σ over rows).
    let mut total: IdMap<(u64, u64)> = IdMap::default();
    let mut simplified: IdMap<ExprId> = IdMap::default();
    for r in 0..n {
        let mut row_max: IdMap<(u64, u64)> = IdMap::default();
        for c in 0..n {
            let mut counts: IdMap<(u64, u64)> = IdMap::default();
            sum_occurrences(m.get(r, c), 1, pool, &mut simplified, &mut counts);
            for (b, (terms, k)) in counts {
                let e = row_max.entry(b).or_insert((terms, 0));
                e.1 = e.1.max(k);
            }
        }
        for (b, (terms, k)) in row_max {
            let e = total.entry(b).or_insert((terms, 0));
            e.1 = e.1.saturating_add(k);
        }
    }
    total.values().all(|&(terms, k)| {
        k <= MAX_EXP
            || terms
                .checked_pow(u32::try_from(k).unwrap_or(u32::MAX))
                .is_some_and(|p| p <= MAX_PRODUCTS)
    })
}

/// Add `mult` to `counts[B]` for every sum `B` in `e` (keyed by
/// `simplify(B)`, with an upper bound on its summand count).
fn sum_occurrences(
    e: ExprId,
    mult: u64,
    pool: &ExprPool,
    simplified: &mut IdMap<ExprId>,
    counts: &mut IdMap<(u64, u64)>,
) {
    enum Node {
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, u64),
        Leaf,
    }
    let node = pool.with(e, |d| match d {
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => {
            let k = pool.with(*exp, |x| match x {
                ExprData::Integer(k) => k.0.to_u64(),
                _ => None,
            });
            Node::Pow(*base, k.unwrap_or(u64::MAX))
        }
        _ => Node::Leaf,
    });
    match node {
        Node::Leaf => {}
        Node::Mul(args) => {
            for a in args {
                sum_occurrences(a, mult, pool, simplified, counts);
            }
        }
        Node::Pow(base, k) => {
            sum_occurrences(base, mult.saturating_mul(k), pool, simplified, counts);
        }
        Node::Add(args) => {
            let key = *simplified
                .entry(e)
                .or_insert_with(|| crate::simplify::engine::simplify(e, pool).value);
            let terms = flat_summands(e, pool).max(flat_summands(key, pool));
            let slot = counts.entry(key).or_insert((terms, 0));
            slot.1 = slot.1.saturating_add(mult);
            for a in args {
                sum_occurrences(a, mult, pool, simplified, counts);
            }
        }
    }
}

/// Number of summands of `e` with nested sums flattened (1 for a non-sum).
fn flat_summands(e: ExprId, pool: &ExprPool) -> u64 {
    let args = pool.with(e, |d| match d {
        ExprData::Add(args) => Some(args.clone()),
        _ => None,
    });
    match args {
        None => 1,
        Some(args) => args.iter().map(|&a| flat_summands(a, pool)).sum(),
    }
}

/// Fraction-free Gaussian elimination on the `k × k` row-major `a`
/// (destroyed). Returns `det(a)`, or `None` if the budget ran out or FLINT
/// reported an inexact division (which Sylvester's identity rules out).
#[cfg(test)]
fn bareiss(a: &mut [FlintMPoly], k: usize, ctx: &Arc<FlintMPolyCtx>) -> Option<FlintMPoly> {
    let mut negate = false;
    let mut prev: Option<FlintMPoly> = None;
    for i in 0..k.saturating_sub(1) {
        crate::budget::check().ok()?;
        if a[i * k + i].is_zero() {
            let Some(r) = (i + 1..k).find(|&r| !a[r * k + i].is_zero()) else {
                return Some(FlintMPoly::new(Arc::clone(ctx)));
            };
            for c in 0..k {
                a.swap(i * k + c, r * k + c);
            }
            negate = !negate;
        }
        for r in i + 1..k {
            for c in i + 1..k {
                let t = a[r * k + c]
                    .mul(&a[i * k + i])
                    .sub(&a[r * k + i].mul(&a[i * k + c]));
                a[r * k + c] = match &prev {
                    None => t,
                    Some(p) => t.divides(p)?,
                };
            }
        }
        prev = Some(a[i * k + i].clone());
    }
    let d = a[(k - 1) * k + (k - 1)].clone();
    Some(if negate {
        FlintMPoly::new(Arc::clone(ctx)).sub(&d)
    } else {
        d
    })
}

/// `p · f` as an `fmpz_mpoly` in `ctx`.
fn multipoly_to_flint_scaled(p: &MultiPoly, f: &Integer, ctx: &Arc<FlintMPolyCtx>) -> FlintMPoly {
    let nvars = ctx.nvars();
    let mut fp = FlintMPoly::new(Arc::clone(ctx));
    let mut e = vec![0u64; nvars];
    for (exp, c) in &p.terms {
        e.iter_mut().for_each(|v| *v = 0);
        for (slot, &x) in e.iter_mut().zip(exp) {
            *slot = u64::from(x);
        }
        fp.push_term(&Integer::from(c * f), &e);
    }
    fp.finish();
    fp
}

/// Append the symbols of `e` to `vars` (first-appearance order), or refuse
/// a node the polynomial route does not take.
fn collect_vars(
    e: ExprId,
    pool: &ExprPool,
    vars: &mut Vec<ExprId>,
    seen: &mut IdMap<()>,
) -> Result<(), NotPolynomial> {
    if seen.insert(e, ()).is_some() {
        return Ok(());
    }
    enum Node {
        Leaf,
        Var,
        Children(Vec<ExprId>),
        Pow(ExprId),
        Refuse,
    }
    let node = pool.with(e, |d| match d {
        ExprData::Symbol { commutative, .. } => {
            if *commutative {
                Node::Var
            } else {
                Node::Refuse
            }
        }
        ExprData::Integer(_) | ExprData::Rational(_) => Node::Leaf,
        ExprData::Add(args) | ExprData::Mul(args) => Node::Children(args.clone()),
        ExprData::Pow { base, exp } => match pool.with(*exp, |x| match x {
            ExprData::Integer(k) => Some(k.0 >= 0),
            _ => None,
        }) {
            Some(true) => Node::Pow(*base),
            _ => Node::Refuse,
        },
        _ => Node::Refuse,
    });
    match node {
        Node::Leaf => Ok(()),
        Node::Var => {
            vars.push(e);
            Ok(())
        }
        Node::Children(args) => {
            for a in args {
                collect_vars(a, pool, vars, seen)?;
            }
            Ok(())
        }
        Node::Pow(base) => collect_vars(base, pool, vars, seen),
        Node::Refuse => Err(NotPolynomial),
    }
}

/// `e` as `num / den` over `ℤ[vars]`. Powers go through
/// `MultiPoly::from_symbolic`, whose size and exponent guards apply; a power
/// of a base with non-integer coefficients is declined.
fn to_qpoly(
    e: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
    memo: &mut IdMap<(MultiPoly, Integer)>,
) -> Result<(MultiPoly, Integer), NotPolynomial> {
    if let Some(r) = memo.get(&e) {
        return Ok(r.clone());
    }
    enum Node {
        Var(usize),
        Int(Integer),
        Rat(Integer, Integer),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow,
        Refuse,
    }
    let node = pool.with(e, |d| match d {
        ExprData::Symbol { .. } => match vars.iter().position(|&v| v == e) {
            Some(i) => Node::Var(i),
            None => Node::Refuse,
        },
        ExprData::Integer(k) => Node::Int(k.0.clone()),
        ExprData::Rational(q) => Node::Rat(q.0.numer().clone(), q.0.denom().clone()),
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { .. } => Node::Pow,
        _ => Node::Refuse,
    });
    let constant = |c: Integer| {
        let mut p = MultiPoly::zero(vars.to_vec());
        if c != 0 {
            p.terms.insert(vec![], c);
        }
        p
    };
    let r = match node {
        Node::Var(i) => {
            let mut exp = vec![0u32; i + 1];
            exp[i] = 1;
            let mut p = MultiPoly::zero(vars.to_vec());
            p.terms.insert(exp, Integer::from(1));
            (p, Integer::from(1))
        }
        Node::Int(k) => (constant(k), Integer::from(1)),
        Node::Rat(num, den) => (constant(num), den),
        Node::Add(args) => {
            let mut acc = MultiPoly::zero(vars.to_vec());
            let mut den = Integer::from(1);
            for a in args {
                let (p, d) = to_qpoly(a, vars, pool, memo)?;
                let l = Integer::from(den.lcm_ref(&d));
                let fa = Integer::from(&l / &den);
                let fp = Integer::from(&l / &d);
                acc = scale(acc, &fa) + scale(p, &fp);
                den = l;
            }
            (acc, den)
        }
        Node::Mul(args) => {
            let mut acc = constant(Integer::from(1));
            let mut den = Integer::from(1);
            for a in args {
                let (p, d) = to_qpoly(a, vars, pool, memo)?;
                acc = acc.checked_mul(&p).map_err(|_| NotPolynomial)?;
                den *= d;
            }
            (acc, den)
        }
        Node::Pow => {
            let p = MultiPoly::from_symbolic(e, vars.to_vec(), pool).map_err(|_| NotPolynomial)?;
            (p, Integer::from(1))
        }
        Node::Refuse => return Err(NotPolynomial),
    };
    // Keep `num / den` reduced so denominators do not grow through sums.
    let (mut num, mut den) = r;
    let g = Integer::from(num.integer_content().gcd_ref(&den));
    if g > 1 {
        num = num.div_integer(&g);
        den /= &g;
    }
    memo.insert(e, (num.clone(), den.clone()));
    Ok((num, den))
}

fn scale(p: MultiPoly, f: &Integer) -> MultiPoly {
    if *f == 1 {
        return p;
    }
    MultiPoly {
        vars: p.vars,
        terms: p.terms.into_iter().map(|(k, c)| (k, c * f)).collect(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// Random dense polynomial matrix: sums of small monomials in `x, y, z`
    /// with integer and rational coefficients, plus zeros and powers of sums.
    pub(crate) fn poly_matrix(n: usize, seed: u64, pool: &ExprPool) -> Matrix {
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let vs = [x, y, z];
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut rnd = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = Vec::with_capacity(n);
            for _ in 0..n {
                let e = match rnd(10) {
                    0 => pool.integer(0_i32),
                    1 => pool.integer(rnd(7) as i64 - 3),
                    2 => pool.rational(
                        rug::Integer::from(rnd(9) as i64 - 4),
                        rug::Integer::from(rnd(3) as i64 + 2),
                    ),
                    3 => pool.pow(
                        pool.add(vec![vs[rnd(3) as usize], pool.integer(rnd(3) as i64 - 1)]),
                        pool.integer(rnd(3) as i64 + 1),
                    ),
                    _ => {
                        let terms = rnd(3) + 1;
                        let ts: Vec<ExprId> = (0..terms)
                            .map(|_| {
                                let c = pool.integer(rnd(7) as i64 - 3);
                                let v = vs[rnd(3) as usize];
                                let d = rnd(3);
                                let m = if d == 0 {
                                    pool.integer(1_i32)
                                } else if d == 1 {
                                    v
                                } else {
                                    pool.pow(v, pool.integer(d as i64))
                                };
                                if rnd(4) == 0 {
                                    pool.mul(vec![
                                        pool.rational(rug::Integer::from(1), rug::Integer::from(2)),
                                        m,
                                    ])
                                } else {
                                    pool.mul(vec![c, m])
                                }
                            })
                            .collect();
                        pool.add(ts)
                    }
                };
                row.push(e);
            }
            rows.push(row);
        }
        Matrix::new(rows).unwrap()
    }

    /// The route `symbolic_inverse` took before: expand the cofactor
    /// expansion. The reference the Bareiss route must reproduce exactly.
    fn reference_det(m: &Matrix, pool: &ExprPool) -> ExprId {
        simplify_expanded(m.det(pool).unwrap(), pool).value
    }

    fn reference_minor(m: &Matrix, r: usize, c: usize, pool: &ExprPool) -> ExprId {
        let mut memo = super::super::DetMemo::new(m, pool);
        simplify_expanded(m.minor_det_memo(r, c, &mut memo, pool), pool).value
    }

    #[test]
    fn bareiss_det_is_the_expanded_cofactor_det() {
        let mut taken = 0;
        for n in 1..=7usize {
            let seeds = if n <= 5 { 40 } else { 8 };
            for seed in 0..seeds {
                // Bareiss first, so it cannot lean on nodes the reference
                // interned.
                let pool = ExprPool::new();
                let m = poly_matrix(n, seed * 131 + n as u64, &pool);
                let Some(pm) = PolyMatrix::from_matrix(&m, &pool) else {
                    continue;
                };
                taken += 1;
                let fast = pm.det_expanded(&pool).unwrap();
                assert_eq!(
                    fast,
                    reference_det(&m, &pool),
                    "n={n} seed={seed}: {}",
                    pool.display(fast)
                );
            }
        }
        assert!(
            taken > 150,
            "only {taken} matrices took the polynomial route"
        );
    }

    #[test]
    fn bareiss_minors_are_the_expanded_cofactor_minors() {
        for n in 2..=5usize {
            for seed in 0..6u64 {
                let pool = ExprPool::new();
                let m = poly_matrix(n, seed * 977 + n as u64, &pool);
                let Some(pm) = PolyMatrix::from_matrix(&m, &pool) else {
                    continue;
                };
                for r in 0..n {
                    for c in 0..n {
                        let fast = pm.minor_det_expanded(r, c, &pool).unwrap();
                        assert_eq!(
                            fast,
                            reference_minor(&m, r, c, &pool),
                            "n={n} seed={seed} minor ({r},{c})"
                        );
                    }
                }
            }
        }
    }

    /// The single Gauss–Jordan pass returns the same determinant and minors
    /// as one Bareiss elimination per minor, including through row swaps.
    #[test]
    fn gauss_jordan_minors_match_per_minor_bareiss() {
        let mut nonsingular = 0;
        for n in 1..=6usize {
            for seed in 0..10u64 {
                let pool = ExprPool::new();
                let m = poly_matrix(n, seed * 389 + n as u64, &pool);
                let Some(pm) = PolyMatrix::from_matrix(&m, &pool) else {
                    continue;
                };
                let (det, minors) = pm.det_and_minors_expanded(&pool).unwrap();
                assert_eq!(det, pm.det_expanded(&pool).unwrap(), "n={n} seed={seed}");
                let Some(minors) = minors else {
                    assert_eq!(det, pool.integer(0_i32));
                    continue;
                };
                nonsingular += 1;
                for j in 0..n {
                    for i in 0..n {
                        let want = if n == 1 {
                            pool.integer(1_i32)
                        } else {
                            pm.minor_det_expanded(j, i, &pool).unwrap()
                        };
                        assert_eq!(minors[j * n + i], want, "n={n} seed={seed} ({j},{i})");
                    }
                }
            }
        }
        assert!(nonsingular > 30, "{nonsingular}");
    }

    /// Singular matrices, zero pivots that need a row swap, and a zero
    /// determinant all come back as the reference's expression.
    #[test]
    fn pivoting_and_singular_matrices() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z0 = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let xy = pool.add(vec![x, y]);
        let two_y = pool.mul(vec![pool.integer(2_i32), y]);
        let cases = vec![
            // Zero leading pivot.
            vec![vec![z0, x, one], vec![y, z0, x], vec![one, y, z0]],
            // Repeated row: det 0.
            vec![vec![x, y, xy], vec![x, y, xy], vec![one, x, y]],
            // Column dependency: c2 = c0 + c1.
            vec![
                vec![x, y, xy],
                vec![one, x, pool.add(vec![one, x])],
                vec![y, y, two_y],
            ],
            // Zero column.
            vec![vec![z0, x], vec![z0, y]],
            // Only a later pivot is non-zero.
            vec![vec![z0, z0, x], vec![z0, y, one], vec![x, one, y]],
        ];
        for rows in cases {
            let m = Matrix::new(rows).unwrap();
            let pm = PolyMatrix::from_matrix(&m, &pool).unwrap();
            let det = reference_det(&m, &pool);
            assert_eq!(pm.det_expanded(&pool).unwrap(), det);
            let (gj_det, minors) = pm.det_and_minors_expanded(&pool).unwrap();
            assert_eq!(gj_det, det);
            if let Some(minors) = minors {
                let n = m.rows;
                for j in 0..n {
                    for i in 0..n {
                        assert_eq!(minors[j * n + i], reference_minor(&m, j, i, &pool));
                    }
                }
            }
        }
    }

    /// The route declines what it does not take, and the matrices where the
    /// cofactor route would stop short of a full expansion.
    #[test]
    fn declines_non_polynomial_and_unexpanded_inputs() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let one = pool.integer(1_i32);
        let two = pool.integer(2_i32);
        let declined = |rows: Vec<Vec<ExprId>>| {
            PolyMatrix::from_matrix(&Matrix::new(rows).unwrap(), &pool).is_none()
        };
        let sinx = pool.func("sin", vec![x]);
        assert!(declined(vec![vec![sinx, one], vec![one, x]]));
        let inv = pool.pow(x, pool.integer(-1_i32));
        assert!(declined(vec![vec![inv, one], vec![one, x]]));
        let symexp = pool.pow(x, y);
        assert!(declined(vec![vec![symexp, one], vec![one, x]]));
        let fl = pool.float(1.5, 53);
        assert!(declined(vec![vec![fl, x], vec![one, x]]));
        let a = pool.symbol_commutative("A", Domain::Real, false);
        assert!(declined(vec![vec![a, x], vec![one, x]]));
        // All numeric: the fmpz_mat route is already fast.
        assert!(declined(vec![vec![one, two], vec![two, one]]));
        // (x + y + 1)^9 = 3^9 > 4096 products: simplify_expanded leaves it.
        let b = pool.add(vec![x, y, one]);
        let big = pool.pow(b, pool.integer(9_i32));
        assert!(declined(vec![vec![big, one], vec![one, x]]));
        // The same sum on the diagonal of an 8×8: the cofactor route collects
        // its product into b^8, 3^8 > 4096.
        let diag = |n: usize| -> Matrix {
            let rows: Vec<Vec<ExprId>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { b } else { pool.integer(0_i32) })
                        .collect()
                })
                .collect();
            Matrix::new(rows).unwrap()
        };
        assert!(PolyMatrix::from_matrix(&diag(8), &pool).is_none());
        // ... but a 4×4 diagonal (b^4) is taken and agrees.
        let m = diag(4);
        let pm = PolyMatrix::from_matrix(&m, &pool).expect("taken");
        assert_eq!(pm.det_expanded(&pool).unwrap(), reference_det(&m, &pool));
    }

    /// Matrices built to repeat a few sums across rows, so the reference
    /// collects powers of them: wherever the route is taken it must agree.
    #[test]
    fn repeated_sums_agree_or_decline() {
        let mut state = 0x1234_5678_9ABC_DEF1_u64;
        let mut rnd = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let (mut taken, mut declined) = (0, 0);
        for trial in 0..60usize {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let one = pool.integer(1_i32);
            let sums = [
                pool.add(vec![x, one]),
                pool.add(vec![x, y]),
                pool.add(vec![x, y, one]),
                pool.add(vec![
                    x,
                    pool.mul(vec![pool.integer(2_i32), y]),
                    pool.integer(-3_i32),
                ]),
            ];
            let n = 2 + (trial % 6);
            let rows: Vec<Vec<ExprId>> = (0..n)
                .map(|_| {
                    (0..n)
                        .map(|_| match rnd(6) {
                            0 => pool.integer(0_i32),
                            1 => pool.pow(sums[rnd(4) as usize], pool.integer(rnd(3) as i64 + 2)),
                            2 => pool
                                .mul(vec![pool.integer(rnd(5) as i64 - 2), sums[rnd(4) as usize]]),
                            _ => sums[rnd(4) as usize],
                        })
                        .collect()
                })
                .collect();
            let m = Matrix::new(rows).unwrap();
            match PolyMatrix::from_matrix(&m, &pool) {
                None => declined += 1,
                Some(pm) => {
                    taken += 1;
                    assert_eq!(
                        pm.det_expanded(&pool).unwrap(),
                        reference_det(&m, &pool),
                        "trial {trial} n={n}"
                    );
                }
            }
        }
        assert!(
            taken > 10 && declined > 0,
            "taken {taken}, declined {declined}"
        );
    }

    /// `cargo test --release -p alkahest-cas poly_det_timing -- --ignored --nocapture`
    #[test]
    #[ignore = "timing report"]
    fn poly_det_timing() {
        for n in 3..=8usize {
            let pool = ExprPool::new();
            let m = poly_matrix(n, 99 + n as u64, &pool);
            let t = std::time::Instant::now();
            let _ = reference_det(&m, &pool);
            let t_old = t.elapsed();
            let pool2 = ExprPool::new();
            let m2 = poly_matrix(n, 99 + n as u64, &pool2);
            let t = std::time::Instant::now();
            let pm = PolyMatrix::from_matrix(&m2, &pool2).unwrap();
            let _ = pm.det_expanded(&pool2).unwrap();
            let t_new = t.elapsed();
            println!("n={n}: expanded cofactor {t_old:?}  bareiss {t_new:?}");
        }
    }
}
