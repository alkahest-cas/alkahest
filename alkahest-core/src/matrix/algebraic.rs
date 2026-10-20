//! Exact linear algebra over a number field `K = ℚ(α)`, for eigenvalues given
//! as `RootOf(p, k)`.
//!
//! # Why a number field
//!
//! An eigenvalue `α = RootOf(p, k)` of a rational matrix `A` is exact, but the
//! symbolic elimination in [`crate::matrix::eigen`] decides every pivot with
//! [`crate::matrix::zero_test`], which knows nothing about the relation
//! `p(α) = 0`: `α⁴ + α + 1` is a non-zero polynomial in an opaque atom to it.
//! Eliminating `A − αI` there either refuses or — worse — takes a pivot that is
//! really zero. So `eigenvects`, `jordan_form` and `matrix_exp` used to refuse a
//! `RootOf` spectrum outright.
//!
//! Here the arithmetic is done in `K = ℚ[x]/(p)` with FLINT's `nf_elem`
//! ([`crate::numfield`]), where equality is exact: an element is its
//! coordinate vector in the basis `1, α, …, α^{d−1}`, and it is zero exactly
//! when every coordinate is. `p` must be irreducible (it is checked, never
//! assumed), which is what makes `K` a field.
//!
//! # One computation, every conjugate
//!
//! Everything computed in `K` is a rational expression in `α` built only from
//! `A`'s rational entries and exact zero tests in `K`. The conjugate embeddings
//! `α ↦ RootOf(p, j)` are field isomorphisms, so they preserve every one of
//! those zero tests: the *same* coordinates, read with `α := RootOf(p, j)`, are
//! the answer for the `j`-th conjugate. Kernels, Jordan chains and spectral
//! projectors are therefore computed once per irreducible factor and printed
//! once per root, as polynomials of degree `< d` in the `RootOf` node.
//!
//! # Scope
//!
//! * `A` must have rational entries (`matrix_exp` also accepts `A·t` for one
//!   symbol `t`). Symbolic entries are not a number field and are refused as
//!   before.
//! * The eigenvalue must be a bare `RootOf` node; its polynomial must be
//!   irreducible over ℚ.
//! * Every loop honours the active [`crate::budget`]: a trip is a refusal, not
//!   a partial answer.

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::matrix::Matrix;
use crate::numfield::{NumberField, NumberFieldElement};
use crate::poly::root_of::{make_root_of, rational_expr, root_of_parts};
use crate::simplify::engine::simplify;
use rug::{Integer, Rational};
use std::collections::HashMap;

type Nfe = NumberFieldElement;
/// A dense matrix over `K`, row-major.
type NfMat = Vec<Vec<Nfe>>;

/// Largest field degree this module will work in. Kernel computations are
/// `O(n³)` field operations, each `O(d²)` coefficient operations; a degree
/// above this is far outside anything an eigenvalue problem here produces and
/// is refused rather than attempted.
const MAX_DEGREE: usize = 64;

// ---------------------------------------------------------------------------
// The field of a RootOf
// ---------------------------------------------------------------------------

/// `K = ℚ[x]/(p)` for the `p` of a `RootOf(p, k)` node; the field's
/// generator is the root the coordinates refer to.
#[derive(Clone)]
pub(crate) struct AlgField {
    field: NumberField,
}

thread_local! {
    /// `p ↦ ℚ[x]/(p)` (or `None` when `p` is reducible). Building a field runs
    /// an irreducibility test; the zero-test hook can ask about the same `p`
    /// many times in one elimination.
    static FIELDS: std::cell::RefCell<HashMap<Vec<Integer>, Option<NumberField>>> =
        std::cell::RefCell::new(HashMap::new());
}

impl AlgField {
    /// The field of the irreducible polynomial with these integer coefficients
    /// (ascending, primitive, positive leading coefficient); `None` when it is
    /// reducible or too large.
    pub(crate) fn from_coeffs(coeffs: &[Integer]) -> Option<Self> {
        if coeffs.len() < 2 || coeffs.len() - 1 > MAX_DEGREE {
            return None;
        }
        let field = FIELDS.with(|f| {
            f.borrow_mut()
                .entry(coeffs.to_vec())
                .or_insert_with(|| NumberField::from_integer_coeffs(coeffs).ok())
                .clone()
        })?;
        // The field normalises its polynomial; the coordinates of an element
        // refer to a root of *that* polynomial, so it must be the same one.
        if field.defining_polynomial() != coeffs {
            return None;
        }
        Some(Self { field })
    }

    /// The field of a `RootOf(p, k)` node, and `k`.
    pub(crate) fn of_root_of(e: ExprId, pool: &ExprPool) -> Option<(Self, usize)> {
        let (coeffs, k) = root_of_parts(e, pool)?;
        Some((Self::from_coeffs(&coeffs)?, k))
    }

    fn alpha(&self) -> Nfe {
        self.field.generator()
    }

    fn rat(&self, r: &Rational) -> Nfe {
        self.field.rational(r)
    }

    fn zero(&self) -> Nfe {
        self.field.zero()
    }

    fn one(&self) -> Nfe {
        self.field.one()
    }
}

// ---------------------------------------------------------------------------
// Elements ⇄ expressions
// ---------------------------------------------------------------------------

/// `c₀ + c₁·r + … + c_{d−1}·r^{d−1}` for the coordinates of `e`, with `r` the
/// expression that stands for the generator (a `RootOf` node, a radical, or a
/// bound `RootSum` variable).
pub(crate) fn element_to_expr(e: &Nfe, r: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (i, c) in e.coefficients().iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let coeff = rational_expr(c, pool);
        terms.push(match i {
            0 => coeff,
            1 => pool.mul(vec![coeff, r]),
            _ => pool.mul(vec![coeff, pool.pow(r, pool.integer(i as i64))]),
        });
    }
    let sum = match terms.len() {
        0 => pool.integer(0),
        1 => terms[0],
        _ => pool.add(terms),
    };
    simplify(sum, pool).value
}

/// The value of `e` in `K`, reading the node `root` as the generator; `None`
/// when `e` is not a rational expression in `root` alone, or divides by zero.
fn expr_to_element(
    e: ExprId,
    root: ExprId,
    k: &AlgField,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, Nfe>,
) -> Option<Nfe> {
    if let Some(v) = memo.get(&e) {
        return Some(v.clone());
    }
    let v = if e == root {
        k.alpha()
    } else {
        match pool.get(e) {
            ExprData::Integer(n) => k.rat(&Rational::from(n.0)),
            ExprData::Rational(r) => k.rat(&r.0),
            ExprData::Add(args) => {
                let mut acc = k.zero();
                for a in args {
                    acc = acc.add(&expr_to_element(a, root, k, pool, memo)?).ok()?;
                }
                acc
            }
            ExprData::Mul(args) => {
                let mut acc = k.one();
                for a in args {
                    acc = acc.mul(&expr_to_element(a, root, k, pool, memo)?).ok()?;
                }
                acc
            }
            ExprData::Pow { base, exp } => {
                let n = match pool.get(exp) {
                    ExprData::Integer(n) => n.0.to_i64()?,
                    _ => return None,
                };
                if n.unsigned_abs() > 4096 {
                    return None;
                }
                let b = expr_to_element(base, root, k, pool, memo)?;
                let b = if n < 0 { b.inverse().ok()? } else { b };
                b.try_pow(n.unsigned_abs()).ok()?
            }
            _ => return None,
        }
    };
    memo.insert(e, v.clone());
    Some(v)
}

/// Exact zero test for an expression that is a rational function, over ℚ, of
/// a **single** `RootOf` node with an irreducible polynomial: `Some(true)` when
/// it is zero, `Some(false)` when it is not, `None` when `e` is not of that
/// shape (another atom, two different `RootOf` nodes, a reducible polynomial,
/// a division by zero).
///
/// Sound by construction: the value is computed in `ℚ[x]/(p) ≅ ℚ(α)`, where
/// zero means every coordinate is zero.
pub(crate) fn root_of_rational_is_zero(e: ExprId, pool: &ExprPool) -> Option<bool> {
    let mut root: Option<ExprId> = None;
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(x) {
            continue;
        }
        if root_of_parts(x, pool).is_some() {
            match root {
                Some(r) if r != x => return None,
                _ => root = Some(x),
            }
            continue;
        }
        let ok = pool.with(x, |d| match d {
            ExprData::Integer(_) | ExprData::Rational(_) => true,
            ExprData::Add(a) | ExprData::Mul(a) => {
                stack.extend(a.iter().copied());
                true
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                matches!(pool.get(*exp), ExprData::Integer(_))
            }
            _ => false,
        });
        if !ok {
            return None;
        }
    }
    let root = root?;
    let (k, _) = AlgField::of_root_of(root, pool)?;
    let v = expr_to_element(e, root, &k, pool, &mut HashMap::new())?;
    Some(v.is_zero())
}

// ---------------------------------------------------------------------------
// Matrices over K
// ---------------------------------------------------------------------------

fn budget_ok() -> Option<()> {
    crate::budget::check().ok()
}

fn mat_from_rational(a: &[Vec<Rational>], k: &AlgField) -> NfMat {
    a.iter()
        .map(|row| row.iter().map(|q| k.rat(q)).collect())
        .collect()
}

/// `A − αI`.
fn shifted(a: &[Vec<Rational>], k: &AlgField) -> Option<NfMat> {
    let mut m = mat_from_rational(a, k);
    let alpha = k.alpha();
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = row[i].sub(&alpha).ok()?;
    }
    Some(m)
}

fn mat_mul(a: &NfMat, b: &NfMat, k: &AlgField) -> Option<NfMat> {
    let (n, inner) = (a.len(), b.len());
    let m = b.first().map_or(0, Vec::len);
    let mut out = vec![vec![k.zero(); m]; n];
    for i in 0..n {
        budget_ok()?;
        for l in 0..inner {
            if a[i][l].is_zero() {
                continue;
            }
            for j in 0..m {
                if b[l][j].is_zero() {
                    continue;
                }
                let t = a[i][l].mul(&b[l][j]).ok()?;
                out[i][j] = out[i][j].add(&t).ok()?;
            }
        }
    }
    Some(out)
}

fn transpose(a: &NfMat) -> NfMat {
    let m = a.first().map_or(0, Vec::len);
    (0..m)
        .map(|j| a.iter().map(|row| row[j].clone()).collect())
        .collect()
}

/// Reduced row echelon form in place; returns the pivot columns.
fn rref(a: &mut NfMat) -> Option<Vec<usize>> {
    let rows = a.len();
    let cols = a.first().map_or(0, Vec::len);
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..cols {
        if r == rows {
            break;
        }
        budget_ok()?;
        let Some(p) = (r..rows).find(|&i| !a[i][c].is_zero()) else {
            continue;
        };
        a.swap(r, p);
        let inv = a[r][c].inverse().ok()?;
        for j in c..cols {
            a[r][j] = a[r][j].mul(&inv).ok()?;
        }
        for i in 0..rows {
            if i == r || a[i][c].is_zero() {
                continue;
            }
            let f = a[i][c].clone();
            for j in c..cols {
                if a[r][j].is_zero() {
                    continue;
                }
                let t = f.mul(&a[r][j]).ok()?;
                a[i][j] = a[i][j].sub(&t).ok()?;
            }
        }
        pivots.push(c);
        r += 1;
    }
    Some(pivots)
}

/// A basis of `{v : a·v = 0}`, one vector per free column with that
/// coordinate `1` (the same normalisation as the rational nullspace).
fn nullspace(a: &NfMat, k: &AlgField) -> Option<Vec<Vec<Nfe>>> {
    let cols = a.first().map_or(0, Vec::len);
    let mut r = a.clone();
    let pivots = rref(&mut r)?;
    let mut basis = Vec::new();
    for free in (0..cols).filter(|c| !pivots.contains(c)) {
        let mut v = vec![k.zero(); cols];
        v[free] = k.one();
        for (row, &pc) in pivots.iter().enumerate() {
            v[pc] = r[row][free].neg();
        }
        basis.push(v);
    }
    Some(basis)
}

/// Rank of a list of vectors (as the rows of a matrix).
fn rank_of(vectors: &[Vec<Nfe>]) -> Option<usize> {
    let mut m = vectors.to_vec();
    Some(rref(&mut m)?.len())
}

fn mat_vec(a: &NfMat, v: &[Nfe], k: &AlgField) -> Option<Vec<Nfe>> {
    let mut out = Vec::with_capacity(a.len());
    for row in a {
        let mut acc = k.zero();
        for (x, y) in row.iter().zip(v) {
            if x.is_zero() || y.is_zero() {
                continue;
            }
            acc = acc.add(&x.mul(y).ok()?).ok()?;
        }
        out.push(acc);
    }
    Some(out)
}

fn inverse(a: &NfMat, k: &AlgField) -> Option<NfMat> {
    let n = a.len();
    let mut aug: NfMat = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { k.one() } else { k.zero() }));
            r
        })
        .collect();
    let pivots = rref(&mut aug)?;
    if pivots.len() < n || pivots.iter().enumerate().any(|(i, &p)| p != i) {
        return None;
    }
    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

fn power(a: &NfMat, e: usize, k: &AlgField) -> Option<NfMat> {
    let n = a.len();
    let mut out: NfMat = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { k.one() } else { k.zero() })
                .collect()
        })
        .collect();
    for _ in 0..e {
        out = mat_mul(&out, a, k)?;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Rational input
// ---------------------------------------------------------------------------

/// The entries of `m` as rationals, or `None` if any is not a rational number.
pub(crate) fn rational_grid(m: &Matrix, pool: &ExprPool) -> Option<Vec<Vec<Rational>>> {
    let mut g = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let mut row = Vec::with_capacity(m.cols);
        for c in 0..m.cols {
            let e = simplify(m.get(r, c), pool).value;
            row.push(match pool.get(e) {
                ExprData::Integer(n) => Rational::from(n.0),
                ExprData::Rational(q) => q.0,
                _ => return None,
            });
        }
        g.push(row);
    }
    Some(g)
}

fn column(v: &[Nfe], r: ExprId, pool: &ExprPool) -> Matrix {
    let mut out = Matrix::zeros(v.len(), 1, pool);
    for (i, e) in v.iter().enumerate() {
        out.set(i, 0, element_to_expr(e, r, pool));
    }
    out
}

// ---------------------------------------------------------------------------
// Which field an eigenvalue lives in
// ---------------------------------------------------------------------------

/// The number field of the eigenvalue `lambda` of the rational matrix `a`.
///
/// * A bare `RootOf(p, k)` is in `ℚ[x]/(p)` by definition.
/// * Any other closed form (a Cardano radical, a `cos(2π/9)` triple) came out
///   of `eigenvalues` as a root of some irreducible factor `f` of
///   `χ_A = Π fᵢ^{mᵢ}`. Which one is **proven**, not guessed: every other
///   factor is shown non-zero at `lambda` by the rigorous zero test, so `f` is
///   the only factor that can vanish there, and since `χ_A(λ) = 0` it does. A
///   floating-point residual is checked on top as a sanity screen. If more
///   than one factor survives the test, the eigenvalue is not placed (`None`).
///
/// Substituting `lambda` for the generator in anything computed over
/// `ℚ[x]/(f)` is then the conjugate embedding that sends the generator to
/// `lambda` — exactly as for a `RootOf`.
fn eigenvalue_field(a: &[Vec<Rational>], lambda: ExprId, pool: &ExprPool) -> Option<AlgField> {
    if let Some((k, _)) = AlgField::of_root_of(lambda, pool) {
        return Some(k);
    }
    // A closed-form *number*: no `RootOf` inside, and it evaluates with no
    // bindings (named constants such as `pi` are fine, a parameter is not).
    if crate::poly::root_of::max_root_of_degree(lambda, pool).is_some() {
        return None;
    }
    let lv = crate::eval::eval_complex_f64(lambda, pool, &HashMap::new()).ok()?;
    let n = a.len();
    let mut am = Matrix::zeros(n, n, pool);
    for i in 0..n {
        for j in 0..n {
            am.set(i, j, rational_expr(&a[i][j], pool));
        }
    }
    let (chi, lam) =
        crate::matrix::eigen::characteristic_polynomial_lambda_minus_m(&am, pool).ok()?;
    let up = crate::poly::UniPoly::from_symbolic_clear_denoms(chi, lam, pool).ok()?;
    let fac = crate::poly::factor_univariate_z(&up).ok()?;
    let mut survivor: Option<Vec<Integer>> = None;
    for (f, _) in &fac.factors {
        if f.degree() < 1 {
            continue;
        }
        let c = primitive_positive(&f.coefficients());
        let at = poly_expr(&c, lambda, pool);
        if matches!(
            crate::matrix::zero_test::zero_status(pool, at),
            crate::matrix::zero_test::ZeroStatus::NonZero
        ) {
            continue;
        }
        if survivor.is_some() {
            return None;
        }
        // Sanity screen: the value really is (numerically) zero.
        let v = crate::eval::eval_complex_f64(at, pool, &HashMap::new()).ok()?;
        let scale: f64 = c.iter().map(|x| x.to_f64().abs()).sum::<f64>().max(1.0);
        let mag = lv.re.hypot(lv.im).max(1.0).powi(c.len() as i32);
        if v.re.hypot(v.im) > 1e-8 * scale * mag {
            return None;
        }
        survivor = Some(c);
    }
    let c = survivor?;
    if c.len() < 3 {
        return None;
    }
    AlgField::from_coeffs(&c)
}

// ---------------------------------------------------------------------------
// Eigenvectors
// ---------------------------------------------------------------------------

/// Eigenvectors of the rational matrix `m` for the eigenvalue `lambda`, a bare
/// `RootOf(p, k)` node with `p` irreducible: a basis of `ker(A − αI)` over
/// `K = ℚ(α)`, each entry printed as a polynomial in `lambda`.
///
/// `None` when the input is outside the scope of the module docs, or the
/// kernel is empty (then `lambda` is not an eigenvalue of `m` at all).
pub(crate) fn field_eigenvectors(
    m: &Matrix,
    lambda: ExprId,
    pool: &ExprPool,
) -> Option<Vec<Matrix>> {
    let a = rational_grid(m, pool)?;
    let k = eigenvalue_field(&a, lambda, pool)?;
    let basis = nullspace(&shifted(&a, &k)?, &k)?;
    if basis.is_empty() {
        return None;
    }
    Some(basis.iter().map(|v| column(v, lambda, pool)).collect())
}

// ---------------------------------------------------------------------------
// Jordan chains
// ---------------------------------------------------------------------------

/// Jordan chains of the rational matrix `a` at a root `α` of the irreducible
/// `p` of `k`, whose algebraic multiplicity is `mult`: one entry per Jordan
/// block, largest first, each `(size, [N^{s−1}g, …, N g, g])` for
/// `N = A − αI` and a generator `g`.
///
/// The construction is the textbook one and is exact: for each size `s`, from
/// the largest down, the generators extend a basis of
/// `ker N^{s−1} + N^{t−s}·(generators of size t > s)` to one of `ker N^s`. With
/// exact rank in `K` that greedy extension is always possible — no search over
/// combinations, no symbolic rank check after the fact.
fn jordan_chains(
    a: &[Vec<Rational>],
    k: &AlgField,
    mult: usize,
) -> Option<Vec<(usize, Vec<Vec<Nfe>>)>> {
    let n_mat = shifted(a, k)?;
    // kernels[s] = basis of ker N^s, s = 0..=mult.
    let mut kernels: Vec<Vec<Vec<Nfe>>> = vec![Vec::new()];
    let mut pow = n_mat.clone();
    for s in 1..=mult {
        if s > 1 {
            pow = mat_mul(&pow, &n_mat, k)?;
        }
        kernels.push(nullspace(&pow, k)?);
    }
    if kernels[mult].len() != mult {
        // The generalised eigenspace has dimension `mult`; anything else means
        // `mult` is not α's multiplicity.
        return None;
    }
    let dims: Vec<usize> = kernels.iter().map(Vec::len).collect();
    // Number of blocks of size ≥ s is dims[s] − dims[s−1].
    let at_least = |s: usize| -> usize {
        if s > mult {
            0
        } else {
            dims[s] - dims[s - 1]
        }
    };
    let top = (1..=mult).rev().find(|&s| at_least(s) > 0)?;
    let mut generators: Vec<(usize, Vec<Nfe>)> = Vec::new();
    for s in (1..=top).rev() {
        let wanted = at_least(s) - at_least(s + 1);
        if wanted == 0 {
            continue;
        }
        let mut span: Vec<Vec<Nfe>> = kernels[s - 1].clone();
        for (t, g) in &generators {
            let mut v = g.clone();
            for _ in 0..(t - s) {
                v = mat_vec(&n_mat, &v, k)?;
            }
            span.push(v);
        }
        let mut r = rank_of(&span)?;
        let mut found = 0;
        for cand in &kernels[s] {
            if found == wanted {
                break;
            }
            span.push(cand.clone());
            let r2 = rank_of(&span)?;
            if r2 > r {
                r = r2;
                generators.push((s, cand.clone()));
                found += 1;
            } else {
                span.pop();
            }
        }
        if found != wanted {
            return None;
        }
    }
    let mut out = Vec::with_capacity(generators.len());
    for (s, g) in generators {
        let mut chain = vec![g];
        for _ in 1..s {
            let next = mat_vec(&n_mat, chain.last().expect("non-empty"), k)?;
            chain.push(next);
        }
        chain.reverse();
        out.push((s, chain));
    }
    Some(out)
}

/// Jordan blocks of the rational matrix `m` at `lambda = RootOf(p, k)` of
/// algebraic multiplicity `mult`: `(size, chain columns)` per block, largest
/// first, the columns printed as polynomials in `lambda`. Scope as
/// [`field_eigenvectors`].
pub(crate) fn field_jordan_blocks(
    m: &Matrix,
    lambda: ExprId,
    mult: usize,
    pool: &ExprPool,
) -> Option<Vec<(usize, Vec<Matrix>)>> {
    let a = rational_grid(m, pool)?;
    let k = eigenvalue_field(&a, lambda, pool)?;
    let chains = jordan_chains(&a, &k, mult)?;
    Some(
        chains
            .into_iter()
            .map(|(s, cols)| (s, cols.iter().map(|v| column(v, lambda, pool)).collect()))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Matrix exponential
// ---------------------------------------------------------------------------

/// `m = s·A` with `A` rational: `(s, A)`, where `s` is `1` for a rational `m`
/// and a single symbol `t` when every entry is a rational multiple of `t`.
fn scalar_times_rational(m: &Matrix, pool: &ExprPool) -> Option<(ExprId, Vec<Vec<Rational>>)> {
    if let Some(a) = rational_grid(m, pool) {
        return Some((pool.integer(1), a));
    }
    let mut free = std::collections::BTreeSet::new();
    for &e in m.entries() {
        free.extend(crate::poly::collect_free_vars(e, pool));
    }
    if free.len() != 1 {
        return None;
    }
    let t = *free.iter().next()?;
    let inv_t = pool.pow(t, pool.integer(-1));
    let mut scaled = Matrix::zeros(m.rows, m.cols, pool);
    for r in 0..m.rows {
        for c in 0..m.cols {
            let e = crate::simplify::engine::simplify_expanded(
                pool.mul(vec![m.get(r, c), inv_t]),
                pool,
            )
            .value;
            scaled.set(r, c, e);
        }
    }
    Some((t, rational_grid(&scaled, pool)?))
}

/// `e^{M}` for `M = s·A`, `A` rational, by spectral projectors computed
/// exactly in the number field of each irreducible factor of `A`'s
/// characteristic polynomial:
///
/// ```text
///   e^{sA} = Σ_f Σ_{α : f(α)=0} e^{sα} Σ_{j<m_f} (s^j/j!)·(A − αI)^j·E_α
/// ```
///
/// with `E_α = V (WᵀV)⁻¹ Wᵀ` the projector onto the generalised eigenspace
/// `V = ker(A − αI)^{m}` along the others (`W = ker(Aᵀ − αI)^{m}` spans the
/// annihilator of the other generalised eigenspaces). Every `E_α`, `(A − αI)^j`
/// entry is an element of `ℚ(α)`, computed once per factor; the inner sum over
/// the conjugates is written out for a rational or quadratic factor, and is a
/// `RootSum` over `f` for degree ≥ 3 — a real number for real `s`, which the
/// numeric evaluators read.
///
/// `None` outside that scope (symbolic entries, a field too large, a budget
/// trip); the caller then refuses as before.
pub(crate) fn matrix_exp_by_projectors(m: &Matrix, pool: &ExprPool) -> Option<Matrix> {
    if m.rows != m.cols || m.rows == 0 {
        return None;
    }
    let n = m.rows;
    let (s, a) = scalar_times_rational(m, pool)?;
    let mut am = Matrix::zeros(n, n, pool);
    for i in 0..n {
        for j in 0..n {
            am.set(i, j, rational_expr(&a[i][j], pool));
        }
    }
    let (chi, lam) =
        crate::matrix::eigen::characteristic_polynomial_lambda_minus_m(&am, pool).ok()?;
    let up = crate::poly::UniPoly::from_symbolic_clear_denoms(chi, lam, pool).ok()?;
    let fac = crate::poly::factor_univariate_z(&up).ok()?;
    let mut total = 0usize;
    let mut out: Vec<Vec<Vec<ExprId>>> = vec![vec![Vec::new(); n]; n];
    for (f, mult) in &fac.factors {
        let d = usize::try_from(f.degree()).ok()?;
        if d == 0 {
            continue;
        }
        let mult = *mult as usize;
        total += d * mult;
        let coeffs = primitive_positive(&f.coefficients());
        let k = AlgField::from_coeffs(&coeffs)?;
        let nm = shifted(&a, &k)?;
        let nm_pow = power(&nm, mult, &k)?;
        let v = nullspace(&nm_pow, &k)?;
        let w = nullspace(&transpose(&nm_pow), &k)?;
        if v.len() != mult || w.len() != mult {
            return None;
        }
        // V: n×m (columns v), Wᵀ: m×n (rows w).
        let v_mat: NfMat = transpose(&v);
        let g = mat_mul(&w, &v_mat, &k)?;
        let g_inv = inverse(&g, &k)?;
        let e = mat_mul(&mat_mul(&v_mat, &g_inv, &k)?, &w, &k)?;
        // F_j = (A − αI)^j E, j < mult.
        let mut fs = vec![e];
        for _ in 1..mult {
            let next = mat_mul(&nm, fs.last().expect("non-empty"), &k)?;
            fs.push(next);
        }
        // body_ij(r) = e^{s r} · Σ_j s^j/j! · F_j[i][j](r).
        let body = |i: usize, jj: usize, r: ExprId| -> Option<ExprId> {
            let mut poly_terms = Vec::new();
            let mut fact = Integer::from(1);
            for (j, fj) in fs.iter().enumerate() {
                if j > 0 {
                    fact *= j as u64;
                }
                let entry = &fj[i][jj];
                if entry.is_zero() {
                    continue;
                }
                let ex = element_to_expr(entry, r, pool);
                let coeff = rational_expr(&Rational::from((Integer::from(1), fact.clone())), pool);
                poly_terms.push(match j {
                    0 => ex,
                    _ => pool.mul(vec![coeff, pool.pow(s, pool.integer(j as i64)), ex]),
                });
            }
            if poly_terms.is_empty() {
                return None;
            }
            let poly = if poly_terms.len() == 1 {
                poly_terms[0]
            } else {
                pool.add(poly_terms)
            };
            let exp = pool.func("exp", vec![pool.mul(vec![s, r])]);
            Some(pool.mul(vec![exp, poly]))
        };
        for i in 0..n {
            for jj in 0..n {
                match d {
                    1 => {
                        let root = rational_expr(
                            &Rational::from((-coeffs[0].clone(), coeffs[1].clone())),
                            pool,
                        );
                        if let Some(b) = body(i, jj, root) {
                            out[i][jj].push(b);
                        }
                    }
                    2 => {
                        // Radicals, rewritten to real form when the pair is
                        // complex (`e^{±i√2 t}` → cos/sin); a pair the
                        // rewrite cannot settle is left to a `RootSum`,
                        // which the real evaluators read.
                        let (r1, r2) = crate::matrix::eigen::quadratic_roots(f, pool).ok()?;
                        let terms: Vec<ExprId> = [r1, r2]
                            .into_iter()
                            .filter_map(|r| body(i, jj, r))
                            .collect();
                        if terms.is_empty() {
                            continue;
                        }
                        let pair = simplify(pool.add(terms), pool).value;
                        let pair = if crate::simplify::real_form::looks_complex(pair, pool) {
                            crate::simplify::real_form::to_real_form_ambient(pair, pool)
                        } else {
                            pair
                        };
                        if crate::simplify::real_form::looks_complex(pair, pool) {
                            let r = pool.symbol("$root$", Domain::Complex);
                            if let Some(b) = body(i, jj, r) {
                                let fpoly = poly_expr(&coeffs, r, pool);
                                out[i][jj].push(pool.root_sum(fpoly, r, b));
                            }
                        } else {
                            out[i][jj].push(pair);
                        }
                    }
                    _ => {
                        // The bound root placeholder, as the integrator's
                        // Rothstein–Trager `RootSum` writes it.
                        let r = pool.symbol("$root$", Domain::Complex);
                        if let Some(b) = body(i, jj, r) {
                            let fpoly = poly_expr(&coeffs, r, pool);
                            out[i][jj].push(pool.root_sum(fpoly, r, b));
                        }
                    }
                }
            }
        }
    }
    if total != n {
        return None;
    }
    let mut res = Matrix::zeros(n, n, pool);
    for i in 0..n {
        for j in 0..n {
            let terms = std::mem::take(&mut out[i][j]);
            let e = match terms.len() {
                0 => pool.integer(0),
                1 => terms[0],
                _ => pool.add(terms),
            };
            res.set(i, j, simplify(e, pool).value);
        }
    }
    Some(res)
}

/// `Σ cᵢ rⁱ`.
fn poly_expr(coeffs: &[Integer], r: ExprId, pool: &ExprPool) -> ExprId {
    let terms: Vec<ExprId> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(i, c)| {
            let c = pool.integer(c.clone());
            match i {
                0 => c,
                1 => pool.mul(vec![c, r]),
                _ => pool.mul(vec![c, pool.pow(r, pool.integer(i as i64))]),
            }
        })
        .collect();
    pool.add(terms)
}

/// `f` divided by its content, with a positive leading coefficient.
fn primitive_positive(c: &[Integer]) -> Vec<Integer> {
    let mut g = Integer::from(0);
    for x in c {
        g.gcd_mut(x);
    }
    if g == 0 {
        return c.to_vec();
    }
    if c.last().is_some_and(|l| *l < 0) {
        g = -g;
    }
    c.iter().map(|x| Integer::from(x / &g)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn companion(c: &[i64], pool: &ExprPool) -> Matrix {
        // monic x^n + c_{n-1} x^{n-1} + … + c_0, c ascending without the 1.
        let n = c.len();
        let mut m = Matrix::zeros(n, n, pool);
        for i in 1..n {
            m.set(i, i - 1, pool.integer(1));
        }
        for i in 0..n {
            m.set(i, n - 1, pool.integer(-c[i]));
        }
        m
    }

    /// `A v = α v` checked in `K`, from the printed expressions.
    fn check_eigvec_exact(m: &Matrix, lambda: ExprId, v: &Matrix, pool: &ExprPool) {
        let (k, _) = AlgField::of_root_of(lambda, pool).unwrap();
        let a = rational_grid(m, pool).unwrap();
        let mut memo = HashMap::new();
        let vk: Vec<Nfe> = (0..v.rows)
            .map(|i| expr_to_element(v.get(i, 0), lambda, &k, pool, &mut memo).unwrap())
            .collect();
        assert!(vk.iter().any(|x| !x.is_zero()));
        let av = mat_vec(&mat_from_rational(&a, &k), &vk, &k).unwrap();
        for (x, y) in av.iter().zip(&vk) {
            assert_eq!(*x, y.mul(&k.alpha()).unwrap());
        }
    }

    #[test]
    fn companion_quartic_eigenvectors_are_exact_for_every_root() {
        if !crate::poly::root_of::root_isolation_available() {
            return;
        }
        let pool = ExprPool::new();
        // x⁴ + x + 1: irreducible, no radical shortcut → RootOf eigenvalues.
        let m = companion(&[1, 1, 0, 0], &pool);
        let coeffs: Vec<Integer> = [1, 1, 0, 0, 1].iter().map(|&c| Integer::from(c)).collect();
        for idx in 0..4 {
            let lam = make_root_of(&coeffs, idx, &pool);
            let vs = field_eigenvectors(&m, lam, &pool).unwrap();
            assert_eq!(vs.len(), 1);
            check_eigvec_exact(&m, lam, &vs[0], &pool);
        }
    }

    /// Cardano eigenvalues (not `RootOf`) used to stall symbolic elimination;
    /// the factor is now identified and the kernel computed in its field.
    #[test]
    fn cubic_cardano_eigenvectors_via_the_factor_field() {
        let pool = ExprPool::new();
        let m = companion(&[-1, -1, 0], &pool);
        let vecs = crate::matrix::eigen::eigenvectors(&m, &pool).unwrap();
        assert_eq!(vecs.len(), 3);
        let env = HashMap::new();
        let a = rational_grid(&m, &pool).unwrap();
        for (lam, mult, vs) in &vecs {
            assert_eq!((*mult, vs.len()), (1, 1));
            let l = crate::eval::eval_complex_f64(*lam, &pool, &env).unwrap();
            let v: Vec<_> = (0..3)
                .map(|i| crate::eval::eval_complex_f64(vs[0].get(i, 0), &pool, &env).unwrap())
                .collect();
            for i in 0..3 {
                let (mut re, mut im) = (0.0, 0.0);
                for j in 0..3 {
                    let c = a[i][j].to_f64();
                    re += c * v[j].re;
                    im += c * v[j].im;
                }
                re -= l.re * v[i].re - l.im * v[i].im;
                im -= l.re * v[i].im + l.im * v[i].re;
                assert!(re.hypot(im) < 1e-9, "{i}: {re} {im}");
            }
        }
    }

    #[test]
    fn zero_test_in_the_field() {
        let pool = ExprPool::new();
        let coeffs: Vec<Integer> = [-1, -1, 0, 1].iter().map(|&c| Integer::from(c)).collect();
        let r = make_root_of(&coeffs, 0, &pool);
        // r³ − r − 1 = 0
        let e = pool.add(vec![
            pool.pow(r, pool.integer(3)),
            pool.mul(vec![pool.integer(-1), r]),
            pool.integer(-1),
        ]);
        assert_eq!(root_of_rational_is_zero(e, &pool), Some(true));
        // r³ − r ≠ 0
        let e2 = pool.add(vec![
            pool.pow(r, pool.integer(3)),
            pool.mul(vec![pool.integer(-1), r]),
        ]);
        assert_eq!(root_of_rational_is_zero(e2, &pool), Some(false));
        // 1/r − (r² − 1) = 0 since r(r² − 1) = 1
        let e3 = pool.add(vec![
            pool.pow(r, pool.integer(-1)),
            pool.mul(vec![pool.integer(-1), pool.pow(r, pool.integer(2))]),
            pool.integer(1),
        ]);
        assert_eq!(root_of_rational_is_zero(e3, &pool), Some(true));
        // Another atom: not decided here.
        let x = pool.symbol("x", Domain::Real);
        assert_eq!(root_of_rational_is_zero(pool.add(vec![r, x]), &pool), None);
        // Two different conjugates: not decided here.
        let r1 = make_root_of(&coeffs, 1, &pool);
        assert_eq!(
            root_of_rational_is_zero(
                pool.add(vec![r, pool.mul(vec![pool.integer(-1), r1])]),
                &pool
            ),
            None
        );
    }

    #[test]
    fn jordan_chains_of_a_repeated_irreducible_factor() {
        // A = [[C, I], [0, C]] with C the companion of x³ − x − 1: χ = p², one
        // Jordan block of size 2 per root.
        let pool = ExprPool::new();
        let c = companion(&[-1, -1, 0], &pool);
        let n = 6;
        let mut a = vec![vec![Rational::from(0); n]; n];
        let cg = rational_grid(&c, &pool).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = cg[i][j].clone();
                a[i + 3][j + 3] = cg[i][j].clone();
            }
            a[i][i + 3] = Rational::from(1);
        }
        let coeffs: Vec<Integer> = [-1, -1, 0, 1].iter().map(|&c| Integer::from(c)).collect();
        let k = AlgField::from_coeffs(&coeffs).unwrap();
        let chains = jordan_chains(&a, &k, 2).unwrap();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].0, 2);
        let nm = shifted(&a, &k).unwrap();
        let (v1, v2) = (&chains[0].1[0], &chains[0].1[1]);
        // N v1 = 0, N v2 = v1, v1 ≠ 0.
        assert!(mat_vec(&nm, v1, &k).unwrap().iter().all(Nfe::is_zero));
        assert_eq!(&mat_vec(&nm, v2, &k).unwrap(), v1);
        assert!(v1.iter().any(|x| !x.is_zero()));
    }
}
