//! Hermite reduction on an algebraic curve — Risch milestone **M3 / P3**.
//!
//! Given an integrand `f ∈ ℚ(x)(y)` (`F(x,y)=0`), Hermite reduction writes
//! `∫ f dx = g + ∫ h dx` where `g ∈ ℚ(x)(y)` is the **algebraic part** and `h`
//! has only **simple poles** (a squarefree denominator over the curve — a
//! differential of the third kind).  `∫ h dx` is then the logarithmic part (MC).
//!
//! For a **simple radical** `yⁿ = a(x)` the integral basis `wᵢ = yⁱ/dᵢ`
//! diagonalizes the derivation: `wᵢ' = ωᵢ·wᵢ` with
//! `ωᵢ = i·a'/(n·a) − dᵢ'/dᵢ ∈ ℚ(x)`.  Hermite then **decouples** into `n`
//! independent *twisted* scalar Hermite reductions — for the operator
//! `L = d/dx + ωᵢ` — one per basis component (Bronstein, *Symbolic Integration
//! Tutorial* §3.2, eq 12).  The twist's pole at a branch point is handled
//! automatically by the `V·ωᵢ` term, so this is correct including at the
//! branch locus.
//!
//! [`hermite_reduce_general`] handles an **arbitrary** curve `F(x,y)=0` over the
//! van Hoeij integral basis.  On the **normal part** (factors coprime to the
//! discriminant, where `wᵢ'` is regular) the leading pole cancellation is the
//! fast componentwise scalar solve (mod `V`).  At a **branch-locus** factor
//! (`V | disc`, where the basis derivation has a pole at `V`) it uses the
//! `different`-aware **lazy** solve: a `ℚ`-linear system for `b`'s coordinates
//! mod `V` derived from `U·[(M−1)V'·b − V·D(b)] ≡ −A (mod V)`.  Repeated poles
//! at *either* locus are reduced to simple poles; a genuine simple pole at the
//! branch locus (the lazy system inconsistent) is left in `h`.
//!
//! Sound by construction: every result is accepted only after the exact field
//! identity `g' + h = f` is verified, and each `h` component is checked to have a
//! squarefree denominator (modulo the branch locus, where simple poles remain).

use rug::Rational;

use super::super::risch::alg_field::{AlgElem, AlgExtension, RatFn, RationalFunctionField};
use super::super::risch::number_field::{mod_inverse, CoeffField};
use super::super::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};
use super::super::risch::rational_rde::{poly_div_exact, poly_gcd};
use super::integral_basis::{discriminant, radical_integral_basis, squarefree_factors};
use super::vanhoeij::integral_basis;

/// Hermite reduction of `∫ f dx` on the curve `yⁿ = a(x)`.  Returns `(g, h)` with
/// `f = g' + h` and every component of `h` having a squarefree denominator
/// (simple poles).  `None` if the shape is unsupported or verification fails.
pub fn hermite_reduce_radical(
    n: usize,
    a: &QPoly,
    integrand: &AlgElem,
) -> Option<(AlgElem, AlgElem)> {
    if n < 2 {
        return None;
    }
    let f = RationalFunctionField;
    let basis = radical_integral_basis(n, a)?;
    let ext = AlgExtension::radical(n, a);

    // dᵢ = denominator of the basis element wᵢ = yⁱ/dᵢ.
    let d: Vec<QPoly> = (0..n)
        .map(|i| {
            basis[i]
                .get(i)
                .map(|c| c.denom().clone())
                .unwrap_or_else(|| vec![Rational::from(1)])
        })
        .collect();

    // Integrand in the w-basis: f = Σ fᵢ yⁱ = Σ (fᵢ·dᵢ) wᵢ  (diagonal basis change).
    let a_prime = poly_deriv(a);
    let mut g_w = vec![RatFn::int(0); n];
    let mut h_w = vec![RatFn::int(0); n];
    for i in 0..n {
        let fi = integrand.get(i).cloned().unwrap_or_else(|| RatFn::int(0));
        if fi.numer().is_empty() {
            continue;
        }
        let coord = f.mul(&fi, &RatFn::from_poly(&d[i])); // fᵢ·dᵢ
                                                          // ωᵢ = i·a'/(n·a) − dᵢ'/dᵢ.
        let omega = omega_i(i, n, a, &a_prime, &d[i]);
        let (gi, hi) = twisted_hermite(&coord, &omega)?;
        g_w[i] = gi;
        h_w[i] = hi;
    }

    // Back to the power basis: g = Σ gᵢ wᵢ = Σ (gᵢ/dᵢ) yⁱ.
    let to_power = |w: &[RatFn]| -> AlgElem {
        w.iter()
            .enumerate()
            .map(|(i, gi)| f.mul(gi, &RatFn::new(vec![Rational::from(1)], d[i].clone())))
            .collect()
    };
    let g = to_power(&g_w);
    let h = to_power(&h_w);

    // Soundness gate 1: every h component has a squarefree denominator.
    for hi in &h {
        let den = hi.denom();
        if degree(&poly_gcd(den, &poly_deriv(den))) > 0 {
            return None;
        }
    }
    // Soundness gate 2: g' + h == f exactly in the field.
    let lhs = ext.add(&ext.derivation(&g), &h);
    if !ext.elem_eq(&lhs, integrand) {
        return None;
    }
    Some((g, h))
}

/// Hermite reduction of `∫ f dx` on a **general** curve `F(x,y)=0` (not
/// necessarily a simple radical), over the van Hoeij integral basis.  Returns
/// `(g, h)` with `f = g' + h`, `g ∈ ℚ(x)(y)` the algebraic part and `h` a
/// differential of the **third kind** — every integral-basis coordinate of `h`
/// has a squarefree denominator on the **normal part** (the locus coprime to the
/// discriminant).  `None` if the basis is unavailable or a soundness gate fails.
///
/// Reduces the repeated factors `V` of the denominator that are **coprime to the
/// discriminant** (where the basis derivation is regular, so `wᵢ'` has no pole at
/// `V`).  There the leading order-`M` pole cancels **componentwise**: with
/// `D = U·V^M`, the integral element `b = Σ bᵢ wᵢ` solving
/// `bᵢ ≡ −Aᵢ·(U·(M−1)·V')⁻¹ (mod V)` makes `(b/V^{M−1})'` match the pole, so
/// `f − (b/V^{M−1})'` drops one power of `V`.  The basis-mixing `wᵢ' = Σ Mᵢⱼ wⱼ`
/// only perturbs lower orders and is handled by iteration.  Branch-locus repeated
/// poles (`V | disc`) are reduced by the lazy `lazy_solve_b` step; a genuine
/// simple pole there is left in `h`.
///
/// Sound by construction: accepted only after the exact field identity
/// `g' + h = f` holds and every `h`-coordinate denominator is squarefree away
/// from the branch locus.
pub fn hermite_reduce_general(
    f_coeffs: &[QPoly],
    integrand: &AlgElem,
) -> Option<(AlgElem, AlgElem)> {
    let ext = AlgExtension::new(f_coeffs);
    let n = ext.degree() as usize;
    if n < 2 {
        return None;
    }
    let basis = integral_basis(f_coeffs)?;
    let disc = discriminant(f_coeffs);

    let mut cur = pad(integrand, n);
    let mut g = ext.from_int(0);

    let denom_total = |e: &AlgElem| -> i64 {
        to_w_coords(&basis, e, n)
            .map(|cs| cs.iter().map(|c| degree(c.denom()).max(0)).sum())
            .unwrap_or(0)
    };
    let cap = 4 * (denom_total(&cur) as usize) + 8;

    for _ in 0..cap {
        let coords = to_w_coords(&basis, &cur, n)?;
        let (d_poly, a_polys) = common_denominator(&coords);
        let sqf = squarefree_factors(&d_poly);

        // Build the reduction term `b/V^{M−1}` for the highest-multiplicity
        // repeated factor `V` (mult `M ≥ 2`).  Prefer a **normal-part** factor
        // (coprime to the discriminant) — there the leading cancellation is the
        // fast componentwise solve `bᵢ ≡ −Aᵢ·(U(M−1)V')⁻¹ (mod V)`.  Otherwise
        // fall back to the **lazy** branch-locus solve (`V | disc`), which
        // accounts for the basis derivation's pole at `V`.
        let normal = sqf
            .iter()
            .enumerate()
            .rev()
            .find(|(k, p)| *k + 1 >= 2 && degree(p) >= 1 && degree(&poly_gcd(p, &disc)) <= 0)
            .map(|(k, p)| (p.clone(), k + 1));

        let term = if let Some((v, m)) = normal {
            let vm = poly_pow(&v, m as u32);
            let u = poly_div_exact(&d_poly, &vm);
            let s = poly_scale(
                &poly_mul(&u, &poly_deriv(&v)),
                &Rational::from((m - 1) as i64),
            );
            let s_inv = mod_inverse(&s, &v)?;
            let mut b_w = vec![RatFn::int(0); n];
            for (i, ai) in a_polys.iter().enumerate() {
                let bi = poly_mod(&poly_mul(ai, &s_inv), &v);
                b_w[i] = RatFn::from_poly(&poly_scale(&bi, &Rational::from(-1)));
            }
            let b_power = w_to_power(&basis, &b_w, &ext, n);
            let inv_vm1 = RatFn::new(vec![Rational::from(1)], poly_pow(&v, (m - 1) as u32));
            Some(scale_elem(&b_power, &inv_vm1))
        } else if let Some((v, m)) = sqf
            .iter()
            .enumerate()
            .rev()
            .find(|(k, p)| *k + 1 >= 2 && degree(p) >= 1)
            .map(|(k, p)| (p.clone(), k + 1))
        {
            // Branch-locus (`V | disc`): lazy coupled solve.
            let vm = poly_pow(&v, m as u32);
            let u = poly_div_exact(&d_poly, &vm);
            lazy_solve_b(&ext, &basis, &a_polys, &u, &v, m, n).map(|b_power| {
                let inv_vm1 = RatFn::new(vec![Rational::from(1)], poly_pow(&v, (m - 1) as u32));
                scale_elem(&b_power, &inv_vm1)
            })
        } else {
            None
        };

        let Some(term) = term else {
            break; // squarefree denominator (or lazy solve unavailable) → done
        };
        if ext.elem_eq(&term, &ext.from_int(0)) {
            break; // no progress possible at this place
        }
        let next = ext.sub(&cur, &ext.derivation(&term));
        g = ext.add(&g, &term);
        // Progress guard: the denominator must strictly drop.
        if denom_total(&next) >= denom_total(&cur) {
            cur = next;
            break;
        }
        cur = next;
    }

    let h = cur;
    // Gate 1: every h-coordinate denominator is squarefree on the normal part.
    let hcoords = to_w_coords(&basis, &h, n)?;
    for c in &hcoords {
        let den = c.denom();
        let g_sq = poly_gcd(den, &poly_deriv(den));
        // Allow repeated factors only at the discriminant (branch) locus — these
        // are left in place when the lazy solve cannot reduce them.
        if degree(&poly_gcd(&g_sq, &disc)) < degree(&g_sq) {
            return None;
        }
    }
    // Gate 2: g' + h = f exactly in the field.
    let lhs = ext.add(&ext.derivation(&g), &h);
    if !ext.elem_eq(&lhs, integrand) {
        return None;
    }
    Some((g, h))
}

/// Pad an `AlgElem` to length `n` with zero coordinates.
fn pad(e: &AlgElem, n: usize) -> AlgElem {
    let mut v = e.clone();
    while v.len() < n {
        v.push(RatFn::int(0));
    }
    v
}

/// Coordinates of `elem` in the integral basis: solve `Σ cᵢ·basisᵢ = elem` over
/// `ℚ(x)`.  `None` if the basis matrix is singular (should not happen).
fn to_w_coords(basis: &[AlgElem], elem: &AlgElem, n: usize) -> Option<Vec<RatFn>> {
    let f = RationalFunctionField;
    let comp = |e: &AlgElem, r: usize| e.get(r).cloned().unwrap_or_else(|| RatFn::int(0));
    // Augmented matrix: column i = basisᵢ (coeff of yʳ in row r), last column = elem.
    let mut m: Vec<Vec<RatFn>> = (0..n)
        .map(|r| {
            let mut row: Vec<RatFn> = (0..n).map(|i| comp(&basis[i], r)).collect();
            row.push(comp(elem, r));
            row
        })
        .collect();
    // Gaussian elimination over ℚ(x).  The pivot row equals `col` at every step.
    for col in 0..n {
        let sel = (col..n).find(|&r| !f.eq(&m[r][col], &f.zero()))?;
        m.swap(col, sel);
        let inv = f.inv(&m[col][col])?;
        for v in m[col].iter_mut() {
            *v = f.mul(v, &inv);
        }
        for r in 0..n {
            if r != col && !f.eq(&m[r][col], &f.zero()) {
                let factor = m[r][col].clone();
                // Two distinct rows (`m[r]` updated from `m[col]`): range loop.
                #[allow(clippy::needless_range_loop)]
                for c in 0..=n {
                    let sub = f.mul(&factor, &m[col][c].clone());
                    m[r][c] = f.sub(&m[r][c], &sub);
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n].clone()).collect())
}

/// `Σ coordsᵢ · basisᵢ` (integral basis → power basis).
fn w_to_power(basis: &[AlgElem], coords: &[RatFn], ext: &AlgExtension, n: usize) -> AlgElem {
    let mut acc = ext.from_int(0);
    for i in 0..n {
        acc = ext.add(&acc, &scale_elem(&basis[i], &coords[i]));
    }
    acc
}

/// Multiply every coordinate of `elem` by the scalar `s ∈ ℚ(x)`.
fn scale_elem(elem: &AlgElem, s: &RatFn) -> AlgElem {
    let f = RationalFunctionField;
    elem.iter().map(|c| f.mul(s, c)).collect()
}

/// Common denominator `D` of the coordinates and the integral numerators
/// `Aᵢ = numer(cᵢ)·(D/denom(cᵢ))`, so `cᵢ = Aᵢ/D`.
fn common_denominator(coords: &[RatFn]) -> (QPoly, Vec<QPoly>) {
    let mut d = vec![Rational::from(1)];
    for c in coords {
        d = poly_lcm(&d, c.denom());
    }
    let a = coords
        .iter()
        .map(|c| poly_mul(c.numer(), &poly_div_exact(&d, c.denom())))
        .collect();
    (d, a)
}

/// Least common multiple `a·b/gcd(a,b)` over `ℚ[x]`.
fn poly_lcm(a: &QPoly, b: &QPoly) -> QPoly {
    if degree(a) < 0 || degree(b) < 0 {
        return vec![Rational::from(1)];
    }
    poly_div_exact(&poly_mul(a, b), &poly_gcd(a, b))
}

/// **Lazy** Hermite solve at a branch-locus factor `V` (`V | disc`, squarefree,
/// multiplicity `M`).  Find `b = Σ bᵢ wᵢ` (`bᵢ ∈ ℚ[x]/V`) with
/// `U·[(M−1)V'·b − V·D(b)] ≡ −A (mod V)` in the integral basis, so that
/// `(b/V^{M−1})'` cancels the order-`M` pole even though the basis derivation
/// `D(wᵢ)` itself has a pole at `V`.  Returns `b` in the power basis, or `None`
/// if the (linear, over `ℚ`) system is inconsistent — i.e. the pole at `V` is a
/// genuine simple pole that cannot be reduced.
fn lazy_solve_b(
    ext: &AlgExtension,
    basis: &[AlgElem],
    a_polys: &[QPoly],
    u: &QPoly,
    v: &QPoly,
    m: usize,
    n: usize,
) -> Option<AlgElem> {
    let dv = degree(v) as usize;
    if dv < 1 {
        return None;
    }
    let nunk = n * dv;
    let mv1 = poly_scale(&poly_deriv(v), &Rational::from((m - 1) as i64)); // (M−1)V'
    let u_rf = RatFn::from_poly(u);
    let v_rf = RatFn::from_poly(v);
    let mv1_rf = RatFn::from_poly(&mv1);

    // Column `(i, p)` = the basis trial `b = xᵖ·wᵢ` mapped through
    // `T(b) = U·[(M−1)V'·b − V·D(b)]`, reduced to w-coords mod V and flattened.
    let mut cols: Vec<Vec<Rational>> = Vec::with_capacity(nunk);
    for i in 0..n {
        for p in 0..dv {
            let mut b_w = vec![RatFn::int(0); n];
            b_w[i] = RatFn::from_poly(&monomial_q(p));
            let b_power = w_to_power(basis, &b_w, ext, n);
            let db = ext.derivation(&b_power);
            let v_db = scale_elem(&db, &v_rf); // V·D(b)
            let mv1_b = scale_elem(&b_power, &mv1_rf); // (M−1)V'·b
            let inner = ext.sub(&mv1_b, &v_db);
            let t_elem = scale_elem(&inner, &u_rf); // U·[…]
            let tcoords = to_w_coords(basis, &t_elem, n)?;
            let mut col = Vec::with_capacity(nunk);
            for c in &tcoords {
                let r = ratfn_mod_v(c, v)?; // None ⇒ not regular at V ⇒ bail
                for k in 0..dv {
                    col.push(r.get(k).cloned().unwrap_or_else(|| Rational::from(0)));
                }
            }
            cols.push(col);
        }
    }

    // RHS = −A mod V (flattened over the n coordinates).
    let mut rhs = Vec::with_capacity(nunk);
    for j in 0..n {
        let aj = a_polys.get(j).cloned().unwrap_or_default();
        let r = poly_mod(&poly_scale(&aj, &Rational::from(-1)), v);
        for k in 0..dv {
            rhs.push(r.get(k).cloned().unwrap_or_else(|| Rational::from(0)));
        }
    }

    // Assemble `mat·x = rhs` (rows = nunk equations, cols = nunk unknowns).
    let mut mat = vec![vec![Rational::from(0); nunk]; nunk];
    for (unk, col) in cols.iter().enumerate() {
        for (eq, val) in col.iter().enumerate() {
            mat[eq][unk] = val.clone();
        }
    }
    let sol = gauss_solve_q(mat, rhs, nunk)?;

    let mut b_w = vec![RatFn::int(0); n];
    for (i, slot) in b_w.iter_mut().enumerate() {
        let poly: QPoly = (0..dv).map(|p| sol[i * dv + p].clone()).collect();
        *slot = RatFn::from_poly(&trim(poly));
    }
    Some(w_to_power(basis, &b_w, ext, n))
}

/// Reduce `r = num/den ∈ ℚ(x)` modulo `V` (requires `gcd(den, V) = 1`):
/// `num · den⁻¹ mod V`, as a polynomial of degree `< deg V`.  `None` if `den`
/// is not invertible mod `V` (i.e. `r` has a pole at `V`).
fn ratfn_mod_v(r: &RatFn, v: &QPoly) -> Option<QPoly> {
    let inv = mod_inverse(r.denom(), v)?;
    Some(poly_mod(&poly_mul(r.numer(), &inv), v))
}

/// The monomial `xᵖ` as a `QPoly`.
fn monomial_q(p: usize) -> QPoly {
    let mut v = vec![Rational::from(0); p + 1];
    v[p] = Rational::from(1);
    v
}

/// Solve `mat·x = rhs` over `ℚ` (square `n×n`); particular solution (free vars 0)
/// or `None` if inconsistent.
fn gauss_solve_q(
    mut mat: Vec<Vec<Rational>>,
    mut rhs: Vec<Rational>,
    n: usize,
) -> Option<Vec<Rational>> {
    let nrows = mat.len();
    let mut pivot_of_col = vec![None; n];
    let mut row = 0usize;
    for col in 0..n {
        if row >= nrows {
            break;
        }
        let Some(sel) = (row..nrows).find(|&r| mat[r][col] != 0) else {
            continue;
        };
        mat.swap(row, sel);
        rhs.swap(row, sel);
        let piv = mat[row][col].clone();
        for v in mat[row].iter_mut() {
            *v /= &piv;
        }
        rhs[row] /= &piv;
        let pr = mat[row].clone();
        let pb = rhs[row].clone();
        for r in 0..nrows {
            if r != row && mat[r][col] != 0 {
                let f = mat[r][col].clone();
                for (dst, pv) in mat[r].iter_mut().zip(pr.iter()) {
                    *dst -= f.clone() * pv;
                }
                rhs[r] -= f * &pb;
            }
        }
        pivot_of_col[col] = Some(row);
        row += 1;
    }
    for r in 0..nrows {
        if mat[r].iter().all(|v| *v == 0) && rhs[r] != 0 {
            return None; // inconsistent
        }
    }
    let mut x = vec![Rational::from(0); n];
    for (col, pr) in pivot_of_col.iter().enumerate() {
        if let Some(r) = pr {
            x[col] = rhs[*r].clone();
        }
    }
    Some(x)
}

/// `ωᵢ = i·a'/(n·a) − dᵢ'/dᵢ`, the basis-derivative coefficient `wᵢ' = ωᵢ wᵢ`.
fn omega_i(i: usize, n: usize, a: &QPoly, a_prime: &QPoly, di: &QPoly) -> RatFn {
    let f = RationalFunctionField;
    let scale = RatFn::new(
        vec![Rational::from(i as i64)],
        vec![Rational::from(n as i64)],
    );
    let log_a = RatFn::new(a_prime.clone(), a.clone()); // a'/a
    let term1 = f.mul(&scale, &log_a);
    if degree(di) < 1 {
        return term1; // dᵢ = 1 ⇒ dᵢ'/dᵢ = 0
    }
    let log_d = RatFn::new(poly_deriv(di), di.clone()); // dᵢ'/dᵢ
    f.add(&term1, &f.neg(&log_d))
}

/// Twisted scalar Hermite reduction for `L = d/dx + ω`: returns `(g, h)` with
/// `c = L(g) + h`, `h` having a squarefree denominator.
fn twisted_hermite(c: &RatFn, omega: &RatFn) -> Option<(RatFn, RatFn)> {
    let f = RationalFunctionField;
    let mut cur = c.clone();
    let mut g = RatFn::int(0);
    // Each step lowers one repeated factor's multiplicity by one.
    let cap = 4 * (degree(c.denom()).max(0) as usize) + 8;
    for _ in 0..cap {
        let den = cur.denom().clone();
        // Find the highest-multiplicity squarefree factor V (mult M ≥ 2).
        let sqf = squarefree_factors(&den);
        let Some((v, m)) = sqf
            .iter()
            .enumerate()
            .rev()
            .find(|(k, p)| *k + 1 >= 2 && degree(p) >= 1)
            .map(|(k, p)| (p.clone(), k + 1))
        else {
            break; // denominator squarefree → done
        };

        // B ≡ (A/U)·inv(ωV − (M−1)V') (mod V), with U = den / V^M, A = numer(cur).
        let vm = poly_pow(&v, m as u32);
        let u = poly_div_exact(&den, &vm);
        let num = cur.numer().clone();
        let au = poly_mod(&poly_mul(&num, &mod_inverse(&u, &v)?), &v); // A/U mod V

        // K = ωV − (M−1)V'  (a ℚ(x) element, regular at V), reduced mod V.
        let v_rf = RatFn::from_poly(&v);
        let vp_rf = RatFn::from_poly(&poly_scale(
            &poly_deriv(&v),
            &Rational::from((m - 1) as i64),
        ));
        let k_rf = f.add(&f.mul(omega, &v_rf), &f.neg(&vp_rf));
        let k_mod = reduce_mod_v(&k_rf, &v)?;
        let k_inv = mod_inverse(&k_mod, &v)?;

        let b = poly_mod(&poly_mul(&au, &k_inv), &v);
        if trim(b.clone()).is_empty() {
            break; // no reduction possible at this place
        }

        // g += B / V^{M−1};  cur -= L(B/V^{M−1}).
        let term = RatFn::new(b.clone(), poly_pow(&v, (m - 1) as u32));
        g = f.add(&g, &term);
        let l_term = f.add(&f.derivation(&term), &f.mul(omega, &term));
        let next = f.add(&cur, &f.neg(&l_term));
        // Guard against non-progress.
        if degree(next.denom()) >= degree(cur.denom()) && next != RatFn::int(0) {
            // The V-power should strictly drop; if not, stop to stay sound.
            cur = next;
            break;
        }
        cur = next;
    }
    Some((g, cur))
}

/// Reduce a `ℚ(x)` element `r = num/den` modulo `V` (requires `gcd(den, V) = 1`):
/// `num · den⁻¹ mod V`.
fn reduce_mod_v(r: &RatFn, v: &QPoly) -> Option<QPoly> {
    let inv = mod_inverse(r.denom(), v)?;
    Some(poly_mod(&poly_mul(r.numer(), &inv), v))
}

fn poly_mod(a: &QPoly, m: &QPoly) -> QPoly {
    // Remainder of a ÷ m over ℚ[x].
    let (_, rem) = poly_divrem(a, m);
    rem
}

/// `a = q·b + r`, `deg r < deg b`, over `ℚ[x]`.
fn poly_divrem(a: &QPoly, b: &QPoly) -> (QPoly, QPoly) {
    let b = trim(b.clone());
    let bd = degree(&b);
    let mut r = trim(a.clone());
    if bd < 0 {
        return (Vec::new(), r);
    }
    let lc = b[bd as usize].clone();
    let mut q = vec![Rational::from(0); (degree(&r) - bd + 1).max(0) as usize];
    while degree(&r) >= bd && !r.is_empty() {
        let rd = degree(&r);
        let shift = (rd - bd) as usize;
        let factor = r[rd as usize].clone() / &lc;
        if (shift as i64) < q.len() as i64 {
            q[shift] = factor.clone();
        }
        for (i, bc) in b.iter().enumerate() {
            r[shift + i] -= factor.clone() * bc;
        }
        r = trim(r);
    }
    (trim(q), r)
}

fn poly_pow(p: &QPoly, e: u32) -> QPoly {
    let mut acc = vec![Rational::from(1)];
    for _ in 0..e {
        acc = poly_mul(&acc, p);
    }
    acc
}

fn poly_scale(p: &QPoly, s: &Rational) -> QPoly {
    p.iter().map(|c| c.clone() * s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }
    fn rf(num: &[i64], den: &[i64]) -> RatFn {
        RatFn::new(qp(num), qp(den))
    }

    /// ∫ y/x³ dx on y² = x : fully algebraic, g = −⅔ y/x², h = 0.
    #[test]
    fn sqrt_double_pole_fully_reduces() {
        // integrand y/x³ = AlgElem [0, 1/x³].
        let integrand = vec![RatFn::int(0), rf(&[1], &[0, 0, 0, 1])];
        let (g, h) = hermite_reduce_radical(2, &qp(&[0, 1]), &integrand).expect("reduce");
        // h = 0.
        assert!(h.iter().all(|c| c.numer().is_empty()));
        // g = −⅔ y/x²  ⇒  component 1 = −2/3 / x².
        assert_eq!(g[1], RatFn::new(qp(&[-2]), qp(&[0, 0, 3])));
    }

    /// ∫ y/((x−1)·x) dx on y²=x : already simple poles ⇒ g = 0, h = f.
    #[test]
    fn sqrt_simple_pole_untouched() {
        // y/((x-1)x) = AlgElem [0, 1/((x-1)x)] ; (x-1)x = x²−x.
        let integrand = vec![RatFn::int(0), rf(&[1], &[0, -1, 1])];
        let (g, h) = hermite_reduce_radical(2, &qp(&[0, 1]), &integrand).expect("reduce");
        assert!(g.iter().all(|c| c.numer().is_empty())); // g = 0
        assert_eq!(h[1], rf(&[1], &[0, -1, 1])); // h = f
    }

    /// Mixed: ∫ y/(x²(x−1)) dx on y²=x reduces the x² pole, leaving simple poles.
    /// Verified by the exact `g' + h = f` gate inside the reducer.
    #[test]
    fn sqrt_mixed_reduction() {
        // x²(x-1) = x³ − x².
        let integrand = vec![RatFn::int(0), rf(&[1], &[0, 0, -1, 1])];
        let (g, h) = hermite_reduce_radical(2, &qp(&[0, 1]), &integrand).expect("reduce");
        // h has squarefree denominator (gate) and the algebraic part is nontrivial.
        assert!(!g.iter().all(|c| c.numer().is_empty()));
        for hi in &h {
            let den = hi.denom();
            assert!(degree(&poly_gcd(den, &poly_deriv(den))) <= 0);
        }
    }

    /// General (non-radical) curve `y² + y − x³ = 0` (a `y`-term ⇒ non-radical;
    /// disc = 1+4x³ squarefree ⇒ nonsingular, basis {1, y}).  Reduce an **exact
    /// derivative**: take `g₀ = y/(x−1)` (a simple pole off the branch locus),
    /// `f = g₀'` has a double pole; Hermite must recover `h = 0` with `g' = f`.
    #[test]
    fn general_curve_exact_derivative_reduces_to_zero() {
        // F = y² + y − x³  ⇒  coeffs [ -x³, 1, 1 ].
        let f_coeffs = [qp(&[0, 0, 0, -1]), qp(&[1]), qp(&[1])];
        let ext = AlgExtension::new(&f_coeffs);
        let g0 = vec![RatFn::int(0), rf(&[1], &[-1, 1])]; // y/(x−1)
        let f = ext.derivation(&g0);
        let (g, h) = hermite_reduce_general(&f_coeffs, &f).expect("reduce");
        // Exact derivative ⇒ no third-kind remainder.
        assert!(h.iter().all(|c| c.numer().is_empty()), "h = {h:?}");
        // g' = f.
        assert!(ext.elem_eq(&ext.derivation(&g), &f));
    }

    /// General curve, a double pole that is **not** an exact derivative: the
    /// reduction still lowers the pole and the `g' + h = f` gate holds with `h`
    /// having a squarefree (normal-part) denominator.
    #[test]
    fn general_curve_double_pole_reduces() {
        let f_coeffs = [qp(&[0, 0, 0, -1]), qp(&[1]), qp(&[1])]; // y²+y−x³
        let ext = AlgExtension::new(&f_coeffs);
        // f = y/(x−1)²  = AlgElem [0, 1/(x−1)²];  (x−1)² = x²−2x+1.
        let f = vec![RatFn::int(0), rf(&[1], &[1, -2, 1])];
        let (g, h) = hermite_reduce_general(&f_coeffs, &f).expect("reduce");
        // Nontrivial algebraic part extracted.
        assert!(!g.iter().all(|c| c.numer().is_empty()));
        // g' + h = f exactly.
        assert!(ext.elem_eq(&ext.add(&ext.derivation(&g), &h), &f));
        // h denominators squarefree (x=1 is off the branch locus disc=1+4x³).
        for hi in &h {
            let den = hi.denom();
            assert!(degree(&poly_gcd(den, &poly_deriv(den))) <= 0);
        }
    }

    /// The general reducer handles a higher-degree radical curve at an
    /// **off-branch** pole: y³ = x (disc ∝ x², branch at 0), integrand y/(x−1)²
    /// (pole at x=1, normal).  The pole is lowered and the `g'+h=f` gate holds
    /// with squarefree `h` — exercising the degree-3 basis derivation mixing.
    #[test]
    fn general_off_branch_pole_cubic_radical() {
        let f_coeffs = [qp(&[0, -1]), qp(&[]), qp(&[]), qp(&[1])]; // y³ − x
        let ext = AlgExtension::new(&f_coeffs);
        let f = vec![RatFn::int(0), rf(&[1], &[1, -2, 1])]; // y/(x−1)²
        let (g, h) = hermite_reduce_general(&f_coeffs, &f).expect("reduce");
        assert!(!g.iter().all(|c| c.numer().is_empty()));
        assert!(ext.elem_eq(&ext.add(&ext.derivation(&g), &h), &f));
        for hi in &h {
            let den = hi.denom();
            assert!(degree(&poly_gcd(den, &poly_deriv(den))) <= 0);
        }
    }

    /// Lazy (branch-locus) Hermite: the **general** reducer now reduces a pole
    /// at the branch point `x=0` (`x | disc`), matching the radical reducer.
    /// ∫ y/x³ on y²=x ⇒ g = −⅔ y/x², h = 0.
    #[test]
    fn general_branch_locus_pole_reduces() {
        let f_coeffs = [qp(&[0, -1]), qp(&[]), qp(&[1])]; // y² − x
        let integrand = vec![RatFn::int(0), rf(&[1], &[0, 0, 0, 1])]; // y/x³
        let (g, h) = hermite_reduce_general(&f_coeffs, &integrand).expect("reduce");
        assert!(h.iter().all(|c| c.numer().is_empty()), "h = {h:?}");
        assert_eq!(g[1], RatFn::new(qp(&[-2]), qp(&[0, 0, 3]))); // −⅔ y/x²
    }

    /// Lazy Hermite on a degree-3 radical at the branch locus: ∫ y²/x⁴ on y³=x
    /// (disc ∝ x²) ⇒ g = −3/7 y²/x³, h = 0, via the general reducer.
    #[test]
    fn general_branch_locus_cubic_radical() {
        let f_coeffs = [qp(&[0, -1]), qp(&[]), qp(&[]), qp(&[1])]; // y³ − x
        let integrand = vec![RatFn::int(0), RatFn::int(0), rf(&[1], &[0, 0, 0, 0, 1])]; // y²/x⁴
        let (g, h) = hermite_reduce_general(&f_coeffs, &integrand).expect("reduce");
        assert!(h.iter().all(|c| c.numer().is_empty()), "h = {h:?}");
        assert_eq!(g[2], RatFn::new(qp(&[-3]), qp(&[0, 0, 0, 7]))); // −3/7 y²/x³
    }

    /// A genuine **simple pole** at the branch locus is left in `h` (the lazy
    /// solve is inconsistent — nothing to reduce): ∫ y/((x−1)x) on y²=x, the
    /// branch-point part stays, `g'+h=f` holds.
    #[test]
    fn general_branch_simple_pole_untouched() {
        let f_coeffs = [qp(&[0, -1]), qp(&[]), qp(&[1])]; // y² − x
        let ext = AlgExtension::new(&f_coeffs);
        let f = vec![RatFn::int(0), rf(&[1], &[0, -1, 1])]; // y/((x−1)x)
        let (g, h) = hermite_reduce_general(&f_coeffs, &f).expect("reduce");
        assert!(ext.elem_eq(&ext.add(&ext.derivation(&g), &h), &f));
    }

    /// Degree-3 radical: ∫ y²/x⁴ dx on y³ = x reduces the x⁴ pole.
    #[test]
    fn cbrt_reduction() {
        // y²/x⁴ = AlgElem [0, 0, 1/x⁴].
        let integrand = vec![RatFn::int(0), RatFn::int(0), rf(&[1], &[0, 0, 0, 0, 1])];
        let (g, h) = hermite_reduce_radical(3, &qp(&[0, 1]), &integrand).expect("reduce");
        // y²/x⁴ = x^{2/3}/x⁴ = x^{2/3-4} ; ∫ = x^{2/3-3}/(2/3-3) = (-3/7) x^{-7/3}.
        // x^{-7/3} = y²/x³ ⇒ g component 2 = (-3/7)/x³, h = 0.
        assert!(h.iter().all(|c| c.numer().is_empty()));
        assert_eq!(g[2], RatFn::new(qp(&[-3]), qp(&[0, 0, 0, 7])));
    }
}
