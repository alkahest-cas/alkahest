//! The radical extension `F[y]/(yⁿ − a)` over a generic differential field
//! `F: DifferentialField`, itself made a [`DifferentialField`] — the Risch
//! **M4 PR3** core.
//!
//! This is what makes the [`DifferentialField`] recursion genuinely
//! *tower-recursive*: a level may be an **algebraic (radical) extension of any
//! lower differential field**, and its Risch DE solver descends into the lower
//! field's solver.  PR2's `integrate_exp_times_radical` was a hand-rolled,
//! *diagonal-twist* specialization of exactly this algorithm; PR3 lifts it into
//! one generic place and PR2 now routes through it.
//!
//! ## Representation
//!
//! An element of the extension is the coefficient vector `[c₀, …, c_{n−1}]` of
//! the power basis `1, y, …, y^{n−1}` with each `cᵢ ∈ F` (the lower field).  The
//! defining relation is `yⁿ = a` (the *radical*/*diagonal* case), where the
//! radicand `a ∈ F`.  Arithmetic is the usual polynomial arithmetic reduced mod
//! `yⁿ − a`.
//!
//! ## Derivation
//!
//! The derivation extends `F`'s by the **diagonal twist**
//!
//! ```text
//!   D(y) = (1/n)·(D(a)/a)·y      ⇒      D(yⁱ) = (i/n)·(D(a)/a)·yⁱ,
//! ```
//!
//! so for `u = Σᵢ cᵢ yⁱ`,
//!
//! ```text
//!   D(u) = Σᵢ ( D(cᵢ) + cᵢ·(i/n)·D(a)/a ) yⁱ.
//! ```
//!
//! No reduction mod `yⁿ = a` is needed for the derivation since every `yⁱ`
//! stays at degree `i < n`.
//!
//! ## `rational_rde` — the `f ∈ base` restriction
//!
//! [`rational_rde`](DifferentialField::rational_rde) solves `D(u) + f·u = g`.
//! For the radical case the twist is **diagonal**, so when `f ∈ F` (i.e. `f` is
//! a base scalar: all its higher `y`-components vanish) multiplication by `f`
//! preserves the power basis and the system **decouples per component** into
//!
//! ```text
//!   D(uᵢ) + ( f₀ + (i/n)·D(a)/a )·uᵢ = gᵢ      over F,
//! ```
//!
//! each solved by `base.rational_rde(…)` — the M4 descent.  This diagonal
//! fast-path is kept unchanged (it is correct and cheaper) and covers PR2's
//! use, where the per-component twist `ω = η' + (i/n)D(a)/a` is itself a base
//! scalar plus the diagonal radical twist.
//!
//! ### Non-diagonal `f` — the coupled case
//!
//! If `f` has any nonzero higher `y`-component, multiplication-by-`f` *mixes*
//! components and the system is **not** diagonal: it becomes a genuinely
//! coupled linear system `b' + M·b = c` over `F`.  Rather than always
//! declining, this impl now delegates to the base field's
//! [`coupled_radical_rde`](DifferentialField::coupled_radical_rde) hook (gather
//! `f`, `g` as length-`n` component vectors over `F`, call the hook, then
//! re-verify `D(u)+f·u=g` exactly in-field before accepting).  The hook's
//! declining default means a base field with no coupled solver still declines
//! (`None`).
//!
//! The tractable, proven slice is the **radical-over-`ℚ(x)`** base case:
//! [`RationalDiffField`](super::diff_field::RationalDiffField)'s
//! `coupled_radical_rde` bridges to an [`AlgExtension`](super::alg_field::AlgExtension)
//! and runs the coupled solver `solve_alg_rde_general` (whose derivation matches
//! `RadicalExt`'s diagonal twist, so the bridge is sound; it is verification-
//! gated regardless).  Tower bases (`ExpTowerField`/`LogTowerField`) keep the
//! declining default — the fully-generic coupled solve over an arbitrary
//! transcendental tower remains future work.
//!
//! `limited_integrate` / `param_log_deriv` remain unimplemented (`None`).

use super::diff_field::DifferentialField;

/// A dense, ascending-degree polynomial over the base field `F` (used internally
/// by [`RadicalExt::inv`] for the extended-Euclid inversion against `yⁿ − a`).
type FPoly<F> = Vec<<F as DifferentialField>::Elem>;

/// The radical extension `F[y]/(yⁿ − a)` over a lower differential field `F`,
/// made a [`DifferentialField`] in its own right.
///
/// Elements are coefficient vectors over the power basis `1, y, …, y^{n−1}`;
/// see the [module docs](self).
#[derive(Clone, Debug)]
pub struct RadicalExt<F: DifferentialField> {
    /// The lower differential field `F`.
    base: F,
    /// The radicand `a ∈ F` (`yⁿ = a`).
    radicand_a: F::Elem,
    /// The radical degree `n ≥ 1`.
    n: usize,
}

impl<F: DifferentialField> RadicalExt<F> {
    /// Build `F[y]/(yⁿ − a)`.  `n` must be `≥ 1`.
    pub fn new(base: F, radicand_a: F::Elem, n: usize) -> Self {
        assert!(n >= 1, "RadicalExt: degree n must be ≥ 1");
        Self {
            base,
            radicand_a,
            n,
        }
    }

    /// The lower field `F`.
    pub fn base(&self) -> &F {
        &self.base
    }

    /// The radical degree `n`.
    pub fn degree(&self) -> usize {
        self.n
    }

    /// The radicand `a`.
    pub fn radicand(&self) -> &F::Elem {
        &self.radicand_a
    }

    /// `D(a)/a` in the lower field — the logarithmic derivative of the radicand,
    /// the building block of the diagonal twist.  `None` if `a` is zero.
    fn log_deriv_a(&self) -> Option<F::Elem> {
        let da = self.base.derivation(&self.radicand_a);
        let inv_a = self.base.inv(&self.radicand_a)?;
        Some(self.base.mul(&da, &inv_a))
    }

    /// The base scalar `m/n ∈ ℚ ⊂ F` (used to scale the twist for component
    /// `i = m`).  Built from `F::one` via repeated addition and one inversion,
    /// since [`DifferentialField`] has no `from_i64`.
    fn base_scalar_ratio(&self, m: usize) -> F::Elem {
        let num = self.base_int(m as i64);
        let den = self.base_int(self.n as i64);
        // n ≥ 1, so den is invertible.
        let den_inv = self
            .base
            .inv(&den)
            .expect("n ≥ 1 ⇒ nonzero base integer is invertible");
        self.base.mul(&num, &den_inv)
    }

    /// Embed an integer into `F` via repeated addition.
    fn base_int(&self, m: i64) -> F::Elem {
        let one = self.base.one();
        let mut acc = self.base.zero();
        for _ in 0..m.unsigned_abs() {
            acc = self.base.add(&acc, &one);
        }
        if m < 0 {
            self.base.neg(&acc)
        } else {
            acc
        }
    }

    /// Trim trailing zero components (canonicalization helper).
    fn trim(&self, mut v: Vec<F::Elem>) -> Vec<F::Elem> {
        while v.last().is_some_and(|c| self.base.is_zero(c)) {
            v.pop();
        }
        v
    }

    /// Reduce an arbitrary `F`-polynomial in `y` (possibly degree ≥ n) modulo
    /// `yⁿ = a` into a canonical element (length ≤ n).
    fn reduce(&self, v: &[F::Elem]) -> Vec<F::Elem> {
        if v.len() <= self.n {
            return self.trim(v.to_vec());
        }
        let mut v = v.to_vec();
        // Fold high powers down: y^k = a · y^{k-n}.
        for k in (self.n..v.len()).rev() {
            let c = v[k].clone();
            if self.base.is_zero(&c) {
                continue;
            }
            let folded = self.base.mul(&c, &self.radicand_a);
            let lower = k - self.n;
            v[lower] = self.base.add(&v[lower], &folded);
            v[k] = self.base.zero();
        }
        v.truncate(self.n);
        self.trim(v)
    }

    /// Coupled (non-diagonal) `D(u) + f·u = g` solve for a non-base `f`, via the
    /// base field's [`coupled_radical_rde`](DifferentialField::coupled_radical_rde)
    /// hook.  Gathers `f` and `g` as length-`n` component vectors over `F`, calls
    /// the hook, and — like the diagonal fast-path — **verifies `D(u) + f·u = g`
    /// in-field** before returning.  Returns `None` if the base field has no
    /// coupled solver (the default), the hook finds no solution, or verification
    /// fails; so a `Some` is always correct.
    fn coupled_nondiagonal_rde(&self, f: &[F::Elem], g: &[F::Elem]) -> Option<Vec<F::Elem>> {
        // Component vectors of length n over F (pad with base zeros).
        let comps = |v: &[F::Elem]| -> Vec<F::Elem> {
            let mut out = vec![self.base.zero(); self.n];
            for (i, c) in v.iter().take(self.n).enumerate() {
                out[i] = c.clone();
            }
            out
        };
        let f_comps = comps(f);
        let g_comps = comps(g);

        let u = self
            .base
            .coupled_radical_rde(self.n, &self.radicand_a, &f_comps, &g_comps)?;
        let u = self.trim(u);

        // In-field verification: D(u) + f·u = g (mirrors the diagonal path).
        let f_elem = f.to_vec();
        let g_elem = self.trim(g.to_vec());
        let lhs = self.add(&self.derivation(&u), &self.mul(&f_elem, &u));
        if self.eq(&lhs, &g_elem) {
            Some(u)
        } else {
            None
        }
    }
}

impl<F: DifferentialField> DifferentialField for RadicalExt<F> {
    /// Coefficient vector `[c₀, …, c_{n−1}]` over the power basis `1,…,y^{n−1}`.
    type Elem = Vec<F::Elem>;

    fn zero(&self) -> Self::Elem {
        Vec::new()
    }

    fn one(&self) -> Self::Elem {
        vec![self.base.one()]
    }

    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        let len = a.len().max(b.len());
        let mut r = Vec::with_capacity(len);
        for i in 0..len {
            let ai = a.get(i).cloned().unwrap_or_else(|| self.base.zero());
            let bi = b.get(i).cloned().unwrap_or_else(|| self.base.zero());
            r.push(self.base.add(&ai, &bi));
        }
        self.trim(r)
    }

    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        let len = a.len().max(b.len());
        let mut r = Vec::with_capacity(len);
        for i in 0..len {
            let ai = a.get(i).cloned().unwrap_or_else(|| self.base.zero());
            let bi = b.get(i).cloned().unwrap_or_else(|| self.base.zero());
            r.push(self.base.sub(&ai, &bi));
        }
        self.trim(r)
    }

    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut raw = vec![self.base.zero(); a.len() + b.len() - 1];
        for (i, ca) in a.iter().enumerate() {
            if self.base.is_zero(ca) {
                continue;
            }
            for (j, cb) in b.iter().enumerate() {
                let p = self.base.mul(ca, cb);
                raw[i + j] = self.base.add(&raw[i + j], &p);
            }
        }
        self.reduce(&raw)
    }

    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        self.trim(a.iter().map(|c| self.base.neg(c)).collect())
    }

    /// `a⁻¹` via the extended Euclidean algorithm in `F[y]` against the modulus
    /// `yⁿ − a`.  `None` if `a` is zero or a zero divisor.
    fn inv(&self, a: &Self::Elem) -> Option<Self::Elem> {
        let a = self.trim(a.to_vec());
        if a.is_empty() {
            return None;
        }
        // Modulus m(y) = yⁿ − a (length n+1).
        let mut modulus = vec![self.base.zero(); self.n + 1];
        modulus[0] = self.base.neg(&self.radicand_a);
        modulus[self.n] = self.base.one();
        let (g, s, _t) = self.fpoly_ext_gcd(&a, &modulus);
        // gcd must be a (nonzero) unit in F (degree 0) for an inverse to exist.
        if g.len() != 1 || self.base.is_zero(&g[0]) {
            return None;
        }
        let g_inv = self.base.inv(&g[0])?;
        let s = self.fpoly_scale(&s, &g_inv);
        Some(self.reduce(&s))
    }

    fn is_zero(&self, a: &Self::Elem) -> bool {
        self.trim(a.to_vec()).is_empty()
    }

    fn eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool {
        let a = self.trim(a.to_vec());
        let b = self.trim(b.to_vec());
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| self.base.eq(x, y))
    }

    /// `D(Σᵢ cᵢ yⁱ) = Σᵢ ( D(cᵢ) + cᵢ·(i/n)·D(a)/a ) yⁱ` — the diagonal twist.
    fn derivation(&self, a: &Self::Elem) -> Self::Elem {
        if a.is_empty() {
            return Vec::new();
        }
        let lda = match self.log_deriv_a() {
            Some(v) => v,
            None => {
                // a = 0: y is undefined; fall back to the base derivation of the
                // constant term only.
                let mut out = vec![self.base.zero(); a.len()];
                if let Some(c0) = a.first() {
                    out[0] = self.base.derivation(c0);
                }
                return self.trim(out);
            }
        };
        let mut out = Vec::with_capacity(a.len());
        for (i, ci) in a.iter().enumerate() {
            let dci = self.base.derivation(ci);
            if i == 0 {
                out.push(dci);
            } else {
                let scale = self.base_scalar_ratio(i); // i/n
                let twist = self.base.mul(ci, &self.base.mul(&scale, &lda));
                out.push(self.base.add(&dci, &twist));
            }
        }
        self.trim(out)
    }

    /// Solve `D(u) + f·u = g` over the radical extension.
    ///
    /// **Diagonal case (`f ∈ F`, only the `1`-component nonzero):** the system
    /// decouples per `y`-power into `D(uᵢ) + (f₀ + (i/n)·D(a)/a)·uᵢ = gᵢ` over
    /// `F`, each solved by the lower field's
    /// [`rational_rde`](DifferentialField::rational_rde).
    ///
    /// **Non-diagonal case (`f` carries higher `y`-powers):** the multiplication
    /// is coupled; this delegates to the base field's
    /// [`coupled_radical_rde`](DifferentialField::coupled_radical_rde) hook (the
    /// `ℚ(x)` impl bridges to `solve_alg_rde_general` over an `AlgExtension`;
    /// tower bases keep the declining default and so still return `None`).
    ///
    /// In both cases the assembled candidate is **verified in-field**
    /// (`D(u) + f·u = g`) before being returned, mirroring PR1's verification
    /// discipline; so a `Some` is always correct and `None` means
    /// declined/not-found.
    fn rational_rde(&self, f: &Self::Elem, g: &Self::Elem) -> Option<Self::Elem> {
        // f ∈ base: every higher component must be zero.
        let f_trim = self.trim(f.to_vec());
        if f_trim.len() > 1 {
            // Non-diagonal: multiplication-by-`f` mixes the power basis into a
            // genuinely coupled system.  Defer to the base field's coupled-radical
            // solver (the `ℚ(x)` impl bridges to `solve_alg_rde_general` over an
            // `AlgExtension`; tower bases keep the declining default).  Any
            // returned candidate is re-verified in-field before being accepted.
            return self.coupled_nondiagonal_rde(f, g);
        }
        let f0 = f_trim
            .into_iter()
            .next()
            .unwrap_or_else(|| self.base.zero());

        let lda = self.log_deriv_a()?; // D(a)/a; needs a ≠ 0

        let g = self.trim(g.to_vec());
        let mut u = vec![self.base.zero(); g.len()];
        for (i, gi) in g.iter().enumerate() {
            if self.base.is_zero(gi) {
                continue;
            }
            // ωᵢ = f₀ + (i/n)·D(a)/a.
            let omega = if i == 0 {
                f0.clone()
            } else {
                let scale = self.base_scalar_ratio(i);
                let twist = self.base.mul(&scale, &lda);
                self.base.add(&f0, &twist)
            };
            let ui = self.base.rational_rde(&omega, gi)?; // M4 descent
            u[i] = ui;
        }
        let u = self.trim(u);

        // In-field verification: D(u) + f·u = g.
        let lhs = self.add(&self.derivation(&u), &self.mul(f, &u));
        if self.eq(&lhs, &g) {
            Some(u)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Small F[y] (dense, ascending) polynomial helpers for `inv`.
// These operate on *unreduced* polynomials over the base field F.
// ---------------------------------------------------------------------------

impl<F: DifferentialField> RadicalExt<F> {
    fn fpoly_trim(&self, mut p: Vec<F::Elem>) -> Vec<F::Elem> {
        while p.last().is_some_and(|c| self.base.is_zero(c)) {
            p.pop();
        }
        p
    }

    fn fpoly_degree(&self, p: &[F::Elem]) -> i64 {
        let mut d = p.len() as i64 - 1;
        while d >= 0 && self.base.is_zero(&p[d as usize]) {
            d -= 1;
        }
        d
    }

    fn fpoly_scale(&self, p: &[F::Elem], s: &F::Elem) -> Vec<F::Elem> {
        if self.base.is_zero(s) {
            return Vec::new();
        }
        self.fpoly_trim(p.iter().map(|c| self.base.mul(c, s)).collect())
    }

    fn fpoly_sub(&self, a: &[F::Elem], b: &[F::Elem]) -> Vec<F::Elem> {
        let n = a.len().max(b.len());
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let ai = a.get(i).cloned().unwrap_or_else(|| self.base.zero());
            let bi = b.get(i).cloned().unwrap_or_else(|| self.base.zero());
            r.push(self.base.sub(&ai, &bi));
        }
        self.fpoly_trim(r)
    }

    fn fpoly_mul(&self, a: &[F::Elem], b: &[F::Elem]) -> Vec<F::Elem> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut r = vec![self.base.zero(); a.len() + b.len() - 1];
        for (i, ca) in a.iter().enumerate() {
            if self.base.is_zero(ca) {
                continue;
            }
            for (j, cb) in b.iter().enumerate() {
                let p = self.base.mul(ca, cb);
                r[i + j] = self.base.add(&r[i + j], &p);
            }
        }
        self.fpoly_trim(r)
    }

    /// Long division `a = q·b + r`, `deg r < deg b`, over `F[y]`.
    fn fpoly_divrem(&self, a: &[F::Elem], b: &[F::Elem]) -> (Vec<F::Elem>, Vec<F::Elem>) {
        let b = self.fpoly_trim(b.to_vec());
        let bd = self.fpoly_degree(&b);
        debug_assert!(bd >= 0, "division by zero polynomial");
        let lc_inv = self
            .base
            .inv(&b[bd as usize])
            .expect("nonzero leading coefficient of a field element is invertible");
        let mut r = self.fpoly_trim(a.to_vec());
        let ad = self.fpoly_degree(&r);
        if ad < bd {
            return (Vec::new(), r);
        }
        let mut q = vec![self.base.zero(); (ad - bd + 1) as usize];
        loop {
            let rd = self.fpoly_degree(&r);
            if rd < bd {
                break;
            }
            let shift = (rd - bd) as usize;
            let factor = self.base.mul(&r[rd as usize], &lc_inv);
            q[shift] = self.base.add(&q[shift], &factor);
            for (i, bc) in b.iter().enumerate() {
                let prod = self.base.mul(&factor, bc);
                r[shift + i] = self.base.sub(&r[shift + i], &prod);
            }
            r = self.fpoly_trim(r);
            if r.is_empty() {
                break;
            }
        }
        (self.fpoly_trim(q), r)
    }

    /// Extended GCD over `F[y]`: returns `(g, s, t)` with `s·a + t·b = g`.
    fn fpoly_ext_gcd(&self, a: &[F::Elem], b: &[F::Elem]) -> (FPoly<F>, FPoly<F>, FPoly<F>) {
        let (mut old_r, mut r) = (self.fpoly_trim(a.to_vec()), self.fpoly_trim(b.to_vec()));
        let one = vec![self.base.one()];
        let (mut old_s, mut s) = (one.clone(), Vec::new());
        let (mut old_t, mut t) = (Vec::new(), one);
        while !r.is_empty() {
            let (q, rem) = self.fpoly_divrem(&old_r, &r);
            old_r = r;
            r = rem;
            let ns = self.fpoly_sub(&old_s, &self.fpoly_mul(&q, &s));
            old_s = s;
            s = ns;
            let nt = self.fpoly_sub(&old_t, &self.fpoly_mul(&q, &t));
            old_t = t;
            t = nt;
        }
        (old_r, old_s, old_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::alg_field::RatFn;
    use crate::integrate::risch::diff_field::RationalDiffField;
    use crate::integrate::risch::poly_rde::QPoly;
    use crate::integrate::risch::tower_field::{LogTowerField, TExpr};
    use rug::Rational;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    fn rf_poly(c: &[i64]) -> RatFn {
        let p: QPoly = c.iter().map(|&n| rat(n)).collect();
        RatFn::from_poly(&p)
    }

    // ---- Arithmetic & derivation over ℚ(x): y = √x  (n = 2, a = x) ----

    /// `RadicalExt<RationalDiffField>` with `y = √x`.
    fn sqrt_x() -> RadicalExt<RationalDiffField> {
        RadicalExt::new(RationalDiffField::new(), rf_poly(&[0, 1]), 2)
    }

    #[test]
    fn arithmetic_mod_y2_eq_x() {
        let ext = sqrt_x();
        // y · y = x.
        let y = vec![ext.base().zero(), ext.base().one()];
        let yy = ext.mul(&y, &y);
        assert!(
            ext.eq(&yy, &vec![rf_poly(&[0, 1])]),
            "y² should reduce to x; got {yy:?}"
        );
        // (1 + y)² = 1 + 2y + x  = (1 + x) + 2y.
        let one_plus_y = ext.add(&ext.one(), &y);
        let sq = ext.mul(&one_plus_y, &one_plus_y);
        let expected = vec![rf_poly(&[1, 1]), rf_poly(&[2])]; // (1+x) + 2y
        assert!(ext.eq(&sq, &expected), "got {sq:?}");
    }

    #[test]
    fn inverse_of_y_is_y_over_x() {
        let ext = sqrt_x();
        let y = vec![ext.base().zero(), ext.base().one()];
        let inv = ext.inv(&y).expect("y invertible");
        // y · y⁻¹ = 1.
        assert!(ext.eq(&ext.mul(&y, &inv), &ext.one()));
        // y⁻¹ = (1/x)·y.
        let expected = vec![
            ext.base().zero(),
            RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]),
        ];
        assert!(ext.eq(&inv, &expected), "got {inv:?}");
    }

    #[test]
    fn derivation_of_y_is_half_over_x_times_y() {
        // D(y) = (1/2)(D(x)/x)·y = (1/(2x))·y.
        let ext = sqrt_x();
        let y = vec![ext.base().zero(), ext.base().one()];
        let dy = ext.derivation(&y);
        let expected = vec![
            ext.base().zero(),
            RatFn::new(vec![rat(1)], vec![rat(0), rat(2)]), // 1/(2x)
        ];
        assert!(ext.eq(&dy, &expected), "D(y) should be y/(2x); got {dy:?}");
    }

    #[test]
    fn derivation_product_rule_holds() {
        // Sanity: D(y·y) = D(x) = 1, and product rule D(y)y + yD(y) agrees.
        let ext = sqrt_x();
        let y = vec![ext.base().zero(), ext.base().one()];
        let d_yy = ext.derivation(&ext.mul(&y, &y));
        assert!(ext.eq(&d_yy, &ext.one()), "D(y²)=D(x)=1; got {d_yy:?}");
        let dy = ext.derivation(&y);
        let pr = ext.add(&ext.mul(&dy, &y), &ext.mul(&y, &dy));
        assert!(ext.eq(&d_yy, &pr), "product rule mismatch");
    }

    // ---- rational_rde: per-component descent + self-verification ----

    /// Battery: for a target `u` and base scalar `f`, build `g = D(u) + f·u`,
    /// then assert `rational_rde(f, g)` recovers a solution verifying the RDE.
    fn check_solvable(ext: &RadicalExt<RationalDiffField>, f: &Vec<RatFn>, u: &Vec<RatFn>) {
        let g = ext.add(&ext.derivation(u), &ext.mul(f, u));
        let sol = ext.rational_rde(f, &g).expect("should be solvable");
        // Self-verify D(sol)+f·sol = g.
        let lhs = ext.add(&ext.derivation(&sol), &ext.mul(f, &sol));
        assert!(ext.eq(&lhs, &g), "RDE not satisfied; sol={sol:?}");
    }

    #[test]
    fn rde_pure_antiderivative_per_component() {
        let ext = sqrt_x();
        let f = ext.zero(); // f = 0  ⇒  D(u) = g
                            // u = x + x²·y  (component 0 = x, component 1 = x²).
        let u = vec![rf_poly(&[0, 1]), rf_poly(&[0, 0, 1])];
        check_solvable(&ext, &f, &u);
    }

    #[test]
    fn rde_base_scalar_f() {
        let ext = sqrt_x();
        let f = vec![ext.base().one()]; // f = 1 ∈ base
        let u = vec![rf_poly(&[0, 1]), rf_poly(&[1])]; // x + 1·y
        check_solvable(&ext, &f, &u);
    }

    #[test]
    fn rde_unsolvable_component_is_none() {
        // f = 0, g = (1/x)·1  ⇒  component 0 needs ∫1/x = log x ∉ ℚ(x): None.
        let ext = sqrt_x();
        let f = ext.zero();
        let g = vec![RatFn::new(vec![rat(1)], vec![rat(0), rat(1)])]; // 1/x in comp 0
        assert!(ext.rational_rde(&f, &g).is_none(), "log x ∉ ℚ(x) ⇒ None");
    }

    // ---- non-diagonal f over ℚ(x): coupled solve via the AlgExtension bridge ----

    /// Assert `rational_rde(f, g)` solves the *non-diagonal* (coupled) RDE for a
    /// non-base `f`, with `g = D(u_true) + f·u_true` constructed in-field, and
    /// that the returned `u` re-verifies `D(u) + f·u = g`.
    fn check_nondiag_solvable(
        ext: &RadicalExt<RationalDiffField>,
        f: &Vec<RatFn>,
        u_true: &Vec<RatFn>,
    ) {
        // Sanity: f really is non-diagonal (has a higher y-component).
        assert!(ext.trim(f.clone()).len() > 1, "test f must be non-diagonal");
        let g = ext.add(&ext.derivation(u_true), &ext.mul(f, u_true));
        let sol = ext
            .rational_rde(f, &g)
            .expect("non-diagonal f over ℚ(x) should now solve");
        let lhs = ext.add(&ext.derivation(&sol), &ext.mul(f, &sol));
        assert!(ext.eq(&lhs, &g), "coupled RDE not satisfied; sol={sol:?}");
    }

    #[test]
    fn rde_nondiagonal_f_sqrt_x_solves() {
        // y = √x (n=2, a=x).  Non-base f = (1/(2x))·y, target u_true = y.
        // Previously this declined (PR4); now the ℚ(x) coupled hook solves it.
        let ext = sqrt_x();
        let inv_2x = RatFn::new(vec![rat(1)], vec![rat(0), rat(2)]); // 1/(2x)
        let f = vec![ext.base().zero(), inv_2x]; // (1/(2x))·y
        let u_true = vec![ext.base().zero(), ext.base().one()]; // y
        check_nondiag_solvable(&ext, &f, &u_true);
    }

    #[test]
    fn rde_nondiagonal_f_cbrt_x_solves() {
        // y = ∛x (n=3, a=x).  Non-base f = (1/(3x))·y, target u_true = y.
        let ext = RadicalExt::new(RationalDiffField::new(), rf_poly(&[0, 1]), 3);
        let inv_3x = RatFn::new(vec![rat(1)], vec![rat(0), rat(3)]); // 1/(3x)
        let f = vec![ext.base().zero(), inv_3x]; // (1/(3x))·y
        let u_true = vec![ext.base().zero(), ext.base().one()]; // y
        check_nondiag_solvable(&ext, &f, &u_true);
    }

    #[test]
    fn rde_nondiagonal_f_over_tower_base_declines() {
        // RadicalExt over a LOG tower base: the base field has no coupled-radical
        // solver (default hook ⇒ None), so a non-diagonal f still declines.  This
        // documents the remaining hard case (coupled solve over a transcendental
        // tower).
        let dh_over_h = RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]); // 1/x
        let log_field = LogTowerField::new(dh_over_h);
        // Radicand a = x + log x.
        let a = {
            let x = TExpr::from_ratfn(rf_poly(&[0, 1]));
            <LogTowerField as DifferentialField>::add(&log_field, &x, &TExpr::t())
        };
        let ext = RadicalExt::new(log_field.clone(), a, 2);
        // Non-base f = 1·y (nonzero y-component).
        let f = vec![
            <LogTowerField as DifferentialField>::zero(&log_field),
            <LogTowerField as DifferentialField>::one(&log_field),
        ];
        let g = ext.one();
        assert!(
            ext.rational_rde(&f, &g).is_none(),
            "tower-base non-diagonal f ⇒ declines (default hook)"
        );
    }

    // ---- recursion depth: radical over a LOG tower ----

    /// `y = √(x + log x)` over the log tower ℚ(x)(log x): a genuine two-level
    /// `DifferentialField` (radical-over-transcendental).  Exercises the M4
    /// descent
    /// `RadicalExt<LogTowerField>::rational_rde → LogTowerField::rational_rde`.
    #[test]
    fn radical_over_log_tower_descent() {
        // Tower: t = log x, D(t) = 1/x.
        let dh_over_h = RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]); // 1/x
        let log_field = LogTowerField::new(dh_over_h);
        // Radicand a = x + t  (∈ ℚ(x)(t)).
        let a = {
            let x = TExpr::from_ratfn(rf_poly(&[0, 1]));
            <LogTowerField as DifferentialField>::add(&log_field, &x, &TExpr::t())
        };
        let ext = RadicalExt::new(log_field.clone(), a, 2);

        // PR2 headline per-component shape: solve D(w) + w = R with target
        // w = √(x+log x) = 1·y  (component 1 = 1).  Here f = 1 (base scalar).
        let f = vec![<LogTowerField as DifferentialField>::one(&log_field)];
        let target = vec![
            <LogTowerField as DifferentialField>::zero(&log_field),
            <LogTowerField as DifferentialField>::one(&log_field),
        ]; // y
        let g = ext.add(&ext.derivation(&target), &ext.mul(&f, &target));
        let sol = ext
            .rational_rde(&f, &g)
            .expect("radical-over-log descent should solve");
        assert!(
            ext.eq(&sol, &target),
            "recovered w should be y; got {sol:?}"
        );
        // In-field verification.
        let lhs = ext.add(&ext.derivation(&sol), &ext.mul(&f, &sol));
        assert!(ext.eq(&lhs, &g), "D(w)+f·w=g must hold in-field");
    }

    #[test]
    fn stubs_decline() {
        let ext = sqrt_x();
        let one = ext.one();
        assert!(ext
            .limited_integrate(&one, std::slice::from_ref(&one))
            .is_none());
        assert!(ext.param_log_deriv(&one, &one).is_none());
    }
}
