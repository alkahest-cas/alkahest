//! MC2 — FIND-ORDER on genus ≥ 2 curves by reduction modulo good primes.
//!
//! For a hyperelliptic curve `y² = a(x)` of genus `g ≥ 2` there is **no uniform
//! bound** on the order of a rational torsion divisor class, so the genus-1
//! Mazur shortcut ([`super::find_order`]) does not apply.  Instead we use the
//! classical *reduction-mod-good-prime* test (Bronstein 1990 Prop. 1.16–1.17;
//! Davenport, *problem of torsion divisors*):
//!
//! 1. Reduce the curve and the residue divisor `δ` modulo several **good primes**
//!    `p` (`p ≠ 2`, `a` stays squarefree of full degree, denominators invertible).
//! 2. Compute the order `mₚ` of the reduced class in the finite group
//!    `Jac(F_p)` — using Cantor's algorithm on Mumford representations.
//! 3. **Reduction is injective on prime-to-`p` torsion** (a theorem).  Hence if
//!    `δ` were torsion over ℚ of order `N`, then for every good prime the
//!    `prime-to-p` part of `N` equals the `prime-to-p` part of `mₚ`.  Comparing
//!    two good primes therefore pins every `v_ℓ(N)` and any **disagreement
//!    certifies `δ` is non-torsion** ⇒ the integral is non-elementary.
//!
//! The decision is **complete and sound** for this class:
//!
//! * When the prime-to-`p` orders are inconsistent across good primes, `δ` is
//!   non-torsion (a real theorem — no false positives) ⇒ [`FindOrder::NonElementary`].
//! * Otherwise the reconstruction pins the **exact** candidate order `N`, and an
//!   exact Cantor computation **over ℚ** tests whether `N·δ` is principal:
//!   principal ⇒ [`FindOrder::Principal`] (torsion of order `N`), else
//!   non-torsion ⇒ [`FindOrder::NonElementary`].  No Coates function
//!   construction is needed for the *decision* — only for emitting the log
//!   argument, which the genus-1 integrator does and the (not-yet-wired)
//!   genus ≥ 2 integrator would.
//! * [`FindOrder::NotDecided`] only when out of scope: too few good primes, the
//!   order exceeds a practical cap, or the curve is outside the handled class.
//!
//! Scope: `n = 2` (hyperelliptic), `a` squarefree of **odd** degree `2g+1`
//! (imaginary model, a single rational place at infinity).  Even degree (real
//! model, two places at infinity) and `n > 2` fall through to `NotDecided`.

use rug::{Integer, Rational};

use super::super::risch::number_field::KElem;
use super::super::risch::poly_rde::{degree, poly_add, poly_mul, poly_scale, trim, QPoly};
use super::super::risch::rational_rde::{poly_divrem, poly_monic, poly_sub};
use super::find_order::FindOrder;
use super::residues::PlacedResidue;

// ===========================================================================
// F_p scalar arithmetic
// ===========================================================================

#[inline]
fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

#[inline]
fn addmod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p {
        s - p
    } else {
        s
    }
}

#[inline]
fn submod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        p - (b - a)
    }
}

fn powmod(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut r = 1u64 % p;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            r = mulmod(r, base, p);
        }
        base = mulmod(base, base, p);
        exp >>= 1;
    }
    r
}

/// Inverse of `a` in `F_p` (`p` prime, `a ≢ 0`).
fn invmod(a: u64, p: u64) -> u64 {
    powmod(a % p, p - 2, p)
}

/// Reduce a rational `r = n/d` to `F_p`; `None` if `p | d`.
fn rat_to_fp(r: &Rational, p: u64) -> Option<u64> {
    let pb = Integer::from(p);
    let dm = {
        let m = r.denom().clone() % pb.clone();
        let m = if m < 0 { m + &pb } else { m };
        m.to_u64().unwrap()
    };
    if dm == 0 {
        return None;
    }
    let nm = {
        let m = r.numer().clone() % pb.clone();
        let m = if m < 0 { m + &pb } else { m };
        m.to_u64().unwrap()
    };
    Some(mulmod(nm, invmod(dm, p), p))
}

// ===========================================================================
// F_p[x] polynomial arithmetic — dense, little-endian (index = degree).
// ===========================================================================

type FpPoly = Vec<u64>;

fn fp_trim(mut a: FpPoly) -> FpPoly {
    while a.last() == Some(&0) {
        a.pop();
    }
    a
}

/// Degree, or `None` for the zero polynomial.
fn fp_deg(a: &[u64]) -> Option<usize> {
    let a = fp_trim(a.to_vec());
    if a.is_empty() {
        None
    } else {
        Some(a.len() - 1)
    }
}

fn fp_add(a: &[u64], b: &[u64], p: u64) -> FpPoly {
    let n = a.len().max(b.len());
    let mut r = vec![0u64; n];
    for (i, slot) in r.iter_mut().enumerate() {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        *slot = addmod(av, bv, p);
    }
    fp_trim(r)
}

fn fp_sub(a: &[u64], b: &[u64], p: u64) -> FpPoly {
    let n = a.len().max(b.len());
    let mut r = vec![0u64; n];
    for (i, slot) in r.iter_mut().enumerate() {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        *slot = submod(av, bv, p);
    }
    fp_trim(r)
}

fn fp_scale(a: &[u64], s: u64, p: u64) -> FpPoly {
    if s == 0 {
        return vec![];
    }
    fp_trim(a.iter().map(|&c| mulmod(c, s, p)).collect())
}

fn fp_mul(a: &[u64], b: &[u64], p: u64) -> FpPoly {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut r = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            r[i + j] = addmod(r[i + j], mulmod(ai, bj, p), p);
        }
    }
    fp_trim(r)
}

/// Make `a` monic; returns `(monic, leading_coeff)`.  Zero polynomial returns
/// `(zero, 0)`.
fn fp_monic(a: &[u64], p: u64) -> (FpPoly, u64) {
    let a = fp_trim(a.to_vec());
    match a.last() {
        None | Some(0) => (vec![], 0),
        Some(&lc) => {
            let inv = invmod(lc, p);
            (fp_scale(&a, inv, p), lc)
        }
    }
}

/// `(q, r)` with `a = q·b + r`, `deg r < deg b`.  `b` must be nonzero.
fn fp_divrem(a: &[u64], b: &[u64], p: u64) -> (FpPoly, FpPoly) {
    let b = fp_trim(b.to_vec());
    let bd = b.len() - 1;
    let blc_inv = invmod(b[bd], p);
    let mut r = fp_trim(a.to_vec());
    if r.len() < b.len() {
        return (vec![], r);
    }
    let mut q = vec![0u64; r.len() - bd];
    while r.len() >= b.len() && !r.is_empty() {
        let rd = r.len() - 1;
        let coeff = mulmod(r[rd], blc_inv, p);
        let shift = rd - bd;
        q[shift] = coeff;
        for (j, &bj) in b.iter().enumerate() {
            r[shift + j] = submod(r[shift + j], mulmod(coeff, bj, p), p);
        }
        r = fp_trim(r);
    }
    (fp_trim(q), r)
}

fn fp_rem(a: &[u64], b: &[u64], p: u64) -> FpPoly {
    fp_divrem(a, b, p).1
}

fn fp_gcd(a: &[u64], b: &[u64], p: u64) -> FpPoly {
    let mut a = fp_trim(a.to_vec());
    let mut b = fp_trim(b.to_vec());
    while !b.is_empty() {
        let r = fp_rem(&a, &b, p);
        a = b;
        b = r;
    }
    fp_monic(&a, p).0
}

/// Extended gcd: returns `(g, s, t)` with `s·a + t·b = g`, `g` monic.
fn fp_ext_gcd(a: &[u64], b: &[u64], p: u64) -> (FpPoly, FpPoly, FpPoly) {
    let mut r0 = fp_trim(a.to_vec());
    let mut r1 = fp_trim(b.to_vec());
    let mut s0 = vec![1u64];
    let mut s1 = vec![];
    let mut t0 = vec![];
    let mut t1 = vec![1u64];
    while !r1.is_empty() {
        let (q, r) = fp_divrem(&r0, &r1, p);
        let new_s = fp_sub(&s0, &fp_mul(&q, &s1, p), p);
        let new_t = fp_sub(&t0, &fp_mul(&q, &t1, p), p);
        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = new_s;
        t0 = t1;
        t1 = new_t;
    }
    // Normalize g monic.
    let (g, lc) = fp_monic(&r0, p);
    if lc == 0 {
        return (vec![], vec![], vec![]);
    }
    let inv = invmod(lc, p);
    (g, fp_scale(&s0, inv, p), fp_scale(&t0, inv, p))
}

fn fp_eval(a: &[u64], x: u64, p: u64) -> u64 {
    let mut acc = 0u64;
    for &c in a.iter().rev() {
        acc = addmod(mulmod(acc, x, p), c, p);
    }
    acc
}

fn fp_deriv(a: &[u64], p: u64) -> FpPoly {
    if a.len() <= 1 {
        return vec![];
    }
    let mut r = vec![0u64; a.len() - 1];
    for (i, slot) in r.iter_mut().enumerate() {
        let k = (i as u64 + 1) % p;
        *slot = mulmod(a[i + 1], k, p);
    }
    fp_trim(r)
}

// ===========================================================================
// Hyperelliptic Jacobian over F_p (imaginary model y² = F, F monic odd degree)
// ===========================================================================

/// A reduced divisor class in Mumford representation `(u, v)`:
/// `u` monic with `deg u ≤ g`, `deg v < deg u`, and `u | (v² − F)`.
/// The identity (zero class) is `(1, 0)`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Mumford {
    u: FpPoly,
    v: FpPoly,
}

/// Imaginary hyperelliptic curve `y² = F(x)` over `F_p`, `F` monic of degree
/// `2g+1`.
struct HypFp {
    p: u64,
    f: FpPoly,
    g: usize,
}

impl HypFp {
    fn identity(&self) -> Mumford {
        Mumford {
            u: vec![1],
            v: vec![],
        }
    }

    fn is_identity(d: &Mumford) -> bool {
        fp_deg(&d.u) == Some(0)
    }

    /// The class of `(X₀, Y₀) − ∞`, i.e. `u = x − X₀`, `v = Y₀`.
    fn point_class(&self, x0: u64, y0: u64) -> Mumford {
        Mumford {
            u: fp_trim(vec![submod(0, x0, self.p), 1]),
            v: if y0 == 0 { vec![] } else { vec![y0] },
        }
    }

    /// Inverse class `(u, −v mod u)`.
    fn neg(&self, d: &Mumford) -> Mumford {
        let nv = fp_sub(&[], &d.v, self.p);
        let nv = if nv.is_empty() {
            vec![]
        } else {
            fp_rem(&nv, &d.u, self.p)
        };
        Mumford {
            u: d.u.clone(),
            v: nv,
        }
    }

    /// Cantor reduction: while `deg u > g`, replace `(u, v)` by the reduced
    /// equivalent class.
    fn reduce(&self, mut u: FpPoly, mut v: FpPoly) -> Mumford {
        let p = self.p;
        while fp_deg(&u).map(|d| d > self.g).unwrap_or(false) {
            // u' = (F − v²) / u   (exact), then make monic
            let v2 = fp_mul(&v, &v, p);
            let num = fp_sub(&self.f, &v2, p);
            let (mut up, _r) = fp_divrem(&num, &u, p);
            up = fp_monic(&up, p).0;
            // v' = (−v) mod u'
            let nv = fp_sub(&[], &v, p);
            let vp = if up.len() <= 1 {
                vec![]
            } else {
                fp_rem(&nv, &up, p)
            };
            u = up;
            v = vp;
        }
        let u = fp_monic(&u, p).0;
        let v = if u.len() <= 1 {
            vec![]
        } else {
            fp_rem(&v, &u, p)
        };
        Mumford { u, v }
    }

    /// Cantor composition + reduction: `d1 + d2` in `Jac(F_p)`.
    fn add(&self, d1: &Mumford, d2: &Mumford) -> Mumford {
        let p = self.p;
        let (u1, v1) = (&d1.u, &d1.v);
        let (u2, v2) = (&d2.u, &d2.v);

        // d = gcd(u1, u2, v1 + v2) = e1·u1 + e2·u2 + c·(v1+v2)
        let (g1, a1, b1) = fp_ext_gcd(u1, u2, p); // g1 = a1·u1 + b1·u2
        let vsum = fp_add(v1, v2, p);
        let (d, c1, c2) = fp_ext_gcd(&g1, &vsum, p); // d = c1·g1 + c2·vsum
                                                     // s1 = c1·a1, s2 = c1·b1, s3 = c2
        let s1 = fp_mul(&c1, &a1, p);
        let s2 = fp_mul(&c1, &b1, p);
        let s3 = c2;

        // u = u1·u2 / d²
        let u1u2 = fp_mul(u1, u2, p);
        let d2sq = fp_mul(&d, &d, p);
        let (u, _r) = fp_divrem(&u1u2, &d2sq, p);
        let u = fp_monic(&u, p).0;

        // v = (s1·u1·v2 + s2·u2·v1 + s3·(v1·v2 + F)) / d   mod u
        let t1 = fp_mul(&fp_mul(&s1, u1, p), v2, p);
        let t2 = fp_mul(&fp_mul(&s2, u2, p), v1, p);
        let v1v2 = fp_mul(v1, v2, p);
        let t3 = fp_mul(&s3, &fp_add(&v1v2, &self.f, p), p);
        let vnum = fp_add(&fp_add(&t1, &t2, p), &t3, p);
        let (vq, _vr) = fp_divrem(&vnum, &d, p);
        let v = if u.len() <= 1 {
            vec![]
        } else {
            fp_rem(&vq, &u, p)
        };

        self.reduce(u, v)
    }

    /// `k · D` by double-and-add (`k ≥ 0`).
    fn mul(&self, k: u64, d: &Mumford) -> Mumford {
        let mut acc = self.identity();
        let mut base = d.clone();
        let mut k = k;
        while k > 0 {
            if k & 1 == 1 {
                acc = self.add(&acc, &base);
            }
            base = self.add(&base, &base);
            k >>= 1;
        }
        acc
    }

    /// Order of the class `d` in `Jac(F_p)` (always finite — a finite group).
    /// Bounded by the Weil ceiling `(√p + 1)^{2g}`; returns `None` if the cap is
    /// somehow exceeded (a bad reduction we failed to filter — caller skips it).
    fn order(&self, d: &Mumford) -> Option<u64> {
        if Self::is_identity(d) {
            return Some(1);
        }
        let bound = weil_upper_bound(self.p, self.g);
        let mut acc = d.clone();
        let mut k = 1u64;
        while !Self::is_identity(&acc) {
            acc = self.add(&acc, d);
            k += 1;
            if k > bound {
                return None;
            }
        }
        Some(k)
    }
}

/// Weil upper bound `⌈(√p + 1)^{2g}⌉` on `#Jac(F_p)` (a ceiling for the search).
fn weil_upper_bound(p: u64, g: usize) -> u64 {
    let s = (p as f64).sqrt() + 1.0;
    let b = s.powi(2 * g as i32).ceil();
    if b.is_finite() && b < (u64::MAX as f64) {
        b as u64 + 4
    } else {
        u64::MAX
    }
}

// ===========================================================================
// Reduction of the ℚ curve + divisor to F_p, and the multi-prime decision
// ===========================================================================

/// A finite place `(α, β)` of the divisor with integer multiplicity `c`.
struct Place {
    x: Rational,
    y: Rational,
    coeff: Integer,
}

/// Reduce `y² = a(x)` (monic-normalized) and the divisor `places` to `F_p`,
/// building the reduced class.  Returns `None` if `p` is a **bad** prime for
/// this data (so the caller tries the next prime).
fn reduce_and_build(
    a: &QPoly,
    g: usize,
    places: &[Place],
    alg: &[AlgPlace],
    p: u64,
) -> Option<(HypFp, Mumford)> {
    if p == 2 {
        return None;
    }
    let d = degree(a) as usize;
    // Reduce a mod p; leading coeff must survive.
    let mut a_fp = Vec::with_capacity(d + 1);
    for c in a.iter() {
        a_fp.push(rat_to_fp(c, p)?);
    }
    let a_fp = fp_trim(a_fp);
    if a_fp.len() != d + 1 {
        return None; // leading coeff vanished mod p
    }
    let lc = a_fp[d];
    // Monicize via X = lc·x, Y = lc^g·y:  F_k = a_k · lc^{d-1-k}.
    let mut f: FpPoly = vec![0u64; d + 1];
    for (k, slot) in f.iter_mut().enumerate() {
        // F_k = a_k · lc^{d-1-k}; exponent d-1-k ∈ [−1, d−1] (k=d gives lc^{-1}).
        let factor = if k == d {
            invmod(lc, p)
        } else {
            powmod(lc, (d - 1 - k) as u64, p)
        };
        *slot = mulmod(a_fp[k], factor, p);
    }
    let f = fp_trim(f);
    if f.len() != d + 1 || f[d] != 1 {
        return None;
    }
    // Good reduction: F squarefree (gcd(F, F') constant).
    if fp_deg(&fp_gcd(&f, &fp_deriv(&f, p), p)) != Some(0) {
        return None;
    }
    let curve = HypFp { p, f, g };

    // Build the class Σ cₐ·[(Xₐ, Yₐ) − ∞] with the monicizing transform.
    let lc_g = powmod(lc, g as u64, p);
    let mut acc = curve.identity();
    for pl in places {
        let xa = rat_to_fp(&pl.x, p)?;
        let ya = rat_to_fp(&pl.y, p)?;
        let big_x = mulmod(lc, xa, p);
        let big_y = mulmod(lc_g, ya, p);
        // Sanity: the reduced point must lie on the curve.
        if mulmod(big_y, big_y, p) != fp_eval(&curve.f, big_x, p) {
            return None;
        }
        let cls = curve.point_class(big_x, big_y);
        let k = pl.coeff.clone().abs().to_u64()?;
        if k == 0 {
            continue;
        }
        let term = curve.mul(k, &cls);
        let term = if pl.coeff < 0 { curve.neg(&term) } else { term };
        acc = curve.add(&acc, &term);
    }

    // Algebraic orbits: usable only at primes where `minpoly` splits completely
    // (every conjugate root `t` reduces to F_p; the place is `(x_coord(t),
    // y_coord(t))`, keeping the sheet consistent across the orbit).
    for ap in alg {
        let roots = fp_roots_split(&ap.minpoly, p)?; // None ⇒ not fully split ⇒ skip prime
        let k = ap.coeff.clone().abs().to_u64()?;
        if k == 0 {
            continue;
        }
        let x_fp = kelem_to_fp(&ap.x_coord, p)?;
        let y_fp = kelem_to_fp(&ap.y_coord, p)?;
        for t in roots {
            let xt = fp_eval(&x_fp, t, p);
            let yt = fp_eval(&y_fp, t, p);
            let big_x = mulmod(lc, xt, p);
            let big_y = mulmod(lc_g, yt, p);
            // On-curve sanity: Y² = F(X).
            if mulmod(big_y, big_y, p) != fp_eval(&curve.f, big_x, p) {
                return None;
            }
            let cls = curve.point_class(big_x, big_y);
            let term = curve.mul(k, &cls);
            let term = if ap.coeff < 0 { curve.neg(&term) } else { term };
            acc = curve.add(&acc, &term);
        }
    }
    Some((curve, acc))
}

/// Reduce a `ℚ(θ)` element (a `KElem` = `QPoly` mod the minpoly) to `F_p[θ]`:
/// each coefficient mod `p`.  `None` if a denominator vanishes mod `p`.
fn kelem_to_fp(k: &KElem, p: u64) -> Option<FpPoly> {
    let mut out = Vec::with_capacity(k.len());
    for c in k.iter() {
        out.push(rat_to_fp(c, p)?);
    }
    Some(fp_trim(out))
}

/// Roots of `q` in `F_p`, but only if `q` **splits completely** (deg q distinct
/// roots) mod `p`; `None` otherwise (or if a denominator vanishes mod `p`).
fn fp_roots_split(q: &QPoly, p: u64) -> Option<Vec<u64>> {
    let dq = degree(q);
    if dq < 1 {
        return None;
    }
    let mut q_fp = Vec::with_capacity(dq as usize + 1);
    for c in q.iter() {
        q_fp.push(rat_to_fp(c, p)?);
    }
    let q_fp = fp_trim(q_fp);
    if q_fp.len() != dq as usize + 1 {
        return None; // leading coeff vanished mod p
    }
    let mut roots = Vec::new();
    for r in 0..p {
        if fp_eval(&q_fp, r, p) == 0 {
            roots.push(r);
        }
    }
    if roots.len() == dq as usize {
        Some(roots) // splits completely into distinct linear factors
    } else {
        None
    }
}

/// `v_ℓ(m)` — exponent of prime `ℓ` in `m`.
fn val(m: u64, l: u64) -> u32 {
    let mut e = 0;
    let mut m = m;
    while m % l == 0 {
        m /= l;
        e += 1;
    }
    e
}

/// Distinct prime factors of `m` (trial division; `m` is bounded by a Weil
/// ceiling, so this is cheap).
fn prime_factors(mut m: u64) -> Vec<u64> {
    let mut fs = Vec::new();
    let mut d = 2u64;
    while d * d <= m {
        if m % d == 0 {
            fs.push(d);
            while m % d == 0 {
                m /= d;
            }
        }
        d += 1;
    }
    if m > 1 {
        fs.push(m);
    }
    fs
}

// ===========================================================================
// Exact Cantor arithmetic over ℚ — for the *principality* test
//
// Once reduction mod good primes pins the exact candidate torsion order `N`
// (sound, by injectivity of reduction on prime-to-p torsion), `δ` is torsion of
// order `N` **iff** `N·δ` is principal, i.e. reduces to the identity class.  We
// decide that exactly over ℚ with the same Cantor/Mumford machinery — no Coates
// function construction is needed for the *decision* (only for emitting the
// log argument, which the genus-1 integrator does and genus ≥ 2 does not yet).
// ===========================================================================

/// Extended gcd over `ℚ[x]`: `(g, s, t)` with `s·a + t·b = g`, `g` monic.
fn q_ext_gcd(a: &QPoly, b: &QPoly) -> (QPoly, QPoly, QPoly) {
    let mut r0 = trim(a.clone());
    let mut r1 = trim(b.clone());
    let mut s0 = vec![Rational::from(1)];
    let mut s1: QPoly = vec![];
    let mut t0: QPoly = vec![];
    let mut t1 = vec![Rational::from(1)];
    while degree(&r1) >= 0 {
        let (q, r) = poly_divrem(&r0, &r1);
        let new_s = poly_sub(&s0, &poly_mul(&q, &s1));
        let new_t = poly_sub(&t0, &poly_mul(&q, &t1));
        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = new_s;
        t0 = t1;
        t1 = new_t;
    }
    let lc = match trim(r0.clone()).last() {
        Some(c) => c.clone(),
        None => return (vec![], vec![], vec![]),
    };
    let inv = Rational::from(1) / lc;
    (
        poly_scale(&trim(r0), &inv),
        poly_scale(&s0, &inv),
        poly_scale(&t0, &inv),
    )
}

fn q_is_zero(p: &QPoly) -> bool {
    degree(p) < 0
}

fn q_rem(a: &QPoly, b: &QPoly) -> QPoly {
    poly_divrem(a, b).1
}

/// A reduced Mumford class `(u, v)` over ℚ (`u` monic, `deg v < deg u ≤ g`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct MumQ {
    u: QPoly,
    v: QPoly,
}

/// Imaginary hyperelliptic curve `y² = F(x)` over ℚ, `F` monic of odd degree.
struct HypQ {
    f: QPoly,
    g: usize,
}

impl HypQ {
    fn identity(&self) -> MumQ {
        MumQ {
            u: vec![Rational::from(1)],
            v: vec![],
        }
    }

    fn is_identity(d: &MumQ) -> bool {
        degree(&d.u) == 0
    }

    fn point_class(&self, x0: &Rational, y0: &Rational) -> MumQ {
        MumQ {
            u: trim(vec![-x0.clone(), Rational::from(1)]),
            v: if *y0 == 0 { vec![] } else { vec![y0.clone()] },
        }
    }

    fn neg(&self, d: &MumQ) -> MumQ {
        let nv = poly_scale(&d.v, &Rational::from(-1));
        let nv = if q_is_zero(&nv) {
            vec![]
        } else {
            q_rem(&nv, &d.u)
        };
        MumQ {
            u: d.u.clone(),
            v: nv,
        }
    }

    fn reduce(&self, mut u: QPoly, mut v: QPoly) -> MumQ {
        while degree(&u) > self.g as i64 {
            let v2 = poly_mul(&v, &v);
            let num = poly_sub(&self.f, &v2);
            let (mut up, _r) = poly_divrem(&num, &u);
            up = poly_monic(&up);
            let nv = poly_scale(&v, &Rational::from(-1));
            let vp = if degree(&up) <= 0 {
                vec![]
            } else {
                q_rem(&nv, &up)
            };
            u = up;
            v = vp;
        }
        let u = poly_monic(&u);
        let v = if degree(&u) <= 0 {
            vec![]
        } else {
            q_rem(&v, &u)
        };
        MumQ { u, v }
    }

    fn add(&self, d1: &MumQ, d2: &MumQ) -> MumQ {
        let (u1, v1) = (&d1.u, &d1.v);
        let (u2, v2) = (&d2.u, &d2.v);
        let (g1, a1, b1) = q_ext_gcd(u1, u2);
        let vsum = poly_add(v1, v2);
        let (d, c1, c2) = q_ext_gcd(&g1, &vsum);
        let s1 = poly_mul(&c1, &a1);
        let s2 = poly_mul(&c1, &b1);
        let s3 = c2;

        let u1u2 = poly_mul(u1, u2);
        let d2sq = poly_mul(&d, &d);
        let (u, _r) = poly_divrem(&u1u2, &d2sq);
        let u = poly_monic(&u);

        let t1 = poly_mul(&poly_mul(&s1, u1), v2);
        let t2 = poly_mul(&poly_mul(&s2, u2), v1);
        let v1v2 = poly_mul(v1, v2);
        let t3 = poly_mul(&s3, &poly_add(&v1v2, &self.f));
        let vnum = poly_add(&poly_add(&t1, &t2), &t3);
        let (vq, _vr) = poly_divrem(&vnum, &d);
        let v = if degree(&u) <= 0 {
            vec![]
        } else {
            q_rem(&vq, &u)
        };
        self.reduce(u, v)
    }

    fn mul(&self, k: u64, d: &MumQ) -> MumQ {
        let mut acc = self.identity();
        let mut base = d.clone();
        let mut k = k;
        while k > 0 {
            if k & 1 == 1 {
                acc = self.add(&acc, &base);
            }
            base = self.add(&base, &base);
            k >>= 1;
        }
        acc
    }
}

/// Build the monic curve `F` over ℚ (`X = lc·x`, `Y = lc^g·y`) and the divisor
/// class `Σ cₐ·[(Xₐ,Yₐ) − ∞]`, then test whether `N·δ` is principal (reduces to
/// the identity).  `None` if a point fails to lie on the (monicized) curve
/// (should not happen for a genuine residue divisor) or `N` overflows the cap.
fn n_delta_is_principal(a: &QPoly, g: usize, places: &[Place], n: u64) -> Option<bool> {
    let d = degree(a) as usize;
    let lc = a[d].clone();
    // F_k = a_k · lc^{d-1-k}.
    let mut f = vec![Rational::from(0); d + 1];
    for (k, slot) in f.iter_mut().enumerate() {
        let e = d as i64 - 1 - k as i64;
        *slot = a[k].clone() * pow_rat(&lc, e);
    }
    let f = trim(f);
    let curve = HypQ { f, g };
    let lc_g = pow_rat(&lc, g as i64);
    let mut delta = curve.identity();
    for pl in places {
        let big_x = lc.clone() * &pl.x;
        let big_y = lc_g.clone() * &pl.y;
        // On-curve sanity: Y² = F(X).
        let fx = eval_q(&curve.f, &big_x);
        if big_y.clone() * &big_y != fx {
            return None;
        }
        let cls = curve.point_class(&big_x, &big_y);
        let k = pl.coeff.clone().abs().to_u64()?;
        if k == 0 {
            continue;
        }
        let term = curve.mul(k, &cls);
        let term = if pl.coeff < 0 { curve.neg(&term) } else { term };
        delta = curve.add(&delta, &term);
    }
    let nd = curve.mul(n, &delta);
    Some(HypQ::is_identity(&nd))
}

/// `r^e` for a rational base and (possibly negative) integer exponent.
fn pow_rat(r: &Rational, e: i64) -> Rational {
    if e >= 0 {
        let mut acc = Rational::from(1);
        for _ in 0..e {
            acc *= r;
        }
        acc
    } else {
        let mut acc = Rational::from(1);
        for _ in 0..(-e) {
            acc *= r;
        }
        Rational::from(1) / acc
    }
}

fn eval_q(p: &QPoly, x: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// FIND-ORDER for genus ≥ 2 hyperelliptic curves `y² = a(x)` (odd-degree
/// squarefree `a`).  **Complete and sound** for this class: reduction mod good
/// primes pins the exact candidate order `N` (injectivity of reduction on
/// prime-to-`p` torsion), then an exact Cantor computation over ℚ tests whether
/// `N·δ` is principal — giving [`FindOrder::Principal`] (torsion, order `N`) or
/// [`FindOrder::NonElementary`] (non-torsion).  Even-degree (real) models with
/// **no residue at infinity** are reduced to an odd model by moving a rational
/// branch point to infinity.  [`FindOrder::NotDecided`] only when out of scope
/// (too few good primes, `N` past the cap, an even model with a residue at ∞ or
/// no usable rational branch point).
/// A Galois orbit of finite places of `y² = a(x)`, parameterized by a root `θ`
/// of an irreducible `minpoly`: the place is `(x_coord(θ), y_coord(θ))` with
/// `x_coord, y_coord ∈ ℚ(θ)` (`KElem` = `QPoly` reduced mod `minpoly`), with
/// integer multiplicity `coeff`.  A **branch** place is `y_coord = 0` (empty);
/// expressing the `y`-coordinate as an element of `ℚ(θ)` keeps the sheet
/// consistent across the conjugate roots and across primes.  Reduced only at
/// primes where `minpoly` **splits completely** (every conjugate becomes
/// `F_p`-rational).
#[derive(Clone, Debug)]
pub(crate) struct AlgPlace {
    pub minpoly: QPoly,
    pub x_coord: KElem,
    pub y_coord: KElem,
    pub coeff: Integer,
}

pub(crate) fn find_order_genus_ge2(n: usize, a: &QPoly, divisor: &[PlacedResidue]) -> FindOrder {
    find_order_genus_ge2_alg(n, a, divisor, &[])
}

/// FIND-ORDER for genus ≥ 2 with **rational** places (`divisor`) **and**
/// **algebraic** places (`alg`, Galois orbits parameterized by a root of an
/// irreducible minpoly — branch or non-branch).  Algebraic places are reduced
/// modulo only those good primes where their minpoly **splits completely** (so
/// every conjugate becomes `F_p`-rational), then folded into the same Cantor /
/// two-prime torsion test.  Odd-degree (imaginary) model only when `alg` is
/// nonempty.  Sound: prime-to-`p` inconsistency ⇒ `NonElementary`; `Principal`
/// asserted only for branch-only divisors (order ∣ 2, exact mod-p order) or a
/// trivial (`N=1`) class; other consistent algebraic candidates ⇒ `NotDecided`
/// (no exact ℚ(θ)-Cantor principality test).
pub(crate) fn find_order_genus_ge2_alg(
    n: usize,
    a: &QPoly,
    divisor: &[PlacedResidue],
    alg_places: &[AlgPlace],
) -> FindOrder {
    if n != 2 {
        return FindOrder::NotDecided;
    }
    let a = trim(a.clone());
    let dd = degree(&a);
    if dd < 5 {
        return FindOrder::NotDecided; // genus ≥ 2 needs deg ≥ 5
    }

    // Collect finite rational places with integer multiplicities (scaled by the
    // common denominator; algebraic coeffs are already integers and join the
    // same primitivizing gcd so the whole divisor is scaled consistently).
    let finite: Vec<&PlacedResidue> = divisor.iter().filter(|r| !r.residue.at_infinity).collect();
    let mut l = Integer::from(1);
    for r in &finite {
        l = l.lcm(r.residue.value.denom());
    }
    let mut rat_coeffs: Vec<Integer> = finite
        .iter()
        .map(|r| {
            (r.residue.value.clone() * Rational::from(l.clone()))
                .numer()
                .clone()
        })
        .collect();
    let alg_coeffs: Vec<Integer> = alg_places.iter().map(|p| p.coeff.clone() * &l).collect();
    let mut gg = Integer::from(0);
    for c in rat_coeffs.iter().chain(alg_coeffs.iter()) {
        gg = gg.gcd(c);
    }
    if gg == 0 {
        return FindOrder::NotDecided; // no logarithmic part
    }
    for c in &mut rat_coeffs {
        *c /= &gg;
    }
    let places: Vec<Place> = finite
        .iter()
        .zip(&rat_coeffs)
        .filter_map(|(r, c)| {
            if *c == 0 {
                None
            } else {
                Some(Place {
                    x: r.residue.point.clone(),
                    y: r.y_coord.clone(),
                    coeff: c.clone(),
                })
            }
        })
        .collect();
    let alg: Vec<AlgPlace> = alg_places
        .iter()
        .zip(&alg_coeffs)
        .filter_map(|(p, c)| {
            let coeff = c.clone() / &gg;
            if coeff == 0 {
                None
            } else {
                Some(AlgPlace {
                    minpoly: trim(p.minpoly.clone()),
                    x_coord: p.x_coord.clone(),
                    y_coord: trim(p.y_coord.clone()),
                    coeff,
                })
            }
        })
        .collect();
    if places.is_empty() && alg.is_empty() {
        return FindOrder::NotDecided;
    }

    if dd % 2 == 1 {
        // Imaginary model: a single rational place at infinity, absorbed by Σ = 0.
        decide_odd(&a, &places, &alg)
    } else if !alg.is_empty() {
        // Even model + algebraic branch places: not handled (the even→odd move
        // would have to transform the algebraic places too).
        FindOrder::NotDecided
    } else {
        // Real model (two places at infinity).  Handle only the case with **no
        // residue at infinity** (mirrors the genus-1 quartic path): move a
        // rational branch point to infinity via `x = r + 1/s` to get an odd
        // model, then decide there.  Any residue at ∞, no rational root, or a
        // residue on the moved fibre ⇒ undecided.
        if divisor
            .iter()
            .any(|r| r.residue.at_infinity && r.residue.value != 0)
        {
            return FindOrder::NotDecided;
        }
        // Pick a rational branch point **off** the divisor support to move to ∞.
        let Some(r) = rational_root_off_support(&a, &places) else {
            return FindOrder::NotDecided;
        };
        let Some((a_odd, places_odd)) = even_to_odd(&a, &r, &places) else {
            return FindOrder::NotDecided;
        };
        decide_odd(&a_odd, &places_odd, &[])
    }
}

/// A rational root of `a` that is **not** the `x`-coordinate of any divisor
/// place (so the moved fibre `x=r` carries no residue).  Found by deflating out
/// each rational root in turn.
fn rational_root_off_support(a: &QPoly, places: &[Place]) -> Option<Rational> {
    let support: Vec<Rational> = places.iter().map(|p| p.x.clone()).collect();
    let mut poly = trim(a.clone());
    while let Some(r) = super::find_order::first_rational_root(&poly) {
        if !support.iter().any(|s| *s == r) {
            return Some(r);
        }
        // Deflate by (x − r) and look for the next rational root.
        let lin = vec![-r.clone(), Rational::from(1)];
        let (q, _rem) = poly_divrem(&poly, &lin);
        poly = trim(q);
        if degree(&poly) < 1 {
            break;
        }
    }
    None
}

/// Reduce an even-degree model `y²=a(x)` (with `a(r)=0`, `r` rational) to the
/// odd model `ỹ²=ã(s)`, `ã(s) = sᵈ·a(r+1/s)`, mapping each finite place
/// `(α,β) ↦ (1/(α−r), β/(α−r)^{g+1})`.  `None` if a place sits on the moved
/// fibre `x=r` (it would go to `s=∞`).
fn even_to_odd(a: &QPoly, r: &Rational, places: &[Place]) -> Option<(QPoly, Vec<Place>)> {
    let d = degree(a) as usize; // 2g+2
    let g = (d - 2) / 2;
    // ã = reverse(a(x+r)); the constant term a(r)=0 drops the top, giving deg d-1.
    let b = poly_shift(a, r);
    let mut a_odd = vec![Rational::from(0); d + 1];
    for (i, slot) in a_odd.iter_mut().enumerate() {
        if let Some(c) = b.get(d - i) {
            *slot = c.clone();
        }
    }
    let a_odd = trim(a_odd);
    if degree(&a_odd) != (d as i64 - 1) {
        return None;
    }
    let mut out = Vec::with_capacity(places.len());
    for pl in places {
        let dx = pl.x.clone() - r;
        if dx == 0 {
            return None; // place on the moved fibre x=r
        }
        let s0 = Rational::from(1) / &dx;
        let y_new = pl.y.clone() * pow_rat(&s0, g as i64 + 1);
        out.push(Place {
            x: s0,
            y: y_new,
            coeff: pl.coeff.clone(),
        });
    }
    Some((a_odd, out))
}

/// `a(x + r)` (Taylor shift).
fn poly_shift(a: &QPoly, r: &Rational) -> QPoly {
    let mut result: QPoly = vec![];
    let mut xr_pow = vec![Rational::from(1)]; // (x+r)^0
    let xr = vec![r.clone(), Rational::from(1)]; // x + r
    for ak in a.iter() {
        if *ak != 0 {
            result = poly_add(&result, &poly_scale(&xr_pow, ak));
        }
        xr_pow = poly_mul(&xr_pow, &xr);
    }
    trim(result)
}

/// The decision core for an **odd-degree** imaginary model `y²=a(x)` with the
/// already-collected primitive integer finite `places` and algebraic branch
/// orbits `alg` (`(minpoly, coeff)`, reduced only at fully-split primes).
fn decide_odd(a: &QPoly, places: &[Place], alg: &[AlgPlace]) -> FindOrder {
    let dd = degree(a);
    if dd < 5 || dd % 2 == 0 {
        return FindOrder::NotDecided;
    }
    let g = (dd as usize - 1) / 2;

    // Gather (prime, order) pairs over several good primes.
    const MAX_PRIME: u64 = 200;
    const WANT: usize = 4;
    let mut data: Vec<(u64, u64)> = Vec::new();
    let mut p = 3u64;
    while p <= MAX_PRIME && data.len() < WANT {
        if crate::modular::is_prime(p) {
            if let Some((curve, cls)) = reduce_and_build(a, g, places, alg, p) {
                if let Some(order) = curve.order(&cls) {
                    data.push((p, order));
                }
            }
        }
        p += 2;
    }
    if data.len() < 2 {
        return FindOrder::NotDecided; // not enough good primes to conclude
    }

    // Prime-to-p reconstruction: `None` ⇒ inconsistent ⇒ non-torsion (sound,
    // works for rational *and* algebraic places); `Some(N)` ⇒ candidate order.
    let Some(n) = reconstruct_candidate_order(&data) else {
        return FindOrder::NonElementary;
    };
    if n == 0 || n > u32::MAX as u64 {
        return FindOrder::NotDecided; // order past a practical cap
    }

    if alg.is_empty() {
        // Rational-place divisor: confirm torsion by the exact ℚ-Cantor test.
        return match n_delta_is_principal(a, g, places, n) {
            Some(true) => FindOrder::Principal { order: n as u32 },
            Some(false) => FindOrder::NonElementary,
            None => FindOrder::NotDecided,
        };
    }

    // Algebraic places present — the exact ℚ-Cantor principality test does not
    // run over a number field, so we can soundly confirm `Principal` only when:
    //   * every place is a **branch** point (`y = 0`): then `2·δ` is principal,
    //     so the order divides 2 and the reconstructed `N` is exact; or
    //   * `N = 1`: the class is trivial modulo every good prime, hence trivial
    //     over ℚ (reduction is injective on prime-to-p torsion).
    // Any other (consistent, N > 1, non-branch) candidate ⇒ undecided.
    let all_branch =
        places.iter().all(|pl| pl.y == 0) && alg.iter().all(|pl| degree(&pl.y_coord) < 0);
    if all_branch {
        FindOrder::Principal { order: n as u32 }
    } else if n == 1 {
        FindOrder::Principal { order: 1 }
    } else {
        FindOrder::NotDecided
    }
}

/// From the per-prime orders `data = [(p, mₚ)]`, reconstruct the candidate
/// torsion order `N` (prime-to-p injectivity pins each `v_ℓ(N)`).  Returns
/// `None` on a prime-to-`ℓ` inconsistency — a sound non-torsion certificate.
fn reconstruct_candidate_order(data: &[(u64, u64)]) -> Option<u64> {
    let mut ls: Vec<u64> = Vec::new();
    for (_, m) in data {
        for f in prime_factors(*m) {
            if !ls.contains(&f) {
                ls.push(f);
            }
        }
    }
    let mut big_n = Integer::from(1);
    for &l in &ls {
        let witnesses: Vec<u32> = data
            .iter()
            .filter(|(pi, _)| *pi != l)
            .map(|(_, m)| val(*m, l))
            .collect();
        if witnesses.is_empty() {
            continue; // can't pin v_ℓ(N) without a prime ≠ ℓ
        }
        let v_n = witnesses[0];
        if witnesses.iter().any(|&w| w != v_n) {
            return None; // prime-to-ℓ inconsistency ⇒ non-torsion
        }
        if let Some((_, m)) = data.iter().find(|(pi, _)| *pi == l) {
            if val(*m, l) > v_n {
                return None; // image order's ℓ-part exceeds v_ℓ(N)
            }
        }
        for _ in 0..v_n {
            big_n *= Integer::from(l);
        }
    }
    big_n.to_u64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    /// Build a monic imaginary curve over F_p directly for arithmetic tests.
    fn curve(p: u64, f: &[u64], g: usize) -> HypFp {
        HypFp {
            p,
            f: fp_trim(f.to_vec()),
            g,
        }
    }

    #[test]
    fn fp_divrem_roundtrip() {
        let p = 13;
        let a = vec![1, 0, 0, 0, 2]; // 2x⁴ + 1
        let b = vec![3, 1]; // x + 3
        let (q, r) = fp_divrem(&a, &b, p);
        // a == q·b + r
        let back = fp_add(&fp_mul(&q, &b, p), &r, p);
        assert_eq!(back, fp_trim(a));
        assert!(fp_deg(&r).map(|d| d < 1).unwrap_or(true));
    }

    #[test]
    fn ext_gcd_identity() {
        let p = 17;
        let a = vec![1, 0, 1]; // x²+1
        let b = vec![2, 1]; // x+2
        let (g, s, t) = fp_ext_gcd(&a, &b, p);
        let lhs = fp_add(&fp_mul(&s, &a, p), &fp_mul(&t, &b, p), p);
        assert_eq!(lhs, g);
    }

    /// `D + (−D) = O` and `order(O) = 1` on a genus-2 curve y²=x⁵+1 over F_11.
    #[test]
    fn neg_cancels() {
        let c = curve(11, &[1, 0, 0, 0, 0, 1], 2); // x⁵+1
                                                   // (0,1) is on the curve: 0+1=1.
        let d = c.point_class(0, 1);
        let sum = c.add(&d, &c.neg(&d));
        assert!(HypFp::is_identity(&sum));
        assert_eq!(c.order(&c.identity()), Some(1));
    }

    /// A finite rational point's class has the order the group assigns; the
    /// branch-point class `(α,0)−∞` is 2-torsion (`2·D = O`, `D ≠ O`).
    #[test]
    fn branch_point_two_torsion() {
        // y² = x⁵ - x = x(x-1)(x+1)(x²+1) over F_11 (genus 2). Roots 0, 1, -1.
        // f = -x + x⁵  →  coeffs [0, -1, 0, 0, 0, 1]; over F_11, -1 = 10.
        let c = curve(11, &[0, 10, 0, 0, 0, 1], 2);
        let d = c.point_class(1, 0); // (1,0), a branch point
        assert!(!HypFp::is_identity(&d));
        let dd = c.add(&d, &d);
        assert!(HypFp::is_identity(&dd));
        assert_eq!(c.order(&d), Some(2));
    }

    /// Difference of two rational branch points is 2-torsion of order 2.
    #[test]
    fn branch_difference_order_two() {
        let c = curve(11, &[0, 10, 0, 0, 0, 1], 2);
        let p0 = c.point_class(0, 0);
        let p1 = c.point_class(1, 0);
        let diff = c.add(&p0, &c.neg(&p1)); // (0,0) - (1,0)
        assert!(!HypFp::is_identity(&diff));
        assert!(HypFp::is_identity(&c.add(&diff, &diff)));
        assert_eq!(c.order(&diff), Some(2));
    }

    fn place(x: i64, y: i64, value: i64) -> PlacedResidue {
        PlacedResidue {
            residue: super::super::residues::Residue {
                point: Rational::from(x),
                at_infinity: false,
                sheet: 0,
                ramification: 1,
                value: Rational::from(value),
            },
            y_coord: Rational::from(y),
        }
    }

    /// Even-degree real model `y²=x⁶+1` with no rational branch point (roots are
    /// complex 12th roots of unity) ⇒ NotDecided (can't move a place to ∞).
    #[test]
    fn even_degree_no_rational_root_not_decided() {
        let a = qp(&[1, 0, 0, 0, 0, 0, 1]);
        let div = [place(0, 1, 1), place(0, -1, -1)];
        assert_eq!(find_order_genus_ge2(2, &a, &div), FindOrder::NotDecided);
    }

    /// Even-degree real model `y²=(x²−1)(x²−4)(x²−9)=x⁶−14x⁴+49x²−36` (genus 2,
    /// rational branch points ±1,±2,±3).  The branch difference `(1,0) − (2,0)`
    /// is 2-torsion: reduce a rational root to ∞, then the exact ℚ test gives
    /// `Principal{2}`.
    #[test]
    fn even_degree_branch_difference_principal() {
        let a = qp(&[-36, 0, 49, 0, -14, 0, 1]);
        assert_eq!(super::super::find_order::genus(2, &a), Some(2));
        let div = [place(1, 0, 1), place(2, 0, -1)];
        assert_eq!(
            find_order_genus_ge2(2, &a, &div),
            FindOrder::Principal { order: 2 }
        );
    }

    /// Genus-2 `y² = x(x-1)(x+1)(x²+1) = x⁵ - x`.  The divisor `(0,0) − (1,0)`
    /// of two distinct rational branch points is 2-torsion: the exact ℚ test
    /// confirms `2·δ` is principal ⇒ `Principal{2}`.
    #[test]
    fn genus2_branch_difference_principal_order_two() {
        let a = qp(&[0, -1, 0, 0, 0, 1]);
        assert_eq!(super::super::find_order::genus(2, &a), Some(2));
        let div = [place(0, 0, 1), place(1, 0, -1)];
        assert_eq!(
            find_order_genus_ge2(2, &a, &div),
            FindOrder::Principal { order: 2 }
        );
    }

    /// **Algebraic branch places** on `y² = (x²−2)(x³+1) = x⁵ − 2x³ + x² − 2`
    /// (genus 2).  The conjugate branch points `(±√2, 0)` (a Galois orbit over
    /// `x²−2`) form a 2-torsion class — wired in via primes that split `x²−2`
    /// completely (order divides 2 ⇒ the mod-p order is exact) ⇒ `Principal{2}`.
    #[test]
    fn genus2_algebraic_branch_orbit_principal() {
        let a = qp(&[-2, 0, 1, -2, 0, 1]); // x⁵ − 2x³ + x² − 2
        assert_eq!(super::super::find_order::genus(2, &a), Some(2));
        let alg = [AlgPlace {
            minpoly: qp(&[-2, 0, 1]), // x² − 2
            x_coord: qp(&[0, 1]),     // x = θ (the root √2)
            y_coord: Vec::new(),      // branch: y = 0
            coeff: Integer::from(1),
        }];
        assert_eq!(
            find_order_genus_ge2_alg(2, &a, &[], &alg),
            FindOrder::Principal { order: 2 }
        );
    }

    /// A rational branch point combined with an algebraic branch orbit:
    /// `(−1,0) − [(√2,0)+(−√2,0)]` on the same curve is still branch-only ⇒
    /// 2-torsion ⇒ `Principal{2}` (mixed rational + algebraic places).
    #[test]
    fn genus2_mixed_rational_algebraic_branch() {
        let a = qp(&[-2, 0, 1, -2, 0, 1]); // x⁵ − 2x³ + x² − 2; x=−1 is a branch (x³+1=0)
        let rat = [place(-1, 0, 2)]; // 2·(−1,0)
        let alg = [AlgPlace {
            minpoly: qp(&[-2, 0, 1]),
            x_coord: qp(&[0, 1]),
            y_coord: Vec::new(),
            coeff: Integer::from(-1), // −1 at each of (±√2,0)
        }];
        assert_eq!(
            find_order_genus_ge2_alg(2, &a, &rat, &alg),
            FindOrder::Principal { order: 2 }
        );
    }

    /// **Non-branch** algebraic places.  On `y² = x⁵ − 2x + 3` we have
    /// `a(√2) = 3+2√2 = (1+√2)²`, so `(±√2, ±(1±√2))` are non-branch algebraic
    /// points (`y`-coordinate `±(1+θ) ∈ ℚ(θ)`, `θ=√2`).  All four together are
    /// the zeros of `x²−2` on the curve, so `Σ[(P)−∞] = div(x²−2)` is principal
    /// ⇒ `Principal{1}` (the `N=1` sound path).  Exercises sheet-consistent
    /// non-branch reduction via `y_coord(θ)`.
    #[test]
    fn genus2_nonbranch_symmetric_principal_one() {
        let a = qp(&[3, -2, 0, 0, 0, 1]); // x⁵ − 2x + 3
        assert_eq!(super::super::find_order::genus(2, &a), Some(2));
        let alg = [
            AlgPlace {
                minpoly: qp(&[-2, 0, 1]),
                x_coord: qp(&[0, 1]), // θ = √2
                y_coord: qp(&[1, 1]), // 1 + θ
                coeff: Integer::from(1),
            },
            AlgPlace {
                minpoly: qp(&[-2, 0, 1]),
                x_coord: qp(&[0, 1]),
                y_coord: qp(&[-1, -1]), // −(1 + θ)
                coeff: Integer::from(1),
            },
        ];
        assert_eq!(
            find_order_genus_ge2_alg(2, &a, &[], &alg),
            FindOrder::Principal { order: 1 }
        );
    }

    /// Genus-2 `y² = x⁵ + x + 1` with the non-torsion divisor `(0,1) − ∞`
    /// (the infinity place is absorbed by Σ coeff = 0).  The reduced class has
    /// orders 6, 44, 47, … at p = 5, 11, 13 — wildly inconsistent prime-to-p
    /// parts ⇒ certified NonElementary.
    #[test]
    fn genus2_non_torsion_certified() {
        // (0,1) on the curve: 0+0+1 = 1 = 1² ✓.
        let a = qp(&[1, 1, 0, 0, 0, 1]);
        assert_eq!(super::super::find_order::genus(2, &a), Some(2));
        let div = [place(0, 1, 1)];
        assert_eq!(find_order_genus_ge2(2, &a, &div), FindOrder::NonElementary);
    }
}
