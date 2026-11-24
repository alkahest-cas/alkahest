//! Grevlex → lex for a zero-dimensional ideal in **shape position**, by a
//! Krylov ("sparse") FGLM modulo primes and a lift through the Kronecker
//! representation.
//!
//! The ideal `I ⊂ ℚ[x_0, …, x_{n-1}]`, with `d = dim ℚ[x]/I`, is in shape
//! position when `1, t, …, t^{d-1}` (`t = x_{n-1}`, the smallest variable in
//! lex) are linearly independent modulo `I`.  Its reduced lex basis is then
//!
//! ```text
//! f(t),   x_i − g_i(t)   (i < n − 1),      deg f = d,  deg g_i < d,
//! ```
//!
//! and all of it comes out of the coordinate vectors `u_k` of `t^k` on the
//! grevlex staircase: `u_{k+1} = M_t·u_k`, `f` expresses `u_d` in the `u_k`,
//! and `g_i` expresses the coordinates of `x_i`.  Modulo a prime that costs `d`
//! products with the (sparse) multiplication matrix of `t` and one dense `d×d`
//! solve with `n` right-hand sides — instead of a run of the general FGLM,
//! whose traversal reduces every candidate monomial against all earlier ones
//! (Faugère–Mou, *Sparse FGLM algorithms*, JSC 80 (2017)).
//!
//! **What is lifted.**  The `g_i` have far larger coefficients than `f`: their
//! height grows like `d` times that of `f` (Dahan–Schost, *Sharp estimates for
//! triangular sets*, ISSAC 2004), so lifting them directly takes `d` times as
//! many primes.  When `f` is squarefree the `h_i = g_i·f' mod f` (the
//! Kronecker representation, as in msolve's RUR) are about as small as `f`, so
//! `f` and the `h_i` are lifted instead: on katsura-7 that is a few dozen
//! primes instead of 2572.  The `g_i = h_i·(f')⁻¹ mod f` are then rebuilt from
//! the exact `f` and `h_i`, modulo as many further primes as their size needs,
//! but each of those costs an inverse and `n − 1` products modulo `f` in
//! machine words, not a run of FGLM.
//!
//! **Why the images are right.**  A prime `p` is used only when it divides no
//! denominator of the multiplication data and `1, …, t^{d-1}` stay independent
//! modulo `p` (the solve finds a pivot in every column).  The data modulo `p`
//! is then the reduction of the ℚ-data, the solution of the system modulo `p`
//! is the reduction of its solution over ℚ, and so `f`, `g_i`, `h_i` modulo `p`
//! are the reductions of the ℚ polynomials, whatever `p` is.  There is no
//! unlucky prime to out-vote.
//!
//! **What is proved.**  Nothing rests on the lift: the lex basis it gives is
//! checked exactly against the grevlex basis by
//! [`super::multimodular::verify_lex`] — every element has zero normal form,
//! and the lex staircase has exactly `d` monomials — as the general path
//! checks its own.

use std::collections::HashMap;

use rug::{Integer, Rational};

use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::modp::{ModPoly, MonoTable, Prime};
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::multimodular::{
    batch_width, lift_coefficient, lift_loop, rational_mod_p, run_jobs, verify_lex, ModMulMatrix,
    MulMatrices, PrimeResult, RandomPrimes, Unavailable,
};
use rug::Assign;

/// Primes tried before deciding the ideal is not in shape position.
const PROBE_PRIMES: usize = 3;

/// `*a += x` for `a, x < p²`, folded back below `p²`.
#[inline(always)]
fn add_folded(a: &mut u64, x: u64, p2: u64) {
    *a += x;
    if *a >= p2 {
        *a -= p2;
    }
}

/// The coordinates of `x_i · 1` modulo `p`, dense.
fn first_column(m: &ModMulMatrix, d: usize) -> Vec<u64> {
    let mut v = vec![0u64; d];
    match &m[0] {
        Ok(k) => v[*k] = 1,
        Err(col) => {
            for &(k, y) in col {
                v[k] = y;
            }
        }
    }
    v
}

/// `f` (ascending, monic, length `d + 1`) and the `g_i` (ascending, length
/// `d`) modulo `p`, from the multiplication data modulo `p`; `None` when
/// `1, t, …, t^{d-1}` are dependent modulo `p`.
pub(crate) fn shape_mod_p(mats: &[ModMulMatrix], d: usize, pr: Prime) -> Option<ShapeImage> {
    let n = mats.len();
    let p = pr.p;
    let p2 = p * p;
    let mt = &mats[n - 1];
    // Krylov vectors u_0 … u_d of t.
    let mut u: Vec<Vec<u64>> = Vec::with_capacity(d + 1);
    let mut cur = vec![0u64; d];
    cur[0] = 1;
    for _ in 0..d {
        let mut next = vec![0u64; d];
        for (j, &x) in cur.iter().enumerate() {
            if x == 0 {
                continue;
            }
            match &mt[j] {
                Ok(k) => add_folded(&mut next[*k], x, p2),
                Err(col) => {
                    for &(k, y) in col {
                        add_folded(&mut next[k], x * y, p2);
                    }
                }
            }
        }
        for x in &mut next {
            *x %= p;
        }
        u.push(std::mem::replace(&mut cur, next));
    }
    u.push(cur);
    // Solve [u_0 … u_{d-1}] · c = b for b = u_d and the coordinates of each
    // x_i (i < n − 1): rows are coordinates, the first d columns the u_k.
    let m = n; // right-hand sides: u_d, then n − 1 variables
    let w = d + m;
    let mut a = vec![0u64; d * w];
    for (k, uk) in u[..d].iter().enumerate() {
        for (r, &x) in uk.iter().enumerate() {
            a[r * w + k] = x;
        }
    }
    for (r, &x) in u[d].iter().enumerate() {
        a[r * w + d] = x;
    }
    for (i, mi) in mats[..n - 1].iter().enumerate() {
        for (r, x) in first_column(mi, d).into_iter().enumerate() {
            a[r * w + d + 1 + i] = x;
        }
    }
    // Forward elimination with entries kept below p² and reduced when read.
    for col in 0..d {
        let mut piv = None;
        for r in col..d {
            let x = a[r * w + col] % p;
            a[r * w + col] = x;
            if x != 0 {
                piv = Some(r);
                break;
            }
        }
        let piv = piv?;
        if piv != col {
            for j in 0..w {
                a.swap(piv * w + j, col * w + j);
            }
        }
        let inv = pr.inv(a[col * w + col]);
        for j in col..w {
            let x = a[col * w + j] % p;
            a[col * w + j] = pr.mul(x, inv);
        }
        let (top, rest) = a.split_at_mut((col + 1) * w);
        let prow = &top[col * w..];
        for r in 0..d - col - 1 {
            let row = &mut rest[r * w..(r + 1) * w];
            let f = row[col] % p;
            row[col] = 0;
            if f == 0 {
                continue;
            }
            let nf = p - f;
            for j in col + 1..w {
                add_folded(&mut row[j], nf * prow[j], p2);
            }
        }
    }
    // Back substitution, one right-hand side at a time.
    let mut sols: Vec<Vec<u64>> = Vec::with_capacity(m);
    for rhs in 0..m {
        let mut x = vec![0u64; d];
        for r in (0..d).rev() {
            let mut acc = a[r * w + d + rhs] % p;
            for k in r + 1..d {
                let c = a[r * w + k] % p;
                if c != 0 && x[k] != 0 {
                    acc = (acc + p - pr.mul(c, x[k])) % p;
                }
            }
            x[r] = acc;
        }
        sols.push(x);
    }
    let mut it = sols.into_iter();
    let c = it.next().expect("u_d");
    // f = t^d − Σ c_k t^k.
    let mut f: Vec<u64> = c.iter().map(|&x| (p - x) % p).collect();
    f.push(1);
    Some(ShapeImage { f, g: it.collect() })
}

/// The lex basis in shape position modulo one prime.
pub(crate) struct ShapeImage {
    /// The univariate element, ascending, monic.
    pub(crate) f: Vec<u64>,
    /// `x_i − g_i(t)`: the `g_i`, ascending, `d` coefficients each.
    pub(crate) g: Vec<Vec<u64>>,
}

/// `a · b mod f` for a monic `f` of degree `d` and `a, b` of degree below `d`,
/// all modulo `p` (coefficients ascending).  Products are accumulated below
/// `p²` and reduced once.
fn mulmod(a: &[u64], b: &[u64], f: &[u64], pr: Prime) -> Vec<u64> {
    let p = pr.p;
    let p2 = p * p;
    let d = f.len() - 1;
    let mut prod = vec![0u64; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (acc, &y) in prod[i..].iter_mut().zip(b) {
            add_folded(acc, x * y, p2);
        }
    }
    for m in (d..prod.len()).rev() {
        let q = pr.reduce(prod[m]);
        prod[m] = 0;
        if q == 0 {
            continue;
        }
        let nq = p - q;
        for (acc, &y) in prod[m - d..m].iter_mut().zip(&f[..d]) {
            add_folded(acc, nq * y, p2);
        }
    }
    prod.truncate(d);
    prod.resize(d, 0);
    for x in &mut prod {
        *x = pr.reduce(*x);
    }
    prod
}

/// `a − c·y mod p` for residues `a, c, y`.
#[inline(always)]
fn sub_mul(a: u64, c: u64, y: u64, pr: Prime) -> u64 {
    let t = pr.reduce(c * y);
    if a >= t {
        a - t
    } else {
        a + pr.p - t
    }
}

/// `a⁻¹ mod f` modulo `p` for a monic `f` (ascending), by the extended
/// Euclidean algorithm; `None` when `gcd(a, f) ≠ 1`.
fn invmod(a: &[u64], f: &[u64], pr: Prime) -> Option<Vec<u64>> {
    let trim = |v: &mut Vec<u64>| {
        while v.last() == Some(&0) {
            v.pop();
        }
    };
    // Invariant: s_i · a ≡ r_i (mod f).
    let (mut r0, mut r1) = (f.to_vec(), a.to_vec());
    let (mut s0, mut s1): (Vec<u64>, Vec<u64>) = (Vec::new(), vec![1]);
    trim(&mut r0);
    trim(&mut r1);
    while !r1.is_empty() {
        // r0 = q·r1 + rem; s_new = s0 − q·s1.
        let inv = pr.inv(*r1.last().expect("non-empty"));
        let mut q = vec![0u64; r0.len().saturating_sub(r1.len()) + 1];
        while r0.len() >= r1.len() && !r0.is_empty() {
            let c = pr.mul(*r0.last().expect("non-empty"), inv);
            let shift = r0.len() - r1.len();
            q[shift] = c;
            for (j, &y) in r1.iter().enumerate() {
                r0[shift + j] = sub_mul(r0[shift + j], c, y, pr);
            }
            trim(&mut r0);
        }
        let mut s_new = s0.clone();
        let need = q.len() + s1.len();
        if s_new.len() < need {
            s_new.resize(need, 0);
        }
        for (i, &qi) in q.iter().enumerate() {
            if qi == 0 {
                continue;
            }
            for (j, &y) in s1.iter().enumerate() {
                s_new[i + j] = sub_mul(s_new[i + j], qi, y, pr);
            }
        }
        trim(&mut s_new);
        s0 = std::mem::replace(&mut s1, s_new);
        r0 = std::mem::replace(&mut r1, r0);
    }
    // gcd = r0, a non-zero constant iff a is invertible.
    if r0.len() != 1 {
        return None;
    }
    let c = pr.inv(r0[0]);
    let d = f.len() - 1;
    // The Bézout cofactor has degree below deg f.
    let mut out: Vec<u64> = s0.iter().map(|&x| pr.mul(x, c)).collect();
    debug_assert!(out.len() <= d);
    out.resize(d, 0);
    Some(out)
}

/// Is `f` (monic, ascending) squarefree modulo `p`, i.e. `gcd(f, f') = 1`?
fn squarefree_mod_p(f: &[u64], pr: Prime) -> bool {
    let p = pr.p;
    let trim = |v: &mut Vec<u64>| {
        while v.last() == Some(&0) {
            v.pop();
        }
    };
    let mut a: Vec<u64> = f.to_vec();
    let mut b: Vec<u64> = (1..f.len()).map(|k| pr.mul(f[k], k as u64 % p)).collect();
    trim(&mut a);
    trim(&mut b);
    while !b.is_empty() {
        // a ← a mod b
        let lb = *b.last().expect("non-empty");
        let inv = pr.inv(lb);
        while a.len() >= b.len() {
            let q = pr.mul(*a.last().expect("non-empty"), inv);
            let shift = a.len() - b.len();
            for (j, &y) in b.iter().enumerate() {
                a[shift + j] = (a[shift + j] + (p - pr.mul(q, y))) % p;
            }
            trim(&mut a);
            if a.is_empty() {
                break;
            }
        }
        std::mem::swap(&mut a, &mut b);
    }
    a.len() == 1
}

/// The coefficients of the `t`-only polynomial `g` (ascending, `len` of
/// them; missing terms are zero).
fn coefficients_in_t(g: &GbPoly, n: usize, len: usize) -> Vec<Rational> {
    let mut out = vec![Rational::new(); len];
    for (e, c) in &g.terms {
        let k = e[n - 1] as usize;
        if k < len && e[..n - 1].iter().all(|&x| x == 0) {
            out[k] = c.clone();
        }
    }
    out
}

/// `g_i = h_i · (f')⁻¹ mod f` over ℚ, for a monic squarefree `f` of degree
/// `d ≥ 1` (ascending) and `h_i` of degree below `d`, by
/// [`kronecker_to_shape_crt`].  `None` when `f` has a repeated factor, or
/// the conversion was inconclusive.
pub(crate) fn kronecker_to_shape(
    f: &[Rational],
    hs: &[Vec<Rational>],
) -> Option<Vec<Vec<Rational>>> {
    let d = f.len() - 1;
    if d == 1 {
        // f' = 1.
        return Some(hs.to_vec());
    }
    kronecker_to_shape_crt(f, hs)
}

/// A product tree over a list of primes, for Chinese remaindering many
/// residue vectors against the same primes: each combination is then
/// `O(M(n) log n)` rather than the `O(n²)` of adding one prime at a time.
struct CrtTree {
    /// `levels[0]` holds the primes; `levels[k + 1][i]` is the product of
    /// `levels[k][2i]` and `levels[k][2i + 1]` (an unpaired last node is
    /// carried up as it is).
    levels: Vec<Vec<Integer>>,
    /// `invs[k][i] = levels[k][2i]⁻¹ mod levels[k][2i + 1]`.
    invs: Vec<Vec<Integer>>,
}

impl CrtTree {
    fn new(primes: &[u64]) -> Self {
        let mut levels = vec![primes.iter().map(|&p| Integer::from(p)).collect::<Vec<_>>()];
        let mut invs = Vec::new();
        while levels.last().expect("a level").len() > 1 {
            let cur = levels.last().expect("a level");
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut inv = Vec::with_capacity(cur.len() / 2);
            for pair in cur.chunks(2) {
                match pair {
                    [a, b] => {
                        inv.push(
                            Integer::from(a % b)
                                .invert(b)
                                .expect("distinct primes are coprime"),
                        );
                        next.push(Integer::from(a * b));
                    }
                    [a] => next.push(a.clone()),
                    _ => unreachable!("chunks of two"),
                }
            }
            invs.push(inv);
            levels.push(next);
        }
        CrtTree { levels, invs }
    }

    fn modulus(&self) -> &Integer {
        &self.levels.last().expect("a level")[0]
    }

    /// The residue modulo the product of the primes, in `[0, M)`, of the
    /// residues `res(i)` modulo `primes[i]`.
    fn combine(&self, res: impl Fn(usize) -> u64) -> Integer {
        let mut cur: Vec<Integer> = (0..self.levels[0].len())
            .map(|i| Integer::from(res(i)))
            .collect();
        let mut t = Integer::new();
        for (k, inv) in self.invs.iter().enumerate() {
            let ms = &self.levels[k];
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut it = cur.into_iter();
            let mut i = 0;
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => {
                        // a + m_a · ((b − a) · m_a⁻¹ mod m_b)
                        t.assign(&b - &a);
                        t *= &inv[i];
                        t.modulo_mut(&ms[2 * i + 1]);
                        t *= &ms[2 * i];
                        next.push(Integer::from(&a + &t));
                    }
                    None => next.push(a),
                }
                i += 1;
            }
            cur = next;
        }
        cur.pop().unwrap_or_default()
    }
}

/// Polynomials with rational coefficients as integer numerators over one
/// denominator each, so that an image modulo `p` costs one inverse per
/// polynomial rather than one per coefficient.
struct OverDen {
    num: Vec<Integer>,
    den: Integer,
}

impl OverDen {
    fn new(v: &[Rational]) -> Self {
        let mut den = Integer::from(1);
        for c in v {
            if !den.is_divisible(c.denom()) {
                den.lcm_mut(c.denom());
            }
        }
        let num = v
            .iter()
            .map(|c| Integer::from(den.div_exact_ref(c.denom())) * c.numer())
            .collect();
        OverDen { num, den }
    }

    /// The coefficients modulo `p`, or `None` if `p` divides the denominator.
    fn mod_p(&self, pr: Prime) -> Option<Vec<u64>> {
        let dp = u64::from(self.den.mod_u(pr.p as u32));
        if dp == 0 {
            return None;
        }
        let inv = pr.inv(dp);
        Some(
            self.num
                .iter()
                .map(|c| pr.mul(u64::from(c.mod_u(pr.p as u32)), inv))
                .collect(),
        )
    }
}

/// The `g_i = h_i · (f')⁻¹ mod f` modulo `p`, flattened (`d` coefficients
/// each), or `None` when `p` divides a denominator or the discriminant.
fn g_mod_p(f: &OverDen, hs: &[OverDen], pr: Prime) -> Option<Vec<u64>> {
    let fp = f.mod_p(pr)?;
    if fp.last() != Some(&1) {
        return None;
    }
    let d = fp.len() - 1;
    let fd: Vec<u64> = (1..=d).map(|k| pr.mul(fp[k], k as u64 % pr.p)).collect();
    let u = invmod(&fd, &fp, pr)?;
    let mut out = Vec::with_capacity(hs.len() * d);
    for h in hs {
        out.extend(mulmod(&h.mod_p(pr)?, &u, &fp, pr));
    }
    Some(out)
}

/// Primes allowed for [`kronecker_to_shape_crt`] before giving up.
const MAX_CONVERSION_PRIMES: usize = 1 << 16;

/// [`kronecker_to_shape`] by Chinese remaindering: the `g_i` modulo
/// successive random primes (an inverse and `n − 1` products modulo `f`, all
/// in machine words), combined through a [`CrtTree`] and lifted by rational
/// reconstruction once a probe coefficient lifts.  The number of primes is
/// what the size of the `g_i` needs, and nothing passes through the
/// resultant-sized cofactors of an extended gcd over ℤ, which on katsura-7
/// are four times the size of the result.
///
/// A lift is returned once it agrees with one more prime; the caller verifies
/// the basis exactly regardless.  `None` when no lift was found within
/// [`MAX_CONVERSION_PRIMES`] or an armed budget fired.
fn kronecker_to_shape_crt(f: &[Rational], hs: &[Vec<Rational>]) -> Option<Vec<Vec<Rational>>> {
    let d = f.len() - 1;
    let total = hs.len() * d;
    if total == 0 {
        return Some(vec![Vec::new(); hs.len()]);
    }
    let f = &OverDen::new(f);
    let hs: &[OverDen] = &hs.iter().map(|h| OverDen::new(h)).collect::<Vec<_>>();
    let mut rp = RandomPrimes::new();
    let mut primes: Vec<u64> = Vec::new();
    // res[k][i]: coefficient k modulo primes[i] (below 2³¹).
    let mut res: Vec<Vec<u32>> = vec![Vec::new(); total];
    let mut probe = total - 1;
    let mut target = 32usize;
    let mut drawn = 0usize;
    loop {
        // A prime is refused only when it divides a denominator or the
        // discriminant of `f`; refusals beyond a few then mean `f` has a
        // repeated factor.
        if crate::budget::interrupted()
            || primes.len() > MAX_CONVERSION_PRIMES
            || drawn > 2 * primes.len() + 64
        {
            return None;
        }
        let want = target.saturating_sub(primes.len()).max(1);
        let batch: Vec<Prime> = rp.by_ref().take(want).collect();
        drawn += batch.len();
        let imgs = run_jobs(batch, |pr| g_mod_p(f, hs, pr).map(|g| (pr, g))).ok()?;
        for (pr, g) in imgs.into_iter().flatten() {
            primes.push(pr.p);
            for (r, x) in res.iter_mut().zip(g) {
                r.push(x as u32);
            }
        }
        if primes.len() < target {
            continue;
        }
        target += target / 4 + 1;
        let tree = CrtTree::new(&primes);
        let m = tree.modulus();
        let r = tree.combine(|i| u64::from(res[probe][i]));
        if lift_coefficient(&r, m, &mut Integer::from(1)).is_none() {
            continue;
        }
        // Every coefficient, lifted in order with the shared denominator
        // carried along; the residues are combined a block at a time in
        // parallel, so a coefficient that does not lift yet wastes at most
        // one block.
        let block = 8 * batch_width();
        let mut den = Integer::from(1);
        let mut out = Vec::with_capacity(total);
        'blocks: for start in (0..total).step_by(block) {
            let ks: Vec<usize> = (start..(start + block).min(total)).collect();
            let combined = run_jobs(ks, |k| tree.combine(|i| u64::from(res[k][i]))).ok()?;
            for (j, r) in combined.into_iter().enumerate() {
                match lift_coefficient(&r, m, &mut den) {
                    Some(x) => out.push(x),
                    None => {
                        probe = start + j;
                        break 'blocks;
                    }
                }
            }
        }
        if out.len() < total {
            continue;
        }
        // One more prime must agree.
        let (pr, check) = loop {
            let pr = rp.next()?;
            drawn += 1;
            if let Some(g) = g_mod_p(f, hs, pr) {
                break (pr, g);
            }
            if drawn > 2 * primes.len() + 64 {
                return None;
            }
        };
        let agrees = out
            .iter()
            .zip(&check)
            .all(|(x, &c)| rational_mod_p(x, pr) == Some(c));
        primes.push(pr.p);
        for (r, x) in res.iter_mut().zip(check) {
            r.push(x as u32);
        }
        if agrees {
            let mut it = out.into_iter();
            return Some(
                (0..hs.len())
                    .map(|_| it.by_ref().take(d).collect())
                    .collect(),
            );
        }
    }
}

/// A univariate `GbPoly` in `t = x_{n-1}` from ascending coefficients.
fn poly_in_t(coefs: &[Rational], n: usize) -> GbPoly {
    let mut terms = std::collections::BTreeMap::new();
    for (k, c) in coefs.iter().enumerate() {
        if *c != 0 {
            let mut e = vec![0u32; n];
            e[n - 1] = k as u32;
            terms.insert(e, c.clone());
        }
    }
    GbPoly { terms, n_vars: n }
}

/// The reduced lex basis `f(t), x_i − g_i(t)`, sorted by increasing leading
/// monomial (`t^d < x_{n-2} < … < x_0`).
fn shape_basis(f: &[Rational], gs: &[Vec<Rational>], n: usize) -> Vec<GbPoly> {
    let mut out = Vec::with_capacity(n);
    out.push(poly_in_t(f, n));
    for i in (0..n - 1).rev() {
        let neg: Vec<Rational> = gs[i].iter().map(|c| Rational::from(-c)).collect();
        let mut p = poly_in_t(&neg, n);
        let mut e = vec![0u32; n];
        e[i] = 1;
        p.terms.insert(e, Rational::from(1));
        out.push(p);
    }
    out
}

/// The reduced lex basis of the zero-dimensional ideal with reduced grevlex
/// basis `g` and staircase `staircase` (sorted increasingly), if the ideal is
/// in shape position.  `Ok(None)` when it is not (the caller runs the general
/// conversion), `Err(Fallback)` when the shape path was inconclusive.
pub(crate) fn shape_lex(
    g: &[GbPoly],
    n: usize,
    staircase: &[Vec<u32>],
    idx: &HashMap<Vec<u32>, usize>,
    modular: bool,
) -> Result<Option<Vec<GbPoly>>, Unavailable> {
    let d = staircase.len();
    if n < 2 || d == 0 {
        return Ok(None);
    }
    let Some(mats) = MulMatrices::new(g, n, staircase, idx, true) else {
        return Err(if crate::budget::interrupt_pending() {
            Unavailable::Interrupted
        } else {
            Unavailable::Fallback
        });
    };
    // Probe: shape position, and whether `f` is squarefree (a squarefree
    // image modulo one prime proves it over ℚ: the discriminant of `f` is
    // then non-zero).
    let mut primes = RandomPrimes::new();
    let mut kronecker = None;
    let mut tried = 0;
    while tried < PROBE_PRIMES {
        let pr = primes.next().ok_or(Unavailable::Fallback)?;
        let Some(mp) = mats.mod_p(pr) else {
            continue;
        };
        tried += 1;
        if let Some(img) = shape_mod_p(&mp, d, pr) {
            kronecker = Some(squarefree_mod_p(&img.f, pr));
            break;
        }
        if crate::budget::interrupted() {
            return Err(Unavailable::Interrupted);
        }
    }
    let Some(kronecker) = kronecker else {
        return Ok(None);
    };

    let mut table = MonoTable::new(n, MonomialOrder::Lex);
    let mut t_pow = Vec::with_capacity(d + 1);
    for k in 0..=d {
        let mut e = vec![0u32; n];
        e[n - 1] = k as u32;
        t_pow.push(table.intern_u32(&e).ok_or(Unavailable::Fallback)?);
    }
    // Images: [f, then h_i (or g_i)], on full supports t^d … 1 and t^{d-1} … 1.
    let compute = |_t: &mut MonoTable, pr: Prime| -> PrimeResult {
        let Some(mp) = mats.mod_p(pr) else {
            return Ok(None);
        };
        let Some(img) = shape_mod_p(&mp, d, pr) else {
            return Ok(None);
        };
        let mut polys = Vec::with_capacity(n);
        polys.push(ModPoly {
            mons: t_pow.iter().rev().copied().collect(),
            coefs: img.f.iter().rev().copied().collect(),
        });
        let fprime: Vec<u64> = (1..img.f.len())
            .map(|k| pr.mul(img.f[k], k as u64))
            .collect();
        for gi in &img.g {
            let h = if kronecker {
                mulmod(gi, &fprime, &img.f, pr)
            } else {
                gi.clone()
            };
            polys.push(ModPoly {
                mons: t_pow[..d].iter().rev().copied().collect(),
                coefs: h.into_iter().rev().collect(),
            });
        }
        Ok(Some(polys))
    };
    let mut result: Option<Vec<GbPoly>> = None;
    let verify = |cand: &[GbPoly]| -> Option<bool> {
        let f = coefficients_in_t(&cand[0], n, d + 1);
        let hs: Vec<Vec<Rational>> = cand[1..]
            .iter()
            .map(|h| coefficients_in_t(h, n, d))
            .collect();
        let gs = if kronecker {
            match kronecker_to_shape(&f, &hs) {
                Some(gs) => gs,
                None if crate::budget::interrupt_pending() => return None,
                // A repeated factor in a lifted `f` that is squarefree over
                // ℚ: a wrong lift.
                None => return Some(false),
            }
        } else {
            hs
        };
        let basis = shape_basis(&f, &gs, n);
        let ok = if modular {
            true
        } else {
            verify_lex(&basis, &mats, n)?
        };
        if ok {
            result = Some(basis);
        }
        Some(ok)
    };
    lift_loop(&mut table, modular, compute, verify)?;
    Ok(Some(result.ok_or(Unavailable::Fallback)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::groebner::multimodular::{
        fglm_lex_general, groebner_basis, tests::katsura_system,
    };

    fn canon(b: &[GbPoly]) -> Vec<Vec<(Vec<u32>, Rational)>> {
        let mut v: Vec<Vec<(Vec<u32>, Rational)>> = b
            .iter()
            .map(|g| {
                g.terms
                    .iter()
                    .map(|(e, c)| (e.clone(), c.clone()))
                    .collect()
            })
            .collect();
        v.sort();
        v
    }

    fn poly(n: usize, terms: &[(&[u32], i64)]) -> GbPoly {
        GbPoly {
            terms: terms
                .iter()
                .filter(|t| t.1 != 0)
                .map(|(e, c)| (e.to_vec(), Rational::from(*c)))
                .collect(),
            n_vars: n,
        }
    }

    /// Both conversions of the grevlex basis of `gens`: the shape path
    /// (`None` if it declined) and the general one.
    fn both(gens: &[GbPoly], modular: bool) -> (Option<Vec<GbPoly>>, Vec<GbPoly>) {
        let n = gens[0].n_vars;
        let g = groebner_basis(gens, MonomialOrder::GRevLex).expect("grevlex");
        let st = crate::poly::groebner::fglm::grevlex_staircase(&g, n).expect("zero-dim");
        let idx: HashMap<Vec<u32>, usize> =
            st.iter().enumerate().map(|(i, e)| (e.clone(), i)).collect();
        let shape = shape_lex(&g, n, &st, &idx, modular).expect("conclusive");
        let general = fglm_lex_general(&g, n, &st, &idx, false).expect("general");
        (shape, general)
    }

    #[test]
    fn shape_path_matches_the_general_conversion() {
        for n in 2..=5 {
            let gens = katsura_system(n);
            for modular in [false, true] {
                let (shape, general) = both(&gens, modular);
                let shape = shape.expect("katsura is in shape position");
                assert_eq!(canon(&shape), canon(&general), "katsura-{n}");
                let leads: Vec<Vec<u32>> = shape
                    .iter()
                    .map(|g| g.leading_exp(MonomialOrder::Lex).unwrap())
                    .collect();
                assert!(leads.windows(2).all(|w| w[0] < w[1]), "increasing leads");
            }
        }
    }

    #[test]
    fn not_in_shape_position_is_declined() {
        // ⟨x² − 1, y² − 1⟩: y alone has a minimal polynomial of degree 2 < 4.
        let gens = vec![
            poly(2, &[(&[2, 0], 1), (&[0, 0], -1)]),
            poly(2, &[(&[0, 2], 1), (&[0, 0], -1)]),
        ];
        let (shape, general) = both(&gens, false);
        assert!(shape.is_none());
        assert_eq!(general.len(), 2);
    }

    #[test]
    fn a_repeated_root_lifts_the_shape_basis_directly() {
        // ⟨(y − 2)², x − y² − 3⟩ = ⟨y² − 4y + 4, x − 4y + 1⟩: in shape position
        // with f = (y − 2)² not squarefree, so g is lifted as it is.
        let gens = vec![
            poly(2, &[(&[0, 2], 1), (&[0, 1], -4), (&[0, 0], 4)]),
            poly(2, &[(&[1, 0], 1), (&[0, 2], -1), (&[0, 0], -3)]),
        ];
        let (shape, general) = both(&gens, false);
        let shape = shape.expect("shape position");
        assert_eq!(canon(&shape), canon(&general));
        assert_eq!(
            canon(&shape),
            canon(&[
                poly(2, &[(&[0, 2], 1), (&[0, 1], -4), (&[0, 0], 4)]),
                poly(2, &[(&[1, 0], 1), (&[0, 1], -4), (&[0, 0], 1)]),
            ])
        );
    }

    #[test]
    fn partial_matrices_verify_only_what_they_can_read() {
        let gens = katsura_system(3);
        let g = groebner_basis(&gens, MonomialOrder::GRevLex).unwrap();
        let st = crate::poly::groebner::fglm::grevlex_staircase(&g, 4).unwrap();
        let idx: HashMap<Vec<u32>, usize> =
            st.iter().enumerate().map(|(i, e)| (e.clone(), i)).collect();
        let mats = MulMatrices::new(&g, 4, &st, &idx, true).unwrap();
        let mut lex = shape_lex(&g, 4, &st, &idx, false).unwrap().unwrap();
        assert_eq!(verify_lex(&lex, &mats, 4), Some(true));
        let last = lex.last_mut().unwrap();
        let k = last.terms.keys().last().unwrap().clone();
        *last.terms.get_mut(&k).unwrap() += 1;
        assert_eq!(verify_lex(&lex, &mats, 4), Some(false));
        // A monomial the partial matrices cannot place is a rejection, not a
        // guess.
        let mut lex = shape_lex(&g, 4, &st, &idx, false).unwrap().unwrap();
        lex[1].terms.insert(vec![1, 1, 0, 0], Rational::from(0));
        assert_eq!(verify_lex(&lex, &mats, 4), Some(false));
    }

    #[test]
    fn mulmod_and_squarefree() {
        let pr = Prime::new(1_000_003);
        // f = t² − 2 (ascending: [-2, 0, 1]); t · t ≡ 2.
        let p = pr.p;
        let f = vec![p - 2, 0, 1];
        assert_eq!(mulmod(&[0, 1], &[0, 1], &f, pr), vec![2, 0]);
        assert!(squarefree_mod_p(&f, pr));
        // (t − 1)² = t² − 2t + 1.
        assert!(!squarefree_mod_p(&[1, p - 2, 1], pr));
        // t⁻¹ ≡ t/2 modulo t² − 2; t − 1 is not invertible modulo (t − 1)².
        let inv2 = pr.inv(2);
        assert_eq!(invmod(&[0, 1], &f, pr), Some(vec![0, inv2]));
        assert_eq!(invmod(&[p - 1, 1], &[1, p - 2, 1], pr), None);
        // A random element times its inverse is one.
        let f5 = vec![3, 1, 4, 1, 5, 1];
        let a = vec![2, 7, 1, 8, 2];
        let u = invmod(&a, &f5, pr).expect("coprime");
        assert_eq!(mulmod(&a, &u, &f5, pr), vec![1, 0, 0, 0, 0]);
    }

    #[test]
    fn crt_tree_combines_like_incremental_crt() {
        let primes = [1_000_003u64, 1_000_033, 1_000_037, 1_000_039, 1_000_081];
        let tree = CrtTree::new(&primes);
        let x = Integer::from(123_456_789_012_345_678u64) * 987_654_321u64;
        let m = primes.iter().fold(Integer::from(1), |acc, &p| acc * p);
        assert_eq!(*tree.modulus(), m);
        let res: Vec<u64> = primes
            .iter()
            .map(|&p| u64::from(x.mod_u(p as u32)))
            .collect();
        assert_eq!(tree.combine(|i| res[i]), Integer::from(&x % &m));
    }

    #[test]
    fn kronecker_inverts_the_derivative() {
        // f = t² − 2, g = 3t + 1, h = g·f' mod f = (3t + 1)·2t = 6t² + 2t ≡ 2t + 12.
        let q = |n: i64| Rational::from(n);
        let f = vec![q(-2), q(0), q(1)];
        let h = vec![q(12), q(2)];
        let g = kronecker_to_shape(&f, &[h]).expect("squarefree");
        assert_eq!(g, vec![vec![q(1), q(3)]]);
        // A repeated factor: f = (t − 1)².
        assert!(kronecker_to_shape(&[q(1), q(-2), q(1)], &[vec![q(1), q(0)]]).is_none());
    }

    #[test]
    fn kronecker_with_rational_coefficients() {
        // f = t³ − t/2 + 1/3, g = t²/5 − 7t + 2/9; h = g·f' mod f, then back.
        let q = |n: i64, d: i64| Rational::from((n, d));
        let f = vec![q(1, 3), q(-1, 2), q(0, 1), q(1, 1)];
        let g = vec![q(2, 9), q(-7, 1), q(1, 5)];
        // f' = 3t² − 1/2.  g·f' = 3/5 t⁴ − 21 t³ + (2/3 − 1/10) t² + 7/2 t − 1/9.
        let mut prod = vec![Rational::new(); 5];
        let fp = [q(-1, 2), q(0, 1), q(3, 1)];
        for (i, a) in g.iter().enumerate() {
            for (j, b) in fp.iter().enumerate() {
                prod[i + j] += Rational::from(a * b);
            }
        }
        for m in (3..5).rev() {
            let c = prod[m].clone();
            for j in 0..4 {
                prod[m - 3 + j] -= Rational::from(&c * &f[j]);
            }
        }
        prod.truncate(3);
        let back = kronecker_to_shape(&f, &[prod]).expect("squarefree");
        assert_eq!(back, vec![g]);
    }
}
