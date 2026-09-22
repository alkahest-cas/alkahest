//! Exact Fincke–Pohst enumeration of lattice points inside a ball.
//!
//! This is the engine behind the shortest vector, the closest vector, the
//! minimal vectors and the theta series. It enumerates **every** `x ∈ ℤ^m` with
//! `Q(x − t) ≤ C` and nothing else — no pruning heuristic, no floating-point
//! Cholesky, no early exit on a "good enough" vector.
//!
//! # Why exact integers, and not a float Cholesky
//!
//! The usual implementation computes the Gram–Schmidt data in `double`s and
//! prunes with `sqrt`. The error then lives in the *pruning bound*, where its
//! effect is to drop lattice vectors near the boundary of the ball — which is
//! precisely where the interesting ones are (the whole content of "240 vectors
//! of norm 2" is a boundary count). A dropped vector is silent: the routine
//! returns a shorter list, or a longer "shortest" vector, with no diagnostic.
//!
//! So the admissible range for each coordinate is computed by integer square
//! root on exact rationals:
//!
//! ```text
//!     B_j · (x_j + u_j)² ≤ rem ,   B_j = p/q > 0 ,  u_j = a/b ,  rem = r/s ≥ 0
//! ⟺   (b·x_j + a)²  ≤  ⌊ (r·q·b²) / (s·p) ⌋            [both sides integers]
//! ⟺   |b·x_j + a|   ≤  ⌊√ … ⌋                          [`Integer::sqrt`, exact]
//! ```
//!
//! and the endpoints come out of floor/ceiling division. No rounding enters
//! anywhere.
//!
//! # Cost
//!
//! Exponential in the rank, as every exact SVP/CVP algorithm is. Callers cap
//! the rank ([`super::MAX_ENUM_RANK`]) and pass a node budget; exceeding either
//! is a typed refusal, never a truncated answer.

use super::error::LatticeGeometryError;
use rug::ops::DivRounding;
use rug::{Integer, Rational};

/// What the enumerator reports for one lattice point it found.
pub(crate) struct Found<'a> {
    /// Coefficients with respect to the basis the Gram–Schmidt data came from.
    pub(crate) x: &'a [Integer],
    /// `Q(x − t)`, exactly.
    pub(crate) norm: Rational,
}

/// Fincke–Pohst state: Gram–Schmidt data plus a node budget.
pub(crate) struct Enumerator {
    /// `mu[i][j]` for `j < i`.
    mu: Vec<Vec<Rational>>,
    /// `b[i] = ‖b*_i‖² > 0`.
    b: Vec<Rational>,
    /// `b[i]` split once into `(numerator, denominator)`, both positive. The
    /// per-node coordinate range is then pure integer arithmetic — no rational
    /// normalisation, which is a GCD per operation and dominates an enumeration
    /// of any size.
    b_split: Vec<(Integer, Integer)>,
    budget: u64,
    nodes: u64,
}

impl Enumerator {
    pub(crate) fn new(mu: Vec<Vec<Rational>>, b: Vec<Rational>, budget: u64) -> Self {
        let b_split = b
            .iter()
            .map(|v| (Integer::from(v.numer()), Integer::from(v.denom())))
            .collect();
        Self {
            mu,
            b,
            b_split,
            budget,
            nodes: 0,
        }
    }

    /// Visit every `x ∈ ℤ^m` with `Q(x − t) ≤ bound`.
    ///
    /// * `center` — `t`, in basis coordinates. `None` means the origin.
    /// * `half` — when set, only visit points whose **last non-zero**
    ///   coefficient is positive, plus the origin. For a centre-symmetric
    ///   search (`center = None`) the omitted points are exactly the negatives
    ///   of the visited ones, so a caller can double the counts. Meaningless,
    ///   and rejected by the caller, when `center` is set.
    pub(crate) fn enumerate<F>(
        &mut self,
        bound: &Rational,
        center: Option<&[Rational]>,
        half: bool,
        visit: &mut F,
    ) -> Result<(), LatticeGeometryError>
    where
        F: FnMut(Found<'_>),
    {
        let m = self.b.len();
        if m == 0 || *bound < 0 {
            return Ok(());
        }
        self.nodes = 0;
        let mut x = vec![Integer::new(); m];
        // `u[j]` accumulates `Σ_{i>j} mu[i][j] · (x_i − t_i)`, minus `t_j`
        // itself so the level test is a plain `(x_j + u_j)`.
        let mut u: Vec<Rational> = match center {
            Some(t) => t.iter().map(|v| -v.clone()).collect(),
            None => vec![Rational::new(); m],
        };
        self.rec(
            m - 1,
            bound.clone(),
            Rational::new(),
            &mut x,
            &mut u,
            center,
            half,
            visit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rec<F>(
        &mut self,
        level: usize,
        rem: Rational,
        used: Rational,
        x: &mut Vec<Integer>,
        u: &mut Vec<Rational>,
        center: Option<&[Rational]>,
        leading_free: bool,
        visit: &mut F,
    ) -> Result<(), LatticeGeometryError>
    where
        F: FnMut(Found<'_>),
    {
        self.nodes += 1;
        if self.nodes > self.budget {
            return Err(LatticeGeometryError::EnumerationBudget {
                budget: self.budget,
            });
        }

        let (lo, hi) = coordinate_range(&self.b_split[level], &u[level], &rem);
        let lo = if leading_free {
            lo.max(Integer::new())
        } else {
            lo
        };

        let mut xi = lo;
        while xi <= hi {
            // Exact term; `coordinate_range` is tight, so this never rejects,
            // but computing it is needed for `rem` anyway.
            let shifted = Rational::from(&xi) + &u[level];
            let term = Rational::from(&shifted * &shifted) * &self.b[level];
            if term <= rem {
                x[level] = xi.clone();
                if level == 0 {
                    // Accumulated exactly down the branch. Rational arithmetic
                    // does not round, so this is the same number
                    // `exact_norm` would recompute — which
                    // `tests::accumulated_norm_matches_recomputation` checks
                    // exhaustively on small lattices, rather than paying an
                    // O(m²) recomputation at every one of (for the Leech
                    // lattice) 98280 leaves.
                    let q = Rational::from(&used + &term);
                    visit(Found { x, norm: q });
                } else {
                    let next_rem = rem.clone() - &term;
                    let next_used = Rational::from(&used + &term);
                    // Push this coordinate into the running projections.
                    let delta_i = match center {
                        Some(t) => Rational::from(&xi) - &t[level],
                        None => Rational::from(&xi),
                    };
                    for (uj, m) in u.iter_mut().zip(self.mu[level].iter()).take(level) {
                        *uj += Rational::from(m * &delta_i);
                    }
                    let still_free = leading_free && xi == 0;
                    self.rec(
                        level - 1,
                        next_rem,
                        next_used,
                        x,
                        u,
                        center,
                        still_free,
                        visit,
                    )?;
                    for (uj, m) in u.iter_mut().zip(self.mu[level].iter()).take(level) {
                        *uj -= Rational::from(m * &delta_i);
                    }
                }
            }
            xi += 1;
        }
        x[level] = Integer::new();
        Ok(())
    }

    /// Babai's nearest-plane rounding of `t`, and the exact `Q(x − t)` it
    /// achieves.
    ///
    /// Used only as the *starting radius* for an exact CVP enumeration — the
    /// ball it defines provably contains a closest vector, because it contains
    /// the Babai vector itself. Nothing downstream trusts it to be optimal.
    pub(crate) fn babai_bound(&self, center: &[Rational]) -> Rational {
        let m = self.b.len();
        let mut x = vec![Integer::new(); m];
        let mut u: Vec<Rational> = center.iter().map(|v| -v.clone()).collect();
        for level in (0..m).rev() {
            let xi = super::quadform::round_rational(&-u[level].clone());
            x[level] = xi.clone();
            let delta_i = Rational::from(&xi) - &center[level];
            for (uj, m) in u.iter_mut().zip(self.mu[level].iter()).take(level) {
                *uj += Rational::from(m * &delta_i);
            }
        }
        self.exact_norm(&x, Some(center))
    }

    /// `Q(x − t)` recomputed from the Gram–Schmidt data, independently of the
    /// running residual — so an accounting slip in the recursion cannot reach a
    /// caller as a wrong norm.
    fn exact_norm(&self, x: &[Integer], center: Option<&[Rational]>) -> Rational {
        let m = self.b.len();
        let mut acc = Rational::new();
        for j in 0..m {
            let mut s = match center {
                Some(t) => Rational::from(&x[j]) - &t[j],
                None => Rational::from(&x[j]),
            };
            for i in (j + 1)..m {
                let di = match center {
                    Some(t) => Rational::from(&x[i]) - &t[i],
                    None => Rational::from(&x[i]),
                };
                s += Rational::from(&self.mu[i][j] * &di);
            }
            acc += Rational::from(&s * &s) * &self.b[j];
        }
        acc
    }
}

/// The exact set of integers `x` with `b·(x + u)² ≤ rem`, as an inclusive range.
///
/// `b = bp/bq > 0` and `rem ≥ 0` are preconditions. When the range is empty the
/// returned `lo` exceeds `hi`. Everything below is integer arithmetic:
///
/// ```text
///   (bp/bq)·(x + a/d)² ≤ r/s
/// ⟺ (d·x + a)²         ≤ (r·bq·d²) / (s·bp)
/// ⟺ |d·x + a|          ≤ ⌊√⌊(r·bq·d²)/(s·bp)⌋⌋
/// ```
///
/// the inner floor being free because `(d·x + a)²` is an integer.
fn coordinate_range(b: &(Integer, Integer), u: &Rational, rem: &Rational) -> (Integer, Integer) {
    let empty = (Integer::from(1), Integer::new());
    if *rem < 0 {
        return empty;
    }
    let (bp, bq) = b;
    debug_assert!(*bp > 0 && *bq > 0);
    let a = Integer::from(u.numer());
    let d = Integer::from(u.denom()); // > 0 by rug's normalisation
    let num = Integer::from(rem.numer()) * bq * &d * &d;
    let den = Integer::from(rem.denom()) * bp;
    let floor = num.div_floor(den);
    if floor < 0 {
        return empty;
    }
    let root = floor.sqrt(); // ⌊√·⌋, exact
    let lo = (Integer::from(-&root) - &a).div_ceil(d.clone());
    let hi = (root - &a).div_floor(d);
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinate_range_is_exact_on_the_boundary() {
        // b = 1, u = 0, rem = 4  ⟹  x² ≤ 4  ⟹  −2 ..= 2 (boundary included).
        let one = (Integer::from(1), Integer::from(1));
        let two = (Integer::from(2), Integer::from(1));
        let nine = (Integer::from(9), Integer::from(1));
        let (lo, hi) = coordinate_range(&one, &Rational::new(), &Rational::from(4));
        assert_eq!((lo, hi), (Integer::from(-2), Integer::from(2)));

        // b = 2, u = 1/2, rem = 2  ⟹  2(x + 1/2)² ≤ 2 ⟹ |x + 1/2| ≤ 1 ⟹ −1..=0.
        let (lo, hi) = coordinate_range(&two, &Rational::from((1, 2)), &Rational::from(2));
        assert_eq!((lo, hi), (Integer::from(-1), Integer::from(0)));

        // Just under the boundary: 2(x)² ≤ 7/4 ⟹ x² ≤ 7/8 ⟹ only x = 0.
        let (lo, hi) = coordinate_range(&two, &Rational::new(), &Rational::from((7, 4)));
        assert_eq!((lo, hi), (Integer::from(0), Integer::from(0)));

        // Empty.
        let (lo, hi) = coordinate_range(&nine, &Rational::from((1, 2)), &Rational::from(1));
        assert!(lo > hi);
    }

    /// The norm reported at a leaf is accumulated down the branch rather than
    /// recomputed, which is worth several times the run time on a rank-24
    /// enumeration. This is the check that the two agree — over every point of
    /// several forms, centred and uncentred.
    #[test]
    fn accumulated_norm_matches_recomputation() {
        // A_2, D_2-ish and a skew form, as (mu, b) pairs derived by hand-rolled
        // Gram–Schmidt from the Gram matrix.
        let forms: Vec<Vec<Vec<Rational>>> = vec![
            vec![
                vec![Rational::from(2), Rational::from(-1)],
                vec![Rational::from(-1), Rational::from(2)],
            ],
            vec![
                vec![Rational::from(5), Rational::from(3)],
                vec![Rational::from(3), Rational::from(4)],
            ],
            vec![
                vec![Rational::from(1), Rational::new(), Rational::new()],
                vec![Rational::new(), Rational::from(2), Rational::from(1)],
                vec![Rational::new(), Rational::from(1), Rational::from(3)],
            ],
        ];
        let centers: Vec<Option<Vec<Rational>>> = vec![
            None,
            Some(vec![Rational::from((1, 3)), Rational::from((1, 2))]),
        ];
        for g in &forms {
            let (mu, b) = crate::lattice::quadform::gram_schmidt(g).unwrap();
            for c in &centers {
                if let Some(t) = c {
                    if t.len() != g.len() {
                        continue;
                    }
                }
                let mut e = Enumerator::new(mu.clone(), b.clone(), 1_000_000);
                let reference = Enumerator::new(mu.clone(), b.clone(), 1);
                let mut checked = 0usize;
                e.enumerate(&Rational::from(12), c.as_deref(), false, &mut |found| {
                    let want = reference.exact_norm(found.x, c.as_deref());
                    assert_eq!(found.norm, want, "x = {:?}", found.x);
                    checked += 1;
                })
                .unwrap();
                assert!(checked > 0);
            }
        }
    }

    #[test]
    fn budget_is_a_refusal_not_a_truncation() {
        // Z^3, bound 100 — far more nodes than the budget of 3.
        let mu = vec![vec![Rational::new(); 3]; 3];
        let b = vec![Rational::from(1), Rational::from(1), Rational::from(1)];
        let mut e = Enumerator::new(mu, b, 3);
        let mut seen = 0usize;
        let err = e.enumerate(&Rational::from(100), None, false, &mut |_| seen += 1);
        assert!(matches!(
            err,
            Err(LatticeGeometryError::EnumerationBudget { .. })
        ));
    }
}
