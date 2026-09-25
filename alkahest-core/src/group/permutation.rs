//! A permutation of `{0, 1, …, n−1}`, stored as an images array.

use super::GroupError;
use rug::Integer;
use std::fmt;

/// A bijection of the point set `{0, 1, …, degree−1}`.
///
/// The representation is the images array: `images()[i]` is the image of point
/// `i`. Equality, ordering and hashing are equality, ordering and hashing of
/// that array, so two permutations of **different degrees are never equal**
/// even when one is the other padded with fixed points — see
/// [`extend_degree`](Permutation::extend_degree).
///
/// # Composition order
///
/// [`compose`](Permutation::compose) is **left-to-right**: `p.compose(&q)`
/// applies `p` first, then `q`. See the [module docs](super) for why, and for
/// what goes wrong if this is mixed with the analysis convention.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Permutation {
    images: Vec<usize>,
}

impl Permutation {
    /// The identity permutation of `degree` points.
    pub fn identity(degree: usize) -> Permutation {
        Permutation {
            images: (0..degree).collect(),
        }
    }

    /// Build from an images array: `images[i]` is the image of point `i`.
    ///
    /// Refuses ([`GroupError::NotAPermutation`]) unless the array is a
    /// bijection of `{0, …, n−1}` — every entry in range, no repeats.
    ///
    /// ```
    /// use alkahest_cas::experimental::Permutation;
    /// let p = Permutation::from_images(vec![1, 0, 2]).unwrap();
    /// assert_eq!(p.degree(), 3);
    /// assert!(Permutation::from_images(vec![0, 0]).is_err());
    /// ```
    pub fn from_images(images: Vec<usize>) -> Result<Permutation, GroupError> {
        let degree = images.len();
        let mut seen = vec![false; degree];
        for (point, &image) in images.iter().enumerate() {
            if image >= degree {
                return Err(GroupError::NotAPermutation {
                    degree,
                    reason: format!("image of {point} is {image}, which is not below {degree}"),
                });
            }
            if seen[image] {
                return Err(GroupError::NotAPermutation {
                    degree,
                    reason: format!("{image} is the image of two different points"),
                });
            }
            seen[image] = true;
        }
        Ok(Permutation { images })
    }

    /// Build from **0-based** disjoint cycles. Points not named are fixed.
    ///
    /// Cycles of length 0 or 1 are allowed and contribute nothing. A point
    /// named twice — within one cycle or across two — is
    /// [`GroupError::NotAPermutation`]: the cycles must be disjoint, because a
    /// non-disjoint list is ambiguous about the intended product order.
    ///
    /// ```
    /// use alkahest_cas::experimental::Permutation;
    /// // (0 1 2) as a permutation of 4 points.
    /// let p = Permutation::from_cycles(4, &[vec![0, 1, 2]]).unwrap();
    /// assert_eq!(p.images(), &[1, 2, 0, 3]);
    /// ```
    pub fn from_cycles(degree: usize, cycles: &[Vec<usize>]) -> Result<Permutation, GroupError> {
        let mut images: Vec<usize> = (0..degree).collect();
        let mut used = vec![false; degree];
        for cycle in cycles {
            for &point in cycle {
                if point >= degree {
                    return Err(GroupError::NotAPermutation {
                        degree,
                        reason: format!("cycle names point {point}, which is not below {degree}"),
                    });
                }
                if used[point] {
                    return Err(GroupError::NotAPermutation {
                        degree,
                        reason: format!("point {point} appears in more than one cycle position"),
                    });
                }
                used[point] = true;
            }
            if cycle.len() < 2 {
                continue;
            }
            for window in cycle.windows(2) {
                images[window[0]] = window[1];
            }
            images[cycle[cycle.len() - 1]] = cycle[0];
        }
        // Built by construction, but validate anyway: the check is O(n) and
        // this is the constructor most likely to be fed hand-typed input.
        Permutation::from_images(images)
    }

    /// Build from **1-based** disjoint cycles on the points `1..=degree`.
    ///
    /// GAP, the ATLAS of Finite Groups and essentially every paper number
    /// points from 1. This constructor exists so that a generator can be
    /// transcribed character for character; it subtracts one from every point
    /// and is otherwise [`from_cycles`](Permutation::from_cycles). Note that
    /// `degree` still means *how many points there are*, so the ATLAS's degree-11
    /// generators for `M₁₁` are `from_cycles_one_based(11, …)`.
    ///
    /// ```
    /// use alkahest_cas::experimental::Permutation;
    /// let p = Permutation::from_cycles_one_based(4, &[vec![1, 2, 3]]).unwrap();
    /// assert_eq!(p.images(), &[1, 2, 0, 3]);
    /// ```
    pub fn from_cycles_one_based(
        degree: usize,
        cycles: &[Vec<usize>],
    ) -> Result<Permutation, GroupError> {
        let mut zero_based = Vec::with_capacity(cycles.len());
        for cycle in cycles {
            let mut c = Vec::with_capacity(cycle.len());
            for &point in cycle {
                if point == 0 || point > degree {
                    return Err(GroupError::NotAPermutation {
                        degree,
                        reason: format!(
                            "1-based cycle names point {point}, which is outside 1..={degree}"
                        ),
                    });
                }
                c.push(point - 1);
            }
            zero_based.push(c);
        }
        Permutation::from_cycles(degree, &zero_based)
    }

    /// The number of points this permutation acts on.
    pub fn degree(&self) -> usize {
        self.images.len()
    }

    /// The images array: `images()[i]` is the image of `i`.
    pub fn images(&self) -> &[usize] {
        &self.images
    }

    /// The image of `point`, or [`GroupError::PointOutOfRange`].
    pub fn apply(&self, point: usize) -> Result<usize, GroupError> {
        self.images
            .get(point)
            .copied()
            .ok_or(GroupError::PointOutOfRange {
                point,
                degree: self.degree(),
            })
    }

    /// The image of `point`, without the range check.
    #[inline]
    pub(crate) fn img(&self, point: usize) -> usize {
        self.images[point]
    }

    /// Re-read this permutation as one of `degree` points, fixing the added
    /// ones.
    ///
    /// Refuses ([`GroupError::DegreeMismatch`]) if `degree` is smaller than the
    /// current degree; a permutation is never *restricted* silently, because
    /// the restriction need not be a permutation of the smaller set.
    pub fn extend_degree(&self, degree: usize) -> Result<Permutation, GroupError> {
        if degree < self.degree() {
            return Err(GroupError::DegreeMismatch {
                left: self.degree(),
                right: degree,
            });
        }
        let mut images = self.images.clone();
        images.extend(self.degree()..degree);
        Ok(Permutation { images })
    }

    /// **Left-to-right** composition: apply `self`, then `other`.
    ///
    /// `(self ∘ other)(i) = other(self(i))`, i.e. `i^(p·q) = (i^p)^q`.
    ///
    /// ```
    /// use alkahest_cas::experimental::Permutation;
    /// let p = Permutation::from_cycles(3, &[vec![0, 1]]).unwrap();     // (0 1)
    /// let q = Permutation::from_cycles(3, &[vec![1, 2]]).unwrap();     // (1 2)
    /// // p then q: 0 -> 1 -> 2.
    /// assert_eq!(p.compose(&q).unwrap().images(), &[2, 0, 1]);
    /// // q then p: 0 -> 0 -> 1.  Composition is not commutative.
    /// assert_eq!(q.compose(&p).unwrap().images(), &[1, 2, 0]);
    /// ```
    pub fn compose(&self, other: &Permutation) -> Result<Permutation, GroupError> {
        if self.degree() != other.degree() {
            return Err(GroupError::DegreeMismatch {
                left: self.degree(),
                right: other.degree(),
            });
        }
        Ok(self.compose_unchecked(other))
    }

    /// [`compose`](Permutation::compose) with the degree check already done.
    #[inline]
    pub(crate) fn compose_unchecked(&self, other: &Permutation) -> Permutation {
        Permutation {
            images: self.images.iter().map(|&i| other.images[i]).collect(),
        }
    }

    /// The inverse permutation. `p.compose(&p.inverse())` is the identity.
    pub fn inverse(&self) -> Permutation {
        let mut images = vec![0usize; self.degree()];
        for (point, &image) in self.images.iter().enumerate() {
            images[image] = point;
        }
        Permutation { images }
    }

    /// `self` raised to an integer power, positive, zero or negative.
    pub fn pow(&self, exponent: i64) -> Permutation {
        let mut base = if exponent < 0 {
            self.inverse()
        } else {
            self.clone()
        };
        let mut k = exponent.unsigned_abs();
        let mut acc = Permutation::identity(self.degree());
        while k > 0 {
            if k & 1 == 1 {
                acc = acc.compose_unchecked(&base);
            }
            base = base.compose_unchecked(&base);
            k >>= 1;
        }
        acc
    }

    /// Is this the identity?
    pub fn is_identity(&self) -> bool {
        self.images.iter().enumerate().all(|(i, &j)| i == j)
    }

    /// The points this permutation moves, ascending.
    pub fn support(&self) -> Vec<usize> {
        self.images
            .iter()
            .enumerate()
            .filter(|(i, &j)| *i != j)
            .map(|(i, _)| i)
            .collect()
    }

    /// The least moved point, or `None` for the identity.
    pub fn first_moved_point(&self) -> Option<usize> {
        self.images
            .iter()
            .enumerate()
            .find(|(i, &j)| *i != j)
            .map(|(i, _)| i)
    }

    /// The non-trivial cycles, each written from its least point, ordered by
    /// that least point.
    ///
    /// Fixed points are **omitted** — the identity has no cycles at all. Use
    /// [`cycle_type`](Permutation::cycle_type) for the full multiset of cycle
    /// lengths, which includes the 1-cycles and partitions the degree.
    ///
    /// ```
    /// use alkahest_cas::experimental::Permutation;
    /// let p = Permutation::from_images(vec![1, 0, 2, 4, 5, 3]).unwrap();
    /// assert_eq!(p.cycles(), vec![vec![0, 1], vec![3, 4, 5]]);
    /// assert_eq!(p.cycle_type(), vec![3, 2, 1]);
    /// ```
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.degree()];
        let mut out = Vec::new();
        for start in 0..self.degree() {
            if seen[start] {
                continue;
            }
            let mut cycle = vec![start];
            seen[start] = true;
            let mut next = self.images[start];
            while next != start {
                seen[next] = true;
                cycle.push(next);
                next = self.images[next];
            }
            if cycle.len() > 1 {
                out.push(cycle);
            }
        }
        out
    }

    /// The multiset of cycle lengths **including fixed points**, descending.
    ///
    /// Sums to [`degree`](Permutation::degree), so it is a partition of `n`.
    pub fn cycle_type(&self) -> Vec<usize> {
        let mut seen = vec![false; self.degree()];
        let mut lengths = Vec::new();
        for start in 0..self.degree() {
            if seen[start] {
                continue;
            }
            let mut len = 1;
            seen[start] = true;
            let mut next = self.images[start];
            while next != start {
                seen[next] = true;
                len += 1;
                next = self.images[next];
            }
            lengths.push(len);
        }
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths
    }

    /// The multiplicative order of this element: the least `k > 0` with
    /// `p^k = 1`, as the lcm of its cycle lengths.
    ///
    /// Arbitrary precision on purpose. The largest order in `S_n` is Landau's
    /// function `g(n)`, which passes `u64` a little after `n = 180`, so a
    /// machine integer here would be a silent overflow waiting for a large
    /// enough degree.
    pub fn order(&self) -> Integer {
        let mut order = Integer::from(1);
        for length in self.cycle_type() {
            if length > 1 {
                order.lcm_mut(&Integer::from(length));
            }
        }
        order
    }

    /// The sign: `+1` for an even permutation, `−1` for an odd one.
    ///
    /// `sign(p·q) = sign(p)·sign(q)` — asserted as a property test.
    pub fn sign(&self) -> i32 {
        let mut parity = 0usize;
        for length in self.cycle_type() {
            parity += length - 1;
        }
        if parity % 2 == 0 {
            1
        } else {
            -1
        }
    }

    /// Is this permutation even?
    pub fn is_even(&self) -> bool {
        self.sign() == 1
    }
}

impl fmt::Display for Permutation {
    /// Cycle notation with **0-based** points; `()` for the identity.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cycles = self.cycles();
        if cycles.is_empty() {
            return write!(f, "()");
        }
        for cycle in &cycles {
            write!(f, "(")?;
            for (i, point) in cycle.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{point}")?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}
