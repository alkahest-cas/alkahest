//! Monomial orderings for Gröbner basis computation.

use std::cmp::Ordering;

/// A monomial ordering for multivariate polynomials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonomialOrder {
    /// Lexicographic order (Lex): x > y > z, compare exponents left-to-right.
    Lex,
    /// Graded lexicographic order (GrLex): total degree first, then Lex.
    GrLex,
    /// Graded reverse lexicographic order (GRevLex): total degree first, then reverse Lex.
    GRevLex,
}

impl MonomialOrder {
    /// Compare two exponent vectors under this ordering.
    /// Exponent vectors are indexed by variable (index 0 = first variable).
    pub fn cmp(&self, a: &[u32], b: &[u32]) -> Ordering {
        match self {
            MonomialOrder::Lex => {
                for (ai, bi) in a.iter().zip(b.iter()) {
                    let c = ai.cmp(bi);
                    if c != Ordering::Equal {
                        return c;
                    }
                }
                a.len().cmp(&b.len())
            }
            MonomialOrder::GrLex => {
                let da: u32 = a.iter().sum();
                let db: u32 = b.iter().sum();
                match da.cmp(&db) {
                    Ordering::Equal => MonomialOrder::Lex.cmp(a, b),
                    c => c,
                }
            }
            MonomialOrder::GRevLex => {
                let da: u32 = a.iter().sum();
                let db: u32 = b.iter().sum();
                match da.cmp(&db) {
                    Ordering::Equal => {
                        for (ai, bi) in a.iter().rev().zip(b.iter().rev()) {
                            let c = bi.cmp(ai); // reversed!
                            if c != Ordering::Equal {
                                return c;
                            }
                        }
                        Ordering::Equal
                    }
                    c => c,
                }
            }
        }
    }

    /// True for graded orders (GrLex, GRevLex) where total degree is the primary key.
    ///
    /// For graded orders, LM(g) cannot divide a term of lower total degree, so the
    /// degree-skip optimization in reduction is sound.
    #[inline]
    pub fn is_graded(self) -> bool {
        matches!(self, MonomialOrder::GrLex | MonomialOrder::GRevLex)
    }

    /// Parse from a string: "lex", "grlex", "grevlex".
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "lex" => Some(MonomialOrder::Lex),
            "grlex" => Some(MonomialOrder::GrLex),
            "grevlex" | "degrevlex" => Some(MonomialOrder::GRevLex),
            _ => None,
        }
    }
}

impl Default for MonomialOrder {
    fn default() -> Self {
        MonomialOrder::GRevLex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_order() {
        let a = vec![2u32, 0, 0];
        let b = vec![0u32, 1, 0];
        assert_eq!(MonomialOrder::Lex.cmp(&a, &b), Ordering::Greater);
    }

    #[test]
    fn grlex_order() {
        let a = vec![2u32, 0];
        let b = vec![1u32, 1];
        assert_eq!(MonomialOrder::GrLex.cmp(&a, &b), Ordering::Greater);
    }

    #[test]
    fn grevlex_order() {
        let a = vec![2u32, 0, 0];
        let b = vec![0u32, 0, 2];
        assert_eq!(MonomialOrder::GRevLex.cmp(&a, &b), Ordering::Greater);
    }

    #[test]
    fn from_str_works() {
        assert_eq!(MonomialOrder::from_str("lex"), Some(MonomialOrder::Lex));
        assert_eq!(MonomialOrder::from_str("grlex"), Some(MonomialOrder::GrLex));
        assert_eq!(
            MonomialOrder::from_str("grevlex"),
            Some(MonomialOrder::GRevLex)
        );
        assert_eq!(
            MonomialOrder::from_str("degrevlex"),
            Some(MonomialOrder::GRevLex)
        );
        assert_eq!(MonomialOrder::from_str("unknown"), None);
    }
}
