//! Discrimination-net index for pattern-rule lookup.
//!
//! Rules whose left-hand sides share no root constructor (Add vs Mul vs Pow, …)
//! are *match-disjoint*: only rules whose pattern head matches the expression
//! head at the current node need to be tried.  This reduces rule scanning from
//! O(|rules|) to O(|rules at head|) per node.

use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// Root constructor of a pattern or expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatternHead {
    /// Wildcard-only or non-constructor root (always considered).
    Any,
    Add,
    Mul,
    Pow,
    Func(String),
    Integer,
    Symbol(String),
}

/// Index mapping expression heads to candidate rule indices.
#[derive(Clone, Debug, Default)]
pub struct DiscriminationIndex {
    by_head: HashMap<PatternHead, Vec<usize>>,
    /// Rules with `PatternHead::Any` — tried after head-specific candidates.
    universal: Vec<usize>,
}

impl DiscriminationIndex {
    /// Build an index from per-rule pattern heads (one head per rule, in order).
    pub fn build(heads: impl IntoIterator<Item = PatternHead>) -> Self {
        let mut by_head: HashMap<PatternHead, Vec<usize>> = HashMap::new();
        let mut universal = Vec::new();
        for (i, head) in heads.into_iter().enumerate() {
            if head == PatternHead::Any {
                universal.push(i);
            } else {
                by_head.entry(head).or_default().push(i);
            }
        }
        DiscriminationIndex { by_head, universal }
    }

    /// Rule indices to try for `expr`, in order (head-specific then universal).
    pub fn candidates<'a>(&'a self, expr: ExprId, pool: &ExprPool) -> CandidateRules<'a> {
        let head = expr_head(expr, pool);
        let specific = self.by_head.get(&head).map(|v| v.as_slice()).unwrap_or(&[]);
        CandidateRules {
            specific,
            universal: &self.universal,
            phase: 0,
            i: 0,
        }
    }
}

/// Iterator over rule indices for a single simplification step.
pub struct CandidateRules<'a> {
    specific: &'a [usize],
    universal: &'a [usize],
    phase: u8,
    i: usize,
}

impl<'a> Iterator for CandidateRules<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.phase == 0 {
            if self.i < self.specific.len() {
                let idx = self.specific[self.i];
                self.i += 1;
                return Some(idx);
            }
            self.phase = 1;
            self.i = 0;
        }
        if self.i < self.universal.len() {
            let idx = self.universal[self.i];
            self.i += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Head constructor of an expression node.
pub fn expr_head(expr: ExprId, pool: &ExprPool) -> PatternHead {
    match pool.get(expr) {
        ExprData::Add(_) => PatternHead::Add,
        ExprData::Mul(_) => PatternHead::Mul,
        ExprData::Pow { .. } => PatternHead::Pow,
        ExprData::Func { name, .. } => PatternHead::Func(name),
        ExprData::Integer(_) => PatternHead::Integer,
        ExprData::Symbol { name, .. } => PatternHead::Symbol(name),
        _ => PatternHead::Any,
    }
}

/// Head constructor of a pattern's root node.
pub fn pattern_head(pat: ExprId, pool: &ExprPool) -> PatternHead {
    match pool.get(pat) {
        ExprData::Add(_) => PatternHead::Add,
        ExprData::Mul(_) => PatternHead::Mul,
        ExprData::Pow { .. } => PatternHead::Pow,
        ExprData::Func { name, .. } => PatternHead::Func(name),
        ExprData::Integer(_) => PatternHead::Integer,
        ExprData::Symbol { name, .. } if is_wildcard(&name) => PatternHead::Any,
        ExprData::Symbol { name, .. } => PatternHead::Symbol(name.clone()),
        _ => PatternHead::Any,
    }
}

fn is_wildcard(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_only_tries_matching_head() {
        let pool = ExprPool::new();
        let index = DiscriminationIndex::build([PatternHead::Add, PatternHead::Mul]);
        let x = pool.symbol("x", crate::kernel::Domain::Real);
        let add_expr = pool.add(vec![x, x]);
        let candidates: Vec<_> = index.candidates(add_expr, &pool).collect();
        assert_eq!(candidates, vec![0]);
        let mul_expr = pool.mul(vec![x, pool.integer(2_i32)]);
        let candidates: Vec<_> = index.candidates(mul_expr, &pool).collect();
        assert_eq!(candidates, vec![1]);
    }
}
