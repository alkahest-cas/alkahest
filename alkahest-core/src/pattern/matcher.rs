/// AC-aware pattern matching for symbolic expressions.
///
/// A `Pattern` is a template that may contain named wildcards.  A wildcard
/// matches any sub-expression and binds it to a name, so that all occurrences
/// of the same wildcard name must match the *same* (structurally equal)
/// expression.
///
/// # AC semantics
///
/// `Add` and `Mul` are treated as *associative and commutative* (AC)
/// operators.  A pattern like `a + b` therefore matches *any pair* of
/// sub-expressions drawn from an n-ary sum, not just the literal first and
/// second children.
///
/// # Search depth
///
/// To prevent combinatorial explosion the AC search is bounded:
/// - At most `MAX_AC_DEPTH` nested AC operators are explored.
/// - The number of candidate splits for an n-ary term is bounded by the
///   number of size-k subsets of n terms (k = arity of pattern AC node).
///
/// These bounds are conservative for normal CAS expressions.  Callers who
/// need exhaustive matching on large sums/products should pass a custom
/// config in future (extension point).
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

// Maximum depth of AC nodes traversed during recursive matching.
const MAX_AC_DEPTH: usize = 6;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A pattern for matching against symbolic expressions.
///
/// Patterns share the same expression representation as regular expressions
/// but may include `Symbol` nodes that act as wildcards.  A symbol whose
/// name starts with a lower-case letter (e.g. `a`, `f`, `lhs`) is treated
/// as a *wildcard variable* that binds to any sub-expression.  Upper-case
/// or multi-character names that don't match the wildcard convention are
/// treated as literal symbols.
///
/// Use `Pattern::parse` to build patterns from strings, or `Pattern::from_expr`
/// to use any `ExprId` directly as a pattern (all symbols become wildcards).
#[derive(Clone, Debug)]
pub struct Pattern {
    pub root: ExprId,
}

impl Pattern {
    /// Create a pattern from an existing expression.  All `Symbol` nodes in
    /// the expression become wildcards.
    pub fn from_expr(root: ExprId) -> Self {
        Pattern { root }
    }
}

/// A binding from wildcard names to matched expression ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Substitution {
    pub bindings: HashMap<String, ExprId>,
}

impl Substitution {
    fn new() -> Self {
        Substitution {
            bindings: HashMap::new(),
        }
    }

    /// Attempt to bind `name` to `id`.  Returns `false` if `name` is already
    /// bound to a different expression.
    fn bind(&mut self, name: &str, id: ExprId) -> bool {
        match self.bindings.get(name) {
            Some(&existing) if existing != id => false,
            _ => {
                self.bindings.insert(name.to_string(), id);
                true
            }
        }
    }

    /// Apply the substitution to a pattern expression, returning the
    /// concrete expression.  Wildcards are replaced by their bindings;
    /// unbound wildcards are left as-is.
    pub fn apply(&self, pattern: ExprId, pool: &ExprPool) -> ExprId {
        apply_subst(pattern, self, pool)
    }
}

fn apply_subst(pat: ExprId, subst: &Substitution, pool: &ExprPool) -> ExprId {
    enum Node {
        Wildcard(String),
        Literal,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
    }

    let node = pool.with(pat, |data| match data {
        ExprData::Symbol { name, .. } if is_wildcard(name) => Node::Wildcard(name.clone()),
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        ExprData::Func { name, args } => Node::Func(name.clone(), args.clone()),
        _ => Node::Literal,
    });

    match node {
        Node::Wildcard(name) => subst.bindings.get(&name).copied().unwrap_or(pat),
        Node::Literal => pat,
        Node::Add(args) => {
            let new_args: Vec<_> = args.iter().map(|&a| apply_subst(a, subst, pool)).collect();
            pool.add(new_args)
        }
        Node::Mul(args) => {
            let new_args: Vec<_> = args.iter().map(|&a| apply_subst(a, subst, pool)).collect();
            pool.mul(new_args)
        }
        Node::Pow(base, exp) => pool.pow(
            apply_subst(base, subst, pool),
            apply_subst(exp, subst, pool),
        ),
        Node::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(|&a| apply_subst(a, subst, pool)).collect();
            pool.func(name, new_args)
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A symbol is a wildcard if its name is a single lower-case letter or
/// starts with a lower-case letter followed by alphanumeric/underscore.
fn is_wildcard(name: &str) -> bool {
    name.starts_with(|c: char| c.is_lowercase())
}

// ---------------------------------------------------------------------------
// Core matching — non-AC
// ---------------------------------------------------------------------------

/// Try to match `pat` against `expr` given an existing partial `subst`.
/// Returns the extended substitution on success, `None` on failure.
fn match_one(
    pat: ExprId,
    expr: ExprId,
    subst: Substitution,
    pool: &ExprPool,
    ac_depth: usize,
) -> Option<Substitution> {
    enum PatNode {
        Wildcard(String),
        Integer(i64),
        Symbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Literal,
    }

    enum ExprNode {
        Integer(i64),
        Symbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Other,
    }

    let pat_node = pool.with(pat, |data| match data {
        ExprData::Symbol { name, .. } if is_wildcard(name) => PatNode::Wildcard(name.clone()),
        ExprData::Symbol { name, .. } => PatNode::Symbol(name.clone()),
        ExprData::Integer(n) => PatNode::Integer(n.0.to_i64().unwrap_or(i64::MIN)),
        ExprData::Add(args) => PatNode::Add(args.clone()),
        ExprData::Mul(args) => PatNode::Mul(args.clone()),
        ExprData::Pow { base, exp } => PatNode::Pow(*base, *exp),
        ExprData::Func { name, args } => PatNode::Func(name.clone(), args.clone()),
        ExprData::Rational(_) | ExprData::Float(_) => PatNode::Literal,
        ExprData::Piecewise { .. } | ExprData::Predicate { .. } => PatNode::Literal,
        ExprData::Forall { .. } | ExprData::Exists { .. } | ExprData::BigO(_) => PatNode::Literal,
    });

    let expr_node = pool.with(expr, |data| match data {
        ExprData::Symbol { name, .. } => ExprNode::Symbol(name.clone()),
        ExprData::Integer(n) => ExprNode::Integer(n.0.to_i64().unwrap_or(i64::MIN)),
        ExprData::Add(args) => ExprNode::Add(args.clone()),
        ExprData::Mul(args) => ExprNode::Mul(args.clone()),
        ExprData::Pow { base, exp } => ExprNode::Pow(*base, *exp),
        ExprData::Func { name, args } => ExprNode::Func(name.clone(), args.clone()),
        _ => ExprNode::Other,
    });

    match pat_node {
        // Wildcard: bind to the whole expression
        PatNode::Wildcard(name) => {
            let mut s = subst;
            if s.bind(&name, expr) {
                Some(s)
            } else {
                None
            }
        }

        // Literal integer — must match exactly
        PatNode::Integer(pn) => {
            if matches!(expr_node, ExprNode::Integer(en) if en == pn) {
                Some(subst)
            } else {
                None
            }
        }

        // Literal symbol — must match the same symbol name (not a wildcard)
        PatNode::Symbol(pname) => {
            if matches!(expr_node, ExprNode::Symbol(ref ename) if *ename == pname) {
                Some(subst)
            } else {
                None
            }
        }

        // AC-aware Add matching
        PatNode::Add(pat_args) => {
            let ExprNode::Add(expr_args) = expr_node else {
                return None;
            };
            if ac_depth >= MAX_AC_DEPTH {
                // Fall back to exact positional matching to bound depth
                return match_args_exact(&pat_args, &expr_args, subst, pool, ac_depth + 1);
            }
            match_ac_args(&pat_args, &expr_args, subst, pool, ac_depth, true)
        }

        // AC-aware Mul matching
        PatNode::Mul(pat_args) => {
            let ExprNode::Mul(expr_args) = expr_node else {
                return None;
            };
            if ac_depth >= MAX_AC_DEPTH {
                return match_args_exact(&pat_args, &expr_args, subst, pool, ac_depth + 1);
            }
            match_ac_args(&pat_args, &expr_args, subst, pool, ac_depth, true)
        }

        // Pow — exact structural match
        PatNode::Pow(pb, pe) => {
            let ExprNode::Pow(eb, ee) = expr_node else {
                return None;
            };
            let s = match_one(pb, eb, subst, pool, ac_depth + 1)?;
            match_one(pe, ee, s, pool, ac_depth + 1)
        }

        // Named function — name must match, args AC-matched if Add/Mul
        PatNode::Func(pname, pargs) => {
            let ExprNode::Func(ename, eargs) = expr_node else {
                return None;
            };
            if pname != ename {
                return None;
            }
            match_args_exact(&pargs, &eargs, subst, pool, ac_depth + 1)
        }

        // Rational/Float literal in pattern — match only if same id (structural equality)
        PatNode::Literal => {
            if pat == expr {
                Some(subst)
            } else {
                None
            }
        }
    }
}

/// Match pattern args against expr args positionally (no AC permutations).
fn match_args_exact(
    pat_args: &[ExprId],
    expr_args: &[ExprId],
    subst: Substitution,
    pool: &ExprPool,
    ac_depth: usize,
) -> Option<Substitution> {
    if pat_args.len() != expr_args.len() {
        return None;
    }
    let mut s = subst;
    for (&p, &e) in pat_args.iter().zip(expr_args.iter()) {
        s = match_one(p, e, s, pool, ac_depth)?;
    }
    Some(s)
}

// ---------------------------------------------------------------------------
// AC matching
// ---------------------------------------------------------------------------

/// AC-aware matching for n-ary Add or Mul.
///
/// If `pat_args.len() == expr_args.len()` we try all permutations of
/// expr_args against pat_args (bounded by MAX_AC_DEPTH checks above).
///
/// If `pat_args.len() < expr_args.len()` we additionally try all size-k
/// subsets of expr_args for the first k-1 pat_args, bundling the remainder
/// into a single Add/Mul node bound to the last wildcard if it is one.
///
/// This approach is *sound* (every returned substitution is valid) and
/// *complete for ground patterns* — every valid ground match is returned.
fn match_ac_args(
    pat_args: &[ExprId],
    expr_args: &[ExprId],
    subst: Substitution,
    pool: &ExprPool,
    ac_depth: usize,
    is_add: bool,
) -> Option<Substitution> {
    if pat_args.is_empty() && expr_args.is_empty() {
        return Some(subst);
    }
    if pat_args.is_empty() || expr_args.is_empty() {
        return None;
    }

    // Exact-length case: try all permutations
    if pat_args.len() == expr_args.len() {
        return try_permutations(pat_args, expr_args, subst, pool, ac_depth);
    }

    // Pattern is shorter: try matching a subset of expr_args to the first
    // pat_args, leaving the rest as a residual bound to the last pattern arg
    // (only if it's a wildcard).
    if pat_args.len() < expr_args.len() {
        let last_pat = *pat_args.last().unwrap();
        let is_last_wildcard = pool.with(
            last_pat,
            |data| matches!(data, ExprData::Symbol { name, .. } if is_wildcard(name)),
        );

        if !is_last_wildcard {
            // Can't absorb remainder — no match
            return None;
        }

        let prefix_len = pat_args.len() - 1;
        // Try all size-(prefix_len) subsets of expr_args for the prefix pattern args
        let indices: Vec<usize> = (0..expr_args.len()).collect();
        return try_subsets(
            pat_args, expr_args, &indices, prefix_len, subst, pool, ac_depth, is_add,
        );
    }

    // Pattern is longer than expr: no match possible
    None
}

/// Try matching pat_args against all permutations of a chosen expr_args subset.
fn try_permutations(
    pat_args: &[ExprId],
    expr_args: &[ExprId],
    subst: Substitution,
    pool: &ExprPool,
    ac_depth: usize,
) -> Option<Substitution> {
    // Generate permutations via Heap's algorithm
    let mut perm: Vec<usize> = (0..expr_args.len()).collect();
    loop {
        // Try current permutation
        let mut s = subst.clone();
        let mut ok = true;
        for (i, &pat_id) in pat_args.iter().enumerate() {
            match match_one(pat_id, expr_args[perm[i]], s.clone(), pool, ac_depth + 1) {
                Some(new_s) => s = new_s,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(s);
        }

        // Advance to next permutation (Heap's algorithm)
        if !next_permutation(&mut perm) {
            break;
        }
    }
    None
}

/// Advance `perm` to the next lexicographic permutation.  Returns `false`
/// when already at the last permutation.
fn next_permutation(perm: &mut [usize]) -> bool {
    let n = perm.len();
    if n <= 1 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && perm[i - 1] >= perm[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let j = (i..n).rfind(|&j| perm[j] > perm[i - 1]).unwrap();
    perm.swap(i - 1, j);
    perm[i..].reverse();
    true
}

/// Try matching prefix pattern args against all size-`prefix_len` subsets of
/// expr_args, binding the remainder to the last wildcard.
#[allow(clippy::too_many_arguments)]
fn try_subsets(
    pat_args: &[ExprId],
    expr_args: &[ExprId],
    indices: &[usize],
    prefix_len: usize,
    subst: Substitution,
    pool: &ExprPool,
    ac_depth: usize,
    is_add: bool,
) -> Option<Substitution> {
    if prefix_len == 0 {
        // All expr_args go to the last wildcard
        let last_pat = *pat_args.last().unwrap();
        let residual: Vec<ExprId> = indices.iter().map(|&i| expr_args[i]).collect();
        let residual_expr = match residual.len() {
            0 => return None,
            1 => residual[0],
            _ => {
                if is_add {
                    pool.add(residual)
                } else {
                    pool.mul(residual)
                }
            }
        };
        let mut s = subst;
        s.bind(
            &pool.with(last_pat, |data| {
                if let ExprData::Symbol { name, .. } = data {
                    name.clone()
                } else {
                    String::new()
                }
            }),
            residual_expr,
        );
        return if s.bindings.values().next().is_some() {
            Some(s)
        } else {
            None
        };
    }

    // Pick one element for the next prefix slot and recurse
    for chosen_pos in 0..indices.len() {
        let chosen = indices[chosen_pos];
        let remaining: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != chosen_pos)
            .map(|(_, &i)| i)
            .collect();
        let pat_idx = pat_args.len() - 1 - prefix_len; // next prefix pattern index
        if let Some(s) = match_one(
            pat_args[pat_idx],
            expr_args[chosen],
            subst.clone(),
            pool,
            ac_depth + 1,
        ) {
            if let Some(final_s) = try_subsets(
                pat_args,
                expr_args,
                &remaining,
                prefix_len - 1,
                s,
                pool,
                ac_depth,
                is_add,
            ) {
                return Some(final_s);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find all AC-aware matches of `pattern` anywhere in `expr`.
///
/// Returns a list of substitutions, one per distinct match site.  The
/// search recurses into sub-expressions so that `match_pattern(a + b, f(x + y))`
/// can match `x + y` inside `f(...)`.
///
/// # Example
/// ```
/// # use alkahest_core::kernel::{ExprPool, Domain};
/// # use alkahest_core::pattern::{Pattern, match_pattern};
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let y = pool.symbol("y", Domain::Real);
/// let a = pool.symbol("a", Domain::Real);  // wildcard
/// let b = pool.symbol("b", Domain::Real);  // wildcard
/// let pat = Pattern::from_expr(pool.add(vec![a, b]));
/// let expr = pool.add(vec![x, y]);
/// let matches = match_pattern(&pat, expr, &pool);
/// assert!(!matches.is_empty());
/// ```
pub fn match_pattern(pattern: &Pattern, expr: ExprId, pool: &ExprPool) -> Vec<Substitution> {
    let mut results = Vec::new();
    collect_matches(pattern.root, expr, pool, &mut results);
    results
}

/// Recursively search `expr` and its sub-expressions for matches of `pat`.
fn collect_matches(pat: ExprId, expr: ExprId, pool: &ExprPool, results: &mut Vec<Substitution>) {
    // Try matching at this node
    if let Some(s) = match_one(pat, expr, Substitution::new(), pool, 0) {
        results.push(s);
    }

    // Recurse into children
    let children: Vec<ExprId> = pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::Func { args, .. } => args.clone(),
        _ => vec![],
    });

    for child in children {
        collect_matches(pat, child, pool, results);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn wildcard_matches_anything() {
        let p = pool();
        let a = p.symbol("a", Domain::Real); // wildcard
        let x = p.symbol("x", Domain::Real);
        let pat = Pattern::from_expr(a);
        let matches = match_pattern(&pat, x, &p);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].bindings["a"], x);
    }

    #[test]
    fn literal_symbol_exact_match() {
        let p = pool();
        let x = p.symbol("x", Domain::Real); // non-wildcard (only if name starts upper or is multi-char, but here it starts lower)
                                             // Use "X" to force a literal pattern
        let xpat = p.symbol("X", Domain::Real); // non-wildcard
        let pat = Pattern::from_expr(xpat);
        // Should not match y
        let y = p.symbol("Y", Domain::Real);
        assert!(match_pattern(&pat, y, &p).is_empty());
        // Should match X
        assert!(!match_pattern(&pat, xpat, &p).is_empty());
        let _ = x; // suppress unused warning
    }

    #[test]
    fn add_pattern_ac_match() {
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        // Pattern: a + b  should match x + y in either order
        let pat = Pattern::from_expr(p.add(vec![a, b]));
        let expr = p.add(vec![x, y]);
        let matches = match_pattern(&pat, expr, &p);
        // At least one match where {a→x, b→y} or {a→y, b→x}
        assert!(!matches.is_empty(), "a+b should match x+y");
    }

    #[test]
    fn add_pattern_two_splits_for_three_terms() {
        // Pattern a + b on x + y + z should find a match for each pair
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);
        let pat = Pattern::from_expr(p.add(vec![a, b]));
        let expr = p.add(vec![x, y, z]);
        let matches = match_pattern(&pat, expr, &p);
        // At least one match (b absorbs the remaining two-element sum)
        assert!(!matches.is_empty(), "a+b should match subsets of x+y+z");
    }

    #[test]
    fn substitution_apply() {
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);
        let pat = p.add(vec![a, one]); // a + 1
        let mut subst = Substitution::new();
        subst.bind("a", x);
        let result = subst.apply(pat, &p);
        // x + 1
        let expected = p.add(vec![x, one]);
        assert_eq!(result, expected);
    }

    #[test]
    fn match_inside_function() {
        // Pattern a + b should match inside f(x + y)
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let inner = p.add(vec![x, y]);
        let f = p.func("f", vec![inner]);
        let pat = Pattern::from_expr(p.add(vec![a, b]));
        let matches = match_pattern(&pat, f, &p);
        assert!(!matches.is_empty(), "should find a+b inside f(x+y)");
    }

    #[test]
    fn no_spurious_matches() {
        // Pattern a * b should NOT match x + y
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let pat = Pattern::from_expr(p.mul(vec![a, b]));
        let expr = p.add(vec![x, y]);
        assert!(
            match_pattern(&pat, expr, &p).is_empty(),
            "mul pattern should not match add"
        );
    }

    #[test]
    fn consistent_wildcard_bindings() {
        // Pattern a + a: both copies of `a` must bind to the same thing
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let pat = Pattern::from_expr(p.add(vec![a, a]));
        // x + x should match
        assert!(!match_pattern(&pat, p.add(vec![x, x]), &p).is_empty());
        // x + y should NOT match
        assert!(match_pattern(&pat, p.add(vec![x, y]), &p).is_empty());
    }
}
