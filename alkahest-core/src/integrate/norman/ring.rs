//! The differential ring `ℚ[x, θ₁, …, θₙ]` used by the Risch–Norman heuristic.
//!
//! The Risch–Norman ansatz equates coefficients of *monomials in the
//! generators*.  That step is only legitimate when the generators are
//! algebraically independent over `ℚ(x)`; otherwise the "linear system" is
//! being read in a basis that is not a basis, and both false declines and (in
//! principle) spurious solutions become possible.  Bronstein, *Structure
//! theorems for parallel integration*, JSC 42(7):757–769 (2007) states the
//! condition.  This module implements a conservative, decidable approximation
//! and **declines** whenever it cannot certify independence:
//!
//! * **Exponentials.**  `exp(η₁), …, exp(η_m)` are handled by reducing the
//!   arguments `ηᵢ` to a `ℤ`-lattice basis inside the `ℚ`-vector space spanned
//!   by their additive *atoms* (see [`lattice_basis`]).  Every `exp(ηᵢ)` then
//!   becomes an integer power product of the basis exponentials, so no
//!   multiplicative relation between them survives — that is exactly the
//!   "`ηᵢ` linearly independent over `ℚ`" half of the structure theorem.  What
//!   remains is the possibility that a basis exponent lies in the `ℚ`-span of
//!   the logarithms already in the tower (`exp(2·log x) = x²`).  The atom
//!   columns are ordered so that logarithm atoms and the constant atom come
//!   last, which forces any such lattice vector into the echelon tail, where
//!   it is detected and declined.
//! * **Logarithms.**  `log(h₁), …, log(h_k)` are independent over
//!   `ℚ(x, exp …)` iff the `hⱼ` are multiplicatively independent modulo
//!   constants.  Each `hⱼ` is factored into irreducibles over `ℤ[x, θ]`
//!   (FLINT) and the resulting exponent vectors — constants dropped, since
//!   `log 2` only shifts a logarithm by a constant — must have full rank.
//!
//! Anything the checks cannot clear is declined, never guessed.

use std::collections::BTreeMap;

use rug::{Integer, Rational};

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::multipoly::MultiPoly;
use crate::poly::rational::{mpoly_exact_div, RationalFunction};

use super::DeclineReason;

/// Maximum number of tower generators (excluding `x`).
///
/// **Observed:** 2 (so 3 variables including `x`) on both the 103-case corpus
/// and the stress set — `exp(x)·exp(exp(x))` and `1/(x·log x·log log x)`.
///
/// Six is not a round number, it is where `ansatz::MAX_UNKNOWNS` takes over.
/// The monomial box carries a per-variable bound of at least 1, so it holds at
/// least `2^nvars` monomials: `2^7 = 128 ≤ 240` but `2^8 = 256 > 240`.  Seven
/// variables — `x` plus six generators — is therefore the widest tower that can
/// produce a solvable ansatz at all; an eighth generator could only ever
/// decline with `TooLarge`, and declining on generator count is a clearer
/// message than declining on box size.
pub(super) const MAX_GENERATORS: usize = 6;

/// Maximum number of distinct additive atoms across all exponential arguments.
///
/// **Observed:** 3 on both corpora (`exp(−x²)`-style arguments contribute one
/// atom each).  The atoms are the columns of the lattice reduction, whose cost
/// is `O(atoms² · rows)` of bignum work; 16 keeps that under a millisecond even
/// in the worst case while being five times anything measured.
pub(super) const MAX_ATOMS: usize = 16;
/// Maximum expression nesting this module will accept.
///
/// `collect` and `to_rf` walk the expression tree recursively, and so — before
/// either of them runs — does `simplify`.  A deeply nested integrand exhausts
/// the stack, which *aborts the process*: strictly worse than a panic, not
/// catchable by the caller, and reachable from user input through `parse`.
/// Measured on this machine, a 5 000-deep chain of `Add` nodes overflows the
/// default 8 MiB stack inside `simplify`.
///
/// 256 is two orders of magnitude below that and more than an order of
/// magnitude above anything real: the deepest integrand in the 103-case corpus
/// nests 6 levels, and `parse` flattens n-ary sums and products rather than
/// nesting them.  [`depth_exceeds`] enforces it iteratively at the entry point,
/// because a recursive depth check would itself overflow.
pub(super) const MAX_DEPTH: u32 = 256;

/// `true` when `expr` nests deeper than `limit`.
///
/// Iterative on purpose — this is the guard that stops a stack overflow, so it
/// must not be able to cause one.  The pool is a DAG, so depths are memoised
/// per node and the walk is linear in the number of distinct subexpressions
/// rather than in the number of paths.
pub(super) fn depth_exceeds(expr: ExprId, pool: &ExprPool, limit: u32) -> bool {
    fn children(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        pool.with(e, |d| match d {
            ExprData::Add(a) | ExprData::Mul(a) => a.clone(),
            ExprData::Pow { base, exp } => vec![*base, *exp],
            ExprData::Func { args, .. } => args.clone(),
            _ => Vec::new(),
        })
    }

    let mut depth: BTreeMap<ExprId, u32> = BTreeMap::new();
    let mut queued: std::collections::BTreeSet<ExprId> = std::collections::BTreeSet::new();
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];
    queued.insert(expr);
    while let Some((e, expanded)) = stack.pop() {
        if expanded {
            let d = 1 + children(e, pool)
                .iter()
                .filter_map(|k| depth.get(k))
                .copied()
                .max()
                .unwrap_or(0);
            if d > limit {
                return true;
            }
            depth.insert(e, d);
            continue;
        }
        if depth.contains_key(&e) {
            continue;
        }
        stack.push((e, true));
        for k in children(e, pool) {
            if !depth.contains_key(&k) && queued.insert(k) {
                stack.push((k, false));
            }
        }
    }
    false
}

/// Maximum magnitude of an integer exponent handled during ring conversion.
///
/// Repeated squaring is not used, so `r^k` costs `k` rational-function
/// multiplications; 64 is the point past which an integrand is better refused
/// than expanded.  The largest exponent in the 103-case corpus is 6.
const MAX_POW: u64 = 64;

/// Which kind of monomial a tower generator is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GenKind {
    /// `θ = exp(η)`, with `D(θ) = D(η)·θ`.
    Exp,
    /// `θ = log(h)`, with `D(θ) = D(h)/h`.
    Log,
}

/// A differential ring `ℚ[x, θ₁, …, θₙ]` together with its derivation `D`.
pub(super) struct NormanRing {
    /// `vars[0]` is the integration variable; `vars[1..]` are the generators.
    /// This vector *is* the variable ordering of every [`MultiPoly`] below.
    pub vars: Vec<ExprId>,
    /// Kind of `vars[i + 1]`, stored at `kinds[i]`.
    pub kinds: Vec<GenKind>,
    /// One common denominator `L` for **every** `D(vars[i])`.
    ///
    /// Together with [`NormanRing::dnum`] this gives `D(vars[i]) = dnum[i]/L`
    /// with a denominator that does not depend on `i`.  That is what lets the
    /// ansatz differentiate a monomial with polynomial arithmetic alone: see
    /// [`NormanRing::deriv_scaled`].
    pub dden: MultiPoly,
    /// `dnum[i] = D(vars[i])·L`, a polynomial.
    pub dnum: Vec<MultiPoly>,
    /// Additive atoms used to normalise exponential arguments.
    atoms: Vec<ExprId>,
    /// Common denominator applied to atom coordinates before lattice reduction.
    atom_denom: Integer,
    /// Echelon lattice basis of the scaled exponential arguments.
    basis: Vec<Vec<Integer>>,
    /// Pivot column of each basis row.
    pivots: Vec<usize>,
}

impl NormanRing {
    /// Number of variables, including `x`.
    pub fn nvars(&self) -> usize {
        self.vars.len()
    }

    /// The constant polynomial `c`.
    pub fn constant_poly(&self, c: i64) -> MultiPoly {
        MultiPoly::constant(self.vars.clone(), c)
    }

    /// The monomial with exponent vector `exp` and coefficient 1.
    pub fn monomial(&self, exp: &[u32]) -> MultiPoly {
        let mut e = exp.to_vec();
        while e.last() == Some(&0) {
            e.pop();
        }
        let mut terms = BTreeMap::new();
        terms.insert(e, Integer::from(1));
        MultiPoly {
            vars: self.vars.clone(),
            terms,
        }
    }

    /// `p / 1` as a rational function.
    pub fn rf(&self, p: MultiPoly) -> Result<RationalFunction, DeclineReason> {
        RationalFunction::new(p, self.constant_poly(1)).map_err(|_| DeclineReason::RingArithmetic)
    }

    /// The zero rational function.
    pub fn rf_zero(&self) -> RationalFunction {
        RationalFunction {
            numer: MultiPoly::zero(self.vars.clone()),
            denom: self.constant_poly(1),
        }
    }

    /// The rational function `1`.
    pub fn rf_one(&self) -> Result<RationalFunction, DeclineReason> {
        self.rf(self.constant_poly(1))
    }

    /// `D(p)·L` for a polynomial `p` — the chain rule `Σᵢ (∂p/∂vᵢ)·dnum[i]`,
    /// with `L = ` [`NormanRing::dden`] the shared derivation denominator.
    ///
    /// Returning the *scaled* numerator rather than a [`RationalFunction`] is
    /// the whole point: `RationalFunction`'s operators normalise, and each
    /// normalisation is a FLINT multivariate GCD plus two exact divisions.
    /// Differentiating an ansatz of `n` atoms through `RationalFunction` cost
    /// `O(n)` GCDs and dominated the module's runtime (measured: 67 % of it).
    /// Every atom's derivative shares the denominator `L·Q²` (rational atoms)
    /// or `L·p` (logarithm atoms), so the reduction is not merely wasted, it is
    /// undone again by the common-denominator step that follows.
    pub fn deriv_scaled(&self, p: &MultiPoly) -> MultiPoly {
        let mut acc = MultiPoly::zero(self.vars.clone());
        for i in 0..self.nvars() {
            let dp = p.partial_derivative(i);
            if dp.is_zero() {
                continue;
            }
            acc = acc + dp * self.dnum[i].clone();
        }
        acc
    }

    /// Convert `expr` into an element of `ℚ(x, θ₁, …, θₙ)`.
    ///
    /// Declines on anything the ring cannot represent exactly: an unknown
    /// symbol, a function that is not a tower generator, a non-integer
    /// exponent, or a floating-point literal.
    pub fn to_rf(&self, expr: ExprId, pool: &ExprPool) -> Result<RationalFunction, DeclineReason> {
        self.to_rf_at(expr, pool, 0)
    }

    fn to_rf_at(
        &self,
        expr: ExprId,
        pool: &ExprPool,
        depth: u32,
    ) -> Result<RationalFunction, DeclineReason> {
        if depth > MAX_DEPTH {
            return Err(DeclineReason::TooLarge("expression nesting depth"));
        }
        if let Some(i) = self.vars.iter().position(|&v| v == expr) {
            let mut e = vec![0u32; i + 1];
            e[i] = 1;
            return self.rf(self.monomial(&e));
        }

        enum Node {
            Int(Integer),
            Rat(Integer, Integer),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow(ExprId, Option<i64>),
            Exp(ExprId),
            Unsupported(String),
        }

        let node = pool.with(expr, |data| match data {
            ExprData::Integer(n) => Node::Int(n.0.clone()),
            ExprData::Rational(r) => Node::Rat(r.0.numer().clone(), r.0.denom().clone()),
            ExprData::Add(a) => Node::Add(a.clone()),
            ExprData::Mul(a) => Node::Mul(a.clone()),
            ExprData::Pow { base, exp } => {
                let k = pool.with(*exp, |e| match e {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                });
                Node::Pow(*base, k)
            }
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => Node::Exp(args[0]),
            ExprData::Symbol { name, .. } => Node::Unsupported(format!("symbol `{name}`")),
            ExprData::Func { name, .. } => Node::Unsupported(format!("`{name}`")),
            ExprData::Float(_) => Node::Unsupported("floating-point literal".to_string()),
            _ => Node::Unsupported("non-algebraic node".to_string()),
        });

        match node {
            Node::Int(n) => {
                let mut terms = BTreeMap::new();
                if n != 0 {
                    terms.insert(Vec::new(), n);
                }
                self.rf(MultiPoly {
                    vars: self.vars.clone(),
                    terms,
                })
            }
            Node::Rat(num, den) => {
                let mut nt = BTreeMap::new();
                if num != 0 {
                    nt.insert(Vec::new(), num);
                }
                let mut dt = BTreeMap::new();
                dt.insert(Vec::new(), den);
                RationalFunction::new(
                    MultiPoly {
                        vars: self.vars.clone(),
                        terms: nt,
                    },
                    MultiPoly {
                        vars: self.vars.clone(),
                        terms: dt,
                    },
                )
                .map_err(|_| DeclineReason::RingArithmetic)
            }
            Node::Add(args) => {
                let mut acc = self.rf_zero();
                for a in args {
                    let r = self.to_rf_at(a, pool, depth + 1)?;
                    acc = (acc + r).map_err(|_| DeclineReason::RingArithmetic)?;
                }
                Ok(acc)
            }
            Node::Mul(args) => {
                let mut acc = self.rf_one()?;
                for a in args {
                    let r = self.to_rf_at(a, pool, depth + 1)?;
                    acc = (acc * r).map_err(|_| DeclineReason::RingArithmetic)?;
                }
                Ok(acc)
            }
            Node::Pow(base, Some(k)) => {
                let r = self.to_rf_at(base, pool, depth + 1)?;
                self.rf_pow(&r, k)
            }
            Node::Pow(_, None) => Err(DeclineReason::UnsupportedIntegrand(
                "non-integer exponent".to_string(),
            )),
            Node::Exp(arg) => self.exp_rf(arg, pool),
            Node::Unsupported(what) => Err(DeclineReason::UnsupportedIntegrand(format!(
                "{what} is not a generator of the ring"
            ))),
        }
    }

    /// `r^k` for an integer `k`; a negative `k` inverts.
    fn rf_pow(&self, r: &RationalFunction, k: i64) -> Result<RationalFunction, DeclineReason> {
        // `unsigned_abs`, not `abs`: `i64::MIN.abs()` overflows, which in a
        // release build wraps back to `i64::MIN` and slips past the cap, and
        // the loop below then runs `2^63` times.  `x^(-9223372036854775808)`
        // is reachable from user input.
        if k.unsigned_abs() > MAX_POW {
            return Err(DeclineReason::TooLarge("exponent magnitude"));
        }
        let base = if k < 0 {
            if r.numer.is_zero() {
                return Err(DeclineReason::RingArithmetic);
            }
            RationalFunction::new(r.denom.clone(), r.numer.clone())
                .map_err(|_| DeclineReason::RingArithmetic)?
        } else {
            r.clone()
        };
        let mut acc = self.rf_one()?;
        for _ in 0..k.unsigned_abs() {
            acc = (acc * base.clone()).map_err(|_| DeclineReason::RingArithmetic)?;
        }
        Ok(acc)
    }

    /// `exp(arg)`, rewritten as an integer power product of the basis
    /// exponential generators.
    fn exp_rf(&self, arg: ExprId, pool: &ExprPool) -> Result<RationalFunction, DeclineReason> {
        let coords = self.atom_coords(arg, pool)?;
        let mut scaled = Vec::with_capacity(coords.len());
        for c in &coords {
            let v = c.clone() * Rational::from(self.atom_denom.clone());
            if *v.denom() != 1 {
                return Err(DeclineReason::DependentGenerators(
                    "exponential argument falls outside the reduced lattice",
                ));
            }
            scaled.push(v.numer().clone());
        }
        let combo = int_combo(&self.basis, &self.pivots, &scaled).ok_or(
            DeclineReason::DependentGenerators("exponential argument falls outside the lattice"),
        )?;
        let mut acc = self.rf_one()?;
        for (k, n) in combo.iter().enumerate() {
            if *n == 0 {
                continue;
            }
            let Some(ni) = n.to_i64() else {
                return Err(DeclineReason::TooLarge("exponential power"));
            };
            let idx = 1 + k;
            let mut e = vec![0u32; idx + 1];
            e[idx] = 1;
            let g = self.rf(self.monomial(&e))?;
            let p = self.rf_pow(&g, ni)?;
            acc = (acc * p).map_err(|_| DeclineReason::RingArithmetic)?;
        }
        Ok(acc)
    }

    /// Coordinates of `expr` in the atom basis, as a `ℚ`-vector.
    fn atom_coords(&self, expr: ExprId, pool: &ExprPool) -> Result<Vec<Rational>, DeclineReason> {
        let terms = atom_decompose(expr, pool).ok_or_else(|| {
            DeclineReason::UnsupportedIntegrand(
                "exponential argument is not a ℚ-combination of atoms".to_string(),
            )
        })?;
        let mut out = vec![Rational::from(0); self.atoms.len()];
        for (atom, coeff) in terms {
            let Some(i) = self.atoms.iter().position(|&a| a == atom) else {
                return Err(DeclineReason::DependentGenerators(
                    "exponential argument uses an atom outside the reduced basis",
                ));
            };
            out[i] += coeff;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Atom decomposition
// ---------------------------------------------------------------------------

/// Split `expr` into a `ℚ`-linear combination of *atoms*.
///
/// `2x + 3·log(x) − 1` becomes `[(x, 2), (log(x), 3), (1, −1)]`.  Returns
/// `None` if a term carries a non-rational numeric factor such as a float.
pub(super) fn atom_decompose(expr: ExprId, pool: &ExprPool) -> Option<Vec<(ExprId, Rational)>> {
    let mut out: Vec<(ExprId, Rational)> = Vec::new();
    let summands = pool.with(expr, |d| match d {
        ExprData::Add(a) => a.clone(),
        _ => vec![expr],
    });
    for s in summands {
        let (atom, coeff) = split_coefficient(s, pool)?;
        if let Some(slot) = out.iter_mut().find(|(a, _)| *a == atom) {
            slot.1 += coeff;
        } else {
            out.push((atom, coeff));
        }
    }
    out.retain(|(_, c)| *c != 0);
    Some(out)
}

/// Split a single product into `(atom, rational coefficient)`.
fn split_coefficient(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Rational)> {
    let factors = pool.with(expr, |d| match d {
        ExprData::Mul(a) => a.clone(),
        _ => vec![expr],
    });
    let mut coeff = Rational::from(1);
    let mut rest: Vec<ExprId> = Vec::new();
    for f in factors {
        enum Kind {
            Num(Rational),
            Float,
            Other,
        }
        let kind = pool.with(f, |d| match d {
            ExprData::Integer(n) => Kind::Num(Rational::from(n.0.clone())),
            ExprData::Rational(r) => Kind::Num(r.0.clone()),
            ExprData::Float(_) => Kind::Float,
            _ => Kind::Other,
        });
        match kind {
            Kind::Float => return None,
            Kind::Num(v) => coeff *= v,
            Kind::Other => rest.push(f),
        }
    }
    let atom = match rest.len() {
        0 => pool.integer(1_i32),
        1 => rest[0],
        _ => pool.mul(rest),
    };
    Some((atom, coeff))
}

// ---------------------------------------------------------------------------
// Integer lattice reduction (row-style Hermite form)
// ---------------------------------------------------------------------------

/// Reduce the rows of an integer matrix to an echelon `ℤ`-basis of the lattice
/// they generate.  Returns `(basis rows, pivot column of each row)`.
///
/// This is what makes `exp(2x)`, `exp(−x)` and `exp(x)` a *single* generator
/// rather than three, and it is the exponential half of the structure-theorem
/// check: after reduction no non-trivial multiplicative relation between the
/// basis exponentials survives.
pub(super) fn lattice_basis(
    mut rows: Vec<Vec<Integer>>,
    ncols: usize,
) -> (Vec<Vec<Integer>>, Vec<usize>) {
    let mut basis = Vec::new();
    let mut pivots = Vec::new();
    let mut top = 0usize;
    for col in 0..ncols {
        loop {
            let nz: Vec<usize> = (top..rows.len()).filter(|&i| rows[i][col] != 0).collect();
            if nz.len() <= 1 {
                break;
            }
            let p = *nz
                .iter()
                .min_by_key(|&&i| rows[i][col].clone().abs())
                .expect("nz is non-empty");
            let pivot_row = rows[p].clone();
            for &i in &nz {
                if i == p {
                    continue;
                }
                let q = Integer::from(&rows[i][col] / &pivot_row[col]);
                if q == 0 {
                    continue;
                }
                for (cell, pv) in rows[i].iter_mut().zip(pivot_row.iter()).skip(col) {
                    *cell -= pv.clone() * q.clone();
                }
            }
        }
        let Some(p) = (top..rows.len()).find(|&i| rows[i][col] != 0) else {
            continue;
        };
        rows.swap(top, p);
        if rows[top][col] < 0 {
            for v in rows[top].iter_mut() {
                *v = -std::mem::take(v);
            }
        }
        basis.push(rows[top].clone());
        pivots.push(col);
        top += 1;
    }
    (basis, pivots)
}

/// Express `w` as an integer combination of an echelon lattice `basis`.
pub(super) fn int_combo(
    basis: &[Vec<Integer>],
    pivots: &[usize],
    w: &[Integer],
) -> Option<Vec<Integer>> {
    let mut rem: Vec<Integer> = w.to_vec();
    let mut out = vec![Integer::from(0); basis.len()];
    for (k, row) in basis.iter().enumerate() {
        let c = pivots[k];
        if rem[c] == 0 {
            continue;
        }
        if !rem[c].is_divisible(&row[c]) {
            return None;
        }
        let q = Integer::from(&rem[c] / &row[c]);
        for (j, rv) in rem.iter_mut().enumerate() {
            *rv -= row[j].clone() * q.clone();
        }
        out[k] = q;
    }
    if rem.iter().all(|v| *v == 0) {
        Some(out)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Ring construction
// ---------------------------------------------------------------------------

/// Every `exp`/`log` node found in the integrand.
struct Collected {
    exp_args: Vec<ExprId>,
    log_nodes: Vec<ExprId>,
    log_args: Vec<ExprId>,
}

fn collect(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    acc: &mut Collected,
    depth: u32,
) -> Result<(), DeclineReason> {
    if depth > MAX_DEPTH {
        return Err(DeclineReason::TooLarge("expression nesting depth"));
    }
    enum Node {
        Leaf,
        Kids(Vec<ExprId>),
        Pow(ExprId, ExprId, bool),
        Exp(ExprId),
        Log(ExprId),
        Bad(String),
    }
    let node = pool.with(expr, |d| match d {
        ExprData::Integer(_) | ExprData::Rational(_) => Node::Leaf,
        ExprData::Float(_) => Node::Bad("floating-point literal".to_string()),
        ExprData::Symbol { name, .. } => {
            if expr == var {
                Node::Leaf
            } else {
                Node::Bad(format!("free symbol `{name}`"))
            }
        }
        ExprData::Add(a) | ExprData::Mul(a) => Node::Kids(a.clone()),
        ExprData::Pow { base, exp } => {
            let int_exp = pool.with(*exp, |e| matches!(e, ExprData::Integer(_)));
            Node::Pow(*base, *exp, int_exp)
        }
        ExprData::Func { name, args } if name == "exp" && args.len() == 1 => Node::Exp(args[0]),
        ExprData::Func { name, args } if name == "log" && args.len() == 1 => Node::Log(args[0]),
        ExprData::Func { name, .. } => Node::Bad(format!("`{name}`")),
        _ => Node::Bad("non-algebraic node".to_string()),
    });

    match node {
        Node::Leaf => Ok(()),
        Node::Bad(what) => Err(DeclineReason::UnsupportedIntegrand(format!(
            "{what} is outside the exp/log ring"
        ))),
        Node::Kids(kids) => {
            for k in kids {
                collect(k, var, pool, acc, depth + 1)?;
            }
            Ok(())
        }
        Node::Pow(base, exp, int_exp) => {
            if !int_exp {
                return Err(DeclineReason::UnsupportedIntegrand(
                    "non-integer exponent (radical or symbolic power)".to_string(),
                ));
            }
            collect(base, var, pool, acc, depth + 1)?;
            collect(exp, var, pool, acc, depth + 1)
        }
        Node::Exp(arg) => {
            collect(arg, var, pool, acc, depth + 1)?;
            if !acc.exp_args.contains(&arg) {
                acc.exp_args.push(arg);
            }
            Ok(())
        }
        Node::Log(arg) => {
            collect(arg, var, pool, acc, depth + 1)?;
            if !acc.log_nodes.contains(&expr) {
                acc.log_nodes.push(expr);
                acc.log_args.push(arg);
            }
            Ok(())
        }
    }
}

/// Build the differential ring for `expr`.
pub(super) fn build(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<NormanRing, DeclineReason> {
    let mut acc = Collected {
        exp_args: Vec::new(),
        log_nodes: Vec::new(),
        log_args: Vec::new(),
    };
    collect(expr, var, pool, &mut acc, 0)?;

    // --- Atoms, ordered: ordinary atoms, then logarithm generators, then `1`.
    let one = pool.integer(1_i32);
    let mut plain: Vec<ExprId> = Vec::new();
    let mut logs: Vec<ExprId> = Vec::new();
    let mut has_const = false;
    let mut decomposed: Vec<Vec<(ExprId, Rational)>> = Vec::new();
    for &a in &acc.exp_args {
        let d = atom_decompose(a, pool).ok_or_else(|| {
            DeclineReason::UnsupportedIntegrand(
                "exponential argument is not a ℚ-combination of atoms".to_string(),
            )
        })?;
        for (atom, _) in &d {
            if *atom == one {
                has_const = true;
            } else if acc.log_nodes.contains(atom) {
                if !logs.contains(atom) {
                    logs.push(*atom);
                }
            } else if !plain.contains(atom) {
                plain.push(*atom);
            }
        }
        decomposed.push(d);
    }
    let mut atoms = plain;
    let n_plain = atoms.len();
    atoms.extend(logs);
    let n_log_atoms = atoms.len() - n_plain;
    if has_const {
        atoms.push(one);
    }
    if atoms.len() > MAX_ATOMS {
        return Err(DeclineReason::TooLarge("exponential atom count"));
    }

    // --- Lattice reduction of the exponential arguments.
    let mut atom_denom = Integer::from(1);
    for d in &decomposed {
        for (_, c) in d {
            atom_denom = atom_denom.clone().lcm(c.denom());
        }
    }
    let ncols = atoms.len();
    let mut rows: Vec<Vec<Integer>> = Vec::new();
    for d in &decomposed {
        let mut row = vec![Integer::from(0); ncols];
        for (atom, c) in d {
            let i = atoms
                .iter()
                .position(|a| a == atom)
                .expect("atom collected above");
            let scaled = c.clone() * Rational::from(atom_denom.clone());
            row[i] += scaled.numer().clone();
        }
        rows.push(row);
    }
    let (basis, pivots) = lattice_basis(rows, ncols);

    // Structure theorem, exponential half: a basis vector supported only on
    // logarithm atoms means `exp(Σ cⱼ·log hⱼ) = ∏ hⱼ^{cⱼ}`, which is algebraic
    // over the field, so the monomials would not be independent.
    for row in &basis {
        let touches_log = (n_plain..n_plain + n_log_atoms).any(|c| row[c] != 0);
        let outside = (0..n_plain).any(|c| row[c] != 0);
        if touches_log && !outside {
            return Err(DeclineReason::DependentGenerators(
                "an exponential argument is a ℚ-combination of the tower's logarithms",
            ));
        }
    }

    // --- Variables: `[x, exp gens…, log gens…]`.
    let mut vars = vec![var];
    let mut kinds = Vec::new();
    let mut basis_exprs = Vec::new();
    for row in &basis {
        let e = lattice_row_to_expr(row, &atoms, &atom_denom, pool);
        let g = pool.func("exp", vec![e]);
        basis_exprs.push(e);
        vars.push(g);
        kinds.push(GenKind::Exp);
    }
    let n_exp = basis.len();
    for &l in &acc.log_nodes {
        vars.push(l);
        kinds.push(GenKind::Log);
    }
    if vars.len() > MAX_GENERATORS + 1 {
        return Err(DeclineReason::TooLarge("generator count"));
    }
    // Hash-consing can collapse a constructed generator onto an existing one;
    // duplicated variables would silently break the coefficient matching.
    for i in 0..vars.len() {
        for j in (i + 1)..vars.len() {
            if vars[i] == vars[j] {
                return Err(DeclineReason::DependentGenerators(
                    "two tower generators are the same expression",
                ));
            }
        }
    }

    let n_vars = vars.len();
    let mut ring = NormanRing {
        vars: vars.clone(),
        kinds,
        dden: MultiPoly::constant(vars, 1),
        dnum: Vec::new(),
        atoms,
        atom_denom,
        basis,
        pivots,
    };

    // --- Derivations.
    let mut dvars = vec![ring.rf_one()?];
    for (k, &b) in basis_exprs.iter().enumerate() {
        let db = derivative_rf(&ring, b, var, pool)?;
        let idx = 1 + k;
        let mut e = vec![0u32; idx + 1];
        e[idx] = 1;
        let gen = ring.rf(ring.monomial(&e))?;
        dvars.push((db * gen).map_err(|_| DeclineReason::RingArithmetic)?);
    }
    for &h in &acc.log_args {
        let dh = derivative_rf(&ring, h, var, pool)?;
        let hr = ring.to_rf(h, pool)?;
        if hr.numer.is_zero() {
            return Err(DeclineReason::RingArithmetic);
        }
        dvars.push((dh / hr).map_err(|_| DeclineReason::RingArithmetic)?);
    }
    debug_assert_eq!(dvars.len(), n_vars);

    // --- One denominator for the whole derivation table.
    //
    // `D(vᵢ) = dnum[i]/L` with a single `L` turns differentiating an ansatz
    // monomial into polynomial arithmetic (see `deriv_scaled`).  The `lcm`
    // here is at most `nvars ≤ 7` GCDs, paid once per integrand.
    let mut dden = MultiPoly::constant(ring.vars.clone(), 1);
    for dv in &dvars {
        dden = lcm(&dden, &dv.denom).ok_or(DeclineReason::RingArithmetic)?;
    }
    let mut dnum = Vec::with_capacity(dvars.len());
    for dv in &dvars {
        let scale = mpoly_exact_div(&dden, &dv.denom).ok_or(DeclineReason::RingArithmetic)?;
        dnum.push(dv.numer.clone() * scale);
    }
    ring.dden = dden;
    ring.dnum = dnum;

    // --- Structure theorem, logarithmic half.
    check_log_independence(&ring, &acc.log_args, n_exp, pool)?;

    Ok(ring)
}

/// `true` for the constant polynomial `1`.
pub(super) fn is_unit(p: &MultiPoly) -> bool {
    p.terms.len() == 1 && p.terms.get(&Vec::new()).is_some_and(|c| *c == 1)
}

/// `lcm(a, b) = a·(b / gcd(a, b))`.
///
/// Only the literal unit is short-circuited.  A *constant* denominator is not
/// a unit here: `RationalFunction` keeps `3/5` as `3/5`, and clearing against
/// `lcm(5, x) = x` would leave an inexact division and a spurious decline.
pub(super) fn lcm(a: &MultiPoly, b: &MultiPoly) -> Option<MultiPoly> {
    if is_unit(a) {
        return Some(b.clone());
    }
    if is_unit(b) {
        return Some(a.clone());
    }
    if a == b {
        return Some(a.clone());
    }
    match a.gcd(b) {
        Some(g) => {
            let q = mpoly_exact_div(b, &g)?;
            Some(a.clone() * q)
        }
        None => Some(a.clone() * b.clone()),
    }
}

/// `c·x^shift·p`, i.e. `p` multiplied by a single monomial term.
///
/// Multiplying by a monomial is a relabelling of the exponent keys, so it costs
/// `O(|p|)` key copies and coefficient multiplications rather than the
/// `O(|p|·|m|)` of a general product.  The ansatz's inner loop is nothing but
/// this operation, which is why it is worth spelling out.
pub(super) fn shift_scale(p: &MultiPoly, shift: &[u32], c: &Integer) -> MultiPoly {
    let mut terms = BTreeMap::new();
    if *c == 0 {
        return MultiPoly {
            vars: p.vars.clone(),
            terms,
        };
    }
    for (exp, coeff) in &p.terms {
        let n = exp.len().max(shift.len());
        let mut key = Vec::with_capacity(n);
        for i in 0..n {
            key.push(exp.get(i).copied().unwrap_or(0) + shift.get(i).copied().unwrap_or(0));
        }
        while key.last() == Some(&0) {
            key.pop();
        }
        // Distinct source keys stay distinct under a shift, so no accumulation
        // is needed — but `insert` is used rather than a bare push so that a
        // future caller passing a non-canonical `p` cannot silently lose terms.
        let entry = terms.entry(key).or_insert_with(|| Integer::from(0));
        *entry += coeff.clone() * c.clone();
    }
    terms.retain(|_, v| *v != 0);
    MultiPoly {
        vars: p.vars.clone(),
        terms,
    }
}

/// `d/dx e`, converted into the ring.
fn derivative_rf(
    ring: &NormanRing,
    e: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<RationalFunction, DeclineReason> {
    let d = crate::diff::diff(e, var, pool).map_err(|_| DeclineReason::RingArithmetic)?;
    let d = crate::simplify::engine::simplify(d.value, pool).value;
    ring.to_rf(d, pool)
}

/// Rebuild `Σ (rowᵢ / denom)·atomᵢ` as an expression.
fn lattice_row_to_expr(
    row: &[Integer],
    atoms: &[ExprId],
    denom: &Integer,
    pool: &ExprPool,
) -> ExprId {
    let one = pool.integer(1_i32);
    let mut terms = Vec::new();
    for (i, c) in row.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let q = Rational::from((c.clone(), denom.clone()));
        let coeff = pool.rational(q.numer().clone(), q.denom().clone());
        let atom = atoms[i];
        let t = if atom == one {
            coeff
        } else if q == 1 {
            atom
        } else {
            pool.mul(vec![coeff, atom])
        };
        terms.push(t);
    }
    let raw = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };
    crate::simplify::engine::simplify(raw, pool).value
}

/// Multiplicative independence of the logarithm arguments, modulo constants.
fn check_log_independence(
    ring: &NormanRing,
    log_args: &[ExprId],
    n_exp: usize,
    pool: &ExprPool,
) -> Result<(), DeclineReason> {
    if log_args.is_empty() {
        return Ok(());
    }
    // Factor each argument and record the exponent vector over the
    // non-constant irreducible factors.  Integer content is dropped: `log 2`
    // only shifts a logarithm by a constant, which is invisible to `D`.
    // Powers of an exponential generator are dropped for the same reason
    // (`log(exp η) = η` is already in the field, not a new logarithm).
    let mut factor_list: Vec<MultiPoly> = Vec::new();
    let mut vectors: Vec<Vec<(usize, i64)>> = Vec::new();
    for &h in log_args {
        let hr = ring.to_rf(h, pool)?;
        let mut entries: Vec<(usize, i64)> = Vec::new();
        for (p, sign) in [(&hr.numer, 1_i64), (&hr.denom, -1_i64)] {
            let (_unit, factors) = p
                .factor_irreducible()
                .ok_or(DeclineReason::RingArithmetic)?;
            for (f, m) in factors {
                if f.total_degree() == 0 || is_pure_exp_generator(&f, n_exp) {
                    continue;
                }
                let neg = -f.clone();
                let idx = match factor_list.iter().position(|g| *g == f || *g == neg) {
                    Some(i) => i,
                    None => {
                        factor_list.push(f);
                        factor_list.len() - 1
                    }
                };
                entries.push((idx, sign * i64::from(m)));
            }
        }
        vectors.push(entries);
    }
    let ncols = factor_list.len();
    let mut mat: Vec<Vec<Rational>> = Vec::new();
    for v in &vectors {
        let mut row = vec![Rational::from(0); ncols];
        for (i, m) in v {
            row[*i] += Rational::from(*m);
        }
        mat.push(row);
    }
    if rank_q(&mut mat, ncols) < log_args.len() {
        return Err(DeclineReason::DependentGenerators(
            "the tower's logarithm arguments are multiplicatively dependent modulo constants",
        ));
    }
    Ok(())
}

/// `true` when `f` is `c·θ^k` for exponential generators `θ` alone.
fn is_pure_exp_generator(f: &MultiPoly, n_exp: usize) -> bool {
    if f.terms.len() != 1 {
        return false;
    }
    let (exp, _) = f.terms.iter().next().expect("exactly one term");
    let support: Vec<usize> = exp
        .iter()
        .enumerate()
        .filter(|(_, &e)| e > 0)
        .map(|(i, _)| i)
        .collect();
    !support.is_empty() && support.iter().all(|&i| i >= 1 && i <= n_exp)
}

/// Rank of a rational matrix, by Gaussian elimination.
fn rank_q(mat: &mut [Vec<Rational>], ncols: usize) -> usize {
    let nrows = mat.len();
    let mut row = 0usize;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        let Some(p) = (row..nrows).find(|&r| mat[r][col] != 0) else {
            continue;
        };
        mat.swap(row, p);
        let piv = mat[row][col].clone();
        let pivot_row = mat[row].clone();
        for (r, mrow) in mat.iter_mut().enumerate() {
            if r == row || mrow[col] == 0 {
                continue;
            }
            let f = mrow[col].clone() / piv.clone();
            for (cell, pv) in mrow.iter_mut().zip(pivot_row.iter()).skip(col) {
                *cell -= pv.clone() * f.clone();
            }
        }
        row += 1;
    }
    row
}
