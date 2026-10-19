//! Lie-algebra commutator tables (F20).
//!
//! A [`LieAlgebra`] is a list of noncommutative generator symbols `T_a` with
//! structure constants defined by `[T_a, T_b] = i Σ_c f_{abc} T_c`. It
//! compiles to word-rewriting rules `T_a T_b → T_b T_a + [T_a, T_b]` for
//! `a > b` (a Poincaré–Birkhoff–Witt ordering), so [`LieAlgebra::commutator`]
//! reduces the commutator of *any* noncommutative polynomial in the
//! generators to the generators' linear span plus normal-ordered words, and
//! [`LieAlgebra::normal_order`] orders any word.
//!
//! Built-in tables: [`su2_structure_constants`] (`f = ε`, generators
//! `σ/2`), [`su3_structure_constants`] and [`su3_d_symbols`] (Gell-Mann
//! `f_{abc}`, `d_{abc}`, generators `λ/2`):
//! `[T_a, T_b] = i f_{abc} T_c`, `{λ_a, λ_b} = (4/3) δ_{ab} + 2 d_{abc} λ_c`.
//!
//! A user-supplied table is checked against the Jacobi identity unless the
//! caller opts out; a table that fails it (or cannot be shown to satisfy it)
//! is refused with `E-NC-008` rather than producing commutators that depend
//! on the order in which they are evaluated.

use super::nc_poly::{
    anticommutator as nc_anticommutator, commutator as nc_commutator, nc_expand, nc_rewrite,
    nc_terms, NcError, DEFAULT_MAX_STEPS,
};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// A structure-constant entry `f_{abc}` (0-based generator indices).
pub type StructureEntry = (usize, usize, usize, ExprId);

/// A Lie algebra given by commutators of its generators.
#[derive(Debug, Clone)]
pub struct LieAlgebra {
    generators: Vec<ExprId>,
    /// `[T_a, T_b] = i Σ_c f_{abc} T_c`, for `a < b`.
    table: HashMap<(usize, usize), Vec<(usize, ExprId)>>,
    rules: Vec<(ExprId, ExprId)>,
}

fn invalid(reason: String) -> NcError {
    NcError::InvalidLieAlgebra { reason }
}

fn imag_times(coeff: ExprId, gen: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.imaginary_unit(), coeff, gen])
}

impl LieAlgebra {
    /// Build the algebra from `[T_a, T_b] = i Σ_c f_{abc} T_c`.
    ///
    /// `constants` lists `(a, b, c, f_abc)` for `a ≠ b`; an entry with
    /// `a > b` is read as `f_{abc} = −f_{bac}`. If both orders of a pair are
    /// given they must agree. Missing entries are zero.
    ///
    /// # Errors
    /// `E-NC-008` for generators that are not distinct noncommutative
    /// symbols, out-of-range or diagonal indices, inconsistent duplicates,
    /// or (with `check_jacobi`) a table that does not satisfy the Jacobi
    /// identity; `E-NC-002`/`E-NC-001` and budget codes from the checks.
    pub fn new(
        generators: Vec<ExprId>,
        constants: &[StructureEntry],
        check_jacobi: bool,
        pool: &ExprPool,
    ) -> Result<LieAlgebra, NcError> {
        let n = generators.len();
        for (i, &g) in generators.iter().enumerate() {
            let ok = pool.with(g, |d| {
                matches!(
                    d,
                    ExprData::Symbol {
                        commutative: false,
                        ..
                    }
                )
            });
            if !ok {
                return Err(invalid(format!(
                    "generator `{}` must be a noncommutative symbol",
                    pool.display(g)
                )));
            }
            if generators[..i].contains(&g) {
                return Err(invalid(format!(
                    "generator `{}` listed twice",
                    pool.display(g)
                )));
            }
        }
        let mut raw: HashMap<(usize, usize, usize), ExprId> = HashMap::new();
        let neg = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
        for &(a, b, c, f) in constants {
            if a >= n || b >= n || c >= n {
                return Err(invalid(format!(
                    "structure-constant index ({a}, {b}, {c}) out of range for {n} generators"
                )));
            }
            if a == b {
                return Err(invalid(format!(
                    "f_({a},{b},{c}): [T_a, T_a] = 0, so diagonal entries must not be given"
                )));
            }
            let (key, val) = if a < b {
                ((a, b, c), f)
            } else {
                ((b, a, c), neg(f))
            };
            match raw.get(&key) {
                Some(&old) if old != val => {
                    let same = crate::simplify::engine::simplify_expanded(
                        pool.add(vec![old, neg(val)]),
                        pool,
                    )
                    .value
                        == pool.integer(0_i32);
                    if !same {
                        return Err(invalid(format!(
                            "f_({},{},{}) is given twice with different values",
                            key.0, key.1, key.2
                        )));
                    }
                }
                _ => {
                    raw.insert(key, val);
                }
            }
        }
        let zero = pool.integer(0_i32);
        let mut table: HashMap<(usize, usize), Vec<(usize, ExprId)>> = HashMap::new();
        let mut keys: Vec<_> = raw.keys().copied().collect();
        keys.sort_unstable();
        for (a, b, c) in keys {
            let f = raw[&(a, b, c)];
            if f != zero {
                table.entry((a, b)).or_default().push((c, f));
            }
        }
        // Rules T_a T_b -> T_b T_a + [T_a, T_b] for a > b.
        let mut rules = Vec::new();
        for a in 0..n {
            for b in 0..a {
                let lhs = pool.mul(vec![generators[a], generators[b]]);
                let mut parts = vec![pool.mul(vec![generators[b], generators[a]])];
                if let Some(row) = table.get(&(b, a)) {
                    // [T_a, T_b] = −[T_b, T_a] = −i Σ f_{bac} T_c
                    for &(c, f) in row {
                        parts.push(pool.mul(vec![
                            pool.integer(-1_i32),
                            imag_times(f, generators[c], pool),
                        ]));
                    }
                }
                rules.push((lhs, pool.add(parts)));
            }
        }
        let alg = LieAlgebra {
            generators,
            table,
            rules,
        };
        if check_jacobi {
            alg.verify_jacobi(pool)?;
        }
        Ok(alg)
    }

    /// The generators.
    pub fn generators(&self) -> &[ExprId] {
        &self.generators
    }

    /// The PBW rewrite rules `T_a T_b → T_b T_a + [T_a, T_b]` (`a > b`).
    pub fn rules(&self) -> &[(ExprId, ExprId)] {
        &self.rules
    }

    /// `f_{abc}` for `a < b` as `(c, f)` pairs (`[T_a, T_b] = i Σ f T_c`).
    pub fn structure_row(&self, a: usize, b: usize) -> &[(usize, ExprId)] {
        self.table.get(&(a, b)).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Rewrite `expr` to PBW normal form (generators in increasing order).
    pub fn normal_order(&self, expr: ExprId, pool: &ExprPool) -> Result<ExprId, NcError> {
        nc_rewrite(expr, &self.rules, pool, DEFAULT_MAX_STEPS)
    }

    /// `[a, b]` of two noncommutative polynomials in the generators, reduced
    /// to normal form (for generators: `i Σ f_{abc} T_c`).
    pub fn commutator(&self, a: ExprId, b: ExprId, pool: &ExprPool) -> Result<ExprId, NcError> {
        let c = nc_commutator(a, b, pool)?;
        self.normal_order(c, pool)
    }

    /// `{a, b}` reduced to normal form.
    pub fn anticommutator(&self, a: ExprId, b: ExprId, pool: &ExprPool) -> Result<ExprId, NcError> {
        let c = nc_anticommutator(a, b, pool)?;
        self.normal_order(c, pool)
    }

    /// The quadratic Casimir `Σ_{ab} g^{ab} T_a T_b`. `inverse_metric` is the
    /// matrix `g^{ab}` (row-major, `n × n`); `None` means the identity, which
    /// is right for a basis orthonormal under the Killing form (su(n) with the
    /// standard `f_{abc}`).
    pub fn casimir(
        &self,
        inverse_metric: Option<&[Vec<ExprId>]>,
        pool: &ExprPool,
    ) -> Result<ExprId, NcError> {
        let n = self.generators.len();
        let mut parts = Vec::new();
        match inverse_metric {
            None => {
                for &g in &self.generators {
                    parts.push(pool.mul(vec![g, g]));
                }
            }
            Some(m) => {
                if m.len() != n || m.iter().any(|r| r.len() != n) {
                    return Err(invalid(format!("inverse metric must be {n} × {n}")));
                }
                let zero = pool.integer(0_i32);
                for (a, row) in m.iter().enumerate() {
                    for (b, &g) in row.iter().enumerate() {
                        if g != zero {
                            parts.push(pool.mul(vec![g, self.generators[a], self.generators[b]]));
                        }
                    }
                }
            }
        }
        nc_expand(pool.add(parts), pool)
    }

    /// Check `[[T_a,T_b],T_c] + [[T_b,T_c],T_a] + [[T_c,T_a],T_b] = 0` for all
    /// triples. Refuses (`E-NC-008`) when a residual is not provably zero.
    pub fn verify_jacobi(&self, pool: &ExprPool) -> Result<(), NcError> {
        let n = self.generators.len();
        let g = &self.generators;
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    let t1 = self.commutator(self.commutator(g[a], g[b], pool)?, g[c], pool)?;
                    let t2 = self.commutator(self.commutator(g[b], g[c], pool)?, g[a], pool)?;
                    let t3 = self.commutator(self.commutator(g[c], g[a], pool)?, g[b], pool)?;
                    let sum = pool.add(vec![t1, t2, t3]);
                    if !nc_terms(sum, pool)?.is_empty() {
                        return Err(invalid(format!(
                            "the Jacobi identity fails (or could not be verified) for ({}, {}, {})",
                            pool.display(g[a]),
                            pool.display(g[b]),
                            pool.display(g[c])
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

fn q(pool: &ExprPool, n: i64, d: i64) -> ExprId {
    pool.rational(n, d)
}

fn sqrt3(pool: &ExprPool) -> ExprId {
    pool.pow(pool.integer(3_i32), pool.rational(1, 2))
}

/// `f_{abc} = ε_{abc}` for su(2) (`[J_a, J_b] = i ε_{abc} J_c`; `J = σ/2`),
/// as 0-based entries with `a < b`.
pub fn su2_structure_constants(pool: &ExprPool) -> Vec<StructureEntry> {
    let one = pool.integer(1_i32);
    vec![(0, 1, 2, one), (1, 2, 0, one), (0, 2, 1, pool.integer(-1_i32))]
}

/// Gell-Mann `f_{abc}` for su(3) (`[T_a, T_b] = i f_{abc} T_c`, `T = λ/2`),
/// as 0-based entries `(a, b, c)` with `a < b`, all `c`.
pub fn su3_structure_constants(pool: &ExprPool) -> Vec<StructureEntry> {
    let h = q(pool, 1, 2);
    let mh = q(pool, -1, 2);
    let s = pool.mul(vec![q(pool, 1, 2), sqrt3(pool)]);
    let one = pool.integer(1_i32);
    // Totally antisymmetric: listed for the sorted triple (1-based).
    let sorted: [((usize, usize, usize), ExprId); 9] = [
        ((1, 2, 3), one),
        ((1, 4, 7), h),
        ((1, 5, 6), mh),
        ((2, 4, 6), h),
        ((2, 5, 7), h),
        ((3, 4, 5), h),
        ((3, 6, 7), mh),
        ((4, 5, 8), s),
        ((6, 7, 8), s),
    ];
    let mut out = Vec::new();
    for ((a, b, c), f) in sorted {
        let (a, b, c) = (a - 1, b - 1, c - 1);
        let neg = pool.mul(vec![pool.integer(-1_i32), f]);
        // (a,b,c) +, (b,c,a) +, (c,a,b) + ; (b,a,c) −, (a,c,b) −, (c,b,a) −,
        // keeping a < b in the first two slots.
        out.push((a, b, c, f));
        out.push((b, c, a, f));
        out.push((a, c, b, neg));
    }
    out
}

/// Gell-Mann `d_{abc}` (`{λ_a, λ_b} = (4/3) δ_{ab} + 2 d_{abc} λ_c`), totally
/// symmetric; returned for sorted triples `a ≤ b ≤ c` (1-based).
pub fn su3_d_symbols(pool: &ExprPool) -> Vec<((usize, usize, usize), ExprId)> {
    let h = q(pool, 1, 2);
    let mh = q(pool, -1, 2);
    let r3 = pool.pow(pool.integer(3_i32), pool.rational(-1, 2));
    let neg_r3 = pool.mul(vec![pool.integer(-1_i32), r3]);
    let neg_half_r3 = pool.mul(vec![q(pool, -1, 2), r3]);
    vec![
        ((1, 1, 8), r3),
        ((2, 2, 8), r3),
        ((3, 3, 8), r3),
        ((4, 4, 8), neg_half_r3),
        ((5, 5, 8), neg_half_r3),
        ((6, 6, 8), neg_half_r3),
        ((7, 7, 8), neg_half_r3),
        ((1, 4, 6), h),
        ((1, 5, 7), h),
        ((2, 4, 7), mh),
        ((2, 5, 6), h),
        ((3, 4, 4), h),
        ((3, 5, 5), h),
        ((3, 6, 6), mh),
        ((3, 7, 7), mh),
        ((8, 8, 8), neg_r3),
    ]
}
