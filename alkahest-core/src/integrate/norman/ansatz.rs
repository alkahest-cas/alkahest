//! The Risch–Norman ansatz and the single linear system it reduces to.
//!
//! Given `f = N/Q` in the differential ring built by [`super::ring`], we posit
//!
//! ```text
//!     F  =  P/Q  +  Σⱼ dⱼ·log(pⱼ)
//! ```
//!
//! where `P` ranges over a bounded box of monomials in `ℚ[x, θ₁, …, θₙ]` and
//! the `pⱼ` are the irreducible factors of `Q` (plus `x` and the tower's own
//! logarithm generators, so that `log(log x)`-shaped answers are reachable).
//! Differentiating each ansatz atom gives a *known* rational function; the
//! unknowns enter only linearly.  Clearing a common denominator turns
//! `D(F) − f = 0` into one polynomial identity, and equating the coefficient
//! of every monomial gives a linear system over `ℚ`.
//!
//! Two things are worth being explicit about.
//!
//! * The construction is sound *by design* — we differentiate the candidate to
//!   build the system — but it is only *complete* to the extent that the
//!   monomial box and the logarithm candidate set happen to contain the
//!   answer.  A failure to solve is a failure of the ansatz, never a proof.
//! * Equating coefficients presumes the generators are algebraically
//!   independent.  [`super::ring`] establishes that before we get here.

use std::collections::{BTreeMap, BTreeSet};

use rug::{Integer, Rational};

use crate::kernel::{ExprId, ExprPool};
use crate::poly::multipoly::MultiPoly;
use crate::poly::rational::{mpoly_exact_div, RationalFunction};

use super::profile::{self, Phase};
use super::ring::{self, is_unit, lcm, GenKind, NormanRing};
use super::solve;
use super::DeclineReason;

// ---------------------------------------------------------------------------
// Caps
//
// Every number below is stated against a measurement.  The two corpora are the
// 103 integrands of `bench::CORPUS` (the 40-case probe plus 63 textbook cases —
// what this module is actually for) and the 18 deliberately oversized ones of
// `bench::STRESS`.  `cargo test -p alkahest-cas --release norman::bench --
// --ignored --nocapture` prints the observed maxima that these clear.
// ---------------------------------------------------------------------------

/// Largest number of unknowns (monomials plus logarithm candidates).
///
/// **Observed:** 30 (`1/(x·log x·log log x)`) on the corpus, 127 (`x⁴⁰·eˣ`) on
/// the stress set.  This is 8× the first and 1.9× the second.
///
/// It is the cap that binds first and the one that matters: the monomial box
/// is a *product* over generators, so it is the only quantity here that grows
/// combinatorially.  At 240 unknowns a `∫xⁿeˣ`-shaped integrand measures about
/// 4 ms end to end — an order of magnitude above the existing engine's median,
/// i.e. the point past which a cheap heuristic has stopped being cheap.
const MAX_UNKNOWNS: usize = 240;

/// Largest number of equations (distinct monomials in the cleared identity).
///
/// **Observed:** 41 on the corpus, 126 on the stress set — 146× and 48× of
/// headroom.
///
/// This cap is deliberately loose and is no longer the operative one.  Rows are
/// nearly free now that the elimination is sparse (an equation costs a bucket
/// entry, not a matrix row), so the ceiling that actually bounds solver work is
/// [`super::solve`]'s fill-in cap.  This one survives to stop the *matrix
/// build* from allocating without limit on an input that produces a huge
/// cleared identity.
const MAX_EQUATIONS: usize = 6000;

/// Largest number of terms tolerated in the cleared common denominator.
///
/// **Observed:** 7 on the corpus, 13 on the stress set.
///
/// The headroom looks absurd against those numbers and is not: the common
/// denominator is `L·Q²`, so it grows *quadratically* in the size of the
/// integrand's denominator, and 2 000 terms is reached by a `Q` of only about
/// 45 terms.  Every cleared column is then multiplied against it, so this is
/// the cap that stops a dense high-degree multivariate denominator from
/// turning one integrand into seconds of polynomial arithmetic.
const MAX_DENOM_TERMS: usize = 2000;

/// One term of the ansatz, together with its derivative.
struct Atom {
    /// The term itself, as an expression builder input.
    kind: AtomKind,
    /// Numerator of `D(term)` over [`Atom::den`].  Deliberately *not* reduced:
    /// the common-denominator step below would undo the reduction anyway, and
    /// each reduction costs a FLINT multivariate GCD.
    num: MultiPoly,
    /// Denominator of `D(term)`.
    den: Den,
}

/// Which denominator an atom's derivative sits over.
enum Den {
    /// The shared `L·Q²` of every rational atom, stored once by the caller.
    Rational,
    /// `L·p` for a `log(p)` atom.
    Log(MultiPoly),
}

enum AtomKind {
    /// `monomial / Q`.
    Rational(Vec<u32>),
    /// `log(p)`.
    Log(MultiPoly),
}

/// Build and solve the ansatz.  Returns the candidate antiderivative
/// expression, *unverified*.
pub(super) fn solve(
    ring: &NormanRing,
    f: &RationalFunction,
    pool: &ExprPool,
) -> Result<ExprId, DeclineReason> {
    let q = f.denom.clone();
    let n = f.numer.clone();

    // ---- monomial box for the numerator of the rational part.
    let (monomials, logs) = profile::timed(Phase::AnsatzSetup, || {
        let nv = ring.nvars();
        let mut bounds = Vec::with_capacity(nv);
        let mut count: usize = 1;
        for i in 0..nv {
            let b = (n.degree_in(i) + q.degree_in(i) + 1) as usize;
            count = count.saturating_mul(b + 1);
            if count > MAX_UNKNOWNS {
                return Err(DeclineReason::TooLarge("ansatz monomial box"));
            }
            bounds.push(b);
        }
        let monomials = monomial_box(&bounds);

        // ---- logarithm candidates.
        let logs = log_candidates(ring, &q)?;

        if monomials.len() + logs.len() > MAX_UNKNOWNS {
            return Err(DeclineReason::TooLarge("ansatz size"));
        }
        if monomials.is_empty() && logs.is_empty() {
            return Err(DeclineReason::NoSolution);
        }
        Ok((monomials, logs))
    })?;

    // ---- derivatives of every ansatz atom, over a *shared* denominator.
    //
    // With `D(vᵢ) = dnum[i]/L` (one `L` for the whole derivation table),
    //
    //     D(m/Q) = (D(m)·Q − m·D(Q)) / Q²
    //            = (Σᵢ eᵢ·(m/vᵢ)·dnum[i]·Q  −  m·D(Q)·L) / (L·Q²)
    //
    // so *every* rational atom lands over the same denominator `L·Q²`, and the
    // numerator is a sum of monomial shifts of the two precomputed polynomials
    // `dnum[i]·Q` and `deriv_scaled(Q)`.  No GCD, no rational-function
    // normalisation, no per-atom polynomial multiplication.  Logarithm atoms
    // are `D(log p) = deriv_scaled(p)/(L·p)`.
    let (atoms, rat_den) = profile::timed(Phase::AtomDerivs, || {
        let q2 = q.clone() * q.clone();
        let rat_den = ring.dden.clone() * q2;
        let nq = ring.deriv_scaled(&q);
        let nv = ring.nvars();
        let dnq: Vec<MultiPoly> = (0..nv).map(|i| ring.dnum[i].clone() * q.clone()).collect();

        let mut atoms: Vec<Atom> = Vec::with_capacity(monomials.len() + logs.len());
        for m in &monomials {
            let mut num = MultiPoly::zero(ring.vars.clone());
            for (i, dq) in dnq.iter().enumerate() {
                let e = m.get(i).copied().unwrap_or(0);
                if e == 0 {
                    continue;
                }
                let mut shift = m.clone();
                shift[i] = e - 1;
                num = num + ring::shift_scale(dq, &shift, &Integer::from(e));
            }
            num = num - ring::shift_scale(&nq, m, &Integer::from(1));
            atoms.push(Atom {
                kind: AtomKind::Rational(m.clone()),
                num,
                den: Den::Rational,
            });
        }
        for p in &logs {
            atoms.push(Atom {
                kind: AtomKind::Log(p.clone()),
                num: ring.deriv_scaled(p),
                den: Den::Log(ring.dden.clone() * p.clone()),
            });
        }
        (atoms, rat_den)
    });

    // ---- common denominator, then clear it.
    let (columns, rhs_poly) = profile::timed(Phase::ClearDenoms, || {
        // All the rational atoms share `L·Q²`, so the common denominator is
        // built from *distinct* denominators — one step per logarithm
        // candidate, not one per atom.
        //
        // Almost every candidate `p` is an irreducible factor of `Q`, so `L·p`
        // already divides `L·Q²` and the running lcm does not move.  The loop
        // therefore asks for the *quotient* directly and only falls back to a
        // GCD (and a restart, because a grown `common` invalidates the
        // quotients computed so far) for a candidate drawn from outside `Q`'s
        // factorisation — `x`, or one of the tower's own logarithm generators.
        // Each FLINT call carries a context init and two polynomial
        // round-trips, so halving their number is what this saves.
        let mut dens: Vec<&MultiPoly> = vec![&rat_den];
        for a in &atoms {
            if let Den::Log(d) = &a.den {
                dens.push(d);
            }
        }
        dens.push(&f.denom);

        let mut common = rat_den.clone();
        let mut scales: Option<Vec<MultiPoly>> = None;
        // Each restart absorbs one more denominator into `common` and never
        // un-absorbs one, so `dens.len() + 1` passes is a proof-carrying bound
        // rather than a guess.  It is written as a bounded loop anyway so that
        // a FLINT `divides`/`gcd` anomaly cannot hang the integrator.
        for _ in 0..=dens.len() {
            let mut built: Vec<MultiPoly> = Vec::with_capacity(dens.len());
            let mut grew = false;
            for d in &dens {
                match mpoly_exact_div(&common, d) {
                    Some(s) => built.push(s),
                    None => {
                        common = lcm(&common, d).ok_or(DeclineReason::RingArithmetic)?;
                        if common.terms.len() > MAX_DENOM_TERMS {
                            return Err(DeclineReason::TooLarge("common denominator"));
                        }
                        grew = true;
                        break;
                    }
                }
            }
            if !grew {
                scales = Some(built);
                break;
            }
        }
        let scales = scales.ok_or(DeclineReason::RingArithmetic)?;

        profile::record(|s| s.denom_terms = common.terms.len());
        let rhs_scale = scales.last().expect("f.denom was pushed");
        let mut columns: Vec<MultiPoly> = Vec::with_capacity(atoms.len());
        let mut next_log = 1usize;
        for a in &atoms {
            let scale = match &a.den {
                Den::Rational => &scales[0],
                Den::Log(_) => {
                    let s = &scales[next_log];
                    next_log += 1;
                    s
                }
            };
            columns.push(if is_unit(scale) {
                a.num.clone()
            } else {
                a.num.clone() * scale.clone()
            });
        }
        let rhs_poly = if is_unit(rhs_scale) {
            f.numer.clone()
        } else {
            f.numer.clone() * rhs_scale.clone()
        };
        Ok((columns, rhs_poly))
    })?;

    // ---- one equation per monomial of the cleared identity.
    //
    // Built column-sparse: the overwhelming majority of `(equation, atom)`
    // pairs are structurally zero, and materialising them as `Rational::from(0)`
    // costs both the allocation and, later, a full row of bignum work in the
    // elimination.  Inverting the loop — walk each column's terms once and
    // scatter them into the rows — is `O(non-zeros)` instead of `O(eqs·atoms)`.
    let system = profile::timed(Phase::MatrixBuild, || {
        let mut keys: BTreeSet<&Vec<u32>> = BTreeSet::new();
        for c in &columns {
            keys.extend(c.terms.keys());
        }
        keys.extend(rhs_poly.terms.keys());
        profile::record(|s| {
            s.equations = keys.len();
            s.unknowns = columns.len();
        });
        if keys.len() > MAX_EQUATIONS {
            return Err(DeclineReason::TooLarge("linear system"));
        }

        let index: BTreeMap<&Vec<u32>, usize> =
            keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let mut cells: Vec<Vec<(usize, Integer)>> = vec![Vec::new(); keys.len()];
        for (j, c) in columns.iter().enumerate() {
            for (k, v) in &c.terms {
                let row = *index.get(k).expect("every column key is in `keys`");
                cells[row].push((j, v.clone()));
            }
        }
        let mut sys = solve::SparseSystem::new(columns.len());
        for (i, k) in keys.iter().enumerate() {
            let rhs = rhs_poly
                .terms
                .get(*k)
                .cloned()
                .unwrap_or_else(|| Integer::from(0));
            // Columns are visited in increasing `j`, so each row's cells are
            // already sorted; `SparseSystem::push_row` asserts it in debug.
            sys.push_row(std::mem::take(&mut cells[i]), rhs);
        }
        Ok(sys)
    })?;

    // A degenerate (empty) system with a non-zero right-hand side cannot be
    // solved; an empty system with a zero right-hand side means `f = 0`.
    let sol = match profile::timed(Phase::Solve, || system.solve()) {
        solve::SolveOutcome::Solved(sol) => sol,
        solve::SolveOutcome::Inconsistent => return Err(DeclineReason::NoSolution),
        solve::SolveOutcome::GaveUp => return Err(DeclineReason::LinearSolver),
    };
    if sol.len() != atoms.len() {
        return Err(DeclineReason::NoSolution);
    }

    profile::timed(Phase::Rebuild, || {
        Ok(build_expression(ring, &atoms, &sol, &q, pool))
    })
}

/// Every exponent vector in the box `0 ≤ eᵢ ≤ bounds[i]`.
fn monomial_box(bounds: &[usize]) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![Vec::new()];
    for &b in bounds {
        let mut next = Vec::with_capacity(out.len() * (b + 1));
        for base in &out {
            for e in 0..=b {
                let mut v = base.clone();
                v.push(e as u32);
                next.push(v);
            }
        }
        out = next;
    }
    for v in out.iter_mut() {
        while v.last() == Some(&0) {
            v.pop();
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Logarithm arguments the ansatz may use.
///
/// The irreducible factors of the denominator (Rothstein–Trager's candidate
/// set, generalised to the tower), plus `x` and the tower's own logarithm
/// generators so that answers of the shape `log(log x)` are reachable.
fn log_candidates(ring: &NormanRing, q: &MultiPoly) -> Result<Vec<MultiPoly>, DeclineReason> {
    let mut out: Vec<MultiPoly> = Vec::new();
    let push = |p: MultiPoly, out: &mut Vec<MultiPoly>| {
        if p.total_degree() == 0 {
            return;
        }
        let neg = -p.clone();
        if !out.iter().any(|g| *g == p || *g == neg) {
            out.push(p);
        }
    };

    let (_unit, factors) = q
        .factor_irreducible()
        .ok_or(DeclineReason::RingArithmetic)?;
    for (f, _m) in factors {
        // An exponential generator is a unit: `log(exp η) = η` is already in
        // the field and adds nothing but noise to the output.
        if is_exp_monomial(ring, &f) {
            continue;
        }
        push(f, &mut out);
    }

    // `x` itself.
    push(ring.monomial(&[1]), &mut out);
    // Each logarithm generator, so that `log(log x)` is reachable.
    for (i, k) in ring.kinds.iter().enumerate() {
        if *k == GenKind::Log {
            let idx = i + 1;
            let mut e = vec![0u32; idx + 1];
            e[idx] = 1;
            push(ring.monomial(&e), &mut out);
        }
    }
    Ok(out)
}

/// `true` when `p` is a single monomial in exponential generators only.
fn is_exp_monomial(ring: &NormanRing, p: &MultiPoly) -> bool {
    if p.terms.len() != 1 {
        return false;
    }
    let (exp, _) = p.terms.iter().next().expect("exactly one term");
    let mut saw = false;
    for (i, &e) in exp.iter().enumerate() {
        if e == 0 {
            continue;
        }
        if i == 0 || ring.kinds.get(i - 1) != Some(&GenKind::Exp) {
            return false;
        }
        saw = true;
    }
    saw
}

/// A scalar multiple `c·e`, with `c = 1` elided and integral `c` emitted as an
/// integer rather than a rational node (which keeps `simplify` on the fast
/// path, and matters because the exactness of the `d/dx F = f` gate depends on
/// the answer normalising cleanly).
fn scale(c: &Rational, e: ExprId, pool: &ExprPool) -> ExprId {
    if *c == 1 {
        return e;
    }
    let coeff = if *c.denom() == 1 {
        pool.integer(c.numer().clone())
    } else {
        pool.rational(c.numer().clone(), c.denom().clone())
    };
    pool.mul(vec![coeff, e])
}

/// Assemble `P/Q + Σ dⱼ log(pⱼ)` from the solved coefficients.
///
/// `P` carries rational coefficients, so it is cleared to `ℤ` by the lcm of
/// their denominators and then reduced against `Q` through
/// [`RationalFunction::new`], which cancels the polynomial gcd.  Without that
/// reduction `∫dx/(x+1)²` would come back as `−(x+1)/(x+1)²`.
fn build_expression(
    ring: &NormanRing,
    atoms: &[Atom],
    sol: &[Rational],
    q: &MultiPoly,
    pool: &ExprPool,
) -> ExprId {
    let mut log_terms: Vec<ExprId> = Vec::new();

    // Common denominator of the rational part's coefficients.
    let mut lcm_den = rug::Integer::from(1);
    for (a, c) in atoms.iter().zip(sol.iter()) {
        if matches!(a.kind, AtomKind::Rational(_)) && *c != 0 {
            lcm_den = lcm_den.lcm(c.denom());
        }
    }

    let mut p_int = MultiPoly::zero(ring.vars.clone());
    for (a, c) in atoms.iter().zip(sol.iter()) {
        if *c == 0 {
            continue;
        }
        match &a.kind {
            AtomKind::Rational(exp) => {
                let scaled = c.clone() * Rational::from(lcm_den.clone());
                debug_assert_eq!(*scaled.denom(), 1);
                let mut term = ring.monomial(exp);
                for v in term.terms.values_mut() {
                    *v *= scaled.numer().clone();
                }
                p_int = p_int + term;
            }
            AtomKind::Log(p) => {
                let arg = p.to_expr(pool);
                let l = pool.func("log", vec![arg]);
                log_terms.push(scale(c, l, pool));
            }
        }
    }

    let mut parts: Vec<ExprId> = Vec::new();
    if !p_int.is_zero() {
        let reduced = RationalFunction::new(p_int, q.clone());
        let (n, d) = match reduced {
            Ok(rf) => (rf.numer, rf.denom),
            Err(_) => (MultiPoly::zero(ring.vars.clone()), ring.constant_poly(1)),
        };
        let n_expr = n.to_expr(pool);
        let body = if is_unit(&d) {
            n_expr
        } else {
            let inv = pool.pow(d.to_expr(pool), pool.integer(-1_i32));
            pool.mul(vec![n_expr, inv])
        };
        let s = Rational::from((rug::Integer::from(1), lcm_den));
        parts.push(scale(&s, body, pool));
    }
    parts.extend(log_terms);

    let raw = match parts.len() {
        0 => pool.integer(0_i32),
        1 => parts[0],
        _ => pool.add(parts),
    };
    crate::simplify::engine::simplify(raw, pool).value
}
