use super::*;
use crate::kernel::{Domain, ExprPool};
use crate::pattern::{match_pattern, Pattern};
use crate::simplify::rulesets::PatternRule;

fn nc(pool: &ExprPool, name: &str) -> ExprId {
    pool.symbol_commutative(name, Domain::Complex, false)
}

fn int(pool: &ExprPool, n: i64) -> ExprId {
    pool.integer(n)
}

fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

/// Two expressions are equal as noncommutative polynomials.
fn same(pool: &ExprPool, a: ExprId, b: ExprId) -> bool {
    nc_terms(pool.add(vec![a, neg(pool, b)]), pool)
        .unwrap()
        .is_empty()
}

#[test]
fn expand_square_of_sum_keeps_order() {
    let p = ExprPool::new();
    let (x, y) = (nc(&p, "X"), nc(&p, "Y"));
    let sq = p.pow(p.add(vec![x, y]), int(&p, 2));
    let t = nc_terms(sq, &p).unwrap();
    let words: Vec<Vec<ExprId>> = t.iter().map(|(_, w)| w.clone()).collect();
    assert_eq!(t.len(), 4);
    for w in [vec![x, x], vec![x, y], vec![y, x], vec![y, y]] {
        assert!(words.contains(&w));
    }
}

#[test]
fn expand_collects_coefficients() {
    let p = ExprPool::new();
    let (x, y) = (nc(&p, "X"), nc(&p, "Y"));
    let c = p.symbol("c", Domain::Real);
    // c·X·Y + 2·X·Y − X·Y·c  =  2·X·Y
    let e = p.add(vec![
        p.mul(vec![c, x, y]),
        p.mul(vec![int(&p, 2), x, y]),
        p.mul(vec![int(&p, -1), x, y, c]),
    ]);
    let t = nc_terms(e, &p).unwrap();
    assert_eq!(t, vec![(int(&p, 2), vec![x, y])]);
}

#[test]
fn commutator_rule_inside_longer_words() {
    // [X, P] = iħ  ⇒  X·P·P − P·P·X = 2iħ·P
    let p = ExprPool::new();
    let (x, pp) = (nc(&p, "X"), nc(&p, "P"));
    let hb = p.symbol("hbar", Domain::Real);
    let i = p.imaginary_unit();
    let rhs = p.add(vec![p.mul(vec![pp, x]), p.mul(vec![i, hb])]);
    let e = p.add(vec![
        p.mul(vec![x, pp, pp]),
        neg(&p, p.mul(vec![pp, pp, x])),
    ]);
    let out = nc_rewrite(e, &[(p.mul(vec![x, pp]), rhs)], &p, DEFAULT_MAX_STEPS).unwrap();
    let want = p.mul(vec![int(&p, 2), i, hb, pp]);
    assert!(same(&p, out, want), "got {}", p.display(out));
}

#[test]
fn nonterminating_rules_refuse() {
    let p = ExprPool::new();
    let (a, b) = (nc(&p, "A"), nc(&p, "B"));
    let rules = [
        (p.mul(vec![a, b]), p.mul(vec![b, a])),
        (p.mul(vec![b, a]), p.mul(vec![a, b])),
    ];
    let err = nc_rewrite(p.mul(vec![a, b]), &rules, &p, 1000).unwrap_err();
    assert!(matches!(err, NcError::StepLimit { .. }));
}

#[test]
fn rule_lhs_must_be_a_word() {
    let p = ExprPool::new();
    let (a, b) = (nc(&p, "A"), nc(&p, "B"));
    let err = nc_rewrite(a, &[(p.add(vec![a, b]), a)], &p, 10).unwrap_err();
    assert!(matches!(err, NcError::InvalidRule { .. }));
}

#[test]
fn pauli_words_reduce_completely() {
    let p = ExprPool::new();
    let (sx, sy, sz) = (nc(&p, "sx"), nc(&p, "sy"), nc(&p, "sz"));
    let i = p.imaginary_unit();
    let two = int(&p, 2);
    // σz² = 1
    assert_eq!(pauli_reduce(p.pow(sz, two), &p).unwrap(), int(&p, 1));
    // σxσyσx = −σy
    let r = pauli_reduce(p.mul(vec![sx, sy, sx]), &p).unwrap();
    assert!(same(&p, r, neg(&p, sy)), "got {}", p.display(r));
    // (σx + σy)² = 2
    let r = pauli_reduce(p.pow(p.add(vec![sx, sy]), two), &p).unwrap();
    assert_eq!(r, two);
    // [σx, σy] = 2iσz
    let c = commutator(sx, sy, &p).unwrap();
    let r = pauli_reduce(c, &p).unwrap();
    assert!(same(&p, r, p.mul(vec![two, i, sz])), "got {}", p.display(r));
    // σxσyσz = i
    let r = pauli_reduce(p.mul(vec![sx, sy, sz]), &p).unwrap();
    assert_eq!(r, i);
}

#[test]
fn pauli_ignores_commutative_namesakes() {
    let p = ExprPool::new();
    let sx = p.symbol("sx", Domain::Complex);
    let e = p.pow(sx, int(&p, 2));
    assert_eq!(pauli_reduce(e, &p).unwrap(), e);
}

#[test]
fn boson_normal_order() {
    let p = ExprPool::new();
    let (a, ad) = (nc(&p, "a"), nc(&p, "ad"));
    let modes = [LadderMode {
        annihilator: a,
        creator: ad,
        fermionic: false,
    }];
    // a·a† = a†·a + 1
    let r = normal_order(p.mul(vec![a, ad]), &modes, &p).unwrap();
    assert!(same(&p, r, p.add(vec![p.mul(vec![ad, a]), int(&p, 1)])));
    // [a, a†³] = 3 a†²
    let ad3 = p.pow(ad, int(&p, 3));
    let c = commutator(a, ad3, &p).unwrap();
    let r = normal_order(c, &modes, &p).unwrap();
    assert!(same(&p, r, p.mul(vec![int(&p, 3), p.pow(ad, int(&p, 2))])));
    // ⟨0| a a a† a† |0⟩ = 2
    let v = vacuum_expectation(p.mul(vec![a, a, ad, ad]), &modes, &p).unwrap();
    assert_eq!(v, int(&p, 2));
}

#[test]
fn fermion_normal_order() {
    let p = ExprPool::new();
    let (c, cd, d, dd) = (nc(&p, "c"), nc(&p, "cd"), nc(&p, "d"), nc(&p, "dd"));
    let modes = [
        LadderMode {
            annihilator: c,
            creator: cd,
            fermionic: true,
        },
        LadderMode {
            annihilator: d,
            creator: dd,
            fermionic: true,
        },
    ];
    // c·c† = 1 − c†·c
    let r = normal_order(p.mul(vec![c, cd]), &modes, &p).unwrap();
    assert!(same(
        &p,
        r,
        p.add(vec![int(&p, 1), neg(&p, p.mul(vec![cd, c]))])
    ));
    // c² = 0, c†² = 0
    assert_eq!(
        normal_order(p.mul(vec![c, c]), &modes, &p).unwrap(),
        int(&p, 0)
    );
    assert_eq!(
        normal_order(p.mul(vec![cd, cd]), &modes, &p).unwrap(),
        int(&p, 0)
    );
    // d·c† = −c†·d ; d†·c† = −c†·d†
    let r = normal_order(p.mul(vec![d, cd]), &modes, &p).unwrap();
    assert!(same(&p, r, neg(&p, p.mul(vec![cd, d]))));
    let r = normal_order(p.mul(vec![dd, cd]), &modes, &p).unwrap();
    assert!(same(&p, r, neg(&p, p.mul(vec![cd, dd]))));
    // ⟨0| c c† |0⟩ = 1
    assert_eq!(
        vacuum_expectation(p.mul(vec![c, cd]), &modes, &p).unwrap(),
        int(&p, 1)
    );
}

#[test]
fn ladder_spec_is_validated() {
    let p = ExprPool::new();
    let a = p.symbol("a", Domain::Complex);
    let ad = nc(&p, "ad");
    let modes = [LadderMode {
        annihilator: a,
        creator: ad,
        fermionic: false,
    }];
    assert!(matches!(
        normal_order(ad, &modes, &p),
        Err(NcError::InvalidOperator { .. })
    ));
    let b = nc(&p, "b");
    let modes = [LadderMode {
        annihilator: b,
        creator: b,
        fermionic: false,
    }];
    assert!(matches!(
        normal_order(b, &modes, &p),
        Err(NcError::InvalidOperator { .. })
    ));
}

#[test]
fn vacuum_expectation_refuses_foreign_atoms() {
    let p = ExprPool::new();
    let (a, ad, x) = (nc(&p, "a"), nc(&p, "ad"), nc(&p, "X"));
    let modes = [LadderMode {
        annihilator: a,
        creator: ad,
        fermionic: false,
    }];
    assert!(matches!(
        vacuum_expectation(p.mul(vec![a, x, ad]), &modes, &p),
        Err(NcError::UnknownOperator { .. })
    ));
}

// --- Dirac traces against explicit 4×4 matrices -------------------------

/// Gaussian-integer 4×4 matrices, (re, im).
type M4 = [[(i64, i64); 4]; 4];

fn cmul(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn mmul(a: &M4, b: &M4) -> M4 {
    let mut c = [[(0, 0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = (0, 0);
            for k in 0..4 {
                let t = cmul(a[i][k], b[k][j]);
                s = (s.0 + t.0, s.1 + t.1);
            }
            c[i][j] = s;
        }
    }
    c
}

/// Dirac representation: γ⁰ = diag(1,1,−1,−1), γᵏ = [[0, σk], [−σk, 0]],
/// γ⁵ = iγ⁰γ¹γ²γ³.
fn dirac(k: u8) -> M4 {
    let z = (0, 0);
    let o = (1, 0);
    let m = (-1, 0);
    let i = (0, 1);
    let mi = (0, -1);
    match k {
        0 => [[o, z, z, z], [z, o, z, z], [z, z, m, z], [z, z, z, m]],
        1 => [[z, z, z, o], [z, z, o, z], [z, m, z, z], [m, z, z, z]],
        2 => [[z, z, z, mi], [z, z, i, z], [z, i, z, z], [mi, z, z, z]],
        3 => [[z, z, o, z], [z, z, z, m], [m, z, z, z], [z, o, z, z]],
        _ => {
            let p = mmul(&mmul(&dirac(0), &dirac(1)), &mmul(&dirac(2), &dirac(3)));
            let mut out = [[z; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    out[r][c] = cmul(i, p[r][c]);
                }
            }
            out
        }
    }
}

fn explicit_trace(ks: &[u8]) -> (i64, i64) {
    let mut m = [[(0, 0); 4]; 4];
    for (d, row) in m.iter_mut().enumerate() {
        row[d] = (1, 0);
    }
    for &k in ks {
        m = mmul(&m, &dirac(k));
    }
    (0..4).fold((0, 0), |s, d| (s.0 + m[d][d].0, s.1 + m[d][d].1))
}

/// Read a Gaussian integer back out of `gamma_trace`'s result.
fn as_gaussian(p: &ExprPool, e: ExprId) -> (i64, i64) {
    if let Some(n) = small_int(e, p) {
        return (n, 0);
    }
    if p.is_imaginary_unit(e) {
        return (0, 1);
    }
    let args = p.with(e, |d| match d {
        ExprData::Mul(v) => v.clone(),
        _ => panic!("unexpected trace {}", p.display(e)),
    });
    assert_eq!(args.len(), 2, "unexpected trace {}", p.display(e));
    let (n, imag) = if p.is_imaginary_unit(args[1]) {
        (small_int(args[0], p), args[1])
    } else {
        (small_int(args[1], p), args[0])
    };
    assert!(p.is_imaginary_unit(imag));
    (0, n.expect("integer coefficient"))
}

fn slot(k: u8) -> GammaSlot {
    if k == 5 {
        GammaSlot::Gamma5
    } else {
        GammaSlot::Index(k)
    }
}

#[test]
fn explicit_traces_match_dirac_matrices() {
    let p = ExprPool::new();
    let alphabet = [0u8, 1, 2, 3, 5];
    // every word of length ≤ 4, and every word of length 6 over a sparser set
    let mut words: Vec<Vec<u8>> = vec![vec![]];
    let mut frontier: Vec<Vec<u8>> = vec![vec![]];
    for _ in 0..4 {
        let mut next = Vec::new();
        for w in &frontier {
            for &k in &alphabet {
                let mut v = w.clone();
                v.push(k);
                next.push(v);
            }
        }
        words.extend(next.iter().cloned());
        frontier = next;
    }
    // length-6 and length-8 samples (deterministic LCG)
    let mut s: u64 = 12345;
    for len in [5usize, 6, 7, 8] {
        for _ in 0..400 {
            let w: Vec<u8> = (0..len)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    alphabet[((s >> 33) % 5) as usize]
                })
                .collect();
            words.push(w);
        }
    }
    for w in words {
        let slots: Vec<GammaSlot> = w.iter().map(|&k| slot(k)).collect();
        let got = gamma_trace(&slots, &p).unwrap();
        assert_eq!(
            as_gaussian(&p, got),
            explicit_trace(&w),
            "tr {:?} = {}",
            w,
            p.display(got)
        );
    }
}

#[test]
fn symbolic_traces() {
    let p = ExprPool::new();
    let (a, b, c, d) = (
        p.symbol("p", Domain::Real),
        p.symbol("q", Domain::Real),
        p.symbol("k", Domain::Real),
        p.symbol("l", Domain::Real),
    );
    let v = GammaSlot::Vector;
    let dot = |x: ExprId, y: ExprId| {
        let (x, y) = if x <= y { (x, y) } else { (y, x) };
        p.func("dot", vec![x, y])
    };
    // tr(p̸ q̸) = 4 p·q
    let t = gamma_trace(&[v(a), v(b)], &p).unwrap();
    assert_eq!(t, p.mul(vec![int(&p, 4), dot(a, b)]));
    // tr(p̸ q̸ k̸ l̸) = 4[(pq)(kl) − (pk)(ql) + (pl)(qk)]
    let t = gamma_trace(&[v(a), v(b), v(c), v(d)], &p).unwrap();
    let want = p.add(vec![
        p.mul(vec![int(&p, 4), dot(a, b), dot(c, d)]),
        p.mul(vec![int(&p, -4), dot(a, c), dot(b, d)]),
        p.mul(vec![int(&p, 4), dot(a, d), dot(b, c)]),
    ]);
    let diff = crate::simplify::engine::simplify(p.add(vec![t, neg(&p, want)]), &p).value;
    assert_eq!(diff, int(&p, 0), "got {}", p.display(t));
    // odd number of slots
    assert_eq!(gamma_trace(&[v(a), v(b), v(c)], &p).unwrap(), int(&p, 0));
    // γ⁵ with four vectors: −4i ε(p,q,k,l)
    let t = gamma_trace(&[GammaSlot::Gamma5, v(a), v(b), v(c), v(d)], &p).unwrap();
    let eps = p.func("levi_civita", vec![a, b, c, d]);
    let want = p.mul(vec![int(&p, -4), eps, p.imaginary_unit()]);
    let diff = crate::simplify::engine::simplify(p.add(vec![t, neg(&p, want)]), &p).value;
    assert_eq!(diff, int(&p, 0), "got {}", p.display(t));
    // a repeated vector kills the ε
    assert_eq!(
        gamma_trace(&[GammaSlot::Gamma5, v(a), v(a), v(c), v(d)], &p).unwrap(),
        int(&p, 0)
    );
    // too long
    let long: Vec<GammaSlot> = (0..16).map(|_| v(a)).collect();
    assert!(matches!(
        gamma_trace(&long, &p),
        Err(NcError::GammaTraceTooLong { .. })
    ));
    assert!(matches!(
        gamma_trace(&[GammaSlot::Index(4)], &p),
        Err(NcError::InvalidGamma { .. })
    ));
}

// --- pattern rules inside products ---------------------------------------

fn apply_rule(p: &ExprPool, lhs: ExprId, rhs: ExprId, e: ExprId) -> ExprId {
    let rule: Vec<Box<dyn RewriteRule>> =
        vec![Box::new(PatternRule::new(Pattern::from_expr(lhs), rhs))];
    simplify_with(e, p, &rule, SimplifyConfig::default()).value
}

#[test]
fn commutative_rule_fires_on_sub_multiset() {
    let p = ExprPool::new();
    let (a, b, c, d) = (
        p.symbol("A", Domain::Real),
        p.symbol("B", Domain::Real),
        p.symbol("C", Domain::Real),
        p.symbol("D", Domain::Real),
    );
    let out = apply_rule(&p, p.mul(vec![a, b]), c, p.mul(vec![a, b, d]));
    assert_eq!(out, p.mul(vec![c, d]));
    // order in the product does not matter
    let out = apply_rule(&p, p.mul(vec![a, b]), c, p.mul(vec![d, b, a]));
    assert_eq!(out, p.mul(vec![c, d]));
    // nothing to match
    let e = p.mul(vec![a, c, d]);
    assert_eq!(apply_rule(&p, p.mul(vec![a, b]), c, e), e);
}

#[test]
fn noncommutative_rule_fires_on_contiguous_run_only() {
    let p = ExprPool::new();
    let (x, pp, y) = (nc(&p, "X"), nc(&p, "P"), nc(&p, "Y"));
    // X·P → Y inside X·P·P
    let out = apply_rule(&p, p.mul(vec![x, pp]), y, p.mul(vec![x, pp, pp]));
    assert_eq!(out, p.mul(vec![y, pp]));
    // not inside P·X·... (no contiguous X·P)
    let e = p.mul(vec![pp, x, y]);
    assert_eq!(apply_rule(&p, p.mul(vec![x, pp]), y, e), e);
    // not across a gap
    let e = p.mul(vec![x, y, pp]);
    assert_eq!(apply_rule(&p, p.mul(vec![x, pp]), y, e), e);
}

#[test]
fn lowercase_noncommutative_symbols_are_not_wildcards() {
    let p = ExprPool::new();
    let (a, ad) = (nc(&p, "a"), nc(&p, "ad"));
    let (x, y) = (nc(&p, "X"), nc(&p, "Y"));
    let rhs = p.add(vec![p.mul(vec![ad, a]), int(&p, 1)]);
    // X·Y is not a·a†
    let e = p.mul(vec![x, y]);
    assert_eq!(apply_rule(&p, p.mul(vec![a, ad]), rhs, e), e);
}

#[test]
fn simplify_with_noncommutative_reaches_normal_form() {
    let p = ExprPool::new();
    let (x, pp) = (nc(&p, "X"), nc(&p, "P"));
    let hb = p.symbol("hbar", Domain::Real);
    let i = p.imaginary_unit();
    let rhs = p.add(vec![p.mul(vec![pp, x]), p.mul(vec![i, hb])]);
    let rules: Vec<Box<dyn RewriteRule>> = vec![Box::new(PatternRule::new(
        Pattern::from_expr(p.mul(vec![x, pp])),
        rhs,
    ))];
    let e = p.add(vec![
        p.mul(vec![x, pp, pp]),
        neg(&p, p.mul(vec![pp, pp, x])),
    ]);
    let out = simplify_with_noncommutative(e, &p, &rules, SimplifyConfig::default()).value;
    assert!(
        same(&p, out, p.mul(vec![int(&p, 2), i, hb, pp])),
        "got {}",
        p.display(out)
    );
}

#[test]
fn match_pattern_respects_order() {
    let p = ExprPool::new();
    let (x, pp) = (nc(&p, "X"), nc(&p, "P"));
    let pat = Pattern::from_expr(p.mul(vec![x, pp]));
    assert!(match_pattern(&pat, p.mul(vec![pp, x]), &p).is_empty());
    assert!(!match_pattern(&pat, p.mul(vec![x, pp, pp]), &p).is_empty());
    // commutative sub-multiset
    let (a, b, d) = (
        p.symbol("A", Domain::Real),
        p.symbol("B", Domain::Real),
        p.symbol("D", Domain::Real),
    );
    let pat = Pattern::from_expr(p.mul(vec![a, b]));
    assert!(!match_pattern(&pat, p.mul(vec![a, b, d]), &p).is_empty());
    // Nested in a function, the product must match whole.
    let f = |e| p.func("f", vec![e]);
    let pat = Pattern::from_expr(f(p.mul(vec![a, b])));
    assert!(match_pattern(&pat, f(p.mul(vec![a, b, d])), &p).is_empty());
    let pat = Pattern::from_expr(f(p.mul(vec![x, pp])));
    assert!(match_pattern(&pat, f(p.mul(vec![x, pp, pp])), &p).is_empty());
    // ...but a match is still found at the inner node itself.
    let pat = Pattern::from_expr(p.mul(vec![x, pp]));
    assert!(!match_pattern(&pat, f(p.mul(vec![x, pp, pp])), &p).is_empty());
    // Sums keep their previous behaviour: no partial match without a wildcard.
    let pat = Pattern::from_expr(p.add(vec![a, b]));
    assert!(match_pattern(&pat, p.add(vec![a, b, d]), &p).is_empty());
}

#[test]
fn noncommutative_rule_sees_through_integer_powers() {
    let p = ExprPool::new();
    let (x, pp, y) = (nc(&p, "X"), nc(&p, "P"), nc(&p, "Y"));
    // X·P² contains X·P
    let e = p.mul(vec![x, p.pow(pp, int(&p, 2))]);
    assert_eq!(apply_rule(&p, p.mul(vec![x, pp]), y, e), p.mul(vec![y, pp]));
    // P²·X does not
    let e = p.mul(vec![p.pow(pp, int(&p, 2)), x]);
    assert_eq!(apply_rule(&p, p.mul(vec![x, pp]), y, e), e);
}
