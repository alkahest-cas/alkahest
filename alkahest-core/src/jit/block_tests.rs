//! The block interpreter (F14): the interpreter tier evaluates a batch a
//! block of points at a time, `Piecewise` included, and must give exactly
//! what the single-point evaluators give — the strict program's own `run`,
//! `eval_interp`, the snapshot tree walk the tier used for `Piecewise`
//! before (where that walk had a value), and, where one is built, a native
//! backend.

use super::*;
use crate::kernel::Domain;
use proptest::prelude::*;

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The block power kernel (exponent case decided once, integer powers eight
/// lanes at a time, scalar tail) equals the scalar kernel element by element.
#[test]
fn power_slices_equal_the_scalar_kernel_bit_for_bit() {
    use crate::kernel::eval_const::pow_f64_slice;
    let mut xs = special_grid(203, 99);
    xs.extend([
        -0.0,
        f64::MAX,
        -f64::MAX,
        5e-324,
        -5e-324,
        1.0 + f64::EPSILON,
        -3.7e15,
        2.5e-160,
    ]);
    let mut exps: Vec<f64> = (-70..=70).map(f64::from).collect();
    exps.extend([
        0.5,
        -0.5,
        1.5,
        -2.25,
        1e300,
        -1e300,
        f64::NAN,
        f64::INFINITY,
    ]);
    for len in [xs.len(), 7, 8, 9, 1] {
        let base = &xs[..len];
        let mut out = vec![0.0; len];
        for &e in &exps {
            pow_f64_slice(base, e, None, &mut out);
            for (i, &b) in base.iter().enumerate() {
                let want = pow_f64(b, e, || None);
                assert!(
                    same(out[i], want),
                    "{b:e}^{e}: slice {:e} vs scalar {want:e}",
                    out[i]
                );
            }
        }
    }
}

fn interp(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> CompiledFn {
    compile_with(
        expr,
        inputs,
        pool,
        CompileConfig {
            expected_evals: None,
            force_tier: Some(CompileTier::Interpreter),
        },
    )
    .unwrap()
}

fn is_program(f: &CompiledFn) -> bool {
    matches!(f.inner, CompiledFnInner::Program(_))
}

/// The tree walk the interpreter tier used for `Piecewise` before this
/// change, at one point.
fn snapshot_reference(expr: ExprId, inputs: &[ExprId], pool: &ExprPool, pt: &[f64]) -> f64 {
    let snap = snapshot_expr(expr, pool);
    let env: HashMap<ExprId, f64> = inputs.iter().copied().zip(pt.iter().copied()).collect();
    let mut memo: IdMap<f64> = IdMap::default();
    eval_interp_snap(expr, &env, &snap, &mut memo).unwrap_or(f64::NAN)
}

/// Values every evaluator must agree on, IEEE specials included, spread so
/// that a batch spans several blocks and ends on a partial one.
fn special_grid(n: usize, seed: u64) -> Vec<f64> {
    const SPECIAL: [f64; 9] = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        1e300,
    ];
    let mut state = seed | 1;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if i % 11 == 0 {
                SPECIAL[(state % SPECIAL.len() as u64) as usize]
            } else {
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 8.0 - 4.0
            }
        })
        .collect()
}

#[test]
fn block_batch_matches_single_point_run() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let xs = special_grid(1000, 7);
    let ys = special_grid(1000, 11);
    for (name, expr) in super::batch_tests::corpus(&pool, x, y) {
        let f = interp(expr, &[x, y], &pool);
        assert!(is_program(&f), "{name}");
        let mut out = vec![0.0; xs.len()];
        f.call_batch(&[&xs, &ys], &mut out);
        let mut bulk = vec![0.0; xs.len()];
        let mut flat = xs.clone();
        flat.extend_from_slice(&ys);
        f.call_bulk(&flat, &mut bulk);
        let env_of = |i: usize| HashMap::from([(x, xs[i]), (y, ys[i])]);
        for i in 0..xs.len() {
            let one = f.call(&[xs[i], ys[i]]);
            let walk = eval_interp(expr, &env_of(i), &pool).unwrap_or(f64::NAN);
            assert!(
                same(out[i], one) && same(out[i], walk) && same(bulk[i], one),
                "{name} at ({}, {}): block {:?}, bulk {:?}, point {one:?}, eval_interp {walk:?}",
                xs[i],
                ys[i],
                out[i],
                bulk[i]
            );
        }
    }
}

/// `Piecewise` of a function that has no value on part of the line.
fn lambert(pool: &ExprPool, x: ExprId) -> ExprId {
    pool.func("lambert_w", vec![x])
}

#[test]
fn piecewise_compiles_to_the_block_program() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let zero = pool.integer(0_i32);
    let pw = pool.piecewise(
        vec![(pool.pred_gt(x, zero), pool.func("sin", vec![x]))],
        pool.pow(x, pool.integer(2_i32)),
    );
    let e = pool.add(vec![pw, pool.integer(1_i32)]);
    for expr in [pw, e] {
        let f = compile(expr, &[x], &pool).unwrap();
        assert!(is_program(&f), "Piecewise kept the tree walk");
    }
}

#[test]
fn a_branch_no_point_selects_cannot_fail_it() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let zero = pool.integer(0_i32);
    let one = pool.integer(1_i32);
    // lambert_w has no value below -1/e; the branch is only taken for x > 0.
    let guarded = pool.piecewise(vec![(pool.pred_gt(x, zero), lambert(&pool, x))], one);
    // The failing kernel in a *condition* fails the point that consults it.
    let in_cond = pool.piecewise(
        vec![(pool.pred_gt(lambert(&pool, x), zero), pool.integer(2_i32))],
        one,
    );
    // Short circuits: `x > 0 ∧ W(x) > 0` never consults W where x ≤ 0;
    // `x < 0 ∨ W(x) > 0` never consults it where x < 0; `x > 0 ∨ W(x) > 0`
    // consults it at every x ≤ 0.
    let w_pos = pool.pred_gt(lambert(&pool, x), zero);
    let and = pool.piecewise(
        vec![(
            pool.pred_and(vec![pool.pred_gt(x, zero), w_pos]),
            pool.integer(3_i32),
        )],
        one,
    );
    let or_safe = pool.piecewise(
        vec![(
            pool.pred_or(vec![pool.pred_lt(x, zero), w_pos]),
            pool.integer(4_i32),
        )],
        one,
    );
    let or_fails = pool.piecewise(
        vec![(
            pool.pred_or(vec![pool.pred_gt(x, zero), w_pos]),
            pool.integer(4_i32),
        )],
        one,
    );
    let not = pool.piecewise(vec![(pool.pred_not(w_pos), pool.integer(5_i32))], one);
    let w = |v: f64| registry().numeric_f64("lambert_w", &[v]).unwrap();
    let nan = f64::NAN;
    // Hand-derived at finite points (W(-5), W(-1) have no value; W(-0.2) < 0,
    // W(0) = 0, W(0.5), W(3) > 0); NaN and ±inf are checked against the walk.
    let xs = [-5.0, -1.0, -0.2, 0.0, 0.5, 3.0];
    let cases: [(&str, ExprId, [f64; 6]); 6] = [
        ("guarded", guarded, [1.0, 1.0, 1.0, 1.0, w(0.5), w(3.0)]),
        ("in_cond", in_cond, [nan, nan, 1.0, 1.0, 2.0, 2.0]),
        ("and", and, [1.0, 1.0, 1.0, 1.0, 3.0, 3.0]),
        ("or_safe", or_safe, [4.0, 4.0, 4.0, 1.0, 4.0, 4.0]),
        ("or_fails", or_fails, [nan, nan, 1.0, 1.0, 4.0, 4.0]),
        ("not", not, [nan, nan, 5.0, 5.0, 1.0, 1.0]),
    ];
    let all = [
        -5.0,
        -1.0,
        -0.2,
        0.0,
        0.5,
        3.0,
        nan,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for (name, expr, want) in cases {
        let f = interp(expr, &[x], &pool);
        assert!(is_program(&f), "{name}");
        let mut out = vec![0.0; all.len()];
        f.call_batch(&[&all], &mut out);
        for (i, &xv) in all.iter().enumerate() {
            let reference = snapshot_reference(expr, &[x], &pool, &[xv]);
            let hand = if i < xs.len() { want[i] } else { reference };
            assert!(
                same(out[i], hand) && same(out[i], reference),
                "{name} at {xv}: block {:?}, want {hand:?}, tree walk {reference:?}",
                out[i],
            );
            assert!(
                same(f.call(&[xv]), out[i]),
                "{name} at {xv}: point vs block"
            );
        }
    }
}

#[test]
fn unbound_symbol_refusal_is_unchanged_for_piecewise() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let z = pool.symbol("z", Domain::Real);
    let zero = pool.integer(0_i32);
    // `z` sits in a branch some points never select: still refused.
    let pw = pool.piecewise(vec![(pool.pred_gt(x, zero), z)], x);
    for tier in [None, Some(CompileTier::Interpreter)] {
        let r = compile_with(
            pw,
            &[x],
            &pool,
            CompileConfig {
                expected_evals: None,
                force_tier: tier,
            },
        );
        assert!(r.is_err(), "{tier:?}");
        let refusal = take_unbound_symbols().expect("an E-JIT-005 refusal");
        assert_eq!(refusal.names(), ["z".to_string()]);
    }
    // An undefined function in a branch: E-JIT-006, as before.
    let f = pool.piecewise(
        vec![(pool.pred_gt(x, zero), pool.func("no_such_head", vec![x]))],
        x,
    );
    assert!(compile(f, &[x], &pool).is_err());
    assert!(take_unevaluable_functions().is_some());
}

/// The program itself (below the compile-time refusals): an unbound symbol or
/// an unknown head in a branch fails exactly the points that select it.
#[test]
fn lazy_program_fails_only_points_that_select_a_valueless_branch() {
    use crate::eval::program::NumericProgram;
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let z = pool.symbol("z", Domain::Real);
    let zero = pool.integer(0_i32);
    for bad in [z, pool.func("no_such_head", vec![x])] {
        let pw = pool.piecewise(vec![(pool.pred_gt(x, zero), bad)], x);
        let prog = NumericProgram::compile_lazy(pw, &[x], &pool).unwrap();
        assert_eq!(prog.eval(&[-2.0]), Some(-2.0));
        assert_eq!(prog.eval(&[2.0]), None);
        let mut out = [0.0; 3];
        prog.eval_columns(&[-1.0, 1.0, -3.0], &mut out);
        assert_eq!(out[0], -1.0);
        assert!(out[1].is_nan());
        assert_eq!(out[2], -3.0);
    }
}

/// Forms the old tree walk had no value for, although `eval_interp` does: a
/// `Piecewise` compared inside a condition, a predicate used as a value, and
/// a `RootOf` constant in a `Piecewise`. The block program gives
/// `eval_interp`'s value.
#[test]
fn piecewise_forms_the_old_walk_dropped_now_have_eval_interps_value() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let zero = pool.integer(0_i32);
    let pw = pool.piecewise(vec![(pool.pred_gt(x, zero), x)], zero);
    let nested_cond = pool.piecewise(vec![(pool.pred_lt(pw, pool.integer(1_i32)), x)], zero);
    let pred_value = pool.add(vec![pool.pred_gt(x, zero), x]);
    let mut cases = vec![("nested_cond", nested_cond), ("pred_value", pred_value)];
    // The positive root of t^2 - 2 (when this FLINT build isolates roots).
    let t = pool.symbol("t", Domain::Real);
    let quad = pool.add(vec![pool.pow(t, pool.integer(2_i32)), pool.integer(-2_i32)]);
    if let Ok(sqrt2) = crate::poly::root_of::root_of(quad, t, 1, &pool) {
        cases.push((
            "with_root",
            pool.piecewise(vec![(pool.pred_gt(x, sqrt2), sqrt2)], x),
        ));
    }
    for (name, e) in cases {
        let f = interp(e, &[x], &pool);
        assert!(is_program(&f), "{name}");
        let xs = [-1.0, 0.5, 2.0, f64::NAN, f64::INFINITY];
        let mut out = vec![0.0; xs.len()];
        f.call_batch(&[&xs], &mut out);
        for (i, &xv) in xs.iter().enumerate() {
            let want = eval_interp(e, &HashMap::from([(x, xv)]), &pool).unwrap_or(f64::NAN);
            assert!(
                same(out[i], want) && same(f.call(&[xv]), want),
                "{name} at {xv}: block {:?}, eval_interp {want:?}",
                out[i]
            );
        }
        assert!(out[1].is_finite(), "{name} at 0.5 has a value");
    }
}

/// A condition that is not a predicate has no truth value anywhere; that
/// form keeps the tree walk, and gives what it gave.
#[test]
fn a_condition_that_is_not_a_predicate_keeps_the_tree_walk() {
    use crate::eval::program::NumericProgram;
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let bare_cond = pool.piecewise(vec![(x, x)], pool.integer(0_i32));
    assert!(NumericProgram::compile_lazy(bare_cond, &[x], &pool).is_none());
    let f = interp(bare_cond, &[x], &pool);
    assert!(!is_program(&f));
    for xv in [-1.0, 0.5, 2.0, f64::NAN] {
        let reference = snapshot_reference(bare_cond, &[x], &pool, &[xv]);
        assert!(same(f.call(&[xv]), reference));
    }
}

// ---------------------------------------------------------------------------
// Random expressions: block ≡ single point ≡ eval_interp ≡ JIT
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Tree {
    X,
    Y,
    Int(i32),
    Rat(i32, i32),
    Add(Vec<Tree>),
    Mul(Vec<Tree>),
    Pow(Box<Tree>, Box<Tree>),
    F1(&'static str, Box<Tree>),
    Atan2(Box<Tree>, Box<Tree>),
    Pw(Vec<(Cond, Tree)>, Box<Tree>),
}

#[derive(Debug, Clone)]
enum Cond {
    Cmp(u8, Tree, Tree),
    And(Vec<Cond>),
    Or(Vec<Cond>),
    Not(Box<Cond>),
    True,
}

fn arb_plain(depth: u32) -> BoxedStrategy<Tree> {
    let leaf = prop_oneof![
        Just(Tree::X),
        Just(Tree::Y),
        (-4i32..=4).prop_map(Tree::Int),
        ((-7i32..=7), (1i32..=5)).prop_map(|(n, d)| Tree::Rat(n, d)),
    ];
    leaf.prop_recursive(depth, 24, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 1..4).prop_map(Tree::Add),
            prop::collection::vec(inner.clone(), 1..4).prop_map(Tree::Mul),
            (inner.clone(), (-3i32..=5))
                .prop_map(|(a, n)| Tree::Pow(Box::new(a), Box::new(Tree::Int(n)))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Pow(Box::new(a), Box::new(b))),
            (
                prop::sample::select(vec![
                    "sin",
                    "exp",
                    "log",
                    "sqrt",
                    "abs",
                    "erf",
                    "gamma",
                    "floor",
                    "lambert_w",
                ]),
                inner.clone()
            )
                .prop_map(|(f, a)| Tree::F1(f, Box::new(a))),
            (inner.clone(), inner).prop_map(|(a, b)| Tree::Atan2(Box::new(a), Box::new(b))),
        ]
    })
    .boxed()
}

fn arb_cond(plain: BoxedStrategy<Tree>) -> BoxedStrategy<Cond> {
    let atom = prop_oneof![
        ((0u8..6), plain.clone(), plain).prop_map(|(k, a, b)| Cond::Cmp(k, a, b)),
        Just(Cond::True),
    ];
    atom.prop_recursive(2, 8, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 1..4).prop_map(Cond::And),
            prop::collection::vec(inner.clone(), 1..4).prop_map(Cond::Or),
            inner.prop_map(|c| Cond::Not(Box::new(c))),
        ]
    })
    .boxed()
}

fn arb_tree() -> BoxedStrategy<Tree> {
    let plain = arb_plain(3);
    let cond = arb_cond(arb_plain(2));
    plain
        .clone()
        .prop_recursive(2, 32, 3, move |inner| {
            prop_oneof![
                (
                    prop::collection::vec((cond.clone(), inner.clone()), 1..4),
                    inner.clone()
                )
                    .prop_map(|(bs, d)| Tree::Pw(bs, Box::new(d))),
                prop::collection::vec(inner.clone(), 1..3).prop_map(Tree::Add),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Mul(vec![a, b])),
                (prop::sample::select(vec!["sin", "exp", "lambert_w"]), inner)
                    .prop_map(|(f, a)| Tree::F1(f, Box::new(a))),
            ]
        })
        .boxed()
}

fn build(t: &Tree, pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
    match t {
        Tree::X => x,
        Tree::Y => y,
        Tree::Int(n) => pool.integer(*n),
        Tree::Rat(n, d) => pool.rational(*n, *d),
        Tree::Add(v) => pool.add(v.iter().map(|c| build(c, pool, x, y)).collect()),
        Tree::Mul(v) => pool.mul(v.iter().map(|c| build(c, pool, x, y)).collect()),
        Tree::Pow(a, b) => pool.pow(build(a, pool, x, y), build(b, pool, x, y)),
        Tree::F1(f, a) => pool.func(*f, vec![build(a, pool, x, y)]),
        Tree::Atan2(a, b) => pool.func("atan2", vec![build(a, pool, x, y), build(b, pool, x, y)]),
        Tree::Pw(bs, d) => pool.piecewise(
            bs.iter()
                .map(|(c, v)| (build_cond(c, pool, x, y), build(v, pool, x, y)))
                .collect(),
            build(d, pool, x, y),
        ),
    }
}

fn build_cond(c: &Cond, pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
    match c {
        Cond::Cmp(k, a, b) => {
            let (a, b) = (build(a, pool, x, y), build(b, pool, x, y));
            match k {
                0 => pool.pred_lt(a, b),
                1 => pool.pred_le(a, b),
                2 => pool.pred_gt(a, b),
                3 => pool.pred_ge(a, b),
                4 => pool.pred_eq(a, b),
                _ => pool.pred_ne(a, b),
            }
        }
        Cond::And(v) => pool.pred_and(v.iter().map(|c| build_cond(c, pool, x, y)).collect()),
        Cond::Or(v) => pool.pred_or(v.iter().map(|c| build_cond(c, pool, x, y)).collect()),
        Cond::Not(c) => pool.pred_not(build_cond(c, pool, x, y)),
        Cond::True => pool.pred_true(),
    }
}

fn check_all_evaluators(
    expr: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
    xs: &[f64],
    ys: &[f64],
) -> Result<(), TestCaseError> {
    let n = xs.len();
    let f = interp(expr, &[x, y], pool);
    let mut block = vec![0.0; n];
    f.call_batch(&[xs, ys], &mut block);
    for i in 0..n {
        let env = HashMap::from([(x, xs[i]), (y, ys[i])]);
        let reference = eval_interp(expr, &env, pool).unwrap_or(f64::NAN);
        let point = f.call(&[xs[i], ys[i]]);
        prop_assert!(
            same(block[i], reference) && same(point, reference),
            "{} at ({}, {}): block {:?}, point {point:?}, eval_interp {reference:?}",
            pool.display(expr),
            xs[i],
            ys[i],
            block[i]
        );
    }
    // A native backend, where one compiles the expression (it declines
    // `Piecewise` and falls back to the interpreter, which must agree too).
    #[cfg(any(feature = "cranelift", feature = "jit"))]
    {
        let native = compile_with(expr, &[x, y], pool, CompileConfig::for_batch(n as u64)).unwrap();
        let mut out = vec![0.0; n];
        native.call_batch(&[xs, ys], &mut out);
        for i in 0..n {
            prop_assert!(
                same(out[i], block[i]),
                "{} ({:?}) at ({}, {}): {:?} vs block {:?}",
                pool.display(expr),
                native.compile_tier(),
                xs[i],
                ys[i],
                out[i],
                block[i]
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(160))]

    #[test]
    fn block_agrees_with_every_evaluator_on_random_piecewise(
        t in arb_tree(),
        seed in any::<u64>(),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = build(&t, &pool, x, y);
        // 300 points: a full block, then a partial one.
        let xs = special_grid(300, seed);
        let ys = special_grid(300, seed.rotate_left(17));
        check_all_evaluators(e, x, y, &pool, &xs, &ys)?;
    }

    #[test]
    fn block_agrees_with_every_evaluator_on_random_arithmetic(
        t in arb_plain(4),
        seed in any::<u64>(),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = build(&t, &pool, x, y);
        let xs = special_grid(300, seed);
        let ys = special_grid(300, seed.rotate_left(17));
        check_all_evaluators(e, x, y, &pool, &xs, &ys)?;
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_block_batches_match_sequential_for_piecewise() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let zero = pool.integer(0_i32);
    let pw = pool.piecewise(
        vec![
            (
                pool.pred_gt(x, y),
                pool.func("exp", vec![pool.mul(vec![x, y])]),
            ),
            (pool.pred_lt(x, zero), lambert(&pool, x)),
        ],
        pool.pow(pool.add(vec![x, y]), pool.integer(3_i32)),
    );
    let n = 3 * PAR_CHUNK + 77;
    let xs = special_grid(n, 3);
    let ys = special_grid(n, 5);
    let f = interp(pw, &[x, y], &pool);
    assert!(is_program(&f));
    let mut seq = vec![0.0; n];
    let mut par = vec![0.0; n];
    f.call_batch(&[&xs, &ys], &mut seq);
    f.call_batch_par(&[&xs, &ys], &mut par);
    for i in 0..n {
        assert!(same(seq[i], par[i]), "at {i}");
        assert!(same(seq[i], f.call(&[xs[i], ys[i]])), "point at {i}");
    }
}

/// The LLVM tier's parallel batch is chunked over its column entry, like
/// Cranelift's: the same values as the sequential batch and the interpreter.
#[cfg(all(feature = "jit", feature = "parallel"))]
#[test]
fn llvm_parallel_batch_is_chunked_and_exact() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let expr = pool.add(vec![
        pool.mul(vec![pool.func("sin", vec![x]), pool.func("exp", vec![y])]),
        pool.pow(pool.add(vec![x, y]), pool.integer(-3_i32)),
        pool.pow(pool.func("abs", vec![x]), y),
    ]);
    let cfg = |tier| CompileConfig {
        expected_evals: None,
        force_tier: Some(tier),
    };
    let llvm = compile_with(expr, &[x, y], &pool, cfg(CompileTier::Llvm)).unwrap();
    assert_eq!(llvm.compile_tier(), CompileTier::Llvm);
    assert!(llvm.native_cols_fn().is_some());
    let reference = interp(expr, &[x, y], &pool);
    let n = 4 * PAR_CHUNK + 321;
    let xs = special_grid(n, 13);
    let ys = special_grid(n, 17);
    let mut seq = vec![0.0; n];
    let mut par = vec![0.0; n];
    let mut want = vec![0.0; n];
    llvm.call_batch(&[&xs, &ys], &mut seq);
    llvm.call_batch_par(&[&xs, &ys], &mut par);
    reference.call_batch(&[&xs, &ys], &mut want);
    let mut flat = xs.clone();
    flat.extend_from_slice(&ys);
    let mut bulk = vec![0.0; n];
    llvm.call_bulk(&flat, &mut bulk);
    for i in 0..n {
        assert!(
            same(seq[i], par[i]) && same(seq[i], want[i]) && same(bulk[i], want[i]),
            "at ({}, {}): seq {:?} par {:?} bulk {:?} interpreter {:?}",
            xs[i],
            ys[i],
            seq[i],
            par[i],
            bulk[i],
            want[i]
        );
        assert!(same(llvm.call(&[xs[i], ys[i]]), want[i]));
    }
}
