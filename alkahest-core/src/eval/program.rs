//! A flat, reusable `f64` program for evaluating one expression many times.
//!
//! [`crate::jit::eval_interp`] walks the pool on every call: it looks every
//! node up, builds a fresh memo table, and resolves each function by name in
//! the primitive registry. That is the right trade for a single evaluation, and
//! the wrong one for a sampler or a batch that evaluates the *same* expression
//! at thousands of points. [`NumericProgram`] does that work once: the DAG is
//! flattened into a post-order list of operations over numbered slots (each
//! shared node gets exactly one slot, so sharing is preserved without a memo),
//! constants are folded to their `f64` value, and every function head is
//! resolved to its primitive.
//!
//! # Agreement with the interpreter
//!
//! A program computes, bit for bit, what [`crate::jit::eval_interp`] computes
//! for the same bindings — same operations, same order, same numeric kernels:
//!
//! * `Add` folds left from `0.0` and `Mul` from `1.0`, exactly as the
//!   interpreter does (`0.0 + -0.0` is `+0.0`, so the seed is not cosmetic);
//! * powers go through [`pow_f64`] with the exponent's exact node, so a wide
//!   integer exponent keeps its parity;
//! * functions call the same [`Primitive::numeric_f64`] kernel the registry
//!   dispatches to;
//! * an unbound symbol (other than `π`) or an unknown function head makes every
//!   evaluation `None`, as it does in the interpreter, where such a node is
//!   always reached.
//!
//! Only forms whose every node is reached on every evaluation are compiled.
//! `Piecewise`, `Predicate` and `RootSum` — where the interpreter evaluates a
//! branch lazily or does something other than arithmetic — are declined
//! (`compile` returns `None`) and the caller keeps its tree walk for them.

use crate::kernel::{integer_to_f64, pow_f64, rational_to_f64, ExprData, ExprId, ExprPool};
use crate::primitive::Primitive;

use super::IdMap;

/// Index of a value slot.
type Slot = u32;

#[derive(Clone, Copy)]
enum OpKind {
    /// `0.0 + s[a] + s[b]`.
    Add2(Slot, Slot),
    /// `0.0 + Σ s[operands[start..start+len]]`, left to right.
    Add { start: u32, len: u32 },
    /// `1.0 · s[a] · s[b]`.
    Mul2(Slot, Slot),
    /// `1.0 · Π s[operands[start..start+len]]`, left to right.
    Mul { start: u32, len: u32 },
    /// `pow_f64(s[base], s[exp], exp_node)`; `wide` indexes
    /// [`NumericProgram::exp_nodes`] when the exponent node is an exact
    /// number (`u32::MAX` otherwise).
    Pow { base: Slot, exp: Slot, wide: u32 },
    /// A one-argument primitive.
    Func1 {
        f: &'static dyn Primitive,
        arg: Slot,
    },
    /// A primitive of any other arity.
    FuncN {
        f: &'static dyn Primitive,
        start: u32,
        len: u32,
    },
}

#[derive(Clone, Copy)]
struct Op {
    kind: OpKind,
    dst: Slot,
}

/// A compiled, reusable evaluation plan. See the [module docs](self).
#[derive(Clone)]
pub(crate) struct NumericProgram {
    n_inputs: usize,
    /// Initial slot contents: `n_inputs` input slots, then constants; slots
    /// written by operations follow. Constants are never overwritten, so one
    /// scratch buffer can be reused across any number of evaluations.
    template: Vec<f64>,
    ops: Vec<Op>,
    operands: Vec<Slot>,
    /// Exponent nodes a wide power needs to see exactly (cold path only).
    exp_nodes: Vec<ExprData>,
    result: Slot,
    /// The expression reaches a node no binding can make evaluable.
    always_fails: bool,
}

/// Which primitive a `Func` head resolves to — the same table
/// [`crate::jit::eval_interp`] consults by name.
fn primitive(name: &str) -> Option<&'static dyn Primitive> {
    crate::jit::registry().get(name)
}

impl NumericProgram {
    /// Compile `expr` with `inputs[i]` bound to input slot `i`.
    ///
    /// A symbol listed twice takes its value from the *last* position, as a
    /// `HashMap` environment built by inserting `inputs` in order would.
    /// Returns `None` for a form the program does not model (see the module
    /// docs); the caller should fall back to the tree-walking interpreter.
    pub(crate) fn compile(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Option<Self> {
        let n_inputs = inputs.len();
        let mut template = vec![0.0f64; n_inputs];
        let mut slot_of: IdMap<Slot> = IdMap::default();
        for (i, &var) in inputs.iter().enumerate() {
            // Environments are consulted only at `Symbol` nodes, so an input
            // that is not a symbol binds nothing — and must not here either.
            if pool.with(var, |d| matches!(d, ExprData::Symbol { .. })) {
                slot_of.insert(var, slot(i)?);
            }
        }
        let mut ops: Vec<Op> = Vec::new();
        let mut operands: Vec<Slot> = Vec::new();
        let mut exp_nodes: Vec<ExprData> = Vec::new();
        let mut always_fails = false;

        // Iterative post-order: a deep DAG must not overflow the stack here.
        let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];
        while let Some((id, expanded)) = stack.pop() {
            if slot_of.contains_key(&id) {
                continue;
            }
            if !expanded {
                enum Visit {
                    Const(f64),
                    Fail,
                    Children,
                    Unsupported,
                }
                let visit = pool.with(id, |data| match data {
                    ExprData::Integer(n) => Visit::Const(integer_to_f64(&n.0)),
                    ExprData::Rational(r) => Visit::Const(rational_to_f64(&r.0)),
                    ExprData::Float(f) => Visit::Const(f.inner.to_f64()),
                    // Not an input (inputs were seeded above): `π` has its
                    // own value, anything else is unbound.
                    ExprData::Symbol { name, .. } => {
                        if name == super::symbols::PI_NAME {
                            Visit::Const(std::f64::consts::PI)
                        } else {
                            Visit::Fail
                        }
                    }
                    ExprData::Add(_)
                    | ExprData::Mul(_)
                    | ExprData::Pow { .. }
                    | ExprData::Func { .. } => Visit::Children,
                    _ => Visit::Unsupported,
                });
                match visit {
                    Visit::Const(v) => {
                        slot_of.insert(id, slot(template.len())?);
                        template.push(v);
                    }
                    Visit::Fail => {
                        always_fails = true;
                        slot_of.insert(id, slot(template.len())?);
                        template.push(f64::NAN);
                    }
                    Visit::Unsupported => return None,
                    Visit::Children => {
                        stack.push((id, true));
                        pool.with(id, |data| match data {
                            ExprData::Add(args)
                            | ExprData::Mul(args)
                            | ExprData::Func { args, .. } => {
                                // Reversed so children are *finished* in
                                // argument order (only matters for layout).
                                for &a in args.iter().rev() {
                                    stack.push((a, false));
                                }
                            }
                            ExprData::Pow { base, exp } => {
                                stack.push((*exp, false));
                                stack.push((*base, false));
                            }
                            _ => unreachable!(),
                        });
                    }
                }
                continue;
            }

            // Every child has a slot now; emit this node's operation.
            let s = |c: &ExprId| slot_of[c];
            let kind = pool.with(id, |data| -> Option<OpKind> {
                Some(match data {
                    ExprData::Add(args) if args.len() == 2 => {
                        OpKind::Add2(s(&args[0]), s(&args[1]))
                    }
                    ExprData::Mul(args) if args.len() == 2 => {
                        OpKind::Mul2(s(&args[0]), s(&args[1]))
                    }
                    ExprData::Add(args) | ExprData::Mul(args) => {
                        let start = u32::try_from(operands.len()).ok()?;
                        operands.extend(args.iter().map(s));
                        let len = u32::try_from(args.len()).ok()?;
                        if matches!(data, ExprData::Add(_)) {
                            OpKind::Add { start, len }
                        } else {
                            OpKind::Mul { start, len }
                        }
                    }
                    ExprData::Pow { base, exp } => {
                        let exact = pool.with(*exp, |e| {
                            matches!(e, ExprData::Integer(_) | ExprData::Rational(_))
                                .then(|| e.clone())
                        });
                        let wide = match exact {
                            Some(node) => {
                                exp_nodes.push(node);
                                u32::try_from(exp_nodes.len() - 1).ok()?
                            }
                            None => u32::MAX,
                        };
                        OpKind::Pow {
                            base: s(base),
                            exp: s(exp),
                            wide,
                        }
                    }
                    ExprData::Func { name, args } => match primitive(name) {
                        None => {
                            always_fails = true;
                            // Placeholder never executed: `run` returns
                            // before the first op when `always_fails`.
                            OpKind::Add { start: 0, len: 0 }
                        }
                        Some(f) if args.len() == 1 => OpKind::Func1 {
                            f,
                            arg: s(&args[0]),
                        },
                        Some(f) => {
                            let start = u32::try_from(operands.len()).ok()?;
                            operands.extend(args.iter().map(s));
                            OpKind::FuncN {
                                f,
                                start,
                                len: u32::try_from(args.len()).ok()?,
                            }
                        }
                    },
                    _ => unreachable!(),
                })
            })?;
            let dst = slot(template.len())?;
            template.push(0.0);
            slot_of.insert(id, dst);
            ops.push(Op { kind, dst });
        }

        Some(Self {
            n_inputs,
            result: slot_of[&expr],
            template,
            ops,
            operands,
            exp_nodes,
            always_fails,
        })
    }

    /// A scratch buffer for [`eval_in`](Self::eval_in); reusable across calls.
    pub(crate) fn scratch(&self) -> Vec<f64> {
        self.template.clone()
    }

    /// Length of a buffer from [`scratch`](Self::scratch).
    pub(crate) fn scratch_len(&self) -> usize {
        self.template.len()
    }

    /// Re-initialise a buffer of [`scratch_len`](Self::scratch_len) slots —
    /// one that may have been used by a *different* program — for this one.
    pub(crate) fn reset_scratch(&self, scratch: &mut [f64]) {
        scratch.copy_from_slice(&self.template);
    }

    /// Evaluate at `inputs` (one value per compiled input), allocating
    /// a scratch buffer. Same value as `eval_interp` with the same bindings.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn eval(&self, inputs: &[f64]) -> Option<f64> {
        let mut scratch = self.scratch();
        self.eval_in(inputs, &mut scratch)
    }

    /// Evaluate at `inputs` using a buffer from [`scratch`](Self::scratch).
    #[inline]
    pub(crate) fn eval_in(&self, inputs: &[f64], scratch: &mut [f64]) -> Option<f64> {
        debug_assert_eq!(inputs.len(), self.n_inputs);
        debug_assert_eq!(scratch.len(), self.template.len());
        scratch[..self.n_inputs].copy_from_slice(inputs);
        self.run(scratch)
    }

    /// Evaluate over `output.len()` points laid out column-major in
    /// `inputs_flat` (input `i` of point `j` at `i * n_points + j`), writing
    /// `NaN` where an evaluation fails — the interpreter tier's convention.
    pub(crate) fn eval_columns(&self, inputs_flat: &[f64], output: &mut [f64]) {
        let n_points = output.len();
        debug_assert_eq!(inputs_flat.len(), self.n_inputs * n_points);
        if self.always_fails {
            output.fill(f64::NAN);
            return;
        }
        let mut scratch = self.scratch();
        for (j, out) in output.iter_mut().enumerate() {
            for i in 0..self.n_inputs {
                scratch[i] = inputs_flat[i * n_points + j];
            }
            *out = self.run(&mut scratch).unwrap_or(f64::NAN);
        }
    }

    /// Like [`eval_columns`](Self::eval_columns) with one slice per input.
    pub(crate) fn eval_slices(&self, inputs: &[&[f64]], output: &mut [f64], offset: usize) {
        if self.always_fails {
            output.fill(f64::NAN);
            return;
        }
        let mut scratch = self.scratch();
        for (j, out) in output.iter_mut().enumerate() {
            for (i, col) in inputs.iter().enumerate() {
                scratch[i] = col[offset + j];
            }
            *out = self.run(&mut scratch).unwrap_or(f64::NAN);
        }
    }

    #[inline]
    fn run(&self, s: &mut [f64]) -> Option<f64> {
        if self.always_fails {
            return None;
        }
        for op in &self.ops {
            let v = match op.kind {
                OpKind::Add2(a, b) => 0.0 + s[a as usize] + s[b as usize],
                OpKind::Mul2(a, b) => 1.0 * s[a as usize] * s[b as usize],
                OpKind::Add { start, len } => {
                    let mut acc = 0.0f64;
                    for &a in &self.operands[start as usize..(start + len) as usize] {
                        acc += s[a as usize];
                    }
                    acc
                }
                OpKind::Mul { start, len } => {
                    let mut acc = 1.0f64;
                    for &a in &self.operands[start as usize..(start + len) as usize] {
                        acc *= s[a as usize];
                    }
                    acc
                }
                OpKind::Pow { base, exp, wide } => {
                    let node = self.exp_nodes.get(wide as usize);
                    pow_f64(s[base as usize], s[exp as usize], || node.cloned())
                }
                OpKind::Func1 { f, arg } => f.numeric_f64(&[s[arg as usize]])?,
                OpKind::FuncN { f, start, len } => {
                    let args = &self.operands[start as usize..(start + len) as usize];
                    if args.len() <= 8 {
                        let mut buf = [0.0f64; 8];
                        for (k, &a) in args.iter().enumerate() {
                            buf[k] = s[a as usize];
                        }
                        f.numeric_f64(&buf[..args.len()])?
                    } else {
                        let vals: Vec<f64> = args.iter().map(|&a| s[a as usize]).collect();
                        f.numeric_f64(&vals)?
                    }
                }
            };
            s[op.dst as usize] = v;
        }
        Some(s[self.result as usize])
    }
}

fn slot(n: usize) -> Option<Slot> {
    Slot::try_from(n).ok().filter(|&s| s != u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;
    use proptest::prelude::*;
    use rug::ops::Pow;
    use std::collections::HashMap;

    fn same(a: Option<f64>, b: Option<f64>) -> bool {
        match (a, b) {
            (Some(x), Some(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            (None, None) => true,
            _ => false,
        }
    }

    fn check(expr: ExprId, vars: &[ExprId], pool: &ExprPool, points: &[Vec<f64>]) {
        let prog = NumericProgram::compile(expr, vars, pool).expect("compilable");
        for pt in points {
            let env: HashMap<ExprId, f64> = vars.iter().copied().zip(pt.iter().copied()).collect();
            let want = eval_interp(expr, &env, pool);
            let got = prog.eval(pt);
            assert!(
                same(want, got),
                "{} at {pt:?}: interp {want:?} vs program {got:?}",
                pool.display(expr)
            );
        }
    }

    #[test]
    fn matches_interpreter_on_mixed_heads() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let pi = crate::eval::symbols::pi_symbol(&pool);
        let e = pool.add(vec![
            pool.mul(vec![pool.func("sin", vec![x]), pool.func("exp", vec![y])]),
            pool.pow(x, pool.integer(3_i32)),
            pool.pow(x, pool.rational(1, 2)),
            pool.pow(y, x),
            pool.mul(vec![pool.rational(-7, 3), pi]),
            pool.func("atan2", vec![y, x]),
            pool.func("tan", vec![pool.add(vec![x, y])]),
            pool.float(1.25, 53),
        ]);
        let pts: Vec<Vec<f64>> = [-2.5, -1.0, -0.0, 0.0, 0.3, 1.0, 7.5]
            .iter()
            .flat_map(|&a| [-1.5, 0.0, 2.0].iter().map(move |&b| vec![a, b]))
            .collect();
        check(e, &[x, y], &pool, &pts);
    }

    #[test]
    fn wide_integer_exponent_keeps_its_parity() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let odd = pool.integer(rug::Integer::from(10).pow(30u32) + 1u32);
        let e = pool.pow(x, odd);
        let prog = NumericProgram::compile(e, &[x], &pool).unwrap();
        assert_eq!(prog.eval(&[-1.0]), Some(-1.0));
        check(
            e,
            &[x],
            &pool,
            &[vec![-1.0], vec![1.0], vec![-2.0], vec![0.5]],
        );
    }

    #[test]
    fn unbound_symbol_and_unknown_head_always_fail() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let e1 = pool.add(vec![x, z]);
        let e2 = pool.func("no_such_function", vec![x]);
        for e in [e1, e2] {
            let prog = NumericProgram::compile(e, &[x], &pool).unwrap();
            assert_eq!(prog.eval(&[1.0]), None);
            check(e, &[x], &pool, &[vec![1.0]]);
            let mut out = [0.0; 3];
            prog.eval_columns(&[1.0, 2.0, 3.0], &mut out);
            assert!(out.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn duplicate_input_takes_the_last_position() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let prog = NumericProgram::compile(x, &[x, x], &pool).unwrap();
        assert_eq!(prog.eval(&[1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn lazy_forms_are_declined() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(
            vec![(pool.pred_ge(x, pool.integer(0_i32)), x)],
            pool.integer(-1_i32),
        );
        assert!(NumericProgram::compile(pw, &[x], &pool).is_none());
        let inside = pool.add(vec![pool.integer(1_i32), pw]);
        assert!(NumericProgram::compile(inside, &[x], &pool).is_none());
    }

    #[test]
    fn deep_chain_compiles_without_recursion() {
        // 200k nested heads on the default 2 MiB test-thread stack: a
        // recursive compile would overflow long before the end.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut e = x;
        for _ in 0..200_000 {
            e = pool.func("sin", vec![e]);
        }
        let prog = NumericProgram::compile(e, &[x], &pool).unwrap();
        let want = (0..200_000).fold(0.5f64, |v, _| v.sin());
        let got = prog.eval(&[0.5]).unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn chebyshev_dag_is_linear_in_distinct_nodes() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let (mut a, mut b) = (pool.integer(1_i32), x);
        for _ in 0..60 {
            let next = pool.add(vec![
                pool.mul(vec![two, x, b]),
                pool.mul(vec![pool.integer(-1_i32), a]),
            ]);
            a = b;
            b = next;
        }
        let prog = NumericProgram::compile(b, &[x], &pool).unwrap();
        assert!(prog.ops.len() < 200, "{} ops", prog.ops.len());
        // T_61(cos t) = cos(61 t).
        let t = 0.3f64;
        let v = prog.eval(&[t.cos()]).unwrap();
        assert!((v - (61.0 * t).cos()).abs() < 1e-9, "{v}");
        check(b, &[x], &pool, &[vec![t.cos()], vec![-0.7], vec![1.0]]);
    }

    // Random expressions over a small vocabulary: the program must agree with
    // the interpreter bit for bit (or both NaN, or both refuse).
    fn arb_expr(depth: u32) -> BoxedStrategy<Tree> {
        let leaf = prop_oneof![
            Just(Tree::X),
            Just(Tree::Y),
            Just(Tree::Pi),
            Just(Tree::Z),
            (-5i32..=5).prop_map(Tree::Int),
            ((-9i32..=9), (1i32..=7)).prop_map(|(n, d)| Tree::Rat(n, d)),
            (-3.0f64..3.0).prop_map(Tree::Float),
        ];
        leaf.prop_recursive(depth, 48, 4, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 1..5).prop_map(Tree::Add),
                prop::collection::vec(inner.clone(), 1..5).prop_map(Tree::Mul),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Tree::Pow(Box::new(a), Box::new(b))),
                (
                    prop::sample::select(vec![
                        "sin", "cos", "exp", "log", "sqrt", "tan", "abs", "atan", "sinh", "erf",
                        "gamma", "floor", "nope"
                    ]),
                    inner.clone()
                )
                    .prop_map(|(f, a)| Tree::F1(f, Box::new(a))),
                (inner.clone(), inner).prop_map(|(a, b)| Tree::Atan2(Box::new(a), Box::new(b))),
            ]
        })
        .boxed()
    }

    #[derive(Debug, Clone)]
    enum Tree {
        X,
        Y,
        Z,
        Pi,
        Int(i32),
        Rat(i32, i32),
        Float(f64),
        Add(Vec<Tree>),
        Mul(Vec<Tree>),
        Pow(Box<Tree>, Box<Tree>),
        F1(&'static str, Box<Tree>),
        Atan2(Box<Tree>, Box<Tree>),
    }

    fn build(t: &Tree, pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
        match t {
            Tree::X => x,
            Tree::Y => y,
            Tree::Z => pool.symbol("z", Domain::Real),
            Tree::Pi => crate::eval::symbols::pi_symbol(pool),
            Tree::Int(n) => pool.integer(*n),
            Tree::Rat(n, d) => pool.rational(*n, *d),
            Tree::Float(f) => pool.float(*f, 53),
            Tree::Add(v) => pool.add(v.iter().map(|c| build(c, pool, x, y)).collect()),
            Tree::Mul(v) => pool.mul(v.iter().map(|c| build(c, pool, x, y)).collect()),
            Tree::Pow(a, b) => pool.pow(build(a, pool, x, y), build(b, pool, x, y)),
            Tree::F1(f, a) => pool.func(*f, vec![build(a, pool, x, y)]),
            Tree::Atan2(a, b) => {
                pool.func("atan2", vec![build(a, pool, x, y), build(b, pool, x, y)])
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn program_agrees_with_eval_interp(
            t in arb_expr(4),
            pts in prop::collection::vec((-4.0f64..4.0, -4.0f64..4.0), 1..6),
        ) {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let e = build(&t, &pool, x, y);
            let prog = NumericProgram::compile(e, &[x, y], &pool).expect("no lazy forms generated");
            let mut scratch = prog.scratch();
            for (a, b) in pts {
                let env = HashMap::from([(x, a), (y, b)]);
                let want = eval_interp(e, &env, &pool);
                let got = prog.eval_in(&[a, b], &mut scratch);
                prop_assert!(same(want, got), "{} at ({a}, {b}): {want:?} vs {got:?}", pool.display(e));
            }
        }
    }
}
